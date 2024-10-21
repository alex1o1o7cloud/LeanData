import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_is_two_l456_45688

noncomputable def sample : Finset ℚ := {-1, 0, 1, 2, 3}

theorem sample_variance_is_two :
  let n : ℕ := sample.card
  let μ : ℚ := (sample.sum id) / n
  let variance : ℚ := (sample.sum (λ x => (x - μ)^2)) / n
  (n = 5 ∧ μ = 1) → variance = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_is_two_l456_45688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duma_common_members_l456_45639

structure Duma where
  deputies : Finset Nat
  committees : Finset (Finset Nat)
  deputyCount : deputies.card = 1600
  committeeCount : committees.card = 16000
  committeeSize : ∀ c ∈ committees, c.card = 80

theorem duma_common_members (d : Duma) :
  ∃ c1 c2, c1 ∈ d.committees ∧ c2 ∈ d.committees ∧ c1 ≠ c2 ∧ (c1 ∩ c2).card ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duma_common_members_l456_45639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_and_optimal_price_l456_45611

/-- Represents the factory's production and pricing model -/
structure Factory where
  original_production : ℝ
  current_production : ℝ
  current_cost : ℝ
  current_price : ℝ
  price_increase_step : ℝ
  quantity_decrease_step : ℝ

/-- The factory satisfies the given conditions -/
def satisfies_conditions (f : Factory) : Prop :=
  f.current_production = f.original_production + 50 ∧
  (600 / f.current_production) = (450 / f.original_production) ∧
  f.current_cost = 600 ∧
  f.current_price = 900 ∧
  f.price_increase_step = 50 ∧
  f.quantity_decrease_step = 20

/-- The profit function based on price increase -/
noncomputable def profit_function (f : Factory) (price_increase : ℝ) : ℝ :=
  (f.current_price + price_increase - f.current_cost) *
  (f.current_production - (f.quantity_decrease_step / f.price_increase_step) * price_increase)

/-- Theorem stating the current production and optimal price -/
theorem production_and_optimal_price (f : Factory) 
  (h : satisfies_conditions f) :
  f.current_production = 200 ∧
  ∃ (optimal_price : ℝ), 
    optimal_price = 1000 ∧
    ∀ (price : ℝ), profit_function f (price - f.current_price) ≤ profit_function f (optimal_price - f.current_price) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_and_optimal_price_l456_45611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_range_l456_45693

/-- Parabola C₁ -/
def C₁ (x y : ℝ) : Prop := y^2 = 16*x

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1

/-- Point M -/
def M : ℝ × ℝ := (8, 0)

/-- Distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem x_coordinate_range :
  ∀ (P Q : ℝ × ℝ),
  C₁ P.1 P.2 →
  C₂ Q.1 Q.2 →
  distance P M = distance P Q →
  39/10 ≤ P.1 ∧ P.1 ≤ 55/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_range_l456_45693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vaccine_cooling_time_l456_45650

noncomputable def vaccine_storage_temp : ℚ := -24
noncomputable def initial_temp : ℚ := -4
noncomputable def cooling_rate : ℚ := 5

noncomputable def cooling_time : ℚ :=
  (vaccine_storage_temp - initial_temp) / cooling_rate

theorem vaccine_cooling_time :
  cooling_time = 4 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vaccine_cooling_time_l456_45650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_f_l456_45627

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt 2 * Real.cos (x - Real.pi/4) + 6*x^2 + x) / (6*x^2 + Real.cos x)

theorem max_min_sum_of_f :
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, m ≤ f x) ∧ M + m = 2 := by
  sorry

#check max_min_sum_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_of_f_l456_45627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markus_family_ages_l456_45686

/-- The age of Markus's great-grandson in years -/
def great_grandson_age : ℝ := sorry

/-- The age of Markus's grandson in years -/
def grandson_age : ℝ := 3.5 * great_grandson_age

/-- The age of Markus's son in years -/
def son_age : ℝ := 2 * grandson_age

/-- Markus's age in years -/
def markus_age : ℝ := 2 * son_age

theorem markus_family_ages :
  great_grandson_age + grandson_age + son_age + markus_age = 140 →
  great_grandson_age = 140 / 25.5 := by
  sorry

#check markus_family_ages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_markus_family_ages_l456_45686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l456_45620

noncomputable def f (x φ a : ℝ) := Real.sin (x + φ) + a * Real.cos x

theorem trigonometric_function_properties :
  ∀ φ a : ℝ,
  |φ| < π/2 →
  f (π/2) φ a = Real.sqrt 2 / 2 →
  (φ = π/4 ∨ φ = -π/4) ∧
  (a = Real.sqrt 3 ∧ φ = -π/3 →
    ∀ x : ℝ, (∃ k : ℤ, x ∈ Set.Icc (-5*π/6 + 2*π*↑k) (π/6 + 2*π*↑k)) →
      Monotone (λ x ↦ f x (-π/3) (Real.sqrt 3))) ∧
  (a = -1 ∧ φ = π/6 →
    ∀ x : ℝ, (∃ k : ℤ, x ∈ Set.Icc (-π/3 + 2*π*↑k) (2*π/3 + 2*π*↑k)) →
      Monotone (λ x ↦ f x (π/6) (-1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_properties_l456_45620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_dihedral_angle_l456_45679

/-- A regular tetrahedron is a tetrahedron with all edges of equal length. -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The cosine of the dihedral angle in a regular tetrahedron. -/
noncomputable def dihedral_angle_cos (t : RegularTetrahedron) : ℝ := 1 / Real.sqrt 3

/-- Theorem: The cosine of each dihedral angle in a regular tetrahedron is 1/√3. -/
theorem regular_tetrahedron_dihedral_angle 
  (t : RegularTetrahedron) : dihedral_angle_cos t = 1 / Real.sqrt 3 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_dihedral_angle_l456_45679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_is_fourteen_l456_45652

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  -- The length of the equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- Assumption that the side length is positive
  side_pos : side > 0
  -- Assumption that the base length is positive
  base_pos : base > 0
  -- Assumption that the triangle inequality holds
  triangle_ineq : base < 2 * side

/-- The length of the segment BN in the isosceles triangle -/
noncomputable def segment_length (t : IsoscelesTriangle) : ℝ :=
  t.side - (t.base * t.side) / (3 * t.side)

/-- Theorem: In the given isosceles triangle, the length of segment BN is 14 cm -/
theorem segment_length_is_fourteen :
  let t : IsoscelesTriangle := {
    side := 18,
    base := 12,
    side_pos := by norm_num,
    base_pos := by norm_num,
    triangle_ineq := by norm_num
  }
  segment_length t = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_is_fourteen_l456_45652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doppler_effect_train_l456_45610

/-- The speed of the train in m/s -/
noncomputable def c : ℝ := 60 * 1000 / 3600

/-- The frequency of the train's whistle when stationary in Hz -/
def N : ℝ := 2048

/-- The speed of sound in air in m/s -/
def V : ℝ := 340

/-- The observed frequency when the train is approaching -/
noncomputable def N_approaching : ℝ := (V * N) / (V - c)

/-- The observed frequency when the train is receding -/
noncomputable def N_receding : ℝ := (V * N) / (V + c)

/-- Theorem stating the observed frequencies are approximately correct -/
theorem doppler_effect_train : 
  (abs (N_approaching - 2153) < 1) ∧ (abs (N_receding - 1952) < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_doppler_effect_train_l456_45610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shift_l456_45655

/-- Given a function f: ℝ → ℝ such that f(x+1) = 2x^2 + 1 for all x ∈ ℝ,
    prove that f(x-1) = 2x^2 - 8x + 9 for all x ∈ ℝ. -/
theorem function_shift (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2 * x^2 + 1) :
  ∀ x : ℝ, f (x - 1) = 2 * x^2 - 8 * x + 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_shift_l456_45655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_uncountable_l456_45612

open Set

/-- S has the property that any open set intersecting (0,1) also intersects S -/
def IntersectsOpenSets (S : Set ℝ) : Prop :=
  ∀ U : Set ℝ, IsOpen U → (U ∩ Ioo 0 1).Nonempty → (U ∩ S).Nonempty

/-- T is a countable collection of open sets containing S -/
def IsCountableOpenCover (T : Set (Set ℝ)) (S : Set ℝ) : Prop :=
  Countable T ∧ (∀ U ∈ T, IsOpen U ∧ S ⊆ U)

theorem intersection_uncountable (S : Set ℝ) (T : Set (Set ℝ))
  (hS : IntersectsOpenSets S) (hT : IsCountableOpenCover T S) :
  ¬(Countable (⋂₀ T)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_uncountable_l456_45612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l456_45619

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := x / (x^2 - 8*x + 15)

-- Define the solution set
noncomputable def solution_set : Set ℝ := Set.Icc (5/2) 3 ∪ Set.Ioo 5 6

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 2} = solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l456_45619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l456_45687

-- Define the circle
def circle_C (x y : ℝ) : Prop := x^2 + (y+1)^2 = 5

-- Define the line with slope angle 120°
def line_l (x y : ℝ) : Prop := -Real.sqrt 3 * x - y + 1 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l A.1 A.2 ∧ line_l B.1 B.2

-- Theorem statement
theorem chord_length (A B : ℝ × ℝ) : 
  intersection_points A B → dist A B = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l456_45687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_alpha_l456_45600

theorem cos_pi_third_plus_alpha (α : ℝ) 
  (h1 : Real.cos α = 3/5) 
  (h2 : α ∈ Set.Ioo 0 (π/2)) : 
  Real.cos (π/3 + α) = (3 - 4*Real.sqrt 3)/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_alpha_l456_45600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l456_45601

-- Define the cost price and gain percent
noncomputable def cost_price : ℝ := 900
noncomputable def gain_percent : ℝ := 27.77777777777778

-- Define the selling price calculation function
noncomputable def selling_price (cp : ℝ) (gp : ℝ) : ℝ :=
  cp * (1 + gp / 100)

-- Theorem statement
theorem cycle_selling_price :
  selling_price cost_price gain_percent = 1150 := by
  -- Unfold the definitions
  unfold selling_price cost_price gain_percent
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l456_45601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_fixed_points_exist_l456_45685

/-- An ellipse with specific properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_properties (e : Ellipse) 
  (h1 : 2 * e.a = 4)
  (h2 : eccentricity e = 1/2) :
  e.a = 2 ∧ e.b = Real.sqrt 3 := by sorry

theorem fixed_points_exist (e : Ellipse) 
  (h1 : e.a = 2 ∧ e.b = Real.sqrt 3)
  (p : PointOnEllipse e)
  (hb : p.x ≠ 2 ∧ p.x ≠ -2) :
  ∃ (m n : ℝ), 
    (m = 1 ∨ m = 7) ∧ 
    n = 0 ∧
    (m - 4)^2 + n^2 - (8 * p.x * p.y - 8 * p.y) / (p.x^2 - 4) * n - 9 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_fixed_points_exist_l456_45685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l456_45632

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  (((0 < a ∧ a < 1) ∨ (a > 5/2 ∨ (0 < a ∧ a < 1/2))) ∧
   ¬((0 < a ∧ a < 1) ∧ (a > 5/2 ∨ (0 < a ∧ a < 1/2)))) →
  a ∈ Set.Icc (1/2) 1 ∪ Set.Ioi (5/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l456_45632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_45_l456_45665

/-- Represents the rowing scenario with given conditions -/
structure RowingScenario where
  downstream_distance : ℝ
  downstream_time : ℝ
  upstream_time : ℝ
  stream_speed : ℝ

/-- Calculates the upstream distance given a rowing scenario -/
noncomputable def upstream_distance (scenario : RowingScenario) : ℝ :=
  let boat_speed := scenario.downstream_distance / scenario.downstream_time + scenario.stream_speed
  (boat_speed - scenario.stream_speed) * scenario.upstream_time

/-- Theorem stating that under the given conditions, the upstream distance is 45 km -/
theorem upstream_distance_is_45 (scenario : RowingScenario) 
  (h1 : scenario.downstream_distance = 75)
  (h2 : scenario.downstream_time = 5)
  (h3 : scenario.upstream_time = 5)
  (h4 : scenario.stream_speed = 3) :
  upstream_distance scenario = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_45_l456_45665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l456_45662

/-- Given a hyperbola with equation y²/4 - x²/a = 1 and asymptote equations y = ±(2√3/3)x,
    prove that its eccentricity is √7/2 -/
theorem hyperbola_eccentricity (a : ℝ) :
  (∀ x y : ℝ, y^2 / 4 - x^2 / a = 1) →
  (∀ x : ℝ, ∃ y : ℝ, y = (2 * Real.sqrt 3 / 3) * x ∨ y = -(2 * Real.sqrt 3 / 3) * x) →
  Real.sqrt (4 + 3) / 2 = Real.sqrt 7 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l456_45662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_nine_l456_45602

/-- Two perpendicular lines intersecting at point B(3,4) with y-intercepts R and S --/
structure PerpendicularLines where
  m₁ : ℝ
  m₂ : ℝ
  c₁ : ℝ
  c₂ : ℝ
  perpendicular : m₁ * m₂ = -1
  intersect_at_B : 3 * m₁ + c₁ = 4 ∧ 3 * m₂ + c₂ = 4
  y_intercepts_sum : c₁ + c₂ = 3

/-- The area of triangle BRS --/
noncomputable def triangleArea (lines : PerpendicularLines) : ℝ :=
  3 / 2 * |2 * lines.c₁ - 3|

theorem area_is_nine (lines : PerpendicularLines) :
  triangleArea lines = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_nine_l456_45602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l456_45624

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem f_properties :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8),
    ∀ y ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8),
    x ≤ y → f x ≤ f y) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 0 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f x = 0) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f x ≤ Real.sqrt 2 + 1) ∧
  (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f x = Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l456_45624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l456_45680

noncomputable section

-- Define the function
def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 6) - 1

-- State the theorem
theorem range_of_f :
  (∀ y ∈ Set.Ioo 0 1, ∃ x ∈ Set.Ioo 0 ((2 : ℝ) * Real.pi / 3), f x = y) ∧
  (∀ x ∈ Set.Ioo 0 ((2 : ℝ) * Real.pi / 3), 0 < f x ∧ f x ≤ 1) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l456_45680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_condition_l456_45638

/-- The function h(x) as defined in the problem -/
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2*x

/-- The derivative of h(x) with respect to x -/
noncomputable def h_derivative (a : ℝ) (x : ℝ) : ℝ := 1/x - a*x - 2

theorem monotonic_decreasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 4, h_derivative a x < 0) ↔ a > -1 := by
  sorry

#check monotonic_decreasing_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_condition_l456_45638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_period_l456_45618

/-- The rotation matrix for 170 degrees -/
noncomputable def R : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos (170 * Real.pi / 180), -Real.sin (170 * Real.pi / 180)],
    ![Real.sin (170 * Real.pi / 180),  Real.cos (170 * Real.pi / 180)]]

/-- The statement that 36 is the smallest positive integer n such that R^n = I -/
theorem smallest_rotation_period :
  (∀ k : ℕ, 0 < k → k < 36 → R ^ k ≠ 1) ∧ R ^ 36 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_period_l456_45618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_marathon_duration_l456_45695

/-- Movie marathon duration calculation -/
theorem movie_marathon_duration :
  let movie1 : ℝ := 2
  let movie2 : ℝ := movie1 * 1.5
  let movie3 : ℝ := (movie1 + movie2) * 0.8
  let movie4 : ℝ := movie2 * 2
  let movie5 : ℝ := movie3 - 0.5
  let movie6 : ℝ := (movie2 + movie4) / 2
  let movie7 : ℝ := 45 / movie5
  ∃ ε > 0, |movie1 + movie2 + movie3 + movie4 + movie5 + movie6 + movie7 - 35.8571| < ε :=
by
  sorry

#check movie_marathon_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_marathon_duration_l456_45695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_root_product_sum_l456_45615

def g (x : ℝ) : ℝ := x^4 + 10*x^3 + 29*x^2 + 30*x + 9

def roots_of_g : Set ℝ := {x | g x = 0}

theorem smallest_root_product_sum : 
  ∃ (w₁ w₂ w₃ w₄ : ℝ), w₁ ∈ roots_of_g ∧ w₂ ∈ roots_of_g ∧ w₃ ∈ roots_of_g ∧ w₄ ∈ roots_of_g ∧
  (∀ (σ : Fin 4 → Fin 4), Function.Bijective σ → 
    |w₁ * w₂ + w₃ * w₄| ≤ |w₁ * w₂ + w₃ * w₄|) ∧
  |w₁ * w₂ + w₃ * w₄| = 6 :=
sorry

#check smallest_root_product_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_root_product_sum_l456_45615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_digits_of_1024_l456_45623

/-- The number of digits in the base-8 representation of a positive integer n -/
noncomputable def base8Digits (n : ℕ+) : ℕ :=
  Nat.floor (Real.log n / Real.log 8) + 1

/-- Theorem: The number of digits in the base-8 representation of 1024 is 4 -/
theorem base8_digits_of_1024 : base8Digits 1024 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_digits_of_1024_l456_45623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_theorem_l456_45629

theorem tan_period_theorem (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, Real.tan (3 * a * x - π / 3) = Real.tan (3 * a * (x + π / 2) - π / 3)) →
  a = 2 / 3 ∨ a = -2 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_theorem_l456_45629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_rounded_l456_45645

noncomputable def round_to_3dp (x : ℝ) : ℝ :=
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

theorem sum_of_fractions_rounded (a b : ℕ+) :
  round_to_3dp ((a : ℝ) / 5 + (b : ℝ) / 7) = 1.51 → a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fractions_rounded_l456_45645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l456_45668

theorem cube_root_equation_solution (x : ℝ) (r s : ℤ)
  (h1 : x ^ (1/3) + (30 - x) ^ (1/3) = 3)
  (h2 : x = r - Real.sqrt s) :
  r + s = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l456_45668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_properties_l456_45634

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 4 * Real.cos (ω * x) * Real.sin (ω * x - Real.pi / 6)

theorem function_and_triangle_properties (ω : ℝ) (A B C : ℝ) (a b c : ℝ) :
  ω > 0 →
  (∀ x, f ω x = f ω (x + Real.pi)) →
  0 < A ∧ A < Real.pi / 2 →
  f ω A = 0 →
  Real.sin B = Real.sqrt 3 * Real.sin C →
  1 / 2 * b * c * Real.sin A = 2 * Real.sqrt 3 →
  ω = 1 ∧ a = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_triangle_properties_l456_45634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_special_angle_l456_45646

/-- Given an angle α whose terminal side passes through the point (3, -4),
    prove that sin α + cos α = -1/5 -/
theorem sin_plus_cos_special_angle (α : ℝ) :
  (3 : ℝ) = 5 * Real.cos α ∧ (-4 : ℝ) = 5 * Real.sin α →
  Real.sin α + Real.cos α = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_special_angle_l456_45646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l456_45607

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom angle_sum : A + B + C = Real.pi
axiom side_a : a = 3 * Real.cos C
axiom side_b : b = 1

-- Define the theorem
theorem triangle_properties :
  (Real.tan C = 2 * Real.tan B) ∧
  ((∀ S : ℝ, S ≤ (1/2 * a * b * Real.sin C)) → Real.cos (2 * B) = 3/5) ∧
  (c = Real.sqrt 10 / 2 → Real.cos (2 * B) = 3/5) :=
by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l456_45607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l456_45642

theorem polynomial_divisibility (k l m n : ℕ) :
  ∃ q : Polynomial ℤ, (X : Polynomial ℤ)^(4*k) + (X : Polynomial ℤ)^(4*l + 1) + 
    (X : Polynomial ℤ)^(4*m + 2) + (X : Polynomial ℤ)^(4*n + 3) = 
    ((X : Polynomial ℤ)^3 + (X : Polynomial ℤ)^2 + (X : Polynomial ℤ) + 1) * q :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l456_45642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2α_values_l456_45631

theorem sin_2α_values (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : 3 * Real.cos (2 * α) = Real.sin (π / 4 - α)) :
  Real.sin (2 * α) = 1 ∨ Real.sin (2 * α) = -17/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2α_values_l456_45631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_price_is_two_l456_45614

/-- Calculates the selling price of tacos given the total beef, beef per taco, cost per taco, and total profit -/
noncomputable def calculate_taco_price (total_beef : ℝ) (beef_per_taco : ℝ) (cost_per_taco : ℝ) (total_profit : ℝ) : ℝ :=
  let num_tacos := total_beef / beef_per_taco
  let total_cost := num_tacos * cost_per_taco
  let total_revenue := total_cost + total_profit
  total_revenue / num_tacos

/-- Theorem stating that the selling price of each taco is $2 under the given conditions -/
theorem taco_price_is_two :
  calculate_taco_price 100 0.25 1.5 200 = 2 := by
  -- Unfold the definition of calculate_taco_price
  unfold calculate_taco_price
  -- Simplify the expressions
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_price_is_two_l456_45614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_min_sum_l456_45635

/-- Given a parabola x² = 2y and a line passing through point P(0,1) intersecting 
    the parabola at points A(x₁,y₁) and B(x₂,y₂), the minimum value of y₁ + y₂ is 2. -/
theorem parabola_intersection_min_sum :
  ∀ (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
  (x₁^2 = 2 * y₁) →  -- Point A on parabola
  (x₂^2 = 2 * y₂) →  -- Point B on parabola
  (y₁ = k * x₁ + 1) →  -- Point A on line
  (y₂ = k * x₂ + 1) →  -- Point B on line
  (x₁ ≠ x₂) →  -- Distinct intersection points
  (∀ (k' : ℝ) (x₁' y₁' x₂' y₂' : ℝ),
    (x₁'^2 = 2 * y₁') →
    (x₂'^2 = 2 * y₂') →
    (y₁' = k' * x₁' + 1) →
    (y₂' = k' * x₂' + 1) →
    (x₁' ≠ x₂') →
    (y₁ + y₂ ≤ y₁' + y₂')) →
  y₁ + y₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_min_sum_l456_45635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l456_45651

-- Define the function f(x) = 3 - 4sin(x) - cos²(x)
noncomputable def f (x : ℝ) : ℝ := 3 - 4 * Real.sin x - (Real.cos x) ^ 2

-- State the theorem about the maximum and minimum values of f(x)
theorem f_max_min :
  (∀ x : ℝ, f x ≤ 7) ∧ (∃ x : ℝ, f x = 7) ∧
  (∀ x : ℝ, f x ≥ -1) ∧ (∃ x : ℝ, f x = -1) := by
  sorry

#check f_max_min

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l456_45651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_origin_movement_l456_45698

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a dilation transformation -/
structure MyDilation where
  factor : ℝ
  center : Point

/-- Applies a dilation to a point -/
def applyDilation (d : MyDilation) (p : Point) : Point :=
  { x := d.center.x + d.factor * (p.x - d.center.x)
  , y := d.center.y + d.factor * (p.y - d.center.y) }

theorem dilation_origin_movement :
  let originalCircle : Circle := { center := { x := 3, y := 1 }, radius := 4 }
  let dilatedCircle : Circle := { center := { x := 7, y := 9 }, radius := 6 }
  let origin : Point := { x := 0, y := 0 }
  ∃ (d : MyDilation),
    (applyDilation d originalCircle.center = dilatedCircle.center) ∧
    (d.factor * originalCircle.radius = dilatedCircle.radius) ∧
    (distance origin (applyDilation d origin) = 0.5 * Real.sqrt 10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_origin_movement_l456_45698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_trailers_added_l456_45661

/-- Given the initial number of trailers, their initial average age,
    time passed, and the new average age, calculate the number of new trailers added. -/
theorem new_trailers_added
  (initial_trailers : ℕ)
  (initial_avg_age : ℝ)
  (years_passed : ℕ)
  (new_avg_age : ℝ)
  (h1 : initial_trailers = 30)
  (h2 : initial_avg_age = 15)
  (h3 : years_passed = 3)
  (h4 : new_avg_age = 12) :
  ∃ (new_trailers : ℕ),
    (initial_trailers * (initial_avg_age + years_passed : ℝ) + new_trailers * (years_passed : ℝ)) /
    (initial_trailers + new_trailers : ℝ) = new_avg_age ∧
    new_trailers = 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_trailers_added_l456_45661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l456_45621

theorem log_equation_solution (b : ℝ) (h : Real.log 216 / Real.log b = -3/2) : b = 1/36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l456_45621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l456_45676

/-- A right pyramid with a square base -/
structure RightSquarePyramid where
  base_side : ℝ
  height : ℝ

/-- Calculate the surface area of a right square pyramid -/
noncomputable def surface_area (p : RightSquarePyramid) : ℝ :=
  p.base_side ^ 2 + 4 * (1/2 * p.base_side * Real.sqrt (p.height ^ 2 + (p.base_side / 2) ^ 2))

/-- Calculate the volume of a right square pyramid -/
noncomputable def volume (p : RightSquarePyramid) : ℝ :=
  (1/3) * p.base_side ^ 2 * p.height

/-- The main theorem -/
theorem pyramid_volume_theorem (p : RightSquarePyramid) 
  (h1 : surface_area p = 540)
  (h2 : (1/2 * p.base_side * Real.sqrt (p.height ^ 2 + (p.base_side / 2) ^ 2)) = (3/4) * p.base_side ^ 2) :
  volume p = 405 * Real.sqrt 111.375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l456_45676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integer_iterate_f_l456_45691

/-- Ceiling function -/
noncomputable def ceil (x : ℝ) : ℤ :=
  Int.ceil x

/-- The function f -/
noncomputable def f (r : ℝ) : ℝ :=
  r * ceil r

/-- Iterate f n times -/
noncomputable def iterate_f : ℕ → ℝ → ℝ
  | 0, r => r
  | n + 1, r => f (iterate_f n r)

/-- The main theorem -/
theorem exists_integer_iterate_f (k : ℕ+) :
  ∃ m : ℕ+, ∃ n : ℤ, iterate_f m (k + 1/2) = n := by
  sorry

#check exists_integer_iterate_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_integer_iterate_f_l456_45691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_b_correct_option_c_correct_l456_45675

-- Define the line l: a(2x+3y+2)+b(x-2y-6)=0
def line_l (a b x y : ℝ) : Prop :=
  a * (2 * x + 3 * y + 2) + b * (x - 2 * y - 6) = 0

-- Define the condition ab≠0
def ab_not_zero (a b : ℝ) : Prop :=
  a * b ≠ 0

-- Define the circle (x-3)²+y²=5
def circle_eq (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 5

-- Theorem for option B
theorem option_b_correct (a b : ℝ) :
  ab_not_zero a b →
  (∃ m n : ℝ, line_l a b m 0 ∧ line_l a b 0 n ∧ m + n = 0) →
  a ≠ 3 * b →
  b = 5 * a :=
by
  sorry

-- Theorem for option C
theorem option_c_correct (a b : ℝ) :
  ab_not_zero a b →
  (∀ x y : ℝ, line_l a b x y → ¬circle_eq x y) →
  (∃ x y : ℝ, line_l a b x y ∧ circle_eq x y) →
  a = -4 * b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_b_correct_option_c_correct_l456_45675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_for_80_yuan_tax_l456_45616

/-- Calculates the tax for a given salary --/
noncomputable def calculateTax (salary : ℝ) : ℝ :=
  if salary ≤ 800 then 0
  else if salary ≤ 1300 then (salary - 800) * 0.05
  else 500 * 0.05 + (salary - 1300) * 0.1

/-- Theorem stating that a salary of 1850 yuan results in 80 yuan of tax --/
theorem salary_for_80_yuan_tax :
  calculateTax 1850 = 80 := by
  -- Unfold the definition of calculateTax
  unfold calculateTax
  -- Simplify the if-then-else expression
  simp
  -- Perform numerical calculations
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_for_80_yuan_tax_l456_45616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l456_45683

def a : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n + 2) => a (n + 1) + a n

noncomputable def series_sum : ℚ := ∑' n, a n / 3^(n+2)

theorem series_sum_value : series_sum = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_value_l456_45683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_equals_four_l456_45643

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 2

-- State the theorem
theorem function_sum_equals_four :
  f (Real.log 5 / Real.log 10) + f (Real.log (1/5) / Real.log 10) = 4 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_equals_four_l456_45643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_values_l456_45670

noncomputable def f (x : ℝ) : ℝ :=
  6 / (Real.sqrt (x - 9) - 10) + 1 / (Real.sqrt (x - 9) - 5) +
  7 / (Real.sqrt (x - 9) + 5) + 12 / (Real.sqrt (x - 9) + 10)

theorem solution_values (x : ℝ) : f x = 0 ↔ x = 9 ∨ x = 109 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_values_l456_45670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_projections_equilateral_triangle_l456_45692

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Perpendicular projection of a point onto a line segment -/
noncomputable def perpendicularProjection (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Check if three points form an equilateral triangle -/
def isEquilateralTriangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Main theorem: For any triangle, there exists a point whose perpendicular projections
    on the sides of the triangle form an equilateral triangle -/
theorem perpendicular_projections_equilateral_triangle (T : Triangle) :
  ∃ P : ℝ × ℝ, 
    isEquilateralTriangle 
      (perpendicularProjection P T.B T.C)
      (perpendicularProjection P T.C T.A)
      (perpendicularProjection P T.A T.B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_projections_equilateral_triangle_l456_45692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l456_45690

noncomputable section

open Real

-- Define the function f
def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x - π / 6) + sin (ω * x - π / 2)

-- State the theorem
theorem function_properties :
  ∀ ω : ℝ, 0 < ω → ω < 3 → f ω (π / 6) = 0 →
  (ω = 2) ∧
  (∃ g : ℝ → ℝ, 
    (∀ x : ℝ, g x = sqrt 3 * sin (x - π / 12)) ∧
    (∀ x : ℝ, -π / 4 ≤ x → x ≤ 3 * π / 4 → g x ≤ sqrt 3) ∧
    (∃ x : ℝ, -π / 4 ≤ x ∧ x ≤ 3 * π / 4 ∧ g x = sqrt 3) ∧
    (∀ x : ℝ, -π / 4 ≤ x → x ≤ 3 * π / 4 → -3 / 2 ≤ g x) ∧
    (∃ x : ℝ, -π / 4 ≤ x ∧ x ≤ 3 * π / 4 ∧ g x = -3 / 2)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l456_45690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l456_45697

theorem tan_alpha_plus_pi_fourth (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.cos α = -4/5) :
  Real.tan (α + π/4) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l456_45697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_neg_five_thirds_l456_45637

/-- The sum of the infinite series Σ(3n+2)/(n(n+1)(n+3)) from n=1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' (n : ℕ), (3 * n + 2) / (n * (n + 1) * (n + 3))

/-- The infinite series Σ(3n+2)/(n(n+1)(n+3)) from n=1 to infinity converges to -5/3 -/
theorem infinite_series_sum_eq_neg_five_thirds : infinite_series_sum = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_neg_five_thirds_l456_45637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l456_45622

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 4/5) : Real.cos (2*α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l456_45622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_product_two_solutions_l456_45641

/-- The product of all possible values of a, where a is determined by the condition that 
    the length of the segment between points (2a, a-4) and (4, -1) is 2√10 units. -/
theorem segment_length_product (a b : ℝ) : 
  (b ≠ a ∧ 
   ((2*a - 4)^2 + (a - 4 - (-1))^2 = 40) ∧ 
   ((2*b - 4)^2 + (b - 4 - (-1))^2 = 40)) →
  a * b = -3 := by
  sorry

/-- The number of distinct real solutions to the equation 
    (2a - 4)^2 + (a - 4 - (-1))^2 = 40 is exactly 2. -/
theorem two_solutions :
  ∃! (s : Set ℝ), (∀ a ∈ s, (2*a - 4)^2 + (a - 4 - (-1))^2 = 40) ∧ s.ncard = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_product_two_solutions_l456_45641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_minus_one_to_one_l456_45647

open Real MeasureTheory Interval

/-- The function to be integrated -/
noncomputable def f (x : ℝ) : ℝ := x^3 + tan x + x^2 * sin x

/-- The theorem stating that the integral of f from -1 to 1 is 0 -/
theorem integral_f_minus_one_to_one :
  ∫ x in (-1:ℝ)..1, f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_minus_one_to_one_l456_45647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_properties_l456_45666

/-- A custom polynomial type for this problem -/
structure CustomPolynomial (R : Type) [Ring R] where
  terms : List (R × List (Nat × Nat))

/-- The degree of a custom polynomial is the highest sum of exponents in any term. -/
def degree (p : CustomPolynomial ℝ) : ℕ := sorry

/-- The number of terms in a custom polynomial is the count of distinct combinations of variables and their powers, each multiplied by a non-zero coefficient. -/
def numTerms (p : CustomPolynomial ℝ) : ℕ := sorry

/-- A specific polynomial $2x^2y-3y^2-1$ -/
def p : CustomPolynomial ℝ := ⟨[
  (2, [(2, 1), (1, 1)]),   -- 2x^2y
  (-3, [(0, 2)]),          -- -3y^2
  (-1, [])                 -- -1
]⟩

theorem polynomial_properties :
  degree p = 3 ∧ numTerms p = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_properties_l456_45666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_n_l456_45672

-- Define the arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Define the sum of the first n terms
noncomputable def sum_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

-- Theorem statement
theorem arithmetic_sequence_max_sum_n (a₁ d : ℝ) :
  d < 0 ∧ arithmetic_sequence a₁ d 3 = -arithmetic_sequence a₁ d 9 →
  ∃ n : ℕ, (n = 5 ∨ n = 6) ∧
    ∀ m : ℕ, sum_n_terms a₁ d n ≥ sum_n_terms a₁ d m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_n_l456_45672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_distinct_abs_values_l456_45658

/-- The set S containing integers from -100 to 100 -/
def S : Set Int := {i | -100 ≤ i ∧ i ≤ 100}

/-- A random 50-element subset of S -/
noncomputable def T : Set Int := sorry

/-- The set of absolute values of elements in T -/
def absT : Set ℕ := {n | ∃ x ∈ T, n = Int.natAbs x}

/-- The expected number of elements in absT -/
noncomputable def expectedAbsT : ℝ := sorry

/-- Theorem stating the expected number of distinct absolute values -/
theorem expected_distinct_abs_values :
  ∃ ε > 0, |expectedAbsT - 22.3688| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_distinct_abs_values_l456_45658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_products_l456_45626

def is_valid_permutation (f g h j : ℕ) : Prop :=
  Multiset.ofList [f, g, h, j] = Multiset.ofList [6, 7, 8, 9]

def sum_of_products (f g h j : ℕ) : ℕ :=
  f * g + g * h + h * j + f * j

theorem max_sum_of_products :
  ∃ f g h j : ℕ,
    is_valid_permutation f g h j ∧
    (∀ f' g' h' j' : ℕ,
      is_valid_permutation f' g' h' j' →
      sum_of_products f' g' h' j' ≤ sum_of_products f g h j) ∧
    sum_of_products f g h j = 221 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_products_l456_45626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_star_circle_area_ratio_proof_l456_45678

/-- The ratio of the area of a hexagonal star formed from a circle of radius 3 to the area of the original circle --/
noncomputable def hexagonal_star_circle_area_ratio : ℝ := 4.5 * Real.sqrt 3 / Real.pi

/-- Proves that the ratio of the area of a hexagonal star formed from a circle of radius 3 to the area of the original circle is 4.5√3/π --/
theorem hexagonal_star_circle_area_ratio_proof (r : ℝ) (h : r = 3) : 
  let circle_area := Real.pi * r^2
  let hexagon_side := 2 * r
  let hexagon_area := 3 * Real.sqrt 3 / 2 * hexagon_side^2
  let triangle_area := Real.sqrt 3 / 4 * r^2
  let star_area := hexagon_area - 6 * triangle_area
  star_area / circle_area = hexagonal_star_circle_area_ratio :=
by
  sorry

#check hexagonal_star_circle_area_ratio
#check hexagonal_star_circle_area_ratio_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_star_circle_area_ratio_proof_l456_45678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_for_wall_l456_45636

/-- Calculates the number of bricks needed to build a wall -/
def bricks_needed (brick_length brick_width brick_height : ℚ) 
                  (wall_length wall_width wall_height : ℚ) : ℕ :=
  let brick_volume := brick_length * brick_width * brick_height
  let wall_volume := wall_length * wall_width * wall_height
  (wall_volume / brick_volume).ceil.toNat

/-- Theorem stating the number of bricks needed for the specified wall and brick dimensions -/
theorem bricks_for_wall : 
  bricks_needed 80 11.25 6 800 600 22.5 = 2000 := by
  -- Proof goes here
  sorry

#eval bricks_needed 80 11.25 6 800 600 22.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_for_wall_l456_45636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_prism_volume_l456_45660

/-- A straight prism with a regular hexagonal base and body diagonals of lengths 12 and 13 -/
structure HexagonalPrism where
  /-- The side length of the hexagonal base -/
  a : ℝ
  /-- The height of the prism -/
  m : ℝ
  /-- Condition for the longer body diagonal -/
  diagonal_long : (2 * a) ^ 2 + m ^ 2 = 13 ^ 2
  /-- Condition for the shorter body diagonal -/
  diagonal_short : (a * Real.sqrt 3) ^ 2 + m ^ 2 = 12 ^ 2

/-- The volume of the hexagonal prism -/
noncomputable def volume (p : HexagonalPrism) : ℝ :=
  (3 * Real.sqrt 3 / 2) * p.a ^ 2 * p.m

/-- Theorem stating the volume of the hexagonal prism -/
theorem hexagonal_prism_volume (p : HexagonalPrism) :
  volume p = 37.5 * Real.sqrt 207 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_prism_volume_l456_45660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheryll_book_purchase_cost_l456_45659

/-- Represents the cost and quantity of books in a category --/
structure BookCategory where
  price : ℚ
  quantity : ℕ
  discount : ℚ

/-- Calculates the total cost after discount for a book category --/
def totalCostAfterDiscount (category : BookCategory) : ℚ :=
  category.price * category.quantity * (1 - category.discount)

/-- Theorem stating the total cost of Sheryll's purchase --/
theorem sheryll_book_purchase_cost :
  let categoryA : BookCategory := ⟨10, 5, 1/10⟩
  let categoryB : BookCategory := ⟨15/2, 4, 3/20⟩
  let categoryC : BookCategory := ⟨5, 3, 1/5⟩
  totalCostAfterDiscount categoryA +
  totalCostAfterDiscount categoryB +
  totalCostAfterDiscount categoryC = 33/4 := by
  sorry

#eval (33 : ℚ) / 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheryll_book_purchase_cost_l456_45659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_l456_45640

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  first : ℚ
  diff : ℚ

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  ap.first + (n - 1 : ℚ) * ap.diff

/-- The sum of the first n terms of an arithmetic progression -/
def ArithmeticProgression.sumFirstN (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.first + (n - 1 : ℚ) * ap.diff)

/-- Theorem: For an arithmetic progression where the sum of the 4th term
    and the 12th term is 12, the sum of the first 15 terms is 90. -/
theorem arithmetic_progression_sum
  (ap : ArithmeticProgression)
  (h : ap.nthTerm 4 + ap.nthTerm 12 = 12) :
  ap.sumFirstN 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_l456_45640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_is_sqrt_three_l456_45649

theorem tan_alpha_is_sqrt_three (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : Real.sin α ^ 2 + Real.cos (2 * α) = 1 / 4) : 
  Real.tan α = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_is_sqrt_three_l456_45649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizer_l456_45654

/-- The probability function for the dragon's head regrowth --/
noncomputable def p (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

/-- The vector of observed head regrowths --/
def K : List ℕ := [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]

/-- The probability of obtaining the vector K --/
noncomputable def prob_K (x : ℝ) : ℝ :=
  (K.map (p x)).prod

/-- The theorem stating that (√97 + 1) / 8 maximizes the probability of obtaining K --/
theorem dragon_resilience_maximizer :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → prob_K x ≥ prob_K y ∧ x = (Real.sqrt 97 + 1) / 8 := by
  sorry

#check dragon_resilience_maximizer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_resilience_maximizer_l456_45654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l456_45664

/-- A geometric sequence with common ratio greater than 1 -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  geom_seq : ∀ n, a (n + 1) = a n * q
  q_gt_one : q > 1

/-- The sum of the first n terms of a geometric sequence -/
def geometricSum (g : GeometricSequence) (n : ℕ) : ℚ :=
  g.a 1 * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_properties (g : GeometricSequence)
    (h1 : g.a 1 + g.a 4 = 9)
    (h2 : g.a 2 * g.a 3 = 8) :
  g.a 1 = 1 ∧ g.q = 2 ∧ geometricSum g 6 = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l456_45664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_projections_perpendicular_l456_45605

/-- Predicate stating that four points form a rectangle -/
def IsRectangle (A B C D : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate stating that a point is on the circumcircle of a quadrilateral -/
def OnCircumcircle (M A B C D : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate stating that a point is the projection of another point onto a line -/
def IsProjection (P M : ℝ × ℝ) (L : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Type representing a line in 2D plane -/
def Line (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Predicate stating that two lines are perpendicular -/
def Perpendicular (L1 L2 : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Given a rectangle ABCD and a point M on its circumcircle, with projections P, Q, R, S,
    prove that PQ is perpendicular to RS -/
theorem rectangle_projections_perpendicular
  (A B C D M : ℝ × ℝ)  -- Points in 2D plane
  (P Q R S : ℝ × ℝ)    -- Projections
  (h_rectangle : IsRectangle A B C D)
  (h_on_circle : OnCircumcircle M A B C D)
  (h_not_vertex : M ≠ A ∧ M ≠ B)
  (h_P : IsProjection P M (Line A D))
  (h_Q : IsProjection Q M (Line A B))
  (h_R : IsProjection R M (Line B C))
  (h_S : IsProjection S M (Line C D)) :
  Perpendicular (Line P Q) (Line R S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_projections_perpendicular_l456_45605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_V_300_determinable_l456_45684

/-- An arithmetic sequence -/
noncomputable def arithmetic_sequence (b : ℝ) (r : ℝ) (n : ℕ) : ℝ := b + (n - 1) * r

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def U (b : ℝ) (r : ℝ) (n : ℕ) : ℝ := n * (2 * b + (n - 1) * r) / 2

/-- Sum of U_1 to U_n -/
noncomputable def V (b : ℝ) (r : ℝ) (n : ℕ) : ℝ := n * (n + 1) * (6 * b + r * (n - 1)) / 12

theorem V_300_determinable (b : ℝ) (r : ℝ) :
  ∃ (f : ℝ → ℝ), V b r 300 = f (U b r 150) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_V_300_determinable_l456_45684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_greater_than_seven_halves_l456_45699

/-- The function g(x) = x^2 - 3x + 2 * ln(x) -/
noncomputable def g (x : ℝ) : ℝ := x^2 - 3*x + 2 * Real.log x

/-- Theorem: If g(x₁) + g(x₂) = 0, then x₁ + x₂ > 7/2 -/
theorem sum_greater_than_seven_halves (x₁ x₂ : ℝ) (h : g x₁ + g x₂ = 0) : x₁ + x₂ > 7/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_greater_than_seven_halves_l456_45699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_is_real_l456_45608

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3*x + 8 - 2*x^2) / (x + 4)

-- Theorem statement
theorem range_of_g_is_real :
  ∀ y : ℝ, ∃ x : ℝ, x ≠ -4 ∧ g x = y :=
by
  sorry

#check range_of_g_is_real

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_is_real_l456_45608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_theorem_l456_45633

/-- Represents a point in 4D space -/
structure Point4D where
  x : ℝ
  y : ℝ
  z : ℝ
  u : ℝ

/-- Calculates the squared distance between two points in 4D space -/
noncomputable def sqDistance (p q : Point4D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2 + (p.u - q.u)^2

/-- Calculates the centroid of four points in 4D space -/
noncomputable def centroid (p₁ p₂ p₃ p₄ : Point4D) : Point4D :=
  { x := (p₁.x + p₂.x + p₃.x + p₄.x) / 4
    y := (p₁.y + p₂.y + p₃.y + p₄.y) / 4
    z := (p₁.z + p₂.z + p₃.z + p₄.z) / 4
    u := (p₁.u + p₂.u + p₃.u + p₄.u) / 4 }

/-- Theorem statement -/
theorem centroid_distance_theorem (p₁ p₂ p₃ p₄ p₅ : Point4D) :
  let s := centroid p₁ p₂ p₃ p₄
  sqDistance p₅ s = 1/4 * (sqDistance p₅ p₁ + sqDistance p₅ p₂ + sqDistance p₅ p₃ + sqDistance p₅ p₄) -
                    1/16 * (sqDistance p₁ p₂ + sqDistance p₁ p₃ + sqDistance p₁ p₄ +
                            sqDistance p₂ p₃ + sqDistance p₂ p₄ + sqDistance p₃ p₄) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_theorem_l456_45633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l456_45696

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
def triangle_condition (A B C a b c : ℝ) : Prop := 
  (2*a + c) * Real.cos B + b * Real.cos C = 0

def side_condition (a : ℝ) : Prop := a = 3

def area_condition (A B C a c : ℝ) : Prop := 
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2

-- Define the theorem
theorem triangle_theorem 
  (h1 : triangle_condition A B C a b c)
  (h2 : side_condition a)
  (h3 : area_condition A B C a c) :
  B = (2 * Real.pi) / 3 ∧ 
  a * c * Real.cos B = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l456_45696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pat_to_mark_ratio_l456_45644

/-- Represents the hours charged by Kate -/
def kate_hours : ℕ := sorry

/-- Represents the hours charged by Pat -/
def pat_hours : ℕ := 2 * kate_hours

/-- Represents the hours charged by Mark -/
def mark_hours : ℕ := kate_hours + 105

/-- The total hours charged by all three -/
def total_hours : ℕ := 189

/-- The theorem stating the ratio of Pat's hours to Mark's hours -/
theorem pat_to_mark_ratio :
  pat_hours * 3 = mark_hours ∧ total_hours = pat_hours + kate_hours + mark_hours :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pat_to_mark_ratio_l456_45644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_multiple_of_300_l456_45625

theorem factors_multiple_of_300 (n : ℕ) (h : n = 2^15 * 3^10 * 5^12) :
  (Finset.filter (fun x => x ∣ n ∧ 300 ∣ x) (Finset.range (n + 1))).card = 1540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_multiple_of_300_l456_45625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_curve_l456_45681

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = 4 - 2*x^2

-- Define point A
noncomputable def A : ℝ × ℝ := (0, -Real.sqrt 2)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_on_curve :
  ∃ (max_dist : ℝ), max_dist = 2 + Real.sqrt 2 ∧
    ∀ (x y : ℝ), C x y →
      distance (x, y) A ≤ max_dist ∧
      ∃ (x' y' : ℝ), C x' y' ∧ distance (x', y') A = max_dist :=
by
  sorry

#check max_distance_on_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_curve_l456_45681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daisy_petals_count_l456_45663

/-- Represents the number of petals on each daisy -/
def petals_per_daisy : ℕ := sorry

/-- The initial number of daisies Mabel has -/
def initial_daisies : ℕ := 5

/-- The number of daisies Mabel gives away -/
def given_away_daisies : ℕ := 2

/-- The total number of petals on the remaining daisies -/
def remaining_petals : ℕ := 24

theorem daisy_petals_count : 
  (initial_daisies - given_away_daisies) * petals_per_daisy = remaining_petals →
  petals_per_daisy = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daisy_petals_count_l456_45663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_odd_even_l456_45657

noncomputable def y (x φ : ℝ) : ℝ := 2 * Real.sin (3 * x + 2 * φ - Real.pi / 3)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem min_phi_odd_even (φ : ℝ) : 
  (φ > 0 ∧ is_odd_function (y · φ) → φ ≥ Real.pi / 6) ∧
  (φ > 0 ∧ is_even_function (y · φ) → φ ≥ 5 * Real.pi / 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_odd_even_l456_45657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l456_45617

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x - k * Real.sqrt (x^2 - 1)

-- State the theorem
theorem range_of_f (k : ℝ) (hk : 0 < k ∧ k < 1) :
  Set.range (fun x => f k x) = Set.Ici (Real.sqrt (1 - k^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l456_45617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_M_equals_one_two_l456_45648

def A : Set ℕ := {1, 2, 3}
def B : Set ℝ := {x : ℝ | x > 2}
def M : Set ℕ := {x ∈ A | x ∉ {y : ℕ | (y : ℝ) ∈ B}}

theorem set_M_equals_one_two : M = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_M_equals_one_two_l456_45648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l456_45630

theorem complex_number_problem (b : ℝ) (z ω : ℂ) :
  z = 3 + b * I ∧
  (1 + 3 * I) * z = I * Complex.im ((1 + 3 * I) * z) →
  z = 3 + I ∧
  ω = z / (2 + I) ∧
  ω = 7/5 - 1/5 * I ∧
  Complex.abs ω = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_problem_l456_45630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_prize_winners_equation_solutions_l456_45694

-- Define variables
variable (x y z a b : ℤ)

-- Define the conditions
def conditions (x y z a b : ℤ) : Prop :=
  (x + y + z + a + 20 = 40) ∧
  (y + a = 7) ∧
  (x + a = 10) ∧
  (z + a = 11) ∧
  (x + y + z + a + b + 20 = 51)

-- Define the theorem for the number of physics prize winners
theorem physics_prize_winners (x y z a b : ℤ) :
  conditions x y z a b → y + z + a + b = 25 := by sorry

-- Define the equation
noncomputable def equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 1 + Real.sqrt (4*x^2 + 4*y^2 - 34) = 2 * abs (x + y) - 2*x*y

-- Define the theorem for the equation solutions
theorem equation_solutions :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    equation x₁ y₁ ∧ equation x₂ y₂ ∧ equation x₃ y₃ ∧ equation x₄ y₄ ∧
    x₁ = 2.5 ∧ y₁ = -1.5 ∧
    x₂ = -2.5 ∧ y₂ = 1.5 ∧
    x₃ = 1.5 ∧ y₃ = -2.5 ∧
    x₄ = -1.5 ∧ y₄ = 2.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_prize_winners_equation_solutions_l456_45694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l456_45653

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two parallel planes -/
noncomputable def distance_between_planes (p1 p2 : Plane) : ℝ :=
  abs (p2.d / p1.a - p1.d / p1.a) / Real.sqrt (p1.a^2 + p1.b^2 + p1.c^2)

theorem distance_between_specific_planes :
  let p1 : Plane := { a := 1, b := 2, c := -2, d := -1 }
  let p2 : Plane := { a := 3, b := 6, c := -6, d := -7 }
  distance_between_planes p1 p2 = 4/3 := by
  sorry

#check distance_between_specific_planes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l456_45653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l456_45671

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

-- State the theorem
theorem function_property (m : ℝ) (a b : ℝ) :
  -- f is a power function
  (∃ k c : ℝ, ∀ x > 0, f m x = c * x^k) →
  -- f is monotonically increasing on (0, +∞)
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f m x₁ - f m x₂) / (x₁ - x₂) > 0) →
  -- Given condition
  f m a + f m b < 0 →
  -- Conclusion
  a + b < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l456_45671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_irrational_l456_45674

theorem unique_solution_iff_irrational (c : ℝ) :
  (∃! x : ℝ, 1 + Real.sin (c * x)^2 = Real.cos x) ↔ Irrational c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_iff_irrational_l456_45674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_22_l456_45682

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

/-- The specific triangle vertices -/
def p1 : ℝ × ℝ := (-4, 2)
def p2 : ℝ × ℝ := (2, 8)
def p3 : ℝ × ℝ := (-2, -2)

theorem triangle_area_is_22 : triangleArea p1 p2 p3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_22_l456_45682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_area_percentage_is_58_l456_45673

/-- Represents the properties of a checkered fabric -/
structure CheckeredFabric where
  totalSideLength : ℚ
  largeSideLength : ℚ
  smallSideLength : ℚ

/-- Calculates the percentage of white area in the checkered fabric -/
def whiteAreaPercentage (fabric : CheckeredFabric) : ℚ :=
  let totalArea := fabric.totalSideLength ^ 2
  let largeWhiteArea := fabric.largeSideLength ^ 2
  let smallWhiteArea := fabric.smallSideLength ^ 2
  let totalWhiteArea := largeWhiteArea + smallWhiteArea
  (totalWhiteArea / totalArea) * 100

/-- Theorem stating that the white area percentage of the given fabric is 58% -/
theorem white_area_percentage_is_58 (fabric : CheckeredFabric) 
  (h1 : fabric.totalSideLength = 20)
  (h2 : fabric.largeSideLength = 14)
  (h3 : fabric.smallSideLength = 6) :
  whiteAreaPercentage fabric = 58 := by
  sorry

#eval whiteAreaPercentage { totalSideLength := 20, largeSideLength := 14, smallSideLength := 6 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_area_percentage_is_58_l456_45673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l456_45604

theorem negation_of_existence :
  (∃ x : ℝ, (4 : ℝ) ^ x > x ^ 4) ↔ ¬(∀ x : ℝ, (4 : ℝ) ^ x ≤ x ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_l456_45604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_x_15_minus_x_l456_45669

theorem factor_count_x_15_minus_x : 
  ∃ (f₁ f₂ f₃ f₄ f₅ : Polynomial ℤ), 
    (∀ i ∈ ({f₁, f₂, f₃, f₄, f₅} : Set (Polynomial ℤ)), Irreducible i) ∧ 
    (X^15 - X : Polynomial ℤ) = f₁ * f₂ * f₃ * f₄ * f₅ ∧
    (∀ (g₁ g₂ g₃ g₄ g₅ g₆ : Polynomial ℤ), 
      (X^15 - X : Polynomial ℤ) ≠ g₁ * g₂ * g₃ * g₄ * g₅ * g₆) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_x_15_minus_x_l456_45669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l456_45606

/-- Given vectors a, b, c in ℝ², if a + b = λc for some λ ∈ ℝ, then λ + x = -29/2 -/
theorem vector_equation_solution (a b c : ℝ × ℝ) (l : ℝ) (x : ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (-3, 5)) 
  (hc : c = (4, x)) 
  (heq : a + b = l • c) : 
  l + x = -29/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l456_45606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l456_45609

/-- The polynomial P(x) -/
def P (a b x : ℚ) : ℚ := (a + b) * x^5 + a * b * x^2 + 1

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := x^2 - 3*x + 2

/-- Theorem stating the divisibility condition -/
theorem divisibility_condition (a b : ℚ) :
  (∀ x, (divisor x) ∣ (P a b x)) ↔ 
  ((a = -1 ∧ b = 31/28) ∨ (a = 31/28 ∧ b = -1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l456_45609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l456_45667

/-- A vector in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The dot product of two 2D vectors -/
noncomputable def dot (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- The squared norm of a 2D vector -/
noncomputable def norm_sq (v : Vector2D) : ℝ := dot v v

/-- A vector on the line y = 3/2x + 3 -/
noncomputable def vector_on_line (a : ℝ) : Vector2D := ⟨a, 3/2 * a + 3⟩

/-- The projection vector u -/
noncomputable def u (d : ℝ) : Vector2D := ⟨-3/2 * d, d⟩

/-- The projection of a vector v onto u -/
noncomputable def proj (v : Vector2D) (u : Vector2D) : Vector2D :=
  let scalar := (dot v u) / (norm_sq u)
  ⟨scalar * u.x, scalar * u.y⟩

theorem projection_theorem (a : ℝ) (d : ℝ) :
  proj (vector_on_line a) (u d) = ⟨-18/13, 12/13⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l456_45667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_score_product_l456_45628

def basketball_scores (scores : List Nat) : Prop :=
  scores.length = 10 ∧
  scores.take 8 = [7, 4, 3, 6, 8, 3, 1, 5] ∧
  scores[8]! < 10 ∧
  scores[9]! < 10 ∧
  (scores.take 9).sum % 9 = 0 ∧
  scores.sum % 10 = 0

theorem basketball_score_product (scores : List Nat) :
  basketball_scores scores → scores[8]! * scores[9]! = 40 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_score_product_l456_45628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l456_45656

theorem train_length_calculation (jogger_speed train_speed initial_distance passing_time : ℝ) :
  jogger_speed = 9 →
  train_speed = 45 →
  initial_distance = 150 →
  passing_time = 25 →
  let jogger_speed_ms : ℝ := jogger_speed * 1000 / 3600
  let train_speed_ms : ℝ := train_speed * 1000 / 3600
  let relative_speed : ℝ := train_speed_ms - jogger_speed_ms
  let distance_covered : ℝ := relative_speed * passing_time
  let train_length : ℝ := distance_covered - initial_distance
  train_length = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l456_45656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_through_point_l456_45689

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ a^x

-- State the theorem
theorem function_through_point (a : ℝ) :
  (f a 2 = 4) → (f 2 = f a) := by
  intro h
  -- The proof goes here
  sorry

#check function_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_through_point_l456_45689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l456_45613

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Calculates the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Represents a triangle with base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  (1/2) * t.base * t.height

theorem ellipse_properties (e : Ellipse) 
  (h_minor_axis : e.b * 2 = Real.sqrt 3 * 2)
  (h_eccentricity : eccentricity e = 1/2) :
  (∃ (x y : ℝ), x^2/4 + y^2/3 = 1) ∧
  (∃ (t : Triangle), triangle_area t = 2 * Real.sqrt 3) := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l456_45613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_with_tide_l456_45677

/-- The distance a man can row in 60 minutes with the help of the tide -/
def D : ℝ := sorry

/-- The man's speed in still water -/
def v_m : ℝ := sorry

/-- The speed of the tide -/
def v_t : ℝ := sorry

/-- The man can row distance D in 60 minutes with the help of the tide -/
axiom condition1 : D = v_m + v_t

/-- The man travels 30 km in 10 hours against the tide -/
axiom condition2 : 30 = 10 * (v_m - v_t)

/-- If the tide hadn't changed, the man would have covered 30 km in 6 hours -/
axiom condition3 : 30 = 6 * (v_m + v_t)

/-- The distance D that the man can row in 60 minutes with the help of the tide is 5 km -/
theorem distance_with_tide : D = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_with_tide_l456_45677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grand_canyon_trip_cost_l456_45603

/-- Calculates the cost of a round trip given car efficiency, distance, and gas price -/
noncomputable def round_trip_cost (city_efficiency highway_efficiency : ℝ) 
                    (city_distance highway_distance : ℝ) 
                    (gas_price : ℝ) : ℝ :=
  let city_gallons := 2 * city_distance / city_efficiency
  let highway_gallons := 2 * highway_distance / highway_efficiency
  let total_gallons := city_gallons + highway_gallons
  total_gallons * gas_price

/-- Proves that the round trip cost to the Grand Canyon is $42.00 -/
theorem grand_canyon_trip_cost : 
  round_trip_cost 30 40 60 200 3 = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grand_canyon_trip_cost_l456_45603
