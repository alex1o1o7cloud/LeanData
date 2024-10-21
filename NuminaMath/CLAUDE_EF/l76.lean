import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arithmetic_sequence_l76_7637

theorem cos_arithmetic_sequence (a : Real) :
  0 < a ∧ a < 2 * Real.pi →
  (Real.cos a + Real.cos (5 * a) = 2 * Real.cos (3 * a)) ↔ (a = Real.pi / 2 ∨ a = 3 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arithmetic_sequence_l76_7637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_over_fifth_root_of_nine_l76_7688

theorem cube_root_over_fifth_root_of_nine :
  (9 : ℝ) ^ (1/3) / (9 : ℝ) ^ (1/5) = 9 ^ (2/15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_over_fifth_root_of_nine_l76_7688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_rowing_distance_l76_7666

/-- Calculates the upstream distance rowed given downstream distance, time, and stream speed -/
noncomputable def upstream_distance (downstream_distance : ℝ) (time : ℝ) (stream_speed : ℝ) : ℝ :=
  (downstream_distance / time - stream_speed) * time - stream_speed * time

theorem upstream_rowing_distance :
  let downstream_distance : ℝ := 60
  let time : ℝ := 3
  let stream_speed : ℝ := 5
  upstream_distance downstream_distance time stream_speed = 30 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_rowing_distance_l76_7666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l76_7635

/-- The function f(x) = x^2 / (x - 3) for x > 3 -/
noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 3)

/-- The domain condition x > 3 -/
def domain (x : ℝ) : Prop := x > 3

theorem f_minimum_value :
  ∃ (min : ℝ), min = 12 ∧ ∀ x, domain x → f x ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l76_7635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_rectangle_l76_7672

/-- Definition: Four points form a rectangle if opposite sides are equal and adjacent sides are perpendicular -/
def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  let dist := λ (P Q : ℝ × ℝ) => Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let perpendicular := λ (P Q R S : ℝ × ℝ) => (Q.1 - P.1) * (S.1 - R.1) + (Q.2 - P.2) * (S.2 - R.2) = 0
  dist A B = dist C D ∧
  dist A D = dist B C ∧
  perpendicular A B B C ∧
  perpendicular B C C D ∧
  perpendicular C D D A ∧
  perpendicular D A A B

/-- The points of intersection of the curves xy = 16 and x^2 + y^2 = 34 form a rectangle -/
theorem intersection_points_form_rectangle :
  ∃ (A B C D : ℝ × ℝ),
    (A.1 * A.2 = 16 ∧ A.1^2 + A.2^2 = 34) ∧
    (B.1 * B.2 = 16 ∧ B.1^2 + B.2^2 = 34) ∧
    (C.1 * C.2 = 16 ∧ C.1^2 + C.2^2 = 34) ∧
    (D.1 * D.2 = 16 ∧ D.1^2 + D.2^2 = 34) ∧
    is_rectangle A B C D :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_rectangle_l76_7672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_translation_phi_value_exists_l76_7634

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

noncomputable def translated_f (x φ : ℝ) : ℝ := f (x + Real.pi/8) φ

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem sin_graph_translation (φ : ℝ) : 
  is_even (translated_f · φ) → ∃ k : ℤ, φ = Real.pi/4 + k * Real.pi :=
sorry

theorem phi_value_exists : 
  ∃ φ, is_even (translated_f · φ) ∧ φ = Real.pi/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_translation_phi_value_exists_l76_7634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_overlapping_caps_max_rays_l76_7631

-- Define the radius of the spherical cap
noncomputable def r : ℝ := Real.pi / 8

-- Define the area of a spherical cap
noncomputable def spherical_cap_area (r : ℝ) : ℝ := 2 * Real.pi * (1 - Real.cos r)

-- Define the surface area of a unit sphere
noncomputable def unit_sphere_area : ℝ := 4 * Real.pi

-- Theorem statement
theorem max_non_overlapping_caps :
  (⌊unit_sphere_area / spherical_cap_area r⌋ : ℤ) ≤ 26 := by
  sorry

-- Additional theorem to connect the floor of the ratio to the number of rays
theorem max_rays :
  ∀ n : ℕ, n > (⌊unit_sphere_area / spherical_cap_area r⌋ : ℤ) → n > 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_overlapping_caps_max_rays_l76_7631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_right_angled_l76_7606

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively, 
    if a = 2b*sin(A) and b^2 + c^2 - a^2 = bc, then ABC is a right triangle. -/
theorem triangle_is_right_angled 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : ∀ {x y z : ℝ}, x + y > z ∧ y + z > x ∧ z + x > y)
  (h_angles : A + B + C = Real.pi)
  (h_side_a : a = 2 * b * Real.sin A)
  (h_sides : b^2 + c^2 - a^2 = b * c) :
  ∃ (x : ℝ), x^2 + x^2 = (2*x*Real.sin A)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_right_angled_l76_7606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l76_7620

-- Define the function f with domain (0,4]
def f : Set ℝ := Set.Ioo 0 4

-- Define the function g
def g (x : ℝ) : Set ℝ := {y | ∃ (y1 y2 : ℝ), y1 ∈ f ∧ y2 ∈ f ∧ y = y1 + y2 ∧ x ∈ f ∧ x^2 ∈ f}

-- Theorem statement
theorem domain_of_g :
  Set.Ioo 0 2 = {x : ℝ | x ∈ f ∧ x^2 ∈ f} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l76_7620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairwise_different_plants_no_exactly_fifty_plants_l76_7685

/-- Represents a plant with 100 binary features -/
def Plant := Fin 100 → Bool

/-- Two plants are considered different if they differ in at least 51 features -/
def are_different (p1 p2 : Plant) : Prop :=
  (Finset.filter (fun i => p1 i ≠ p2 i) (Finset.univ : Finset (Fin 100))).card ≥ 51

/-- A set of pairwise different plants -/
def PairwiseDifferentPlants (s : Set Plant) : Prop :=
  ∀ p1 p2 : Plant, p1 ∈ s → p2 ∈ s → p1 ≠ p2 → are_different p1 p2

theorem max_pairwise_different_plants :
  ∀ s : Set Plant, PairwiseDifferentPlants s → s.ncard ≤ 50 := by
  sorry

theorem no_exactly_fifty_plants :
  ¬∃ s : Set Plant, PairwiseDifferentPlants s ∧ s.ncard = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairwise_different_plants_no_exactly_fifty_plants_l76_7685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_male_percentage_is_40_l76_7670

/-- Represents the percentage of male students in the class -/
noncomputable def male_percentage : ℝ := sorry

/-- Represents the percentage of female students in the class -/
noncomputable def female_percentage : ℝ := sorry

/-- The sum of male and female percentages is 100% -/
axiom total_percentage : male_percentage + female_percentage = 100

/-- 60% of male students are 25 years old or older -/
def male_older_percentage : ℝ := 60

/-- 40% of female students are 25 years old or older -/
def female_older_percentage : ℝ := 40

/-- Probability of selecting a student less than 25 years old -/
noncomputable def prob_younger : ℝ := sorry

axiom prob_younger_value : prob_younger = 0.52

/-- The probability of selecting a younger student is a weighted average -/
axiom prob_younger_equation : 
  prob_younger = (100 - male_older_percentage) / 100 * male_percentage / 100 + 
                 (100 - female_older_percentage) / 100 * female_percentage / 100

/-- Theorem: The percentage of male students in the class is 40% -/
theorem male_percentage_is_40 : male_percentage = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_male_percentage_is_40_l76_7670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l76_7616

-- Define a line by its equation coefficients
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to calculate x-intercept of a line
noncomputable def xIntercept (l : Line) : ℝ :=
  -l.c / l.a

-- Function to calculate y-intercept of a line
noncomputable def yIntercept (l : Line) : ℝ :=
  -l.c / l.b

-- Theorem statement
theorem line_equation : 
  ∃ (l1 l2 : Line), 
    -- The lines pass through the point (5,6)
    pointOnLine ⟨5, 6⟩ l1 ∧ 
    pointOnLine ⟨5, 6⟩ l2 ∧ 
    -- The x-intercept is twice the y-intercept for both lines
    xIntercept l1 = 2 * yIntercept l1 ∧
    xIntercept l2 = 2 * yIntercept l2 ∧
    -- The lines have the equations x+2y-17=0 and 6x-5y=0
    (l1.a = 1 ∧ l1.b = 2 ∧ l1.c = -17) ∧
    (l2.a = 6 ∧ l2.b = -5 ∧ l2.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l76_7616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l76_7689

/-- Defines the minimum distance from the center of symmetry to the axis of symmetry for a function. -/
noncomputable def min_distance_center_to_axis (f : ℝ → ℝ) : ℝ :=
sorry

/-- Defines when a real number is an x-coordinate of an axis of symmetry for a function. -/
def is_axis_of_symmetry (f : ℝ → ℝ) (x : ℝ) : Prop :=
sorry

/-- Defines when a point is a center of symmetry for a function. -/
def is_center_of_symmetry (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
sorry

/-- Given a function f and a real number ω, proves properties about f when certain conditions are met. -/
theorem function_properties (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.cos (2 * ω * x - π / 3) - 2 * (Real.cos (ω * x))^2 + 2) →
  (∃ d, d = π / 4 ∧ d = min_distance_center_to_axis f) →
  (ω = 1) ∧
  (∀ k : ℤ, is_axis_of_symmetry f ((1 / 2 : ℝ) * k * π + π / 3)) ∧
  (∀ k : ℤ, is_center_of_symmetry f (k * π / 2 + π / 12, 1)) ∧
  (∀ y ∈ Set.Icc (-Real.sqrt 3 / 2 + 1) 2, ∃ x ∈ Set.Icc (-π / 12) (π / 2), f x = y) ∧
  (∀ x ∈ Set.Icc (-π / 12) (π / 2), f x ∈ Set.Icc (-Real.sqrt 3 / 2 + 1) 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l76_7689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_theorem_l76_7651

/-- Represents the minimum number of workers needed for profit -/
noncomputable def min_workers_for_profit (maintenance_fee : ℝ) (hourly_wage : ℝ) (pens_per_hour : ℝ) 
  (pen_price : ℝ) (work_hours : ℝ) : ℕ :=
  Int.natAbs (Int.ceil (maintenance_fee / (pens_per_hour * pen_price * work_hours - hourly_wage * work_hours)))

/-- Theorem stating the minimum number of workers needed for profit -/
theorem min_workers_theorem :
  min_workers_for_profit 600 20 7 2.80 9 = 167 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_workers_theorem_l76_7651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l76_7664

theorem tan_difference (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l76_7664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_mean_inequality_weighted_power_mean_inequality_l76_7647

theorem power_mean_inequality (n : ℕ) (x₁ x₂ : ℝ) (q₁ q₂ : ℝ) 
  (hn : n > 0) (hq : q₁ + q₂ = 1) (hq₁ : 0 ≤ q₁) (hq₂ : 0 ≤ q₂) :
  ((q₁ * x₁^n + q₂ * x₂^n) / 2)^(1/(n:ℝ)) ≤ ((q₁ * x₁^(n+1) + q₂ * x₂^(n+1)) / 2)^(1/((n+1):ℝ)) :=
sorry

theorem weighted_power_mean_inequality (n : ℕ) (x₁ x₂ : ℝ) (q₁ q₂ : ℝ) 
  (hn : n > 0) (hq : q₁ + q₂ = 1) (hq₁ : 0 ≤ q₁) (hq₂ : 0 ≤ q₂) :
  (q₁ * x₁^n + q₂ * x₂^n)^(1/(n:ℝ)) ≤ (q₁ * x₁^(n+1) + q₂ * x₂^(n+1))^(1/((n+1):ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_mean_inequality_weighted_power_mean_inequality_l76_7647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expressions_equality_l76_7663

theorem complex_expressions_equality : 
  (∃ x : ℝ, (Real.sqrt 27 * Real.sqrt 2) / Real.sqrt (2/3) - (Real.sqrt 12 + 3 * Real.sqrt 6) * Real.sqrt 3 = 3 - 9 * Real.sqrt 2) ∧
  (∃ y : ℝ, (3 - Real.sqrt 2)^2 - (Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3) = 9 - 6 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expressions_equality_l76_7663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l76_7629

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - 2*x - 15)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < 5) ∨ 5 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l76_7629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_arg_l76_7695

/-- The sum of complex exponentials -/
noncomputable def complex_sum : ℂ :=
  Complex.exp (11 * Real.pi * Complex.I / 60) +
  Complex.exp (23 * Real.pi * Complex.I / 60) +
  Complex.exp (35 * Real.pi * Complex.I / 60) +
  Complex.exp (47 * Real.pi * Complex.I / 60) +
  Complex.exp (59 * Real.pi * Complex.I / 60)

/-- The theorem stating that the argument of the complex sum is 35π/60 -/
theorem complex_sum_arg :
  0 ≤ Complex.arg complex_sum ∧
  Complex.arg complex_sum < 2 * Real.pi ∧
  Complex.arg complex_sum = 35 * Real.pi / 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_arg_l76_7695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l76_7646

noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) / (x^2 - 16)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -4 ∨ (-4 < x ∧ x < 4) ∨ 4 < x} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l76_7646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_graph_l76_7650

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem symmetry_of_sine_graph :
  ∀ (x : ℝ), f (Real.pi / 6 - x) = f (Real.pi / 6 + x) := by
  intro x
  simp [f]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_graph_l76_7650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_l76_7680

-- Define the necessary structures
structure Line where

structure Plane where

-- Define the relationships
def perpendicular_to_plane (l : Line) (p : Plane) : Prop :=
  sorry

def parallel (l1 l2 : Line) : Prop :=
  sorry

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (γ : Plane) :
  m ≠ n →
  perpendicular_to_plane m γ →
  perpendicular_to_plane n γ →
  parallel m n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_parallel_l76_7680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_ball_is_0_36_l76_7655

/-- Represents a basket with balls of different colors -/
structure Basket where
  white : ℕ
  red : ℕ
  yellow : ℕ
  black : ℕ

/-- The probability of picking a red ball from two baskets -/
noncomputable def prob_red_ball (basketA basketB : Basket) (probA probB : ℝ) : ℝ :=
  let totalA := basketA.white + basketA.red
  let totalB := basketB.yellow + basketB.red + basketB.black
  (basketA.red : ℝ) / totalA * probA + (basketB.red : ℝ) / totalB * probB

/-- Theorem: The probability of picking a red ball is 0.36 -/
theorem prob_red_ball_is_0_36 :
  let basketA : Basket := ⟨10, 5, 0, 0⟩
  let basketB : Basket := ⟨0, 6, 4, 5⟩
  prob_red_ball basketA basketB 0.6 0.4 = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_ball_is_0_36_l76_7655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_eq_3_range_of_a_for_f_geq_2_l76_7638

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for the first part of the problem
theorem solution_set_for_a_eq_3 :
  {x : ℝ | f x 3 ≥ 5} = Set.Iic (-1/2) ∪ Set.Ici (9/2) := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_f_geq_2 :
  ∀ a : ℝ, (∀ x : ℝ, f x a ≥ 2) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_eq_3_range_of_a_for_f_geq_2_l76_7638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l76_7641

noncomputable def f (x : ℝ) := Real.exp x + x

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-1) 1 ∧
  f x = Real.exp 1 + 1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-1) 1 → f y ≤ Real.exp 1 + 1 := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l76_7641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_ratio_sum_l76_7681

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the radii of circumcircle and incircle
variable (R r : ℝ)

-- Define the incenter
variable (I : ℝ × ℝ)

-- Define the intersection points
variable (D D' E E' F F' : ℝ × ℝ)

-- State the theorem
theorem triangle_circle_intersection_ratio_sum :
  let DD' := dist D D'
  let D'A := dist D' A
  let EE' := dist E E'
  let E'B := dist E' B
  let FF' := dist F F'
  let F'C := dist F' C
  (DD' / D'A + EE' / E'B + FF' / F'C = (R - r) / r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_ratio_sum_l76_7681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l76_7642

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The sum of the specific geometric series 3 + 3(1/3) + 3(1/3)² + 3(1/3)³ + ... -/
noncomputable def specific_sum : ℝ := geometric_sum 3 (1/3)

theorem geometric_series_sum : specific_sum = 9/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l76_7642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_minimum_value_bound_l76_7614

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_minimum_value_bound 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (hf'0 : 2 * a * 0 + b > 0)
  (hnonneg : ∀ x, QuadraticFunction a b c x ≥ 0) :
  ∀ y : ℝ, y = (QuadraticFunction a b c 1) / (2 * a * 0 + b) → y ≥ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_minimum_value_bound_l76_7614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_count_l76_7668

/-- The count of positive integers less than 800 that can be written as the sum of two positive perfect squares -/
def count_sum_squares : ℕ :=
  Finset.card (Finset.filter (fun p : ℕ × ℕ => 
    1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ 28 ∧ 
    p.1^2 + p.2^2 < 800) (Finset.product (Finset.range 29) (Finset.range 29)))

theorem sum_squares_count :
  count_sum_squares = 
  (Finset.range 28).card.choose 2 - 
  Finset.card (Finset.filter (fun p : ℕ × ℕ => 
    1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ 28 ∧ 
    p.1^2 + p.2^2 ≥ 800) (Finset.product (Finset.range 29) (Finset.range 29))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_count_l76_7668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_ratio_l76_7699

/-- Given a triangle ABC with points M and N, prove that if AM = 2MC and BN = 3NC,
    then for MN = xAB + yAC, x/y = 3 -/
theorem triangle_vector_ratio (A B C M N : EuclideanSpace ℝ (Fin 2)) 
  (h1 : A - M = 2 • (M - C))
  (h2 : B - N = 3 • (N - C))
  (h3 : ∃ x y : ℝ, M - N = x • (A - B) + y • (A - C)) :
  ∃ x y : ℝ, M - N = x • (A - B) + y • (A - C) ∧ x / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_ratio_l76_7699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_4_sup_a_for_f_minus_a_geq_0_l76_7639

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 3|

-- Theorem for the solution set of f(x) ≤ 4
theorem solution_set_f_leq_4 :
  {x : ℝ | f x ≤ 4} = Set.Icc (-8) 2 := by sorry

-- Theorem for the supremum of a
theorem sup_a_for_f_minus_a_geq_0 :
  ⨆ (a : ℝ) (h : ∀ x, f x - a ≥ 0), a = -7/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_4_sup_a_for_f_minus_a_geq_0_l76_7639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_determine_plane_three_points_not_always_unique_two_lines_not_always_unique_line_and_point_not_always_unique_l76_7636

-- Define the basic types
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line3D where
  point : Point3D
  direction : Point3D

structure Plane where
  normal : Point3D
  d : ℝ

-- Define parallel lines
def parallel (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, l1.direction = { x := k * l2.direction.x, y := k * l2.direction.y, z := k * l2.direction.z }

-- Define membership for Point3D in Line3D and Plane
def pointInLine (pt : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, pt = { x := l.point.x + t * l.direction.x,
                  y := l.point.y + t * l.direction.y,
                  z := l.point.z + t * l.direction.z }

def pointInPlane (pt : Point3D) (p : Plane) : Prop :=
  p.normal.x * pt.x + p.normal.y * pt.y + p.normal.z * pt.z = p.d

-- State the theorem
theorem parallel_lines_determine_plane (l1 l2 : Line3D) :
  parallel l1 l2 → ∃! p : Plane, (∀ pt : Point3D, pointInLine pt l1 → pointInPlane pt p) ∧
                                 (∀ pt : Point3D, pointInLine pt l2 → pointInPlane pt p) :=
sorry

-- Counter-examples for other conditions
theorem three_points_not_always_unique (p1 p2 p3 : Point3D) :
  ¬(∃! p : Plane, pointInPlane p1 p ∧ pointInPlane p2 p ∧ pointInPlane p3 p) :=
sorry

theorem two_lines_not_always_unique (l1 l2 : Line3D) :
  ¬(∃! p : Plane, (∀ pt : Point3D, pointInLine pt l1 → pointInPlane pt p) ∧
                  (∀ pt : Point3D, pointInLine pt l2 → pointInPlane pt p)) :=
sorry

theorem line_and_point_not_always_unique (l : Line3D) (pt : Point3D) :
  ¬(∃! p : Plane, (∀ pt' : Point3D, pointInLine pt' l → pointInPlane pt' p) ∧ pointInPlane pt p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_determine_plane_three_points_not_always_unique_two_lines_not_always_unique_line_and_point_not_always_unique_l76_7636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l76_7673

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det (B^2 - 3 • B) = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_squared_minus_3B_l76_7673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroids_form_line_l76_7610

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- A triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.a.x + t.b.x + t.c.x) / 3
    y := (t.a.y + t.b.y + t.c.y) / 3 }

/-- Check if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  ∃ t : ℝ, p = { x := l.p1.x + t * (l.p2.x - l.p1.x), y := l.p1.y + t * (l.p2.y - l.p1.y) }

/-- A set of triangles sharing a common base -/
def TrianglesWithCommonBase (base : Line) (vertices : Line) :=
  {t : Triangle | t.a = base.p1 ∧ t.b = base.p2 ∧ t.c.onLine vertices}

/-- Theorem: Centroids of triangles with common base and opposite vertices on a line form a line -/
theorem centroids_form_line (base : Line) (vertices : Line) :
  ∃ l : Line, ∀ t : Triangle, t ∈ TrianglesWithCommonBase base vertices →
    (centroid t).onLine l := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroids_form_line_l76_7610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_queen_diamond_is_one_52_l76_7632

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)
  (valid : ∀ c ∈ cards, c.1 ∈ Finset.range 13 ∧ c.2 ∈ Finset.range 4)

/-- Represents the event of drawing a Queen first and a Diamond second -/
def queen_diamond_event (d : Deck) : Finset (Nat × Nat) :=
  d.cards.filter (fun c => c.1 = 11 ∨ (c.1 ≠ 11 ∧ c.2 = 1))

/-- The probability of the queen_diamond_event -/
def prob_queen_diamond (d : Deck) : Rat :=
  (queen_diamond_event d).card / d.cards.card

theorem prob_queen_diamond_is_one_52 (d : Deck) :
  prob_queen_diamond d = 1 / 52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_queen_diamond_is_one_52_l76_7632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_symmetry_l76_7604

-- Define a type for points on a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define the theorem
theorem closest_point_symmetry (K P : Fin 8 → Point) :
  (∀ i : Fin 8, ∀ j : Fin 8, j ≠ i → distance (P i) (K i) ≤ distance (P i) (K j)) →
  (∀ j : Fin 8, j ≠ 0 → distance (P 0) (K 0) ≤ distance (P 0) (K j)) :=
by
  sorry

#check closest_point_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_symmetry_l76_7604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_count_l76_7684

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- Checks if three points are aligned --/
def aligned (p1 p2 p3 : Fin 5 × Fin 5) : Prop :=
  (p1.1 = p2.1 ∧ p2.1 = p3.1) ∨ -- vertical
  (p1.2 = p2.2 ∧ p2.2 = p3.2) ∨ -- horizontal
  (p1.1 - p2.1 = p2.1 - p3.1 ∧ p1.2 - p2.2 = p2.2 - p3.2) -- diagonal

/-- Checks if a grid configuration is valid (no three X's aligned) --/
def valid_configuration (g : Grid) : Prop :=
  ∀ p1 p2 p3 : Fin 5 × Fin 5,
    g p1.1 p1.2 ∧ g p2.1 p2.2 ∧ g p3.1 p3.2 → ¬aligned p1 p2 p3

/-- Counts the number of X's in a grid --/
def count_x (g : Grid) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 5)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 5)) fun j =>
      if g i j then 1 else 0

/-- The main theorem: maximum number of X's is 12 --/
theorem max_x_count :
  (∃ g : Grid, valid_configuration g ∧ count_x g = 12) ∧
  (∀ g : Grid, valid_configuration g → count_x g ≤ 12) := by
  sorry

#check max_x_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_count_l76_7684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lyle_friends_served_l76_7669

/-- The number of friends who can have a sandwich and a pack of juice --/
def friends_served (sandwich_cost juice_cost total_money : ℚ) : ℕ :=
  (((total_money / (sandwich_cost + juice_cost)).floor : ℤ) - 1).toNat

/-- Theorem stating the number of friends Lyle can serve --/
theorem lyle_friends_served :
  friends_served (30/100) (20/100) (250/100) = 4 := by
  rfl

#eval friends_served (30/100) (20/100) (250/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lyle_friends_served_l76_7669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_color_class_l76_7683

/-- A coloring of integers from 1 to n satisfying the given condition -/
def ValidColoring (n : ℕ) (color : ℕ → ℕ) : Prop :=
  ∀ a b, 0 < a → a < b → a + b ≤ n →
    (color a = color b ∨ color a = color (a + b) ∨ color b = color (a + b))

/-- The main theorem -/
theorem exists_large_color_class (n : ℕ) (color : ℕ → ℕ) (hn : n > 0) 
  (hcolor : ValidColoring n color) :
  ∃ c : ℕ, (Finset.filter (λ i => color i = c) (Finset.range n)).card ≥ (2 * n) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_color_class_l76_7683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_theorem_l76_7698

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

theorem ellipse_circle_theorem (e : Ellipse) 
  (A B M N : Point) (F : Point) :
  ellipse_equation e A ∧ 
  ellipse_equation e B ∧
  ellipse_equation e M ∧
  ellipse_equation e N ∧
  A.x = -2 ∧ A.y = 0 ∧
  F.x = 1 ∧ F.y = 0 ∧
  B.x = 3/5 * M.x + 4/5 * N.x ∧
  B.y = 3/5 * M.y + 4/5 * N.y →
  ∃ (C : Point),
    (C.x = -1/2) ∧
    ((C.y = Real.sqrt 21/4) ∨ (C.y = -Real.sqrt 21/4)) ∧
    ∀ (P : Point),
      (P.x - C.x)^2 + (P.y - C.y)^2 = 57/16 ↔
      ((P = A) ∨ (P = F) ∨ 
       ∃ (t : ℝ), P.x = (1 - t) * M.x + t * N.x ∧
                  P.y = (1 - t) * M.y + t * N.y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_theorem_l76_7698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l76_7677

/-- Calculates the cost price given the selling price and profit percentage -/
noncomputable def costPrice (sellingPrice : ℝ) (profitPercentage : ℝ) : ℝ :=
  sellingPrice / (1 + profitPercentage / 100)

/-- Proves that the cost price is approximately $83.33 given the conditions -/
theorem cost_price_calculation (sellingPrice : ℝ) (profitPercentage : ℝ) 
  (h1 : sellingPrice = 100) 
  (h2 : profitPercentage = 20) :
  ∃ ε > 0, |costPrice sellingPrice profitPercentage - 83.33| < ε :=
by
  sorry

/-- Computes an approximation of the cost price -/
def approxCostPrice : ℚ := 
  (100 : ℚ) / (1 + 20 / 100)

#eval approxCostPrice

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l76_7677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l76_7687

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the point A
def A : ℝ × ℝ := (0, -2)

-- Define the focus F
structure Focus (e : Ellipse) where
  x : ℝ
  h : x > 0

-- Define the line AF
noncomputable def slope_AF (e : Ellipse) (f : Focus e) : ℝ := 2 * Real.sqrt 3 / 3

-- Define the ratio of long axis to short axis
def axis_ratio (e : Ellipse) : ℝ := 2

-- Define the equation of line l
structure Line where
  k : ℝ
  b : ℝ

-- Define the area of triangle POQ
noncomputable def area_POQ (e : Ellipse) (l : Line) : ℝ := sorry

theorem ellipse_and_max_area_line 
  (e : Ellipse) 
  (f : Focus e) 
  (h1 : axis_ratio e = 2) 
  (h2 : slope_AF e f = 2 * Real.sqrt 3 / 3) :
  (∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  ∃ k : ℝ, 
    (∀ l : Line, area_POQ e l ≤ area_POQ e { k := k, b := -2 }) ∧
    (∀ l : Line, area_POQ e l ≤ area_POQ e { k := -k, b := -2 }) ∧
    k^2 = 7/4 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l76_7687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_120_l76_7628

-- Define the shapes
structure Square where
  side : ℝ
  height : ℝ

structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

structure Parallelogram where
  base : ℝ
  height : ℝ

-- Define the problem setup
def problem_setup (s : Square) (t : IsoscelesTriangle) (p : Parallelogram) : Prop :=
  s.side = 12 ∧
  s.height = 12 ∧
  t.base = 12 ∧
  t.height = s.height ∧
  p.base = 8 ∧
  p.height = t.height

-- Define the shaded area
noncomputable def shaded_area (s : Square) (t : IsoscelesTriangle) (p : Parallelogram) : ℝ :=
  let base := s.side + t.base + p.base
  let height := s.height
  (1 / 2) * base * height

-- Theorem statement
theorem shaded_area_is_120 (s : Square) (t : IsoscelesTriangle) (p : Parallelogram) :
  problem_setup s t p → shaded_area s t p = 120 := by
  intro h
  simp [problem_setup, shaded_area] at *
  -- The actual proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_120_l76_7628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hazel_speed_is_59_l76_7682

/-- Hazel's running speed in meters per second -/
noncomputable def hazel_speed (distance_km : ℝ) (time_min : ℝ) : ℝ :=
  (distance_km * 1000) / (time_min * 60)

/-- Theorem stating Hazel's speed is 59 m/s given the conditions -/
theorem hazel_speed_is_59 :
  hazel_speed 17.7 5 = 59 := by
  -- Unfold the definition of hazel_speed
  unfold hazel_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hazel_speed_is_59_l76_7682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_is_810_l76_7625

/-- Represents the number of days it takes for a worker to complete the entire job alone -/
structure WorkerEfficiency where
  days : ℚ
  days_positive : days > 0

/-- Represents the payment information for a worker -/
structure WorkerPayment where
  daily_wage : ℚ
  days_worked : ℚ
  days_worked_nonnegative : days_worked ≥ 0

/-- Calculates the total payment for the entire work -/
noncomputable def calculate_total_payment (a b : WorkerEfficiency) (b_payment : WorkerPayment) : ℚ :=
  let total_work_fraction := 5 * (1 / a.days + 1 / b.days)
  let b_work_fraction := 5 * (1 / b.days)
  (b_payment.daily_wage * b_payment.days_worked) * (1 / b_work_fraction)

/-- Theorem stating that the total payment for the work is 810 -/
theorem total_payment_is_810 
  (a : WorkerEfficiency)
  (b : WorkerEfficiency)
  (b_payment : WorkerPayment)
  (ha : a.days = 12)
  (hb : b.days = 15)
  (hb_days_worked : b_payment.days_worked = 5)
  (hb_daily_wage : b_payment.daily_wage = 54) :
  calculate_total_payment a b b_payment = 810 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_is_810_l76_7625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l76_7645

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-3) 0 then -2 - x
  else if x ∈ Set.Icc 0 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if x ∈ Set.Icc 2 3 then 2 * (x - 2)
  else 0  -- undefined outside [-3, 3]

-- Define the function g as f(|x + 1|)
noncomputable def g (x : ℝ) : ℝ := f (|x + 1|)

-- State the theorem
theorem g_properties :
  (∀ x, x ∈ Set.Icc (-4) 2 → g x = f (|x + 1|)) ∧
  (∀ x, x ∈ Set.Icc (-4) (-1) → g x = f (-(x + 1))) ∧
  (∀ x, x ∈ Set.Icc (-1) 2 → g x = f (x + 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l76_7645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_half_l76_7678

def terminal_side_point (α : Real) : ℝ × ℝ := (1, -2)

theorem cos_alpha_plus_pi_half (α : Real) : 
  terminal_side_point α = (1, -2) → Real.cos (α + Real.pi/2) = 2*Real.sqrt 5/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_half_l76_7678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radiator_capacity_is_six_l76_7686

/-- Represents the capacity of a car radiator in liters. -/
def radiator_capacity : ℝ → Prop := sorry

/-- The initial concentration of antifreeze in the radiator. -/
def initial_concentration : ℝ := 0.40

/-- The final concentration of antifreeze in the radiator after replacement. -/
def final_concentration : ℝ := 0.50

/-- The amount of liquid replaced in liters. -/
def replacement_amount : ℝ := 1

/-- Theorem stating that the radiator capacity is 6 liters given the conditions. -/
theorem radiator_capacity_is_six :
  ∀ C : ℝ,
  radiator_capacity C →
  (initial_concentration * C - initial_concentration * replacement_amount + replacement_amount) 
    = final_concentration * C →
  C = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radiator_capacity_is_six_l76_7686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_integers_negative15_to_6_l76_7623

theorem sum_integers_negative15_to_6 : 
  (Finset.range 22).sum (fun i ↦ (i : Int) - 15) = -99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_integers_negative15_to_6_l76_7623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_197_l76_7626

/-- An arithmetic sequence {a_n} with given first three terms -/
def arithmetic_sequence (a : ℝ) : ℕ → ℝ
| 0 => a - 1  -- Added case for 0
| 1 => a - 1
| 2 => a + 1
| 3 => 2 * a + 3
| n + 4 => arithmetic_sequence a (n + 3) + (arithmetic_sequence a 2 - arithmetic_sequence a 1)

/-- Theorem stating that the 100th term of the sequence is 197 -/
theorem hundredth_term_is_197 (a : ℝ) : arithmetic_sequence a 100 = 197 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_197_l76_7626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_theorem_l76_7660

/-- Curve C in rectangular coordinates -/
def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

/-- Line l in general form -/
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

/-- Point P -/
def point_P : ℝ × ℝ := (-4, -2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Geometric sequence property -/
def is_geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c

theorem curve_line_intersection_theorem (a : ℝ) (M N : ℝ × ℝ) :
  curve_C a M.1 M.2 ∧ curve_C a N.1 N.2 ∧
  line_l M.1 M.2 ∧ line_l N.1 N.2 ∧
  is_geometric_sequence (distance point_P M) (distance M N) (distance point_P N) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_theorem_l76_7660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_possible_l76_7697

/-- Represents a circular arrangement of 20 integers -/
def CircularArrangement := Fin 20 → ℤ

/-- The operation that replaces three consecutive numbers (x, y, z) with (x+y, -y, z+y) -/
def transform (arr : CircularArrangement) (i : Fin 20) : CircularArrangement :=
  fun j => if j = i then arr i + arr (i + 1)
           else if j = i + 1 then -arr (i + 1)
           else if j = i + 2 then arr (i + 2) + arr (i + 1)
           else arr j

/-- The initial arrangement -/
def initial : CircularArrangement :=
  fun i => if i.val < 10 then (i.val + 1 : ℤ) else -((i.val - 9) : ℤ)

/-- The target arrangement -/
def target : CircularArrangement :=
  fun i => if i.val < 10 then (10 - i.val : ℤ) else -(i.val - 9 : ℤ)

/-- Auxiliary function to apply a sequence of transformations -/
def applyTransforms (arr : CircularArrangement) : List (Fin 20) → CircularArrangement
  | [] => arr
  | i :: is => applyTransforms (transform arr i) is

/-- Theorem stating that it's possible to transform the initial arrangement into the target arrangement -/
theorem transform_possible : ∃ (l : List (Fin 20)), applyTransforms initial l = target := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_possible_l76_7697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_one_l76_7633

/-- The volume function of a rectangular box formed from an 18m wire,
    where one side of the base is twice as long as the other. -/
noncomputable def volume (x : ℝ) : ℝ := x * (2 * x) * ((18 - 6 * x) / 4)

/-- The derivative of the volume function -/
noncomputable def volume_derivative (x : ℝ) : ℝ := 
  (18 * x - 18 * x^2) / 2

theorem volume_maximized_at_one :
  ∃ (x : ℝ), x > 0 ∧ x < 3/2 ∧
  (∀ y, y > 0 → y < 3/2 → volume y ≤ volume x) ∧
  x = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_one_l76_7633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_equation_correct_l76_7621

/-- Represents the price reduction in yuan -/
def x : Real := sorry

/-- The original cost per box in yuan -/
def original_cost : ℝ := 80

/-- The original selling price per box in yuan -/
def original_price : ℝ := 150

/-- The initial daily sales in boxes -/
def initial_sales : ℝ := 160

/-- The increase in sales for every 1 yuan price reduction -/
def sales_increase_rate : ℝ := 8

/-- The target daily profit in yuan -/
def target_profit : ℝ := 16000

/-- The new selling price after reduction -/
def new_price (x : ℝ) : ℝ := original_price - x

/-- The new daily sales volume after price reduction -/
def new_sales (x : ℝ) : ℝ := initial_sales + sales_increase_rate * x

/-- The profit per box after price reduction -/
def profit_per_box (x : ℝ) : ℝ := new_price x - original_cost

/-- Theorem stating that the equation correctly represents the relationship
    between price reduction and daily profit goal -/
theorem profit_equation_correct (x : ℝ) :
  profit_per_box x * new_sales x = target_profit ↔
  (150 - 80 - x) * (160 + 8 * x) = 16000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_equation_correct_l76_7621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chopped_cube_height_is_correct_l76_7615

/-- The height of a unit cube with a corner chopped off, when placed on the freshly-cut face --/
noncomputable def choppedCubeHeight : ℝ := 2 * Real.sqrt 3 / 3

/-- Definition of a cube --/
structure Cube where
  sideLength : ℝ

/-- Definition of a chopped cube --/
structure ChoppedCube where
  originalCube : Cube
  cutPlaneDistance : ℝ

/-- Definition of a unit cube --/
def Cube.unit : Cube :=
  { sideLength := 1 }

/-- Operation to chop off a corner of a cube --/
def Cube.chopCorner (cube : Cube) : ChoppedCube :=
  { originalCube := cube
    cutPlaneDistance := 1 }

/-- The height of a chopped cube when placed on its freshly-cut face --/
noncomputable def ChoppedCube.heightOnFreshlyCutFace (choppedCube : ChoppedCube) : ℝ :=
  sorry -- This would be calculated based on the geometry of the chopped cube

/-- Theorem stating the height of a unit cube with a corner chopped off --/
theorem chopped_cube_height_is_correct :
  let cube := Cube.unit
  let choppedCube := cube.chopCorner
  choppedCube.heightOnFreshlyCutFace = choppedCubeHeight :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chopped_cube_height_is_correct_l76_7615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_primitive_roots_count_l76_7643

theorem primitive_roots_count (p : Nat) (h_prime : Nat.Prime p) (h_form : ∃ k, p = 4 * k + 1) 
  (h_ineq : p - 1 < 3 * Nat.totient (p - 1)) : 
  ∃ S : Finset Nat, 
    (∀ k ∈ S, IsPrimitiveRoot k p ∧ Nat.Coprime k (p - 1)) ∧ 
    S.card ≥ (3 * Nat.totient (p - 1) - (p - 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_primitive_roots_count_l76_7643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_minus_multicircle_equals_two_l76_7691

def circle' (a b : ℕ) : ℕ := (a + 2) * (b + 2) - 2

def multiCircle (list : List ℕ) : ℕ :=
  match list with
  | [] => 0
  | [x] => x
  | x :: xs => circle' x (multiCircle xs)

theorem product_minus_multicircle_equals_two :
  1 * 3 * 5 * 7 * 9 * 11 * 13 - multiCircle [1, 3, 5, 7, 9, 11, 13] = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_minus_multicircle_equals_two_l76_7691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l76_7656

/-- Represents the color of clothes a person is wearing -/
inductive Color
  | Red
  | Yellow
  deriving BEq, Repr

/-- Represents an arrangement of people -/
def Arrangement := List Color

/-- Checks if an arrangement is valid (no adjacent same colors) -/
def isValidArrangement (arr : Arrangement) : Bool :=
  arr.zipWith (·!=·) (arr.tail) |>.all id

/-- Generates all possible arrangements of 4 people with 2 red and 2 yellow -/
def allArrangements : List Arrangement :=
  [Color.Red, Color.Red, Color.Yellow, Color.Yellow].permutations

/-- Counts the number of valid arrangements -/
def countValidArrangements : Nat :=
  allArrangements.filter isValidArrangement |>.length

theorem valid_arrangements_count :
  countValidArrangements = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l76_7656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_defectiveness_l76_7608

-- Define the sample space
variable (Ω : Type)

-- Define the events as sets
variable (A B C : Set Ω)

-- Define defective as a predicate
variable (defective : Ω → Fin 3 → Prop)

-- Define the conditions
axiom h1 : A = {ω : Ω | ∀ i : Fin 3, ¬defective ω i}
axiom h2 : B = {ω : Ω | ∀ i : Fin 3, defective ω i}
axiom h3 : C = {ω : Ω | ∃ i : Fin 3, ¬defective ω i}

-- Theorem to prove
theorem product_defectiveness :
  (A ∩ B = ∅) ∧
  (B ∩ C = ∅) ∧
  (B ∪ C = Ω) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_defectiveness_l76_7608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grassy_pathway_area_l76_7675

/-- Given a rectangular plot and a grass path around it, calculate the area of the grassy pathway. -/
theorem grassy_pathway_area
  (plot_length : ℝ)
  (plot_width : ℝ)
  (path_width : ℝ)
  (h_plot_length : plot_length = 15)
  (h_plot_width : plot_width = 10)
  (h_path_width : path_width = 2) :
  (plot_length + 2 * path_width) * (plot_width + 2 * path_width) - plot_length * plot_width = 116 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grassy_pathway_area_l76_7675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_numbers_l76_7676

noncomputable def numbers : List ℝ := [5, 8, 11]

noncomputable def mean (list : List ℝ) : ℝ := (list.sum) / (list.length : ℝ)

noncomputable def variance (list : List ℝ) : ℝ :=
  let m := mean list
  (list.map (fun x => (x - m) ^ 2)).sum / (list.length : ℝ)

noncomputable def standardDeviation (list : List ℝ) : ℝ :=
  Real.sqrt (variance list)

theorem standard_deviation_of_numbers : standardDeviation numbers = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_numbers_l76_7676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_pyramid_angle_of_inclination_l76_7653

/-- Represents a cone with an inscribed triangular pyramid -/
structure ConePyramid where
  /-- Angle of the sector formed by the cone's lateral surface when unwrapped -/
  sector_angle : ℝ
  /-- Difference between consecutive angles in the arithmetic progression of base angles -/
  angle_diff : ℝ

/-- The cosine of the angle of inclination of the smallest lateral face to the base plane -/
noncomputable def angle_of_inclination (cp : ConePyramid) : ℝ :=
  1 / Real.sqrt 17

/-- Theorem stating the result for the specific cone and pyramid configuration -/
theorem cone_pyramid_angle_of_inclination :
  ∀ (cp : ConePyramid),
  cp.sector_angle = 120 ∧ cp.angle_diff = 15 →
  angle_of_inclination cp = 1 / Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_pyramid_angle_of_inclination_l76_7653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_to_one_control_l76_7611

/-- Represents the state of a switch or light -/
inductive State
| On
| Off

/-- Represents a system of switches and lights -/
structure SwitchLightSystem (n : ℕ) where
  switches : (Fin n → State) → (Fin n → State)
  lights : (Fin n → State) → (Fin n → State)
  all_combinations : ∀ (s : Fin n → State), ∃ (l : Fin n → State), switches s = s ∧ lights l = l
  unique_dependence : ∀ (s₁ s₂ : Fin n → State), switches s₁ = switches s₂ → lights s₁ = lights s₂
  one_switch_one_light : ∀ (s₁ s₂ : Fin n → State), 
    (∃! (i : Fin n), switches s₁ i ≠ switches s₂ i) → 
    (∃! (j : Fin n), lights s₁ j ≠ lights s₂ j)

/-- Main theorem: Each light is controlled by exactly one switch -/
theorem one_to_one_control (n : ℕ) (sys : SwitchLightSystem n) : 
  ∀ (i : Fin n), ∃! (j : Fin n), ∀ (s₁ s₂ : Fin n → State), 
    sys.switches s₁ j ≠ sys.switches s₂ j → sys.lights s₁ i ≠ sys.lights s₂ i :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_to_one_control_l76_7611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_4_equals_10_point_5_l76_7654

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := 3 - 4 / x

-- Define the function h using f_inv
noncomputable def h (x : ℝ) : ℝ := 1 / (f_inv x) + 10

-- Theorem statement
theorem h_of_4_equals_10_point_5 : h 4 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_4_equals_10_point_5_l76_7654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothes_fraction_proof_l76_7618

noncomputable def salary : ℝ := 140000
noncomputable def food_fraction : ℝ := 1/5
noncomputable def rent_fraction : ℝ := 1/10
noncomputable def remaining : ℝ := 14000

theorem clothes_fraction_proof :
  let food_expense := food_fraction * salary
  let rent_expense := rent_fraction * salary
  let total_expense := food_expense + rent_expense + remaining
  let clothes_expense := salary - total_expense
  clothes_expense / salary = 3/5 := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothes_fraction_proof_l76_7618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_fruit_sufficient_l76_7609

/-- Represents the content of a bin -/
inductive BinContent
  | Apples
  | Oranges
  | Mixed

/-- Represents a label on a bin -/
inductive BinLabel
  | Apples
  | Oranges
  | Mixed

/-- Represents a bin with its content and label -/
structure Bin where
  content : BinContent
  label : BinLabel

/-- The configuration of three bins -/
def BinConfiguration := (Bin × Bin × Bin)

/-- Predicate to check if all labels are incorrect -/
def allLabelsIncorrect (config : BinConfiguration) : Prop :=
  let (bin1, bin2, bin3) := config
  (bin1.content ≠ BinContent.Apples ∨ bin1.label ≠ BinLabel.Apples) ∧
  (bin1.content ≠ BinContent.Oranges ∨ bin1.label ≠ BinLabel.Oranges) ∧
  (bin1.content ≠ BinContent.Mixed ∨ bin1.label ≠ BinLabel.Mixed) ∧
  (bin2.content ≠ BinContent.Apples ∨ bin2.label ≠ BinLabel.Apples) ∧
  (bin2.content ≠ BinContent.Oranges ∨ bin2.label ≠ BinLabel.Oranges) ∧
  (bin2.content ≠ BinContent.Mixed ∨ bin2.label ≠ BinLabel.Mixed) ∧
  (bin3.content ≠ BinContent.Apples ∨ bin3.label ≠ BinLabel.Apples) ∧
  (bin3.content ≠ BinContent.Oranges ∨ bin3.label ≠ BinLabel.Oranges) ∧
  (bin3.content ≠ BinContent.Mixed ∨ bin3.label ≠ BinLabel.Mixed)

/-- Predicate to check if the configuration contains one of each content type -/
def hasOneOfEachContent (config : BinConfiguration) : Prop :=
  let (bin1, bin2, bin3) := config
  (bin1.content = BinContent.Apples ∧ bin2.content = BinContent.Oranges ∧ bin3.content = BinContent.Mixed) ∨
  (bin1.content = BinContent.Apples ∧ bin2.content = BinContent.Mixed ∧ bin3.content = BinContent.Oranges) ∨
  (bin1.content = BinContent.Oranges ∧ bin2.content = BinContent.Apples ∧ bin3.content = BinContent.Mixed) ∨
  (bin1.content = BinContent.Oranges ∧ bin2.content = BinContent.Mixed ∧ bin3.content = BinContent.Apples) ∨
  (bin1.content = BinContent.Mixed ∧ bin2.content = BinContent.Apples ∧ bin3.content = BinContent.Oranges) ∨
  (bin1.content = BinContent.Mixed ∧ bin2.content = BinContent.Oranges ∧ bin3.content = BinContent.Apples)

/-- Theorem stating that examining one piece of fruit is sufficient to determine all bin labels -/
theorem one_fruit_sufficient (config : BinConfiguration) 
  (h1 : allLabelsIncorrect config) 
  (h2 : hasOneOfEachContent config) : 
  ∃ (n : Nat), n = 1 ∧ n = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_fruit_sufficient_l76_7609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_doubles_in_two_years_l76_7652

/-- Annual production function -/
noncomputable def production (a : ℝ) (p : ℝ) (x : ℕ) : ℝ := a * (1 + p / 100) ^ x

/-- Condition for doubling production in two years -/
def double_in_two_years (a : ℝ) (p : ℝ) : Prop :=
  production a p 2 = 2 * a

theorem production_doubles_in_two_years (a : ℝ) (h : a > 0) :
  ∃ p : ℝ, double_in_two_years a p ∧ p = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_doubles_in_two_years_l76_7652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l76_7619

noncomputable section

-- Define the circumferences of the two circles
def circumference1 : ℝ := 528
def circumference2 : ℝ := 704

-- Define a function to calculate the radius from the circumference
noncomputable def radius_from_circumference (c : ℝ) : ℝ := c / (2 * Real.pi)

-- Define a function to calculate the area of a circle given its radius
noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

-- Define the radii of the two circles
noncomputable def radius1 : ℝ := radius_from_circumference circumference1
noncomputable def radius2 : ℝ := radius_from_circumference circumference2

-- Define the areas of the two circles
noncomputable def area1 : ℝ := area_of_circle radius1
noncomputable def area2 : ℝ := area_of_circle radius2

-- State the theorem
theorem area_difference_approx : 
  ‖area2 - area1 - 54197.5‖ < 0.1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_approx_l76_7619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l76_7613

/-- Conversion from spherical coordinates to rectangular coordinates -/
noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

/-- Theorem: The given spherical coordinates convert to the specified rectangular coordinates -/
theorem spherical_to_rectangular_conversion :
  let (x, y, z) := spherical_to_rectangular (-5) ((7 * Real.pi) / 4) (Real.pi / 3)
  x = -(5 * Real.sqrt 6) / 4 ∧
  y = -(5 * Real.sqrt 6) / 4 ∧
  z = -5 / 2 := by
  sorry

-- Remove the #eval statement as it may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l76_7613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_third_component_l76_7659

/-- Given two vectors a and b in ℝ³, prove that if they are perpendicular
    and have specific components, then the third component of b must be 14. -/
theorem perpendicular_vectors_third_component
  (a b : Fin 3 → ℝ)
  (h1 : a 0 = 2 ∧ a 1 = -3 ∧ a 2 = 1)
  (h2 : b 0 = -4 ∧ b 1 = 2)
  (h3 : a 0 * b 0 + a 1 * b 1 + a 2 * b 2 = 0) :
  b 2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_third_component_l76_7659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unchanged_100th_position_l76_7679

def swap_process (s : List ℕ) : List ℕ := sorry

theorem unchanged_100th_position (s : List ℕ) :
  (∀ n ∈ s, 1951 ≤ n ∧ n ≤ 1982) →
  s.length = 32 →
  let s' := swap_process s
  s.get? 99 = s'.get? 99 →
  s.get? 99 = some 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unchanged_100th_position_l76_7679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_after_discount_l76_7603

noncomputable section

-- Define the cost of 1 kg of flour
def flour_cost : ℝ := 21

-- Define the relationship between mangos and rice
def mango_rice_ratio : ℝ := 10 / 24

-- Define the relationship between flour and rice
def flour_rice_ratio : ℝ := 3 / 1

-- Define the discount rate
def discount_rate : ℝ := 0.1

-- Define the quantities of each item
def mango_quantity : ℝ := 4
def rice_quantity : ℝ := 3
def flour_quantity : ℝ := 5

-- Theorem statement
theorem total_cost_after_discount : 
  let rice_cost := flour_cost * flour_rice_ratio
  let mango_cost := rice_cost / mango_rice_ratio
  let total_cost := mango_cost * mango_quantity + rice_cost * rice_quantity + flour_cost * flour_quantity
  let discounted_cost := total_cost * (1 - discount_rate)
  discounted_cost = 808.92 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_after_discount_l76_7603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l76_7627

-- Define the function f(x) = ln(x/2) - 1/x
noncomputable def f (x : ℝ) : ℝ := Real.log (x / 2) - 1 / x

-- State the theorem
theorem zero_in_interval :
  (∀ x > 0, ContinuousAt (f) x) →  -- f is continuous for x > 0
  (∀ x > 0, ∀ y > x, f x < f y) →  -- f is strictly increasing for x > 0
  f 2 < 0 →  -- f(2) < 0
  f 3 > 0 →  -- f(3) > 0
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=  -- There exists c in (2, 3) such that f(c) = 0
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l76_7627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equals_two_thirds_pi_l76_7662

/-- The function f(x) = 2cos²(x) - 2√3sin(x)cos(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

/-- Theorem: The sum of the two distinct real roots of f(x) = -1/3 in [0, π/2] is 2π/3 -/
theorem sum_of_roots_equals_two_thirds_pi :
  ∃ (α β : ℝ), 0 ≤ α ∧ α < β ∧ β ≤ Real.pi / 2 ∧
  f α = -1/3 ∧ f β = -1/3 ∧
  α + β = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equals_two_thirds_pi_l76_7662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_sum_squared_bound_l76_7612

-- Define the semicircle
structure Semicircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the triangle
structure Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)

def is_inscribed (t : Triangle) (s : Semicircle) : Prop :=
  -- D and E are on the diameter
  t.D.1 = s.center.1 - s.radius ∧ t.D.2 = s.center.2 ∧
  t.E.1 = s.center.1 + s.radius ∧ t.E.2 = s.center.2 ∧
  -- F is on the semicircle
  (t.F.1 - s.center.1)^2 + (t.F.2 - s.center.2)^2 = s.radius^2 ∧
  t.F.2 ≥ s.center.2

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem inscribed_triangle_sum_squared_bound
  (s : Semicircle) (t : Triangle) (h : is_inscribed t s) :
  (distance t.D t.F + distance t.E t.F)^2 ≤ 32 * s.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_sum_squared_bound_l76_7612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_placement_l76_7671

def number_of_placements (n m : ℕ) : ℕ :=
  m^n

theorem letter_placement (n m : ℕ) (hn : n = 3) (hm : m = 4) :
  (number_of_placements n m) = m^n := by
  rfl

#check letter_placement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_placement_l76_7671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_population_calculation_l76_7692

/-- The total number of students in the school -/
def total_students : ℕ := sorry

/-- The number of boys in the school -/
def num_boys : ℕ := sorry

/-- The number of students representing a certain percent of boys -/
def num_representing : ℕ := 98

/-- The percent of the school population that is boys -/
def percent_boys : ℝ := 0.5

/-- The percent of boys that the 98 students represent -/
def percent_representing : ℝ := sorry

theorem school_population_calculation :
  total_students = num_representing / (percent_representing * percent_boys) :=
by
  sorry

#check school_population_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_population_calculation_l76_7692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_florida_movers_per_hour_l76_7658

/-- The number of people moving to Florida in a week -/
def people_moving : ℕ := 5000

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The average number of people moving to Florida per hour -/
def average_per_hour : ℚ :=
  people_moving / (days_per_week * hours_per_day)

/-- Rounds a rational number to two decimal places -/
def round_to_two_decimals (q : ℚ) : ℚ :=
  (q * 100).floor / 100

theorem florida_movers_per_hour :
  round_to_two_decimals average_per_hour = 2976 / 100 := by
  sorry

#eval round_to_two_decimals average_per_hour

end NUMINAMATH_CALUDE_ERRORFEEDBACK_florida_movers_per_hour_l76_7658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_consecutive_useful_numbers_l76_7648

def is_useful (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  (∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0) ∧
  (∀ d₁ d₂ : ℕ, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ ≠ d₂ → d₁ ≠ d₂) ∧
  (n.digits 10).prod % (n.digits 10).sum = 0

theorem exists_consecutive_useful_numbers :
  ∃ n : ℕ, is_useful n ∧ is_useful (n + 1) :=
by
  -- Proof goes here
  sorry

#check exists_consecutive_useful_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_consecutive_useful_numbers_l76_7648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_and_converse_of_P_l76_7600

-- Define Triangle as a structure
structure Triangle where
  -- You might want to add more fields to define a triangle
  area : ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the main proposition
def P : Prop := ∀ t1 t2 : Triangle, t1.area = t2.area → congruent t1 t2

theorem negation_and_converse_of_P :
  (¬P ↔ ∃ t1 t2 : Triangle, t1.area = t2.area ∧ ¬(congruent t1 t2)) ∧
  (∀ t1 t2 : Triangle, ¬(t1.area = t2.area) → ¬(congruent t1 t2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_and_converse_of_P_l76_7600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l76_7693

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a * Real.cos B - b * Real.cos A = c →
  C = Real.pi / 5 →
  B = 3 * Real.pi / 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l76_7693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l76_7622

/-- Calculates the speed of a train given its length and time to cross a stationary point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time * 3.6

/-- Theorem: A train of length 250 meters that crosses a stationary point in 3 seconds has a speed of approximately 300 km/h -/
theorem train_speed_calculation :
  let length : ℝ := 250
  let time : ℝ := 3
  let speed := train_speed length time
  ∃ ε > 0, |speed - 300| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l76_7622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l76_7667

/-- Given a hyperbola x²/a² - y²/b² = 1 and a function y = x², 
    if the tangent line to the function at (1,1) is parallel to 
    one of the hyperbola's asymptotes, then the eccentricity of 
    the hyperbola is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (b / a = 2) →
  Real.sqrt 5 = Real.sqrt (1 + (b/a)^2) := by
  intros ha hb hyperbola tangent_parallel
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l76_7667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_card_value_change_l76_7649

noncomputable def initial_value : ℝ := 100

noncomputable def year1_change : ℝ := -0.15
noncomputable def year2_change : ℝ := 0.10
noncomputable def year3_change : ℝ := -0.20
noncomputable def year4_change : ℝ := -0.25

noncomputable def final_value : ℝ := initial_value * (1 + year1_change) * (1 + year2_change) * (1 + year3_change) * (1 + year4_change)

noncomputable def total_percent_change : ℝ := (final_value - initial_value) / initial_value * 100

theorem baseball_card_value_change :
  abs (total_percent_change + 43.9) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_card_value_change_l76_7649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_determinant_zero_l76_7657

/-- Three lines intersecting at one point implies determinant is zero -/
theorem lines_intersect_determinant_zero (a : ℝ) :
  (∃ x y : ℝ, a * x + y + 3 = 0 ∧ x + y + 2 = 0 ∧ 2 * x - y + 1 = 0) →
  Matrix.det !![a, 1, 3; 1, 1, 2; 2, -1, 1] = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_determinant_zero_l76_7657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_points_even_l76_7624

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of six points in general position -/
def SixPoints := Fin 6 → Point3D

/-- Predicate to check if six points are in general position -/
def InGeneralPosition (points : SixPoints) : Prop := sorry

/-- A red point is an intersection between a segment and a tetrahedron surface -/
def RedPoint (points : SixPoints) : Type := Unit

/-- The set of all red points for given six points -/
def AllRedPoints (points : SixPoints) : Finset (RedPoint points) :=
  sorry

/-- The main theorem: the number of red points is even -/
theorem red_points_even (points : SixPoints) (h : InGeneralPosition points) :
  Even (Finset.card (AllRedPoints points)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_points_even_l76_7624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l76_7617

/-- An arithmetic sequence is a sequence where the difference between 
    consecutive terms is constant. -/
def is_arithmetic (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_solution :
  ∀ y : ℚ,
  let a : ℕ → ℚ := fun n =>
    match n with
    | 0 => -1/3
    | 1 => y + 2
    | 2 => 4*y
    | _ => 0  -- placeholder for other terms
  is_arithmetic a → y = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l76_7617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_selection_probability_l76_7644

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- The probability of an individual being selected in a sample -/
noncomputable def selectionProbability (N : ℕ) (n : ℕ) (method : SamplingMethod) : ℝ :=
  match method with
  | SamplingMethod.SimpleRandom => (n : ℝ) / N
  | SamplingMethod.Systematic => (n : ℝ) / N
  | SamplingMethod.Stratified => (n : ℝ) / N

theorem equal_selection_probability (N : ℕ) (n : ℕ) (h : 0 < N) (h' : n ≤ N) :
  selectionProbability N n SamplingMethod.SimpleRandom =
  selectionProbability N n SamplingMethod.Systematic ∧
  selectionProbability N n SamplingMethod.SimpleRandom =
  selectionProbability N n SamplingMethod.Stratified :=
by
  sorry

#check equal_selection_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_selection_probability_l76_7644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l76_7665

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (b + c) * (b + c) - (a^2 + b * c) = 0 →
  (∃ (k : ℝ), a = k * (Real.sin A) ∧ b = k * (Real.sin B) ∧ c = k * (Real.sin C)) →
  (A = 2 * π / 3 ∧
   (a = 3 → ∃ (p : ℝ), p ≤ 3 + 2 * Real.sqrt 3 ∧
                       ∀ (q : ℝ), q = a + b + c → q ≤ p)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l76_7665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_is_18_l76_7601

/-- Represents the fishing scenario with Daisy and Tom --/
structure FishingScenario where
  initial_distance : ℚ
  water_intake_rate : ℚ
  max_water_capacity : ℚ
  first_half_speed : ℚ
  second_half_speed : ℚ

/-- Calculates the minimum bailing rate required to reach shore safely --/
def min_bailing_rate (scenario : FishingScenario) : ℚ :=
  let first_half_time := scenario.initial_distance / (2 * scenario.first_half_speed)
  let second_half_time := scenario.initial_distance / (2 * scenario.second_half_speed)
  let total_time := first_half_time + second_half_time
  let total_water_intake := scenario.water_intake_rate * total_time
  (total_water_intake - scenario.max_water_capacity) / total_time

/-- The main theorem stating the minimum bailing rate for the given scenario --/
theorem min_bailing_rate_is_18 (scenario : FishingScenario) 
  (h1 : scenario.initial_distance = 3)
  (h2 : scenario.water_intake_rate = 20)
  (h3 : scenario.max_water_capacity = 120)
  (h4 : scenario.first_half_speed = 6)
  (h5 : scenario.second_half_speed = 3) :
  ⌈min_bailing_rate scenario⌉ = 18 := by
  sorry

def scenario : FishingScenario := {
  initial_distance := 3,
  water_intake_rate := 20,
  max_water_capacity := 120,
  first_half_speed := 6,
  second_half_speed := 3
}

#eval ⌈min_bailing_rate scenario⌉

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_is_18_l76_7601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inference_reasonability_l76_7640

/-- An inference is a logical reasoning process --/
structure Inference where
  premise : Prop
  conclusion : Prop

/-- A reasonable inference is one that follows logical principles --/
def reasonable (i : Inference) : Prop :=
  -- This definition is left abstract as the problem doesn't provide specific criteria
  sorry

/-- Inference A: Properties of a sphere inferred from properties of a circle --/
def inference_A : Inference where
  premise := ∃ p : Prop, p  -- Represents "Properties of a circle are known"
  conclusion := ∃ q : Prop, q  -- Represents "Properties of a sphere are inferred"

/-- Inference B: Sum of internal angles of all triangles inferred from specific types --/
def inference_B : Inference where
  premise := ∀ t : Nat, t ≤ 3 → (∃ angle : Nat, angle = 180)  -- Represents specific triangle types
  conclusion := ∀ t : Nat, ∃ angle : Nat, angle = 180  -- Represents all triangles

/-- Inference C: All students' scores inferred from one student's score --/
def inference_C : Inference where
  premise := ∃ s : Nat, s = 100  -- Represents "Zhang Jun scored 100 points"
  conclusion := ∀ s : Nat, s = 100  -- Represents "All students scored 100 points"

/-- Inference D: All reptiles' breathing method inferred from specific reptiles --/
def inference_D : Inference where
  premise := ∃ r : Nat, r ≤ 3 → (∃ b : Prop, b)  -- Represents specific reptiles breathe with lungs
  conclusion := ∀ r : Nat, ∃ b : Prop, b  -- Represents all reptiles breathe with lungs

/-- Theorem stating that inference C is not reasonable while others are --/
theorem inference_reasonability :
  reasonable inference_A ∧ 
  reasonable inference_B ∧ 
  ¬reasonable inference_C ∧ 
  reasonable inference_D :=
by
  sorry  -- Proof is omitted as per the problem statement


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inference_reasonability_l76_7640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l76_7605

open Real

-- Define the function
noncomputable def f (x : ℝ) := x + 4 / (x - 1)

-- State the theorem
theorem f_minimum_value :
  ∀ x : ℝ, x > 1 → f x ≥ 5 ∧ ∃ x₀ : ℝ, x₀ > 1 ∧ f x₀ = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l76_7605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l76_7674

noncomputable section

variable (f : ℝ → ℝ)

theorem f_derivative_at_2 (h : ∀ x, f x = x^2 + 2*x*(deriv f 2) - Real.log x) : 
  deriv f 2 = -7/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l76_7674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rooms_with_two_windows_l76_7694

structure Room where
  windows : ℕ

theorem rooms_with_two_windows
  (total_windows : ℕ)
  (rooms_with_four : ℕ)
  (rooms_with_three : ℕ)
  (h1 : total_windows = 122)
  (h2 : rooms_with_four = 5)
  (h3 : rooms_with_three = 8)
  (h4 : ∀ room : Room, 2 ≤ room.windows ∧ room.windows ≤ 4) :
  ∃ rooms_with_two : ℕ,
    rooms_with_two * 2 + rooms_with_three * 3 + rooms_with_four * 4 = total_windows ∧
    rooms_with_two = 39 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rooms_with_two_windows_l76_7694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_blue_percentage_l76_7690

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue : ℝ
  red : ℝ
  yellow : ℝ

/-- The maroon paint composition -/
def maroon : PaintMixture := { blue := 0.5, red := 0.5, yellow := 0 }

/-- The green paint composition -/
def green (b : ℝ) : PaintMixture := { blue := b, red := 0, yellow := 0.7 }

/-- The brown paint composition -/
def brown : PaintMixture := { blue := 0.4, red := 0.25, yellow := 0.35 }

/-- The total weight of the brown paint in grams -/
def brownWeight : ℝ := 10

/-- Theorem stating that the percentage of blue pigment in the green paint is 30% -/
theorem green_blue_percentage :
  ∃ b : ℝ, green b = green 0.3 ∧ 
  maroon.blue * 5 + b * (brownWeight - 5) = brown.blue * brownWeight := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_blue_percentage_l76_7690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l76_7602

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - 2 * x + a)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l76_7602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_overall_income_l76_7630

/-- Calculate the overall total income after one year given the following conditions:
  * Initial deposit in savings account: 4,500 Rs
  * Savings deposit is 28% of monthly income
  * Initial deposit in fixed deposit account: 3,000 Rs
  * Fixed deposit annual interest rate: 6% compounded annually
  * Savings account monthly interest rate: 2%
  * Time period: 1 year
-/
theorem calculate_overall_income (savings_deposit : ℝ) (savings_percentage : ℝ)
    (fixed_deposit : ℝ) (fixed_rate : ℝ) (savings_rate : ℝ) (time : ℕ) :
  savings_deposit = 4500 →
  savings_percentage = 0.28 →
  fixed_deposit = 3000 →
  fixed_rate = 0.06 →
  savings_rate = 0.02 →
  time = 1 →
  ∃ total_income : ℝ,
    (total_income ≥ 194117.15 ∧ total_income ≤ 194117.17) ∧
    total_income = (savings_deposit / savings_percentage * 12) +
                   (savings_deposit * (savings_rate * 12)) +
                   (fixed_deposit * fixed_rate) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_overall_income_l76_7630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_remainder_l76_7607

/-- Calculates the number of towers that can be built with cubes of sizes 1 to n -/
def towerCount : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => 4 * towerCount (n + 1)

/-- The problem statement -/
theorem tower_remainder :
  towerCount 9 % 1000 = 536 := by
  -- Evaluate towerCount 9
  have h1 : towerCount 9 = 65536 := by native_decide
  -- Calculate the remainder
  have h2 : 65536 % 1000 = 536 := by native_decide
  -- Combine the results
  rw [h1, h2]

#eval towerCount 9 % 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_remainder_l76_7607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_is_correct_l76_7696

/-- A lattice point in an xy-coordinate system is any point (x, y) where both x and y are integers. -/
def is_lattice_point (x y : ℤ) : Prop := true

/-- The line y = mx + 3 passes through no lattice point with 0 < x ≤ n -/
def no_lattice_points (m : ℚ) (n : ℕ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ n → is_lattice_point x y → y ≠ (m * ↑x + 3).floor

/-- The maximum possible value of a -/
noncomputable def max_a (n : ℕ) : ℚ := (n + 1) / (3 * n + 1)

theorem max_a_is_correct (n : ℕ) :
  (∀ m : ℚ, 1/3 < m ∧ m < max_a n → no_lattice_points m n) ∧
  ∀ a : ℚ, a > max_a n →
    ∃ m : ℚ, 1/3 < m ∧ m < a ∧ ¬(no_lattice_points m n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_is_correct_l76_7696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_of_coefficients_l76_7661

/-- Given a binomial expansion (x² - 3/√x)ⁿ where the 5th term is constant, 
    prove that the sum of all coefficients is -32 -/
theorem binomial_expansion_sum_of_coefficients :
  ∀ n : ℕ,
  (∃ k : ℕ, k = 4 ∧ (n.choose k) * (-3)^k * (2*n - 5*k : ℤ) = 0) →
  (1 - 3)^n = -32 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_of_coefficients_l76_7661
