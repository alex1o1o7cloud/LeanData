import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1298_129814

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/4) * (t.a^2 + t.b^2 - t.c^2)

/-- Theorem: In triangle ABC with area S = (1/4)(a² + b² - c²), 
    angle C = 45° and cos B = √6/3 when b = 2 and c = √6 -/
theorem triangle_theorem (t : Triangle) 
    (h1 : area t = (1/4) * (t.a^2 + t.b^2 - t.c^2))
    (h2 : t.b = 2)
    (h3 : t.c = Real.sqrt 6) : 
    t.C = 45 * (π / 180) ∧ Real.cos t.B = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1298_129814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1298_129892

-- Define the sequence a_n
def a : ℕ → ℚ
| 0 => 3/2
| (n+1) => (n+2 : ℚ)

-- Define the sequence S_n
def S (n : ℕ) : ℚ := ((n+1)^2 + (n+1) + 1) / 2

-- Define the sequence b_n
def b : ℕ → ℚ
| 0 => 3
| (n+1) => 3 * b n

theorem problem_solution :
  (∀ n : ℕ, S n = ((n+1)^2 + (n+1) + 1) / 2) ∧
  (a 0 = 1 ∧ a 1 = 2) ∧
  (∀ n : ℕ, b n = a (2*n) + a (2*n + 1)) ∧
  (∀ n : ℕ, b (n+1) = 3 * b n) →
  (∀ n : ℕ, a n = if n = 0 then 3/2 else (n+1 : ℚ)) ∧
  (∀ n : ℕ, S (2*n + 1) = (3/2) * (3^(n+1) - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1298_129892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l1298_129861

noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_value (α : ℝ) (h : powerFunction α (Real.sqrt 3) = 1/3) : 
  powerFunction α (1/2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l1298_129861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_P_and_parallel_to_tangent_l1298_129813

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at M
noncomputable def m : ℝ := (deriv f) M.1

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

theorem line_passes_through_P_and_parallel_to_tangent : 
  line_equation P.1 P.2 ∧ 
  (∀ (x y : ℝ), line_equation x y → (y - P.2) = m * (x - P.1)) := by
  sorry

#check line_passes_through_P_and_parallel_to_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_P_and_parallel_to_tangent_l1298_129813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_to_equation_l1298_129818

theorem unique_solution_to_equation : 
  ∃! x : ℝ, x^2 + 6*x + 6*x * Real.sqrt (x + 4) = 24 ∧ x = (17 - Real.sqrt 241) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_to_equation_l1298_129818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1298_129821

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1298_129821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_cost_l1298_129831

def michael_money : ℕ := 50
def bouquet_cost : ℕ := 36
def balloons_cost : ℕ := 5
def additional_money_needed : ℕ := 11

theorem cake_cost : 
  michael_money + additional_money_needed - bouquet_cost - balloons_cost = 20 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_cost_l1298_129831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_length_is_125_optimal_length_maximizes_area_l1298_129839

/-- Represents a rectangular sheep pasture -/
structure SheepPasture where
  barn_length : ℝ
  fence_cost_per_foot : ℝ
  total_fence_cost : ℝ

/-- Calculates the area of the pasture given the length of the side perpendicular to the barn -/
noncomputable def pasture_area (p : SheepPasture) (y : ℝ) : ℝ :=
  y * (p.total_fence_cost / p.fence_cost_per_foot - 2 * y)

/-- Finds the length of the side parallel to the barn that maximizes the area -/
noncomputable def optimal_parallel_length (p : SheepPasture) : ℝ :=
  p.total_fence_cost / p.fence_cost_per_foot - 2 * (p.total_fence_cost / (2 * p.fence_cost_per_foot))

theorem optimal_length_is_125 (p : SheepPasture) 
  (h1 : p.barn_length = 500)
  (h2 : p.fence_cost_per_foot = 10)
  (h3 : p.total_fence_cost = 2500) :
  optimal_parallel_length p = 125 := by
  sorry

theorem optimal_length_maximizes_area (p : SheepPasture) (y : ℝ) :
  pasture_area p (p.total_fence_cost / (2 * p.fence_cost_per_foot)) ≥ pasture_area p y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_length_is_125_optimal_length_maximizes_area_l1298_129839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1298_129862

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the dimensions of the original rectangle -/
noncomputable def rectangleWidth : ℝ := 15 * Real.sqrt 2
noncomputable def rectangleHeight : ℝ := 14 * Real.sqrt 2

/-- The distance from Q to points E, F, and G -/
noncomputable def edgeLength : ℝ := Real.sqrt 1031 / 2

/-- Theorem stating the volume of the pyramid -/
theorem pyramid_volume : 
  ∀ (G H F Q : Point3D),
  G.x = 7 * Real.sqrt 2 ∧ G.y = 0 ∧ G.z = 0 →
  H.x = -7 * Real.sqrt 2 ∧ H.y = 0 ∧ H.z = 0 →
  F.x = 0 ∧ F.y = Real.sqrt 391 ∧ F.z = 0 →
  (Q.x - F.x)^2 + (Q.y - F.y)^2 + (Q.z - F.z)^2 = edgeLength^2 →
  (Q.x - G.x)^2 + (Q.y - G.y)^2 + (Q.z - G.z)^2 = edgeLength^2 →
  (Q.x - H.x)^2 + (Q.y - H.y)^2 + (Q.z - H.z)^2 = edgeLength^2 →
  (1/3) * (rectangleWidth * rectangleHeight) * Q.z = 735 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1298_129862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_proof_l1298_129802

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem investment_duration_proof :
  ∃ t : ℝ, (t ≥ 1.48 ∧ t ≤ 1.50) ∧
  (compound_interest 3000 0.04 2 t ≥ 3181.78 ∧ compound_interest 3000 0.04 2 t ≤ 3181.79) := by
  sorry

#check investment_duration_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_duration_proof_l1298_129802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_cross_product_equality_l1298_129855

open Real

noncomputable def a : ℝ × ℝ × ℝ := (1, 2, 1)
noncomputable def b : ℝ × ℝ × ℝ := (2, 0, -1)
noncomputable def v : ℝ × ℝ × ℝ := (3, 2, 0)

def cross (x y : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (x.2.1 * y.2.2 - x.2.2 * y.2.1,
   x.2.2 * y.1 - x.1 * y.2.2,
   x.1 * y.2.1 - x.2.1 * y.1)

theorem vector_cross_product_equality :
  (cross v a = cross b a) ∧ (cross v b = cross a b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_cross_product_equality_l1298_129855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_to_curve_l1298_129807

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 2 * y + 1 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop := ∃ t : ℝ, x = t ∧ y = 1/4 * t^2

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_line_to_curve :
  ∃ (x1 y1 x2 y2 : ℝ),
    line_l x1 y1 ∧ 
    curve_C x2 y2 ∧
    (∀ (x3 y3 x4 y4 : ℝ), line_l x3 y3 → curve_C x4 y4 →
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4) ∧
    distance x1 y1 x2 y2 = Real.sqrt 5 / 10 := by
  sorry

#check min_distance_line_to_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_to_curve_l1298_129807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosC_value_l1298_129875

theorem cosC_value (A B C : Real) 
  (h : A + B + C = Real.pi)
  (h1 : ∃ (x y : Real), x^2 - 10*x + 6 = 0 ∧ y^2 - 10*y + 6 = 0 ∧ x = Real.tan A ∧ y = Real.tan B) :
  Real.cos C = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosC_value_l1298_129875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l1298_129834

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 1)

theorem inverse_function_domain :
  ∀ y : ℝ, y ∈ Set.range f ↔ y ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l1298_129834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1298_129811

theorem triangle_problem (A B C : ℝ) (m n : ℝ × ℝ) (a b c s : ℝ) : 
  m = (Real.cos (C / 2), Real.sin (C / 2)) →
  n = (Real.cos (C / 2), -Real.sin (C / 2)) →
  Real.arccos ((m.1 * n.1 + m.2 * n.2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2))) = π / 3 →
  c = 7 / 2 →
  s = 3 * Real.sqrt 3 / 2 →
  C = π / 3 ∧ a + b = 11 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1298_129811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_different_points_num_second_quadrant_num_not_on_line_l1298_129871

-- Define the set M
def M : Finset Int := {-3, -2, -1, 0, 1, 2}

-- Define the point P
def P (a b : Int) : Prod Int Int := (a, b)

-- Theorem for the number of different points
theorem num_different_points : 
  Finset.card (Finset.product M M) = 36 := by
  sorry

-- Theorem for the number of points in the second quadrant
theorem num_second_quadrant : 
  Finset.card (Finset.filter (fun p => p.1 < 0 ∧ p.2 > 0) (Finset.product M M)) = 6 := by
  sorry

-- Theorem for the number of points not on the line y = x
theorem num_not_on_line : 
  Finset.card (Finset.filter (fun p => p.1 ≠ p.2) (Finset.product M M)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_different_points_num_second_quadrant_num_not_on_line_l1298_129871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_both_clubs_value_l1298_129882

def total_students : ℕ := 30
def chess_club_members : ℕ := 22
def drama_club_members : ℕ := 20

def probability_both_clubs : ℚ :=
  1 - (Nat.choose (chess_club_members + drama_club_members - total_students) 2 +
       Nat.choose (chess_club_members - (chess_club_members + drama_club_members - total_students)) 2) /
      Nat.choose total_students 2

theorem probability_both_clubs_value :
  probability_both_clubs = 362 / 435 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_both_clubs_value_l1298_129882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1298_129894

noncomputable def y (a b x : ℝ) : ℝ := a - b * Real.cos (2 * x + Real.pi / 6)

noncomputable def g (a b x : ℝ) : ℝ := 4 * a * Real.sin (b * x - Real.pi / 3)

theorem function_properties (a b : ℝ) (h₁ : b > 0) 
  (h₂ : ∀ x, y a b x ≤ 3) 
  (h₃ : ∃ x, y a b x = 3)
  (h₄ : ∀ x, y a b x ≥ -1) 
  (h₅ : ∃ x, y a b x = -1) :
  (a = 1 ∧ b = 2) ∧
  (∀ x, x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 6) → 
    g a b x ∈ Set.Icc (-2 * Real.sqrt 3) 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1298_129894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_A_l1298_129893

def A : Set (ℕ × ℕ) := {p | p.1 + p.2 = 10 ∧ p.1 > 0 ∧ p.2 > 0}

theorem cardinality_of_A : Finset.card (Finset.filter (fun p => p.1 + p.2 = 10) (Finset.range 10 ×ˢ Finset.range 10)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_A_l1298_129893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l1298_129854

noncomputable def f (x : ℝ) : ℝ := Real.log ((-x^2 + 4*x - 3) : ℝ) / Real.log (1/2 : ℝ)

theorem monotonic_increasing_interval_of_f :
  ∃ (a b : ℝ), a = 2 ∧ b = 3 ∧
  (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∧
  (∀ ε > 0, ∃ x y, a - ε < x ∧ x < a ∧ a < y ∧ y < a + ε ∧ f x ≥ f y) ∧
  (∀ ε > 0, ∃ x y, b - ε < x ∧ x < b ∧ b < y ∧ y < b + ε ∧ f x > f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l1298_129854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jumping_theorem_l1298_129845

/-- A regular 2n-gon inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  n_ge_two : n ≥ 2

/-- A jumping scheme for frogs on the vertices of a regular polygon -/
structure JumpingScheme (n : ℕ) where
  jump : Fin (2*n) → Fin (2*n)

/-- Predicate to check if a line segment goes through the center of the circle -/
def line_through_center (n : ℕ) (v1 v2 : Fin (2*n)) : Prop :=
  sorry

/-- Predicate to check if a jumping scheme is valid (no line through center) -/
def valid_jumping_scheme (n : ℕ) (scheme : JumpingScheme n) : Prop :=
  ∀ v1 v2 : Fin (2*n), v1 ≠ v2 →
    ¬(line_through_center n (scheme.jump v1) (scheme.jump v2))

theorem frog_jumping_theorem (n : ℕ) (polygon : RegularPolygon n) :
  (∃ (scheme : JumpingScheme n), valid_jumping_scheme n scheme) ↔ n % 4 = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jumping_theorem_l1298_129845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tile_probability_l1298_129897

-- Define the total number of tiles
def total_tiles : ℕ := 50

-- Define a function to check if a tile is blue
def is_blue (n : ℕ) : Bool := n % 5 = 2

-- Define the set of blue tiles
def blue_tiles : Finset ℕ := Finset.filter (fun n => is_blue n) (Finset.range total_tiles)

-- State the theorem
theorem blue_tile_probability :
  (Finset.card blue_tiles : ℚ) / total_tiles = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tile_probability_l1298_129897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1298_129850

noncomputable def point1 : ℝ × ℝ := (-3, 5)
noncomputable def point2 : ℝ × ℝ := (7, -10)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  distance point1 point2 = Real.sqrt 325 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1298_129850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetric_points_k_range_l1298_129859

/-- Given two distinct points M and N on the parabola y = x^2 that are symmetric about the line y = kx + 9/2, 
    the range of possible values for k is (-∞, -1/4) ∪ (1/4, +∞). -/
theorem parabola_symmetric_points_k_range (M N : ℝ × ℝ) (k : ℝ) :
  M ≠ N ∧
  (∀ (x : ℝ), (x, x^2) = M ∨ (x, x^2) = N) ∧
  (∃ (m : ℝ × ℝ), m = ((M.1 + N.1)/2, (M.2 + N.2)/2) ∧ m.2 = k * m.1 + 9/2) →
  k ∈ Set.Ioi (1/4) ∪ Set.Iio (-1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetric_points_k_range_l1298_129859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_exponents_l1298_129822

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The main theorem -/
theorem max_sum_of_exponents (x y : ℤ) :
  (∃ k : ℕ, (factorial 30 : ℚ) = k * ((36 : ℚ)^x * (25 : ℚ)^y)) →
  x + y ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_exponents_l1298_129822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1298_129840

-- Define the hyperbola
noncomputable def is_hyperbola (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the line
def is_line (x y : ℝ) : Prop :=
  y = 2 * x

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

-- Theorem statement
theorem hyperbola_eccentricity_range (a b : ℝ) :
  (∀ x y : ℝ, is_hyperbola a b x y → ¬is_line x y) →
  1 < eccentricity a b ∧ eccentricity a b ≤ Real.sqrt 5 := by
  sorry

#check hyperbola_eccentricity_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1298_129840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_dissipated_on_r3_l1298_129809

/-- Represents a resistor with a given resistance in Ohms -/
structure Resistor where
  resistance : ℝ

/-- Represents a series circuit of resistors -/
structure SeriesCircuit where
  resistors : List Resistor

/-- Represents a voltage source -/
structure VoltageSource where
  voltage : ℝ

/-- Calculates the total resistance of a series circuit -/
noncomputable def totalResistance (circuit : SeriesCircuit) : ℝ :=
  (circuit.resistors.map (·.resistance)).sum

/-- Calculates the power dissipated on a specific resistor in the circuit -/
noncomputable def powerDissipated (circuit : SeriesCircuit) (source : VoltageSource) (resistor : Resistor) : ℝ :=
  (source.voltage ^ 2 * resistor.resistance) / (totalResistance circuit) ^ 2

theorem power_dissipated_on_r3 (circuit : SeriesCircuit) (source : VoltageSource) :
  circuit.resistors = [
    Resistor.mk 1,
    Resistor.mk 2,
    Resistor.mk 3,
    Resistor.mk 4,
    Resistor.mk 5,
    Resistor.mk 6
  ] →
  source.voltage = 12 →
  powerDissipated circuit source (Resistor.mk 3) = 432 / 441 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_dissipated_on_r3_l1298_129809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_ellipse_find_m_l1298_129826

/-- The chord length of an ellipse intersected by a line -/
noncomputable def chord_length (m : ℝ) : ℝ :=
  let a := 4 -- coefficient of x^2 in ellipse equation
  let b := 1 -- coefficient of y^2 in ellipse equation
  let c := 1 -- constant term in ellipse equation
  Real.sqrt ((a + 1) * ((2 * m)^2 / (a + 1)^2 - 4 * (m^2 - c) / (a + 1)))

/-- The theorem stating the relationship between m and the chord length -/
theorem chord_length_ellipse (m : ℝ) :
  4 * m^2 = 5 ↔ chord_length m = 2 * Real.sqrt 2 / 5 :=
sorry

/-- The main theorem proving the value of m -/
theorem find_m :
  ∃ m : ℝ, (m = Real.sqrt 5 / 2 ∨ m = -Real.sqrt 5 / 2) ∧ chord_length m = 2 * Real.sqrt 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_ellipse_find_m_l1298_129826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangential_quadrilateral_l1298_129830

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
structure PointOnCircle (k : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - k.center.1)^2 + (point.2 - k.center.2)^2 = k.radius^2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem tangential_quadrilateral 
  (k : Circle) 
  (A B C D : PointOnCircle k) 
  (h_distinct : A.point ≠ B.point ∧ B.point ≠ C.point ∧ C.point ≠ D.point ∧ D.point ≠ A.point)
  (h_sum : distance A.point B.point + distance C.point D.point = 
           distance B.point C.point + distance A.point D.point) :
  ∃ (I : ℝ × ℝ), 
    distance I A.point = distance I B.point ∧
    distance I B.point = distance I C.point ∧
    distance I C.point = distance I D.point ∧
    distance I D.point = distance I A.point :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangential_quadrilateral_l1298_129830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_roots_of_10000_l1298_129853

theorem integer_roots_of_10000 : 
  (∃! (count : ℕ), ∀ (n : ℕ), n > 0 → (∃ (m : ℕ), m^n = 10000) ↔ n ∈ Finset.range (count + 1)) ∧ 
  (∃ (count : ℕ), ∀ (n : ℕ), n > 0 → (∃ (m : ℕ), m^n = 10000) ↔ n ∈ Finset.range (count + 1) ∧ count = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_roots_of_10000_l1298_129853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_from_vectors_l1298_129836

theorem equilateral_triangle_from_vectors
  (u v w : ℝ)
  (h_order : u ≤ v ∧ v ≤ w)
  (h_positive : 0 < u ∧ 0 < v ∧ 0 < w) :
  (∃ (A B C : ℝ × ℝ),
    let d := dist A B
    dist A B = d ∧
    dist B C = d ∧
    dist C A = d ∧
    dist (0, 0) A = u ∧
    dist (0, 0) B = v ∧
    dist (0, 0) C = w) ↔
  u + v ≥ w :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_from_vectors_l1298_129836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_inequality_l1298_129823

/-- Triangle inequality theorem for bisector distances -/
theorem triangle_bisector_inequality 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (l_a l_b l_c : ℝ) 
  (h_l_a : l_a > 0) 
  (h_l_b : l_b > 0) 
  (h_l_c : l_c > 0) 
  (h_A h_B h_C : ℝ)
  (h_def_h_A : h_A > 0)
  (h_def_h_B : h_B > 0)
  (h_def_h_C : h_C > 0)
  (h_bisector_a : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ h_A = x + y)
  (h_bisector_b : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ h_B = x + y)
  (h_bisector_c : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ h_C = x + y) :
  h_A * h_B * h_C ≥ (1/8) * (a+b) * (b+c) * (c+a) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_inequality_l1298_129823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_square_problem_l1298_129874

/-- The area of the region bounded by RT, RU, and the minor arc connecting T and U -/
noncomputable def shaded_area (R : ℝ) (s : ℝ) : ℝ :=
  (R^2 * Real.pi * 3/8) - (2 * (R^2 - (s^2/4))^(1/2))^2 / 2

theorem circle_square_problem (R s : ℝ) (h1 : R = 5) (h2 : s = 3) :
  shaded_area R s = (25 * Real.pi / 24) - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_square_problem_l1298_129874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1298_129865

theorem constant_term_expansion (a b x : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : x > 0) 
  (h4 : x^(2*a) = 1/x^b) : 
  ∃ c : ℕ → ℝ, (∀ n, c n = (Nat.choose 9 n) * (x^a)^(9-n) * (-2*x^b)^n) ∧ 
  (∃ k, c k = -672 ∧ ∀ n ≠ k, (x^a)^(9-n) * x^(b*n) = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1298_129865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1298_129884

/-- The function g(t) representing the difference between max and min of f(x) -/
noncomputable def g (t : ℝ) : ℝ := (8 * Real.sqrt (t^2 + 1) * (2 * t^2 + 5)) / (16 * t^2 + 25)

/-- Theorem stating the properties of g(t) and the inequality involving g(tan u_i) -/
theorem g_properties (t : ℝ) (u₁ u₂ u₃ : ℝ) : 
  let α := (2 * t - Real.sqrt (4 * t^2 + 1)) / 4
  let β := (2 * t + Real.sqrt (4 * t^2 + 1)) / 4
  let f (x : ℝ) := (2 * x - t) / (x^2 + 1)
  (∀ x ∈ Set.Icc α β, f x ≤ f β ∧ f x ≥ f α) →
  0 < u₁ ∧ u₁ < π/2 →
  0 < u₂ ∧ u₂ < π/2 →
  0 < u₃ ∧ u₃ < π/2 →
  Real.sin u₁ + Real.sin u₂ + Real.sin u₃ = 1 →
  g t = (8 * Real.sqrt (t^2 + 1) * (2 * t^2 + 5)) / (16 * t^2 + 25) ∧
  1 / g (Real.tan u₁) + 1 / g (Real.tan u₂) + 1 / g (Real.tan u₃) < 3/4 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1298_129884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_minus_2cos_squared_alpha_l1298_129841

theorem sin_2alpha_minus_2cos_squared_alpha (α : ℝ) :
  Real.tan (α - π / 4) = 2 → Real.sin (2 * α) - 2 * (Real.cos α)^2 = -4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_minus_2cos_squared_alpha_l1298_129841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_eggs_taken_l1298_129837

theorem amy_eggs_taken (start end_ taken : ℕ) : 
  start = 96 → end_ = 93 → taken = start - end_ → taken = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amy_eggs_taken_l1298_129837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_16cm_2rad_l1298_129881

/-- The area of a circular sector with given perimeter and central angle -/
noncomputable def sectorArea (perimeter : ℝ) (centralAngle : ℝ) : ℝ :=
  let radius := perimeter / (2 + centralAngle)
  (centralAngle * radius^2) / 2

/-- Theorem: The area of a circular sector with perimeter 16 cm and central angle 2 rad is 16 cm² -/
theorem sector_area_16cm_2rad :
  sectorArea 16 2 = 16 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_16cm_2rad_l1298_129881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_price_for_revenue_increase_l1298_129847

/-- Represents the revenue function for the electricity department -/
noncomputable def revenue (a k x : ℝ) : ℝ := (a + k / (x - 0.40)) * (x - 0.30)

/-- Theorem stating the minimum electricity price for a 20% revenue increase -/
theorem min_price_for_revenue_increase (a : ℝ) (h_a : a > 0) : 
  ∃ x : ℝ, x ≥ 0.55 ∧ x ≤ 0.75 ∧ 
    (∀ y : ℝ, y ≥ 0.55 → y ≤ 0.75 → 
      revenue a (0.2 * a) y ≥ 0.60 * a → x ≤ y) ∧
    revenue a (0.2 * a) x ≥ 0.60 * a ∧
    x = 0.60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_price_for_revenue_increase_l1298_129847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_votes_l1298_129879

theorem winning_candidate_votes (v : ℕ) : 
  let total_votes : ℚ := 3136 + 7636 + v
  let winning_percentage : ℚ := 51910714285714285 / 100000000000000000
  (7636 : ℚ) = winning_percentage * total_votes
  ∧ 7636 ≥ 3136 ∧ 7636 ≥ v := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_votes_l1298_129879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_one_two_l1298_129805

open Real

theorem solution_in_interval_one_two :
  ∃ x : ℝ, x > 1 ∧ x < 2 ∧ log x = 2 - x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_one_two_l1298_129805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_ellipse_C_circle_D_inside_ellipse_C_l1298_129810

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the circle D
def circle_D (x y : ℝ) : Prop := (x+1)^2 + y^2 = 1/4

-- Define eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

-- Theorem 1: The eccentricity of ellipse C is √3/2
theorem eccentricity_of_ellipse_C :
  eccentricity 2 1 = Real.sqrt 3 / 2 := by
  sorry

-- Theorem 2: Circle D lies entirely inside ellipse C
theorem circle_D_inside_ellipse_C :
  ∀ x y : ℝ, circle_D x y → ellipse_C x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_ellipse_C_circle_D_inside_ellipse_C_l1298_129810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_theorem_l1298_129819

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℝ  -- first term
  d : ℝ  -- common difference
  n : ℕ  -- number of terms

/-- Sum of first k terms of an arithmetic progression -/
noncomputable def sum_first (ap : ArithmeticProgression) (k : ℕ) : ℝ :=
  k / 2 * (2 * ap.a + (k - 1) * ap.d)

/-- Sum of last k terms of an arithmetic progression -/
noncomputable def sum_last (ap : ArithmeticProgression) (k : ℕ) : ℝ :=
  k / 2 * (2 * ap.a + (2 * ap.n - k - 1) * ap.d)

/-- Sum of all terms except first k terms of an arithmetic progression -/
noncomputable def sum_except_first (ap : ArithmeticProgression) (k : ℕ) : ℝ :=
  (ap.n - k) / 2 * (2 * ap.a + (ap.n + k - 1) * ap.d)

/-- Sum of all terms except last k terms of an arithmetic progression -/
noncomputable def sum_except_last (ap : ArithmeticProgression) (k : ℕ) : ℝ :=
  (ap.n - k) / 2 * (2 * ap.a + (ap.n - k - 1) * ap.d)

theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  sum_first ap 13 = 0.5 * sum_last ap 13 ∧
  sum_except_first ap 3 / sum_except_last ap 3 = 5 / 4 →
  ap.n = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_theorem_l1298_129819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l1298_129832

/-- Represents a four-digit number formed by digits 1, 2, and 3 -/
def FourDigitNumber := Fin 4 → Fin 3

/-- Checks if a four-digit number uses each digit at least once -/
def uses_all_digits (n : FourDigitNumber) : Prop :=
  (∃ i, n i = 0) ∧ (∃ j, n j = 1) ∧ (∃ k, n k = 2)

/-- Checks if a four-digit number has no adjacent identical digits -/
def no_adjacent_identical (n : FourDigitNumber) : Prop :=
  ∀ i : Fin 3, n i ≠ n (i.succ)

/-- The set of valid four-digit numbers according to the problem conditions -/
def ValidNumbers : Set FourDigitNumber :=
  {n | uses_all_digits n ∧ no_adjacent_identical n}

/-- Proof that ValidNumbers is finite -/
instance : Fintype ValidNumbers := by
  sorry

/-- The main theorem stating that there are 18 valid numbers -/
theorem valid_numbers_count : Fintype.card ValidNumbers = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l1298_129832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_max_value_of_a_plus_b_l1298_129829

-- Define the function f as noncomputable
noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - (1/2) * a * x^2 - b

-- Part 1
theorem range_of_f_on_interval :
  let f₁ := f 1 1
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, 
    Real.exp (-1 : ℝ) - (3/2) ≤ f₁ x ∧ f₁ x ≤ Real.exp 1 - (3/2) := by
  sorry

-- Part 2
theorem max_value_of_a_plus_b :
  ∀ a b : ℝ, (∀ x : ℝ, f a b x ≥ 0) → a + b ≤ Real.exp (-Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_on_interval_max_value_of_a_plus_b_l1298_129829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_v_cost_l1298_129887

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The cost to fill a cylinder with gasoline -/
noncomputable def fillCost (c : Cylinder) (price_per_volume : ℝ) : ℝ :=
  Real.pi * c.radius^2 * c.height * price_per_volume

theorem can_v_cost (can_b can_v : Cylinder) (half_b_cost : ℝ) :
  can_v.radius = 2 * can_b.radius →
  can_v.height = can_b.height / 2 →
  fillCost can_v (half_b_cost / (Real.pi * can_b.radius^2 * can_b.height / 2)) = 4 * half_b_cost := by
  sorry

#check can_v_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_v_cost_l1298_129887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_primes_l1298_129842

theorem sum_of_special_primes : 
  (Finset.filter (fun p => Nat.Prime p ∧ 
                           p > 1 ∧ 
                           p < 100 ∧ 
                           p % 4 = 1 ∧ 
                           p % 5 = 4) 
                 (Finset.range 100)).sum id = 139 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_primes_l1298_129842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1298_129863

theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h1 : Real.tan (α + π/4) = 1/2) 
  (h2 : -π/2 < α ∧ α < 0) : 
  (2 * Real.sin α^2 + Real.sin (2*α)) / Real.cos (α - π/4) = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l1298_129863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_remainder_l1298_129866

theorem p_remainder (p : ℕ) (h : (p : ℝ) / 35 = 17.45) : p % 35 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_remainder_l1298_129866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_eight_points_l1298_129868

/-- The minimum total distance traveled by n equally spaced points on a circle
    visiting all non-adjacent points -/
noncomputable def minTotalDistance (r : ℝ) (n : ℕ) : ℝ :=
  let chordLength := r * Real.sqrt (2 - 2 * Real.cos (2 * Real.pi / n))
  let diameterLength := 2 * r
  let distancePerPoint := 3 * chordLength + 2 * diameterLength
  n * distancePerPoint

/-- Theorem stating the minimum total distance for 8 points on a circle of radius 50 -/
theorem min_distance_eight_points :
  minTotalDistance 50 8 = 1200 * Real.sqrt 2 + 1600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_eight_points_l1298_129868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_table_height_l1298_129812

noncomputable def triangle_height (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h_a := 2 * area / a
  let h_b := 2 * area / b
  let h_c := 2 * area / c
  min (h_a * h_b / (h_a + h_b)) (min (h_b * h_c / (h_b + h_c)) (h_c * h_a / (h_c + h_a)))

theorem max_table_height (de ef fd : ℝ) (h_de : de = 26) (h_ef : ef = 28) (h_fd : fd = 32) :
  triangle_height de ef fd = triangle_height 26 28 32 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_table_height_l1298_129812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_plane_halves_volume_l1298_129873

/-- Regular square pyramid -/
structure RegularSquarePyramid where
  baseEdge : ℝ
  height : ℝ

/-- Plane that intersects the pyramid -/
structure IntersectingPlane (p : RegularSquarePyramid) where
  baseEdge : ℝ
  intersectionSegment : ℝ
  alongBaseEdge : baseEdge = p.baseEdge
  goldenRatio : p.baseEdge / intersectionSegment = (Real.sqrt 5 + 1) / 2

/-- Volume of a portion of the pyramid -/
noncomputable def partialVolume (p : RegularSquarePyramid) (base : ℝ) : ℝ :=
  (1 / 3) * base * base * p.height

/-- Theorem: The intersecting plane halves the volume of the pyramid -/
theorem intersecting_plane_halves_volume (p : RegularSquarePyramid) (plane : IntersectingPlane p) :
  partialVolume p plane.intersectionSegment = partialVolume p (p.baseEdge - plane.intersectionSegment) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_plane_halves_volume_l1298_129873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_formula_l1298_129827

/-- A tetrahedron with a sphere touching all its edges -/
structure SphericalTetrahedron where
  /-- Length of one of the two opposite edges -/
  a : ℝ
  /-- Length of the other opposite edge -/
  b : ℝ
  /-- Length of all other edges -/
  x : ℝ
  /-- All edges are positive -/
  ha : a > 0
  hb : b > 0
  hx : x > 0
  /-- The sphere touches all edges -/
  touches_all_edges : True

/-- The radius of the sphere touching all edges of the tetrahedron -/
noncomputable def sphere_radius (t : SphericalTetrahedron) : ℝ :=
  Real.sqrt (2 * t.a * t.b) / 2

/-- Theorem stating that the radius of the sphere is √(2ab)/2 -/
theorem sphere_radius_formula (t : SphericalTetrahedron) :
    sphere_radius t = Real.sqrt (2 * t.a * t.b) / 2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_formula_l1298_129827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_of_two_eq_nineteen_l1298_129806

/-- Given a function P(t) = a^t + b^t where a and b are complex numbers,
    if P(1) = 7 and P(3) = 28, then P(2) = 19 -/
theorem p_of_two_eq_nineteen (a b : ℂ) (P : ℝ → ℂ)
  (h_def : ∀ t : ℝ, P t = a^(t:ℂ) + b^(t:ℂ))
  (h_one : P 1 = 7)
  (h_three : P 3 = 28) :
  P 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_of_two_eq_nineteen_l1298_129806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_S_l1298_129835

-- Define the set of points
def S : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p;
                    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
                    x + y ≤ 1 ∧ y + z ≤ 1 ∧ z + x ≤ 1}

-- State the theorem
theorem volume_of_S : MeasureTheory.volume S = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_S_l1298_129835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_circle_distance_l1298_129858

/-- The distance between two non-adjacent points on a regular pentagon inscribed in a circle of radius 20 -/
noncomputable def d : ℝ := 2 * 20 * Real.sqrt (1 - (1 + Real.sqrt 5) / 4)

/-- The total distance traveled by five friends on a circle -/
noncomputable def total_distance : ℝ := 400 + 10 * d

/-- Theorem stating the total distance traveled by five friends -/
theorem friends_circle_distance :
  ∀ (radius : ℝ) (num_friends : ℕ) (friends_visited : ℕ),
    radius = 20 →
    num_friends = 5 →
    friends_visited = 2 →
    total_distance = 400 + 10 * d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_circle_distance_l1298_129858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_show_scientific_notation_l1298_129846

noncomputable def number : ℝ := 81600

-- Scientific notation representation
noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

-- Definition of valid scientific notation
def is_valid_scientific_notation (a : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10

theorem auto_show_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    number = scientific_notation a n ∧ 
    is_valid_scientific_notation a ∧
    a = 8.16 ∧ 
    n = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auto_show_scientific_notation_l1298_129846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1298_129896

/-- Calculates the speed of a train given its length, the time it takes to cross a person
    moving in the opposite direction, and the person's speed. -/
noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) (person_speed : ℝ) : ℝ :=
  let relative_speed := train_length / crossing_time
  let train_speed_ms := relative_speed - (person_speed * 1000 / 3600)
  train_speed_ms * 3600 / 1000

/-- Theorem stating that a train with the given parameters has a speed of 25 km/h -/
theorem train_speed_problem : train_speed 210 28 2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1298_129896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_segment_bound_l1298_129800

/-- A type representing a segment in the plane -/
structure Segment where
  isHorizontal : Bool

/-- A type representing a curve in the plane -/
structure Curve where
  intersections : Finset Segment
  different_pairs : intersections.card = 2

/-- The main theorem statement -/
theorem curve_segment_bound 
  (n : ℕ) 
  (segments : Finset Segment) 
  (m : ℕ) 
  (curves : Finset Curve) 
  (h_segments : segments.card = n)
  (h_curves : curves.card = m)
  (h_intersections : ∀ c ∈ curves, c.intersections ⊆ segments)
  (h_different : ∀ c₁ c₂, c₁ ∈ curves → c₂ ∈ curves → c₁ ≠ c₂ → c₁.intersections ≠ c₂.intersections) :
  ∃ (c : ℝ) (n₀ : ℕ), ∀ n ≥ n₀, (m : ℝ) ≤ c * n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_segment_bound_l1298_129800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1298_129883

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

theorem f_properties :
  (∀ x y : ℝ, x < y ∧ ((x ≤ -2 ∧ y ≤ -2) ∨ (x ≥ 2 ∧ y ≥ 2)) → f x < f y) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 4 → f x ≥ -4/3) ∧
  (f 2 = -4/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1298_129883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_double_weight_8n_plus_5_eq_4n_plus_3_weight_2_pow_n_minus_1_l1298_129889

-- Define the binary representation of a non-negative integer
def binary_rep (n : ℕ) : List Bool :=
  sorry

-- Define the weight function ω
def ω (n : ℕ) : ℕ :=
  (binary_rep n).foldl (λ acc b => if b then acc + 1 else acc) 0

-- Theorem statements
theorem weight_double (n : ℕ) : ω (2 * n) = ω n := by
  sorry

theorem weight_8n_plus_5_eq_4n_plus_3 (n : ℕ) : ω (8 * n + 5) = ω (4 * n + 3) := by
  sorry

theorem weight_2_pow_n_minus_1 (n : ℕ) : ω (2^n - 1) = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_double_weight_8n_plus_5_eq_4n_plus_3_weight_2_pow_n_minus_1_l1298_129889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1298_129869

/-- Given a complex number z₁ such that z₂ = z₁ + 1/z₁ is real and -1 ≤ z₂ ≤ 1,
    prove that |z₁| = 1, the real part of z₁ is in [-1/2, 1/2],
    and ω = (1 - z₁)/(1 + z₁) is purely imaginary. -/
theorem complex_number_properties (z₁ : ℂ) 
    (h1 : (z₁ + z₁⁻¹).im = 0)
    (h2 : -1 ≤ (z₁ + z₁⁻¹).re ∧ (z₁ + z₁⁻¹).re ≤ 1) :
  Complex.abs z₁ = 1 ∧ 
  -1/2 ≤ z₁.re ∧ z₁.re ≤ 1/2 ∧
  ((1 - z₁) / (1 + z₁)).re = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1298_129869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1298_129801

-- Define the hyperbola
def is_hyperbola (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (P.1^2 / a^2) - (P.2^2 / b^2) = 1

-- Define the foci of the hyperbola
noncomputable def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 + b^2)
  ((c, 0), (-c, 0))

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

-- The main theorem
theorem hyperbola_eccentricity_range (a b : ℝ) (P : ℝ × ℝ) :
  is_hyperbola a b P →
  (let (F₁, F₂) := foci a b
   distance P F₁ = 3 * distance P F₂) →
  1 < eccentricity a b ∧ eccentricity a b ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1298_129801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_100_degree_angle_impossible_l1298_129860

theorem right_triangle_with_100_degree_angle_impossible :
  ¬ ∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧  -- positive side lengths
    a^2 + b^2 = c^2 ∧        -- Pythagorean theorem (right angle)
    (a / c) = Real.sin (100 * π / 180) -- one angle is 100 degrees
    := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_100_degree_angle_impossible_l1298_129860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l1298_129872

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem min_translation_for_symmetry :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ (x : ℝ), f (x + φ) = f (-x + φ)) ∧
  (∀ (ψ : ℝ), ψ > 0 ∧ (∀ (x : ℝ), f (x + ψ) = f (-x + ψ)) → ψ ≥ φ) ∧
  φ = Real.pi / 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l1298_129872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_track_length_l1298_129849

/-- The vertical rise of the railroad track in feet -/
noncomputable def vertical_rise : ℝ := 800

/-- The initial grade of the track as a percentage -/
noncomputable def initial_grade : ℝ := 4

/-- The desired grade of the track as a percentage -/
noncomputable def desired_grade : ℝ := 2.5

/-- Calculate the horizontal length for a given grade percentage -/
noncomputable def horizontal_length (grade : ℝ) : ℝ := vertical_rise / (grade / 100)

/-- The additional track length required to reduce the grade -/
noncomputable def additional_length : ℝ := horizontal_length desired_grade - horizontal_length initial_grade

theorem additional_track_length :
  additional_length = 12000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_track_length_l1298_129849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_product_l1298_129851

theorem parallel_vectors_product (m n : ℝ) : 
  let a : Fin 3 → ℝ := ![m, -6, 2]
  let b : Fin 3 → ℝ := ![4, n, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) → m * n = -24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_product_l1298_129851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_problem_l1298_129825

noncomputable def average_of_multiples (x : ℝ) : ℝ :=
  (x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x) / 7

noncomputable def median_of_multiples (n : ℕ) : ℝ :=
  2 * (n : ℝ)

theorem multiples_problem (x : ℝ) (h : x > 0) :
  (average_of_multiples x)^2 = (median_of_multiples 8)^2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_problem_l1298_129825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_m_bound_l1298_129815

-- Define the function f as noncomputable due to the use of Real.instPowReal
noncomputable def f (x m : ℝ) : ℝ := 2^(|2*x - m|)

-- State the theorem
theorem f_increasing_implies_m_bound (m : ℝ) :
  (∀ x y : ℝ, 2 ≤ x → x < y → f x m < f y m) →
  m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_m_bound_l1298_129815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finitely_many_zeros_l1298_129880

/-- A polynomial with real coefficients of degree at least 1 -/
def RealPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ (n : ℕ), n ≥ 1 ∧ ∃ (coeffs : Finset ℝ), p = λ x ↦ (Finset.sum coeffs (λ c ↦ c * x^n))

/-- The set of α values satisfying the integral conditions -/
noncomputable def IntegralZeros (p : ℝ → ℝ) : Set ℝ :=
  {α : ℝ | ∫ x in Set.Icc 0 α, p x * Real.sin x = 0 ∧ ∫ x in Set.Icc 0 α, p x * Real.cos x = 0}

/-- The main theorem -/
theorem finitely_many_zeros (p : ℝ → ℝ) (hp : RealPolynomial p) :
    (IntegralZeros p).Finite := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finitely_many_zeros_l1298_129880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_percentage_l1298_129833

noncomputable def net_salary : ℝ := 3500

noncomputable def discretionary_income : ℝ := (1/5) * net_salary

noncomputable def savings_percentage : ℝ := 20

noncomputable def eating_out_socializing_percentage : ℝ := 35

noncomputable def gifts_charitable_amount : ℝ := 105

theorem vacation_fund_percentage :
  let remaining_percentage : ℝ := 100 - savings_percentage - eating_out_socializing_percentage - (gifts_charitable_amount / discretionary_income * 100)
  remaining_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_percentage_l1298_129833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_segment_volume_ratio_theorem_l1298_129885

/-- The ratio of the volume of a spherical segment to the volume of the entire sphere -/
noncomputable def spherical_segment_volume_ratio (α : Real) : Real :=
  (Real.sin (α / 4))^4 * (2 + Real.cos (α / 2))

/-- Volume of a spherical segment -/
noncomputable def spherical_segment_volume (α R : Real) : Real :=
  (4 / 3) * Real.pi * R^3 * (Real.sin (α / 4))^4 * (2 + Real.cos (α / 2))

/-- Volume of a sphere -/
noncomputable def sphere_volume (R : Real) : Real :=
  (4 / 3) * Real.pi * R^3

/-- Theorem: The ratio of the volume of a spherical segment to the volume of the entire sphere,
    where the arc in the axial section of the segment corresponds to a central angle α,
    is equal to sin⁴(α/4) * (2 + cos(α/2)). -/
theorem spherical_segment_volume_ratio_theorem (α R : Real) :
  let V_segment := spherical_segment_volume α R
  let V_sphere := sphere_volume R
  V_segment / V_sphere = spherical_segment_volume_ratio α :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_segment_volume_ratio_theorem_l1298_129885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1298_129890

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

noncomputable def f' (x : ℝ) : ℝ := 2*x - 1/x^2

theorem tangent_line_at_one :
  ∃ (m b : ℝ), 
    (f' 1 = m) ∧ 
    (f 1 = 2) ∧ 
    (∀ x y : ℝ, y - 2 = m * (x - 1) ↔ x - y + 1 = 0) := by
  -- We'll use m = 1 and b = 1
  use 1, 1
  constructor
  · -- Prove f' 1 = 1
    sorry
  constructor
  · -- Prove f 1 = 2
    sorry
  · -- Prove the equivalence of the line equations
    intro x y
    apply Iff.intro
    · intro h
      linarith
    · intro h
      linarith

-- The proof is incomplete, but the structure is correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l1298_129890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_l1298_129867

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x - Real.sqrt 3 * y - 6 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the slope of line AP
noncomputable def slope_AP : ℝ := Real.tan (30 * Real.pi / 180)

-- Define point A as the intersection of line l and line AP
def point_A (x y : ℝ) : Prop :=
  line_l x y ∧ y - 2 = slope_AP * (x - 0)

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_min_distance :
  ∀ (px py ax ay : ℝ),
    curve_C px py →
    point_A ax ay →
    (∃ (d : ℝ), d = distance px py ax ay ∧
      d ≤ 2 * Real.sqrt 3 + 4 ∧
      d ≥ 2 * Real.sqrt 3 ∧
      (d = 2 * Real.sqrt 3 + 4 ∨ d = 2 * Real.sqrt 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_l1298_129867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sqrt_ratio_ge_one_l1298_129824

open BigOperators

theorem sum_sqrt_ratio_ge_one (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, a i > 0) :
  ∑ i, Real.sqrt (
    (a i ^ (n - 1)) / 
    ((a i ^ (n - 1)) + ((n^2 - 1) * ∏ j, if j ≠ i then a j else 1))
  ) ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_sqrt_ratio_ge_one_l1298_129824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equal_l1298_129870

/-- A function is of second degree if it can be written as ax² + bx + c where a ≠ 0 -/
def SecondDegree (g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, g x = a * x^2 + b * x + c

/-- A function has solutions for a linear equation -/
def HasSolutions (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∃ x : ℝ, f x = m * x + n

theorem functions_equal
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h₁ : SecondDegree g)
  (h₂ : ∀ m n : ℝ, HasSolutions f m n ↔ HasSolutions g m n) :
  ∀ x : ℝ, f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_equal_l1298_129870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_rotational_phenomena_l1298_129864

-- Define the type for phenomena
inductive Phenomenon
  | GroundwaterDecrease
  | ConveyorBelt
  | SteeringWheel
  | Faucet
  | Pendulum
  | Swing

-- Define a function to check if a phenomenon is rotational
def isRotational (p : Phenomenon) : Bool :=
  match p with
  | Phenomenon.SteeringWheel => true
  | Phenomenon.Faucet => true
  | Phenomenon.Pendulum => true
  | Phenomenon.Swing => true
  | _ => false

-- Define the list of all phenomena
def allPhenomena : List Phenomenon :=
  [Phenomenon.GroundwaterDecrease, Phenomenon.ConveyorBelt, Phenomenon.SteeringWheel,
   Phenomenon.Faucet, Phenomenon.Pendulum, Phenomenon.Swing]

-- Theorem stating that there are exactly four rotational phenomena
theorem four_rotational_phenomena :
  (allPhenomena.filter isRotational).length = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_rotational_phenomena_l1298_129864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1298_129820

def M : Set ℤ := {x | x^2 - 3*x - 4 ≤ 0}
def N : Set ℤ := {x : ℤ | 0 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1298_129820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1298_129856

/-- Given a number of people and an amount of work, calculates the time taken to complete the work -/
noncomputable def time_to_complete (people : ℕ) (work : ℝ) (days : ℝ) : ℝ :=
  work / (people : ℝ) * days

theorem work_completion_time 
  (P : ℕ) -- Original number of people
  (W : ℝ) -- Total work to be done
  (h : time_to_complete (2 * P) (W / 2) 5 = W / 2) : -- Given condition
  time_to_complete P W 20 = W := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1298_129856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_thirty_degrees_l1298_129895

theorem tangent_thirty_degrees (x y : ℝ) :
  (∃ (A : ℝ × ℝ), A = (x, y) ∧ A ≠ (0, 0) ∧ 
   x * Real.cos (30 * π / 180) = y * Real.sin (30 * π / 180)) →
  y / x = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_thirty_degrees_l1298_129895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_budget_is_12200_l1298_129888

/-- Represents the annual budget of Centerville in dollars -/
noncomputable def total_budget : ℝ := 3000 / 0.15

/-- Represents the amount spent on the public library in dollars -/
def library_budget : ℝ := 3000

/-- Represents the percentage of the budget spent on the public library -/
def library_percentage : ℝ := 0.15

/-- Represents the percentage of the budget spent on public parks -/
def parks_percentage : ℝ := 0.24

/-- Represents the amount spent on public parks in dollars -/
noncomputable def parks_budget : ℝ := total_budget * parks_percentage

/-- Theorem stating that the remaining budget is $12,200 -/
theorem remaining_budget_is_12200 :
  total_budget - (library_budget + parks_budget) = 12200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_budget_is_12200_l1298_129888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l1298_129848

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / (2^x + 1) + Real.sin x

-- State the theorem
theorem sum_of_f_values : 
  f (-2) + f (-1) + f 0 + f 1 + f 2 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l1298_129848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1298_129877

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6) * Real.sin (2 * x) - 1 / 4

-- Define the interval
def interval : Set ℝ := Set.Icc (-Real.pi / 4) 0

-- Statement of the theorem
theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∃ T : ℝ, T = Real.pi / 2 ∧ T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x ∈ interval, f x ≤ 1 / 4) ∧
  (∃ x ∈ interval, f x = 1 / 4) ∧
  (∀ x ∈ interval, f x ≥ -1 / 2) ∧
  (∃ x ∈ interval, f x = -1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1298_129877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_ratio_of_men_to_women_l1298_129899

theorem initial_ratio_of_men_to_women 
  (initial_men : ℕ) 
  (initial_women : ℕ) 
  (men_entered : ℕ) 
  (women_left : ℕ) 
  (final_men : ℕ) 
  (final_women : ℕ) 
  (h1 : men_entered = 2) 
  (h2 : women_left = 3) 
  (h3 : final_men = initial_men + men_entered) 
  (h4 : final_women = (initial_women - women_left) * 2) 
  (h5 : final_men = 14) 
  (h6 : final_women = 24) : 
  (initial_men : ℚ) / initial_women = 4 / 5 := by
  sorry

#check initial_ratio_of_men_to_women

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_ratio_of_men_to_women_l1298_129899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_half_equals_third_nick_full_l1298_129878

-- Define the cleaning rates and times
noncomputable def john_full_time : ℝ := 6
noncomputable def together_time : ℝ := 3.6

-- Define Nick's cleaning time as a function of John's and their combined time
noncomputable def nick_full_time : ℝ := john_full_time * together_time / (john_full_time - together_time)

-- Define John's time to clean half the house
noncomputable def john_half_time : ℝ := john_full_time / 2

-- Theorem to prove
theorem john_half_equals_third_nick_full :
  john_half_time = (1 / 3) * nick_full_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_half_equals_third_nick_full_l1298_129878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_minimum_distance_sum_l1298_129843

/-- Given a triangle ABC with side lengths a, b, c, the minimum value of 
    |MA|^2 + |MB|^2 + |MC|^2 for any point M in the plane is (a^2 + b^2 + c^2) / 3 -/
theorem triangle_centroid_minimum_distance_sum 
  (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (A B C M : ℝ × ℝ),
    let dist_squared := fun (P Q : ℝ × ℝ) ↦ (P.1 - Q.1)^2 + (P.2 - Q.2)^2
    let side_lengths := (dist_squared A B = c^2) ∧ 
                        (dist_squared B C = a^2) ∧ 
                        (dist_squared C A = b^2)
    let sum_dist_squared := fun (P : ℝ × ℝ) ↦ 
      dist_squared P A + dist_squared P B + dist_squared P C
    side_lengths ∧ 
    (∀ P : ℝ × ℝ, sum_dist_squared M ≤ sum_dist_squared P) ∧
    sum_dist_squared M = (a^2 + b^2 + c^2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_minimum_distance_sum_l1298_129843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_intersects_all_lines_l1298_129852

/-- Definition of a line in 3D space using parametric form -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Definition of the four given lines -/
noncomputable def L1 : Line3D := ⟨(1, 0, 0), (0, 0, 1)⟩
noncomputable def L2 : Line3D := ⟨(0, 1, 0), (1, 0, 0)⟩
noncomputable def L3 : Line3D := ⟨(0, 0, 1), (0, 1, 0)⟩
noncomputable def L4 : Line3D := ⟨(0, 0, 0), (1, 1, -1/6)⟩

/-- Function to check if two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- The two lines that intersect all given lines -/
noncomputable def solution1 : Line3D := ⟨(1, 0, -1/2), (-1/3, 1, 1/2)⟩
noncomputable def solution2 : Line3D := ⟨(1, 0, 1/3), (1/2, 1, -1/3)⟩

/-- Theorem stating that the solution lines intersect all given lines -/
theorem solution_intersects_all_lines :
  (intersect solution1 L1 ∧ intersect solution1 L2 ∧ intersect solution1 L3 ∧ intersect solution1 L4) ∧
  (intersect solution2 L1 ∧ intersect solution2 L2 ∧ intersect solution2 L3 ∧ intersect solution2 L4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_intersects_all_lines_l1298_129852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measures_l1298_129808

/-- Given a triangle ABC where angle B = angle A + 10° and angle C = angle B + 10°,
    prove that angle A = 50°, angle B = 60°, and angle C = 70°. -/
theorem triangle_angle_measures (A B C : ℝ) 
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) :
  A = 50 ∧ B = 60 ∧ C = 70 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#check triangle_angle_measures

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measures_l1298_129808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_games_l1298_129857

theorem first_part_games : ∃ (first_part_games : ℕ),
  let total_games : ℕ := 120
  let first_win_rate : ℚ := 2/5
  let second_win_rate : ℚ := 4/5
  let overall_win_rate : ℚ := 7/10
  first_part_games * first_win_rate + (total_games - first_part_games) * second_win_rate = 
    total_games * overall_win_rate ∧ first_part_games = 30 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_games_l1298_129857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_is_17_l1298_129804

/-- A point in the 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a right triangle given its three vertices -/
noncomputable def rightTriangleArea (p1 p2 p3 : Point) : ℝ :=
  let base := |p2.x - p1.x|
  let height := |p3.y - p1.y|
  (1/2) * base * height

/-- The total area of two right triangles in the coordinate plane -/
noncomputable def totalArea : ℝ :=
  let t1p1 : Point := ⟨3, 7⟩
  let t1p2 : Point := ⟨3, 2⟩
  let t1p3 : Point := ⟨8, 2⟩
  let t2p1 : Point := ⟨-4, -1⟩
  let t2p2 : Point := ⟨-1, -1⟩
  let t2p3 : Point := ⟨-1, -4⟩
  rightTriangleArea t1p1 t1p2 t1p3 + rightTriangleArea t2p1 t2p2 t2p3

theorem total_area_is_17 : totalArea = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_is_17_l1298_129804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_values_l1298_129844

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

-- Theorem statement
theorem common_ratio_values (a₁ : ℝ) (q : ℝ) (h : a₁ ≠ 0) :
  sum_geometric a₁ q 3 = 3 * a₁ → q = 1 ∨ q = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_values_l1298_129844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_monotonic_decreasing_when_m_nonpositive_monotonicity_when_m_positive_l1298_129817

-- Define the function f(x) = me^x - x
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x - x

-- Define the derivative of f
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x - 1

-- Theorem for the tangent line equation
theorem tangent_line_at_zero (m : ℝ) (h : m = 2) :
  ∃ (a b : ℝ), a = 1 ∧ b = 2 ∧
  ∀ x, a * x + b = (f_deriv m 0) * x + f m 0 := by
  sorry

-- Theorem for monotonicity when m ≤ 0
theorem monotonic_decreasing_when_m_nonpositive (m : ℝ) (h : m ≤ 0) :
  ∀ x, f_deriv m x < 0 := by
  sorry

-- Theorem for monotonicity when m > 0
theorem monotonicity_when_m_positive (m : ℝ) (h : m > 0) :
  (∀ x, x < -Real.log m → f_deriv m x < 0) ∧
  (∀ x, x > -Real.log m → f_deriv m x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_monotonic_decreasing_when_m_nonpositive_monotonicity_when_m_positive_l1298_129817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_focus_chord_length_l1298_129886

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 6

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Theorem 1: When line l passes through the focus of the parabola, m = -1
theorem line_through_focus (m : ℝ) : 
  line_l m (parabola_focus.1) (parabola_focus.2) → m = -1 :=
by sorry

-- Theorem 2: When m = -1, the length of chord AB is 2√6
theorem chord_length : 
  ∃ (A B : ℝ × ℝ), line_l (-1) A.1 A.2 ∧ line_l (-1) B.1 B.2 ∧ 
  my_circle A.1 A.2 ∧ my_circle B.1 B.2 ∧ 
  ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) = 2 * 6^(1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_focus_chord_length_l1298_129886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1298_129803

noncomputable def f (x : ℝ) := Real.exp (-(x - 1)^2)

theorem f_inequality : f (Real.sqrt 3 / 2) > f (Real.sqrt 6 / 2) ∧ f (Real.sqrt 6 / 2) > f (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1298_129803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sum_l1298_129891

def factorial (n : ℕ) : ℕ := 
  (List.range n).foldl (· * ·) 1

def S (n : ℕ) : ℕ := 
  (List.range n).map (λ i => (i + 1) * factorial (i + 1)) |>.sum

theorem simplify_sum (n : ℕ) : S n = factorial (n + 1) - 1 := by
  sorry

#eval S 5  -- This line is optional, for testing purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sum_l1298_129891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l1298_129828

/-- The probability of drawing two balls of different colors from a box containing 3 white balls and 1 black ball is 1/2. -/
theorem different_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  white_balls = 3 →
  black_balls = 1 →
  (Nat.choose total_balls 2 : ℚ) ≠ 0 →
  (Nat.choose white_balls 1 * Nat.choose black_balls 1 : ℚ) / (Nat.choose total_balls 2 : ℚ) = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l1298_129828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_N_l1298_129838

def N : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_factors_N : (Finset.filter (· ∣ N) (Finset.range (N + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_N_l1298_129838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l1298_129816

/-- Converts a point from cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : Real) : (Real × Real × Real) :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : (Real × Real × Real) := (6, Real.pi / 3, 2)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : (Real × Real × Real) := (3, 3 * Real.sqrt 3, 2)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l1298_129816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_decorative_number_l1298_129876

/-- Represents a painter with a starting position and interval --/
structure Painter where
  start : Nat
  interval : Nat

/-- Checks if a painter paints a given picket --/
def paintsPicket (painter : Painter) (picket : Nat) : Prop :=
  ∃ k : Nat, picket = painter.start + k * painter.interval

/-- Checks if any painter from a list paints a given picket --/
def anyPaintsPicket (painters : List Painter) (picket : Nat) : Prop :=
  ∃ painter, painter ∈ painters ∧ paintsPicket painter picket

/-- Checks if exactly one painter from a list paints a given picket --/
def uniquePainter (painters : List Painter) (picket : Nat) : Prop :=
  ∃! painter, painter ∈ painters ∧ paintsPicket painter picket

/-- Checks if all pickets are painted by exactly one painter --/
def allPicketsUnique (painters : List Painter) (totalPickets : Nat) : Prop :=
  ∀ picket : Nat, picket > 0 ∧ picket ≤ totalPickets → uniquePainter painters picket

/-- Calculates the decorative number for given painter intervals --/
def decorativeNumber (g h i : Nat) : Nat :=
  100 * g + 10 * h + i

/-- Main theorem: The only decorative number is 264 --/
theorem only_decorative_number : 
  ∀ g h i : Nat, 
  g > 0 ∧ h > 0 ∧ i > 0 →
  allPicketsUnique 
    [Painter.mk 1 g, Painter.mk 3 h, Painter.mk 2 i] 
    60 →
  decorativeNumber g h i = 264 := by
  sorry

#check only_decorative_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_decorative_number_l1298_129876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_implies_a_eq_one_thirtieth_l1298_129898

noncomputable def f (a x : ℝ) : ℝ := x^2 + (Real.log (3*x))^2 - 2*a*(x + 3*Real.log (3*x)) + 10*a^2

theorem exists_x0_implies_a_eq_one_thirtieth :
  ∀ a : ℝ, (∃ x₀ : ℝ, f a x₀ ≤ 1/10) → a = 1/30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_implies_a_eq_one_thirtieth_l1298_129898
