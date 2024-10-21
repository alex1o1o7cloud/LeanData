import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l359_35987

-- Define the function f
def f (f'₁ : ℝ) (x : ℝ) : ℝ := x^2 + 2*x*f'₁ - 6

-- State the theorem
theorem f_derivative_at_one :
  ∃ (f'₁ : ℝ), (∀ x, deriv (f f'₁) x = 2*x + 2*f'₁) ∧ f'₁ = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l359_35987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stewart_farm_sheep_count_l359_35981

/-- Stewart farm problem -/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℚ),
  sheep / horses = 2 / 7 →
  230 * horses = 12880 →
  150 * sheep = 21500 →
  sheep = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stewart_farm_sheep_count_l359_35981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_eq_neg_three_l359_35959

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x^2 - x else -(2 * (-x)^2 - (-x))

-- State the theorem
theorem f_of_one_eq_neg_three :
  (∀ x, f (-x) = -f x) →  -- f is odd
  f 1 = -3 := by
  intro h
  -- The proof steps would go here
  sorry

-- You can add more lemmas or theorems as needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_eq_neg_three_l359_35959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_tetrahedron_volume_l359_35984

/-- Represents a cube with side length 2 units -/
noncomputable def cube_side_length : ℝ := 2

/-- Represents the volume of a tetrahedron -/
noncomputable def tetrahedron_volume (base_area height : ℝ) : ℝ := (1/3) * base_area * height

/-- Represents the area of a triangle -/
noncomputable def triangle_area (side1 side2 : ℝ) : ℝ := (1/2) * side1 * side2

/-- Theorem stating the volume of the inner tetrahedron -/
theorem inner_tetrahedron_volume :
  let outer_base_area := triangle_area cube_side_length cube_side_length
  let outer_height := cube_side_length
  let outer_volume := tetrahedron_volume outer_base_area outer_height
  let inner_base_area := triangle_area (cube_side_length/2) (cube_side_length/2)
  let inner_height := cube_side_length/2
  tetrahedron_volume inner_base_area inner_height = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_tetrahedron_volume_l359_35984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l359_35996

-- Define the equation
noncomputable def f (x a : ℝ) : ℝ := (2 : ℝ)^(2*x) + a * (2 : ℝ)^x + a + 1

-- Define the condition that the equation has real roots
def has_real_roots (a : ℝ) : Prop := ∃ x : ℝ, f x a = 0

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  has_real_roots a → a ≤ 2 - 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l359_35996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l359_35997

-- Define complex numbers
noncomputable def z₁ (a : ℝ) : ℂ := 2 * a - Complex.I * Real.sqrt 3
noncomputable def z₂ (b : ℝ) : ℂ := 2 * b + Complex.I
def z₃ (a b : ℝ) : ℂ := a + Complex.I * b

-- Define the theorem
theorem complex_problem (a b : ℝ) :
  Complex.abs (z₃ a b) = 1 →
  (z₁ a - z₂ b).re = 0 →
  (z₃ a b = -Real.sqrt 2 / 2 - Complex.I * Real.sqrt 2 / 2 ∨
   z₃ a b = Real.sqrt 2 / 2 + Complex.I * Real.sqrt 2 / 2) ∧
  ((a = -1/2 ∧ b = -Real.sqrt 3 / 2) ∨ (a = 1/2 ∧ b = Real.sqrt 3 / 2)) ∧
  (let S := |a + Real.sqrt 3 * b|
   S ≤ 2 ∧ ∃ (a₀ b₀ : ℝ), S = 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l359_35997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_formula_properties_l359_35903

-- Define Euler's formula
noncomputable def euler_formula (x : ℝ) : ℂ := Complex.exp (x * Complex.I)

-- Theorem statement
theorem euler_formula_properties (x : ℝ) :
  (Complex.abs (euler_formula x) = 1) ∧
  (euler_formula x + euler_formula (-x) = 2 * Complex.cos x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_formula_properties_l359_35903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_inequality_l359_35940

theorem lcm_inequality (n : ℕ) (a : ℕ → ℕ) :
  (∀ i : ℕ, i < n → a i > 0) →
  (∀ i j : ℕ, i < j → j < n → a i < a j) →
  (∀ i : ℕ, i < n → a i ≤ 2 * n) →
  (∀ i j : ℕ, i < n → j < n → i ≠ j → Nat.lcm (a i) (a j) > 2 * n) →
  a 0 > Nat.floor (2 * n / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_inequality_l359_35940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_point_l359_35934

-- Define the points and circle
noncomputable def M : ℝ × ℝ := (Real.sqrt 3, 0)
noncomputable def N : ℝ × ℝ := (-Real.sqrt 3, 0)

def circle_N (x y : ℝ) : Prop :=
  (x + Real.sqrt 3)^2 + y^2 = 16

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (k m x y : ℝ) : Prop :=
  y = k * x + m

-- Define point D
def D : ℝ × ℝ := (0, 1)

-- Define the theorem
theorem trajectory_and_fixed_point :
  ∀ (P Q : ℝ × ℝ) (k m : ℝ),
    circle_N P.1 P.2 →
    (∃ (A B : ℝ × ℝ),
      trajectory_C A.1 A.2 ∧
      trajectory_C B.1 B.2 ∧
      line_l k m A.1 A.2 ∧
      line_l k m B.1 B.2 ∧
      (A.1 - D.1) * (B.1 - D.1) + (A.2 - D.2) * (B.2 - D.2) = 0) →
    trajectory_C Q.1 Q.2 ∧
    line_l k (-3/5) 0 (-3/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_point_l359_35934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l359_35907

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- The theorem stating that if S₄/S₂ = 5 for a geometric sequence, then S₈/S₄ = 17 -/
theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  (geometricSum a₁ q 4) / (geometricSum a₁ q 2) = 5 →
  (geometricSum a₁ q 8) / (geometricSum a₁ q 4) = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l359_35907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l359_35962

-- Define the function f(x) = -x^2 - x + 4
def f (x : ℝ) : ℝ := -x^2 - x + 4

-- Define the decreasing interval
def decreasingInterval : Set ℝ := Set.Ioi (-1/2)

-- Theorem statement
theorem f_decreasing_interval :
  ∀ x y, x ∈ decreasingInterval → y ∈ decreasingInterval → x < y → f x > f y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l359_35962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l359_35902

noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x^(-m^2 - 2*m + 3)

theorem power_function_value (m : ℤ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- f is an even function
  (∀ x y : ℝ, 0 < x → x < y → f m x < f m y) →  -- f is increasing on (0, +∞)
  f m 2 = 16 := by
  intro h_even h_increasing
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l359_35902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_xoz_plane_intersection_l359_35983

/-- Given points A and B in ℝ³, and a point M on the intersection of the perpendicular bisector of AB and the xOz plane, prove the relationship among the coordinates of M. -/
theorem perpendicular_bisector_xoz_plane_intersection 
  (A B M : ℝ × ℝ × ℝ) 
  (hA : A = (1, 2, -1)) 
  (hB : B = (2, 0, 2)) 
  (h_perp_bisector : (M.1 - A.1)^2 + (M.2.1 - A.2.1)^2 + (M.2.2 - A.2.2)^2 = 
                     (M.1 - B.1)^2 + (M.2.1 - B.2.1)^2 + (M.2.2 - B.2.2)^2)
  (h_xoz_plane : M.2.1 = 0) : 
  M.1 + 3 * M.2.2 - 1 = 0 ∧ M.2.1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_xoz_plane_intersection_l359_35983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_solution_l359_35905

noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

def satisfies_equation (x : ℤ) : Prop :=
  Real.cos (deg_to_rad (x : ℝ)) = Real.sin (deg_to_rad ((x^2 : ℤ) : ℝ))

theorem least_integer_solution :
  (∃ (x : ℤ), x ≥ 1 ∧ satisfies_equation x) ∧
  (∀ (y : ℤ), y ≥ 1 ∧ satisfies_equation y → y ≥ 9) ∧
  satisfies_equation 9 :=
by
  sorry

#check least_integer_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_solution_l359_35905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_implies_omega_l359_35973

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (Real.pi / 3 - ω * x)

theorem period_implies_omega (ω : ℝ) (h : ω ≠ 0) :
  (∀ x, f ω (x + 4 * Real.pi) = f ω x) ∧ 
  (∀ T, 0 < T ∧ T < 4 * Real.pi → ∃ x, f ω (x + T) ≠ f ω x) →
  ω = 1/2 ∨ ω = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_implies_omega_l359_35973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_observed_angle_equals_original_l359_35974

/-- The observed angle through a magnifying glass is equal to the original angle -/
theorem observed_angle_equals_original (θ : ℝ) (magnification : ℝ) :
  θ > 0 → magnification > 0 → θ = θ := by
  intro h_θ h_mag
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_observed_angle_equals_original_l359_35974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l359_35917

theorem triangle_inequality (α β γ R A : ℝ) : 
  0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π ∧ 0 < R ∧ 0 < A →
  Real.tan (α/2) + Real.tan (β/2) + Real.tan (γ/2) ≤ 9 * R^2 / (4 * A) ∧
  (Real.tan (α/2) + Real.tan (β/2) + Real.tan (γ/2) = 9 * R^2 / (4 * A) ↔ α = β ∧ β = γ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l359_35917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_theorem_l359_35906

-- Define the angle measure type
def AngleMeasure : Type := ℝ

-- Define the concept of parallel lines
def Parallel (l m : Set (ℝ × ℝ)) : Prop := sorry

-- Define the angle measure function
def MeasureAngle (A B C : ℝ × ℝ) : AngleMeasure := sorry

-- State the theorem
theorem angle_measure_theorem 
  (l m : Set (ℝ × ℝ)) 
  (A B C : ℝ × ℝ) :
  Parallel l m →
  MeasureAngle A A C = (100 : ℝ) →
  MeasureAngle B B C = (140 : ℝ) →
  MeasureAngle A C B = (120 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_theorem_l359_35906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basil_price_is_correct_l359_35939

/-- Represents the ingredients and costs for Scott's ratatouille recipe --/
structure Ratatouille where
  eggplant_pounds : ℚ
  eggplant_price : ℚ
  zucchini_pounds : ℚ
  zucchini_price : ℚ
  tomato_pounds : ℚ
  tomato_price : ℚ
  onion_pounds : ℚ
  onion_price : ℚ
  basil_pounds : ℚ
  quart_yield : ℚ
  quart_price : ℚ

/-- Calculates the price per half pound of basil --/
def basil_half_pound_price (r : Ratatouille) : ℚ :=
  let total_cost := r.eggplant_pounds * r.eggplant_price +
                    r.zucchini_pounds * r.zucchini_price +
                    r.tomato_pounds * r.tomato_price +
                    r.onion_pounds * r.onion_price
  let total_revenue := r.quart_yield * r.quart_price
  let basil_cost := total_revenue - total_cost
  basil_cost / (2 * r.basil_pounds)

/-- Theorem stating that the price per half pound of basil is $2.50 --/
theorem basil_price_is_correct (r : Ratatouille)
  (h1 : r.eggplant_pounds = 5)
  (h2 : r.eggplant_price = 2)
  (h3 : r.zucchini_pounds = 4)
  (h4 : r.zucchini_price = 2)
  (h5 : r.tomato_pounds = 4)
  (h6 : r.tomato_price = 7/2)
  (h7 : r.onion_pounds = 3)
  (h8 : r.onion_price = 1)
  (h9 : r.basil_pounds = 1)
  (h10 : r.quart_yield = 4)
  (h11 : r.quart_price = 10) :
  basil_half_pound_price r = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basil_price_is_correct_l359_35939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersects_parabola_l359_35954

noncomputable def normal_intersection (a : ℝ × ℝ) : ℝ × ℝ :=
  (9/4, 81/16)

noncomputable def parabola (x : ℝ) : ℝ := x^2

noncomputable def normal_line (x : ℝ) : ℝ := -1/4 * x + 9/2

theorem normal_intersects_parabola (a : ℝ × ℝ) (h1 : a = (2, 4)) :
  let b := normal_intersection a
  (parabola b.1 = b.2) ∧ 
  (normal_line b.1 = b.2) ∧ 
  (b ≠ a) := by
  sorry

#check normal_intersects_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_intersects_parabola_l359_35954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_l359_35958

/-- Calculates the distance traveled in one direction for a round trip -/
noncomputable def calculate_distance (total_time : ℝ) (outbound_speed : ℝ) (return_speed : ℝ) : ℝ :=
  (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed)

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_trip_distance :
  let total_time : ℝ := 3
  let outbound_speed : ℝ := 16
  let return_speed : ℝ := 24
  round_to_hundredth (calculate_distance total_time outbound_speed return_speed) = 28.80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_distance_l359_35958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adults_divisible_by_max_tables_l359_35948

/-- Represents the number of adults at the event -/
def num_adults : ℕ := sorry

/-- Represents the number of children at the event -/
def num_children : ℕ := 20

/-- Represents the maximum number of tables that can be set up -/
def max_tables : ℕ := 4

/-- Theorem stating that the number of adults must be divisible by the maximum number of tables -/
theorem adults_divisible_by_max_tables :
  num_children % max_tables = 0 →
  ∃ k : ℕ, num_adults = k * max_tables :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adults_divisible_by_max_tables_l359_35948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y6_is_zero_l359_35922

/-- The coefficient of x³y⁶ in the expansion of ((x-y)²(x+y)⁷) -/
def coefficient_x3y6 : ℤ :=
  (Nat.choose 7 6) - 2 * (Nat.choose 7 5) + (Nat.choose 7 4)

/-- Theorem stating that the coefficient of x³y⁶ in ((x-y)²(x+y)⁷) is 0 -/
theorem coefficient_x3y6_is_zero : coefficient_x3y6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y6_is_zero_l359_35922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l359_35972

noncomputable def g (x : ℝ) : ℝ := |⌊x⌋ + 1| - |⌊2 - x⌋ + 1|

theorem g_symmetry (x : ℝ) : g x = g (3 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l359_35972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_may_savings_l359_35978

/-- Calculates the savings for a given month in Ofelia's savings challenge -/
def savings : Nat → Nat
  | 0 => 10  -- Base case (January)
  | n + 1 => 2 * savings n  -- Recursive case for subsequent months

/-- Theorem stating that Ofelia's savings in May (5th month) will be $160 -/
theorem may_savings : savings 4 = 160 := by
  -- Proof steps would go here
  sorry

#eval savings 4  -- This will evaluate the function for the 5th month (index 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_may_savings_l359_35978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sixth_term_l359_35900

/-- Represents a geometric sequence with first term 1 -/
noncomputable def GeometricSequence (q : ℝ) : ℕ → ℝ :=
  λ n => q ^ (n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def GeometricSum (q : ℝ) (n : ℕ) : ℝ :=
  (1 - q^n) / (1 - q)

theorem geometric_sequence_sixth_term (q : ℝ) :
  (GeometricSum q 6 = 9 * GeometricSum q 3) →
  GeometricSequence q 6 = 32 := by
  intro h
  sorry

#check geometric_sequence_sixth_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sixth_term_l359_35900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_quotient_l359_35937

/-- S_k is an integer whose base-ten representation is a sequence of k ones -/
def S (k : ℕ) : ℕ := (13^k - 1) / 12

/-- The quotient of S_30 divided by S_5 -/
def Q : ℕ := S 30 / S 5

/-- The number of zeros in the quotient Q -/
def num_zeros : ℕ := 5 * (13^5 - 1)

/-- Convert a natural number to a string -/
def natToString (n : ℕ) : String := toString n

/-- Count occurrences of a character in a string -/
def countChar (s : String) (c : Char) : ℕ := s.toList.filter (· == c) |>.length

theorem zeros_in_quotient : num_zeros = (natToString Q).length - countChar (natToString Q) '1' := by
  sorry

#eval num_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_in_quotient_l359_35937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_approx_l359_35999

-- Define the dimensions of the vessel and the water level rise
noncomputable def vessel_length : ℝ := 20
noncomputable def vessel_width : ℝ := 15
noncomputable def water_level_rise : ℝ := 16.376666666666665

-- Define the volume of the cube
noncomputable def cube_volume : ℝ := vessel_length * vessel_width * water_level_rise

-- Define the edge length of the cube
noncomputable def cube_edge_length : ℝ := cube_volume ^ (1/3 : ℝ)

-- Theorem statement
theorem cube_edge_length_approx :
  ∀ ε > 0, |cube_edge_length - 17.1| < ε := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_approx_l359_35999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_quadratic_inequality_l359_35929

noncomputable section

-- Define the quadratic function
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x - 1/m

-- State the theorem
theorem solve_quadratic_inequality (m n : ℝ) :
  (∀ x, f m n x < 0 ↔ x < -1/2 ∨ x > 2) →
  m - n = -5/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_quadratic_inequality_l359_35929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_of_f_l359_35998

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 1) * exp x

-- State the theorem
theorem max_difference_of_f (t : ℝ) (h₁ : t ∈ Set.Icc (-3) (-1)) :
  ∃ (M : ℝ), M = 4 * exp 1 ∧ 
  ∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc t (t + 2) → x₂ ∈ Set.Icc t (t + 2) → 
  |f x₁ - f x₂| ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_of_f_l359_35998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l359_35941

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * Real.sin x

theorem range_of_f :
  Set.range f = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l359_35941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_regular_ngon_on_grid_l359_35942

/-- A grid point in a 2D plane -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  vertices : Fin n → GridPoint

/-- Theorem: No regular n-gon (n ≠ 4) exists with vertices on grid points -/
theorem no_regular_ngon_on_grid (n : ℕ) (h : n ≠ 4) :
  ¬∃ (p : RegularPolygon), p.n = n ∧ (∀ i, p.vertices i ∈ Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_regular_ngon_on_grid_l359_35942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andy_questions_wrong_l359_35936

/-- Represents the number of questions a person got wrong in a test -/
abbrev QuestionsWrong := Nat

/-- Represents a group of four people taking a test -/
structure TestGroup where
  andy : QuestionsWrong
  beth : QuestionsWrong
  charlie : QuestionsWrong
  daniel : QuestionsWrong

/-- The conditions of the test problem -/
def TestConditions (g : TestGroup) : Prop :=
  (g.andy + g.beth = g.charlie + g.daniel + 6) ∧
  (g.andy + g.daniel = g.beth + g.charlie + 4) ∧
  (g.charlie = 10)

theorem andy_questions_wrong (g : TestGroup) 
  (h : TestConditions g) : g.andy = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_andy_questions_wrong_l359_35936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brady_earnings_l359_35957

/-- Represents the payment structure and completed work for Brady's recipe card transcription job. -/
structure RecipeCardJob where
  basic_pay : ℚ := 70 / 100
  gourmet_pay : ℚ := 90 / 100
  advanced_pay : ℚ := 110 / 100
  basic_count : ℕ := 110
  gourmet_count : ℕ := 60
  advanced_count : ℕ := 40
  total_count : ℕ := basic_count + gourmet_count + advanced_count

/-- Calculates the bonus based on the total number of cards completed. -/
def calculate_bonus (total_cards : ℕ) : ℚ :=
  if total_cards ≥ 200 then 40
  else if total_cards ≥ 150 then 30
  else if total_cards ≥ 100 then 20
  else if total_cards ≥ 50 then 10
  else 0

/-- Calculates the total earnings for the RecipeCardJob. -/
def total_earnings (job : RecipeCardJob) : ℚ :=
  job.basic_pay * job.basic_count +
  job.gourmet_pay * job.gourmet_count +
  job.advanced_pay * job.advanced_count +
  calculate_bonus job.total_count

/-- Theorem stating that Brady's total earnings are $215.00. -/
theorem brady_earnings (job : RecipeCardJob) : total_earnings job = 215 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brady_earnings_l359_35957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_percentage_of_u_l359_35920

-- Define variables
variable (x y z w v u : ℝ)

-- Define the conditions
def condition1 (x y : ℝ) : Prop := x = 1.30 * y
def condition2 (y z : ℝ) : Prop := y = 0.60 * z
def condition3 (w x : ℝ) : Prop := w = 1.25 * x^2
def condition4 (v w : ℝ) : Prop := v = 0.85 * w^2
def condition5 (u z : ℝ) : Prop := u = 1.20 * z^2

-- Theorem statement
theorem v_percentage_of_u 
  (h1 : condition1 x y)
  (h2 : condition2 y z)
  (h3 : condition3 w x)
  (h4 : condition4 v w)
  (h5 : condition5 u z) :
  ∃ ε > 0, |v / u - 0.3414| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_percentage_of_u_l359_35920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l359_35918

/-- The area of a trapezoid with bases 4h and 5h, and height h, is 9h²/2 -/
theorem trapezoid_area (h : ℝ) : 
  (1 / 2) * ((4 * h) + (5 * h)) * h = (9 / 2) * h^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l359_35918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recorder_price_theorem_l359_35909

/-- Represents a two-digit price -/
structure TwoDigitPrice where
  tens : Nat
  units : Nat
  is_valid : tens < 10 ∧ units < 10 ∧ tens * 10 + units < 50

/-- Calculates the numeric value of a two-digit price -/
def TwoDigitPrice.value (p : TwoDigitPrice) : Nat :=
  p.tens * 10 + p.units

/-- Swaps the digits of a two-digit price -/
def TwoDigitPrice.swap (p : TwoDigitPrice) : TwoDigitPrice where
  tens := p.units
  units := p.tens
  is_valid := by sorry

/-- Increases a price by 20% -/
def increase_by_20_percent (price : Nat) : Nat :=
  (price * 120) / 100

theorem recorder_price_theorem (old_price : TwoDigitPrice) :
  increase_by_20_percent old_price.value = (TwoDigitPrice.swap old_price).value →
  (TwoDigitPrice.swap old_price).value = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recorder_price_theorem_l359_35909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_circular_track_l359_35964

noncomputable def time_to_meet (track_length : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  track_length / relative_speed

noncomputable def kmph_to_ms (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

theorem meeting_time_circular_track :
  let track_length : ℝ := 250
  let speed1_kmph : ℝ := 20
  let speed2_kmph : ℝ := 40
  let speed1_ms := kmph_to_ms speed1_kmph
  let speed2_ms := kmph_to_ms speed2_kmph
  let meeting_time := time_to_meet track_length speed1_ms speed2_ms
  ∃ ε > 0, |meeting_time - 15| < ε := by
  sorry

-- The following lines are commented out because they depend on noncomputable functions
-- #eval kmph_to_ms 20
-- #eval kmph_to_ms 40
-- #eval time_to_meet 250 (kmph_to_ms 20) (kmph_to_ms 40)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_circular_track_l359_35964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l359_35990

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (Real.arctan x)^2 + (Real.arctan (1/x))^2

-- State the theorem
theorem g_range : ∀ x > 0, π^2/8 ≤ g x ∧ g x ≤ π^2/4 ∧
  (∃ x₁ > 0, g x₁ = π^2/8) ∧ (∃ x₂ > 0, g x₂ = π^2/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l359_35990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_problem_l359_35955

def taxi_distances : List Int := [9, -3, -6, 4, -8, 6, -3, -6, -4, 10]
def price_per_km : Rat := 2

theorem taxi_problem :
  (taxi_distances.sum = -1) ∧
  (2 * (taxi_distances.map Int.natAbs).sum = 118) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_problem_l359_35955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roller_coaster_probability_l359_35956

/-- The number of cars in the roller coaster -/
def num_cars : ℕ := 5

/-- The number of rides the passenger takes -/
def num_rides : ℕ := 5

/-- The probability of riding in a specific car on a single ride -/
def prob_single_car : ℚ := 1 / num_cars

/-- The total number of possible outcomes for all rides -/
def total_outcomes : ℕ := num_cars ^ num_rides

/-- The number of ways to ride in each car exactly once -/
def favorable_outcomes : ℕ := Nat.factorial num_cars

theorem roller_coaster_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 120 / 3125 := by
  sorry

#eval favorable_outcomes
#eval total_outcomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roller_coaster_probability_l359_35956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sine_graph_l359_35985

noncomputable section

-- Define the original function
def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

-- Define the target function
def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

-- Theorem statement
theorem shift_sine_graph :
  ∀ x : ℝ, f (x - Real.pi / 3) = g x :=
by
  intro x
  simp [f, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sine_graph_l359_35985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l359_35953

def sequence_a (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => 3^n - 2 * sequence_a a₁ n

theorem sequence_properties (a₁ : ℝ) :
  -- Part 1: {a_n - 3^n/5} is a geometric sequence
  (∃ r : ℝ, ∀ n : ℕ, (sequence_a a₁ (n + 1) - (3^(n + 1))/5) = r * (sequence_a a₁ n - (3^n)/5)) ∧
  -- Part 2: When a₁ = 3/2, a₄, a₅, a₆ form an arithmetic sequence
  (a₁ = 3/2 → 2 * sequence_a a₁ 5 = sequence_a a₁ 4 + sequence_a a₁ 6) ∧
  -- Part 3: Range of a₁ for increasing sequence is (0, 1)
  (∀ n : ℕ, sequence_a a₁ (n + 1) > sequence_a a₁ n ↔ 0 < a₁ ∧ a₁ < 1) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l359_35953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_expenditure_l359_35943

/-- The amount spent on oil given price reduction and quantity increase -/
noncomputable def amount_spent (price_reduction : ℝ) (quantity_increase : ℝ) (reduced_price : ℝ) : ℝ :=
  let original_price := reduced_price / (1 - price_reduction)
  quantity_increase * reduced_price * original_price / (original_price - reduced_price)

/-- Theorem stating the amount spent on oil under given conditions -/
theorem oil_expenditure :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |amount_spent 0.12 6 24 - 1188| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_expenditure_l359_35943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_attendance_l359_35944

theorem fair_attendance (P : ℝ) (h : P > 0) : 
  let projected_attendance := 1.25 * P
  let actual_attendance := 0.8 * P
  (actual_attendance / projected_attendance) * 100 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_attendance_l359_35944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_value_proof_l359_35991

def initial_mean : ℝ := 140
def correct_mean : ℝ := 140.33333333333334
def wrong_value : ℝ := 135
def total_count : ℕ := 30

theorem correct_value_proof :
  ∃ (correct_value : ℝ),
    (total_count - 1) * initial_mean + wrong_value = total_count * initial_mean ∧
    (total_count - 1) * initial_mean + correct_value = total_count * correct_mean ∧
    abs (correct_value - 145) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_value_proof_l359_35991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_eq_neg_seven_l359_35951

noncomputable def f (x : ℝ) : ℝ := x + 1/x - 2

theorem f_negative_a_eq_neg_seven (a : ℝ) (ha : a ≠ 0) (h : f a = 3) : f (-a) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_eq_neg_seven_l359_35951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_5_4_3_l359_35935

/-- The area of a triangle with sides 5, 4, and 3 units is 6 square units. -/
theorem triangle_area_5_4_3 : ∃ (A : ℝ), A > 0 ∧ A^2 = 6^2 ∧ A = Real.sqrt (6 * (6 - 5) * (6 - 4) * (6 - 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_5_4_3_l359_35935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l359_35979

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_max_sum
  (a₁ d : ℝ) (h₁ : a₁ > 0) (h₂ : sum_arithmetic_sequence a₁ d 4 = sum_arithmetic_sequence a₁ d 9) :
  ∃ (n : ℕ), n = 6 ∨ n = 7 ∧
    ∀ (k : ℕ), sum_arithmetic_sequence a₁ d n ≥ sum_arithmetic_sequence a₁ d k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l359_35979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_sum_l359_35930

theorem simplify_sqrt_sum : Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_sum_l359_35930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_b_l359_35924

/-- Given three weights a, b, and c, prove that b equals 35 when:
    1. The average of a, b, and c is 45.
    2. The average of a and b is 42.
    3. The average of b and c is 43. -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 42)
  (h3 : (b + c) / 2 = 43) : 
  b = 35 := by
  sorry

#check weight_of_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_b_l359_35924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_blue_after_operation_l359_35982

-- Define the grid
def Grid := Fin 4 → Fin 4 → Bool

-- Define the probability of a single square being blue initially
noncomputable def initial_blue_prob : ℝ := 1 / 2

-- Define the rotation operation
def rotate (g : Grid) : Grid :=
  fun i j => g (3 - j) i

-- Define the repainting operation
def repaint (g₁ g₂ : Grid) : Grid :=
  fun i j => g₂ i j || g₁ i j

-- Define the probability of the entire grid being blue after rotation and repainting
noncomputable def prob_all_blue (g : Grid) : ℝ :=
  (initial_blue_prob ^ 4) * (initial_blue_prob ^ 12)

-- State the theorem
theorem prob_all_blue_after_operation :
  ∀ g : Grid, prob_all_blue (repaint g (rotate g)) = 1 / 65536 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_blue_after_operation_l359_35982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_g_iff_a_in_range_l359_35901

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -(a + 1) / x

-- State the theorem
theorem f_greater_than_g_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 (exp 1) → f a x > g a x) ↔ 
  (-2 < a ∧ a < (exp 2 + 1) / (exp 1 - 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_g_iff_a_in_range_l359_35901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_milk_proportion_l359_35968

/-- Given that 18 cookies require 3 quarts of milk and 1 quart equals 2 pints,
    prove that 9 cookies require 3 pints of milk. -/
theorem cookies_milk_proportion :
  (∀ (cookies quarts : ℚ), cookies = 18 → quarts = 3 → cookies / 18 * 3 = quarts) →
  (∀ (quarts pints : ℚ), pints = 2 * quarts) →
  9 / 18 * 3 * 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_milk_proportion_l359_35968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_megatek_manufacturing_percentage_l359_35995

/-- Represents the number of degrees in a full circle. -/
noncomputable def full_circle : ℝ := 360

/-- Represents the number of degrees in the manufacturing department's sector. -/
noncomputable def manufacturing_sector : ℝ := 72

/-- Calculates the percentage of employees in a department given the sector's degrees. -/
noncomputable def sector_percentage (sector_degrees : ℝ) : ℝ :=
  (sector_degrees / full_circle) * 100

/-- Theorem stating that the percentage of Megatek employees in manufacturing is 20%. -/
theorem megatek_manufacturing_percentage :
  sector_percentage manufacturing_sector = 20 := by
  -- Unfold the definition of sector_percentage
  unfold sector_percentage
  -- Simplify the expression
  simp [full_circle, manufacturing_sector]
  -- Perform the calculation
  norm_num
  -- QED
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_megatek_manufacturing_percentage_l359_35995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_range_l359_35921

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - a * x else -x^2 - (a + 2) * x + 1

theorem three_zeros_implies_a_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  a > Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_range_l359_35921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l359_35945

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define the point through which the tangent line passes
def tangent_point : ℝ × ℝ := (2, 0)

-- Define a function to check if a line is tangent to the circle
def is_tangent (a b c : ℝ) : Prop :=
  ∀ x y, circle_eq x y → (a*x + b*y + c = 0 → 
    ∀ x' y', circle_eq x' y' → a*x' + b*y' + c ≥ 0)

-- Theorem statement
theorem tangent_lines_to_circle :
  (∀ a b c : ℝ, is_tangent a b c ∧ a * tangent_point.1 + b * tangent_point.2 + c = 0 →
    (a = 1 ∧ b = 0 ∧ c = -2) ∨ (a = 0 ∧ b = 1 ∧ c = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l359_35945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_correct_l359_35971

-- Define the calculations
noncomputable def calc_A : ℝ := (-3/3)^2
noncomputable def calc_B : ℝ := |8/2| - |2/2|
noncomputable def calc_C : ℝ := 2 * |1/2|
def calc_D : ℝ → Prop := λ x ↦ x = |1/(1 - 2/2)| ∧ x = 1 + |2/2|

-- State the theorem
theorem only_C_is_correct :
  calc_A ≠ 3 ∧
  calc_B ≠ |2| ∧
  calc_C = |2/2| ∧
  ¬∃x, calc_D x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_correct_l359_35971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_a_not_right_triangle_l359_35931

-- Define a function to check if three lengths can form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Define the sets of lengths
def set_a : (ℝ × ℝ × ℝ) := (4, 7, 5)
noncomputable def set_b : (ℝ × ℝ × ℝ) := (2, 3, Real.sqrt 5)
def set_c : (ℝ × ℝ × ℝ) := (5, 13, 12)
noncomputable def set_d : (ℝ × ℝ × ℝ) := (1, Real.sqrt 2, Real.sqrt 3)

-- Theorem statement
theorem only_set_a_not_right_triangle :
  ¬(is_right_triangle set_a.1 set_a.2.1 set_a.2.2) ∧
  (is_right_triangle set_b.1 set_b.2.1 set_b.2.2) ∧
  (is_right_triangle set_c.1 set_c.2.1 set_c.2.2) ∧
  (is_right_triangle set_d.1 set_d.2.1 set_d.2.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_a_not_right_triangle_l359_35931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rescue_vehicle_accessible_area_l359_35963

/-- Represents the speed of the rescue vehicle on roads in miles per hour -/
noncomputable def road_speed : ℝ := 60

/-- Represents the speed of the rescue vehicle through the desert in miles per hour -/
noncomputable def desert_speed : ℝ := 18

/-- Represents the time limit in hours -/
noncomputable def time_limit : ℝ := 1/6

/-- Represents the maximum distance the vehicle can travel on road in the given time -/
noncomputable def max_road_distance : ℝ := road_speed * time_limit

/-- Represents the radius of the circle when the vehicle starts off-road at t=0 -/
noncomputable def initial_radius : ℝ := desert_speed * time_limit

/-- Represents a circular segment formed by the intersection of a circle and a triangle -/
def circular_segment (r : ℝ) (b : ℝ) (h : ℝ) : Set (ℝ × ℝ) := sorry

/-- The area accessible by the rescue vehicle within the time limit -/
def accessible_area : Set (ℝ × ℝ) := sorry

/-- Mirror a set about the y-axis -/
def mirror_y (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {(x, y) | (-x, y) ∈ s}

/-- Mirror a set about the x-axis -/
def mirror_x (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {(x, y) | (x, -y) ∈ s}

theorem rescue_vehicle_accessible_area :
  accessible_area = 
    (Set.Icc 0 max_road_distance ×ˢ Set.Icc 0 max_road_distance) ∪
    (circular_segment initial_radius max_road_distance initial_radius) ∪
    (mirror_y (circular_segment initial_radius max_road_distance initial_radius)) ∪
    (mirror_x (circular_segment initial_radius max_road_distance initial_radius)) ∪
    (mirror_x (mirror_y (circular_segment initial_radius max_road_distance initial_radius))) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rescue_vehicle_accessible_area_l359_35963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_boys_than_girls_l359_35913

theorem more_boys_than_girls (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h1 : total_students = 100)
  (h2 : boy_ratio = 3)
  (h3 : girl_ratio = 2) :
  (total_students * boy_ratio) / (boy_ratio + girl_ratio) -
  (total_students * girl_ratio) / (boy_ratio + girl_ratio) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_boys_than_girls_l359_35913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l359_35970

/-- Represents a journey with two parts at different speeds -/
structure Journey where
  totalTime : ℝ
  speed1 : ℝ
  speed2 : ℝ

/-- Calculates the total distance of a journey -/
noncomputable def totalDistance (j : Journey) : ℝ :=
  (j.totalTime / 2) * j.speed1 + (j.totalTime / 2) * j.speed2

/-- Theorem stating that the journey with given conditions has a total distance of 336 km -/
theorem journey_distance :
  ∀ j : Journey,
  j.totalTime = 15 ∧
  j.speed1 = 21 ∧
  j.speed2 = 24 →
  totalDistance j = 336 := by
  intro j ⟨h_time, h_speed1, h_speed2⟩
  unfold totalDistance
  rw [h_time, h_speed1, h_speed2]
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l359_35970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l359_35988

noncomputable def f (y : ℝ) : ℝ := y^2 + 9*y + 81/y^3

theorem min_value_of_f :
  (∀ y : ℝ, y > 0 → f y ≥ 39) ∧
  (∃ y : ℝ, y > 0 ∧ f y = 39) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l359_35988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l359_35927

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

theorem non_monotonic_interval (k : ℝ) :
  (∀ x, x ∈ Set.Icc (k - 1) (k + 1) → x > 0) →
  (∃ x y, x ∈ Set.Icc (k - 1) (k + 1) ∧ y ∈ Set.Icc (k - 1) (k + 1) ∧ x < y ∧ f x > f y) →
  1 < k ∧ k < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l359_35927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_condition_l359_35986

-- Define the power function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 4*m + 4) * x^(m^2 - 6*m + 8)

-- Define the condition for a decreasing function on (0, +∞)
def is_decreasing_on_positive_reals (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → g y < g x

-- Theorem statement
theorem power_function_decreasing_condition (m : ℝ) :
  is_decreasing_on_positive_reals (f m) ↔ m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_condition_l359_35986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l359_35916

-- Define the curves
def C₁ (x : ℝ) : ℝ := x^2
noncomputable def C₂ (a x : ℝ) : ℝ := a * Real.exp x

-- Define the condition that a > 0
variable (a : ℝ) (h : a > 0)

-- Define the condition of common tangent
def has_common_tangent (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), (2 * x₁ = a * Real.exp x₂) ∧ (2 * x₁ = (a * Real.exp x₂ - x₁^2) / (x₂ - x₁))

-- Theorem statement
theorem range_of_a (a : ℝ) (h : a > 0) (h_tangent : has_common_tangent a) :
  a ∈ Set.Ioo 0 (4 / Real.exp 2) := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l359_35916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_gas_consumption_theorem_l359_35914

/-- Represents the gas consumption for Janet's doctor appointments -/
noncomputable def janet_gas_consumption (miles_to_dermatologist miles_to_gynecologist miles_to_cardiologist : ℝ)
  (miles_per_gallon : ℝ) (extra_gas_dermatologist extra_gas_gynecologist extra_gas_cardiologist : ℝ) : ℝ :=
  let total_miles := 2 * (miles_to_dermatologist + miles_to_gynecologist + miles_to_cardiologist)
  let normal_gas_consumption := total_miles / miles_per_gallon
  let extra_gas_consumption := extra_gas_dermatologist + extra_gas_gynecologist + extra_gas_cardiologist
  normal_gas_consumption + extra_gas_consumption

/-- Theorem stating that Janet's total gas consumption for her doctor appointments is 34.2 gallons -/
theorem janet_gas_consumption_theorem :
  janet_gas_consumption 60 80 100 15 0.5 0.7 1 = 34.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_gas_consumption_theorem_l359_35914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l359_35919

def A : Set ℤ := {x : ℤ | x^2 - 1 ≤ 0}
def B : Set ℤ := {x : ℤ | x^2 - x - 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l359_35919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_vehicles_for_zoo_trip_l359_35933

/-- Represents the types of vehicles available --/
inductive Vehicle
| Van
| Minibus

/-- Represents a person on the trip --/
inductive Person
| Student
| Adult
| AllergicStudent

/-- Represents the field trip setup --/
structure FieldTrip where
  students : Nat
  adults : Nat
  allergicStudents : Nat
  vanCapacity : Nat
  minibusCapacity : Nat

/-- Calculates the minimum number of vehicles needed for the field trip --/
def minVehiclesNeeded (trip : FieldTrip) : Nat :=
  sorry

/-- Theorem stating the minimum number of vehicles needed for the given conditions --/
theorem min_vehicles_for_zoo_trip :
  let trip : FieldTrip := {
    students := 24,
    adults := 3,
    allergicStudents := 2,
    vanCapacity := 8,
    minibusCapacity := 14
  }
  minVehiclesNeeded trip = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_vehicles_for_zoo_trip_l359_35933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_x_value_l359_35923

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, x^(-(1/2:ℝ))}
def B : Set ℝ := {0, 1, 2}

-- State the theorem
theorem subset_implies_x_value (x : ℝ) (h : A x ⊆ B) : x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_x_value_l359_35923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l359_35969

-- Define set A
def A : Set ℝ := {0, 1, 2}

-- Define set B
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l359_35969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_l359_35950

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem: Points satisfying the distance condition form an ellipse -/
theorem ellipse_trajectory (x y : ℝ) : 
  distance x y 1 0 + distance x y (-1) 0 = 4 → x^2/4 + y^2/3 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_l359_35950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_total_spend_l359_35946

/-- Represents a video recorder with its wholesale cost, markup percentage, and employee discounts -/
structure VideoRecorder where
  wholesale_cost : ℚ
  markup_percentage : ℚ
  discount_week1 : ℚ
  discount_week2 : ℚ

/-- Calculates the retail price of a video recorder -/
def retail_price (vr : VideoRecorder) : ℚ :=
  vr.wholesale_cost * (1 + vr.markup_percentage)

/-- Calculates the discounted price for an employee -/
def discounted_price (vr : VideoRecorder) : ℚ :=
  retail_price vr * (1 - max vr.discount_week1 vr.discount_week2)

/-- The three video recorders available in the store -/
def v1 : VideoRecorder := ⟨200, 1/5, 1/5, 1/4⟩
def v2 : VideoRecorder := ⟨250, 1/4, 3/20, 1/5⟩
def v3 : VideoRecorder := ⟨300, 3/10, 1/10, 3/20⟩

/-- The theorem stating the minimum total amount an employee would spend -/
theorem minimum_total_spend : 
  discounted_price v1 + discounted_price v2 + discounted_price v3 = 761.5 := by
  sorry

#eval (discounted_price v1 + discounted_price v2 + discounted_price v3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_total_spend_l359_35946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_theta_value_l359_35912

theorem sin_two_theta_value (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1/3) : 
  Real.sin (2 * θ) = -8/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_theta_value_l359_35912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_existence_l359_35975

theorem perfect_square_existence (S : Finset ℕ) (h_card : S.card = 1986) 
  (h_prime_factors : (S.prod id).factors.toFinset.card = 1985) :
  (∃ n ∈ S, ∃ m : ℕ, n = m ^ 2) ∨ 
  (∃ T : Finset ℕ, T ⊆ S ∧ T.Nonempty ∧ ∃ m : ℕ, T.prod id = m ^ 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_existence_l359_35975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_box_width_l359_35960

/-- Represents a rectangular milk box -/
structure MilkBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Converts gallons to cubic feet -/
noncomputable def gallonsToCubicFeet (gallons : ℝ) : ℝ := gallons / 7.5

/-- Calculates the volume of a rectangular box -/
noncomputable def boxVolume (box : MilkBox) : ℝ := box.length * box.width * box.height

/-- Theorem: The width of the milk box is 25 feet -/
theorem milk_box_width : 
  ∃ (box : MilkBox), 
    box.length = 56 ∧ 
    gallonsToCubicFeet 5250 = boxVolume { length := box.length, width := box.width, height := 0.5 } ∧ 
    box.width = 25 := by
  sorry

#check milk_box_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_box_width_l359_35960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l359_35976

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (3, 0)

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The main theorem -/
theorem distance_right_focus_to_line :
  distance_point_to_line (right_focus.1) (right_focus.2) 1 2 (-8) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l359_35976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_driving_time_at_max_speeds_l359_35965

/-- Represents the driving speeds for each segment of the journey -/
structure DrivingSpeeds where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total driving time given the driving speeds -/
noncomputable def totalDrivingTime (speeds : DrivingSpeeds) : ℝ :=
  120 / 40 + 60 / speeds.x + 90 / speeds.y + 200 / speeds.z

/-- Theorem stating that the minimum driving time is achieved at maximum allowed speeds -/
theorem min_driving_time_at_max_speeds
  (speeds : DrivingSpeeds)
  (hx : 30 ≤ speeds.x ∧ speeds.x ≤ 60)
  (hy : 40 ≤ speeds.y ∧ speeds.y ≤ 70)
  (hz : 50 ≤ speeds.z ∧ speeds.z ≤ 90)
  (ht : totalDrivingTime speeds ≤ 10) :
  totalDrivingTime speeds ≥ totalDrivingTime { x := 60, y := 70, z := 90 } := by
  sorry

#check min_driving_time_at_max_speeds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_driving_time_at_max_speeds_l359_35965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_properties_l359_35925

noncomputable def arithmeticProgression (a d : ℝ) (n : ℕ) : ℕ → ℝ
  | 0 => a
  | k + 1 => arithmeticProgression a d n k + d

noncomputable def sumFirstTerms (a d : ℝ) (n k : ℕ) : ℝ :=
  (k : ℝ) / 2 * (2 * a + (k - 1) * d)

noncomputable def sumLastTerms (a d : ℝ) (n k : ℕ) : ℝ :=
  (k : ℝ) / 2 * (2 * (a + (n - k) * d) + (k - 1) * d)

noncomputable def sumExceptFirstM (a d : ℝ) (n m : ℕ) : ℝ :=
  ((n - m) : ℝ) / 2 * (2 * (a + m * d) + (n - m - 1) * d)

noncomputable def sumExceptLastM (a d : ℝ) (n m : ℕ) : ℝ :=
  ((n - m) : ℝ) / 2 * (2 * a + (n - m - 1) * d)

theorem arithmetic_progression_properties (a d : ℝ) (n : ℕ) :
  (sumFirstTerms a d n 13 = 0.5 * sumLastTerms a d n 13) ∧
  (sumExceptFirstM a d n 3 / sumExceptLastM a d n 3 = 6 / 5) →
  n = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_properties_l359_35925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_representable_amount_l359_35977

/-- Represents the coin denominations in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |> List.map (fun i => 3^(n - i) * 4^i)

/-- Checks if an amount can be represented using given coin denominations -/
def is_representable (amount : ℕ) (coins : List ℕ) : Prop :=
  ∃ (coeffs : List ℕ), amount = List.sum (List.zipWith (· * ·) coeffs coins)

/-- The main theorem about the largest non-representable amount -/
theorem largest_non_representable_amount (n : ℕ) :
  ¬ is_representable (2 * 4^(n+1) - 3^(n+2)) (coin_denominations n) ∧
  ∀ m : ℕ, m > 2 * 4^(n+1) - 3^(n+2) → is_representable m (coin_denominations n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_representable_amount_l359_35977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_zeros_range_of_x_for_inequality_l359_35993

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then x else x^3 - 3*x

-- Theorem for the first part of the problem
theorem range_of_a_for_two_zeros (a : ℝ) :
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z, f a z = 0 → z = x ∨ z = y) ↔
  -Real.sqrt 3 < a ∧ a ≤ Real.sqrt 3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_x_for_inequality (a : ℝ) (h : a ≤ -2) :
  ∀ x, f a x + f a (x - 1) > -3 ↔ x > -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_zeros_range_of_x_for_inequality_l359_35993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trigonometric_expression_l359_35915

theorem simplify_trigonometric_expression : 
  1 / Real.sqrt (1 + Real.tan (160 * Real.pi / 180)) ^ 2 = - Real.cos (160 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trigonometric_expression_l359_35915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_digit_at_position_l359_35910

/-- The decimal representation of 5/13 -/
def decimal_rep : ℚ := 5 / 13

/-- The length of the repeating sequence in the decimal representation of 5/13 -/
def repeat_length : ℕ := 6

/-- The repeating sequence in the decimal representation of 5/13 -/
def repeat_seq : List ℕ := [3, 8, 4, 6, 1, 5]

/-- The position we're interested in -/
def position : ℕ := 534

theorem decimal_digit_at_position :
  (repeat_seq.get! ((position - 1) % repeat_length)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_digit_at_position_l359_35910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l359_35980

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  h_angles : A + B + C = Real.pi
  h_sides : (Real.cos A - 2 * Real.cos B) / Real.cos C = (2 * b - a) / c

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h_cosC : Real.cos t.C = 1/4) (h_c : t.c = 2) : 
  (Real.sin t.B / Real.sin t.A = 2) ∧ 
  (1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 15 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l359_35980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l359_35926

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := ((2 * x + 6) / 5) ^ (1/4)

-- State the theorem
theorem function_equality : 
  ∀ x : ℝ, f (3 * x) = 3 * f x ↔ x = -40/13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l359_35926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_is_65_l359_35928

-- Define the original price and reduced price
noncomputable def original_price : ℝ := sorry
noncomputable def reduced_price : ℝ := sorry

-- Define the reduction percentage
def reduction_percentage : ℝ := 0.25

-- Define the additional quantity obtained after reduction
def additional_quantity : ℝ := 5

-- Define the total amount spent
def total_amount : ℝ := 1300

-- Define the relationship between original and reduced price
axiom price_reduction : reduced_price = original_price * (1 - reduction_percentage)

-- Define the relationship between prices and quantities
axiom quantity_increase : total_amount = original_price * (total_amount / original_price) ∧
                          total_amount = reduced_price * (total_amount / original_price + additional_quantity)

-- State the theorem
theorem reduced_price_is_65 : ∃ ε > 0, |reduced_price - 65| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_is_65_l359_35928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l359_35911

-- Define the logarithms
noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 8 / Real.log 4
noncomputable def c : ℝ := Real.log 10 / Real.log 5

-- Theorem statement
theorem log_inequality : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l359_35911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l359_35938

noncomputable def f (x y : ℝ) : ℝ := Real.sqrt (y - Real.sqrt (8 - Real.sqrt x))

theorem domain_of_f :
  {p : ℝ × ℝ | ∃ z, f p.1 p.2 = z} = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 64 ∧ 0 ≤ p.2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l359_35938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_identity_l359_35947

/-- A type representing a line in a plane -/
structure Line where
  -- We'll use a simple representation for now
  a : ℝ
  b : ℝ
  c : ℝ

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across a line -/
def reflect (p : Point) (l : Line) : Point :=
  sorry

/-- Represents the sequence of reflections across the given lines -/
def reflectSequence (lines : List Line) (p : Point) : Point :=
  sorry

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem reflection_identity 
  (n : ℕ) 
  (lines : List Line) 
  (h1 : lines.length = 2 * n) 
  (h2 : ∀ (l1 l2 : Line), l1 ∈ lines → l2 ∈ lines → ∃ (p : Point), pointOnLine p l1 ∧ pointOnLine p l2)
  (M : Point)
  (h3 : reflectSequence lines M = M) :
  ∀ (p : Point), reflectSequence lines p = p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_identity_l359_35947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_angle_BXY_l359_35932

-- Define the points
variable (A B C D X Y E : EuclideanSpace ℝ (Fin 2))

-- Define the angles
noncomputable def angle_AXE : ℝ := sorry
noncomputable def angle_CYX : ℝ := sorry
noncomputable def angle_BXY : ℝ := sorry

-- Define the parallel lines condition
def AB_parallel_CD : Prop := (B - A).Orthogonal ((C - D).Orthogonal (B - A))

-- Define the angle relation condition
def angle_relation : Prop := angle_AXE = 4 * angle_CYX - 120

-- Theorem statement
theorem find_angle_BXY (h1 : AB_parallel_CD) (h2 : angle_relation) : angle_BXY = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_angle_BXY_l359_35932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_arrangements_eq_3456_l359_35952

/-- Represents a seating arrangement for siblings in a bus -/
structure SeatingArrangement where
  rows : Fin 2 → Fin 4 → Fin 8
  no_adjacent_siblings : ∀ (i : Fin 2) (j : Fin 3), rows i j ≠ rows i (j.succ) + 4
  no_front_back_siblings : ∀ (j : Fin 4), rows 0 j ≠ rows 1 j + 4

/-- The number of permissible seating arrangements -/
def num_arrangements : ℕ := sorry

/-- Theorem stating the number of permissible seating arrangements -/
theorem num_arrangements_eq_3456 : num_arrangements = 3456 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_arrangements_eq_3456_l359_35952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sum_l359_35949

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 3 * n - 1

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℝ := (-1)^n * a n + 2^(n + 1)

-- Define the sum of the first n terms of a_n
noncomputable def S (n : ℕ) : ℝ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

-- Define the sum of the first 2n terms of b_n
noncomputable def T (n : ℕ) : ℝ := 3 * n + 4^(n + 1) - 4

theorem arithmetic_sequence_and_sum :
  a 3 = 8 ∧ S 5 = 2 * a 7 ∧
  (∀ n : ℕ, a n = 3 * n - 1) ∧
  (∀ n : ℕ, T n = 3 * n + 4^(n + 1) - 4) := by
  sorry

#check arithmetic_sequence_and_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sum_l359_35949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_zeros_a_range_l359_35967

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x - a

theorem f_three_zeros_a_range (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  a > -9 ∧ a < 5/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_zeros_a_range_l359_35967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_properties_l359_35989

def Z (a b : ℕ) : ℚ := (Nat.factorial (3 * a) * Nat.factorial (4 * b)) / 
  ((Nat.factorial a)^4 * (Nat.factorial b)^3)

theorem Z_properties :
  (∀ a b : ℕ, a ≤ b → (Z a b).num % (Z a b).den = 0) ∧
  (∀ b : ℕ, ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ a ∈ S, (Z a b).num % (Z a b).den ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_properties_l359_35989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_cosine_value_l359_35961

open Real

-- Define a triangle ABC
variable (A B C O : EuclideanSpace ℝ (Fin 2))

-- Define that O is the circumcenter of triangle ABC
def is_circumcenter (O A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the vector equality condition
def vector_condition (O A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (O - A) = (1/3 : ℝ) • ((B - A) + (C - A))

-- Define the angle measure
noncomputable def angle_BAC (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem circumcenter_cosine_value
  (h1 : is_circumcenter O A B C)
  (h2 : vector_condition O A B C) :
  cos (angle_BAC A B C) = 1/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_cosine_value_l359_35961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_max_value_of_M_l359_35904

-- Part I
noncomputable def f (x : ℝ) := Real.exp x - x + 1

theorem range_of_f : 
  ∃ (y : ℝ), y ∈ Set.Icc 2 (Real.exp 2 - 1) ↔ ∃ (x : ℝ), x ∈ Set.Icc (-1) 2 ∧ f x = y :=
sorry

-- Part II
noncomputable def g (a b x : ℝ) := Real.exp x - a * x + b

def M (a b : ℝ) := a - b

theorem max_value_of_M (a b : ℝ) :
  (∀ x : ℝ, g a b x ≥ 0) → M a b ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_max_value_of_M_l359_35904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_difference_l359_35908

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The difference between 54210 (base 8) and 43210 (base 9) in base 10 --/
theorem base_difference : 
  (to_base_10 [0, 1, 2, 4, 5] 8 : Int) - (to_base_10 [0, 1, 2, 3, 4] 9 : Int) = -5938 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_difference_l359_35908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_count_10_l359_35966

/-- A function that counts the number of triangles with integer side lengths not exceeding a given maximum value. -/
def countTriangles (maxSide : ℕ) : ℕ :=
  let countForSide (a : ℕ) : ℕ :=
    if a ≤ maxSide then
      let halfCeil := (a + 1) / 2
      (a - halfCeil) * (halfCeil + 1)
    else 0
  (List.range (maxSide + 1)).map countForSide |>.sum

/-- Theorem stating that the number of triangles with integer side lengths not exceeding 10 is 125. -/
theorem triangles_count_10 : countTriangles 10 = 125 := by
  sorry

#eval countTriangles 10  -- Should output 125

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_count_10_l359_35966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_h_l359_35994

/-- Given that h(3y + 2) = 2y + 7 for all y, prove that 17 is the unique solution to h(x) = x -/
theorem fixed_point_of_h (h : ℝ → ℝ) (h_def : ∀ y, h (3*y + 2) = 2*y + 7) :
  (∃! x, h x = x) ∧ (h 17 = 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_h_l359_35994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_five_digit_numbers_l359_35992

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ n % 100 = 0

def uses_distinct_digits (a b : Nat) : Prop :=
  let digits := (Nat.repr a ++ Nat.repr b).data
  digits.eraseDups.length = 10 ∧ digits.all (λ d => '0' ≤ d ∧ d ≤ '9')

theorem max_sum_five_digit_numbers :
  ∃ (a b : Nat), is_valid_number a ∧ is_valid_number b ∧
  uses_distinct_digits a b ∧
  (∀ (x y : Nat), is_valid_number x ∧ is_valid_number y ∧ uses_distinct_digits x y →
    x + y ≤ a + b) ∧
  (a = 96400 ∨ a = 87500 ∨ b = 96400 ∨ b = 87500) := by
  sorry

#check max_sum_five_digit_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_five_digit_numbers_l359_35992
