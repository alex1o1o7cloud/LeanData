import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_expansion_l2231_223107

theorem polynomial_expansion (x : ℝ) : 
  (7 * x + 3) * (5 * x^2 + 2 * x + 4) = 35 * x^3 + 29 * x^2 + 34 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2231_223107


namespace NUMINAMATH_CALUDE_system_solution_l2231_223123

theorem system_solution (x y : ℚ) (h1 : x + 2*y = -1) (h2 : 2*x + y = 3) : x + y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2231_223123


namespace NUMINAMATH_CALUDE_total_original_cost_of_cars_l2231_223137

/-- Calculates the original price of a car before depreciation -/
def originalPrice (soldPrice : ℚ) (depreciationRate : ℚ) : ℚ :=
  soldPrice / (1 - depreciationRate)

/-- Proves that the total original cost of two cars is $3058.82 -/
theorem total_original_cost_of_cars 
  (oldCarSoldPrice : ℚ) 
  (secondOldestCarSoldPrice : ℚ) 
  (oldCarDepreciationRate : ℚ) 
  (secondOldestCarDepreciationRate : ℚ)
  (h1 : oldCarSoldPrice = 1800)
  (h2 : secondOldestCarSoldPrice = 900)
  (h3 : oldCarDepreciationRate = 1/10)
  (h4 : secondOldestCarDepreciationRate = 3/20) :
  originalPrice oldCarSoldPrice oldCarDepreciationRate + 
  originalPrice secondOldestCarSoldPrice secondOldestCarDepreciationRate = 3058.82 := by
  sorry

#eval originalPrice 1800 (1/10) + originalPrice 900 (3/20)

end NUMINAMATH_CALUDE_total_original_cost_of_cars_l2231_223137


namespace NUMINAMATH_CALUDE_expression_simplification_l2231_223162

theorem expression_simplification (b : ℝ) :
  ((3 * b - 3) - 5 * b) / 3 = -2/3 * b - 1 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2231_223162


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2231_223151

theorem cube_root_equation_solution :
  ∃ y : ℝ, (30 * y + (30 * y + 24) ^ (1/3)) ^ (1/3) = 24 ∧ y = 460 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2231_223151


namespace NUMINAMATH_CALUDE_non_officers_count_correct_l2231_223191

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := 450

/-- Represents the number of officers in an office -/
def num_officers : ℕ := 15

/-- Average salary of all employees in rupees per month -/
def avg_salary_all : ℚ := 120

/-- Average salary of officers in rupees per month -/
def avg_salary_officers : ℚ := 420

/-- Average salary of non-officers in rupees per month -/
def avg_salary_non_officers : ℚ := 110

/-- Theorem stating that the number of non-officers is correct given the salary information -/
theorem non_officers_count_correct :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / 
  (num_officers + num_non_officers : ℚ) = avg_salary_all := by
  sorry


end NUMINAMATH_CALUDE_non_officers_count_correct_l2231_223191


namespace NUMINAMATH_CALUDE_pond_volume_l2231_223197

/-- The volume of a rectangular prism given its length, width, and height -/
def volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a rectangular prism with dimensions 20 m × 10 m × 8 m is 1600 cubic meters -/
theorem pond_volume : volume 20 10 8 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_l2231_223197


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2231_223181

theorem rectangular_prism_diagonal (width length height : ℝ) 
  (hw : width = 12) (hl : length = 16) (hh : height = 9) : 
  Real.sqrt (width^2 + length^2 + height^2) = Real.sqrt 481 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2231_223181


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2231_223196

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → 
  x₂^2 - 3*x₂ - 1 = 0 → 
  x₁^2 + x₂^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2231_223196


namespace NUMINAMATH_CALUDE_extended_altitude_triangle_l2231_223134

/-- Given a triangle ABC with sides a, b, c, angles α, β, γ, and area t,
    we extend its altitudes beyond the sides by their own lengths to form
    a new triangle A'B'C' with sides a', b', c' and area t'. -/
theorem extended_altitude_triangle
  (a b c a' b' c' : ℝ)
  (α β γ : ℝ)
  (t t' : ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧ β > 0 ∧ γ > 0)
  (h_angles : α + β + γ = Real.pi)
  (h_area : t = (1/2) * a * b * Real.sin γ)
  (h_extended : a' > a ∧ b' > b ∧ c' > c) :
  (a'^2 + b'^2 + c'^2 - (a^2 + b^2 + c^2) = 32 * t * Real.sin α * Real.sin β * Real.sin γ) ∧
  (t' = t * (3 + 8 * Real.cos α * Real.cos β * Real.cos γ)) := by
  sorry

end NUMINAMATH_CALUDE_extended_altitude_triangle_l2231_223134


namespace NUMINAMATH_CALUDE_ball_distribution_problem_l2231_223174

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute n identical balls into 3 distinct boxes,
    where box i must contain at least i balls (for i = 1, 2, 3) --/
def distributeWithMinimum (n : ℕ) : ℕ :=
  distribute (n - (1 + 2 + 3)) 3

theorem ball_distribution_problem :
  distributeWithMinimum 10 = 15 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_problem_l2231_223174


namespace NUMINAMATH_CALUDE_max_percentage_both_amenities_l2231_223163

/-- Represents the percentage of companies with type A planes -/
def percentage_type_A : ℝ := 0.4

/-- Represents the percentage of companies with type B planes -/
def percentage_type_B : ℝ := 0.6

/-- Represents the percentage of type A planes with wireless internet -/
def wireless_A : ℝ := 0.8

/-- Represents the percentage of type B planes with wireless internet -/
def wireless_B : ℝ := 0.1

/-- Represents the percentage of type A planes offering free snacks -/
def snacks_A : ℝ := 0.9

/-- Represents the percentage of type B planes offering free snacks -/
def snacks_B : ℝ := 0.5

/-- Theorem stating the maximum percentage of companies offering both amenities -/
theorem max_percentage_both_amenities :
  let max_both_A := min wireless_A snacks_A
  let max_both_B := min wireless_B snacks_B
  let max_percentage := percentage_type_A * max_both_A + percentage_type_B * max_both_B
  max_percentage = 0.38 := by sorry

end NUMINAMATH_CALUDE_max_percentage_both_amenities_l2231_223163


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2231_223177

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 / 9 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := 2*x + 3*y = 0 ∨ 2*x - 3*y = 0

-- Theorem statement
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2231_223177


namespace NUMINAMATH_CALUDE_basketball_percentage_l2231_223102

theorem basketball_percentage (total_students : ℕ) (chess_percent : ℚ) (chess_or_basketball : ℕ) : 
  total_students = 250 →
  chess_percent = 1/10 →
  chess_or_basketball = 125 →
  ∃ (basketball_percent : ℚ), 
    basketball_percent = 2/5 ∧ 
    (basketball_percent + chess_percent) * total_students = chess_or_basketball :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_percentage_l2231_223102


namespace NUMINAMATH_CALUDE_inverse_function_sum_l2231_223195

-- Define the function g and its inverse
def g (c d : ℝ) (x : ℝ) : ℝ := c * x + d
def g_inv (c d : ℝ) (x : ℝ) : ℝ := d * x + c

-- State the theorem
theorem inverse_function_sum (c d : ℝ) : 
  (∀ x : ℝ, g c d (g_inv c d x) = x) ∧ 
  (∀ x : ℝ, g_inv c d (g c d x) = x) → 
  c + d = -2 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_sum_l2231_223195


namespace NUMINAMATH_CALUDE_product_of_solutions_l2231_223144

theorem product_of_solutions (x : ℝ) : 
  (|5 * x| + 7 = 47) → (∃ y : ℝ, |5 * y| + 7 = 47 ∧ x * y = -64) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2231_223144


namespace NUMINAMATH_CALUDE_max_chocolates_ben_l2231_223118

theorem max_chocolates_ben (total : ℕ) (ben carol : ℕ) (k : ℕ) : 
  total = 30 →
  ben + carol = total →
  carol = k * ben →
  k > 0 →
  ben ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_max_chocolates_ben_l2231_223118


namespace NUMINAMATH_CALUDE_S_is_bounded_region_l2231_223172

/-- The set S of points (x,y) in the coordinate plane where one of 5, x+1, and y-5 is greater than or equal to the other two -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (5 ≥ p.1 + 1 ∧ 5 ≥ p.2 - 5) ∨
               (p.1 + 1 ≥ 5 ∧ p.1 + 1 ≥ p.2 - 5) ∨
               (p.2 - 5 ≥ 5 ∧ p.2 - 5 ≥ p.1 + 1)}

/-- S is a single bounded region in the quadrant -/
theorem S_is_bounded_region : 
  ∃ (a b c d : ℝ), a < b ∧ c < d ∧
  S = {p : ℝ × ℝ | a ≤ p.1 ∧ p.1 ≤ b ∧ c ≤ p.2 ∧ p.2 ≤ d} :=
by
  sorry

end NUMINAMATH_CALUDE_S_is_bounded_region_l2231_223172


namespace NUMINAMATH_CALUDE_solution_value_l2231_223150

theorem solution_value (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 3/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2231_223150


namespace NUMINAMATH_CALUDE_truck_fuel_relationship_l2231_223112

/-- Represents the fuel consumption model of a truck -/
structure TruckFuelModel where
  tankCapacity : ℝ
  fuelConsumptionRate : ℝ

/-- Calculates the remaining fuel in the tank after a given time -/
def remainingFuel (model : TruckFuelModel) (time : ℝ) : ℝ :=
  model.tankCapacity - model.fuelConsumptionRate * time

/-- Theorem: The relationship between remaining fuel and traveling time for the given truck -/
theorem truck_fuel_relationship (model : TruckFuelModel) 
  (h1 : model.tankCapacity = 60)
  (h2 : model.fuelConsumptionRate = 8) :
  ∀ t : ℝ, remainingFuel model t = 60 - 8 * t :=
by sorry

end NUMINAMATH_CALUDE_truck_fuel_relationship_l2231_223112


namespace NUMINAMATH_CALUDE_buses_per_week_is_165_l2231_223117

/-- Calculates the number of buses leaving a station in a week -/
def total_buses_per_week (
  weekday_interval : ℕ
  ) (weekday_hours : ℕ
  ) (weekday_count : ℕ
  ) (weekend_interval : ℕ
  ) (weekend_hours : ℕ
  ) (weekend_count : ℕ
  ) : ℕ :=
  let weekday_buses := weekday_count * (weekday_hours * 60 / weekday_interval)
  let weekend_buses := weekend_count * (weekend_hours * 60 / weekend_interval)
  weekday_buses + weekend_buses

/-- Theorem stating that the total number of buses leaving the station in a week is 165 -/
theorem buses_per_week_is_165 :
  total_buses_per_week 40 14 5 20 10 2 = 165 := by
  sorry


end NUMINAMATH_CALUDE_buses_per_week_is_165_l2231_223117


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l2231_223152

theorem difference_of_squares_factorization (y : ℝ) :
  49 - 16 * y^2 = (7 - 4*y) * (7 + 4*y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l2231_223152


namespace NUMINAMATH_CALUDE_trig_expression_value_l2231_223103

theorem trig_expression_value (x : Real) (h : Real.tan x = -1/2) :
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l2231_223103


namespace NUMINAMATH_CALUDE_function_properties_l2231_223135

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + Real.log (2 - x) + a * x

-- State the theorem
theorem function_properties (a : ℝ) (h : a > 0) :
  -- 1. Domain of f(x) is (0, 2)
  (∀ x, f a x ≠ Real.log 0 → 0 < x ∧ x < 2) ∧
  -- 2. When a = 1, f(x) is increasing on (0, √2) and decreasing on (√2, 2)
  (a = 1 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.sqrt 2 → f 1 x₁ < f 1 x₂) ∧
    (∀ x₁ x₂, Real.sqrt 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f 1 x₁ > f 1 x₂)) ∧
  -- 3. If the maximum value of f(x) on (0, 1] is 1/2, then a = 1/2
  ((∃ x, 0 < x ∧ x ≤ 1 ∧ f a x = 1/2 ∧ ∀ y, 0 < y ∧ y ≤ 1 → f a y ≤ 1/2) → a = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2231_223135


namespace NUMINAMATH_CALUDE_fraction_sum_l2231_223160

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 4) : (a + b) / b = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2231_223160


namespace NUMINAMATH_CALUDE_river_name_proof_l2231_223178

theorem river_name_proof :
  ∃! (x y z : ℕ),
    x + y + z = 35 ∧
    x - y = y - (z + 1) ∧
    (x + 3) * z = y^2 ∧
    x = 5 ∧ y = 12 ∧ z = 18 := by
  sorry

end NUMINAMATH_CALUDE_river_name_proof_l2231_223178


namespace NUMINAMATH_CALUDE_vector_coordinates_proof_l2231_223192

/-- Given points A, B, C in a 2D plane, and points M and N satisfying certain conditions,
    prove that M, N, and vector MN have specific coordinates. -/
theorem vector_coordinates_proof (A B C M N : ℝ × ℝ) : 
  A = (-2, 4) → 
  B = (3, -1) → 
  C = (-3, -4) → 
  M - C = 3 • (A - C) → 
  N - C = 2 • (B - C) → 
  M = (0, 20) ∧ 
  N = (9, 2) ∧ 
  N - M = (9, -18) := by
  sorry

end NUMINAMATH_CALUDE_vector_coordinates_proof_l2231_223192


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2231_223155

/-- The perimeter of a trapezoid JKLM with given coordinates is 36 units -/
theorem trapezoid_perimeter : 
  let J : ℝ × ℝ := (-2, -4)
  let K : ℝ × ℝ := (-2, 2)
  let L : ℝ × ℝ := (6, 8)
  let M : ℝ × ℝ := (6, -4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist J K + dist K L + dist L M + dist M J = 36 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2231_223155


namespace NUMINAMATH_CALUDE_basketball_games_l2231_223114

theorem basketball_games (c : ℕ) : 
  (3 * c / 4 : ℚ) = (7 * c / 10 : ℚ) - 5 ∧ 
  (c / 4 : ℚ) = (3 * c / 10 : ℚ) - 5 → 
  c = 100 := by
  sorry

end NUMINAMATH_CALUDE_basketball_games_l2231_223114


namespace NUMINAMATH_CALUDE_constant_function_proof_l2231_223147

theorem constant_function_proof (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 + x) = 2 - f x) 
  (h2 : ∀ x, f (x + 3) ≥ f x) : 
  ∀ x, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l2231_223147


namespace NUMINAMATH_CALUDE_exists_even_non_zero_from_step_two_l2231_223115

/-- Represents the state of the sequence at a given step -/
def SequenceState := ℤ → ℤ

/-- The initial state of the sequence -/
def initial_state : SequenceState :=
  fun i => if i = 0 then 1 else 0

/-- Updates the sequence for one step -/
def update_sequence (s : SequenceState) : SequenceState :=
  fun i => s (i - 1) + s i + s (i + 1)

/-- Checks if a number is even and non-zero -/
def is_even_non_zero (n : ℤ) : Prop :=
  n ≠ 0 ∧ n % 2 = 0

/-- The sequence after n steps -/
def sequence_at_step (n : ℕ) : SequenceState :=
  match n with
  | 0 => initial_state
  | n + 1 => update_sequence (sequence_at_step n)

/-- The main theorem to be proved -/
theorem exists_even_non_zero_from_step_two (n : ℕ) (h : n ≥ 2) :
  ∃ i : ℤ, is_even_non_zero ((sequence_at_step n) i) :=
sorry

end NUMINAMATH_CALUDE_exists_even_non_zero_from_step_two_l2231_223115


namespace NUMINAMATH_CALUDE_solution_set_length_l2231_223124

theorem solution_set_length (a : ℝ) (h1 : a > 0) : 
  (∃ x1 x2 : ℝ, x1 < x2 ∧ 
    (∀ x : ℝ, x1 ≤ x ∧ x ≤ x2 ↔ Real.sqrt (x + a) + Real.sqrt (x - a) ≤ Real.sqrt (2 * (x + 1))) ∧
    x2 - x1 = 1/2) →
  a = 3/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_length_l2231_223124


namespace NUMINAMATH_CALUDE_function_symmetry_l2231_223156

/-- The function f(x) = 2sin(4x + π/4) is symmetric about the point (-π/16, 0) -/
theorem function_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (4 * x + π / 4)
  ∀ y : ℝ, f ((-π/16) + y) = f ((-π/16) - y) :=
by sorry

end NUMINAMATH_CALUDE_function_symmetry_l2231_223156


namespace NUMINAMATH_CALUDE_salt_solution_problem_l2231_223126

theorem salt_solution_problem (x : ℝ) : 
  x > 0 →  -- Ensure x is positive
  let initial_salt := 0.2 * x
  let after_evaporation := 0.75 * x
  let final_volume := after_evaporation + 7 + 14
  let final_salt := initial_salt + 14
  (final_salt / final_volume = 1/3) →
  x = 140 :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_problem_l2231_223126


namespace NUMINAMATH_CALUDE_target_breaking_orders_l2231_223186

theorem target_breaking_orders : 
  (Nat.factorial 8) / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 2) = 560 := by
  sorry

end NUMINAMATH_CALUDE_target_breaking_orders_l2231_223186


namespace NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l2231_223189

/-- Given a point P(3, -4), prove that its symmetric point P' about the x-axis has coordinates (3, 4) -/
theorem symmetric_point_about_x_axis :
  let P : ℝ × ℝ := (3, -4)
  let P' : ℝ × ℝ := (P.1, -P.2)
  P' = (3, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l2231_223189


namespace NUMINAMATH_CALUDE_apex_distance_theorem_l2231_223111

/-- Represents a right octagonal pyramid with two parallel cross sections -/
structure RightOctagonalPyramid where
  small_area : ℝ
  large_area : ℝ
  plane_distance : ℝ

/-- The distance from the apex to the plane of the larger cross section -/
def apex_to_large_section (p : RightOctagonalPyramid) : ℝ :=
  36 -- We define this as 36 based on the problem statement

/-- Theorem stating the relationship between the pyramid's properties and the apex distance -/
theorem apex_distance_theorem (p : RightOctagonalPyramid) 
  (h1 : p.small_area = 256 * Real.sqrt 2)
  (h2 : p.large_area = 576 * Real.sqrt 2)
  (h3 : p.plane_distance = 12) :
  apex_to_large_section p = 36 := by
  sorry

#check apex_distance_theorem

end NUMINAMATH_CALUDE_apex_distance_theorem_l2231_223111


namespace NUMINAMATH_CALUDE_women_per_table_women_per_table_solution_l2231_223166

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) : ℕ :=
  let total_men := num_tables * men_per_table
  let total_women := total_customers - total_men
  total_women / num_tables

theorem women_per_table_solution :
  women_per_table 9 3 90 = 7 := by
  sorry

end NUMINAMATH_CALUDE_women_per_table_women_per_table_solution_l2231_223166


namespace NUMINAMATH_CALUDE_expression_simplification_l2231_223138

theorem expression_simplification :
  let x : ℝ := Real.sqrt 2 + 1
  let expr := ((2 * x - 1) / (x + 1) - x + 1) / ((x - 2) / (x^2 + 2*x + 1))
  expr = -12 * Real.sqrt 2 - 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2231_223138


namespace NUMINAMATH_CALUDE_set_union_problem_l2231_223136

-- Define the sets A and B as functions of x
def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

-- State the theorem
theorem set_union_problem (x : ℝ) :
  (A x ∩ B x = {9}) →
  (∃ y, A y ∪ B y = {-8, -7, -4, 4, 9}) :=
by sorry

end NUMINAMATH_CALUDE_set_union_problem_l2231_223136


namespace NUMINAMATH_CALUDE_square_property_l2231_223185

theorem square_property (n : ℕ) :
  (∃ (d : Finset ℕ), d.card = 6 ∧ ∀ x ∈ d, x ∣ (n^5 + n^4 + 1)) →
  ∃ k : ℕ, n^3 - n + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_property_l2231_223185


namespace NUMINAMATH_CALUDE_selling_price_for_loss_is_40_l2231_223132

/-- The selling price that yields the same loss as the profit for an article -/
def selling_price_for_loss (cost_price : ℕ) (profit_selling_price : ℕ) : ℕ :=
  cost_price - (profit_selling_price - cost_price)

/-- Proof that the selling price for loss is 40 given the conditions -/
theorem selling_price_for_loss_is_40 :
  selling_price_for_loss 47 54 = 40 := by
  sorry

#eval selling_price_for_loss 47 54

end NUMINAMATH_CALUDE_selling_price_for_loss_is_40_l2231_223132


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l2231_223128

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l2231_223128


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2231_223154

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 3.17171717 ∧ 
  n + d = 413 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2231_223154


namespace NUMINAMATH_CALUDE_cow_horse_ratio_l2231_223171

theorem cow_horse_ratio (total : ℕ) (cows : ℕ) (horses : ℕ) (h1 : total = 168) (h2 : cows = 140) (h3 : total = cows + horses) (h4 : ∃ r : ℕ, cows = r * horses) : 
  cows / horses = 5 := by
  sorry

end NUMINAMATH_CALUDE_cow_horse_ratio_l2231_223171


namespace NUMINAMATH_CALUDE_even_sum_probability_l2231_223113

/-- The set of the first twenty prime numbers -/
def first_twenty_primes : Finset ℕ := sorry

/-- The number of ways to select 6 numbers from a set of 20 -/
def total_selections : ℕ := Nat.choose 20 6

/-- The number of ways to select 6 odd numbers from the set of odd primes in first_twenty_primes -/
def odd_selections : ℕ := Nat.choose 19 6

/-- The probability of selecting six prime numbers from first_twenty_primes such that their sum is even -/
def prob_even_sum : ℚ := odd_selections / total_selections

theorem even_sum_probability : prob_even_sum = 354 / 505 := by sorry

end NUMINAMATH_CALUDE_even_sum_probability_l2231_223113


namespace NUMINAMATH_CALUDE_path_area_calculation_l2231_223148

/-- Calculates the area of a path surrounding a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

theorem path_area_calculation (field_length field_width path_width : ℝ)
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 3.5) :
  path_area field_length field_width path_width = 959 := by
  sorry

#eval path_area 75 55 3.5

end NUMINAMATH_CALUDE_path_area_calculation_l2231_223148


namespace NUMINAMATH_CALUDE_min_workers_theorem_l2231_223141

-- Define the problem parameters
def total_days : ℕ := 40
def days_worked : ℕ := 10
def initial_workers : ℕ := 10
def work_completed : ℚ := 1/4

-- Define the function to calculate the minimum number of workers
def min_workers_needed (total_days : ℕ) (days_worked : ℕ) (initial_workers : ℕ) (work_completed : ℚ) : ℕ :=
  -- Implementation details are not provided in the statement
  sorry

-- Theorem statement
theorem min_workers_theorem :
  min_workers_needed total_days days_worked initial_workers work_completed = 10 :=
by sorry

end NUMINAMATH_CALUDE_min_workers_theorem_l2231_223141


namespace NUMINAMATH_CALUDE_angle_double_quadrant_l2231_223146

/-- Given that α is an angle in the second quadrant, prove that 2α is an angle in the third or fourth quadrant. -/
theorem angle_double_quadrant (α : Real) (h : π/2 < α ∧ α < π) :
  π < 2*α ∧ 2*α < 2*π :=
by sorry

end NUMINAMATH_CALUDE_angle_double_quadrant_l2231_223146


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l2231_223167

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : 7 < x)
  (h3 : 13 > x ∧ x > 2)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) :
  ∀ y : ℤ, x < y → 12 ≤ y := by
  sorry

#check smallest_upper_bound

end NUMINAMATH_CALUDE_smallest_upper_bound_l2231_223167


namespace NUMINAMATH_CALUDE_sum_of_456_terms_l2231_223180

/-- An arithmetic progression with first term 2 and sum of second and third terms 13 -/
def ArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, 
    (a 1 = 2) ∧ 
    (a 2 + a 3 = 13) ∧ 
    ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 4th, 5th, and 6th terms of the arithmetic progression is 42 -/
theorem sum_of_456_terms (a : ℕ → ℝ) (h : ArithmeticProgression a) : 
  a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_456_terms_l2231_223180


namespace NUMINAMATH_CALUDE_max_triangle_area_is_sqrt3_div_2_l2231_223110

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line in 2D space -/
structure Line where
  k : ℝ
  m : ℝ

/-- The maximum area of triangle AOB for the given ellipse and line conditions -/
def max_triangle_area (e : Ellipse) (l : Line) : ℝ :=
  sorry

/-- Main theorem: The maximum area of triangle AOB is √3/2 under the given conditions -/
theorem max_triangle_area_is_sqrt3_div_2 
  (e : Ellipse) 
  (h_vertex : e.b = 1)
  (h_eccentricity : Real.sqrt (e.a^2 - e.b^2) / e.a = Real.sqrt 6 / 3)
  (l : Line)
  (h_distance : |l.m| / Real.sqrt (1 + l.k^2) = Real.sqrt 3 / 2) :
  max_triangle_area e l = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_is_sqrt3_div_2_l2231_223110


namespace NUMINAMATH_CALUDE_terminal_side_second_quadrant_l2231_223159

-- Define the quadrants
inductive Quadrant
| I
| II
| III
| IV

-- Define a function to determine the quadrant of an angle
def angle_quadrant (θ : ℝ) : Quadrant := sorry

-- Define a function to determine the quadrant of the terminal side of an angle
def terminal_side_quadrant (θ : ℝ) : Quadrant := sorry

-- Theorem statement
theorem terminal_side_second_quadrant (α : ℝ) :
  angle_quadrant α = Quadrant.III →
  |Real.cos (α/2)| = -Real.cos (α/2) →
  terminal_side_quadrant (α/2) = Quadrant.II :=
by sorry

end NUMINAMATH_CALUDE_terminal_side_second_quadrant_l2231_223159


namespace NUMINAMATH_CALUDE_ali_baba_max_camels_l2231_223169

/-- Represents the problem of maximizing the number of camels Ali Baba can buy --/
theorem ali_baba_max_camels :
  let gold_capacity : ℝ := 200
  let diamond_capacity : ℝ := 40
  let max_weight : ℝ := 100
  let gold_camel_rate : ℝ := 20
  let diamond_camel_rate : ℝ := 60
  
  ∃ (gold_weight diamond_weight : ℝ),
    gold_weight ≥ 0 ∧
    diamond_weight ≥ 0 ∧
    gold_weight + diamond_weight ≤ max_weight ∧
    gold_weight / gold_capacity + diamond_weight / diamond_capacity ≤ 1 ∧
    ∀ (g d : ℝ),
      g ≥ 0 →
      d ≥ 0 →
      g + d ≤ max_weight →
      g / gold_capacity + d / diamond_capacity ≤ 1 →
      gold_camel_rate * g + diamond_camel_rate * d ≤ gold_camel_rate * gold_weight + diamond_camel_rate * diamond_weight ∧
    gold_camel_rate * gold_weight + diamond_camel_rate * diamond_weight = 3000 := by
  sorry


end NUMINAMATH_CALUDE_ali_baba_max_camels_l2231_223169


namespace NUMINAMATH_CALUDE_candy_store_sampling_theorem_l2231_223100

/-- The percentage of customers who sample candy but are not caught -/
def uncaught_samplers (total_samplers caught_samplers : ℝ) : ℝ :=
  total_samplers - caught_samplers

theorem candy_store_sampling_theorem 
  (total_samplers : ℝ) 
  (caught_samplers : ℝ) 
  (h1 : caught_samplers = 22)
  (h2 : total_samplers = 23.913043478260867) :
  uncaught_samplers total_samplers caught_samplers = 1.913043478260867 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_sampling_theorem_l2231_223100


namespace NUMINAMATH_CALUDE_tree_space_calculation_l2231_223149

/-- Given a road of length 151 feet where 11 trees are planted with 14 feet between each tree,
    prove that each tree occupies 1 square foot of sidewalk space. -/
theorem tree_space_calculation (road_length : ℕ) (num_trees : ℕ) (gap_between_trees : ℕ) :
  road_length = 151 →
  num_trees = 11 →
  gap_between_trees = 14 →
  (road_length - (num_trees - 1) * gap_between_trees) / num_trees = 1 := by
  sorry

end NUMINAMATH_CALUDE_tree_space_calculation_l2231_223149


namespace NUMINAMATH_CALUDE_minimum_pass_rate_four_subjects_l2231_223153

theorem minimum_pass_rate_four_subjects 
  (math_pass : Real) (chinese_pass : Real) (english_pass : Real) (chemistry_pass : Real)
  (h_math : math_pass = 0.99)
  (h_chinese : chinese_pass = 0.98)
  (h_english : english_pass = 0.96)
  (h_chemistry : chemistry_pass = 0.92) :
  1 - (1 - math_pass + 1 - chinese_pass + 1 - english_pass + 1 - chemistry_pass) = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_minimum_pass_rate_four_subjects_l2231_223153


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2231_223175

/-- A hyperbola with center at the origin -/
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_axes : Bool
  asymptote_angle : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: The eccentricity of the given hyperbola is either 2 or 2√3/3 -/
theorem hyperbola_eccentricity (C : Hyperbola) 
  (h1 : C.center = (0, 0))
  (h2 : C.foci_on_axes = true)
  (h3 : C.asymptote_angle = π / 3) :
  eccentricity C = 2 ∨ eccentricity C = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2231_223175


namespace NUMINAMATH_CALUDE_matrix_vector_computation_l2231_223165

variable {n : ℕ}
variable (N : Matrix (Fin 2) (Fin n) ℝ)
variable (a b : Fin n → ℝ)

theorem matrix_vector_computation 
  (ha : N.mulVec a = ![3, 4]) 
  (hb : N.mulVec b = ![1, -2]) :
  N.mulVec (2 • a - 4 • b) = ![2, 16] := by
  sorry

end NUMINAMATH_CALUDE_matrix_vector_computation_l2231_223165


namespace NUMINAMATH_CALUDE_add_point_three_to_twenty_nine_point_eight_l2231_223161

theorem add_point_three_to_twenty_nine_point_eight : 
  29.8 + 0.3 = 30.1 := by
  sorry

end NUMINAMATH_CALUDE_add_point_three_to_twenty_nine_point_eight_l2231_223161


namespace NUMINAMATH_CALUDE_third_group_students_l2231_223105

/-- The number of tissues in each mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The number of students in the first kindergartner group -/
def group1_students : ℕ := 9

/-- The number of students in the second kindergartner group -/
def group2_students : ℕ := 10

/-- The total number of tissues brought by all groups -/
def total_tissues : ℕ := 1200

/-- Theorem stating that the number of students in the third kindergartner group is 11 -/
theorem third_group_students :
  ∃ (x : ℕ), x = 11 ∧ 
  tissues_per_box * (group1_students + group2_students + x) = total_tissues :=
sorry

end NUMINAMATH_CALUDE_third_group_students_l2231_223105


namespace NUMINAMATH_CALUDE_smallest_n_for_interval_multiple_l2231_223129

theorem smallest_n_for_interval_multiple : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 → 
    ∃ (k : ℕ), (m : ℚ) / 1993 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m + 1) : ℚ) / 1994) ∧
  (∀ (n' : ℕ), 0 < n' ∧ n' < n → 
    ∃ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 ∧
      ∀ (k : ℕ), ¬((m : ℚ) / 1993 < (k : ℚ) / n' ∧ (k : ℚ) / n' < ((m + 1) : ℚ) / 1994)) ∧
  n = 3987 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_interval_multiple_l2231_223129


namespace NUMINAMATH_CALUDE_rectangle_difference_l2231_223193

theorem rectangle_difference (x y : ℝ) : 
  y = x / 3 →
  2 * x + 2 * y = 32 →
  x^2 + y^2 = 17^2 →
  x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_difference_l2231_223193


namespace NUMINAMATH_CALUDE_smallest_multiple_with_remainder_three_l2231_223140

theorem smallest_multiple_with_remainder_three : 
  (∀ n : ℕ, n > 1 ∧ n < 843 → 
    ¬(n % 4 = 3 ∧ n % 5 = 3 ∧ n % 6 = 3 ∧ n % 7 = 3 ∧ n % 8 = 3)) ∧ 
  (843 % 4 = 3 ∧ 843 % 5 = 3 ∧ 843 % 6 = 3 ∧ 843 % 7 = 3 ∧ 843 % 8 = 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_remainder_three_l2231_223140


namespace NUMINAMATH_CALUDE_existence_of_solutions_l2231_223157

theorem existence_of_solutions (n : ℕ) : ∃ x y z : ℕ, x^2 + y^2 = z^n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solutions_l2231_223157


namespace NUMINAMATH_CALUDE_no_integer_roots_l2231_223173

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluate a polynomial at a given integer -/
def eval (p : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

theorem no_integer_roots (P : IntPolynomial) 
  (h2020 : eval P 2020 = 2021) 
  (h2021 : eval P 2021 = 2021) : 
  ∀ x : ℤ, eval P x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l2231_223173


namespace NUMINAMATH_CALUDE_arithmetic_geometric_relation_l2231_223127

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
structure GeometricSequence where
  b : ℕ → ℝ
  r : ℝ
  h_geom : ∀ n, b (n + 1) = r * b n

/-- The theorem statement -/
theorem arithmetic_geometric_relation (seq : ArithmeticSequence)
    (h_geom : ∃ (g : GeometricSequence), 
      g.b 1 = seq.a 2 ∧ g.b 2 = seq.a 3 ∧ g.b 3 = seq.a 7) :
    (∃ (g : GeometricSequence), 
      g.b 1 = seq.a 2 ∧ g.b 2 = seq.a 3 ∧ g.b 3 = seq.a 7 ∧ g.r = 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_relation_l2231_223127


namespace NUMINAMATH_CALUDE_average_speed_theorem_l2231_223143

def speed_1 : ℝ := 100
def speed_2 : ℝ := 80
def speed_3_4 : ℝ := 90
def speed_5 : ℝ := 60
def speed_6 : ℝ := 70

def duration_1 : ℝ := 1
def duration_2 : ℝ := 1
def duration_3_4 : ℝ := 2
def duration_5 : ℝ := 1
def duration_6 : ℝ := 1

def total_distance : ℝ := 
  speed_1 * duration_1 + 
  speed_2 * duration_2 + 
  speed_3_4 * duration_3_4 + 
  speed_5 * duration_5 + 
  speed_6 * duration_6

def total_time : ℝ := 
  duration_1 + duration_2 + duration_3_4 + duration_5 + duration_6

theorem average_speed_theorem : 
  total_distance / total_time = 490 / 6 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_theorem_l2231_223143


namespace NUMINAMATH_CALUDE_number_equality_l2231_223109

theorem number_equality : ∃ x : ℝ, x * 120 = 173 * 240 ∧ x = 346 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2231_223109


namespace NUMINAMATH_CALUDE_distance_home_to_school_l2231_223199

/-- Represents the scenario of a boy traveling between home and school. -/
structure TravelScenario where
  speed : ℝ  -- Speed in km/hr
  time_diff : ℝ  -- Time difference in hours (positive for late, negative for early)

/-- The distance between home and school satisfies the given travel scenarios. -/
def distance_satisfies (d : ℝ) (s1 s2 : TravelScenario) : Prop :=
  ∃ t : ℝ, 
    d = s1.speed * (t + s1.time_diff) ∧
    d = s2.speed * (t - s2.time_diff)

/-- The theorem stating the distance between home and school. -/
theorem distance_home_to_school : 
  ∃ d : ℝ, d = 1.5 ∧ 
    distance_satisfies d 
      { speed := 3, time_diff := 7/60 }
      { speed := 6, time_diff := -8/60 } := by
  sorry

end NUMINAMATH_CALUDE_distance_home_to_school_l2231_223199


namespace NUMINAMATH_CALUDE_number_value_proof_l2231_223108

theorem number_value_proof (x : ℝ) : (8^3 * x^3) / 679 = 549.7025036818851 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_value_proof_l2231_223108


namespace NUMINAMATH_CALUDE_sarah_father_age_double_l2231_223184

/-- Given Sarah's age and her father's age in 2010, find the year when the father's age will be double Sarah's age -/
theorem sarah_father_age_double (sarah_age_2010 : ℕ) (father_age_2010 : ℕ) 
  (h1 : sarah_age_2010 = 10)
  (h2 : father_age_2010 = 6 * sarah_age_2010) :
  ∃ (year : ℕ), 
    year > 2010 ∧ 
    (father_age_2010 + (year - 2010)) = 2 * (sarah_age_2010 + (year - 2010)) ∧
    year = 2030 :=
by sorry

end NUMINAMATH_CALUDE_sarah_father_age_double_l2231_223184


namespace NUMINAMATH_CALUDE_harveys_steak_sales_l2231_223120

/-- Given the initial number of steaks, the number left after the first sale,
    and the number of additional steaks sold, calculate the total number of steaks sold. -/
def total_steaks_sold (initial : ℕ) (left_after_first_sale : ℕ) (additional_sold : ℕ) : ℕ :=
  (initial - left_after_first_sale) + additional_sold

/-- Theorem stating that for Harvey's specific case, the total number of steaks sold is 17. -/
theorem harveys_steak_sales : total_steaks_sold 25 12 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_harveys_steak_sales_l2231_223120


namespace NUMINAMATH_CALUDE_last_second_occurrence_is_two_l2231_223125

-- Define the Fibonacci sequence modulo 10
def fib_mod_10 : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => (fib_mod_10 n + fib_mod_10 (n + 1)) % 10

-- Define a function to check if a digit has appeared at least twice up to a given index
def appears_twice (d : ℕ) (n : ℕ) : Prop :=
  ∃ i j, i < j ∧ j ≤ n ∧ fib_mod_10 i = d ∧ fib_mod_10 j = d

-- State the theorem
theorem last_second_occurrence_is_two :
  ∀ d, d ≠ 2 → ∃ n, appears_twice d n ∧ ¬appears_twice 2 n :=
sorry

end NUMINAMATH_CALUDE_last_second_occurrence_is_two_l2231_223125


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2231_223104

/-- A geometric sequence with negative terms -/
def NegativeGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n < 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  NegativeGeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = -6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2231_223104


namespace NUMINAMATH_CALUDE_ones_digit_73_pow_351_l2231_223190

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- The ones digit pattern for powers of 3 -/
def onesDigitPattern : List ℕ := [3, 9, 7, 1]

theorem ones_digit_73_pow_351 : onesDigit (73^351) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_73_pow_351_l2231_223190


namespace NUMINAMATH_CALUDE_pit_stop_duration_is_20_minutes_l2231_223168

-- Define the parameters of the problem
def total_trip_time_without_stops : ℕ := 14 -- hours
def stop_interval : ℕ := 2 -- hours
def additional_food_stops : ℕ := 2
def additional_gas_stops : ℕ := 3
def total_trip_time_with_stops : ℕ := 18 -- hours

-- Calculate the number of stops
def total_stops : ℕ := 
  (total_trip_time_without_stops / stop_interval) + additional_food_stops + additional_gas_stops

-- Define the theorem
theorem pit_stop_duration_is_20_minutes : 
  (total_trip_time_with_stops - total_trip_time_without_stops) * 60 / total_stops = 20 := by
  sorry

end NUMINAMATH_CALUDE_pit_stop_duration_is_20_minutes_l2231_223168


namespace NUMINAMATH_CALUDE_quadratic_power_function_l2231_223170

/-- A function is a power function if it's of the form f(x) = x^a for some real number a -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

/-- A function is quadratic if it's of the form f(x) = ax^2 + bx + c for some real numbers a, b, c with a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

theorem quadratic_power_function (f : ℝ → ℝ) :
  IsQuadratic f ∧ IsPowerFunction f → ∀ x : ℝ, f x = x^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_power_function_l2231_223170


namespace NUMINAMATH_CALUDE_system_solution_l2231_223164

theorem system_solution :
  ∀ (x y z : ℝ),
    (y + z = x * y * z ∧
     z + x = x * y * z ∧
     x + y = x * y * z) →
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨
     (x = Real.sqrt 2 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 2) ∨
     (x = -Real.sqrt 2 ∧ y = -Real.sqrt 2 ∧ z = -Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2231_223164


namespace NUMINAMATH_CALUDE_different_color_probability_l2231_223122

/-- The probability of drawing two balls of different colors from a box -/
theorem different_color_probability (total : ℕ) (red : ℕ) (yellow : ℕ) : 
  total = red + yellow →
  red = 3 →
  yellow = 2 →
  (red.choose 1 * yellow.choose 1 : ℚ) / total.choose 2 = 3/5 :=
sorry

end NUMINAMATH_CALUDE_different_color_probability_l2231_223122


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l2231_223130

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |x - 4| - a

-- Theorem for the minimum value when a = 1
theorem min_value_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 4 ∧ ∃ y : ℝ, f 1 y = 4 :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 4/a + 1) ↔ (a < 0 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l2231_223130


namespace NUMINAMATH_CALUDE_path_area_is_675_l2231_223142

/-- Calculates the area of a path surrounding a rectangular field. -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Theorem: The area of the path surrounding the given rectangular field is 675 sq m. -/
theorem path_area_is_675 (field_length field_width path_width cost_per_sqm total_cost : ℝ) :
  field_length = 75 →
  field_width = 55 →
  path_width = 2.5 →
  cost_per_sqm = 10 →
  total_cost = 6750 →
  path_area field_length field_width path_width = 675 :=
by
  sorry

#eval path_area 75 55 2.5

end NUMINAMATH_CALUDE_path_area_is_675_l2231_223142


namespace NUMINAMATH_CALUDE_card_house_47_floors_l2231_223179

/-- The number of cards needed for the nth floor of a card house -/
def cards_for_floor (n : ℕ) : ℕ := 2 + (n - 1) * 3

/-- The total number of cards needed for a card house with n floors -/
def total_cards (n : ℕ) : ℕ := 
  n * (cards_for_floor 1 + cards_for_floor n) / 2

/-- Theorem: A card house with 47 floors requires 3337 cards -/
theorem card_house_47_floors : total_cards 47 = 3337 := by
  sorry

end NUMINAMATH_CALUDE_card_house_47_floors_l2231_223179


namespace NUMINAMATH_CALUDE_difference_of_squares_81_49_l2231_223182

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_81_49_l2231_223182


namespace NUMINAMATH_CALUDE_scientific_notation_of_384000_l2231_223116

/-- Given a number 384000, prove that its scientific notation representation is 3.84 × 10^5 -/
theorem scientific_notation_of_384000 :
  (384000 : ℝ) = 3.84 * (10 : ℝ)^5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_384000_l2231_223116


namespace NUMINAMATH_CALUDE_equality_of_exponents_l2231_223183

theorem equality_of_exponents (x y z : ℝ) 
  (h : x * (y + z - x) / x = y * (z + x - y) / y ∧ 
       y * (z + x - y) / y = z * (x + y - z) / z) : 
  x^y * y^x = z^y * y^z ∧ z^y * y^z = x^z * z^x := by
  sorry

end NUMINAMATH_CALUDE_equality_of_exponents_l2231_223183


namespace NUMINAMATH_CALUDE_no_valid_coloring_l2231_223121

theorem no_valid_coloring : ¬∃ (f : ℕ+ → Bool), 
  (∀ n : ℕ+, f n ≠ f (n + 5)) ∧ 
  (∀ n : ℕ+, f n ≠ f (2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l2231_223121


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l2231_223131

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l2231_223131


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2231_223176

theorem quadratic_roots_sum (a b : ℝ) : 
  a ≠ b → 
  a^2 - 8*a + 5 = 0 → 
  b^2 - 8*b + 5 = 0 → 
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2231_223176


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l2231_223188

theorem quadratic_root_sum (m n : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 2*n = 0 ∧ x = 2) → m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l2231_223188


namespace NUMINAMATH_CALUDE_circle_radii_order_l2231_223133

theorem circle_radii_order (r_A r_B r_C : ℝ) : 
  r_A = 2 →
  2 * Real.pi * r_B = 10 * Real.pi →
  Real.pi * r_C^2 = 16 * Real.pi →
  r_A < r_C ∧ r_C < r_B := by
sorry

end NUMINAMATH_CALUDE_circle_radii_order_l2231_223133


namespace NUMINAMATH_CALUDE_ellipse_sum_theorem_l2231_223198

/-- Theorem: For an ellipse with given parameters, the sum h + k + a + b + 2c equals 9 + 2√33 -/
theorem ellipse_sum_theorem (h k a b c : ℝ) : 
  h = 3 → 
  k = -5 → 
  a = 7 → 
  b = 4 → 
  c = Real.sqrt (a^2 - b^2) → 
  h + k + a + b + 2*c = 9 + 2 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_theorem_l2231_223198


namespace NUMINAMATH_CALUDE_remainder_876539_mod_7_l2231_223158

theorem remainder_876539_mod_7 : 876539 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_876539_mod_7_l2231_223158


namespace NUMINAMATH_CALUDE_sum_irrational_implies_component_irrational_l2231_223106

theorem sum_irrational_implies_component_irrational (a b c : ℝ) : 
  ¬ (∃ (q : ℚ), (a + b + c : ℝ) = q) → 
  ¬ (∃ (q₁ q₂ q₃ : ℚ), (a = q₁ ∧ b = q₂ ∧ c = q₃)) :=
by sorry

end NUMINAMATH_CALUDE_sum_irrational_implies_component_irrational_l2231_223106


namespace NUMINAMATH_CALUDE_product_of_odds_over_sum_of_squares_l2231_223101

theorem product_of_odds_over_sum_of_squares : 
  (1 * 3 * 5 * 7) / (1^2 + 2^2 + 3^2 + 4^2) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_odds_over_sum_of_squares_l2231_223101


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_eleven_satisfies_smallest_integer_is_eleven_l2231_223145

theorem smallest_integer_fraction (y : ℤ) : (5 : ℚ) / 8 < (y : ℚ) / 17 → y ≥ 11 :=
by sorry

theorem eleven_satisfies (y : ℤ) : (5 : ℚ) / 8 < (11 : ℚ) / 17 :=
by sorry

theorem smallest_integer_is_eleven : 
  ∃ y : ℤ, ((5 : ℚ) / 8 < (y : ℚ) / 17) ∧ (∀ z : ℤ, (5 : ℚ) / 8 < (z : ℚ) / 17 → z ≥ y) ∧ y = 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_eleven_satisfies_smallest_integer_is_eleven_l2231_223145


namespace NUMINAMATH_CALUDE_sum_a_b_is_negative_two_l2231_223139

theorem sum_a_b_is_negative_two (a b : ℝ) (h : |a - 1| + (b + 3)^2 = 0) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_is_negative_two_l2231_223139


namespace NUMINAMATH_CALUDE_normal_distribution_probability_bagged_rice_probability_l2231_223119

/-- The cumulative distribution function of the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The probability density function of the normal distribution with mean μ and variance σ² -/
noncomputable def normalPDF (μ σ : ℝ) (x : ℝ) : ℝ := sorry

theorem normal_distribution_probability (μ σ x₁ x₂ : ℝ) (hσ : σ > 0) :
  (∫ x in x₁..x₂, normalPDF μ σ x) = Φ ((x₂ - μ) / σ) - Φ ((x₁ - μ) / σ) :=
sorry

/-- The probability that a value from N(10, 0.01) is between 9.8 and 10.2 -/
theorem bagged_rice_probability :
  (∫ x in 9.8..10.2, normalPDF 10 0.1 x) = 2 * Φ 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_bagged_rice_probability_l2231_223119


namespace NUMINAMATH_CALUDE_parabola_properties_l2231_223194

/-- Definition of the parabola C: x² = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Definition of the line y = x + 1 -/
def line (x y : ℝ) : Prop := y = x + 1

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 1)

/-- The length of the chord AB -/
def chord_length : ℝ := 8

/-- Theorem stating the properties of the parabola and its intersection with the line -/
theorem parabola_properties :
  (∀ x y, parabola x y → (x, y) ≠ focus → (x - focus.1)^2 + (y - focus.2)^2 > 0) ∧
  (∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l2231_223194


namespace NUMINAMATH_CALUDE_shadow_length_l2231_223187

theorem shadow_length (h₁ s₁ h₂ : ℝ) (h_h₁ : h₁ = 20) (h_s₁ : s₁ = 10) (h_h₂ : h₂ = 40) :
  ∃ s₂ : ℝ, s₂ = 20 ∧ h₁ / s₁ = h₂ / s₂ :=
by sorry

end NUMINAMATH_CALUDE_shadow_length_l2231_223187
