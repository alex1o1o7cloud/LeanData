import Mathlib

namespace NUMINAMATH_CALUDE_min_buses_for_field_trip_l281_28189

def min_buses (total_students : ℕ) (bus_capacity_1 : ℕ) (bus_capacity_2 : ℕ) : ℕ :=
  let large_buses := total_students / bus_capacity_1
  let remaining_students := total_students % bus_capacity_1
  if remaining_students = 0 then
    large_buses
  else
    large_buses + 1

theorem min_buses_for_field_trip :
  min_buses 530 45 40 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_buses_for_field_trip_l281_28189


namespace NUMINAMATH_CALUDE_max_value_theorem_l281_28113

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_eq : x^2 - 3*x*y + 4*y^2 - z = 0) :
  (∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 ∧
    z'/(x'*y') ≤ z/(x*y) ∧
    x + 2*y - z ≤ x' + 2*y' - z') ∧
  (∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 →
    x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 →
    z'/(x'*y') ≤ z/(x*y) →
    x' + 2*y' - z' ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l281_28113


namespace NUMINAMATH_CALUDE_quadratic_minimum_l281_28114

def f (x : ℝ) := 5 * x^2 - 15 * x + 2

theorem quadratic_minimum :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -9.25 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l281_28114


namespace NUMINAMATH_CALUDE_tree_distance_l281_28131

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 10) (h2 : d = 100) :
  let tree_spacing := d / 5
  tree_spacing * (n - 1) = 180 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l281_28131


namespace NUMINAMATH_CALUDE_square_diagonal_l281_28187

theorem square_diagonal (A : ℝ) (h : A = 800) :
  ∃ d : ℝ, d = 40 ∧ d^2 = 2 * A :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_l281_28187


namespace NUMINAMATH_CALUDE_solve_exponent_equation_l281_28160

theorem solve_exponent_equation (n : ℕ) : 2 * 2^2 * 2^n = 2^10 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponent_equation_l281_28160


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l281_28138

theorem min_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  (1 / x + 1 / y + 1 / z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l281_28138


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l281_28155

theorem quadratic_inequality_solution (x : ℝ) : x^2 + x - 12 ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l281_28155


namespace NUMINAMATH_CALUDE_pumpkin_weight_difference_l281_28198

/-- Given three pumpkin weights with specific relationships, 
    prove the difference between the heaviest and lightest is 81 pounds. -/
theorem pumpkin_weight_difference :
  ∀ (brad jessica betty : ℝ),
  brad = 54 →
  jessica = brad / 2 →
  betty = 4 * jessica →
  max brad (max jessica betty) - min brad (min jessica betty) = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_pumpkin_weight_difference_l281_28198


namespace NUMINAMATH_CALUDE_susan_spending_l281_28178

def carnival_spending (initial_amount food_cost : ℕ) : ℕ :=
  let ride_cost := 2 * food_cost
  let total_spent := food_cost + ride_cost
  initial_amount - total_spent

theorem susan_spending :
  carnival_spending 50 12 = 14 := by
  sorry

end NUMINAMATH_CALUDE_susan_spending_l281_28178


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l281_28121

-- Problem 1
theorem problem_1 : (-3)^2 + (Real.pi - 1/2)^0 - |(-4)| = 6 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ -1) : 
  (1 - 1/(a+1)) * ((a^2 + 2*a + 1)/a) = a + 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l281_28121


namespace NUMINAMATH_CALUDE_no_function_exists_for_part_a_function_exists_for_part_b_l281_28123

-- Part a
theorem no_function_exists_for_part_a :
  ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1 :=
sorry

-- Part b
theorem function_exists_for_part_b :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = 2 * n :=
sorry

end NUMINAMATH_CALUDE_no_function_exists_for_part_a_function_exists_for_part_b_l281_28123


namespace NUMINAMATH_CALUDE_cricket_average_score_l281_28127

theorem cricket_average_score 
  (total_matches : ℕ) 
  (overall_average : ℝ) 
  (first_six_average : ℝ) 
  (last_four_sum_lower : ℝ) 
  (last_four_sum_upper : ℝ) 
  (last_four_lowest : ℝ) 
  (h1 : total_matches = 10)
  (h2 : overall_average = 38.9)
  (h3 : first_six_average = 41)
  (h4 : last_four_sum_lower = 120)
  (h5 : last_four_sum_upper = 200)
  (h6 : last_four_lowest = 20)
  (h7 : last_four_sum_lower ≤ (overall_average * total_matches - first_six_average * 6))
  (h8 : (overall_average * total_matches - first_six_average * 6) ≤ last_four_sum_upper) :
  (overall_average * total_matches - first_six_average * 6) / 4 = 35.75 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_score_l281_28127


namespace NUMINAMATH_CALUDE_bread_tear_ratio_l281_28103

/-- Represents the number of bread slices -/
def num_slices : ℕ := 2

/-- Represents the total number of pieces after tearing -/
def total_pieces : ℕ := 8

/-- Represents the number of pieces each slice is torn into -/
def pieces_per_slice : ℕ := total_pieces / num_slices

/-- Proves that the ratio of pieces after the first tear to pieces after the second tear is 1:1 -/
theorem bread_tear_ratio :
  pieces_per_slice = pieces_per_slice → (pieces_per_slice : ℚ) / pieces_per_slice = 1 := by
  sorry

end NUMINAMATH_CALUDE_bread_tear_ratio_l281_28103


namespace NUMINAMATH_CALUDE_watermelon_cost_proof_l281_28151

/-- Represents the number of fruits a container can hold -/
def ContainerCapacity : ℕ := 150

/-- Represents the total value of fruits in rubles -/
def TotalValue : ℕ := 24000

/-- Represents the capacity of the container in terms of melons -/
def MelonCapacity : ℕ := 120

/-- Represents the capacity of the container in terms of watermelons -/
def WatermelonCapacity : ℕ := 160

/-- Represents the cost of a single watermelon in rubles -/
def WatermelonCost : ℕ := 100

theorem watermelon_cost_proof :
  ∃ (num_watermelons num_melons : ℕ),
    num_watermelons + num_melons = ContainerCapacity ∧
    num_watermelons * WatermelonCost = num_melons * (TotalValue / num_melons) ∧
    num_watermelons * WatermelonCost + num_melons * (TotalValue / num_melons) = TotalValue ∧
    num_watermelons * (1 / WatermelonCapacity) + num_melons * (1 / MelonCapacity) = 1 :=
by sorry

end NUMINAMATH_CALUDE_watermelon_cost_proof_l281_28151


namespace NUMINAMATH_CALUDE_least_m_for_no_real_roots_l281_28139

theorem least_m_for_no_real_roots : 
  ∃ (m : ℤ), (∀ (x : ℝ), 3 * x * (m * x + 6) - 2 * x^2 + 8 ≠ 0) ∧
  (∀ (k : ℤ), k < m → ∃ (x : ℝ), 3 * x * (k * x + 6) - 2 * x^2 + 8 = 0) ∧
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_m_for_no_real_roots_l281_28139


namespace NUMINAMATH_CALUDE_square_difference_one_l281_28185

theorem square_difference_one : (726 : ℕ) * 726 - 725 * 727 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_one_l281_28185


namespace NUMINAMATH_CALUDE_logarithm_identity_l281_28105

theorem logarithm_identity (a : ℝ) (ha : a > 0) : 
  a^(Real.log (Real.log a)) - (Real.log a)^(Real.log a) = 0 :=
by sorry

end NUMINAMATH_CALUDE_logarithm_identity_l281_28105


namespace NUMINAMATH_CALUDE_christine_siri_money_difference_l281_28154

theorem christine_siri_money_difference :
  ∀ (christine_amount siri_amount : ℝ),
    christine_amount + siri_amount = 21 →
    christine_amount = 20.5 →
    christine_amount - siri_amount = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_christine_siri_money_difference_l281_28154


namespace NUMINAMATH_CALUDE_janet_total_cost_l281_28126

/-- Calculates the total cost for Janet's group at the waterpark -/
def waterpark_cost (adult_price : ℚ) (group_size : ℕ) (child_count : ℕ) (soda_price : ℚ) : ℚ :=
  let child_price := adult_price / 2
  let adult_count := group_size - child_count
  let total_ticket_cost := adult_price * adult_count + child_price * child_count
  let discount := total_ticket_cost * (1/5)
  (total_ticket_cost - discount) + soda_price

/-- Proves that Janet's total cost is $197 -/
theorem janet_total_cost : 
  waterpark_cost 30 10 4 5 = 197 := by
  sorry

end NUMINAMATH_CALUDE_janet_total_cost_l281_28126


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l281_28166

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ (x y : ℝ), x = 2 ∧ y = 2 ∧ y = a^(x - 2) + 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l281_28166


namespace NUMINAMATH_CALUDE_max_value_of_f_l281_28143

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 1 ∧
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x ≤ f c) ∧
  f c = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l281_28143


namespace NUMINAMATH_CALUDE_absolute_value_greater_than_x_l281_28196

theorem absolute_value_greater_than_x (x : ℝ) : (x < 0) ↔ (abs x > x) := by sorry

end NUMINAMATH_CALUDE_absolute_value_greater_than_x_l281_28196


namespace NUMINAMATH_CALUDE_prime_sum_equality_l281_28134

theorem prime_sum_equality (n : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p * (p + 1) + q * (q + 1) = n * (n + 1)) → 
  n = 3 ∨ n = 6 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_equality_l281_28134


namespace NUMINAMATH_CALUDE_quarterback_passes_l281_28192

theorem quarterback_passes (total passes_left passes_right passes_center : ℕ) : 
  total = 50 → 
  passes_left = 12 → 
  passes_right = 2 * passes_left → 
  total = passes_left + passes_right + passes_center → 
  passes_center - passes_left = 2 := by
  sorry

end NUMINAMATH_CALUDE_quarterback_passes_l281_28192


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l281_28176

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) ↔ x ≥ 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l281_28176


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l281_28150

theorem complex_magnitude_example : Complex.abs (-3 + (8/5)*Complex.I) = 17/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l281_28150


namespace NUMINAMATH_CALUDE_cone_volume_l281_28195

theorem cone_volume (cylinder_volume : ℝ) (h : cylinder_volume = 30) :
  let cone_volume := cylinder_volume / 3
  cone_volume = 10 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l281_28195


namespace NUMINAMATH_CALUDE_tree_height_after_four_years_l281_28148

/-- The height of a tree that doubles every year -/
def treeHeight (initialHeight : ℝ) (years : ℕ) : ℝ :=
  initialHeight * (2 ^ years)

theorem tree_height_after_four_years
  (h : treeHeight 1 7 = 64) :
  treeHeight 1 4 = 8 :=
sorry

end NUMINAMATH_CALUDE_tree_height_after_four_years_l281_28148


namespace NUMINAMATH_CALUDE_circle_center_and_difference_l281_28136

/-- 
Given a circle described by the equation x^2 + y^2 - 10x + 4y + 13 = 0,
prove that its center is (5, -2) and x - y = 7.
-/
theorem circle_center_and_difference (x y : ℝ) :
  x^2 + y^2 - 10*x + 4*y + 13 = 0 →
  (∃ (r : ℝ), (x - 5)^2 + (y + 2)^2 = r^2) ∧
  x - y = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_and_difference_l281_28136


namespace NUMINAMATH_CALUDE_book_price_difference_l281_28156

def necklace_price : ℕ := 34
def spending_limit : ℕ := 70
def overspent_amount : ℕ := 3

theorem book_price_difference (book_price : ℕ) : 
  book_price > necklace_price →
  book_price + necklace_price = spending_limit + overspent_amount →
  book_price - necklace_price = 5 := by
sorry

end NUMINAMATH_CALUDE_book_price_difference_l281_28156


namespace NUMINAMATH_CALUDE_four_digit_harmonious_divisible_by_11_l281_28117

/-- A four-digit harmonious number with 'a' as the first and last digit, and 'b' as the second and third digit. -/
def four_digit_harmonious (a b : ℕ) : ℕ := 1000 * a + 100 * b + 10 * b + a

/-- Proposition: All four-digit harmonious numbers are divisible by 11. -/
theorem four_digit_harmonious_divisible_by_11 (a b : ℕ) :
  ∃ k : ℕ, four_digit_harmonious a b = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_four_digit_harmonious_divisible_by_11_l281_28117


namespace NUMINAMATH_CALUDE_polynomial_roots_arithmetic_progression_l281_28167

theorem polynomial_roots_arithmetic_progression 
  (a b : ℝ) 
  (ha : a = 3 * Real.sqrt 3) 
  (hroots : ∀ (r s t : ℝ), 
    (r^3 - a*r^2 + b*r + a = 0 ∧ 
     s^3 - a*s^2 + b*s + a = 0 ∧ 
     t^3 - a*t^2 + b*t + a = 0) → 
    (r > 0 ∧ s > 0 ∧ t > 0) ∧ 
    ∃ (d : ℝ), (s = r + d ∧ t = r + 2*d) ∨ (s = r ∧ t = r)) : 
  b = 3 * (Real.sqrt 3 + 1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_arithmetic_progression_l281_28167


namespace NUMINAMATH_CALUDE_martha_lasagna_cost_l281_28194

/-- The cost of ingredients for Martha's lasagna -/
def lasagna_cost (cheese_quantity : Real) (cheese_price : Real) 
                 (meat_quantity : Real) (meat_price : Real) : Real :=
  cheese_quantity * cheese_price + meat_quantity * meat_price

/-- Theorem: The cost of ingredients for Martha's lasagna is $13 -/
theorem martha_lasagna_cost : 
  lasagna_cost 1.5 6 0.5 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_martha_lasagna_cost_l281_28194


namespace NUMINAMATH_CALUDE_base_b_problem_l281_28142

theorem base_b_problem (b : ℕ) : 
  (6 * b^2 + 5 * b + 5 = (2 * b + 5)^2) → b = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_b_problem_l281_28142


namespace NUMINAMATH_CALUDE_power_of_five_divides_l281_28100

/-- Sequence of positive integers defined recursively -/
def x : ℕ → ℕ
  | 0 => 2  -- We use 0-based indexing in Lean
  | n + 1 => 2 * (x n)^3 + x n

/-- The statement to be proved -/
theorem power_of_five_divides (n : ℕ) : 
  ∃ k : ℕ, x n^2 + 1 = 5^(n+1) * k ∧ ¬(5 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_divides_l281_28100


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l281_28175

def f (x : ℝ) : ℝ := x^3 + x

theorem f_strictly_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l281_28175


namespace NUMINAMATH_CALUDE_exponent_sum_zero_polynomial_simplification_l281_28197

-- Problem 1
theorem exponent_sum_zero (m : ℝ) : m^3 * m^6 + (-m^3)^3 = 0 := by sorry

-- Problem 2
theorem polynomial_simplification (a : ℝ) : a*(a-2) - 2*a*(1-3*a) = 7*a^2 - 4*a := by sorry

end NUMINAMATH_CALUDE_exponent_sum_zero_polynomial_simplification_l281_28197


namespace NUMINAMATH_CALUDE_valid_solutions_l281_28129

/-- Defines a function that checks if a triple of digits forms a valid solution --/
def is_valid_solution (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  ∃ (k : ℤ), k * (10*a + b + 10*b + c + 10*c + a) = 100*a + 10*b + c + a + b + c

/-- The main theorem stating the valid solutions --/
theorem valid_solutions :
  ∀ a b c : ℕ,
    is_valid_solution a b c ↔
      (a = 5 ∧ b = 1 ∧ c = 6) ∨
      (a = 9 ∧ b = 1 ∧ c = 2) ∨
      (a = 6 ∧ b = 4 ∧ c = 5) ∨
      (a = 3 ∧ b = 7 ∧ c = 8) ∨
      (a = 5 ∧ b = 7 ∧ c = 6) ∨
      (a = 7 ∧ b = 7 ∧ c = 4) ∨
      (a = 9 ∧ b = 7 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_valid_solutions_l281_28129


namespace NUMINAMATH_CALUDE_identity_function_satisfies_conditions_l281_28183

theorem identity_function_satisfies_conditions (f : ℕ → ℕ) 
  (h1 : ∀ n, f (2 * n) = 2 * f n) 
  (h2 : ∀ n, f (2 * n + 1) = 2 * f n + 1) : 
  ∀ n, f n = n := by sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_conditions_l281_28183


namespace NUMINAMATH_CALUDE_starting_number_proof_l281_28184

theorem starting_number_proof (x : ℕ) : 
  (∃ (l : List ℕ), l.length = 12 ∧ 
    (∀ n ∈ l, x ≤ n ∧ n ≤ 47 ∧ n % 3 = 0) ∧
    (∀ m, x ≤ m ∧ m ≤ 47 ∧ m % 3 = 0 → m ∈ l)) ↔ 
  x = 12 := by
  sorry

#check starting_number_proof

end NUMINAMATH_CALUDE_starting_number_proof_l281_28184


namespace NUMINAMATH_CALUDE_polygon_side_length_theorem_l281_28110

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields for a convex polygon

/-- Represents a way to divide a polygon into equilateral triangles and squares. -/
structure Division where
  -- Add necessary fields for a division

/-- Counts the number of ways to divide a polygon into equilateral triangles and squares. -/
def countDivisions (M : ConvexPolygon) : ℕ :=
  sorry

/-- Checks if a number is prime. -/
def isPrime (n : ℕ) : Prop :=
  sorry

/-- Gets the length of a side of a polygon. -/
def sideLength (M : ConvexPolygon) (side : ℕ) : ℕ :=
  sorry

theorem polygon_side_length_theorem (M : ConvexPolygon) (p : ℕ) :
  isPrime p → countDivisions M = p → ∃ side, sideLength M side = p - 1 :=
by sorry

end NUMINAMATH_CALUDE_polygon_side_length_theorem_l281_28110


namespace NUMINAMATH_CALUDE_multiplication_division_equality_l281_28128

theorem multiplication_division_equality : (3 * 4) / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_equality_l281_28128


namespace NUMINAMATH_CALUDE_dilation_rotation_composition_l281_28115

def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; -1, 0]

theorem dilation_rotation_composition :
  rotation_matrix * dilation_matrix = !![0, 2; -2, 0] := by sorry

end NUMINAMATH_CALUDE_dilation_rotation_composition_l281_28115


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l281_28173

theorem solve_equation_and_evaluate : ∃ x : ℝ, 
  (5 * x - 3 = 15 * x + 15) ∧ (6 * (x + 5) = 19.2) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l281_28173


namespace NUMINAMATH_CALUDE_cartesian_angle_properties_l281_28104

/-- An angle in the Cartesian coordinate system -/
structure CartesianAngle where
  /-- The x-coordinate of the point on the terminal side -/
  x : ℝ
  /-- The y-coordinate of the point on the terminal side -/
  y : ℝ

/-- Theorem about properties of a specific angle in the Cartesian coordinate system -/
theorem cartesian_angle_properties (α : CartesianAngle) 
  (h1 : α.x = -1) 
  (h2 : α.y = 2) : 
  (Real.sin α.y * Real.tan α.y = -4 * Real.sqrt 5 / 5) ∧ 
  ((Real.sin (α.y + Real.pi / 2) * Real.cos (7 * Real.pi / 2 - α.y) * Real.tan (2 * Real.pi - α.y)) / 
   (Real.sin (2 * Real.pi - α.y) * Real.tan (-α.y)) = -Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_cartesian_angle_properties_l281_28104


namespace NUMINAMATH_CALUDE_rihanna_shopping_theorem_l281_28133

def calculate_remaining_money (initial_amount : ℕ) (mango_count : ℕ) (juice_count : ℕ) (mango_price : ℕ) (juice_price : ℕ) : ℕ :=
  initial_amount - (mango_count * mango_price + juice_count * juice_price)

theorem rihanna_shopping_theorem (initial_amount : ℕ) (mango_count : ℕ) (juice_count : ℕ) (mango_price : ℕ) (juice_price : ℕ) :
  calculate_remaining_money initial_amount mango_count juice_count mango_price juice_price =
  initial_amount - (mango_count * mango_price + juice_count * juice_price) :=
by
  sorry

#eval calculate_remaining_money 50 6 6 3 3

end NUMINAMATH_CALUDE_rihanna_shopping_theorem_l281_28133


namespace NUMINAMATH_CALUDE_infinite_special_integers_l281_28137

theorem infinite_special_integers :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, (n ∣ 2^(2^n + 1) + 1) ∧ ¬(n ∣ 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_infinite_special_integers_l281_28137


namespace NUMINAMATH_CALUDE_hyperbola_properties_l281_28145

/-- A hyperbola with the given properties -/
def hyperbola (x y : ℝ) : Prop :=
  y^2 / 4 - x^2 = 1

theorem hyperbola_properties :
  ∃ (x y : ℝ),
    -- The hyperbola is centered at the origin
    hyperbola 0 0 ∧
    -- One of its asymptotes is x - 2y = 0
    (∃ (t : ℝ), x = 2*t ∧ y = t) ∧
    -- The hyperbola passes through the point P(√(5/2), 3)
    hyperbola (Real.sqrt (5/2)) 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l281_28145


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l281_28182

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def has_six_consecutive_nonprimes (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ 
    ∀ i : ℕ, i ≥ k ∧ i < k + 6 → ¬(is_prime i)

theorem smallest_prime_after_six_nonprimes :
  (is_prime 97) ∧ 
  (has_six_consecutive_nonprimes 96) ∧ 
  (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ has_six_consecutive_nonprimes (p - 1))) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l281_28182


namespace NUMINAMATH_CALUDE_smallest_number_with_rearranged_double_l281_28122

def digits_to_num (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

def num_to_digits (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

def rearrange_digits (digits : List Nat) : List Nat :=
  (digits.take 2).reverse ++ (digits.drop 2).reverse

theorem smallest_number_with_rearranged_double :
  ∃ (n : Nat),
    n = 263157894736842105 ∧
    (∀ m : Nat, m < n →
      let digits_m := num_to_digits m
      let r_m := digits_to_num (rearrange_digits digits_m)
      r_m ≠ 2 * m) ∧
    let digits_n := num_to_digits n
    let r_n := digits_to_num (rearrange_digits digits_n)
    r_n = 2 * n :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_rearranged_double_l281_28122


namespace NUMINAMATH_CALUDE_race_head_start_l281_28162

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (51 / 44) * Vb) :
  let H := L * (7 / 51)
  L / Va = (L - H) / Vb := by sorry

end NUMINAMATH_CALUDE_race_head_start_l281_28162


namespace NUMINAMATH_CALUDE_base_eight_to_ten_l281_28170

theorem base_eight_to_ten : 
  (1 * 8^3 + 7 * 8^2 + 2 * 8^1 + 4 * 8^0 : ℕ) = 980 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_to_ten_l281_28170


namespace NUMINAMATH_CALUDE_divisibility_of_sum_and_powers_l281_28102

theorem divisibility_of_sum_and_powers (a b c : ℤ) : 
  (6 ∣ (a + b + c)) → (6 ∣ (a^5 + b^3 + c)) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_and_powers_l281_28102


namespace NUMINAMATH_CALUDE_first_chapter_pages_l281_28158

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  chapter2_pages : ℕ

/-- The number of pages in the first chapter of a book -/
def pages_in_chapter1 (b : Book) : ℕ := b.total_pages - b.chapter2_pages

/-- Theorem: For a book with 81 total pages and 68 pages in the second chapter,
    the first chapter has 13 pages -/
theorem first_chapter_pages :
  ∀ (b : Book), b.total_pages = 81 → b.chapter2_pages = 68 →
  pages_in_chapter1 b = 13 := by
  sorry

end NUMINAMATH_CALUDE_first_chapter_pages_l281_28158


namespace NUMINAMATH_CALUDE_ellipse_equation_constants_l281_28107

def ellipse_constants (f1 f2 p : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

theorem ellipse_equation_constants :
  let f1 : ℝ × ℝ := (3, 1)
  let f2 : ℝ × ℝ := (3, 7)
  let p : ℝ × ℝ := (12, 2)
  let (a, b, h, k) := ellipse_constants f1 f2 p
  (a = (Real.sqrt 82 + Real.sqrt 106) / 2) ∧
  (b = Real.sqrt ((Real.sqrt 82 + Real.sqrt 106)^2 / 4 - 9)) ∧
  (h = 3) ∧
  (k = 4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_constants_l281_28107


namespace NUMINAMATH_CALUDE_inequality_preservation_l281_28177

theorem inequality_preservation (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l281_28177


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l281_28190

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (12 * x₁^2 + 16 * x₁ - 21 = 0) → 
  (12 * x₂^2 + 16 * x₂ - 21 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 95/18) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l281_28190


namespace NUMINAMATH_CALUDE_complex_power_of_one_plus_i_l281_28159

theorem complex_power_of_one_plus_i : (1 + Complex.I) ^ 6 = -8 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_of_one_plus_i_l281_28159


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l281_28180

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l281_28180


namespace NUMINAMATH_CALUDE_weight_distribution_l281_28179

theorem weight_distribution :
  ∃! (x y z : ℕ), x + y + z = 11 ∧ 3 * x + 7 * y + 14 * z = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_weight_distribution_l281_28179


namespace NUMINAMATH_CALUDE_banana_cost_theorem_l281_28112

/-- The cost of bananas given a specific rate and quantity -/
def banana_cost (rate_price : ℚ) (rate_quantity : ℚ) (buy_quantity : ℚ) : ℚ :=
  (rate_price / rate_quantity) * buy_quantity

/-- Theorem stating that 20 pounds of bananas cost $30 given the rate of $6 for 4 pounds -/
theorem banana_cost_theorem :
  banana_cost 6 4 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_theorem_l281_28112


namespace NUMINAMATH_CALUDE_correct_factorization_l281_28188

theorem correct_factorization (x y : ℝ) : x * (x - y) + y * (y - x) = (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l281_28188


namespace NUMINAMATH_CALUDE_sqrt_two_sufficient_not_necessary_l281_28186

/-- The line x + y = 0 is tangent to the circle x^2 + (y - a)^2 = 1 -/
def is_tangent (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x + y = 0 ∧ x^2 + (y - a)^2 = 1 ∧
  ∀ (x' y' : ℝ), x' + y' = 0 → x'^2 + (y' - a)^2 ≥ 1

/-- a = √2 is a sufficient but not necessary condition for the line to be tangent to the circle -/
theorem sqrt_two_sufficient_not_necessary :
  (∀ a : ℝ, a = Real.sqrt 2 → is_tangent a) ∧
  ¬(∀ a : ℝ, is_tangent a → a = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_sufficient_not_necessary_l281_28186


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l281_28120

theorem root_sum_reciprocal (p q r : ℝ) (A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ (x : ℝ), x^3 - 23*x^2 + 85*x - 72 = (x - p)*(x - q)*(x - r)) →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 23*s^2 + 85*s - 72) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 248 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l281_28120


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_m_l281_28124

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |1 - x|

-- Theorem for the first part of the problem
theorem solution_set_of_inequality (x : ℝ) :
  f x ≤ 3*x + 4 ↔ x ≥ -1/3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f x ≥ (m^2 - 3*m + 3) * |x|) ↔ 1 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_m_l281_28124


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l281_28147

theorem diophantine_equation_solutions :
  (∃ (S : Set (ℕ × ℕ × ℕ)), 
    (∀ (m n p : ℕ), (m, n, p) ∈ S ↔ 4 * m * n - m - n = p^2 - 1) ∧ 
    Set.Infinite S) ∧
  (∀ (m n p : ℕ), 4 * m * n - m - n ≠ p^2) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l281_28147


namespace NUMINAMATH_CALUDE_fort_block_count_l281_28108

/-- Calculates the number of one-foot cubical blocks required to construct a rectangular fort -/
def fort_blocks (length width height : ℕ) (wall_thickness : ℕ) : ℕ :=
  length * width * height - 
  (length - 2 * wall_thickness) * (width - 2 * wall_thickness) * (height - wall_thickness)

/-- Proves that a fort with given dimensions requires 430 blocks -/
theorem fort_block_count : fort_blocks 15 12 6 1 = 430 := by
  sorry

end NUMINAMATH_CALUDE_fort_block_count_l281_28108


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l281_28181

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 10*x^2 + 28*x > 0 ↔ (x > 0 ∧ x < 4) ∨ x > 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l281_28181


namespace NUMINAMATH_CALUDE_dress_cost_difference_l281_28119

theorem dress_cost_difference (patty ida jean pauline : ℕ) : 
  patty = ida + 10 →
  ida = jean + 30 →
  jean < pauline →
  pauline = 30 →
  patty + ida + jean + pauline = 160 →
  pauline - jean = 10 := by
sorry

end NUMINAMATH_CALUDE_dress_cost_difference_l281_28119


namespace NUMINAMATH_CALUDE_integer_remainder_properties_l281_28153

theorem integer_remainder_properties (n : ℤ) (h : n % 20 = 13) :
  (n % 4 + n % 5 = 4) ∧ n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_remainder_properties_l281_28153


namespace NUMINAMATH_CALUDE_sum_of_cubes_theorem_l281_28172

theorem sum_of_cubes_theorem (a b : ℤ) : 
  a * b = 12 → a^3 + b^3 = 91 → a^3 + b^3 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_theorem_l281_28172


namespace NUMINAMATH_CALUDE_exists_divisible_by_11_in_39_consecutive_integers_l281_28106

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_divisible_by_11_in_39_consecutive_integers :
  ∀ (start : ℕ), ∃ (k : ℕ), k ∈ Finset.range 39 ∧ (sumOfDigits (start + k) % 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_11_in_39_consecutive_integers_l281_28106


namespace NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l281_28164

theorem cos_36_minus_cos_72_eq_half :
  Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l281_28164


namespace NUMINAMATH_CALUDE_paperclip_capacity_l281_28118

/-- Given a box of volume 16 cm³ that holds 50 paperclips, 
    prove that a box of volume 48 cm³ will hold 150 paperclips, 
    assuming a direct proportion between volume and paperclip capacity. -/
theorem paperclip_capacity (v₁ v₂ : ℝ) (c₁ c₂ : ℕ) :
  v₁ = 16 → v₂ = 48 → c₁ = 50 →
  (v₁ * c₂ = v₂ * c₁) →
  c₂ = 150 := by
  sorry

#check paperclip_capacity

end NUMINAMATH_CALUDE_paperclip_capacity_l281_28118


namespace NUMINAMATH_CALUDE_sum_due_calculation_l281_28140

/-- The relationship between banker's discount, true discount, and sum due -/
def banker_discount_relation (bd td sd : ℝ) : Prop :=
  bd = td + td^2 / sd

/-- The problem statement -/
theorem sum_due_calculation (bd td : ℝ) (h1 : bd = 36) (h2 : td = 30) :
  ∃ sd : ℝ, banker_discount_relation bd td sd ∧ sd = 150 := by
  sorry

end NUMINAMATH_CALUDE_sum_due_calculation_l281_28140


namespace NUMINAMATH_CALUDE_parallelogram_area_l281_28157

/-- The area of a parallelogram with base 24 cm and height 16 cm is 384 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 24 → height = 16 → area = base * height → area = 384 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l281_28157


namespace NUMINAMATH_CALUDE_paths_count_is_40_l281_28130

/-- Represents the arrangement of letters and numerals --/
structure Arrangement where
  centralA : Unit
  adjacentM : Fin 4
  adjacentC : Fin 4 → Fin 3
  adjacent1 : Unit
  adjacent0 : Fin 2

/-- Counts the number of paths to spell AMC10 in the given arrangement --/
def countPaths (arr : Arrangement) : ℕ :=
  let pathsFromM (m : Fin 4) := arr.adjacentC m * 1 * 2
  (pathsFromM 0 + pathsFromM 1 + pathsFromM 2 + pathsFromM 3)

/-- The theorem stating that the number of paths is 40 --/
theorem paths_count_is_40 (arr : Arrangement) : countPaths arr = 40 := by
  sorry

#check paths_count_is_40

end NUMINAMATH_CALUDE_paths_count_is_40_l281_28130


namespace NUMINAMATH_CALUDE_people_not_playing_sports_l281_28165

theorem people_not_playing_sports (total_people : ℕ) (tennis_players : ℕ) (baseball_players : ℕ) (both_players : ℕ) :
  total_people = 310 →
  tennis_players = 138 →
  baseball_players = 255 →
  both_players = 94 →
  total_people - (tennis_players + baseball_players - both_players) = 11 :=
by sorry

end NUMINAMATH_CALUDE_people_not_playing_sports_l281_28165


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l281_28125

theorem diophantine_equation_solutions :
  {(a, b) : ℕ × ℕ | 12 * a + 11 * b = 2002} =
    {(11, 170), (22, 158), (33, 146), (44, 134), (55, 122), (66, 110),
     (77, 98), (88, 86), (99, 74), (110, 62), (121, 50), (132, 38),
     (143, 26), (154, 14), (165, 2)} := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l281_28125


namespace NUMINAMATH_CALUDE_woman_lawyer_probability_l281_28199

/-- Represents a study group with given proportions of women and lawyers --/
structure StudyGroup where
  total_members : ℕ
  women_percentage : ℝ
  lawyer_percentage : ℝ
  women_percentage_valid : 0 ≤ women_percentage ∧ women_percentage ≤ 1
  lawyer_percentage_valid : 0 ≤ lawyer_percentage ∧ lawyer_percentage ≤ 1

/-- Calculates the probability of selecting a woman lawyer from the study group --/
def probability_woman_lawyer (group : StudyGroup) : ℝ :=
  group.women_percentage * group.lawyer_percentage

/-- Theorem stating that the probability of selecting a woman lawyer is 0.32 
    given the specified conditions --/
theorem woman_lawyer_probability (group : StudyGroup) 
  (h1 : group.women_percentage = 0.8) 
  (h2 : group.lawyer_percentage = 0.4) : 
  probability_woman_lawyer group = 0.32 := by
  sorry


end NUMINAMATH_CALUDE_woman_lawyer_probability_l281_28199


namespace NUMINAMATH_CALUDE_binary_101101110_equals_octal_556_l281_28146

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_natural (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to its octal representation as a list of digits. -/
def natural_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: natural_to_octal (n / 8)

theorem binary_101101110_equals_octal_556 :
  let binary : List Bool := [true, false, true, true, false, true, true, true, false]
  let octal : List ℕ := [6, 5, 5]
  binary_to_natural binary = (natural_to_octal (binary_to_natural binary)).reverse.foldl (fun acc d => acc * 8 + d) 0 ∧
  natural_to_octal (binary_to_natural binary) = octal.reverse :=
by sorry

end NUMINAMATH_CALUDE_binary_101101110_equals_octal_556_l281_28146


namespace NUMINAMATH_CALUDE_star_sum_squared_l281_28168

/-- The star operation defined on real numbers -/
def star (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating the result of (x+y)² ⋆ (y+x)² -/
theorem star_sum_squared (x y : ℝ) : star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end NUMINAMATH_CALUDE_star_sum_squared_l281_28168


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l281_28174

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  is_geometric a → a 1 = 8 → a 2 * a 3 = -8 → a 4 = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l281_28174


namespace NUMINAMATH_CALUDE_science_club_neither_subject_l281_28171

theorem science_club_neither_subject (total : ℕ) (chemistry : ℕ) (biology : ℕ) (both : ℕ) 
  (h1 : total = 75)
  (h2 : chemistry = 42)
  (h3 : biology = 33)
  (h4 : both = 18) :
  total - (chemistry + biology - both) = 18 := by
  sorry

end NUMINAMATH_CALUDE_science_club_neither_subject_l281_28171


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l281_28152

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball placement process described in the problem -/
def ballPlacementProcess (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of balls after n steps is equal to
    the sum of digits in the base 6 representation of n -/
theorem ball_placement_theorem (n : ℕ) :
  ballPlacementProcess n = sumDigits (toBase6 n) :=
  sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l281_28152


namespace NUMINAMATH_CALUDE_box_volume_solutions_l281_28135

def box_volume (x : ℤ) : ℤ :=
  (x + 3) * (x - 3) * (x^3 - 5*x + 25)

def satisfies_condition (x : ℤ) : Prop :=
  x > 0 ∧ box_volume x < 1500

theorem box_volume_solutions :
  (∃ (S : Finset ℤ), (∀ x ∈ S, satisfies_condition x) ∧
                     (∀ x : ℤ, satisfies_condition x → x ∈ S) ∧
                     Finset.card S = 4) := by
  sorry

end NUMINAMATH_CALUDE_box_volume_solutions_l281_28135


namespace NUMINAMATH_CALUDE_min_cost_for_nine_hamburgers_l281_28109

/-- Represents the cost of hamburgers under a "buy two, get one free" promotion -/
def hamburger_cost (unit_price : ℕ) (quantity : ℕ) : ℕ :=
  let sets := quantity / 3
  let remainder := quantity % 3
  sets * (2 * unit_price) + remainder * unit_price

/-- Theorem stating the minimum cost for 9 hamburgers under the given promotion -/
theorem min_cost_for_nine_hamburgers :
  hamburger_cost 10 9 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_nine_hamburgers_l281_28109


namespace NUMINAMATH_CALUDE_candy_bar_profit_l281_28191

theorem candy_bar_profit :
  let total_bars : ℕ := 1200
  let buy_price : ℚ := 3 / 4
  let sell_price : ℚ := 2 / 3
  let discount_threshold : ℕ := 1000
  let discount_per_bar : ℚ := 1 / 10
  let cost : ℚ := total_bars * buy_price
  let revenue_before_discount : ℚ := total_bars * sell_price
  let discounted_bars : ℕ := total_bars - discount_threshold
  let discount : ℚ := discounted_bars * discount_per_bar
  let revenue_after_discount : ℚ := revenue_before_discount - discount
  let profit : ℚ := revenue_after_discount - cost
  profit = -116
:= by sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l281_28191


namespace NUMINAMATH_CALUDE_min_sum_fraction_min_sum_fraction_achievable_l281_28141

theorem min_sum_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 1 / (2 * Real.rpow 2 (1/3)) :=
sorry

theorem min_sum_fraction_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    a / (3 * b) + b / (6 * c) + c / (9 * a) = 1 / (2 * Real.rpow 2 (1/3)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_fraction_min_sum_fraction_achievable_l281_28141


namespace NUMINAMATH_CALUDE_pyramid_solution_l281_28169

/-- Represents a pyramid of numbers --/
structure Pyramid :=
  (bottom_row : List ℝ)
  (is_valid : bottom_row.length = 4)

/-- Checks if a pyramid satisfies the given conditions --/
def satisfies_conditions (p : Pyramid) : Prop :=
  ∃ x : ℝ,
    p.bottom_row = [13, x, 11, 2*x] ∧
    (13 + x) + (11 + 2*x) = 42

/-- The main theorem to prove --/
theorem pyramid_solution {p : Pyramid} (h : satisfies_conditions p) :
  ∃ x : ℝ, x = 6 ∧ p.bottom_row = [13, x, 11, 2*x] := by
  sorry

end NUMINAMATH_CALUDE_pyramid_solution_l281_28169


namespace NUMINAMATH_CALUDE_distinct_committees_l281_28149

/-- The number of teams in the volleyball league -/
def numTeams : ℕ := 5

/-- The number of players in each team -/
def playersPerTeam : ℕ := 8

/-- The number of committee members selected from the host team -/
def hostCommitteeMembers : ℕ := 4

/-- The number of committee members selected from each non-host team -/
def nonHostCommitteeMembers : ℕ := 1

/-- The total number of distinct tournament committees over one complete rotation -/
def totalCommittees : ℕ := numTeams * (Nat.choose playersPerTeam hostCommitteeMembers) * (playersPerTeam ^ (numTeams - 1))

theorem distinct_committees :
  totalCommittees = 1433600 := by
  sorry

end NUMINAMATH_CALUDE_distinct_committees_l281_28149


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_tenth_polygon_l281_28161

/-- The number of sides of the nth polygon in the sequence -/
def sides (n : ℕ) : ℕ := n + 2

/-- The sum of interior angles of a polygon with n sides -/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The 10th polygon in the sequence -/
def tenth_polygon : ℕ := 10

theorem sum_of_interior_angles_tenth_polygon :
  interior_angle_sum (sides tenth_polygon) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_tenth_polygon_l281_28161


namespace NUMINAMATH_CALUDE_abs_plus_one_nonzero_l281_28193

theorem abs_plus_one_nonzero (a : ℚ) : |a| + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_one_nonzero_l281_28193


namespace NUMINAMATH_CALUDE_hex_tile_difference_specific_hex_tile_difference_l281_28144

/-- Represents a hexagonal tile arrangement with blue and green tiles -/
structure HexTileArrangement where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Adds a border of hexagonal tiles around an existing arrangement -/
def add_border (arrangement : HexTileArrangement) (border_color : String) : HexTileArrangement :=
  match border_color with
  | "green" => { blue_tiles := arrangement.blue_tiles, 
                 green_tiles := arrangement.green_tiles + (arrangement.blue_tiles + arrangement.green_tiles) / 2 }
  | "blue" => { blue_tiles := arrangement.blue_tiles + (arrangement.blue_tiles + arrangement.green_tiles) / 2 + 3, 
                green_tiles := arrangement.green_tiles }
  | _ => arrangement

/-- The main theorem stating the difference in tile counts after adding two borders -/
theorem hex_tile_difference (initial : HexTileArrangement) :
  let with_green_border := add_border initial "green"
  let final := add_border with_green_border "blue"
  final.blue_tiles - final.green_tiles = 16 :=
by
  sorry

/-- The specific instance of the hexagonal tile arrangement -/
def initial_arrangement : HexTileArrangement := { blue_tiles := 20, green_tiles := 10 }

/-- Applying the theorem to the specific instance -/
theorem specific_hex_tile_difference :
  let with_green_border := add_border initial_arrangement "green"
  let final := add_border with_green_border "blue"
  final.blue_tiles - final.green_tiles = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_hex_tile_difference_specific_hex_tile_difference_l281_28144


namespace NUMINAMATH_CALUDE_replaced_student_weight_l281_28101

/-- Given 5 students, if replacing one student with a 72 kg student causes
    the average weight to decrease by 4 kg, then the replaced student's weight was 92 kg. -/
theorem replaced_student_weight
  (n : ℕ) -- number of students
  (old_avg : ℝ) -- original average weight
  (new_avg : ℝ) -- new average weight after replacement
  (new_student_weight : ℝ) -- weight of the new student
  (h1 : n = 5) -- there are 5 students
  (h2 : new_avg = old_avg - 4) -- average weight decreases by 4 kg
  (h3 : new_student_weight = 72) -- new student weighs 72 kg
  : n * old_avg - (n * new_avg + new_student_weight) = 92 := by
  sorry

end NUMINAMATH_CALUDE_replaced_student_weight_l281_28101


namespace NUMINAMATH_CALUDE_two_digit_prime_sum_20180500_prime_l281_28132

theorem two_digit_prime_sum_20180500_prime (n : ℕ) : 
  (10 ≤ n ∧ n < 100) →  -- n is a two-digit number
  Nat.Prime n →         -- n is prime
  Nat.Prime (n + 20180500) → -- n + 20180500 is prime
  n = 61 := by
sorry

end NUMINAMATH_CALUDE_two_digit_prime_sum_20180500_prime_l281_28132


namespace NUMINAMATH_CALUDE_cos_750_degrees_l281_28111

theorem cos_750_degrees : Real.cos (750 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_750_degrees_l281_28111


namespace NUMINAMATH_CALUDE_burger_cost_is_five_l281_28163

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℝ := 4

/-- The cost of a smoothie in dollars -/
def smoothie_cost : ℝ := 4

/-- The number of smoothies ordered -/
def num_smoothies : ℕ := 2

/-- The total cost of the order in dollars -/
def total_cost : ℝ := 17

/-- The cost of the burger in dollars -/
def burger_cost : ℝ := total_cost - (sandwich_cost + num_smoothies * smoothie_cost)

theorem burger_cost_is_five :
  burger_cost = 5 := by sorry

end NUMINAMATH_CALUDE_burger_cost_is_five_l281_28163


namespace NUMINAMATH_CALUDE_congruence_solution_l281_28116

theorem congruence_solution (n : ℤ) : 
  (13 * n) % 47 = 9 % 47 ↔ n % 47 = 39 % 47 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l281_28116
