import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2449_244983

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where a_2 + a_6 = 10, prove that a_4 = 5. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 2 + a 6 = 10) : 
  a 4 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2449_244983


namespace NUMINAMATH_CALUDE_calculation_proof_l2449_244916

theorem calculation_proof : 
  Real.rpow 27 (1/3) + (Real.sqrt 2 - 1)^2 - (1/2)⁻¹ + 2 / (Real.sqrt 2 - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2449_244916


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l2449_244956

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define the problem
theorem circle_and_tangent_lines 
  (C : Circle) 
  (h1 : (0, -6) ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2})
  (h2 : (1, -5) ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2})
  (h3 : C.center ∈ Line 1 (-1) 1) :
  (∀ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 25 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2}) ∧
  (∀ x y : ℝ, (x = 2 ∨ 3*x - 4*y + 26 = 0) ↔ 
    ((x, y) ∈ Line 1 (-1) (-2) ∧ 
     ((x - 2)^2 + (y - 8)^2) * C.radius^2 = ((x - C.center.1)^2 + (y - C.center.2)^2) * ((2 - C.center.1)^2 + (8 - C.center.2)^2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l2449_244956


namespace NUMINAMATH_CALUDE_derived_function_coefficients_target_point_coords_two_base_points_and_distance_range_l2449_244992

/-- Definition of a derived function -/
def is_derived_function (a b c : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
  (a * x₁ + b = c / x₂) ∧
  (x₁ = -x₂)

/-- Part 1: Derived function coefficients -/
theorem derived_function_coefficients :
  is_derived_function 2 4 5 := by sorry

/-- Part 2: Target point coordinates -/
theorem target_point_coords (b c : ℝ) :
  is_derived_function 1 b c →
  (∃ (x : ℝ), x^2 + b*x + c = 0) →
  (1 + b = -c) →
  (∃ (x y : ℝ), x = -1 ∧ y = -1 ∧ y = c / x) := by sorry

/-- Part 3: Existence of two base points and their distance range -/
theorem two_base_points_and_distance_range (a b : ℝ) :
  a > b ∧ b > 0 →
  is_derived_function a (2*b) (-2) →
  (∃ (x : ℝ), a*x^2 + 2*b*x - 2 = 6) →
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a*x₁ + 2*b = a*x₁^2 + 2*b*x₁ - 2 ∧ a*x₂ + 2*b = a*x₂^2 + 2*b*x₂ - 2) ∧
  (∃ (x₁ x₂ : ℝ), 2 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_derived_function_coefficients_target_point_coords_two_base_points_and_distance_range_l2449_244992


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2449_244928

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z + x * y + y * z + z * x = 4) :
  Real.sqrt ((x * y + x + y) / z) + Real.sqrt ((y * z + y + z) / x) + Real.sqrt ((z * x + z + x) / y) ≥ 
  3 * Real.sqrt (3 * (x + 2) * (y + 2) * (z + 2) / ((2 * x + 1) * (2 * y + 1) * (2 * z + 1))) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2449_244928


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_values_l2449_244915

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 ≠ 1}

-- State the theorem
theorem set_inclusion_implies_a_values :
  ∀ a : ℝ, (B a ⊆ A) ↔ (a ≤ -1 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_values_l2449_244915


namespace NUMINAMATH_CALUDE_ratio_last_year_ratio_future_ronaldo_is_36_l2449_244931

/-- Ronaldo's age one year ago -/
def ronaldo_age_last_year : ℕ := 35

/-- Roonie's age one year ago -/
def roonie_age_last_year : ℕ := 30

/-- The ratio of Roonie's to Ronaldo's age one year ago -/
theorem ratio_last_year : 
  roonie_age_last_year / ronaldo_age_last_year = 6 / 7 := by sorry

/-- The ratio of their ages four years from now -/
theorem ratio_future : 
  (roonie_age_last_year + 5) / (ronaldo_age_last_year + 5) = 7 / 8 := by sorry

/-- Ronaldo's current age -/
def ronaldo_current_age : ℕ := ronaldo_age_last_year + 1

/-- Proof that Ronaldo's current age is 36 -/
theorem ronaldo_is_36 : ronaldo_current_age = 36 := by sorry

end NUMINAMATH_CALUDE_ratio_last_year_ratio_future_ronaldo_is_36_l2449_244931


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2449_244974

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a line given by the equation (3+k)x + (1-2k)y + 1 + 5k = 0 -/
def lies_on_line (p : Point) (k : ℝ) : Prop :=
  (3 + k) * p.x + (1 - 2*k) * p.y + 1 + 5*k = 0

/-- The theorem stating that (-1, 2) is the unique fixed point for all lines -/
theorem fixed_point_theorem :
  ∃! p : Point, ∀ k : ℝ, lies_on_line p k :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2449_244974


namespace NUMINAMATH_CALUDE_myrtle_egg_count_l2449_244993

/-- The number of eggs Myrtle has after her trip -/
def myrtle_eggs (num_hens : ℕ) (eggs_per_hen : ℕ) (days_gone : ℕ) (neighbor_took : ℕ) (eggs_dropped : ℕ) : ℕ :=
  num_hens * eggs_per_hen * days_gone - neighbor_took - eggs_dropped

/-- Proof that Myrtle has 46 eggs given the conditions -/
theorem myrtle_egg_count : myrtle_eggs 3 3 7 12 5 = 46 := by
  sorry

end NUMINAMATH_CALUDE_myrtle_egg_count_l2449_244993


namespace NUMINAMATH_CALUDE_water_polo_team_selection_l2449_244975

theorem water_polo_team_selection (total_members : ℕ) (starting_team_size : ℕ) (goalie_count : ℕ) :
  total_members = 18 →
  starting_team_size = 8 →
  goalie_count = 1 →
  (total_members.choose goalie_count) * ((total_members - goalie_count).choose (starting_team_size - goalie_count)) = 222768 :=
by sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_l2449_244975


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2449_244939

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ ∀ x > 0, ¬ P x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x > 0, x^2 - x ≤ 0) ↔ (∀ x > 0, x^2 - x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2449_244939


namespace NUMINAMATH_CALUDE_equation_solution_l2449_244935

theorem equation_solution : 
  ∃ (x : ℚ), (x + 2) / 4 - 1 = (2 * x + 1) / 3 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2449_244935


namespace NUMINAMATH_CALUDE_correct_multiplication_l2449_244918

theorem correct_multiplication (x : ℚ) : 14 * x = 42 → 12 * x = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplication_l2449_244918


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l2449_244929

theorem smallest_five_digit_multiple_of_18 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 18 ∣ n → 10008 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l2449_244929


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2449_244938

theorem complex_expression_simplification (x : ℝ) :
  x * (x * (x * (3 - x) - 3) + 5) + 1 = -x^4 + 3*x^3 - 3*x^2 + 5*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2449_244938


namespace NUMINAMATH_CALUDE_fifteenth_digit_of_sum_one_seventh_one_eleventh_l2449_244995

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sum_decimal_representations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in a decimal representation -/
def nth_digit_after_decimal (rep : ℕ → ℕ) (n : ℕ) : ℕ := sorry

theorem fifteenth_digit_of_sum_one_seventh_one_eleventh :
  nth_digit_after_decimal (sum_decimal_representations (1/7) (1/11)) 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_digit_of_sum_one_seventh_one_eleventh_l2449_244995


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l2449_244948

theorem more_girls_than_boys (girls boys : ℝ) (h1 : girls = 542.0) (h2 : boys = 387.0) :
  girls - boys = 155.0 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l2449_244948


namespace NUMINAMATH_CALUDE_paint_jar_capacity_l2449_244968

theorem paint_jar_capacity (mary_dragon : ℝ) (mike_castle : ℝ) (sun : ℝ) 
  (h1 : mary_dragon = 3)
  (h2 : mike_castle = mary_dragon + 2)
  (h3 : sun = 5) :
  mary_dragon + mike_castle + sun = 13 := by
  sorry

end NUMINAMATH_CALUDE_paint_jar_capacity_l2449_244968


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_45_l2449_244960

-- Define a function to represent a four-digit number of the form a43b
def number (a b : Nat) : Nat := a * 1000 + 430 + b

-- Define the divisibility condition
def isDivisibleBy45 (n : Nat) : Prop := n % 45 = 0

-- State the theorem
theorem four_digit_divisible_by_45 :
  ∀ a b : Nat, a < 10 ∧ b < 10 →
    (isDivisibleBy45 (number a b) ↔ (a = 2 ∧ b = 0) ∨ (a = 6 ∧ b = 5)) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_45_l2449_244960


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_repeats_eq_252_l2449_244926

/-- The number of three-digit numbers with repeated digits using digits 0 to 9 -/
def three_digit_numbers_with_repeats : ℕ :=
  let total_three_digit_numbers := 9 * 10 * 10  -- First digit can't be 0
  let three_digit_numbers_without_repeats := 9 * 9 * 8
  total_three_digit_numbers - three_digit_numbers_without_repeats

/-- Theorem stating that the number of three-digit numbers with repeated digits is 252 -/
theorem three_digit_numbers_with_repeats_eq_252 : 
  three_digit_numbers_with_repeats = 252 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_repeats_eq_252_l2449_244926


namespace NUMINAMATH_CALUDE_first_number_is_1841_l2449_244910

/-- Represents one operation of replacing the first number with the average of the other two -/
def operation (x y z : ℤ) : ℤ × ℤ × ℤ := (y, z, (y + z) / 2)

/-- Applies the operation n times -/
def apply_operations (n : ℕ) (x y z : ℤ) : ℤ × ℤ × ℤ :=
  match n with
  | 0 => (x, y, z)
  | n + 1 => 
    let (a, b, c) := apply_operations n x y z
    operation a b c

theorem first_number_is_1841 (a b c : ℤ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ -- all numbers are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ -- all numbers are different
  a + b + c = 2013 ∧ -- initial sum
  (let (x, y, z) := apply_operations 7 a b c; x + y + z = 195) → -- sum after 7 operations
  a = 1841 := by sorry

end NUMINAMATH_CALUDE_first_number_is_1841_l2449_244910


namespace NUMINAMATH_CALUDE_triangle_side_length_l2449_244954

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 2)
  (h2 : A = π / 6)  -- 30° in radians
  (h3 : C = 3 * π / 4)  -- 135° in radians
  (h4 : A + B + C = π)  -- sum of angles in a triangle
  (h5 : a / Real.sin A = b / Real.sin B)  -- Law of Sines
  : b = (Real.sqrt 2 - Real.sqrt 6) / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2449_244954


namespace NUMINAMATH_CALUDE_subtract_three_from_binary_l2449_244978

/-- Converts a binary number (represented as a list of bits) to decimal --/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts a decimal number to binary (represented as a list of bits) --/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2) :: acc)
  aux n []

theorem subtract_three_from_binary :
  let M : List Nat := [0, 1, 0, 1, 0, 1]  -- 101010 in binary
  let M_decimal : Nat := binary_to_decimal M
  let result : List Nat := decimal_to_binary (M_decimal - 3)
  result = [1, 1, 1, 0, 0, 1] -- 100111 in binary
  := by sorry

end NUMINAMATH_CALUDE_subtract_three_from_binary_l2449_244978


namespace NUMINAMATH_CALUDE_shooting_probability_l2449_244934

/-- The probability of person A hitting the target -/
def prob_A : ℚ := 3/4

/-- The probability of person B hitting the target -/
def prob_B : ℚ := 4/5

/-- The probability of the event where A has shot twice when they stop -/
def prob_A_shoots_twice : ℚ := 19/400

theorem shooting_probability :
  let prob_A_miss := 1 - prob_A
  let prob_B_miss := 1 - prob_B
  prob_A_shoots_twice = 
    (prob_A_miss * prob_B_miss * prob_A) + 
    (prob_A_miss * prob_B_miss * prob_A_miss * prob_B) :=
by sorry

end NUMINAMATH_CALUDE_shooting_probability_l2449_244934


namespace NUMINAMATH_CALUDE_circle_product_values_l2449_244922

noncomputable section

open Real Set

def circle_product (α β : ℝ × ℝ) : ℝ := 
  (α.1 * β.1 + α.2 * β.2) / (β.1 * β.1 + β.2 * β.2)

def angle (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem circle_product_values (a b : ℝ × ℝ) 
  (h1 : a ≠ (0, 0)) 
  (h2 : b ≠ (0, 0)) 
  (h3 : π/6 < angle a b ∧ angle a b < π/2) 
  (h4 : ∃ n : ℤ, circle_product a b = n/2) 
  (h5 : ∃ m : ℤ, circle_product b a = m/2) : 
  circle_product a b = 1 ∨ circle_product a b = 1/2 := by
sorry

end

end NUMINAMATH_CALUDE_circle_product_values_l2449_244922


namespace NUMINAMATH_CALUDE_rental_cost_calculation_l2449_244961

/-- Calculates the total cost of renting a truck given the daily rate, mileage rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that the total rental cost for the given conditions is $230. -/
theorem rental_cost_calculation :
  let daily_rate : ℚ := 35
  let mileage_rate : ℚ := 1/4
  let days : ℕ := 3
  let miles : ℕ := 500
  total_rental_cost daily_rate mileage_rate days miles = 230 := by
sorry


end NUMINAMATH_CALUDE_rental_cost_calculation_l2449_244961


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2449_244920

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 86 ∧ (13605 - x) % 87 = 0 ∧ ∀ y : ℕ, y < x → (13605 - y) % 87 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2449_244920


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_rotated_triangle_l2449_244997

/-- The lateral surface area of a cone formed by rotating a right-angled triangle -/
theorem lateral_surface_area_of_rotated_triangle (AC BC : ℝ) (h_right_angle : AC * AC + BC * BC = 5 * 5) :
  AC = 3 → BC = 4 → π * BC * 5 = 20 * π := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_rotated_triangle_l2449_244997


namespace NUMINAMATH_CALUDE_max_value_expression_l2449_244913

theorem max_value_expression (a b c d : ℝ) 
  (ha : -8.5 ≤ a ∧ a ≤ 8.5)
  (hb : -8.5 ≤ b ∧ b ≤ 8.5)
  (hc : -8.5 ≤ c ∧ c ≤ 8.5)
  (hd : -8.5 ≤ d ∧ d ≤ 8.5) :
  ∃ (m : ℝ), m = 306 ∧ 
  ∀ (a' b' c' d' : ℝ), 
    -8.5 ≤ a' ∧ a' ≤ 8.5 → 
    -8.5 ≤ b' ∧ b' ≤ 8.5 → 
    -8.5 ≤ c' ∧ c' ≤ 8.5 → 
    -8.5 ≤ d' ∧ d' ≤ 8.5 → 
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2449_244913


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2449_244904

theorem complex_fraction_simplification :
  ∃ (i : ℂ), i * i = -1 → (5 * i) / (1 - 2 * i) = -2 + i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2449_244904


namespace NUMINAMATH_CALUDE_smallest_w_l2449_244965

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) (hw : w > 0) 
  (h1 : is_factor (2^5) (936 * w))
  (h2 : is_factor (3^3) (936 * w))
  (h3 : is_factor (10^2) (936 * w)) :
  w ≥ 900 ∧ ∃ w', w' = 900 ∧ w' > 0 ∧ 
    is_factor (2^5) (936 * w') ∧ 
    is_factor (3^3) (936 * w') ∧ 
    is_factor (10^2) (936 * w') :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l2449_244965


namespace NUMINAMATH_CALUDE_equation_solution_l2449_244901

theorem equation_solution : ∃ (x y z : ℝ), 
  2 * Real.sqrt (x - 4) + 3 * Real.sqrt (y - 9) + 4 * Real.sqrt (z - 16) = (1/2) * (x + y + z) ∧
  x = 8 ∧ y = 18 ∧ z = 32 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2449_244901


namespace NUMINAMATH_CALUDE_basketball_lineup_selection_l2449_244923

theorem basketball_lineup_selection (n m k : ℕ) (hn : n = 12) (hm : m = 5) (hk : k = 1) :
  n * Nat.choose (n - k) (m - k) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_selection_l2449_244923


namespace NUMINAMATH_CALUDE_kyle_lifts_320_l2449_244933

/-- Kyle's lifting capacity over the years -/
structure KylesLift where
  two_years_ago : ℝ
  last_year : ℝ
  this_year : ℝ

/-- Given information about Kyle's lifting capacity -/
def kyle_info (k : KylesLift) : Prop :=
  k.this_year = 1.6 * k.last_year ∧
  0.6 * k.last_year = 3 * k.two_years_ago ∧
  k.two_years_ago = 40

/-- Theorem: Kyle can lift 320 pounds this year -/
theorem kyle_lifts_320 (k : KylesLift) (h : kyle_info k) : k.this_year = 320 := by
  sorry


end NUMINAMATH_CALUDE_kyle_lifts_320_l2449_244933


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l2449_244953

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem monotonic_increase_interval (x : ℝ) :
  StrictMonoOn f (Set.Ici 2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l2449_244953


namespace NUMINAMATH_CALUDE_sum_of_e_and_f_l2449_244936

theorem sum_of_e_and_f (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 5.2)
  (h2 : (c + d) / 2 = 5.8)
  (h3 : (a + b + c + d + e + f) / 6 = 5.4) :
  e + f = 10.4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_e_and_f_l2449_244936


namespace NUMINAMATH_CALUDE_parallel_vectors_iff_k_eq_neg_two_l2449_244985

-- Define the vectors
def a : Fin 2 → ℝ := ![1, -2]
def b (k : ℝ) : Fin 2 → ℝ := ![k, 4]

-- Define parallel vectors
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ (∀ i, u i = c * v i)

-- Theorem statement
theorem parallel_vectors_iff_k_eq_neg_two :
  parallel a (b k) ↔ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_iff_k_eq_neg_two_l2449_244985


namespace NUMINAMATH_CALUDE_complex_equation_imag_part_l2449_244972

theorem complex_equation_imag_part :
  ∀ z : ℂ, z * (1 + Complex.I) = (3 : ℂ) + 2 * Complex.I →
  Complex.im z = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_imag_part_l2449_244972


namespace NUMINAMATH_CALUDE_relationship_abc_l2449_244991

theorem relationship_abc (a b c : ℝ) (ha : a = Real.exp 0.3) (hb : b = 0.9^2) (hc : c = Real.log 0.9) :
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2449_244991


namespace NUMINAMATH_CALUDE_all_propositions_false_l2449_244900

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel_lines
local infix:50 " ∥ " => parallel_line_plane
local infix:50 " ⊂ " => line_in_plane

-- Theorem statement
theorem all_propositions_false :
  (∀ (a b : Line) (α : Plane), (a ∥ b) → (b ⊂ α) → (a ∥ α)) = False ∧
  (∀ (a b : Line) (α : Plane), (a ∥ α) → (b ∥ α) → (a ∥ b)) = False ∧
  (∀ (a b : Line) (α : Plane), (a ∥ b) → (b ∥ α) → (a ∥ α)) = False ∧
  (∀ (a b : Line) (α : Plane), (a ∥ α) → (b ⊂ α) → (a ∥ b)) = False :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l2449_244900


namespace NUMINAMATH_CALUDE_smallest_batch_size_l2449_244962

theorem smallest_batch_size (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ (21 * N)) : 
  (∀ M : ℕ, M > 70 ∧ 70 ∣ (21 * M) → N ≤ M) → N = 80 :=
by sorry

end NUMINAMATH_CALUDE_smallest_batch_size_l2449_244962


namespace NUMINAMATH_CALUDE_remaining_kittens_l2449_244937

def initial_kittens : ℕ := 8
def given_away : ℕ := 2

theorem remaining_kittens : initial_kittens - given_away = 6 := by
  sorry

end NUMINAMATH_CALUDE_remaining_kittens_l2449_244937


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2449_244952

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 225) : x + y = 650 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2449_244952


namespace NUMINAMATH_CALUDE_simplify_expression_l2449_244979

theorem simplify_expression (a b : ℝ) : (25*a + 70*b) + (15*a + 34*b) - (12*a + 55*b) = 28*a + 49*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2449_244979


namespace NUMINAMATH_CALUDE_solve_movie_problem_l2449_244943

def movie_problem (regular_price child_discount adults_count money_given change : ℕ) : Prop :=
  let child_price := regular_price - child_discount
  let total_spent := money_given - change
  let adults_cost := adults_count * regular_price
  let children_cost := total_spent - adults_cost
  ∃ (children_count : ℕ), children_count * child_price = children_cost

theorem solve_movie_problem :
  movie_problem 9 2 2 40 1 = ∃ (children_count : ℕ), children_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_movie_problem_l2449_244943


namespace NUMINAMATH_CALUDE_shaded_area_proof_l2449_244966

theorem shaded_area_proof (t : ℝ) (h : t = 5) : 
  let larger_side := 2 * t - 4
  let smaller_side := 4
  (larger_side ^ 2) - (smaller_side ^ 2) = 20 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l2449_244966


namespace NUMINAMATH_CALUDE_vertex_C_coordinates_l2449_244921

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the median CM and altitude BH
def median_CM (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - p.2 - 5 = 0

def altitude_BH (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => p.1 - 2 * p.2 - 5 = 0

-- Theorem statement
theorem vertex_C_coordinates (t : Triangle) :
  t.A = (5, 1) →
  median_CM t t.C →
  altitude_BH t t.B →
  (t.C.1 - t.A.1) * (t.B.1 - t.C.1) + (t.C.2 - t.A.2) * (t.B.2 - t.C.2) = 0 →
  t.C = (4, 3) := by
  sorry

end NUMINAMATH_CALUDE_vertex_C_coordinates_l2449_244921


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2449_244958

/-- A geometric sequence with its third term and sum of first three terms given -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  is_geometric : ∀ n, a (n + 1) = q * a n
  third_term : a 3 = 3/2
  third_sum : (a 1) + (a 2) + (a 3) = 9/2

/-- The common ratio of the geometric sequence is either 1 or -1/2 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : 
  seq.q = 1 ∨ seq.q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2449_244958


namespace NUMINAMATH_CALUDE_fraction_equality_l2449_244964

theorem fraction_equality : (3 * 4 * 5) / (2 * 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2449_244964


namespace NUMINAMATH_CALUDE_banana_cream_pie_slice_degrees_l2449_244999

/-- Proves that the banana cream pie slice in a pie chart is 44° given the classroom preferences. -/
theorem banana_cream_pie_slice_degrees (total_students : ℕ) (strawberry_pref : ℕ) (pecan_pref : ℕ) (pumpkin_pref : ℕ) 
  (h_total : total_students = 45)
  (h_strawberry : strawberry_pref = 15)
  (h_pecan : pecan_pref = 10)
  (h_pumpkin : pumpkin_pref = 9)
  (h_remaining : (total_students - (strawberry_pref + pecan_pref + pumpkin_pref)) % 2 = 0) :
  (((total_students - (strawberry_pref + pecan_pref + pumpkin_pref)) / 2 : ℚ) / total_students) * 360 = 44 := by
  sorry

#check banana_cream_pie_slice_degrees

end NUMINAMATH_CALUDE_banana_cream_pie_slice_degrees_l2449_244999


namespace NUMINAMATH_CALUDE_even_product_implies_even_factor_l2449_244932

theorem even_product_implies_even_factor (a b : ℕ) : 
  a > 0 → b > 0 → Even (a * b) → Even a ∨ Even b :=
by sorry

end NUMINAMATH_CALUDE_even_product_implies_even_factor_l2449_244932


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_twelve_l2449_244911

theorem five_digit_divisible_by_twelve : ∃! (n : Nat), n < 10 ∧ 51470 + n ≡ 0 [MOD 12] := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_twelve_l2449_244911


namespace NUMINAMATH_CALUDE_weight_identification_unbiased_weight_identification_biased_l2449_244927

/-- Represents a weight with a mass in grams -/
structure Weight where
  mass : ℕ

/-- Represents a balance scale -/
inductive BalanceResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing operation -/
def weighing (left right : List Weight) (bias : ℕ := 0) : BalanceResult :=
  sorry

/-- Represents the process of identifying weights -/
def identifyWeights (weights : List Weight) (numWeighings : ℕ) (bias : ℕ := 0) : Bool :=
  sorry

/-- The set of weights Tanya has -/
def tanyasWeights : List Weight :=
  [⟨1000⟩, ⟨1002⟩, ⟨1004⟩, ⟨1005⟩]

theorem weight_identification_unbiased :
  ¬ (identifyWeights tanyasWeights 4 0) :=
sorry

theorem weight_identification_biased :
  identifyWeights tanyasWeights 4 1 :=
sorry

end NUMINAMATH_CALUDE_weight_identification_unbiased_weight_identification_biased_l2449_244927


namespace NUMINAMATH_CALUDE_unique_intersection_l2449_244973

-- Define the complex plane
variable (z : ℂ)

-- Define the equations
def equation1 (z : ℂ) : Prop := Complex.abs (z - 5) = 3 * Complex.abs (z + 5)
def equation2 (z : ℂ) (k : ℝ) : Prop := Complex.abs z = k

-- Define the intersection condition
def intersectsOnce (k : ℝ) : Prop :=
  ∃! z, equation1 z ∧ equation2 z k

-- Theorem statement
theorem unique_intersection :
  ∃! k, intersectsOnce k ∧ k = 12.5 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l2449_244973


namespace NUMINAMATH_CALUDE_group_average_calculation_l2449_244950

theorem group_average_calculation (initial_group_size : ℕ) 
  (new_member_amount : ℚ) (new_average : ℚ) : 
  initial_group_size = 7 → 
  new_member_amount = 56 → 
  new_average = 20 → 
  (initial_group_size * new_average + new_member_amount) / (initial_group_size + 1) = new_average → 
  new_average = 20 := by
  sorry

end NUMINAMATH_CALUDE_group_average_calculation_l2449_244950


namespace NUMINAMATH_CALUDE_kite_perimeter_l2449_244912

/-- A kite with given side lengths -/
structure Kite where
  short_side : ℝ
  long_side : ℝ

/-- The perimeter of a kite -/
def perimeter (k : Kite) : ℝ :=
  2 * k.short_side + 2 * k.long_side

/-- Theorem: The perimeter of a kite with short sides 10 inches and long sides 15 inches is 50 inches -/
theorem kite_perimeter : 
  let k : Kite := { short_side := 10, long_side := 15 }
  perimeter k = 50 := by sorry

end NUMINAMATH_CALUDE_kite_perimeter_l2449_244912


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2449_244977

theorem simplify_and_evaluate (m : ℤ) (h : m = -1) :
  -(m^2 - 3*m) + 2*(m^2 - m - 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2449_244977


namespace NUMINAMATH_CALUDE_factor_million_three_ways_l2449_244907

/-- The number of ways to factor 1,000,000 into three factors, ignoring order -/
def factor_ways : ℕ := 139

/-- The prime factorization of 1,000,000 -/
def million_factorization : ℕ × ℕ := (6, 6)

theorem factor_million_three_ways :
  let (a, b) := million_factorization
  (2^a * 5^b = 1000000) →
  (factor_ways = 
    (1 : ℕ) + -- case where all factors are equal
    15 + -- case where exactly two factors are equal
    ((28 * 28 - 15 * 3 - 1) / 6 : ℕ) -- case where all factors are different
  ) := by sorry

end NUMINAMATH_CALUDE_factor_million_three_ways_l2449_244907


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2449_244925

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 0}
def B : Set ℝ := {x | x^2 - 1 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2449_244925


namespace NUMINAMATH_CALUDE_bulb_toggling_theorem_l2449_244951

/-- Represents the state of a light bulb (on or off) -/
inductive BulbState
| Off
| On

/-- Toggles the state of a light bulb -/
def toggleBulb : BulbState → BulbState
| BulbState.Off => BulbState.On
| BulbState.On => BulbState.Off

/-- Returns the number of positive divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- Returns true if a natural number is a perfect square, false otherwise -/
def isPerfectSquare (n : ℕ) : Bool := sorry

/-- Simulates the process of students toggling light bulbs -/
def toggleBulbs (n : ℕ) : List BulbState := sorry

/-- Counts the number of bulbs that are on after the toggling process -/
def countOnBulbs (bulbs : List BulbState) : ℕ := sorry

/-- Counts the number of perfect squares less than or equal to a given number -/
def countPerfectSquares (n : ℕ) : ℕ := sorry

theorem bulb_toggling_theorem :
  countOnBulbs (toggleBulbs 100) = countPerfectSquares 100 := by sorry

end NUMINAMATH_CALUDE_bulb_toggling_theorem_l2449_244951


namespace NUMINAMATH_CALUDE_combined_efficiency_approx_38_l2449_244941

-- Define the fuel efficiencies and distance
def jane_efficiency : ℚ := 30
def mike_efficiency : ℚ := 15
def carl_efficiency : ℚ := 20
def distance : ℚ := 100

-- Define the combined fuel efficiency function
def combined_efficiency (e1 e2 e3 d : ℚ) : ℚ :=
  (3 * d) / (d / e1 + d / e2 + d / e3)

-- State the theorem
theorem combined_efficiency_approx_38 :
  ∃ ε > 0, abs (combined_efficiency jane_efficiency mike_efficiency carl_efficiency distance - 38) < ε :=
by sorry

end NUMINAMATH_CALUDE_combined_efficiency_approx_38_l2449_244941


namespace NUMINAMATH_CALUDE_factorization_equality_l2449_244949

theorem factorization_equality (x : ℝ) : 3 * x^2 - 9 * x = 3 * x * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2449_244949


namespace NUMINAMATH_CALUDE_calculate_expression_l2449_244994

theorem calculate_expression : 2003^3 - 2001^3 - 6 * 2003^2 + 24 * 1001 = -4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2449_244994


namespace NUMINAMATH_CALUDE_identity_proof_l2449_244982

theorem identity_proof (a b c : ℝ) 
  (h1 : (a - c) / (a + c) ≠ 0)
  (h2 : (b - c) / (b + c) ≠ 0)
  (h3 : (a + c) / (a - c) + (b + c) / (b - c) ≠ 0) :
  ((((a - c) / (a + c) + (b - c) / (b + c)) / ((a + c) / (a - c) + (b + c) / (b - c))) ^ 2) = 
  ((((a - c) / (a + c)) ^ 2 + ((b - c) / (b + c)) ^ 2) / (((a + c) / (a - c)) ^ 2 + ((b + c) / (b - c)) ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_identity_proof_l2449_244982


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2449_244924

-- Define the point (x, y) in the Cartesian coordinate system
def point (a : ℝ) : ℝ × ℝ := (-2, a^2 + 1)

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem point_in_second_quadrant (a : ℝ) :
  in_second_quadrant (point a) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2449_244924


namespace NUMINAMATH_CALUDE_probability_second_science_question_l2449_244945

/-- Given a set of questions with science and humanities questions,
    prove the probability of drawing a second science question
    after drawing a science question first. -/
theorem probability_second_science_question
  (total_questions : ℕ)
  (science_questions : ℕ)
  (humanities_questions : ℕ)
  (h1 : total_questions = 6)
  (h2 : science_questions = 4)
  (h3 : humanities_questions = 2)
  (h4 : total_questions = science_questions + humanities_questions)
  (h5 : science_questions > 0) :
  (science_questions - 1 : ℚ) / (total_questions - 1) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_probability_second_science_question_l2449_244945


namespace NUMINAMATH_CALUDE_root_implies_a_values_l2449_244919

theorem root_implies_a_values (a : ℝ) : 
  (2 * (-1)^2 + a * (-1) - a^2 = 0) → (a = 1 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_values_l2449_244919


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2449_244976

-- Define the inverse variation relationship
def inverse_variation (y z : ℝ) : Prop := ∃ k : ℝ, y^2 * Real.sqrt z = k

-- Define the theorem
theorem inverse_variation_problem (y₁ y₂ z₁ z₂ : ℝ) 
  (h1 : inverse_variation y₁ z₁)
  (h2 : y₁ = 3)
  (h3 : z₁ = 4)
  (h4 : y₂ = 6) :
  z₂ = 1/4 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2449_244976


namespace NUMINAMATH_CALUDE_two_primes_sum_and_product_l2449_244957

theorem two_primes_sum_and_product : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p * q = 166 ∧ p + q = 85 := by
sorry

end NUMINAMATH_CALUDE_two_primes_sum_and_product_l2449_244957


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2449_244970

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r ≠ 0 ∧ a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 1 * a 9 = 16) :
  a 2 * a 5 * a 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2449_244970


namespace NUMINAMATH_CALUDE_ice_cube_volume_l2449_244905

theorem ice_cube_volume (original_volume : ℝ) : 
  (original_volume > 0) →
  (original_volume * (1/4) * (1/4) = 0.25) →
  (original_volume = 4) := by
sorry

end NUMINAMATH_CALUDE_ice_cube_volume_l2449_244905


namespace NUMINAMATH_CALUDE_line_slope_point_sum_l2449_244902

/-- Given a line with slope 5 passing through (5, 3), prove m + b^2 = 489 --/
theorem line_slope_point_sum (m b : ℝ) : 
  m = 5 →                   -- The slope is 5
  3 = 5 * 5 + b →           -- The line passes through (5, 3)
  m + b^2 = 489 :=          -- Prove that m + b^2 = 489
by sorry

end NUMINAMATH_CALUDE_line_slope_point_sum_l2449_244902


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l2449_244971

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (36 * π / 180) * Real.cos (24 * π / 180) +
  Real.cos (36 * π / 180) * Real.sin (156 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l2449_244971


namespace NUMINAMATH_CALUDE_points_four_units_from_negative_three_l2449_244988

def distance_from_point (x y : ℝ) : ℝ := |x - y|

theorem points_four_units_from_negative_three :
  {x : ℝ | distance_from_point x (-3) = 4} = {1, -7} := by
  sorry

end NUMINAMATH_CALUDE_points_four_units_from_negative_three_l2449_244988


namespace NUMINAMATH_CALUDE_beshmi_investment_l2449_244981

theorem beshmi_investment (savings : ℝ) : 
  (1 / 5 : ℝ) * savings + 0.42 * savings + (savings - (1 / 5 : ℝ) * savings - 0.42 * savings) = savings
    → 0.42 * savings = 10500
    → savings - (1 / 5 : ℝ) * savings - 0.42 * savings = 9500 :=
by
  sorry

end NUMINAMATH_CALUDE_beshmi_investment_l2449_244981


namespace NUMINAMATH_CALUDE_games_played_l2449_244942

/-- Given that Andrew spent $9.00 for each game and $45 in total,
    prove that the number of games played is 5. -/
theorem games_played (cost_per_game : ℝ) (total_spent : ℝ) : 
  cost_per_game = 9 → total_spent = 45 → (total_spent / cost_per_game : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_games_played_l2449_244942


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l2449_244940

theorem floor_equation_solutions (a : ℝ) (n : ℕ) (h1 : a > 1) (h2 : n ≥ 2) :
  (∃ (S : Finset ℝ), S.card = n ∧ (∀ x ∈ S, ⌊a * x⌋ = x)) ↔ 
  (1 + 1 / n : ℝ) ≤ a ∧ a < 1 + 1 / (n - 1) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l2449_244940


namespace NUMINAMATH_CALUDE_garden_area_increase_l2449_244996

/-- Proves that adding 60 feet of fence to a rectangular garden of 80x20 feet
    to make it square increases the area by 2625 square feet. -/
theorem garden_area_increase : 
  ∀ (original_length original_width added_fence : ℕ),
    original_length = 80 →
    original_width = 20 →
    added_fence = 60 →
    let original_perimeter := 2 * (original_length + original_width)
    let new_perimeter := original_perimeter + added_fence
    let new_side := new_perimeter / 4
    let original_area := original_length * original_width
    let new_area := new_side * new_side
    new_area - original_area = 2625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_increase_l2449_244996


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l2449_244906

theorem solve_system_of_equations (x y z : ℤ) 
  (eq1 : 4 * x + y + z = 80)
  (eq2 : 2 * x - y - z = 40)
  (eq3 : 3 * x + y - z = 20) :
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l2449_244906


namespace NUMINAMATH_CALUDE_kims_money_l2449_244984

/-- Given the money relationships between Kim, Sal, Phil, and Alex, prove Kim's amount. -/
theorem kims_money (sal phil kim alex : ℝ) 
  (h1 : kim = 1.4 * sal)  -- Kim has 40% more money than Sal
  (h2 : sal = 0.8 * phil)  -- Sal has 20% less money than Phil
  (h3 : alex = 1.25 * (sal + kim))  -- Alex has 25% more money than Sal and Kim combined
  (h4 : sal + phil + alex = 3.6)  -- Sal, Phil, and Alex have a combined total of $3.60
  : kim = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_kims_money_l2449_244984


namespace NUMINAMATH_CALUDE_janes_pudding_purchase_l2449_244989

/-- The number of cups of pudding Jane purchased -/
def cups_of_pudding : ℕ := 5

/-- The number of ice cream cones Jane purchased -/
def ice_cream_cones : ℕ := 15

/-- The cost of one ice cream cone in dollars -/
def ice_cream_cost : ℕ := 5

/-- The cost of one cup of pudding in dollars -/
def pudding_cost : ℕ := 2

/-- The difference between the total cost of ice cream and pudding in dollars -/
def cost_difference : ℕ := 65

theorem janes_pudding_purchase :
  cups_of_pudding * pudding_cost + cost_difference = ice_cream_cones * ice_cream_cost :=
by sorry

end NUMINAMATH_CALUDE_janes_pudding_purchase_l2449_244989


namespace NUMINAMATH_CALUDE_sequence_sixth_term_l2449_244967

theorem sequence_sixth_term (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 2)
  (h3 : ∀ n ≥ 2, 2 * (a n)^2 = (a (n+1))^2 + (a (n-1))^2)
  (h4 : ∀ n, a n > 0) :
  a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_sixth_term_l2449_244967


namespace NUMINAMATH_CALUDE_upstream_distance_l2449_244930

/-- Represents the speed of a boat in km/hr -/
def BoatSpeed : ℝ := 11

/-- Represents the distance traveled downstream in km -/
def DownstreamDistance : ℝ := 16

/-- Represents the time of travel in hours -/
def TravelTime : ℝ := 1

/-- Calculates the speed of the stream based on the boat's speed and downstream distance -/
def StreamSpeed : ℝ := DownstreamDistance - BoatSpeed

/-- Theorem: The distance traveled upstream in one hour is 6 km -/
theorem upstream_distance :
  BoatSpeed - StreamSpeed = 6 := by
  sorry

end NUMINAMATH_CALUDE_upstream_distance_l2449_244930


namespace NUMINAMATH_CALUDE_triangle_cosine_B_l2449_244944

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_cosine_B (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.B - abc.a * Real.sin abc.A = (1/2) * abc.a * Real.sin abc.C)
  (h2 : (1/2) * abc.a * abc.c * Real.sin abc.B = abc.a^2 * Real.sin abc.B) :
  Real.cos abc.B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_B_l2449_244944


namespace NUMINAMATH_CALUDE_P_no_real_roots_l2449_244955

/-- Recursive definition of the polynomial sequence P_n(x) -/
def P (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => λ _ => 1
  | n + 1 => λ x => x^(11*(n+1)) - P n x

/-- Theorem stating that P_n(x) has no real roots for all n ≥ 0 -/
theorem P_no_real_roots (n : ℕ) : ∀ x : ℝ, P n x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_P_no_real_roots_l2449_244955


namespace NUMINAMATH_CALUDE_three_rug_overlap_l2449_244909

theorem three_rug_overlap (total_area floor_area double_layer_area : ℝ) 
  (h1 : total_area = 200)
  (h2 : floor_area = 140)
  (h3 : double_layer_area = 24) : 
  ∃ (triple_layer_area : ℝ), 
    triple_layer_area = 18 ∧ 
    total_area = floor_area + double_layer_area + 2 * triple_layer_area :=
by
  sorry

end NUMINAMATH_CALUDE_three_rug_overlap_l2449_244909


namespace NUMINAMATH_CALUDE_distribution_counterexample_l2449_244969

-- Define a type for random variables
def RandomVariable := Real → Real

-- Define a type for distribution functions
def DistributionFunction := Real → Real

-- Function to get the distribution function of a random variable
def getDistribution (X : RandomVariable) : DistributionFunction := sorry

-- Function to check if two distribution functions are identical
def distributionsIdentical (F G : DistributionFunction) : Prop := sorry

-- Function to multiply two random variables
def multiply (X Y : RandomVariable) : RandomVariable := sorry

theorem distribution_counterexample :
  ∃ (ξ η ζ : RandomVariable),
    distributionsIdentical (getDistribution ξ) (getDistribution η) ∧
    ¬distributionsIdentical (getDistribution (multiply ξ ζ)) (getDistribution (multiply η ζ)) := by
  sorry

end NUMINAMATH_CALUDE_distribution_counterexample_l2449_244969


namespace NUMINAMATH_CALUDE_root_implies_b_value_l2449_244986

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 15

-- State the theorem
theorem root_implies_b_value (a : ℚ) :
  (∃ b : ℚ, f a b (3 + Real.sqrt 5) = 0) →
  (∃ b : ℚ, b = -37/2) :=
by sorry

end NUMINAMATH_CALUDE_root_implies_b_value_l2449_244986


namespace NUMINAMATH_CALUDE_tangent_line_coincidence_l2449_244946

/-- Given a differentiable function f where the tangent line of y = f(x) at (0,0) 
    coincides with the tangent line of y = f(x)/x at (2,1), prove that f'(2) = 2 -/
theorem tangent_line_coincidence (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, x ≠ 0 → (f x) / x = ((f 0) + (deriv f 0) * x)) →
  (f 2) / 2 = 1 →
  deriv f 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_coincidence_l2449_244946


namespace NUMINAMATH_CALUDE_other_communities_count_l2449_244990

theorem other_communities_count (total_boys : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) :
  total_boys = 700 →
  muslim_percent = 44/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 126 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l2449_244990


namespace NUMINAMATH_CALUDE_unique_monic_quadratic_with_complex_root_l2449_244998

def is_monic_quadratic (p : ℝ → ℂ) : Prop :=
  ∃ a b : ℝ, ∀ x, p x = x^2 + a*x + b

theorem unique_monic_quadratic_with_complex_root :
  ∃! p : ℝ → ℂ, is_monic_quadratic p ∧ p (3 - 4*I) = 0 ∧ 
  ∀ x : ℝ, p x = x^2 - 6*x + 25 :=
sorry

end NUMINAMATH_CALUDE_unique_monic_quadratic_with_complex_root_l2449_244998


namespace NUMINAMATH_CALUDE_trig_identity_l2449_244908

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) - 2 / Real.sin (70 * π / 180) = 
  2 * Real.tan (50 * π / 180) - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2449_244908


namespace NUMINAMATH_CALUDE_chef_eggs_per_cake_l2449_244947

/-- Given a chef with initial eggs, eggs in fridge, and number of cakes made,
    calculate the number of eggs used per cake. -/
def eggs_per_cake (initial_eggs : ℕ) (fridge_eggs : ℕ) (cakes_made : ℕ) : ℕ :=
  (initial_eggs - fridge_eggs) / cakes_made

/-- Theorem stating that with 60 initial eggs, 10 eggs in fridge, and 10 cakes made,
    the number of eggs per cake is 5. -/
theorem chef_eggs_per_cake :
  eggs_per_cake 60 10 10 = 5 := by
  sorry

#eval eggs_per_cake 60 10 10

end NUMINAMATH_CALUDE_chef_eggs_per_cake_l2449_244947


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2449_244963

theorem sqrt_sum_inequality (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 19/3) :
  Real.sqrt (x - 1) + Real.sqrt (2 * x + 9) + Real.sqrt (19 - 3 * x) < 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2449_244963


namespace NUMINAMATH_CALUDE_max_value_of_a_max_value_is_tight_l2449_244987

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x - a ≥ 0) → a ≤ -1 := by
  sorry

theorem max_value_is_tight : ∃ a : ℝ, a = -1 ∧ (∀ x : ℝ, x^2 - 2*x - a ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_a_max_value_is_tight_l2449_244987


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2449_244980

theorem container_volume_ratio :
  ∀ (v1 v2 : ℚ),
  v1 > 0 → v2 > 0 →
  (5 / 6 : ℚ) * v1 = (3 / 4 : ℚ) * v2 →
  v1 / v2 = (9 / 10 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2449_244980


namespace NUMINAMATH_CALUDE_journey_start_time_l2449_244903

/-- Two people moving towards each other -/
structure Journey where
  start_time : ℝ
  meet_time : ℝ
  a_finish_time : ℝ
  b_finish_time : ℝ

/-- The journey satisfies the problem conditions -/
def satisfies_conditions (j : Journey) : Prop :=
  j.meet_time = 12 ∧ 
  j.a_finish_time = 16 ∧ 
  j.b_finish_time = 21 ∧ 
  0 < j.start_time ∧ j.start_time < j.meet_time

/-- The equation representing the journey -/
def journey_equation (j : Journey) : Prop :=
  1 / (j.meet_time - j.start_time) + 
  1 / (j.a_finish_time - j.meet_time) + 
  1 / (j.b_finish_time - j.meet_time) = 1

theorem journey_start_time (j : Journey) 
  (h1 : satisfies_conditions j) 
  (h2 : journey_equation j) : 
  j.start_time = 6 := by
  sorry


end NUMINAMATH_CALUDE_journey_start_time_l2449_244903


namespace NUMINAMATH_CALUDE_five_digit_with_four_or_five_l2449_244917

/-- The number of five-digit positive integers -/
def total_five_digit : ℕ := 90000

/-- The number of digits that are not 4 or 5 -/
def non_four_five_digits : ℕ := 8

/-- The number of options for the first digit (excluding 0, 4, and 5) -/
def first_digit_options : ℕ := 7

/-- The number of five-digit positive integers without 4 or 5 -/
def without_four_five : ℕ := first_digit_options * non_four_five_digits^4

theorem five_digit_with_four_or_five :
  total_five_digit - without_four_five = 61328 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_with_four_or_five_l2449_244917


namespace NUMINAMATH_CALUDE_min_radius_point_l2449_244959

/-- The point that minimizes the radius of a circle centered at the origin -/
theorem min_radius_point (x y : ℝ) :
  (∀ a b : ℝ, x^2 + y^2 ≤ a^2 + b^2) → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_radius_point_l2449_244959


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l2449_244914

theorem arithmetic_sequence_solution :
  let a₁ : ℚ := 2/3
  let a₂ := y - 2
  let a₃ := 4*y - 1
  (a₂ - a₁ = a₃ - a₂) → y = 11/6 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l2449_244914
