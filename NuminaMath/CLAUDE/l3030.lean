import Mathlib

namespace NUMINAMATH_CALUDE_complex_magnitude_special_angle_l3030_303020

theorem complex_magnitude_special_angle : 
  let z : ℂ := Complex.mk (Real.sin (π / 3)) (Real.cos (π / 6))
  ‖z‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_special_angle_l3030_303020


namespace NUMINAMATH_CALUDE_distance_proof_l3030_303010

/-- The distance between three equidistant points A, B, and C. -/
def distance_between_points : ℝ := 26

/-- The speed of the cyclist traveling from A to B in km/h. -/
def cyclist_speed : ℝ := 15

/-- The speed of the tourist traveling from B to C in km/h. -/
def tourist_speed : ℝ := 5

/-- The time at which the cyclist and tourist are at their shortest distance, in hours. -/
def time_shortest_distance : ℝ := 1.4

/-- The theorem stating that the distance between the points is 26 km under the given conditions. -/
theorem distance_proof :
  ∀ (S : ℝ),
  (S > 0) →
  (S = distance_between_points) →
  (∀ (t : ℝ), 
    (t > 0) →
    (cyclist_speed * t ≤ S) →
    (tourist_speed * t ≤ S) →
    (S^2 - 35*t*S + 325*t^2 ≥ S^2 - 35*time_shortest_distance*S + 325*time_shortest_distance^2)) →
  (S = 26) :=
sorry

end NUMINAMATH_CALUDE_distance_proof_l3030_303010


namespace NUMINAMATH_CALUDE_f_properties_l3030_303051

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (10 - 2 * x) / Real.log (1/2)

-- Theorem statement
theorem f_properties :
  -- 1. Domain of f(x) is (-∞, 5)
  (∀ x, f x ≠ 0 → x < 5) ∧
  -- 2. f(x) is increasing on its domain
  (∀ x y, x < y → x < 5 → y < 5 → f x < f y) ∧
  -- 3. Maximum value of m for which f(x) ≥ (1/2)ˣ + m holds for all x ∈ [3, 4] is -17/8
  (∀ m, (∀ x, x ∈ Set.Icc 3 4 → f x ≥ (1/2)^x + m) → m ≤ -17/8) ∧
  (∃ x, x ∈ Set.Icc 3 4 ∧ f x = (1/2)^x + (-17/8)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3030_303051


namespace NUMINAMATH_CALUDE_tan_pi_sevenths_l3030_303080

theorem tan_pi_sevenths (y₁ y₂ y₃ : ℝ) 
  (h : y₁^3 - 21*y₁^2 + 35*y₁ - 7 = 0 ∧ 
       y₂^3 - 21*y₂^2 + 35*y₂ - 7 = 0 ∧ 
       y₃^3 - 21*y₃^2 + 35*y₃ - 7 = 0) 
  (h₁ : y₁ = Real.tan (π/7)^2) 
  (h₂ : y₂ = Real.tan (2*π/7)^2) 
  (h₃ : y₃ = Real.tan (3*π/7)^2) : 
  Real.tan (π/7) * Real.tan (2*π/7) * Real.tan (3*π/7) = Real.sqrt 7 ∧
  Real.tan (π/7)^2 + Real.tan (2*π/7)^2 + Real.tan (3*π/7)^2 = 21 := by
sorry

end NUMINAMATH_CALUDE_tan_pi_sevenths_l3030_303080


namespace NUMINAMATH_CALUDE_largest_angle_is_80_l3030_303077

-- Define a right angle in degrees
def right_angle : ℝ := 90

-- Define the triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.angle1 + t.angle2 = (4/3) * right_angle ∧
  t.angle2 = t.angle1 + 40 ∧
  t.angle1 + t.angle2 + t.angle3 = 180

-- Theorem statement
theorem largest_angle_is_80 (t : Triangle) :
  triangle_conditions t → (max t.angle1 (max t.angle2 t.angle3) = 80) :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_is_80_l3030_303077


namespace NUMINAMATH_CALUDE_intersection_M_N_l3030_303096

def M : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3030_303096


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3030_303053

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 4 = 0 ∧ y^2 + m*y + 4 = 0) ↔ 
  (m < -4 ∨ m > 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3030_303053


namespace NUMINAMATH_CALUDE_correct_sums_l3030_303025

theorem correct_sums (total : ℕ) (wrong : ℕ → ℕ) (h1 : total = 48) (h2 : wrong = λ x => 2 * x) : 
  ∃ x : ℕ, x + wrong x = total ∧ x = 16 := by
sorry

end NUMINAMATH_CALUDE_correct_sums_l3030_303025


namespace NUMINAMATH_CALUDE_solve_equation_l3030_303007

theorem solve_equation (x : ℝ) : 3 * x = 2 * x + 6 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3030_303007


namespace NUMINAMATH_CALUDE_one_ton_equals_2000_pounds_l3030_303035

-- Define the basic units
def ounce : ℕ := 1
def pound : ℕ := 16 * ounce
def ton : ℕ := 2000 * pound

-- Define the packet weight
def packet_weight : ℕ := 16 * pound + 4 * ounce

-- Define the gunny bag capacity
def gunny_bag_capacity : ℕ := 13 * ton

-- Theorem statement
theorem one_ton_equals_2000_pounds : 
  (2000 * packet_weight = gunny_bag_capacity) → ton = 2000 * pound := by
  sorry

end NUMINAMATH_CALUDE_one_ton_equals_2000_pounds_l3030_303035


namespace NUMINAMATH_CALUDE_race_distance_l3030_303039

/-- Represents a race between two participants A and B -/
structure Race where
  distance : ℝ
  timeA : ℝ
  timeB : ℝ
  speedA : ℝ
  speedB : ℝ

/-- The conditions of the race -/
def raceConditions (r : Race) : Prop :=
  r.timeA = 18 ∧
  r.timeB = r.timeA + 7 ∧
  r.distance = r.speedA * r.timeA ∧
  r.distance = r.speedB * r.timeB ∧
  r.distance - r.speedB * r.timeA = 56

theorem race_distance (r : Race) (h : raceConditions r) : r.distance = 200 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l3030_303039


namespace NUMINAMATH_CALUDE_solve_for_C_l3030_303012

theorem solve_for_C : ∃ C : ℝ, (4 * C + 5 = 37) ∧ (C = 8) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_C_l3030_303012


namespace NUMINAMATH_CALUDE_equal_coins_count_l3030_303098

/-- Represents the value of a coin in cents -/
def coin_value (coin_type : String) : ℕ :=
  match coin_type with
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Represents the total value of coins in cents -/
def total_value : ℕ := 120

/-- Represents the number of different types of coins -/
def num_coin_types : ℕ := 3

theorem equal_coins_count (num_each : ℕ) :
  (num_each * coin_value "nickel" +
   num_each * coin_value "dime" +
   num_each * coin_value "quarter" = total_value) →
  (num_each * num_coin_types = 9) := by
  sorry

#check equal_coins_count

end NUMINAMATH_CALUDE_equal_coins_count_l3030_303098


namespace NUMINAMATH_CALUDE_officer_selection_count_l3030_303087

/-- The number of members in the club -/
def club_members : ℕ := 12

/-- The number of officer positions to be filled -/
def officer_positions : ℕ := 4

/-- The number of ways to choose officers from the club members -/
def ways_to_choose_officers : ℕ := club_members * (club_members - 1) * (club_members - 2) * (club_members - 3)

theorem officer_selection_count :
  ways_to_choose_officers = 11880 :=
sorry

end NUMINAMATH_CALUDE_officer_selection_count_l3030_303087


namespace NUMINAMATH_CALUDE_dog_age_ratio_l3030_303061

/-- Given information about five dogs' ages, prove the ratio of the 4th to 3rd fastest dog's age --/
theorem dog_age_ratio :
  ∀ (age1 age2 age3 age4 age5 : ℕ),
  -- Average age of 1st and 5th fastest dogs is 18 years
  (age1 + age5) / 2 = 18 →
  -- 1st fastest dog is 10 years old
  age1 = 10 →
  -- 2nd fastest dog is 2 years younger than the 1st fastest dog
  age2 = age1 - 2 →
  -- 3rd fastest dog is 4 years older than the 2nd fastest dog
  age3 = age2 + 4 →
  -- 4th fastest dog is half the age of the 3rd fastest dog
  2 * age4 = age3 →
  -- 5th fastest dog is 20 years older than the 4th fastest dog
  age5 = age4 + 20 →
  -- Ratio of 4th fastest dog's age to 3rd fastest dog's age is 1:2
  2 * age4 = age3 := by
  sorry

end NUMINAMATH_CALUDE_dog_age_ratio_l3030_303061


namespace NUMINAMATH_CALUDE_subset_implies_b_equals_two_l3030_303017

theorem subset_implies_b_equals_two :
  (∀ x y : ℝ, x + y - 2 = 0 ∧ x - 2*y + 4 = 0 → y = 3*x + b) →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_b_equals_two_l3030_303017


namespace NUMINAMATH_CALUDE_simplify_expression_l3030_303075

theorem simplify_expression (a b c : ℝ) :
  (15*a + 45*b + 20*c) + (25*a - 35*b - 10*c) - (10*a + 55*b + 30*c) = 30*a - 45*b - 20*c :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3030_303075


namespace NUMINAMATH_CALUDE_ratio_of_x_to_y_l3030_303044

theorem ratio_of_x_to_y (x y : ℝ) (h1 : 5 * x = 6 * y) (h2 : x * y ≠ 0) :
  (1/3 * x) / (1/5 * y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_to_y_l3030_303044


namespace NUMINAMATH_CALUDE_adams_collection_worth_80_dollars_l3030_303028

/-- The value of Adam's coin collection -/
def adams_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℕ) : ℕ :=
  total_coins * (sample_value / sample_coins)

/-- Theorem: Adam's coin collection is worth 80 dollars -/
theorem adams_collection_worth_80_dollars :
  adams_collection_value 20 4 16 = 80 := by
  sorry

end NUMINAMATH_CALUDE_adams_collection_worth_80_dollars_l3030_303028


namespace NUMINAMATH_CALUDE_black_friday_sales_l3030_303090

/-- Proves that if a store sells 477 televisions three years from now, 
    and the number of televisions sold increases by 50 each year, 
    then the store sold 327 televisions this year. -/
theorem black_friday_sales (current_sales : ℕ) : 
  (current_sales + 3 * 50 = 477) → current_sales = 327 := by
  sorry

end NUMINAMATH_CALUDE_black_friday_sales_l3030_303090


namespace NUMINAMATH_CALUDE_weight_of_b_l3030_303032

/-- Given the average weights of three people (a, b, c) and two pairs (a, b) and (b, c),
    prove that the weight of b is 31 kg. -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)  -- average weight of a, b, and c is 45 kg
  (h2 : (a + b) / 2 = 40)      -- average weight of a and b is 40 kg
  (h3 : (b + c) / 2 = 43) :    -- average weight of b and c is 43 kg
  b = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l3030_303032


namespace NUMINAMATH_CALUDE_function_maximum_implies_inequality_l3030_303029

/-- Given a function f(x) = ln x - mx² + 2nx where m is real and n is positive,
    if f(x) ≤ f(1) for all positive x, then ln n < 8m -/
theorem function_maximum_implies_inequality (m : ℝ) (n : ℝ) (h_n_pos : n > 0) :
  (∀ x > 0, Real.log x - m * x^2 + 2 * n * x ≤ Real.log 1 - m * 1^2 + 2 * n * 1) →
  Real.log n < 8 * m :=
by sorry

end NUMINAMATH_CALUDE_function_maximum_implies_inequality_l3030_303029


namespace NUMINAMATH_CALUDE_dogGroupings_eq_4200_l3030_303091

/-- The number of ways to divide 12 dogs into groups of 4, 5, and 3,
    with Fluffy in the 4-dog group and Nipper in the 5-dog group -/
def dogGroupings : ℕ :=
  let totalDogs : ℕ := 12
  let group1Size : ℕ := 4  -- Fluffy's group
  let group2Size : ℕ := 5  -- Nipper's group
  let group3Size : ℕ := 3
  let remainingDogs : ℕ := totalDogs - 2  -- Excluding Fluffy and Nipper

  (remainingDogs.choose (group1Size - 1)) * ((remainingDogs - (group1Size - 1)).choose (group2Size - 1))

theorem dogGroupings_eq_4200 : dogGroupings = 4200 := by
  sorry

end NUMINAMATH_CALUDE_dogGroupings_eq_4200_l3030_303091


namespace NUMINAMATH_CALUDE_smallest_m_is_ten_l3030_303042

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℤ
  first_term : a 1 = -19
  difference : a 7 - a 4 = 6
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ+) : ℤ :=
  (seq.a 1 + seq.a n) * n / 2

/-- The theorem to be proved -/
theorem smallest_m_is_ten (seq : ArithmeticSequence) :
  ∃ m : ℕ+, (∀ n : ℕ+, sum_n seq n ≥ sum_n seq m) ∧ 
    (∀ k : ℕ+, k < m → ∃ n : ℕ+, sum_n seq n < sum_n seq k) ∧
    m = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_is_ten_l3030_303042


namespace NUMINAMATH_CALUDE_unique_number_with_remainder_l3030_303054

theorem unique_number_with_remainder (n : ℕ) : n < 5000 ∧ 
  (∀ k ∈ Finset.range 9, n % (k + 2) = 1) ↔ n = 2521 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_remainder_l3030_303054


namespace NUMINAMATH_CALUDE_jacob_ladder_price_l3030_303050

/-- The price per rung for Jacob's ladders -/
def price_per_rung : ℚ := 2

/-- The number of ladders with 50 rungs -/
def ladders_50 : ℕ := 10

/-- The number of ladders with 60 rungs -/
def ladders_60 : ℕ := 20

/-- The number of rungs per ladder in the first group -/
def rungs_per_ladder_50 : ℕ := 50

/-- The number of rungs per ladder in the second group -/
def rungs_per_ladder_60 : ℕ := 60

/-- The total cost for all ladders -/
def total_cost : ℚ := 3400

theorem jacob_ladder_price : 
  price_per_rung * (ladders_50 * rungs_per_ladder_50 + ladders_60 * rungs_per_ladder_60) = total_cost := by
  sorry

end NUMINAMATH_CALUDE_jacob_ladder_price_l3030_303050


namespace NUMINAMATH_CALUDE_line_mb_product_l3030_303097

/-- A line passing through two points (0, -2) and (2, 4) has mb = -6 --/
theorem line_mb_product (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  (-2 : ℝ) = b →                -- Line passes through (0, -2)
  4 = m * 2 + b →               -- Line passes through (2, 4)
  m * b = -6 := by sorry

end NUMINAMATH_CALUDE_line_mb_product_l3030_303097


namespace NUMINAMATH_CALUDE_platform_length_l3030_303046

/-- Given a train of length 200 meters that crosses a platform in 50 seconds
    and a signal pole in 42 seconds, the length of the platform is 38 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 200)
  (h2 : platform_crossing_time = 50)
  (h3 : pole_crossing_time = 42) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 38 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l3030_303046


namespace NUMINAMATH_CALUDE_binomial_plus_five_l3030_303079

theorem binomial_plus_five : Nat.choose 7 4 + 5 = 40 := by sorry

end NUMINAMATH_CALUDE_binomial_plus_five_l3030_303079


namespace NUMINAMATH_CALUDE_triangle_side_length_l3030_303055

theorem triangle_side_length (A B C : ℝ) (b c : ℝ) :
  A = π / 3 →
  b = 16 →
  (1 / 2) * b * c * Real.sin A = 64 * Real.sqrt 3 →
  c = 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3030_303055


namespace NUMINAMATH_CALUDE_greatest_integer_cube_root_three_l3030_303015

theorem greatest_integer_cube_root_three : ⌊(2 + Real.sqrt 3)^3⌋ = 51 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_cube_root_three_l3030_303015


namespace NUMINAMATH_CALUDE_largest_two_digit_one_less_multiple_l3030_303019

theorem largest_two_digit_one_less_multiple : ∃ n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ k : ℕ, n + 1 = 60 * k) ∧
  (∀ m : ℕ, m > n → m < 100 → ¬∃ j : ℕ, m + 1 = 60 * j) ∧
  n = 59 := by
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_one_less_multiple_l3030_303019


namespace NUMINAMATH_CALUDE_angle_D_measure_l3030_303084

-- Define the pentagon and its properties
structure Pentagon where
  A : ℝ  -- Measure of angle A
  B : ℝ  -- Measure of angle B
  C : ℝ  -- Measure of angle C
  D : ℝ  -- Measure of angle D
  E : ℝ  -- Measure of angle E
  convex : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ E > 0
  sum_angles : A + B + C + D + E = 540
  congruent_ABC : A = B ∧ B = C
  congruent_DE : D = E
  A_less_D : A = D - 40

-- Theorem statement
theorem angle_D_measure (p : Pentagon) : p.D = 132 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l3030_303084


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3030_303056

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | 12 * x^2 - a * x > a^2}
  (a > 0 → S = {x | x < -a/4} ∪ {x | x > a/3}) ∧
  (a = 0 → S = {x | x ≠ 0}) ∧
  (a < 0 → S = {x | x < a/3} ∪ {x | x > -a/4}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3030_303056


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_5_5_minus_5_3_l3030_303083

theorem smallest_prime_factor_of_5_5_minus_5_3 :
  Nat.minFac (5^5 - 5^3) = 2 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_5_5_minus_5_3_l3030_303083


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3030_303088

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (1 : ℂ) / ((1 + Complex.I)^2 + 1) + Complex.I^4
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3030_303088


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3030_303089

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {y | (y - 2) * (y + 3) < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3030_303089


namespace NUMINAMATH_CALUDE_w_range_l3030_303052

theorem w_range (x y w : ℝ) : 
  -x + y = 2 → x < 3 → y ≥ 0 → w = x + y - 2 → -4 ≤ w ∧ w < 6 :=
by sorry

end NUMINAMATH_CALUDE_w_range_l3030_303052


namespace NUMINAMATH_CALUDE_matrix_property_implies_k_one_and_n_even_l3030_303082

open Matrix

theorem matrix_property_implies_k_one_and_n_even 
  (k n : ℕ) 
  (hk : k ≥ 1) 
  (hn : n ≥ 2) 
  (A B : Matrix (Fin n) (Fin n) ℤ) 
  (h1 : A ^ 3 = 0)
  (h2 : A ^ k * B + B * A = 1) :
  k = 1 ∧ Even n :=
sorry

end NUMINAMATH_CALUDE_matrix_property_implies_k_one_and_n_even_l3030_303082


namespace NUMINAMATH_CALUDE_system_inequalities_solution_equation_solution_l3030_303021

-- Define the system of inequalities
def system_inequalities (x : ℝ) : Prop :=
  2 * (x - 1) ≥ -4 ∧ (3 * x - 6) / 2 < x - 1

-- Define the set of positive integer solutions
def positive_integer_solutions : Set ℕ :=
  {1, 2, 3}

-- Define the equation
def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ 3 / (x - 2) = 5 / (2 - x) - 1

-- Theorem for the system of inequalities
theorem system_inequalities_solution :
  ∀ n : ℕ, n ∈ positive_integer_solutions ↔ system_inequalities (n : ℝ) :=
sorry

-- Theorem for the equation
theorem equation_solution :
  ∀ x : ℝ, equation x ↔ x = -6 :=
sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_equation_solution_l3030_303021


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3030_303002

/-- Represents a repeating decimal of the form 0.abab̄ab -/
def repeating_decimal_2 (a b : ℕ) : ℚ :=
  (100 * a + 10 * b + a + b : ℚ) / 9999

/-- Represents a repeating decimal of the form 0.abcabc̄abc -/
def repeating_decimal_3 (a b c : ℕ) : ℚ :=
  (100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c : ℚ) / 999999

/-- The main theorem stating that if the sum of the two repeating decimals
    equals 33/37, then abc must be 447 -/
theorem repeating_decimal_sum (a b c : ℕ) 
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10) 
  (h_sum : repeating_decimal_2 a b + repeating_decimal_3 a b c = 33/37) :
  100 * a + 10 * b + c = 447 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3030_303002


namespace NUMINAMATH_CALUDE_least_multiple_32_over_500_l3030_303063

theorem least_multiple_32_over_500 : ∃ (n : ℕ), n * 32 > 500 ∧ n * 32 = 512 ∧ ∀ (m : ℕ), m * 32 > 500 → m * 32 ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_32_over_500_l3030_303063


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l3030_303071

theorem min_value_of_expression (x y : ℝ) : (x * y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, (x * y - 1)^2 + (x + y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l3030_303071


namespace NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_l3030_303066

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_increasing_iff (a : ℕ → ℝ) (h : is_arithmetic_sequence a) :
  (a 1 < a 3) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_l3030_303066


namespace NUMINAMATH_CALUDE_gcd_2952_1386_l3030_303040

theorem gcd_2952_1386 : Nat.gcd 2952 1386 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2952_1386_l3030_303040


namespace NUMINAMATH_CALUDE_num_valid_selections_eq_twenty_l3030_303085

/-- Represents a volunteer --/
inductive Volunteer : Type
  | A | B | C | D | E

/-- Represents a role --/
inductive Role : Type
  | Translator | TourGuide | Etiquette | Driver

/-- Predicate to check if a volunteer can perform a given role --/
def can_perform (v : Volunteer) (r : Role) : Prop :=
  match v, r with
  | Volunteer.A, Role.Translator => True
  | Volunteer.A, Role.TourGuide => True
  | Volunteer.B, Role.Translator => True
  | Volunteer.B, Role.TourGuide => True
  | Volunteer.C, _ => True
  | Volunteer.D, _ => True
  | Volunteer.E, _ => True
  | _, _ => False

/-- A valid selection is a function from Role to Volunteer satisfying the constraints --/
def ValidSelection : Type :=
  { f : Role → Volunteer // ∀ r, can_perform (f r) r ∧ ∀ r' ≠ r, f r ≠ f r' }

/-- The number of valid selections --/
def num_valid_selections : ℕ := sorry

/-- Theorem stating that the number of valid selections is 20 --/
theorem num_valid_selections_eq_twenty : num_valid_selections = 20 := by sorry

end NUMINAMATH_CALUDE_num_valid_selections_eq_twenty_l3030_303085


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3030_303072

theorem rectangle_area_change (initial_area : ℝ) : 
  initial_area = 500 →
  (0.8 * 1.2 * initial_area) = 480 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3030_303072


namespace NUMINAMATH_CALUDE_plough_time_for_A_l3030_303094

/-- Given two workers A and B who can plough a field together in 10 hours,
    and B alone takes 30 hours, prove that A alone would take 15 hours. -/
theorem plough_time_for_A (time_together time_B : ℝ) (time_together_pos : time_together > 0)
    (time_B_pos : time_B > 0) (h1 : time_together = 10) (h2 : time_B = 30) :
    ∃ time_A : ℝ, time_A > 0 ∧ 1 / time_A + 1 / time_B = 1 / time_together ∧ time_A = 15 := by
  sorry

end NUMINAMATH_CALUDE_plough_time_for_A_l3030_303094


namespace NUMINAMATH_CALUDE_arithmetic_progression_terms_l3030_303045

/-- 
Given an arithmetic progression with:
- First term: 2
- Last term: 62
- Common difference: 2

Prove that the number of terms in this arithmetic progression is 31.
-/
theorem arithmetic_progression_terms : 
  let a := 2  -- First term
  let L := 62 -- Last term
  let d := 2  -- Common difference
  let n := (L - a) / d + 1 -- Number of terms formula
  n = 31 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_terms_l3030_303045


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l3030_303047

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(B) - b * cos(A) = (4/5) * c, then tan(A) / tan(B) = 9 -/
theorem triangle_tangent_ratio (a b c : ℝ) (A B C : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = (4/5) * c →
  Real.tan A / Real.tan B = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l3030_303047


namespace NUMINAMATH_CALUDE_g_range_and_density_l3030_303014

noncomputable def g (a b c : ℝ) : ℝ := a / (a + b) + b / (b + c) + c / (c + a)

theorem g_range_and_density :
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 1 < g a b c ∧ g a b c < 2) ∧
  (∀ ε : ℝ, ε > 0 → 
    (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ |g a b c - 1| < ε) ∧
    (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ |g a b c - 2| < ε)) :=
by sorry

end NUMINAMATH_CALUDE_g_range_and_density_l3030_303014


namespace NUMINAMATH_CALUDE_congruence_solution_l3030_303008

theorem congruence_solution (n : ℤ) : 13 * n ≡ 19 [ZMOD 47] ↔ n ≡ 30 [ZMOD 47] := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l3030_303008


namespace NUMINAMATH_CALUDE_min_sum_of_digits_for_odd_primes_l3030_303030

def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def n (p : ℕ) : ℕ := p^4 - 5*p^2 + 13

theorem min_sum_of_digits_for_odd_primes :
  ∀ p : ℕ, is_odd_prime p → sum_of_digits (n p) ≥ sum_of_digits (n 5) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_digits_for_odd_primes_l3030_303030


namespace NUMINAMATH_CALUDE_company_managers_count_l3030_303041

theorem company_managers_count 
  (num_associates : ℕ) 
  (avg_salary_managers : ℝ) 
  (avg_salary_associates : ℝ) 
  (avg_salary_company : ℝ) 
  (h1 : num_associates = 75)
  (h2 : avg_salary_managers = 90000)
  (h3 : avg_salary_associates = 30000)
  (h4 : avg_salary_company = 40000) :
  ∃ (num_managers : ℕ), 
    (num_managers : ℝ) * avg_salary_managers + (num_associates : ℝ) * avg_salary_associates = 
    ((num_managers : ℝ) + (num_associates : ℝ)) * avg_salary_company ∧ 
    num_managers = 15 := by
  sorry

end NUMINAMATH_CALUDE_company_managers_count_l3030_303041


namespace NUMINAMATH_CALUDE_absolute_value_quadratic_equation_solution_l3030_303004

theorem absolute_value_quadratic_equation_solution :
  let y₁ : ℝ := (-1 + Real.sqrt 241) / 6
  let y₂ : ℝ := (1 - Real.sqrt 145) / 6
  (|y₁ - 4| + 3 * y₁^2 = 16) ∧ (|y₂ - 4| + 3 * y₂^2 = 16) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_quadratic_equation_solution_l3030_303004


namespace NUMINAMATH_CALUDE_range_of_a_l3030_303031

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp (x - 1) + 2 * x - log a * x / log (sqrt 2)

theorem range_of_a (a : ℝ) (h1 : a > 0) :
  (∃ x y, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) →
  a > 2^(3/2) ∧ a < 2^((exp 1 + 4)/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3030_303031


namespace NUMINAMATH_CALUDE_lecture_arrangements_l3030_303037

theorem lecture_arrangements (n : ℕ) (h : n = 3) : Nat.factorial n = 6 := by
  sorry

end NUMINAMATH_CALUDE_lecture_arrangements_l3030_303037


namespace NUMINAMATH_CALUDE_min_value_x2_2xy_y2_l3030_303099

theorem min_value_x2_2xy_y2 :
  (∀ x y : ℝ, x^2 + 2*x*y + y^2 ≥ 0) ∧
  (∃ x y : ℝ, x^2 + 2*x*y + y^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x2_2xy_y2_l3030_303099


namespace NUMINAMATH_CALUDE_inequalities_theorem_l3030_303064

theorem inequalities_theorem (a b c : ℝ) 
  (h1 : a < 0) (h2 : b > 0) (h3 : a < b) (h4 : b < c) : 
  (a * b < b * c) ∧ 
  (a * c < b * c) ∧ 
  (a * b < a * c) ∧ 
  (a + b < b + c) ∧ 
  (c / a < c / b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3030_303064


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3030_303065

/-- Calculate the combined tax rate for two individuals given their incomes and tax rates -/
theorem combined_tax_rate
  (john_income : ℝ)
  (john_tax_rate : ℝ)
  (ingrid_income : ℝ)
  (ingrid_tax_rate : ℝ)
  (h1 : john_income = 56000)
  (h2 : john_tax_rate = 0.30)
  (h3 : ingrid_income = 72000)
  (h4 : ingrid_tax_rate = 0.40) :
  (john_income * john_tax_rate + ingrid_income * ingrid_tax_rate) / (john_income + ingrid_income) = 0.35625 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l3030_303065


namespace NUMINAMATH_CALUDE_original_price_l3030_303001

-- Define the discount rate
def discount_rate : ℝ := 0.4

-- Define the discounted price
def discounted_price : ℝ := 120

-- Theorem stating the original price
theorem original_price : 
  ∃ (price : ℝ), price * (1 - discount_rate) = discounted_price ∧ price = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_original_price_l3030_303001


namespace NUMINAMATH_CALUDE_intersection_A_B_l3030_303016

-- Define set A
def A : Set ℝ := {x | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3030_303016


namespace NUMINAMATH_CALUDE_overlapping_triangle_is_equilateral_l3030_303038

/-- Represents a right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- Represents the overlapping triangle formed by two identical right-angled triangles -/
structure OverlappingTriangle where
  original : RightTriangle
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- 
Given two identical right-angled triangles arranged such that the right angle vertex 
of one triangle lies on the side of the other, the resulting overlapping triangle is equilateral.
-/
theorem overlapping_triangle_is_equilateral (t : RightTriangle) 
  (ot : OverlappingTriangle) (h : ot.original = t) : 
  ot.side1 = ot.side2 ∧ ot.side2 = ot.side3 := by
  sorry


end NUMINAMATH_CALUDE_overlapping_triangle_is_equilateral_l3030_303038


namespace NUMINAMATH_CALUDE_log_4_30_l3030_303048

theorem log_4_30 (a c : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 5 / Real.log 10 = c) :
  Real.log 30 / Real.log 4 = 1 / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_log_4_30_l3030_303048


namespace NUMINAMATH_CALUDE_flight_duration_sum_main_flight_theorem_l3030_303033

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents a flight with departure and arrival times -/
structure Flight where
  departure : Time
  arrival : Time

def Flight.duration (f : Flight) : ℕ × ℕ :=
  sorry

theorem flight_duration_sum (f : Flight) (time_zone_diff : ℕ) (daylight_saving : ℕ) :
  let (h, m) := f.duration
  h + m = 32 :=
by
  sorry

/-- The main theorem proving the flight duration sum -/
theorem main_flight_theorem : ∃ (f : Flight) (time_zone_diff daylight_saving : ℕ),
  f.departure = ⟨7, 15, sorry⟩ ∧
  f.arrival = ⟨17, 40, sorry⟩ ∧
  time_zone_diff = 3 ∧
  daylight_saving = 1 ∧
  (let (h, m) := f.duration
   0 < m ∧ m < 60 ∧ h + m = 32) :=
by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_main_flight_theorem_l3030_303033


namespace NUMINAMATH_CALUDE_contest_order_l3030_303081

-- Define the contestants
variable (Andy Beth Carol Dave : ℝ)

-- Define the conditions
axiom sum_equality : Andy + Carol = Beth + Dave
axiom interchange_inequality : Beth + Andy > Dave + Carol
axiom carol_highest : Carol > Andy + Beth
axiom nonnegative_scores : Andy ≥ 0 ∧ Beth ≥ 0 ∧ Carol ≥ 0 ∧ Dave ≥ 0

-- Theorem to prove
theorem contest_order : Carol > Beth ∧ Beth > Andy ∧ Andy > Dave := by
  sorry

end NUMINAMATH_CALUDE_contest_order_l3030_303081


namespace NUMINAMATH_CALUDE_factors_of_2520_l3030_303086

theorem factors_of_2520 : Nat.card (Nat.divisors 2520) = 48 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_2520_l3030_303086


namespace NUMINAMATH_CALUDE_number_difference_l3030_303000

theorem number_difference (a b : ℕ) (h1 : b = 10 * a) (h2 : a + b = 30000) :
  b - a = 24543 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3030_303000


namespace NUMINAMATH_CALUDE_parabola_roots_l3030_303060

/-- Given a parabola y = ax^2 - 2ax + c where a ≠ 0 that passes through the point (3, 0),
    prove that the solutions to ax^2 - 2ax + c = 0 are x₁ = -1 and x₂ = 3. -/
theorem parabola_roots (a c : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 - 2*a*x + c = 0 ↔ x = -1 ∨ x = 3) ↔
  a * 3^2 - 2*a*3 + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_roots_l3030_303060


namespace NUMINAMATH_CALUDE_rectangle_area_solution_l3030_303013

/-- A rectangle with dimensions (2x - 3) by (3x + 4) has an area of 14x - 12. -/
theorem rectangle_area_solution (x : ℝ) : 
  (2 * x - 3 > 0) → 
  (3 * x + 4 > 0) → 
  (2 * x - 3) * (3 * x + 4) = 14 * x - 12 → 
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_solution_l3030_303013


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l3030_303057

theorem quadratic_equation_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + a = 0) → a ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l3030_303057


namespace NUMINAMATH_CALUDE_emerald_density_conversion_l3030_303059

/-- Density of a material in g/cm³ -/
def density : ℝ := 2.7

/-- Conversion factor from grams to carats -/
def gramsToCarat : ℝ := 5

/-- Conversion factor from cubic centimeters to cubic inches -/
def cmCubedToInchCubed : ℝ := 16.387

/-- Density of emerald in carats per cubic inch -/
def emeraldDensityCaratsPerCubicInch : ℝ :=
  density * gramsToCarat * cmCubedToInchCubed

theorem emerald_density_conversion :
  ⌊emeraldDensityCaratsPerCubicInch⌋ = 221 := by sorry

end NUMINAMATH_CALUDE_emerald_density_conversion_l3030_303059


namespace NUMINAMATH_CALUDE_tangent_circles_bc_length_l3030_303093

/-- Two externally tangent circles with centers A and B -/
structure TangentCircles where
  A : ℝ × ℝ  -- Center of first circle
  B : ℝ × ℝ  -- Center of second circle
  radius_A : ℝ  -- Radius of first circle
  radius_B : ℝ  -- Radius of second circle
  externally_tangent : ‖A - B‖ = radius_A + radius_B

/-- A line tangent to both circles intersecting ray AB at point C -/
def tangent_line (tc : TangentCircles) (C : ℝ × ℝ) : Prop :=
  ∃ D E : ℝ × ℝ,
    ‖D - tc.A‖ = tc.radius_A ∧
    ‖E - tc.B‖ = tc.radius_B ∧
    (D - C) • (tc.A - C) = 0 ∧
    (E - C) • (tc.B - C) = 0 ∧
    (C - tc.A) • (tc.B - tc.A) ≥ 0

/-- The main theorem -/
theorem tangent_circles_bc_length 
  (tc : TangentCircles) 
  (hA : tc.radius_A = 7)
  (hB : tc.radius_B = 4)
  (C : ℝ × ℝ)
  (h_tangent : tangent_line tc C) :
  ‖C - tc.B‖ = 44 / 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_bc_length_l3030_303093


namespace NUMINAMATH_CALUDE_horners_method_polynomial_transformation_l3030_303009

theorem horners_method_polynomial_transformation (x : ℝ) :
  6 * x^3 + 5 * x^2 + 4 * x + 3 = x * (x * (6 * x + 5) + 4) + 3 := by
  sorry

end NUMINAMATH_CALUDE_horners_method_polynomial_transformation_l3030_303009


namespace NUMINAMATH_CALUDE_sandal_pairs_bought_l3030_303095

def shirt_price : ℕ := 5
def sandal_price : ℕ := 3
def num_shirts : ℕ := 10
def total_paid : ℕ := 100
def change_received : ℕ := 41

theorem sandal_pairs_bought : ℕ := by
  sorry

end NUMINAMATH_CALUDE_sandal_pairs_bought_l3030_303095


namespace NUMINAMATH_CALUDE_well_digging_time_l3030_303034

/-- Represents the time taken to dig a meter at a given depth -/
def digTime (depth : ℕ) : ℕ := 40 + (depth - 1) * 10

/-- Converts minutes to hours -/
def minutesToHours (minutes : ℕ) : ℚ := minutes / 60

theorem well_digging_time :
  minutesToHours (digTime 21) = 4 := by
  sorry

end NUMINAMATH_CALUDE_well_digging_time_l3030_303034


namespace NUMINAMATH_CALUDE_distance_between_trees_l3030_303022

/-- Given a yard of length 150 meters with 11 trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is 15 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 150 →
  num_trees = 11 →
  let num_segments := num_trees - 1
  yard_length / num_segments = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3030_303022


namespace NUMINAMATH_CALUDE_correct_verb_forms_l3030_303074

/-- Represents the grammatical number of a subject --/
inductive GrammaticalNumber
| Singular
| Plural

/-- Represents a subject in a sentence --/
structure Subject where
  text : String
  number : GrammaticalNumber

/-- Represents a verb in a sentence --/
structure Verb where
  singular_form : String
  plural_form : String

/-- Checks if a verb agrees with a subject --/
def verb_agrees (s : Subject) (v : Verb) : Prop :=
  match s.number with
  | GrammaticalNumber.Singular => v.singular_form = "is"
  | GrammaticalNumber.Plural => v.plural_form = "want"

/-- The main theorem stating the correct verb forms for the given subjects --/
theorem correct_verb_forms 
  (subject1 : Subject)
  (subject2 : Subject)
  (h1 : subject1.text = "The number of the stamps")
  (h2 : subject2.text = "a number of people")
  (h3 : subject1.number = GrammaticalNumber.Singular)
  (h4 : subject2.number = GrammaticalNumber.Plural) :
  ∃ (v1 v2 : Verb), 
    verb_agrees subject1 v1 ∧ 
    verb_agrees subject2 v2 ∧ 
    v1.singular_form = "is" ∧ 
    v2.plural_form = "want" := by
  sorry


end NUMINAMATH_CALUDE_correct_verb_forms_l3030_303074


namespace NUMINAMATH_CALUDE_seed_mixture_percentage_l3030_303026

/-- Given two seed mixtures X and Y, and a final mixture composed of X and Y,
    this theorem proves the percentage of X in the final mixture. -/
theorem seed_mixture_percentage
  (x_ryegrass : Real) (x_bluegrass : Real)
  (y_ryegrass : Real) (y_fescue : Real)
  (final_ryegrass : Real) :
  x_ryegrass = 0.40 →
  x_bluegrass = 0.60 →
  y_ryegrass = 0.25 →
  y_fescue = 0.75 →
  final_ryegrass = 0.27 →
  x_ryegrass + x_bluegrass = 1 →
  y_ryegrass + y_fescue = 1 →
  ∃ (p : Real), p * x_ryegrass + (1 - p) * y_ryegrass = final_ryegrass ∧ p = 200 / 15 := by
  sorry

end NUMINAMATH_CALUDE_seed_mixture_percentage_l3030_303026


namespace NUMINAMATH_CALUDE_factorization_of_2x2_minus_2y2_l3030_303006

theorem factorization_of_2x2_minus_2y2 (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x2_minus_2y2_l3030_303006


namespace NUMINAMATH_CALUDE_product_of_square_roots_equals_one_l3030_303018

theorem product_of_square_roots_equals_one :
  let P := Real.sqrt 2012 + Real.sqrt 2013
  let Q := -Real.sqrt 2012 - Real.sqrt 2013
  let R := Real.sqrt 2012 - Real.sqrt 2013
  let S := Real.sqrt 2013 - Real.sqrt 2012
  P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_equals_one_l3030_303018


namespace NUMINAMATH_CALUDE_initial_cookies_count_l3030_303024

/-- The number of cookies Paul took out in 4 days -/
def cookies_taken_4_days : ℕ := 24

/-- The number of days Paul took cookies out -/
def days_taken : ℕ := 4

/-- The number of cookies remaining after a week -/
def cookies_remaining : ℕ := 28

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Proves that the initial number of cookies in the jar is 52 -/
theorem initial_cookies_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_initial_cookies_count_l3030_303024


namespace NUMINAMATH_CALUDE_find_a_l3030_303049

-- Define the sets U and A
def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}
def A (a : ℝ) : Set ℝ := {|2*a - 1|, 2}

-- Define the theorem
theorem find_a : ∃ (a : ℝ), 
  (U a \ A a = {5}) ∧ 
  (A a ⊆ U a) ∧
  (a = 2) := by sorry

end NUMINAMATH_CALUDE_find_a_l3030_303049


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l3030_303005

theorem min_value_quadratic (x : ℝ) : 3 * x^2 - 18 * x + 2048 ≥ 2021 := by sorry

theorem min_value_quadratic_achieved : ∃ x : ℝ, 3 * x^2 - 18 * x + 2048 = 2021 := by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l3030_303005


namespace NUMINAMATH_CALUDE_g_range_l3030_303003

noncomputable def g (x : ℝ) : ℝ :=
  (Real.sin x ^ 4 + 3 * Real.sin x ^ 3 + 5 * Real.sin x ^ 2 + 4 * Real.sin x + 3 * Real.cos x ^ 2 - 9) /
  (Real.sin x - 1)

theorem g_range :
  ∀ x : ℝ, Real.sin x ≠ 1 → 2 ≤ g x ∧ g x < 15 := by
  sorry

end NUMINAMATH_CALUDE_g_range_l3030_303003


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l3030_303070

/-- Two cyclists on a circular track problem -/
theorem cyclists_meeting_time
  (circumference : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : circumference = 180)
  (h2 : speed1 = 7)
  (h3 : speed2 = 8) :
  circumference / (speed1 + speed2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l3030_303070


namespace NUMINAMATH_CALUDE_broker_commission_slump_l3030_303036

theorem broker_commission_slump (initial_rate final_rate : ℝ) 
  (initial_business final_business : ℝ) (h1 : initial_rate = 0.04) 
  (h2 : final_rate = 0.05) (h3 : initial_rate * initial_business = final_rate * final_business) :
  (initial_business - final_business) / initial_business = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_broker_commission_slump_l3030_303036


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3030_303011

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (1 - i) / (1 + i) = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3030_303011


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3030_303027

theorem sufficient_not_necessary (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x - y ≥ 2 → 2^x - 2^y ≥ 3) ∧
  (∃ x y, 2^x - 2^y ≥ 3 ∧ x - y < 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3030_303027


namespace NUMINAMATH_CALUDE_no_solution_condition_l3030_303092

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (3 - 2*x)/(x - 3) - (m*x - 2)/(3 - x) ≠ -1) ↔ 
  (m = 5/3 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3030_303092


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l3030_303069

theorem sequence_sum_problem (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 8)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 81) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ = 425 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l3030_303069


namespace NUMINAMATH_CALUDE_sphere_radius_from_cone_volume_l3030_303068

/-- Given a cone with radius 2 inches and height 8 inches, 
    prove that a sphere with twice the volume of this cone 
    has a radius of 2^(4/3) inches. -/
theorem sphere_radius_from_cone_volume 
  (cone_radius : ℝ) 
  (cone_height : ℝ) 
  (sphere_radius : ℝ) :
  cone_radius = 2 ∧ 
  cone_height = 8 ∧ 
  (4/3) * π * sphere_radius^3 = 2 * ((1/3) * π * cone_radius^2 * cone_height) →
  sphere_radius = 2^(4/3) :=
by
  sorry

#check sphere_radius_from_cone_volume

end NUMINAMATH_CALUDE_sphere_radius_from_cone_volume_l3030_303068


namespace NUMINAMATH_CALUDE_orangeade_water_ratio_l3030_303023

/-- Represents the orangeade mixing and selling scenario over two days -/
structure OrangeadeScenario where
  orange_juice : ℝ  -- Amount of orange juice used (same for both days)
  water_day1 : ℝ    -- Amount of water used on day 1
  water_day2 : ℝ    -- Amount of water used on day 2
  price_day1 : ℝ    -- Price per glass on day 1
  price_day2 : ℝ    -- Price per glass on day 2
  glasses_day1 : ℝ  -- Number of glasses sold on day 1
  glasses_day2 : ℝ  -- Number of glasses sold on day 2

/-- The conditions of the orangeade scenario -/
def scenario_conditions (s : OrangeadeScenario) : Prop :=
  s.orange_juice > 0 ∧
  s.water_day1 = s.orange_juice ∧
  s.price_day1 = 0.48 ∧
  s.glasses_day1 * (s.orange_juice + s.water_day1) = s.glasses_day2 * (s.orange_juice + s.water_day2) ∧
  s.price_day1 * s.glasses_day1 = s.price_day2 * s.glasses_day2

/-- The main theorem: under the given conditions, the ratio of water used on day 2 to orange juice is 1:1 -/
theorem orangeade_water_ratio (s : OrangeadeScenario) 
  (h : scenario_conditions s) : s.water_day2 = s.orange_juice :=
sorry


end NUMINAMATH_CALUDE_orangeade_water_ratio_l3030_303023


namespace NUMINAMATH_CALUDE_pyramid_base_edge_length_l3030_303078

/-- The configuration of five identical balls and a circumscribing square pyramid. -/
structure BallPyramidConfig where
  -- Radius of each ball
  ball_radius : ℝ
  -- Distance between centers of adjacent bottom balls
  bottom_center_distance : ℝ
  -- Height from floor to center of top ball
  top_ball_height : ℝ
  -- Edge length of the square base of the pyramid
  pyramid_base_edge : ℝ

/-- The theorem stating the edge length of the square base of the pyramid. -/
theorem pyramid_base_edge_length 
  (config : BallPyramidConfig) 
  (h1 : config.ball_radius = 2)
  (h2 : config.bottom_center_distance = 2 * config.ball_radius)
  (h3 : config.top_ball_height = 3 * config.ball_radius)
  (h4 : config.pyramid_base_edge = config.bottom_center_distance * Real.sqrt 2) :
  config.pyramid_base_edge = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_base_edge_length_l3030_303078


namespace NUMINAMATH_CALUDE_intersection_points_problem_l3030_303058

/-- The number of intersection points in the first quadrant given points on x and y axes -/
def intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- The theorem stating the number of intersection points for the given problem -/
theorem intersection_points_problem :
  intersection_points 15 10 = 4725 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_problem_l3030_303058


namespace NUMINAMATH_CALUDE_max_trip_weight_l3030_303076

def max_crates : ℕ := 5
def min_crate_weight : ℕ := 1250

theorem max_trip_weight :
  max_crates * min_crate_weight = 6250 :=
by sorry

end NUMINAMATH_CALUDE_max_trip_weight_l3030_303076


namespace NUMINAMATH_CALUDE_vector_properties_l3030_303073

/-- Given vectors a and b in ℝ², prove they are perpendicular and satisfy certain magnitude properties -/
theorem vector_properties (a b : ℝ × ℝ) (h1 : a = (2, 4)) (h2 : b = (-2, 1)) :
  (a.1 * b.1 + a.2 * b.2 = 0) ∧ 
  (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5) ∧
  (Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l3030_303073


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3030_303062

theorem absolute_value_equality (x : ℝ) : 
  |(-x)| = |(-8)| → x = 8 ∨ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3030_303062


namespace NUMINAMATH_CALUDE_a_squared_plus_reciprocal_squared_is_integer_l3030_303043

theorem a_squared_plus_reciprocal_squared_is_integer (a : ℝ) (h : ∃ k : ℤ, a + 1 / a = k) :
  ∃ m : ℤ, a^2 + 1 / a^2 = m := by
sorry

end NUMINAMATH_CALUDE_a_squared_plus_reciprocal_squared_is_integer_l3030_303043


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l3030_303067

/-- The function f(x) = x^2 + 12x - 15 -/
def f (x : ℝ) : ℝ := x^2 + 12*x - 15

theorem root_exists_in_interval :
  (f 1.1 < 0) → (f 1.2 > 0) → ∃ x ∈ Set.Ioo 1.1 1.2, f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l3030_303067
