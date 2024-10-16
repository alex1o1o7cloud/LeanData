import Mathlib

namespace NUMINAMATH_CALUDE_sixth_term_is_three_l3298_329854

/-- An arithmetic sequence with 10 terms where the sum of even-numbered terms is 15 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n m, a (n + 1) - a n = a (m + 1) - a m) ∧ 
  (a 2 + a 4 + a 6 + a 8 + a 10 = 15)

/-- The 6th term of the arithmetic sequence is 3 -/
theorem sixth_term_is_three (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  a 6 = 3 := by sorry

end NUMINAMATH_CALUDE_sixth_term_is_three_l3298_329854


namespace NUMINAMATH_CALUDE_late_attendees_fraction_l3298_329878

theorem late_attendees_fraction 
  (total : ℕ) 
  (total_pos : total > 0)
  (male_fraction : Rat)
  (male_on_time_fraction : Rat)
  (female_on_time_fraction : Rat)
  (h_male : male_fraction = 2 / 3)
  (h_male_on_time : male_on_time_fraction = 3 / 4)
  (h_female_on_time : female_on_time_fraction = 5 / 6) :
  (1 : Rat) - (male_fraction * male_on_time_fraction + (1 - male_fraction) * female_on_time_fraction) = 2 / 9 := by
  sorry

#check late_attendees_fraction

end NUMINAMATH_CALUDE_late_attendees_fraction_l3298_329878


namespace NUMINAMATH_CALUDE_exotic_fruit_distribution_l3298_329813

theorem exotic_fruit_distribution (eldest_fruits second_fruits third_fruits : ℕ) 
  (gold_to_eldest gold_to_second : ℕ) :
  eldest_fruits = 2 * second_fruits / 3 →
  third_fruits = 0 →
  gold_to_eldest + gold_to_second = 180 →
  eldest_fruits - gold_to_eldest / (180 / (gold_to_eldest + gold_to_second)) = 
    second_fruits - gold_to_second / (180 / (gold_to_eldest + gold_to_second)) →
  eldest_fruits - gold_to_eldest / (180 / (gold_to_eldest + gold_to_second)) = 
    (gold_to_eldest + gold_to_second) / (180 / (gold_to_eldest + gold_to_second)) →
  gold_to_second = 144 :=
by sorry

end NUMINAMATH_CALUDE_exotic_fruit_distribution_l3298_329813


namespace NUMINAMATH_CALUDE_ptolemy_special_cases_l3298_329833

/-- Ptolemy's theorem for cyclic quadrilaterals -/
def ptolemyTheorem (a b c d e f : ℝ) : Prop := a * c + b * d = e * f

/-- A cyclic quadrilateral with one side zero -/
def cyclicQuadrilateralOneSideZero (b c d e f : ℝ) : Prop :=
  ptolemyTheorem 0 b c d e f

/-- A rectangle -/
def rectangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

/-- An isosceles trapezoid -/
def isoscelesTrapezoid (a b c e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ e > 0

theorem ptolemy_special_cases :
  (∀ b c d e f : ℝ, cyclicQuadrilateralOneSideZero b c d e f → b * d = e * f) ∧
  (∀ a b : ℝ, rectangle a b → 2 * a * b = a^2 + b^2) ∧
  (∀ a b c e : ℝ, isoscelesTrapezoid a b c e → e^2 = c^2 + a * b) :=
sorry

end NUMINAMATH_CALUDE_ptolemy_special_cases_l3298_329833


namespace NUMINAMATH_CALUDE_four_row_triangle_count_l3298_329856

/-- Calculates the total number of triangles in a triangular grid with n rows -/
def triangleCount (n : ℕ) : ℕ :=
  let smallTriangles := n * (n + 1) / 2
  let mediumTriangles := (n - 1) * (n - 2) / 2
  let largeTriangles := n - 2
  smallTriangles + mediumTriangles + largeTriangles

/-- Theorem stating that a triangular grid with 4 rows contains 14 triangles in total -/
theorem four_row_triangle_count : triangleCount 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_four_row_triangle_count_l3298_329856


namespace NUMINAMATH_CALUDE_age_difference_l3298_329865

-- Define the ages as natural numbers
def rona_age : ℕ := 8
def rachel_age : ℕ := 2 * rona_age
def collete_age : ℕ := rona_age / 2

-- Theorem statement
theorem age_difference : rachel_age - collete_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3298_329865


namespace NUMINAMATH_CALUDE_solve_for_a_l3298_329886

theorem solve_for_a (m d b a : ℝ) (h1 : m = (d * a * b) / (a - b)) (h2 : m ≠ d * b) :
  a = (m * b) / (m - d * b) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3298_329886


namespace NUMINAMATH_CALUDE_renovation_project_equation_l3298_329859

/-- Represents the relationship between the number of workers hired in a renovation project --/
theorem renovation_project_equation (x y : ℕ) : 
  (∀ (carpenter_wage mason_wage labor_budget : ℕ), 
    carpenter_wage = 50 ∧ 
    mason_wage = 40 ∧ 
    labor_budget = 2000 → 
    50 * x + 40 * y ≤ 2000) ↔ 
  5 * x + 4 * y ≤ 200 :=
by sorry

end NUMINAMATH_CALUDE_renovation_project_equation_l3298_329859


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l3298_329896

theorem angle_in_third_quadrant (α : Real) (h1 : π < α ∧ α < 3*π/2) : 
  (Real.sin (π/2 - α) * Real.cos (-α) * Real.tan (π + α)) / Real.cos (π - α) = 2 * Real.sqrt 5 / 5 →
  Real.cos α = -(Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l3298_329896


namespace NUMINAMATH_CALUDE_equation_solution_l3298_329893

theorem equation_solution :
  ∃ x : ℚ, (2 * x + 5 * x = 500 - (4 * x + 6 * x + 10)) ∧ x = 490 / 17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3298_329893


namespace NUMINAMATH_CALUDE_g_of_3_equals_101_l3298_329816

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := 5 * x^3 - 4 * x^2 + 3 * x - 7

-- Theorem stating that g(3) = 101
theorem g_of_3_equals_101 : g 3 = 101 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_101_l3298_329816


namespace NUMINAMATH_CALUDE_problem_solution_l3298_329844

theorem problem_solution (a b m n : ℚ) 
  (ha_neg : a < 0) 
  (ha_abs : |a| = 7/4)
  (hb_recip : 1/b = -3/2)
  (hmn_opp : m = -n) :
  4*a / b + 3*(m + n) = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3298_329844


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_congruence_l3298_329803

theorem smallest_four_digit_number_congruence (x : ℕ) : 
  (x ≥ 1000 ∧ x < 10000) →
  (3 * x ≡ 9 [ZMOD 18]) →
  (5 * x + 20 ≡ 30 [ZMOD 15]) →
  (3 * x - 4 ≡ 2 * x [ZMOD 35]) →
  x ≥ 1004 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_congruence_l3298_329803


namespace NUMINAMATH_CALUDE_perfect_square_function_characterization_l3298_329861

/-- A function g: ℕ → ℕ satisfies the perfect square property if 
    (g(m) + n)(m + g(n)) is a perfect square for all m, n ∈ ℕ -/
def PerfectSquareProperty (g : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (m + g n) = k * k

/-- The main theorem characterizing functions with the perfect square property -/
theorem perfect_square_function_characterization :
  ∀ g : ℕ → ℕ, PerfectSquareProperty g ↔ ∃ c : ℕ, ∀ n : ℕ, g n = n + c :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_function_characterization_l3298_329861


namespace NUMINAMATH_CALUDE_arithmetic_progression_product_power_l3298_329818

theorem arithmetic_progression_product_power : ∃ (a b : ℕ), 
  a > 0 ∧ 
  (a * (2*a) * (3*a) * (4*a) * (5*a) = b^2008) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_product_power_l3298_329818


namespace NUMINAMATH_CALUDE_skater_practice_hours_l3298_329848

/-- Given a skater's practice schedule, calculate the total weekly practice hours. -/
theorem skater_practice_hours (weekend_hours : ℕ) (additional_weekday_hours : ℕ) : 
  weekend_hours = 8 → additional_weekday_hours = 17 → 
  weekend_hours + (weekend_hours + additional_weekday_hours) = 33 := by
  sorry

#check skater_practice_hours

end NUMINAMATH_CALUDE_skater_practice_hours_l3298_329848


namespace NUMINAMATH_CALUDE_apple_pricing_discrepancy_l3298_329873

def vendor1_rate : ℚ := 2 / 3
def vendor2_rate : ℚ := 1
def day1_total_apples : ℕ := 60
def day1_total_earnings : ℕ := 50
def day2_rate : ℚ := 4 / 5
def day2_total_apples : ℕ := 60
def day2_total_earnings : ℕ := 48

theorem apple_pricing_discrepancy :
  ∃ (day1_avg_price day2_avg_price : ℚ),
    day1_avg_price = day1_total_earnings / day1_total_apples ∧
    day2_avg_price = day2_rate ∧
    (day1_avg_price - day2_avg_price) * day1_total_apples = 2 :=
by sorry

end NUMINAMATH_CALUDE_apple_pricing_discrepancy_l3298_329873


namespace NUMINAMATH_CALUDE_game_d_higher_prob_l3298_329804

def coin_prob_tails : ℚ := 3/4
def coin_prob_heads : ℚ := 1/4

def game_c_win_prob : ℚ := 2 * (coin_prob_heads * coin_prob_tails)

def game_d_win_prob : ℚ := 
  3 * (coin_prob_tails^2 * coin_prob_heads) + coin_prob_tails^3

theorem game_d_higher_prob : 
  game_d_win_prob - game_c_win_prob = 15/32 :=
sorry

end NUMINAMATH_CALUDE_game_d_higher_prob_l3298_329804


namespace NUMINAMATH_CALUDE_modified_goldbach_for_2024_l3298_329817

theorem modified_goldbach_for_2024 :
  ∃ (p q : ℕ), p ≠ q ∧ Prime p ∧ Prime q ∧ p + q = 2024 :=
by
  sorry

#check modified_goldbach_for_2024

end NUMINAMATH_CALUDE_modified_goldbach_for_2024_l3298_329817


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l3298_329876

open Complex

theorem complex_equation_solutions :
  ∃ (S : Finset ℂ), 
    (∀ z ∈ S, Complex.abs z < 25 ∧ Complex.exp z = (z + 1) / (z - 1)) ∧
    S.card = 8 ∧
    (∀ z : ℂ, Complex.abs z < 25 ∧ Complex.exp z = (z + 1) / (z - 1) → z ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l3298_329876


namespace NUMINAMATH_CALUDE_robert_remaining_kicks_l3298_329815

/-- Calculates the remaining kicks needed to reach a goal. -/
def remaining_kicks (total_goal : ℕ) (kicks_before_break : ℕ) (kicks_after_break : ℕ) : ℕ :=
  total_goal - (kicks_before_break + kicks_after_break)

/-- Proves that given the specific conditions, the remaining kicks is 19. -/
theorem robert_remaining_kicks : 
  remaining_kicks 98 43 36 = 19 := by
  sorry

end NUMINAMATH_CALUDE_robert_remaining_kicks_l3298_329815


namespace NUMINAMATH_CALUDE_seokgi_money_problem_l3298_329883

theorem seokgi_money_problem (initial_money : ℕ) : 
  (initial_money / 2) / 2 = 1250 → initial_money = 5000 := by
  sorry

end NUMINAMATH_CALUDE_seokgi_money_problem_l3298_329883


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3298_329846

theorem division_remainder_problem (x y : ℤ) (r : ℕ) 
  (h1 : x > 0)
  (h2 : x = 10 * y + r)
  (h3 : 0 ≤ r ∧ r < 10)
  (h4 : 2 * x = 7 * (3 * y) + 1)
  (h5 : 11 * y - x = 2) :
  r = 3 := by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3298_329846


namespace NUMINAMATH_CALUDE_rotated_square_distance_l3298_329835

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the configuration of four squares -/
structure SquareConfiguration where
  squares : Fin 4 → Square
  aligned : Bool
  rotatedSquareIndex : Fin 4
  rotatedSquareTouching : Bool

/-- The distance from the top vertex of the rotated square to the original line -/
def distanceToOriginalLine (config : SquareConfiguration) : ℝ :=
  sorry

theorem rotated_square_distance
  (config : SquareConfiguration)
  (h1 : ∀ i, (config.squares i).sideLength = 2)
  (h2 : config.aligned)
  (h3 : config.rotatedSquareIndex = 1)
  (h4 : config.rotatedSquareTouching) :
  distanceToOriginalLine config = 2 :=
sorry

end NUMINAMATH_CALUDE_rotated_square_distance_l3298_329835


namespace NUMINAMATH_CALUDE_no_real_roots_l3298_329800

theorem no_real_roots (c : ℝ) (h : c > 1) : ∀ x : ℝ, x^2 + 2*x + c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3298_329800


namespace NUMINAMATH_CALUDE_gcd_seven_factorial_six_factorial_l3298_329860

theorem gcd_seven_factorial_six_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 6) = Nat.factorial 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_seven_factorial_six_factorial_l3298_329860


namespace NUMINAMATH_CALUDE_incorrect_statement_l3298_329892

theorem incorrect_statement :
  ¬(∀ (p q : Prop), (¬p ∧ ¬q) → ¬(p ∧ q)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_statement_l3298_329892


namespace NUMINAMATH_CALUDE_ratio_equality_l3298_329863

theorem ratio_equality (x : ℝ) :
  (0.75 / x = 5 / 8) → x = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3298_329863


namespace NUMINAMATH_CALUDE_fish_count_l3298_329823

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 9

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 19 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l3298_329823


namespace NUMINAMATH_CALUDE_composite_polynomial_l3298_329806

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 9*n^2 + 27*n + 35 = a * b :=
by sorry

end NUMINAMATH_CALUDE_composite_polynomial_l3298_329806


namespace NUMINAMATH_CALUDE_one_third_coloring_ways_l3298_329819

/-- The number of ways to choose k items from a set of n items -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The number of triangles in the square -/
def total_triangles : ℕ := 18

/-- The number of triangles to be colored -/
def colored_triangles : ℕ := 6

/-- Theorem stating that the number of ways to color one-third of the square is 18564 -/
theorem one_third_coloring_ways :
  binomial total_triangles colored_triangles = 18564 := by
  sorry

end NUMINAMATH_CALUDE_one_third_coloring_ways_l3298_329819


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l3298_329805

/-- The probability of having a boy or a girl -/
def child_probability : ℚ := 1 / 2

/-- The number of children in the family -/
def family_size : ℕ := 4

/-- The probability of having at least one boy and one girl in a family with four children -/
theorem prob_at_least_one_boy_and_girl : 
  (1 : ℚ) - (child_probability ^ family_size + child_probability ^ family_size) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l3298_329805


namespace NUMINAMATH_CALUDE_encyclopedia_cost_l3298_329857

/-- Proves that the cost of encyclopedias is approximately $1002.86 given the specified conditions --/
theorem encyclopedia_cost (down_payment : ℝ) (monthly_payment : ℝ) (num_monthly_payments : ℕ)
  (final_payment : ℝ) (interest_rate : ℝ) :
  down_payment = 300 →
  monthly_payment = 57 →
  num_monthly_payments = 9 →
  final_payment = 21 →
  interest_rate = 18.666666666666668 / 100 →
  ∃ (cost : ℝ), abs (cost - 1002.86) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_encyclopedia_cost_l3298_329857


namespace NUMINAMATH_CALUDE_sin_sum_product_zero_l3298_329851

theorem sin_sum_product_zero : 
  Real.sin (523 * π / 180) * Real.sin (943 * π / 180) + 
  Real.sin (1333 * π / 180) * Real.sin (313 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_product_zero_l3298_329851


namespace NUMINAMATH_CALUDE_count_numbers_divisible_by_33_l3298_329864

def is_valid_number (x y : ℕ) : Prop :=
  x ≤ 9 ∧ y ≤ 9

def number_value (x y : ℕ) : ℕ :=
  2007000000 + x * 100000 + 2008 + y

theorem count_numbers_divisible_by_33 :
  ∃ (S : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ S ↔ 
      is_valid_number p.1 p.2 ∧ 
      (number_value p.1 p.2) % 33 = 0) ∧
    Finset.card S = 3 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_divisible_by_33_l3298_329864


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3298_329837

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 6*x^2 - 4 = (x-1)*(x+1)*(x-Real.sqrt 2)^2*(x+Real.sqrt 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3298_329837


namespace NUMINAMATH_CALUDE_alex_age_l3298_329867

/-- Given the ages of Alex, Bella, and Carlos, prove that Alex is 20 years old. -/
theorem alex_age (bella_age carlos_age alex_age : ℕ) : 
  bella_age = 21 →
  carlos_age = bella_age + 5 →
  alex_age = carlos_age - 6 →
  alex_age = 20 := by
sorry

end NUMINAMATH_CALUDE_alex_age_l3298_329867


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3298_329828

theorem largest_multiple_of_15_under_500 :
  ∃ (n : ℕ), n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ (m : ℕ), m * 15 < 500 → m * 15 ≤ 495 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3298_329828


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3298_329802

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x > 1}
def N : Set ℝ := {x | ∃ y, y = Real.log (3/2 - x)}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 3/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3298_329802


namespace NUMINAMATH_CALUDE_inequality_proof_l3298_329820

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : a * (c^2 - 1) = b * (b^2 + c^2)) 
  (h_d : d ≤ 1) : 
  d * (a * Real.sqrt (1 - d^2) + b^2 * Real.sqrt (1 + d^2)) ≤ (a + b) * c / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3298_329820


namespace NUMINAMATH_CALUDE_marks_initial_fries_l3298_329801

/-- Given that Sally had 14 fries initially, Mark gave her one-third of his fries,
    and Sally ended up with 26 fries, prove that Mark initially had 36 fries. -/
theorem marks_initial_fries (sally_initial : ℕ) (sally_final : ℕ) (mark_fraction : ℚ) :
  sally_initial = 14 →
  sally_final = 26 →
  mark_fraction = 1 / 3 →
  ∃ (mark_initial : ℕ), 
    mark_initial = 36 ∧
    sally_final = sally_initial + mark_fraction * mark_initial :=
by sorry

end NUMINAMATH_CALUDE_marks_initial_fries_l3298_329801


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3298_329830

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3298_329830


namespace NUMINAMATH_CALUDE_eight_people_twentyeight_handshakes_l3298_329808

/-- The number of handshakes in a function where every person shakes hands with every other person exactly once -/
def total_handshakes : ℕ := 28

/-- Calculates the number of handshakes given the number of people -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Proves that 8 people results in 28 handshakes -/
theorem eight_people_twentyeight_handshakes :
  ∃ (n : ℕ), n > 0 ∧ handshakes n = total_handshakes ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_eight_people_twentyeight_handshakes_l3298_329808


namespace NUMINAMATH_CALUDE_no_integer_roots_for_both_equations_l3298_329850

theorem no_integer_roots_for_both_equations :
  ¬∃ (b c : ℝ), 
    (∃ (p q : ℤ), p ≠ q ∧ (p : ℝ)^2 + b*(p : ℝ) + c = 0 ∧ (q : ℝ)^2 + b*(q : ℝ) + c = 0) ∧
    (∃ (r s : ℤ), r ≠ s ∧ 2*(r : ℝ)^2 + (b+1)*(r : ℝ) + (c+1) = 0 ∧ 2*(s : ℝ)^2 + (b+1)*(s : ℝ) + (c+1) = 0) :=
sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_both_equations_l3298_329850


namespace NUMINAMATH_CALUDE_multiple_factor_statement_l3298_329870

theorem multiple_factor_statement (h : 8 * 9 = 72) : ¬(∃ k : ℕ, 72 = 8 * k ∧ ∃ m : ℕ, 72 = m * 8) :=
sorry

end NUMINAMATH_CALUDE_multiple_factor_statement_l3298_329870


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3298_329840

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3298_329840


namespace NUMINAMATH_CALUDE_team_transfer_equation_l3298_329866

theorem team_transfer_equation (x : ℤ) : 
  let team_a_initial : ℤ := 37
  let team_b_initial : ℤ := 23
  let team_a_final : ℤ := team_a_initial + x
  let team_b_final : ℤ := team_b_initial - x
  team_a_final = 2 * team_b_final →
  37 + x = 2 * (23 - x) :=
by
  sorry

end NUMINAMATH_CALUDE_team_transfer_equation_l3298_329866


namespace NUMINAMATH_CALUDE_armans_sister_age_l3298_329894

theorem armans_sister_age (arman_age sister_age : ℚ) : 
  arman_age = 6 * sister_age →
  arman_age + 4 = 40 →
  sister_age - 4 = 16 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_armans_sister_age_l3298_329894


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3298_329822

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ, (2 * a^2 + 7 * a - 15 = 0) ∧ (2 * b^2 + 7 * b - 15 = 0) → (a - b)^2 = 169/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3298_329822


namespace NUMINAMATH_CALUDE_lucy_groceries_l3298_329809

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 2

/-- The number of packs of cake Lucy bought -/
def cake : ℕ := 12

/-- The total number of grocery packs Lucy bought -/
def total_groceries : ℕ := cookies + cake

theorem lucy_groceries : total_groceries = 14 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l3298_329809


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3298_329899

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3298_329899


namespace NUMINAMATH_CALUDE_set_A_equals_interval_rep_l3298_329872

-- Define the set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 5 ∨ x > 10}

-- Define the interval representation
def intervalRep : Set ℝ := Set.Ici 0 ∩ Set.Iio 5 ∪ Set.Ioi 10

-- Theorem statement
theorem set_A_equals_interval_rep : A = intervalRep := by sorry

end NUMINAMATH_CALUDE_set_A_equals_interval_rep_l3298_329872


namespace NUMINAMATH_CALUDE_simplify_expression_l3298_329832

theorem simplify_expression : (45000 - 32000) * 10 + (2500 / 5) - 21005 * 3 = 67485 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3298_329832


namespace NUMINAMATH_CALUDE_ball_travel_distance_l3298_329814

/-- The distance traveled by the center of a ball on a track with three semicircular arcs -/
theorem ball_travel_distance (ball_diameter : ℝ) (R₁ R₂ R₃ : ℝ) : 
  ball_diameter = 8 →
  R₁ = 110 →
  R₂ = 70 →
  R₃ = 90 →
  (π / 2) * ((R₁ - ball_diameter / 2) + (R₂ + ball_diameter / 2) + (R₃ - ball_diameter / 2)) = 266 * π := by
  sorry

end NUMINAMATH_CALUDE_ball_travel_distance_l3298_329814


namespace NUMINAMATH_CALUDE_stimulus_savings_amount_l3298_329849

def stimulus_distribution (initial_amount : ℚ) : ℚ :=
  let wife_share := (2 / 5) * initial_amount
  let after_wife := initial_amount - wife_share
  let first_son_share := (2 / 5) * after_wife
  let after_first_son := after_wife - first_son_share
  let second_son_share := (40 / 100) * after_first_son
  after_first_son - second_son_share

theorem stimulus_savings_amount :
  stimulus_distribution 2000 = 432 := by
  sorry

end NUMINAMATH_CALUDE_stimulus_savings_amount_l3298_329849


namespace NUMINAMATH_CALUDE_geometric_sum_abs_l3298_329875

def geometric_sequence (n : ℕ) (a₁ : ℝ) (r : ℝ) : ℝ := a₁ * r^(n-1)

theorem geometric_sum_abs (a₁ r : ℝ) (h : a₁ = 1 ∧ r = -2) :
  let a := geometric_sequence
  a 1 a₁ r + |a 2 a₁ r| + |a 3 a₁ r| + a 4 a₁ r = 15 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_abs_l3298_329875


namespace NUMINAMATH_CALUDE_factorization_proof_l3298_329862

theorem factorization_proof (a b x y m : ℝ) : 
  (3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2) ∧ 
  (x^2 * (m - 2) + y^2 * (2 - m) = (m - 2) * (x + y) * (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3298_329862


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l3298_329898

theorem cistern_emptying_time (empty_rate : ℚ) (time : ℕ) (portion : ℚ) :
  empty_rate = 1/3 ∧ time = 6 ∧ portion = 2/3 →
  (portion / empty_rate) * time = 12 := by
sorry

end NUMINAMATH_CALUDE_cistern_emptying_time_l3298_329898


namespace NUMINAMATH_CALUDE_no_partition_exists_l3298_329881

theorem no_partition_exists : ¬∃ (A B C : Set ℕ), 
  (A ∪ B ∪ C = Set.univ) ∧ 
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
  (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
  (∀ a b, a ∈ A → b ∈ B → a + b + 2008 ∈ C) ∧
  (∀ b c, b ∈ B → c ∈ C → b + c + 2008 ∈ A) ∧
  (∀ c a, c ∈ C → a ∈ A → c + a + 2008 ∈ B) := by
sorry

end NUMINAMATH_CALUDE_no_partition_exists_l3298_329881


namespace NUMINAMATH_CALUDE_subset_condition_l3298_329824

def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 2| < a}

def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

theorem subset_condition (a : ℝ) : B ⊆ A a ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l3298_329824


namespace NUMINAMATH_CALUDE_max_value_when_m_3_solution_f_geq_0_l3298_329871

-- Define the function f(x, m)
def f (x m : ℝ) : ℝ := |x - m| - 2 * |x - 1|

-- Theorem for the maximum value when m = 3
theorem max_value_when_m_3 :
  ∃ (max : ℝ), max = 2 ∧ ∀ x, f x 3 ≤ max :=
sorry

-- Theorem for the solution of f(x) ≥ 0
theorem solution_f_geq_0 (m : ℝ) :
  (m > 1 → ∀ x, f x m ≥ 0 ↔ 2 - m ≤ x ∧ x ≤ (2 + m) / 3) ∧
  (m = 1 → ∀ x, f x m ≥ 0 ↔ x = 1) ∧
  (m < 1 → ∀ x, f x m ≥ 0 ↔ (2 + m) / 3 ≤ x ∧ x ≤ 2 - m) :=
sorry

end NUMINAMATH_CALUDE_max_value_when_m_3_solution_f_geq_0_l3298_329871


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l3298_329831

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- Probability function for a normal random variable -/
noncomputable def P (ξ : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry 
  (ξ : NormalRV) 
  (h1 : ξ.μ = 2) 
  (h2 : P ξ 4 = 0.84) : 
  P ξ 0 = 0.16 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l3298_329831


namespace NUMINAMATH_CALUDE_tan_x_minus_pi_sixth_l3298_329812

theorem tan_x_minus_pi_sixth (x : Real) 
  (h : Real.sin (π / 3 - x) = (1 / 2) * Real.cos (x - π / 2)) : 
  Real.tan (x - π / 6) = Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_minus_pi_sixth_l3298_329812


namespace NUMINAMATH_CALUDE_opposite_numbers_and_reciprocal_l3298_329811

theorem opposite_numbers_and_reciprocal (a b c : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : 1 / c = 4)  -- the reciprocal of c is 4
  : 3 * a + 3 * b - 4 * c = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_and_reciprocal_l3298_329811


namespace NUMINAMATH_CALUDE_f_recursive_relation_l3298_329821

def f (n : ℕ) : ℕ := (Finset.range (2 * n + 1)).sum (λ i => i * i)

theorem f_recursive_relation (k : ℕ) : f (k + 1) = f k + (2 * k + 1)^2 + (2 * k + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_f_recursive_relation_l3298_329821


namespace NUMINAMATH_CALUDE_choir_average_age_l3298_329855

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℝ) 
  (avg_age_males : ℝ) 
  (h1 : num_females = 8)
  (h2 : avg_age_females = 25)
  (h3 : num_males = 12)
  (h4 : avg_age_males = 40)
  (h5 : num_females + num_males = 20) :
  (num_females * avg_age_females + num_males * avg_age_males) / (num_females + num_males) = 34 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l3298_329855


namespace NUMINAMATH_CALUDE_subcommittee_count_l3298_329825

/-- The number of members in the planning committee -/
def total_members : ℕ := 12

/-- The number of teachers in the planning committee -/
def teacher_count : ℕ := 5

/-- The size of the subcommittee to be formed -/
def subcommittee_size : ℕ := 4

/-- The minimum number of teachers required in the subcommittee -/
def min_teachers : ℕ := 2

/-- Calculates the number of valid subcommittees -/
def valid_subcommittees : ℕ := 285

theorem subcommittee_count :
  (Nat.choose total_members subcommittee_size) -
  (Nat.choose (total_members - teacher_count) subcommittee_size) -
  (Nat.choose teacher_count 1 * Nat.choose (total_members - teacher_count) (subcommittee_size - 1)) =
  valid_subcommittees :=
sorry

end NUMINAMATH_CALUDE_subcommittee_count_l3298_329825


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l3298_329842

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 5) :
  let min_area := min (a * b / 2) ((a * Real.sqrt (a^2 - b^2)) / 2)
  min_area = (5 * Real.sqrt 11) / 2 := by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l3298_329842


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l3298_329897

theorem subtraction_multiplication_equality : (3.456 - 1.234) * 0.5 = 1.111 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l3298_329897


namespace NUMINAMATH_CALUDE_train_length_l3298_329847

/-- The length of a train given its speed and time to cross a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 * (5 / 18) → time = 16 → speed * time = 320 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3298_329847


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_A_l3298_329868

theorem partial_fraction_decomposition_A (x A B C : ℝ) :
  (1 : ℝ) / (x^3 - x^2 - 17*x + 45) = A / (x + 5) + B / (x - 3) + C / (x + 3) →
  A = (1 : ℝ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_A_l3298_329868


namespace NUMINAMATH_CALUDE_smallest_value_theorem_l3298_329874

theorem smallest_value_theorem (a : ℝ) (h : 8 * a^3 + 6 * a^2 + 7 * a + 5 = 4) :
  ∃ (min_val : ℝ), min_val = (1 : ℝ) / 2 ∧ ∀ (x : ℝ), 8 * x^3 + 6 * x^2 + 7 * x + 5 = 4 → 3 * x + 2 ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_theorem_l3298_329874


namespace NUMINAMATH_CALUDE_min_lines_for_37_segments_l3298_329887

/-- Represents an open non-self-intersecting broken line -/
structure BrokenLine where
  segments : ℕ
  is_open : Bool
  is_non_self_intersecting : Bool

/-- Represents the minimum number of lines needed to cover all segments of a broken line -/
def minimum_lines (bl : BrokenLine) : ℕ := sorry

/-- The theorem stating the minimum number of lines for a 37-segment broken line -/
theorem min_lines_for_37_segments (bl : BrokenLine) : 
  bl.segments = 37 → bl.is_open = true → bl.is_non_self_intersecting = true →
  minimum_lines bl = 9 := by sorry

end NUMINAMATH_CALUDE_min_lines_for_37_segments_l3298_329887


namespace NUMINAMATH_CALUDE_one_in_A_l3298_329826

def A : Set ℝ := {x : ℝ | x ≥ -1}

theorem one_in_A : (1 : ℝ) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_one_in_A_l3298_329826


namespace NUMINAMATH_CALUDE_no_real_m_for_single_root_l3298_329888

theorem no_real_m_for_single_root : 
  ¬∃ (m : ℝ), (∀ (x : ℝ), x^2 + (4*m+2)*x + m = 0 ↔ x = -2*m-1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_m_for_single_root_l3298_329888


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l3298_329895

/-- The function f(x) = (1/5)^(x+1) + m does not pass through the first quadrant if and only if m ≤ -1/5 -/
theorem function_not_in_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → (1/5)^(x+1) + m ≤ 0) ↔ m ≤ -1/5 := by sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l3298_329895


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l3298_329839

theorem arithmetic_simplification :
  (0.25 * 4 - (5/6 + 1/12) * 6/5 = 1/10) ∧
  ((5/12 - 5/16) * 4/5 + 2/3 - 3/4 = 0) := by sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l3298_329839


namespace NUMINAMATH_CALUDE_line_parameterization_l3298_329884

/-- Given a line y = 2x - 30 parameterized by (x, y) = (f t, 20t - 10),
    prove that f t = 10t + 10 for all t. -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t, 2 * (f t) - 30 = 20 * t - 10) → 
  (∀ t, f t = 10 * t + 10) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3298_329884


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3298_329845

/-- The first equation has a unique solution x = 4 -/
theorem equation_one_solution :
  ∃! x : ℝ, (5 / (x + 1) = 1 / (x - 3)) ∧ (x + 1 ≠ 0) ∧ (x - 3 ≠ 0) :=
sorry

/-- The second equation has no solution -/
theorem equation_two_no_solution :
  ¬∃ x : ℝ, ((3 - x) / (x - 4) = 1 / (4 - x) - 2) ∧ (x - 4 ≠ 0) ∧ (4 - x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3298_329845


namespace NUMINAMATH_CALUDE_investment_change_l3298_329869

theorem investment_change (initial_value : ℝ) (h : initial_value > 0) : 
  let day1_value := initial_value * 1.4
  let day2_value := day1_value * 0.75
  (day2_value - initial_value) / initial_value = 0.05 := by
sorry

end NUMINAMATH_CALUDE_investment_change_l3298_329869


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3298_329836

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- Distance squared between two points -/
def distanceSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- A line in parametric form -/
structure Line where
  a : ℝ  -- x-intercept
  α : ℝ  -- angle of inclination

/-- Get points where a line intersects the parabola -/
def lineParabolaIntersection (l : Line) : Set Point :=
  {p : Point | p ∈ Parabola ∧ ∃ t : ℝ, p.x = l.a + t * Real.cos l.α ∧ p.y = t * Real.sin l.α}

/-- The theorem to be proved -/
theorem parabola_intersection_theorem (a : ℝ) :
  (∀ l : Line, l.a = a →
    let M : Point := ⟨a, 0⟩
    let intersections := lineParabolaIntersection l
    ∃ k : ℝ, ∀ P Q : Point, P ∈ intersections → Q ∈ intersections → P ≠ Q →
      1 / distanceSquared P M + 1 / distanceSquared Q M = k) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3298_329836


namespace NUMINAMATH_CALUDE_combined_stock_cost_value_l3298_329853

/-- Calculate the final cost of a stock given its initial parameters -/
def calculate_stock_cost (initial_price discount brokerage tax_rate transaction_fee : ℚ) : ℚ :=
  let discounted_price := initial_price * (1 - discount)
  let brokerage_fee := discounted_price * brokerage
  let net_purchase_price := discounted_price + brokerage_fee
  let tax := net_purchase_price * tax_rate
  net_purchase_price + tax + transaction_fee

/-- The combined cost of three stocks with given parameters -/
def combined_stock_cost : ℚ :=
  calculate_stock_cost 100 (4/100) (1/500) (12/100) 2 +
  calculate_stock_cost 200 (6/100) (1/400) (10/100) 3 +
  calculate_stock_cost 150 (3/100) (1/200) (15/100) 1

/-- Theorem stating the combined cost of the three stocks -/
theorem combined_stock_cost_value : 
  combined_stock_cost = 489213665/1000000 := by sorry

end NUMINAMATH_CALUDE_combined_stock_cost_value_l3298_329853


namespace NUMINAMATH_CALUDE_rectangle_tileable_iff_divisible_l3298_329891

/-- An (0, b)-tile is a 2 × b rectangle. -/
structure ZeroBTile (b : ℕ) :=
  (width : Fin 2)
  (height : Fin b)

/-- A tiling of an m × n rectangle with (0, b)-tiles. -/
def Tiling (m n b : ℕ) := List (ZeroBTile b)

/-- Predicate to check if a tiling is valid for an m × n rectangle. -/
def IsValidTiling (m n b : ℕ) (t : Tiling m n b) : Prop :=
  sorry  -- Definition of valid tiling omitted for brevity

/-- An m × n rectangle is (0, b)-tileable if there exists a valid tiling. -/
def IsTileable (m n b : ℕ) : Prop :=
  ∃ t : Tiling m n b, IsValidTiling m n b t

/-- Main theorem: An m × n rectangle is (0, b)-tileable iff 2b divides m or 2b divides n. -/
theorem rectangle_tileable_iff_divisible (m n b : ℕ) (hm : m > 0) (hn : n > 0) (hb : b > 0) :
  IsTileable m n b ↔ (2 * b ∣ m) ∨ (2 * b ∣ n) :=
sorry

end NUMINAMATH_CALUDE_rectangle_tileable_iff_divisible_l3298_329891


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l3298_329882

theorem fenced_area_calculation : 
  let yard_length : ℕ := 20
  let yard_width : ℕ := 18
  let large_cutout_side : ℕ := 4
  let small_cutout_side : ℕ := 2
  let yard_area := yard_length * yard_width
  let large_cutout_area := large_cutout_side * large_cutout_side
  let small_cutout_area := small_cutout_side * small_cutout_side
  yard_area - large_cutout_area - small_cutout_area = 340 := by
sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l3298_329882


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3298_329890

theorem algebraic_expression_value (x y : ℝ) (h : x + y = 2) :
  (1/2) * x^2 + x * y + (1/2) * y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3298_329890


namespace NUMINAMATH_CALUDE_contrapositive_example_l3298_329834

theorem contrapositive_example :
  (∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∀ x : ℝ, x^2 ≤ 1 → x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l3298_329834


namespace NUMINAMATH_CALUDE_divisible_by_two_and_three_l3298_329885

theorem divisible_by_two_and_three (n : ℕ) : 
  (∃ (k : ℕ), k = 33 ∧ k = (n.div 6).succ) ↔ n = 204 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_two_and_three_l3298_329885


namespace NUMINAMATH_CALUDE_completing_square_addition_l3298_329877

theorem completing_square_addition (x : ℝ) : 
  (∃ k : ℝ, (x^2 - 4*x + k)^(1/2) = x - 2) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_addition_l3298_329877


namespace NUMINAMATH_CALUDE_total_fish_caught_l3298_329838

/-- Calculates the total number of fish caught by Brian, Chris, and Dave given specific fishing frequencies and catch rates. -/
theorem total_fish_caught
  (chris_trips : ℕ)
  (brian_fish_per_trip : ℕ)
  (h1 : chris_trips = 10)
  (h2 : brian_fish_per_trip = 400)
  (h3 : (2 : ℚ) / 5 < 1) :
  ∃ (total : ℕ),
    total = 
      (2 * chris_trips * brian_fish_per_trip) +
      (chris_trips * (brian_fish_per_trip * 5 / 3 : ℚ).ceil) +
      (3 * chris_trips * ((brian_fish_per_trip * 5 / 3 : ℚ).ceil * 3 / 2 : ℚ).ceil) ∧
    total = 38800 :=
by sorry

end NUMINAMATH_CALUDE_total_fish_caught_l3298_329838


namespace NUMINAMATH_CALUDE_constant_speed_l3298_329841

-- Define the type of the position function
def PositionFunction := ℝ → ℝ → ℝ

-- Define the property of being a polynomial in two variables
def IsPolynomial (f : PositionFunction) : Prop := sorry

-- Define the functional equation property
def SatisfiesFunctionalEquation (p : PositionFunction) : Prop :=
  ∀ t k x, p t x - p k x = p (t - k) (p k x)

-- State the theorem
theorem constant_speed
  (p : PositionFunction)
  (h_poly : IsPolynomial p)
  (h_func_eq : SatisfiesFunctionalEquation p) :
  ∃ a : ℝ, ∀ t x, p t x = x + a * t :=
sorry

end NUMINAMATH_CALUDE_constant_speed_l3298_329841


namespace NUMINAMATH_CALUDE_joe_monthly_income_correct_l3298_329879

/-- Joe's monthly income in dollars -/
def monthly_income : ℝ := 2120

/-- The fraction of Joe's income that goes to taxes -/
def tax_rate : ℝ := 0.4

/-- The amount Joe pays in taxes each month in dollars -/
def tax_paid : ℝ := 848

/-- Theorem stating that Joe's monthly income is correct given the tax rate and tax paid -/
theorem joe_monthly_income_correct : 
  tax_rate * monthly_income = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_joe_monthly_income_correct_l3298_329879


namespace NUMINAMATH_CALUDE_probability_yellow_chalk_l3298_329843

/-- The number of yellow chalks in the box -/
def yellow_chalks : ℕ := 3

/-- The number of red chalks in the box -/
def red_chalks : ℕ := 2

/-- The total number of chalks in the box -/
def total_chalks : ℕ := yellow_chalks + red_chalks

/-- The probability of selecting a yellow chalk -/
def prob_yellow : ℚ := yellow_chalks / total_chalks

theorem probability_yellow_chalk :
  prob_yellow = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_yellow_chalk_l3298_329843


namespace NUMINAMATH_CALUDE_sum_after_100_operations_l3298_329889

def initial_sequence : List ℕ := [2, 11, 8, 9]

def operation (seq : List ℤ) : List ℤ :=
  seq ++ (seq.zip (seq.tail!)).map (fun (a, b) => b - a)

def sum_after_n_operations (n : ℕ) : ℤ :=
  30 + 7 * n

theorem sum_after_100_operations :
  sum_after_n_operations 100 = 730 :=
sorry

end NUMINAMATH_CALUDE_sum_after_100_operations_l3298_329889


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l3298_329810

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l3298_329810


namespace NUMINAMATH_CALUDE_sqrt_seven_fraction_l3298_329852

theorem sqrt_seven_fraction (p q : ℝ) (hp : p > 0) (hq : q > 0) (h : Real.sqrt 7 = p / q) :
  Real.sqrt 7 = (7 * q - 2 * p) / (p - 2 * q) ∧ p - 2 * q > 0 ∧ p - 2 * q < q := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_fraction_l3298_329852


namespace NUMINAMATH_CALUDE_girls_percentage_increase_l3298_329858

theorem girls_percentage_increase (initial_boys : ℕ) (final_total : ℕ) : 
  initial_boys = 15 →
  final_total = 51 →
  ∃ (initial_girls : ℕ),
    initial_girls = initial_boys + (initial_boys * 20 / 100) ∧
    final_total = initial_boys + 2 * initial_girls :=
by sorry

end NUMINAMATH_CALUDE_girls_percentage_increase_l3298_329858


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_negation_equivalence_necessary_sufficient_condition_l3298_329829

-- Statement 1
theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  ¬(∀ x : ℝ, |x| > 1 → x > 1) :=
sorry

-- Statement 2
theorem negation_equivalence :
  ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
sorry

-- Statement 3
theorem necessary_sufficient_condition (a b c : ℝ) :
  (a + b + c = 0) ↔ (a * 1^2 + b * 1 + c = 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_negation_equivalence_necessary_sufficient_condition_l3298_329829


namespace NUMINAMATH_CALUDE_floor_sqrt_27_squared_l3298_329827

theorem floor_sqrt_27_squared : ⌊Real.sqrt 27⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_27_squared_l3298_329827


namespace NUMINAMATH_CALUDE_slope_implies_y_coordinate_l3298_329807

/-- Given two points A and B in a coordinate plane, if the slope of the line through A and B is 1/3, then the y-coordinate of B is 12. -/
theorem slope_implies_y_coordinate
  (xA yA xB : ℝ)
  (h1 : xA = -3)
  (h2 : yA = 9)
  (h3 : xB = 6) :
  (yB - yA) / (xB - xA) = 1/3 → yB = 12 :=
by sorry

end NUMINAMATH_CALUDE_slope_implies_y_coordinate_l3298_329807


namespace NUMINAMATH_CALUDE_circle_triangle_area_l3298_329880

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def CircleTangentToLine (c : Circle) (l : Line) : Prop := sorry

def CirclesInternallyTangent (c1 c2 : Circle) : Prop := sorry

def CirclesExternallyTangent (c1 c2 : Circle) : Prop := sorry

def PointBetween (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

def AreaOfTriangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem circle_triangle_area 
  (A B C : Circle)
  (m : Line)
  (A' B' C' : ℝ × ℝ) :
  A.radius = 3 →
  B.radius = 4 →
  C.radius = 5 →
  CircleTangentToLine A m →
  CircleTangentToLine B m →
  CircleTangentToLine C m →
  PointBetween A' B' C' →
  CirclesInternallyTangent A B →
  CirclesExternallyTangent B C →
  AreaOfTriangle A.center B.center C.center = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_area_l3298_329880
