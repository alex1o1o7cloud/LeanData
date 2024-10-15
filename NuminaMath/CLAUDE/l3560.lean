import Mathlib

namespace NUMINAMATH_CALUDE_ball_max_height_l3560_356075

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

-- Theorem statement
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 40 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l3560_356075


namespace NUMINAMATH_CALUDE_smallest_divisible_by_8_9_11_l3560_356036

theorem smallest_divisible_by_8_9_11 : ∀ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 9 ∣ n ∧ 11 ∣ n → n ≥ 792 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_8_9_11_l3560_356036


namespace NUMINAMATH_CALUDE_lining_fabric_cost_l3560_356047

/-- The cost of lining fabric per yard -/
def lining_cost : ℝ := 30.69

theorem lining_fabric_cost :
  let velvet_cost : ℝ := 24
  let pattern_cost : ℝ := 15
  let thread_cost : ℝ := 3 * 2
  let buttons_cost : ℝ := 14
  let trim_cost : ℝ := 19 * 3
  let velvet_yards : ℝ := 5
  let lining_yards : ℝ := 4
  let discount_rate : ℝ := 0.1
  let total_cost : ℝ := 310.50
  
  total_cost = (1 - discount_rate) * (velvet_cost * velvet_yards + lining_cost * lining_yards) +
               pattern_cost + thread_cost + buttons_cost + trim_cost :=
by sorry


end NUMINAMATH_CALUDE_lining_fabric_cost_l3560_356047


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3560_356024

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 + Complex.I) ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3560_356024


namespace NUMINAMATH_CALUDE_total_books_stu_and_albert_l3560_356060

/-- Given that Stu has 9 books and Albert has 4 times as many books as Stu,
    prove that the total number of books Stu and Albert have is 45. -/
theorem total_books_stu_and_albert :
  let stu_books : ℕ := 9
  let albert_books : ℕ := 4 * stu_books
  stu_books + albert_books = 45 := by
sorry

end NUMINAMATH_CALUDE_total_books_stu_and_albert_l3560_356060


namespace NUMINAMATH_CALUDE_unique_prime_pair_solution_l3560_356079

theorem unique_prime_pair_solution : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (3 * p^(q-1) + 1) ∣ (11^p + 17^p) → 
    p = 3 ∧ q = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_solution_l3560_356079


namespace NUMINAMATH_CALUDE_nancy_files_distribution_l3560_356004

/-- Given the initial number of files, number of deleted files, and number of folders,
    calculate the number of files in each folder after distribution. -/
def filesPerFolder (initialFiles deletedFiles numFolders : ℕ) : ℕ :=
  (initialFiles - deletedFiles) / numFolders

/-- Prove that given 80 initial files, after deleting 31 files and distributing
    the remaining files equally into 7 folders, each folder contains 7 files. -/
theorem nancy_files_distribution :
  filesPerFolder 80 31 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_files_distribution_l3560_356004


namespace NUMINAMATH_CALUDE_vector_opposite_direction_l3560_356026

/-- Given two vectors a and b in ℝ², where a = (1, -1), |b| = |a|, and b is in the opposite direction of a, prove that b = (-1, 1). -/
theorem vector_opposite_direction (a b : ℝ × ℝ) : 
  a = (1, -1) → 
  ‖b‖ = ‖a‖ → 
  ∃ (k : ℝ), k < 0 ∧ b = k • a → 
  b = (-1, 1) := by
sorry

end NUMINAMATH_CALUDE_vector_opposite_direction_l3560_356026


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3560_356073

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let parabola := fun (x y : ℝ) ↦ y^2 = 4 * x
  let directrix := fun (x : ℝ) ↦ x = -1
  let asymptote1 := fun (x y : ℝ) ↦ y = (b / a) * x
  let asymptote2 := fun (x y : ℝ) ↦ y = -(b / a) * x
  let triangle_area := 2 * Real.sqrt 3
  let eccentricity := Real.sqrt ((a^2 + b^2) / a^2)
  (∃ A B : ℝ × ℝ, 
    directrix A.1 ∧ asymptote1 A.1 A.2 ∧
    directrix B.1 ∧ asymptote2 B.1 B.2 ∧
    (1/2) * (A.2 - B.2) = triangle_area) →
  eccentricity = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3560_356073


namespace NUMINAMATH_CALUDE_even_divisors_of_factorial_8_l3560_356022

/-- The factorial of 8 -/
def factorial_8 : ℕ := 40320

/-- The prime factorization of 8! -/
axiom factorial_8_factorization : factorial_8 = 2^7 * 3^2 * 5 * 7

/-- A function that counts the number of even divisors of a natural number -/
def count_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating that 8! has 84 even divisors -/
theorem even_divisors_of_factorial_8 :
  count_even_divisors factorial_8 = 84 := by sorry

end NUMINAMATH_CALUDE_even_divisors_of_factorial_8_l3560_356022


namespace NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l3560_356069

theorem five_fourths_of_twelve_fifths (x : ℚ) : x = 12 / 5 → (5 / 4) * x = 3 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l3560_356069


namespace NUMINAMATH_CALUDE_cost_per_bag_first_is_24_l3560_356059

/-- The cost per bag of zongzi in the first batch -/
def cost_per_bag_first : ℝ := 24

/-- The total cost of the first batch of zongzi -/
def total_cost_first : ℝ := 3000

/-- The total cost of the second batch of zongzi -/
def total_cost_second : ℝ := 7500

/-- The number of bags in the second batch is three times the number in the first batch -/
def batch_ratio : ℝ := 3

/-- The cost difference per bag between the first and second batch -/
def cost_difference : ℝ := 4

theorem cost_per_bag_first_is_24 :
  cost_per_bag_first = 24 ∧
  total_cost_first = 3000 ∧
  total_cost_second = 7500 ∧
  batch_ratio = 3 ∧
  cost_difference = 4 →
  cost_per_bag_first = 24 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_bag_first_is_24_l3560_356059


namespace NUMINAMATH_CALUDE_solve_for_a_l3560_356041

-- Define the equation
def equation (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

-- Define the theorem
theorem solve_for_a :
  ∀ a : ℝ, equation a 15 7 → a * 15 * 7 = 1.5 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3560_356041


namespace NUMINAMATH_CALUDE_total_selling_price_proof_l3560_356028

def calculate_selling_price (cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost * (1 + profit_percentage)

theorem total_selling_price_proof :
  let cost_prices : List ℝ := [280, 350, 500]
  let profit_percentages : List ℝ := [0.30, 0.45, 0.25]
  let selling_prices := List.zipWith calculate_selling_price cost_prices profit_percentages
  List.sum selling_prices = 1496.50 := by
sorry

end NUMINAMATH_CALUDE_total_selling_price_proof_l3560_356028


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l3560_356029

/-- The number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  (man_age - 2 * son_age) / (2 - 1)

/-- Theorem: Given the son's age is 35 and the age difference is 37, 
    the number of years until the man's age is twice his son's age is 2 -/
theorem double_age_in_two_years (son_age : ℕ) (age_difference : ℕ) 
  (h1 : son_age = 35) (h2 : age_difference = 37) : 
  years_until_double_age son_age age_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_age_in_two_years_l3560_356029


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l3560_356070

theorem man_son_age_ratio :
  ∀ (man_age son_age : ℕ),
    man_age = 36 →
    son_age = 12 →
    man_age + 12 = 2 * (son_age + 12) →
    man_age / son_age = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l3560_356070


namespace NUMINAMATH_CALUDE_john_total_spent_l3560_356051

/-- The total amount John spends on t-shirts and pants -/
def total_spent (num_tshirts : ℕ) (price_per_tshirt : ℕ) (pants_cost : ℕ) : ℕ :=
  num_tshirts * price_per_tshirt + pants_cost

/-- Theorem: John spends $110 in total -/
theorem john_total_spent :
  total_spent 3 20 50 = 110 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spent_l3560_356051


namespace NUMINAMATH_CALUDE_largest_number_l3560_356091

theorem largest_number (a b c d : ℝ) (h1 : a = -1) (h2 : b = 0) (h3 : c = 1) (h4 : d = 2) :
  d ≥ a ∧ d ≥ b ∧ d ≥ c := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3560_356091


namespace NUMINAMATH_CALUDE_point_on_y_axis_l3560_356001

/-- A point lies on the y-axis if and only if its x-coordinate is 0 -/
def lies_on_y_axis (x y : ℝ) : Prop := x = 0

/-- The theorem states that if the point (a+1, a-1) lies on the y-axis, then a = -1 -/
theorem point_on_y_axis (a : ℝ) : lies_on_y_axis (a + 1) (a - 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3560_356001


namespace NUMINAMATH_CALUDE_harvard_mit_puzzle_l3560_356018

/-- Given that the product of letters in "harvard", "mit", and "hmmt" all equal 100,
    prove that the product of letters in "rad" and "trivia" equals 10000. -/
theorem harvard_mit_puzzle (h a r v d m i t : ℕ) : 
  h * a * r * v * a * r * d = 100 →
  m * i * t = 100 →
  h * m * m * t = 100 →
  (r * a * d) * (t * r * i * v * i * a) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_harvard_mit_puzzle_l3560_356018


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l3560_356031

theorem quadratic_perfect_square (p : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 + 27*x + p = (a*x + b)^2) → p = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l3560_356031


namespace NUMINAMATH_CALUDE_problem_1_l3560_356081

theorem problem_1 : (1/3)⁻¹ + Real.sqrt 18 - 4 * Real.cos (π/4) = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3560_356081


namespace NUMINAMATH_CALUDE_complex_modulus_l3560_356063

theorem complex_modulus (x y : ℝ) (z : ℂ) : 
  z = x + y * Complex.I → 
  (1/2 * x - y : ℂ) + (x + y) * Complex.I = 3 * Complex.I → 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3560_356063


namespace NUMINAMATH_CALUDE_car_profit_percent_l3560_356092

/-- Calculate the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent 
  (purchase_price : ℝ) 
  (repair_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 42000) 
  (h2 : repair_cost = 12000) 
  (h3 : selling_price = 64900) : 
  ∃ (profit_percent : ℝ), abs (profit_percent - 20.19) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_car_profit_percent_l3560_356092


namespace NUMINAMATH_CALUDE_john_crab_earnings_l3560_356019

/-- Calculates the weekly earnings from crab sales given the following conditions:
  * Number of baskets reeled in per collection
  * Number of collections per week
  * Number of crabs per basket
  * Price per crab
-/
def weekly_crab_earnings (baskets_per_collection : ℕ) (collections_per_week : ℕ) (crabs_per_basket : ℕ) (price_per_crab : ℕ) : ℕ :=
  baskets_per_collection * collections_per_week * crabs_per_basket * price_per_crab

/-- Theorem stating that under the given conditions, John makes $72 per week from selling crabs -/
theorem john_crab_earnings :
  weekly_crab_earnings 3 2 4 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_john_crab_earnings_l3560_356019


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l3560_356061

theorem salary_reduction_percentage (S : ℝ) (R : ℝ) :
  S > 0 →
  (S - (R / 100 * S)) * (1 + 1 / 3) = S →
  R = 25 := by
sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l3560_356061


namespace NUMINAMATH_CALUDE_average_score_of_group_specific_group_average_l3560_356085

theorem average_score_of_group (total_people : ℕ) (group1_size : ℕ) (group2_size : ℕ) 
  (group1_avg : ℝ) (group2_avg : ℝ) :
  total_people = group1_size + group2_size →
  (group1_size : ℝ) * group1_avg + (group2_size : ℝ) * group2_avg = 
    (total_people : ℝ) * ((group1_size : ℝ) * group1_avg + (group2_size : ℝ) * group2_avg) / (total_people : ℝ) :=
by
  sorry

-- The specific problem instance
theorem specific_group_average :
  let total_people : ℕ := 10
  let group1_size : ℕ := 6
  let group2_size : ℕ := 4
  let group1_avg : ℝ := 90
  let group2_avg : ℝ := 80
  ((group1_size : ℝ) * group1_avg + (group2_size : ℝ) * group2_avg) / (total_people : ℝ) = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_average_score_of_group_specific_group_average_l3560_356085


namespace NUMINAMATH_CALUDE_larger_number_problem_l3560_356042

theorem larger_number_problem (x y : ℝ) (h1 : x - y = 8) (h2 : (1/4) * (x + y) = 6) : max x y = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3560_356042


namespace NUMINAMATH_CALUDE_barry_fifth_game_yards_l3560_356045

theorem barry_fifth_game_yards (game1 game2 game3 game4 game6 : ℕ) 
  (h1 : game1 = 98)
  (h2 : game2 = 107)
  (h3 : game3 = 85)
  (h4 : game4 = 89)
  (h5 : game6 ≥ 130)
  (h6 : (game1 + game2 + game3 + game4 + game6 : ℚ) / 6 > 100) :
  ∃ game5 : ℕ, game5 = 91 ∧ (game1 + game2 + game3 + game4 + game5 + game6 : ℚ) / 6 > 100 := by
sorry

end NUMINAMATH_CALUDE_barry_fifth_game_yards_l3560_356045


namespace NUMINAMATH_CALUDE_remainder_problem_l3560_356087

theorem remainder_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) (hmn : m > n) 
  (hm_mod : m % 6 = 2) (hdiff_mod : (m - n) % 6 = 5) : n % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3560_356087


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l3560_356020

/-- Represents an equilateral triangle with pegs -/
structure TriangleWithPegs where
  sideLength : ℕ
  pegDistance : ℕ

/-- Counts the number of ways to choose pegs that divide the triangle into 9 regions -/
def countValidPegChoices (t : TriangleWithPegs) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem triangle_division_theorem (t : TriangleWithPegs) :
  t.sideLength = 6 ∧ t.pegDistance = 1 → countValidPegChoices t = 456 :=
sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l3560_356020


namespace NUMINAMATH_CALUDE_jellybean_problem_l3560_356000

theorem jellybean_problem (J : ℕ) : 
  J - 15 + 5 - 4 = 23 → J = 33 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l3560_356000


namespace NUMINAMATH_CALUDE_total_cost_is_240000_l3560_356014

/-- The total cost of three necklaces and a set of earrings -/
def total_cost (necklace_price : ℕ) : ℕ :=
  3 * necklace_price + 3 * necklace_price

/-- Proof that the total cost is $240,000 -/
theorem total_cost_is_240000 :
  total_cost 40000 = 240000 := by
  sorry

#eval total_cost 40000

end NUMINAMATH_CALUDE_total_cost_is_240000_l3560_356014


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l3560_356064

def arithmetic_sequence (a d : ℝ) : ℕ → ℝ
  | 0 => a
  | n+1 => arithmetic_sequence a d n + d

theorem ratio_a_to_b (a d : ℝ) :
  let b := a + 3 * d
  (arithmetic_sequence a d 0 = a) ∧
  (arithmetic_sequence a d 1 = a + 2*d) ∧
  (arithmetic_sequence a d 2 = a + 3*d) ∧
  (arithmetic_sequence a d 3 = a + 5*d) →
  a / b = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l3560_356064


namespace NUMINAMATH_CALUDE_zoo_count_l3560_356099

/-- Represents the number of peacocks in the zoo -/
def num_peacocks : ℕ := 7

/-- Represents the number of tortoises in the zoo -/
def num_tortoises : ℕ := 17 - num_peacocks

/-- The total number of legs in the zoo -/
def total_legs : ℕ := 54

/-- The total number of heads in the zoo -/
def total_heads : ℕ := 17

/-- Each peacock has 2 legs -/
def peacock_legs : ℕ := 2

/-- Each peacock has 1 head -/
def peacock_head : ℕ := 1

/-- Each tortoise has 4 legs -/
def tortoise_legs : ℕ := 4

/-- Each tortoise has 1 head -/
def tortoise_head : ℕ := 1

theorem zoo_count :
  num_peacocks * peacock_legs + num_tortoises * tortoise_legs = total_legs ∧
  num_peacocks * peacock_head + num_tortoises * tortoise_head = total_heads :=
by sorry

end NUMINAMATH_CALUDE_zoo_count_l3560_356099


namespace NUMINAMATH_CALUDE_statement_true_except_two_and_five_l3560_356090

theorem statement_true_except_two_and_five (x : ℝ) :
  (x - 2) * (x - 5) ≠ 0 ↔ x ≠ 2 ∧ x ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_statement_true_except_two_and_five_l3560_356090


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3560_356032

theorem expand_and_simplify (x : ℝ) : 
  (x + 2)^2 + x * (3 - x) = 7 * x + 4 := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3560_356032


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l3560_356025

theorem stratified_sampling_sample_size 
  (high_school_students : ℕ) 
  (junior_high_students : ℕ) 
  (high_school_sample : ℕ) 
  (h1 : high_school_students = 3500)
  (h2 : junior_high_students = 1500)
  (h3 : high_school_sample = 70) :
  let total_students := high_school_students + junior_high_students
  let sample_proportion := high_school_sample / high_school_students
  let total_sample_size := total_students * sample_proportion
  total_sample_size = 100 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l3560_356025


namespace NUMINAMATH_CALUDE_rectangle_area_l3560_356072

theorem rectangle_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (a^2 * b) / (b^2 * a) = 5/8 →
  (a + 6) * (b + 6) - a * b = 114 →
  a * b = 40 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3560_356072


namespace NUMINAMATH_CALUDE_units_digit_of_large_power_l3560_356015

/-- The units' digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The result of 3824^428 -/
def large_power : ℕ := 3824^428

theorem units_digit_of_large_power :
  units_digit large_power = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_large_power_l3560_356015


namespace NUMINAMATH_CALUDE_fraction_equality_l3560_356048

theorem fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b * (a + b) = 1) :
  a / (a^3 + a + 1) = b / (b^3 + b + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3560_356048


namespace NUMINAMATH_CALUDE_max_fridays_12th_non_leap_year_max_fridays_12th_leap_year_l3560_356077

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

/-- Represents a month of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Returns the number of days in a given month for a given year -/
def daysInMonth (m : Month) (isLeapYear : Bool) : Nat :=
  match m with
  | .January => 31
  | .February => if isLeapYear then 29 else 28
  | .March => 31
  | .April => 30
  | .May => 31
  | .June => 30
  | .July => 31
  | .August => 31
  | .September => 30
  | .October => 31
  | .November => 30
  | .December => 31

/-- Returns the day of the week for the 12th of a given month, 
    given the day of the week of January 1st -/
def dayOfWeekOn12th (m : Month) (jan1 : DayOfWeek) (isLeapYear : Bool) : DayOfWeek :=
  sorry

/-- Counts the number of Fridays that fall on the 12th in a year -/
def countFridays12th (jan1 : DayOfWeek) (isLeapYear : Bool) : Nat :=
  sorry

/-- Theorem: In a non-leap year, there can be at most 3 Fridays 
    that fall on the 12th of a month -/
theorem max_fridays_12th_non_leap_year :
  ∀ (jan1 : DayOfWeek), countFridays12th jan1 false ≤ 3 :=
  sorry

/-- Theorem: In a leap year, there can be at most 4 Fridays 
    that fall on the 12th of a month -/
theorem max_fridays_12th_leap_year :
  ∀ (jan1 : DayOfWeek), countFridays12th jan1 true ≤ 4 :=
  sorry

end NUMINAMATH_CALUDE_max_fridays_12th_non_leap_year_max_fridays_12th_leap_year_l3560_356077


namespace NUMINAMATH_CALUDE_adlai_chickens_l3560_356033

def num_dogs : ℕ := 2
def total_legs : ℕ := 10
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

theorem adlai_chickens :
  (total_legs - num_dogs * legs_per_dog) / legs_per_chicken = 1 := by
  sorry

end NUMINAMATH_CALUDE_adlai_chickens_l3560_356033


namespace NUMINAMATH_CALUDE_profit_percentage_example_l3560_356030

/-- Calculates the profit percentage given the selling price and cost price. -/
def profit_percentage (selling_price cost_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that for a selling price of 250 and a cost price of 200, 
    the profit percentage is 25%. -/
theorem profit_percentage_example : 
  profit_percentage 250 200 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_example_l3560_356030


namespace NUMINAMATH_CALUDE_division_calculation_l3560_356043

theorem division_calculation : 250 / (5 + 15 * 3^2) = 25 / 14 := by
  sorry

end NUMINAMATH_CALUDE_division_calculation_l3560_356043


namespace NUMINAMATH_CALUDE_larger_cube_volume_l3560_356068

theorem larger_cube_volume (v : ℝ) (k : ℝ) : 
  v = 216 → k = 2.5 → (k * (v ^ (1/3 : ℝ)))^3 = 3375 := by
  sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l3560_356068


namespace NUMINAMATH_CALUDE_beetle_distance_theorem_l3560_356027

def beetle_crawl (start : ℤ) (stop1 : ℤ) (stop2 : ℤ) : ℕ :=
  (Int.natAbs (stop1 - start)) + (Int.natAbs (stop2 - stop1))

theorem beetle_distance_theorem :
  beetle_crawl 3 (-5) 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_beetle_distance_theorem_l3560_356027


namespace NUMINAMATH_CALUDE_power_division_l3560_356003

theorem power_division (a : ℝ) (h : a ≠ 0) : a^6 / a^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l3560_356003


namespace NUMINAMATH_CALUDE_tennis_players_count_l3560_356071

theorem tennis_players_count (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 40)
  (h2 : badminton = 20)
  (h3 : neither = 5)
  (h4 : both = 3)
  : ∃ tennis : ℕ, tennis = 18 ∧ 
    total = badminton + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_tennis_players_count_l3560_356071


namespace NUMINAMATH_CALUDE_smallest_x_value_l3560_356017

theorem smallest_x_value (x : ℝ) :
  (x^2 - 5*x - 84) / (x - 9) = 4 / (x + 6) →
  x ≥ (-13 - Real.sqrt 17) / 2 ∧
  ∃ y : ℝ, y < (-13 - Real.sqrt 17) / 2 ∧ (y^2 - 5*y - 84) / (y - 9) ≠ 4 / (y + 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3560_356017


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3560_356088

theorem absolute_value_inequality (x : ℝ) : |2*x - 1| ≤ 1 ↔ 0 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3560_356088


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l3560_356011

/-- Calculates the gain percent given the cost price and selling price -/
def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The gain percent is 8% when a cycle is bought for Rs. 1000 and sold for Rs. 1080 -/
theorem cycle_gain_percent :
  gain_percent 1000 1080 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l3560_356011


namespace NUMINAMATH_CALUDE_stones_per_bracelet_l3560_356021

theorem stones_per_bracelet (total_stones : Float) (num_bracelets : Float) 
  (h1 : total_stones = 88.0)
  (h2 : num_bracelets = 8.0) :
  total_stones / num_bracelets = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_stones_per_bracelet_l3560_356021


namespace NUMINAMATH_CALUDE_parabola_coef_sum_zero_l3560_356050

/-- A parabola with equation y = ax^2 + bx + c, vertex (3, 4), and passing through (1, 0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 4
  point_x : ℝ := 1
  point_y : ℝ := 0
  eq_at_vertex : vertex_y = a * vertex_x^2 + b * vertex_x + c
  eq_at_point : point_y = a * point_x^2 + b * point_x + c

/-- The sum of coefficients a, b, and c for the specified parabola is 0 -/
theorem parabola_coef_sum_zero (p : Parabola) : p.a + p.b + p.c = 0 := by
  sorry


end NUMINAMATH_CALUDE_parabola_coef_sum_zero_l3560_356050


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_no_minimum_l3560_356038

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_monotone_decreasing_no_minimum :
  (∀ x y : ℝ, x < y → f x > f y) ∧ 
  (∀ ε > 0, ∃ x : ℝ, f x < ε) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_no_minimum_l3560_356038


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l3560_356034

-- Problem 1
theorem calculation_proof :
  Real.sqrt 4 - 2 * Real.sin (45 * π / 180) + (1/3)⁻¹ + |-(Real.sqrt 2)| = 5 := by sorry

-- Problem 2
theorem inequality_system_solution (x : ℝ) :
  (3*x + 1 < 2*x + 3 ∧ 2*x > (3*x - 1)/2) ↔ (-1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l3560_356034


namespace NUMINAMATH_CALUDE_helen_cookies_theorem_l3560_356005

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 31

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_day_before_yesterday : ℕ := 419

/-- The total number of cookies Helen baked till last night -/
def total_cookies_till_last_night : ℕ := cookies_yesterday + cookies_day_before_yesterday

theorem helen_cookies_theorem : total_cookies_till_last_night = 450 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_theorem_l3560_356005


namespace NUMINAMATH_CALUDE_fair_coin_heads_then_tails_l3560_356023

/-- The probability of getting heads on a single flip of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The probability of getting tails on a single flip of a fair coin. -/
def prob_tails : ℚ := 1/2

/-- The probability of getting heads on the first flip and tails on the second flip
    of a fair coin. -/
def prob_heads_then_tails : ℚ := prob_heads * prob_tails

theorem fair_coin_heads_then_tails :
  prob_heads_then_tails = 1/4 := by sorry

end NUMINAMATH_CALUDE_fair_coin_heads_then_tails_l3560_356023


namespace NUMINAMATH_CALUDE_ac_values_l3560_356057

theorem ac_values (a c : ℝ) (h : ∀ x, 2 * Real.sin (3 * x) = a * Real.cos (3 * x + c)) :
  ∃ k : ℤ, a * c = (4 * k - 1) * Real.pi :=
sorry

end NUMINAMATH_CALUDE_ac_values_l3560_356057


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3560_356058

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (x - 1)^4 = x^4 - 4*x^3 + 6*x^2 - 4*x + 1 := by
  sorry

#check coefficient_x_squared_in_expansion

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3560_356058


namespace NUMINAMATH_CALUDE_basketball_score_proof_l3560_356074

theorem basketball_score_proof :
  ∀ (two_pointers three_pointers free_throws : ℕ),
    2 * two_pointers = 3 * three_pointers →
    free_throws = 2 * two_pointers →
    2 * two_pointers + 3 * three_pointers + free_throws = 78 →
    free_throws = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l3560_356074


namespace NUMINAMATH_CALUDE_zeta_power_sum_l3560_356002

theorem zeta_power_sum (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 20) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 54 := by
  sorry

end NUMINAMATH_CALUDE_zeta_power_sum_l3560_356002


namespace NUMINAMATH_CALUDE_six_digit_integers_count_l3560_356016

/-- The number of different six-digit integers that can be formed using the digits 2, 2, 2, 5, 5, and 9 -/
def six_digit_integers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of different six-digit integers
    formed using the digits 2, 2, 2, 5, 5, and 9 is equal to 60 -/
theorem six_digit_integers_count : six_digit_integers = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_integers_count_l3560_356016


namespace NUMINAMATH_CALUDE_seven_eighths_of_64_l3560_356094

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_64_l3560_356094


namespace NUMINAMATH_CALUDE_ellipse_equation_from_hyperbola_l3560_356065

/-- Given a hyperbola with equation 3x^2 - y^2 = 3, prove that an ellipse with the same foci
    and reciprocal eccentricity has the equation x^2/16 + y^2/12 = 1 -/
theorem ellipse_equation_from_hyperbola (x y : ℝ) :
  (3 * x^2 - y^2 = 3) →
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧
    a^2 - b^2 = 4 ∧ 2 / a = 1 / 2) →
  x^2 / 16 + y^2 / 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_hyperbola_l3560_356065


namespace NUMINAMATH_CALUDE_workshop_attendees_count_l3560_356035

/-- Calculates the total number of people at a workshop given the number of novelists and the ratio of novelists to poets -/
def total_workshop_attendees (num_novelists : ℕ) (novelist_ratio : ℕ) (poet_ratio : ℕ) : ℕ :=
  num_novelists + (num_novelists * poet_ratio) / novelist_ratio

/-- Theorem stating that for a workshop with 15 novelists and a 5:3 ratio of novelists to poets, there are 24 people in total -/
theorem workshop_attendees_count :
  total_workshop_attendees 15 5 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_workshop_attendees_count_l3560_356035


namespace NUMINAMATH_CALUDE_one_third_of_four_equals_two_l3560_356098

-- Define the country's multiplication operation
noncomputable def country_mul (a b : ℚ) : ℚ := sorry

-- Define the property that 1/8 of 4 equals 3 in this system
axiom country_property : country_mul (1/8) 4 = 3

-- Theorem statement
theorem one_third_of_four_equals_two : 
  country_mul (1/3) 4 = 2 := by sorry

end NUMINAMATH_CALUDE_one_third_of_four_equals_two_l3560_356098


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3560_356049

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : isArithmeticSequence a)
  (h_sum : a 6 + a 9 = 16)
  (h_a4 : a 4 = 1) :
  a 11 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3560_356049


namespace NUMINAMATH_CALUDE_rectangular_shape_perimeter_and_area_l3560_356040

/-- A rectangular shape composed of 5 cm segments -/
structure RectangularShape where
  segmentLength : ℝ
  length : ℝ
  height : ℝ

/-- Calculate the perimeter of the rectangular shape -/
def perimeter (shape : RectangularShape) : ℝ :=
  2 * (shape.length + shape.height)

/-- Calculate the area of the rectangular shape -/
def area (shape : RectangularShape) : ℝ :=
  shape.length * shape.height

theorem rectangular_shape_perimeter_and_area 
  (shape : RectangularShape)
  (h1 : shape.segmentLength = 5)
  (h2 : shape.length = 45)
  (h3 : shape.height = 30) :
  perimeter shape = 200 ∧ area shape = 725 := by
  sorry

#check rectangular_shape_perimeter_and_area

end NUMINAMATH_CALUDE_rectangular_shape_perimeter_and_area_l3560_356040


namespace NUMINAMATH_CALUDE_simple_interest_duration_l3560_356053

/-- Simple interest calculation -/
theorem simple_interest_duration (P R SI : ℝ) (h1 : P = 10000) (h2 : R = 9) (h3 : SI = 900) :
  (SI * 100) / (P * R) * 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_duration_l3560_356053


namespace NUMINAMATH_CALUDE_theater_seat_interpretation_l3560_356044

/-- Represents a theater seat as an ordered pair of natural numbers -/
structure TheaterSeat :=
  (row : ℕ)
  (seat : ℕ)

/-- Interprets a TheaterSeat as a description -/
def interpret (s : TheaterSeat) : String :=
  s!"seat {s.seat} in row {s.row}"

theorem theater_seat_interpretation :
  interpret ⟨6, 2⟩ = "seat 2 in row 6" := by
  sorry

end NUMINAMATH_CALUDE_theater_seat_interpretation_l3560_356044


namespace NUMINAMATH_CALUDE_grid_state_theorem_l3560_356093

/-- Represents the number of times a 2x2 square was picked -/
structure SquarePicks where
  topLeft : ℕ
  topRight : ℕ
  bottomLeft : ℕ
  bottomRight : ℕ

/-- Represents the state of the 3x3 grid -/
def GridState (p : SquarePicks) : Matrix (Fin 3) (Fin 3) ℕ :=
  fun i j =>
    match i, j with
    | 0, 0 => p.topLeft
    | 0, 2 => p.topRight
    | 2, 0 => p.bottomLeft
    | 2, 2 => p.bottomRight
    | 0, 1 => p.topLeft + p.topRight
    | 1, 0 => p.topLeft + p.bottomLeft
    | 1, 2 => p.topRight + p.bottomRight
    | 2, 1 => p.bottomLeft + p.bottomRight
    | 1, 1 => p.topLeft + p.topRight + p.bottomLeft + p.bottomRight

theorem grid_state_theorem (p : SquarePicks) :
  (GridState p 2 0 = 13) →
  (GridState p 0 1 = 18) →
  (GridState p 1 1 = 47) →
  (GridState p 2 2 = 16) := by
    sorry

end NUMINAMATH_CALUDE_grid_state_theorem_l3560_356093


namespace NUMINAMATH_CALUDE_family_composition_l3560_356080

/-- Represents a family with boys and girls -/
structure Family where
  boys : Nat
  girls : Nat

/-- A boy in the family has equal number of brothers and sisters -/
def equal_siblings (f : Family) : Prop :=
  f.boys - 1 = f.girls

/-- A girl in the family has twice as many brothers as sisters -/
def double_brothers (f : Family) : Prop :=
  f.boys = 2 * (f.girls - 1)

/-- The family satisfies both conditions and has 4 boys and 3 girls -/
theorem family_composition :
  ∃ (f : Family), equal_siblings f ∧ double_brothers f ∧ f.boys = 4 ∧ f.girls = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_family_composition_l3560_356080


namespace NUMINAMATH_CALUDE_line_length_difference_l3560_356006

theorem line_length_difference : 
  let white_line : ℝ := 7.67
  let blue_line : ℝ := 3.33
  white_line - blue_line = 4.34 := by
sorry

end NUMINAMATH_CALUDE_line_length_difference_l3560_356006


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l3560_356046

-- Define the reduction factors
def first_reduction : ℝ := 0.85  -- 1 - 0.15
def second_reduction : ℝ := 0.90 -- 1 - 0.10

-- Theorem statement
theorem price_reduction_theorem :
  first_reduction * second_reduction * 100 = 76.5 := by
  sorry

#eval first_reduction * second_reduction * 100

end NUMINAMATH_CALUDE_price_reduction_theorem_l3560_356046


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3560_356013

theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃! x, q * x^2 - 8 * x + 2 = 0) ↔ q = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3560_356013


namespace NUMINAMATH_CALUDE_words_with_b_count_l3560_356039

/-- The number of letters in the alphabet we're using -/
def alphabet_size : ℕ := 5

/-- The length of the words we're forming -/
def word_length : ℕ := 4

/-- The number of letters in the alphabet excluding B -/
def alphabet_size_without_b : ℕ := 4

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words without B -/
def words_without_b : ℕ := alphabet_size_without_b ^ word_length

/-- The number of words with at least one B -/
def words_with_b : ℕ := total_words - words_without_b

theorem words_with_b_count : words_with_b = 369 := by
  sorry

end NUMINAMATH_CALUDE_words_with_b_count_l3560_356039


namespace NUMINAMATH_CALUDE_train_speed_problem_l3560_356082

/-- Proves that given the conditions of the train problem, the speeds of the regular and high-speed trains are 100 km/h and 250 km/h respectively. -/
theorem train_speed_problem (regular_speed : ℝ) (bullet_speed : ℝ) (high_speed : ℝ) (express_speed : ℝ)
  (h1 : bullet_speed = 2 * regular_speed)
  (h2 : high_speed = bullet_speed * 1.25)
  (h3 : (high_speed + regular_speed) / 2 = express_speed + 15)
  (h4 : (bullet_speed + regular_speed) / 2 = express_speed - 10) :
  regular_speed = 100 ∧ high_speed = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3560_356082


namespace NUMINAMATH_CALUDE_dance_to_electropop_ratio_l3560_356055

def total_requests : ℕ := 30
def electropop_requests : ℕ := total_requests / 2
def rock_requests : ℕ := 5
def oldies_requests : ℕ := rock_requests - 3
def dj_choice_requests : ℕ := oldies_requests / 2
def rap_requests : ℕ := 2

def non_electropop_requests : ℕ := rock_requests + oldies_requests + dj_choice_requests + rap_requests

def dance_music_requests : ℕ := total_requests - non_electropop_requests

theorem dance_to_electropop_ratio :
  dance_music_requests = electropop_requests :=
sorry

end NUMINAMATH_CALUDE_dance_to_electropop_ratio_l3560_356055


namespace NUMINAMATH_CALUDE_batsman_average_after_12_innings_l3560_356096

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat

/-- Calculates the average score of a batsman -/
def calculateAverage (b : Batsman) : Nat :=
  b.totalRuns / b.innings

/-- Theorem: A batsman's average after 12 innings is 70 runs -/
theorem batsman_average_after_12_innings (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.totalRuns = calculateAverage b * 11 + 92)
  (h3 : b.averageIncrease = 2)
  : calculateAverage b = 70 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_12_innings_l3560_356096


namespace NUMINAMATH_CALUDE_circle_radius_from_inscribed_rectangle_l3560_356086

theorem circle_radius_from_inscribed_rectangle (r : ℝ) : 
  (∃ (s : ℝ), s^2 = 72 ∧ s^2 = 2 * r^2) → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_inscribed_rectangle_l3560_356086


namespace NUMINAMATH_CALUDE_pencil_count_l3560_356097

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of these two numbers. -/
theorem pencil_count (initial : ℕ) (added : ℕ) :
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l3560_356097


namespace NUMINAMATH_CALUDE_total_peanuts_l3560_356084

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := 10

/-- The number of peanuts Mary adds to the box -/
def added_peanuts : ℕ := 8

/-- Theorem stating the total number of peanuts in the box -/
theorem total_peanuts : initial_peanuts + added_peanuts = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_peanuts_l3560_356084


namespace NUMINAMATH_CALUDE_matthew_crackers_l3560_356052

def crackers_problem (initial_crackers : ℕ) (friends : ℕ) (crackers_per_friend : ℕ) : Prop :=
  initial_crackers - (friends * crackers_per_friend) = 3

theorem matthew_crackers : crackers_problem 24 3 7 := by
  sorry

end NUMINAMATH_CALUDE_matthew_crackers_l3560_356052


namespace NUMINAMATH_CALUDE_olympic_medals_l3560_356009

theorem olympic_medals (total gold silver bronze : ℕ) : 
  total = 89 → 
  gold + silver = 4 * bronze - 6 → 
  gold + silver + bronze = total → 
  bronze = 19 := by
sorry

end NUMINAMATH_CALUDE_olympic_medals_l3560_356009


namespace NUMINAMATH_CALUDE_player_pay_is_23000_l3560_356078

/-- Represents the player's performance in a single game -/
structure GamePerformance :=
  (points : ℕ)
  (assists : ℕ)
  (rebounds : ℕ)
  (steals : ℕ)

/-- Calculates the base pay based on average points per game -/
def basePay (games : List GamePerformance) : ℕ :=
  if (games.map GamePerformance.points).sum / games.length ≥ 30 then 10000 else 8000

/-- Calculates the assists bonus based on total assists -/
def assistsBonus (games : List GamePerformance) : ℕ :=
  let totalAssists := (games.map GamePerformance.assists).sum
  if totalAssists ≥ 20 then 5000
  else if totalAssists ≥ 10 then 3000
  else 1000

/-- Calculates the rebounds bonus based on total rebounds -/
def reboundsBonus (games : List GamePerformance) : ℕ :=
  let totalRebounds := (games.map GamePerformance.rebounds).sum
  if totalRebounds ≥ 40 then 5000
  else if totalRebounds ≥ 20 then 3000
  else 1000

/-- Calculates the steals bonus based on total steals -/
def stealsBonus (games : List GamePerformance) : ℕ :=
  let totalSteals := (games.map GamePerformance.steals).sum
  if totalSteals ≥ 15 then 5000
  else if totalSteals ≥ 5 then 3000
  else 1000

/-- Calculates the total pay for the week -/
def totalPay (games : List GamePerformance) : ℕ :=
  basePay games + assistsBonus games + reboundsBonus games + stealsBonus games

/-- Theorem: Given the player's performance, the total pay for the week is $23,000 -/
theorem player_pay_is_23000 (games : List GamePerformance) 
  (h1 : games = [
    ⟨30, 5, 7, 3⟩, 
    ⟨28, 6, 5, 2⟩, 
    ⟨32, 4, 9, 1⟩, 
    ⟨34, 3, 11, 2⟩, 
    ⟨26, 2, 8, 3⟩
  ]) : 
  totalPay games = 23000 := by
  sorry


end NUMINAMATH_CALUDE_player_pay_is_23000_l3560_356078


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3560_356076

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (((x - y) * x^4 < 0 → x < y) ∧
   ∃ a b : ℝ, a < b ∧ (a - b) * a^4 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3560_356076


namespace NUMINAMATH_CALUDE_bac_is_105_l3560_356037

/-- Represents the encoding of a base-5 digit --/
inductive Encoding
  | A
  | B
  | C
  | D
  | E

/-- Converts an Encoding to its corresponding base-5 digit --/
def encoding_to_digit (e : Encoding) : Nat :=
  match e with
  | Encoding.A => 1
  | Encoding.B => 4
  | Encoding.C => 0
  | Encoding.D => 3
  | Encoding.E => 4

/-- Converts a sequence of Encodings to its base-10 representation --/
def encodings_to_base10 (encodings : List Encoding) : Nat :=
  encodings.enum.foldl (fun acc (i, e) => acc + encoding_to_digit e * (5 ^ (encodings.length - 1 - i))) 0

/-- The main theorem stating that BAC in the given encoding system represents 105 in base 10 --/
theorem bac_is_105 (h1 : encodings_to_base10 [Encoding.A, Encoding.B, Encoding.E] + 1 = encodings_to_base10 [Encoding.A, Encoding.B, Encoding.D])
                   (h2 : encodings_to_base10 [Encoding.A, Encoding.B, Encoding.D] + 1 = encodings_to_base10 [Encoding.A, Encoding.A, Encoding.C]) :
  encodings_to_base10 [Encoding.B, Encoding.A, Encoding.C] = 105 := by
  sorry

end NUMINAMATH_CALUDE_bac_is_105_l3560_356037


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3560_356089

theorem fraction_sum_equality : 
  (4 : ℚ) / 3 + 13 / 9 + 40 / 27 + 121 / 81 - 8 / 3 = 171 / 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3560_356089


namespace NUMINAMATH_CALUDE_earth_hour_seating_l3560_356008

theorem earth_hour_seating (x : ℕ) : 30 * x + 8 = 31 * x - 26 := by
  sorry

end NUMINAMATH_CALUDE_earth_hour_seating_l3560_356008


namespace NUMINAMATH_CALUDE_laura_weekly_mileage_l3560_356095

/-- Represents the total miles driven by Laura in a week -/
def total_miles_per_week (
  house_school_round_trip : ℕ)
  (supermarket_extra_distance : ℕ)
  (gym_distance : ℕ)
  (friend_distance : ℕ)
  (workplace_distance : ℕ)
  (school_days : ℕ)
  (supermarket_trips : ℕ)
  (gym_trips : ℕ)
  (friend_trips : ℕ) : ℕ :=
  -- Weekday trips (work and school)
  (workplace_distance + (house_school_round_trip / 2 - workplace_distance) + (house_school_round_trip / 2)) * school_days +
  -- Supermarket trips
  ((house_school_round_trip / 2 + supermarket_extra_distance) * 2) * supermarket_trips +
  -- Gym trips
  (gym_distance * 2) * gym_trips +
  -- Friend's house trips
  (friend_distance * 2) * friend_trips

/-- Theorem stating that Laura drives 234 miles per week -/
theorem laura_weekly_mileage :
  total_miles_per_week 20 10 5 12 8 5 2 3 1 = 234 := by
  sorry

end NUMINAMATH_CALUDE_laura_weekly_mileage_l3560_356095


namespace NUMINAMATH_CALUDE_count_with_six_seven_l3560_356007

/-- The number of integers from 1 to 512 in base 8 that don't use digits 6 or 7 -/
def count_without_six_seven : ℕ := 215

/-- The total number of integers we're considering -/
def total_count : ℕ := 512

theorem count_with_six_seven :
  total_count - count_without_six_seven = 297 := by
  sorry

end NUMINAMATH_CALUDE_count_with_six_seven_l3560_356007


namespace NUMINAMATH_CALUDE_sams_seashells_l3560_356067

theorem sams_seashells (mary_seashells : ℕ) (total_seashells : ℕ) 
  (h1 : mary_seashells = 47)
  (h2 : total_seashells = 65) :
  total_seashells - mary_seashells = 18 := by
  sorry

end NUMINAMATH_CALUDE_sams_seashells_l3560_356067


namespace NUMINAMATH_CALUDE_problem_solution_l3560_356062

def A (a : ℝ) : Set ℝ := {x | 2 * a - 3 < x ∧ x < a + 1}

def B : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem problem_solution :
  (∀ x : ℝ, x ∈ A 0 ∩ B ↔ -1 < x ∧ x < 1) ∧
  (∀ a : ℝ, A a ∩ (Set.univ \ B) = A a ↔ a ≤ -2 ∨ a ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3560_356062


namespace NUMINAMATH_CALUDE_pony_discount_rate_l3560_356083

-- Define the regular prices
def fox_price : ℝ := 15
def pony_price : ℝ := 18

-- Define the number of jeans purchased
def fox_quantity : ℕ := 3
def pony_quantity : ℕ := 2

-- Define the total savings
def total_savings : ℝ := 9

-- Define the sum of discount rates
def total_discount_rate : ℝ := 22

-- Theorem statement
theorem pony_discount_rate :
  ∃ (fox_discount pony_discount : ℝ),
    fox_discount + pony_discount = total_discount_rate ∧
    fox_quantity * fox_price * (fox_discount / 100) +
    pony_quantity * pony_price * (pony_discount / 100) = total_savings ∧
    pony_discount = 10 := by
  sorry

end NUMINAMATH_CALUDE_pony_discount_rate_l3560_356083


namespace NUMINAMATH_CALUDE_greatest_NPM_value_l3560_356010

theorem greatest_NPM_value : ∀ M N P : ℕ,
  (M ≥ 1 ∧ M ≤ 9) →  -- M is a one-digit integer
  (N ≥ 1 ∧ N ≤ 9) →  -- N is a one-digit integer
  (P ≥ 0 ∧ P ≤ 9) →  -- P is a one-digit integer
  (10 * M + M) * M = 100 * N + 10 * P + M →  -- MM * M = NPM
  100 * N + 10 * P + M ≤ 396 :=
by sorry

end NUMINAMATH_CALUDE_greatest_NPM_value_l3560_356010


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_volume_l3560_356012

theorem rectangular_prism_surface_area_volume (x : ℝ) (h : x > 0) :
  let a := Real.log x
  let b := Real.exp (Real.log x)
  let c := x
  let surface_area := 2 * (a * b + b * c + c * a)
  let volume := a * b * c
  surface_area = 3 * volume → x = Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_volume_l3560_356012


namespace NUMINAMATH_CALUDE_cos_product_pi_ninths_l3560_356056

theorem cos_product_pi_ninths : 
  Real.cos (π / 9) * Real.cos (2 * π / 9) * Real.cos (4 * π / 9) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_pi_ninths_l3560_356056


namespace NUMINAMATH_CALUDE_tablet_savings_l3560_356054

/-- Proves that buying a tablet in cash saves $70 compared to an installment plan -/
theorem tablet_savings : 
  let cash_price : ℕ := 450
  let down_payment : ℕ := 100
  let first_four_months : ℕ := 4 * 40
  let next_four_months : ℕ := 4 * 35
  let last_four_months : ℕ := 4 * 30
  let total_installment : ℕ := down_payment + first_four_months + next_four_months + last_four_months
  total_installment - cash_price = 70 := by
  sorry

end NUMINAMATH_CALUDE_tablet_savings_l3560_356054


namespace NUMINAMATH_CALUDE_mikes_bills_l3560_356066

theorem mikes_bills (total_amount : ℕ) (bill_denomination : ℕ) (h1 : total_amount = 45) (h2 : bill_denomination = 5) :
  total_amount / bill_denomination = 9 := by
  sorry

end NUMINAMATH_CALUDE_mikes_bills_l3560_356066
