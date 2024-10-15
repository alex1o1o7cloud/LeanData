import Mathlib

namespace NUMINAMATH_CALUDE_farmhand_work_hours_l681_68118

/-- Represents the number of golden delicious apples needed for one pint of cider -/
def golden_delicious_per_pint : ℕ := 20

/-- Represents the number of pink lady apples needed for one pint of cider -/
def pink_lady_per_pint : ℕ := 40

/-- Represents the number of farmhands -/
def num_farmhands : ℕ := 6

/-- Represents the number of apples each farmhand can pick per hour -/
def apples_per_hour_per_farmhand : ℕ := 240

/-- Represents the ratio of golden delicious to pink lady apples -/
def apple_ratio : Rat := 1 / 2

/-- Represents the number of pints of cider Haley can make with the gathered apples -/
def pints_of_cider : ℕ := 120

/-- Theorem stating that the farmhands will work for 5 hours -/
theorem farmhand_work_hours : 
  ∃ (hours : ℕ), 
    hours = 5 ∧ 
    hours * (num_farmhands * apples_per_hour_per_farmhand) = 
      pints_of_cider * (golden_delicious_per_pint + pink_lady_per_pint) ∧
    apple_ratio = (pints_of_cider * golden_delicious_per_pint : ℚ) / 
                  (pints_of_cider * pink_lady_per_pint : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_farmhand_work_hours_l681_68118


namespace NUMINAMATH_CALUDE_functional_inequality_solution_l681_68150

open Real

-- Define the function type
def ContinuousRealFunction := {f : ℝ → ℝ // Continuous f}

-- State the theorem
theorem functional_inequality_solution 
  (f : ContinuousRealFunction) 
  (h1 : f.val 0 = 0) 
  (h2 : ∀ x y : ℝ, f.val ((x + y) / (1 + x * y)) ≥ f.val x + f.val y) :
  ∃ c : ℝ, ∀ x : ℝ, f.val x = (c / 2) * log (abs ((x + 1) / (x - 1))) :=
sorry

end NUMINAMATH_CALUDE_functional_inequality_solution_l681_68150


namespace NUMINAMATH_CALUDE_range_of_a_l681_68155

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (A ∪ B a = B a) → a < -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l681_68155


namespace NUMINAMATH_CALUDE_remainder_problem_l681_68183

theorem remainder_problem (j : ℕ+) (h : 75 % (j^2 : ℕ) = 3) : 
  (130 % (j : ℕ) = 0) ∨ (130 % (j : ℕ) = 1) := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l681_68183


namespace NUMINAMATH_CALUDE_bobby_shoes_count_l681_68164

/-- Given information about the number of shoes owned by Bonny, Becky, and Bobby, 
    prove that Bobby has 27 pairs of shoes. -/
theorem bobby_shoes_count : 
  ∀ (becky_shoes : ℕ), 
  (13 = 2 * becky_shoes - 5) →  -- Bonny's shoes are 5 less than twice Becky's
  (27 = 3 * becky_shoes) -- Bobby has 3 times as many shoes as Becky
  := by sorry

end NUMINAMATH_CALUDE_bobby_shoes_count_l681_68164


namespace NUMINAMATH_CALUDE_smallest_norm_w_l681_68102

/-- Given a vector w such that ‖w + (4, 2)‖ = 10, 
    the smallest possible value of ‖w‖ is 10 - 2√5 -/
theorem smallest_norm_w (w : ℝ × ℝ) 
    (h : ‖w + (4, 2)‖ = 10) : 
    ∃ (w_min : ℝ × ℝ), ‖w_min‖ = 10 - 2 * Real.sqrt 5 ∧ 
    ∀ (v : ℝ × ℝ), ‖v + (4, 2)‖ = 10 → ‖w_min‖ ≤ ‖v‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_w_l681_68102


namespace NUMINAMATH_CALUDE_problem_solving_percentage_l681_68171

theorem problem_solving_percentage (total : ℕ) (multiple_choice : ℕ) : 
  total = 50 → multiple_choice = 10 → 
  (((total - multiple_choice) : ℚ) / total) * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_percentage_l681_68171


namespace NUMINAMATH_CALUDE_non_similar_1500_pointed_stars_l681_68119

/-- The number of non-similar regular n-pointed stars -/
def num_non_similar_stars (n : ℕ) : ℕ := 
  (Nat.totient n - 2) / 2

/-- Properties of regular n-pointed stars -/
axiom regular_star_properties (n : ℕ) : 
  ∃ (prop : ℕ → Prop), prop n ∧ prop 1000

theorem non_similar_1500_pointed_stars : 
  num_non_similar_stars 1500 = 199 := by
  sorry

end NUMINAMATH_CALUDE_non_similar_1500_pointed_stars_l681_68119


namespace NUMINAMATH_CALUDE_baseball_season_games_l681_68125

/-- Calculates the total number of games played in a baseball season given the number of wins and a relationship between wins and losses. -/
theorem baseball_season_games (wins losses : ℕ) : 
  wins = 101 ∧ wins = 3 * losses + 14 → wins + losses = 130 :=
by sorry

end NUMINAMATH_CALUDE_baseball_season_games_l681_68125


namespace NUMINAMATH_CALUDE_evolute_of_ellipse_l681_68149

/-- The equation of the evolute of an ellipse -/
theorem evolute_of_ellipse (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^2 / a^2 + y^2 / b^2 = 1 →
  (a * x)^(2/3) + (b * y)^(2/3) = (a^2 - b^2)^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_evolute_of_ellipse_l681_68149


namespace NUMINAMATH_CALUDE_age_sum_five_years_ago_l681_68193

theorem age_sum_five_years_ago (djibo_age : ℕ) (sister_age : ℕ) : 
  djibo_age = 17 → sister_age = 28 → djibo_age - 5 + (sister_age - 5) = 35 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_five_years_ago_l681_68193


namespace NUMINAMATH_CALUDE_f_properties_l681_68137

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) - Real.cos (2 * x) + 1) / (2 * Real.sin x)

theorem f_properties :
  (∃ (S : Set ℝ), S = {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi} ∧ (∀ x : ℝ, x ∈ S ↔ f x ≠ 0)) ∧
  (Set.range f = Set.Icc (-Real.sqrt 2) (-1) ∪ Set.Ioo (-1) 1 ∪ Set.Icc 1 (Real.sqrt 2)) ∧
  (∀ α : ℝ, 0 < α ∧ α < Real.pi / 2 → Real.tan (α / 2) = 1 / 2 → f α = 7 / 5) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l681_68137


namespace NUMINAMATH_CALUDE_binomial_congruence_characterization_l681_68146

theorem binomial_congruence_characterization (n : ℕ) (hn : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i → i ≤ j → j ≤ n →
    (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  ∃ p : ℕ, p > 0 ∧ n = 2^p - 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_congruence_characterization_l681_68146


namespace NUMINAMATH_CALUDE_download_time_proof_l681_68184

def internet_speed : ℝ := 2
def file1_size : ℝ := 80
def file2_size : ℝ := 90
def file3_size : ℝ := 70
def minutes_per_hour : ℝ := 60

theorem download_time_proof :
  let total_size := file1_size + file2_size + file3_size
  let download_time_minutes := total_size / internet_speed
  let download_time_hours := download_time_minutes / minutes_per_hour
  download_time_hours = 2 := by
sorry

end NUMINAMATH_CALUDE_download_time_proof_l681_68184


namespace NUMINAMATH_CALUDE_ellas_food_calculation_l681_68188

/-- The amount of food Ella eats each day, in pounds -/
def ellas_daily_food : ℝ := 20

/-- The number of days considered -/
def days : ℕ := 10

/-- The total amount of food Ella and her dog eat in the given number of days, in pounds -/
def total_food : ℝ := 1000

/-- The ratio of food Ella's dog eats compared to Ella -/
def dog_food_ratio : ℝ := 4

theorem ellas_food_calculation :
  ellas_daily_food * (1 + dog_food_ratio) * days = total_food :=
by sorry

end NUMINAMATH_CALUDE_ellas_food_calculation_l681_68188


namespace NUMINAMATH_CALUDE_plywood_length_l681_68197

/-- The length of a rectangular piece of plywood with given area and width -/
theorem plywood_length (area width : ℝ) (h1 : area = 24) (h2 : width = 6) :
  area / width = 4 := by
  sorry

end NUMINAMATH_CALUDE_plywood_length_l681_68197


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_range_l681_68104

theorem function_inequality_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, (1/2) * x^4 - 2 * x^3 + 3 * m + 6 ≥ 0) → m ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_range_l681_68104


namespace NUMINAMATH_CALUDE_opposite_quadratics_solution_l681_68115

theorem opposite_quadratics_solution (x : ℚ) : 
  (2 * x^2 + 1 = -(4 * x^2 - 2 * x - 5)) → (x = 1 ∨ x = -2/3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_quadratics_solution_l681_68115


namespace NUMINAMATH_CALUDE_nearest_city_distance_l681_68147

theorem nearest_city_distance (d : ℝ) : 
  (¬ (d ≥ 13)) ∧ (¬ (d ≤ 10)) ∧ (¬ (d ≤ 8)) → d ∈ Set.Ioo 10 13 :=
by sorry

end NUMINAMATH_CALUDE_nearest_city_distance_l681_68147


namespace NUMINAMATH_CALUDE_craig_apple_count_l681_68130

/-- The number of apples Craig has initially -/
def craig_initial_apples : ℝ := 20.0

/-- The number of apples Craig receives from Eugene -/
def apples_from_eugene : ℝ := 7.0

/-- The total number of apples Craig will have -/
def craig_total_apples : ℝ := craig_initial_apples + apples_from_eugene

theorem craig_apple_count : craig_total_apples = 27.0 := by
  sorry

end NUMINAMATH_CALUDE_craig_apple_count_l681_68130


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l681_68180

theorem largest_integer_with_remainder : 
  ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l681_68180


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l681_68176

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing on ℝ
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem increasing_function_inequality (h_incr : IsIncreasing f) (m : ℝ) :
  f (2 * m) > f (-m + 9) → m > 3 := by
  sorry


end NUMINAMATH_CALUDE_increasing_function_inequality_l681_68176


namespace NUMINAMATH_CALUDE_square_division_exists_l681_68181

theorem square_division_exists : ∃ (n : ℕ) (a b c : ℝ), 
  n > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ c^2 = n * (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_square_division_exists_l681_68181


namespace NUMINAMATH_CALUDE_smallest_with_twelve_odd_eighteen_even_divisors_l681_68151

def count_odd_divisors (n : ℕ) : ℕ := 
  (Finset.filter (λ d => d % 2 = 1) (Nat.divisors n)).card

def count_even_divisors (n : ℕ) : ℕ := 
  (Finset.filter (λ d => d % 2 = 0) (Nat.divisors n)).card

theorem smallest_with_twelve_odd_eighteen_even_divisors :
  ∀ n : ℕ, n > 0 → 
    (count_odd_divisors n = 12 ∧ count_even_divisors n = 18) → 
    n ≥ 900 :=
sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_odd_eighteen_even_divisors_l681_68151


namespace NUMINAMATH_CALUDE_square_equals_eight_times_reciprocal_l681_68170

theorem square_equals_eight_times_reciprocal (x : ℝ) : 
  x > 0 → x^2 = 8 * (1/x) → x = 2 := by sorry

end NUMINAMATH_CALUDE_square_equals_eight_times_reciprocal_l681_68170


namespace NUMINAMATH_CALUDE_simplify_tan_product_l681_68124

theorem simplify_tan_product (tan30 tan15 : ℝ) : 
  tan30 + tan15 = 1 - tan30 * tan15 → (1 + tan30) * (1 + tan15) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_tan_product_l681_68124


namespace NUMINAMATH_CALUDE_final_digit_is_nine_l681_68163

/-- Represents the sequence of digits formed by concatenating numbers from 1 to 1995 -/
def initial_sequence : List Nat := sorry

/-- Removes digits at even positions from a list of digits -/
def remove_even_positions (digits : List Nat) : List Nat := sorry

/-- Removes digits at odd positions from a list of digits -/
def remove_odd_positions (digits : List Nat) : List Nat := sorry

/-- Applies the alternating removal process until one digit remains -/
def process_sequence (digits : List Nat) : Nat := sorry

theorem final_digit_is_nine : 
  process_sequence initial_sequence = 9 := by sorry

end NUMINAMATH_CALUDE_final_digit_is_nine_l681_68163


namespace NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l681_68154

theorem gcd_8_factorial_6_factorial_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l681_68154


namespace NUMINAMATH_CALUDE_modulus_of_z_l681_68143

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 3 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l681_68143


namespace NUMINAMATH_CALUDE_prime_sum_square_fourth_power_l681_68156

theorem prime_sum_square_fourth_power : 
  ∀ p q r : ℕ, 
    Prime p → Prime q → Prime r → 
    p + q^2 = r^4 → 
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_square_fourth_power_l681_68156


namespace NUMINAMATH_CALUDE_one_acrobat_l681_68111

/-- Represents the count of animals at the zoo -/
structure ZooCount where
  acrobats : ℕ
  elephants : ℕ
  monkeys : ℕ

/-- Checks if the given ZooCount satisfies the conditions of the problem -/
def isValidCount (count : ZooCount) : Prop :=
  2 * count.acrobats + 4 * count.elephants + 2 * count.monkeys = 134 ∧
  count.acrobats + count.elephants + count.monkeys = 45

/-- Theorem stating that there is exactly one acrobat in the valid zoo count -/
theorem one_acrobat :
  ∃! (count : ZooCount), isValidCount count ∧ count.acrobats = 1 := by
  sorry

#check one_acrobat

end NUMINAMATH_CALUDE_one_acrobat_l681_68111


namespace NUMINAMATH_CALUDE_eighth_grade_students_l681_68174

/-- The number of students in eighth grade -/
theorem eighth_grade_students :
  let girls : ℕ := 28
  let boys : ℕ := 2 * girls - 16
  let total : ℕ := boys + girls
  total = 68 := by sorry

end NUMINAMATH_CALUDE_eighth_grade_students_l681_68174


namespace NUMINAMATH_CALUDE_power_of_512_l681_68186

theorem power_of_512 : (512 : ℝ) ^ (4/3) = 4096 := by sorry

end NUMINAMATH_CALUDE_power_of_512_l681_68186


namespace NUMINAMATH_CALUDE_same_solution_k_value_l681_68178

theorem same_solution_k_value (x k : ℝ) : 
  (2 * x + 4 = 4 * (x - 2) ∧ -x + k = 2 * x - 1) ↔ k = 17 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_k_value_l681_68178


namespace NUMINAMATH_CALUDE_sequence_product_l681_68139

theorem sequence_product (a : ℕ → ℝ) (h1 : ∀ n, a (n - 1) = 2 * a n) (h2 : a 5 = 4) :
  a 4 * a 5 * a 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l681_68139


namespace NUMINAMATH_CALUDE_sum_of_parts_l681_68108

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 54) (h2 : y = 34) : 10 * x + 22 * y = 948 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_l681_68108


namespace NUMINAMATH_CALUDE_swimming_frequency_l681_68162

def runs_every : ℕ := 4
def cycles_every : ℕ := 16
def all_activities_every : ℕ := 48

theorem swimming_frequency :
  ∃ (swims_every : ℕ),
    swims_every > 0 ∧
    (Nat.lcm swims_every runs_every = Nat.lcm (Nat.lcm swims_every runs_every) cycles_every) ∧
    Nat.lcm (Nat.lcm swims_every runs_every) cycles_every = all_activities_every ∧
    swims_every = 3 := by
  sorry

end NUMINAMATH_CALUDE_swimming_frequency_l681_68162


namespace NUMINAMATH_CALUDE_basketball_score_ratio_l681_68159

/-- Represents the points scored in basketball games -/
structure BasketballScores where
  first_away : ℕ
  second_away : ℕ
  third_away : ℕ
  last_home : ℕ
  next_game : ℕ

/-- Theorem stating the ratio of last home game points to first away game points -/
theorem basketball_score_ratio (scores : BasketballScores) : 
  scores.last_home = 62 →
  scores.second_away = scores.first_away + 18 →
  scores.third_away = scores.second_away + 2 →
  scores.next_game = 55 →
  scores.first_away + scores.second_away + scores.third_away + scores.last_home + scores.next_game = 4 * scores.last_home →
  (scores.last_home : ℚ) / scores.first_away = 2 := by
  sorry


end NUMINAMATH_CALUDE_basketball_score_ratio_l681_68159


namespace NUMINAMATH_CALUDE_min_value_of_expression_limit_at_one_l681_68166

open Real

theorem min_value_of_expression (x : ℝ) (h1 : -3 < x) (h2 : x < 2) (h3 : x ≠ 1) :
  (x^2 - 4*x + 5) / (3*x - 3) ≥ 2/3 :=
sorry

theorem limit_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |(x^2 - 4*x + 5) / (3*x - 3) - 2/3| < ε :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_limit_at_one_l681_68166


namespace NUMINAMATH_CALUDE_resort_tips_multiple_l681_68144

theorem resort_tips_multiple (total_months : ℕ) (august_ratio : ℝ) : 
  total_months = 7 → 
  august_ratio = 0.25 → 
  (7 * august_ratio) / (1 - august_ratio) = 1.75 := by
sorry

end NUMINAMATH_CALUDE_resort_tips_multiple_l681_68144


namespace NUMINAMATH_CALUDE_average_age_increase_l681_68129

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 23 →
  student_avg_age = 22 →
  teacher_age = 46 →
  ((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1) - student_avg_age = 1 :=
by sorry

end NUMINAMATH_CALUDE_average_age_increase_l681_68129


namespace NUMINAMATH_CALUDE_unique_integer_fraction_l681_68122

theorem unique_integer_fraction (m n : ℕ) (h1 : m ≥ 3) (h2 : n ≥ 3) :
  (∃ S : Set ℕ, (Set.Infinite S ∧
    ∀ a ∈ S, ∃ k : ℤ, (a^m + a - 1 : ℤ) = k * (a^n + a^2 - 1)))
  ↔ m = 5 ∧ n = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_fraction_l681_68122


namespace NUMINAMATH_CALUDE_negation_of_divisible_by_5_is_odd_l681_68126

theorem negation_of_divisible_by_5_is_odd :
  (¬ ∀ n : ℤ, n % 5 = 0 → Odd n) ↔ (∃ n : ℤ, n % 5 = 0 ∧ ¬ Odd n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_divisible_by_5_is_odd_l681_68126


namespace NUMINAMATH_CALUDE_connie_watch_savings_l681_68172

/-- The amount of money Connie needs to buy a watch -/
theorem connie_watch_savings (saved : ℕ) (watch_cost : ℕ) (h1 : saved = 39) (h2 : watch_cost = 55) :
  watch_cost - saved = 16 := by
  sorry

end NUMINAMATH_CALUDE_connie_watch_savings_l681_68172


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l681_68190

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the solution set
def solution_set : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l681_68190


namespace NUMINAMATH_CALUDE_min_value_on_circle_l681_68157

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 20 = 0

/-- A point on the circle -/
structure PointOnCircle where
  a : ℝ
  b : ℝ
  on_circle : circle_equation a b

/-- The theorem stating the minimum value of a^2 + b^2 for points on the circle -/
theorem min_value_on_circle :
  ∀ P : PointOnCircle, ∃ m : ℝ, 
    (∀ Q : PointOnCircle, m ≤ Q.a^2 + Q.b^2) ∧
    m = 30 - 10 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l681_68157


namespace NUMINAMATH_CALUDE_unique_solution_l681_68185

theorem unique_solution : ∃! x : ℝ, 
  -1 < x ∧ x ≤ 2 ∧ 
  Real.sqrt (2 - x) + Real.sqrt (2 + 2*x) = Real.sqrt ((x^4 + 1)/(x^2 + 1)) + (x + 3)/(x + 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l681_68185


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l681_68192

theorem sqrt_sum_squares_eq_sum (a b : ℝ) :
  Real.sqrt (a^2 + b^2) = a + b ↔ a * b = 0 ∧ a + b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l681_68192


namespace NUMINAMATH_CALUDE_patio_perimeter_l681_68128

/-- A rectangular patio with length 40 feet and width equal to one-fourth of its length has a perimeter of 100 feet. -/
theorem patio_perimeter : 
  ∀ (length width : ℝ), 
  length = 40 → 
  width = length / 4 → 
  2 * length + 2 * width = 100 := by
sorry

end NUMINAMATH_CALUDE_patio_perimeter_l681_68128


namespace NUMINAMATH_CALUDE_athletes_seating_arrangements_l681_68179

def number_of_arrangements (team_sizes : List Nat) : Nat :=
  (team_sizes.length.factorial) * (team_sizes.map Nat.factorial).prod

theorem athletes_seating_arrangements :
  number_of_arrangements [4, 3, 3] = 5184 := by
  sorry

end NUMINAMATH_CALUDE_athletes_seating_arrangements_l681_68179


namespace NUMINAMATH_CALUDE_special_sequence_max_length_l681_68198

/-- A finite sequence of real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∀ i, i + 2 < n → a i + a (i + 1) + a (i + 2) < 0) ∧
  (∀ i, i + 3 < n → a i + a (i + 1) + a (i + 2) + a (i + 3) > 0)

/-- The maximum length of a SpecialSequence is 5 -/
theorem special_sequence_max_length :
  ∀ n : ℕ, ∀ a : ℕ → ℝ, SpecialSequence a n → n ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_special_sequence_max_length_l681_68198


namespace NUMINAMATH_CALUDE_general_solution_zero_a_case_degenerate_case_l681_68132

-- Define the system of equations
def system (a b c x y z : ℝ) : Prop :=
  a * x + b * y - c * z = a * b ∧
  3 * a * x - b * y + 2 * c * z = a * (5 * c - b) ∧
  3 * y + 2 * z = 5 * a

-- Theorem for the general solution
theorem general_solution (a b c : ℝ) :
  ∃ x y z, system a b c x y z ∧ x = c ∧ y = a ∧ z = a :=
sorry

-- Theorem for the case when a = 0
theorem zero_a_case (b c : ℝ) :
  ∃ x y z, system 0 b c x y z ∧ y = 0 ∧ z = 0 :=
sorry

-- Theorem for the case when 8b + 15c = 0
theorem degenerate_case (a b : ℝ) :
  8 * b + 15 * (-8 * b / 15) = 0 →
  ∃ x y, ∀ z, system a b (-8 * b / 15) x y z :=
sorry

end NUMINAMATH_CALUDE_general_solution_zero_a_case_degenerate_case_l681_68132


namespace NUMINAMATH_CALUDE_trigonometric_identity_l681_68168

theorem trigonometric_identity (α : ℝ) : 
  3 + 4 * Real.sin (4 * α + 3 / 2 * Real.pi) + Real.sin (8 * α + 5 / 2 * Real.pi) = 8 * (Real.sin (2 * α))^4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l681_68168


namespace NUMINAMATH_CALUDE_age_multiple_l681_68110

def rons_current_age : ℕ := 43
def maurices_current_age : ℕ := 7
def years_passed : ℕ := 5

theorem age_multiple : 
  (rons_current_age + years_passed) / (maurices_current_age + years_passed) = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_multiple_l681_68110


namespace NUMINAMATH_CALUDE_tent_count_solution_l681_68120

def total_value : ℕ := 940000
def total_tents : ℕ := 600
def cost_A : ℕ := 1700
def cost_B : ℕ := 1300

theorem tent_count_solution :
  ∃ (x y : ℕ),
    x + y = total_tents ∧
    cost_A * x + cost_B * y = total_value ∧
    x = 400 ∧
    y = 200 := by
  sorry

end NUMINAMATH_CALUDE_tent_count_solution_l681_68120


namespace NUMINAMATH_CALUDE_local_min_implies_a_equals_one_l681_68195

/-- Given a function f(x) = ax^3 - 2x^2 + a^2x, where a is a real number,
    if f has a local minimum at x = 1, then a = 1. -/
theorem local_min_implies_a_equals_one (a : ℝ) :
  let f := λ x : ℝ => a * x^3 - 2 * x^2 + a^2 * x
  (∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_local_min_implies_a_equals_one_l681_68195


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l681_68105

/-- An arithmetic sequence with its sum function and common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  d : ℝ       -- Common difference
  sum_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2
  seq_def : ∀ n, a n = a 1 + (n - 1) * d

/-- Theorem: If 2S_3 = 3S_2 + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l681_68105


namespace NUMINAMATH_CALUDE_class_size_proof_l681_68189

/-- The number of students in a class with English and German courses -/
def class_size (english_only german_only both : ℕ) : ℕ :=
  english_only + german_only + both

theorem class_size_proof (english_only german_only both : ℕ) 
  (h1 : both = 12)
  (h2 : german_only + both = 22)
  (h3 : english_only = 30) :
  class_size english_only german_only both = 52 := by
  sorry

#check class_size_proof

end NUMINAMATH_CALUDE_class_size_proof_l681_68189


namespace NUMINAMATH_CALUDE_wage_decrease_increase_l681_68127

theorem wage_decrease_increase (original : ℝ) (h : original > 0) :
  let decreased := original * 0.5
  let increased := decreased * 1.5
  increased = original * 0.75 :=
by sorry

end NUMINAMATH_CALUDE_wage_decrease_increase_l681_68127


namespace NUMINAMATH_CALUDE_percentage_of_filled_seats_l681_68158

/-- Given a hall with 600 seats and 240 vacant seats, prove that 60% of the seats were filled. -/
theorem percentage_of_filled_seats (total_seats : ℕ) (vacant_seats : ℕ) : 
  total_seats = 600 → vacant_seats = 240 → 
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 = 60) := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_filled_seats_l681_68158


namespace NUMINAMATH_CALUDE_function_through_point_l681_68133

theorem function_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (fun x : ℝ ↦ a^x) (-1) = 2 → (fun x : ℝ ↦ a^x) = (fun x : ℝ ↦ (1/2)^x) := by
  sorry

end NUMINAMATH_CALUDE_function_through_point_l681_68133


namespace NUMINAMATH_CALUDE_calculate_X_l681_68100

theorem calculate_X : ∀ M N X : ℚ,
  M = 3009 / 3 →
  N = M / 4 →
  X = M + 2 * N →
  X = 1504.5 := by
sorry

end NUMINAMATH_CALUDE_calculate_X_l681_68100


namespace NUMINAMATH_CALUDE_partition_cases_num_partitions_formula_l681_68160

/-- The number of partitions of a set with n+1 elements into n subsets -/
def num_partitions (n : ℕ) : ℕ := (2^n - 1)^(n+1)

/-- Theorem stating the number of partitions for specific cases -/
theorem partition_cases :
  (num_partitions 2 = 3^3) ∧
  (num_partitions 3 = 7^4) ∧
  (num_partitions 4 = 15^5) := by sorry

/-- Main theorem: The number of partitions of a set with n+1 elements into n subsets is (2^n - 1)^(n+1) -/
theorem num_partitions_formula (n : ℕ) :
  num_partitions n = (2^n - 1)^(n+1) := by sorry

end NUMINAMATH_CALUDE_partition_cases_num_partitions_formula_l681_68160


namespace NUMINAMATH_CALUDE_final_class_size_l681_68140

theorem final_class_size (initial_size second_year_join final_year_leave : ℕ) :
  initial_size = 150 →
  second_year_join = 30 →
  final_year_leave = 15 →
  initial_size + second_year_join - final_year_leave = 165 := by
  sorry

end NUMINAMATH_CALUDE_final_class_size_l681_68140


namespace NUMINAMATH_CALUDE_remaining_tickets_l681_68106

def tickets_from_whack_a_mole : ℕ := 32
def tickets_from_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

theorem remaining_tickets :
  tickets_from_whack_a_mole + tickets_from_skee_ball - tickets_spent_on_hat = 50 := by
  sorry

end NUMINAMATH_CALUDE_remaining_tickets_l681_68106


namespace NUMINAMATH_CALUDE_constructible_heights_count_l681_68194

/-- A function that returns the number of constructible heights given a number of bricks and possible height increments. -/
def countConstructibleHeights (numBricks : ℕ) (heightIncrements : List ℕ) : ℕ :=
  sorry

/-- The theorem stating that with 25 bricks and height increments of 0, 3, and 4, there are 98 constructible heights. -/
theorem constructible_heights_count : 
  countConstructibleHeights 25 [0, 3, 4] = 98 :=
sorry

end NUMINAMATH_CALUDE_constructible_heights_count_l681_68194


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l681_68152

/-- The line y - 1 = k(x - 1) is tangent to the circle x^2 + y^2 - 2y = 0 for any real k -/
theorem line_tangent_to_circle (k : ℝ) : 
  ∃! (x y : ℝ), (y - 1 = k * (x - 1)) ∧ (x^2 + y^2 - 2*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l681_68152


namespace NUMINAMATH_CALUDE_particle_probability_l681_68134

def probability (x y : ℕ) : ℚ :=
  sorry

theorem particle_probability :
  let start_x : ℕ := 5
  let start_y : ℕ := 5
  probability start_x start_y = 1 / 243 :=
by
  sorry

axiom probability_recursive (x y : ℕ) :
  x > 0 → y > 0 →
  probability x y = (1/3) * probability (x-1) y + 
                    (1/3) * probability x (y-1) + 
                    (1/3) * probability (x-1) (y-1)

axiom probability_boundary_zero (x y : ℕ) :
  (x = 0 ∧ y > 0) ∨ (x > 0 ∧ y = 0) →
  probability x y = 0

axiom probability_origin :
  probability 0 0 = 1

end NUMINAMATH_CALUDE_particle_probability_l681_68134


namespace NUMINAMATH_CALUDE_mean_of_set_l681_68117

theorem mean_of_set (m : ℝ) : 
  (m + 8 = 16) → 
  (m + (m + 6) + (m + 8) + (m + 14) + (m + 21)) / 5 = 89 / 5 := by
sorry

end NUMINAMATH_CALUDE_mean_of_set_l681_68117


namespace NUMINAMATH_CALUDE_function_max_at_zero_implies_a_geq_three_l681_68196

/-- Given a function f(x) = x + a / (x + 1) defined on [0, 2] with maximum at x = 0, prove a ≥ 3 -/
theorem function_max_at_zero_implies_a_geq_three (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 2 → x + a / (x + 1) ≤ a) →
  a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_max_at_zero_implies_a_geq_three_l681_68196


namespace NUMINAMATH_CALUDE_essay_word_ratio_l681_68145

def johnny_words : ℕ := 150
def timothy_words (madeline_words : ℕ) : ℕ := madeline_words + 30
def total_pages : ℕ := 3
def words_per_page : ℕ := 260

theorem essay_word_ratio (madeline_words : ℕ) :
  (johnny_words + madeline_words + timothy_words madeline_words = total_pages * words_per_page) →
  (madeline_words : ℚ) / johnny_words = 2 := by
  sorry

end NUMINAMATH_CALUDE_essay_word_ratio_l681_68145


namespace NUMINAMATH_CALUDE_area_of_triangle_from_centers_area_is_sqrt_three_l681_68109

/-- The area of an equilateral triangle formed by connecting the centers of three equilateral
    triangles of side length 2, arranged around a vertex of a square. -/
theorem area_of_triangle_from_centers : ℝ :=
  let side_length : ℝ := 2
  let triangle_centers_distance : ℝ := side_length
  let area_formula (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2
  area_formula triangle_centers_distance

/-- The area of the triangle formed by connecting the centers is √3. -/
theorem area_is_sqrt_three : area_of_triangle_from_centers = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_from_centers_area_is_sqrt_three_l681_68109


namespace NUMINAMATH_CALUDE_student_count_third_row_l681_68175

/-- The number of students in the first row -/
def students_first_row : ℕ := 12

/-- The number of students in the second row -/
def students_second_row : ℕ := 12

/-- The change in average age (in weeks) for the first row after rearrangement -/
def change_first_row : ℤ := 1

/-- The change in average age (in weeks) for the second row after rearrangement -/
def change_second_row : ℤ := 2

/-- The change in average age (in weeks) for the third row after rearrangement -/
def change_third_row : ℤ := -4

/-- The number of students in the third row -/
def students_third_row : ℕ := 9

theorem student_count_third_row : 
  students_first_row * change_first_row + 
  students_second_row * change_second_row + 
  students_third_row * change_third_row = 0 :=
by sorry

end NUMINAMATH_CALUDE_student_count_third_row_l681_68175


namespace NUMINAMATH_CALUDE_geometric_sequence_tangent_l681_68113

open Real

theorem geometric_sequence_tangent (x : ℝ) : 
  (∃ (r : ℝ), (tan (π/12 - x) = tan (π/12) * r ∧ tan (π/12) = tan (π/12 + x) * r) ∨
               (tan (π/12 - x) = tan (π/12 + x) * r ∧ tan (π/12) = tan (π/12 - x) * r) ∨
               (tan (π/12) = tan (π/12 - x) * r ∧ tan (π/12 + x) = tan (π/12) * r)) ↔ 
  (∃ (ε : ℤ) (n : ℤ), ε ∈ ({-1, 0, 1} : Set ℤ) ∧ x = ε * (π/3) + n * π) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_tangent_l681_68113


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l681_68135

theorem least_n_satisfying_inequality : ∀ n : ℕ, n > 0 → 
  ((1 : ℚ) / n - (1 : ℚ) / (n + 2) < (1 : ℚ) / 15) ↔ n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l681_68135


namespace NUMINAMATH_CALUDE_even_function_sum_l681_68123

def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 2) a, f a b x = f a b ((a - 2) + a - x)) →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_even_function_sum_l681_68123


namespace NUMINAMATH_CALUDE_distance_sum_on_corresponding_segments_l681_68136

/-- Given two line segments AB and A'B' with lengths 6 and 16 respectively,
    and a linear correspondence between points on these segments,
    prove that the sum of distances from A to P and A' to P' is 18/5 * a,
    where a is the distance from A to P. -/
theorem distance_sum_on_corresponding_segments
  (AB : Real) (A'B' : Real)
  (a : Real)
  (h1 : AB = 6)
  (h2 : A'B' = 16)
  (h3 : 0 ≤ a ∧ a ≤ AB)
  (correspondence : Real → Real)
  (h4 : correspondence 1 = 3)
  (h5 : ∀ x, 0 ≤ x ∧ x ≤ AB → 0 ≤ correspondence x ∧ correspondence x ≤ A'B')
  (h6 : ∀ x y, (0 ≤ x ∧ x ≤ AB ∧ 0 ≤ y ∧ y ≤ AB) →
              (correspondence x - correspondence y) / (x - y) = (correspondence 1 - 0) / (1 - 0)) :
  a + correspondence a = 18/5 * a := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_on_corresponding_segments_l681_68136


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_min_value_achievable_l681_68191

theorem min_value_quadratic_form (x y : ℝ) : 
  3 * x^2 + 2 * x * y + 3 * y^2 + 5 ≥ 5 :=
by sorry

theorem min_value_achievable : 
  ∃ (x y : ℝ), 3 * x^2 + 2 * x * y + 3 * y^2 + 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_min_value_achievable_l681_68191


namespace NUMINAMATH_CALUDE_apple_distribution_l681_68101

theorem apple_distribution (total_apples : ℕ) (ratio_1_2 ratio_1_3 ratio_2_3 : ℚ) :
  total_apples = 169 →
  ratio_1_2 = 1 / 2 →
  ratio_1_3 = 1 / 3 →
  ratio_2_3 = 1 / 2 →
  ∃ (boy1 boy2 boy3 : ℕ),
    boy1 + boy2 + boy3 = total_apples ∧
    boy1 = 78 ∧
    boy2 = 52 ∧
    boy3 = 39 ∧
    (boy1 : ℚ) / (boy2 : ℚ) = ratio_1_2 ∧
    (boy1 : ℚ) / (boy3 : ℚ) = ratio_1_3 ∧
    (boy2 : ℚ) / (boy3 : ℚ) = ratio_2_3 :=
by
  sorry

#check apple_distribution

end NUMINAMATH_CALUDE_apple_distribution_l681_68101


namespace NUMINAMATH_CALUDE_base_for_888_l681_68173

theorem base_for_888 :
  ∃! b : ℕ,
    (b > 1) ∧
    (∃ a B : ℕ,
      a ≠ B ∧
      a < b ∧
      B < b ∧
      888 = a * b^3 + a * b^2 + B * b + B) ∧
    (b^3 ≤ 888) ∧
    (888 < b^4) :=
by sorry

end NUMINAMATH_CALUDE_base_for_888_l681_68173


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l681_68142

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x - 2 < 0) ↔ a ∈ Set.Ioc (-2) 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l681_68142


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l681_68148

/-- A function satisfying the given inequality -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + y ≤ f (f (f x))

/-- The main theorem stating the form of functions satisfying the inequality -/
theorem characterize_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfiesInequality f →
  ∃ C : ℝ, ∀ x : ℝ, f x = -x + C :=
by sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l681_68148


namespace NUMINAMATH_CALUDE_not_p_and_not_q_is_false_l681_68107

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.tan x = 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 > 0

-- Theorem statement
theorem not_p_and_not_q_is_false : ¬(¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_is_false_l681_68107


namespace NUMINAMATH_CALUDE_golden_ratio_approximation_l681_68112

theorem golden_ratio_approximation :
  (∃ (S : Set ℚ), Set.Infinite S ∧
    ∀ r ∈ S, ∃ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 ∧ r = q / p ∧
      |r - (Real.sqrt 5 - 1) / 2| < 1 / p^2) ∧
  (∀ p q : ℤ, p > 0 → Int.gcd p q = 1 →
    |(q : ℝ) / p - (Real.sqrt 5 - 1) / 2| > 1 / (Real.sqrt 5 + 1) / p^2) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_approximation_l681_68112


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l681_68114

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ (i, bit) acc => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec to_binary_aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
    (to_binary_aux n).reverse

theorem binary_arithmetic_equality :
  let a := binary_to_decimal [true, false, true, true]  -- 1101₂
  let b := binary_to_decimal [false, true, true]        -- 110₂
  let c := binary_to_decimal [false, true, true, true]  -- 1110₂
  let d := binary_to_decimal [true, true, true, true]   -- 1111₂
  let result := decimal_to_binary (a + b - c + d)
  result = [false, true, false, false, false, true]     -- 100010₂
:= by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l681_68114


namespace NUMINAMATH_CALUDE_root_difference_cubic_equation_l681_68138

theorem root_difference_cubic_equation :
  ∃ (α β γ : ℝ),
    (81 * α^3 - 162 * α^2 + 90 * α - 10 = 0) ∧
    (81 * β^3 - 162 * β^2 + 90 * β - 10 = 0) ∧
    (81 * γ^3 - 162 * γ^2 + 90 * γ - 10 = 0) ∧
    (β = 2 * α ∨ γ = 2 * α ∨ γ = 2 * β) ∧
    (max α (max β γ) - min α (min β γ) = 1) :=
sorry

end NUMINAMATH_CALUDE_root_difference_cubic_equation_l681_68138


namespace NUMINAMATH_CALUDE_megans_books_l681_68121

theorem megans_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (h1 : books_per_shelf = 7)
  (h2 : mystery_shelves = 8)
  (h3 : picture_shelves = 2) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 70 :=
by sorry

end NUMINAMATH_CALUDE_megans_books_l681_68121


namespace NUMINAMATH_CALUDE_expression_evaluation_l681_68199

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := -3
  3 * (x^2 - 2*x*y) - (3*x^2 - 2*y + 2*(x*y + y)) = -12 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l681_68199


namespace NUMINAMATH_CALUDE_blue_balls_count_l681_68153

theorem blue_balls_count (total : ℕ) (green blue yellow white : ℕ) : 
  green = total / 4 →
  blue = total / 8 →
  yellow = total / 12 →
  white = 26 →
  total = green + blue + yellow + white →
  blue = 6 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l681_68153


namespace NUMINAMATH_CALUDE_parabola_symmetry_problem_l681_68103

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = 2 * x^2

/-- The problem statement -/
theorem parabola_symmetry_problem (A B : ParabolaPoint) (m : ℝ) 
  (h_symmetric : ∃ (t : ℝ), (A.x + B.x) / 2 = t ∧ (A.y + B.y) / 2 = t + m)
  (h_product : A.x * B.x = -1/2) :
  m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_problem_l681_68103


namespace NUMINAMATH_CALUDE_toy_cost_price_l681_68131

/-- Given the sale of toys, prove the cost price of a single toy. -/
theorem toy_cost_price (num_sold : ℕ) (total_price : ℕ) (gain_equiv : ℕ) (cost_price : ℕ) :
  num_sold = 36 →
  total_price = 45000 →
  gain_equiv = 6 →
  total_price = num_sold * cost_price + gain_equiv * cost_price →
  cost_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_price_l681_68131


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l681_68165

-- Define the function f(x) = -x
def f (x : ℝ) : ℝ := -x

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x ≤ y → f y ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l681_68165


namespace NUMINAMATH_CALUDE_ounces_per_pound_l681_68182

def cat_food_bags : ℕ := 2
def cat_food_weight : ℕ := 3
def dog_food_bags : ℕ := 2
def dog_food_extra_weight : ℕ := 2
def total_ounces : ℕ := 256

theorem ounces_per_pound :
  ∃ (x : ℕ),
    x * (cat_food_bags * cat_food_weight + 
         dog_food_bags * (cat_food_weight + dog_food_extra_weight)) = total_ounces ∧
    x = 16 := by
  sorry

end NUMINAMATH_CALUDE_ounces_per_pound_l681_68182


namespace NUMINAMATH_CALUDE_biology_score_calculation_l681_68161

def math_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 67
def average_score : ℕ := 69
def total_subjects : ℕ := 5

theorem biology_score_calculation :
  let known_subjects_total := math_score + science_score + social_studies_score + english_score
  let all_subjects_total := average_score * total_subjects
  all_subjects_total - known_subjects_total = 55 := by
sorry

end NUMINAMATH_CALUDE_biology_score_calculation_l681_68161


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l681_68169

/-- Given two parallel vectors a and b, prove that the magnitude of 3a + 2b is √5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  b.1 = -2 → 
  ∃ y, b.2 = y → 
  (∃ k : ℝ, k ≠ 0 ∧ a = k • b) → 
  ‖3 • a + 2 • b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l681_68169


namespace NUMINAMATH_CALUDE_shortest_path_length_l681_68116

/-- The shortest path length from (0,0) to (12,16) avoiding a circle -/
theorem shortest_path_length (start end_ circle_center : ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  let path_length := 10 * Real.sqrt 3 + 5 * Real.pi / 3
  by
    sorry

#check shortest_path_length (0, 0) (12, 16) (6, 8) 5

end NUMINAMATH_CALUDE_shortest_path_length_l681_68116


namespace NUMINAMATH_CALUDE_six_ways_to_make_50_yuan_l681_68167

/-- The number of ways to make 50 yuan using 5 yuan and 10 yuan notes -/
def ways_to_make_50_yuan : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 5 * p.1 + 10 * p.2 = 50) (Finset.product (Finset.range 11) (Finset.range 6))).card

/-- Theorem stating that there are exactly 6 ways to make 50 yuan using 5 yuan and 10 yuan notes -/
theorem six_ways_to_make_50_yuan : ways_to_make_50_yuan = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_ways_to_make_50_yuan_l681_68167


namespace NUMINAMATH_CALUDE_mixture_weight_l681_68177

/-- Given substances a and b mixed in a ratio of 9:11, prove that the total weight
    of the mixture is 58 kg when 26.1 kg of substance a is used. -/
theorem mixture_weight (a b : ℝ) (h1 : a / b = 9 / 11) (h2 : a = 26.1) :
  a + b = 58 := by
  sorry

end NUMINAMATH_CALUDE_mixture_weight_l681_68177


namespace NUMINAMATH_CALUDE_weeks_to_cover_all_combinations_l681_68141

/-- Represents a lottery ticket grid -/
structure LotteryGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (row_constraint : rows ≥ 5)
  (col_constraint : cols ≥ 14)

/-- Represents the marking strategy -/
structure MarkingStrategy :=
  (square_size : ℕ)
  (extra_number : ℕ)
  (square_constraint : square_size = 2)
  (extra_constraint : extra_number = 1)

/-- Represents the weekly ticket filling strategy -/
def weekly_tickets : ℕ := 4

/-- Theorem stating the time required to cover all combinations -/
theorem weeks_to_cover_all_combinations 
  (grid : LotteryGrid) 
  (strategy : MarkingStrategy) : 
  (((grid.rows - 2) * (grid.cols - 2)) + weekly_tickets - 1) / weekly_tickets = 52 :=
sorry

end NUMINAMATH_CALUDE_weeks_to_cover_all_combinations_l681_68141


namespace NUMINAMATH_CALUDE_exchange_problem_l681_68187

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Represents the exchange scenario -/
def exchangeScenario (d : ℕ) : Prop :=
  (8 : ℚ) / 5 * d - 80 = d

theorem exchange_problem :
  ∃ d : ℕ, exchangeScenario d ∧ sumOfDigits d = 9 := by sorry

end NUMINAMATH_CALUDE_exchange_problem_l681_68187
