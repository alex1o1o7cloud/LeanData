import Mathlib

namespace NUMINAMATH_CALUDE_officers_selection_count_l331_33103

/-- The number of ways to select 3 distinct individuals from a group of 6 people to fill 3 distinct positions -/
def selectOfficers (n : ℕ) : ℕ :=
  if n < 3 then 0
  else n * (n - 1) * (n - 2)

/-- Theorem stating that selecting 3 officers from 6 people results in 120 possibilities -/
theorem officers_selection_count : selectOfficers 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_officers_selection_count_l331_33103


namespace NUMINAMATH_CALUDE_no_integer_solution_l331_33169

theorem no_integer_solution : ¬∃ (x y : ℤ), (x + 2020) * (x + 2021) + (x + 2021) * (x + 2022) + (x + 2020) * (x + 2022) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l331_33169


namespace NUMINAMATH_CALUDE_equation_solutions_l331_33119

theorem equation_solutions :
  (∀ x, (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4) ∧
  (∀ x, x * (x - 6) = 6 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l331_33119


namespace NUMINAMATH_CALUDE_quadratic_factorization_l331_33179

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 15 * y^2 - 74 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l331_33179


namespace NUMINAMATH_CALUDE_expression_bounds_l331_33186

theorem expression_bounds (a b c d x : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ x) (hc : 0 ≤ c ∧ c ≤ x) (hd : 0 ≤ d ∧ d ≤ x)
  (hx : 0 < x ∧ x ≤ 10) : 
  2 * x * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (x - b)^2) + Real.sqrt (b^2 + (x - c)^2) + 
    Real.sqrt (c^2 + (x - d)^2) + Real.sqrt (d^2 + (x - a)^2) ∧
  Real.sqrt (a^2 + (x - b)^2) + Real.sqrt (b^2 + (x - c)^2) + 
    Real.sqrt (c^2 + (x - d)^2) + Real.sqrt (d^2 + (x - a)^2) ≤ 4 * x ∧
  ∃ (a' b' c' d' : ℝ), 
    0 ≤ a' ∧ a' ≤ x ∧ 0 ≤ b' ∧ b' ≤ x ∧ 0 ≤ c' ∧ c' ≤ x ∧ 0 ≤ d' ∧ d' ≤ x ∧
    Real.sqrt (a'^2 + (x - b')^2) + Real.sqrt (b'^2 + (x - c')^2) + 
    Real.sqrt (c'^2 + (x - d')^2) + Real.sqrt (d'^2 + (x - a')^2) = 2 * x * Real.sqrt 2 ∧
  ∃ (a'' b'' c'' d'' : ℝ), 
    0 ≤ a'' ∧ a'' ≤ x ∧ 0 ≤ b'' ∧ b'' ≤ x ∧ 0 ≤ c'' ∧ c'' ≤ x ∧ 0 ≤ d'' ∧ d'' ≤ x ∧
    Real.sqrt (a''^2 + (x - b''^2)) + Real.sqrt (b''^2 + (x - c''^2)) + 
    Real.sqrt (c''^2 + (x - d''^2)) + Real.sqrt (d''^2 + (x - a''^2)) = 4 * x :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l331_33186


namespace NUMINAMATH_CALUDE_similar_cube_volume_l331_33165

theorem similar_cube_volume (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 27 → scale_factor = 2 → 
  (scale_factor^3 * original_volume : ℝ) = 216 := by
  sorry

end NUMINAMATH_CALUDE_similar_cube_volume_l331_33165


namespace NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l331_33174

-- Define the geometric sequence a_n
def a (n : ℕ) : ℝ := 2^n

-- Define the arithmetic sequence b_n
def b (n : ℕ) : ℝ := 12 * n - 28

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := 6 * n^2 - 22 * n

theorem geometric_and_arithmetic_sequences :
  (a 1 = 2) ∧ 
  (a 4 = 16) ∧ 
  (∀ n : ℕ, a n = 2^n) ∧
  (b 3 = a 3) ∧
  (b 5 = a 5) ∧
  (∀ n : ℕ, b n = 12 * n - 28) ∧
  (∀ n : ℕ, S n = 6 * n^2 - 22 * n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l331_33174


namespace NUMINAMATH_CALUDE_cow_spots_multiple_l331_33170

/-- 
Given a cow with spots on both sides:
* The left side has 16 spots
* The total number of spots is 71
* The right side has 16x + 7 spots, where x is some multiple

Prove that x = 3
-/
theorem cow_spots_multiple (x : ℚ) : 
  16 + (16 * x + 7) = 71 → x = 3 := by sorry

end NUMINAMATH_CALUDE_cow_spots_multiple_l331_33170


namespace NUMINAMATH_CALUDE_ball_probabilities_l331_33193

theorem ball_probabilities (total_balls : ℕ) (red_prob black_prob white_prob green_prob : ℚ)
  (h_total : total_balls = 12)
  (h_red : red_prob = 5 / 12)
  (h_black : black_prob = 1 / 3)
  (h_white : white_prob = 1 / 6)
  (h_green : green_prob = 1 / 12)
  (h_sum : red_prob + black_prob + white_prob + green_prob = 1) :
  (red_prob + black_prob = 3 / 4) ∧ (red_prob + black_prob + white_prob = 11 / 12) := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l331_33193


namespace NUMINAMATH_CALUDE_cupcake_price_correct_l331_33136

/-- The original price of cupcakes before the discount -/
def original_cupcake_price : ℝ := 3

/-- The original price of cookies before the discount -/
def original_cookie_price : ℝ := 2

/-- The number of cupcakes sold -/
def cupcakes_sold : ℕ := 16

/-- The number of cookies sold -/
def cookies_sold : ℕ := 8

/-- The total revenue from the sale -/
def total_revenue : ℝ := 32

/-- Theorem stating that the original cupcake price satisfies the given conditions -/
theorem cupcake_price_correct : 
  cupcakes_sold * (original_cupcake_price / 2) + cookies_sold * (original_cookie_price / 2) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_cupcake_price_correct_l331_33136


namespace NUMINAMATH_CALUDE_mabels_daisies_l331_33156

/-- Given initial daisies, petals per daisy, and daisies given away, 
    calculate the number of petals on remaining daisies -/
def remaining_petals (initial_daisies : ℕ) (petals_per_daisy : ℕ) (daisies_given : ℕ) : ℕ :=
  (initial_daisies - daisies_given) * petals_per_daisy

/-- Theorem: Given 5 initial daisies with 8 petals each, 
    after giving away 2 daisies, 24 petals remain -/
theorem mabels_daisies : remaining_petals 5 8 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_mabels_daisies_l331_33156


namespace NUMINAMATH_CALUDE_prob_four_friends_same_group_l331_33162

/-- Represents the total number of students -/
def total_students : ℕ := 800

/-- Represents the number of lunch groups -/
def num_groups : ℕ := 4

/-- Represents the size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- Represents the probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- Represents the four friends -/
inductive Friend : Type
  | Al | Bob | Carol | Dan

/-- 
Theorem: The probability that four specific students (friends) are assigned 
to the same lunch group in a random assignment is 1/64.
-/
theorem prob_four_friends_same_group : 
  (prob_assigned_to_group ^ 3 : ℚ) = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_prob_four_friends_same_group_l331_33162


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l331_33178

/-- The proposition p -/
def p (m a : ℝ) : Prop := m^2 - 4*a*m + 3*a^2 < 0 ∧ a < 0

/-- The proposition q -/
def q (m : ℝ) : Prop := ∀ x > 0, x + 4/x ≥ 1 - m

theorem p_necessary_not_sufficient_for_q :
  (∃ a m : ℝ, q m → p m a) ∧
  (∃ a m : ℝ, p m a ∧ ¬(q m)) ∧
  (∀ a : ℝ, (∃ m : ℝ, p m a ∧ q m) ↔ a ∈ Set.Icc (-1) 0) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l331_33178


namespace NUMINAMATH_CALUDE_one_weighing_sufficient_l331_33164

/-- Represents the types of balls -/
inductive BallType
| Aluminum
| Duralumin

/-- The total number of balls -/
def totalBalls : ℕ := 2000

/-- The number of balls in each group -/
def groupSize : ℕ := 1000

/-- The weight of an aluminum ball in grams -/
def aluminumWeight : ℚ := 10

/-- The weight of a duralumin ball in grams -/
def duraluminWeight : ℚ := 9.9

/-- A function that returns the weight of a ball given its type -/
def ballWeight (t : BallType) : ℚ :=
  match t with
  | BallType.Aluminum => aluminumWeight
  | BallType.Duralumin => duraluminWeight

/-- Represents a group of balls -/
structure BallGroup where
  aluminum : ℕ
  duralumin : ℕ

/-- The total weight of a group of balls -/
def groupWeight (g : BallGroup) : ℚ :=
  g.aluminum * aluminumWeight + g.duralumin * duraluminWeight

/-- Theorem stating that it's possible to separate the balls into two groups
    with equal size but different weights using one weighing -/
theorem one_weighing_sufficient :
  ∃ (g1 g2 : BallGroup),
    g1.aluminum + g1.duralumin = groupSize ∧
    g2.aluminum + g2.duralumin = groupSize ∧
    g1.aluminum + g2.aluminum = groupSize ∧
    g1.duralumin + g2.duralumin = groupSize ∧
    groupWeight g1 ≠ groupWeight g2 :=
  sorry

end NUMINAMATH_CALUDE_one_weighing_sufficient_l331_33164


namespace NUMINAMATH_CALUDE_least_positive_angle_l331_33100

open Real

theorem least_positive_angle (θ : ℝ) : 
  (θ > 0 ∧ ∀ φ, φ > 0 ∧ (cos (10 * π / 180) = sin (40 * π / 180) + cos φ) → θ ≤ φ) →
  cos (10 * π / 180) = sin (40 * π / 180) + cos θ →
  θ = 70 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_least_positive_angle_l331_33100


namespace NUMINAMATH_CALUDE_complex_equation_solution_l331_33117

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 + 3 * Complex.I → z = 3 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l331_33117


namespace NUMINAMATH_CALUDE_sum_of_roots_l331_33152

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 12 → ∃ (x₁ x₂ : ℝ), 
  (x₁ + 3) * (x₁ - 4) = 12 ∧ 
  (x₂ + 3) * (x₂ - 4) = 12 ∧ 
  x₁ + x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l331_33152


namespace NUMINAMATH_CALUDE_largest_2010_digit_prime_squared_minus_one_div_24_l331_33132

/-- The largest prime number with 2010 digits -/
def p : ℕ := sorry

/-- p is prime -/
axiom p_prime : Nat.Prime p

/-- p has 2010 digits -/
axiom p_digits : 10^2009 ≤ p ∧ p < 10^2010

/-- p is the largest prime with 2010 digits -/
axiom p_largest : ∀ q, Nat.Prime q → 10^2009 ≤ q → q < 10^2010 → q ≤ p

theorem largest_2010_digit_prime_squared_minus_one_div_24 : 
  24 ∣ (p^2 - 1) := by sorry

end NUMINAMATH_CALUDE_largest_2010_digit_prime_squared_minus_one_div_24_l331_33132


namespace NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l331_33184

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

/-- Check if a point satisfies a line equation -/
def satisfies_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop := sorry

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

/-- The main theorem -/
theorem perpendicular_line_through_intersection :
  let l1 : ℝ → ℝ → Prop := λ x y => 2*x - y = 0
  let l2 : ℝ → ℝ → Prop := λ x y => x + y - 6 = 0
  let l3 : ℝ → ℝ → Prop := λ x y => 2*x + y - 1 = 0
  let l4 : ℝ → ℝ → Prop := λ x y => x - 2*y + 6 = 0
  let p := intersection_point l1 l2
  satisfies_line p l4 ∧ perpendicular l3 l4 := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l331_33184


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l331_33175

/-- The line equation y = -3x + 4 -/
def line_equation (x y : ℝ) : Prop := y = -3 * x + 4

/-- Point A with coordinates (2, y₁) -/
def point_A (y₁ : ℝ) : ℝ × ℝ := (2, y₁)

/-- Point B with coordinates (-1, y₂) -/
def point_B (y₂ : ℝ) : ℝ × ℝ := (-1, y₂)

theorem y1_less_than_y2 (y₁ y₂ : ℝ) 
  (hA : line_equation (point_A y₁).1 (point_A y₁).2)
  (hB : line_equation (point_B y₂).1 (point_B y₂).2) :
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l331_33175


namespace NUMINAMATH_CALUDE_july_birth_percentage_l331_33153

/-- The percentage of people born in July, given 15 out of 120 famous Americans were born in July -/
theorem july_birth_percentage :
  let total_people : ℕ := 120
  let july_births : ℕ := 15
  (july_births : ℚ) / total_people * 100 = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l331_33153


namespace NUMINAMATH_CALUDE_sum_proper_divisors_81_l331_33112

def proper_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d ∣ n)

theorem sum_proper_divisors_81 :
  (proper_divisors 81).sum id = 40 :=
sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_81_l331_33112


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l331_33182

theorem least_k_for_inequality : ∃ k : ℤ, k = 5 ∧ 
  (∀ n : ℤ, 0.0010101 * (10 : ℝ)^n > 10 → n ≥ k) ∧
  (0.0010101 * (10 : ℝ)^k > 10) := by
  sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l331_33182


namespace NUMINAMATH_CALUDE_rock_band_fuel_cost_l331_33177

theorem rock_band_fuel_cost (x : ℝ) :
  (2 * (0.5 * x + 100) + 2 * (0.75 * x + 100) = 550) →
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_rock_band_fuel_cost_l331_33177


namespace NUMINAMATH_CALUDE_circle_symmetry_max_ab_l331_33196

/-- Given a circle x^2 + y^2 - 4ax + 2by + b^2 = 0 (where a > 0 and b > 0) 
    symmetric about the line x - y - 1 = 0, the maximum value of ab is 1/8 -/
theorem circle_symmetry_max_ab (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, x^2 + y^2 - 4*a*x + 2*b*y + b^2 = 0 → 
    ∃ x' y' : ℝ, x' - y' - 1 = 0 ∧ x^2 + y^2 - 4*a*x + 2*b*y + b^2 = (x' - x)^2 + (y' - y)^2) →
  a * b ≤ 1/8 :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_max_ab_l331_33196


namespace NUMINAMATH_CALUDE_right_triangle_side_difference_l331_33141

theorem right_triangle_side_difference (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧ 
  C = π / 2 ∧ 
  a = 6 ∧ 
  B = π / 6 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  c / Real.sin C = a / Real.sin A
  → c - b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_difference_l331_33141


namespace NUMINAMATH_CALUDE_daves_initial_files_l331_33142

theorem daves_initial_files (initial_apps : ℕ) (final_apps : ℕ) (final_files : ℕ) :
  initial_apps = 24 →
  final_apps = 12 →
  final_files = 5 →
  final_apps = final_files + 7 →
  initial_apps - final_apps + final_files = 17 := by
  sorry

end NUMINAMATH_CALUDE_daves_initial_files_l331_33142


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l331_33191

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / i
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l331_33191


namespace NUMINAMATH_CALUDE_crazy_silly_school_remaining_books_l331_33149

/-- Given a book series with a total number of books and a number of books already read,
    calculate the number of books remaining to be read. -/
def books_remaining (total : ℕ) (read : ℕ) : ℕ := total - read

/-- Theorem stating that for the 'crazy silly school' series with 14 total books
    and 8 books already read, there are 6 books remaining to be read. -/
theorem crazy_silly_school_remaining_books :
  books_remaining 14 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_remaining_books_l331_33149


namespace NUMINAMATH_CALUDE_triangle_problem_l331_33124

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : a - c = (Real.sqrt 6 / 6) * b) 
  (h2 : Real.sin B = Real.sqrt 6 * Real.sin C) : 
  Real.cos A = Real.sqrt 6 / 4 ∧ 
  Real.sin (2 * A + π / 6) = (3 * Real.sqrt 5 - 1) / 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l331_33124


namespace NUMINAMATH_CALUDE_chef_apples_left_l331_33183

/-- The number of apples the chef has left after making a pie -/
def applesLeft (initialApples usedApples : ℕ) : ℕ :=
  initialApples - usedApples

theorem chef_apples_left : applesLeft 19 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_chef_apples_left_l331_33183


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l331_33192

theorem oil_leak_calculation (total_leak : ℕ) (leak_during_fix : ℕ) 
  (h1 : total_leak = 6206)
  (h2 : leak_during_fix = 3731) :
  total_leak - leak_during_fix = 2475 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l331_33192


namespace NUMINAMATH_CALUDE_holiday_rain_probability_l331_33157

/-- Probability of rain on Monday -/
def prob_rain_monday : ℝ := 0.3

/-- Probability of rain on Tuesday -/
def prob_rain_tuesday : ℝ := 0.6

/-- Probability of rain continuing to the next day -/
def prob_rain_continue : ℝ := 0.8

/-- Probability of rain on at least one day during the two-day holiday period -/
def prob_rain_at_least_one_day : ℝ :=
  1 - (1 - prob_rain_monday) * (1 - prob_rain_tuesday)

theorem holiday_rain_probability :
  prob_rain_at_least_one_day = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_holiday_rain_probability_l331_33157


namespace NUMINAMATH_CALUDE_three_heads_probability_l331_33130

-- Define a fair coin
def fair_coin_prob : ℚ := 1 / 2

-- Define the probability of three heads in a row
def three_heads_prob : ℚ := fair_coin_prob * fair_coin_prob * fair_coin_prob

-- Theorem statement
theorem three_heads_probability :
  three_heads_prob = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_three_heads_probability_l331_33130


namespace NUMINAMATH_CALUDE_rachel_hourly_wage_l331_33194

/-- Rachel's earnings as a waitress in a coffee shop -/
def rachel_earnings (people_served : ℕ) (tip_per_person : ℚ) (total_earnings : ℚ) : Prop :=
  let total_tips : ℚ := people_served * tip_per_person
  let hourly_wage_without_tips : ℚ := total_earnings - total_tips
  hourly_wage_without_tips = 12

theorem rachel_hourly_wage :
  rachel_earnings 20 (25/20) 37 := by
  sorry

end NUMINAMATH_CALUDE_rachel_hourly_wage_l331_33194


namespace NUMINAMATH_CALUDE_free_throws_count_l331_33111

/-- Represents a basketball team's scoring -/
structure BasketballScore where
  two_pointers : ℕ
  three_pointers : ℕ
  free_throws : ℕ

/-- Calculates the total score -/
def total_score (s : BasketballScore) : ℕ :=
  2 * s.two_pointers + 3 * s.three_pointers + s.free_throws

/-- Theorem: Given the conditions, the number of free throws is 13 -/
theorem free_throws_count (s : BasketballScore) :
  (2 * s.two_pointers = 3 * s.three_pointers) →
  (s.free_throws = s.two_pointers + 1) →
  (total_score s = 61) →
  s.free_throws = 13 := by
  sorry

#check free_throws_count

end NUMINAMATH_CALUDE_free_throws_count_l331_33111


namespace NUMINAMATH_CALUDE_dress_pocket_ratio_l331_33138

/-- Proves that the ratio of dresses with pockets to the total number of dresses is 1:2 --/
theorem dress_pocket_ratio :
  ∀ (total_dresses : ℕ) (dresses_with_pockets : ℕ) (total_pockets : ℕ),
    total_dresses = 24 →
    total_pockets = 32 →
    dresses_with_pockets * 2 = total_dresses * 1 →
    dresses_with_pockets * 2 + dresses_with_pockets * 3 = total_pockets * 3 →
    (dresses_with_pockets : ℚ) / total_dresses = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dress_pocket_ratio_l331_33138


namespace NUMINAMATH_CALUDE_greatest_a_value_l331_33185

theorem greatest_a_value (a : ℝ) : 
  (7 * Real.sqrt ((2 * a) ^ 2 + 1 ^ 2) - 4 * a ^ 2 - 1) / (Real.sqrt (1 + 4 * a ^ 2) + 3) = 2 →
  a ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_greatest_a_value_l331_33185


namespace NUMINAMATH_CALUDE_parabolas_coefficient_sum_zero_l331_33166

/-- Two distinct parabolas with leading coefficients p and q, where the vertex of each parabola lies on the other parabola -/
structure DistinctParabolas (p q : ℝ) : Prop where
  distinct : p ≠ q
  vertex_on_other : ∃ (a b : ℝ), a ≠ 0 ∧ b = p * a^2 ∧ 0 = q * a^2 + b

/-- The sum of leading coefficients of two distinct parabolas with vertices on each other is zero -/
theorem parabolas_coefficient_sum_zero {p q : ℝ} (h : DistinctParabolas p q) : p + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabolas_coefficient_sum_zero_l331_33166


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l331_33187

open Real

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) +
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) +
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l331_33187


namespace NUMINAMATH_CALUDE_garden_area_l331_33105

theorem garden_area (width : ℝ) (length : ℝ) :
  length = 3 * width + 30 →
  2 * (length + width) = 800 →
  width * length = 28443.75 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l331_33105


namespace NUMINAMATH_CALUDE_bus_stoppage_time_l331_33134

/-- Given a bus with speeds excluding and including stoppages, 
    calculate the number of minutes the bus stops per hour -/
theorem bus_stoppage_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 48)
  (h2 : speed_with_stops = 12) :
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 45 := by
  sorry

end NUMINAMATH_CALUDE_bus_stoppage_time_l331_33134


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l331_33199

theorem triangle_max_perimeter :
  ∀ a b : ℕ,
  b = 4 * a →
  (a + b > 16 ∧ a + 16 > b ∧ b + 16 > a) →
  a + b + 16 ≤ 41 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l331_33199


namespace NUMINAMATH_CALUDE_distance_to_point_l331_33122

/-- The distance from the origin to the point (8, -3, 6) in 3D space is √109. -/
theorem distance_to_point : Real.sqrt 109 = Real.sqrt (8^2 + (-3)^2 + 6^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_l331_33122


namespace NUMINAMATH_CALUDE_no_solution_system_l331_33143

theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 18) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l331_33143


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l331_33129

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 9 10))) = 630 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l331_33129


namespace NUMINAMATH_CALUDE_problem_solution_l331_33151

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l331_33151


namespace NUMINAMATH_CALUDE_toms_dimes_calculation_l331_33107

/-- Calculates the final number of dimes Tom has after receiving and spending some. -/
def final_dimes (initial : ℕ) (received : ℕ) (spent : ℕ) : ℕ :=
  initial + received - spent

/-- Proves that Tom's final number of dimes is correct given the initial amount, 
    the amount received, and the amount spent. -/
theorem toms_dimes_calculation (initial : ℕ) (received : ℕ) (spent : ℕ) :
  final_dimes initial received spent = initial + received - spent :=
by sorry

end NUMINAMATH_CALUDE_toms_dimes_calculation_l331_33107


namespace NUMINAMATH_CALUDE_number_of_women_in_first_group_l331_33146

/-- The number of women in the first group -/
def W : ℕ := sorry

/-- The work rate of the first group in units per hour -/
def work_rate_1 : ℚ := 75 / (8 * 5)

/-- The work rate of the second group in units per hour -/
def work_rate_2 : ℚ := 30 / (3 * 8)

/-- Theorem stating that the number of women in the first group is 5 -/
theorem number_of_women_in_first_group : W = 5 := by sorry

end NUMINAMATH_CALUDE_number_of_women_in_first_group_l331_33146


namespace NUMINAMATH_CALUDE_product_of_third_and_fourth_primes_above_20_l331_33125

def third_prime_above_20 : ℕ := 31

def fourth_prime_above_20 : ℕ := 37

theorem product_of_third_and_fourth_primes_above_20 :
  third_prime_above_20 * fourth_prime_above_20 = 1147 := by
  sorry

end NUMINAMATH_CALUDE_product_of_third_and_fourth_primes_above_20_l331_33125


namespace NUMINAMATH_CALUDE_clothes_thrown_away_l331_33128

theorem clothes_thrown_away (initial : ℕ) (donated_first : ℕ) (remaining : ℕ) :
  initial = 100 →
  donated_first = 5 →
  remaining = 65 →
  initial - remaining - donated_first - 3 * donated_first = 15 :=
by sorry

end NUMINAMATH_CALUDE_clothes_thrown_away_l331_33128


namespace NUMINAMATH_CALUDE_divide_and_add_l331_33150

theorem divide_and_add (x : ℝ) : x = 72 → (x / 6) + 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_divide_and_add_l331_33150


namespace NUMINAMATH_CALUDE_cows_husk_consumption_l331_33189

/-- The number of bags of husk eaten by a given number of cows in 45 days -/
def bags_eaten (num_cows : ℕ) : ℕ :=
  num_cows

/-- Theorem stating that 45 cows eat 45 bags of husk in 45 days -/
theorem cows_husk_consumption :
  bags_eaten 45 = 45 := by
  sorry

end NUMINAMATH_CALUDE_cows_husk_consumption_l331_33189


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l331_33110

theorem quadratic_form_sum (x : ℝ) :
  ∃ (b c : ℝ), x^2 - 18*x + 45 = (x + b)^2 + c ∧ b + c = -45 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l331_33110


namespace NUMINAMATH_CALUDE_function_properties_l331_33161

def FunctionProperties (f : ℝ → ℝ) : Prop :=
  (∀ x y, f x + f y = f (x + y)) ∧ (∀ x, x > 0 → f x < 0)

theorem function_properties (f : ℝ → ℝ) (h : FunctionProperties f) :
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l331_33161


namespace NUMINAMATH_CALUDE_relationship_abc_l331_33109

theorem relationship_abc : 
  let a := Real.sin (15 * π / 180) * Real.cos (15 * π / 180)
  let b := Real.cos (π / 6) ^ 2 - Real.sin (π / 6) ^ 2
  let c := Real.tan (30 * π / 180) / (1 - Real.tan (30 * π / 180) ^ 2)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l331_33109


namespace NUMINAMATH_CALUDE_g_2016_equals_1_l331_33172

-- Define the properties of function f
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧
  (∀ x : ℝ, f (x + 5) ≥ f x + 5) ∧
  (∀ x : ℝ, f (x + 1) ≤ f x + 1)

-- Define function g
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1 - x

-- Theorem statement
theorem g_2016_equals_1 (f : ℝ → ℝ) (h : satisfies_conditions f) :
  g f 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_2016_equals_1_l331_33172


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l331_33116

/-- Theorem: Relationship between heights of two cylinders with equal volumes and different radii -/
theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l331_33116


namespace NUMINAMATH_CALUDE_quadratic_solution_square_l331_33145

theorem quadratic_solution_square (y : ℝ) (h : 3 * y^2 + 2 = 7 * y + 15) : (6 * y - 5)^2 = 269 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_square_l331_33145


namespace NUMINAMATH_CALUDE_regular_price_is_80_l331_33139

/-- The regular price of one tire -/
def regular_price : ℝ := 80

/-- The total cost for four tires -/
def total_cost : ℝ := 250

/-- Theorem: The regular price of one tire is 80 dollars -/
theorem regular_price_is_80 : regular_price = 80 :=
  by
    have h1 : total_cost = 3 * regular_price + 10 := by sorry
    have h2 : total_cost = 250 := by rfl
    sorry

#check regular_price_is_80

end NUMINAMATH_CALUDE_regular_price_is_80_l331_33139


namespace NUMINAMATH_CALUDE_volunteer_distribution_l331_33198

/-- The number of ways to distribute n distinguishable volunteers among k distinguishable places,
    such that each place has at least one volunteer. -/
def distribute_volunteers (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

/-- The problem statement -/
theorem volunteer_distribution :
  distribute_volunteers 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l331_33198


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l331_33123

/-- A parallelogram with opposite vertices (3, -4) and (13, 8) has its diagonals intersecting at (8, 2) -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (3, -4)
  let v2 : ℝ × ℝ := (13, 8)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 2) := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l331_33123


namespace NUMINAMATH_CALUDE_captain_selection_theorem_l331_33106

/-- The number of ways to choose 4 captains from a team of 12 people,
    where two of the captains must be from a subset of 4 specific players. -/
def captain_selection_ways (total_team : ℕ) (total_captains : ℕ) (specific_subset : ℕ) (required_from_subset : ℕ) : ℕ :=
  (Nat.choose specific_subset required_from_subset) * 
  (Nat.choose (total_team - specific_subset) (total_captains - required_from_subset))

/-- Theorem stating that the number of ways to choose 4 captains from a team of 12 people,
    where two of the captains must be from a subset of 4 specific players, is 168. -/
theorem captain_selection_theorem : 
  captain_selection_ways 12 4 4 2 = 168 := by
  sorry

end NUMINAMATH_CALUDE_captain_selection_theorem_l331_33106


namespace NUMINAMATH_CALUDE_luncheon_invitees_l331_33180

theorem luncheon_invitees (no_shows : ℕ) (people_per_table : ℕ) (tables_needed : ℕ) :
  no_shows = 35 →
  people_per_table = 2 →
  tables_needed = 5 →
  no_shows + (people_per_table * tables_needed) = 45 :=
by sorry

end NUMINAMATH_CALUDE_luncheon_invitees_l331_33180


namespace NUMINAMATH_CALUDE_distance_to_big_rock_big_rock_distance_l331_33190

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock (v : ℝ) (c : ℝ) (t : ℝ) : 
  v > c ∧ v > 0 ∧ c > 0 ∧ t > 0 → 
  (v + c)⁻¹ * d + (v - c)⁻¹ * d = t → 
  d = (t * v^2 - t * c^2) / (2 * v) :=
by sorry

/-- The specific case for the given problem -/
theorem big_rock_distance : 
  let v := 6 -- rower's speed in still water
  let c := 1 -- river current speed
  let t := 1 -- total time for round trip
  let d := (t * v^2 - t * c^2) / (2 * v) -- distance to Big Rock
  d = 35 / 12 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_big_rock_distance_l331_33190


namespace NUMINAMATH_CALUDE_two_different_pitchers_l331_33168

-- Define the type for pitchers
structure Pitcher :=
  (shape : ℕ)
  (color : ℕ)

-- Define the theorem
theorem two_different_pitchers 
  (pitchers : Set Pitcher) 
  (h1 : ∃ (a b : Pitcher), a ∈ pitchers ∧ b ∈ pitchers ∧ a.shape ≠ b.shape)
  (h2 : ∃ (c d : Pitcher), c ∈ pitchers ∧ d ∈ pitchers ∧ c.color ≠ d.color) :
  ∃ (x y : Pitcher), x ∈ pitchers ∧ y ∈ pitchers ∧ x.shape ≠ y.shape ∧ x.color ≠ y.color :=
sorry

end NUMINAMATH_CALUDE_two_different_pitchers_l331_33168


namespace NUMINAMATH_CALUDE_triangle_classification_l331_33133

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem triangle_classification :
  ¬(is_right_triangle 1.5 2 3) ∧
  (is_right_triangle 7 24 25) ∧
  (is_right_triangle 3 4 5) ∧
  (is_right_triangle 9 12 15) :=
by sorry

end NUMINAMATH_CALUDE_triangle_classification_l331_33133


namespace NUMINAMATH_CALUDE_min_value_and_inequality_solution_l331_33135

def f (x a : ℝ) : ℝ := |x - a| + |x - 1|

theorem min_value_and_inequality_solution 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : ∀ x, f x a ≥ 2) 
  (h3 : ∃ x, f x a = 2) :
  (a = 3) ∧ 
  (∀ x, f x a ≥ 4 ↔ x ∈ Set.Iic 0 ∪ Set.Ici 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_solution_l331_33135


namespace NUMINAMATH_CALUDE_cash_refund_per_bottle_l331_33101

/-- The number of bottles of kombucha Henry drinks per month -/
def bottles_per_month : ℕ := 15

/-- The cost of each bottle of kombucha in dollars -/
def bottle_cost : ℚ := 3

/-- The number of bottles Henry can buy with his cash refund after 1 year -/
def bottles_from_refund : ℕ := 6

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: The cash refund per bottle is $0.10 -/
theorem cash_refund_per_bottle :
  (bottles_from_refund * bottle_cost) / (bottles_per_month * months_in_year) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_cash_refund_per_bottle_l331_33101


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l331_33176

theorem max_area_rectangular_pen (fencing : ℝ) (h_fencing : fencing = 60) :
  let width := fencing / 6
  let length := 2 * width
  let area := width * length
  area = 200 := by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l331_33176


namespace NUMINAMATH_CALUDE_sum_edge_face_angles_less_than_plane_angles_sum_edge_face_angles_greater_than_half_plane_angles_if_acute_l331_33173

-- Define a trihedral angle
structure TrihedralAngle where
  -- Angles between edges and opposite faces
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Plane angles at vertex
  θ₁ : ℝ
  θ₂ : ℝ
  θ₃ : ℝ
  -- Ensure all angles are positive
  α_pos : 0 < α
  β_pos : 0 < β
  γ_pos : 0 < γ
  θ₁_pos : 0 < θ₁
  θ₂_pos : 0 < θ₂
  θ₃_pos : 0 < θ₃

-- Theorem 1: Sum of angles between edges and opposite faces is less than sum of plane angles
theorem sum_edge_face_angles_less_than_plane_angles (t : TrihedralAngle) :
  t.α + t.β + t.γ < t.θ₁ + t.θ₂ + t.θ₃ := by
  sorry

-- Theorem 2: If all plane angles are acute, sum of angles between edges and opposite faces 
-- is greater than half the sum of plane angles
theorem sum_edge_face_angles_greater_than_half_plane_angles_if_acute (t : TrihedralAngle)
  (h₁ : t.θ₁ < π/2) (h₂ : t.θ₂ < π/2) (h₃ : t.θ₃ < π/2) :
  t.α + t.β + t.γ > (t.θ₁ + t.θ₂ + t.θ₃) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_edge_face_angles_less_than_plane_angles_sum_edge_face_angles_greater_than_half_plane_angles_if_acute_l331_33173


namespace NUMINAMATH_CALUDE_brand_a_households_l331_33188

theorem brand_a_households (total : ℕ) (neither : ℕ) (both : ℕ) (ratio : ℕ) :
  total = 160 →
  neither = 80 →
  both = 5 →
  ratio = 3 →
  ∃ (only_a only_b : ℕ),
    total = neither + only_a + only_b + both ∧
    only_b = ratio * both ∧
    only_a = 60 :=
by sorry

end NUMINAMATH_CALUDE_brand_a_households_l331_33188


namespace NUMINAMATH_CALUDE_cos_135_degrees_l331_33120

theorem cos_135_degrees : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l331_33120


namespace NUMINAMATH_CALUDE_helen_raisin_cookies_l331_33160

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_chocolate : ℕ := 519

/-- The number of raisin cookies Helen baked yesterday -/
def yesterday_raisin : ℕ := 300

/-- The number of chocolate chip cookies Helen baked today -/
def today_chocolate : ℕ := 359

/-- The difference in raisin cookies baked between yesterday and today -/
def raisin_difference : ℕ := 20

/-- The number of raisin cookies Helen baked today -/
def today_raisin : ℕ := yesterday_raisin - raisin_difference

theorem helen_raisin_cookies : today_raisin = 280 := by
  sorry

end NUMINAMATH_CALUDE_helen_raisin_cookies_l331_33160


namespace NUMINAMATH_CALUDE_exists_negative_f_iff_a_less_than_three_halves_max_value_one_implies_a_values_l331_33113

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + (2*a - 1)*x - 3

-- Part 1
theorem exists_negative_f_iff_a_less_than_three_halves (a : ℝ) :
  (∃ x : ℝ, x > 1 ∧ f a x < 0) ↔ a < 3/2 :=
sorry

-- Part 2
theorem max_value_one_implies_a_values (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 3, f a x ≤ 1) ∧ (∃ x ∈ Set.Icc (-1) 3, f a x = 1) →
  a = -1/3 ∨ a = -1 :=
sorry

end NUMINAMATH_CALUDE_exists_negative_f_iff_a_less_than_three_halves_max_value_one_implies_a_values_l331_33113


namespace NUMINAMATH_CALUDE_tower_height_proof_l331_33121

def sum_of_arithmetic_series (n : ℕ) : ℕ := n * (n + 1) / 2

theorem tower_height_proof :
  let initial_blocks := 35
  let additional_blocks := 65
  let initial_height := sum_of_arithmetic_series initial_blocks
  let additional_height := sum_of_arithmetic_series additional_blocks
  initial_height + additional_height = 2775 :=
by sorry

end NUMINAMATH_CALUDE_tower_height_proof_l331_33121


namespace NUMINAMATH_CALUDE_parallelogram_area_l331_33115

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the conditions of the triangle
def triangle_conditions (t : Triangle) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.C = 6 ∧ 
  d t.B t.C = 7 ∧
  ((t.B.1 - t.A.1) * (t.C.1 - t.A.1) + (t.B.2 - t.A.2) * (t.C.2 - t.A.2)) / 
    (d t.A t.B * d t.A t.C) = 1/5

-- Theorem statement
theorem parallelogram_area (t : Triangle) (h : triangle_conditions t) :
  let O := incenter t
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B * d O t.A * Real.sin (Real.arccos (((t.B.1 - O.1) * (t.A.1 - O.1) + 
    (t.B.2 - O.2) * (t.A.2 - O.2)) / (d O t.B * d O t.A))) = 10 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l331_33115


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l331_33126

theorem quadratic_coefficient (b m : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/5 = (x + m)^2 + 1/20) → 
  b = -2 * Real.sqrt (3/20) := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l331_33126


namespace NUMINAMATH_CALUDE_waiter_tables_l331_33118

theorem waiter_tables (total_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ)
  (h1 : total_customers = 21)
  (h2 : left_customers = 12)
  (h3 : people_per_table = 3) :
  (total_customers - left_customers) / people_per_table = 3 :=
by sorry

end NUMINAMATH_CALUDE_waiter_tables_l331_33118


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l331_33155

-- Problem 1
theorem simplify_expression_1 (x : ℝ) :
  (x + 3) * (3 * x - 2) + x * (1 - 3 * x) = 8 * x - 6 := by sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) (h1 : m ≠ 2) (h2 : m ≠ -2) :
  (1 - m / (m + 2)) / ((m^2 - 4*m + 4) / (m^2 - 4)) = 2 / (m - 2) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l331_33155


namespace NUMINAMATH_CALUDE_james_run_calories_l331_33195

/-- Calculates the calories burned per minute during James' run -/
def caloriesBurnedPerMinute (bagsEaten : ℕ) (ouncesPerBag : ℕ) (caloriesPerOunce : ℕ) 
  (runDuration : ℕ) (excessCalories : ℕ) : ℕ :=
  let totalOunces := bagsEaten * ouncesPerBag
  let totalCaloriesConsumed := totalOunces * caloriesPerOunce
  let caloriesBurned := totalCaloriesConsumed - excessCalories
  caloriesBurned / runDuration

theorem james_run_calories : 
  caloriesBurnedPerMinute 3 2 150 40 420 = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_run_calories_l331_33195


namespace NUMINAMATH_CALUDE_max_stores_visited_l331_33102

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ)
  (two_store_visitors : ℕ) (three_store_visitors : ℕ) (four_store_visitors : ℕ)
  (h1 : total_stores = 15)
  (h2 : total_visits = 60)
  (h3 : total_shoppers = 30)
  (h4 : two_store_visitors = 12)
  (h5 : three_store_visitors = 6)
  (h6 : four_store_visitors = 4)
  (h7 : two_store_visitors * 2 + three_store_visitors * 3 + four_store_visitors * 4 < total_visits)
  (h8 : ∀ n : ℕ, n ≤ total_shoppers → n > 0) :
  ∃ (max_visited : ℕ), max_visited = 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visited :=
sorry

end NUMINAMATH_CALUDE_max_stores_visited_l331_33102


namespace NUMINAMATH_CALUDE_steaks_needed_l331_33154

def family_members : ℕ := 5
def pounds_per_person : ℕ := 1
def ounces_per_steak : ℕ := 20
def ounces_per_pound : ℕ := 16

def total_ounces : ℕ := family_members * pounds_per_person * ounces_per_pound

theorem steaks_needed : (total_ounces / ounces_per_steak : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_steaks_needed_l331_33154


namespace NUMINAMATH_CALUDE_soldiers_per_tower_l331_33147

/-- Proves that the number of soldiers in each tower is 2 -/
theorem soldiers_per_tower (wall_length : ℕ) (tower_interval : ℕ) (total_soldiers : ℕ)
  (h1 : wall_length = 7300)
  (h2 : tower_interval = 5)
  (h3 : total_soldiers = 2920) :
  total_soldiers / (wall_length / tower_interval) = 2 := by
  sorry

end NUMINAMATH_CALUDE_soldiers_per_tower_l331_33147


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l331_33181

theorem fourth_degree_polynomial_roots : 
  let p (x : ℂ) := x^4 - 16*x^2 + 51
  ∀ r : ℂ, r^2 = 8 + Real.sqrt 13 → p r = 0 :=
sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l331_33181


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l331_33158

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

/-- Theorem stating that (2, -1) is the pre-image of (3, 1) under f -/
theorem preimage_of_3_1 : f (2, -1) = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l331_33158


namespace NUMINAMATH_CALUDE_people_with_banners_l331_33108

/-- Given a stadium with a certain number of seats, prove that the number of people
    holding banners is equal to the number of attendees minus the number of empty seats. -/
theorem people_with_banners (total_seats attendees empty_seats : ℕ) :
  total_seats = 92 →
  attendees = 47 →
  empty_seats = 45 →
  attendees - empty_seats = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_people_with_banners_l331_33108


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l331_33167

theorem gold_coin_distribution (x y : ℕ) (h1 : x + y = 25) :
  ∃ k : ℕ, x^2 - y^2 = k * (x - y) → k = 25 := by
sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l331_33167


namespace NUMINAMATH_CALUDE_min_value_problem_l331_33144

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, a * x + b * y = 1 → x^2 + y^2 - 2*x - 2*y - 2 = 0 → x = 1 ∧ y = 1) →
  (∀ c d : ℝ, c > 0 → d > 0 → c * 1 + d * 1 = 1 → 1/c + 2/d ≥ 1/a + 2/b) →
  1/a + 2/b = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l331_33144


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l331_33114

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_intersect : ∃ (x y : ℝ), y = 2*x ∧ x^2/a^2 - y^2/b^2 = 1) :
  let e := Real.sqrt (1 + (b/a)^2)
  e > Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l331_33114


namespace NUMINAMATH_CALUDE_hundredth_figure_count_l331_33148

/-- Represents the number of nonoverlapping unit triangles in the nth figure of the pattern. -/
def triangle_count (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The first four terms of the sequence match the given pattern. -/
axiom first_four_correct : 
  triangle_count 0 = 1 ∧ 
  triangle_count 1 = 7 ∧ 
  triangle_count 2 = 19 ∧ 
  triangle_count 3 = 37

/-- The 100th figure contains 30301 nonoverlapping unit triangles. -/
theorem hundredth_figure_count : triangle_count 100 = 30301 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_figure_count_l331_33148


namespace NUMINAMATH_CALUDE_min_sum_cube_relation_l331_33159

theorem min_sum_cube_relation (m n : ℕ+) (h : 90 * m.val = n.val ^ 3) : 
  (∀ (x y : ℕ+), 90 * x.val = y.val ^ 3 → m.val + n.val ≤ x.val + y.val) → 
  m.val + n.val = 330 := by
sorry

end NUMINAMATH_CALUDE_min_sum_cube_relation_l331_33159


namespace NUMINAMATH_CALUDE_farmer_extra_days_l331_33140

/-- The number of extra days a farmer needs to work given initial and actual ploughing rates, total area, and remaining area. -/
theorem farmer_extra_days (initial_rate actual_rate total_area remaining_area : ℕ) : 
  initial_rate = 90 →
  actual_rate = 85 →
  total_area = 3780 →
  remaining_area = 40 →
  (total_area - remaining_area) % actual_rate = 0 →
  (remaining_area + actual_rate - 1) / actual_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_farmer_extra_days_l331_33140


namespace NUMINAMATH_CALUDE_prove_weekly_earnings_l331_33137

def total_earnings : ℕ := 133
def num_weeks : ℕ := 19
def weekly_earnings : ℚ := total_earnings / num_weeks

theorem prove_weekly_earnings : weekly_earnings = 7 := by
  sorry

end NUMINAMATH_CALUDE_prove_weekly_earnings_l331_33137


namespace NUMINAMATH_CALUDE_total_paths_correct_l331_33163

/-- Represents the number of paths from A to the first red arrow -/
def paths_A_to_red1 : ℕ := 1

/-- Represents the number of paths from A to the second red arrow -/
def paths_A_to_red2 : ℕ := 2

/-- Represents the number of paths from the first red arrow to each of the first two blue arrows -/
def paths_red1_to_blue : ℕ := 3

/-- Represents the number of paths from the second red arrow to each of the first two blue arrows -/
def paths_red2_to_blue : ℕ := 4

/-- Represents the number of paths from each of the first two blue arrows to each of the first two green arrows -/
def paths_blue12_to_green : ℕ := 5

/-- Represents the number of paths from each of the third and fourth blue arrows to each of the first two green arrows -/
def paths_blue34_to_green : ℕ := 6

/-- Represents the number of paths from each green arrow to point B -/
def paths_green_to_B : ℕ := 3

/-- Represents the number of paths from each green arrow to point C -/
def paths_green_to_C : ℕ := 4

/-- The total number of distinct paths from A to B and C -/
def total_paths : ℕ := 4312

theorem total_paths_correct : 
  total_paths = 
    (paths_A_to_red1 * paths_red1_to_blue * 2 + paths_A_to_red2 * paths_red2_to_blue * 2) *
    (paths_blue12_to_green * 4 + paths_blue34_to_green * 4) *
    (paths_green_to_B + paths_green_to_C) := by
  sorry

end NUMINAMATH_CALUDE_total_paths_correct_l331_33163


namespace NUMINAMATH_CALUDE_min_cost_square_base_l331_33104

/-- Represents the dimensions and cost parameters of a rectangular open-top tank. -/
structure Tank where
  volume : ℝ
  depth : ℝ
  base_cost : ℝ
  wall_cost : ℝ

/-- Calculates the total cost of constructing the tank given its length and width. -/
def total_cost (t : Tank) (length width : ℝ) : ℝ :=
  t.base_cost * length * width + t.wall_cost * 2 * t.depth * (length + width)

/-- Theorem stating that the minimum cost for the specified tank is achieved with a square base of side length 3m. -/
theorem min_cost_square_base (t : Tank) 
    (h_volume : t.volume = 18)
    (h_depth : t.depth = 2)
    (h_base_cost : t.base_cost = 200)
    (h_wall_cost : t.wall_cost = 150) :
    ∃ (cost : ℝ), cost = 5400 ∧ 
    ∀ (l w : ℝ), l * w * t.depth = t.volume → total_cost t l w ≥ cost ∧
    total_cost t 3 3 = cost :=
  sorry

#check min_cost_square_base

end NUMINAMATH_CALUDE_min_cost_square_base_l331_33104


namespace NUMINAMATH_CALUDE_unique_prime_pair_solution_l331_33127

theorem unique_prime_pair_solution : 
  ∃! p q : ℕ, Prime p ∧ Prime q ∧ p^3 - q^5 = (p + q)^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_solution_l331_33127


namespace NUMINAMATH_CALUDE_johns_phd_time_l331_33131

/-- Represents the duration of John's PhD journey in years -/
def total_phd_time (
  acclimation_period : ℝ)
  (basics_period : ℝ)
  (research_ratio : ℝ)
  (sabbatical1 : ℝ)
  (sabbatical2 : ℝ)
  (conference1 : ℝ)
  (conference2 : ℝ)
  (dissertation_ratio : ℝ)
  (dissertation_conference : ℝ) : ℝ :=
  acclimation_period +
  basics_period +
  (basics_period * (1 + research_ratio) + sabbatical1 + sabbatical2 + conference1 + conference2) +
  (acclimation_period * dissertation_ratio + dissertation_conference)

/-- Theorem stating that John's total PhD time is 8.75 years -/
theorem johns_phd_time :
  total_phd_time 1 2 0.75 0.5 0.25 (4/12) (5/12) 0.5 0.25 = 8.75 := by
  sorry


end NUMINAMATH_CALUDE_johns_phd_time_l331_33131


namespace NUMINAMATH_CALUDE_certain_number_problem_l331_33197

theorem certain_number_problem (A B : ℝ) (h1 : A + B = 15) (h2 : A = 7) :
  ∃ C : ℝ, C * B = 5 * A - 11 ∧ C = 3 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l331_33197


namespace NUMINAMATH_CALUDE_color_film_fraction_l331_33171

theorem color_film_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw := 20 * x
  let total_color := 8 * y
  let selected_bw := y / 5
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  selected_color / total_selected = 40 / 41 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l331_33171
