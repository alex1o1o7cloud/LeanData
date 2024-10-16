import Mathlib

namespace NUMINAMATH_CALUDE_parabola_points_relation_l1745_174541

/-- Prove that for points A(1, y₁) and B(2, y₂) lying on the parabola y = a(x+1)² + 2 where a < 0, 
    the relationship 2 > y₁ > y₂ holds. -/
theorem parabola_points_relation (a y₁ y₂ : ℝ) : 
  a < 0 → 
  y₁ = a * (1 + 1)^2 + 2 → 
  y₂ = a * (2 + 1)^2 + 2 → 
  2 > y₁ ∧ y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_relation_l1745_174541


namespace NUMINAMATH_CALUDE_right_triangle_sum_squares_l1745_174574

theorem right_triangle_sum_squares (A B C : ℝ × ℝ) : 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →  -- Right triangle condition
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = ((A.1 - C.1)^2 + (A.2 - C.2)^2) →  -- BC is hypotenuse
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 →  -- BC = 2
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.1 - C.1)^2 + (A.2 - C.2)^2 + (C.1 - B.1)^2 + (C.2 - B.2)^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sum_squares_l1745_174574


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l1745_174572

/-- Represents the number of coins in the final distribution step -/
def x : ℕ := 13

/-- The sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Pete's coins after the distribution -/
def pete_coins : ℕ := 5 * x^2

/-- Paul's coins after the distribution -/
def paul_coins : ℕ := x^2

/-- The total number of coins -/
def total_coins : ℕ := pete_coins + paul_coins

theorem pirate_treasure_distribution :
  (sum_of_squares x = pete_coins) ∧
  (total_coins = 1014) := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l1745_174572


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1745_174547

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x₀ : ℝ, x₀^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1745_174547


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l1745_174567

theorem price_reduction_theorem (x : ℝ) : 
  (1 - x / 100) * 1.8 = 1.17 → x = 35 := by sorry

end NUMINAMATH_CALUDE_price_reduction_theorem_l1745_174567


namespace NUMINAMATH_CALUDE_infinite_power_tower_eq_four_l1745_174535

/-- The limit of the infinite power tower x^(x^(x^...)) -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := Real.log x / Real.log (Real.log x)

/-- Theorem stating that if the infinite power tower of x equals 4, then x equals √2 -/
theorem infinite_power_tower_eq_four (x : ℝ) (h : x > 0) :
  infinitePowerTower x = 4 → x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_power_tower_eq_four_l1745_174535


namespace NUMINAMATH_CALUDE_equation_solution_l1745_174557

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (x^2 + 2*x + 3) / (x + 2) = x + 3 ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1745_174557


namespace NUMINAMATH_CALUDE_cube_sum_over_product_is_18_l1745_174563

theorem cube_sum_over_product_is_18 
  (a b c : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_15 : a + b + c = 15)
  (squared_diff_sum : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_is_18_l1745_174563


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l1745_174527

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1 -/
theorem man_son_age_ratio :
  ∀ (man_age son_age : ℕ),
    man_age = son_age + 32 →
    son_age = 30 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l1745_174527


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_sum_l1745_174526

/-- Given two arithmetic sequences {a_n} and {b_n} with the sum of their first n terms
    denoted as (A_n, B_n), where A_n / B_n = (5n + 12) / (2n + 3) for all n,
    prove that a_5 / b_5 + a_7 / b_12 = 30/7. -/
theorem arithmetic_sequence_ratio_sum (a b : ℕ → ℚ) (A B : ℕ → ℚ) :
  (∀ n, A n / B n = (5 * n + 12) / (2 * n + 3)) →
  (∀ n, A n = n * (a 1 + a n) / 2) →
  (∀ n, B n = n * (b 1 + b n) / 2) →
  a 5 / b 5 + a 7 / b 12 = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_sum_l1745_174526


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1745_174554

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ) 
    (h1 : along_stream = 21) 
    (h2 : against_stream = 9) : 
    (along_stream + against_stream) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1745_174554


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1745_174508

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 4*m + 3)
  (z.re = 0 ∧ z.im ≠ 0) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1745_174508


namespace NUMINAMATH_CALUDE_number_puzzle_l1745_174580

theorem number_puzzle : ∃ x : ℚ, (x / 4 + 15 = 4 * x - 15) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1745_174580


namespace NUMINAMATH_CALUDE_orange_balls_count_l1745_174532

def ball_problem (total red blue pink orange : ℕ) : Prop :=
  total = 50 ∧
  red = 20 ∧
  blue = 10 ∧
  total = red + blue + pink + orange ∧
  pink = 3 * orange

theorem orange_balls_count :
  ∀ total red blue pink orange : ℕ,
  ball_problem total red blue pink orange →
  orange = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_balls_count_l1745_174532


namespace NUMINAMATH_CALUDE_find_a_l1745_174501

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- State the theorem
theorem find_a : 
  ∃ (a : ℝ), (∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1745_174501


namespace NUMINAMATH_CALUDE_cyclist_journey_l1745_174596

theorem cyclist_journey (v t : ℝ) 
  (h1 : (v + 1) * (t - 0.5) = v * t)
  (h2 : (v - 1) * (t + 1) = v * t)
  : v * t = 6 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_journey_l1745_174596


namespace NUMINAMATH_CALUDE_selection_theorem_l1745_174597

/-- The number of ways to choose a president, vice-president, and 2-person committee from 10 people -/
def selection_ways (n : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) 2)

/-- Theorem stating the number of ways to make the selection -/
theorem selection_theorem :
  selection_ways 10 = 2520 :=
by sorry

end NUMINAMATH_CALUDE_selection_theorem_l1745_174597


namespace NUMINAMATH_CALUDE_linear_equation_k_value_l1745_174520

theorem linear_equation_k_value (k : ℤ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (k - 3) * x^(abs k - 2) + 5 = a * x + b) → 
  k = -3 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_k_value_l1745_174520


namespace NUMINAMATH_CALUDE_stating_sidorov_cash_sum_l1745_174584

/-- The disposable cash of the Sidorov family as of June 1, 2018 -/
def sidorov_cash : ℝ := 724506.3

/-- The first part of the Sidorov family's cash -/
def first_part : ℝ := 496941.3

/-- The second part of the Sidorov family's cash -/
def second_part : ℝ := 227565

/-- 
Theorem stating that the disposable cash of the Sidorov family 
as of June 1, 2018, is the sum of two given parts
-/
theorem sidorov_cash_sum : 
  sidorov_cash = first_part + second_part := by
  sorry

end NUMINAMATH_CALUDE_stating_sidorov_cash_sum_l1745_174584


namespace NUMINAMATH_CALUDE_diophantine_approximation_l1745_174506

theorem diophantine_approximation (x : ℝ) : 
  ∀ N : ℕ, ∃ p q : ℤ, q > N ∧ |x - (p : ℝ) / (q : ℝ)| < 1 / (q : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_approximation_l1745_174506


namespace NUMINAMATH_CALUDE_shekar_average_marks_l1745_174590

/-- Represents Shekar's scores in different subjects -/
structure ShekarScores where
  mathematics : ℕ
  science : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average of Shekar's scores -/
def average_marks (scores : ShekarScores) : ℚ :=
  (scores.mathematics + scores.science + scores.social_studies + scores.english + scores.biology) / 5

/-- Theorem stating that Shekar's average marks are 77 -/
theorem shekar_average_marks :
  let scores : ShekarScores := {
    mathematics := 76,
    science := 65,
    social_studies := 82,
    english := 67,
    biology := 95
  }
  average_marks scores = 77 := by sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l1745_174590


namespace NUMINAMATH_CALUDE_prob_at_least_one_is_correct_l1745_174514

/-- The probability that person A tells the truth -/
def prob_A : ℝ := 0.8

/-- The probability that person B tells the truth -/
def prob_B : ℝ := 0.6

/-- The probability that person C tells the truth -/
def prob_C : ℝ := 0.75

/-- The probability that at least one person tells the truth -/
def prob_at_least_one : ℝ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

theorem prob_at_least_one_is_correct : prob_at_least_one = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_is_correct_l1745_174514


namespace NUMINAMATH_CALUDE_binomial_divisibility_l1745_174503

theorem binomial_divisibility (k : ℤ) : k ≠ 1 ↔ ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ i : ℕ, (f i + k : ℤ) ∣ Nat.choose (2 * f i) (f i) → False :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l1745_174503


namespace NUMINAMATH_CALUDE_diagonal_sum_property_l1745_174543

/-- A convex regular polygon with 3k sides, where k > 4 is an integer -/
structure RegularPolygon (k : ℕ) :=
  (sides : ℕ)
  (convex : Bool)
  (regular : Bool)
  (k_gt_4 : k > 4)
  (sides_eq_3k : sides = 3 * k)

/-- A diagonal in a polygon -/
structure Diagonal (P : RegularPolygon k) :=
  (length : ℝ)

/-- Theorem: In a convex regular polygon with 3k sides (k > 4), 
    there exist diagonals whose lengths are equal to the sum of 
    the lengths of two shorter diagonals -/
theorem diagonal_sum_property (k : ℕ) (P : RegularPolygon k) :
  ∃ (d1 d2 d3 : Diagonal P), 
    d1.length = d2.length + d3.length ∧ 
    d1.length > d2.length ∧ 
    d1.length > d3.length :=
  sorry

end NUMINAMATH_CALUDE_diagonal_sum_property_l1745_174543


namespace NUMINAMATH_CALUDE_day_365_is_tuesday_l1745_174539

/-- Days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to get the next day of the week -/
def nextDay (d : Weekday) : Weekday :=
  match d with
  | Weekday.Monday => Weekday.Tuesday
  | Weekday.Tuesday => Weekday.Wednesday
  | Weekday.Wednesday => Weekday.Thursday
  | Weekday.Thursday => Weekday.Friday
  | Weekday.Friday => Weekday.Saturday
  | Weekday.Saturday => Weekday.Sunday
  | Weekday.Sunday => Weekday.Monday

/-- Function to advance a day by n days -/
def advanceDay (d : Weekday) (n : Nat) : Weekday :=
  match n with
  | 0 => d
  | n + 1 => nextDay (advanceDay d n)

/-- Theorem: If the 15th day of a 365-day year is a Tuesday, 
    then the 365th day is also a Tuesday -/
theorem day_365_is_tuesday (h : advanceDay Weekday.Tuesday 14 = Weekday.Tuesday) : 
  advanceDay Weekday.Tuesday 364 = Weekday.Tuesday := by
  sorry


end NUMINAMATH_CALUDE_day_365_is_tuesday_l1745_174539


namespace NUMINAMATH_CALUDE_total_bills_is_30_l1745_174592

/-- Represents the number of $10 bills -/
def num_ten_bills : ℕ := 27

/-- Represents the number of $20 bills -/
def num_twenty_bills : ℕ := 3

/-- Represents the total value of all bills in dollars -/
def total_value : ℕ := 330

/-- Theorem stating that the total number of bills is 30 -/
theorem total_bills_is_30 : num_ten_bills + num_twenty_bills = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_bills_is_30_l1745_174592


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l1745_174565

/-- The area of a circle inscribed in an equilateral triangle with side length 24 cm is 48π cm². -/
theorem inscribed_circle_area (s : ℝ) (h : s = 24) : 
  let r := s * Real.sqrt 3 / 6
  π * r^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_l1745_174565


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_range_l1745_174568

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (3/2) 3 := by sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_range_l1745_174568


namespace NUMINAMATH_CALUDE_nearest_year_with_more_zeros_than_ones_l1745_174505

/-- Given a natural number, returns the number of ones in its binary representation. -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Given a natural number, returns the number of zeros in its binary representation. -/
def countZeros (n : ℕ) : ℕ := sorry

/-- Theorem: 2048 is the smallest integer greater than 2017 such that in its binary representation, 
    the number of ones is less than or equal to the number of zeros. -/
theorem nearest_year_with_more_zeros_than_ones : 
  ∀ k : ℕ, k > 2017 → k < 2048 → countOnes k > countZeros k :=
by sorry

end NUMINAMATH_CALUDE_nearest_year_with_more_zeros_than_ones_l1745_174505


namespace NUMINAMATH_CALUDE_product_of_exponents_l1745_174515

theorem product_of_exponents (p r s : ℕ) : 
  4^p + 4^2 = 280 → 
  3^r + 29 = 56 → 
  7^s + 6^3 + 7^2 = 728 → 
  p * r * s = 27 := by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l1745_174515


namespace NUMINAMATH_CALUDE_factory_solution_l1745_174573

def factory_problem (total_employees : ℕ) : Prop :=
  ∃ (employees_17 : ℕ),
    -- 200 employees earn $12/hour
    -- 40 employees earn $14/hour
    -- The rest earn $17/hour
    total_employees = 200 + 40 + employees_17 ∧
    -- The cost for one 8-hour shift is $31840
    31840 = (200 * 12 + 40 * 14 + employees_17 * 17) * 8

theorem factory_solution : ∃ (total_employees : ℕ), factory_problem total_employees ∧ total_employees = 300 := by
  sorry

end NUMINAMATH_CALUDE_factory_solution_l1745_174573


namespace NUMINAMATH_CALUDE_solution_set_transformation_l1745_174516

/-- Given that the solution set of ax^2 + bx + c > 0 is (1, 2),
    prove that the solution set of cx^2 + bx + a > 0 is (1/2, 1) -/
theorem solution_set_transformation (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + c > 0 ↔ 1 < x ∧ x < 2) →
  (∀ x : ℝ, c*x^2 + b*x + a > 0 ↔ 1/2 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_transformation_l1745_174516


namespace NUMINAMATH_CALUDE_last_digit_sum_l1745_174509

theorem last_digit_sum (x y : ℕ) : 
  (135^x + 31^y + 56^(x+y)) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_last_digit_sum_l1745_174509


namespace NUMINAMATH_CALUDE_karen_cindy_crayon_difference_l1745_174582

theorem karen_cindy_crayon_difference :
  let karen_crayons : ℕ := 639
  let cindy_crayons : ℕ := 504
  karen_crayons - cindy_crayons = 135 :=
by sorry

end NUMINAMATH_CALUDE_karen_cindy_crayon_difference_l1745_174582


namespace NUMINAMATH_CALUDE_lamps_remain_lighted_l1745_174576

def toggle_lamps (n : ℕ) : ℕ :=
  n - (n / 2 + n / 3 + n / 5 - n / 6 - n / 10 - n / 15 + n / 30)

theorem lamps_remain_lighted :
  toggle_lamps 2015 = 1006 := by
  sorry

end NUMINAMATH_CALUDE_lamps_remain_lighted_l1745_174576


namespace NUMINAMATH_CALUDE_sequence_limit_property_l1745_174530

theorem sequence_limit_property (a : ℕ → ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a (n + 2) - a n| < ε) →
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |((a (n + 1) - a n) : ℝ) / n| < ε) :=
by sorry

end NUMINAMATH_CALUDE_sequence_limit_property_l1745_174530


namespace NUMINAMATH_CALUDE_discounted_price_theorem_l1745_174564

/-- The original price of an article before discounts -/
def original_price : ℝ := 150

/-- The first discount rate -/
def discount1 : ℝ := 0.1

/-- The second discount rate -/
def discount2 : ℝ := 0.2

/-- The final sale price after discounts -/
def final_price : ℝ := 108

/-- Theorem stating that the original price results in the final price after discounts -/
theorem discounted_price_theorem :
  final_price = original_price * (1 - discount1) * (1 - discount2) := by
  sorry

#check discounted_price_theorem

end NUMINAMATH_CALUDE_discounted_price_theorem_l1745_174564


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l1745_174581

/-- Three points are collinear if and only if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The value of k for which the points (2, -3), (k, k + 2), and (-3k + 4, 1) are collinear. -/
theorem collinear_points_k_value :
  ∃ k : ℝ, collinear 2 (-3) k (k + 2) (-3 * k + 4) 1 ∧
    (k = (17 + Real.sqrt 505) / (-6) ∨ k = (17 - Real.sqrt 505) / (-6)) := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l1745_174581


namespace NUMINAMATH_CALUDE_new_boarders_count_new_boarders_joined_school_l1745_174548

theorem new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders : ℕ) (initial_ratio_day : ℕ)
                            (new_ratio_boarders : ℕ) (new_ratio_day : ℕ) : ℕ :=
  let initial_day_students := initial_boarders * initial_ratio_day / initial_ratio_boarders
  let new_boarders := initial_day_students * new_ratio_boarders / new_ratio_day - initial_boarders
  new_boarders

theorem new_boarders_joined_school :
  new_boarders_count 60 2 5 1 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_new_boarders_count_new_boarders_joined_school_l1745_174548


namespace NUMINAMATH_CALUDE_sierpinski_carpet_area_sum_l1745_174559

/-- Sierpinski carpet area calculation -/
theorem sierpinski_carpet_area_sum (n : ℕ) : 
  let initial_area : ℝ := Real.sqrt 3 / 4
  let removed_area_sum : ℝ → ℕ → ℝ := λ a k => a * (1 - (3/4)^k)
  removed_area_sum initial_area n = (Real.sqrt 3 / 4) * (1 - (3/4)^n) := by
  sorry

end NUMINAMATH_CALUDE_sierpinski_carpet_area_sum_l1745_174559


namespace NUMINAMATH_CALUDE_fountain_walkway_ratio_l1745_174512

-- Define the park setup
def park (n : ℕ) (s d : ℝ) : Prop :=
  ∃ (total_area fountain_area : ℝ),
    total_area = (n * s + 2 * n * d)^2 ∧
    fountain_area = n^2 * s^2 ∧
    fountain_area / total_area = 0.4

-- Theorem statement
theorem fountain_walkway_ratio :
  ∀ (n : ℕ) (s d : ℝ),
    n = 10 →
    park n s d →
    d / s = 1 / 3.44 :=
by sorry

end NUMINAMATH_CALUDE_fountain_walkway_ratio_l1745_174512


namespace NUMINAMATH_CALUDE_factor_proof_l1745_174570

theorem factor_proof :
  (∃ n : ℤ, 24 = 4 * n) ∧ (∃ m : ℤ, 180 = 9 * m) := by
  sorry

end NUMINAMATH_CALUDE_factor_proof_l1745_174570


namespace NUMINAMATH_CALUDE_max_basketballs_min_basketballs_for_profit_max_profit_l1745_174550

/-- Represents the sports equipment store problem -/
structure StoreProblem where
  total_balls : ℕ
  max_payment : ℕ
  basketball_wholesale : ℕ
  volleyball_wholesale : ℕ
  basketball_retail : ℕ
  volleyball_retail : ℕ
  min_profit : ℕ

/-- The specific instance of the store problem -/
def store_instance : StoreProblem :=
  { total_balls := 100
  , max_payment := 11815
  , basketball_wholesale := 130
  , volleyball_wholesale := 100
  , basketball_retail := 160
  , volleyball_retail := 120
  , min_profit := 2580
  }

/-- Calculates the total cost of purchasing basketballs and volleyballs -/
def total_cost (p : StoreProblem) (basketballs : ℕ) : ℕ :=
  p.basketball_wholesale * basketballs + p.volleyball_wholesale * (p.total_balls - basketballs)

/-- Calculates the profit from selling all balls -/
def profit (p : StoreProblem) (basketballs : ℕ) : ℕ :=
  (p.basketball_retail - p.basketball_wholesale) * basketballs +
  (p.volleyball_retail - p.volleyball_wholesale) * (p.total_balls - basketballs)

/-- Theorem stating the maximum number of basketballs that can be purchased -/
theorem max_basketballs (p : StoreProblem) :
  ∃ (max_basketballs : ℕ),
    (∀ (b : ℕ), total_cost p b ≤ p.max_payment → b ≤ max_basketballs) ∧
    total_cost p max_basketballs ≤ p.max_payment ∧
    max_basketballs = 60 :=
  sorry

/-- Theorem stating the minimum number of basketballs needed for desired profit -/
theorem min_basketballs_for_profit (p : StoreProblem) :
  ∃ (min_basketballs : ℕ),
    (∀ (b : ℕ), profit p b ≥ p.min_profit → b ≥ min_basketballs) ∧
    profit p min_basketballs ≥ p.min_profit ∧
    min_basketballs = 58 :=
  sorry

/-- Theorem stating the maximum profit achievable -/
theorem max_profit (p : StoreProblem) :
  ∃ (max_profit : ℕ),
    (∀ (b : ℕ), total_cost p b ≤ p.max_payment → profit p b ≤ max_profit) ∧
    (∃ (b : ℕ), total_cost p b ≤ p.max_payment ∧ profit p b = max_profit) ∧
    max_profit = 2600 :=
  sorry

end NUMINAMATH_CALUDE_max_basketballs_min_basketballs_for_profit_max_profit_l1745_174550


namespace NUMINAMATH_CALUDE_cannot_determine_books_left_l1745_174562

def initial_pens : ℕ := 42
def initial_books : ℕ := 143
def pens_sold : ℕ := 23
def pens_left : ℕ := 19

theorem cannot_determine_books_left : 
  ∀ (books_left : ℕ), 
  initial_pens = pens_sold + pens_left →
  ¬(∀ (books_sold : ℕ), initial_books = books_sold + books_left) :=
by
  sorry

end NUMINAMATH_CALUDE_cannot_determine_books_left_l1745_174562


namespace NUMINAMATH_CALUDE_no_functions_exist_l1745_174534

theorem no_functions_exist : ¬ ∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_functions_exist_l1745_174534


namespace NUMINAMATH_CALUDE_chinese_character_multiplication_l1745_174589

theorem chinese_character_multiplication : ∃! (x y : Nat), 
  x ≠ y ∧ x ≠ 3 ∧ x ≠ 0 ∧ y ≠ 3 ∧ y ≠ 0 ∧
  (3000 + 100 * x + y) * (3000 + 100 * x + y) ≥ 10000000 ∧
  (3000 + 100 * x + y) * (3000 + 100 * x + y) < 100000000 :=
by sorry

#check chinese_character_multiplication

end NUMINAMATH_CALUDE_chinese_character_multiplication_l1745_174589


namespace NUMINAMATH_CALUDE_lowest_price_calculation_l1745_174500

/-- Calculates the lowest price per component to avoid loss --/
def lowest_price_per_component (production_cost shipping_cost : ℚ) 
  (fixed_monthly_cost : ℚ) (production_volume : ℕ) : ℚ :=
  let total_cost := production_volume * (production_cost + shipping_cost) + fixed_monthly_cost
  total_cost / production_volume

/-- Theorem: The lowest price per component is the total cost divided by the number of components --/
theorem lowest_price_calculation (production_cost shipping_cost : ℚ) 
  (fixed_monthly_cost : ℚ) (production_volume : ℕ) :
  lowest_price_per_component production_cost shipping_cost fixed_monthly_cost production_volume = 
  (production_volume * (production_cost + shipping_cost) + fixed_monthly_cost) / production_volume :=
by
  sorry

#eval lowest_price_per_component 80 4 16500 150

end NUMINAMATH_CALUDE_lowest_price_calculation_l1745_174500


namespace NUMINAMATH_CALUDE_exponent_increase_l1745_174583

theorem exponent_increase (x : ℝ) (y : ℝ) (h : 3^x = y) : 3^(x+1) = 3*y := by
  sorry

end NUMINAMATH_CALUDE_exponent_increase_l1745_174583


namespace NUMINAMATH_CALUDE_highest_score_is_103_l1745_174598

def base_score : ℕ := 100

def score_adjustments : List ℤ := [3, -8, 0]

def actual_scores : List ℕ := score_adjustments.map (λ x => (base_score : ℤ) + x |>.toNat)

theorem highest_score_is_103 : actual_scores.maximum? = some 103 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_is_103_l1745_174598


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1745_174566

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1745_174566


namespace NUMINAMATH_CALUDE_fourth_derivative_y_l1745_174517

noncomputable def y (x : ℝ) : ℝ := (5 * x - 8) * (2 ^ (-x))

theorem fourth_derivative_y (x : ℝ) :
  (deriv^[4] y) x = 2^(-x) * (Real.log 2)^4 * (5*x - 9) := by sorry

end NUMINAMATH_CALUDE_fourth_derivative_y_l1745_174517


namespace NUMINAMATH_CALUDE_purple_ring_weight_l1745_174577

/-- The weight of the purple ring in Karin's science class experiment -/
theorem purple_ring_weight (orange_weight white_weight total_weight : ℚ)
  (h_orange : orange_weight = 8/100)
  (h_white : white_weight = 42/100)
  (h_total : total_weight = 83/100) :
  total_weight - orange_weight - white_weight = 33/100 := by
  sorry

end NUMINAMATH_CALUDE_purple_ring_weight_l1745_174577


namespace NUMINAMATH_CALUDE_bicycle_weight_is_12_l1745_174537

-- Define the weight of a bicycle and a car
def bicycle_weight : ℝ := sorry
def car_weight : ℝ := sorry

-- State the theorem
theorem bicycle_weight_is_12 :
  (10 * bicycle_weight = 4 * car_weight) →
  (3 * car_weight = 90) →
  bicycle_weight = 12 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_weight_is_12_l1745_174537


namespace NUMINAMATH_CALUDE_vending_machine_probability_l1745_174510

def num_toys : ℕ := 10
def min_cost : ℚ := 1/2
def max_cost : ℚ := 5
def cost_difference : ℚ := 1/2
def initial_half_dollars : ℕ := 10
def favorite_toy_cost : ℚ := 9/2

theorem vending_machine_probability :
  let total_permutations : ℕ := num_toys.factorial
  let favorable_outcomes : ℕ := (num_toys - 1).factorial + (num_toys - 2).factorial
  (1 : ℚ) - (favorable_outcomes : ℚ) / (total_permutations : ℚ) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_vending_machine_probability_l1745_174510


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1745_174578

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 8)
  (h_fourth : a 4 = 7) : 
  a 5 = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1745_174578


namespace NUMINAMATH_CALUDE_marathon_time_l1745_174523

theorem marathon_time (dean_time : ℝ) 
  (h1 : dean_time > 0)
  (h2 : dean_time * (2/3) * (1 + 1/3) + dean_time * (3/2) + dean_time = 23) : 
  dean_time = 23/3 := by
sorry

end NUMINAMATH_CALUDE_marathon_time_l1745_174523


namespace NUMINAMATH_CALUDE_cube_root_equation_solutions_l1745_174521

theorem cube_root_equation_solutions :
  let f (x : ℝ) := (18 * x - 3)^(1/3) + (12 * x + 3)^(1/3) - 5 * x^(1/3)
  { x : ℝ | f x = 0 } = 
    { 0, (-27 + Real.sqrt 18477) / 1026, (-27 - Real.sqrt 18477) / 1026 } := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solutions_l1745_174521


namespace NUMINAMATH_CALUDE_larger_number_is_588_l1745_174545

/-- Given two positive integers with HCF 42 and LCM factors 12 and 14, the larger number is 588 -/
theorem larger_number_is_588 (a b : ℕ+) (hcf : Nat.gcd a b = 42) 
  (lcm_factors : ∃ (x y : ℕ+), x = 12 ∧ y = 14 ∧ Nat.lcm a b = 42 * x * y) :
  max a b = 588 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_588_l1745_174545


namespace NUMINAMATH_CALUDE_adam_first_half_correct_l1745_174528

/-- Represents the trivia game scenario -/
structure TriviaGame where
  pointsPerQuestion : ℕ
  secondHalfCorrect : ℕ
  finalScore : ℕ

/-- Calculates the number of correctly answered questions in the first half -/
def firstHalfCorrect (game : TriviaGame) : ℕ :=
  (game.finalScore - game.secondHalfCorrect * game.pointsPerQuestion) / game.pointsPerQuestion

/-- Theorem stating that Adam answered 8 questions correctly in the first half -/
theorem adam_first_half_correct :
  let game : TriviaGame := {
    pointsPerQuestion := 8,
    secondHalfCorrect := 2,
    finalScore := 80
  }
  firstHalfCorrect game = 8 := by sorry

end NUMINAMATH_CALUDE_adam_first_half_correct_l1745_174528


namespace NUMINAMATH_CALUDE_line_arrangement_result_l1745_174553

/-- The number of ways to arrange 3 boys and 3 girls in a line with two girls together -/
def line_arrangement (num_boys num_girls : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating that the number of arrangements is 432 -/
theorem line_arrangement_result : line_arrangement 3 3 = 432 := by
  sorry

end NUMINAMATH_CALUDE_line_arrangement_result_l1745_174553


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1745_174561

/-- The y-intercept of the line 4x + 7y - 3xy = 28 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) : 
  (4 * x + 7 * y - 3 * x * y = 28) → 
  (x = 0 → y = 4) :=
by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1745_174561


namespace NUMINAMATH_CALUDE_expand_product_l1745_174599

theorem expand_product (x : ℝ) : (x + 3)^2 * (x - 5) = x^3 + x^2 - 21*x - 45 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1745_174599


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l1745_174556

theorem product_from_lcm_and_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 90) 
  (h2 : Nat.gcd a b = 10) : 
  a * b = 900 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l1745_174556


namespace NUMINAMATH_CALUDE_periodic_function_value_l1745_174524

-- Define a periodic function with period 2
def isPeriodic2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f x

-- Theorem statement
theorem periodic_function_value (f : ℝ → ℝ) (h1 : isPeriodic2 f) (h2 : f 2 = 2) :
  f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l1745_174524


namespace NUMINAMATH_CALUDE_vector_operation_result_l1745_174536

theorem vector_operation_result :
  let a : ℝ × ℝ × ℝ := (-3, 5, 2)
  let b : ℝ × ℝ × ℝ := (1, -1, 3)
  let c : ℝ × ℝ × ℝ := (2, 0, -4)
  a - 4 • b + c = (-5, 9, -14) :=
by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l1745_174536


namespace NUMINAMATH_CALUDE_roots_opposite_signs_l1745_174540

theorem roots_opposite_signs (c d e : ℝ) (n : ℝ) : 
  (∃ y₁ y₂ : ℝ, y₁ = -y₂ ∧ y₁ ≠ 0 ∧
    (y₁^2 - d*y₁) / (c*y₁ - e) = (n - 2) / (n + 2) ∧
    (y₂^2 - d*y₂) / (c*y₂ - e) = (n - 2) / (n + 2)) →
  n = -2 :=
by sorry

end NUMINAMATH_CALUDE_roots_opposite_signs_l1745_174540


namespace NUMINAMATH_CALUDE_half_floors_full_capacity_l1745_174518

/-- Represents a building with floors, apartments, and occupants. -/
structure Building where
  total_floors : ℕ
  apartments_per_floor : ℕ
  people_per_apartment : ℕ
  total_people : ℕ

/-- Calculates the number of full-capacity floors in the building. -/
def full_capacity_floors (b : Building) : ℕ :=
  let people_per_full_floor := b.apartments_per_floor * b.people_per_apartment
  let total_full_floor_capacity := b.total_floors * people_per_full_floor
  (2 * b.total_people - total_full_floor_capacity) / people_per_full_floor

/-- Theorem stating that for a building with specific parameters,
    the number of full-capacity floors is half the total floors. -/
theorem half_floors_full_capacity (b : Building)
    (h1 : b.total_floors = 12)
    (h2 : b.apartments_per_floor = 10)
    (h3 : b.people_per_apartment = 4)
    (h4 : b.total_people = 360) :
    full_capacity_floors b = b.total_floors / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_floors_full_capacity_l1745_174518


namespace NUMINAMATH_CALUDE_infinite_solutions_exist_l1745_174504

theorem infinite_solutions_exist : 
  ∀ n : ℕ, n > 0 → 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 
  x = (1 + 1 / n) * y ∧
  (3 * x^3 + x * y^2) * (x^2 * y + 3 * y^3) = (x - y)^7 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_exist_l1745_174504


namespace NUMINAMATH_CALUDE_goldfish_feeding_l1745_174513

/-- Given that one scoop of fish food can feed 8 goldfish, 
    prove that 4 scoops can feed 32 goldfish -/
theorem goldfish_feeding (scoop_capacity : ℕ) (num_scoops : ℕ) : 
  scoop_capacity = 8 → num_scoops = 4 → num_scoops * scoop_capacity = 32 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_feeding_l1745_174513


namespace NUMINAMATH_CALUDE_camping_hike_distance_l1745_174525

theorem camping_hike_distance (total_distance stream_to_meadow meadow_to_campsite : ℝ)
  (h1 : total_distance = 0.7)
  (h2 : stream_to_meadow = 0.4)
  (h3 : meadow_to_campsite = 0.1) :
  total_distance - (stream_to_meadow + meadow_to_campsite) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_camping_hike_distance_l1745_174525


namespace NUMINAMATH_CALUDE_green_rotten_no_smell_count_l1745_174579

/-- Represents the types of fruits in the orchard -/
inductive Fruit
| Apple
| Orange
| Pear

/-- Represents the colors of fruits -/
inductive Color
| Red
| Green
| Orange
| Yellow
| Brown

structure OrchardData where
  total_fruits : Fruit → ℕ
  color_distribution : Fruit → Color → ℚ
  rotten_percentage : Fruit → ℚ
  strong_smell_percentage : Fruit → ℚ

def orchard_data : OrchardData := {
  total_fruits := λ f => match f with
    | Fruit.Apple => 200
    | Fruit.Orange => 150
    | Fruit.Pear => 100,
  color_distribution := λ f c => match f, c with
    | Fruit.Apple, Color.Red => 1/2
    | Fruit.Apple, Color.Green => 1/2
    | Fruit.Orange, Color.Orange => 2/5
    | Fruit.Orange, Color.Yellow => 3/5
    | Fruit.Pear, Color.Green => 3/10
    | Fruit.Pear, Color.Brown => 7/10
    | _, _ => 0,
  rotten_percentage := λ f => match f with
    | Fruit.Apple => 2/5
    | Fruit.Orange => 1/4
    | Fruit.Pear => 7/20,
  strong_smell_percentage := λ f => match f with
    | Fruit.Apple => 7/10
    | Fruit.Orange => 1/2
    | Fruit.Pear => 4/5
}

/-- Calculates the number of green rotten fruits without a strong smell in the orchard -/
def green_rotten_no_smell (data : OrchardData) : ℕ :=
  sorry

theorem green_rotten_no_smell_count :
  green_rotten_no_smell orchard_data = 14 :=
sorry

end NUMINAMATH_CALUDE_green_rotten_no_smell_count_l1745_174579


namespace NUMINAMATH_CALUDE_sum_of_counts_l1745_174544

/-- A function that returns the count of four-digit even numbers -/
def count_four_digit_even : ℕ :=
  sorry

/-- A function that returns the count of four-digit numbers divisible by both 5 and 3 -/
def count_four_digit_div_by_5_and_3 : ℕ :=
  sorry

/-- Theorem stating that the sum of four-digit even numbers and four-digit numbers
    divisible by both 5 and 3 is equal to 5100 -/
theorem sum_of_counts : count_four_digit_even + count_four_digit_div_by_5_and_3 = 5100 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_counts_l1745_174544


namespace NUMINAMATH_CALUDE_disjunction_true_when_second_true_l1745_174585

theorem disjunction_true_when_second_true (p q : Prop) (hp : ¬p) (hq : q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_true_when_second_true_l1745_174585


namespace NUMINAMATH_CALUDE_counterexample_exists_l1745_174591

theorem counterexample_exists : ∃ (a b : ℝ), a < b ∧ -3 * a ≥ -3 * b := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1745_174591


namespace NUMINAMATH_CALUDE_distance_for_equilateral_hyperbola_locus_l1745_174558

/-- Two circles C1 and C2 with variable tangent t to C1 intersecting C2 at A and B.
    Tangents to C2 through A and B intersect at P. -/
structure TwoCirclesConfig where
  r1 : ℝ  -- radius of C1
  r2 : ℝ  -- radius of C2
  d : ℝ   -- distance between centers of C1 and C2

/-- The locus of P is contained in an equilateral hyperbola -/
def isEquilateralHyperbolaLocus (config : TwoCirclesConfig) : Prop :=
  config.d = config.r1 * Real.sqrt 2

/-- Theorem: The distance between centers for equilateral hyperbola locus -/
theorem distance_for_equilateral_hyperbola_locus 
  (config : TwoCirclesConfig) (h1 : config.r1 > 0) (h2 : config.r2 > 0) :
  isEquilateralHyperbolaLocus config ↔ config.d = config.r1 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_distance_for_equilateral_hyperbola_locus_l1745_174558


namespace NUMINAMATH_CALUDE_max_value_of_g_l1745_174593

/-- The quadratic function f(x, y) -/
def f (x y : ℝ) : ℝ := 10*x - 4*x^2 + 2*x*y

/-- The function g(x) is f(x, 3) -/
def g (x : ℝ) : ℝ := f x 3

theorem max_value_of_g :
  ∃ (m : ℝ), ∀ (x : ℝ), g x ≤ m ∧ ∃ (x₀ : ℝ), g x₀ = m ∧ m = 16 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l1745_174593


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_l1745_174511

theorem reciprocal_equals_self (x : ℝ) : x ≠ 0 → (x = 1/x ↔ x = 1 ∨ x = -1) := by sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_l1745_174511


namespace NUMINAMATH_CALUDE_log_equation_solution_l1745_174552

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 2 = 9 →
  x = 2^(27/10) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1745_174552


namespace NUMINAMATH_CALUDE_equal_products_l1745_174533

theorem equal_products : 2 * 20212021 * 1011 * 202320232023 = 43 * 47 * 20232023 * 202220222022 := by
  sorry

end NUMINAMATH_CALUDE_equal_products_l1745_174533


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1739_l1745_174571

theorem largest_prime_factor_of_1739 :
  (Nat.factors 1739).maximum? = some 47 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1739_l1745_174571


namespace NUMINAMATH_CALUDE_square_plot_area_l1745_174569

/-- Given a square plot with a fence, prove that the area is 289 square feet
    when the price per foot is 59 and the total cost is 4012. -/
theorem square_plot_area (side_length : ℝ) (perimeter : ℝ) (price_per_foot : ℝ) (total_cost : ℝ) :
  price_per_foot = 59 →
  total_cost = 4012 →
  perimeter = 4 * side_length →
  total_cost = perimeter * price_per_foot →
  side_length ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_area_l1745_174569


namespace NUMINAMATH_CALUDE_decimal_89_to_base5_l1745_174586

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Checks if a list of digits is a valid base-5 representation --/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

theorem decimal_89_to_base5 :
  let base5_representation := toBase5 89
  isValidBase5 base5_representation ∧ base5_representation = [4, 2, 3] :=
by sorry

end NUMINAMATH_CALUDE_decimal_89_to_base5_l1745_174586


namespace NUMINAMATH_CALUDE_johns_total_time_l1745_174546

/-- The total time John spent on his book and exploring is 5 years. -/
theorem johns_total_time (exploring_time note_writing_time book_writing_time : ℝ) :
  exploring_time = 3 →
  note_writing_time = exploring_time / 2 →
  book_writing_time = 0.5 →
  exploring_time + note_writing_time + book_writing_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_johns_total_time_l1745_174546


namespace NUMINAMATH_CALUDE_polynomial_value_at_five_l1745_174531

theorem polynomial_value_at_five (p : ℝ → ℝ) :
  (∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) →
  p 1 = 1 →
  p 2 = 2 →
  p 3 = 3 →
  p 4 = 4 →
  p 5 = 29 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_five_l1745_174531


namespace NUMINAMATH_CALUDE_translation_lori_to_alex_l1745_174522

/-- Represents a 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a 2D translation --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Lori's house location --/
def lori_house : Point := ⟨6, 3⟩

/-- Alex's house location --/
def alex_house : Point := ⟨-2, -4⟩

/-- Calculates the translation between two points --/
def calculate_translation (p1 p2 : Point) : Translation :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

/-- Theorem: The translation from Lori's house to Alex's house is 8 units left and 7 units down --/
theorem translation_lori_to_alex :
  let t := calculate_translation lori_house alex_house
  t.dx = -8 ∧ t.dy = -7 := by sorry

end NUMINAMATH_CALUDE_translation_lori_to_alex_l1745_174522


namespace NUMINAMATH_CALUDE_abs_of_negative_2023_l1745_174555

theorem abs_of_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_of_negative_2023_l1745_174555


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1745_174588

theorem water_tank_capacity (initial_fraction : Rat) (final_fraction : Rat) (added_volume : Rat) (total_capacity : Rat) : 
  initial_fraction = 1/3 →
  final_fraction = 2/5 →
  added_volume = 5 →
  initial_fraction * total_capacity + added_volume = final_fraction * total_capacity →
  total_capacity = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1745_174588


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1745_174575

theorem other_root_of_quadratic (m : ℝ) : 
  (1 : ℝ)^2 + m * 1 - 5 = 0 → 
  (-5 : ℝ)^2 + m * (-5) - 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1745_174575


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l1745_174587

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) : 
  total_length = 140 → ratio = 2/5 → 
  ∃ (shorter_piece longer_piece : ℝ), 
    shorter_piece + longer_piece = total_length ∧ 
    shorter_piece = ratio * longer_piece ∧
    shorter_piece = 40 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l1745_174587


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1745_174560

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation x^2 - 4x - 11 = 0 has discriminant 60 -/
theorem quadratic_discriminant :
  discriminant 1 (-4) (-11) = 60 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1745_174560


namespace NUMINAMATH_CALUDE_range_of_a_l1745_174551

def proposition_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  proposition_p a ∧ proposition_q a → a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1745_174551


namespace NUMINAMATH_CALUDE_kaprekar_convergence_l1745_174594

/-- Reverses a four-digit number -/
def reverseNumber (n : Nat) : Nat :=
  sorry

/-- Rearranges digits of a four-digit number from largest to smallest -/
def rearrangeDigits (n : Nat) : Nat :=
  sorry

/-- Applies the Kaprekar transformation to a four-digit number -/
def kaprekarTransform (n : Nat) : Nat :=
  let m := rearrangeDigits n
  let r := reverseNumber m
  m - r

/-- Applies the Kaprekar transformation k times -/
def kaprekarTransformK (n : Nat) (k : Nat) : Nat :=
  sorry

theorem kaprekar_convergence (n : Nat) (h : n = 5298 ∨ n = 4852) :
  ∃ (k : Nat) (t : Nat), k = 7 ∧ t = 6174 ∧
    kaprekarTransformK n k = t ∧
    kaprekarTransform t = t :=
  sorry

end NUMINAMATH_CALUDE_kaprekar_convergence_l1745_174594


namespace NUMINAMATH_CALUDE_watch_loss_percentage_l1745_174542

/-- Proves that the loss percentage is 10% given the conditions of the watch sale problem -/
theorem watch_loss_percentage (cost_price : ℝ) (additional_price : ℝ) (gain_percentage : ℝ) 
  (h1 : cost_price = 1428.57)
  (h2 : additional_price = 200)
  (h3 : gain_percentage = 4) : 
  ∃ (loss_percentage : ℝ), 
    loss_percentage = 10 ∧ 
    cost_price + additional_price = cost_price * (1 + gain_percentage / 100) ∧
    cost_price * (1 - loss_percentage / 100) + additional_price = cost_price * (1 + gain_percentage / 100) :=
by
  sorry


end NUMINAMATH_CALUDE_watch_loss_percentage_l1745_174542


namespace NUMINAMATH_CALUDE_max_sphere_volume_in_prism_max_sphere_volume_in_specific_prism_l1745_174595

/-- The maximum volume of a sphere inscribed in a right triangular prism -/
theorem max_sphere_volume_in_prism (a b h : ℝ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) :
  let r := min (a * b / (a + b)) (h / 2)
  (4 / 3) * Real.pi * r^3 ≤ (4 / 3) * Real.pi * ((3 : ℝ) / 2)^3 :=
by sorry

/-- The specific case for the given prism dimensions -/
theorem max_sphere_volume_in_specific_prism :
  let max_volume := (4 / 3) * Real.pi * ((3 : ℝ) / 2)^3
  max_volume = 9 * Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_volume_in_prism_max_sphere_volume_in_specific_prism_l1745_174595


namespace NUMINAMATH_CALUDE_equation_graph_is_axes_l1745_174538

/-- The set of points (x, y) satisfying (x-y)^2 = x^2 + y^2 is equivalent to the union of the x-axis and y-axis -/
theorem equation_graph_is_axes (x y : ℝ) : 
  (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_graph_is_axes_l1745_174538


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1745_174519

/-- Given two quadratic equations with a specific relationship between their roots,
    prove that the ratio of certain coefficients is 27. -/
theorem quadratic_root_relation (k n p : ℝ) : 
  k ≠ 0 → n ≠ 0 → p ≠ 0 →
  (∃ r₁ r₂ : ℝ, r₁ + r₂ = -p ∧ r₁ * r₂ = k) →
  (∃ s₁ s₂ : ℝ, s₁ + s₂ = -k ∧ s₁ * s₂ = n ∧ s₁ = 3*r₁ ∧ s₂ = 3*r₂) →
  n / p = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1745_174519


namespace NUMINAMATH_CALUDE_power_product_l1745_174507

theorem power_product (x y : ℝ) (h1 : (10 : ℝ) ^ x = 3) (h2 : (10 : ℝ) ^ y = 4) : 
  (10 : ℝ) ^ (x * y) = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_product_l1745_174507


namespace NUMINAMATH_CALUDE_even_odd_function_sum_l1745_174529

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem even_odd_function_sum (f g : ℝ → ℝ) 
  (hf : IsEven f) (hg : IsOdd g) 
  (h : ∀ x, f x + g x = x^2 + 3*x + 1) : 
  ∀ x, f x = x^2 + 1 := by
sorry

end NUMINAMATH_CALUDE_even_odd_function_sum_l1745_174529


namespace NUMINAMATH_CALUDE_restaurant_students_l1745_174502

theorem restaurant_students (burgers hot_dogs pizza_slices sandwiches : ℕ) : 
  burgers = 30 ∧ 
  burgers = 2 * hot_dogs ∧ 
  pizza_slices = hot_dogs + 5 ∧ 
  sandwiches = 3 * pizza_slices → 
  burgers + hot_dogs + pizza_slices + sandwiches = 125 := by
sorry

end NUMINAMATH_CALUDE_restaurant_students_l1745_174502


namespace NUMINAMATH_CALUDE_g10_diamonds_l1745_174549

/-- Number of diamonds in figure G_n -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 10
  else 2 + 2 * n^2 + 2 * n - 4

/-- The sequence of figures G_n satisfies the given properties -/
axiom sequence_property (n : ℕ) (h : n ≥ 3) :
  num_diamonds n = num_diamonds (n - 1) + 4 * (n + 1)

/-- G_1 has 2 diamonds -/
axiom g1_diamonds : num_diamonds 1 = 2

/-- G_2 has 10 diamonds -/
axiom g2_diamonds : num_diamonds 2 = 10

/-- Theorem: G_10 has 218 diamonds -/
theorem g10_diamonds : num_diamonds 10 = 218 := by sorry

end NUMINAMATH_CALUDE_g10_diamonds_l1745_174549
