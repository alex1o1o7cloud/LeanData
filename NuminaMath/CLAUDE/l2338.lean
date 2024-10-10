import Mathlib

namespace quadratic_inequality_l2338_233832

theorem quadratic_inequality (y : ℝ) : y^2 - 6*y - 16 > 0 ↔ y < -2 ∨ y > 8 := by
  sorry

end quadratic_inequality_l2338_233832


namespace work_completion_time_l2338_233838

/-- The number of days it takes for worker a to complete the work alone -/
def days_a : ℝ := 4

/-- The number of days it takes for worker b to complete the work alone -/
def days_b : ℝ := 9

/-- The number of days it takes for workers a, b, and c to complete the work together -/
def days_together : ℝ := 2

/-- The number of days it takes for worker c to complete the work alone -/
def days_c : ℝ := 7.2

theorem work_completion_time :
  (1 / days_a) + (1 / days_b) + (1 / days_c) = (1 / days_together) :=
sorry

end work_completion_time_l2338_233838


namespace nth_row_equation_l2338_233883

theorem nth_row_equation (n : ℕ) : 2 * n + 1 = (n + 1)^2 - n^2 := by
  sorry

end nth_row_equation_l2338_233883


namespace welch_distance_before_pie_l2338_233866

/-- The distance Mr. Welch drove before buying a pie -/
def distance_before_pie (total_distance : ℕ) (distance_after_pie : ℕ) : ℕ :=
  total_distance - distance_after_pie

/-- Theorem: Mr. Welch drove 35 miles before buying a pie -/
theorem welch_distance_before_pie :
  distance_before_pie 78 43 = 35 := by
  sorry

end welch_distance_before_pie_l2338_233866


namespace nested_fraction_equality_l2338_233823

theorem nested_fraction_equality : 1 + (1 / (1 + (1 / (1 + (1 / (1 + 2)))))) = 7 / 4 := by
  sorry

end nested_fraction_equality_l2338_233823


namespace pauls_crayons_left_l2338_233852

/-- Represents the number of crayons Paul had left at the end of the school year. -/
def crayons_left (initial_erasers initial_crayons : ℕ) (extra_crayons : ℕ) : ℕ :=
  initial_erasers + extra_crayons

/-- Theorem stating that Paul had 523 crayons left at the end of the school year. -/
theorem pauls_crayons_left :
  crayons_left 457 617 66 = 523 := by
  sorry

end pauls_crayons_left_l2338_233852


namespace infinitely_many_divisible_by_digit_sum_l2338_233839

/-- Function to create a number with n digits of 1 -/
def oneDigits (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Function to calculate the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sumOfDigits (n / 10)

/-- Theorem stating that there are infinitely many integers divisible by the sum of their digits -/
theorem infinitely_many_divisible_by_digit_sum :
  ∀ n : ℕ, ∃ k : ℕ,
    k > 0 ∧
    (∀ d : ℕ, d > 0 → d < 10 → k % d ≠ 0) ∧ 
    (k % (sumOfDigits k) = 0) :=
by
  intro n
  use oneDigits (3^n)
  sorry

/-- Lemma: The number created by oneDigits(3^n) has exactly 3^n digits, all of which are 1 -/
lemma oneDigits_all_ones (n : ℕ) :
  ∀ d : ℕ, d > 0 → d < 10 → (oneDigits (3^n)) % d ≠ 0 :=
by sorry

/-- Lemma: The sum of digits of oneDigits(3^n) is equal to 3^n -/
lemma sum_of_digits_oneDigits (n : ℕ) :
  sumOfDigits (oneDigits (3^n)) = 3^n :=
by sorry

/-- Lemma: oneDigits(3^n) is divisible by 3^n -/
lemma oneDigits_divisible (n : ℕ) :
  (oneDigits (3^n)) % (3^n) = 0 :=
by sorry

end infinitely_many_divisible_by_digit_sum_l2338_233839


namespace orange_packing_l2338_233837

/-- Given a fruit farm that packs oranges in boxes with a variable capacity,
    this theorem proves the relationship between the number of boxes used,
    the total number of oranges, and the capacity of each box. -/
theorem orange_packing (x : ℕ+) :
  (5623 : ℕ) / x.val = (5623 : ℕ) / x.val := by sorry

end orange_packing_l2338_233837


namespace equation_roots_existence_and_bounds_l2338_233867

theorem equation_roots_existence_and_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ x₁ x₂ : ℝ, 
    (1 / x₁ + 1 / (x₁ - a) + 1 / (x₁ + b) = 0) ∧
    (1 / x₂ + 1 / (x₂ - a) + 1 / (x₂ + b) = 0) ∧
    (a / 3 < x₁ ∧ x₁ < 2 * a / 3) ∧
    (-2 * b / 3 < x₂ ∧ x₂ < -b / 3) :=
by sorry

end equation_roots_existence_and_bounds_l2338_233867


namespace joans_kittens_l2338_233800

theorem joans_kittens (initial_kittens : ℕ) (given_away : ℕ) (remaining : ℕ) 
  (h1 : given_away = 2) 
  (h2 : remaining = 6) 
  (h3 : initial_kittens = remaining + given_away) : initial_kittens = 8 :=
by sorry

end joans_kittens_l2338_233800


namespace parallelogram_area_l2338_233896

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 and 12 is 60 square units. -/
theorem parallelogram_area (a b : ℝ) (angle : ℝ) (h1 : a = 10) (h2 : b = 12) (h3 : angle = 150) :
  a * b * Real.sin (angle * π / 180) = 60 := by
  sorry

end parallelogram_area_l2338_233896


namespace quadratic_expression_value_l2338_233802

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 9) 
  (eq2 : x + 3 * y = 10) : 
  10 * x^2 + 19 * x * y + 10 * y^2 = 181 := by
  sorry

end quadratic_expression_value_l2338_233802


namespace brick_wall_theorem_l2338_233893

/-- Represents a brick wall with a given number of rows, total bricks, and bricks in the bottom row. -/
structure BrickWall where
  rows : ℕ
  totalBricks : ℕ
  bottomRowBricks : ℕ

/-- Calculates the number of bricks in a given row of the wall. -/
def bricksInRow (wall : BrickWall) (rowNumber : ℕ) : ℕ :=
  wall.bottomRowBricks - (rowNumber - 1)

theorem brick_wall_theorem (wall : BrickWall) 
    (h1 : wall.rows = 5)
    (h2 : wall.totalBricks = 100)
    (h3 : wall.bottomRowBricks = 18) :
    ∀ (r : ℕ), 1 < r ∧ r ≤ wall.rows → 
    bricksInRow wall r = bricksInRow wall (r - 1) - 1 := by
  sorry

end brick_wall_theorem_l2338_233893


namespace josh_marbles_l2338_233875

def marble_problem (initial_marbles found_marbles : ℕ) : Prop :=
  initial_marbles + found_marbles = 28

theorem josh_marbles : marble_problem 21 7 := by
  sorry

end josh_marbles_l2338_233875


namespace smallest_n_equal_l2338_233892

/-- Geometric series C_n -/
def C (n : ℕ) : ℚ :=
  352 * (1 - (1/2)^n) / (1 - 1/2)

/-- Geometric series D_n -/
def D (n : ℕ) : ℚ :=
  992 * (1 - (1/(-2))^n) / (1 + 1/2)

/-- The smallest n ≥ 1 for which C_n = D_n is 1 -/
theorem smallest_n_equal (n : ℕ) (h : n ≥ 1) : (C n = D n) → n = 1 :=
by sorry

end smallest_n_equal_l2338_233892


namespace geometric_sequence_ratio_l2338_233861

/-- A geometric sequence with common ratio q -/
def geometric_sequence (q : ℝ) : ℕ → ℝ := fun n => q ^ (n - 1)

/-- The common ratio of a geometric sequence where a₄ = 27 and a₇ = -729 -/
theorem geometric_sequence_ratio : ∃ q : ℝ, 
  geometric_sequence q 4 = 27 ∧ 
  geometric_sequence q 7 = -729 ∧ 
  q = -3 := by
  sorry

end geometric_sequence_ratio_l2338_233861


namespace work_time_calculation_l2338_233814

theorem work_time_calculation (a_time b_time : ℝ) (b_fraction : ℝ) : 
  a_time = 6 →
  b_time = 3 →
  b_fraction = 1/9 →
  (1 - b_fraction) / (1 / a_time) = 16/3 :=
by sorry

end work_time_calculation_l2338_233814


namespace units_digit_of_large_product_l2338_233821

theorem units_digit_of_large_product : ∃ n : ℕ, n < 10 ∧ 2^1007 * 6^1008 * 14^1009 ≡ n [ZMOD 10] ∧ n = 2 := by
  sorry

end units_digit_of_large_product_l2338_233821


namespace necessary_but_not_sufficient_l2338_233868

theorem necessary_but_not_sufficient (a b : ℝ) :
  (∀ x y : ℝ, x * y ≠ 0 → x ≠ 0) ∧
  ¬(∀ x y : ℝ, x ≠ 0 → x * y ≠ 0) :=
sorry

end necessary_but_not_sufficient_l2338_233868


namespace polynomial_intersection_theorem_l2338_233809

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- Define the theorem
theorem polynomial_intersection_theorem (a b c d : ℝ) : 
  -- f and g are distinct polynomials
  (∃ x, f a b x ≠ g c d x) →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- The graphs intersect at (50, -200)
  f a b 50 = -200 ∧ g c d 50 = -200 →
  -- The minimum value of f is 50 less than the minimum value of g
  (-a^2/4 + b) = (-c^2/4 + d - 50) →
  -- There exists a unique value for a + c
  ∃! x, x = a + c :=
by sorry

end polynomial_intersection_theorem_l2338_233809


namespace league_games_count_l2338_233822

/-- Calculates the number of games in a round-robin tournament. -/
def numGames (n : ℕ) (k : ℕ) : ℕ := n * (n - 1) / 2 * k

/-- Proves that in a league with 20 teams, where each team plays every other team 4 times, 
    the total number of games played in the season is 760. -/
theorem league_games_count : numGames 20 4 = 760 := by
  sorry

end league_games_count_l2338_233822


namespace hundred_day_previous_year_is_saturday_l2338_233856

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek := sorry

/-- Returns true if the year is a leap year, false otherwise -/
def isLeapYear (y : Year) : Bool := sorry

/-- The number of days in a year -/
def daysInYear (y : Year) : ℕ :=
  if isLeapYear y then 366 else 365

theorem hundred_day_previous_year_is_saturday 
  (N : Year)
  (h1 : dayOfWeek N 400 = DayOfWeek.Friday)
  (h2 : dayOfWeek (Year.mk (N.value + 1)) 300 = DayOfWeek.Friday) :
  dayOfWeek (Year.mk (N.value - 1)) 100 = DayOfWeek.Saturday := by
  sorry

end hundred_day_previous_year_is_saturday_l2338_233856


namespace blue_balls_in_box_l2338_233846

theorem blue_balls_in_box (purple_balls yellow_balls min_tries : ℕ) 
  (h1 : purple_balls = 7)
  (h2 : yellow_balls = 11)
  (h3 : min_tries = 19) :
  ∃! blue_balls : ℕ, 
    blue_balls > 0 ∧ 
    purple_balls + yellow_balls + blue_balls = min_tries :=
by
  sorry

end blue_balls_in_box_l2338_233846


namespace intersection_implies_a_values_l2338_233833

theorem intersection_implies_a_values (a : ℝ) : 
  let M : Set ℝ := {5, a^2 - 3*a + 5}
  let N : Set ℝ := {1, 3}
  (M ∩ N).Nonempty → a = 1 ∨ a = 2 := by
sorry

end intersection_implies_a_values_l2338_233833


namespace equal_value_nickels_l2338_233894

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters in the first set -/
def quarters_set1 : ℕ := 30

/-- The number of nickels in the first set -/
def nickels_set1 : ℕ := 15

/-- The number of quarters in the second set -/
def quarters_set2 : ℕ := 15

theorem equal_value_nickels : 
  ∃ n : ℕ, 
    quarters_set1 * quarter_value + nickels_set1 * nickel_value = 
    quarters_set2 * quarter_value + n * nickel_value ∧ 
    n = 90 := by
  sorry

end equal_value_nickels_l2338_233894


namespace ravi_selection_probability_l2338_233863

theorem ravi_selection_probability 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 4/7)
  (h2 : p_both = 0.11428571428571428) :
  p_both / p_ram = 0.2 := by
  sorry

end ravi_selection_probability_l2338_233863


namespace function_domain_implies_m_range_l2338_233849

/-- Given a function f(x) = 1 / √(mx² + mx + 1) with domain R, 
    prove that m must be in the range [0, 4) -/
theorem function_domain_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, mx^2 + mx + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end function_domain_implies_m_range_l2338_233849


namespace max_negative_integers_in_equation_l2338_233859

theorem max_negative_integers_in_equation (a b c d : ℤ) 
  (eq : (2 : ℝ)^a + (2 : ℝ)^b = (5 : ℝ)^c + (5 : ℝ)^d) : 
  ∀ (n : ℕ), n ≤ (if a < 0 then 1 else 0) + 
              (if b < 0 then 1 else 0) + 
              (if c < 0 then 1 else 0) + 
              (if d < 0 then 1 else 0) → n = 0 :=
sorry

end max_negative_integers_in_equation_l2338_233859


namespace intersection_A_B_complement_A_in_U_l2338_233840

-- Define the universal set U
def U : Set ℝ := {x | 1 < x ∧ x < 7}

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 5} := by sorry

-- Theorem for complement of A in U
theorem complement_A_in_U : (U \ A) = {x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7)} := by sorry

end intersection_A_B_complement_A_in_U_l2338_233840


namespace xavier_yasmin_age_ratio_l2338_233818

/-- Represents a person's age -/
structure Person where
  age : ℕ

/-- Xavier is older than Yasmin -/
def xavier_older (x y : Person) : Prop :=
  x.age > y.age

/-- Xavier will be 30 in 6 years -/
def xavier_future_age (x : Person) : Prop :=
  x.age + 6 = 30

/-- The sum of Xavier and Yasmin's ages is 36 -/
def total_age (x y : Person) : Prop :=
  x.age + y.age = 36

/-- The ratio of Xavier's age to Yasmin's age is 2:1 -/
def age_ratio (x y : Person) : Prop :=
  2 * y.age = x.age

theorem xavier_yasmin_age_ratio (x y : Person) 
  (h1 : xavier_older x y) 
  (h2 : xavier_future_age x) 
  (h3 : total_age x y) : 
  age_ratio x y := by
  sorry

end xavier_yasmin_age_ratio_l2338_233818


namespace fraction_value_proof_l2338_233807

theorem fraction_value_proof (a b c : ℚ) (h1 : a = 5) (h2 : b = -3) (h3 : c = 4) :
  2 * c / (a + b) = 4 := by
  sorry

end fraction_value_proof_l2338_233807


namespace new_person_weight_l2338_233854

def group_weight_change (initial_count : ℕ) (leaving_weight : ℝ) (average_increase : ℝ) : ℝ :=
  let final_count : ℕ := initial_count
  let intermediate_count : ℕ := initial_count - 1
  (final_count : ℝ) * average_increase + leaving_weight

theorem new_person_weight 
  (initial_count : ℕ) 
  (leaving_weight : ℝ) 
  (average_increase : ℝ) 
  (h1 : initial_count = 15) 
  (h2 : leaving_weight = 90) 
  (h3 : average_increase = 3.7) : 
  group_weight_change initial_count leaving_weight average_increase = 55.5 := by
sorry

#eval group_weight_change 15 90 3.7

end new_person_weight_l2338_233854


namespace abc_inequality_and_reciprocal_sum_l2338_233873

theorem abc_inequality_and_reciprocal_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 2) : 
  a * b * c ≤ 8 / 27 ∧ 1 / a + 1 / b + 1 / c ≥ 9 / 2 := by
  sorry

end abc_inequality_and_reciprocal_sum_l2338_233873


namespace book_pages_proof_l2338_233874

/-- Proves that a book has 72 pages given the reading conditions -/
theorem book_pages_proof (total_days : ℕ) (fraction_per_day : ℚ) (extra_pages : ℕ) : 
  total_days = 3 → 
  fraction_per_day = 1/4 → 
  extra_pages = 6 → 
  (total_days : ℚ) * (fraction_per_day * (72 : ℚ) + extra_pages) = 72 := by
  sorry

#check book_pages_proof

end book_pages_proof_l2338_233874


namespace common_chord_length_l2338_233884

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y - 4 = 0
def circle_C2 (x y : ℝ) : Prop := (x + 3/2)^2 + (y - 3/2)^2 = 11/2

-- Theorem statement
theorem common_chord_length :
  ∃ (a b c d : ℝ),
    (circle_C1 a b ∧ circle_C1 c d) ∧
    (circle_C2 a b ∧ circle_C2 c d) ∧
    (a ≠ c ∨ b ≠ d) ∧
    Real.sqrt ((a - c)^2 + (b - d)^2) = 2 :=
sorry

end common_chord_length_l2338_233884


namespace bill_calculation_l2338_233827

/-- Given an initial bill amount, calculate the final amount after applying two successive late charges -/
def final_bill_amount (initial_amount : ℝ) (first_charge_rate : ℝ) (second_charge_rate : ℝ) : ℝ :=
  initial_amount * (1 + first_charge_rate) * (1 + second_charge_rate)

/-- Theorem: The final bill amount after applying late charges is $525.30 -/
theorem bill_calculation : 
  final_bill_amount 500 0.02 0.03 = 525.30 := by
  sorry

#eval final_bill_amount 500 0.02 0.03

end bill_calculation_l2338_233827


namespace polynomial_evaluation_l2338_233862

theorem polynomial_evaluation (x : ℝ) (hx_pos : x > 0) (hx_eq : x^2 - 3*x - 9 = 0) :
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = -8 := by
  sorry

end polynomial_evaluation_l2338_233862


namespace factorization_equality_l2338_233899

theorem factorization_equality (x y z : ℝ) : x^2 + x*y - x*z - y*z = (x + y)*(x - z) := by
  sorry

end factorization_equality_l2338_233899


namespace tank_capacity_l2338_233835

/-- Represents a tank with a leak and an inlet pipe -/
structure Tank where
  capacity : ℝ
  leakRate : ℝ
  inletRate : ℝ

/-- The conditions of the problem -/
def tankProblem (t : Tank) : Prop :=
  t.leakRate = t.capacity / 6 ∧ 
  t.inletRate = 3.5 * 60 ∧ 
  t.inletRate - t.leakRate = t.capacity / 8

/-- The theorem stating that under the given conditions, the tank's capacity is 720 liters -/
theorem tank_capacity (t : Tank) : tankProblem t → t.capacity = 720 := by
  sorry

end tank_capacity_l2338_233835


namespace children_count_l2338_233860

theorem children_count (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 6) (h2 : total_pencils = 12) :
  total_pencils / pencils_per_child = 2 :=
by sorry

end children_count_l2338_233860


namespace units_digit_of_sum_units_digit_of_power_units_digit_of_expression_l2338_233830

theorem units_digit_of_sum (a b : ℕ) : ∃ (x y : ℕ), 
  x = a % 10 ∧ 
  y = b % 10 ∧ 
  (a + b) % 10 = (x + y) % 10 :=
by sorry

theorem units_digit_of_power (base exp : ℕ) : 
  (base ^ exp) % 10 = (base % 10 ^ (exp % 4 + 4)) % 10 :=
by sorry

theorem units_digit_of_expression : (5^12 + 4^2) % 10 = 1 :=
by sorry

end units_digit_of_sum_units_digit_of_power_units_digit_of_expression_l2338_233830


namespace yunas_grandfather_age_l2338_233879

/-- Calculates the age of Yuna's grandfather given the ages and age differences of family members. -/
def grandfatherAge (yunaAge : ℕ) (fatherAgeDiff : ℕ) (grandfatherAgeDiff : ℕ) : ℕ :=
  yunaAge + fatherAgeDiff + grandfatherAgeDiff

/-- Proves that Yuna's grandfather is 59 years old given the provided conditions. -/
theorem yunas_grandfather_age :
  grandfatherAge 9 27 23 = 59 := by
  sorry

#eval grandfatherAge 9 27 23

end yunas_grandfather_age_l2338_233879


namespace cloth_cost_price_theorem_l2338_233887

/-- Calculates the cost price per meter of cloth given the total length,
    selling price, and profit per meter. -/
def costPricePerMeter (totalLength : ℕ) (sellingPrice : ℕ) (profitPerMeter : ℕ) : ℚ :=
  (sellingPrice - totalLength * profitPerMeter) / totalLength

/-- Theorem stating that for the given conditions, the cost price per meter is 86. -/
theorem cloth_cost_price_theorem (totalLength : ℕ) (sellingPrice : ℕ) (profitPerMeter : ℕ)
    (h1 : totalLength = 45)
    (h2 : sellingPrice = 4500)
    (h3 : profitPerMeter = 14) :
    costPricePerMeter totalLength sellingPrice profitPerMeter = 86 := by
  sorry

#eval costPricePerMeter 45 4500 14

end cloth_cost_price_theorem_l2338_233887


namespace line_segment_theorem_l2338_233853

/-- Represents a line segment on a straight line -/
structure LineSegment where
  left : ℝ
  right : ℝ
  h : left ≤ right

/-- Given a list of line segments, returns true if there exists a point common to at least n of them -/
def has_common_point (segments : List LineSegment) (n : ℕ) : Prop :=
  ∃ p : ℝ, (segments.filter (λ s => s.left ≤ p ∧ p ≤ s.right)).length ≥ n

/-- Given a list of line segments, returns true if there exist n pairwise disjoint segments -/
def has_disjoint_segments (segments : List LineSegment) (n : ℕ) : Prop :=
  ∃ disjoint : List LineSegment, disjoint.length = n ∧
    ∀ i j, i < j → j < disjoint.length →
      (disjoint.get ⟨i, by sorry⟩).right < (disjoint.get ⟨j, by sorry⟩).left

/-- The main theorem -/
theorem line_segment_theorem (segments : List LineSegment) 
    (h : segments.length = 50) :
    has_common_point segments 8 ∨ has_disjoint_segments segments 8 := by
  sorry

end line_segment_theorem_l2338_233853


namespace at_most_one_greater_than_one_l2338_233869

theorem at_most_one_greater_than_one (x y : ℝ) (h : x + y < 2) :
  ¬(x > 1 ∧ y > 1) := by
  sorry

end at_most_one_greater_than_one_l2338_233869


namespace jerry_video_games_l2338_233825

theorem jerry_video_games (initial_games new_games : ℕ) : 
  initial_games = 7 → new_games = 2 → initial_games + new_games = 9 :=
by sorry

end jerry_video_games_l2338_233825


namespace water_removal_for_concentration_l2338_233813

/-- 
Proves that the amount of water removed to concentrate a 40% acidic liquid to 60% acidic liquid 
is 5 liters, given that the final volume is 5 liters less than the initial volume.
-/
theorem water_removal_for_concentration (initial_volume : ℝ) : 
  initial_volume > 0 →
  let initial_concentration : ℝ := 0.4
  let final_concentration : ℝ := 0.6
  let volume_decrease : ℝ := 5
  let final_volume : ℝ := initial_volume - volume_decrease
  let water_removed : ℝ := volume_decrease
  initial_concentration * initial_volume = final_concentration * final_volume →
  water_removed = 5 := by
sorry

end water_removal_for_concentration_l2338_233813


namespace intersection_point_lines_parallel_line_equation_y_intercept_4_equation_l2338_233870

-- Define the lines and point M
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0
def l (x y : ℝ) : Prop := 2 * x + 4 * y - 5 = 0

-- M is the intersection point of l₁ and l₂
def M : ℝ × ℝ := (-1, 2)

-- Define the equations of the lines we want to prove
def line_parallel (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def line_y_intercept_4 (x y : ℝ) : Prop := 2 * x - y + 4 = 0

theorem intersection_point_lines (x y : ℝ) :
  l₁ x y ∧ l₂ x y ↔ (x, y) = M :=
sorry

theorem parallel_line_equation :
  ∀ x y : ℝ, (x, y) = M → line_parallel x y ∧ ∃ k : ℝ, ∀ x y : ℝ, line_parallel x y ↔ l (x + k) (y + k) :=
sorry

theorem y_intercept_4_equation :
  ∀ x y : ℝ, (x, y) = M → line_y_intercept_4 x y ∧ line_y_intercept_4 0 4 :=
sorry

end intersection_point_lines_parallel_line_equation_y_intercept_4_equation_l2338_233870


namespace expression_simplification_l2338_233864

theorem expression_simplification (p : ℝ) :
  (2 * (3 * p + 4) - 5 * p * 2)^2 + (6 - 2 / 2) * (9 * p - 12) = 16 * p^2 - 19 * p + 4 := by
  sorry

end expression_simplification_l2338_233864


namespace perfect_seventh_power_l2338_233815

theorem perfect_seventh_power (x y z : ℕ+) (h : ∃ (n : ℕ+), x^3 * y^5 * z^6 = n^7) :
  ∃ (m : ℕ+), x^5 * y^6 * z^3 = m^7 := by sorry

end perfect_seventh_power_l2338_233815


namespace quadratic_root_square_relation_l2338_233803

theorem quadratic_root_square_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = x^2) →
  b^2 = 3 * a * c + c^2 := by
  sorry

end quadratic_root_square_relation_l2338_233803


namespace condition_D_not_sufficient_condition_A_sufficient_condition_B_sufficient_condition_C_sufficient_l2338_233842

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles

-- Define similarity relation between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the four conditions
def condition_A (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β

def condition_B (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.γ = t2.γ

def condition_C (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

def condition_D (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b

-- Theorem stating that condition D is not sufficient for similarity
theorem condition_D_not_sufficient :
  ∃ t1 t2 : Triangle, condition_D t1 t2 ∧ ¬(similar t1 t2) := by sorry

-- Theorems stating that the other conditions are sufficient for similarity
theorem condition_A_sufficient (t1 t2 : Triangle) :
  condition_A t1 t2 → similar t1 t2 := by sorry

theorem condition_B_sufficient (t1 t2 : Triangle) :
  condition_B t1 t2 → similar t1 t2 := by sorry

theorem condition_C_sufficient (t1 t2 : Triangle) :
  condition_C t1 t2 → similar t1 t2 := by sorry

end condition_D_not_sufficient_condition_A_sufficient_condition_B_sufficient_condition_C_sufficient_l2338_233842


namespace taller_tree_height_taller_tree_height_proof_l2338_233828

/-- Given two trees where one is 20 feet taller than the other and their heights are in the ratio 2:3,
    the height of the taller tree is 60 feet. -/
theorem taller_tree_height : ℝ → ℝ → Prop :=
  fun h₁ h₂ => (h₁ = h₂ + 20 ∧ h₂ / h₁ = 2 / 3) → h₁ = 60

/-- Proof of the theorem -/
theorem taller_tree_height_proof : taller_tree_height 60 40 := by
  sorry

end taller_tree_height_taller_tree_height_proof_l2338_233828


namespace smallest_factorial_with_1987_zeros_l2338_233806

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The smallest natural number n such that n! ends with exactly 1987 zeros -/
def smallestFactorialWith1987Zeros : ℕ := 7960

theorem smallest_factorial_with_1987_zeros :
  (∀ m < smallestFactorialWith1987Zeros, trailingZeros m < 1987) ∧
  trailingZeros smallestFactorialWith1987Zeros = 1987 := by
  sorry

end smallest_factorial_with_1987_zeros_l2338_233806


namespace sin_A_in_special_triangle_l2338_233891

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem sin_A_in_special_triangle (t : Triangle) (h1 : t.a = 8) (h2 : t.b = 7) (h3 : t.B = 30 * π / 180) :
  Real.sin t.A = 4 / 7 := by
  sorry

end sin_A_in_special_triangle_l2338_233891


namespace largest_whole_number_less_than_150_l2338_233876

theorem largest_whole_number_less_than_150 :
  ∀ x : ℕ, x ≤ 24 ↔ 6 * x + 3 < 150 :=
by sorry

end largest_whole_number_less_than_150_l2338_233876


namespace m_3_sufficient_m_3_not_necessary_l2338_233888

/-- Represents an ellipse with equation x²/4 + y²/m = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ (x y : ℝ), x^2/4 + y^2/m = 1

/-- The focal length of an ellipse -/
def focal_length (e : Ellipse m) : ℝ := 
  sorry

/-- Theorem stating that m = 3 is sufficient for focal length 2 -/
theorem m_3_sufficient (e : Ellipse 3) : focal_length e = 2 :=
  sorry

/-- Theorem stating that m = 3 is not necessary for focal length 2 -/
theorem m_3_not_necessary : ∃ (m : ℝ), m ≠ 3 ∧ ∃ (e : Ellipse m), focal_length e = 2 :=
  sorry

end m_3_sufficient_m_3_not_necessary_l2338_233888


namespace equation_solution_l2338_233895

theorem equation_solution : ∃! x : ℝ, (1/4 : ℝ)^(2*x+8) = 16^(2*x+5) :=
  by
    use -3
    constructor
    · -- Prove that x = -3 satisfies the equation
      sorry
    · -- Prove uniqueness
      sorry

#check equation_solution

end equation_solution_l2338_233895


namespace conors_potato_chopping_l2338_233885

/-- The number of potatoes Conor can chop in a day -/
def potatoes_per_day : ℕ := sorry

/-- The number of eggplants Conor can chop in a day -/
def eggplants_per_day : ℕ := 12

/-- The number of carrots Conor can chop in a day -/
def carrots_per_day : ℕ := 9

/-- The number of days Conor works per week -/
def work_days_per_week : ℕ := 4

/-- The total number of vegetables Conor chops in a week -/
def total_vegetables_per_week : ℕ := 116

theorem conors_potato_chopping :
  potatoes_per_day = 8 ∧
  work_days_per_week * (eggplants_per_day + carrots_per_day + potatoes_per_day) = total_vegetables_per_week :=
by sorry

end conors_potato_chopping_l2338_233885


namespace clothes_cost_l2338_233851

def total_spending : ℕ := 10000

def adidas_cost : ℕ := 800

def nike_cost : ℕ := 2 * adidas_cost

def skechers_cost : ℕ := 4 * adidas_cost

def puma_cost : ℕ := nike_cost / 2

def total_sneakers_cost : ℕ := adidas_cost + nike_cost + skechers_cost + puma_cost

theorem clothes_cost : total_spending - total_sneakers_cost = 3600 := by
  sorry

end clothes_cost_l2338_233851


namespace simplify_expression_l2338_233810

theorem simplify_expression : 
  2 - (2 / (2 + Real.sqrt 5)) + (2 / (2 - Real.sqrt 5)) = 2 + 4 * Real.sqrt 5 := by
sorry

end simplify_expression_l2338_233810


namespace greatest_k_value_l2338_233857

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 117 := by
sorry

end greatest_k_value_l2338_233857


namespace lab_expense_ratio_l2338_233847

/-- Given a laboratory budget and expenses, prove the ratio of test tube cost to flask cost -/
theorem lab_expense_ratio (total_budget flask_cost remaining : ℚ) : 
  total_budget = 325 →
  flask_cost = 150 →
  remaining = 25 →
  ∃ (test_tube_cost : ℚ),
    total_budget = flask_cost + test_tube_cost + (test_tube_cost / 2) + remaining →
    test_tube_cost / flask_cost = 2 / 3 := by
  sorry


end lab_expense_ratio_l2338_233847


namespace root_sum_fraction_l2338_233890

theorem root_sum_fraction (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 →
  q^3 - 8*q^2 + 10*q - 3 = 0 →
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 59/22 := by
  sorry

end root_sum_fraction_l2338_233890


namespace cubic_equation_solution_l2338_233871

theorem cubic_equation_solution :
  ∃! x : ℝ, (x^3 - 5*x^2 + 5*x - 1) + (x - 1) = 0 :=
by
  -- The unique solution is x = 2
  use 2
  sorry

end cubic_equation_solution_l2338_233871


namespace specific_trapezoid_area_l2338_233811

/-- A trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of BM, where M is the point where the circle touches AB -/
  bm : ℝ
  /-- The length of the top side CD -/
  cd : ℝ

/-- The area of a trapezoid with an inscribed circle -/
def trapezoidArea (t : InscribedCircleTrapezoid) : ℝ :=
  sorry

/-- Theorem: The area of the specific trapezoid is 108 -/
theorem specific_trapezoid_area :
  ∃ t : InscribedCircleTrapezoid, t.r = 4 ∧ t.bm = 16 ∧ t.cd = 3 ∧ trapezoidArea t = 108 := by
  sorry

end specific_trapezoid_area_l2338_233811


namespace triangle_properties_l2338_233865

noncomputable section

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The triangle satisfies the given condition -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a + 2 * t.c = t.b * Real.cos t.C + Real.sqrt 3 * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (t.b = 3 → 
    6 < t.a + t.b + t.c ∧ 
    t.a + t.b + t.c ≤ 3 + 2 * Real.sqrt 3) := by
  sorry

end

end triangle_properties_l2338_233865


namespace eddie_number_l2338_233845

theorem eddie_number (n : ℕ) (m : ℕ) (h1 : n ≥ 40) (h2 : n % 5 = 0) (h3 : n % m = 0) :
  (∀ k : ℕ, k ≥ 40 ∧ k % 5 = 0 ∧ ∃ j : ℕ, k % j = 0 → k ≥ n) →
  n = 40 ∧ m = 2 := by
sorry

end eddie_number_l2338_233845


namespace angle_measure_l2338_233843

theorem angle_measure : ∃ x : ℝ, 
  (x + (5 * x + 12) = 180) ∧ x = 28 := by
  sorry

end angle_measure_l2338_233843


namespace range_of_a_l2338_233878

-- Define the conditions P and Q
def P (a : ℝ) : Prop := ∀ x y : ℝ, ∃ k : ℝ, k > 0 ∧ x^2 / (3 - a) + y^2 / (1 + a) = k

def Q (a : ℝ) : Prop := ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

-- Theorem statement
theorem range_of_a (a : ℝ) (hP : P a) (hQ : Q a) : 
  -1 < a ∧ a ≤ 2 ∧ a ≠ 1 :=
sorry

end range_of_a_l2338_233878


namespace grid_intersection_sum_zero_l2338_233824

/-- Represents a cell in the grid -/
inductive CellValue
  | Plus : CellValue
  | Minus : CellValue
  | Zero : CellValue

/-- Represents the grid -/
def Grid := Matrix (Fin 1980) (Fin 1981) CellValue

/-- The sum of all numbers in the grid is zero -/
def sumIsZero (g : Grid) : Prop := sorry

/-- The sum of four numbers at the intersections of two rows and two columns -/
def intersectionSum (g : Grid) (r1 r2 : Fin 1980) (c1 c2 : Fin 1981) : Int := sorry

theorem grid_intersection_sum_zero (g : Grid) (h : sumIsZero g) :
  ∃ (r1 r2 : Fin 1980) (c1 c2 : Fin 1981), intersectionSum g r1 r2 c1 c2 = 0 := by
  sorry

end grid_intersection_sum_zero_l2338_233824


namespace calculation_1_l2338_233836

theorem calculation_1 : (1 * (-1/9)) - (1/2) = -11/18 := by sorry

end calculation_1_l2338_233836


namespace razorback_shop_revenue_l2338_233858

theorem razorback_shop_revenue :
  let tshirt_price : ℕ := 98
  let hat_price : ℕ := 45
  let scarf_price : ℕ := 60
  let tshirts_sold : ℕ := 42
  let hats_sold : ℕ := 32
  let scarves_sold : ℕ := 15
  tshirt_price * tshirts_sold + hat_price * hats_sold + scarf_price * scarves_sold = 6456 :=
by sorry

end razorback_shop_revenue_l2338_233858


namespace home_learning_percentage_l2338_233855

-- Define the percentage of students present in school
def students_present : ℝ := 30

-- Define the theorem
theorem home_learning_percentage :
  let total_percentage : ℝ := 100
  let non_home_learning : ℝ := 2 * students_present
  let home_learning : ℝ := total_percentage - non_home_learning
  home_learning = 40 := by sorry

end home_learning_percentage_l2338_233855


namespace triangle_inequality_l2338_233880

theorem triangle_inequality (a b c : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = 1) (h8 : n ≥ 2) :
  (a^n + b^n)^(1/n) + (b^n + c^n)^(1/n) + (c^n + a^n)^(1/n) < 1 + (2^(1/n))/2 := by
sorry

end triangle_inequality_l2338_233880


namespace power_of_64_equals_128_l2338_233841

theorem power_of_64_equals_128 : (64 : ℝ) ^ (7/6) = 128 := by
  have h : 64 = 2^6 := by sorry
  sorry

end power_of_64_equals_128_l2338_233841


namespace min_value_quadratic_l2338_233819

theorem min_value_quadratic (x : ℝ) : 
  ∃ (min_y : ℝ), ∀ (y : ℝ), y = 5*x^2 + 10*x + 15 → y ≥ min_y ∧ ∃ (x₀ : ℝ), 5*x₀^2 + 10*x₀ + 15 = min_y :=
by
  sorry

end min_value_quadratic_l2338_233819


namespace same_heads_probability_l2338_233848

-- Define the probability of getting a specific number of heads when tossing two coins
def prob_heads (n : Nat) : ℚ :=
  if n = 0 then 1/4
  else if n = 1 then 1/2
  else if n = 2 then 1/4
  else 0

-- Define the probability of both people getting the same number of heads
def prob_same_heads : ℚ :=
  (prob_heads 0)^2 + (prob_heads 1)^2 + (prob_heads 2)^2

-- Theorem statement
theorem same_heads_probability : prob_same_heads = 3/8 := by
  sorry

end same_heads_probability_l2338_233848


namespace reaching_penglai_sufficient_for_immortal_l2338_233801

/-- Reaching Penglai implies becoming an immortal -/
def reaching_penglai_implies_immortal (reaching_penglai becoming_immortal : Prop) : Prop :=
  reaching_penglai → becoming_immortal

/-- Not reaching Penglai implies not becoming an immortal -/
axiom not_reaching_penglai_implies_not_immortal {reaching_penglai becoming_immortal : Prop} :
  ¬reaching_penglai → ¬becoming_immortal

/-- Prove that reaching Penglai is a sufficient condition for becoming an immortal -/
theorem reaching_penglai_sufficient_for_immortal
  {reaching_penglai becoming_immortal : Prop}
  (h : ¬reaching_penglai → ¬becoming_immortal) :
  reaching_penglai_implies_immortal reaching_penglai becoming_immortal :=
by sorry

end reaching_penglai_sufficient_for_immortal_l2338_233801


namespace chemists_self_receipts_l2338_233882

/-- Represents a chemist in the laboratory -/
structure Chemist where
  id : Nat
  reagents : Finset Nat

/-- Represents the state of the laboratory -/
structure Laboratory where
  chemists : Finset Chemist
  num_chemists : Nat

/-- Checks if a chemist has received all reagents -/
def has_all_reagents (c : Chemist) (lab : Laboratory) : Prop :=
  c.reagents.card = lab.num_chemists

/-- Checks if no chemist has received any reagent more than once -/
def no_double_receipts (lab : Laboratory) : Prop :=
  ∀ c ∈ lab.chemists, ∀ r ∈ c.reagents, (c.reagents.filter (λ x => x = r)).card ≤ 1

/-- Counts the number of chemists who received their own reagent -/
def count_self_receipts (lab : Laboratory) : Nat :=
  (lab.chemists.filter (λ c => c.id ∈ c.reagents)).card

/-- The main theorem to be proved -/
theorem chemists_self_receipts (lab : Laboratory) 
  (h1 : ∀ c ∈ lab.chemists, has_all_reagents c lab)
  (h2 : no_double_receipts lab) :
  count_self_receipts lab ≥ lab.num_chemists - 1 :=
sorry

end chemists_self_receipts_l2338_233882


namespace inscribed_circle_radius_l2338_233804

theorem inscribed_circle_radius (PQ PR QR : ℝ) (h_PQ : PQ = 30) (h_PR : PR = 26) (h_QR : QR = 28) :
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  area / s = 8 := by sorry

end inscribed_circle_radius_l2338_233804


namespace right_to_left_evaluation_l2338_233816

-- Define a custom operation that evaluates from right to left
noncomputable def rightToLeftEval (a b c d : ℝ) : ℝ :=
  a * (b / (c + d^2))

-- Theorem statement
theorem right_to_left_evaluation (a b c d : ℝ) :
  rightToLeftEval a b c d = (a * b) / (c + d^2) :=
by sorry

end right_to_left_evaluation_l2338_233816


namespace blue_jelly_bean_probability_l2338_233844

/-- The probability of selecting a blue jelly bean from a bag -/
theorem blue_jelly_bean_probability :
  let red : ℕ := 5
  let green : ℕ := 6
  let yellow : ℕ := 7
  let blue : ℕ := 8
  let total : ℕ := red + green + yellow + blue
  (blue : ℚ) / total = 4 / 13 := by
  sorry

end blue_jelly_bean_probability_l2338_233844


namespace james_work_hours_l2338_233820

/-- Calculates the number of hours James works at his main job --/
theorem james_work_hours (main_rate : ℝ) (second_rate_reduction : ℝ) (total_earnings : ℝ) :
  main_rate = 20 →
  second_rate_reduction = 0.2 →
  total_earnings = 840 →
  ∃ h : ℝ, h = 30 ∧ 
    main_rate * h + (main_rate * (1 - second_rate_reduction)) * (h / 2) = total_earnings :=
by sorry

end james_work_hours_l2338_233820


namespace inequality_system_solution_l2338_233834

def inequality_system (x : ℝ) : Prop :=
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4

theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ -2 < x ∧ x ≤ 1 :=
by sorry

end inequality_system_solution_l2338_233834


namespace max_book_price_l2338_233889

theorem max_book_price (total_money : ℕ) (num_books : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) :
  total_money = 200 →
  num_books = 20 →
  entrance_fee = 5 →
  tax_rate = 7 / 100 →
  ∃ (max_price : ℕ),
    (max_price ≤ (total_money - entrance_fee) / (num_books * (1 + tax_rate))) ∧
    (∀ (price : ℕ), price > max_price →
      price * num_books * (1 + tax_rate) > (total_money - entrance_fee)) ∧
    max_price = 9 :=
by sorry

end max_book_price_l2338_233889


namespace equation_solutions_l2338_233817

theorem equation_solutions : 
  ∀ x : ℝ, (x^2 + x)^2 + (x^2 + x) - 6 = 0 ↔ x = -2 ∨ x = 1 := by
  sorry

end equation_solutions_l2338_233817


namespace parabola_line_intersection_l2338_233829

/-- Theorem stating the relationship between k, a, m, and n for a parabola and line intersection -/
theorem parabola_line_intersection
  (a m n k b : ℝ)
  (ha : a ≠ 0)
  (h_intersect : ∃ (y₁ y₂ : ℝ),
    a * (1 - m) * (1 - n) = k * 1 + b ∧
    a * (6 - m) * (6 - n) = k * 6 + b) :
  k = a * (7 - m - n) := by
  sorry

end parabola_line_intersection_l2338_233829


namespace geometric_sequence_properties_l2338_233805

/-- For non-zero real numbers a, b, c, if they form a geometric sequence,
    then their reciprocals and their squares also form geometric sequences. -/
theorem geometric_sequence_properties (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
    (h_geometric : b^2 = a * c) : 
  (1 / b)^2 = (1 / a) * (1 / c) ∧ (b^2)^2 = a^2 * c^2 := by
  sorry


end geometric_sequence_properties_l2338_233805


namespace unique_solution_l2338_233897

/-- The number of positive integer solutions to the equation 2x + 3y = 8 -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ 2 * p.1 + 3 * p.2 = 8) (Finset.product (Finset.range 9) (Finset.range 9))).card

/-- Theorem stating that there is exactly one positive integer solution to 2x + 3y = 8 -/
theorem unique_solution : solution_count = 1 := by
  sorry

end unique_solution_l2338_233897


namespace buttons_problem_l2338_233850

theorem buttons_problem (sue kendra mari : ℕ) : 
  sue = kendra / 2 →
  sue = 6 →
  mari = 5 * kendra + 4 →
  mari = 64 := by
sorry

end buttons_problem_l2338_233850


namespace quadratic_root_sum_l2338_233898

theorem quadratic_root_sum (a b : ℝ) : 
  (1 : ℝ) ^ 2 * a + 1 * b - 3 = 0 → a + b = 3 := by
sorry

end quadratic_root_sum_l2338_233898


namespace not_in_range_iff_b_in_interval_l2338_233872

/-- The function f(x) defined as x^2 + bx + 5 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 5

/-- Theorem stating that -3 is not in the range of f(x) if and only if b is in the open interval (-4√2, 4√2) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, f b x ≠ -3) ↔ b ∈ Set.Ioo (-4 * Real.sqrt 2) (4 * Real.sqrt 2) :=
sorry

end not_in_range_iff_b_in_interval_l2338_233872


namespace division_of_composite_products_l2338_233886

-- Define the first six positive composite integers
def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]

-- Define the product of the first three composite integers
def product_first_three : Nat := (first_six_composites.take 3).prod

-- Define the product of the next three composite integers
def product_next_three : Nat := (first_six_composites.drop 3).prod

-- Theorem to prove
theorem division_of_composite_products :
  (product_first_three : ℚ) / product_next_three = 8 / 45 := by
  sorry

end division_of_composite_products_l2338_233886


namespace largest_domain_of_g_l2338_233812

/-- A function g satisfying the given property -/
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → g x + g (1 / x^2) = x^2

/-- The domain of g is the set of all non-zero real numbers -/
theorem largest_domain_of_g :
  ∃ g : ℝ → ℝ, g_property g ∧
  ∀ S : Set ℝ, (∃ h : ℝ → ℝ, g_property h ∧ ∀ x ∈ S, h x ≠ 0) →
  S ⊆ {x : ℝ | x ≠ 0} :=
sorry

end largest_domain_of_g_l2338_233812


namespace ellipse_k_value_l2338_233831

/-- Theorem: For an ellipse with equation 5x^2 - ky^2 = 5 and one focus at (0, 2), the value of k is -1. -/
theorem ellipse_k_value (k : ℝ) : 
  (∃ (x y : ℝ), 5 * x^2 - k * y^2 = 5) → -- Ellipse equation
  (∃ (c : ℝ), c = 2 ∧ c^2 = 5 - (-5/k)) → -- Focus at (0, 2) and standard form relation
  k = -1 := by
sorry

end ellipse_k_value_l2338_233831


namespace f_min_value_l2338_233826

/-- The quadratic function f(x) = 5x^2 - 15x - 2 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 15 * x - 2

/-- The minimum value of f(x) is -13.25 -/
theorem f_min_value : ∃ (x : ℝ), f x = -13.25 ∧ ∀ (y : ℝ), f y ≥ -13.25 :=
sorry

end f_min_value_l2338_233826


namespace square_difference_39_40_square_41_from_40_l2338_233808

theorem square_difference_39_40 :
  (40 : ℕ)^2 - (39 : ℕ)^2 = 79 :=
by
  sorry

-- Additional theorem to represent the given condition
theorem square_41_from_40 :
  (41 : ℕ)^2 = (40 : ℕ)^2 + 81 :=
by
  sorry

end square_difference_39_40_square_41_from_40_l2338_233808


namespace distance_traveled_l2338_233881

theorem distance_traveled (speed1 speed2 distance_diff : ℝ) (h1 : speed1 = 10)
  (h2 : speed2 = 20) (h3 : distance_diff = 40) :
  let time := distance_diff / (speed2 - speed1)
  let actual_distance := speed1 * time
  actual_distance = 40 :=
by sorry

end distance_traveled_l2338_233881


namespace delivery_problem_l2338_233877

theorem delivery_problem (total_bottles : ℕ) (cider_bottles : ℕ) (beer_bottles : ℕ) 
  (h1 : total_bottles = 180)
  (h2 : cider_bottles = 40)
  (h3 : beer_bottles = 80)
  (h4 : cider_bottles + beer_bottles < total_bottles) :
  (cider_bottles / 2) + (beer_bottles / 2) + ((total_bottles - cider_bottles - beer_bottles) / 2) = 90 := by
  sorry

end delivery_problem_l2338_233877
