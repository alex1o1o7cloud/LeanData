import Mathlib

namespace NUMINAMATH_CALUDE_smaller_square_area_percentage_l1525_152593

/-- A circle with an inscribed square and a smaller square -/
structure CircleWithSquares where
  /-- Radius of the circle -/
  r : ℝ
  /-- Side length of the larger square (inscribed in the circle) -/
  large_side : ℝ
  /-- Side length of the smaller square -/
  small_side : ℝ
  /-- The larger square is inscribed in the circle -/
  large_inscribed : large_side = 2 * r
  /-- The smaller square shares one side with the larger square -/
  shared_side : small_side ≤ large_side
  /-- Two vertices of the smaller square are on the circle -/
  vertices_on_circle : small_side^2 + (large_side/2 + small_side/2)^2 = r^2

/-- The area of the smaller square is 0% of the area of the larger square -/
theorem smaller_square_area_percentage (c : CircleWithSquares) :
  (c.small_side^2) / (c.large_side^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_percentage_l1525_152593


namespace NUMINAMATH_CALUDE_parabola_chord_length_l1525_152577

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a parabola x^2 = 2py -/
def Parabola (p : ℝ) : Type :=
  {point : ParabolaPoint // point.x^2 = 2 * p * point.y}

theorem parabola_chord_length (p : ℝ) (h_p : p > 0) :
  ∀ (A B : Parabola p),
    (A.val.y + B.val.y) / 2 = 3 →
    (∀ C D : Parabola p, |C.val.y - D.val.y + p| ≤ 8) →
    (∃ E F : Parabola p, |E.val.y - F.val.y + p| = 8) →
    p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l1525_152577


namespace NUMINAMATH_CALUDE_arc_length_radius_l1525_152560

/-- Given an arc length and central angle, calculate the radius of the circle. -/
theorem arc_length_radius (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = 2) :
  s / θ = 2 := by sorry

end NUMINAMATH_CALUDE_arc_length_radius_l1525_152560


namespace NUMINAMATH_CALUDE_geometric_sum_specific_l1525_152598

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_specific :
  geometric_sum (3/4) (3/4) 12 = 48758625/16777216 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_specific_l1525_152598


namespace NUMINAMATH_CALUDE_product_of_decimals_l1525_152535

theorem product_of_decimals : 3.6 * 0.04 = 0.144 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1525_152535


namespace NUMINAMATH_CALUDE_birds_to_asia_count_l1525_152576

/-- The number of bird families that flew away to Asia -/
def birds_to_asia : ℕ := sorry

/-- The number of bird families living near the mountain -/
def birds_near_mountain : ℕ := 38

/-- The number of bird families that flew away to Africa -/
def birds_to_africa : ℕ := 47

theorem birds_to_asia_count : birds_to_asia = 94 := by
  sorry

end NUMINAMATH_CALUDE_birds_to_asia_count_l1525_152576


namespace NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l1525_152528

theorem sphere_cylinder_equal_area (r : ℝ) : 
  (4 * Real.pi * r^2 = 2 * Real.pi * 6 * 12) → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l1525_152528


namespace NUMINAMATH_CALUDE_book_sale_revenue_l1525_152563

theorem book_sale_revenue (total_books : ℕ) (sold_books : ℕ) (price_per_book : ℕ) 
  (h1 : sold_books = (2 * total_books) / 3)
  (h2 : total_books - sold_books = 36)
  (h3 : price_per_book = 4) :
  sold_books * price_per_book = 288 := by
sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l1525_152563


namespace NUMINAMATH_CALUDE_bus_fare_cost_l1525_152595

/-- Represents the cost of a bus fare for one person one way -/
def bus_fare : ℝ := 1.5

/-- Represents the cost of zoo entry for one person -/
def zoo_entry : ℝ := 5

/-- Represents the total money brought -/
def total_money : ℝ := 40

/-- Represents the money left after zoo entry and bus fare -/
def money_left : ℝ := 24

/-- Proves that the bus fare cost per person one way is $1.50 -/
theorem bus_fare_cost : 
  2 * zoo_entry + 4 * bus_fare = total_money - money_left :=
by sorry

end NUMINAMATH_CALUDE_bus_fare_cost_l1525_152595


namespace NUMINAMATH_CALUDE_problem_solution_l1525_152556

def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 8 = 0}
def C (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem problem_solution :
  (∀ a : ℝ, A a = B → a = 2) ∧
  (∀ m : ℝ, B ∪ C m = B → m = -1/4 ∨ m = 0 ∨ m = 1/2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1525_152556


namespace NUMINAMATH_CALUDE_shirts_purchased_l1525_152559

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_cost : ℕ := 51
def num_jeans : ℕ := 2
def num_hats : ℕ := 4

theorem shirts_purchased : 
  ∃ (num_shirts : ℕ), 
    num_shirts * shirt_cost + num_jeans * jeans_cost + num_hats * hat_cost = total_cost ∧ 
    num_shirts = 3 := by
  sorry

end NUMINAMATH_CALUDE_shirts_purchased_l1525_152559


namespace NUMINAMATH_CALUDE_square_area_l1525_152549

/-- The area of a square with side length 11 cm is 121 cm². -/
theorem square_area (side_length : ℝ) (h : side_length = 11) : side_length ^ 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l1525_152549


namespace NUMINAMATH_CALUDE_complex_exponential_identity_l1525_152592

theorem complex_exponential_identity :
  (Complex.exp (Complex.I * Real.pi * (125 / 180)))^40 = Complex.exp (Complex.I * Real.pi * (40 / 180) * (-1)) :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_identity_l1525_152592


namespace NUMINAMATH_CALUDE_meetings_percentage_of_workday_l1525_152530

-- Define the work day in minutes
def work_day_minutes : ℕ := 8 * 60

-- Define the durations of the meetings
def meeting1_duration : ℕ := 30
def meeting2_duration : ℕ := 60
def meeting3_duration : ℕ := meeting1_duration + meeting2_duration

-- Define the total meeting time
def total_meeting_time : ℕ := meeting1_duration + meeting2_duration + meeting3_duration

-- Theorem to prove
theorem meetings_percentage_of_workday :
  (total_meeting_time : ℚ) / work_day_minutes * 100 = 37.5 := by
  sorry


end NUMINAMATH_CALUDE_meetings_percentage_of_workday_l1525_152530


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_12_mod_19_l1525_152594

theorem largest_four_digit_congruent_to_12_mod_19 : ∀ n : ℕ,
  n < 10000 → n ≡ 12 [ZMOD 19] → n ≤ 9987 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_12_mod_19_l1525_152594


namespace NUMINAMATH_CALUDE_w_squared_value_l1525_152553

theorem w_squared_value (w : ℚ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l1525_152553


namespace NUMINAMATH_CALUDE_rain_probability_jan20_l1525_152558

-- Define the initial probability and the number of days
def initial_prob : ℚ := 1/2
def days : ℕ := 5

-- Define the daily probability adjustment factors
def factor1 : ℚ := 2017/2016
def factor2 : ℚ := 1007/2016

-- Define the function to calculate the probability after n days
def prob_after_n_days (n : ℕ) : ℚ :=
  initial_prob * (((factor1 + factor2) / 2) ^ n)

-- The theorem to prove
theorem rain_probability_jan20 :
  prob_after_n_days days = 243/2048 := by
  sorry


end NUMINAMATH_CALUDE_rain_probability_jan20_l1525_152558


namespace NUMINAMATH_CALUDE_point_movement_l1525_152584

def number_line_move (start : ℤ) (move : ℤ) : ℤ :=
  start + move

theorem point_movement :
  let point_A : ℤ := -3
  let movement : ℤ := 6
  let point_B : ℤ := number_line_move point_A movement
  point_B = 3 := by sorry

end NUMINAMATH_CALUDE_point_movement_l1525_152584


namespace NUMINAMATH_CALUDE_square_25_solutions_l1525_152562

theorem square_25_solutions (x : ℝ) (h : x^2 = 25) :
  (∃ y : ℝ, y^2 = 25 ∧ y ≠ x) ∧
  (∀ z : ℝ, z^2 = 25 → z = x ∨ z = -x) ∧
  x + (-x) = 0 ∧
  x * (-x) = -25 := by
sorry

end NUMINAMATH_CALUDE_square_25_solutions_l1525_152562


namespace NUMINAMATH_CALUDE_total_leaves_l1525_152509

/-- The number of ferns Karen hangs around her house. -/
def num_ferns : ℕ := 12

/-- The number of fronds each fern has. -/
def fronds_per_fern : ℕ := 15

/-- The number of leaves each frond has. -/
def leaves_per_frond : ℕ := 45

/-- Theorem stating the total number of leaves on all ferns. -/
theorem total_leaves : num_ferns * fronds_per_fern * leaves_per_frond = 8100 := by
  sorry

end NUMINAMATH_CALUDE_total_leaves_l1525_152509


namespace NUMINAMATH_CALUDE_cousins_money_correct_l1525_152527

/-- The amount of money Jim's cousin brought to the restaurant --/
def cousins_money : ℝ := 10

/-- The total cost of the meal --/
def meal_cost : ℝ := 24

/-- The percentage of their combined money spent on the meal --/
def spent_percentage : ℝ := 0.8

/-- The amount of money Jim brought --/
def jims_money : ℝ := 20

/-- Theorem stating that the calculated amount Jim's cousin brought is correct --/
theorem cousins_money_correct : 
  cousins_money = (meal_cost / spent_percentage) - jims_money := by
  sorry

end NUMINAMATH_CALUDE_cousins_money_correct_l1525_152527


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l1525_152580

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 16) 
  (h2 : x * y = 6) : 
  x^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l1525_152580


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1525_152541

theorem min_value_of_expression :
  (∀ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 ≥ 2022) ∧
  (∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 = 2022) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1525_152541


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l1525_152586

theorem smallest_multiple_of_45_and_75_not_20 : 
  ∃ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 75 ∣ n ∧ ¬(20 ∣ n) ∧ 
  ∀ m : ℕ, m > 0 → 45 ∣ m → 75 ∣ m → ¬(20 ∣ m) → n ≤ m :=
by
  use 225
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l1525_152586


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1525_152515

theorem polynomial_division_remainder 
  (dividend : Polynomial ℚ) 
  (divisor : Polynomial ℚ) 
  (quotient : Polynomial ℚ) 
  (remainder : Polynomial ℚ) :
  dividend = 3 * X^4 + 7 * X^3 - 28 * X^2 - 32 * X + 53 →
  divisor = X^2 + 5 * X + 3 →
  dividend = divisor * quotient + remainder →
  Polynomial.degree remainder < Polynomial.degree divisor →
  remainder = 97 * X + 116 := by
    sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1525_152515


namespace NUMINAMATH_CALUDE_distinct_outfits_l1525_152504

theorem distinct_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (hats : ℕ) 
  (h_shirts : shirts = 7)
  (h_pants : pants = 5)
  (h_ties : ties = 6)
  (h_hats : hats = 4) :
  shirts * pants * (ties + 1) * (hats + 1) = 1225 := by
  sorry

end NUMINAMATH_CALUDE_distinct_outfits_l1525_152504


namespace NUMINAMATH_CALUDE_total_fruits_is_54_l1525_152574

/-- The total number of fruits picked by all people -/
def total_fruits (melanie_plums melanie_apples dan_plums dan_oranges sally_plums sally_cherries thomas_plums thomas_peaches : ℕ) : ℕ :=
  melanie_plums + melanie_apples + dan_plums + dan_oranges + sally_plums + sally_cherries + thomas_plums + thomas_peaches

/-- Theorem stating that the total number of fruits picked is 54 -/
theorem total_fruits_is_54 :
  total_fruits 4 6 9 2 3 10 15 5 = 54 := by
  sorry

#eval total_fruits 4 6 9 2 3 10 15 5

end NUMINAMATH_CALUDE_total_fruits_is_54_l1525_152574


namespace NUMINAMATH_CALUDE_sequence_problem_l1525_152566

/-- Given a sequence a and a geometric sequence b, prove that a_2016 = 1 -/
theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  a 1 = 1 →
  (∀ n, b n = a (n + 1) / a n) →
  (∀ n, b (n + 1) / b n = b 2 / b 1) →
  b 1008 = 1 →
  a 2016 = 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1525_152566


namespace NUMINAMATH_CALUDE_kim_shirts_l1525_152568

theorem kim_shirts (D : ℕ) : 
  (2 / 3 : ℚ) * (12 * D) = 32 → D = 4 := by
  sorry

end NUMINAMATH_CALUDE_kim_shirts_l1525_152568


namespace NUMINAMATH_CALUDE_contest_age_fraction_l1525_152585

theorem contest_age_fraction (total_participants : ℕ) (F : ℚ) : 
  total_participants = 500 →
  (F + F / 8 : ℚ) = 0.5625 →
  F = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_contest_age_fraction_l1525_152585


namespace NUMINAMATH_CALUDE_math_club_team_selection_l1525_152599

theorem math_club_team_selection (boys girls team_size : ℕ) 
  (h_boys : boys = 10) 
  (h_girls : girls = 12) 
  (h_team_size : team_size = 8) : 
  (Nat.choose (boys + girls) team_size) - 
  (Nat.choose girls team_size) - 
  (Nat.choose boys team_size) = 319230 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l1525_152599


namespace NUMINAMATH_CALUDE_polynomial_always_positive_l1525_152529

theorem polynomial_always_positive (x : ℝ) : x^12 - x^9 + x^4 - x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_always_positive_l1525_152529


namespace NUMINAMATH_CALUDE_sum_of_base6_series_l1525_152589

/-- Represents a number in base 6 -/
def Base6 := Nat

/-- Converts a base 6 number to decimal -/
def to_decimal (n : Base6) : Nat :=
  sorry

/-- Converts a decimal number to base 6 -/
def to_base6 (n : Nat) : Base6 :=
  sorry

/-- The sum of an arithmetic series in base 6 -/
def arithmetic_sum_base6 (first last : Base6) (common_diff : Base6) : Base6 :=
  sorry

/-- Theorem: The sum of the series 2₆ + 4₆ + 6₆ + ⋯ + 100₆ in base 6 is 1330₆ -/
theorem sum_of_base6_series : 
  arithmetic_sum_base6 (to_base6 2) (to_base6 36) (to_base6 2) = to_base6 342 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_base6_series_l1525_152589


namespace NUMINAMATH_CALUDE_tan_value_on_sqrt_graph_l1525_152512

/-- If the point (4, a) is on the graph of y = x^(1/2), then tan(aπ/6) = √3 -/
theorem tan_value_on_sqrt_graph (a : ℝ) : 
  a = 4^(1/2) → Real.tan (a * π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_on_sqrt_graph_l1525_152512


namespace NUMINAMATH_CALUDE_stating_italian_regular_clock_coincidences_l1525_152516

/-- Represents a clock with specified rotations for hour and minute hands per day. -/
structure Clock :=
  (hour_rotations : ℕ)
  (minute_rotations : ℕ)

/-- The Italian clock with 1 hour hand rotation and 24 minute hand rotations per day. -/
def italian_clock : Clock :=
  { hour_rotations := 1, minute_rotations := 24 }

/-- The regular clock with 2 hour hand rotations and 24 minute hand rotations per day. -/
def regular_clock : Clock :=
  { hour_rotations := 2, minute_rotations := 24 }

/-- 
  The number of times the hands of an Italian clock coincide with 
  the hands of a regular clock in a 24-hour period.
-/
def coincidence_count (ic : Clock) (rc : Clock) : ℕ := sorry

/-- 
  Theorem stating that the number of hand coincidences between 
  the Italian clock and regular clock in a 24-hour period is 12.
-/
theorem italian_regular_clock_coincidences : 
  coincidence_count italian_clock regular_clock = 12 := by sorry

end NUMINAMATH_CALUDE_stating_italian_regular_clock_coincidences_l1525_152516


namespace NUMINAMATH_CALUDE_expression_equality_l1525_152524

theorem expression_equality : 
  (2^3 ≠ 3^2) ∧ 
  (-2^3 = (-2)^3) ∧ 
  ((-2 * 3)^2 ≠ -2 * 3^2) ∧ 
  ((-5)^2 ≠ -5^2) :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l1525_152524


namespace NUMINAMATH_CALUDE_no_roots_implies_not_integer_l1525_152520

theorem no_roots_implies_not_integer (a b : ℝ) (h1 : a ≠ b)
  (h2 : ∀ x : ℝ, (x^2 + 20*a*x + 10*b) * (x^2 + 20*b*x + 10*a) ≠ 0) :
  ¬ ∃ n : ℤ, 20*(b - a) = n := by
  sorry

end NUMINAMATH_CALUDE_no_roots_implies_not_integer_l1525_152520


namespace NUMINAMATH_CALUDE_factorization_bound_l1525_152591

/-- The number of ways to factorize k into a product of integers greater than 1 -/
def f (k : ℕ) : ℕ :=
  sorry

/-- Theorem: For any integer n > 1 and any prime factor p of n,
    the number of ways to factorize n is less than or equal to n/p -/
theorem factorization_bound (n : ℕ) (p : ℕ) (h1 : n > 1) (h2 : Nat.Prime p) (h3 : p ∣ n) :
  f n ≤ n / p :=
sorry

end NUMINAMATH_CALUDE_factorization_bound_l1525_152591


namespace NUMINAMATH_CALUDE_four_valid_numbers_l1525_152567

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧
  (∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
    ((a = 1 ∧ b = 9 ∧ c = 8 ∧ d = 8) ∨
     (a = 1 ∧ b = 8 ∧ c = 9 ∧ d = 8) ∨
     (a = 1 ∧ b = 8 ∧ c = 8 ∧ d = 9) ∨
     (a = 9 ∧ b = 1 ∧ c = 8 ∧ d = 8) ∨
     (a = 9 ∧ b = 8 ∧ c = 1 ∧ d = 8) ∨
     (a = 9 ∧ b = 8 ∧ c = 8 ∧ d = 1) ∨
     (a = 8 ∧ b = 1 ∧ c = 9 ∧ d = 8) ∨
     (a = 8 ∧ b = 1 ∧ c = 8 ∧ d = 9) ∨
     (a = 8 ∧ b = 9 ∧ c = 1 ∧ d = 8) ∨
     (a = 8 ∧ b = 9 ∧ c = 8 ∧ d = 1) ∨
     (a = 8 ∧ b = 8 ∧ c = 1 ∧ d = 9) ∨
     (a = 8 ∧ b = 8 ∧ c = 9 ∧ d = 1))) ∧
  n % 11 = 8

theorem four_valid_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_valid_number n) ∧ s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_four_valid_numbers_l1525_152567


namespace NUMINAMATH_CALUDE_david_average_speed_l1525_152539

/-- Calculates the average speed given distance and time -/
def average_speed (distance : ℚ) (time : ℚ) : ℚ :=
  distance / time

/-- Converts hours and minutes to hours -/
def hours_and_minutes_to_hours (hours : ℕ) (minutes : ℕ) : ℚ :=
  hours + (minutes : ℚ) / 60

theorem david_average_speed :
  let distance : ℚ := 49/3  -- 16 1/3 miles as an improper fraction
  let time : ℚ := hours_and_minutes_to_hours 2 20
  average_speed distance time = 7 := by sorry

end NUMINAMATH_CALUDE_david_average_speed_l1525_152539


namespace NUMINAMATH_CALUDE_fraction_decomposition_l1525_152517

theorem fraction_decomposition (x A B : ℝ) : 
  (8 * x - 17) / (3 * x^2 + 4 * x - 15) = A / (3 * x + 5) + B / (x - 3) →
  A = 6.5 ∧ B = 0.5 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l1525_152517


namespace NUMINAMATH_CALUDE_smallest_1755_more_than_sum_of_digits_l1525_152590

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Property that a number is 1755 more than the sum of its digits -/
def is1755MoreThanSumOfDigits (n : ℕ) : Prop :=
  n = sumOfDigits n + 1755

/-- Theorem stating that 1770 is the smallest natural number that is 1755 more than the sum of its digits -/
theorem smallest_1755_more_than_sum_of_digits :
  (1770 = sumOfDigits 1770 + 1755) ∧
  ∀ m : ℕ, m < 1770 → m ≠ sumOfDigits m + 1755 :=
by sorry

end NUMINAMATH_CALUDE_smallest_1755_more_than_sum_of_digits_l1525_152590


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_2_5_7_l1525_152505

theorem least_three_digit_multiple_of_2_5_7 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 140 → ¬(2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_2_5_7_l1525_152505


namespace NUMINAMATH_CALUDE_symmetric_absolute_value_function_l1525_152506

/-- A function f is symmetric about a point c if f(c + x) = f(c - x) for all x -/
def IsSymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetric_absolute_value_function (a : ℝ) :
  IsSymmetricAbout (fun x ↦ |x + 2*a| - 1) 1 → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_absolute_value_function_l1525_152506


namespace NUMINAMATH_CALUDE_school_student_count_l1525_152557

theorem school_student_count (total : ℕ) (junior_increase senior_increase total_increase : ℚ) 
  (h1 : total = 4200)
  (h2 : junior_increase = 8 / 100)
  (h3 : senior_increase = 11 / 100)
  (h4 : total_increase = 10 / 100) :
  ∃ (junior senior : ℕ), 
    junior + senior = total ∧
    (1 + junior_increase) * junior + (1 + senior_increase) * senior = (1 + total_increase) * total ∧
    junior = 1400 ∧
    senior = 2800 := by
  sorry

end NUMINAMATH_CALUDE_school_student_count_l1525_152557


namespace NUMINAMATH_CALUDE_max_m_value_l1525_152583

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → x^2 - x + 1/2 ≥ m) → 
  m ≤ 1/4 := by
sorry

end NUMINAMATH_CALUDE_max_m_value_l1525_152583


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1525_152526

theorem quadratic_no_real_roots (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + a*x - 4*a = 0) ↔ (-16 < a ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1525_152526


namespace NUMINAMATH_CALUDE_sports_club_size_l1525_152551

/-- The number of members in a sports club -/
def sports_club_members (badminton tennis both neither : ℕ) : ℕ :=
  badminton + tennis - both + neither

/-- Theorem: The sports club has 28 members -/
theorem sports_club_size :
  ∃ (badminton tennis both neither : ℕ),
    badminton = 17 ∧
    tennis = 19 ∧
    both = 10 ∧
    neither = 2 ∧
    sports_club_members badminton tennis both neither = 28 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_size_l1525_152551


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_of_games_l1525_152521

/-- The total amount Tom spent on video games -/
def total_spent : ℝ := 35.52

/-- The cost of the football game -/
def football_cost : ℝ := 14.02

/-- The cost of the strategy game -/
def strategy_cost : ℝ := 9.46

/-- The cost of the Batman game -/
def batman_cost : ℝ := 12.04

/-- Theorem stating that the total amount spent is equal to the sum of individual game costs -/
theorem total_spent_equals_sum_of_games :
  total_spent = football_cost + strategy_cost + batman_cost := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_of_games_l1525_152521


namespace NUMINAMATH_CALUDE_fraction_equality_l1525_152555

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) (h1 : a / b = 3 / 4) : a / (a + b) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1525_152555


namespace NUMINAMATH_CALUDE_radio_selling_price_l1525_152542

/-- Calculates the selling price of an item given its cost price and loss percentage. -/
def selling_price (cost_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  cost_price * (1 - loss_percentage / 100)

/-- Theorem: The selling price of a radio with a cost price of 1500 and a loss percentage of 17 is 1245. -/
theorem radio_selling_price :
  selling_price 1500 17 = 1245 := by
  sorry

end NUMINAMATH_CALUDE_radio_selling_price_l1525_152542


namespace NUMINAMATH_CALUDE_gcd_193116_127413_properties_l1525_152533

theorem gcd_193116_127413_properties :
  let g := Nat.gcd 193116 127413
  ∃ (g : ℕ),
    g = 3 ∧
    g ∣ 3 ∧
    ¬(2 ∣ g) ∧
    ¬(9 ∣ g) ∧
    ¬(11 ∣ g) ∧
    ¬(33 ∣ g) ∧
    ¬(99 ∣ g) := by
  sorry

end NUMINAMATH_CALUDE_gcd_193116_127413_properties_l1525_152533


namespace NUMINAMATH_CALUDE_tank_filling_time_l1525_152579

theorem tank_filling_time (fill_rate_1 fill_rate_2 remaining_time : ℝ) 
  (h1 : fill_rate_1 = 1 / 20)
  (h2 : fill_rate_2 = 1 / 60)
  (h3 : remaining_time = 20.000000000000004)
  : ∃ t : ℝ, t * (fill_rate_1 + fill_rate_2) + remaining_time * fill_rate_2 = 1 ∧ t = 10 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l1525_152579


namespace NUMINAMATH_CALUDE_line_through_interior_point_no_intersection_l1525_152508

/-- Theorem: A line through a point inside a parabola has no intersection with the parabola --/
theorem line_through_interior_point_no_intersection 
  (x y_o : ℝ) (h : y_o^2 < 4*x) : 
  ∀ y : ℝ, (y^2 = 4*((y*y_o)/(2) - x)) → False :=
by sorry

end NUMINAMATH_CALUDE_line_through_interior_point_no_intersection_l1525_152508


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_27_factorial_l1525_152502

/-- The largest power of 3 that divides n! -/
def largest_power_of_three_dividing_factorial (n : ℕ) : ℕ :=
  (n / 3) + (n / 9) + (n / 27)

/-- The ones digit of 3^n -/
def ones_digit_of_power_of_three (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0 -- This case is unreachable, but needed for exhaustiveness

theorem ones_digit_of_largest_power_of_three_dividing_27_factorial :
  ones_digit_of_power_of_three (largest_power_of_three_dividing_factorial 27) = 3 :=
by sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_27_factorial_l1525_152502


namespace NUMINAMATH_CALUDE_percentage_calculation_l1525_152575

theorem percentage_calculation (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 
  0.5 * (0.4 * 0.3 * x) = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1525_152575


namespace NUMINAMATH_CALUDE_prob_coprime_42_l1525_152503

/-- The number of positive integers less than or equal to n that are relatively prime to n -/
def phi (n : ℕ) : ℕ := sorry

theorem prob_coprime_42 : (phi 42 : ℚ) / 42 = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_prob_coprime_42_l1525_152503


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1525_152552

theorem inequality_solution_set (x : ℝ) : (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1525_152552


namespace NUMINAMATH_CALUDE_degree_of_g_l1525_152588

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := -9*x^5 + 5*x^4 + 2*x^2 - x + 6

-- Define the theorem
theorem degree_of_g (g : ℝ → ℝ) :
  (∃ c : ℝ, ∀ x : ℝ, f x + g x = c) →  -- degree of f(x) + g(x) is 0
  (∃ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ, a₅ ≠ 0 ∧ 
    ∀ x : ℝ, g x = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →  -- g(x) is a polynomial of degree 5
  true :=
by sorry

end NUMINAMATH_CALUDE_degree_of_g_l1525_152588


namespace NUMINAMATH_CALUDE_queen_middle_teachers_l1525_152545

/-- Represents a school with students, teachers, and classes. -/
structure School where
  num_students : ℕ
  classes_per_student : ℕ
  classes_per_teacher : ℕ
  students_per_class : ℕ

/-- Calculates the number of teachers in a school. -/
def num_teachers (s : School) : ℕ :=
  (s.num_students * s.classes_per_student) / (s.students_per_class * s.classes_per_teacher)

/-- Queen Middle School -/
def queen_middle : School :=
  { num_students := 1500
  , classes_per_student := 6
  , classes_per_teacher := 5
  , students_per_class := 25
  }

/-- Theorem stating that Queen Middle School has 72 teachers -/
theorem queen_middle_teachers : num_teachers queen_middle = 72 := by
  sorry

end NUMINAMATH_CALUDE_queen_middle_teachers_l1525_152545


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_of_complex_l1525_152510

theorem sum_real_imag_parts_of_complex (z : ℂ) : z = (1 + 2*I) / (1 - 2*I) → (z.re + z.im = 1/5) := by
  sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_of_complex_l1525_152510


namespace NUMINAMATH_CALUDE_number_of_hiding_snakes_l1525_152534

/-- Given a cage with snakes, some of which are hiding, this theorem proves
    the number of hiding snakes. -/
theorem number_of_hiding_snakes
  (total_snakes : ℕ)
  (visible_snakes : ℕ)
  (h1 : total_snakes = 95)
  (h2 : visible_snakes = 31) :
  total_snakes - visible_snakes = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_hiding_snakes_l1525_152534


namespace NUMINAMATH_CALUDE_simplify_expression_l1525_152538

theorem simplify_expression (x : ℝ) : 
  2*x + 4*x^2 - 3 + (5 - 3*x - 9*x^2) + (x+1)^2 = -4*x^2 + x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1525_152538


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1525_152543

theorem absolute_value_inequality (x : ℝ) :
  |((3 * x + 2) / (x - 2))| > 3 ↔ 2/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1525_152543


namespace NUMINAMATH_CALUDE_equation_solutions_l1525_152571

theorem equation_solutions (n : ℕ+) : 
  (∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    s.card = 10 ∧ 
    ∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ 3*x + 3*y + 2*z = n) ↔ 
  n = 17 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1525_152571


namespace NUMINAMATH_CALUDE_largest_package_size_l1525_152550

theorem largest_package_size (a b c : ℕ) (ha : a = 60) (hb : b = 36) (hc : c = 48) :
  Nat.gcd a (Nat.gcd b c) = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l1525_152550


namespace NUMINAMATH_CALUDE_m_range_theorem_l1525_152531

-- Define propositions p and q as functions of x and m
def p (x m : ℝ) : Prop := (x - m)^2 > 3*(x - m)
def q (x : ℝ) : Prop := x^2 + 3*x - 4 < 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ -7 ∨ m ≥ 1

-- Theorem statement
theorem m_range_theorem :
  (∀ x m : ℝ, q x → p x m) ∧ 
  (∃ x m : ℝ, p x m ∧ ¬(q x)) →
  ∀ m : ℝ, m_range m ↔ ∃ x : ℝ, p x m :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1525_152531


namespace NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l1525_152513

/-- The radius of a spherical ball that leaves a circular hole in a frozen lake surface. -/
def ball_radius (hole_width : ℝ) (hole_depth : ℝ) : ℝ :=
  hole_depth

/-- Theorem stating that if a spherical ball leaves a hole 32 cm wide and 16 cm deep
    in a frozen lake surface, its radius is 16 cm. -/
theorem ball_radius_from_hole_dimensions :
  ball_radius 32 16 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l1525_152513


namespace NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l1525_152570

/-- Given a parabola y^2 = -2px where p > 0, if its directrix is tangent to the circle (x-5)^2 + y^2 = 25, then p = 20 -/
theorem parabola_directrix_tangent_circle (p : ℝ) : 
  p > 0 →
  (∃ x y : ℝ, y^2 = -2*p*x) →
  (∃ x : ℝ, x = p/2 ∧ (x-5)^2 = 25) →
  p = 20 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l1525_152570


namespace NUMINAMATH_CALUDE_square_elements_iff_odd_order_l1525_152544

theorem square_elements_iff_odd_order (G : Type*) [Group G] [Fintype G] :
  (∀ g : G, ∃ h : G, h ^ 2 = g) ↔ Odd (Fintype.card G) :=
sorry

end NUMINAMATH_CALUDE_square_elements_iff_odd_order_l1525_152544


namespace NUMINAMATH_CALUDE_least_absolute_prime_l1525_152578

theorem least_absolute_prime (n : ℤ) : 
  Nat.Prime n.natAbs → 101 * n^2 ≤ 3600 → (∀ m : ℤ, Nat.Prime m.natAbs → 101 * m^2 ≤ 3600 → n.natAbs ≤ m.natAbs) → n.natAbs = 2 :=
by sorry

end NUMINAMATH_CALUDE_least_absolute_prime_l1525_152578


namespace NUMINAMATH_CALUDE_original_cost_l1525_152554

theorem original_cost (final_cost : ℝ) : 
  final_cost = 72 → 
  ∃ (original_cost : ℝ), 
    original_cost * (1 + 0.2) * (1 - 0.2) = final_cost ∧ 
    original_cost = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_original_cost_l1525_152554


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1525_152547

theorem sum_of_reciprocals_positive (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_neg_hundred : a * b * c = -100) : 
  1/a + 1/b + 1/c > 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1525_152547


namespace NUMINAMATH_CALUDE_max_total_length_tetrahedron_l1525_152581

/-- Represents a tetrahedron with edge lengths a, b, c, d, e, f --/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f

/-- The condition that at most one edge is longer than 1 --/
def atMostOneLongerThanOne (t : Tetrahedron) : Prop :=
  (t.a ≤ 1 ∨ (t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.b ≤ 1 ∨ (t.a ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.c ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.d ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.e ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.f ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1))

/-- The total length of all edges in a tetrahedron --/
def totalLength (t : Tetrahedron) : ℝ := t.a + t.b + t.c + t.d + t.e + t.f

/-- The theorem stating the maximum total length of edges in a tetrahedron --/
theorem max_total_length_tetrahedron :
  ∀ t : Tetrahedron, atMostOneLongerThanOne t → totalLength t ≤ 5 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_total_length_tetrahedron_l1525_152581


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l1525_152500

theorem min_value_and_inequality (a b x₁ x₂ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hab : a + b = 1) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' = 1 → a'^2 + b'^2/4 ≥ 1/5) ∧
  (a^2 + b^2/4 = 1/5 → a = 1/5 ∧ b = 4/5) ∧
  (a*x₁ + b*x₂) * (b*x₁ + a*x₂) ≥ x₁*x₂ := by
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l1525_152500


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1525_152540

theorem sum_of_fractions_equals_one
  (a b c p q r : ℝ)
  (eq1 : 19 * p + b * q + c * r = 0)
  (eq2 : a * p + 29 * q + c * r = 0)
  (eq3 : a * p + b * q + 56 * r = 0)
  (ha : a ≠ 19)
  (hp : p ≠ 0) :
  a / (a - 19) + b / (b - 29) + c / (c - 56) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1525_152540


namespace NUMINAMATH_CALUDE_factor_expression_l1525_152564

theorem factor_expression (x : ℝ) : 63 * x^19 + 147 * x^38 = 21 * x^19 * (3 + 7 * x^19) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1525_152564


namespace NUMINAMATH_CALUDE_series_convergence_l1525_152561

noncomputable def x : ℕ → ℝ
  | 0 => 1
  | n + 1 => Real.log (Real.exp (x n) - x n)

theorem series_convergence :
  (∑' n, x n) = Real.exp 1 - 1 := by sorry

end NUMINAMATH_CALUDE_series_convergence_l1525_152561


namespace NUMINAMATH_CALUDE_power_of_two_pairs_l1525_152532

theorem power_of_two_pairs (a b : ℕ) (h1 : a ≠ b) (h2 : ∃ k : ℕ, a + b = 2^k) (h3 : ∃ m : ℕ, a * b + 1 = 2^m) :
  (∃ k : ℕ, k ≥ 1 ∧ ((a = 1 ∧ b = 2^k - 1) ∨ (a = 2^k - 1 ∧ b = 1))) ∨
  (∃ k : ℕ, k ≥ 2 ∧ ((a = 2^k - 1 ∧ b = 2^k + 1) ∨ (a = 2^k + 1 ∧ b = 2^k - 1))) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_pairs_l1525_152532


namespace NUMINAMATH_CALUDE_sum_in_base7_l1525_152518

/-- Converts a base 7 number to base 10 --/
def toBase10 (x : List Nat) : Nat :=
  x.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- Converts a base 10 number to base 7 --/
def toBase7 (x : Nat) : List Nat :=
  if x = 0 then [0] else
  let rec aux (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc else aux (n / 7) ((n % 7) :: acc)
  aux x []

theorem sum_in_base7 :
  let a := [2, 3, 4]  -- 432 in base 7
  let b := [4, 5]     -- 54 in base 7
  let c := [6]        -- 6 in base 7
  let sum := toBase10 a + toBase10 b + toBase10 c
  toBase7 sum = [5, 2, 5] := by sorry

end NUMINAMATH_CALUDE_sum_in_base7_l1525_152518


namespace NUMINAMATH_CALUDE_quadratic_real_roots_quadratic_integer_roots_l1525_152569

/-- The quadratic equation kx^2 + (k+1)x + (k-1) = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  k * x^2 + (k + 1) * x + (k - 1) = 0

/-- The set of k values for which the equation has real roots -/
def real_roots_set : Set ℝ :=
  {k | (3 - 2 * Real.sqrt 3) / 3 ≤ k ∧ k ≤ (3 + 2 * Real.sqrt 3) / 3}

/-- The set of k values for which the equation has integer roots -/
def integer_roots_set : Set ℝ :=
  {0, 1, -1/7}

theorem quadratic_real_roots :
  ∀ k : ℝ, (∃ x : ℝ, quadratic_equation k x) ↔ k ∈ real_roots_set :=
sorry

theorem quadratic_integer_roots :
  ∀ k : ℝ, (∃ x : ℤ, quadratic_equation k (x : ℝ)) ↔ k ∈ integer_roots_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_quadratic_integer_roots_l1525_152569


namespace NUMINAMATH_CALUDE_meter_to_skips_conversion_l1525_152596

/-- Proves that 1 meter is equivalent to (g*b*f*d)/(a*e*h*c) skips given the measurement relationships -/
theorem meter_to_skips_conversion
  (a b c d e f g h : ℝ)
  (hops_to_skips : a * 1 = b)
  (jumps_to_hops : c * 1 = d)
  (leaps_to_jumps : e * 1 = f)
  (leaps_to_meters : g * 1 = h)
  (a_pos : 0 < a)
  (c_pos : 0 < c)
  (e_pos : 0 < e)
  (h_pos : 0 < h) :
  1 = (g * b * f * d) / (a * e * h * c) :=
sorry

end NUMINAMATH_CALUDE_meter_to_skips_conversion_l1525_152596


namespace NUMINAMATH_CALUDE_widget_difference_formula_l1525_152523

/-- Represents the widget production difference between Monday and Tuesday -/
def widget_difference (t : ℝ) : ℝ :=
  let w : ℝ := 3 * t - 1
  let monday_production : ℝ := w * t
  let tuesday_production : ℝ := (w + 6) * (t - 3)
  monday_production - tuesday_production

/-- Theorem stating the widget production difference -/
theorem widget_difference_formula (t : ℝ) :
  widget_difference t = 3 * t + 15 := by
  sorry

#check widget_difference_formula

end NUMINAMATH_CALUDE_widget_difference_formula_l1525_152523


namespace NUMINAMATH_CALUDE_simultaneous_ring_theorem_l1525_152587

def bell_ring_time (start_hour start_minute : ℕ) (interval_minutes : ℕ) : ℕ × ℕ := sorry

def next_simultaneous_ring 
  (start_hour start_minute : ℕ) 
  (interval1 interval2 interval3 : ℕ) : ℕ × ℕ := sorry

theorem simultaneous_ring_theorem 
  (h1 : interval1 = 18)
  (h2 : interval2 = 24)
  (h3 : interval3 = 30)
  (h4 : start_hour = 10)
  (h5 : start_minute = 0) :
  next_simultaneous_ring start_hour start_minute interval1 interval2 interval3 = (16, 0) := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_ring_theorem_l1525_152587


namespace NUMINAMATH_CALUDE_conference_center_distance_l1525_152536

/-- Represents the problem of calculating the distance to the conference center --/
theorem conference_center_distance :
  -- Initial speed
  ∀ (initial_speed : ℝ),
  -- Speed increase
  ∀ (speed_increase : ℝ),
  -- Distance covered in first hour
  ∀ (first_hour_distance : ℝ),
  -- Late arrival time if continued at initial speed
  ∀ (late_arrival_time : ℝ),
  -- Early arrival time with increased speed
  ∀ (early_arrival_time : ℝ),
  -- Conditions from the problem
  initial_speed = 40 →
  speed_increase = 20 →
  first_hour_distance = 40 →
  late_arrival_time = 1.5 →
  early_arrival_time = 1 →
  -- Conclusion: The distance to the conference center is 100 miles
  ∃ (distance : ℝ), distance = 100 := by
  sorry


end NUMINAMATH_CALUDE_conference_center_distance_l1525_152536


namespace NUMINAMATH_CALUDE_integer_square_root_of_seven_minus_x_l1525_152522

theorem integer_square_root_of_seven_minus_x (x : ℕ+) :
  (∃ (n : ℤ), n^2 = 7 - x.val) → x.val = 3 ∨ x.val = 6 ∨ x.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_integer_square_root_of_seven_minus_x_l1525_152522


namespace NUMINAMATH_CALUDE_a_power_m_plus_2n_l1525_152573

theorem a_power_m_plus_2n (a : ℝ) (m n : ℤ) (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(m + 2*n) = 18 := by
  sorry

end NUMINAMATH_CALUDE_a_power_m_plus_2n_l1525_152573


namespace NUMINAMATH_CALUDE_difference_of_squares_l1525_152507

theorem difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1525_152507


namespace NUMINAMATH_CALUDE_trevors_age_problem_l1525_152511

theorem trevors_age_problem (trevor_age_decade_ago : ℕ) (brother_current_age : ℕ) : 
  trevor_age_decade_ago = 16 →
  brother_current_age = 32 →
  ∃ x : ℕ, x = 20 ∧ 2 * (trevor_age_decade_ago + 10 - x) = brother_current_age - x :=
by sorry

end NUMINAMATH_CALUDE_trevors_age_problem_l1525_152511


namespace NUMINAMATH_CALUDE_unique_whole_number_between_l1525_152572

theorem unique_whole_number_between (N : ℤ) : 
  (5.5 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 6) ↔ N = 23 := by
sorry

end NUMINAMATH_CALUDE_unique_whole_number_between_l1525_152572


namespace NUMINAMATH_CALUDE_four_spheres_existence_l1525_152597

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray starting from a point
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a point is inside a sphere
def isInside (p : Point3D) (s : Sphere) : Prop := sorry

-- Function to check if two spheres intersect
def intersect (s1 s2 : Sphere) : Prop := sorry

-- Function to check if a ray intersects a sphere
def rayIntersectsSphere (r : Ray) (s : Sphere) : Prop := sorry

-- Theorem statement
theorem four_spheres_existence (A : Point3D) : 
  ∃ (s1 s2 s3 s4 : Sphere),
    (¬ isInside A s1) ∧ (¬ isInside A s2) ∧ (¬ isInside A s3) ∧ (¬ isInside A s4) ∧
    (¬ intersect s1 s2) ∧ (¬ intersect s1 s3) ∧ (¬ intersect s1 s4) ∧
    (¬ intersect s2 s3) ∧ (¬ intersect s2 s4) ∧ (¬ intersect s3 s4) ∧
    (∀ (r : Ray), r.origin = A → 
      rayIntersectsSphere r s1 ∨ rayIntersectsSphere r s2 ∨ 
      rayIntersectsSphere r s3 ∨ rayIntersectsSphere r s4) :=
by
  sorry

end NUMINAMATH_CALUDE_four_spheres_existence_l1525_152597


namespace NUMINAMATH_CALUDE_determine_M_value_l1525_152514

theorem determine_M_value (a b c d : ℤ) (M : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  a + b + c + d = 0 →
  M = (b * c - a * d) * (a * c - b * d) * (a * b - c * d) →
  96100 < M ∧ M < 98000 →
  M = 97344 := by
  sorry

end NUMINAMATH_CALUDE_determine_M_value_l1525_152514


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l1525_152519

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  (1 / (1 + a)) + (4 / (4 + b)) ≥ 9/8 := by sorry

theorem min_value_achievable (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 3 ∧
  (1 / (1 + a₀)) + (4 / (4 + b₀)) = 9/8 := by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l1525_152519


namespace NUMINAMATH_CALUDE_two_rolls_give_target_prob_l1525_152525

-- Define a type for a six-sided die
def Die := Fin 6

-- Define the number of sides on the die
def numSides : ℕ := 6

-- Define the target sum
def targetSum : ℕ := 9

-- Define the target probability
def targetProb : ℚ := 1 / 9

-- Function to calculate the number of ways to get a sum of 9 with n rolls
def waysToGetSum (n : ℕ) : ℕ := sorry

-- Function to calculate the total number of possible outcomes with n rolls
def totalOutcomes (n : ℕ) : ℕ := numSides ^ n

-- Theorem stating that rolling the die twice gives the target probability
theorem two_rolls_give_target_prob :
  ∃ (n : ℕ), (waysToGetSum n : ℚ) / (totalOutcomes n) = targetProb ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_two_rolls_give_target_prob_l1525_152525


namespace NUMINAMATH_CALUDE_bar_chart_ratio_difference_l1525_152546

theorem bar_chart_ratio_difference 
  (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
  a / (a + b) - c / (c + d) = (a * d - b * c) / ((a + b) * (c + d)) :=
by sorry

end NUMINAMATH_CALUDE_bar_chart_ratio_difference_l1525_152546


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l1525_152565

theorem quadratic_roots_properties (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (h_roots : ∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₁^2 - p*r₁ + q = 0 ∧ r₂^2 - p*r₂ + q = 0) :
  (∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₁^2 - p*r₁ + q = 0 ∧ r₂^2 - p*r₂ + q = 0 ∧ 
    (∃ (k : ℕ), r₁ - r₂ = 2*k + 1 ∨ r₂ - r₁ = 2*k + 1)) ∧ 
  (∃ (r : ℕ), (r^2 - p*r + q = 0) ∧ Prime r) ∧
  Prime (p^2 - q) ∧
  Prime (p + q) := by
  sorry

#check quadratic_roots_properties

end NUMINAMATH_CALUDE_quadratic_roots_properties_l1525_152565


namespace NUMINAMATH_CALUDE_expression_value_l1525_152582

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  2 * x^2 + 3 * y^2 - z^2 + 4 * x * y - 6 * x * z = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1525_152582


namespace NUMINAMATH_CALUDE_algebraic_equality_l1525_152548

theorem algebraic_equality (a b c : ℝ) : a - b + c = a - (b - c) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equality_l1525_152548


namespace NUMINAMATH_CALUDE_orange_harvest_total_l1525_152537

/-- The number of days the orange harvest lasts -/
def harvest_days : ℕ := 4

/-- The number of sacks harvested per day -/
def sacks_per_day : ℕ := 14

/-- The total number of sacks harvested -/
def total_sacks : ℕ := harvest_days * sacks_per_day

theorem orange_harvest_total :
  total_sacks = 56 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_total_l1525_152537


namespace NUMINAMATH_CALUDE_different_color_probability_l1525_152501

/-- The probability of drawing two chips of different colors from a bag -/
theorem different_color_probability (total_chips : ℕ) (blue_chips : ℕ) (yellow_chips : ℕ) 
  (h1 : total_chips = blue_chips + yellow_chips)
  (h2 : blue_chips = 7)
  (h3 : yellow_chips = 2) :
  (blue_chips * yellow_chips + yellow_chips * blue_chips) / (total_chips * (total_chips - 1)) = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l1525_152501
