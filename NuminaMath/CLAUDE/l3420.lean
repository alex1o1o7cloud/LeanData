import Mathlib

namespace max_a_correct_l3420_342018

/-- The inequality x^2 - 4x - a - 1 ≥ 0 has solutions for x ∈ [1, 4] -/
def has_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc 1 4 ∧ x^2 - 4*x - a - 1 ≥ 0

/-- The maximum value of a for which the inequality has solutions -/
def max_a : ℝ := -1

theorem max_a_correct :
  ∀ a : ℝ, has_solutions a ↔ a ≤ max_a :=
by sorry

end max_a_correct_l3420_342018


namespace initial_sets_count_l3420_342031

/-- The number of letters available (A through J) -/
def n : ℕ := 10

/-- The length of each set of initials -/
def k : ℕ := 3

/-- The number of different three-letter sets of initials possible using letters A through J, with no repetition -/
def num_initial_sets : ℕ := n * (n - 1) * (n - 2)

theorem initial_sets_count : num_initial_sets = 720 := by
  sorry

end initial_sets_count_l3420_342031


namespace tangent_line_equation_l3420_342097

/-- The equation of the line passing through the tangency points of two tangent lines drawn from a point to a circle. -/
theorem tangent_line_equation (P : ℝ × ℝ) (r : ℝ) :
  P = (5, 3) →
  r = 3 →
  ∃ A B : ℝ × ℝ,
    (A.1^2 + A.2^2 = r^2) ∧
    (B.1^2 + B.2^2 = r^2) ∧
    ((A.1 - P.1)^2 + (A.2 - P.2)^2 = ((P.1)^2 + (P.2)^2 - r^2)) ∧
    ((B.1 - P.1)^2 + (B.2 - P.2)^2 = ((P.1)^2 + (P.2)^2 - r^2)) ∧
    (∀ x y : ℝ, 5*x + 3*y - 9 = 0 ↔ (x - A.1)*(B.2 - A.2) = (y - A.2)*(B.1 - A.1)) :=
by sorry

end tangent_line_equation_l3420_342097


namespace valid_seating_count_l3420_342066

/-- Represents a seating arrangement around a round table -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Checks if two positions are adjacent on a round table with 12 chairs -/
def isAdjacent (a b : Fin 12) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 11 ∧ b = 0) ∨ (a = 0 ∧ b = 11)

/-- Checks if two positions are across from each other on a round table with 12 chairs -/
def isAcross (a b : Fin 12) : Prop := (a + 6 = b) ∨ (b + 6 = a)

/-- Represents a valid seating arrangement for 6 married couples -/
def ValidSeating (s : SeatingArrangement) : Prop :=
  ∀ i j : Fin 12,
    -- Men and women alternate
    (i.val % 2 = 0 → s i < 6) ∧
    (i.val % 2 = 1 → s i ≥ 6) ∧
    -- No one sits next to or across from their spouse
    (s i < 6 ∧ s j ≥ 6 ∧ s i + 6 = s j →
      ¬(isAdjacent i j ∨ isAcross i j))

/-- The number of valid seating arrangements -/
def numValidSeatings : ℕ := sorry

theorem valid_seating_count : numValidSeatings = 5184 := by sorry

end valid_seating_count_l3420_342066


namespace abs_neg_four_minus_two_l3420_342017

theorem abs_neg_four_minus_two : |(-4 : ℤ) - 2| = 6 := by
  sorry

end abs_neg_four_minus_two_l3420_342017


namespace M_enumeration_l3420_342074

def M : Set ℕ := {a | a > 0 ∧ ∃ k : ℤ, 4 / (1 - a) = k}

theorem M_enumeration : M = {2, 3, 5} := by sorry

end M_enumeration_l3420_342074


namespace ratio_max_min_sequence_diff_l3420_342083

def geometric_sequence (n : ℕ) : ℚ :=
  (3/2) * (-1/2) ^ (n - 1)

def sum_n_terms (n : ℕ) : ℚ :=
  (3/2) * (1 - (-1/2)^n) / (1 + 1/2)

def sequence_diff (n : ℕ) : ℚ :=
  sum_n_terms n - 1 / sum_n_terms n

theorem ratio_max_min_sequence_diff :
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
    sequence_diff m / sequence_diff n = -10/7 ∧
    ∀ (k : ℕ), k > 0 → 
      sequence_diff m ≥ sequence_diff k ∧
      sequence_diff k ≥ sequence_diff n) :=
sorry

end ratio_max_min_sequence_diff_l3420_342083


namespace ball_hitting_ground_time_l3420_342085

/-- The time when the ball hits the ground, given the initial conditions and equation of motion -/
theorem ball_hitting_ground_time : ∃ t : ℝ, t > 0 ∧ -16 * t^2 + 32 * t + 180 = 0 ∧ t = 4.5 := by
  sorry

end ball_hitting_ground_time_l3420_342085


namespace team_a_more_uniform_l3420_342080

/-- Represents a team of girls in a duet --/
structure Team where
  members : Fin 6 → ℝ
  variance : ℝ

/-- The problem setup --/
def problem_setup (team_a team_b : Team) : Prop :=
  team_a.variance = 1.2 ∧ team_b.variance = 2.0

/-- Definition of more uniform heights --/
def more_uniform (team1 team2 : Team) : Prop :=
  team1.variance < team2.variance

/-- The main theorem --/
theorem team_a_more_uniform (team_a team_b : Team) 
  (h : problem_setup team_a team_b) : 
  more_uniform team_a team_b := by
  sorry

#check team_a_more_uniform

end team_a_more_uniform_l3420_342080


namespace tangent_triangle_area_l3420_342016

theorem tangent_triangle_area (a : ℝ) : 
  a > 0 → 
  (1/2 * a/2 * a^2 = 2) → 
  a = 2 := by
sorry

end tangent_triangle_area_l3420_342016


namespace least_even_integer_for_300p_perfect_square_l3420_342094

theorem least_even_integer_for_300p_perfect_square :
  ∀ p : ℕ,
    p % 2 = 0 →
    (∃ n : ℕ, 300 * p = n^2) →
    p ≥ 18 :=
by sorry

end least_even_integer_for_300p_perfect_square_l3420_342094


namespace nine_ants_nine_trips_l3420_342059

/-- Represents the number of grains of rice that can be moved by a given number of ants in a given number of trips -/
def rice_moved (ants : ℕ) (trips : ℕ) : ℚ :=
  (24 : ℚ) * ants * trips / (12 * 6)

/-- Theorem stating that 9 ants can move 27 grains of rice in 9 trips -/
theorem nine_ants_nine_trips :
  rice_moved 9 9 = 27 := by
  sorry

end nine_ants_nine_trips_l3420_342059


namespace greatest_integer_solution_l3420_342041

theorem greatest_integer_solution (x : ℝ) : 
  (((|x^2 - 2| - 7) * (|x + 3| - 5)) / (|x - 3| - |x - 1|) > 0) → 
  (∃ (n : ℤ), n ≤ x ∧ n ≤ 1 ∧ ∀ (m : ℤ), m ≤ x → m ≤ n) :=
by sorry

end greatest_integer_solution_l3420_342041


namespace systematic_sampling_seventh_group_l3420_342021

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (m : Nat) (k : Nat) : Nat :=
  (m + k) % 10

/-- The population size -/
def populationSize : Nat := 100

/-- The number of groups -/
def numGroups : Nat := 10

/-- The size of each group -/
def groupSize : Nat := populationSize / numGroups

/-- The starting number of the k-th group -/
def groupStart (k : Nat) : Nat :=
  (k - 1) * groupSize

theorem systematic_sampling_seventh_group :
  ∀ m : Nat,
    m = 6 →
    ∃ n : Nat,
      n = 63 ∧
      n ≥ groupStart 7 ∧
      n < groupStart 7 + groupSize ∧
      n % 10 = systematicSample m 7 :=
sorry

end systematic_sampling_seventh_group_l3420_342021


namespace emily_sixth_score_l3420_342055

def emily_scores : List ℕ := [94, 90, 85, 90, 105]

def arithmetic_mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem emily_sixth_score :
  ∃ (sixth_score : ℕ),
    sixth_score > emily_scores.minimum ∧
    arithmetic_mean (emily_scores ++ [sixth_score]) = 95 ∧
    sixth_score = 106 := by
  sorry

end emily_sixth_score_l3420_342055


namespace greatest_whole_number_satisfying_inequality_l3420_342044

theorem greatest_whole_number_satisfying_inequality :
  ∀ (n : ℤ), n ≤ 0 ↔ (3 : ℝ) * n + 2 < 5 - 2 * n :=
by sorry

end greatest_whole_number_satisfying_inequality_l3420_342044


namespace digit_156_is_zero_l3420_342052

-- Define the fraction
def fraction : ℚ := 37 / 740

-- Define a function to get the nth digit after the decimal point
def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem digit_156_is_zero : nthDigitAfterDecimal fraction 156 = 0 := by sorry

end digit_156_is_zero_l3420_342052


namespace smallest_number_of_editors_l3420_342082

/-- The total number of people at the conference -/
def total : ℕ := 90

/-- The number of writers at the conference -/
def writers : ℕ := 45

/-- The number of people who are both writers and editors -/
def both : ℕ := 6

/-- The number of people who are neither writers nor editors -/
def neither : ℕ := 2 * both

/-- The number of editors at the conference -/
def editors : ℕ := total - writers - neither + both

theorem smallest_number_of_editors : editors = 39 := by
  sorry

end smallest_number_of_editors_l3420_342082


namespace cubic_equation_one_positive_root_l3420_342006

theorem cubic_equation_one_positive_root (a b : ℝ) (hb : b > 0) :
  ∃! x : ℝ, x > 0 ∧ x^3 + a*x^2 - b = 0 :=
sorry

end cubic_equation_one_positive_root_l3420_342006


namespace probability_nine_heads_in_twelve_flips_l3420_342035

def n : ℕ := 12
def k : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (n.choose k : ℚ) / (2 ^ n : ℚ) = 220 / 4096 := by
  sorry

end probability_nine_heads_in_twelve_flips_l3420_342035


namespace consecutive_odd_numbers_sum_l3420_342025

theorem consecutive_odd_numbers_sum (n : ℕ) : 
  (n % 2 = 1) → (n + (n + 2) = 48) → n = 23 := by
  sorry

end consecutive_odd_numbers_sum_l3420_342025


namespace set_operations_and_inclusion_l3420_342093

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a - 1}

theorem set_operations_and_inclusion (a : ℝ) :
  (a = 7/2 → M ∪ N a = {x | -2 ≤ x ∧ x ≤ 6} ∧
             (Set.univ \ M) ∩ N a = {x | 5 < x ∧ x ≤ 6}) ∧
  (M ⊇ N a ↔ a ∈ Set.Iic 3) :=
sorry

end set_operations_and_inclusion_l3420_342093


namespace temperature_difference_l3420_342030

theorem temperature_difference (highest lowest : ℤ) (h1 : highest = -9) (h2 : lowest = -22) :
  highest - lowest = 13 := by
  sorry

end temperature_difference_l3420_342030


namespace characterization_of_p_l3420_342003

/-- The polynomial equation in x with parameter p -/
def f (p : ℝ) (x : ℝ) : ℝ := x^4 + 3*p*x^3 + x^2 + 3*p*x + 1

/-- A function has at least two distinct positive real roots -/
def has_two_distinct_positive_roots (g : ℝ → ℝ) : Prop :=
  ∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ g x = 0 ∧ g y = 0

/-- The main theorem: characterization of p for which f has at least two distinct positive real roots -/
theorem characterization_of_p (p : ℝ) : 
  has_two_distinct_positive_roots (f p) ↔ p < 1/4 := by sorry

end characterization_of_p_l3420_342003


namespace caramel_chews_theorem_l3420_342038

/-- Represents the distribution of candy bags -/
structure CandyDistribution where
  totalCandies : ℕ
  totalBags : ℕ
  heartsCount : ℕ
  kissesCount : ℕ
  jelliesCount : ℕ
  heartsExtra : ℕ
  jelliesMultiplier : ℚ

/-- Calculates the number of candies in caramel chews bags -/
def caramelChewsCandies (d : CandyDistribution) : ℕ :=
  let remainingBags := d.totalBags - (d.heartsCount + d.kissesCount + d.jelliesCount)
  let baseCandy := (d.totalCandies - d.heartsCount * d.heartsExtra) / d.totalBags
  remainingBags * baseCandy

/-- Theorem stating that for the given distribution, caramel chews bags contain 44 candies -/
theorem caramel_chews_theorem (d : CandyDistribution) 
  (h1 : d.totalCandies = 500)
  (h2 : d.totalBags = 20)
  (h3 : d.heartsCount = 6)
  (h4 : d.kissesCount = 8)
  (h5 : d.jelliesCount = 4)
  (h6 : d.heartsExtra = 2)
  (h7 : d.jelliesMultiplier = 3/2) :
  caramelChewsCandies d = 44 := by
  sorry

end caramel_chews_theorem_l3420_342038


namespace integer_solutions_count_l3420_342060

theorem integer_solutions_count : 
  ∃! (S : Finset ℤ), 
    (∀ a ∈ S, ∃ x : ℤ, x^2 + a*x - 6*a = 0) ∧ 
    (∀ a : ℤ, (∃ x : ℤ, x^2 + a*x - 6*a = 0) → a ∈ S) ∧
    S.card = 9 :=
by sorry

end integer_solutions_count_l3420_342060


namespace cos_pi_third_minus_alpha_l3420_342077

theorem cos_pi_third_minus_alpha (α : ℝ) (h : Real.sin (π / 6 + α) = 2 / 3) :
  Real.cos (π / 3 - α) = 2 / 3 := by
sorry

end cos_pi_third_minus_alpha_l3420_342077


namespace angle_measure_in_triangle_l3420_342061

theorem angle_measure_in_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = π ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  (2 * b - c) * Real.cos A = a * Real.cos C →
  A = π / 3 := by
sorry

end angle_measure_in_triangle_l3420_342061


namespace quadratic_root_condition_l3420_342042

/-- Given a quadratic equation x^2 + 2(a-1)x + 2a + 6 = 0 with one positive and one negative real root,
    prove that a < -3 --/
theorem quadratic_root_condition (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ 
    x^2 + 2*(a-1)*x + 2*a + 6 = 0 ∧
    y^2 + 2*(a-1)*y + 2*a + 6 = 0) →
  a < -3 := by
sorry

end quadratic_root_condition_l3420_342042


namespace inequality_solution_range_l3420_342046

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 1| - |x - 3| > a) → a < 2 := by
  sorry

end inequality_solution_range_l3420_342046


namespace leap_years_in_200_years_l3420_342028

/-- A calendrical system where leap years occur every four years without exception. -/
structure CalendarSystem where
  /-- The period in years -/
  period : ℕ
  /-- The frequency of leap years -/
  leap_year_frequency : ℕ
  /-- Assertion that leap years occur every four years -/
  leap_year_every_four : leap_year_frequency = 4

/-- The number of leap years in a given period for a calendar system -/
def num_leap_years (c : CalendarSystem) : ℕ :=
  c.period / c.leap_year_frequency

/-- Theorem stating that in a 200-year period with leap years every 4 years, there are 50 leap years -/
theorem leap_years_in_200_years (c : CalendarSystem) 
  (h_period : c.period = 200) : num_leap_years c = 50 := by
  sorry

end leap_years_in_200_years_l3420_342028


namespace elephant_exodus_rate_calculation_l3420_342056

/-- The rate of elephants leaving Utopia National Park during an exodus --/
def elephant_exodus_rate (initial_elephants : ℕ) (exodus_duration : ℕ) 
  (new_elephants_duration : ℕ) (new_elephants_rate : ℕ) (final_elephants : ℕ) : ℕ :=
  (initial_elephants - final_elephants + new_elephants_duration * new_elephants_rate) / exodus_duration

/-- Theorem stating the rate of elephants leaving during the exodus --/
theorem elephant_exodus_rate_calculation :
  elephant_exodus_rate 30000 4 7 1500 28980 = 2880 :=
by sorry

end elephant_exodus_rate_calculation_l3420_342056


namespace line_intersections_l3420_342081

/-- The line equation 4y - 5x = 20 -/
def line_equation (x y : ℝ) : Prop := 4 * y - 5 * x = 20

/-- The x-axis intercept of the line -/
def x_intercept : ℝ × ℝ := (-4, 0)

/-- The y-axis intercept of the line -/
def y_intercept : ℝ × ℝ := (0, 5)

/-- Theorem stating that the line intersects the x-axis and y-axis at the given points -/
theorem line_intersections :
  (line_equation x_intercept.1 x_intercept.2) ∧
  (line_equation y_intercept.1 y_intercept.2) :=
by sorry

end line_intersections_l3420_342081


namespace intersection_of_A_and_B_l3420_342002

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by sorry

end intersection_of_A_and_B_l3420_342002


namespace two_negative_roots_l3420_342004

/-- The polynomial function we're analyzing -/
def f (q : ℝ) (x : ℝ) : ℝ := x^4 + 2*q*x^3 - 3*x^2 + 2*q*x + 1

/-- Theorem stating that for any q < 1/4, the equation f q x = 0 has at least two distinct negative real roots -/
theorem two_negative_roots (q : ℝ) (h : q < 1/4) : 
  ∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ f q x₁ = 0 ∧ f q x₂ = 0 := by
  sorry

end two_negative_roots_l3420_342004


namespace january_oil_bill_l3420_342075

theorem january_oil_bill (january february : ℝ) 
  (h1 : february / january = 5 / 4)
  (h2 : (february + 30) / january = 3 / 2) :
  january = 120 := by
sorry

end january_oil_bill_l3420_342075


namespace distance_center_to_endpoint_l3420_342067

/-- Given two points representing the endpoints of a circle's diameter,
    calculate the distance from the center of the circle to one of the endpoints. -/
theorem distance_center_to_endpoint
  (p1 : ℝ × ℝ)
  (p2 : ℝ × ℝ)
  (h1 : p1 = (12, -8))
  (h2 : p2 = (-6, 4))
  : Real.sqrt ((12 - ((p1.1 + p2.1) / 2))^2 + (-8 - ((p1.2 + p2.2) / 2))^2) = Real.sqrt 117 :=
by sorry

end distance_center_to_endpoint_l3420_342067


namespace special_number_pair_l3420_342009

/-- Given two distinct positive integers a and b, such that b is a multiple of a,
    both a and b consist of 2n digits in decimal form with no leading zeros,
    and the first n digits of a are the same as the last n digits of b (and vice versa),
    prove that a = (10^(2n) - 1) / 7 and b = 6 * (10^(2n) - 1) / 7 -/
theorem special_number_pair (n : ℕ) (a b : ℕ) :
  (a ≠ b) →
  (a > 0) →
  (b > 0) →
  (∃ (k : ℕ), b = k * a) →
  (10^n ≤ a) →
  (a < 10^(2*n)) →
  (10^n ≤ b) →
  (b < 10^(2*n)) →
  (∃ (x y : ℕ), a = 10^n * x + y ∧ b = 10^n * y + x ∧ x < 10^n ∧ y < 10^n) →
  (a = (10^(2*n) - 1) / 7 ∧ b = 6 * (10^(2*n) - 1) / 7) := by
sorry

end special_number_pair_l3420_342009


namespace harvester_equations_l3420_342010

theorem harvester_equations (x y : ℝ) : True → ∃ (eq1 eq2 : ℝ → ℝ → Prop),
  (∀ a b, eq1 a b ↔ 2 * (2 * a + 5 * b) = 3.6) ∧
  (∀ a b, eq2 a b ↔ 5 * (3 * a + 2 * b) = 8) ∧
  (eq1 x y ∧ eq2 x y) :=
by
  sorry

end harvester_equations_l3420_342010


namespace triangle_side_cube_l3420_342089

/-- Given a triangle ABC with positive integer side lengths a, b, and c, 
    where gcd(a,b,c) = 1 and ∠A = 3∠B, at least one of a, b, and c is a cube. -/
theorem triangle_side_cube (a b c : ℕ+) (angleA angleB : ℝ) : 
  (a.val.gcd (b.val.gcd c.val) = 1) →
  (angleA = 3 * angleB) →
  (∃ (x : ℕ+), x^3 = a ∨ x^3 = b ∨ x^3 = c) := by
sorry

end triangle_side_cube_l3420_342089


namespace rabbit_race_l3420_342057

theorem rabbit_race (pink_speed white_speed : ℝ) (time_difference : ℝ) :
  pink_speed = 15 →
  white_speed = 10 →
  time_difference = 0.5 →
  ∃ (pink_time : ℝ),
    pink_time * pink_speed = (pink_time + time_difference) * white_speed ∧
    pink_time = 1 :=
by sorry

end rabbit_race_l3420_342057


namespace words_per_page_larger_type_l3420_342014

/-- Given an article with a total of 48,000 words printed on 21 pages,
    where 17 pages use smaller type with 2,400 words each,
    prove that the remaining pages in larger type contain 1,800 words each. -/
theorem words_per_page_larger_type :
  ∀ (total_words total_pages smaller_type_pages words_per_page_smaller : ℕ),
    total_words = 48000 →
    total_pages = 21 →
    smaller_type_pages = 17 →
    words_per_page_smaller = 2400 →
    (total_pages - smaller_type_pages) * 
      ((total_words - smaller_type_pages * words_per_page_smaller) / (total_pages - smaller_type_pages)) = 1800 := by
  sorry

end words_per_page_larger_type_l3420_342014


namespace unique_triangle_number_three_identical_digits_l3420_342096

/-- The sum of the first n positive integers -/
def triangle_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Checks if a number is a three-digit number composed of identical digits -/
def is_three_identical_digits (n : ℕ) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ d < 10 ∧ n = d * 111

theorem unique_triangle_number_three_identical_digits :
  ∃! (n : ℕ), n > 0 ∧ is_three_identical_digits (triangle_number n) :=
sorry

end unique_triangle_number_three_identical_digits_l3420_342096


namespace diamond_neg_one_six_l3420_342092

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a * b^2 - b + 1

-- Theorem statement
theorem diamond_neg_one_six : diamond (-1) 6 = -41 := by
  sorry

end diamond_neg_one_six_l3420_342092


namespace trig_identities_l3420_342029

/-- Theorem: Trigonometric identities for specific angles -/
theorem trig_identities :
  (∃ (x y : ℝ), x = 263 * π / 180 ∧ y = 203 * π / 180 ∧
    Real.cos x * Real.cos y + Real.sin (83 * π / 180) * Real.sin (23 * π / 180) = 1/2) ∧
  (∃ (z : ℝ), z = 8 * π / 180 ∧
    (Real.cos (7 * π / 180) - Real.sin (15 * π / 180) * Real.sin z) / Real.cos z =
    (Real.sqrt 6 + Real.sqrt 2) / 4) :=
by
  sorry

end trig_identities_l3420_342029


namespace smallest_base_for_perfect_fourth_power_l3420_342078

/-- Given that 5n is a positive integer represented as 777 in base b,
    and n is a perfect fourth power, prove that the smallest positive
    integer b satisfying these conditions is 41. -/
theorem smallest_base_for_perfect_fourth_power (n : ℕ) (b : ℕ) : 
  (5 * n : ℕ) > 0 ∧ 
  (5 * n = 7 * b^2 + 7 * b + 7) ∧
  (∃ (x : ℕ), n = x^4) →
  (∀ (b' : ℕ), b' ≥ 1 ∧ 
    (∃ (n' : ℕ), (5 * n' : ℕ) > 0 ∧ 
      (5 * n' = 7 * b'^2 + 7 * b' + 7) ∧
      (∃ (x : ℕ), n' = x^4)) →
    b' ≥ b) ∧
  b = 41 :=
by sorry

end smallest_base_for_perfect_fourth_power_l3420_342078


namespace y_value_theorem_l3420_342090

theorem y_value_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 := by
  sorry

end y_value_theorem_l3420_342090


namespace complement_intersection_theorem_l3420_342048

universe u

def U : Set Nat := {0, 1, 2, 3}
def A : Set Nat := {0, 1}
def B : Set Nat := {1, 2, 3}

theorem complement_intersection_theorem :
  (Set.compl A ∩ B) = {2, 3} := by sorry

end complement_intersection_theorem_l3420_342048


namespace largest_common_divisor_360_450_l3420_342051

theorem largest_common_divisor_360_450 : Nat.gcd 360 450 = 90 := by
  sorry

end largest_common_divisor_360_450_l3420_342051


namespace subSubfaces_12_9_l3420_342064

/-- The number of k-dimensional sub-subfaces in an n-dimensional cube -/
def subSubfaces (n k : ℕ) : ℕ := 2^(n - k) * (Nat.choose n k)

/-- Theorem: The number of 9-dimensional sub-subfaces in a 12-dimensional cube is 1760 -/
theorem subSubfaces_12_9 : subSubfaces 12 9 = 1760 := by
  sorry

end subSubfaces_12_9_l3420_342064


namespace thirty_six_times_sum_of_digits_l3420_342005

def sum_of_digits (x : ℕ) : ℕ := sorry

theorem thirty_six_times_sum_of_digits :
  ∀ x : ℕ, x = 36 * sum_of_digits x ↔ x = 324 ∨ x = 648 := by sorry

end thirty_six_times_sum_of_digits_l3420_342005


namespace not_perfect_square_l3420_342084

theorem not_perfect_square (n : ℕ+) : ¬ ∃ m : ℤ, (2551 * 543^n.val - 2008 * 7^n.val : ℤ) = m^2 := by
  sorry

end not_perfect_square_l3420_342084


namespace f_positive_solution_a_range_l3420_342036

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 3|

-- Theorem for the solution of f(x) > 0
theorem f_positive_solution :
  ∀ x : ℝ, f x > 0 ↔ x < -4 ∨ x > 2/3 := by sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∀ x : ℝ, a - 3*|x - 3| < f x) ↔ a < 7 := by sorry

end f_positive_solution_a_range_l3420_342036


namespace no_point_satisfies_conditions_l3420_342011

-- Define a triangle as a structure with three points
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to check if a point is inside a triangle
def isInside (T : Triangle) (D : ℝ × ℝ) : Prop :=
  sorry

-- Define a function to get the shortest side of a triangle
def shortestSide (T : Triangle) : ℝ :=
  sorry

-- Main theorem
theorem no_point_satisfies_conditions (ABC : Triangle) :
  ¬ ∃ D : ℝ × ℝ,
    isInside ABC D ∧
    shortestSide (Triangle.mk ABC.B ABC.C D) = 1 ∧
    shortestSide (Triangle.mk ABC.A ABC.C D) = 2 ∧
    shortestSide (Triangle.mk ABC.A ABC.B D) = 3 :=
by
  sorry

end no_point_satisfies_conditions_l3420_342011


namespace equilateral_triangle_side_length_l3420_342013

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (A : Point) (B : Point) (C : Point)

/-- The perpendicular distance from a point to a line -/
def perpendicularDistance (P : Point) (A : Point) (B : Point) : ℝ := sorry

theorem equilateral_triangle_side_length 
  (ABC : EquilateralTriangle) (P : Point) :
  perpendicularDistance P ABC.A ABC.B = 2 →
  perpendicularDistance P ABC.B ABC.C = 4 →
  perpendicularDistance P ABC.C ABC.A = 6 →
  ∃ (side : ℝ), side = 8 * Real.sqrt 3 ∧ 
    (perpendicularDistance ABC.A ABC.B ABC.C = side ∧
     perpendicularDistance ABC.B ABC.C ABC.A = side ∧
     perpendicularDistance ABC.C ABC.A ABC.B = side) :=
by
  sorry

end equilateral_triangle_side_length_l3420_342013


namespace solve_equation_l3420_342095

-- Define the ⊗ operation
def otimes (m n : ℝ) : ℝ := m^2 - 2*m*n

-- State the theorem
theorem solve_equation (x : ℝ) : 
  otimes (x + 1) (x - 2) = 5 → x = 0 ∨ x = 4 := by
  sorry

end solve_equation_l3420_342095


namespace lucinda_jelly_beans_l3420_342020

/-- The number of grape jelly beans Lucinda originally had -/
def original_grape : ℕ := 180

/-- The number of lemon jelly beans Lucinda originally had -/
def original_lemon : ℕ := original_grape / 3

/-- The number of grape jelly beans Lucinda has after gifting -/
def remaining_grape : ℕ := original_grape - 20

/-- The number of lemon jelly beans Lucinda has after gifting -/
def remaining_lemon : ℕ := original_lemon - 20

theorem lucinda_jelly_beans :
  (original_grape = 3 * original_lemon) ∧
  (remaining_grape = 4 * remaining_lemon) →
  original_grape = 180 :=
by sorry

end lucinda_jelly_beans_l3420_342020


namespace other_person_speed_l3420_342068

/-- Proves that given Roja's speed and the final distance after a certain time, 
    the other person's speed can be determined. -/
theorem other_person_speed 
  (roja_speed : ℝ) 
  (time : ℝ) 
  (final_distance : ℝ) 
  (h1 : roja_speed = 2) 
  (h2 : time = 4) 
  (h3 : final_distance = 20) : 
  ∃ other_speed : ℝ, 
    other_speed = 3 ∧ 
    final_distance = (roja_speed + other_speed) * time :=
by
  sorry

#check other_person_speed

end other_person_speed_l3420_342068


namespace cubic_function_extremum_value_l3420_342053

/-- A cubic function with an extremum at x = 1 and f(1) = 10 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_extremum_value (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → f a b 2 = 18 := by
  sorry

end cubic_function_extremum_value_l3420_342053


namespace complex_equation_solution_l3420_342033

theorem complex_equation_solution (x y : ℝ) :
  (x + y * Complex.I) / (3 - 2 * Complex.I) = 1 + Complex.I →
  Complex.im (x + y * Complex.I) = 1 ∧ Complex.abs (x + y * Complex.I) = Real.sqrt 26 := by
  sorry

end complex_equation_solution_l3420_342033


namespace quadratic_root_square_l3420_342070

theorem quadratic_root_square (a : ℚ) : 
  (∃ x y : ℚ, x^2 - (15/4)*x + a^3 = 0 ∧ y^2 - (15/4)*y + a^3 = 0 ∧ x = y^2) ↔ 
  (a = 3/2 ∨ a = -5/2) :=
sorry

end quadratic_root_square_l3420_342070


namespace turtleneck_profit_percentage_l3420_342072

/-- Calculates the profit percentage on turtleneck sweaters sold in February 
    given specific markup and discount conditions. -/
theorem turtleneck_profit_percentage :
  let initial_markup : ℝ := 0.20
  let new_year_markup : ℝ := 0.25
  let february_discount : ℝ := 0.09
  let first_price := 1 + initial_markup
  let second_price := first_price + new_year_markup * first_price
  let final_price := second_price * (1 - february_discount)
  let profit_percentage := final_price - 1
  profit_percentage = 0.365 := by sorry

end turtleneck_profit_percentage_l3420_342072


namespace sqrt_29_between_5_and_6_l3420_342062

theorem sqrt_29_between_5_and_6 : 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 := by
  sorry

end sqrt_29_between_5_and_6_l3420_342062


namespace brown_dog_weight_l3420_342024

/-- The weight of the brown dog -/
def brown_weight : ℝ := sorry

/-- The weight of the black dog -/
def black_weight : ℝ := brown_weight + 1

/-- The weight of the white dog -/
def white_weight : ℝ := 2 * brown_weight

/-- The weight of the grey dog -/
def grey_weight : ℝ := black_weight - 2

/-- The average weight of all dogs -/
def average_weight : ℝ := 5

theorem brown_dog_weight :
  (brown_weight + black_weight + white_weight + grey_weight) / 4 = average_weight →
  brown_weight = 4 := by sorry

end brown_dog_weight_l3420_342024


namespace average_of_five_quantities_l3420_342034

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 21.5) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 11 := by
  sorry

end average_of_five_quantities_l3420_342034


namespace last_digit_of_even_ten_digit_with_sum_89_l3420_342015

/-- A ten-digit integer -/
def TenDigitInt : Type := { n : ℕ // 1000000000 ≤ n ∧ n < 10000000000 }

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem last_digit_of_even_ten_digit_with_sum_89 (n : TenDigitInt) 
  (h_even : Even n.val)
  (h_sum : sum_of_digits n.val = 89) :
  n.val % 10 = 8 := by sorry

end last_digit_of_even_ten_digit_with_sum_89_l3420_342015


namespace intersection_M_N_l3420_342079

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| > 1}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x ≤ 3} := by sorry

end intersection_M_N_l3420_342079


namespace problem_statement_l3420_342069

theorem problem_statement (a b : ℝ) : (a - 1)^2 + |b + 2| = 0 → (a + b)^2023 = -1 := by
  sorry

end problem_statement_l3420_342069


namespace fermat_prime_l3420_342037

theorem fermat_prime (m : ℕ) (h : m > 0) :
  (2^(m+1) + 1) ∣ (3^(2^m) + 1) → Nat.Prime (2^(m+1) + 1) :=
by sorry

end fermat_prime_l3420_342037


namespace toms_profit_l3420_342091

/-- Calculates Tom's profit from lawn mowing and weed pulling -/
def calculate_profit (lawns_mowed : ℕ) (price_per_lawn : ℕ) (gas_expense : ℕ) (weed_pulling_income : ℕ) : ℕ :=
  lawns_mowed * price_per_lawn + weed_pulling_income - gas_expense

/-- Theorem: Tom's profit last month was $29 -/
theorem toms_profit :
  calculate_profit 3 12 17 10 = 29 := by
  sorry

end toms_profit_l3420_342091


namespace adult_ticket_cost_l3420_342049

/-- Proves that the cost of an adult ticket is $16 given the conditions of the problem -/
theorem adult_ticket_cost (child_ticket_cost : ℕ) (total_attendance : ℕ) 
  (total_revenue : ℕ) (child_attendance : ℕ) :
  child_ticket_cost = 9 →
  total_attendance = 24 →
  total_revenue = 258 →
  child_attendance = 18 →
  (total_attendance - child_attendance) * 16 + child_attendance * child_ticket_cost = total_revenue :=
by sorry

end adult_ticket_cost_l3420_342049


namespace evaluate_P_l3420_342001

-- Define the polynomial P(a)
def P (a : ℝ) : ℝ := (6 * a^2 - 14 * a + 5) * (3 * a - 4)

-- Theorem stating the values of P(4/3) and P(2)
theorem evaluate_P : P (4/3) = 0 ∧ P 2 = 2 := by sorry

end evaluate_P_l3420_342001


namespace bing_dwen_dwen_sales_equation_l3420_342022

/-- The sales equation for Bing Dwen Dwen mascot -/
theorem bing_dwen_dwen_sales_equation (x : ℝ) : 
  (5000 : ℝ) * (1 + x) + (5000 : ℝ) * (1 + x)^2 = 22500 ↔ 
  (∃ (sales_feb4 sales_feb5 sales_feb6 : ℝ),
    sales_feb4 = 5000 ∧
    sales_feb5 = sales_feb4 * (1 + x) ∧
    sales_feb6 = sales_feb5 * (1 + x) ∧
    sales_feb5 + sales_feb6 = 22500) :=
by
  sorry

end bing_dwen_dwen_sales_equation_l3420_342022


namespace some_fire_breathing_mystical_l3420_342054

-- Define the sets
variable (U : Type) -- Universe set
variable (Dragon MysticalCreature FireBreathingCreature : Set U)

-- Define the conditions
variable (h1 : Dragon ⊆ FireBreathingCreature)
variable (h2 : ∃ x, x ∈ MysticalCreature ∧ x ∈ Dragon)

-- Theorem to prove
theorem some_fire_breathing_mystical :
  ∃ x, x ∈ FireBreathingCreature ∧ x ∈ MysticalCreature :=
by
  sorry


end some_fire_breathing_mystical_l3420_342054


namespace m_less_than_n_l3420_342086

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence, M and N are defined as follows -/
def M (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.a n * seq.a (n + 3)

def N (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.a (n + 1) * seq.a (n + 2)

/-- For any arithmetic sequence with non-zero common difference, M < N -/
theorem m_less_than_n (seq : ArithmeticSequence) (n : ℕ) : M seq n < N seq n := by
  sorry

end m_less_than_n_l3420_342086


namespace closed_polygonal_line_even_segments_l3420_342087

/-- Represents a segment of the polygonal line -/
structure Segment where
  x : Int
  y : Int

/-- Represents a closed polygonal line on a grid -/
structure ClosedPolygonalLine where
  segments : List Segment
  is_closed : segments.length > 0
  same_length : ∀ s ∈ segments, s.x^2 + s.y^2 = 1
  on_grid : ∀ s ∈ segments, s.x = 0 ∨ s.y = 0

/-- The main theorem stating that the number of segments in a closed polygonal line is even -/
theorem closed_polygonal_line_even_segments (p : ClosedPolygonalLine) : 
  Even p.segments.length := by
  sorry

end closed_polygonal_line_even_segments_l3420_342087


namespace denver_temperature_peak_l3420_342071

/-- The temperature function modeling a day in Denver, CO -/
def temperature (t : ℝ) : ℝ := -2 * t^2 + 24 * t + 100

/-- Theorem stating that 6 is the smallest non-negative real solution to the temperature equation -/
theorem denver_temperature_peak :
  (∀ t : ℝ, t ≥ 0 → temperature t = 148 → t ≥ 6) ∧
  temperature 6 = 148 := by
  sorry

end denver_temperature_peak_l3420_342071


namespace factor_expression_l3420_342027

theorem factor_expression (c : ℝ) : 180 * c^2 + 36 * c = 36 * c * (5 * c + 1) := by
  sorry

end factor_expression_l3420_342027


namespace problem_solution_l3420_342073

theorem problem_solution (x y : ℝ) (h1 : x^(3*y) = 16) (h2 : x = 16) : y = 1/3 := by
  sorry

end problem_solution_l3420_342073


namespace fraction_problem_l3420_342000

theorem fraction_problem (x y : ℚ) (h1 : x + y = 14/15) (h2 : x * y = 1/10) :
  min x y = 1/5 := by
  sorry

end fraction_problem_l3420_342000


namespace smallest_multiple_l3420_342043

theorem smallest_multiple (x : ℕ) : x = 16 ↔ (
  x > 0 ∧
  450 * x % 800 = 0 ∧
  ∀ y : ℕ, y > 0 → y < x → 450 * y % 800 ≠ 0
) := by sorry

end smallest_multiple_l3420_342043


namespace sufficient_not_necessary_conditions_l3420_342040

theorem sufficient_not_necessary_conditions (a b : ℝ) :
  (∀ (a b : ℝ), a + b > 2 → a + b > 0) ∧
  (∀ (a b : ℝ), (a > 0 ∧ b > 0) → a + b > 0) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a + b > 2)) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a > 0 ∧ b > 0)) ∧
  (∃ (a b : ℝ), ¬(ab > 0) ∧ a + b > 0) ∧
  (∃ (a b : ℝ), ¬(a > 0 ∨ b > 0) ∧ a + b > 0) :=
by sorry

end sufficient_not_necessary_conditions_l3420_342040


namespace parabola_properties_l3420_342088

-- Define the parabola and its properties
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_a_neg : a < 0
  h_m_bounds : 1 < m ∧ m < 2
  h_passes_through : a * (-1)^2 + b * (-1) + c = 0 ∧ a * m^2 + b * m + c = 0

-- Theorem statements
theorem parabola_properties (p : Parabola) :
  (p.b > 0) ∧
  (∀ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ < x₂ → x₁ + x₂ > 1 → 
    p.a * x₁^2 + p.b * x₁ + p.c = y₁ → 
    p.a * x₂^2 + p.b * x₂ + p.c = y₂ → 
    y₁ > y₂) ∧
  (p.a ≤ -1 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ p.a * x₁^2 + p.b * x₁ + p.c = 1 ∧ p.a * x₂^2 + p.b * x₂ + p.c = 1) :=
by sorry

end parabola_properties_l3420_342088


namespace area_ADC_approx_l3420_342098

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle bisector AD
def angleBisector (t : Triangle) : ℝ × ℝ := sorry

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop := sorry
def hasAngleBisector (t : Triangle) : Prop := sorry
def sideAB (t : Triangle) : ℝ := sorry
def sideBC (t : Triangle) : ℝ := sorry
def sideAC (t : Triangle) : ℝ := sorry

-- Define the area calculation function
def areaADC (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem area_ADC_approx (t : Triangle) 
  (h1 : isRightTriangle t)
  (h2 : hasAngleBisector t)
  (h3 : sideAB t = 80)
  (h4 : ∃ x, sideBC t = x ∧ sideAC t = 2*x - 10) :
  ∃ ε > 0, |areaADC t - 949| < ε :=
sorry

end area_ADC_approx_l3420_342098


namespace janes_garden_area_l3420_342019

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  total_posts : ℕ
  post_spacing : ℕ
  long_side_posts : ℕ
  short_side_posts : ℕ

/-- Calculates the area of the garden -/
def garden_area (g : Garden) : ℕ :=
  (g.short_side_posts - 1) * g.post_spacing * (g.long_side_posts - 1) * g.post_spacing

/-- Theorem stating the area of Jane's garden -/
theorem janes_garden_area :
  ∀ g : Garden,
    g.total_posts = 24 →
    g.post_spacing = 3 →
    g.long_side_posts = 3 * g.short_side_posts →
    g.total_posts = 2 * (g.short_side_posts + g.long_side_posts) - 4 →
    garden_area g = 144 := by
  sorry


end janes_garden_area_l3420_342019


namespace point_on_line_l3420_342099

/-- Given that point A (3, a) lies on the line 2x + y - 7 = 0, prove that a = 1 -/
theorem point_on_line (a : ℝ) : 2 * 3 + a - 7 = 0 → a = 1 := by
  sorry

end point_on_line_l3420_342099


namespace painting_frame_ratio_l3420_342012

theorem painting_frame_ratio {x l : ℝ} (h_positive : x > 0 ∧ l > 0) 
  (h_area_equality : (x + 2*l) * ((3/2)*x + 2*l) = 2 * (x * (3/2)*x)) :
  (x + 2*l) / ((3/2)*x + 2*l) = 3/4 := by
  sorry

end painting_frame_ratio_l3420_342012


namespace train_length_l3420_342065

/-- The length of a train given its speed, bridge length, and time to cross the bridge. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  bridge_length = 265 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 110 := by
  sorry

#check train_length

end train_length_l3420_342065


namespace adams_stairs_l3420_342008

theorem adams_stairs (total_steps : ℕ) (steps_left : ℕ) (steps_climbed : ℕ) : 
  total_steps = 96 → steps_left = 22 → steps_climbed = total_steps - steps_left → steps_climbed = 74 := by
  sorry

end adams_stairs_l3420_342008


namespace double_inequality_abc_l3420_342047

theorem double_inequality_abc (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b ≤ 1) (hbc : b + c ≤ 1) (hca : c + a ≤ 1) :
  a^2 + b^2 + c^2 ≤ a + b + c - a*b - b*c - c*a ∧ 
  a + b + c - a*b - b*c - c*a ≤ (1 + a^2 + b^2 + c^2) / 2 :=
by sorry

end double_inequality_abc_l3420_342047


namespace board_numbers_divisibility_l3420_342023

theorem board_numbers_divisibility (X Y N A B : ℤ) 
  (sum_eq : X + Y = N) 
  (tanya_div : (A * X + B * Y) % N = 0) : 
  (B * X + A * Y) % N = 0 := by
  sorry

end board_numbers_divisibility_l3420_342023


namespace evaluate_expression_l3420_342045

theorem evaluate_expression : ((3^1 - 2 + 6^2 - 0)⁻¹ * 3 : ℚ) = 3 / 37 := by
  sorry

end evaluate_expression_l3420_342045


namespace min_value_expression_l3420_342076

theorem min_value_expression (a θ : ℝ) : 
  (a - 2 * Real.cos θ)^2 + (a - 5 * Real.sqrt 2 - 2 * Real.sin θ)^2 ≥ 9 ∧
  ∃ a θ : ℝ, (a - 2 * Real.cos θ)^2 + (a - 5 * Real.sqrt 2 - 2 * Real.sin θ)^2 = 9 :=
by sorry

end min_value_expression_l3420_342076


namespace min_value_theorem_l3420_342050

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b ∧ g x = a * x + c

/-- The theorem statement -/
theorem min_value_theorem (funcs : ParallelLinearFunctions) 
  (h : ∃ (x : ℝ), ∀ (y : ℝ), (funcs.f y)^2 + 8 * funcs.g y ≥ (funcs.f x)^2 + 8 * funcs.g x)
  (min_value : (funcs.f x)^2 + 8 * funcs.g x = -29) :
  ∃ (z : ℝ), ∀ (w : ℝ), (funcs.g w)^2 + 8 * funcs.f w ≥ (funcs.g z)^2 + 8 * funcs.f z ∧ 
  (funcs.g z)^2 + 8 * funcs.f z = -3 :=
sorry

end min_value_theorem_l3420_342050


namespace solve_fish_problem_l3420_342007

def fish_problem (total_spent : ℕ) (cost_per_fish : ℕ) (fish_for_dog : ℕ) : Prop :=
  let total_fish : ℕ := total_spent / cost_per_fish
  let fish_for_cat : ℕ := total_fish - fish_for_dog
  (fish_for_cat : ℚ) / fish_for_dog = 1 / 2

theorem solve_fish_problem :
  fish_problem 240 4 40 := by
  sorry

end solve_fish_problem_l3420_342007


namespace consecutive_integers_square_sum_l3420_342058

theorem consecutive_integers_square_sum (a b : ℤ) (h : b = a + 1) :
  a^2 + b^2 + (a*b)^2 = (a*b + 1)^2 := by
  sorry

end consecutive_integers_square_sum_l3420_342058


namespace number_and_square_relationship_l3420_342063

theorem number_and_square_relationship (n : ℕ) (h : n = 8) : n^2 + n = 72 := by
  sorry

end number_and_square_relationship_l3420_342063


namespace average_income_problem_l3420_342039

theorem average_income_problem (M N O : ℕ) : 
  (M + N) / 2 = 5050 →
  (N + O) / 2 = 6250 →
  M = 4000 →
  (M + O) / 2 = 5200 := by
  sorry

end average_income_problem_l3420_342039


namespace festival_attendance_ratio_l3420_342032

/-- Represents a 3-day music festival attendance --/
structure FestivalAttendance where
  total : ℕ
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- The conditions of the festival attendance --/
def festivalConditions (f : FestivalAttendance) : Prop :=
  f.total = 2700 ∧
  f.day2 = f.day1 / 2 ∧
  f.day2 = 300 ∧
  f.total = f.day1 + f.day2 + f.day3

/-- The theorem stating the ratio of third day to first day attendance --/
theorem festival_attendance_ratio (f : FestivalAttendance) 
  (h : festivalConditions f) : f.day3 = 3 * f.day1 := by
  sorry

#check festival_attendance_ratio

end festival_attendance_ratio_l3420_342032


namespace star_example_l3420_342026

-- Define the * operation
def star (a b : ℕ) : ℕ := a + 2 * b

-- State the theorem
theorem star_example : star (star 2 3) 4 = 16 := by sorry

end star_example_l3420_342026
