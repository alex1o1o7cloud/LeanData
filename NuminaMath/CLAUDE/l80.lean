import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l80_8072

/-- Given an ellipse mx^2 + ny^2 = 1 intersecting with a line x + y - 1 = 0,
    if the slope of the line passing through the origin and the midpoint of
    the intersection points is √2/2, then n/m = √2 -/
theorem ellipse_line_intersection (m n : ℝ) :
  (∃ A B : ℝ × ℝ,
    m * A.1^2 + n * A.2^2 = 1 ∧
    m * B.1^2 + n * B.2^2 = 1 ∧
    A.1 + A.2 = 1 ∧
    B.1 + B.2 = 1 ∧
    (A ≠ B) ∧
    ((A.2 + B.2)/2) / ((A.1 + B.1)/2) = Real.sqrt 2 / 2) →
  n / m = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l80_8072


namespace NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l80_8060

-- Define the vectors
def m (a : ℝ) : ℝ × ℝ := (a, a - 4)
def n (b : ℝ) : ℝ × ℝ := (b, 1 - b)

-- Define parallelism condition
def are_parallel (a b : ℝ) : Prop :=
  a * (1 - b) = b * (a - 4)

theorem min_sum_of_parallel_vectors (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_parallel : are_parallel a b) :
  a + b ≥ 9/2 ∧ (a + b = 9/2 ↔ a = 4 ∧ b = 2) := by
  sorry


end NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l80_8060


namespace NUMINAMATH_CALUDE_total_new_people_value_l80_8015

/-- The number of people born in the country last year -/
def people_born : ℕ := 90171

/-- The number of people who immigrated to the country last year -/
def people_immigrated : ℕ := 16320

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := people_born + people_immigrated

/-- Theorem stating that the total number of new people is 106491 -/
theorem total_new_people_value : total_new_people = 106491 := by
  sorry

end NUMINAMATH_CALUDE_total_new_people_value_l80_8015


namespace NUMINAMATH_CALUDE_sum_six_consecutive_integers_l80_8006

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_integers_l80_8006


namespace NUMINAMATH_CALUDE_train_length_calculation_l80_8008

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 15 → ∃ (length : ℝ), abs (length - 500) < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l80_8008


namespace NUMINAMATH_CALUDE_f_properties_l80_8086

/-- The function f(x) = x³ - 3mx + n --/
def f (m n x : ℝ) : ℝ := x^3 - 3*m*x + n

/-- Theorem stating the values of m and n, and the extrema in [0,3] --/
theorem f_properties (m n : ℝ) (hm : m > 0) 
  (hmax : ∃ x, ∀ y, f m n y ≤ f m n x)
  (hmin : ∃ x, ∀ y, f m n x ≤ f m n y)
  (hmax_val : ∃ x, f m n x = 6)
  (hmin_val : ∃ x, f m n x = 2) :
  m = 1 ∧ n = 4 ∧ 
  (∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, f 1 4 y ≤ f 1 4 x) ∧
  (∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, f 1 4 x ≤ f 1 4 y) ∧
  (∃ x ∈ Set.Icc 0 3, f 1 4 x = 2) ∧
  (∃ x ∈ Set.Icc 0 3, f 1 4 x = 22) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l80_8086


namespace NUMINAMATH_CALUDE_light_travel_distance_l80_8016

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℕ := 50

/-- Theorem stating the distance light travels in 50 years -/
theorem light_travel_distance : light_year_distance * years = 2935 * (10 : ℝ)^11 := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l80_8016


namespace NUMINAMATH_CALUDE_minimum_score_for_target_average_l80_8002

def test_count : ℕ := 6
def max_score : ℕ := 100
def target_average : ℕ := 85
def scores : List ℕ := [82, 70, 88]

theorem minimum_score_for_target_average :
  ∃ (x y z : ℕ), 
    x ≤ max_score ∧ y ≤ max_score ∧ z ≤ max_score ∧
    (scores.sum + x + y + z) / test_count = target_average ∧
    (∀ w, w < 70 → (scores.sum + w + max_score + max_score) / test_count < target_average) := by
  sorry

end NUMINAMATH_CALUDE_minimum_score_for_target_average_l80_8002


namespace NUMINAMATH_CALUDE_lulu_poptarts_count_l80_8023

/-- Represents the number of pastries baked by Lola and Lulu -/
structure PastryCounts where
  lola_cupcakes : ℕ
  lola_poptarts : ℕ
  lola_pies : ℕ
  lulu_cupcakes : ℕ
  lulu_poptarts : ℕ
  lulu_pies : ℕ

/-- The total number of pastries baked by Lola and Lulu -/
def total_pastries (counts : PastryCounts) : ℕ :=
  counts.lola_cupcakes + counts.lola_poptarts + counts.lola_pies +
  counts.lulu_cupcakes + counts.lulu_poptarts + counts.lulu_pies

/-- Theorem stating that Lulu baked 12 pop tarts -/
theorem lulu_poptarts_count (counts : PastryCounts) 
  (h1 : counts.lola_cupcakes = 13)
  (h2 : counts.lola_poptarts = 10)
  (h3 : counts.lola_pies = 8)
  (h4 : counts.lulu_cupcakes = 16)
  (h5 : counts.lulu_pies = 14)
  (h6 : total_pastries counts = 73) :
  counts.lulu_poptarts = 12 := by
  sorry

end NUMINAMATH_CALUDE_lulu_poptarts_count_l80_8023


namespace NUMINAMATH_CALUDE_find_number_l80_8082

theorem find_number : ∃! x : ℝ, 7 * x + 37 = 100 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l80_8082


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_property_l80_8062

/-- A hyperbola in a 2D plane -/
structure Hyperbola where
  -- Add necessary fields to define a hyperbola
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Foci of the hyperbola -/
def foci (h : Hyperbola) : (Point × Point) := sorry

/-- Check if a point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop := sorry

/-- Diameter of the director circle -/
def director_circle_diameter (h : Hyperbola) : ℝ := sorry

theorem hyperbola_focal_distance_property (h : Hyperbola) (p : Point) :
  is_on_hyperbola h p →
  let (f1, f2) := foci h
  |distance p f1 - distance p f2| = director_circle_diameter h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_property_l80_8062


namespace NUMINAMATH_CALUDE_angle_equivalence_l80_8056

-- Define α in degrees
def α : ℝ := 2010

-- Theorem statement
theorem angle_equivalence (α : ℝ) : 
  -- Part 1: Rewrite α in the form θ + 2kπ
  (α * π / 180 = 7 * π / 6 + 10 * π) ∧
  -- Part 2: Find equivalent angles in [-5π, 0)
  (∀ β : ℝ, -5 * π ≤ β ∧ β < 0 ∧ 
    (∃ k : ℤ, β = 7 * π / 6 + 2 * k * π) ↔ 
    (β = -29 * π / 6 ∨ β = -17 * π / 6 ∨ β = -5 * π / 6)) :=
by sorry

end NUMINAMATH_CALUDE_angle_equivalence_l80_8056


namespace NUMINAMATH_CALUDE_third_candidate_votes_correct_l80_8079

/-- The number of votes received by the third candidate in an election with three candidates,
    where two candidates received 7636 and 11628 votes respectively,
    and the winning candidate got 54.336448598130836% of the total votes. -/
def third_candidate_votes : ℕ :=
  let total_votes : ℕ := 7636 + 11628 + 2136
  let winning_votes : ℕ := 11628
  let winning_percentage : ℚ := 54336448598130836 / 100000000000000000
  2136

theorem third_candidate_votes_correct :
  let total_votes : ℕ := 7636 + 11628 + third_candidate_votes
  let winning_votes : ℕ := 11628
  let winning_percentage : ℚ := 54336448598130836 / 100000000000000000
  (winning_votes : ℚ) / (total_votes : ℚ) = winning_percentage :=
by sorry

#eval third_candidate_votes

end NUMINAMATH_CALUDE_third_candidate_votes_correct_l80_8079


namespace NUMINAMATH_CALUDE_reflection_of_line_l80_8017

/-- Given a line with equation 2x + 3y - 5 = 0, its reflection about the line y = x
    is the line with equation 3x + 2y - 5 = 0 -/
theorem reflection_of_line :
  let original_line : ℝ → ℝ → Prop := λ x y ↦ 2*x + 3*y - 5 = 0
  let reflection_axis : ℝ → ℝ → Prop := λ x y ↦ y = x
  let reflected_line : ℝ → ℝ → Prop := λ x y ↦ 3*x + 2*y - 5 = 0
  ∀ (x y : ℝ), original_line x y ↔ reflected_line y x :=
by sorry

end NUMINAMATH_CALUDE_reflection_of_line_l80_8017


namespace NUMINAMATH_CALUDE_acute_triangle_side_range_l80_8088

-- Define an acute triangle with sides 3, 4, and a
def is_acute_triangle (a : ℝ) : Prop :=
  a > 0 ∧ 3 > 0 ∧ 4 > 0 ∧
  a + 3 > 4 ∧ a + 4 > 3 ∧ 3 + 4 > a ∧
  a^2 < 3^2 + 4^2 ∧ 3^2 < a^2 + 4^2 ∧ 4^2 < a^2 + 3^2

-- Theorem statement
theorem acute_triangle_side_range :
  ∀ a : ℝ, is_acute_triangle a → Real.sqrt 7 < a ∧ a < 5 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_side_range_l80_8088


namespace NUMINAMATH_CALUDE_work_completion_time_l80_8005

/-- 
Given:
- A can do a work in 14 days
- A and B together can do the same work in 10 days

Prove that B can do the work alone in 35 days
-/
theorem work_completion_time (work : ℝ) (a_rate b_rate : ℝ) 
  (h1 : a_rate = work / 14)
  (h2 : a_rate + b_rate = work / 10) :
  b_rate = work / 35 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l80_8005


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l80_8004

theorem fractional_inequality_solution_set (x : ℝ) : 
  (x - 1) / (2 * x + 3) > 1 ↔ -4 < x ∧ x < -3/2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l80_8004


namespace NUMINAMATH_CALUDE_lawsuit_probability_comparison_l80_8058

theorem lawsuit_probability_comparison :
  let p1_win : ℝ := 0.30
  let p2_win : ℝ := 0.50
  let p3_win : ℝ := 0.40
  let p4_win : ℝ := 0.25
  
  let p1_lose : ℝ := 1 - p1_win
  let p2_lose : ℝ := 1 - p2_win
  let p3_lose : ℝ := 1 - p3_win
  let p4_lose : ℝ := 1 - p4_win
  
  let p_win_all : ℝ := p1_win * p2_win * p3_win * p4_win
  let p_lose_all : ℝ := p1_lose * p2_lose * p3_lose * p4_lose
  
  (p_lose_all - p_win_all) / p_win_all = 9.5
:= by sorry

end NUMINAMATH_CALUDE_lawsuit_probability_comparison_l80_8058


namespace NUMINAMATH_CALUDE_least_four_digit_9_heavy_l80_8098

def is_9_heavy (n : ℕ) : Prop := n % 9 = 6

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem least_four_digit_9_heavy : 
  (∀ m : ℕ, is_four_digit m → is_9_heavy m → 1005 ≤ m) ∧ 
  is_four_digit 1005 ∧ 
  is_9_heavy 1005 := by sorry

end NUMINAMATH_CALUDE_least_four_digit_9_heavy_l80_8098


namespace NUMINAMATH_CALUDE_triangle_altitude_equals_twice_base_l80_8028

/-- Given a square with side length x and a triangle with base x, 
    if their areas are equal, then the altitude of the triangle is 2x. -/
theorem triangle_altitude_equals_twice_base (x : ℝ) (h : x > 0) : 
  x^2 = (1/2) * x * (2*x) := by sorry

end NUMINAMATH_CALUDE_triangle_altitude_equals_twice_base_l80_8028


namespace NUMINAMATH_CALUDE_derek_water_addition_l80_8044

/-- The amount of water Derek added to the bucket -/
def water_added (initial final : ℝ) : ℝ := final - initial

theorem derek_water_addition (initial final : ℝ) 
  (h1 : initial = 3)
  (h2 : final = 9.8) :
  water_added initial final = 6.8 := by
  sorry

end NUMINAMATH_CALUDE_derek_water_addition_l80_8044


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l80_8039

/-- Represents the number of frogs of each color -/
structure FrogCounts where
  green : Nat
  red : Nat
  blue : Nat

/-- Represents the arrangement rules for frogs -/
structure FrogRules where
  green_red_adjacent : Bool
  green_blue_adjacent : Bool
  red_blue_adjacent : Bool
  blue_blue_adjacent : Bool

/-- Calculates the number of valid frog arrangements -/
def countFrogArrangements (counts : FrogCounts) (rules : FrogRules) : Nat :=
  sorry

/-- The main theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  let counts : FrogCounts := ⟨2, 3, 2⟩
  let rules : FrogRules := ⟨false, true, true, true⟩
  countFrogArrangements counts rules = 72 := by sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l80_8039


namespace NUMINAMATH_CALUDE_potato_difference_l80_8075

/-- The number of potato wedges Cynthia makes -/
def x : ℕ := 8 * 13

/-- The number of potatoes used for french fries or potato chips -/
def k : ℕ := (67 - 13) / 2

/-- The number of potato chips Cynthia makes -/
def z : ℕ := 20 * k

/-- The difference between the number of potato chips and potato wedges -/
def d : ℤ := z - x

theorem potato_difference : d = 436 := by
  sorry

end NUMINAMATH_CALUDE_potato_difference_l80_8075


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l80_8043

/-- Given an arithmetic sequence with first term 5, second term 12, and last term 40,
    the sum of the two terms immediately preceding 40 is 59. -/
theorem arithmetic_sequence_sum (a : ℕ → ℕ) : 
  a 0 = 5 → a 1 = 12 → 
  (∃ n : ℕ, a n = 40 ∧ ∀ k < n, a k < 40) →
  (∀ i j k : ℕ, i < j → j < k → a j - a i = a k - a j) →
  (∃ m : ℕ, a m + a (m + 1) = 59 ∧ a (m + 2) = 40) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l80_8043


namespace NUMINAMATH_CALUDE_teacher_number_game_l80_8040

theorem teacher_number_game (x : ℝ) : 
  let max_result := 2 * (3 * (x + 1))
  let lisa_result := 2 * ((max_result / 2) - 1)
  lisa_result = 2 * x + 2 := by sorry

end NUMINAMATH_CALUDE_teacher_number_game_l80_8040


namespace NUMINAMATH_CALUDE_circle_triangle_area_constraint_l80_8055

/-- The range of r for which there are exactly two points on the circle
    (x-2)^2 + y^2 = r^2 that form triangles with area 4 with given points A and B -/
theorem circle_triangle_area_constraint (r : ℝ) : 
  r > 0 →
  (∃! M N : ℝ × ℝ, 
    (M.1 - 2)^2 + M.2^2 = r^2 ∧
    (N.1 - 2)^2 + N.2^2 = r^2 ∧
    abs ((M.1 + 3) * (-2) - (M.2 - 0) * (-2)) / 2 = 4 ∧
    abs ((N.1 + 3) * (-2) - (N.2 - 0) * (-2)) / 2 = 4) →
  r ∈ Set.Ioo (Real.sqrt 2 / 2) (9 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_triangle_area_constraint_l80_8055


namespace NUMINAMATH_CALUDE_cow_ratio_l80_8099

/-- Proves the ratio of black cows to total cows given the conditions -/
theorem cow_ratio (total : ℕ) (non_black : ℕ) (black : ℕ) : 
  total = 18 → 
  non_black = 4 → 
  black = (total / 2) + 5 → 
  black + non_black = total →
  (black : ℚ) / (total : ℚ) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cow_ratio_l80_8099


namespace NUMINAMATH_CALUDE_connie_marbles_l80_8093

/-- Calculates the remaining marbles after giving some away. -/
def remaining_marbles (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proves that Connie has 3 marbles left after giving away 70 from her initial 73 marbles. -/
theorem connie_marbles : remaining_marbles 73 70 = 3 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l80_8093


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l80_8024

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l80_8024


namespace NUMINAMATH_CALUDE_green_fruits_vs_red_peaches_green_peaches_vs_yellow_apples_l80_8085

/-- Represents the number of red peaches in the basket -/
def red_peaches : ℕ := 5

/-- Represents the number of green peaches in the basket -/
def green_peaches : ℕ := 11

/-- Represents the number of yellow apples in the basket -/
def yellow_apples : ℕ := 8

/-- Represents the number of green apples in the basket -/
def green_apples : ℕ := 15

/-- Theorem stating the difference between green fruits and red peaches -/
theorem green_fruits_vs_red_peaches : 
  green_peaches + green_apples - red_peaches = 21 := by sorry

/-- Theorem stating the difference between green peaches and yellow apples -/
theorem green_peaches_vs_yellow_apples : 
  green_peaches - yellow_apples = 3 := by sorry

end NUMINAMATH_CALUDE_green_fruits_vs_red_peaches_green_peaches_vs_yellow_apples_l80_8085


namespace NUMINAMATH_CALUDE_unique_solution_abs_equation_l80_8089

theorem unique_solution_abs_equation :
  ∃! y : ℝ, y * |y| = -3 * y + 5 :=
by
  -- The unique solution is (-3 + √29) / 2
  use (-3 + Real.sqrt 29) / 2
  sorry

end NUMINAMATH_CALUDE_unique_solution_abs_equation_l80_8089


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_five_l80_8041

theorem polynomial_value_at_negative_five (a b c : ℝ) : 
  (5^5 * a + 5^3 * b + 5 * c + 2 = 8) → 
  ((-5)^5 * a + (-5)^3 * b + (-5) * c - 3 = -9) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_five_l80_8041


namespace NUMINAMATH_CALUDE_probability_of_science_second_draw_l80_8077

/-- Represents the type of questions --/
inductive QuestionType
| Science
| LiberalArts

/-- Represents the state of the questions after the first draw --/
structure QuestionState :=
  (total : Nat)
  (science : Nat)
  (liberal_arts : Nat)

/-- The initial state of questions --/
def initial_state : QuestionState :=
  ⟨5, 3, 2⟩

/-- The state after drawing a science question --/
def after_first_draw (s : QuestionState) : QuestionState :=
  ⟨s.total - 1, s.science - 1, s.liberal_arts⟩

/-- The probability of drawing a science question on the second draw --/
def prob_science_second_draw (s : QuestionState) : Rat :=
  s.science / s.total

theorem probability_of_science_second_draw :
  prob_science_second_draw (after_first_draw initial_state) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_science_second_draw_l80_8077


namespace NUMINAMATH_CALUDE_H2O_formation_l80_8069

-- Define the molecules and their molar quantities
def HCl_moles : ℚ := 2
def CaCO3_moles : ℚ := 1

-- Define the balanced equation coefficients
def HCl_coeff : ℚ := 2
def CaCO3_coeff : ℚ := 1
def H2O_coeff : ℚ := 1

-- Define the function to calculate the amount of H2O formed
def H2O_formed (HCl : ℚ) (CaCO3 : ℚ) : ℚ :=
  min (HCl / HCl_coeff) (CaCO3 / CaCO3_coeff) * H2O_coeff

-- State the theorem
theorem H2O_formation :
  H2O_formed HCl_moles CaCO3_moles = 1 := by
  sorry

end NUMINAMATH_CALUDE_H2O_formation_l80_8069


namespace NUMINAMATH_CALUDE_driver_net_pay_driver_net_pay_result_l80_8032

/-- Calculate the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay (travel_time : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (earnings_per_mile : ℝ) (gas_price : ℝ) : ℝ :=
  let total_distance := travel_time * speed
  let gas_used := total_distance / fuel_efficiency
  let total_earnings := earnings_per_mile * total_distance
  let gas_cost := gas_price * gas_used
  let net_earnings := total_earnings - gas_cost
  let net_rate := net_earnings / travel_time
  net_rate

/-- The driver's net rate of pay is $39.75 per hour --/
theorem driver_net_pay_result : 
  driver_net_pay 3 75 25 0.65 3 = 39.75 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_driver_net_pay_result_l80_8032


namespace NUMINAMATH_CALUDE_seating_arrangements_l80_8081

-- Define the number of seats, adults, and children
def numSeats : ℕ := 6
def numAdults : ℕ := 3
def numChildren : ℕ := 3

-- Define a function to calculate permutations
def permutations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

-- Theorem statement
theorem seating_arrangements :
  2 * (permutations numAdults numAdults) * (permutations numChildren numChildren) = 72 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l80_8081


namespace NUMINAMATH_CALUDE_intersection_dot_product_l80_8009

/-- Given a line and a parabola that intersect at points A and B, 
    and the focus of the parabola F, prove that the dot product 
    of vectors FA and FB is -11. -/
theorem intersection_dot_product 
  (A B : ℝ × ℝ) 
  (hA : A.2 = 2 * A.1 - 2 ∧ A.2^2 = 8 * A.1) 
  (hB : B.2 = 2 * B.1 - 2 ∧ B.2^2 = 8 * B.1) 
  (hAB_distinct : A ≠ B) : 
  let F : ℝ × ℝ := (2, 0)
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = -11 := by
  sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l80_8009


namespace NUMINAMATH_CALUDE_first_car_speed_l80_8071

/-- 
Given two cars starting from opposite ends of a highway, this theorem proves
that the speed of the first car is 25 mph under the given conditions.
-/
theorem first_car_speed 
  (highway_length : ℝ) 
  (second_car_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : highway_length = 175) 
  (h2 : second_car_speed = 45) 
  (h3 : meeting_time = 2.5) :
  ∃ (first_car_speed : ℝ), 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    first_car_speed = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_first_car_speed_l80_8071


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l80_8036

/-- Given three lines that intersect at one point, prove the value of a -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃! p : ℝ × ℝ, a * p.1 + 2 * p.2 + 8 = 0 ∧ 
                  4 * p.1 + 3 * p.2 = 10 ∧ 
                  2 * p.1 - p.2 = 10) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l80_8036


namespace NUMINAMATH_CALUDE_bedroom_size_calculation_l80_8095

theorem bedroom_size_calculation (total_area : ℝ) (difference : ℝ) :
  total_area = 300 →
  difference = 60 →
  ∃ (smaller_room : ℝ),
    smaller_room + (smaller_room + difference) = total_area ∧
    smaller_room = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_bedroom_size_calculation_l80_8095


namespace NUMINAMATH_CALUDE_sum_of_two_integers_l80_8080

theorem sum_of_two_integers (x y : ℤ) : x = 32 → y = 2 * x → x + y = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_integers_l80_8080


namespace NUMINAMATH_CALUDE_fair_rides_calculation_fair_rides_proof_l80_8070

/-- Calculates the number of rides taken by each person at a fair given specific conditions. -/
theorem fair_rides_calculation (entrance_fee_under_18 : ℚ) (ride_cost : ℚ) 
  (total_spent : ℚ) (num_people : ℕ) : ℚ :=
  let entrance_fee_18_plus := entrance_fee_under_18 * (1 + 1/5)
  let total_entrance_fee := entrance_fee_18_plus + 2 * entrance_fee_under_18
  let rides_cost := total_spent - total_entrance_fee
  let total_rides := rides_cost / ride_cost
  total_rides / num_people

/-- Proves that under the given conditions, each person took 3 rides. -/
theorem fair_rides_proof :
  fair_rides_calculation 5 (1/2) (41/2) 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fair_rides_calculation_fair_rides_proof_l80_8070


namespace NUMINAMATH_CALUDE_parking_savings_yearly_parking_savings_l80_8050

theorem parking_savings : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun weekly_rate monthly_rate weeks_per_year months_per_year savings =>
    weekly_rate * weeks_per_year - monthly_rate * months_per_year = savings

/-- Proof of yearly savings when renting monthly instead of weekly --/
theorem yearly_parking_savings : parking_savings 10 40 52 12 40 := by
  sorry

end NUMINAMATH_CALUDE_parking_savings_yearly_parking_savings_l80_8050


namespace NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l80_8045

theorem children_neither_happy_nor_sad 
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (neither_happy_nor_sad_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : boys = 16)
  (h5 : girls = 44)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neither_happy_nor_sad_boys = 4)
  : total_children - happy_children - sad_children = 20 := by
  sorry

end NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l80_8045


namespace NUMINAMATH_CALUDE_max_intersections_l80_8087

/-- Represents a polynomial of degree 5 or less -/
def Polynomial5 := Fin 6 → ℝ

/-- The set of ten 5-degree polynomials -/
def TenPolynomials := Fin 10 → Polynomial5

/-- A linear function representing an arithmetic sequence -/
def ArithmeticSequence := ℝ → ℝ

/-- The number of intersections between a polynomial and a linear function -/
def intersections (p : Polynomial5) (f : ArithmeticSequence) : ℕ :=
  sorry

/-- The total number of intersections between ten polynomials and a linear function -/
def totalIntersections (polynomials : TenPolynomials) (f : ArithmeticSequence) : ℕ :=
  sorry

theorem max_intersections (polynomials : TenPolynomials) (f : ArithmeticSequence) :
  totalIntersections polynomials f ≤ 50 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_l80_8087


namespace NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l80_8026

def required_average : ℝ := 85
def num_quarters : ℕ := 4
def first_quarter_score : ℝ := 84
def second_quarter_score : ℝ := 80
def third_quarter_score : ℝ := 78

theorem minimum_fourth_quarter_score :
  let total_required := required_average * num_quarters
  let current_total := first_quarter_score + second_quarter_score + third_quarter_score
  let minimum_score := total_required - current_total
  minimum_score = 98 := by sorry

end NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l80_8026


namespace NUMINAMATH_CALUDE_flying_scotsman_norwich_difference_l80_8067

/-- Proves that Flying Scotsman had 20 more carriages than Norwich -/
theorem flying_scotsman_norwich_difference :
  let euston : ℕ := 130
  let norwich : ℕ := 100
  let total : ℕ := 460
  let norfolk : ℕ := euston - 20
  let flying_scotsman : ℕ := total - (euston + norfolk + norwich)
  flying_scotsman - norwich = 20 := by
  sorry

end NUMINAMATH_CALUDE_flying_scotsman_norwich_difference_l80_8067


namespace NUMINAMATH_CALUDE_plums_for_oranges_l80_8057

-- Define the cost of fruits as real numbers
variables (orange pear plum : ℝ)

-- Define the conditions
def condition1 : Prop := 5 * orange = 3 * pear
def condition2 : Prop := 4 * pear = 6 * plum

-- Theorem to prove
theorem plums_for_oranges 
  (h1 : condition1 orange pear) 
  (h2 : condition2 pear plum) : 
  20 * orange = 18 * plum :=
sorry

end NUMINAMATH_CALUDE_plums_for_oranges_l80_8057


namespace NUMINAMATH_CALUDE_athlete_A_one_win_one_loss_l80_8021

/-- The probability of athlete A winning against athlete B -/
def prob_A_wins_B : ℝ := 0.8

/-- The probability of athlete A winning against athlete C -/
def prob_A_wins_C : ℝ := 0.7

/-- The probability of athlete A achieving one win and one loss -/
def prob_one_win_one_loss : ℝ := prob_A_wins_B * (1 - prob_A_wins_C) + (1 - prob_A_wins_B) * prob_A_wins_C

theorem athlete_A_one_win_one_loss : prob_one_win_one_loss = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_athlete_A_one_win_one_loss_l80_8021


namespace NUMINAMATH_CALUDE_equivalence_of_inequalities_l80_8037

theorem equivalence_of_inequalities (a : ℝ) : a - 1 > 0 ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_equivalence_of_inequalities_l80_8037


namespace NUMINAMATH_CALUDE_pitcher_problem_l80_8012

theorem pitcher_problem (C : ℝ) (h : C > 0) :
  let juice_volume := C / 2
  let num_cups := 8
  let cup_volume := juice_volume / num_cups
  (cup_volume / C) * 100 = 6.25 := by sorry

end NUMINAMATH_CALUDE_pitcher_problem_l80_8012


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l80_8090

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 150 → s^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l80_8090


namespace NUMINAMATH_CALUDE_vase_capacity_l80_8027

/-- The number of flowers each vase can hold -/
def flowers_per_vase (carnations roses vases : ℕ) : ℕ :=
  (carnations + roses) / vases

/-- Theorem: Given 7 carnations, 47 roses, and 9 vases, each vase can hold 6 flowers -/
theorem vase_capacity :
  flowers_per_vase 7 47 9 = 6 := by
sorry

end NUMINAMATH_CALUDE_vase_capacity_l80_8027


namespace NUMINAMATH_CALUDE_nilpotent_is_zero_fourth_power_eq_self_l80_8052

class SpecialRing (A : Type*) extends Ring A where
  special_property : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6

variable {A : Type*} [SpecialRing A]

theorem nilpotent_is_zero (x : A) (n : ℕ) (hn : n ≥ 2) (hx : x^n = 0) : x = 0 := by
  sorry

theorem fourth_power_eq_self (x : A) : x^4 = x := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_is_zero_fourth_power_eq_self_l80_8052


namespace NUMINAMATH_CALUDE_sum_of_nth_row_sum_of_100th_row_l80_8076

/-- The sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * f (n - 1) + 2 * n

/-- Theorem: The closed form of the sum of numbers in the nth row -/
theorem sum_of_nth_row (n : ℕ) : f n = 3 * 2^(n-1) - 2 * n := by
  sorry

/-- Corollary: The sum of numbers in the 100th row -/
theorem sum_of_100th_row : f 100 = 3 * 2^99 - 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_nth_row_sum_of_100th_row_l80_8076


namespace NUMINAMATH_CALUDE_min_days_person_A_l80_8097

/-- Represents the number of days a person takes to complete the project alone -/
structure PersonSpeed where
  days : ℕ
  days_positive : days > 0

/-- Represents the work done by a person in a day -/
def work_rate (speed : PersonSpeed) : ℚ :=
  1 / speed.days

/-- The total project work is 1 -/
def total_work : ℚ := 1

/-- Theorem stating the minimum number of days person A must work -/
theorem min_days_person_A (
  speed_A speed_B speed_C : PersonSpeed)
  (h_A : speed_A.days = 24)
  (h_B : speed_B.days = 36)
  (h_C : speed_C.days = 60)
  (total_days : ℕ)
  (h_total_days : total_days ≤ 18)
  (h_integer_days : ∃ (days_A days_B days_C : ℕ),
    days_A + days_B + days_C = total_days ∧
    days_A * work_rate speed_A + days_B * work_rate speed_B + days_C * work_rate speed_C = total_work) :
  ∃ (min_days_A : ℕ), min_days_A = 6 ∧
    ∀ (days_A : ℕ), 
      (∃ (days_B days_C : ℕ),
        days_A + days_B + days_C = total_days ∧
        days_A * work_rate speed_A + days_B * work_rate speed_B + days_C * work_rate speed_C = total_work) →
      days_A ≥ min_days_A :=
by sorry

end NUMINAMATH_CALUDE_min_days_person_A_l80_8097


namespace NUMINAMATH_CALUDE_ellipse_constant_product_l80_8031

def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def focus (x y : ℝ) : Prop := x = -1 ∧ y = 0

def min_distance (d : ℝ) : Prop := d = Real.sqrt 2 - 1

def point_M (x y : ℝ) : Prop := x = -5/4 ∧ y = 0

def line_intersects_ellipse (l : ℝ → ℝ → Prop) (a b : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂

def product_MA_MB (xₐ yₐ xₘ yₘ xb yb : ℝ) : ℝ :=
  ((xₐ - xₘ)^2 + (yₐ - yₘ)^2) * ((xb - xₘ)^2 + (yb - yₘ)^2)

theorem ellipse_constant_product (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  ∀ l : ℝ → ℝ → Prop,
    (∃ x y, focus x y) →
    (∃ d, min_distance d) →
    (∃ xₘ yₘ, point_M xₘ yₘ) →
    line_intersects_ellipse l a b →
    (∃ xₐ yₐ xb yb xₘ yₘ,
      l xₐ yₐ ∧ l xb yb ∧ point_M xₘ yₘ ∧
      product_MA_MB xₐ yₐ xₘ yₘ xb yb = -7/16) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_constant_product_l80_8031


namespace NUMINAMATH_CALUDE_equation_solution_l80_8092

theorem equation_solution (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  (x + y) / x = (y + 1) / (x + y) →
  (x = (-y + Real.sqrt (4 - 3 * y^2)) / 2 ∨ x = (-y - Real.sqrt (4 - 3 * y^2)) / 2) ∧
  -2 / Real.sqrt 3 ≤ y ∧ y ≤ 2 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l80_8092


namespace NUMINAMATH_CALUDE_solve_equation_l80_8007

theorem solve_equation : 
  ∃ x : ℝ, 3 + 2 * (8 - x) = 24.16 ∧ x = -2.58 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l80_8007


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l80_8042

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^4 + 18 * b^4 + 72 * c^4 + 1 / (27 * a * b * c) ≥ 4 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^4 + 18 * b^4 + 72 * c^4 + 1 / (27 * a * b * c) = 4 ↔
  a = ((9/4)^(1/4) * (1/(18 * ((9/4)^(1/4)) * (2^(1/4))))^(1/3)) ∧
  b = (2^(1/4) * (1/(18 * ((9/4)^(1/4)) * (2^(1/4))))^(1/3)) ∧
  c = (1/(18 * ((9/4)^(1/4)) * (2^(1/4))))^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l80_8042


namespace NUMINAMATH_CALUDE_first_question_percentage_l80_8038

theorem first_question_percentage
  (second_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : second_correct = 55)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 40)
  : ℝ :=
by
  -- The percentage answering the first question correctly is 65%
  sorry

#check first_question_percentage

end NUMINAMATH_CALUDE_first_question_percentage_l80_8038


namespace NUMINAMATH_CALUDE_opposite_expression_implies_ab_zero_l80_8084

/-- Given that for all x, ax + bx^2 = -(a(-x) + b(-x)^2), prove that ab = 0 -/
theorem opposite_expression_implies_ab_zero (a b : ℝ) 
  (h : ∀ x : ℝ, a * x + b * x^2 = -(a * (-x) + b * (-x)^2)) : 
  a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_expression_implies_ab_zero_l80_8084


namespace NUMINAMATH_CALUDE_cookie_problem_l80_8074

theorem cookie_problem (tom mike millie lucy frank : ℕ) : 
  tom = 16 →
  lucy * lucy = tom →
  millie = 2 * lucy →
  mike = 3 * millie →
  frank = mike / 2 - 3 →
  frank = 9 :=
by sorry

end NUMINAMATH_CALUDE_cookie_problem_l80_8074


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l80_8053

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {3, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l80_8053


namespace NUMINAMATH_CALUDE_max_value_f_range_of_m_l80_8014

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.log x - (1/2) * x^2

-- Define the interval [1/e, e]
def I : Set ℝ := { x | 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 }

-- Statement for part (I)
theorem max_value_f : 
  ∃ (x : ℝ), x ∈ I ∧ f x = Real.log 2 - 1 ∧ ∀ y ∈ I, f y ≤ f x :=
sorry

-- Define the function g for part (II)
def g (a x : ℝ) : ℝ := a * Real.log x

-- Define the intervals for a and x in part (II)
def A : Set ℝ := { a | 0 ≤ a ∧ a ≤ 3/2 }
def X : Set ℝ := { x | 1 < x ∧ x ≤ Real.exp 2 }

-- Statement for part (II)
theorem range_of_m :
  ∀ m : ℝ, (∀ a ∈ A, ∀ x ∈ X, g a x ≥ m + x) ↔ m ≤ -(Real.exp 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_f_range_of_m_l80_8014


namespace NUMINAMATH_CALUDE_special_table_sum_l80_8065

/-- Represents a 2 × 7 table where each column after the first is the sum and difference of the previous column --/
def SpecialTable := Fin 7 → Fin 2 → ℤ

/-- The rule for generating subsequent columns --/
def nextColumn (col : Fin 2 → ℤ) : Fin 2 → ℤ :=
  fun i => if i = 0 then col 0 + col 1 else col 0 - col 1

/-- Checks if the table follows the special rule --/
def isValidTable (t : SpecialTable) : Prop :=
  ∀ j : Fin 6, t (j.succ) = nextColumn (t j)

/-- The theorem to be proved --/
theorem special_table_sum (t : SpecialTable) : 
  isValidTable t → t 6 0 = 96 → t 6 1 = 64 → t 0 0 + t 0 1 = 20 := by
  sorry

#check special_table_sum

end NUMINAMATH_CALUDE_special_table_sum_l80_8065


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l80_8083

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n = 3 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l80_8083


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l80_8018

theorem arithmetic_calculations :
  ((-7) * (-5) - 90 / (-15) = 41) ∧
  ((-1)^10 * 2 - (-2)^3 / 4 = 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l80_8018


namespace NUMINAMATH_CALUDE_unique_number_with_appended_digits_sum_l80_8022

theorem unique_number_with_appended_digits_sum (A : ℕ) : 
  (∃ B : ℕ, B ≤ 999 ∧ 1000 * A + B = A * (A + 1) / 2) ↔ A = 1999 :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_appended_digits_sum_l80_8022


namespace NUMINAMATH_CALUDE_right_triangle_condition_l80_8046

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π

-- State the theorem
theorem right_triangle_condition (t : Triangle) :
  Real.sin (t.A + t.B) * Real.sin (t.A - t.B) = (Real.sin t.C)^2 →
  t.C = π / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l80_8046


namespace NUMINAMATH_CALUDE_inequality_proof_l80_8064

theorem inequality_proof (a b : ℝ) (h : a * b ≥ 0) :
  a^4 + 2*a^3*b + 2*a*b^3 + b^4 ≥ 6*a^2*b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l80_8064


namespace NUMINAMATH_CALUDE_base_7_23456_equals_6068_l80_8066

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_23456_equals_6068 :
  base_7_to_10 [6, 5, 4, 3, 2] = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base_7_23456_equals_6068_l80_8066


namespace NUMINAMATH_CALUDE_acute_angles_sum_l80_8061

theorem acute_angles_sum (x y : Real) : 
  0 < x ∧ x < π/2 →
  0 < y ∧ y < π/2 →
  4 * (Real.cos x)^2 + 3 * (Real.cos y)^2 = 1 →
  4 * Real.cos (2*x) - 3 * Real.cos (2*y) = 0 →
  x + 3*y = π/2 := by
sorry

end NUMINAMATH_CALUDE_acute_angles_sum_l80_8061


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l80_8051

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℚ := 56 / 5

/-- The weight of one canoe in pounds -/
def canoe_weight : ℚ := 28

theorem bowling_ball_weight_proof :
  (5 : ℚ) * bowling_ball_weight = 2 * canoe_weight ∧
  (3 : ℚ) * canoe_weight = 84 →
  bowling_ball_weight = 56 / 5 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l80_8051


namespace NUMINAMATH_CALUDE_path_width_is_three_l80_8000

/-- Represents a rectangular garden surrounded by a path of constant width. -/
structure GardenWithPath where
  garden_length : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the perimeter of the garden. -/
def garden_perimeter (g : GardenWithPath) : ℝ :=
  2 * (g.garden_length + g.garden_width)

/-- Calculates the perimeter of the outer edge of the path. -/
def outer_perimeter (g : GardenWithPath) : ℝ :=
  2 * ((g.garden_length + 2 * g.path_width) + (g.garden_width + 2 * g.path_width))

/-- Theorem: If the perimeter of the garden is 24 m shorter than the outer perimeter,
    then the path width is 3 m. -/
theorem path_width_is_three (g : GardenWithPath) :
  outer_perimeter g = garden_perimeter g + 24 → g.path_width = 3 := by
  sorry

#check path_width_is_three

end NUMINAMATH_CALUDE_path_width_is_three_l80_8000


namespace NUMINAMATH_CALUDE_complex_product_example_l80_8059

theorem complex_product_example : (1 + Complex.I) * (2 + Complex.I) = 1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_example_l80_8059


namespace NUMINAMATH_CALUDE_smallest_integer_linear_combination_l80_8078

theorem smallest_integer_linear_combination (m n : ℤ) : 
  ∃ (k : ℕ), k > 0 ∧ (∃ (a b : ℤ), k = 5013 * a + 111111 * b) ∧
  ∀ (l : ℕ), l > 0 → (∃ (c d : ℤ), l = 5013 * c + 111111 * d) → k ≤ l :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_linear_combination_l80_8078


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l80_8013

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₀ + a₂ + a₄ = -121 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l80_8013


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l80_8034

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1 ↔ n = 1) := by sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l80_8034


namespace NUMINAMATH_CALUDE_diamond_three_eight_l80_8019

/-- Definition of the diamond operation -/
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

/-- Theorem stating that 3 ◇ 8 = 60 -/
theorem diamond_three_eight : diamond 3 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_eight_l80_8019


namespace NUMINAMATH_CALUDE_product_xy_in_parallelogram_l80_8068

/-- A parallelogram with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ
  is_parallelogram : EF = GH 1 ∧ FG 1 = HE

/-- The product of x and y in the given parallelogram is 18√3 -/
theorem product_xy_in_parallelogram (p : Parallelogram) 
    (h1 : p.EF = 42)
    (h2 : p.FG = fun y ↦ 4 * y^2 + 1)
    (h3 : p.GH = fun x ↦ 3 * x + 6)
    (h4 : p.HE = 28) :
    ∃ x y, p.GH x = p.EF ∧ p.FG y = p.HE ∧ x * y = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_in_parallelogram_l80_8068


namespace NUMINAMATH_CALUDE_pencils_per_row_indeterminate_l80_8025

theorem pencils_per_row_indeterminate (rows : ℕ) (crayons_per_row : ℕ) (total_crayons : ℕ) :
  rows = 7 →
  crayons_per_row = 30 →
  total_crayons = 210 →
  ∀ (pencils_per_row : ℕ), ∃ (total_pencils : ℕ),
    total_pencils = rows * pencils_per_row :=
by sorry

end NUMINAMATH_CALUDE_pencils_per_row_indeterminate_l80_8025


namespace NUMINAMATH_CALUDE_rhombus_perimeter_given_side_l80_8048

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  side_length : ℝ
  side_length_positive : side_length > 0

/-- The perimeter of a rhombus is four times its side length -/
def perimeter (r : Rhombus) : ℝ := 4 * r.side_length

theorem rhombus_perimeter_given_side (r : Rhombus) (h : r.side_length = 2) : perimeter r = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_given_side_l80_8048


namespace NUMINAMATH_CALUDE_equation_solution_l80_8063

theorem equation_solution (x : ℝ) :
  x > 6 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3) ↔
  x ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l80_8063


namespace NUMINAMATH_CALUDE_simplified_sqrt_expression_l80_8029

theorem simplified_sqrt_expression (x : ℝ) : 
  Real.sqrt (9 * x^4 + 3 * x^2) = Real.sqrt 3 * |x| * Real.sqrt (3 * x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplified_sqrt_expression_l80_8029


namespace NUMINAMATH_CALUDE_hyperbola_equation_l80_8073

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    x = Real.sqrt 2 ∧ y = Real.sqrt 3) →
  (Real.sqrt (1 + b^2 / a^2) = 2) →
  (∀ x y : ℝ, x^2 - y^2 / 3 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l80_8073


namespace NUMINAMATH_CALUDE_reciprocal_and_opposite_of_negative_four_l80_8020

theorem reciprocal_and_opposite_of_negative_four :
  (1 / (-4 : ℝ) = -1/4) ∧ (-((-4) : ℝ) = 4) := by sorry

end NUMINAMATH_CALUDE_reciprocal_and_opposite_of_negative_four_l80_8020


namespace NUMINAMATH_CALUDE_function_upper_bound_l80_8011

open Real

theorem function_upper_bound 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : a ∈ Set.Icc (-1 / Real.exp 1) 0) 
  (h2 : ∀ x > 0, f x = (x + 1) / Real.exp x - a * log x) :
  ∀ x ∈ Set.Ioo 0 2, f x < (1 - a - a^2) / Real.exp (-a) := by
sorry

end NUMINAMATH_CALUDE_function_upper_bound_l80_8011


namespace NUMINAMATH_CALUDE_professor_chair_selections_eq_24_l80_8033

/-- Represents the number of chairs in a row -/
def total_chairs : ℕ := 11

/-- Represents the number of professors -/
def num_professors : ℕ := 3

/-- Represents the minimum number of chairs between professors -/
def min_separation : ℕ := 2

/-- Calculates the number of ways to select chairs for professors -/
def professor_chair_selections : ℕ := sorry

/-- Theorem stating that the number of ways to select chairs for professors is 24 -/
theorem professor_chair_selections_eq_24 :
  professor_chair_selections = 24 := by sorry

end NUMINAMATH_CALUDE_professor_chair_selections_eq_24_l80_8033


namespace NUMINAMATH_CALUDE_vector_sum_proof_l80_8030

/-- Given two vectors a and b in ℝ², prove that their sum is (-1, 5) -/
theorem vector_sum_proof (a b : ℝ × ℝ) (ha : a = (2, 1)) (hb : b = (-3, 4)) :
  a + b = (-1, 5) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l80_8030


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l80_8094

def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x^2 < 4}

theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l80_8094


namespace NUMINAMATH_CALUDE_collection_for_44_members_l80_8035

/-- Calculates the total collection amount in rupees for a group of students -/
def total_collection_rupees (num_members : ℕ) (paise_per_rupee : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / paise_per_rupee

/-- Proves that the total collection amount for 44 members is 19.36 rupees -/
theorem collection_for_44_members :
  total_collection_rupees 44 100 = 19.36 := by
  sorry

#eval total_collection_rupees 44 100

end NUMINAMATH_CALUDE_collection_for_44_members_l80_8035


namespace NUMINAMATH_CALUDE_f_properties_and_value_l80_8091

/-- A linear function satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The theorem stating the properties of f and its value at -1 -/
theorem f_properties_and_value :
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧ 
  (∀ x, f x = 3 * f⁻¹ x + 5) ∧
  (f 0 = 3) →
  f (-1) = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_f_properties_and_value_l80_8091


namespace NUMINAMATH_CALUDE_thirtieth_in_base_five_l80_8001

def to_base_five (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem thirtieth_in_base_five :
  to_base_five 30 = [1, 1, 0] :=
sorry

end NUMINAMATH_CALUDE_thirtieth_in_base_five_l80_8001


namespace NUMINAMATH_CALUDE_bus_fare_impossible_l80_8003

/-- Represents the denominations of coins available --/
inductive Coin : Type
  | ten : Coin
  | fifteen : Coin
  | twenty : Coin

/-- The value of a coin in kopecks --/
def coin_value : Coin → Nat
  | Coin.ten => 10
  | Coin.fifteen => 15
  | Coin.twenty => 20

/-- A configuration of coins --/
def CoinConfig := List Coin

/-- The total value of a coin configuration in kopecks --/
def total_value (config : CoinConfig) : Nat :=
  config.foldl (fun acc c => acc + coin_value c) 0

/-- The number of coins in a configuration --/
def coin_count (config : CoinConfig) : Nat := config.length

theorem bus_fare_impossible : 
  ∀ (config : CoinConfig), 
    (coin_count config = 49) → 
    (total_value config = 200) → 
    False :=
sorry

end NUMINAMATH_CALUDE_bus_fare_impossible_l80_8003


namespace NUMINAMATH_CALUDE_range_of_a_l80_8010

-- Define the polynomials p and q
def p (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
def q (x a : ℝ) : ℝ := x^2 - (2 * a + 1) * x + a^2 + a

-- Define the condition for p
def p_condition (x : ℝ) : Prop := p x ≤ 0

-- Define the condition for q
def q_condition (x a : ℝ) : Prop := q x a ≤ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p_condition x → q_condition x a) ∧
  (∃ x, q_condition x a ∧ ¬p_condition x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l80_8010


namespace NUMINAMATH_CALUDE_community_center_ticket_sales_l80_8049

/-- Calculates the total amount collected from ticket sales given the ticket prices and quantities sold. -/
def total_amount_collected (adult_price child_price : ℕ) (total_tickets adult_tickets : ℕ) : ℕ :=
  adult_price * adult_tickets + child_price * (total_tickets - adult_tickets)

/-- Theorem stating that given the specific conditions of the problem, the total amount collected is $275. -/
theorem community_center_ticket_sales :
  let adult_price : ℕ := 5
  let child_price : ℕ := 2
  let total_tickets : ℕ := 85
  let adult_tickets : ℕ := 35
  total_amount_collected adult_price child_price total_tickets adult_tickets = 275 := by
sorry

end NUMINAMATH_CALUDE_community_center_ticket_sales_l80_8049


namespace NUMINAMATH_CALUDE_simplify_roots_l80_8096

theorem simplify_roots : 
  Real.sqrt 27 - Real.sqrt (1/3) + Real.sqrt 12 = 14 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_roots_l80_8096


namespace NUMINAMATH_CALUDE_sphere_surface_area_cuboid_l80_8054

/-- The surface area of a sphere circumscribing a cuboid with dimensions 2, 1, and 1 is 6π. -/
theorem sphere_surface_area_cuboid : 
  ∃ (r : ℝ), 
    r > 0 ∧ 
    (2 : ℝ)^2 + 1^2 + 1^2 = (2*r)^2 ∧ 
    4 * Real.pi * r^2 = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_cuboid_l80_8054


namespace NUMINAMATH_CALUDE_inequality_proof_l80_8047

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) (h4 : n ≥ 2) :
  (3 / 2 : ℝ) < 1 / (a^n + 1) + 1 / (b^n + 1) ∧ 
  1 / (a^n + 1) + 1 / (b^n + 1) ≤ (2^(n+1) : ℝ) / (2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l80_8047
