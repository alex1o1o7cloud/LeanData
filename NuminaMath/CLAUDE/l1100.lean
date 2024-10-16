import Mathlib

namespace NUMINAMATH_CALUDE_david_subtraction_l1100_110034

theorem david_subtraction (n : ℕ) (h : n = 40) : n^2 - 79 = (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_david_subtraction_l1100_110034


namespace NUMINAMATH_CALUDE_linear_correlation_proof_l1100_110062

/-- Determines if two variables are linearly correlated based on the correlation coefficient and critical value -/
def are_linearly_correlated (r : ℝ) (r_critical : ℝ) : Prop :=
  |r| > r_critical

/-- Theorem stating that given conditions lead to linear correlation -/
theorem linear_correlation_proof (r r_critical : ℝ) 
  (h1 : r = -0.9362)
  (h2 : r_critical = 0.8013) :
  are_linearly_correlated r r_critical :=
by
  sorry

#check linear_correlation_proof

end NUMINAMATH_CALUDE_linear_correlation_proof_l1100_110062


namespace NUMINAMATH_CALUDE_quadratic_radical_range_l1100_110043

theorem quadratic_radical_range : 
  {x : ℝ | ∃ y : ℝ, y^2 = 3*x - 1} = {x : ℝ | x ≥ 1/3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_range_l1100_110043


namespace NUMINAMATH_CALUDE_fraction_equality_l1100_110089

theorem fraction_equality : (250 : ℚ) / ((20 + 15 * 3) - 10) = 250 / 55 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1100_110089


namespace NUMINAMATH_CALUDE_field_distance_l1100_110093

theorem field_distance (D : ℝ) (mary edna lucy : ℝ) : 
  mary = (3/8) * D →
  edna = (2/3) * mary →
  lucy = (5/6) * edna →
  lucy + 4 = mary →
  D = 24 := by
sorry

end NUMINAMATH_CALUDE_field_distance_l1100_110093


namespace NUMINAMATH_CALUDE_area_of_AGKIJEFB_l1100_110049

-- Define the hexagons and point K
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ
  area : ℝ

def Point := ℝ × ℝ

-- Define the problem setup
axiom hexagon1 : Hexagon
axiom hexagon2 : Hexagon
axiom K : Point

-- State the conditions
axiom shared_side : hexagon1.vertices 4 = hexagon2.vertices 4 ∧ hexagon1.vertices 5 = hexagon2.vertices 5
axiom equal_areas : hexagon1.area = 36 ∧ hexagon2.area = 36
axiom K_on_AB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ K = (1 - t) • hexagon1.vertices 0 + t • hexagon1.vertices 1
axiom AK_KB_ratio : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a / b = 1 / 2 ∧
  K = (b / (a + b)) • hexagon1.vertices 0 + (a / (a + b)) • hexagon1.vertices 1
axiom K_midpoint_GH : K = (1 / 2) • hexagon2.vertices 0 + (1 / 2) • hexagon2.vertices 1

-- Define the polygon AGKIJEFB
def polygon_AGKIJEFB_area : ℝ := sorry

-- State the theorem to be proved
theorem area_of_AGKIJEFB : polygon_AGKIJEFB_area = 36 + Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_area_of_AGKIJEFB_l1100_110049


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1100_110020

theorem fraction_sum_equals_decimal : 
  2/5 + 3/25 + 4/125 + 1/625 = 0.5536 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1100_110020


namespace NUMINAMATH_CALUDE_f_extrema_and_monotonicity_l1100_110082

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + a)

theorem f_extrema_and_monotonicity :
  (∃ (x_max x_min : ℝ), f (-3) x_max = 6 * Real.exp (-3) ∧
                        f (-3) x_min = -2 * Real.exp 1 ∧
                        ∀ x, f (-3) x ≤ f (-3) x_max ∧
                              f (-3) x ≥ f (-3) x_min) ∧
  (∀ a, (∀ x y, x < y → f a x < f a y) → a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_and_monotonicity_l1100_110082


namespace NUMINAMATH_CALUDE_power_calculation_l1100_110084

theorem power_calculation : 16^10 * 8^12 / 4^28 = 2^20 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1100_110084


namespace NUMINAMATH_CALUDE_area_enclosed_by_curves_l1100_110091

-- Define the curves
def curve1 (x y : ℝ) : Prop := y^2 = x
def curve2 (x y : ℝ) : Prop := y = x^2

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := sorry

-- Theorem statement
theorem area_enclosed_by_curves : enclosed_area = 1/3 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_curves_l1100_110091


namespace NUMINAMATH_CALUDE_friend_c_spent_26_l1100_110012

/-- Friend C's lunch cost given the conditions of the problem -/
def friend_c_cost (your_cost friend_a_extra friend_b_less : ℕ) : ℕ :=
  2 * (your_cost + friend_a_extra - friend_b_less)

/-- Theorem stating that Friend C's lunch cost is $26 -/
theorem friend_c_spent_26 : friend_c_cost 12 4 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_friend_c_spent_26_l1100_110012


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l1100_110051

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 20 = 12 →
  arithmetic_sequence a₁ d 21 = 16 →
  arithmetic_sequence a₁ d 5 = -48 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l1100_110051


namespace NUMINAMATH_CALUDE_no_97_points_l1100_110004

/-- Represents the score on a test with the given scoring system -/
structure TestScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ
  total : correct + unanswered + incorrect = 20

/-- Calculates the total points for a given TestScore -/
def calculatePoints (score : TestScore) : ℕ :=
  5 * score.correct + score.unanswered

/-- Theorem stating that 97 points is not possible on the test -/
theorem no_97_points : ¬ ∃ (score : TestScore), calculatePoints score = 97 := by
  sorry


end NUMINAMATH_CALUDE_no_97_points_l1100_110004


namespace NUMINAMATH_CALUDE_equation_solution_l1100_110053

theorem equation_solution (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x ≠ 1/16) 
  (h3 : x ≠ 1/2) 
  (h4 : x ≠ 1) : 
  (Real.log 2 / Real.log (4 * Real.sqrt x)) / (Real.log 2 / Real.log (2 * x)) + 
  (Real.log 2 / Real.log (2 * x)) * (Real.log (2 * x) / Real.log (1/2)) = 0 ↔ 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1100_110053


namespace NUMINAMATH_CALUDE_cos_three_pi_halves_l1100_110003

theorem cos_three_pi_halves : Real.cos (3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_halves_l1100_110003


namespace NUMINAMATH_CALUDE_find_M_l1100_110092

theorem find_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1230) ∧ (M = 3690) := by sorry

end NUMINAMATH_CALUDE_find_M_l1100_110092


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1100_110069

-- Problem 1
theorem problem_1 : -1^2023 * ((-8) + 2 / (1/2)) - |(-3)| = 1 := by sorry

-- Problem 2
theorem problem_2 : ∃ x : ℚ, (x + 2) / 3 - (x - 1) / 2 = x + 2 ∧ x = -5/7 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1100_110069


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_3_l1100_110050

/-- The function f(x) = x^2 + 4ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4*a*x + 2

/-- The theorem stating that if f(x) is monotonically decreasing in (-∞, 6), then a ≤ 3 -/
theorem monotone_decreasing_implies_a_leq_3 (a : ℝ) :
  (∀ x y, x < y → y < 6 → f a x > f a y) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_leq_3_l1100_110050


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1100_110016

theorem polynomial_division_theorem (x : ℝ) : 
  2*x^4 - 3*x^3 + x^2 + 5*x - 7 = (x + 1)*(2*x^3 - 5*x^2 + 6*x - 1) + (-6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1100_110016


namespace NUMINAMATH_CALUDE_deepak_age_l1100_110075

/-- Given the ratio between Rahul and Deepak's ages is 4:3, and that Rahul will be 26 years old after 6 years, prove that Deepak's present age is 15 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  rahul_age = 4 * (rahul_age / 4) → 
  deepak_age = 3 * (rahul_age / 4) → 
  rahul_age + 6 = 26 → 
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1100_110075


namespace NUMINAMATH_CALUDE_exactly_one_prop_true_l1100_110017

-- Define a type for lines
structure Line where
  -- Add necessary fields for a line

-- Define what it means for two lines to form equal angles with a third line
def form_equal_angles (l1 l2 l3 : Line) : Prop := sorry

-- Define what it means for a line to be perpendicular to another line
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the three propositions
def prop1 : Prop := ∀ l1 l2 l3 : Line, form_equal_angles l1 l2 l3 → parallel l1 l2
def prop2 : Prop := ∀ l1 l2 l3 : Line, perpendicular l1 l3 ∧ perpendicular l2 l3 → parallel l1 l2
def prop3 : Prop := ∀ l1 l2 l3 : Line, parallel l1 l3 ∧ parallel l2 l3 → parallel l1 l2

-- Theorem stating that exactly one proposition is true
theorem exactly_one_prop_true : (prop1 ∧ ¬prop2 ∧ ¬prop3) ∨ (¬prop1 ∧ prop2 ∧ ¬prop3) ∨ (¬prop1 ∧ ¬prop2 ∧ prop3) :=
  sorry

end NUMINAMATH_CALUDE_exactly_one_prop_true_l1100_110017


namespace NUMINAMATH_CALUDE_flower_shop_relation_l1100_110031

theorem flower_shop_relation (C V T R : ℕ) (total : ℕ) : 
  V = C / 3 →
  T = V / 3 →
  C = (64423765211166780 : ℕ) * total / 100000000000000000 →
  C + V + T + R = total →
  T = C / 9 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_relation_l1100_110031


namespace NUMINAMATH_CALUDE_distance_AB_is_correct_l1100_110000

/-- The distance between two points A and B, where two people start simultaneously
    and move towards each other under specific conditions. -/
def distance_AB : ℝ :=
  let speed_A : ℝ := 12.5 -- km/h
  let speed_B : ℝ := 10   -- km/h
  let time_to_bank : ℝ := 0.5 -- hours
  let time_to_return : ℝ := 0.5 -- hours
  let time_to_find_card : ℝ := 0.5 -- hours
  let remaining_time : ℝ := 0.25 -- hours
  62.5 -- km

theorem distance_AB_is_correct :
  let speed_A : ℝ := 12.5 -- km/h
  let speed_B : ℝ := 10   -- km/h
  let time_to_bank : ℝ := 0.5 -- hours
  let time_to_return : ℝ := 0.5 -- hours
  let time_to_find_card : ℝ := 0.5 -- hours
  let remaining_time : ℝ := 0.25 -- hours
  distance_AB = 62.5 := by
  sorry

#check distance_AB_is_correct

end NUMINAMATH_CALUDE_distance_AB_is_correct_l1100_110000


namespace NUMINAMATH_CALUDE_quadratic_roots_uniqueness_l1100_110095

/-- Given two quadratic polynomials with specific root relationships, 
    prove that there is only one set of values for the roots and coefficients. -/
theorem quadratic_roots_uniqueness (p q u v : ℝ) : 
  p ≠ 0 ∧ q ≠ 0 ∧ u ≠ 0 ∧ v ≠ 0 ∧  -- non-zero roots
  p ≠ q ∧ u ≠ v ∧  -- distinct roots
  (∀ x, x^2 + u*x - v = (x - p)*(x - q)) ∧  -- first polynomial
  (∀ x, x^2 + p*x - q = (x - u)*(x - v)) →  -- second polynomial
  p = -1 ∧ q = 2 ∧ u = -1 ∧ v = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_uniqueness_l1100_110095


namespace NUMINAMATH_CALUDE_prob_green_is_0_15_l1100_110025

/-- The probability of selecting a green jelly bean from a jar -/
def prob_green (prob_red prob_orange prob_blue prob_yellow : ℝ) : ℝ :=
  1 - (prob_red + prob_orange + prob_blue + prob_yellow)

/-- Theorem: The probability of selecting a green jelly bean is 0.15 -/
theorem prob_green_is_0_15 :
  prob_green 0.15 0.35 0.2 0.15 = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_is_0_15_l1100_110025


namespace NUMINAMATH_CALUDE_golden_retriever_adult_weight_l1100_110001

/-- Represents the weight of a golden retriever at different stages of growth -/
structure DogWeight where
  initial : ℕ  -- Weight at 7 weeks
  week9 : ℕ    -- Weight at 9 weeks
  month3 : ℕ   -- Weight at 3 months
  month5 : ℕ   -- Weight at 5 months
  adult : ℕ    -- Adult weight at 1 year

/-- Calculates the adult weight of a golden retriever based on its growth pattern -/
def calculateAdultWeight (w : DogWeight) : ℕ :=
  w.initial * 2 * 2 * 2 + 30

/-- Theorem stating that the adult weight of the golden retriever is 78 pounds -/
theorem golden_retriever_adult_weight (w : DogWeight) 
  (h1 : w.initial = 6)
  (h2 : w.week9 = w.initial * 2)
  (h3 : w.month3 = w.week9 * 2)
  (h4 : w.month5 = w.month3 * 2)
  (h5 : w.adult = w.month5 + 30) :
  w.adult = 78 := by
  sorry


end NUMINAMATH_CALUDE_golden_retriever_adult_weight_l1100_110001


namespace NUMINAMATH_CALUDE_minimum_games_for_percentage_l1100_110040

theorem minimum_games_for_percentage (N : ℕ) : N = 7 ↔ 
  (N ≥ 0) ∧ 
  (∀ k : ℕ, k ≥ 0 → (2 : ℚ) / (3 + k) ≥ (9 : ℚ) / 10 → k ≥ N) ∧
  ((2 : ℚ) / (3 + N) ≥ (9 : ℚ) / 10) :=
by sorry

end NUMINAMATH_CALUDE_minimum_games_for_percentage_l1100_110040


namespace NUMINAMATH_CALUDE_angle_measure_l1100_110018

theorem angle_measure (C D : ℝ) : 
  C + D = 180 →  -- Angles C and D are supplementary
  C = 5 * D →    -- Measure of angle C is 5 times angle D
  C = 150 :=     -- Measure of angle C is 150 degrees
by sorry

end NUMINAMATH_CALUDE_angle_measure_l1100_110018


namespace NUMINAMATH_CALUDE_starting_lineup_count_l1100_110097

def total_team_members : ℕ := 12
def offensive_linemen : ℕ := 4
def linemen_quarterbacks : ℕ := 2
def running_backs : ℕ := 3

def starting_lineup_combinations : ℕ := 
  offensive_linemen * linemen_quarterbacks * running_backs * (total_team_members - 3)

theorem starting_lineup_count : starting_lineup_combinations = 216 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l1100_110097


namespace NUMINAMATH_CALUDE_monotonic_absolute_value_function_l1100_110047

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem monotonic_absolute_value_function (a : ℝ) :
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f a x ≤ f a y ∨ f a x ≥ f a y) →
  a ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_absolute_value_function_l1100_110047


namespace NUMINAMATH_CALUDE_min_tests_correct_l1100_110027

/-- Represents the result of a test between two balls -/
inductive TestResult
| Same
| Different

/-- Represents a ball -/
structure Ball :=
  (id : Nat)
  (metal : Bool)  -- True for copper, False for zinc

/-- Represents a test between two balls -/
structure Test :=
  (ball1 : Ball)
  (ball2 : Ball)
  (result : TestResult)

/-- The minimum number of tests required to determine the material of each ball -/
def min_tests (n : Nat) (copper_count : Nat) (zinc_count : Nat) : Nat :=
  n - 1

theorem min_tests_correct (n : Nat) (copper_count : Nat) (zinc_count : Nat) 
  (h1 : n = 99)
  (h2 : copper_count = 50)
  (h3 : zinc_count = 49)
  (h4 : copper_count + zinc_count = n) :
  min_tests n copper_count zinc_count = 98 := by
  sorry

#eval min_tests 99 50 49

end NUMINAMATH_CALUDE_min_tests_correct_l1100_110027


namespace NUMINAMATH_CALUDE_B_current_age_l1100_110052

-- Define variables for A's and B's current ages
variable (A B : ℕ)

-- Define the conditions
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 6

-- Theorem statement
theorem B_current_age (h1 : condition1 A B) (h2 : condition2 A B) : B = 36 := by
  sorry

end NUMINAMATH_CALUDE_B_current_age_l1100_110052


namespace NUMINAMATH_CALUDE_min_value_of_function_l1100_110059

theorem min_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∃ y : ℝ, y = 9 ∧ ∀ z : ℝ, (4 / x + 1 / (1 - x)) ≥ y := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1100_110059


namespace NUMINAMATH_CALUDE_number_of_factors_of_power_l1100_110028

theorem number_of_factors_of_power (b n : ℕ+) (hb : b = 8) (hn : n = 15) :
  (Finset.range ((n * (Nat.factorization b).sum (fun _ e => e)) + 1)).card = 46 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_power_l1100_110028


namespace NUMINAMATH_CALUDE_spider_cylinder_ratio_l1100_110038

/-- In a cylindrical room, a spider can reach the opposite point on the floor
    by two paths of equal length. This theorem proves the ratio of the cylinder's
    height to its diameter given these conditions. -/
theorem spider_cylinder_ratio (m r : ℝ) (h_positive : m > 0 ∧ r > 0) :
  (m + 2*r = Real.sqrt (m^2 + (r*Real.pi)^2)) →
  m / (2*r) = (Real.pi^2 - 4) / 8 := by
  sorry

#check spider_cylinder_ratio

end NUMINAMATH_CALUDE_spider_cylinder_ratio_l1100_110038


namespace NUMINAMATH_CALUDE_impossibility_of_crossing_plan_l1100_110088

/-- Represents a group of friends -/
def FriendGroup := Finset (Fin 5)

/-- The set of all possible non-empty groups of friends -/
def AllGroups : Set FriendGroup :=
  {g : FriendGroup | g.Nonempty}

/-- A crossing plan is a function that assigns each group to a number of crossings -/
def CrossingPlan := FriendGroup → ℕ

/-- A valid crossing plan assigns exactly one crossing to each non-empty group -/
def IsValidPlan (plan : CrossingPlan) : Prop :=
  ∀ g : FriendGroup, g ∈ AllGroups → plan g = 1

theorem impossibility_of_crossing_plan :
  ¬∃ (plan : CrossingPlan), IsValidPlan plan :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_crossing_plan_l1100_110088


namespace NUMINAMATH_CALUDE_work_day_end_time_l1100_110061

-- Define the start time, lunch start time, and total work hours
def start_time : ℕ := 465  -- 7:45 AM in minutes since midnight
def lunch_start : ℕ := 720  -- 12:00 PM in minutes since midnight
def total_work_hours : ℕ := 540  -- 9 hours in minutes

-- Define the lunch break duration
def lunch_duration : ℕ := 75  -- 1 hour and 15 minutes in minutes

-- Define the end time we want to prove
def end_time : ℕ := 1080  -- 6:00 PM in minutes since midnight

-- Theorem to prove
theorem work_day_end_time : 
  start_time + total_work_hours + lunch_duration = end_time :=
sorry

end NUMINAMATH_CALUDE_work_day_end_time_l1100_110061


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1100_110039

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1100_110039


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1100_110006

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (∀ x, x < -1 → x^2 - 1 > 0) ∧ 
  (∃ x, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1100_110006


namespace NUMINAMATH_CALUDE_area_of_R_in_unit_square_l1100_110014

/-- A square with side length 1 -/
def UnitSquare : Set (ℝ × ℝ) :=
  {p | -1/2 ≤ p.1 ∧ p.1 ≤ 1/2 ∧ -1/2 ≤ p.2 ∧ p.2 ≤ 1/2}

/-- The subset of points closer to the center than to any side -/
def R (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p ∈ S | ∀ q ∈ S, dist p (0, 0) < dist p q}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem area_of_R_in_unit_square :
  area (R UnitSquare) = (4 * Real.sqrt 2 - 5) / 3 := by sorry

end NUMINAMATH_CALUDE_area_of_R_in_unit_square_l1100_110014


namespace NUMINAMATH_CALUDE_problem_solution_l1100_110023

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 2) (h3 : c^2 / a = 3) : 
  a = 12^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1100_110023


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l1100_110065

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 926 → ¬(n + 10 ∣ n^3 + 64) ∧ (926 + 10 ∣ 926^3 + 64) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l1100_110065


namespace NUMINAMATH_CALUDE_function_properties_l1100_110008

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x) + (2^x - 1) / (2^x + 1) + 3

def g (x : ℝ) : ℝ := sorry

theorem function_properties :
  (∀ x : ℝ, f x + f (-x) = 6) ∧
  (∀ x : ℝ, g x + g (-x) = 6) ∧
  (∀ a b : ℝ, f a + f b > 6 → a + b > 0) := by sorry

end NUMINAMATH_CALUDE_function_properties_l1100_110008


namespace NUMINAMATH_CALUDE_logarithm_inequality_l1100_110032

theorem logarithm_inequality (x : ℝ) (h : 1 < x ∧ x < 10) :
  Real.log (Real.log x) < (Real.log x)^2 ∧ (Real.log x)^2 < Real.log (x^2) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l1100_110032


namespace NUMINAMATH_CALUDE_no_a_in_either_subject_l1100_110080

theorem no_a_in_either_subject (total_students : ℕ) (physics_a : ℕ) (chemistry_a : ℕ) (both_a : ℕ)
  (h1 : total_students = 40)
  (h2 : physics_a = 10)
  (h3 : chemistry_a = 18)
  (h4 : both_a = 6) :
  total_students - (physics_a + chemistry_a - both_a) = 18 :=
by sorry

end NUMINAMATH_CALUDE_no_a_in_either_subject_l1100_110080


namespace NUMINAMATH_CALUDE_fraction_equality_l1100_110055

theorem fraction_equality (a b : ℚ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1100_110055


namespace NUMINAMATH_CALUDE_boys_without_pencils_l1100_110011

theorem boys_without_pencils (total_students : ℕ) (total_boys : ℕ) (students_with_pencils : ℕ) (girls_with_pencils : ℕ)
  (h1 : total_students = 30)
  (h2 : total_boys = 18)
  (h3 : students_with_pencils = 25)
  (h4 : girls_with_pencils = 15) :
  total_boys - (students_with_pencils - girls_with_pencils) = 8 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_pencils_l1100_110011


namespace NUMINAMATH_CALUDE_f_geq_6_iff_min_value_sum_reciprocals_min_value_sum_reciprocals_equality_l1100_110044

-- Part 1
def f (x : ℝ) : ℝ := |x + 1| + |2*x - 4|

theorem f_geq_6_iff (x : ℝ) : f x ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 := by sorry

-- Part 2
theorem min_value_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + 2*b + 4*c = 8) : 
  (1/a + 1/b + 1/c) ≥ (11 + 6*Real.sqrt 2) / 8 := by sorry

theorem min_value_sum_reciprocals_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + 2*b + 4*c = 8) : 
  (1/a + 1/b + 1/c) = (11 + 6*Real.sqrt 2) / 8 ↔ a = Real.sqrt 2 * b ∧ b = 2 * c := by sorry

end NUMINAMATH_CALUDE_f_geq_6_iff_min_value_sum_reciprocals_min_value_sum_reciprocals_equality_l1100_110044


namespace NUMINAMATH_CALUDE_bakery_sales_projection_l1100_110078

theorem bakery_sales_projection (white_bread_ratio : ℕ) (wheat_bread_ratio : ℕ) 
  (projected_white_bread : ℕ) (expected_wheat_bread : ℕ) : 
  white_bread_ratio = 5 → 
  wheat_bread_ratio = 8 → 
  projected_white_bread = 45 →
  expected_wheat_bread = wheat_bread_ratio * projected_white_bread / white_bread_ratio →
  expected_wheat_bread = 72 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sales_projection_l1100_110078


namespace NUMINAMATH_CALUDE_stating_plane_landing_time_l1100_110041

/-- Represents the scenario of a mail delivery between a post office and an airfield -/
structure MailDeliveryScenario where
  usual_travel_time : ℕ  -- Usual one-way travel time in minutes
  early_arrival : ℕ      -- How many minutes earlier the Moskvich arrived
  truck_travel_time : ℕ  -- How long the truck traveled before meeting Moskvich

/-- 
Theorem stating that under the given conditions, the plane must have landed 40 minutes early.
-/
theorem plane_landing_time (scenario : MailDeliveryScenario) 
  (h1 : scenario.early_arrival = 20)
  (h2 : scenario.truck_travel_time = 30) :
  40 = (scenario.truck_travel_time + (scenario.early_arrival / 2)) :=
by sorry

end NUMINAMATH_CALUDE_stating_plane_landing_time_l1100_110041


namespace NUMINAMATH_CALUDE_emma_last_page_l1100_110094

/-- Represents a reader with their reading speed in seconds per page -/
structure Reader where
  name : String
  speed : ℕ

/-- Represents the novel reading scenario -/
structure NovelReading where
  totalPages : ℕ
  emma : Reader
  liam : Reader
  noah : Reader
  noahPages : ℕ

/-- Calculates the last page Emma should read -/
def lastPageForEmma (scenario : NovelReading) : ℕ :=
  sorry

/-- Theorem stating that the last page Emma should read is 525 -/
theorem emma_last_page (scenario : NovelReading) 
  (h1 : scenario.totalPages = 900)
  (h2 : scenario.emma = ⟨"Emma", 15⟩)
  (h3 : scenario.liam = ⟨"Liam", 45⟩)
  (h4 : scenario.noah = ⟨"Noah", 30⟩)
  (h5 : scenario.noahPages = 200)
  : lastPageForEmma scenario = 525 := by
  sorry

end NUMINAMATH_CALUDE_emma_last_page_l1100_110094


namespace NUMINAMATH_CALUDE_jacket_discount_percentage_l1100_110045

/-- Calculates the discount percentage on a jacket sale --/
theorem jacket_discount_percentage
  (purchase_price : ℝ)
  (markup_percentage : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 48)
  (h2 : markup_percentage = 0.4)
  (h3 : gross_profit = 16) :
  let selling_price := purchase_price / (1 - markup_percentage)
  let sale_price := purchase_price + gross_profit
  (selling_price - sale_price) / selling_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_jacket_discount_percentage_l1100_110045


namespace NUMINAMATH_CALUDE_card_area_theorem_l1100_110013

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

theorem card_area_theorem (card : Rectangle) 
  (h1 : card.width = 3 ∧ card.height = 7)
  (h2 : ∃ (shortened : Rectangle), 
    (shortened.width = card.width ∧ shortened.height = card.height - 1) ∨
    (shortened.width = card.width - 1 ∧ shortened.height = card.height) ∧
    area shortened = 15) :
  ∃ (other_shortened : Rectangle),
    ((other_shortened.width = card.width - 1 ∧ other_shortened.height = card.height) ∨
     (other_shortened.width = card.width ∧ other_shortened.height = card.height - 1)) ∧
    area other_shortened = 10 := by
  sorry

end NUMINAMATH_CALUDE_card_area_theorem_l1100_110013


namespace NUMINAMATH_CALUDE_line_slope_range_l1100_110021

/-- The range of m for a line x - my + √3m = 0 with a point M satisfying certain conditions -/
theorem line_slope_range (m : ℝ) : 
  (∃ (x y : ℝ), x - m * y + Real.sqrt 3 * m = 0 ∧ 
    y^2 = 3 * x^2 - 3) →
  (m ≤ -Real.sqrt 6 / 6 ∨ m ≥ Real.sqrt 6 / 6) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_range_l1100_110021


namespace NUMINAMATH_CALUDE_a_range_l1100_110057

theorem a_range (a : ℝ) (h : a^(3/2) < a^(Real.sqrt 2)) : 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l1100_110057


namespace NUMINAMATH_CALUDE_rectangular_box_diagonals_l1100_110036

theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + a * c) = 118) 
  (edge_sum : 4 * (a + b + c) = 52) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 51 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonals_l1100_110036


namespace NUMINAMATH_CALUDE_sum_of_sequences_l1100_110060

def sequence1 : List ℕ := [1, 12, 23, 34, 45]
def sequence2 : List ℕ := [10, 20, 30, 40, 50]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum) = 265 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l1100_110060


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_inning_l1100_110072

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (b : Batsman) (runsInLastInning : ℕ) : ℚ :=
  (b.totalRuns + runsInLastInning : ℚ) / (b.innings + 1)

theorem batsman_average_after_12th_inning 
  (b : Batsman) 
  (h1 : b.innings = 11)
  (h2 : newAverage b 60 = b.average + 4) :
  newAverage b 60 = 16 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_inning_l1100_110072


namespace NUMINAMATH_CALUDE_cubic_polynomial_bound_l1100_110085

theorem cubic_polynomial_bound (p q r : ℝ) : 
  ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ |x^3 + p*x^2 + q*x + r| ≥ (1/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_bound_l1100_110085


namespace NUMINAMATH_CALUDE_zeros_when_m_zero_one_zero_in_interval_l1100_110070

/-- The function f(x) defined in terms of m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (2*m + 1)*x + m*(m + 1)

/-- Theorem stating the zeros of f(x) when m = 0 -/
theorem zeros_when_m_zero :
  ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 1 ∧ f 0 x₁ = 0 ∧ f 0 x₂ = 0 :=
sorry

/-- Theorem stating the range of m for which f(x) has exactly one zero in (1,3) -/
theorem one_zero_in_interval (m : ℝ) :
  (∃! x, 1 < x ∧ x < 3 ∧ f m x = 0) ↔ (0 < m ∧ m ≤ 1) ∨ (2 ≤ m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_zeros_when_m_zero_one_zero_in_interval_l1100_110070


namespace NUMINAMATH_CALUDE_rational_square_difference_l1100_110009

theorem rational_square_difference (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by sorry

end NUMINAMATH_CALUDE_rational_square_difference_l1100_110009


namespace NUMINAMATH_CALUDE_intersection_and_inequality_l1100_110099

-- Define the solution sets
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection
def intersection : Set ℝ := A ∩ B

-- Define the quadratic inequality with parameters a and b
def quadratic_inequality (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- Define the linear inequality with parameters a and b
def linear_inequality (a b : ℝ) : Set ℝ := {x | a*x^2 + x + b < 0}

theorem intersection_and_inequality :
  (intersection = Set.Ioo (-1) 2) ∧
  (∃ a b : ℝ, quadratic_inequality a b = Set.Ioo (-1) 2 → linear_inequality a b = Set.univ) :=
sorry


end NUMINAMATH_CALUDE_intersection_and_inequality_l1100_110099


namespace NUMINAMATH_CALUDE_admission_cutoff_score_admission_cutoff_score_is_96_l1100_110081

theorem admission_cutoff_score (total_average : ℝ) (admitted_fraction : ℝ) 
  (admitted_score_diff : ℝ) (non_admitted_score_diff : ℝ) : ℝ :=
  let cutoff := total_average + (admitted_fraction * admitted_score_diff - 
    (1 - admitted_fraction) * non_admitted_score_diff)
  cutoff

theorem admission_cutoff_score_is_96 :
  admission_cutoff_score 90 (2/5) 15 20 = 96 := by
  sorry

end NUMINAMATH_CALUDE_admission_cutoff_score_admission_cutoff_score_is_96_l1100_110081


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1100_110076

theorem quadratic_equation_roots : ∃ x : ℝ, x^2 + 6*x + 9 = 0 ∧ 
  (∀ y : ℝ, y^2 + 6*y + 9 = 0 → y = x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1100_110076


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1100_110096

def U : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {0, 1, 2}
def N : Finset ℕ := {2, 3}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1100_110096


namespace NUMINAMATH_CALUDE_swimmer_journey_l1100_110048

/-- Swimmer's journey problem -/
theorem swimmer_journey 
  (swimmer_speed : ℝ) 
  (current_speed : ℝ) 
  (distance_PQ : ℝ) 
  (distance_QR : ℝ) 
  (h1 : swimmer_speed = 1)
  (h2 : distance_PQ / (swimmer_speed + current_speed) + distance_QR / swimmer_speed = 3)
  (h3 : distance_QR / (swimmer_speed - current_speed) + distance_PQ / (swimmer_speed - current_speed) = 6)
  (h4 : (distance_PQ + distance_QR) / (swimmer_speed + current_speed) = 5/2)
  : (distance_QR + distance_PQ) / (swimmer_speed - current_speed) = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_journey_l1100_110048


namespace NUMINAMATH_CALUDE_missing_sale_is_3920_l1100_110073

/-- Calculates the missing sale amount given the sales for 5 months and the desired average -/
def calculate_missing_sale (sales : List ℕ) (average : ℕ) : ℕ :=
  6 * average - sales.sum

/-- The list of known sales amounts -/
def known_sales : List ℕ := [3435, 3855, 4230, 3560, 2000]

/-- The desired average sale -/
def desired_average : ℕ := 3500

theorem missing_sale_is_3920 :
  calculate_missing_sale known_sales desired_average = 3920 := by
  sorry

#eval calculate_missing_sale known_sales desired_average

end NUMINAMATH_CALUDE_missing_sale_is_3920_l1100_110073


namespace NUMINAMATH_CALUDE_car_trip_speed_l1100_110037

/-- Given a car trip with the following properties:
  1. The car averages a certain speed for the first 6 hours.
  2. The car averages 46 miles per hour for each additional hour.
  3. The average speed for the entire trip is 34 miles per hour.
  4. The trip is 8 hours long.
  Prove that the average speed for the first 6 hours of the trip is 30 miles per hour. -/
theorem car_trip_speed (initial_speed : ℝ) : initial_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_speed_l1100_110037


namespace NUMINAMATH_CALUDE_stating_plant_distribution_theorem_l1100_110071

/-- Represents the number of ways to distribute plants among lamps -/
def plant_distribution_ways : ℕ := 9

/-- The number of cactus plants -/
def num_cactus : ℕ := 3

/-- The number of bamboo plants -/
def num_bamboo : ℕ := 2

/-- The number of blue lamps -/
def num_blue_lamps : ℕ := 3

/-- The number of green lamps -/
def num_green_lamps : ℕ := 2

/-- 
Theorem stating that the number of ways to distribute the plants among the lamps is 9,
given the specified numbers of plants and lamps.
-/
theorem plant_distribution_theorem : 
  plant_distribution_ways = 9 := by sorry

end NUMINAMATH_CALUDE_stating_plant_distribution_theorem_l1100_110071


namespace NUMINAMATH_CALUDE_complex_plane_theorem_l1100_110090

def complex_plane_problem (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im
  Complex.abs z = Real.sqrt 2 ∧
  (z^2).im = 2 ∧
  x > 0 ∧ y > 0 →
  z = Complex.mk 1 1 ∧
  let A := z
  let B := z^2
  let C := z - z^2
  let cos_ABC := ((B.re - A.re) * (C.re - B.re) + (B.im - A.im) * (C.im - B.im)) /
                 (Complex.abs (B - A) * Complex.abs (C - B))
  cos_ABC = 3 * Real.sqrt 10 / 10

theorem complex_plane_theorem :
  ∃ z : ℂ, complex_plane_problem z :=
sorry

end NUMINAMATH_CALUDE_complex_plane_theorem_l1100_110090


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l1100_110098

/-- The number of basic ice cream flavors -/
def num_flavors : ℕ := 4

/-- The number of scoops used to create a new flavor -/
def num_scoops : ℕ := 5

/-- The number of ways to distribute n identical objects into k distinct categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem ice_cream_flavors : 
  distribute num_scoops num_flavors = 56 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l1100_110098


namespace NUMINAMATH_CALUDE_inequality_proof_l1100_110033

theorem inequality_proof (x y z : ℝ) (h1 : x < 0) (h2 : x < y) (h3 : y < z) :
  x + y < y + z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1100_110033


namespace NUMINAMATH_CALUDE_some_number_value_l1100_110029

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 45 * 49) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1100_110029


namespace NUMINAMATH_CALUDE_pizza_difference_l1100_110087

/-- Given that Seung-hyeon gave Su-yeon 2 pieces of pizza and then had 5 more pieces than Su-yeon,
    prove that Seung-hyeon had 9 more pieces than Su-yeon before giving. -/
theorem pizza_difference (s y : ℕ) : 
  s - 2 = y + 2 + 5 → s - y = 9 := by
  sorry

end NUMINAMATH_CALUDE_pizza_difference_l1100_110087


namespace NUMINAMATH_CALUDE_parallelogram_area_36_24_l1100_110083

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 36 cm and height 24 cm is 864 cm² -/
theorem parallelogram_area_36_24 :
  parallelogram_area 36 24 = 864 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_36_24_l1100_110083


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1100_110056

theorem trigonometric_simplification :
  let sin15 := Real.sin (15 * π / 180)
  let sin30 := Real.sin (30 * π / 180)
  let sin45 := Real.sin (45 * π / 180)
  let sin60 := Real.sin (60 * π / 180)
  let sin75 := Real.sin (75 * π / 180)
  let cos15 := Real.cos (15 * π / 180)
  let cos30 := Real.cos (30 * π / 180)
  (sin15 + sin30 + sin45 + sin60 + sin75) / (sin15 * cos15 * cos30) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1100_110056


namespace NUMINAMATH_CALUDE_max_value_theorem_l1100_110077

theorem max_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 3*x*y + 5*y^2 = 9) :
  ∃ (m : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 → a^2 - 3*a*b + 5*b^2 = 9 →
  a^2 + 3*a*b + 5*b^2 ≤ m ∧ m = (315 + 297 * Real.sqrt 5) / 55 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1100_110077


namespace NUMINAMATH_CALUDE_value_of_a_l1100_110058

theorem value_of_a (a b c : ℝ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1100_110058


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1100_110005

theorem modulus_of_complex_number : Complex.abs (2 / (1 + Complex.I)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1100_110005


namespace NUMINAMATH_CALUDE_sine_inequality_l1100_110074

theorem sine_inequality (t : ℝ) (h1 : 0 < t) (h2 : t ≤ π / 2) :
  1 / (Real.sin t)^2 ≤ 1 / t^2 + 1 - 4 / π^2 := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l1100_110074


namespace NUMINAMATH_CALUDE_simplify_expression_l1100_110064

theorem simplify_expression : 80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1100_110064


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1100_110024

/-- If a point P(a-1, a+2) lies on the x-axis, then P = (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a - 1 ∧ P.2 = a + 2 ∧ P.2 = 0) → 
  ∃ P : ℝ × ℝ, P = (-3, 0) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1100_110024


namespace NUMINAMATH_CALUDE_m_plus_n_equals_one_l1100_110046

theorem m_plus_n_equals_one (m n : ℤ) (h : |m - 2| + (n + 1)^2 = 0) : m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_m_plus_n_equals_one_l1100_110046


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1100_110079

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    if a_2 * a_3 = 2 * a_1 and 5/4 is the arithmetic mean of a_4 and 2 * a_7,
    then q = 1/2 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q))
  (h3 : a 2 * a 3 = 2 * a 1)
  (h4 : (a 4 + 2 * a 7) / 2 = 5 / 4)
  : q = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1100_110079


namespace NUMINAMATH_CALUDE_simplify_expression_l1100_110054

theorem simplify_expression (w : ℝ) : 3*w + 6*w + 9*w + 12*w + 15*w + 18 + 24 = 45*w + 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1100_110054


namespace NUMINAMATH_CALUDE_complex_modulus_evaluation_l1100_110019

theorem complex_modulus_evaluation :
  Complex.abs (3 - 5*I + (-2 + (3/4)*I)) = (Real.sqrt 305) / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_evaluation_l1100_110019


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l1100_110086

theorem average_age_after_leaving (initial_people : ℕ) (initial_avg : ℚ) (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 7 →
  initial_avg = 32 →
  leaving_age = 22 →
  remaining_people = 6 →
  (initial_people * initial_avg - leaving_age) / remaining_people = 34 := by
sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l1100_110086


namespace NUMINAMATH_CALUDE_f_min_value_iff_a_range_l1100_110068

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2*a*x - 2 else x + 36/x - 6*a

-- State the theorem
theorem f_min_value_iff_a_range (a : ℝ) :
  (∀ x : ℝ, f a 2 ≤ f a x) ↔ 2 ≤ a ∧ a ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_iff_a_range_l1100_110068


namespace NUMINAMATH_CALUDE_sequence_divisibility_and_conditions_l1100_110010

def a (n : ℕ) : ℤ := 15 * n + 2 + (15 * n - 32) * 16^(n - 1)

theorem sequence_divisibility_and_conditions :
  (∀ n : ℕ, (15^3 : ℤ) ∣ a n) ∧
  (∀ n : ℕ, (1991 : ℤ) ∣ a n ∧ (1991 : ℤ) ∣ a (n + 1) ∧ (1991 : ℤ) ∣ a (n + 2) ↔ ∃ k : ℕ, n = 89595 * k) :=
sorry

end NUMINAMATH_CALUDE_sequence_divisibility_and_conditions_l1100_110010


namespace NUMINAMATH_CALUDE_complement_angle_l1100_110063

theorem complement_angle (A : ℝ) (h : A = 25) : 90 - A = 65 := by
  sorry

end NUMINAMATH_CALUDE_complement_angle_l1100_110063


namespace NUMINAMATH_CALUDE_percentage_increase_l1100_110030

theorem percentage_increase (initial final : ℝ) (h : initial > 0) :
  let increase := (final - initial) / initial * 100
  initial = 200 ∧ final = 250 → increase = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1100_110030


namespace NUMINAMATH_CALUDE_series_sum_equals_two_l1100_110026

/-- Given a real number k > 1 such that the infinite sum of (7n-3)/k^n from n=1 to infinity equals 2,
    prove that k = 2 + (3√2)/2 -/
theorem series_sum_equals_two (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 3) / k^n = 2) : k = 2 + 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_two_l1100_110026


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l1100_110066

theorem complex_roots_theorem (x y : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (x + 3 * Complex.I) * (x + 3 * Complex.I) - (13 + 12 * Complex.I) * (x + 3 * Complex.I) + (15 + 72 * Complex.I) = 0 ∧
  (y + 6 * Complex.I) * (y + 6 * Complex.I) - (13 + 12 * Complex.I) * (y + 6 * Complex.I) + (15 + 72 * Complex.I) = 0 →
  x = 11 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l1100_110066


namespace NUMINAMATH_CALUDE_prairie_area_l1100_110042

/-- The total area of a prairie given the dust-covered and untouched areas -/
theorem prairie_area (dust_covered : ℕ) (untouched : ℕ) 
  (h1 : dust_covered = 64535) 
  (h2 : untouched = 522) : 
  dust_covered + untouched = 65057 := by
  sorry

end NUMINAMATH_CALUDE_prairie_area_l1100_110042


namespace NUMINAMATH_CALUDE_total_towels_calculation_l1100_110015

/-- The number of loads of laundry washed -/
def loads : ℕ := 6

/-- The number of towels in each load -/
def towels_per_load : ℕ := 7

/-- The total number of towels washed -/
def total_towels : ℕ := loads * towels_per_load

theorem total_towels_calculation : total_towels = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_towels_calculation_l1100_110015


namespace NUMINAMATH_CALUDE_solve_system_l1100_110022

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1100_110022


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1100_110067

theorem unique_solution_for_equation :
  ∀ x y : ℝ,
    (Real.sqrt (1 / (4 - x^2)) + Real.sqrt (y^2 / (y - 1)) = 5/2) →
    (x = 0 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1100_110067


namespace NUMINAMATH_CALUDE_system_solution_l1100_110002

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := (y^5 / x)^(Real.log x) = y^(2 * Real.log (x * y))
def equation2 (x y : ℝ) : Prop := x^2 - 2*x*y - 4*x - 3*y^2 + 12*y = 0

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(2, 2), (9, 3), ((9 - Real.sqrt 17) / 2, (Real.sqrt 17 - 1) / 2)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), x > 0 ∧ y > 0 →
  (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l1100_110002


namespace NUMINAMATH_CALUDE_difference_not_one_l1100_110035

theorem difference_not_one (a b : ℝ) (h : a^2 - b^2 + 2*a - 4*b - 3 ≠ 0) : a - b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_not_one_l1100_110035


namespace NUMINAMATH_CALUDE_probability_not_triangle_l1100_110007

theorem probability_not_triangle (total : ℕ) (triangles : ℕ) 
  (h1 : total = 10) (h2 : triangles = 4) : 
  (total - triangles : ℚ) / total = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_triangle_l1100_110007
