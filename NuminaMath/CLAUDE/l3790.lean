import Mathlib

namespace reinforcement_size_correct_l3790_379089

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days before reinforcement, and remaining provision duration after reinforcement. -/
def reinforcement_size (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  let remaining_provisions := initial_garrison * (initial_duration - days_before_reinforcement)
  let total_men_days := remaining_provisions
  (total_men_days / remaining_duration) - initial_garrison

theorem reinforcement_size_correct (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (remaining_duration : ℕ) 
    (h1 : initial_garrison = 2000)
    (h2 : initial_duration = 54)
    (h3 : days_before_reinforcement = 15)
    (h4 : remaining_duration = 20) :
  reinforcement_size initial_garrison initial_duration days_before_reinforcement remaining_duration = 1900 := by
  sorry

#eval reinforcement_size 2000 54 15 20

end reinforcement_size_correct_l3790_379089


namespace men_in_room_l3790_379019

theorem men_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (2 * (initial_women - 3) = 24) →
  (initial_men + 2 = 14) :=
by
  sorry

#check men_in_room

end men_in_room_l3790_379019


namespace athletes_division_l3790_379010

theorem athletes_division (n : ℕ) (k : ℕ) : n = 10 ∧ k = 5 → (n.choose k) / 2 = 126 := by
  sorry

end athletes_division_l3790_379010


namespace graph_intersection_sum_l3790_379007

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop :=
  (x^2 + y^2 - 2*x)^2 = 2*(x^2 + y^2)^2

/-- The number of points where the graph meets the x-axis -/
def p : ℕ :=
  3  -- This is given as a fact from the problem, not derived from the solution

/-- The number of points where the graph meets the y-axis -/
def q : ℕ :=
  1  -- This is given as a fact from the problem, not derived from the solution

/-- The theorem to be proved -/
theorem graph_intersection_sum : 100 * p + 100 * q = 400 := by
  sorry

end graph_intersection_sum_l3790_379007


namespace point_minimizing_distance_sum_l3790_379004

/-- The point that minimizes the sum of distances to two fixed points on a line --/
theorem point_minimizing_distance_sum 
  (M N P : ℝ × ℝ) 
  (hM : M = (1, 2)) 
  (hN : N = (4, 6)) 
  (hP : P.2 = P.1 - 1) 
  (h_min : ∀ Q : ℝ × ℝ, Q.2 = Q.1 - 1 → 
    dist P M + dist P N ≤ dist Q M + dist Q N) : 
  P = (17/5, 12/5) := by
  sorry

-- where
-- dist : ℝ × ℝ → ℝ × ℝ → ℝ 
-- represents the Euclidean distance between two points

end point_minimizing_distance_sum_l3790_379004


namespace car_wash_group_composition_l3790_379084

theorem car_wash_group_composition (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls : ℚ) / total = 2 / 5 →
  ((initial_girls : ℚ) - 2) / total = 3 / 10 →
  initial_girls = 8 := by
sorry

end car_wash_group_composition_l3790_379084


namespace point_C_values_l3790_379087

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Represents the number line with three points -/
structure NumberLine where
  A : Point
  B : Point
  C : Point

/-- Checks if folding at one point makes the other two coincide -/
def foldingCondition (line : NumberLine) : Prop :=
  (abs (line.A.value - line.B.value) = 2 * abs (line.A.value - line.C.value)) ∨
  (abs (line.A.value - line.B.value) = 2 * abs (line.B.value - line.C.value)) ∨
  (abs (line.A.value - line.C.value) = abs (line.B.value - line.C.value))

/-- The main theorem to prove -/
theorem point_C_values (line : NumberLine) :
  ((line.A.value + 3)^2 + abs (line.B.value - 1) = 0) →
  foldingCondition line →
  (line.C.value = -7 ∨ line.C.value = -1 ∨ line.C.value = 5) := by
  sorry

end point_C_values_l3790_379087


namespace geometry_rhyme_probability_l3790_379042

def geometry_letters : Finset Char := {'G', 'E', 'O', 'M', 'E', 'T', 'R', 'Y'}
def rhyme_letters : Finset Char := {'R', 'H', 'Y', 'M', 'E'}

theorem geometry_rhyme_probability :
  (geometry_letters ∩ rhyme_letters).card / geometry_letters.card = 1 / 2 := by
  sorry

end geometry_rhyme_probability_l3790_379042


namespace geometric_sequence_product_l3790_379058

theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (a 3 * a 3 - 5 * a 3 + 4 = 0) →             -- a_3 is a root of x^2 - 5x + 4 = 0
  (a 5 * a 5 - 5 * a 5 + 4 = 0) →             -- a_5 is a root of x^2 - 5x + 4 = 0
  (a 2 * a 4 * a 6 = 8 ∨ a 2 * a 4 * a 6 = -8) :=
by sorry

end geometric_sequence_product_l3790_379058


namespace inequality_counterexample_l3790_379096

theorem inequality_counterexample (a b : ℝ) (h : a < b) :
  ∃ c : ℝ, ¬(a * c < b * c) := by
  sorry

end inequality_counterexample_l3790_379096


namespace megans_earnings_l3790_379067

/-- Calculates the total earnings for a given number of months based on daily work hours, hourly rate, and days worked per month. -/
def total_earnings (hours_per_day : ℕ) (hourly_rate : ℚ) (days_per_month : ℕ) (months : ℕ) : ℚ :=
  hours_per_day * hourly_rate * days_per_month * months

/-- Proves that Megan's total earnings for two months of work is $2400. -/
theorem megans_earnings :
  total_earnings 8 (15/2) 20 2 = 2400 := by
  sorry

#eval total_earnings 8 (15/2) 20 2

end megans_earnings_l3790_379067


namespace max_third_altitude_exists_max_altitude_l3790_379099

/-- An isosceles triangle with specific altitude properties -/
structure IsoscelesTriangle where
  -- The lengths of the sides
  AB : ℝ
  BC : ℝ
  -- The altitudes
  h_AB : ℝ
  h_AC : ℝ
  h_BC : ℕ
  -- Isosceles property
  isIsosceles : AB = BC
  -- Given altitude lengths
  alt_AB : h_AB = 6
  alt_AC : h_AC = 18
  -- Triangle inequality
  triangle_inequality : AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB

/-- The theorem stating the maximum possible integer length of the third altitude -/
theorem max_third_altitude (t : IsoscelesTriangle) : t.h_BC ≤ 6 := by
  sorry

/-- The existence of such a triangle with the maximum third altitude -/
theorem exists_max_altitude : ∃ t : IsoscelesTriangle, t.h_BC = 6 := by
  sorry

end max_third_altitude_exists_max_altitude_l3790_379099


namespace problem_solution_l3790_379048

theorem problem_solution : (((3⁻¹ : ℚ) + 7^3 - 2)⁻¹ * 7 : ℚ) = 21 / 1024 := by
  sorry

end problem_solution_l3790_379048


namespace hexagon_area_l3790_379006

theorem hexagon_area (s : ℝ) (t : ℝ) : 
  s > 0 → t > 0 →
  (4 * s = 6 * t) →  -- Equal perimeters
  (s^2 = 16) →       -- Area of square is 16
  (6 * (t^2 * Real.sqrt 3) / 4) = (64 * Real.sqrt 3) / 3 := by
  sorry

end hexagon_area_l3790_379006


namespace sequence_convergence_l3790_379027

def converges (a : ℕ → ℝ) : Prop :=
  ∃ (l : ℝ), ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l| < ε

theorem sequence_convergence
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_ineq : ∀ n ≥ 2, a (n + 1) ≤ (a n * (a (n - 1))^2)^(1/3)) :
  converges a :=
sorry

end sequence_convergence_l3790_379027


namespace simplify_expression_l3790_379070

theorem simplify_expression : ((4 + 6) * 2) / 4 - 1 / 4 = 4.75 := by
  sorry

end simplify_expression_l3790_379070


namespace smallest_whole_number_above_sum_l3790_379081

theorem smallest_whole_number_above_sum : ∃ n : ℕ, 
  (n : ℚ) > (3 + 1/3 : ℚ) + (4 + 1/6 : ℚ) + (5 + 1/12 : ℚ) + (6 + 1/8 : ℚ) ∧
  ∀ m : ℕ, (m : ℚ) > (3 + 1/3 : ℚ) + (4 + 1/6 : ℚ) + (5 + 1/12 : ℚ) + (6 + 1/8 : ℚ) → n ≤ m :=
by
  -- Proof goes here
  sorry

end smallest_whole_number_above_sum_l3790_379081


namespace solve_product_equation_l3790_379035

theorem solve_product_equation : 
  ∀ x : ℝ, 6 * (x - 3) * (x + 5) = 0 ↔ x = -5 ∨ x = 3 := by
  sorry

end solve_product_equation_l3790_379035


namespace smallest_common_multiple_of_8_and_6_l3790_379052

theorem smallest_common_multiple_of_8_and_6 : 
  ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ (m : ℕ), m > 0 → 8 ∣ m → 6 ∣ m → n ≤ m := by
  sorry

end smallest_common_multiple_of_8_and_6_l3790_379052


namespace distance_between_trees_l3790_379072

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : 
  yard_length = 414 → num_trees = 24 → 
  (yard_length : ℚ) / (num_trees - 1 : ℚ) = 18 := by
  sorry

end distance_between_trees_l3790_379072


namespace arithmetic_sequence_sum_property_l3790_379040

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_4 + a_8 = 16, then a_2 + a_10 = 16 -/
theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ) (h_arithmetic : arithmetic_sequence a) (h_sum : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 := by
  sorry

end arithmetic_sequence_sum_property_l3790_379040


namespace f_value_at_negative_pi_third_l3790_379002

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (Real.cos x)^2 - b * Real.sin x * Real.cos x - a / 2

theorem f_value_at_negative_pi_third (a b : ℝ) :
  (∃ (x : ℝ), f a b x ≤ 1/2) ∧
  (f a b (π/3) = Real.sqrt 3 / 4) →
  (f a b (-π/3) = 0 ∨ f a b (-π/3) = -Real.sqrt 3 / 4) :=
by sorry

end f_value_at_negative_pi_third_l3790_379002


namespace sum_of_derived_geometric_progression_l3790_379095

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℕ  -- First term
  d : ℕ  -- Common difference
  sum_first_three : a + (a + d) + (a + 2 * d) = 21
  increasing : d > 0

/-- A geometric progression derived from the arithmetic progression -/
def geometric_from_arithmetic (ap : ArithmeticProgression) : Fin 3 → ℕ
  | 0 => ap.a - 1
  | 1 => ap.a + ap.d - 1
  | 2 => ap.a + 2 * ap.d + 2

/-- The theorem to be proved -/
theorem sum_of_derived_geometric_progression (ap : ArithmeticProgression) :
  let gp := geometric_from_arithmetic ap
  let q := gp 1 / gp 0  -- Common ratio of the geometric progression
  gp 0 * (q^8 - 1) / (q - 1) = 765 := by
  sorry


end sum_of_derived_geometric_progression_l3790_379095


namespace solve_system_l3790_379078

theorem solve_system (x y : ℝ) (h1 : 3 * x + y = 75) (h2 : 2 * (3 * x + y) - y = 138) : x = 21 := by
  sorry

end solve_system_l3790_379078


namespace num_ways_to_achieve_18_with_5_dice_l3790_379005

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 5

/-- The target sum we're aiming for -/
def targetSum : ℕ := 18

/-- A function that calculates the number of ways to achieve the target sum -/
def numWaysToAchieveSum (faces : ℕ) (dice : ℕ) (sum : ℕ) : ℕ := sorry

theorem num_ways_to_achieve_18_with_5_dice : 
  numWaysToAchieveSum numFaces numDice targetSum = 651 := by sorry

end num_ways_to_achieve_18_with_5_dice_l3790_379005


namespace lunch_cost_proof_l3790_379063

theorem lunch_cost_proof (adam_cost rick_cost jose_cost total_cost : ℚ) : 
  adam_cost = (2 : ℚ) / (3 : ℚ) * rick_cost →
  rick_cost = jose_cost →
  jose_cost = 45 →
  total_cost = adam_cost + rick_cost + jose_cost →
  total_cost = 120 := by
sorry

end lunch_cost_proof_l3790_379063


namespace complement_A_in_U_l3790_379080

-- Define the universal set U
def U : Set ℝ := {x | x > 1}

-- Define set A
def A : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end complement_A_in_U_l3790_379080


namespace jos_number_l3790_379041

theorem jos_number (n : ℕ) : 
  (∃ k l : ℕ, n = 9 * k - 2 ∧ n = 6 * l - 4) ∧ 
  n < 100 ∧ 
  (∀ m : ℕ, m < 100 → (∃ k' l' : ℕ, m = 9 * k' - 2 ∧ m = 6 * l' - 4) → m ≤ n) → 
  n = 86 := by
sorry

end jos_number_l3790_379041


namespace no_perfect_power_in_sequence_l3790_379014

/-- Represents a triple in the sequence -/
structure Triple where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Generates the next triple from the current one -/
def nextTriple (t : Triple) : Triple :=
  { a := t.a * t.b,
    b := t.b * t.c,
    c := t.c * t.a }

/-- Checks if a number is a perfect power -/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ k m : ℕ, m ≥ 2 ∧ n = k^m

/-- The sequence of triples starting with (2,3,5) -/
def tripleSequence : ℕ → Triple
  | 0 => { a := 2, b := 3, c := 5 }
  | n + 1 => nextTriple (tripleSequence n)

/-- Theorem: No number in any triple of the sequence is a perfect power -/
theorem no_perfect_power_in_sequence :
  ∀ n : ℕ, ¬(isPerfectPower (tripleSequence n).a ∨
            isPerfectPower (tripleSequence n).b ∨
            isPerfectPower (tripleSequence n).c) :=
by
  sorry


end no_perfect_power_in_sequence_l3790_379014


namespace arithmetic_mean_problem_l3790_379045

theorem arithmetic_mean_problem : ∃ (x y : ℝ), 
  ((x + 12) + y + 3*x + 18 + (3*x + 6)) / 5 = 30 := by
  sorry

end arithmetic_mean_problem_l3790_379045


namespace probability_of_mixed_team_l3790_379039

def num_girls : ℕ := 3
def num_boys : ℕ := 2
def team_size : ℕ := 2
def total_group_size : ℕ := num_girls + num_boys

def num_total_combinations : ℕ := (total_group_size.choose team_size)
def num_mixed_combinations : ℕ := num_girls * num_boys

theorem probability_of_mixed_team :
  (num_mixed_combinations : ℚ) / num_total_combinations = 3 / 5 := by sorry

end probability_of_mixed_team_l3790_379039


namespace dacid_weighted_average_score_l3790_379044

/-- Calculates the weighted average score for a student given their marks and subject weightages --/
def weighted_average_score (
  english_mark : ℚ)
  (math_mark : ℚ)
  (physics_mark : ℚ)
  (chemistry_mark : ℚ)
  (biology_mark : ℚ)
  (cs_mark : ℚ)
  (sports_mark : ℚ)
  (english_weight : ℚ)
  (math_weight : ℚ)
  (physics_weight : ℚ)
  (chemistry_weight : ℚ)
  (biology_weight : ℚ)
  (cs_weight : ℚ)
  (sports_weight : ℚ) : ℚ :=
  english_mark * english_weight +
  math_mark * math_weight +
  physics_mark * physics_weight +
  chemistry_mark * chemistry_weight +
  biology_mark * biology_weight +
  (cs_mark * 100 / 150) * cs_weight +
  (sports_mark * 100 / 150) * sports_weight

/-- Theorem stating that Dacid's weighted average score is approximately 86.82 --/
theorem dacid_weighted_average_score :
  ∃ ε > 0, abs (weighted_average_score 96 95 82 97 95 88 83 0.25 0.20 0.10 0.15 0.10 0.15 0.05 - 86.82) < ε :=
by
  sorry

end dacid_weighted_average_score_l3790_379044


namespace A_minus_2B_A_minus_2B_special_case_A_minus_2B_independent_of_x_l3790_379093

-- Define the expressions A and B
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y
def B (x y : ℝ) : ℝ := x^2 - x * y + x

-- Theorem 1: A - 2B = 5xy - 2x + 2y
theorem A_minus_2B (x y : ℝ) : A x y - 2 * B x y = 5 * x * y - 2 * x + 2 * y := by sorry

-- Theorem 2: When x² = 9 and |y| = 2, A - 2B ∈ {28, -40, -20, 32}
theorem A_minus_2B_special_case (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 2) :
  A x y - 2 * B x y ∈ ({28, -40, -20, 32} : Set ℝ) := by sorry

-- Theorem 3: If A - 2B is independent of x, then y = 2/5
theorem A_minus_2B_independent_of_x (y : ℝ) :
  (∀ x : ℝ, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2/5 := by sorry

end A_minus_2B_A_minus_2B_special_case_A_minus_2B_independent_of_x_l3790_379093


namespace rectangle_area_value_l3790_379053

theorem rectangle_area_value (y : ℝ) : 
  y > 1 → 
  (3 : ℝ) * (y - 1) = 36 → 
  y = 13 := by
sorry

end rectangle_area_value_l3790_379053


namespace parallelogram_height_l3790_379082

theorem parallelogram_height (area base height : ℝ) : 
  area = 364 ∧ base = 26 ∧ area = base * height → height = 14 := by
  sorry

end parallelogram_height_l3790_379082


namespace cars_without_ac_l3790_379077

/-- Given a group of cars with the following properties:
  * There are 100 cars in total
  * At least 53 cars have racing stripes
  * The greatest number of cars that could have air conditioning but not racing stripes is 47
  Prove that the number of cars without air conditioning is 47. -/
theorem cars_without_ac (total : ℕ) (with_stripes : ℕ) (ac_no_stripes : ℕ)
  (h1 : total = 100)
  (h2 : with_stripes ≥ 53)
  (h3 : ac_no_stripes = 47) :
  total - (ac_no_stripes + (with_stripes - ac_no_stripes)) = 47 := by
  sorry

end cars_without_ac_l3790_379077


namespace bag_probability_l3790_379013

theorem bag_probability (n : ℕ) : 
  (5 : ℚ) / (n + 5) = 1 / 3 → n = 10 := by
  sorry

end bag_probability_l3790_379013


namespace light_distance_half_year_l3790_379012

/-- The speed of light in kilometers per second -/
def speed_of_light : ℝ := 299792

/-- The number of days in half a year -/
def half_year_days : ℝ := 182.5

/-- The distance light travels in half a year -/
def light_distance : ℝ := speed_of_light * half_year_days * 24 * 3600

theorem light_distance_half_year :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 * 10^12 ∧ 
  |light_distance - 4.73 * 10^12| < ε :=
sorry

end light_distance_half_year_l3790_379012


namespace min_f_1998_l3790_379024

/-- A function from positive integers to positive integers satisfying the given property -/
def SpecialFunction (f : ℕ+ → ℕ+) : Prop :=
  ∀ (s t : ℕ+), f (t^2 * f s) = s * (f t)^2

/-- The theorem stating the minimum value of f(1998) -/
theorem min_f_1998 (f : ℕ+ → ℕ+) (h : SpecialFunction f) :
  ∃ (m : ℕ+), f 1998 = m ∧ ∀ (g : ℕ+ → ℕ+), SpecialFunction g → m ≤ g 1998 :=
sorry

end min_f_1998_l3790_379024


namespace least_xy_value_l3790_379056

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) :
  (x * y : ℕ) ≥ 96 ∧ ∃ (a b : ℕ+), (a : ℚ) / b + (1 : ℚ) / (3 * b) = (1 : ℚ) / 8 ∧ (a * b : ℕ) = 96 :=
sorry

end least_xy_value_l3790_379056


namespace jordan_running_time_l3790_379036

/-- Given Steve's running time and distance, and Jordan's relative speed,
    calculate Jordan's time to run a specified distance. -/
theorem jordan_running_time
  (steve_distance : ℝ)
  (steve_time : ℝ)
  (jordan_relative_speed : ℝ)
  (jordan_distance : ℝ)
  (h1 : steve_distance = 6)
  (h2 : steve_time = 36)
  (h3 : jordan_relative_speed = 3)
  (h4 : jordan_distance = 8) :
  (jordan_distance * steve_time) / (steve_distance * jordan_relative_speed) = 24 :=
by sorry

end jordan_running_time_l3790_379036


namespace car_value_after_depreciation_l3790_379017

/-- Calculates the current value of a car given its initial price and depreciation rate. -/
def currentCarValue (initialPrice : ℝ) (depreciationRate : ℝ) : ℝ :=
  initialPrice * (1 - depreciationRate)

/-- Theorem stating that a car initially priced at $4000 with 30% depreciation is now worth $2800. -/
theorem car_value_after_depreciation :
  currentCarValue 4000 0.3 = 2800 := by
  sorry

end car_value_after_depreciation_l3790_379017


namespace missing_number_value_l3790_379064

theorem missing_number_value (a b some_number : ℕ) : 
  a = 105 → 
  b = 147 → 
  a^3 = 21 * 25 * some_number * b → 
  some_number = 3 := by
sorry

end missing_number_value_l3790_379064


namespace precy_age_l3790_379030

/-- Represents the ages of Alex and Precy -/
structure Ages where
  alex : ℕ
  precy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.alex = 15 ∧
  (ages.alex + 3) = 3 * (ages.precy + 3) ∧
  (ages.alex - 1) = 7 * (ages.precy - 1)

/-- The theorem stating that under the given conditions, Precy's age is 3 -/
theorem precy_age (ages : Ages) : problem_conditions ages → ages.precy = 3 := by
  sorry

end precy_age_l3790_379030


namespace equal_intercept_line_equation_l3790_379020

/-- A line passing through (1, 2) with equal intercepts on both axes -/
def EqualInterceptLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 = 1 ∧ p.2 = 2) ∨ 
               (p.2 = 2 * p.1) ∨ 
               (p.1 + p.2 = 3)}

theorem equal_intercept_line_equation :
  ∀ (x y : ℝ), (x, y) ∈ EqualInterceptLine ↔ (y = 2 * x ∨ x + y = 3) :=
by sorry

end equal_intercept_line_equation_l3790_379020


namespace shortest_side_of_right_triangle_l3790_379011

theorem shortest_side_of_right_triangle (a b c : ℝ) : 
  a = 7 → b = 24 → c^2 = a^2 + b^2 → a ≤ b ∧ a ≤ c := by
  sorry

end shortest_side_of_right_triangle_l3790_379011


namespace regression_prediction_l3790_379061

/-- Represents the regression equation y = mx + b -/
structure RegressionLine where
  m : ℝ
  b : ℝ

/-- Calculates the y-value for a given x using the regression line -/
def RegressionLine.predict (line : RegressionLine) (x : ℝ) : ℝ :=
  line.m * x + line.b

theorem regression_prediction 
  (line : RegressionLine)
  (h1 : line.m = 9.4)
  (h2 : line.predict 3.5 = 42)
  : line.predict 6 = 65.5 := by
  sorry

end regression_prediction_l3790_379061


namespace max_inscribed_right_triangles_l3790_379018

/-- Represents an ellipse with equation x^2 + a^2 * y^2 = a^2 where a > 1 -/
structure Ellipse where
  a : ℝ
  h_a_gt_one : a > 1

/-- Represents a right triangle inscribed in the ellipse with C(0, 1) as the right angle -/
structure InscribedRightTriangle (e : Ellipse) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_ellipse_A : A.1^2 + e.a^2 * A.2^2 = e.a^2
  h_on_ellipse_B : B.1^2 + e.a^2 * B.2^2 = e.a^2
  h_right_angle : (A.1 - 0) * (B.1 - 0) + (A.2 - 1) * (B.2 - 1) = 0

/-- The theorem stating the maximum number of inscribed right triangles -/
theorem max_inscribed_right_triangles (e : Ellipse) : 
  (∃ (n : ℕ), ∀ (m : ℕ), (∃ (f : Fin m → InscribedRightTriangle e), Function.Injective f) → m ≤ n) ∧ 
  (∃ (f : Fin 3 → InscribedRightTriangle e), Function.Injective f) := by
  sorry

end max_inscribed_right_triangles_l3790_379018


namespace tysons_three_pointers_l3790_379032

/-- Tyson's basketball scoring problem -/
theorem tysons_three_pointers (x : ℕ) : 
  (3 * x + 2 * 12 + 1 * 6 = 75) → x = 15 := by
  sorry

end tysons_three_pointers_l3790_379032


namespace soda_price_proof_l3790_379083

/-- Given a regular price per can of soda, prove that it equals $0.55 under the given conditions -/
theorem soda_price_proof (P : ℝ) : 
  (∃ (discounted_price : ℝ), 
    discounted_price = 0.75 * P ∧ 
    70 * discounted_price = 28.875) → 
  P = 0.55 := by
sorry

end soda_price_proof_l3790_379083


namespace student_mistake_fraction_l3790_379098

theorem student_mistake_fraction (x y : ℚ) : 
  (x / y) * 576 = 480 → x / y = 5 / 6 :=
by sorry

end student_mistake_fraction_l3790_379098


namespace sqrt_sum_reciprocal_l3790_379088

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end sqrt_sum_reciprocal_l3790_379088


namespace tenth_difference_optimal_number_l3790_379025

/-- A positive integer that can be expressed as the difference of squares of two positive integers m and n, where m - n > 1 -/
def DifferenceOptimalNumber (k : ℕ) : Prop :=
  ∃ m n : ℕ, m > n ∧ m - n > 1 ∧ k = m^2 - n^2

/-- The sequence of difference optimal numbers in ascending order -/
def DifferenceOptimalSequence : ℕ → ℕ :=
  sorry

theorem tenth_difference_optimal_number :
  DifferenceOptimalNumber (DifferenceOptimalSequence 10) ∧ 
  DifferenceOptimalSequence 10 = 32 := by
  sorry

end tenth_difference_optimal_number_l3790_379025


namespace intersection_M_N_l3790_379034

def M : Set ℝ := {x | x^2 > 1}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, 2} := by sorry

end intersection_M_N_l3790_379034


namespace remainder_of_sum_of_fourth_powers_l3790_379097

theorem remainder_of_sum_of_fourth_powers (x y : ℕ+) (P Q : ℕ) :
  x^4 + y^4 = (x + y) * (P + 13) + Q ∧ Q < x + y →
  Q = 8 :=
by sorry

end remainder_of_sum_of_fourth_powers_l3790_379097


namespace palindrome_product_sum_theorem_l3790_379071

/-- A positive three-digit palindrome is a number between 100 and 999 inclusive,
    where the first and third digits are the same. -/
def IsPositiveThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- There exist two positive three-digit palindromes whose product is 589185 and whose sum is 1534. -/
theorem palindrome_product_sum_theorem :
  ∃ (a b : ℕ), IsPositiveThreeDigitPalindrome a ∧
                IsPositiveThreeDigitPalindrome b ∧
                a * b = 589185 ∧
                a + b = 1534 := by
  sorry

end palindrome_product_sum_theorem_l3790_379071


namespace figure_segments_length_l3790_379021

theorem figure_segments_length 
  (rectangle_length : ℝ) 
  (rectangle_breadth : ℝ) 
  (square_side : ℝ) 
  (h1 : rectangle_length = 10) 
  (h2 : rectangle_breadth = 6) 
  (h3 : square_side = 4) :
  square_side + 2 * rectangle_length + rectangle_breadth / 2 = 27 :=
by sorry

end figure_segments_length_l3790_379021


namespace geometric_series_sum_l3790_379038

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum (1/4) (1/4) 6 = 4095/12288 := by
  sorry

end geometric_series_sum_l3790_379038


namespace sum_of_binary_digits_157_l3790_379094

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ := sorry

/-- Sums the digits of a natural number in base 10 -/
def sumDigits (n : ℕ) : ℕ := sorry

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ := sorry

theorem sum_of_binary_digits_157 : 
  let binary157 := toBinary 157
  let sumBinary157 := sumList binary157
  let sumDigits157 := sumDigits 157
  let binarySumDigits157 := toBinary sumDigits157
  let sumBinarySumDigits157 := sumList binarySumDigits157
  sumBinary157 + sumBinarySumDigits157 = 8 := by sorry

end sum_of_binary_digits_157_l3790_379094


namespace snow_probability_l3790_379086

theorem snow_probability (p1 p2 : ℚ) : 
  p1 = 1/4 → p2 = 1/3 → 
  (1 - (1 - p1)^4 * (1 - p2)^3 : ℚ) = 29/32 := by
  sorry

end snow_probability_l3790_379086


namespace even_and_divisible_by_six_l3790_379074

theorem even_and_divisible_by_six (n : ℕ) : 
  (2 ∣ n * (n + 1)) ∧ (6 ∣ n * (n + 1) * (2 * n + 1)) := by
  sorry

end even_and_divisible_by_six_l3790_379074


namespace martha_final_cards_l3790_379022

-- Define the initial number of cards Martha has
def initial_cards : ℝ := 76.0

-- Define the number of cards Martha gives away
def cards_given_away : ℝ := 3.0

-- Theorem statement
theorem martha_final_cards : 
  initial_cards - cards_given_away = 73.0 := by
  sorry

end martha_final_cards_l3790_379022


namespace bus_distance_ratio_l3790_379065

theorem bus_distance_ratio (total_distance : ℝ) (foot_fraction : ℝ) (car_distance : ℝ) :
  total_distance = 40 →
  foot_fraction = 1 / 4 →
  car_distance = 10 →
  (total_distance - (foot_fraction * total_distance + car_distance)) / total_distance = 1 / 2 := by
  sorry

end bus_distance_ratio_l3790_379065


namespace triangle_quadratic_no_roots_l3790_379069

/-- Given a, b, and c are side lengths of a triangle, 
    the quadratic equation (a+b)x^2 + 2cx + a+b = 0 has no real roots -/
theorem triangle_quadratic_no_roots (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) : 
  ∀ x : ℝ, (a + b) * x^2 + 2 * c * x + (a + b) ≠ 0 := by
  sorry

#check triangle_quadratic_no_roots

end triangle_quadratic_no_roots_l3790_379069


namespace sqrt_difference_inequality_l3790_379031

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end sqrt_difference_inequality_l3790_379031


namespace equation_solution_l3790_379090

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (4*x+1)*(3*x+1)*(2*x+1)*(x+1) - 3*x^4
  ∀ x : ℝ, f x = 0 ↔ x = (-5 + Real.sqrt 13) / 6 ∨ x = (-5 - Real.sqrt 13) / 6 := by
sorry

end equation_solution_l3790_379090


namespace square_sum_ge_twice_product_l3790_379055

theorem square_sum_ge_twice_product {x y : ℝ} (h : x ≥ y) : x^2 + y^2 ≥ 2*x*y := by
  sorry

end square_sum_ge_twice_product_l3790_379055


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3790_379000

/-- Proposition p: For all real x, x^2 - 4x + 2m ≥ 0 -/
def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 4*x + 2*m ≥ 0

/-- m ≥ 3 is a sufficient condition for proposition p -/
theorem sufficient_condition (m : ℝ) :
  m ≥ 3 → proposition_p m :=
sorry

/-- m ≥ 3 is not a necessary condition for proposition p -/
theorem not_necessary_condition :
  ∃ m : ℝ, m < 3 ∧ proposition_p m :=
sorry

/-- m ≥ 3 is a sufficient but not necessary condition for proposition p -/
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, m ≥ 3 → proposition_p m) ∧
  (∃ m : ℝ, m < 3 ∧ proposition_p m) :=
sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3790_379000


namespace total_distance_run_l3790_379037

theorem total_distance_run (num_students : ℕ) (avg_distance : ℕ) (h1 : num_students = 18) (h2 : avg_distance = 106) :
  num_students * avg_distance = 1908 := by
  sorry

end total_distance_run_l3790_379037


namespace normal_distribution_probability_l3790_379046

/-- The normal distribution with mean μ and standard deviation σ -/
noncomputable def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) : ℝ → ℝ := sorry

/-- The cumulative distribution function of a normal distribution -/
noncomputable def normal_cdf (μ σ : ℝ) : ℝ → ℝ := sorry

/-- The probability of a random variable X falling within an interval [a, b] -/
noncomputable def prob_interval (μ σ : ℝ) (a b : ℝ) : ℝ :=
  normal_cdf μ σ b - normal_cdf μ σ a

theorem normal_distribution_probability (μ σ : ℝ) :
  (normal_pdf μ σ 0 = 1 / (3 * Real.sqrt (2 * Real.pi))) →
  (prob_interval μ σ (μ - σ) (μ + σ) = 0.6826) →
  (prob_interval μ σ (μ - 2*σ) (μ + 2*σ) = 0.9544) →
  (prob_interval μ σ 3 6 = 0.1359) := by sorry

end normal_distribution_probability_l3790_379046


namespace parabola_unique_values_l3790_379009

/-- A parabola passing through three given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ := fun x ↦ x^2 + b*x + c
  point1 : eq (-2) = -8
  point2 : eq 4 = 28
  point3 : eq 1 = 4

/-- The unique values of b and c for the parabola -/
theorem parabola_unique_values (p : Parabola) : p.b = 4 ∧ p.c = -1 := by
  sorry

end parabola_unique_values_l3790_379009


namespace weekly_social_media_time_l3790_379001

/-- Charlotte's daily phone usage in hours -/
def daily_phone_usage : ℕ := 16

/-- The fraction of phone time spent on social media -/
def social_media_fraction : ℚ := 1/2

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Charlotte spends 56 hours on social media in a week -/
theorem weekly_social_media_time : 
  (daily_phone_usage * social_media_fraction * days_in_week : ℚ) = 56 := by
sorry

end weekly_social_media_time_l3790_379001


namespace inequality_condition_l3790_379029

-- Define the conditions
def has_solutions (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x - a ≤ 0

def condition_q (a : ℝ) : Prop := a > 0 ∨ a < -1

-- State the theorem
theorem inequality_condition :
  (∀ a : ℝ, condition_q a → has_solutions a) ∧
  ¬(∀ a : ℝ, has_solutions a → condition_q a) :=
sorry

end inequality_condition_l3790_379029


namespace triangle_properties_l3790_379028

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) (R : ℝ) :
  R = Real.sqrt 3 →
  (2 * Real.sin A - Real.sin C) / Real.sin B = Real.cos C / Real.cos B →
  (∀ x y z, x + y + z = Real.pi → Real.sin x / a = Real.sin y / b) →
  (b = 2 * R * Real.sin B) →
  (b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) →
  (∃ (S : ℝ), S = 1/2 * a * c * Real.sin B) →
  (B = Real.pi / 3 ∧ 
   b = 3 ∧ 
   (∃ (S_max : ℝ), S_max = 9 * Real.sqrt 3 / 4 ∧ 
     (∀ S, S ≤ S_max) ∧ 
     (S = S_max ↔ a = c ∧ a = 3))) := by sorry

end triangle_properties_l3790_379028


namespace remainder_problem_l3790_379054

theorem remainder_problem (D : ℕ) (R : ℕ) (h1 : D > 0) 
  (h2 : 242 % D = 4) 
  (h3 : 698 % D = R) 
  (h4 : 940 % D = 7) : R = 3 := by
  sorry

end remainder_problem_l3790_379054


namespace kyles_presents_l3790_379085

theorem kyles_presents (cost1 cost2 cost3 : ℝ) : 
  cost2 = cost1 + 7 →
  cost3 = cost1 - 11 →
  cost1 + cost2 + cost3 = 50 →
  cost1 = 18 := by
sorry

end kyles_presents_l3790_379085


namespace lara_miles_walked_l3790_379073

/-- Represents a pedometer with a maximum count before resetting --/
structure Pedometer where
  max_count : ℕ
  reset_count : ℕ
  final_reading : ℕ
  steps_per_mile : ℕ

/-- Calculates the total steps walked based on pedometer data --/
def total_steps (p : Pedometer) : ℕ :=
  p.reset_count * (p.max_count + 1) + p.final_reading

/-- Calculates the approximate miles walked based on total steps --/
def miles_walked (p : Pedometer) : ℕ :=
  (total_steps p + p.steps_per_mile - 1) / p.steps_per_mile

/-- Theorem stating the approximate miles walked --/
theorem lara_miles_walked (p : Pedometer) 
  (h1 : p.max_count = 99999)
  (h2 : p.reset_count = 52)
  (h3 : p.final_reading = 38200)
  (h4 : p.steps_per_mile = 2000) :
  miles_walked p = 2619 := by
  sorry

end lara_miles_walked_l3790_379073


namespace pure_imaginary_condition_l3790_379076

/-- A complex number is pure imaginary if its real part is zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0

/-- The theorem states that if (a - i)² * i³ is a pure imaginary number,
    then the real number a must be equal to 0 -/
theorem pure_imaginary_condition (a : ℝ) :
  IsPureImaginary ((a - Complex.I)^2 * Complex.I^3) → a = 0 := by
  sorry

end pure_imaginary_condition_l3790_379076


namespace problem_proof_l3790_379079

theorem problem_proof : ((12^12 / 12^11)^2 * 4^2) / 2^4 = 144 := by
  sorry

end problem_proof_l3790_379079


namespace ball_bounce_theorem_l3790_379091

theorem ball_bounce_theorem (h : Real) (r : Real) (target : Real) :
  h = 700 ∧ r = 1/3 ∧ target = 2 →
  (∀ k : ℕ, h * r^k < target ↔ k ≥ 6) :=
by sorry

end ball_bounce_theorem_l3790_379091


namespace two_sqrt_two_gt_sqrt_seven_l3790_379015

theorem two_sqrt_two_gt_sqrt_seven : 2 * Real.sqrt 2 > Real.sqrt 7 := by
  sorry

end two_sqrt_two_gt_sqrt_seven_l3790_379015


namespace difference_is_perfect_square_l3790_379016

theorem difference_is_perfect_square (m n : ℕ+) 
  (h : 2001 * m^2 + m = 2002 * n^2 + n) : 
  ∃ k : ℕ, (m : ℤ) - (n : ℤ) = k^2 :=
sorry

end difference_is_perfect_square_l3790_379016


namespace largest_n_sin_cos_inequality_l3790_379066

theorem largest_n_sin_cos_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * Real.sqrt (n : ℝ))) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 1 / (2 * Real.sqrt (m : ℝ))) ∧
  n = 2 := by
  sorry

end largest_n_sin_cos_inequality_l3790_379066


namespace exponent_sum_l3790_379049

theorem exponent_sum (x m n : ℝ) (hm : x^m = 6) (hn : x^n = 2) : x^(m+n) = 12 := by
  sorry

end exponent_sum_l3790_379049


namespace equation_solution_l3790_379075

theorem equation_solution : ∃ y : ℚ, y + 5/8 = 2/9 + 1/2 ∧ y = 7/72 := by sorry

end equation_solution_l3790_379075


namespace inequality_proof_l3790_379033

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2) / 2 := by
  sorry

end inequality_proof_l3790_379033


namespace z_modulus_range_l3790_379059

-- Define the complex number z
def z (a : ℝ) : ℂ := Complex.mk (a - 2) (a + 1)

-- Define the condition for z to be in the second quadrant
def second_quadrant (a : ℝ) : Prop := a - 2 < 0 ∧ a + 1 > 0

-- State the theorem
theorem z_modulus_range :
  ∃ (min max : ℝ), min = 3 * Real.sqrt 2 / 2 ∧ max = 3 ∧
  ∀ a : ℝ, second_quadrant a →
    Complex.abs (z a) ≥ min ∧ Complex.abs (z a) ≤ max ∧
    (∃ a₁ a₂ : ℝ, second_quadrant a₁ ∧ second_quadrant a₂ ∧
      Complex.abs (z a₁) = min ∧ Complex.abs (z a₂) = max) :=
by sorry

end z_modulus_range_l3790_379059


namespace pyramid_volume_scaling_l3790_379043

theorem pyramid_volume_scaling (V₀ : ℝ) (l w h : ℝ) : 
  V₀ = (1/3) * l * w * h → 
  V₀ = 60 → 
  (1/3) * (3*l) * (4*w) * (2*h) = 1440 := by sorry

end pyramid_volume_scaling_l3790_379043


namespace erased_number_l3790_379026

/-- Given nine consecutive integers where the sum of eight of them is 1703, prove that the missing number is 214. -/
theorem erased_number (a : ℤ) (b : ℤ) (h1 : -4 ≤ b ∧ b ≤ 4) (h2 : 8*a - b = 1703) : a + b = 214 := by
  sorry

end erased_number_l3790_379026


namespace equation_condition_l3790_379057

theorem equation_condition (a b c : ℕ) 
  (ha : 0 < a ∧ a < 20) 
  (hb : 0 < b ∧ b < 20) 
  (hc : 0 < c ∧ c < 20) : 
  (20 * a + b) * (20 * a + c) = 400 * a^2 + 200 * a + b * c ↔ b + c = 10 := by
sorry

end equation_condition_l3790_379057


namespace negation_of_implication_l3790_379062

theorem negation_of_implication :
  (¬(∀ x : ℝ, x = 3 → x^2 - 2*x - 3 = 0)) ↔
  (∀ x : ℝ, x ≠ 3 → x^2 - 2*x - 3 ≠ 0) :=
by sorry

end negation_of_implication_l3790_379062


namespace wednesday_to_monday_ratio_l3790_379051

/-- Represents the number of cars passing through a toll booth on each day of the week -/
structure TollBoothWeek where
  total : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- The ratio of cars on Wednesday to cars on Monday is 2:1 -/
theorem wednesday_to_monday_ratio (week : TollBoothWeek)
  (h1 : week.total = 450)
  (h2 : week.monday = 50)
  (h3 : week.tuesday = 50)
  (h4 : week.wednesday = week.thursday)
  (h5 : week.friday = 50)
  (h6 : week.saturday = 50)
  (h7 : week.sunday = 50)
  (h8 : week.total = week.monday + week.tuesday + week.wednesday + week.thursday + 
                     week.friday + week.saturday + week.sunday) :
  week.wednesday = 2 * week.monday := by
  sorry

#check wednesday_to_monday_ratio

end wednesday_to_monday_ratio_l3790_379051


namespace recipe_fat_calculation_l3790_379050

/-- Calculates the grams of fat per serving in a recipe -/
def fat_per_serving (servings : ℕ) (cream_cups : ℚ) (fat_per_cup : ℕ) : ℚ :=
  (cream_cups * fat_per_cup) / servings

theorem recipe_fat_calculation :
  fat_per_serving 4 (1/2) 88 = 11 := by
  sorry

end recipe_fat_calculation_l3790_379050


namespace final_card_values_card_game_2004_l3790_379060

def card_game (n : ℕ) : ℕ :=
  3^(2*n) - 2 * 3^n + 2

theorem final_card_values (n : ℕ) :
  let initial_cards := 3^(2*n)
  let final_values := card_game n
  ∀ c : ℕ, c ≥ 3^n ∧ c ≤ 3^(2*n) - 3^n + 1 →
    c ∈ Finset.range final_values :=
by sorry

theorem card_game_2004 :
  card_game 1002 = 3^2004 - 2 * 3^1002 + 2 :=
by sorry

end final_card_values_card_game_2004_l3790_379060


namespace scooter_depreciation_l3790_379023

theorem scooter_depreciation (initial_value : ℝ) : 
  (((initial_value * (3/4)) * (3/4)) = 22500) → initial_value = 40000 := by
  sorry

end scooter_depreciation_l3790_379023


namespace purely_imaginary_m_l3790_379068

theorem purely_imaginary_m (m : ℝ) : 
  (m^2 - m : ℂ) + 3*I = (0 : ℝ) + I * (3 : ℝ) → m = 0 ∨ m = 1 := by
  sorry

end purely_imaginary_m_l3790_379068


namespace marks_reading_increase_l3790_379047

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Mark's current daily reading time in hours -/
def current_daily_reading : ℕ := 2

/-- Mark's desired weekly reading time in hours -/
def desired_weekly_reading : ℕ := 18

/-- Calculate the increase in Mark's weekly reading time -/
def reading_time_increase : ℕ :=
  desired_weekly_reading - (current_daily_reading * days_in_week)

/-- Theorem stating that Mark's weekly reading time increase is 4 hours -/
theorem marks_reading_increase : reading_time_increase = 4 := by
  sorry

end marks_reading_increase_l3790_379047


namespace max_second_term_arithmetic_sequence_l3790_379003

theorem max_second_term_arithmetic_sequence (a d : ℕ) (h1 : 0 < a) (h2 : 0 < d) :
  (a + (a + d) + (a + 2*d) + (a + 3*d) = 58) →
  ∀ b e : ℕ, (0 < b) → (0 < e) →
  (b + (b + e) + (b + 2*e) + (b + 3*e) = 58) →
  (a + d ≤ 10) :=
by sorry

end max_second_term_arithmetic_sequence_l3790_379003


namespace total_hours_worked_l3790_379008

def ordinary_rate : ℚ := 60 / 100
def overtime_rate : ℚ := 90 / 100
def total_earnings : ℚ := 3240 / 100
def overtime_hours : ℕ := 8

theorem total_hours_worked : ℕ := by
  sorry

#check total_hours_worked = 50

end total_hours_worked_l3790_379008


namespace c_can_be_any_real_l3790_379092

theorem c_can_be_any_real (a b c d : ℝ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : a / b + c / d < 0) :
  ∃ (a' b' d' : ℝ) (h1' : b' ≠ 0) (h2' : d' ≠ 0),
    (∀ c' : ℝ, ∃ (h3' : a' / b' + c' / d' < 0), True) :=
sorry

end c_can_be_any_real_l3790_379092
