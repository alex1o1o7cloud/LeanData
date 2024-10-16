import Mathlib

namespace NUMINAMATH_CALUDE_raft_minimum_capacity_l2128_212846

/-- Represents an animal with its weight -/
structure Animal where
  weight : ℕ

/-- Represents the raft with its capacity -/
structure Raft where
  capacity : ℕ

/-- Checks if the raft can carry at least two of the lightest animals -/
def canCarryTwoLightest (r : Raft) (animals : List Animal) : Prop :=
  r.capacity ≥ 2 * (animals.map Animal.weight).minimum

/-- Checks if all animals can be transported using the given raft -/
def canTransportAll (r : Raft) (animals : List Animal) : Prop :=
  canCarryTwoLightest r animals

/-- The theorem to be proved -/
theorem raft_minimum_capacity 
  (mice : List Animal) 
  (moles : List Animal) 
  (hamsters : List Animal) 
  (h_mice : mice.length = 5 ∧ ∀ m ∈ mice, m.weight = 70)
  (h_moles : moles.length = 3 ∧ ∀ m ∈ moles, m.weight = 90)
  (h_hamsters : hamsters.length = 4 ∧ ∀ h ∈ hamsters, h.weight = 120)
  : ∃ (r : Raft), r.capacity = 140 ∧ canTransportAll r (mice ++ moles ++ hamsters) :=
sorry

end NUMINAMATH_CALUDE_raft_minimum_capacity_l2128_212846


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2128_212830

-- Define the universal set U
def U : Set Int := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set A
def A : Set Int := {0, 1, 2, 3}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = {-3, -2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2128_212830


namespace NUMINAMATH_CALUDE_toothpicks_200th_stage_l2128_212800

def toothpicks (n : ℕ) : ℕ :=
  if n ≤ 49 then
    4 + 4 * (n - 1)
  else if n ≤ 99 then
    toothpicks 49 + 5 * (n - 49)
  else if n ≤ 149 then
    toothpicks 99 + 6 * (n - 99)
  else
    toothpicks 149 + 7 * (n - 149)

theorem toothpicks_200th_stage :
  toothpicks 200 = 1082 := by sorry

end NUMINAMATH_CALUDE_toothpicks_200th_stage_l2128_212800


namespace NUMINAMATH_CALUDE_milk_leftover_problem_l2128_212816

/-- Calculates the amount of milk left over from yesterday given today's milk production and sales --/
def milk_leftover (morning_milk : ℕ) (evening_milk : ℕ) (sold_milk : ℕ) (total_left : ℕ) : ℕ :=
  total_left - ((morning_milk + evening_milk) - sold_milk)

/-- Theorem stating that given the problem conditions, the milk leftover from yesterday is 15 gallons --/
theorem milk_leftover_problem : milk_leftover 365 380 612 148 = 15 := by
  sorry

end NUMINAMATH_CALUDE_milk_leftover_problem_l2128_212816


namespace NUMINAMATH_CALUDE_desired_depth_is_50_l2128_212895

/-- Calculates the desired depth to be dug given initial and new working conditions -/
def desired_depth (initial_men : ℕ) (initial_hours : ℕ) (initial_depth : ℕ) 
                  (new_hours : ℕ) (extra_men : ℕ) : ℕ :=
  let total_men := initial_men + extra_men
  let numerator := total_men * new_hours * initial_depth
  let denominator := initial_men * initial_hours
  numerator / denominator

/-- Theorem stating that the desired depth is 50 meters under given conditions -/
theorem desired_depth_is_50 :
  desired_depth 54 8 30 6 66 = 50 := by
  sorry

end NUMINAMATH_CALUDE_desired_depth_is_50_l2128_212895


namespace NUMINAMATH_CALUDE_employee_salary_proof_l2128_212803

-- Define the total weekly salary
def total_salary : ℝ := 594

-- Define the ratio of m's salary to n's salary
def salary_ratio : ℝ := 1.2

-- Define n's salary
def n_salary : ℝ := 270

-- Theorem statement
theorem employee_salary_proof :
  n_salary * (1 + salary_ratio) = total_salary :=
by sorry

end NUMINAMATH_CALUDE_employee_salary_proof_l2128_212803


namespace NUMINAMATH_CALUDE_ninas_ants_l2128_212860

theorem ninas_ants (spider_count : ℕ) (spider_eyes : ℕ) (ant_eyes : ℕ) (total_eyes : ℕ) :
  spider_count = 3 →
  spider_eyes = 8 →
  ant_eyes = 2 →
  total_eyes = 124 →
  (total_eyes - spider_count * spider_eyes) / ant_eyes = 50 := by
  sorry

end NUMINAMATH_CALUDE_ninas_ants_l2128_212860


namespace NUMINAMATH_CALUDE_min_value_of_g_l2128_212894

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^3

-- State the theorem
theorem min_value_of_g :
  ∃ (min : ℝ), min = 16 * Real.sqrt 3 / 9 ∧
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → g x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_g_l2128_212894


namespace NUMINAMATH_CALUDE_paving_cost_l2128_212873

/-- The cost of paving a rectangular floor given its dimensions and the paving rate. -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 800) :
  length * width * rate = 16500 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l2128_212873


namespace NUMINAMATH_CALUDE_fathers_age_triple_weiweis_l2128_212828

/-- Proves that after 5 years, the father's age will be three times Weiwei's age -/
theorem fathers_age_triple_weiweis (weiwei_age : ℕ) (father_age : ℕ) 
  (h1 : weiwei_age = 8) (h2 : father_age = 34) : 
  ∃ (years : ℕ), years = 5 ∧ father_age + years = 3 * (weiwei_age + years) :=
by sorry

end NUMINAMATH_CALUDE_fathers_age_triple_weiweis_l2128_212828


namespace NUMINAMATH_CALUDE_david_total_cost_l2128_212869

/-- Calculates the total cost of a cell phone plan given usage and plan details -/
def calculateTotalCost (baseCost monthlyTexts monthlyHours monthlyData : ℕ)
                       (extraTextCost extraMinuteCost extraGBCost : ℚ)
                       (usedTexts usedHours usedData : ℕ) : ℚ :=
  let extraTexts := max (usedTexts - monthlyTexts) 0
  let extraMinutes := max (usedHours * 60 - monthlyHours * 60) 0
  let extraData := max (usedData - monthlyData) 0
  baseCost + extraTextCost * extraTexts + extraMinuteCost * extraMinutes + extraGBCost * extraData

/-- Theorem stating that David's total cost is $54.50 -/
theorem david_total_cost :
  calculateTotalCost 25 200 40 3 (3/100) (15/100) 10 250 42 4 = 54.5 := by
  sorry

end NUMINAMATH_CALUDE_david_total_cost_l2128_212869


namespace NUMINAMATH_CALUDE_P_when_a_is_3_range_of_a_for_Q_subset_P_l2128_212805

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x : ℝ | (x - a) * (x + 1) ≤ 0}
def Q : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

-- Theorem 1: When a = 3, P = {x | -1 ≤ x ≤ 3}
theorem P_when_a_is_3 : P 3 = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem 2: The range of positive a such that Q ⊆ P is [2, +∞)
theorem range_of_a_for_Q_subset_P : 
  {a : ℝ | a > 0 ∧ Q ⊆ P a} = {a : ℝ | a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_P_when_a_is_3_range_of_a_for_Q_subset_P_l2128_212805


namespace NUMINAMATH_CALUDE_fewer_twos_equals_hundred_l2128_212823

theorem fewer_twos_equals_hundred : (222 / 2) - (22 / 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_fewer_twos_equals_hundred_l2128_212823


namespace NUMINAMATH_CALUDE_factorization_equality_l2128_212834

theorem factorization_equality (x y : ℝ) : x^2 + x*y + x = x*(x + y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2128_212834


namespace NUMINAMATH_CALUDE_smallest_multiple_ending_in_three_l2128_212848

theorem smallest_multiple_ending_in_three : 
  ∀ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 5 = 0 → n ≥ 53 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_ending_in_three_l2128_212848


namespace NUMINAMATH_CALUDE_apples_used_for_pie_l2128_212868

theorem apples_used_for_pie (initial_apples : ℕ) (remaining_apples : ℕ) 
  (h1 : initial_apples = 19) 
  (h2 : remaining_apples = 4) : 
  initial_apples - remaining_apples = 15 := by
  sorry

end NUMINAMATH_CALUDE_apples_used_for_pie_l2128_212868


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l2128_212855

/-- The total amount of oil leaked from a broken pipe -/
def total_oil_leaked (before_fixing : ℕ) (while_fixing : ℕ) : ℕ :=
  before_fixing + while_fixing

/-- Theorem: Given the specific amounts of oil leaked before and during fixing,
    the total amount of oil leaked is 6206 gallons -/
theorem oil_leak_calculation :
  total_oil_leaked 2475 3731 = 6206 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l2128_212855


namespace NUMINAMATH_CALUDE_product_invariant_under_decrease_l2128_212822

theorem product_invariant_under_decrease :
  ∃ (a b c d e : ℝ),
    a * b * c * d * e ≠ 0 ∧
    (a - 1) * (b - 1) * (c - 1) * (d - 1) * (e - 1) = a * b * c * d * e :=
by sorry

end NUMINAMATH_CALUDE_product_invariant_under_decrease_l2128_212822


namespace NUMINAMATH_CALUDE_tv_show_cost_l2128_212857

/-- Calculates the total cost of a TV show season -/
def season_cost (total_episodes : ℕ) (first_half_cost : ℝ) (second_half_increase : ℝ) : ℝ :=
  let half_episodes := total_episodes / 2
  let first_half_total := first_half_cost * half_episodes
  let second_half_cost := first_half_cost * (1 + second_half_increase)
  let second_half_total := second_half_cost * half_episodes
  first_half_total + second_half_total

/-- Theorem stating the total cost of the TV show season -/
theorem tv_show_cost : 
  season_cost 22 1000 1.2 = 35200 := by
  sorry

#eval season_cost 22 1000 1.2

end NUMINAMATH_CALUDE_tv_show_cost_l2128_212857


namespace NUMINAMATH_CALUDE_bridge_length_l2128_212866

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 →
  train_speed_kmh = 42.3 →
  crossing_time = 40 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 320 := by
  sorry

#check bridge_length

end NUMINAMATH_CALUDE_bridge_length_l2128_212866


namespace NUMINAMATH_CALUDE_two_stamps_theorem_l2128_212838

/-- The cost of a single stamp in dollars -/
def single_stamp_cost : ℚ := 34/100

/-- The cost of three stamps in dollars -/
def three_stamps_cost : ℚ := 102/100

/-- The cost of two stamps in dollars -/
def two_stamps_cost : ℚ := 68/100

theorem two_stamps_theorem :
  (single_stamp_cost * 2 = two_stamps_cost) ∧
  (single_stamp_cost * 3 = three_stamps_cost) := by
  sorry

end NUMINAMATH_CALUDE_two_stamps_theorem_l2128_212838


namespace NUMINAMATH_CALUDE_two_absent_one_present_probability_l2128_212879

-- Define the probability of a student being absent
def p_absent : ℚ := 1 / 20

-- Define the probability of a student being present
def p_present : ℚ := 1 - p_absent

-- Define the number of students
def n_students : ℕ := 3

-- Define the number of absent students we're interested in
def n_absent : ℕ := 2

-- Theorem statement
theorem two_absent_one_present_probability :
  (n_students.choose n_absent : ℚ) * p_absent ^ n_absent * p_present ^ (n_students - n_absent) = 57 / 8000 := by
  sorry

end NUMINAMATH_CALUDE_two_absent_one_present_probability_l2128_212879


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_power_of_two_l2128_212802

theorem negation_of_existence_squared_greater_than_power_of_two :
  (¬ ∃ (n : ℕ+), n.val ^ 2 > 2 ^ n.val) ↔ (∀ (n : ℕ+), n.val ^ 2 ≤ 2 ^ n.val) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_power_of_two_l2128_212802


namespace NUMINAMATH_CALUDE_distance_run_by_p_l2128_212807

/-- A race between two runners p and q, where p is faster but q gets a head start -/
structure Race where
  /-- The speed of runner q in meters per minute -/
  v : ℝ
  /-- The time of the race in minutes -/
  t : ℝ
  /-- The head start distance given to runner q in meters -/
  d : ℝ
  /-- The speed of runner p is 30% faster than q -/
  h_speed : v > 0
  /-- The race time is positive -/
  h_time : t > 0
  /-- The head start distance is non-negative -/
  h_headstart : d ≥ 0
  /-- The race ends in a tie -/
  h_tie : d + v * t = (v + 0.3 * v) * t

/-- The theorem stating the distance run by p in the race -/
theorem distance_run_by_p (race : Race) : 
  (race.v + 0.3 * race.v) * race.t = 1.3 * race.v * race.t :=
by sorry

end NUMINAMATH_CALUDE_distance_run_by_p_l2128_212807


namespace NUMINAMATH_CALUDE_tim_total_amount_l2128_212839

-- Define the value of each coin type
def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10
def half_dollar_value : ℚ := 0.50

-- Define the number of each coin type Tim received
def nickels_from_shining : ℕ := 3
def dimes_from_shining : ℕ := 13
def dimes_from_tip_jar : ℕ := 7
def half_dollars_from_tip_jar : ℕ := 9

-- Calculate the total amount Tim received
def total_amount : ℚ :=
  nickels_from_shining * nickel_value +
  (dimes_from_shining + dimes_from_tip_jar) * dime_value +
  half_dollars_from_tip_jar * half_dollar_value

-- Theorem statement
theorem tim_total_amount : total_amount = 6.65 := by
  sorry

end NUMINAMATH_CALUDE_tim_total_amount_l2128_212839


namespace NUMINAMATH_CALUDE_system_solution_l2128_212835

theorem system_solution : ∃ (x y : ℝ), 2*x - 3*y = -7 ∧ 5*x + 4*y = -6 ∧ (x, y) = (-2, 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2128_212835


namespace NUMINAMATH_CALUDE_fraction_simplification_l2128_212858

theorem fraction_simplification :
  (3/7 + 5/8 + 2/9) / (5/12 + 1/4) = 643/336 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2128_212858


namespace NUMINAMATH_CALUDE_average_headcount_is_11600_l2128_212808

def fall_02_03_headcount : ℕ := 11700
def fall_03_04_headcount : ℕ := 11500
def fall_04_05_headcount : ℕ := 11600

def average_headcount : ℚ :=
  (fall_02_03_headcount + fall_03_04_headcount + fall_04_05_headcount) / 3

theorem average_headcount_is_11600 :
  average_headcount = 11600 := by sorry

end NUMINAMATH_CALUDE_average_headcount_is_11600_l2128_212808


namespace NUMINAMATH_CALUDE_gcf_75_100_l2128_212829

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_75_100_l2128_212829


namespace NUMINAMATH_CALUDE_harvey_number_l2128_212812

/-- Represents the skipping rule for a single student -/
def skip_rule (s : List Nat) : List Nat :=
  s.enum.filterMap fun (i, n) => if (i + 1) % 4 ≠ 2 then some n else none

/-- Applies the skipping rule n times to the initial list -/
def apply_skip_n_times (n : Nat) : List Nat → List Nat
  | s => match n with
    | 0 => s
    | m + 1 => apply_skip_n_times m (skip_rule s)

theorem harvey_number :
  let initial_list := List.range 1100
  (apply_skip_n_times 7 initial_list).head? = some 365 := by
  sorry


end NUMINAMATH_CALUDE_harvey_number_l2128_212812


namespace NUMINAMATH_CALUDE_class_mean_calculation_l2128_212893

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_mean : ℚ)
  (group2_students : ℕ) (group2_mean : ℚ) :
  total_students = 28 →
  group1_students = 24 →
  group2_students = 4 →
  group1_mean = 68 / 100 →
  group2_mean = 82 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 70 / 100 := by
sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l2128_212893


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_necessary_not_sufficient_l2128_212856

-- Define the propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3*a) < 0

def q (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

-- Theorem 1: When a = 1, the range of x satisfying both p and q is (2, 3)
theorem range_of_x_when_a_is_one :
  {x : ℝ | p x 1 ∧ q x} = Set.Ioo 2 3 := by sorry

-- Theorem 2: The range of a when p is necessary but not sufficient for q is [1, 2]
theorem range_of_a_necessary_not_sufficient :
  {a : ℝ | a > 0 ∧ (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x)} = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_necessary_not_sufficient_l2128_212856


namespace NUMINAMATH_CALUDE_intersection_point_on_lines_unique_intersection_point_l2128_212821

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (25/29, -57/29)

/-- First line equation: 6x - 5y = 15 -/
def line1 (x y : ℚ) : Prop := 6*x - 5*y = 15

/-- Second line equation: 8x + 3y = 1 -/
def line2 (x y : ℚ) : Prop := 8*x + 3*y = 1

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_lines : 
  line1 intersection_point.1 intersection_point.2 ∧ 
  line2 intersection_point.1 intersection_point.2 :=
sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point (x y : ℚ) :
  line1 x y ∧ line2 x y → (x, y) = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_on_lines_unique_intersection_point_l2128_212821


namespace NUMINAMATH_CALUDE_average_book_cost_l2128_212885

theorem average_book_cost (initial_amount : ℕ) (books_bought : ℕ) (amount_left : ℕ) : 
  initial_amount = 236 → 
  books_bought = 6 → 
  amount_left = 14 → 
  (initial_amount - amount_left) / books_bought = 37 :=
by sorry

end NUMINAMATH_CALUDE_average_book_cost_l2128_212885


namespace NUMINAMATH_CALUDE_ellipse_intersection_product_l2128_212887

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define a point on the ellipse
def point_on_ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  ellipse a b p.1 p.2

-- Define a diameter of the ellipse
def is_diameter (a b : ℝ) (c d : ℝ × ℝ) : Prop :=
  point_on_ellipse a b c ∧ point_on_ellipse a b d ∧ 
  c.1 = -d.1 ∧ c.2 = -d.2

-- Define a line parallel to CD passing through A
def parallel_line (a b : ℝ) (c d n m : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, n.1 = -a + t * (d.1 - c.1) ∧ 
            n.2 = t * (d.2 - c.2) ∧
            m.1 = -a + (a / (d.1 - c.1)) * (d.1 - c.1) ∧
            m.2 = (a / (d.1 - c.1)) * (d.2 - c.2)

-- Theorem statement
theorem ellipse_intersection_product (a b : ℝ) (c d n m : ℝ × ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hcd : is_diameter a b c d)
  (hnm : parallel_line a b c d n m)
  (hn : point_on_ellipse a b n) :
  let a := (-a, 0)
  let o := (0, 0)
  (dist a m) * (dist a n) = (dist c o) * (dist c d) := by sorry


end NUMINAMATH_CALUDE_ellipse_intersection_product_l2128_212887


namespace NUMINAMATH_CALUDE_simplest_form_expression_l2128_212853

theorem simplest_form_expression (x y a : ℝ) (h : x ≠ 2) : 
  (∀ k : ℝ, k ≠ 0 → (1 : ℝ) / (x - 2) ≠ k * (1 : ℝ) / (x - 2)) ∧ 
  (∃ k : ℝ, k ≠ 0 ∧ (x^2 * y) / (2 * x) = k * (x * y) / 2) ∧
  (∃ k : ℝ, k ≠ 0 ∧ (2 * a) / 8 = k * a / 4) :=
sorry

end NUMINAMATH_CALUDE_simplest_form_expression_l2128_212853


namespace NUMINAMATH_CALUDE_snail_reaches_top_in_ten_days_l2128_212889

/-- Represents the snail's climbing problem -/
structure SnailClimb where
  treeHeight : ℕ
  climbUp : ℕ
  slideDown : ℕ

/-- Calculates the number of days needed for the snail to reach the top of the tree -/
def daysToReachTop (s : SnailClimb) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the snail reaches the top in 10 days -/
theorem snail_reaches_top_in_ten_days :
  let s : SnailClimb := ⟨24, 6, 4⟩
  daysToReachTop s = 10 := by
  sorry

end NUMINAMATH_CALUDE_snail_reaches_top_in_ten_days_l2128_212889


namespace NUMINAMATH_CALUDE_cos_angle_relation_l2128_212865

theorem cos_angle_relation (α : Real) (h : Real.cos (75 * Real.pi / 180 + α) = 1/2) :
  Real.cos (105 * Real.pi / 180 - α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_relation_l2128_212865


namespace NUMINAMATH_CALUDE_result_calculation_l2128_212831

/-- Definition of x as the solution to x = 2 + (√3 / (2 + (√3 / (2 + ...)))) -/
noncomputable def x : ℝ := 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / 2))))))

/-- Theorem stating the result of the calculation -/
theorem result_calculation : 1 / ((x + 2) * (x - 3)) = (5 + Real.sqrt 3) / -22 := by
  sorry

end NUMINAMATH_CALUDE_result_calculation_l2128_212831


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_097_billion_l2128_212877

theorem scientific_notation_of_1_097_billion :
  ∃ (a : ℝ) (n : ℤ), 1.097e9 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.097 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_097_billion_l2128_212877


namespace NUMINAMATH_CALUDE_evaluate_expression_l2128_212810

theorem evaluate_expression (b : ℕ) (h : b = 4) : b^3 * b^4 * b^2 = 262144 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2128_212810


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2128_212876

theorem largest_constant_inequality (x y : ℝ) :
  ∃ (D : ℝ), D = 2 * Real.sqrt 3 ∧
  (∀ (x y : ℝ), 2 * x^2 + 2 * y^2 + 3 ≥ D * (x + y)) ∧
  (∀ (D' : ℝ), (∀ (x y : ℝ), 2 * x^2 + 2 * y^2 + 3 ≥ D' * (x + y)) → D' ≤ D) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2128_212876


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2128_212882

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (36 - 18*x - x^2 = 0) → 
  (∃ r s : ℝ, (36 - 18*r - r^2 = 0) ∧ (36 - 18*s - s^2 = 0) ∧ (r + s = -18)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2128_212882


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2128_212859

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|) ∧ 
  (∃ y : ℝ, y < 1 ∧ |y + 1| + |y - 1| = 2 * |y|) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2128_212859


namespace NUMINAMATH_CALUDE_fraction_power_product_l2128_212825

theorem fraction_power_product : (2/3 : ℚ)^2023 * (-3/2 : ℚ)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l2128_212825


namespace NUMINAMATH_CALUDE_apartment_occupancy_l2128_212842

/-- Calculates the number of people in an apartment building given specific conditions. -/
def people_in_building (total_floors : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  let full_floors := total_floors / 2
  let half_full_floors := total_floors - full_floors
  let full_apartments := full_floors * apartments_per_floor
  let half_full_apartments := half_full_floors * (apartments_per_floor / 2)
  let total_apartments := full_apartments + half_full_apartments
  total_apartments * people_per_apartment

/-- Theorem stating that under given conditions, the number of people in the building is 360. -/
theorem apartment_occupancy : 
  people_in_building 12 10 4 = 360 := by
  sorry


end NUMINAMATH_CALUDE_apartment_occupancy_l2128_212842


namespace NUMINAMATH_CALUDE_function_determination_l2128_212874

-- Define a first-degree function
def first_degree_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- State the theorem
theorem function_determination (f : ℝ → ℝ) 
  (h1 : first_degree_function f)
  (h2 : 2 * f 2 - 3 * f 1 = 5)
  (h3 : 2 * f 0 - f (-1) = 1) :
  ∀ x, f x = 3 * x - 2 := by
sorry

end NUMINAMATH_CALUDE_function_determination_l2128_212874


namespace NUMINAMATH_CALUDE_last_two_digits_of_root_sum_power_l2128_212832

theorem last_two_digits_of_root_sum_power : 
  ∃ n : ℤ, (n : ℝ) = (Real.sqrt 29 + Real.sqrt 21)^1984 ∧ n % 100 = 71 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_root_sum_power_l2128_212832


namespace NUMINAMATH_CALUDE_smaller_cuboid_width_l2128_212852

/-- Proves that the width of smaller cuboids is 6 meters given the dimensions of the original cuboid,
    the length and height of smaller cuboids, and the number of smaller cuboids. -/
theorem smaller_cuboid_width
  (original_length : ℝ)
  (original_width : ℝ)
  (original_height : ℝ)
  (small_length : ℝ)
  (small_height : ℝ)
  (num_small_cuboids : ℕ)
  (h1 : original_length = 18)
  (h2 : original_width = 15)
  (h3 : original_height = 2)
  (h4 : small_length = 5)
  (h5 : small_height = 3)
  (h6 : num_small_cuboids = 6) :
  ∃ (small_width : ℝ), small_width = 6 ∧
    original_length * original_width * original_height =
    num_small_cuboids * small_length * small_width * small_height :=
by sorry

end NUMINAMATH_CALUDE_smaller_cuboid_width_l2128_212852


namespace NUMINAMATH_CALUDE_glass_bowls_problem_l2128_212820

/-- The number of glass bowls initially bought -/
def initial_bowls : ℕ := 139

/-- The cost per bowl in Rupees -/
def cost_per_bowl : ℚ := 13

/-- The selling price per bowl in Rupees -/
def selling_price : ℚ := 17

/-- The number of bowls sold -/
def bowls_sold : ℕ := 108

/-- The percentage gain -/
def percentage_gain : ℚ := 23.88663967611336

theorem glass_bowls_problem :
  (percentage_gain / 100 * (initial_bowls * cost_per_bowl) = 
   bowls_sold * selling_price - bowls_sold * cost_per_bowl) ∧
  (initial_bowls ≥ bowls_sold) := by
  sorry

end NUMINAMATH_CALUDE_glass_bowls_problem_l2128_212820


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l2128_212851

universe u

def U : Set (Fin 5) := {0, 1, 2, 3, 4}
def A : Set (Fin 5) := {2, 3, 4}
def B : Set (Fin 5) := {0, 1, 4}

theorem complement_A_inter_B :
  (Aᶜ ∩ B : Set (Fin 5)) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l2128_212851


namespace NUMINAMATH_CALUDE_largest_side_of_crate_with_cylinder_l2128_212896

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  width : ℝ
  depth : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate -/
def cylinderFitsInCrate (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (2 * cylinder.radius ≤ crate.width ∧ 2 * cylinder.radius ≤ crate.depth) ∨
  (2 * cylinder.radius ≤ crate.width ∧ 2 * cylinder.radius ≤ crate.height) ∨
  (2 * cylinder.radius ≤ crate.depth ∧ 2 * cylinder.radius ≤ crate.height)

theorem largest_side_of_crate_with_cylinder 
  (crate : CrateDimensions) 
  (cylinder : Cylinder) 
  (h1 : crate.width = 7)
  (h2 : crate.depth = 8)
  (h3 : cylinder.radius = 7)
  (h4 : cylinderFitsInCrate crate cylinder) :
  max crate.width (max crate.depth crate.height) = 14 := by
  sorry

#check largest_side_of_crate_with_cylinder

end NUMINAMATH_CALUDE_largest_side_of_crate_with_cylinder_l2128_212896


namespace NUMINAMATH_CALUDE_bracket_removal_l2128_212863

theorem bracket_removal (a b c : ℝ) : a - (b - c) = a - b + c := by
  sorry

end NUMINAMATH_CALUDE_bracket_removal_l2128_212863


namespace NUMINAMATH_CALUDE_kolya_purchase_options_l2128_212817

def store_price (a : ℕ) : ℕ := 100 * a + 99

def kolya_total : ℕ := 20083

theorem kolya_purchase_options :
  ∀ n : ℕ, (∃ a : ℕ, n * store_price a = kolya_total) ↔ (n = 17 ∨ n = 117) :=
by sorry

end NUMINAMATH_CALUDE_kolya_purchase_options_l2128_212817


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l2128_212844

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l2128_212844


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l2128_212837

/-- Given a convex hexagon ABCDEF with the following properties:
  - Angles A, B, and C are congruent
  - Angles D and E are congruent
  - Angle A is 30° less than angle D
  - Angle F is equal to angle A
  Prove that the measure of angle D is 140° -/
theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  A = B ∧ B = C ∧                      -- Angles A, B, and C are congruent
  D = E ∧                              -- Angles D and E are congruent
  A = D - 30 ∧                         -- Angle A is 30° less than angle D
  F = A ∧                              -- Angle F is equal to angle A
  A + B + C + D + E + F = 720          -- Sum of angles in a hexagon
  → D = 140 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l2128_212837


namespace NUMINAMATH_CALUDE_gummy_bear_production_l2128_212897

/-- The number of gummy bears in each packet -/
def bears_per_packet : ℕ := 50

/-- The number of packets filled in 40 minutes -/
def packets_filled : ℕ := 240

/-- The time taken to fill the packets (in minutes) -/
def time_taken : ℕ := 40

/-- The number of gummy bears manufactured per minute -/
def bears_per_minute : ℕ := packets_filled * bears_per_packet / time_taken

theorem gummy_bear_production :
  bears_per_minute = 300 := by
  sorry

end NUMINAMATH_CALUDE_gummy_bear_production_l2128_212897


namespace NUMINAMATH_CALUDE_negative_of_negative_five_l2128_212815

theorem negative_of_negative_five : -(- 5) = 5 := by sorry

end NUMINAMATH_CALUDE_negative_of_negative_five_l2128_212815


namespace NUMINAMATH_CALUDE_age_ratio_sandy_molly_l2128_212888

/-- Given that Sandy is 70 years old and Molly is 20 years older than Sandy,
    prove that the ratio of their ages is 7:9. -/
theorem age_ratio_sandy_molly :
  let sandy_age : ℕ := 70
  let age_difference : ℕ := 20
  let molly_age : ℕ := sandy_age + age_difference
  (sandy_age : ℚ) / (molly_age : ℚ) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_age_ratio_sandy_molly_l2128_212888


namespace NUMINAMATH_CALUDE_small_rectangle_perimeter_l2128_212814

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the problem setup -/
structure ProblemSetup where
  large_rectangle : Rectangle
  num_vertical_cuts : ℕ
  num_horizontal_cuts : ℕ
  total_cut_length : ℝ

/-- Theorem stating the solution to the problem -/
theorem small_rectangle_perimeter
  (setup : ProblemSetup)
  (h1 : setup.large_rectangle.perimeter = 100)
  (h2 : setup.num_vertical_cuts = 6)
  (h3 : setup.num_horizontal_cuts = 9)
  (h4 : setup.total_cut_length = 405)
  (h5 : (setup.num_vertical_cuts + 1) * (setup.num_horizontal_cuts + 1) = 70) :
  let small_rectangle := Rectangle.mk
    (setup.large_rectangle.width / (setup.num_vertical_cuts + 1))
    (setup.large_rectangle.height / (setup.num_horizontal_cuts + 1))
  small_rectangle.perimeter = 13 := by
  sorry

end NUMINAMATH_CALUDE_small_rectangle_perimeter_l2128_212814


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l2128_212827

/-- Given that 20 cows eat 20 bags of husk in 20 days, prove that one cow will eat one bag of husk in 20 days. -/
theorem cow_husk_consumption (num_cows : ℕ) (num_bags : ℕ) (num_days : ℕ) 
  (h1 : num_cows = 20) 
  (h2 : num_bags = 20) 
  (h3 : num_days = 20) : 
  (num_days : ℚ) = (num_cows : ℚ) * (num_bags : ℚ) / ((num_cows : ℚ) * (num_bags : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l2128_212827


namespace NUMINAMATH_CALUDE_multiples_of_15_between_16_and_181_l2128_212892

theorem multiples_of_15_between_16_and_181 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ 16 < n ∧ n < 181) (Finset.range 181)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_16_and_181_l2128_212892


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l2128_212836

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l2128_212836


namespace NUMINAMATH_CALUDE_larger_integer_value_l2128_212899

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 3 / 2)
  (h_product : (a : ℕ) * b = 180) :
  (a : ℝ) = 3 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l2128_212899


namespace NUMINAMATH_CALUDE_product_simplification_l2128_212872

theorem product_simplification (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((2*x + 2*y + 2*z)⁻¹) * (x⁻¹ + y⁻¹ + z⁻¹) * ((x*y + y*z + x*z)⁻¹) * 
  (2*(x*y)⁻¹ + 2*(y*z)⁻¹ + 2*(x*z)⁻¹) = (x^2 * y^2 * z^2)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_product_simplification_l2128_212872


namespace NUMINAMATH_CALUDE_units_digit_of_47_to_47_l2128_212843

theorem units_digit_of_47_to_47 : (47^47 % 10 = 3) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_47_to_47_l2128_212843


namespace NUMINAMATH_CALUDE_concentric_circles_properties_l2128_212840

def inner_radius : ℝ := 25
def track_width : ℝ := 15

def outer_radius : ℝ := inner_radius + track_width

theorem concentric_circles_properties :
  let inner_circ := 2 * Real.pi * inner_radius
  let outer_circ := 2 * Real.pi * outer_radius
  let inner_area := Real.pi * inner_radius^2
  let outer_area := Real.pi * outer_radius^2
  (outer_circ - inner_circ = 30 * Real.pi) ∧
  (outer_area - inner_area = 975 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_properties_l2128_212840


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l2128_212864

theorem sandy_shopping_money (remaining_money : ℝ) (spent_percentage : ℝ) 
  (h1 : remaining_money = 224)
  (h2 : spent_percentage = 0.3)
  : (remaining_money / (1 - spent_percentage)) = 320 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l2128_212864


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2128_212883

theorem quadratic_inequality_solution_sets (p q : ℝ) :
  (∀ x : ℝ, x^2 + p*x + q < 0 ↔ -1/2 < x ∧ x < 1/3) →
  (∀ x : ℝ, q*x^2 + p*x + 1 > 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2128_212883


namespace NUMINAMATH_CALUDE_remainder_of_nested_division_l2128_212898

theorem remainder_of_nested_division (P D K Q R R'q R'r : ℕ) :
  D > 0 →
  K > 0 →
  K < D →
  P = Q * D + R →
  R = R'q * K + R'r →
  R'r < K →
  P % (D * K) = R'r :=
sorry

end NUMINAMATH_CALUDE_remainder_of_nested_division_l2128_212898


namespace NUMINAMATH_CALUDE_blender_sales_at_600_l2128_212878

/-- Represents the relationship between price and number of customers for blenders. -/
structure BlenderSales where
  price : ℝ
  customers : ℝ

/-- The inverse proportionality constant for blender sales. -/
def k : ℝ := 10 * 300

/-- Axiom: The number of customers is inversely proportional to the price of blenders. -/
axiom inverse_proportion (b : BlenderSales) : b.price * b.customers = k

/-- The theorem to be proved. -/
theorem blender_sales_at_600 :
  ∃ (b : BlenderSales), b.price = 600 ∧ b.customers = 5 :=
sorry

end NUMINAMATH_CALUDE_blender_sales_at_600_l2128_212878


namespace NUMINAMATH_CALUDE_determinant_of_quartic_roots_l2128_212809

theorem determinant_of_quartic_roots (r s t : ℝ) (a b c d : ℂ) : 
  (a^4 + r*a^2 + s*a + t = 0) →
  (b^4 + r*b^2 + s*b + t = 0) →
  (c^4 + r*c^2 + s*c + t = 0) →
  (d^4 + r*d^2 + s*d + t = 0) →
  let M : Matrix (Fin 4) (Fin 4) ℂ := !![1+a, 1, 1, 1;
                                       1, 1+b, 1, 1;
                                       1, 1, 1+c, 1;
                                       1, 1, 1, 1+d]
  Matrix.det M = r + s - t := by
sorry

end NUMINAMATH_CALUDE_determinant_of_quartic_roots_l2128_212809


namespace NUMINAMATH_CALUDE_triangle_bottom_number_l2128_212891

/-- Define the triangle structure -/
def Triangle (n : ℕ) : Type :=
  Fin n → Fin n → ℕ

/-- The first row of the triangle contains numbers from 1 to 2000 -/
def first_row_condition (t : Triangle 2000) : Prop :=
  ∀ i : Fin 2000, t 0 i = i.val + 1

/-- Each subsequent number is the sum of the two numbers immediately above it -/
def sum_condition (t : Triangle 2000) : Prop :=
  ∀ i j : Fin 2000, i > 0 → t i j = t (i-1) j + t (i-1) (j+1)

/-- The theorem to be proved -/
theorem triangle_bottom_number (t : Triangle 2000) 
  (h1 : first_row_condition t) (h2 : sum_condition t) : 
  t 1999 0 = 2^1998 * 2001 := by
  sorry

end NUMINAMATH_CALUDE_triangle_bottom_number_l2128_212891


namespace NUMINAMATH_CALUDE_point_movement_l2128_212845

/-- Given a point A with coordinates (-3, -2), moving it up by 3 units
    and then left by 2 units results in a point B with coordinates (-5, 1). -/
theorem point_movement :
  let A : ℝ × ℝ := (-3, -2)
  let up_movement : ℝ := 3
  let left_movement : ℝ := 2
  let B : ℝ × ℝ := (A.1 - left_movement, A.2 + up_movement)
  B = (-5, 1) := by sorry

end NUMINAMATH_CALUDE_point_movement_l2128_212845


namespace NUMINAMATH_CALUDE_a_formula_a_2_2_l2128_212880

/-- The number of ordered subset groups with empty intersection -/
def a (i j : ℕ+) : ℕ :=
  (2^j.val - 1)^i.val

/-- The theorem stating the formula for a(i,j) -/
theorem a_formula (i j : ℕ+) :
  a i j = (Finset.univ.filter (fun s : Finset (Fin i.val) => s.card > 0)).card ^ j.val :=
by sorry

/-- Specific case for a(2,2) -/
theorem a_2_2 : a 2 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_a_formula_a_2_2_l2128_212880


namespace NUMINAMATH_CALUDE_function_periodic_l2128_212841

/-- A function satisfying the given functional equation is periodic -/
theorem function_periodic (f : ℝ → ℝ) (a : ℝ) (ha : a > 0)
  (h : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodic_l2128_212841


namespace NUMINAMATH_CALUDE_player_B_more_consistent_l2128_212804

def player_A_scores : List ℕ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def player_B_scores : List ℕ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def variance (scores : List ℕ) : ℚ :=
  let m := mean scores
  (scores.map (λ x => ((x : ℚ) - m) ^ 2)).sum / scores.length

theorem player_B_more_consistent :
  variance player_B_scores < variance player_A_scores :=
by sorry

end NUMINAMATH_CALUDE_player_B_more_consistent_l2128_212804


namespace NUMINAMATH_CALUDE_min_distinct_values_l2128_212881

theorem min_distinct_values (n : ℕ) (mode_freq : ℕ) (second_freq : ℕ) 
  (h1 : n = 3000)
  (h2 : mode_freq = 15)
  (h3 : second_freq = 14)
  (h4 : ∀ k : ℕ, k ≠ mode_freq → k ≤ second_freq) :
  (∃ x : ℕ, x * mode_freq + x * second_freq + (n - x * mode_freq - x * second_freq) ≤ n ∧ 
   ∀ y : ℕ, y < x → y * mode_freq + y * second_freq + (n - y * mode_freq - y * second_freq) > n) →
  x = 232 := by
sorry

end NUMINAMATH_CALUDE_min_distinct_values_l2128_212881


namespace NUMINAMATH_CALUDE_a_minus_b_values_l2128_212819

theorem a_minus_b_values (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a + b > 0) :
  a - b = -1 ∨ a - b = -7 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_values_l2128_212819


namespace NUMINAMATH_CALUDE_daily_houses_count_l2128_212806

/-- Represents Kyle's newspaper delivery route --/
structure NewspaperRoute where
  /-- Number of houses receiving daily paper Monday through Saturday --/
  daily_houses : ℕ
  /-- Total number of papers delivered in a week --/
  total_weekly_papers : ℕ
  /-- Number of regular customers not receiving Sunday paper --/
  sunday_skip : ℕ
  /-- Number of houses receiving paper only on Sunday --/
  sunday_only : ℕ
  /-- Ensures the total weekly papers match the given conditions --/
  papers_match : total_weekly_papers = 
    (6 * daily_houses) + (daily_houses - sunday_skip + sunday_only)

/-- Theorem stating the number of houses receiving daily paper --/
theorem daily_houses_count (route : NewspaperRoute) 
  (h1 : route.total_weekly_papers = 720)
  (h2 : route.sunday_skip = 10)
  (h3 : route.sunday_only = 30) : 
  route.daily_houses = 100 := by
  sorry

#check daily_houses_count

end NUMINAMATH_CALUDE_daily_houses_count_l2128_212806


namespace NUMINAMATH_CALUDE_max_distinct_ten_blocks_l2128_212850

/-- Represents a binary string of length 10^4 -/
def BinaryString := Fin 10000 → Bool

/-- A k-block is a contiguous substring of length k -/
def kBlock (s : BinaryString) (start : Fin 10000) (k : Nat) : Fin k → Bool :=
  fun i => s ⟨start + i, sorry⟩

/-- Two k-blocks are identical if all their corresponding elements are equal -/
def kBlocksEqual (b1 b2 : Fin k → Bool) : Prop :=
  ∀ i : Fin k, b1 i = b2 i

/-- Count the number of distinct 3-blocks in a binary string -/
def distinctThreeBlocks (s : BinaryString) : Nat :=
  sorry

/-- Count the number of distinct 10-blocks in a binary string -/
def distinctTenBlocks (s : BinaryString) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem max_distinct_ten_blocks :
  ∀ s : BinaryString,
    distinctThreeBlocks s ≤ 7 →
    distinctTenBlocks s ≤ 504 :=
  sorry

end NUMINAMATH_CALUDE_max_distinct_ten_blocks_l2128_212850


namespace NUMINAMATH_CALUDE_woman_work_time_l2128_212854

/-- Represents the time taken to complete a work unit -/
structure WorkTime where
  men : ℕ
  women : ℕ
  days : ℚ

/-- The work rate of a single person -/
def work_rate (wt : WorkTime) : ℚ := 1 / wt.days

theorem woman_work_time (wt1 wt2 : WorkTime) : 
  wt1.men = 10 ∧ 
  wt1.women = 15 ∧ 
  wt1.days = 6 ∧
  wt2.men = 1 ∧ 
  wt2.women = 0 ∧ 
  wt2.days = 100 →
  ∃ wt3 : WorkTime, wt3.men = 0 ∧ wt3.women = 1 ∧ wt3.days = 225 :=
by sorry

#check woman_work_time

end NUMINAMATH_CALUDE_woman_work_time_l2128_212854


namespace NUMINAMATH_CALUDE_gcd_sequence_is_one_l2128_212811

theorem gcd_sequence_is_one (n : ℕ) : 
  Nat.gcd ((7^n - 1) / 6) ((7^(n+1) - 1) / 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sequence_is_one_l2128_212811


namespace NUMINAMATH_CALUDE_impossible_coverage_l2128_212818

/-- Represents a rectangle with width 1 and length either 6 or 7 -/
inductive SmallRectangle
  | sixLength : SmallRectangle
  | sevenLength : SmallRectangle

/-- Represents the grid to be covered -/
structure Grid :=
  (width : Nat)
  (height : Nat)

/-- Represents a configuration of small rectangles -/
structure Configuration :=
  (sixCount : Nat)
  (sevenCount : Nat)

def totalRectangles (config : Configuration) : Nat :=
  config.sixCount + config.sevenCount

def coveredArea (config : Configuration) : Nat :=
  6 * config.sixCount + 7 * config.sevenCount

def gridArea (grid : Grid) : Nat :=
  grid.width * grid.height

/-- Theorem stating that it's impossible to cover the 11x12 grid with the given configuration -/
theorem impossible_coverage (grid : Grid) (config : Configuration) :
  grid.width = 11 → grid.height = 12 → totalRectangles config = 19 →
  ¬(coveredArea config = gridArea grid ∧ 
    ∃ (arrangement : List SmallRectangle), 
      arrangement.length = totalRectangles config ∧ 
      -- Additional conditions for a valid arrangement would be defined here
      True) :=
sorry

end NUMINAMATH_CALUDE_impossible_coverage_l2128_212818


namespace NUMINAMATH_CALUDE_parentheses_equivalence_l2128_212890

theorem parentheses_equivalence (a b c : ℝ) : a - b + c = a - (b - c) := by
  sorry

end NUMINAMATH_CALUDE_parentheses_equivalence_l2128_212890


namespace NUMINAMATH_CALUDE_decagon_ratio_l2128_212862

/-- A decagon made up of unit squares with specific properties -/
structure Decagon where
  /-- The total number of unit squares in the decagon -/
  num_squares : ℕ
  /-- The total area of the decagon in square units -/
  total_area : ℝ
  /-- LZ is a line segment intersecting the left and right vertices of the decagon -/
  lz : ℝ × ℝ
  /-- XZ is a segment from LZ to a vertex -/
  xz : ℝ
  /-- ZY is a segment from LZ to another vertex -/
  zy : ℝ
  /-- The number of unit squares is 12 -/
  h_num_squares : num_squares = 12
  /-- The total area is 12 square units -/
  h_total_area : total_area = 12
  /-- LZ bisects the area of the decagon -/
  h_bisects : lz.1 = total_area / 2

/-- The ratio of XZ to ZY is 1 -/
theorem decagon_ratio (d : Decagon) : d.xz / d.zy = 1 := by
  sorry


end NUMINAMATH_CALUDE_decagon_ratio_l2128_212862


namespace NUMINAMATH_CALUDE_find_A_l2128_212801

theorem find_A (A : ℕ) (h : A % 9 = 6 ∧ A / 9 = 2) : A = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l2128_212801


namespace NUMINAMATH_CALUDE_f_at_4_l2128_212886

/-- The polynomial function f(x) = x^5 + 3x^4 - 5x^3 + 7x^2 - 9x + 11 -/
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

/-- Theorem: The value of f(4) is 1559 -/
theorem f_at_4 : f 4 = 1559 := by
  sorry

end NUMINAMATH_CALUDE_f_at_4_l2128_212886


namespace NUMINAMATH_CALUDE_parallel_lines_x_value_l2128_212884

/-- A line in a 2D plane --/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Check if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  (l1.point1.1 = l1.point2.1) = (l2.point1.1 = l2.point2.1)

theorem parallel_lines_x_value (l1 l2 : Line) (x : ℝ) :
  l1.point1 = (-1, -2) →
  l1.point2 = (-1, 4) →
  l2.point1 = (2, 1) →
  l2.point2 = (x, 6) →
  parallel l1 l2 →
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_x_value_l2128_212884


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l2128_212826

theorem greatest_multiple_of_four (x : ℕ) : 
  x % 4 = 0 → 
  x > 0 → 
  x^3 < 5000 → 
  x ≤ 16 ∧ 
  ∃ y : ℕ, y % 4 = 0 ∧ y > 0 ∧ y^3 < 5000 ∧ y = 16 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l2128_212826


namespace NUMINAMATH_CALUDE_flower_planting_area_l2128_212871

/-- Represents a square lawn with flowers -/
structure FlowerLawn where
  side_length : ℝ
  flower_area : ℝ

/-- Theorem: A square lawn with side length 16 meters can have a flower planting area of 144 square meters -/
theorem flower_planting_area (lawn : FlowerLawn) (h1 : lawn.side_length = 16) 
  (h2 : lawn.flower_area = 144) : 
  lawn.flower_area ≤ lawn.side_length ^ 2 ∧ lawn.flower_area > 0 := by
  sorry

#check flower_planting_area

end NUMINAMATH_CALUDE_flower_planting_area_l2128_212871


namespace NUMINAMATH_CALUDE_paving_cost_theorem_l2128_212833

/-- The cost of paving a rectangular floor -/
theorem paving_cost_theorem (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 300) :
  length * width * rate = 6187.5 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_theorem_l2128_212833


namespace NUMINAMATH_CALUDE_investment_growth_l2128_212875

theorem investment_growth (initial_investment : ℝ) (interest_rate : ℝ) (years : ℕ) (final_amount : ℝ) :
  initial_investment = 400 →
  interest_rate = 0.12 →
  years = 5 →
  final_amount = 705.03 →
  initial_investment * (1 + interest_rate) ^ years = final_amount :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l2128_212875


namespace NUMINAMATH_CALUDE_probability_y_div_x_geq_4_probability_equals_one_eighth_l2128_212824

/-- The probability that y/x ≥ 4 when x and y are randomly selected from [0,2] -/
theorem probability_y_div_x_geq_4 : Real :=
  let total_area := 4
  let favorable_area := 1/2
  favorable_area / total_area

/-- The probability is equal to 1/8 -/
theorem probability_equals_one_eighth : probability_y_div_x_geq_4 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_y_div_x_geq_4_probability_equals_one_eighth_l2128_212824


namespace NUMINAMATH_CALUDE_solve_system_for_y_l2128_212847

theorem solve_system_for_y (x y : ℚ) 
  (eq1 : 2 * x - y = 10) 
  (eq2 : x + 3 * y = 2) : 
  y = -6/7 := by sorry

end NUMINAMATH_CALUDE_solve_system_for_y_l2128_212847


namespace NUMINAMATH_CALUDE_correct_num_kettles_l2128_212849

/-- The number of kettles of hawks the ornithologists are tracking -/
def num_kettles : ℕ := 6

/-- The average number of pregnancies per kettle -/
def pregnancies_per_kettle : ℕ := 15

/-- The number of babies per pregnancy -/
def babies_per_pregnancy : ℕ := 4

/-- The survival rate of babies -/
def survival_rate : ℚ := 3/4

/-- The total number of expected babies this season -/
def total_babies : ℕ := 270

/-- Theorem stating that the number of kettles is correct given the conditions -/
theorem correct_num_kettles : 
  num_kettles = total_babies / (pregnancies_per_kettle * babies_per_pregnancy * survival_rate) :=
sorry

end NUMINAMATH_CALUDE_correct_num_kettles_l2128_212849


namespace NUMINAMATH_CALUDE_recipe_sugar_amount_l2128_212861

/-- The amount of sugar in cups already added to the recipe -/
def sugar_added : ℕ := 4

/-- The amount of sugar in cups still needed to be added to the recipe -/
def sugar_needed : ℕ := 3

/-- The total amount of sugar in cups required by the recipe -/
def total_sugar : ℕ := sugar_added + sugar_needed

theorem recipe_sugar_amount : total_sugar = 7 := by sorry

end NUMINAMATH_CALUDE_recipe_sugar_amount_l2128_212861


namespace NUMINAMATH_CALUDE_three_number_sum_l2128_212870

theorem three_number_sum (a b c : ℝ) (h1 : a < b) (h2 : b < c) 
  (h3 : ((a + b)/2 + (b + c)/2) / 2 = (a + b + c) / 3)
  (h4 : (a + c) / 2 = 2022) : 
  a + b + c = 6066 := by
  sorry

end NUMINAMATH_CALUDE_three_number_sum_l2128_212870


namespace NUMINAMATH_CALUDE_sequence_a_monotonicity_l2128_212867

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def sequence_a (x y : V) (n : ℕ) : ℝ := ‖x - n • y‖

theorem sequence_a_monotonicity (x y : V) (hx : x ≠ 0) (hy : y ≠ 0) :
  (∀ n : ℕ, sequence_a V x y n < sequence_a V x y (n + 1)) ↔
  (3 * ‖y‖ > 2 * ‖x‖ * ‖y‖⁻¹ * (inner x y)) ∧
  ¬(∀ n : ℕ, sequence_a V x y (n + 1) < sequence_a V x y n) :=
sorry

end NUMINAMATH_CALUDE_sequence_a_monotonicity_l2128_212867


namespace NUMINAMATH_CALUDE_billboard_perimeter_l2128_212813

/-- A rectangular billboard with given area and shorter side length has a specific perimeter. -/
theorem billboard_perimeter (area : ℝ) (short_side : ℝ) (h1 : area = 120) (h2 : short_side = 8) :
  2 * (area / short_side) + 2 * short_side = 46 := by
  sorry

#check billboard_perimeter

end NUMINAMATH_CALUDE_billboard_perimeter_l2128_212813
