import Mathlib

namespace NUMINAMATH_CALUDE_pages_written_theorem_l2480_248090

/-- Calculates the number of pages written in a year given the specified writing habits -/
def pages_written_per_year (pages_per_letter : ℕ) (num_friends : ℕ) (writing_frequency_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  pages_per_letter * num_friends * writing_frequency_per_week * weeks_per_year

/-- Proves that given the specified writing habits, the total number of pages written in a year is 624 -/
theorem pages_written_theorem :
  pages_written_per_year 3 2 2 52 = 624 := by
  sorry

end NUMINAMATH_CALUDE_pages_written_theorem_l2480_248090


namespace NUMINAMATH_CALUDE_compare_with_one_twentieth_l2480_248052

theorem compare_with_one_twentieth : 
  (1 / 15 : ℚ) > 1 / 20 ∧ 
  (1 / 25 : ℚ) < 1 / 20 ∧ 
  (1 / 2 : ℚ) > 1 / 20 ∧ 
  (55 / 1000 : ℚ) > 1 / 20 ∧ 
  (1 / 10 : ℚ) > 1 / 20 := by
  sorry

#check compare_with_one_twentieth

end NUMINAMATH_CALUDE_compare_with_one_twentieth_l2480_248052


namespace NUMINAMATH_CALUDE_simplify_sqrt_500_l2480_248010

theorem simplify_sqrt_500 : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_500_l2480_248010


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2480_248048

theorem simplify_sqrt_expression : 
  Real.sqrt 5 - Real.sqrt 40 + Real.sqrt 45 = 4 * Real.sqrt 5 - 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2480_248048


namespace NUMINAMATH_CALUDE_work_completion_time_l2480_248053

/-- Given that A can do a work in 9 days and A and B together can do the work in 6 days,
    prove that B can do the work alone in 18 days. -/
theorem work_completion_time (a_time b_time ab_time : ℝ) 
    (ha : a_time = 9)
    (hab : ab_time = 6)
    (h_work_rate : 1 / a_time + 1 / b_time = 1 / ab_time) : 
  b_time = 18 := by
sorry


end NUMINAMATH_CALUDE_work_completion_time_l2480_248053


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2480_248002

theorem polynomial_factorization (a b c : ℝ) :
  (a - 2*b) * (a - 2*b - 4) + 4 - c^2 = ((a - 2*b) - 2 + c) * ((a - 2*b) - 2 - c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2480_248002


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l2480_248035

theorem trig_expression_equals_one :
  let α : Real := 37 * π / 180
  let β : Real := 53 * π / 180
  (1 - 1 / Real.cos α) * (1 + 1 / Real.sin β) * (1 - 1 / Real.sin α) * (1 + 1 / Real.cos β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l2480_248035


namespace NUMINAMATH_CALUDE_train_speed_problem_l2480_248033

/-- Prove that given two trains on a 200 km track, where one starts at 7 am and the other at 8 am
    traveling towards each other, meeting at 12 pm, and the second train travels at 25 km/h,
    the speed of the first train is 20 km/h. -/
theorem train_speed_problem (total_distance : ℝ) (second_train_speed : ℝ) 
  (first_train_start_time : ℝ) (second_train_start_time : ℝ) (meeting_time : ℝ) :
  total_distance = 200 →
  second_train_speed = 25 →
  first_train_start_time = 7 →
  second_train_start_time = 8 →
  meeting_time = 12 →
  ∃ (first_train_speed : ℝ), first_train_speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2480_248033


namespace NUMINAMATH_CALUDE_melanie_initial_dimes_l2480_248096

/-- Proves that Melanie initially had 7 dimes given the problem conditions. -/
theorem melanie_initial_dimes :
  ∀ (initial : ℕ),
  initial + 8 + 4 = 19 →
  initial = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_melanie_initial_dimes_l2480_248096


namespace NUMINAMATH_CALUDE_initial_apples_count_l2480_248011

/-- The number of apples in a package -/
def apples_per_package : ℕ := 11

/-- The number of apples added to the pile -/
def apples_added : ℕ := 5

/-- The final number of apples in the pile -/
def final_apples : ℕ := 13

/-- The initial number of apples in the pile -/
def initial_apples : ℕ := final_apples - apples_added

theorem initial_apples_count : initial_apples = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_count_l2480_248011


namespace NUMINAMATH_CALUDE_not_parallel_implies_m_eq_one_perpendicular_implies_m_eq_neg_five_thirds_l2480_248095

/-- Two lines l₁ and l₂ in the plane -/
structure TwoLines (m : ℝ) where
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → Prop
  l₁_eq : ∀ x y, l₁ x y ↔ (3 + m) * x + 4 * y = 5 - 3 * m
  l₂_eq : ∀ x y, l₂ x y ↔ 2 * x + (m + 1) * y = -20

/-- Condition for two lines to be not parallel -/
def NotParallel (m : ℝ) (lines : TwoLines m) : Prop :=
  (3 + m) * (1 + m) - 4 * 2 ≠ 0

/-- Condition for two lines to be perpendicular -/
def Perpendicular (m : ℝ) (lines : TwoLines m) : Prop :=
  2 * (3 + m) + 4 * (1 + m) = 0

/-- Theorem: If the lines are not parallel, then m = 1 -/
theorem not_parallel_implies_m_eq_one (m : ℝ) (lines : TwoLines m) :
  NotParallel m lines → m = 1 := by sorry

/-- Theorem: If the lines are perpendicular, then m = -5/3 -/
theorem perpendicular_implies_m_eq_neg_five_thirds (m : ℝ) (lines : TwoLines m) :
  Perpendicular m lines → m = -5/3 := by sorry

end NUMINAMATH_CALUDE_not_parallel_implies_m_eq_one_perpendicular_implies_m_eq_neg_five_thirds_l2480_248095


namespace NUMINAMATH_CALUDE_marbles_selection_count_l2480_248012

def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def marbles_to_choose : ℕ := 5
def special_marbles_to_choose : ℕ := 2

theorem marbles_selection_count :
  (Nat.choose special_marbles special_marbles_to_choose) *
  (Nat.choose (total_marbles - special_marbles) (marbles_to_choose - special_marbles_to_choose)) =
  990 := by sorry

end NUMINAMATH_CALUDE_marbles_selection_count_l2480_248012


namespace NUMINAMATH_CALUDE_add_12345_seconds_to_10am_l2480_248001

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time -/
def initialTime : Time :=
  { hours := 10, minutes := 0, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 12345

/-- The expected final time -/
def expectedFinalTime : Time :=
  { hours := 13, minutes := 25, seconds := 45 }

theorem add_12345_seconds_to_10am :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_12345_seconds_to_10am_l2480_248001


namespace NUMINAMATH_CALUDE_solution_value_l2480_248074

/-- Given that (a, b) is a solution to the linear equation 2x-7y=8,
    prove that the value of the algebraic expression 17-4a+14b is 1 -/
theorem solution_value (a b : ℝ) (h : 2*a - 7*b = 8) : 17 - 4*a + 14*b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2480_248074


namespace NUMINAMATH_CALUDE_linear_function_m_value_l2480_248089

theorem linear_function_m_value :
  ∃! m : ℝ, m ≠ 0 ∧ (∀ x y : ℝ, y = m * x^(|m + 1|) - 2 → ∃ a b : ℝ, y = a * x + b) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_m_value_l2480_248089


namespace NUMINAMATH_CALUDE_olya_always_wins_l2480_248004

/-- Represents an archipelago with a given number of islands -/
structure Archipelago where
  num_islands : Nat
  connections : List (Nat × Nat)

/-- Represents a game played on an archipelago -/
inductive GameResult
  | OlyaWins
  | MaximWins

/-- The game played by Olya and Maxim on the archipelago -/
def play_game (a : Archipelago) : GameResult :=
  sorry

/-- Theorem stating that Olya always wins the game on an archipelago with 2009 islands -/
theorem olya_always_wins :
  ∀ (a : Archipelago), a.num_islands = 2009 → play_game a = GameResult.OlyaWins :=
sorry

end NUMINAMATH_CALUDE_olya_always_wins_l2480_248004


namespace NUMINAMATH_CALUDE_faster_train_length_l2480_248030

/-- The length of a train given its speed relative to another train and the time it takes to pass --/
def train_length (relative_speed : ℝ) (passing_time : ℝ) : ℝ :=
  relative_speed * passing_time

theorem faster_train_length :
  let faster_speed : ℝ := 108 * (1000 / 3600)  -- Convert km/h to m/s
  let slower_speed : ℝ := 36 * (1000 / 3600)   -- Convert km/h to m/s
  let relative_speed : ℝ := faster_speed - slower_speed
  let passing_time : ℝ := 17
  train_length relative_speed passing_time = 340 := by
  sorry

#check faster_train_length

end NUMINAMATH_CALUDE_faster_train_length_l2480_248030


namespace NUMINAMATH_CALUDE_count_tricycles_l2480_248029

/-- The number of tricycles in a bike shop, given the number of bicycles,
    the number of wheels per bicycle and tricycle, and the total number of wheels. -/
theorem count_tricycles (num_bicycles : ℕ) (wheels_per_bicycle : ℕ) (wheels_per_tricycle : ℕ) 
    (total_wheels : ℕ) (h1 : num_bicycles = 50) (h2 : wheels_per_bicycle = 2) 
    (h3 : wheels_per_tricycle = 3) (h4 : total_wheels = 160) : 
    (total_wheels - num_bicycles * wheels_per_bicycle) / wheels_per_tricycle = 20 := by
  sorry

end NUMINAMATH_CALUDE_count_tricycles_l2480_248029


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2480_248046

theorem consecutive_integers_sum (a b c : ℤ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c = 13) → a + b + c = 36 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2480_248046


namespace NUMINAMATH_CALUDE_nancy_and_rose_bracelets_l2480_248084

/-- The number of beads in each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has -/
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20

/-- The number of crystal beads Rose has -/
def rose_crystal_beads : ℕ := 20

/-- The number of stone beads Rose has -/
def rose_stone_beads : ℕ := 2 * rose_crystal_beads

/-- The total number of beads Nancy and Rose have -/
def total_beads : ℕ := nancy_metal_beads + nancy_pearl_beads + rose_crystal_beads + rose_stone_beads

/-- The number of bracelets Nancy and Rose can make -/
def bracelets_made : ℕ := total_beads / beads_per_bracelet

theorem nancy_and_rose_bracelets : bracelets_made = 20 := by
  sorry

end NUMINAMATH_CALUDE_nancy_and_rose_bracelets_l2480_248084


namespace NUMINAMATH_CALUDE_compound_has_one_hydrogen_l2480_248019

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  hydrogen : ℕ
  bromine : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (h_weight o_weight br_weight : ℚ) : ℚ :=
  c.hydrogen * h_weight + c.bromine * br_weight + c.oxygen * o_weight

/-- The theorem stating that a compound with 1 Br, 3 O, and molecular weight 129 has 1 H atom -/
theorem compound_has_one_hydrogen :
  ∃ (c : Compound),
    c.bromine = 1 ∧
    c.oxygen = 3 ∧
    molecularWeight c 1 16 79.9 = 129 ∧
    c.hydrogen = 1 := by
  sorry


end NUMINAMATH_CALUDE_compound_has_one_hydrogen_l2480_248019


namespace NUMINAMATH_CALUDE_highest_power_of_three_in_N_l2480_248050

def N : ℕ := sorry

-- Define the property that N is formed by writing down two-digit integers from 19 to 92 continuously
def is_valid_N (n : ℕ) : Prop := sorry

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem highest_power_of_three_in_N :
  is_valid_N N →
  ∃ m : ℕ, (sum_of_digits N = 3^2 * m) ∧ (m % 3 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_three_in_N_l2480_248050


namespace NUMINAMATH_CALUDE_reflection_y_transforms_points_l2480_248025

/-- Reflection in the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-(p.1), p.2)

theorem reflection_y_transforms_points :
  let C : ℝ × ℝ := (-3, 2)
  let D : ℝ × ℝ := (-4, -2)
  let C' : ℝ × ℝ := (3, 2)
  let D' : ℝ × ℝ := (4, -2)
  (reflect_y C = C') ∧ (reflect_y D = D') :=
by sorry

end NUMINAMATH_CALUDE_reflection_y_transforms_points_l2480_248025


namespace NUMINAMATH_CALUDE_points_collinearity_l2480_248041

/-- Checks if three points are collinear -/
def are_collinear (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

theorem points_collinearity :
  (are_collinear 1 2 2 4 3 6) ∧
  ¬(are_collinear 2 3 (-2) 1 3 4) := by
  sorry

end NUMINAMATH_CALUDE_points_collinearity_l2480_248041


namespace NUMINAMATH_CALUDE_sequence_a_odd_l2480_248068

def sequence_a : ℕ → ℤ
  | 0 => 2
  | 1 => 7
  | (n + 2) => sequence_a (n + 1)

axiom sequence_a_positive (n : ℕ) : 0 < sequence_a n

axiom sequence_a_inequality (n : ℕ) (h : n ≥ 2) :
  -1/2 < (sequence_a n - (sequence_a (n-1))^2 / sequence_a (n-2)) ∧
  (sequence_a n - (sequence_a (n-1))^2 / sequence_a (n-2)) ≤ 1/2

theorem sequence_a_odd (n : ℕ) (h : n > 1) : Odd (sequence_a n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_odd_l2480_248068


namespace NUMINAMATH_CALUDE_lime_score_difference_l2480_248049

/-- Given a ratio of white to black scores and a total number of lime scores,
    calculate 2/3 of the difference between the number of white and black scores. -/
theorem lime_score_difference (white_ratio black_ratio total_lime_scores : ℕ) : 
  white_ratio = 13 → 
  black_ratio = 8 → 
  total_lime_scores = 270 → 
  (2 : ℚ) / 3 * (white_ratio * (total_lime_scores / (white_ratio + black_ratio)) - 
                 black_ratio * (total_lime_scores / (white_ratio + black_ratio))) = 43 := by
  sorry

#eval (2 : ℚ) / 3 * (13 * (270 / (13 + 8)) - 8 * (270 / (13 + 8)))

end NUMINAMATH_CALUDE_lime_score_difference_l2480_248049


namespace NUMINAMATH_CALUDE_min_value_expression_l2480_248015

theorem min_value_expression (r s t : ℝ) 
  (h1 : 1 ≤ r) (h2 : r ≤ s) (h3 : s ≤ t) (h4 : t ≤ 4) :
  (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2 ≥ 12 - 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2480_248015


namespace NUMINAMATH_CALUDE_distribute_four_to_three_l2480_248058

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container having at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r objects from n distinct objects. -/
def choose (n r : ℕ) : ℕ := sorry

/-- The number of ways to arrange n distinct objects in k positions. -/
def arrange (n k : ℕ) : ℕ := sorry

theorem distribute_four_to_three :
  distribute 4 3 = 36 :=
by
  have h1 : distribute 4 3 = choose 4 2 * arrange 3 3 := sorry
  sorry


end NUMINAMATH_CALUDE_distribute_four_to_three_l2480_248058


namespace NUMINAMATH_CALUDE_function_no_zeros_implies_a_less_than_neg_one_l2480_248031

theorem function_no_zeros_implies_a_less_than_neg_one (a : ℝ) : 
  (∀ x : ℝ, 4^x - 2^(x+1) - a ≠ 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_function_no_zeros_implies_a_less_than_neg_one_l2480_248031


namespace NUMINAMATH_CALUDE_total_undeveloped_area_is_18750_l2480_248099

/-- The number of undeveloped land sections -/
def num_sections : ℕ := 5

/-- The area of each undeveloped land section in square feet -/
def area_per_section : ℕ := 3750

/-- The total area of undeveloped land in square feet -/
def total_undeveloped_area : ℕ := num_sections * area_per_section

/-- Theorem stating that the total area of undeveloped land is 18,750 square feet -/
theorem total_undeveloped_area_is_18750 : total_undeveloped_area = 18750 := by
  sorry

end NUMINAMATH_CALUDE_total_undeveloped_area_is_18750_l2480_248099


namespace NUMINAMATH_CALUDE_rectangle_area_l2480_248021

/-- Given a rectangle with perimeter 50 cm and length 13 cm, its area is 156 cm² -/
theorem rectangle_area (perimeter width length : ℝ) : 
  perimeter = 50 → 
  length = 13 → 
  width = (perimeter - 2 * length) / 2 → 
  length * width = 156 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2480_248021


namespace NUMINAMATH_CALUDE_steven_owes_jeremy_l2480_248026

/-- The amount Steven owes Jeremy for cleaning rooms --/
def amount_owed (base_rate : ℚ) (rooms_cleaned : ℚ) (bonus_threshold : ℚ) (bonus_rate : ℚ) : ℚ :=
  let base_payment := base_rate * rooms_cleaned
  let bonus_payment := if rooms_cleaned > bonus_threshold then rooms_cleaned * bonus_rate else 0
  base_payment + bonus_payment

/-- Theorem: Steven owes Jeremy 145/12 dollars --/
theorem steven_owes_jeremy :
  let base_rate : ℚ := 13/3
  let rooms_cleaned : ℚ := 5/2
  let bonus_threshold : ℚ := 2
  let bonus_rate : ℚ := 1/2
  amount_owed base_rate rooms_cleaned bonus_threshold bonus_rate = 145/12 := by
  sorry


end NUMINAMATH_CALUDE_steven_owes_jeremy_l2480_248026


namespace NUMINAMATH_CALUDE_derivative_of_y_l2480_248069

noncomputable def y (x : ℝ) : ℝ := Real.sin x - 2^x

theorem derivative_of_y (x : ℝ) :
  deriv y x = Real.cos x - 2^x * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l2480_248069


namespace NUMINAMATH_CALUDE_shopkeeper_oranges_l2480_248073

/-- The number of oranges bought by a shopkeeper -/
def oranges : ℕ := sorry

/-- The number of bananas bought by the shopkeeper -/
def bananas : ℕ := 400

/-- The percentage of oranges that are not rotten -/
def good_orange_percentage : ℚ := 85 / 100

/-- The percentage of bananas that are not rotten -/
def good_banana_percentage : ℚ := 95 / 100

/-- The overall percentage of fruits in good condition -/
def total_good_percentage : ℚ := 89 / 100

theorem shopkeeper_oranges :
  (good_orange_percentage * oranges + good_banana_percentage * bananas) / (oranges + bananas) = total_good_percentage ∧
  oranges = 600 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_oranges_l2480_248073


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2480_248014

theorem complex_fraction_equality : ∃ z : ℂ, z = (2 - I) / (1 - I) ∧ z = (3/2 : ℂ) + (1/2 : ℂ) * I :=
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2480_248014


namespace NUMINAMATH_CALUDE_next_simultaneous_ring_l2480_248016

def library_period : ℕ := 18
def fire_station_period : ℕ := 24
def hospital_period : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_simultaneous_ring (start_time : ℕ) :
  ∃ (t : ℕ), t > 0 ∧ 
    t % library_period = 0 ∧ 
    t % fire_station_period = 0 ∧ 
    t % hospital_period = 0 ∧
    t / minutes_in_hour = 6 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_ring_l2480_248016


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2480_248055

theorem hyperbola_asymptote_angle (c d : ℝ) (h1 : c > d) (h2 : c > 0) (h3 : d > 0) :
  (∀ x y : ℝ, x^2 / c^2 - y^2 / d^2 = 1) →
  (Real.arctan (d / c) - Real.arctan (-d / c) = π / 4) →
  c / d = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2480_248055


namespace NUMINAMATH_CALUDE_company_gender_distribution_l2480_248009

theorem company_gender_distribution (total : ℕ) 
  (h1 : total / 3 = total - 2 * total / 3)  -- One-third of workers don't have a retirement plan
  (h2 : (3 * total / 5) / 3 = total / 3 - 2 * total / 5 / 3)  -- 60% of workers without a retirement plan are women
  (h3 : (2 * total / 5) / 3 = total / 3 - 3 * total / 5 / 3)  -- 40% of workers without a retirement plan are men
  (h4 : 4 * (2 * total / 3) / 10 = 2 * total / 3 - 6 * (2 * total / 3) / 10)  -- 40% of workers with a retirement plan are men
  (h5 : 6 * (2 * total / 3) / 10 = 2 * total / 3 - 4 * (2 * total / 3) / 10)  -- 60% of workers with a retirement plan are women
  (h6 : (2 * total / 5) / 3 + 4 * (2 * total / 3) / 10 = 120)  -- There are 120 men in total
  : total - 120 = 180 := by
  sorry

end NUMINAMATH_CALUDE_company_gender_distribution_l2480_248009


namespace NUMINAMATH_CALUDE_digit_equation_solution_l2480_248066

/-- Represents a four-digit number ABBD --/
def ABBD (A B D : Nat) : Nat := A * 1000 + B * 100 + B * 10 + D

/-- Represents a four-digit number BCAC --/
def BCAC (B C A : Nat) : Nat := B * 1000 + C * 100 + A * 10 + C

/-- Represents a five-digit number DDBBD --/
def DDBBD (D B : Nat) : Nat := D * 10000 + D * 1000 + B * 100 + B * 10 + D

theorem digit_equation_solution 
  (A B C D : Nat) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) 
  (h_equation : ABBD A B D + BCAC B C A = DDBBD D B) : 
  D = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l2480_248066


namespace NUMINAMATH_CALUDE_rachel_video_game_score_l2480_248017

/-- Rachel's video game scoring problem -/
theorem rachel_video_game_score :
  let level1_treasures : ℕ := 5
  let level1_points : ℕ := 9
  let level2_treasures : ℕ := 2
  let level2_points : ℕ := 12
  let level3_treasures : ℕ := 8
  let level3_points : ℕ := 15
  let total_score := 
    level1_treasures * level1_points +
    level2_treasures * level2_points +
    level3_treasures * level3_points
  total_score = 189 := by sorry

end NUMINAMATH_CALUDE_rachel_video_game_score_l2480_248017


namespace NUMINAMATH_CALUDE_addison_raffle_tickets_l2480_248054

/-- The number of raffle tickets Addison sold on Friday -/
def friday_tickets : ℕ := 181

/-- The number of raffle tickets Addison sold on Saturday -/
def saturday_tickets : ℕ := 2 * friday_tickets

/-- The number of raffle tickets Addison sold on Sunday -/
def sunday_tickets : ℕ := 78

theorem addison_raffle_tickets :
  friday_tickets = 181 ∧
  saturday_tickets = 2 * friday_tickets ∧
  sunday_tickets = 78 ∧
  saturday_tickets = sunday_tickets + 284 :=
by sorry

end NUMINAMATH_CALUDE_addison_raffle_tickets_l2480_248054


namespace NUMINAMATH_CALUDE_tetrahedron_sum_l2480_248065

/-- A tetrahedron is a three-dimensional geometric shape with four faces, four vertices, and six edges. --/
structure Tetrahedron where
  edges : Nat
  corners : Nat
  faces : Nat

/-- The sum of edges, corners, and faces of a tetrahedron is 14. --/
theorem tetrahedron_sum (t : Tetrahedron) : t.edges + t.corners + t.faces = 14 := by
  sorry

#check tetrahedron_sum

end NUMINAMATH_CALUDE_tetrahedron_sum_l2480_248065


namespace NUMINAMATH_CALUDE_total_students_l2480_248042

theorem total_students (students_per_group : ℕ) (groups_per_class : ℕ) (classes : ℕ)
  (h1 : students_per_group = 7)
  (h2 : groups_per_class = 9)
  (h3 : classes = 13) :
  students_per_group * groups_per_class * classes = 819 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l2480_248042


namespace NUMINAMATH_CALUDE_sum_of_possible_A_values_l2480_248007

/-- The sum of digits of a number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by 9 -/
def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

/-- The given number with A as a parameter -/
def given_number (A : ℕ) : ℕ := 7456291 * 10 + A * 10 + 2

theorem sum_of_possible_A_values : 
  (∀ A : ℕ, A < 10 → is_divisible_by_9 (given_number A) → 
    sum_of_digits (given_number A) = sum_of_digits 7456291 + A + 2) →
  (∃ A₁ A₂ : ℕ, A₁ < 10 ∧ A₂ < 10 ∧ 
    is_divisible_by_9 (given_number A₁) ∧ 
    is_divisible_by_9 (given_number A₂) ∧
    A₁ + A₂ = 9) :=
sorry

end NUMINAMATH_CALUDE_sum_of_possible_A_values_l2480_248007


namespace NUMINAMATH_CALUDE_factorization_eq_l2480_248063

theorem factorization_eq (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_eq_l2480_248063


namespace NUMINAMATH_CALUDE_jar_water_problem_l2480_248044

theorem jar_water_problem (S L : ℝ) (hS : S > 0) (hL : L > 0) (h_capacities : S ≠ L) : 
  let water := (1/5) * S
  (water = (1/4) * L) → ((2 * water) / L = 1/2) := by sorry

end NUMINAMATH_CALUDE_jar_water_problem_l2480_248044


namespace NUMINAMATH_CALUDE_granddaughter_mother_age_ratio_l2480_248018

/-- The ratio of a granddaughter's age to her mother's age, given the ages of three generations. -/
theorem granddaughter_mother_age_ratio
  (betty_age : ℕ)
  (daughter_age : ℕ)
  (granddaughter_age : ℕ)
  (h1 : betty_age = 60)
  (h2 : daughter_age = betty_age - (40 * betty_age / 100))
  (h3 : granddaughter_age = 12) :
  granddaughter_age / daughter_age = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_granddaughter_mother_age_ratio_l2480_248018


namespace NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_equals_9_l2480_248064

theorem number_divided_by_6_multiplied_by_12_equals_9 (x : ℝ) : (x / 6) * 12 = 9 → x = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_equals_9_l2480_248064


namespace NUMINAMATH_CALUDE_division_and_addition_l2480_248039

theorem division_and_addition : (10 / (1/5)) + 6 = 56 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l2480_248039


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l2480_248086

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 - a 2 = 6 →
  a 5 - a 1 = 15 →
  a 3 = 4 ∨ a 3 = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l2480_248086


namespace NUMINAMATH_CALUDE_total_fruit_weight_l2480_248061

/-- The total weight of fruit sold by an orchard -/
theorem total_fruit_weight (frozen_fruit fresh_fruit : ℕ) 
  (h1 : frozen_fruit = 3513)
  (h2 : fresh_fruit = 6279) :
  frozen_fruit + fresh_fruit = 9792 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_weight_l2480_248061


namespace NUMINAMATH_CALUDE_min_fraction_value_l2480_248076

theorem min_fraction_value (x y : ℝ) (hx : 3 ≤ x ∧ x ≤ 5) (hy : -5 ≤ y ∧ y ≤ -3) :
  (x + y) / x ≥ 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_value_l2480_248076


namespace NUMINAMATH_CALUDE_red_tile_probability_l2480_248080

theorem red_tile_probability (n : ℕ) (h : n = 77) : 
  let red_tiles := (Finset.range n).filter (λ x => (x + 1) % 7 = 3)
  Finset.card red_tiles = 10 ∧ 
  (Finset.card red_tiles : ℚ) / n = 10 / 77 := by
  sorry

end NUMINAMATH_CALUDE_red_tile_probability_l2480_248080


namespace NUMINAMATH_CALUDE_zero_points_product_bound_l2480_248088

open Real

theorem zero_points_product_bound (a : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h_zero₁ : Real.log x₁ = a * x₁)
  (h_zero₂ : Real.log x₂ = a * x₂) :
  x₁ * x₂ > Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_zero_points_product_bound_l2480_248088


namespace NUMINAMATH_CALUDE_sin_cos_sum_27_63_l2480_248081

theorem sin_cos_sum_27_63 : 
  Real.sin (27 * π / 180) * Real.cos (63 * π / 180) + 
  Real.cos (27 * π / 180) * Real.sin (63 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_27_63_l2480_248081


namespace NUMINAMATH_CALUDE_inductive_reasoning_methods_l2480_248072

-- Define the type for reasoning methods
inductive ReasoningMethod
  | InferBallFromCircle
  | InferTriangleAngles
  | DeductBrokenChairs
  | InferPolygonAngles

-- Define a predicate for inductive reasoning
def isInductiveReasoning : ReasoningMethod → Prop
  | ReasoningMethod.InferBallFromCircle => False
  | ReasoningMethod.InferTriangleAngles => True
  | ReasoningMethod.DeductBrokenChairs => False
  | ReasoningMethod.InferPolygonAngles => True

-- Theorem stating which methods are inductive reasoning
theorem inductive_reasoning_methods :
  (isInductiveReasoning ReasoningMethod.InferTriangleAngles) ∧
  (isInductiveReasoning ReasoningMethod.InferPolygonAngles) ∧
  (¬ isInductiveReasoning ReasoningMethod.InferBallFromCircle) ∧
  (¬ isInductiveReasoning ReasoningMethod.DeductBrokenChairs) :=
by sorry


end NUMINAMATH_CALUDE_inductive_reasoning_methods_l2480_248072


namespace NUMINAMATH_CALUDE_remaining_money_is_48_6_l2480_248040

/-- Calculates the remaining money in Country B's currency after shopping in Country A -/
def remaining_money_country_b (initial_amount : ℝ) (grocery_ratio : ℝ) (household_ratio : ℝ) 
  (personal_ratio : ℝ) (household_tax : ℝ) (personal_discount : ℝ) (exchange_rate : ℝ) : ℝ :=
  let groceries := initial_amount * grocery_ratio
  let household := initial_amount * household_ratio * (1 + household_tax)
  let personal := initial_amount * personal_ratio * (1 - personal_discount)
  let total_spent := groceries + household + personal
  let remaining_a := initial_amount - total_spent
  remaining_a * exchange_rate

/-- Theorem stating that the remaining money in Country B's currency is 48.6 units -/
theorem remaining_money_is_48_6 : 
  remaining_money_country_b 450 (3/5) (1/6) (1/10) 0.05 0.1 0.8 = 48.6 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_is_48_6_l2480_248040


namespace NUMINAMATH_CALUDE_group1_larger_than_group2_l2480_248078

/-- A point on a circle -/
structure CirclePoint where
  angle : ℝ

/-- A convex polygon formed by points on a circle -/
structure ConvexPolygon where
  vertices : List CirclePoint
  is_convex : Bool

/-- The set of n points on the circle -/
def circle_points (n : ℕ) : List CirclePoint :=
  sorry

/-- Group 1: Polygons that include A₁ as a vertex -/
def group1 (n : ℕ) : List ConvexPolygon :=
  sorry

/-- Group 2: Polygons that do not include A₁ as a vertex -/
def group2 (n : ℕ) : List ConvexPolygon :=
  sorry

/-- Theorem: Group 1 contains more polygons than Group 2 -/
theorem group1_larger_than_group2 (n : ℕ) : 
  (group1 n).length > (group2 n).length :=
  sorry

end NUMINAMATH_CALUDE_group1_larger_than_group2_l2480_248078


namespace NUMINAMATH_CALUDE_at_least_one_irrational_l2480_248070

theorem at_least_one_irrational (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :
  ¬(∃ (q r : ℚ), (↑q : ℝ) = a ∧ (↑r : ℝ) = b) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_irrational_l2480_248070


namespace NUMINAMATH_CALUDE_product_inspection_probabilities_l2480_248083

/-- Given a total of 10 products with 8 first-grade and 2 second-grade products,
    calculate probabilities when 2 products are randomly inspected. -/
theorem product_inspection_probabilities :
  let total_products : ℕ := 10
  let first_grade_products : ℕ := 8
  let second_grade_products : ℕ := 2
  let inspected_products : ℕ := 2

  -- Probability that both products are first-grade
  (Nat.choose first_grade_products inspected_products : ℚ) / 
  (Nat.choose total_products inspected_products : ℚ) = 28/45 ∧
  
  -- Probability that at least one product is second-grade
  1 - (Nat.choose first_grade_products inspected_products : ℚ) / 
  (Nat.choose total_products inspected_products : ℚ) = 17/45 :=
by
  sorry


end NUMINAMATH_CALUDE_product_inspection_probabilities_l2480_248083


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_sixteen_l2480_248008

theorem sqrt_of_sqrt_sixteen : Real.sqrt (Real.sqrt 16) = 2 ∨ Real.sqrt (Real.sqrt 16) = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_sixteen_l2480_248008


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2480_248059

theorem trigonometric_identities :
  (Real.cos (780 * π / 180) = 1 / 2) ∧ 
  (Real.sin (-45 * π / 180) = -Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2480_248059


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2480_248067

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

theorem tangent_line_equation :
  (∃ L : ℝ → ℝ, (L 0 = 0) ∧ (∀ x : ℝ, L x = 2*x) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x| < δ → |f x - L x| < ε * |x|)) ∧
  (∀ x₀ : ℝ, x₀ ≠ 0 →
    (∃ L : ℝ → ℝ, (L x₀ = f x₀) ∧ (∀ x : ℝ, L x = f' x₀ * (x - x₀) + f x₀) ∧
      (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x₀| < δ → |f x - L x| < ε * |x - x₀|)) →
    f' x₀ = -1/4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2480_248067


namespace NUMINAMATH_CALUDE_equilateral_triangle_from_inscribed_circles_l2480_248003

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The angles of the triangle -/
  angles : Fin 3 → ℝ
  /-- Sum of angles is 180° -/
  sum_angles : (angles 0) + (angles 1) + (angles 2) = π
  /-- All angles are positive -/
  all_positive : ∀ i, 0 < angles i

/-- Represents the process of inscribing circles and forming new triangles -/
def inscribe_circle (t : TriangleWithInscribedCircle) : TriangleWithInscribedCircle :=
  sorry

/-- The theorem to be proved -/
theorem equilateral_triangle_from_inscribed_circles 
  (t : TriangleWithInscribedCircle) : 
  (∀ i, (inscribe_circle (inscribe_circle t)).angles i = t.angles i) → 
  (∀ i, t.angles i = π / 3) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_from_inscribed_circles_l2480_248003


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_one_l2480_248034

theorem arithmetic_expression_equals_one : 3 * (7 - 5) - 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_one_l2480_248034


namespace NUMINAMATH_CALUDE_income_redistribution_l2480_248098

/-- Represents the income distribution in a city --/
structure CityIncome where
  x : ℝ
  poor_income : ℝ
  middle_income : ℝ
  rich_income : ℝ
  tax_rate : ℝ

/-- Theorem stating the income redistribution after tax --/
theorem income_redistribution (c : CityIncome) 
  (h1 : c.poor_income = c.x)
  (h2 : c.middle_income = 3 * c.x)
  (h3 : c.rich_income = 6 * c.x)
  (h4 : c.poor_income + c.middle_income + c.rich_income = 100)
  (h5 : c.tax_rate = c.x^2 / 5 + c.x)
  (h6 : c.x = 10) :
  let tax_amount := c.rich_income * c.tax_rate / 100
  let poor_new := c.poor_income + 2 * tax_amount / 3
  let middle_new := c.middle_income + tax_amount / 3
  let rich_new := c.rich_income - tax_amount
  (poor_new = 22 ∧ middle_new = 36 ∧ rich_new = 42) := by
  sorry


end NUMINAMATH_CALUDE_income_redistribution_l2480_248098


namespace NUMINAMATH_CALUDE_range_of_a_l2480_248037

def complex_number (a : ℝ) : ℂ := (1 - a * Complex.I) * (a + 2 * Complex.I)

def in_first_quadrant (z : ℂ) : Prop := Complex.re z > 0 ∧ Complex.im z > 0

theorem range_of_a (a : ℝ) :
  in_first_quadrant (complex_number a) → 0 < a ∧ a < Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2480_248037


namespace NUMINAMATH_CALUDE_sin_double_angle_l2480_248071

theorem sin_double_angle (θ : Real) (h : Real.sin θ = 3/5) : Real.sin (2 * θ) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_l2480_248071


namespace NUMINAMATH_CALUDE_second_boy_speed_l2480_248062

/-- Given two boys walking in the same direction for 7 hours, with the first boy
    walking at 4 kmph and ending up 10.5 km apart, prove that the speed of the
    second boy is 5.5 kmph. -/
theorem second_boy_speed (v : ℝ) 
  (h1 : (v - 4) * 7 = 10.5) : v = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_second_boy_speed_l2480_248062


namespace NUMINAMATH_CALUDE_label_difference_less_than_distance_l2480_248032

open Set

theorem label_difference_less_than_distance :
  ∀ f : ℝ × ℝ → ℝ, ∃ P Q : ℝ × ℝ, P ≠ Q ∧ |f P - f Q| < ‖P - Q‖ :=
by sorry

end NUMINAMATH_CALUDE_label_difference_less_than_distance_l2480_248032


namespace NUMINAMATH_CALUDE_system_solution_l2480_248038

theorem system_solution : ∃! (x y z : ℝ),
  (3 * x - 2 * y + z = 7) ∧
  (9 * y - 6 * x - 3 * z = -21) ∧
  (x + y + z = 5) ∧
  (x = 1 ∧ y = 0 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2480_248038


namespace NUMINAMATH_CALUDE_coupon1_best_discount_l2480_248022

def coupon1_discount (x : ℝ) : ℝ := 0.1 * x

def coupon2_discount : ℝ := 20

def coupon3_discount (x : ℝ) : ℝ := 0.18 * (x - 100)

theorem coupon1_best_discount (x : ℝ) : 
  (coupon1_discount x > coupon2_discount ∧ 
   coupon1_discount x > coupon3_discount x) ↔ 
  (200 < x ∧ x < 225) :=
sorry

end NUMINAMATH_CALUDE_coupon1_best_discount_l2480_248022


namespace NUMINAMATH_CALUDE_player_a_wins_l2480_248043

/-- Represents a game state --/
structure GameState :=
  (current : ℕ)

/-- Defines a valid move in the game --/
def validMove (s : GameState) (next : ℕ) : Prop :=
  next > s.current ∧ next ≤ 2 * s.current - 1

/-- Defines the winning condition --/
def isWinningState (s : GameState) : Prop :=
  s.current = 2004

/-- Defines a winning strategy for Player A --/
def hasWinningStrategy (player : ℕ → GameState → Prop) : Prop :=
  ∀ s : GameState, s.current = 2 → 
    ∃ (strategy : GameState → ℕ),
      (∀ s, validMove s (strategy s)) ∧
      (∀ s, player 0 s → isWinningState (GameState.mk (strategy s)) ∨
        (∀ next, validMove (GameState.mk (strategy s)) next → 
          player 1 (GameState.mk next) → 
          player 0 (GameState.mk (strategy (GameState.mk next)))))

/-- The main theorem stating that Player A has a winning strategy --/
theorem player_a_wins : 
  ∃ (player : ℕ → GameState → Prop), hasWinningStrategy player :=
sorry

end NUMINAMATH_CALUDE_player_a_wins_l2480_248043


namespace NUMINAMATH_CALUDE_simplify_expression_l2480_248005

theorem simplify_expression (x y : ℝ) : 3*y - 5*x + 2*y + 4*x = 5*y - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2480_248005


namespace NUMINAMATH_CALUDE_cubic_minus_three_divisibility_l2480_248056

theorem cubic_minus_three_divisibility (n : ℕ) (h : n > 1) :
  (n - 1) ∣ (n^3 - 3) ↔ n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_three_divisibility_l2480_248056


namespace NUMINAMATH_CALUDE_brownies_calculation_l2480_248094

/-- The number of brownies in each batch -/
def brownies_per_batch : ℕ := 200

/-- The number of batches baked -/
def num_batches : ℕ := 10

/-- The fraction of brownies set aside for the bake sale -/
def bake_sale_fraction : ℚ := 3/4

/-- The fraction of remaining brownies put in a container -/
def container_fraction : ℚ := 3/5

/-- The number of brownies given out -/
def brownies_given_out : ℕ := 20

theorem brownies_calculation (b : ℕ) (h : b = brownies_per_batch) : 
  (1 - bake_sale_fraction) * (1 - container_fraction) * (b * num_batches) = brownies_given_out := by
  sorry

#check brownies_calculation

end NUMINAMATH_CALUDE_brownies_calculation_l2480_248094


namespace NUMINAMATH_CALUDE_right_triangle_from_equations_l2480_248077

theorem right_triangle_from_equations (a b c x : ℝ) :
  (∃ α : ℝ, α^2 + 2*a*α + b^2 = 0 ∧ α^2 + 2*c*α - b^2 = 0) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  a^2 = b^2 + c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_from_equations_l2480_248077


namespace NUMINAMATH_CALUDE_angle_relationship_indeterminate_l2480_248085

-- Define a plane
def Plane : Type := ℝ × ℝ → ℝ

-- Define a point in 3D space
def Point : Type := ℝ × ℝ × ℝ

-- Define a ray in 3D space
def Ray : Type := Point × Point

-- Function to calculate angle between two rays
def angle_between_rays : Ray → Ray → ℝ := sorry

-- Function to project a ray onto a plane
def project_ray : Ray → Plane → Ray := sorry

-- Function to check if a point is outside a plane
def is_outside_plane : Point → Plane → Prop := sorry

-- Theorem statement
theorem angle_relationship_indeterminate 
  (M : Plane) (P : Point) (r1 r2 : Ray) 
  (h_outside : is_outside_plane P M)
  (h_alpha : 0 < angle_between_rays r1 r2 ∧ angle_between_rays r1 r2 < π)
  (h_beta : 0 < angle_between_rays (project_ray r1 M) (project_ray r2 M) ∧ 
            angle_between_rays (project_ray r1 M) (project_ray r2 M) < π) :
  ¬ ∃ (R : ℝ → ℝ → Prop), 
    ∀ (α β : ℝ), 
      α = angle_between_rays r1 r2 → 
      β = angle_between_rays (project_ray r1 M) (project_ray r2 M) → 
      R α β :=
sorry

end NUMINAMATH_CALUDE_angle_relationship_indeterminate_l2480_248085


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2480_248075

-- Define the concept of a line in a plane
def Line : Type := Unit

-- Define the concept of a plane
def Plane : Type := Unit

-- Define the perpendicular relation between two lines in a plane
def perpendicular (p : Plane) (l1 l2 : Line) : Prop := sorry

-- Define the parallel relation between two lines in a plane
def parallel (p : Plane) (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_lines_parallel (p : Plane) (a b c : Line) :
  perpendicular p a c → perpendicular p b c → parallel p a b := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2480_248075


namespace NUMINAMATH_CALUDE_distinct_permutations_with_repetition_l2480_248000

theorem distinct_permutations_with_repetition : 
  let total_elements : ℕ := 5
  let repeated_elements : ℕ := 3
  let factorial (n : ℕ) := Nat.factorial n
  factorial total_elements / (factorial repeated_elements * factorial 1 * factorial 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_distinct_permutations_with_repetition_l2480_248000


namespace NUMINAMATH_CALUDE_evil_vile_live_l2480_248079

theorem evil_vile_live (E V I L : Nat) : 
  E ≠ 0 → V ≠ 0 → I ≠ 0 → L ≠ 0 →
  E < 10 → V < 10 → I < 10 → L < 10 →
  (1000 * E + 100 * V + 10 * I + L) % 73 = 0 →
  (1000 * V + 100 * I + 10 * L + E) % 74 = 0 →
  1000 * L + 100 * I + 10 * V + E = 5499 := by
sorry

end NUMINAMATH_CALUDE_evil_vile_live_l2480_248079


namespace NUMINAMATH_CALUDE_orange_price_is_60_l2480_248024

/- Define the problem parameters -/
def apple_price : ℚ := 40
def initial_total_fruits : ℕ := 10
def initial_avg_price : ℚ := 48
def oranges_removed : ℕ := 2
def final_avg_price : ℚ := 45

/- Define the function to calculate the orange price -/
def calculate_orange_price : ℚ :=
  let initial_total_cost : ℚ := initial_total_fruits * initial_avg_price
  let final_total_fruits : ℕ := initial_total_fruits - oranges_removed
  let final_total_cost : ℚ := final_total_fruits * final_avg_price
  60  -- The calculated price of each orange

/- Theorem statement -/
theorem orange_price_is_60 :
  calculate_orange_price = 60 :=
sorry

end NUMINAMATH_CALUDE_orange_price_is_60_l2480_248024


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l2480_248060

theorem infinitely_many_perfect_squares (n k : ℕ+) : 
  ∃ (S : Set (ℕ+ × ℕ+)), Set.Infinite S ∧ 
  ∀ (pair : ℕ+ × ℕ+), pair ∈ S → 
  ∃ (m : ℕ), (pair.1 * 2^(pair.2.val) - 7 : ℤ) = m^2 := by
sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l2480_248060


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2480_248047

def TwoStepSelection (n : ℕ) (m : ℕ) (k : ℕ) :=
  (n > m) ∧ (m > k) ∧ (k > 0)

theorem equal_selection_probability
  (n m k : ℕ)
  (h : TwoStepSelection n m k)
  (eliminate_one : ℕ → ℕ)
  (systematic_sample : ℕ → Finset ℕ)
  (h_eliminate : ∀ i, i ∈ Finset.range n → eliminate_one i ∈ Finset.range (n - 1))
  (h_sample : ∀ i, i ∈ Finset.range (n - 1) → systematic_sample i ⊆ Finset.range (n - 1) ∧ (systematic_sample i).card = k) :
  ∀ j ∈ Finset.range n, (∃ i ∈ Finset.range (n - 1), j ∈ systematic_sample (eliminate_one i)) ↔ true :=
sorry

#check equal_selection_probability

end NUMINAMATH_CALUDE_equal_selection_probability_l2480_248047


namespace NUMINAMATH_CALUDE_total_mission_time_is_11_days_l2480_248023

/-- Calculates the total time spent on missions given the planned duration of the first mission,
    the percentage increase in duration, and the duration of the second mission. -/
def total_mission_time (planned_duration : ℝ) (percentage_increase : ℝ) (second_mission_duration : ℝ) : ℝ :=
  (planned_duration * (1 + percentage_increase)) + second_mission_duration

/-- Proves that the total time spent on missions is 11 days. -/
theorem total_mission_time_is_11_days : 
  total_mission_time 5 0.6 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_mission_time_is_11_days_l2480_248023


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2480_248091

theorem greatest_divisor_with_remainders : 
  let a := 1657
  let b := 2037
  let r1 := 6
  let r2 := 5
  Int.gcd (a - r1) (b - r2) = 127 := by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2480_248091


namespace NUMINAMATH_CALUDE_mixture_problem_l2480_248082

/-- Represents the initial ratio of liquid A to liquid B -/
def initial_ratio : ℚ := 7 / 5

/-- Represents the amount of mixture drawn off in liters -/
def drawn_off : ℚ := 9

/-- Represents the new ratio of liquid A to liquid B after refilling -/
def new_ratio : ℚ := 7 / 9

/-- Represents the initial amount of liquid A in the can -/
def initial_amount_A : ℚ := 21

theorem mixture_problem :
  ∃ (total : ℚ),
    total > 0 ∧
    initial_amount_A / (total - initial_amount_A) = initial_ratio ∧
    (initial_amount_A - (initial_amount_A / total) * drawn_off) /
    (total - initial_amount_A - ((total - initial_amount_A) / total) * drawn_off + drawn_off) = new_ratio :=
by sorry

end NUMINAMATH_CALUDE_mixture_problem_l2480_248082


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l2480_248093

-- Define a monomial type
structure Monomial where
  coefficient : ℤ
  x_exponent : ℕ
  y_exponent : ℕ

-- Define our specific monomial
def our_monomial : Monomial := ⟨-3, 2, 1⟩

-- Theorem for the coefficient
theorem coefficient_of_monomial :
  our_monomial.coefficient = -3 := by sorry

-- Theorem for the degree
theorem degree_of_monomial :
  our_monomial.x_exponent + our_monomial.y_exponent = 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l2480_248093


namespace NUMINAMATH_CALUDE_science_club_neither_subject_l2480_248028

theorem science_club_neither_subject (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ)
  (h_total : total = 75)
  (h_biology : biology = 40)
  (h_chemistry : chemistry = 30)
  (h_both : both = 18) :
  total - (biology + chemistry - both) = 23 := by
  sorry

end NUMINAMATH_CALUDE_science_club_neither_subject_l2480_248028


namespace NUMINAMATH_CALUDE_part_one_part_two_l2480_248092

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.A = -4/5

-- Part I
theorem part_one (t : Triangle) (h : triangle_conditions t) (ha : t.a = 4) :
  Real.sin t.B = 3/10 := by sorry

-- Part II
theorem part_two (t : Triangle) (h : triangle_conditions t) (hs : (1/2) * t.b * t.c * Real.sin t.A = 6) :
  t.a = 2 * Real.sqrt 34 ∧ t.c = 10 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2480_248092


namespace NUMINAMATH_CALUDE_spencer_journey_distance_l2480_248006

def walking_distances : List Float := [1.2, 0.4, 0.6, 1.5]
def biking_distances : List Float := [1.8, 2]
def bus_distance : Float := 3

def biking_to_walking_factor : Float := 0.5
def bus_to_walking_factor : Float := 0.8

def total_walking_equivalent (walking : List Float) (biking : List Float) (bus : Float) 
  (bike_factor : Float) (bus_factor : Float) : Float :=
  (walking.sum) + 
  (biking.sum * bike_factor) + 
  (bus * bus_factor)

theorem spencer_journey_distance :
  total_walking_equivalent walking_distances biking_distances bus_distance
    biking_to_walking_factor bus_to_walking_factor = 8 := by
  sorry

end NUMINAMATH_CALUDE_spencer_journey_distance_l2480_248006


namespace NUMINAMATH_CALUDE_area_FGCD_l2480_248057

/-- Represents a trapezoid ABCD with the given properties -/
structure Trapezoid where
  ab : ℝ
  cd : ℝ
  altitude : ℝ
  ab_positive : 0 < ab
  cd_positive : 0 < cd
  altitude_positive : 0 < altitude

/-- Theorem stating the area of quadrilateral FGCD in the given trapezoid -/
theorem area_FGCD (t : Trapezoid) (h1 : t.ab = 10) (h2 : t.cd = 26) (h3 : t.altitude = 15) :
  let fg := (t.ab + t.cd) / 2 - 5 / 2
  (fg + t.cd) / 2 * t.altitude = 311.25 := by sorry

end NUMINAMATH_CALUDE_area_FGCD_l2480_248057


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2480_248087

theorem trigonometric_inequality (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π / 2) :
  π / 2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z >
  Real.sin (2 * x) + Real.sin (2 * y) + Real.sin (2 * z) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2480_248087


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l2480_248036

/-- The number of distinct arrangements of n beads on a bracelet, 
    considering rotational symmetry but not reflection -/
def bracelet_arrangements (n : ℕ) : ℕ := Nat.factorial n / n

/-- Theorem: The number of distinct arrangements of 8 beads on a bracelet, 
    considering rotational symmetry but not reflection, is 5040 -/
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements 8 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l2480_248036


namespace NUMINAMATH_CALUDE_dolphin_edge_probability_l2480_248045

/-- The probability of a point being within 2 m of the edge in a 30 m by 20 m rectangle is 23/75. -/
theorem dolphin_edge_probability : 
  let pool_length : ℝ := 30
  let pool_width : ℝ := 20
  let edge_distance : ℝ := 2
  let total_area := pool_length * pool_width
  let inner_length := pool_length - 2 * edge_distance
  let inner_width := pool_width - 2 * edge_distance
  let inner_area := inner_length * inner_width
  let edge_area := total_area - inner_area
  edge_area / total_area = 23 / 75 := by sorry

end NUMINAMATH_CALUDE_dolphin_edge_probability_l2480_248045


namespace NUMINAMATH_CALUDE_cos_six_arccos_two_fifths_l2480_248013

theorem cos_six_arccos_two_fifths :
  Real.cos (6 * Real.arccos (2/5)) = 12223/15625 := by
  sorry

end NUMINAMATH_CALUDE_cos_six_arccos_two_fifths_l2480_248013


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l2480_248097

theorem rectangular_box_volume 
  (a b c : ℝ) 
  (edge_sum : 4 * a + 4 * b + 4 * c = 180) 
  (diagonal : a^2 + b^2 + c^2 = 25^2) : 
  a * b * c = 32125 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l2480_248097


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l2480_248051

/-- Given a point P(4, -5) and its symmetric point P1 with respect to the y-axis, 
    prove that P1 has coordinates (-4, -5) -/
theorem symmetric_point_y_axis : 
  let P : ℝ × ℝ := (4, -5)
  let P1 : ℝ × ℝ := (-P.1, P.2)  -- Definition of symmetry with respect to y-axis
  P1 = (-4, -5) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l2480_248051


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2480_248027

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 300 →
  initial_money = 750 := by
sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2480_248027


namespace NUMINAMATH_CALUDE_third_stack_difference_l2480_248020

/-- Represents the heights of five stacks of blocks -/
structure BlockStacks where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- The properties of the block stacks as described in the problem -/
def validBlockStacks (s : BlockStacks) : Prop :=
  s.first = 7 ∧
  s.second = s.first + 3 ∧
  s.third < s.second ∧
  s.fourth = s.third + 10 ∧
  s.fifth = 2 * s.second ∧
  s.first + s.second + s.third + s.fourth + s.fifth = 55

theorem third_stack_difference (s : BlockStacks) 
  (h : validBlockStacks s) : s.second - s.third = 1 := by
  sorry

end NUMINAMATH_CALUDE_third_stack_difference_l2480_248020
