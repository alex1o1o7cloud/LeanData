import Mathlib

namespace distance_calculation_l2340_234020

/-- Represents the distance to the destination in kilometers -/
def distance : ℝ := 96

/-- Represents the rowing speed in still water in km/h -/
def rowing_speed : ℝ := 10

/-- Represents the current velocity in km/h -/
def current_velocity : ℝ := 2

/-- Represents the total round trip time in hours -/
def total_time : ℝ := 20

/-- Theorem stating that the given conditions result in the correct distance -/
theorem distance_calculation :
  let speed_with_current := rowing_speed + current_velocity
  let speed_against_current := rowing_speed - current_velocity
  (distance / speed_with_current) + (distance / speed_against_current) = total_time :=
sorry

end distance_calculation_l2340_234020


namespace amber_max_ounces_l2340_234061

def amber_money : ℚ := 7
def candy_price : ℚ := 1
def candy_ounces : ℚ := 12
def chips_price : ℚ := 1.4
def chips_ounces : ℚ := 17

def candy_bags : ℚ := amber_money / candy_price
def chips_bags : ℚ := amber_money / chips_price

def total_candy_ounces : ℚ := candy_bags * candy_ounces
def total_chips_ounces : ℚ := chips_bags * chips_ounces

theorem amber_max_ounces :
  max total_candy_ounces total_chips_ounces = 85 :=
by sorry

end amber_max_ounces_l2340_234061


namespace smallest_n_squared_existence_of_solution_smallest_n_is_11_l2340_234075

theorem smallest_n_squared (n : ℕ+) : 
  (∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11) →
  n ≥ 11 :=
by sorry

theorem existence_of_solution : 
  ∃ (x y z : ℕ+), 11^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11 :=
by sorry

theorem smallest_n_is_11 : 
  (∃ (n : ℕ+), ∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11) ∧
  (∀ (m : ℕ+), (∃ (x y z : ℕ+), m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11) → m ≥ 11) :=
by sorry

end smallest_n_squared_existence_of_solution_smallest_n_is_11_l2340_234075


namespace terminating_decimal_expansion_of_13_375_l2340_234043

theorem terminating_decimal_expansion_of_13_375 :
  ∃ (n : ℕ) (k : ℕ), (13 : ℚ) / 375 = (34666 : ℚ) / 10^6 + k / (10^6 * 10^n) :=
sorry

end terminating_decimal_expansion_of_13_375_l2340_234043


namespace cafeteria_apples_l2340_234025

/-- The number of apples initially in the cafeteria -/
def initial_apples : ℕ := sorry

/-- The number of apples handed out to students -/
def apples_to_students : ℕ := 8

/-- The number of pies made -/
def pies_made : ℕ := 6

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 9

/-- Theorem stating that the initial number of apples is 62 -/
theorem cafeteria_apples : initial_apples = 62 := by
  sorry

end cafeteria_apples_l2340_234025


namespace shaded_cubes_is_14_l2340_234067

/-- Represents a 3x3x3 cube with a specific shading pattern -/
structure ShadedCube where
  /-- Total number of smaller cubes -/
  total_cubes : Nat
  /-- Number of cubes per edge -/
  edge_length : Nat
  /-- Number of shaded cubes per face -/
  shaded_per_face : Nat
  /-- Total number of faces -/
  total_faces : Nat
  /-- Condition: total cubes is 27 -/
  total_is_27 : total_cubes = 27
  /-- Condition: edge length is 3 -/
  edge_is_3 : edge_length = 3
  /-- Condition: 5 cubes shaded per face (4 corners + 1 center) -/
  shaded_is_5 : shaded_per_face = 5
  /-- Condition: cube has 6 faces -/
  faces_is_6 : total_faces = 6

/-- Function to calculate the number of uniquely shaded cubes -/
def uniquelyShadedCubes (c : ShadedCube) : Nat :=
  8 + 6 -- 8 corner cubes + 6 center face cubes

/-- Theorem: The number of uniquely shaded cubes is 14 -/
theorem shaded_cubes_is_14 (c : ShadedCube) : uniquelyShadedCubes c = 14 := by
  sorry

#check shaded_cubes_is_14

end shaded_cubes_is_14_l2340_234067


namespace kelvin_expected_score_l2340_234013

/-- Represents the coin flipping game --/
structure CoinGame where
  /-- The number of coins Kelvin starts with --/
  initialCoins : ℕ
  /-- The probability of getting heads on a single coin flip --/
  headsProbability : ℝ

/-- Calculates the expected score for the game --/
noncomputable def expectedScore (game : CoinGame) : ℝ :=
  sorry

/-- Theorem stating the expected score for Kelvin's specific game --/
theorem kelvin_expected_score :
  let game : CoinGame := { initialCoins := 2, headsProbability := 1/2 }
  expectedScore game = 64/9 := by
  sorry

end kelvin_expected_score_l2340_234013


namespace audrey_heracles_age_difference_l2340_234032

/-- The age difference between Audrey and Heracles -/
def ageDifference (audreyAge : ℕ) (heraclesAge : ℕ) : ℕ :=
  audreyAge - heraclesAge

theorem audrey_heracles_age_difference :
  ∃ (audreyAge : ℕ),
    heraclesAge = 10 ∧
    audreyAge + 3 = 2 * heraclesAge ∧
    ageDifference audreyAge heraclesAge = 7 :=
by
  sorry

end audrey_heracles_age_difference_l2340_234032


namespace candy_redistribution_theorem_l2340_234083

/-- Represents the number of candies each friend has at a given stage -/
structure CandyState where
  vasya : ℕ
  petya : ℕ
  kolya : ℕ

/-- Represents a round of candy redistribution -/
def redistribute (state : CandyState) (giver : Fin 3) : CandyState :=
  match giver with
  | 0 => ⟨state.vasya - (state.petya + state.kolya), 2 * state.petya, 2 * state.kolya⟩
  | 1 => ⟨2 * state.vasya, state.petya - (state.vasya + state.kolya), 2 * state.kolya⟩
  | 2 => ⟨2 * state.vasya, 2 * state.petya, state.kolya - (state.vasya + state.petya)⟩

theorem candy_redistribution_theorem (initial : CandyState) :
  initial.kolya = 36 →
  (redistribute (redistribute (redistribute initial 0) 1) 2).kolya = 36 →
  initial.vasya + initial.petya + initial.kolya = 252 := by
  sorry


end candy_redistribution_theorem_l2340_234083


namespace cricket_average_l2340_234046

theorem cricket_average (current_innings : Nat) (next_innings_runs : Nat) (average_increase : Nat) (current_average : Nat) : 
  current_innings = 20 →
  next_innings_runs = 116 →
  average_increase = 4 →
  (current_innings * current_average + next_innings_runs) / (current_innings + 1) = current_average + average_increase →
  current_average = 32 := by
sorry

end cricket_average_l2340_234046


namespace double_root_equations_l2340_234012

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₂ = 2 * x₁

/-- Theorem about double root equations -/
theorem double_root_equations :
  (is_double_root_equation 1 (-3) 2) ∧
  (∀ m n : ℝ, is_double_root_equation 1 (-2-m) (2*n) → 4*m^2 + 5*m*n + n^2 = 0) ∧
  (∀ p q : ℝ, p * q = 2 → is_double_root_equation p 3 q) := by
  sorry


end double_root_equations_l2340_234012


namespace isabella_marble_problem_l2340_234070

def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem isabella_marble_problem :
  ∀ k : ℕ, k < 45 → P k ≥ 1 / 2023 ∧ P 45 < 1 / 2023 := by
  sorry

end isabella_marble_problem_l2340_234070


namespace distance_AB_is_70_l2340_234078

/-- The distance between two points A and B, given specific travel conditions of two couriers --/
def distance_AB : ℝ := by sorry

theorem distance_AB_is_70 :
  let t₁ := 14 -- Travel time for first courier in hours
  let d := 10 -- Distance behind A where second courier starts in km
  let x := distance_AB -- Distance from A to B in km
  let v₁ := x / t₁ -- Speed of first courier
  let v₂ := (x + d) / t₁ -- Speed of second courier
  let t₁_20 := 20 / v₁ -- Time for first courier to travel 20 km
  let t₂_20 := 20 / v₂ -- Time for second courier to travel 20 km
  t₁_20 = t₂_20 + 0.5 → -- Second courier is half hour faster over 20 km
  x = 70 := by sorry

end distance_AB_is_70_l2340_234078


namespace friends_new_games_l2340_234040

theorem friends_new_games (katie_new : ℕ) (total_new : ℕ) (h1 : katie_new = 84) (h2 : total_new = 92) :
  total_new - katie_new = 8 := by
  sorry

end friends_new_games_l2340_234040


namespace grandfather_age_proof_l2340_234045

/-- The age of the grandfather -/
def grandfather_age : ℕ := 84

/-- The age of the older grandson -/
def older_grandson_age : ℕ := grandfather_age / 3

/-- The age of the younger grandson -/
def younger_grandson_age : ℕ := grandfather_age / 4

theorem grandfather_age_proof :
  (grandfather_age = 3 * older_grandson_age) ∧
  (grandfather_age = 4 * younger_grandson_age) ∧
  (older_grandson_age + younger_grandson_age = 49) →
  grandfather_age = 84 :=
by sorry

end grandfather_age_proof_l2340_234045


namespace max_product_sum_2024_l2340_234022

theorem max_product_sum_2024 : 
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144) ∧ 
  (∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) := by
  sorry

end max_product_sum_2024_l2340_234022


namespace not_right_triangle_condition_l2340_234096

/-- Triangle ABC with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of a right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

/-- The theorem to be proved -/
theorem not_right_triangle_condition (t : Triangle) 
  (h1 : t.a = 3^2)
  (h2 : t.b = 4^2)
  (h3 : t.c = 5^2) : 
  ¬ is_right_triangle t := by
  sorry

end not_right_triangle_condition_l2340_234096


namespace smallest_number_l2340_234069

theorem smallest_number (π : Real) : 
  -π < -3 ∧ -π < -Real.sqrt 2 ∧ -π < -(5/2) :=
by sorry

end smallest_number_l2340_234069


namespace sum_of_zeros_negative_l2340_234054

/-- The function f(x) = ln(x) - x + m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - x + m

/-- The function g(x) = f(x+m) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m (x + m)

/-- Theorem: Given f(x) = ln(x) - x + m, m > 1, g(x) = f(x+m), 
    and x₁, x₂ are zeros of g(x), then x₁ + x₂ < 0 -/
theorem sum_of_zeros_negative (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m > 1) 
  (hx₁ : g m x₁ = 0) 
  (hx₂ : g m x₂ = 0) : 
  x₁ + x₂ < 0 := by
  sorry

end sum_of_zeros_negative_l2340_234054


namespace joe_journey_time_l2340_234059

/-- Represents the scenario of Joe's journey from home to school -/
structure JourneyScenario where
  d : ℝ  -- Total distance from home to school
  walk_speed : ℝ  -- Joe's walking speed
  run_speed : ℝ  -- Joe's running speed
  walk_time : ℝ  -- Time Joe spent walking
  walk_distance : ℝ  -- Distance Joe walked
  run_distance : ℝ  -- Distance Joe ran

/-- The theorem stating the total time of Joe's journey -/
theorem joe_journey_time (scenario : JourneyScenario) :
  scenario.walk_distance = scenario.d / 3 ∧
  scenario.run_distance = 2 * scenario.d / 3 ∧
  scenario.run_speed = 4 * scenario.walk_speed ∧
  scenario.walk_time = 9 ∧
  scenario.walk_distance = scenario.walk_speed * scenario.walk_time ∧
  scenario.run_distance = scenario.run_speed * (13.5 - scenario.walk_time) →
  13.5 = scenario.walk_time + (scenario.run_distance / scenario.run_speed) :=
by sorry


end joe_journey_time_l2340_234059


namespace cubic_root_theorem_l2340_234055

theorem cubic_root_theorem (b c : ℚ) : 
  (∃ (x : ℝ), x^3 + b*x + c = 0 ∧ x = 3 - Real.sqrt 7) →
  ((-6 : ℝ)^3 + b*(-6) + c = 0) :=
sorry

end cubic_root_theorem_l2340_234055


namespace greatest_number_of_bouquets_l2340_234085

theorem greatest_number_of_bouquets (white_tulips red_tulips : ℕ) 
  (h1 : white_tulips = 21) (h2 : red_tulips = 91) : 
  (Nat.gcd white_tulips red_tulips) = 7 := by
  sorry

end greatest_number_of_bouquets_l2340_234085


namespace triangle_ab_length_l2340_234041

/-- Given a triangle ABC with angles B and C both 45 degrees and side BC of length 10,
    prove that the length of side AB is 5√2. -/
theorem triangle_ab_length (A B C : ℝ × ℝ) : 
  let triangle := (A, B, C)
  let angle (X Y Z : ℝ × ℝ) := Real.arccos ((X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2)) / 
    (((X.1 - Y.1)^2 + (X.2 - Y.2)^2).sqrt * ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2).sqrt)
  let distance (X Y : ℝ × ℝ) := ((X.1 - Y.1)^2 + (X.2 - Y.2)^2).sqrt
  angle B A C = π/4 →
  angle C B A = π/4 →
  distance B C = 10 →
  distance A B = 5 * Real.sqrt 2 := by
sorry


end triangle_ab_length_l2340_234041


namespace cauchy_schwarz_like_inequality_l2340_234015

theorem cauchy_schwarz_like_inequality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end cauchy_schwarz_like_inequality_l2340_234015


namespace cos_alpha_value_l2340_234089

theorem cos_alpha_value (α : Real) (h : Real.cos (Real.pi + α) = -1/3) : Real.cos α = 1/3 := by
  sorry

end cos_alpha_value_l2340_234089


namespace rose_friends_count_l2340_234056

def total_apples : ℕ := 9
def apples_per_friend : ℕ := 3

theorem rose_friends_count : 
  total_apples / apples_per_friend = 3 :=
by sorry

end rose_friends_count_l2340_234056


namespace geometric_sequence_common_ratio_l2340_234087

/-- Given a geometric sequence {a_n} with common ratio q, prove that if the sum of the first n terms S_n
    satisfies S_2 = 2a_2 + 3 and S_3 = 2a_3 + 3, then q = 2. -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (q : ℝ)      -- The common ratio
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, a (n + 1) = a n * q)  -- Definition of geometric sequence
  (h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q))  -- Sum formula for geometric sequence
  (h3 : S 2 = 2 * a 2 + 3)  -- Given condition for S_2
  (h4 : S 3 = 2 * a 3 + 3)  -- Given condition for S_3
  : q = 2 := by
  sorry

end geometric_sequence_common_ratio_l2340_234087


namespace student_test_score_l2340_234024

theorem student_test_score (max_marks : ℕ) (pass_percentage : ℚ) (fail_margin : ℕ) : 
  max_marks = 300 → 
  pass_percentage = 60 / 100 → 
  fail_margin = 100 → 
  ∃ (student_score : ℕ), 
    student_score = max_marks * pass_percentage - fail_margin ∧ 
    student_score = 80 := by
  sorry

end student_test_score_l2340_234024


namespace count_ways_2024_l2340_234033

/-- The target sum -/
def target_sum : ℕ := 2024

/-- The set of allowed numbers -/
def allowed_numbers : Finset ℕ := {2, 3, 4}

/-- A function that counts the number of ways to express a target sum
    as a sum of non-negative integer multiples of allowed numbers,
    ignoring the order of summands -/
noncomputable def count_ways (target : ℕ) (allowed : Finset ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 57231 ways to express 2024
    as a sum of non-negative integer multiples of 2, 3, and 4,
    ignoring the order of summands -/
theorem count_ways_2024 :
  count_ways target_sum allowed_numbers = 57231 :=
by sorry

end count_ways_2024_l2340_234033


namespace pencil_cost_l2340_234002

theorem pencil_cost (total_spent notebook_cost ruler_cost num_pencils : ℕ) 
  (h1 : total_spent = 74)
  (h2 : notebook_cost = 35)
  (h3 : ruler_cost = 18)
  (h4 : num_pencils = 3) :
  (total_spent - notebook_cost - ruler_cost) / num_pencils = 7 := by
  sorry

end pencil_cost_l2340_234002


namespace workshop_workers_l2340_234005

/-- The total number of workers in a workshop -/
def total_workers : ℕ := 49

/-- The average salary of all workers -/
def avg_salary_all : ℕ := 8000

/-- The number of technicians -/
def num_technicians : ℕ := 7

/-- The average salary of technicians -/
def avg_salary_technicians : ℕ := 20000

/-- The average salary of non-technicians -/
def avg_salary_others : ℕ := 6000

/-- Theorem stating that the total number of workers is 49 -/
theorem workshop_workers :
  total_workers = 49 ∧
  avg_salary_all * total_workers = 
    avg_salary_technicians * num_technicians + 
    avg_salary_others * (total_workers - num_technicians) :=
by sorry

end workshop_workers_l2340_234005


namespace sum_of_fractions_l2340_234003

def fraction_sequence : List ℚ := 
  [1/10, 2/10, 3/10, 4/10, 5/10, 6/10, 7/10, 8/10, 9/10, 10/10, 11/10, 12/10, 13/10]

theorem sum_of_fractions : 
  fraction_sequence.sum = 91/10 := by sorry

end sum_of_fractions_l2340_234003


namespace sum_of_ratios_l2340_234010

theorem sum_of_ratios (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x/y + y/z + z/x + y/x + z/y + x/z = 9)
  (h2 : x + y + z = 3) :
  x/y + y/z + z/x = 4.5 ∧ y/x + z/y + x/z = 4.5 := by
  sorry

end sum_of_ratios_l2340_234010


namespace gcd_lcm_sum_75_4500_l2340_234042

theorem gcd_lcm_sum_75_4500 : Nat.gcd 75 4500 + Nat.lcm 75 4500 = 4575 := by
  sorry

end gcd_lcm_sum_75_4500_l2340_234042


namespace max_value_of_function_l2340_234031

theorem max_value_of_function (x : ℝ) (hx : x < 0) :
  ∃ (max : ℝ), max = -4 * Real.sqrt 3 ∧ ∀ y, y = 3 * x + 4 / x → y ≤ max :=
sorry

end max_value_of_function_l2340_234031


namespace courier_cost_formula_l2340_234019

def courier_cost (P : ℕ+) : ℕ :=
  if P ≤ 2 then 15 else 15 + 5 * (P - 2)

theorem courier_cost_formula (P : ℕ+) :
  courier_cost P = if P ≤ 2 then 15 else 15 + 5 * (P - 2) :=
by sorry

end courier_cost_formula_l2340_234019


namespace shopkeeper_profit_l2340_234035

/-- Calculates the mean daily profit for a month given the mean profits of two halves --/
def mean_daily_profit (days : ℕ) (first_half_mean : ℚ) (second_half_mean : ℚ) : ℚ :=
  (first_half_mean * (days / 2) + second_half_mean * (days / 2)) / days

/-- Proves that the mean daily profit for the given scenario is 350 --/
theorem shopkeeper_profit : mean_daily_profit 30 245 455 = 350 := by
  sorry

end shopkeeper_profit_l2340_234035


namespace lcm_of_20_28_45_l2340_234053

theorem lcm_of_20_28_45 : Nat.lcm (Nat.lcm 20 28) 45 = 1260 := by
  sorry

end lcm_of_20_28_45_l2340_234053


namespace min_value_implies_a_l2340_234092

/-- The function f(x) defined as |x+1| + |2x+a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |2*x + a|

/-- The theorem stating that if the minimum value of f(x) is 3, then a = -4 or a = 8 -/
theorem min_value_implies_a (a : ℝ) : (∀ x, f a x ≥ 3) ∧ (∃ x, f a x = 3) → a = -4 ∨ a = 8 := by
  sorry


end min_value_implies_a_l2340_234092


namespace sum_of_xyz_l2340_234082

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + y^2 + x*y = 1)
  (eq2 : y^2 + z^2 + y*z = 4)
  (eq3 : z^2 + x^2 + z*x = 5) :
  x + y + z = Real.sqrt (5 + 2 * Real.sqrt 3) := by
sorry

end sum_of_xyz_l2340_234082


namespace additive_is_odd_l2340_234048

/-- A function satisfying the given additive property -/
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- Theorem stating that an additive function is odd -/
theorem additive_is_odd (f : ℝ → ℝ) (h : is_additive f) : 
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end additive_is_odd_l2340_234048


namespace equal_cost_for_54_students_option2_cheaper_for_50_students_students_more_than_40_l2340_234057

/-- Represents the discount options for movie tickets. -/
inductive DiscountOption
  | Option1
  | Option2

/-- Calculates the total cost for a given number of students and discount option. -/
def calculateCost (students : ℕ) (option : DiscountOption) : ℚ :=
  match option with
  | DiscountOption.Option1 => 30 * students * (1 - 1/5)
  | DiscountOption.Option2 => 30 * (students - 6) * (1 - 1/10)

/-- Theorem stating that both discount options result in the same cost for 54 students. -/
theorem equal_cost_for_54_students :
  calculateCost 54 DiscountOption.Option1 = calculateCost 54 DiscountOption.Option2 :=
by sorry

/-- Theorem stating that Option 2 is cheaper for 50 students. -/
theorem option2_cheaper_for_50_students :
  calculateCost 50 DiscountOption.Option2 < calculateCost 50 DiscountOption.Option1 :=
by sorry

/-- Theorem stating that the number of students is more than 40. -/
theorem students_more_than_40 : 54 > 40 :=
by sorry

end equal_cost_for_54_students_option2_cheaper_for_50_students_students_more_than_40_l2340_234057


namespace same_solution_implies_a_and_b_l2340_234018

theorem same_solution_implies_a_and_b (a b : ℝ) :
  (∃ x y : ℝ, x - y = 0 ∧ 2*a*x + b*y = 4 ∧ 2*x + y = 3 ∧ a*x + b*y = 3) →
  a = 1 ∧ b = 2 := by
sorry

end same_solution_implies_a_and_b_l2340_234018


namespace binomial_equation_solution_l2340_234094

theorem binomial_equation_solution (x : ℕ) : 
  (Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4) → (x = 5 ∨ x = 2) :=
by sorry

end binomial_equation_solution_l2340_234094


namespace bounded_g_given_bounded_f_l2340_234077

/-- Given real functions f and g defined on the entire real line, 
    satisfying certain conditions, prove that |g(y)| ≤ 1 for all y -/
theorem bounded_g_given_bounded_f (f g : ℝ → ℝ) 
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := by
  sorry

end bounded_g_given_bounded_f_l2340_234077


namespace max_value_of_z_l2340_234036

theorem max_value_of_z (x y : ℝ) (h1 : x + 2*y - 5 ≥ 0) (h2 : x - 2*y + 3 ≥ 0) (h3 : x - 5 ≤ 0) :
  ∀ x' y', x' + 2*y' - 5 ≥ 0 → x' - 2*y' + 3 ≥ 0 → x' - 5 ≤ 0 → x + y ≥ x' + y' ∧ x + y ≤ 9 :=
by sorry

end max_value_of_z_l2340_234036


namespace implicit_second_derivative_l2340_234091

noncomputable def y (x : ℝ) : ℝ := Real.exp x

theorem implicit_second_derivative 
  (h : ∀ x, y x * Real.exp x + Real.exp (y x) = Real.exp 1 + 1) :
  ∀ x, deriv (deriv y) x = 
    (-2 * Real.exp (2*x) * y x * (Real.exp x + Real.exp (y x)) + 
     y x * Real.exp x * (Real.exp x + Real.exp (y x))^2 + 
     (y x)^2 * Real.exp (y x) * Real.exp (2*x)) / 
    (Real.exp x + Real.exp (y x))^3 :=
by
  sorry

end implicit_second_derivative_l2340_234091


namespace bug_crawl_distance_l2340_234006

-- Define the bug's starting position
def start : Int := 3

-- Define the bug's first destination
def first_dest : Int := 9

-- Define the bug's final destination
def final_dest : Int := -4

-- Define the function to calculate distance between two points
def distance (a b : Int) : Nat := Int.natAbs (b - a)

-- Theorem statement
theorem bug_crawl_distance : 
  distance start first_dest + distance first_dest final_dest = 19 := by
  sorry

end bug_crawl_distance_l2340_234006


namespace set_element_value_l2340_234066

theorem set_element_value (a : ℝ) : 2 ∈ ({0, a, a^2 - 3*a + 2} : Set ℝ) → a = 3 := by
  sorry

end set_element_value_l2340_234066


namespace gold_coin_percentage_is_48_percent_l2340_234008

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  bead_percentage : ℝ
  silver_coin_percentage : ℝ
  gold_coin_percentage : ℝ

/-- Calculates the percentage of gold coins in the urn --/
def gold_coin_percentage (u : UrnComposition) : ℝ :=
  u.gold_coin_percentage

/-- The urn composition satisfies the given conditions --/
def valid_urn_composition (u : UrnComposition) : Prop :=
  u.bead_percentage = 0.2 ∧
  u.silver_coin_percentage + u.gold_coin_percentage = 0.8 ∧
  u.silver_coin_percentage = 0.4 * (u.silver_coin_percentage + u.gold_coin_percentage)

theorem gold_coin_percentage_is_48_percent (u : UrnComposition) 
  (h : valid_urn_composition u) : gold_coin_percentage u = 0.48 := by
  sorry

end gold_coin_percentage_is_48_percent_l2340_234008


namespace binomial_square_exists_l2340_234063

theorem binomial_square_exists : ∃ (b t u : ℝ), ∀ x : ℝ, b * x^2 + 12 * x + 9 = (t * x + u)^2 := by
  sorry

end binomial_square_exists_l2340_234063


namespace equation_solution_l2340_234071

theorem equation_solution : 
  ∃ c : ℚ, (c - 35) / 14 = (2 * c + 9) / 49 ∧ c = 1841 / 21 := by
  sorry

end equation_solution_l2340_234071


namespace construction_time_theorem_l2340_234095

/-- Represents the time taken to construct a wall given the number of boys and girls -/
def constructionTime (boys : ℕ) (girls : ℕ) : ℝ :=
  sorry

/-- Theorem stating that if 16 boys or 24 girls can construct a wall in 6 days,
    then 8 boys and 4 girls will take 12 days to construct the same wall -/
theorem construction_time_theorem :
  (constructionTime 16 0 = 6 ∧ constructionTime 0 24 = 6) →
  constructionTime 8 4 = 12 :=
sorry

end construction_time_theorem_l2340_234095


namespace picture_book_shelves_l2340_234044

theorem picture_book_shelves 
  (books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) 
  (total_books : ℕ) : 
  books_per_shelf = 4 → 
  mystery_shelves = 5 → 
  total_books = 32 → 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 3 := by
sorry

end picture_book_shelves_l2340_234044


namespace min_value_a5_plus_a6_l2340_234001

/-- A positive arithmetic geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r d : ℝ), r > 1 ∧ d > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n + d

theorem min_value_a5_plus_a6 (a : ℕ → ℝ) :
  ArithmeticGeometricSequence a →
  a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6 →
  ∃ (min : ℝ), min = 48 ∧ ∀ x, (∃ b, ArithmeticGeometricSequence b ∧ 
    b 4 + b 3 - 2 * b 2 - 2 * b 1 = 6 ∧ b 5 + b 6 = x) → x ≥ min :=
by sorry

end min_value_a5_plus_a6_l2340_234001


namespace difference_of_squares_65_35_l2340_234060

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by sorry

end difference_of_squares_65_35_l2340_234060


namespace set_equality_implies_a_equals_three_l2340_234074

theorem set_equality_implies_a_equals_three (a : ℝ) : 
  let A : Set ℝ := {0, 1, a^2}
  let B : Set ℝ := {1, 0, 2*a+3}
  A ∩ B = A ∪ B → a = 3 := by
sorry

end set_equality_implies_a_equals_three_l2340_234074


namespace trapezoid_circle_radii_l2340_234028

/-- An isosceles trapezoid with inscribed and circumscribed circles -/
structure IsoscelesTrapezoid where
  -- Base lengths
  BC : ℝ
  AD : ℝ
  -- Inscribed circle exists
  has_inscribed_circle : Bool
  -- Circumscribed circle exists
  has_circumscribed_circle : Bool

/-- The radii of inscribed and circumscribed circles of an isosceles trapezoid -/
def circle_radii (t : IsoscelesTrapezoid) : ℝ × ℝ :=
  sorry

theorem trapezoid_circle_radii (t : IsoscelesTrapezoid) 
  (h1 : t.BC = 4)
  (h2 : t.AD = 16)
  (h3 : t.has_inscribed_circle = true)
  (h4 : t.has_circumscribed_circle = true) :
  circle_radii t = (4, 5 * Real.sqrt 41 / 4) :=
sorry

end trapezoid_circle_radii_l2340_234028


namespace no_natural_solutions_l2340_234090

theorem no_natural_solutions :
  ∀ (x y : ℕ), y^2 ≠ x^2 + x + 1 := by
  sorry

end no_natural_solutions_l2340_234090


namespace compound_molecular_weight_l2340_234023

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_P : ℝ := 30.97
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms for each element
def num_Al : ℕ := 1
def num_P : ℕ := 1
def num_O : ℕ := 4

-- Define the molecular weight calculation function
def molecular_weight (w_Al w_P w_O : ℝ) (n_Al n_P n_O : ℕ) : ℝ :=
  w_Al * n_Al + w_P * n_P + w_O * n_O

-- Theorem statement
theorem compound_molecular_weight :
  molecular_weight atomic_weight_Al atomic_weight_P atomic_weight_O num_Al num_P num_O = 121.95 := by
  sorry

end compound_molecular_weight_l2340_234023


namespace binomial_25_2_l2340_234049

theorem binomial_25_2 : Nat.choose 25 2 = 300 := by
  sorry

end binomial_25_2_l2340_234049


namespace point_A_coordinates_l2340_234058

/-- A point lies on the x-axis if and only if its y-coordinate is 0 -/
def lies_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- The coordinates of point A as a function of x -/
def point_A (x : ℝ) : ℝ × ℝ := (2 - x, x + 3)

/-- Theorem stating that if point A lies on the x-axis, its coordinates are (5, 0) -/
theorem point_A_coordinates :
  ∃ x : ℝ, lies_on_x_axis (point_A x) → point_A x = (5, 0) := by
  sorry


end point_A_coordinates_l2340_234058


namespace p_necessary_not_sufficient_l2340_234068

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a ≥ 0

def q (a : ℝ) : Prop := -1 < a ∧ a < 0

-- State the theorem
theorem p_necessary_not_sufficient :
  (∀ a : ℝ, q a → p a) ∧ (∃ a : ℝ, p a ∧ ¬q a) := by sorry

end p_necessary_not_sufficient_l2340_234068


namespace max_distance_for_specific_car_l2340_234004

/-- Represents the lifespan of a set of tires in kilometers. -/
structure TireLifespan where
  km : ℕ

/-- Represents a car with front and rear tires. -/
structure Car where
  frontTires : TireLifespan
  rearTires : TireLifespan

/-- Calculates the maximum distance a car can travel with optimal tire swapping. -/
def maxDistance (car : Car) : ℕ :=
  sorry

/-- Theorem stating the maximum distance for a specific car configuration. -/
theorem max_distance_for_specific_car :
  let car := Car.mk (TireLifespan.mk 20000) (TireLifespan.mk 30000)
  maxDistance car = 24000 := by
  sorry

end max_distance_for_specific_car_l2340_234004


namespace nth_ring_area_l2340_234009

/-- Represents the area of a ring in a square garden system -/
def ring_area (n : ℕ) : ℝ :=
  36 * n

/-- Theorem stating the area of the nth ring in a square garden system -/
theorem nth_ring_area (n : ℕ) :
  ring_area n = 36 * n :=
by
  -- The proof goes here
  sorry

/-- The area of the 50th ring -/
def area_50th_ring : ℝ := ring_area 50

#eval area_50th_ring  -- Should evaluate to 1800

end nth_ring_area_l2340_234009


namespace pythagorean_chord_l2340_234079

theorem pythagorean_chord (m : ℕ) (h : m ≥ 3) : 
  let width := 2 * m
  let height := m^2 - 1
  let diagonal := height + 2
  width^2 + height^2 = diagonal^2 ∧ diagonal = m^2 + 1 := by
sorry

end pythagorean_chord_l2340_234079


namespace smallest_bdf_value_l2340_234052

theorem smallest_bdf_value (a b c d e f : ℕ+) : 
  let expr := (a : ℚ) / b * (c : ℚ) / d * (e : ℚ) / f
  (expr + 3 = ((a + 1 : ℕ+) : ℚ) / b * (c : ℚ) / d * (e : ℚ) / f) →
  (expr + 4 = (a : ℚ) / b * ((c + 1 : ℕ+) : ℚ) / d * (e : ℚ) / f) →
  (expr + 5 = (a : ℚ) / b * (c : ℚ) / d * ((e + 1 : ℕ+) : ℚ) / f) →
  (∀ k : ℕ+, (b * d * f : ℕ) = k → k ≥ 60) ∧ 
  (∃ b' d' f' : ℕ+, (b' * d' * f' : ℕ) = 60) :=
by sorry

end smallest_bdf_value_l2340_234052


namespace juan_number_puzzle_l2340_234034

theorem juan_number_puzzle (n : ℝ) : ((((n + 2) * 2) - 2) / 2) = 7 → n = 6 := by
  sorry

end juan_number_puzzle_l2340_234034


namespace some_number_value_l2340_234088

theorem some_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end some_number_value_l2340_234088


namespace marion_additional_points_l2340_234081

/-- Represents the additional points Marion got on the exam -/
def additional_points (total_items : ℕ) (ella_incorrect : ℕ) (marion_score : ℕ) : ℕ :=
  marion_score - (total_items - ella_incorrect) / 2

/-- Proves that Marion got 6 additional points given the exam conditions -/
theorem marion_additional_points :
  additional_points 40 4 24 = 6 := by
  sorry

end marion_additional_points_l2340_234081


namespace football_lineup_combinations_l2340_234017

def team_size : ℕ := 12
def offensive_linemen : ℕ := 4
def positions : ℕ := 5

def lineup_combinations : ℕ := 31680

theorem football_lineup_combinations :
  team_size = 12 ∧ 
  offensive_linemen = 4 ∧ 
  positions = 5 →
  lineup_combinations = offensive_linemen * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4) :=
by sorry

end football_lineup_combinations_l2340_234017


namespace baking_time_ratio_l2340_234093

def usual_assembly_time : ℝ := 1
def usual_baking_time : ℝ := 1.5
def usual_decorating_time : ℝ := 1
def total_time_on_failed_day : ℝ := 5

theorem baking_time_ratio :
  let usual_total_time := usual_assembly_time + usual_baking_time + usual_decorating_time
  let baking_time_on_failed_day := total_time_on_failed_day - usual_assembly_time - usual_decorating_time
  baking_time_on_failed_day / usual_baking_time = 2 := by
sorry

end baking_time_ratio_l2340_234093


namespace nickels_maximize_value_expected_value_is_7480_l2340_234098

/-- Represents the types of coins --/
inductive Coin
| Quarter
| Nickel
| Dime

/-- Represents the material of a coin --/
inductive Material
| Regular
| Iron

/-- The number of quarters Alice has --/
def initial_quarters : ℕ := 20

/-- The exchange rate for quarters to nickels --/
def quarters_to_nickels : ℕ := 4

/-- The exchange rate for quarters to dimes --/
def quarters_to_dimes : ℕ := 2

/-- The probability of a nickel being iron --/
def iron_nickel_prob : ℚ := 3/10

/-- The probability of a dime being iron --/
def iron_dime_prob : ℚ := 1/10

/-- The value of an iron nickel in cents --/
def iron_nickel_value : ℕ := 300

/-- The value of an iron dime in cents --/
def iron_dime_value : ℕ := 500

/-- The value of a regular nickel in cents --/
def regular_nickel_value : ℕ := 5

/-- The value of a regular dime in cents --/
def regular_dime_value : ℕ := 10

/-- Calculates the expected value of a nickel in cents --/
def expected_nickel_value : ℚ :=
  iron_nickel_prob * iron_nickel_value + (1 - iron_nickel_prob) * regular_nickel_value

/-- Calculates the expected value of a dime in cents --/
def expected_dime_value : ℚ :=
  iron_dime_prob * iron_dime_value + (1 - iron_dime_prob) * regular_dime_value

/-- Theorem stating that exchanging for nickels maximizes expected value --/
theorem nickels_maximize_value :
  expected_nickel_value * quarters_to_nickels > expected_dime_value * quarters_to_dimes :=
sorry

/-- Calculates the total number of nickels Alice can get --/
def total_nickels : ℕ := initial_quarters * quarters_to_nickels

/-- Calculates the expected total value in cents after exchanging for nickels --/
def expected_total_value : ℚ := total_nickels * expected_nickel_value

/-- Theorem stating that the expected total value is 7480 cents ($74.80) --/
theorem expected_value_is_7480 : expected_total_value = 7480 := sorry

end nickels_maximize_value_expected_value_is_7480_l2340_234098


namespace sqrt_equation_solution_l2340_234016

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 - 2*x + x^2) = 9 ↔ x = 1 + Real.sqrt 78 ∨ x = 1 - Real.sqrt 78 :=
by sorry

end sqrt_equation_solution_l2340_234016


namespace triangle_max_perimeter_l2340_234084

theorem triangle_max_perimeter (A B C : ℝ) (b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  (1 - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C →
  1 + b + c ≤ 3 :=
sorry

end triangle_max_perimeter_l2340_234084


namespace a_gt_b_neither_sufficient_nor_necessary_for_abs_a_gt_abs_b_l2340_234065

theorem a_gt_b_neither_sufficient_nor_necessary_for_abs_a_gt_abs_b :
  ¬(∀ a b : ℝ, a > b → |a| > |b|) ∧ ¬(∀ a b : ℝ, |a| > |b| → a > b) := by
  sorry

end a_gt_b_neither_sufficient_nor_necessary_for_abs_a_gt_abs_b_l2340_234065


namespace determine_original_prices_l2340_234047

/-- Represents a purchase of products A and B -/
structure Purchase where
  quantityA : ℕ
  quantityB : ℕ
  totalPrice : ℕ

/-- Represents the store's pricing system -/
structure Store where
  priceA : ℕ
  priceB : ℕ

/-- Checks if a purchase is consistent with the store's pricing -/
def isPurchaseConsistent (s : Store) (p : Purchase) : Prop :=
  s.priceA * p.quantityA + s.priceB * p.quantityB = p.totalPrice

/-- The theorem stating that given the purchase data, we can determine the original prices -/
theorem determine_original_prices 
  (p1 p2 : Purchase)
  (h1 : p1.quantityA = 6 ∧ p1.quantityB = 5 ∧ p1.totalPrice = 1140)
  (h2 : p2.quantityA = 3 ∧ p2.quantityB = 7 ∧ p2.totalPrice = 1110) :
  ∃ (s : Store), 
    s.priceA = 90 ∧ 
    s.priceB = 120 ∧ 
    isPurchaseConsistent s p1 ∧ 
    isPurchaseConsistent s p2 :=
  sorry

end determine_original_prices_l2340_234047


namespace probability_of_drawing_math_books_l2340_234073

/-- The number of Chinese books -/
def chinese_books : ℕ := 10

/-- The number of math books -/
def math_books : ℕ := 2

/-- The total number of books -/
def total_books : ℕ := chinese_books + math_books

/-- The number of books to be drawn -/
def books_drawn : ℕ := 2

theorem probability_of_drawing_math_books :
  (Nat.choose total_books books_drawn - Nat.choose chinese_books books_drawn) / Nat.choose total_books books_drawn = 7 / 22 := by
  sorry

end probability_of_drawing_math_books_l2340_234073


namespace distance_before_gas_is_32_l2340_234000

/-- The distance driven before stopping for gas -/
def distance_before_gas (total_distance remaining_distance : ℕ) : ℕ :=
  total_distance - remaining_distance

/-- Theorem: The distance driven before stopping for gas is 32 miles -/
theorem distance_before_gas_is_32 :
  distance_before_gas 78 46 = 32 := by
  sorry

end distance_before_gas_is_32_l2340_234000


namespace geli_workout_results_l2340_234099

/-- Represents a workout routine with push-ups and runs -/
structure WorkoutRoutine where
  workoutsPerWeek : ℕ
  weeks : ℕ
  initialPushups : ℕ
  pushupIncrement : ℕ
  pushupsMileRatio : ℕ

/-- Calculates the total number of push-ups for a given workout routine -/
def totalPushups (routine : WorkoutRoutine) : ℕ :=
  let totalDays := routine.workoutsPerWeek * routine.weeks
  let lastDayPushups := routine.initialPushups + (totalDays - 1) * routine.pushupIncrement
  totalDays * (routine.initialPushups + lastDayPushups) / 2

/-- Calculates the number of one-mile runs based on the total push-ups -/
def totalRuns (routine : WorkoutRoutine) : ℕ :=
  totalPushups routine / routine.pushupsMileRatio

/-- Theorem stating the results for Geli's specific workout routine -/
theorem geli_workout_results :
  let routine : WorkoutRoutine := {
    workoutsPerWeek := 3,
    weeks := 4,
    initialPushups := 10,
    pushupIncrement := 5,
    pushupsMileRatio := 30
  }
  totalPushups routine = 450 ∧ totalRuns routine = 15 := by
  sorry


end geli_workout_results_l2340_234099


namespace min_value_exponential_function_l2340_234029

theorem min_value_exponential_function (x : ℝ) : Real.exp x + 4 * Real.exp (-x) ≥ 4 := by
  sorry

end min_value_exponential_function_l2340_234029


namespace cubic_roots_property_l2340_234011

theorem cubic_roots_property (a b c t u v : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = t ∨ x = u ∨ x = v) →
  (∀ x : ℝ, x^3 + a^3*x^2 + b^3*x + c^3 = 0 ↔ x = t^3 ∨ x = u^3 ∨ x = v^3) ↔
  ∃ t : ℝ, a = t ∧ b = 0 ∧ c = 0 :=
by sorry

end cubic_roots_property_l2340_234011


namespace sum_of_largest_and_smallest_prime_factors_l2340_234027

def number : ℕ := 1386

theorem sum_of_largest_and_smallest_prime_factors :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ 
    largest.Prime ∧
    smallest ∣ number ∧ 
    largest ∣ number ∧
    (∀ p : ℕ, p.Prime → p ∣ number → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ number → p ≥ smallest) ∧
    smallest + largest = 13 :=
by sorry

end sum_of_largest_and_smallest_prime_factors_l2340_234027


namespace min_cuts_for_100_polygons_l2340_234072

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- Represents the state of the paper after cutting -/
structure PaperState where
  pieces : ℕ
  total_vertices : ℕ

/-- Initial state of the square paper -/
def initial_state : PaperState :=
  { pieces := 1, total_vertices := 4 }

/-- Function to model a single cut -/
def cut (state : PaperState) (new_vertices : ℕ) : PaperState :=
  { pieces := state.pieces + 1,
    total_vertices := state.total_vertices + new_vertices }

/-- Predicate to check if the final state is valid -/
def is_valid_final_state (state : PaperState) : Prop :=
  state.pieces = 100 ∧ state.total_vertices = 100 * 20

/-- Theorem stating the minimum number of cuts required -/
theorem min_cuts_for_100_polygons :
  ∃ (n : ℕ), n = 1699 ∧
  ∃ (cut_sequence : List ℕ),
    cut_sequence.length = n ∧
    (cut_sequence.all (λ x => x ∈ [2, 3, 4])) ∧
    is_valid_final_state (cut_sequence.foldl cut initial_state) ∧
    ∀ (m : ℕ) (other_sequence : List ℕ),
      m < n →
      other_sequence.length = m →
      (other_sequence.all (λ x => x ∈ [2, 3, 4])) →
      ¬is_valid_final_state (other_sequence.foldl cut initial_state) :=
sorry


end min_cuts_for_100_polygons_l2340_234072


namespace postman_june_distance_l2340_234007

/-- Represents a step counter with a maximum count before resetting -/
structure StepCounter where
  max_count : ℕ
  resets : ℕ
  final_count : ℕ

/-- Calculates the total number of steps based on the step counter data -/
def total_steps (counter : StepCounter) : ℕ :=
  counter.max_count * counter.resets + counter.final_count

/-- Converts steps to miles given the number of steps per mile -/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

/-- Theorem stating that given the specified conditions, the total distance walked is 2615 miles -/
theorem postman_june_distance :
  let counter : StepCounter := ⟨100000, 52, 30000⟩
  let steps_per_mile : ℕ := 2000
  steps_to_miles (total_steps counter) steps_per_mile = 2615 := by
  sorry

end postman_june_distance_l2340_234007


namespace cube_painting_probability_l2340_234037

/-- Represents the three possible colors for painting cube faces -/
inductive Color
  | Black
  | White
  | Red

/-- Represents a painted cube -/
def Cube := Fin 6 → Color

/-- The number of ways to paint a single cube -/
def total_single_cube_paintings : ℕ := 729

/-- The total number of ways to paint two cubes -/
def total_two_cube_paintings : ℕ := 531441

/-- The number of ways to paint two cubes so they are identical after rotation -/
def identical_after_rotation : ℕ := 1178

/-- The probability that two independently painted cubes are identical after rotation -/
def probability_identical_after_rotation : ℚ := 1178 / 531441

theorem cube_painting_probability :
  probability_identical_after_rotation = identical_after_rotation / total_two_cube_paintings :=
by sorry

end cube_painting_probability_l2340_234037


namespace quadratic_function_max_value_l2340_234080

theorem quadratic_function_max_value (m n : ℝ) : 
  m^2 - 4*n ≥ 0 →
  (m - 1)^2 + (n - 1)^2 + (m - n)^2 ≤ 9/8 :=
by sorry

end quadratic_function_max_value_l2340_234080


namespace sum_reciprocals_inequality_l2340_234030

theorem sum_reciprocals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := by
  sorry

end sum_reciprocals_inequality_l2340_234030


namespace time_difference_for_trips_l2340_234051

/-- Given a truck traveling at a constant speed, this theorem proves the time difference
    between two trips of different distances. -/
theorem time_difference_for_trips
  (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ)
  (h1 : speed = 60)  -- Speed in miles per hour
  (h2 : distance1 = 570)  -- Distance of first trip in miles
  (h3 : distance2 = 540)  -- Distance of second trip in miles
  : (distance1 - distance2) / speed * 60 = 30 := by
  sorry

end time_difference_for_trips_l2340_234051


namespace isotomic_lines_not_intersect_in_medial_triangle_l2340_234062

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The medial triangle of a given triangle --/
def medialTriangle (t : Triangle) : Triangle := sorry

/-- Checks if a point is inside or on the boundary of a triangle --/
def isInsideOrOnTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Checks if two lines are isotomic with respect to a triangle --/
def areIsotomicLines (l1 l2 : Line) (t : Triangle) : Prop := sorry

/-- The intersection point of two lines, if it exists --/
def lineIntersection (l1 l2 : Line) : Option (ℝ × ℝ) := sorry

theorem isotomic_lines_not_intersect_in_medial_triangle (t : Triangle) (l1 l2 : Line) :
  areIsotomicLines l1 l2 t →
  match lineIntersection l1 l2 with
  | some p => ¬isInsideOrOnTriangle p (medialTriangle t)
  | none => True
  := by sorry

end isotomic_lines_not_intersect_in_medial_triangle_l2340_234062


namespace total_spider_legs_l2340_234038

/-- The number of spiders in Zoey's room -/
def num_spiders : ℕ := 5

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs in Zoey's room is 40 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 40 := by
  sorry

end total_spider_legs_l2340_234038


namespace min_area_PJ1J2_l2340_234086

/-- Triangle PQR with given side lengths -/
structure Triangle (PQ QR PR : ℝ) where
  side_PQ : PQ = 26
  side_QR : QR = 28
  side_PR : PR = 30

/-- Point Y on side QR -/
def Y (QR : ℝ) := ℝ

/-- Incenter of a triangle -/
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem min_area_PJ1J2 (t : Triangle 26 28 30) (y : Y 28) :
  ∃ (P Q R : ℝ × ℝ),
    let J1 := incenter P Q (0, y)
    let J2 := incenter P R (0, y)
    ∀ (y' : Y 28),
      let J1' := incenter P Q (0, y')
      let J2' := incenter P R (0, y')
      triangle_area P J1 J2 ≤ triangle_area P J1' J2' ∧
      (∃ (y_min : Y 28), triangle_area P J1 J2 = 51) := by
  sorry

end min_area_PJ1J2_l2340_234086


namespace quarters_needed_l2340_234064

/-- Represents the cost of items in cents -/
def CandyBarCost : ℕ := 25
def ChocolatePieceCost : ℕ := 75
def JuicePackCost : ℕ := 50

/-- Represents the number of each item to be purchased -/
def CandyBarCount : ℕ := 3
def ChocolatePieceCount : ℕ := 2
def JuicePackCount : ℕ := 1

/-- Represents the value of a quarter in cents -/
def QuarterValue : ℕ := 25

/-- Calculates the total cost in cents -/
def TotalCost : ℕ := 
  CandyBarCost * CandyBarCount + 
  ChocolatePieceCost * ChocolatePieceCount + 
  JuicePackCost * JuicePackCount

/-- Theorem: The number of quarters needed is 11 -/
theorem quarters_needed : TotalCost / QuarterValue = 11 := by
  sorry

end quarters_needed_l2340_234064


namespace factorization_x_10_minus_1024_l2340_234021

theorem factorization_x_10_minus_1024 (x : ℝ) : 
  x^10 - 1024 = (x^5 + 32) * (x^5 - 32) :=
by
  sorry

end factorization_x_10_minus_1024_l2340_234021


namespace square_diff_fourth_power_l2340_234097

theorem square_diff_fourth_power : (7^2 - 5^2)^4 = 331776 := by sorry

end square_diff_fourth_power_l2340_234097


namespace sum_mod_nine_l2340_234026

theorem sum_mod_nine : (2 + 44 + 666 + 8888 + 111110 + 13131312 + 1414141414) % 9 = 6 := by
  sorry

end sum_mod_nine_l2340_234026


namespace smallest_integer_fraction_l2340_234076

theorem smallest_integer_fraction (y : ℤ) : (8 : ℚ) / 12 < (y : ℚ) / 15 ↔ 11 ≤ y := by
  sorry

end smallest_integer_fraction_l2340_234076


namespace f_g_four_zeros_implies_a_range_l2340_234050

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*a*x - a + 1 else Real.log (-x)

def g (a : ℝ) (x : ℝ) : ℝ := x^2 + 1 - 2*a

def has_four_zeros (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧
    ∀ x, f x = 0 → (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

theorem f_g_four_zeros_implies_a_range (a : ℝ) :
  has_four_zeros (f a ∘ g a) →
  a ∈ Set.Ioo ((Real.sqrt 5 - 1) / 2) 1 ∪ Set.Ioi 1 :=
by sorry

end f_g_four_zeros_implies_a_range_l2340_234050


namespace problem_solution_l2340_234039

theorem problem_solution : 
  let M : ℤ := 2007 / 3
  let N : ℤ := M / 3
  let X : ℤ := M - N
  X = 446 := by sorry

end problem_solution_l2340_234039


namespace frog_jump_probability_l2340_234014

-- Define the grid size
def gridSize : ℕ := 6

-- Define the jump size
def jumpSize : ℕ := 2

-- Define a position on the grid
structure Position where
  x : ℕ
  y : ℕ

-- Define the starting position
def startPos : Position := ⟨2, 3⟩

-- Define a function to check if a position is on the vertical side
def isOnVerticalSide (p : Position) : Prop :=
  p.x = 0 ∨ p.x = gridSize

-- Define a function to check if a position is on any side
def isOnAnySide (p : Position) : Prop :=
  p.x = 0 ∨ p.x = gridSize ∨ p.y = 0 ∨ p.y = gridSize

-- Define the probability of ending on a vertical side
def probEndVertical (p : Position) : ℝ := sorry

-- State the theorem
theorem frog_jump_probability :
  probEndVertical startPos = 3/4 := by sorry

end frog_jump_probability_l2340_234014
