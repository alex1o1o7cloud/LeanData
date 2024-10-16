import Mathlib

namespace NUMINAMATH_CALUDE_hamburger_combinations_l2503_250360

/-- The number of available condiments -/
def num_condiments : ℕ := 9

/-- The number of patty options -/
def num_patty_options : ℕ := 4

/-- The number of possible combinations for condiments -/
def condiment_combinations : ℕ := 2^num_condiments

/-- The total number of different hamburger combinations -/
def total_combinations : ℕ := num_patty_options * condiment_combinations

/-- Theorem stating that the total number of different hamburger combinations is 2048 -/
theorem hamburger_combinations : total_combinations = 2048 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l2503_250360


namespace NUMINAMATH_CALUDE_axis_of_symmetry_sin_l2503_250337

theorem axis_of_symmetry_sin (x : ℝ) : 
  x = π / 12 → 
  ∃ k : ℤ, 2 * x + π / 3 = π / 2 + k * π :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_sin_l2503_250337


namespace NUMINAMATH_CALUDE_min_participants_l2503_250367

/-- Represents a participant in the race -/
structure Participant where
  name : String
  position : Nat

/-- Represents the race -/
structure Race where
  participants : List Participant
  /-- No two participants finished simultaneously -/
  no_ties : ∀ p1 p2 : Participant, p1 ∈ participants → p2 ∈ participants → p1 ≠ p2 → p1.position ≠ p2.position

/-- The number of people who finished before a given participant -/
def finished_before (race : Race) (p : Participant) : Nat :=
  (race.participants.filter (fun q => q.position < p.position)).length

/-- The number of people who finished after a given participant -/
def finished_after (race : Race) (p : Participant) : Nat :=
  (race.participants.filter (fun q => q.position > p.position)).length

/-- The theorem stating the minimum number of participants in the race -/
theorem min_participants (race : Race) 
  (andrei dima lenya : Participant)
  (andrei_in : andrei ∈ race.participants)
  (dima_in : dima ∈ race.participants)
  (lenya_in : lenya ∈ race.participants)
  (andrei_cond : finished_before race andrei = (finished_after race andrei) / 2)
  (dima_cond : finished_before race dima = (finished_after race dima) / 3)
  (lenya_cond : finished_before race lenya = (finished_after race lenya) / 4) :
  race.participants.length ≥ 61 := by
  sorry

end NUMINAMATH_CALUDE_min_participants_l2503_250367


namespace NUMINAMATH_CALUDE_function_value_at_one_l2503_250338

theorem function_value_at_one (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x - x^2| ≤ 1/4)
  (h2 : ∀ x, |f x + 1 - x^2| ≤ 3/4) : 
  f 1 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_one_l2503_250338


namespace NUMINAMATH_CALUDE_counterfeit_coin_determination_l2503_250389

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a group of coins -/
structure CoinGroup where
  size : Nat
  hasFake : Bool

/-- Represents a weighing action -/
structure Weighing where
  left : CoinGroup
  right : CoinGroup

/-- The state of knowledge about the counterfeit coins -/
inductive FakeState
  | Unknown : FakeState
  | Heavier : FakeState
  | Lighter : FakeState

/-- A strategy for determining the state of the counterfeit coins -/
def Strategy := List Weighing

/-- The result of applying a strategy -/
def StrategyResult := FakeState

/-- Axiom: There are 239 coins in total -/
axiom total_coins : Nat
axiom total_coins_eq : total_coins = 239

/-- Axiom: There are exactly two counterfeit coins -/
axiom num_fake_coins : Nat
axiom num_fake_coins_eq : num_fake_coins = 2

/-- Theorem: It is possible to determine whether the counterfeit coins are heavier or lighter in exactly three weighings -/
theorem counterfeit_coin_determination :
  ∃ (s : Strategy),
    (s.length = 3) ∧
    (∀ (fake_heavier : Bool),
      ∃ (result : StrategyResult),
        (result = FakeState.Heavier ∧ fake_heavier = true) ∨
        (result = FakeState.Lighter ∧ fake_heavier = false)) :=
by sorry

end NUMINAMATH_CALUDE_counterfeit_coin_determination_l2503_250389


namespace NUMINAMATH_CALUDE_calf_cost_l2503_250314

/-- Given a cow and a calf where the total cost is $990 and the cow costs 8 times as much as the calf, 
    the cost of the calf is $110. -/
theorem calf_cost (total_cost : ℕ) (cow_calf_ratio : ℕ) (calf_cost : ℕ) : 
  total_cost = 990 → 
  cow_calf_ratio = 8 → 
  calf_cost + cow_calf_ratio * calf_cost = total_cost → 
  calf_cost = 110 := by
  sorry

end NUMINAMATH_CALUDE_calf_cost_l2503_250314


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2503_250399

theorem polygon_sides_count (n : ℕ) : n > 2 → (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2503_250399


namespace NUMINAMATH_CALUDE_volunteer_assignment_l2503_250394

def number_of_volunteers : ℕ := 6
def number_for_training : ℕ := 4
def number_per_location : ℕ := 2

def select_and_assign (n m k : ℕ) : Prop :=
  ∃ (total : ℕ),
    total = Nat.choose (n - 1) k * Nat.choose (n - k - 1) k +
            Nat.choose (n - 1) 1 * Nat.choose (n - 2) k ∧
    total = 60

theorem volunteer_assignment :
  select_and_assign number_of_volunteers number_for_training number_per_location :=
sorry

end NUMINAMATH_CALUDE_volunteer_assignment_l2503_250394


namespace NUMINAMATH_CALUDE_carrot_cost_theorem_l2503_250365

/-- Calculates the total cost of carrots for a year given the daily consumption, carrots per bag, and cost per bag. -/
theorem carrot_cost_theorem (carrots_per_day : ℕ) (carrots_per_bag : ℕ) (cost_per_bag : ℚ) :
  carrots_per_day = 1 →
  carrots_per_bag = 5 →
  cost_per_bag = 2 →
  (365 * carrots_per_day / carrots_per_bag : ℚ).ceil * cost_per_bag = 146 := by
  sorry

#eval (365 * 1 / 5 : ℚ).ceil * 2

end NUMINAMATH_CALUDE_carrot_cost_theorem_l2503_250365


namespace NUMINAMATH_CALUDE_sum_of_mixed_numbers_l2503_250377

theorem sum_of_mixed_numbers : 
  (481 + 1/6 : ℚ) + (265 + 1/12 : ℚ) + (904 + 1/20 : ℚ) - 
  (184 + 29/30 : ℚ) - (160 + 41/42 : ℚ) - (703 + 55/56 : ℚ) = 
  603 + 3/8 := by sorry

end NUMINAMATH_CALUDE_sum_of_mixed_numbers_l2503_250377


namespace NUMINAMATH_CALUDE_percentage_married_employees_l2503_250339

/-- The percentage of married employees in a company given specific conditions -/
theorem percentage_married_employees :
  let percent_women : ℝ := 0.61
  let percent_married_women : ℝ := 0.7704918032786885
  let percent_single_men : ℝ := 2/3
  
  let percent_men : ℝ := 1 - percent_women
  let percent_married_men : ℝ := 1 - percent_single_men
  
  let married_women : ℝ := percent_women * percent_married_women
  let married_men : ℝ := percent_men * percent_married_men
  
  let total_married : ℝ := married_women + married_men
  
  total_married = 0.60020016000000005
  := by sorry

end NUMINAMATH_CALUDE_percentage_married_employees_l2503_250339


namespace NUMINAMATH_CALUDE_range_of_a_l2503_250388

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 4

-- State the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x ∈ Set.Icc (a - 2) (a^2), f a x ∈ Set.Icc (-4) 0) →
  a ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2503_250388


namespace NUMINAMATH_CALUDE_cody_tickets_l2503_250312

def arcade_tickets (initial_tickets spent_tickets additional_tickets : ℕ) : ℕ :=
  initial_tickets - spent_tickets + additional_tickets

theorem cody_tickets : arcade_tickets 49 25 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cody_tickets_l2503_250312


namespace NUMINAMATH_CALUDE_base_10_729_to_base_7_l2503_250329

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7 + d

/-- The base-10 number 729 is equal to 2061 in base 7 --/
theorem base_10_729_to_base_7 : base7ToBase10 2 0 6 1 = 729 := by
  sorry

end NUMINAMATH_CALUDE_base_10_729_to_base_7_l2503_250329


namespace NUMINAMATH_CALUDE_tangent_triangle_angle_theorem_l2503_250327

-- Define the circle
variable (O : Point)

-- Define the triangle
variable (P A B : Point)

-- Define the property that PAB is formed by tangents to circle O
def is_tangent_triangle (O P A B : Point) : Prop := sorry

-- Define the measure of an angle
def angle_measure (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem tangent_triangle_angle_theorem 
  (h_tangent : is_tangent_triangle O P A B)
  (h_angle : angle_measure A P B = 50) :
  angle_measure A O B = 65 := by sorry

end NUMINAMATH_CALUDE_tangent_triangle_angle_theorem_l2503_250327


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l2503_250390

/-- Given a person walking at two different speeds, prove the actual distance traveled -/
theorem actual_distance_traveled 
  (initial_speed : ℝ) 
  (increased_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : initial_speed = 10) 
  (h2 : increased_speed = 15) 
  (h3 : additional_distance = 15) 
  (h4 : ∃ t : ℝ, increased_speed * t = initial_speed * t + additional_distance) : 
  ∃ d : ℝ, d = 30 ∧ d = initial_speed * (additional_distance / (increased_speed - initial_speed)) :=
by sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l2503_250390


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l2503_250322

/-- Given a cubic function f(x) = ax³ + x + 1, prove that if its tangent line at 
    (1, f(1)) passes through (2, 7), then a = 1. -/
theorem tangent_line_cubic (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + x + 1
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 1
  let tangent_slope : ℝ := f' 1
  let point_on_curve : ℝ := f 1
  (point_on_curve - 7) / (1 - 2) = tangent_slope → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l2503_250322


namespace NUMINAMATH_CALUDE_shorts_cost_l2503_250358

def total_spent : ℝ := 33.56
def shirt_cost : ℝ := 12.14
def jacket_cost : ℝ := 7.43

theorem shorts_cost (shorts_cost : ℝ) : 
  shorts_cost = total_spent - shirt_cost - jacket_cost → shorts_cost = 13.99 := by
  sorry

end NUMINAMATH_CALUDE_shorts_cost_l2503_250358


namespace NUMINAMATH_CALUDE_garys_money_l2503_250398

/-- Gary's initial amount of money -/
def initial_amount : ℕ := sorry

/-- Amount Gary spent on the snake -/
def spent_amount : ℕ := 55

/-- Amount Gary had left after buying the snake -/
def remaining_amount : ℕ := 18

/-- Theorem stating that Gary's initial amount equals the sum of spent and remaining amounts -/
theorem garys_money : initial_amount = spent_amount + remaining_amount := by sorry

end NUMINAMATH_CALUDE_garys_money_l2503_250398


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l2503_250349

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.15 * last_year_earnings
  let this_year_rent := 0.25 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 143.75 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l2503_250349


namespace NUMINAMATH_CALUDE_age_ratio_seven_years_ago_l2503_250366

-- Define the present ages of Henry and Jill
def henry_present_age : ℕ := 25
def jill_present_age : ℕ := 16

-- Define the sum of their present ages
def sum_present_ages : ℕ := henry_present_age + jill_present_age

-- Define their ages 7 years ago
def henry_past_age : ℕ := henry_present_age - 7
def jill_past_age : ℕ := jill_present_age - 7

-- Define the theorem
theorem age_ratio_seven_years_ago :
  sum_present_ages = 41 →
  ∃ k : ℕ, henry_past_age = k * jill_past_age →
  henry_past_age / jill_past_age = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_seven_years_ago_l2503_250366


namespace NUMINAMATH_CALUDE_candy_bar_sales_l2503_250309

theorem candy_bar_sales (members : ℕ) (price : ℚ) (total_earnings : ℚ) 
  (h1 : members = 20)
  (h2 : price = 1/2)
  (h3 : total_earnings = 80) :
  (total_earnings / price) / members = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_sales_l2503_250309


namespace NUMINAMATH_CALUDE_decreasing_quadratic_function_m_range_l2503_250340

/-- A function f(x) = mx^2 + (m-1)x + 1 is decreasing on (-∞, 1] if and only if m ∈ [0, 1/3] -/
theorem decreasing_quadratic_function_m_range (m : ℝ) : 
  (∀ x ≤ 1, ∀ y ≤ 1, x < y → m * x^2 + (m - 1) * x + 1 > m * y^2 + (m - 1) * y + 1) ↔ 
  0 ≤ m ∧ m ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_function_m_range_l2503_250340


namespace NUMINAMATH_CALUDE_logarithm_expression_evaluation_l2503_250386

theorem logarithm_expression_evaluation :
  (Real.log 50 / Real.log 4) / (Real.log 4 / Real.log 25) -
  (Real.log 100 / Real.log 4) / (Real.log 4 / Real.log 50) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_evaluation_l2503_250386


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2503_250376

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2503_250376


namespace NUMINAMATH_CALUDE_interval_intersection_l2503_250333

-- Define the conditions
def condition1 (x : ℝ) : Prop := 2 < 5 * x ∧ 5 * x < 3
def condition2 (x : ℝ) : Prop := 4 < 7 * x ∧ 7 * x < 6

-- Define the theorem
theorem interval_intersection :
  ∀ x : ℝ, (condition1 x ∧ condition2 x) ↔ (4/7 < x ∧ x < 3/5) :=
sorry

end NUMINAMATH_CALUDE_interval_intersection_l2503_250333


namespace NUMINAMATH_CALUDE_ship_passengers_asia_fraction_l2503_250343

theorem ship_passengers_asia_fraction (total : ℕ) 
  (north_america : ℚ) (europe : ℚ) (africa : ℚ) (other : ℕ) :
  total = 108 →
  north_america = 1 / 12 →
  europe = 1 / 4 →
  africa = 1 / 9 →
  other = 42 →
  (north_america + europe + africa + (other : ℚ) / total + 1 / 6 : ℚ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ship_passengers_asia_fraction_l2503_250343


namespace NUMINAMATH_CALUDE_f_value_at_2_l2503_250335

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 2) : f a b 2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l2503_250335


namespace NUMINAMATH_CALUDE_direct_proportion_b_value_l2503_250380

/-- A function f is a direct proportion function if there exists a constant k such that f x = k * x for all x. -/
def IsDirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function we're considering -/
def f (b : ℝ) (x : ℝ) : ℝ := x + b - 2

theorem direct_proportion_b_value :
  (IsDirectProportion (f b)) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_b_value_l2503_250380


namespace NUMINAMATH_CALUDE_competition_theorem_l2503_250305

/-- Represents a team in the competition -/
inductive Team
| A
| B
| E

/-- Represents an event in the competition -/
inductive Event
| Vaulting
| GrenadeThrowingv
| Other1
| Other2
| Other3

/-- Represents a place in an event -/
inductive Place
| First
| Second
| Third

/-- The scoring system for the competition -/
structure ScoringSystem where
  first : ℕ
  second : ℕ
  third : ℕ
  first_gt_second : first > second
  second_gt_third : second > third
  third_pos : third > 0

/-- The result of a single event -/
structure EventResult where
  event : Event
  first : Team
  second : Team
  third : Team

/-- The final scores of the teams -/
structure FinalScores where
  team_A : ℕ
  team_B : ℕ
  team_E : ℕ

/-- The competition results -/
structure CompetitionResults where
  scoring : ScoringSystem
  events : List EventResult
  scores : FinalScores

/-- The main theorem to prove -/
theorem competition_theorem (r : CompetitionResults) : 
  r.scores.team_A = 22 ∧ 
  r.scores.team_B = 9 ∧ 
  r.scores.team_E = 9 ∧
  (∃ e : EventResult, e ∈ r.events ∧ e.event = Event.Vaulting ∧ e.first = Team.B) →
  r.events.length = 5 ∧
  (∃ e : EventResult, e ∈ r.events ∧ e.event = Event.GrenadeThrowingv ∧ e.second = Team.B) :=
by sorry

end NUMINAMATH_CALUDE_competition_theorem_l2503_250305


namespace NUMINAMATH_CALUDE_quadratic_sum_l2503_250396

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -3 and 5, and minimum value 36 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x ≥ 36) ∧
  QuadraticFunction a b c (-3) = 0 ∧
  QuadraticFunction a b c 5 = 0 →
  a + b + c = 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2503_250396


namespace NUMINAMATH_CALUDE_line_relationships_l2503_250347

-- Define a type for lines in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- For simplicity, we'll just use an opaque type
  mk :: (dummy : Unit)

-- Define the relationships between lines
def parallel (l1 l2 : Line3D) : Prop := sorry

def intersects (l1 l2 : Line3D) : Prop := sorry

def skew (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem line_relationships (a b c : Line3D) 
  (h1 : parallel a b) 
  (h2 : intersects a c) :
  skew b c ∨ intersects b c := by sorry

end NUMINAMATH_CALUDE_line_relationships_l2503_250347


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2503_250355

theorem least_positive_integer_with_remainders (N : ℕ) : 
  (N % 7 = 5) ∧ 
  (N % 8 = 6) ∧ 
  (N % 9 = 7) ∧ 
  (N % 10 = 8) ∧ 
  (∀ m : ℕ, m < N → 
    (m % 7 ≠ 5) ∨ 
    (m % 8 ≠ 6) ∨ 
    (m % 9 ≠ 7) ∨ 
    (m % 10 ≠ 8)) → 
  N = 2518 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2503_250355


namespace NUMINAMATH_CALUDE_sqrt5_diamond_sqrt5_equals_20_l2503_250359

-- Define the custom operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt5_diamond_sqrt5_equals_20 : diamond (Real.sqrt 5) (Real.sqrt 5) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sqrt5_diamond_sqrt5_equals_20_l2503_250359


namespace NUMINAMATH_CALUDE_cos_960_degrees_l2503_250303

theorem cos_960_degrees : Real.cos (960 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_960_degrees_l2503_250303


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2503_250328

theorem fraction_equals_zero (x : ℝ) (h : 6 * x ≠ 0) :
  (x - 5) / (6 * x) = 0 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2503_250328


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_magnitude_l2503_250319

theorem complex_reciprocal_sum_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_magnitude_l2503_250319


namespace NUMINAMATH_CALUDE_sqrt_t4_4t2_4_l2503_250310

theorem sqrt_t4_4t2_4 (t : ℝ) : Real.sqrt (t^4 + 4*t^2 + 4) = |t^2 + 2| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_t4_4t2_4_l2503_250310


namespace NUMINAMATH_CALUDE_least_possible_c_l2503_250357

theorem least_possible_c (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 20 →
  a ≤ b ∧ b ≤ c →
  b - a ≥ 2 ∧ c - b ≥ 2 →
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 →
  b = a + 13 →
  c ≥ 33 ∧ ∀ (c' : ℕ), c' ≥ 33 → c' % 3 = 0 → c' - b ≥ 2 → c ≤ c' :=
by sorry

end NUMINAMATH_CALUDE_least_possible_c_l2503_250357


namespace NUMINAMATH_CALUDE_sqrt_27_plus_sqrt_75_l2503_250370

theorem sqrt_27_plus_sqrt_75 : Real.sqrt 27 + Real.sqrt 75 = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_plus_sqrt_75_l2503_250370


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l2503_250320

theorem quiz_competition_participants (initial_participants : ℕ) : 
  (initial_participants : ℝ) * 0.4 * 0.25 = 30 → initial_participants = 300 := by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l2503_250320


namespace NUMINAMATH_CALUDE_rectangle_tiling_divisibility_l2503_250395

/-- An L-shaped piece made of 4 unit squares -/
structure LPiece :=
  (squares : Fin 4 → (Nat × Nat))

/-- A tiling of an m × n rectangle with L-shaped pieces -/
def Tiling (m n : Nat) := List LPiece

/-- Predicate to check if a tiling is valid for an m × n rectangle -/
def IsValidTiling (t : Tiling m n) : Prop := sorry

theorem rectangle_tiling_divisibility (m n : Nat) (t : Tiling m n) :
  IsValidTiling t → (m * n) % 8 = 0 := by sorry

end NUMINAMATH_CALUDE_rectangle_tiling_divisibility_l2503_250395


namespace NUMINAMATH_CALUDE_local_politics_coverage_l2503_250342

/-- The percentage of reporters covering politics -/
def politics_coverage : ℝ := 100 - 92.85714285714286

/-- The percentage of reporters covering local politics among those covering politics -/
def local_coverage_ratio : ℝ := 100 - 30

theorem local_politics_coverage :
  (local_coverage_ratio * politics_coverage / 100) = 5 := by sorry

end NUMINAMATH_CALUDE_local_politics_coverage_l2503_250342


namespace NUMINAMATH_CALUDE_complex_product_ab_l2503_250348

theorem complex_product_ab (z : ℂ) (a b : ℝ) : 
  z = a + b * Complex.I → 
  z = (4 + 3 * Complex.I) * Complex.I → 
  a * b = -12 := by sorry

end NUMINAMATH_CALUDE_complex_product_ab_l2503_250348


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l2503_250318

theorem consecutive_pages_sum (x y : ℕ) : 
  x + y = 125 → y = x + 1 → y = 63 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l2503_250318


namespace NUMINAMATH_CALUDE_marys_next_birthday_age_l2503_250315

/-- Proves that Mary's age on her next birthday is 11 years, given the conditions of the problem. -/
theorem marys_next_birthday_age :
  ∀ (m s d : ℝ),
  m = 1.3 * s →  -- Mary is 30% older than Sally
  s = 0.75 * d →  -- Sally is 25% younger than Danielle
  m + s + d = 30 →  -- Sum of their ages is 30 years
  ⌈m⌉ = 11  -- Mary's age on her next birthday (ceiling of her current age)
  := by sorry

end NUMINAMATH_CALUDE_marys_next_birthday_age_l2503_250315


namespace NUMINAMATH_CALUDE_part1_part2_l2503_250354

-- Part 1
def U1 : Set ℕ := {2, 3, 4}
def A1 : Set ℕ := {4, 3}
def B1 : Set ℕ := ∅

theorem part1 :
  (U1 \ A1 = {2}) ∧ (U1 \ B1 = U1) := by sorry

-- Part 2
def U2 : Set ℝ := {x | x ≤ 4}
def A2 : Set ℝ := {x | -2 < x ∧ x < 3}
def B2 : Set ℝ := {x | -3 < x ∧ x ≤ 3}

theorem part2 :
  (U2 \ A2 = {x | x ≤ -2 ∨ (3 ≤ x ∧ x ≤ 4)}) ∧
  (A2 ∩ B2 = {x | -2 < x ∧ x < 3}) ∧
  (U2 \ (A2 ∩ B2) = {x | x ≤ -2 ∨ (3 ≤ x ∧ x ≤ 4)}) ∧
  ((U2 \ A2) ∩ B2 = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3}) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2503_250354


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2503_250352

theorem simplify_fraction_product : 8 * (15 / 4) * (-25 / 45) = -50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2503_250352


namespace NUMINAMATH_CALUDE_average_difference_l2503_250334

theorem average_difference (x : ℝ) : (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 8 ↔ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2503_250334


namespace NUMINAMATH_CALUDE_women_per_table_l2503_250304

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 5 →
  men_per_table = 3 →
  total_customers = 40 →
  ∃ (women_per_table : ℕ),
    women_per_table * num_tables + men_per_table * num_tables = total_customers ∧
    women_per_table = 5 :=
by sorry

end NUMINAMATH_CALUDE_women_per_table_l2503_250304


namespace NUMINAMATH_CALUDE_stool_height_is_75cm_l2503_250345

/-- Represents the problem setup for Alice's light bulb replacement task -/
structure LightBulbProblem where
  ceiling_height : ℝ
  bulb_below_ceiling : ℝ
  alice_height : ℝ
  alice_reach : ℝ
  decorative_item_below_ceiling : ℝ

/-- Calculates the required stool height for Alice to reach the light bulb -/
def calculate_stool_height (p : LightBulbProblem) : ℝ :=
  p.ceiling_height - p.bulb_below_ceiling - (p.alice_height + p.alice_reach)

/-- Theorem stating that the stool height Alice needs is 75 cm -/
theorem stool_height_is_75cm (p : LightBulbProblem) 
    (h1 : p.ceiling_height = 300)
    (h2 : p.bulb_below_ceiling = 15)
    (h3 : p.alice_height = 160)
    (h4 : p.alice_reach = 50)
    (h5 : p.decorative_item_below_ceiling = 20) :
    calculate_stool_height p = 75 := by
  sorry

#eval calculate_stool_height {
  ceiling_height := 300,
  bulb_below_ceiling := 15,
  alice_height := 160,
  alice_reach := 50,
  decorative_item_below_ceiling := 20
}

end NUMINAMATH_CALUDE_stool_height_is_75cm_l2503_250345


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l2503_250375

/-- Represents the profit distribution in a partnership --/
structure ProfitDistribution where
  investment_A : ℕ
  investment_B : ℕ
  investment_C : ℕ
  profit_share_C : ℕ

/-- Calculates the total profit given a ProfitDistribution --/
def calculate_total_profit (pd : ProfitDistribution) : ℕ :=
  let total_investment := pd.investment_A + pd.investment_B + pd.investment_C
  let profit_per_unit := pd.profit_share_C * total_investment / pd.investment_C
  profit_per_unit

/-- Theorem stating that given the specific investments and C's profit share, the total profit is 86400 --/
theorem partnership_profit_calculation (pd : ProfitDistribution) 
  (h1 : pd.investment_A = 12000)
  (h2 : pd.investment_B = 16000)
  (h3 : pd.investment_C = 20000)
  (h4 : pd.profit_share_C = 36000) :
  calculate_total_profit pd = 86400 := by
  sorry


end NUMINAMATH_CALUDE_partnership_profit_calculation_l2503_250375


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l2503_250364

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Stratified
  | Systematic

/-- Represents a population with subgroups -/
structure Population where
  total : ℕ
  subgroups : List ℕ

/-- Represents a sampling scenario -/
structure SamplingScenario where
  population : Population
  sample_size : ℕ

/-- Determines the most appropriate sampling method for a given scenario -/
def most_appropriate_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- The student population -/
def student_population : Population :=
  { total := 10000, subgroups := [2000, 4500, 3500] }

/-- The product population -/
def product_population : Population :=
  { total := 1002, subgroups := [1002] }

/-- The student sampling scenario -/
def student_scenario : SamplingScenario :=
  { population := student_population, sample_size := 200 }

/-- The product sampling scenario -/
def product_scenario : SamplingScenario :=
  { population := product_population, sample_size := 20 }

theorem appropriate_sampling_methods :
  (most_appropriate_method student_scenario = SamplingMethod.Stratified) ∧
  (most_appropriate_method product_scenario = SamplingMethod.Systematic) :=
sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l2503_250364


namespace NUMINAMATH_CALUDE_population_increase_l2503_250382

/-- The birth rate in people per two seconds -/
def birth_rate : ℚ := 7

/-- The death rate in people per two seconds -/
def death_rate : ℚ := 1

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The net population increase in one day -/
def net_increase : ℕ := 259200

theorem population_increase :
  (birth_rate - death_rate) / 2 * seconds_per_day = net_increase := by
  sorry

end NUMINAMATH_CALUDE_population_increase_l2503_250382


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2503_250363

/-- Sum of a finite geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- First term of the geometric series -/
def a : ℚ := 2

/-- Common ratio of the geometric series -/
def r : ℚ := 2/5

/-- Number of terms in the series -/
def n : ℕ := 5

theorem geometric_series_sum :
  geometric_sum a r n = 2062/375 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2503_250363


namespace NUMINAMATH_CALUDE_subtract_two_percent_l2503_250332

theorem subtract_two_percent (a : ℝ) : a - (0.02 * a) = 0.98 * a := by
  sorry

end NUMINAMATH_CALUDE_subtract_two_percent_l2503_250332


namespace NUMINAMATH_CALUDE_job_completion_time_l2503_250373

theorem job_completion_time (y : ℝ) 
  (h1 : (1 : ℝ) / (y + 8) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) : y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2503_250373


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l2503_250336

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFraction : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The problem statement -/
theorem ball_bounce_distance :
  totalDistance 150 (3/4) 4 = 765.234375 :=
sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l2503_250336


namespace NUMINAMATH_CALUDE_larger_number_proof_l2503_250341

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 45) (h2 : x - y = 5) (h3 : x ≥ y) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2503_250341


namespace NUMINAMATH_CALUDE_camel_inheritance_theorem_l2503_250379

theorem camel_inheritance_theorem :
  let total_camels : ℕ := 17
  let eldest_share : ℚ := 1/2
  let middle_share : ℚ := 1/3
  let youngest_share : ℚ := 1/9
  eldest_share + middle_share + youngest_share = 17/18 := by
  sorry

end NUMINAMATH_CALUDE_camel_inheritance_theorem_l2503_250379


namespace NUMINAMATH_CALUDE_circle_outside_square_area_l2503_250391

/-- The area inside a circle with radius 1/2 but outside a square with side length 1, 
    when both shapes share the same center, is equal to π/4 - 1. -/
theorem circle_outside_square_area :
  let square_side : ℝ := 1
  let circle_radius : ℝ := 1/2
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  circle_area - square_area = π/4 - 1 := by
sorry

end NUMINAMATH_CALUDE_circle_outside_square_area_l2503_250391


namespace NUMINAMATH_CALUDE_exists_far_reaching_quadrilateral_with_bounded_area_l2503_250308

/-- A point in the 2D plane with integer coordinates. -/
structure Point where
  x : ℤ
  y : ℤ

/-- A rectangle defined by its width and height. -/
structure Rectangle where
  width : ℤ
  height : ℤ

/-- A quadrilateral defined by its four vertices. -/
structure Quadrilateral where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Predicate to check if a point is on or inside a rectangle. -/
def pointInRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

/-- Predicate to check if a quadrilateral is far-reaching in a rectangle. -/
def isFarReaching (q : Quadrilateral) (r : Rectangle) : Prop :=
  (pointInRectangle q.v1 r ∧ pointInRectangle q.v2 r ∧ pointInRectangle q.v3 r ∧ pointInRectangle q.v4 r) ∧
  (q.v1.x = 0 ∨ q.v2.x = 0 ∨ q.v3.x = 0 ∨ q.v4.x = 0) ∧
  (q.v1.y = 0 ∨ q.v2.y = 0 ∨ q.v3.y = 0 ∨ q.v4.y = 0) ∧
  (q.v1.x = r.width ∨ q.v2.x = r.width ∨ q.v3.x = r.width ∨ q.v4.x = r.width) ∧
  (q.v1.y = r.height ∨ q.v2.y = r.height ∨ q.v3.y = r.height ∨ q.v4.y = r.height)

/-- Calculate the area of a quadrilateral. -/
def quadrilateralArea (q : Quadrilateral) : ℚ :=
  sorry  -- The actual area calculation would go here

/-- The main theorem to be proved. -/
theorem exists_far_reaching_quadrilateral_with_bounded_area
  (n m : ℕ) (hn : n ≤ 10^10) (hm : m ≤ 10^10) :
  ∃ (q : Quadrilateral), isFarReaching q (Rectangle.mk n m) ∧ quadrilateralArea q ≤ 10^6 := by
  sorry

end NUMINAMATH_CALUDE_exists_far_reaching_quadrilateral_with_bounded_area_l2503_250308


namespace NUMINAMATH_CALUDE_collection_forms_set_iff_well_defined_l2503_250325

-- Define a type for collections of elements
def Collection := Type

-- Define a property for well-defined elements
def HasWellDefinedElements (c : Collection) : Prop := sorry

-- Define a property for forming a set
def CanFormSet (c : Collection) : Prop := sorry

-- Theorem: A collection can form a set if and only if its elements are well-defined
theorem collection_forms_set_iff_well_defined (c : Collection) :
  CanFormSet c ↔ HasWellDefinedElements c := by sorry

end NUMINAMATH_CALUDE_collection_forms_set_iff_well_defined_l2503_250325


namespace NUMINAMATH_CALUDE_sum_of_digits_2010_5012_6_l2503_250356

def digit_sum (n : ℕ) : ℕ := sorry

theorem sum_of_digits_2010_5012_6 :
  digit_sum (2^2010 * 5^2012 * 6) = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_2010_5012_6_l2503_250356


namespace NUMINAMATH_CALUDE_fraction_meaningful_range_l2503_250371

theorem fraction_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y = 2 / (x + 3)) → x ≠ -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_range_l2503_250371


namespace NUMINAMATH_CALUDE_cube_with_holes_volume_l2503_250311

/-- The volume of a cube with holes drilled through it -/
theorem cube_with_holes_volume :
  let cube_edge : ℝ := 3
  let hole_side : ℝ := 1
  let cube_volume := cube_edge ^ 3
  let hole_volume := hole_side ^ 2 * cube_edge
  let num_hole_pairs := 3
  cube_volume - (num_hole_pairs * hole_volume) = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_volume_l2503_250311


namespace NUMINAMATH_CALUDE_special_square_numbers_l2503_250368

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A function that returns the first two digits of a six-digit number -/
def first_two_digits (n : ℕ) : ℕ :=
  n / 10000

/-- A function that returns the middle two digits of a six-digit number -/
def middle_two_digits (n : ℕ) : ℕ :=
  (n / 100) % 100

/-- A function that returns the last two digits of a six-digit number -/
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

/-- A function that checks if all digits of a six-digit number are non-zero -/
def all_digits_nonzero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [n / 100000, (n / 10000) % 10, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10] → d ≠ 0

/-- The main theorem stating that there are exactly 2 special square numbers -/
theorem special_square_numbers :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, 100000 ≤ n ∧ n < 1000000 ∧
              all_digits_nonzero n ∧
              is_perfect_square n ∧
              is_perfect_square (first_two_digits n) ∧
              is_perfect_square (middle_two_digits n) ∧
              is_perfect_square (last_two_digits n)) ∧
    s.card = 2 := by
  sorry


end NUMINAMATH_CALUDE_special_square_numbers_l2503_250368


namespace NUMINAMATH_CALUDE_three_digit_one_more_than_multiple_l2503_250351

/-- The least common multiple of 2, 3, 5, and 7 -/
def lcm_2357 : ℕ := 210

/-- Checks if a number is a three-digit positive integer -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Checks if a number is one more than a multiple of 2, 3, 5, and 7 -/
def is_one_more_than_multiple (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * lcm_2357 + 1

theorem three_digit_one_more_than_multiple :
  ∀ n : ℕ, is_three_digit n ∧ is_one_more_than_multiple n ↔ n = 211 ∨ n = 421 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_one_more_than_multiple_l2503_250351


namespace NUMINAMATH_CALUDE_divisibility_property_l2503_250321

theorem divisibility_property (a b c : ℤ) (h : a + b + c = 0) :
  (∃ k : ℤ, a^4 + b^4 + c^4 = k * (a^2 + b^2 + c^2)) ∧
  (∃ m : ℤ, a^100 + b^100 + c^100 = m * (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2503_250321


namespace NUMINAMATH_CALUDE_total_marks_is_660_l2503_250374

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ
  history : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science + scores.history

/-- Eva's scores for the second semester -/
def secondSemester : SemesterScores :=
  { maths := 80, arts := 90, science := 90, history := 85 }

/-- Eva's scores for the first semester -/
def firstSemester : SemesterScores :=
  { maths := secondSemester.maths + 10,
    arts := secondSemester.arts - 15,
    science := secondSemester.science - (secondSemester.science / 3),
    history := secondSemester.history + 5 }

/-- Theorem: The total number of marks in all semesters is 660 -/
theorem total_marks_is_660 :
  totalScore firstSemester + totalScore secondSemester = 660 := by
  sorry


end NUMINAMATH_CALUDE_total_marks_is_660_l2503_250374


namespace NUMINAMATH_CALUDE_susan_bob_cat_difference_l2503_250397

/-- The number of cats Susan has initially -/
def susan_initial_cats : ℕ := 21

/-- The number of cats Bob has -/
def bob_cats : ℕ := 3

/-- The number of cats Susan gives to Robert -/
def cats_given_to_robert : ℕ := 4

/-- Theorem stating the difference between Susan's remaining cats and Bob's cats -/
theorem susan_bob_cat_difference : 
  susan_initial_cats - cats_given_to_robert - bob_cats = 14 := by
  sorry

end NUMINAMATH_CALUDE_susan_bob_cat_difference_l2503_250397


namespace NUMINAMATH_CALUDE_minimum_raft_capacity_l2503_250302

/-- Represents an animal with a specific weight -/
structure Animal where
  weight : ℕ

/-- Represents the raft with a weight capacity -/
structure Raft where
  capacity : ℕ

/-- Checks if a raft can carry at least two mice -/
def canCarryTwoMice (r : Raft) (mouseWeight : ℕ) : Prop :=
  r.capacity ≥ 2 * mouseWeight

/-- Checks if all animals can be transported given a raft capacity -/
def canTransportAll (r : Raft) (mice moles hamsters : List Animal) : Prop :=
  (mice ++ moles ++ hamsters).all (fun a => a.weight ≤ r.capacity)

theorem minimum_raft_capacity
  (mice : List Animal)
  (moles : List Animal)
  (hamsters : List Animal)
  (h_mice_count : mice.length = 5)
  (h_moles_count : moles.length = 3)
  (h_hamsters_count : hamsters.length = 4)
  (h_mice_weight : ∀ m ∈ mice, m.weight = 70)
  (h_moles_weight : ∀ m ∈ moles, m.weight = 90)
  (h_hamsters_weight : ∀ h ∈ hamsters, h.weight = 120)
  : ∃ (r : Raft), r.capacity = 140 ∧ 
    canCarryTwoMice r 70 ∧
    canTransportAll r mice moles hamsters :=
  sorry

#check minimum_raft_capacity

end NUMINAMATH_CALUDE_minimum_raft_capacity_l2503_250302


namespace NUMINAMATH_CALUDE_equality_for_all_n_l2503_250324

theorem equality_for_all_n (x y a b : ℝ) 
  (h1 : x + y = a + b) 
  (h2 : x^2 + y^2 = a^2 + b^2) : 
  ∀ n : ℤ, x^n + y^n = a^n + b^n := by sorry

end NUMINAMATH_CALUDE_equality_for_all_n_l2503_250324


namespace NUMINAMATH_CALUDE_expression_upper_bound_l2503_250331

theorem expression_upper_bound (α β γ δ ε : ℝ) : 
  (1 - α) * Real.exp α + 
  (1 - β) * Real.exp (α + β) + 
  (1 - γ) * Real.exp (α + β + γ) + 
  (1 - δ) * Real.exp (α + β + γ + δ) + 
  (1 - ε) * Real.exp (α + β + γ + δ + ε) ≤ Real.exp 4 := by
  sorry

#check expression_upper_bound

end NUMINAMATH_CALUDE_expression_upper_bound_l2503_250331


namespace NUMINAMATH_CALUDE_arithmetic_sequence_convex_condition_l2503_250362

/-- A sequence a is convex if a(n+1) + a(n-1) ≤ 2*a(n) for all n ≥ 2 -/
def IsConvexSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) + a (n - 1) ≤ 2 * a n

/-- The nth term of an arithmetic sequence with first term b₁ and common difference d -/
def ArithmeticSequence (b₁ d : ℝ) (n : ℕ) : ℝ :=
  b₁ + (n - 1) * d

theorem arithmetic_sequence_convex_condition (d : ℝ) :
  let b := ArithmeticSequence 2 (Real.log d)
  IsConvexSequence (fun n => b n / n) → d ≥ Real.exp 2 := by
  sorry

#check arithmetic_sequence_convex_condition

end NUMINAMATH_CALUDE_arithmetic_sequence_convex_condition_l2503_250362


namespace NUMINAMATH_CALUDE_circle_equation_satisfies_conditions_l2503_250353

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def PointOnLine (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

theorem circle_equation_satisfies_conditions :
  ∃ (h k r : ℝ),
    -- The circle's equation
    (∀ x y, CircleEquation h k r x y ↔ (x - 2)^2 + (y + 1)^2 = 5) ∧
    -- The center lies on the line 3x + y - 5 = 0
    PointOnLine 3 1 (-5) h k ∧
    -- The circle passes through (0, 0)
    CircleEquation h k r 0 0 ∧
    -- The circle passes through (4, 0)
    CircleEquation h k r 4 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_satisfies_conditions_l2503_250353


namespace NUMINAMATH_CALUDE_age_difference_l2503_250372

theorem age_difference (matt_age john_age : ℕ) 
  (h1 : matt_age + john_age = 52)
  (h2 : ∃ k : ℕ, matt_age + k = 4 * john_age) : 
  4 * john_age - matt_age = 3 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2503_250372


namespace NUMINAMATH_CALUDE_factorial_ratio_l2503_250392

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2503_250392


namespace NUMINAMATH_CALUDE_problem_statement_l2503_250300

theorem problem_statement :
  (∃ x : ℝ, x^2 + 1 ≤ 2*x) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt ((x^2 + y^2)/2) ≥ (2*x*y)/(x + y)) ∧
  ¬(∀ x : ℝ, x ≠ 0 → x + 1/x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2503_250300


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l2503_250384

theorem smallest_five_digit_divisible_by_53 : ∀ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) → -- five-digit number condition
  n % 53 = 0 → -- divisibility by 53 condition
  n ≥ 10017 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l2503_250384


namespace NUMINAMATH_CALUDE_min_digit_divisible_by_72_l2503_250350

theorem min_digit_divisible_by_72 :
  ∃ (x : ℕ), x < 10 ∧ (983480 + x) % 72 = 0 ∧
  ∀ (y : ℕ), y < x → (983480 + y) % 72 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_digit_divisible_by_72_l2503_250350


namespace NUMINAMATH_CALUDE_three_classes_five_spots_l2503_250316

/-- The number of ways for classes to choose scenic spots -/
def num_selection_methods (num_classes : ℕ) (num_spots : ℕ) : ℕ :=
  num_spots ^ num_classes

/-- Theorem: Three classes choosing from five scenic spots results in 5^3 selection methods -/
theorem three_classes_five_spots : num_selection_methods 3 5 = 5^3 := by
  sorry

end NUMINAMATH_CALUDE_three_classes_five_spots_l2503_250316


namespace NUMINAMATH_CALUDE_coefficient_of_x_six_in_expansion_l2503_250326

theorem coefficient_of_x_six_in_expansion (x : ℝ) : 
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ), 
    (2*x^2 + 1)^5 = a₀ + a₁*x^2 + a₂*x^4 + a₃*x^6 + a₄*x^8 + a₅*x^10 ∧ 
    a₃ = 80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_six_in_expansion_l2503_250326


namespace NUMINAMATH_CALUDE_ellipse_equation_l2503_250387

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    if its eccentricity is √3/2 and the distance from one endpoint of
    the minor axis to the right focus is 2, then its equation is x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := (Real.sqrt 3) / 2
  let d := 2
  (e = Real.sqrt (1 - b^2 / a^2) ∧ d = a) →
  a^2 = 4 ∧ b^2 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2503_250387


namespace NUMINAMATH_CALUDE_line_symmetry_l2503_250301

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Symmetry condition for two lines with respect to y = x -/
def symmetric_about_y_eq_x (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = 1 ∧ l1.intercept + l2.intercept = 0

theorem line_symmetry (a b : ℝ) :
  let l1 : Line := ⟨a, 2⟩
  let l2 : Line := ⟨3, -b⟩
  symmetric_about_y_eq_x l1 l2 → a = 1/3 ∧ b = 6 := by
  sorry

#check line_symmetry

end NUMINAMATH_CALUDE_line_symmetry_l2503_250301


namespace NUMINAMATH_CALUDE_triangle_properties_l2503_250313

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  sum_angles : A + B + C = π

/-- The vector m in the problem -/
def m (t : Triangle) : ℝ × ℝ := (t.a + t.c, t.b - t.a)

/-- The vector n in the problem -/
def n (t : Triangle) : ℝ × ℝ := (t.a - t.c, t.b)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem triangle_properties (t : Triangle) 
  (h_perp : dot_product (m t) (n t) = 0)
  (h_sin : 2 * Real.sin (t.A / 2) ^ 2 + 2 * Real.sin (t.B / 2) ^ 2 = 1) :
  t.C = π / 3 ∧ t.A = π / 3 ∧ t.B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2503_250313


namespace NUMINAMATH_CALUDE_amount_after_two_years_l2503_250383

/-- The final amount after compound interest --/
def final_amount (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate) ^ years

/-- The problem statement --/
theorem amount_after_two_years :
  let initial := 2880
  let rate := 1 / 8
  let years := 2
  final_amount initial rate years = 3645 := by
sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l2503_250383


namespace NUMINAMATH_CALUDE_abs_ratio_sqrt_five_halves_l2503_250317

theorem abs_ratio_sqrt_five_halves (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^2 = 18*a*b) : 
  |((a+b)/(a-b))| = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_abs_ratio_sqrt_five_halves_l2503_250317


namespace NUMINAMATH_CALUDE_age_difference_l2503_250307

theorem age_difference (x y z : ℕ) : 
  x + y = y + z + 18 → (x - z : ℚ) / 10 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2503_250307


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l2503_250369

theorem floor_ceiling_sum : ⌊(3.67 : ℝ)⌋ + ⌈(-14.2 : ℝ)⌉ = -11 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l2503_250369


namespace NUMINAMATH_CALUDE_value_after_seven_years_l2503_250385

/-- Calculates the value after n years given initial value, annual increase rate, inflation rate, and tax rate -/
def value_after_years (initial_value : ℝ) (increase_rate : ℝ) (inflation_rate : ℝ) (tax_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * ((1 - tax_rate) * (1 - inflation_rate) * (1 + increase_rate)) ^ years

/-- Theorem stating that the value after 7 years is approximately 126469.75 -/
theorem value_after_seven_years :
  let initial_value : ℝ := 59000
  let increase_rate : ℝ := 1/8
  let inflation_rate : ℝ := 0.03
  let tax_rate : ℝ := 0.07
  let years : ℕ := 7
  abs (value_after_years initial_value increase_rate inflation_rate tax_rate years - 126469.75) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_value_after_seven_years_l2503_250385


namespace NUMINAMATH_CALUDE_boys_girls_points_not_equal_l2503_250346

/-- Represents a round-robin chess tournament with boys and girls -/
structure ChessTournament where
  num_boys : Nat
  num_girls : Nat

/-- Calculate the total number of games in a round-robin tournament -/
def total_games (t : ChessTournament) : Nat :=
  (t.num_boys + t.num_girls) * (t.num_boys + t.num_girls - 1) / 2

/-- Calculate the number of games between boys -/
def boys_games (t : ChessTournament) : Nat :=
  t.num_boys * (t.num_boys - 1) / 2

/-- Calculate the number of games between girls -/
def girls_games (t : ChessTournament) : Nat :=
  t.num_girls * (t.num_girls - 1) / 2

/-- Calculate the number of games between boys and girls -/
def mixed_games (t : ChessTournament) : Nat :=
  t.num_boys * t.num_girls

/-- Theorem: In a round-robin chess tournament with 9 boys and 3 girls,
    the total points scored by all boys cannot equal the total points scored by all girls -/
theorem boys_girls_points_not_equal (t : ChessTournament) 
        (h1 : t.num_boys = 9) 
        (h2 : t.num_girls = 3) : 
        ¬ (boys_games t + mixed_games t / 2 = girls_games t + mixed_games t / 2) := by
  sorry

#eval boys_games ⟨9, 3⟩
#eval girls_games ⟨9, 3⟩
#eval mixed_games ⟨9, 3⟩

end NUMINAMATH_CALUDE_boys_girls_points_not_equal_l2503_250346


namespace NUMINAMATH_CALUDE_gas_needed_is_eighteen_l2503_250361

/-- Calculates the total amount of gas needed to fill both a truck and car tank completely. -/
def total_gas_needed (truck_capacity car_capacity : ℚ) (truck_fullness car_fullness : ℚ) : ℚ :=
  (truck_capacity - truck_capacity * truck_fullness) + (car_capacity - car_capacity * car_fullness)

/-- Proves that the total amount of gas needed to fill both tanks is 18 gallons. -/
theorem gas_needed_is_eighteen :
  total_gas_needed 20 12 (1/2) (1/3) = 18 := by
  sorry

end NUMINAMATH_CALUDE_gas_needed_is_eighteen_l2503_250361


namespace NUMINAMATH_CALUDE_total_monthly_payment_l2503_250344

/-- Calculates the total monthly payment for employees after new hires --/
theorem total_monthly_payment
  (initial_employees : ℕ)
  (hourly_rate : ℚ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (additional_hires : ℕ)
  (h1 : initial_employees = 500)
  (h2 : hourly_rate = 12)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : additional_hires = 200) :
  (initial_employees + additional_hires) *
  (hourly_rate * hours_per_day * days_per_week * weeks_per_month) = 1680000 := by
  sorry

#eval (500 + 200) * (12 * 10 * 5 * 4)

end NUMINAMATH_CALUDE_total_monthly_payment_l2503_250344


namespace NUMINAMATH_CALUDE_marys_double_counted_sheep_l2503_250306

/-- Given Mary's animal counting problem, prove that she double-counted 7 sheep. -/
theorem marys_double_counted_sheep :
  let marys_count : ℕ := 60
  let actual_animals : ℕ := 56
  let forgotten_pigs : ℕ := 3
  let double_counted_sheep : ℕ := marys_count - actual_animals + forgotten_pigs
  double_counted_sheep = 7 := by sorry

end NUMINAMATH_CALUDE_marys_double_counted_sheep_l2503_250306


namespace NUMINAMATH_CALUDE_total_amount_correct_l2503_250323

/-- The rate for painting fences in dollars per meter -/
def painting_rate : ℚ := 0.20

/-- The number of fences to be painted -/
def number_of_fences : ℕ := 50

/-- The length of each fence in meters -/
def fence_length : ℕ := 500

/-- The total amount earned from painting all fences -/
def total_amount : ℚ := 5000

/-- Theorem stating that the total amount earned is correct given the conditions -/
theorem total_amount_correct : 
  painting_rate * (number_of_fences * fence_length : ℚ) = total_amount := by
  sorry

end NUMINAMATH_CALUDE_total_amount_correct_l2503_250323


namespace NUMINAMATH_CALUDE_dog_bunny_ratio_l2503_250393

/-- Given a total of 375 dogs and bunnies, with 75 dogs, prove that the ratio of dogs to bunnies is 1:4 -/
theorem dog_bunny_ratio (total : ℕ) (dogs : ℕ) (h1 : total = 375) (h2 : dogs = 75) :
  (dogs : ℚ) / (total - dogs : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_dog_bunny_ratio_l2503_250393


namespace NUMINAMATH_CALUDE_circle_inequality_l2503_250378

theorem circle_inequality (r s d : ℝ) (h1 : r > s) (h2 : r > 0) (h3 : s > 0) (h4 : d > 0) :
  r - s ≤ d :=
sorry

end NUMINAMATH_CALUDE_circle_inequality_l2503_250378


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2503_250330

theorem y_in_terms_of_x (m : ℕ) (x y : ℝ) 
  (hx : x = 2^m + 1) 
  (hy : y = 3 + 2^(m+1)) : 
  y = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2503_250330


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2503_250381

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

theorem f_max_min_on_interval :
  let a : ℝ := -3
  let b : ℝ := 3
  ∃ (x_max x_min : ℝ), a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    f x_max = 59 ∧ f x_min = -49 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2503_250381
