import Mathlib

namespace binary_addition_proof_l3383_338338

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then
    []
  else
    (n % 2 = 1) :: nat_to_binary (n / 2)

theorem binary_addition_proof :
  let a := [false, true, false, true]  -- 1010₂
  let b := [false, true]               -- 10₂
  let sum := [false, false, true, true] -- 1100₂
  binary_to_nat a + binary_to_nat b = binary_to_nat sum := by
  sorry

end binary_addition_proof_l3383_338338


namespace boat_speed_in_still_water_l3383_338305

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (travel_time_minutes : ℝ) :
  current_speed = 7 →
  downstream_distance = 35.93 →
  travel_time_minutes = 44 →
  ∃ (v : ℝ), abs (v - 42) < 0.01 ∧ downstream_distance = (v + current_speed) * (travel_time_minutes / 60) :=
by sorry

end boat_speed_in_still_water_l3383_338305


namespace ferry_tourists_sum_l3383_338304

/-- The number of trips the ferry makes in a day -/
def num_trips : ℕ := 9

/-- The initial number of tourists on the first trip -/
def initial_tourists : ℕ := 120

/-- The decrease in number of tourists for each subsequent trip -/
def tourist_decrease : ℤ := 2

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℕ) * d)

theorem ferry_tourists_sum :
  arithmetic_sum initial_tourists (-tourist_decrease) num_trips = 1008 := by
  sorry

end ferry_tourists_sum_l3383_338304


namespace white_bread_cost_l3383_338336

/-- Represents the cost of bread items in dollars -/
structure BreadCosts where
  white : ℝ
  baguette : ℝ := 1.50
  sourdough : ℝ := 4.50
  croissant : ℝ := 2.00

/-- Represents the weekly purchase of bread items -/
structure WeeklyPurchase where
  white : ℕ := 2
  baguette : ℕ := 1
  sourdough : ℕ := 2
  croissant : ℕ := 1

def total_spent_over_4_weeks : ℝ := 78

/-- Calculates the weekly cost of non-white bread items -/
def weekly_non_white_cost (costs : BreadCosts) (purchase : WeeklyPurchase) : ℝ :=
  costs.baguette * purchase.baguette + 
  costs.sourdough * purchase.sourdough + 
  costs.croissant * purchase.croissant

/-- Theorem stating that the cost of each loaf of white bread is $3.50 -/
theorem white_bread_cost (costs : BreadCosts) (purchase : WeeklyPurchase) :
  costs.white = 3.50 ↔ 
  total_spent_over_4_weeks = 
    4 * (weekly_non_white_cost costs purchase + costs.white * purchase.white) :=
sorry

end white_bread_cost_l3383_338336


namespace right_triangle_angle_identity_l3383_338316

theorem right_triangle_angle_identity (α β γ : Real) 
  (h_right_triangle : α + β + γ = π)
  (h_right_angle : α = π/2 ∨ β = π/2 ∨ γ = π/2) : 
  Real.sin α * Real.sin β * Real.sin (α - β) + 
  Real.sin β * Real.sin γ * Real.sin (β - γ) + 
  Real.sin γ * Real.sin α * Real.sin (γ - α) + 
  Real.sin (α - β) * Real.sin (β - γ) * Real.sin (γ - α) = 0 := by
  sorry

end right_triangle_angle_identity_l3383_338316


namespace secretary_discussions_l3383_338378

/-- Represents the number of emails sent in a small discussion -/
def small_discussion_emails : ℕ := 7 * 6

/-- Represents the number of emails sent in a large discussion -/
def large_discussion_emails : ℕ := 15 * 14

/-- Represents the total number of emails sent excluding the secretary's -/
def total_emails : ℕ := 1994

/-- Represents the maximum number of discussions a jury member can participate in -/
def max_discussions : ℕ := 10

theorem secretary_discussions (m b : ℕ) :
  m + b ≤ max_discussions →
  small_discussion_emails * m + large_discussion_emails * b + 6 * m + 14 * b = total_emails →
  m = 6 ∧ b = 2 := by
  sorry

#check secretary_discussions

end secretary_discussions_l3383_338378


namespace airplane_seats_l3383_338391

/-- Given an airplane with a total of 387 seats, where the number of coach class seats
    is 2 more than 4 times the number of first-class seats, prove that there are
    77 first-class seats. -/
theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) (coach : ℕ)
    (h1 : total_seats = 387)
    (h2 : coach = 4 * first_class + 2)
    (h3 : total_seats = first_class + coach) :
    first_class = 77 := by
  sorry

end airplane_seats_l3383_338391


namespace delta_value_l3383_338345

theorem delta_value (Δ : ℤ) : 4 * (-3) = Δ + 5 → Δ = -17 := by
  sorry

end delta_value_l3383_338345


namespace vivian_yogurt_count_l3383_338308

/-- The number of banana slices per yogurt -/
def slices_per_yogurt : ℕ := 8

/-- The number of slices one banana yields -/
def slices_per_banana : ℕ := 10

/-- The number of bananas Vivian needs to buy -/
def bananas_to_buy : ℕ := 4

/-- The number of yogurts Vivian needs to make -/
def yogurts_to_make : ℕ := (bananas_to_buy * slices_per_banana) / slices_per_yogurt

theorem vivian_yogurt_count : yogurts_to_make = 5 := by
  sorry

end vivian_yogurt_count_l3383_338308


namespace city_population_ratio_l3383_338344

/-- Given the population relationships between cities X, Y, and Z, 
    prove that the ratio of City X's population to City Z's population is 6:1 -/
theorem city_population_ratio 
  (Z : ℕ) -- Population of City Z
  (Y : ℕ) -- Population of City Y
  (X : ℕ) -- Population of City X
  (h1 : Y = 2 * Z) -- City Y's population is twice City Z's
  (h2 : ∃ k : ℕ, X = k * Y) -- City X's population is some multiple of City Y's
  (h3 : X = 6 * Z) -- The ratio of City X's to City Z's population is 6
  : X / Z = 6 := by
  sorry

end city_population_ratio_l3383_338344


namespace business_profit_calculation_l3383_338356

/-- Represents the total profit of a business partnership --/
def total_profit (a_investment b_investment : ℕ) (a_management_fee : ℚ) (a_total_received : ℕ) : ℚ :=
  let total_investment := a_investment + b_investment
  let remaining_profit_share := 1 - a_management_fee
  let a_profit_share := (a_investment : ℚ) / (total_investment : ℚ) * remaining_profit_share
  (a_total_received : ℚ) / (a_management_fee + a_profit_share)

/-- Theorem stating the total profit of the business partnership --/
theorem business_profit_calculation :
  total_profit 3500 2500 (1/10) 6000 = 9600 := by
  sorry

#eval total_profit 3500 2500 (1/10) 6000

end business_profit_calculation_l3383_338356


namespace intersection_y_diff_zero_l3383_338368

def f (x : ℝ) : ℝ := 2 - x^2 + x^4
def g (x : ℝ) : ℝ := -1 + x^2 + x^4

theorem intersection_y_diff_zero :
  let intersection_points := {x : ℝ | f x = g x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧
    ∀ (y₁ y₂ : ℝ), y₁ = f x₁ ∧ y₂ = f x₂ → |y₁ - y₂| = 0 :=
sorry

end intersection_y_diff_zero_l3383_338368


namespace inequality_system_solution_l3383_338339

/-- Proves that the solution set of the given inequality system is (-2, 1]. -/
theorem inequality_system_solution :
  ∀ x : ℝ, (x > -6 - 2*x ∧ x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end inequality_system_solution_l3383_338339


namespace min_value_sum_squares_l3383_338331

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + 2*y + z + 8 = 0) :
  ∃ (m : ℝ), m = 9 ∧ ∀ (x' y' z' : ℝ), 2*x' + 2*y' + z' + 8 = 0 →
    (x' - 1)^2 + (y' + 2)^2 + (z' - 3)^2 ≥ m :=
by sorry

end min_value_sum_squares_l3383_338331


namespace inequality_system_solution_range_l3383_338367

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x - a ≥ 0 ∧ 2*x < 4))) → 
  (-2 < a ∧ a ≤ -1) :=
by sorry

end inequality_system_solution_range_l3383_338367


namespace stratified_sampling_theorem_l3383_338396

/-- Represents the number of employees in each title category -/
structure TitleCount where
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Represents the result of a stratified sampling -/
structure SampleResult where
  senior : Nat
  intermediate : Nat
  junior : Nat

def total_employees : Nat := 150
def sample_size : Nat := 30

def population : TitleCount := {
  senior := 45,
  intermediate := 90,
  junior := 15
}

def stratified_sample (pop : TitleCount) (total : Nat) (sample : Nat) : SampleResult :=
  { senior := sample * pop.senior / total,
    intermediate := sample * pop.intermediate / total,
    junior := sample * pop.junior / total }

theorem stratified_sampling_theorem :
  stratified_sample population total_employees sample_size =
  { senior := 9, intermediate := 18, junior := 3 } := by
  sorry

end stratified_sampling_theorem_l3383_338396


namespace complement_A_intersect_B_l3383_338322

-- Define the universal set U
def U : Set ℤ := {x | 0 < x ∧ x < 5}

-- Define sets A and B
def A : Set ℤ := {1, 2}
def B : Set ℤ := {2, 3}

-- State the theorem
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {3} := by sorry

end complement_A_intersect_B_l3383_338322


namespace random_function_iff_stochastic_process_l3383_338365

open MeasureTheory ProbabilityTheory

/-- A random function X = (X_t)_{t ∈ T} taking values in (ℝ^T, ℬ(ℝ^T)) -/
def RandomFunction (T : Type) (Ω : Type) [MeasurableSpace Ω] : Type :=
  Ω → (T → ℝ)

/-- A stochastic process (collection of random variables X_t) -/
def StochasticProcess (T : Type) (Ω : Type) [MeasurableSpace Ω] : Type :=
  T → (Ω → ℝ)

/-- Theorem stating the equivalence between random functions and stochastic processes -/
theorem random_function_iff_stochastic_process (T : Type) (Ω : Type) [MeasurableSpace Ω] :
  (∃ X : RandomFunction T Ω, Measurable X) ↔ (∃ Y : StochasticProcess T Ω, ∀ t, Measurable (Y t)) :=
sorry


end random_function_iff_stochastic_process_l3383_338365


namespace eduardo_flour_amount_l3383_338334

/-- Represents the number of cookies in the original recipe -/
def original_cookies : ℕ := 30

/-- Represents the amount of flour (in cups) needed for the original recipe -/
def original_flour : ℕ := 2

/-- Represents the number of cookies Eduardo wants to bake -/
def eduardo_cookies : ℕ := 90

/-- Calculates the amount of flour needed for a given number of cookies -/
def flour_needed (cookies : ℕ) : ℕ :=
  (cookies * original_flour) / original_cookies

theorem eduardo_flour_amount : flour_needed eduardo_cookies = 6 := by
  sorry

end eduardo_flour_amount_l3383_338334


namespace intersection_M_P_union_M_P_condition_l3383_338319

-- Define the sets M and P
def M (m : ℝ) : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4*m - 2}
def P : Set ℝ := {x : ℝ | x > 2 ∨ x ≤ 1}

-- Theorem 1: Intersection of M and P when m = 2
theorem intersection_M_P : 
  M 2 ∩ P = {x : ℝ | (-1 ≤ x ∧ x ≤ 1) ∨ (2 < x ∧ x ≤ 6)} := by sorry

-- Theorem 2: Union of M and P is ℝ iff m ≥ 1
theorem union_M_P_condition (m : ℝ) : 
  M m ∪ P = Set.univ ↔ m ≥ 1 := by sorry

end intersection_M_P_union_M_P_condition_l3383_338319


namespace solve_sticker_price_l3383_338360

def sticker_price_problem (p : ℝ) : Prop :=
  let store_a_price := 1.08 * (0.8 * p - 120)
  let store_b_price := 1.08 * (0.7 * p + 50)
  store_b_price - store_a_price = 27 ∧ p = 1450

theorem solve_sticker_price : ∃ p : ℝ, sticker_price_problem p := by
  sorry

end solve_sticker_price_l3383_338360


namespace ball_probability_theorem_l3383_338398

/-- Given a bag with m red balls and n white balls, where m ≥ n ≥ 2, prove that if the probability
    of drawing two red balls is an integer multiple of the probability of drawing one red and one
    white ball, then m must be odd. Also, find all pairs (m, n) such that m + n ≤ 40 and the
    probability of drawing two balls of the same color equals the probability of drawing two balls
    of different colors. -/
theorem ball_probability_theorem (m n : ℕ) (h1 : m ≥ n) (h2 : n ≥ 2) :
  (∃ k : ℕ, Nat.choose m 2 * (Nat.choose (m + n) 2) = k * m * n * (Nat.choose (m + n) 2)) →
  Odd m ∧
  (m + n ≤ 40 →
    Nat.choose m 2 + Nat.choose n 2 = m * n →
    ∃ (p q : ℕ), p = m ∧ q = n) :=
by sorry

end ball_probability_theorem_l3383_338398


namespace square_roots_ratio_l3383_338312

-- Define the complex polynomial z^2 + az + b
def complex_polynomial (a b z : ℂ) : ℂ := z^2 + a*z + b

-- Define the theorem
theorem square_roots_ratio (a b z₁ : ℂ) :
  (complex_polynomial a b z₁ = 0) →
  (complex_polynomial a b (Complex.I * z₁) = 0) →
  a^2 / b = 2 := by
  sorry

end square_roots_ratio_l3383_338312


namespace convergence_and_bound_l3383_338311

def u : ℕ → ℚ
  | 0 => 1/6
  | n + 1 => 2 * u n - 2 * (u n)^2 + 1/3

def L : ℚ := 5/6

theorem convergence_and_bound :
  (∃ (k : ℕ), ∀ (n : ℕ), n ≥ k → |u n - L| ≤ 1 / 2^500) ∧
  (∀ (k : ℕ), k < 9 → ∃ (n : ℕ), n ≥ k ∧ |u n - L| > 1 / 2^500) ∧
  (∀ (n : ℕ), n ≥ 9 → |u n - L| ≤ 1 / 2^500) :=
sorry

end convergence_and_bound_l3383_338311


namespace modular_power_congruence_l3383_338303

theorem modular_power_congruence (p : ℕ) (n : ℕ) (a b : ℤ) 
  (h_prime : Nat.Prime p) (h_cong : a ≡ b [ZMOD p^n]) :
  a^p ≡ b^p [ZMOD p^(n+1)] := by sorry

end modular_power_congruence_l3383_338303


namespace weekly_jog_distance_l3383_338355

/-- The total distance jogged throughout the week in kilometers -/
def total_distance (mon tue wed thu fri_miles : ℝ) (mile_to_km : ℝ) : ℝ :=
  mon + tue + wed + thu + (fri_miles * mile_to_km)

/-- Theorem stating the total distance jogged throughout the week -/
theorem weekly_jog_distance :
  let mon := 3
  let tue := 5.5
  let wed := 9.7
  let thu := 10.8
  let fri_miles := 2
  let mile_to_km := 1.60934
  total_distance mon tue wed thu fri_miles mile_to_km = 32.21868 := by
  sorry

end weekly_jog_distance_l3383_338355


namespace fraction_value_l3383_338395

theorem fraction_value (a b c : ℚ) (h1 : a = 5) (h2 : b = -3) (h3 : c = 2) :
  3 * c / (a + b) = 3 := by
  sorry

end fraction_value_l3383_338395


namespace twenty_four_is_seventy_five_percent_of_thirty_two_l3383_338348

theorem twenty_four_is_seventy_five_percent_of_thirty_two (x : ℝ) :
  24 / x = 75 / 100 → x = 32 := by
  sorry

end twenty_four_is_seventy_five_percent_of_thirty_two_l3383_338348


namespace sphere_volume_from_surface_area_l3383_338349

/-- Given a sphere with surface area 400π cm², prove its volume is (4000/3)π cm³ -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
  (4 * π * r^2 = 400 * π) →  -- Surface area formula
  ((4 / 3) * π * r^3 = (4000 / 3) * π) -- Volume formula
  := by sorry

end sphere_volume_from_surface_area_l3383_338349


namespace max_A_value_l3383_338342

/-- Represents the board configuration after chip removal operations -/
structure BoardConfig where
  white_columns : Nat
  white_rows : Nat
  black_columns : Nat
  black_rows : Nat

/-- Calculates the number of remaining chips for a given color -/
def remaining_chips (config : BoardConfig) (color : Bool) : Nat :=
  if color then config.white_columns * config.white_rows
  else config.black_columns * config.black_rows

/-- The size of the board -/
def board_size : Nat := 2018

/-- Theorem stating the maximum value of A -/
theorem max_A_value :
  ∃ (config : BoardConfig),
    config.white_columns + config.black_columns = board_size ∧
    config.white_rows + config.black_rows = board_size ∧
    ∀ (other_config : BoardConfig),
      other_config.white_columns + other_config.black_columns = board_size →
      other_config.white_rows + other_config.black_rows = board_size →
      min (remaining_chips config true) (remaining_chips config false) ≥
      min (remaining_chips other_config true) (remaining_chips other_config false) ∧
    min (remaining_chips config true) (remaining_chips config false) = 1018081 :=
sorry

end max_A_value_l3383_338342


namespace janet_dress_pockets_janet_dress_problem_l3383_338301

theorem janet_dress_pockets (total_dresses : ℕ) (dresses_with_pockets : ℕ) 
  (dresses_unknown_pockets : ℕ) (known_pockets : ℕ) (total_pockets : ℕ) : ℕ :=
  let dresses_known_pockets := dresses_with_pockets - dresses_unknown_pockets
  let unknown_pockets := (total_pockets - dresses_known_pockets * known_pockets) / dresses_unknown_pockets
  unknown_pockets

theorem janet_dress_problem : janet_dress_pockets 24 12 4 3 32 = 2 := by
  sorry

end janet_dress_pockets_janet_dress_problem_l3383_338301


namespace triangle_inequality_from_squared_sum_l3383_338387

theorem triangle_inequality_from_squared_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b := by sorry

end triangle_inequality_from_squared_sum_l3383_338387


namespace buy_one_get_one_free_cost_l3383_338399

/-- Calculates the total cost of cans under a "buy 1 get one free" offer -/
def totalCost (totalCans : ℕ) (pricePerCan : ℚ) : ℚ :=
  (totalCans / 2 : ℚ) * pricePerCan

/-- Proves that the total cost for 30 cans at $0.60 each under a "buy 1 get one free" offer is $9 -/
theorem buy_one_get_one_free_cost :
  totalCost 30 (60 / 100) = 9 := by
  sorry

end buy_one_get_one_free_cost_l3383_338399


namespace neg_eight_celsius_meaning_l3383_338323

/-- Represents temperature in Celsius -/
structure Temperature where
  value : ℤ
  unit : String
  deriving Repr

/-- Converts a temperature to its representation relative to zero -/
def tempRelativeToZero (t : Temperature) : String :=
  if t.value > 0 then
    s!"{t.value}°C above zero"
  else if t.value < 0 then
    s!"{-t.value}°C below zero"
  else
    "0°C"

/-- The convention for representing temperatures -/
axiom temp_convention (t : Temperature) : 
  t.value > 0 → tempRelativeToZero t = s!"{t.value}°C above zero"

/-- Theorem: -8°C represents 8°C below zero -/
theorem neg_eight_celsius_meaning :
  let t : Temperature := ⟨-8, "C"⟩
  tempRelativeToZero t = "8°C below zero" := by
  sorry

end neg_eight_celsius_meaning_l3383_338323


namespace chord_length_is_2_sqrt_2_l3383_338371

-- Define the line
def line (x y : ℝ) : Prop := x + y = 3

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- Define the chord length
def chord_length (l : (ℝ → ℝ → Prop)) (c : (ℝ → ℝ → Prop)) : ℝ := 
  sorry  -- The actual computation of chord length would go here

-- Theorem statement
theorem chord_length_is_2_sqrt_2 :
  chord_length line curve = 2 * Real.sqrt 2 := by sorry

end chord_length_is_2_sqrt_2_l3383_338371


namespace analytic_method_characterization_l3383_338361

/-- Enumeration of proof methods --/
inductive ProofMethod
  | MathematicalInduction
  | ProofByContradiction
  | AnalyticMethod
  | SyntheticMethod

/-- Characteristic of a proof method --/
def isCharacterizedBy (m : ProofMethod) (c : String) : Prop :=
  match m with
  | ProofMethod.AnalyticMethod => c = "seeking the cause from the effect"
  | _ => c ≠ "seeking the cause from the effect"

/-- Theorem stating that the Analytic Method is characterized by "seeking the cause from the effect" --/
theorem analytic_method_characterization :
  isCharacterizedBy ProofMethod.AnalyticMethod "seeking the cause from the effect" :=
by sorry

end analytic_method_characterization_l3383_338361


namespace biology_marks_l3383_338309

def marks_english : ℕ := 86
def marks_mathematics : ℕ := 85
def marks_physics : ℕ := 82
def marks_chemistry : ℕ := 87
def average_marks : ℕ := 85
def total_subjects : ℕ := 5

theorem biology_marks :
  ∃ (marks_biology : ℕ),
    marks_biology = average_marks * total_subjects - (marks_english + marks_mathematics + marks_physics + marks_chemistry) ∧
    marks_biology = 85 := by
  sorry

end biology_marks_l3383_338309


namespace distance_from_origin_and_point_specific_distances_l3383_338357

theorem distance_from_origin_and_point (d : ℝ) (p : ℝ) :
  -- A point at distance d from the origin represents either d or -d
  (∃ x : ℝ, x = d ∨ x = -d ∧ |x| = d) ∧
  -- A point at distance d from p represents either p + d or p - d
  (∃ y : ℝ, y = p + d ∨ y = p - d ∧ |y - p| = d) :=
by
  sorry

-- Specific instances for the given problem
theorem specific_distances :
  -- A point at distance √5 from the origin represents either √5 or -√5
  (∃ x : ℝ, x = Real.sqrt 5 ∨ x = -Real.sqrt 5 ∧ |x| = Real.sqrt 5) ∧
  -- A point at distance 2√5 from √5 represents either 3√5 or -√5
  (∃ y : ℝ, y = 3 * Real.sqrt 5 ∨ y = -Real.sqrt 5 ∧ |y - Real.sqrt 5| = 2 * Real.sqrt 5) :=
by
  sorry

end distance_from_origin_and_point_specific_distances_l3383_338357


namespace hallway_floor_design_ratio_l3383_338306

/-- Given a rectangle with semicircles on either side, where the ratio of length to width
    is 4:1 and the width is 20 inches, the ratio of the area of the rectangle to the
    combined area of the semicircles is 16/π. -/
theorem hallway_floor_design_ratio : 
  ∀ (length width : ℝ),
  width = 20 →
  length = 4 * width →
  (length * width) / (π * (width / 2)^2) = 16 / π :=
by sorry

end hallway_floor_design_ratio_l3383_338306


namespace gauss_family_mean_age_l3383_338327

/-- The ages of the Gauss family children -/
def gauss_ages : List ℕ := [7, 7, 7, 14, 15]

/-- The number of children in the Gauss family -/
def num_children : ℕ := gauss_ages.length

/-- The mean age of the Gauss family children -/
def mean_age : ℚ := (gauss_ages.sum : ℚ) / num_children

theorem gauss_family_mean_age : mean_age = 10 := by
  sorry

end gauss_family_mean_age_l3383_338327


namespace triangle_angle_proof_l3383_338373

theorem triangle_angle_proof (a b c A B C : Real) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧
  b = a * (Real.sin C + Real.cos C) →
  A = π / 4 := by
  sorry

end triangle_angle_proof_l3383_338373


namespace expression_equality_l3383_338382

theorem expression_equality : (-1)^2023 - Real.sqrt 9 + |1 - Real.sqrt 2| - ((-8) ^ (1/3 : ℝ)) = Real.sqrt 2 - 3 := by
  sorry

end expression_equality_l3383_338382


namespace books_sum_is_67_l3383_338392

/-- The total number of books Sandy, Benny, and Tim have together -/
def total_books (sandy_books benny_books tim_books : ℕ) : ℕ :=
  sandy_books + benny_books + tim_books

/-- Theorem stating that the total number of books is 67 -/
theorem books_sum_is_67 :
  total_books 10 24 33 = 67 := by
  sorry

end books_sum_is_67_l3383_338392


namespace cat_stickers_count_l3383_338388

theorem cat_stickers_count (space_stickers : Nat) (friends : Nat) (leftover : Nat) (cat_stickers : Nat) : 
  space_stickers = 100 →
  friends = 3 →
  leftover = 3 →
  (space_stickers + cat_stickers - leftover) % friends = 0 →
  cat_stickers = 2 := by
  sorry

end cat_stickers_count_l3383_338388


namespace area_between_line_and_curve_l3383_338329

theorem area_between_line_and_curve : 
  let f (x : ℝ) := 3 * x
  let g (x : ℝ) := x^2
  let lower_bound := (0 : ℝ)
  let upper_bound := (3 : ℝ)
  let area := ∫ x in lower_bound..upper_bound, (f x - g x)
  area = 9/2 := by sorry

end area_between_line_and_curve_l3383_338329


namespace range_of_a_l3383_338314

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x : ℝ, x^2 - a ≥ 0)
  (h2 : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a ≤ -2 := by
sorry

end range_of_a_l3383_338314


namespace exactly_one_greater_than_one_l3383_338353

theorem exactly_one_greater_than_one 
  (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (prod_eq_one : a * b * c = 1)
  (sum_gt_recip_sum : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end exactly_one_greater_than_one_l3383_338353


namespace abs_sum_inequality_l3383_338389

theorem abs_sum_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), |x - 4| + |x - 6| ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), |x - 4| + |x - 6| ≥ b) → b ≤ 2) := by
  sorry

end abs_sum_inequality_l3383_338389


namespace third_stick_shorter_by_one_cm_l3383_338346

/-- The length difference between the second and third stick -/
def stick_length_difference (first_stick second_stick third_stick : ℝ) : ℝ :=
  second_stick - third_stick

/-- Proof that the third stick is 1 cm shorter than the second stick -/
theorem third_stick_shorter_by_one_cm 
  (first_stick : ℝ)
  (second_stick : ℝ)
  (third_stick : ℝ)
  (h1 : first_stick = 3)
  (h2 : second_stick = 2 * first_stick)
  (h3 : first_stick + second_stick + third_stick = 14) :
  stick_length_difference first_stick second_stick third_stick = 1 := by
sorry

end third_stick_shorter_by_one_cm_l3383_338346


namespace apollo_chariot_payment_l3383_338362

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months before the price increase -/
def months_before_increase : ℕ := 6

/-- The initial price in golden apples per month -/
def initial_price : ℕ := 3

/-- The price increase factor -/
def price_increase_factor : ℕ := 2

/-- The total number of golden apples paid for the year -/
def total_apples : ℕ := 
  initial_price * months_before_increase + 
  initial_price * price_increase_factor * (months_in_year - months_before_increase)

theorem apollo_chariot_payment :
  total_apples = 54 := by sorry

end apollo_chariot_payment_l3383_338362


namespace share_difference_l3383_338363

/-- Given a distribution ratio and Vasim's share, calculate the difference between Ranjith's and Faruk's shares -/
theorem share_difference (faruk_ratio vasim_ratio ranjith_ratio : ℕ) (vasim_share : ℕ) : 
  faruk_ratio = 3 →
  vasim_ratio = 5 →
  ranjith_ratio = 6 →
  vasim_share = 1500 →
  (ranjith_ratio * vasim_share / vasim_ratio) - (faruk_ratio * vasim_share / vasim_ratio) = 900 := by
sorry

end share_difference_l3383_338363


namespace ideal_function_fixed_point_l3383_338375

/-- An ideal function is a function f: [0,1] → ℝ satisfying:
    1) ∀ x ∈ [0,1], f(x) ≥ 0
    2) f(1) = 1
    3) ∀ x₁ x₂ ≥ 0 with x₁ + x₂ ≤ 1, f(x₁ + x₂) ≥ f(x₁) + f(x₂) -/
def IdealFunction (f : Real → Real) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧ 
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

theorem ideal_function_fixed_point 
  (f : Real → Real) (h : IdealFunction f) 
  (x₀ : Real) (hx₀ : x₀ ∈ Set.Icc 0 1) 
  (hfx₀ : f x₀ ∈ Set.Icc 0 1) (hffx₀ : f (f x₀) = x₀) : 
  f x₀ = x₀ := by
  sorry

end ideal_function_fixed_point_l3383_338375


namespace plane_perpendicular_parallel_implies_perpendicular_l3383_338386

-- Define the plane type
structure Plane where
  -- Add necessary fields or leave it abstract

-- Define the perpendicular and parallel relations
def perpendicular (p q : Plane) : Prop := sorry

def parallel (p q : Plane) : Prop := sorry

-- State the theorem
theorem plane_perpendicular_parallel_implies_perpendicular 
  (α β γ : Plane) 
  (h1 : α ≠ β) (h2 : β ≠ γ) (h3 : α ≠ γ)
  (h4 : perpendicular α β) 
  (h5 : parallel β γ) : 
  perpendicular α γ := by sorry

end plane_perpendicular_parallel_implies_perpendicular_l3383_338386


namespace fibonacci_fourth_term_divisible_by_three_l3383_338330

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_fourth_term_divisible_by_three (k : ℕ) :
  3 ∣ fibonacci (4 * k) := by
  sorry

end fibonacci_fourth_term_divisible_by_three_l3383_338330


namespace batch_size_calculation_l3383_338347

theorem batch_size_calculation (N : ℕ) (sample_size : ℕ) (prob : ℚ) 
  (h1 : sample_size = 30)
  (h2 : prob = 1/4)
  (h3 : (sample_size : ℚ) / N = prob) : 
  N = 120 := by
  sorry

end batch_size_calculation_l3383_338347


namespace fourth_guard_distance_l3383_338384

/-- Represents a rectangular facility with guards -/
structure Facility :=
  (length : ℝ)
  (width : ℝ)
  (perimeter : ℝ)
  (three_guards_distance : ℝ)

/-- The theorem to prove -/
theorem fourth_guard_distance (f : Facility) 
  (h1 : f.length = 200)
  (h2 : f.width = 300)
  (h3 : f.perimeter = 2 * (f.length + f.width))
  (h4 : f.three_guards_distance = 850) :
  f.perimeter - f.three_guards_distance = 150 := by
  sorry

#check fourth_guard_distance

end fourth_guard_distance_l3383_338384


namespace triangle_side_length_l3383_338320

/-- Given a triangle ABC with circumradius R, prove that if cos B and cos A are known,
    then the length of side c can be determined. -/
theorem triangle_side_length (A B C : Real) (R : Real) (h1 : R = 5/6)
  (h2 : Real.cos B = 3/5) (h3 : Real.cos A = 12/13) :
  2 * R * Real.sin (A + B) = 21/13 := by
  sorry

end triangle_side_length_l3383_338320


namespace f_properties_l3383_338394

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + Real.log x

theorem f_properties (a : ℝ) :
  (∃ x ∈ Set.Icc 1 2, f a x ≥ 0 → a ≤ 2 + 1/2 * Real.log 2) ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioi 1 ∧ 
    (∀ x : ℝ, deriv (f a) x = 0 ↔ x = x₁ ∨ x = x₂) →
    f a x₁ - f a x₂ < -3/4 + Real.log 2) := by
  sorry

end f_properties_l3383_338394


namespace alternating_dodecagon_area_l3383_338381

/-- An equilateral 12-gon with alternating interior angles -/
structure AlternatingDodecagon where
  side_length : ℝ
  interior_angles : Fin 12 → ℝ
  is_equilateral : ∀ i : Fin 12, side_length > 0
  angle_pattern : ∀ i : Fin 12, interior_angles i = 
    if i % 3 = 0 ∨ i % 3 = 1 then 90 else 270

/-- The area of the alternating dodecagon -/
noncomputable def area (d : AlternatingDodecagon) : ℝ := sorry

/-- Theorem stating that the area of the specific alternating dodecagon is 500 -/
theorem alternating_dodecagon_area :
  ∀ d : AlternatingDodecagon, d.side_length = 10 → area d = 500 := by sorry

end alternating_dodecagon_area_l3383_338381


namespace triangle_abc_properties_l3383_338359

theorem triangle_abc_properties (A B C : Real) (m n : Real × Real) :
  -- Given conditions
  m = (Real.sin A, Real.sin B) →
  n = (Real.cos B, Real.cos A) →
  m.1 * n.1 + m.2 * n.2 = Real.sin (2 * C) →
  A + B + C = π →
  2 * Real.sin C = Real.sin A + Real.sin B →
  Real.sin A * Real.sin C * (Real.sin B - Real.sin A) = 18 →
  -- Conclusions
  C = π / 3 ∧ 
  2 * Real.sin A * Real.sin B * Real.cos C = 18 ∧
  Real.sin A * Real.sin B = 16 ∧
  Real.sin C = Real.sin A + Real.sin B - Real.sin A * Real.sin B / 2 := by
sorry

end triangle_abc_properties_l3383_338359


namespace price_change_equivalence_l3383_338372

theorem price_change_equivalence (initial_price : ℝ) (x : ℝ) 
  (h1 : initial_price > 0)
  (h2 : x > 0 ∧ x < 100) :
  (1.25 * initial_price) * (1 - x / 100) = 1.125 * initial_price → x = 10 := by
sorry

end price_change_equivalence_l3383_338372


namespace composition_equation_solution_l3383_338310

/-- Given functions f and g, and a condition on their composition, prove the value of a. -/
theorem composition_equation_solution (a : ℝ) : 
  (let f (x : ℝ) := (x + 4) / 7 + 2
   let g (x : ℝ) := 5 - 2 * x
   f (g a) = 8) → 
  a = -33/2 := by
sorry

end composition_equation_solution_l3383_338310


namespace total_slices_needed_l3383_338326

/-- The number of sandwiches Ryan wants to make -/
def num_sandwiches : ℕ := 5

/-- The number of bread slices needed for each sandwich -/
def slices_per_sandwich : ℕ := 3

/-- Theorem stating the total number of bread slices needed -/
theorem total_slices_needed : num_sandwiches * slices_per_sandwich = 15 := by
  sorry

end total_slices_needed_l3383_338326


namespace line_intersection_range_l3383_338313

theorem line_intersection_range (a : ℝ) : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ 2 * x + (3 - a) = 0) ↔ 5 ≤ a ∧ a ≤ 9 := by
  sorry

end line_intersection_range_l3383_338313


namespace jims_remaining_distance_l3383_338337

/-- Calculates the remaining distance in a journey. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Proves that for Jim's journey, the remaining distance is 1,068 miles. -/
theorem jims_remaining_distance :
  remaining_distance 2450 1382 = 1068 := by
  sorry

end jims_remaining_distance_l3383_338337


namespace hyperbola_conjugate_axis_length_l3383_338324

/-- Given a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 and eccentricity 2,
    if the product of the distances from a point on the hyperbola to its two asymptotes is 3/4,
    then the length of the conjugate axis is 2√3. -/
theorem hyperbola_conjugate_axis_length 
  (a b : ℝ) 
  (h1 : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → 
    (|b*x - a*y| * |b*x + a*y|) / (a^2 + b^2) = 3/4)
  (h2 : a^2 + b^2 = 5*a^2) :
  2*b = 2*Real.sqrt 3 := by sorry

end hyperbola_conjugate_axis_length_l3383_338324


namespace skateboard_speed_l3383_338390

/-- 
Given Pedro's skateboarding speed and time, prove Liam's required speed 
to cover the same distance in a different time.
-/
theorem skateboard_speed 
  (pedro_speed : ℝ) 
  (pedro_time : ℝ) 
  (liam_time : ℝ) 
  (h1 : pedro_speed = 10) 
  (h2 : pedro_time = 4) 
  (h3 : liam_time = 5) : 
  (pedro_speed * pedro_time) / liam_time = 8 := by
  sorry

#check skateboard_speed

end skateboard_speed_l3383_338390


namespace first_line_time_l3383_338393

/-- Represents the productivity of a production line -/
structure ProductivityRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a production line -/
structure ProductionLine where
  productivity : ProductivityRate

/-- Represents a system of three production lines -/
structure ProductionSystem where
  line1 : ProductionLine
  line2 : ProductionLine
  line3 : ProductionLine
  combined_productivity : ProductivityRate
  first_second_productivity : ProductivityRate
  combined_vs_first_second : combined_productivity.rate = 1.5 * first_second_productivity.rate
  second_faster_than_first : line2.productivity.rate = line1.productivity.rate + (1 / 2)
  second_third_vs_first : 
    1 / line1.productivity.rate - (24 / 5) = 
    1 / (line2.productivity.rate + line3.productivity.rate)

theorem first_line_time (system : ProductionSystem) : 
  1 / system.line1.productivity.rate = 8 := by
  sorry

end first_line_time_l3383_338393


namespace prime_sum_implies_prime_exponent_l3383_338335

theorem prime_sum_implies_prime_exponent (p d : ℕ) : 
  Prime p → p = (10^d - 1) / 9 → Prime d := by
  sorry

end prime_sum_implies_prime_exponent_l3383_338335


namespace max_value_expression_max_value_achievable_l3383_338300

theorem max_value_expression (x : ℝ) :
  x^6 / (x^12 + 3*x^8 - 6*x^6 + 12*x^4 + 36) ≤ 1/18 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, x^6 / (x^12 + 3*x^8 - 6*x^6 + 12*x^4 + 36) = 1/18 :=
by sorry

end max_value_expression_max_value_achievable_l3383_338300


namespace shaded_area_is_three_l3383_338377

def grid_area : ℕ := 3 * 2 + 4 * 6 + 5 * 3

def unshaded_triangle_area : ℕ := (14 * 6) / 2

def shaded_area : ℕ := grid_area - unshaded_triangle_area

theorem shaded_area_is_three : shaded_area = 3 := by
  sorry

end shaded_area_is_three_l3383_338377


namespace special_polynomial_exists_l3383_338376

/-- A fifth-degree polynomial with specific root properties -/
def exists_special_polynomial : Prop :=
  ∃ (P : ℝ → ℝ),
    (∀ x : ℝ, ∃ (a b c d e f : ℝ), P x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) ∧
    (∀ r : ℝ, P r = 0 → r < 0) ∧
    (∀ s : ℝ, (deriv P) s = 0 → s > 0) ∧
    (∃ t : ℝ, P t = 0) ∧
    (∃ u : ℝ, (deriv P) u = 0)

/-- Theorem stating the existence of a special polynomial -/
theorem special_polynomial_exists : exists_special_polynomial :=
sorry

end special_polynomial_exists_l3383_338376


namespace exists_valid_coloring_l3383_338366

-- Define the color type
inductive Color
  | White
  | Red
  | Black

-- Define the coloring function type
def ColoringFunction := ℤ × ℤ → Color

-- Define what it means for a color to appear on infinitely many lines
def AppearsOnInfinitelyManyLines (f : ColoringFunction) (c : Color) : Prop :=
  ∀ n : ℕ, ∃ y : ℤ, y > n ∧ (∀ m : ℕ, ∃ x : ℤ, x > m ∧ f (x, y) = c)

-- Define what it means to be a parallelogram
def IsParallelogram (A B C D : ℤ × ℤ) : Prop :=
  B.1 - A.1 = D.1 - C.1 ∧ B.2 - A.2 = D.2 - C.2

-- Main theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction,
  (AppearsOnInfinitelyManyLines f Color.White) ∧
  (AppearsOnInfinitelyManyLines f Color.Red) ∧
  (AppearsOnInfinitelyManyLines f Color.Black) ∧
  (∀ A B C : ℤ × ℤ, f A = Color.White → f B = Color.Red → f C = Color.Black →
    ∃ D : ℤ × ℤ, f D = Color.Red ∧ IsParallelogram A B C D) :=
sorry

end exists_valid_coloring_l3383_338366


namespace balloon_permutations_l3383_338315

def balloon_letters : Nat := 7
def l_count : Nat := 2
def o_count : Nat := 3

theorem balloon_permutations :
  (balloon_letters.factorial) / (l_count.factorial * o_count.factorial) = 420 := by
  sorry

end balloon_permutations_l3383_338315


namespace A_sufficient_not_necessary_for_D_l3383_338340

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : (B ↔ C))
variable (h3 : (D → C) ∧ ¬(C → D))

-- Theorem to prove
theorem A_sufficient_not_necessary_for_D : 
  (A → D) ∧ ¬(D → A) :=
sorry

end A_sufficient_not_necessary_for_D_l3383_338340


namespace birthday_problem_l3383_338385

/-- The number of months in the fantasy world -/
def num_months : ℕ := 10

/-- The number of people in the room -/
def num_people : ℕ := 60

/-- The largest number n such that at least n people are guaranteed to have birthdays in the same month -/
def largest_guaranteed_group : ℕ := 6

theorem birthday_problem :
  ∀ (birthday_distribution : Fin num_people → Fin num_months),
  ∃ (month : Fin num_months),
  (Finset.filter (λ person => birthday_distribution person = month) Finset.univ).card ≥ largest_guaranteed_group ∧
  ∀ n > largest_guaranteed_group,
  ∃ (bad_distribution : Fin num_people → Fin num_months),
  ∀ (month : Fin num_months),
  (Finset.filter (λ person => bad_distribution person = month) Finset.univ).card < n :=
sorry

end birthday_problem_l3383_338385


namespace geometric_sequence_common_ratio_l3383_338317

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the condition given in the problem
def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = (16 : ℝ) ^ n

-- Theorem statement
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : satisfies_condition a) : 
  ∃ r : ℝ, (∀ n : ℕ, a (n + 1) = r * a n) ∧ r = 4 :=
sorry

end geometric_sequence_common_ratio_l3383_338317


namespace smallest_k_inequality_l3383_338333

theorem smallest_k_inequality (x y z : ℝ) :
  ∃ (k : ℝ), k = 3 ∧ (x^2 + y^2 + z^2)^2 ≤ k * (x^4 + y^4 + z^4) ∧
  ∀ (k' : ℝ), (∀ (a b c : ℝ), (a^2 + b^2 + c^2)^2 ≤ k' * (a^4 + b^4 + c^4)) → k' ≥ k :=
sorry

end smallest_k_inequality_l3383_338333


namespace simplify_nested_roots_l3383_338383

theorem simplify_nested_roots : 
  (65536 : ℝ) = 2^16 →
  (((1 / 65536)^(1/2))^(1/3))^(1/4) = 1 / (4^(1/3)) :=
by sorry

end simplify_nested_roots_l3383_338383


namespace sandy_paint_area_l3383_338358

/-- The area Sandy needs to paint on a wall with a decorative region -/
theorem sandy_paint_area (wall_height wall_length decor_height decor_length : ℝ)
  (h_wall_height : wall_height = 10)
  (h_wall_length : wall_length = 15)
  (h_decor_height : decor_height = 3)
  (h_decor_length : decor_length = 5) :
  wall_height * wall_length - decor_height * decor_length = 135 := by
  sorry

end sandy_paint_area_l3383_338358


namespace real_numbers_closed_closed_set_contains_zero_l3383_338350

-- Definition of a closed set
def is_closed_set (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x - y) ∈ S ∧ (x * y) ∈ S

-- Theorem 1: The set of real numbers is a closed set
theorem real_numbers_closed : is_closed_set Set.univ := by sorry

-- Theorem 2: If S is a closed set, then 0 is an element of S
theorem closed_set_contains_zero (S : Set ℝ) (h : is_closed_set S) (h_nonempty : S.Nonempty) : 
  (0 : ℝ) ∈ S := by sorry

end real_numbers_closed_closed_set_contains_zero_l3383_338350


namespace ship_speed_calculation_l3383_338307

/-- The speed of the train in km/h -/
def train_speed : ℝ := 48

/-- The total distance traveled in km -/
def total_distance : ℝ := 480

/-- The additional time taken by the train in hours -/
def additional_time : ℝ := 2

/-- The speed of the ship in km/h -/
def ship_speed : ℝ := 60

theorem ship_speed_calculation : 
  (total_distance / ship_speed) + additional_time = total_distance / train_speed := by
  sorry

#check ship_speed_calculation

end ship_speed_calculation_l3383_338307


namespace ellipse_foci_distance_l3383_338370

/-- Given an ellipse with equation x^2 + 9y^2 = 8100, 
    the distance between its foci is 120√2 -/
theorem ellipse_foci_distance : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, x^2 + 9*y^2 = 8100 → x^2/a^2 + y^2/b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    2*c = 120*Real.sqrt 2 := by
  sorry

end ellipse_foci_distance_l3383_338370


namespace fibonacci_arithmetic_sequence_l3383_338351

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the theorem
theorem fibonacci_arithmetic_sequence (a b c : ℕ) :
  (fib a < fib b) ∧ (fib b < fib c) ∧  -- Fₐ, Fₑ, Fₒ form an increasing sequence
  (fib (a + 1) < fib (b + 1)) ∧ (fib (b + 1) < fib (c + 1)) ∧  -- Fₐ₊₁, Fₑ₊₁, Fₒ₊₁ form an increasing sequence
  (fib c - fib b = fib b - fib a) ∧  -- Arithmetic sequence condition
  (fib (c + 1) - fib (b + 1) = fib (b + 1) - fib (a + 1)) ∧  -- Arithmetic sequence condition for next terms
  (a + b + c = 3000) →  -- Sum condition
  a = 999 := by
  sorry

end fibonacci_arithmetic_sequence_l3383_338351


namespace total_pet_owners_l3383_338369

/-- The number of people who own only dogs -/
def only_dogs : ℕ := 15

/-- The number of people who own only cats -/
def only_cats : ℕ := 10

/-- The number of people who own only cats and dogs -/
def cats_and_dogs : ℕ := 5

/-- The number of people who own cats, dogs, and snakes -/
def cats_dogs_snakes : ℕ := 3

/-- The total number of snakes -/
def total_snakes : ℕ := 29

/-- Theorem stating the total number of pet owners -/
theorem total_pet_owners : 
  only_dogs + only_cats + cats_and_dogs + cats_dogs_snakes = 33 := by
  sorry


end total_pet_owners_l3383_338369


namespace convex_pentagon_angles_obtuse_l3383_338321

/-- A convex pentagon with equal sides and each angle less than 120° -/
structure ConvexPentagon where
  -- The pentagon is convex
  is_convex : Bool
  -- All sides are equal
  equal_sides : Bool
  -- Each angle is less than 120°
  angles_less_than_120 : Bool

/-- Theorem: In a convex pentagon with equal sides and each angle less than 120°, 
    each angle is greater than 90° -/
theorem convex_pentagon_angles_obtuse (p : ConvexPentagon) : 
  p.is_convex ∧ p.equal_sides ∧ p.angles_less_than_120 → 
  ∀ angle, angle > 90 := by sorry

end convex_pentagon_angles_obtuse_l3383_338321


namespace craft_sales_sum_l3383_338318

/-- The sum of an arithmetic sequence with first term 3 and common difference 4 for 10 terms -/
theorem craft_sales_sum : 
  let a : ℕ → ℕ := fun n => 3 + 4 * (n - 1)
  let S : ℕ → ℕ := fun n => n * (a 1 + a n) / 2
  S 10 = 210 := by
sorry

end craft_sales_sum_l3383_338318


namespace coefficient_x3_is_80_l3383_338332

/-- The coefficient of x^3 in the expansion of (1+2x)^5 -/
def coefficient_x3 : ℕ :=
  Nat.choose 5 3 * 2^3

/-- Theorem stating that the coefficient of x^3 in (1+2x)^5 is 80 -/
theorem coefficient_x3_is_80 : coefficient_x3 = 80 := by
  sorry

end coefficient_x3_is_80_l3383_338332


namespace product_of_squares_l3383_338325

theorem product_of_squares (N : ℕ+) 
  (h : ∃! (a₁ b₁ a₂ b₂ a₃ b₃ : ℕ+), 
    a₁^2 * b₁^2 = N ∧ 
    a₂^2 * b₂^2 = N ∧ 
    a₃^2 * b₃^2 = N ∧
    (a₁, b₁) ≠ (a₂, b₂) ∧ 
    (a₁, b₁) ≠ (a₃, b₃) ∧ 
    (a₂, b₂) ≠ (a₃, b₃)) :
  ∃ (a₁ b₁ a₂ b₂ a₃ b₃ : ℕ+), 
    a₁^2 * b₁^2 * a₂^2 * b₂^2 * a₃^2 * b₃^2 = N^3 :=
sorry

end product_of_squares_l3383_338325


namespace expression_bounds_l3383_338352

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + 
    Real.sqrt (e^2 + (1-a)^2) ∧
  Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
  Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + 
  Real.sqrt (e^2 + (1-a)^2) ≤ 5 :=
by sorry

end expression_bounds_l3383_338352


namespace football_field_length_prove_football_field_length_l3383_338380

theorem football_field_length : ℝ → Prop :=
  fun length =>
    (4 * length + 500 = 1172) →
    length = 168

-- The proof is omitted
theorem prove_football_field_length : football_field_length 168 := by
  sorry

end football_field_length_prove_football_field_length_l3383_338380


namespace wedding_guests_l3383_338379

/-- The number of guests at Jenny's wedding --/
def total_guests : ℕ := 80

/-- The number of guests who want chicken --/
def chicken_guests : ℕ := 20

/-- The number of guests who want steak --/
def steak_guests : ℕ := 60

/-- The cost of a chicken entree in dollars --/
def chicken_cost : ℕ := 18

/-- The cost of a steak entree in dollars --/
def steak_cost : ℕ := 25

/-- The total catering budget in dollars --/
def total_budget : ℕ := 1860

theorem wedding_guests :
  (chicken_guests + steak_guests = total_guests) ∧
  (steak_guests = 3 * chicken_guests) ∧
  (chicken_cost * chicken_guests + steak_cost * steak_guests = total_budget) := by
  sorry

end wedding_guests_l3383_338379


namespace P₁_subset_P₂_l3383_338397

/-- P₁ is the set of real numbers x such that x² + ax + 1 > 0 -/
def P₁ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}

/-- P₂ is the set of real numbers x such that x² + ax + 2 > 0 -/
def P₂ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}

/-- For all real numbers a, P₁(a) is a subset of P₂(a) -/
theorem P₁_subset_P₂ : ∀ a : ℝ, P₁ a ⊆ P₂ a := by
  sorry

end P₁_subset_P₂_l3383_338397


namespace kannon_oranges_last_night_l3383_338341

/-- Represents the number of fruits Kannon ate --/
structure FruitCount where
  apples : ℕ
  bananas : ℕ
  oranges : ℕ

/-- The total number of fruits eaten over two meals --/
def totalFruits : ℕ := 39

/-- Kannon's fruit consumption for last night --/
def lastNight : FruitCount where
  apples := 3
  bananas := 1
  oranges := 4  -- This is what we want to prove

/-- Kannon's fruit consumption for today --/
def today : FruitCount where
  apples := lastNight.apples + 4
  bananas := 10 * lastNight.bananas
  oranges := 2 * (lastNight.apples + 4)

/-- The theorem to prove --/
theorem kannon_oranges_last_night :
  lastNight.oranges = 4 ∧
  lastNight.apples + lastNight.bananas + lastNight.oranges +
  today.apples + today.bananas + today.oranges = totalFruits := by
  sorry


end kannon_oranges_last_night_l3383_338341


namespace alpha_range_l3383_338302

theorem alpha_range (α : Real) 
  (h1 : 0 ≤ α ∧ α < 2 * Real.pi) 
  (h2 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  Real.pi / 3 < α ∧ α < 4 * Real.pi / 3 := by
  sorry

end alpha_range_l3383_338302


namespace optimal_strategy_l3383_338374

-- Define the set of available numbers
def availableNumbers : Finset Nat := Finset.range 17

-- Define the rules of the game
def isValidChoice (chosen : Finset Nat) (n : Nat) : Bool :=
  n ∈ availableNumbers ∧
  n ∉ chosen ∧
  ¬(∃m ∈ chosen, n = 2 * m ∨ 2 * n = m)

-- Define the state after Player A's move
def initialState : Finset Nat := {8}

-- Define Player B's optimal move
def optimalMove : Nat := 6

-- Theorem to prove
theorem optimal_strategy :
  isValidChoice initialState optimalMove ∧
  ∀ n : Nat, n ≠ optimalMove → 
    (isValidChoice initialState n → 
      ∃ m : Nat, isValidChoice (insert n initialState) m) →
    ¬(∃ m : Nat, isValidChoice (insert optimalMove initialState) m) :=
sorry

end optimal_strategy_l3383_338374


namespace necessary_but_not_sufficient_l3383_338328

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ a, (0 < a ∧ a < 1) → (a + 1) * (a - 2) < 0) ∧
  (∃ a, (a + 1) * (a - 2) < 0 ∧ (a ≤ 0 ∨ a ≥ 1)) :=
by sorry

end necessary_but_not_sufficient_l3383_338328


namespace intersection_A_complement_B_l3383_338354

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - x)}

-- Theorem to prove
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end intersection_A_complement_B_l3383_338354


namespace manicure_cost_proof_l3383_338364

/-- The cost of a hair updo in dollars -/
def hair_updo_cost : ℝ := 50

/-- The total cost including tips for both services in dollars -/
def total_cost_with_tips : ℝ := 96

/-- The tip percentage as a decimal -/
def tip_percentage : ℝ := 0.20

/-- The cost of a manicure in dollars -/
def manicure_cost : ℝ := 30

theorem manicure_cost_proof :
  (hair_updo_cost + manicure_cost) * (1 + tip_percentage) = total_cost_with_tips := by
  sorry

end manicure_cost_proof_l3383_338364


namespace spaceship_journey_l3383_338343

def total_distance : Real := 0.7
def earth_to_x : Real := 0.5
def y_to_earth : Real := 0.1

theorem spaceship_journey : 
  total_distance - earth_to_x - y_to_earth = 0.1 := by sorry

end spaceship_journey_l3383_338343
