import Mathlib

namespace max_min_on_interval_l3403_340350

def f (x : ℝ) : ℝ := 3 * x^4 + 4 * x^3 + 34

theorem max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = min) ∧
    max = 50 ∧ min = 33 := by
  sorry

end max_min_on_interval_l3403_340350


namespace unique_prime_p_squared_plus_two_prime_l3403_340366

theorem unique_prime_p_squared_plus_two_prime : 
  ∃! p : ℕ, Prime p ∧ Prime (p^2 + 2) :=
by sorry

end unique_prime_p_squared_plus_two_prime_l3403_340366


namespace sine_domain_range_constraint_l3403_340318

theorem sine_domain_range_constraint (a b : Real) : 
  (∀ x ∈ Set.Icc a b, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1/2) →
  (∃ x ∈ Set.Icc a b, Real.sin x = -1) →
  (∃ x ∈ Set.Icc a b, Real.sin x = 1/2) →
  b - a ≠ π/3 := by
  sorry

end sine_domain_range_constraint_l3403_340318


namespace expected_winnings_is_five_thirds_l3403_340338

/-- A coin with three possible outcomes -/
inductive CoinOutcome
  | Heads
  | Tails
  | Edge

/-- The probability of each outcome -/
def probability (outcome : CoinOutcome) : ℚ :=
  match outcome with
  | .Heads => 1/3
  | .Tails => 1/2
  | .Edge => 1/6

/-- The payoff for each outcome in dollars -/
def payoff (outcome : CoinOutcome) : ℤ :=
  match outcome with
  | .Heads => 2
  | .Tails => 4
  | .Edge => -6

/-- The expected winnings from flipping the coin -/
def expectedWinnings : ℚ :=
  (probability CoinOutcome.Heads * payoff CoinOutcome.Heads) +
  (probability CoinOutcome.Tails * payoff CoinOutcome.Tails) +
  (probability CoinOutcome.Edge * payoff CoinOutcome.Edge)

theorem expected_winnings_is_five_thirds :
  expectedWinnings = 5/3 := by
  sorry

end expected_winnings_is_five_thirds_l3403_340338


namespace ravi_coin_value_l3403_340376

/-- Represents the number of coins of each type Ravi has -/
structure CoinCounts where
  nickels : ℕ
  quarters : ℕ
  dimes : ℕ
  half_dollars : ℕ
  pennies : ℕ

/-- Calculates the total value of coins in cents -/
def total_value (counts : CoinCounts) : ℕ :=
  counts.nickels * 5 +
  counts.quarters * 25 +
  counts.dimes * 10 +
  counts.half_dollars * 50 +
  counts.pennies * 1

/-- Theorem stating that Ravi's coin collection is worth $12.51 -/
theorem ravi_coin_value : ∃ (counts : CoinCounts),
  counts.nickels = 6 ∧
  counts.quarters = counts.nickels + 2 ∧
  counts.dimes = counts.quarters + 4 ∧
  counts.half_dollars = counts.dimes + 5 ∧
  counts.pennies = counts.half_dollars * 3 ∧
  total_value counts = 1251 := by
  sorry

end ravi_coin_value_l3403_340376


namespace units_digit_of_expression_l3403_340321

theorem units_digit_of_expression : ∃ n : ℕ, (9 * 19 * 1989 - 9^4) % 10 = 8 ∧ n * 10 + 8 = 9 * 19 * 1989 - 9^4 := by
  sorry

end units_digit_of_expression_l3403_340321


namespace total_appetizers_l3403_340326

def hotdogs : ℕ := 60
def cheese_pops : ℕ := 40
def chicken_nuggets : ℕ := 80
def mini_quiches : ℕ := 100
def stuffed_mushrooms : ℕ := 50

theorem total_appetizers : 
  hotdogs + cheese_pops + chicken_nuggets + mini_quiches + stuffed_mushrooms = 330 := by
  sorry

end total_appetizers_l3403_340326


namespace solve_ice_cubes_problem_l3403_340342

def ice_cubes_problem (x : ℕ) : Prop :=
  let glass_ice := x
  let pitcher_ice := 2 * x
  let total_ice := glass_ice + pitcher_ice
  let tray_capacity := 2 * 12
  total_ice = tray_capacity

theorem solve_ice_cubes_problem :
  ∃ x : ℕ, ice_cubes_problem x ∧ x = 8 := by
  sorry

end solve_ice_cubes_problem_l3403_340342


namespace power_multiplication_l3403_340316

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end power_multiplication_l3403_340316


namespace basement_water_pump_time_l3403_340386

/-- Proves that it takes 225 minutes to pump out water from a flooded basement --/
theorem basement_water_pump_time : 
  let basement_length : ℝ := 30
  let basement_width : ℝ := 40
  let water_depth_inches : ℝ := 12
  let num_pumps : ℕ := 4
  let pump_rate : ℝ := 10  -- gallons per minute
  let gallons_per_cubic_foot : ℝ := 7.5
  let inches_per_foot : ℝ := 12

  let water_depth_feet : ℝ := water_depth_inches / inches_per_foot
  let water_volume_cubic_feet : ℝ := basement_length * basement_width * water_depth_feet
  let water_volume_gallons : ℝ := water_volume_cubic_feet * gallons_per_cubic_foot
  let total_pump_rate : ℝ := num_pumps * pump_rate
  let pump_time_minutes : ℝ := water_volume_gallons / total_pump_rate

  pump_time_minutes = 225 := by
  sorry

end basement_water_pump_time_l3403_340386


namespace class_average_l3403_340300

theorem class_average (total_students : ℕ) 
                      (top_scorers : ℕ) 
                      (zero_scorers : ℕ) 
                      (top_score : ℝ) 
                      (rest_average : ℝ) :
  total_students = 25 →
  top_scorers = 5 →
  zero_scorers = 3 →
  top_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - top_scorers - zero_scorers
  let total_score := top_scorers * top_score + zero_scorers * 0 + rest_students * rest_average
  total_score / total_students = 49.6 := by
  sorry

end class_average_l3403_340300


namespace complex_equation_solution_l3403_340374

theorem complex_equation_solution (i : ℂ) (m : ℝ) :
  i * i = -1 →
  (1 - m * i) / (i^3) = 1 + i →
  m = 1 := by
sorry

end complex_equation_solution_l3403_340374


namespace barbi_monthly_loss_is_one_point_five_l3403_340339

/-- Represents the weight loss scenario of Barbi and Luca -/
structure WeightLossScenario where
  barbi_monthly_loss : ℝ
  months_in_year : ℕ
  luca_yearly_loss : ℝ
  luca_years : ℕ
  difference : ℝ

/-- The weight loss scenario satisfies the given conditions -/
def satisfies_conditions (scenario : WeightLossScenario) : Prop :=
  scenario.months_in_year = 12 ∧
  scenario.luca_yearly_loss = 9 ∧
  scenario.luca_years = 11 ∧
  scenario.difference = 81 ∧
  scenario.luca_yearly_loss * scenario.luca_years = 
    scenario.barbi_monthly_loss * scenario.months_in_year + scenario.difference

/-- Theorem stating that under the given conditions, Barbi's monthly weight loss is 1.5 kg -/
theorem barbi_monthly_loss_is_one_point_five 
  (scenario : WeightLossScenario) 
  (h : satisfies_conditions scenario) : 
  scenario.barbi_monthly_loss = 1.5 := by
  sorry


end barbi_monthly_loss_is_one_point_five_l3403_340339


namespace cos_sin_sum_equals_half_l3403_340391

theorem cos_sin_sum_equals_half : 
  Real.cos (π / 4) * Real.cos (π / 12) - Real.sin (π / 4) * Real.sin (π / 12) = 1 / 2 := by
  sorry

end cos_sin_sum_equals_half_l3403_340391


namespace coin_combination_difference_l3403_340329

def coin_values : List ℕ := [10, 20, 50]
def target_amount : ℕ := 45

def valid_combination (coins : List ℕ) : Prop :=
  coins.all (λ c => c ∈ coin_values) ∧ coins.sum = target_amount

def num_coins (coins : List ℕ) : ℕ := coins.length

theorem coin_combination_difference :
  ∃ (min_coins max_coins : List ℕ),
    valid_combination min_coins ∧
    valid_combination max_coins ∧
    (∀ coins, valid_combination coins → num_coins min_coins ≤ num_coins coins) ∧
    (∀ coins, valid_combination coins → num_coins coins ≤ num_coins max_coins) ∧
    num_coins max_coins - num_coins min_coins = 0 :=
sorry

end coin_combination_difference_l3403_340329


namespace business_profit_theorem_l3403_340367

def business_profit_distribution (total_profit : ℝ) : ℝ :=
  let majority_owner_share := 0.25 * total_profit
  let remaining_profit := total_profit - majority_owner_share
  let partner_share := 0.25 * remaining_profit
  majority_owner_share + 2 * partner_share

theorem business_profit_theorem :
  business_profit_distribution 80000 = 50000 := by
  sorry

end business_profit_theorem_l3403_340367


namespace clown_balloons_l3403_340328

theorem clown_balloons (initial_balloons : ℕ) : 
  initial_balloons + 13 = 60 → initial_balloons = 47 := by
  sorry

end clown_balloons_l3403_340328


namespace max_decimal_places_is_14_complex_expression_decimal_places_l3403_340380

/-- The number of decimal places in 3.456789 -/
def decimal_places_a : ℕ := 6

/-- The number of decimal places in 6.78901234 -/
def decimal_places_b : ℕ := 8

/-- The expression ((10 ^ 5 * 3.456789) ^ 12) / (6.78901234 ^ 4)) ^ 9 -/
noncomputable def complex_expression : ℝ := 
  (((10 ^ 5 * 3.456789) ^ 12) / (6.78901234 ^ 4)) ^ 9

/-- The maximum number of decimal places in the result -/
def max_decimal_places : ℕ := decimal_places_a + decimal_places_b

theorem max_decimal_places_is_14 : 
  max_decimal_places = 14 := by sorry

theorem complex_expression_decimal_places : 
  ∃ (n : ℕ), n ≤ max_decimal_places ∧ 
  complex_expression * (10 ^ n) = ⌊complex_expression * (10 ^ n)⌋ := by sorry

end max_decimal_places_is_14_complex_expression_decimal_places_l3403_340380


namespace tan_monotonic_interval_l3403_340340

/-- The monotonic increasing interval of tan(x + π/4) -/
theorem tan_monotonic_interval (k : ℤ) :
  ∀ x : ℝ, (k * π - 3 * π / 4 < x ∧ x < k * π + π / 4) →
    Monotone (fun x => Real.tan (x + π / 4)) := by
  sorry

end tan_monotonic_interval_l3403_340340


namespace farm_area_calculation_l3403_340363

/-- Given a farm divided into sections, calculate its total area -/
def farm_total_area (num_sections : ℕ) (section_area : ℕ) : ℕ :=
  num_sections * section_area

/-- Theorem: The total area of a farm with 5 sections of 60 acres each is 300 acres -/
theorem farm_area_calculation : farm_total_area 5 60 = 300 := by
  sorry

end farm_area_calculation_l3403_340363


namespace divisibility_of_expression_l3403_340348

theorem divisibility_of_expression (m : ℕ) 
  (h1 : m > 0) 
  (h2 : Odd m) 
  (h3 : ¬(3 ∣ m)) : 
  112 ∣ (Int.floor (4^m - (2 + Real.sqrt 2)^m)) := by
sorry

end divisibility_of_expression_l3403_340348


namespace quadratic_and_fractional_equations_l3403_340305

theorem quadratic_and_fractional_equations :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 5 ∧ x₂ = 1 - Real.sqrt 5 ∧
    x₁^2 - 2*x₁ - 4 = 0 ∧ x₂^2 - 2*x₂ - 4 = 0) ∧
  (∀ x : ℝ, x ≠ 4 → ((x - 5) / (x - 4) = 1 - x / (4 - x)) ↔ x = -1) :=
by sorry

end quadratic_and_fractional_equations_l3403_340305


namespace min_beta_delta_sum_l3403_340357

open Complex

/-- A complex-valued function with specific properties -/
def f (β δ : ℂ) (z : ℂ) : ℂ := (3 + 2*I)*z^2 + β*z + δ

/-- The theorem stating the minimum value of |β| + |δ| -/
theorem min_beta_delta_sum :
  ∃ (β δ : ℂ), 
    (∀ (β' δ' : ℂ), (f β' δ' (1 + I)).im = 0 ∧ (f β' δ' (-I)).im = 0 → 
      Complex.abs β + Complex.abs δ ≤ Complex.abs β' + Complex.abs δ') ∧
    Complex.abs β + Complex.abs δ = Real.sqrt 5 + 3 := by
  sorry

end min_beta_delta_sum_l3403_340357


namespace f_log2_32_equals_17_l3403_340308

noncomputable def f (x : ℝ) : ℝ :=
  if x < 4 then Real.log 4 / Real.log 2
  else 1 + 2^(x - 1)

theorem f_log2_32_equals_17 : f (Real.log 32 / Real.log 2) = 17 := by
  sorry

end f_log2_32_equals_17_l3403_340308


namespace impossible_equal_checkers_l3403_340352

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Represents an L-shape on the grid -/
inductive LShape
  | topLeft : LShape
  | topRight : LShape
  | bottomLeft : LShape
  | bottomRight : LShape

/-- Applies a move to the grid -/
def applyMove (grid : Grid) (shape : LShape) : Grid :=
  sorry

/-- Checks if all cells in the grid have the same non-zero value -/
def allCellsSame (grid : Grid) : Prop :=
  sorry

/-- Theorem stating the impossibility of reaching a state where all cells have the same non-zero value -/
theorem impossible_equal_checkers :
  ¬ ∃ (initial : Grid) (moves : List LShape),
    (∀ i j, initial i j = 0) ∧ 
    allCellsSame (moves.foldl applyMove initial) :=
  sorry

end impossible_equal_checkers_l3403_340352


namespace sufficient_fabric_l3403_340330

/-- Represents the dimensions of a rectangular piece of fabric -/
structure FabricDimensions where
  length : ℕ
  width : ℕ

/-- Checks if a piece of fabric can be cut into at least n smaller pieces -/
def canCutInto (fabric : FabricDimensions) (piece : FabricDimensions) (n : ℕ) : Prop :=
  ∃ (l w : ℕ), 
    l * piece.length ≤ fabric.length ∧ 
    w * piece.width ≤ fabric.width ∧ 
    l * w ≥ n

theorem sufficient_fabric : 
  let fabric := FabricDimensions.mk 140 75
  let dress := FabricDimensions.mk 45 26
  canCutInto fabric dress 8 := by
  sorry

end sufficient_fabric_l3403_340330


namespace cone_lateral_surface_area_l3403_340398

/-- Lateral surface area of a cone with given base radius and volume -/
theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 3) (hv : (1/3) * π * r^2 * h = 12 * π) :
  π * r * (Real.sqrt (r^2 + h^2)) = 15 * π := by
  sorry

end cone_lateral_surface_area_l3403_340398


namespace propositions_p_and_q_l3403_340327

theorem propositions_p_and_q : 
  (∀ x : ℝ, (2 : ℝ)^x < (3 : ℝ)^x) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := by
  sorry

end propositions_p_and_q_l3403_340327


namespace intersection_when_a_is_neg_two_intersection_equals_A_iff_l3403_340334

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x + a < 0}

-- Theorem 1: When a = -2, A ∩ B = {x | 1/2 ≤ x < 2}
theorem intersection_when_a_is_neg_two :
  A ∩ B (-2) = {x : ℝ | 1/2 ≤ x ∧ x < 2} := by sorry

-- Theorem 2: A ∩ B = A if and only if a < -3
theorem intersection_equals_A_iff (a : ℝ) :
  A ∩ B a = A ↔ a < -3 := by sorry

end intersection_when_a_is_neg_two_intersection_equals_A_iff_l3403_340334


namespace limit_sin_squared_minus_tan_squared_over_x_fourth_l3403_340303

open Real

theorem limit_sin_squared_minus_tan_squared_over_x_fourth : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((sin x)^2 - (tan x)^2) / x^4 + 1| < ε :=
by sorry

end limit_sin_squared_minus_tan_squared_over_x_fourth_l3403_340303


namespace randys_trip_distance_l3403_340372

theorem randys_trip_distance :
  ∀ y : ℝ,
  (y / 4 : ℝ) + 30 + (y / 3 : ℝ) = y →
  y = 72 := by
sorry

end randys_trip_distance_l3403_340372


namespace remainder_problem_l3403_340346

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k % 5 = 2) 
  (h3 : k % 6 = 5) 
  (h4 : k < 42) : 
  k % 7 = 3 := by
sorry

end remainder_problem_l3403_340346


namespace absent_men_calculation_l3403_340395

/-- Represents the number of men who became absent -/
def absentMen (totalMen originalDays actualDays : ℕ) : ℕ :=
  totalMen - (totalMen * originalDays) / actualDays

theorem absent_men_calculation (totalMen originalDays actualDays : ℕ) 
  (h1 : totalMen = 15)
  (h2 : originalDays = 8)
  (h3 : actualDays = 10)
  (h4 : totalMen > 0)
  (h5 : originalDays > 0)
  (h6 : actualDays > 0)
  (h7 : (totalMen * originalDays) % actualDays = 0) :
  absentMen totalMen originalDays actualDays = 3 := by
  sorry

#eval absentMen 15 8 10

end absent_men_calculation_l3403_340395


namespace annual_interest_rate_l3403_340382

/-- Calculate the annual interest rate given the borrowed amount and repayment amount after one year. -/
theorem annual_interest_rate (borrowed : ℝ) (repaid : ℝ) (h1 : borrowed = 150) (h2 : repaid = 165) :
  (repaid - borrowed) / borrowed * 100 = 10 := by
  sorry

end annual_interest_rate_l3403_340382


namespace rectangular_box_dimensions_l3403_340311

theorem rectangular_box_dimensions (X Y Z : ℝ) 
  (h1 : X * Y = 32)
  (h2 : X * Z = 50)
  (h3 : Y * Z = 80) :
  X + Y + Z = 25.5 * Real.sqrt 2 := by
  sorry

end rectangular_box_dimensions_l3403_340311


namespace min_cost_for_family_trip_l3403_340341

/-- Represents the ticket prices in rubles -/
structure TicketPrices where
  adult_single : ℕ
  child_single : ℕ
  day_pass_single : ℕ
  day_pass_group : ℕ
  three_day_pass_single : ℕ
  three_day_pass_group : ℕ

/-- Calculates the minimum cost for a family's subway tickets -/
def min_family_ticket_cost (prices : TicketPrices) (days : ℕ) (trips_per_day : ℕ) (adults : ℕ) (children : ℕ) : ℕ :=
  sorry

/-- The theorem stating the minimum cost for the given family and conditions -/
theorem min_cost_for_family_trip (prices : TicketPrices) 
  (h1 : prices.adult_single = 40)
  (h2 : prices.child_single = 20)
  (h3 : prices.day_pass_single = 350)
  (h4 : prices.day_pass_group = 1500)
  (h5 : prices.three_day_pass_single = 900)
  (h6 : prices.three_day_pass_group = 3500) :
  min_family_ticket_cost prices 5 10 2 2 = 5200 :=
by sorry

end min_cost_for_family_trip_l3403_340341


namespace first_or_third_quadrant_set_l3403_340364

def first_or_third_quadrant (α : ℝ) : Prop :=
  (∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2) ∨
  (∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2)

theorem first_or_third_quadrant_set : 
  {α : ℝ | first_or_third_quadrant α} = 
  {α : ℝ | ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2} ∪
  {α : ℝ | ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2} :=
by sorry

end first_or_third_quadrant_set_l3403_340364


namespace sixth_grade_boys_count_l3403_340394

/-- Represents the set of boys in the 6th "A" grade. -/
def Boys : Type := Unit

/-- Represents the set of girls in the 6th "A" grade. -/
inductive Girls : Type
  | tanya : Girls
  | dasha : Girls
  | katya : Girls

/-- Represents the friendship relation between boys and girls. -/
def IsFriend : Boys → Girls → Prop := sorry

/-- The number of boys in the 6th "A" grade. -/
def numBoys : ℕ := sorry

theorem sixth_grade_boys_count :
  (∀ (b1 b2 b3 : Boys), ∃ (g : Girls), IsFriend b1 g ∨ IsFriend b2 g ∨ IsFriend b3 g) →
  (∃ (boys : Finset Boys), Finset.card boys = 12 ∧ ∀ b ∈ boys, IsFriend b Girls.tanya) →
  (∃ (boys : Finset Boys), Finset.card boys = 12 ∧ ∀ b ∈ boys, IsFriend b Girls.dasha) →
  (∃ (boys : Finset Boys), Finset.card boys = 13 ∧ ∀ b ∈ boys, IsFriend b Girls.katya) →
  numBoys = 13 ∨ numBoys = 14 := by
  sorry

end sixth_grade_boys_count_l3403_340394


namespace solution_implies_a_value_l3403_340349

theorem solution_implies_a_value (a : ℝ) :
  (5 * a - 8 = 10 + 4 * a) → a = 18 := by
  sorry

end solution_implies_a_value_l3403_340349


namespace product_of_g_at_roots_of_f_l3403_340360

theorem product_of_g_at_roots_of_f (x₁ x₂ x₃ x₄ x₅ : ℂ) : 
  (x₁^5 - x₁^3 + 2*x₁^2 + 1 = 0) →
  (x₂^5 - x₂^3 + 2*x₂^2 + 1 = 0) →
  (x₃^5 - x₃^3 + 2*x₃^2 + 1 = 0) →
  (x₄^5 - x₄^3 + 2*x₄^2 + 1 = 0) →
  (x₅^5 - x₅^3 + 2*x₅^2 + 1 = 0) →
  (x₁^2 - 3) * (x₂^2 - 3) * (x₃^2 - 3) * (x₄^2 - 3) * (x₅^2 - 3) = -59 :=
by sorry

end product_of_g_at_roots_of_f_l3403_340360


namespace greatest_common_divisor_under_30_l3403_340375

theorem greatest_common_divisor_under_30 : ∃ (n : ℕ), n = 18 ∧ 
  n ∣ 540 ∧ n < 30 ∧ n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 → m < 30 → m ∣ 180 → m ≤ n :=
by sorry

end greatest_common_divisor_under_30_l3403_340375


namespace chris_sick_one_week_l3403_340347

/-- Calculates the number of weeks Chris got sick based on Cathy's work hours -/
def weeks_chris_sick (hours_per_week : ℕ) (total_weeks : ℕ) (cathy_total_hours : ℕ) : ℕ :=
  (cathy_total_hours - (hours_per_week * total_weeks)) / hours_per_week

/-- Proves that Chris got sick for 1 week given the conditions in the problem -/
theorem chris_sick_one_week :
  let hours_per_week : ℕ := 20
  let months : ℕ := 2
  let weeks_per_month : ℕ := 4
  let total_weeks : ℕ := months * weeks_per_month
  let cathy_total_hours : ℕ := 180
  weeks_chris_sick hours_per_week total_weeks cathy_total_hours = 1 := by
  sorry

#eval weeks_chris_sick 20 8 180

end chris_sick_one_week_l3403_340347


namespace M_superset_P_l3403_340396

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x^2 - 4}

-- Define the set P
def P : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}

-- Define the transformation function
def f (x : ℝ) : ℝ := x^2 - 4

-- Theorem statement
theorem M_superset_P : M ⊇ f '' P := by sorry

end M_superset_P_l3403_340396


namespace rectangle_area_l3403_340336

/-- Given a rectangle with length L and width W, if 2L + W = 34 and L + 2W = 38, then the area of the rectangle is 140. -/
theorem rectangle_area (L W : ℝ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) :
  L * W = 140 := by sorry

end rectangle_area_l3403_340336


namespace lowest_unique_score_l3403_340384

/-- The scoring function for the modified AHSME -/
def score (c w : ℕ) : ℤ := 30 + 4 * c - 2 * w

/-- Predicate to check if a score uniquely determines c and w -/
def uniquely_determines (s : ℤ) : Prop :=
  ∃! (c w : ℕ), score c w = s ∧ c + w ≤ 30

theorem lowest_unique_score : 
  (∀ s : ℤ, 100 < s → s < 116 → ¬ uniquely_determines s) ∧
  uniquely_determines 116 := by
  sorry

end lowest_unique_score_l3403_340384


namespace arithmetic_sequence_general_term_l3403_340365

/-- An arithmetic sequence with first term a₁ and common ratio q -/
structure ArithmeticSequence (α : Type*) [Semiring α] where
  a₁ : α
  q : α

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm {α : Type*} [Semiring α] (seq : ArithmeticSequence α) (n : ℕ) : α :=
  seq.a₁ * seq.q ^ (n - 1)

/-- Theorem: The general term of an arithmetic sequence -/
theorem arithmetic_sequence_general_term {α : Type*} [Semiring α] (seq : ArithmeticSequence α) (n : ℕ) :
  seq.nthTerm n = seq.a₁ * seq.q ^ (n - 1) := by
  sorry

end arithmetic_sequence_general_term_l3403_340365


namespace divisible_by_36_sum_6_l3403_340325

/-- Represents a 7-digit number in the form 457q89f -/
def number (q f : Nat) : Nat :=
  457000 + q * 1000 + 89 * 10 + f

/-- Predicate to check if two natural numbers are distinct digits -/
def distinct_digits (a b : Nat) : Prop :=
  a ≠ b ∧ a < 10 ∧ b < 10

theorem divisible_by_36_sum_6 (q f : Nat) :
  distinct_digits q f →
  number q f % 36 = 0 →
  q + f = 6 := by
sorry

end divisible_by_36_sum_6_l3403_340325


namespace largest_angle_in_special_triangle_l3403_340320

/-- Given a scalene triangle with angles in the ratio 1:2:3 and the smallest angle being 30°,
    the largest angle is 90°. -/
theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c →  -- angles are positive
    a < b ∧ b < c →  -- scalene triangle condition
    a + b + c = 180 →  -- sum of angles in a triangle
    b = 2*a ∧ c = 3*a →  -- ratio of angles is 1:2:3
    a = 30 →  -- smallest angle is 30°
    c = 90 := by
  sorry

end largest_angle_in_special_triangle_l3403_340320


namespace range_of_a_l3403_340361

theorem range_of_a (a : ℝ) : 
  ((∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
   (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0)) ↔ 
  (a ≤ -2 ∨ a = 1) := by
sorry

end range_of_a_l3403_340361


namespace hiking_time_theorem_l3403_340354

/-- Calculates the total time for a hiker to return to the starting point given their hiking rate and distances. -/
def total_hiking_time (rate : ℝ) (initial_distance : ℝ) (total_distance : ℝ) : ℝ :=
  let additional_distance := total_distance - initial_distance
  let time_additional := additional_distance * rate
  let time_return := total_distance * rate
  time_additional + time_return

/-- Theorem stating that under given conditions, the total hiking time is 40 minutes. -/
theorem hiking_time_theorem (rate : ℝ) (initial_distance : ℝ) (total_distance : ℝ) :
  rate = 12 →
  initial_distance = 2.75 →
  total_distance = 3.041666666666667 →
  total_hiking_time rate initial_distance total_distance = 40 := by
  sorry

#eval total_hiking_time 12 2.75 3.041666666666667

end hiking_time_theorem_l3403_340354


namespace remaining_balloons_l3403_340356

-- Define the type for balloon labels
inductive BalloonLabel
| A | B | C | D | E | F | G | H | I | J | K | L

-- Define the function to get the next balloon to pop
def nextBalloon (current : BalloonLabel) : BalloonLabel :=
  match current with
  | BalloonLabel.A => BalloonLabel.D
  | BalloonLabel.B => BalloonLabel.E
  | BalloonLabel.C => BalloonLabel.F
  | BalloonLabel.D => BalloonLabel.G
  | BalloonLabel.E => BalloonLabel.H
  | BalloonLabel.F => BalloonLabel.I
  | BalloonLabel.G => BalloonLabel.J
  | BalloonLabel.H => BalloonLabel.K
  | BalloonLabel.I => BalloonLabel.L
  | BalloonLabel.J => BalloonLabel.A
  | BalloonLabel.K => BalloonLabel.B
  | BalloonLabel.L => BalloonLabel.C

-- Define the function to pop balloons
def popBalloons (start : BalloonLabel) (n : Nat) : List BalloonLabel :=
  if n = 0 then []
  else start :: popBalloons (nextBalloon (nextBalloon start)) (n - 1)

-- Theorem statement
theorem remaining_balloons :
  popBalloons BalloonLabel.C 10 = [BalloonLabel.C, BalloonLabel.F, BalloonLabel.I, BalloonLabel.L, BalloonLabel.D, BalloonLabel.H, BalloonLabel.A, BalloonLabel.G, BalloonLabel.B, BalloonLabel.K] ∧
  (∀ b : BalloonLabel, b ∉ popBalloons BalloonLabel.C 10 → b = BalloonLabel.E ∨ b = BalloonLabel.J) :=
by sorry


end remaining_balloons_l3403_340356


namespace train_length_l3403_340353

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 → time = 30 → speed * time * (5 / 18) = 300 :=
by sorry

end train_length_l3403_340353


namespace a_union_b_iff_c_l3403_340389

-- Define the sets A, B, and C
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

-- State the theorem
theorem a_union_b_iff_c : ∀ x : ℝ, x ∈ (A ∪ B) ↔ x ∈ C := by
  sorry

end a_union_b_iff_c_l3403_340389


namespace periodic_function_l3403_340387

def isPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f (x + 1) + f (x - 1) = Real.sqrt 2 * f x) : 
  isPeriodic f := by
  sorry

end periodic_function_l3403_340387


namespace sum_of_data_l3403_340317

theorem sum_of_data (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a = 12) :
  a + b + c = 96 := by
  sorry

end sum_of_data_l3403_340317


namespace investment_average_rate_l3403_340331

/-- Proves that given a total investment split between two schemes with different rates,
    if the annual returns from both parts are equal, then the average rate of interest
    on the total investment is as calculated. -/
theorem investment_average_rate
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (h_total : total_investment = 5000)
  (h_rates : rate1 = 0.03 ∧ rate2 = 0.05)
  (h_equal_returns : ∃ (x : ℝ), x ≥ 0 ∧ x ≤ total_investment ∧
    rate1 * (total_investment - x) = rate2 * x) :
  (rate1 * (total_investment - x) + rate2 * x) / total_investment = 0.0375 :=
sorry

end investment_average_rate_l3403_340331


namespace f_derivative_at_zero_l3403_340343

def f (x : ℝ) : ℝ := x*(x+1)*(x+2)*(x+3)*(x+4)*(x+5) + 6

theorem f_derivative_at_zero : 
  deriv f 0 = 120 := by sorry

end f_derivative_at_zero_l3403_340343


namespace complex_number_i_properties_l3403_340337

/-- Given a complex number i such that i^2 = -1, prove the properties of i raised to different powers -/
theorem complex_number_i_properties (i : ℂ) (n : ℕ) (h : i^2 = -1) :
  i^(4*n + 1) = i ∧ i^(4*n + 2) = -1 ∧ i^(4*n + 3) = -i := by
  sorry

end complex_number_i_properties_l3403_340337


namespace complex_simplification_l3403_340368

theorem complex_simplification :
  (7 * (4 - 2 * Complex.I) + 4 * Complex.I * (7 - 3 * Complex.I)) = (40 : ℂ) + 14 * Complex.I :=
by sorry

end complex_simplification_l3403_340368


namespace remaining_files_l3403_340397

theorem remaining_files (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 13)
  (h2 : video_files = 30)
  (h3 : deleted_files = 10) :
  music_files + video_files - deleted_files = 33 := by
  sorry

end remaining_files_l3403_340397


namespace sum_of_triangle_and_rectangle_edges_l3403_340381

/-- The number of edges in a triangle -/
def triangle_edges : ℕ := 3

/-- The number of edges in a rectangle -/
def rectangle_edges : ℕ := 4

/-- The sum of edges in a triangle and a rectangle -/
def total_edges : ℕ := triangle_edges + rectangle_edges

theorem sum_of_triangle_and_rectangle_edges :
  total_edges = 7 := by sorry

end sum_of_triangle_and_rectangle_edges_l3403_340381


namespace travel_ways_proof_l3403_340377

/-- The number of roads from village A to village B -/
def roads_A_to_B : ℕ := 3

/-- The number of roads from village B to village C -/
def roads_B_to_C : ℕ := 2

/-- The total number of ways to travel from village A to village C via village B -/
def total_ways : ℕ := roads_A_to_B * roads_B_to_C

theorem travel_ways_proof : total_ways = 6 := by
  sorry

end travel_ways_proof_l3403_340377


namespace sqrt_sum_ge_product_sum_l3403_340370

theorem sqrt_sum_ge_product_sum {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 3) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + c * a := by
  sorry

end sqrt_sum_ge_product_sum_l3403_340370


namespace test_question_percentage_l3403_340306

theorem test_question_percentage (second_correct : ℝ) (neither_correct : ℝ) (both_correct : ℝ)
  (h1 : second_correct = 0.55)
  (h2 : neither_correct = 0.20)
  (h3 : both_correct = 0.50) :
  ∃ first_correct : ℝ,
    first_correct = 0.75 ∧
    first_correct + second_correct - both_correct + neither_correct = 1 :=
by sorry

end test_question_percentage_l3403_340306


namespace div_point_one_eq_mul_ten_l3403_340378

theorem div_point_one_eq_mul_ten (a : ℝ) : a / 0.1 = a * 10 := by sorry

end div_point_one_eq_mul_ten_l3403_340378


namespace power_sum_equality_l3403_340323

theorem power_sum_equality : (-2)^1999 + (-2)^2000 = 2^1999 := by
  sorry

end power_sum_equality_l3403_340323


namespace infinite_geometric_series_ratio_specific_geometric_series_ratio_l3403_340335

/-- For an infinite geometric series with first term a and sum S, 
    the common ratio r is given by r = 1 - (a / S) -/
theorem infinite_geometric_series_ratio 
  (a : ℝ) (S : ℝ) (h1 : a > 0) (h2 : S > a) :
  let r := 1 - (a / S)
  (∀ n : ℕ, a * r^n = a * (1 - a/S)^n) ∧ 
  (∑' n, a * r^n = S) →
  r = (S - a) / S :=
sorry

/-- The common ratio of an infinite geometric series with 
    first term 520 and sum 3250 is 273/325 -/
theorem specific_geometric_series_ratio :
  let a : ℝ := 520
  let S : ℝ := 3250
  let r := 1 - (a / S)
  (∀ n : ℕ, a * r^n = 520 * (1 - 520/3250)^n) ∧ 
  (∑' n, a * r^n = 3250) →
  r = 273 / 325 :=
sorry

end infinite_geometric_series_ratio_specific_geometric_series_ratio_l3403_340335


namespace polynomial_equality_sum_of_squares_l3403_340383

theorem polynomial_equality_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by sorry

end polynomial_equality_sum_of_squares_l3403_340383


namespace binomial_coefficient_problem_l3403_340345

theorem binomial_coefficient_problem (m : ℕ+) 
  (a b : ℕ) 
  (ha : a = Nat.choose (2 * m) m)
  (hb : b = Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 := by
sorry

end binomial_coefficient_problem_l3403_340345


namespace imo_1993_function_exists_l3403_340393

/-- A strictly increasing function from positive integers to positive integers -/
def StrictlyIncreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m < n → f m < f n

/-- The existence of a function satisfying the IMO 1993 conditions -/
theorem imo_1993_function_exists : ∃ f : ℕ+ → ℕ+, 
  f 1 = 2 ∧ 
  StrictlyIncreasing f ∧ 
  ∀ n : ℕ+, f (f n) = f n + n :=
sorry

end imo_1993_function_exists_l3403_340393


namespace solve_birthday_money_problem_l3403_340301

def birthday_money_problem (aunt uncle friend1 friend2 friend3 sister : ℝ)
  (mean : ℝ) (total_gifts : ℕ) (unknown_gift : ℝ) : Prop :=
  aunt = 9 ∧
  uncle = 9 ∧
  friend1 = 22 ∧
  friend2 = 22 ∧
  friend3 = 22 ∧
  sister = 7 ∧
  mean = 16.3 ∧
  total_gifts = 7 ∧
  (aunt + uncle + friend1 + unknown_gift + friend2 + friend3 + sister) / total_gifts = mean ∧
  unknown_gift = 23.1

theorem solve_birthday_money_problem :
  ∃ (aunt uncle friend1 friend2 friend3 sister : ℝ)
    (mean : ℝ) (total_gifts : ℕ) (unknown_gift : ℝ),
  birthday_money_problem aunt uncle friend1 friend2 friend3 sister mean total_gifts unknown_gift :=
by sorry

end solve_birthday_money_problem_l3403_340301


namespace problem_statement_l3403_340315

/-- Given a function f(x) = ax^5 + bx^3 + cx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem problem_statement (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x - 8)
    (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end problem_statement_l3403_340315


namespace square_side_lengths_average_l3403_340322

theorem square_side_lengths_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 16) (h₂ : a₂ = 49) (h₃ : a₃ = 169) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 8 := by
  sorry

end square_side_lengths_average_l3403_340322


namespace triangle_area_l3403_340351

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c^2 = (a - b)^2 + 6 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by sorry

end triangle_area_l3403_340351


namespace vacation_days_calculation_l3403_340313

theorem vacation_days_calculation (families : Nat) (people_per_family : Nat) 
  (towels_per_person_per_day : Nat) (towels_per_load : Nat) (total_loads : Nat) :
  families = 3 →
  people_per_family = 4 →
  towels_per_person_per_day = 1 →
  towels_per_load = 14 →
  total_loads = 6 →
  (total_loads * towels_per_load) / (families * people_per_family * towels_per_person_per_day) = 7 := by
  sorry

end vacation_days_calculation_l3403_340313


namespace max_a_squared_b_l3403_340302

theorem max_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) :
  a^2 * b ≤ 54 := by
sorry

end max_a_squared_b_l3403_340302


namespace annes_speed_l3403_340373

/-- Given a distance of 6 miles traveled in 3 hours, prove that the speed is 2 miles per hour. -/
theorem annes_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 6 → time = 3 → speed = distance / time → speed = 2 := by sorry

end annes_speed_l3403_340373


namespace initial_marbles_count_l3403_340314

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The total number of marbles Carla has now -/
def total_marbles_now : ℕ := 187

/-- The initial number of marbles Carla had -/
def initial_marbles : ℕ := total_marbles_now - marbles_bought

theorem initial_marbles_count : initial_marbles = 53 := by
  sorry

end initial_marbles_count_l3403_340314


namespace median_mean_difference_l3403_340388

structure ArticleData where
  frequencies : List (Nat × Nat)
  total_students : Nat
  sum_articles : Nat

def median (data : ArticleData) : Rat := 2

def mean (data : ArticleData) : Rat := data.sum_articles / data.total_students

theorem median_mean_difference (data : ArticleData) 
  (h1 : data.frequencies = [(0, 4), (1, 3), (2, 2), (3, 2), (4, 3), (5, 4)])
  (h2 : data.total_students = 18)
  (h3 : data.sum_articles = 45) :
  mean data - median data = 1/2 := by sorry

end median_mean_difference_l3403_340388


namespace parallel_lines_condition_l3403_340358

/-- Given two lines in the real plane, determine if a specific value of a parameter is sufficient but not necessary for their parallelism. -/
theorem parallel_lines_condition (a : ℝ) : 
  (∃ (x y : ℝ), a * x + 2 * y - 1 = 0) →  -- l₁ exists
  (∃ (x y : ℝ), x + (a + 1) * y + 4 = 0) →  -- l₂ exists
  (a = 1 → (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    a * x₁ + 2 * y₁ - 1 = 0 → 
    x₂ + (a + 1) * y₂ + 4 = 0 → 
    (y₂ - y₁) * (1 - 0) = (x₂ - x₁) * (2 - (a + 1)))) ∧ 
  (∃ b : ℝ, b ≠ 1 ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), 
      b * x₁ + 2 * y₁ - 1 = 0 → 
      x₂ + (b + 1) * y₂ + 4 = 0 → 
      (y₂ - y₁) * (1 - 0) = (x₂ - x₁) * (2 - (b + 1)))) :=
by sorry

end parallel_lines_condition_l3403_340358


namespace pyarelal_loss_l3403_340399

theorem pyarelal_loss (p a : ℝ) (total_loss : ℝ) : 
  a = (1 / 9) * p → 
  total_loss = 900 → 
  (p / (p + a)) * total_loss = 810 :=
by sorry

end pyarelal_loss_l3403_340399


namespace triangle_is_equilateral_l3403_340362

theorem triangle_is_equilateral (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + b^2 = c^2 + a*b →
  Real.cos A * Real.cos B = 1/4 →
  A = B ∧ B = C ∧ C = π/3 :=
by sorry

end triangle_is_equilateral_l3403_340362


namespace snowdrift_depth_change_l3403_340344

/-- Given a snowdrift with certain depth changes over four days, 
    calculate the amount of snow added on the fourth day. -/
theorem snowdrift_depth_change (initial_depth final_depth third_day_addition : ℕ) : 
  initial_depth = 20 →
  final_depth = 34 →
  third_day_addition = 6 →
  final_depth - (initial_depth / 2 + third_day_addition) = 18 := by
  sorry

#check snowdrift_depth_change

end snowdrift_depth_change_l3403_340344


namespace age_ratio_is_two_to_one_l3403_340371

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- Conditions for the ages -/
def age_conditions (a : Ages) : Prop :=
  a.roy = a.julia + 6 ∧
  a.roy + 2 = 2 * (a.julia + 2) ∧
  (a.roy + 2) * (a.kelly + 2) = 108

/-- The theorem to be proved -/
theorem age_ratio_is_two_to_one (a : Ages) :
  age_conditions a →
  (a.roy - a.julia) / (a.roy - a.kelly) = 2 := by
  sorry

#check age_ratio_is_two_to_one

end age_ratio_is_two_to_one_l3403_340371


namespace difference_of_squares_l3403_340355

theorem difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end difference_of_squares_l3403_340355


namespace crossing_over_result_l3403_340392

/-- Represents a chromatid with its staining pattern -/
structure Chromatid where
  staining : ℕ → Bool  -- True for darker staining, False for lighter

/-- Represents a chromosome with two sister chromatids -/
structure Chromosome where
  chromatid1 : Chromatid
  chromatid2 : Chromatid

/-- Represents the process of DNA replication with BrdU -/
def dnaReplication (c : Chromosome) : Chromosome :=
  { chromatid1 := { staining := fun _ => true },
    chromatid2 := c.chromatid1 }

/-- Represents the process of crossing over between sister chromatids -/
def crossingOver (c : Chromosome) : Chromosome :=
  { chromatid1 := { staining := fun n => if n % 2 = 0 then c.chromatid1.staining n else c.chromatid2.staining n },
    chromatid2 := { staining := fun n => if n % 2 = 0 then c.chromatid2.staining n else c.chromatid1.staining n } }

/-- Theorem stating the result of the experiment -/
theorem crossing_over_result (initialChromosome : Chromosome) :
  ∃ (n m : ℕ), 
    let finalChromosome := crossingOver (dnaReplication (dnaReplication initialChromosome))
    finalChromosome.chromatid1.staining n ≠ finalChromosome.chromatid1.staining m ∧
    finalChromosome.chromatid2.staining n ≠ finalChromosome.chromatid2.staining m :=
  sorry


end crossing_over_result_l3403_340392


namespace right_triangle_area_l3403_340385

/-- A right-angled triangle with specific properties -/
structure RightTriangle where
  -- The legs of the triangle
  a : ℝ
  b : ℝ
  -- The hypotenuse of the triangle
  c : ℝ
  -- Conditions
  right_angle : a^2 + b^2 = c^2
  perimeter : a + b + c = 2 + Real.sqrt 6
  hypotenuse : c = 2
  median : (a + b) / 2 = 1

/-- The area of a right-angled triangle with the given properties is 1/2 -/
theorem right_triangle_area (t : RightTriangle) : (t.a * t.b) / 2 = 1/2 := by
  sorry

end right_triangle_area_l3403_340385


namespace goose_egg_hatch_fraction_l3403_340333

theorem goose_egg_hatch_fraction (total_eggs : ℕ) (survived_year : ℕ) 
  (h1 : total_eggs = 550)
  (h2 : survived_year = 110)
  (h3 : ∀ x : ℚ, x * total_eggs * (3/4 : ℚ) * (2/5 : ℚ) = survived_year → x = 2/3) :
  ∃ x : ℚ, x * total_eggs = (total_eggs : ℚ) * (2/3 : ℚ) := by sorry

end goose_egg_hatch_fraction_l3403_340333


namespace savings_equality_l3403_340324

/-- Prove that A's savings equal B's savings given the conditions -/
theorem savings_equality (total_salary : ℝ) (a_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ)
  (h1 : total_salary = 7000)
  (h2 : a_salary = 5250)
  (h3 : a_spend_rate = 0.95)
  (h4 : b_spend_rate = 0.85) :
  a_salary * (1 - a_spend_rate) = (total_salary - a_salary) * (1 - b_spend_rate) :=
by
  sorry

end savings_equality_l3403_340324


namespace power_of_complex_root_of_unity_l3403_340359

open Complex

theorem power_of_complex_root_of_unity : ((1 - I) / (Real.sqrt 2)) ^ 20 = 1 := by
  sorry

end power_of_complex_root_of_unity_l3403_340359


namespace cube_equation_solution_l3403_340307

theorem cube_equation_solution (c : ℤ) : 
  c^3 + 3*c + 3/c + 1/c^3 = 8 → c = 1 := by
  sorry

end cube_equation_solution_l3403_340307


namespace circle_area_quadrupled_l3403_340390

theorem circle_area_quadrupled (r n : ℝ) : 
  (r > 0) → (n > 0) → (π * (r + n)^2 = 4 * π * r^2) → r = n / 3 :=
by sorry

end circle_area_quadrupled_l3403_340390


namespace cube_painting_theorem_l3403_340309

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- The number of ways to paint all faces of a cube the same color -/
def all_same_color : ℕ := num_colors

/-- The number of ways to paint 5 faces the same color and 1 face a different color -/
def five_same_one_different : ℕ := num_faces * (num_colors - 1)

/-- The number of ways to paint all faces of a cube different colors, considering rotational symmetry -/
def all_different_colors : ℕ := (Nat.factorial num_colors) / cube_symmetries

theorem cube_painting_theorem :
  all_same_color = 6 ∧
  five_same_one_different = 30 ∧
  all_different_colors = 30 := by
  sorry

end cube_painting_theorem_l3403_340309


namespace rabbit_population_l3403_340312

theorem rabbit_population (breeding_rabbits : ℕ) (first_spring_ratio : ℕ) 
  (second_spring_kittens : ℕ) (second_spring_adopted : ℕ) (total_rabbits : ℕ) :
  breeding_rabbits = 10 →
  second_spring_kittens = 60 →
  second_spring_adopted = 4 →
  total_rabbits = 121 →
  breeding_rabbits + (first_spring_ratio * breeding_rabbits / 2 + 5) + 
    (second_spring_kittens - second_spring_adopted) = total_rabbits →
  first_spring_ratio = 10 := by
sorry

end rabbit_population_l3403_340312


namespace perpendicular_slope_l3403_340379

/-- Given a line with equation 5x - 2y = 10, 
    the slope of the perpendicular line is -2/5 -/
theorem perpendicular_slope (x y : ℝ) :
  (5 * x - 2 * y = 10) → 
  (∃ m : ℝ, m = -2/5 ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      (5 * x₁ - 2 * y₁ = 10) → 
      (5 * x₂ - 2 * y₂ = 10) → 
      x₁ ≠ x₂ → 
      m * ((x₂ - x₁) / (y₂ - y₁)) = -1) :=
by sorry

end perpendicular_slope_l3403_340379


namespace single_loop_probability_six_threads_l3403_340319

/-- Represents the game with threads and pairings -/
structure ThreadGame where
  num_threads : ℕ
  num_pairs : ℕ

/-- Calculates the total number of possible pairings -/
def total_pairings (game : ThreadGame) : ℕ :=
  (2 * game.num_threads - 1) * (2 * game.num_threads - 3)

/-- Calculates the number of pairings that form a single loop -/
def single_loop_pairings (game : ThreadGame) : ℕ :=
  (2 * game.num_threads - 2) * (game.num_threads - 1)

/-- Theorem stating the probability of forming a single loop in the game with 6 threads -/
theorem single_loop_probability_six_threads :
  let game : ThreadGame := { num_threads := 6, num_pairs := 3 }
  (single_loop_pairings game : ℚ) / (total_pairings game) = 8 / 15 := by
  sorry


end single_loop_probability_six_threads_l3403_340319


namespace quadratic_root_property_l3403_340304

theorem quadratic_root_property (m : ℝ) : 
  m^2 - 3*m + 1 = 0 → 2*m^2 - 6*m - 2024 = -2026 := by
sorry

end quadratic_root_property_l3403_340304


namespace percentage_difference_l3403_340310

theorem percentage_difference : 
  (0.80 * 40) - ((4 / 5) * 20) = 16 := by
  sorry

end percentage_difference_l3403_340310


namespace juvy_chives_count_l3403_340369

/-- Calculates the number of chives planted in Juvy's garden. -/
def chives_count (total_rows : ℕ) (plants_per_row : ℕ) (parsley_rows : ℕ) (rosemary_rows : ℕ) : ℕ :=
  (total_rows - (parsley_rows + rosemary_rows)) * plants_per_row

/-- Theorem stating that the number of chives Juvy will plant is 150. -/
theorem juvy_chives_count :
  chives_count 20 10 3 2 = 150 := by
  sorry

end juvy_chives_count_l3403_340369


namespace x_divisibility_l3403_340332

def x : ℤ := 64 + 96 + 128 + 160 + 288 + 352 + 3232

theorem x_divisibility :
  (∃ k : ℤ, x = 4 * k) ∧
  (∃ k : ℤ, x = 8 * k) ∧
  (∃ k : ℤ, x = 16 * k) ∧
  (∃ k : ℤ, x = 32 * k) :=
by sorry

end x_divisibility_l3403_340332
