import Mathlib

namespace NUMINAMATH_GPT_Agnes_birth_year_l504_50473

theorem Agnes_birth_year (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9)
  (h3 : (11 * x + 2 * y + x * y = 92)) : 1948 = 1900 + (10 * x + y) :=
sorry

end NUMINAMATH_GPT_Agnes_birth_year_l504_50473


namespace NUMINAMATH_GPT_simplify_fraction_l504_50468

theorem simplify_fraction : (48 / 72 : ℚ) = (2 / 3) := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l504_50468


namespace NUMINAMATH_GPT_sector_area_eq_13pi_l504_50484

theorem sector_area_eq_13pi
    (O A B C : Type)
    (r : ℝ)
    (θ : ℝ)
    (h1 : θ = 130)
    (h2 : r = 6) :
    (θ / 360) * (π * r^2) = 13 * π := by
  sorry

end NUMINAMATH_GPT_sector_area_eq_13pi_l504_50484


namespace NUMINAMATH_GPT_max_d_for_range_of_fx_l504_50479

theorem max_d_for_range_of_fx : 
  ∀ (d : ℝ), (∃ x : ℝ, x^2 + 4*x + d = -3) → d ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_max_d_for_range_of_fx_l504_50479


namespace NUMINAMATH_GPT_find_cost_price_l504_50422

-- Condition 1: The owner charges his customer 15% more than the cost price.
def selling_price (C : Real) : Real := C * 1.15

-- Condition 2: A customer paid Rs. 8325 for the computer table.
def paid_amount : Real := 8325

-- Define the cost price and its expected value
def cost_price : Real := 7239.13

-- The theorem to prove that the cost price matches the expected value
theorem find_cost_price : 
  ∃ C : Real, selling_price C = paid_amount ∧ C = cost_price :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l504_50422


namespace NUMINAMATH_GPT_relation_among_a_b_c_l504_50480

open Real

theorem relation_among_a_b_c 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = log 3 / log 2)
  (h2 : b = log 7 / (2 * log 2))
  (h3 : c = 0.7 ^ 4) :
  a > b ∧ b > c :=
by
  -- we leave the proof as an exercise
  sorry

end NUMINAMATH_GPT_relation_among_a_b_c_l504_50480


namespace NUMINAMATH_GPT_value_of_f_750_l504_50409

theorem value_of_f_750 (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y^2)
    (hf500 : f 500 = 4) :
    f 750 = 16 / 9 :=
sorry

end NUMINAMATH_GPT_value_of_f_750_l504_50409


namespace NUMINAMATH_GPT_sum_of_extreme_values_eq_four_l504_50436

-- Given conditions in problem statement
variables (x y z : ℝ)
variables (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 8)

-- Statement to be proved: sum of smallest and largest possible values of x is 4
theorem sum_of_extreme_values_eq_four : m + M = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_extreme_values_eq_four_l504_50436


namespace NUMINAMATH_GPT_election_votes_l504_50476

theorem election_votes (V : ℝ) (h1 : ∃ geoff_votes : ℝ, geoff_votes = 0.01 * V)
                       (h2 : ∀ candidate_votes : ℝ, (candidate_votes > 0.51 * V) → candidate_votes > 0.51 * V)
                       (h3 : ∃ needed_votes : ℝ, needed_votes = 3000 ∧ 0.01 * V + needed_votes = 0.51 * V) :
                       V = 6000 :=
by sorry

end NUMINAMATH_GPT_election_votes_l504_50476


namespace NUMINAMATH_GPT_max_geometric_progression_terms_l504_50411

theorem max_geometric_progression_terms :
  ∀ a0 q : ℕ, (∀ k, a0 * q^k ≥ 100 ∧ a0 * q^k < 1000) →
  (∃ r s : ℕ, r > s ∧ q = r / s) →
  (∀ n, ∃ r s : ℕ, (r^n < 1000) ∧ ((r / s)^n < 10)) →
  n ≤ 5 :=
sorry

end NUMINAMATH_GPT_max_geometric_progression_terms_l504_50411


namespace NUMINAMATH_GPT_barry_more_votes_than_joey_l504_50488

theorem barry_more_votes_than_joey {M B J X : ℕ} 
  (h1 : M = 66)
  (h2 : J = 8)
  (h3 : M = 3 * B)
  (h4 : B = 2 * (J + X)) :
  B - J = 14 := by
  sorry

end NUMINAMATH_GPT_barry_more_votes_than_joey_l504_50488


namespace NUMINAMATH_GPT_maximize_area_l504_50459

noncomputable def max_area : ℝ :=
  let l := 100
  let w := 100
  l * w

theorem maximize_area (l w : ℝ) (h1 : 2 * l + 2 * w = 400) (h2 : l ≥ 100) (h3 : w ≥ 50) :
  (l * w ≤ 10000) :=
sorry

end NUMINAMATH_GPT_maximize_area_l504_50459


namespace NUMINAMATH_GPT_ab_minus_c_eq_six_l504_50493

theorem ab_minus_c_eq_six (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a * b + b - 9) : 
  a * b - c = 6 := 
by
  sorry

end NUMINAMATH_GPT_ab_minus_c_eq_six_l504_50493


namespace NUMINAMATH_GPT_percentage_employed_females_is_16_l504_50430

/- 
  In Town X, the population is divided into three age groups: 18-34, 35-54, and 55+.
  For each age group, the percentage of the employed population is 64%, and the percentage of employed males is 48%.
  We need to prove that the percentage of employed females in each age group is 16%.
-/

theorem percentage_employed_females_is_16
  (percentage_employed_population : ℝ)
  (percentage_employed_males : ℝ)
  (h1 : percentage_employed_population = 0.64)
  (h2 : percentage_employed_males = 0.48) :
  percentage_employed_population - percentage_employed_males = 0.16 :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end NUMINAMATH_GPT_percentage_employed_females_is_16_l504_50430


namespace NUMINAMATH_GPT_sum_cubes_eq_power_l504_50464

/-- Given the conditions, prove that 1^3 + 2^3 + 3^3 + 4^3 = 10^2 -/
theorem sum_cubes_eq_power : 1 + 2 + 3 + 4 = 10 → 1^3 + 2^3 + 3^3 + 4^3 = 10^2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_cubes_eq_power_l504_50464


namespace NUMINAMATH_GPT_bhanu_spends_on_petrol_l504_50405

-- Define the conditions as hypotheses
variable (income : ℝ)
variable (spend_on_rent : income * 0.7 * 0.14 = 98)

-- Define the theorem to prove
theorem bhanu_spends_on_petrol : (income * 0.3 = 300) :=
by
  sorry

end NUMINAMATH_GPT_bhanu_spends_on_petrol_l504_50405


namespace NUMINAMATH_GPT_problem_f_17_l504_50447

/-- Assume that f(1) = 0 and f(m + n) = f(m) + f(n) + 4 * (9 * m * n - 1) for all natural numbers m and n.
    Prove that f(17) = 4832.
-/
theorem problem_f_17 (f : ℕ → ℤ) 
  (h1 : f 1 = 0) 
  (h_func : ∀ m n : ℕ, f (m + n) = f m + f n + 4 * (9 * m * n - 1)) 
  : f 17 = 4832 := 
sorry

end NUMINAMATH_GPT_problem_f_17_l504_50447


namespace NUMINAMATH_GPT_min_number_of_stamps_exists_l504_50498

theorem min_number_of_stamps_exists : 
  ∃ s t : ℕ, 5 * s + 7 * t = 50 ∧ ∀ (s' t' : ℕ), 5 * s' + 7 * t' = 50 → s + t ≤ s' + t' := 
by
  sorry

end NUMINAMATH_GPT_min_number_of_stamps_exists_l504_50498


namespace NUMINAMATH_GPT_eqD_is_linear_l504_50443

-- Definitions for the given equations
def eqA (x y : ℝ) : Prop := 3 * x - 2 * y = 1
def eqB (x : ℝ) : Prop := 1 + (1 / x) = x
def eqC (x : ℝ) : Prop := x^2 = 9
def eqD (x : ℝ) : Prop := 2 * x - 3 = 5

-- Definition of a linear equation in one variable
def isLinear (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (∀ x : ℝ, eq x ↔ a * x + b = c)

-- Theorem stating that eqD is a linear equation
theorem eqD_is_linear : isLinear eqD :=
  sorry

end NUMINAMATH_GPT_eqD_is_linear_l504_50443


namespace NUMINAMATH_GPT_rational_function_solution_l504_50481

theorem rational_function_solution (g : ℝ → ℝ) (h : ∀ x ≠ 0, 4 * g (1 / x) + 3 * g x / x = x^3) :
  g (-3) = 135 / 4 := 
sorry

end NUMINAMATH_GPT_rational_function_solution_l504_50481


namespace NUMINAMATH_GPT_value_of_b_minus_a_l504_50495

variable (a b : ℕ)

theorem value_of_b_minus_a 
  (h1 : b = 10)
  (h2 : a * b = 2 * (a + b) + 12) : b - a = 6 :=
by sorry

end NUMINAMATH_GPT_value_of_b_minus_a_l504_50495


namespace NUMINAMATH_GPT_zeros_of_f_x_minus_1_l504_50454

def f (x : ℝ) : ℝ := x^2 - 1

theorem zeros_of_f_x_minus_1 :
  (f (0 - 1) = 0) ∧ (f (2 - 1) = 0) :=
by
  sorry

end NUMINAMATH_GPT_zeros_of_f_x_minus_1_l504_50454


namespace NUMINAMATH_GPT_calc_result_l504_50425

theorem calc_result : (-2 * -3 + 2) = 8 := sorry

end NUMINAMATH_GPT_calc_result_l504_50425


namespace NUMINAMATH_GPT_solve_m_range_l504_50416

-- Define the propositions
def p (m : ℝ) := m + 1 ≤ 0

def q (m : ℝ) := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Provide the Lean statement for the problem
theorem solve_m_range (m : ℝ) (hpq_false : ¬ (p m ∧ q m)) (hpq_true : p m ∨ q m) :
  m ≤ -2 ∨ (-1 < m ∧ m < 2) :=
sorry

end NUMINAMATH_GPT_solve_m_range_l504_50416


namespace NUMINAMATH_GPT_maximize_profit_l504_50486

variables (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ)

-- Definitions for the conditions
def nonneg_x := (0 ≤ x)
def nonneg_y := (0 ≤ y)
def constraint1 := (a1 * x + a2 * y ≤ c1)
def constraint2 := (b1 * x + b2 * y ≤ c2)
def profit := (z = d1 * x + d2 * y)

-- Proof of constraints and profit condition
theorem maximize_profit (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ) :
    nonneg_x x ∧ nonneg_y y ∧ constraint1 a1 a2 c1 x y ∧ constraint2 b1 b2 c2 x y → profit d1 d2 x y z :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l504_50486


namespace NUMINAMATH_GPT_max_gcd_b_n_b_n_plus_1_l504_50420

noncomputable def b (n : ℕ) : ℚ := (2 ^ n - 1) / 3

theorem max_gcd_b_n_b_n_plus_1 : ∀ n : ℕ, Int.gcd (b n).num (b (n + 1)).num = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_gcd_b_n_b_n_plus_1_l504_50420


namespace NUMINAMATH_GPT_central_angle_of_sector_l504_50490

theorem central_angle_of_sector (l S : ℝ) (r : ℝ) (θ : ℝ) 
  (h1 : l = 5) 
  (h2 : S = 5) 
  (h3 : S = (1 / 2) * l * r) 
  (h4 : l = θ * r): θ = 2.5 := by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l504_50490


namespace NUMINAMATH_GPT_number_of_chocolate_bars_l504_50483

theorem number_of_chocolate_bars (C : ℕ) (h1 : 50 * C = 250) : C = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_chocolate_bars_l504_50483


namespace NUMINAMATH_GPT_man_upstream_rate_l504_50400

theorem man_upstream_rate (rate_downstream : ℝ) (rate_still_water : ℝ) (rate_current : ℝ) 
    (h1 : rate_downstream = 32) (h2 : rate_still_water = 24.5) (h3 : rate_current = 7.5) : 
    rate_still_water - rate_current = 17 := 
by 
  sorry

end NUMINAMATH_GPT_man_upstream_rate_l504_50400


namespace NUMINAMATH_GPT_immortal_flea_can_visit_every_natural_l504_50463

theorem immortal_flea_can_visit_every_natural :
  ∀ (k : ℕ), ∃ (jumps : ℕ → ℤ), (∀ n : ℕ, ∃ m : ℕ, jumps m = n) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_immortal_flea_can_visit_every_natural_l504_50463


namespace NUMINAMATH_GPT_divisible_by_eight_l504_50410

def expr (n : ℕ) : ℕ := 3^(4*n + 1) + 5^(2*n + 1)

theorem divisible_by_eight (n : ℕ) : expr n % 8 = 0 :=
  sorry

end NUMINAMATH_GPT_divisible_by_eight_l504_50410


namespace NUMINAMATH_GPT_probability_2x_less_y_equals_one_over_eight_l504_50434

noncomputable def probability_2x_less_y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1 / 2) * 3 * 1.5
  let area_rectangle : ℚ := 6 * 3
  area_triangle / area_rectangle

theorem probability_2x_less_y_equals_one_over_eight :
  probability_2x_less_y_in_rectangle = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_2x_less_y_equals_one_over_eight_l504_50434


namespace NUMINAMATH_GPT_expand_and_simplify_l504_50449

theorem expand_and_simplify : ∀ x : ℝ, (7 * x + 5) * 3 * x^2 = 21 * x^3 + 15 * x^2 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l504_50449


namespace NUMINAMATH_GPT_solve_quadratic_eq_l504_50421

theorem solve_quadratic_eq {x : ℝ} (h : x^2 - 5*x + 6 = 0) : x = 2 ∨ x = 3 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l504_50421


namespace NUMINAMATH_GPT_expected_winnings_l504_50467

-- Define the probabilities
def prob_heads : ℚ := 1/2
def prob_tails : ℚ := 1/3
def prob_edge : ℚ := 1/6

-- Define the winnings
def win_heads : ℚ := 1
def win_tails : ℚ := 3
def lose_edge : ℚ := -5

-- Define the expected value function
def expected_value (p1 p2 p3 : ℚ) (w1 w2 w3 : ℚ) : ℚ :=
  p1 * w1 + p2 * w2 + p3 * w3

-- The expected winnings from flipping this coin
theorem expected_winnings : expected_value prob_heads prob_tails prob_edge win_heads win_tails lose_edge = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_expected_winnings_l504_50467


namespace NUMINAMATH_GPT_solution_set_condition_l504_50477

theorem solution_set_condition (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - 1| > a) ↔ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_condition_l504_50477


namespace NUMINAMATH_GPT_total_votes_cast_l504_50469

def votes_witch : ℕ := 7
def votes_unicorn : ℕ := 3 * votes_witch
def votes_dragon : ℕ := votes_witch + 25
def votes_total : ℕ := votes_witch + votes_unicorn + votes_dragon

theorem total_votes_cast : votes_total = 60 := by
  sorry

end NUMINAMATH_GPT_total_votes_cast_l504_50469


namespace NUMINAMATH_GPT_no_day_income_is_36_l504_50444

theorem no_day_income_is_36 : ∀ (n : ℕ), 3 * 3^(n-1) ≠ 36 :=
by
  intro n
  sorry

end NUMINAMATH_GPT_no_day_income_is_36_l504_50444


namespace NUMINAMATH_GPT_ratio_P_to_A_l504_50465

variable (M P A : ℕ) -- Define variables for Matthew, Patrick, and Alvin's egg rolls

theorem ratio_P_to_A (hM : M = 6) (hM_to_P : M = 3 * P) (hA : A = 4) : P / A = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_P_to_A_l504_50465


namespace NUMINAMATH_GPT_sugar_per_larger_cookie_l504_50485

theorem sugar_per_larger_cookie (c₁ c₂ : ℕ) (s₁ s₂ : ℝ) (h₁ : c₁ = 50) (h₂ : s₁ = 1 / 10) (h₃ : c₂ = 25) (h₄ : c₁ * s₁ = c₂ * s₂) : s₂ = 1 / 5 :=
by
  simp [h₁, h₂, h₃, h₄]
  sorry

end NUMINAMATH_GPT_sugar_per_larger_cookie_l504_50485


namespace NUMINAMATH_GPT_naomi_total_time_l504_50455

-- Definitions
def time_to_parlor : ℕ := 60
def speed_ratio : ℕ := 2 -- because her returning speed is half of the going speed
def first_trip_delay : ℕ := 15
def coffee_break : ℕ := 10
def second_trip_delay : ℕ := 20
def detour_time : ℕ := 30

-- Calculate total round trip times
def first_round_trip_time : ℕ := time_to_parlor + speed_ratio * time_to_parlor + first_trip_delay + coffee_break
def second_round_trip_time : ℕ := time_to_parlor + speed_ratio * time_to_parlor + second_trip_delay + detour_time

-- Hypothesis
def total_round_trip_time : ℕ := first_round_trip_time + second_round_trip_time

-- Main theorem statement
theorem naomi_total_time : total_round_trip_time = 435 := by
  sorry

end NUMINAMATH_GPT_naomi_total_time_l504_50455


namespace NUMINAMATH_GPT_price_of_cheese_cookie_pack_l504_50441

theorem price_of_cheese_cookie_pack
    (cartons : ℕ) (boxes_per_carton : ℕ) (packs_per_box : ℕ) (total_cost : ℕ)
    (h_cartons : cartons = 12)
    (h_boxes_per_carton : boxes_per_carton = 12)
    (h_packs_per_box : packs_per_box = 10)
    (h_total_cost : total_cost = 1440) :
  (total_cost / (cartons * boxes_per_carton * packs_per_box) = 1) :=
by
  -- conditions are explicitly given in the theorem statement
  sorry

end NUMINAMATH_GPT_price_of_cheese_cookie_pack_l504_50441


namespace NUMINAMATH_GPT_circle_radius_l504_50417

theorem circle_radius (m : ℝ) (h : 2 * 1 + (-m / 2) = 0) :
  let radius := 1 / 2 * Real.sqrt (4 + m ^ 2 + 16)
  radius = 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l504_50417


namespace NUMINAMATH_GPT_total_wage_calculation_l504_50478

def basic_pay_rate : ℝ := 20
def weekly_hours : ℝ := 40
def overtime_rate : ℝ := basic_pay_rate * 1.25
def total_hours_worked : ℝ := 48
def overtime_hours : ℝ := total_hours_worked - weekly_hours

theorem total_wage_calculation : 
  (weekly_hours * basic_pay_rate) + (overtime_hours * overtime_rate) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_total_wage_calculation_l504_50478


namespace NUMINAMATH_GPT_apples_difference_l504_50437

theorem apples_difference
    (adam_apples : ℕ)
    (jackie_apples : ℕ)
    (h_adam : adam_apples = 10)
    (h_jackie : jackie_apples = 2) :
    adam_apples - jackie_apples = 8 :=
by
    sorry

end NUMINAMATH_GPT_apples_difference_l504_50437


namespace NUMINAMATH_GPT_hiker_displacement_l504_50412

theorem hiker_displacement :
  let start_point := (0, 0)
  let move_east := (24, 0)
  let move_north := (0, 20)
  let move_west := (-7, 0)
  let move_south := (0, -9)
  let final_position := (start_point.1 + move_east.1 + move_west.1, start_point.2 + move_north.2 + move_south.2)
  let distance_from_start := Real.sqrt (final_position.1^2 + final_position.2^2)
  distance_from_start = Real.sqrt 410
:= by 
  sorry

end NUMINAMATH_GPT_hiker_displacement_l504_50412


namespace NUMINAMATH_GPT_brother_highlighters_spent_l504_50491

-- Define the total money given by the father
def total_money : ℕ := 100

-- Define the amount Heaven spent (2 sharpeners + 4 notebooks at $5 each)
def heaven_spent : ℕ := 30

-- Define the amount Heaven's brother spent on erasers (10 erasers at $4 each)
def erasers_spent : ℕ := 40

-- Prove the amount Heaven's brother spent on highlighters
theorem brother_highlighters_spent : total_money - heaven_spent - erasers_spent == 30 :=
by
  sorry

end NUMINAMATH_GPT_brother_highlighters_spent_l504_50491


namespace NUMINAMATH_GPT_mos_to_ory_bus_encounter_l504_50415

def encounter_buses (departure_time : Nat) (encounter_bus_time : Nat) (travel_time : Nat) : Nat := sorry

theorem mos_to_ory_bus_encounter :
  encounter_buses 0 30 5 = 10 :=
sorry

end NUMINAMATH_GPT_mos_to_ory_bus_encounter_l504_50415


namespace NUMINAMATH_GPT_unique_solution_l504_50494

theorem unique_solution (x y a : ℝ) :
  (x^2 + y^2 = 2 * a ∧ x + Real.log (y^2 + 1) / Real.log 2 = a) ↔ a = 0 ∧ x = 0 ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l504_50494


namespace NUMINAMATH_GPT_find_a_l504_50428

-- Define sets A and B
def A : Set ℕ := {1, 2, 5}
def B (a : ℕ) : Set ℕ := {2, a}

-- Given condition: A ∪ B = {1, 2, 3, 5}
def union_condition (a : ℕ) : Prop := A ∪ B a = {1, 2, 3, 5}

-- Theorem we want to prove
theorem find_a (a : ℕ) : union_condition a → a = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l504_50428


namespace NUMINAMATH_GPT_solution_set_of_inequality_l504_50418

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)
-- f(x) is symmetric about the origin
variable (symmetric_f : ∀ x, f (-x) = -f x)
-- f(2) = 2
variable (f_at_2 : f 2 = 2)
-- For any 0 < x2 < x1, the slope condition holds
variable (slope_cond : ∀ x1 x2, 0 < x2 ∧ x2 < x1 → (f x1 - f x2) / (x1 - x2) < 1)

theorem solution_set_of_inequality :
  {x : ℝ | f x - x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l504_50418


namespace NUMINAMATH_GPT_number_of_true_propositions_is_one_l504_50438

theorem number_of_true_propositions_is_one :
  (¬ ∀ x : ℝ, x^4 > x^2) ∧
  (¬ (∀ (p q : Prop), ¬ (p ∧ q) → (¬ p ∧ ¬ q))) ∧
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) →
  1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_is_one_l504_50438


namespace NUMINAMATH_GPT_time_to_shovel_snow_l504_50440

noncomputable def initial_rate : ℕ := 30
noncomputable def decay_rate : ℕ := 2
noncomputable def driveway_width : ℕ := 6
noncomputable def driveway_length : ℕ := 15
noncomputable def snow_depth : ℕ := 2

noncomputable def total_snow_volume : ℕ := driveway_width * driveway_length * snow_depth

def snow_shoveling_time (initial_rate decay_rate total_volume : ℕ) : ℕ :=
-- Function to compute the time needed, assuming definition provided
sorry

theorem time_to_shovel_snow 
  : snow_shoveling_time initial_rate decay_rate total_snow_volume = 8 :=
sorry

end NUMINAMATH_GPT_time_to_shovel_snow_l504_50440


namespace NUMINAMATH_GPT_acute_triangle_exterior_angles_obtuse_l504_50475

theorem acute_triangle_exterior_angles_obtuse
  (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h_sum : A + B + C = π) :
  ∀ α β γ, α = A + B → β = B + C → γ = C + A → α > π / 2 ∧ β > π / 2 ∧ γ > π / 2 :=
by
  sorry

end NUMINAMATH_GPT_acute_triangle_exterior_angles_obtuse_l504_50475


namespace NUMINAMATH_GPT_grass_coverage_day_l504_50471

theorem grass_coverage_day (coverage : ℕ → ℚ) : 
  (∀ n : ℕ, coverage (n + 1) = 2 * coverage n) → 
  coverage 24 = 1 → 
  coverage 21 = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_grass_coverage_day_l504_50471


namespace NUMINAMATH_GPT_sofia_total_time_l504_50474

def distance1 : ℕ := 150
def speed1 : ℕ := 5
def distance2 : ℕ := 150
def speed2 : ℕ := 6
def laps : ℕ := 8
def time_per_lap := (distance1 / speed1) + (distance2 / speed2)
def total_time := 440  -- 7 minutes and 20 seconds in seconds

theorem sofia_total_time :
  laps * time_per_lap = total_time :=
by
  -- Proof steps are omitted and represented by sorry.
  sorry

end NUMINAMATH_GPT_sofia_total_time_l504_50474


namespace NUMINAMATH_GPT_arithmetic_mean_l504_50457

theorem arithmetic_mean (a b : ℚ) (h1 : a = 3/7) (h2 : b = 6/11) :
  (a + b) / 2 = 75 / 154 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_l504_50457


namespace NUMINAMATH_GPT_bridge_length_at_least_200_l504_50419

theorem bridge_length_at_least_200 :
  ∀ (length_train : ℝ) (speed_kmph : ℝ) (time_secs : ℝ),
  length_train = 200 ∧ speed_kmph = 32 ∧ time_secs = 20 →
  ∃ l : ℝ, l ≥ length_train :=
by
  sorry

end NUMINAMATH_GPT_bridge_length_at_least_200_l504_50419


namespace NUMINAMATH_GPT_length_of_field_l504_50435

-- Define the known conditions
def width := 50
def total_distance_run := 1800
def num_laps := 6

-- Define the problem statement
theorem length_of_field :
  ∃ L : ℕ, 6 * (2 * (L + width)) = total_distance_run ∧ L = 100 :=
by
  sorry

end NUMINAMATH_GPT_length_of_field_l504_50435


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l504_50402

theorem largest_angle_in_triangle
    (a b c : ℝ)
    (h_sum_two_angles : a + b = (7 / 5) * 90)
    (h_angle_difference : b = a + 40) :
    max a (max b c) = 83 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l504_50402


namespace NUMINAMATH_GPT_fraction_irreducible_l504_50462

theorem fraction_irreducible (n : ℤ) : gcd (2 * n ^ 2 + 9 * n - 17) (n + 6) = 1 := by
  sorry

end NUMINAMATH_GPT_fraction_irreducible_l504_50462


namespace NUMINAMATH_GPT_vacation_cost_l504_50404

theorem vacation_cost (C : ℝ) (h1 : C / 3 - C / 4 = 30) : C = 360 :=
by
  sorry

end NUMINAMATH_GPT_vacation_cost_l504_50404


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l504_50461

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem problem_part1 (h : ∀ x : ℝ, f (-x) = -f x) : a = 1 :=
sorry

theorem problem_part2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l504_50461


namespace NUMINAMATH_GPT_find_k_multiple_l504_50450

theorem find_k_multiple (a b k : ℕ) (h1 : a = b + 5) (h2 : a + b = 13) 
  (h3 : 3 * (a + 7) = k * (b + 7)) : k = 4 := sorry

end NUMINAMATH_GPT_find_k_multiple_l504_50450


namespace NUMINAMATH_GPT_ab_cd_ge_ac_bd_squared_eq_condition_ad_eq_bc_l504_50453

theorem ab_cd_ge_ac_bd_squared (a b c d : ℝ) : ((a^2 + b^2) * (c^2 + d^2)) ≥ (a * c + b * d)^2 := 
by sorry

theorem eq_condition_ad_eq_bc (a b c d : ℝ) (h : a * d = b * c) : ((a^2 + b^2) * (c^2 + d^2)) = (a * c + b * d)^2 := 
by sorry

end NUMINAMATH_GPT_ab_cd_ge_ac_bd_squared_eq_condition_ad_eq_bc_l504_50453


namespace NUMINAMATH_GPT_original_square_side_length_l504_50427

theorem original_square_side_length :
  ∃ n k : ℕ, (n + k) * (n + k) - n * n = 47 ∧ k ≤ 5 ∧ k % 2 = 1 ∧ n = 23 :=
by
  sorry

end NUMINAMATH_GPT_original_square_side_length_l504_50427


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l504_50408

-- Definitions based on conditions from step a
def first_term : ℕ := 1
def last_term : ℕ := 36
def num_terms : ℕ := 8

-- The problem statement in Lean 4
theorem arithmetic_sequence_sum :
  (num_terms / 2) * (first_term + last_term) = 148 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l504_50408


namespace NUMINAMATH_GPT_common_sale_days_in_july_l504_50429

def BookstoreSaleDays (d : ℕ) : Prop :=
  (d ≥ 1) ∧ (d ≤ 31) ∧ (d % 4 = 0)

def ShoeStoreSaleDays (d : ℕ) : Prop :=
  (d ≥ 1) ∧ (d ≤ 31) ∧ (∃ k : ℕ, d = 2 + k * 7)

theorem common_sale_days_in_july : ∃! d, (BookstoreSaleDays d) ∧ (ShoeStoreSaleDays d) :=
by {
  sorry
}

end NUMINAMATH_GPT_common_sale_days_in_july_l504_50429


namespace NUMINAMATH_GPT_isabella_haircut_length_l504_50456

-- Define the original length of Isabella's hair.
def original_length : ℕ := 18

-- Define the length of hair cut off.
def cut_off_length : ℕ := 9

-- The length of Isabella's hair after the haircut.
def length_after_haircut : ℕ := original_length - cut_off_length

-- Statement of the theorem we want to prove.
theorem isabella_haircut_length : length_after_haircut = 9 :=
by
  sorry

end NUMINAMATH_GPT_isabella_haircut_length_l504_50456


namespace NUMINAMATH_GPT_vanessa_score_l504_50466

-- Define the total score of the team
def total_points : ℕ := 60

-- Define the score of the seven other players
def other_players_points : ℕ := 7 * 4

-- Mathematics statement for proof
theorem vanessa_score : total_points - other_players_points = 32 :=
by
    sorry

end NUMINAMATH_GPT_vanessa_score_l504_50466


namespace NUMINAMATH_GPT_range_of_s_l504_50487

def double_value_point (s t : ℝ) (ht : t ≠ -1) :
  Prop := 
  ∀ k : ℝ, (t + 1) * k^2 + t * k + s = 0 →
  (t^2 - 4 * s * (t + 1) > 0)

theorem range_of_s (s t : ℝ) (ht : t ≠ -1) :
  double_value_point s t ht ↔ -1 < s ∧ s < 0 :=
sorry

end NUMINAMATH_GPT_range_of_s_l504_50487


namespace NUMINAMATH_GPT_ufo_convention_attendees_l504_50432

theorem ufo_convention_attendees 
  (F M : ℕ) 
  (h1 : F + M = 450) 
  (h2 : M = F + 26) : 
  M = 238 := 
sorry

end NUMINAMATH_GPT_ufo_convention_attendees_l504_50432


namespace NUMINAMATH_GPT_greg_savings_l504_50489

-- Definitions based on the conditions
def scooter_cost : ℕ := 90
def money_needed : ℕ := 33

-- The theorem to prove
theorem greg_savings : scooter_cost - money_needed = 57 := 
by
  -- sorry is used to skip the actual mathematical proof steps
  sorry

end NUMINAMATH_GPT_greg_savings_l504_50489


namespace NUMINAMATH_GPT_sum_of_terms_in_arithmetic_sequence_eq_l504_50472

variable {a : ℕ → ℕ}

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_terms_in_arithmetic_sequence_eq :
  arithmetic_sequence a →
  (a 2 + a 3 + a 10 + a 11 = 36) →
  (a 3 + a 10 = 18) :=
by
  intros h_seq h_sum
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_sum_of_terms_in_arithmetic_sequence_eq_l504_50472


namespace NUMINAMATH_GPT_yuna_average_score_l504_50451

theorem yuna_average_score (avg_may_june : ℕ) (score_july : ℕ) (h1 : avg_may_june = 84) (h2 : score_july = 96) :
  (avg_may_june * 2 + score_july) / 3 = 88 := by
  sorry

end NUMINAMATH_GPT_yuna_average_score_l504_50451


namespace NUMINAMATH_GPT_t_shirts_sold_l504_50424

theorem t_shirts_sold (total_money : ℕ) (money_per_tshirt : ℕ) (n : ℕ) 
  (h1 : total_money = 2205) (h2 : money_per_tshirt = 9) (h3 : total_money = n * money_per_tshirt) : 
  n = 245 :=
by
  sorry

end NUMINAMATH_GPT_t_shirts_sold_l504_50424


namespace NUMINAMATH_GPT_line_symmetric_y_axis_eqn_l504_50448

theorem line_symmetric_y_axis_eqn (x y : ℝ) : 
  (∀ x y : ℝ, x - y + 1 = 0 → x + y - 1 = 0) := 
sorry

end NUMINAMATH_GPT_line_symmetric_y_axis_eqn_l504_50448


namespace NUMINAMATH_GPT_calf_rope_length_l504_50492

noncomputable def new_rope_length (initial_length : ℝ) (additional_area : ℝ) : ℝ :=
  let A1 := Real.pi * initial_length ^ 2
  let A2 := A1 + additional_area
  let new_length_squared := A2 / Real.pi
  Real.sqrt new_length_squared

theorem calf_rope_length :
  new_rope_length 12 565.7142857142857 = 18 := by
  sorry

end NUMINAMATH_GPT_calf_rope_length_l504_50492


namespace NUMINAMATH_GPT_age_ratio_l504_50406

/-- 
Axiom: Kareem's age is 42 and his son's age is 14. 
-/
axiom Kareem_age : ℕ
axiom Son_age : ℕ

/-- 
Conditions: 
  - Kareem's age after 10 years plus his son's age after 10 years equals 76.
  - Kareem's current age is 42.
  - His son's current age is 14.
-/
axiom age_condition : Kareem_age + 10 + Son_age + 10 = 76
axiom Kareem_current_age : Kareem_age = 42
axiom Son_current_age : Son_age = 14

/-- 
Theorem: The ratio of Kareem's age to his son's age is 3:1.
-/
theorem age_ratio : Kareem_age / Son_age = 3 / 1 := by {
  -- Proof skipped
  sorry 
}

end NUMINAMATH_GPT_age_ratio_l504_50406


namespace NUMINAMATH_GPT_problem_statement_l504_50414

theorem problem_statement
  (a b m n c : ℝ)
  (h1 : a = -b)
  (h2 : m * n = 1)
  (h3 : |c| = 3)
  : a + b + m * n - |c| = -2 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l504_50414


namespace NUMINAMATH_GPT_product_of_four_integers_negative_l504_50446

theorem product_of_four_integers_negative {a b c d : ℤ}
  (h : a * b * c * d < 0) :
  (∃ n : ℕ, n ≤ 3 ∧ (n = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0))) :=
sorry

end NUMINAMATH_GPT_product_of_four_integers_negative_l504_50446


namespace NUMINAMATH_GPT_polynomial_identity_l504_50496

theorem polynomial_identity (a0 a1 a2 a3 a4 a5 : ℤ) (x : ℤ) :
  (1 + 3 * x) ^ 5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 →
  a0 - a1 + a2 - a3 + a4 - a5 = -32 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l504_50496


namespace NUMINAMATH_GPT_nancy_yearly_payment_l504_50452

open Real

-- Define the monthly cost of the car insurance
def monthly_cost : ℝ := 80

-- Nancy's percentage contribution
def percentage : ℝ := 0.40

-- Calculate the monthly payment Nancy will make
def monthly_payment : ℝ := percentage * monthly_cost

-- Calculate the yearly payment Nancy will make
def yearly_payment : ℝ := 12 * monthly_payment

-- State the proof problem
theorem nancy_yearly_payment : yearly_payment = 384 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_nancy_yearly_payment_l504_50452


namespace NUMINAMATH_GPT_proposition_judgement_l504_50470

theorem proposition_judgement (p q : Prop) (a b c x : ℝ) :
  (¬ (p ∨ q) → (¬ p ∧ ¬ q)) ∧
  (¬ (a > b → a * c^2 > b * c^2)) ∧
  (¬ (∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)) ∧
  ((x^2 - 3*x + 2 = 0) → (x = 2)) =
  false := sorry

end NUMINAMATH_GPT_proposition_judgement_l504_50470


namespace NUMINAMATH_GPT_max_pieces_from_cake_l504_50439

theorem max_pieces_from_cake (large_cake_area small_piece_area : ℕ) 
  (h_large_cake : large_cake_area = 15 * 15) 
  (h_small_piece : small_piece_area = 5 * 5) :
  large_cake_area / small_piece_area = 9 := 
by
  sorry

end NUMINAMATH_GPT_max_pieces_from_cake_l504_50439


namespace NUMINAMATH_GPT_op_value_l504_50426

noncomputable def op (a b c : ℝ) (k : ℤ) : ℝ :=
  b^2 - k * a^2 * c

theorem op_value : op 2 5 3 3 = -11 := by
  sorry

end NUMINAMATH_GPT_op_value_l504_50426


namespace NUMINAMATH_GPT_gcd_of_powers_l504_50442

theorem gcd_of_powers (m n : ℕ) (h1 : m = 2^2024 - 1) (h2 : n = 2^2007 - 1) : 
  Nat.gcd m n = 131071 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_powers_l504_50442


namespace NUMINAMATH_GPT_least_number_to_subtract_l504_50458

theorem least_number_to_subtract 
  (n : ℤ) 
  (h1 : 7 ∣ (90210 - n + 12)) 
  (h2 : 11 ∣ (90210 - n + 12)) 
  (h3 : 13 ∣ (90210 - n + 12)) 
  (h4 : 17 ∣ (90210 - n + 12)) 
  (h5 : 19 ∣ (90210 - n + 12)) : 
  n = 90198 :=
sorry

end NUMINAMATH_GPT_least_number_to_subtract_l504_50458


namespace NUMINAMATH_GPT_amy_small_gardens_l504_50497

-- Define the initial number of seeds
def initial_seeds : ℕ := 101

-- Define the number of seeds planted in the big garden
def big_garden_seeds : ℕ := 47

-- Define the number of seeds planted in each small garden
def seeds_per_small_garden : ℕ := 6

-- Define the number of small gardens
def number_of_small_gardens : ℕ := (initial_seeds - big_garden_seeds) / seeds_per_small_garden

-- Prove that Amy has 9 small gardens
theorem amy_small_gardens : number_of_small_gardens = 9 := by
  sorry

end NUMINAMATH_GPT_amy_small_gardens_l504_50497


namespace NUMINAMATH_GPT_max_value_correct_l504_50445

noncomputable def max_value (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^5 * y + x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 + x * y^5

theorem max_value_correct (x y : ℝ) (h : x + y = 5) : max_value x y h ≤ 22884 :=
  sorry

end NUMINAMATH_GPT_max_value_correct_l504_50445


namespace NUMINAMATH_GPT_find_a_bi_c_l504_50433

theorem find_a_bi_c (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_eq : (a - (b : ℤ)*I)^2 + c = 13 - 8*I) :
  a = 2 ∧ b = 2 ∧ c = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_a_bi_c_l504_50433


namespace NUMINAMATH_GPT_score_analysis_l504_50431

open Real

noncomputable def deviations : List ℝ := [8, -3, 12, -7, -10, -4, -8, 1, 0, 10]
def benchmark : ℝ := 85

theorem score_analysis :
  let highest_score := benchmark + List.maximum deviations
  let lowest_score := benchmark + List.minimum deviations
  let sum_deviations := List.sum deviations
  let average_deviation := sum_deviations / List.length deviations
  let average_score := benchmark + average_deviation
  highest_score = 97 ∧ lowest_score = 75 ∧ average_score = 84.9 :=
by
  sorry -- This is the placeholder for the proof

end NUMINAMATH_GPT_score_analysis_l504_50431


namespace NUMINAMATH_GPT_andrew_paid_1428_to_shopkeeper_l504_50499

-- Given conditions
def rate_per_kg_grapes : ℕ := 98
def quantity_of_grapes : ℕ := 11
def rate_per_kg_mangoes : ℕ := 50
def quantity_of_mangoes : ℕ := 7

-- Definitions for costs
def cost_of_grapes : ℕ := rate_per_kg_grapes * quantity_of_grapes
def cost_of_mangoes : ℕ := rate_per_kg_mangoes * quantity_of_mangoes
def total_amount_paid : ℕ := cost_of_grapes + cost_of_mangoes

-- Theorem to prove the total amount paid
theorem andrew_paid_1428_to_shopkeeper : total_amount_paid = 1428 := by
  sorry

end NUMINAMATH_GPT_andrew_paid_1428_to_shopkeeper_l504_50499


namespace NUMINAMATH_GPT_sin_geq_tan_minus_half_tan_cubed_l504_50403

theorem sin_geq_tan_minus_half_tan_cubed (x : ℝ) (hx : 0 ≤ x ∧ x < π / 2) :
  Real.sin x ≥ Real.tan x - 1/2 * (Real.tan x) ^ 3 := 
sorry

end NUMINAMATH_GPT_sin_geq_tan_minus_half_tan_cubed_l504_50403


namespace NUMINAMATH_GPT_inequality_d_over_c_lt_d_plus_4_over_c_plus_4_l504_50413

theorem inequality_d_over_c_lt_d_plus_4_over_c_plus_4
  (a b c d : ℝ)
  (h1 : a > b)
  (h2 : c > d)
  (h3 : d > 0) :
  (d / c) < ((d + 4) / (c + 4)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_d_over_c_lt_d_plus_4_over_c_plus_4_l504_50413


namespace NUMINAMATH_GPT_fraction_money_left_zero_l504_50407

-- Defining variables and conditions
variables {m c : ℝ} -- m: total money, c: total cost of CDs

-- Condition under the problem statement
def uses_one_fourth_of_money_to_buy_one_fourth_of_CDs (m c : ℝ) := (1 / 4) * m = (1 / 4) * c

-- The conjecture to be proven
theorem fraction_money_left_zero 
  (h: uses_one_fourth_of_money_to_buy_one_fourth_of_CDs m c) 
  (h_eq: c = m) : 
  (m - c) / m = 0 := 
by
  sorry

end NUMINAMATH_GPT_fraction_money_left_zero_l504_50407


namespace NUMINAMATH_GPT_most_probable_hits_l504_50401

theorem most_probable_hits (p : ℝ) (q : ℝ) (k0 : ℕ) (n : ℤ) 
  (h1 : p = 0.7) (h2 : q = 1 - p) (h3 : k0 = 16) 
  (h4 : 21 < (n : ℝ) * 0.7) (h5 : (n : ℝ) * 0.7 < 23.3) : 
  n = 22 ∨ n = 23 :=
sorry

end NUMINAMATH_GPT_most_probable_hits_l504_50401


namespace NUMINAMATH_GPT_arithmetic_sequence_length_l504_50482

theorem arithmetic_sequence_length :
  ∀ (a₁ d an : ℤ), a₁ = -5 → d = 3 → an = 40 → (∃ n : ℕ, an = a₁ + (n - 1) * d ∧ n = 16) :=
by
  intros a₁ d an h₁ hd han
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_length_l504_50482


namespace NUMINAMATH_GPT_total_logs_combined_l504_50460

theorem total_logs_combined 
  (a1 l1 a2 l2 : ℕ) 
  (n1 n2 : ℕ) 
  (S1 S2 : ℕ) 
  (h1 : a1 = 15) 
  (h2 : l1 = 10) 
  (h3 : n1 = 6) 
  (h4 : S1 = n1 * (a1 + l1) / 2) 
  (h5 : a2 = 9) 
  (h6 : l2 = 5) 
  (h7 : n2 = 5) 
  (h8 : S2 = n2 * (a2 + l2) / 2) : 
  S1 + S2 = 110 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_logs_combined_l504_50460


namespace NUMINAMATH_GPT_average_percentage_score_is_71_l504_50423

-- Define the number of students.
def number_of_students : ℕ := 150

-- Define the scores and their corresponding frequencies.
def scores_and_frequencies : List (ℕ × ℕ) :=
  [(100, 10), (95, 20), (85, 45), (75, 30), (65, 25), (55, 15), (45, 5)]

-- Define the total points scored by all students.
def total_points_scored : ℕ := 
  scores_and_frequencies.foldl (λ acc pair => acc + pair.1 * pair.2) 0

-- Define the average percentage score.
def average_score : ℚ := total_points_scored / number_of_students

-- Statement of the proof problem.
theorem average_percentage_score_is_71 :
  average_score = 71.0 := by
  sorry

end NUMINAMATH_GPT_average_percentage_score_is_71_l504_50423
