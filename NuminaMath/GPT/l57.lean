import Mathlib

namespace determine_k_coplanar_l57_5780

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable {A B C D : V}
variable (k : ℝ)

theorem determine_k_coplanar (h : 4 • A - 3 • B + 6 • C + k • D = 0) : k = -13 :=
sorry

end determine_k_coplanar_l57_5780


namespace distance_from_left_focal_to_line_l57_5790

noncomputable def ellipse_eq_line_dist : Prop :=
  let a := 2
  let b := Real.sqrt 3
  let c := 1
  let x₀ := -1
  let y₀ := 0
  let x₁ := 0
  let y₁ := Real.sqrt 3
  let x₂ := 1
  let y₂ := 0
  
  -- Equation of the line derived from the upper vertex and right focal point
  let m := -(y₁ - y₂) / (x₁ - x₂)
  let line_eq (x y : ℝ) := (Real.sqrt 3 * x + y - Real.sqrt 3 = 0)
  
  -- Distance formula from point to line
  let d := abs (Real.sqrt 3 * x₀ + y₀ - Real.sqrt 3) / Real.sqrt ((Real.sqrt 3)^2 + 1^2)

  -- The assertion that the distance is √3
  d = Real.sqrt 3

theorem distance_from_left_focal_to_line : ellipse_eq_line_dist := 
  sorry  -- Proof is omitted as per the instruction

end distance_from_left_focal_to_line_l57_5790


namespace find_n_l57_5713

-- Define the operation €
def operation (x y : ℕ) : ℕ := 2 * x * y

-- State the theorem
theorem find_n (n : ℕ) (h : operation 8 (operation 4 n) = 640) : n = 5 :=
  by
  sorry

end find_n_l57_5713


namespace desired_gain_percentage_l57_5784

theorem desired_gain_percentage (cp16 sp16 cp12881355932203391 sp12881355932203391 : ℝ) :
  sp16 = 1 →
  sp16 = 0.95 * cp16 →
  sp12881355932203391 = 1 →
  cp12881355932203391 = (12.881355932203391 / 16) * cp16 →
  (sp12881355932203391 - cp12881355932203391) / cp12881355932203391 * 100 = 18.75 :=
by sorry

end desired_gain_percentage_l57_5784


namespace tickets_sold_l57_5704

theorem tickets_sold (T : ℕ) (h1 : 3 * T / 4 > 0)
    (h2 : 5 * (T / 4) / 9 > 0)
    (h3 : 80 > 0)
    (h4 : 20 > 0) :
    (1 / 4 * T - 5 / 36 * T = 100) -> T = 900 :=
by
  sorry

end tickets_sold_l57_5704


namespace logarithmic_relationship_l57_5773

theorem logarithmic_relationship (a b : ℝ) (h1 : a = Real.logb 16 625) (h2 : b = Real.logb 2 25) : a = b / 2 :=
sorry

end logarithmic_relationship_l57_5773


namespace coffee_price_decrease_is_37_5_l57_5739

-- Define the initial and new prices
def initial_price_per_packet := 12 / 3
def new_price_per_packet := 10 / 4

-- Define the calculation of the percent decrease
def percent_decrease (initial_price : ℚ) (new_price : ℚ) : ℚ :=
  ((initial_price - new_price) / initial_price) * 100

-- The theorem statement
theorem coffee_price_decrease_is_37_5 :
  percent_decrease initial_price_per_packet new_price_per_packet = 37.5 := by
  sorry

end coffee_price_decrease_is_37_5_l57_5739


namespace max_consecutive_interesting_numbers_l57_5796

def is_interesting (n : ℕ) : Prop :=
  (n / 100 % 3 = 0) ∨ (n / 10 % 10 % 3 = 0) ∨ (n % 10 % 3 = 0)

theorem max_consecutive_interesting_numbers :
  ∃ l r, 100 ≤ l ∧ r ≤ 999 ∧ r - l + 1 = 122 ∧ (∀ n, l ≤ n ∧ n ≤ r → is_interesting n) ∧ 
  ∀ l' r', 100 ≤ l' ∧ r' ≤ 999 ∧ r' - l' + 1 > 122 → ∃ n, l' ≤ n ∧ n ≤ r' ∧ ¬ is_interesting n := 
sorry

end max_consecutive_interesting_numbers_l57_5796


namespace parabola_distance_to_y_axis_l57_5785

theorem parabola_distance_to_y_axis :
  ∀ (M : ℝ × ℝ), (M.2 ^ 2 = 4 * M.1) → 
  dist (M, (1, 0)) = 10 →
  abs (M.1) = 9 :=
by
  intros M hParabola hDist
  sorry

end parabola_distance_to_y_axis_l57_5785


namespace range_of_x_l57_5772

theorem range_of_x (x p : ℝ) (hp : 0 ≤ p ∧ p ≤ 4) :
  (x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) :=
by {
  sorry
}

end range_of_x_l57_5772


namespace englishman_land_earnings_l57_5709

noncomputable def acres_to_square_yards (acres : ℝ) : ℝ := acres * 4840
noncomputable def square_yards_to_square_meters (sq_yards : ℝ) : ℝ := sq_yards * (0.9144 ^ 2)
noncomputable def square_meters_to_hectares (sq_meters : ℝ) : ℝ := sq_meters / 10000
noncomputable def cost_of_land (hectares : ℝ) (price_per_hectare : ℝ) : ℝ := hectares * price_per_hectare

theorem englishman_land_earnings
  (acres_owned : ℝ)
  (price_per_hectare : ℝ)
  (acre_to_yard : ℝ)
  (yard_to_meter : ℝ)
  (hectare_to_meter : ℝ)
  (h1 : acres_owned = 2)
  (h2 : price_per_hectare = 500000)
  (h3 : acre_to_yard = 4840)
  (h4 : yard_to_meter = 0.9144)
  (h5 : hectare_to_meter = 10000)
  : cost_of_land (square_meters_to_hectares (square_yards_to_square_meters (acres_to_square_yards acres_owned))) price_per_hectare = 404685.6 := sorry

end englishman_land_earnings_l57_5709


namespace sequence_bound_l57_5789

open Real

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ)
  (h₀ : ∀ i : ℕ, 0 < i → 0 ≤ a i ∧ a i ≤ c)
  (h₁ : ∀ (i j : ℕ), 0 < i → 0 < j → i ≠ j → abs (a i - a j) ≥ 1 / (i + j)) :
  c ≥ 1 :=
by {
  sorry
}

end sequence_bound_l57_5789


namespace tin_to_copper_ratio_l57_5717

theorem tin_to_copper_ratio (L_A T_A T_B C_B : ℝ) 
  (h_total_mass_A : L_A + T_A = 90)
  (h_ratio_A : L_A / T_A = 3 / 4)
  (h_total_mass_B : T_B + C_B = 140)
  (h_total_tin : T_A + T_B = 91.42857142857143) :
  T_B / C_B = 2 / 5 :=
sorry

end tin_to_copper_ratio_l57_5717


namespace solve_for_z_l57_5738

theorem solve_for_z :
  ∃ z : ℤ, (∀ x y : ℤ, x = 11 → y = 8 → 2 * x + 3 * z = 5 * y) → z = 6 :=
by
  sorry

end solve_for_z_l57_5738


namespace school_xx_percentage_increase_l57_5729

theorem school_xx_percentage_increase
  (X Y : ℕ) -- denote the number of students at school XX and YY last year
  (H_Y : Y = 2400) -- condition: school YY had 2400 students last year
  (H_total : X + Y = 4000) -- condition: total number of students last year was 4000
  (H_increase_YY : YY_increase = (3 * Y) / 100) -- condition: 3 percent increase at school YY
  (H_difference : XX_increase = YY_increase + 40) -- condition: school XX grew by 40 more students than YY
  : (XX_increase * 100) / X = 7 :=
by
  sorry

end school_xx_percentage_increase_l57_5729


namespace find_k_l57_5708

noncomputable def k := 3

theorem find_k :
  (∀ x : ℝ, (Real.sin x ^ k) * (Real.sin (k * x)) + (Real.cos x ^ k) * (Real.cos (k * x)) = Real.cos (2 * x) ^ k) ↔ k = 3 :=
sorry

end find_k_l57_5708


namespace factorize_diff_of_squares_factorize_common_factor_diff_of_squares_l57_5771

-- Problem 1: Prove the factorization of 4x^2 - 25y^2
theorem factorize_diff_of_squares (x y : ℝ) : 4 * x^2 - 25 * y^2 = (2 * x + 5 * y) * (2 * x - 5 * y) := 
sorry

-- Problem 2: Prove the factorization of -3xy^3 + 27x^3y
theorem factorize_common_factor_diff_of_squares (x y : ℝ) : 
  -3 * x * y^3 + 27 * x^3 * y = -3 * x * y * (y + 3 * x) * (y - 3 * x) := 
sorry

end factorize_diff_of_squares_factorize_common_factor_diff_of_squares_l57_5771


namespace partnership_investment_l57_5724

theorem partnership_investment
  (a_investment : ℕ := 30000)
  (b_investment : ℕ)
  (c_investment : ℕ := 50000)
  (c_profit_share : ℕ := 36000)
  (total_profit : ℕ := 90000)
  (total_investment := a_investment + b_investment + c_investment)
  (c_defined_share : ℚ := 2/5)
  (profit_proportionality : (c_profit_share : ℚ) / total_profit = (c_investment : ℚ) / total_investment) :
  b_investment = 45000 :=
by
  sorry

end partnership_investment_l57_5724


namespace frequency_of_middle_group_l57_5775

theorem frequency_of_middle_group :
  ∃ m : ℝ, m + (1/3) * m = 200 ∧ (1/3) * m = 50 :=
by
  sorry

end frequency_of_middle_group_l57_5775


namespace find_initial_children_l57_5711

variables (x y : ℕ)

-- Defining the conditions 
def initial_children_on_bus (x : ℕ) : Prop :=
  ∃ y : ℕ, x - 68 + y = 12 ∧ 68 - y = 24 + y

-- Theorem statement
theorem find_initial_children : initial_children_on_bus x → x = 58 :=
by
  -- Skipping the proof for now
  sorry

end find_initial_children_l57_5711


namespace compute_t_minus_s_l57_5700

noncomputable def t : ℚ := (40 + 30 + 30 + 20) / 4

noncomputable def s : ℚ := (40 * (40 / 120) + 30 * (30 / 120) + 30 * (30 / 120) + 20 * (20 / 120))

theorem compute_t_minus_s : t - s = -1.67 := by
  sorry

end compute_t_minus_s_l57_5700


namespace john_moves_540kg_l57_5726

-- Conditions
def used_to_back_squat : ℝ := 200
def increased_by : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9

-- Definitions based on conditions
def current_back_squat : ℝ := used_to_back_squat + increased_by
def current_front_squat : ℝ := front_squat_ratio * current_back_squat
def one_triple : ℝ := triple_ratio * current_front_squat
def three_triples : ℝ := 3 * one_triple

-- The proof statement
theorem john_moves_540kg : three_triples = 540 := by
  sorry

end john_moves_540kg_l57_5726


namespace train_speed_l57_5765

-- Definitions to capture the conditions
def length_of_train : ℝ := 100
def length_of_bridge : ℝ := 300
def time_to_cross_bridge : ℝ := 36

-- The speed of the train calculated according to the condition
def total_distance : ℝ := length_of_train + length_of_bridge

theorem train_speed : total_distance / time_to_cross_bridge = 11.11 :=
by
  sorry

end train_speed_l57_5765


namespace leos_time_is_1230_l57_5754

theorem leos_time_is_1230
  (theo_watch_slow: Int)
  (theo_watch_fast_belief: Int)
  (leo_watch_fast: Int)
  (leo_watch_slow_belief: Int)
  (theo_thinks_time: Int):
  theo_watch_slow = 10 ∧
  theo_watch_fast_belief = 5 ∧
  leo_watch_fast = 5 ∧
  leo_watch_slow_belief = 10 ∧
  theo_thinks_time = 720
  → leo_thinks_time = 750 :=
by
  sorry

end leos_time_is_1230_l57_5754


namespace simplify_expr_l57_5750

open Real

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) : 
  sqrt (1 + ( (x^6 - 2) / (3 * x^3) )^2) = sqrt (x^12 + 5 * x^6 + 4) / (3 * x^3) :=
by
  sorry

end simplify_expr_l57_5750


namespace satisfy_third_eq_l57_5703

theorem satisfy_third_eq 
  (x y : ℝ) 
  (h1 : x^2 - 3 * x * y + 2 * y^2 + x - y = 0)
  (h2 : x^2 - 2 * x * y + y^2 - 5 * x + 7 * y = 0) 
  : x * y - 12 * x + 15 * y = 0 :=
by
  sorry

end satisfy_third_eq_l57_5703


namespace minimum_revenue_maximum_marginal_cost_minimum_profit_l57_5735

noncomputable def R (x : ℕ) : ℝ := x^2 + 16 / x^2 + 40
noncomputable def C (x : ℕ) : ℝ := 10 * x + 40 / x
noncomputable def MC (x : ℕ) : ℝ := C (x + 1) - C x
noncomputable def z (x : ℕ) : ℝ := R x - C x

theorem minimum_revenue :
  ∀ x : ℕ, 1 ≤ x → x ≤ 10 → R x ≥ 72 :=
sorry

theorem maximum_marginal_cost :
  ∀ x : ℕ, 1 ≤ x → x ≤ 9 → MC x ≤ 86 / 9 :=
sorry

theorem minimum_profit :
  ∀ x : ℕ, 1 ≤ x → x ≤ 10 → (x = 1 ∨ x = 4) → z x ≥ 7 :=
sorry

end minimum_revenue_maximum_marginal_cost_minimum_profit_l57_5735


namespace polynomial_range_open_interval_l57_5722

theorem polynomial_range_open_interval :
  ∀ (k : ℝ), k > 0 → ∃ (x y : ℝ), (1 - x * y)^2 + x^2 = k :=
by
  sorry

end polynomial_range_open_interval_l57_5722


namespace bluegrass_percentage_l57_5767

-- Define the problem conditions
def seed_mixture_X_ryegrass_percentage : ℝ := 40
def seed_mixture_Y_ryegrass_percentage : ℝ := 25
def seed_mixture_Y_fescue_percentage : ℝ := 75
def mixture_X_Y_ryegrass_percentage : ℝ := 30
def mixture_weight_percentage_X : ℝ := 33.33333333333333

-- Prove that the percentage of bluegrass in seed mixture X is 60%
theorem bluegrass_percentage (X_ryegrass : ℝ) (Y_ryegrass : ℝ) (Y_fescue : ℝ) (mixture_ryegrass : ℝ) (weight_percentage_X : ℝ) :
  X_ryegrass = seed_mixture_X_ryegrass_percentage →
  Y_ryegrass = seed_mixture_Y_ryegrass_percentage →
  Y_fescue = seed_mixture_Y_fescue_percentage →
  mixture_ryegrass = mixture_X_Y_ryegrass_percentage →
  weight_percentage_X = mixture_weight_percentage_X →
  (100 - X_ryegrass) = 60 :=
by
  intro hX_ryegrass hY_ryegrass hY_fescue hmixture_ryegrass hweight_X
  rw [hX_ryegrass]
  sorry

end bluegrass_percentage_l57_5767


namespace find_lower_rate_l57_5786

-- Definitions
def total_investment : ℝ := 20000
def total_interest : ℝ := 1440
def higher_rate : ℝ := 0.09
def fraction_higher : ℝ := 0.55

-- The amount invested at the higher rate
def x := fraction_higher * total_investment
-- The amount invested at the lower rate
def y := total_investment - x

-- The interest contributions
def interest_higher := x * higher_rate
def interest_lower (r : ℝ) := y * r

-- The equation we need to solve to find the lower interest rate
theorem find_lower_rate (r : ℝ) : interest_higher + interest_lower r = total_interest → r = 0.05 :=
by
  sorry

end find_lower_rate_l57_5786


namespace jessie_interest_l57_5787

noncomputable def compoundInterest 
  (P : ℝ) -- Principal
  (r : ℝ) -- annual interest rate
  (n : ℕ) -- number of times interest applied per time period
  (t : ℝ) -- time periods elapsed
  : ℝ :=
  P * (1 + r / n)^(n * t)

theorem jessie_interest :
  let P := 1200
  let annual_rate := 0.08
  let periods_per_year := 2
  let years := 5
  let A := compoundInterest P annual_rate periods_per_year years
  let interest := A - P
  interest = 576.29 :=
by
  sorry

end jessie_interest_l57_5787


namespace max_gcd_15n_plus_4_8n_plus_1_l57_5728

theorem max_gcd_15n_plus_4_8n_plus_1 (n : ℕ) (h : n > 0) : 
  ∃ g, g = gcd (15 * n + 4) (8 * n + 1) ∧ g ≤ 17 :=
sorry

end max_gcd_15n_plus_4_8n_plus_1_l57_5728


namespace Sam_needs_16_more_hours_l57_5769

noncomputable def Sam_hourly_rate : ℝ :=
  460 / 23

noncomputable def Sam_earnings_Sep_to_Feb : ℝ :=
  8 * Sam_hourly_rate

noncomputable def Sam_total_earnings : ℝ :=
  460 + Sam_earnings_Sep_to_Feb

noncomputable def Sam_remaining_money : ℝ :=
  Sam_total_earnings - 340

noncomputable def Sam_needed_money : ℝ :=
  600 - Sam_remaining_money

noncomputable def Sam_additional_hours_needed : ℝ :=
  Sam_needed_money / Sam_hourly_rate

theorem Sam_needs_16_more_hours : Sam_additional_hours_needed = 16 :=
by 
  sorry

end Sam_needs_16_more_hours_l57_5769


namespace track_extension_needed_l57_5752

noncomputable def additional_track_length (r : ℝ) (g1 g2 : ℝ) : ℝ :=
  let l1 := r / g1
  let l2 := r / g2
  l2 - l1

theorem track_extension_needed :
  additional_track_length 800 0.04 0.015 = 33333 :=
by
  sorry

end track_extension_needed_l57_5752


namespace find_constants_u_v_l57_5757

theorem find_constants_u_v
  (n p r1 r2 : ℝ)
  (h1 : r1 + r2 = n)
  (h2 : r1 * r2 = p) :
  ∃ u v, (r1^4 + r2^4 = -u) ∧ (r1^4 * r2^4 = v) ∧ u = -(n^4 - 4*p*n^2 + 2*p^2) ∧ v = p^4 :=
by
  sorry

end find_constants_u_v_l57_5757


namespace evaluate_composite_l57_5706

def f (x : ℕ) : ℕ := 2 * x + 5
def g (x : ℕ) : ℕ := 3 * x + 4

theorem evaluate_composite : f (g (f 3)) = 79 := by
  sorry

end evaluate_composite_l57_5706


namespace find_function_l57_5714

noncomputable def solution_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = x + f y → ∃ c : ℝ, ∀ x : ℝ, f x = x + c

theorem find_function (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = x + f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end find_function_l57_5714


namespace max_value_of_z_l57_5778

theorem max_value_of_z (x y : ℝ) (h1 : x + 2 * y ≤ 2) (h2 : x + y ≥ 0) (h3 : x ≤ 4) : 
  ∃ (z : ℝ), z = 2 * x + y ∧ z ≤ 11 :=
by
  sorry

end max_value_of_z_l57_5778


namespace frac_inequality_l57_5794

theorem frac_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : d > c) (h4 : c > 0) : (a/c) > (b/d) := 
sorry

end frac_inequality_l57_5794


namespace no_solutions_interval_length_l57_5795

theorem no_solutions_interval_length : 
  (∀ x a : ℝ, |x| ≠ ax - 2) → ([-1, 1].length = 2) :=
by {
  sorry
}

end no_solutions_interval_length_l57_5795


namespace pqr_value_l57_5730

theorem pqr_value (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h1 : p + q + r = 24)
  (h2 : (1 / p : ℚ) + (1 / q) + (1 / r) + 240 / (p * q * r) = 1): 
  p * q * r = 384 :=
by
  sorry

end pqr_value_l57_5730


namespace sum_of_first_four_terms_of_arithmetic_sequence_l57_5782

theorem sum_of_first_four_terms_of_arithmetic_sequence
  (a d : ℤ)
  (h1 : a + 4 * d = 10)  -- Condition for the fifth term
  (h2 : a + 5 * d = 14)  -- Condition for the sixth term
  (h3 : a + 6 * d = 18)  -- Condition for the seventh term
  : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 0 :=  -- Prove the sum of the first four terms is 0
by
  sorry

end sum_of_first_four_terms_of_arithmetic_sequence_l57_5782


namespace inverse_proportional_example_l57_5774

variable (x y : ℝ)

def inverse_proportional (x y : ℝ) := y = 8 / (x - 1)

theorem inverse_proportional_example
  (h1 : y = 4)
  (h2 : x = 3) :
  inverse_proportional x y :=
by
  sorry

end inverse_proportional_example_l57_5774


namespace seashells_total_l57_5749

theorem seashells_total (s m : Nat) (hs : s = 18) (hm : m = 47) : s + m = 65 := 
by
  -- We are just specifying the theorem statement here
  sorry

end seashells_total_l57_5749


namespace find_solutions_l57_5747

def is_solution (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ Int.gcd (Int.gcd a b) c = 1 ∧
  (a + b + c) ∣ (a^12 + b^12 + c^12) ∧
  (a + b + c) ∣ (a^23 + b^23 + c^23) ∧
  (a + b + c) ∣ (a^11004 + b^11004 + c^11004)

theorem find_solutions :
  (is_solution 1 1 1) ∧ (is_solution 1 1 4) ∧ 
  (∀ a b c : ℕ, is_solution a b c → 
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 4)) := 
sorry

end find_solutions_l57_5747


namespace minimum_distinct_lines_l57_5788

theorem minimum_distinct_lines (n : ℕ) (h : n = 31) : 
  ∃ (k : ℕ), k = 9 :=
by
  sorry

end minimum_distinct_lines_l57_5788


namespace general_term_seq_l57_5725

theorem general_term_seq 
  (a : ℕ → ℚ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 5/3) 
  (h_rec : ∀ n, n > 0 → a (n + 2) = (5 / 3) * a (n + 1) - (2 / 3) * a n) : 
  ∀ n, a n = 2 - (3 / 2) * (2 / 3)^n :=
by
  sorry

end general_term_seq_l57_5725


namespace range_of_a_l57_5701

theorem range_of_a
  (h : ∀ x : ℝ, |x - 1| + |x - 2| > Real.log (a ^ 2) / Real.log 4) :
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2 :=
sorry

end range_of_a_l57_5701


namespace modified_determinant_l57_5723

def determinant_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem modified_determinant (x y z w : ℝ)
  (h : determinant_2x2 x y z w = 6) :
  determinant_2x2 x (5 * x + 4 * y) z (5 * z + 4 * w) = 24 := by
  sorry

end modified_determinant_l57_5723


namespace expected_value_of_unfair_die_l57_5705

noncomputable def seven_sided_die_expected_value : ℝ :=
  let p7 := 1 / 3
  let p_other := (2 / 3) / 6
  ((1 + 2 + 3 + 4 + 5 + 6) * p_other + 7 * p7)

theorem expected_value_of_unfair_die :
  seven_sided_die_expected_value = 14 / 3 :=
by
  sorry

end expected_value_of_unfair_die_l57_5705


namespace shelves_used_l57_5756

def coloring_books := 87
def sold_books := 33
def books_per_shelf := 6

theorem shelves_used (h1: coloring_books - sold_books = 54) : 54 / books_per_shelf = 9 :=
by
  sorry

end shelves_used_l57_5756


namespace intersection_point_exists_l57_5737

theorem intersection_point_exists :
  ∃ (x y z t : ℝ), (x = 1 - 2 * t) ∧ (y = 2 + t) ∧ (z = -1 - t) ∧
                   (x - 2 * y + 5 * z + 17 = 0) ∧ 
                   (x = -1) ∧ (y = 3) ∧ (z = -2) :=
by
  sorry

end intersection_point_exists_l57_5737


namespace find_k_l57_5791

theorem find_k (k : ℤ) :
  (∃ a b c : ℤ, a = 49 + k ∧ b = 441 + k ∧ c = 961 + k ∧
  (∃ r : ℚ, b = r * a ∧ c = r * r * a)) ↔ k = 1152 := by
  sorry

end find_k_l57_5791


namespace factorize_expression_l57_5745

variable (m n : ℤ)

theorem factorize_expression : 2 * m * n^2 - 12 * m * n + 18 * m = 2 * m * (n - 3)^2 := by
  sorry

end factorize_expression_l57_5745


namespace hexagon_inequality_l57_5751

variables {Point : Type} [MetricSpace Point]

-- Definitions of points and distances
variables (A B C D E F G H : Point) 
variables (dist : Point → Point → ℝ)
variables (angle : Point → Point → Point → ℝ)

-- Conditions
variables (hABCDEF : ConvexHexagon A B C D E F)
variables (hAB_BC_CD : dist A B = dist B C ∧ dist B C = dist C D)
variables (hDE_EF_FA : dist D E = dist E F ∧ dist E F = dist F A)
variables (hBCD_60 : angle B C D = 60)
variables (hEFA_60 : angle E F A = 60)
variables (hAGB_120 : angle A G B = 120)
variables (hDHE_120 : angle D H E = 120)

-- Objective statement
theorem hexagon_inequality : 
  dist A G + dist G B + dist G H + dist D H + dist H E ≥ dist C F :=
sorry

end hexagon_inequality_l57_5751


namespace cars_pass_same_order_l57_5715

theorem cars_pass_same_order (num_cars : ℕ) (num_points : ℕ)
    (cities_speeds speeds_outside_cities : Fin num_cars → ℝ) :
    num_cars = 10 → num_points = 2011 → 
    ∃ (p1 p2 : Fin num_points), p1 ≠ p2 ∧ (∀ i j : Fin num_cars, (i < j) → 
    (cities_speeds i) / (cities_speeds i + speeds_outside_cities i) = 
    (cities_speeds j) / (cities_speeds j + speeds_outside_cities j) → p1 = p2 ) :=
by
  sorry

end cars_pass_same_order_l57_5715


namespace focus_of_parabola_l57_5731

theorem focus_of_parabola (p : ℝ) :
  (∃ p, x ^ 2 = 4 * p * y ∧ x ^ 2 = 4 * 1 * y) → (0, p) = (0, 1) :=
by
  sorry

end focus_of_parabola_l57_5731


namespace sum_of_numbers_l57_5727

theorem sum_of_numbers : 
  5678 + 6785 + 7856 + 8567 = 28886 := 
by 
  sorry

end sum_of_numbers_l57_5727


namespace distinct_p_q_r_s_t_sum_l57_5781

theorem distinct_p_q_r_s_t_sum (p q r s t : ℤ) (h1 : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120)
    (h2 : p ≠ q) (h3 : p ≠ r) (h4 : p ≠ s) (h5 : p ≠ t) 
    (h6 : q ≠ r) (h7 : q ≠ s) (h8 : q ≠ t)
    (h9 : r ≠ s) (h10 : r ≠ t)
    (h11 : s ≠ t) : p + q + r + s + t = 25 := by
  sorry

end distinct_p_q_r_s_t_sum_l57_5781


namespace megan_initial_acorns_l57_5764

def initial_acorns (given_away left: ℕ) : ℕ := 
  given_away + left

theorem megan_initial_acorns :
  initial_acorns 7 9 = 16 := 
by 
  unfold initial_acorns
  rfl

end megan_initial_acorns_l57_5764


namespace speed_of_man_rowing_upstream_l57_5732

theorem speed_of_man_rowing_upstream (V_m V_downstream V_upstream V_s : ℝ) 
  (h1 : V_m = 25) 
  (h2 : V_downstream = 38) :
  V_upstream = V_m - (V_downstream - V_m) :=
by
  sorry

end speed_of_man_rowing_upstream_l57_5732


namespace find_value_of_A_l57_5702

theorem find_value_of_A (x : ℝ) (h₁ : x - 3 * (x - 2) ≥ 2) (h₂ : 4 * x - 2 < 5 * x - 1) (h₃ : x ≠ 1) (h₄ : x ≠ -1) (h₅ : x ≠ 0) (hx : x = 2) :
  let A := (3 * x / (x - 1) - x / (x + 1)) / (x / (x^2 - 1))
  A = 8 :=
by
  -- Proof will be filled in
  sorry

end find_value_of_A_l57_5702


namespace ending_number_of_range_l57_5748

theorem ending_number_of_range (n : ℕ) (h : ∃ k, 1 ≤ k ∧ k ≤ 5 ∧ n = 29 + 11 * k) : n = 77 := by
  sorry

end ending_number_of_range_l57_5748


namespace lara_total_space_larger_by_1500_square_feet_l57_5770

theorem lara_total_space_larger_by_1500_square_feet :
  let length_rect := 30
  let width_rect := 50
  let area_rect := length_rect * width_rect
  let side_square := width_rect
  let area_square := side_square * side_square
  let total_area := area_rect + area_square
  total_area - area_square = 1500 :=
by
  -- Definitions
  let length_rect := 30
  let width_rect := 50
  let area_rect := length_rect * width_rect
  let side_square := width_rect
  let area_square := side_square * side_square
  let total_area := area_rect + area_square
  
  -- Calculation
  have h_area_rect : area_rect = 1500 := by
    norm_num [area_rect, length_rect, width_rect]

  have h_area_square : area_square = 2500 := by
    norm_num [area_square, side_square]

  have h_total_area : total_area = 4000 := by
    norm_num [total_area, h_area_rect, h_area_square]

  -- Final comparison
  have h_difference : total_area - area_square = 1500 := by
    norm_num [total_area, area_square, h_area_square]

  exact h_difference

end lara_total_space_larger_by_1500_square_feet_l57_5770


namespace percent_of_whole_l57_5740

theorem percent_of_whole (Part Whole : ℝ) (Percent : ℝ) (hPart : Part = 160) (hWhole : Whole = 50) :
  Percent = (Part / Whole) * 100 → Percent = 320 :=
by
  rw [hPart, hWhole]
  sorry

end percent_of_whole_l57_5740


namespace total_votes_cast_l57_5744

theorem total_votes_cast (V : ℕ) (C R : ℕ) 
  (hC : C = 30 * V / 100) 
  (hR1 : R = C + 4000) 
  (hR2 : R = 70 * V / 100) : 
  V = 10000 :=
by
  sorry

end total_votes_cast_l57_5744


namespace total_amount_paid_l57_5792

-- Define the given conditions
def q_g : ℕ := 9        -- Quantity of grapes
def r_g : ℕ := 70       -- Rate per kg of grapes
def q_m : ℕ := 9        -- Quantity of mangoes
def r_m : ℕ := 55       -- Rate per kg of mangoes

-- Define the total amount paid calculation and prove it equals 1125
theorem total_amount_paid : (q_g * r_g + q_m * r_m) = 1125 :=
by
  -- Proof will be provided here. Currently using 'sorry' to skip it.
  sorry

end total_amount_paid_l57_5792


namespace subtract_29_after_46_l57_5743

theorem subtract_29_after_46 (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 :=
sorry

end subtract_29_after_46_l57_5743


namespace total_passengers_l57_5779

theorem total_passengers (P : ℕ) 
  (h1 : P = (1/12 : ℚ) * P + (1/4 : ℚ) * P + (1/9 : ℚ) * P + (1/6 : ℚ) * P + 42) :
  P = 108 :=
sorry

end total_passengers_l57_5779


namespace correct_answer_is_B_l57_5753

def is_permutation_problem (desc : String) : Prop :=
  desc = "Permutation"

def check_problem_A : Prop :=
  ¬ is_permutation_problem "Selecting 2 out of 8 students to participate in a knowledge competition"

def check_problem_B : Prop :=
  is_permutation_problem "If 10 people write letters to each other once, how many letters are written in total"

def check_problem_C : Prop :=
  ¬ is_permutation_problem "There are 5 points on a plane, with no three points collinear, what is the maximum number of lines that can be determined by these 5 points"

def check_problem_D : Prop :=
  ¬ is_permutation_problem "From the numbers 1, 2, 3, 4, choose any two numbers to multiply, how many different results are there"

theorem correct_answer_is_B : check_problem_A ∧ check_problem_B ∧ check_problem_C ∧ check_problem_D → 
  ("B" = "B") := by
  sorry

end correct_answer_is_B_l57_5753


namespace tablet_value_is_2100_compensation_for_m_days_l57_5776

-- Define the given conditions
def monthly_compensation: ℕ := 30
def monthly_tablet_value (x: ℕ) (cash: ℕ): ℕ := x + cash

def daily_compensation (days: ℕ) (x: ℕ) (cash: ℕ): ℕ :=
  days * (x / monthly_compensation + cash / monthly_compensation)

def received_compensation (tablet_value: ℕ) (cash: ℕ): ℕ :=
  tablet_value + cash

-- The proofs we need:
-- Proof that the tablet value is 2100 yuan
theorem tablet_value_is_2100:
  ∀ (x: ℕ) (cash_1 cash_2: ℕ), 
  ((20 * (x / monthly_compensation + 1500 / monthly_compensation)) = (x + 300)) → 
  x = 2100 := sorry

-- Proof that compensation for m days is 120m yuan
theorem compensation_for_m_days (m: ℕ):
  ∀ (x: ℕ), 
  ((x + 1500) / monthly_compensation) = 120 → 
  x = 2100 → 
  m * 120 = 120 * m := sorry

end tablet_value_is_2100_compensation_for_m_days_l57_5776


namespace necessary_sufficient_condition_l57_5761

theorem necessary_sufficient_condition (n : ℕ) (hn : n > 0) :
  (∃ x : ℚ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 :=
sorry

end necessary_sufficient_condition_l57_5761


namespace caleb_hamburgers_total_l57_5707

def total_spent : ℝ := 66.50
def cost_single : ℝ := 1.00
def cost_double : ℝ := 1.50
def num_double : ℕ := 33

theorem caleb_hamburgers_total : 
  ∃ n : ℕ,  n = 17 + num_double ∧ 
            (num_double * cost_double) + (n - num_double) * cost_single = total_spent := by
sorry

end caleb_hamburgers_total_l57_5707


namespace sums_of_squares_divisibility_l57_5768

theorem sums_of_squares_divisibility :
  (∀ n : ℤ, (3 * n^2 + 2) % 3 ≠ 0) ∧ (∃ n : ℤ, (3 * n^2 + 2) % 11 = 0) := 
by
  sorry

end sums_of_squares_divisibility_l57_5768


namespace sufficient_but_not_necessary_not_necessary_l57_5721

theorem sufficient_but_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (a * (b + 1) > a^2) :=
sorry

theorem not_necessary (a b : ℝ) : (a * (b + 1) > a^2 → b > a ∧ a > 0) → false :=
sorry

end sufficient_but_not_necessary_not_necessary_l57_5721


namespace solve_proof_problem_l57_5783

noncomputable def proof_problem (f g : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x + g y) = 2 * x + y → g (x + f y) = x / 2 + y

theorem solve_proof_problem (f g : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + g y) = 2 * x + y) :
  ∀ x y : ℝ, g (x + f y) = x / 2 + y :=
sorry

end solve_proof_problem_l57_5783


namespace interest_rate_B_lent_to_C_l57_5755

noncomputable def principal : ℝ := 1500
noncomputable def rate_A : ℝ := 10
noncomputable def time : ℝ := 3
noncomputable def gain_B : ℝ := 67.5
noncomputable def interest_paid_by_B_to_A : ℝ := principal * rate_A * time / 100
noncomputable def interest_received_by_B_from_C : ℝ := interest_paid_by_B_to_A + gain_B
noncomputable def expected_rate : ℝ := 11.5

theorem interest_rate_B_lent_to_C :
  interest_received_by_B_from_C = principal * (expected_rate) * time / 100 := 
by
  -- the proof will go here
  sorry

end interest_rate_B_lent_to_C_l57_5755


namespace birthday_count_l57_5799

theorem birthday_count (N : ℕ) (P : ℝ) (days : ℕ) (hN : N = 1200) (hP1 : P = 1 / 365 ∨ P = 1 / 366) 
  (hdays : days = 365 ∨ days = 366) : 
  N * P = 4 :=
by
  sorry

end birthday_count_l57_5799


namespace calc_residue_modulo_l57_5741

theorem calc_residue_modulo :
  let a := 320
  let b := 16
  let c := 28
  let d := 5
  let e := 7
  let n := 14
  (a * b - c * d + e) % n = 3 :=
by
  sorry

end calc_residue_modulo_l57_5741


namespace greatest_possible_sum_l57_5742

theorem greatest_possible_sum (x y : ℤ) (h : x^2 + y^2 = 100) : x + y ≤ 14 :=
sorry

end greatest_possible_sum_l57_5742


namespace part1_part2_part3_l57_5763

-- Part 1: There exists a real number a such that a + 1/a ≤ 2
theorem part1 : ∃ a : ℝ, a + 1/a ≤ 2 := sorry

-- Part 2: For all positive real numbers a and b, b/a + a/b ≥ 2
theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : b / a + a / b ≥ 2 := sorry

-- Part 3: For positive real numbers x and y such that x + 2y = 1, then 2/x + 1/y ≥ 8
theorem part3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 2 / x + 1 / y ≥ 8 := sorry

end part1_part2_part3_l57_5763


namespace no_snow_five_days_l57_5712

-- Define the probability of snow on a given day
def prob_snow : ℚ := 2 / 3

-- Define the probability of no snow on a given day
def prob_not_snow : ℚ := 1 - prob_snow

-- Define the probability of no snow for five consecutive days
def prob_no_snow_five_days : ℚ := (prob_not_snow)^5

-- Statement to prove the probability of no snow for the next five days is 1/243
theorem no_snow_five_days : prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l57_5712


namespace arithmetic_seq_term_six_l57_5733

theorem arithmetic_seq_term_six {a : ℕ → ℝ} (a1 : ℝ) (S3 : ℝ) (h1 : a1 = 2) (h2 : S3 = 12) :
  a 6 = 12 :=
sorry

end arithmetic_seq_term_six_l57_5733


namespace ratio_of_B_to_C_l57_5766

variables (A B C : ℕ)

-- Conditions from the problem
axiom h1 : A = B + 2
axiom h2 : A + B + C = 12
axiom h3 : B = 4

-- Goal: Prove that the ratio of B's age to C's age is 2
theorem ratio_of_B_to_C : B / C = 2 :=
by {
  sorry
}

end ratio_of_B_to_C_l57_5766


namespace ending_number_of_range_divisible_by_five_l57_5762

theorem ending_number_of_range_divisible_by_five
  (first_number : ℕ)
  (number_of_terms : ℕ)
  (h_first : first_number = 15)
  (h_terms : number_of_terms = 10)
  : ∃ ending_number : ℕ, ending_number = first_number + 5 * (number_of_terms - 1) := 
by
  sorry

end ending_number_of_range_divisible_by_five_l57_5762


namespace hexagon_perimeter_is_24_l57_5734

-- Conditions given in the problem
def AB : ℝ := 3
def EF : ℝ := 3
def BE : ℝ := 4
def AF : ℝ := 4
def CD : ℝ := 5
def DF : ℝ := 5

-- Statement to show that the perimeter is 24 units
theorem hexagon_perimeter_is_24 :
  AB + BE + CD + DF + EF + AF = 24 :=
by
  sorry

end hexagon_perimeter_is_24_l57_5734


namespace base8_difference_divisible_by_7_l57_5758

theorem base8_difference_divisible_by_7 (A B : ℕ) (h₁ : A < 8) (h₂ : B < 8) (h₃ : A ≠ B) : 
  ∃ k : ℕ, k * 7 = (if 8 * A + B > 8 * B + A then 8 * A + B - (8 * B + A) else 8 * B + A - (8 * A + B)) :=
by
  sorry

end base8_difference_divisible_by_7_l57_5758


namespace hagrid_divisible_by_three_l57_5736

def distinct_digits (n : ℕ) : Prop :=
  n < 10

theorem hagrid_divisible_by_three (H A G R I D : ℕ) (H_dist A_dist G_dist R_dist I_dist D_dist : distinct_digits H ∧ distinct_digits A ∧ distinct_digits G ∧ distinct_digits R ∧ distinct_digits I ∧ distinct_digits D)
  (distinct_letters: H ≠ A ∧ H ≠ G ∧ H ≠ R ∧ H ≠ I ∧ H ≠ D ∧ A ≠ G ∧ A ≠ R ∧ A ≠ I ∧ A ≠ D ∧ G ≠ R ∧ G ≠ I ∧ G ≠ D ∧ R ≠ I ∧ R ≠ D ∧ I ≠ D) :
  3 ∣ (H * 100000 + A * 10000 + G * 1000 + R * 100 + I * 10 + D) * H * A * G * R * I * D :=
sorry

end hagrid_divisible_by_three_l57_5736


namespace fraction_division_l57_5746

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := 
by
  sorry

end fraction_division_l57_5746


namespace square_side_length_l57_5716

-- Define the conditions
def rectangle_width : ℝ := 4
def rectangle_length : ℝ := 4
def area_rectangle : ℝ := rectangle_width * rectangle_length
def area_square : ℝ := area_rectangle

-- Prove the side length of the square
theorem square_side_length :
  ∃ s : ℝ, s * s = area_square ∧ s = 4 := 
  by {
    -- Here you'd write the proof step, but it's omitted as per instructions
    sorry
  }

end square_side_length_l57_5716


namespace max_partial_sum_l57_5720

variable (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ)
variable (S : ℕ → ℤ)

-- Define the arithmetic sequence and the conditions given
def arithmetic_sequence (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a_n n = a_1 + n * d

def condition1 (a_1 : ℤ) : Prop := a_1 > 0

def condition2 (a_n : ℕ → ℤ) (d : ℤ) : Prop := 3 * (a_n 8) = 5 * (a_n 13)

-- Define the partial sum of the arithmetic sequence
def partial_sum (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
∀ n : ℕ, S n = n * (a_n 1 + a_n n) / 2

-- Define the main problem: Prove that S_20 is the greatest
theorem max_partial_sum (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) (S : ℕ → ℤ) :
  arithmetic_sequence a_n a_1 d →
  condition1 a_1 →
  condition2 a_n d →
  partial_sum S a_n →
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → S 20 ≥ S n := by
  sorry

end max_partial_sum_l57_5720


namespace exists_same_color_points_at_distance_one_l57_5760

theorem exists_same_color_points_at_distance_one (coloring : ℝ × ℝ → Fin 3) :
  ∃ (p q : ℝ × ℝ), (coloring p = coloring q) ∧ (dist p q = 1) := sorry

end exists_same_color_points_at_distance_one_l57_5760


namespace nancy_total_savings_l57_5710

noncomputable def total_savings : ℝ :=
  let cost_this_month := 9 * 5
  let cost_last_month := 8 * 4
  let cost_next_month := 7 * 6
  let discount_this_month := 0.20 * cost_this_month
  let discount_last_month := 0.20 * cost_last_month
  let discount_next_month := 0.20 * cost_next_month
  discount_this_month + discount_last_month + discount_next_month

theorem nancy_total_savings : total_savings = 23.80 :=
by
  sorry

end nancy_total_savings_l57_5710


namespace chess_tournament_total_players_l57_5793

-- Define the conditions

def total_points_calculation (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 132

def games_played (n : ℕ) : ℕ :=
  ((n + 12) * (n + 11)) / 2

theorem chess_tournament_total_players :
  ∃ n, total_points_calculation n = games_played n ∧ n + 12 = 34 :=
by {
  -- Assume n is found such that all conditions are satisfied
  use 22,
  -- Provide the necessary equations and conditions
  sorry
}

end chess_tournament_total_players_l57_5793


namespace unique_polynomial_solution_l57_5797

def polynomial_homogeneous_of_degree_n (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (t x y : ℝ), P (t * x) (t * y) = t^n * P x y

def polynomial_symmetric_condition (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y z : ℝ), P (y + z) x + P (z + x) y + P (x + y) z = 0

def polynomial_value_at_point (P : ℝ → ℝ → ℝ) : Prop :=
  P 1 0 = 1

theorem unique_polynomial_solution (P : ℝ → ℝ → ℝ) (n : ℕ) :
  polynomial_homogeneous_of_degree_n P n →
  polynomial_symmetric_condition P →
  polynomial_value_at_point P →
  ∀ x y : ℝ, P x y = (x + y)^n * (x - 2 * y) := 
by
  intros h_deg h_symm h_value x y
  sorry

end unique_polynomial_solution_l57_5797


namespace avg_korean_language_score_l57_5777

theorem avg_korean_language_score (male_avg : ℝ) (female_avg : ℝ) (male_students : ℕ) (female_students : ℕ) 
    (male_avg_given : male_avg = 83.1) (female_avg_given : female_avg = 84) (male_students_given : male_students = 10) (female_students_given : female_students = 8) :
    (male_avg * male_students + female_avg * female_students) / (male_students + female_students) = 83.5 :=
by sorry

end avg_korean_language_score_l57_5777


namespace number_of_good_colorings_l57_5759

theorem number_of_good_colorings (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) : 
  ∃ (good_colorings : ℕ), good_colorings = 6 * (2^n - 4 + 4 * 2^(m-2)) :=
sorry

end number_of_good_colorings_l57_5759


namespace track_length_l57_5718

theorem track_length (x : ℝ) (b_speed s_speed : ℝ) (b_dist1 s_dist1 s_dist2 : ℝ)
  (h1 : b_dist1 = 80)
  (h2 : s_dist1 = x / 2 - 80)
  (h3 : s_dist2 = s_dist1 + 180)
  (h4 : x / 4 * b_speed = (x / 2 - 80) * s_speed)
  (h5 : x / 4 * ((x / 2) - 100) = (x / 2 + 100) * s_speed) :
  x = 520 := 
sorry

end track_length_l57_5718


namespace find_n_l57_5719

theorem find_n (n : ℕ) (h : (2 * n + 1) / 3 = 2022) : n = 3033 :=
sorry

end find_n_l57_5719


namespace jill_arrives_before_jack_l57_5798

theorem jill_arrives_before_jack
  (distance : ℝ)
  (jill_speed : ℝ)
  (jack_speed : ℝ)
  (jill_time_minutes : ℝ)
  (jack_time_minutes : ℝ) :
  distance = 2 →
  jill_speed = 15 →
  jack_speed = 6 →
  jill_time_minutes = (distance / jill_speed) * 60 →
  jack_time_minutes = (distance / jack_speed) * 60 →
  jack_time_minutes - jill_time_minutes = 12 :=
by
  sorry

end jill_arrives_before_jack_l57_5798
