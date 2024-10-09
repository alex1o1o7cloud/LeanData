import Mathlib

namespace geometric_series_proof_l299_29917

theorem geometric_series_proof (y : ℝ) :
  ((1 + (1/3) + (1/9) + (1/27) + ∑' n : ℕ, (1 / 3^(n+1))) * 
   (1 - (1/3) + (1/9) - (1/27) + ∑' n : ℕ, ((-1)^n * (1 / 3^(n+1)))) = 
   1 + (1/y) + (1/y^2) + (∑' n : ℕ, (1 / y^(n+1)))) → y = 9 := by
  sorry

end geometric_series_proof_l299_29917


namespace prism_sides_plus_two_l299_29969

theorem prism_sides_plus_two (E V S : ℕ) (h1 : E + V = 30) (h2 : E = 3 * S) (h3 : V = 2 * S) : S + 2 = 8 :=
by
  sorry

end prism_sides_plus_two_l299_29969


namespace problem1_problem2_l299_29933

theorem problem1 : (-1 / 2) * (-8) + (-6) = -2 := by
  sorry

theorem problem2 : -(1^4) - 2 / (-1 / 3) - abs (-9) = -4 := by
  sorry

end problem1_problem2_l299_29933


namespace john_walks_farther_l299_29976

theorem john_walks_farther :
  let john_distance : ℝ := 1.74
  let nina_distance : ℝ := 1.235
  john_distance - nina_distance = 0.505 :=
by
  sorry

end john_walks_farther_l299_29976


namespace marge_final_plants_l299_29989

-- Definitions corresponding to the conditions
def seeds_planted := 23
def seeds_never_grew := 5
def plants_grew := seeds_planted - seeds_never_grew
def plants_eaten := plants_grew / 3
def uneaten_plants := plants_grew - plants_eaten
def plants_strangled := uneaten_plants / 3
def survived_plants := uneaten_plants - plants_strangled
def effective_addition := 1

-- The main statement we need to prove
theorem marge_final_plants : 
  (plants_grew - plants_eaten - plants_strangled + effective_addition) = 9 := 
by
  sorry

end marge_final_plants_l299_29989


namespace grass_knot_segments_butterfly_knot_segments_l299_29960

-- Definitions for the grass knot problem
def outer_loops_cut : Nat := 5
def segments_after_outer_loops_cut : Nat := 6

-- Theorem for the grass knot
theorem grass_knot_segments (n : Nat) (h : n = outer_loops_cut) : (n + 1 = segments_after_outer_loops_cut) :=
sorry

-- Definitions for the butterfly knot problem
def butterfly_wings_loops_per_wing : Nat := 7
def segments_after_butterfly_wings_cut : Nat := 15

-- Theorem for the butterfly knot
theorem butterfly_knot_segments (w : Nat) (h : w = butterfly_wings_loops_per_wing) : ((w * 2 * 2 + 2) / 2 = segments_after_butterfly_wings_cut) :=
sorry

end grass_knot_segments_butterfly_knot_segments_l299_29960


namespace solution_l299_29937

noncomputable def problem (x : ℝ) : Prop :=
  (Real.sqrt (Real.sqrt (53 - 3 * x)) + Real.sqrt (Real.sqrt (39 + 3 * x))) = 5

theorem solution :
  ∀ x : ℝ, problem x → x = -23 / 3 :=
by
  intro x
  intro h
  sorry

end solution_l299_29937


namespace bobby_jumps_per_second_as_adult_l299_29914

-- Define the conditions as variables
def child_jumps_per_minute : ℕ := 30
def additional_jumps_as_adult : ℕ := 30

-- Theorem statement
theorem bobby_jumps_per_second_as_adult :
  (child_jumps_per_minute + additional_jumps_as_adult) / 60 = 1 :=
by
  -- placeholder for the proof
  sorry

end bobby_jumps_per_second_as_adult_l299_29914


namespace exists_overlapping_pairs_l299_29931

-- Definition of conditions:
def no_boy_danced_with_all_girls (B : Type) (G : Type) (danced : B → G → Prop) :=
  ∀ b : B, ∃ g : G, ¬ danced b g

def each_girl_danced_with_at_least_one_boy (B : Type) (G : Type) (danced : B → G → Prop) :=
  ∀ g : G, ∃ b : B, danced b g

-- The main theorem to prove:
theorem exists_overlapping_pairs
  (B : Type) (G : Type) (danced : B → G → Prop)
  (h1 : no_boy_danced_with_all_girls B G danced)
  (h2 : each_girl_danced_with_at_least_one_boy B G danced) :
  ∃ (b1 b2 : B) (g1 g2 : G), b1 ≠ b2 ∧ g1 ≠ g2 ∧ danced b1 g1 ∧ danced b2 g2 :=
sorry

end exists_overlapping_pairs_l299_29931


namespace seven_digit_palindromes_l299_29993

def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

theorem seven_digit_palindromes : 
  (∃ l : List ℕ, l = [1, 1, 4, 4, 4, 6, 6] ∧ 
  ∃ pl : List ℕ, pl.length = 7 ∧ is_palindrome pl ∧ 
  ∀ d, d ∈ pl → d ∈ l) →
  ∃! n, n = 12 :=
by
  sorry

end seven_digit_palindromes_l299_29993


namespace sqrt_12_same_type_sqrt_3_l299_29982

-- We define that two square roots are of the same type if one is a multiple of the other
def same_type (a b : ℝ) : Prop := ∃ k : ℝ, b = k * a

-- We need to show that sqrt(12) is of the same type as sqrt(3), and check options
theorem sqrt_12_same_type_sqrt_3 : same_type (Real.sqrt 3) (Real.sqrt 12) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 8) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 18) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 6) :=
by
  sorry -- Proof is omitted


end sqrt_12_same_type_sqrt_3_l299_29982


namespace percentage_calculation_l299_29932

theorem percentage_calculation 
  (number : ℝ)
  (h1 : 0.035 * number = 700) :
  0.024 * (1.5 * number) = 720 := 
by
  sorry

end percentage_calculation_l299_29932


namespace CDs_per_rack_l299_29938

theorem CDs_per_rack (racks_on_shelf : ℕ) (CDs_on_shelf : ℕ) (h1 : racks_on_shelf = 4) (h2 : CDs_on_shelf = 32) : 
  CDs_on_shelf / racks_on_shelf = 8 :=
by
  sorry

end CDs_per_rack_l299_29938


namespace jose_share_of_profit_l299_29925

def investment_months (amount : ℕ) (months : ℕ) : ℕ := amount * months

def profit_share (investment_months : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment_months * total_profit) / total_investment_months

theorem jose_share_of_profit :
  let tom_investment := 30000
  let jose_investment := 45000
  let total_profit := 36000
  let tom_months := 12
  let jose_months := 10
  let tom_investment_months := investment_months tom_investment tom_months
  let jose_investment_months := investment_months jose_investment jose_months
  let total_investment_months := tom_investment_months + jose_investment_months
  profit_share jose_investment_months total_investment_months total_profit = 20000 :=
by
  sorry

end jose_share_of_profit_l299_29925


namespace sum_of_three_consecutive_even_l299_29958

theorem sum_of_three_consecutive_even (a1 a2 a3 : ℤ) (h1 : a1 % 2 = 0) (h2 : a2 = a1 + 2) (h3 : a3 = a1 + 4) (h4 : a1 + a3 = 128) : a1 + a2 + a3 = 192 :=
sorry

end sum_of_three_consecutive_even_l299_29958


namespace count_zeros_in_10000_power_50_l299_29904

theorem count_zeros_in_10000_power_50 :
  10000^50 = 10^200 :=
by
  have h1 : 10000 = 10^4 := by sorry
  have h2 : (10^4)^50 = 10^(4 * 50) := by sorry
  exact h2.trans (by norm_num)

end count_zeros_in_10000_power_50_l299_29904


namespace find_q_l299_29959

theorem find_q {q : ℕ} (h : 27^8 = 9^q) : q = 12 := by
  sorry

end find_q_l299_29959


namespace min_a_squared_plus_b_squared_l299_29977

theorem min_a_squared_plus_b_squared (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 :=
sorry

end min_a_squared_plus_b_squared_l299_29977


namespace Jose_played_football_l299_29981

theorem Jose_played_football :
  ∀ (total_hours : ℝ) (basketball_minutes : ℕ) (minutes_per_hour : ℕ), total_hours = 1.5 → basketball_minutes = 60 →
  (total_hours * minutes_per_hour - basketball_minutes = 30) :=
by
  intros total_hours basketball_minutes minutes_per_hour h1 h2
  sorry

end Jose_played_football_l299_29981


namespace value_spent_more_than_l299_29940

theorem value_spent_more_than (x : ℕ) (h : 8 * 12 + (x + 8) = 117) : x = 13 :=
by
  sorry

end value_spent_more_than_l299_29940


namespace balloons_per_school_l299_29975

theorem balloons_per_school (yellow black total : ℕ) 
  (hyellow : yellow = 3414)
  (hblack : black = yellow + 1762)
  (htotal : total = yellow + black)
  (hdivide : total % 10 = 0) : 
  total / 10 = 859 :=
by sorry

end balloons_per_school_l299_29975


namespace possible_values_of_a₁_l299_29924

-- Define arithmetic progression with first term a₁ and common difference d
def arithmetic_progression (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

-- Define the sum of the first 7 terms of the arithmetic progression
def sum_first_7_terms (a₁ d : ℤ) : ℤ := 7 * a₁ + 21 * d

-- Define the conditions given
def condition1 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 7) * (arithmetic_progression a₁ d 12) > (sum_first_7_terms a₁ d) + 20

def condition2 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 9) * (arithmetic_progression a₁ d 10) < (sum_first_7_terms a₁ d) + 44

-- The main problem to prove
def problem (a₁ : ℤ) (d : ℤ) : Prop := 
  condition1 a₁ d ∧ condition2 a₁ d

-- The theorem statement to prove
theorem possible_values_of_a₁ (a₁ d : ℤ) : problem a₁ d → a₁ = -9 ∨ a₁ = -8 ∨ a₁ = -7 ∨ a₁ = -6 ∨ a₁ = -4 ∨ a₁ = -3 ∨ a₁ = -2 ∨ a₁ = -1 := 
by sorry

end possible_values_of_a₁_l299_29924


namespace negation_of_existence_l299_29912

theorem negation_of_existence (T : Type) (triangle : T → Prop) (sum_interior_angles : T → ℝ) :
  (¬ ∃ t : T, sum_interior_angles t ≠ 180) ↔ (∀ t : T, sum_interior_angles t = 180) :=
by 
  sorry

end negation_of_existence_l299_29912


namespace area_of_side_face_of_box_l299_29970

theorem area_of_side_face_of_box:
  ∃ (l w h : ℝ), (w * h = (1/2) * (l * w)) ∧
                 (l * w = 1.5 * (l * h)) ∧
                 (l * w * h = 3000) ∧
                 ((l * h) = 200) :=
sorry

end area_of_side_face_of_box_l299_29970


namespace athleteA_time_to_complete_race_l299_29966

theorem athleteA_time_to_complete_race
    (v : ℝ)
    (t : ℝ)
    (h1 : v = 1000 / t)
    (h2 : v = 948 / (t + 18)) :
    t = 18000 / 52 := by
  sorry

end athleteA_time_to_complete_race_l299_29966


namespace triangle_subsegment_length_l299_29934

noncomputable def length_of_shorter_subsegment (PQ QR PR PS SR : ℝ) :=
  PQ < QR ∧ 
  PR = 15 ∧ 
  PQ / QR = 1 / 5 ∧ 
  PS + SR = PR ∧ 
  PS = PQ / QR * SR → 
  PS = 5 / 2

theorem triangle_subsegment_length (PQ QR PR PS SR : ℝ) 
  (h1 : PQ < QR) 
  (h2 : PR = 15) 
  (h3 : PQ / QR = 1 / 5) 
  (h4 : PS + SR = PR) 
  (h5 : PS = PQ / QR * SR) : 
  length_of_shorter_subsegment PQ QR PR PS SR := 
sorry

end triangle_subsegment_length_l299_29934


namespace no_correct_option_l299_29939

-- Define the given table as a list of pairs
def table :=
  [(1, -2), (2, 0), (3, 2), (4, 6), (5, 12), (6, 20)]

-- Define the given functions as potential options
def optionA (x : ℕ) : ℤ := x^2 - 5 * x + 4
def optionB (x : ℕ) : ℤ := x^2 - 3 * x
def optionC (x : ℕ) : ℤ := x^3 - 3 * x^2 + 2 * x
def optionD (x : ℕ) : ℤ := 2 * x^2 - 4 * x - 2
def optionE (x : ℕ) : ℤ := x^2 - 4 * x + 2

-- Prove that there is no correct option among the given options that matches the table
theorem no_correct_option : 
  ¬(∀ p ∈ table, p.snd = optionA p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionB p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionC p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionD p.fst) ∧
  ¬(∀ p ∈ table, p.snd = optionE p.fst) :=
by sorry

end no_correct_option_l299_29939


namespace rational_number_div_eq_l299_29921

theorem rational_number_div_eq :
  ∃ x : ℚ, (-2 : ℚ) / x = 8 ∧ x = -1 / 4 :=
by
  existsi (-1 / 4 : ℚ)
  sorry

end rational_number_div_eq_l299_29921


namespace find_a_minus_b_l299_29915

-- Given definitions for conditions
variables (a b : ℤ)

-- Given conditions as hypotheses
def condition1 := a + 2 * b = 5
def condition2 := a * b = -12

theorem find_a_minus_b (h1 : condition1 a b) (h2 : condition2 a b) : a - b = -7 :=
sorry

end find_a_minus_b_l299_29915


namespace linear_regression_intercept_l299_29907

theorem linear_regression_intercept :
  let x_values := [1, 2, 3, 4, 5]
  let y_values := [0.5, 0.8, 1.0, 1.2, 1.5]
  let x_mean := (x_values.sum / x_values.length : ℝ)
  let y_mean := (y_values.sum / y_values.length : ℝ)
  let slope := 0.24
  (x_mean = 3) →
  (y_mean = 1) →
  y_mean = slope * x_mean + 0.28 :=
by
  sorry

end linear_regression_intercept_l299_29907


namespace parabola_hyperbola_tangent_l299_29990

noncomputable def parabola : ℝ → ℝ := λ x => x^2 + 5

noncomputable def hyperbola (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y^2 - m * x^2 = 1

theorem parabola_hyperbola_tangent (m : ℝ) : 
  (m = 10 + 4*Real.sqrt 6 ∨ m = 10 - 4*Real.sqrt 6) →
  ∃ x y, parabola x = y ∧ hyperbola m x y ∧ 
    ∃ c b a, a * y^2 + b * y + c = 0 ∧ a = 1 ∧ c = 5 * m - 1 ∧ b = -m ∧ b^2 - 4*a*c = 0 :=
by
  sorry

end parabola_hyperbola_tangent_l299_29990


namespace puppy_weight_is_3_8_l299_29985

noncomputable def puppy_weight_problem (p s l : ℝ) : Prop :=
  p + 2 * s + l = 38 ∧
  p + l = 3 * s ∧
  p + 2 * s = l

theorem puppy_weight_is_3_8 :
  ∃ p s l : ℝ, puppy_weight_problem p s l ∧ p = 3.8 :=
by
  sorry

end puppy_weight_is_3_8_l299_29985


namespace point_not_on_graph_and_others_on_l299_29979

theorem point_not_on_graph_and_others_on (y : ℝ → ℝ) (h₁ : ∀ x, y x = x / (x - 1))
  : ¬ (1 = (1 : ℝ) / ((1 : ℝ) - 1)) 
  ∧ (2 = (2 : ℝ) / ((2 : ℝ) - 1)) 
  ∧ ((-1 : ℝ) = (1/2 : ℝ) / ((1/2 : ℝ) - 1)) 
  ∧ (0 = (0 : ℝ) / ((0 : ℝ) - 1)) 
  ∧ (3/2 = (3 : ℝ) / ((3 : ℝ) - 1)) := 
sorry

end point_not_on_graph_and_others_on_l299_29979


namespace complement_of_A_in_U_l299_29995

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 6}
def complement : Set ℕ := U \ A

theorem complement_of_A_in_U :
  complement = {1, 3, 5} := by
  sorry

end complement_of_A_in_U_l299_29995


namespace third_of_ten_l299_29920

theorem third_of_ten : (1/3 : ℝ) * 10 = 8 / 3 :=
by
  have h : (1/4 : ℝ) * 20 = 4 := by sorry
  sorry

end third_of_ten_l299_29920


namespace length_of_chord_EF_l299_29965

theorem length_of_chord_EF 
  (rO rN rP : ℝ)
  (AB BC CD : ℝ)
  (AG_EF_intersec_E AG_EF_intersec_F : ℝ)
  (EF : ℝ)
  (cond1 : rO = 10)
  (cond2 : rN = 20)
  (cond3 : rP = 30)
  (cond4 : AB = 2 * rO)
  (cond5 : BC = 2 * rN)
  (cond6 : CD = 2 * rP)
  (cond7 : EF = 6 * Real.sqrt (24 + 2/3)) :
  EF = 6 * Real.sqrt 24.6666 := sorry

end length_of_chord_EF_l299_29965


namespace polynomial_divisible_by_cube_l299_29918

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := 
  n^2 * x^(n+2) - (2 * n^2 + 2 * n - 1) * x^(n+1) + (n + 1)^2 * x^n - x - 1

theorem polynomial_divisible_by_cube (n : ℕ) (h : n > 0) : 
  ∃ Q, P n x = (x - 1)^3 * Q :=
sorry

end polynomial_divisible_by_cube_l299_29918


namespace prove_odd_function_definition_l299_29909

theorem prove_odd_function_definition (f : ℝ → ℝ) 
  (odd : ∀ x : ℝ, f (-x) = -f x)
  (pos_def : ∀ x : ℝ, 0 < x → f x = 2 * x ^ 2 - x + 1) :
  ∀ x : ℝ, x < 0 → f x = -2 * x ^ 2 - x - 1 :=
by
  intro x hx
  sorry

end prove_odd_function_definition_l299_29909


namespace price_reduction_2100_yuan_l299_29900

-- Definitions based on conditions
def initial_sales : ℕ := 30
def initial_profit_per_item : ℕ := 50
def additional_sales_per_yuan (x : ℕ) : ℕ := 2 * x
def new_profit_per_item (x : ℕ) : ℕ := 50 - x
def target_profit : ℕ := 2100

-- Final proof statement, showing the price reduction needed
theorem price_reduction_2100_yuan (x : ℕ) 
  (h : (50 - x) * (30 + 2 * x) = 2100) : 
  x = 20 := 
by 
  sorry

end price_reduction_2100_yuan_l299_29900


namespace percentage_of_water_in_mixture_l299_29943

-- Definitions based on conditions from a)
def original_price : ℝ := 1 -- assuming $1 per liter for pure dairy
def selling_price : ℝ := 1.25 -- 25% profit means selling at $1.25
def profit_percentage : ℝ := 0.25 -- 25% profit

-- Theorem statement based on the equivalent problem in c)
theorem percentage_of_water_in_mixture : 
  (selling_price - original_price) / selling_price * 100 = 20 :=
by
  sorry

end percentage_of_water_in_mixture_l299_29943


namespace Luka_water_requirement_l299_29961

-- Declare variables and conditions
variables (L S W O : ℕ)  -- All variables are natural numbers
-- Conditions
variable (h1 : S = 2 * L)  -- Twice as much sugar as lemon juice
variable (h2 : W = 5 * S)  -- 5 times as much water as sugar
variable (h3 : O = S)      -- Orange juice equals the amount of sugar 
variable (L_eq_5 : L = 5)  -- Lemon juice is 5 cups

-- The goal statement to prove
theorem Luka_water_requirement : W = 50 :=
by
  -- Note: The proof steps would go here, but as per instructions, we leave it as sorry.
  sorry

end Luka_water_requirement_l299_29961


namespace Sheila_weekly_earnings_l299_29964

-- Definitions based on the conditions
def hours_per_day_MWF : ℕ := 8
def hours_per_day_TT : ℕ := 6
def hourly_wage : ℕ := 7
def days_MWF : ℕ := 3
def days_TT : ℕ := 2

-- Theorem that Sheila earns $252 per week
theorem Sheila_weekly_earnings : (hours_per_day_MWF * hourly_wage * days_MWF) + (hours_per_day_TT * hourly_wage * days_TT) = 252 :=
by 
  sorry

end Sheila_weekly_earnings_l299_29964


namespace total_copper_mined_l299_29991

theorem total_copper_mined :
  let daily_production_A := 4500
  let daily_production_B := 6000
  let daily_production_C := 5000
  let daily_production_D := 3500
  let copper_percentage_A := 0.055
  let copper_percentage_B := 0.071
  let copper_percentage_C := 0.147
  let copper_percentage_D := 0.092
  (daily_production_A * copper_percentage_A +
   daily_production_B * copper_percentage_B +
   daily_production_C * copper_percentage_C +
   daily_production_D * copper_percentage_D) = 1730.5 :=
by
  sorry

end total_copper_mined_l299_29991


namespace least_number_divisible_l299_29998

theorem least_number_divisible (n : ℕ) :
  ((∀ d ∈ [24, 32, 36, 54, 72, 81, 100], (n + 21) % d = 0) ↔ n = 64779) :=
sorry

end least_number_divisible_l299_29998


namespace find_k_l299_29956

variable (a b : ℝ → ℝ → ℝ)
variable {k : ℝ}

-- Defining conditions
axiom a_perpendicular_b : ∀ x y, a x y = 0
axiom a_unit_vector : a 1 0 = 1
axiom b_unit_vector : b 0 1 = 1
axiom sum_perpendicular_to_k_diff : ∀ x y, (a x y + b x y) * (k * a x y - b x y) = 0

theorem find_k : k = 1 :=
sorry

end find_k_l299_29956


namespace mix_ratios_l299_29916

theorem mix_ratios (milk1 water1 milk2 water2 : ℕ) 
  (h1 : milk1 = 7) (h2 : water1 = 2)
  (h3 : milk2 = 8) (h4 : water2 = 1) :
  (milk1 + milk2) / (water1 + water2) = 5 :=
by
  -- Proof required here
  sorry

end mix_ratios_l299_29916


namespace find_F_l299_29951

-- Define the condition and the equation
def C (F : ℤ) : ℤ := (5 * (F - 30)) / 9

-- Define the assumption that C = 25
def C_condition : ℤ := 25

-- The theorem to prove that F = 75 given the conditions
theorem find_F (F : ℤ) (h : C F = C_condition) : F = 75 :=
sorry

end find_F_l299_29951


namespace fraction_apple_juice_in_mixture_l299_29973

theorem fraction_apple_juice_in_mixture :
  let pitcher1_capacity := 800
  let pitcher2_capacity := 500
  let fraction_juice_pitcher1 := (1 : ℚ) / 4
  let fraction_juice_pitcher2 := (3 : ℚ) / 8
  let apple_juice_pitcher1 := pitcher1_capacity * fraction_juice_pitcher1
  let apple_juice_pitcher2 := pitcher2_capacity * fraction_juice_pitcher2
  let total_apple_juice := apple_juice_pitcher1 + apple_juice_pitcher2
  let total_capacity := pitcher1_capacity + pitcher2_capacity
  (total_apple_juice / total_capacity = 31 / 104) :=
by
  sorry

end fraction_apple_juice_in_mixture_l299_29973


namespace unique_non_overtaken_city_l299_29901

structure City :=
(size_left : ℕ)
(size_right : ℕ)

def canOvertake (A B : City) : Prop :=
  A.size_right > B.size_left 

theorem unique_non_overtaken_city (n : ℕ) (H : n > 0) (cities : Fin n → City) : 
  ∃! i : Fin n, ∀ j : Fin n, ¬ canOvertake (cities j) (cities i) :=
by
  sorry

end unique_non_overtaken_city_l299_29901


namespace total_erasers_is_35_l299_29911

def Celine : ℕ := 10

def Gabriel : ℕ := Celine / 2

def Julian : ℕ := Celine * 2

def total_erasers : ℕ := Celine + Gabriel + Julian

theorem total_erasers_is_35 : total_erasers = 35 :=
  by
  sorry

end total_erasers_is_35_l299_29911


namespace negation_of_universal_proposition_l299_29941

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 - 2 * x + 3 ≥ 0) ↔ ∃ x : ℝ, x^2 - 2 * x + 3 < 0 := 
sorry

end negation_of_universal_proposition_l299_29941


namespace largest_integer_y_l299_29999

theorem largest_integer_y (y : ℤ) : (y / (4:ℚ) + 3 / 7 < 2 / 3) → y ≤ 0 :=
by
  sorry

end largest_integer_y_l299_29999


namespace larger_rectangle_area_l299_29928

/-- Given a smaller rectangle made out of three squares each of area 25 cm²,
    where two vertices of the smaller rectangle lie on the midpoints of the
    shorter sides of the larger rectangle and the other two vertices lie on
    the longer sides, prove the area of the larger rectangle is 150 cm². -/
theorem larger_rectangle_area (s : ℝ) (l W S_Larger W_Larger : ℝ)
  (h_s : s^2 = 25) 
  (h_small_dim : l = 3 * s ∧ W = s ∧ l * W = 3 * s^2) 
  (h_vertices : 2 * W = W_Larger ∧ l = S_Larger) :
  (S_Larger * W_Larger = 150) := 
by
  sorry

end larger_rectangle_area_l299_29928


namespace travel_from_A_to_C_l299_29906

def num_ways_A_to_B : ℕ := 5 + 2  -- 5 buses and 2 trains
def num_ways_B_to_C : ℕ := 3 + 2  -- 3 buses and 2 ferries

theorem travel_from_A_to_C :
  num_ways_A_to_B * num_ways_B_to_C = 35 :=
by
  -- The proof environment will be added here. 
  -- We include 'sorry' here for now.
  sorry

end travel_from_A_to_C_l299_29906


namespace value_of_expression_when_x_is_3_l299_29947

theorem value_of_expression_when_x_is_3 :
  (3^6 - 6*3 = 711) :=
by
  sorry

end value_of_expression_when_x_is_3_l299_29947


namespace jack_walking_rate_l299_29984

variables (distance : ℝ) (time_hours : ℝ)
#check distance  -- ℝ (real number)
#check time_hours  -- ℝ (real number)

-- Define the conditions
def jack_distance : Prop := distance = 9
def jack_time : Prop := time_hours = 1 + 15 / 60

-- Define the statement to prove
theorem jack_walking_rate (h1 : jack_distance distance) (h2 : jack_time time_hours) :
  (distance / time_hours) = 7.2 :=
sorry

end jack_walking_rate_l299_29984


namespace gcd_sum_and_lcm_eq_gcd_l299_29919

theorem gcd_sum_and_lcm_eq_gcd (a b : ℤ) :  Int.gcd (a + b) (Int.lcm a b) = Int.gcd a b :=
sorry

end gcd_sum_and_lcm_eq_gcd_l299_29919


namespace condition_equiv_l299_29922

theorem condition_equiv (p q : Prop) : (¬ (p ∧ q) ∧ (p ∨ q)) ↔ ((p ∨ q) ∧ (¬ p ↔ q)) :=
  sorry

end condition_equiv_l299_29922


namespace max_principals_in_10_years_l299_29953

theorem max_principals_in_10_years (h : ∀ p : ℕ, 4 * p ≤ 10) :
  ∃ n : ℕ, n ≤ 3 ∧ n = 3 :=
sorry

end max_principals_in_10_years_l299_29953


namespace problem_solution_l299_29962

def arithmetic_sequence (a_1 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a_1 + (n - 1) * d

def sum_of_terms (a_1 : ℕ) (a_n : ℕ) (n : ℕ) : ℕ :=
  n * (a_1 + a_n) / 2

theorem problem_solution 
  (a_1 : ℕ) (d : ℕ) (a_n : ℕ) (S_n : ℕ)
  (h1 : a_1 = 2)
  (h2 : S_2 = arithmetic_sequence a_1 d 3):
  a_2 = 4 ∧ S_10 = 110 :=
by
  sorry

end problem_solution_l299_29962


namespace Barons_theorem_correct_l299_29902

theorem Barons_theorem_correct (a b : ℕ) (ha: 0 < a) (hb: 0 < b) : 
  ∃ n : ℕ, 0 < n ∧ ∃ k1 k2 : ℕ, an = k1 ^ 2 ∧ bn = k2 ^ 3 := 
sorry

end Barons_theorem_correct_l299_29902


namespace spring_length_function_l299_29936

noncomputable def spring_length (x : ℝ) : ℝ :=
  12 + 3 * x

theorem spring_length_function :
  ∀ (x : ℝ), spring_length x = 12 + 3 * x :=
by
  intro x
  rfl

end spring_length_function_l299_29936


namespace number_of_stadiums_to_visit_l299_29997

def average_cost_per_stadium : ℕ := 900
def annual_savings : ℕ := 1500
def years_saving : ℕ := 18

theorem number_of_stadiums_to_visit (c : ℕ) (s : ℕ) (n : ℕ) (h1 : c = average_cost_per_stadium) (h2 : s = annual_savings) (h3 : n = years_saving) : n * s / c = 30 := 
by 
  rw [h1, h2, h3]
  exact sorry

end number_of_stadiums_to_visit_l299_29997


namespace more_non_product_eight_digit_numbers_l299_29927

def num_eight_digit_numbers := 10^8 - 10^7
def num_four_digit_numbers := 9999 - 1000 + 1
def num_unique_products := (num_four_digit_numbers.choose 2) + num_four_digit_numbers

theorem more_non_product_eight_digit_numbers :
  (num_eight_digit_numbers - num_unique_products) > num_unique_products := by sorry

end more_non_product_eight_digit_numbers_l299_29927


namespace remainder_when_7n_divided_by_4_l299_29944

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l299_29944


namespace jason_total_spent_l299_29930

-- Conditions
def shorts_cost : ℝ := 14.28
def jacket_cost : ℝ := 4.74

-- Statement to prove
theorem jason_total_spent : shorts_cost + jacket_cost = 19.02 := by
  -- Proof to be filled in
  sorry

end jason_total_spent_l299_29930


namespace ThreeStudentsGotA_l299_29903

-- Definitions of students receiving A grades
variable (Edward Fiona George Hannah Ian : Prop)

-- Conditions given in the problem
axiom H1 : Edward → Fiona
axiom H2 : Fiona → George
axiom H3 : George → Hannah
axiom H4 : Hannah → Ian
axiom H5 : (Edward → False) ∧ (Fiona → False)

-- Theorem stating the final result
theorem ThreeStudentsGotA : (George ∧ Hannah ∧ Ian) ∧ 
                            (¬Edward ∧ ¬Fiona) ∧ 
                            (Edward ∨ Fiona ∨ George ∨ Hannah ∨ Ian) :=
by
  sorry

end ThreeStudentsGotA_l299_29903


namespace levi_additional_baskets_to_score_l299_29994

def levi_scored_initial := 8
def brother_scored_initial := 12
def brother_likely_to_score := 3
def levi_goal_margin := 5

theorem levi_additional_baskets_to_score : 
  levi_scored_initial + 12 >= brother_scored_initial + brother_likely_to_score + levi_goal_margin :=
by
  sorry

end levi_additional_baskets_to_score_l299_29994


namespace ratio_equivalence_l299_29929

theorem ratio_equivalence (a b : ℝ) (hb : b ≠ 0) (h : a / b = 5 / 4) : (4 * a + 3 * b) / (4 * a - 3 * b) = 4 :=
sorry

end ratio_equivalence_l299_29929


namespace range_of_a_l299_29983

-- Defining the function f : ℝ → ℝ
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x + a * Real.log x

-- Main theorem statement
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (f a x1 = 0 ∧ f a x2 = 0)) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l299_29983


namespace sum_of_ages_l299_29949

theorem sum_of_ages {a b c : ℕ} (h1 : a * b * c = 72) (h2 : b < a) (h3 : a < c) : a + b + c = 13 :=
sorry

end sum_of_ages_l299_29949


namespace area_under_cos_l299_29950

theorem area_under_cos :
  ∫ x in (0 : ℝ)..(3 * Real.pi / 2), |Real.cos x| = 3 :=
by
  sorry

end area_under_cos_l299_29950


namespace distance_covered_at_40_kmph_l299_29980

theorem distance_covered_at_40_kmph (x : ℝ) (h : 0 ≤ x ∧ x ≤ 250) 
  (total_distance : x + (250 - x) = 250) 
  (total_time : x / 40 + (250 - x) / 60 = 5.5) : 
  x = 160 :=
sorry

end distance_covered_at_40_kmph_l299_29980


namespace min_value_at_2_l299_29996

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_at_2 : ∃ x : ℝ, f x = 2 :=
sorry

end min_value_at_2_l299_29996


namespace unique_integral_root_of_equation_l299_29957

theorem unique_integral_root_of_equation :
  ∀ x : ℤ, (x - 9 / (x - 5) = 7 - 9 / (x - 5)) ↔ (x = 7) :=
by
  sorry

end unique_integral_root_of_equation_l299_29957


namespace inequality_add_one_l299_29971

variable {α : Type*} [LinearOrderedField α]

theorem inequality_add_one {a b : α} (h : a > b) : a + 1 > b + 1 :=
sorry

end inequality_add_one_l299_29971


namespace find_x_l299_29954

noncomputable def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

noncomputable def vec_dot (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt ((v.1)^2 + (v.2)^2)

theorem find_x (x : ℝ) (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (1, x)) 
  (h3 : magnitude (vec_sub a b) = vec_dot a b) : 
  x = 1 / 3 :=
by
  sorry

end find_x_l299_29954


namespace family_ages_l299_29988

-- Define the conditions
variables (D M S F : ℕ)

-- Condition 1: In the year 2000, the mother was 4 times the daughter's age.
axiom mother_age : M = 4 * D

-- Condition 2: In the year 2000, the father was 6 times the son's age.
axiom father_age : F = 6 * S

-- Condition 3: The son is 1.5 times the age of the daughter.
axiom son_age_ratio : S = 3 * D / 2

-- Condition 4: In the year 2010, the father became twice the mother's age.
axiom father_mother_2010 : F + 10 = 2 * (M + 10)

-- Condition 5: The age gap between the mother and father has always been the same.
axiom age_gap_constant : F - M = (F + 10) - (M + 10)

-- Define the theorem
theorem family_ages :
  D = 10 ∧ S = 15 ∧ M = 40 ∧ F = 90 ∧ (F - M = 50) := sorry

end family_ages_l299_29988


namespace gcd_13642_19236_34176_l299_29955

theorem gcd_13642_19236_34176 : Int.gcd (Int.gcd 13642 19236) 34176 = 2 := 
sorry

end gcd_13642_19236_34176_l299_29955


namespace beka_flies_more_l299_29926

-- Definitions
def beka_flight_distance : ℕ := 873
def jackson_flight_distance : ℕ := 563

-- The theorem we need to prove
theorem beka_flies_more : beka_flight_distance - jackson_flight_distance = 310 :=
by
  sorry

end beka_flies_more_l299_29926


namespace intersection_point_l299_29935

noncomputable def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem intersection_point :
  ∃ a : ℝ, g a = a ∧ a = -3 :=
by
  sorry

end intersection_point_l299_29935


namespace platform_length_1000_l299_29908

open Nat Real

noncomputable def length_of_platform (train_length : ℝ) (time_pole : ℝ) (time_platform : ℝ) : ℝ :=
  let speed := train_length / time_pole
  let platform_length := (speed * time_platform) - train_length
  platform_length

theorem platform_length_1000 :
  length_of_platform 300 9 39 = 1000 := by
  sorry

end platform_length_1000_l299_29908


namespace find_stream_speed_l299_29986

variable (r w : ℝ)

noncomputable def stream_speed:
    Prop := 
    (21 / (r + w) + 4 = 21 / (r - w)) ∧ 
    (21 / (3 * r + w) + 0.5 = 21 / (3 * r - w)) ∧ 
    w = 3 

theorem find_stream_speed : ∃ w, stream_speed r w := 
by
  sorry

end find_stream_speed_l299_29986


namespace total_number_of_eyes_l299_29942

theorem total_number_of_eyes (n_spiders n_ants eyes_per_spider eyes_per_ant : ℕ)
  (h1 : n_spiders = 3) (h2 : n_ants = 50) (h3 : eyes_per_spider = 8) (h4 : eyes_per_ant = 2) :
  (n_spiders * eyes_per_spider + n_ants * eyes_per_ant) = 124 :=
by
  sorry

end total_number_of_eyes_l299_29942


namespace pages_revised_only_once_l299_29952

variable (x : ℕ)

def rate_first_time_typing := 6
def rate_revision := 4
def total_pages := 100
def pages_revised_twice := 15
def total_cost := 860

theorem pages_revised_only_once : 
  rate_first_time_typing * total_pages 
  + rate_revision * x 
  + rate_revision * pages_revised_twice * 2 
  = total_cost 
  → x = 35 :=
by
  sorry

end pages_revised_only_once_l299_29952


namespace max_value_of_exp_diff_l299_29923

open Real

theorem max_value_of_exp_diff : ∀ x : ℝ, ∃ y : ℝ, y = 2^x - 4^x ∧ y ≤ 1/4 := sorry

end max_value_of_exp_diff_l299_29923


namespace at_least_one_not_less_than_neg_two_l299_29948

theorem at_least_one_not_less_than_neg_two (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≥ -2 ∨ b + 1/c ≥ -2 ∨ c + 1/a ≥ -2) :=
sorry

end at_least_one_not_less_than_neg_two_l299_29948


namespace banana_pie_angle_l299_29992

theorem banana_pie_angle
  (total_students : ℕ := 48)
  (chocolate_students : ℕ := 15)
  (apple_students : ℕ := 10)
  (blueberry_students : ℕ := 9)
  (remaining_students := total_students - (chocolate_students + apple_students + blueberry_students))
  (banana_students := remaining_students / 2) :
  (banana_students : ℝ) / total_students * 360 = 52.5 :=
by
  sorry

end banana_pie_angle_l299_29992


namespace solve_mod_equiv_l299_29974

theorem solve_mod_equiv : ∃ (n : ℤ), 0 ≤ n ∧ n < 9 ∧ (-2222 ≡ n [ZMOD 9]) → n = 6 := by
  sorry

end solve_mod_equiv_l299_29974


namespace quadratic_has_one_solution_positive_value_of_n_l299_29905

theorem quadratic_has_one_solution_positive_value_of_n :
  ∃ n : ℝ, (4 * x ^ 2 + n * x + 1 = 0 → n ^ 2 - 16 = 0) ∧ n > 0 ∧ n = 4 :=
sorry

end quadratic_has_one_solution_positive_value_of_n_l299_29905


namespace value_of_f_of_1_plus_g_of_2_l299_29978

def f (x : ℝ) := 2 * x - 3
def g (x : ℝ) := x + 1

theorem value_of_f_of_1_plus_g_of_2 : f (1 + g 2) = 5 :=
by
  sorry

end value_of_f_of_1_plus_g_of_2_l299_29978


namespace k_range_l299_29910

noncomputable def valid_k (k : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → x / Real.exp x < 1 / (k + 2 * x - x^2)

theorem k_range : {k : ℝ | valid_k k} = {k : ℝ | 0 ≤ k ∧ k < Real.exp 1 - 1} :=
by sorry

end k_range_l299_29910


namespace diesel_fuel_usage_l299_29913

theorem diesel_fuel_usage (weekly_spending : ℝ) (cost_per_gallon : ℝ) (weeks : ℝ) (result : ℝ): 
  weekly_spending = 36 → cost_per_gallon = 3 → weeks = 2 → result = 24 → 
  (weekly_spending / cost_per_gallon) * weeks = result :=
by
  intros
  sorry

end diesel_fuel_usage_l299_29913


namespace quadratic_one_solution_m_l299_29946

theorem quadratic_one_solution_m (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 7 * x + m = 0) → 
  (∀ (x y : ℝ), 3 * x^2 - 7 * x + m = 0 → 3 * y^2 - 7 * y + m = 0 → x = y) → 
  m = 49 / 12 :=
by
  sorry

end quadratic_one_solution_m_l299_29946


namespace maximize_x3y4_l299_29967

noncomputable def max_product (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 50) : ℝ :=
  x^3 * y^4

theorem maximize_x3y4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 50) :
  max_product x y hx hy h ≤ max_product (150/7) (200/7) (by norm_num) (by norm_num) (by norm_num) :=
  sorry

end maximize_x3y4_l299_29967


namespace find_length_AB_l299_29987

-- Definitions for the problem conditions.
def angle_B : ℝ := 90
def angle_A : ℝ := 30
def BC : ℝ := 24

-- Main theorem to prove.
theorem find_length_AB (angle_B_eq : angle_B = 90) (angle_A_eq : angle_A = 30) (BC_eq : BC = 24) : 
  ∃ AB : ℝ, AB = 12 := 
by
  sorry

end find_length_AB_l299_29987


namespace student_number_choice_l299_29945

theorem student_number_choice (x : ℤ) (h : 3 * x - 220 = 110) : x = 110 :=
by sorry

end student_number_choice_l299_29945


namespace dot_product_a_a_sub_2b_l299_29968

-- Define the vectors a and b
def a : (ℝ × ℝ) := (2, 3)
def b : (ℝ × ℝ) := (-1, 2)

-- Define the subtraction of vector a and 2 * vector b
def a_sub_2b : (ℝ × ℝ) := (a.1 - 2 * b.1, a.2 - 2 * b.2)

-- Define the dot product of two vectors
def dot_product (u v : (ℝ × ℝ)) : ℝ := u.1 * v.1 + u.2 * v.2

-- State that the dot product of a and (a - 2b) is 5
theorem dot_product_a_a_sub_2b : dot_product a a_sub_2b = 5 := 
by 
  -- proof omitted
  sorry

end dot_product_a_a_sub_2b_l299_29968


namespace stephen_speed_second_third_l299_29972

theorem stephen_speed_second_third
  (first_third_speed : ℝ)
  (last_third_speed : ℝ)
  (total_distance : ℝ)
  (travel_time : ℝ)
  (time_in_hours : ℝ)
  (h1 : first_third_speed = 16)
  (h2 : last_third_speed = 20)
  (h3 : total_distance = 12)
  (h4 : travel_time = 15)
  (h5 : time_in_hours = travel_time / 60) :
  time_in_hours * (total_distance - (first_third_speed * time_in_hours + last_third_speed * time_in_hours)) = 12 := 
by 
  sorry

end stephen_speed_second_third_l299_29972


namespace remainder_is_one_l299_29963

theorem remainder_is_one (N : ℤ) (R : ℤ)
  (h1 : N % 100 = R)
  (h2 : N % R = 1) :
  R = 1 :=
by
  sorry

end remainder_is_one_l299_29963
