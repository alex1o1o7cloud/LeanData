import Mathlib

namespace domain_inequality_l1934_193491

theorem domain_inequality (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (1 - m) * x + m ≥ 0) ↔ (m ≥ 1/3) :=
by
  sorry

end domain_inequality_l1934_193491


namespace smallest_value_N_l1934_193418

theorem smallest_value_N (N : ℕ) (a b c : ℕ) (h1 : N = a * b * c) (h2 : (a - 1) * (b - 1) * (c - 1) = 252) : N = 392 :=
sorry

end smallest_value_N_l1934_193418


namespace postage_arrangements_11_cents_l1934_193408

-- Definitions for the problem settings, such as stamp denominations and counts
def stamp_collection : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

-- Function to calculate all unique arrangements of stamps that sum to a given value (11 cents)
def count_arrangements (total_cents : ℕ) : ℕ :=
  -- The implementation would involve a combinatorial counting taking into account the problem conditions
  sorry

-- The main theorem statement asserting the solution
theorem postage_arrangements_11_cents :
  count_arrangements 11 = 71 :=
  sorry

end postage_arrangements_11_cents_l1934_193408


namespace razorback_tshirt_profit_l1934_193434

theorem razorback_tshirt_profit :
  let profit_per_tshirt := 9
  let cost_per_tshirt := 4
  let num_tshirts_sold := 245
  let discount := 0.2
  let selling_price := profit_per_tshirt + cost_per_tshirt
  let discount_amount := discount * selling_price
  let discounted_price := selling_price - discount_amount
  let total_revenue := discounted_price * num_tshirts_sold
  let total_production_cost := cost_per_tshirt * num_tshirts_sold
  let total_profit := total_revenue - total_production_cost
  total_profit = 1568 :=
by
  sorry

end razorback_tshirt_profit_l1934_193434


namespace opposite_of_2023_l1934_193475

def opposite (n x : ℤ) := n + x = 0 

theorem opposite_of_2023 : ∃ x : ℤ, opposite 2023 x ∧ x = -2023 := 
by
  sorry

end opposite_of_2023_l1934_193475


namespace min_edge_disjoint_cycles_l1934_193483

noncomputable def minEdgesForDisjointCycles (n : ℕ) (h : n ≥ 6) : ℕ := 3 * (n - 2)

theorem min_edge_disjoint_cycles (n : ℕ) (h : n ≥ 6) : minEdgesForDisjointCycles n h = 3 * (n - 2) := 
by
  sorry

end min_edge_disjoint_cycles_l1934_193483


namespace ellipse_slope_product_constant_l1934_193416

noncomputable def ellipse_constant_slope_product (a b : ℝ) (P M : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧
  (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
  (N.1 = -M.1 ∧ N.2 = -M.2) ∧
  (∃ k_PM k_PN : ℝ, k_PM = (P.2 - M.2) / (P.1 - M.1) ∧ k_PN = (P.2 - N.2) / (P.1 - N.1)) ∧
  ((P.2 - M.2) / (P.1 - M.1) * (P.2 - N.2) / (P.1 - N.1) = -b^2 / a^2)

theorem ellipse_slope_product_constant (a b : ℝ) (P M N : ℝ × ℝ) :
  ellipse_constant_slope_product a b P M N := 
sorry

end ellipse_slope_product_constant_l1934_193416


namespace value_of_f_g_5_l1934_193460

def g (x : ℕ) : ℕ := 4 * x - 5
def f (x : ℕ) : ℕ := 6 * x + 11

theorem value_of_f_g_5 : f (g 5) = 101 := by
  sorry

end value_of_f_g_5_l1934_193460


namespace cost_price_of_watch_l1934_193495

variable (CP : ℝ)
variable (SP_loss SP_gain : ℝ)
variable (h1 : SP_loss = CP * 0.725)
variable (h2 : SP_gain = CP * 1.125)
variable (h3 : SP_gain - SP_loss = 275)

theorem cost_price_of_watch : CP = 687.50 :=
by
  sorry

end cost_price_of_watch_l1934_193495


namespace triangle_ABC_BC_length_l1934_193423

theorem triangle_ABC_BC_length 
  (A B C D : ℝ)
  (AB AD DC AC BD BC : ℝ)
  (h1 : BD = 20)
  (h2 : AC = 69)
  (h3 : AB = 29)
  (h4 : BD^2 + DC^2 = BC^2)
  (h5 : AD^2 + BD^2 = AB^2)
  (h6 : AC = AD + DC) : 
  BC = 52 := 
by
  sorry

end triangle_ABC_BC_length_l1934_193423


namespace find_length_of_smaller_rectangle_l1934_193480

theorem find_length_of_smaller_rectangle
  (w : ℝ)
  (h_original : 10 * 15 = 150)
  (h_new_rectangle : 2 * w * w = 150)
  (h_z : w = 5 * Real.sqrt 3) :
  z = 5 * Real.sqrt 3 :=
by
  sorry

end find_length_of_smaller_rectangle_l1934_193480


namespace conic_sections_are_parabolas_l1934_193469

theorem conic_sections_are_parabolas (x y : ℝ) :
  y^6 - 9*x^6 = 3*y^3 - 1 → ∃ k : ℝ, (y^3 - 1 = k * 3 * x^3 ∨ y^3 = -k * 3 * x^3 + 1) := by
  sorry

end conic_sections_are_parabolas_l1934_193469


namespace simplify_expression_l1934_193476

theorem simplify_expression :
  ((3 + 4 + 6 + 7) / 3) + ((4 * 3 + 5 - 2) / 4) = 125 / 12 := by
  sorry

end simplify_expression_l1934_193476


namespace first_discount_percentage_l1934_193466

theorem first_discount_percentage (d : ℝ) (h : d > 0) :
  (∃ x : ℝ, (0 < x) ∧ (x < 100) ∧ 0.6 * d = (d * (1 - x / 100)) * 0.8) → x = 25 :=
by
  sorry

end first_discount_percentage_l1934_193466


namespace determinant_matrix_example_l1934_193488

open Matrix

def matrix_example : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -2], ![-3, 6]]

noncomputable def compute_det_and_add_5 : ℤ := (matrix_example.det) + 5

theorem determinant_matrix_example :
  compute_det_and_add_5 = 41 := by
  sorry

end determinant_matrix_example_l1934_193488


namespace debate_team_group_size_l1934_193487

theorem debate_team_group_size (boys girls groups : ℕ) (h_boys : boys = 11) (h_girls : girls = 45) (h_groups : groups = 8) : 
  (boys + girls) / groups = 7 := by
  sorry

end debate_team_group_size_l1934_193487


namespace find_values_of_cubes_l1934_193482

def N (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, c, b], ![c, b, a], ![b, a, c]]

theorem find_values_of_cubes (a b c : ℂ) (h1 : (N a b c) ^ 2 = 1) (h2 : a * b * c = 1) :
  a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 :=
by
  sorry

end find_values_of_cubes_l1934_193482


namespace mean_age_of_all_children_l1934_193494

def euler_ages : List ℕ := [10, 12, 8]
def gauss_ages : List ℕ := [8, 8, 8, 16, 18]
def all_ages : List ℕ := euler_ages ++ gauss_ages
def total_children : ℕ := all_ages.length
def total_age : ℕ := all_ages.sum
def mean_age : ℕ := total_age / total_children

theorem mean_age_of_all_children : mean_age = 11 := by
  sorry

end mean_age_of_all_children_l1934_193494


namespace horses_put_by_c_l1934_193470

theorem horses_put_by_c (a_horses a_months b_horses b_months c_months total_cost b_cost : ℕ) (x : ℕ) 
  (h1 : a_horses = 12) 
  (h2 : a_months = 8) 
  (h3 : b_horses = 16) 
  (h4 : b_months = 9) 
  (h5 : c_months = 6) 
  (h6 : total_cost = 870) 
  (h7 : b_cost = 360) 
  (h8 : 144 / (96 + 144 + 6 * x) = 360 / 870) : 
  x = 18 := 
by 
  sorry

end horses_put_by_c_l1934_193470


namespace bob_favorite_number_is_correct_l1934_193454

def bob_favorite_number : ℕ :=
  99

theorem bob_favorite_number_is_correct :
  50 < bob_favorite_number ∧
  bob_favorite_number < 100 ∧
  bob_favorite_number % 11 = 0 ∧
  bob_favorite_number % 2 ≠ 0 ∧
  (bob_favorite_number / 10 + bob_favorite_number % 10) % 3 = 0 :=
by
  sorry

end bob_favorite_number_is_correct_l1934_193454


namespace dino_remaining_balance_is_4650_l1934_193444

def gigA_hours : Nat := 20
def gigA_rate : Nat := 10

def gigB_hours : Nat := 30
def gigB_rate : Nat := 20

def gigC_hours : Nat := 5
def gigC_rate : Nat := 40

def gigD_hours : Nat := 15
def gigD_rate : Nat := 25

def gigE_hours : Nat := 10
def gigE_rate : Nat := 30

def january_expense : Nat := 500
def february_expense : Nat := 550
def march_expense : Nat := 520
def april_expense : Nat := 480

theorem dino_remaining_balance_is_4650 :
  let gigA_earnings := gigA_hours * gigA_rate
  let gigB_earnings := gigB_hours * gigB_rate
  let gigC_earnings := gigC_hours * gigC_rate
  let gigD_earnings := gigD_hours * gigD_rate
  let gigE_earnings := gigE_hours * gigE_rate

  let total_monthly_earnings := gigA_earnings + gigB_earnings + gigC_earnings + gigD_earnings + gigE_earnings

  let total_expenses := january_expense + february_expense + march_expense + april_expense

  let total_earnings_four_months := total_monthly_earnings * 4

  total_earnings_four_months - total_expenses = 4650 :=
by {
  sorry
}

end dino_remaining_balance_is_4650_l1934_193444


namespace scientific_notation_l1934_193492

theorem scientific_notation :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end scientific_notation_l1934_193492


namespace divisor_problem_l1934_193421

theorem divisor_problem (n : ℕ) (hn_pos : 0 < n) (h72 : Nat.totient n = 72) (h5n : Nat.totient (5 * n) = 96) : ∃ k : ℕ, (n = 5^k * m ∧ Nat.gcd m 5 = 1) ∧ k = 2 :=
by
  sorry

end divisor_problem_l1934_193421


namespace appeared_candidates_l1934_193473

noncomputable def number_of_candidates_that_appeared_from_each_state (X : ℝ) : Prop :=
  (8 / 100) * X + 220 = (12 / 100) * X

theorem appeared_candidates (X : ℝ) (h : number_of_candidates_that_appeared_from_each_state X) : X = 5500 :=
  sorry

end appeared_candidates_l1934_193473


namespace smallest_sum_is_4_9_l1934_193430

theorem smallest_sum_is_4_9 :
  min
    (min
      (min
        (min (1/3 + 1/4) (1/3 + 1/5))
        (min (1/3 + 1/6) (1/3 + 1/7)))
      (1/3 + 1/9)) = 4/9 :=
  by sorry

end smallest_sum_is_4_9_l1934_193430


namespace Hari_investment_contribution_l1934_193498

noncomputable def Praveen_investment : ℕ := 3780
noncomputable def Praveen_time : ℕ := 12
noncomputable def Hari_time : ℕ := 7
noncomputable def profit_ratio : ℚ := 2 / 3

theorem Hari_investment_contribution :
  ∃ H : ℕ, (Praveen_investment * Praveen_time) / (H * Hari_time) = (2 : ℕ) / 3 ∧ H = 9720 :=
by
  sorry

end Hari_investment_contribution_l1934_193498


namespace lines_perpendicular_l1934_193445

structure Vec3 :=
(x : ℝ) 
(y : ℝ) 
(z : ℝ)

def line1_dir (x : ℝ) : Vec3 := ⟨x, -1, 2⟩
def line2_dir : Vec3 := ⟨2, 1, 4⟩

def dot_product (v1 v2 : Vec3) : ℝ :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

theorem lines_perpendicular (x : ℝ) :
  dot_product (line1_dir x) line2_dir = 0 ↔ x = -7 / 2 :=
by sorry

end lines_perpendicular_l1934_193445


namespace fraction_sent_for_production_twice_l1934_193462

variable {x : ℝ} (hx : x > 0)

theorem fraction_sent_for_production_twice :
  let initial_sulfur := (1.5 / 100 : ℝ)
  let first_sulfur_addition := (0.5 / 100 : ℝ)
  let second_sulfur_addition := (2 / 100 : ℝ) 
  (initial_sulfur - initial_sulfur * x + first_sulfur_addition * x -
    ((initial_sulfur - initial_sulfur * x + first_sulfur_addition * x) * x) + 
    second_sulfur_addition * x = initial_sulfur) → x = 1 / 2 :=
sorry

end fraction_sent_for_production_twice_l1934_193462


namespace intersection_of_A_and_B_l1934_193435

def A : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def B : Set ℝ := { y | 2 ≤ y ∧ y ≤ 5 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 2 ≤ x ∧ x < 3 } :=
sorry

end intersection_of_A_and_B_l1934_193435


namespace dhoni_toys_average_cost_l1934_193425

theorem dhoni_toys_average_cost (A : ℝ) (h1 : ∃ x1 x2 x3 x4 x5, (x1 + x2 + x3 + x4 + x5) / 5 = A)
  (h2 : 5 * A = 5 * A)
  (h3 : ∃ x6, x6 = 16)
  (h4 : (5 * A + 16) / 6 = 11) : A = 10 :=
by
  sorry

end dhoni_toys_average_cost_l1934_193425


namespace total_value_of_gold_l1934_193485

theorem total_value_of_gold (legacy_bars : ℕ) (aleena_bars : ℕ) (bar_value : ℕ) (total_gold_value : ℕ) 
  (h1 : legacy_bars = 12) 
  (h2 : aleena_bars = legacy_bars - 4)
  (h3 : bar_value = 3500) : 
  total_gold_value = (legacy_bars + aleena_bars) * bar_value := 
by 
  sorry

end total_value_of_gold_l1934_193485


namespace regression_line_passes_through_sample_mean_point_l1934_193478

theorem regression_line_passes_through_sample_mean_point
  (a b : ℝ) (x y : ℝ)
  (hx : x = a + b*x) :
  y = a + b*x :=
by sorry

end regression_line_passes_through_sample_mean_point_l1934_193478


namespace total_animals_in_farm_l1934_193406

theorem total_animals_in_farm (C B : ℕ) (h1 : C = 5) (h2 : 2 * C + 4 * B = 26) : C + B = 9 :=
by
  sorry

end total_animals_in_farm_l1934_193406


namespace smallest_digit_divisibility_l1934_193428

theorem smallest_digit_divisibility : 
  ∃ d : ℕ, (d < 10) ∧ (∃ k1 k2 : ℤ, 5 + 2 + 8 + d + 7 + 4 = 9 * k1 ∧ 5 + 2 + 8 + d + 7 + 4 = 3 * k2) ∧ (∀ d' : ℕ, (d' < 10) ∧ 
  (∃ k1 k2 : ℤ, 5 + 2 + 8 + d' + 7 + 4 = 9 * k1 ∧ 5 + 2 + 8 + d' + 7 + 4 = 3 * k2) → d ≤ d') :=
by
  sorry

end smallest_digit_divisibility_l1934_193428


namespace problem_solution_l1934_193464

noncomputable def corrected_angles 
  (x1_star x2_star x3_star : ℝ) 
  (σ : ℝ) 
  (h_sum : x1_star + x2_star + x3_star - 180.0 = 0)  
  (h_var : σ^2 = (0.1)^2) : ℝ × ℝ × ℝ :=
  let Δ := 2.0 / 3.0 * 0.667
  let Δx1 := Δ * (σ^2 / 2)
  let Δx2 := Δ * (σ^2 / 2)
  let Δx3 := Δ * (σ^2 / 2)
  let corrected_x1 := x1_star - Δx1
  let corrected_x2 := x2_star - Δx2
  let corrected_x3 := x3_star - Δx3
  (corrected_x1, corrected_x2, corrected_x3)

theorem problem_solution :
  corrected_angles 31 62 89 (0.1) sorry sorry = (30.0 + 40 / 60, 61.0 + 40 / 60, 88 + 20 / 60) := 
  sorry

end problem_solution_l1934_193464


namespace class_total_students_l1934_193472

def initial_boys : ℕ := 15
def initial_girls : ℕ := (120 * initial_boys) / 100 -- 1.2 * initial_boys

def final_boys : ℕ := initial_boys
def final_girls : ℕ := 2 * initial_girls

def total_students : ℕ := final_boys + final_girls

theorem class_total_students : total_students = 51 := 
by 
  -- the actual proof will go here
  sorry

end class_total_students_l1934_193472


namespace abc_div_def_eq_1_div_20_l1934_193413

-- Definitions
variables (a b c d e f : ℝ)

-- Conditions
axiom condition1 : a / b = 1 / 3
axiom condition2 : b / c = 2
axiom condition3 : c / d = 1 / 2
axiom condition4 : d / e = 3
axiom condition5 : e / f = 1 / 10

-- Proof statement
theorem abc_div_def_eq_1_div_20 : (a * b * c) / (d * e * f) = 1 / 20 :=
by 
  -- The actual proof is omitted, as the problem only requires the statement.
  sorry

end abc_div_def_eq_1_div_20_l1934_193413


namespace fraction_never_reducible_by_11_l1934_193484

theorem fraction_never_reducible_by_11 :
  ∀ (n : ℕ), Nat.gcd (1 + n) (3 + 7 * n) ≠ 11 := by
  sorry

end fraction_never_reducible_by_11_l1934_193484


namespace combined_weight_is_150_l1934_193414

-- Definitions based on conditions
def tracy_weight : ℕ := 52
def jake_weight : ℕ := tracy_weight + 8
def weight_range : ℕ := 14
def john_weight : ℕ := tracy_weight - 14

-- Proving the combined weight
theorem combined_weight_is_150 :
  tracy_weight + jake_weight + john_weight = 150 := by
  sorry

end combined_weight_is_150_l1934_193414


namespace top_card_is_king_l1934_193452

noncomputable def num_cards := 52
noncomputable def num_kings := 4
noncomputable def probability_king := num_kings / num_cards

theorem top_card_is_king :
  probability_king = 1 / 13 := by
  sorry

end top_card_is_king_l1934_193452


namespace integer_a_values_l1934_193415

theorem integer_a_values (a : ℤ) :
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x - 7 = 0) ↔ a = -70 ∨ a = -29 ∨ a = -5 ∨ a = 3 :=
by
  sorry

end integer_a_values_l1934_193415


namespace arithmetic_evaluation_l1934_193401

theorem arithmetic_evaluation : 8 + 18 / 3 - 4 * 2 = 6 := 
by
  sorry

end arithmetic_evaluation_l1934_193401


namespace system_of_equations_solution_l1934_193405

theorem system_of_equations_solution :
  ∃ x y : ℚ, (4 * x - 3 * y = -8) ∧ (5 * x + 9 * y = -18) ∧ x = -14 / 3 ∧ y = -32 / 9 :=
by {
  sorry  -- Proof goes here
}

end system_of_equations_solution_l1934_193405


namespace new_interest_rate_l1934_193412

theorem new_interest_rate 
    (i₁ : ℝ) (r₁ : ℝ) (p : ℝ) (additional_interest : ℝ) (i₂ : ℝ) (r₂ : ℝ)
    (h1 : r₁ = 0.05)
    (h2 : i₁ = 101.20)
    (h3 : additional_interest = 20.24)
    (h4 : i₂ = i₁ + additional_interest)
    (h5 : p = i₁ / (r₁ * 1))
    (h6 : i₂ = p * r₂ * 1) :
  r₂ = 0.06 :=
by
  sorry

end new_interest_rate_l1934_193412


namespace james_initial_amount_l1934_193477

noncomputable def initial_amount (total_amount_invested_per_week: ℕ) 
                                (number_of_weeks_in_year: ℕ) 
                                (windfall_factor: ℚ) 
                                (amount_after_windfall: ℕ) : ℚ :=
  let total_investment := total_amount_invested_per_week * number_of_weeks_in_year
  let amount_without_windfall := (amount_after_windfall : ℚ) / (1 + windfall_factor)
  amount_without_windfall - total_investment

theorem james_initial_amount:
  initial_amount 2000 52 0.5 885000 = 250000 := sorry

end james_initial_amount_l1934_193477


namespace pirates_coins_l1934_193440

noncomputable def coins (x : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0     => x
  | k + 1 => (coins x k) - (coins x k * (k + 2) / 15)

theorem pirates_coins (x : ℕ) (H : x = 2^15 * 3^8 * 5^14) :
  ∃ n : ℕ, n = coins x 14 :=
sorry

end pirates_coins_l1934_193440


namespace number_of_solutions_l1934_193474

theorem number_of_solutions (f : ℕ → ℕ) (n : ℕ) : 
  (∀ n, f n = n^4 + 2 * n^3 - 20 * n^2 + 2 * n - 21) →
  (∀ n, 0 ≤ n ∧ n < 2013 → 2013 ∣ f n) → 
  ∃ k, k = 6 :=
by
  sorry

end number_of_solutions_l1934_193474


namespace power_six_sum_l1934_193486

theorem power_six_sum (x : ℝ) (h : x + 1 / x = 3) : x^6 + 1 / x^6 = 322 := 
by 
  sorry

end power_six_sum_l1934_193486


namespace verify_system_of_equations_l1934_193417

/-- Define a structure to hold the conditions of the problem -/
structure TreePurchasing :=
  (cost_A : ℕ)
  (cost_B : ℕ)
  (diff_A_B : ℕ)
  (total_cost : ℕ)
  (x : ℕ)
  (y : ℕ)

/-- Given conditions for purchasing trees -/
def example_problem : TreePurchasing :=
  { cost_A := 100,
    cost_B := 80,
    diff_A_B := 8,
    total_cost := 8000,
    x := 0,
    y := 0 }

/-- The theorem to prove that the equations match given conditions -/
theorem verify_system_of_equations (data : TreePurchasing) (h_diff : data.x - data.y = data.diff_A_B) (h_cost : data.cost_A * data.x + data.cost_B * data.y = data.total_cost) : 
  (data.x - data.y = 8) ∧ (100 * data.x + 80 * data.y = 8000) :=
  by
    sorry

end verify_system_of_equations_l1934_193417


namespace inequality_inequality_l1934_193431

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end inequality_inequality_l1934_193431


namespace expand_and_simplify_l1934_193468

theorem expand_and_simplify (x y : ℝ) : 
  (x + 6) * (x + 8 + y) = x^2 + 14 * x + x * y + 48 + 6 * y :=
by sorry

end expand_and_simplify_l1934_193468


namespace alyssa_puppies_l1934_193407

theorem alyssa_puppies (initial now given : ℕ) (h1 : initial = 12) (h2 : now = 5) : given = 7 :=
by
  have h3 : given = initial - now := by sorry
  rw [h1, h2] at h3
  exact h3

end alyssa_puppies_l1934_193407


namespace find_x_when_y_is_minus_21_l1934_193457

variable (x y k : ℝ)

theorem find_x_when_y_is_minus_21
  (h1 : x * y = k)
  (h2 : x + y = 35)
  (h3 : y = 3 * x)
  (h4 : y = -21) :
  x = -10.9375 := by
  sorry

end find_x_when_y_is_minus_21_l1934_193457


namespace solve_star_eq_five_l1934_193499

def star (a b : ℝ) : ℝ := a + b^2

theorem solve_star_eq_five :
  ∃ x₁ x₂ : ℝ, star x₁ (x₁ + 1) = 5 ∧ star x₂ (x₂ + 1) = 5 ∧ x₁ = 1 ∧ x₂ = -4 :=
by
  sorry

end solve_star_eq_five_l1934_193499


namespace marbles_solid_color_non_yellow_l1934_193429

theorem marbles_solid_color_non_yellow (total_marble solid_colored solid_yellow : ℝ)
    (h1: solid_colored = 0.90 * total_marble)
    (h2: solid_yellow = 0.05 * total_marble) :
    (solid_colored - solid_yellow) / total_marble = 0.85 := by
  -- sorry is used to skip the proof
  sorry

end marbles_solid_color_non_yellow_l1934_193429


namespace students_like_basketball_or_cricket_or_both_l1934_193455

theorem students_like_basketball_or_cricket_or_both :
  let basketball_lovers := 9
  let cricket_lovers := 8
  let both_lovers := 6
  basketball_lovers + cricket_lovers - both_lovers = 11 :=
by
  sorry

end students_like_basketball_or_cricket_or_both_l1934_193455


namespace mean_of_two_fractions_l1934_193446

theorem mean_of_two_fractions :
  ( (2 : ℚ) / 3 + (4 : ℚ) / 9 ) / 2 = 5 / 9 :=
by
  sorry

end mean_of_two_fractions_l1934_193446


namespace positive_sequence_unique_l1934_193419

theorem positive_sequence_unique (x : Fin 2021 → ℝ) (h : ∀ i : Fin 2020, x i.succ = (x i ^ 3 + 2) / (3 * x i ^ 2)) (h' : x 2020 = x 0) : ∀ i, x i = 1 := by
  sorry

end positive_sequence_unique_l1934_193419


namespace solve_system_l1934_193402

theorem solve_system 
  (x y z : ℝ)
  (h1 : x + 2 * y = 10)
  (h2 : y = 3)
  (h3 : x - 3 * y + z = 7) :
  x = 4 ∧ y = 3 ∧ z = 12 :=
by
  sorry

end solve_system_l1934_193402


namespace hawks_points_l1934_193403

theorem hawks_points (x y z : ℤ) 
  (h_total_points: x + y = 82)
  (h_margin: x - y = 18)
  (h_eagles_points: x = 12 + z) : 
  y = 32 := 
sorry

end hawks_points_l1934_193403


namespace minimum_value_of_expression_l1934_193458

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) :
  2 * x + 1 / x^6 ≥ 3 :=
sorry

end minimum_value_of_expression_l1934_193458


namespace pencil_case_cost_l1934_193443

-- Defining given conditions
def initial_amount : ℕ := 10
def toy_truck_cost : ℕ := 3
def remaining_amount : ℕ := 5
def total_spent : ℕ := initial_amount - remaining_amount

-- Proof statement
theorem pencil_case_cost : total_spent - toy_truck_cost = 2 :=
by
  sorry

end pencil_case_cost_l1934_193443


namespace triangle_angle_properties_l1934_193437

theorem triangle_angle_properties
  (a b : ℕ)
  (h₁ : a = 45)
  (h₂ : b = 70) :
  ∃ (c : ℕ), a + b + c = 180 ∧ c = 65 ∧ max (max a b) c = 70 := by
  sorry

end triangle_angle_properties_l1934_193437


namespace floor_of_smallest_zero_l1934_193481
noncomputable def g (x : ℝ) := 3 * Real.sin x - Real.cos x + 2 * Real.tan x
def smallest_zero (s : ℝ) : Prop := s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem floor_of_smallest_zero (s : ℝ) (h : smallest_zero s) : ⌊s⌋ = 4 :=
sorry

end floor_of_smallest_zero_l1934_193481


namespace vinegar_solution_concentration_l1934_193490

theorem vinegar_solution_concentration
  (original_volume : ℝ) (water_volume : ℝ)
  (original_concentration : ℝ)
  (h1 : original_volume = 12)
  (h2 : water_volume = 50)
  (h3 : original_concentration = 36.166666666666664) :
  original_concentration / 100 * original_volume / (original_volume + water_volume) = 0.07 :=
by
  sorry

end vinegar_solution_concentration_l1934_193490


namespace proportional_function_l1934_193438

theorem proportional_function (k m : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = k * x) →
  f 2 = -4 →
  (∀ x, f x + m = -2 * x + m) →
  f 2 = -4 ∧ (f 1 + m = 1) →
  k = -2 ∧ m = 3 := 
by
  intros h1 h2 h3 h4
  sorry

end proportional_function_l1934_193438


namespace points_enclosed_in_circle_l1934_193404

open Set

variable (points : Set (ℝ × ℝ))
variable (radius : ℝ)
variable (h1 : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points → 
  ∃ (c : ℝ × ℝ), dist c A ≤ radius ∧ dist c B ≤ radius ∧ dist c C ≤ radius)

theorem points_enclosed_in_circle
  (h1 : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points →
    ∃ (c : ℝ × ℝ), dist c A ≤ 1 ∧ dist c B ≤ 1 ∧ dist c C ≤ 1) :
  ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ points → dist c p ≤ 1 :=
sorry

end points_enclosed_in_circle_l1934_193404


namespace Xiaobing_jumps_189_ropes_per_minute_l1934_193465

-- Define conditions and variables
variable (x : ℕ) -- The number of ropes Xiaohan jumps per minute

-- Conditions:
-- 1. Xiaobing jumps x + 21 ropes per minute
-- 2. Time taken for Xiaobing to jump 135 ropes is the same as the time taken for Xiaohan to jump 120 ropes

theorem Xiaobing_jumps_189_ropes_per_minute (h : 135 * x = 120 * (x + 21)) :
    x + 21 = 189 :=
by
  sorry -- Proof is not required as per instructions

end Xiaobing_jumps_189_ropes_per_minute_l1934_193465


namespace prove_length_square_qp_l1934_193461

noncomputable def length_square_qp (r1 r2 d : ℝ) (x : ℝ) : Prop :=
  r1 = 10 ∧ r2 = 8 ∧ d = 15 ∧ (2*r1*x - (x^2 + r2^2 - d^2) = 0) → x^2 = 164

theorem prove_length_square_qp : length_square_qp 10 8 15 x :=
sorry

end prove_length_square_qp_l1934_193461


namespace tangent_line_circle_l1934_193496

open Real

theorem tangent_line_circle (m n : ℝ) :
  (∀ x y : ℝ, ((m + 1) * x + (n + 1) * y - 2 = 0) ↔ (x - 1)^2 + (y - 1)^2 = 1) →
  ((m + n) ≤ 2 - 2 * sqrt 2) ∨ (2 + 2 * sqrt 2 ≤ (m + n)) := by
  sorry

end tangent_line_circle_l1934_193496


namespace line_common_chord_eq_l1934_193432

theorem line_common_chord_eq (a b : ℝ) :
  (∀ (x1 x2 y1 y2 : ℝ), x1^2 + y1^2 = 1 → (x2 - a)^2 + (y2 - b)^2 = 1 → 
    2 * a * x2 + 2 * b * y2 - 3 = 0) :=
sorry

end line_common_chord_eq_l1934_193432


namespace largest_q_value_l1934_193427

theorem largest_q_value : ∃ q, q >= 1 ∧ q^4 - q^3 - q - 1 ≤ 0 ∧ (∀ r, r >= 1 ∧ r^4 - r^3 - r - 1 ≤ 0 → r ≤ q) ∧ q = (Real.sqrt 5 + 1) / 2 := 
sorry

end largest_q_value_l1934_193427


namespace sequence_induction_l1934_193409

theorem sequence_induction (a b : ℕ → ℕ)
  (h₁ : a 1 = 2)
  (h₂ : b 1 = 4)
  (h₃ : ∀ n : ℕ, 0 < n → 2 * b n = a n + a (n + 1))
  (h₄ : ∀ n : ℕ, 0 < n → (a (n + 1))^2 = b n * b (n + 1)) :
  (∀ n : ℕ, 0 < n → a n = n * (n + 1)) ∧ (∀ n : ℕ, 0 < n → b n = (n + 1)^2) :=
by
  sorry

end sequence_induction_l1934_193409


namespace option_D_forms_triangle_l1934_193463

theorem option_D_forms_triangle (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 9) : 
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end option_D_forms_triangle_l1934_193463


namespace problem_correctness_l1934_193449

theorem problem_correctness (a b x y m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : x * y = 1) 
  (h3 : |m| = 2) : 
  (m = 2 ∨ m = -2) ∧ (m^2 + (a + b) / 2 + (- (x * y)) ^ 2023 = 3) := 
by
  sorry

end problem_correctness_l1934_193449


namespace problem_stated_l1934_193497

-- Definitions of constants based on conditions
def a : ℕ := 5
def b : ℕ := 4
def c : ℕ := 3
def d : ℕ := 400
def x : ℕ := 401

-- Mathematical theorem stating the question == answer given conditions
theorem problem_stated : a * x + b * x + c * x + d = 5212 := 
by 
  sorry

end problem_stated_l1934_193497


namespace find_f2_l1934_193424

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f2 (h : f a b (-2) = 3) : f a b 2 = -1 :=
by
  sorry

end find_f2_l1934_193424


namespace kirill_height_l1934_193439

theorem kirill_height (K B : ℕ) (h1 : K = B - 14) (h2 : K + B = 112) : K = 49 :=
by
  sorry

end kirill_height_l1934_193439


namespace garden_width_l1934_193456

theorem garden_width (w : ℝ) (h : ℝ) 
  (h1 : w * h ≥ 150)
  (h2 : h = w + 20)
  (h3 : 2 * (w + h) ≤ 70) :
  w = -10 + 5 * Real.sqrt 10 :=
by sorry

end garden_width_l1934_193456


namespace contrapositive_inequality_l1934_193442

theorem contrapositive_inequality {x y : ℝ} (h : x^2 ≤ y^2) : x ≤ y :=
  sorry

end contrapositive_inequality_l1934_193442


namespace max_ages_acceptable_within_one_std_dev_l1934_193489

theorem max_ages_acceptable_within_one_std_dev
  (average_age : ℤ)
  (std_deviation : ℤ)
  (acceptable_range_lower : ℤ)
  (acceptable_range_upper : ℤ)
  (h1 : average_age = 31)
  (h2 : std_deviation = 5)
  (h3 : acceptable_range_lower = average_age - std_deviation)
  (h4 : acceptable_range_upper = average_age + std_deviation) :
  ∃ n : ℕ, n = acceptable_range_upper - acceptable_range_lower + 1 ∧ n = 11 :=
by
  sorry

end max_ages_acceptable_within_one_std_dev_l1934_193489


namespace mole_can_sustain_l1934_193436

noncomputable def mole_winter_sustainability : Prop :=
  ∃ (grain millet : ℕ), 
    grain = 8 ∧ 
    millet = 0 ∧ 
    ∀ (month : ℕ), 1 ≤ month ∧ month ≤ 3 → 
      ((grain ≥ 3 ∧ (grain - 3) + millet <= 12) ∨ 
      (grain ≥ 1 ∧ millet ≥ 3 ∧ (grain - 1) + (millet - 3) <= 12)) ∧
      ((∃ grain_exchanged millet_gained : ℕ, 
         grain_exchanged ≤ grain ∧
         millet_gained = 2 * grain_exchanged ∧
         grain - grain_exchanged + millet_gained <= 12 ∧
         grain = grain - grain_exchanged) → 
      (grain = 0 ∧ millet = 0))

theorem mole_can_sustain : mole_winter_sustainability := 
sorry 

end mole_can_sustain_l1934_193436


namespace find_percentage_l1934_193448

theorem find_percentage (P : ℝ) : 
  (∀ x : ℝ, x = 0.40 * 800 → x = P / 100 * 650 + 190) → P = 20 := 
by
  intro h
  sorry

end find_percentage_l1934_193448


namespace point_in_fourth_quadrant_l1934_193441

def is_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  is_fourth_quadrant 2 (-3) :=
by
  sorry

end point_in_fourth_quadrant_l1934_193441


namespace projectile_reaches_49_first_time_at_1_point_4_l1934_193411

-- Define the equation for the height of the projectile
def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t

-- State the theorem to prove
theorem projectile_reaches_49_first_time_at_1_point_4 :
  ∃ t : ℝ, height t = 49 ∧ (∀ t' : ℝ, height t' = 49 → t ≤ t') :=
sorry

end projectile_reaches_49_first_time_at_1_point_4_l1934_193411


namespace son_age_l1934_193467

theorem son_age {x : ℕ} {father son : ℕ} 
  (h1 : father = 4 * son)
  (h2 : (son - 10) + (father - 10) = 60)
  (h3 : son = x)
  : x = 16 := 
sorry

end son_age_l1934_193467


namespace nina_homework_total_l1934_193422

def ruby_math_homework : ℕ := 6

def ruby_reading_homework : ℕ := 2

def nina_math_homework : ℕ := ruby_math_homework * 4 + ruby_math_homework

def nina_reading_homework : ℕ := ruby_reading_homework * 8 + ruby_reading_homework

def nina_total_homework : ℕ := nina_math_homework + nina_reading_homework

theorem nina_homework_total :
  nina_total_homework = 48 :=
by
  unfold nina_total_homework
  unfold nina_math_homework
  unfold nina_reading_homework
  unfold ruby_math_homework
  unfold ruby_reading_homework
  sorry

end nina_homework_total_l1934_193422


namespace find_f_minus_1_l1934_193459

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_at_2 : f 2 = 4

theorem find_f_minus_1 : f (-1) = -2 := 
by 
  sorry

end find_f_minus_1_l1934_193459


namespace compute_binomial_sum_l1934_193410

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem compute_binomial_sum :
  binomial 12 11 + binomial 12 1 = 24 :=
by
  sorry

end compute_binomial_sum_l1934_193410


namespace quotient_is_8_l1934_193433

def dividend : ℕ := 64
def divisor : ℕ := 8
def quotient := dividend / divisor

theorem quotient_is_8 : quotient = 8 := 
by 
  show quotient = 8 
  sorry

end quotient_is_8_l1934_193433


namespace speed_of_man_in_still_water_correct_l1934_193453

def upstream_speed : ℝ := 25 -- Upstream speed in kmph
def downstream_speed : ℝ := 39 -- Downstream speed in kmph
def speed_in_still_water : ℝ := 32 -- The speed of the man in still water

theorem speed_of_man_in_still_water_correct :
  (upstream_speed + downstream_speed) / 2 = speed_in_still_water :=
by
  sorry

end speed_of_man_in_still_water_correct_l1934_193453


namespace land_area_decreases_l1934_193426

theorem land_area_decreases (a : ℕ) (h : a > 4) : (a * a) > ((a + 4) * (a - 4)) :=
by
  sorry

end land_area_decreases_l1934_193426


namespace find_min_value_l1934_193420

noncomputable def problem (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧
  (27^x + y^4 - 3^x - 1 = 0)

theorem find_min_value :
  ∃ x y : ℝ, problem x y ∧ 
  (∀ (x' y' : ℝ), problem x' y' → (x^3 + y^3) ≤ (x'^3 + y'^3)) ∧ (x^3 + y^3 = -1) := 
sorry

end find_min_value_l1934_193420


namespace regression_analysis_notes_l1934_193493

-- Define the conditions
def applicable_population (reg_eq: Type) (sample: Type) : Prop := sorry
def temporality (reg_eq: Type) : Prop := sorry
def sample_value_range_influence (reg_eq: Type) (sample: Type) : Prop := sorry
def prediction_precision (reg_eq: Type) : Prop := sorry

-- Define the key points to note
def key_points_to_note (reg_eq: Type) (sample: Type) : Prop :=
  applicable_population reg_eq sample ∧
  temporality reg_eq ∧
  sample_value_range_influence reg_eq sample ∧
  prediction_precision reg_eq

-- The main statement
theorem regression_analysis_notes (reg_eq: Type) (sample: Type) :
  key_points_to_note reg_eq sample := sorry

end regression_analysis_notes_l1934_193493


namespace problem1_problem2_l1934_193471

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : cos B - 2 * cos A = (2 * a - b) * cos C / c)
variable (h2 : a = 2 * b)

theorem problem1 : a / b = 2 :=
by sorry

theorem problem2 (h3 : A > π / 2) (h4 : c = 3) : 0 < b ∧ b < 3 :=
by sorry

end problem1_problem2_l1934_193471


namespace students_per_group_l1934_193479

theorem students_per_group (n m : ℕ) (h_n : n = 36) (h_m : m = 9) : 
  (n - m) / 3 = 9 := 
by
  sorry

end students_per_group_l1934_193479


namespace larger_cuboid_length_is_16_l1934_193450

def volume (l w h : ℝ) : ℝ := l * w * h

def cuboid_length_proof : Prop :=
  ∀ (length_large : ℝ), 
  (volume 5 4 3 * 32 = volume length_large 10 12) → 
  length_large = 16

theorem larger_cuboid_length_is_16 : cuboid_length_proof :=
by
  intros length_large eq_volume
  sorry

end larger_cuboid_length_is_16_l1934_193450


namespace students_per_configuration_l1934_193451

theorem students_per_configuration (students_per_column : ℕ → ℕ) :
  students_per_column 1 = 15 ∧
  students_per_column 2 = 1 ∧
  students_per_column 3 = 1 ∧
  students_per_column 4 = 6 ∧
  ∀ i j, (i ≠ j ∧ i ≤ 12 ∧ j ≤ 12) → students_per_column i ≠ students_per_column j →
  (∃ n, 13 ≤ n ∧ ∀ k, k < 13 → students_per_column k * n = 60) :=
by
  sorry

end students_per_configuration_l1934_193451


namespace product_evaluation_l1934_193400

theorem product_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) = 5^32 - 4^32 :=
by 
sorry

end product_evaluation_l1934_193400


namespace relationship_among_log_exp_powers_l1934_193447

theorem relationship_among_log_exp_powers :
  let a := Real.log 0.3 / Real.log 2
  let b := Real.exp (0.3 * Real.log 2)
  let c := Real.exp (0.2 * Real.log 0.3)
  a < c ∧ c < b :=
by
  sorry

end relationship_among_log_exp_powers_l1934_193447
