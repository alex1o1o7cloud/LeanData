import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l61_6120

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def pow_log2 (x : ℝ) : ℝ := x ^ log2 x

theorem problem_statement (a b c : ℝ)
  (h0 : 1 ≤ a)
  (h1 : 1 ≤ b)
  (h2 : 1 ≤ c)
  (h3 : a * b * c = 10)
  (h4 : pow_log2 a * pow_log2 b * pow_log2 c ≥ 10) :
  a + b + c = 12 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l61_6120


namespace NUMINAMATH_GPT_solve_inequality_l61_6134

theorem solve_inequality :
  {x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4} = {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3)} :=
sorry

end NUMINAMATH_GPT_solve_inequality_l61_6134


namespace NUMINAMATH_GPT_product_of_b_l61_6170

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

noncomputable def g_inv (b : ℝ) (y : ℝ) : ℝ := (y + 4) / 3

theorem product_of_b (b : ℝ) :
  g b 3 = g_inv b (b + 2) → b = 3 := 
by
  sorry

end NUMINAMATH_GPT_product_of_b_l61_6170


namespace NUMINAMATH_GPT_orchestra_member_count_l61_6124

theorem orchestra_member_count :
  ∃ x : ℕ, 150 ≤ x ∧ x ≤ 250 ∧ 
           x % 4 = 2 ∧
           x % 5 = 3 ∧
           x % 8 = 4 ∧
           x % 9 = 5 :=
sorry

end NUMINAMATH_GPT_orchestra_member_count_l61_6124


namespace NUMINAMATH_GPT_functional_equation_true_l61_6190

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) : f x > 0
axiom f_property (a b : ℝ) : f a * f b = f (a + b)

theorem functional_equation_true :
  (f 0 = 1) ∧ 
  (∀ a, f (-a) = 1 / f a) ∧ 
  (∀ a, f a = (f (4 * a)) ^ (1 / 4)) ∧ 
  (∀ a, f (a^2) = (f a)^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_functional_equation_true_l61_6190


namespace NUMINAMATH_GPT_root_exponent_equiv_l61_6181

theorem root_exponent_equiv :
  (7 ^ (1 / 2)) / (7 ^ (1 / 4)) = 7 ^ (1 / 4) := by
  sorry

end NUMINAMATH_GPT_root_exponent_equiv_l61_6181


namespace NUMINAMATH_GPT_initial_tomatoes_l61_6192

theorem initial_tomatoes (T : ℕ) (picked : ℕ) (remaining_total : ℕ) (potatoes : ℕ) :
  potatoes = 12 →
  picked = 53 →
  remaining_total = 136 →
  T + picked = remaining_total - potatoes →
  T = 71 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_tomatoes_l61_6192


namespace NUMINAMATH_GPT_right_triangle_example_find_inverse_450_mod_3599_l61_6169

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def multiplicative_inverse (a b m : ℕ) : Prop :=
  (a * b) % m = 1

theorem right_triangle_example : is_right_triangle 60 221 229 :=
by
  sorry

theorem find_inverse_450_mod_3599 : ∃ n, 0 ≤ n ∧ n < 3599 ∧ multiplicative_inverse 450 n 3599 :=
by
  use 8
  sorry

end NUMINAMATH_GPT_right_triangle_example_find_inverse_450_mod_3599_l61_6169


namespace NUMINAMATH_GPT_ral_current_age_l61_6141

variable (ral suri : ℕ)

-- Conditions
axiom age_relation : ral = 3 * suri
axiom suri_future_age : suri + 3 = 16

-- Statement
theorem ral_current_age : ral = 39 := by
  sorry

end NUMINAMATH_GPT_ral_current_age_l61_6141


namespace NUMINAMATH_GPT_geometric_sequence_terms_sum_l61_6173

theorem geometric_sequence_terms_sum :
  ∀ (a_n : ℕ → ℝ) (q : ℝ),
    (∀ n, a_n (n + 1) = a_n n * q) ∧ a_n 1 = 3 ∧
    (a_n 1 + a_n 2 + a_n 3) = 21 →
    (a_n (1 + 2) + a_n (1 + 3) + a_n (1 + 4)) = 84 :=
by
  intros a_n q h
  sorry

end NUMINAMATH_GPT_geometric_sequence_terms_sum_l61_6173


namespace NUMINAMATH_GPT_Will_old_cards_l61_6160

theorem Will_old_cards (new_cards pages cards_per_page : ℕ) (h1 : new_cards = 8) (h2 : pages = 6) (h3 : cards_per_page = 3) :
  (pages * cards_per_page) - new_cards = 10 :=
by
  sorry

end NUMINAMATH_GPT_Will_old_cards_l61_6160


namespace NUMINAMATH_GPT_total_value_of_item_l61_6123

theorem total_value_of_item (V : ℝ) (h1 : 0.07 * (V - 1000) = 87.50) :
  V = 2250 :=
by
  sorry

end NUMINAMATH_GPT_total_value_of_item_l61_6123


namespace NUMINAMATH_GPT_star_example_l61_6165

def star (a b : ℤ) : ℤ := a * b^3 - 2 * b + 2

theorem star_example : star 2 3 = 50 := by
  sorry

end NUMINAMATH_GPT_star_example_l61_6165


namespace NUMINAMATH_GPT_dhoni_initial_toys_l61_6199

theorem dhoni_initial_toys (x : ℕ) (T : ℕ) 
    (h1 : T = 10 * x) 
    (h2 : T + 16 = 66) : x = 5 := by
  sorry

end NUMINAMATH_GPT_dhoni_initial_toys_l61_6199


namespace NUMINAMATH_GPT_ramu_profit_percent_l61_6189

-- Definitions of the given conditions
def usd_to_inr (usd : ℤ) : ℤ := usd * 45 / 10
def eur_to_inr (eur : ℤ) : ℤ := eur * 567 / 100
def jpy_to_inr (jpy : ℤ) : ℤ := jpy * 1667 / 10000

def cost_of_car_in_inr := usd_to_inr 10000
def engine_repair_cost_in_inr := eur_to_inr 3000
def bodywork_repair_cost_in_inr := jpy_to_inr 150000
def total_cost_in_inr := cost_of_car_in_inr + engine_repair_cost_in_inr + bodywork_repair_cost_in_inr

def selling_price_in_inr : ℤ := 80000
def profit_or_loss_in_inr : ℤ := selling_price_in_inr - total_cost_in_inr

-- Profit percent calculation
def profit_percent (profit_or_loss total_cost : ℤ) : ℚ := (profit_or_loss : ℚ) / (total_cost : ℚ) * 100

-- The theorem stating the mathematically equivalent problem
theorem ramu_profit_percent :
  profit_percent profit_or_loss_in_inr total_cost_in_inr = -8.06 := by
  sorry

end NUMINAMATH_GPT_ramu_profit_percent_l61_6189


namespace NUMINAMATH_GPT_smallest_n_for_modulo_eq_l61_6174

theorem smallest_n_for_modulo_eq :
  ∃ (n : ℕ), (3^n % 4 = n^3 % 4) ∧ (∀ m : ℕ, m < n → 3^m % 4 ≠ m^3 % 4) ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_modulo_eq_l61_6174


namespace NUMINAMATH_GPT_range_of_m_l61_6182

def A (x : ℝ) : Prop := x^2 - x - 6 > 0
def B (x m : ℝ) : Prop := (x - m) * (x - 2 * m) ≤ 0
def is_disjoint (A B : ℝ → Prop) : Prop := ∀ x, ¬ (A x ∧ B x)

theorem range_of_m (m : ℝ) : 
  is_disjoint (A) (B m) ↔ -1 ≤ m ∧ m ≤ 3 / 2 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l61_6182


namespace NUMINAMATH_GPT_width_of_canal_at_bottom_l61_6198

theorem width_of_canal_at_bottom (h : Real) (b : Real) : 
  (A = 1/2 * (top_width + b) * d) ∧ 
  (A = 840) ∧ 
  (top_width = 12) ∧ 
  (d = 84) 
  → b = 8 := 
by
  intros
  sorry

end NUMINAMATH_GPT_width_of_canal_at_bottom_l61_6198


namespace NUMINAMATH_GPT_value_of_m_l61_6131

theorem value_of_m 
    (x : ℝ) (m : ℝ) 
    (h : 0 < x)
    (h_eq : (2 / (x - 2)) - ((2 * x - m) / (2 - x)) = 3) : 
    m = 6 := 
sorry

end NUMINAMATH_GPT_value_of_m_l61_6131


namespace NUMINAMATH_GPT_driver_speed_ratio_l61_6183

theorem driver_speed_ratio (V1 V2 x : ℝ) (h : V1 > 0 ∧ V2 > 0 ∧ x > 0)
  (meet_halfway : ∀ t1 t2, t1 = x / (2 * V1) ∧ t2 = x / (2 * V2))
  (earlier_start : ∀ t1 t2, t1 = t2 + x / (2 * (V1 + V2))) :
  V2 / V1 = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_GPT_driver_speed_ratio_l61_6183


namespace NUMINAMATH_GPT_quadratic_solutions_l61_6138

theorem quadratic_solutions :
  ∀ x : ℝ, (x^2 - 4 * x = 0) → (x = 0 ∨ x = 4) :=
by sorry

end NUMINAMATH_GPT_quadratic_solutions_l61_6138


namespace NUMINAMATH_GPT_polynomial_evaluation_l61_6144

noncomputable def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem polynomial_evaluation : f 2 = 123 := by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l61_6144


namespace NUMINAMATH_GPT_geometric_sequence_product_l61_6109

theorem geometric_sequence_product {a : ℕ → ℝ} 
(h₁ : a 1 = 2) 
(h₂ : a 5 = 8) 
(h_geom : ∀ n, a (n+1) / a n = a (n+2) / a (n+1)) :
a 2 * a 3 * a 4 = 64 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l61_6109


namespace NUMINAMATH_GPT_ratio_first_term_to_common_difference_l61_6101

theorem ratio_first_term_to_common_difference
  (a d : ℝ)
  (S_n : ℕ → ℝ)
  (hS_n : ∀ n, S_n n = (n / 2) * (2 * a + (n - 1) * d))
  (h : S_n 15 = 3 * S_n 10) :
  a / d = -2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_first_term_to_common_difference_l61_6101


namespace NUMINAMATH_GPT_complex_addition_l61_6140

namespace ComplexProof

def B := (3 : ℂ) + (2 * Complex.I)
def Q := (-5 : ℂ)
def R := (2 * Complex.I)
def T := (3 : ℂ) + (5 * Complex.I)

theorem complex_addition :
  B - Q + R + T = (1 : ℂ) + (9 * Complex.I) := 
by
  sorry

end ComplexProof

end NUMINAMATH_GPT_complex_addition_l61_6140


namespace NUMINAMATH_GPT_roadsters_paving_company_total_cement_l61_6194

noncomputable def cement_lexi : ℝ := 10
noncomputable def cement_tess : ℝ := cement_lexi + 0.20 * cement_lexi
noncomputable def cement_ben : ℝ := cement_tess - 0.10 * cement_tess
noncomputable def cement_olivia : ℝ := 2 * cement_ben

theorem roadsters_paving_company_total_cement :
  cement_lexi + cement_tess + cement_ben + cement_olivia = 54.4 := by
  sorry

end NUMINAMATH_GPT_roadsters_paving_company_total_cement_l61_6194


namespace NUMINAMATH_GPT_solve_for_x_l61_6147

theorem solve_for_x (x : ℚ) : ((1/3 - x) ^ 2 = 4) → (x = -5/3 ∨ x = 7/3) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l61_6147


namespace NUMINAMATH_GPT_num_girls_l61_6102

theorem num_girls (boys girls : ℕ) (h1 : girls = boys + 228) (h2 : boys = 469) : girls = 697 :=
sorry

end NUMINAMATH_GPT_num_girls_l61_6102


namespace NUMINAMATH_GPT_simplify_expression_l61_6122

theorem simplify_expression :
  6^6 + 6^6 + 6^6 + 6^6 + 6^6 + 6^6 = 6^7 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l61_6122


namespace NUMINAMATH_GPT_two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one_l61_6171

theorem two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one :
  (2.85 = 2850 * 0.001) := by
  sorry

end NUMINAMATH_GPT_two_point_eight_five_contains_two_thousand_eight_hundred_fifty_of_point_zero_zero_one_l61_6171


namespace NUMINAMATH_GPT_graph_shift_correct_l61_6139

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x) - Real.sqrt 3 * Real.cos (3 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (3 * x)

theorem graph_shift_correct :
  ∀ (x : ℝ), f x = g (x - (5 * Real.pi / 18)) :=
sorry

end NUMINAMATH_GPT_graph_shift_correct_l61_6139


namespace NUMINAMATH_GPT_fraction_power_l61_6157

theorem fraction_power : (2 / 5 : ℚ) ^ 3 = 8 / 125 := by
  sorry

end NUMINAMATH_GPT_fraction_power_l61_6157


namespace NUMINAMATH_GPT_multiplication_distributive_example_l61_6118

theorem multiplication_distributive_example : 23 * 4 = 20 * 4 + 3 * 4 := by
  sorry

end NUMINAMATH_GPT_multiplication_distributive_example_l61_6118


namespace NUMINAMATH_GPT_order_of_even_function_l61_6164

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem order_of_even_function {f : ℝ → ℝ}
  (h_even : is_even f)
  (h_mono_inc : is_monotonically_increasing_on_nonneg f) :
  f (-π) > f (3) ∧ f (3) > f (-2) :=
sorry

end NUMINAMATH_GPT_order_of_even_function_l61_6164


namespace NUMINAMATH_GPT_parallel_lines_slope_l61_6100

theorem parallel_lines_slope (d : ℝ) (h : 3 = 4 * d) : d = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l61_6100


namespace NUMINAMATH_GPT_find_a_l61_6119

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x ^ 2 + a * Real.cos (Real.pi * x) else 2

theorem find_a (a : ℝ) :
  (∀ x, f (-x) a = -f x a) → f 1 a = 2 → a = - 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l61_6119


namespace NUMINAMATH_GPT_find_factorial_number_l61_6193

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_factorial_number (n : ℕ) : Prop :=
  ∃ x y z : ℕ, (0 ≤ x ∧ x ≤ 5) ∧
               (0 ≤ y ∧ y ≤ 5) ∧
               (0 ≤ z ∧ z ≤ 5) ∧
               n = 100 * x + 10 * y + z ∧
               n = x.factorial + y.factorial + z.factorial

theorem find_factorial_number : ∃ n, is_three_digit_number n ∧ is_factorial_number n ∧ n = 145 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_factorial_number_l61_6193


namespace NUMINAMATH_GPT_handshake_count_l61_6188

def total_employees : ℕ := 50
def dept_X : ℕ := 30
def dept_Y : ℕ := 20
def handshakes_between_departments : ℕ := dept_X * dept_Y

theorem handshake_count : handshakes_between_departments = 600 :=
by
  sorry

end NUMINAMATH_GPT_handshake_count_l61_6188


namespace NUMINAMATH_GPT_loot_box_cost_l61_6179

variable (C : ℝ) -- Declare cost of each loot box as a real number

-- Conditions (average value of items, money spent, loss)
def avg_value : ℝ := 3.5
def money_spent : ℝ := 40
def avg_loss : ℝ := 12

-- Derived equation
def equation := avg_value * (money_spent / C) = money_spent - avg_loss

-- Statement to prove
theorem loot_box_cost : equation C → C = 5 := by
  sorry

end NUMINAMATH_GPT_loot_box_cost_l61_6179


namespace NUMINAMATH_GPT_volume_tetrahedron_l61_6104

variables (AB AC AD : ℝ) (β γ D : ℝ)
open Real

/-- Prove that the volume of tetrahedron ABCD is equal to 
    (AB * AC * AD * sin β * sin γ * sin D) / 6,
    where β and γ are the plane angles at vertex A opposite to edges AB and AC, 
    and D is the dihedral angle at edge AD. 
-/
theorem volume_tetrahedron (h₁: β ≠ 0) (h₂: γ ≠ 0) (h₃: D ≠ 0):
  (AB * AC * AD * sin β * sin γ * sin D) / 6 =
    abs (AB * AC * AD * sin β * sin γ * sin D) / 6 :=
by sorry

end NUMINAMATH_GPT_volume_tetrahedron_l61_6104


namespace NUMINAMATH_GPT_max_value_of_expression_l61_6127

noncomputable def maximum_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (h : x^2 - x*y + 2*y^2 = 8) : ℝ :=
  x^2 + x*y + 2*y^2

theorem max_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x^2 - x*y + 2*y^2 = 8) : maximum_value hx hy h = (72 + 32 * Real.sqrt 2) / 7 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l61_6127


namespace NUMINAMATH_GPT_total_flowers_l61_6196

noncomputable def yellow_flowers : ℕ := 10
noncomputable def purple_flowers : ℕ := yellow_flowers + (80 * yellow_flowers) / 100
noncomputable def green_flowers : ℕ := (25 * (yellow_flowers + purple_flowers)) / 100
noncomputable def red_flowers : ℕ := (35 * (yellow_flowers + purple_flowers + green_flowers)) / 100

theorem total_flowers :
  yellow_flowers + purple_flowers + green_flowers + red_flowers = 47 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_total_flowers_l61_6196


namespace NUMINAMATH_GPT_units_digit_47_4_plus_28_4_l61_6176

theorem units_digit_47_4_plus_28_4 (units_digit_47 : Nat := 7) (units_digit_28 : Nat := 8) :
  (47^4 + 28^4) % 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_47_4_plus_28_4_l61_6176


namespace NUMINAMATH_GPT_traffic_light_probability_change_l61_6153

theorem traffic_light_probability_change :
  let cycle_time := 100
  let intervals := [(0, 50), (50, 55), (55, 100)]
  let time_changing := [((45, 50), 5), ((50, 55), 5), ((95, 100), 5)]
  let total_change_time := time_changing.map Prod.snd |>.sum
  let probability := (total_change_time : ℚ) / cycle_time
  probability = 3 / 20 := sorry

end NUMINAMATH_GPT_traffic_light_probability_change_l61_6153


namespace NUMINAMATH_GPT_triangle_to_pentagon_ratio_l61_6186

theorem triangle_to_pentagon_ratio (t p : ℕ) 
  (h1 : 3 * t = 15) 
  (h2 : 5 * p = 15) : (t : ℚ) / (p : ℚ) = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_to_pentagon_ratio_l61_6186


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l61_6152

theorem sum_of_roots_of_quadratic (a b : ℝ) (h : (a - 3)^2 = 16) (h' : (b - 3)^2 = 16) (a_neq_b : a ≠ b) : a + b = 6 := 
sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l61_6152


namespace NUMINAMATH_GPT_children_left_on_bus_l61_6149

-- Definitions based on the conditions
def initial_children := 43
def children_got_off := 22

-- The theorem we want to prove
theorem children_left_on_bus (initial_children children_got_off : ℕ) : 
  initial_children - children_got_off = 21 :=
by
  sorry

end NUMINAMATH_GPT_children_left_on_bus_l61_6149


namespace NUMINAMATH_GPT_charlie_extra_fee_l61_6106

-- Conditions
def data_limit_week1 : ℕ := 2 -- in GB
def data_limit_week2 : ℕ := 3 -- in GB
def data_limit_week3 : ℕ := 2 -- in GB
def data_limit_week4 : ℕ := 1 -- in GB

def additional_fee_week1 : ℕ := 12 -- dollars per GB
def additional_fee_week2 : ℕ := 10 -- dollars per GB
def additional_fee_week3 : ℕ := 8 -- dollars per GB
def additional_fee_week4 : ℕ := 6 -- dollars per GB

def data_used_week1 : ℕ := 25 -- in 0.1 GB
def data_used_week2 : ℕ := 40 -- in 0.1 GB
def data_used_week3 : ℕ := 30 -- in 0.1 GB
def data_used_week4 : ℕ := 50 -- in 0.1 GB

-- Additional fee calculation
def extra_data_fee := 
  let extra_data_week1 := max (data_used_week1 - data_limit_week1 * 10) 0
  let extra_fee_week1 := extra_data_week1 * additional_fee_week1 / 10
  let extra_data_week2 := max (data_used_week2 - data_limit_week2 * 10) 0
  let extra_fee_week2 := extra_data_week2 * additional_fee_week2 / 10
  let extra_data_week3 := max (data_used_week3 - data_limit_week3 * 10) 0
  let extra_fee_week3 := extra_data_week3 * additional_fee_week3 / 10
  let extra_data_week4 := max (data_used_week4 - data_limit_week4 * 10) 0
  let extra_fee_week4 := extra_data_week4 * additional_fee_week4 / 10
  extra_fee_week1 + extra_fee_week2 + extra_fee_week3 + extra_fee_week4

-- The math proof problem
theorem charlie_extra_fee : extra_data_fee = 48 := sorry

end NUMINAMATH_GPT_charlie_extra_fee_l61_6106


namespace NUMINAMATH_GPT_no_nonzero_solution_l61_6108

theorem no_nonzero_solution (a b c n : ℤ) 
  (h : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := 
by 
  sorry

end NUMINAMATH_GPT_no_nonzero_solution_l61_6108


namespace NUMINAMATH_GPT_ratio_of_black_to_blue_l61_6150

universe u

-- Define the types of black and red pens
variables (B R : ℕ)

-- Define the conditions
def condition1 : Prop := 2 + B + R = 12
def condition2 : Prop := R = 2 * B - 2

-- Define the proof statement
theorem ratio_of_black_to_blue (h1 : condition1 B R) (h2 : condition2 B R) : B / 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_black_to_blue_l61_6150


namespace NUMINAMATH_GPT_natalie_bushes_needed_l61_6185

theorem natalie_bushes_needed (b c p : ℕ) 
  (h1 : ∀ b, b * 10 = c) 
  (h2 : ∀ c, c * 2 = p)
  (target_p : p = 36) :
  ∃ b, b * 10 ≥ 72 :=
by
  sorry

end NUMINAMATH_GPT_natalie_bushes_needed_l61_6185


namespace NUMINAMATH_GPT_rectangle_area_l61_6175

variable (a b : ℝ)

-- Given conditions
axiom h1 : (a + b)^2 = 16 
axiom h2 : (a - b)^2 = 4

-- Objective: Prove that the area of the rectangle ab equals 3
theorem rectangle_area : a * b = 3 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l61_6175


namespace NUMINAMATH_GPT_gcd_m_n_eq_one_l61_6136

/-- Mathematical definitions of m and n. --/
def m : ℕ := 123^2 + 235^2 + 347^2
def n : ℕ := 122^2 + 234^2 + 348^2

/-- Listing the conditions and deriving the result that gcd(m, n) = 1. --/
theorem gcd_m_n_eq_one : gcd m n = 1 :=
by sorry

end NUMINAMATH_GPT_gcd_m_n_eq_one_l61_6136


namespace NUMINAMATH_GPT_rational_powers_implies_rational_a_rational_powers_implies_rational_b_l61_6162

open Real

theorem rational_powers_implies_rational_a (x : ℝ) :
  (∃ r₁ r₂ : ℚ, x^7 = r₁ ∧ x^12 = r₂) → (∃ q : ℚ, x = q) :=
by
  sorry

theorem rational_powers_implies_rational_b (x : ℝ) :
  (∃ r₁ r₂ : ℚ, x^9 = r₁ ∧ x^12 = r₂) → (∃ q : ℚ, x = q) :=
by
  sorry

end NUMINAMATH_GPT_rational_powers_implies_rational_a_rational_powers_implies_rational_b_l61_6162


namespace NUMINAMATH_GPT_christian_age_in_eight_years_l61_6187

-- Definitions from the conditions
def christian_current_age : ℕ := 72
def brian_age_in_eight_years : ℕ := 40

-- Theorem to prove
theorem christian_age_in_eight_years : ∃ (age : ℕ), age = christian_current_age + 8 ∧ age = 80 := by
  sorry

end NUMINAMATH_GPT_christian_age_in_eight_years_l61_6187


namespace NUMINAMATH_GPT_larger_number_is_25_l61_6137

-- Let x and y be real numbers, with x being the larger number
variables (x y : ℝ)

-- The sum of the two numbers is 45
axiom sum_eq_45 : x + y = 45

-- The difference of the two numbers is 5
axiom diff_eq_5 : x - y = 5

-- We need to prove that the larger number x is 25
theorem larger_number_is_25 : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_25_l61_6137


namespace NUMINAMATH_GPT_find_m_range_of_x_l61_6132

def f (m x : ℝ) : ℝ := (m^2 - 1) * x + m^2 - 3 * m + 2

theorem find_m (m : ℝ) (H_dec : m^2 - 1 < 0) (H_f1 : f m 1 = 0) : 
  m = 1 / 2 :=
sorry

theorem range_of_x (x : ℝ) :
  f (1 / 2) (x + 1) ≥ x^2 ↔ -3 / 4 ≤ x ∧ x ≤ 0 :=
sorry

end NUMINAMATH_GPT_find_m_range_of_x_l61_6132


namespace NUMINAMATH_GPT_harmonic_mean_of_4_and_5040_is_8_closest_l61_6103

noncomputable def harmonicMean (a b : ℕ) : ℝ :=
  (2 * a * b) / (a + b)

theorem harmonic_mean_of_4_and_5040_is_8_closest :
  abs (harmonicMean 4 5040 - 8) < 1 :=
by
  -- The proof process would go here
  sorry

end NUMINAMATH_GPT_harmonic_mean_of_4_and_5040_is_8_closest_l61_6103


namespace NUMINAMATH_GPT_min_f_value_l61_6197

open Real

theorem min_f_value (a b c d e : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) :
    ∃ (x : ℝ), (∀ y : ℝ, (|y - a| + |y - b| + |y - c| + |y - d| + |y - e|) ≥ -a - b + d + e) ∧ 
    (|x - a| + |x - b| + |x - c| + |x - d| + |x - e| = -a - b + d + e) :=
sorry

end NUMINAMATH_GPT_min_f_value_l61_6197


namespace NUMINAMATH_GPT_middle_card_is_five_l61_6142

section card_numbers

variables {a b c : ℕ}

-- Conditions
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def sum_fifteen (a b c : ℕ) : Prop := a + b + c = 15
def sum_two_smallest_less_than_ten (a b : ℕ) : Prop := a + b < 10
def ascending_order (a b c : ℕ) : Prop := a < b ∧ b < c 

-- Main theorem statement
theorem middle_card_is_five 
  (h1 : distinct a b c)
  (h2 : sum_fifteen a b c)
  (h3 : sum_two_smallest_less_than_ten a b) 
  (h4 : ascending_order a b c)
  (h5 : ∀ x, (x = a → (∃ b c, sum_fifteen x b c ∧ distinct x b c ∧ sum_two_smallest_less_than_ten x b ∧ ascending_order x b c ∧ ¬ (b = 5 ∧ c = 10))) →
           (∃ b c, sum_fifteen x b c ∧ distinct x b c ∧ sum_two_smallest_less_than_ten b c ∧ ascending_order x b c ∧ ¬ (b = 2 ∧ c = 7)))
  (h6 : ∀ x, (x = c → (∃ a b, sum_fifteen a b x ∧ distinct a b x ∧ sum_two_smallest_less_than_ten a b ∧ ascending_order a b x ∧ ¬ (a = 1 ∧ b = 4))) →
           (∃ a b, sum_fifteen a b x ∧ distinct a b x ∧ sum_two_smallest_less_than_ten a b ∧ ascending_order a b x ∧ ¬ (a = 2 ∧ b = 6)))
  (h7 : ∀ x, (x = b → (∃ a c, sum_fifteen a x c ∧ distinct a x c ∧ sum_two_smallest_less_than_ten a c ∧ ascending_order a x c ∧ ¬ (a = 1 ∧ c = 9 ∨ a = 2 ∧ c = 8))) →
           (∃ a c, sum_fifteen a x c ∧ distinct a x c ∧ sum_two_smallest_less_than_ten a c ∧ ascending_order a x c ∧ ¬ (a = 1 ∧ c = 6 ∨ a = 2 ∧ c = 5)))
  : b = 5 := sorry

end card_numbers

end NUMINAMATH_GPT_middle_card_is_five_l61_6142


namespace NUMINAMATH_GPT_bike_cost_l61_6195

theorem bike_cost (price_per_apple repairs_share remaining_share apples_sold earnings repairs_cost bike_cost : ℝ) :
  price_per_apple = 1.25 →
  repairs_share = 0.25 →
  remaining_share = 1/5 →
  apples_sold = 20 →
  earnings = apples_sold * price_per_apple →
  repairs_cost = earnings * 4/5 →
  repairs_cost = bike_cost * repairs_share →
  bike_cost = 80 :=
by
  intros;
  sorry

end NUMINAMATH_GPT_bike_cost_l61_6195


namespace NUMINAMATH_GPT_veronica_cans_of_food_is_multiple_of_4_l61_6145

-- Definitions of the given conditions
def number_of_water_bottles : ℕ := 20
def number_of_kits : ℕ := 4

-- Proof statement
theorem veronica_cans_of_food_is_multiple_of_4 (F : ℕ) :
  F % number_of_kits = 0 :=
sorry

end NUMINAMATH_GPT_veronica_cans_of_food_is_multiple_of_4_l61_6145


namespace NUMINAMATH_GPT_price_after_two_reductions_l61_6155

variable (orig_price : ℝ) (m : ℝ)

def current_price (orig_price : ℝ) (m : ℝ) : ℝ :=
  orig_price * (1 - m) * (1 - m)

theorem price_after_two_reductions (h1 : orig_price = 100) (h2 : 0 ≤ m ∧ m ≤ 1) :
  current_price orig_price m = 100 * (1 - m) ^ 2 := by
    sorry

end NUMINAMATH_GPT_price_after_two_reductions_l61_6155


namespace NUMINAMATH_GPT_min_blocks_for_wall_l61_6121

noncomputable def min_blocks_needed (length height : ℕ) (block_sizes : List (ℕ × ℕ)) : ℕ :=
  sorry

theorem min_blocks_for_wall :
  min_blocks_needed 120 8 [(1, 3), (1, 2), (1, 1)] = 404 := by
  sorry

end NUMINAMATH_GPT_min_blocks_for_wall_l61_6121


namespace NUMINAMATH_GPT_rancher_steers_cows_solution_l61_6113

theorem rancher_steers_cows_solution :
  ∃ (s c : ℕ), s > 0 ∧ c > 0 ∧ (30 * s + 31 * c = 1200) ∧ (s = 9) ∧ (c = 30) :=
by
  sorry

end NUMINAMATH_GPT_rancher_steers_cows_solution_l61_6113


namespace NUMINAMATH_GPT_heating_time_correct_l61_6159

def initial_temp : ℤ := 20

def desired_temp : ℤ := 100

def heating_rate : ℤ := 5

def time_to_heat (initial desired rate : ℤ) : ℤ :=
  (desired - initial) / rate

theorem heating_time_correct :
  time_to_heat initial_temp desired_temp heating_rate = 16 :=
by
  sorry

end NUMINAMATH_GPT_heating_time_correct_l61_6159


namespace NUMINAMATH_GPT_problem1_problem2_l61_6167

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)

theorem problem1 (m : ℝ) (h₀ : m > 3) (h₁ : ∃ m, (1/2) * (((m - 1) / 2) - (-(m + 1) / 2) + 3) * (m - 3) = 7 / 2) : m = 4 := by
  sorry

theorem problem2 (a : ℝ) (h₂ : ∃ x, (0 ≤ x ∧ x ≤ 2) ∧ f x ≥ abs (a - 3)) : -2 ≤ a ∧ a ≤ 8 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l61_6167


namespace NUMINAMATH_GPT_square_b_perimeter_l61_6110

theorem square_b_perimeter (a b : ℝ) 
  (ha : a^2 = 65) 
  (prob : (65 - b^2) / 65 = 0.7538461538461538) : 
  4 * b = 16 :=
by 
  sorry

end NUMINAMATH_GPT_square_b_perimeter_l61_6110


namespace NUMINAMATH_GPT_positive_integer_pairs_count_l61_6156

theorem positive_integer_pairs_count :
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs → a > 0 ∧ b > 0 ∧ (1 : ℚ) / a - (1 : ℚ) / b = (1 : ℚ) / 2021) ∧ 
    pairs.length = 4 :=
by sorry

end NUMINAMATH_GPT_positive_integer_pairs_count_l61_6156


namespace NUMINAMATH_GPT_petya_time_comparison_l61_6129

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end NUMINAMATH_GPT_petya_time_comparison_l61_6129


namespace NUMINAMATH_GPT_contradiction_proof_l61_6133

theorem contradiction_proof (x y : ℝ) (h1 : x + y ≤ 0) (h2 : x > 0) (h3 : y > 0) : false :=
by
  sorry

end NUMINAMATH_GPT_contradiction_proof_l61_6133


namespace NUMINAMATH_GPT_days_left_in_year_is_100_l61_6161

noncomputable def days_left_in_year 
    (daily_average_rain_before : ℝ) 
    (total_rainfall_so_far : ℝ) 
    (average_rain_needed : ℝ) 
    (total_days_in_year : ℕ) : ℕ :=
    sorry

theorem days_left_in_year_is_100 :
    days_left_in_year 2 430 3 365 = 100 := 
sorry

end NUMINAMATH_GPT_days_left_in_year_is_100_l61_6161


namespace NUMINAMATH_GPT_john_must_deliver_1063_pizzas_l61_6130

-- Declare all the given conditions
def car_cost : ℕ := 8000
def maintenance_cost : ℕ := 500
def pizza_income (p : ℕ) : ℕ := 12 * p
def gas_cost (p : ℕ) : ℕ := 4 * p

-- Define the function that returns the net earnings
def net_earnings (p : ℕ) := pizza_income p - gas_cost p

-- Define the total expenses
def total_expenses : ℕ := car_cost + maintenance_cost

-- Define the minimum number of pizzas John must deliver
def minimum_pizzas (p : ℕ) : Prop := net_earnings p ≥ total_expenses

-- State the theorem that needs to be proved
theorem john_must_deliver_1063_pizzas : minimum_pizzas 1063 := by
  sorry

end NUMINAMATH_GPT_john_must_deliver_1063_pizzas_l61_6130


namespace NUMINAMATH_GPT_largest_partner_share_l61_6151

def total_profit : ℕ := 48000
def partner_ratios : List ℕ := [3, 4, 4, 6, 7]
def value_per_part : ℕ := total_profit / partner_ratios.sum
def largest_share : ℕ := 7 * value_per_part

theorem largest_partner_share :
  largest_share = 14000 := by
  sorry

end NUMINAMATH_GPT_largest_partner_share_l61_6151


namespace NUMINAMATH_GPT_percent_of_workday_in_meetings_l61_6111

theorem percent_of_workday_in_meetings (h1 : 9 > 0) (m1 m2 : ℕ) (h2 : m1 = 45) (h3 : m2 = 2 * m1) : 
  (135 / 540 : ℚ) * 100 = 25 := 
by
  -- Just for structure, the proof should go here
  sorry

end NUMINAMATH_GPT_percent_of_workday_in_meetings_l61_6111


namespace NUMINAMATH_GPT_average_weight_of_11_children_l61_6148

theorem average_weight_of_11_children (b: ℕ) (g: ℕ) (avg_b: ℕ) (avg_g: ℕ) (hb: b = 8) (hg: g = 3) (havg_b: avg_b = 155) (havg_g: avg_g = 115) : 
  (b * avg_b + g * avg_g) / (b + g) = 144 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_weight_of_11_children_l61_6148


namespace NUMINAMATH_GPT_odd_power_preserves_order_l61_6168

theorem odd_power_preserves_order {n : ℤ} (h1 : n > 0) (h2 : n % 2 = 1) :
  ∀ (a b : ℝ), a > b → a^n > b^n :=
by
  sorry

end NUMINAMATH_GPT_odd_power_preserves_order_l61_6168


namespace NUMINAMATH_GPT_population_decrease_is_25_percent_l61_6180

def initial_population : ℕ := 20000
def final_population_first_year : ℕ := initial_population + (initial_population * 25 / 100)
def final_population_second_year : ℕ := 18750

def percentage_decrease (initial final : ℕ) : ℚ :=
  ((initial - final : ℚ) * 100) / initial 

theorem population_decrease_is_25_percent :
  percentage_decrease final_population_first_year final_population_second_year = 25 :=
by
  sorry

end NUMINAMATH_GPT_population_decrease_is_25_percent_l61_6180


namespace NUMINAMATH_GPT_even_function_l61_6135

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ := (x + 2)^2 + (2 * x - 1)^2

theorem even_function : is_even_function f :=
by
  sorry

end NUMINAMATH_GPT_even_function_l61_6135


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_of_f_l61_6114

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_of_f_l61_6114


namespace NUMINAMATH_GPT_yi_successful_shots_l61_6125

-- Defining the basic conditions
variables {x y : ℕ} -- Number of successful shots made by Jia and Yi respectively

-- Each hit gains 20 points and each miss deducts 12 points.
-- Both person A (Jia) and person B (Yi) made 10 shots each.
def total_shots (x y : ℕ) : Prop := 
  (20 * x - 12 * (10 - x)) + (20 * y - 12 * (10 - y)) = 208 ∧ x + y = 14 ∧ x - y = 2

theorem yi_successful_shots (x y : ℕ) (h : total_shots x y) : y = 6 := 
  by sorry

end NUMINAMATH_GPT_yi_successful_shots_l61_6125


namespace NUMINAMATH_GPT_janet_used_clips_correct_l61_6172

-- Define the initial number of paper clips
def initial_clips : ℕ := 85

-- Define the remaining number of paper clips
def remaining_clips : ℕ := 26

-- Define the number of clips Janet used
def used_clips (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

-- The theorem to state the correctness of the calculation
theorem janet_used_clips_correct : used_clips initial_clips remaining_clips = 59 :=
by
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_janet_used_clips_correct_l61_6172


namespace NUMINAMATH_GPT_find_radius_l61_6117

def radius_of_circle (d : ℤ) (PQ : ℕ) (QR : ℕ) (r : ℕ) : Prop := 
  let PR := PQ + QR
  (PQ * PR = (d - r) * (d + r)) ∧ (d = 15) ∧ (PQ = 11) ∧ (QR = 8) ∧ (r = 4)

-- Now stating the theorem to prove the radius r given the conditions
theorem find_radius (r : ℕ) : radius_of_circle 15 11 8 r := by
  sorry

end NUMINAMATH_GPT_find_radius_l61_6117


namespace NUMINAMATH_GPT_dartboard_odd_score_probability_l61_6116

theorem dartboard_odd_score_probability :
  let π := Real.pi
  let r_outer := 4
  let r_inner := 2
  let area_inner := π * r_inner * r_inner
  let area_outer := π * r_outer * r_outer
  let area_annulus := area_outer - area_inner
  let area_inner_region := area_inner / 3
  let area_outer_region := area_annulus / 3
  let odd_inner_regions := 1
  let even_inner_regions := 2
  let odd_outer_regions := 2
  let even_outer_regions := 1
  let prob_odd_inner := (odd_inner_regions * area_inner_region) / (area_inner + area_annulus)
  let prob_even_inner := (even_inner_regions * area_inner_region) / (area_inner + area_annulus)
  let prob_odd_outer := (odd_outer_regions * area_outer_region) / (area_inner + area_annulus)
  let prob_even_outer := (even_outer_regions * area_outer_region) / (area_inner + area_annulus)
  let prob_odd_region := prob_odd_inner + prob_odd_outer
  let prob_even_region := prob_even_inner + prob_even_outer
  let prob_odd_score := (prob_odd_region * prob_even_region) + (prob_even_region * prob_odd_region)
  prob_odd_score = 5 / 9 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_dartboard_odd_score_probability_l61_6116


namespace NUMINAMATH_GPT_largest_divisor_of_n_l61_6105

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n^2 = 18 * k) : ∃ l : ℕ, n = 6 * l :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_n_l61_6105


namespace NUMINAMATH_GPT_lunks_needed_for_20_apples_l61_6177

-- Define the conditions as given in the problem
def lunks_to_kunks (lunks : ℤ) : ℤ := (4 * lunks) / 7
def kunks_to_apples (kunks : ℤ) : ℤ := (5 * kunks) / 3

-- Define the target function to calculate the number of lunks needed for given apples
def apples_to_lunks (apples : ℤ) : ℤ := 
  let kunks := (3 * apples) / 5
  let lunks := (7 * kunks) / 4
  lunks

-- Prove the given problem
theorem lunks_needed_for_20_apples : apples_to_lunks 20 = 21 := by
  sorry

end NUMINAMATH_GPT_lunks_needed_for_20_apples_l61_6177


namespace NUMINAMATH_GPT_find_A_l61_6154

theorem find_A (A7B : ℕ) (H1 : (A7B % 100) / 10 = 7) (H2 : A7B + 23 = 695) : (A7B / 100) = 6 := 
  sorry

end NUMINAMATH_GPT_find_A_l61_6154


namespace NUMINAMATH_GPT_spherical_coordinates_convert_l61_6128

theorem spherical_coordinates_convert (ρ θ φ ρ' θ' φ' : ℝ) 
  (h₀ : ρ > 0) 
  (h₁ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₂ : 0 ≤ φ ∧ φ ≤ Real.pi) 
  (h_initial : (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5)) 
  (h_final : (ρ', θ', φ') = (4, (11 * Real.pi) / 8,  Real.pi / 5)) : 
  (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5) → 
  (ρ, θ, φ) = (ρ', θ', φ') := 
by
  sorry

end NUMINAMATH_GPT_spherical_coordinates_convert_l61_6128


namespace NUMINAMATH_GPT_find_fraction_value_l61_6163

noncomputable section

open Real

theorem find_fraction_value (α : ℝ) (h : sin (α / 2) - 2 * cos (α / 2) = 1) :
  (1 + sin α + cos α) / (1 + sin α - cos α) = 1 :=
sorry

end NUMINAMATH_GPT_find_fraction_value_l61_6163


namespace NUMINAMATH_GPT_part1_part2_part3_l61_6158

-- Part 1
theorem part1 (x : ℝ) :
  (2 * x - 5 > 3 * x - 8 ∧ -4 * x + 3 < x - 4) ↔ x = 2 :=
sorry

-- Part 2
theorem part2 (x : ℤ) :
  (x - 1 / 4 < 1 ∧ 4 + 2 * x > -7 * x + 5) ↔ x = 1 :=
sorry

-- Part 3
theorem part3 (m : ℝ) :
  (∀ x, m < x ∧ x <= m + 2 → (x = 3 ∨ x = 2)) ↔ 1 ≤ m ∧ m < 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l61_6158


namespace NUMINAMATH_GPT_c_payment_l61_6178

theorem c_payment 
  (A_rate : ℝ) (B_rate : ℝ) (days : ℝ) (total_payment : ℝ) (C_fraction : ℝ) 
  (hA : A_rate = 1 / 6) 
  (hB : B_rate = 1 / 8) 
  (hdays : days = 3) 
  (hpayment : total_payment = 3200)
  (hC_fraction : C_fraction = 1 / 8) :
  total_payment * C_fraction = 400 :=
by {
  -- The proof would go here
  sorry
}

end NUMINAMATH_GPT_c_payment_l61_6178


namespace NUMINAMATH_GPT_dot_product_result_parallelism_condition_l61_6126

-- Definitions of the vectors
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 2)

-- 1. Prove the dot product result
theorem dot_product_result :
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  let a_minus_2b := (a.1 - 2 * b.1, a.2 - 2 * b.2)
  a_plus_b.1 * a_minus_2b.1 + a_plus_b.2 * a_minus_2b.2 = -14 :=
by
  sorry

-- 2. Prove parallelism condition
theorem parallelism_condition (k : ℝ) :
  let k_a_plus_b := (k * a.1 + b.1, k * a.2 + b.2)
  let a_minus_3b := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  k = -1/3 → k_a_plus_b.1 * a_minus_3b.2 = k_a_plus_b.2 * a_minus_3b.1 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_result_parallelism_condition_l61_6126


namespace NUMINAMATH_GPT_infinite_series_sum_eq_l61_6184

theorem infinite_series_sum_eq : 
  (∑' n : ℕ, if n = 0 then 0 else ((1 : ℝ) / (n * (n + 3)))) = (11 / 18 : ℝ) :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_eq_l61_6184


namespace NUMINAMATH_GPT_roof_shingle_width_l61_6112

theorem roof_shingle_width (L A W : ℕ) (hL : L = 10) (hA : A = 70) (hArea : A = L * W) : W = 7 :=
by
  sorry

end NUMINAMATH_GPT_roof_shingle_width_l61_6112


namespace NUMINAMATH_GPT_ImpossibleNonConformists_l61_6191

open Int

def BadPairCondition (f : ℤ → ℤ) (n : ℤ) :=
  ∃ (pairs : Finset (ℤ × ℤ)), 
    pairs.card ≤ ⌊0.001 * (n.natAbs^2 : ℝ)⌋₊ ∧ 
    ∀ (x y : ℤ), (x, y) ∈ pairs → max (abs x) (abs y) ≤ n ∧ f (x + y) ≠ f x + f y

def NonConformistCondition (f : ℤ → ℤ) (n : ℤ) :=
  ∃ (conformists : Finset ℤ), 
    conformists.card > n ∧ 
    ∀ (a : ℤ), abs a ≤ n → (f a ≠ a * f 1 → a ∈ conformists)

theorem ImpossibleNonConformists (f : ℤ → ℤ) :
  (∀ (n : ℤ), n ≥ 0 → BadPairCondition f n) → 
  ¬ ∃ (n : ℤ), n ≥ 0 ∧ NonConformistCondition f n :=
  by 
    intros h_cond h_ex
    sorry

end NUMINAMATH_GPT_ImpossibleNonConformists_l61_6191


namespace NUMINAMATH_GPT_fabric_cost_equation_l61_6107

theorem fabric_cost_equation (x : ℝ) :
  (3 * x + 5 * (138 - x) = 540) :=
sorry

end NUMINAMATH_GPT_fabric_cost_equation_l61_6107


namespace NUMINAMATH_GPT_total_time_taken_l61_6166

theorem total_time_taken 
  (R : ℝ) -- Rickey's speed
  (T_R : ℝ := 40) -- Rickey's time
  (T_P : ℝ := (40 * (4 / 3))) -- Prejean's time derived from given conditions
  (P : ℝ := (3 / 4) * R) -- Prejean's speed
  (k : ℝ := 40 * R) -- constant k for distance
 
  (h1 : T_R = 40)
  (h2 : T_P = 40 * (4 / 3))
  -- Main goal: Prove total time taken equals 93.33 minutes
  : (T_R + T_P) = 93.33 := 
  sorry

end NUMINAMATH_GPT_total_time_taken_l61_6166


namespace NUMINAMATH_GPT_sum_equals_120_l61_6143

def rectangular_parallelepiped := (3, 4, 5)

def face_dimensions : List (ℕ × ℕ) := [(4, 5), (3, 5), (3, 4)]

def number_assignment (d : ℕ × ℕ) : ℕ :=
  if d = (4, 5) then 9
  else if d = (3, 5) then 8
  else if d = (3, 4) then 5
  else 0

def sum_checkerboard_ring_one_width (rect_dims : ℕ × ℕ × ℕ) (number_assignment : ℕ × ℕ → ℕ) : ℕ :=
  let (x, y, z) := rect_dims
  let l1 := number_assignment (4, 5) * 2 * (4 * 5)
  let l2 := number_assignment (3, 5) * 2 * (3 * 5)
  let l3 := number_assignment (3, 4) * 2 * (3 * 4) 
  l1 + l2 + l3

theorem sum_equals_120 : ∀ rect_dims number_assignment,
  rect_dims = rectangular_parallelepiped → sum_checkerboard_ring_one_width rect_dims number_assignment = 720 := sorry

end NUMINAMATH_GPT_sum_equals_120_l61_6143


namespace NUMINAMATH_GPT_probability_single_shot_l61_6146

-- Define the event and probability given
def event_A := "shooter hits the target at least once out of three shots"
def probability_event_A : ℝ := 0.875

-- The probability of missing in one shot is q, and missing all three is q^3, 
-- which leads to hitting at least once being 1 - q^3
theorem probability_single_shot (q : ℝ) (h : 1 - q^3 = 0.875) : 1 - q = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_probability_single_shot_l61_6146


namespace NUMINAMATH_GPT_selene_total_payment_l61_6115

def price_instant_camera : ℝ := 110
def num_instant_cameras : ℕ := 2
def discount_instant_camera : ℝ := 0.07
def price_photo_frame : ℝ := 120
def num_photo_frames : ℕ := 3
def discount_photo_frame : ℝ := 0.05
def sales_tax : ℝ := 0.06

theorem selene_total_payment :
  let total_instant_cameras := num_instant_cameras * price_instant_camera
  let discount_instant := total_instant_cameras * discount_instant_camera
  let discounted_instant := total_instant_cameras - discount_instant
  let total_photo_frames := num_photo_frames * price_photo_frame
  let discount_photo := total_photo_frames * discount_photo_frame
  let discounted_photo := total_photo_frames - discount_photo
  let subtotal := discounted_instant + discounted_photo
  let tax := subtotal * sales_tax
  let total_payment := subtotal + tax
  total_payment = 579.40 :=
by
  sorry

end NUMINAMATH_GPT_selene_total_payment_l61_6115
