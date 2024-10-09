import Mathlib

namespace original_cost_prices_l407_40702

variable (COST_A COST_B COST_C : ℝ)

theorem original_cost_prices :
  (COST_A * 0.8 + 100 = COST_A * 1.05) →
  (COST_B * 1.1 - 80 = COST_B * 0.92) →
  (COST_C * 0.85 + 120 = COST_C * 1.07) →
  COST_A = 400 ∧
  COST_B = 4000 / 9 ∧
  COST_C = 6000 / 11 := by
  intro h1 h2 h3
  sorry

end original_cost_prices_l407_40702


namespace eagles_per_section_l407_40780

theorem eagles_per_section (total_eagles sections : ℕ) (h1 : total_eagles = 18) (h2 : sections = 3) :
  total_eagles / sections = 6 := by
  sorry

end eagles_per_section_l407_40780


namespace solve_equation_l407_40741

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 1) = 2 * x - 2 ↔ (x = 1 ∨ x = 2 / 3) := 
by 
  intro x
  sorry

end solve_equation_l407_40741


namespace cos_135_eq_neg_inv_sqrt2_l407_40728

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l407_40728


namespace intersecting_lines_l407_40750

theorem intersecting_lines (c d : ℝ) :
  (∀ x y, (x = (1/3) * y + c) ∧ (y = (1/3) * x + d) → x = 3 ∧ y = 6) →
  c + d = 6 :=
by
  sorry

end intersecting_lines_l407_40750


namespace combined_weight_is_correct_l407_40763

-- Frank and Gwen's candy weights
def frank_candy : ℕ := 10
def gwen_candy : ℕ := 7

-- The combined weight of candy
def combined_weight : ℕ := frank_candy + gwen_candy

-- Theorem that states the combined weight is 17 pounds
theorem combined_weight_is_correct : combined_weight = 17 :=
by
  -- proves that 10 + 7 = 17
  sorry

end combined_weight_is_correct_l407_40763


namespace clean_car_time_l407_40723

theorem clean_car_time (t_outside : ℕ) (t_inside : ℕ) (h_outside : t_outside = 80) (h_inside : t_inside = t_outside / 4) : 
  t_outside + t_inside = 100 := 
by 
  sorry

end clean_car_time_l407_40723


namespace ellipse_ratio_sum_l407_40777

theorem ellipse_ratio_sum :
  (∃ x y : ℝ, 3 * x^2 + 2 * x * y + 4 * y^2 - 20 * x - 30 * y + 60 = 0) →
  (∃ a b : ℝ, (∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 20 * x - 30 * y + 60 = 0 → 
    (y = a * x ∨ y = b * x)) ∧ (a + b = 9)) :=
  sorry

end ellipse_ratio_sum_l407_40777


namespace value_of_x_plus_y_l407_40711

theorem value_of_x_plus_y (x y : ℝ) (h : |x - 1| + (y - 2)^2 = 0) : x + y = 3 := by
  sorry

end value_of_x_plus_y_l407_40711


namespace arccos_cos_8_eq_1_point_72_l407_40756

noncomputable def arccos_cos_eight : Real :=
  Real.arccos (Real.cos 8)

theorem arccos_cos_8_eq_1_point_72 : arccos_cos_eight = 1.72 :=
by
  sorry

end arccos_cos_8_eq_1_point_72_l407_40756


namespace cube_surface_area_l407_40755

theorem cube_surface_area (v : ℝ) (h : v = 1000) : ∃ (s : ℝ), s^3 = v ∧ 6 * s^2 = 600 :=
by
  sorry

end cube_surface_area_l407_40755


namespace remainder_x2023_plus_1_l407_40765

noncomputable def remainder (a b : Polynomial ℂ) : Polynomial ℂ :=
a % b

theorem remainder_x2023_plus_1 :
  remainder (Polynomial.X ^ 2023 + 1) (Polynomial.X ^ 8 - Polynomial.X ^ 6 + Polynomial.X ^ 4 - Polynomial.X ^ 2 + 1) =
  - Polynomial.X ^ 3 + 1 :=
by
  sorry

end remainder_x2023_plus_1_l407_40765


namespace min_sum_of_factors_l407_40794

theorem min_sum_of_factors (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 3960) : 
  a + b + c = 72 :=
sorry

end min_sum_of_factors_l407_40794


namespace christen_peeled_potatoes_l407_40745

open Nat

theorem christen_peeled_potatoes :
  ∀ (total_potatoes homer_rate homer_time christen_rate : ℕ) (combined_rate : ℕ),
    total_potatoes = 60 →
    homer_rate = 4 →
    homer_time = 6 →
    christen_rate = 6 →
    combined_rate = homer_rate + christen_rate →
    Nat.ceil ((total_potatoes - (homer_rate * homer_time)) / combined_rate * christen_rate) = 21 :=
by
  intros total_potatoes homer_rate homer_time christen_rate combined_rate
  intros htp hr ht cr cr_def
  rw [htp, hr, ht, cr, cr_def]
  sorry

end christen_peeled_potatoes_l407_40745


namespace find_m_l407_40739

def a (m : ℝ) : ℝ × ℝ := (2 * m - 1, 3)
def b : ℝ × ℝ := (1, -1)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_m (m : ℝ) (h : dot_product (a m) b = 2) : m = 3 :=
by sorry

end find_m_l407_40739


namespace max_2a_b_2c_l407_40772

theorem max_2a_b_2c (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 2 * a + b + 2 * c ≤ 3 :=
sorry

end max_2a_b_2c_l407_40772


namespace a_share_correct_l407_40752

-- Investment periods for each individual in months
def investment_a := 12
def investment_b := 6
def investment_c := 4
def investment_d := 9
def investment_e := 7
def investment_f := 5

-- Investment multiplier for each individual
def multiplier_b := 2
def multiplier_c := 3
def multiplier_d := 4
def multiplier_e := 5
def multiplier_f := 6

-- Total annual gain
def total_gain := 38400

-- Calculate individual shares
def share_a (x : ℝ) := x * investment_a
def share_b (x : ℝ) := multiplier_b * x * investment_b
def share_c (x : ℝ) := multiplier_c * x * investment_c
def share_d (x : ℝ) := multiplier_d * x * investment_d
def share_e (x : ℝ) := multiplier_e * x * investment_e
def share_f (x : ℝ) := multiplier_f * x * investment_f

-- Calculate total investment
def total_investment (x : ℝ) :=
  share_a x + share_b x + share_c x + share_d x + share_e x + share_f x

-- Prove that a's share of the annual gain is Rs. 3360
theorem a_share_correct : 
  ∃ x : ℝ, (12 * x / total_investment x) * total_gain = 3360 := 
sorry

end a_share_correct_l407_40752


namespace find_d_l407_40701

-- Given conditions
def line_eq (x y : ℚ) : Prop := y = (3 * x - 4) / 4

def parametrized_eq (v d : ℚ × ℚ) (t x y : ℚ) : Prop :=
  (x, y) = (v.1 + t * d.1, v.2 + t * d.2)

def distance_eq (x y : ℚ) (t : ℚ) : Prop :=
  (x - 3) * (x - 3) + (y - 1) * (y - 1) = t * t

-- The proof problem statement
theorem find_d (d : ℚ × ℚ) 
  (h_d : d = (7/2, 5/2)) :
  ∀ (x y t : ℚ) (v : ℚ × ℚ) (h_v : v = (3, 1)),
    (x ≥ 3) → 
    line_eq x y → 
    parametrized_eq v d t x y → 
    distance_eq x y t → 
    d = (7/2, 5/2) := 
by 
  intros;
  sorry


end find_d_l407_40701


namespace lowest_die_exactly_3_prob_l407_40715

noncomputable def fair_die_prob_at_least (n : ℕ) : ℚ :=
  if h : 1 ≤ n ∧ n ≤ 6 then (6 - n + 1) / 6 else 0

noncomputable def prob_lowest_die_exactly_3 : ℚ :=
  let p_at_least_3 := fair_die_prob_at_least 3
  let p_at_least_4 := fair_die_prob_at_least 4
  (p_at_least_3 ^ 4) - (p_at_least_4 ^ 4)

theorem lowest_die_exactly_3_prob :
  prob_lowest_die_exactly_3 = 175 / 1296 := by
  sorry

end lowest_die_exactly_3_prob_l407_40715


namespace exam_rule_l407_40705

variable (P R Q : Prop)

theorem exam_rule (hp : P ∧ R → Q) : ¬ Q → ¬ P ∨ ¬ R :=
by
  sorry

end exam_rule_l407_40705


namespace problem1_no_solution_problem2_solution_l407_40795

theorem problem1_no_solution (x : ℝ) 
  (h : (5*x - 4)/(x - 2) = (4*x + 10)/(3*x - 6) - 1) : false :=
by
  -- The original equation turns out to have no solution
  sorry

theorem problem2_solution (x : ℝ) 
  (h : 1 - (x - 2)/(2 + x) = 16/(x^2 - 4)) : x = 6 :=
by
  -- The equation has a solution x = 6
  sorry

end problem1_no_solution_problem2_solution_l407_40795


namespace gregory_current_age_l407_40732

-- Given conditions
variables (D G y : ℕ)
axiom dm_is_three_times_greg_was (x : ℕ) : D = 3 * y
axiom future_age_sum : D + (3 * y) = 49
axiom greg_age_difference x y : D - (3 * y) = (3 * y) - x

-- Prove statement: Gregory's current age is 14
theorem gregory_current_age : G = 14 := by
  sorry

end gregory_current_age_l407_40732


namespace max_of_inverse_power_sums_l407_40729

theorem max_of_inverse_power_sums (s p r1 r2 : ℝ) 
  (h_eq_roots : r1 + r2 = s ∧ r1 * r2 = p)
  (h_eq_powers : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2023 → r1^n + r2^n = s) :
  1 / r1^(2024:ℕ) + 1 / r2^(2024:ℕ) ≤ 2 :=
sorry

end max_of_inverse_power_sums_l407_40729


namespace cos_45_deg_l407_40717

theorem cos_45_deg : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_45_deg_l407_40717


namespace rectangle_length_width_difference_l407_40742

theorem rectangle_length_width_difference
  (x y : ℝ)
  (h1 : y = 1 / 3 * x)
  (h2 : 2 * x + 2 * y = 32)
  (h3 : Real.sqrt (x^2 + y^2) = 17) :
  abs (x - y) = 8 :=
sorry

end rectangle_length_width_difference_l407_40742


namespace coffee_shop_sales_l407_40716

def number_of_coffee_customers : Nat := 7
def price_per_coffee : Nat := 5

def number_of_tea_customers : Nat := 8
def price_per_tea : Nat := 4

def total_sales : Nat :=
  (number_of_coffee_customers * price_per_coffee)
  + (number_of_tea_customers * price_per_tea)

theorem coffee_shop_sales : total_sales = 67 := by
  sorry

end coffee_shop_sales_l407_40716


namespace min_red_chips_l407_40798

variable (w b r : ℕ)

theorem min_red_chips :
  (b ≥ w / 3) → (b ≤ r / 4) → (w + b ≥ 70) → r ≥ 72 :=
by
  sorry

end min_red_chips_l407_40798


namespace question1_question2_l407_40713

-- Define the sets A and B based on the conditions
def setA (a : ℝ) : Set ℝ := { x | 2 * x + a > 0 }
def setB : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Question 1: When a = 2, find the set A ∩ B
theorem question1 : A ∩ B = { x | x > 3 } :=
  sorry

-- Question 2: If A ∩ (complement of B) = ∅, find the range of a
theorem question2 : A ∩ (U \ B) = ∅ → a ≤ -6 :=
  sorry

end question1_question2_l407_40713


namespace range_of_m_l407_40768

open Set Real

theorem range_of_m (M N : Set ℝ) (m : ℝ) :
    (M = {x | x ≤ m}) →
    (N = {y | ∃ x : ℝ, y = 2^(-x)}) →
    (M ∩ N ≠ ∅) → m > 0 := by
  intros hM hN hMN
  sorry

end range_of_m_l407_40768


namespace boat_travel_times_l407_40738

theorem boat_travel_times (d_AB d_BC : ℕ) 
  (t_against_current t_with_current t_total_A t_total_C : ℕ) 
  (h_AB : d_AB = 3) (h_BC : d_BC = 3) 
  (h_against_current : t_against_current = 10) 
  (h_with_current : t_with_current = 8)
  (h_total_A : t_total_A = 24)
  (h_total_C : t_total_C = 72) :
  (t_total_A = 24 ∨ t_total_A = 72) ∧ (t_total_C = 24 ∨ t_total_C = 72) := 
by 
  sorry

end boat_travel_times_l407_40738


namespace number_of_even_divisors_of_factorial_eight_l407_40721

-- Definition of 8! and its prime factorization
def factorial_eight : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def prime_factorization_factorial_eight : Prop :=
  factorial_eight = 2^7 * 3^2 * 5 * 7

-- The main theorem statement
theorem number_of_even_divisors_of_factorial_eight :
  prime_factorization_factorial_eight →
  ∃ n, n = 7 * 3 * 2 * 2 ∧
  (∀ d, d ∣ factorial_eight → (∃ a b c d, 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧ d = 2^a * 3^b * 5^c * 7^d) →
  (7 * 3 * 2 * 2 = n)) :=
by
  intro h
  use 84
  sorry

end number_of_even_divisors_of_factorial_eight_l407_40721


namespace nicky_speed_l407_40784

theorem nicky_speed
  (head_start : ℕ := 36)
  (cristina_speed : ℕ := 6)
  (time_to_catch_up : ℕ := 12)
  (distance_cristina_runs : ℕ := cristina_speed * time_to_catch_up)
  (distance_nicky_runs : ℕ := distance_cristina_runs - head_start)
  (nicky_speed : ℕ := distance_nicky_runs / time_to_catch_up) :
  nicky_speed = 3 :=
by
  sorry

end nicky_speed_l407_40784


namespace sides_of_triangle_expr_negative_l407_40751

theorem sides_of_triangle_expr_negative (a b c : ℝ) 
(h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
(a - c)^2 - b^2 < 0 :=
sorry

end sides_of_triangle_expr_negative_l407_40751


namespace interval_of_decrease_l407_40734

noncomputable def f (x : ℝ) := x * Real.exp x + 1

theorem interval_of_decrease : {x : ℝ | x < -1} = {x : ℝ | (x + 1) * Real.exp x < 0} :=
by
  sorry

end interval_of_decrease_l407_40734


namespace value_of_a_l407_40776

-- Conditions
def A (a : ℝ) : Set ℝ := {2, a}
def B (a : ℝ) : Set ℝ := {-1, a^2 - 2}

-- Theorem statement asserting the condition and the correct answer
theorem value_of_a (a : ℝ) : (A a ∩ B a).Nonempty → a = -2 :=
by
  sorry

end value_of_a_l407_40776


namespace no_common_complex_roots_l407_40789

theorem no_common_complex_roots (a b : ℚ) :
  ¬ ∃ α : ℂ, (α^5 - α - 1 = 0) ∧ (α^2 + a * α + b = 0) :=
sorry

end no_common_complex_roots_l407_40789


namespace decimal_89_to_binary_l407_40769

def decimal_to_binary (n : ℕ) : ℕ := sorry

theorem decimal_89_to_binary :
  decimal_to_binary 89 = 1011001 :=
sorry

end decimal_89_to_binary_l407_40769


namespace lcm_of_numbers_with_ratio_and_hcf_l407_40753

theorem lcm_of_numbers_with_ratio_and_hcf (a b : ℕ) (h1 : a = 3 * x) (h2 : b = 4 * x) (h3 : Nat.gcd a b = 3) : Nat.lcm a b = 36 := 
  sorry

end lcm_of_numbers_with_ratio_and_hcf_l407_40753


namespace helicopter_A_highest_altitude_helicopter_A_final_altitude_helicopter_B_5th_performance_l407_40778

def heights_A : List ℝ := [3.6, -2.4, 2.8, -1.5, 0.9]
def heights_B : List ℝ := [3.8, -2, 4.1, -2.3]

theorem helicopter_A_highest_altitude :
  List.maximum heights_A = some 3.6 :=
by sorry

theorem helicopter_A_final_altitude :
  List.sum heights_A = 3.4 :=
by sorry

theorem helicopter_B_5th_performance :
  ∃ (x : ℝ), List.sum heights_B + x = 3.4 ∧ x = -0.2 :=
by sorry

end helicopter_A_highest_altitude_helicopter_A_final_altitude_helicopter_B_5th_performance_l407_40778


namespace hexagon_monochromatic_triangle_probability_l407_40719

open Classical

-- Define the total number of edges in the hexagon
def total_edges : ℕ := 15

-- Define the number of triangles from 6 vertices
def total_triangles : ℕ := Nat.choose 6 3

-- Define the probability that a given triangle is not monochromatic
def prob_not_monochromatic_triangle : ℚ := 3 / 4

-- Calculate the probability of having at least one monochromatic triangle
def prob_at_least_one_monochromatic_triangle : ℚ := 
  1 - (prob_not_monochromatic_triangle ^ total_triangles)

theorem hexagon_monochromatic_triangle_probability :
  abs ((prob_at_least_one_monochromatic_triangle : ℝ) - 0.9968) < 0.0001 :=
by
  sorry

end hexagon_monochromatic_triangle_probability_l407_40719


namespace find_slope_l407_40735

theorem find_slope 
  (k : ℝ)
  (y : ℝ -> ℝ)
  (P : ℝ × ℝ)
  (l : ℝ -> ℝ -> Prop)
  (A B F : ℝ × ℝ)
  (C : ℝ × ℝ -> Prop)
  (d : ℝ × ℝ -> ℝ × ℝ -> ℝ)
  (k_pos : P = (3, 0))
  (k_slope : ∀ x, y x = k * (x - 3))
  (k_int_hyperbola_A : C A)
  (k_int_hyperbola_B : C B)
  (k_focus : F = (2, 0))
  (k_sum_dist : d A F + d B F = 16) :
  k = 1 ∨ k = -1 :=
sorry

end find_slope_l407_40735


namespace selling_price_per_pound_l407_40730

-- Definitions based on conditions
def cost_per_pound_type1 : ℝ := 2.00
def cost_per_pound_type2 : ℝ := 3.00
def weight_type1 : ℝ := 64
def weight_type2 : ℝ := 16
def total_weight : ℝ := 80

-- The selling price per pound of the mixture
theorem selling_price_per_pound :
  let total_cost := (weight_type1 * cost_per_pound_type1) + (weight_type2 * cost_per_pound_type2)
  (total_cost / total_weight) = 2.20 :=
by
  sorry

end selling_price_per_pound_l407_40730


namespace ellipse_equation_and_slope_range_l407_40766

theorem ellipse_equation_and_slope_range (a b : ℝ) (e : ℝ) (k : ℝ) :
  a > b ∧ b > 0 ∧ e = (Real.sqrt 3) / 3 ∧
  ∃! ℓ : ℝ × ℝ, (ℓ.2 = 1 ∧ ℓ.1 = -2) ∧
  ∀ x y : ℝ, x^2 + y^2 = b^2 → y = x + 2 →
  ((x - 0)^2 + (y - 0)^2 = b^2) ∧
  (
    (a^2 = (3 * b^2)) ∧ (b = Real.sqrt 2) ∧
    a > 0 ∧
    (∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1) ∧
    (-((Real.sqrt 2) / 2) < k ∧ k < 0) ∨ (0 < k ∧ k < ((Real.sqrt 2) / 2))
  ) :=
by
  sorry

end ellipse_equation_and_slope_range_l407_40766


namespace determine_x_l407_40790

-- Definitions based on conditions
variables {x : ℝ}

-- Problem statement
theorem determine_x (h : (6 * x)^5 = (18 * x)^4) (hx : x ≠ 0) : x = 27 / 2 :=
by
  sorry

end determine_x_l407_40790


namespace congruence_solutions_count_number_of_solutions_l407_40785

theorem congruence_solutions_count (x : ℕ) (hx_pos : x > 0) (hx_lt : x < 200) :
  (x + 17) % 52 = 75 % 52 ↔ x = 6 ∨ x = 58 ∨ x = 110 ∨ x = 162 :=
by sorry

theorem number_of_solutions :
  (∃ x : ℕ, (0 < x ∧ x < 200 ∧ (x + 17) % 52 = 75 % 52)) ∧
  (∃ x1 x2 x3 x4 : ℕ, x1 = 6 ∧ x2 = 58 ∧ x3 = 110 ∧ x4 = 162) ∧
  4 = 4 :=
by sorry

end congruence_solutions_count_number_of_solutions_l407_40785


namespace cyclists_meet_time_l407_40731

theorem cyclists_meet_time 
  (v1 v2 : ℕ) (C : ℕ) (h1 : v1 = 7) (h2 : v2 = 8) (hC : C = 675) : 
  C / (v1 + v2) = 45 :=
by
  sorry

end cyclists_meet_time_l407_40731


namespace prime_sum_exists_even_n_l407_40781

theorem prime_sum_exists_even_n (n : ℕ) :
  (∃ a b c : ℤ, a + b + c = 0 ∧ Prime (a^n + b^n + c^n)) ↔ Even n := 
by
  sorry

end prime_sum_exists_even_n_l407_40781


namespace problem_statement_l407_40706

theorem problem_statement (a b c : ℝ) 
  (h1 : 2011 * (a + b + c) = 1)
  (h2 : a * b + a * c + b * c = 2011 * a * b * c) :
  a ^ 2011 * b ^ 2011 + c ^ 2011 = 1 / 2011^2011 :=
by
  sorry

end problem_statement_l407_40706


namespace sequence_le_zero_l407_40764

noncomputable def sequence_property (N : ℕ) (a : ℕ → ℝ) : Prop :=
  (a 0 = 0) ∧ (a N = 0) ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ N - 1 → a (i + 1) - 2 * a i + a (i - 1) = a i ^ 2)

theorem sequence_le_zero {N : ℕ} (a : ℕ → ℝ) (h : sequence_property N a) : 
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ N - 1 → a i ≤ 0 :=
sorry

end sequence_le_zero_l407_40764


namespace max_ab_perpendicular_l407_40787

theorem max_ab_perpendicular (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 2 * a + b = 3) : ab <= (9 / 8) := 
sorry

end max_ab_perpendicular_l407_40787


namespace range_of_a_l407_40700

variable (a : ℝ) (f : ℝ → ℝ)
axiom func_def : ∀ x, f x = a^x
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1
axiom decreasing : ∀ m n : ℝ, m > n → f m < f n

theorem range_of_a : 0 < a ∧ a < 1 :=
sorry

end range_of_a_l407_40700


namespace simplify_expression_l407_40749

theorem simplify_expression (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) : 
  1 - (1 / (1 + (a^2 / (1 - a^2)))) = a^2 :=
sorry

end simplify_expression_l407_40749


namespace problem1_problem2_l407_40796

/-- Problem 1: Prove the solution to the system of equations is x = 1/2 and y = 5 -/
theorem problem1 (x y : ℚ) (h1 : 2 * x - y = -4) (h2 : 4 * x - 5 * y = -23) : 
  x = 1 / 2 ∧ y = 5 := 
sorry

/-- Problem 2: Prove the value of the expression (x-3y)^{2} - (2x+y)(y-2x) when x = 2 and y = -1 is 40 -/
theorem problem2 (x y : ℚ) (h1 : x = 2) (h2 : y = -1) : 
  (x - 3 * y) ^ 2 - (2 * x + y) * (y - 2 * x) = 40 := 
sorry

end problem1_problem2_l407_40796


namespace percentage_of_earrings_l407_40774

theorem percentage_of_earrings (B M R : ℕ) (hB : B = 10) (hM : M = 2 * R) (hTotal : B + M + R = 70) : 
  (B * 100) / M = 25 := 
by
  sorry

end percentage_of_earrings_l407_40774


namespace min_value_of_a_l407_40724

theorem min_value_of_a (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : a + b + c + d = 2004) (h5 : a^2 - b^2 + c^2 - d^2 = 2004) : a = 503 :=
sorry

end min_value_of_a_l407_40724


namespace subset_A_imp_range_a_disjoint_A_imp_range_a_l407_40712

-- Definition of sets A and B
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a)*(x - 3*a) < 0}

-- Proof problem for Question 1
theorem subset_A_imp_range_a (a : ℝ) (h : A ⊆ B a) : 
  (4 / 3) ≤ a ∧ a ≤ 2 ∧ a ≠ 0 :=
sorry

-- Proof problem for Question 2
theorem disjoint_A_imp_range_a (a : ℝ) (h : A ∩ B a = ∅) : 
  a ≤ (2 / 3) ∨ a ≥ 4 :=
sorry

end subset_A_imp_range_a_disjoint_A_imp_range_a_l407_40712


namespace bob_same_color_probability_is_1_over_28_l407_40758

def num_marriages : ℕ := 9
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 3

def david_marbles : ℕ := 3
def alice_marbles : ℕ := 3
def bob_marbles : ℕ := 3

def total_ways : ℕ := 1680
def favorable_ways : ℕ := 60
def probability_bob_same_color := favorable_ways / total_ways

theorem bob_same_color_probability_is_1_over_28 : probability_bob_same_color = (1 : ℚ) / 28 := by
  sorry

end bob_same_color_probability_is_1_over_28_l407_40758


namespace number_of_students_with_no_pets_l407_40710

-- Define the number of students in the class
def total_students : ℕ := 25

-- Define the number of students with cats
def students_with_cats : ℕ := (3 * total_students) / 5

-- Define the number of students with dogs
def students_with_dogs : ℕ := (20 * total_students) / 100

-- Define the number of students with elephants
def students_with_elephants : ℕ := 3

-- Calculate the number of students with no pets
def students_with_no_pets : ℕ := total_students - (students_with_cats + students_with_dogs + students_with_elephants)

-- Statement to be proved
theorem number_of_students_with_no_pets : students_with_no_pets = 2 :=
sorry

end number_of_students_with_no_pets_l407_40710


namespace number_of_true_propositions_l407_40744

theorem number_of_true_propositions :
  let P1 := false -- Swinging on a swing can be regarded as a translation motion.
  let P2 := false -- Two lines intersected by a third line have equal corresponding angles.
  let P3 := true  -- There is one and only one line passing through a point parallel to a given line.
  let P4 := false -- Angles that are not vertical angles are not equal.
  (if P1 then 1 else 0) + (if P2 then 1 else 0) + (if P3 then 1 else 0) + (if P4 then 1 else 0) = 1 :=
by
  sorry

end number_of_true_propositions_l407_40744


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l407_40762

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l407_40762


namespace grain_storage_bins_total_l407_40737

theorem grain_storage_bins_total
  (b20 : ℕ) (b20_tonnage : ℕ) (b15_tonnage : ℕ) (total_capacity : ℕ) (b20_count : ℕ)
  (h_b20_capacity : b20_count * b20_tonnage = b20)
  (h_total_capacity : b20 + (total_capacity - b20) = total_capacity)
  (h_b20_given : b20_count = 12)
  (h_b20_tonnage : b20_tonnage = 20)
  (h_b15_tonnage : b15_tonnage = 15)
  (h_total_capacity_given : total_capacity = 510) :
  ∃ b_total : ℕ, b_total = b20_count + ((total_capacity - (b20_count * b20_tonnage)) / b15_tonnage) ∧ b_total = 30 :=
by
  sorry

end grain_storage_bins_total_l407_40737


namespace num_workers_in_factory_l407_40703

theorem num_workers_in_factory 
  (average_salary_total : ℕ → ℕ → ℕ)
  (old_supervisor_salary : ℕ)
  (average_salary_9_new : ℕ)
  (new_supervisor_salary : ℕ) :
  ∃ (W : ℕ), 
  average_salary_total (W + 1) 430 = W * 430 + 870 ∧ 
  average_salary_9_new = 9 * 390 ∧ 
  W + 1 = (9 * 390 - 510 + 870) / 430 := 
by {
  sorry
}

end num_workers_in_factory_l407_40703


namespace average_salary_l407_40722

def A_salary : ℝ := 9000
def B_salary : ℝ := 5000
def C_salary : ℝ := 11000
def D_salary : ℝ := 7000
def E_salary : ℝ := 9000
def number_of_people : ℝ := 5
def total_salary : ℝ := A_salary + B_salary + C_salary + D_salary + E_salary

theorem average_salary : (total_salary / number_of_people) = 8200 := by
  sorry

end average_salary_l407_40722


namespace count_students_in_meets_l407_40788

theorem count_students_in_meets (A B : Finset ℕ) (hA : A.card = 13) (hB : B.card = 12) (hAB : (A ∩ B).card = 6) :
  (A ∪ B).card = 19 :=
by
  sorry

end count_students_in_meets_l407_40788


namespace greatest_common_divisor_l407_40725

theorem greatest_common_divisor (n : ℕ) (h1 : ∃ d : ℕ, d = gcd 180 n ∧ (∃ (l : List ℕ), l.length = 5 ∧ ∀ x : ℕ, x ∈ l → x ∣ d)) :
  ∃ x : ℕ, x = 27 :=
by
  sorry

end greatest_common_divisor_l407_40725


namespace age_difference_constant_l407_40743

theorem age_difference_constant (seokjin_age_mother_age_diff : ∀ (t : ℕ), 33 - 7 = 26) : 
  ∀ (n : ℕ), 33 + n - (7 + n) = 26 := 
by
  sorry

end age_difference_constant_l407_40743


namespace ellen_bought_chairs_l407_40733

-- Define the conditions
def cost_per_chair : ℕ := 15
def total_amount_spent : ℕ := 180

-- State the theorem to be proven
theorem ellen_bought_chairs :
  (total_amount_spent / cost_per_chair = 12) := 
sorry

end ellen_bought_chairs_l407_40733


namespace least_x_value_l407_40736

variable (a b : ℕ)
variable (positive_int_a : 0 < a)
variable (positive_int_b : 0 < b)
variable (h : 2 * a^5 = 3 * b^2)

theorem least_x_value (h : 2 * a^5 = 3 * b^2) (positive_int_a : 0 < a) (positive_int_b : 0 < b) : ∃ x, x = 15552 ∧ x = 2 * a^5 ∧ x = 3 * b^2 :=
sorry

end least_x_value_l407_40736


namespace problem_f_x_sum_neg_l407_40799

open Function

-- Definitions for monotonic decreasing and odd properties of the function
def isOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def isMonotonicallyDecreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f y ≤ f x

-- The main theorem to prove
theorem problem_f_x_sum_neg
  (f : ℝ → ℝ)
  (h_odd : isOdd f)
  (h_monotone : isMonotonicallyDecreasing f)
  (x₁ x₂ x₃ : ℝ)
  (h₁ : x₁ + x₂ > 0)
  (h₂ : x₂ + x₃ > 0)
  (h₃ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
by
  sorry

end problem_f_x_sum_neg_l407_40799


namespace problem1_problem2_l407_40767

-- Problem (1): Maximum value of (a + 1/a)(b + 1/b)
theorem problem1 {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/a) * (b + 1/b) ≤ 25 / 4 := 
sorry

-- Problem (2): Minimum value of u = (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3
theorem problem2 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000 / 9 :=
sorry

end problem1_problem2_l407_40767


namespace Amy_finish_time_l407_40740

-- Definitions and assumptions based on conditions
def Patrick_time : ℕ := 60
def Manu_time : ℕ := Patrick_time + 12
def Amy_time : ℕ := Manu_time / 2

-- Theorem statement to be proved
theorem Amy_finish_time : Amy_time = 36 :=
by
  sorry

end Amy_finish_time_l407_40740


namespace pie_filling_cans_l407_40779

-- Conditions
def price_per_pumpkin : ℕ := 3
def total_pumpkins : ℕ := 83
def total_revenue : ℕ := 96
def pumpkins_per_can : ℕ := 3

-- Definition
def cans_of_pie_filling (price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can : ℕ) : ℕ :=
  let pumpkins_sold := total_revenue / price_per_pumpkin
  let pumpkins_remaining := total_pumpkins - pumpkins_sold
  pumpkins_remaining / pumpkins_per_can

-- Theorem
theorem pie_filling_cans : cans_of_pie_filling price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can = 17 :=
  by sorry

end pie_filling_cans_l407_40779


namespace min_value_fraction_l407_40726

theorem min_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 3) : 
  ∃ v, v = (x + y) / x ∧ v = -2 := 
by 
  sorry

end min_value_fraction_l407_40726


namespace jake_sister_weight_ratio_l407_40761

theorem jake_sister_weight_ratio (Jake_initial_weight : ℕ) (total_weight : ℕ) (weight_loss : ℕ) (sister_weight : ℕ) 
(h₁ : Jake_initial_weight = 156) 
(h₂ : total_weight = 224) 
(h₃ : weight_loss = 20) 
(h₄ : total_weight = Jake_initial_weight + sister_weight) :
(Jake_initial_weight - weight_loss) / sister_weight = 2 := by
  sorry

end jake_sister_weight_ratio_l407_40761


namespace sum_of_vertices_l407_40707

theorem sum_of_vertices (vertices_rectangle : ℕ) (vertices_pentagon : ℕ) 
  (h_rect : vertices_rectangle = 4) (h_pent : vertices_pentagon = 5) : 
  vertices_rectangle + vertices_pentagon = 9 :=
by
  sorry

end sum_of_vertices_l407_40707


namespace value_range_f_in_0_to_4_l407_40709

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem value_range_f_in_0_to_4 :
  ∀ (x : ℝ), (0 < x ∧ x ≤ 4) → (1 ≤ f x ∧ f x ≤ 10) :=
sorry

end value_range_f_in_0_to_4_l407_40709


namespace max_xy_value_l407_40708

theorem max_xy_value {x y : ℝ} (h : 2 * x + y = 1) : ∃ z, z = x * y ∧ z = 1 / 8 :=
by sorry

end max_xy_value_l407_40708


namespace five_points_distance_ratio_ge_two_sin_54_l407_40775

theorem five_points_distance_ratio_ge_two_sin_54
  (points : Fin 5 → ℝ × ℝ)
  (distinct : Function.Injective points) :
  let distances := {d : ℝ | ∃ (i j : Fin 5), i ≠ j ∧ d = dist (points i) (points j)}
  ∃ (max_dist min_dist : ℝ), max_dist ∈ distances ∧ min_dist ∈ distances ∧ max_dist / min_dist ≥ 2 * Real.sin (54 * Real.pi / 180) := by
  sorry

end five_points_distance_ratio_ge_two_sin_54_l407_40775


namespace basketball_team_points_l407_40760

theorem basketball_team_points (total_points : ℕ) (number_of_players : ℕ) (points_per_player : ℕ) 
  (h1 : total_points = 18) (h2 : number_of_players = 9) : points_per_player = 2 :=
by {
  sorry -- Proof goes here
}

end basketball_team_points_l407_40760


namespace problem_l407_40718

theorem problem
  (a b c d e : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : e < 0)
  (h4 : |e| = 1) :
  (- (a * b))^2009 - (c + d)^2010 - e^2011 = 0 := 
by
  sorry

end problem_l407_40718


namespace rectangular_prism_volume_is_60_l407_40754

def rectangularPrismVolume (a b c : ℕ) : ℕ := a * b * c 

theorem rectangular_prism_volume_is_60 (a b c : ℕ) 
  (h_ge_2 : a ≥ 2) (h_ge_2_b : b ≥ 2) (h_ge_2_c : c ≥ 2)
  (h_one_face : 2 * ((a-2)*(b-2) + (b-2)*(c-2) + (a-2)*(c-2)) = 24)
  (h_two_faces : 4 * ((a-2) + (b-2) + (c-2)) = 28) :
  rectangularPrismVolume a b c = 60 := 
  by sorry

end rectangular_prism_volume_is_60_l407_40754


namespace find_m_for_min_value_l407_40720

theorem find_m_for_min_value :
  ∃ (m : ℝ), ( ∀ x : ℝ, (y : ℝ) = m * x^2 - 4 * x + 1 → (∃ x_min : ℝ, (∀ x : ℝ, (m * x_min^2 - 4 * x_min + 1 ≤ m * x^2 - 4 * x + 1) → y = -3))) :=
sorry

end find_m_for_min_value_l407_40720


namespace tangent_line_slope_l407_40746

theorem tangent_line_slope (m : ℝ) :
  (∀ x y, (x^2 + y^2 - 4*x + 2 = 0) → (y = m * x)) → (m = 1 ∨ m = -1) := 
by
  intro h
  sorry

end tangent_line_slope_l407_40746


namespace fg_eq_neg7_l407_40791

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem fg_eq_neg7 : f (g 2) = -7 :=
  by
    sorry

end fg_eq_neg7_l407_40791


namespace line_properties_l407_40783

theorem line_properties : ∃ m x_intercept, 
  (∀ (x y : ℝ), 4 * x + 7 * y = 28 → y = m * x + 4) ∧ 
  (∀ (x y : ℝ), y = 0 → 4 * x + 7 * y = 28 → x = x_intercept) ∧ 
  m = -4 / 7 ∧ 
  x_intercept = 7 :=
by 
  sorry

end line_properties_l407_40783


namespace find_expression_value_l407_40782

theorem find_expression_value (a b : ℝ)
  (h1 : a^2 - a - 3 = 0)
  (h2 : b^2 - b - 3 = 0) :
  2 * a^3 + b^2 + 3 * a^2 - 11 * a - b + 5 = 23 :=
  sorry

end find_expression_value_l407_40782


namespace find_b_l407_40704

variables {a b : ℝ}

theorem find_b (h1 : (x - 3) * (x - a) = x^2 - b * x - 10) : b = -1/3 :=
  sorry

end find_b_l407_40704


namespace remainder_is_15_l407_40748

-- Definitions based on conditions
def S : ℕ := 476
def L : ℕ := S + 2395
def quotient : ℕ := 6

-- The proof statement
theorem remainder_is_15 : ∃ R : ℕ, L = quotient * S + R ∧ R = 15 := by
  sorry

end remainder_is_15_l407_40748


namespace simplified_value_of_expression_l407_40727

theorem simplified_value_of_expression :
  (12 ^ 0.6) * (12 ^ 0.4) * (8 ^ 0.2) * (8 ^ 0.8) = 96 := 
by
  sorry

end simplified_value_of_expression_l407_40727


namespace percentage_increase_l407_40714

variables (P : ℝ) (buy_price : ℝ := 0.60 * P) (sell_price : ℝ := 1.08000000000000007 * P)

theorem percentage_increase (h: (0.60 : ℝ) * P = buy_price) (h1: (1.08000000000000007 : ℝ) * P = sell_price) :
  ((sell_price - buy_price) / buy_price) * 100 = 80.00000000000001 :=
  sorry

end percentage_increase_l407_40714


namespace edge_ratio_of_cubes_l407_40773

theorem edge_ratio_of_cubes (a b : ℝ) (h : (a^3) / (b^3) = 64) : a / b = 4 :=
sorry

end edge_ratio_of_cubes_l407_40773


namespace number_of_real_roots_l407_40786

theorem number_of_real_roots :
  ∃ (roots_count : ℕ), roots_count = 2 ∧
  (∀ x : ℝ, x^2 - |2 * x - 1| - 4 = 0 → (x = -1 - Real.sqrt 6 ∨ x = 3)) :=
sorry

end number_of_real_roots_l407_40786


namespace area_shaded_region_is_75_l407_40747

-- Define the side length of the larger square
def side_length_large_square : ℝ := 10

-- Define the side length of the smaller square
def side_length_small_square : ℝ := 5

-- Define the area of the larger square
def area_large_square : ℝ := side_length_large_square ^ 2

-- Define the area of the smaller square
def area_small_square : ℝ := side_length_small_square ^ 2

-- Define the area of the shaded region
def area_shaded_region : ℝ := area_large_square - area_small_square

-- The theorem that states the area of the shaded region is 75 square units
theorem area_shaded_region_is_75 : area_shaded_region = 75 := by
  -- The proof will be filled in here when required
  sorry

end area_shaded_region_is_75_l407_40747


namespace hot_dogs_served_for_dinner_l407_40797

theorem hot_dogs_served_for_dinner
  (l t : ℕ) 
  (h_cond1 : l = 9) 
  (h_cond2 : t = 11) :
  ∃ d : ℕ, d = t - l ∧ d = 2 := by
  sorry

end hot_dogs_served_for_dinner_l407_40797


namespace points_per_round_l407_40757

def total_points : ℕ := 78
def num_rounds : ℕ := 26

theorem points_per_round : total_points / num_rounds = 3 := by
  sorry

end points_per_round_l407_40757


namespace math_problem_l407_40793

theorem math_problem
  (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : x^2 + y^2 = 18) :
  x^2 + y^2 = 18 :=
sorry

end math_problem_l407_40793


namespace a5_value_l407_40771

theorem a5_value (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a2 - a1 = 2)
  (h2 : a3 - a2 = 4)
  (h3 : a4 - a3 = 8)
  (h4 : a5 - a4 = 16) :
  a5 = 31 := by
  sorry

end a5_value_l407_40771


namespace glass_original_water_l407_40770

theorem glass_original_water 
  (O : ℝ)  -- Ounces of water originally in the glass
  (evap_per_day : ℝ)  -- Ounces of water evaporated per day
  (total_days : ℕ)    -- Total number of days evaporation occurs
  (percent_evaporated : ℝ)  -- Percentage of the original amount that evaporated
  (h1 : evap_per_day = 0.06)  -- 0.06 ounces of water evaporated each day
  (h2 : total_days = 20)  -- Evaporation occurred over a period of 20 days
  (h3 : percent_evaporated = 0.12)  -- 12% of the original amount evaporated during this period
  (h4 : evap_per_day * total_days = 1.2)  -- 0.06 ounces per day for 20 days total gives 1.2 ounces
  (h5 : percent_evaporated * O = evap_per_day * total_days) :  -- 1.2 ounces is 12% of the original amount
  O = 10 :=  -- Prove that the original amount is 10 ounces
sorry

end glass_original_water_l407_40770


namespace cube_of_720_diamond_1001_l407_40759

-- Define the operation \diamond
def diamond (a b : ℕ) : ℕ :=
  (Nat.factors (a * b)).toFinset.card

-- Define the specific numbers 720 and 1001
def n1 : ℕ := 720
def n2 : ℕ := 1001

-- Calculate the cubic of the result of diamond operation
def cube_of_diamond : ℕ := (diamond n1 n2) ^ 3

-- The statement to be proved
theorem cube_of_720_diamond_1001 : cube_of_diamond = 216 :=
by {
  sorry
}

end cube_of_720_diamond_1001_l407_40759


namespace complement_M_l407_40792

open Set

-- Definitions and conditions
def U : Set ℝ := univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- Theorem stating the complement of M with respect to the universal set U
theorem complement_M : compl M = {x | x < -2 ∨ x > 2} :=
by
  sorry

end complement_M_l407_40792
