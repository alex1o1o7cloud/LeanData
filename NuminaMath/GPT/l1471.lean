import Mathlib

namespace cylinder_height_l1471_147126

theorem cylinder_height (base_area : ℝ) (h s : ℝ)
  (h_base : base_area > 0)
  (h_ratio : (1 / 3 * base_area * 4.5) / (base_area * h) = 1 / 6)
  (h_cone_height : s = 4.5) :
  h = 9 :=
by
  -- Proof omitted
  sorry

end cylinder_height_l1471_147126


namespace card_stack_partition_l1471_147159

theorem card_stack_partition (n k : ℕ) (cards : Multiset ℕ) (h1 : ∀ x ∈ cards, x ∈ Finset.range (n + 1)) (h2 : cards.sum = k * n!) :
  ∃ stacks : List (Multiset ℕ), stacks.length = k ∧ ∀ stack ∈ stacks, stack.sum = n! :=
sorry

end card_stack_partition_l1471_147159


namespace determine_c_l1471_147134

noncomputable def ab5c_decimal (a b c : ℕ) : ℕ :=
  729 * a + 81 * b + 45 + c

theorem determine_c (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : ∃ k : ℕ, ab5c_decimal a b c = k^2) :
  c = 0 ∨ c = 7 :=
by
  sorry

end determine_c_l1471_147134


namespace yogurt_combinations_l1471_147148

theorem yogurt_combinations (f : ℕ) (t : ℕ) (h_f : f = 4) (h_t : t = 6) :
  (f * (t.choose 2) = 60) :=
by
  rw [h_f, h_t]
  sorry

end yogurt_combinations_l1471_147148


namespace total_depreciation_correct_residual_value_correct_sales_price_correct_l1471_147132

-- Definitions and conditions
def initial_cost := 500000
def max_capacity := 100000
def jul_bottles := 200
def aug_bottles := 15000
def sep_bottles := 12300

def depreciation_per_bottle := initial_cost / max_capacity

-- Part (a)
def total_depreciation_jul := jul_bottles * depreciation_per_bottle
def total_depreciation_aug := aug_bottles * depreciation_per_bottle
def total_depreciation_sep := sep_bottles * depreciation_per_bottle
def total_depreciation := total_depreciation_jul + total_depreciation_aug + total_depreciation_sep

theorem total_depreciation_correct :
  total_depreciation = 137500 := 
by sorry

-- Part (b)
def residual_value := initial_cost - total_depreciation

theorem residual_value_correct :
  residual_value = 362500 := 
by sorry

-- Part (c)
def desired_profit := 10000
def sales_price := residual_value + desired_profit

theorem sales_price_correct :
  sales_price = 372500 := 
by sorry

end total_depreciation_correct_residual_value_correct_sales_price_correct_l1471_147132


namespace sum_interior_numbers_eighth_row_of_pascals_triangle_l1471_147194

theorem sum_interior_numbers_eighth_row_of_pascals_triangle :
  let n := 8
  let sum_all_elements := 2 ^ (n - 1)
  let sum_interior_numbers := sum_all_elements - 2
  sum_interior_numbers = 126 :=
by
  let n := 8
  let sum_all_elements := 2 ^ (n - 1)
  let sum_interior_numbers := sum_all_elements - 2
  show sum_interior_numbers = 126
  sorry

end sum_interior_numbers_eighth_row_of_pascals_triangle_l1471_147194


namespace speed_of_boat_is_correct_l1471_147128

theorem speed_of_boat_is_correct (t : ℝ) (V_b : ℝ) (V_s : ℝ) 
  (h1 : V_s = 19) 
  (h2 : ∀ t, (V_b - V_s) * (2 * t) = (V_b + V_s) * t) :
  V_b = 57 :=
by
  -- Proof will go here
  sorry

end speed_of_boat_is_correct_l1471_147128


namespace number_of_people_in_group_l1471_147147

theorem number_of_people_in_group :
  ∃ (N : ℕ), (∀ (avg_weight : ℝ), 
  ∃ (new_person_weight : ℝ) (replaced_person_weight : ℝ),
  new_person_weight = 85 ∧ replaced_person_weight = 65 ∧
  avg_weight + 2.5 = ((N * avg_weight + (new_person_weight - replaced_person_weight)) / N) ∧ 
  N = 8) :=
by
  sorry

end number_of_people_in_group_l1471_147147


namespace triangle_side_lengths_l1471_147101

noncomputable def side_lengths (a b c : ℝ) : Prop :=
  a = 10 ∧ (a^2 + b^2 + c^2 = 2050) ∧ (c^2 = a^2 + b^2)

theorem triangle_side_lengths :
  ∃ b c : ℝ, side_lengths 10 b c ∧ b = Real.sqrt 925 ∧ c = Real.sqrt 1025 :=
by
  sorry

end triangle_side_lengths_l1471_147101


namespace jim_miles_remaining_l1471_147187

theorem jim_miles_remaining (total_miles : ℕ) (miles_driven : ℕ) (total_miles_eq : total_miles = 1200) (miles_driven_eq : miles_driven = 384) :
  total_miles - miles_driven = 816 :=
by
  sorry

end jim_miles_remaining_l1471_147187


namespace calibration_measurements_l1471_147196

theorem calibration_measurements (holes : Fin 15 → ℝ) (diameter : ℝ)
  (h1 : ∀ i : Fin 15, holes i = 10 + i.val * 0.04)
  (h2 : 10 ≤ diameter ∧ diameter ≤ 10 + 14 * 0.04) :
  ∃ tries : ℕ, (tries ≤ 4) ∧ (∀ (i : Fin 15), if diameter ≤ holes i then True else False) :=
sorry

end calibration_measurements_l1471_147196


namespace mickey_horses_per_week_l1471_147133

def days_in_week : ℕ := 7

def horses_minnie_per_day : ℕ := days_in_week + 3

def horses_twice_minnie_per_day : ℕ := 2 * horses_minnie_per_day

def horses_mickey_per_day : ℕ := horses_twice_minnie_per_day - 6

def horses_mickey_per_week : ℕ := days_in_week * horses_mickey_per_day

theorem mickey_horses_per_week : horses_mickey_per_week = 98 := sorry

end mickey_horses_per_week_l1471_147133


namespace selling_price_of_article_l1471_147119

theorem selling_price_of_article (cost_price : ℕ) (gain_percent : ℕ) (profit : ℕ) (selling_price : ℕ) : 
  cost_price = 100 → gain_percent = 10 → profit = (gain_percent * cost_price) / 100 → selling_price = cost_price + profit → selling_price = 110 :=
by
  intros
  sorry

end selling_price_of_article_l1471_147119


namespace solution_part_for_a_l1471_147163

noncomputable def find_k (k x y n : ℕ) : Prop :=
  gcd x y = 1 ∧ 
  x > 0 ∧ y > 0 ∧ 
  k % (x^2) = 0 ∧ 
  k % (y^2) = 0 ∧ 
  k / (x^2) = n ∧ 
  k / (y^2) = n + 148

theorem solution_part_for_a (k x y n : ℕ) (h : find_k k x y n) : k = 467856 :=
sorry

end solution_part_for_a_l1471_147163


namespace expression_meaningful_l1471_147123

theorem expression_meaningful (x : ℝ) : (∃ y, y = 4 / (x - 5)) ↔ x ≠ 5 :=
by
  sorry

end expression_meaningful_l1471_147123


namespace min_value_a_p_a_q_l1471_147193

theorem min_value_a_p_a_q (a : ℕ → ℕ) (p q : ℕ) (h_arith_geom : ∀ n, a (n + 2) = a (n + 1) + a n * 2)
(h_a9 : a 9 = a 8 + 2 * a 7)
(h_ap_aq : a p * a q = 8 * a 1 ^ 2) :
    (1 / p : ℝ) + (4 / q : ℝ) = 9 / 5 := by
    sorry

end min_value_a_p_a_q_l1471_147193


namespace determine_coordinates_of_M_l1471_147173

def point_in_fourth_quadrant (M : ℝ × ℝ) : Prop :=
  M.1 > 0 ∧ M.2 < 0

def distance_to_x_axis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  |M.2| = d

def distance_to_y_axis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  |M.1| = d

theorem determine_coordinates_of_M :
  ∃ M : ℝ × ℝ, point_in_fourth_quadrant M ∧ distance_to_x_axis M 3 ∧ distance_to_y_axis M 4 ∧ M = (4, -3) :=
by
  sorry

end determine_coordinates_of_M_l1471_147173


namespace proof_of_k_bound_l1471_147120

noncomputable def sets_with_nonempty_intersection_implies_k_bound (k : ℝ) : Prop :=
  let M := {x : ℝ | -1 ≤ x ∧ x < 2}
  let N := {x : ℝ | x ≤ k + 3}
  M ∩ N ≠ ∅ → k ≥ -4

theorem proof_of_k_bound (k : ℝ) : sets_with_nonempty_intersection_implies_k_bound k := by
  intro h
  have : -1 ≤ k + 3 := sorry
  linarith

end proof_of_k_bound_l1471_147120


namespace measure_of_angle_F_l1471_147117

theorem measure_of_angle_F (D E F : ℝ) (hD : D = E) 
  (hF : F = D + 40) (h_sum : D + E + F = 180) : F = 140 / 3 + 40 :=
by
  sorry

end measure_of_angle_F_l1471_147117


namespace ellipse_hyperbola_tangent_l1471_147107

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y - 1)^2 = 4) →
  (m = 6 ∨ m = 12) := by
  sorry

end ellipse_hyperbola_tangent_l1471_147107


namespace nonneg_reals_ineq_l1471_147153

theorem nonneg_reals_ineq 
  (a b x y : ℝ)
  (ha : 0 ≤ a) (hb : 0 ≤ b)
  (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hab : a^5 + b^5 ≤ 1)
  (hxy : x^5 + y^5 ≤ 1) :
  a^2 * x^3 + b^2 * y^3 ≤ 1 :=
sorry

end nonneg_reals_ineq_l1471_147153


namespace flower_pots_problem_l1471_147115

noncomputable def cost_of_largest_pot (x : ℝ) : ℝ := x + 5 * 0.15

theorem flower_pots_problem
  (x : ℝ)       -- cost of the smallest pot
  (total_cost : ℝ) -- total cost of all pots
  (h_total_cost : total_cost = 8.25)
  (h_price_relation : total_cost = 6 * x + (0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.15)) :
  cost_of_largest_pot x = 1.75 :=
by
  sorry

end flower_pots_problem_l1471_147115


namespace no_nat_numbers_satisfy_eqn_l1471_147198

theorem no_nat_numbers_satisfy_eqn (a b : ℕ) : a^2 - 3 * b^2 ≠ 8 := by
  sorry

end no_nat_numbers_satisfy_eqn_l1471_147198


namespace find_a_l1471_147139

open Set

noncomputable def A : Set ℝ := {x | x^2 - 2 * x - 8 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + a * x + a^2 - 12 = 0}

theorem find_a (a : ℝ) : (A ∪ (B a) = A) ↔ (a = -2 ∨ a ≥ 4 ∨ a < -4) := by
  sorry

end find_a_l1471_147139


namespace rebecca_eggs_l1471_147125

theorem rebecca_eggs (groups : ℕ) (eggs_per_group : ℕ) (total_eggs : ℕ) 
  (h1 : groups = 3) (h2 : eggs_per_group = 3) : total_eggs = 9 :=
by
  sorry

end rebecca_eggs_l1471_147125


namespace calculate_expression_l1471_147182

theorem calculate_expression : -Real.sqrt 9 - 4 * (-2) + 2 * Real.cos (Real.pi / 3) = 6 :=
by
  sorry

end calculate_expression_l1471_147182


namespace fish_to_apples_l1471_147169

variable {Fish Loaf Rice Apple : Type}
variable (f : Fish → ℝ) (l : Loaf → ℝ) (r : Rice → ℝ) (a : Apple → ℝ)
variable (F : Fish) (L : Loaf) (A : Apple) (R : Rice)

-- Conditions
axiom cond1 : 4 * f F = 3 * l L
axiom cond2 : l L = 5 * r R
axiom cond3 : r R = 2 * a A

-- Proof statement
theorem fish_to_apples : f F = 7.5 * a A :=
by
  sorry

end fish_to_apples_l1471_147169


namespace scientific_notation_example_l1471_147145

theorem scientific_notation_example :
  284000000 = 2.84 * 10^8 :=
by
  sorry

end scientific_notation_example_l1471_147145


namespace min_value_x_l1471_147157

open Real 

variable (x : ℝ)

theorem min_value_x (hx_pos : 0 < x) 
    (ineq : log x ≥ 2 * log 3 + (1 / 3) * log x + 1) : 
    x ≥ 27 * exp (3 / 2) :=
by 
  sorry

end min_value_x_l1471_147157


namespace colony_fungi_day_l1471_147124

theorem colony_fungi_day (n : ℕ): 
  (4 * 2^n > 150) = (n = 6) :=
sorry

end colony_fungi_day_l1471_147124


namespace three_in_A_even_not_in_A_l1471_147144

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

-- (1) Prove that 3 ∈ A
theorem three_in_A : 3 ∈ A :=
sorry

-- (2) Prove that ∀ k ∈ ℤ, 4k - 2 ∉ A
theorem even_not_in_A (k : ℤ) : (4 * k - 2) ∉ A :=
sorry

end three_in_A_even_not_in_A_l1471_147144


namespace expected_winnings_is_correct_l1471_147185

noncomputable def peculiar_die_expected_winnings : ℝ :=
  (1/4) * 2 + (1/2) * 5 + (1/4) * (-10)

theorem expected_winnings_is_correct :
  peculiar_die_expected_winnings = 0.5 := by
  sorry

end expected_winnings_is_correct_l1471_147185


namespace find_P20_l1471_147190

theorem find_P20 (a b : ℝ) (P : ℝ → ℝ) (hP : ∀ x, P x = x^2 + a * x + b) 
  (h_condition : P 10 + P 30 = 40) : P 20 = -80 :=
by {
  -- Additional statements to structure the proof can go here
  sorry
}

end find_P20_l1471_147190


namespace percent_yield_hydrogen_gas_l1471_147175

theorem percent_yield_hydrogen_gas
  (moles_fe : ℝ) (moles_h2so4 : ℝ) (actual_yield_h2 : ℝ) (theoretical_yield_h2 : ℝ) :
  moles_fe = 3 →
  moles_h2so4 = 4 →
  actual_yield_h2 = 1 →
  theoretical_yield_h2 = moles_fe →
  (actual_yield_h2 / theoretical_yield_h2) * 100 = 33.33 :=
by
  intros h_moles_fe h_moles_h2so4 h_actual_yield_h2 h_theoretical_yield_h2
  sorry

end percent_yield_hydrogen_gas_l1471_147175


namespace farmer_animals_l1471_147181

theorem farmer_animals : 
  ∃ g s : ℕ, 
    35 * g + 40 * s = 2000 ∧ 
    g = 2 * s ∧ 
    (0 < g ∧ 0 < s) ∧ 
    g = 36 ∧ s = 18 := 
by 
  sorry

end farmer_animals_l1471_147181


namespace find_c_l1471_147114

theorem find_c 
  (b c : ℝ) 
  (h1 : 4 = 2 * (1:ℝ)^2 + b * (1:ℝ) + c)
  (h2 : 4 = 2 * (5:ℝ)^2 + b * (5:ℝ) + c) : 
  c = 14 := 
sorry

end find_c_l1471_147114


namespace swimmer_speed_is_4_4_l1471_147110

noncomputable def swimmer_speed_in_still_water (distance : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
(distance / time) + current_speed

theorem swimmer_speed_is_4_4 :
  swimmer_speed_in_still_water 7 2.5 3.684210526315789 = 4.4 :=
by
  -- This part would contain the proof to show that the calculated speed is 4.4
  sorry

end swimmer_speed_is_4_4_l1471_147110


namespace trig_solution_l1471_147167

noncomputable def solve_trig_system (x y : ℝ) : Prop :=
  (3 * Real.cos x + 4 * Real.sin x = -1.4) ∧ 
  (13 * Real.cos x - 41 * Real.cos y = -45) ∧ 
  (13 * Real.sin x + 41 * Real.sin y = 3)

theorem trig_solution :
  solve_trig_system (112.64 * Real.pi / 180) (347.32 * Real.pi / 180) ∧ 
  solve_trig_system (239.75 * Real.pi / 180) (20.31 * Real.pi / 180) :=
by {
    repeat { sorry }
  }

end trig_solution_l1471_147167


namespace upper_bound_exists_l1471_147189

theorem upper_bound_exists (U : ℤ) :
  (∀ n : ℤ, 1 < 4 * n + 7 ∧ 4 * n + 7 < U) →
  (∃ n_min n_max : ℤ, n_max = n_min + 29 ∧ 4 * n_max + 7 < U ∧ 4 * n_min + 7 > 1) →
  (U = 120) :=
by
  intros h1 h2
  sorry

end upper_bound_exists_l1471_147189


namespace arithmetic_sequence_ratio_l1471_147191

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d

variable {a b : ℕ → ℝ}
variable {S T : ℕ → ℝ}

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 0 + a n) / 2

variable (S_eq_k_mul_n_plus_2 : ∀ n, S n = (n + 2) * (S 0 / (n + 2)))
variable (T_eq_k_mul_n_plus_1 : ∀ n, T n = (n + 1) * (T 0 / (n + 1)))

theorem arithmetic_sequence_ratio (h₁ : arithmetic_sequence a) (h₂ : arithmetic_sequence b)
  (h₃ : ∀ n, S n = sum_first_n_terms a n)
  (h₄ : ∀ n, T n = sum_first_n_terms b n)
  (h₅ : ∀ n, (S n) / (T n) = (n + 2) / (n + 1))
  : a 6 / b 8 = 13 / 16 := 
sorry

end arithmetic_sequence_ratio_l1471_147191


namespace sets_B_C_D_represent_same_function_l1471_147180

theorem sets_B_C_D_represent_same_function :
  (∀ x : ℝ, (2 * x = 2 * (x ^ (3 : ℝ) ^ (1 / 3)))) ∧
  (∀ x t : ℝ, (x ^ 2 + x + 3 = t ^ 2 + t + 3)) ∧
  (∀ x : ℝ, (x ^ 2 = (x ^ 4) ^ (1 / 2))) :=
by
  sorry

end sets_B_C_D_represent_same_function_l1471_147180


namespace solve_system_l1471_147160

variable {R : Type*} [CommRing R] {a b c x y z : R}

theorem solve_system (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h₁ : z + a*y + a^2*x + a^3 = 0) 
  (h₂ : z + b*y + b^2*x + b^3 = 0) 
  (h₃ : z + c*y + c^2*x + c^3 = 0) :
  x = -(a + b + c) ∧ y = (a * b + a * c + b * c) ∧ z = -(a * b * c) := 
sorry

end solve_system_l1471_147160


namespace pq_sum_of_harmonic_and_geometric_sequences_l1471_147171

theorem pq_sum_of_harmonic_and_geometric_sequences
  (x y z : ℝ)
  (h1 : (1 / x - 1 / y) / (1 / y - 1 / z) = 1)
  (h2 : 3 * x * y = 7 * z) :
  ∃ p q : ℕ, (Nat.gcd p q = 1) ∧ p + q = 79 :=
by
  sorry

end pq_sum_of_harmonic_and_geometric_sequences_l1471_147171


namespace height_of_triangle_l1471_147121

theorem height_of_triangle (base height area : ℝ) (h1 : base = 6) (h2 : area = 24) (h3 : area = 1 / 2 * base * height) : height = 8 :=
by sorry

end height_of_triangle_l1471_147121


namespace complex_expression_evaluation_l1471_147137

theorem complex_expression_evaluation (z : ℂ) (h : z = 1 - I) :
  (z^2 - 2 * z) / (z - 1) = -2 * I :=
by
  sorry

end complex_expression_evaluation_l1471_147137


namespace correct_options_l1471_147136

theorem correct_options (a b : ℝ) (h : a > 0) (ha : a^2 = 4 * b) :
  ((a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬ (∃ x1 x2, x1 * x2 > 0 ∧ x^2 + a * x - b < 0)) ∧ 
  (∀ (x1 x2 : ℝ), |x1 - x2| = 4 → x^2 + a * x + b < 4 → 4 = 4)) :=
sorry

end correct_options_l1471_147136


namespace parabola_focus_l1471_147177

theorem parabola_focus (x y : ℝ) : (y^2 = -8 * x) → (x, y) = (-2, 0) :=
by
  sorry

end parabola_focus_l1471_147177


namespace total_population_estimate_l1471_147168

def average_population_min : ℕ := 3200
def average_population_max : ℕ := 3600
def towns : ℕ := 25

theorem total_population_estimate : 
    ∃ x : ℕ, average_population_min ≤ x ∧ x ≤ average_population_max ∧ towns * x = 85000 :=
by 
  sorry

end total_population_estimate_l1471_147168


namespace period_pi_omega_l1471_147164

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  3 * (Real.sin (ω * x)) * (Real.cos (ω * x)) - 4 * (Real.cos (ω * x))^2

theorem period_pi_omega (ω : ℝ) (hω : ω > 0) (period_condition : ∀ x, f x ω = f (x + π) ω)
  (theta : ℝ) (h_f_theta : f theta ω = 1 / 2) :
  f (theta + π / 2) ω + f (theta - π / 4) ω = -13 / 2 :=
by
  sorry

end period_pi_omega_l1471_147164


namespace measured_percentage_weight_loss_l1471_147162

variable (W : ℝ) -- W is the starting weight.
variable (weight_loss_percent : ℝ := 0.12) -- 12% weight loss.
variable (clothes_weight_percent : ℝ := 0.03) -- 3% clothes weight addition.
variable (beverage_weight_percent : ℝ := 0.005) -- 0.5% beverage weight addition.

theorem measured_percentage_weight_loss : 
  (W - ((0.88 * W) + (clothes_weight_percent * W) + (beverage_weight_percent * W))) / W * 100 = 8.5 :=
by
  sorry

end measured_percentage_weight_loss_l1471_147162


namespace find_x_l1471_147109

theorem find_x (x : ℝ) :
  (x * 13.26 + x * 9.43 + x * 77.31 = 470) → (x = 4.7) :=
by
  sorry

end find_x_l1471_147109


namespace alpha_half_in_II_IV_l1471_147112

theorem alpha_half_in_II_IV (k : ℤ) (α : ℝ) (h : 2 * k * π - π / 2 < α ∧ α < 2 * k * π) : 
  (k * π - π / 4 < (α / 2) ∧ (α / 2) < k * π) :=
by
  sorry

end alpha_half_in_II_IV_l1471_147112


namespace mairiad_distance_ratio_l1471_147183

open Nat

theorem mairiad_distance_ratio :
  ∀ (x : ℕ),
  let miles_run := 40
  let miles_walked := 3 * miles_run / 5
  let total_distance := miles_run + miles_walked + x * miles_run
  total_distance = 184 →
  24 + x * 40 = 144 →
  (24 + 3 * 40) / 40 = 3.6 := 
sorry

end mairiad_distance_ratio_l1471_147183


namespace tank_capacity_75_l1471_147118

theorem tank_capacity_75 (c w : ℝ) 
  (h₁ : w = c / 3) 
  (h₂ : (w + 5) / c = 2 / 5) : 
  c = 75 := 
  sorry

end tank_capacity_75_l1471_147118


namespace minimum_value_x_plus_4_div_x_l1471_147103

theorem minimum_value_x_plus_4_div_x (x : ℝ) (hx : x > 0) : x + 4 / x ≥ 4 :=
sorry

end minimum_value_x_plus_4_div_x_l1471_147103


namespace prime_p4_minus_one_sometimes_divisible_by_48_l1471_147140

theorem prime_p4_minus_one_sometimes_divisible_by_48 (p : ℕ) (hp : Nat.Prime p) (hge : p ≥ 7) : 
  ∃ k : ℕ, k ≥ 1 ∧ 48 ∣ p^4 - 1 :=
sorry

end prime_p4_minus_one_sometimes_divisible_by_48_l1471_147140


namespace parabola_shifted_l1471_147155

-- Define the original parabola
def originalParabola (x : ℝ) : ℝ := (x + 2)^2 + 3

-- Shift the parabola by 3 units to the right
def shiftedRight (x : ℝ) : ℝ := originalParabola (x - 3)

-- Then shift the parabola 2 units down
def shiftedRightThenDown (x : ℝ) : ℝ := shiftedRight x - 2

-- The problem asks to prove that the final expression is equal to (x - 1)^2 + 1
theorem parabola_shifted (x : ℝ) : shiftedRightThenDown x = (x - 1)^2 + 1 :=
by
  sorry

end parabola_shifted_l1471_147155


namespace katya_classmates_l1471_147151

-- Let N be the number of Katya's classmates
variable (N : ℕ)

-- Let K be the number of candies Artyom initially received
variable (K : ℕ)

-- Condition 1: After distributing some candies, Katya had 10 more candies left than Artyom
def condition_1 := K + 10

-- Condition 2: Katya gave each child, including herself, one more candy, so she gave out N + 1 candies in total
def condition_2 := N + 1

-- Condition 3: After giving out these N + 1 candies, everyone in the class has the same number of candies.
def condition_3 : Prop := (K + 1) = (condition_1 K - condition_2 N) / (N + 1)


-- Goal: Prove the number of Katya's classmates N is 9.
theorem katya_classmates : N = 9 :=
by
  -- Restate the conditions in Lean
  
  -- Apply the conditions to find that the only viable solution is N = 9
  sorry

end katya_classmates_l1471_147151


namespace simplify_expression1_simplify_expression2_l1471_147146

-- Problem 1 statement
theorem simplify_expression1 (a b : ℤ) : 2 * (2 * b - 3 * a) + 3 * (2 * a - 3 * b) = -5 * b :=
  by
  sorry

-- Problem 2 statement
theorem simplify_expression2 (a b : ℤ) : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 :=
  by
  sorry

end simplify_expression1_simplify_expression2_l1471_147146


namespace sum_of_first_11_terms_l1471_147154

theorem sum_of_first_11_terms (a1 d : ℝ) (h : 2 * a1 + 10 * d = 8) : 
  (11 / 2) * (2 * a1 + 10 * d) = 44 := 
by sorry

end sum_of_first_11_terms_l1471_147154


namespace f_not_monotonic_l1471_147141

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-(x:ℝ)) = -f x

def is_not_monotonic (f : ℝ → ℝ) : Prop :=
  ¬ ( (∀ x y, x < y → f x ≤ f y) ∨ (∀ x y, x < y → f y ≤ f x) )

variable (f : ℝ → ℝ)

axiom periodicity : ∀ x, f (x + 3/2) = -f x 
axiom odd_shifted : is_odd_function (λ x => f (x - 3/4))

theorem f_not_monotonic : is_not_monotonic f := by
  sorry

end f_not_monotonic_l1471_147141


namespace winning_candidate_percentage_l1471_147166

noncomputable def votes : List ℝ := [15236.71, 20689.35, 12359.23, 30682.49, 25213.17, 18492.93]

theorem winning_candidate_percentage :
  (List.foldr max 0 votes / (List.foldr (· + ·) 0 votes) * 100) = 25.01 :=
by
  sorry

end winning_candidate_percentage_l1471_147166


namespace eggs_leftover_l1471_147129

theorem eggs_leftover (eggs_abigail eggs_beatrice eggs_carson cartons : ℕ)
  (h_abigail : eggs_abigail = 37)
  (h_beatrice : eggs_beatrice = 49)
  (h_carson : eggs_carson = 14)
  (h_cartons : cartons = 12) :
  ((eggs_abigail + eggs_beatrice + eggs_carson) % cartons) = 4 :=
by
  sorry

end eggs_leftover_l1471_147129


namespace scientific_notation_eight_million_l1471_147138

theorem scientific_notation_eight_million :
  ∃ a n, 8000000 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8 ∧ n = 6 :=
by
  use 8
  use 6
  sorry

end scientific_notation_eight_million_l1471_147138


namespace parallelogram_area_l1471_147161

theorem parallelogram_area (base height : ℝ) (h_base : base = 20) (h_height : height = 16) :
  base * height = 320 :=
by
  sorry

end parallelogram_area_l1471_147161


namespace arithmetic_mean_of_reciprocals_of_first_five_primes_l1471_147184

theorem arithmetic_mean_of_reciprocals_of_first_five_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7 + 1 / 11) / 5 = 2927 / 11550 := 
sorry

end arithmetic_mean_of_reciprocals_of_first_five_primes_l1471_147184


namespace q_true_given_not_p_and_p_or_q_l1471_147127

theorem q_true_given_not_p_and_p_or_q (p q : Prop) (hnp : ¬p) (hpq : p ∨ q) : q :=
by
  sorry

end q_true_given_not_p_and_p_or_q_l1471_147127


namespace joan_books_l1471_147188

theorem joan_books (initial_books sold_books result_books : ℕ) 
  (h_initial : initial_books = 33) 
  (h_sold : sold_books = 26) 
  (h_result : initial_books - sold_books = result_books) : 
  result_books = 7 := 
by
  sorry

end joan_books_l1471_147188


namespace circle_radius_square_l1471_147156

-- Definition of the problem setup
variables {EF GH ER RF GS SH R S : ℝ}

-- Given conditions
def condition1 : ER = 23 := by sorry
def condition2 : RF = 23 := by sorry
def condition3 : GS = 31 := by sorry
def condition4 : SH = 15 := by sorry

-- Circle radius to be proven
def radius_squared : ℝ := 706

-- Lean 4 theorem statement
theorem circle_radius_square (h1 : ER = 23) (h2 : RF = 23) (h3 : GS = 31) (h4 : SH = 15) :
  (r : ℝ) ^ 2 = 706 := sorry

end circle_radius_square_l1471_147156


namespace y_pow_expression_l1471_147100

theorem y_pow_expression (y : ℝ) (h : y + 1/y = 3) : y^13 - 5 * y^9 + y^5 = 0 :=
sorry

end y_pow_expression_l1471_147100


namespace partial_fraction_decomposition_l1471_147108

noncomputable def A := 29 / 15
noncomputable def B := 13 / 12
noncomputable def C := 37 / 15

theorem partial_fraction_decomposition :
  let ABC := A * B * C;
  ABC = 13949 / 2700 :=
by
  sorry

end partial_fraction_decomposition_l1471_147108


namespace expr_value_l1471_147135

theorem expr_value : (34 + 7)^2 - (7^2 + 34^2 + 7 * 34) = 238 := by
  sorry

end expr_value_l1471_147135


namespace option_C_true_l1471_147104

variable {a b : ℝ}

theorem option_C_true (h : a < b) : a / 3 < b / 3 := sorry

end option_C_true_l1471_147104


namespace cos_alpha_plus_pi_over_2_l1471_147178

theorem cos_alpha_plus_pi_over_2 (α : ℝ) (h : Real.sin α = 1/3) : 
    Real.cos (α + Real.pi / 2) = -(1/3) :=
by
  sorry

end cos_alpha_plus_pi_over_2_l1471_147178


namespace alice_savings_l1471_147186

noncomputable def commission (sales : ℝ) : ℝ := 0.02 * sales
noncomputable def totalEarnings (basic_salary commission : ℝ) : ℝ := basic_salary + commission
noncomputable def savings (total_earnings : ℝ) : ℝ := 0.10 * total_earnings

theorem alice_savings (sales basic_salary : ℝ) (commission_rate savings_rate : ℝ) :
  commission_rate = 0.02 →
  savings_rate = 0.10 →
  sales = 2500 →
  basic_salary = 240 →
  savings (totalEarnings basic_salary (commission_rate * sales)) = 29 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_savings_l1471_147186


namespace unique_solution_l1471_147179

-- Given conditions in the problem:
def prime (p : ℕ) : Prop := Nat.Prime p
def is_solution (p n k : ℕ) : Prop :=
  3 ^ p + 4 ^ p = n ^ k ∧ k > 1 ∧ prime p

-- The only solution:
theorem unique_solution (p n k : ℕ) :
  is_solution p n k → (p, n, k) = (2, 5, 2) := 
by
  sorry

end unique_solution_l1471_147179


namespace farmer_plants_rows_per_bed_l1471_147149

theorem farmer_plants_rows_per_bed 
    (bean_seedlings : ℕ) (beans_per_row : ℕ)
    (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
    (radishes : ℕ) (radishes_per_row : ℕ)
    (plant_beds : ℕ)
    (h1 : bean_seedlings = 64)
    (h2 : beans_per_row = 8)
    (h3 : pumpkin_seeds = 84)
    (h4 : pumpkins_per_row = 7)
    (h5 : radishes = 48)
    (h6 : radishes_per_row = 6)
    (h7 : plant_beds = 14) : 
    (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row + radishes / radishes_per_row) / plant_beds = 2 :=
by
  sorry

end farmer_plants_rows_per_bed_l1471_147149


namespace complex_in_third_quadrant_l1471_147192

theorem complex_in_third_quadrant (x : ℝ) : 
  (x^2 - 6*x + 5 < 0) ∧ (x - 2 < 0) ↔ (1 < x ∧ x < 2) := 
by
  sorry

end complex_in_third_quadrant_l1471_147192


namespace find_value_of_a3_a6_a9_l1471_147122

-- Definitions from conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d

variables {a : ℕ → ℤ} (d : ℤ)

-- Given conditions
axiom cond1 : a 1 + a 4 + a 7 = 45
axiom cond2 : a 2 + a 5 + a 8 = 29

-- Lean 4 Statement
theorem find_value_of_a3_a6_a9 : a 3 + a 6 + a 9 = 13 :=
sorry

end find_value_of_a3_a6_a9_l1471_147122


namespace find_k_when_root_is_zero_l1471_147165

-- Define the quadratic equation and what it implies
theorem find_k_when_root_is_zero (k : ℝ) (h : (k-1) * 0^2 + 6 * 0 + k^2 - k = 0) :
  k = 0 :=
by
  -- The proof steps would go here, but we're skipping it as instructed
  sorry

end find_k_when_root_is_zero_l1471_147165


namespace total_income_in_june_l1471_147170

-- Establishing the conditions
def daily_production : ℕ := 200
def days_in_june : ℕ := 30
def price_per_gallon : ℝ := 3.55

-- Defining total milk production in June as a function of daily production and days in June
def total_milk_production_in_june : ℕ :=
  daily_production * days_in_june

-- Defining total income as a function of milk production and price per gallon
def total_income (milk_production : ℕ) (price : ℝ) : ℝ :=
  milk_production * price

-- Stating the theorem that we need to prove
theorem total_income_in_june :
  total_income total_milk_production_in_june price_per_gallon = 21300 := 
sorry

end total_income_in_june_l1471_147170


namespace domain_shift_l1471_147152

theorem domain_shift (f : ℝ → ℝ) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2} = {x | -2 ≤ x ∧ x ≤ -1} →
  {x : ℝ | ∃ y : ℝ, x = y - 1 ∧ 1 ≤ y ∧ y ≤ 2} =
  {x : ℝ | ∃ y : ℝ, x = y + 2 ∧ -2 ≤ y ∧ y ≤ -1} :=
by
  sorry

end domain_shift_l1471_147152


namespace no_three_by_three_red_prob_l1471_147197

theorem no_three_by_three_red_prob : 
  ∃ (m n : ℕ), 
  Nat.gcd m n = 1 ∧ 
  m / n = 340 / 341 ∧ 
  m + n = 681 :=
by
  sorry

end no_three_by_three_red_prob_l1471_147197


namespace rockets_win_30_l1471_147116

-- Given conditions
def hawks_won (h : ℕ) (w : ℕ) : Prop := h > w
def rockets_won (r : ℕ) (k : ℕ) (l : ℕ) : Prop := r > k ∧ r < l
def knicks_at_least (k : ℕ) : Prop := k ≥ 15
def clippers_won (c : ℕ) (l : ℕ) : Prop := c < l

-- Possible number of games won
def possible_games : List ℕ := [15, 20, 25, 30, 35, 40]

-- Prove Rockets won 30 games
theorem rockets_win_30 (h w r k l c : ℕ) 
  (h_w: hawks_won h w)
  (r_kl : rockets_won r k l)
  (k_15: knicks_at_least k)
  (c_l : clippers_won c l)
  (h_mem : h ∈ possible_games)
  (w_mem : w ∈ possible_games)
  (r_mem : r ∈ possible_games)
  (k_mem : k ∈ possible_games)
  (l_mem : l ∈ possible_games)
  (c_mem : c ∈ possible_games) :
  r = 30 :=
sorry

end rockets_win_30_l1471_147116


namespace avg_mark_excluded_students_l1471_147158

-- Define the given conditions
variables (n : ℕ) (A A_remaining : ℕ) (excluded_count : ℕ)
variable (T : ℕ := n * A)
variable (T_remaining : ℕ := (n - excluded_count) * A_remaining)
variable (T_excluded : ℕ := T - T_remaining)

-- Define the problem statement
theorem avg_mark_excluded_students (h1: n = 14) (h2: A = 65) (h3: A_remaining = 90) (h4: excluded_count = 5) :
   T_excluded / excluded_count = 20 :=
by
  sorry

end avg_mark_excluded_students_l1471_147158


namespace masha_can_climb_10_steps_l1471_147106

def ways_to_climb_stairs : ℕ → ℕ 
| 0 => 1
| 1 => 1
| n + 2 => ways_to_climb_stairs (n + 1) + ways_to_climb_stairs n

theorem masha_can_climb_10_steps : ways_to_climb_stairs 10 = 89 :=
by
  -- proof omitted here as per instruction
  sorry

end masha_can_climb_10_steps_l1471_147106


namespace example_problem_l1471_147102

theorem example_problem
  (h1 : 0.25 < 1) 
  (h2 : 0.15 < 0.25) : 
  3.04 / 0.25 > 1 :=
by
  sorry

end example_problem_l1471_147102


namespace ratio_a_to_c_l1471_147105

-- Declaring the variables a, b, c, and d as real numbers.
variables (a b c d : ℝ)

-- Define the conditions given in the problem.
def ratio_conditions : Prop :=
  (a / b = 5 / 4) ∧ (c / d = 4 / 3) ∧ (d / b = 1 / 5)

-- State the theorem we need to prove based on the conditions.
theorem ratio_a_to_c (h : ratio_conditions a b c d) : a / c = 75 / 16 :=
by
  sorry

end ratio_a_to_c_l1471_147105


namespace frequency_of_8th_group_l1471_147172

theorem frequency_of_8th_group :
  let sample_size := 100
  let freq1 := 15
  let freq2 := 17
  let freq3 := 11
  let freq4 := 13
  let freq_5_to_7 := 0.32 * sample_size
  let total_freq_1_to_4 := freq1 + freq2 + freq3 + freq4
  let remaining_freq := sample_size - total_freq_1_to_4
  let freq8 := remaining_freq - freq_5_to_7
  (freq8 / sample_size = 0.12) :=
by
  sorry

end frequency_of_8th_group_l1471_147172


namespace ratio_expression_value_l1471_147174

theorem ratio_expression_value (p q s u : ℚ) (h1 : p / q = 5 / 2) (h2 : s / u = 11 / 7) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 2 * p * s) = -233 / 12 :=
by {
  -- Proof will be provided here.
  sorry
}

end ratio_expression_value_l1471_147174


namespace max_sum_of_roots_l1471_147195

theorem max_sum_of_roots (a b : ℝ) (h_a : a ≠ 0) (m : ℝ) :
  (∀ x : ℝ, (2 * x ^ 2 - 5 * x + m = 0) → 25 - 8 * m ≥ 0) →
  (∃ s, s = -5 / 2) → m = 25 / 8 :=
by
  sorry

end max_sum_of_roots_l1471_147195


namespace sum_of_distances_minimized_l1471_147130

theorem sum_of_distances_minimized (x : ℝ) (h : 0 ≤ x ∧ x ≤ 50) : 
  abs (x - 0) + abs (x - 50) = 50 := 
by
  sorry

end sum_of_distances_minimized_l1471_147130


namespace find_a_l1471_147111

theorem find_a (a : ℝ) (h_pos : a > 0)
  (h_eq : ∀ (f g : ℝ → ℝ), (f = λ x => x^2 + 10) → (g = λ x => x^2 - 6) → f (g a) = 14) :
  a = 2 * Real.sqrt 2 ∨ a = 2 :=
by 
  sorry

end find_a_l1471_147111


namespace sufficient_condition_not_necessary_condition_l1471_147131

theorem sufficient_condition (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : ab > 0 := by
  sorry

theorem not_necessary_condition (a b : ℝ) : ¬(a > 0 ∧ b > 0) → ab > 0 := by
  sorry

end sufficient_condition_not_necessary_condition_l1471_147131


namespace full_size_mustang_length_l1471_147199

theorem full_size_mustang_length 
  (smallest_model_length : ℕ)
  (mid_size_factor : ℕ)
  (full_size_factor : ℕ)
  (h1 : smallest_model_length = 12)
  (h2 : mid_size_factor = 2)
  (h3 : full_size_factor = 10) :
  (smallest_model_length * mid_size_factor) * full_size_factor = 240 := 
sorry

end full_size_mustang_length_l1471_147199


namespace find_constant_a_l1471_147176

theorem find_constant_a (a : ℝ) : 
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ ax^2 + 2 * a * x + 1 = 9) → (a = -8 ∨ a = 1) :=
by
  sorry

end find_constant_a_l1471_147176


namespace smallest_n_for_terminating_decimal_l1471_147113

def is_terminating_decimal (n d : ℕ) : Prop :=
  ∀ (m : ℕ), d = 2^m ∨ d = 5^m ∨ d = (2^m) * (5 : ℕ) ∨ d = (5^m) * (2 : ℕ)
  
theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, 0 < n ∧ is_terminating_decimal n (n + 150) ∧ (∀ m: ℕ, (is_terminating_decimal m (m + 150) ∧ 0 < m) → n ≤ m) :=
sorry

end smallest_n_for_terminating_decimal_l1471_147113


namespace triangle_ineq_l1471_147143

noncomputable def TriangleSidesProof (AB AC BC : ℝ) :=
  AB = AC ∧ BC = 10 ∧ 2 * AB + BC ≤ 44 → 5 < AB ∧ AB ≤ 17

-- Statement for the proof problem
theorem triangle_ineq (AB AC BC : ℝ) (h1 : AB = AC) (h2 : BC = 10) (h3 : 2 * AB + BC ≤ 44) :
  5 < AB ∧ AB ≤ 17 :=
sorry

end triangle_ineq_l1471_147143


namespace range_m_if_neg_p_implies_neg_q_range_x_if_m_is_5_and_p_or_q_true_p_and_q_false_l1471_147142

-- Question 1
def prop_p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def prop_q (x m : ℝ) : Prop := 1 - m ≤ x + 1 ∧ x + 1 < 1 + m ∧ m > 0
def neg_p (x : ℝ) : Prop := ¬ prop_p x
def neg_q (x m : ℝ) : Prop := ¬ prop_q x m

theorem range_m_if_neg_p_implies_neg_q : 
  (∀ x, neg_p x → neg_q x m) → 0 < m ∧ m ≤ 1 :=
by
  sorry

-- Question 2
theorem range_x_if_m_is_5_and_p_or_q_true_p_and_q_false : 
  (∀ x, (prop_p x ∨ prop_q x 5) ∧ ¬ (prop_p x ∧ prop_q x 5)) → 
  ∀ x, (x = 5 ∨ (-5 ≤ x ∧ x < -1)) :=
by
  sorry

end range_m_if_neg_p_implies_neg_q_range_x_if_m_is_5_and_p_or_q_true_p_and_q_false_l1471_147142


namespace total_sticks_of_gum_in_12_brown_boxes_l1471_147150

-- Definitions based on the conditions
def packs_per_carton := 7
def sticks_per_pack := 5
def cartons_in_full_box := 6
def cartons_in_partial_box := 3
def num_brown_boxes := 12
def num_partial_boxes := 2

-- Calculation definitions
def sticks_per_carton := packs_per_carton * sticks_per_pack
def sticks_per_full_box := cartons_in_full_box * sticks_per_carton
def sticks_per_partial_box := cartons_in_partial_box * sticks_per_carton
def num_full_boxes := num_brown_boxes - num_partial_boxes

-- Final total sticks of gum
def total_sticks_of_gum := (num_full_boxes * sticks_per_full_box) + (num_partial_boxes * sticks_per_partial_box)

-- The theorem to be proved
theorem total_sticks_of_gum_in_12_brown_boxes :
  total_sticks_of_gum = 2310 :=
by
  -- The proof is omitted.
  sorry

end total_sticks_of_gum_in_12_brown_boxes_l1471_147150
