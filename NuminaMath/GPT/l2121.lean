import Mathlib

namespace largest_non_representable_as_sum_of_composites_l2121_212147

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end largest_non_representable_as_sum_of_composites_l2121_212147


namespace maximum_p_value_l2121_212135

noncomputable def max_p_value (a b c : ℝ) : ℝ :=
  2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1)

theorem maximum_p_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c + a + c = b) :
  ∃ p_max, p_max = 10 / 3 ∧ ∀ p, p = max_p_value a b c → p ≤ p_max :=
sorry

end maximum_p_value_l2121_212135


namespace area_EFGH_l2121_212190

theorem area_EFGH (n : ℕ) (n_pos : 1 < n) (S_ABCD : ℝ) (h₁ : S_ABCD = 1) :
  ∃ S_EFGH : ℝ, S_EFGH = (n - 2) / n :=
by sorry

end area_EFGH_l2121_212190


namespace theo_needs_84_eggs_l2121_212117

def customers_hour1 := 5
def customers_hour2 := 7
def customers_hour3 := 3
def customers_hour4 := 8

def eggs_per_omelette_3 := 3
def eggs_per_omelette_4 := 4

def total_eggs_needed : Nat :=
  (customers_hour1 * eggs_per_omelette_3) +
  (customers_hour2 * eggs_per_omelette_4) +
  (customers_hour3 * eggs_per_omelette_3) +
  (customers_hour4 * eggs_per_omelette_4)

theorem theo_needs_84_eggs : total_eggs_needed = 84 :=
by
  sorry

end theo_needs_84_eggs_l2121_212117


namespace polynomial_solution_l2121_212116

theorem polynomial_solution (P : Polynomial ℝ) (h : ∀ x, (x + 2019) * (P.eval x) = x * (P.eval (x + 1))) :
  ∃ C : ℝ, P = Polynomial.C C * Polynomial.X * (Polynomial.X + 2018) :=
sorry

end polynomial_solution_l2121_212116


namespace f_8_plus_f_9_l2121_212159

noncomputable def f : ℝ → ℝ := sorry
axiom f_odd : ∀ x : ℝ, f (-x) = -f x 
axiom f_even_transformed : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom f_at_1 : f 1 = 1

theorem f_8_plus_f_9 : f 8 + f 9 = 1 :=
sorry

end f_8_plus_f_9_l2121_212159


namespace point_on_y_axis_m_value_l2121_212156

theorem point_on_y_axis_m_value (m : ℝ) (h : 6 - 2 * m = 0) : m = 3 := by
  sorry

end point_on_y_axis_m_value_l2121_212156


namespace smallest_w_factor_l2121_212145

theorem smallest_w_factor:
  ∃ w : ℕ, (∃ n : ℕ, n = 936 * w ∧ 
              2 ^ 5 ∣ n ∧ 
              3 ^ 3 ∣ n ∧ 
              14 ^ 2 ∣ n) ∧ 
              w = 1764 :=
sorry

end smallest_w_factor_l2121_212145


namespace line_equation_is_correct_l2121_212114

def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

theorem line_equation_is_correct (x y t : ℝ)
  (h1: x = 3 * t + 6)
  (h2: y = 5 * t - 7) :
  y = (5 / 3) * x - 17 :=
sorry

end line_equation_is_correct_l2121_212114


namespace Carl_avg_gift_bags_l2121_212104

theorem Carl_avg_gift_bags :
  ∀ (known expected extravagant remaining : ℕ), 
  known = 50 →
  expected = 40 →
  extravagant = 10 →
  remaining = 60 →
  (known + expected) - extravagant - remaining = 30 := by
  intros
  sorry

end Carl_avg_gift_bags_l2121_212104


namespace cos_B_in_triangle_l2121_212139

theorem cos_B_in_triangle (A B C : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = Real.pi) : 
  Real.cos B = 1 / 2 :=
sorry

end cos_B_in_triangle_l2121_212139


namespace part_I_part_II_l2121_212198

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos x) ^ 2 - Real.sin (2 * x - (7 * Real.pi / 6))

theorem part_I :
  (∀ x, f x ≤ 2) ∧ (∃ x, f x = 2 ∧ ∃ k : ℤ, x = k * Real.pi + Real.pi / 6) :=
by
  sorry

theorem part_II (A a b c : ℝ) (h1 : f A = 3 / 2) (h2 : b + c = 2) :
  a >= 1 :=
by
  sorry

end part_I_part_II_l2121_212198


namespace booster_club_tickets_l2121_212199

theorem booster_club_tickets (x : ℕ) : 
  (11 * 9 + x * 7 = 225) → 
  (x + 11 = 29) := 
by
  sorry

end booster_club_tickets_l2121_212199


namespace savings_percentage_l2121_212154

theorem savings_percentage (I S : ℝ) (h1 : I > 0) (h2 : S > 0) (h3 : S ≤ I) 
  (h4 : 1.25 * I - 2 * S + I - S = 2 * (I - S)) :
  (S / I) * 100 = 25 :=
by
  sorry

end savings_percentage_l2121_212154


namespace correct_divisor_l2121_212181

variable (D X : ℕ)

-- Conditions
def condition1 : Prop := X = D * 24
def condition2 : Prop := X = (D - 12) * 42

theorem correct_divisor (D X : ℕ) (h1 : condition1 D X) (h2 : condition2 D X) : D = 28 := by
  sorry

end correct_divisor_l2121_212181


namespace SarahCansYesterday_l2121_212102

variable (S : ℕ)
variable (LaraYesterday : ℕ := S + 30)
variable (SarahToday : ℕ := 40)
variable (LaraToday : ℕ := 70)
variable (YesterdayTotal : ℕ := LaraYesterday + S)
variable (TodayTotal : ℕ := SarahToday + LaraToday)

theorem SarahCansYesterday : 
  TodayTotal + 20 = YesterdayTotal -> 
  S = 50 :=
by
  sorry

end SarahCansYesterday_l2121_212102


namespace find_x_y_l2121_212189

theorem find_x_y (x y : ℝ) : (3 * x + 4 * -2 = 0) ∧ (3 * 1 + 4 * y = 0) → x = 8 / 3 ∧ y = -3 / 4 :=
by
  sorry

end find_x_y_l2121_212189


namespace average_of_first_12_l2121_212122

theorem average_of_first_12 (avg25 : ℝ) (avg12 : ℝ) (avg_last12 : ℝ) (result_13th : ℝ) : 
  (avg25 = 18) → (avg_last12 = 17) → (result_13th = 78) → 
  25 * avg25 = (12 * avg12) + result_13th + (12 * avg_last12) → avg12 = 14 :=
by 
  sorry

end average_of_first_12_l2121_212122


namespace largest_divisor_of_n_l2121_212192

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 360 ∣ n^2) : 60 ∣ n := 
sorry

end largest_divisor_of_n_l2121_212192


namespace smallest_number_l2121_212179

theorem smallest_number:
  ∃ n : ℕ, (∀ d ∈ [12, 16, 18, 21, 28, 35, 39], (n - 7) % d = 0) ∧ n = 65527 :=
by
  sorry

end smallest_number_l2121_212179


namespace find_roots_of_polynomial_l2121_212169

noncomputable def polynomial_roots : Set ℝ :=
  {x | (6 * x^4 + 25 * x^3 - 59 * x^2 + 28 * x) = 0 }

theorem find_roots_of_polynomial :
  polynomial_roots = {0, 1, (-31 + Real.sqrt 1633) / 12, (-31 - Real.sqrt 1633) / 12} :=
by
  sorry

end find_roots_of_polynomial_l2121_212169


namespace Anne_mom_toothpaste_usage_l2121_212176

theorem Anne_mom_toothpaste_usage
  (total_toothpaste : ℕ)
  (dad_usage_per_brush : ℕ)
  (sibling_usage_per_brush : ℕ)
  (num_brushes_per_day : ℕ)
  (total_days : ℕ)
  (total_toothpaste_used : ℕ)
  (M : ℕ)
  (family_use_model : total_toothpaste = total_toothpaste_used + 3 * num_brushes_per_day * M)
  (total_toothpaste_used_def : total_toothpaste_used = 5 * (dad_usage_per_brush * num_brushes_per_day + 2 * sibling_usage_per_brush * num_brushes_per_day))
  (given_values : total_toothpaste = 105 ∧ dad_usage_per_brush = 3 ∧ sibling_usage_per_brush = 1 ∧ num_brushes_per_day = 3 ∧ total_days = 5)
  : M = 2 := by
  sorry

end Anne_mom_toothpaste_usage_l2121_212176


namespace total_notebooks_l2121_212131

-- Define the problem conditions
theorem total_notebooks (x : ℕ) (hx : x*x + 20 = (x+1)*(x+1) - 9) : x*x + 20 = 216 :=
by
  have h1 : x*x + 20 = 216 := sorry
  exact h1

end total_notebooks_l2121_212131


namespace find_a_plus_b_l2121_212141

theorem find_a_plus_b (a b : ℝ) (h₁ : ∀ x, x - b < 0 → x < b) 
  (h₂ : ∀ x, x + a > 0 → x > -a) 
  (h₃ : ∀ x, 2 < x ∧ x < 3 → -a < x ∧ x < b) : 
  a + b = 1 :=
by
  sorry

end find_a_plus_b_l2121_212141


namespace evaluate_g_at_neg1_l2121_212165

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem evaluate_g_at_neg1 : g (-1) = -5 := by
  sorry

end evaluate_g_at_neg1_l2121_212165


namespace find_a_l2121_212168

theorem find_a (a x : ℝ) 
  (h : x^2 + 3 * x + a = (x + 1) * (x + 2)) : 
  a = 2 :=
sorry

end find_a_l2121_212168


namespace percentage_for_overnight_stays_l2121_212151

noncomputable def total_bill : ℝ := 5000
noncomputable def medication_percentage : ℝ := 0.50
noncomputable def food_cost : ℝ := 175
noncomputable def ambulance_cost : ℝ := 1700

theorem percentage_for_overnight_stays :
  let medication_cost := medication_percentage * total_bill
  let remaining_bill := total_bill - medication_cost
  let cost_for_overnight_stays := remaining_bill - food_cost - ambulance_cost
  (cost_for_overnight_stays / remaining_bill) * 100 = 25 :=
by
  sorry

end percentage_for_overnight_stays_l2121_212151


namespace team_A_has_more_uniform_heights_l2121_212153

-- Definitions of the conditions
def avg_height_team_A : ℝ := 1.65
def avg_height_team_B : ℝ := 1.65

def variance_team_A : ℝ := 1.5
def variance_team_B : ℝ := 2.4

-- Theorem stating the problem solution
theorem team_A_has_more_uniform_heights :
  variance_team_A < variance_team_B :=
by
  -- Proof omitted
  sorry

end team_A_has_more_uniform_heights_l2121_212153


namespace abigail_fence_building_l2121_212118

theorem abigail_fence_building :
  ∀ (initial_fences : Nat) (time_per_fence : Nat) (hours_building : Nat) (minutes_per_hour : Nat),
    initial_fences = 10 →
    time_per_fence = 30 →
    hours_building = 8 →
    minutes_per_hour = 60 →
    initial_fences + (minutes_per_hour / time_per_fence) * hours_building = 26 :=
by
  intros initial_fences time_per_fence hours_building minutes_per_hour
  sorry

end abigail_fence_building_l2121_212118


namespace geometric_sequence_sum_l2121_212174

theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℕ := λ k => 2^k) 
  (S : ℕ → ℕ := λ k => (1 - 2^k) / (1 - 2)) :
  S (n + 1) = 2 * a n - 1 :=
by
  sorry

end geometric_sequence_sum_l2121_212174


namespace original_number_conditions_l2121_212177

theorem original_number_conditions (a : ℕ) :
  ∃ (y1 y2 : ℕ), (7 * a = 10 * 9 + y1) ∧ (9 * 9 = 10 * 8 + y2) ∧ y2 = 1 ∧ (a = 13 ∨ a = 14) := sorry

end original_number_conditions_l2121_212177


namespace probability_A_not_losing_l2121_212155

variable (P_A_wins : ℝ)
variable (P_draw : ℝ)
variable (P_A_not_losing : ℝ)

theorem probability_A_not_losing 
  (h1 : P_A_wins = 0.3) 
  (h2 : P_draw = 0.5) 
  (h3 : P_A_not_losing = P_A_wins + P_draw) :
  P_A_not_losing = 0.8 :=
sorry

end probability_A_not_losing_l2121_212155


namespace calculate_expression_l2121_212161

theorem calculate_expression :
  50 * 24.96 * 2.496 * 500 = (1248)^2 :=
by
  sorry

end calculate_expression_l2121_212161


namespace standard_equation_of_ellipse_l2121_212191

-- Definitions for clarity
def is_ellipse (E : Type) := true
def major_axis (e : is_ellipse E) : ℝ := sorry
def minor_axis (e : is_ellipse E) : ℝ := sorry
def focus (e : is_ellipse E) : ℝ := sorry

theorem standard_equation_of_ellipse (E : Type)
  (e : is_ellipse E)
  (major_sum : major_axis e + minor_axis e = 9)
  (focus_position : focus e = 3) :
  ∀ x y, (x^2 / 25) + (y^2 / 16) = 1 :=
by sorry

end standard_equation_of_ellipse_l2121_212191


namespace roots_absolute_value_l2121_212150

noncomputable def quadratic_roots_property (p : ℝ) (r1 r2 : ℝ) : Prop :=
  r1 ≠ r2 ∧
  r1 + r2 = -p ∧
  r1 * r2 = 16 ∧
  ∃ r : ℝ, r = r1 ∨ r = r2 ∧ abs r > 4

theorem roots_absolute_value (p : ℝ) (r1 r2 : ℝ) :
  quadratic_roots_property p r1 r2 → ∃ r : ℝ, (r = r1 ∨ r = r2) ∧ abs r > 4 :=
sorry

end roots_absolute_value_l2121_212150


namespace cuboid_volume_l2121_212129

theorem cuboid_volume (base_area height : ℝ) (h_base_area : base_area = 14) (h_height : height = 13) : base_area * height = 182 := by
  sorry

end cuboid_volume_l2121_212129


namespace retail_price_l2121_212142

theorem retail_price (W M : ℝ) (hW : W = 20) (hM : M = 80) : W + (M / 100) * W = 36 := by
  sorry

end retail_price_l2121_212142


namespace max_halls_visitable_max_triangles_in_chain_l2121_212123

-- Definition of the problem conditions
def castle_side_length : ℝ := 100
def num_halls : ℕ := 100
def hall_side_length : ℝ := 10
def max_visitable_halls : ℕ := 91

-- Theorem statements
theorem max_halls_visitable (S : ℝ) (n : ℕ) (H : ℝ) :
  S = 100 ∧ n = 100 ∧ H = 10 → max_visitable_halls = 91 :=
by sorry

-- Definitions for subdividing an equilateral triangle and the chain of triangles
def side_divisions (k : ℕ) : ℕ := k
def total_smaller_triangles (k : ℕ) : ℕ := k^2
def max_chain_length (k : ℕ) : ℕ := k^2 - k + 1

-- Theorem statements
theorem max_triangles_in_chain (k : ℕ) :
  max_chain_length k = k^2 - k + 1 :=
by sorry

end max_halls_visitable_max_triangles_in_chain_l2121_212123


namespace cos_double_angle_l2121_212120

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 := 
sorry

end cos_double_angle_l2121_212120


namespace matrix_addition_correct_l2121_212103

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![0, 5]]
def matrixB : Matrix (Fin 2) (Fin 2) ℤ := ![![-6, 2], ![7, -10]]
def matrixC : Matrix (Fin 2) (Fin 2) ℤ := ![![-2, -1], ![7, -5]]

theorem matrix_addition_correct : matrixA + matrixB = matrixC := by
  sorry

end matrix_addition_correct_l2121_212103


namespace unique_real_solution_N_l2121_212130

theorem unique_real_solution_N (N : ℝ) :
  (∃! (x y : ℝ), 2 * x^2 + 4 * x * y + 7 * y^2 - 12 * x - 2 * y + N = 0) ↔ N = 23 :=
by
  sorry

end unique_real_solution_N_l2121_212130


namespace sufficient_not_necessary_a_eq_one_l2121_212108

noncomputable def f (a x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + a^2) - x)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x + f (-x) = 0

theorem sufficient_not_necessary_a_eq_one 
  (a : ℝ) 
  (h₁ : a = 1) 
  : is_odd_function (f a) := sorry

end sufficient_not_necessary_a_eq_one_l2121_212108


namespace coeffs_sum_eq_40_l2121_212105

theorem coeffs_sum_eq_40 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℝ)
  (h : (2 * x - 1) ^ 5 = a_0 * x ^ 5 + a_1 * x ^ 4 + a_2 * x ^ 3 + a_3 * x ^ 2 + a_4 * x + a_5) :
  a_2 + a_3 = 40 :=
sorry

end coeffs_sum_eq_40_l2121_212105


namespace no_real_solution_intersection_l2121_212186

theorem no_real_solution_intersection :
  ¬ ∃ x y : ℝ, (y = 8 / (x^3 + 4 * x + 3)) ∧ (x + y = 5) :=
by
  sorry

end no_real_solution_intersection_l2121_212186


namespace total_cloth_sold_l2121_212195

variable (commissionA commissionB salesA salesB totalWorth : ℝ)

def agentA_commission := 0.025 * salesA
def agentB_commission := 0.03 * salesB
def total_worth_of_cloth_sold := salesA + salesB

theorem total_cloth_sold 
  (hA : agentA_commission = 21) 
  (hB : agentB_commission = 27)
  : total_worth_of_cloth_sold = 1740 :=
by
  sorry

end total_cloth_sold_l2121_212195


namespace solve_eq_l2121_212173

open Real

noncomputable def solution : Set ℝ := { x | ∃ (n : ℤ), x = π / 12 + π * (n : ℝ) }

theorem solve_eq : { x : ℝ | ∃ (n : ℤ), x = π / 12 + π * (n : ℝ) } = solution := by sorry

end solve_eq_l2121_212173


namespace arithmetic_geometric_seq_l2121_212125

theorem arithmetic_geometric_seq (A B C D : ℤ) (h_arith_seq : A + C = 2 * B)
  (h_geom_seq : B * D = C * C) (h_frac : 7 * B = 3 * C) (h_posB : B > 0)
  (h_posC : C > 0) (h_posD : D > 0) : A + B + C + D = 76 :=
sorry

end arithmetic_geometric_seq_l2121_212125


namespace reduction_of_cycle_l2121_212100

noncomputable def firstReductionPercentage (P : ℝ) (x : ℝ) : Prop :=
  P * (1 - (x / 100)) * 0.8 = 0.6 * P

theorem reduction_of_cycle (P x : ℝ) (hP : 0 < P) : firstReductionPercentage P x → x = 25 :=
by
  intros h
  unfold firstReductionPercentage at h
  sorry

end reduction_of_cycle_l2121_212100


namespace werewolf_knight_is_A_l2121_212146

structure Person :=
  (isKnight : Prop)
  (isLiar : Prop)
  (isWerewolf : Prop)

variables (A B C : Person)

-- A's statement: "At least one of us is a liar."
def statementA (A B C : Person) : Prop := A.isLiar ∨ B.isLiar ∨ C.isLiar

-- B's statement: "C is a knight."
def statementB (C : Person) : Prop := C.isKnight

theorem werewolf_knight_is_A (A B C : Person) 
  (hA : statementA A B C)
  (hB : statementB C)
  (hWerewolfKnight : ∃ x : Person, x.isWerewolf ∧ x.isKnight ∧ ¬ (A ≠ x ∧ B ≠ x ∧ C ≠ x))
  : A.isWerewolf ∧ A.isKnight :=
sorry

end werewolf_knight_is_A_l2121_212146


namespace can_form_sets_l2121_212185

def clearly_defined (s : Set α) : Prop := ∀ x ∈ s, True
def not_clearly_defined (s : Set α) : Prop := ¬clearly_defined s

def cubes := {x : Type | True} -- Placeholder for the actual definition
def major_supermarkets := {x : Type | True} -- Placeholder for the actual definition
def difficult_math_problems := {x : Type | True} -- Placeholder for the actual definition
def famous_dancers := {x : Type | True} -- Placeholder for the actual definition
def products_2012 := {x : Type | True} -- Placeholder for the actual definition
def points_on_axes := {x : ℝ × ℝ | x.1 = 0 ∨ x.2 = 0}

theorem can_form_sets :
  (clearly_defined cubes) ∧
  (not_clearly_defined major_supermarkets) ∧
  (not_clearly_defined difficult_math_problems) ∧
  (not_clearly_defined famous_dancers) ∧
  (clearly_defined products_2012) ∧
  (clearly_defined points_on_axes) →
  True := 
by {
  -- Your proof goes here
  sorry
}

end can_form_sets_l2121_212185


namespace gifts_receiving_ribbon_l2121_212127

def total_ribbon := 18
def ribbon_per_gift := 2
def remaining_ribbon := 6

theorem gifts_receiving_ribbon : (total_ribbon - remaining_ribbon) / ribbon_per_gift = 6 := by
  sorry

end gifts_receiving_ribbon_l2121_212127


namespace monotonicity_and_extremum_of_f_l2121_212184

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem monotonicity_and_extremum_of_f :
  (∀ x, 1 < x → ∀ y, x < y → f x < f y) ∧
  (∀ x, 0 < x → x < 1 → ∀ y, x < y → y < 1 → f x > f y) ∧
  (f 1 = -1) :=
by
  sorry

end monotonicity_and_extremum_of_f_l2121_212184


namespace lap_time_improvement_l2121_212149

theorem lap_time_improvement (initial_laps : ℕ) (initial_time : ℕ) (current_laps : ℕ) (current_time : ℕ)
  (h1 : initial_laps = 15) (h2 : initial_time = 45) (h3 : current_laps = 18) (h4 : current_time = 42) :
  (45 / 15 - 42 / 18 : ℚ) = 2 / 3 :=
by
  sorry

end lap_time_improvement_l2121_212149


namespace rhombus_longer_diagonal_l2121_212136

theorem rhombus_longer_diagonal (a b d : ℝ) (h : a = 65) (h_d : d = 56) :
  ∃ l, l = 118 :=
by
  sorry

end rhombus_longer_diagonal_l2121_212136


namespace altitude_of_triangle_l2121_212187

theorem altitude_of_triangle (x : ℝ) (h : ℝ) 
  (h1 : x^2 = (1/2) * x * h) : h = 2 * x :=
by
  sorry

end altitude_of_triangle_l2121_212187


namespace train_length_proof_l2121_212107

-- Definitions for conditions
def jogger_speed_kmh : ℕ := 9
def train_speed_kmh : ℕ := 45
def initial_distance_ahead_m : ℕ := 280
def time_to_pass_s : ℕ := 40

-- Conversion factors
def km_per_hr_to_m_per_s (speed_kmh : ℕ) : ℕ := speed_kmh * 1000 / 3600

-- Converted speeds
def jogger_speed_m_per_s : ℕ := km_per_hr_to_m_per_s jogger_speed_kmh
def train_speed_m_per_s : ℕ := km_per_hr_to_m_per_s train_speed_kmh

-- Relative speed
def relative_speed_m_per_s : ℕ := train_speed_m_per_s - jogger_speed_m_per_s

-- Distance covered relative to the jogger
def distance_covered_relative_m : ℕ := relative_speed_m_per_s * time_to_pass_s

-- Length of the train
def length_of_train_m : ℕ := distance_covered_relative_m + initial_distance_ahead_m

-- Theorem to prove 
theorem train_length_proof : length_of_train_m = 680 := 
by
   sorry

end train_length_proof_l2121_212107


namespace find_Q_l2121_212180

variable {x P Q : ℝ}

theorem find_Q (h₁ : x + 1 / x = P) (h₂ : P = 1) : x^6 + 1 / x^6 = 2 :=
by
  sorry

end find_Q_l2121_212180


namespace time_to_chop_an_onion_is_4_minutes_l2121_212113

noncomputable def time_to_chop_pepper := 3
noncomputable def time_to_grate_cheese_per_omelet := 1
noncomputable def time_to_cook_omelet := 5
noncomputable def peppers_needed := 4
noncomputable def onions_needed := 2
noncomputable def omelets_needed := 5
noncomputable def total_time := 50

theorem time_to_chop_an_onion_is_4_minutes : 
  (total_time - (peppers_needed * time_to_chop_pepper + omelets_needed * time_to_grate_cheese_per_omelet + omelets_needed * time_to_cook_omelet)) / onions_needed = 4 := by sorry

end time_to_chop_an_onion_is_4_minutes_l2121_212113


namespace range_of_a_l2121_212143

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 5^x = a + 3) → a > -3 :=
by
  sorry

end range_of_a_l2121_212143


namespace distance_between_points_l2121_212140

noncomputable def distance (x1 y1 x2 y2 : ℝ) := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points : 
  distance (-3) (1/2) 4 (-7) = Real.sqrt 105.25 := 
by 
  sorry

end distance_between_points_l2121_212140


namespace find_a2_l2121_212196

theorem find_a2 (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + 2)
  (h_geom : (a 1) * (a 5) = (a 2) * (a 2)) : a 2 = 3 :=
by
  -- We are given the conditions and need to prove the statement.
  sorry

end find_a2_l2121_212196


namespace find_functions_satisfying_condition_l2121_212163

noncomputable def function_satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 → a * b * c * d = 1 →
  (f a + f b) * (f c + f d) = (a + b) * (c + d)

theorem find_functions_satisfying_condition :
  ∀ f : ℝ → ℝ, function_satisfies_condition f →
    (∀ x : ℝ, x > 0 → f x = x) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x) :=
sorry

end find_functions_satisfying_condition_l2121_212163


namespace probability_of_three_primes_from_30_l2121_212166

noncomputable def primes_up_to_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_three_primes_from_30 :
  ((primes_up_to_30.card.choose 3) / ((Finset.range 31).card.choose 3)) = (6 / 203) :=
by
  sorry

end probability_of_three_primes_from_30_l2121_212166


namespace even_function_value_of_a_l2121_212112

theorem even_function_value_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x * (Real.exp x + a * Real.exp (-x))) (h_even : ∀ x : ℝ, f x = f (-x)) : a = -1 := 
by
  sorry

end even_function_value_of_a_l2121_212112


namespace distinct_intersection_points_l2121_212164

theorem distinct_intersection_points : 
  ∃! (x y : ℝ), (x + 2*y = 6 ∧ x - 3*y = 2) ∨ (x + 2*y = 6 ∧ 4*x + y = 14) :=
by
  -- proof would be here
  sorry

end distinct_intersection_points_l2121_212164


namespace new_supervisor_salary_l2121_212182

-- Definitions
def average_salary_old (W : ℕ) : Prop :=
  (W + 870) / 9 = 430

def average_salary_new (W : ℕ) (S_new : ℕ) : Prop :=
  (W + S_new) / 9 = 430

-- Problem statement
theorem new_supervisor_salary (W : ℕ) (S_new : ℕ) :
  average_salary_old W →
  average_salary_new W S_new →
  S_new = 870 :=
by
  sorry

end new_supervisor_salary_l2121_212182


namespace gcd_282_470_l2121_212160

theorem gcd_282_470 : Int.gcd 282 470 = 94 :=
by
  sorry

end gcd_282_470_l2121_212160


namespace exists_strictly_increasing_sequence_l2121_212152

theorem exists_strictly_increasing_sequence 
  (N : ℕ) : 
  (∃ (t : ℕ), t^2 ≤ N ∧ N < t^2 + t) →
  (∃ (s : ℕ → ℕ), (∀ n : ℕ, s n < s (n + 1)) ∧ 
   (∃ k : ℕ, ∀ n : ℕ, s (n + 1) - s n = k) ∧
   (∀ n : ℕ, s (s n) - s (s (n - 1)) ≤ N 
      ∧ N < s (1 + s n) - s (s (n - 1)))) :=
by
  sorry

end exists_strictly_increasing_sequence_l2121_212152


namespace find_sixth_term_of_geometric_sequence_l2121_212193

noncomputable def common_ratio (a b : ℚ) : ℚ := b / a

noncomputable def geometric_sequence_term (a r : ℚ) (k : ℕ) : ℚ := a * (r ^ (k - 1))

theorem find_sixth_term_of_geometric_sequence :
  geometric_sequence_term 5 (common_ratio 5 1.25) 6 = 5 / 1024 :=
by
  sorry

end find_sixth_term_of_geometric_sequence_l2121_212193


namespace minimum_workers_needed_l2121_212124

theorem minimum_workers_needed 
  (total_days : ℕ)
  (completed_days : ℕ)
  (initial_workers : ℕ)
  (fraction_completed : ℚ)
  (remaining_fraction : ℚ)
  (remaining_days : ℕ)
  (rate_completed_per_day : ℚ)
  (required_rate_per_day : ℚ)
  (equal_productivity : Prop) 
  : initial_workers = 10 :=
by
  -- Definitions
  let total_days := 40
  let completed_days := 10
  let initial_workers := 10
  let fraction_completed := 1 / 4
  let remaining_fraction := 1 - fraction_completed
  let remaining_days := total_days - completed_days
  let rate_completed_per_day := fraction_completed / completed_days
  let required_rate_per_day := remaining_fraction / remaining_days
  let equal_productivity := true

  -- Sorry is used to skip the proof
  sorry

end minimum_workers_needed_l2121_212124


namespace intersection_M_N_l2121_212132

def M : Set ℝ :=
  {x | |x| ≤ 2}

def N : Set ℝ :=
  {x | Real.exp x ≥ 1}

theorem intersection_M_N :
  (M ∩ N) = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l2121_212132


namespace ratio_of_jars_to_pots_l2121_212134

theorem ratio_of_jars_to_pots 
  (jars : ℕ)
  (pots : ℕ)
  (k : ℕ)
  (marbles_total : ℕ)
  (h1 : jars = 16)
  (h2 : jars = k * pots)
  (h3 : ∀ j, j = 5)
  (h4 : ∀ p, p = 15)
  (h5 : marbles_total = 200) :
  (jars / pots = 2) :=
by
  sorry

end ratio_of_jars_to_pots_l2121_212134


namespace cells_that_remain_open_l2121_212162

/-- A cell q remains open after iterative toggling if and only if it is a perfect square. -/
theorem cells_that_remain_open (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k ^ 2 = n) ↔ 
  (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ m : ℕ, i = m ^ 2)) := 
sorry

end cells_that_remain_open_l2121_212162


namespace stone_travel_distance_l2121_212148

/-- Define the radii --/
def radius_fountain := 15
def radius_stone := 3

/-- Prove the distance the stone needs to travel along the fountain's edge --/
theorem stone_travel_distance :
  let circumference_fountain := 2 * Real.pi * ↑radius_fountain
  let circumference_stone := 2 * Real.pi * ↑radius_stone
  let distance_traveled := circumference_stone
  distance_traveled = 6 * Real.pi := by
  -- Placeholder for proof, based on conditions given
  sorry

end stone_travel_distance_l2121_212148


namespace compound_interest_example_l2121_212171

theorem compound_interest_example :
  let P := 5000
  let r := 0.08
  let n := 4
  let t := 0.5
  let A := P * (1 + r / n) ^ (n * t)
  A = 5202 :=
by
  sorry

end compound_interest_example_l2121_212171


namespace hoseok_multiplied_number_l2121_212194

theorem hoseok_multiplied_number (n : ℕ) (h : 11 * n = 99) : n = 9 := 
sorry

end hoseok_multiplied_number_l2121_212194


namespace forest_coverage_2009_min_annual_growth_rate_l2121_212157

variables (a : ℝ)

-- Conditions
def initially_forest_coverage (a : ℝ) := a
def annual_natural_growth_rate := 0.02

-- Questions reformulated:
-- Part 1: Prove the forest coverage at the end of 2009
theorem forest_coverage_2009 : (∃ a : ℝ, (y : ℝ) = a * (1 + 0.02)^5 ∧ y = 1.104 * a) :=
by sorry

-- Part 2: Prove the minimum annual average growth rate by 2014
theorem min_annual_growth_rate : (∀ p : ℝ, (a : ℝ) * (1 + p)^10 ≥ 2 * a → p ≥ 0.072) :=
by sorry

end forest_coverage_2009_min_annual_growth_rate_l2121_212157


namespace yuna_initial_marbles_l2121_212178

theorem yuna_initial_marbles (M : ℕ) :
  (M - 12 + 5) / 2 + 3 = 17 → M = 35 := by
  sorry

end yuna_initial_marbles_l2121_212178


namespace monotonic_decreasing_interval_l2121_212128

noncomputable def y (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem monotonic_decreasing_interval :
  {x : ℝ | (∃ y', y' = 3 * x^2 - 3 ∧ y' < 0)} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end monotonic_decreasing_interval_l2121_212128


namespace milton_sold_15_pies_l2121_212183

theorem milton_sold_15_pies
  (apple_pie_slices_per_pie : ℕ) (peach_pie_slices_per_pie : ℕ)
  (ordered_apple_pie_slices : ℕ) (ordered_peach_pie_slices : ℕ)
  (h1 : apple_pie_slices_per_pie = 8) (h2 : peach_pie_slices_per_pie = 6)
  (h3 : ordered_apple_pie_slices = 56) (h4 : ordered_peach_pie_slices = 48) :
  (ordered_apple_pie_slices / apple_pie_slices_per_pie) + (ordered_peach_pie_slices / peach_pie_slices_per_pie) = 15 := 
by
  sorry

end milton_sold_15_pies_l2121_212183


namespace exponent_multiplication_identity_l2121_212133

theorem exponent_multiplication_identity : 2^4 * 3^2 * 5^2 * 7 = 6300 := sorry

end exponent_multiplication_identity_l2121_212133


namespace days_to_complete_work_together_l2121_212188

theorem days_to_complete_work_together :
  (20 * 35) / (20 + 35) = 140 / 11 :=
by
  sorry

end days_to_complete_work_together_l2121_212188


namespace number_of_machines_l2121_212111

theorem number_of_machines (X : ℕ)
  (h1 : 20 = (10 : ℝ) * X * 0.4) :
  X = 5 := sorry

end number_of_machines_l2121_212111


namespace find_son_age_l2121_212170

theorem find_son_age (F S : ℕ) (h1 : F + S = 55)
  (h2 : ∃ Y, S + Y = F ∧ (F + Y) + (S + Y) = 93)
  (h3 : F = 18 ∨ S = 18) : S = 18 :=
by
  sorry  -- Proof to be filled in

end find_son_age_l2121_212170


namespace tunnel_length_correct_l2121_212126

noncomputable def length_of_tunnel
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time_min : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let crossing_time_s := crossing_time_min * 60
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem tunnel_length_correct :
  length_of_tunnel 800 78 1 = 500.2 :=
by
  -- The proof will be filled later.
  sorry

end tunnel_length_correct_l2121_212126


namespace percent_value_in_quarters_l2121_212175

theorem percent_value_in_quarters
  (num_dimes num_quarters num_nickels : ℕ)
  (value_dime value_quarter value_nickel : ℕ)
  (h_dimes : num_dimes = 70)
  (h_quarters : num_quarters = 30)
  (h_nickels : num_nickels = 40)
  (h_value_dime : value_dime = 10)
  (h_value_quarter : value_quarter = 25)
  (h_value_nickel : value_nickel = 5) :
  ((num_quarters * value_quarter : ℕ) * 100 : ℚ) / 
  (num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel) = 45.45 :=
by
  sorry

end percent_value_in_quarters_l2121_212175


namespace tiffany_bags_l2121_212137

/-!
## Problem Statement
Tiffany was collecting cans for recycling. On Monday she had some bags of cans. 
She found 3 bags of cans on the next day and 7 bags of cans the day after that. 
She had altogether 20 bags of cans. Prove that the number of bags of cans she had on Monday is 10.
-/

theorem tiffany_bags (M : ℕ) (h1 : M + 3 + 7 = 20) : M = 10 :=
by {
  sorry
}

end tiffany_bags_l2121_212137


namespace value_of_f_neg2_l2121_212106

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem value_of_f_neg2 (a b c : ℝ) (h1 : f a b c 5 + f a b c (-5) = 6) (h2 : f a b c 2 = 8) :
  f a b c (-2) = -2 := by
  sorry

end value_of_f_neg2_l2121_212106


namespace geometric_sequence_common_ratio_l2121_212101

theorem geometric_sequence_common_ratio (a_1 q : ℝ) (hne1 : q ≠ 1)
  (h : (a_1 * (1 - q^4) / (1 - q)) = 5 * (a_1 * (1 - q^2) / (1 - q))) :
  q = -1 ∨ q = 2 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l2121_212101


namespace inequality_solution_l2121_212138

theorem inequality_solution (x : ℝ) :
  4 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 → x ∈ Set.Ioc (5 / 2 : ℝ) (20 / 7 : ℝ) := by
  sorry

end inequality_solution_l2121_212138


namespace cuboid_edge_length_l2121_212167

theorem cuboid_edge_length (x : ℝ) (h1 : 5 * 6 * x = 120) : x = 4 :=
by
  sorry

end cuboid_edge_length_l2121_212167


namespace new_volume_of_cylinder_l2121_212121

theorem new_volume_of_cylinder
  (r h : ℝ) -- original radius and height
  (V : ℝ) -- original volume
  (h_volume : V = π * r^2 * h) -- volume formula for the original cylinder
  (new_radius : ℝ := 3 * r) -- new radius is three times the original radius
  (new_volume : ℝ) -- new volume to be determined
  (h_original_volume : V = 10) -- original volume equals 10 cubic feet
  : new_volume = 9 * V := -- new volume should be 9 times the original volume
by
  sorry

end new_volume_of_cylinder_l2121_212121


namespace sum_in_range_l2121_212197

noncomputable def mixed_number_sum : ℚ :=
  3 + 1/8 + 4 + 3/7 + 6 + 2/21

theorem sum_in_range : 13.5 ≤ mixed_number_sum ∧ mixed_number_sum < 14 := by
  sorry

end sum_in_range_l2121_212197


namespace vertex_of_parabola_find_shift_m_l2121_212144

-- Problem 1: Vertex of the given parabola
theorem vertex_of_parabola : 
  ∃ x y: ℝ, (y = 2 * x^2 + 4 * x - 6) ∧ (x, y) = (-1, -8) := 
by
  -- Proof goes here
  sorry

-- Problem 2: Finding the shift m
theorem find_shift_m (m : ℝ) (h : m > 0) : 
  (∀ x (hx : (x = (x + m)) ∧ (2 * x^2 + 4 * x - 6 = 0)), x = 1 ∨ x = -3) ∧ 
  ((-3 + m) = 0) → m = 3 :=
by
  -- Proof goes here
  sorry

end vertex_of_parabola_find_shift_m_l2121_212144


namespace triple_supplementary_angle_l2121_212109

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l2121_212109


namespace min_area_triangle_ABC_l2121_212158

def point (α : Type*) := (α × α)

def area_of_triangle (A B C : point ℤ) : ℚ :=
  (1/2 : ℚ) * abs (36 * (C.snd) - 15 * (C.fst))

theorem min_area_triangle_ABC :
  ∃ (C : point ℤ), area_of_triangle (0, 0) (36, 15) C = 3 / 2 :=
by
  sorry

end min_area_triangle_ABC_l2121_212158


namespace school_adding_seats_l2121_212115

theorem school_adding_seats (row_seats : ℕ) (seat_cost : ℕ) (discount_rate : ℝ) (total_cost : ℕ) (n : ℕ) 
                         (total_seats : ℕ) (discounted_seat_cost : ℕ)
                         (total_groups : ℕ) (rows : ℕ) :
  row_seats = 8 →
  seat_cost = 30 →
  discount_rate = 0.10 →
  total_cost = 1080 →
  discounted_seat_cost = seat_cost * (1 - discount_rate) →
  total_seats = total_cost / discounted_seat_cost →
  total_groups = total_seats / 10 →
  rows = total_seats / row_seats →
  rows = 5 :=
by
  intros hrowseats hseatcost hdiscountrate htotalcost hdiscountedseatcost htotalseats htotalgroups hrows
  sorry

end school_adding_seats_l2121_212115


namespace children_being_catered_l2121_212119

-- Define the total meal units available
def meal_units_for_adults : ℕ := 70
def meal_units_for_children : ℕ := 90
def meals_eaten_by_adults : ℕ := 14
def remaining_meal_units : ℕ := meal_units_for_adults - meals_eaten_by_adults

theorem children_being_catered :
  (remaining_meal_units * meal_units_for_children) / meal_units_for_adults = 72 := by
{
  sorry
}

end children_being_catered_l2121_212119


namespace b7_in_form_l2121_212172

theorem b7_in_form (a : ℕ → ℚ) (b : ℕ → ℚ) : 
  a 0 = 3 → 
  b 0 = 5 → 
  (∀ n : ℕ, a (n + 1) = (a n)^2 / (b n)) → 
  (∀ n : ℕ, b (n + 1) = (b n)^2 / (a n)) → 
  b 7 = (5^50 : ℚ) / (3^41 : ℚ) := 
by 
  intros h1 h2 h3 h4 
  sorry

end b7_in_form_l2121_212172


namespace inequality_solution_set_l2121_212110

theorem inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - 5 * x + b > 0) :
  ∀ x : ℝ, x < -1/3 ∨ x > 1/2 ↔ b * x^2 - 5 * x + a > 0 :=
sorry

end inequality_solution_set_l2121_212110
