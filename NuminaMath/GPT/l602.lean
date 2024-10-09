import Mathlib

namespace op_4_3_equals_23_l602_60292

def op (a b : ℕ) : ℕ := a ^ 2 + a * b + a - b ^ 2

theorem op_4_3_equals_23 : op 4 3 = 23 := by
  -- Proof steps would go here
  sorry

end op_4_3_equals_23_l602_60292


namespace fraction_computation_l602_60295

theorem fraction_computation :
  ((11^4 + 324) * (23^4 + 324) * (35^4 + 324) * (47^4 + 324) * (59^4 + 324)) / 
  ((5^4 + 324) * (17^4 + 324) * (29^4 + 324) * (41^4 + 324) * (53^4 + 324)) = 295.615 := 
sorry

end fraction_computation_l602_60295


namespace range_of_m_l602_60232

theorem range_of_m (x y m : ℝ) (h1 : x - 2 * y = 1) (h2 : 2 * x + y = 4 * m) (h3 : x + 3 * y < 6) : m < 7 / 4 :=
sorry

end range_of_m_l602_60232


namespace cannot_cut_out_rect_l602_60237

noncomputable def square_area : ℝ := 400
noncomputable def rect_area : ℝ := 300
noncomputable def length_to_width_ratio : ℝ × ℝ := (3, 2)

theorem cannot_cut_out_rect (h1: square_area = 400) (h2: rect_area = 300) (h3: length_to_width_ratio = (3, 2)) : 
  false := sorry

end cannot_cut_out_rect_l602_60237


namespace number_of_unit_fraction_pairs_l602_60204

/-- 
 The number of ways that 1/2007 can be expressed as the sum of two distinct positive unit fractions is 7.
-/
theorem number_of_unit_fraction_pairs : 
  ∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 ≠ p.2 ∧ (1 : ℚ) / 2007 = 1 / ↑p.1 + 1 / ↑p.2) ∧ 
    pairs.card = 7 :=
sorry

end number_of_unit_fraction_pairs_l602_60204


namespace parabola_focus_coords_l602_60268

theorem parabola_focus_coords :
  ∀ (x y : ℝ), y^2 = -4 * x → (x, y) = (-1, 0) :=
by
  intros x y h
  sorry

end parabola_focus_coords_l602_60268


namespace smallest_solution_is_9_l602_60294

noncomputable def smallest_positive_solution (x : ℝ) : Prop :=
  (3*x / (x - 3) + (3*x^2 - 45) / (x + 3) = 14) ∧ (x > 3) ∧ (∀ y : ℝ, (3*y / (y - 3) + (3*y^2 - 45) / (y + 3) = 14) → (y > 3) → (y ≥ 9))

theorem smallest_solution_is_9 : ∃ x : ℝ, smallest_positive_solution x ∧ x = 9 :=
by
  exists 9
  have : smallest_positive_solution 9 := sorry
  exact ⟨this, rfl⟩

end smallest_solution_is_9_l602_60294


namespace width_decrease_l602_60288

-- Given conditions and known values
variable (L W : ℝ) -- original length and width
variable (P : ℝ)   -- percentage decrease in width

-- The known condition for the area comparison
axiom area_condition : 1.4 * (L * (W * (1 - P / 100))) = 1.1199999999999999 * (L * W)

-- The property we want to prove
theorem width_decrease (L W: ℝ) (h : L > 0) (h1 : W > 0) :
  P = 20 := 
by
  sorry

end width_decrease_l602_60288


namespace max_value_of_y_l602_60202

theorem max_value_of_y (x : ℝ) (h₁ : 0 < x) (h₂ : x < 4) : 
  ∃ y : ℝ, (y = x * (8 - 2 * x)) ∧ (∀ z : ℝ, z = x * (8 - 2 * x) → z ≤ 8) :=
sorry

end max_value_of_y_l602_60202


namespace trig_identity_example_l602_60231

theorem trig_identity_example (α : ℝ) (h : Real.tan α = 1 / 2) : 
  (4 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 2 / 3 :=
by
  sorry

end trig_identity_example_l602_60231


namespace can_form_triangle_l602_60251

theorem can_form_triangle (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_condition : c^2 ≤ 4 * a * b) : 
  a + b > c ∧ a + c > b ∧ b + c > a := 
sorry

end can_form_triangle_l602_60251


namespace range_abs_plus_one_l602_60266

 theorem range_abs_plus_one : 
   ∀ y : ℝ, (∃ x : ℝ, y = |x| + 1) ↔ y ≥ 1 := 
 by
   sorry
 
end range_abs_plus_one_l602_60266


namespace infinite_geometric_series_l602_60235

theorem infinite_geometric_series
  (p q r : ℝ)
  (h_series : ∑' n : ℕ, p / q^(n+1) = 9) :
  (∑' n : ℕ, p / (p + r)^(n+1)) = (9 * (q - 1)) / (9 * q + r - 10) :=
by 
  sorry

end infinite_geometric_series_l602_60235


namespace possible_values_of_expression_l602_60222

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∃ v : ℝ, v = (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|) ∧ 
            (v = 5 ∨ v = 1 ∨ v = -3 ∨ v = -5)) :=
by
  sorry

end possible_values_of_expression_l602_60222


namespace number_of_sides_of_polygon_l602_60287

theorem number_of_sides_of_polygon :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 2 * n + 7 ∧ n = 8 := 
by
  sorry

end number_of_sides_of_polygon_l602_60287


namespace cost_per_liter_of_gas_today_l602_60284

-- Definition of the conditions
def oil_price_rollback : ℝ := 0.4
def liters_today : ℝ := 10
def liters_friday : ℝ := 25
def total_liters := liters_today + liters_friday
def total_cost : ℝ := 39

-- The theorem to prove
theorem cost_per_liter_of_gas_today (C : ℝ) :
  (liters_today * C) + (liters_friday * (C - oil_price_rollback)) = total_cost →
  C = 1.4 := 
by 
  sorry

end cost_per_liter_of_gas_today_l602_60284


namespace chemical_transformations_correct_l602_60215

def ethylbenzene : String := "C6H5CH2CH3"
def brominate (A : String) : String := "C6H5CH(Br)CH3"
def hydrolyze (B : String) : String := "C6H5CH(OH)CH3"
def dehydrate (C : String) : String := "C6H5CH=CH2"
def oxidize (D : String) : String := "C6H5COOH"
def brominate_with_catalyst (E : String) : String := "m-C6H4(Br)COOH"

def sequence_of_transformations : Prop :=
  ethylbenzene = "C6H5CH2CH3" ∧
  brominate ethylbenzene = "C6H5CH(Br)CH3" ∧
  hydrolyze (brominate ethylbenzene) = "C6H5CH(OH)CH3" ∧
  dehydrate (hydrolyze (brominate ethylbenzene)) = "C6H5CH=CH2" ∧
  oxidize (dehydrate (hydrolyze (brominate ethylbenzene))) = "C6H5COOH" ∧
  brominate_with_catalyst (oxidize (dehydrate (hydrolyze (brominate ethylbenzene)))) = "m-C6H4(Br)COOH"

theorem chemical_transformations_correct : sequence_of_transformations :=
by
  -- proof would go here
  sorry

end chemical_transformations_correct_l602_60215


namespace equation_of_line_l602_60206

theorem equation_of_line (P : ℝ × ℝ) (m : ℝ) : 
  P = (3, 3) → m = 2 * 1 → ∃ b : ℝ, ∀ x : ℝ, P.2 = m * (x - P.1) + b ↔ y = 2 * x - 3 := 
by {
  sorry
}

end equation_of_line_l602_60206


namespace foci_distance_l602_60236

variable (x y : ℝ)

def ellipse_eq : Prop := (x^2 / 45) + (y^2 / 5) = 9

theorem foci_distance : ellipse_eq x y → (distance_between_foci : ℝ) = 12 * Real.sqrt 10 :=
by
  sorry

end foci_distance_l602_60236


namespace value_expression_l602_60240

theorem value_expression (p q : ℚ) (h : p / q = 4 / 5) : 18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by 
  sorry

end value_expression_l602_60240


namespace find_sum_x_y_l602_60298

theorem find_sum_x_y (x y : ℝ) 
  (h1 : x^3 - 3 * x^2 + 2026 * x = 2023)
  (h2 : y^3 + 6 * y^2 + 2035 * y = -4053) : 
  x + y = -1 := 
sorry

end find_sum_x_y_l602_60298


namespace triangle_inequality_l602_60252

variable (a b c : ℝ)

theorem triangle_inequality (h₁ : a + b + c = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : c > 0) (h₅ : a + b > c) (h₆ : b + c > a) (h₇ : c + a > b) : 
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 := 
sorry

end triangle_inequality_l602_60252


namespace factor_expression_l602_60244

variable (x : ℝ)

theorem factor_expression : 
  (10 * x^3 + 50 * x^2 - 5) - (-5 * x^3 + 15 * x^2 - 5) = 5 * x^2 * (3 * x + 7) := 
by 
  sorry

end factor_expression_l602_60244


namespace shooting_accuracy_l602_60279

theorem shooting_accuracy 
  (P_A : ℚ) 
  (P_AB : ℚ) 
  (h1 : P_A = 9 / 10) 
  (h2 : P_AB = 1 / 2) 
  : P_AB / P_A = 5 / 9 := 
by
  sorry

end shooting_accuracy_l602_60279


namespace triangle_area_is_9_point_5_l602_60230

def Point : Type := (ℝ × ℝ)

def A : Point := (0, 1)
def B : Point := (4, 0)
def C : Point := (3, 5)

noncomputable def areaOfTriangle (A B C : Point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_is_9_point_5 :
  areaOfTriangle A B C = 9.5 :=
by
  sorry

end triangle_area_is_9_point_5_l602_60230


namespace expression_divisible_by_7_l602_60262

theorem expression_divisible_by_7 (k : ℕ) : 
  (∀ n : ℕ, n > 0 → ∃ m : ℤ, 3^(6*n-1) - k * 2^(3*n-2) + 1 = 7 * m) ↔ ∃ m' : ℤ, k = 7 * m' + 3 := 
by
  sorry

end expression_divisible_by_7_l602_60262


namespace find_number_of_women_in_first_group_l602_60208

variables (W : ℕ)

-- Conditions
def women_coloring_rate := 10
def total_cloth_colored_in_3_days := 180
def women_in_first_group := total_cloth_colored_in_3_days / 3

theorem find_number_of_women_in_first_group
  (h1 : 5 * women_coloring_rate * 4 = 200)
  (h2 : W * women_coloring_rate = women_in_first_group) :
  W = 6 :=
by
  sorry

end find_number_of_women_in_first_group_l602_60208


namespace find_x_plus_y_l602_60218

theorem find_x_plus_y (x y : ℝ) (hx : abs x - x + y = 6) (hy : x + abs y + y = 16) : x + y = 10 :=
sorry

end find_x_plus_y_l602_60218


namespace nth_equation_l602_60276

theorem nth_equation (n : ℕ) : (2 * n + 1)^2 - 1 = 4 * n * (n + 1) := 
by
  sorry

end nth_equation_l602_60276


namespace intersection_A_B_when_m_eq_2_range_of_m_for_p_implies_q_l602_60261

noncomputable def A := {x : ℝ | -4 < x ∧ x < 2}
noncomputable def B (m : ℝ) := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem intersection_A_B_when_m_eq_2 : (A ∩ B 2) = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

theorem range_of_m_for_p_implies_q : {m : ℝ | m ≥ 5} = {m : ℝ | ∀ x, ((x^2 + 2 * x - 8 < 0) → ((x - 1 + m) * (x - 1 - m) ≤ 0)) ∧ ¬((x - 1 + m) * (x - 1 - m) ≤ 0 → (x^2 + 2 * x - 8 < 0))} :=
by
  sorry

end intersection_A_B_when_m_eq_2_range_of_m_for_p_implies_q_l602_60261


namespace pipe_R_fill_time_l602_60293

theorem pipe_R_fill_time (P_rate Q_rate combined_rate : ℝ) (hP : P_rate = 1 / 2) (hQ : Q_rate = 1 / 4)
  (h_combined : combined_rate = 1 / 1.2) : (∃ R_rate : ℝ, R_rate = 1 / 12) :=
by
  sorry

end pipe_R_fill_time_l602_60293


namespace smallest_n_divisible_by_24_and_864_l602_60289

theorem smallest_n_divisible_by_24_and_864 :
  ∃ n : ℕ, (0 < n) ∧ (24 ∣ n^2) ∧ (864 ∣ n^3) ∧ (∀ m : ℕ, (0 < m) → (24 ∣ m^2) → (864 ∣ m^3) → (n ≤ m)) :=
sorry

end smallest_n_divisible_by_24_and_864_l602_60289


namespace equivalence_of_statements_l602_60267

variable (S M : Prop)

theorem equivalence_of_statements : 
  (S → M) ↔ ((¬M → ¬S) ∧ (¬S ∨ M)) :=
by
  sorry

end equivalence_of_statements_l602_60267


namespace largest_k_sum_of_consecutive_odds_l602_60285

theorem largest_k_sum_of_consecutive_odds (k m : ℕ) (h1 : k * (2 * m + k) = 2^15) : k ≤ 128 :=
by {
  sorry
}

end largest_k_sum_of_consecutive_odds_l602_60285


namespace max_value_q_l602_60242

noncomputable def q (A M C : ℕ) : ℕ :=
  A * M * C + A * M + M * C + C * A + A + M + C

theorem max_value_q : ∀ A M C : ℕ, A + M + C = 15 → q A M C ≤ 215 :=
by 
  sorry

end max_value_q_l602_60242


namespace equal_distribution_l602_60272

def earnings : List ℕ := [30, 35, 45, 55, 65]

def total_earnings : ℕ := earnings.sum

def equal_share (total: ℕ) : ℕ := total / earnings.length

def redistribution_amount (earner: ℕ) (equal: ℕ) : ℕ := earner - equal

theorem equal_distribution :
  redistribution_amount 65 (equal_share total_earnings) = 19 :=
by
  sorry

end equal_distribution_l602_60272


namespace exists_lcm_lt_l602_60273

theorem exists_lcm_lt (p q : ℕ) (hpq_coprime : Nat.gcd p q = 1) (hp_gt_one : p > 1) (hq_gt_one : q > 1) (hpq_diff_gt_one : (p < q ∧ q - p > 1) ∨ (p > q ∧ p - q > 1)) :
  ∃ n : ℕ, Nat.lcm (p + n) (q + n) < Nat.lcm p q := by
  sorry

end exists_lcm_lt_l602_60273


namespace geometric_sequence_sum_inv_l602_60234

theorem geometric_sequence_sum_inv
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 1 + a 3 + a 5 = 14) :
  (1 / a 1) + (1 / a 3) + (1 / a 5) = 7 / 8 :=
by
  sorry

end geometric_sequence_sum_inv_l602_60234


namespace root_expression_value_l602_60201

theorem root_expression_value (p m n : ℝ) 
  (h1 : m^2 + (p - 2) * m + 1 = 0) 
  (h2 : n^2 + (p - 2) * n + 1 = 0) : 
  (m^2 + p * m + 1) * (n^2 + p * n + 1) - 2 = 2 :=
by
  sorry

end root_expression_value_l602_60201


namespace sheryll_paid_total_l602_60274

-- Variables/conditions
variables (cost_per_book : ℝ) (num_books : ℕ) (discount_per_book : ℝ)

-- Given conditions
def assumption1 : cost_per_book = 5 := by sorry
def assumption2 : num_books = 10 := by sorry
def assumption3 : discount_per_book = 0.5 := by sorry

-- Theorem statement
theorem sheryll_paid_total : cost_per_book = 5 → num_books = 10 → discount_per_book = 0.5 → 
  (cost_per_book - discount_per_book) * num_books = 45 := by
  sorry

end sheryll_paid_total_l602_60274


namespace sphere_touches_pyramid_edges_l602_60224

theorem sphere_touches_pyramid_edges :
  ∃ (KL : ℝ), 
  ∃ (K L M N : ℝ) (MN LN NK : ℝ) (AC: ℝ) (BC: ℝ), 
  MN = 7 ∧ 
  NK = 5 ∧ 
  LN = 2 * Real.sqrt 29 ∧ 
  KL = L ∧ 
  KL = M ∧ 
  KL = 9 :=
sorry

end sphere_touches_pyramid_edges_l602_60224


namespace num_seven_digit_palindromes_l602_60241

theorem num_seven_digit_palindromes : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices = 9000 :=
by
  sorry

end num_seven_digit_palindromes_l602_60241


namespace problem1_l602_60280

theorem problem1 :
  (2021 - Real.pi)^0 + (Real.sqrt 3 - 1) - 2 + (2 * Real.sqrt 3) = 3 * Real.sqrt 3 - 2 :=
by
  sorry

end problem1_l602_60280


namespace lcm_of_lap_times_l602_60203

theorem lcm_of_lap_times :
  Nat.lcm (Nat.lcm 5 8) 10 = 40 := by
  sorry

end lcm_of_lap_times_l602_60203


namespace suff_but_not_nec_l602_60270

-- Definition of proposition p
def p (m : ℝ) : Prop := m = -1

-- Definition of proposition q
def q (m : ℝ) : Prop := 
  let line1 := fun (x y : ℝ) => x - y = 0
  let line2 := fun (x y : ℝ) => x + (m^2) * y = 0
  ∀ (x1 y1 x2 y2 : ℝ), line1 x1 y1 → line2 x2 y2 → (x1 = x2 → y1 = -y2)

-- The proof problem
theorem suff_but_not_nec (m : ℝ) : p m → q m ∧ (q m → m = -1 ∨ m = 1) :=
sorry

end suff_but_not_nec_l602_60270


namespace total_amount_correct_l602_60254

-- Define the prices of jeans and tees
def price_jean : ℕ := 11
def price_tee : ℕ := 8

-- Define the quantities sold
def quantity_jeans_sold : ℕ := 4
def quantity_tees_sold : ℕ := 7

-- Calculate the total amount earned
def total_amount : ℕ := (price_jean * quantity_jeans_sold) + (price_tee * quantity_tees_sold)

-- Now, we state and prove the theorem
theorem total_amount_correct : total_amount = 100 :=
by
  -- Here we assert the correctness of the calculation
  sorry

end total_amount_correct_l602_60254


namespace find_number_l602_60205

theorem find_number (x : ℕ) (h : 5 * x = 100) : x = 20 :=
by
  sorry

end find_number_l602_60205


namespace exponential_fraction_l602_60282

theorem exponential_fraction :
  (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5 / 3 := 
by
  sorry

end exponential_fraction_l602_60282


namespace range_of_m_l602_60257

variable (x m : ℝ)

theorem range_of_m (h1 : ∀ x : ℝ, 2 * x^2 - 2 * m * x + m < 0) 
    (h2 : ∃ a b : ℤ, a ≠ b ∧ ∀ x : ℝ, (a < x ∧ x < b) → 2 * x^2 - 2 * m * x + m < 0): 
    -8 / 5 ≤ m ∧ m < -2 / 3 ∨ 8 / 3 < m ∧ m ≤ 18 / 5 :=
sorry

end range_of_m_l602_60257


namespace problem1_problem2_l602_60277

section
variable {α : Real}
variable (tan_α : Real)
variable (sin_α cos_α : Real)

def trigonometric_identities (tan_α sin_α cos_α : Real) : Prop :=
  tan_α = 2 ∧ sin_α = tan_α * cos_α

theorem problem1 (h : trigonometric_identities tan_α sin_α cos_α) :
  (4 * sin_α - 2 * cos_α) / (5 * cos_α + 3 * sin_α) = 6 / 11 := by
  sorry

theorem problem2 (h : trigonometric_identities tan_α sin_α cos_α) :
  (1 / 4 * sin_α^2 + 1 / 3 * sin_α * cos_α + 1 / 2 * cos_α^2) = 13 / 30 := by
  sorry
end

end problem1_problem2_l602_60277


namespace t_le_s_l602_60226

theorem t_le_s (a b : ℝ) (t s : ℝ) (h1 : t = a + 2 * b) (h2 : s = a + b^2 + 1) : t ≤ s :=
by
  sorry

end t_le_s_l602_60226


namespace squirrel_acorns_l602_60219

theorem squirrel_acorns :
  ∃ (c s r : ℕ), (4 * c = 5 * s) ∧ (3 * r = 4 * c) ∧ (r = s + 3) ∧ (5 * s = 40) :=
by
  sorry

end squirrel_acorns_l602_60219


namespace trajectory_equation_l602_60259

theorem trajectory_equation (m x y : ℝ) (a b : ℝ × ℝ)
  (ha : a = (m * x, y + 1))
  (hb : b = (x, y - 1))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  m * x^2 + y^2 = 1 :=
sorry

end trajectory_equation_l602_60259


namespace intersection_eq_one_l602_60209

def M : Set ℕ := {0, 1}
def N : Set ℕ := {y | ∃ x ∈ M, y = x^2 + 1}

theorem intersection_eq_one : M ∩ N = {1} := 
by
  sorry

end intersection_eq_one_l602_60209


namespace Harkamal_total_payment_l602_60238

theorem Harkamal_total_payment :
  let cost_grapes := 10 * 70
  let cost_mangoes := 9 * 55
  let cost_apples := 12 * 80
  let cost_papayas := 7 * 45
  let cost_oranges := 15 * 30
  let cost_bananas := 5 * 25
  cost_grapes + cost_mangoes + cost_apples + cost_papayas + cost_oranges + cost_bananas = 3045 := by
  sorry

end Harkamal_total_payment_l602_60238


namespace tank_capacity_l602_60239

theorem tank_capacity (C : ℝ) (h_leak : ∀ t, t = 6 -> C / 6 = C / t)
    (h_inlet : ∀ r, r = 240 -> r = 4 * 60)
    (h_net : ∀ t, t = 8 -> 240 - C / 6 = C / 8) :
    C = 5760 / 7 := 
by 
  sorry

end tank_capacity_l602_60239


namespace repeating_decimal_product_l602_60296

noncomputable def x : ℚ := 1 / 33
noncomputable def y : ℚ := 1 / 3

theorem repeating_decimal_product :
  (x * y) = 1 / 99 :=
by
  -- Definitions of x and y
  sorry

end repeating_decimal_product_l602_60296


namespace sum_of_digits_of_N_l602_60250

theorem sum_of_digits_of_N (N : ℕ) (hN : N * (N + 1) / 2 = 3003) :
  (Nat.digits 10 N).sum = 14 :=
sorry

end sum_of_digits_of_N_l602_60250


namespace bronson_yellow_leaves_l602_60212

-- Bronson collects 12 leaves on Thursday
def leaves_thursday : ℕ := 12

-- Bronson collects 13 leaves on Friday
def leaves_friday : ℕ := 13

-- 20% of the leaves are Brown (as a fraction)
def percent_brown : ℚ := 0.2

-- 20% of the leaves are Green (as a fraction)
def percent_green : ℚ := 0.2

theorem bronson_yellow_leaves : 
  (leaves_thursday + leaves_friday) * (1 - percent_brown - percent_green) = 15 := by
sorry

end bronson_yellow_leaves_l602_60212


namespace downloaded_data_l602_60211

/-- 
  Mason is trying to download a 880 MB game to his phone. After downloading some amount, his Internet
  connection slows to 3 MB/minute. It will take him 190 more minutes to download the game. Prove that 
  Mason has downloaded 310 MB before his connection slowed down. 
-/
theorem downloaded_data (total_size : ℕ) (speed : ℕ) (time_remaining : ℕ) (remaining_data : ℕ) (downloaded : ℕ) :
  total_size = 880 ∧
  speed = 3 ∧
  time_remaining = 190 ∧
  remaining_data = speed * time_remaining ∧
  downloaded = total_size - remaining_data →
  downloaded = 310 := 
by 
  sorry

end downloaded_data_l602_60211


namespace factorize_expression_l602_60297

theorem factorize_expression : (x^2 + 9)^2 - 36*x^2 = (x + 3)^2 * (x - 3)^2 := 
by 
  sorry

end factorize_expression_l602_60297


namespace initial_ratio_l602_60207

variable (A B : ℕ) (a b : ℕ)
variable (h1 : B = 6)
variable (h2 : (A + 2) / (B + 2) = 3 / 2)

theorem initial_ratio (A B : ℕ) (h1 : B = 6) (h2 : (A + 2) / (B + 2) = 3 / 2) : A / B = 5 / 3 := 
by 
    sorry

end initial_ratio_l602_60207


namespace number_of_men_in_first_group_l602_60260

theorem number_of_men_in_first_group (M : ℕ) : (M * 15 = 25 * 18) → M = 30 :=
by
  sorry

end number_of_men_in_first_group_l602_60260


namespace sum_of_coefficients_l602_60253

theorem sum_of_coefficients (d : ℤ) (h : d ≠ 0) :
    let a := 3 + 2
    let b := 17 + 2
    let c := 10 + 5
    let e := 16 + 4
    a + b + c + e = 59 :=
by
  let a := 3 + 2
  let b := 17 + 2
  let c := 10 + 5
  let e := 16 + 4
  sorry

end sum_of_coefficients_l602_60253


namespace evaluate_expression_l602_60200

theorem evaluate_expression (x : ℤ) (z : ℤ) (hx : x = 4) (hz : z = -2) : z * (z - 4 * x) = 36 :=
by
  sorry

end evaluate_expression_l602_60200


namespace average_home_runs_correct_l602_60217

-- Define the number of players hitting specific home runs
def players_5_hr : ℕ := 3
def players_7_hr : ℕ := 2
def players_9_hr : ℕ := 1
def players_11_hr : ℕ := 2
def players_13_hr : ℕ := 1

-- Calculate the total number of home runs and total number of players
def total_hr : ℕ := 5 * players_5_hr + 7 * players_7_hr + 9 * players_9_hr + 11 * players_11_hr + 13 * players_13_hr
def total_players : ℕ := players_5_hr + players_7_hr + players_9_hr + players_11_hr + players_13_hr

-- Calculate the average number of home runs
def average_home_runs : ℚ := total_hr / total_players

-- The theorem we need to prove
theorem average_home_runs_correct : average_home_runs = 73 / 9 :=
by
  sorry

end average_home_runs_correct_l602_60217


namespace shop_owner_profitable_l602_60245

noncomputable def shop_owner_profit (CP_SP_difference_percentage: ℚ) (CP: ℚ) (buy_cheat_percentage: ℚ) (sell_cheat_percentage: ℚ) (buy_discount_percentage: ℚ) (sell_markup_percentage: ℚ) : ℚ := 
  CP_SP_difference_percentage * 100

theorem shop_owner_profitable :
  shop_owner_profit ((114 * (110 / 80 / 100) - 90) / 90) 1 0.14 0.20 0.10 0.10 = 74.17 := 
by
  sorry

end shop_owner_profitable_l602_60245


namespace find_minimal_product_l602_60227

theorem find_minimal_product : ∃ x y : ℤ, (20 * x + 19 * y = 2019) ∧ (x * y = 2623) ∧ (∀ z w : ℤ, (20 * z + 19 * w = 2019) → |x - y| ≤ |z - w|) :=
by
  -- definitions and theorems to prove the problem would be placed here
  sorry

end find_minimal_product_l602_60227


namespace largest_x_l602_60290

theorem largest_x (x : ℝ) (h : ⌊x⌋ / x = 7 / 8) : x = 48 / 7 := 
sorry

end largest_x_l602_60290


namespace cos_double_angle_given_tan_l602_60278

theorem cos_double_angle_given_tan (x : ℝ) (h : Real.tan x = 2) : Real.cos (2 * x) = -3 / 5 :=
by sorry

end cos_double_angle_given_tan_l602_60278


namespace numberOfBoysInClass_l602_60233

-- Define the problem condition: students sit in a circle and boy at 5th position is opposite to boy at 20th position
def studentsInCircle (n : ℕ) : Prop :=
  (n > 5) ∧ (n > 20) ∧ ((20 - 5) * 2 + 2 = n)

-- The main theorem: Given the conditions, prove the total number of boys equals 32
theorem numberOfBoysInClass : ∀ n : ℕ, studentsInCircle n → n = 32 :=
by
  intros n hn
  sorry

end numberOfBoysInClass_l602_60233


namespace joan_total_seashells_l602_60220

-- Definitions of the conditions
def joan_initial_seashells : ℕ := 79
def mike_additional_seashells : ℕ := 63

-- Definition of the proof problem statement
theorem joan_total_seashells : joan_initial_seashells + mike_additional_seashells = 142 :=
by
  -- Proof would go here
  sorry

end joan_total_seashells_l602_60220


namespace ratio_girls_to_boys_l602_60248

theorem ratio_girls_to_boys (g b : ℕ) (h1 : g = b + 4) (h2 : g + b = 28) :
  g / gcd g b = 4 ∧ b / gcd g b = 3 :=
by
  sorry

end ratio_girls_to_boys_l602_60248


namespace fraction_ratio_equivalence_l602_60264

theorem fraction_ratio_equivalence :
  ∃ (d : ℚ), d = 240 / 1547 ∧ ((2 / 13) / d) = ((5 / 34) / (7 / 48)) := 
by
  sorry

end fraction_ratio_equivalence_l602_60264


namespace stratified_sampling_third_grade_students_l602_60299

variable (total_students : ℕ) (second_year_female_probability : ℚ) (sample_size : ℕ)

theorem stratified_sampling_third_grade_students
  (h_total : total_students = 2000)
  (h_probability : second_year_female_probability = 0.19)
  (h_sample_size : sample_size = 64) :
  let sampling_fraction := 64 / 2000
  let third_grade_students := 2000 * sampling_fraction
  third_grade_students = 16 :=
by
  -- the proof would go here, but we're skipping it per instructions
  sorry

end stratified_sampling_third_grade_students_l602_60299


namespace cyclic_quadrilateral_angle_D_l602_60247

theorem cyclic_quadrilateral_angle_D (A B C D : ℝ) (h₁ : A + B + C + D = 360) (h₂ : ∃ x, A = 3 * x ∧ B = 4 * x ∧ C = 6 * x) :
  D = 100 :=
by
  sorry

end cyclic_quadrilateral_angle_D_l602_60247


namespace max_min_values_of_f_l602_60243

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_min_values_of_f :
  (∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x = 2) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x ≥ -18) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x = -18)
:= by
  sorry  -- To be replaced with the actual proof

end max_min_values_of_f_l602_60243


namespace gcd_of_gx_and_x_l602_60246

theorem gcd_of_gx_and_x (x : ℕ) (h : 7200 ∣ x) : Nat.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 5) * (4 * x + 7)) x = 30 := 
by 
  sorry

end gcd_of_gx_and_x_l602_60246


namespace no_tangent_line_l602_60214

-- Define the function f(x) = x^3 - 3ax
def f (a x : ℝ) : ℝ := x^3 - 3 * a * x

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3 * x^2 - 3 * a

-- Proposition stating no b exists in ℝ such that y = -x + b is tangent to f
theorem no_tangent_line (a : ℝ) (H : ∀ b : ℝ, ¬ ∃ x : ℝ, f' a x = -1) : a < 1 / 3 :=
by
  sorry

end no_tangent_line_l602_60214


namespace diameter_percentage_l602_60265

theorem diameter_percentage (d_R d_S : ℝ) (h : π * (d_R / 2)^2 = 0.25 * π * (d_S / 2)^2) : 
  d_R = 0.5 * d_S :=
by 
  sorry

end diameter_percentage_l602_60265


namespace electronics_weight_l602_60281

variable (B C E : ℝ)

-- Conditions
def initial_ratio : Prop := B / 5 = C / 4 ∧ C / 4 = E / 2
def removed_clothes : Prop := B / 10 = (C - 9) / 4

-- Proof statement
theorem electronics_weight (h1 : initial_ratio B C E) (h2 : removed_clothes B C) : E = 9 := 
by
  sorry

end electronics_weight_l602_60281


namespace max_value_of_S_n_divided_l602_60283

noncomputable def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def S_n (a₁ d n : ℕ) : ℕ :=
  n * (n + 4)

theorem max_value_of_S_n_divided (a₁ d : ℕ) (h₁ : ∀ n, a₁ + (2 * n - 1) * d = 2 * (a₁ + (n - 1) * d) - 3)
  (h₂ : (a₁ + 5 * d)^2 = a₁ * (a₁ + 20 * d)) :
  ∃ n, 2 * S_n a₁ d n / 2^n = 6 := 
sorry

end max_value_of_S_n_divided_l602_60283


namespace sum_arithmetic_series_l602_60263

theorem sum_arithmetic_series :
  let a1 := 1000
  let an := 5000
  let d := 4
  let n := (an - a1) / d + 1
  let Sn := n * (a1 + an) / 2
  Sn = 3003000 := by
    sorry

end sum_arithmetic_series_l602_60263


namespace correct_operation_l602_60213

noncomputable def valid_operation (n : ℕ) (a b : ℕ) (c d : ℤ) (x : ℚ) : Prop :=
  match n with
  | 0 => (x ^ a / x ^ b = x ^ (a - b))
  | 1 => (x ^ a * x ^ b = x ^ (a + b))
  | 2 => (c * x ^ a + d * x ^ a = (c + d) * x ^ a)
  | 3 => ((c * x ^ a) ^ b = c ^ b * x ^ (a * b))
  | _ => False

theorem correct_operation (x : ℚ) : valid_operation 1 2 3 0 0 x :=
by sorry

end correct_operation_l602_60213


namespace smallest_solution_eq_l602_60271

theorem smallest_solution_eq :
  (∀ x : ℝ, x ≠ 3 →
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 15) → 
  x = 1 - Real.sqrt 10 ∨ (∃ y : ℝ, y ≤ 1 - Real.sqrt 10 ∧ y ≠ 3 ∧ 3 * y / (y - 3) + (3 * y^2 - 27) / y = 15)) :=
sorry

end smallest_solution_eq_l602_60271


namespace allocation_schemes_correct_l602_60269

noncomputable def allocation_schemes : Nat :=
  let C (n k : Nat) : Nat := Nat.choose n k
  -- Calculate category 1: one school gets 1 professor, two get 2 professors each
  let category1 := C 3 1 * C 5 1 * C 4 2 * C 2 2 / 2
  -- Calculate category 2: one school gets 3 professors, two get 1 professor each
  let category2 := C 3 1 * C 5 3 * C 2 1 * C 1 1 / 2
  -- Total allocation ways
  let totalWays := 6 * (category1 + category2)
  totalWays

theorem allocation_schemes_correct : allocation_schemes = 900 := by
  sorry

end allocation_schemes_correct_l602_60269


namespace max_arithmetic_subsequences_l602_60286

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d c : ℤ), ∀ n : ℕ, a n = d * n + c

-- Condition that the sum of the indices is even
def sum_indices_even (n m : ℕ) : Prop :=
  (n % 2 = 0 ∧ m % 2 = 0) ∨ (n % 2 = 1 ∧ m % 2 = 1)

-- Maximum count of 3-term arithmetic sequences in a sequence of 20 terms
theorem max_arithmetic_subsequences (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) :
  ∃ n : ℕ, n = 180 :=
by
  sorry

end max_arithmetic_subsequences_l602_60286


namespace probability_of_one_triplet_without_any_pairs_l602_60256

noncomputable def probability_one_triplet_no_pairs : ℚ :=
  let total_outcomes := 6^5
  let choices_for_triplet := 6
  let ways_to_choose_triplet_dice := Nat.choose 5 3
  let choices_for_remaining_dice := 5 * 4
  let successful_outcomes := choices_for_triplet * ways_to_choose_triplet_dice * choices_for_remaining_dice
  successful_outcomes / total_outcomes

theorem probability_of_one_triplet_without_any_pairs :
  probability_one_triplet_no_pairs = 25 / 129 := by
  sorry

end probability_of_one_triplet_without_any_pairs_l602_60256


namespace probability_first_spade_second_ace_l602_60229

theorem probability_first_spade_second_ace :
  let n : ℕ := 52
  let spades : ℕ := 13
  let aces : ℕ := 4
  let ace_of_spades : ℕ := 1
  let non_ace_spades : ℕ := spades - ace_of_spades
  (non_ace_spades / n : ℚ) * (aces / (n - 1) : ℚ) +
  (ace_of_spades / n : ℚ) * ((aces - 1) / (n - 1) : ℚ) =
  (1 / n : ℚ) :=
by {
  -- proof goes here
  sorry
}

end probability_first_spade_second_ace_l602_60229


namespace time_period_for_investment_l602_60216

variable (P R₁₅ R₁₀ I₁₅ I₁₀ : ℝ)
variable (T : ℝ)

noncomputable def principal := 8400
noncomputable def rate15 := 15
noncomputable def rate10 := 10
noncomputable def interestDifference := 840

theorem time_period_for_investment :
  ∀ (T : ℝ),
    P = principal →
    R₁₅ = rate15 →
    R₁₀ = rate10 →
    I₁₅ = P * (R₁₅ / 100) * T →
    I₁₀ = P * (R₁₀ / 100) * T →
    (I₁₅ - I₁₀) = interestDifference →
    T = 2 :=
  sorry

end time_period_for_investment_l602_60216


namespace third_factorial_is_7_l602_60223

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Problem conditions
def b : ℕ := 9
def factorial_b_minus_2 : ℕ := factorial (b - 2)
def factorial_b_plus_1 : ℕ := factorial (b + 1)
def GCD_value : ℕ := Nat.gcd (Nat.gcd factorial_b_minus_2 factorial_b_plus_1) (factorial 7)

-- Theorem statement
theorem third_factorial_is_7 :
  Nat.gcd (Nat.gcd (factorial (b - 2)) (factorial (b + 1))) (factorial 7) = 5040 →
  ∃ k : ℕ, factorial k = 5040 ∧ k = 7 :=
by
  sorry

end third_factorial_is_7_l602_60223


namespace condition1_condition2_condition3_condition4_l602_60258

-- Proof for the equivalence of conditions and point descriptions

theorem condition1 (x y : ℝ) : 
  (x >= -2) ↔ ∃ y : ℝ, x = -2 ∨ x > -2 := 
by
  sorry

theorem condition2 (x y : ℝ) : 
  (-2 < x ∧ x < 2) ↔ ∃ y : ℝ, -2 < x ∧ x < 2 := 
by
  sorry

theorem condition3 (x y : ℝ) : 
  (|x| < 2) ↔ -2 < x ∧ x < 2 :=
by
  sorry

theorem condition4 (x y : ℝ) : 
  (|x| ≥ 2) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by 
  sorry

end condition1_condition2_condition3_condition4_l602_60258


namespace three_digit_number_base_10_l602_60228

theorem three_digit_number_base_10 (A B C : ℕ) (x : ℕ)
  (h1 : x = 100 * A + 10 * B + 6)
  (h2 : x = 82 * C + 36)
  (hA : 1 ≤ A ∧ A ≤ 9)
  (hB : 0 ≤ B ∧ B ≤ 9)
  (hC : 0 ≤ C ∧ C ≤ 8) :
  x = 446 := by
  sorry

end three_digit_number_base_10_l602_60228


namespace total_employees_l602_60210

theorem total_employees (female_employees managers male_associates female_managers : ℕ)
  (h_female_employees : female_employees = 90)
  (h_managers : managers = 40)
  (h_male_associates : male_associates = 160)
  (h_female_managers : female_managers = 40) :
  female_employees - female_managers + male_associates + managers = 250 :=
by {
  sorry
}

end total_employees_l602_60210


namespace larger_number_solution_l602_60275

theorem larger_number_solution (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
by
  sorry

end larger_number_solution_l602_60275


namespace remainder_of_polynomial_division_is_88_l602_60221

def p (x : ℝ) : ℝ := 4*x^5 - 3*x^4 + 5*x^3 - 7*x^2 + 3*x - 10

theorem remainder_of_polynomial_division_is_88 :
  p 2 = 88 :=
by
  sorry

end remainder_of_polynomial_division_is_88_l602_60221


namespace jane_baking_time_l602_60291

-- Definitions based on the conditions
variables (J : ℝ) (J_time : J > 0) -- J is the time it takes Jane to bake cakes individually
variables (Roy_time : 5 > 0) -- Roy can bake cakes in 5 hours
variables (together_time : 2 > 0) -- They work together for 2 hours
variables (remaining_time : 0.4 > 0) -- Jane completes the remaining task in 0.4 hours alone

-- Lean statement to prove Jane's individual baking time
theorem jane_baking_time : 
  (2 * (1 / J + 1 / 5) + 0.4 * (1 / J) = 1) → 
  J = 4 :=
by 
  sorry

end jane_baking_time_l602_60291


namespace geom_seq_sum_l602_60249

theorem geom_seq_sum (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 + a 2 = 16) 
  (h2 : a 3 + a 4 = 24) 
  (h_geom : ∀ n, a (n+1) = r * a n):
  a 7 + a 8 = 54 :=
sorry

end geom_seq_sum_l602_60249


namespace no_real_roots_ffx_l602_60225

noncomputable def quadratic_f (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem no_real_roots_ffx (a b c : ℝ) (h : (b - 1)^2 < 4 * a * c) :
  ∀ x : ℝ, quadratic_f a b c (quadratic_f a b c x) ≠ x :=
by
  sorry

end no_real_roots_ffx_l602_60225


namespace cheeseburger_cost_l602_60255

-- Definitions for given conditions
def milkshake_price : ℝ := 5
def cheese_fries_price : ℝ := 8
def jim_money : ℝ := 20
def cousin_money : ℝ := 10
def combined_money := jim_money + cousin_money
def spending_percentage : ℝ := 0.80
def total_spent := spending_percentage * combined_money
def number_of_milkshakes : ℝ := 2
def number_of_cheeseburgers : ℝ := 2

-- Prove the cost of one cheeseburger
theorem cheeseburger_cost : (total_spent - (number_of_milkshakes * milkshake_price) - cheese_fries_price) / number_of_cheeseburgers = 3 :=
by
  sorry

end cheeseburger_cost_l602_60255
