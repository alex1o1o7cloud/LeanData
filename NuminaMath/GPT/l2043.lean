import Mathlib

namespace regular_polygon_sides_l2043_204379

theorem regular_polygon_sides (n : ℕ) (h₁ : n > 2) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n → True) (h₃ : (360 / n : ℝ) = 30) : n = 12 := by
  sorry

end regular_polygon_sides_l2043_204379


namespace small_disks_radius_l2043_204346

theorem small_disks_radius (r : ℝ) (h : r > 0) :
  (2 * r ≥ 1 + r) → (r ≥ 1 / 2) := by
  intro hr
  linarith

end small_disks_radius_l2043_204346


namespace pandas_and_bamboo_l2043_204354

-- Definitions for the conditions
def number_of_pandas (x : ℕ) :=
  (∃ y : ℕ, y = 5 * x + 11 ∧ y = 2 * (3 * x - 5) - 8)

-- Theorem stating the solution
theorem pandas_and_bamboo (x y : ℕ) (h1 : y = 5 * x + 11) (h2 : y = 2 * (3 * x - 5) - 8) : x = 29 ∧ y = 156 :=
by {
  sorry
}

end pandas_and_bamboo_l2043_204354


namespace volume_to_surface_area_ratio_l2043_204329

-- Define the shape as described in the problem
structure Shape :=
(center_cube : ℕ)  -- Center cube
(surrounding_cubes : ℕ)  -- Surrounding cubes
(unit_volume : ℕ)  -- Volume of each unit cube
(unit_face_area : ℕ)  -- Surface area of each face of the unit cube

-- Conditions and definitions
def is_special_shape (s : Shape) : Prop :=
  s.center_cube = 1 ∧ s.surrounding_cubes = 7 ∧ s.unit_volume = 1 ∧ s.unit_face_area = 1

-- Theorem statement
theorem volume_to_surface_area_ratio (s : Shape) (h : is_special_shape s) : (s.center_cube + s.surrounding_cubes) * s.unit_volume / (s.surrounding_cubes * 5 * s.unit_face_area) = 8 / 35 :=
by
  sorry

end volume_to_surface_area_ratio_l2043_204329


namespace correct_order_option_C_l2043_204309

def length_unit_ordered (order : List String) : Prop :=
  order = ["kilometer", "meter", "centimeter", "millimeter"]

def option_A := ["kilometer", "meter", "millimeter", "centimeter"]
def option_B := ["meter", "kilometer", "centimeter", "millimeter"]
def option_C := ["kilometer", "meter", "centimeter", "millimeter"]

theorem correct_order_option_C : length_unit_ordered option_C := by
  sorry

end correct_order_option_C_l2043_204309


namespace afternoon_sales_l2043_204393

theorem afternoon_sales (x : ℕ) (H1 : 2 * x + x = 390) : 2 * x = 260 :=
by
  sorry

end afternoon_sales_l2043_204393


namespace instantaneous_speed_at_4_l2043_204333

def motion_equation (t : ℝ) : ℝ := t^2 - 2 * t + 5

theorem instantaneous_speed_at_4 :
  (deriv motion_equation 4) = 6 :=
by
  sorry

end instantaneous_speed_at_4_l2043_204333


namespace solve_fractional_equation_l2043_204335

theorem solve_fractional_equation (x : ℝ) (h : (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 3) : 
  x = 1 :=
sorry

end solve_fractional_equation_l2043_204335


namespace ratio_surfer_malibu_santa_monica_l2043_204370

theorem ratio_surfer_malibu_santa_monica (M S : ℕ) (hS : S = 20) (hTotal : M + S = 60) : M / S = 2 :=
by 
  sorry

end ratio_surfer_malibu_santa_monica_l2043_204370


namespace value_of_quotient_l2043_204384

variable (a b c d : ℕ)

theorem value_of_quotient 
  (h1 : a = 3 * b)
  (h2 : b = 2 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 15 :=
by
  sorry

end value_of_quotient_l2043_204384


namespace tangent_points_sum_constant_l2043_204320

theorem tangent_points_sum_constant 
  (a : ℝ) (x1 y1 x2 y2 : ℝ)
  (hC1 : x1^2 = 4 * y1)
  (hC2 : x2^2 = 4 * y2)
  (hT1 : y1 - (-2) = (1/2)*x1*(x1 - a))
  (hT2 : y2 - (-2) = (1/2)*x2*(x2 - a)) :
  x1 * x2 + y1 * y2 = -4 :=
sorry

end tangent_points_sum_constant_l2043_204320


namespace tv_horizontal_length_l2043_204318

noncomputable def rectangleTvLengthRatio (l h : ℝ) : Prop :=
  l / h = 16 / 9

noncomputable def rectangleTvDiagonal (l h d : ℝ) : Prop :=
  l^2 + h^2 = d^2

theorem tv_horizontal_length
  (h : ℝ)
  (h_positive : h > 0)
  (d : ℝ)
  (h_ratio : rectangleTvLengthRatio l h)
  (h_diagonal : rectangleTvDiagonal l h d)
  (h_diagonal_value : d = 36) :
  l = 56.27 :=
by
  sorry

end tv_horizontal_length_l2043_204318


namespace product_seqFrac_l2043_204337

def seqFrac (n : ℕ) : ℚ := (n : ℚ) / (n + 5 : ℚ)

theorem product_seqFrac :
  ((List.range 53).map seqFrac).prod = 1 / 27720 := by
  sorry

end product_seqFrac_l2043_204337


namespace find_number_to_be_multiplied_l2043_204358

def correct_multiplier := 43
def incorrect_multiplier := 34
def difference := 1224

theorem find_number_to_be_multiplied (x : ℕ) : correct_multiplier * x - incorrect_multiplier * x = difference → x = 136 :=
by
  sorry

end find_number_to_be_multiplied_l2043_204358


namespace oliver_used_fraction_l2043_204317

variable (x : ℚ)

/--
Oliver had 135 stickers. He used a fraction x of his stickers, gave 2/5 of the remaining to his friend, and kept the remaining 54 stickers. Prove that he used 1/3 of his stickers.
-/
theorem oliver_used_fraction (h : 135 - (135 * x) - (2 / 5) * (135 - 135 * x) = 54) : 
  x = 1 / 3 := 
sorry

end oliver_used_fraction_l2043_204317


namespace train_length_l2043_204373

theorem train_length (speed_km_per_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_per_hr = 80) (h_time : time_sec = 9) :
  ∃ length_m : ℕ, length_m = 200 :=
by
  sorry

end train_length_l2043_204373


namespace increase_in_license_plates_l2043_204305

/-- The number of old license plates and new license plates in MiraVille. -/
def old_license_plates : ℕ := 26^2 * 10^3
def new_license_plates : ℕ := 26^2 * 10^4

/-- The ratio of the number of new license plates to the number of old license plates is 10. -/
theorem increase_in_license_plates : new_license_plates / old_license_plates = 10 := by
  unfold old_license_plates new_license_plates
  sorry

end increase_in_license_plates_l2043_204305


namespace max_value_OP_OQ_l2043_204334

def circle_1_polar_eq (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta

def circle_2_polar_eq (rho theta : ℝ) : Prop :=
  rho = 2 * Real.sin theta

theorem max_value_OP_OQ (alpha : ℝ) :
  (∃ rho1 rho2 : ℝ, circle_1_polar_eq rho1 alpha ∧ circle_2_polar_eq rho2 alpha) ∧
  (∃ max_OP_OQ : ℝ, max_OP_OQ = 4) :=
sorry

end max_value_OP_OQ_l2043_204334


namespace find_fx_l2043_204310

theorem find_fx (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = x^2 - 2 * x) : ∀ x, f x = x^2 - 1 :=
by
  intro x
  sorry

end find_fx_l2043_204310


namespace heather_shared_blocks_l2043_204300

-- Define the initial number of blocks Heather starts with
def initial_blocks : ℕ := 86

-- Define the final number of blocks Heather ends up with
def final_blocks : ℕ := 45

-- Define the number of blocks Heather shared
def blocks_shared (initial final : ℕ) : ℕ := initial - final

-- Prove that the number of blocks Heather shared is 41
theorem heather_shared_blocks : blocks_shared initial_blocks final_blocks = 41 := by
  -- Proof steps will be added here
  sorry

end heather_shared_blocks_l2043_204300


namespace profit_percentage_is_twenty_percent_l2043_204316

def selling_price : ℕ := 900
def profit : ℕ := 150
def cost_price : ℕ := selling_price - profit
def profit_percentage : ℕ := (profit * 100) / cost_price

theorem profit_percentage_is_twenty_percent : profit_percentage = 20 := by
  sorry

end profit_percentage_is_twenty_percent_l2043_204316


namespace proof_statements_imply_negation_l2043_204341

-- Define propositions p, q, and r
variables (p q r : Prop)

-- Statement (1): p, q, and r are all true.
def statement_1 : Prop := p ∧ q ∧ r

-- Statement (2): p is true, q is false, and r is true.
def statement_2 : Prop := p ∧ ¬ q ∧ r

-- Statement (3): p is false, q is true, and r is false.
def statement_3 : Prop := ¬ p ∧ q ∧ ¬ r

-- Statement (4): p and r are false, q is true.
def statement_4 : Prop := ¬ p ∧ q ∧ ¬ r

-- The negation of "p and q are true, and r is false" is "¬(p ∧ q) ∨ r"
def negation : Prop := ¬(p ∧ q) ∨ r

-- Proof statement that each of the 4 statements implies the negation
theorem proof_statements_imply_negation :
  (statement_1 p q r → negation p q r) ∧
  (statement_2 p q r → negation p q r) ∧
  (statement_3 p q r → negation p q r) ∧
  (statement_4 p q r → negation p q r) :=
by
  sorry

end proof_statements_imply_negation_l2043_204341


namespace smallest_n_l2043_204319

theorem smallest_n (n : ℕ) : 
  (n > 0 ∧ ((n^2 + n + 1)^2 > 1999) ∧ ∀ m : ℕ, (m > 0 ∧ (m^2 + m + 1)^2 > 1999) → m ≥ n) → n = 7 :=
sorry

end smallest_n_l2043_204319


namespace least_six_digit_divisible_by_198_l2043_204359

/-- The least 6-digit natural number that is divisible by 198 is 100188. -/
theorem least_six_digit_divisible_by_198 : 
  ∃ n : ℕ, n ≥ 100000 ∧ n % 198 = 0 ∧ n = 100188 :=
by
  use 100188
  sorry

end least_six_digit_divisible_by_198_l2043_204359


namespace find_fourth_vertex_l2043_204389

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_parallelogram (A B C D : Point) : Prop :=
  is_midpoint ({x := 0, y := -9}) A C ∧ is_midpoint ({x := 2, y := 6}) B D ∧
  is_midpoint ({x := 4, y := 5}) C D ∧ is_midpoint ({x := 0, y := -9}) A D

theorem find_fourth_vertex :
  ∃ D : Point,
    (is_parallelogram ({x := 0, y := -9}) ({x := 2, y := 6}) ({x := 4, y := 5}) D)
    ∧ ((D = {x := 2, y := -10}) ∨ (D = {x := -2, y := -8}) ∨ (D = {x := 6, y := 20})) :=
sorry

end find_fourth_vertex_l2043_204389


namespace A_equals_4_of_rounded_to_tens_9430_l2043_204361

variable (A B : ℕ)

theorem A_equals_4_of_rounded_to_tens_9430
  (h1 : 9430 = 9000 + 100 * A + 10 * 3 + B)
  (h2 : B < 5)
  (h3 : 0 ≤ A ∧ A ≤ 9)
  (h4 : 0 ≤ B ∧ B ≤ 9) :
  A = 4 :=
by
  sorry

end A_equals_4_of_rounded_to_tens_9430_l2043_204361


namespace total_weight_of_5_moles_of_cai2_l2043_204347

-- Definitions based on the conditions
def weight_of_calcium : Real := 40.08
def weight_of_iodine : Real := 126.90
def iodine_atoms_in_cai2 : Nat := 2
def moles_of_calcium_iodide : Nat := 5

-- Lean 4 statement for the proof problem
theorem total_weight_of_5_moles_of_cai2 :
  (weight_of_calcium + (iodine_atoms_in_cai2 * weight_of_iodine)) * moles_of_calcium_iodide = 1469.4 := by
  sorry

end total_weight_of_5_moles_of_cai2_l2043_204347


namespace Tim_age_l2043_204367

theorem Tim_age (T t : ℕ) (h1 : T = 22) (h2 : T = 2 * t + 6) : t = 8 := by
  sorry

end Tim_age_l2043_204367


namespace square_area_increase_l2043_204342

variable (s : ℝ)

theorem square_area_increase (h : s > 0) : 
  let s_new := 1.30 * s
  let A_original := s^2
  let A_new := s_new^2
  let percentage_increase := ((A_new - A_original) / A_original) * 100
  percentage_increase = 69 := by
sorry

end square_area_increase_l2043_204342


namespace geom_seq_a11_l2043_204392

variable {α : Type*} [LinearOrderedField α]

def geom_seq (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geom_seq_a11
  (a : ℕ → α)
  (q : α)
  (ha3 : a 3 = 3)
  (ha7 : a 7 = 6)
  (hgeom : geom_seq a q) :
  a 11 = 12 :=
by
  sorry

end geom_seq_a11_l2043_204392


namespace output_for_input_8_is_8_over_65_l2043_204321

def function_f (n : ℕ) : ℚ := n / (n^2 + 1)

theorem output_for_input_8_is_8_over_65 : function_f 8 = 8 / 65 := by
  sorry

end output_for_input_8_is_8_over_65_l2043_204321


namespace turnip_pulled_by_mice_l2043_204380

theorem turnip_pulled_by_mice :
  ∀ (M B G D J C : ℕ),
    D = 2 * B →
    B = 3 * G →
    G = 4 * J →
    J = 5 * C →
    C = 6 * M →
    (D + B + G + J + C + M) ≥ (D + B + G + J + C) + M → 
    1237 * M ≤ (D + B + G + J + C + M) :=
by
  intros M B G D J C h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5]
  linarith

end turnip_pulled_by_mice_l2043_204380


namespace product_of_values_l2043_204302

-- Given definitions: N as a real number and R as a real constant
variables (N R : ℝ)

-- Condition
def condition : Prop := N - 5 / N = R

-- The proof statement
theorem product_of_values (h : condition N R) : ∀ (N1 N2 : ℝ), ((N1 - 5 / N1 = R) ∧ (N2 - 5 / N2 = R)) → (N1 * N2 = -5) :=
by sorry

end product_of_values_l2043_204302


namespace taxi_fare_l2043_204303

theorem taxi_fare (fare : ℕ → ℝ) (distance : ℕ) :
  (∀ d, d > 10 → fare d = 20 + (d - 10) * (140 / 70)) →
  fare 80 = 160 →
  fare 100 = 200 :=
by
  intros h_fare h_fare_80
  show fare 100 = 200
  sorry

end taxi_fare_l2043_204303


namespace find_k_l2043_204336

def geom_seq (c : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = c * (a n)

def sum_first_n_terms (S : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n, S n = 3^n + k

theorem find_k {c : ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ} {k : ℝ} (hGeom : geom_seq c a) (hSum : sum_first_n_terms S k) :
  k = -1 :=
by
  sorry

end find_k_l2043_204336


namespace circle_radius_l2043_204340

theorem circle_radius (x y : ℝ) :
  (∃ r, r > 0 ∧ (∀ x y, x^2 - 8*x + y^2 - 4*y + 16 = 0 → r = 2)) :=
sorry

end circle_radius_l2043_204340


namespace dance_lessons_l2043_204349

theorem dance_lessons (cost_per_lesson : ℕ) (free_lessons : ℕ) (amount_paid : ℕ) 
  (H1 : cost_per_lesson = 10) 
  (H2 : free_lessons = 2) 
  (H3 : amount_paid = 80) : 
  (amount_paid / cost_per_lesson + free_lessons = 10) :=
by
  sorry

end dance_lessons_l2043_204349


namespace numbers_composite_l2043_204382

theorem numbers_composite (a b c d : ℕ) (h : a * b = c * d) : ∃ x y : ℕ, (x > 1 ∧ y > 1) ∧ a^2000 + b^2000 + c^2000 + d^2000 = x * y := 
sorry

end numbers_composite_l2043_204382


namespace same_side_interior_not_complementary_l2043_204313

-- Defining the concept of same-side interior angles and complementary angles
def same_side_interior (α β : ℝ) : Prop := 
  α + β = 180 

def complementary (α β : ℝ) : Prop :=
  α + β = 90

-- To state the proposition that should be proven false
theorem same_side_interior_not_complementary (α β : ℝ) (h : same_side_interior α β) : ¬ complementary α β :=
by
  -- We state the observable contradiction here, and since the proof is not required we use sorry
  sorry

end same_side_interior_not_complementary_l2043_204313


namespace medians_form_right_triangle_medians_inequality_l2043_204395

variable {α : Type*}
variables {a b c : ℝ}
variables {m_a m_b m_c : ℝ}
variable (orthogonal_medians : m_a * m_b = 0)

-- Part (a)
theorem medians_form_right_triangle
  (orthogonal_medians : m_a * m_b = 0) :
  m_a^2 + m_b^2 = m_c^2 :=
sorry

-- Part (b)
theorem medians_inequality
  (orthogonal_medians : m_a * m_b = 0)
  (triangle_sides : a^2 + b^2 = 5 * c^2): 
  5 * (a^2 + b^2 - c^2) ≥ 8 * a * b :=
sorry

end medians_form_right_triangle_medians_inequality_l2043_204395


namespace smallest_b_for_45_b_square_l2043_204397

theorem smallest_b_for_45_b_square :
  ∃ b : ℕ, b > 5 ∧ ∃ n : ℕ, 4 * b + 5 = n^2 ∧ b = 11 :=
by
  sorry

end smallest_b_for_45_b_square_l2043_204397


namespace sports_parade_children_l2043_204322

theorem sports_parade_children :
  ∃ (a : ℤ), a ≡ 5 [ZMOD 8] ∧ a ≡ 7 [ZMOD 10] ∧ 100 ≤ a ∧ a ≤ 150 ∧ a = 125 := by
sorry

end sports_parade_children_l2043_204322


namespace calculate_expression_l2043_204338

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := 
by
  -- proof goes here
  sorry

end calculate_expression_l2043_204338


namespace shiela_paintings_l2043_204344

theorem shiela_paintings (h1 : 18 % 2 = 0) : 18 / 2 = 9 := 
by sorry

end shiela_paintings_l2043_204344


namespace probability_to_form_computers_l2043_204325

def letters_in_campus : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def letters_in_threads : Finset Char := {'T', 'H', 'R', 'E', 'A', 'D', 'S'}
def letters_in_glow : Finset Char := {'G', 'L', 'O', 'W'}
def letters_in_computers : Finset Char := {'C', 'O', 'M', 'P', 'U', 'T', 'E', 'R', 'S'}

noncomputable def probability_campus : ℚ := 1 / Nat.choose 6 3
noncomputable def probability_threads : ℚ := 1 / Nat.choose 7 5
noncomputable def probability_glow : ℚ := 1 / (Nat.choose 4 2 / Nat.choose 3 1)

noncomputable def overall_probability : ℚ :=
  probability_campus * probability_threads * probability_glow

theorem probability_to_form_computers :
  overall_probability = 1 / 840 := by
  sorry

end probability_to_form_computers_l2043_204325


namespace heads_count_l2043_204385

theorem heads_count (H T : ℕ) (h1 : H + T = 128) (h2 : H = T + 12) : H = 70 := by
  sorry

end heads_count_l2043_204385


namespace number_of_dolls_of_jane_l2043_204323

-- Given conditions
def total_dolls (J D : ℕ) := J + D = 32
def jill_has_more (J D : ℕ) := D = J + 6

-- Statement to prove
theorem number_of_dolls_of_jane (J D : ℕ) (h1 : total_dolls J D) (h2 : jill_has_more J D) : J = 13 :=
by
  sorry

end number_of_dolls_of_jane_l2043_204323


namespace range_of_a_l2043_204351

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then -x^2 - 1 else Real.log (x + 1)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x ≤ a * x) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l2043_204351


namespace grown_ups_in_milburg_l2043_204391

def number_of_children : ℕ := 2987
def total_population : ℕ := 8243

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 := 
by 
  sorry

end grown_ups_in_milburg_l2043_204391


namespace determine_a_b_l2043_204399

theorem determine_a_b (a b : ℝ) :
  (∀ x, y = x^2 + a * x + b) ∧ (∀ t, t = 0 → 3 * t - (t^2 + a * t + b) + 1 = 0) →
  a = 3 ∧ b = 1 :=
by
  sorry

end determine_a_b_l2043_204399


namespace impossible_to_arrange_circle_l2043_204374

theorem impossible_to_arrange_circle : 
  ¬∃ (f : Fin 10 → Fin 10), 
    (∀ i : Fin 10, (abs ((f i).val - (f (i + 1)).val : Int) = 3 
                ∨ abs ((f i).val - (f (i + 1)).val : Int) = 4 
                ∨ abs ((f i).val - (f (i + 1)).val : Int) = 5)) :=
sorry

end impossible_to_arrange_circle_l2043_204374


namespace pencils_to_sell_for_profit_l2043_204332

theorem pencils_to_sell_for_profit 
    (total_pencils : ℕ) 
    (buy_price sell_price : ℝ) 
    (desired_profit : ℝ) 
    (h_total_pencils : total_pencils = 2000) 
    (h_buy_price : buy_price = 0.15) 
    (h_sell_price : sell_price = 0.30) 
    (h_desired_profit : desired_profit = 150) :
    total_pencils * buy_price + desired_profit = total_pencils * sell_price → total_pencils = 1500 :=
by
    sorry

end pencils_to_sell_for_profit_l2043_204332


namespace domain_of_function_l2043_204348

def quadratic_inequality (x : ℝ) : Prop := -8 * x^2 - 14 * x + 9 ≥ 0

theorem domain_of_function :
  {x : ℝ | quadratic_inequality x} = {x : ℝ | x ≤ -1} ∪ {x : ℝ | x ≥ 9 / 8} :=
by
  -- The detailed proof would go here, but we're focusing on the statement structure.
  sorry

end domain_of_function_l2043_204348


namespace radius_of_sphere_is_approximately_correct_l2043_204378

noncomputable def radius_of_sphere_in_cylinder_cone : ℝ :=
  let radius_cylinder := 12
  let height_cylinder := 30
  let radius_sphere := 21 - 0.5 * Real.sqrt (30^2 + 12^2)
  radius_sphere

theorem radius_of_sphere_is_approximately_correct : abs (radius_of_sphere_in_cylinder_cone - 4.84) < 0.01 :=
by
  sorry

end radius_of_sphere_is_approximately_correct_l2043_204378


namespace line_slope_intercept_product_l2043_204350

theorem line_slope_intercept_product :
  ∃ (m b : ℝ), (b = -1) ∧ ((1 - (m * -1 + b) = 0) ∧ (mb = m * b)) ∧ (mb = 2) :=
by sorry

end line_slope_intercept_product_l2043_204350


namespace joan_kittens_count_correct_l2043_204377

def joan_initial_kittens : Nat := 8
def kittens_from_friends : Nat := 2
def joan_total_kittens (initial: Nat) (added: Nat) : Nat := initial + added

theorem joan_kittens_count_correct : joan_total_kittens joan_initial_kittens kittens_from_friends = 10 := 
by
  sorry

end joan_kittens_count_correct_l2043_204377


namespace gcd_polynomials_l2043_204306

noncomputable def b : ℤ := sorry -- since b is given as an odd multiple of 997

theorem gcd_polynomials (h : ∃ k : ℤ, b = 997 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 41 * b + 101) (b + 17) = 1 :=
sorry

end gcd_polynomials_l2043_204306


namespace spring_membership_decrease_l2043_204394

theorem spring_membership_decrease (init_members : ℝ) (increase_percent : ℝ) (total_change_percent : ℝ) 
  (fall_members := init_members * (1 + increase_percent / 100)) 
  (spring_members := init_members * (1 + total_change_percent / 100)) :
  increase_percent = 8 → total_change_percent = -12.52 → 
  (fall_members - spring_members) / fall_members * 100 = 19 :=
by
  intros h1 h2
  -- The complicated proof goes here.
  sorry

end spring_membership_decrease_l2043_204394


namespace find_y_in_similar_triangles_l2043_204365

-- Define the variables and conditions of the problem
def is_similar (a1 b1 a2 b2 : ℚ) : Prop :=
  a1 / b1 = a2 / b2

-- Problem statement
theorem find_y_in_similar_triangles
  (a1 b1 a2 b2 : ℚ)
  (h1 : a1 = 15)
  (h2 : b1 = 12)
  (h3 : b2 = 10)
  (similarity_condition : is_similar a1 b1 a2 b2) :
  a2 = 25 / 2 :=
by
  rw [h1, h2, h3, is_similar] at similarity_condition
  sorry

end find_y_in_similar_triangles_l2043_204365


namespace average_tree_height_is_800_l2043_204396

def first_tree_height : ℕ := 1000
def other_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200
def total_height : ℕ := first_tree_height + other_tree_height + other_tree_height + last_tree_height
def average_height : ℕ := total_height / 4

theorem average_tree_height_is_800 :
  average_height = 800 := by
  sorry

end average_tree_height_is_800_l2043_204396


namespace no_solution_5x_plus_2_eq_17y_l2043_204315

theorem no_solution_5x_plus_2_eq_17y :
  ¬∃ (x y : ℕ), 5^x + 2 = 17^y :=
sorry

end no_solution_5x_plus_2_eq_17y_l2043_204315


namespace abs_inequality_solution_l2043_204372

theorem abs_inequality_solution (x : ℝ) (h : |x - 4| ≤ 6) : -2 ≤ x ∧ x ≤ 10 := 
sorry

end abs_inequality_solution_l2043_204372


namespace rectangle_side_ratio_l2043_204369

noncomputable def sin_30_deg := 1 / 2

theorem rectangle_side_ratio 
  (a b c : ℝ) 
  (h1 : a + b = 2 * c) 
  (h2 : a * b = (c ^ 2) / 2) :
  (a / b = 3 + 2 * Real.sqrt 2) ∨ (a / b = 3 - 2 * Real.sqrt 2) :=
by
  sorry

end rectangle_side_ratio_l2043_204369


namespace first_term_geometric_sequence_l2043_204362

theorem first_term_geometric_sequence (a r : ℕ) (h1 : r = 3) (h2 : a * r^4 = 81) : a = 1 :=
by
  sorry

end first_term_geometric_sequence_l2043_204362


namespace total_pages_read_correct_l2043_204375

-- Definition of the problem conditions
def first_week_books := 5
def first_week_book_pages := 300
def first_week_magazines := 3
def first_week_magazine_pages := 120
def first_week_newspapers := 2
def first_week_newspaper_pages := 50

def second_week_books := 2 * first_week_books
def second_week_book_pages := 350
def second_week_magazines := 4
def second_week_magazine_pages := 150
def second_week_newspapers := 1
def second_week_newspaper_pages := 60

def third_week_books := 3 * first_week_books
def third_week_book_pages := 400
def third_week_magazines := 5
def third_week_magazine_pages := 125
def third_week_newspapers := 1
def third_week_newspaper_pages := 70

-- Total pages read in each week
def first_week_total_pages : Nat :=
  (first_week_books * first_week_book_pages) +
  (first_week_magazines * first_week_magazine_pages) +
  (first_week_newspapers * first_week_newspaper_pages)

def second_week_total_pages : Nat :=
  (second_week_books * second_week_book_pages) +
  (second_week_magazines * second_week_magazine_pages) +
  (second_week_newspapers * second_week_newspaper_pages)

def third_week_total_pages : Nat :=
  (third_week_books * third_week_book_pages) +
  (third_week_magazines * third_week_magazine_pages) +
  (third_week_newspapers * third_week_newspaper_pages)

-- Grand total pages read over three weeks
def total_pages_read : Nat :=
  first_week_total_pages + second_week_total_pages + third_week_total_pages

-- Theorem statement to be proven
theorem total_pages_read_correct :
  total_pages_read = 12815 :=
by
  -- Proof will be provided here
  sorry

end total_pages_read_correct_l2043_204375


namespace jack_keeps_deers_weight_is_correct_l2043_204368

-- Define conditions
def monthly_hunt_count : Float := 7.5
def fraction_of_year_hunting_season : Float := 1 / 3
def deers_per_hunt : Float := 2.5
def weight_per_deer : Float := 600
def weight_kept_per_deer : Float := 0.65

-- Prove the total weight of the deer Jack keeps
theorem jack_keeps_deers_weight_is_correct :
  (12 * fraction_of_year_hunting_season) * monthly_hunt_count * deers_per_hunt * weight_per_deer * weight_kept_per_deer = 29250 :=
by
  sorry

end jack_keeps_deers_weight_is_correct_l2043_204368


namespace factor_expression_l2043_204388

theorem factor_expression (x : ℝ) : 72 * x ^ 5 - 162 * x ^ 9 = -18 * x ^ 5 * (9 * x ^ 4 - 4) :=
by
  sorry

end factor_expression_l2043_204388


namespace number_of_carbon_atoms_l2043_204304

/-- A proof to determine the number of carbon atoms in a compound given specific conditions
-/
theorem number_of_carbon_atoms
  (H_atoms : ℕ) (O_atoms : ℕ) (C_weight : ℕ) (H_weight : ℕ) (O_weight : ℕ) (Molecular_weight : ℕ) :
  H_atoms = 6 →
  O_atoms = 1 →
  C_weight = 12 →
  H_weight = 1 →
  O_weight = 16 →
  Molecular_weight = 58 →
  (Molecular_weight - (H_atoms * H_weight + O_atoms * O_weight)) / C_weight = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end number_of_carbon_atoms_l2043_204304


namespace jack_handing_in_amount_l2043_204307

theorem jack_handing_in_amount :
  let total_100_bills := 2 * 100
  let total_50_bills := 1 * 50
  let total_20_bills := 5 * 20
  let total_10_bills := 3 * 10
  let total_5_bills := 7 * 5
  let total_1_bills := 27 * 1
  let total_notes := total_100_bills + total_50_bills + total_20_bills + total_10_bills + total_5_bills + total_1_bills
  let amount_in_till := 300
  let amount_to_hand_in := total_notes - amount_in_till
  amount_to_hand_in = 142 := by
  sorry

end jack_handing_in_amount_l2043_204307


namespace mats_weaved_by_mat_weavers_l2043_204364

variable (M : ℕ)

theorem mats_weaved_by_mat_weavers :
  -- 10 mat-weavers can weave 25 mats in 10 days
  (10 * 10) * M / (4 * 4) = 25 / (10 / 4)  →
  -- number of mats woven by 4 mat-weavers in 4 days
  M = 4 :=
sorry

end mats_weaved_by_mat_weavers_l2043_204364


namespace airplane_speed_l2043_204328

noncomputable def distance : ℝ := 378.6   -- Distance in km
noncomputable def time : ℝ := 693.5       -- Time in seconds

noncomputable def altitude : ℝ := 10      -- Altitude in km
noncomputable def earth_radius : ℝ := 6370 -- Earth's radius in km

noncomputable def speed : ℝ := distance / time * 3600  -- Speed in km/h
noncomputable def adjusted_speed : ℝ := speed * (earth_radius + altitude) / earth_radius

noncomputable def min_distance : ℝ := 378.6 - 0.03     -- Minimum possible distance in km
noncomputable def max_distance : ℝ := 378.6 + 0.03     -- Maximum possible distance in km
noncomputable def min_time : ℝ := 693.5 - 1.5          -- Minimum possible time in s
noncomputable def max_time : ℝ := 693.5 + 1.5          -- Maximum possible time in s

noncomputable def max_speed : ℝ := max_distance / min_time * 3600 -- Max speed with uncertainty
noncomputable def min_speed : ℝ := min_distance / max_time * 3600 -- Min speed with uncertainty

theorem airplane_speed :
  1960 < adjusted_speed ∧ adjusted_speed < 1970 :=
by
  sorry

end airplane_speed_l2043_204328


namespace symmetric_line_eq_l2043_204330

theorem symmetric_line_eq : ∀ (x y : ℝ), (x - 2*y - 1 = 0) ↔ (2*x - y + 1 = 0) :=
by sorry

end symmetric_line_eq_l2043_204330


namespace retirement_total_l2043_204324

/-- A company retirement plan allows an employee to retire when their age plus years of employment total a specific number.
A female employee was hired in 1990 on her 32nd birthday. She could first be eligible to retire under this provision in 2009. -/
def required_total_age_years_of_employment : ℕ :=
  let hire_year := 1990
  let retirement_year := 2009
  let age_when_hired := 32
  let years_of_employment := retirement_year - hire_year
  let age_at_retirement := age_when_hired + years_of_employment
  age_at_retirement + years_of_employment

theorem retirement_total :
  required_total_age_years_of_employment = 70 :=
by
  sorry

end retirement_total_l2043_204324


namespace largest_possible_a_l2043_204308

theorem largest_possible_a :
  ∀ (a b c d : ℕ), a < 3 * b ∧ b < 4 * c ∧ c < 5 * d ∧ d < 80 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d → a ≤ 4724 := by
  sorry

end largest_possible_a_l2043_204308


namespace cost_price_correct_l2043_204353

variables (sp : ℕ) (profitPerMeter : ℕ) (metersSold : ℕ)

def total_profit (profitPerMeter metersSold : ℕ) : ℕ := profitPerMeter * metersSold
def total_cost_price (sp total_profit : ℕ) : ℕ := sp - total_profit
def cost_price_per_meter (total_cost_price metersSold : ℕ) : ℕ := total_cost_price / metersSold

theorem cost_price_correct (h1 : sp = 8925) (h2 : profitPerMeter = 10) (h3 : metersSold = 85) :
  cost_price_per_meter (total_cost_price sp (total_profit profitPerMeter metersSold)) metersSold = 95 :=
by
  rw [h1, h2, h3];
  sorry

end cost_price_correct_l2043_204353


namespace smallest_m_l2043_204363

theorem smallest_m (m : ℕ) (p q : ℤ) (h_eq : 10*(p:ℤ)^2 - m*(p:ℤ) + 360 = 0) (h_cond : q = 2 * p) :
  p * q = 36 → 3 * p + 3 * q = m → m = 90 :=
by sorry

end smallest_m_l2043_204363


namespace compute_g_x_h_l2043_204386

def g (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 4

theorem compute_g_x_h (x h : ℝ) : 
  g (x + h) - g x = h * (12 * x + 6 * h - 3) := by
  sorry

end compute_g_x_h_l2043_204386


namespace find_x_for_g_inv_eq_3_l2043_204352

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 5

-- State the theorem
theorem find_x_for_g_inv_eq_3 : ∃ x : ℝ, g x = 113 :=
by
  exists 3
  unfold g
  norm_num

end find_x_for_g_inv_eq_3_l2043_204352


namespace smallest_n_divisible_31_l2043_204301

theorem smallest_n_divisible_31 (n : ℕ) : 31 ∣ (5 ^ n + n) → n = 30 :=
by
  sorry

end smallest_n_divisible_31_l2043_204301


namespace max_vertex_sum_l2043_204343

theorem max_vertex_sum
  (a U : ℤ)
  (hU : U ≠ 0)
  (hA : 0 = a * 0 * (0 - 3 * U))
  (hB : 0 = a * (3 * U) * ((3 * U) - 3 * U))
  (hC : 12 = a * (3 * U - 1) * ((3 * U - 1) - 3 * U))
  : ∃ N : ℝ, N = (3 * U) / 2 - (9 * a * U^2) / 4 ∧ N ≤ 17.75 :=
by sorry

end max_vertex_sum_l2043_204343


namespace smallest_brownie_pan_size_l2043_204311

theorem smallest_brownie_pan_size :
  ∃ s : ℕ, (s - 2) ^ 2 = 4 * s - 4 ∧ ∀ t : ℕ, (t - 2) ^ 2 = 4 * t - 4 → s <= t :=
by
  sorry

end smallest_brownie_pan_size_l2043_204311


namespace typeA_cloth_typeB_cloth_typeC_cloth_l2043_204356

section ClothPrices

variables (CPA CPB CPC : ℝ)

theorem typeA_cloth :
  (300 * CPA * 0.90 = 9000) → CPA = 33.33 :=
by
  intro hCPA
  sorry

theorem typeB_cloth :
  (250 * CPB * 1.05 = 7000) → CPB = 26.67 :=
by
  intro hCPB
  sorry

theorem typeC_cloth :
  (400 * (CPC + 8) = 12000) → CPC = 22 :=
by
  intro hCPC
  sorry

end ClothPrices

end typeA_cloth_typeB_cloth_typeC_cloth_l2043_204356


namespace largest_p_q_sum_l2043_204331

theorem largest_p_q_sum 
  (p q : ℝ)
  (A := (p, q))
  (B := (12, 19))
  (C := (23, 20))
  (area_ABC : ℝ := 70)
  (slope_median : ℝ := -5)
  (midpoint_BC := ((12 + 23) / 2, (19 + 20) / 2))
  (eq_median : (q - midpoint_BC.2) = slope_median * (p - midpoint_BC.1))
  (area_eq : 140 = 240 - 437 - 20 * p + 23 * q + 19 * p - 12 * q) :
  p + q ≤ 47 :=
sorry

end largest_p_q_sum_l2043_204331


namespace area_of_triangle_AMN_l2043_204326

theorem area_of_triangle_AMN
  (α : ℝ) -- Angle at vertex A
  (S : ℝ) -- Area of triangle ABC
  (area_AMN_eq : ∀ (α : ℝ) (S : ℝ), ∃ (area_AMN : ℝ), area_AMN = S * (Real.cos α)^2) :
  ∃ area_AMN, area_AMN = S * (Real.cos α)^2 := by
  sorry

end area_of_triangle_AMN_l2043_204326


namespace street_lights_per_side_l2043_204312

theorem street_lights_per_side
  (neighborhoods : ℕ)
  (roads_per_neighborhood : ℕ)
  (total_street_lights : ℕ)
  (total_neighborhoods : neighborhoods = 10)
  (roads_in_each_neighborhood : roads_per_neighborhood = 4)
  (street_lights_in_town : total_street_lights = 20000) :
  (total_street_lights / (neighborhoods * roads_per_neighborhood * 2) = 250) :=
by
  sorry

end street_lights_per_side_l2043_204312


namespace radius_of_inner_tangent_circle_l2043_204339

theorem radius_of_inner_tangent_circle (side_length : ℝ) (num_semicircles_per_side : ℝ) (semicircle_radius : ℝ)
  (h_side_length : side_length = 4) (h_num_semicircles_per_side : num_semicircles_per_side = 3) 
  (h_semicircle_radius : semicircle_radius = side_length / (2 * num_semicircles_per_side)) :
  ∃ (inner_circle_radius : ℝ), inner_circle_radius = 7 / 6 :=
by
  sorry

end radius_of_inner_tangent_circle_l2043_204339


namespace triangle_vertex_y_coordinate_l2043_204357

theorem triangle_vertex_y_coordinate (h : ℝ) :
  let A := (0, 0)
  let C := (8, 0)
  let B := (4, h)
  (1/2) * (8) * h = 32 → h = 8 :=
by
  intro h
  intro H
  sorry

end triangle_vertex_y_coordinate_l2043_204357


namespace find_S6_l2043_204398

noncomputable def geometric_series_nth_term (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n - 1)

noncomputable def geometric_series_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

variables (a2 q : ℝ)

-- Conditions
axiom a_n_pos : ∀ n, n > 0 → geometric_series_nth_term a2 q n > 0
axiom q_gt_one : q > 1
axiom condition1 : geometric_series_nth_term a2 q 3 + geometric_series_nth_term a2 q 5 = 20
axiom condition2 : geometric_series_nth_term a2 q 2 * geometric_series_nth_term a2 q 6 = 64

-- Question/statement of the theorem
theorem find_S6 : geometric_series_sum 1 q 6 = 63 :=
  sorry

end find_S6_l2043_204398


namespace even_three_digit_numbers_count_l2043_204314

theorem even_three_digit_numbers_count :
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  count = 18 :=
by
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  show count = 18
  sorry

end even_three_digit_numbers_count_l2043_204314


namespace point_and_sum_of_coordinates_l2043_204360

-- Definitions
def point_on_graph_of_g_over_3 (g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = (g p.1) / 3

def point_on_graph_of_inv_g_over_3 (g : ℝ → ℝ) (q : ℝ × ℝ) : Prop :=
  q.2 = (g⁻¹ q.1) / 3

-- Main statement
theorem point_and_sum_of_coordinates {g : ℝ → ℝ} (h : point_on_graph_of_g_over_3 g (2, 3)) :
  point_on_graph_of_inv_g_over_3 g (9, 2 / 3) ∧ (9 + 2 / 3 = 29 / 3) :=
by
  sorry

end point_and_sum_of_coordinates_l2043_204360


namespace P_at_6_l2043_204381

noncomputable def P (x : ℕ) : ℚ := (720 * x) / (x^2 - 1)

theorem P_at_6 : P 6 = 48 :=
by
  -- Definitions and conditions derived from the problem.
  -- Establishing given condition and deriving P(6) value.
  sorry

end P_at_6_l2043_204381


namespace somu_age_relation_l2043_204366

-- Somu’s present age (S) is 20 years
def somu_present_age : ℕ := 20

-- Somu’s age is one-third of his father’s age (F)
def father_present_age : ℕ := 3 * somu_present_age

-- Proof statement: Y years ago, Somu's age was one-fifth of his father's age
theorem somu_age_relation : ∃ (Y : ℕ), somu_present_age - Y = (1 : ℕ) / 5 * (father_present_age - Y) ∧ Y = 10 :=
by
  have h := "" -- Placeholder for the proof steps
  sorry

end somu_age_relation_l2043_204366


namespace g_symmetric_l2043_204390

noncomputable def g (x : ℝ) : ℝ := |⌊2 * x⌋| - |⌊2 - 2 * x⌋|

theorem g_symmetric : ∀ x : ℝ, g x = g (1 - x) := by
  sorry

end g_symmetric_l2043_204390


namespace train_speed_45_kmph_l2043_204387

variable (length_train length_bridge time_passed : ℕ)

def total_distance (length_train length_bridge : ℕ) : ℕ :=
  length_train + length_bridge

def speed_m_per_s (length_train length_bridge time_passed : ℕ) : ℚ :=
  (total_distance length_train length_bridge) / time_passed

def speed_km_per_h (length_train length_bridge time_passed : ℕ) : ℚ :=
  (speed_m_per_s length_train length_bridge time_passed) * 3.6

theorem train_speed_45_kmph :
  length_train = 360 → length_bridge = 140 → time_passed = 40 → speed_km_per_h length_train length_bridge time_passed = 45 := 
by
  sorry

end train_speed_45_kmph_l2043_204387


namespace remainder_of_3_pow_2023_mod_7_l2043_204327

theorem remainder_of_3_pow_2023_mod_7 : (3^2023) % 7 = 3 :=
by
  sorry

end remainder_of_3_pow_2023_mod_7_l2043_204327


namespace initial_bags_count_l2043_204376

theorem initial_bags_count
  (points_per_bag : ℕ)
  (non_recycled_bags : ℕ)
  (total_possible_points : ℕ)
  (points_earned : ℕ)
  (B : ℕ)
  (h1 : points_per_bag = 5)
  (h2 : non_recycled_bags = 2)
  (h3 : total_possible_points = 45)
  (h4 : points_earned = 5 * (B - non_recycled_bags))
  : B = 11 :=
by {
  sorry
}

end initial_bags_count_l2043_204376


namespace max_value_of_quadratic_l2043_204383

theorem max_value_of_quadratic : ∃ x : ℝ, (∀ y : ℝ, (-3 * y^2 + 9 * y - 1) ≤ (-3 * (3/2)^2 + 9 * (3/2) - 1)) ∧ x = 3/2 :=
by
  sorry

end max_value_of_quadratic_l2043_204383


namespace pow_mod_79_l2043_204371

theorem pow_mod_79 (a : ℕ) (h : a = 7) : a^79 % 11 = 6 := by
  sorry

end pow_mod_79_l2043_204371


namespace muffins_equation_l2043_204355

def remaining_muffins : ℕ := 48
def total_muffins : ℕ := 83
def initially_baked_muffins : ℕ := 35

theorem muffins_equation : initially_baked_muffins + remaining_muffins = total_muffins :=
  by
    -- Skipping the proof here
    sorry

end muffins_equation_l2043_204355


namespace breakfast_time_correct_l2043_204345

noncomputable def breakfast_time_calc (x : ℚ) : ℚ :=
  (7 * 60) + (300 / 13)

noncomputable def coffee_time_calc (y : ℚ) : ℚ :=
  (7 * 60) + (420 / 11)

noncomputable def total_breakfast_time : ℚ :=
  coffee_time_calc ((420 : ℚ) / 11) - breakfast_time_calc ((300 : ℚ) / 13)

theorem breakfast_time_correct :
  total_breakfast_time = 15 + (6 / 60) :=
by
  sorry

end breakfast_time_correct_l2043_204345
