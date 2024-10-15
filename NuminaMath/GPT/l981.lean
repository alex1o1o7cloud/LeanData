import Mathlib

namespace NUMINAMATH_GPT_inequality_abc_l981_98156

variable {a b c : ℝ}

theorem inequality_abc
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l981_98156


namespace NUMINAMATH_GPT_incorrect_mode_l981_98137

theorem incorrect_mode (data : List ℕ) (hdata : data = [1, 2, 4, 3, 5]) : ¬ (∃ mode, mode = 5 ∧ (data.count mode > 1)) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_mode_l981_98137


namespace NUMINAMATH_GPT_veranda_width_l981_98165

theorem veranda_width (l w : ℝ) (room_area veranda_area : ℝ) (h1 : l = 20) (h2 : w = 12) (h3 : veranda_area = 144) : 
  ∃ w_v : ℝ, (l + 2 * w_v) * (w + 2 * w_v) - l * w = veranda_area ∧ w_v = 2 := 
by
  sorry

end NUMINAMATH_GPT_veranda_width_l981_98165


namespace NUMINAMATH_GPT_problem_statement_l981_98160

def assoc_number (x : ℚ) : ℚ :=
  if x >= 0 then 2 * x - 1 else -2 * x + 1

theorem problem_statement (a b : ℚ) (ha : a > 0) (hb : b < 0) (hab : assoc_number a = assoc_number b) :
  (a + b)^2 - 2 * a - 2 * b = -1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l981_98160


namespace NUMINAMATH_GPT_original_price_of_sarees_l981_98188

theorem original_price_of_sarees (P : ℝ) (h : 0.95 * 0.80 * P = 456) : P = 600 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_sarees_l981_98188


namespace NUMINAMATH_GPT_obtuse_triangle_side_range_l981_98148

theorem obtuse_triangle_side_range
  (a : ℝ)
  (h1 : a > 0)
  (h2 : (a + 4)^2 > a^2 + (a + 2)^2)
  (h3 : (a + 2)^2 + (a + 4)^2 < a^2) : 
  2 < a ∧ a < 6 := 
sorry

end NUMINAMATH_GPT_obtuse_triangle_side_range_l981_98148


namespace NUMINAMATH_GPT_find_range_of_a_l981_98142

-- Definitions
def is_decreasing_function (a : ℝ) : Prop :=
  0 < a ∧ a < 1

def no_real_roots_of_poly (a : ℝ) : Prop :=
  4 * a < 1

def problem_statement (a : ℝ) : Prop :=
  (is_decreasing_function a ∨ no_real_roots_of_poly a) ∧ ¬ (is_decreasing_function a ∧ no_real_roots_of_poly a)

-- Main theorem
theorem find_range_of_a (a : ℝ) : problem_statement a ↔ (0 < a ∧ a ≤ 1 / 4) ∨ (a ≥ 1) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_range_of_a_l981_98142


namespace NUMINAMATH_GPT_inequality_negatives_l981_98149

theorem inequality_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : (b / a) < 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_negatives_l981_98149


namespace NUMINAMATH_GPT_solution_set_of_inequality_l981_98179

theorem solution_set_of_inequality: 
  {x : ℝ | (2 * x - 1) / x < 1} = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l981_98179


namespace NUMINAMATH_GPT_tiffany_lives_next_level_l981_98105

theorem tiffany_lives_next_level (L1 L2 L3 : ℝ)
    (h1 : L1 = 43.0)
    (h2 : L2 = 14.0)
    (h3 : L3 = 84.0) :
    L3 - (L1 + L2) = 27 :=
by
  rw [h1, h2, h3]
  -- The proof is skipped with "sorry"
  sorry

end NUMINAMATH_GPT_tiffany_lives_next_level_l981_98105


namespace NUMINAMATH_GPT_small_gate_width_l981_98159

-- Bob's garden dimensions
def garden_length : ℝ := 225
def garden_width : ℝ := 125

-- Total fencing needed, including the gates
def total_fencing : ℝ := 687

-- Width of the large gate
def large_gate_width : ℝ := 10

-- Perimeter of the garden without gates
def garden_perimeter : ℝ := 2 * (garden_length + garden_width)

-- Width of the small gate
theorem small_gate_width :
  2 * (garden_length + garden_width) + small_gate + large_gate_width = total_fencing → small_gate = 3 :=
by
  sorry

end NUMINAMATH_GPT_small_gate_width_l981_98159


namespace NUMINAMATH_GPT_count_integers_congruent_to_7_mod_13_l981_98120

theorem count_integers_congruent_to_7_mod_13:
  (∃ S : Finset ℕ, S.card = 154 ∧ ∀ n ∈ S, n < 2000 ∧ n % 13 = 7) :=
by
  sorry

end NUMINAMATH_GPT_count_integers_congruent_to_7_mod_13_l981_98120


namespace NUMINAMATH_GPT_chris_newspapers_l981_98136

theorem chris_newspapers (C L : ℕ) 
  (h1 : L = C + 23) 
  (h2 : C + L = 65) : 
  C = 21 := 
by 
  sorry

end NUMINAMATH_GPT_chris_newspapers_l981_98136


namespace NUMINAMATH_GPT_find_c_l981_98181

def f (x : ℤ) : ℤ := x - 2

def F (x y : ℤ) : ℤ := y^2 + x

theorem find_c : ∃ c, c = F 3 (f 16) ∧ c = 199 :=
by
  use F 3 (f 16)
  sorry

end NUMINAMATH_GPT_find_c_l981_98181


namespace NUMINAMATH_GPT_additional_money_spent_on_dvds_correct_l981_98155

def initial_money : ℕ := 320
def spent_on_books : ℕ := initial_money / 4 + 10
def remaining_after_books : ℕ := initial_money - spent_on_books
def spent_on_dvds_portion : ℕ := 2 * remaining_after_books / 5
def remaining_after_dvds : ℕ := 130
def total_spent_on_dvds : ℕ := remaining_after_books - remaining_after_dvds
def additional_spent_on_dvds : ℕ := total_spent_on_dvds - spent_on_dvds_portion

theorem additional_money_spent_on_dvds_correct : additional_spent_on_dvds = 8 :=
by
  sorry

end NUMINAMATH_GPT_additional_money_spent_on_dvds_correct_l981_98155


namespace NUMINAMATH_GPT_S2014_value_l981_98131

variable (S : ℕ → ℤ) -- S_n represents sum of the first n terms of the arithmetic sequence
variable (a1 : ℤ) -- First term of the arithmetic sequence
variable (d : ℤ) -- Common difference of the arithmetic sequence

-- Given conditions
variable (h1 : a1 = -2016)
variable (h2 : (S 2016) / 2016 - (S 2010) / 2010 = 6)

-- The proof problem
theorem S2014_value :
  S 2014 = -6042 :=
sorry -- Proof omitted

end NUMINAMATH_GPT_S2014_value_l981_98131


namespace NUMINAMATH_GPT_smallest_k_power_l981_98129

theorem smallest_k_power (k : ℕ) (hk : ∀ m : ℕ, m < 14 → 7^m ≤ 4^19) : 7^14 > 4^19 :=
sorry

end NUMINAMATH_GPT_smallest_k_power_l981_98129


namespace NUMINAMATH_GPT_regular_tiles_area_l981_98177

theorem regular_tiles_area (L W : ℝ) (T : ℝ) (h₁ : 1/3 * T * (3 * L * W) + 2/3 * T * (L * W) = 385) : 
  (2/3 * T * (L * W) = 154) :=
by
  sorry

end NUMINAMATH_GPT_regular_tiles_area_l981_98177


namespace NUMINAMATH_GPT_sum_first_10_terms_abs_a_n_l981_98114

noncomputable def a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else 3 * n - 7

def abs_a_n (n : ℕ) : ℤ :=
  if n = 1 ∨ n = 2 then -3 * n + 7 else 3 * n - 7

def sum_abs_a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else List.sum (List.map abs_a_n (List.range n))

theorem sum_first_10_terms_abs_a_n : sum_abs_a_n 10 = 105 := 
  sorry

end NUMINAMATH_GPT_sum_first_10_terms_abs_a_n_l981_98114


namespace NUMINAMATH_GPT_interval_of_decrease_for_f_x_plus_1_l981_98157

def f_prime (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem interval_of_decrease_for_f_x_plus_1 : 
  ∀ x, (f_prime (x + 1) < 0 ↔ 0 < x ∧ x < 2) :=
by 
  intro x
  sorry

end NUMINAMATH_GPT_interval_of_decrease_for_f_x_plus_1_l981_98157


namespace NUMINAMATH_GPT_clive_change_l981_98192

theorem clive_change (money : ℝ) (olives_needed : ℕ) (olives_per_jar : ℕ) (price_per_jar : ℝ) : 
  (money = 10) → 
  (olives_needed = 80) → 
  (olives_per_jar = 20) →
  (price_per_jar = 1.5) →
  money - (olives_needed / olives_per_jar) * price_per_jar = 4 := by
  sorry

end NUMINAMATH_GPT_clive_change_l981_98192


namespace NUMINAMATH_GPT_speed_of_current_l981_98110
  
  theorem speed_of_current (v c : ℝ)
    (h1 : 64 = (v + c) * 8)
    (h2 : 24 = (v - c) * 8) :
    c = 2.5 :=
  by {
    sorry
  }
  
end NUMINAMATH_GPT_speed_of_current_l981_98110


namespace NUMINAMATH_GPT_problem_A_value_l981_98106

theorem problem_A_value (x y A : ℝ) (h : (x + 2 * y) ^ 2 = (x - 2 * y) ^ 2 + A) : A = 8 * x * y :=
by {
    sorry
}

end NUMINAMATH_GPT_problem_A_value_l981_98106


namespace NUMINAMATH_GPT_order_of_mnpq_l981_98147

theorem order_of_mnpq 
(m n p q : ℝ) 
(h1 : m < n)
(h2 : p < q)
(h3 : (p - m) * (p - n) < 0)
(h4 : (q - m) * (q - n) < 0) 
: m < p ∧ p < q ∧ q < n := 
by
  sorry

end NUMINAMATH_GPT_order_of_mnpq_l981_98147


namespace NUMINAMATH_GPT_sum_of_integers_l981_98153

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 6) (h2 : x * y = 112) (h3 : x > y) : x + y = 22 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_l981_98153


namespace NUMINAMATH_GPT_number_of_non_congruent_triangles_with_perimeter_20_l981_98170

theorem number_of_non_congruent_triangles_with_perimeter_20 :
  ∃ T : Finset (Finset ℕ), 
    (∀ t ∈ T, ∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 20 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧
    T.card = 14 :=
by
  sorry

end NUMINAMATH_GPT_number_of_non_congruent_triangles_with_perimeter_20_l981_98170


namespace NUMINAMATH_GPT_segment_outside_spheres_l981_98103

noncomputable def fraction_outside_spheres (α : ℝ) : ℝ :=
  (1 - (Real.cos (α / 2))^2) / (1 + (Real.cos (α / 2))^2)

theorem segment_outside_spheres (R α : ℝ) (hR : R > 0) (hα : 0 < α ∧ α < 2 * Real.pi) :
  fraction_outside_spheres α = (1 - Real.cos (α / 2)^2) / (1 + (Real.cos (α / 2))^2) :=
  by sorry

end NUMINAMATH_GPT_segment_outside_spheres_l981_98103


namespace NUMINAMATH_GPT_chris_score_l981_98191

variable (s g c : ℕ)

theorem chris_score  (h1 : s = g + 60) (h2 : (s + g) / 2 = 110) (h3 : c = 110 * 120 / 100) :
  c = 132 := by
  sorry

end NUMINAMATH_GPT_chris_score_l981_98191


namespace NUMINAMATH_GPT_chocolate_bars_in_large_box_l981_98171

theorem chocolate_bars_in_large_box
  (number_of_small_boxes : ℕ)
  (chocolate_bars_per_box : ℕ)
  (h1 : number_of_small_boxes = 21)
  (h2 : chocolate_bars_per_box = 25) :
  number_of_small_boxes * chocolate_bars_per_box = 525 :=
by {
  sorry
}

end NUMINAMATH_GPT_chocolate_bars_in_large_box_l981_98171


namespace NUMINAMATH_GPT_multiple_of_shirt_cost_l981_98162

theorem multiple_of_shirt_cost (S C M : ℕ) (h1 : S = 97) (h2 : C = 300 - S)
  (h3 : C = M * S + 9) : M = 2 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_multiple_of_shirt_cost_l981_98162


namespace NUMINAMATH_GPT_valid_points_region_equivalence_l981_98173

def valid_point (x y : ℝ) : Prop :=
  |x - 1| + |x + 1| + |2 * y| ≤ 4

def region1 (x y : ℝ) : Prop :=
  x ≤ -1 ∧ y ≤ x + 2 ∧ y ≥ -x - 2

def region2 (x y : ℝ) : Prop :=
  -1 < x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1

def region3 (x y : ℝ) : Prop :=
  1 < x ∧ y ≤ 2 - x ∧ y ≥ x - 2

def solution_region (x y : ℝ) : Prop :=
  region1 x y ∨ region2 x y ∨ region3 x y

theorem valid_points_region_equivalence : 
  ∀ x y : ℝ, valid_point x y ↔ solution_region x y :=
sorry

end NUMINAMATH_GPT_valid_points_region_equivalence_l981_98173


namespace NUMINAMATH_GPT_sum_of_coefficients_of_poly_l981_98122

-- Define the polynomial
def poly (x y : ℕ) := (2 * x + 3 * y) ^ 12

-- Define the sum of coefficients
def sum_of_coefficients := poly 1 1

-- The theorem stating the result
theorem sum_of_coefficients_of_poly : sum_of_coefficients = 244140625 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_of_poly_l981_98122


namespace NUMINAMATH_GPT_inverse_proportion_inequality_l981_98172

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_inequality
  (h1 : y1 = 6 / x1)
  (h2 : y2 = 6 / x2)
  (hx : x1 < 0 ∧ 0 < x2) :
  y1 < y2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_inequality_l981_98172


namespace NUMINAMATH_GPT_percentage_cut_in_magazine_budget_l981_98185

noncomputable def magazine_budget_cut (original_budget : ℕ) (cut_amount : ℕ) : ℕ :=
  (cut_amount * 100) / original_budget

theorem percentage_cut_in_magazine_budget : 
  magazine_budget_cut 940 282 = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_cut_in_magazine_budget_l981_98185


namespace NUMINAMATH_GPT_max_f_value_l981_98118

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x / (x^2 + m)

theorem max_f_value (m : ℝ) : 
  (m > 1) ↔ (∀ x : ℝ, f x m < 1) ∧ ¬((∀ x : ℝ, f x m < 1) → (m > 1)) :=
by
  sorry

end NUMINAMATH_GPT_max_f_value_l981_98118


namespace NUMINAMATH_GPT_first_term_geometric_progression_l981_98112

theorem first_term_geometric_progression (a r : ℝ) 
  (h1 : a / (1 - r) = 6)
  (h2 : a + a * r = 9 / 2) :
  a = 3 ∨ a = 9 := 
sorry -- Proof omitted

end NUMINAMATH_GPT_first_term_geometric_progression_l981_98112


namespace NUMINAMATH_GPT_find_salary_of_Thomas_l981_98104

-- Declare the variables representing the salaries of Raj, Roshan, and Thomas
variables (R S T : ℝ)

-- Given conditions as definitions
def avg_salary_Raj_Roshan : Prop := (R + S) / 2 = 4000
def avg_salary_Raj_Roshan_Thomas : Prop := (R + S + T) / 3 = 5000

-- Stating the theorem
theorem find_salary_of_Thomas
  (h1 : avg_salary_Raj_Roshan R S)
  (h2 : avg_salary_Raj_Roshan_Thomas R S T) : T = 7000 :=
by
  sorry

end NUMINAMATH_GPT_find_salary_of_Thomas_l981_98104


namespace NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l981_98152

def a : ℚ := 2 / 3
def d : ℚ := 2 / 3

theorem tenth_term_arithmetic_sequence : 
  let a := 2 / 3
  let d := 2 / 3
  let n := 10
  a + (n - 1) * d = 20 / 3 := by
  sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l981_98152


namespace NUMINAMATH_GPT_find_x_l981_98140

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (hxy : x + y + x * y = 80) : x = 26 :=
sorry

end NUMINAMATH_GPT_find_x_l981_98140


namespace NUMINAMATH_GPT_m_minus_n_is_square_l981_98163

theorem m_minus_n_is_square (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 2001 * m ^ 2 + m = 2002 * n ^ 2 + n) : ∃ k : ℕ, m - n = k ^ 2 :=
sorry

end NUMINAMATH_GPT_m_minus_n_is_square_l981_98163


namespace NUMINAMATH_GPT_goods_train_pass_time_l981_98119

theorem goods_train_pass_time 
  (speed_mans_train_kmph : ℝ) (speed_goods_train_kmph : ℝ) (length_goods_train_m : ℝ) :
  speed_mans_train_kmph = 20 → 
  speed_goods_train_kmph = 92 → 
  length_goods_train_m = 280 → 
  abs ((length_goods_train_m / ((speed_mans_train_kmph + speed_goods_train_kmph) * 1000 / 3600)) - 8.99) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_goods_train_pass_time_l981_98119


namespace NUMINAMATH_GPT_gcd_1248_585_l981_98109

theorem gcd_1248_585 : Nat.gcd 1248 585 = 39 := by
  sorry

end NUMINAMATH_GPT_gcd_1248_585_l981_98109


namespace NUMINAMATH_GPT_find_p_q_l981_98151

noncomputable def cubicFunction (p q : ℝ) (x : ℂ) : ℂ :=
  2 * x^3 + p * x^2 + q * x

theorem find_p_q (p q : ℝ) :
  cubicFunction p q (2 * Complex.I - 3) = 0 ∧ 
  cubicFunction p q (-2 * Complex.I - 3) = 0 → 
  p = 12 ∧ q = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_l981_98151


namespace NUMINAMATH_GPT_prove_nabla_squared_l981_98143

theorem prove_nabla_squared:
  ∃ (odot nabla : ℕ), odot < 20 ∧ nabla < 20 ∧ odot ≠ nabla ∧
  (nabla * nabla * odot = nabla) ∧ (nabla * nabla = 64) :=
by
  sorry

end NUMINAMATH_GPT_prove_nabla_squared_l981_98143


namespace NUMINAMATH_GPT_jack_leftover_money_l981_98115

theorem jack_leftover_money :
  let saved_money_base8 : ℕ := 3 * 8^3 + 7 * 8^2 + 7 * 8^1 + 7 * 8^0
  let ticket_cost_base10 : ℕ := 1200
  saved_money_base8 - ticket_cost_base10 = 847 :=
by
  let saved_money_base8 := 3 * 8^3 + 7 * 8^2 + 7 * 8^1 + 7 * 8^0
  let ticket_cost_base10 := 1200
  show saved_money_base8 - ticket_cost_base10 = 847
  sorry

end NUMINAMATH_GPT_jack_leftover_money_l981_98115


namespace NUMINAMATH_GPT_original_price_of_racket_l981_98138

theorem original_price_of_racket (P : ℝ) (h : (3 / 2) * P = 90) : P = 60 :=
sorry

end NUMINAMATH_GPT_original_price_of_racket_l981_98138


namespace NUMINAMATH_GPT_arrange_pencils_l981_98121

-- Definition to express the concept of pencil touching
def pencil_touches (a b : Type) : Prop := sorry

-- Assume we have six pencils represented as 6 distinct variables.
variables (A B C D E F : Type)

-- Main theorem statement
theorem arrange_pencils :
  ∃ (A B C D E F : Type), (pencil_touches A B) ∧ (pencil_touches A C) ∧ 
  (pencil_touches A D) ∧ (pencil_touches A E) ∧ (pencil_touches A F) ∧ 
  (pencil_touches B C) ∧ (pencil_touches B D) ∧ (pencil_touches B E) ∧ 
  (pencil_touches B F) ∧ (pencil_touches C D) ∧ (pencil_touches C E) ∧ 
  (pencil_touches C F) ∧ (pencil_touches D E) ∧ (pencil_touches D F) ∧ 
  (pencil_touches E F) :=
sorry

end NUMINAMATH_GPT_arrange_pencils_l981_98121


namespace NUMINAMATH_GPT_f_5times_8_eq_l981_98182

def f (x : ℚ) : ℚ := 1 / x ^ 2

theorem f_5times_8_eq :
  f (f (f (f (f (8 : ℚ))))) = 1 / 79228162514264337593543950336 := 
  by
    sorry

end NUMINAMATH_GPT_f_5times_8_eq_l981_98182


namespace NUMINAMATH_GPT_tip_percentage_l981_98101

theorem tip_percentage
  (original_bill : ℝ)
  (shared_per_person : ℝ)
  (num_people : ℕ)
  (total_shared : ℝ)
  (tip_percent : ℝ)
  (h1 : original_bill = 139.0)
  (h2 : shared_per_person = 50.97)
  (h3 : num_people = 3)
  (h4 : total_shared = shared_per_person * num_people)
  (h5 : total_shared - original_bill = 13.91) :
  tip_percent = 13.91 / 139.0 * 100 := 
sorry

end NUMINAMATH_GPT_tip_percentage_l981_98101


namespace NUMINAMATH_GPT_find_b_neg_l981_98123

noncomputable def h (x : ℝ) : ℝ := if x ≤ 0 then -x else 3 * x - 50

theorem find_b_neg (b : ℝ) (h_neg_b : b < 0) : 
  h (h (h 15)) = h (h (h b)) → b = - (55 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_b_neg_l981_98123


namespace NUMINAMATH_GPT_best_fit_of_regression_model_l981_98117

-- Define the context of regression analysis and the coefficient of determination
def regression_analysis : Type := sorry
def coefficient_of_determination (r : regression_analysis) : ℝ := sorry

-- Definitions of each option for clarity in our context
def A (r : regression_analysis) : Prop := sorry -- the linear relationship is stronger
def B (r : regression_analysis) : Prop := sorry -- the linear relationship is weaker
def C (r : regression_analysis) : Prop := sorry -- better fit of the model
def D (r : regression_analysis) : Prop := sorry -- worse fit of the model

-- The formal statement we need to prove
theorem best_fit_of_regression_model (r : regression_analysis) (R2 : ℝ) (h1 : coefficient_of_determination r = R2) (h2 : R2 = 1) : C r :=
by
  sorry

end NUMINAMATH_GPT_best_fit_of_regression_model_l981_98117


namespace NUMINAMATH_GPT_batsman_average_after_12th_inning_l981_98193

variable (A : ℕ) (total_balls_faced : ℕ)

theorem batsman_average_after_12th_inning 
  (h1 : ∃ A, ∀ total_runs, total_runs = 11 * A)
  (h2 : ∃ A, ∀ total_runs_new, total_runs_new = 12 * (A + 4) ∧ total_runs_new - 60 = 11 * A)
  (h3 : 8 * 4 ≤ 60)
  (h4 : 6000 / total_balls_faced ≥ 130) 
  : (A + 4 = 16) :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_12th_inning_l981_98193


namespace NUMINAMATH_GPT_abc_value_l981_98135

variables (a b c : ℂ)

theorem abc_value :
  (a * b + 4 * b = -16) →
  (b * c + 4 * c = -16) →
  (c * a + 4 * a = -16) →
  a * b * c = 64 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_abc_value_l981_98135


namespace NUMINAMATH_GPT_abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l981_98169

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end NUMINAMATH_GPT_abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l981_98169


namespace NUMINAMATH_GPT_general_term_arithmetic_sequence_l981_98154

theorem general_term_arithmetic_sequence {a : ℕ → ℕ} (d : ℕ) (h_d : d ≠ 0)
  (h1 : a 3 + a 10 = 15)
  (h2 : (a 2 + d) * (a 2 + 10 * d) = (a 2 + 4 * d) * (a 2 + d))
  : ∀ n, a n = n + 1 :=
sorry

end NUMINAMATH_GPT_general_term_arithmetic_sequence_l981_98154


namespace NUMINAMATH_GPT_train_speed_160m_6sec_l981_98183

noncomputable def train_speed (distance time : ℕ) : ℚ :=
(distance : ℚ) / (time : ℚ)

theorem train_speed_160m_6sec : train_speed 160 6 = 26.67 :=
by
  simp [train_speed]
  norm_num
  sorry

end NUMINAMATH_GPT_train_speed_160m_6sec_l981_98183


namespace NUMINAMATH_GPT_simplify_complex_expression_l981_98190

theorem simplify_complex_expression (i : ℂ) (h : i^2 = -1) : 
  7 * (4 - 2 * i) + 4 * i * (7 - 3 * i) = 40 + 14 * i :=
by
  sorry

end NUMINAMATH_GPT_simplify_complex_expression_l981_98190


namespace NUMINAMATH_GPT_increase_a1_intervals_of_increase_l981_98199

noncomputable def f (x a : ℝ) : ℝ := x - (a + 1) * Real.log x - a / x

-- Prove that when a = 1, f(x) has no extreme points (i.e., it is monotonically increasing in (0, +∞))
theorem increase_a1 : ∀ x : ℝ, 0 < x → f x 1 = x - 2 * Real.log x - 1 / x :=
sorry

-- Find the intervals of increase for f(x) = x - (a+1) ln x - a/x
theorem intervals_of_increase (a : ℝ) : 
  (a ≤ 0 → ∀ x : ℝ, 1 < x → 0 ≤ (f x a - f 1 a)) ∧ 
  (0 < a ∧ a < 1 → (∀ x : ℝ, 0 < x ∧ x < a → 0 ≤ f x a) ∧ ∀ x : ℝ, 1 < x → 0 ≤ f x a ) ∧ 
  (a = 1 → ∀ x : ℝ, 0 < x → 0 ≤ f x a) ∧ 
  (a > 1 → (∀ x : ℝ, 0 < x ∧ x < 1 → 0 ≤ f x a) ∧ ∀ x : ℝ, a < x → 0 ≤ f x a ) :=
sorry

end NUMINAMATH_GPT_increase_a1_intervals_of_increase_l981_98199


namespace NUMINAMATH_GPT_find_like_term_l981_98176

-- Definition of the problem conditions
def monomials : List (String × String) := 
  [("A", "-2a^2b"), 
   ("B", "a^2b^2"), 
   ("C", "ab^2"), 
   ("D", "3ab")]

-- A function to check if two terms can be combined (like terms)
def like_terms(a b : String) : Prop :=
  a = "a^2b" ∧ b = "-2a^2b"

-- The theorem we need to prove
theorem find_like_term : ∃ x, x ∈ monomials ∧ like_terms "a^2b" (x.2) ∧ x.2 = "-2a^2b" :=
  sorry

end NUMINAMATH_GPT_find_like_term_l981_98176


namespace NUMINAMATH_GPT_seeds_germination_percentage_l981_98158

theorem seeds_germination_percentage :
  ∀ (total_seeds first_plot_seeds second_plot_seeds germinated_percentage_total germinated_percentage_second_plot germinated_seeds_total germinated_seeds_second_plot germinated_seeds_first_plot x : ℕ),
    total_seeds = 300 + 200 → 
    germinated_percentage_second_plot = 35 → 
    germinated_percentage_total = 32 → 
    second_plot_seeds = 200 → 
    germinated_seeds_second_plot = (germinated_percentage_second_plot * second_plot_seeds) / 100 → 
    germinated_seeds_total = (germinated_percentage_total * total_seeds) / 100 → 
    germinated_seeds_first_plot = germinated_seeds_total - germinated_seeds_second_plot → 
    x = 30 → 
    x = (germinated_seeds_first_plot * 100) / 300 → 
    x = 30 :=
  by 
    intros total_seeds first_plot_seeds second_plot_seeds germinated_percentage_total germinated_percentage_second_plot germinated_seeds_total germinated_seeds_second_plot germinated_seeds_first_plot x
    sorry

end NUMINAMATH_GPT_seeds_germination_percentage_l981_98158


namespace NUMINAMATH_GPT_students_in_class_l981_98141

theorem students_in_class {S : ℕ} 
  (h1 : 20 < S)
  (h2 : S < 30)
  (chess_club_condition : ∃ (n : ℕ), S = 3 * n) 
  (draughts_club_condition : ∃ (m : ℕ), S = 4 * m) : 
  S = 24 := 
sorry

end NUMINAMATH_GPT_students_in_class_l981_98141


namespace NUMINAMATH_GPT_perpendicular_lines_l981_98126

-- Definitions of conditions
def condition1 (α β γ δ : ℝ) : Prop := α = 90 ∧ α + β = 180 ∧ α + γ = 180 ∧ α + δ = 180
def condition2 (α β γ δ : ℝ) : Prop := α = β ∧ β = γ ∧ γ = δ
def condition3 (α β : ℝ) : Prop := α = β ∧ α + β = 180
def condition4 (α β : ℝ) : Prop := α = β ∧ α + β = 180

-- Main theorem statement
theorem perpendicular_lines (α β γ δ : ℝ) :
  (condition1 α β γ δ ∨ condition2 α β γ δ ∨
   condition3 α β ∨ condition4 α β) → α = 90 :=
by sorry

end NUMINAMATH_GPT_perpendicular_lines_l981_98126


namespace NUMINAMATH_GPT_find_value_of_f_f_neg1_l981_98168

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -2 / x else 3 + Real.log x / Real.log 2

theorem find_value_of_f_f_neg1 :
  f (f (-1)) = 4 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_value_of_f_f_neg1_l981_98168


namespace NUMINAMATH_GPT_smallest_n_divisible_by_31997_l981_98180

noncomputable def smallest_n_divisible_by_prime : Nat :=
  let p := 31997
  let k := p
  2 * k

theorem smallest_n_divisible_by_31997 :
  smallest_n_divisible_by_prime = 63994 :=
by
  unfold smallest_n_divisible_by_prime
  rfl

end NUMINAMATH_GPT_smallest_n_divisible_by_31997_l981_98180


namespace NUMINAMATH_GPT_highest_score_is_174_l981_98102

theorem highest_score_is_174
  (avg_40_innings : ℝ)
  (highest_exceeds_lowest : ℝ)
  (avg_excl_two : ℝ)
  (total_runs_40 : ℝ)
  (total_runs_38 : ℝ)
  (sum_H_L : ℝ)
  (new_avg_38 : ℝ)
  (H : ℝ)
  (L : ℝ)
  (H_eq_L_plus_172 : H = L + 172)
  (total_runs_40_eq : total_runs_40 = 40 * avg_40_innings)
  (total_runs_38_eq : total_runs_38 = 38 * new_avg_38)
  (sum_H_L_eq : sum_H_L = total_runs_40 - total_runs_38)
  (new_avg_eq : new_avg_38 = avg_40_innings - 2)
  (sum_H_L_val : sum_H_L = 176)
  (avg_40_val : avg_40_innings = 50) :
  H = 174 :=
sorry

end NUMINAMATH_GPT_highest_score_is_174_l981_98102


namespace NUMINAMATH_GPT_surveyed_households_count_l981_98144

theorem surveyed_households_count 
  (neither : ℕ) (only_R : ℕ) (both_B : ℕ) (both : ℕ) (h_main : Ξ)
  (H1 : neither = 80)
  (H2 : only_R = 60)
  (H3 : both = 40)
  (H4 : both_B = 3 * both) : 
  neither + only_R + both_B + both = 300 :=
by
  sorry

end NUMINAMATH_GPT_surveyed_households_count_l981_98144


namespace NUMINAMATH_GPT_abc_value_l981_98178

theorem abc_value (a b c : ℂ) 
  (h1 : a * b + 5 * b = -20)
  (h2 : b * c + 5 * c = -20)
  (h3 : c * a + 5 * a = -20) : 
  a * b * c = -100 := 
by {
  sorry
}

end NUMINAMATH_GPT_abc_value_l981_98178


namespace NUMINAMATH_GPT_largest_whole_number_solution_for_inequality_l981_98113

theorem largest_whole_number_solution_for_inequality :
  ∀ (x : ℕ), ((1 : ℝ) / 4 + (x : ℝ) / 5 < 2) → x ≤ 23 :=
by sorry

end NUMINAMATH_GPT_largest_whole_number_solution_for_inequality_l981_98113


namespace NUMINAMATH_GPT_no_nat_solutions_no_int_solutions_l981_98128

theorem no_nat_solutions (x y : ℕ) : x^3 + 5 * y = y^3 + 5 * x → x = y :=
by sorry

theorem no_int_solutions (x y : ℤ) : x^3 + 5 * y = y^3 + 5 * x → x = y :=
by sorry

end NUMINAMATH_GPT_no_nat_solutions_no_int_solutions_l981_98128


namespace NUMINAMATH_GPT_sufficient_condition_for_q_l981_98150

def p (a : ℝ) : Prop := a ≥ 0
def q (a : ℝ) : Prop := a^2 + a ≥ 0

theorem sufficient_condition_for_q (a : ℝ) : p a → q a := by 
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_q_l981_98150


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l981_98198

-- Define sets A and B
def A := {x : ℝ | x > 0}
def B := {x : ℝ | x < 1}

-- Statement of the proof problem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 1} := by
  sorry -- The proof goes here

end NUMINAMATH_GPT_intersection_of_A_and_B_l981_98198


namespace NUMINAMATH_GPT_max_digits_in_product_l981_98184

theorem max_digits_in_product :
  let n := (99999 : Nat)
  let m := (999 : Nat)
  let product := n * m
  ∃ d : Nat, product < 10^d ∧ 10^(d-1) ≤ product :=
by
  sorry

end NUMINAMATH_GPT_max_digits_in_product_l981_98184


namespace NUMINAMATH_GPT_olivia_earnings_this_week_l981_98125

variable (hourly_rate : ℕ) (hours_monday hours_wednesday hours_friday : ℕ)

theorem olivia_earnings_this_week : 
  hourly_rate = 9 → 
  hours_monday = 4 → 
  hours_wednesday = 3 → 
  hours_friday = 6 → 
  (hourly_rate * hours_monday + hourly_rate * hours_wednesday + hourly_rate * hours_friday) = 117 := 
by
  intros
  sorry

end NUMINAMATH_GPT_olivia_earnings_this_week_l981_98125


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l981_98145

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_sum_ratio : (a 0 + a 1 + a 2) / a 2 = 7) :
  q = 1 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l981_98145


namespace NUMINAMATH_GPT_problem1_problem2_l981_98194

theorem problem1 : 1 - 2 + 3 + (-4) = -2 :=
sorry

theorem problem2 : (-6) / 3 - (-10) - abs (-8) = 0 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l981_98194


namespace NUMINAMATH_GPT_sufficient_condition_for_proposition_l981_98139

theorem sufficient_condition_for_proposition :
  ∀ (a : ℝ), (0 < a ∧ a < 4) → (∀ x : ℝ, a * x ^ 2 + a * x + 1 > 0) := 
sorry

end NUMINAMATH_GPT_sufficient_condition_for_proposition_l981_98139


namespace NUMINAMATH_GPT_solve_quadratic_eq_l981_98175

theorem solve_quadratic_eq (x : ℝ) : (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l981_98175


namespace NUMINAMATH_GPT_find_x_l981_98146

def f (x : ℝ) : ℝ := 2 * x - 3 -- Definition of the function f

def c : ℝ := 11 -- Definition of the constant c

theorem find_x : 
  ∃ x : ℝ, 2 * f x - c = f (x - 2) ↔ x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l981_98146


namespace NUMINAMATH_GPT_meaningful_fraction_condition_l981_98132

theorem meaningful_fraction_condition (x : ℝ) : (4 - 2 * x ≠ 0) ↔ (x ≠ 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_meaningful_fraction_condition_l981_98132


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l981_98189

-- System (1)
theorem system1_solution (x y : ℚ) (h1 : 3 * y - 4 * x = 0) (h2 : 4 * x + y = 8) : 
  x = 3 / 2 ∧ y = 2 :=
by
  sorry

-- System (2)
theorem system2_solution (x y : ℚ) (h1 : x + y = 3) (h2 : (x - 1) / 4 + y / 2 = 3 / 4) : 
  x = 2 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l981_98189


namespace NUMINAMATH_GPT_derivative_at_one_eq_neg_one_l981_98124

variable {α : Type*} [TopologicalSpace α] {f : ℝ → ℝ}
-- condition: f is differentiable
variable (hf_diff : Differentiable ℝ f)
-- condition: limit condition
variable (h_limit : Tendsto (fun Δx => (f (1 + 2 * Δx) - f 1) / Δx) (𝓝 0) (𝓝 (-2)))

-- proof goal: f'(1) = -1
theorem derivative_at_one_eq_neg_one : deriv f 1 = -1 := 
by
  sorry

end NUMINAMATH_GPT_derivative_at_one_eq_neg_one_l981_98124


namespace NUMINAMATH_GPT_least_positive_multiple_of_primes_l981_98166

theorem least_positive_multiple_of_primes :
  11 * 13 * 17 * 19 = 46189 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_multiple_of_primes_l981_98166


namespace NUMINAMATH_GPT_candle_burning_time_l981_98134

theorem candle_burning_time :
  ∃ t : ℚ, (1 - t / 5) = 3 * (1 - t / 4) ∧ t = 40 / 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_candle_burning_time_l981_98134


namespace NUMINAMATH_GPT_calculate_speed_of_boat_in_still_water_l981_98167

noncomputable def speed_of_boat_in_still_water (V : ℝ) : Prop :=
    let downstream_speed := 16
    let upstream_speed := 9
    let first_half_current := 3 
    let second_half_current := 5
    let wind_speed := 2
    let effective_current_1 := first_half_current - wind_speed
    let effective_current_2 := second_half_current - wind_speed
    let V1 := downstream_speed - effective_current_1
    let V2 := upstream_speed + effective_current_2
    V = (V1 + V2) / 2

theorem calculate_speed_of_boat_in_still_water : 
    ∃ V : ℝ, speed_of_boat_in_still_water V ∧ V = 13.5 := 
sorry

end NUMINAMATH_GPT_calculate_speed_of_boat_in_still_water_l981_98167


namespace NUMINAMATH_GPT_sequence_a_2016_value_l981_98108

theorem sequence_a_2016_value (a : ℕ → ℕ) 
  (h1 : a 4 = 1)
  (h2 : a 11 = 9)
  (h3 : ∀ n : ℕ, a n + a (n+1) + a (n+2) = 15) :
  a 2016 = 5 :=
sorry

end NUMINAMATH_GPT_sequence_a_2016_value_l981_98108


namespace NUMINAMATH_GPT_cube_volume_given_face_perimeter_l981_98187

-- Define the perimeter condition
def is_face_perimeter (perimeter : ℝ) (side_length : ℝ) : Prop :=
  4 * side_length = perimeter

-- Define volume computation
def cube_volume (side_length : ℝ) : ℝ :=
  side_length^3

-- Theorem stating the relationship between face perimeter and cube volume
theorem cube_volume_given_face_perimeter : 
  ∀ (side_length perimeter : ℝ), is_face_perimeter 40 side_length → cube_volume side_length = 1000 :=
by
  intros side_length perimeter h
  sorry

end NUMINAMATH_GPT_cube_volume_given_face_perimeter_l981_98187


namespace NUMINAMATH_GPT_simplest_fraction_is_one_l981_98116

theorem simplest_fraction_is_one :
  ∃ m : ℕ, 
  (∃ k : ℕ, 45 * m = k^2) ∧ 
  (∃ n : ℕ, 56 * m = n^3) → 
  45 * m / 56 * m = 1 := by
  sorry

end NUMINAMATH_GPT_simplest_fraction_is_one_l981_98116


namespace NUMINAMATH_GPT_initial_deposit_l981_98127

/-- 
A person deposits some money in a bank at an interest rate of 7% per annum (of the original amount). 
After two years, the total amount in the bank is $6384. Prove that the initial amount deposited is $5600.
-/
theorem initial_deposit (P : ℝ) (h : (P + 0.07 * P) + 0.07 * P = 6384) : P = 5600 :=
by
  sorry

end NUMINAMATH_GPT_initial_deposit_l981_98127


namespace NUMINAMATH_GPT_inequality_preserves_neg_half_l981_98186

variable (a b : ℝ)

theorem inequality_preserves_neg_half (h : a ≤ b) : -a / 2 ≥ -b / 2 := by
  sorry

end NUMINAMATH_GPT_inequality_preserves_neg_half_l981_98186


namespace NUMINAMATH_GPT_max_winners_at_least_three_matches_l981_98197

theorem max_winners_at_least_three_matches (n : ℕ) (h : n = 200) :
  (∃ k : ℕ, k ≤ n ∧ ∀ m : ℕ, ((m ≥ 3) → ∃ x : ℕ, x = k → k = 66)) := 
sorry

end NUMINAMATH_GPT_max_winners_at_least_three_matches_l981_98197


namespace NUMINAMATH_GPT_prime_factorization_count_l981_98196

theorem prime_factorization_count :
  (∃ (S : Finset ℕ), S = {97, 101, 2, 13, 107, 109} ∧ S.card = 6) :=
by
  sorry

end NUMINAMATH_GPT_prime_factorization_count_l981_98196


namespace NUMINAMATH_GPT_chip_price_reduction_equation_l981_98133

-- Define initial price
def initial_price : ℝ := 400

-- Define final price after reductions
def final_price : ℝ := 144

-- Define the price reduction percentage
variable (x : ℝ)

-- The equation we need to prove
theorem chip_price_reduction_equation :
  initial_price * (1 - x) ^ 2 = final_price :=
sorry

end NUMINAMATH_GPT_chip_price_reduction_equation_l981_98133


namespace NUMINAMATH_GPT_prob_A_is_15_16_prob_B_is_3_4_prob_C_is_5_9_prob_exactly_two_good_ratings_is_77_576_l981_98174

-- Define the probability of success for student A, B, and C on a single jump
def p_A1 := 3 / 4
def p_B1 := 1 / 2
def p_C1 := 1 / 3

-- Calculate the total probability of excellence for A, B, and C
def P_A := p_A1 + (1 - p_A1) * p_A1
def P_B := p_B1 + (1 - p_B1) * p_B1
def P_C := p_C1 + (1 - p_C1) * p_C1

-- Statement to prove probabilities
theorem prob_A_is_15_16 : P_A = 15 / 16 := sorry
theorem prob_B_is_3_4 : P_B = 3 / 4 := sorry
theorem prob_C_is_5_9 : P_C = 5 / 9 := sorry

-- Definition for P(Good_Ratings) - exactly two students get a good rating
def P_Good_Ratings := 
  P_A * (1 - P_B) * (1 - P_C) + 
  (1 - P_A) * P_B * (1 - P_C) + 
  (1 - P_A) * (1 - P_B) * P_C

-- Statement to prove the given condition about good ratings
theorem prob_exactly_two_good_ratings_is_77_576 : P_Good_Ratings = 77 / 576 := sorry

end NUMINAMATH_GPT_prob_A_is_15_16_prob_B_is_3_4_prob_C_is_5_9_prob_exactly_two_good_ratings_is_77_576_l981_98174


namespace NUMINAMATH_GPT_number_of_ways_to_prepare_all_elixirs_l981_98195

def fairy_methods : ℕ := 2
def elf_methods : ℕ := 2
def fairy_elixirs : ℕ := 3
def elf_elixirs : ℕ := 4

theorem number_of_ways_to_prepare_all_elixirs : 
  (fairy_methods * fairy_elixirs) + (elf_methods * elf_elixirs) = 14 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_prepare_all_elixirs_l981_98195


namespace NUMINAMATH_GPT_maria_carrots_l981_98164

theorem maria_carrots :
  ∀ (picked initially thrownOut moreCarrots totalLeft : ℕ),
    initially = 48 →
    thrownOut = 11 →
    totalLeft = 52 →
    moreCarrots = totalLeft - (initially - thrownOut) →
    moreCarrots = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_maria_carrots_l981_98164


namespace NUMINAMATH_GPT_newspaper_pages_l981_98111

theorem newspaper_pages (p : ℕ) (h₁ : p >= 21) (h₂ : 8•2 - 1 ≤ p) (h₃ : p ≤ 8•3) : p = 28 :=
sorry

end NUMINAMATH_GPT_newspaper_pages_l981_98111


namespace NUMINAMATH_GPT_seed_mixture_Y_is_25_percent_ryegrass_l981_98161

variables (X Y : ℝ) (R : ℝ)

def proportion_X_is_40_percent_ryegrass : Prop :=
  X = 40 / 100

def proportion_Y_contains_percent_ryegrass (R : ℝ) : Prop :=
  100 - R = 75 / 100 * 100

def mixture_contains_30_percent_ryegrass (X Y R : ℝ) : Prop :=
  (1/3) * (40 / 100) * 100 + (2/3) * (R / 100) * 100 = 30

def weight_of_mixture_is_33_percent_X (X Y : ℝ) : Prop :=
  X / (X + Y) = 1 / 3

theorem seed_mixture_Y_is_25_percent_ryegrass
  (X Y : ℝ) (R : ℝ) 
  (h1 : proportion_X_is_40_percent_ryegrass X)
  (h2 : proportion_Y_contains_percent_ryegrass R)
  (h3 : weight_of_mixture_is_33_percent_X X Y)
  (h4 : mixture_contains_30_percent_ryegrass X Y R) :
  R = 25 :=
sorry

end NUMINAMATH_GPT_seed_mixture_Y_is_25_percent_ryegrass_l981_98161


namespace NUMINAMATH_GPT_find_a_l981_98130

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end NUMINAMATH_GPT_find_a_l981_98130


namespace NUMINAMATH_GPT_probability_of_perfect_square_sum_l981_98107

def two_dice_probability_of_perfect_square_sum : ℚ :=
  let totalOutcomes := 12 * 12
  let perfectSquareOutcomes := 3 + 8 + 9 -- ways to get sums 4, 9, and 16
  (perfectSquareOutcomes : ℚ) / (totalOutcomes : ℚ)

theorem probability_of_perfect_square_sum :
  two_dice_probability_of_perfect_square_sum = 5 / 36 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_perfect_square_sum_l981_98107


namespace NUMINAMATH_GPT_geometric_sequence_iff_q_neg_one_l981_98100

theorem geometric_sequence_iff_q_neg_one {p q : ℝ} (h1 : p ≠ 0) (h2 : p ≠ 1)
  (S : ℕ → ℝ) (hS : ∀ n, S n = p^n + q) :
  (∃ (a : ℕ → ℝ), (∀ n, a (n+1) = (p - 1) * p^n) ∧ (∀ n, a (n+1) = S (n+1) - S n) ∧
                    (∀ n, a (n+1) / a n = p)) ↔ q = -1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_iff_q_neg_one_l981_98100
