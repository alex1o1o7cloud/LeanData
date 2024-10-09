import Mathlib

namespace algebraic_identity_l233_23360

variables {R : Type*} [CommRing R] (a b : R)

theorem algebraic_identity : 2 * (a - b) + 3 * b = 2 * a + b :=
by
  sorry

end algebraic_identity_l233_23360


namespace increase_function_a_seq_increasing_b_seq_decreasing_seq_relation_l233_23377

open Real

-- Defining the sequences
noncomputable def a_seq (n : ℕ) : ℝ := (1 + (1 : ℝ) / n) ^ n
noncomputable def b_seq (n : ℕ) : ℝ := (1 + (1 : ℝ) / n) ^ (n + 1)

theorem increase_function (x : ℝ) (hx : 0 < x) : 
  ((1:ℝ) + 1 / x) ^ x < (1 + 1 / (x + 1)) ^ (x + 1) := sorry

theorem a_seq_increasing (n : ℕ) (hn : 0 < n) : 
  a_seq n < a_seq (n + 1) := sorry

theorem b_seq_decreasing (n : ℕ) (hn : 0 < n) : 
  b_seq (n + 1) < b_seq n := sorry

theorem seq_relation (n : ℕ) (hn : 0 < n) : 
  a_seq n < b_seq n := sorry

end increase_function_a_seq_increasing_b_seq_decreasing_seq_relation_l233_23377


namespace remainder_3_pow_2040_mod_11_l233_23314

theorem remainder_3_pow_2040_mod_11 : (3 ^ 2040) % 11 = 1 := by
  have h1 : 3 % 11 = 3 := by norm_num
  have h2 : (3 ^ 2) % 11 = 9 := by norm_num
  have h3 : (3 ^ 3) % 11 = 5 := by norm_num
  have h4 : (3 ^ 4) % 11 = 4 := by norm_num
  have h5 : (3 ^ 5) % 11 = 1 := by norm_num
  have h_mod : 2040 % 5 = 0 := by norm_num
  sorry

end remainder_3_pow_2040_mod_11_l233_23314


namespace prime_numbers_satisfying_equation_l233_23369

theorem prime_numbers_satisfying_equation :
  ∀ p : ℕ, Nat.Prime p →
    (∃ x y : ℕ, 1 ≤ x ∧ 1 ≤ y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) →
    p = 2 ∨ p = 3 ∨ p = 7 := 
by 
  intro p hpprime h
  sorry

end prime_numbers_satisfying_equation_l233_23369


namespace two_truth_tellers_are_B_and_C_l233_23367

-- Definitions of students and their statements
def A_statement_false (A_said : Prop) (A_truth_teller : Prop) := ¬A_said = A_truth_teller
def B_statement_true (B_said : Prop) (B_truth_teller : Prop) := B_said = B_truth_teller
def C_statement_true (C_said : Prop) (C_truth_teller : Prop) := C_said = C_truth_teller
def D_statement_false (D_said : Prop) (D_truth_teller : Prop) := ¬D_said = D_truth_teller

-- Given statements
def A_said := ¬ (False : Prop)
def B_said := True
def C_said := B_said ∨ D_statement_false True True
def D_said := False

-- Define who is telling the truth
def A_truth_teller := False
def B_truth_teller := True
def C_truth_teller := True
def D_truth_teller := False

-- Proof problem statement
theorem two_truth_tellers_are_B_and_C :
  (A_statement_false A_said A_truth_teller) ∧
  (B_statement_true B_said B_truth_teller) ∧
  (C_statement_true C_said C_truth_teller) ∧
  (D_statement_false D_said D_truth_teller) →
  ((A_truth_teller = False) ∧
  (B_truth_teller = True) ∧
  (C_truth_teller = True) ∧
  (D_truth_teller = False)) := 
by {
  sorry
}

end two_truth_tellers_are_B_and_C_l233_23367


namespace min_value_is_nine_l233_23315

noncomputable def min_value_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  ℝ :=
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)

theorem min_value_is_nine (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  min_value_expression a b c h_pos h_sum = 9 :=
sorry

end min_value_is_nine_l233_23315


namespace total_students_in_class_l233_23376

theorem total_students_in_class (S R : ℕ)
  (h1 : S = 2 + 12 + 4 + R)
  (h2 : 0 * 2 + 1 * 12 + 2 * 4 + 3 * R = 2 * S) : S = 34 :=
by { sorry }

end total_students_in_class_l233_23376


namespace parabola_line_intersection_length_l233_23355

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y k : ℝ) : Prop := y = k * x - 1
def focus : ℝ × ℝ := (1, 0)

theorem parabola_line_intersection_length (k x1 x2 y1 y2 : ℝ)
  (h_focus : line 1 0 k)
  (h_parabola1 : parabola x1 y1)
  (h_parabola2 : parabola x2 y2)
  (h_line1 : line x1 y1 k)
  (h_line2 : line x2 y2 k) :
  k = 1 ∧ (x1 + x2 + 2) = 8 :=
by
  sorry

end parabola_line_intersection_length_l233_23355


namespace arithmetic_square_root_of_16_l233_23357

theorem arithmetic_square_root_of_16 : ∃ x : ℝ, x^2 = 16 ∧ x > 0 ∧ x = 4 :=
by
  sorry

end arithmetic_square_root_of_16_l233_23357


namespace cream_cheese_volume_l233_23392

theorem cream_cheese_volume
  (raw_spinach : ℕ)
  (spinach_reduction : ℕ)
  (eggs_volume : ℕ)
  (total_volume : ℕ)
  (cooked_spinach : ℕ)
  (cream_cheese : ℕ) :
  raw_spinach = 40 →
  spinach_reduction = 20 →
  eggs_volume = 4 →
  total_volume = 18 →
  cooked_spinach = raw_spinach * spinach_reduction / 100 →
  cream_cheese = total_volume - cooked_spinach - eggs_volume →
  cream_cheese = 6 :=
by
  intros h_raw_spinach h_spinach_reduction h_eggs_volume h_total_volume h_cooked_spinach h_cream_cheese
  sorry

end cream_cheese_volume_l233_23392


namespace intersection_of_sets_l233_23341

theorem intersection_of_sets:
  let A := {-2, -1, 0, 1}
  let B := {x : ℤ | x^3 + 1 ≤ 0 }
  A ∩ B = {-2, -1} :=
by
  sorry

end intersection_of_sets_l233_23341


namespace sum_of_interior_angles_l233_23393

theorem sum_of_interior_angles (n : ℕ) (h : 180 * (n - 2) = 1800) : 180 * ((n - 3) - 2) = 1260 :=
by
  sorry

end sum_of_interior_angles_l233_23393


namespace min_y_value_l233_23381

theorem min_y_value (x : ℝ) : 
  ∃ y : ℝ, y = 4 * x^2 + 8 * x + 12 ∧ ∀ z, (z = 4 * x^2 + 8 * x + 12) → y ≤ z := sorry

end min_y_value_l233_23381


namespace max_right_angles_in_triangular_prism_l233_23309

theorem max_right_angles_in_triangular_prism 
  (n_triangles : ℕ) 
  (n_rectangles : ℕ) 
  (max_right_angles_triangle : ℕ) 
  (max_right_angles_rectangle : ℕ)
  (h1 : n_triangles = 2)
  (h2 : n_rectangles = 3)
  (h3 : max_right_angles_triangle = 1)
  (h4 : max_right_angles_rectangle = 4) : 
  (n_triangles * max_right_angles_triangle + n_rectangles * max_right_angles_rectangle = 14) :=
by
  sorry

end max_right_angles_in_triangular_prism_l233_23309


namespace right_triangle_divisibility_l233_23354

theorem right_triangle_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (a % 3 = 0 ∨ b % 3 = 0) ∧ (a % 5 = 0 ∨ b % 5 = 0 ∨ c % 5 = 0) :=
by
  -- skipping the proof
  sorry

end right_triangle_divisibility_l233_23354


namespace g_diff_l233_23356

noncomputable section

-- Definition of g(n) as given in the problem statement
def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((1 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((1 - Real.sqrt 3) / 2)^n

-- The statement to prove g(n+2) - g(n) = -1/4 * g(n)
theorem g_diff (n : ℕ) : g (n + 2) - g n = -1 / 4 * g n :=
by
  sorry

end g_diff_l233_23356


namespace original_selling_price_l233_23361

theorem original_selling_price (P : ℝ) (d1 d2 d3 t : ℝ) (final_price : ℝ) :
  d1 = 0.32 → -- first discount
  d2 = 0.10 → -- loyalty discount
  d3 = 0.05 → -- holiday discount
  t = 0.15 → -- state tax
  final_price = 650 → 
  1.15 * P * (1 - d1) * (1 - d2) * (1 - d3) = final_price →
  P = 722.57 :=
sorry

end original_selling_price_l233_23361


namespace number_of_pages_correct_number_of_ones_correct_l233_23321

noncomputable def number_of_pages (total_digits : ℕ) : ℕ :=
  let single_digit_odd_pages := 5
  let double_digit_odd_pages := 45
  let triple_digit_odd_pages := (total_digits - (single_digit_odd_pages + 2 * double_digit_odd_pages)) / 3
  single_digit_odd_pages + double_digit_odd_pages + triple_digit_odd_pages

theorem number_of_pages_correct : number_of_pages 125 = 60 :=
by sorry

noncomputable def number_of_ones (total_digits : ℕ) : ℕ :=
  let ones_in_units_place := 12
  let ones_in_tens_place := 18
  let ones_in_hundreds_place := 10
  ones_in_units_place + ones_in_tens_place + ones_in_hundreds_place

theorem number_of_ones_correct : number_of_ones 125 = 40 :=
by sorry

end number_of_pages_correct_number_of_ones_correct_l233_23321


namespace sum_of_ages_l233_23303

theorem sum_of_ages (X_c Y_c : ℕ) (h1 : X_c = 45) 
  (h2 : X_c - 3 = 2 * (Y_c - 3)) : 
  (X_c + 7) + (Y_c + 7) = 83 := 
by
  sorry

end sum_of_ages_l233_23303


namespace sum_of_factors_1656_l233_23383

theorem sum_of_factors_1656 : ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 1656 ∧ a + b = 110 := by
  sorry

end sum_of_factors_1656_l233_23383


namespace cats_not_eating_either_l233_23344

theorem cats_not_eating_either (total_cats : ℕ) (cats_liking_apples : ℕ) (cats_liking_fish : ℕ) (cats_liking_both : ℕ)
  (h1 : total_cats = 75) (h2 : cats_liking_apples = 15) (h3 : cats_liking_fish = 55) (h4 : cats_liking_both = 8) :
  ∃ cats_not_eating_either : ℕ, cats_not_eating_either = total_cats - (cats_liking_apples - cats_liking_both + cats_liking_fish - cats_liking_both + cats_liking_both) ∧ cats_not_eating_either = 13 :=
by
  sorry

end cats_not_eating_either_l233_23344


namespace smallest_n_inequality_l233_23330

theorem smallest_n_inequality :
  ∃ n : ℕ, (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
    ∀ m : ℕ, m < n → ¬ (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ m * (x^4 + y^4 + z^4)) :=
by
  sorry

end smallest_n_inequality_l233_23330


namespace S_7_is_28_l233_23395

-- Define the arithmetic sequence and sum of first n terms
def a : ℕ → ℝ := sorry  -- placeholder for arithmetic sequence
def S (n : ℕ) : ℝ := sorry  -- placeholder for the sum of first n terms

-- Given conditions
def a_3 : ℝ := 3
def a_10 : ℝ := 10

-- Define properties of the arithmetic sequence
axiom a_n_property (n : ℕ) : a n = a 1 + (n - 1) * (a 10 - a 3) / (10 - 3)

-- Define the sum of first n terms
axiom sum_property (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Given specific elements of the sequence
axiom a_3_property : a 3 = 3
axiom a_10_property : a 10 = 10

-- The statement to prove
theorem S_7_is_28 : S 7 = 28 :=
sorry

end S_7_is_28_l233_23395


namespace math_problem_l233_23302

theorem math_problem : (-4)^2 * ((-1)^2023 + (3 / 4) + (-1 / 2)^3) = -6 := 
by 
  sorry

end math_problem_l233_23302


namespace sum_series_eq_three_l233_23326

theorem sum_series_eq_three :
  (∑' k : ℕ, (9^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 :=
by 
  sorry

end sum_series_eq_three_l233_23326


namespace sum_of_values_l233_23385

def r (x : ℝ) : ℝ := abs (x + 1) - 3
def s (x : ℝ) : ℝ := -(abs (x + 2))

theorem sum_of_values :
  (s (r (-5)) + s (r (-4)) + s (r (-3)) + s (r (-2)) + s (r (-1)) + s (r (0)) + s (r (1)) + s (r (2)) + s (r (3))) = -37 :=
by {
  sorry
}

end sum_of_values_l233_23385


namespace find_solutions_l233_23349

-- Define the conditions
variable (n : ℕ)
noncomputable def valid_solution (a b c d : ℕ) : Prop := 
  a^2 + b^2 + c^2 + d^2 = 7 * 4^n

-- Define each possible solution
def sol1 : ℕ × ℕ × ℕ × ℕ := (5 * 2 ^ (n - 1), 2 ^ (n - 1), 2 ^ (n - 1), 2 ^ (n - 1))
def sol2 : ℕ × ℕ × ℕ × ℕ := (2 ^ (n + 1), 2 ^ n, 2 ^ n, 2 ^ n)
def sol3 : ℕ × ℕ × ℕ × ℕ := (3 * 2 ^ (n - 1), 3 * 2 ^ (n - 1), 3 * 2 ^ (n - 1), 2 ^ (n - 1))

-- State the theorem
theorem find_solutions (a b c d : ℕ) (n : ℕ) :
  valid_solution n a b c d →
  (a, b, c, d) = sol1 n ∨
  (a, b, c, d) = sol2 n ∨
  (a, b, c, d) = sol3 n :=
sorry

end find_solutions_l233_23349


namespace reflection_points_reflection_line_l233_23306

-- Definitions of given points and line equation
def original_point : ℝ × ℝ := (2, 3)
def reflected_point : ℝ × ℝ := (8, 7)

-- Definitions of line parameters for y = mx + b
variable {m b : ℝ}

-- Statement of the reflection condition
theorem reflection_points_reflection_line : m + b = 9.5 := by
  -- sorry to skip the actual proof
  sorry

end reflection_points_reflection_line_l233_23306


namespace sum_of_solutions_l233_23372

theorem sum_of_solutions (y : ℝ) (h : y + 16 / y = 12) : y = 4 ∨ y = 8 → 4 + 8 = 12 :=
by sorry

end sum_of_solutions_l233_23372


namespace jameson_badminton_medals_l233_23307

theorem jameson_badminton_medals :
  ∃ (b : ℕ),  (∀ (t s : ℕ), t = 5 → s = 2 * t → t + s + b = 20) ∧ b = 5 :=
by {
sorry
}

end jameson_badminton_medals_l233_23307


namespace orthocenter_PQR_l233_23325

structure Point3D :=
  (x : ℚ)
  (y : ℚ)
  (z : ℚ)

def orthocenter (P Q R : Point3D) : Point3D :=
  sorry

theorem orthocenter_PQR :
  orthocenter ⟨2, 3, 4⟩ ⟨6, 4, 2⟩ ⟨4, 5, 6⟩ = ⟨1/2, 13/2, 15/2⟩ :=
by {
  sorry
}

end orthocenter_PQR_l233_23325


namespace rosie_purchase_price_of_art_piece_l233_23358

-- Define the conditions as hypotheses
variables (P : ℝ)
variables (future_value increase : ℝ)

-- Given conditions
def conditions := future_value = 3 * P ∧ increase = 8000 ∧ increase = future_value - P

-- The statement to be proved
theorem rosie_purchase_price_of_art_piece (h : conditions P future_value increase) : P = 4000 :=
sorry

end rosie_purchase_price_of_art_piece_l233_23358


namespace repair_cost_l233_23382

theorem repair_cost (C : ℝ) (repair_cost : ℝ) (profit : ℝ) (selling_price : ℝ)
  (h1 : repair_cost = 0.10 * C)
  (h2 : profit = 1100)
  (h3 : selling_price = 1.20 * C)
  (h4 : profit = selling_price - C) :
  repair_cost = 550 :=
by
  sorry

end repair_cost_l233_23382


namespace geometric_progression_x_l233_23374

theorem geometric_progression_x :
  ∃ x : ℝ, (70 + x) ^ 2 = (30 + x) * (150 + x) ∧ x = 10 :=
by sorry

end geometric_progression_x_l233_23374


namespace simplify_fraction_l233_23396

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by 
  sorry

end simplify_fraction_l233_23396


namespace dividend_calculation_l233_23313

theorem dividend_calculation
  (divisor : Int)
  (quotient : Int)
  (remainder : Int)
  (dividend : Int)
  (h_divisor : divisor = 800)
  (h_quotient : quotient = 594)
  (h_remainder : remainder = -968)
  (h_dividend : dividend = (divisor * quotient) + remainder) :
  dividend = 474232 := by
  sorry

end dividend_calculation_l233_23313


namespace members_who_play_both_sports_l233_23319

theorem members_who_play_both_sports 
  (N B T Neither BT : ℕ) 
  (h1 : N = 27)
  (h2 : B = 17)
  (h3 : T = 19)
  (h4 : Neither = 2)
  (h5 : BT = B + T - N + Neither) : 
  BT = 11 := 
by 
  have h6 : 17 + 19 - 27 + 2 = 11 := by norm_num
  rw [h2, h3, h1, h4, h6] at h5
  exact h5

end members_who_play_both_sports_l233_23319


namespace polyhedron_faces_same_edges_l233_23343

theorem polyhedron_faces_same_edges (n : ℕ) (h_n : n ≥ 4) : 
  ∃ (f1 f2 : ℕ), f1 ≠ f2 ∧ 3 ≤ f1 ∧ f1 ≤ n - 1 ∧ 3 ≤ f2 ∧ f2 ≤ n - 1 ∧ f1 = f2 := 
by
  sorry

end polyhedron_faces_same_edges_l233_23343


namespace ratio_man_to_son_in_two_years_l233_23378

-- Define the conditions
def son_current_age : ℕ := 32
def man_current_age : ℕ := son_current_age + 34

-- Define the ages in two years
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- The theorem to prove the ratio in two years
theorem ratio_man_to_son_in_two_years : 
  (man_age_in_two_years : ℚ) / son_age_in_two_years = 2 :=
by
  -- Skip the proof
  sorry

end ratio_man_to_son_in_two_years_l233_23378


namespace adi_change_l233_23389

theorem adi_change : 
  let pencil := 0.35
  let notebook := 1.50
  let colored_pencils := 2.75
  let discount := 0.05
  let tax := 0.10
  let payment := 20.00
  let total_cost_before_discount := pencil + notebook + colored_pencils
  let discount_amount := discount * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let tax_amount := tax * total_cost_after_discount
  let total_cost := total_cost_after_discount + tax_amount
  let change := payment - total_cost
  change = 15.19 :=
by
  sorry

end adi_change_l233_23389


namespace min_value_S_l233_23352

noncomputable def S (x y : ℝ) : ℝ := 2 * x ^ 2 - x * y + y ^ 2 + 2 * x + 3 * y

theorem min_value_S : ∃ x y : ℝ, S x y = -4 ∧ ∀ (a b : ℝ), S a b ≥ -4 := 
by
  sorry

end min_value_S_l233_23352


namespace find_a_decreasing_l233_23305

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem find_a_decreasing : 
  (∀ x : ℝ, x < 6 → f x a ≤ f (x - 1) a) → a ≥ 6 := 
sorry

end find_a_decreasing_l233_23305


namespace product_increase_2022_l233_23342

theorem product_increase_2022 (a b c : ℕ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 678) :
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2022 :=
by {
  -- The proof would go here, but it's not required per the instructions.
  sorry
}

end product_increase_2022_l233_23342


namespace ratio_four_of_v_m_l233_23328

theorem ratio_four_of_v_m (m v : ℝ) (h : m < v) 
  (h_eq : 5 * (3 / 4 * m) = v - 1 / 4 * m) : v / m = 4 :=
sorry

end ratio_four_of_v_m_l233_23328


namespace distance_between_foci_of_ellipse_l233_23370

theorem distance_between_foci_of_ellipse (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∀ (x y : ℝ), 9 * x^2 + y^2 = 36 → 2 * Real.sqrt (b^2 - a^2) = 8 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end distance_between_foci_of_ellipse_l233_23370


namespace point_on_x_axis_l233_23301

theorem point_on_x_axis (m : ℝ) (h : (m, m - 1).snd = 0) : m = 1 :=
by
  sorry

end point_on_x_axis_l233_23301


namespace actual_average_height_l233_23353

theorem actual_average_height
  (incorrect_avg_height : ℝ)
  (n : ℕ)
  (incorrect_height : ℝ)
  (actual_height : ℝ)
  (h1 : incorrect_avg_height = 184)
  (h2 : n = 35)
  (h3 : incorrect_height = 166)
  (h4 : actual_height = 106) :
  let incorrect_total_height := incorrect_avg_height * n
  let difference := incorrect_height - actual_height
  let correct_total_height := incorrect_total_height - difference
  let correct_avg_height := correct_total_height / n
  correct_avg_height = 182.29 :=
by {
  sorry
}

end actual_average_height_l233_23353


namespace triangle_is_obtuse_l233_23388

def is_obtuse_triangle (a b c : ℕ) : Prop := a^2 + b^2 < c^2

theorem triangle_is_obtuse :
    is_obtuse_triangle 4 6 8 :=
by
    sorry

end triangle_is_obtuse_l233_23388


namespace average_weight_decrease_l233_23365

theorem average_weight_decrease :
  let original_avg := 102
  let new_weight := 40
  let original_boys := 30
  let total_boys := original_boys + 1
  (original_avg - ((original_boys * original_avg + new_weight) / total_boys)) = 2 :=
by
  sorry

end average_weight_decrease_l233_23365


namespace child_height_at_age_10_l233_23346

theorem child_height_at_age_10 (x y : ℝ) (h : y = 7.19 * x + 73.93) (hx : x = 10) : abs (y - 145.83) < 1 :=
by {
  sorry
}

end child_height_at_age_10_l233_23346


namespace div_add_fraction_l233_23329

theorem div_add_fraction :
  (-75) / (-25) + 1/2 = 7/2 := by
  sorry

end div_add_fraction_l233_23329


namespace kamal_marks_physics_l233_23336

-- Define the marks in subjects
def marks_english := 66
def marks_mathematics := 65
def marks_chemistry := 62
def marks_biology := 75
def average_marks := 69
def number_of_subjects := 5

-- Calculate the total marks from the average
def total_marks := average_marks * number_of_subjects

-- Calculate the known total marks
def known_total_marks := marks_english + marks_mathematics + marks_chemistry + marks_biology

-- Define Kamal's marks in Physics
def marks_physics := total_marks - known_total_marks

-- Prove the marks in Physics are 77
theorem kamal_marks_physics : marks_physics = 77 := by
  sorry

end kamal_marks_physics_l233_23336


namespace divide_and_add_l233_23379

theorem divide_and_add (x : ℤ) (h1 : x = 95) : (x / 5) + 23 = 42 := by
  sorry

end divide_and_add_l233_23379


namespace even_integers_count_form_3k_plus_4_l233_23348

theorem even_integers_count_form_3k_plus_4 
  (n : ℕ) (h1 : 20 ≤ n ∧ n ≤ 250)
  (h2 : ∃ k : ℕ, n = 3 * k + 4 ∧ Even n) : 
  ∃ N : ℕ, N = 39 :=
by {
  sorry
}

end even_integers_count_form_3k_plus_4_l233_23348


namespace fg_of_2_l233_23362

def g (x : ℝ) : ℝ := 2 * x^2
def f (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2 : f (g 2) = 15 :=
by
  have h1 : g 2 = 8 := by sorry
  have h2 : f 8 = 15 := by sorry
  rw [h1]
  exact h2

end fg_of_2_l233_23362


namespace odd_number_expression_parity_l233_23316

theorem odd_number_expression_parity (o n : ℕ) (ho : ∃ k : ℕ, o = 2 * k + 1) :
  (o^2 + n * o) % 2 = 1 ↔ n % 2 = 0 :=
by
  sorry

end odd_number_expression_parity_l233_23316


namespace adam_and_simon_time_to_be_80_miles_apart_l233_23384

theorem adam_and_simon_time_to_be_80_miles_apart :
  ∃ x : ℝ, (10 * x)^2 + (8 * x)^2 = 80^2 ∧ x = 6.25 :=
by
  sorry

end adam_and_simon_time_to_be_80_miles_apart_l233_23384


namespace people_own_only_cats_and_dogs_l233_23399

-- Define the given conditions
def total_people : ℕ := 59
def only_dogs : ℕ := 15
def only_cats : ℕ := 10
def cats_dogs_snakes : ℕ := 3
def total_snakes : ℕ := 29

-- Define the proof problem
theorem people_own_only_cats_and_dogs : ∃ x : ℕ, 15 + 10 + x + 3 + (29 - 3) = 59 ∧ x = 5 :=
by {
  sorry
}

end people_own_only_cats_and_dogs_l233_23399


namespace projection_matrix_determinant_l233_23386

theorem projection_matrix_determinant (a c : ℚ) (h : (a^2 + (20 / 49 : ℚ) * c = a) ∧ ((20 / 49 : ℚ) * a + 580 / 2401 = 20 / 49) ∧ (a * c + (29 / 49 : ℚ) * c = c) ∧ ((20 / 49 : ℚ) * c + 841 / 2401 = 29 / 49)) :
  (a = 41 / 49) ∧ (c = 204 / 1225) := 
by {
  sorry
}

end projection_matrix_determinant_l233_23386


namespace decagon_diagonal_relation_l233_23339

-- Define side length, shortest diagonal, and longest diagonal in a regular decagon
variable (a b d : ℝ)
variable (h1 : a > 0) -- Side length must be positive
variable (h2 : b > 0) -- Shortest diagonal length must be positive
variable (h3 : d > 0) -- Longest diagonal length must be positive

theorem decagon_diagonal_relation (ha : d^2 = 5 * a^2) (hb : b^2 = 3 * a^2) : b^2 = a * d :=
sorry

end decagon_diagonal_relation_l233_23339


namespace functional_equation_solution_l233_23397

open Nat

theorem functional_equation_solution (f : ℕ+ → ℕ+) 
  (H : ∀ (m n : ℕ+), f (f (f m) * f (f m) + 2 * f (f n) * f (f n)) = m * m + 2 * n * n) : 
  ∀ n : ℕ+, f n = n := 
sorry

end functional_equation_solution_l233_23397


namespace fraction_power_l233_23380

theorem fraction_power (a b : ℕ) (ha : a = 5) (hb : b = 6) : (a / b : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end fraction_power_l233_23380


namespace balls_in_boxes_l233_23375

theorem balls_in_boxes :
  let balls := 5
  let boxes := 4
  boxes ^ balls = 1024 :=
by
  sorry

end balls_in_boxes_l233_23375


namespace sphere_surface_area_l233_23324

theorem sphere_surface_area (r : ℝ) (hr : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi :=
by
  rw [hr]
  norm_num
  sorry

end sphere_surface_area_l233_23324


namespace shape_is_cone_l233_23345

-- Define the spherical coordinate system and the condition
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

def shape (c : ℝ) (p : SphericalCoord) : Prop := p.φ ≤ c

-- The shape described by \(\exists c, \forall p \in SphericalCoord, shape c p\) is a cone
theorem shape_is_cone (c : ℝ) (p : SphericalCoord) : shape c p → (c ≥ 0 ∧ c ≤ π → shape c p = Cone) :=
by
  sorry

end shape_is_cone_l233_23345


namespace range_of_values_for_a_l233_23398

theorem range_of_values_for_a (a : ℝ) :
  (∀ x : ℝ, (x + 2) / 3 - x / 2 > 1 → 2 * (x - a) ≤ 0) → a ≥ -2 :=
by
  intro h
  sorry

end range_of_values_for_a_l233_23398


namespace simplify_expression_l233_23366

-- Define the given expression
def expr : ℚ := (5^6 + 5^3) / (5^5 - 5^2)

-- State the proof problem
theorem simplify_expression : expr = 315 / 62 := 
by sorry

end simplify_expression_l233_23366


namespace prob_white_ball_second_l233_23338

structure Bag :=
  (black_balls : ℕ)
  (white_balls : ℕ)

def total_balls (bag : Bag) := bag.black_balls + bag.white_balls

def prob_white_second_after_black_first (bag : Bag) : ℚ :=
  if bag.black_balls > 0 ∧ bag.white_balls > 0 ∧ total_balls bag > 1 then
    (bag.white_balls : ℚ) / (total_balls bag - 1)
  else 0

theorem prob_white_ball_second 
  (bag : Bag)
  (h_black : bag.black_balls = 4)
  (h_white : bag.white_balls = 3)
  (h_total : total_balls bag = 7) :
  prob_white_second_after_black_first bag = 1 / 2 :=
by
  sorry

end prob_white_ball_second_l233_23338


namespace cos_value_l233_23351

theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 4 - α) = 1 / 3) :
  Real.cos (Real.pi / 4 + α) = 1 / 3 :=
sorry

end cos_value_l233_23351


namespace find_value_of_A_l233_23368

theorem find_value_of_A (A B : ℤ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 :=
by
  sorry

end find_value_of_A_l233_23368


namespace students_more_than_turtles_l233_23317

theorem students_more_than_turtles
  (students_per_classroom : ℕ)
  (turtles_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (h1 : students_per_classroom = 20)
  (h2 : turtles_per_classroom = 3)
  (h3 : number_of_classrooms = 5) :
  (students_per_classroom * number_of_classrooms)
  - (turtles_per_classroom * number_of_classrooms) = 85 :=
by
  sorry

end students_more_than_turtles_l233_23317


namespace domain_is_all_real_l233_23300

-- Definitions and conditions
def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 18

def domain_of_f (x : ℝ) : Prop := ∃ (y : ℝ), y = 1 / (⌊quadratic_expression x⌋)

-- Theorem statement
theorem domain_is_all_real : ∀ x : ℝ, domain_of_f x :=
by
  sorry

end domain_is_all_real_l233_23300


namespace g_zero_value_l233_23332

variables {R : Type*} [Ring R]

def polynomial_h (f g h : Polynomial R) : Prop :=
  h = f * g

def constant_term (p : Polynomial R) : R :=
  p.coeff 0

variables {f g h : Polynomial R}

theorem g_zero_value
  (Hf : constant_term f = 6)
  (Hh : constant_term h = -18)
  (H : polynomial_h f g h) :
  g.coeff 0 = -3 :=
by
  sorry

end g_zero_value_l233_23332


namespace gpa_of_entire_class_l233_23340

def students : ℕ := 200

def gpa1_num : ℕ := 18 * students / 100
def gpa2_num : ℕ := 27 * students / 100
def gpa3_num : ℕ := 22 * students / 100
def gpa4_num : ℕ := 12 * students / 100
def gpa5_num : ℕ := students - (gpa1_num + gpa2_num + gpa3_num + gpa4_num)

def gpa1 : ℕ := 58
def gpa2 : ℕ := 63
def gpa3 : ℕ := 69
def gpa4 : ℕ := 75
def gpa5 : ℕ := 85

def total_points : ℕ :=
  (gpa1_num * gpa1) + (gpa2_num * gpa2) + (gpa3_num * gpa3) + (gpa4_num * gpa4) + (gpa5_num * gpa5)

def class_gpa : ℚ := total_points / students

theorem gpa_of_entire_class :
  class_gpa = 69.48 := 
  by
  sorry

end gpa_of_entire_class_l233_23340


namespace necessary_but_not_sufficient_cond_l233_23311

noncomputable
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_cond (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (hseq : geometric_sequence a a1 q)
  (hpos : a1 > 0) :
  (q < 0 ↔ (∀ n : ℕ, a (2 * n + 1) + a (2 * n + 2) < 0)) :=
sorry

end necessary_but_not_sufficient_cond_l233_23311


namespace second_train_start_time_l233_23359

-- Define the conditions as hypotheses
def station_distance : ℝ := 200
def speed_train_A : ℝ := 20
def speed_train_B : ℝ := 25
def meet_time : ℝ := 12 - 7 -- Time they meet after the first train starts, in hours.

-- The theorem statement corresponding to the proof problem
theorem second_train_start_time :
  ∃ T : ℝ, 0 <= T ∧ T <= 5 ∧ (5 * speed_train_A) + ((5 - T) * speed_train_B) = station_distance → T = 1 :=
by
  -- Placeholder for actual proof
  sorry

end second_train_start_time_l233_23359


namespace inverse_proportion_function_l233_23308

theorem inverse_proportion_function (x y : ℝ) (h : y = 6 / x) : x * y = 6 :=
by
  sorry

end inverse_proportion_function_l233_23308


namespace remainder_of_poly_division_l233_23304

theorem remainder_of_poly_division :
  ∀ (x : ℂ), ((x + 1)^2048) % (x^2 - x + 1) = x + 1 :=
by
  sorry

end remainder_of_poly_division_l233_23304


namespace part1_combined_time_part2_copier_A_insufficient_part3_combined_after_repair_l233_23391

-- Definitions for times needed by copiers A and B
def time_A : ℕ := 90
def time_B : ℕ := 60

-- (1) Combined time for both copiers
theorem part1_combined_time : 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 36 = 1 := 
by sorry

-- (2) Time left for copier A alone
theorem part2_copier_A_insufficient (mins_combined : ℕ) (time_left : ℕ) : 
  mins_combined = 30 → time_left = 13 → 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 30 + time_left / (time_A : ℝ) ≠ 1 := 
by sorry

-- (3) Combined time with B after repair is sufficient
theorem part3_combined_after_repair (mins_combined : ℕ) (mins_repair_B : ℕ) (time_left : ℕ) : 
  mins_combined = 30 → mins_repair_B = 9 → time_left = 13 →
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 30 + 9 / (time_A : ℝ) + 
  (1 / (time_A : ℝ) + 1 / (time_B : ℝ)) * 2.4 = 1 := 
by sorry

end part1_combined_time_part2_copier_A_insufficient_part3_combined_after_repair_l233_23391


namespace units_digit_of_product_l233_23333

-- Define the three given even composite numbers
def a := 4
def b := 6
def c := 8

-- Define the product of the three numbers
def product := a * b * c

-- State the units digit of the product
theorem units_digit_of_product : product % 10 = 2 :=
by
  -- Proof is skipped here
  sorry

end units_digit_of_product_l233_23333


namespace domain_of_function_l233_23373

def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1
def g (x : ℝ) : ℝ := x^2 - 9

theorem domain_of_function :
  {x : ℝ | g x ≠ 0} = {x : ℝ | x < -3} ∪ {x : ℝ | -3 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_function_l233_23373


namespace count_irreducible_fractions_l233_23318

theorem count_irreducible_fractions (s : Finset ℕ) (h1 : ∀ n ∈ s, 15*n > 15/16) (h2 : ∀ n ∈ s, n < 1) (h3 : ∀ n ∈ s, Nat.gcd n 15 = 1) :
  s.card = 8 := 
sorry

end count_irreducible_fractions_l233_23318


namespace Derek_test_score_l233_23327

def Grant_score (John_score : ℕ) : ℕ := John_score + 10
def John_score (Hunter_score : ℕ) : ℕ := 2 * Hunter_score
def Hunter_score : ℕ := 45
def Sarah_score (Grant_score : ℕ) : ℕ := Grant_score - 5
def Derek_score (John_score Grant_score : ℕ) : ℕ := (John_score + Grant_score) / 2

theorem Derek_test_score :
  Derek_score (John_score Hunter_score) (Grant_score (John_score Hunter_score)) = 95 :=
  by
  -- proof here
  sorry

end Derek_test_score_l233_23327


namespace avg_pages_hr_difference_l233_23335

noncomputable def avg_pages_hr_diff (total_pages_ryan : ℕ) (hours_ryan : ℕ) (books_brother : ℕ) (pages_per_book : ℕ) (hours_brother : ℕ) : ℚ :=
  (total_pages_ryan / hours_ryan : ℚ) - (books_brother * pages_per_book / hours_brother : ℚ)

theorem avg_pages_hr_difference :
  avg_pages_hr_diff 4200 78 15 250 90 = 12.18 :=
by
  sorry

end avg_pages_hr_difference_l233_23335


namespace beka_distance_l233_23390

theorem beka_distance (jackson_distance : ℕ) (beka_more_than_jackson : ℕ) :
  jackson_distance = 563 → beka_more_than_jackson = 310 → 
  (jackson_distance + beka_more_than_jackson = 873) :=
by
  sorry

end beka_distance_l233_23390


namespace min_length_intersection_l233_23347

theorem min_length_intersection (m n : ℝ) (h_m1 : 0 ≤ m) (h_m2 : m + 7 / 10 ≤ 1) 
                                (h_n1 : 2 / 5 ≤ n) (h_n2 : n ≤ 1) : 
  ∃ (min_length : ℝ), min_length = 1 / 10 :=
by
  sorry

end min_length_intersection_l233_23347


namespace number_of_real_solutions_l233_23350

noncomputable def system_of_equations (n : ℕ) (a b c : ℝ) (x : Fin n → ℝ) : Prop :=
∀ i : Fin n, a * (x i) ^ 2 + b * (x i) + c = x (⟨(i + 1) % n, sorry⟩)

theorem number_of_real_solutions
  (a b c : ℝ)
  (h : a ≠ 0)
  (n : ℕ)
  (x : Fin n → ℝ) :
  (b - 1) ^ 2 - 4 * a * c < 0 → ¬(∃ x : Fin n → ℝ, system_of_equations n a b c x) ∧
  (b - 1) ^ 2 - 4 * a * c = 0 → ∃! x : Fin n → ℝ, system_of_equations n a b c x ∧
  (b - 1) ^ 2 - 4 * a * c > 0 → ∃ x : Fin n → ℝ, ∃ y : Fin n → ℝ, x ≠ y ∧ system_of_equations n a b c x ∧ system_of_equations n a b c y := 
sorry

end number_of_real_solutions_l233_23350


namespace sum_of_first_eleven_terms_l233_23331

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_first_eleven_terms 
  (h_arith : is_arithmetic_sequence a)
  (h_S : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_condition : 2 * a 7 - a 8 = 5) :
  S 11 = 55 :=
sorry

end sum_of_first_eleven_terms_l233_23331


namespace find_number_l233_23387

theorem find_number : ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 51 ∧ x = 50 :=
by
  sorry

end find_number_l233_23387


namespace monica_expected_winnings_l233_23334

def monica_die_winnings : List ℤ := [2, 3, 5, 7, 0, 0, 0, -4]

def expected_value (values : List ℤ) : ℚ :=
  (List.sum values) / (values.length : ℚ)

theorem monica_expected_winnings :
  expected_value monica_die_winnings = 1.625 := by
  sorry

end monica_expected_winnings_l233_23334


namespace fourth_term_geometric_progression_l233_23323

theorem fourth_term_geometric_progression (x : ℝ) (h : ∀ n : ℕ, 0 < n → 
  (x ≠ 0 ∧ (2 * (x) + 2 * (n - 1)) ≠ 0 ∧ (3 * (x) + 3 * (n - 1)) ≠ 0)
  → ((2 * x + 2) / x) = (3 * x + 3) / (2 * x + 2)) : 
  ∃ r : ℝ, r = -13.5 := 
by 
  sorry

end fourth_term_geometric_progression_l233_23323


namespace domain_of_v_l233_23312

-- Define the function v
noncomputable def v (x y : ℝ) : ℝ := 1 / (x^(2/3) - y^(2/3))

-- State the domain of v
def domain_v : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≠ p.2 }

-- State the main theorem
theorem domain_of_v :
  ∀ x y : ℝ, x ≠ y ↔ (x, y) ∈ domain_v :=
by
  intro x y
  -- We don't need to provide proof
  sorry

end domain_of_v_l233_23312


namespace f_recurrence_l233_23394

noncomputable def f (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem f_recurrence (n : ℕ) : f (n + 1) - f (n - 1) = (3 * Real.sqrt 7 / 14) * f n := 
  sorry

end f_recurrence_l233_23394


namespace units_digit_of_52_cubed_plus_29_cubed_l233_23320

-- Define the units digit of a number n
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions as definitions in Lean
def units_digit_of_2_cubed : ℕ := units_digit (2^3)  -- 8
def units_digit_of_9_cubed : ℕ := units_digit (9^3)  -- 9

-- The main theorem to prove
theorem units_digit_of_52_cubed_plus_29_cubed : units_digit (52^3 + 29^3) = 7 :=
by
  sorry

end units_digit_of_52_cubed_plus_29_cubed_l233_23320


namespace find_r_l233_23337

variable {x y r k : ℝ}

theorem find_r (h1 : y^2 + 4 * y + 4 + Real.sqrt (x + y + k) = 0)
               (h2 : r = |x * y|) :
    r = 2 :=
by
  sorry

end find_r_l233_23337


namespace special_lines_count_l233_23322

noncomputable def count_special_lines : ℕ :=
  sorry

theorem special_lines_count :
  count_special_lines = 3 :=
by sorry

end special_lines_count_l233_23322


namespace max_value_fraction_l233_23363

theorem max_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (∀ x y : ℝ, (0 < x → 0 < y → (x / (2 * x + y) + y / (x + 2 * y)) ≤ 2 / 3)) :=
by
  sorry

end max_value_fraction_l233_23363


namespace min_correct_answers_for_score_above_60_l233_23310

theorem min_correct_answers_for_score_above_60 :
  ∃ (x : ℕ), 6 * x - 2 * (15 - x) > 60 ∧ x = 12 :=
by
  sorry

end min_correct_answers_for_score_above_60_l233_23310


namespace remainder_of_product_mod_7_l233_23364

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l233_23364


namespace log_comparison_theorem_CauchySchwarz_inequality_theorem_trigonometric_minimum_theorem_l233_23371

noncomputable def log_comparison (n : ℕ) (hn : 0 < n) : Prop := 
  Real.log n / Real.log (n + 1) < Real.log (n + 1) / Real.log (n + 2)

theorem log_comparison_theorem (n : ℕ) (hn : 0 < n) : log_comparison n hn := 
  sorry

def inequality_CauchySchwarz (a b x y : ℝ) : Prop :=
  (a*a + b*b) * (x*x + y*y) ≥ (a*x + b*y) * (a*x + b*y)

theorem CauchySchwarz_inequality_theorem (a b x y : ℝ) : inequality_CauchySchwarz a b x y :=
  sorry

noncomputable def trigonometric_minimum (x : ℝ) : ℝ := 
  (Real.sin x)^2 + (Real.cos x)^2

theorem trigonometric_minimum_theorem : ∀ x : ℝ, trigonometric_minimum x ≥ 9 :=
  sorry

end log_comparison_theorem_CauchySchwarz_inequality_theorem_trigonometric_minimum_theorem_l233_23371
