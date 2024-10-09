import Mathlib

namespace pyramid_coloring_methods_l2183_218309

theorem pyramid_coloring_methods : 
  ∀ (P A B C D : ℕ),
    (P ≠ A) ∧ (P ≠ B) ∧ (P ≠ C) ∧ (P ≠ D) ∧
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D) ∧
    (P < 5) ∧ (A < 5) ∧ (B < 5) ∧ (C < 5) ∧ (D < 5) →
  ∃! (num_methods : ℕ), num_methods = 420 :=
by
  sorry

end pyramid_coloring_methods_l2183_218309


namespace number_of_zeros_of_g_l2183_218371

open Real

noncomputable def g (x : ℝ) : ℝ := cos (π * log x + x)

theorem number_of_zeros_of_g : ¬ ∃ (x : ℝ), 1 < x ∧ x < exp 2 ∧ g x = 0 :=
sorry

end number_of_zeros_of_g_l2183_218371


namespace sample_size_of_survey_l2183_218367

def eighth_grade_students : ℕ := 350
def selected_students : ℕ := 50

theorem sample_size_of_survey : selected_students = 50 :=
by sorry

end sample_size_of_survey_l2183_218367


namespace rational_sum_zero_l2183_218375

theorem rational_sum_zero {a b c : ℚ} (h : (a + b + c) * (a + b - c) = 4 * c^2) : a + b = 0 := 
sorry

end rational_sum_zero_l2183_218375


namespace smallest_k_divides_polynomial_l2183_218348

theorem smallest_k_divides_polynomial :
  ∃ k : ℕ, 0 < k ∧ (∀ z : ℂ, (z^10 + z^9 + z^8 + z^6 + z^5 + z^4 + z + 1) ∣ (z^k - 1)) ∧ k = 84 :=
by
  sorry

end smallest_k_divides_polynomial_l2183_218348


namespace number_of_ways_to_form_team_l2183_218335

theorem number_of_ways_to_form_team (boys girls : ℕ) (select_boys select_girls : ℕ)
    (H_boys : boys = 7) (H_girls : girls = 9) (H_select_boys : select_boys = 2) (H_select_girls : select_girls = 3) :
    (Nat.choose boys select_boys) * (Nat.choose girls select_girls) = 1764 := by
  rw [H_boys, H_girls, H_select_boys, H_select_girls]
  sorry

end number_of_ways_to_form_team_l2183_218335


namespace find_m_value_l2183_218381

/-- 
If the function y = (m + 1)x^(m^2 + 3m + 4) is a quadratic function, 
then the value of m is -2.
--/
theorem find_m_value 
  (m : ℝ)
  (h1 : m^2 + 3 * m + 4 = 2)
  (h2 : m + 1 ≠ 0) : 
  m = -2 := 
sorry

end find_m_value_l2183_218381


namespace min_value_quadratic_expr_l2183_218395

-- Define the quadratic function
def quadratic_expr (x : ℝ) : ℝ := 8 * x^2 - 24 * x + 1729

-- State the theorem to prove the minimum value
theorem min_value_quadratic_expr : (∃ x : ℝ, ∀ y : ℝ, quadratic_expr y ≥ quadratic_expr x) ∧ ∃ x : ℝ, quadratic_expr x = 1711 :=
by
  -- The proof will go here
  sorry

end min_value_quadratic_expr_l2183_218395


namespace cannot_divide_1980_into_four_groups_l2183_218324

theorem cannot_divide_1980_into_four_groups :
  ¬∃ (S₁ S₂ S₃ S₄ : ℕ),
    S₂ = S₁ + 10 ∧
    S₃ = S₂ + 10 ∧
    S₄ = S₃ + 10 ∧
    (1 + 1980) * 1980 / 2 = S₁ + S₂ + S₃ + S₄ := 
sorry

end cannot_divide_1980_into_four_groups_l2183_218324


namespace am_gm_inequality_l2183_218337

-- Let's define the problem statement
theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : a + b + c ≥ 3 := by
  sorry

end am_gm_inequality_l2183_218337


namespace trillion_value_l2183_218360

def ten_thousand : ℕ := 10^4
def million : ℕ := 10^6
def billion : ℕ := ten_thousand * million

theorem trillion_value : (ten_thousand * ten_thousand * billion) = 10^16 :=
by
  sorry

end trillion_value_l2183_218360


namespace system_solution_l2183_218345

theorem system_solution (x y : ℝ) (h1 : x + 5*y = 5) (h2 : 3*x - y = 3) : x + y = 2 := 
by
  sorry

end system_solution_l2183_218345


namespace time_to_cover_escalator_l2183_218307

variable (v_e v_p L : ℝ)

theorem time_to_cover_escalator
  (h_v_e : v_e = 15)
  (h_v_p : v_p = 5)
  (h_L : L = 180) :
  (L / (v_e + v_p) = 9) :=
by
  -- Set up the given conditions
  rw [h_v_e, h_v_p, h_L]
  -- This will now reduce to proving 180 / (15 + 5) = 9
  sorry

end time_to_cover_escalator_l2183_218307


namespace find_x_l2183_218334

theorem find_x :
  (2 + 3 = 5) →
  (3 + 4 = 7) →
  (1 / (2 + 3)) * (1 / (3 + 4)) = 1 / (x + 5) →
  x = 30 :=
by
  intros
  sorry

end find_x_l2183_218334


namespace Lauryn_earnings_l2183_218303

variables (L : ℝ)

theorem Lauryn_earnings (h1 : 0.70 * L + L = 3400) : L = 2000 :=
sorry

end Lauryn_earnings_l2183_218303


namespace no_solution_exists_l2183_218377

theorem no_solution_exists (m n : ℕ) : ¬ (m^2 = n^2 + 2014) :=
by
  sorry

end no_solution_exists_l2183_218377


namespace average_of_first_21_multiples_of_7_l2183_218338

theorem average_of_first_21_multiples_of_7 :
  let a1 := 7
  let d := 7
  let n := 21
  let an := a1 + (n - 1) * d
  let Sn := n / 2 * (a1 + an)
  Sn / n = 77 :=
by
  let a1 := 7
  let d := 7
  let n := 21
  let an := a1 + (n - 1) * d
  let Sn := n / 2 * (a1 + an)
  have h1 : an = 147 := by
    sorry
  have h2 : Sn = 1617 := by
    sorry
  have h3 : Sn / n = 77 := by
    sorry
  exact h3

end average_of_first_21_multiples_of_7_l2183_218338


namespace g_extreme_values_l2183_218346

-- Definitions based on the conditions
def f (x : ℝ) := x^3 - 2 * x^2 + x
def g (x : ℝ) := f x + 1

-- Theorem statement
theorem g_extreme_values : 
  (g (1/3) = 31/27) ∧ (g 1 = 1) := sorry

end g_extreme_values_l2183_218346


namespace actual_length_of_tunnel_in_km_l2183_218350

-- Define the conditions
def scale_factor : ℝ := 30000
def length_on_map_cm : ℝ := 7

-- Using the conditions, we need to prove the actual length is 2.1 km
theorem actual_length_of_tunnel_in_km :
  (length_on_map_cm * scale_factor / 100000) = 2.1 :=
by sorry

end actual_length_of_tunnel_in_km_l2183_218350


namespace num_two_digit_math_representation_l2183_218357

-- Define the problem space
def unique_digits (n : ℕ) : Prop := 
  n >= 1 ∧ n <= 9

-- Representation of the characters' assignment
def representation (x y z w : ℕ) : Prop :=
  unique_digits x ∧ unique_digits y ∧ unique_digits z ∧ unique_digits w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧ 
  x = z ∧ 3 * (10 * y + z) = 10 * w + x

-- The main theorem to prove
theorem num_two_digit_math_representation : 
  ∃ x y z w, representation x y z w :=
sorry

end num_two_digit_math_representation_l2183_218357


namespace incorrect_option_C_l2183_218390

theorem incorrect_option_C (a b : ℝ) (h1 : a > b) (h2 : b > a + b) : ¬ (ab > (a + b)^2) :=
by {
  sorry
}

end incorrect_option_C_l2183_218390


namespace number_of_lines_at_least_two_points_4_by_4_grid_l2183_218333

-- Definition of 4-by-4 grid
def grid : Type := (Fin 4) × (Fin 4)

-- Definition of a line passing through at least two points in this grid
def line_through_at_least_two_points (points : List grid) : Prop := 
  points.length ≥ 2
  ∧ ∃ m b, ∀ (x y : Fin 4 × Fin 4), (x ∈ points ∧ y ∈ points) → (y.snd : ℕ) = m * (x.fst : ℕ) + b

-- Defining the total number of points choosing 2 out of 16
def total_points : Nat := Nat.choose 16 2

-- Defining the overcount of vertical, horizontal,
-- major diagonals, and secondary diagonals lines
def overcount : Nat := 8 + 2 + 4

-- Total distinct count of lines passing through at least two points
def correct_answer : Nat := total_points - overcount

-- Main theorem stating that the total count is 106
theorem number_of_lines_at_least_two_points_4_by_4_grid : correct_answer = 106 := 
by
  sorry

end number_of_lines_at_least_two_points_4_by_4_grid_l2183_218333


namespace probability_no_3by3_red_grid_correct_l2183_218342

noncomputable def probability_no_3by3_red_grid : ℚ := 813 / 819

theorem probability_no_3by3_red_grid_correct :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∀ square : Fin 4 × Fin 4, square.1 = square.2 ∨ square.1 ≠ square.2) →
    m = 813 ∧ n = 819 ∧ probability_no_3by3_red_grid = m / n :=
by
  sorry

end probability_no_3by3_red_grid_correct_l2183_218342


namespace books_loaned_out_l2183_218383

theorem books_loaned_out (initial_books returned_percent final_books : ℕ) (h1 : initial_books = 75) (h2 : returned_percent = 65) (h3 : final_books = 61) : 
  ∃ x : ℕ, initial_books - final_books = x - (returned_percent * x / 100) ∧ x = 40 :=
by {
  sorry 
}

end books_loaned_out_l2183_218383


namespace find_exact_speed_l2183_218314

variable (d t v : ℝ)

-- Conditions as Lean definitions
def distance_eq1 : d = 50 * (t - 1/12) := sorry
def distance_eq2 : d = 70 * (t + 1/12) := sorry
def travel_time : t = 1/2 := sorry -- deduced travel time from the equations and given conditions
def correct_speed : v = 42 := sorry -- Mr. Bird needs to drive at 42 mph to be exactly on time

-- Lean 4 statement proving the required speed is 42 mph
theorem find_exact_speed : v = d / t :=
  by
    sorry

end find_exact_speed_l2183_218314


namespace projectiles_meet_time_l2183_218301

def distance : ℕ := 2520
def speed1 : ℕ := 432
def speed2 : ℕ := 576
def combined_speed : ℕ := speed1 + speed2

theorem projectiles_meet_time :
  (distance * 60) / combined_speed = 150 := 
by
  sorry

end projectiles_meet_time_l2183_218301


namespace find_number_to_be_multiplied_l2183_218327

-- Define the conditions of the problem
variable (x : ℕ)

-- Condition 1: The correct multiplication would have been 43x
-- Condition 2: The actual multiplication done was 34x
-- Condition 3: The difference between correct and actual result is 1242

theorem find_number_to_be_multiplied (h : 43 * x - 34 * x = 1242) : 
  x = 138 := by
  sorry

end find_number_to_be_multiplied_l2183_218327


namespace product_of_three_numbers_l2183_218343

theorem product_of_three_numbers (x y z n : ℝ)
  (h_sum : x + y + z = 180)
  (h_n_eq_8x : n = 8 * x)
  (h_n_eq_y_minus_10 : n = y - 10)
  (h_n_eq_z_plus_10 : n = z + 10) :
  x * y * z = (180 / 17) * ((1440 / 17) ^ 2 - 100) := by
  sorry

end product_of_three_numbers_l2183_218343


namespace k_value_l2183_218351

theorem k_value (k : ℝ) (h : (k / 4) + (-k / 3) = 2) : k = -24 :=
by
  sorry

end k_value_l2183_218351


namespace sarah_bought_3_bottle_caps_l2183_218349

theorem sarah_bought_3_bottle_caps
  (orig_caps : ℕ)
  (new_caps : ℕ)
  (h_orig_caps : orig_caps = 26)
  (h_new_caps : new_caps = 29) :
  new_caps - orig_caps = 3 :=
by
  sorry

end sarah_bought_3_bottle_caps_l2183_218349


namespace ratio_of_heights_l2183_218310

def min_height := 140
def brother_height := 180
def grow_needed := 20

def mary_height := min_height - grow_needed
def height_ratio := mary_height / brother_height

theorem ratio_of_heights : height_ratio = (2 / 3) := 
  sorry

end ratio_of_heights_l2183_218310


namespace find_b8_l2183_218329

noncomputable section

def increasing_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = b (n + 1) + b n

axiom b_seq : ℕ → ℕ

axiom seq_inc : increasing_sequence b_seq

axiom b7_eq : b_seq 7 = 198

theorem find_b8 : b_seq 8 = 321 := by
  sorry

end find_b8_l2183_218329


namespace first_term_arithmetic_series_l2183_218316

theorem first_term_arithmetic_series 
  (a d : ℚ) 
  (h1 : 30 * (2 * a + 59 * d) = 240)
  (h2 : 30 * (2 * a + 179 * d) = 3600) : 
  a = -353 / 15 :=
by
  have eq1 : 2 * a + 59 * d = 8 := by sorry
  have eq2 : 2 * a + 179 * d = 120 := by sorry
  sorry

end first_term_arithmetic_series_l2183_218316


namespace opposite_of_neg_three_sevenths_l2183_218322

theorem opposite_of_neg_three_sevenths:
  ∀ x : ℚ, (x = -3 / 7) → (∃ y : ℚ, y + x = 0 ∧ y = 3 / 7) :=
by
  sorry

end opposite_of_neg_three_sevenths_l2183_218322


namespace tan_150_eq_neg_inv_sqrt3_l2183_218372

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l2183_218372


namespace molecular_weight_of_Carbonic_acid_l2183_218356

theorem molecular_weight_of_Carbonic_acid :
  let H_weight := 1.008
  let C_weight := 12.011
  let O_weight := 15.999
  let H_atoms := 2
  let C_atoms := 1
  let O_atoms := 3
  (H_atoms * H_weight + C_atoms * C_weight + O_atoms * O_weight) = 62.024 :=
by 
  let H_weight := 1.008
  let C_weight := 12.011
  let O_weight := 15.999
  let H_atoms := 2
  let C_atoms := 1
  let O_atoms := 3
  sorry

end molecular_weight_of_Carbonic_acid_l2183_218356


namespace area_S_inequality_l2183_218330

noncomputable def F (t : ℝ) : ℝ := 2 * (t - ⌊t⌋)

def S (t : ℝ) : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 - F t) * (p.1 - F t) + p.2 * p.2 ≤ (F t) * (F t) }

theorem area_S_inequality (t : ℝ) : 0 ≤ π * (F t) ^ 2 ∧ π * (F t) ^ 2 ≤ 4 * π := 
by sorry

end area_S_inequality_l2183_218330


namespace value_of_expression_l2183_218353

-- Given conditions as definitions
axiom cond1 (x y : ℝ) : -x + 2*y = 5

-- The theorem we want to prove
theorem value_of_expression (x y : ℝ) (h : -x + 2*y = 5) : 
  5 * (x - 2 * y)^2 - 3 * (x - 2 * y) - 60 = 80 :=
by
  -- The proof part is omitted here.
  sorry

end value_of_expression_l2183_218353


namespace find_positive_integer_solutions_l2183_218332

theorem find_positive_integer_solutions :
  ∃ a b : ℤ, a > 0 ∧ b > 0 ∧ (1 / (a : ℚ)) - (1 / (b : ℚ)) = 1 / 37 ∧ (a, b) = (38, 1332) :=
by
  sorry

end find_positive_integer_solutions_l2183_218332


namespace divide_square_into_smaller_squares_l2183_218382

def P (n : ℕ) : Prop := sorry /- Define the property of dividing a square into n smaller squares -/

theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
  sorry

end divide_square_into_smaller_squares_l2183_218382


namespace product_equals_sum_only_in_two_cases_l2183_218313

theorem product_equals_sum_only_in_two_cases (x y : ℤ) : 
  x * y = x + y ↔ (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) :=
by 
  sorry

end product_equals_sum_only_in_two_cases_l2183_218313


namespace geometric_sequence_common_ratio_l2183_218354

theorem geometric_sequence_common_ratio (a : ℕ → ℝ)
  (h : ∀ n, a n * a (n + 1) = 16^n) :
  ∃ r : ℝ, r = 4 ∧ ∀ n, a (n + 1) = a n * r :=
sorry

end geometric_sequence_common_ratio_l2183_218354


namespace sin_2017pi_div_3_l2183_218370

theorem sin_2017pi_div_3 : Real.sin (2017 * Real.pi / 3) = Real.sqrt 3 / 2 := 
  sorry

end sin_2017pi_div_3_l2183_218370


namespace determine_x_l2183_218344

theorem determine_x 
  (w : ℤ) (hw : w = 90)
  (z : ℤ) (hz : z = 4 * w + 40)
  (y : ℤ) (hy : y = 3 * z + 15)
  (x : ℤ) (hx : x = 2 * y + 6) :
  x = 2436 := 
by
  sorry

end determine_x_l2183_218344


namespace reflection_correct_l2183_218325

def point := (ℝ × ℝ)

def reflect_x_axis (p : point) : point :=
  (p.1, -p.2)

def M : point := (3, 2)

theorem reflection_correct : reflect_x_axis M = (3, -2) :=
  sorry

end reflection_correct_l2183_218325


namespace maximal_inradius_of_tetrahedron_l2183_218391

-- Define the properties and variables
variables (A B C D : ℝ) (h_A h_B h_C h_D : ℝ) (V r : ℝ)

-- Assumptions
variable (h_A_ge_1 : h_A ≥ 1)
variable (h_B_ge_1 : h_B ≥ 1)
variable (h_C_ge_1 : h_C ≥ 1)
variable (h_D_ge_1 : h_D ≥ 1)

-- Volume expressed in terms of altitudes and face areas
axiom vol_eq_Ah : V = (1 / 3) * A * h_A
axiom vol_eq_Bh : V = (1 / 3) * B * h_B
axiom vol_eq_Ch : V = (1 / 3) * C * h_C
axiom vol_eq_Dh : V = (1 / 3) * D * h_D

-- Volume expressed in terms of inradius and sum of face areas
axiom vol_eq_inradius : V = (1 / 3) * (A + B + C + D) * r

-- The theorem to prove
theorem maximal_inradius_of_tetrahedron : r = 1 / 4 :=
sorry

end maximal_inradius_of_tetrahedron_l2183_218391


namespace trig_fraction_identity_l2183_218352

noncomputable def cos_63 := Real.cos (Real.pi * 63 / 180)
noncomputable def cos_3 := Real.cos (Real.pi * 3 / 180)
noncomputable def cos_87 := Real.cos (Real.pi * 87 / 180)
noncomputable def cos_27 := Real.cos (Real.pi * 27 / 180)
noncomputable def cos_132 := Real.cos (Real.pi * 132 / 180)
noncomputable def cos_72 := Real.cos (Real.pi * 72 / 180)
noncomputable def cos_42 := Real.cos (Real.pi * 42 / 180)
noncomputable def cos_18 := Real.cos (Real.pi * 18 / 180)
noncomputable def tan_24 := Real.tan (Real.pi * 24 / 180)

theorem trig_fraction_identity :
  (cos_63 * cos_3 - cos_87 * cos_27) / 
  (cos_132 * cos_72 - cos_42 * cos_18) = 
  -tan_24 := 
by
  sorry

end trig_fraction_identity_l2183_218352


namespace other_person_time_to_complete_job_l2183_218388

-- Define the conditions
def SureshTime : ℕ := 15
def SureshWorkHours : ℕ := 9
def OtherPersonWorkHours : ℕ := 4

-- The proof problem: Prove that the other person can complete the job in 10 hours.
theorem other_person_time_to_complete_job (x : ℕ) 
  (h1 : ∀ SureshWorkHours SureshTime, SureshWorkHours * (1 / SureshTime) = (SureshWorkHours / SureshTime) ∧ 
       4 * (SureshWorkHours / SureshTime / 4) = 1) : 
  (x = 10) :=
sorry

end other_person_time_to_complete_job_l2183_218388


namespace sum_of_fractions_l2183_218306

theorem sum_of_fractions :
  (1 / (1^2 * 2^2) + 1 / (2^2 * 3^2) + 1 / (3^2 * 4^2) + 1 / (4^2 * 5^2)
  + 1 / (5^2 * 6^2) + 1 / (6^2 * 7^2)) = 48 / 49 := 
by
  sorry

end sum_of_fractions_l2183_218306


namespace union_A_B_complement_U_A_intersection_B_range_of_a_l2183_218311

-- Define the sets A, B, C, and U
def setA (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 8
def setB (x : ℝ) : Prop := 1 < x ∧ x < 6
def setC (a : ℝ) (x : ℝ) : Prop := x > a
def U (x : ℝ) : Prop := True  -- U being the universal set of all real numbers

-- Define complements and intersections
def complement (A : ℝ → Prop) (x : ℝ) : Prop := ¬ A x
def intersection (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∧ B x
def union (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∨ B x

-- Proof problems
theorem union_A_B : ∀ x, union setA setB x ↔ (1 < x ∧ x ≤ 8) :=
by 
  intros x
  sorry

theorem complement_U_A_intersection_B : ∀ x, intersection (complement setA) setB x ↔ (1 < x ∧ x < 2) :=
by 
  intros x
  sorry

theorem range_of_a (a : ℝ) : (∃ x, intersection setA (setC a) x) → a < 8 :=
by
  intros h
  sorry

end union_A_B_complement_U_A_intersection_B_range_of_a_l2183_218311


namespace find_function_g_l2183_218392

noncomputable def g (x : ℝ) : ℝ := (5^x - 3^x) / 8

theorem find_function_g (x y : ℝ) (h1 : g 2 = 2) (h2 : ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y) :
  g x = (5^x - 3^x) / 8 :=
by
  sorry

end find_function_g_l2183_218392


namespace factorize_expr_l2183_218304

theorem factorize_expr (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := 
sorry

end factorize_expr_l2183_218304


namespace constant_is_arithmetic_l2183_218364

def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem constant_is_arithmetic (a : ℕ → ℝ) (h : is_constant_sequence a) : is_arithmetic_sequence a := by
  sorry

end constant_is_arithmetic_l2183_218364


namespace rowing_upstream_speed_l2183_218319

def speed_in_still_water : ℝ := 31
def speed_downstream : ℝ := 37

def speed_stream : ℝ := speed_downstream - speed_in_still_water

def speed_upstream : ℝ := speed_in_still_water - speed_stream

theorem rowing_upstream_speed :
  speed_upstream = 25 := by
  sorry

end rowing_upstream_speed_l2183_218319


namespace max_at_pi_six_l2183_218347

theorem max_at_pi_six : ∃ (x : ℝ), (0 ≤ x ∧ x ≤ π / 2) ∧ (∀ y, (0 ≤ y ∧ y ≤ π / 2) → (x + 2 * Real.cos x) ≥ (y + 2 * Real.cos y)) ∧ x = π / 6 := sorry

end max_at_pi_six_l2183_218347


namespace solve_for_x_l2183_218340

def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

theorem solve_for_x (x : ℝ) : (custom_mul 3 (custom_mul 6 x) = 2) → (x = 19 / 2) :=
sorry

end solve_for_x_l2183_218340


namespace gcd_example_l2183_218361

theorem gcd_example : Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end gcd_example_l2183_218361


namespace victor_weight_is_correct_l2183_218393

-- Define the given conditions
def bear_daily_food : ℕ := 90
def victors_food_in_3_weeks : ℕ := 15
def days_in_3_weeks : ℕ := 21

-- Define the equivalent weight of Victor based on the given conditions
def victor_weight : ℕ := bear_daily_food * days_in_3_weeks / victors_food_in_3_weeks

-- Prove that the weight of Victor is 126 pounds
theorem victor_weight_is_correct : victor_weight = 126 := by
  sorry

end victor_weight_is_correct_l2183_218393


namespace polynomial_at_x_is_minus_80_l2183_218328

def polynomial (x : ℤ) : ℤ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def x_value : ℤ := 2

theorem polynomial_at_x_is_minus_80 : polynomial x_value = -80 := 
by
  sorry

end polynomial_at_x_is_minus_80_l2183_218328


namespace matching_pair_probability_l2183_218362

theorem matching_pair_probability :
  let total_socks := 22
  let blue_socks := 12
  let red_socks := 10
  let total_ways := (total_socks * (total_socks - 1)) / 2
  let blue_ways := (blue_socks * (blue_socks - 1)) / 2
  let red_ways := (red_socks * (red_socks - 1)) / 2
  let matching_ways := blue_ways + red_ways
  total_ways = 231 →
  blue_ways = 66 →
  red_ways = 45 →
  matching_ways = 111 →
  (matching_ways : ℝ) / total_ways = 111 / 231 := by sorry

end matching_pair_probability_l2183_218362


namespace range_of_a_l2183_218341

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 3| - |x + 1| ≤ a^2 - 3 * a) ↔ a ≤ -1 ∨ 4 ≤ a := 
sorry

end range_of_a_l2183_218341


namespace slope_and_angle_of_inclination_l2183_218386

noncomputable def line_slope_and_inclination : Prop :=
  ∀ (x y : ℝ), (x - y - 3 = 0) → (∃ m : ℝ, m = 1) ∧ (∃ θ : ℝ, θ = 45)

theorem slope_and_angle_of_inclination (x y : ℝ) (h : x - y - 3 = 0) : line_slope_and_inclination :=
by
  sorry

end slope_and_angle_of_inclination_l2183_218386


namespace solve_equation_1_solve_equation_2_l2183_218305

theorem solve_equation_1 (x : ℝ) : 2 * (x + 1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6 := sorry

theorem solve_equation_2 (x : ℝ) : (1 / 2) * (x - 1)^3 = -4 ↔ x = -1 := sorry

end solve_equation_1_solve_equation_2_l2183_218305


namespace function_increasing_on_interval_l2183_218320

theorem function_increasing_on_interval {x : ℝ} (hx : x < 1) : 
  (-1/2) * x^2 + x + 4 < -1/2 * (x + 1)^2 + (x + 1) + 4 :=
sorry

end function_increasing_on_interval_l2183_218320


namespace smallest_integer_solution_of_inequality_l2183_218368

theorem smallest_integer_solution_of_inequality : ∃ x : ℤ, (3 * x ≥ x - 5) ∧ (∀ y : ℤ, 3 * y ≥ y - 5 → y ≥ -2) := 
sorry

end smallest_integer_solution_of_inequality_l2183_218368


namespace base_b_digit_sum_l2183_218315

theorem base_b_digit_sum :
  ∃ (b : ℕ), ((b^2 / 2 + b / 2) % b = 2) ∧ (b = 8) :=
by
  sorry

end base_b_digit_sum_l2183_218315


namespace rectangular_field_length_l2183_218378

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

noncomputable def length_rectangle (area width : ℝ) : ℝ :=
  area / width

theorem rectangular_field_length (base height width : ℝ) (h_base : base = 7.2) (h_height : height = 7) (h_width : width = 4) :
  length_rectangle (area_triangle base height) width = 6.3 :=
by
  -- sorry would be replaced by the actual proof.
  sorry

end rectangular_field_length_l2183_218378


namespace odd_function_m_value_l2183_218326

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x - m

theorem odd_function_m_value :
  ∃ m : ℝ, (∀ (x : ℝ), g (-x) m + g x m = 0) ∧ m = 2 :=
by
  sorry

end odd_function_m_value_l2183_218326


namespace card_area_after_shortening_l2183_218359

/-- Given a card with dimensions 3 inches by 7 inches, prove that 
  if the length is shortened by 1 inch and the width is shortened by 2 inches, 
  then the resulting area is 10 square inches. -/
theorem card_area_after_shortening :
  let length := 3
  let width := 7
  let new_length := length - 1
  let new_width := width - 2
  new_length * new_width = 10 :=
by
  let length := 3
  let width := 7
  let new_length := length - 1
  let new_width := width - 2
  show new_length * new_width = 10
  sorry

end card_area_after_shortening_l2183_218359


namespace rectangular_plot_breadth_l2183_218300

theorem rectangular_plot_breadth :
  ∀ (l b : ℝ), (l = 3 * b) → (l * b = 588) → (b = 14) :=
by
  intros l b h1 h2
  sorry

end rectangular_plot_breadth_l2183_218300


namespace bryden_receives_10_dollars_l2183_218374

-- Define the face value of one quarter
def face_value_quarter : ℝ := 0.25

-- Define the number of quarters Bryden has
def num_quarters : ℕ := 8

-- Define the multiplier for 500%
def multiplier : ℝ := 5

-- Calculate the total face value of eight quarters
def total_face_value : ℝ := num_quarters * face_value_quarter

-- Calculate the amount Bryden will receive
def amount_received : ℝ := total_face_value * multiplier

-- The proof goal: Bryden will receive 10 dollars
theorem bryden_receives_10_dollars : amount_received = 10 :=
by
  sorry

end bryden_receives_10_dollars_l2183_218374


namespace dozen_pen_cost_l2183_218376

-- Definitions based on the conditions
def cost_of_pen (x : ℝ) : ℝ := 5 * x
def cost_of_pencil (x : ℝ) : ℝ := x
def total_cost (x : ℝ) (y : ℝ) : ℝ := 3 * cost_of_pen x + y * cost_of_pencil x

open Classical
noncomputable def cost_dozen_pens (x : ℝ) : ℝ := 12 * cost_of_pen x

theorem dozen_pen_cost (x y : ℝ) (h : total_cost x y = 150) : cost_dozen_pens x = 60 * x :=
by
  sorry

end dozen_pen_cost_l2183_218376


namespace xy_identity_l2183_218373

theorem xy_identity (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : x^2 + y^2 = 5 :=
  sorry

end xy_identity_l2183_218373


namespace find_fg_l2183_218308

def f (x : ℕ) : ℕ := 3 * x^2 + 2
def g (x : ℕ) : ℕ := 4 * x + 1

theorem find_fg :
  f (g 3) = 509 :=
by
  sorry

end find_fg_l2183_218308


namespace algebraic_expression_value_l2183_218387

variable (a b : ℝ)

theorem algebraic_expression_value
  (h : a^2 + 2 * b^2 - 1 = 0) :
  (a - b)^2 + b * (2 * a + b) = 1 :=
by
  sorry

end algebraic_expression_value_l2183_218387


namespace one_number_is_zero_l2183_218369

variable {a b c : ℤ}
variable (cards : Fin 30 → ℤ)

theorem one_number_is_zero (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
    (h_cards : ∀ i : Fin 30, cards i = a ∨ cards i = b ∨ cards i = c)
    (h_sum_zero : ∀ (S : Finset (Fin 30)) (hS : S.card = 5),
        ∃ T : Finset (Fin 30), T.card = 5 ∧ (S ∪ T).sum cards = 0) :
    b = 0 := 
sorry

end one_number_is_zero_l2183_218369


namespace choice_first_question_range_of_P2_l2183_218336

theorem choice_first_question (P1 P2 a b : ℚ) (hP1 : P1 = 1/2) (hP2 : P2 = 1/3) :
  (P1 * (1 - P2) * a + P1 * P2 * (a + b) - P2 * (1 - P1) * b - P1 * P2 * (a + b) > 0) ↔ a > b / 2 :=
sorry

theorem range_of_P2 (a b P1 P2 : ℚ) (ha : a = 10) (hb : b = 20) (hP1 : P1 = 2/5) :
  P1 * (1 - P2) * a + P1 * P2 * (a + b) - P2 * (1 - P1) * b - P1 * P2 * (a + b) ≥ 0 ↔ (0 ≤ P2 ∧ P2 ≤ P1 / (2 - P1)) :=
sorry

end choice_first_question_range_of_P2_l2183_218336


namespace calc_3_pow_6_mul_4_pow_6_l2183_218363

theorem calc_3_pow_6_mul_4_pow_6 : (3^6) * (4^6) = 2985984 :=
by 
  sorry

end calc_3_pow_6_mul_4_pow_6_l2183_218363


namespace find_richards_score_l2183_218365

variable (R B : ℕ)

theorem find_richards_score (h1 : B = R - 14) (h2 : B = 48) : R = 62 := by
  sorry

end find_richards_score_l2183_218365


namespace ammonium_nitrate_formed_l2183_218358

-- Definitions based on conditions in the problem
def NH3_moles : ℕ := 3
def HNO3_moles (NH3 : ℕ) : ℕ := NH3 -- 1:1 molar ratio with NH3 for HNO3

-- Definition of the outcome
def NH4NO3_moles (NH3 NH4NO3 : ℕ) : Prop :=
  NH4NO3 = NH3

-- The theorem to prove that 3 moles of NH3 combined with sufficient HNO3 produces 3 moles of NH4NO3
theorem ammonium_nitrate_formed (NH3 NH4NO3 : ℕ) (h : NH3 = 3) :
  NH4NO3_moles NH3 NH4NO3 → NH4NO3 = 3 :=
by
  intro hn
  rw [h] at hn
  exact hn

end ammonium_nitrate_formed_l2183_218358


namespace sales_price_reduction_l2183_218385

theorem sales_price_reduction
  (current_sales : ℝ := 20)
  (current_profit_per_shirt : ℝ := 40)
  (sales_increase_per_dollar : ℝ := 2)
  (desired_profit : ℝ := 1200) :
  ∃ x : ℝ, (40 - x) * (20 + 2 * x) = 1200 ∧ x = 20 :=
by
  use 20
  sorry

end sales_price_reduction_l2183_218385


namespace find_2a_plus_b_l2183_218379

open Real

-- Define the given conditions
variables (a b : ℝ)

-- a and b are acute angles
axiom acute_a : 0 < a ∧ a < π / 2
axiom acute_b : 0 < b ∧ b < π / 2

axiom condition1 : 4 * sin a ^ 2 + 3 * sin b ^ 2 = 1
axiom condition2 : 4 * sin (2 * a) - 3 * sin (2 * b) = 0

-- Define the theorem we want to prove
theorem find_2a_plus_b : 2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l2183_218379


namespace smallest_k_l2183_218366

theorem smallest_k (a b : ℚ) (h_a_period : ∀ n, a ≠ (10^30 - 1) * n)
  (h_b_period : ∀ n, b ≠ (10^30 - 1) * n)
  (h_diff_period : ∀ n, a - b ≠ (10^15 - 1) * n) :
  ∃ k : ℕ, k = 6 ∧ (a + (k:ℚ) * b) ≠ (10^15 - 1) :=
sorry

end smallest_k_l2183_218366


namespace implies_neg_p_and_q_count_l2183_218317

-- Definitions of the logical conditions
variables (p q : Prop)

def cond1 : Prop := p ∧ q
def cond2 : Prop := p ∧ ¬ q
def cond3 : Prop := ¬ p ∧ q
def cond4 : Prop := ¬ p ∧ ¬ q

-- Negative of the statement "p and q are both true"
def neg_p_and_q := ¬ (p ∧ q)

-- The Lean 4 statement to prove
theorem implies_neg_p_and_q_count :
  (cond2 p q → neg_p_and_q p q) ∧ 
  (cond3 p q → neg_p_and_q p q) ∧ 
  (cond4 p q → neg_p_and_q p q) ∧ 
  ¬ (cond1 p q → neg_p_and_q p q) :=
sorry

end implies_neg_p_and_q_count_l2183_218317


namespace find_min_value_l2183_218398

theorem find_min_value (a x y : ℝ) (h : y = -x^2 + 3 * Real.log x) : ∃ x, ∃ y, (a - x)^2 + (a + 2 - y)^2 = 8 :=
by
  sorry

end find_min_value_l2183_218398


namespace sequence_arithmetic_and_find_an_l2183_218397

theorem sequence_arithmetic_and_find_an (a : ℕ → ℝ)
  (h1 : a 9 = 1 / 7)
  (h2 : ∀ n, a (n + 1) = a n / (3 * a n + 1)) :
  (∀ n, 1 / a (n + 1) = 3 + 1 / a n) ∧ (∀ n, a n = 1 / (3 * n - 20)) :=
by
  sorry

end sequence_arithmetic_and_find_an_l2183_218397


namespace two_times_difference_eq_20_l2183_218302

theorem two_times_difference_eq_20 (x y : ℕ) (hx : x = 30) (hy : y = 20) (hsum : x + y = 50) : 2 * (x - y) = 20 := by
  sorry

end two_times_difference_eq_20_l2183_218302


namespace tracy_feeds_dogs_times_per_day_l2183_218355

theorem tracy_feeds_dogs_times_per_day : 
  let cups_per_meal_per_dog := 1.5
  let dogs := 2
  let total_pounds_per_day := 4
  let cups_per_pound := 2.25
  (total_pounds_per_day * cups_per_pound) / (dogs * cups_per_meal_per_dog) = 3 :=
by
  sorry

end tracy_feeds_dogs_times_per_day_l2183_218355


namespace fifth_group_members_l2183_218331

-- Define the number of members in the choir
def total_members : ℕ := 150 

-- Define the number of members in each group
def group1 : ℕ := 18 
def group2 : ℕ := 29 
def group3 : ℕ := 34 
def group4 : ℕ := 23 

-- Define the fifth group as the remaining members
def group5 : ℕ := total_members - (group1 + group2 + group3 + group4)

theorem fifth_group_members : group5 = 46 := sorry

end fifth_group_members_l2183_218331


namespace butterflies_in_the_garden_l2183_218394

variable (total_butterflies : Nat) (fly_away : Nat)

def butterflies_left (total_butterflies : Nat) (fly_away : Nat) : Nat :=
  total_butterflies - fly_away

theorem butterflies_in_the_garden :
  (total_butterflies = 9) → (fly_away = 1 / 3 * total_butterflies) → butterflies_left total_butterflies fly_away = 6 :=
by
  intro h1 h2
  sorry

end butterflies_in_the_garden_l2183_218394


namespace tom_climbing_time_l2183_218384

theorem tom_climbing_time (elizabeth_time : ℕ) (multiplier : ℕ) 
  (h1 : elizabeth_time = 30) (h2 : multiplier = 4) : (elizabeth_time * multiplier) / 60 = 2 :=
by
  sorry

end tom_climbing_time_l2183_218384


namespace trains_meet_in_approx_17_45_seconds_l2183_218399

noncomputable def train_meet_time
  (length1 length2 distance_between : ℕ)
  (speed1_kmph speed2_kmph : ℕ)
  : ℕ :=
  let speed1_mps := (speed1_kmph * 1000) / 3600
  let speed2_mps := (speed2_kmph * 1000) / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := distance_between + length1 + length2
  total_distance / relative_speed

theorem trains_meet_in_approx_17_45_seconds :
  train_meet_time 100 200 660 90 108 = 17 := by
  sorry

end trains_meet_in_approx_17_45_seconds_l2183_218399


namespace jeans_more_than_scarves_l2183_218323

def num_ties := 34
def num_belts := 40
def num_black_shirts := 63
def num_white_shirts := 42
def num_jeans := (2 / 3) * (num_black_shirts + num_white_shirts)
def num_scarves := (1 / 2) * (num_ties + num_belts)

theorem jeans_more_than_scarves : num_jeans - num_scarves = 33 := by
  sorry

end jeans_more_than_scarves_l2183_218323


namespace regular_n_gon_center_inside_circle_l2183_218380

-- Define a regular n-gon
structure RegularNGon (n : ℕ) :=
(center : ℝ × ℝ)
(vertices : Fin n → (ℝ × ℝ))

-- Define the condition to be able to roll and reflect the n-gon over any of its sides
def canReflectSymmetrically (n : ℕ) (g : RegularNGon n) : Prop := sorry

-- Definition of a circle with a given center and radius
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the problem for determining if reflection can bring the center of n-gon inside any circle
def canCenterBeInsideCircle (n : ℕ) (g : RegularNGon n) (c : Circle) : Prop :=
  ∃ (f : ℝ × ℝ → ℝ × ℝ), -- Some function representing the reflections
    canReflectSymmetrically n g ∧ f g.center = c.center

-- State the main theorem determining for which n-gons the assertion is true
theorem regular_n_gon_center_inside_circle (n : ℕ) 
  (h : n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6) : 
  ∀ (g : RegularNGon n) (c : Circle), canCenterBeInsideCircle n g c :=
sorry

end regular_n_gon_center_inside_circle_l2183_218380


namespace average_of_two_numbers_l2183_218312

theorem average_of_two_numbers (A B C : ℝ) (h1 : (A + B + C)/3 = 48) (h2 : C = 32) : (A + B)/2 = 56 := by
  sorry

end average_of_two_numbers_l2183_218312


namespace parallelogram_side_length_l2183_218389

theorem parallelogram_side_length (a b : ℕ) (h1 : 2 * (a + b) = 16) (h2 : a = 5) : b = 3 :=
by
  sorry

end parallelogram_side_length_l2183_218389


namespace number_of_questions_in_test_l2183_218396

-- Definitions based on the conditions:
def marks_per_question : ℕ := 2
def jose_wrong_questions : ℕ := 5  -- number of questions Jose got wrong
def total_combined_score : ℕ := 210  -- total score of Meghan, Jose, and Alisson combined

-- Let A be Alisson's score
variables (A Jose Meghan : ℕ)

-- Conditions
axiom joe_more_than_alisson : Jose = A + 40
axiom megh_less_than_jose : Meghan = Jose - 20
axiom combined_scores : A + Jose + Meghan = total_combined_score

-- Function to compute the total possible score for Jose without wrong answers:
noncomputable def jose_improvement_score : ℕ := Jose + (jose_wrong_questions * marks_per_question)

-- Proof problem statement
theorem number_of_questions_in_test :
  (jose_improvement_score Jose) / marks_per_question = 50 :=
by
  -- Sorry is used here to indicate that the proof is omitted.
  sorry

end number_of_questions_in_test_l2183_218396


namespace probability_at_least_one_blown_l2183_218339

theorem probability_at_least_one_blown (P_A P_B P_AB : ℝ)  
  (hP_A : P_A = 0.085) 
  (hP_B : P_B = 0.074) 
  (hP_AB : P_AB = 0.063) : 
  P_A + P_B - P_AB = 0.096 :=
by
  sorry

end probability_at_least_one_blown_l2183_218339


namespace roots_equation_value_l2183_218318

theorem roots_equation_value (α β : ℝ) (h1 : α^2 - α - 1 = 0) (h2 : β^2 - β - 1 = 0) (h3 : α + β = 1) :
    α^4 + 3 * β = 5 := by
sorry

end roots_equation_value_l2183_218318


namespace smallest_lcm_l2183_218321

/-- If k and l are positive 4-digit integers such that gcd(k, l) = 5, 
the smallest value for lcm(k, l) is 201000. -/
theorem smallest_lcm (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h₅ : Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l2183_218321
