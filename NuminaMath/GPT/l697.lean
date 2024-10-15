import Mathlib

namespace NUMINAMATH_GPT_unique_real_root_eq_l697_69710

theorem unique_real_root_eq (x : ℝ) : (∃! x, x = Real.sin x + 1993) :=
sorry

end NUMINAMATH_GPT_unique_real_root_eq_l697_69710


namespace NUMINAMATH_GPT_Mary_chewing_gums_count_l697_69764

variable (Mary_gums Sam_gums Sue_gums : ℕ)

-- Define the given conditions
axiom Sam_chewing_gums : Sam_gums = 10
axiom Sue_chewing_gums : Sue_gums = 15
axiom Total_chewing_gums : Mary_gums + Sam_gums + Sue_gums = 30

theorem Mary_chewing_gums_count : Mary_gums = 5 := by
  sorry

end NUMINAMATH_GPT_Mary_chewing_gums_count_l697_69764


namespace NUMINAMATH_GPT_find_other_number_l697_69781

theorem find_other_number (w : ℕ) (x : ℕ) 
    (h1 : w = 468)
    (h2 : x * w = 2^4 * 3^3 * 13^3) 
    : x = 2028 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l697_69781


namespace NUMINAMATH_GPT_smallest_AAB_value_exists_l697_69786

def is_consecutive_digits (A B : ℕ) : Prop :=
  (B = A + 1 ∨ A = B + 1) ∧ 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9

def two_digit_to_int (A B : ℕ) : ℕ :=
  10 * A + B

def three_digit_to_int (A B : ℕ) : ℕ :=
  110 * A + B

theorem smallest_AAB_value_exists :
  ∃ (A B: ℕ), is_consecutive_digits A B ∧ two_digit_to_int A B = (1 / 7 : ℝ) * ↑(three_digit_to_int A B) ∧ three_digit_to_int A B = 889 :=
sorry

end NUMINAMATH_GPT_smallest_AAB_value_exists_l697_69786


namespace NUMINAMATH_GPT_Lance_must_read_today_l697_69731

def total_pages : ℕ := 100
def pages_read_yesterday : ℕ := 35
def pages_read_tomorrow : ℕ := 27

noncomputable def pages_read_today : ℕ :=
  pages_read_yesterday - 5

noncomputable def pages_left_today : ℕ :=
  total_pages - (pages_read_yesterday + pages_read_today + pages_read_tomorrow)

theorem Lance_must_read_today :
  pages_read_today + pages_left_today = 38 :=
by 
  rw [pages_read_today, pages_left_today, pages_read_yesterday, pages_read_tomorrow, total_pages]
  simp
  sorry

end NUMINAMATH_GPT_Lance_must_read_today_l697_69731


namespace NUMINAMATH_GPT_domain_ln_l697_69798

def domain_of_ln (x : ℝ) : Prop := x^2 - x > 0

theorem domain_ln (x : ℝ) :
  domain_of_ln x ↔ (x < 0 ∨ x > 1) :=
by sorry

end NUMINAMATH_GPT_domain_ln_l697_69798


namespace NUMINAMATH_GPT_remainder_div_modulo_l697_69721

theorem remainder_div_modulo (N : ℕ) (h1 : N % 19 = 7) : N % 20 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_modulo_l697_69721


namespace NUMINAMATH_GPT_salad_dressing_oil_percentage_l697_69774

theorem salad_dressing_oil_percentage 
  (vinegar_P : ℝ) (vinegar_Q : ℝ) (oil_Q : ℝ)
  (new_vinegar : ℝ) (proportion_P : ℝ) :
  vinegar_P = 0.30 ∧ vinegar_Q = 0.10 ∧ oil_Q = 0.90 ∧ new_vinegar = 0.12 ∧ proportion_P = 0.10 →
  (1 - vinegar_P) = 0.70 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_salad_dressing_oil_percentage_l697_69774


namespace NUMINAMATH_GPT_value_of_a_l697_69799

theorem value_of_a (a : ℕ) (A_a B_a : ℕ)
  (h1 : A_a = 10)
  (h2 : B_a = 11)
  (h3 : 2 * a^2 + 10 * a + 3 + 5 * a^2 + 7 * a + 8 = 8 * a^2 + 4 * a + 11) :
  a = 13 :=
sorry

end NUMINAMATH_GPT_value_of_a_l697_69799


namespace NUMINAMATH_GPT_fraction_power_l697_69714

variables (a b c : ℝ)

theorem fraction_power :
  ( ( -2 * a^2 * b ) / (3 * c) )^2 = ( 4 * a^4 * b^2 ) / ( 9 * c^2 ) := 
by sorry

end NUMINAMATH_GPT_fraction_power_l697_69714


namespace NUMINAMATH_GPT_min_value_xyz_l697_69783

-- Definition of the problem
theorem min_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 108):
  x^2 + 9 * x * y + 9 * y^2 + 3 * z^2 ≥ 324 :=
sorry

end NUMINAMATH_GPT_min_value_xyz_l697_69783


namespace NUMINAMATH_GPT_area_of_rectangular_field_l697_69769

theorem area_of_rectangular_field (length width : ℝ) (h_length: length = 5.9) (h_width: width = 3) : 
  length * width = 17.7 := 
by
  sorry

end NUMINAMATH_GPT_area_of_rectangular_field_l697_69769


namespace NUMINAMATH_GPT_jacqueline_erasers_l697_69779

def num_boxes : ℕ := 4
def erasers_per_box : ℕ := 10
def total_erasers : ℕ := num_boxes * erasers_per_box

theorem jacqueline_erasers : total_erasers = 40 := by
  sorry

end NUMINAMATH_GPT_jacqueline_erasers_l697_69779


namespace NUMINAMATH_GPT_problem_statement_l697_69772

def setS : Set (ℝ × ℝ) := {p | p.1 * p.2 > 0}
def setT : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 > 0}

theorem problem_statement : setS ∪ setT = setS ∧ setS ∩ setT = setT :=
by
  -- To be proved
  sorry

end NUMINAMATH_GPT_problem_statement_l697_69772


namespace NUMINAMATH_GPT_remainder_form_l697_69744

open Polynomial Int

-- Define the conditions
variable (f : Polynomial ℤ)
variable (h1 : ∀ n : ℤ, 3 ∣ eval n f)

-- Define the proof problem statement
theorem remainder_form (h1 : ∀ n : ℤ, 3 ∣ eval n f) :
  ∃ (M r : Polynomial ℤ), f = (X^3 - X) * M + C 3 * r :=
sorry

end NUMINAMATH_GPT_remainder_form_l697_69744


namespace NUMINAMATH_GPT_john_money_l697_69738

theorem john_money (cost_given : ℝ) : cost_given = 14 :=
by
  have gift_cost := 28
  have half_cost := gift_cost / 2
  exact sorry

end NUMINAMATH_GPT_john_money_l697_69738


namespace NUMINAMATH_GPT_common_ratio_of_geo_seq_l697_69754

variable {a : ℕ → ℝ} (q : ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem common_ratio_of_geo_seq :
  (∀ n, 0 < a n) →
  geometric_sequence a q →
  a 6 = a 5 + 2 * a 4 →
  q = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_common_ratio_of_geo_seq_l697_69754


namespace NUMINAMATH_GPT_total_outfits_l697_69745

-- Define the quantities of each item.
def red_shirts : ℕ := 7
def green_shirts : ℕ := 8
def pants : ℕ := 10
def blue_hats : ℕ := 10
def red_hats : ℕ := 10
def scarves : ℕ := 5

-- The total number of outfits without having the same color of shirts and hats.
theorem total_outfits : 
  (red_shirts * pants * blue_hats * scarves) + (green_shirts * pants * red_hats * scarves) = 7500 := 
by sorry

end NUMINAMATH_GPT_total_outfits_l697_69745


namespace NUMINAMATH_GPT_number_of_pairs_of_shoes_l697_69700

/-- A box contains some pairs of shoes with a total of 10 shoes.
    If two shoes are selected at random, the probability that they are matching shoes is 1/9.
    Prove that the number of pairs of shoes in the box is 5. -/
theorem number_of_pairs_of_shoes (n : ℕ) (h1 : 2 * n = 10) 
  (h2 : ((n * (n - 1)) / (10 * (10 - 1))) = 1 / 9) : n = 5 := 
sorry

end NUMINAMATH_GPT_number_of_pairs_of_shoes_l697_69700


namespace NUMINAMATH_GPT_arithmetic_seq_first_term_l697_69737

theorem arithmetic_seq_first_term (S : ℕ → ℚ) (a : ℚ) (n : ℕ) (h1 : ∀ n, S n = (n * (2 * a + (n - 1) * 5)) / 2)
  (h2 : ∀ n, S (4 * n) / S n = 16) : a = 5 / 2 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_first_term_l697_69737


namespace NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l697_69750

theorem max_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (a1 : a 1 = 29) 
  (S10_eq_S20 : S 10 = S 20) :
  (∃ n, ∀ m, S n ≥ S m) ∧ ∃ n, (S n = S 15) :=
sorry

end NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l697_69750


namespace NUMINAMATH_GPT_factorization_a_minus_b_l697_69751

theorem factorization_a_minus_b (a b : ℤ) (y : ℝ) 
  (h1 : 3 * y ^ 2 - 7 * y - 6 = (3 * y + a) * (y + b)) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) : 
  a - b = 5 :=
sorry

end NUMINAMATH_GPT_factorization_a_minus_b_l697_69751


namespace NUMINAMATH_GPT_simultaneous_messengers_l697_69762

theorem simultaneous_messengers (m n : ℕ) (h : m * n = 2010) : 
  m ≠ n → ((m, n) = (1, 2010) ∨ (m, n) = (2, 1005) ∨ (m, n) = (3, 670) ∨ 
          (m, n) = (5, 402) ∨ (m, n) = (6, 335) ∨ (m, n) = (10, 201) ∨ 
          (m, n) = (15, 134) ∨ (m, n) = (30, 67)) :=
sorry

end NUMINAMATH_GPT_simultaneous_messengers_l697_69762


namespace NUMINAMATH_GPT_sphere_volume_of_hexagonal_prism_l697_69782

noncomputable def volume_of_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem sphere_volume_of_hexagonal_prism
  (a h : ℝ)
  (volume : ℝ)
  (base_perimeter : ℝ)
  (vertices_on_sphere : ∀ (x y : ℝ) (hx : x^2 + y^2 = a^2) (hy : y = h / 2), x^2 + y^2 = 1) :
  volume = 9 / 8 ∧ base_perimeter = 3 →
  volume_of_sphere 1 = 4 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_sphere_volume_of_hexagonal_prism_l697_69782


namespace NUMINAMATH_GPT_depth_of_ship_l697_69707

-- Condition definitions
def rate : ℝ := 80  -- feet per minute
def time : ℝ := 50  -- minutes

-- Problem Statement
theorem depth_of_ship : rate * time = 4000 :=
by
  sorry

end NUMINAMATH_GPT_depth_of_ship_l697_69707


namespace NUMINAMATH_GPT_tens_digit_36_pow_12_l697_69797

theorem tens_digit_36_pow_12 : ((36 ^ 12) % 100) / 10 % 10 = 1 := 
by 
sorry

end NUMINAMATH_GPT_tens_digit_36_pow_12_l697_69797


namespace NUMINAMATH_GPT_min_tangent_length_l697_69777

-- Definitions and conditions as given in the problem context
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 3 = 0

def symmetry_line (a b x y : ℝ) : Prop :=
  2 * a * x + b * y + 6 = 0

-- Proving the minimum length of the tangent line
theorem min_tangent_length (a b : ℝ) (h_sym : ∀ x y, circle_equation x y → symmetry_line a b x y) :
  ∃ l, l = 4 :=
sorry

end NUMINAMATH_GPT_min_tangent_length_l697_69777


namespace NUMINAMATH_GPT_find_two_digit_number_with_cubic_ending_in_9_l697_69761

theorem find_two_digit_number_with_cubic_ending_in_9:
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n^3 % 10 = 9 ∧ n = 19 := 
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_with_cubic_ending_in_9_l697_69761


namespace NUMINAMATH_GPT_tires_usage_l697_69712

theorem tires_usage :
  let total_miles := 50000
  let first_part_miles := 40000
  let second_part_miles := 10000
  let num_tires_first_part := 5
  let num_tires_total := 7
  let total_tire_miles_first := first_part_miles * num_tires_first_part
  let total_tire_miles_second := second_part_miles * num_tires_total
  let combined_tire_miles := total_tire_miles_first + total_tire_miles_second
  let miles_per_tire := combined_tire_miles / num_tires_total
  miles_per_tire = 38571 := 
by
  sorry

end NUMINAMATH_GPT_tires_usage_l697_69712


namespace NUMINAMATH_GPT_find_a_of_inequality_solution_l697_69757

theorem find_a_of_inequality_solution (a : ℝ) :
  (∀ x : ℝ, -3 < ax - 2 ∧ ax - 2 < 3 ↔ -5/3 < x ∧ x < 1/3) →
  a = -3 := by
  sorry

end NUMINAMATH_GPT_find_a_of_inequality_solution_l697_69757


namespace NUMINAMATH_GPT_lost_marble_count_l697_69708

def initial_marble_count : ℕ := 16
def remaining_marble_count : ℕ := 9

theorem lost_marble_count : initial_marble_count - remaining_marble_count = 7 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_lost_marble_count_l697_69708


namespace NUMINAMATH_GPT_vowel_initial_probability_is_correct_l697_69784

-- Given conditions as definitions
def total_students : ℕ := 34
def vowels : List Char := ['A', 'E', 'I', 'O', 'U', 'Y']
def vowels_count_per_vowel : ℕ := 2
def total_vowels_count := vowels.length * vowels_count_per_vowel

-- The probabilistic statement we want to prove
def vowel_probability : ℚ := total_vowels_count / total_students

-- The final statement to prove
theorem vowel_initial_probability_is_correct :
  vowel_probability = 6 / 17 :=
by
  unfold vowel_probability total_vowels_count
  -- Simplification to verify our statement.
  sorry

end NUMINAMATH_GPT_vowel_initial_probability_is_correct_l697_69784


namespace NUMINAMATH_GPT_find_digit_B_l697_69788

theorem find_digit_B (A B : ℕ) (h1 : A3B = 100 * A + 30 + B)
  (h2 : 0 ≤ A ∧ A ≤ 9)
  (h3 : 0 ≤ B ∧ B ≤ 9)
  (h4 : A3B - 41 = 591) : 
  B = 2 := 
by sorry

end NUMINAMATH_GPT_find_digit_B_l697_69788


namespace NUMINAMATH_GPT_relationship_between_M_and_P_l697_69704

def M := {y : ℝ | ∃ x : ℝ, y = x^2 - 4}
def P := {x : ℝ | 2 ≤ x ∧ x ≤ 4}

theorem relationship_between_M_and_P : ∀ y ∈ {y : ℝ | ∃ x ∈ P, y = x^2 - 4}, y ∈ M :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_M_and_P_l697_69704


namespace NUMINAMATH_GPT_f_plus_2012_odd_l697_69720

def f : ℝ → ℝ → ℝ := sorry

lemma f_property (α β : ℝ) : f α β = 2012 := sorry

theorem f_plus_2012_odd : ∀ x : ℝ, f (-x) + 2012 = -(f x + 2012) :=
by
  sorry

end NUMINAMATH_GPT_f_plus_2012_odd_l697_69720


namespace NUMINAMATH_GPT_number_of_lines_intersecting_circle_l697_69758

theorem number_of_lines_intersecting_circle : 
  ∃ l : ℕ, 
  (∀ a b x y : ℤ, (x^2 + y^2 = 50 ∧ (x / a + y / b = 1))) → 
  (∃ n : ℕ, n = 60) :=
sorry

end NUMINAMATH_GPT_number_of_lines_intersecting_circle_l697_69758


namespace NUMINAMATH_GPT_opposite_of_neg_two_l697_69770

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_two_l697_69770


namespace NUMINAMATH_GPT_cookie_radius_l697_69716

theorem cookie_radius (x y : ℝ) (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 13 := 
sorry

end NUMINAMATH_GPT_cookie_radius_l697_69716


namespace NUMINAMATH_GPT_isosceles_trapezoid_ratio_ab_cd_l697_69791

theorem isosceles_trapezoid_ratio_ab_cd (AB CD : ℝ) (P : ℝ → ℝ → Prop)
  (area1 area2 area3 area4 : ℝ)
  (h1 : AB > CD)
  (h2 : area1 = 5)
  (h3 : area2 = 7)
  (h4 : area3 = 3)
  (h5 : area4 = 9) :
  AB / CD = 1 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_isosceles_trapezoid_ratio_ab_cd_l697_69791


namespace NUMINAMATH_GPT_matrix_B_pow_66_l697_69701

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0], 
    ![-1, 0, 0], 
    ![0, 0, 1]]

theorem matrix_B_pow_66 : B^66 = ![![-1, 0, 0], ![0, -1, 0], ![0, 0, 1]] := by
  sorry

end NUMINAMATH_GPT_matrix_B_pow_66_l697_69701


namespace NUMINAMATH_GPT_tim_runs_more_than_sarah_l697_69748

-- Definitions based on the conditions
def street_width : ℕ := 25
def side_length : ℕ := 450

-- Perimeters of the paths
def sarah_perimeter : ℕ := 4 * side_length
def tim_perimeter : ℕ := 4 * (side_length + 2 * street_width)

-- The theorem to prove
theorem tim_runs_more_than_sarah : tim_perimeter - sarah_perimeter = 200 := by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_tim_runs_more_than_sarah_l697_69748


namespace NUMINAMATH_GPT_sum_A_J_l697_69736

variable (A B C D E F G H I J : ℕ)

-- Conditions
axiom h1 : C = 7
axiom h2 : A + B + C = 40
axiom h3 : B + C + D = 40
axiom h4 : C + D + E = 40
axiom h5 : D + E + F = 40
axiom h6 : E + F + G = 40
axiom h7 : F + G + H = 40
axiom h8 : G + H + I = 40
axiom h9 : H + I + J = 40

-- Proof statement
theorem sum_A_J : A + J = 33 :=
by
  sorry

end NUMINAMATH_GPT_sum_A_J_l697_69736


namespace NUMINAMATH_GPT_number_of_packs_l697_69792

theorem number_of_packs (total_towels towels_per_pack : ℕ) (h1 : total_towels = 27) (h2 : towels_per_pack = 3) :
  total_towels / towels_per_pack = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_packs_l697_69792


namespace NUMINAMATH_GPT_max_marks_l697_69749

theorem max_marks (M: ℝ) (h1: 0.95 * M = 285):
  M = 300 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l697_69749


namespace NUMINAMATH_GPT_sum_angles_star_l697_69766

theorem sum_angles_star (β : ℝ) (h : β = 90) : 
  8 * β = 720 :=
by
  sorry

end NUMINAMATH_GPT_sum_angles_star_l697_69766


namespace NUMINAMATH_GPT_lunch_cost_before_tip_l697_69733

theorem lunch_cost_before_tip (C : ℝ) (h : C + 0.2 * C = 60.6) : C = 50.5 :=
sorry

end NUMINAMATH_GPT_lunch_cost_before_tip_l697_69733


namespace NUMINAMATH_GPT_certain_number_is_10000_l697_69765

theorem certain_number_is_10000 (n : ℕ) (h1 : n - 999 = 9001) : n = 10000 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_10000_l697_69765


namespace NUMINAMATH_GPT_sale_in_first_month_is_5420_l697_69789

-- Definitions of the sales in months 2 to 6
def sale_month2 : ℕ := 5660
def sale_month3 : ℕ := 6200
def sale_month4 : ℕ := 6350
def sale_month5 : ℕ := 6500
def sale_month6 : ℕ := 6470

-- Definition of the average sale goal
def average_sale_goal : ℕ := 6100

-- Calculating the total needed sales to achieve the average sale goal
def total_required_sales := 6 * average_sale_goal

-- Known sales for months 2 to 6
def known_sales := sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6

-- Definition of the sale in the first month
def sale_month1 := total_required_sales - known_sales

-- The proof statement that the sale in the first month is 5420
theorem sale_in_first_month_is_5420 : sale_month1 = 5420 := by
  sorry

end NUMINAMATH_GPT_sale_in_first_month_is_5420_l697_69789


namespace NUMINAMATH_GPT_magnitude_of_vec_sum_l697_69705

noncomputable def vec_a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def vec_b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))
noncomputable def vec_sum : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_vec_sum : magnitude vec_sum = Real.sqrt 7 := 
by 
  sorry

end NUMINAMATH_GPT_magnitude_of_vec_sum_l697_69705


namespace NUMINAMATH_GPT_total_leftover_tarts_l697_69709

def cherry_tarts := 0.08
def blueberry_tarts := 0.75
def peach_tarts := 0.08

theorem total_leftover_tarts : cherry_tarts + blueberry_tarts + peach_tarts = 0.91 := by
  sorry

end NUMINAMATH_GPT_total_leftover_tarts_l697_69709


namespace NUMINAMATH_GPT_proposition_not_true_at_3_l697_69730

variable (P : ℕ → Prop)

theorem proposition_not_true_at_3
  (h1 : ∀ k : ℕ, P k → P (k + 1))
  (h2 : ¬ P 4) :
  ¬ P 3 :=
sorry

end NUMINAMATH_GPT_proposition_not_true_at_3_l697_69730


namespace NUMINAMATH_GPT_average_difference_l697_69732

def daily_differences : List ℤ := [2, -1, 3, 1, -2, 2, 1]

theorem average_difference :
  (daily_differences.sum : ℚ) / daily_differences.length = 0.857 :=
by
  sorry

end NUMINAMATH_GPT_average_difference_l697_69732


namespace NUMINAMATH_GPT_triangle_base_length_l697_69768

theorem triangle_base_length :
  ∀ (base height area : ℕ), height = 4 → area = 16 → area = (base * height) / 2 → base = 8 :=
by
  intros base height area h_height h_area h_formula
  sorry

end NUMINAMATH_GPT_triangle_base_length_l697_69768


namespace NUMINAMATH_GPT_one_gt_one_others_lt_one_l697_69755

theorem one_gt_one_others_lt_one 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_prod : a * b * c = 1)
  (h_ineq : a + b + c > (1 / a) + (1 / b) + (1 / c)) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ (b > 1 ∧ a < 1 ∧ c < 1) ∨ (c > 1 ∧ a < 1 ∧ b < 1) :=
sorry

end NUMINAMATH_GPT_one_gt_one_others_lt_one_l697_69755


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l697_69727

noncomputable def f (a x : ℝ) := x^2 + 2 * a * x - 2

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, x ≤ -2 → deriv (f a) x ≤ 0) ↔ a = 2 :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l697_69727


namespace NUMINAMATH_GPT_flight_up_speed_l697_69702

variable (v : ℝ) -- speed on the flight up
variable (d : ℝ) -- distance to mother's place

/--
Given:
1. The speed on the way home was 72 mph.
2. The average speed for the trip was 91 mph.

Prove:
The speed on the flight up was 123.62 mph.
-/
theorem flight_up_speed
  (h1 : 72 > 0)
  (h2 : 91 > 0)
  (avg_speed_def : 91 = (2 * d) / ((d / v) + (d / 72))) :
  v = 123.62 :=
by
  sorry

end NUMINAMATH_GPT_flight_up_speed_l697_69702


namespace NUMINAMATH_GPT_standard_circle_equation_passing_through_P_l697_69760

-- Define the condition that a point P is a solution to the system of equations derived from the line
def PointPCondition (x y : ℝ) : Prop :=
  (2 * x + 3 * y - 1 = 0) ∧ (3 * x - 2 * y + 5 = 0)

-- Define the center and radius of the given circle C
def CenterCircleC : ℝ × ℝ := (2, -3)
def RadiusCircleC : ℝ := 4  -- Since the radius squared is 16

-- Define the condition that a point is on a circle with a given center and radius
def OnCircle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.fst)^2 + (y + center.snd)^2 = radius^2

-- State the problem
theorem standard_circle_equation_passing_through_P :
  ∃ (x y : ℝ), PointPCondition x y ∧ OnCircle CenterCircleC 5 x y :=
sorry

end NUMINAMATH_GPT_standard_circle_equation_passing_through_P_l697_69760


namespace NUMINAMATH_GPT_negation_of_p_l697_69795

-- Given conditions
def p : Prop := ∃ x : ℝ, x^2 + 3 * x = 4

-- The proof problem to be solved 
theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x^2 + 3 * x ≠ 4 := by
  sorry

end NUMINAMATH_GPT_negation_of_p_l697_69795


namespace NUMINAMATH_GPT_initial_mat_weavers_eq_4_l697_69711

theorem initial_mat_weavers_eq_4 :
  ∃ x : ℕ, (x * 4 = 4) ∧ (14 * 14 = 49) ∧ (x = 4) :=
sorry

end NUMINAMATH_GPT_initial_mat_weavers_eq_4_l697_69711


namespace NUMINAMATH_GPT_cosine_lt_sine_neg_four_l697_69776

theorem cosine_lt_sine_neg_four : ∀ (m n : ℝ), m = Real.cos (-4) → n = Real.sin (-4) → m < n :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end NUMINAMATH_GPT_cosine_lt_sine_neg_four_l697_69776


namespace NUMINAMATH_GPT_equal_distribution_l697_69728

namespace MoneyDistribution

def Ann_initial := 777
def Bill_initial := 1111
def Charlie_initial := 1555
def target_amount := 1148
def Bill_to_Ann := 371
def Charlie_to_Bill := 408

theorem equal_distribution :
  (Bill_initial - Bill_to_Ann + Charlie_to_Bill = target_amount) ∧
  (Ann_initial + Bill_to_Ann = target_amount) ∧
  (Charlie_initial - Charlie_to_Bill = target_amount) :=
by
  sorry

end MoneyDistribution

end NUMINAMATH_GPT_equal_distribution_l697_69728


namespace NUMINAMATH_GPT_tennis_balls_ordered_l697_69746

def original_white_balls : ℕ := sorry
def original_yellow_balls_with_error : ℕ := sorry

theorem tennis_balls_ordered 
  (W Y : ℕ)
  (h1 : W = Y)
  (h2 : Y + 70 = original_yellow_balls_with_error)
  (h3 : W = 8 / 13 * (Y + 70)):
  W + Y = 224 := sorry

end NUMINAMATH_GPT_tennis_balls_ordered_l697_69746


namespace NUMINAMATH_GPT_find_lightest_bead_l697_69741

theorem find_lightest_bead (n : ℕ) (h : 0 < n) (H : ∀ b1 b2 b3 : ℕ, b1 + b2 + b3 = n → b1 > 0 ∧ b2 > 0 ∧ b3 > 0 → b1 ≤ 3 ∧ b2 ≤ 9 ∧ b3 ≤ 27) : n = 27 :=
sorry

end NUMINAMATH_GPT_find_lightest_bead_l697_69741


namespace NUMINAMATH_GPT_inequality_cannot_hold_l697_69742

noncomputable def f (a b c x : ℝ) := a * x ^ 2 + b * x + c

theorem inequality_cannot_hold
  (a b c : ℝ)
  (h_symm : ∀ x, f a b c x = f a b c (2 - x)) :
  ¬ (f a b c (1 - a) < f a b c (1 - 2 * a) ∧ f a b c (1 - 2 * a) < f a b c 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_cannot_hold_l697_69742


namespace NUMINAMATH_GPT_daily_wage_of_c_l697_69767

-- Define the conditions
variables (a b c : ℝ)
variables (h_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5)
variables (h_days : 6 * a + 9 * b + 4 * c = 1702)

-- Define the proof problem; to prove c = 115
theorem daily_wage_of_c (h_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5) (h_days : 6 * a + 9 * b + 4 * c = 1702) : 
  c = 115 :=
sorry

end NUMINAMATH_GPT_daily_wage_of_c_l697_69767


namespace NUMINAMATH_GPT_gcd_lcm_product_l697_69778

theorem gcd_lcm_product (a b : ℕ) (ha : a = 100) (hb : b = 120) :
  Nat.gcd a b * Nat.lcm a b = 12000 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l697_69778


namespace NUMINAMATH_GPT_quadratic_root_unique_l697_69759

theorem quadratic_root_unique 
  (a b c : ℝ)
  (hf1 : b^2 - 4 * a * c = 0)
  (hf2 : (b - 30 * a)^2 - 4 * a * (17 * a - 7 * b + c) = 0)
  (ha_pos : a ≠ 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = -11 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_root_unique_l697_69759


namespace NUMINAMATH_GPT_smallest_base_for_100_l697_69747

theorem smallest_base_for_100 :
  ∃ b : ℕ, b^2 ≤ 100 ∧ 100 < b^3 ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
sorry

end NUMINAMATH_GPT_smallest_base_for_100_l697_69747


namespace NUMINAMATH_GPT_find_m_interval_l697_69713

def seq (x : ℕ → ℚ) : Prop :=
  (x 0 = 7) ∧ (∀ n : ℕ, x (n + 1) = (x n ^ 2 + 8 * x n + 9) / (x n + 7))

def m_spec (x : ℕ → ℚ) (m : ℕ) : Prop :=
  (x m ≤ 5 + 1 / 2^15)

theorem find_m_interval :
  ∃ (x : ℕ → ℚ) (m : ℕ), seq x ∧ m_spec x m ∧ 81 ≤ m ∧ m ≤ 242 :=
sorry

end NUMINAMATH_GPT_find_m_interval_l697_69713


namespace NUMINAMATH_GPT_erin_days_to_receive_30_l697_69780

theorem erin_days_to_receive_30 (x : ℕ) (h : 3 * x = 30) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_erin_days_to_receive_30_l697_69780


namespace NUMINAMATH_GPT_chocolates_remaining_l697_69775

theorem chocolates_remaining 
  (total_chocolates : ℕ)
  (ate_day1 : ℕ) (ate_day2 : ℕ) (ate_day3 : ℕ) (ate_day4 : ℕ) (ate_day5 : ℕ) (remaining_chocolates : ℕ) 
  (h_total : total_chocolates = 48)
  (h_day1 : ate_day1 = 6) 
  (h_day2 : ate_day2 = 2 * ate_day1 + 2) 
  (h_day3 : ate_day3 = ate_day1 - 3) 
  (h_day4 : ate_day4 = 2 * ate_day3 + 1) 
  (h_day5 : ate_day5 = ate_day2 / 2) 
  (h_rem : remaining_chocolates = total_chocolates - (ate_day1 + ate_day2 + ate_day3 + ate_day4 + ate_day5)) :
  remaining_chocolates = 14 :=
sorry

end NUMINAMATH_GPT_chocolates_remaining_l697_69775


namespace NUMINAMATH_GPT_principal_amount_l697_69715

theorem principal_amount (SI P R T : ℝ) 
  (h1 : R = 12) (h2 : T = 3) (h3 : SI = 3600) : 
  SI = P * R * T / 100 → P = 10000 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_principal_amount_l697_69715


namespace NUMINAMATH_GPT_john_pool_cleanings_per_month_l697_69794

noncomputable def tip_percent : ℝ := 0.10
noncomputable def cost_per_cleaning : ℝ := 150
noncomputable def total_cost_per_cleaning : ℝ := cost_per_cleaning + (tip_percent * cost_per_cleaning)
noncomputable def chemical_cost_bi_monthly : ℝ := 200
noncomputable def monthly_chemical_cost : ℝ := 2 * chemical_cost_bi_monthly
noncomputable def total_monthly_pool_cost : ℝ := 2050
noncomputable def total_cleaning_cost : ℝ := total_monthly_pool_cost - monthly_chemical_cost

theorem john_pool_cleanings_per_month : total_cleaning_cost / total_cost_per_cleaning = 10 := by
  sorry

end NUMINAMATH_GPT_john_pool_cleanings_per_month_l697_69794


namespace NUMINAMATH_GPT_ratio_sheep_horses_eq_six_seven_l697_69706

noncomputable def total_food_per_day : ℕ := 12880
noncomputable def food_per_horse_per_day : ℕ := 230
noncomputable def num_sheep : ℕ := 48
noncomputable def num_horses : ℕ := total_food_per_day / food_per_horse_per_day
noncomputable def ratio_sheep_to_horses := num_sheep / num_horses

theorem ratio_sheep_horses_eq_six_seven :
  ratio_sheep_to_horses = 6 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_sheep_horses_eq_six_seven_l697_69706


namespace NUMINAMATH_GPT_complement_of_A_l697_69735

def U : Set ℤ := {-1, 2, 4}
def A : Set ℤ := {-1, 4}

theorem complement_of_A : U \ A = {2} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_l697_69735


namespace NUMINAMATH_GPT_graph_representation_l697_69785

theorem graph_representation {x y : ℝ} (h : x^2 * (x - y - 2) = y^2 * (x - y - 2)) :
  ( ∃ a : ℝ, ∀ (x : ℝ), y = a * x ) ∨ 
  ( ∃ b : ℝ, ∀ (x : ℝ), y = b * x ) ∨ 
  ( ∃ c : ℝ, ∀ (x : ℝ), y = x - 2 ) ∧ 
  (¬ ∃ d : ℝ, ∀ (x : ℝ), y = d * x ∧ y = d * x - 2) :=
sorry

end NUMINAMATH_GPT_graph_representation_l697_69785


namespace NUMINAMATH_GPT_prime_divides_diff_of_cubes_l697_69787

theorem prime_divides_diff_of_cubes (a b c : ℕ) [Fact (Nat.Prime a)] [Fact (Nat.Prime b)]
  (h1 : c ∣ (a + b)) (h2 : c ∣ (a * b)) : c ∣ (a^3 - b^3) :=
by
  sorry

end NUMINAMATH_GPT_prime_divides_diff_of_cubes_l697_69787


namespace NUMINAMATH_GPT_grooming_time_5_dogs_3_cats_l697_69796

theorem grooming_time_5_dogs_3_cats :
  (2.5 * 5 + 0.5 * 3) * 60 = 840 :=
by
  -- Prove that grooming 5 dogs and 3 cats takes 840 minutes.
  sorry

end NUMINAMATH_GPT_grooming_time_5_dogs_3_cats_l697_69796


namespace NUMINAMATH_GPT_difference_area_octagon_shaded_l697_69740

-- Definitions based on the given conditions
def radius : ℝ := 10
def pi_value : ℝ := 3.14

-- Lean statement for the given proof problem
theorem difference_area_octagon_shaded :
  ∃ S_octagon S_shaded, 
    10^2 * pi_value = 314 ∧
    (20 / 2^0.5)^2 = 200 ∧
    S_octagon = 200 - 114 ∧ -- transposed to reverse engineering step
    S_shaded = 28 ∧ -- needs refinement
    S_octagon - S_shaded = 86 :=
sorry

end NUMINAMATH_GPT_difference_area_octagon_shaded_l697_69740


namespace NUMINAMATH_GPT_value_of_expression_l697_69718

theorem value_of_expression : 50^4 + 4 * 50^3 + 6 * 50^2 + 4 * 50 + 1 = 6765201 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l697_69718


namespace NUMINAMATH_GPT_min_value_l697_69703

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : 1 / (a + 3) + 1 / (b + 3) + 1 / (c + 3) = 1 / 4) : 
  22.75 ≤ a + 3 * b + 2 * c := 
sorry

end NUMINAMATH_GPT_min_value_l697_69703


namespace NUMINAMATH_GPT_find_equation_of_line_l_l697_69753

-- Define the conditions
def point_P : ℝ × ℝ := (2, 3)

noncomputable def angle_of_inclination : ℝ := 2 * Real.pi / 3

def intercept_condition (a b : ℝ) : Prop := a + b = 0

-- The proof statement
theorem find_equation_of_line_l :
  ∃ (k : ℝ), k = Real.tan angle_of_inclination ∧
  ∃ (C : ℝ), ∀ (x y : ℝ), (y - 3 = k * (x - 2)) ∧ C = (3 + 2 * (Real.sqrt 3)) ∨ 
             (intercept_condition (x / point_P.1) (y / point_P.2) ∧ C = 1) ∨ 
             -- The standard forms of the line equation
             ((Real.sqrt 3 * x + y - C = 0) ∨ (x - y + 1 = 0)) :=
sorry

end NUMINAMATH_GPT_find_equation_of_line_l_l697_69753


namespace NUMINAMATH_GPT_painting_time_eq_l697_69722

theorem painting_time_eq (t : ℚ) : 
  (1/6 + 1/8 + 1/10) * (t - 2) = 1 := 
sorry

end NUMINAMATH_GPT_painting_time_eq_l697_69722


namespace NUMINAMATH_GPT_initial_pencils_count_l697_69763

theorem initial_pencils_count (pencils_taken : ℕ) (pencils_left : ℕ) (h1 : pencils_taken = 4) (h2 : pencils_left = 75) : 
  pencils_left + pencils_taken = 79 :=
by
  sorry

end NUMINAMATH_GPT_initial_pencils_count_l697_69763


namespace NUMINAMATH_GPT_negation_equivalence_l697_69771

theorem negation_equivalence : (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
  sorry

end NUMINAMATH_GPT_negation_equivalence_l697_69771


namespace NUMINAMATH_GPT_min_value_inequality_l697_69773

noncomputable def min_value (x y z w : ℝ) : ℝ :=
  x^2 + 4 * x * y + 9 * y^2 + 6 * y * z + 8 * z^2 + 3 * x * w + 4 * w^2

theorem min_value_inequality 
  (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_prod : x * y * z * w = 3) : 
  min_value x y z w ≥ 81.25 := 
sorry

end NUMINAMATH_GPT_min_value_inequality_l697_69773


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l697_69726

theorem arithmetic_geometric_sequence :
  ∀ (a : ℕ → ℕ) (b : ℕ → ℕ),
    (a 1 + a 2 = 10) →
    (a 4 - a 3 = 2) →
    (b 2 = a 3) →
    (b 3 = a 7) →
    a 15 = b 4 :=
by
  intros a b h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l697_69726


namespace NUMINAMATH_GPT_exists_smaller_circle_with_at_least_as_many_lattice_points_l697_69752

theorem exists_smaller_circle_with_at_least_as_many_lattice_points
  (R : ℝ) (hR : 0 < R) :
  ∃ R' : ℝ, (R' < R) ∧ (∀ (x y : ℤ), x^2 + y^2 ≤ R^2 → ∃ (x' y' : ℤ), (x')^2 + (y')^2 ≤ (R')^2) := sorry

end NUMINAMATH_GPT_exists_smaller_circle_with_at_least_as_many_lattice_points_l697_69752


namespace NUMINAMATH_GPT_identical_digits_divisible_l697_69725

  theorem identical_digits_divisible (n : ℕ) (hn : n > 0) : 
    ∀ a : ℕ, (10^(3^n - 1) * a / 9) % 3^n = 0 := 
  by
    intros
    sorry
  
end NUMINAMATH_GPT_identical_digits_divisible_l697_69725


namespace NUMINAMATH_GPT_negation_proposition_l697_69756

theorem negation_proposition :
  ¬(∀ x : ℝ, |x - 2| < 3) ↔ ∃ x : ℝ, |x - 2| ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l697_69756


namespace NUMINAMATH_GPT_prime_fraction_sum_l697_69739

theorem prime_fraction_sum (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
    (h : a + b + c + a * b * c = 99) :
    |(1 / a : ℚ) - (1 / b : ℚ)| + |(1 / b : ℚ) - (1 / c : ℚ)| + |(1 / c : ℚ) - (1 / a : ℚ)| = 9 / 11 := 
sorry

end NUMINAMATH_GPT_prime_fraction_sum_l697_69739


namespace NUMINAMATH_GPT_Aaron_initial_erasers_l697_69729

/-- 
  Given:
  - Aaron gives 34 erasers to Doris.
  - Aaron ends with 47 erasers.
  Prove:
  - Aaron started with 81 erasers.
-/ 
theorem Aaron_initial_erasers (gives : ℕ) (ends : ℕ) (start : ℕ) :
  gives = 34 → ends = 47 → start = ends + gives → start = 81 :=
by
  intros h_gives h_ends h_start
  sorry

end NUMINAMATH_GPT_Aaron_initial_erasers_l697_69729


namespace NUMINAMATH_GPT_train_bus_difference_l697_69790

variable (T : ℝ)  -- T is the cost of a train ride

-- conditions
def cond1 := T + 1.50 = 9.85
def cond2 := 1.50 = 1.50

theorem train_bus_difference (h1 : cond1 T) (h2 : cond2) : T - 1.50 = 6.85 := 
sorry

end NUMINAMATH_GPT_train_bus_difference_l697_69790


namespace NUMINAMATH_GPT_train_speed_conversion_l697_69793

def km_per_hour_to_m_per_s (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

theorem train_speed_conversion (speed_kmph : ℕ) (h : speed_kmph = 108) :
  km_per_hour_to_m_per_s speed_kmph = 30 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_train_speed_conversion_l697_69793


namespace NUMINAMATH_GPT_matching_pair_probability_l697_69743

def total_pairs : ℕ := 17

def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 3
def red_pairs : ℕ := 2

def total_shoes : ℕ := 2 * (black_pairs + brown_pairs + gray_pairs + red_pairs)

def prob_match (n_pairs : ℕ) (total_shoes : ℕ) :=
  (2 * n_pairs / total_shoes) * (n_pairs / (total_shoes - 1))

noncomputable def probability_of_matching_pair :=
  (prob_match black_pairs total_shoes) +
  (prob_match brown_pairs total_shoes) +
  (prob_match gray_pairs total_shoes) +
  (prob_match red_pairs total_shoes)

theorem matching_pair_probability :
  probability_of_matching_pair = 93 / 551 :=
sorry

end NUMINAMATH_GPT_matching_pair_probability_l697_69743


namespace NUMINAMATH_GPT_track_circumference_l697_69723

theorem track_circumference (x : ℕ) 
  (A_B_uniform_speeds_opposite : True) 
  (diametrically_opposite_start : True) 
  (same_start_time : True) 
  (first_meeting_B_150_yards : True) 
  (second_meeting_A_90_yards_before_complete_lap : True) : 
  2 * x = 720 :=
by
  sorry

end NUMINAMATH_GPT_track_circumference_l697_69723


namespace NUMINAMATH_GPT_sum_of_xy_l697_69719

theorem sum_of_xy (x y : ℝ) (h1 : x^3 - 6*x^2 + 12*x = 13) (h2 : y^3 + 3*y - 3*y^2 = -4) : x + y = 3 :=
by sorry

end NUMINAMATH_GPT_sum_of_xy_l697_69719


namespace NUMINAMATH_GPT_sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions_l697_69724

theorem sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions :
  ∃ (a b : ℕ), (4 < a ∧ a < b ∧ b < 16) ∧
  (∃ r : ℚ, a = 4 * r ∧ b = 4 * r * r) ∧
  (a + b = 2 * b - a + 16) ∧
  a + b = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions_l697_69724


namespace NUMINAMATH_GPT_flowers_sold_l697_69734

theorem flowers_sold (lilacs roses gardenias : ℕ) 
  (h1 : lilacs = 10)
  (h2 : roses = 3 * lilacs)
  (h3 : gardenias = lilacs / 2) : 
  lilacs + roses + gardenias = 45 :=
by
  sorry

end NUMINAMATH_GPT_flowers_sold_l697_69734


namespace NUMINAMATH_GPT_isosceles_triangle_base_angle_l697_69717

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle : a + b + c = 180)
  (h_iso : a = b ∨ b = c ∨ a = c) (h_interior : a = 50 ∨ b = 50 ∨ c = 50) :
  c = 50 ∨ c = 65 :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angle_l697_69717
