import Mathlib

namespace NUMINAMATH_GPT_insurance_covers_80_percent_l1520_152022

def xray_cost : ℕ := 250
def mri_cost : ℕ := 3 * xray_cost
def total_cost : ℕ := xray_cost + mri_cost
def mike_payment : ℕ := 200
def insurance_coverage : ℕ := total_cost - mike_payment
def insurance_percentage : ℕ := (insurance_coverage * 100) / total_cost

theorem insurance_covers_80_percent : insurance_percentage = 80 := by
  -- Carry out the necessary calculations
  sorry

end NUMINAMATH_GPT_insurance_covers_80_percent_l1520_152022


namespace NUMINAMATH_GPT_sum_mod_7_eq_5_l1520_152052

theorem sum_mod_7_eq_5 : 
  (51730 + 51731 + 51732 + 51733 + 51734 + 51735) % 7 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_sum_mod_7_eq_5_l1520_152052


namespace NUMINAMATH_GPT_discount_percentage_l1520_152058

theorem discount_percentage (wholesale_price retail_price selling_price profit: ℝ) 
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : profit = 0.20 * wholesale_price)
  (h4 : selling_price = wholesale_price + profit):
  (retail_price - selling_price) / retail_price * 100 = 10 :=
by 
  sorry

end NUMINAMATH_GPT_discount_percentage_l1520_152058


namespace NUMINAMATH_GPT_largest_angle_of_quadrilateral_l1520_152002

open Real

theorem largest_angle_of_quadrilateral 
  (PQ QR RS : ℝ)
  (angle_RQP angle_SRQ largest_angle : ℝ)
  (h1: PQ = QR) 
  (h2: QR = RS) 
  (h3: angle_RQP = 60)
  (h4: angle_SRQ = 100)
  (h5: largest_angle = 130) : 
  largest_angle = 130 := by
  sorry

end NUMINAMATH_GPT_largest_angle_of_quadrilateral_l1520_152002


namespace NUMINAMATH_GPT_geometric_sum_S12_l1520_152090

theorem geometric_sum_S12 
  (S : ℕ → ℝ)
  (h_S4 : S 4 = 2) 
  (h_S8 : S 8 = 6) 
  (geom_property : ∀ n, (S (2 * n + 4) - S n) ^ 2 = S n * (S (3 * n + 4) - S (2 * n + 4))) 
  : S 12 = 14 := 
by sorry

end NUMINAMATH_GPT_geometric_sum_S12_l1520_152090


namespace NUMINAMATH_GPT_ral_current_age_l1520_152097

theorem ral_current_age (Ral_age Suri_age : ℕ) (h1 : Ral_age = 2 * Suri_age) (h2 : Suri_age + 3 = 16) : Ral_age = 26 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_ral_current_age_l1520_152097


namespace NUMINAMATH_GPT_hall_length_l1520_152074

theorem hall_length (b : ℕ) (h1 : b + 5 > 0) (h2 : (b + 5) * b = 750) : b + 5 = 30 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_hall_length_l1520_152074


namespace NUMINAMATH_GPT_solution_Y_required_l1520_152019

theorem solution_Y_required (V_total V_ratio_Y : ℝ) (h_total : V_total = 0.64) (h_ratio : V_ratio_Y = 3 / 8) : 
  (0.64 * (3 / 8) = 0.24) :=
by
  sorry

end NUMINAMATH_GPT_solution_Y_required_l1520_152019


namespace NUMINAMATH_GPT_mark_first_part_playing_time_l1520_152050

open Nat

theorem mark_first_part_playing_time (x : ℕ) (total_game_time second_part_playing_time sideline_time : ℕ)
  (h1 : total_game_time = 90) (h2 : second_part_playing_time = 35) (h3 : sideline_time = 35) 
  (h4 : x + second_part_playing_time + sideline_time = total_game_time) : x = 20 := 
by
  sorry

end NUMINAMATH_GPT_mark_first_part_playing_time_l1520_152050


namespace NUMINAMATH_GPT_tom_total_distance_l1520_152078

/-- Tom swims for 1.5 hours at 2.5 miles per hour. 
    Tom runs for 0.75 hours at 6.5 miles per hour. 
    Tom bikes for 3 hours at 12 miles per hour. 
    The total distance Tom covered is 44.625 miles.
-/
theorem tom_total_distance
  (swim_time : ℝ := 1.5) (swim_speed : ℝ := 2.5)
  (run_time : ℝ := 0.75) (run_speed : ℝ := 6.5)
  (bike_time : ℝ := 3) (bike_speed : ℝ := 12) :
  swim_time * swim_speed + run_time * run_speed + bike_time * bike_speed = 44.625 :=
by
  sorry

end NUMINAMATH_GPT_tom_total_distance_l1520_152078


namespace NUMINAMATH_GPT_arrangement_count_l1520_152023

-- Definitions
def volunteers := 4
def elderly := 2
def total_people := volunteers + elderly
def criteria := "The 2 elderly people must be adjacent but not at the ends of the row."

-- Theorem: The number of different valid arrangements is 144
theorem arrangement_count : 
  ∃ (arrangements : Nat), arrangements = (volunteers.factorial * 3 * elderly.factorial) ∧ arrangements = 144 := 
  by 
    sorry

end NUMINAMATH_GPT_arrangement_count_l1520_152023


namespace NUMINAMATH_GPT_correct_division_result_l1520_152051

theorem correct_division_result : 
  ∀ (a b : ℕ),
  (1722 / (10 * b + a) = 42) →
  (10 * a + b = 14) →
  1722 / 14 = 123 :=
by
  intros a b h1 h2
  sorry

end NUMINAMATH_GPT_correct_division_result_l1520_152051


namespace NUMINAMATH_GPT_widgets_per_shipping_box_l1520_152088

theorem widgets_per_shipping_box :
  let widget_per_carton := 3
  let carton_width := 4
  let carton_length := 4
  let carton_height := 5
  let shipping_box_width := 20
  let shipping_box_length := 20
  let shipping_box_height := 20
  let carton_volume := carton_width * carton_length * carton_height
  let shipping_box_volume := shipping_box_width * shipping_box_length * shipping_box_height
  let cartons_per_shipping_box := shipping_box_volume / carton_volume
  cartons_per_shipping_box * widget_per_carton = 300 :=
by
  sorry

end NUMINAMATH_GPT_widgets_per_shipping_box_l1520_152088


namespace NUMINAMATH_GPT_value_of_a_purely_imaginary_l1520_152031

-- Define the conditions under which a given complex number is purely imaginary
def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = Complex.im z * Complex.I ∧ b ≠ 0

-- Define the complex number based on the variable a
def given_complex_number (a : ℝ) : ℂ :=
  ⟨a^2 - 3*a + 2, a - 1⟩

-- The proof statement
theorem value_of_a_purely_imaginary :
  is_purely_imaginary (given_complex_number 2) := sorry

end NUMINAMATH_GPT_value_of_a_purely_imaginary_l1520_152031


namespace NUMINAMATH_GPT_absent_children_on_teachers_day_l1520_152040

theorem absent_children_on_teachers_day (A : ℕ) (h1 : ∀ n : ℕ, n = 190)
(h2 : ∀ s : ℕ, s = 38) (h3 : ∀ extra : ℕ, extra = 14) :
  (190 - A) * 38 = 190 * 24 → A = 70 :=
by
  sorry

end NUMINAMATH_GPT_absent_children_on_teachers_day_l1520_152040


namespace NUMINAMATH_GPT_mindy_emails_l1520_152072

theorem mindy_emails (P E : ℕ) 
    (h1 : E = 9 * P - 7)
    (h2 : E + P = 93) :
    E = 83 := 
    sorry

end NUMINAMATH_GPT_mindy_emails_l1520_152072


namespace NUMINAMATH_GPT_n_minus_m_eq_singleton_6_l1520_152025

def set_difference (A B : Set α) : Set α :=
  {x | x ∈ A ∧ x ∉ B}

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 6}

theorem n_minus_m_eq_singleton_6 : set_difference N M = {6} :=
by
  sorry

end NUMINAMATH_GPT_n_minus_m_eq_singleton_6_l1520_152025


namespace NUMINAMATH_GPT_cyclic_quadrilateral_equality_l1520_152043

variables {A B C D : ℝ} (AB BC CD DA AC BD : ℝ)

theorem cyclic_quadrilateral_equality 
  (h_cyclic: A * B * C * D = AB * BC * CD * DA)
  (h_sides: AB = A ∧ BC = B ∧ CD = C ∧ DA = D)
  (h_diagonals: AC = E ∧ BD = F) :
  E * (A * B + C * D) = F * (D * A + B * C) :=
sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_equality_l1520_152043


namespace NUMINAMATH_GPT_simplify_expression_l1520_152099

variable (a b : ℝ)

theorem simplify_expression :
  -2 * (a^3 - 3 * b^2) + 4 * (-b^2 + a^3) = 2 * a^3 + 2 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1520_152099


namespace NUMINAMATH_GPT_binomial_distrib_not_equiv_binom_expansion_l1520_152076

theorem binomial_distrib_not_equiv_binom_expansion (a b : ℝ) (n : ℕ) (p : ℝ) (h1: a = p) (h2: b = 1 - p):
    ¬ (∃ k : ℕ, p ^ k * (1 - p) ^ (n - k) = (a + b) ^ n) := sorry

end NUMINAMATH_GPT_binomial_distrib_not_equiv_binom_expansion_l1520_152076


namespace NUMINAMATH_GPT_planes_parallel_l1520_152049

variables (m n : Line) (α β : Plane)

-- Non-overlapping lines and planes conditions
axiom non_overlapping_lines : m ≠ n
axiom non_overlapping_planes : α ≠ β

-- Parallel and perpendicular definitions
axiom parallel_lines (l k : Line) : Prop
axiom parallel_planes (π ρ : Plane) : Prop
axiom perpendicular (l : Line) (π : Plane) : Prop

-- Given conditions
axiom m_perpendicular_to_alpha : perpendicular m α
axiom m_perpendicular_to_beta : perpendicular m β

-- Proof statement
theorem planes_parallel (m_perpendicular_to_alpha : perpendicular m α)
  (m_perpendicular_to_beta : perpendicular m β) :
  parallel_planes α β := sorry

end NUMINAMATH_GPT_planes_parallel_l1520_152049


namespace NUMINAMATH_GPT_find_alpha_l1520_152027

-- Define the given condition that alpha is inversely proportional to beta
def inv_proportional (α β : ℝ) (k : ℝ) : Prop := α * β = k

-- Main theorem statement
theorem find_alpha (α β k : ℝ) (h1 : inv_proportional 2 5 k) (h2 : inv_proportional α (-10) k) : α = -1 := by
  -- Given the conditions, the proof would follow, but it's not required here.
  sorry

end NUMINAMATH_GPT_find_alpha_l1520_152027


namespace NUMINAMATH_GPT_car_mileage_before_modification_l1520_152075

theorem car_mileage_before_modification (miles_per_gallon_before : ℝ) 
  (fuel_efficiency_modifier : ℝ := 0.75) (tank_capacity : ℝ := 12) 
  (extra_miles_after_modification : ℝ := 96) :
  (1 / fuel_efficiency_modifier) * miles_per_gallon_before * (tank_capacity - 1) = 24 :=
by
  sorry

end NUMINAMATH_GPT_car_mileage_before_modification_l1520_152075


namespace NUMINAMATH_GPT_quadrilateral_pyramid_plane_intersection_l1520_152038

-- Definitions:
-- Let MA, MB, MC, MD, MK, ML, MP, MN be lengths of respective segments
-- Let S_ABC, S_ABD, S_ACD, S_BCD be areas of respective triangles
variables {MA MB MC MD MK ML MP MN : ℝ}
variables {S_ABC S_ABD S_ACD S_BCD : ℝ}

-- Given a quadrilateral pyramid MABCD with a convex quadrilateral ABCD as base, and a plane intersecting edges MA, MB, MC, and MD at points K, L, P, and N respectively. Prove the following relation.
theorem quadrilateral_pyramid_plane_intersection :
  S_BCD * (MA / MK) + S_ADB * (MC / MP) = S_ABC * (MD / MN) + S_ACD * (MB / ML) :=
sorry

end NUMINAMATH_GPT_quadrilateral_pyramid_plane_intersection_l1520_152038


namespace NUMINAMATH_GPT_other_acute_angle_measure_l1520_152094

-- Definitions based on the conditions
def right_triangle_sum (a b : ℝ) : Prop := a + b = 90
def is_right_triangle (a b : ℝ) : Prop := right_triangle_sum a b ∧ a = 20

-- The statement to prove
theorem other_acute_angle_measure {a b : ℝ} (h : is_right_triangle a b) : b = 70 :=
sorry

end NUMINAMATH_GPT_other_acute_angle_measure_l1520_152094


namespace NUMINAMATH_GPT_gcd_72_108_l1520_152095

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end NUMINAMATH_GPT_gcd_72_108_l1520_152095


namespace NUMINAMATH_GPT_count_valid_triples_l1520_152021

-- Define the necessary conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_positive (n : ℕ) : Prop := n > 0
def valid_triple (p q n : ℕ) : Prop := is_prime p ∧ is_prime q ∧ is_positive n ∧ (1/p + 2013/q = n/5)

-- Lean statement for the proof problem
theorem count_valid_triples : 
  ∃ c : ℕ, c = 5 ∧ 
  (∀ p q n : ℕ, valid_triple p q n → true) :=
sorry

end NUMINAMATH_GPT_count_valid_triples_l1520_152021


namespace NUMINAMATH_GPT_friends_with_Ron_l1520_152064

-- Ron is eating pizza with his friends 
def total_slices : Nat := 12
def slices_per_person : Nat := 4
def total_people := total_slices / slices_per_person
def ron_included := 1

theorem friends_with_Ron : total_people - ron_included = 2 := by
  sorry

end NUMINAMATH_GPT_friends_with_Ron_l1520_152064


namespace NUMINAMATH_GPT_percent_difference_l1520_152055

theorem percent_difference : 
  let a := 0.60 * 50
  let b := 0.45 * 30
  a - b = 16.5 :=
by
  let a := 0.60 * 50
  let b := 0.45 * 30
  sorry

end NUMINAMATH_GPT_percent_difference_l1520_152055


namespace NUMINAMATH_GPT_points_per_correct_answer_l1520_152079

theorem points_per_correct_answer (x : ℕ) : 
  let total_questions := 30
  let points_deducted_per_incorrect := 5
  let total_score := 325
  let correct_answers := 19
  let incorrect_answers := total_questions - correct_answers
  let points_lost_from_incorrect := incorrect_answers * points_deducted_per_incorrect
  let score_from_correct := correct_answers * x
  (score_from_correct - points_lost_from_incorrect = total_score) → x = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_points_per_correct_answer_l1520_152079


namespace NUMINAMATH_GPT_problem1_problem2_l1520_152004

-- Definition of f(x)
def f (x : ℝ) : ℝ := abs (x - 1)

-- Definition of g(x)
def g (x t : ℝ) : ℝ := t * abs x - 2

-- Problem 1: Proof that f(x) > 2x + 1 implies x < 0
theorem problem1 (x : ℝ) : f x > 2 * x + 1 → x < 0 := by
  sorry

-- Problem 2: Proof that if f(x) ≥ g(x) for all x, then t ≤ 1
theorem problem2 (t : ℝ) : (∀ x : ℝ, f x ≥ g x t) → t ≤ 1 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1520_152004


namespace NUMINAMATH_GPT_divide_bill_evenly_l1520_152012

variable (totalBill amountPaid : ℕ)
variable (numberOfFriends : ℕ)

theorem divide_bill_evenly (h1 : totalBill = 135) (h2 : amountPaid = 45) (h3 : numberOfFriends * amountPaid = totalBill) :
  numberOfFriends = 3 := by
  sorry

end NUMINAMATH_GPT_divide_bill_evenly_l1520_152012


namespace NUMINAMATH_GPT_largest_possible_x_l1520_152057

theorem largest_possible_x :
  ∃ x : ℝ, (3*x^2 + 18*x - 84 = x*(x + 10)) ∧ ∀ y : ℝ, (3*y^2 + 18*y - 84 = y*(y + 10)) → y ≤ x :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_x_l1520_152057


namespace NUMINAMATH_GPT_custom_operator_example_l1520_152028

def custom_operator (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem custom_operator_example : custom_operator 5 3 = 4 := by
  sorry

end NUMINAMATH_GPT_custom_operator_example_l1520_152028


namespace NUMINAMATH_GPT_minimize_square_sum_l1520_152020

theorem minimize_square_sum (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : 
  ∃ x y z, (x + 2 * y + 3 * z = 1) ∧ (x^2 + y^2 + z^2 ≥ 0) ∧ ((x^2 + y^2 + z^2) = 1 / 14) :=
sorry

end NUMINAMATH_GPT_minimize_square_sum_l1520_152020


namespace NUMINAMATH_GPT_pencil_distribution_l1520_152005

-- Formalize the problem in Lean
theorem pencil_distribution (x1 x2 x3 x4 : ℕ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 5) (hx2 : 1 ≤ x2 ∧ x2 ≤ 5) (hx3 : 1 ≤ x3 ∧ x3 ≤ 5) (hx4 : 1 ≤ x4 ∧ x4 ≤ 5) :
  x1 + x2 + x3 + x4 = 10 → 64 = 64 :=
by {
  sorry
}

end NUMINAMATH_GPT_pencil_distribution_l1520_152005


namespace NUMINAMATH_GPT_book_selection_l1520_152034

def num_books_in_genre (mystery fantasy biography : ℕ) : ℕ :=
  mystery + fantasy + biography

def num_combinations_two_diff_genres (mystery fantasy biography : ℕ) : ℕ :=
  if mystery = 4 ∧ fantasy = 4 ∧ biography = 4 then 48 else 0

theorem book_selection : 
  ∀ (mystery fantasy biography : ℕ),
  num_books_in_genre mystery fantasy biography = 12 →
  num_combinations_two_diff_genres mystery fantasy biography = 48 :=
by
  intros mystery fantasy biography h
  sorry

end NUMINAMATH_GPT_book_selection_l1520_152034


namespace NUMINAMATH_GPT_solutions_equiv_cond_l1520_152053

theorem solutions_equiv_cond (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → x^2 + 3 * x + 1 / (x - 1) = a + 1 / (x - 1)) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 + 3 * x = a) ∧ (∃ x : ℝ, x = 1 → a ≠ 4)  :=
sorry

end NUMINAMATH_GPT_solutions_equiv_cond_l1520_152053


namespace NUMINAMATH_GPT_min_value_function_l1520_152093

theorem min_value_function (x y : ℝ) (h1 : -2 < x ∧ x < 2) (h2 : -2 < y ∧ y < 2) (h3 : x * y = -1) :
  ∃ u : ℝ, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) ∧ u = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_function_l1520_152093


namespace NUMINAMATH_GPT_greatest_third_term_of_arithmetic_sequence_l1520_152096

def is_arithmetic_sequence (a b c d : ℤ) : Prop := (b - a = c - b) ∧ (c - b = d - c)

theorem greatest_third_term_of_arithmetic_sequence :
  ∃ a b c d : ℤ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  is_arithmetic_sequence a b c d ∧
  (a + b + c + d = 52) ∧
  (c = 17) :=
sorry

end NUMINAMATH_GPT_greatest_third_term_of_arithmetic_sequence_l1520_152096


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l1520_152068

theorem largest_divisor_of_expression (x : ℤ) (hx : x % 2 = 1) :
  864 ∣ (12 * x + 2) * (12 * x + 6) * (12 * x + 10) * (6 * x + 3) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l1520_152068


namespace NUMINAMATH_GPT_soda_count_l1520_152018

theorem soda_count
  (W : ℕ) (S : ℕ) (B : ℕ) (T : ℕ)
  (hW : W = 26) (hB : B = 17) (hT : T = 31) :
  W + S - B = T → S = 22 :=
by
  sorry

end NUMINAMATH_GPT_soda_count_l1520_152018


namespace NUMINAMATH_GPT_three_digit_diff_no_repeated_digits_l1520_152070

theorem three_digit_diff_no_repeated_digits :
  let largest := 987
  let smallest := 102
  largest - smallest = 885 := by
  sorry

end NUMINAMATH_GPT_three_digit_diff_no_repeated_digits_l1520_152070


namespace NUMINAMATH_GPT_intersection_M_N_l1520_152071

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

theorem intersection_M_N : M ∩ N = {0, 3} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1520_152071


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_15_l1520_152016

theorem smallest_four_digit_multiple_of_15 :
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 15 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 15 = 0) → n ≤ m) ∧ n = 1005 :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_15_l1520_152016


namespace NUMINAMATH_GPT_marble_draw_probability_l1520_152013

def marble_probabilities : ℚ :=
  let prob_white_a := 5 / 10
  let prob_black_a := 5 / 10
  let prob_yellow_b := 8 / 15
  let prob_yellow_c := 3 / 10
  let prob_green_d := 6 / 10
  let prob_white_then_yellow_then_green := prob_white_a * prob_yellow_b * prob_green_d
  let prob_black_then_yellow_then_green := prob_black_a * prob_yellow_c * prob_green_d
  prob_white_then_yellow_then_green + prob_black_then_yellow_then_green

theorem marble_draw_probability :
  marble_probabilities = 17 / 50 := by
  sorry

end NUMINAMATH_GPT_marble_draw_probability_l1520_152013


namespace NUMINAMATH_GPT_remainder_when_x_plus_3uy_div_y_l1520_152039

theorem remainder_when_x_plus_3uy_div_y (x y u v : ℕ) (hx : x = u * y + v) (v_lt_y : v < y) :
  ((x + 3 * u * y) % y) = v := 
sorry

end NUMINAMATH_GPT_remainder_when_x_plus_3uy_div_y_l1520_152039


namespace NUMINAMATH_GPT_tin_can_allocation_l1520_152042

-- Define the total number of sheets of tinplate available
def total_sheets := 108

-- Define the number of sheets used for can bodies
variable (x : ℕ)

-- Define the number of can bodies a single sheet makes
def can_bodies_per_sheet := 15

-- Define the number of can bottoms a single sheet makes
def can_bottoms_per_sheet := 42

-- Define the equation to be proven
theorem tin_can_allocation :
  2 * can_bodies_per_sheet * x = can_bottoms_per_sheet * (total_sheets - x) :=
  sorry

end NUMINAMATH_GPT_tin_can_allocation_l1520_152042


namespace NUMINAMATH_GPT_fraction_shaded_l1520_152069

-- Define relevant elements
def quilt : ℕ := 9
def rows : ℕ := 3
def shaded_rows : ℕ := 1
def shaded_fraction := shaded_rows / rows

-- We are to prove the fraction of the quilt that is shaded
theorem fraction_shaded (h : quilt = 3 * 3) : shaded_fraction = 1 / 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fraction_shaded_l1520_152069


namespace NUMINAMATH_GPT_prime_5p_plus_4p4_is_perfect_square_l1520_152098

theorem prime_5p_plus_4p4_is_perfect_square (p : ℕ) [Fact (Nat.Prime p)] :
  ∃ q : ℕ, 5^p + 4 * p^4 = q^2 ↔ p = 5 :=
by
  sorry

end NUMINAMATH_GPT_prime_5p_plus_4p4_is_perfect_square_l1520_152098


namespace NUMINAMATH_GPT_Kaarel_wins_l1520_152091

theorem Kaarel_wins (p : ℕ) (hp : Prime p) (hp_gt3 : p > 3) :
  ∃ (x y a : ℕ), x ∈ Finset.range (p-1) ∧ y ∈ Finset.range (p-1) ∧ a ∈ Finset.range (p-1) ∧ 
  x ≠ y ∧ y ≠ (p - x) ∧ a ≠ x ∧ a ≠ (p - x) ∧ a ≠ y ∧ 
  (x * (p - x) + y * a) % p = 0 :=
sorry

end NUMINAMATH_GPT_Kaarel_wins_l1520_152091


namespace NUMINAMATH_GPT_correct_subtraction_l1520_152047

theorem correct_subtraction (x : ℕ) (h : x - 42 = 50) : x - 24 = 68 :=
  sorry

end NUMINAMATH_GPT_correct_subtraction_l1520_152047


namespace NUMINAMATH_GPT_pipe_filling_time_l1520_152046

theorem pipe_filling_time 
  (rate_A : ℚ := 1/8) 
  (rate_L : ℚ := 1/24) :
  (1 / (rate_A - rate_L) = 12) :=
by
  sorry

end NUMINAMATH_GPT_pipe_filling_time_l1520_152046


namespace NUMINAMATH_GPT_glucose_amount_in_45cc_l1520_152087

noncomputable def glucose_in_container (concentration : ℝ) (total_volume : ℝ) (poured_volume : ℝ) : ℝ :=
  (concentration * poured_volume) / total_volume

theorem glucose_amount_in_45cc (concentration : ℝ) (total_volume : ℝ) (poured_volume : ℝ) :
  concentration = 10 → total_volume = 100 → poured_volume = 45 →
  glucose_in_container concentration total_volume poured_volume = 4.5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_glucose_amount_in_45cc_l1520_152087


namespace NUMINAMATH_GPT_part_one_solution_part_two_solution_l1520_152029

-- (I) Prove the solution set for the given inequality with m = 2.
theorem part_one_solution (x : ℝ) : 
  (|x - 2| > 7 - |x - 1|) ↔ (x < -4 ∨ x > 5) :=
sorry

-- (II) Prove the range of m given the condition.
theorem part_two_solution (m : ℝ) : 
  (∃ x : ℝ, |x - m| > 7 + |x - 1|) ↔ (m ∈ Set.Iio (-6) ∪ Set.Ioi (8)) :=
sorry

end NUMINAMATH_GPT_part_one_solution_part_two_solution_l1520_152029


namespace NUMINAMATH_GPT_total_pencils_correct_l1520_152067

-- Define the number of pencils Reeta has
def ReetaPencils : ℕ := 20

-- Define the number of pencils Anika has based on the conditions
def AnikaPencils : ℕ := 2 * ReetaPencils + 4

-- Define the total number of pencils Anika and Reeta have together
def TotalPencils : ℕ := ReetaPencils + AnikaPencils

-- Statement to prove
theorem total_pencils_correct : TotalPencils = 64 :=
by
  sorry

end NUMINAMATH_GPT_total_pencils_correct_l1520_152067


namespace NUMINAMATH_GPT_actual_distance_traveled_l1520_152080

theorem actual_distance_traveled (D : ℕ) 
  (h : D / 10 = (D + 36) / 16) : D = 60 := by
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l1520_152080


namespace NUMINAMATH_GPT_total_amount_l1520_152010

theorem total_amount (N50 N: ℕ) (h1: N = 90) (h2: N50 = 77) : 
  (N50 * 50 + (N - N50) * 500) = 10350 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_l1520_152010


namespace NUMINAMATH_GPT_sufficient_condition_p_or_q_false_p_and_q_false_l1520_152054

variables (p q : Prop)

theorem sufficient_condition_p_or_q_false_p_and_q_false :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ ¬ ( (¬ (p ∧ q)) → ¬ (p ∨ q)) :=
by 
  -- Proof: If ¬ (p ∨ q), then (p ∨ q) is false, which means (p ∧ q) must also be false.
  -- The other direction would mean if at least one of p or q is false, then (p ∨ q) is false,
  -- which is not necessarily true. Therefore, it's not a necessary condition.
  sorry

end NUMINAMATH_GPT_sufficient_condition_p_or_q_false_p_and_q_false_l1520_152054


namespace NUMINAMATH_GPT_volunteer_group_selection_l1520_152030

theorem volunteer_group_selection :
  let M := 4  -- Number of male teachers
  let F := 5  -- Number of female teachers
  let G := 3  -- Total number of teachers in the group
  -- Calculate the number of ways to select 2 male teachers and 1 female teacher
  let ways1 := (Nat.choose M 2) * (Nat.choose F 1)
  -- Calculate the number of ways to select 1 male teacher and 2 female teachers
  let ways2 := (Nat.choose M 1) * (Nat.choose F 2)
  -- The total number of ways to form the group
  ways1 + ways2 = 70 := by sorry

end NUMINAMATH_GPT_volunteer_group_selection_l1520_152030


namespace NUMINAMATH_GPT_true_universal_quantifier_l1520_152092

theorem true_universal_quantifier :
  ∀ (a b : ℝ), a^2 + b^2 ≥ 2 * (a - b - 1) := by
  sorry

end NUMINAMATH_GPT_true_universal_quantifier_l1520_152092


namespace NUMINAMATH_GPT_tangents_parallel_l1520_152007

-- Definitions based on the conditions in part A
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_line (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

def secant_intersection (c1 c2 : Circle) (A : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  sorry

-- Main theorem statement
theorem tangents_parallel 
  (c1 c2 : Circle) (A B C : ℝ × ℝ) 
  (h1 : c1.center ≠ c2.center) 
  (h2 : dist c1.center c2.center = c1.radius + c2.radius) 
  (h3 : (B, C) = secant_intersection c1 c2 A) 
  (h4 : tangent_line c1 B ≠ tangent_line c2 C) :
  tangent_line c1 B = tangent_line c2 C :=
sorry

end NUMINAMATH_GPT_tangents_parallel_l1520_152007


namespace NUMINAMATH_GPT_sum_of_three_ints_product_5_4_l1520_152082

theorem sum_of_three_ints_product_5_4 :
  ∃ (a b c: ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 51 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_ints_product_5_4_l1520_152082


namespace NUMINAMATH_GPT_interest_difference_20_years_l1520_152065

def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n
def simple_interest (P r : ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

theorem interest_difference_20_years :
  compound_interest 15000 0.06 20 - simple_interest 15000 0.08 20 = 9107 :=
by
  sorry

end NUMINAMATH_GPT_interest_difference_20_years_l1520_152065


namespace NUMINAMATH_GPT_positive_3_digit_numbers_divisible_by_13_count_l1520_152015

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end NUMINAMATH_GPT_positive_3_digit_numbers_divisible_by_13_count_l1520_152015


namespace NUMINAMATH_GPT_range_of_a_part1_range_of_a_part2_l1520_152000

theorem range_of_a_part1 (a : ℝ) :
  (∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) → 0 < a ∧ a < 4 :=
sorry

theorem range_of_a_part2 (a : ℝ) :
  ((∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) ∨ (∃ x : ℝ, x^2 - x + a = 0)) ∧ ¬((∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) ∧ (∃ x : ℝ, x^2 - x + a = 0)) →
  a ≤ 0 ∨ (1 / 4) < a ∧ a < 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_part1_range_of_a_part2_l1520_152000


namespace NUMINAMATH_GPT_required_circle_equation_l1520_152084

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line equation on which the center of the required circle lies
def center_line (x y : ℝ) : Prop := 3 * x + 4 * y - 1 = 0

-- State the final proof that the equation of the required circle is (x + 1)^2 + (y - 1)^2 = 13 under the given conditions
theorem required_circle_equation (x y : ℝ) :
  ( ∃ (x1 y1 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧
    (∃ (cx cy r : ℝ), center_line cx cy ∧ (x - cx)^2 + (y - cy)^2 = r^2 ∧ (x1 - cx)^2 + (y1 - cy)^2 = r^2 ∧
      (x + 1)^2 + (y - 1)^2 = 13) )
 := sorry

end NUMINAMATH_GPT_required_circle_equation_l1520_152084


namespace NUMINAMATH_GPT_Jill_arrives_9_minutes_later_l1520_152036

theorem Jill_arrives_9_minutes_later
  (distance : ℝ)
  (Jack_speed : ℝ)
  (Jill_speed : ℝ)
  (h1 : distance = 1)
  (h2 : Jack_speed = 10)
  (h3 : Jill_speed = 4) :
  ((distance / Jill_speed) - (distance / Jack_speed)) * 60 = 9 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_Jill_arrives_9_minutes_later_l1520_152036


namespace NUMINAMATH_GPT_number_of_players_is_correct_l1520_152045

-- Defining the problem conditions
def wristband_cost : ℕ := 6
def jersey_cost : ℕ := wristband_cost + 7
def wristbands_per_player : ℕ := 4
def jerseys_per_player : ℕ := 2
def total_expenditure : ℕ := 3774

-- Calculating cost per player and stating the proof problem
def cost_per_player : ℕ := wristbands_per_player * wristband_cost +
                           jerseys_per_player * jersey_cost

def number_of_players : ℕ := total_expenditure / cost_per_player

-- The final proof statement to show that number_of_players is 75
theorem number_of_players_is_correct : number_of_players = 75 :=
by sorry

end NUMINAMATH_GPT_number_of_players_is_correct_l1520_152045


namespace NUMINAMATH_GPT_combined_sleep_hours_l1520_152066

def connor_hours : ℕ := 6
def luke_hours : ℕ := connor_hours + 2
def emma_hours : ℕ := connor_hours - 1
def puppy_hours : ℕ := 2 * luke_hours

theorem combined_sleep_hours :
  connor_hours + luke_hours + emma_hours + puppy_hours = 35 := by
  sorry

end NUMINAMATH_GPT_combined_sleep_hours_l1520_152066


namespace NUMINAMATH_GPT_small_circles_sixth_figure_l1520_152085

-- Defining the function to calculate the number of circles in the nth figure
def small_circles (n : ℕ) : ℕ :=
  n * (n + 1) + 4

-- Statement of the theorem
theorem small_circles_sixth_figure :
  small_circles 6 = 46 :=
by sorry

end NUMINAMATH_GPT_small_circles_sixth_figure_l1520_152085


namespace NUMINAMATH_GPT_all_three_pets_l1520_152044

-- Definitions of the given conditions
def total_students : ℕ := 40
def dog_owners : ℕ := 20
def cat_owners : ℕ := 13
def other_pet_owners : ℕ := 8
def no_pets : ℕ := 7

-- Definitions from Venn diagram
def dogs_only : ℕ := 12
def cats_only : ℕ := 3
def other_pets_only : ℕ := 2

-- Intersection variables
variables (a b c d : ℕ)

-- Translated problem
theorem all_three_pets :
  dogs_only + cats_only + other_pets_only + a + b + c + d = total_students - no_pets ∧
  dogs_only + a + c + d = dog_owners ∧
  cats_only + a + b + d = cat_owners ∧
  other_pets_only + b + c + d = other_pet_owners ∧
  d = 2 :=
sorry

end NUMINAMATH_GPT_all_three_pets_l1520_152044


namespace NUMINAMATH_GPT_assistant_increases_output_by_100_percent_l1520_152077

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end NUMINAMATH_GPT_assistant_increases_output_by_100_percent_l1520_152077


namespace NUMINAMATH_GPT_tenth_term_of_sequence_l1520_152059

-- Define the first term and the common difference
def a1 : ℤ := 10
def d : ℤ := -2

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) : ℤ := a1 + d * (n - 1)

-- State the theorem about the 10th term
theorem tenth_term_of_sequence : a_n 10 = -8 := by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_tenth_term_of_sequence_l1520_152059


namespace NUMINAMATH_GPT_single_intersection_l1520_152086

theorem single_intersection (k : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, y^2 = x ∧ y + 1 = k * x) ↔ (k = 0 ∨ k = -1 / 4) :=
sorry

end NUMINAMATH_GPT_single_intersection_l1520_152086


namespace NUMINAMATH_GPT_reciprocal_solution_l1520_152001

theorem reciprocal_solution {x : ℝ} (h : x * -9 = 1) : x = -1/9 :=
sorry

end NUMINAMATH_GPT_reciprocal_solution_l1520_152001


namespace NUMINAMATH_GPT_arithmetic_sequence_S9_l1520_152032

-- Define the sum of an arithmetic sequence: S_n
def arithmetic_sequence_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ := (n * (2 * a (0) + (n - 1) * d (0))) / 2

-- Conditions
variable (a d : ℕ → ℕ)
variable (S_n : ℕ → ℕ)
variable (h1 : S_n 3 = 9)
variable (h2 : S_n 6 = 27)

-- Question: Prove that S_9 = 54
theorem arithmetic_sequence_S9 : S_n 9 = 54 := by
    sorry

end NUMINAMATH_GPT_arithmetic_sequence_S9_l1520_152032


namespace NUMINAMATH_GPT_jamie_paid_0_more_than_alex_l1520_152041

/-- Conditions:
     1. Alex and Jamie shared a pizza cut into 10 equally-sized slices.
     2. Alex wanted a plain pizza.
     3. Jamie wanted a special spicy topping on one-third of the pizza.
     4. The cost of a plain pizza was $10.
     5. The spicy topping on one-third of the pizza cost an additional $3.
     6. Jamie ate all the slices with the spicy topping and two extra plain slices.
     7. Alex ate the remaining plain slices.
     8. They each paid for what they ate.
    
     Question: How many more dollars did Jamie pay than Alex?
     Answer: 0
-/
theorem jamie_paid_0_more_than_alex :
  let total_slices := 10
  let cost_plain := 10
  let cost_spicy := 3
  let total_cost := cost_plain + cost_spicy
  let cost_per_slice := total_cost / total_slices
  let jamie_slices := 5
  let alex_slices := total_slices - jamie_slices
  let jamie_cost := jamie_slices * cost_per_slice
  let alex_cost := alex_slices * cost_per_slice
  jamie_cost - alex_cost = 0 :=
by
  sorry

end NUMINAMATH_GPT_jamie_paid_0_more_than_alex_l1520_152041


namespace NUMINAMATH_GPT_ratio_of_sides_l1520_152026
-- Import the complete math library

-- Define the conditions as hypotheses
variables (s x y : ℝ)
variable (h_outer_area : (3 * s)^2 = 9 * s^2)
variable (h_side_lengths : 3 * s = s + 2 * x)
variable (h_y_length : y + x = 3 * s)

-- State the theorem
theorem ratio_of_sides (h_outer_area : (3 * s)^2 = 9 * s^2)
  (h_side_lengths : 3 * s = s + 2 * x)
  (h_y_length : y + x = 3 * s) :
  y / x = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_sides_l1520_152026


namespace NUMINAMATH_GPT_gcd_in_range_l1520_152024

theorem gcd_in_range :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Int.gcd n 30 = 10 :=
sorry

end NUMINAMATH_GPT_gcd_in_range_l1520_152024


namespace NUMINAMATH_GPT_chocolate_bars_gigantic_box_l1520_152037

def large_boxes : ℕ := 50
def medium_boxes : ℕ := 25
def small_boxes : ℕ := 10
def chocolate_bars_per_small_box : ℕ := 45

theorem chocolate_bars_gigantic_box : 
  large_boxes * medium_boxes * small_boxes * chocolate_bars_per_small_box = 562500 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bars_gigantic_box_l1520_152037


namespace NUMINAMATH_GPT_hex_A08_to_decimal_l1520_152017

noncomputable def hex_A := 10
noncomputable def hex_A08_base_10 : ℕ :=
  (hex_A * 16^2) + (0 * 16^1) + (8 * 16^0)

theorem hex_A08_to_decimal :
  hex_A08_base_10 = 2568 :=
by
  sorry

end NUMINAMATH_GPT_hex_A08_to_decimal_l1520_152017


namespace NUMINAMATH_GPT_log_expression_l1520_152009

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression : 
  log 2 * log 50 + log 25 - log 5 * log 20 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_log_expression_l1520_152009


namespace NUMINAMATH_GPT_sum_y_coordinates_of_intersection_with_y_axis_l1520_152003

-- Define the center and radius of the circle
def center : ℝ × ℝ := (-4, 5)
def radius : ℝ := 9

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + center.1)^2 + (y - center.2)^2 = radius^2

theorem sum_y_coordinates_of_intersection_with_y_axis : 
  ∃ y1 y2 : ℝ, circle_eq 0 y1 ∧ circle_eq 0 y2 ∧ y1 + y2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_y_coordinates_of_intersection_with_y_axis_l1520_152003


namespace NUMINAMATH_GPT_part1_part2_l1520_152035

noncomputable def A (a : ℝ) : Set ℝ := { x | a * x^2 - 3 * x + 2 = 0 }

theorem part1 (a : ℝ) : (A a = ∅) ↔ (a > 9/8) := sorry

theorem part2 (a : ℝ) : 
  (∃ x, A a = {x}) ↔ 
  (a = 0 ∧ A a = {2 / 3})
  ∨ (a = 9 / 8 ∧ A a = {4 / 3}) := sorry

end NUMINAMATH_GPT_part1_part2_l1520_152035


namespace NUMINAMATH_GPT_art_gallery_ratio_l1520_152014

theorem art_gallery_ratio (A : ℕ) (D : ℕ) (S_not_displayed : ℕ) (P_not_displayed : ℕ)
  (h1 : A = 2700)
  (h2 : 1 / 6 * D = D / 6)
  (h3 : P_not_displayed = S_not_displayed / 3)
  (h4 : S_not_displayed = 1200) :
  D / A = 11 / 27 := by
  sorry

end NUMINAMATH_GPT_art_gallery_ratio_l1520_152014


namespace NUMINAMATH_GPT_distance_between_stripes_l1520_152062

theorem distance_between_stripes (d₁ d₂ L W : ℝ) (h : ℝ)
  (h₁ : d₁ = 60)  -- distance between parallel curbs
  (h₂ : L = 30)  -- length of the curb between stripes
  (h₃ : d₂ = 80)  -- length of each stripe
  (area_eq : W * L = 1800) -- area of the parallelogram with base L
: h = 22.5 :=
by
  -- This is to assume the equation derived from area calculation
  have area_eq' : d₂ * h = 1800 := by sorry
  -- Solving for h using the derived area equation
  have h_calc : h = 1800 / 80 := by sorry
  -- Simplifying the result
  have h_simplified : h = 22.5 := by sorry
  exact h_simplified

end NUMINAMATH_GPT_distance_between_stripes_l1520_152062


namespace NUMINAMATH_GPT_find_AD_l1520_152061

-- Define the geometrical context and constraints
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB AC AD BD CD : ℝ) (x : ℝ)

-- Assume the given conditions
def problem_conditions := 
  (AB = 50) ∧
  (AC = 41) ∧
  (BD = 10 * x) ∧
  (CD = 3 * x) ∧
  (AB^2 = AD^2 + BD^2) ∧
  (AC^2 = AD^2 + CD^2)

-- Formulate the problem question and the correct answer
theorem find_AD (h : problem_conditions AB AC AD BD CD x) : AD = 40 :=
sorry

end NUMINAMATH_GPT_find_AD_l1520_152061


namespace NUMINAMATH_GPT_sum_of_midpoints_l1520_152048

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_l1520_152048


namespace NUMINAMATH_GPT_symmetric_point_proof_l1520_152083

noncomputable def point_symmetric_to_x_axis (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, -A.2)

theorem symmetric_point_proof :
  point_symmetric_to_x_axis (-2, 3) = (-2, -3) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_proof_l1520_152083


namespace NUMINAMATH_GPT_range_of_2a_plus_b_range_of_a_minus_b_range_of_a_div_b_l1520_152006

variable {a b : ℝ}

theorem range_of_2a_plus_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -10 < 2*a + b ∧ 2*a + b < 19 :=
by
  sorry

theorem range_of_a_minus_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -9 < a - b ∧ a - b < 6 :=
by
  sorry

theorem range_of_a_div_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -2 < a / b ∧ a / b < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_2a_plus_b_range_of_a_minus_b_range_of_a_div_b_l1520_152006


namespace NUMINAMATH_GPT_solve_system_of_equations_l1520_152081

def proof_problem (a b c : ℚ) : Prop :=
  ((a - b = 2) ∧ (c = -5) ∧ (2 * a - 6 * b = 2)) → 
  (a = 5 / 2 ∧ b = 1 / 2 ∧ c = -5)

theorem solve_system_of_equations (a b c : ℚ) :
  proof_problem a b c :=
  by
    sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1520_152081


namespace NUMINAMATH_GPT_num_positive_integers_l1520_152011

theorem num_positive_integers (n : ℕ) :
    (0 < n ∧ n < 40 ∧ ∃ k : ℕ, k > 0 ∧ n = 40 * k / (k + 1)) ↔ 
    (n = 20 ∨ n = 30 ∨ n = 32 ∨ n = 35 ∨ n = 36 ∨ n = 38 ∨ n = 39) :=
sorry

end NUMINAMATH_GPT_num_positive_integers_l1520_152011


namespace NUMINAMATH_GPT_smallest_y_l1520_152060

theorem smallest_y (y : ℕ) (h : 56 * y + 8 ≡ 6 [MOD 26]) : y = 6 := by
  sorry

end NUMINAMATH_GPT_smallest_y_l1520_152060


namespace NUMINAMATH_GPT_clea_ride_escalator_time_l1520_152033

theorem clea_ride_escalator_time
  (s v d : ℝ)
  (h1 : 75 * s = d)
  (h2 : 30 * (s + v) = d) :
  t = 50 :=
by
  sorry

end NUMINAMATH_GPT_clea_ride_escalator_time_l1520_152033


namespace NUMINAMATH_GPT_profit_percentage_l1520_152008

theorem profit_percentage (selling_price profit : ℝ) (h1 : selling_price = 900) (h2 : profit = 300) : 
  (profit / (selling_price - profit)) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l1520_152008


namespace NUMINAMATH_GPT_Tom_total_spend_l1520_152056

theorem Tom_total_spend :
  let notebook_price := 2
  let notebook_discount := 0.75
  let notebook_count := 4
  let magazine_price := 5
  let magazine_count := 2
  let pen_price := 1.50
  let pen_discount := 0.75
  let pen_count := 3
  let book_price := 12
  let book_count := 1
  let discount_threshold := 30
  let coupon_discount := 10
  let total_cost :=
    (notebook_count * (notebook_price * notebook_discount)) +
    (magazine_count * magazine_price) +
    (pen_count * (pen_price * pen_discount)) +
    (book_count * book_price)
  let final_cost := if total_cost >= discount_threshold then total_cost - coupon_discount else total_cost
  final_cost = 21.375 :=
by
  sorry

end NUMINAMATH_GPT_Tom_total_spend_l1520_152056


namespace NUMINAMATH_GPT_problem_statement_l1520_152089

noncomputable def term_with_largest_binomial_coefficient
  (M N P : ℕ)
  (h_sum : M + N - P = 2016)
  (n : ℕ) : ℤ :=
-8064

noncomputable def term_with_largest_absolute_value_coefficient
  (M N P : ℕ)
  (h_sum : M + N - P = 2016)
  (n : ℕ) : ℤ × ℕ :=
(-15360, 8)

theorem problem_statement (M N P : ℕ) (h_sum : M + N - P = 2016) (n : ℕ) :
  ((term_with_largest_binomial_coefficient M N P h_sum n = -8064) ∧ 
   (term_with_largest_absolute_value_coefficient M N P h_sum n = (-15360, 8))) :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1520_152089


namespace NUMINAMATH_GPT_smallest_y_for_perfect_square_l1520_152063

theorem smallest_y_for_perfect_square (x y: ℕ) (h : x = 5 * 32 * 45) (hY: y = 2) : 
  ∃ v: ℕ, (x * y = v ^ 2) :=
by
  use 2
  rw [h, hY]
  -- expand and simplify
  sorry

end NUMINAMATH_GPT_smallest_y_for_perfect_square_l1520_152063


namespace NUMINAMATH_GPT_simplify_expression_l1520_152073

theorem simplify_expression (x : ℝ) : 
  ((3 * x - 6) - 5 * x) / 3 = - (2 / 3) * x - 2 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1520_152073
