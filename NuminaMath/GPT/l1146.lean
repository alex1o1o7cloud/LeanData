import Mathlib

namespace NUMINAMATH_GPT_Qing_Dynasty_Problem_l1146_114643

variable {x y : ℕ}

theorem Qing_Dynasty_Problem (h1 : 4 * x + 6 * y = 48) (h2 : 2 * x + 5 * y = 38) :
  (4 * x + 6 * y = 48) ∧ (2 * x + 5 * y = 38) := by
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_Qing_Dynasty_Problem_l1146_114643


namespace NUMINAMATH_GPT_total_pieces_of_junk_mail_l1146_114657

def pieces_per_block : ℕ := 48
def num_blocks : ℕ := 4

theorem total_pieces_of_junk_mail : (pieces_per_block * num_blocks) = 192 := by
  sorry

end NUMINAMATH_GPT_total_pieces_of_junk_mail_l1146_114657


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1146_114671

def A : Set ℝ := {x | x - 1 > 1}
def B : Set ℝ := {x | x < 3}

theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1146_114671


namespace NUMINAMATH_GPT_sale_in_fifth_month_l1146_114619

theorem sale_in_fifth_month (a1 a2 a3 a4 a5 a6 avg : ℝ)
  (h1 : a1 = 5420) (h2 : a2 = 5660) (h3 : a3 = 6200) (h4 : a4 = 6350) (h6 : a6 = 6470) (h_avg : avg = 6100) :
  a5 = 6500 :=
by
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l1146_114619


namespace NUMINAMATH_GPT_tan_alpha_implication_l1146_114684

theorem tan_alpha_implication (α : ℝ) (h : Real.tan α = 2) :
    (2 * Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_implication_l1146_114684


namespace NUMINAMATH_GPT_find_x2_plus_y2_l1146_114618

open Real

theorem find_x2_plus_y2 (x y : ℝ) 
  (h1 : (x + y) ^ 4 + (x - y) ^ 4 = 4112)
  (h2 : x ^ 2 - y ^ 2 = 16) :
  x ^ 2 + y ^ 2 = 34 := 
sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l1146_114618


namespace NUMINAMATH_GPT_z_real_iff_z_complex_iff_z_pure_imaginary_iff_l1146_114687

-- Definitions for the problem conditions
def z_real (m : ℝ) : Prop := (m^2 - 2 * m - 15 = 0)
def z_pure_imaginary (m : ℝ) : Prop := (m^2 - 9 * m - 36 = 0) ∧ (m^2 - 2 * m - 15 ≠ 0)

-- Question 1: Prove that z is a real number if and only if m = -3 or m = 5
theorem z_real_iff (m : ℝ) : z_real m ↔ m = -3 ∨ m = 5 := sorry

-- Question 2: Prove that z is a complex number with non-zero imaginary part if and only if m ≠ -3 and m ≠ 5
theorem z_complex_iff (m : ℝ) : ¬z_real m ↔ m ≠ -3 ∧ m ≠ 5 := sorry

-- Question 3: Prove that z is a pure imaginary number if and only if m = 12
theorem z_pure_imaginary_iff (m : ℝ) : z_pure_imaginary m ↔ m = 12 := sorry

end NUMINAMATH_GPT_z_real_iff_z_complex_iff_z_pure_imaginary_iff_l1146_114687


namespace NUMINAMATH_GPT_max_value_expression_l1146_114633

variable (a b : ℝ)

theorem max_value_expression (h : a^2 + b^2 = 3 + a * b) : 
  ∃ a b : ℝ, (2 * a - 3 * b)^2 + (a + 2 * b) * (a - 2 * b) = 22 :=
by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_max_value_expression_l1146_114633


namespace NUMINAMATH_GPT_probability_of_picking_letter_in_mathematics_l1146_114640

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_picking_letter_in_mathematics :
  (unique_letters_in_mathematics.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_picking_letter_in_mathematics_l1146_114640


namespace NUMINAMATH_GPT_line_through_point_area_T_l1146_114667

variable (a T : ℝ)

def triangle_line_equation (a T : ℝ) : Prop :=
  ∃ y x : ℝ, (a^2 * y + 2 * T * x - 2 * a * T = 0) ∧ (y = -((2 * T)/a^2) * x + (2 * T) / a) ∧ (x ≥ 0) ∧ (y ≥ 0)

theorem line_through_point_area_T (a T : ℝ) (h₁ : a > 0) (h₂ : T > 0) :
  triangle_line_equation a T :=
sorry

end NUMINAMATH_GPT_line_through_point_area_T_l1146_114667


namespace NUMINAMATH_GPT_roses_cut_from_garden_l1146_114602

-- Define the variables and conditions
variables {x : ℕ} -- x is the number of freshly cut roses

def initial_roses : ℕ := 17
def roses_thrown_away : ℕ := 8
def roses_final_vase : ℕ := 42
def roses_given_away : ℕ := 6

-- The condition that describes the total roses now
def condition (x : ℕ) : Prop :=
  initial_roses - roses_thrown_away + (1/3 : ℚ) * x = roses_final_vase

-- The verification step that checks the total roses concerning given away roses
def verification (x : ℕ) : Prop :=
  (1/3 : ℚ) * x + roses_given_away = roses_final_vase + roses_given_away

-- The main theorem to prove the number of roses cut
theorem roses_cut_from_garden (x : ℕ) (h1 : condition x) (h2 : verification x) : x = 99 :=
  sorry

end NUMINAMATH_GPT_roses_cut_from_garden_l1146_114602


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l1146_114691

theorem parabola_focus_coordinates (x y : ℝ) (h : y = -2 * x^2) : (0, -1 / 8) = (0, (-1 / 2) * (y: ℝ)) :=
sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l1146_114691


namespace NUMINAMATH_GPT_counting_error_l1146_114674

theorem counting_error
  (b g : ℕ)
  (initial_balloons := 5 * b + 4 * g)
  (popped_balloons := g + 2 * b)
  (remaining_balloons := initial_balloons - popped_balloons)
  (Dima_count := 100) :
  remaining_balloons ≠ Dima_count := by
  sorry

end NUMINAMATH_GPT_counting_error_l1146_114674


namespace NUMINAMATH_GPT_one_fifth_greater_than_decimal_by_term_l1146_114606

noncomputable def one_fifth := (1 : ℝ) / 5
noncomputable def decimal_value := 20000001 / 10^8
noncomputable def term := 1 / (5 * 10^8)

theorem one_fifth_greater_than_decimal_by_term :
  one_fifth > decimal_value ∧ one_fifth - decimal_value = term :=
  sorry

end NUMINAMATH_GPT_one_fifth_greater_than_decimal_by_term_l1146_114606


namespace NUMINAMATH_GPT_jessica_needs_stamps_l1146_114654

-- Define the weights and conditions
def weight_of_paper := 1 / 5
def total_papers := 8
def weight_of_envelope := 2 / 5
def stamps_per_ounce := 1

-- Calculate the total weight and determine the number of stamps needed
theorem jessica_needs_stamps : 
  total_papers * weight_of_paper + weight_of_envelope = 2 :=
by
  sorry

end NUMINAMATH_GPT_jessica_needs_stamps_l1146_114654


namespace NUMINAMATH_GPT_brianna_books_gift_l1146_114613

theorem brianna_books_gift (books_per_month : ℕ) (months_per_year : ℕ) (books_bought : ℕ) 
  (borrow_difference : ℕ) (books_reread : ℕ) (total_books_needed : ℕ) : 
  (books_per_month * months_per_year = total_books_needed) →
  ((books_per_month * months_per_year) - books_reread - 
  (books_bought + (books_bought - borrow_difference)) = 
  books_given) →
  books_given = 6 := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_brianna_books_gift_l1146_114613


namespace NUMINAMATH_GPT_product_of_real_values_l1146_114629

theorem product_of_real_values (r : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (1 / (3 * x)) = (r - x) / 8 → (3 * x * x - 3 * r * x + 8 = 0)) →
  r = 4 * Real.sqrt 6 / 3 ∨ r = -(4 * Real.sqrt 6 / 3) →
  r * -r = -32 / 3 :=
by
  intro h_x
  intro h_r
  sorry

end NUMINAMATH_GPT_product_of_real_values_l1146_114629


namespace NUMINAMATH_GPT_fractional_equation_root_l1146_114695

theorem fractional_equation_root (k : ℚ) (x : ℚ) (h : (2 * k) / (x - 1) - 3 / (1 - x) = 1) : k = -3 / 2 :=
sorry

end NUMINAMATH_GPT_fractional_equation_root_l1146_114695


namespace NUMINAMATH_GPT_total_cans_to_collect_l1146_114697

def cans_for_project (marthas_cans : ℕ) (additional_cans_needed : ℕ) (total_cans_needed : ℕ) : Prop :=
  ∃ diegos_cans : ℕ, diegos_cans = (marthas_cans / 2) + 10 ∧ 
  total_cans_needed = marthas_cans + diegos_cans + additional_cans_needed

theorem total_cans_to_collect : 
  cans_for_project 90 5 150 :=
by
  -- Insert proof here in actual usage
  sorry

end NUMINAMATH_GPT_total_cans_to_collect_l1146_114697


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1146_114665

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1146_114665


namespace NUMINAMATH_GPT_chess_club_members_l1146_114650

theorem chess_club_members {n : ℤ} (h10 : n % 10 = 6) (h11 : n % 11 = 6) (rng : 300 ≤ n ∧ n ≤ 400) : n = 336 :=
  sorry

end NUMINAMATH_GPT_chess_club_members_l1146_114650


namespace NUMINAMATH_GPT_missing_coins_l1146_114656

-- Definition representing the total number of coins Charlie received
variable (y : ℚ)

-- Conditions
def initial_lost_coins (y : ℚ) := (1 / 3) * y
def recovered_coins (y : ℚ) := (2 / 9) * y

-- Main Theorem
theorem missing_coins (y : ℚ) :
  y - (y * (8 / 9)) = y * (1 / 9) :=
by
  sorry

end NUMINAMATH_GPT_missing_coins_l1146_114656


namespace NUMINAMATH_GPT_find_M_l1146_114659

theorem find_M :
  (∃ M: ℕ, (10 + 11 + 12) / 3 = (2022 + 2023 + 2024) / M) → M = 551 :=
by
  sorry

end NUMINAMATH_GPT_find_M_l1146_114659


namespace NUMINAMATH_GPT_solution_one_solution_two_l1146_114675

section

variables {a x : ℝ}

def f (x : ℝ) (a : ℝ) := |2 * x - a| - |x + 1|

-- (1) Prove the solution set for f(x) > 2 when a = 1 is (-∞, -2/3) ∪ (4, ∞)
theorem solution_one (x : ℝ) : f x 1 > 2 ↔ x < -2/3 ∨ x > 4 :=
by sorry

-- (2) Prove the range of a for which f(x) + |x + 1| + x > a² - 1/2 always holds for x ∈ ℝ is (-1/2, 1)
theorem solution_two (a : ℝ) : 
  (∀ x, f x a + |x + 1| + x > a^2 - 1/2) ↔ -1/2 < a ∧ a < 1 :=
by sorry

end

end NUMINAMATH_GPT_solution_one_solution_two_l1146_114675


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_subsequence_l1146_114614

theorem arithmetic_sequence_geometric_subsequence (a : ℕ → ℕ)
  (h1 : ∀ n, a (n + 1) = a n + 1)
  (h2 : (a 3)^2 = a 1 * a 7) :
  a 5 = 6 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_subsequence_l1146_114614


namespace NUMINAMATH_GPT_integral_exp_neg_l1146_114679

theorem integral_exp_neg : ∫ x in (Set.Ioi 0), Real.exp (-x) = 1 := sorry

end NUMINAMATH_GPT_integral_exp_neg_l1146_114679


namespace NUMINAMATH_GPT_sticks_per_pot_is_181_l1146_114672

/-- Define the problem conditions -/
def number_of_pots : ℕ := 466
def flowers_per_pot : ℕ := 53
def total_flowers_and_sticks : ℕ := 109044

/-- Define the function to calculate the number of sticks per pot -/
def sticks_per_pot (S : ℕ) : Prop :=
  (number_of_pots * flowers_per_pot + number_of_pots * S = total_flowers_and_sticks)

/-- State the theorem -/
theorem sticks_per_pot_is_181 : sticks_per_pot 181 :=
by
  sorry

end NUMINAMATH_GPT_sticks_per_pot_is_181_l1146_114672


namespace NUMINAMATH_GPT_pregnant_dogs_count_l1146_114655

-- Definitions as conditions stated in the problem
def total_puppies (P : ℕ) : ℕ := 4 * P
def total_shots (P : ℕ) : ℕ := 2 * total_puppies P
def total_cost (P : ℕ) : ℕ := total_shots P * 5

-- Proof statement without proof
theorem pregnant_dogs_count : ∃ P : ℕ, total_cost P = 120 → P = 3 :=
by sorry

end NUMINAMATH_GPT_pregnant_dogs_count_l1146_114655


namespace NUMINAMATH_GPT_abs_sin_diff_le_abs_sin_sub_l1146_114646

theorem abs_sin_diff_le_abs_sin_sub (A B : ℝ) (hA : 0 ≤ A) (hA' : A ≤ π) (hB : 0 ≤ B) (hB' : B ≤ π) :
  |Real.sin A - Real.sin B| ≤ |Real.sin (A - B)| :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_abs_sin_diff_le_abs_sin_sub_l1146_114646


namespace NUMINAMATH_GPT_max_third_altitude_l1146_114669

theorem max_third_altitude (h1 h2 : ℕ) (h1_eq : h1 = 6) (h2_eq : h2 = 18) (triangle_scalene : true)
: (exists h3 : ℕ, (∀ h3_alt > h3, h3_alt > 8)) := 
sorry

end NUMINAMATH_GPT_max_third_altitude_l1146_114669


namespace NUMINAMATH_GPT_find_a_plus_b_l1146_114690

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 + b * x + 3 * a + b

theorem find_a_plus_b (a b : ℝ) (h1 : ∀ x : ℝ, f a b x = f a b (-x)) (h2 : 2 * a = 3 - a) : a + b = 1 :=
by
  unfold f at h1
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1146_114690


namespace NUMINAMATH_GPT_factorization_correct_l1146_114611

theorem factorization_correct (x : ℝ) :
  (x - 3) * (x - 1) * (x - 2) * (x + 4) + 24 = (x - 2) * (x + 3) * (x^2 + x - 8) := 
sorry

end NUMINAMATH_GPT_factorization_correct_l1146_114611


namespace NUMINAMATH_GPT_opposite_sides_line_range_a_l1146_114600

theorem opposite_sides_line_range_a (a : ℝ) :
  (3 * 2 - 2 * 1 + a) * (3 * -1 - 2 * 3 + a) < 0 → -4 < a ∧ a < 9 := by
  sorry

end NUMINAMATH_GPT_opposite_sides_line_range_a_l1146_114600


namespace NUMINAMATH_GPT_right_triangle_side_lengths_l1146_114617

theorem right_triangle_side_lengths :
  ¬ (4^2 + 5^2 = 6^2) ∧
  (12^2 + 16^2 = 20^2) ∧
  ¬ (5^2 + 10^2 = 13^2) ∧
  ¬ (8^2 + 40^2 = 41^2) := by
  sorry

end NUMINAMATH_GPT_right_triangle_side_lengths_l1146_114617


namespace NUMINAMATH_GPT_product_of_roots_l1146_114638

theorem product_of_roots (a b c : ℤ) (h_eq : a = 24 ∧ b = 60 ∧ c = -600) :
  ∀ x : ℂ, (a * x^2 + b * x + c = 0) → (x * (-b - x) = -25) := sorry

end NUMINAMATH_GPT_product_of_roots_l1146_114638


namespace NUMINAMATH_GPT_cookies_remaining_percentage_l1146_114692

theorem cookies_remaining_percentage: 
  ∀ (total initial_remaining eduardo_remaining final_remaining: ℕ),
  total = 600 → 
  initial_remaining = total - (2 * total / 5) → 
  eduardo_remaining = initial_remaining - (3 * initial_remaining / 5) → 
  final_remaining = eduardo_remaining → 
  (final_remaining * 100) / total = 24 := 
by
  intros total initial_remaining eduardo_remaining final_remaining h_total h_initial_remaining h_eduardo_remaining h_final_remaining
  sorry

end NUMINAMATH_GPT_cookies_remaining_percentage_l1146_114692


namespace NUMINAMATH_GPT_jim_ran_16_miles_in_2_hours_l1146_114609

-- Given conditions
variables (j f : ℝ) -- miles Jim ran in 2 hours, miles Frank ran in 2 hours
variables (h1 : f = 20) -- Frank ran 20 miles in 2 hours
variables (h2 : f / 2 = (j / 2) + 2) -- Frank ran 2 miles more than Jim in an hour

-- Statement to prove
theorem jim_ran_16_miles_in_2_hours (j f : ℝ) (h1 : f = 20) (h2 : f / 2 = (j / 2) + 2) : j = 16 :=
by
  sorry

end NUMINAMATH_GPT_jim_ran_16_miles_in_2_hours_l1146_114609


namespace NUMINAMATH_GPT_car_distribution_l1146_114601

theorem car_distribution :
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let total_distributed_first_three := first_supplier + second_supplier + third_supplier
  let remaining_cars := total_cars - total_distributed_first_three
  let fourth_supplier := remaining_cars / 2
  let fifth_supplier := remaining_cars / 2
  fourth_supplier = 325000 :=
by
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let total_distributed_first_three := first_supplier + second_supplier + third_supplier
  let remaining_cars := total_cars - total_distributed_first_three
  let fourth_supplier := remaining_cars / 2
  let fifth_supplier := remaining_cars / 2
  sorry

end NUMINAMATH_GPT_car_distribution_l1146_114601


namespace NUMINAMATH_GPT_total_apples_picked_l1146_114698

-- Define the number of apples picked by Benny
def applesBenny : Nat := 2

-- Define the number of apples picked by Dan
def applesDan : Nat := 9

-- The theorem we want to prove
theorem total_apples_picked : applesBenny + applesDan = 11 := 
by 
  sorry

end NUMINAMATH_GPT_total_apples_picked_l1146_114698


namespace NUMINAMATH_GPT_curve_C2_equation_l1146_114637

theorem curve_C2_equation (x y : ℝ) :
  (∀ x, y = 2 * Real.sin (2 * x + π / 3) → 
    y = 2 * Real.sin (4 * (( x - π / 6) / 2))) := 
  sorry

end NUMINAMATH_GPT_curve_C2_equation_l1146_114637


namespace NUMINAMATH_GPT_fraction_in_classroom_l1146_114616

theorem fraction_in_classroom (total_students absent_fraction canteen_students present_students class_students : ℕ) 
  (h_total : total_students = 40)
  (h_absent_fraction : absent_fraction = 1 / 10)
  (h_canteen_students : canteen_students = 9)
  (h_absent_students : absent_fraction * total_students = 4)
  (h_present_students : present_students = total_students - absent_fraction * total_students)
  (h_class_students : class_students = present_students - canteen_students) :
  class_students / present_students = 3 / 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_fraction_in_classroom_l1146_114616


namespace NUMINAMATH_GPT_range_of_a_l1146_114682

theorem range_of_a :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ -1 ≤ a ∧ a ≤ 3 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1146_114682


namespace NUMINAMATH_GPT_fuchsia_to_mauve_l1146_114635

theorem fuchsia_to_mauve (F : ℝ) :
  (5 / 8) * F + (3 * 26.67 : ℝ) = (3 / 8) * F + (5 / 8) * F →
  F = 106.68 :=
by
  intro h
  -- Step to implement the solution would go here
  sorry

end NUMINAMATH_GPT_fuchsia_to_mauve_l1146_114635


namespace NUMINAMATH_GPT_circle_tangent_line_l1146_114689

noncomputable def line_eq (x : ℝ) : ℝ := 2 * x + 1
noncomputable def circle_eq (x y b : ℝ) : ℝ := x^2 + (y - b)^2

theorem circle_tangent_line 
  (b : ℝ) 
  (tangency : ∃ b, (1 - b) / (0 - 1) = -(1 / 2)) 
  (center_point : 1^2 + (3 - b)^2 = 5 / 4) : 
  circle_eq 1 3 b = circle_eq 0 b (7/2) :=
sorry

end NUMINAMATH_GPT_circle_tangent_line_l1146_114689


namespace NUMINAMATH_GPT_pork_price_increase_l1146_114639

variable (x : ℝ)
variable (P_aug P_oct : ℝ)
variable (P_aug := 32)
variable (P_oct := 64)

theorem pork_price_increase :
  P_aug * (1 + x) ^ 2 = P_oct :=
sorry

end NUMINAMATH_GPT_pork_price_increase_l1146_114639


namespace NUMINAMATH_GPT_multiple_statements_l1146_114699

theorem multiple_statements (c d : ℤ)
  (hc4 : ∃ k : ℤ, c = 4 * k)
  (hd8 : ∃ k : ℤ, d = 8 * k) :
  (∃ k : ℤ, d = 4 * k) ∧
  (∃ k : ℤ, c + d = 4 * k) ∧
  (∃ k : ℤ, c + d = 2 * k) :=
by
  sorry

end NUMINAMATH_GPT_multiple_statements_l1146_114699


namespace NUMINAMATH_GPT_alice_steps_l1146_114696

noncomputable def num_sticks (n : ℕ) : ℕ :=
  (n + 1 : ℕ) ^ 2

theorem alice_steps (n : ℕ) (h : num_sticks n = 169) : n = 13 :=
by sorry

end NUMINAMATH_GPT_alice_steps_l1146_114696


namespace NUMINAMATH_GPT_unique_H_value_l1146_114626

theorem unique_H_value :
  ∀ (T H R E F I V S : ℕ),
    T = 8 →
    E % 2 = 1 →
    E ≠ T ∧ E ≠ H ∧ E ≠ R ∧ E ≠ F ∧ E ≠ I ∧ E ≠ V ∧ E ≠ S ∧ 
    H ≠ T ∧ H ≠ R ∧ H ≠ F ∧ H ≠ I ∧ H ≠ V ∧ H ≠ S ∧
    F ≠ T ∧ F ≠ I ∧ F ≠ V ∧ F ≠ S ∧
    I ≠ T ∧ I ≠ V ∧ I ≠ S ∧
    V ≠ T ∧ V ≠ S ∧
    S ≠ T ∧
    (8 + 8) = 10 + F ∧
    (E + E) % 10 = 6 →
    H + H = 10 + 4 →
    H = 7 := 
sorry

end NUMINAMATH_GPT_unique_H_value_l1146_114626


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1146_114624

def A : Set (ℝ × ℝ) := {p | p.snd = 3 * p.fst - 2}
def B : Set (ℝ × ℝ) := {p | p.snd = p.fst ^ 2}

theorem intersection_of_A_and_B :
  {p : ℝ × ℝ | p ∈ A ∧ p ∈ B} = {(1, 1), (2, 4)} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1146_114624


namespace NUMINAMATH_GPT_andrews_age_l1146_114632

-- Define Andrew's age
variable (a g : ℚ)

-- Problem conditions
axiom condition1 : g = 10 * a
axiom condition2 : g - (a + 2) = 57

theorem andrews_age : a = 59 / 9 := 
by
  -- Set the proof steps aside for now
  sorry

end NUMINAMATH_GPT_andrews_age_l1146_114632


namespace NUMINAMATH_GPT_correct_conclusions_l1146_114644

variable (f : ℝ → ℝ)

def condition_1 := ∀ x : ℝ, f (x + 2) = f (2 - (x + 2))
def condition_2 := ∀ x : ℝ, f (-2*x - 1) = -f (2*x + 1)

theorem correct_conclusions 
  (h1 : condition_1 f) 
  (h2 : condition_2 f) : 
  f 1 = f 3 ∧ 
  f 2 + f 4 = 0 ∧ 
  f (-1 / 2) * f (11 / 2) ≤ 0 := 
by 
  sorry

end NUMINAMATH_GPT_correct_conclusions_l1146_114644


namespace NUMINAMATH_GPT_series_sum_equality_l1146_114688

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, 12^k / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem series_sum_equality : sum_series = 1 := 
by sorry

end NUMINAMATH_GPT_series_sum_equality_l1146_114688


namespace NUMINAMATH_GPT_distance_from_origin_to_line_AB_is_sqrt6_div_3_l1146_114694

open Real

structure Point where
  x : ℝ
  y : ℝ

def ellipse (p : Point) : Prop :=
  p.x^2 / 2 + p.y^2 = 1

def left_focus : Point := ⟨-1, 0⟩

def line_through_focus (t : ℝ) (p : Point) : Prop :=
  p.x = t * p.y - 1

def origin : Point := ⟨0, 0⟩

def perpendicular (A B : Point) : Prop :=
  A.x * B.x + A.y * B.y = 0

noncomputable def distance (O : Point) (A B : Point) : ℝ :=
  let a := A.y - B.y
  let b := B.x - A.x
  let c := A.x * B.y - A.y * B.x
  abs (a * O.x + b * O.y + c) / sqrt (a^2 + b^2)

theorem distance_from_origin_to_line_AB_is_sqrt6_div_3 
  (A B : Point)
  (hA_on_ellipse : ellipse A)
  (hB_on_ellipse : ellipse B)
  (h_line_through_focus : ∃ t : ℝ, line_through_focus t A ∧ line_through_focus t B)
  (h_perpendicular : perpendicular A B) :
  distance origin A B = sqrt 6 / 3 := sorry

end NUMINAMATH_GPT_distance_from_origin_to_line_AB_is_sqrt6_div_3_l1146_114694


namespace NUMINAMATH_GPT_students_at_year_end_l1146_114681

theorem students_at_year_end (initial_students left_students new_students end_students : ℕ)
  (h_initial : initial_students = 31)
  (h_left : left_students = 5)
  (h_new : new_students = 11)
  (h_end : end_students = initial_students - left_students + new_students) :
  end_students = 37 :=
by
  sorry

end NUMINAMATH_GPT_students_at_year_end_l1146_114681


namespace NUMINAMATH_GPT_find_angle_C_l1146_114608

open Real -- Opening Real to directly use real number functions and constants

noncomputable def triangle_angles_condition (A B C: ℝ) : Prop :=
  2 * sin A + 5 * cos B = 5 ∧ 5 * sin B + 2 * cos A = 2

-- Theorem statement
theorem find_angle_C (A B C: ℝ) (h: triangle_angles_condition A B C):
  C = arcsin (1 / 5) ∨ C = 180 - arcsin (1 / 5) :=
sorry

end NUMINAMATH_GPT_find_angle_C_l1146_114608


namespace NUMINAMATH_GPT_length_of_second_platform_l1146_114678

-- Given conditions
def length_of_train : ℕ := 310
def length_of_first_platform : ℕ := 110
def time_to_cross_first_platform : ℕ := 15
def time_to_cross_second_platform : ℕ := 20

-- Calculated based on conditions
def total_distance_first_platform : ℕ :=
  length_of_train + length_of_first_platform

def speed_of_train : ℕ :=
  total_distance_first_platform / time_to_cross_first_platform

def total_distance_second_platform : ℕ :=
  speed_of_train * time_to_cross_second_platform

-- Statement to prove
theorem length_of_second_platform :
  total_distance_second_platform = length_of_train + 250 := sorry

end NUMINAMATH_GPT_length_of_second_platform_l1146_114678


namespace NUMINAMATH_GPT_sum_of_powers_of_i_l1146_114645

-- Define the imaginary unit and its property
def i : ℂ := Complex.I -- ℂ represents the complex numbers, Complex.I is the imaginary unit

-- The statement we need to prove
theorem sum_of_powers_of_i : i + i^2 + i^3 + i^4 = 0 := 
by {
  -- Lean requires the proof, but we will use sorry to skip it.
  -- Define the properties of i directly or use in-built properties
  sorry
}

end NUMINAMATH_GPT_sum_of_powers_of_i_l1146_114645


namespace NUMINAMATH_GPT_outfit_choices_l1146_114649

theorem outfit_choices (tops pants : ℕ) (TopsCount : tops = 4) (PantsCount : pants = 3) :
  tops * pants = 12 := by
  sorry

end NUMINAMATH_GPT_outfit_choices_l1146_114649


namespace NUMINAMATH_GPT_find_f2_l1146_114683

-- Define the function f and the condition it satisfies
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
def condition : Prop := ∀ x, x ≠ 1 / 3 → f x + f ((x + 1) / (1 - 3 * x)) = x

-- State the theorem to prove the value of f(2)
theorem find_f2 (h : condition f) : f 2 = 48 / 35 := 
by
  sorry

end NUMINAMATH_GPT_find_f2_l1146_114683


namespace NUMINAMATH_GPT_sum_of_g_31_values_l1146_114634

def f (x : ℝ) : ℝ := 4 * x^2 - 3
def g (y : ℝ) : ℝ := y ^ 2 - y + 2

theorem sum_of_g_31_values :
  g 31 + g 31 = 21 := sorry

end NUMINAMATH_GPT_sum_of_g_31_values_l1146_114634


namespace NUMINAMATH_GPT_value_of_m_l1146_114612

def p (m : ℝ) : Prop :=
  4 < m ∧ m < 10

def q (m : ℝ) : Prop :=
  8 < m ∧ m < 12

theorem value_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1146_114612


namespace NUMINAMATH_GPT_prob_black_yellow_l1146_114676

theorem prob_black_yellow:
  ∃ (x y : ℚ), 12 > 0 ∧
  (∃ (r b y' : ℚ), r = 1/3 ∧ b - y' = 1/6 ∧ b + y' = 2/3 ∧ r + b + y' = 1) ∧
  x = 5/12 ∧ y = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_prob_black_yellow_l1146_114676


namespace NUMINAMATH_GPT_total_problems_completed_l1146_114622

variables (p t : ℕ)
variables (hp_pos : 15 < p) (ht_pos : 0 < t)
variables (eq1 : (3 * p - 6) * (t - 3) = p * t)

theorem total_problems_completed : p * t = 120 :=
by sorry

end NUMINAMATH_GPT_total_problems_completed_l1146_114622


namespace NUMINAMATH_GPT_intersection_of_sets_l1146_114647

noncomputable def A : Set ℝ := {x | -1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 5}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_of_sets : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1146_114647


namespace NUMINAMATH_GPT_a_plus_d_eq_zero_l1146_114670

noncomputable def f (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (c * x + 2 * d)

theorem a_plus_d_eq_zero (a b c d : ℝ) (h : a * b * c * d ≠ 0) (hff : ∀ x, f a b c d (f a b c d x) = 3 * x - 4) : a + d = 0 :=
by
  sorry

end NUMINAMATH_GPT_a_plus_d_eq_zero_l1146_114670


namespace NUMINAMATH_GPT_new_average_after_increase_and_bonus_l1146_114603

theorem new_average_after_increase_and_bonus 
  (n : ℕ) (initial_avg : ℝ) (k : ℝ) (bonus : ℝ) 
  (h1: n = 37) 
  (h2: initial_avg = 73) 
  (h3: k = 1.65) 
  (h4: bonus = 15) 
  : (initial_avg * k) + bonus = 135.45 := 
sorry

end NUMINAMATH_GPT_new_average_after_increase_and_bonus_l1146_114603


namespace NUMINAMATH_GPT_mean_temperature_is_correct_l1146_114636

-- Defining the list of temperatures
def temperatures : List ℝ := [75, 74, 76, 77, 80, 81, 83, 85, 83, 85]

-- Lean statement asserting the mean temperature is 79.9
theorem mean_temperature_is_correct : temperatures.sum / (temperatures.length: ℝ) = 79.9 := 
by
  sorry

end NUMINAMATH_GPT_mean_temperature_is_correct_l1146_114636


namespace NUMINAMATH_GPT_tree_cost_calculation_l1146_114668

theorem tree_cost_calculation :
  let c := 1500 -- park circumference in meters
  let i := 30 -- interval distance in meters
  let p := 5000 -- price per tree in mill
  let n := c / i -- number of trees
  let cost := n * p -- total cost in mill
  cost = 250000 :=
by
  sorry

end NUMINAMATH_GPT_tree_cost_calculation_l1146_114668


namespace NUMINAMATH_GPT_expand_remains_same_l1146_114685

variable (m n : ℤ)

-- Define a function that represents expanding m and n by a factor of 3
def expand_by_factor_3 (m n : ℤ) : ℤ := 
  2 * (3 * m) / (3 * m - 3 * n)

-- Define the original fraction
def original_fraction (m n : ℤ) : ℤ :=
  2 * m / (m - n)

-- Theorem to prove that expanding m and n by a factor of 3 does not change the fraction
theorem expand_remains_same (m n : ℤ) : 
  expand_by_factor_3 m n = original_fraction m n := 
by sorry

end NUMINAMATH_GPT_expand_remains_same_l1146_114685


namespace NUMINAMATH_GPT_smallest_r_for_B_in_C_l1146_114666

def A : Set ℝ := {t | 0 < t ∧ t < 2 * Real.pi}

def B : Set (ℝ × ℝ) := 
  {p | ∃ t ∈ A, p.1 = Real.sin t ∧ p.2 = 2 * Real.sin t * Real.cos t}

def C (r : ℝ) : Set (ℝ × ℝ) := 
  {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

theorem smallest_r_for_B_in_C : ∃ r, (B ⊆ C r ∧ ∀ r', r' < r → ¬ (B ⊆ C r')) :=
  sorry

end NUMINAMATH_GPT_smallest_r_for_B_in_C_l1146_114666


namespace NUMINAMATH_GPT_relationship_y1_y2_l1146_114653

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem relationship_y1_y2 :
  ∀ (a b c x₀ x₁ x₂ : ℝ),
    (quadratic_function a b c 0 = 4) →
    (quadratic_function a b c 1 = 1) →
    (quadratic_function a b c 2 = 0) →
    1 < x₁ → 
    x₁ < 2 → 
    3 < x₂ → 
    x₂ < 4 → 
    (quadratic_function a b c x₁ < quadratic_function a b c x₂) :=
by 
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_l1146_114653


namespace NUMINAMATH_GPT_part_one_l1146_114620

theorem part_one (m : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = m * Real.exp x - x - 2) :
  (∀ x : ℝ, f x > 0) → m > Real.exp 1 :=
sorry

end NUMINAMATH_GPT_part_one_l1146_114620


namespace NUMINAMATH_GPT_quadratic_equation_roots_l1146_114607

theorem quadratic_equation_roots (a b c : ℝ) (h_a_nonzero : a ≠ 0) 
  (h_roots : ∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -1) : 
  a + b + c = 0 ∧ b = 0 :=
by
  -- Using Vieta's formulas and the properties given, we should show:
  -- h_roots means the sum of roots = -(b/a) = 0 → b = 0
  -- and the product of roots = (c/a) = -1/a → c = -a
  -- Substituting these into ax^2 + bx + c = 0 should give us:
  -- a + b + c = 0 → we need to show both parts to complete the proof.
  sorry

end NUMINAMATH_GPT_quadratic_equation_roots_l1146_114607


namespace NUMINAMATH_GPT_dennis_teaching_years_l1146_114662

noncomputable def years_taught (V A D E N : ℕ) := V + A + D + E + N
noncomputable def sum_of_ages := 375
noncomputable def teaching_years : Prop :=
  ∃ (A V D E N : ℕ),
    V + A + D + E + N = 225 ∧
    V = A + 9 ∧
    V = D - 15 ∧
    E = A - 3 ∧
    E = 2 * N ∧
    D = 101

theorem dennis_teaching_years : teaching_years :=
by
  sorry

end NUMINAMATH_GPT_dennis_teaching_years_l1146_114662


namespace NUMINAMATH_GPT_find_p_l1146_114663

theorem find_p
  (A B C r s p q : ℝ)
  (h1 : A ≠ 0)
  (h2 : r + s = -B / A)
  (h3 : r * s = C / A)
  (h4 : r^3 + s^3 = -p) :
  p = (B^3 - 3 * A * B * C) / A^3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_p_l1146_114663


namespace NUMINAMATH_GPT_find_m_value_l1146_114648

def vectors_parallel (a1 a2 b1 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

theorem find_m_value (m : ℝ) :
  let a := (6, 3)
  let b := (m, 2)
  vectors_parallel a.1 a.2 b.1 b.2 ↔ m = 4 :=
by
  intro H
  obtain ⟨_, _⟩ := H
  sorry

end NUMINAMATH_GPT_find_m_value_l1146_114648


namespace NUMINAMATH_GPT_probability_closer_to_center_radius6_eq_1_4_l1146_114641

noncomputable def probability_closer_to_center (radius : ℝ) (r_inner : ℝ) :=
    let area_outer := Real.pi * radius ^ 2
    let area_inner := Real.pi * r_inner ^ 2
    area_inner / area_outer

theorem probability_closer_to_center_radius6_eq_1_4 :
    probability_closer_to_center 6 3 = 1 / 4 := by
    sorry

end NUMINAMATH_GPT_probability_closer_to_center_radius6_eq_1_4_l1146_114641


namespace NUMINAMATH_GPT_not_necessarily_divisible_by_20_l1146_114651

theorem not_necessarily_divisible_by_20 (k : ℤ) (h : ∃ k : ℤ, 5 ∣ k * (k+1) * (k+2)) : ¬ ∀ k : ℤ, 20 ∣ k * (k+1) * (k+2) :=
by
  sorry

end NUMINAMATH_GPT_not_necessarily_divisible_by_20_l1146_114651


namespace NUMINAMATH_GPT_total_toothpicks_correct_l1146_114658

noncomputable def total_toothpicks_in_grid 
  (height : ℕ) (width : ℕ) (partition_interval : ℕ) : ℕ :=
  let horizontal_lines := height + 1
  let vertical_lines := width + 1
  let num_partitions := height / partition_interval
  (horizontal_lines * width) + (vertical_lines * height) + (num_partitions * width)

theorem total_toothpicks_correct :
  total_toothpicks_in_grid 25 15 5 = 850 := 
by 
  sorry

end NUMINAMATH_GPT_total_toothpicks_correct_l1146_114658


namespace NUMINAMATH_GPT_yacht_actual_cost_l1146_114652

theorem yacht_actual_cost
  (discount_percentage : ℝ)
  (amount_paid : ℝ)
  (original_cost : ℝ)
  (h1 : discount_percentage = 0.72)
  (h2 : amount_paid = 3200000)
  (h3 : amount_paid = (1 - discount_percentage) * original_cost) :
  original_cost = 11428571.43 :=
by
  sorry

end NUMINAMATH_GPT_yacht_actual_cost_l1146_114652


namespace NUMINAMATH_GPT_probability_both_selected_is_correct_l1146_114605

def prob_selection_x : ℚ := 1 / 7
def prob_selection_y : ℚ := 2 / 9
def prob_both_selected : ℚ := prob_selection_x * prob_selection_y

theorem probability_both_selected_is_correct : prob_both_selected = 2 / 63 := 
by 
  sorry

end NUMINAMATH_GPT_probability_both_selected_is_correct_l1146_114605


namespace NUMINAMATH_GPT_regina_final_earnings_l1146_114664

-- Define the number of animals Regina has
def cows := 20
def pigs := 4 * cows
def goats := pigs / 2
def chickens := 2 * cows
def rabbits := 30

-- Define sale prices for each animal
def cow_price := 800
def pig_price := 400
def goat_price := 600
def chicken_price := 50
def rabbit_price := 25

-- Define annual earnings from animal products
def cow_milk_income := 500
def rabbit_meat_income := 10

-- Define annual farm maintenance and animal feed costs
def maintenance_cost := 10000

-- Define a calculation for the final earnings
def final_earnings : ℕ :=
  let cow_income := cows * cow_price
  let pig_income := pigs * pig_price
  let goat_income := goats * goat_price
  let chicken_income := chickens * chicken_price
  let rabbit_income := rabbits * rabbit_price
  let total_animal_sale_income := cow_income + pig_income + goat_income + chicken_income + rabbit_income

  let cow_milk_earning := cows * cow_milk_income
  let rabbit_meat_earning := rabbits * rabbit_meat_income
  let total_annual_income := cow_milk_earning + rabbit_meat_earning

  let total_income := total_animal_sale_income + total_annual_income
  let final_income := total_income - maintenance_cost

  final_income

-- Prove that the final earnings is as calculated
theorem regina_final_earnings : final_earnings = 75050 := by
  sorry

end NUMINAMATH_GPT_regina_final_earnings_l1146_114664


namespace NUMINAMATH_GPT_J_of_given_values_l1146_114623

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_of_given_values : J 3 (-15) 10 = 49 / 30 := 
by 
  sorry

end NUMINAMATH_GPT_J_of_given_values_l1146_114623


namespace NUMINAMATH_GPT_total_surface_area_excluding_bases_l1146_114642

def lower_base_radius : ℝ := 8
def upper_base_radius : ℝ := 5
def frustum_height : ℝ := 6
def cylinder_section_height : ℝ := 2
def cylinder_section_radius : ℝ := 5

theorem total_surface_area_excluding_bases :
  let l := Real.sqrt (frustum_height ^ 2 + (lower_base_radius - upper_base_radius) ^ 2)
  let lateral_surface_area_frustum := π * (lower_base_radius + upper_base_radius) * l
  let lateral_surface_area_cylinder := 2 * π * cylinder_section_radius * cylinder_section_height
  lateral_surface_area_frustum + lateral_surface_area_cylinder = 39 * π * Real.sqrt 5 + 20 * π :=
by
  sorry

end NUMINAMATH_GPT_total_surface_area_excluding_bases_l1146_114642


namespace NUMINAMATH_GPT_find_integer_pairs_l1146_114627

theorem find_integer_pairs (x y : ℕ) (h : x ^ 5 = y ^ 5 + 10 * y ^ 2 + 20 * y + 1) : (x, y) = (1, 0) :=
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l1146_114627


namespace NUMINAMATH_GPT_correct_equation_l1146_114673

variable (x : ℝ) (h1 : x > 0)

def length_pipeline : ℝ := 3000
def efficiency_increase : ℝ := 0.2
def days_ahead : ℝ := 10

theorem correct_equation :
  (length_pipeline / x) - (length_pipeline / ((1 + efficiency_increase) * x)) = days_ahead :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_l1146_114673


namespace NUMINAMATH_GPT_side_length_of_square_ground_l1146_114677

theorem side_length_of_square_ground
    (radius : ℝ)
    (Q_area : ℝ)
    (pi : ℝ)
    (quarter_circle_area : Q_area = (pi * (radius^2) / 4))
    (pi_approx : pi = 3.141592653589793)
    (Q_area_val : Q_area = 15393.804002589986)
    (radius_val : radius = 140) :
    ∃ (s : ℝ), s^2 = radius^2 :=
by
  sorry -- Proof not required per the instructions

end NUMINAMATH_GPT_side_length_of_square_ground_l1146_114677


namespace NUMINAMATH_GPT_unknown_number_l1146_114686

theorem unknown_number (n : ℕ) (h1 : Nat.lcm 24 n = 168) (h2 : Nat.gcd 24 n = 4) : n = 28 :=
by
  sorry

end NUMINAMATH_GPT_unknown_number_l1146_114686


namespace NUMINAMATH_GPT_percent_flowers_are_carnations_l1146_114680

-- Define the conditions
def one_third_pink_are_roses (total_flower pink_flower pink_roses : ℕ) : Prop :=
  pink_roses = (1/3) * pink_flower

def three_fourths_red_are_carnations (total_flower red_flower red_carnations : ℕ) : Prop :=
  red_carnations = (3/4) * red_flower

def six_tenths_are_pink (total_flower pink_flower : ℕ) : Prop :=
  pink_flower = (6/10) * total_flower

-- Define the proof problem statement
theorem percent_flowers_are_carnations (total_flower pink_flower pink_roses red_flower red_carnations : ℕ) :
  one_third_pink_are_roses total_flower pink_flower pink_roses →
  three_fourths_red_are_carnations total_flower red_flower red_carnations →
  six_tenths_are_pink total_flower pink_flower →
  (red_flower = total_flower - pink_flower) →
  (pink_flower - pink_roses + red_carnations = (4/10) * total_flower) →
  ((pink_flower - pink_roses) + red_carnations) * 100 / total_flower = 40 := 
sorry

end NUMINAMATH_GPT_percent_flowers_are_carnations_l1146_114680


namespace NUMINAMATH_GPT_combined_reach_l1146_114661

theorem combined_reach (barry_reach : ℝ) (larry_height : ℝ) (shoulder_ratio : ℝ) :
  barry_reach = 5 → larry_height = 5 → shoulder_ratio = 0.80 → 
  (larry_height * shoulder_ratio + barry_reach) = 9 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_combined_reach_l1146_114661


namespace NUMINAMATH_GPT_median_of_consecutive_integers_l1146_114625

theorem median_of_consecutive_integers (n : ℕ) (S : ℤ) (h1 : n = 35) (h2 : S = 1225) : 
  n % 2 = 1 → S / n = 35 := 
sorry

end NUMINAMATH_GPT_median_of_consecutive_integers_l1146_114625


namespace NUMINAMATH_GPT_misha_problem_l1146_114631

theorem misha_problem (N : ℕ) (h : ∀ a, a ∈ {a | a > 1 → ∃ b > 0, b ∈ {b' | b' < a ∧ a % b' = 0}}) :
  (∀ t : ℕ, (t > 1) → (1 / t ^ 2) < (1 / t * (t - 1))) →
  (∃ (n : ℕ), n = 1) → (N = 1 ↔ ∃ (k : ℕ), k = N^2) :=
by
  sorry

end NUMINAMATH_GPT_misha_problem_l1146_114631


namespace NUMINAMATH_GPT_find_correct_answer_l1146_114610

theorem find_correct_answer (x : ℕ) (h : 3 * x = 135) : x / 3 = 15 :=
sorry

end NUMINAMATH_GPT_find_correct_answer_l1146_114610


namespace NUMINAMATH_GPT_commission_percentage_l1146_114693

def commission_rate (amount: ℕ) : ℚ :=
  if amount <= 500 then
    0.20 * amount
  else
    0.20 * 500 + 0.50 * (amount - 500)

theorem commission_percentage (total_sale : ℕ) (h : total_sale = 800) :
  (commission_rate total_sale) / total_sale * 100 = 31.25 :=
by
  sorry

end NUMINAMATH_GPT_commission_percentage_l1146_114693


namespace NUMINAMATH_GPT_total_sheets_folded_l1146_114604

theorem total_sheets_folded (initially_folded : ℕ) (additionally_folded : ℕ) (total_folded : ℕ) :
  initially_folded = 45 → additionally_folded = 18 → total_folded = initially_folded + additionally_folded → total_folded = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end NUMINAMATH_GPT_total_sheets_folded_l1146_114604


namespace NUMINAMATH_GPT_least_possible_xy_l1146_114621

theorem least_possible_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 48 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_xy_l1146_114621


namespace NUMINAMATH_GPT_necessarily_negative_l1146_114628

theorem necessarily_negative (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : -2 < z ∧ z < -1) : 
  y + z < 0 := 
sorry

end NUMINAMATH_GPT_necessarily_negative_l1146_114628


namespace NUMINAMATH_GPT_max_value_l1146_114615

variable (a b c d : ℝ)

theorem max_value 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) 
  (h5 : b ≠ d) (h6 : c ≠ d)
  (cond1 : a / b + b / c + c / d + d / a = 4)
  (cond2 : a * c = b * d) :
  (a / c + b / d + c / a + d / b) ≤ -12 :=
sorry

end NUMINAMATH_GPT_max_value_l1146_114615


namespace NUMINAMATH_GPT_monotonic_increasing_m_ge_neg4_l1146_114660

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ a → y > x → f y ≥ f x

def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x - 2

theorem monotonic_increasing_m_ge_neg4 (m : ℝ) :
  is_monotonic_increasing (f m) 2 → m ≥ -4 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_m_ge_neg4_l1146_114660


namespace NUMINAMATH_GPT_abs_eq_iff_mul_nonpos_l1146_114630

theorem abs_eq_iff_mul_nonpos (a b : ℝ) : |a - b| = |a| + |b| ↔ a * b ≤ 0 :=
sorry

end NUMINAMATH_GPT_abs_eq_iff_mul_nonpos_l1146_114630
