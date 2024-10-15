import Mathlib

namespace NUMINAMATH_GPT_evaluate_expression_l2217_221746

theorem evaluate_expression :
  8^(-1/3 : ℝ) + (49^(-1/2 : ℝ))^(1/2 : ℝ) = (Real.sqrt 7 + 2) / (2 * Real.sqrt 7) := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2217_221746


namespace NUMINAMATH_GPT_fractions_with_same_denominators_fractions_with_same_numerators_fractions_with_different_numerators_and_denominators_l2217_221733

theorem fractions_with_same_denominators {a b c : ℤ} (h_c : c ≠ 0) :
  (a > b → a / (c:ℚ) > b / (c:ℚ)) ∧ (a < b → a / (c:ℚ) < b / (c:ℚ)) :=
by sorry

theorem fractions_with_same_numerators {a c d : ℤ} (h_c : c ≠ 0) (h_d : d ≠ 0) :
  (c < d → a / (c:ℚ) > a / (d:ℚ)) ∧ (c > d → a / (c:ℚ) < a / (d:ℚ)) :=
by sorry

theorem fractions_with_different_numerators_and_denominators {a b c d : ℤ} (h_c : c ≠ 0) (h_d : d ≠ 0) :
  a > b ∧ c < d → a / (c:ℚ) > b / (d:ℚ) :=
by sorry

end NUMINAMATH_GPT_fractions_with_same_denominators_fractions_with_same_numerators_fractions_with_different_numerators_and_denominators_l2217_221733


namespace NUMINAMATH_GPT_A_alone_finishes_work_in_30_days_l2217_221778

noncomputable def work_rate_A (B : ℝ) : ℝ := 2 * B

noncomputable def total_work (B : ℝ) : ℝ := 60 * B

theorem A_alone_finishes_work_in_30_days (B : ℝ) : (total_work B) / (work_rate_A B) = 30 := by
  sorry

end NUMINAMATH_GPT_A_alone_finishes_work_in_30_days_l2217_221778


namespace NUMINAMATH_GPT_solve_for_x_l2217_221765

theorem solve_for_x (x : ℝ) (h : -3 * x - 12 = 8 * x + 5) : x = -17 / 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2217_221765


namespace NUMINAMATH_GPT_volume_of_snow_correct_l2217_221740

noncomputable def volume_of_snow : ℝ :=
  let sidewalk_length := 30
  let sidewalk_width := 3
  let depth := 3 / 4
  let sidewalk_volume := sidewalk_length * sidewalk_width * depth
  
  let garden_path_leg1 := 3
  let garden_path_leg2 := 4
  let garden_path_area := (garden_path_leg1 * garden_path_leg2) / 2
  let garden_path_volume := garden_path_area * depth
  
  let total_volume := sidewalk_volume + garden_path_volume
  total_volume

theorem volume_of_snow_correct : volume_of_snow = 72 := by
  sorry

end NUMINAMATH_GPT_volume_of_snow_correct_l2217_221740


namespace NUMINAMATH_GPT_largest_of_seven_consecutive_integers_l2217_221729

-- Define the main conditions as hypotheses
theorem largest_of_seven_consecutive_integers (n : ℕ) (h_sum : 7 * n + 21 = 2401) : 
  n + 6 = 346 :=
by
  -- Conditions from the problem are utilized here
  sorry

end NUMINAMATH_GPT_largest_of_seven_consecutive_integers_l2217_221729


namespace NUMINAMATH_GPT_angle_B_measure_triangle_area_l2217_221774

noncomputable def triangle (A B C : ℝ) : Type := sorry

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Given conditions:
axiom eq1 : b * Real.cos C = (2 * a - c) * Real.cos B

-- Part 1: Prove the measure of angle B
theorem angle_B_measure : B = Real.pi / 3 :=
by
  have b_cos_C := eq1
  sorry

-- Part 2: Given additional conditions and find the area
variable (b_value : ℝ := Real.sqrt 7)
variable (sum_ac : ℝ := 4)

theorem triangle_area : (1 / 2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) :=
by
  have b_value_def := b_value
  have sum_ac_def := sum_ac
  sorry

end NUMINAMATH_GPT_angle_B_measure_triangle_area_l2217_221774


namespace NUMINAMATH_GPT_linear_system_solution_l2217_221775

theorem linear_system_solution :
  ∃ (x y z : ℝ), (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧
  (x + (85/3) * y + 4 * z = 0) ∧ 
  (4 * x + (85/3) * y + z = 0) ∧ 
  (3 * x + 5 * y - 2 * z = 0) ∧ 
  (x * z) / (y ^ 2) = 25 := 
sorry

end NUMINAMATH_GPT_linear_system_solution_l2217_221775


namespace NUMINAMATH_GPT_outfit_combinations_l2217_221794

theorem outfit_combinations 
  (shirts : Fin 5)
  (pants : Fin 6)
  (restricted_shirt : Fin 1)
  (restricted_pants : Fin 2) :
  ∃ total_combinations : ℕ, total_combinations = 28 :=
sorry

end NUMINAMATH_GPT_outfit_combinations_l2217_221794


namespace NUMINAMATH_GPT_distance_to_other_asymptote_is_8_l2217_221741

-- Define the hyperbola and the properties
def hyperbola (x y : ℝ) : Prop := (x^2) / 2 - (y^2) / 8 = 1

-- Define the asymptotes
def asymptote_1 (x y : ℝ) : Prop := y = 2 * x
def asymptote_2 (x y : ℝ) : Prop := y = -2 * x

-- Given conditions
variables (P : ℝ × ℝ)
variable (distance_to_one_asymptote : ℝ)
variable (distance_to_other_asymptote : ℝ)

axiom point_on_hyperbola : hyperbola P.1 P.2
axiom distance_to_one_asymptote_is_1_over_5 : distance_to_one_asymptote = 1 / 5

-- The proof statement
theorem distance_to_other_asymptote_is_8 :
  distance_to_other_asymptote = 8 := sorry

end NUMINAMATH_GPT_distance_to_other_asymptote_is_8_l2217_221741


namespace NUMINAMATH_GPT_solve_for_x_l2217_221724

theorem solve_for_x (x y z w : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) 
(h3 : x * z + y * w = 50) (h4 : z - w = 5) : x = 20 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l2217_221724


namespace NUMINAMATH_GPT_paul_initial_books_l2217_221791

theorem paul_initial_books (sold_books : ℕ) (left_books : ℕ) (initial_books : ℕ) 
  (h_sold_books : sold_books = 109)
  (h_left_books : left_books = 27)
  (h_initial_books_formula : initial_books = sold_books + left_books) : 
  initial_books = 136 :=
by
  rw [h_sold_books, h_left_books] at h_initial_books_formula
  exact h_initial_books_formula

end NUMINAMATH_GPT_paul_initial_books_l2217_221791


namespace NUMINAMATH_GPT_find_metal_sheet_width_l2217_221723

-- The given conditions
def metalSheetLength : ℝ := 100
def cutSquareSide : ℝ := 10
def boxVolume : ℝ := 24000

-- Statement to prove
theorem find_metal_sheet_width (w : ℝ) (h : w - 2 * cutSquareSide > 0):
  boxVolume = (metalSheetLength - 2 * cutSquareSide) * (w - 2 * cutSquareSide) * cutSquareSide → 
  w = 50 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_metal_sheet_width_l2217_221723


namespace NUMINAMATH_GPT_minimum_value_condition_l2217_221757

theorem minimum_value_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) 
                                (h_line : ∀ x y : ℝ, m * x + n * y + 2 = 0 → (x + 3)^2 + (y + 1)^2 = 1) 
                                (h_chord : ∀ x1 y1 x2 y2 : ℝ, m * x1 + n * y1 + 2 = 0 ∧ (x1 + 3)^2 + (y1 + 1)^2 = 1 ∧
                                           m * x2 + n * y2 + 2 = 0 ∧ (x2 + 3)^2 + (y2 + 1)^2 = 1 ∧
                                           (x1 - x2)^2 + (y1 - y2)^2 = 4) 
                                (h_relation : 3 * m + n = 2) : 
    ∃ (C : ℝ), C = 6 ∧ (C = (1 / m + 3 / n)) := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_condition_l2217_221757


namespace NUMINAMATH_GPT_delivery_parcels_problem_l2217_221705

theorem delivery_parcels_problem (x : ℝ) (h1 : 2 + 2 * (1 + x) + 2 * (1 + x) ^ 2 = 7.28) : 
  2 + 2 * (1 + x) + 2 * (1 + x) ^ 2 = 7.28 :=
by
  exact h1

end NUMINAMATH_GPT_delivery_parcels_problem_l2217_221705


namespace NUMINAMATH_GPT_negation_of_p_l2217_221759

open Classical

variable {x : ℝ}

def p : Prop := ∃ x : ℝ, x > 1

theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l2217_221759


namespace NUMINAMATH_GPT_number_of_black_squares_in_56th_row_l2217_221755

def total_squares (n : Nat) : Nat := 3 + 2 * (n - 1)

def black_squares (n : Nat) : Nat :=
  if total_squares n % 2 == 1 then
    (total_squares n - 1) / 2
  else
    total_squares n / 2

theorem number_of_black_squares_in_56th_row :
  black_squares 56 = 56 :=
by
  sorry

end NUMINAMATH_GPT_number_of_black_squares_in_56th_row_l2217_221755


namespace NUMINAMATH_GPT_combined_boys_average_l2217_221725

noncomputable def average_boys_score (C c D d : ℕ) : ℚ :=
  (68 * C + 74 * 3 * c / 4) / (C + 3 * c / 4)

theorem combined_boys_average:
  ∀ (C c D d : ℕ),
  (68 * C + 72 * c) / (C + c) = 70 →
  (74 * D + 88 * d) / (D + d) = 82 →
  (72 * c + 88 * d) / (c + d) = 83 →
  C = c →
  4 * D = 3 * d →
  average_boys_score C c D d = 48.57 :=
by
  intros C c D d h_clinton h_dixon h_combined_girls h_C_eq_c h_D_eq_d
  sorry

end NUMINAMATH_GPT_combined_boys_average_l2217_221725


namespace NUMINAMATH_GPT_minimum_value_l2217_221707

theorem minimum_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ x : ℝ, 
    (x = 2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b) ^ 2) ∧ 
    (∀ y, y = 2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b) ^ 2 → x ≤ y) ∧ 
    x = 7 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_l2217_221707


namespace NUMINAMATH_GPT_evaluate_expression_l2217_221770

theorem evaluate_expression :
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = 137 / 52 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2217_221770


namespace NUMINAMATH_GPT_smallest_square_side_length_paintings_l2217_221750

theorem smallest_square_side_length_paintings (n : ℕ) :
  ∃ n : ℕ, (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 2020 → 1 * i ≤ n * n) → n = 1430 :=
by
  sorry

end NUMINAMATH_GPT_smallest_square_side_length_paintings_l2217_221750


namespace NUMINAMATH_GPT_ratio_of_Phil_to_Bob_l2217_221720

-- There exists real numbers P, J, and B such that
theorem ratio_of_Phil_to_Bob (P J B : ℝ) (h1 : J = 2 * P) (h2 : B = 60) (h3 : J = B - 20) : P / B = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_Phil_to_Bob_l2217_221720


namespace NUMINAMATH_GPT_find_divisor_l2217_221713

variable (n : ℤ) (d : ℤ)

theorem find_divisor 
    (h1 : ∃ k : ℤ, n = k * d + 4)
    (h2 : ∃ m : ℤ, n + 15 = m * 5 + 4) :
    d = 5 :=
sorry

end NUMINAMATH_GPT_find_divisor_l2217_221713


namespace NUMINAMATH_GPT_find_a1_l2217_221727

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

def is_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem find_a1 (h1 : is_arithmetic_seq a (-2)) 
               (h2 : sum_n_terms S a) 
               (h3 : S 10 = S 11) : 
  a 1 = 20 :=
sorry

end NUMINAMATH_GPT_find_a1_l2217_221727


namespace NUMINAMATH_GPT_y_divides_x_squared_l2217_221752

-- Define the conditions and proof problem in Lean 4
theorem y_divides_x_squared (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : ∃ (n : ℕ), n = (x^2 / y) + (y^2 / x)) : y ∣ x^2 :=
by {
  -- Proof steps are skipped
  sorry
}

end NUMINAMATH_GPT_y_divides_x_squared_l2217_221752


namespace NUMINAMATH_GPT_percentage_change_difference_l2217_221706

-- Define the initial and final percentages of students
def initial_liked_percentage : ℝ := 0.4
def initial_disliked_percentage : ℝ := 0.6
def final_liked_percentage : ℝ := 0.8
def final_disliked_percentage : ℝ := 0.2

-- Define the problem statement
theorem percentage_change_difference :
  (final_liked_percentage - initial_liked_percentage) + 
  (initial_disliked_percentage - final_disliked_percentage) = 0.6 :=
sorry

end NUMINAMATH_GPT_percentage_change_difference_l2217_221706


namespace NUMINAMATH_GPT_mean_of_six_numbers_sum_three_quarters_l2217_221764

theorem mean_of_six_numbers_sum_three_quarters :
  let sum := (3 / 4 : ℝ)
  let n := 6
  (sum / n) = (1 / 8 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_mean_of_six_numbers_sum_three_quarters_l2217_221764


namespace NUMINAMATH_GPT_rational_numbers_countable_l2217_221738

theorem rational_numbers_countable : ∃ (f : ℚ → ℕ), Function.Bijective f :=
by
  sorry

end NUMINAMATH_GPT_rational_numbers_countable_l2217_221738


namespace NUMINAMATH_GPT_simplify_T_l2217_221787

noncomputable def T (x : ℝ) : ℝ :=
  (x+1)^4 - 4*(x+1)^3 + 6*(x+1)^2 - 4*(x+1) + 1

theorem simplify_T (x : ℝ) : T x = x^4 :=
  sorry

end NUMINAMATH_GPT_simplify_T_l2217_221787


namespace NUMINAMATH_GPT_gcd_problem_l2217_221762

theorem gcd_problem : ∃ b : ℕ, gcd (20 * b) (18 * 24) = 2 :=
by { sorry }

end NUMINAMATH_GPT_gcd_problem_l2217_221762


namespace NUMINAMATH_GPT_sally_remaining_cards_l2217_221777

variable (total_cards : ℕ) (torn_cards : ℕ) (bought_cards : ℕ)

def intact_cards (total_cards : ℕ) (torn_cards : ℕ) : ℕ := total_cards - torn_cards
def remaining_cards (intact_cards : ℕ) (bought_cards : ℕ) : ℕ := intact_cards - bought_cards

theorem sally_remaining_cards :
  intact_cards 39 9 - 24 = 6 :=
by
  -- sorry for proof
  sorry

end NUMINAMATH_GPT_sally_remaining_cards_l2217_221777


namespace NUMINAMATH_GPT_f_2007_eq_0_l2217_221789

-- Define even function and odd function properties
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define functions f and g
variables (f g : ℝ → ℝ)

-- Assume the given conditions
axiom even_f : is_even f
axiom odd_g : is_odd g
axiom g_def : ∀ x, g x = f (x - 1)

-- Prove that f(2007) = 0
theorem f_2007_eq_0 : f 2007 = 0 :=
sorry

end NUMINAMATH_GPT_f_2007_eq_0_l2217_221789


namespace NUMINAMATH_GPT_arithmetic_mean_reciprocals_primes_l2217_221722

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_reciprocals_primes_l2217_221722


namespace NUMINAMATH_GPT_peanuts_remaining_l2217_221795

theorem peanuts_remaining (initial_peanuts brock_ate bonita_ate brock_fraction : ℕ) (h_initial : initial_peanuts = 148) (h_brock_fraction : brock_fraction = 4) (h_brock_ate : brock_ate = initial_peanuts / brock_fraction) (h_bonita_ate : bonita_ate = 29) :
  (initial_peanuts - brock_ate - bonita_ate) = 82 :=
by
  sorry

end NUMINAMATH_GPT_peanuts_remaining_l2217_221795


namespace NUMINAMATH_GPT_log_expression_value_l2217_221769

noncomputable def log_expression : ℝ :=
  (Real.log (Real.sqrt 27) + Real.log 8 - 3 * Real.log (Real.sqrt 10)) / Real.log 1.2

theorem log_expression_value : log_expression = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_log_expression_value_l2217_221769


namespace NUMINAMATH_GPT_martha_no_daughters_count_l2217_221734

-- Definitions based on conditions
def total_people : ℕ := 40
def martha_daughters : ℕ := 8
def granddaughters_per_child (x : ℕ) : ℕ := if x = 1 then 8 else 0

-- Statement of the problem
theorem martha_no_daughters_count : 
  (total_people - martha_daughters) +
  (martha_daughters - (total_people - martha_daughters) / 8) = 36 := 
  by
    sorry

end NUMINAMATH_GPT_martha_no_daughters_count_l2217_221734


namespace NUMINAMATH_GPT_students_who_like_both_l2217_221776

def total_students : ℕ := 50
def apple_pie_lovers : ℕ := 22
def chocolate_cake_lovers : ℕ := 20
def neither_dessert_lovers : ℕ := 15

theorem students_who_like_both : 
  (apple_pie_lovers + chocolate_cake_lovers) - (total_students - neither_dessert_lovers) = 7 :=
by
  -- Calculation steps (skipped)
  sorry

end NUMINAMATH_GPT_students_who_like_both_l2217_221776


namespace NUMINAMATH_GPT_product_of_two_numbers_l2217_221782

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 26) (h2 : x - y = 8) : x * y = 153 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2217_221782


namespace NUMINAMATH_GPT_parabola_line_intersection_sum_l2217_221708

theorem parabola_line_intersection_sum (r s : ℝ) (h_r : r = 20 - 10 * Real.sqrt 38) (h_s : s = 20 + 10 * Real.sqrt 38) :
  r + s = 40 := by
  sorry

end NUMINAMATH_GPT_parabola_line_intersection_sum_l2217_221708


namespace NUMINAMATH_GPT_cost_of_ingredients_l2217_221701

theorem cost_of_ingredients :
  let popcorn_earnings := 50
  let cotton_candy_earnings := 3 * popcorn_earnings
  let total_earnings_per_day := popcorn_earnings + cotton_candy_earnings
  let total_earnings := total_earnings_per_day * 5
  let rent := 30
  let earnings_after_rent := total_earnings - rent
  earnings_after_rent - 895 = 75 :=
by
  let popcorn_earnings := 50
  let cotton_candy_earnings := 3 * popcorn_earnings
  let total_earnings_per_day := popcorn_earnings + cotton_candy_earnings
  let total_earnings := total_earnings_per_day * 5
  let rent := 30
  let earnings_after_rent := total_earnings - rent
  show earnings_after_rent - 895 = 75
  sorry

end NUMINAMATH_GPT_cost_of_ingredients_l2217_221701


namespace NUMINAMATH_GPT_houses_after_boom_l2217_221739

theorem houses_after_boom (h_pre_boom : ℕ) (h_built : ℕ) (h_count : ℕ)
  (H1 : h_pre_boom = 1426)
  (H2 : h_built = 574)
  (H3 : h_count = h_pre_boom + h_built) :
  h_count = 2000 :=
by {
  sorry
}

end NUMINAMATH_GPT_houses_after_boom_l2217_221739


namespace NUMINAMATH_GPT_unique_element_in_set_l2217_221767

theorem unique_element_in_set (A : Set ℝ) (h₁ : ∃ x, A = {x})
(h₂ : ∀ x ∈ A, (x + 3) / (x - 1) ∈ A) : ∃ x, x ∈ A ∧ (x = 3 ∨ x = -1) := by
  sorry

end NUMINAMATH_GPT_unique_element_in_set_l2217_221767


namespace NUMINAMATH_GPT_friend_P_distance_l2217_221798

theorem friend_P_distance (v t : ℝ) (hv : v > 0)
  (distance_trail : 22 = (1.20 * v * t) + (v * t))
  (h_t : t = 22 / (2.20 * v)) : 
  (1.20 * v * t = 12) :=
by
  sorry

end NUMINAMATH_GPT_friend_P_distance_l2217_221798


namespace NUMINAMATH_GPT_student_total_marks_l2217_221771

variable (M P C : ℕ)

theorem student_total_marks :
  C = P + 20 ∧ (M + C) / 2 = 25 → M + P = 30 :=
by
  sorry

end NUMINAMATH_GPT_student_total_marks_l2217_221771


namespace NUMINAMATH_GPT_line_intersects_circle_l2217_221719

theorem line_intersects_circle (m : ℝ) : 
  ∃ (x y : ℝ), y = m * x - 3 ∧ x^2 + (y - 1)^2 = 25 :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_l2217_221719


namespace NUMINAMATH_GPT_distance_between_stations_l2217_221772

theorem distance_between_stations
  (v₁ v₂ : ℝ)
  (D₁ D₂ : ℝ)
  (T : ℝ)
  (h₁ : v₁ = 20)
  (h₂ : v₂ = 25)
  (h₃ : D₂ = D₁ + 70)
  (h₄ : D₁ = v₁ * T)
  (h₅ : D₂ = v₂ * T) : 
  D₁ + D₂ = 630 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_stations_l2217_221772


namespace NUMINAMATH_GPT_probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18_l2217_221717

open Real

noncomputable def probability_event : ℝ :=
  ((327.61 - 324) / (361 - 324))

theorem probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18 :
  probability_event = 361 / 3700 :=
by
  -- Conditions and calculations supplied in the problem
  sorry

end NUMINAMATH_GPT_probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18_l2217_221717


namespace NUMINAMATH_GPT_part1_part2_l2217_221747

-- Part 1: Define the sequence and sum function, then state the problem.
def a_1 : ℚ := 3 / 2
def d : ℚ := 1

def S_n (n : ℕ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem part1 (k : ℕ) (h : S_n (k^2) = (S_n k)^2) : k = 4 := sorry

-- Part 2: Define the general sequence and state the problem.
def arith_seq (a_1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a_1 + (n - 1) * d

def S_n_general (a_1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n * a_1) + (n * (n - 1) / 2) * d

theorem part2 (a_1 : ℚ) (d : ℚ) :
  (∀ k : ℕ, S_n_general a_1 d (k^2) = (S_n_general a_1 d k)^2) ↔
  (a_1 = 0 ∧ d = 0) ∨
  (a_1 = 1 ∧ d = 0) ∨
  (a_1 = 1 ∧ d = 2) := sorry

end NUMINAMATH_GPT_part1_part2_l2217_221747


namespace NUMINAMATH_GPT_total_height_of_pipes_l2217_221796

theorem total_height_of_pipes 
  (diameter : ℝ) (radius : ℝ) (total_pipes : ℕ) (first_row_pipes : ℕ) (second_row_pipes : ℕ) 
  (h : ℝ) 
  (h_diam : diameter = 10)
  (h_radius : radius = 5)
  (h_total_pipes : total_pipes = 5)
  (h_first_row : first_row_pipes = 2)
  (h_second_row : second_row_pipes = 3) :
  h = 10 + 5 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_total_height_of_pipes_l2217_221796


namespace NUMINAMATH_GPT_octagon_diagonal_ratio_l2217_221710

theorem octagon_diagonal_ratio (P : ℝ → ℝ → Prop) (d1 d2 : ℝ) (h1 : P d1 d2) : d1 / d2 = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_octagon_diagonal_ratio_l2217_221710


namespace NUMINAMATH_GPT_average_words_per_hour_l2217_221743

theorem average_words_per_hour
  (total_words : ℕ := 60000)
  (total_hours : ℕ := 150)
  (first_period_hours : ℕ := 50)
  (first_period_words : ℕ := total_words / 2) :
  first_period_words / first_period_hours = 600 ∧ total_words / total_hours = 400 := 
by
  sorry

end NUMINAMATH_GPT_average_words_per_hour_l2217_221743


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2217_221730

variable (a b : ℚ)

theorem simplify_and_evaluate_expression
  (ha : a = 1 / 2)
  (hb : b = -1 / 3) :
  b^2 - a^2 + 2 * (a^2 + a * b) - (a^2 + b^2) = -1 / 3 :=
by
  -- The proof will be inserted here
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2217_221730


namespace NUMINAMATH_GPT_star_example_l2217_221766

def star (x y : ℝ) : ℝ := 2 * x * y - 3 * x + y

theorem star_example : (star 6 4) - (star 4 6) = -8 := by
  sorry

end NUMINAMATH_GPT_star_example_l2217_221766


namespace NUMINAMATH_GPT_shorts_more_than_checkered_l2217_221781

noncomputable def total_students : ℕ := 81

noncomputable def striped_shirts : ℕ := (2 * total_students) / 3

noncomputable def checkered_shirts : ℕ := total_students - striped_shirts

noncomputable def shorts : ℕ := striped_shirts - 8

theorem shorts_more_than_checkered :
  shorts - checkered_shirts = 19 :=
by
  sorry

end NUMINAMATH_GPT_shorts_more_than_checkered_l2217_221781


namespace NUMINAMATH_GPT_shirt_cost_is_ten_l2217_221792

theorem shirt_cost_is_ten (S J : ℝ) (h1 : J = 2 * S) 
    (h2 : 20 * S + 10 * J = 400) : S = 10 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_shirt_cost_is_ten_l2217_221792


namespace NUMINAMATH_GPT_goldie_total_earnings_l2217_221712

-- Define weekly earnings based on hours and rates
def earnings_first_week (hours_dog_walking hours_medication : ℕ) : ℕ :=
  (hours_dog_walking * 5) + (hours_medication * 8)

def earnings_second_week (hours_feeding hours_cleaning hours_playing : ℕ) : ℕ :=
  (hours_feeding * 6) + (hours_cleaning * 4) + (hours_playing * 3)

-- Given conditions for hours worked each task in two weeks
def hours_dog_walking : ℕ := 12
def hours_medication : ℕ := 8
def hours_feeding : ℕ := 10
def hours_cleaning : ℕ := 15
def hours_playing : ℕ := 5

-- Proof statement: Total earnings over two weeks equals $259
theorem goldie_total_earnings : 
  (earnings_first_week hours_dog_walking hours_medication) + 
  (earnings_second_week hours_feeding hours_cleaning hours_playing) = 259 :=
by
  sorry

end NUMINAMATH_GPT_goldie_total_earnings_l2217_221712


namespace NUMINAMATH_GPT_am_gm_inequality_l2217_221709

theorem am_gm_inequality (a b c : ℝ) (h : a * b * c = 1 / 8) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 :=
sorry

end NUMINAMATH_GPT_am_gm_inequality_l2217_221709


namespace NUMINAMATH_GPT_sum_eq_sum_l2217_221760

theorem sum_eq_sum {a b c d : ℝ} (h1 : a + b = c + d) (h2 : ac = bd) (h3 : a + b ≠ 0) : a + c = b + d := 
by
  sorry

end NUMINAMATH_GPT_sum_eq_sum_l2217_221760


namespace NUMINAMATH_GPT_square_area_problem_l2217_221736

theorem square_area_problem
    (x1 y1 x2 y2 : ℝ)
    (h1 : y1 = x1^2)
    (h2 : y2 = x2^2)
    (line_eq : ∃ a : ℝ, a = 2 ∧ ∃ b : ℝ, b = -22 ∧ ∀ x y : ℝ, y = 2 * x - 22 → (y = y1 ∨ y = y2)) :
    ∃ area : ℝ, area = 180 ∨ area = 980 :=
sorry

end NUMINAMATH_GPT_square_area_problem_l2217_221736


namespace NUMINAMATH_GPT_value_of_f_at_2_l2217_221783

theorem value_of_f_at_2 (a b : ℝ) (h : (a + -b + 8) = (9 * a + 3 * b + 8)) :
  (a * 2 ^ 2 + b * 2 + 8) = 8 := 
by
  sorry

end NUMINAMATH_GPT_value_of_f_at_2_l2217_221783


namespace NUMINAMATH_GPT_sixth_term_geometric_mean_l2217_221704

variable (a d : ℝ)

-- Define the arithmetic progression terms
def a_n (n : ℕ) := a + (n - 1) * d

-- Provided condition: second term is the geometric mean of the 1st and 4th terms
def condition (a d : ℝ) := a_n a d 2 = Real.sqrt (a_n a d 1 * a_n a d 4)

-- The goal to be proved: sixth term is the geometric mean of the 4th and 9th terms
theorem sixth_term_geometric_mean (a d : ℝ) (h : condition a d) : 
  a_n a d 6 = Real.sqrt (a_n a d 4 * a_n a d 9) :=
sorry

end NUMINAMATH_GPT_sixth_term_geometric_mean_l2217_221704


namespace NUMINAMATH_GPT_card_arrangement_probability_l2217_221773

/-- 
This problem considers the probability of arranging four distinct cards,
each labeled with a unique character, in such a way that they form one of two specific
sequences. Specifically, the sequences are "我爱数学" (I love mathematics) and "数学爱我" (mathematics loves me).
-/
theorem card_arrangement_probability :
  let cards := ["我", "爱", "数", "学"]
  let total_permutations := 24
  let favorable_outcomes := 2
  let probability := favorable_outcomes / total_permutations
  probability = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_card_arrangement_probability_l2217_221773


namespace NUMINAMATH_GPT_retail_price_before_discounts_l2217_221763

theorem retail_price_before_discounts 
  (wholesale_price profit_rate tax_rate discount1 discount2 total_effective_price : ℝ) 
  (h_wholesale_price : wholesale_price = 108)
  (h_profit_rate : profit_rate = 0.20)
  (h_tax_rate : tax_rate = 0.15)
  (h_discount1 : discount1 = 0.10)
  (h_discount2 : discount2 = 0.05)
  (h_total_effective_price : total_effective_price = 126.36) :
  ∃ (retail_price_before_discounts : ℝ), retail_price_before_discounts = 147.78 := 
by
  sorry

end NUMINAMATH_GPT_retail_price_before_discounts_l2217_221763


namespace NUMINAMATH_GPT_card_statements_has_four_true_l2217_221718

noncomputable def statement1 (S : Fin 5 → Bool) : Prop := S 0 = true -> (S 1 = false ∧ S 2 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement2 (S : Fin 5 → Bool) : Prop := S 1 = true -> (S 0 = false ∧ S 2 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement3 (S : Fin 5 → Bool) : Prop := S 2 = true -> (S 0 = false ∧ S 1 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement4 (S : Fin 5 → Bool) : Prop := S 3 = true -> (S 0 = false ∧ S 1 = false ∧ S 2 = false ∧ S 4 = false)
noncomputable def statement5 (S : Fin 5 → Bool) : Prop := S 4 = true -> (S 0 = false ∧ S 1 = false ∧ S 2 = false ∧ S 3 = false)

theorem card_statements_has_four_true : ∃ (S : Fin 5 → Bool), 
  (statement1 S ∧ statement2 S ∧ statement3 S ∧ statement4 S ∧ statement5 S ∧ 
  ((S 0 = true ∨ S 1 = true ∨ S 2 = true ∨ S 3 = true ∨ S 4 = true) ∧ 
  4 = (if S 0 then 1 else 0) + (if S 1 then 1 else 0) + 
      (if S 2 then 1 else 0) + (if S 3 then 1 else 0) + 
      (if S 4 then 1 else 0))) :=
sorry

end NUMINAMATH_GPT_card_statements_has_four_true_l2217_221718


namespace NUMINAMATH_GPT_birdseed_mix_percentage_l2217_221785

theorem birdseed_mix_percentage (x : ℝ) :
  (0.40 * x + 0.65 * (100 - x) = 50) → x = 60 :=
by
  sorry

end NUMINAMATH_GPT_birdseed_mix_percentage_l2217_221785


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l2217_221742

theorem hcf_of_two_numbers (A B : ℕ) (h1 : A * B = 4107) (h2 : A = 111) : (Nat.gcd A B) = 37 :=
by
  -- Given conditions
  have h3 : B = 37 := by
    -- Deduce B from given conditions
    sorry
  -- Prove hcf (gcd) is 37
  sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_l2217_221742


namespace NUMINAMATH_GPT_parallel_line_slope_l2217_221784

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 9) : ∃ (m : ℝ), m = 1 / 2 := 
sorry

end NUMINAMATH_GPT_parallel_line_slope_l2217_221784


namespace NUMINAMATH_GPT_min_value_of_expression_l2217_221716

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) : 
  36 ≤ (1/x + 4/y + 9/z) :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l2217_221716


namespace NUMINAMATH_GPT_jake_bitcoins_l2217_221711

theorem jake_bitcoins (initial : ℕ) (donation1 : ℕ) (fraction : ℕ) (multiplier : ℕ) (donation2 : ℕ) :
  initial = 80 →
  donation1 = 20 →
  fraction = 2 →
  multiplier = 3 →
  donation2 = 10 →
  (initial - donation1) / fraction * multiplier - donation2 = 80 :=
by
  sorry

end NUMINAMATH_GPT_jake_bitcoins_l2217_221711


namespace NUMINAMATH_GPT_coefficient_of_q_l2217_221715

theorem coefficient_of_q (q' : ℤ → ℤ) (h : ∀ q, q' q = 3 * q - 3) (h₁ : q' (q' 4) = 72) : 
  ∀ q, q' q = 3 * q - 3 :=
  sorry

end NUMINAMATH_GPT_coefficient_of_q_l2217_221715


namespace NUMINAMATH_GPT_exists_digit_sum_divisible_by_27_not_number_l2217_221732

-- Definitions
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

-- Theorem statement
theorem exists_digit_sum_divisible_by_27_not_number (n : ℕ) :
  divisible_by (sum_of_digits n) 27 ∧ ¬ divisible_by n 27 :=
  sorry

end NUMINAMATH_GPT_exists_digit_sum_divisible_by_27_not_number_l2217_221732


namespace NUMINAMATH_GPT_min_value_a_l2217_221731

theorem min_value_a (a b c : ℤ) (α β : ℝ)
  (h_a_pos : a > 0) 
  (h_eq : ∀ x : ℝ, a * x^2 + b * x + c = 0 → (x = α ∨ x = β))
  (h_alpha_beta_order : 0 < α ∧ α < β ∧ β < 1) :
  a ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_a_l2217_221731


namespace NUMINAMATH_GPT_inscribed_pentagon_angles_sum_l2217_221797

theorem inscribed_pentagon_angles_sum (α β γ δ ε : ℝ) (h1 : α + β + γ + δ + ε = 360) 
(h2 : α / 2 + β / 2 + γ / 2 + δ / 2 + ε / 2 = 180) : 
(α / 2) + (β / 2) + (γ / 2) + (δ / 2) + (ε / 2) = 180 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_pentagon_angles_sum_l2217_221797


namespace NUMINAMATH_GPT_factorize_polynomial_l2217_221721

variable (a x y : ℝ)

theorem factorize_polynomial (a x y : ℝ) :
  3 * a * x ^ 2 - 3 * a * y ^ 2 = 3 * a * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l2217_221721


namespace NUMINAMATH_GPT_average_difference_l2217_221703

theorem average_difference :
  let avg1 := (200 + 400) / 2
  let avg2 := (100 + 200) / 2
  avg1 - avg2 = 150 :=
by
  sorry

end NUMINAMATH_GPT_average_difference_l2217_221703


namespace NUMINAMATH_GPT_find_a_from_log_condition_l2217_221779

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_a_from_log_condition (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1)
  (h₂ : f a 9 = 2) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_from_log_condition_l2217_221779


namespace NUMINAMATH_GPT_area_of_trapezium_l2217_221788

/-- Two parallel sides of a trapezium are 4 cm and 5 cm respectively. 
    The perpendicular distance between the parallel sides is 6 cm.
    Prove that the area of the trapezium is 27 cm². -/
theorem area_of_trapezium (a b h : ℝ) (ha : a = 4) (hb : b = 5) (hh : h = 6) : 
  (1/2) * (a + b) * h = 27 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_trapezium_l2217_221788


namespace NUMINAMATH_GPT_least_whole_number_l2217_221751

theorem least_whole_number (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 1)
  (h4 : n % 5 = 1)
  (h5 : n % 6 = 1)
  (h6 : 7 ∣ n) : 
  n = 301 := 
sorry

end NUMINAMATH_GPT_least_whole_number_l2217_221751


namespace NUMINAMATH_GPT_prove_inequalities_l2217_221749

theorem prove_inequalities (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^3 * b > a * b^3 ∧ a - b / a > b - a / b :=
by
  sorry

end NUMINAMATH_GPT_prove_inequalities_l2217_221749


namespace NUMINAMATH_GPT_b_is_arithmetic_sequence_a_general_formula_l2217_221744

open Nat

-- Define the sequence a_n
def a : ℕ → ℤ
| 0     => 1
| 1     => 2
| (n+2) => 2 * (a (n+1)) - (a n) + 2

-- Define the sequence b_n
def b (n : ℕ) : ℤ := a (n+1) - a n

-- Part 1: The sequence b_n is an arithmetic sequence
theorem b_is_arithmetic_sequence : ∀ n : ℕ, b (n+1) - b n = 2 := by
  sorry

-- Part 2: Find the general formula for a_n
theorem a_general_formula : ∀ n : ℕ, a (n+1) = n^2 + 1 := by
  sorry

end NUMINAMATH_GPT_b_is_arithmetic_sequence_a_general_formula_l2217_221744


namespace NUMINAMATH_GPT_actual_price_of_good_l2217_221714

variables (P : Real)

theorem actual_price_of_good:
  (∀ (P : ℝ), 0.5450625 * P = 6500 → P = 6500 / 0.5450625) :=
  by sorry

end NUMINAMATH_GPT_actual_price_of_good_l2217_221714


namespace NUMINAMATH_GPT_remainder_division_l2217_221702

theorem remainder_division (x : ℤ) (hx : x % 82 = 5) : (x + 7) % 41 = 12 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_division_l2217_221702


namespace NUMINAMATH_GPT_math_problem_l2217_221799

theorem math_problem 
  (f : ℝ → ℝ)
  (phi : ℝ)
  (h_def : ∀ x, f x = 2 * Real.sin (2 * x + phi) + 1)
  (h_point : f 0 = 0)
  (h_phi_range : -Real.pi / 2 < phi ∧ phi < 0) : 
  (phi = -Real.pi / 6) ∧ (∃ k : ℤ, ∀ x, f x = 3 ↔ x = k * Real.pi + 2 * Real.pi / 3) :=
sorry

end NUMINAMATH_GPT_math_problem_l2217_221799


namespace NUMINAMATH_GPT_dihedral_angle_ge_l2217_221726

-- Define the problem conditions and goal in Lean
theorem dihedral_angle_ge (n : ℕ) (h : 3 ≤ n) (ϕ : ℝ) :
  ϕ ≥ π * (1 - 2 / n) := 
sorry

end NUMINAMATH_GPT_dihedral_angle_ge_l2217_221726


namespace NUMINAMATH_GPT_train_speed_is_25_kmph_l2217_221761

noncomputable def train_speed_kmph (train_length_m : ℕ) (man_speed_kmph : ℕ) (cross_time_s : ℕ) : ℕ :=
  let man_speed_mps := (man_speed_kmph * 1000) / 3600
  let relative_speed_mps := train_length_m / cross_time_s
  let train_speed_mps := relative_speed_mps - man_speed_mps
  let train_speed_kmph := (train_speed_mps * 3600) / 1000
  train_speed_kmph

theorem train_speed_is_25_kmph : train_speed_kmph 270 2 36 = 25 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_25_kmph_l2217_221761


namespace NUMINAMATH_GPT_other_continent_passengers_l2217_221756

noncomputable def totalPassengers := 240
noncomputable def northAmericaFraction := (1 / 3 : ℝ)
noncomputable def europeFraction := (1 / 8 : ℝ)
noncomputable def africaFraction := (1 / 5 : ℝ)
noncomputable def asiaFraction := (1 / 6 : ℝ)

theorem other_continent_passengers :
  (totalPassengers : ℝ) - (totalPassengers * northAmericaFraction +
                           totalPassengers * europeFraction +
                           totalPassengers * africaFraction +
                           totalPassengers * asiaFraction) = 42 :=
by
  sorry

end NUMINAMATH_GPT_other_continent_passengers_l2217_221756


namespace NUMINAMATH_GPT_original_number_l2217_221735

theorem original_number (n : ℕ) (h : (2 * (n + 2) - 2) / 2 = 7) : n = 6 := by
  sorry

end NUMINAMATH_GPT_original_number_l2217_221735


namespace NUMINAMATH_GPT_sum_mod_30_l2217_221758

theorem sum_mod_30 (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 7) 
  (h3 : c % 30 = 18) : 
  (a + 2 * b + c) % 30 = 17 := 
by
  sorry

end NUMINAMATH_GPT_sum_mod_30_l2217_221758


namespace NUMINAMATH_GPT_patanjali_distance_first_day_l2217_221728

theorem patanjali_distance_first_day
  (h : ℕ)
  (H1 : 3 * h + 4 * (h - 1) + 4 * h = 62) :
  3 * h = 18 :=
by
  sorry

end NUMINAMATH_GPT_patanjali_distance_first_day_l2217_221728


namespace NUMINAMATH_GPT_fresh_fruit_sold_l2217_221786

-- Define the conditions
def total_fruit_sold : ℕ := 9792
def frozen_fruit_sold : ℕ := 3513

-- Define what we need to prove
theorem fresh_fruit_sold : (total_fruit_sold - frozen_fruit_sold = 6279) := by
  sorry

end NUMINAMATH_GPT_fresh_fruit_sold_l2217_221786


namespace NUMINAMATH_GPT_lower_limit_b_l2217_221754

theorem lower_limit_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : b < 29) 
  (h4 : ∃ min_b max_b, min_b = 4 ∧ max_b ≤ 29 ∧ 3.75 = (16 : ℚ) / (min_b : ℚ) - (7 : ℚ) / (max_b : ℚ)) : 
  b ≥ 4 :=
sorry

end NUMINAMATH_GPT_lower_limit_b_l2217_221754


namespace NUMINAMATH_GPT_unique_sequence_exists_and_bounded_l2217_221745

theorem unique_sequence_exists_and_bounded (a : ℝ) (n : ℕ) :
  ∃! (x : ℕ → ℝ), -- There exists a unique sequence x : ℕ → ℝ
    (x 1 = x (n - 1)) ∧ -- x_1 = x_{n-1}
    (∀ i, 1 ≤ i ∧ i ≤ n → (1 / 2) * (x (i - 1) + x i) = x i + x i ^ 3 - a ^ 3) ∧ -- Condition for all 1 ≤ i ≤ n
    (∀ i, 0 ≤ i ∧ i ≤ n + 1 → |x i| ≤ |a|) -- Bounding condition for all 0 ≤ i ≤ n + 1
:= sorry

end NUMINAMATH_GPT_unique_sequence_exists_and_bounded_l2217_221745


namespace NUMINAMATH_GPT_sum_of_powers_eq_zero_l2217_221753

theorem sum_of_powers_eq_zero
  (a b c : ℝ)
  (n : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = 0) :
  a^(2* ⌊n⌋ + 1) + b^(2* ⌊n⌋ + 1) + c^(2* ⌊n⌋ + 1) = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_eq_zero_l2217_221753


namespace NUMINAMATH_GPT_marys_score_l2217_221700

theorem marys_score (C ω S : ℕ) (H1 : S = 30 + 4 * C - ω) (H2 : S > 80)
  (H3 : (∀ C1 ω1 C2 ω2, (C1 ≠ C2 → 30 + 4 * C1 - ω1 ≠ 30 + 4 * C2 - ω2))) : 
  S = 119 :=
sorry

end NUMINAMATH_GPT_marys_score_l2217_221700


namespace NUMINAMATH_GPT_B_Bons_wins_probability_l2217_221748

theorem B_Bons_wins_probability :
  let roll_six := (1 : ℚ) / 6
  let not_roll_six := (5 : ℚ) / 6
  let p := (5 : ℚ) / 11
  p = (5 / 36) + (25 / 36) * p :=
by
  sorry

end NUMINAMATH_GPT_B_Bons_wins_probability_l2217_221748


namespace NUMINAMATH_GPT_increasing_interval_l2217_221768

noncomputable def function_y (x : ℝ) : ℝ :=
  2 * (Real.logb (1/2) x) ^ 2 - 2 * Real.logb (1/2) x + 1

theorem increasing_interval :
  ∀ x : ℝ, x > 0 → (∀ {y}, y ≥ x → function_y y ≥ function_y x) ↔ x ∈ Set.Ici (Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_increasing_interval_l2217_221768


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2217_221793

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 3 * x - 18 < 0 ↔ -6 < x ∧ x < 3 := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2217_221793


namespace NUMINAMATH_GPT_final_price_jacket_l2217_221780

-- Defining the conditions as per the problem
def original_price : ℚ := 250
def first_discount_rate : ℚ := 0.40
def second_discount_rate : ℚ := 0.15
def tax_rate : ℚ := 0.05

-- Defining the calculation steps
def first_discounted_price : ℚ := original_price * (1 - first_discount_rate)
def second_discounted_price : ℚ := first_discounted_price * (1 - second_discount_rate)
def final_price_inclusive_tax : ℚ := second_discounted_price * (1 + tax_rate)

-- The proof problem statement
theorem final_price_jacket : final_price_inclusive_tax = 133.88 := sorry

end NUMINAMATH_GPT_final_price_jacket_l2217_221780


namespace NUMINAMATH_GPT_quadratic_inequality_l2217_221737

variable (a b c A B C : ℝ)

theorem quadratic_inequality
  (h₁ : a ≠ 0)
  (h₂ : A ≠ 0)
  (h₃ : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_l2217_221737


namespace NUMINAMATH_GPT_part_I_part_II_l2217_221790

-- Part I
theorem part_I :
  ∀ (x_0 y_0 : ℝ),
  (x_0 ^ 2 + y_0 ^ 2 = 8) ∧
  (x_0 ^ 2 / 12 + y_0 ^ 2 / 6 = 1) →
  ∃ a b : ℝ, (a = 2 ∧ b = 2) →
  (∀ x y : ℝ, (x - 2) ^ 2 + (y - 2) ^ 2 = 8) :=
by 
sorry

-- Part II
theorem part_II :
  ¬ ∃ (x_0 y_0 k_1 k_2 : ℝ),
  (x_0 ^ 2 / 12 + y_0 ^ 2 / 6 = 1) ∧
  (k_1k_2 = (y_0^2 - 4) / (x_0^2 - 4)) ∧
  (k_1 + k_2 = 2 * x_0 * y_0 / (x_0^2 - 4)) ∧
  (k_1k_2 - (k_1 + k_2) / (x_0 * y_0) + 1 = 0) :=
by 
sorry

end NUMINAMATH_GPT_part_I_part_II_l2217_221790
