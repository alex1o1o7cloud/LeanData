import Mathlib

namespace part1_part2_l1506_150643

-- Statement for Part 1
theorem part1 : 
  ∃ (a : Fin 8 → ℕ), 
    (∀ i : Fin 8, 1 ≤ a i ∧ a i ≤ 8) ∧ 
    (∀ i j : Fin 8, i ≠ j → a i ≠ a j) ∧ 
    (∀ i : Fin 8, (a i + a (i + 1) + a (i + 2)) > 11) := sorry

-- Statement for Part 2
theorem part2 : 
  ¬ ∃ (a : Fin 8 → ℕ), 
    (∀ i : Fin 8, 1 ≤ a i ∧ a i ≤ 8) ∧ 
    (∀ i j : Fin 8, i ≠ j → a i ≠ a j) ∧ 
    (∀ i : Fin 8, (a i + a (i + 1) + a (i + 2)) > 13) := sorry

end part1_part2_l1506_150643


namespace eval_g_231_l1506_150601

def g (a b c : ℤ) : ℚ :=
  (c ^ 2 + a ^ 2) / (c - b)

theorem eval_g_231 : g 2 (-3) 1 = 5 / 4 :=
by
  sorry

end eval_g_231_l1506_150601


namespace actual_average_height_is_correct_l1506_150667

-- Definitions based on given conditions
def number_of_students : ℕ := 20
def incorrect_average_height : ℝ := 175.0
def incorrect_height_of_student : ℝ := 151.0
def actual_height_of_student : ℝ := 136.0

-- Prove that the actual average height is 174.25 cm
theorem actual_average_height_is_correct :
  (incorrect_average_height * number_of_students - (incorrect_height_of_student - actual_height_of_student)) / number_of_students = 174.25 :=
sorry

end actual_average_height_is_correct_l1506_150667


namespace abc_value_l1506_150679

theorem abc_value (a b c : ℂ) 
  (h1 : a * b + 5 * b = -20)
  (h2 : b * c + 5 * c = -20)
  (h3 : c * a + 5 * a = -20) : 
  a * b * c = -100 := 
by {
  sorry
}

end abc_value_l1506_150679


namespace hemisphere_surface_area_l1506_150639

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 225 * π) : 2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end hemisphere_surface_area_l1506_150639


namespace smallest_solution_l1506_150647

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l1506_150647


namespace find_a_from_conditions_l1506_150619

noncomputable def f (x b : ℤ) : ℤ := 4 * x + b

theorem find_a_from_conditions (b a : ℤ) (h1 : a = f (-4) b) (h2 : -4 = f a b) : a = -4 :=
by
  sorry

end find_a_from_conditions_l1506_150619


namespace higher_profit_percentage_l1506_150622

theorem higher_profit_percentage (P : ℝ) :
  (P / 100 * 800 = 144) ↔ (P = 18) :=
by
  sorry

end higher_profit_percentage_l1506_150622


namespace problem_solution_l1506_150631

theorem problem_solution (a b c : ℝ)
  (h₁ : 10 = (6 / 100) * a)
  (h₂ : 6 = (10 / 100) * b)
  (h₃ : c = b / a) : c = 0.36 :=
by sorry

end problem_solution_l1506_150631


namespace sequence_bounds_l1506_150636

theorem sequence_bounds (n : ℕ) (hpos : 0 < n) :
  ∃ (a : ℕ → ℝ), (a 0 = 1/2) ∧
  (∀ k < n, a (k + 1) = a k + (1/n) * (a k)^2) ∧
  (1 - 1 / n < a n ∧ a n < 1) :=
sorry

end sequence_bounds_l1506_150636


namespace calculation_equality_l1506_150610

theorem calculation_equality : ((8^5 / 8^2) * 4^4) = 2^17 := by
  sorry

end calculation_equality_l1506_150610


namespace simplify_complex_expression_l1506_150698

theorem simplify_complex_expression (i : ℂ) (h : i^2 = -1) : 
  7 * (4 - 2 * i) + 4 * i * (7 - 3 * i) = 40 + 14 * i :=
by
  sorry

end simplify_complex_expression_l1506_150698


namespace range_of_a_for_increasing_l1506_150691

noncomputable def f (a x : ℝ) : ℝ := x * abs (2 * a - x) + 2 * x

theorem range_of_a_for_increasing (a : ℝ) :
  -1 ≤ a ∧ a ≤ 1 ↔ ∀ x y : ℝ, x < y → f a x ≤ f a y :=
sorry

end range_of_a_for_increasing_l1506_150691


namespace average_weight_of_whole_class_l1506_150665

theorem average_weight_of_whole_class (n_a n_b : ℕ) (w_a w_b : ℕ) (avg_w_a avg_w_b : ℕ)
  (h_a : n_a = 36) (h_b : n_b = 24) (h_avg_a : avg_w_a = 30) (h_avg_b : avg_w_b = 30) :
  ((n_a * avg_w_a + n_b * avg_w_b) / (n_a + n_b) = 30) := 
by
  sorry

end average_weight_of_whole_class_l1506_150665


namespace negation_of_P_l1506_150607

open Classical

variable (x : ℝ)

def P (x : ℝ) : Prop :=
  x^2 + 2 > 2 * x

theorem negation_of_P : (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x := by
  sorry

end negation_of_P_l1506_150607


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l1506_150687

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

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l1506_150687


namespace sum_of_three_quadratics_no_rot_l1506_150611

def quad_poly_sum_no_root (p q : ℝ -> ℝ) : Prop :=
  ∀ x : ℝ, (p x + q x ≠ 0)

theorem sum_of_three_quadratics_no_rot (a b c d e f : ℝ)
    (h1 : quad_poly_sum_no_root (λ x => x^2 + a*x + b) (λ x => x^2 + c*x + d))
    (h2 : quad_poly_sum_no_root (λ x => x^2 + c*x + d) (λ x => x^2 + e*x + f))
    (h3 : quad_poly_sum_no_root (λ x => x^2 + e*x + f) (λ x => x^2 + a*x + b)) :
    quad_poly_sum_no_root (λ x => x^2 + a*x + b) 
                         (λ x => x^2 + c*x + d + x^2 + e*x + f) :=
sorry

end sum_of_three_quadratics_no_rot_l1506_150611


namespace find_value_of_f_f_neg1_l1506_150686

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -2 / x else 3 + Real.log x / Real.log 2

theorem find_value_of_f_f_neg1 :
  f (f (-1)) = 4 := by
  -- proof omitted
  sorry

end find_value_of_f_f_neg1_l1506_150686


namespace gcd_of_XY_is_6_l1506_150625

theorem gcd_of_XY_is_6 (X Y : ℕ) (h1 : Nat.lcm X Y = 180)
  (h2 : X * 6 = Y * 5) : Nat.gcd X Y = 6 :=
sorry

end gcd_of_XY_is_6_l1506_150625


namespace students_like_basketball_l1506_150605

variable (B C B_inter_C B_union_C : ℕ)

theorem students_like_basketball (hC : C = 8) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 17) 
    (h_incl_excl : B_union_C = B + C - B_inter_C) : B = 12 := by 
  -- Given: 
  --   C = 8
  --   B_inter_C = 3
  --   B_union_C = 17
  --   B_union_C = B + C - B_inter_C
  -- Prove: 
  --   B = 12
  sorry

end students_like_basketball_l1506_150605


namespace smallest_n_divisible_by_31997_l1506_150689

noncomputable def smallest_n_divisible_by_prime : Nat :=
  let p := 31997
  let k := p
  2 * k

theorem smallest_n_divisible_by_31997 :
  smallest_n_divisible_by_prime = 63994 :=
by
  unfold smallest_n_divisible_by_prime
  rfl

end smallest_n_divisible_by_31997_l1506_150689


namespace yellow_chip_value_l1506_150677

theorem yellow_chip_value
  (y b g : ℕ)
  (hb : b = g)
  (hchips : y^4 * (4 * b)^b * (5 * g)^g = 16000)
  (h4yellow : y = 2) :
  y = 2 :=
by {
  sorry
}

end yellow_chip_value_l1506_150677


namespace plane_through_Ox_and_point_plane_parallel_Oz_and_points_l1506_150612

-- Definitions for first plane problem
def plane1_through_Ox_axis (y z : ℝ) : Prop := 3 * y + 2 * z = 0

-- Definitions for second plane problem
def plane2_parallel_Oz (x y : ℝ) : Prop := x + 3 * y - 1 = 0

theorem plane_through_Ox_and_point : plane1_through_Ox_axis 2 (-3) := 
by {
  -- Hint: Prove that substituting y = 2 and z = -3 in the equation results in LHS equals RHS.
  -- proof
  sorry 
}

theorem plane_parallel_Oz_and_points : 
  plane2_parallel_Oz 1 0 ∧ plane2_parallel_Oz (-2) 1 :=
by {
  -- Hint: Prove that substituting the points (1, 0) and (-2, 1) in the equation results in LHS equals RHS.
  -- proof
  sorry
}

end plane_through_Ox_and_point_plane_parallel_Oz_and_points_l1506_150612


namespace frank_total_pages_read_l1506_150637

-- Definitions of given conditions
def first_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days
def second_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days
def third_book_pages (pages_per_day : ℕ) (days : ℕ) := pages_per_day * days

-- Given values
def pages_first_book := first_book_pages 22 569
def pages_second_book := second_book_pages 35 315
def pages_third_book := third_book_pages 18 450

-- Total number of pages read by Frank
def total_pages := pages_first_book + pages_second_book + pages_third_book

-- Statement to prove
theorem frank_total_pages_read : total_pages = 31643 := by
  sorry

end frank_total_pages_read_l1506_150637


namespace small_gate_width_l1506_150693

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

end small_gate_width_l1506_150693


namespace roots_solution_l1506_150603

theorem roots_solution (p q : ℝ) (h1 : (∀ x : ℝ, (x - 3) * (3 * x + 8) = x^2 - 5 * x + 6 → (x = p ∨ x = q)))
  (h2 : p + q = 0) (h3 : p * q = -9) : (p + 4) * (q + 4) = 7 :=
by
  sorry

end roots_solution_l1506_150603


namespace matching_pair_probability_l1506_150624

theorem matching_pair_probability :
  let gray_socks := 12
  let white_socks := 10
  let black_socks := 6
  let total_socks := gray_socks + white_socks + black_socks
  let total_ways := total_socks.choose 2
  let gray_matching := gray_socks.choose 2
  let white_matching := white_socks.choose 2
  let black_matching := black_socks.choose 2
  let matching_ways := gray_matching + white_matching + black_matching
  let probability := matching_ways / total_ways
  probability = 1 / 3 :=
by sorry

end matching_pair_probability_l1506_150624


namespace largest_multiple_of_45_l1506_150676

theorem largest_multiple_of_45 (m : ℕ) 
  (h₁ : m % 45 = 0) 
  (h₂ : ∀ d : ℕ, d ∈ m.digits 10 → d = 8 ∨ d = 0) : 
  m / 45 = 197530 := 
sorry

end largest_multiple_of_45_l1506_150676


namespace reading_schedule_l1506_150666

-- Definitions of reading speeds and conditions
def total_pages := 910
def alice_speed := 30  -- seconds per page
def bob_speed := 60    -- seconds per page
def chandra_speed := 45  -- seconds per page

-- Mathematical problem statement
theorem reading_schedule :
  ∃ (x y : ℕ), 
    (x < y) ∧ 
    (y ≤ total_pages) ∧ 
    (30 * x = 45 * (y - x) ∧ 45 * (y - x) = 60 * (total_pages - y)) ∧ 
    x = 420 ∧ 
    y = 700 :=
  sorry

end reading_schedule_l1506_150666


namespace sasha_quarters_max_l1506_150626

/-- Sasha has \$4.80 in U.S. coins. She has four times as many dimes as she has nickels 
and the same number of quarters as nickels. Prove that the greatest number 
of quarters she could have is 6. -/
theorem sasha_quarters_max (q n d : ℝ) (h1 : 0.25 * q + 0.05 * n + 0.1 * d = 4.80)
  (h2 : n = q) (h3 : d = 4 * n) : q = 6 := 
sorry

end sasha_quarters_max_l1506_150626


namespace salary_reduction_l1506_150620

theorem salary_reduction (S : ℝ) (R : ℝ) :
  ((S - (R / 100 * S)) * 1.25 = S) → (R = 20) :=
by
  sorry

end salary_reduction_l1506_150620


namespace m_minus_n_is_square_l1506_150699

theorem m_minus_n_is_square (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 2001 * m ^ 2 + m = 2002 * n ^ 2 + n) : ∃ k : ℕ, m - n = k ^ 2 :=
sorry

end m_minus_n_is_square_l1506_150699


namespace rational_values_of_expressions_l1506_150648

theorem rational_values_of_expressions {x : ℚ} :
  (∃ a : ℚ, x / (x^2 + x + 1) = a) → (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) :=
by
  sorry

end rational_values_of_expressions_l1506_150648


namespace larger_number_is_50_l1506_150627

variable (a b : ℕ)
-- Conditions given in the problem
axiom cond1 : 4 * b = 5 * a
axiom cond2 : b - a = 10

-- The proof statement
theorem larger_number_is_50 : b = 50 :=
sorry

end larger_number_is_50_l1506_150627


namespace quadratic_real_root_iff_b_range_l1506_150629

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_real_root_iff_b_range_l1506_150629


namespace madeline_refills_l1506_150664

theorem madeline_refills :
  let total_water := 100
  let bottle_capacity := 12
  let remaining_to_drink := 16
  let already_drank := total_water - remaining_to_drink
  let initial_refills := already_drank / bottle_capacity
  let refills := initial_refills + 1
  refills = 8 :=
by
  sorry

end madeline_refills_l1506_150664


namespace probability_first_queen_second_diamond_l1506_150655

/-- 
Given that two cards are dealt at random from a standard deck of 52 cards,
the probability that the first card is a Queen and the second card is a Diamonds suit is 1/52.
-/
theorem probability_first_queen_second_diamond : 
  let total_cards := 52
  let probability (num events : ℕ) := (events : ℝ) / (num + 0 : ℝ)
  let prob_first_queen := probability total_cards 4
  let prob_second_diamond_if_first_queen_diamond := probability (total_cards - 1) 12
  let prob_second_diamond_if_first_other_queen := probability (total_cards - 1) 13
  let prob_first_queen_diamond := probability total_cards 1
  let prob_first_queen_other := probability total_cards 3
  let joint_prob_first_queen_diamond := prob_first_queen_diamond * prob_second_diamond_if_first_queen_diamond
  let joint_prob_first_queen_other := prob_first_queen_other * prob_second_diamond_if_first_other_queen
  let total_probability := joint_prob_first_queen_diamond + joint_prob_first_queen_other
  total_probability = probability total_cards 1 := 
by
  -- Proof goes here
  sorry

end probability_first_queen_second_diamond_l1506_150655


namespace geom_seq_a_sum_first_n_terms_l1506_150632

noncomputable def a (n : ℕ) : ℕ := 2^(n + 1)

def b (n : ℕ) : ℕ := 3 * (n + 1) - 2

def a_b_product (n : ℕ) : ℕ := (3 * (n + 1) - 2) * 2^(n + 1)

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => a_b_product k)

theorem geom_seq_a (n : ℕ) : a (n + 1) = 2 * a n :=
by sorry

theorem sum_first_n_terms (n : ℕ) : S n = 10 + (3 * n - 5) * 2^(n + 1) :=
by sorry

end geom_seq_a_sum_first_n_terms_l1506_150632


namespace optionA_not_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_right_triangle_l1506_150616
-- Import necessary libraries

-- Define each of the conditions as Lean definitions
def OptionA (a b c : ℝ) : Prop := a = 1.5 ∧ b = 2 ∧ c = 3
def OptionB (a b c : ℝ) : Prop := a = 7 ∧ b = 24 ∧ c = 25
def OptionC (a b c : ℝ) : Prop := ∃ k : ℕ, a = (3 : ℝ)*k ∧ b = (4 : ℝ)*k ∧ c = (5 : ℝ)*k
def OptionD (a b c : ℝ) : Prop := a = 9 ∧ b = 12 ∧ c = 15

-- Define the Pythagorean theorem predicate
def Pythagorean (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- State the theorem to prove Option A cannot form a right triangle
theorem optionA_not_right_triangle : ¬ Pythagorean 1.5 2 3 := by sorry

-- State the remaining options can form a right triangle
theorem optionB_right_triangle : Pythagorean 7 24 25 := by sorry
theorem optionC_right_triangle (k : ℕ) : Pythagorean (3 * k) (4 * k) (5 * k) := by sorry
theorem optionD_right_triangle : Pythagorean 9 12 15 := by sorry

end optionA_not_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_right_triangle_l1506_150616


namespace find_a_plus_b_l1506_150635

theorem find_a_plus_b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 4 * a + 3 * b = 39) :
  a + b = 82 / 7 :=
sorry

end find_a_plus_b_l1506_150635


namespace last_two_digits_of_17_pow_17_l1506_150669

theorem last_two_digits_of_17_pow_17 : (17 ^ 17) % 100 = 77 := 
by sorry

end last_two_digits_of_17_pow_17_l1506_150669


namespace product_of_repeating_decimal_l1506_150617

theorem product_of_repeating_decimal 
  (t : ℚ) 
  (h : t = 456 / 999) : 
  8 * t = 1216 / 333 :=
by
  sorry

end product_of_repeating_decimal_l1506_150617


namespace batsman_average_after_12th_inning_l1506_150684

variable (A : ℕ) (total_balls_faced : ℕ)

theorem batsman_average_after_12th_inning 
  (h1 : ∃ A, ∀ total_runs, total_runs = 11 * A)
  (h2 : ∃ A, ∀ total_runs_new, total_runs_new = 12 * (A + 4) ∧ total_runs_new - 60 = 11 * A)
  (h3 : 8 * 4 ≤ 60)
  (h4 : 6000 / total_balls_faced ≥ 130) 
  : (A + 4 = 16) :=
by
  sorry

end batsman_average_after_12th_inning_l1506_150684


namespace problem_1_problem_2_l1506_150674

theorem problem_1 (A B C : ℝ) (h_cond : (abs (B - A)) * (abs (C - A)) * (Real.cos A) = 3 * (abs (A - B)) * (abs (C - B)) * (Real.cos B)) : 
  (Real.tan B = 3 * Real.tan A) := 
sorry

theorem problem_2 (A B C : ℝ) (h_cosC : Real.cos C = Real.sqrt 5 / 5) (h_tanB : Real.tan B = 3 * Real.tan A) : 
  (A = Real.pi / 4) := 
sorry

end problem_1_problem_2_l1506_150674


namespace negation_all_dogs_playful_l1506_150642

variable {α : Type} (dog playful : α → Prop)

theorem negation_all_dogs_playful :
  (¬ ∀ x, dog x → playful x) ↔ (∃ x, dog x ∧ ¬ playful x) :=
by sorry

end negation_all_dogs_playful_l1506_150642


namespace solve_quadratic_equation_1_solve_quadratic_equation_2_l1506_150608

theorem solve_quadratic_equation_1 (x : ℝ) :
  3 * x^2 + 2 * x - 1 = 0 ↔ x = 1/3 ∨ x = -1 :=
by sorry

theorem solve_quadratic_equation_2 (x : ℝ) :
  (x + 2) * (x - 3) = 5 * x - 15 ↔ x = 3 :=
by sorry

end solve_quadratic_equation_1_solve_quadratic_equation_2_l1506_150608


namespace number_of_B_students_l1506_150623

-- Conditions
def prob_A (prob_B : ℝ) := 0.6 * prob_B
def prob_C (prob_B : ℝ) := 1.6 * prob_B
def prob_D (prob_B : ℝ) := 0.3 * prob_B

-- Total students
def total_students : ℝ := 50

-- Main theorem statement
theorem number_of_B_students (x : ℝ) (h1 : prob_A x + x + prob_C x + prob_D x = total_students) :
  x = 14 :=
  by
-- Proof skipped
  sorry

end number_of_B_students_l1506_150623


namespace complement_computation_l1506_150663

open Set

theorem complement_computation (U A : Set ℕ) :
  U = {1, 2, 3, 4, 5, 6, 7} → A = {2, 4, 5} →
  U \ A = {1, 3, 6, 7} :=
by
  intros hU hA
  rw [hU, hA]
  ext
  simp
  sorry

end complement_computation_l1506_150663


namespace value_of_expression_l1506_150653

theorem value_of_expression (a : ℝ) (h : a = 1/2) : 
  (2 * a⁻¹ + a⁻¹ / 2) / a = 10 :=
by
  sorry

end value_of_expression_l1506_150653


namespace at_least_one_worker_must_wait_l1506_150641

/-- 
Given five workers who collectively have a salary of 1500 rubles, 
and each tape recorder costs 320 rubles, we need to prove that 
at least one worker will not be able to buy a tape recorder immediately. 
-/
theorem at_least_one_worker_must_wait 
  (num_workers : ℕ) 
  (total_salary : ℕ) 
  (tape_recorder_cost : ℕ) 
  (h_workers : num_workers = 5) 
  (h_salary : total_salary = 1500) 
  (h_cost : tape_recorder_cost = 320) :
  ∀ (tape_recorders_required : ℕ), 
    tape_recorders_required = num_workers → total_salary < tape_recorder_cost * tape_recorders_required → ∃ (k : ℕ), 1 ≤ k ∧ k ≤ num_workers ∧ total_salary < k * tape_recorder_cost :=
by 
  intros tape_recorders_required h_required h_insufficient
  sorry

end at_least_one_worker_must_wait_l1506_150641


namespace calculate_speed_of_boat_in_still_water_l1506_150695

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

end calculate_speed_of_boat_in_still_water_l1506_150695


namespace solve_quadratic_eq_l1506_150685

theorem solve_quadratic_eq (x : ℝ) : (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2) := by
  sorry

end solve_quadratic_eq_l1506_150685


namespace probability_of_winning_exactly_once_l1506_150671

-- Define the probability of player A winning a match
def prob_win_A (p : ℝ) : Prop := (1 - p) ^ 3 = 1 - 63 / 64

-- Define the binomial probability for exactly one win in three matches
def binomial_prob (p : ℝ) : ℝ := 3 * p * (1 - p) ^ 2

theorem probability_of_winning_exactly_once (p : ℝ) (h : prob_win_A p) : binomial_prob p = 9 / 64 :=
sorry

end probability_of_winning_exactly_once_l1506_150671


namespace Yura_catches_up_in_five_minutes_l1506_150668

-- Define the speeds and distances
variables (v_Lena v_Yura d_Lena d_Yura : ℝ)
-- Assume v_Yura = 2 * v_Lena (Yura is twice as fast)
axiom h1 : v_Yura = 2 * v_Lena 
-- Assume Lena walks for 5 minutes before Yura starts
axiom h2 : d_Lena = v_Lena * 5
-- Assume they walk at constant speeds
noncomputable def t_to_catch_up := 10 / 2 -- time Yura takes to catch up Lena

-- Define the proof problem
theorem Yura_catches_up_in_five_minutes :
    t_to_catch_up = 5 :=
by
    sorry

end Yura_catches_up_in_five_minutes_l1506_150668


namespace find_d_minus_c_l1506_150644

variable (c d : ℝ)

def rotate180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (2 * cx - x, 2 * cy - y)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

def transformations (q : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_eq_x (rotate180 q (2, 3))

theorem find_d_minus_c :
  transformations (c, d) = (1, -4) → d - c = 7 :=
by
  intro h
  sorry

end find_d_minus_c_l1506_150644


namespace total_leaves_l1506_150600

theorem total_leaves (ferns fronds leaves : ℕ) (h1 : ferns = 12) (h2 : fronds = 15) (h3 : leaves = 45) :
  ferns * fronds * leaves = 8100 :=
by
  sorry

end total_leaves_l1506_150600


namespace legs_per_bee_l1506_150634

def number_of_bees : ℕ := 8
def total_legs : ℕ := 48

theorem legs_per_bee : (total_legs / number_of_bees) = 6 := by
  sorry

end legs_per_bee_l1506_150634


namespace minimum_bailing_rate_l1506_150662

-- Conditions
def distance_to_shore : ℝ := 2 -- miles
def rowing_speed : ℝ := 3 -- miles per hour
def water_intake_rate : ℝ := 15 -- gallons per minute
def max_water_capacity : ℝ := 50 -- gallons

-- Result to prove
theorem minimum_bailing_rate (r : ℝ) : 
  (distance_to_shore / rowing_speed * 60 * water_intake_rate - distance_to_shore / rowing_speed * 60 * r) ≤ max_water_capacity →
  r ≥ 13.75 :=
by
  sorry

end minimum_bailing_rate_l1506_150662


namespace new_train_distance_l1506_150672

theorem new_train_distance (old_train_distance : ℕ) (additional_factor : ℕ) (h₀ : old_train_distance = 300) (h₁ : additional_factor = 50) :
  let new_train_distance := old_train_distance + (additional_factor * old_train_distance / 100)
  new_train_distance = 450 :=
by
  sorry

end new_train_distance_l1506_150672


namespace Sara_spent_on_each_movie_ticket_l1506_150618

def Sara_spent_on_each_movie_ticket_correct : Prop :=
  let T := 36.78
  let R := 1.59
  let B := 13.95
  (T - R - B) / 2 = 10.62

theorem Sara_spent_on_each_movie_ticket : 
  Sara_spent_on_each_movie_ticket_correct :=
by
  sorry

end Sara_spent_on_each_movie_ticket_l1506_150618


namespace increase_a1_intervals_of_increase_l1506_150683

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

end increase_a1_intervals_of_increase_l1506_150683


namespace find_y_l1506_150656

theorem find_y (x y z : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 8) (h4 : z = 2) : y = 3 :=
    sorry

end find_y_l1506_150656


namespace system1_solution_system2_solution_l1506_150673

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

end system1_solution_system2_solution_l1506_150673


namespace total_books_correct_l1506_150606

-- Define the number of books each person has
def joan_books : ℕ := 10
def tom_books : ℕ := 38
def lisa_books : ℕ := 27
def steve_books : ℕ := 45

-- Calculate the total number of books they have together
def total_books : ℕ := joan_books + tom_books + lisa_books + steve_books

-- State the theorem that needs to be proved
theorem total_books_correct : total_books = 120 :=
by
  sorry

end total_books_correct_l1506_150606


namespace combined_work_time_l1506_150646

def Worker_A_time : ℝ := 10
def Worker_B_time : ℝ := 15

theorem combined_work_time :
  (1 / Worker_A_time + 1 / Worker_B_time)⁻¹ = 6 := by
  sorry

end combined_work_time_l1506_150646


namespace vector_at_t5_l1506_150652

theorem vector_at_t5 (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) 
  (h1 : (a, b) = (2, 5)) 
  (h2 : (a + 3 * c, b + 3 * d) = (8, -7)) :
  (a + 5 * c, b + 5 * d) = (10, -11) :=
by
  sorry

end vector_at_t5_l1506_150652


namespace g_interval_l1506_150657

noncomputable def g (a b c : ℝ) : ℝ :=
  a / (a + b) + b / (b + c) + c / (c + a)

theorem g_interval (a b c : ℝ) (ha : 0 < a) (hb: 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 :=
sorry

end g_interval_l1506_150657


namespace seed_mixture_Y_is_25_percent_ryegrass_l1506_150681

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

end seed_mixture_Y_is_25_percent_ryegrass_l1506_150681


namespace range_of_x_l1506_150638

noncomputable def is_valid_x (x : ℝ) : Prop :=
  x ≥ 0 ∧ x ≠ 4

theorem range_of_x (x : ℝ) : 
  is_valid_x x ↔ x ≥ 0 ∧ x ≠ 4 :=
by sorry

end range_of_x_l1506_150638


namespace fraction_to_terminating_decimal_l1506_150613

-- Lean statement for the mathematical problem
theorem fraction_to_terminating_decimal: (13 : ℚ) / 200 = 0.26 := 
sorry

end fraction_to_terminating_decimal_l1506_150613


namespace intersection_of_A_and_B_l1506_150690

-- Define sets A and B
def A := {x : ℝ | x > 0}
def B := {x : ℝ | x < 1}

-- Statement of the proof problem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 1} := by
  sorry -- The proof goes here

end intersection_of_A_and_B_l1506_150690


namespace number_of_ways_to_prepare_all_elixirs_l1506_150678

def fairy_methods : ℕ := 2
def elf_methods : ℕ := 2
def fairy_elixirs : ℕ := 3
def elf_elixirs : ℕ := 4

theorem number_of_ways_to_prepare_all_elixirs : 
  (fairy_methods * fairy_elixirs) + (elf_methods * elf_elixirs) = 14 :=
by
  sorry

end number_of_ways_to_prepare_all_elixirs_l1506_150678


namespace russian_pairing_probability_l1506_150696

-- Definitions based on conditions
def total_players : ℕ := 10
def russian_players : ℕ := 4
def non_russian_players : ℕ := total_players - russian_players

-- Probability calculation as a hypothesis
noncomputable def pairing_probability (rs: ℕ) (ns: ℕ) : ℚ :=
  (rs * (rs - 1)) / (total_players * (total_players - 1))

theorem russian_pairing_probability :
  pairing_probability russian_players non_russian_players = 1 / 21 :=
sorry

end russian_pairing_probability_l1506_150696


namespace horizontal_length_of_monitor_l1506_150628

def monitor_diagonal := 32
def aspect_ratio_horizontal := 16
def aspect_ratio_height := 9

theorem horizontal_length_of_monitor :
  ∃ (horizontal_length : ℝ), horizontal_length = 512 / Real.sqrt 337 := by
  sorry

end horizontal_length_of_monitor_l1506_150628


namespace smaller_triangle_perimeter_l1506_150659

theorem smaller_triangle_perimeter (p : ℕ) (p1 : ℕ) (p2 : ℕ) (p3 : ℕ) 
  (h₀ : p = 11)
  (h₁ : p1 = 5)
  (h₂ : p2 = 7)
  (h₃ : p3 = 9) : 
  p1 + p2 + p3 - p = 10 := by
  sorry

end smaller_triangle_perimeter_l1506_150659


namespace problem1_problem2_l1506_150675

theorem problem1 : 1 - 2 + 3 + (-4) = -2 :=
sorry

theorem problem2 : (-6) / 3 - (-10) - abs (-8) = 0 :=
sorry

end problem1_problem2_l1506_150675


namespace original_price_of_sarees_l1506_150692

theorem original_price_of_sarees (P : ℝ) (h : 0.95 * 0.80 * P = 456) : P = 600 :=
by
  sorry

end original_price_of_sarees_l1506_150692


namespace age_difference_l1506_150654

theorem age_difference (a b : ℕ) (ha : a < 10) (hb : b < 10)
  (h1 : 10 * a + b + 10 = 3 * (10 * b + a + 10)) :
  10 * a + b - (10 * b + a) = 54 :=
by sorry

end age_difference_l1506_150654


namespace seventh_graders_count_l1506_150604

-- Define the problem conditions
def total_students (T : ℝ) : Prop := 0.38 * T = 76
def seventh_grade_ratio : ℝ := 0.32
def seventh_graders (S : ℝ) (T : ℝ) : Prop := S = seventh_grade_ratio * T

-- The goal statement
theorem seventh_graders_count {T S : ℝ} (h : total_students T) : seventh_graders S T → S = 64 :=
by
  sorry

end seventh_graders_count_l1506_150604


namespace seeds_germination_percentage_l1506_150694

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

end seeds_germination_percentage_l1506_150694


namespace average_weight_decrease_l1506_150660

theorem average_weight_decrease 
  (A1 : ℝ) (new_person_weight : ℝ) (num_initial : ℕ) (num_total : ℕ) 
  (hA1 : A1 = 55) (hnew_person_weight : new_person_weight = 50) 
  (hnum_initial : num_initial = 20) (hnum_total : num_total = 21) :
  A1 - ((A1 * num_initial + new_person_weight) / num_total) = 0.24 :=
by
  rw [hA1, hnew_person_weight, hnum_initial, hnum_total]
  -- Further proof steps would go here
  sorry

end average_weight_decrease_l1506_150660


namespace journey_speed_condition_l1506_150649

theorem journey_speed_condition (v : ℝ) :
  (10 : ℝ) = 112 / v + 112 / 24 → (224 / 2 = 112) → v = 21 := by
  intros
  apply sorry

end journey_speed_condition_l1506_150649


namespace valid_twenty_letter_words_l1506_150614

noncomputable def number_of_valid_words : ℕ := sorry

theorem valid_twenty_letter_words :
  number_of_valid_words = 3 * 2^18 := sorry

end valid_twenty_letter_words_l1506_150614


namespace multiple_of_shirt_cost_l1506_150682

theorem multiple_of_shirt_cost (S C M : ℕ) (h1 : S = 97) (h2 : C = 300 - S)
  (h3 : C = M * S + 9) : M = 2 :=
by
  -- The proof will be filled in here
  sorry

end multiple_of_shirt_cost_l1506_150682


namespace LTE_divisibility_l1506_150697

theorem LTE_divisibility (m : ℕ) (h_pos : 0 < m) :
  (∀ k : ℕ, k % 2 = 1 ∧ k ≥ 3 → 2^m ∣ k^m - 1) ↔ m = 1 ∨ m = 2 ∨ m = 4 :=
by
  sorry

end LTE_divisibility_l1506_150697


namespace coffee_ratio_is_one_to_five_l1506_150645

-- Given conditions
def thermos_capacity : ℕ := 20 -- capacity in ounces
def times_filled_per_day : ℕ := 2
def school_days_per_week : ℕ := 5
def new_weekly_coffee_consumption : ℕ := 40 -- in ounces

-- Definitions based on the conditions
def old_daily_coffee_consumption := thermos_capacity * times_filled_per_day
def old_weekly_coffee_consumption := old_daily_coffee_consumption * school_days_per_week

-- Theorem: The ratio of the new weekly coffee consumption to the old weekly coffee consumption is 1:5
theorem coffee_ratio_is_one_to_five : 
  new_weekly_coffee_consumption / old_weekly_coffee_consumption = 1 / 5 := 
by
  -- Proof is omitted
  sorry

end coffee_ratio_is_one_to_five_l1506_150645


namespace power_function_properties_l1506_150670

def power_function (f : ℝ → ℝ) (x : ℝ) (a : ℝ) : Prop :=
  f x = x ^ a

theorem power_function_properties :
  ∃ (f : ℝ → ℝ) (a : ℝ), power_function f 2 a ∧ f 2 = 1/2 ∧ 
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → 0 < x2 → 
    (f x1 + f x2) / 2 > f ((x1 + x2) / 2)) :=
sorry

end power_function_properties_l1506_150670


namespace sufficient_not_necessary_condition_l1506_150651

-- Define the condition on a
def condition (a : ℝ) : Prop := a > 0

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) : Prop := a^2 + a ≥ 0

-- The proof statement that "a > 0" is a sufficient but not necessary condition for "a^2 + a ≥ 0"
theorem sufficient_not_necessary_condition (a : ℝ) : condition a → quadratic_inequality a :=
by
    intro ha
    -- [The remaining part of the proof is skipped.]
    sorry

end sufficient_not_necessary_condition_l1506_150651


namespace price_of_turban_l1506_150658

theorem price_of_turban (T : ℝ) (h1 : ∀ (T : ℝ), 3 / 4 * (90 + T) = 40 + T) : T = 110 :=
by
  sorry

end price_of_turban_l1506_150658


namespace monomial_completes_square_l1506_150633

variable (x : ℝ)

theorem monomial_completes_square :
  ∃ (m : ℝ), ∀ (x : ℝ), ∃ (a b : ℝ), (16 * x^2 + 1 + m) = (a * x + b)^2 :=
sorry

end monomial_completes_square_l1506_150633


namespace ab_equals_4_l1506_150661

theorem ab_equals_4 (a b : ℝ) (h_pos : a > 0 ∧ b > 0)
  (h_area : (1/2) * (12 / a) * (8 / b) = 12) : a * b = 4 :=
by
  sorry

end ab_equals_4_l1506_150661


namespace smallest_angle_in_right_triangle_l1506_150615

noncomputable def is_consecutive_primes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ ∀ r, Nat.Prime r → p < r → r < q → False

theorem smallest_angle_in_right_triangle : ∃ p : ℕ, ∃ q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ p + q = 90 ∧ is_consecutive_primes p q ∧ p = 43 :=
by
  sorry

end smallest_angle_in_right_triangle_l1506_150615


namespace change_combinations_l1506_150602

def isValidCombination (nickels dimes quarters : ℕ) : Prop :=
  nickels * 5 + dimes * 10 + quarters * 25 = 50 ∧ quarters ≤ 1

theorem change_combinations : {n // ∃ (combinations : ℕ) (nickels dimes quarters : ℕ), 
  n = combinations ∧ isValidCombination nickels dimes quarters ∧ 
  ((nickels, dimes, quarters) = (10, 0, 0) ∨
   (nickels, dimes, quarters) = (8, 1, 0) ∨
   (nickels, dimes, quarters) = (6, 2, 0) ∨
   (nickels, dimes, quarters) = (4, 3, 0) ∨
   (nickels, dimes, quarters) = (2, 4, 0) ∨
   (nickels, dimes, quarters) = (0, 5, 0) ∨
   (nickels, dimes, quarters) = (5, 0, 1) ∨
   (nickels, dimes, quarters) = (3, 1, 1) ∨
   (nickels, dimes, quarters) = (1, 2, 1))}
  :=
  ⟨9, sorry⟩

end change_combinations_l1506_150602


namespace triangle_side_length_c_l1506_150621

theorem triangle_side_length_c
  (a b A B C : ℝ)
  (ha : a = Real.sqrt 3)
  (hb : b = 1)
  (hA : A = 2 * B)
  (hAngleSum : A + B + C = Real.pi) :
  ∃ c : ℝ, c = 2 := 
by
  sorry

end triangle_side_length_c_l1506_150621


namespace solution_set_of_inequality_l1506_150688

theorem solution_set_of_inequality: 
  {x : ℝ | (2 * x - 1) / x < 1} = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end solution_set_of_inequality_l1506_150688


namespace value_of_g_at_2_l1506_150609

def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem value_of_g_at_2 : g 2 = 0 :=
by
  sorry

end value_of_g_at_2_l1506_150609


namespace chris_score_l1506_150680

variable (s g c : ℕ)

theorem chris_score  (h1 : s = g + 60) (h2 : (s + g) / 2 = 110) (h3 : c = 110 * 120 / 100) :
  c = 132 := by
  sorry

end chris_score_l1506_150680


namespace abigail_collected_43_l1506_150640

noncomputable def cans_needed : ℕ := 100
noncomputable def collected_by_alyssa : ℕ := 30
noncomputable def more_to_collect : ℕ := 27
noncomputable def collected_by_abigail : ℕ := cans_needed - (collected_by_alyssa + more_to_collect)

theorem abigail_collected_43 : collected_by_abigail = 43 := by
  sorry

end abigail_collected_43_l1506_150640


namespace binary_to_decimal_101101_l1506_150630

def binary_to_decimal (digits : List ℕ) : ℕ :=
  digits.foldr (λ (digit : ℕ) (acc : ℕ × ℕ) => (acc.1 + digit * 2 ^ acc.2, acc.2 + 1)) (0, 0) |>.1

theorem binary_to_decimal_101101 : binary_to_decimal [1, 0, 1, 1, 0, 1] = 45 :=
by
  -- Proof is needed but here we use sorry as placeholder.
  sorry

end binary_to_decimal_101101_l1506_150630


namespace rate_of_current_l1506_150650

theorem rate_of_current : 
  ∀ (v c : ℝ), v = 3.3 → (∀ d: ℝ, d > 0 → (d / (v - c) = 2 * (d / (v + c))) → c = 1.1) :=
by
  intros v c hv h
  sorry

end rate_of_current_l1506_150650
