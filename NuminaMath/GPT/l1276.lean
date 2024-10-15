import Mathlib

namespace NUMINAMATH_GPT_failed_in_english_l1276_127644

/- Lean definitions and statement -/

def total_percentage := 100
def failed_H := 32
def failed_H_and_E := 12
def passed_H_or_E := 24

theorem failed_in_english (total_percentage failed_H failed_H_and_E passed_H_or_E : ℕ) (h1 : total_percentage = 100) (h2 : failed_H = 32) (h3 : failed_H_and_E = 12) (h4 : passed_H_or_E = 24) :
  total_percentage - (failed_H + (total_percentage - passed_H_or_E - failed_H_and_E)) = 56 :=
by sorry

end NUMINAMATH_GPT_failed_in_english_l1276_127644


namespace NUMINAMATH_GPT_find_t_l1276_127695

variable (s t : ℚ) -- Using the rational numbers since the correct answer involves a fraction

theorem find_t (h1 : 8 * s + 7 * t = 145) (h2 : s = t + 3) : t = 121 / 15 :=
by 
  sorry

end NUMINAMATH_GPT_find_t_l1276_127695


namespace NUMINAMATH_GPT_people_stools_chairs_l1276_127615

def total_legs (x y z : ℕ) : ℕ := 2 * x + 3 * y + 4 * z 

theorem people_stools_chairs (x y z : ℕ) : 
  (x > y) → (x > z) → (x < y + z) → (total_legs x y z = 32) → 
  (x = 5 ∧ y = 2 ∧ z = 4) :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_people_stools_chairs_l1276_127615


namespace NUMINAMATH_GPT_count_solutions_congruence_l1276_127699

theorem count_solutions_congruence : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, x + 20 ≡ 75 [MOD 45] ∧ x < 150 :=
sorry

end NUMINAMATH_GPT_count_solutions_congruence_l1276_127699


namespace NUMINAMATH_GPT_exists_nat_sum_of_squares_two_ways_l1276_127659

theorem exists_nat_sum_of_squares_two_ways :
  ∃ n : ℕ, n < 100 ∧ ∃ a b c d : ℕ, a ≠ b ∧ c ≠ d ∧ n = a^2 + b^2 ∧ n = c^2 + d^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_nat_sum_of_squares_two_ways_l1276_127659


namespace NUMINAMATH_GPT_total_amount_is_105_l1276_127602

theorem total_amount_is_105 (x_amount y_amount z_amount : ℝ) 
  (h1 : ∀ x, y_amount = x * 0.45) 
  (h2 : ∀ x, z_amount = x * 0.30) 
  (h3 : y_amount = 27) : 
  (x_amount + y_amount + z_amount = 105) := 
sorry

end NUMINAMATH_GPT_total_amount_is_105_l1276_127602


namespace NUMINAMATH_GPT_gretchen_total_earnings_l1276_127649

-- Define the conditions
def price_per_drawing : ℝ := 20.0
def caricatures_sold_saturday : ℕ := 24
def caricatures_sold_sunday : ℕ := 16

-- The total caricatures sold
def total_caricatures_sold : ℕ := caricatures_sold_saturday + caricatures_sold_sunday

-- The total amount of money made
def total_money_made : ℝ := total_caricatures_sold * price_per_drawing

-- The theorem to be proven
theorem gretchen_total_earnings : total_money_made = 800.0 := by
  sorry

end NUMINAMATH_GPT_gretchen_total_earnings_l1276_127649


namespace NUMINAMATH_GPT_reflection_y_axis_correct_l1276_127620

-- Define the coordinates and reflection across the y-axis
def reflect_y_axis (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (-p.1, p.2)

-- Define the original point M
def M : (ℝ × ℝ) := (3, 2)

-- State the theorem we want to prove
theorem reflection_y_axis_correct : reflect_y_axis M = (-3, 2) :=
by
  -- The proof would go here, but it is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_reflection_y_axis_correct_l1276_127620


namespace NUMINAMATH_GPT_min_socks_to_guarantee_10_pairs_l1276_127607

/--
Given a drawer containing 100 red socks, 80 green socks, 60 blue socks, and 40 black socks, 
and socks are selected one at a time without seeing their color. 
The minimum number of socks that must be selected to guarantee at least 10 pairs is 23.
-/
theorem min_socks_to_guarantee_10_pairs 
  (red_socks green_socks blue_socks black_socks : ℕ) 
  (total_pairs : ℕ)
  (h_red : red_socks = 100)
  (h_green : green_socks = 80)
  (h_blue : blue_socks = 60)
  (h_black : black_socks = 40)
  (h_total_pairs : total_pairs = 10) :
  ∃ (n : ℕ), n = 23 := 
sorry

end NUMINAMATH_GPT_min_socks_to_guarantee_10_pairs_l1276_127607


namespace NUMINAMATH_GPT_loss_percentage_on_book_sold_at_loss_l1276_127638

theorem loss_percentage_on_book_sold_at_loss :
  ∀ (total_cost cost1 : ℝ) (gain_percent : ℝ),
    total_cost = 420 → cost1 = 245 → gain_percent = 0.19 →
    (∀ (cost2 SP : ℝ), cost2 = total_cost - cost1 →
                       SP = cost2 * (1 + gain_percent) →
                       SP = 208.25 →
                       ((cost1 - SP) / cost1 * 100) = 15) :=
by
  intros total_cost cost1 gain_percent h_total_cost h_cost1 h_gain_percent cost2 SP h_cost2 h_SP h_SP_value
  sorry

end NUMINAMATH_GPT_loss_percentage_on_book_sold_at_loss_l1276_127638


namespace NUMINAMATH_GPT_largest_real_root_range_l1276_127616

theorem largest_real_root_range (b0 b1 b2 b3 : ℝ) (h0 : |b0| ≤ 1) (h1 : |b1| ≤ 1) (h2 : |b2| ≤ 1) (h3 : |b3| ≤ 1) :
  ∀ r : ℝ, (Polynomial.eval r (Polynomial.C (1:ℝ) + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C b0) = 0) → (5 / 2) < r ∧ r < 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_real_root_range_l1276_127616


namespace NUMINAMATH_GPT_max_arith_seq_20_terms_l1276_127697

noncomputable def max_arithmetic_sequences :
  Nat :=
  180

theorem max_arith_seq_20_terms (a : Nat → Nat) :
  (∀ (k : Nat), k ≥ 1 ∧ k ≤ 20 → ∃ d : Nat, a (k + 1) = a k + d) →
  (P : Nat) = max_arithmetic_sequences :=
  by
  -- here's where the proof would go
  sorry

end NUMINAMATH_GPT_max_arith_seq_20_terms_l1276_127697


namespace NUMINAMATH_GPT_greatest_solution_of_equation_l1276_127610

theorem greatest_solution_of_equation : ∀ x : ℝ, x ≠ 9 ∧ (x^2 - x - 90) / (x - 9) = 4 / (x + 6) → x ≤ -7 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_greatest_solution_of_equation_l1276_127610


namespace NUMINAMATH_GPT_triangle_area_l1276_127656

theorem triangle_area (A B C : ℝ) (AB AC : ℝ) (A_angle : ℝ) (h1 : A_angle = π / 6)
  (h2 : AB * AC * Real.cos A_angle = Real.tan A_angle) :
  1 / 2 * AB * AC * Real.sin A_angle = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1276_127656


namespace NUMINAMATH_GPT_expression_is_odd_l1276_127612

-- Define positive integers
def is_positive (n : ℕ) := n > 0

-- Define odd integer
def is_odd (n : ℕ) := n % 2 = 1

-- Define multiple of 3
def is_multiple_of_3 (n : ℕ) := ∃ k : ℕ, n = 3 * k

-- The Lean 4 statement to prove the problem
theorem expression_is_odd (a b c : ℕ)
  (ha : is_positive a) (hb : is_positive b) (hc : is_positive c)
  (h_odd_a : is_odd a) (h_odd_b : is_odd b) (h_mult_3_c : is_multiple_of_3 c) :
  is_odd (5^a + (b-1)^2 * c) :=
by
  sorry

end NUMINAMATH_GPT_expression_is_odd_l1276_127612


namespace NUMINAMATH_GPT_striped_to_total_ratio_l1276_127634

theorem striped_to_total_ratio (total_students shorts_checkered_diff striped_shorts_diff : ℕ)
    (h_total : total_students = 81)
    (h_shorts_checkered : ∃ checkered, shorts_checkered_diff = checkered + 19)
    (h_striped_shorts : ∃ shorts, striped_shorts_diff = shorts + 8) :
    (striped_shorts_diff : ℚ) / total_students = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_striped_to_total_ratio_l1276_127634


namespace NUMINAMATH_GPT_calc_expr_correct_l1276_127626

noncomputable def eval_expr : ℚ :=
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 12.5

theorem calc_expr_correct : eval_expr = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_calc_expr_correct_l1276_127626


namespace NUMINAMATH_GPT_weekly_allowance_l1276_127617

theorem weekly_allowance (A : ℝ) (H1 : A - (3/5) * A = (2/5) * A)
(H2 : (2/5) * A - (1/3) * ((2/5) * A) = (4/15) * A)
(H3 : (4/15) * A = 0.96) : A = 3.6 := 
sorry

end NUMINAMATH_GPT_weekly_allowance_l1276_127617


namespace NUMINAMATH_GPT_total_musicians_is_98_l1276_127691

-- Define the number of males and females in the orchestra
def males_in_orchestra : ℕ := 11
def females_in_orchestra : ℕ := 12

-- Define the total number of musicians in the orchestra
def total_in_orchestra : ℕ := males_in_orchestra + females_in_orchestra

-- Define the number of musicians in the band as twice the number in the orchestra
def total_in_band : ℕ := 2 * total_in_orchestra

-- Define the number of males and females in the choir
def males_in_choir : ℕ := 12
def females_in_choir : ℕ := 17

-- Define the total number of musicians in the choir
def total_in_choir : ℕ := males_in_choir + females_in_choir

-- Prove that the total number of musicians in the orchestra, band, and choir is 98
theorem total_musicians_is_98 : total_in_orchestra + total_in_band + total_in_choir = 98 :=
by {
  -- Adding placeholders for the proof steps
  sorry
}

end NUMINAMATH_GPT_total_musicians_is_98_l1276_127691


namespace NUMINAMATH_GPT_tan_2beta_l1276_127627

theorem tan_2beta {α β : ℝ} 
  (h₁ : Real.tan (α + β) = 2) 
  (h₂ : Real.tan (α - β) = 3) : 
  Real.tan (2 * β) = -1 / 7 :=
by 
  sorry

end NUMINAMATH_GPT_tan_2beta_l1276_127627


namespace NUMINAMATH_GPT_grid_labelings_count_l1276_127648

theorem grid_labelings_count :
  ∃ (labeling_count : ℕ), 
    labeling_count = 2448 ∧ 
    (∀ (grid : Matrix (Fin 3) (Fin 3) ℕ),
      grid 0 0 = 1 ∧ 
      grid 2 2 = 2009 ∧ 
      (∀ (i j : Fin 3), j < 2 → grid i j ∣ grid i (j + 1)) ∧ 
      (∀ (i j : Fin 3), i < 2 → grid i j ∣ grid (i + 1) j)) :=
sorry

end NUMINAMATH_GPT_grid_labelings_count_l1276_127648


namespace NUMINAMATH_GPT_proof_2_in_M_l1276_127631

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end NUMINAMATH_GPT_proof_2_in_M_l1276_127631


namespace NUMINAMATH_GPT_scarlett_initial_oil_amount_l1276_127606

theorem scarlett_initial_oil_amount (x : ℝ) (h : x + 0.67 = 0.84) : x = 0.17 :=
by sorry

end NUMINAMATH_GPT_scarlett_initial_oil_amount_l1276_127606


namespace NUMINAMATH_GPT_exists_triangle_perimeter_lt_1cm_circumradius_gt_1km_l1276_127635

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem exists_triangle_perimeter_lt_1cm_circumradius_gt_1km :
  ∃ (A B C : ℝ) (a b c : ℝ), a + b + c < 0.01 ∧ circumradius a b c > 1000 :=
by
  sorry

end NUMINAMATH_GPT_exists_triangle_perimeter_lt_1cm_circumradius_gt_1km_l1276_127635


namespace NUMINAMATH_GPT_find_C_coordinates_l1276_127630

-- Define the points A, B, and the vector relationship
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)
def C : ℝ × ℝ := (-3, 9)

-- The condition stating vector AC is twice vector AB
def vector_condition (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1, C.2 - A.2) = (2 * (B.1 - A.1), 2 * (B.2 - A.2))

-- The theorem we need to prove
theorem find_C_coordinates (A B C : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (-1, 5))
  (hCondition : vector_condition A B C) : C = (-3, 9) :=
by
  rw [hA, hB] at hCondition
  -- sorry here skips the proof
  sorry

end NUMINAMATH_GPT_find_C_coordinates_l1276_127630


namespace NUMINAMATH_GPT_largest_divisor_of_n_l1276_127675

theorem largest_divisor_of_n 
  (n : ℕ) (h_pos : n > 0) (h_div : 72 ∣ n^2) : 
  ∃ v : ℕ, v = 12 ∧ v ∣ n :=
by
  use 12
  sorry

end NUMINAMATH_GPT_largest_divisor_of_n_l1276_127675


namespace NUMINAMATH_GPT_max_marks_exam_l1276_127660

theorem max_marks_exam (M : ℝ) 
  (h1 : 0.80 * M = 400) :
  M = 500 := 
by
  sorry

end NUMINAMATH_GPT_max_marks_exam_l1276_127660


namespace NUMINAMATH_GPT_total_students_in_lunchroom_l1276_127663

theorem total_students_in_lunchroom 
  (students_per_table : ℕ) 
  (number_of_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : number_of_tables = 34) : 
  students_per_table * number_of_tables = 204 := by
  sorry

end NUMINAMATH_GPT_total_students_in_lunchroom_l1276_127663


namespace NUMINAMATH_GPT_checkerboard_inequivalent_color_schemes_l1276_127678

/-- 
  We consider a 7x7 checkerboard where two squares are painted yellow, and the remaining 
  are painted green. Two color schemes are equivalent if one can be obtained from 
  the other by rotations of 0°, 90°, 180°, or 270°. We aim to prove that the 
  number of inequivalent color schemes is 312. 
-/
theorem checkerboard_inequivalent_color_schemes : 
  let n := 7
  let total_squares := n * n
  let total_pairs := total_squares.choose 2
  let symmetric_pairs := 24
  let nonsymmetric_pairs := total_pairs - symmetric_pairs
  let unique_symmetric_pairs := symmetric_pairs 
  let unique_nonsymmetric_pairs := nonsymmetric_pairs / 4
  unique_symmetric_pairs + unique_nonsymmetric_pairs = 312 :=
by sorry

end NUMINAMATH_GPT_checkerboard_inequivalent_color_schemes_l1276_127678


namespace NUMINAMATH_GPT_area_of_triangle_PQR_l1276_127677

theorem area_of_triangle_PQR 
  (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 8)
  (P_is_center : ∃ P : ℝ, True) -- Simplified assumption that P exists
  (bases_on_same_line : True) -- Assumed true, as touching condition implies it
  : ∃ area : ℝ, area = 20 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_PQR_l1276_127677


namespace NUMINAMATH_GPT_amy_balloons_l1276_127650

theorem amy_balloons (james_balloons amy_balloons : ℕ) (h1 : james_balloons = 1222) (h2 : james_balloons = amy_balloons + 709) : amy_balloons = 513 :=
by
  sorry

end NUMINAMATH_GPT_amy_balloons_l1276_127650


namespace NUMINAMATH_GPT_maximum_time_for_3_digit_combination_lock_l1276_127662

def max_time_to_open_briefcase : ℕ :=
  let num_combinations := 9 * 9 * 9
  let time_per_trial := 3
  num_combinations * time_per_trial

theorem maximum_time_for_3_digit_combination_lock :
  max_time_to_open_briefcase = 2187 :=
by
  sorry

end NUMINAMATH_GPT_maximum_time_for_3_digit_combination_lock_l1276_127662


namespace NUMINAMATH_GPT_estimated_germination_probability_l1276_127600

-- This definition represents the conditions of the problem in Lean.
def germination_data : List (ℕ × ℕ × Real) :=
  [(2, 2, 1.000), (5, 4, 0.800), (10, 9, 0.900), (50, 44, 0.880), (100, 92, 0.920),
   (500, 463, 0.926), (1000, 928, 0.928), (1500, 1396, 0.931), (2000, 1866, 0.933), (3000, 2794, 0.931)]

-- The theorem states that the germination probability is approximately 0.93.
theorem estimated_germination_probability (data : List (ℕ × ℕ × Real)) (h : data = germination_data) :
  ∃ p : Real, p = 0.93 ∧ ∀ n m r, (n, m, r) ∈ data → |r - p| < 0.01 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_estimated_germination_probability_l1276_127600


namespace NUMINAMATH_GPT_anna_lemonade_difference_l1276_127639

variables (x y p s : ℝ)

theorem anna_lemonade_difference (h : x * p = 1.5 * (y * s)) : (x * p) - (y * s) = 0.5 * (y * s) :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_anna_lemonade_difference_l1276_127639


namespace NUMINAMATH_GPT_sqrt_cosine_identity_l1276_127683

theorem sqrt_cosine_identity :
  Real.sqrt ((3 - Real.cos (Real.pi / 8)^2) * (3 - Real.cos (3 * Real.pi / 8)^2)) = (3 * Real.sqrt 5) / 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_cosine_identity_l1276_127683


namespace NUMINAMATH_GPT_cube_vertices_count_l1276_127636

-- Defining the conditions of the problem
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def euler_formula (V E F : ℕ) : Prop := V - E + F = 2

-- Stating the proof problem
theorem cube_vertices_count : ∃ V : ℕ, euler_formula V num_edges num_faces ∧ V = 8 :=
by
  sorry

end NUMINAMATH_GPT_cube_vertices_count_l1276_127636


namespace NUMINAMATH_GPT_abs_iff_sq_gt_l1276_127671

theorem abs_iff_sq_gt (x y : ℝ) : (|x| > |y|) ↔ (x^2 > y^2) :=
by sorry

end NUMINAMATH_GPT_abs_iff_sq_gt_l1276_127671


namespace NUMINAMATH_GPT_product_has_correct_sign_and_units_digit_l1276_127681

noncomputable def product_negative_integers_divisible_by_3_less_than_198 : ℤ :=
  sorry

theorem product_has_correct_sign_and_units_digit :
  product_negative_integers_divisible_by_3_less_than_198 < 0 ∧
  product_negative_integers_divisible_by_3_less_than_198 % 10 = 6 :=
by
  sorry

end NUMINAMATH_GPT_product_has_correct_sign_and_units_digit_l1276_127681


namespace NUMINAMATH_GPT_total_people_in_boats_l1276_127605

theorem total_people_in_boats (bo_num : ℝ) (avg_people : ℝ) (bo_num_eq : bo_num = 3.0) (avg_people_eq : avg_people = 1.66666666699999) : ∃ total_people : ℕ, total_people = 6 := 
by
  sorry

end NUMINAMATH_GPT_total_people_in_boats_l1276_127605


namespace NUMINAMATH_GPT_mangoes_in_basket_B_l1276_127685

theorem mangoes_in_basket_B :
  ∀ (A C D E B : ℕ), 
    (A = 15) →
    (C = 20) →
    (D = 25) →
    (E = 35) →
    (5 * 25 = A + C + D + E + B) →
    (B = 30) :=
by
  intros A C D E B hA hC hD hE hSum
  sorry

end NUMINAMATH_GPT_mangoes_in_basket_B_l1276_127685


namespace NUMINAMATH_GPT_AC_amount_l1276_127669

variable (A B C : ℝ)

theorem AC_amount
  (h1 : A + B + C = 400)
  (h2 : B + C = 150)
  (h3 : C = 50) :
  A + C = 300 := by
  sorry

end NUMINAMATH_GPT_AC_amount_l1276_127669


namespace NUMINAMATH_GPT_value_of_expression_l1276_127618

theorem value_of_expression (a : ℝ) (h : a^2 - 2 * a = 1) : 3 * a^2 - 6 * a - 4 = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1276_127618


namespace NUMINAMATH_GPT_larger_number_is_26_l1276_127601

theorem larger_number_is_26 {x y : ℤ} 
  (h1 : x + y = 45) 
  (h2 : x - y = 7) : 
  max x y = 26 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_26_l1276_127601


namespace NUMINAMATH_GPT_card_sorting_moves_upper_bound_l1276_127604

theorem card_sorting_moves_upper_bound (n : ℕ) (cells : Fin (n+1) → Fin (n+1)) (cards : Fin (n+1) → Fin (n+1)) : 
  (∃ (moves : (Fin (n+1) × Fin (n+1)) → ℕ),
    (∀ (i : Fin (n+1)), moves (i, cards i) ≤ 2 * n - 1) ∧ 
    (cards 0 = 0 → moves (0, 0) = 2 * n - 1) ∧ 
    (∃! start_pos : Fin (n+1) → Fin (n+1), 
      moves (start_pos (n), start_pos (0)) = 2 * n - 1)) := sorry

end NUMINAMATH_GPT_card_sorting_moves_upper_bound_l1276_127604


namespace NUMINAMATH_GPT_area_of_rectangle_l1276_127647

theorem area_of_rectangle (P : ℝ) (w : ℝ) (h : ℝ) (A : ℝ) 
  (hP : P = 28) 
  (hw : w = 6) 
  (hP_formula : P = 2 * (h + w)) 
  (hA_formula : A = h * w) : 
  A = 48 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1276_127647


namespace NUMINAMATH_GPT_fraction_of_juan_chocolates_given_to_tito_l1276_127686

variable (n : ℕ)
variable (Juan Angela Tito : ℕ)
variable (f : ℝ)

-- Conditions
def chocolates_Angela_Tito : Angela = 3 * Tito := 
by sorry

def chocolates_Juan_Angela : Juan = 4 * Angela := 
by sorry

def equal_distribution : (Juan + Angela + Tito) = 16 * n := 
by sorry

-- Theorem to prove
theorem fraction_of_juan_chocolates_given_to_tito (n : ℕ) 
  (H1 : Angela = 3 * Tito)
  (H2 : Juan = 4 * Angela)
  (H3 : Juan + Angela + Tito = 16 * n) :
  f = 13 / 36 :=
by sorry

end NUMINAMATH_GPT_fraction_of_juan_chocolates_given_to_tito_l1276_127686


namespace NUMINAMATH_GPT_ratio_perimeters_of_squares_l1276_127654

theorem ratio_perimeters_of_squares 
  (s₁ s₂ : ℝ)
  (h : (s₁ ^ 2) / (s₂ ^ 2) = 25 / 36) :
  (4 * s₁) / (4 * s₂) = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_perimeters_of_squares_l1276_127654


namespace NUMINAMATH_GPT_seafood_regular_price_l1276_127664

theorem seafood_regular_price (y : ℝ) (h : y / 4 = 4) : 2 * y = 32 := by
  sorry

end NUMINAMATH_GPT_seafood_regular_price_l1276_127664


namespace NUMINAMATH_GPT_find_expression_for_a_n_l1276_127621

-- Definitions for conditions in the problem
variable (a : ℕ → ℝ) -- Sequence is of positive real numbers
variable (S : ℕ → ℝ) -- Sum of the first n terms of the sequence

-- Condition that all terms in the sequence a_n are positive and indexed by natural numbers starting from 1
axiom pos_seq : ∀ n : ℕ, 0 < a (n + 1)
-- Condition for the sum of the terms: 4S_n = a_n^2 + 2a_n for n ∈ ℕ*
axiom sum_condition : ∀ n : ℕ, 4 * S (n + 1) = (a (n + 1))^2 + 2 * a (n + 1)

-- Theorem stating that sequence a_n = 2n given the above conditions
theorem find_expression_for_a_n : ∀ n : ℕ, a (n + 1) = 2 * (n + 1) := by
  sorry

end NUMINAMATH_GPT_find_expression_for_a_n_l1276_127621


namespace NUMINAMATH_GPT_fraction_transformation_l1276_127698

theorem fraction_transformation (a b x: ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + 2 * b) / (a - 2 * b) = (x + 2) / (x - 2) :=
by sorry

end NUMINAMATH_GPT_fraction_transformation_l1276_127698


namespace NUMINAMATH_GPT_sector_area_l1276_127646

theorem sector_area (r : ℝ) (θ : ℝ) (h_r : r = 12) (h_θ : θ = 40) : (θ / 360) * π * r^2 = 16 * π :=
by
  rw [h_r, h_θ]
  sorry

end NUMINAMATH_GPT_sector_area_l1276_127646


namespace NUMINAMATH_GPT_squares_to_nine_l1276_127625

theorem squares_to_nine (x : ℤ) : x^2 = 9 ↔ x = 3 ∨ x = -3 :=
sorry

end NUMINAMATH_GPT_squares_to_nine_l1276_127625


namespace NUMINAMATH_GPT_triangle_area_is_correct_l1276_127692

-- Defining the points
structure Point where
  x : ℝ
  y : ℝ

-- Defining vertices A, B, C
def A : Point := { x := 2, y := -3 }
def B : Point := { x := 0, y := 4 }
def C : Point := { x := 3, y := -1 }

-- Vector from C to A
def v : Point := { x := A.x - C.x, y := A.y - C.y }

-- Vector from C to B
def w : Point := { x := B.x - C.x, y := B.y - C.y }

-- Cross product of vectors v and w in 2D
noncomputable def cross_product (v w : Point) : ℝ :=
  v.x * w.y - v.y * w.x

-- Absolute value of the cross product
noncomputable def abs_cross_product (v w : Point) : ℝ :=
  |cross_product v w|

-- Area of the triangle
noncomputable def area_of_triangle (v w : Point) : ℝ :=
  (1 / 2) * abs_cross_product v w

-- Prove the area of the triangle is 5.5
theorem triangle_area_is_correct : area_of_triangle v w = 5.5 :=
  sorry

end NUMINAMATH_GPT_triangle_area_is_correct_l1276_127692


namespace NUMINAMATH_GPT_min_value_expr_l1276_127633

open Real

theorem min_value_expr (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) :
  3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ ≥ (3 : ℝ) * (12 * sqrt 2)^((1 : ℝ) / (3 : ℝ)) := sorry

end NUMINAMATH_GPT_min_value_expr_l1276_127633


namespace NUMINAMATH_GPT_AM_GM_inequality_example_l1276_127628

open Real

theorem AM_GM_inequality_example 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ((a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2)) ≥ 9 * (a^2 * b^2 * c^2) :=
sorry

end NUMINAMATH_GPT_AM_GM_inequality_example_l1276_127628


namespace NUMINAMATH_GPT_second_fish_length_l1276_127657

-- Defining the conditions
def first_fish_length : ℝ := 0.3
def length_difference : ℝ := 0.1

-- Proof statement
theorem second_fish_length : ∀ (second_fish : ℝ), first_fish_length = second_fish + length_difference → second_fish = 0.2 :=
by 
  intro second_fish
  intro h
  sorry

end NUMINAMATH_GPT_second_fish_length_l1276_127657


namespace NUMINAMATH_GPT_greatest_x_l1276_127666

-- Define x as a positive multiple of 4.
def is_positive_multiple_of_four (x : ℕ) : Prop :=
  x > 0 ∧ ∃ k : ℕ, x = 4 * k

-- Statement of the equivalent proof problem
theorem greatest_x (x : ℕ) (h1: is_positive_multiple_of_four x) (h2: x^3 < 4096) : x ≤ 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_x_l1276_127666


namespace NUMINAMATH_GPT_time_to_decorate_l1276_127608

variable (mia_rate billy_rate total_eggs : ℕ)

theorem time_to_decorate (h_mia : mia_rate = 24) (h_billy : billy_rate = 10) (h_total : total_eggs = 170) :
  total_eggs / (mia_rate + billy_rate) = 5 :=
by
  sorry

end NUMINAMATH_GPT_time_to_decorate_l1276_127608


namespace NUMINAMATH_GPT_line_equation_of_projection_l1276_127624

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_v2 := v.1 * v.1 + v.2 * v.2
  (dot_uv / norm_v2 * v.1, dot_uv / norm_v2 * v.2)

theorem line_equation_of_projection (x y : ℝ) :
  proj (x, y) (3, -4) = (9 / 5, -12 / 5) ↔ y = (3 / 4) * x - 15 / 4 :=
sorry

end NUMINAMATH_GPT_line_equation_of_projection_l1276_127624


namespace NUMINAMATH_GPT_study_group_number_l1276_127672

theorem study_group_number (b : ℤ) :
  (¬ (b % 2 = 0) ∧ (b + b^3 < 8000) ∧ ¬ (∃ r : ℚ, r^2 = 13) ∧ (b % 7 = 0)
  ∧ (∃ r : ℚ, r = b) ∧ ¬ (b % 14 = 0)) →
  b = 7 :=
by
  sorry

end NUMINAMATH_GPT_study_group_number_l1276_127672


namespace NUMINAMATH_GPT_minimize_distance_l1276_127689

-- Definitions of points and distances
structure Point where
  x : ℝ
  y : ℝ

def distanceSquared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Condition points A, B, and C
def A := Point.mk 7 3
def B := Point.mk 3 0

-- Mathematical problem: Find the value of k that minimizes the sum of distances squared
theorem minimize_distance : ∃ k : ℝ, ∀ k', 
  (distanceSquared A (Point.mk 0 k) + distanceSquared B (Point.mk 0 k) ≤ 
   distanceSquared A (Point.mk 0 k') + distanceSquared B (Point.mk 0 k')) → 
  k = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimize_distance_l1276_127689


namespace NUMINAMATH_GPT_divide_triangle_in_half_l1276_127629

def triangle_vertices : Prop :=
  let A := (0, 2)
  let B := (0, 0)
  let C := (10, 0)
  let base := 10
  let height := 2
  let total_area := (1 / 2) * base * height

  ∀ (a : ℝ),
  (1 / 2) * a * height = total_area / 2 → a = 5

theorem divide_triangle_in_half : triangle_vertices := 
  sorry

end NUMINAMATH_GPT_divide_triangle_in_half_l1276_127629


namespace NUMINAMATH_GPT_triangle_side_length_l1276_127684

/-
  Given a triangle ABC with sides |AB| = c, |AC| = b, and centroid G, incenter I,
  if GI is perpendicular to BC, then we need to prove that |BC| = (b+c)/2.
-/
variable {A B C G I : Type}
variable {AB AC BC : ℝ} -- Lengths of the sides
variable {b c : ℝ} -- Given lengths
variable {G_centroid : IsCentroid A B C G} -- G is the centroid of triangle ABC
variable {I_incenter : IsIncenter A B C I} -- I is the incenter of triangle ABC
variable {G_perp_BC : IsPerpendicular G I BC} -- G I ⊥ BC

theorem triangle_side_length (h1 : |AB| = c) (h2 : |AC| = b) :
  |BC| = (b + c) / 2 := 
sorry

end NUMINAMATH_GPT_triangle_side_length_l1276_127684


namespace NUMINAMATH_GPT_degree_at_least_three_l1276_127688

noncomputable def p : Polynomial ℤ := sorry
noncomputable def q : Polynomial ℤ := sorry

theorem degree_at_least_three (h1 : p.degree ≥ 1)
                              (h2 : q.degree ≥ 1)
                              (h3 : (∃ xs : Fin 33 → ℤ, ∀ i, p.eval (xs i) * q.eval (xs i) - 2015 = 0)) :
  p.degree ≥ 3 ∧ q.degree ≥ 3 := 
sorry

end NUMINAMATH_GPT_degree_at_least_three_l1276_127688


namespace NUMINAMATH_GPT_jung_kook_blue_balls_l1276_127655

def num_boxes := 2
def blue_balls_per_box := 5
def total_blue_balls := num_boxes * blue_balls_per_box

theorem jung_kook_blue_balls : total_blue_balls = 10 :=
by
  sorry

end NUMINAMATH_GPT_jung_kook_blue_balls_l1276_127655


namespace NUMINAMATH_GPT_total_frogs_seen_by_hunter_l1276_127632

/-- Hunter saw 5 frogs sitting on lily pads in the pond. -/
def initial_frogs : ℕ := 5

/-- Three more frogs climbed out of the water onto logs floating in the pond. -/
def frogs_on_logs : ℕ := 3

/-- Two dozen baby frogs (24 frogs) hopped onto a big rock jutting out from the pond. -/
def baby_frogs : ℕ := 24

/--
The total number of frogs Hunter saw in the pond.
-/
theorem total_frogs_seen_by_hunter : initial_frogs + frogs_on_logs + baby_frogs = 32 := by
sorry

end NUMINAMATH_GPT_total_frogs_seen_by_hunter_l1276_127632


namespace NUMINAMATH_GPT_min_elements_l1276_127696

-- Definitions for conditions in part b
def num_elements (n : ℕ) : ℕ := 2 * n + 1
def sum_upper_bound (n : ℕ) : ℕ := 15 * n + 2
def sum_arithmetic_mean (n : ℕ) : ℕ := 14 * n + 7

-- Prove that for conditions, the number of elements should be at least 11
theorem min_elements (n : ℕ) (h : 14 * n + 7 ≤ 15 * n + 2) : 2 * n + 1 ≥ 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_elements_l1276_127696


namespace NUMINAMATH_GPT_pythagorean_triple_square_l1276_127661

theorem pythagorean_triple_square (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pythagorean : a^2 + b^2 = c^2) : ∃ k : ℤ, k^2 = (c - a) * (c - b) / 2 := 
sorry

end NUMINAMATH_GPT_pythagorean_triple_square_l1276_127661


namespace NUMINAMATH_GPT_rectangular_plot_area_l1276_127668

/-- The ratio between the length and the breadth of a rectangular plot is 7 : 5.
    If the perimeter of the plot is 288 meters, then the area of the plot is 5040 square meters.
-/
theorem rectangular_plot_area
    (L B : ℝ)
    (h1 : L / B = 7 / 5)
    (h2 : 2 * (L + B) = 288) :
    L * B = 5040 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_plot_area_l1276_127668


namespace NUMINAMATH_GPT_sum_integer_solutions_correct_l1276_127611

noncomputable def sum_of_integer_solutions (m : ℝ) : ℝ :=
  if (3 ≤ m ∧ m < 6) ∨ (-6 ≤ m ∧ m < -3) then -9 else 0

theorem sum_integer_solutions_correct (m : ℝ) :
  (∀ x : ℝ, (3 * x + m < 0 ∧ x > -5) → (∃ s : ℝ, s = sum_of_integer_solutions m ∧ s = -9)) :=
by
  sorry

end NUMINAMATH_GPT_sum_integer_solutions_correct_l1276_127611


namespace NUMINAMATH_GPT_Isabel_initial_flowers_l1276_127652

-- Constants for conditions
def b := 7  -- Number of bouquets after wilting
def fw := 10  -- Number of wilted flowers
def n := 8  -- Number of flowers in each bouquet

-- Theorem statement
theorem Isabel_initial_flowers (h1 : b = 7) (h2 : fw = 10) (h3 : n = 8) : 
  (b * n + fw = 66) := by
  sorry

end NUMINAMATH_GPT_Isabel_initial_flowers_l1276_127652


namespace NUMINAMATH_GPT_number_of_packages_needed_l1276_127676

-- Define the problem constants and constraints
def students_per_class := 30
def number_of_classes := 4
def buns_per_student := 2
def buns_per_package := 8

-- Calculate the total number of students
def total_students := number_of_classes * students_per_class

-- Calculate the total number of buns needed
def total_buns := total_students * buns_per_student

-- Calculate the required number of packages
def required_packages := total_buns / buns_per_package

-- Prove that the required number of packages is 30
theorem number_of_packages_needed : required_packages = 30 := by
  -- The proof would be here, but for now we assume it is correct
  sorry

end NUMINAMATH_GPT_number_of_packages_needed_l1276_127676


namespace NUMINAMATH_GPT_wheel_radius_increase_l1276_127694

theorem wheel_radius_increase 
  (d₁ d₂ : ℝ) -- distances according to the odometer (600 and 580 miles)
  (r₀ : ℝ)   -- original radius (17 inches)
  (C₁: d₁ = 600)
  (C₂: d₂ = 580)
  (C₃: r₀ = 17) :
  ∃ Δr : ℝ, Δr = 0.57 :=
by
  sorry

end NUMINAMATH_GPT_wheel_radius_increase_l1276_127694


namespace NUMINAMATH_GPT_problem_I2_1_problem_I2_2_problem_I2_3_problem_I2_4_l1276_127693

-- Problem I2.1
theorem problem_I2_1 (a : ℕ) (h₁ : a > 0) (h₂ : a^2 - 1 = 123 * 125) : a = 124 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2.2
theorem problem_I2_2 (b : ℕ) (h₁ : b = (2^3 - 16*2^2 - 9*2 + 124)) : b = 50 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2.3
theorem problem_I2_3 (n : ℕ) (h₁ : (n * (n - 3)) / 2 = 54) : n = 12 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2_4
theorem problem_I2_4 (d : ℤ) (n : ℤ) (h₁ : n = 12) 
  (h₂ : (d - 1) * 2 = (1 - n) * 2) : d = -10 :=
by {
  -- This proof needs to be filled in
  sorry
}

end NUMINAMATH_GPT_problem_I2_1_problem_I2_2_problem_I2_3_problem_I2_4_l1276_127693


namespace NUMINAMATH_GPT_incorrect_statement_D_l1276_127651

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + x
else -(x^2 + x)

theorem incorrect_statement_D : ¬(∀ x : ℝ, x ≤ 0 → f x = x^2 + x) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_D_l1276_127651


namespace NUMINAMATH_GPT_photo_students_count_l1276_127619

theorem photo_students_count (n m : ℕ) 
  (h1 : m - 1 = n + 4) 
  (h2 : m - 2 = n) : 
  n * m = 24 := 
by 
  sorry

end NUMINAMATH_GPT_photo_students_count_l1276_127619


namespace NUMINAMATH_GPT_solve_system_of_equations_in_nat_numbers_l1276_127614

theorem solve_system_of_equations_in_nat_numbers :
  ∃ a b c d : ℕ, a * b = c + d ∧ c * d = a + b ∧ a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_in_nat_numbers_l1276_127614


namespace NUMINAMATH_GPT_find_some_number_l1276_127613

theorem find_some_number :
  ∃ some_number : ℝ, (3.242 * 10 / some_number) = 0.032420000000000004 ∧ some_number = 1000 :=
by
  sorry

end NUMINAMATH_GPT_find_some_number_l1276_127613


namespace NUMINAMATH_GPT_parabola_translation_correct_l1276_127673

noncomputable def translate_parabola (x y : ℝ) (h : y = -2 * x^2 - 4 * x - 6) : Prop :=
  let x' := x - 1
  let y' := y + 3
  y' = -2 * x'^2 - 1

theorem parabola_translation_correct (x y : ℝ) (h : y = -2 * x^2 - 4 * x - 6) :
  translate_parabola x y h :=
sorry

end NUMINAMATH_GPT_parabola_translation_correct_l1276_127673


namespace NUMINAMATH_GPT_y1_greater_than_y2_l1276_127609

-- Definitions of the conditions.
def point1_lies_on_line (y₁ b : ℝ) : Prop := y₁ = -3 * (-2 : ℝ) + b
def point2_lies_on_line (y₂ b : ℝ) : Prop := y₂ = -3 * (-1 : ℝ) + b

-- The theorem to prove: y₁ > y₂ given the conditions.
theorem y1_greater_than_y2 (y₁ y₂ b : ℝ) (h1 : point1_lies_on_line y₁ b) (h2 : point2_lies_on_line y₂ b) : y₁ > y₂ :=
by {
  sorry
}

end NUMINAMATH_GPT_y1_greater_than_y2_l1276_127609


namespace NUMINAMATH_GPT_minimum_value_f_l1276_127643

noncomputable def f (x : ℝ) : ℝ := (x^2 / 8) + x * (Real.cos x) + (Real.cos (2 * x))

theorem minimum_value_f : ∃ x : ℝ, f x = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_f_l1276_127643


namespace NUMINAMATH_GPT_white_bread_served_l1276_127665

theorem white_bread_served (total_bread : ℝ) (wheat_bread : ℝ) (white_bread : ℝ) 
  (h1 : total_bread = 0.9) (h2 : wheat_bread = 0.5) : white_bread = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_white_bread_served_l1276_127665


namespace NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l1276_127680

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → 4 * a (n + 1) = 4 * a n - 7) →
  a 1 = 25 →
  (∀ n : ℕ, S n = (n * (50 - (7/4 : ℚ) * (n - 1))) / 2) →
  ∃ n : ℕ, n = 15 ∧ S n = 765 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_of_arithmetic_sequence_l1276_127680


namespace NUMINAMATH_GPT_suitable_bases_for_346_l1276_127682

theorem suitable_bases_for_346 (b : ℕ) (hb : b^3 ≤ 346 ∧ 346 < b^4 ∧ (346 % b) % 2 = 0) : b = 6 ∨ b = 7 :=
sorry

end NUMINAMATH_GPT_suitable_bases_for_346_l1276_127682


namespace NUMINAMATH_GPT_fixed_monthly_fee_l1276_127690

/-
  We want to prove that given two conditions:
  1. x + y = 12.48
  2. x + 2y = 17.54
  The fixed monthly fee (x) is 7.42.
-/

theorem fixed_monthly_fee (x y : ℝ) 
  (h1 : x + y = 12.48) 
  (h2 : x + 2 * y = 17.54) : 
  x = 7.42 := 
sorry

end NUMINAMATH_GPT_fixed_monthly_fee_l1276_127690


namespace NUMINAMATH_GPT_sin_30_eq_one_half_cos_11pi_over_4_eq_neg_sqrt2_over_2_l1276_127641

theorem sin_30_eq_one_half : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
by 
  -- This is the statement only, the proof will be here
  sorry

theorem cos_11pi_over_4_eq_neg_sqrt2_over_2 : Real.cos (11 * Real.pi / 4) = - Real.sqrt 2 / 2 :=
by 
  -- This is the statement only, the proof will be here
  sorry

end NUMINAMATH_GPT_sin_30_eq_one_half_cos_11pi_over_4_eq_neg_sqrt2_over_2_l1276_127641


namespace NUMINAMATH_GPT_carpet_dimensions_l1276_127637

theorem carpet_dimensions
  (x y q : ℕ)
  (h_dim : y = 2 * x)
  (h_room1 : ((q^2 + 50^2) = (q * 2 - 50)^2 + (50 * 2 - q)^2))
  (h_room2 : ((q^2 + 38^2) = (q * 2 - 38)^2 + (38 * 2 - q)^2)) :
  x = 25 ∧ y = 50 :=
sorry

end NUMINAMATH_GPT_carpet_dimensions_l1276_127637


namespace NUMINAMATH_GPT_phone_price_increase_is_40_percent_l1276_127658

-- Definitions based on the conditions
def initial_price_tv := 500
def increased_fraction_tv := 2 / 5
def initial_price_phone := 400
def total_amount_received := 1260

-- The price increase of the TV
def final_price_tv := initial_price_tv * (1 + increased_fraction_tv)

-- The final price of the phone
def final_price_phone := total_amount_received - final_price_tv

-- The percentage increase in the phone's price
def percentage_increase_phone := ((final_price_phone - initial_price_phone) / initial_price_phone) * 100

-- The theorem to prove
theorem phone_price_increase_is_40_percent :
  percentage_increase_phone = 40 := by
  sorry

end NUMINAMATH_GPT_phone_price_increase_is_40_percent_l1276_127658


namespace NUMINAMATH_GPT_simplify_sqrt_l1276_127670

theorem simplify_sqrt (a : ℝ) (h : a < 2) : Real.sqrt ((a - 2)^2) = 2 - a :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_l1276_127670


namespace NUMINAMATH_GPT_sin_45_eq_1_div_sqrt_2_l1276_127642

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_45_eq_1_div_sqrt_2_l1276_127642


namespace NUMINAMATH_GPT_cost_of_mens_t_shirt_l1276_127674

-- Definitions based on conditions
def womens_price : ℕ := 18
def womens_interval : ℕ := 30
def mens_interval : ℕ := 40
def shop_open_hours_per_day : ℕ := 12
def total_earnings_per_week : ℕ := 4914

-- Auxiliary definitions based on conditions
def t_shirts_sold_per_hour (interval : ℕ) : ℕ := 60 / interval
def t_shirts_sold_per_day (interval : ℕ) : ℕ := shop_open_hours_per_day * t_shirts_sold_per_hour interval
def t_shirts_sold_per_week (interval : ℕ) : ℕ := t_shirts_sold_per_day interval * 7

def weekly_earnings_womens : ℕ := womens_price * t_shirts_sold_per_week womens_interval
def weekly_earnings_mens : ℕ := total_earnings_per_week - weekly_earnings_womens
def mens_price : ℚ := weekly_earnings_mens / t_shirts_sold_per_week mens_interval

-- The statement to be proved
theorem cost_of_mens_t_shirt : mens_price = 15 := by
  sorry

end NUMINAMATH_GPT_cost_of_mens_t_shirt_l1276_127674


namespace NUMINAMATH_GPT_number_of_triangles_l1276_127645

theorem number_of_triangles (points_AB points_BC points_AC : ℕ)
                            (hAB : points_AB = 12)
                            (hBC : points_BC = 9)
                            (hAC : points_AC = 10) :
    let total_points := points_AB + points_BC + points_AC
    let total_combinations := Nat.choose total_points 3
    let degenerate_AB := Nat.choose points_AB 3
    let degenerate_BC := Nat.choose points_BC 3
    let degenerate_AC := Nat.choose points_AC 3
    let valid_triangles := total_combinations - (degenerate_AB + degenerate_BC + degenerate_AC)
    valid_triangles = 4071 :=
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_l1276_127645


namespace NUMINAMATH_GPT_addition_of_two_negatives_l1276_127667

theorem addition_of_two_negatives (a b : ℤ) (ha : a < 0) (hb : b < 0) : a + b < a ∧ a + b < b :=
by
  sorry

end NUMINAMATH_GPT_addition_of_two_negatives_l1276_127667


namespace NUMINAMATH_GPT_annual_interest_rate_l1276_127679

-- Define the conditions as given in the problem
def principal : ℝ := 5000
def maturity_amount : ℝ := 5080
def interest_tax_rate : ℝ := 0.2

-- Define the annual interest rate x
variable (x : ℝ)

-- Statement to be proved: the annual interest rate x is 0.02
theorem annual_interest_rate :
  principal + principal * x - interest_tax_rate * (principal * x) = maturity_amount → x = 0.02 :=
by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_l1276_127679


namespace NUMINAMATH_GPT_product_of_roots_l1276_127687

theorem product_of_roots (r1 r2 r3 : ℝ) : 
  (∀ x : ℝ, 2 * x^3 - 24 * x^2 + 96 * x + 56 = 0 → x = r1 ∨ x = r2 ∨ x = r3) →
  r1 * r2 * r3 = -28 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_l1276_127687


namespace NUMINAMATH_GPT_find_first_number_l1276_127653

theorem find_first_number (x : ℕ) (h : x + 15 = 20) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l1276_127653


namespace NUMINAMATH_GPT_horse_food_per_day_l1276_127603

theorem horse_food_per_day
  (total_horse_food_per_day : ℕ)
  (sheep_count : ℕ)
  (sheep_to_horse_ratio : ℕ)
  (horse_to_sheep_ratio : ℕ)
  (horse_food_per_horse_per_day : ℕ) :
  sheep_to_horse_ratio * horse_food_per_horse_per_day = total_horse_food_per_day / (sheep_count / sheep_to_horse_ratio * horse_to_sheep_ratio) :=
by
  -- Given
  let total_horse_food_per_day := 12880
  let sheep_count := 24
  let sheep_to_horse_ratio := 3
  let horse_to_sheep_ratio := 7

  -- We need to show that horse_food_per_horse_per_day = 230
  have horse_count : ℕ := (sheep_count / sheep_to_horse_ratio) * horse_to_sheep_ratio
  have horse_food_per_horse_per_day : ℕ := total_horse_food_per_day / horse_count

  -- Desired proof statement
  sorry

end NUMINAMATH_GPT_horse_food_per_day_l1276_127603


namespace NUMINAMATH_GPT_tan_double_angle_l1276_127623

theorem tan_double_angle (α : ℝ) (h1 : α > 0) (h2 : α < Real.pi)
  (h3 : Real.cos α + Real.sin α = -1 / 5) : Real.tan (2 * α) = -24 / 7 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l1276_127623


namespace NUMINAMATH_GPT_constant_in_denominator_l1276_127640

theorem constant_in_denominator (x y z : ℝ) (some_constant : ℝ)
  (h : ((x - y)^3 + (y - z)^3 + (z - x)^3) / (some_constant * (x - y) * (y - z) * (z - x)) = 0.2) :
  some_constant = 15 := 
sorry

end NUMINAMATH_GPT_constant_in_denominator_l1276_127640


namespace NUMINAMATH_GPT_verify_parabola_D_l1276_127622

def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

def parabola_vertex (y : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, y x = vertex_form (-1) h k x

-- Given conditions
def h : ℝ := 2
def k : ℝ := 3

-- Possible expressions
def parabola_A (x : ℝ) : ℝ := -((x + 2)^2) - 3
def parabola_B (x : ℝ) : ℝ := -((x - 2)^2) - 3
def parabola_C (x : ℝ) : ℝ := -((x + 2)^2) + 3
def parabola_D (x : ℝ) : ℝ := -((x - 2)^2) + 3

theorem verify_parabola_D : parabola_vertex parabola_D 2 3 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_verify_parabola_D_l1276_127622
