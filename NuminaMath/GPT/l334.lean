import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_geometric_sequence_sum_first_n_terms_l334_33403

noncomputable def arithmetic_sequence (a n d : ℝ) : ℝ := 
  a + (n - 1) * d

noncomputable def geometric_sequence_sum (b1 r n : ℝ) : ℝ := 
  b1 * (1 - r^n) / (1 - r)

theorem arithmetic_sequence_general_formula (a1 d : ℝ) (h1 : a1 + 2 * d = 2) (h2 : 3 * a1 + 3 * d = 9 / 2) : 
  ∀ n, arithmetic_sequence a1 n d = (n + 1) / 2 :=
by 
  sorry

theorem geometric_sequence_sum_first_n_terms (a1 d b1 b4 : ℝ) (h1 : a1 + 2 * d = 2) (h2 : 3 * a1 + 3 * d = 9 / 2) 
  (h3 : b1 = a1) (h4 : b4 = arithmetic_sequence a1 15 d) (h5 : b4 = 8) :
  ∀ n, geometric_sequence_sum b1 2 n = 2^n - 1 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_geometric_sequence_sum_first_n_terms_l334_33403


namespace NUMINAMATH_GPT_total_number_of_bills_received_l334_33402

open Nat

-- Definitions based on the conditions:
def total_withdrawal_amount : ℕ := 600
def bill_denomination : ℕ := 20

-- Mathematically equivalent proof problem
theorem total_number_of_bills_received : (total_withdrawal_amount / bill_denomination) = 30 := 
by
  sorry

end NUMINAMATH_GPT_total_number_of_bills_received_l334_33402


namespace NUMINAMATH_GPT_sin_double_angle_l334_33409

theorem sin_double_angle (alpha : ℝ) (h1 : Real.cos (alpha + π / 4) = 3 / 5)
  (h2 : π / 2 ≤ alpha ∧ alpha ≤ 3 * π / 2) : Real.sin (2 * alpha) = 7 / 25 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_l334_33409


namespace NUMINAMATH_GPT_fraction_red_marbles_l334_33496

theorem fraction_red_marbles (x : ℕ) (h : x > 0) :
  let blue := (2/3 : ℚ) * x
  let red := (1/3 : ℚ) * x
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = (3/5 : ℚ) := by
  sorry

end NUMINAMATH_GPT_fraction_red_marbles_l334_33496


namespace NUMINAMATH_GPT_no_such_n_exists_l334_33473

-- Definition of the sum of the digits function s(n)
def s (n : ℕ) : ℕ := n.digits 10 |> List.sum

-- Statement of the proof problem
theorem no_such_n_exists : ¬ ∃ n : ℕ, n * s n = 20222022 :=
by
  -- argument based on divisibility rules as presented in the problem
  sorry

end NUMINAMATH_GPT_no_such_n_exists_l334_33473


namespace NUMINAMATH_GPT_drinking_ratio_l334_33457

variable (t_mala t_usha : ℝ) (d_usha : ℝ)

theorem drinking_ratio :
  (t_mala = t_usha) → 
  (d_usha = 2 / 10) →
  (1 - d_usha = 8 / 10) →
  (4 * d_usha = 8) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_drinking_ratio_l334_33457


namespace NUMINAMATH_GPT_find_a_value_l334_33440

theorem find_a_value (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x1 + x2 = 15) 
  (h3 : ∀ x, x^2 - 2 * a * x - 8 * a^2 < 0) : a = 15 / 2 :=
  sorry

end NUMINAMATH_GPT_find_a_value_l334_33440


namespace NUMINAMATH_GPT_charcoal_drawings_count_l334_33447

-- Defining the conditions
def total_drawings : Nat := 25
def colored_pencil_drawings : Nat := 14
def blending_marker_drawings : Nat := 7

-- Defining the target value for charcoal drawings
def charcoal_drawings : Nat := total_drawings - (colored_pencil_drawings + blending_marker_drawings)

-- The theorem we need to prove
theorem charcoal_drawings_count : charcoal_drawings = 4 :=
by
  -- Lean proof goes here, but since we skip the proof, we'll just use 'sorry'
  sorry

end NUMINAMATH_GPT_charcoal_drawings_count_l334_33447


namespace NUMINAMATH_GPT_max_x1_sq_plus_x2_sq_l334_33442

theorem max_x1_sq_plus_x2_sq (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = k - 2) 
  (h2 : x1 * x2 = k^2 + 3 * k + 5)
  (h3 : -4 ≤ k ∧ k ≤ -4 / 3) : 
  x1^2 + x2^2 ≤ 18 :=
by sorry

end NUMINAMATH_GPT_max_x1_sq_plus_x2_sq_l334_33442


namespace NUMINAMATH_GPT_find_number_l334_33453

theorem find_number (a : ℕ) (h : a = 105) : 
  a^3 / (49 * 45 * 25) = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l334_33453


namespace NUMINAMATH_GPT_factorable_iff_m_eq_2_l334_33478

theorem factorable_iff_m_eq_2 (m : ℤ) :
  (∃ (A B C D : ℤ), (x y : ℤ) -> (x^2 + 2*x*y + 2*x + m*y + 2*m = (x + A*y + B) * (x + C*y + D))) ↔ m = 2 :=
sorry

end NUMINAMATH_GPT_factorable_iff_m_eq_2_l334_33478


namespace NUMINAMATH_GPT_opposite_of_7_l334_33420

-- Define the concept of an opposite number for real numbers
def is_opposite (x y : ℝ) : Prop := x = -y

-- Theorem statement
theorem opposite_of_7 :
  is_opposite 7 (-7) :=
by {
  sorry
}

end NUMINAMATH_GPT_opposite_of_7_l334_33420


namespace NUMINAMATH_GPT_ratio_of_installing_to_downloading_l334_33432

noncomputable def timeDownloading : ℕ := 10

noncomputable def ratioTimeSpent (installingTime : ℕ) : ℚ :=
  let tutorialTime := 3 * (timeDownloading + installingTime)
  let totalTime := timeDownloading + installingTime + tutorialTime
  if totalTime = 60 then
    (installingTime : ℚ) / (timeDownloading : ℚ)
  else 0

theorem ratio_of_installing_to_downloading : ratioTimeSpent 5 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_installing_to_downloading_l334_33432


namespace NUMINAMATH_GPT_contest_score_order_l334_33450

variables (E F G H : ℕ) -- nonnegative scores of Emily, Fran, Gina, and Harry respectively

-- Conditions
axiom cond1 : E - F = G + H + 10
axiom cond2 : G + E > F + H + 5
axiom cond3 : H = F + 8

-- Statement to prove
theorem contest_score_order : (H > E) ∧ (E > F) ∧ (F > G) :=
sorry

end NUMINAMATH_GPT_contest_score_order_l334_33450


namespace NUMINAMATH_GPT_juniors_score_l334_33484

/-- Mathematical proof problem stated in Lean 4 -/
theorem juniors_score 
  (total_students : ℕ) 
  (juniors seniors : ℕ)
  (junior_score senior_avg total_avg : ℝ)
  (h_total_students : total_students > 0)
  (h_juniors : juniors = total_students / 10)
  (h_seniors : seniors = (total_students * 9) / 10)
  (h_total_avg : total_avg = 84)
  (h_senior_avg : senior_avg = 83)
  (h_junior_score_same : ∀ j : ℕ, j < juniors → ∃ s : ℝ, s = junior_score)
  :
  junior_score = 93 :=
by
  sorry

end NUMINAMATH_GPT_juniors_score_l334_33484


namespace NUMINAMATH_GPT_find_complement_intersection_find_union_complement_subset_implies_a_range_l334_33464

-- Definitions for sets A and B
def A : Set ℝ := { x | 3 ≤ x ∧ x < 6 }
def B : Set ℝ := { x | 2 < x ∧ x < 9 }

-- Definitions for complements and subsets
def complement (S : Set ℝ) : Set ℝ := { x | x ∉ S }
def intersection (S T : Set ℝ) : Set ℝ := { x | x ∈ S ∧ x ∈ T }
def union (S T : Set ℝ) : Set ℝ := { x | x ∈ S ∨ x ∈ T }

-- Definition for set C as a parameterized set by a
def C (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

-- Proof statements
theorem find_complement_intersection :
  complement (intersection A B) = { x | x < 3 ∨ x ≥ 6 } :=
by sorry

theorem find_union_complement :
  union (complement B) A = { x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9 } :=
by sorry

theorem subset_implies_a_range (a : ℝ) :
  C a ⊆ B → a ∈ {x | 2 ≤ x ∧ x ≤ 8} :=
by sorry

end NUMINAMATH_GPT_find_complement_intersection_find_union_complement_subset_implies_a_range_l334_33464


namespace NUMINAMATH_GPT_product_pattern_l334_33459

theorem product_pattern (a b : ℕ) (h1 : b < 10) (h2 : 10 - b < 10) :
    (10 * a + b) * (10 * a + (10 - b)) = 100 * a * (a + 1) + b * (10 - b) :=
by
  sorry

end NUMINAMATH_GPT_product_pattern_l334_33459


namespace NUMINAMATH_GPT_weight_of_10_moles_approx_l334_33444

def atomic_mass_C : ℝ := 12.01
def atomic_mass_H : ℝ := 1.008
def atomic_mass_O : ℝ := 16.00

def molar_mass_C6H8O6 : ℝ := 
  (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)

def moles : ℝ := 10
def given_total_weight : ℝ := 1760

theorem weight_of_10_moles_approx (ε : ℝ) (hε : ε > 0) :
  abs ((moles * molar_mass_C6H8O6) - given_total_weight) < ε := by
  -- proof will go here.
  sorry

end NUMINAMATH_GPT_weight_of_10_moles_approx_l334_33444


namespace NUMINAMATH_GPT_line_equation_l334_33482

theorem line_equation (k : ℝ) (x1 y1 : ℝ) (P : x1 = 1 ∧ y1 = -1) (angle_slope : k = Real.tan (135 * Real.pi / 180)) : 
  ∃ (a b : ℝ), a = -1 ∧ b = -1 ∧ (y1 = k * x1 + b) ∧ (y1 = a * x1 + b) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l334_33482


namespace NUMINAMATH_GPT_optimal_pricing_l334_33427

-- Define the conditions given in the problem
def cost_price : ℕ := 40
def selling_price : ℕ := 60
def weekly_sales : ℕ := 300

def sales_volume (price : ℕ) : ℕ := weekly_sales - 10 * (price - selling_price)
def profit (price : ℕ) : ℕ := (price - cost_price) * sales_volume price

-- Statement to prove
theorem optimal_pricing : ∃ (price : ℕ), price = 65 ∧ profit price = 6250 :=
by {
  sorry
}

end NUMINAMATH_GPT_optimal_pricing_l334_33427


namespace NUMINAMATH_GPT_burger_share_per_person_l334_33407

-- Definitions based on conditions
def foot_to_inches : ℕ := 12
def burger_length_foot : ℕ := 1
def burger_length_inches : ℕ := burger_length_foot * foot_to_inches

theorem burger_share_per_person : (burger_length_inches / 2) = 6 := by
  sorry

end NUMINAMATH_GPT_burger_share_per_person_l334_33407


namespace NUMINAMATH_GPT_find_abc_l334_33479

-- Definitions based on given conditions
variables (a b c : ℝ)
variable (h1 : a * b = 30 * (3 ^ (1/3)))
variable (h2 : a * c = 42 * (3 ^ (1/3)))
variable (h3 : b * c = 18 * (3 ^ (1/3)))

-- Formal statement of the proof problem
theorem find_abc : a * b * c = 90 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_abc_l334_33479


namespace NUMINAMATH_GPT_find_f_2015_plus_f_2016_l334_33425

def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom functional_equation (x : ℝ) : f (3/2 - x) = f x
axiom value_at_minus2 : f (-2) = -3

theorem find_f_2015_plus_f_2016 : f 2015 + f 2016 = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_f_2015_plus_f_2016_l334_33425


namespace NUMINAMATH_GPT_largest_z_l334_33477

theorem largest_z (x y z : ℝ) 
  (h1 : x + y + z = 5)  
  (h2 : x * y + y * z + x * z = 3) 
  : z ≤ 13 / 3 := sorry

end NUMINAMATH_GPT_largest_z_l334_33477


namespace NUMINAMATH_GPT_parabola_c_value_l334_33491

theorem parabola_c_value (b c : ℝ) 
  (h1 : 6 = 2^2 + 2 * b + c) 
  (h2 : 20 = 4^2 + 4 * b + c) : 
  c = 0 :=
by {
  -- We state that we're skipping the proof
  sorry
}

end NUMINAMATH_GPT_parabola_c_value_l334_33491


namespace NUMINAMATH_GPT_one_minus_repeating_three_l334_33476

theorem one_minus_repeating_three : ∀ b : ℚ, b = 1 / 3 → 1 - b = 2 / 3 :=
by
  intro b hb
  rw [hb]
  norm_num

end NUMINAMATH_GPT_one_minus_repeating_three_l334_33476


namespace NUMINAMATH_GPT_minerals_found_today_l334_33468

noncomputable def yesterday_gemstones := 21
noncomputable def today_minerals := 48
noncomputable def today_gemstones := 21

theorem minerals_found_today :
  (today_minerals - (2 * yesterday_gemstones) = 6) :=
by
  sorry

end NUMINAMATH_GPT_minerals_found_today_l334_33468


namespace NUMINAMATH_GPT_number_913n_divisible_by_18_l334_33466

theorem number_913n_divisible_by_18 (n : ℕ) (h1 : 9130 % 2 = 0) (h2 : (9 + 1 + 3 + n) % 9 = 0) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_913n_divisible_by_18_l334_33466


namespace NUMINAMATH_GPT_equal_intercepts_l334_33416

theorem equal_intercepts (a : ℝ) (h : ∃ (x y : ℝ), (x = (2 + a) / a ∧ y = 2 + a ∧ x = y)) :
  a = -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_GPT_equal_intercepts_l334_33416


namespace NUMINAMATH_GPT_rectangle_area_l334_33487

theorem rectangle_area (a b c d : ℝ) 
  (ha : a = 4) 
  (hb : b = 4) 
  (hc : c = 4) 
  (hd : d = 1) :
  ∃ E F G H : ℝ,
    (E = 0 ∧ F = 3 ∧ G = 4 ∧ H = 0) →
    (a + b + c + d) = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rectangle_area_l334_33487


namespace NUMINAMATH_GPT_B_necessary_not_sufficient_for_A_l334_33438

def A (x : ℝ) : Prop := 0 < x ∧ x < 5
def B (x : ℝ) : Prop := |x - 2| < 3

theorem B_necessary_not_sufficient_for_A (x : ℝ) :
  (A x → B x) ∧ (∃ x, B x ∧ ¬ A x) :=
by
  sorry

end NUMINAMATH_GPT_B_necessary_not_sufficient_for_A_l334_33438


namespace NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l334_33431

-- Define the conditions and prove the correctness of solutions to the equations
theorem equation_one_solution (x : ℝ) (h : 3 / (x - 2) = 9 / x) : x = 3 :=
by
  sorry

theorem equation_two_solution (x : ℝ) (h : x / (x + 1) = 2 * x / (3 * x + 3) - 1) : x = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l334_33431


namespace NUMINAMATH_GPT_find_m_n_sum_l334_33434

noncomputable def point (x : ℝ) (y : ℝ) := (x, y)

def center_line (P : ℝ × ℝ) : Prop := P.1 - P.2 - 2 = 0

def on_circle (C : ℝ × ℝ) (P : ℝ × ℝ) (r : ℝ) : Prop := 
  (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2

def circles_intersect (A B C D : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  on_circle A C r₁ ∧ on_circle A D r₂ ∧ on_circle B C r₁ ∧ on_circle B D r₂

theorem find_m_n_sum 
  (A : ℝ × ℝ) (m n : ℝ)
  (C D : ℝ × ℝ)
  (r₁ r₂ : ℝ)
  (H1 : A = point 1 3)
  (H2 : circles_intersect A (point m n) C D r₁ r₂)
  (H3 : center_line C ∧ center_line D) :
  m + n = 4 :=
sorry

end NUMINAMATH_GPT_find_m_n_sum_l334_33434


namespace NUMINAMATH_GPT_half_abs_diff_squares_eq_40_l334_33458

theorem half_abs_diff_squares_eq_40 (x y : ℤ) (hx : x = 21) (hy : y = 19) :
  (|x^2 - y^2| / 2) = 40 :=
by
  sorry

end NUMINAMATH_GPT_half_abs_diff_squares_eq_40_l334_33458


namespace NUMINAMATH_GPT_eq_x_add_q_l334_33470

theorem eq_x_add_q (x q : ℝ) (h1 : abs (x - 5) = q) (h2 : x > 5) : x + q = 5 + 2*q :=
by {
  sorry
}

end NUMINAMATH_GPT_eq_x_add_q_l334_33470


namespace NUMINAMATH_GPT_eval_g_six_times_at_2_l334_33404

def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem eval_g_six_times_at_2 : g (g (g (g (g (g 2))))) = 4 := sorry

end NUMINAMATH_GPT_eval_g_six_times_at_2_l334_33404


namespace NUMINAMATH_GPT_find_a_l334_33436

open Set

theorem find_a (A : Set ℝ) (B : Set ℝ) (f : ℝ → ℝ) (a : ℝ)
  (hA : A = Ici 0) 
  (hB : B = univ)
  (hf : ∀ x ∈ A, f x = 2^x - 1) 
  (ha_in_A : a ∈ A) 
  (ha_f_eq_3 : f a = 3) :
  a = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l334_33436


namespace NUMINAMATH_GPT_arithmetic_sequence_a8_l334_33454

/-- In an arithmetic sequence with the given sum of terms, prove the value of a_8 is 14. -/
theorem arithmetic_sequence_a8 (a : ℕ → ℕ) (d : ℕ) (h1 : ∀ (n : ℕ), a (n+1) = a n + d)
    (h2 : a 2 + a 7 + a 8 + a 9 + a 14 = 70) : a 8 = 14 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a8_l334_33454


namespace NUMINAMATH_GPT_find_pairs_l334_33419

theorem find_pairs :
  { (m, n) : ℕ × ℕ | (m > 0) ∧ (n > 0) ∧ (m^2 - n ∣ m + n^2)
      ∧ (n^2 - m ∣ n + m^2) } = { (2, 2), (3, 3), (1, 2), (2, 1), (3, 2), (2, 3) } :=
sorry

end NUMINAMATH_GPT_find_pairs_l334_33419


namespace NUMINAMATH_GPT_unique_solution_p_zero_l334_33422

theorem unique_solution_p_zero :
  ∃! (x y p : ℝ), 
    (x^2 - y^2 = 0) ∧ 
    (x * y + p * x - p * y = p^2) ↔ 
    p = 0 :=
by sorry

end NUMINAMATH_GPT_unique_solution_p_zero_l334_33422


namespace NUMINAMATH_GPT_eval_expr_l334_33497

theorem eval_expr : (3^3)^2 = 729 := 
by
  sorry

end NUMINAMATH_GPT_eval_expr_l334_33497


namespace NUMINAMATH_GPT_negation_equivalence_l334_33443

-- Definition of the original proposition
def proposition (x : ℝ) : Prop := x > 1 → Real.log x > 0

-- Definition of the negated proposition
def negation (x : ℝ) : Prop := ¬ (x > 1 → Real.log x > 0)

-- The mathematically equivalent proof problem as Lean statement
theorem negation_equivalence (x : ℝ) : 
  (¬ (x > 1 → Real.log x > 0)) ↔ (x ≤ 1 → Real.log x ≤ 0) := 
by 
  sorry

end NUMINAMATH_GPT_negation_equivalence_l334_33443


namespace NUMINAMATH_GPT_ellipse_foci_x_axis_l334_33483

theorem ellipse_foci_x_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1) : 0 < a ∧ a < b :=
sorry

end NUMINAMATH_GPT_ellipse_foci_x_axis_l334_33483


namespace NUMINAMATH_GPT_arithmetic_mean_of_17_29_45_64_l334_33493

theorem arithmetic_mean_of_17_29_45_64 : (17 + 29 + 45 + 64) / 4 = 38.75 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_17_29_45_64_l334_33493


namespace NUMINAMATH_GPT_each_friend_gets_four_pieces_l334_33410

noncomputable def pieces_per_friend : ℕ :=
  let oranges := 80
  let pieces_per_orange := 10
  let friends := 200
  (oranges * pieces_per_orange) / friends

theorem each_friend_gets_four_pieces :
  pieces_per_friend = 4 :=
by
  sorry

end NUMINAMATH_GPT_each_friend_gets_four_pieces_l334_33410


namespace NUMINAMATH_GPT_correct_calculated_value_l334_33449

theorem correct_calculated_value (n : ℕ) (h : n + 9 = 30) : n + 7 = 28 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculated_value_l334_33449


namespace NUMINAMATH_GPT_sum_of_products_of_two_at_a_time_l334_33462

theorem sum_of_products_of_two_at_a_time (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : a + b + c = 21) : 
  a * b + b * c + a * c = 100 := 
  sorry

end NUMINAMATH_GPT_sum_of_products_of_two_at_a_time_l334_33462


namespace NUMINAMATH_GPT_product_of_abc_l334_33417

noncomputable def abc_product (a b c : ℝ) : ℝ :=
  a * b * c

theorem product_of_abc (a b c m : ℝ) 
    (h1 : a + b + c = 300)
    (h2 : m = 5 * a)
    (h3 : m = b + 14)
    (h4 : m = c - 14) : 
    abc_product a b c = 664500 :=
by sorry

end NUMINAMATH_GPT_product_of_abc_l334_33417


namespace NUMINAMATH_GPT_average_playtime_l334_33490

-- Definitions based on conditions
def h_w := 2 -- Hours played on Wednesday
def h_t := 2 -- Hours played on Thursday
def h_f := h_w + 3 -- Hours played on Friday (3 hours more than Wednesday)

-- Statement to prove
theorem average_playtime :
  (h_w + h_t + h_f) / 3 = 3 := by
  sorry

end NUMINAMATH_GPT_average_playtime_l334_33490


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l334_33428

variable (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
variable (h3 : (1 + b^2 / a^2) = (6 / 4))

theorem asymptotes_of_hyperbola :
  ∃ (m : ℝ), m = b / a ∧ (m = Real.sqrt 2 / 2) ∧ ∀ x : ℝ, (y = m*x) ∨ (y = -m*x) :=
by
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l334_33428


namespace NUMINAMATH_GPT_p_suff_but_not_nec_q_l334_33494

variable (p q : Prop)

-- Given conditions: ¬p is a necessary but not sufficient condition for ¬q.
def neg_p_nec_but_not_suff_neg_q : Prop :=
  (¬q → ¬p) ∧ ¬(¬p → ¬q)

-- Concluding statement: p is a sufficient but not necessary condition for q.
theorem p_suff_but_not_nec_q 
  (h : neg_p_nec_but_not_suff_neg_q p q) : (p → q) ∧ ¬(q → p) := 
sorry

end NUMINAMATH_GPT_p_suff_but_not_nec_q_l334_33494


namespace NUMINAMATH_GPT_reggie_marbles_bet_l334_33446

theorem reggie_marbles_bet 
  (initial_marbles : ℕ) (final_marbles : ℕ) (games_played : ℕ) (games_lost : ℕ) (bet_per_game : ℕ)
  (h_initial : initial_marbles = 100) 
  (h_final : final_marbles = 90) 
  (h_games : games_played = 9) 
  (h_losses : games_lost = 1) : 
  bet_per_game = 13 :=
by
  sorry

end NUMINAMATH_GPT_reggie_marbles_bet_l334_33446


namespace NUMINAMATH_GPT_karl_present_salary_l334_33455

def original_salary : ℝ := 20000
def reduction_percentage : ℝ := 0.10
def increase_percentage : ℝ := 0.10

theorem karl_present_salary :
  let reduced_salary := original_salary * (1 - reduction_percentage)
  let present_salary := reduced_salary * (1 + increase_percentage)
  present_salary = 19800 :=
by
  sorry

end NUMINAMATH_GPT_karl_present_salary_l334_33455


namespace NUMINAMATH_GPT_find_x_l334_33474

noncomputable def h (x : ℚ) : ℚ :=
  (5 * ((x - 2) / 3) - 3)

theorem find_x : h (19/2) = 19/2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l334_33474


namespace NUMINAMATH_GPT_common_factor_of_polynomial_l334_33499

noncomputable def polynomial_common_factor (m : ℤ) : ℤ :=
  let polynomial := 2 * m^3 - 8 * m
  let common_factor := 2 * m
  common_factor  -- We're stating that the common factor is 2 * m

-- The theorem to verify that the common factor of each term in the polynomial is 2m
theorem common_factor_of_polynomial (m : ℤ) : 
  polynomial_common_factor m = 2 * m := by
  sorry

end NUMINAMATH_GPT_common_factor_of_polynomial_l334_33499


namespace NUMINAMATH_GPT_smallest_prime_8_less_than_square_l334_33439

theorem smallest_prime_8_less_than_square :
  ∃ p : ℕ, (∃ n : ℤ, p = n^2 - 8) ∧ Nat.Prime p ∧ p > 0 ∧ (∀ q : ℕ, (∃ m : ℤ, q = m^2 - 8) ∧ Nat.Prime q → q ≥ p) :=
sorry

end NUMINAMATH_GPT_smallest_prime_8_less_than_square_l334_33439


namespace NUMINAMATH_GPT_original_height_l334_33475

theorem original_height (h : ℝ) (h_rebound : ∀ n : ℕ, h / (4/3)^(n+1) > 0) (total_distance : ∀ h : ℝ, h*(1 + 1.5 + 1.5*(0.75) + 1.5*(0.75)^2 + 1.5*(0.75)^3 + (0.75)^4) = 305) :
  h = 56.3 := 
sorry

end NUMINAMATH_GPT_original_height_l334_33475


namespace NUMINAMATH_GPT_kelsey_more_than_ekon_l334_33400

theorem kelsey_more_than_ekon :
  ∃ (K E U : ℕ), (K = 160) ∧ (E = U - 17) ∧ (K + E + U = 411) ∧ (K - E = 43) :=
by
  sorry

end NUMINAMATH_GPT_kelsey_more_than_ekon_l334_33400


namespace NUMINAMATH_GPT_nancy_carrots_l334_33498

-- Definitions based on the conditions
def initial_carrots := 12
def carrots_to_cook := 2
def new_carrot_seeds := 5
def growth_factor := 3
def kept_carrots := 10
def poor_quality_ratio := 3

-- Calculate new carrots grown from seeds
def new_carrots := new_carrot_seeds * growth_factor

-- Total carrots after new ones are added
def total_carrots := kept_carrots + new_carrots

-- Calculate poor quality carrots (integer part only)
def poor_quality_carrots := total_carrots / poor_quality_ratio

-- Calculate good quality carrots
def good_quality_carrots := total_carrots - poor_quality_carrots

-- Statement to prove
theorem nancy_carrots : good_quality_carrots = 17 :=
by
  sorry -- proof is not required

end NUMINAMATH_GPT_nancy_carrots_l334_33498


namespace NUMINAMATH_GPT_sub_frac_pow_eq_l334_33421

theorem sub_frac_pow_eq :
  7 - (2 / 5)^3 = 867 / 125 := by
  sorry

end NUMINAMATH_GPT_sub_frac_pow_eq_l334_33421


namespace NUMINAMATH_GPT_days_c_worked_l334_33463

theorem days_c_worked (Da Db Dc : ℕ) (Wa Wb Wc : ℕ)
  (h1 : Da = 6) (h2 : Db = 9) (h3 : Wc = 100) (h4 : 3 * Wc = 5 * Wa)
  (h5 : 4 * Wc = 5 * Wb)
  (h6 : Wa * Da + Wb * Db + Wc * Dc = 1480) : Dc = 4 :=
by
  sorry

end NUMINAMATH_GPT_days_c_worked_l334_33463


namespace NUMINAMATH_GPT_purchasing_plans_count_l334_33406

theorem purchasing_plans_count :
  ∃ n : ℕ, n = 2 ∧ (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = 35) :=
sorry

end NUMINAMATH_GPT_purchasing_plans_count_l334_33406


namespace NUMINAMATH_GPT_solution_set_of_inequality_l334_33489

theorem solution_set_of_inequality (x : ℝ) : x^2 - 5 * |x| + 6 < 0 ↔ (-3 < x ∧ x < -2) ∨ (2 < x ∧ x < 3) :=
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l334_33489


namespace NUMINAMATH_GPT_arctan_sum_pi_l334_33465

open Real

theorem arctan_sum_pi : arctan (1 / 3) + arctan (3 / 8) + arctan (8 / 3) = π := 
sorry

end NUMINAMATH_GPT_arctan_sum_pi_l334_33465


namespace NUMINAMATH_GPT_find_x_l334_33413

theorem find_x (x : ℝ) (h : (3 * x) / 7 = 6) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l334_33413


namespace NUMINAMATH_GPT_fish_remaining_l334_33485

def fish_caught_per_hour := 7
def hours_fished := 9
def fish_lost := 15

theorem fish_remaining : 
  (fish_caught_per_hour * hours_fished - fish_lost) = 48 :=
by
  sorry

end NUMINAMATH_GPT_fish_remaining_l334_33485


namespace NUMINAMATH_GPT_total_cost_of_repair_l334_33472

theorem total_cost_of_repair (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) (H1 : hours = 2) (H2 : hourly_rate = 75) (H3 : part_cost = 150) :
  hours * hourly_rate + part_cost = 300 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_of_repair_l334_33472


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l334_33418

noncomputable def repeating_decimal_sum : ℚ :=
  let x := 1 / 3
  let y := 7 / 99
  let z := 8 / 999
  x + y + z

theorem sum_of_repeating_decimals :
  repeating_decimal_sum = 418 / 999 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_l334_33418


namespace NUMINAMATH_GPT_ascetic_height_l334_33424

theorem ascetic_height (h m : ℝ) (x : ℝ) (hx : h * (m + 1) = (x + h)^2 + (m * h)^2) : x = h * m / (m + 2) :=
sorry

end NUMINAMATH_GPT_ascetic_height_l334_33424


namespace NUMINAMATH_GPT_karens_speed_l334_33430

noncomputable def average_speed_karen (k : ℝ) : Prop :=
  let late_start_in_hours := 4 / 60
  let total_distance_karen := 24 + 4
  let time_karen := total_distance_karen / k
  let distance_tom_start := 45 * late_start_in_hours
  let distance_tom_total := distance_tom_start + 45 * time_karen
  distance_tom_total = 24

theorem karens_speed : average_speed_karen 60 :=
by
  sorry

end NUMINAMATH_GPT_karens_speed_l334_33430


namespace NUMINAMATH_GPT_point_on_xoz_plane_l334_33488

def Point := ℝ × ℝ × ℝ

def lies_on_plane_xoz (p : Point) : Prop :=
  p.2 = 0

theorem point_on_xoz_plane :
  lies_on_plane_xoz (-2, 0, 3) :=
by
  sorry

end NUMINAMATH_GPT_point_on_xoz_plane_l334_33488


namespace NUMINAMATH_GPT_henry_walks_distance_l334_33467

noncomputable def gym_distance : ℝ := 3

noncomputable def walk_factor : ℝ := 2 / 3

noncomputable def c_limit_position : ℝ := 1.5

noncomputable def d_limit_position : ℝ := 2.5

theorem henry_walks_distance :
  abs (c_limit_position - d_limit_position) = 1 := by
  sorry

end NUMINAMATH_GPT_henry_walks_distance_l334_33467


namespace NUMINAMATH_GPT_total_age_l334_33415

variable (A B : ℝ)

-- Conditions
def condition1 : Prop := A / B = 3 / 4
def condition2 : Prop := A - 10 = (1 / 2) * (B - 10)

-- Statement
theorem total_age : condition1 A B → condition2 A B → A + B = 35 := by
  sorry

end NUMINAMATH_GPT_total_age_l334_33415


namespace NUMINAMATH_GPT_intersection_M_N_l334_33412

noncomputable def M : Set ℝ := { x | x^2 ≤ x }
noncomputable def N : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_M_N :
  M ∩ N = { x | 0 < x ∧ x ≤ 1 } :=
  sorry

end NUMINAMATH_GPT_intersection_M_N_l334_33412


namespace NUMINAMATH_GPT_incorrect_locus_proof_l334_33405

-- Conditions given in the problem
def condition_A (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∈ locus ↔ conditions p)

def condition_B (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∉ locus ↔ ¬ conditions p) ∧ (conditions p ↔ p ∈ locus)

def condition_C (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∈ locus → conditions p) ∧ (∃ q, conditions q ∧ q ∈ locus)

def condition_D (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∉ locus ↔ ¬ conditions p) ∧ (p ∈ locus ↔ conditions p)

def condition_E (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (conditions p ↔ p ∈ locus) ∧ (¬ conditions p ↔ p ∉ locus)

-- Statement to be proved
theorem incorrect_locus_proof (locus : Set Point) (conditions : Point → Prop) :
  ¬ condition_C locus conditions :=
sorry

end NUMINAMATH_GPT_incorrect_locus_proof_l334_33405


namespace NUMINAMATH_GPT_Jerry_average_speed_l334_33441

variable (J : ℝ) -- Jerry's average speed in miles per hour
variable (C : ℝ) -- Carla's average speed in miles per hour
variable (T_J : ℝ) -- Time Jerry has been driving in hours
variable (T_C : ℝ) -- Time Carla has been driving in hours
variable (D : ℝ) -- Distance covered in miles

-- Given conditions
axiom Carla_speed : C = 35
axiom Carla_time : T_C = 3
axiom Jerry_time : T_J = T_C + 0.5

-- Distance covered by Carla in T_C hours at speed C
axiom Carla_distance : D = C * T_C

-- Distance covered by Jerry in T_J hours at speed J
axiom Jerry_distance : D = J * T_J

-- The goal to prove
theorem Jerry_average_speed : J = 30 :=
by
  sorry

end NUMINAMATH_GPT_Jerry_average_speed_l334_33441


namespace NUMINAMATH_GPT_half_angle_quadrant_l334_33435

-- Define the given condition
def is_angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

-- Define the result that needs to be proved
def is_angle_in_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, (k * 180 < α / 2 ∧ α / 2 < k * 180 + 45) ∨ (k * 180 + 180 < α / 2 ∧ α / 2 < k * 180 + 225)

-- The main theorem statement
theorem half_angle_quadrant (α : ℝ) (h : is_angle_in_first_quadrant α) : is_angle_in_first_or_third_quadrant α :=
sorry

end NUMINAMATH_GPT_half_angle_quadrant_l334_33435


namespace NUMINAMATH_GPT_total_dogs_in_kennel_l334_33456

-- Definition of the given conditions
def T := 45       -- Number of dogs that wear tags
def C := 40       -- Number of dogs that wear flea collars
def B := 6        -- Number of dogs that wear both tags and collars
def D_neither := 1 -- Number of dogs that wear neither a collar nor tags

-- Theorem statement
theorem total_dogs_in_kennel : T + C - B + D_neither = 80 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_dogs_in_kennel_l334_33456


namespace NUMINAMATH_GPT_quadratic_real_roots_l334_33429

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l334_33429


namespace NUMINAMATH_GPT_diagonal_less_than_half_perimeter_l334_33401

theorem diagonal_less_than_half_perimeter (a b c d x : ℝ) 
  (h1 : x < a + b) (h2 : x < c + d) : x < (a + b + c + d) / 2 := 
by
  sorry

end NUMINAMATH_GPT_diagonal_less_than_half_perimeter_l334_33401


namespace NUMINAMATH_GPT_probability_five_blue_marbles_is_correct_l334_33452

noncomputable def probability_of_five_blue_marbles : ℝ :=
let p_blue := (9 : ℝ) / 15
let p_red := (6 : ℝ) / 15
let specific_sequence_prob := p_blue ^ 5 * p_red ^ 3
let number_of_ways := (Nat.choose 8 5 : ℝ)
(number_of_ways * specific_sequence_prob)

theorem probability_five_blue_marbles_is_correct :
  probability_of_five_blue_marbles = 0.279 := by
sorry

end NUMINAMATH_GPT_probability_five_blue_marbles_is_correct_l334_33452


namespace NUMINAMATH_GPT_remainder_when_divided_by_11_l334_33460

theorem remainder_when_divided_by_11 (n : ℕ) 
  (h1 : 10 ≤ n ∧ n < 100) 
  (h2 : n % 9 = 1) 
  (h3 : n % 10 = 3) : 
  n % 11 = 7 := 
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_11_l334_33460


namespace NUMINAMATH_GPT_card_selection_l334_33492

noncomputable def count_ways := 438400

theorem card_selection :
  let decks := 2
  let total_cards := 52 * decks
  let suits := 4
  let non_royal_count := 10 * decks
  let royal_count := 3 * decks
  let non_royal_options := non_royal_count * decks
  let royal_options := royal_count * decks
  1 * (non_royal_options)^4 + (suits.choose 1) * royal_options * (non_royal_options)^3 + (suits.choose 2) * (royal_options)^2 * (non_royal_options)^2 = count_ways :=
sorry

end NUMINAMATH_GPT_card_selection_l334_33492


namespace NUMINAMATH_GPT_part1_part2_l334_33426

def P : Set ℝ := {x | x ≥ 1 / 2 ∧ x ≤ 2}

def Q (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x + 2 > 0}

def R (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x + 2 = 0}

theorem part1 (a : ℝ) : (∃ x, x ∈ P ∧ x ∈ Q a) → a > -1 / 2 :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x, x ∈ P ∧ x ∈ R a) → a ≥ -1 / 2 ∧ a ≤ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l334_33426


namespace NUMINAMATH_GPT_ants_rice_transport_l334_33408

/-- 
Given:
  1) 12 ants can move 24 grains of rice in 6 trips.

Prove:
  How many grains of rice can 9 ants move in 9 trips?
-/
theorem ants_rice_transport :
  (9 * 9 * (24 / (12 * 6))) = 27 := 
sorry

end NUMINAMATH_GPT_ants_rice_transport_l334_33408


namespace NUMINAMATH_GPT_square_area_is_8_point_0_l334_33448

theorem square_area_is_8_point_0 (A B C D E F : ℝ) 
    (h_square : E + F = 4)
    (h_diag : 1 + 2 + 1 = 4) : 
    ∃ (s : ℝ), s^2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_square_area_is_8_point_0_l334_33448


namespace NUMINAMATH_GPT_complex_powers_i_l334_33411

theorem complex_powers_i (i : ℂ) (h : i^2 = -1) :
  (i^123 - i^321 + i^432 = -2 * i + 1) :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_complex_powers_i_l334_33411


namespace NUMINAMATH_GPT_find_number_l334_33437

theorem find_number (x : ℝ) : (x / 2 = x - 5) → x = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l334_33437


namespace NUMINAMATH_GPT_x_y_sum_l334_33469

theorem x_y_sum (x y : ℝ) 
  (h1 : (x-1)^3 + 1997*(x-1) = -1)
  (h2 : (y-1)^3 + 1997*(y-1) = 1) :
  x + y = 2 :=
sorry

end NUMINAMATH_GPT_x_y_sum_l334_33469


namespace NUMINAMATH_GPT_geo_seq_second_term_l334_33480

theorem geo_seq_second_term (b r : Real) 
  (h1 : 280 * r = b) 
  (h2 : b * r = 90 / 56) 
  (h3 : b > 0) 
  : b = 15 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_geo_seq_second_term_l334_33480


namespace NUMINAMATH_GPT_expected_number_of_adjacent_black_pairs_l334_33486

theorem expected_number_of_adjacent_black_pairs :
  let total_cards := 52
  let black_cards := 26
  let adjacent_probability := (black_cards - 1) / (total_cards - 1)
  let expected_per_black_card := black_cards * adjacent_probability / total_cards
  let expected_total := black_cards * adjacent_probability
  expected_total = 650 / 51 := 
by
  let total_cards := 52
  let black_cards := 26
  let adjacent_probability := (black_cards - 1) / (total_cards - 1)
  let expected_total := black_cards * adjacent_probability
  sorry

end NUMINAMATH_GPT_expected_number_of_adjacent_black_pairs_l334_33486


namespace NUMINAMATH_GPT_car_distribution_l334_33433

theorem car_distribution :
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let fourth_supplier := (total_cars - (first_supplier + second_supplier + third_supplier)) / 2
  let fifth_supplier := fourth_supplier
  fourth_supplier = 325000 ∧ fifth_supplier = 325000 := 
by 
  sorry

end NUMINAMATH_GPT_car_distribution_l334_33433


namespace NUMINAMATH_GPT_logan_buys_15_pounds_of_corn_l334_33481

theorem logan_buys_15_pounds_of_corn (c b : ℝ) 
    (h1 : 1.20 * c + 0.60 * b = 27) 
    (h2 : b + c = 30) : 
    c = 15.0 :=
by
  sorry

end NUMINAMATH_GPT_logan_buys_15_pounds_of_corn_l334_33481


namespace NUMINAMATH_GPT_find_xy_l334_33461

-- Defining the initial conditions
variable (x y : ℕ)

-- Defining the rectangular prism dimensions and the volume equation
def prism_volume_original : ℕ := 15 * 5 * 4 -- Volume = 300
def remaining_volume : ℕ := 120

-- The main theorem statement to prove the conditions and their solution
theorem find_xy (h1 : prism_volume_original - 5 * y * x = remaining_volume)
    (h2 : x < 4) 
    (h3 : y < 15) : 
    x = 3 ∧ y = 12 := sorry

end NUMINAMATH_GPT_find_xy_l334_33461


namespace NUMINAMATH_GPT_max_value_of_expr_l334_33423

noncomputable def max_value (t : ℕ) : ℝ := (3^t - 2*t)*t / 9^t

theorem max_value_of_expr :
  ∃ t : ℕ, max_value t = 1 / 8 :=
sorry

end NUMINAMATH_GPT_max_value_of_expr_l334_33423


namespace NUMINAMATH_GPT_y_coord_equidistant_l334_33445

theorem y_coord_equidistant (y : ℝ) :
  (dist (0, y) (-3, 0) = dist (0, y) (2, 5)) ↔ y = 2 := by
  sorry

end NUMINAMATH_GPT_y_coord_equidistant_l334_33445


namespace NUMINAMATH_GPT_well_depth_and_rope_length_l334_33471

theorem well_depth_and_rope_length (h x : ℝ) : 
  (h / 3 = x + 4) ∧ (h / 4 = x + 1) :=
sorry

end NUMINAMATH_GPT_well_depth_and_rope_length_l334_33471


namespace NUMINAMATH_GPT_abs_sum_factors_l334_33495

theorem abs_sum_factors (a b c d : ℤ) : 
  (6 * x ^ 2 + x - 12 = (a * x + b) * (c * x + d)) →
  (|a| + |b| + |c| + |d| = 12) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_abs_sum_factors_l334_33495


namespace NUMINAMATH_GPT_find_n_solution_l334_33414

def product_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.prod

theorem find_n_solution : ∃ n : ℕ, n > 0 ∧ n^2 - 17 * n + 56 = product_of_digits n ∧ n = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_n_solution_l334_33414


namespace NUMINAMATH_GPT_fraction_of_females_l334_33451

variable (participants_last_year males_last_year females_last_year males_this_year females_this_year participants_this_year : ℕ)

-- The conditions
def conditions :=
  males_last_year = 20 ∧
  participants_this_year = (110 * (participants_last_year/100)) ∧
  males_this_year = (105 * males_last_year / 100) ∧
  females_this_year = (120 * females_last_year / 100) ∧
  participants_last_year = males_last_year + females_last_year ∧
  participants_this_year = males_this_year + females_this_year

-- The proof statement
theorem fraction_of_females (h : conditions males_last_year females_last_year males_this_year females_this_year participants_last_year participants_this_year) :
  (females_this_year : ℚ) / (participants_this_year : ℚ) = 4 / 11 :=
  sorry

end NUMINAMATH_GPT_fraction_of_females_l334_33451
