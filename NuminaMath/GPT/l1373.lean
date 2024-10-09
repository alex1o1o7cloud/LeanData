import Mathlib

namespace marble_weight_l1373_137389

theorem marble_weight (W : ℝ) (h : 2 * W + 0.08333333333333333 = 0.75) : 
  W = 0.33333333333333335 := 
by 
  -- Skipping the proof as specified
  sorry

end marble_weight_l1373_137389


namespace percentage_of_boys_is_60_percent_l1373_137319

-- Definition of the problem conditions
def totalPlayers := 50
def juniorGirls := 10
def half (n : ℕ) := n / 2
def girls := 2 * juniorGirls
def boys := totalPlayers - girls
def percentage_of_boys := (boys * 100) / totalPlayers

-- The theorem stating the proof problem
theorem percentage_of_boys_is_60_percent : percentage_of_boys = 60 := 
by 
  -- Proof omitted
  sorry

end percentage_of_boys_is_60_percent_l1373_137319


namespace Esha_behind_Anusha_l1373_137316

/-- Define conditions for the race -/

def Anusha_speed := 100
def Banu_behind_when_Anusha_finishes := 10
def Banu_run_when_Anusha_finishes := Anusha_speed - Banu_behind_when_Anusha_finishes
def Esha_behind_when_Banu_finishes := 10
def Esha_run_when_Banu_finishes := Anusha_speed - Esha_behind_when_Banu_finishes
def Banu_speed_ratio := Banu_run_when_Anusha_finishes / Anusha_speed
def Esha_speed_ratio := Esha_run_when_Banu_finishes / Anusha_speed
def Esha_to_Anusha_speed_ratio := Esha_speed_ratio * Banu_speed_ratio
def Esha_run_when_Anusha_finishes := Anusha_speed * Esha_to_Anusha_speed_ratio

/-- Prove that Esha is 19 meters behind Anusha when Anusha finishes the race -/
theorem Esha_behind_Anusha {V_A V_B V_E : ℝ} :
  (V_B / V_A = 9 / 10) →
  (V_E / V_B = 9 / 10) →
  (Esha_run_when_Anusha_finishes = Anusha_speed * (9 / 10 * 9 / 10)) →
  Anusha_speed - Esha_run_when_Anusha_finishes = 19 := 
by
  intros h1 h2 h3
  sorry

end Esha_behind_Anusha_l1373_137316


namespace double_even_l1373_137353

-- Define even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Lean statement of the mathematically equivalent proof problem
theorem double_even (f : ℝ → ℝ) (h : is_even_function f) : is_even_function (f ∘ f) :=
by
  sorry

end double_even_l1373_137353


namespace find_x_l1373_137398

theorem find_x (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y + x * y = 119) : x = 39 :=
sorry

end find_x_l1373_137398


namespace breadth_of_rectangular_plot_l1373_137307

variable (A b l : ℝ)

theorem breadth_of_rectangular_plot :
  (A = 15 * b) ∧ (l = b + 10) ∧ (A = l * b) → b = 5 :=
by
  intro h
  sorry

end breadth_of_rectangular_plot_l1373_137307


namespace part_I_part_II_part_III_l1373_137379

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem part_I (a : ℝ) : (∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f a x ≥ f a 1) ↔ a ≥ -1/2 :=
by
  sorry

theorem part_II : ∀ x : ℝ, f (-Real.exp 1) x + 2 ≤ 0 :=
by
  sorry

theorem part_III : ¬ ∃ x : ℝ, |f (-Real.exp 1) x| = Real.log x / x + 3 / 2 :=
by
  sorry

end part_I_part_II_part_III_l1373_137379


namespace largest_satisfying_n_correct_l1373_137349
noncomputable def largest_satisfying_n : ℕ := 4

theorem largest_satisfying_n_correct :
  ∀ n x, (1 < x ∧ x < 2 ∧ 2 < x^2 ∧ x^2 < 3 ∧ 3 < x^3 ∧ x^3 < 4 ∧ 4 < x^4 ∧ x^4 < 5) 
  → n = largest_satisfying_n ∧
  ¬ (∃ x, (1 < x ∧ x < 2 ∧ 2 < x^2 ∧ x^2 < 3 ∧ 3 < x^3 ∧ x^3 < 4 ∧ 4 < x^4 ∧ x^4 < 5 ∧ 5 < x^5 ∧ x^5 < 6)) := sorry

end largest_satisfying_n_correct_l1373_137349


namespace evens_in_triangle_l1373_137325

theorem evens_in_triangle (a : ℕ → ℕ → ℕ) (h : ∀ i j, a i.succ j = (a i (j - 1) + a i j + a i (j + 1)) % 2) :
  ∀ n ≥ 2, ∃ j, a n j % 2 = 0 :=
  sorry

end evens_in_triangle_l1373_137325


namespace lisa_interest_correct_l1373_137394

noncomputable def lisa_interest : ℝ :=
  let P := 2000
  let r := 0.035
  let n := 10
  let A := P * (1 + r) ^ n
  A - P

theorem lisa_interest_correct :
  lisa_interest = 821 := by
  sorry

end lisa_interest_correct_l1373_137394


namespace question_l1373_137396

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.log x - 7

theorem question (x : ℝ) (n : ℕ) (h1 : 2 < x ∧ x < 3) (h2 : f x = 0) : n = 2 := by
  sorry

end question_l1373_137396


namespace tom_to_luke_ratio_l1373_137338

theorem tom_to_luke_ratio (Tom Luke Anthony : ℕ) 
  (hAnthony : Anthony = 44) 
  (hTom : Tom = 33) 
  (hLuke : Luke = Anthony / 4) : 
  Tom / Nat.gcd Tom Luke = 3 ∧ Luke / Nat.gcd Tom Luke = 1 := 
by
  sorry

end tom_to_luke_ratio_l1373_137338


namespace chameleons_impossible_all_white_l1373_137360

/--
On Easter Island, there are initial counts of blue (12), white (25), and red (8) chameleons.
When two chameleons of different colors meet, they both change to the third color.
Prove that it is impossible for all chameleons to become white.
--/
theorem chameleons_impossible_all_white :
  let n1 := 12 -- Blue chameleons
  let n2 := 25 -- White chameleons
  let n3 := 8  -- Red chameleons
  (∀ (n1 n2 n3 : ℕ), (n1 + n2 + n3 = 45) → 
   ∀ (k : ℕ), ∃ m1 m2 m3 : ℕ, (m1 - m2) % 3 = (n1 - n2) % 3 ∧ (m1 - m3) % 3 = (n1 - n3) % 3 ∧ 
   (m2 - m3) % 3 = (n2 - n3) % 3) → False := sorry

end chameleons_impossible_all_white_l1373_137360


namespace solve_quadratic_l1373_137375

theorem solve_quadratic : ∃ x : ℚ, 3 * x^2 + 11 * x - 20 = 0 ∧ x > 0 ∧ x = 4 / 3 :=
by
  sorry

end solve_quadratic_l1373_137375


namespace square_difference_l1373_137399

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 64) (h₂ : x * y = 12) : (x - y)^2 = 16 := by
  -- proof would go here
  sorry

end square_difference_l1373_137399


namespace work_completion_time_l1373_137334

/-- q can complete the work in 9 days, r can complete the work in 12 days, they work together
for 3 days, and p completes the remaining work in 10.000000000000002 days. Prove that
p alone can complete the work in approximately 24 days. -/
theorem work_completion_time (W : ℝ) (q : ℝ) (r : ℝ) (p : ℝ) :
  q = 9 → r = 12 → (p * 10.000000000000002 = (5 / 12) * W) →
  p = 24.000000000000004 :=
by 
  intros hq hr hp
  sorry

end work_completion_time_l1373_137334


namespace fraction_is_three_eighths_l1373_137318

theorem fraction_is_three_eighths (F N : ℝ) 
  (h1 : (4 / 5) * F * N = 24) 
  (h2 : (250 / 100) * N = 199.99999999999997) : 
  F = 3 / 8 :=
by 
  sorry

end fraction_is_three_eighths_l1373_137318


namespace triangle_angle_contradiction_l1373_137302

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (sum_angles : A + B + C = 180) : 
  (¬ (A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60)) = (A > 60 ∧ B > 60 ∧ C > 60) :=
by sorry

end triangle_angle_contradiction_l1373_137302


namespace vector_perpendicular_l1373_137304

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 3)
def vec_diff : ℝ × ℝ := (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_perpendicular :
  dot_product vec_a vec_diff = 0 := by
  sorry

end vector_perpendicular_l1373_137304


namespace compare_y_values_l1373_137313

noncomputable def parabola (x : ℝ) : ℝ := -2 * (x + 1) ^ 2 - 1

theorem compare_y_values :
  ∃ y1 y2 y3, (parabola (-3) = y1) ∧ (parabola (-2) = y2) ∧ (parabola 2 = y3) ∧ (y3 < y1) ∧ (y1 < y2) :=
by
  sorry

end compare_y_values_l1373_137313


namespace exists_m_for_n_divides_2_pow_m_plus_m_l1373_137320

theorem exists_m_for_n_divides_2_pow_m_plus_m (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, 0 < m ∧ n ∣ 2^m + m :=
sorry

end exists_m_for_n_divides_2_pow_m_plus_m_l1373_137320


namespace liquid_X_percent_in_mixed_solution_l1373_137357

theorem liquid_X_percent_in_mixed_solution (wP wQ : ℝ) (xP xQ : ℝ) (mP mQ : ℝ) :
  xP = 0.005 * wP →
  xQ = 0.015 * wQ →
  wP = 200 →
  wQ = 800 →
  13 / 1000 * 100 = 1.3 :=
by
  intros h1 h2 h3 h4
  sorry

end liquid_X_percent_in_mixed_solution_l1373_137357


namespace power_function_point_l1373_137382

theorem power_function_point (n : ℕ) (hn : 2^n = 8) : n = 3 := 
by
  sorry

end power_function_point_l1373_137382


namespace minimum_positive_period_of_f_decreasing_intervals_of_f_maximum_value_of_f_l1373_137373

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + 3 / 2

theorem minimum_positive_period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x := by sorry

theorem decreasing_intervals_of_f : ∀ k : ℤ, ∀ x : ℝ,
  (Real.pi / 6 + k * Real.pi) ≤ x ∧ x ≤ (2 * Real.pi / 3 + k * Real.pi) → ∀ y : ℝ, 
  (Real.pi / 6 + k * Real.pi) ≤ y ∧ y ≤ (2 * Real.pi / 3 + k * Real.pi) → x ≤ y → f y ≤ f x := by sorry

theorem maximum_value_of_f : ∃ k : ℤ, ∃ x : ℝ, x = (Real.pi / 6 + k * Real.pi) ∧ f x = 5 / 2 := by sorry

end minimum_positive_period_of_f_decreasing_intervals_of_f_maximum_value_of_f_l1373_137373


namespace solve_quadratic_eq_l1373_137384

theorem solve_quadratic_eq (x : ℝ) : x^2 + 2 * x - 1 = 0 ↔ (x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) :=
by
  sorry

end solve_quadratic_eq_l1373_137384


namespace longest_pencil_l1373_137340

/-- Hallway dimensions and the longest pencil problem -/
theorem longest_pencil (L : ℝ) : 
    (∃ P : ℝ, P = 3 * L) :=
sorry

end longest_pencil_l1373_137340


namespace isosceles_right_triangle_leg_length_l1373_137391

theorem isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ) 
  (h1 : m = 12) 
  (h2 : m = h / 2)
  (h3 : h = x * Real.sqrt 2) :
  x = 12 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_leg_length_l1373_137391


namespace greatest_divisible_by_11_l1373_137365

theorem greatest_divisible_by_11 :
  ∃ (A B C : ℕ), A ≠ C ∧ A ≠ B ∧ B ≠ C ∧ 
  (∀ n, n = 10000 * A + 1000 * B + 100 * C + 10 * B + A → n = 96569) ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 11 = 0 :=
sorry

end greatest_divisible_by_11_l1373_137365


namespace example_inequality_l1373_137392

variable (a b c : ℝ)

theorem example_inequality 
  (h : a^6 + b^6 + c^6 = 3) : a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := 
by
  sorry

end example_inequality_l1373_137392


namespace seated_men_l1373_137362

def passengers : Nat := 48
def fraction_of_women : Rat := 2/3
def fraction_of_men_standing : Rat := 1/8

theorem seated_men (men women standing seated : Nat) 
  (h1 : women = passengers * fraction_of_women)
  (h2 : men = passengers - women)
  (h3 : standing = men * fraction_of_men_standing)
  (h4 : seated = men - standing) :
  seated = 14 := by
  sorry

end seated_men_l1373_137362


namespace c_seq_formula_l1373_137309

def x_seq (n : ℕ) : ℕ := 2 * n - 1
def y_seq (n : ℕ) : ℕ := n ^ 2
def c_seq (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem c_seq_formula (n : ℕ) : ∀ k, (c_seq k) = (2 * k - 1) ^ 2 :=
by
  sorry

end c_seq_formula_l1373_137309


namespace system_solution_unique_l1373_137327

theorem system_solution_unique : 
  ∀ (x y z : ℝ),
  (4 * x^2) / (1 + 4 * x^2) = y ∧
  (4 * y^2) / (1 + 4 * y^2) = z ∧
  (4 * z^2) / (1 + 4 * z^2) = x 
  → (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solution_unique_l1373_137327


namespace find_angle_C_l1373_137369

noncomputable def ABC_triangle (A B C a b c : ℝ) : Prop :=
b = c * Real.cos A + Real.sqrt 3 * a * Real.sin C

theorem find_angle_C (A B C a b c : ℝ) (h : ABC_triangle A B C a b c) :
  C = π / 6 :=
sorry

end find_angle_C_l1373_137369


namespace factorization_quad_l1373_137312

theorem factorization_quad (c d : ℕ) (h_factor : (x^2 - 18 * x + 77 = (x - c) * (x - d)))
  (h_nonneg : c ≥ 0 ∧ d ≥ 0) (h_lt : c > d) : 4 * d - c = 17 := by
  sorry

end factorization_quad_l1373_137312


namespace problem1_problem2_problem3_l1373_137381

-- Prove \(2x = 4\) is a "difference solution equation"
theorem problem1 (x : ℝ) : (2 * x = 4) → x = 4 - 2 :=
by
  sorry

-- Given \(4x = ab + a\) is a "difference solution equation", prove \(3(ab + a) = 16\)
theorem problem2 (x ab a : ℝ) : (4 * x = ab + a) → 3 * (ab + a) = 16 :=
by
  sorry

-- Given \(4x = mn + m\) and \(-2x = mn + n\) are both "difference solution equations", prove \(3(mn + m) - 9(mn + n)^2 = 0\)
theorem problem3 (x mn m n : ℝ) :
  (4 * x = mn + m) ∧ (-2 * x = mn + n) → 3 * (mn + m) - 9 * (mn + n)^2 = 0 :=
by
  sorry

end problem1_problem2_problem3_l1373_137381


namespace values_of_a_and_b_l1373_137370

theorem values_of_a_and_b (a b : ℝ) 
  (hT : (2, 1) ∈ {p : ℝ × ℝ | ∃ (a : ℝ), p.1 * a + p.2 - 3 = 0})
  (hS : (2, 1) ∈ {p : ℝ × ℝ | ∃ (b : ℝ), p.1 - p.2 - b = 0}) :
  a = 1 ∧ b = 1 :=
by
  sorry

end values_of_a_and_b_l1373_137370


namespace isosceles_triangle_area_l1373_137330

theorem isosceles_triangle_area (PQ PR QR : ℝ) (PS : ℝ) (h1 : PQ = PR)
  (h2 : QR = 10) (h3 : PS^2 + (QR / 2)^2 = PQ^2) : 
  (1/2) * QR * PS = 60 :=
by
  sorry

end isosceles_triangle_area_l1373_137330


namespace sum_of_cubes_decomposition_l1373_137372

theorem sum_of_cubes_decomposition :
  ∃ a b c d e : ℤ, (∀ x : ℤ, 1728 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ (a + b + c + d + e = 132) :=
by
  sorry

end sum_of_cubes_decomposition_l1373_137372


namespace age_solution_l1373_137355

noncomputable def age_problem : Prop :=
  ∃ (A B x : ℕ),
    A = B + 5 ∧
    A + B = 13 ∧
    3 * (A + x) = 4 * (B + x) ∧
    x = 11

theorem age_solution : age_problem :=
  sorry

end age_solution_l1373_137355


namespace range_of_a_l1373_137335

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
  (∀ x, |x^3 - a * x^2| = x → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) →
  a > 2 :=
by
  -- The proof is to be provided here.
  sorry

end range_of_a_l1373_137335


namespace cos_squared_identity_l1373_137336

theorem cos_squared_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * Real.cos (π / 6 + α / 2) ^ 2 + 1 = 7 / 3 := 
by
    sorry

end cos_squared_identity_l1373_137336


namespace simplify_expression_l1373_137331

theorem simplify_expression (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - a^2) / (b^2 - a^2) =
  (a^3 - 3 * a * b^2 + 2 * b^3) / (a * b * (b + a)) :=
by
  sorry

end simplify_expression_l1373_137331


namespace norm_of_5v_l1373_137305

noncomputable def norm_scale (v : ℝ × ℝ) (c : ℝ) : ℝ := c * (Real.sqrt (v.1^2 + v.2^2))

theorem norm_of_5v (v : ℝ × ℝ) (h : Real.sqrt (v.1^2 + v.2^2) = 6) : norm_scale v 5 = 30 := by
  sorry

end norm_of_5v_l1373_137305


namespace ratio_of_areas_l1373_137337

theorem ratio_of_areas (OR : ℝ) (h : OR > 0) :
  let OY := (1 / 3) * OR
  let area_OY := π * OY^2
  let area_OR := π * OR^2
  (area_OY / area_OR) = (1 / 9) :=
by
  -- Definitions
  let OY := (1 / 3) * OR
  let area_OY := π * OY^2
  let area_OR := π * OR^2
  sorry

end ratio_of_areas_l1373_137337


namespace ratio_x_to_y_is_12_l1373_137371

noncomputable def ratio_x_y (x y : ℝ) (h1 : y = x * (1 - 0.9166666666666666)) : ℝ := x / y

theorem ratio_x_to_y_is_12 (x y : ℝ) (h1 : y = x * (1 - 0.9166666666666666)) : ratio_x_y x y h1 = 12 :=
sorry

end ratio_x_to_y_is_12_l1373_137371


namespace M_intersection_N_eq_N_l1373_137366

def M := { x : ℝ | x < 4 }
def N := { x : ℝ | x ≤ -2 }

theorem M_intersection_N_eq_N : M ∩ N = N :=
by
  sorry

end M_intersection_N_eq_N_l1373_137366


namespace base5_representation_three_consecutive_digits_l1373_137324

theorem base5_representation_three_consecutive_digits :
  ∃ (digits : ℕ), 
    (digits = 3) ∧ 
    (∃ (a1 a2 a3 : ℕ), 
      94 = a1 * 5^2 + a2 * 5^1 + a3 * 5^0 ∧
      a1 = 3 ∧ a2 = 3 ∧ a3 = 4 ∧
      (a1 = a3 + 1) ∧ (a2 = a3 + 2)) := 
    sorry

end base5_representation_three_consecutive_digits_l1373_137324


namespace carol_packs_l1373_137347

theorem carol_packs (invitations_per_pack total_invitations packs_bought : ℕ) 
  (h1 : invitations_per_pack = 9)
  (h2 : total_invitations = 45) 
  (h3 : packs_bought = total_invitations / invitations_per_pack) : 
  packs_bought = 5 :=
by 
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end carol_packs_l1373_137347


namespace perfect_square_trinomial_k_l1373_137346

theorem perfect_square_trinomial_k (a k : ℝ) : (∃ b : ℝ, (a - b)^2 = a^2 - ka + 25) ↔ k = 10 ∨ k = -10 := 
sorry

end perfect_square_trinomial_k_l1373_137346


namespace multiples_of_three_l1373_137352

theorem multiples_of_three (a b : ℤ) (h : 9 ∣ (a^2 + a * b + b^2)) : 3 ∣ a ∧ 3 ∣ b :=
by {
  sorry
}

end multiples_of_three_l1373_137352


namespace find_y_l1373_137317

variable (x y : ℤ)

-- Conditions
def cond1 : Prop := x + y = 280
def cond2 : Prop := x - y = 200

-- Proof statement
theorem find_y (h1 : cond1 x y) (h2 : cond2 x y) : y = 40 := 
by 
  sorry

end find_y_l1373_137317


namespace vector_expression_identity_l1373_137339

variables (E : Type) [AddCommGroup E] [Module ℝ E]
variables (e1 e2 : E)
variables (a b : E)
variables (cond1 : a = (3 : ℝ) • e1 - (2 : ℝ) • e2) (cond2 : b = (e2 - (2 : ℝ) • e1))

theorem vector_expression_identity :
  (1 / 3 : ℝ) • a + b + a - (3 / 2 : ℝ) • b + 2 • b - a = -2 • e1 + (5 / 6 : ℝ) • e2 :=
sorry

end vector_expression_identity_l1373_137339


namespace div_fraction_eq_l1373_137387

theorem div_fraction_eq :
  (5 / 3) / (1 / 4) = 20 / 3 := 
by
  sorry

end div_fraction_eq_l1373_137387


namespace deepak_present_age_l1373_137315

-- We start with the conditions translated into Lean definitions.

variables (R D : ℕ)

-- Condition 1: The ratio between Rahul's and Deepak's ages is 4:3.
def age_ratio := R * 3 = D * 4

-- Condition 2: After 6 years, Rahul's age will be 38 years.
def rahul_future_age := R + 6 = 38

-- The goal is to prove that D = 24 given the above conditions.
theorem deepak_present_age 
  (h1: age_ratio R D) 
  (h2: rahul_future_age R) : D = 24 :=
sorry

end deepak_present_age_l1373_137315


namespace custom_op_4_2_l1373_137308

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem to prove the result
theorem custom_op_4_2 : custom_op 4 2 = 24 :=
by
  sorry

end custom_op_4_2_l1373_137308


namespace minimum_stamps_satisfying_congruences_l1373_137358

theorem minimum_stamps_satisfying_congruences (n : ℕ) :
  (n % 4 = 3) ∧ (n % 5 = 2) ∧ (n % 7 = 1) → n = 107 :=
by
  sorry

end minimum_stamps_satisfying_congruences_l1373_137358


namespace pond_capacity_l1373_137368

theorem pond_capacity :
  let normal_rate := 6 -- gallons per minute
  let restriction_rate := (2/3 : ℝ) * normal_rate -- gallons per minute
  let time := 50 -- minutes
  let capacity := restriction_rate * time -- total capacity in gallons
  capacity = 200 := sorry

end pond_capacity_l1373_137368


namespace find_x_add_inv_l1373_137311

theorem find_x_add_inv (x : ℝ) (h : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 :=
sorry

end find_x_add_inv_l1373_137311


namespace cut_scene_length_l1373_137306

theorem cut_scene_length (original_length final_length : ℕ) (h1 : original_length = 60) (h2 : final_length = 52) : original_length - final_length = 8 := 
by 
  sorry

end cut_scene_length_l1373_137306


namespace bike_price_l1373_137395

variable (p : ℝ)

def percent_upfront_payment : ℝ := 0.20
def upfront_payment : ℝ := 200

theorem bike_price (h : percent_upfront_payment * p = upfront_payment) : p = 1000 := by
  sorry

end bike_price_l1373_137395


namespace arithmetic_progression_common_difference_zero_l1373_137332

theorem arithmetic_progression_common_difference_zero {a d : ℤ} (h₁ : a = 12) 
  (h₂ : ∀ n : ℕ, a + n * d = (a + (n + 1) * d + a + (n + 2) * d) / 2) : d = 0 :=
  sorry

end arithmetic_progression_common_difference_zero_l1373_137332


namespace BKING_2023_reappears_at_20_l1373_137326

-- Defining the basic conditions of the problem
def cycle_length_BKING : ℕ := 5
def cycle_length_2023 : ℕ := 4

-- Formulating the proof problem statement
theorem BKING_2023_reappears_at_20 :
  Nat.lcm cycle_length_BKING cycle_length_2023 = 20 :=
by
  sorry

end BKING_2023_reappears_at_20_l1373_137326


namespace least_positive_integer_l1373_137378

theorem least_positive_integer (n : ℕ) :
  (∃ n : ℕ, 25^n + 16^n ≡ 1 [MOD 121] ∧ ∀ m : ℕ, (m < n ∧ 25^m + 16^m ≡ 1 [MOD 121]) → false) ↔ n = 32 :=
sorry

end least_positive_integer_l1373_137378


namespace inequality_solution_set_l1373_137383

theorem inequality_solution_set :
  (∀ x : ℝ, (3 * x - 2 < 2 * (x + 1) ∧ (x - 1) / 2 > 1) ↔ (3 < x ∧ x < 4)) :=
by
  sorry

end inequality_solution_set_l1373_137383


namespace cdf_of_Z_pdf_of_Z_l1373_137374

noncomputable def f1 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then 0.5 else 0

noncomputable def f2 (y : ℝ) : ℝ :=
  if 0 < y ∧ y < 2 then 0.5 else 0

noncomputable def G (z : ℝ) : ℝ :=
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z^2 / 8
  else if 2 < z ∧ z ≤ 4 then 1 - (4 - z)^2 / 8
  else 1

noncomputable def g (z : ℝ) : ℝ :=
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z / 4
  else if 2 < z ∧ z ≤ 4 then 1 - z / 4
  else 0

theorem cdf_of_Z (z : ℝ) : G z = 
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z^2 / 8
  else if 2 < z ∧ z ≤ 4 then 1 - (4 - z)^2 / 8
  else 1 := sorry

theorem pdf_of_Z (z : ℝ) : g z = 
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z / 4
  else if 2 < z ∧ z ≤ 4 then 1 - z / 4
  else 0 := sorry

end cdf_of_Z_pdf_of_Z_l1373_137374


namespace melanie_food_total_weight_l1373_137301

def total_weight (brie_oz : ℕ) (bread_lb : ℕ) (tomatoes_lb : ℕ) (zucchini_lb : ℕ) 
           (chicken_lb : ℕ) (raspberries_oz : ℕ) (blueberries_oz : ℕ) : ℕ :=
  let brie_lb := brie_oz / 16
  let raspberries_lb := raspberries_oz / 16
  let blueberries_lb := blueberries_oz / 16
  brie_lb + raspberries_lb + blueberries_lb + bread_lb + tomatoes_lb + zucchini_lb + chicken_lb

theorem melanie_food_total_weight : total_weight 8 1 1 2 (3 / 2) 8 8 = 7 :=
by
  -- result placeholder
  sorry

end melanie_food_total_weight_l1373_137301


namespace sqrt_mult_simplify_l1373_137385

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l1373_137385


namespace stratified_sampling_third_grade_l1373_137390

theorem stratified_sampling_third_grade (total_students : ℕ)
  (ratio_first_second_third : ℕ × ℕ × ℕ)
  (sample_size : ℕ) (r1 r2 r3 : ℕ) (h_ratio : ratio_first_second_third = (r1, r2, r3)) :
  total_students = 3000  ∧ ratio_first_second_third = (2, 3, 1)  ∧ sample_size = 180 →
  (sample_size * r3 / (r1 + r2 + r3) = 30) :=
sorry

end stratified_sampling_third_grade_l1373_137390


namespace find_alpha_l1373_137343

theorem find_alpha (α : ℝ) (h0 : 0 ≤ α) (h1 : α < 360)
    (h_point : (Real.sin 215) = (Real.sin α) ∧ (Real.cos 215) = (Real.cos α)) :
    α = 235 :=
sorry

end find_alpha_l1373_137343


namespace evan_45_l1373_137367

theorem evan_45 (k n : ℤ) (h1 : n + (k * (2 * k - 1)) = 60) : 60 - n = 45 :=
by sorry

end evan_45_l1373_137367


namespace math_problem_l1373_137310

theorem math_problem (x : ℕ) (h : (2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x = 512)) : (x + 2) * (x - 2) = 32 :=
sorry

end math_problem_l1373_137310


namespace greatest_x_solution_l1373_137321

theorem greatest_x_solution (x : ℝ) (h₁ : (x^2 - x - 30) / (x - 6) = 2 / (x + 4)) : x ≤ -3 :=
sorry

end greatest_x_solution_l1373_137321


namespace probability_C_D_l1373_137356

variable (P : String → ℚ)

axiom h₁ : P "A" = 1/4
axiom h₂ : P "B" = 1/3
axiom h₃ : P "A" + P "B" + P "C" + P "D" = 1

theorem probability_C_D : P "C" + P "D" = 5/12 := by
  sorry

end probability_C_D_l1373_137356


namespace janice_weekly_earnings_l1373_137380

-- define the conditions
def regular_days_per_week : Nat := 5
def regular_earnings_per_day : Nat := 30
def overtime_earnings_per_shift : Nat := 15
def overtime_shifts_per_week : Nat := 3

-- define the total earnings calculation
def total_earnings (regular_days : Nat) (regular_rate : Nat) (overtime_shifts : Nat) (overtime_rate : Nat) : Nat :=
  (regular_days * regular_rate) + (overtime_shifts * overtime_rate)

-- state the problem to be proved
theorem janice_weekly_earnings : total_earnings regular_days_per_week regular_earnings_per_day overtime_shifts_per_week overtime_earnings_per_shift = 195 :=
by
  sorry

end janice_weekly_earnings_l1373_137380


namespace difference_of_triangular_23_and_21_l1373_137348

def triangular (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem difference_of_triangular_23_and_21 : triangular 23 - triangular 21 = 45 :=
sorry

end difference_of_triangular_23_and_21_l1373_137348


namespace a5_a6_less_than_a4_squared_l1373_137386

variable {a : ℕ → ℝ}
variable {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem a5_a6_less_than_a4_squared
  (h_geo : is_geometric_sequence a q)
  (h_cond : a 5 * a 6 < (a 4) ^ 2) :
  0 < q ∧ q < 1 :=
sorry

end a5_a6_less_than_a4_squared_l1373_137386


namespace remaining_days_temperature_l1373_137342

theorem remaining_days_temperature (avg_temp : ℕ) (d1 d2 d3 d4 d5 : ℕ) :
  avg_temp = 60 →
  d1 = 40 →
  d2 = 40 →
  d3 = 40 →
  d4 = 80 →
  d5 = 80 →
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  total_temp - known_temp = 140 := 
by
  intros _ _ _ _ _ _
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  sorry

end remaining_days_temperature_l1373_137342


namespace probability_of_selecting_particular_girl_l1373_137322

-- Define the numbers involved
def total_population : ℕ := 60
def num_girls : ℕ := 25
def num_boys : ℕ := 35
def sample_size : ℕ := 5

-- Total number of basic events
def total_combinations : ℕ := Nat.choose total_population sample_size

-- Number of basic events that include a particular girl
def girl_combinations : ℕ := Nat.choose (total_population - 1) (sample_size - 1)

-- Probability of selecting a particular girl
def probability_of_girl_selection : ℚ := girl_combinations / total_combinations

-- The theorem to be proved
theorem probability_of_selecting_particular_girl :
  probability_of_girl_selection = 1 / 12 :=
by sorry

end probability_of_selecting_particular_girl_l1373_137322


namespace correct_statement_dice_roll_l1373_137328

theorem correct_statement_dice_roll :
  (∃! s, s ∈ ["When flipping a coin, the head side will definitely face up.",
              "The probability of precipitation tomorrow is 80% means that 80% of the areas will have rain tomorrow.",
              "To understand the lifespan of a type of light bulb, it is appropriate to use a census method.",
              "When rolling a dice, the number will definitely not be greater than 6."] ∧
          s = "When rolling a dice, the number will definitely not be greater than 6.") :=
by {
  sorry
}

end correct_statement_dice_roll_l1373_137328


namespace unique_passenger_counts_l1373_137354

def train_frequencies : Nat × Nat × Nat := (6, 4, 3)
def train_passengers_leaving : Nat × Nat × Nat := (200, 300, 150)
def train_passengers_taking : Nat × Nat × Nat := (320, 400, 280)
def trains_per_hour (freq : Nat) : Nat := 60 / freq

def total_passengers_leaving : Nat :=
  let t1 := (trains_per_hour 10) * 200
  let t2 := (trains_per_hour 15) * 300
  let t3 := (trains_per_hour 20) * 150
  t1 + t2 + t3

def total_passengers_taking : Nat :=
  let t1 := (trains_per_hour 10) * 320
  let t2 := (trains_per_hour 15) * 400
  let t3 := (trains_per_hour 20) * 280
  t1 + t2 + t3

theorem unique_passenger_counts :
  total_passengers_leaving = 2850 ∧ total_passengers_taking = 4360 := by
  sorry

end unique_passenger_counts_l1373_137354


namespace annie_budget_l1373_137388

theorem annie_budget :
  let budget := 120
  let hamburger_count := 8
  let milkshake_count := 6
  let hamburgerA := 4
  let milkshakeA := 5
  let hamburgerB := 3.5
  let milkshakeB := 6
  let hamburgerC := 5
  let milkshakeC := 4
  let costA := hamburgerA * hamburger_count + milkshakeA * milkshake_count
  let costB := hamburgerB * hamburger_count + milkshakeB * milkshake_count
  let costC := hamburgerC * hamburger_count + milkshakeC * milkshake_count
  let min_cost := min costA (min costB costC)
  budget - min_cost = 58 :=
by {
  sorry
}

end annie_budget_l1373_137388


namespace total_distance_correct_l1373_137345

def liters_U := 50
def liters_V := 50
def liters_W := 50
def liters_X := 50

def fuel_efficiency_U := 20 -- liters per 100 km
def fuel_efficiency_V := 25 -- liters per 100 km
def fuel_efficiency_W := 5 -- liters per 100 km
def fuel_efficiency_X := 10 -- liters per 100 km

def distance_U := (liters_U / fuel_efficiency_U) * 100 -- Distance for U in km
def distance_V := (liters_V / fuel_efficiency_V) * 100 -- Distance for V in km
def distance_W := (liters_W / fuel_efficiency_W) * 100 -- Distance for W in km
def distance_X := (liters_X / fuel_efficiency_X) * 100 -- Distance for X in km

def total_distance := distance_U + distance_V + distance_W + distance_X -- Total distance of all cars

theorem total_distance_correct :
  total_distance = 1950 := 
by {
  sorry
}

end total_distance_correct_l1373_137345


namespace number_of_comic_books_l1373_137300

def fairy_tale_books := 305
def science_and_technology_books := fairy_tale_books + 115
def total_books := fairy_tale_books + science_and_technology_books
def comic_books := total_books * 4

theorem number_of_comic_books : comic_books = 2900 := by
  sorry

end number_of_comic_books_l1373_137300


namespace candy_bar_sugar_calories_l1373_137364

theorem candy_bar_sugar_calories
  (candy_bars : Nat)
  (soft_drink_calories : Nat)
  (soft_drink_sugar_percentage : Float)
  (recommended_sugar_intake : Nat)
  (excess_percentage : Nat)
  (sugar_in_each_bar : Nat) :
  candy_bars = 7 ∧
  soft_drink_calories = 2500 ∧
  soft_drink_sugar_percentage = 0.05 ∧
  recommended_sugar_intake = 150 ∧
  excess_percentage = 100 →
  sugar_in_each_bar = 25 := by
  sorry

end candy_bar_sugar_calories_l1373_137364


namespace tetrahedron_edge_length_l1373_137363

-- Define the problem specifications
def mutuallyTangent (r : ℝ) (a b c d : ℝ → ℝ → ℝ → Prop) :=
  a = b ∧ a = c ∧ a = d ∧ b = c ∧ b = d ∧ c = d

noncomputable def tetrahedronEdgeLength (r : ℝ) : ℝ :=
  2 + 2 * Real.sqrt 6

-- Proof goal: edge length of tetrahedron containing four mutually tangent balls each of radius 1
theorem tetrahedron_edge_length (r : ℝ) (a b c d : ℝ → ℝ → ℝ → Prop)
  (h1 : r = 1)
  (h2 : mutuallyTangent r a b c d)
  : tetrahedronEdgeLength r = 2 + 2 * Real.sqrt 6 :=
sorry

end tetrahedron_edge_length_l1373_137363


namespace question_1_question_2_l1373_137303

-- Condition: The coordinates of point P are given by the equations x = -3a - 4, y = 2 + a

-- Question 1: Prove coordinates when P lies on the x-axis
theorem question_1 (a : ℝ) (x : ℝ) (y : ℝ) (h1 : x = -3 * a - 4) (h2 : y = 2 + a) (hy0 : y = 0) :
  a = -2 ∧ x = 2 ∧ y = 0 :=
sorry

-- Question 2: Prove coordinates when PQ is parallel to the y-axis
theorem question_2 (a : ℝ) (x : ℝ) (y : ℝ) (h1 : x = -3 * a - 4) (h2 : y = 2 + a) (hx5 : x = 5) :
  a = -3 ∧ x = 5 ∧ y = -1 :=
sorry

end question_1_question_2_l1373_137303


namespace pencils_per_student_l1373_137351

theorem pencils_per_student
  (boxes : ℝ) (pencils_per_box : ℝ) (students : ℝ)
  (h1 : boxes = 4.0)
  (h2 : pencils_per_box = 648.0)
  (h3 : students = 36.0) :
  (boxes * pencils_per_box) / students = 72.0 :=
by
  sorry

end pencils_per_student_l1373_137351


namespace find_x_plus_y_l1373_137323

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y = 2009) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := 
by
  sorry

end find_x_plus_y_l1373_137323


namespace find_y_l1373_137350

theorem find_y
  (x y : ℝ)
  (h1 : x - y = 10)
  (h2 : x + y = 8) : y = -1 :=
by
  sorry

end find_y_l1373_137350


namespace cylinder_original_radius_l1373_137333

theorem cylinder_original_radius 
  (r h : ℝ) 
  (hr_eq : h = 3)
  (volume_increase_radius : Real.pi * (r + 8)^2 * 3 = Real.pi * r^2 * 11) :
  r = 8 :=
by
  -- the proof steps will be here
  sorry

end cylinder_original_radius_l1373_137333


namespace cylinder_volume_increase_l1373_137361

theorem cylinder_volume_increase 
  (r h : ℝ) 
  (V : ℝ := π * r^2 * h) 
  (new_h : ℝ := 3 * h) 
  (new_r : ℝ := 2 * r) : 
  (π * new_r^2 * new_h) = 12 * V := 
by
  sorry

end cylinder_volume_increase_l1373_137361


namespace circle_diameter_C_l1373_137377

theorem circle_diameter_C {D C : ℝ} (hD : D = 20) (h_ratio : (π * (D/2)^2 - π * (C/2)^2) / (π * (C/2)^2) = 4) : C = 4 * Real.sqrt 5 := 
sorry

end circle_diameter_C_l1373_137377


namespace area_ratio_eq_two_l1373_137376

/-- 
  Given a unit square, let circle B be the inscribed circle and circle A be the circumscribed circle.
  Prove the ratio of the area of circle A to the area of circle B is 2.
--/
theorem area_ratio_eq_two (r_B r_A : ℝ) (hB : r_B = 1 / 2) (hA : r_A = Real.sqrt 2 / 2):
  (π * r_A ^ 2) / (π * r_B ^ 2) = 2 := by
  sorry

end area_ratio_eq_two_l1373_137376


namespace tan_theta_correct_l1373_137393

noncomputable def cos_double_angle (θ : ℝ) : ℝ := 2 * Real.cos θ ^ 2 - 1

theorem tan_theta_correct (θ : ℝ) (hθ₁ : θ > 0) (hθ₂ : θ < Real.pi / 2) 
  (h : 15 * cos_double_angle θ - 14 * Real.cos θ + 11 = 0) : Real.tan θ = Real.sqrt 5 / 2 :=
sorry

end tan_theta_correct_l1373_137393


namespace fraction_of_green_balls_l1373_137344

theorem fraction_of_green_balls (T G : ℝ)
    (h1 : (1 / 8) * T = 6)
    (h2 : (1 / 12) * T + (1 / 8) * T + 26 = T - G)
    (h3 : (1 / 8) * T = 6)
    (h4 : 26 ≥ 0):
  G / T = 1 / 4 :=
by
  sorry

end fraction_of_green_balls_l1373_137344


namespace ordered_pair_proportional_l1373_137341

theorem ordered_pair_proportional (p q : ℝ) (h : (3 : ℝ) • (-4 : ℝ) = (5 : ℝ) • p ∧ (3 : ℝ) • q = (5 : ℝ) • (-4 : ℝ)) :
  (p, q) = (5 / 2, -8) :=
by
  sorry

end ordered_pair_proportional_l1373_137341


namespace minimum_pencils_l1373_137397

-- Define the given conditions
def red_pencils : ℕ := 15
def blue_pencils : ℕ := 13
def green_pencils : ℕ := 8

-- Define the requirement for pencils to ensure the conditions are met
def required_red : ℕ := 1
def required_blue : ℕ := 2
def required_green : ℕ := 3

-- The minimum number of pencils Constanza should take out
noncomputable def minimum_pencils_to_ensure : ℕ := 21 + 1

theorem minimum_pencils (red_pencils blue_pencils green_pencils : ℕ)
    (required_red required_blue required_green minimum_pencils_to_ensure : ℕ) :
    red_pencils = 15 →
    blue_pencils = 13 →
    green_pencils = 8 →
    required_red = 1 →
    required_blue = 2 →
    required_green = 3 →
    minimum_pencils_to_ensure = 22 :=
by
    intros h1 h2 h3 h4 h5 h6
    sorry

end minimum_pencils_l1373_137397


namespace range_of_f_l1373_137329

def f (x : ℤ) : ℤ := x ^ 2 - 2 * x
def domain : Set ℤ := {0, 1, 2, 3}
def expectedRange : Set ℤ := {-1, 0, 3}

theorem range_of_f : (Set.image f domain) = expectedRange :=
  sorry

end range_of_f_l1373_137329


namespace cat_daytime_catches_l1373_137359

theorem cat_daytime_catches
  (D : ℕ)
  (night_catches : ℕ := 2 * D)
  (total_catches : ℕ := D + night_catches)
  (h : total_catches = 24) :
  D = 8 := by
  sorry

end cat_daytime_catches_l1373_137359


namespace solve_chimney_bricks_l1373_137314

noncomputable def chimney_bricks (x : ℝ) : Prop :=
  let brenda_rate := x / 8
  let brandon_rate := x / 12
  let combined_rate := brenda_rate + brandon_rate - 15
  (combined_rate * 6) = x

theorem solve_chimney_bricks : ∃ (x : ℝ), chimney_bricks x ∧ x = 360 :=
by
  use 360
  unfold chimney_bricks
  sorry

end solve_chimney_bricks_l1373_137314
