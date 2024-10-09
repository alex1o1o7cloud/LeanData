import Mathlib

namespace dogs_for_sale_l1945_194526

variable (D : ℕ)
def number_of_cats := D / 2
def number_of_birds := 2 * D
def number_of_fish := 3 * D
def total_animals := D + number_of_cats D + number_of_birds D + number_of_fish D

theorem dogs_for_sale (h : total_animals D = 39) : D = 6 :=
by
  sorry

end dogs_for_sale_l1945_194526


namespace problem_evaluation_l1945_194543

theorem problem_evaluation (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (ab + bc + cd + da + ac + bd)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (cd)⁻¹ + (da)⁻¹ + (ac)⁻¹ + (bd)⁻¹) = 
  (1 / (a * b * c * d)) * (1 / (a * b * c * d)) :=
by
  sorry

end problem_evaluation_l1945_194543


namespace angle_W_in_quadrilateral_l1945_194567

theorem angle_W_in_quadrilateral 
  (W X Y Z : ℝ) 
  (h₀ : W + X + Y + Z = 360) 
  (h₁ : W = 3 * X) 
  (h₂ : W = 4 * Y) 
  (h₃ : W = 6 * Z) : 
  W = 206 :=
by
  sorry

end angle_W_in_quadrilateral_l1945_194567


namespace solve_equation1_solve_equation2_l1945_194527

def equation1 (x : ℝ) : Prop := 3 * x^2 + 2 * x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 2) * (x - 1) = 2 - 2 * x

theorem solve_equation1 :
  (equation1 (-1) ∨ equation1 (1 / 3)) ∧ 
  (∀ x, equation1 x → x = -1 ∨ x = 1 / 3) :=
sorry

theorem solve_equation2 :
  (equation2 1 ∨ equation2 (-4)) ∧ 
  (∀ x, equation2 x → x = 1 ∨ x = -4) :=
sorry

end solve_equation1_solve_equation2_l1945_194527


namespace johns_allowance_is_3_45_l1945_194577

noncomputable def johns_weekly_allowance (A : ℝ) : Prop :=
  -- Condition 1: John spent 3/5 of his allowance at the arcade
  let spent_at_arcade := (3/5) * A
  -- Remaining allowance
  let remaining_after_arcade := A - spent_at_arcade
  -- Condition 2: He spent 1/3 of the remaining allowance at the toy store
  let spent_at_toy_store := (1/3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - spent_at_toy_store
  -- Condition 3: He spent his last $0.92 at the candy store
  let spent_at_candy_store := 0.92
  -- Remaining amount after the candy store expenditure should be 0
  remaining_after_toy_store = spent_at_candy_store

theorem johns_allowance_is_3_45 : johns_weekly_allowance 3.45 :=
sorry

end johns_allowance_is_3_45_l1945_194577


namespace complex_number_real_imag_equal_l1945_194588

theorem complex_number_real_imag_equal (a : ℝ) (h : (a + 6) = (3 - 2 * a)) : a = -1 :=
by
  sorry

end complex_number_real_imag_equal_l1945_194588


namespace simplify_fraction_l1945_194505

theorem simplify_fraction (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by
  sorry

end simplify_fraction_l1945_194505


namespace ice_cream_flavors_l1945_194551

theorem ice_cream_flavors (n k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n + k - 1).choose (k - 1) = 84 :=
by
  have h3 : n = 6 := h1
  have h4 : k = 4 := h2
  rw [h3, h4]
  sorry

end ice_cream_flavors_l1945_194551


namespace line_of_intersecting_circles_l1945_194598

theorem line_of_intersecting_circles
  (A B : ℝ × ℝ)
  (hAB1 : A.1^2 + A.2^2 + 4 * A.1 - 4 * A.2 = 0)
  (hAB2 : B.1^2 + B.2^2 + 4 * B.1 - 4 * B.2 = 0)
  (hAB3 : A.1^2 + A.2^2 + 2 * A.1 - 12 = 0)
  (hAB4 : B.1^2 + B.2^2 + 2 * B.1 - 12 = 0) :
  ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧ a * B.1 + b * B.2 + c = 0 ∧
                  a = 1 ∧ b = -2 ∧ c = 6 :=
sorry

end line_of_intersecting_circles_l1945_194598


namespace unique_solution_conditions_l1945_194555

-- Definitions based on the conditions
variables {x y a : ℝ}

def inequality_condition (x y a : ℝ) : Prop := 
  x^2 + y^2 + 2 * x ≤ 1

def equation_condition (x y a : ℝ) : Prop := 
  x - y = -a

-- Main Theorem Statement
theorem unique_solution_conditions (a : ℝ) : 
  (∃! x y : ℝ, inequality_condition x y a ∧ equation_condition x y a) ↔ (a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2) :=
sorry

end unique_solution_conditions_l1945_194555


namespace washing_machines_total_pounds_l1945_194518

theorem washing_machines_total_pounds (pounds_per_machine_per_day : ℕ) (number_of_machines : ℕ)
  (h1 : pounds_per_machine_per_day = 28) (h2 : number_of_machines = 8) :
  number_of_machines * pounds_per_machine_per_day = 224 :=
by
  sorry

end washing_machines_total_pounds_l1945_194518


namespace relation_between_x_and_y_l1945_194521

open Real

noncomputable def x (t : ℝ) : ℝ := t^(1 / (t - 1))
noncomputable def y (t : ℝ) : ℝ := t^(t / (t - 1))

theorem relation_between_x_and_y (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1) : (y t)^(x t) = (x t)^(y t) :=
by sorry

end relation_between_x_and_y_l1945_194521


namespace part1_part2_l1945_194554

noncomputable def f (a c x : ℝ) : ℝ :=
  if x >= c then a * Real.log x + (x - c) ^ 2
  else a * Real.log x - (x - c) ^ 2

theorem part1 (a c : ℝ)
  (h_a : a = 2 * c - 2)
  (h_c_gt_0 : c > 0)
  (h_f_geq : ∀ x, x ∈ (Set.Ioi c) → f a c x >= 1 / 4) :
    a ∈ Set.Icc (-2 : ℝ) (-1 : ℝ) :=
  sorry

theorem part2 (a c x1 x2 : ℝ)
  (h_a_lt_0 : a < 0)
  (h_c_gt_0 : c > 0)
  (h_x1 : x1 = Real.sqrt (- a / 2))
  (h_x2 : x2 = c)
  (h_tangents_intersect : deriv (f a c) x1 * deriv (f a c) x2 = -1) :
    c >= 3 * Real.sqrt 3 / 2 :=
  sorry

end part1_part2_l1945_194554


namespace roses_in_each_bouquet_l1945_194570

theorem roses_in_each_bouquet (R : ℕ)
(roses_bouquets daisies_bouquets total_bouquets total_flowers daisies_per_bouquet total_daisies : ℕ)
(h1 : total_bouquets = 20)
(h2 : roses_bouquets = 10)
(h3 : daisies_bouquets = 10)
(h4 : total_flowers = 190)
(h5 : daisies_per_bouquet = 7)
(h6 : total_daisies = daisies_bouquets * daisies_per_bouquet)
(h7 : total_flowers - total_daisies = roses_bouquets * R) :
R = 12 :=
by
  sorry

end roses_in_each_bouquet_l1945_194570


namespace remainder_of_power_mod_l1945_194514

theorem remainder_of_power_mod (a b n : ℕ) (h_prime : Nat.Prime n) (h_a_not_div : ¬ (n ∣ a)) :
  a ^ b % n = 82 :=
by
  have : n = 379 := sorry
  have : a = 6 := sorry
  have : b = 97 := sorry
  sorry

end remainder_of_power_mod_l1945_194514


namespace cubic_eq_root_nature_l1945_194594

-- Definitions based on the problem statement
def cubic_eq (x : ℝ) : Prop := x^3 + 3 * x^2 - 4 * x - 12 = 0

-- The main theorem statement
theorem cubic_eq_root_nature :
  (∃ p n₁ n₂ : ℝ, cubic_eq p ∧ cubic_eq n₁ ∧ cubic_eq n₂ ∧ p > 0 ∧ n₁ < 0 ∧ n₂ < 0 ∧ p ≠ n₁ ∧ p ≠ n₂ ∧ n₁ ≠ n₂) :=
sorry

end cubic_eq_root_nature_l1945_194594


namespace problem_statement_l1945_194536

theorem problem_statement (a b : ℝ) (h1 : 1/a < 1/b) (h2 : 1/b < 0) :
  (a + b < a * b) ∧ ¬(a^2 > b^2) ∧ ¬(a < b) ∧ (b/a + a/b > 2) := by
  sorry

end problem_statement_l1945_194536


namespace algebraic_expression_value_l1945_194508

variable (x : ℝ)

theorem algebraic_expression_value (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 2 = 4 :=
by
  -- This is where the detailed proof would go, but we are skipping it with sorry.
  sorry

end algebraic_expression_value_l1945_194508


namespace find_p_l1945_194579

variables (p q : ℚ)
variables (h1 : 2 * p + 5 * q = 10) (h2 : 5 * p + 2 * q = 20)

theorem find_p : p = 80 / 21 :=
by sorry

end find_p_l1945_194579


namespace puppies_start_count_l1945_194517

theorem puppies_start_count (x : ℕ) (given_away : ℕ) (left : ℕ) (h1 : given_away = 7) (h2 : left = 5) (h3 : x = given_away + left) : x = 12 :=
by
  rw [h1, h2] at h3
  exact h3

end puppies_start_count_l1945_194517


namespace correct_quotient_remainder_sum_l1945_194589

theorem correct_quotient_remainder_sum :
  ∃ N : ℕ, (N % 23 = 17 ∧ N / 23 = 3) ∧ (∃ q r : ℕ, N = 32 * q + r ∧ r < 32 ∧ q + r = 24) :=
by
  sorry

end correct_quotient_remainder_sum_l1945_194589


namespace pure_imaginary_condition_l1945_194583

theorem pure_imaginary_condition (a b : ℝ) : 
  (a = 0) ↔ (∃ b : ℝ, b ≠ 0 ∧ z = a + b * I) :=
sorry

end pure_imaginary_condition_l1945_194583


namespace combined_students_yellow_blue_l1945_194547

theorem combined_students_yellow_blue {total_students blue_percent red_percent yellow_combined : ℕ} :
  total_students = 200 →
  blue_percent = 30 →
  red_percent = 40 →
  yellow_combined = (total_students * 3 / 10) + ((total_students - (total_students * 3 / 10)) * 6 / 10) →
  yellow_combined = 144 :=
by
  intros
  sorry

end combined_students_yellow_blue_l1945_194547


namespace find_coefficients_l1945_194541

variables {x1 x2 x3 x4 x5 x6 x7 : ℝ}

theorem find_coefficients
  (h1 : x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 49*x7 = 5)
  (h2 : 4*x1 + 9*x2 + 16*x3 + 25*x4 + 36*x5 + 49*x6 + 64*x7 = 14)
  (h3 : 9*x1 + 16*x2 + 25*x3 + 36*x4 + 49*x5 + 64*x6 + 81*x7 = 30)
  (h4 : 16*x1 + 25*x2 + 36*x3 + 49*x4 + 64*x5 + 81*x6 + 100*x7 = 70) :
  25*x1 + 36*x2 + 49*x3 + 64*x4 + 81*x5 + 100*x6 + 121*x7 = 130 :=
sorry

end find_coefficients_l1945_194541


namespace asia_paid_140_l1945_194564

noncomputable def original_price : ℝ := 350
noncomputable def discount_percentage : ℝ := 0.60
noncomputable def discount_amount : ℝ := original_price * discount_percentage
noncomputable def final_price : ℝ := original_price - discount_amount

theorem asia_paid_140 : final_price = 140 := by
  unfold final_price
  unfold discount_amount
  unfold original_price
  unfold discount_percentage
  sorry

end asia_paid_140_l1945_194564


namespace arithmetic_sequence_problem_l1945_194597

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (a1 : ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :=
  ∀ n, a n = a1 + n * d

-- Given condition
variable (h1 : a 3 + a 4 + a 5 = 36)

-- The goal is to prove that a 0 + a 8 = 24
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :
  arithmetic_sequence a a1 d →
  a 3 + a 4 + a 5 = 36 →
  a 0 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_problem_l1945_194597


namespace extremum_f_at_neg_four_thirds_monotonicity_g_l1945_194574

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + x ^ 2
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f a x) * Real.exp x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 2 * x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 
  let f_a_x := f a x
  ( f' a x * Real.exp x ) + ( f_a_x * Real.exp x)

theorem extremum_f_at_neg_four_thirds (a : ℝ) :
  f' a (-4/3) = 0 ↔ a = 1/2 := sorry

-- Assuming a = 1/2 from the previous theorem
theorem monotonicity_g :
  let a := 1/2
  ∀ x : ℝ, 
    ((x < -4 → g' a x < 0) ∧ 
     (-4 < x ∧ x < -1 → g' a x > 0) ∧
     (-1 < x ∧ x < 0 → g' a x < 0) ∧
     (x > 0 → g' a x > 0)) := sorry

end extremum_f_at_neg_four_thirds_monotonicity_g_l1945_194574


namespace extremum_areas_extremum_areas_case_b_equal_areas_l1945_194575

variable (a b x : ℝ)
variable (h1 : b > 0) (h2 : a ≥ b) (h_cond : 0 < x ∧ x ≤ b)

def area_t1 (a b x : ℝ) : ℝ := 2 * x^2 - (a + b) * x + a * b
def area_t2 (a b x : ℝ) : ℝ := -2 * x^2 + (a + b) * x

noncomputable def x0 (a b : ℝ) : ℝ := (a + b) / 4

-- Problem 1
theorem extremum_areas :
  b ≥ a / 3 → area_t1 a b (x0 a b) ≤ area_t1 a b x ∧ area_t2 a b (x0 a b) ≥ area_t2 a b x :=
sorry

theorem extremum_areas_case_b :
  b < a / 3 → (area_t1 a b b = b^2) ∧ (area_t2 a b b = a * b - b^2) :=
sorry

-- Problem 2
theorem equal_areas :
  b ≤ a ∧ a ≤ 2 * b → (area_t1 a b (a / 2) = area_t2 a b (a / 2)) ∧ (area_t1 a b (b / 2) = area_t2 a b (b / 2)) :=
sorry

end extremum_areas_extremum_areas_case_b_equal_areas_l1945_194575


namespace expression_is_integer_iff_divisible_l1945_194504

theorem expression_is_integer_iff_divisible (k n : ℤ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ m : ℤ, n = m * (k + 2) ↔ (∃ C : ℤ, (3 * n - 4 * k + 2) / (k + 2) * C = (3 * n - 4 * k + 2) / (k + 2)) :=
sorry

end expression_is_integer_iff_divisible_l1945_194504


namespace tan_alpha_calc_l1945_194509

theorem tan_alpha_calc (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (2 * α) / Real.cos α ^ 2) = 6 :=
by sorry

end tan_alpha_calc_l1945_194509


namespace anne_speed_ratio_l1945_194565

theorem anne_speed_ratio (B A A' : ℝ) (h_A : A = 1/12) (h_together_current : (B + A) * 4 = 1) (h_together_new : (B + A') * 3 = 1) :
  A' / A = 2 := 
by
  sorry

end anne_speed_ratio_l1945_194565


namespace unique_a_for_fx_eq_2ax_l1945_194553

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * Real.log x

theorem unique_a_for_fx_eq_2ax (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, f x a = 2 * a * x → x = (a + Real.sqrt (a^2 + 4 * a)) / 2) →
  a = 1 / 2 :=
sorry

end unique_a_for_fx_eq_2ax_l1945_194553


namespace expression_evaluation_l1945_194539

theorem expression_evaluation : 
  (3.14 - Real.pi)^0 + abs (Real.sqrt 2 - 1) + (1 / 2)^(-1:ℤ) - Real.sqrt 8 = 2 - Real.sqrt 2 :=
by sorry

end expression_evaluation_l1945_194539


namespace inverse_matrix_l1945_194524

theorem inverse_matrix
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (B : Matrix (Fin 2) (Fin 2) ℚ)
  (H : A * B = ![![1, 2], ![0, 6]]) :
  A⁻¹ = ![![-1, 0], ![0, 2]] :=
sorry

end inverse_matrix_l1945_194524


namespace paintable_sum_l1945_194510

theorem paintable_sum :
  ∃ (h t u v : ℕ), h > 0 ∧ t > 0 ∧ u > 0 ∧ v > 0 ∧
  (∀ k, k % h = 1 ∨ k % t = 2 ∨ k % u = 3 ∨ k % v = 4) ∧
  (∀ k k', k ≠ k' → (k % h ≠ k' % h ∧ k % t ≠ k' % t ∧ k % u ≠ k' % u ∧ k % v ≠ k' % v)) ∧
  1000 * h + 100 * t + 10 * u + v = 4536 :=
by
  sorry

end paintable_sum_l1945_194510


namespace meeting_at_centroid_l1945_194522

theorem meeting_at_centroid :
  let A := (2, 9)
  let B := (-3, -4)
  let C := (6, -1)
  let centroid := ((2 - 3 + 6) / 3, (9 - 4 - 1) / 3)
  centroid = (5 / 3, 4 / 3) := sorry

end meeting_at_centroid_l1945_194522


namespace plates_are_multiple_of_eleven_l1945_194525

theorem plates_are_multiple_of_eleven
    (P : ℕ)    -- Number of plates
    (S : ℕ := 33)    -- Number of spoons
    (g : ℕ := 11)    -- Greatest number of groups
    (hS : S % g = 0)    -- Condition: All spoons can be divided into these groups evenly
    (hP : ∀ (k : ℕ), P = k * g) : ∃ x : ℕ, P = 11 * x :=
by
  sorry

end plates_are_multiple_of_eleven_l1945_194525


namespace jesters_on_stilts_count_l1945_194566

theorem jesters_on_stilts_count :
  ∃ j e : ℕ, 3 * j + 4 * e = 50 ∧ j + e = 18 ∧ j = 22 :=
by 
  sorry

end jesters_on_stilts_count_l1945_194566


namespace problem1_problem2_l1945_194584

namespace TriangleProofs

-- Problem 1: Prove that A + B = π / 2
theorem problem1 (a b c : ℝ) (A B C : ℝ) 
  (m n : ℝ × ℝ) 
  (h1 : m = (a, Real.cos B))
  (h2 : n = (b, Real.cos A))
  (h_parallel : m.1 * n.2 = m.2 * n.1)
  (h_neq : m ≠ n)
  : A + B = Real.pi / 2 :=
sorry

-- Problem 2: Determine the range of x
theorem problem2 (A B : ℝ) (x : ℝ) 
  (h : A + B = Real.pi / 2) 
  (hx : x * Real.sin A * Real.sin B = Real.sin A + Real.sin B) 
  : 2 * Real.sqrt 2 ≤ x :=
sorry

end TriangleProofs

end problem1_problem2_l1945_194584


namespace salary_january_l1945_194512

theorem salary_january
  (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8300)
  (h3 : May = 6500) :
  J = 5300 :=
by
  sorry

end salary_january_l1945_194512


namespace sum_of_consecutive_integers_l1945_194503

theorem sum_of_consecutive_integers (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + 1 = b) (h4 : b + 1 = c) (h5 : a * b * c = 336) : a + b + c = 21 :=
sorry

end sum_of_consecutive_integers_l1945_194503


namespace g_of_g_of_g_of_g_of_3_l1945_194596

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else x^2 + 2

theorem g_of_g_of_g_of_g_of_3 : g (g (g (g 3))) = 3 :=
by sorry

end g_of_g_of_g_of_g_of_3_l1945_194596


namespace LynsDonation_l1945_194592

theorem LynsDonation (X : ℝ)
  (h1 : 1/3 * X + 1/2 * X + 1/4 * (X - (1/3 * X + 1/2 * X)) = 3/4 * X)
  (h2 : (X - 3/4 * X)/4 = 30) :
  X = 240 := by
  sorry

end LynsDonation_l1945_194592


namespace coords_reflect_origin_l1945_194591

def P : Type := (ℤ × ℤ)

def reflect_origin (p : P) : P :=
  (-p.1, -p.2)

theorem coords_reflect_origin (p : P) (hx : p = (2, -1)) : reflect_origin p = (-2, 1) :=
by
  sorry

end coords_reflect_origin_l1945_194591


namespace percentage_increase_l1945_194535

theorem percentage_increase (P Q : ℝ)
  (price_decreased : ∀ P', P' = 0.80 * P)
  (revenue_increased : ∀ R R', R = P * Q ∧ R' = 1.28000000000000025 * R)
  : ∃ Q', Q' = 1.6000000000000003125 * Q :=
by
  sorry

end percentage_increase_l1945_194535


namespace division_value_l1945_194560

theorem division_value (x : ℝ) (h : 1376 / x - 160 = 12) : x = 8 := 
by sorry

end division_value_l1945_194560


namespace number_of_pizza_varieties_l1945_194520

-- Definitions for the problem conditions
def number_of_flavors : Nat := 8
def toppings : List String := ["C", "M", "O", "J", "L"]

-- Function to count valid combinations of toppings
def valid_combinations (n : Nat) : Nat :=
  match n with
  | 1 => 5
  | 2 => 10 - 1 -- Subtracting the invalid combination (O, J)
  | 3 => 10 - 3 -- Subtracting the 3 invalid combinations containing (O, J)
  | _ => 0

def total_topping_combinations : Nat :=
  valid_combinations 1 + valid_combinations 2 + valid_combinations 3

-- The final proof stating the number of pizza varieties
theorem number_of_pizza_varieties : total_topping_combinations * number_of_flavors = 168 := by
  -- Calculation steps can be inserted here, we use sorry for now
  sorry

end number_of_pizza_varieties_l1945_194520


namespace total_amount_saved_l1945_194568

def priceX : ℝ := 575
def surcharge_rateX : ℝ := 0.04
def installation_chargeX : ℝ := 82.50
def total_chargeX : ℝ := priceX + surcharge_rateX * priceX + installation_chargeX

def priceY : ℝ := 530
def surcharge_rateY : ℝ := 0.03
def installation_chargeY : ℝ := 93.00
def total_chargeY : ℝ := priceY + surcharge_rateY * priceY + installation_chargeY

def savings : ℝ := total_chargeX - total_chargeY

theorem total_amount_saved : savings = 41.60 :=
by
  sorry

end total_amount_saved_l1945_194568


namespace angle_in_second_quadrant_l1945_194556

def inSecondQuadrant (θ : ℤ) : Prop :=
  90 < θ ∧ θ < 180

theorem angle_in_second_quadrant :
  ∃ k : ℤ, inSecondQuadrant (-2015 + 360 * k) :=
by {
  sorry
}

end angle_in_second_quadrant_l1945_194556


namespace calculate_expression_l1945_194585

theorem calculate_expression : 
  -3^2 + Real.sqrt ((-2)^4) - (-27)^(1/3 : ℝ) = -2 := 
by
  sorry

end calculate_expression_l1945_194585


namespace find_F_of_circle_l1945_194546

def circle_equation (x y F : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + F = 0

def is_circle_with_radius (x y F r : ℝ) : Prop := 
  ∃ k h, (x - k)^2 + (y + h)^2 = r

theorem find_F_of_circle {F : ℝ} :
  (∀ x y : ℝ, circle_equation x y F) ∧ 
  is_circle_with_radius 1 1 F 4 → F = -2 := 
by
  sorry

end find_F_of_circle_l1945_194546


namespace find_number_that_gives_200_9_when_8_036_divided_by_it_l1945_194542

theorem find_number_that_gives_200_9_when_8_036_divided_by_it (
  x : ℝ
) : (8.036 / x = 200.9) → (x = 0.04) :=
by
  intro h
  sorry

end find_number_that_gives_200_9_when_8_036_divided_by_it_l1945_194542


namespace range_of_a_for_increasing_function_l1945_194593

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 2 * a else a ^ x

theorem range_of_a_for_increasing_function (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (3/2 ≤ a ∧ a < 6) := sorry

end range_of_a_for_increasing_function_l1945_194593


namespace sin_2B_minus_5pi_over_6_area_of_triangle_l1945_194571

-- Problem (I)
theorem sin_2B_minus_5pi_over_6 {A B C : ℝ} (a b c : ℝ)
  (h: 3 * Real.cos A * Real.cos C * (Real.tan A * Real.tan C - 1) = 1) :
  Real.sin (2 * B - 5 * Real.pi / 6) = (7 - 4 * Real.sqrt 6) / 18 :=
sorry

-- Problem (II)
theorem area_of_triangle {A B C : ℝ} (a b c : ℝ)
  (h1: a + c = 3 * Real.sqrt 3 / 2) (h2: b = Real.sqrt 3) :
  Real.sqrt (a * c) * Real.sin B / 2 = 15 * Real.sqrt 2 / 32 :=
sorry

end sin_2B_minus_5pi_over_6_area_of_triangle_l1945_194571


namespace lowest_positive_integer_divisible_by_primes_between_10_and_50_l1945_194529

def primes_10_to_50 : List ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def lcm_list (lst : List ℕ) : ℕ :=
lst.foldr Nat.lcm 1

theorem lowest_positive_integer_divisible_by_primes_between_10_and_50 :
  lcm_list primes_10_to_50 = 614889782588491410 :=
by
  sorry

end lowest_positive_integer_divisible_by_primes_between_10_and_50_l1945_194529


namespace tangent_line_at_P_exists_c_for_a_l1945_194586

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_P :
  ∀ x y : ℝ, y = f x → x = 1 → y = 0 → x - y - 1 = 0 := 
by 
  sorry

theorem exists_c_for_a :
  ∀ a : ℝ, 1 < a → ∃ c : ℝ, 0 < c ∧ c < 1 / a ∧ ∀ x : ℝ, c < x → x < 1 → f x > a * x * (x - 1) :=
by 
  sorry

end tangent_line_at_P_exists_c_for_a_l1945_194586


namespace range_of_a_l1945_194573

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x^2
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 4 * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, x0 > 0 ∧ f x0 = g (-x0) a) → a ≤ -1 := 
by
  sorry

end range_of_a_l1945_194573


namespace total_cost_of_bicycles_is_2000_l1945_194552

noncomputable def calculate_total_cost_of_bicycles (SP1 SP2 : ℝ) (profit1 profit2 : ℝ) : ℝ :=
  let C1 := SP1 / (1 + profit1)
  let C2 := SP2 / (1 - profit2)
  C1 + C2

theorem total_cost_of_bicycles_is_2000 :
  calculate_total_cost_of_bicycles 990 990 0.10 0.10 = 2000 :=
by
  -- Proof will be provided here
  sorry

end total_cost_of_bicycles_is_2000_l1945_194552


namespace sqrt_frac_meaningful_l1945_194519

theorem sqrt_frac_meaningful (x : ℝ) (h : 1 / (x - 1) > 0) : x > 1 :=
sorry

end sqrt_frac_meaningful_l1945_194519


namespace worker_followed_instructions_l1945_194572

def initial_trees (grid_size : ℕ) : ℕ := grid_size * grid_size

noncomputable def rows_of_trees (rows left each_row : ℕ) : ℕ := rows * each_row

theorem worker_followed_instructions :
  initial_trees 7 = 49 →
  rows_of_trees 5 20 4 = 20 →
  rows_of_trees 5 10 4 = 39 →
  (∃ T : Finset (Fin 7 × Fin 7), T.card = 10) :=
by
  sorry

end worker_followed_instructions_l1945_194572


namespace sin_C_eq_sqrt14_div_8_area_triangle_eq_sqrt7_div_4_l1945_194511

theorem sin_C_eq_sqrt14_div_8 (b c : ℝ) (cosB : ℝ) (h1 : b = Real.sqrt 2) (h2 : c = 1) (h3 : cosB = 3 / 4) : 
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := c * sinB / b
  sinC = Real.sqrt 14 / 8 := 
by
  -- Proof is omitted
  sorry

theorem area_triangle_eq_sqrt7_div_4 (b c : ℝ) (cosB : ℝ) (h1 : b = Real.sqrt 2) (h2 : c = 1) (h3 : cosB = 3 / 4) : 
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := c * sinB / b
  let cosC := Real.sqrt (1 - sinC^2)
  let sinA := sinB * cosC + cosB * sinC
  let area := 1 / 2 * b * c * sinA
  area = Real.sqrt 7 / 4 := 
by
  -- Proof is omitted
  sorry

end sin_C_eq_sqrt14_div_8_area_triangle_eq_sqrt7_div_4_l1945_194511


namespace Maya_takes_longer_l1945_194500

-- Define the constants according to the conditions
def Xavier_reading_speed : ℕ := 120
def Maya_reading_speed : ℕ := 60
def novel_pages : ℕ := 360
def minutes_per_hour : ℕ := 60

-- Define the times it takes for Xavier and Maya to read the novel
def Xavier_time : ℕ := novel_pages / Xavier_reading_speed
def Maya_time : ℕ := novel_pages / Maya_reading_speed

-- Define the time difference in hours and then in minutes
def time_difference_hours : ℕ := Maya_time - Xavier_time
def time_difference_minutes : ℕ := time_difference_hours * minutes_per_hour

-- The statement to prove
theorem Maya_takes_longer :
  time_difference_minutes = 180 :=
by
  sorry

end Maya_takes_longer_l1945_194500


namespace simplify_fraction_l1945_194599

theorem simplify_fraction : (1 / (2 + (2/3))) = (3 / 8) :=
by
  sorry

end simplify_fraction_l1945_194599


namespace find_interest_rate_l1945_194548

noncomputable def amount : ℝ := 896
noncomputable def principal : ℝ := 799.9999999999999
noncomputable def time : ℝ := 2 + 2 / 5
noncomputable def interest : ℝ := amount - principal
noncomputable def rate : ℝ := interest / (principal * time)

theorem find_interest_rate :
  rate * 100 = 5 := by
  sorry

end find_interest_rate_l1945_194548


namespace train_length_l1945_194569

theorem train_length {L : ℝ} (h_equal_lengths : ∃ (L: ℝ), L = L) (h_cross_time : ∃ (t : ℝ), t = 60) (h_speed : ∃ (v : ℝ), v = 20) : L = 600 :=
by
  sorry

end train_length_l1945_194569


namespace inequality_xyz_l1945_194562

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end inequality_xyz_l1945_194562


namespace not_sufficient_nor_necessary_geometric_seq_l1945_194578

theorem not_sufficient_nor_necessary_geometric_seq {a : ℕ → ℝ} (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q) :
    (a 1 < a 3) ↔ (¬(a 2 < a 4) ∨ ¬(a 4 < a 2)) :=
by
  sorry

end not_sufficient_nor_necessary_geometric_seq_l1945_194578


namespace base_area_of_cone_with_slant_height_10_and_semi_lateral_surface_l1945_194537

theorem base_area_of_cone_with_slant_height_10_and_semi_lateral_surface :
  (l = 10) → (l = 2 * r) → (A = 25 * π) :=
  by
  intros l_eq_ten l_eq_two_r
  have r_is_five : r = 5 := by sorry
  have A_is_25pi : A = 25 * π := by sorry
  exact A_is_25pi

end base_area_of_cone_with_slant_height_10_and_semi_lateral_surface_l1945_194537


namespace max_mineral_value_l1945_194533

/-- Jane discovers three types of minerals with given weights and values:
6-pound mineral chunks worth $16 each,
3-pound mineral chunks worth $9 each,
and 2-pound mineral chunks worth $3 each. 
There are at least 30 of each type available.
She can haul a maximum of 21 pounds in her cart.
Prove that the maximum value, in dollars, that Jane can transport is $63. -/
theorem max_mineral_value : 
  ∃ (value : ℕ), (∀ (x y z : ℕ), 6 * x + 3 * y + 2 * z ≤ 21 → 
    (x ≤ 30 ∧ y ≤ 30 ∧ z ≤ 30) → value ≥ 16 * x + 9 * y + 3 * z) ∧ value = 63 :=
by sorry

end max_mineral_value_l1945_194533


namespace desired_average_sale_is_5600_l1945_194531

-- Define the sales for five consecutive months
def sale1 : ℕ := 5266
def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029

-- Define the required sale for the sixth month
def sale6 : ℕ := 4937

-- Calculate total sales for the first five months
def total_five_months := sale1 + sale2 + sale3 + sale4 + sale5

-- Calculate total sales for six months
def total_six_months := total_five_months + sale6

-- Calculate the desired average sale for six months
def desired_average := total_six_months / 6

-- The theorem statement: desired average sale for the six months
theorem desired_average_sale_is_5600 : desired_average = 5600 :=
by
  sorry

end desired_average_sale_is_5600_l1945_194531


namespace problem_statement_l1945_194513

-- Definitions corresponding to the given condition
noncomputable def sum_to_n (n : ℕ) : ℤ := (n * (n + 1)) / 2
noncomputable def alternating_sum_to_n (n : ℕ) : ℤ := if n % 2 = 0 then -(n / 2) else (n / 2 + 1)

-- Lean statement for the problem
theorem problem_statement :
  (alternating_sum_to_n 2022) * (sum_to_n 2023 - 1) - (alternating_sum_to_n 2023) * (sum_to_n 2022 - 1) = 2023 :=
sorry

end problem_statement_l1945_194513


namespace M_inter_N_eq_l1945_194516

def set_M (x : ℝ) : Prop := x^2 - 3 * x < 0
def set_N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 4

def M := { x : ℝ | set_M x }
def N := { x : ℝ | set_N x }

theorem M_inter_N_eq : M ∩ N = { x | 1 ≤ x ∧ x < 3 } :=
by sorry

end M_inter_N_eq_l1945_194516


namespace evaluate_expression_l1945_194581

theorem evaluate_expression : 
  (196 * (1 / 17 - 1 / 21) + 361 * (1 / 21 - 1 / 13) + 529 * (1 / 13 - 1 / 17)) /
    (14 * (1 / 17 - 1 / 21) + 19 * (1 / 21 - 1 / 13) + 23 * (1 / 13 - 1 / 17)) = 56 :=
by
  sorry

end evaluate_expression_l1945_194581


namespace masha_nonnegative_l1945_194502

theorem masha_nonnegative (a b c d : ℝ) (h1 : a + b = c * d) (h2 : a * b = c + d) : 
  (a + 1) * (b + 1) * (c + 1) * (d + 1) ≥ 0 := 
by
  -- Proof is omitted
  sorry

end masha_nonnegative_l1945_194502


namespace percentage_of_men_l1945_194530

variable (M W : ℝ)
variable (h1 : M + W = 100)
variable (h2 : 0.20 * W + 0.70 * M = 40)

theorem percentage_of_men : M = 40 :=
by
  sorry

end percentage_of_men_l1945_194530


namespace linear_function_increasing_l1945_194587

theorem linear_function_increasing (x1 x2 y1 y2 : ℝ) (h1 : y1 = 2 * x1 - 1) (h2 : y2 = 2 * x2 - 1) (h3 : x1 > x2) : y1 > y2 :=
by
  sorry

end linear_function_increasing_l1945_194587


namespace find_percentage_l1945_194558

noncomputable def percentage_condition (P : ℝ) : Prop :=
  9000 + (P / 100) * 9032 = 10500

theorem find_percentage (P : ℝ) (h : percentage_condition P) : P = 16.61 :=
sorry

end find_percentage_l1945_194558


namespace plane_through_A_perpendicular_to_BC_l1945_194515

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_between_points (P Q : Point3D) : Point3D :=
  { x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z }

def plane_eq (n : Point3D) (P : Point3D) (x y z : ℝ) : ℝ :=
  n.x * (x - P.x) + n.y * (y - P.y) + n.z * (z - P.z)

def A := Point3D.mk 0 (-2) 8
def B := Point3D.mk 4 3 2
def C := Point3D.mk 1 4 3

def n := vector_between_points B C
def plane := plane_eq n A

theorem plane_through_A_perpendicular_to_BC :
  ∀ x y z : ℝ, plane x y z = 0 ↔ -3 * x + y + z - 6 = 0 :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l1945_194515


namespace thirteen_pow_seven_mod_eight_l1945_194590

theorem thirteen_pow_seven_mod_eight : 
  (13^7) % 8 = 5 := by
  sorry

end thirteen_pow_seven_mod_eight_l1945_194590


namespace abc_inequality_l1945_194545

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) : 
    (ab / (a^5 + ab + b^5)) + (bc / (b^5 + bc + c^5)) + (ca / (c^5 + ca + a^5)) ≤ 1 := 
sorry

end abc_inequality_l1945_194545


namespace abs_neg_three_halves_l1945_194506

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end abs_neg_three_halves_l1945_194506


namespace number_of_teams_l1945_194538

-- Define the necessary conditions and variables
variable (n : ℕ)
variable (num_games : ℕ)

-- Define the condition that each team plays each other team exactly once 
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The main theorem to prove
theorem number_of_teams (h : total_games n = 91) : n = 14 :=
sorry

end number_of_teams_l1945_194538


namespace range_of_a_l1945_194576

theorem range_of_a (a : ℝ) :
  (∃ (x : ℝ), (2 - 2^(-|x - 3|))^2 = 3 + a) ↔ -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l1945_194576


namespace find_symmetric_point_l1945_194528

structure Point := (x : Int) (y : Int)

def translate_right (p : Point) (n : Int) : Point :=
  { x := p.x + n, y := p.y }

def symmetric_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem find_symmetric_point : 
  ∀ (A B C : Point),
  A = ⟨-1, 2⟩ →
  B = translate_right A 2 →
  C = symmetric_x_axis B →
  C = ⟨1, -2⟩ :=
by
  intros A B C hA hB hC
  sorry

end find_symmetric_point_l1945_194528


namespace exists_fifth_degree_polynomial_l1945_194534

noncomputable def p (x : ℝ) : ℝ :=
  12.4 * (x^5 - 1.38 * x^3 + 0.38 * x)

theorem exists_fifth_degree_polynomial :
  (∃ x1 x2 : ℝ, -1 < x1 ∧ x1 < 1 ∧ -1 < x2 ∧ x2 < 1 ∧ x1 ≠ x2 ∧ 
    p x1 = 1 ∧ p x2 = -1 ∧ p (-1) = 0 ∧ p 1 = 0) :=
  sorry

end exists_fifth_degree_polynomial_l1945_194534


namespace discount_percentage_l1945_194549

theorem discount_percentage (discount amount_paid : ℝ) (h_discount : discount = 40) (h_paid : amount_paid = 120) : 
  (discount / (discount + amount_paid)) * 100 = 25 := by
  sorry

end discount_percentage_l1945_194549


namespace probability_reaching_five_without_returning_to_zero_l1945_194559

def reach_position_without_return_condition (tosses : ℕ) (target : ℤ) (return_limit : ℤ) : ℕ :=
  -- Ideally we should implement the logic to find the number of valid paths here (as per problem constraints)
  sorry

theorem probability_reaching_five_without_returning_to_zero {a b : ℕ} (h_rel_prime : Nat.gcd a b = 1)
    (h_paths_valid : reach_position_without_return_condition 10 5 3 = 15) :
    a = 15 ∧ b = 256 ∧ a + b = 271 :=
by
  sorry

end probability_reaching_five_without_returning_to_zero_l1945_194559


namespace problem_solution_l1945_194550

def sequence_graphical_representation_isolated (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ x : ℝ, x = a n

def sequence_terms_infinite (a : ℕ → ℝ) : Prop :=
  ∃ l : List ℝ, ∃ n : ℕ, l.length = n

def sequence_general_term_formula_unique (a : ℕ → ℝ) : Prop :=
  ∀ f g : ℕ → ℝ, (∀ n, f n = g n) → f = g

theorem problem_solution
  (h1 : ∀ a : ℕ → ℝ, sequence_graphical_representation_isolated a)
  (h2 : ¬ ∀ a : ℕ → ℝ, sequence_terms_infinite a)
  (h3 : ¬ ∀ a : ℕ → ℝ, sequence_general_term_formula_unique a) :
  ∀ a : ℕ → ℝ, sequence_graphical_representation_isolated a ∧ 
                ¬ (sequence_terms_infinite a) ∧
                ¬ (sequence_general_term_formula_unique a) := by
  sorry

end problem_solution_l1945_194550


namespace find_a7_l1945_194523

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (n : ℕ)

-- Condition 1: The sequence {a_n} is geometric with all positive terms.
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

-- Condition 2: a₄ * a₁₀ = 16
axiom geo_seq_condition : is_geometric_sequence a r ∧ a 4 * a 10 = 16

-- The goal to prove
theorem find_a7 : (is_geometric_sequence a r ∧ a 4 * a 10 = 16) → a 7 = 4 :=
by {
  sorry
}

end find_a7_l1945_194523


namespace units_digit_2016_pow_2017_add_2017_pow_2016_l1945_194595

theorem units_digit_2016_pow_2017_add_2017_pow_2016 :
  (2016 ^ 2017 + 2017 ^ 2016) % 10 = 7 :=
by
  sorry

end units_digit_2016_pow_2017_add_2017_pow_2016_l1945_194595


namespace max_value_expression_l1945_194557

theorem max_value_expression (x : ℝ) : 
  ∃ m : ℝ, m = 1 / 37 ∧ ∀ x : ℝ, (x^6) / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) ≤ m :=
sorry

end max_value_expression_l1945_194557


namespace chocolate_chip_cookies_l1945_194561

theorem chocolate_chip_cookies (chocolate_chips_per_recipe : ℕ) (num_recipes : ℕ) (total_chocolate_chips : ℕ) 
  (h1 : chocolate_chips_per_recipe = 2) 
  (h2 : num_recipes = 23) 
  (h3 : total_chocolate_chips = chocolate_chips_per_recipe * num_recipes) : 
  total_chocolate_chips = 46 :=
by
  rw [h1, h2] at h3
  exact h3

-- sorry

end chocolate_chip_cookies_l1945_194561


namespace remainder_when_divided_by_8_l1945_194563

theorem remainder_when_divided_by_8 :
  (481207 % 8) = 7 :=
by
  sorry

end remainder_when_divided_by_8_l1945_194563


namespace probability_of_death_each_month_l1945_194540

-- Defining the variables and expressions used in conditions
def p : ℝ := 0.1
def N : ℝ := 400
def surviving_after_3_months : ℝ := 291.6

-- The main theorem to be proven
theorem probability_of_death_each_month (prob : ℝ) :
  (N * (1 - prob)^3 = surviving_after_3_months) → (prob = p) :=
by
  sorry

end probability_of_death_each_month_l1945_194540


namespace part1_part2_part3_l1945_194580

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + 2)

theorem part1 (h_odd : ∀ x : ℝ, f x b = -f (-x) b) : b = 1 :=
sorry

theorem part2 (h_b : b = 1) : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 > f x2 1 :=
sorry

theorem part3 (h_monotonic : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 > f x2 1) 
  : ∀ t : ℝ, f (t^2 - 2 * t) 1 + f (2 * t^2 - k) 1 < 0 → k < -1/3 :=
sorry

end part1_part2_part3_l1945_194580


namespace num_positive_integers_l1945_194544

-- Definitions
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

-- Problem statement
theorem num_positive_integers (n : ℕ) (h : n = 2310) :
  (∃ count, count = 3 ∧ (∀ m : ℕ, m > 0 → is_divisor (m^2 - 2) n → count = 3)) := by
  sorry

end num_positive_integers_l1945_194544


namespace at_least_one_greater_than_zero_l1945_194582

noncomputable def a (x : ℝ) : ℝ := x^2 - 2 * x + (Real.pi / 2)
noncomputable def b (y : ℝ) : ℝ := y^2 - 2 * y + (Real.pi / 2)
noncomputable def c (z : ℝ) : ℝ := z^2 - 2 * z + (Real.pi / 2)

theorem at_least_one_greater_than_zero (x y z : ℝ) : (a x > 0) ∨ (b y > 0) ∨ (c z > 0) :=
by sorry

end at_least_one_greater_than_zero_l1945_194582


namespace joe_used_paint_total_l1945_194507

theorem joe_used_paint_total :
  let first_airport_paint := 360
  let second_airport_paint := 600
  let first_week_first_airport := (1/4 : ℝ) * first_airport_paint
  let remaining_first_airport := first_airport_paint - first_week_first_airport
  let second_week_first_airport := (1/6 : ℝ) * remaining_first_airport
  let total_first_airport := first_week_first_airport + second_week_first_airport
  let first_week_second_airport := (1/3 : ℝ) * second_airport_paint
  let remaining_second_airport := second_airport_paint - first_week_second_airport
  let second_week_second_airport := (1/5 : ℝ) * remaining_second_airport
  let total_second_airport := first_week_second_airport + second_week_second_airport
  total_first_airport + total_second_airport = 415 :=
by
  let first_airport_paint := 360
  let second_airport_paint := 600
  let first_week_first_airport := (1/4 : ℝ) * first_airport_paint
  let remaining_first_airport := first_airport_paint - first_week_first_airport
  let second_week_first_airport := (1/6 : ℝ) * remaining_first_airport
  let total_first_airport := first_week_first_airport + second_week_first_airport
  let first_week_second_airport := (1/3 : ℝ) * second_airport_paint
  let remaining_second_airport := second_airport_paint - first_week_second_airport
  let second_week_second_airport := (1/5 : ℝ) * remaining_second_airport
  let total_second_airport := first_week_second_airport + second_week_second_airport
  show total_first_airport + total_second_airport = 415
  sorry

end joe_used_paint_total_l1945_194507


namespace cost_function_discrete_points_l1945_194501

def cost (n : ℕ) : ℕ :=
  if n <= 10 then 20 * n
  else if n <= 25 then 18 * n
  else 0

theorem cost_function_discrete_points :
  (∀ n, 1 ≤ n ∧ n ≤ 25 → ∃ y, cost n = y) ∧
  (∀ m n, 1 ≤ m ∧ m ≤ 25 ∧ 1 ≤ n ∧ n ≤ 25 ∧ m ≠ n → cost m ≠ cost n) :=
sorry

end cost_function_discrete_points_l1945_194501


namespace bob_can_order_199_sandwiches_l1945_194532

-- Define the types of bread, meat, and cheese
def number_of_bread : ℕ := 5
def number_of_meat : ℕ := 7
def number_of_cheese : ℕ := 6

-- Define the forbidden combinations
def forbidden_turkey_swiss : ℕ := number_of_bread -- 5
def forbidden_rye_roastbeef : ℕ := number_of_cheese -- 6

-- Calculate the total sandwiches and subtract forbidden combinations
def total_sandwiches : ℕ := number_of_bread * number_of_meat * number_of_cheese
def forbidden_sandwiches : ℕ := forbidden_turkey_swiss + forbidden_rye_roastbeef

def sandwiches_bob_can_order : ℕ := total_sandwiches - forbidden_sandwiches

theorem bob_can_order_199_sandwiches :
  sandwiches_bob_can_order = 199 :=
by
  -- The calculation steps are encapsulated in definitions and are considered done
  sorry

end bob_can_order_199_sandwiches_l1945_194532
