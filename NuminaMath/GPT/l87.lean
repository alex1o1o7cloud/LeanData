import Mathlib

namespace NUMINAMATH_GPT_product_of_largest_and_second_largest_l87_8713

theorem product_of_largest_and_second_largest (a b c : ℕ) (h₁ : a = 10) (h₂ : b = 11) (h₃ : c = 12) :
  (max (max a b) c * (max (min a (max b c)) (min b (max a c)))) = 132 :=
by
  sorry

end NUMINAMATH_GPT_product_of_largest_and_second_largest_l87_8713


namespace NUMINAMATH_GPT_roots_real_roots_equal_l87_8799

noncomputable def discriminant (a : ℝ) : ℝ :=
  let b := 4 * a
  let c := 2 * a^2 - 1 + 3 * a
  b^2 - 4 * 1 * c

theorem roots_real (a : ℝ) : discriminant a ≥ 0 ↔ a ≤ 1/2 ∨ a ≥ 1 := sorry

theorem roots_equal (a : ℝ) : discriminant a = 0 ↔ a = 1 ∨ a = 1/2 := sorry

end NUMINAMATH_GPT_roots_real_roots_equal_l87_8799


namespace NUMINAMATH_GPT_problem_I_l87_8714

theorem problem_I (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : 
  (1 + 1 / a) * (1 + 1 / b) ≥ 9 := 
by
  sorry

end NUMINAMATH_GPT_problem_I_l87_8714


namespace NUMINAMATH_GPT_triplet_solution_l87_8778

theorem triplet_solution (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) :
  (a + b + c = (1 / a) + (1 / b) + (1 / c) ∧ a ^ 2 + b ^ 2 + c ^ 2 = (1 / a ^ 2) + (1 / b ^ 2) + (1 / c ^ 2))
  ↔ (∃ x, (a = 1 ∨ a = -1 ∨ a = x ∨ a = 1/x) ∧
           (b = 1 ∨ b = -1 ∨ b = x ∨ b = 1/x) ∧
           (c = 1 ∨ c = -1 ∨ c = x ∨ c = 1/x)) := 
sorry

end NUMINAMATH_GPT_triplet_solution_l87_8778


namespace NUMINAMATH_GPT_fraction_identity_l87_8780

theorem fraction_identity (a b : ℚ) (h : (a - 2 * b) / b = 3 / 5) : a / b = 13 / 5 :=
sorry

end NUMINAMATH_GPT_fraction_identity_l87_8780


namespace NUMINAMATH_GPT_terminal_side_in_third_quadrant_l87_8752

def is_equivalent_angle (a b : ℝ) : Prop := ∃ k : ℤ, a = b + k * 360

def in_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

theorem terminal_side_in_third_quadrant : 
  ∀ θ, θ = 600 → in_third_quadrant (θ % 360) :=
by
  intro θ
  intro hθ
  sorry

end NUMINAMATH_GPT_terminal_side_in_third_quadrant_l87_8752


namespace NUMINAMATH_GPT_yellow_balls_are_24_l87_8781

theorem yellow_balls_are_24 (x y z : ℕ) (h1 : x + y + z = 68) 
                             (h2 : y = 2 * x) (h3 : 3 * z = 4 * y) : y = 24 :=
by
  sorry

end NUMINAMATH_GPT_yellow_balls_are_24_l87_8781


namespace NUMINAMATH_GPT_distinct_ball_placement_l87_8772

def num_distributions (balls boxes : ℕ) : ℕ :=
  if boxes = 3 then 243 - 32 + 16 else 0

theorem distinct_ball_placement : num_distributions 5 3 = 227 :=
by
  sorry

end NUMINAMATH_GPT_distinct_ball_placement_l87_8772


namespace NUMINAMATH_GPT_geometric_sequence_sum_l87_8794

-- Define the problem conditions and the result
theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ), a 1 + a 2 = 16 ∧ a 3 + a 4 = 24 → a 7 + a 8 = 54 :=
by
  -- Preliminary steps and definitions to prove the theorem
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l87_8794


namespace NUMINAMATH_GPT_range_of_a_l87_8725

theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ, n ≥ 8 → (a * (n^2) + n + 5) > (a * ((n + 1)^2) + (n + 1) + 5)) → 
  (a * (1^2) + 1 + 5 < a * (2^2) + 2 + 5) →
  (a * (2^2) + 2 + 5 < a * (3^2) + 3 + 5) →
  (a * (3^2) + 3 + 5 < a * (4^2) + 4 + 5) →
  (- (1 / 7) < a ∧ a < - (1 / 17)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l87_8725


namespace NUMINAMATH_GPT_sequence_explicit_formula_l87_8728

theorem sequence_explicit_formula (a : ℕ → ℤ) (n : ℕ) :
  a 0 = 2 →
  (∀ n, a (n+1) = a n - n + 3) →
  a n = -((n * (n + 1)) / 2) + 3 * n + 2 :=
by
  intros h0 h_rec
  sorry

end NUMINAMATH_GPT_sequence_explicit_formula_l87_8728


namespace NUMINAMATH_GPT_remainder_correct_l87_8769

noncomputable def p (x : ℝ) : ℝ := 3 * x ^ 8 - 2 * x ^ 5 + 5 * x ^ 3 - 9
noncomputable def d (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1
noncomputable def r (x : ℝ) : ℝ := 29 * x - 32

theorem remainder_correct (x : ℝ) :
  ∃ q : ℝ → ℝ, p x = d x * q x + r x :=
sorry

end NUMINAMATH_GPT_remainder_correct_l87_8769


namespace NUMINAMATH_GPT_vanya_correct_answers_l87_8788

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end NUMINAMATH_GPT_vanya_correct_answers_l87_8788


namespace NUMINAMATH_GPT_unique_zero_function_l87_8733

theorem unique_zero_function
    (f : ℝ → ℝ)
    (H : ∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)) :
    ∀ x : ℝ, f x = 0 := 
by 
     sorry

end NUMINAMATH_GPT_unique_zero_function_l87_8733


namespace NUMINAMATH_GPT_tan_of_neg_23_over_3_pi_l87_8734

theorem tan_of_neg_23_over_3_pi : (Real.tan (- 23 / 3 * Real.pi) = Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_tan_of_neg_23_over_3_pi_l87_8734


namespace NUMINAMATH_GPT_problem_mod_l87_8707

theorem problem_mod (a b c d : ℕ) (h1 : a = 2011) (h2 : b = 2012) (h3 : c = 2013) (h4 : d = 2014) :
  (a * b * c * d) % 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_mod_l87_8707


namespace NUMINAMATH_GPT_divisible_by_xyz_l87_8773

/-- 
Prove that the expression K = (x+y+z)^5 - (-x+y+z)^5 - (x-y+z)^5 - (x+y-z)^5 
is divisible by each of x, y, z.
-/
theorem divisible_by_xyz (x y z : ℝ) :
  ∃ t : ℝ, (x + y + z)^5 - (-x + y + z)^5 - (x - y + z)^5 - (x + y - z)^5 = t * x * y * z :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_divisible_by_xyz_l87_8773


namespace NUMINAMATH_GPT_cubic_polynomial_roots_l87_8791

variables (a b c : ℚ)

theorem cubic_polynomial_roots (a b c : ℚ) :
  (c = 0 → ∃ x y z : ℚ, (x = 0 ∧ y = 1 ∧ z = -2) ∧ (-a = x + y + z) ∧ (b = x*y + y*z + z*x) ∧ (-c = x*y*z)) ∧
  (c ≠ 0 → ∃ x y z : ℚ, (x = 1 ∧ y = -1 ∧ z = -1) ∧ (-a = x + y + z) ∧ (b = x*y + y*z + z*x) ∧ (-c = x*y*z)) :=
by
  sorry

end NUMINAMATH_GPT_cubic_polynomial_roots_l87_8791


namespace NUMINAMATH_GPT_final_position_D_l87_8755

open Function

-- Define the original points of the parallelogram
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, 8)
def C : ℝ × ℝ := (9, 4)
def D : ℝ × ℝ := (7, 0)

-- Define the reflection across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define the translation by (0, 1)
def translate_up (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + 1)
def translate_down (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 - 1)

-- Define the reflection across y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Combine the transformations to get the final reflection across y = x - 1
def reflect_across_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_down (reflect_y_eq_x (translate_up p))

-- Prove that the final position of D after the two transformations is (1, -8)
theorem final_position_D'' : reflect_across_y_eq_x_minus_1 (reflect_y_axis D) = (1, -8) :=
  sorry

end NUMINAMATH_GPT_final_position_D_l87_8755


namespace NUMINAMATH_GPT_each_sibling_gets_13_pencils_l87_8745

theorem each_sibling_gets_13_pencils (colored_pencils : ℕ) (black_pencils : ℕ) (siblings : ℕ) (kept_pencils : ℕ) 
  (hyp1 : colored_pencils = 14) (hyp2 : black_pencils = 35) (hyp3 : siblings = 3) (hyp4 : kept_pencils = 10) :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by sorry

end NUMINAMATH_GPT_each_sibling_gets_13_pencils_l87_8745


namespace NUMINAMATH_GPT_geom_series_min_q_l87_8741

theorem geom_series_min_q (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h_geom : ∃ k : ℝ, q = p * k ∧ r = q * k)
  (hpqr : p * q * r = 216) : q = 6 :=
sorry

end NUMINAMATH_GPT_geom_series_min_q_l87_8741


namespace NUMINAMATH_GPT_students_table_tennis_not_basketball_l87_8784

variable (total_students : ℕ)
variable (students_like_basketball : ℕ)
variable (students_like_table_tennis : ℕ)
variable (students_dislike_both : ℕ)

theorem students_table_tennis_not_basketball 
  (h_total : total_students = 40)
  (h_basketball : students_like_basketball = 17)
  (h_table_tennis : students_like_table_tennis = 20)
  (h_dislike : students_dislike_both = 8) : 
  ∃ (students_table_tennis_not_basketball : ℕ), students_table_tennis_not_basketball = 15 :=
by
  sorry

end NUMINAMATH_GPT_students_table_tennis_not_basketball_l87_8784


namespace NUMINAMATH_GPT_pond_water_amount_l87_8766

-- Definitions based on the problem conditions
def initial_gallons := 500
def evaporation_rate := 1
def additional_gallons := 10
def days_period := 35
def additional_days_interval := 7

-- Calculations based on the conditions
def total_evaporation := days_period * evaporation_rate
def total_additional_gallons := (days_period / additional_days_interval) * additional_gallons

-- Theorem stating the final amount of water
theorem pond_water_amount : initial_gallons - total_evaporation + total_additional_gallons = 515 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_pond_water_amount_l87_8766


namespace NUMINAMATH_GPT_product_of_numbers_eq_120_l87_8744

theorem product_of_numbers_eq_120 (x y P : ℝ) (h1 : x + y = 23) (h2 : x^2 + y^2 = 289) (h3 : x * y = P) : P = 120 := 
sorry

end NUMINAMATH_GPT_product_of_numbers_eq_120_l87_8744


namespace NUMINAMATH_GPT_tan_45_degrees_l87_8779

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_45_degrees_l87_8779


namespace NUMINAMATH_GPT_buratino_cafe_workdays_l87_8757

-- Define the conditions as given in the problem statement
def days_in_april (d : Nat) : Prop := d >= 1 ∧ d <= 30
def is_monday (d : Nat) : Prop := d = 1 ∨ d = 8 ∨ d = 15 ∨ d = 22 ∨ d = 29

-- Define the period April 1 to April 13
def period_1_13 (d : Nat) : Prop := d >= 1 ∧ d <= 13

-- Define the statements made by Kolya
def kolya_statement_1 : Prop := ∀ d : Nat, days_in_april d → (d >= 1 ∧ d <= 20) → ¬is_monday d → ∃ n : Nat, n = 18
def kolya_statement_2 : Prop := ∀ d : Nat, days_in_april d → (d >= 10 ∧ d <= 30) → ¬is_monday d → ∃ n : Nat, n = 18

-- Define the condition stating Kolya made a mistake once
def kolya_made_mistake_once : Prop := kolya_statement_1 ∨ kolya_statement_2

-- The proof problem: Prove the number of working days from April 1 to April 13 is 11
theorem buratino_cafe_workdays : period_1_13 (d) → (¬is_monday d → (∃ n : Nat, n = 11)) := sorry

end NUMINAMATH_GPT_buratino_cafe_workdays_l87_8757


namespace NUMINAMATH_GPT_graph_of_equation_l87_8703

theorem graph_of_equation :
  ∀ (x y : ℝ), (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (x + y + 2 = 0 ∨ x+y = 0 ∨ x-y = 0) ∧ 
  ¬(∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    (x₁ + y₁ + 2 = 0 ∧ x₁ + y₁ = 0) ∧ 
    (x₂ + y₂ + 2 = 0 ∧ x₂ = -x₂) ∧ 
    (x₃ + y₃ + 2 = 0 ∧ x₃ - y₃ = 0)) := 
sorry

end NUMINAMATH_GPT_graph_of_equation_l87_8703


namespace NUMINAMATH_GPT_gasoline_price_increase_percentage_l87_8771

theorem gasoline_price_increase_percentage : 
  ∀ (highest_price lowest_price : ℝ), highest_price = 24 → lowest_price = 18 → 
  ((highest_price - lowest_price) / lowest_price) * 100 = 33.33 :=
by
  intros highest_price lowest_price h_highest h_lowest
  rw [h_highest, h_lowest]
  -- To be completed in the proof
  sorry

end NUMINAMATH_GPT_gasoline_price_increase_percentage_l87_8771


namespace NUMINAMATH_GPT_wallpaper_three_layers_l87_8721

theorem wallpaper_three_layers
  (A B C : ℝ)
  (hA : A = 300)
  (hB : B = 30)
  (wall_area : ℝ)
  (h_wall_area : wall_area = 180)
  (hC : C = A - (wall_area - B) - B)
  : C = 120 := by
  sorry

end NUMINAMATH_GPT_wallpaper_three_layers_l87_8721


namespace NUMINAMATH_GPT_apples_picked_correct_l87_8785

-- Define the conditions as given in the problem
def apples_given_to_Melanie : ℕ := 27
def apples_left : ℕ := 16

-- Define the problem statement
def total_apples_picked := apples_given_to_Melanie + apples_left

-- Prove that the total apples picked is equal to 43 given the conditions
theorem apples_picked_correct : total_apples_picked = 43 := by
  sorry

end NUMINAMATH_GPT_apples_picked_correct_l87_8785


namespace NUMINAMATH_GPT_adult_ticket_cost_is_19_l87_8792

variable (A : ℕ) -- the cost for an adult ticket
def child_ticket_cost : ℕ := 15
def total_receipts : ℕ := 7200
def total_attendance : ℕ := 400
def adults_attendance : ℕ := 280
def children_attendance : ℕ := 120

-- The equation representing the total receipts
theorem adult_ticket_cost_is_19 (h : total_receipts = 280 * A + 120 * child_ticket_cost) : A = 19 :=
  by sorry

end NUMINAMATH_GPT_adult_ticket_cost_is_19_l87_8792


namespace NUMINAMATH_GPT_initial_amount_of_liquid_A_l87_8709

-- Definitions for liquids A and B and their ratios in the initial and modified mixtures
def initial_ratio_A_over_B : ℚ := 4 / 1
def final_ratio_A_over_B_after_replacement : ℚ := 2 / 3
def mixture_replacement_volume : ℚ := 30

-- Proof of the initial amount of liquid A
theorem initial_amount_of_liquid_A (x : ℚ) (A B : ℚ) (initial_mixture : ℚ) :
  (initial_ratio_A_over_B = 4 / 1) →
  (final_ratio_A_over_B_after_replacement = 2 / 3) →
  (mixture_replacement_volume = 30) →
  (A + B = 5 * x) →
  (A / B = 4 / 1) →
  ((A - 24) / (B - 6 + 30) = 2 / 3) →
  A = 48 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_amount_of_liquid_A_l87_8709


namespace NUMINAMATH_GPT_minimize_travel_time_l87_8768

theorem minimize_travel_time
  (a b c d : ℝ)
  (v₁ v₂ v₃ v₄ : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > d)
  (h4 : v₁ > v₂)
  (h5 : v₂ > v₃)
  (h6 : v₃ > v₄) : 
  (a / v₁ + b / v₂ + c / v₃ + d / v₄) ≤ (a / v₁ + b / v₄ + c / v₃ + d / v₂) :=
sorry

end NUMINAMATH_GPT_minimize_travel_time_l87_8768


namespace NUMINAMATH_GPT_sequence_properties_l87_8724

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d : ℤ} {q : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a n = a 1 + (n - 1) * d

def geometric_sequence (b : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n, b n = b 1 * q^(n - 1)

theorem sequence_properties
  (ha : arithmetic_sequence a d)
  (hb : geometric_sequence b q)
  (h1 : 2 * a 5 - a 3 = 3)
  (h2 : b 2 = 1)
  (h3 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (q = 2 ∨ q = -2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l87_8724


namespace NUMINAMATH_GPT_suff_but_not_necessary_condition_l87_8722

theorem suff_but_not_necessary_condition (x y : ℝ) :
  (xy ≠ 6 → x ≠ 2 ∨ y ≠ 3) ∧ ¬ (x ≠ 2 ∨ y ≠ 3 → xy ≠ 6) :=
by
  sorry

end NUMINAMATH_GPT_suff_but_not_necessary_condition_l87_8722


namespace NUMINAMATH_GPT_probability_digits_different_l87_8786

noncomputable def probability_all_digits_different : ℚ :=
  have tens_digits_probability := (9 / 9) * (8 / 9) * (7 / 9)
  have ones_digits_probability := (10 / 10) * (9 / 10) * (8 / 10)
  (tens_digits_probability * ones_digits_probability)

theorem probability_digits_different :
  probability_all_digits_different = 112 / 225 :=
by 
  -- The proof would go here, but it is not required for this task.
  sorry

end NUMINAMATH_GPT_probability_digits_different_l87_8786


namespace NUMINAMATH_GPT_initial_price_after_markup_l87_8740

theorem initial_price_after_markup 
  (wholesale_price : ℝ) 
  (h_markup_80 : ∀ P, P = wholesale_price → 1.80 * P = 1.80 * wholesale_price)
  (h_markup_diff : ∀ P, P = wholesale_price → 2.00 * P - 1.80 * P = 3) 
  : 1.80 * wholesale_price = 27 := 
by
  sorry

end NUMINAMATH_GPT_initial_price_after_markup_l87_8740


namespace NUMINAMATH_GPT_cheaper_joint_work_l87_8743

theorem cheaper_joint_work (r L P : ℝ) (hr_pos : 0 < r) (hL_pos : 0 < L) (hP_pos : 0 < P) : 
  (2 * P * L) / (3 * r) < (3 * P * L) / (4 * r) :=
by
  sorry

end NUMINAMATH_GPT_cheaper_joint_work_l87_8743


namespace NUMINAMATH_GPT_sum_of_exponents_correct_l87_8735

-- Define the initial expression
def original_expr (a b c : ℤ) : ℤ := 40 * a^6 * b^9 * c^14

-- Define the simplified expression outside the radical
def simplified_outside_expr (a b c : ℤ) : ℤ := a * b^3 * c^3

-- Define the sum of the exponents
def sum_of_exponents : ℕ := 1 + 3 + 3

-- Prove that the given conditions lead to the sum of the exponents being 7
theorem sum_of_exponents_correct (a b c : ℤ) :
  original_expr a b c = 40 * a^6 * b^9 * c^14 →
  simplified_outside_expr a b c = a * b^3 * c^3 →
  sum_of_exponents = 7 :=
by
  intros
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_exponents_correct_l87_8735


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l87_8796

theorem sum_of_consecutive_integers (x y z : ℤ) (h1 : y = x + 1) (h2 : z = y + 1) (h3 : z = 12) :
  x + y + z = 33 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l87_8796


namespace NUMINAMATH_GPT_triangle_DEF_all_acute_l87_8712

theorem triangle_DEF_all_acute
  (α : ℝ)
  (hα : 0 < α ∧ α < 90)
  (DEF : Type)
  (D : DEF) (E : DEF) (F : DEF)
  (angle_DFE : DEF → DEF → DEF → ℝ) 
  (angle_FED : DEF → DEF → DEF → ℝ) 
  (angle_EFD : DEF → DEF → DEF → ℝ)
  (h1 : angle_DFE D F E = 45)
  (h2 : angle_FED F E D = 90 - α / 2)
  (h3 : angle_EFD E D F = 45 + α / 2) :
  (0 < angle_DFE D F E ∧ angle_DFE D F E < 90) ∧ 
  (0 < angle_FED F E D ∧ angle_FED F E D < 90) ∧ 
  (0 < angle_EFD E D F ∧ angle_EFD E D F < 90) := by
  sorry

end NUMINAMATH_GPT_triangle_DEF_all_acute_l87_8712


namespace NUMINAMATH_GPT_find_f_minus1_plus_f_2_l87_8789

variable (f : ℝ → ℝ)

def even_function := ∀ x : ℝ, f (-x) = f x

def symmetric_about_origin := ∀ x : ℝ, f (x + 1) = -f (-(x + 1))

def f_value_at_zero := f 0 = 1

theorem find_f_minus1_plus_f_2 :
  even_function f →
  symmetric_about_origin f →
  f_value_at_zero f →
  f (-1) + f 2 = -1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_f_minus1_plus_f_2_l87_8789


namespace NUMINAMATH_GPT_inequality_range_l87_8736

theorem inequality_range (a : ℝ) : (∀ x : ℝ, x^2 - 1 ≥ a * |x - 1|) → a ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_range_l87_8736


namespace NUMINAMATH_GPT_maximize_sum_of_arithmetic_seq_l87_8715

theorem maximize_sum_of_arithmetic_seq (a d : ℤ) (n : ℤ) : d < 0 → a^2 = (a + 10 * d)^2 → n = 5 ∨ n = 6 :=
by
  intro h_d_neg h_a1_eq_a11
  have h_a1_5d_neg : a + 5 * d = 0 := sorry
  have h_sum_max : n = 5 ∨ n = 6 := sorry
  exact h_sum_max

end NUMINAMATH_GPT_maximize_sum_of_arithmetic_seq_l87_8715


namespace NUMINAMATH_GPT_problem_1_problem_2_l87_8798

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 / 6 + 1 / x - a * Real.log x

theorem problem_1 (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 3 → f x a ≤ f 3 a) → a ≥ 8 / 3 :=
sorry

theorem problem_2 (a : ℝ) (h1 : 0 < a) (x0 : ℝ) :
  (∃! t : ℝ, 0 < t ∧ f t a = 0) → Real.log x0 = (x0^3 + 6) / (2 * (x0^3 - 3)) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l87_8798


namespace NUMINAMATH_GPT_candy_bar_calories_l87_8777

theorem candy_bar_calories :
  let calA := 150
  let calB := 200
  let calC := 250
  let countA := 2
  let countB := 3
  let countC := 4
  (countA * calA + countB * calB + countC * calC) = 1900 :=
by
  sorry

end NUMINAMATH_GPT_candy_bar_calories_l87_8777


namespace NUMINAMATH_GPT_lcm_one_to_twelve_l87_8790

theorem lcm_one_to_twelve : 
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 
  (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 10 (Nat.lcm 11 12)))))))))) = 27720 := 
by sorry

end NUMINAMATH_GPT_lcm_one_to_twelve_l87_8790


namespace NUMINAMATH_GPT_carolyn_shared_with_diana_l87_8774

theorem carolyn_shared_with_diana (initial final shared : ℕ) 
    (h_initial : initial = 47) 
    (h_final : final = 5)
    (h_shared : shared = initial - final) : shared = 42 := by
  rw [h_initial, h_final] at h_shared
  exact h_shared

end NUMINAMATH_GPT_carolyn_shared_with_diana_l87_8774


namespace NUMINAMATH_GPT_students_at_start_of_year_l87_8729

-- Define the initial number of students as a variable S
variables (S : ℕ)

-- Define the conditions
def condition_1 := S - 18 + 14 = 29

-- State the theorem to be proved
theorem students_at_start_of_year (h : condition_1 S) : S = 33 :=
sorry

end NUMINAMATH_GPT_students_at_start_of_year_l87_8729


namespace NUMINAMATH_GPT_ribbon_fraction_per_box_l87_8759

theorem ribbon_fraction_per_box 
  (total_ribbon_used : ℚ)
  (number_of_boxes : ℕ)
  (h1 : total_ribbon_used = 5/8)
  (h2 : number_of_boxes = 5) :
  (total_ribbon_used / number_of_boxes = 1/8) :=
by
  sorry

end NUMINAMATH_GPT_ribbon_fraction_per_box_l87_8759


namespace NUMINAMATH_GPT_find_square_tiles_l87_8730

variables (t s p : ℕ)

theorem find_square_tiles
  (h1 : t + s + p = 30)
  (h2 : 3 * t + 4 * s + 5 * p = 120) :
  s = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_square_tiles_l87_8730


namespace NUMINAMATH_GPT_decreasing_interval_l87_8762

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 15 * x^4 - 15 * x^2

-- State the theorem
theorem decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → f' x < 0 :=
by sorry

end NUMINAMATH_GPT_decreasing_interval_l87_8762


namespace NUMINAMATH_GPT_ira_addition_olya_subtraction_addition_l87_8776

theorem ira_addition (x : ℤ) (h : (11 + x) / (41 + x : ℚ) = 3 / 8) : x = 7 :=
  sorry

theorem olya_subtraction_addition (y : ℤ) (h : (37 - y) / (63 + y : ℚ) = 3 / 17) : y = 22 :=
  sorry

end NUMINAMATH_GPT_ira_addition_olya_subtraction_addition_l87_8776


namespace NUMINAMATH_GPT_min_value_of_fraction_l87_8750

noncomputable def min_val (a b : ℝ) : ℝ :=
  1 / a + 2 * b

theorem min_value_of_fraction (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 2 * a * b + 3 = b) :
  min_val a b = 8 + 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_fraction_l87_8750


namespace NUMINAMATH_GPT_positive_diff_after_add_five_l87_8783

theorem positive_diff_after_add_five (y : ℝ) 
  (h : (45 + y)/2 = 32) : |45 - (y + 5)| = 21 :=
by 
  sorry

end NUMINAMATH_GPT_positive_diff_after_add_five_l87_8783


namespace NUMINAMATH_GPT_remainder_when_squared_mod_seven_l87_8711

theorem remainder_when_squared_mod_seven
  (x y : ℤ) (k m : ℤ)
  (hx : x = 52 * k + 19)
  (hy : 3 * y = 7 * m + 5) :
  ((x + 2 * y)^2 % 7) = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_when_squared_mod_seven_l87_8711


namespace NUMINAMATH_GPT_pos_divisors_180_l87_8731

theorem pos_divisors_180 : 
  (∃ a b c : ℕ, 180 = (2^a) * (3^b) * (5^c) ∧ a = 2 ∧ b = 2 ∧ c = 1) →
  (∃ n : ℕ, n = 18 ∧ n = (a + 1) * (b + 1) * (c + 1)) := by
  sorry

end NUMINAMATH_GPT_pos_divisors_180_l87_8731


namespace NUMINAMATH_GPT_factorizations_of_4050_l87_8747

theorem factorizations_of_4050 :
  ∃! (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4050 :=
by
  sorry

end NUMINAMATH_GPT_factorizations_of_4050_l87_8747


namespace NUMINAMATH_GPT_unique_real_solution_l87_8727

theorem unique_real_solution (x y z : ℝ) :
  (x^3 - 3 * x = 4 - y) ∧ 
  (2 * y^3 - 6 * y = 6 - z) ∧ 
  (3 * z^3 - 9 * z = 8 - x) ↔ 
  x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end NUMINAMATH_GPT_unique_real_solution_l87_8727


namespace NUMINAMATH_GPT_power_mod_result_l87_8787

-- Define the modulus and base
def mod : ℕ := 8
def base : ℕ := 7
def exponent : ℕ := 202

-- State the theorem
theorem power_mod_result :
  (base ^ exponent) % mod = 1 :=
by
  sorry

end NUMINAMATH_GPT_power_mod_result_l87_8787


namespace NUMINAMATH_GPT_number_of_people_in_first_group_l87_8760

variable (W : ℝ)  -- Amount of work
variable (P : ℝ)  -- Number of people in the first group

-- Condition 1: P people can do 3W work in 3 days
def condition1 : Prop := P * (W / 1) * 3 = 3 * W

-- Condition 2: 5 people can do 5W work in 3 days
def condition2 : Prop := 5 * (W / 1) * 3 = 5 * W

-- Theorem to prove: The number of people in the first group is 3
theorem number_of_people_in_first_group (h1 : condition1 W P) (h2 : condition2 W) : P = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_in_first_group_l87_8760


namespace NUMINAMATH_GPT_roots_calc_l87_8742

theorem roots_calc {a b c d : ℝ} (h1: a ≠ 0) (h2 : 125 * a + 25 * b + 5 * c + d = 0) (h3 : -27 * a + 9 * b - 3 * c + d = 0) :
  (b + c) / a = -19 :=
by
  sorry

end NUMINAMATH_GPT_roots_calc_l87_8742


namespace NUMINAMATH_GPT_minimum_value_l87_8797

theorem minimum_value (x y z : ℝ) (h : x + y + z = 1) : 2 * x^2 + y^2 + 3 * z^2 ≥ 3 / 7 := by
  sorry

end NUMINAMATH_GPT_minimum_value_l87_8797


namespace NUMINAMATH_GPT_find_x_l87_8718

theorem find_x (x : ℝ) 
  (h: 3 * x + 6 * x + 2 * x + x = 360) : 
  x = 30 := 
sorry

end NUMINAMATH_GPT_find_x_l87_8718


namespace NUMINAMATH_GPT_jellybeans_original_count_l87_8770

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end NUMINAMATH_GPT_jellybeans_original_count_l87_8770


namespace NUMINAMATH_GPT_melissa_coupe_sale_l87_8749

theorem melissa_coupe_sale :
  ∃ x : ℝ, (0.02 * x + 0.02 * 2 * x = 1800) ∧ x = 30000 :=
by
  sorry

end NUMINAMATH_GPT_melissa_coupe_sale_l87_8749


namespace NUMINAMATH_GPT_largest_k_statement_l87_8746

noncomputable def largest_k (n : ℕ) : ℕ :=
  n - 2

theorem largest_k_statement (S : Finset ℕ) (A : Finset (Finset ℕ)) (h1 : ∀ (A_i : Finset ℕ), A_i ∈ A → 2 ≤ A_i.card ∧ A_i.card < S.card) : 
  largest_k S.card = S.card - 2 :=
by
  sorry

end NUMINAMATH_GPT_largest_k_statement_l87_8746


namespace NUMINAMATH_GPT_find_f_l87_8701

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (h : ∀ x, x ≠ -1 → f ((1-x) / (1+x)) = (1 - x^2) / (1 + x^2)) 
               (hx : x ≠ -1) :
  f x = 2 * x / (1 + x^2) :=
sorry

end NUMINAMATH_GPT_find_f_l87_8701


namespace NUMINAMATH_GPT_circle_radius_tangent_l87_8756

theorem circle_radius_tangent (a : ℝ) (R : ℝ) (h1 : a = 25)
  (h2 : ∀ BP DE CP CE, BP = 2 ∧ DE = 2 ∧ CP = 23 ∧ CE = 23 ∧ BP + CP = a ∧ DE + CE = a)
  : R = 17 :=
sorry

end NUMINAMATH_GPT_circle_radius_tangent_l87_8756


namespace NUMINAMATH_GPT_shadow_boundary_l87_8775

theorem shadow_boundary (r : ℝ) (O P : ℝ × ℝ × ℝ) :
  r = 2 → O = (0, 0, 2) → P = (0, -2, 4) → ∀ x : ℝ, ∃ y : ℝ, y = -10 :=
by sorry

end NUMINAMATH_GPT_shadow_boundary_l87_8775


namespace NUMINAMATH_GPT_ella_age_l87_8702

theorem ella_age (s t e : ℕ) (h1 : s + t + e = 36) (h2 : e - 5 = s) (h3 : t + 4 = (3 * (s + 4)) / 4) : e = 15 := by
  sorry

end NUMINAMATH_GPT_ella_age_l87_8702


namespace NUMINAMATH_GPT_problem_proof_l87_8706

theorem problem_proof (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = 3) : 3 * a^2 * b + 3 * a * b^2 = 18 := 
by
  sorry

end NUMINAMATH_GPT_problem_proof_l87_8706


namespace NUMINAMATH_GPT_change_occurs_in_3_years_l87_8782

theorem change_occurs_in_3_years (P A1 A2 : ℝ) (R T : ℝ) (h1 : P = 825) (h2 : A1 = 956) (h3 : A2 = 1055)
    (h4 : A1 = P + (P * R * T) / 100)
    (h5 : A2 = P + (P * (R + 4) * T) / 100) : T = 3 :=
by
  sorry

end NUMINAMATH_GPT_change_occurs_in_3_years_l87_8782


namespace NUMINAMATH_GPT_sufficient_condition_for_perpendicular_l87_8751

variables (m n : Line) (α β : Plane)

def are_parallel (l1 l2 : Line) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

theorem sufficient_condition_for_perpendicular :
  (are_parallel m n) ∧ (line_perpendicular_to_plane n α) → (line_perpendicular_to_plane m α) :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_perpendicular_l87_8751


namespace NUMINAMATH_GPT_bad_games_count_l87_8795

/-- 
  Oliver bought a total of 11 video games, and 6 of them worked.
  Prove that the number of bad games he bought is 5.
-/
theorem bad_games_count (total_games : ℕ) (working_games : ℕ) (h1 : total_games = 11) (h2 : working_games = 6) : total_games - working_games = 5 :=
by
  sorry

end NUMINAMATH_GPT_bad_games_count_l87_8795


namespace NUMINAMATH_GPT_arccos_half_eq_pi_div_three_l87_8758

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end NUMINAMATH_GPT_arccos_half_eq_pi_div_three_l87_8758


namespace NUMINAMATH_GPT_square_area_l87_8704

theorem square_area (x : ℝ) (G H : ℝ) (hyp_1 : 0 ≤ G) (hyp_2 : G ≤ x) (hyp_3 : 0 ≤ H) (hyp_4 : H ≤ x) (AG : ℝ) (GH : ℝ) (HD : ℝ)
  (hyp_5 : AG = 20) (hyp_6 : GH = 20) (hyp_7 : HD = 20) (hyp_8 : x = 20 * Real.sqrt 2) :
  x^2 = 800 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l87_8704


namespace NUMINAMATH_GPT_smallest_digit_divisible_by_9_l87_8726

theorem smallest_digit_divisible_by_9 : 
  ∃ d : ℕ, (∃ m : ℕ, m = 2 + 4 + d + 6 + 0 ∧ m % 9 = 0 ∧ d < 10) ∧ d = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_digit_divisible_by_9_l87_8726


namespace NUMINAMATH_GPT_has_root_in_interval_l87_8738

def f (x : ℝ) := x^3 - 3*x - 3

theorem has_root_in_interval : ∃ c ∈ (Set.Ioo (2:ℝ) 3), f c = 0 :=
by 
    sorry

end NUMINAMATH_GPT_has_root_in_interval_l87_8738


namespace NUMINAMATH_GPT_no_fractional_linear_function_l87_8764

noncomputable def fractional_linear_function (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem no_fractional_linear_function (a b c d : ℝ) :
  ∀ x : ℝ, c ≠ 0 → 
  (fractional_linear_function a b c d x + fractional_linear_function b (-d) c (-a) x ≠ -2) :=
by
  sorry

end NUMINAMATH_GPT_no_fractional_linear_function_l87_8764


namespace NUMINAMATH_GPT_polynomial_form_l87_8708

theorem polynomial_form (P : ℝ → ℝ) (h₁ : P 0 = 0) (h₂ : ∀ x, P x = (P (x + 1) + P (x - 1)) / 2) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
sorry

end NUMINAMATH_GPT_polynomial_form_l87_8708


namespace NUMINAMATH_GPT_composite_number_property_l87_8761

theorem composite_number_property (n : ℕ) 
  (h1 : n > 1) 
  (h2 : ¬ Prime n) 
  (h3 : ∀ (d : ℕ), d ∣ n → 1 ≤ d → d < n → n - 20 ≤ d ∧ d ≤ n - 12) :
  n = 21 ∨ n = 25 :=
by
  sorry

end NUMINAMATH_GPT_composite_number_property_l87_8761


namespace NUMINAMATH_GPT_game_terminates_if_n_lt_1994_game_does_not_terminate_if_n_eq_1994_l87_8710

-- Definitions and conditions for the problem
def num_girls : ℕ := 1994
def tokens (n : ℕ) := n

-- Main theorem statements
theorem game_terminates_if_n_lt_1994 (n : ℕ) (h : n < num_girls) :
  ∃ (S : ℕ) (invariant : ℕ) (steps : ℕ), (∀ j : ℕ, 1 ≤ j ∧ j ≤ num_girls → (tokens n % num_girls) ≤ 1) :=
by
  sorry

theorem game_does_not_terminate_if_n_eq_1994 :
  ∃ (S : ℕ) (invariant : ℕ) (steps : ℕ), (tokens 1994 % num_girls = 0) :=
by
  sorry

end NUMINAMATH_GPT_game_terminates_if_n_lt_1994_game_does_not_terminate_if_n_eq_1994_l87_8710


namespace NUMINAMATH_GPT_jane_doe_gift_l87_8700

theorem jane_doe_gift (G : ℝ) (h1 : 0.25 * G + 0.1125 * (0.75 * G) = 15000) : G = 41379 := 
sorry

end NUMINAMATH_GPT_jane_doe_gift_l87_8700


namespace NUMINAMATH_GPT_find_x_in_equation_l87_8739

theorem find_x_in_equation :
  ∃ x : ℝ, x / 18 * (x / 162) = 1 ∧ x = 54 :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_equation_l87_8739


namespace NUMINAMATH_GPT_filling_material_heavier_than_sand_l87_8737

noncomputable def percentage_increase (full_sandbag_weight : ℝ) (partial_fill_percent : ℝ) (full_material_weight : ℝ) : ℝ :=
  let sand_weight := (partial_fill_percent / 100) * full_sandbag_weight
  let material_weight := full_material_weight
  let weight_increase := material_weight - sand_weight
  (weight_increase / sand_weight) * 100

theorem filling_material_heavier_than_sand :
  let full_sandbag_weight := 250
  let partial_fill_percent := 80
  let full_material_weight := 280
  percentage_increase full_sandbag_weight partial_fill_percent full_material_weight = 40 :=
by
  sorry

end NUMINAMATH_GPT_filling_material_heavier_than_sand_l87_8737


namespace NUMINAMATH_GPT_cost_per_book_l87_8763

theorem cost_per_book (num_animal_books : ℕ) (num_space_books : ℕ) (num_train_books : ℕ) (total_cost : ℕ) 
                      (h1 : num_animal_books = 10) (h2 : num_space_books = 1) (h3 : num_train_books = 3) (h4 : total_cost = 224) :
  (total_cost / (num_animal_books + num_space_books + num_train_books) = 16) :=
by sorry

end NUMINAMATH_GPT_cost_per_book_l87_8763


namespace NUMINAMATH_GPT_mix_solutions_l87_8754

variables (Vx : ℚ)

def alcohol_content_x (Vx : ℚ) : ℚ := 0.10 * Vx
def alcohol_content_y : ℚ := 0.30 * 450
def final_alcohol_content (Vx : ℚ) : ℚ := 0.22 * (Vx + 450)

theorem mix_solutions (Vx : ℚ) (h : 0.10 * Vx + 0.30 * 450 = 0.22 * (Vx + 450)) :
  Vx = 300 :=
sorry

end NUMINAMATH_GPT_mix_solutions_l87_8754


namespace NUMINAMATH_GPT_macey_saving_weeks_l87_8719

-- Definitions for conditions
def shirt_cost : ℝ := 3
def amount_saved : ℝ := 1.5
def weekly_saving : ℝ := 0.5

-- Statement of the proof problem
theorem macey_saving_weeks : (shirt_cost - amount_saved) / weekly_saving = 3 := by
  sorry

end NUMINAMATH_GPT_macey_saving_weeks_l87_8719


namespace NUMINAMATH_GPT_frank_reads_pages_per_day_l87_8720

theorem frank_reads_pages_per_day (pages_per_book : ℕ) (days_per_book : ℕ) (h1 : pages_per_book = 249) (h2 : days_per_book = 3) : pages_per_book / days_per_book = 83 :=
by {
  sorry
}

end NUMINAMATH_GPT_frank_reads_pages_per_day_l87_8720


namespace NUMINAMATH_GPT_bridge_length_l87_8767

theorem bridge_length (rate : ℝ) (time_minutes : ℝ) (length : ℝ) 
    (rate_condition : rate = 10) 
    (time_condition : time_minutes = 15) : 
    length = 2.5 := 
by
  sorry

end NUMINAMATH_GPT_bridge_length_l87_8767


namespace NUMINAMATH_GPT_circle_radius_l87_8717

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 180 * π) : r = 10 := 
by
  sorry

end NUMINAMATH_GPT_circle_radius_l87_8717


namespace NUMINAMATH_GPT_remainder_of_large_number_l87_8793

theorem remainder_of_large_number :
  (102938475610 % 12) = 10 :=
by
  have h1 : (102938475610 % 4) = 2 := sorry
  have h2 : (102938475610 % 3) = 1 := sorry
  sorry

end NUMINAMATH_GPT_remainder_of_large_number_l87_8793


namespace NUMINAMATH_GPT_matrix_scalars_exist_l87_8732

namespace MatrixProof

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![4, -1]]

theorem matrix_scalars_exist :
  ∃ r s : ℝ, B^6 = r • B + s • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ r = 0 ∧ s = 64 := by
  sorry

end MatrixProof

end NUMINAMATH_GPT_matrix_scalars_exist_l87_8732


namespace NUMINAMATH_GPT_smallest_four_digit_int_mod_9_l87_8753

theorem smallest_four_digit_int_mod_9 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 5 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 9 = 5 → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_int_mod_9_l87_8753


namespace NUMINAMATH_GPT_loss_per_metre_l87_8705

def total_metres : ℕ := 500
def selling_price : ℕ := 18000
def cost_price_per_metre : ℕ := 41

theorem loss_per_metre :
  (cost_price_per_metre * total_metres - selling_price) / total_metres = 5 :=
by sorry

end NUMINAMATH_GPT_loss_per_metre_l87_8705


namespace NUMINAMATH_GPT_volume_of_circumscribed_sphere_l87_8765

theorem volume_of_circumscribed_sphere (vol_cube : ℝ) (h : vol_cube = 8) :
  ∃ (vol_sphere : ℝ), vol_sphere = 4 * Real.sqrt 3 * Real.pi := 
sorry

end NUMINAMATH_GPT_volume_of_circumscribed_sphere_l87_8765


namespace NUMINAMATH_GPT_inequality_solution_l87_8748

theorem inequality_solution (x : ℝ) (h : 0 < x) : x^3 - 9*x^2 + 52*x > 0 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l87_8748


namespace NUMINAMATH_GPT_cody_books_reading_l87_8723

theorem cody_books_reading :
  ∀ (total_books first_week_books second_week_books subsequent_week_books : ℕ),
    total_books = 54 →
    first_week_books = 6 →
    second_week_books = 3 →
    subsequent_week_books = 9 →
    (2 + (total_books - (first_week_books + second_week_books)) / subsequent_week_books) = 7 :=
by
  -- Using sorry to mark the proof as incomplete.
  sorry

end NUMINAMATH_GPT_cody_books_reading_l87_8723


namespace NUMINAMATH_GPT_construct_points_PQ_l87_8716

-- Given Conditions
variable (a b c : ℝ)
def triangle_ABC_conditions : Prop := 
  let s := (a + b + c) / 2
  s^2 ≥ 2 * a * b

-- Main Statement
theorem construct_points_PQ (a b c : ℝ) (P Q : ℝ) 
(h1 : triangle_ABC_conditions a b c) :
  let s := (a + b + c) / 2
  let x := (s + Real.sqrt (s^2 - 2 * a * b)) / 2
  let y := (s - Real.sqrt (s^2 - 2 * a * b)) / 2
  x + y = s ∧ x * y = (a * b) / 2 :=
by
  sorry

end NUMINAMATH_GPT_construct_points_PQ_l87_8716
