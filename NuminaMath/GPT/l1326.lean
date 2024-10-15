import Mathlib

namespace NUMINAMATH_GPT_battery_change_month_battery_change_in_november_l1326_132626

theorem battery_change_month :
  (119 % 12) = 11 := by
  sorry

theorem battery_change_in_november (n : Nat) (h1 : n = 18) :
  let month := ((n - 1) * 7) % 12
  month = 11 := by
  sorry

end NUMINAMATH_GPT_battery_change_month_battery_change_in_november_l1326_132626


namespace NUMINAMATH_GPT_percentage_j_of_k_theorem_l1326_132639

noncomputable def percentage_j_of_k 
  (j k l m : ℝ) (x : ℝ) 
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100)) : Prop :=
  x = 500

theorem percentage_j_of_k_theorem 
  (j k l m : ℝ) (x : ℝ)
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100)) : percentage_j_of_k j k l m x h1 h2 h3 h4 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_j_of_k_theorem_l1326_132639


namespace NUMINAMATH_GPT_evaluate_expression_l1326_132628

theorem evaluate_expression : (2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9) := by
    sorry

end NUMINAMATH_GPT_evaluate_expression_l1326_132628


namespace NUMINAMATH_GPT_total_cost_of_tennis_balls_l1326_132698

theorem total_cost_of_tennis_balls
  (packs : ℕ) (balls_per_pack : ℕ) (cost_per_ball : ℕ)
  (h1 : packs = 4) (h2 : balls_per_pack = 3) (h3 : cost_per_ball = 2) : 
  packs * balls_per_pack * cost_per_ball = 24 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_tennis_balls_l1326_132698


namespace NUMINAMATH_GPT_ball_probability_l1326_132646

theorem ball_probability :
  ∀ (total_balls red_balls white_balls : ℕ),
  total_balls = 10 → red_balls = 6 → white_balls = 4 →
  -- Given conditions: Total balls, red balls, and white balls.
  -- First ball drawn is red
  ∀ (first_ball_red : true),
  -- Prove that the probability of the second ball being red is 5/9.
  (red_balls - 1) / (total_balls - 1) = 5/9 :=
by
  intros total_balls red_balls white_balls h_total h_red h_white first_ball_red
  sorry

end NUMINAMATH_GPT_ball_probability_l1326_132646


namespace NUMINAMATH_GPT_interval_monotonic_decrease_min_value_g_l1326_132635

noncomputable def a (x : ℝ) : ℝ × ℝ := (3 * Real.sqrt 3 * Real.sin x, Real.sqrt 3 * Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := let (a1, a2) := a x; let (b1, b2) := b x; a1 * b1 + a2 * b2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x + m

theorem interval_monotonic_decrease (x : ℝ) (k : ℤ) :
  0 ≤ x ∧ x ≤ Real.pi ∧ (2 * x + Real.pi / 6) ∈ [Real.pi/2 + 2 * (k : ℝ) * Real.pi, 3 * Real.pi/2 + 2 * (k : ℝ) * Real.pi] →
  x ∈ [Real.pi / 6 + (k : ℝ) * Real.pi, 2 * Real.pi / 3 + (k : ℝ) * Real.pi] := sorry

theorem min_value_g (x : ℝ) :
  x ∈ [- Real.pi / 3, Real.pi / 3] →
  ∃ x₀, g x₀ 1 = -1/2 ∧ x₀ = - Real.pi / 3 := sorry

end NUMINAMATH_GPT_interval_monotonic_decrease_min_value_g_l1326_132635


namespace NUMINAMATH_GPT_division_identity_l1326_132666

theorem division_identity :
  (0.0048 * 3.5) / (0.05 * 0.1 * 0.004) = 840 :=
by
  -- TODO: Provide the proof here
  sorry

end NUMINAMATH_GPT_division_identity_l1326_132666


namespace NUMINAMATH_GPT_intersection_A_complement_B_l1326_132638

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 > 0}
def comR (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

theorem intersection_A_complement_B : A ∩ (comR B) = {x | -1 < x ∧ x ≤ 3} := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_complement_B_l1326_132638


namespace NUMINAMATH_GPT_tickets_spent_l1326_132610

theorem tickets_spent (initial_tickets : ℕ) (tickets_left : ℕ) (tickets_spent : ℕ) 
  (h1 : initial_tickets = 11) (h2 : tickets_left = 8) : tickets_spent = 3 :=
by
  sorry

end NUMINAMATH_GPT_tickets_spent_l1326_132610


namespace NUMINAMATH_GPT_problem_statement_l1326_132667

-- Define A as the number of four-digit odd numbers
def A : ℕ := 4500

-- Define B as the number of four-digit multiples of 3
def B : ℕ := 3000

-- The main theorem stating the sum A + B equals 7500
theorem problem_statement : A + B = 7500 := by
  -- The exact proof is omitted using sorry
  sorry

end NUMINAMATH_GPT_problem_statement_l1326_132667


namespace NUMINAMATH_GPT_sin_cos_sum_identity_l1326_132631

noncomputable def trigonometric_identity (x y z w : ℝ) := 
  (Real.sin x * Real.cos y + Real.sin z * Real.cos w) = Real.sqrt 2 / 2

theorem sin_cos_sum_identity :
  trigonometric_identity 347 148 77 58 :=
by sorry

end NUMINAMATH_GPT_sin_cos_sum_identity_l1326_132631


namespace NUMINAMATH_GPT_earnings_from_cauliflower_correct_l1326_132668

-- Define the earnings from each vegetable
def earnings_from_broccoli : ℕ := 57
def earnings_from_carrots : ℕ := 2 * earnings_from_broccoli
def earnings_from_spinach : ℕ := (earnings_from_carrots / 2) + 16
def total_earnings : ℕ := 380

-- Define the total earnings from vegetables other than cauliflower
def earnings_from_others : ℕ := earnings_from_broccoli + earnings_from_carrots + earnings_from_spinach

-- Define the earnings from cauliflower
def earnings_from_cauliflower : ℕ := total_earnings - earnings_from_others

-- Theorem to prove the earnings from cauliflower
theorem earnings_from_cauliflower_correct : earnings_from_cauliflower = 136 :=
by
  sorry

end NUMINAMATH_GPT_earnings_from_cauliflower_correct_l1326_132668


namespace NUMINAMATH_GPT_range_of_k_l1326_132637

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x > k → (3 / (x + 1) < 1)) ↔ k ≥ 2 := sorry

end NUMINAMATH_GPT_range_of_k_l1326_132637


namespace NUMINAMATH_GPT_smallest_lambda_inequality_l1326_132652

theorem smallest_lambda_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x * y * (x^2 + y^2) + y * z * (y^2 + z^2) + z * x * (z^2 + x^2) ≤ (1 / 8) * (x + y + z)^4 :=
sorry

end NUMINAMATH_GPT_smallest_lambda_inequality_l1326_132652


namespace NUMINAMATH_GPT_division_of_fractions_l1326_132640

theorem division_of_fractions : (5 / 6) / (1 + 3 / 9) = 5 / 8 := by
  sorry

end NUMINAMATH_GPT_division_of_fractions_l1326_132640


namespace NUMINAMATH_GPT_solve_q_l1326_132636

-- Definitions of conditions
variable (p q : ℝ)
variable (k : ℝ) 

-- Initial conditions
axiom h1 : p = 1500
axiom h2 : q = 0.5
axiom h3 : p * q = k
axiom h4 : k = 750

-- Goal
theorem solve_q (hp : p = 3000) : q = 0.250 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_solve_q_l1326_132636


namespace NUMINAMATH_GPT_problem_solution_l1326_132664

-- Definitions based on conditions
def p (a b : ℝ) : Prop := a > b → a^2 > b^2
def neg_p (a b : ℝ) : Prop := a > b → a^2 ≤ b^2
def disjunction (p q : Prop) : Prop := p ∨ q
def suff_but_not_nec (x : ℝ) : Prop := x > 2 → x > 1 ∧ ¬(x > 1 → x > 2)
def congruent_triangles (T1 T2 : Prop) : Prop := T1 → T2
def neg_congruent_triangles (T1 T2 : Prop) : Prop := ¬(T1 → T2)

-- Mathematical problem as Lean statements
theorem problem_solution :
  ( (∀ a b : ℝ, p a b = (a > b → a^2 > b^2) ∧ neg_p a b = (a > b → a^2 ≤ b^2)) ∧
    (∀ p q : Prop, (disjunction p q) = false → p = false ∧ q = false) ∧
    (∀ x : ℝ, suff_but_not_nec x = (x > 2 → x > 1 ∧ ¬(x > 1 → x > 2))) ∧
    (∀ T1 T2 : Prop, (neg_congruent_triangles T1 T2) = true ↔ ¬(T1 → T2)) ) →
  ( (∀ a b : ℝ, neg_p a b = (a > b → a^2 ≤ b^2)) ∧
    (∀ p q : Prop, (disjunction p q) = false → p = false ∧ q = false) ∧
    (∀ x : ℝ, suff_but_not_nec x = (x > 2 → x > 1 ∧ ¬(x > 1 → x > 2))) ∧
    (∀ T1 T2 : Prop, (neg_congruent_triangles T1 T2) = false) ) :=
sorry

end NUMINAMATH_GPT_problem_solution_l1326_132664


namespace NUMINAMATH_GPT_seq_a_2012_value_l1326_132675

theorem seq_a_2012_value :
  ∀ (a : ℕ → ℕ),
  (a 1 = 0) →
  (∀ n : ℕ, a (n + 1) = a n + 2 * n) →
  a 2012 = 2011 * 2012 :=
by
  intros a h₁ h₂
  sorry

end NUMINAMATH_GPT_seq_a_2012_value_l1326_132675


namespace NUMINAMATH_GPT_compound_interest_correct_l1326_132633

variables (SI : ℚ) (R : ℚ) (T : ℕ) (P : ℚ)

def calculate_principal (SI R T : ℚ) : ℚ := SI * 100 / (R * T)

def calculate_compound_interest (P R : ℚ) (T : ℕ) : ℚ :=
  P * ((1 + R / 100)^T - 1)

theorem compound_interest_correct (h1: SI = 52) (h2: R = 5) (h3: T = 2) :
  calculate_compound_interest (calculate_principal SI R T) R T = 53.30 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_correct_l1326_132633


namespace NUMINAMATH_GPT_min_translation_phi_l1326_132649

theorem min_translation_phi (φ : ℝ) (hφ : φ > 0) : 
  (∃ k : ℤ, φ = (π / 3) - k * π) → φ = π / 3 := 
by 
  sorry

end NUMINAMATH_GPT_min_translation_phi_l1326_132649


namespace NUMINAMATH_GPT_sin_cos_term_side_l1326_132641

theorem sin_cos_term_side (a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, (k = 2 * (if a > 0 then -3/5 else 3/5) + (if a > 0 then 4/5 else -4/5)) ∧ (k = 2/5 ∨ k = -2/5) := by
  sorry

end NUMINAMATH_GPT_sin_cos_term_side_l1326_132641


namespace NUMINAMATH_GPT_g_self_inverse_if_one_l1326_132634

variables (f : ℝ → ℝ) (symm_about : ∀ x, f (f x) = x - 1)

def g (b : ℝ) (x : ℝ) : ℝ := f (x + b)

theorem g_self_inverse_if_one (b : ℝ) :
  (∀ x, g f b (g f b x) = x) ↔ b = 1 := 
by
  sorry

end NUMINAMATH_GPT_g_self_inverse_if_one_l1326_132634


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l1326_132696

variables (A B C : ℝ) (a b c : ℝ) (R : ℝ) (area : ℝ)

-- Given conditions
def sides_ratio := a / b = 7 / 5 ∧ b / c = 5 / 3
def triangle_area := area = 45 * Real.sqrt 3
def sides := (a, b, c)
def angles := (A, B, C)

-- Prove radius
theorem circumscribed_circle_radius 
  (h_ratio : sides_ratio a b c)
  (h_area : triangle_area area) :
  R = 14 :=
sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l1326_132696


namespace NUMINAMATH_GPT_shaded_area_ratio_l1326_132663

-- Definitions based on conditions
def large_square_area : ℕ := 16
def shaded_components : ℕ := 4
def component_fraction : ℚ := 1 / 2
def shaded_square_area : ℚ := shaded_components * component_fraction
def large_square_area_q : ℚ := large_square_area

-- Goal statement
theorem shaded_area_ratio : (shaded_square_area / large_square_area_q) = (1 / 8) :=
by sorry

end NUMINAMATH_GPT_shaded_area_ratio_l1326_132663


namespace NUMINAMATH_GPT_find_time_when_velocity_is_one_l1326_132682

-- Define the equation of motion
def equation_of_motion (t : ℝ) : ℝ := 7 * t^2 + 8

-- Define the velocity function as the derivative of the equation of motion
def velocity (t : ℝ) : ℝ := by
  let s := equation_of_motion t
  exact 14 * t  -- Since we calculated the derivative above

-- Statement of the problem to be proved
theorem find_time_when_velocity_is_one : 
  (velocity (1 / 14)) = 1 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_time_when_velocity_is_one_l1326_132682


namespace NUMINAMATH_GPT_solve_equation_l1326_132690

theorem solve_equation (x : ℝ) : (⌊Real.sin x⌋:ℝ)^2 = Real.cos x ^ 2 - 1 ↔ ∃ n : ℤ, x = n * Real.pi := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1326_132690


namespace NUMINAMATH_GPT_range_of_2a_sub_b_l1326_132603

theorem range_of_2a_sub_b (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 2) : -4 < 2 * a - b ∧ 2 * a - b < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_2a_sub_b_l1326_132603


namespace NUMINAMATH_GPT_tiffany_initial_lives_l1326_132694

theorem tiffany_initial_lives (x : ℕ) 
    (H1 : x - 14 + 27 = 56) : x = 43 :=
sorry

end NUMINAMATH_GPT_tiffany_initial_lives_l1326_132694


namespace NUMINAMATH_GPT_rectangle_area_unchanged_l1326_132611

theorem rectangle_area_unchanged (x y : ℕ) (h1 : x * y = (x + 5/2) * (y - 2/3)) (h2 : x * y = (x - 5/2) * (y + 4/3)) : x * y = 20 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_unchanged_l1326_132611


namespace NUMINAMATH_GPT_roller_coaster_costs_4_l1326_132612

-- Definitions from conditions
def tickets_initial: ℕ := 5                     -- Jeanne initially has 5 tickets
def tickets_to_buy: ℕ := 8                      -- Jeanne needs to buy 8 more tickets
def total_tickets_needed: ℕ := tickets_initial + tickets_to_buy -- Total tickets needed
def tickets_ferris_wheel: ℕ := 5                -- Ferris wheel costs 5 tickets
def tickets_total_after_ferris_wheel: ℕ := total_tickets_needed - tickets_ferris_wheel -- Remaining tickets after Ferris wheel

-- Definition to be proved (question = answer)
def cost_roller_coaster_bumper_cars: ℕ := tickets_total_after_ferris_wheel / 2 -- Each of roller coaster and bumper cars cost

-- The theorem that corresponds to the solution
theorem roller_coaster_costs_4 :
  cost_roller_coaster_bumper_cars = 4 :=
by
  sorry

end NUMINAMATH_GPT_roller_coaster_costs_4_l1326_132612


namespace NUMINAMATH_GPT_examination_is_30_hours_l1326_132648

noncomputable def examination_time_in_hours : ℝ :=
  let total_questions := 200
  let type_a_problems := 10
  let total_time_on_type_a := 17.142857142857142
  let time_per_type_a := total_time_on_type_a / type_a_problems
  let time_per_type_b := time_per_type_a / 2
  let type_b_problems := total_questions - type_a_problems
  let total_time_on_type_b := time_per_type_b * type_b_problems
  let total_time_in_minutes := total_time_on_type_a * type_a_problems + total_time_on_type_b
  total_time_in_minutes / 60

theorem examination_is_30_hours :
  examination_time_in_hours = 30 := by
  sorry

end NUMINAMATH_GPT_examination_is_30_hours_l1326_132648


namespace NUMINAMATH_GPT_coin_value_is_630_l1326_132604

theorem coin_value_is_630 :
  (∃ x : ℤ, x > 0 ∧ 406 * x = 63000) :=
by {
  sorry
}

end NUMINAMATH_GPT_coin_value_is_630_l1326_132604


namespace NUMINAMATH_GPT_carrots_total_l1326_132620

theorem carrots_total (sandy_carrots : Nat) (sam_carrots : Nat) (h1 : sandy_carrots = 6) (h2 : sam_carrots = 3) :
  sandy_carrots + sam_carrots = 9 :=
by
  sorry

end NUMINAMATH_GPT_carrots_total_l1326_132620


namespace NUMINAMATH_GPT_range_f_contained_in_0_1_l1326_132632

theorem range_f_contained_in_0_1 (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_range_f_contained_in_0_1_l1326_132632


namespace NUMINAMATH_GPT_product_of_consecutive_integers_sqrt_50_l1326_132671

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_sqrt_50_l1326_132671


namespace NUMINAMATH_GPT_max_value_of_a_l1326_132642

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) ∧
  (∃ x : ℝ, x < a ∧ ¬(x^2 - 2*x - 3 > 0)) →
  a = -1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_a_l1326_132642


namespace NUMINAMATH_GPT_gcd_182_98_l1326_132655

theorem gcd_182_98 : Nat.gcd 182 98 = 14 :=
by
  -- Provide the proof here, but as per instructions, we'll use sorry to skip it.
  sorry

end NUMINAMATH_GPT_gcd_182_98_l1326_132655


namespace NUMINAMATH_GPT_find_m_l1326_132660

theorem find_m
  (θ : Real)
  (m : Real)
  (h_sin_cos_roots : ∀ x : Real, 4 * x^2 + 2 * m * x + m = 0 → x = Real.sin θ ∨ x = Real.cos θ)
  (h_real_roots : ∃ x : Real, 4 * x^2 + 2 * m * x + m = 0) :
  m = 1 - Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_find_m_l1326_132660


namespace NUMINAMATH_GPT_even_function_zero_coefficient_l1326_132688

theorem even_function_zero_coefficient: ∀ a : ℝ, (∀ x : ℝ, (x^2 + a * x + 1) = ((-x)^2 + a * (-x) + 1)) → a = 0 :=
by
  intros a h
  sorry

end NUMINAMATH_GPT_even_function_zero_coefficient_l1326_132688


namespace NUMINAMATH_GPT_a_eq_b_if_conditions_l1326_132618

theorem a_eq_b_if_conditions (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := 
sorry

end NUMINAMATH_GPT_a_eq_b_if_conditions_l1326_132618


namespace NUMINAMATH_GPT_exists_numbers_with_prime_sum_and_product_l1326_132685

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem exists_numbers_with_prime_sum_and_product :
  ∃ a b c : ℕ, is_prime (a + b + c) ∧ is_prime (a * b * c) :=
  by
    -- First import the prime definitions and variables.
    let a := 1
    let b := 1
    let c := 3
    have h1 : is_prime (a + b + c) := by sorry
    have h2 : is_prime (a * b * c) := by sorry
    exact ⟨a, b, c, h1, h2⟩

end NUMINAMATH_GPT_exists_numbers_with_prime_sum_and_product_l1326_132685


namespace NUMINAMATH_GPT_element_in_set_l1326_132679

variable (A : Set ℕ) (a b : ℕ)
def condition : Prop := A = {a, b, 1}

theorem element_in_set (h : condition A a b) : 1 ∈ A :=
by sorry

end NUMINAMATH_GPT_element_in_set_l1326_132679


namespace NUMINAMATH_GPT_quadratic_linear_common_solution_l1326_132691

theorem quadratic_linear_common_solution
  (a x1 x2 d e : ℝ)
  (ha : a ≠ 0) (hx1x2 : x1 ≠ x2) (hd : d ≠ 0)
  (h_quad : ∀ x, a * (x - x1) * (x - x2) = 0 → x = x1 ∨ x = x2)
  (h_linear : d * x1 + e = 0)
  (h_combined : ∀ x, a * (x - x1) * (x - x2) + d * x + e = 0 → x = x1) :
  d = a * (x2 - x1) :=
by sorry

end NUMINAMATH_GPT_quadratic_linear_common_solution_l1326_132691


namespace NUMINAMATH_GPT_apples_left_is_ten_l1326_132615

noncomputable def appleCost : ℝ := 0.80
noncomputable def orangeCost : ℝ := 0.50
def initialApples : ℕ := 50
def initialOranges : ℕ := 40
def totalEarnings : ℝ := 49
def orangesLeft : ℕ := 6

theorem apples_left_is_ten (A : ℕ) :
  (50 - A) * appleCost + (40 - orangesLeft) * orangeCost = 49 → A = 10 :=
by
  sorry

end NUMINAMATH_GPT_apples_left_is_ten_l1326_132615


namespace NUMINAMATH_GPT_solve_inequality_1_solve_inequality_2_l1326_132689

-- Definitions based on given conditions
noncomputable def f (x : ℝ) : ℝ := abs (x + 1)

-- Lean statement for the first proof problem
theorem solve_inequality_1 :
  ∀ x : ℝ, f x ≤ 5 - f (x - 3) ↔ -2 ≤ x ∧ x ≤ 3 :=
by
  sorry

-- Lean statement for the second proof problem
theorem solve_inequality_2 (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ 2 * f x + abs (x + a) ≤ x + 4) ↔ -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_1_solve_inequality_2_l1326_132689


namespace NUMINAMATH_GPT_mens_wages_l1326_132659

theorem mens_wages
  (M : ℝ) (WW : ℝ) (B : ℝ)
  (h1 : 5 * M = WW)
  (h2 : WW = 8 * B)
  (h3 : 5 * M + WW + 8 * B = 60) :
  5 * M = 30 :=
by
  sorry

end NUMINAMATH_GPT_mens_wages_l1326_132659


namespace NUMINAMATH_GPT_three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3_l1326_132607

theorem three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3 :
  ∃ x1 x2 x3 : ℕ, ((x1 = 414 ∧ x2 = 444 ∧ x3 = 474) ∧ 
  (∀ n, (100 * 4 + 10 * n + 4 = x1 ∨ 100 * 4 + 10 * n + 4 = x2 ∨ 100 * 4 + 10 * n + 4 = x3) 
  → (100 * 4 + 10 * n + 4) % 3 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3_l1326_132607


namespace NUMINAMATH_GPT_sum_of_values_l1326_132613

theorem sum_of_values (N : ℝ) (R : ℝ) (hN : N ≠ 0) (h_eq : N - 3 / N = R) :
  let N1 := (-R + Real.sqrt (R^2 + 12)) / 2
  let N2 := (-R - Real.sqrt (R^2 + 12)) / 2
  N1 + N2 = R :=
by
  sorry

end NUMINAMATH_GPT_sum_of_values_l1326_132613


namespace NUMINAMATH_GPT_smallest_possible_w_l1326_132673

theorem smallest_possible_w 
  (h1 : 936 = 2^3 * 3 * 13)
  (h2 : 2^5 = 32)
  (h3 : 3^3 = 27)
  (h4 : 14^2 = 196) :
  ∃ w : ℕ, (w > 0) ∧ (936 * w) % 32 = 0 ∧ (936 * w) % 27 = 0 ∧ (936 * w) % 196 = 0 ∧ w = 1764 :=
sorry

end NUMINAMATH_GPT_smallest_possible_w_l1326_132673


namespace NUMINAMATH_GPT_inequality_proof_l1326_132680

open Real

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h1 : a ^ x = b * c) 
  (h2 : b ^ y = c * a) 
  (h3 : c ^ z = a * b) :
  (1 / (2 + x) + 1 / (2 + y) + 1 / (2 + z)) ≤ 3 / 4 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1326_132680


namespace NUMINAMATH_GPT_functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l1326_132653

-- Definitions for the problem conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def max_selling_price : ℝ := 38
def base_sales_volume : ℝ := 250
def price_decrease_effect : ℝ := 10
def profit_requirement : ℝ := 2000

-- Given the initial conditions
noncomputable def sales_volume (x : ℝ) : ℝ := base_sales_volume - price_decrease_effect * (x - min_selling_price)

-- Target problem statement
-- Part 1: Functional relationship between y and x
theorem functional_relationship (x : ℝ) : sales_volume x = -10 * x + 500 := by
sorry

-- Part 2: Maximizing profit
noncomputable def profit (x : ℝ) : ℝ := (x - cost_per_box) * sales_volume x

theorem maximizing_profit : ∃ (x : ℝ), x = 35 ∧ profit x = 2250 := by
sorry

-- Part 3: Minimum number of boxes to sell for at least 2000 yuan profit
theorem minimum_boxes_for_2000_profit (x : ℝ) : x ≤ max_selling_price → profit x ≥ profit_requirement → sales_volume x ≥ 120 := by
sorry

end NUMINAMATH_GPT_functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l1326_132653


namespace NUMINAMATH_GPT_set_intersection_l1326_132619
noncomputable def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2 }
noncomputable def B : Set ℝ := {x : ℝ | x ≥ 1 }

theorem set_intersection (x : ℝ) : x ∈ A ∩ B ↔ x ∈ A := sorry

end NUMINAMATH_GPT_set_intersection_l1326_132619


namespace NUMINAMATH_GPT_problem_l1326_132650

def f (x : ℝ) : ℝ := (x^4 + 2*x^3 + 4*x - 5) ^ 2004 + 2004

theorem problem (x : ℝ) (h : x = Real.sqrt 3 - 1) : f x = 2005 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1326_132650


namespace NUMINAMATH_GPT_price_decrease_percentage_l1326_132651

-- Define the conditions
variables {P : ℝ} (original_price increased_price decreased_price : ℝ)
variables (y : ℝ) -- percentage by which increased price is decreased

-- Given conditions
def store_conditions :=
  increased_price = 1.20 * original_price ∧
  decreased_price = increased_price * (1 - y/100) ∧
  decreased_price = 0.75 * original_price

-- The proof problem
theorem price_decrease_percentage 
  (original_price increased_price decreased_price : ℝ)
  (y : ℝ) 
  (h : store_conditions original_price increased_price decreased_price y) :
  y = 37.5 :=
by 
  sorry

end NUMINAMATH_GPT_price_decrease_percentage_l1326_132651


namespace NUMINAMATH_GPT_find_may_monday_l1326_132699

noncomputable def weekday (day_of_month : ℕ) (first_day_weekday : ℕ) : ℕ :=
(day_of_month + first_day_weekday - 1) % 7

theorem find_may_monday (r n : ℕ) (condition1 : weekday r 5 = 5) (condition2 : weekday n 5 = 1) (condition3 : 15 < n ∧ n < 25) : 
  n = 20 :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_find_may_monday_l1326_132699


namespace NUMINAMATH_GPT_forty_percent_of_number_l1326_132697

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 17) : 0.4 * N = 204 :=
sorry

end NUMINAMATH_GPT_forty_percent_of_number_l1326_132697


namespace NUMINAMATH_GPT_range_of_a_l1326_132681

-- Define the set A
def A (a x : ℝ) := 6 * x + a > 0

-- Theorem stating the range of a given the conditions
theorem range_of_a (a : ℝ) (h : ¬ A a 1) : a ≤ -6 :=
by
  -- Here we would provide the proof
  sorry

end NUMINAMATH_GPT_range_of_a_l1326_132681


namespace NUMINAMATH_GPT_vector_CD_l1326_132662

-- Define the vector space and the vectors a and b
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)
variables (a b : V)

-- Define the conditions
def is_on_line (D A B : V) := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (D = t • A + (1 - t) • B)
def da_eq_2bd (D A B : V) := (A - D) = 2 • (D - B)

-- Define the triangle ABC and the specific vectors CA and CB
variables (CA := C - A) (CB := C - B)
variable (H1 : is_on_line D A B)
variable (H2 : da_eq_2bd D A B)
variable (H3 : CA = a)
variable (H4 : CB = b)

-- Prove the conclusion
theorem vector_CD (H1 : is_on_line D A B) (H2 : da_eq_2bd D A B)
  (H3 : CA = a) (H4 : CB = b) : 
  (C - D) = (1/3 : ℝ) • a + (2/3 : ℝ) • b :=
sorry

end NUMINAMATH_GPT_vector_CD_l1326_132662


namespace NUMINAMATH_GPT_numberOfCows_l1326_132602

-- Definitions coming from the conditions
def hasFoxes (n : Nat) := n = 15
def zebrasFromFoxes (z f : Nat) := z = 3 * f
def totalAnimalRequirement (total : Nat) := total = 100
def addedSheep (s : Nat) := s = 20

-- Theorem stating the desired proof
theorem numberOfCows (f z total s c : Nat) 
 (h1 : hasFoxes f)
 (h2 : zebrasFromFoxes z f) 
 (h3 : totalAnimalRequirement total) 
 (h4 : addedSheep s) :
 c = total - s - (f + z) := by
 sorry

end NUMINAMATH_GPT_numberOfCows_l1326_132602


namespace NUMINAMATH_GPT_number_of_triangles_in_decagon_l1326_132614

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end NUMINAMATH_GPT_number_of_triangles_in_decagon_l1326_132614


namespace NUMINAMATH_GPT_find_k_l1326_132665

theorem find_k (k : ℝ) : 2 + (2 + k) / 3 + (2 + 2 * k) / 3^2 + (2 + 3 * k) / 3^3 + 
  ∑' (n : ℕ), (2 + (n + 1) * k) / 3^(n + 1) = 7 ↔ k = 16 / 3 := 
sorry

end NUMINAMATH_GPT_find_k_l1326_132665


namespace NUMINAMATH_GPT_initial_wage_illiterate_l1326_132625

variable (I : ℕ) -- initial daily average wage of illiterate employees

theorem initial_wage_illiterate (h1 : 20 * I - 20 * 10 = 300) : I = 25 :=
by
  simp at h1
  sorry

end NUMINAMATH_GPT_initial_wage_illiterate_l1326_132625


namespace NUMINAMATH_GPT_moles_ethane_and_hexachloroethane_l1326_132695

-- Define the conditions
def balanced_eq (a b c d : ℕ) : Prop :=
  a * 6 = b ∧ d * 6 = c

-- The main theorem statement
theorem moles_ethane_and_hexachloroethane (moles_Cl2 : ℕ) :
  moles_Cl2 = 18 → balanced_eq 1 1 18 3 :=
by
  sorry

end NUMINAMATH_GPT_moles_ethane_and_hexachloroethane_l1326_132695


namespace NUMINAMATH_GPT_field_trip_total_l1326_132630

-- Define the conditions
def vans := 2
def buses := 3
def people_per_van := 8
def people_per_bus := 20

-- The total number of people
def total_people := (vans * people_per_van) + (buses * people_per_bus)

theorem field_trip_total : total_people = 76 :=
by
  -- skip the proof here
  sorry

end NUMINAMATH_GPT_field_trip_total_l1326_132630


namespace NUMINAMATH_GPT_series_sum_eq_half_l1326_132622

theorem series_sum_eq_half :
  ∑' (n : ℕ), 2^n / (3^(2^n) + 1) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_series_sum_eq_half_l1326_132622


namespace NUMINAMATH_GPT_base6_to_decimal_l1326_132601

theorem base6_to_decimal (m : ℕ) (h : 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 4934) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_base6_to_decimal_l1326_132601


namespace NUMINAMATH_GPT_lattice_points_in_region_l1326_132617

theorem lattice_points_in_region : ∃! n : ℕ, n = 14 ∧ ∀ (x y : ℤ), (y = |x| ∨ y = -x^2 + 4) ∧ (-2 ≤ x ∧ x ≤ 1) → 
  (y = -x^2 + 4 ∧ y = |x|) :=
sorry

end NUMINAMATH_GPT_lattice_points_in_region_l1326_132617


namespace NUMINAMATH_GPT_min_major_axis_length_l1326_132656

theorem min_major_axis_length (a b c : ℝ) (h_area : b * c = 1) (h_focal_relation : 2 * a = 2 * Real.sqrt (b^2 + c^2)) :
  2 * a = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_major_axis_length_l1326_132656


namespace NUMINAMATH_GPT_T_value_l1326_132669

variable (x : ℝ)

def T : ℝ := (x-2)^4 + 4 * (x-2)^3 + 6 * (x-2)^2 + 4 * (x-2) + 1

theorem T_value : T x = (x-1)^4 := by
  sorry

end NUMINAMATH_GPT_T_value_l1326_132669


namespace NUMINAMATH_GPT_sum_of_powers_eq_123_l1326_132657

section

variables {a b : Real}

-- Conditions provided in the problem
axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7

-- Define the theorem to be proved
theorem sum_of_powers_eq_123 : a^10 + b^10 = 123 :=
sorry

end

end NUMINAMATH_GPT_sum_of_powers_eq_123_l1326_132657


namespace NUMINAMATH_GPT_bus_seat_capacity_l1326_132661

theorem bus_seat_capacity (x : ℕ) : 15 * x + (15 - 3) * x + 11 = 92 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_bus_seat_capacity_l1326_132661


namespace NUMINAMATH_GPT_find_value_given_conditions_l1326_132647

def equation_result (x y k : ℕ) : Prop := x ^ y + y ^ x = k

theorem find_value_given_conditions (y : ℕ) (k : ℕ) : 
  equation_result 2407 y k := 
by 
  sorry

end NUMINAMATH_GPT_find_value_given_conditions_l1326_132647


namespace NUMINAMATH_GPT_parallel_vectors_t_eq_neg1_l1326_132606

theorem parallel_vectors_t_eq_neg1 (t : ℝ) :
  let a := (1, -1)
  let b := (t, 1)
  (a.1 + b.1, a.2 + b.2) = (k * (a.1 - b.1), k * (a.2 - b.2)) -> t = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_t_eq_neg1_l1326_132606


namespace NUMINAMATH_GPT_problem_statement_l1326_132623

theorem problem_statement 
  (p q r x y z a b c : ℝ)
  (h1 : p / x = q / y ∧ q / y = r / z)
  (h2 : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1) :
  p^2 / a^2 + q^2 / b^2 + r^2 / c^2 = (p^2 + q^2 + r^2) / (x^2 + y^2 + z^2) :=
sorry  -- Proof omitted

end NUMINAMATH_GPT_problem_statement_l1326_132623


namespace NUMINAMATH_GPT_playground_area_l1326_132654

theorem playground_area :
  ∃ (l w : ℝ), 2 * l + 2 * w = 84 ∧ l = 3 * w ∧ l * w = 330.75 :=
by
  sorry

end NUMINAMATH_GPT_playground_area_l1326_132654


namespace NUMINAMATH_GPT_function_range_is_interval_l1326_132670

theorem function_range_is_interval :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ (256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x) ∧ 
  (256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x) ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_function_range_is_interval_l1326_132670


namespace NUMINAMATH_GPT_fruit_basket_combinations_l1326_132658

namespace FruitBasket

def apples := 3
def oranges := 8
def min_apples := 1
def min_oranges := 1

theorem fruit_basket_combinations : 
  (apples + 1 - min_apples) * (oranges + 1 - min_oranges) = 36 := by
  sorry

end FruitBasket

end NUMINAMATH_GPT_fruit_basket_combinations_l1326_132658


namespace NUMINAMATH_GPT_probability_of_one_of_each_color_l1326_132644

-- Definitions based on the conditions
def total_marbles : ℕ := 12
def marbles_of_each_color : ℕ := 3
def number_of_selected_marbles : ℕ := 4

-- Calculation based on problem requirements
def total_ways_to_choose_marbles : ℕ := Nat.choose total_marbles number_of_selected_marbles
def favorable_ways_to_choose : ℕ := marbles_of_each_color ^ number_of_selected_marbles

-- The main theorem to prove the probability
theorem probability_of_one_of_each_color :
  (favorable_ways_to_choose : ℚ) / total_ways_to_choose = 9 / 55 := by
  sorry

end NUMINAMATH_GPT_probability_of_one_of_each_color_l1326_132644


namespace NUMINAMATH_GPT_fixed_point_of_exponential_function_l1326_132683

theorem fixed_point_of_exponential_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : 
  ∃ p : ℝ × ℝ, p = (-2, 1) ∧ ∀ x : ℝ, (x, a^(x + 2)) = p → x = -2 ∧ a^(x + 2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_exponential_function_l1326_132683


namespace NUMINAMATH_GPT_largest_y_coordinate_of_degenerate_ellipse_l1326_132605

theorem largest_y_coordinate_of_degenerate_ellipse :
  ∀ (x y : ℝ), (x^2 / 36 + (y + 5)^2 / 16 = 0) → y = -5 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_largest_y_coordinate_of_degenerate_ellipse_l1326_132605


namespace NUMINAMATH_GPT_prove_a_star_b_l1326_132621

variable (a b : ℤ)
variable (h1 : a + b = 12)
variable (h2 : a * b = 35)

def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

theorem prove_a_star_b : star a b = 12 / 35 :=
by
  sorry

end NUMINAMATH_GPT_prove_a_star_b_l1326_132621


namespace NUMINAMATH_GPT_Tim_sweets_are_multiple_of_4_l1326_132676

-- Define the conditions
def sweets_are_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Given definitions
def Peter_sweets : ℕ := 44
def largest_possible_number_per_tray : ℕ := 4

-- Define the proposition to be proven
theorem Tim_sweets_are_multiple_of_4 (O : ℕ) (h1 : sweets_are_divisible_by_4 Peter_sweets) (h2 : sweets_are_divisible_by_4 largest_possible_number_per_tray) :
  sweets_are_divisible_by_4 O :=
sorry

end NUMINAMATH_GPT_Tim_sweets_are_multiple_of_4_l1326_132676


namespace NUMINAMATH_GPT_scientific_notation_of_number_l1326_132608

theorem scientific_notation_of_number : 15300000000 = 1.53 * (10 : ℝ)^10 := sorry

end NUMINAMATH_GPT_scientific_notation_of_number_l1326_132608


namespace NUMINAMATH_GPT_total_votes_l1326_132645

theorem total_votes (V : ℝ) 
  (h1 : 0.5 / 100 * V = 0.005 * V) 
  (h2 : 50.5 / 100 * V = 0.505 * V) 
  (h3 : 0.505 * V - 0.005 * V = 3000) : 
  V = 6000 := 
by
  sorry

end NUMINAMATH_GPT_total_votes_l1326_132645


namespace NUMINAMATH_GPT_total_weight_of_remaining_eggs_is_correct_l1326_132686

-- Define the initial conditions and the question as Lean definitions
def total_eggs : Nat := 12
def weight_per_egg : Nat := 10
def num_boxes : Nat := 4
def melted_boxes : Nat := 1

-- Calculate the total weight of the eggs
def total_weight : Nat := total_eggs * weight_per_egg

-- Calculate the number of eggs per box
def eggs_per_box : Nat := total_eggs / num_boxes

-- Calculate the weight per box
def weight_per_box : Nat := eggs_per_box * weight_per_egg

-- Calculate the number of remaining boxes after one is tossed out
def remaining_boxes : Nat := num_boxes - melted_boxes

-- Calculate the total weight of the remaining chocolate eggs
def remaining_weight : Nat := remaining_boxes * weight_per_box

-- The proof task
theorem total_weight_of_remaining_eggs_is_correct : remaining_weight = 90 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_remaining_eggs_is_correct_l1326_132686


namespace NUMINAMATH_GPT_cos_b4_b6_l1326_132616

theorem cos_b4_b6 (a b : ℕ → ℝ) (d : ℝ) 
  (ha_geom : ∀ n, a (n + 1) / a n = a 1)
  (hb_arith : ∀ n, b (n + 1) = b n + d)
  (ha_prod : a 1 * a 5 * a 9 = -8)
  (hb_sum : b 2 + b 5 + b 8 = 6 * Real.pi) : 
  Real.cos ((b 4 + b 6) / (1 - a 3 * a 7)) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_cos_b4_b6_l1326_132616


namespace NUMINAMATH_GPT_smallest_value_of_n_l1326_132672

theorem smallest_value_of_n :
  ∃ o y m n : ℕ, 10 * o = 16 * y ∧ 16 * y = 18 * m ∧ 18 * m = 18 * n ∧ n = 40 := 
sorry

end NUMINAMATH_GPT_smallest_value_of_n_l1326_132672


namespace NUMINAMATH_GPT_pie_shop_total_earnings_l1326_132629

theorem pie_shop_total_earnings :
  let price_per_slice_custard := 3
  let price_per_slice_apple := 4
  let price_per_slice_blueberry := 5
  let slices_per_whole_custard := 10
  let slices_per_whole_apple := 8
  let slices_per_whole_blueberry := 12
  let num_whole_custard_pies := 6
  let num_whole_apple_pies := 4
  let num_whole_blueberry_pies := 5
  let total_earnings :=
    (num_whole_custard_pies * slices_per_whole_custard * price_per_slice_custard) +
    (num_whole_apple_pies * slices_per_whole_apple * price_per_slice_apple) +
    (num_whole_blueberry_pies * slices_per_whole_blueberry * price_per_slice_blueberry)
  total_earnings = 608 := by
  sorry

end NUMINAMATH_GPT_pie_shop_total_earnings_l1326_132629


namespace NUMINAMATH_GPT_luna_budget_l1326_132684

variable {H F P : ℝ}

theorem luna_budget (h1: F = 0.60 * H) (h2: P = 0.10 * F) (h3: H + F + P = 249) :
  H + F = 240 :=
by
  -- The proof will be filled in here. For now, we use sorry.
  sorry

end NUMINAMATH_GPT_luna_budget_l1326_132684


namespace NUMINAMATH_GPT_combined_time_is_45_l1326_132692

-- Definitions based on conditions
def Pulsar_time : ℕ := 10
def Polly_time : ℕ := 3 * Pulsar_time
def Petra_time : ℕ := (1 / 6 ) * Polly_time

-- Total combined time
def total_time : ℕ := Pulsar_time + Polly_time + Petra_time

-- Theorem to prove
theorem combined_time_is_45 : total_time = 45 := by
  sorry

end NUMINAMATH_GPT_combined_time_is_45_l1326_132692


namespace NUMINAMATH_GPT_triangle_inradius_l1326_132674

theorem triangle_inradius (p A : ℝ) (h_p : p = 20) (h_A : A = 30) : 
  ∃ r : ℝ, r = 3 ∧ A = r * p / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inradius_l1326_132674


namespace NUMINAMATH_GPT_tim_movie_marathon_duration_l1326_132687

-- Define the durations of each movie
def first_movie_duration : ℕ := 2

def second_movie_duration : ℕ := 
  first_movie_duration + (first_movie_duration / 2)

def combined_first_two_movies_duration : ℕ :=
  first_movie_duration + second_movie_duration

def last_movie_duration : ℕ := 
  combined_first_two_movies_duration - 1

-- Define the total movie marathon duration
def total_movie_marathon_duration : ℕ := 
  first_movie_duration + second_movie_duration + last_movie_duration

-- Problem statement to be proved
theorem tim_movie_marathon_duration : total_movie_marathon_duration = 9 := by
  sorry

end NUMINAMATH_GPT_tim_movie_marathon_duration_l1326_132687


namespace NUMINAMATH_GPT_frequency_not_equal_probability_l1326_132643

theorem frequency_not_equal_probability
  (N : ℕ) -- Total number of trials
  (N1 : ℕ) -- Number of times student A is selected
  (hN : N > 0) -- Ensure the number of trials is positive
  (rand_int_gen : ℕ → ℕ) -- A function generating random integers from 1 to 6
  (h_gen : ∀ n, 1 ≤ rand_int_gen n ∧ rand_int_gen n ≤ 6) -- Generator produces numbers between 1 to 6
: (N1/N : ℚ) ≠ (1/6 : ℚ) := 
sorry

end NUMINAMATH_GPT_frequency_not_equal_probability_l1326_132643


namespace NUMINAMATH_GPT_expand_expression_l1326_132677

theorem expand_expression (x : ℝ) : 
  (x - 3) * (x + 3) * (x^2 + 5) = x^4 - 4 * x^2 - 45 := 
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1326_132677


namespace NUMINAMATH_GPT_neg_P_l1326_132627

/-
Proposition: There exists a natural number n such that 2^n > 1000.
-/
def P : Prop := ∃ n : ℕ, 2^n > 1000

/-
Theorem: The negation of the above proposition P is:
For all natural numbers n, 2^n ≤ 1000.
-/
theorem neg_P : ¬ P ↔ ∀ n : ℕ, 2^n ≤ 1000 :=
by
  sorry

end NUMINAMATH_GPT_neg_P_l1326_132627


namespace NUMINAMATH_GPT_gcd_poly_l1326_132609

theorem gcd_poly (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) : 
  Int.gcd (4 * b ^ 2 + 63 * b + 144) (2 * b + 7) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_poly_l1326_132609


namespace NUMINAMATH_GPT_passing_grade_fraction_l1326_132600

variables (students : ℕ) -- total number of students in Mrs. Susna's class

-- Conditions
def fraction_A : ℚ := 1/4
def fraction_B : ℚ := 1/2
def fraction_C : ℚ := 1/8
def fraction_D : ℚ := 1/12
def fraction_F : ℚ := 1/24

-- Prove the fraction of students getting a passing grade (C or higher) is 7/8
theorem passing_grade_fraction : 
  fraction_A + fraction_B + fraction_C = 7/8 :=
by
  sorry

end NUMINAMATH_GPT_passing_grade_fraction_l1326_132600


namespace NUMINAMATH_GPT_pears_sold_l1326_132693

theorem pears_sold (m a total : ℕ) (h1 : a = 2 * m) (h2 : m = 120) (h3 : a = 240) : total = 360 :=
by
  sorry

end NUMINAMATH_GPT_pears_sold_l1326_132693


namespace NUMINAMATH_GPT_find_x_in_list_l1326_132678

theorem find_x_in_list :
  ∃ x : ℕ, x > 0 ∧ x ≤ 120 ∧ (45 + 76 + 110 + x + x) / 5 = 2 * x ∧ x = 29 :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_list_l1326_132678


namespace NUMINAMATH_GPT_pyramid_surface_area_l1326_132624

-- Definitions for the conditions
structure Rectangle where
  length : ℝ
  width : ℝ

structure Pyramid where
  base : Rectangle
  height : ℝ

-- Create instances representing the given conditions
noncomputable def givenRectangle : Rectangle := {
  length := 8,
  width := 6
}

noncomputable def givenPyramid : Pyramid := {
  base := givenRectangle,
  height := 15
}

-- Statement to prove the surface area of the pyramid
theorem pyramid_surface_area
  (rect: Rectangle)
  (length := rect.length)
  (width := rect.width)
  (height: ℝ)
  (hy1: length = 8)
  (hy2: width = 6)
  (hy3: height = 15) :
  let base_area := length * width
  let slant_height := Real.sqrt (height^2 + (length / 2)^2)
  let lateral_area := 2 * ((length * slant_height) / 2 + (width * slant_height) / 2)
  let total_surface_area := base_area + lateral_area 
  total_surface_area = 48 + 7 * Real.sqrt 241 := 
  sorry

end NUMINAMATH_GPT_pyramid_surface_area_l1326_132624
