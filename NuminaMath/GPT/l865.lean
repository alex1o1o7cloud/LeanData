import Mathlib

namespace NUMINAMATH_GPT_quadratic_inequalities_solution_l865_86554

noncomputable def a : Type := sorry
noncomputable def b : Type := sorry
noncomputable def c : Type := sorry

theorem quadratic_inequalities_solution (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + bx + c > 0 ↔ -1/3 < x ∧ x < 2) :
  ∀ y, cx^2 + bx + a < 0 ↔ -3 < y ∧ y < 1/2 :=
sorry

end NUMINAMATH_GPT_quadratic_inequalities_solution_l865_86554


namespace NUMINAMATH_GPT_modulo_multiplication_l865_86590

theorem modulo_multiplication (m : ℕ) (h : 0 ≤ m ∧ m < 50) :
  152 * 936 % 50 = 22 :=
by
  sorry

end NUMINAMATH_GPT_modulo_multiplication_l865_86590


namespace NUMINAMATH_GPT_intersect_at_point_m_eq_1_3_n_eq_neg_73_9_lines_parallel_pass_through_lines_perpendicular_y_intercept_l865_86563

theorem intersect_at_point_m_eq_1_3_n_eq_neg_73_9 
  (m : ℚ) (n : ℚ) : 
  (m^2 + 8 + n = 0) ∧ (3*m - 1 = 0) → 
  (m = 1/3 ∧ n = -73/9) := 
by 
  sorry

theorem lines_parallel_pass_through 
  (m : ℚ) (n : ℚ) :
  (m ≠ 0) → (m^2 = 16) ∧ (3*m - 8 + n = 0) → 
  (m = 4 ∧ n = -4) ∨ (m = -4 ∧ n = 20) :=
by 
  sorry

theorem lines_perpendicular_y_intercept 
  (m : ℚ) (n : ℚ) :
  (m = 0 ∧ 8*(-1) + n = 0) → 
  (m = 0 ∧ n = 8) :=
by 
  sorry

end NUMINAMATH_GPT_intersect_at_point_m_eq_1_3_n_eq_neg_73_9_lines_parallel_pass_through_lines_perpendicular_y_intercept_l865_86563


namespace NUMINAMATH_GPT_proof_problem_l865_86561

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * Real.pi * x)

theorem proof_problem
  (a : ℝ)
  (h1 : ∀ x : ℝ, f (x - 1/2) = f (x + 1/2))
  (h2 : f (-1/4) = a) :
  f (9/4) = -a :=
by sorry

end NUMINAMATH_GPT_proof_problem_l865_86561


namespace NUMINAMATH_GPT_num_combinations_l865_86565

-- The conditions given in the problem.
def num_pencil_types : ℕ := 2
def num_eraser_types : ℕ := 3

-- The theorem to prove.
theorem num_combinations (pencils : ℕ) (erasers : ℕ) (h1 : pencils = num_pencil_types) (h2 : erasers = num_eraser_types) : pencils * erasers = 6 :=
by 
  have hp : pencils = 2 := h1
  have he : erasers = 3 := h2
  cases hp
  cases he
  rfl

end NUMINAMATH_GPT_num_combinations_l865_86565


namespace NUMINAMATH_GPT_eval_expression_l865_86566

theorem eval_expression (x y : ℕ) (h_x : x = 2001) (h_y : y = 2002) :
  (x^3 - 3*x^2*y + 5*x*y^2 - y^3 - 2) / (x * y) = 1999 :=
  sorry

end NUMINAMATH_GPT_eval_expression_l865_86566


namespace NUMINAMATH_GPT_ratio_fifth_term_l865_86553

-- Definitions of arithmetic sequences and sums
def arithmetic_seq_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ := n * (2 * a 1 + (n - 1) * d 1) / 2

-- Conditions
variables (S_n S'_n : ℕ → ℕ) (n : ℕ)

-- Given conditions
axiom ratio_sum : ∀ (n : ℕ), S_n n / S'_n n = (5 * n + 3) / (2 * n + 7)
axiom sums_at_9 : S_n 9 = 9 * (S_n 1 + S_n 9) / 2
axiom sums'_at_9 : S'_n 9 = 9 * (S'_n 1 + S'_n 9) / 2

-- Theorem to prove
theorem ratio_fifth_term : (9 * (S_n 1 + S_n 9) / 2) / (9 * (S'_n 1 + S'_n 9) / 2) = 48 / 25 := sorry

end NUMINAMATH_GPT_ratio_fifth_term_l865_86553


namespace NUMINAMATH_GPT_initial_investment_proof_l865_86517

noncomputable def initial_investment (A : ℝ) (r t : ℕ) : ℝ := 
  A / (1 + r / 100) ^ t

theorem initial_investment_proof : 
  initial_investment 1000 8 8 = 630.17 := sorry

end NUMINAMATH_GPT_initial_investment_proof_l865_86517


namespace NUMINAMATH_GPT_average_age_at_marriage_l865_86564

theorem average_age_at_marriage
  (A : ℕ)
  (combined_age_at_marriage : husband_age + wife_age = 2 * A)
  (combined_age_after_5_years : (A + 5) + (A + 5) + 1 = 57) :
  A = 23 := 
sorry

end NUMINAMATH_GPT_average_age_at_marriage_l865_86564


namespace NUMINAMATH_GPT_cakes_initially_made_l865_86579

variables (sold bought total initial_cakes : ℕ)

theorem cakes_initially_made (h1 : sold = 105) (h2 : bought = 170) (h3 : total = 186) :
  initial_cakes = total - (sold - bought) :=
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_cakes_initially_made_l865_86579


namespace NUMINAMATH_GPT_darrel_will_receive_l865_86599

noncomputable def darrel_coins_value : ℝ := 
  let quarters := 127 
  let dimes := 183 
  let nickels := 47 
  let pennies := 237 
  let half_dollars := 64 
  let euros := 32 
  let pounds := 55 
  let quarter_fee_rate := 0.12 
  let dime_fee_rate := 0.07 
  let nickel_fee_rate := 0.15 
  let penny_fee_rate := 0.10 
  let half_dollar_fee_rate := 0.05 
  let euro_exchange_rate := 1.18 
  let euro_fee_rate := 0.03 
  let pound_exchange_rate := 1.39 
  let pound_fee_rate := 0.04 
  let quarters_value := 127 * 0.25 
  let quarters_fee := quarters_value * 0.12 
  let quarters_after_fee := quarters_value - quarters_fee 
  let dimes_value := 183 * 0.10 
  let dimes_fee := dimes_value * 0.07 
  let dimes_after_fee := dimes_value - dimes_fee 
  let nickels_value := 47 * 0.05 
  let nickels_fee := nickels_value * 0.15 
  let nickels_after_fee := nickels_value - nickels_fee 
  let pennies_value := 237 * 0.01 
  let pennies_fee := pennies_value * 0.10 
  let pennies_after_fee := pennies_value - pennies_fee 
  let half_dollars_value := 64 * 0.50 
  let half_dollars_fee := half_dollars_value * 0.05 
  let half_dollars_after_fee := half_dollars_value - half_dollars_fee 
  let euros_value := 32 * 1.18 
  let euros_fee := euros_value * 0.03 
  let euros_after_fee := euros_value - euros_fee 
  let pounds_value := 55 * 1.39 
  let pounds_fee := pounds_value * 0.04 
  let pounds_after_fee := pounds_value - pounds_fee 
  quarters_after_fee + dimes_after_fee + nickels_after_fee + pennies_after_fee + half_dollars_after_fee + euros_after_fee + pounds_after_fee

theorem darrel_will_receive : darrel_coins_value = 189.51 := by
  unfold darrel_coins_value
  sorry

end NUMINAMATH_GPT_darrel_will_receive_l865_86599


namespace NUMINAMATH_GPT_range_of_m_l865_86500

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_m (h1 : ∀ x : ℝ, f (-x) = f x)
                   (h2 : ∀ a b : ℝ, a ≠ b → a ≤ 0 → b ≤ 0 → (f a - f b) / (a - b) < 0)
                   (h3 : f (m + 1) < f 2) : 
  ∃ m : ℝ, -3 < m ∧ m < 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l865_86500


namespace NUMINAMATH_GPT_f_x_neg_l865_86526

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1 else -x^2 - 1

theorem f_x_neg (x : ℝ) (h : x < 0) : f x = -x^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_f_x_neg_l865_86526


namespace NUMINAMATH_GPT_solve_quadratic_for_negative_integer_l865_86515

theorem solve_quadratic_for_negative_integer (N : ℤ) (h_neg : N < 0) (h_eq : 2 * N^2 + N = 20) : N = -4 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_for_negative_integer_l865_86515


namespace NUMINAMATH_GPT_solution_set_of_fraction_inequality_l865_86595

theorem solution_set_of_fraction_inequality
  (a b : ℝ) (h₀ : ∀ x : ℝ, x > 1 → ax - b > 0) :
  {x : ℝ | (ax + b) / (x - 2) > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_fraction_inequality_l865_86595


namespace NUMINAMATH_GPT_determine_ordered_triple_l865_86506

open Real

theorem determine_ordered_triple (a b c : ℝ) (h₁ : 5 < a) (h₂ : 5 < b) (h₃ : 5 < c) 
  (h₄ : (a + 3)^2 / (b + c - 3) + (b + 6)^2 / (c + a - 6) + (c + 9)^2 / (a + b - 9) = 81) : 
  a = 15 ∧ b = 12 ∧ c = 9 := 
sorry

end NUMINAMATH_GPT_determine_ordered_triple_l865_86506


namespace NUMINAMATH_GPT_simplify_expression_l865_86594

theorem simplify_expression (y : ℝ) : 2 - (2 - (2 - (2 - (2 - y)))) = 4 - y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l865_86594


namespace NUMINAMATH_GPT_problem_statement_l865_86569

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem problem_statement :
  (M ∩ N) = N :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l865_86569


namespace NUMINAMATH_GPT_find_blue_weights_l865_86528

theorem find_blue_weights (B : ℕ) :
  (2 * B + 15 + 2 = 25) → B = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_blue_weights_l865_86528


namespace NUMINAMATH_GPT_g_product_of_roots_l865_86513

def f (x : ℂ) : ℂ := x^6 + x^3 + 1
def g (x : ℂ) : ℂ := x^2 + 1

theorem g_product_of_roots (x_1 x_2 x_3 x_4 x_5 x_6 : ℂ) 
    (h1 : ∀ x, (x - x_1) * (x - x_2) * (x - x_3) * (x - x_4) * (x - x_5) * (x - x_6) = f x) :
    g x_1 * g x_2 * g x_3 * g x_4 * g x_5 * g x_6 = 1 :=
by 
    sorry

end NUMINAMATH_GPT_g_product_of_roots_l865_86513


namespace NUMINAMATH_GPT_min_sum_weights_l865_86574

theorem min_sum_weights (S : ℕ) (h1 : S > 280) (h2 : S % 70 = 30) : S = 310 :=
sorry

end NUMINAMATH_GPT_min_sum_weights_l865_86574


namespace NUMINAMATH_GPT_parabola_equation_l865_86537

-- Define the conditions of the problem
def parabola_vertex := (0, 0)
def parabola_focus_x_axis := true
def line_eq (x y : ℝ) : Prop := x = y
def midpoint_of_AB (x1 y1 x2 y2 mx my: ℝ) : Prop := (mx, my) = ((x1 + x2) / 2, (y1 + y2) / 2)
def point_P := (1, 1)

theorem parabola_equation (A B : ℝ × ℝ) :
  (parabola_vertex = (0, 0)) →
  (parabola_focus_x_axis) →
  (line_eq A.1 A.2) →
  (line_eq B.1 B.2) →
  midpoint_of_AB A.1 A.2 B.1 B.2 point_P.1 point_P.2 →
  A = (0, 0) ∨ B = (0, 0) →
  B = A ∨ A = (0, 0) → B = (2, 2) →
  ∃ a, ∀ x y, y^2 = a * x → a = 2 :=
sorry

end NUMINAMATH_GPT_parabola_equation_l865_86537


namespace NUMINAMATH_GPT_unique_zero_point_condition1_unique_zero_point_condition2_l865_86531

noncomputable def func (x a b : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

theorem unique_zero_point_condition1 {a b : ℝ} (h1 : 1 / 2 < a) (h2 : a ≤ Real.exp 2 / 2) (h3 : b > 2 * a) :
  ∃! x, func x a b = 0 :=
sorry

theorem unique_zero_point_condition2 {a b : ℝ} (h1 : 0 < a) (h2 : a < 1 / 2) (h3 : b ≤ 2 * a) :
  ∃! x, func x a b = 0 :=
sorry

end NUMINAMATH_GPT_unique_zero_point_condition1_unique_zero_point_condition2_l865_86531


namespace NUMINAMATH_GPT_find_ab_l865_86509

theorem find_ab (A B : Set ℝ) (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = Set.univ) → 
  (A ∩ B = {x | 3 < x ∧ x ≤ 4}) →
  a + b = -7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_ab_l865_86509


namespace NUMINAMATH_GPT_count_valid_numbers_is_31_l865_86539

def is_valid_digit (n : Nat) : Prop := n = 0 ∨ n = 2 ∨ n = 6 ∨ n = 8

def count_valid_numbers : Nat :=
  let valid_digits := [0, 2, 6, 8]
  let one_digit := valid_digits.filter (λ n => n % 4 = 0)
  let two_digits := valid_digits.product valid_digits |>.filter (λ (a, b) => (10*a + b) % 4 = 0)
  let three_digits := valid_digits.product two_digits |>.filter (λ (a, (b, c)) => (100*a + 10*b + c) % 4 = 0)
  one_digit.length + two_digits.length + three_digits.length

theorem count_valid_numbers_is_31 : count_valid_numbers = 31 := by
  sorry

end NUMINAMATH_GPT_count_valid_numbers_is_31_l865_86539


namespace NUMINAMATH_GPT_percentage_women_red_and_men_dark_l865_86593

-- Define the conditions as variables
variables (w_fair_hair w_dark_hair w_red_hair m_fair_hair m_dark_hair m_red_hair : ℝ)

-- Define the percentage of women with red hair and men with dark hair
def women_red_men_dark (w_red_hair m_dark_hair : ℝ) : ℝ := w_red_hair + m_dark_hair

-- Define the main theorem to be proven
theorem percentage_women_red_and_men_dark 
  (hw_fair_hair : w_fair_hair = 30)
  (hw_dark_hair : w_dark_hair = 28)
  (hw_red_hair : w_red_hair = 12)
  (hm_fair_hair : m_fair_hair = 20)
  (hm_dark_hair : m_dark_hair = 35)
  (hm_red_hair : m_red_hair = 5) :
  women_red_men_dark w_red_hair m_dark_hair = 47 := 
sorry

end NUMINAMATH_GPT_percentage_women_red_and_men_dark_l865_86593


namespace NUMINAMATH_GPT_number_of_people_l865_86548

theorem number_of_people (total_cookies : ℕ) (cookies_per_person : ℝ) (h1 : total_cookies = 144) (h2 : cookies_per_person = 24.0) : total_cookies / cookies_per_person = 6 := 
by 
  -- Placeholder for actual proof.
  sorry

end NUMINAMATH_GPT_number_of_people_l865_86548


namespace NUMINAMATH_GPT_max_even_a_exists_max_even_a_l865_86546

theorem max_even_a (a : ℤ): (a^2 - 12 * a + 32 ≤ 0 ∧ ∃ k : ℤ, a = 2 * k) → a ≤ 8 := sorry

theorem exists_max_even_a : ∃ a : ℤ, (a^2 - 12 * a + 32 ≤ 0 ∧ ∃ k : ℤ, a = 2 * k ∧ a = 8) := sorry

end NUMINAMATH_GPT_max_even_a_exists_max_even_a_l865_86546


namespace NUMINAMATH_GPT_circle_equation_l865_86558

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * y = 0

theorem circle_equation
  (x y : ℝ)
  (center_on_y_axis : ∃ r : ℝ, r > 0 ∧ x^2 + (y - r)^2 = r^2)
  (tangent_to_x_axis : ∃ r : ℝ, r > 0 ∧ y = r)
  (passes_through_point : x = 3 ∧ y = 1) :
  equation_of_circle x y :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l865_86558


namespace NUMINAMATH_GPT_categorize_numbers_l865_86589

noncomputable def positive_numbers (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x > 0}

noncomputable def non_neg_integers (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x ≥ 0 ∧ ∃ n : ℤ, x = n}

noncomputable def negative_fractions (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x < 0 ∧ ∃ n d : ℤ, d ≠ 0 ∧ (x = n / d)}

def given_set : Set ℝ := {6, -3, 2.4, -3/4, 0, -3.14, 2, -7/2, 2/3}

theorem categorize_numbers :
  positive_numbers given_set = {6, 2.4, 2, 2/3} ∧
  non_neg_integers given_set = {6, 0, 2} ∧
  negative_fractions given_set = {-3/4, -3.14, -7/2} :=
by
  sorry

end NUMINAMATH_GPT_categorize_numbers_l865_86589


namespace NUMINAMATH_GPT_find_metal_molecular_weight_l865_86507

noncomputable def molecular_weight_of_metal (compound_mw: ℝ) (oh_mw: ℝ) : ℝ :=
  compound_mw - oh_mw

theorem find_metal_molecular_weight :
  let compound_mw := 171.00
  let oxygen_mw := 16.00
  let hydrogen_mw := 1.01
  let oh_ions := 2
  let oh_mw := oh_ions * (oxygen_mw + hydrogen_mw)
  molecular_weight_of_metal compound_mw oh_mw = 136.98 :=
by
  sorry

end NUMINAMATH_GPT_find_metal_molecular_weight_l865_86507


namespace NUMINAMATH_GPT_smallest_x_for_gx_eq_1024_l865_86505

noncomputable def g : ℝ → ℝ
  | x => if 2 ≤ x ∧ x ≤ 6 then 2 - |x - 3| else 0

axiom g_property1 : ∀ x : ℝ, 0 < x → g (4 * x) = 4 * g x
axiom g_property2 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 6 → g x = 2 - |x - 3|
axiom g_2004 : g 2004 = 1024

theorem smallest_x_for_gx_eq_1024 : ∃ x : ℝ, g x = 1024 ∧ ∀ y : ℝ, g y = 1024 → x ≤ y := sorry

end NUMINAMATH_GPT_smallest_x_for_gx_eq_1024_l865_86505


namespace NUMINAMATH_GPT_brad_start_time_after_maxwell_l865_86573

-- Assuming time is measured in hours, distance in kilometers, and speed in km/h
def meet_time (d : ℕ) (v_m : ℕ) (v_b : ℕ) (t_m : ℕ) : ℕ :=
  let d_m := t_m * v_m
  let t_b := t_m - 1
  let d_b := t_b * v_b
  d_m + d_b

theorem brad_start_time_after_maxwell (d : ℕ) (v_m : ℕ) (v_b : ℕ) (t_m : ℕ) :
  d = 54 → v_m = 4 → v_b = 6 → t_m = 6 → 
  meet_time d v_m v_b t_m = 54 :=
by
  intros hd hv_m hv_b ht_m
  have : meet_time d v_m v_b t_m = t_m * v_m + (t_m - 1) * v_b := rfl
  rw [hd, hv_m, hv_b, ht_m] at this
  sorry

end NUMINAMATH_GPT_brad_start_time_after_maxwell_l865_86573


namespace NUMINAMATH_GPT_most_appropriate_method_to_solve_4x2_minus_9_eq_0_l865_86510

theorem most_appropriate_method_to_solve_4x2_minus_9_eq_0 :
  (∀ x : ℤ, 4 * x^2 - 9 = 0 ↔ x = 3 / 2 ∨ x = -3 / 2) → true :=
by
  sorry

end NUMINAMATH_GPT_most_appropriate_method_to_solve_4x2_minus_9_eq_0_l865_86510


namespace NUMINAMATH_GPT_no_solution_for_ab_ba_l865_86547

theorem no_solution_for_ab_ba (a b x : ℕ)
  (ab ba : ℕ)
  (h_ab : ab = 10 * a + b)
  (h_ba : ba = 10 * b + a) :
  (ab^x - 2 = ba^x - 7) → false :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_ab_ba_l865_86547


namespace NUMINAMATH_GPT_magnitude_range_l865_86570

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def vector_b : ℝ × ℝ := (Real.sqrt 3, -1)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem magnitude_range (θ : ℝ) : 
  0 ≤ (vector_magnitude (2 • vector_a θ - vector_b)) ∧ (vector_magnitude (2 • vector_a θ - vector_b)) ≤ 4 := 
sorry

end NUMINAMATH_GPT_magnitude_range_l865_86570


namespace NUMINAMATH_GPT_max_k_inequality_l865_86516

theorem max_k_inequality (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  ∀ k ≤ 2, ( ( (b - c) ^ 2 * (b + c) / a ) + 
             ( (c - a) ^ 2 * (c + a) / b ) + 
             ( (a - b) ^ 2 * (a + b) / c ) 
             ≥ k * ( a^2 + b^2 + c^2 - a*b - b*c - c*a ) ) :=
by
  sorry

end NUMINAMATH_GPT_max_k_inequality_l865_86516


namespace NUMINAMATH_GPT_range_of_m_l865_86560

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (m * x^2 + (m - 3) * x + 1 = 0)) →
  m ∈ Set.Iic 1 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l865_86560


namespace NUMINAMATH_GPT_ab_cd_is_1_or_minus_1_l865_86552

theorem ab_cd_is_1_or_minus_1 (a b c d : ℤ) (h1 : ∃ k₁ : ℤ, a = k₁ * (a * b - c * d))
  (h2 : ∃ k₂ : ℤ, b = k₂ * (a * b - c * d)) (h3 : ∃ k₃ : ℤ, c = k₃ * (a * b - c * d))
  (h4 : ∃ k₄ : ℤ, d = k₄ * (a * b - c * d)) :
  a * b - c * d = 1 ∨ a * b - c * d = -1 := 
sorry

end NUMINAMATH_GPT_ab_cd_is_1_or_minus_1_l865_86552


namespace NUMINAMATH_GPT_n_greater_than_7_l865_86586

theorem n_greater_than_7 (m n : ℕ) (hmn : m > n) (h : ∃k:ℕ, 22220038^m - 22220038^n = 10^8 * k) : n > 7 :=
sorry

end NUMINAMATH_GPT_n_greater_than_7_l865_86586


namespace NUMINAMATH_GPT_largest_possible_expression_value_l865_86532

-- Definition of the conditions.
def distinct_digits (X Y Z : ℕ) : Prop := X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X < 10 ∧ Y < 10 ∧ Z < 10

-- The main theorem statement.
theorem largest_possible_expression_value : ∀ (X Y Z : ℕ), distinct_digits X Y Z → 
  (100 * X + 10 * Y + Z - 10 * Z - Y - X) ≤ 900 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_expression_value_l865_86532


namespace NUMINAMATH_GPT_blue_balls_balance_l865_86519

variables {R B O P : ℝ}

-- Given conditions
def cond1 : 4 * R = 8 * B := sorry
def cond2 : 3 * O = 7 * B := sorry
def cond3 : 8 * B = 6 * P := sorry

-- Proof problem: proving equal balance of 5 red balls, 3 orange balls, and 4 purple balls
theorem blue_balls_balance : 5 * R + 3 * O + 4 * P = (67 / 3) * B :=
by
  sorry

end NUMINAMATH_GPT_blue_balls_balance_l865_86519


namespace NUMINAMATH_GPT_probability_more_sons_or_daughters_correct_l865_86523

noncomputable def probability_more_sons_or_daughters : ℚ :=
  let total_combinations := (2 : ℕ) ^ 8
  let equal_sons_daughters := Nat.choose 8 4
  let more_sons_or_daughters := total_combinations - equal_sons_daughters
  more_sons_or_daughters / total_combinations

theorem probability_more_sons_or_daughters_correct :
  probability_more_sons_or_daughters = 93 / 128 := by
  sorry 

end NUMINAMATH_GPT_probability_more_sons_or_daughters_correct_l865_86523


namespace NUMINAMATH_GPT_lilith_caps_collection_l865_86512

noncomputable def monthlyCollectionYear1 := 3
noncomputable def monthlyCollectionAfterYear1 := 5
noncomputable def christmasCaps := 40
noncomputable def yearlyCapsLost := 15
noncomputable def totalYears := 5

noncomputable def totalCapsCollectedByLilith :=
  let firstYearCaps := monthlyCollectionYear1 * 12
  let remainingYearsCaps := monthlyCollectionAfterYear1 * 12 * (totalYears - 1)
  let christmasCapsTotal := christmasCaps * totalYears
  let totalCapsBeforeLosses := firstYearCaps + remainingYearsCaps + christmasCapsTotal
  let lostCapsTotal := yearlyCapsLost * totalYears
  let totalCapsAfterLosses := totalCapsBeforeLosses - lostCapsTotal
  totalCapsAfterLosses

theorem lilith_caps_collection : totalCapsCollectedByLilith = 401 := by
  sorry

end NUMINAMATH_GPT_lilith_caps_collection_l865_86512


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l865_86585

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 16 - y^2 / 9 = 1) → (y = 3/4 * x ∨ y = -3/4 * x) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l865_86585


namespace NUMINAMATH_GPT_Johnson_Martinez_tied_at_end_of_september_l865_86582

open Nat

-- Define the monthly home runs for Johnson and Martinez
def Johnson_runs : List Nat := [3, 8, 15, 12, 5, 7, 14]
def Martinez_runs : List Nat := [0, 3, 9, 20, 7, 12, 13]

-- Define the cumulated home runs for Johnson and Martinez over the months
def total_runs (runs : List Nat) : List Nat :=
  runs.scanl (· + ·) 0

-- State the theorem to prove that they are tied in total runs at the end of September
theorem Johnson_Martinez_tied_at_end_of_september :
  (total_runs Johnson_runs).getLast (by decide) =
  (total_runs Martinez_runs).getLast (by decide) := by
  sorry

end NUMINAMATH_GPT_Johnson_Martinez_tied_at_end_of_september_l865_86582


namespace NUMINAMATH_GPT_length_of_one_pencil_l865_86588

theorem length_of_one_pencil (l : ℕ) (h1 : 2 * l = 24) : l = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_of_one_pencil_l865_86588


namespace NUMINAMATH_GPT_problem_a_lt_b_lt_0_implies_ab_gt_b_sq_l865_86549

theorem problem_a_lt_b_lt_0_implies_ab_gt_b_sq (a b : ℝ) (h : a < b ∧ b < 0) : ab > b^2 := by
  sorry

end NUMINAMATH_GPT_problem_a_lt_b_lt_0_implies_ab_gt_b_sq_l865_86549


namespace NUMINAMATH_GPT_shaded_area_l865_86518

-- Definition for the conditions provided in the problem
def side_length := 6
def area_square := side_length ^ 2
def area_square_unit := area_square * 4

-- The problem and proof statement
theorem shaded_area (sl : ℕ) (asq : ℕ) (nsq : ℕ):
    sl = 6 ∧
    asq = sl ^ 2 ∧
    nsq = asq * 4 →
    nsq - (4 * (sl^2 / 2)) = 72 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l865_86518


namespace NUMINAMATH_GPT_instantaneous_velocity_at_1_l865_86522

noncomputable def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

theorem instantaneous_velocity_at_1 :
  (deriv h 1) = -3.3 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_1_l865_86522


namespace NUMINAMATH_GPT_lines_intersect_value_k_l865_86521

theorem lines_intersect_value_k :
  ∀ (x y k : ℝ), (-3 * x + y = k) → (2 * x + y = 20) → (x = -10) → (k = 70) :=
by
  intros x y k h1 h2 h3
  sorry

end NUMINAMATH_GPT_lines_intersect_value_k_l865_86521


namespace NUMINAMATH_GPT_log_eq_solution_l865_86587

open Real

theorem log_eq_solution (x : ℝ) (h : x > 0) : log x + log (x + 1) = 2 ↔ x = (-1 + sqrt 401) / 2 :=
by
  sorry

end NUMINAMATH_GPT_log_eq_solution_l865_86587


namespace NUMINAMATH_GPT_perpendicular_lines_k_value_l865_86557

theorem perpendicular_lines_k_value (k : ℝ) : 
  k * (k - 1) + (1 - k) * (2 * k + 3) = 0 ↔ k = -3 ∨ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_k_value_l865_86557


namespace NUMINAMATH_GPT_total_weight_of_fish_is_correct_l865_86543

noncomputable def totalWeightInFirstTank := 15 * 0.08 + 12 * 0.05

noncomputable def totalWeightInSecondTank := 2 * 15 * 0.08 + 3 * 12 * 0.05

noncomputable def totalWeightInThirdTank := 3 * 15 * 0.08 + 2 * 12 * 0.05 + 5 * 0.14

noncomputable def totalWeightAllTanks := totalWeightInFirstTank + totalWeightInSecondTank + totalWeightInThirdTank

theorem total_weight_of_fish_is_correct : 
  totalWeightAllTanks = 11.5 :=
by         
  sorry

end NUMINAMATH_GPT_total_weight_of_fish_is_correct_l865_86543


namespace NUMINAMATH_GPT_improper_fraction_decomposition_l865_86578

theorem improper_fraction_decomposition (x : ℝ) :
  (6 * x^3 + 5 * x^2 + 3 * x - 4) / (x^2 + 4) = 6 * x + 5 - (21 * x + 24) / (x^2 + 4) := 
sorry

end NUMINAMATH_GPT_improper_fraction_decomposition_l865_86578


namespace NUMINAMATH_GPT_finding_breadth_and_length_of_floor_l865_86514

noncomputable def length_of_floor (b : ℝ) := 3 * b
noncomputable def area_of_floor (b : ℝ) := (length_of_floor b) * b

theorem finding_breadth_and_length_of_floor
  (breadth : ℝ)
  (length : ℝ := length_of_floor breadth)
  (area : ℝ := area_of_floor breadth)
  (painting_cost : ℝ)
  (cost_per_sqm : ℝ)
  (h1 : painting_cost = 100)
  (h2 : cost_per_sqm = 2)
  (h3 : area = painting_cost / cost_per_sqm) :
  length = Real.sqrt 150 :=
by
  sorry

end NUMINAMATH_GPT_finding_breadth_and_length_of_floor_l865_86514


namespace NUMINAMATH_GPT_cost_of_bench_eq_150_l865_86572

theorem cost_of_bench_eq_150 (B : ℕ) (h : B + 2 * B = 450) : B = 150 :=
sorry

end NUMINAMATH_GPT_cost_of_bench_eq_150_l865_86572


namespace NUMINAMATH_GPT_sum_of_yellow_and_blue_is_red_l865_86597

theorem sum_of_yellow_and_blue_is_red (a b : ℕ) : ∃ k : ℕ, (4 * a + 3) + (4 * b + 2) = 4 * k + 1 :=
by sorry

end NUMINAMATH_GPT_sum_of_yellow_and_blue_is_red_l865_86597


namespace NUMINAMATH_GPT_natural_numbers_condition_l865_86533

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem natural_numbers_condition (n : ℕ) (p1 p2 : ℕ)
  (hp1_prime : is_prime p1) (hp2_prime : is_prime p2)
  (hn : n = p1 ^ 2) (hn72 : n + 72 = p2 ^ 2) :
  n = 49 ∨ n = 289 :=
  sorry

end NUMINAMATH_GPT_natural_numbers_condition_l865_86533


namespace NUMINAMATH_GPT_supply_lasts_for_8_months_l865_86591

-- Define the conditions
def pills_per_supply : ℕ := 120
def days_per_pill : ℕ := 2
def days_per_month : ℕ := 30

-- Define the function to calculate the duration in days
def supply_duration_in_days (pills : ℕ) (days_per_pill : ℕ) : ℕ :=
  pills * days_per_pill

-- Define the function to convert days to months
def days_to_months (days : ℕ) (days_per_month : ℕ) : ℕ :=
  days / days_per_month

-- Main statement to prove
theorem supply_lasts_for_8_months :
  days_to_months (supply_duration_in_days pills_per_supply days_per_pill) days_per_month = 8 :=
by
  sorry

end NUMINAMATH_GPT_supply_lasts_for_8_months_l865_86591


namespace NUMINAMATH_GPT_farmer_field_area_l865_86556

variable (x : ℕ) (A : ℕ)

def planned_days : Type := {x : ℕ // 120 * x = 85 * (x + 2) + 40}

theorem farmer_field_area (h : {x : ℕ // 120 * x = 85 * (x + 2) + 40}) : A = 720 :=
by
  sorry

end NUMINAMATH_GPT_farmer_field_area_l865_86556


namespace NUMINAMATH_GPT_value_of_a_l865_86541

theorem value_of_a (a x y : ℤ) (h1 : x = 1) (h2 : y = -3) (h3 : a * x - y = 1) : a = -2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_value_of_a_l865_86541


namespace NUMINAMATH_GPT_algebraic_expression_value_l865_86592

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a - 2 * b + 2 = 0) :
  2024 + 2 * a - b = 2023 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l865_86592


namespace NUMINAMATH_GPT_card_length_l865_86508

noncomputable def width_card : ℕ := 2
noncomputable def side_poster_board : ℕ := 12
noncomputable def total_cards : ℕ := 24

theorem card_length :
  ∃ (card_length : ℕ),
    (side_poster_board / width_card) * (side_poster_board / card_length) = total_cards ∧ 
    card_length = 3 := by
  sorry

end NUMINAMATH_GPT_card_length_l865_86508


namespace NUMINAMATH_GPT_correct_statement_is_D_l865_86501

axiom three_points_determine_plane : Prop
axiom line_and_point_determine_plane : Prop
axiom quadrilateral_is_planar_figure : Prop
axiom two_intersecting_lines_determine_plane : Prop

theorem correct_statement_is_D : two_intersecting_lines_determine_plane = True := 
by sorry

end NUMINAMATH_GPT_correct_statement_is_D_l865_86501


namespace NUMINAMATH_GPT_average_side_length_of_squares_l865_86577

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_side_length_of_squares_l865_86577


namespace NUMINAMATH_GPT_product_of_numbers_l865_86583

theorem product_of_numbers (x y : ℕ) (h1 : x + y = 15) (h2 : x - y = 11) : x * y = 26 :=
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l865_86583


namespace NUMINAMATH_GPT_square_table_seats_4_pupils_l865_86567

-- Define the conditions given in the problem
def num_rectangular_tables := 7
def seats_per_rectangular_table := 10
def total_pupils := 90
def num_square_tables := 5

-- Define what we want to prove
theorem square_table_seats_4_pupils (x : ℕ) :
  total_pupils = num_rectangular_tables * seats_per_rectangular_table + num_square_tables * x →
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_square_table_seats_4_pupils_l865_86567


namespace NUMINAMATH_GPT_average_words_per_puzzle_l865_86562

-- Define the conditions
def uses_up_pencil_every_two_weeks : Prop := ∀ (days_used : ℕ), days_used = 14
def words_to_use_up_pencil : ℕ := 1050
def puzzles_completed_per_day : ℕ := 1

-- Problem statement: Prove the average number of words in each crossword puzzle
theorem average_words_per_puzzle :
  (words_to_use_up_pencil / 14 = 75) :=
by
  -- Definitions used directly from the conditions
  sorry

end NUMINAMATH_GPT_average_words_per_puzzle_l865_86562


namespace NUMINAMATH_GPT_find_least_x_divisible_by_17_l865_86503

theorem find_least_x_divisible_by_17 (x k : ℕ) (h : x + 2 = 17 * k) : x = 15 :=
sorry

end NUMINAMATH_GPT_find_least_x_divisible_by_17_l865_86503


namespace NUMINAMATH_GPT_exactly_one_three_digit_perfect_cube_divisible_by_25_l865_86540

theorem exactly_one_three_digit_perfect_cube_divisible_by_25 :
  ∃! (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ (∃ k : ℕ, n = k^3) ∧ n % 25 = 0 :=
sorry

end NUMINAMATH_GPT_exactly_one_three_digit_perfect_cube_divisible_by_25_l865_86540


namespace NUMINAMATH_GPT_value_of_b_over_a_l865_86551

def rectangle_ratio (a b : ℝ) : Prop :=
  let d := Real.sqrt (a^2 + b^2)
  let P := 2 * (a + b)
  (b / d) = (d / (a + b))

theorem value_of_b_over_a (a b : ℝ) (h : rectangle_ratio a b) : b / a = 1 :=
by sorry

end NUMINAMATH_GPT_value_of_b_over_a_l865_86551


namespace NUMINAMATH_GPT_product_of_D_l865_86581

theorem product_of_D:
  ∀ (D : ℝ × ℝ), 
  (∃ M C : ℝ × ℝ, 
    M.1 = 4 ∧ M.2 = 3 ∧ 
    C.1 = 6 ∧ C.2 = -1 ∧ 
    M.1 = (C.1 + D.1) / 2 ∧ 
    M.2 = (C.2 + D.2) / 2) 
  → (D.1 * D.2 = 14) :=
sorry

end NUMINAMATH_GPT_product_of_D_l865_86581


namespace NUMINAMATH_GPT_speed_with_stream_l865_86529

variable (V_as V_m V_ws : ℝ)

theorem speed_with_stream (h1 : V_as = 6) (h2 : V_m = 2) : V_ws = V_m + (V_as - V_m) :=
by
  sorry

end NUMINAMATH_GPT_speed_with_stream_l865_86529


namespace NUMINAMATH_GPT_initial_hamburgers_count_is_nine_l865_86575

-- Define the conditions
def hamburgers_initial (total_hamburgers : ℕ) (additional_hamburgers : ℕ) : ℕ :=
  total_hamburgers - additional_hamburgers

-- The statement to be proved
theorem initial_hamburgers_count_is_nine :
  hamburgers_initial 12 3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_initial_hamburgers_count_is_nine_l865_86575


namespace NUMINAMATH_GPT_minimum_value_l865_86555

/-- The minimum value of the expression (x+2)^2 / (y-2) + (y+2)^2 / (x-2)
    for real numbers x > 2 and y > 2 is 50. -/
theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ z, z = (x + 2) ^ 2 / (y - 2) + (y + 2) ^ 2 / (x - 2) ∧ z = 50 :=
sorry

end NUMINAMATH_GPT_minimum_value_l865_86555


namespace NUMINAMATH_GPT_find_b_l865_86542

theorem find_b
  (b : ℝ)
  (h1 : ∃ r : ℝ, 2 * r^2 + b * r - 65 = 0 ∧ r = 5)
  (h2 : 2 * 5^2 + b * 5 - 65 = 0) :
  b = 3 := by
  sorry

end NUMINAMATH_GPT_find_b_l865_86542


namespace NUMINAMATH_GPT_domain_of_sqrt_sum_l865_86598

theorem domain_of_sqrt_sum (x : ℝ) (h1 : 3 + x ≥ 0) (h2 : 1 - x ≥ 0) : -3 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_sum_l865_86598


namespace NUMINAMATH_GPT_max_slope_avoiding_lattice_points_l865_86584

theorem max_slope_avoiding_lattice_points :
  ∃ a : ℝ, (1 < a ∧ ∀ m : ℝ, (1 < m ∧ m < a) → (∀ x : ℤ, (10 < x ∧ x ≤ 200) → ∃ k : ℝ, y = m * x + 5 ∧ (m * x + 5 ≠ k))) ∧ a = 101 / 100 :=
sorry

end NUMINAMATH_GPT_max_slope_avoiding_lattice_points_l865_86584


namespace NUMINAMATH_GPT_magic_square_solution_l865_86534

theorem magic_square_solution (d e k f g h x y : ℤ)
  (h1 : x + 4 + f = 87 + d + f)
  (h2 : x + d + h = 87 + e + h)
  (h3 : x + y + 87 = 4 + d + e)
  (h4 : f + g + h = x + y + 87)
  (h5 : d = x - 83)
  (h6 : e = 2 * x - 170)
  (h7 : y = 3 * x - 274)
  (h8 : f = g)
  (h9 : g = h) :
  x = 62 ∧ y = -88 :=
by
  sorry

end NUMINAMATH_GPT_magic_square_solution_l865_86534


namespace NUMINAMATH_GPT_solve_for_x_l865_86504

theorem solve_for_x : ∃ x : ℚ, 7 * (4 * x + 3) - 3 = -3 * (2 - 5 * x) + 5 * x / 2 ∧ x = -16 / 7 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l865_86504


namespace NUMINAMATH_GPT_sin_double_angle_plus_pi_over_4_l865_86527

theorem sin_double_angle_plus_pi_over_4 (α : ℝ) 
  (h : Real.tan α = 3) : 
  Real.sin (2 * α + Real.pi / 4) = -Real.sqrt 2 / 10 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_plus_pi_over_4_l865_86527


namespace NUMINAMATH_GPT_finished_year_eq_183_l865_86502

theorem finished_year_eq_183 (x : ℕ) (h1 : x < 200) 
  (h2 : x ^ 13 = 258145266804692077858261512663) : x = 183 :=
sorry

end NUMINAMATH_GPT_finished_year_eq_183_l865_86502


namespace NUMINAMATH_GPT_total_amount_correct_l865_86520

noncomputable def total_amount : ℝ :=
  let nissin_noodles := 24 * 1.80 * 0.80
  let master_kong_tea := 6 * 1.70 * 0.80
  let shanlin_soup := 5 * 3.40
  let shuanghui_sausage := 3 * 11.20 * 0.90
  nissin_noodles + master_kong_tea + shanlin_soup + shuanghui_sausage

theorem total_amount_correct : total_amount = 89.96 := by
  sorry

end NUMINAMATH_GPT_total_amount_correct_l865_86520


namespace NUMINAMATH_GPT_barbeck_steve_guitar_ratio_l865_86544

theorem barbeck_steve_guitar_ratio (b s d : ℕ) 
  (h1 : b = s) 
  (h2 : d = 3 * b) 
  (h3 : b + s + d = 27) 
  (h4 : d = 18) : 
  b / s = 2 / 1 := 
by 
  sorry

end NUMINAMATH_GPT_barbeck_steve_guitar_ratio_l865_86544


namespace NUMINAMATH_GPT_simplified_expression_l865_86596

theorem simplified_expression :
  ( (81 / 16) ^ (3 / 4) - (-1) ^ 0 ) = 19 / 8 := 
by 
  -- It is a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_simplified_expression_l865_86596


namespace NUMINAMATH_GPT_estimate_pi_l865_86530

theorem estimate_pi :
  ∀ (r : ℝ) (side_length : ℝ) (total_beans : ℕ) (beans_in_circle : ℕ),
  r = 1 →
  side_length = 2 →
  total_beans = 80 →
  beans_in_circle = 64 →
  (π = 3.2) :=
by
  intros r side_length total_beans beans_in_circle hr hside htotal hin_circle
  sorry

end NUMINAMATH_GPT_estimate_pi_l865_86530


namespace NUMINAMATH_GPT_price_of_turbans_l865_86571

theorem price_of_turbans : 
  ∀ (salary_A salary_B salary_C : ℝ) (months_A months_B months_C : ℕ) (payment_A payment_B payment_C : ℝ)
    (prorated_salary_A prorated_salary_B prorated_salary_C : ℝ),
  salary_A = 120 → 
  salary_B = 150 → 
  salary_C = 180 → 
  months_A = 8 → 
  months_B = 7 → 
  months_C = 10 → 
  payment_A = 80 → 
  payment_B = 87.50 → 
  payment_C = 150 → 
  prorated_salary_A = (salary_A * (months_A / 12 : ℝ)) → 
  prorated_salary_B = (salary_B * (months_B / 12 : ℝ)) → 
  prorated_salary_C = (salary_C * (months_C / 12 : ℝ)) → 
  ∃ (price_A price_B price_C : ℝ),
  price_A = payment_A - prorated_salary_A ∧ 
  price_B = payment_B - prorated_salary_B ∧ 
  price_C = payment_C - prorated_salary_C ∧ 
  price_A = 0 ∧ price_B = 0 ∧ price_C = 0 := 
by
  sorry

end NUMINAMATH_GPT_price_of_turbans_l865_86571


namespace NUMINAMATH_GPT_rabbit_speed_l865_86568

theorem rabbit_speed (x : ℕ) :
  2 * (2 * x + 4) = 188 → x = 45 := by
  sorry

end NUMINAMATH_GPT_rabbit_speed_l865_86568


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_purely_imaginary_l865_86525

theorem necessary_but_not_sufficient_condition_for_purely_imaginary (m : ℂ) :
  (1 - m^2 + (1 + m) * Complex.I = 0 → m = 1) ∧ 
  ((1 - m^2 + (1 + m) * Complex.I = 0 ↔ m = 1) = false) := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_purely_imaginary_l865_86525


namespace NUMINAMATH_GPT_salary_of_E_l865_86580

theorem salary_of_E (A B C D E : ℕ) (avg_salary : ℕ) 
  (hA : A = 8000) 
  (hB : B = 5000) 
  (hC : C = 11000) 
  (hD : D = 7000) 
  (h_avg : avg_salary = 8000) 
  (h_total_avg : avg_salary * 5 = A + B + C + D + E) : 
  E = 9000 :=
by {
  sorry
}

end NUMINAMATH_GPT_salary_of_E_l865_86580


namespace NUMINAMATH_GPT_total_original_grain_l865_86576

-- Define initial conditions
variables (initial_warehouse1 : ℕ) (initial_warehouse2 : ℕ)
-- Define the amount of grain transported away from the first warehouse
def transported_away := 2500
-- Define the amount of grain in the second warehouse
def warehouse2_initial := 50200

-- Prove the total original amount of grain in the two warehouses
theorem total_original_grain 
  (h1 : transported_away = 2500)
  (h2 : warehouse2_initial = 50200)
  (h3 : initial_warehouse1 - transported_away = warehouse2_initial) : 
  initial_warehouse1 + warehouse2_initial = 102900 :=
sorry

end NUMINAMATH_GPT_total_original_grain_l865_86576


namespace NUMINAMATH_GPT_solve_linear_system_l865_86559

theorem solve_linear_system :
  ∃ x y : ℤ, x + 9773 = 13200 ∧ 2 * x - 3 * y = 1544 ∧ x = 3427 ∧ y = 1770 := by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l865_86559


namespace NUMINAMATH_GPT_fred_speed_l865_86524

variable {F : ℝ} -- Fred's speed
variable {T : ℝ} -- Time in hours

-- Conditions
def initial_distance : ℝ := 35
def sam_speed : ℝ := 5
def sam_distance : ℝ := 25
def fred_distance := initial_distance - sam_distance

-- Theorem to prove
theorem fred_speed (h1 : T = sam_distance / sam_speed) (h2 : fred_distance = F * T) :
  F = 2 :=
by
  sorry

end NUMINAMATH_GPT_fred_speed_l865_86524


namespace NUMINAMATH_GPT_find_b_for_intersection_l865_86538

theorem find_b_for_intersection (b : ℝ) :
  (∀ x : ℝ, bx^2 + 2 * x + 3 = 3 * x + 4 → bx^2 - x - 1 = 0) →
  (∀ x : ℝ, x^2 * b - x - 1 = 0 → (1 + 4 * b = 0) → b = -1/4) :=
by
  intros h_eq h_discriminant h_solution
  sorry

end NUMINAMATH_GPT_find_b_for_intersection_l865_86538


namespace NUMINAMATH_GPT_exists_power_of_two_with_last_n_digits_ones_and_twos_l865_86536

theorem exists_power_of_two_with_last_n_digits_ones_and_twos (N : ℕ) (hN : 0 < N) :
  ∃ k : ℕ, ∀ i < N, ∃ (d : ℕ), d = 1 ∨ d = 2 ∧ 
    (2^k % 10^N) / 10^i % 10 = d :=
sorry

end NUMINAMATH_GPT_exists_power_of_two_with_last_n_digits_ones_and_twos_l865_86536


namespace NUMINAMATH_GPT_range_of_a_l865_86550

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a+3) / (5-a)) → -3 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l865_86550


namespace NUMINAMATH_GPT_g_neg6_eq_neg1_l865_86511

def f : ℝ → ℝ := fun x => 4 * x - 6
def g : ℝ → ℝ := fun x => 2 * x^2 + 7 * x - 1

theorem g_neg6_eq_neg1 : g (-6) = -1 := by
  sorry

end NUMINAMATH_GPT_g_neg6_eq_neg1_l865_86511


namespace NUMINAMATH_GPT_firefighter_remaining_money_correct_l865_86545

noncomputable def firefighter_weekly_earnings : ℕ := 30 * 48
noncomputable def firefighter_monthly_earnings : ℕ := firefighter_weekly_earnings * 4
noncomputable def firefighter_rent_expense : ℕ := firefighter_monthly_earnings / 3
noncomputable def firefighter_food_expense : ℕ := 500
noncomputable def firefighter_tax_expense : ℕ := 1000
noncomputable def firefighter_total_expenses : ℕ := firefighter_rent_expense + firefighter_food_expense + firefighter_tax_expense
noncomputable def firefighter_remaining_money : ℕ := firefighter_monthly_earnings - firefighter_total_expenses

theorem firefighter_remaining_money_correct :
  firefighter_remaining_money = 2340 :=
by 
  rfl

end NUMINAMATH_GPT_firefighter_remaining_money_correct_l865_86545


namespace NUMINAMATH_GPT_sara_picked_peaches_l865_86535

def peaches_original : ℕ := 24
def peaches_now : ℕ := 61
def peaches_picked (p_o p_n : ℕ) : ℕ := p_n - p_o

theorem sara_picked_peaches : peaches_picked peaches_original peaches_now = 37 :=
by
  sorry

end NUMINAMATH_GPT_sara_picked_peaches_l865_86535
