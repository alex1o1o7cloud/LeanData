import Mathlib

namespace uncovered_area_l104_10408

def shoebox_height : ℕ := 4
def shoebox_width : ℕ := 6
def block_side : ℕ := 4

theorem uncovered_area (height width side : ℕ) (h : height = shoebox_height) (w : width = shoebox_width) (s : side = block_side) :
  (width * height) - (side * side) = 8 :=
by
  rw [h, w, s]
  -- Area of shoebox bottom = width * height
  -- Area of square block = side * side
  -- Uncovered area = (width * height) - (side * side)
  -- Therefore, (6 * 4) - (4 * 4) = 24 - 16 = 8
  sorry

end uncovered_area_l104_10408


namespace koschei_coins_l104_10410

theorem koschei_coins :
  ∃ a : ℕ, a % 10 = 7 ∧ a % 12 = 9 ∧ 300 ≤ a ∧ a ≤ 400 ∧ a = 357 :=
by
  sorry

end koschei_coins_l104_10410


namespace tangent_line_eq_bounded_area_l104_10459

-- Given two parabolas and a tangent line, and a positive constant a
variables (a : ℝ)
variables (y1 y2 l : ℝ → ℝ)

-- Conditions:
def parabola1 := ∀ (x : ℝ), y1 x = x^2 + a * x
def parabola2 := ∀ (x : ℝ), y2 x = x^2 - 2 * a * x
def tangent_line := ∀ (x : ℝ), l x = - (a / 2) * x - (9 * a^2 / 16)
def a_positive := a > 0

-- Proof goals:
theorem tangent_line_eq : 
  parabola1 a y1 ∧ parabola2 a y2 ∧ tangent_line a l ∧ a_positive a 
  → ∀ x, (y1 x = l x ∨ y2 x = l x) :=
sorry

theorem bounded_area : 
  parabola1 a y1 ∧ parabola2 a y2 ∧ tangent_line a l ∧ a_positive a 
  → ∫ (x : ℝ) in (-3 * a / 4)..(3 * a / 4), (y1 x - l x) + (y2 x - l x) = 9 * a^3 / 8 :=
sorry

end tangent_line_eq_bounded_area_l104_10459


namespace sin_theta_plus_2pi_div_3_cos_theta_minus_5pi_div_6_l104_10474

variable (θ : ℝ)

theorem sin_theta_plus_2pi_div_3 (h : Real.sin (θ - Real.pi / 3) = 1 / 3) :
  Real.sin (θ + 2 * Real.pi / 3) = -1 / 3 :=
  sorry

theorem cos_theta_minus_5pi_div_6 (h : Real.sin (θ - Real.pi / 3) = 1 / 3) :
  Real.cos (θ - 5 * Real.pi / 6) = 1 / 3 :=
  sorry

end sin_theta_plus_2pi_div_3_cos_theta_minus_5pi_div_6_l104_10474


namespace delta_discount_percentage_l104_10435

theorem delta_discount_percentage (original_delta : ℝ) (original_united : ℝ)
  (united_discount_percent : ℝ) (savings : ℝ) (delta_discounted : ℝ) : 
  original_delta - delta_discounted = 0.2 * original_delta := by
  -- Given conditions
  let discounted_united := original_united * (1 - united_discount_percent / 100)
  have : delta_discounted = discounted_united - savings := sorry
  let delta_discount_amount := original_delta - delta_discounted
  have : delta_discount_amount = 0.2 * original_delta := sorry
  exact this

end delta_discount_percentage_l104_10435


namespace dividend_is_2160_l104_10417

theorem dividend_is_2160 (d q r : ℕ) (h₁ : d = 2016 + d) (h₂ : q = 15) (h₃ : r = 0) : d = 2160 :=
by
  sorry

end dividend_is_2160_l104_10417


namespace smallest_possible_Y_l104_10469

def digits (n : ℕ) : List ℕ := -- hypothetical function to get the digits of a number
  sorry

def is_divisible (n d : ℕ) : Prop := d ∣ n

theorem smallest_possible_Y :
  ∃ (U : ℕ), (∀ d ∈ digits U, d = 0 ∨ d = 1) ∧ is_divisible U 18 ∧ U / 18 = 61728395 :=
by
  sorry

end smallest_possible_Y_l104_10469


namespace area_of_rectangle_l104_10414

def length : ℝ := 0.5
def width : ℝ := 0.24

theorem area_of_rectangle :
  length * width = 0.12 :=
by
  sorry

end area_of_rectangle_l104_10414


namespace manager_salary_l104_10470

theorem manager_salary 
    (avg_salary_18 : ℕ)
    (new_avg_salary : ℕ)
    (num_employees : ℕ)
    (num_employees_with_manager : ℕ)
    (old_total_salary : ℕ := num_employees * avg_salary_18)
    (new_total_salary : ℕ := num_employees_with_manager * new_avg_salary) :
    (new_avg_salary = avg_salary_18 + 200) →
    (old_total_salary = 18 * 2000) →
    (new_total_salary = 19 * (2000 + 200)) →
    new_total_salary - old_total_salary = 5800 :=
by
  intros h1 h2 h3
  sorry

end manager_salary_l104_10470


namespace sum_of_possible_two_digit_values_l104_10436

theorem sum_of_possible_two_digit_values (d : ℕ) (h1 : 0 < d) (h2 : d < 100) (h3 : 137 % d = 6) : d = 131 :=
by
  sorry

end sum_of_possible_two_digit_values_l104_10436


namespace determine_y_increase_volume_l104_10463

noncomputable def volume_increase_y (r h y : ℝ) : Prop :=
  (1/3) * Real.pi * (r + y)^2 * h = (1/3) * Real.pi * r^2 * (h + y)

theorem determine_y_increase_volume (y : ℝ) :
  volume_increase_y 5 12 y ↔ y = 31 / 12 :=
by
  sorry

end determine_y_increase_volume_l104_10463


namespace alice_needs_to_add_stamps_l104_10424

variable (A B E P D : ℕ)
variable (h₁ : B = 4 * E)
variable (h₂ : E = 3 * P)
variable (h₃ : P = 2 * D)
variable (h₄ : D = A + 5)
variable (h₅ : A = 65)

theorem alice_needs_to_add_stamps : (1680 - A = 1615) :=
by
  sorry

end alice_needs_to_add_stamps_l104_10424


namespace x_squared_minus_y_squared_l104_10423

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : x - y = 4) :
  x^2 - y^2 = 80 :=
by
  -- Proof goes here
  sorry

end x_squared_minus_y_squared_l104_10423


namespace somu_age_to_father_age_ratio_l104_10406

theorem somu_age_to_father_age_ratio
  (S : ℕ) (F : ℕ)
  (h1 : S = 10)
  (h2 : S - 5 = (1/5) * (F - 5)) :
  S / F = 1 / 3 :=
by
  sorry

end somu_age_to_father_age_ratio_l104_10406


namespace crackers_count_l104_10437

theorem crackers_count (crackers_Marcus crackers_Mona crackers_Nicholas : ℕ) 
  (h1 : crackers_Marcus = 3 * crackers_Mona)
  (h2 : crackers_Nicholas = crackers_Mona + 6)
  (h3 : crackers_Marcus = 27) : crackers_Nicholas = 15 := 
by 
  sorry

end crackers_count_l104_10437


namespace regression_correlation_relation_l104_10415

variable (b r : ℝ)

theorem regression_correlation_relation (h : b = 0) : r = 0 := 
sorry

end regression_correlation_relation_l104_10415


namespace nancy_packs_of_crayons_l104_10429

theorem nancy_packs_of_crayons (total_crayons : ℕ) (crayons_per_pack : ℕ) (h1 : total_crayons = 615) (h2 : crayons_per_pack = 15) : total_crayons / crayons_per_pack = 41 :=
by
  sorry

end nancy_packs_of_crayons_l104_10429


namespace inequality_inequality_always_holds_l104_10488

theorem inequality_inequality_always_holds (x y : ℝ) (h : x > y) : |x| > y :=
sorry

end inequality_inequality_always_holds_l104_10488


namespace hillary_sunday_spend_l104_10452

noncomputable def spend_per_sunday (total_spent : ℕ) (weeks : ℕ) (weekday_price : ℕ) (weekday_papers : ℕ) : ℕ :=
  (total_spent - weeks * weekday_papers * weekday_price) / weeks

theorem hillary_sunday_spend :
  spend_per_sunday 2800 8 50 3 = 200 :=
sorry

end hillary_sunday_spend_l104_10452


namespace final_price_of_pencil_l104_10447

-- Define the initial constants
def initialCost : ℝ := 4.00
def christmasDiscount : ℝ := 0.63
def seasonalDiscountRate : ℝ := 0.07
def finalDiscountRate : ℝ := 0.05
def taxRate : ℝ := 0.065

-- Define the steps of the problem concisely
def priceAfterChristmasDiscount := initialCost - christmasDiscount
def priceAfterSeasonalDiscount := priceAfterChristmasDiscount * (1 - seasonalDiscountRate)
def priceAfterFinalDiscount := priceAfterSeasonalDiscount * (1 - finalDiscountRate)
def finalPrice := priceAfterFinalDiscount * (1 + taxRate)

-- The theorem to be proven
theorem final_price_of_pencil :
  abs (finalPrice - 3.17) < 0.01 := by
  sorry

end final_price_of_pencil_l104_10447


namespace determine_angle_range_l104_10427

variable (α : ℝ)

theorem determine_angle_range 
  (h1 : 0 < α) 
  (h2 : α < 2 * π) 
  (h_sin : Real.sin α < 0) 
  (h_cos : Real.cos α > 0) : 
  (3 * π / 2 < α ∧ α < 2 * π) := 
sorry

end determine_angle_range_l104_10427


namespace find_m_if_purely_imaginary_l104_10431

theorem find_m_if_purely_imaginary : ∀ m : ℝ, (m^2 - 5*m + 6 = 0) → (m = 2) :=
by 
  intro m
  intro h
  sorry

end find_m_if_purely_imaginary_l104_10431


namespace find_value_of_expression_l104_10448

-- Define non-negative variables
variables (x y z : ℝ) 

-- Conditions
def cond1 := x ^ 2 + x * y + y ^ 2 / 3 = 25
def cond2 := y ^ 2 / 3 + z ^ 2 = 9
def cond3 := z ^ 2 + z * x + x ^ 2 = 16

-- Target statement to be proven
theorem find_value_of_expression (h1 : cond1 x y) (h2 : cond2 y z) (h3 : cond3 z x) : 
  x * y + 2 * y * z + 3 * z * x = 24 * Real.sqrt 3 :=
sorry

end find_value_of_expression_l104_10448


namespace infinite_n_exists_r_s_t_l104_10478

noncomputable def a (n : ℕ) : ℝ := n^(1/3 : ℝ)
noncomputable def b (n : ℕ) : ℝ := 1 / (a n - ⌊a n⌋)
noncomputable def c (n : ℕ) : ℝ := 1 / (b n - ⌊b n⌋)

theorem infinite_n_exists_r_s_t :
  ∃ (n : ℕ) (r s t : ℤ), (0 < n ∧ ¬∃ k : ℕ, n = k^3) ∧ (¬(r = 0 ∧ s = 0 ∧ t = 0)) ∧ (r * a n + s * b n + t * c n = 0) :=
sorry

end infinite_n_exists_r_s_t_l104_10478


namespace intersection_complement_N_l104_10403

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {4, 5, 6}
def C_U_M : Set ℕ := U \ M

theorem intersection_complement_N : (C_U_M ∩ N) = {4, 6} :=
by
  sorry

end intersection_complement_N_l104_10403


namespace solve_quadratic_and_linear_equations_l104_10457

theorem solve_quadratic_and_linear_equations :
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 → x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) ∧
  (∀ x : ℝ, (x + 3) * (x - 3) = 3 * (x + 3) → x = -3 ∨ x = 6) :=
by
  sorry

end solve_quadratic_and_linear_equations_l104_10457


namespace roots_reciprocal_sum_l104_10441

theorem roots_reciprocal_sum
  {a b c : ℂ}
  (h_roots : ∀ x : ℂ, (x - a) * (x - b) * (x - c) = x^3 - x + 1) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) = -2 :=
by
  sorry

end roots_reciprocal_sum_l104_10441


namespace cos_a3_value_l104_10404

theorem cos_a3_value (a : ℕ → ℝ) (h : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 1 + a 3 + a 5 = Real.pi) : 
  Real.cos (a 3) = 1/2 := 
by 
  sorry

end cos_a3_value_l104_10404


namespace find_f_0_plus_f_neg_1_l104_10400

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - x^2 else
if x < 0 then -(2^(-x) - (-x)^2) else 0

theorem find_f_0_plus_f_neg_1 : f 0 + f (-1) = -1 := by
  sorry

end find_f_0_plus_f_neg_1_l104_10400


namespace julia_baking_days_l104_10466

variable (bakes_per_day : ℕ)
variable (clifford_eats_per_two_days : ℕ)
variable (final_cakes : ℕ)

def number_of_baking_days : ℕ :=
  2 * (final_cakes / (bakes_per_day * 2 - clifford_eats_per_two_days))

theorem julia_baking_days (h1 : bakes_per_day = 4)
                        (h2 : clifford_eats_per_two_days = 1)
                        (h3 : final_cakes = 21) :
  number_of_baking_days bakes_per_day clifford_eats_per_two_days final_cakes = 6 :=
by {
  sorry
}

end julia_baking_days_l104_10466


namespace area_of_park_l104_10479

theorem area_of_park (L B : ℝ) (h1 : L / B = 1 / 3) (h2 : 12 * 1000 / 60 * 4 = 2 * (L + B)) : 
  L * B = 30000 :=
by
  sorry

end area_of_park_l104_10479


namespace range_of_y0_l104_10464

theorem range_of_y0
  (y0 : ℝ)
  (h_tangent : ∃ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧ ((↑(Real.sqrt 3 - N.1)^2 + (y0 - N.2)^2) = 1))
  (h_angle : ∀ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧ ((↑(Real.sqrt 3 - N.1)^2 + (y0 - N.2)^2 = 1)) → (Real.arccos ((Real.sqrt 3 - N.1)/Real.sqrt ((3 - 2 * N.1 * Real.sqrt 3 + N.1^2) + (y0 - N.2)^2)) ≥ π / 6)) :
  -1 ≤ y0 ∧ y0 ≤ 1 :=
by
  sorry

end range_of_y0_l104_10464


namespace fraction_value_l104_10413

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the theorem
theorem fraction_value (h : 2 * x = -y) : (x * y) / (x^2 - y^2) = 2 / 3 :=
by
  sorry

end fraction_value_l104_10413


namespace simplify_fraction_l104_10473

theorem simplify_fraction (d : ℤ) : (5 + 4 * d) / 9 - 3 = (4 * d - 22) / 9 :=
by
  sorry

end simplify_fraction_l104_10473


namespace derivative_of_y_l104_10454

noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x + x * Real.exp x

theorem derivative_of_y (x : ℝ) (hx : x > 0) : 
  deriv y x = (1 - Real.log x) / (x^2) + (x + 1) * Real.exp x := by
  sorry

end derivative_of_y_l104_10454


namespace find_n_for_perfect_square_l104_10420

theorem find_n_for_perfect_square :
  ∃ (n : ℕ), n > 0 ∧ ∃ (m : ℤ), n^2 + 5 * n + 13 = m^2 ∧ n = 4 :=
by
  sorry

end find_n_for_perfect_square_l104_10420


namespace wood_cost_l104_10418

theorem wood_cost (C : ℝ) (h1 : 20 * 15 = 300) (h2 : 300 - C = 200) : C = 100 :=
by
  -- The proof is to be filled here, but it is currently skipped with 'sorry'.
  sorry

end wood_cost_l104_10418


namespace value_of_m_l104_10402

theorem value_of_m (m x : ℝ) (h : x - 4 ≠ 0) (hx_pos : x > 0) 
  (eqn : m / (x - 4) - (1 - x) / (4 - x) = 0) : m = 3 := 
by
  sorry

end value_of_m_l104_10402


namespace eggs_in_box_l104_10433

-- Given conditions as definitions in Lean 4
def initial_eggs : ℕ := 7
def additional_whole_eggs : ℕ := 3

-- The proof statement
theorem eggs_in_box : initial_eggs + additional_whole_eggs = 10 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end eggs_in_box_l104_10433


namespace A_3_2_eq_29_l104_10465

def A : ℕ → ℕ → ℕ
| 0, n     => n + 1
| (m + 1), 0 => A m 1
| (m + 1), (n + 1) => A m (A (m + 1) n)

theorem A_3_2_eq_29 : A 3 2 = 29 := by
  sorry

end A_3_2_eq_29_l104_10465


namespace complex_purely_imaginary_l104_10489

theorem complex_purely_imaginary (m : ℝ) :
  (m^2 - 3*m + 2 = 0) ∧ (m^2 - 2*m ≠ 0) → m = 1 :=
by {
  sorry
}

end complex_purely_imaginary_l104_10489


namespace n_cubed_plus_5n_divisible_by_6_l104_10467

theorem n_cubed_plus_5n_divisible_by_6 (n : ℕ) : ∃ k : ℤ, n^3 + 5 * n = 6 * k :=
by
  sorry

end n_cubed_plus_5n_divisible_by_6_l104_10467


namespace product_of_two_numbers_l104_10412

theorem product_of_two_numbers (x y : ℕ) 
  (h1 : y = 15 * x) 
  (h2 : x + y = 400) : 
  x * y = 9375 :=
by
  sorry

end product_of_two_numbers_l104_10412


namespace combined_weight_l104_10496

theorem combined_weight (x y z : ℕ) (h1 : x + y = 110) (h2 : y + z = 130) (h3 : z + x = 150) : x + y + z = 195 :=
by
  sorry

end combined_weight_l104_10496


namespace negation_equiv_l104_10451

variable (p : Prop) [Nonempty ℝ]

def proposition := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

def negation_of_proposition : Prop := ∀ x : ℝ, Real.exp x - x - 1 > 0

theorem negation_equiv
  (h : proposition = p) : (¬ proposition) = negation_of_proposition := by
  sorry

end negation_equiv_l104_10451


namespace find_a10_l104_10458

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def a2_eq_4 (a : ℕ → ℝ) := a 2 = 4

def a6_eq_6 (a : ℕ → ℝ) := a 6 = 6

theorem find_a10 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h2 : a2_eq_4 a) (h6 : a6_eq_6 a) : 
  a 10 = 9 :=
sorry

end find_a10_l104_10458


namespace convert_to_rectangular_form_l104_10493

theorem convert_to_rectangular_form :
  2 * Real.sqrt 3 * Complex.exp (13 * Real.pi * Complex.I / 6) = 3 + Complex.I * Real.sqrt 3 :=
by
  sorry

end convert_to_rectangular_form_l104_10493


namespace ratio_of_mustang_models_length_l104_10453

theorem ratio_of_mustang_models_length :
  ∀ (full_size_length mid_size_length smallest_model_length : ℕ),
    full_size_length = 240 →
    mid_size_length = full_size_length / 10 →
    smallest_model_length = 12 →
    smallest_model_length / mid_size_length = 1/2 :=
by
  intros full_size_length mid_size_length smallest_model_length h1 h2 h3
  sorry

end ratio_of_mustang_models_length_l104_10453


namespace beth_lost_red_marbles_l104_10450

-- Definitions from conditions
def total_marbles : ℕ := 72
def marbles_per_color : ℕ := total_marbles / 3
variable (R : ℕ)  -- Number of red marbles Beth lost
def blue_marbles_lost : ℕ := 2 * R
def yellow_marbles_lost : ℕ := 3 * R
def marbles_left : ℕ := 42

-- Theorem we want to prove
theorem beth_lost_red_marbles (h : total_marbles - (R + blue_marbles_lost R + yellow_marbles_lost R) = marbles_left) :
  R = 5 :=
by
  sorry

end beth_lost_red_marbles_l104_10450


namespace spend_on_candy_l104_10490

variable (initial_money spent_on_oranges spent_on_apples remaining_money spent_on_candy : ℕ)

-- Conditions
axiom initial_amount : initial_money = 95
axiom spent_on_oranges_value : spent_on_oranges = 14
axiom spent_on_apples_value : spent_on_apples = 25
axiom remaining_amount : remaining_money = 50

-- Question as a theorem
theorem spend_on_candy :
  spent_on_candy = initial_money - (spent_on_oranges + spent_on_apples) - remaining_money :=
by sorry

end spend_on_candy_l104_10490


namespace total_pencils_bought_l104_10426

theorem total_pencils_bought (x y : ℕ) (y_pos : 0 < y) (initial_cost : y * (x + 10) = 5 * x) (later_cost : (4 * y) * (x + 10) = 20 * x) :
    x = 15 → (40 = x + x + 10) ∨ x = 40 → (90 = x + (x + 10)) :=
by
  sorry

end total_pencils_bought_l104_10426


namespace quadratic_transformed_correct_l104_10497

noncomputable def quadratic_transformed (a b c : ℝ) (r s : ℝ) (h1 : a ≠ 0) 
  (h_roots : r + s = -b / a ∧ r * s = c / a) : Polynomial ℝ :=
Polynomial.C (a * b * c) + Polynomial.C ((-(a + b) * b)) * Polynomial.X + Polynomial.X^2

-- The theorem statement
theorem quadratic_transformed_correct (a b c r s : ℝ) (h1 : a ≠ 0)
  (h_roots : r + s = -b / a ∧ r * s = c / a) :
  (quadratic_transformed a b c r s h1 h_roots).roots = {a * (r + b), a * (s + b)} :=
sorry

end quadratic_transformed_correct_l104_10497


namespace problem_prove_divisibility_l104_10416

theorem problem_prove_divisibility (n : ℕ) : 11 ∣ (5^(2*n) + 3^(n+2) + 3^n) :=
sorry

end problem_prove_divisibility_l104_10416


namespace pyramid_value_l104_10407

theorem pyramid_value (a b c d e f : ℕ) (h_b : b = 6) (h_d : d = 20) (h_prod1 : d = b * (20 / b)) (h_prod2 : e = (20 / b) * c) (h_prod3 : f = c * (72 / c)) : a = b * c → a = 54 :=
by 
  -- Assuming the proof would assert the calculations done in the solution.
  sorry

end pyramid_value_l104_10407


namespace Ramya_reads_total_124_pages_l104_10439

theorem Ramya_reads_total_124_pages :
  let total_pages : ℕ := 300
  let pages_read_monday := (1/5 : ℚ) * total_pages
  let pages_remaining := total_pages - pages_read_monday
  let pages_read_tuesday := (4/15 : ℚ) * pages_remaining
  pages_read_monday + pages_read_tuesday = 124 := 
by
  sorry

end Ramya_reads_total_124_pages_l104_10439


namespace ticket_queue_correct_l104_10419

-- Define the conditions
noncomputable def ticket_queue_count (m n : ℕ) (h : n ≥ m) : ℕ :=
  (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1))

-- State the theorem
theorem ticket_queue_correct (m n : ℕ) (h : n ≥ m) :
  ticket_queue_count m n h = (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1)) :=
by
  sorry

end ticket_queue_correct_l104_10419


namespace domain_range_of_p_l104_10445

variable (h : ℝ → ℝ)
variable (h_domain : ∀ x, -1 ≤ x ∧ x ≤ 3)
variable (h_range : ∀ x, 0 ≤ h x ∧ h x ≤ 2)

def p (x : ℝ) : ℝ := 2 - h (x - 1)

theorem domain_range_of_p :
  (∀ x, 0 ≤ x ∧ x ≤ 4) ∧ (∀ y, 0 ≤ y ∧ y ≤ 2) :=
by
  -- Proof to show that the domain of p(x) is [0, 4] and the range is [0, 2]
  sorry

end domain_range_of_p_l104_10445


namespace solution_set_of_inequality_l104_10421

theorem solution_set_of_inequality (x : ℝ) : (x + 3) * (x - 5) < 0 ↔ (-3 < x ∧ x < 5) :=
by
  sorry

end solution_set_of_inequality_l104_10421


namespace cubic_difference_l104_10409

theorem cubic_difference (x y : ℝ) (h1 : x + y = 15) (h2 : 2 * x + y = 20) : x^3 - y^3 = -875 := 
by
  sorry

end cubic_difference_l104_10409


namespace negative_square_inequality_l104_10484

theorem negative_square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end negative_square_inequality_l104_10484


namespace rho_square_max_value_l104_10461

variable {a b x y c : ℝ}
variable (ha_pos : a > 0) (hb_pos : b > 0)
variable (ha_ge_b : a ≥ b)
variable (hx_range : 0 ≤ x ∧ x < a)
variable (hy_range : 0 ≤ y ∧ y < b)
variable (h_eq1 : a^2 + y^2 = b^2 + x^2)
variable (h_eq2 : b^2 + x^2 = (a - x)^2 + (b - y)^2 + c^2)

theorem rho_square_max_value : (a / b) ^ 2 ≤ 4 / 3 :=
sorry

end rho_square_max_value_l104_10461


namespace range_of_m_l104_10495

def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (m : ℝ) := {x : ℝ | x^2 - 2*x + m = 0}

theorem range_of_m (m : ℝ) : (A ∪ B m = A) ↔ m ∈ Set.Ici 1 :=
by
  sorry

end range_of_m_l104_10495


namespace lcm_hcf_product_l104_10430

theorem lcm_hcf_product (lcm hcf a b : ℕ) (hlcm : lcm = 2310) (hhcf : hcf = 30) (ha : a = 330) (eq : lcm * hcf = a * b) : b = 210 :=
by {
  sorry
}

end lcm_hcf_product_l104_10430


namespace quadratic_to_general_form_l104_10455

theorem quadratic_to_general_form (x : ℝ) :
  ∃ b : ℝ, (∀ a c : ℝ, (a = 3) ∧ (c = 1) → (a * x^2 + c = 6 * x) → b = -6) :=
by
  sorry

end quadratic_to_general_form_l104_10455


namespace exponential_equality_l104_10468

theorem exponential_equality (n : ℕ) (h : 4 ^ n = 64 ^ 2) : n = 6 :=
  sorry

end exponential_equality_l104_10468


namespace find_x_l104_10440

theorem find_x (x : ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (x, 1)) :
  ((2 * a.fst - x, 2 * a.snd + 1) • b = 0) → x = -1 ∨ x = 3 :=
by
  sorry

end find_x_l104_10440


namespace remainder_mod_of_a_squared_subtract_3b_l104_10446

theorem remainder_mod_of_a_squared_subtract_3b (a b : ℕ) (h₁ : a % 7 = 2) (h₂ : b % 7 = 5) (h₃ : a^2 > 3 * b) : 
  (a^2 - 3 * b) % 7 = 3 := 
sorry

end remainder_mod_of_a_squared_subtract_3b_l104_10446


namespace find_expression_value_l104_10486

def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem find_expression_value (p q r s : ℝ) (h1 : g p q r s (-1) = 2) (h2 : g p q r s (-2) = -1) (h3 : g p q r s (1) = -2) :
  9 * p - 3 * q + 3 * r - s = -2 :=
by
  sorry

end find_expression_value_l104_10486


namespace equal_phrases_impossible_l104_10481

-- Define the inhabitants and the statements they make.
def inhabitants : ℕ := 1234

-- Define what it means to be a knight or a liar.
inductive Person
| knight : Person
| liar : Person

-- Define the statements "He is a knight!" and "He is a liar!"
inductive Statement
| is_knight : Statement
| is_liar : Statement

-- Define the pairings and types of statements 
def pairings (inhabitant1 inhabitant2 : Person) : Statement :=
match inhabitant1, inhabitant2 with
| Person.knight, Person.knight => Statement.is_knight
| Person.liar, Person.liar => Statement.is_knight
| Person.knight, Person.liar => Statement.is_liar
| Person.liar, Person.knight => Statement.is_knight

-- Define the total number of statements
def total_statements (pairs : ℕ) : ℕ := 2 * pairs

-- Theorem stating the mathematical equivalent proof problem
theorem equal_phrases_impossible :
  ¬ ∃ n : ℕ, n = inhabitants / 2 ∧ total_statements n = inhabitants ∧
    (pairings Person.knight Person.liar = Statement.is_knight ∧
     pairings Person.liar Person.knight = Statement.is_knight ∧
     (pairings Person.knight Person.knight = Statement.is_knight ∧
      pairings Person.liar Person.liar = Statement.is_knight) ∨
      (pairings Person.knight Person.liar = Statement.is_liar ∧
       pairings Person.liar Person.knight = Statement.is_liar)) :=
sorry

end equal_phrases_impossible_l104_10481


namespace jenna_remaining_money_l104_10499

theorem jenna_remaining_money (m c : ℝ) (h : (1 / 4) * m = (1 / 2) * c) : (m - c) / m = 1 / 2 :=
by
  sorry

end jenna_remaining_money_l104_10499


namespace cos_squared_alpha_plus_pi_over_4_correct_l104_10460

variable (α : ℝ)
axiom sin_two_alpha : Real.sin (2 * α) = 2 / 3

theorem cos_squared_alpha_plus_pi_over_4_correct :
  Real.cos (α + Real.pi / 4) ^ 2 = 1 / 6 :=
by
  sorry

end cos_squared_alpha_plus_pi_over_4_correct_l104_10460


namespace solution_interval_l104_10443

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem solution_interval :
  ∃ x_0, f x_0 = 0 ∧ 2 < x_0 ∧ x_0 < 3 :=
by
  sorry

end solution_interval_l104_10443


namespace probability_green_jelly_bean_l104_10498

theorem probability_green_jelly_bean :
  let red := 10
  let green := 9
  let yellow := 5
  let blue := 7
  let total := red + green + yellow + blue
  (green : ℚ) / (total : ℚ) = 9 / 31 := by
  sorry

end probability_green_jelly_bean_l104_10498


namespace number_of_ways_to_assign_roles_l104_10485

theorem number_of_ways_to_assign_roles : 
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let men := 4
  let women := 5
  let total_roles := male_roles + female_roles + either_gender_roles
  let ways_to_assign_males := men * (men-1) * (men-2)
  let ways_to_assign_females := women * (women-1)
  let remaining_actors := men + women - male_roles - female_roles
  let ways_to_assign_either_gender := remaining_actors
  let total_ways := ways_to_assign_males * ways_to_assign_females * ways_to_assign_either_gender

  total_ways = 1920 :=
by
  sorry

end number_of_ways_to_assign_roles_l104_10485


namespace reese_practice_hours_l104_10405

-- Define the average number of weeks in a month
def avg_weeks_per_month : ℝ := 4.345

-- Define the number of hours Reese practices per week
def hours_per_week : ℝ := 4 

-- Define the number of months under consideration
def num_months : ℝ := 5

-- Calculate the total hours Reese will practice after five months
theorem reese_practice_hours :
  (num_months * avg_weeks_per_month * hours_per_week) = 86.9 :=
by
  -- We'll skip the proof part by adding sorry here
  sorry

end reese_practice_hours_l104_10405


namespace tub_drain_time_l104_10425

theorem tub_drain_time (time_for_five_sevenths : ℝ)
  (time_for_five_sevenths_eq_four : time_for_five_sevenths = 4) :
  let rate := time_for_five_sevenths / (5 / 7)
  let time_for_two_sevenths := 2 * rate
  time_for_two_sevenths = 11.2 := by
  -- Definitions and initial conditions
  sorry

end tub_drain_time_l104_10425


namespace sum_of_rationals_l104_10492

theorem sum_of_rationals (r1 r2 : ℚ) : ∃ r : ℚ, r = r1 + r2 :=
sorry

end sum_of_rationals_l104_10492


namespace sum_of_products_l104_10475

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62)
  (h2 : a + b + c = 18) : 
  a * b + b * c + c * a = 131 :=
sorry

end sum_of_products_l104_10475


namespace asphalt_road_proof_l104_10487

-- We define the initial conditions given in the problem
def man_hours (men days hours_per_day : Nat) : Nat :=
  men * days * hours_per_day

-- Given the conditions for asphalting 1 km road
def conditions_1 (men1 days1 hours_per_day1 : Nat) : Prop :=
  man_hours men1 days1 hours_per_day1 = 2880

-- Given that the second road is 2 km long
def conditions_2 (man_hours1 : Nat) : Prop :=
  2 * man_hours1 = 5760

-- Given the working conditions for the second road
def conditions_3 (men2 days2 hours_per_day2 : Nat) : Prop :=
  men2 * days2 * hours_per_day2 = 5760

-- The theorem to prove
theorem asphalt_road_proof 
  (men1 days1 hours_per_day1 days2 hours_per_day2 men2 : Nat)
  (H1 : conditions_1 men1 days1 hours_per_day1)
  (H2 : conditions_2 (man_hours men1 days1 hours_per_day1))
  (H3 : men2 * days2 * hours_per_day2 = 5760)
  : men2 = 20 :=
by
  sorry

end asphalt_road_proof_l104_10487


namespace solve_exponent_equation_l104_10422

theorem solve_exponent_equation (x y z : ℕ) :
  7^x + 1 = 3^y + 5^z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  sorry

end solve_exponent_equation_l104_10422


namespace roots_eq_202_l104_10449

theorem roots_eq_202 (p q : ℝ) 
  (h1 : ∀ x : ℝ, ((x + p) * (x + q) * (x + 10) = 0 ↔ (x = -p ∨ x = -q ∨ x = -10)) ∧ 
       ∀ x : ℝ, ((x + 5) ^ 2 = 0 ↔ x = -5)) 
  (h2 : ∀ x : ℝ, ((x + 2 * p) * (x + 4) * (x + 8) = 0 ↔ (x = -2 * p ∨ x = -4 ∨ x = -8)) ∧ 
       ∀ x : ℝ, ((x + q) * (x + 10) = 0 ↔ (x = -q ∨ x = -10))) 
  (hpq : p = q) (neq_5 : q ≠ 5) (p_2 : p = 2):
  100 * p + q = 202 := sorry

end roots_eq_202_l104_10449


namespace chef_bought_kilograms_of_almonds_l104_10462

def total_weight_of_nuts : ℝ := 0.52
def weight_of_pecans : ℝ := 0.38
def weight_of_almonds : ℝ := total_weight_of_nuts - weight_of_pecans

theorem chef_bought_kilograms_of_almonds : weight_of_almonds = 0.14 := by
  sorry

end chef_bought_kilograms_of_almonds_l104_10462


namespace number_of_terms_in_arithmetic_sequence_l104_10482

-- Definitions derived directly from the conditions
def first_term : ℕ := 2
def common_difference : ℕ := 4
def last_term : ℕ := 2010

-- Lean statement for the proof problem
theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, last_term = first_term + (n - 1) * common_difference ∧ n = 503 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l104_10482


namespace total_games_single_elimination_l104_10471

theorem total_games_single_elimination (teams : ℕ) (h_teams : teams = 24)
  (preliminary_matches : ℕ) (h_preliminary_matches : preliminary_matches = 8)
  (preliminary_teams : ℕ) (h_preliminary_teams : preliminary_teams = 16)
  (idle_teams : ℕ) (h_idle_teams : idle_teams = 8)
  (main_draw_teams : ℕ) (h_main_draw_teams : main_draw_teams = 16) :
  (games : ℕ) -> games = 23 :=
by
  sorry

end total_games_single_elimination_l104_10471


namespace proof_problem_l104_10401

theorem proof_problem
  (a b c : ℂ)
  (h1 : ac / (a + b) + ba / (b + c) + cb / (c + a) = -4)
  (h2 : bc / (a + b) + ca / (b + c) + ab / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 7 := 
sorry

end proof_problem_l104_10401


namespace find_other_number_l104_10476

theorem find_other_number
  (B : ℕ)
  (hcf_condition : Nat.gcd 24 B = 12)
  (lcm_condition : Nat.lcm 24 B = 396) :
  B = 198 :=
by
  sorry

end find_other_number_l104_10476


namespace distance_from_origin_12_5_l104_10456

def distance_from_origin (x y : ℕ) : ℕ := 
  Int.natAbs (Nat.sqrt (x * x + y * y))

theorem distance_from_origin_12_5 : distance_from_origin 12 5 = 13 := by
  sorry

end distance_from_origin_12_5_l104_10456


namespace find_f2_l104_10438

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

theorem find_f2 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3) : f a 2 = 7 := 
by 
  sorry

end find_f2_l104_10438


namespace carter_cheesecakes_l104_10480

theorem carter_cheesecakes (C : ℕ) (nm : ℕ) (nr : ℕ) (increase : ℕ) (this_week_cakes : ℕ) (usual_cakes : ℕ) :
  nm = 5 → nr = 8 → increase = 38 → 
  this_week_cakes = 3 * C + 3 * nm + 3 * nr → 
  usual_cakes = C + nm + nr → 
  this_week_cakes = usual_cakes + increase → 
  C = 6 :=
by
  intros hnm hnr hinc htw husual hcakes
  sorry

end carter_cheesecakes_l104_10480


namespace income_increase_correct_l104_10477

noncomputable def income_increase_percentage (I1 : ℝ) (S1 : ℝ) (E1 : ℝ) (I2 : ℝ) (S2 : ℝ) (E2 : ℝ) (P : ℝ) :=
  S1 = 0.5 * I1 ∧
  S2 = 2 * S1 ∧
  E1 = 0.5 * I1 ∧
  E2 = I2 - S2 ∧
  I2 = I1 * (1 + P / 100) ∧
  E1 + E2 = 2 * E1

theorem income_increase_correct (I1 : ℝ) (S1 : ℝ) (E1 : ℝ) (I2 : ℝ) (S2 : ℝ) (E2 : ℝ) (P : ℝ)
  (h1 : income_increase_percentage I1 S1 E1 I2 S2 E2 P) : P = 50 :=
sorry

end income_increase_correct_l104_10477


namespace bus_trip_cost_l104_10483

-- Problem Statement Definitions
def distance_AB : ℕ := 4500
def cost_per_kilometer_bus : ℚ := 0.20

-- Theorem Statement
theorem bus_trip_cost : distance_AB * cost_per_kilometer_bus = 900 := by
  sorry

end bus_trip_cost_l104_10483


namespace find_k_l104_10428

theorem find_k (x y k : ℝ) (hx : x = 2) (hy : y = 1) (h : k * x - y = 3) : k = 2 := by
  sorry

end find_k_l104_10428


namespace FG_square_l104_10472

def trapezoid_EFGH (EF FG GH EH : ℝ) : Prop :=
  ∃ x y : ℝ, 
  EF = 4 ∧
  EH = 31 ∧
  FG = x ∧
  GH = y ∧
  x^2 + (y - 4)^2 = 961 ∧
  x^2 = 4 * y

theorem FG_square (EF EH FG GH x y : ℝ) (h : trapezoid_EFGH EF FG GH EH) :
  FG^2 = 132 :=
by
  obtain ⟨x, y, h1, h2, h3, h4, h5, h6⟩ := h
  exact sorry

end FG_square_l104_10472


namespace value_of_unknown_number_l104_10442

theorem value_of_unknown_number (x n : ℤ) 
  (h1 : x = 88320) 
  (h2 : x + n + 9211 - 1569 = 11901) : 
  n = -84061 :=
by
  sorry

end value_of_unknown_number_l104_10442


namespace intersection_of_sets_l104_10411

noncomputable def A : Set ℝ := { x | x^2 - 1 > 0 }
noncomputable def B : Set ℝ := { x | Real.log x / Real.log 2 > 0 }

theorem intersection_of_sets :
  A ∩ B = { x | x > 1 } :=
by {
  sorry
}

end intersection_of_sets_l104_10411


namespace doug_lost_marbles_l104_10491

-- Definitions based on the conditions
variables (D D' : ℕ) -- D is the number of marbles Doug originally had, D' is the number Doug has now

-- Condition 1: Ed had 10 more marbles than Doug originally.
def ed_marble_initial (D : ℕ) : ℕ := D + 10

-- Condition 2: Ed had 45 marbles originally.
axiom ed_initial_marble_count : ed_marble_initial D = 45

-- Solve for D from condition 2
noncomputable def doug_initial_marble_count : ℕ := 45 - 10

-- Condition 3: Ed now has 21 more marbles than Doug.
axiom ed_current_marble_difference : 45 = D' + 21

-- Translate what we need to prove
theorem doug_lost_marbles : (doug_initial_marble_count - D') = 11 :=
by
    -- Insert math proof steps here
    sorry

end doug_lost_marbles_l104_10491


namespace no_real_solution_for_pairs_l104_10444

theorem no_real_solution_for_pairs (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬ (1 / a + 1 / b = 1 / (a + b)) :=
by
  sorry

end no_real_solution_for_pairs_l104_10444


namespace A_B_symmetric_x_axis_l104_10494

-- Definitions of points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -3)

-- Theorem stating the symmetry relationship between points A and B with respect to the x-axis
theorem A_B_symmetric_x_axis (xA yA xB yB : ℝ) (hA : A = (xA, yA)) (hB : B = (xB, yB)) :
  xA = xB ∧ yA = -yB := by
  sorry

end A_B_symmetric_x_axis_l104_10494


namespace fraction_budget_paid_l104_10434

variable (B : ℝ) (b k : ℝ)

-- Conditions
def condition1 : b = 0.30 * (B - k) := by sorry
def condition2 : k = 0.10 * (B - b) := by sorry

-- Proof that Jenny paid 35% of her budget for her book and snack
theorem fraction_budget_paid :
  b + k = 0.35 * B :=
by
  -- use condition1 and condition2 to prove the theorem
  sorry

end fraction_budget_paid_l104_10434


namespace path_length_of_dot_l104_10432

-- Define the edge length of the cube
def edge_length : ℝ := 3

-- Define the conditions of the problem
def cube_condition (l : ℝ) (rolling_without_slipping : Prop) (at_least_two_vertices_touching : Prop) (dot_at_one_corner : Prop) (returns_to_original_position : Prop) : Prop :=
  l = edge_length ∧ rolling_without_slipping ∧ at_least_two_vertices_touching ∧ dot_at_one_corner ∧ returns_to_original_position

-- Define the theorem to be proven
theorem path_length_of_dot (rolling_without_slipping : Prop) (at_least_two_vertices_touching : Prop) (dot_at_one_corner : Prop) (returns_to_original_position : Prop) :
  cube_condition edge_length rolling_without_slipping at_least_two_vertices_touching dot_at_one_corner returns_to_original_position →
  ∃ c : ℝ, c = 6 ∧ (c * Real.pi) = 6 * Real.pi :=
by
  intro h
  sorry

end path_length_of_dot_l104_10432
