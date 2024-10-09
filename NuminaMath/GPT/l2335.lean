import Mathlib

namespace infinite_triangles_with_sides_x_y_10_l2335_233540

theorem infinite_triangles_with_sides_x_y_10 (x y : Nat) (hx : 0 < x) (hy : 0 < y) : 
  (∃ n : Nat, n > 5 ∧ ∀ m ≥ n, ∃ x y : Nat, 0 < x ∧ 0 < y ∧ x + y > 10 ∧ x + 10 > y ∧ y + 10 > x) :=
sorry

end infinite_triangles_with_sides_x_y_10_l2335_233540


namespace find_m_value_l2335_233595

theorem find_m_value (m : ℤ) : (∃ a : ℤ, x^2 + 2 * (m + 1) * x + 25 = (x + a)^2) ↔ (m = 4 ∨ m = -6) := 
sorry

end find_m_value_l2335_233595


namespace dealer_gross_profit_l2335_233599

theorem dealer_gross_profit
  (purchase_price : ℝ)
  (markup_rate : ℝ)
  (discount_rate : ℝ)
  (initial_selling_price : ℝ)
  (final_selling_price : ℝ)
  (gross_profit : ℝ)
  (h0 : purchase_price = 150)
  (h1 : markup_rate = 0.5)
  (h2 : discount_rate = 0.2)
  (h3 : initial_selling_price = purchase_price + markup_rate * initial_selling_price)
  (h4 : final_selling_price = initial_selling_price - discount_rate * initial_selling_price)
  (h5 : gross_profit = final_selling_price - purchase_price) :
  gross_profit = 90 :=
sorry

end dealer_gross_profit_l2335_233599


namespace solve_for_x_l2335_233508

theorem solve_for_x (x : ℝ) (h : (x - 5)^3 = (1 / 27)⁻¹) : x = 8 :=
sorry

end solve_for_x_l2335_233508


namespace endpoints_undetermined_l2335_233505

theorem endpoints_undetermined (m : ℝ → ℝ) :
  (∀ x, m x = x - 2) ∧ (∃ mid : ℝ × ℝ, ∃ (x1 x2 y1 y2 : ℝ), 
    mid = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
    m mid.1 = mid.2) → 
  ¬ (∃ (x1 x2 y1 y2 : ℝ), mid = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
    m ((x1 + x2) / 2) = (y1 + y2) / 2 ∧
    x1 = the_exact_endpoint ∧ x2 = the_exact_other_endpoint) :=
by sorry

end endpoints_undetermined_l2335_233505


namespace roots_of_equation_l2335_233531

theorem roots_of_equation :
  (∃ x, (18 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = 3 ∨ x = -4.5)) :=
by
  sorry

end roots_of_equation_l2335_233531


namespace jaymee_is_22_l2335_233503

-- Definitions based on the problem conditions
def shara_age : ℕ := 10
def jaymee_age : ℕ := 2 + 2 * shara_age

-- The theorem we need to prove
theorem jaymee_is_22 : jaymee_age = 22 :=
by
  sorry

end jaymee_is_22_l2335_233503


namespace solution_set_of_f_gt_0_range_of_m_l2335_233561

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) - abs (x + 2)

theorem solution_set_of_f_gt_0 :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1 / 3} ∪ {x | x > 3} :=
by sorry

theorem range_of_m (m : ℝ) :
  (∃ x_0 : ℝ, f x_0 + 2 * m^2 < 4 * m) ↔ -1 / 2 < m ∧ m < 5 / 2 :=
by sorry

end solution_set_of_f_gt_0_range_of_m_l2335_233561


namespace negation_of_every_planet_orbits_the_sun_l2335_233534

variables (Planet : Type) (orbits_sun : Planet → Prop)

theorem negation_of_every_planet_orbits_the_sun :
  (¬ ∀ x : Planet, (¬ (¬ (exists x : Planet, true)) → orbits_sun x)) ↔
  ∃ x : Planet, ¬ orbits_sun x :=
by sorry

end negation_of_every_planet_orbits_the_sun_l2335_233534


namespace two_colonies_limit_l2335_233588

def doubles_each_day (size: ℕ) (day: ℕ) : ℕ := size * 2 ^ day

theorem two_colonies_limit (habitat_limit: ℕ) (initial_size: ℕ) : 
  (∀ t, doubles_each_day initial_size t = habitat_limit → t = 20) → 
  initial_size > 0 →
  ∀ t, doubles_each_day (2 * initial_size) t = habitat_limit → t = 20 :=
by
  sorry

end two_colonies_limit_l2335_233588


namespace jennifer_spent_124_dollars_l2335_233568

theorem jennifer_spent_124_dollars 
  (initial_cans : ℕ := 40)
  (cans_per_set : ℕ := 5)
  (additional_cans_per_set : ℕ := 6)
  (total_cans_mark : ℕ := 30)
  (price_per_can_whole : ℕ := 2)
  (discount_threshold_whole : ℕ := 10)
  (discount_amount_whole : ℕ := 4) : 
  (initial_cans + additional_cans_per_set * (total_cans_mark / cans_per_set)) * price_per_can_whole - 
  (discount_amount_whole * ((initial_cans + additional_cans_per_set * (total_cans_mark / cans_per_set)) / discount_threshold_whole)) = 124 := by
  sorry

end jennifer_spent_124_dollars_l2335_233568


namespace area_of_quadrilateral_l2335_233556

noncomputable def quadrilateral_area
  (AB CD r : ℝ) (k : ℝ) 
  (h_perpendicular : AB * CD = 0)
  (h_equal_diameters : AB = 2 * r ∧ CD = 2 * r)
  (h_ratio : BC / AD = k) : ℝ := 
  (3 * r^2 * abs (1 - k^2)) / (1 + k^2)

theorem area_of_quadrilateral
  (AB CD r : ℝ) (k : ℝ)
  (h_perpendicular : AB * CD = 0)
  (h_equal_diameters : AB = 2 * r ∧ CD = 2 * r)
  (h_ratio : BC / AD = k) :
  quadrilateral_area AB CD r k h_perpendicular h_equal_diameters h_ratio = (3 * r^2 * abs (1 - k^2)) / (1 + k^2) :=
sorry

end area_of_quadrilateral_l2335_233556


namespace caitlin_age_l2335_233550

theorem caitlin_age (aunt_anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ) 
  (h1 : aunt_anna_age = 60)
  (h2 : brianna_age = aunt_anna_age / 3)
  (h3 : caitlin_age = brianna_age - 7)
  : caitlin_age = 13 :=
by
  sorry

end caitlin_age_l2335_233550


namespace total_area_of_field_l2335_233565

noncomputable def total_field_area (A1 A2 : ℝ) : ℝ := A1 + A2

theorem total_area_of_field :
  ∀ (A1 A2 : ℝ),
    A1 = 405 ∧ (A2 - A1 = (1/5) * ((A1 + A2) / 2)) →
    total_field_area A1 A2 = 900 :=
by
  intros A1 A2 h
  sorry

end total_area_of_field_l2335_233565


namespace peter_age_problem_l2335_233564

theorem peter_age_problem
  (P J : ℕ) 
  (h1 : J = P + 12)
  (h2 : P - 10 = 1/3 * (J - 10)) : P = 16 :=
sorry

end peter_age_problem_l2335_233564


namespace gcd_polynomials_l2335_233552

def P (n : ℤ) : ℤ := n^3 - 6 * n^2 + 11 * n - 6
def Q (n : ℤ) : ℤ := n^2 - 4 * n + 4

theorem gcd_polynomials (n : ℤ) (h : n ≥ 3) : Int.gcd (P n) (Q n) = n - 2 :=
by
  sorry

end gcd_polynomials_l2335_233552


namespace contrapositive_example_l2335_233545

theorem contrapositive_example 
  (x y : ℝ) (h : x^2 + y^2 = 0 → x = 0 ∧ y = 0) : 
  (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0 :=
sorry

end contrapositive_example_l2335_233545


namespace find_other_endpoint_l2335_233504

theorem find_other_endpoint (x1 y1 x2 y2 xm ym : ℝ)
  (midpoint_formula_x : xm = (x1 + x2) / 2)
  (midpoint_formula_y : ym = (y1 + y2) / 2)
  (h_midpoint : xm = -3 ∧ ym = 2)
  (h_endpoint : x1 = -7 ∧ y1 = 6) :
  x2 = 1 ∧ y2 = -2 := 
sorry

end find_other_endpoint_l2335_233504


namespace difference_in_number_of_girls_and_boys_l2335_233515

def ratio_boys_girls (b g : ℕ) : Prop := b * 3 = g * 2

def total_students (b g : ℕ) : Prop := b + g = 30

theorem difference_in_number_of_girls_and_boys
  (b g : ℕ)
  (h1 : ratio_boys_girls b g)
  (h2 : total_students b g) :
  g - b = 6 :=
sorry

end difference_in_number_of_girls_and_boys_l2335_233515


namespace there_are_six_bases_ending_in_one_for_625_in_decimal_l2335_233573

theorem there_are_six_bases_ending_in_one_for_625_in_decimal :
  (∃ ls : List ℕ, ls = [2, 3, 4, 6, 8, 12] ∧ ∀ b ∈ ls, 2 ≤ b ∧ b ≤ 12 ∧ 624 % b = 0 ∧ List.length ls = 6) :=
by
  sorry

end there_are_six_bases_ending_in_one_for_625_in_decimal_l2335_233573


namespace cake_sharing_l2335_233507

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l2335_233507


namespace eq_three_div_x_one_of_eq_l2335_233546

theorem eq_three_div_x_one_of_eq (x : ℝ) (hx : 1 - 6 / x + 9 / (x ^ 2) = 0) : (3 / x) = 1 :=
sorry

end eq_three_div_x_one_of_eq_l2335_233546


namespace find_k_value_l2335_233577

variable (S : ℕ → ℤ) (n : ℕ)

-- Conditions
def is_arithmetic_sum (S : ℕ → ℤ) : Prop :=
  ∃ (a d : ℤ), ∀ n : ℕ, S n = n * (2 * a + (n - 1) * d) / 2

axiom S3_eq_S8 (S : ℕ → ℤ) (hS : is_arithmetic_sum S) : S 3 = S 8
axiom Sk_eq_S7 (S : ℕ → ℤ) (k : ℕ) (hS: is_arithmetic_sum S)  : S 7 = S k

theorem find_k_value (S : ℕ → ℤ) (hS: is_arithmetic_sum S) :  S 3 = S 8 → S 7 = S 4 :=
by
  sorry

end find_k_value_l2335_233577


namespace distance_to_cut_pyramid_l2335_233567

theorem distance_to_cut_pyramid (V A V1 : ℝ) (h1 : V > 0) (h2 : A > 0) :
  ∃ d : ℝ, d = (3 / A) * (V - (V^2 * (V - V1))^(1 / 3)) :=
by
  sorry

end distance_to_cut_pyramid_l2335_233567


namespace find_x_l2335_233544

-- Given condition
def condition (x : ℝ) : Prop := 3 * x - 5 * x + 8 * x = 240

-- Statement (problem to prove)
theorem find_x (x : ℝ) (h : condition x) : x = 40 :=
by 
  sorry

end find_x_l2335_233544


namespace geom_seq_common_ratio_l2335_233535

noncomputable def log_custom_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem geom_seq_common_ratio (a : ℝ) :
  let u₁ := a + log_custom_base 2 3
  let u₂ := a + log_custom_base 4 3
  let u₃ := a + log_custom_base 8 3
  u₂ / u₁ = u₃ / u₂ →
  u₂ / u₁ = 1 / 3 :=
by
  intro h
  sorry

end geom_seq_common_ratio_l2335_233535


namespace find_a5_over_T9_l2335_233574

-- Define arithmetic sequences and their sums
variables {a_n : ℕ → ℚ} {b_n : ℕ → ℚ}
variables {S_n : ℕ → ℚ} {T_n : ℕ → ℚ}

-- Conditions
def arithmetic_seq_a (a_n : ℕ → ℚ) : Prop :=
  ∀ n, a_n n = a_n 1 + (n - 1) * (a_n 2 - a_n 1)

def arithmetic_seq_b (b_n : ℕ → ℚ) : Prop :=
  ∀ n, b_n n = b_n 1 + (n - 1) * (b_n 2 - b_n 1)

def sum_a (S_n : ℕ → ℚ) (a_n : ℕ → ℚ) : Prop :=
  ∀ n, S_n n = n * (a_n 1 + a_n n) / 2

def sum_b (T_n : ℕ → ℚ) (b_n : ℕ → ℚ) : Prop :=
  ∀ n, T_n n = n * (b_n 1 + b_n n) / 2

def given_condition (S_n : ℕ → ℚ) (T_n : ℕ → ℚ) : Prop :=
  ∀ n, S_n n / T_n n = (n + 3) / (2 * n - 1)

-- Goal statement
theorem find_a5_over_T9 (h_a : arithmetic_seq_a a_n) (h_b : arithmetic_seq_b b_n)
  (sum_a_S : sum_a S_n a_n) (sum_b_T : sum_b T_n b_n) (cond : given_condition S_n T_n) :
  a_n 5 / T_n 9 = 4 / 51 :=
  sorry

end find_a5_over_T9_l2335_233574


namespace functional_eq_implies_odd_l2335_233510

variable {f : ℝ → ℝ}

def functional_eq (f : ℝ → ℝ) :=
∀ a b, f (a + b) + f (a - b) = 2 * f a * Real.cos b

theorem functional_eq_implies_odd (h : functional_eq f) (hf_non_zero : ¬∀ x, f x = 0) : 
  ∀ x, f (-x) = -f x := 
by
  sorry

end functional_eq_implies_odd_l2335_233510


namespace quadratic_completion_l2335_233575

theorem quadratic_completion :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) ↔ ((x - 2)^2 = 3) :=
by
  sorry

end quadratic_completion_l2335_233575


namespace parabola_shift_l2335_233592

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2

-- Define the shift function for shifting the parabola right by 3 units
def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Define the shift function for shifting the parabola up by 4 units
def shift_up (f : ℝ → ℝ) (b : ℝ) (y : ℝ) : ℝ := y + b

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := shift_up (shift_right initial_parabola 3) 4 (initial_parabola x)

-- Goal: Prove that the transformed parabola is y = (x - 3)^2 + 4
theorem parabola_shift (x : ℝ) : transformed_parabola x = (x - 3)^2 + 4 := sorry

end parabola_shift_l2335_233592


namespace probability_both_selected_l2335_233518

theorem probability_both_selected (P_R : ℚ) (P_V : ℚ) (h1 : P_R = 3 / 7) (h2 : P_V = 1 / 5) :
  P_R * P_V = 3 / 35 :=
by {
  sorry
}

end probability_both_selected_l2335_233518


namespace ball_box_distribution_l2335_233578

theorem ball_box_distribution:
  ∃ (C : ℕ → ℕ → ℕ) (A : ℕ → ℕ → ℕ),
  C 4 2 * A 3 3 = sorry := 
by sorry

end ball_box_distribution_l2335_233578


namespace least_k_divisible_480_l2335_233547

theorem least_k_divisible_480 (k : ℕ) (h : k^4 % 480 = 0) : k = 101250 :=
sorry

end least_k_divisible_480_l2335_233547


namespace perfect_square_trinomial_l2335_233585

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ (x : ℝ), 4 * x^2 + b * x + 1 = (2 * x + 1) ^ 2) ↔ (b = 4 ∨ b = -4) := 
by 
  sorry

end perfect_square_trinomial_l2335_233585


namespace eq_fractions_l2335_233582

theorem eq_fractions : 
  (1 + 1 / (1 + 1 / (1 + 1 / 2))) = 8 / 5 := 
  sorry

end eq_fractions_l2335_233582


namespace part1_part2_l2335_233500

variable (a b : ℝ) (x : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) : (a^2 / b) + (b^2 / a) ≥ a + b :=
sorry

theorem part2 (h3 : 0 < x) (h4 : x < 1) : 
(∀ y : ℝ, y = ((1 - x)^2 / x) + (x^2 / (1 - x)) → y ≥ 1) ∧ ((1 - x) = x → y = 1) :=
sorry

end part1_part2_l2335_233500


namespace tangent_triangle_area_l2335_233579

noncomputable def area_of_tangent_triangle : ℝ :=
  let f : ℝ → ℝ := fun x => Real.log x
  let f' : ℝ → ℝ := fun x => 1 / x
  let tangent_line : ℝ → ℝ := fun x => x - 1
  let x_intercept : ℝ := 1
  let y_intercept : ℝ := -1
  let base := 1
  let height := 1
  (1 / 2) * base * height

theorem tangent_triangle_area :
  area_of_tangent_triangle = 1 / 2 :=
sorry

end tangent_triangle_area_l2335_233579


namespace denomination_of_second_note_l2335_233591

theorem denomination_of_second_note
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)
  (h1 : x = y)
  (h2 : y = z)
  (h3 : x + y + z = 75)
  (h4 : 1 * x + y * x + 10 * x = 400):
  y = 5 := by
  sorry

end denomination_of_second_note_l2335_233591


namespace marika_father_age_twice_l2335_233580

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end marika_father_age_twice_l2335_233580


namespace white_balls_count_l2335_233566

theorem white_balls_count {T W : ℕ} (h1 : 3 * 4 = T) (h2 : T - 3 = W) : W = 9 :=
by 
    sorry

end white_balls_count_l2335_233566


namespace theta_in_third_or_fourth_quadrant_l2335_233513

-- Define the conditions as Lean definitions
def theta_condition (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = k * Real.pi + (-1 : ℝ)^(k + 1) * (Real.pi / 4)

-- Formulate the statement we need to prove
theorem theta_in_third_or_fourth_quadrant (θ : ℝ) (h : theta_condition θ) :
  ∃ q : ℤ, q = 3 ∨ q = 4 :=
sorry

end theta_in_third_or_fourth_quadrant_l2335_233513


namespace complex_exponential_sum_angle_l2335_233514

theorem complex_exponential_sum_angle :
  ∃ r : ℝ, r ≥ 0 ∧ (e^(Complex.I * 11 * Real.pi / 60) + 
                     e^(Complex.I * 21 * Real.pi / 60) + 
                     e^(Complex.I * 31 * Real.pi / 60) + 
                     e^(Complex.I * 41 * Real.pi / 60) + 
                     e^(Complex.I * 51 * Real.pi / 60) = r * Complex.exp (Complex.I * 31 * Real.pi / 60)) := 
by
  sorry

end complex_exponential_sum_angle_l2335_233514


namespace rain_difference_l2335_233590

theorem rain_difference
    (rain_monday : ℕ → ℝ)
    (rain_tuesday : ℕ → ℝ)
    (rain_wednesday : ℕ → ℝ)
    (rain_thursday : ℕ → ℝ)
    (h_monday : ∀ n : ℕ, n = 10 → rain_monday n = 1.25)
    (h_tuesday : ∀ n : ℕ, n = 12 → rain_tuesday n = 2.15)
    (h_wednesday : ∀ n : ℕ, n = 8 → rain_wednesday n = 1.60)
    (h_thursday : ∀ n : ℕ, n = 6 → rain_thursday n = 2.80) :
    let total_rain_monday := 10 * 1.25
    let total_rain_tuesday := 12 * 2.15
    let total_rain_wednesday := 8 * 1.60
    let total_rain_thursday := 6 * 2.80
    (total_rain_tuesday + total_rain_thursday) - (total_rain_monday + total_rain_wednesday) = 17.3 :=
by
  sorry

end rain_difference_l2335_233590


namespace people_between_katya_and_polina_l2335_233558

-- Definitions based on given conditions
def is_next_to (a b : ℕ) : Prop := (b = a + 1) ∨ (b = a - 1)
def position_alena : ℕ := 1
def position_lena : ℕ := 5
def position_sveta (pos_sveta : ℕ) : Prop := pos_sveta + 1 = position_lena
def position_katya (pos_katya : ℕ) : Prop := pos_katya = 3
def position_polina (pos_polina : ℕ) : Prop := (is_next_to position_alena pos_polina)

-- The question: prove the number of people between Katya and Polina is 0
theorem people_between_katya_and_polina : 
  ∃ (pos_katya pos_polina : ℕ),
    position_katya pos_katya ∧ 
    position_polina pos_polina ∧ 
    pos_polina + 1 = pos_katya ∧
    pos_katya = 3 ∧ pos_polina = 2 := 
sorry

end people_between_katya_and_polina_l2335_233558


namespace test_point_selection_0618_method_l2335_233516

theorem test_point_selection_0618_method :
  ∀ (x1 x2 x3 : ℝ),
    1000 + 0.618 * (2000 - 1000) = x1 →
    1000 + (2000 - x1) = x2 →
    x2 < x1 →
    (∀ (f : ℝ → ℝ), f x2 < f x1) →
    x1 + (1000 - x2) = x3 →
    x3 = 1236 :=
by
  intros x1 x2 x3 h1 h2 h3 h4 h5
  sorry

end test_point_selection_0618_method_l2335_233516


namespace v_is_82_875_percent_of_z_l2335_233571

theorem v_is_82_875_percent_of_z (x y z w v : ℝ) 
  (h1 : x = 1.30 * y)
  (h2 : y = 0.60 * z)
  (h3 : w = 1.25 * x)
  (h4 : v = 0.85 * w) : 
  v = 0.82875 * z :=
by
  sorry

end v_is_82_875_percent_of_z_l2335_233571


namespace smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50_l2335_233539

def is_not_prime (n : ℕ) : Prop := ¬ Prime n

def is_not_square (n : ℕ) : Prop := ∀ m : ℕ, m * m ≠ n

def no_prime_factor_less_than_50 (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → p ≥ 50

theorem smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50 :
  (∃ n : ℕ, 0 < n ∧ is_not_prime n ∧ is_not_square n ∧ no_prime_factor_less_than_50 n ∧
  (∀ m : ℕ, 0 < m ∧ is_not_prime m ∧ is_not_square m ∧ no_prime_factor_less_than_50 m → n ≤ m)) →
  ∃ n : ℕ, n = 3127 :=
by {
  sorry
}

end smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50_l2335_233539


namespace sum_of_digits_product_is_13_l2335_233553

def base_eight_to_base_ten (n : ℕ) : ℕ := sorry
def product_base_eight (n1 n2 : ℕ) : ℕ := sorry
def digits_sum_base_ten (n : ℕ) : ℕ := sorry

theorem sum_of_digits_product_is_13 :
  let N1 := base_eight_to_base_ten 35
  let N2 := base_eight_to_base_ten 42
  let product := product_base_eight N1 N2
  digits_sum_base_ten product = 13 :=
by
  sorry

end sum_of_digits_product_is_13_l2335_233553


namespace find_geometric_sequence_values_l2335_233537

structure GeometricSequence (a b c d : ℝ) : Prop where
  ratio1 : b / a = c / b
  ratio2 : c / b = d / c

theorem find_geometric_sequence_values (x u v y : ℝ)
    (h1 : x + y = 20)
    (h2 : u + v = 34)
    (h3 : x^2 + u^2 + v^2 + y^2 = 1300) :
    (GeometricSequence x u v y ∧ ((x = 16 ∧ u = 4 ∧ v = 32 ∧ y = 2) ∨ (x = 4 ∧ u = 16 ∧ v = 2 ∧ y = 32))) :=
by
  sorry

end find_geometric_sequence_values_l2335_233537


namespace fraction_of_paper_per_book_l2335_233549

theorem fraction_of_paper_per_book (total_fraction_used : ℚ) (num_books : ℕ) (h1 : total_fraction_used = 5 / 8) (h2 : num_books = 5) : 
  (total_fraction_used / num_books) = 1 / 8 :=
by
  sorry

end fraction_of_paper_per_book_l2335_233549


namespace ellipse_foci_distance_l2335_233594

noncomputable def distance_between_foci_of_ellipse (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), (a = 5) → (b = 2) →
  distance_between_foci_of_ellipse a b = Real.sqrt 21 :=
by
  intros a b ha hb
  rw [ha, hb]
  -- The rest of the proof is omitted
  sorry

end ellipse_foci_distance_l2335_233594


namespace distribution_problem_l2335_233527

theorem distribution_problem (cards friends : ℕ) (h1 : cards = 7) (h2 : friends = 9) :
  (Nat.choose friends cards) * (Nat.factorial cards) = 181440 :=
by
  -- According to the combination formula and factorial definition
  -- We can insert specific values and calculations here, but as per the task requirements, 
  -- we are skipping the actual proof.
  sorry

end distribution_problem_l2335_233527


namespace length_of_train_l2335_233584

-- Definitions of given conditions
def train_speed (kmh : ℤ) := 25
def man_speed (kmh : ℤ) := 2
def crossing_time (sec : ℤ) := 28

-- Relative speed calculation (in meters per second)
def relative_speed := (train_speed 1 + man_speed 1) * (5 / 18 : ℚ)

-- Distance calculation (in meters)
def distance_covered := relative_speed * (crossing_time 1 : ℚ)

-- The theorem statement: Length of the train equals distance covered in crossing time
theorem length_of_train : distance_covered = 210 := by
  sorry

end length_of_train_l2335_233584


namespace original_class_strength_l2335_233528

theorem original_class_strength 
  (orig_avg_age : ℕ) (new_students_num : ℕ) (new_avg_age : ℕ) 
  (avg_age_decrease : ℕ) (orig_strength : ℕ) :
  orig_avg_age = 40 →
  new_students_num = 12 →
  new_avg_age = 32 →
  avg_age_decrease = 4 →
  (orig_strength + new_students_num) * (orig_avg_age - avg_age_decrease) = orig_strength * orig_avg_age + new_students_num * new_avg_age →
  orig_strength = 12 := 
by
  intros
  sorry

end original_class_strength_l2335_233528


namespace philips_painting_total_l2335_233511

def total_paintings_after_days (daily_paintings : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + daily_paintings * days

theorem philips_painting_total (daily_paintings initial_paintings days : ℕ) 
  (h1 : daily_paintings = 2) (h2 : initial_paintings = 20) (h3 : days = 30) : 
  total_paintings_after_days daily_paintings initial_paintings days = 80 := 
by
  sorry

end philips_painting_total_l2335_233511


namespace inequality_proof_l2335_233555

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (c + a)) * (1 + 4 * c / (a + b)) > 25 := by
  sorry

end inequality_proof_l2335_233555


namespace cevian_concurrency_l2335_233502

theorem cevian_concurrency
  (A B C Z X Y : ℝ)
  (a b c s : ℝ)
  (h1 : s = (a + b + c) / 2)
  (h2 : AZ = s - c) (h3 : ZB = s - b)
  (h4 : BX = s - a) (h5 : XC = s - c)
  (h6 : CY = s - b) (h7 : YA = s - a)
  : (AZ / ZB) * (BX / XC) * (CY / YA) = 1 :=
by
  sorry

end cevian_concurrency_l2335_233502


namespace boat_speed_still_water_l2335_233570

theorem boat_speed_still_water : 
  ∀ (b s : ℝ), (b + s = 11) → (b - s = 5) → b = 8 := 
by 
  intros b s h1 h2
  sorry

end boat_speed_still_water_l2335_233570


namespace incorrect_desc_is_C_l2335_233587
noncomputable def incorrect_geometric_solid_desc : Prop :=
  ¬ (∀ (plane_parallel: Prop), 
      plane_parallel ∧ 
      (∀ (frustum: Prop), frustum ↔ 
        (∃ (base section_cut cone : Prop), 
          cone ∧ 
          (section_cut = plane_parallel) ∧ 
          (frustum = (base ∧ section_cut)))))

theorem incorrect_desc_is_C (plane_parallel frustum base section_cut cone : Prop) :
  incorrect_geometric_solid_desc := 
by
  sorry

end incorrect_desc_is_C_l2335_233587


namespace distance_AB_polar_l2335_233598

open Real

/-- The distance between points A and B in polar coordinates, given that θ₁ - θ₂ = π. -/
theorem distance_AB_polar (A B : ℝ × ℝ) (r1 r2 : ℝ) (θ1 θ2 : ℝ) (hA : A = (r1, θ1)) (hB : B = (r2, θ2)) (hθ : θ1 - θ2 = π) :
  dist (r1 * cos θ1, r1 * sin θ1) (r2 * cos θ2, r2 * sin θ2) = r1 + r2 :=
sorry

end distance_AB_polar_l2335_233598


namespace speed_of_man_in_still_water_l2335_233576

variable (v_m v_s : ℝ)

theorem speed_of_man_in_still_water
  (h1 : (v_m + v_s) * 4 = 24)
  (h2 : (v_m - v_s) * 5 = 20) :
  v_m = 5 := 
sorry

end speed_of_man_in_still_water_l2335_233576


namespace greta_received_more_letters_l2335_233538

noncomputable def number_of_letters_difference : ℕ :=
  let B := 40
  let M (G : ℕ) := 2 * (G + B)
  let total (G : ℕ) := G + B + M G
  let G := 50 -- Solved from the total equation
  G - B

theorem greta_received_more_letters : number_of_letters_difference = 10 :=
by
  sorry

end greta_received_more_letters_l2335_233538


namespace length_of_CD_l2335_233541

theorem length_of_CD
    (AB BC AC AD CD : ℝ)
    (h1 : AB = 6)
    (h2 : BC = 1 / 2 * AB)
    (h3 : AC = AB + BC)
    (h4 : AD = AC)
    (h5 : CD = AD + AC) :
    CD = 18 := by
  sorry

end length_of_CD_l2335_233541


namespace min_value_fraction_l2335_233509

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 1) : 
  (1 / x) + (1 / (3 * y)) ≥ 3 :=
sorry

end min_value_fraction_l2335_233509


namespace interest_difference_l2335_233512

theorem interest_difference :
  let P := 10000
  let R := 6
  let T := 2
  let SI := P * R * T / 100
  let CI := P * (1 + R / 100)^T - P
  CI - SI = 36 :=
by
  let P := 10000
  let R := 6
  let T := 2
  let SI := P * R * T / 100
  let CI := P * (1 + R / 100)^T - P
  show CI - SI = 36
  sorry

end interest_difference_l2335_233512


namespace solve_inequality_l2335_233525

theorem solve_inequality {a b : ℝ} (h : -2 * a + 1 < -2 * b + 1) : a > b :=
by
  sorry

end solve_inequality_l2335_233525


namespace arithmetic_sequence_problem_l2335_233519

-- Define the arithmetic sequence and given properties
variable {a : ℕ → ℝ} -- an arithmetic sequence such that for all n, a_{n+1} - a_{n} is constant
variable (d : ℝ) (a1 : ℝ) -- common difference 'd' and first term 'a1'

-- Express the terms using the common difference 'd' and first term 'a1'
def a_n (n : ℕ) : ℝ := a1 + (n-1) * d

-- Given condition
axiom given_condition : a_n 3 + a_n 8 = 10

-- Proof goal
theorem arithmetic_sequence_problem : 3 * a_n 5 + a_n 7 = 20 :=
by
  -- Define the sequence in terms of common difference and the first term
  let a_n := fun n => a1 + (n-1) * d
  -- Simplify using the given condition
  sorry

end arithmetic_sequence_problem_l2335_233519


namespace cos_minus_sin_of_tan_eq_sqrt3_l2335_233526

theorem cos_minus_sin_of_tan_eq_sqrt3 (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = (Real.sqrt 3 - 1) / 2 := 
by
  sorry

end cos_minus_sin_of_tan_eq_sqrt3_l2335_233526


namespace sum_of_ages_l2335_233523

theorem sum_of_ages (P K : ℕ) (h1 : P - 7 = 3 * (K - 7)) (h2 : P + 2 = 2 * (K + 2)) : P + K = 50 :=
by
  sorry

end sum_of_ages_l2335_233523


namespace trajectory_of_point_inside_square_is_conic_or_degenerates_l2335_233597

noncomputable def is_conic_section (a : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ (m n l : ℝ) (x y : ℝ), 
    x = P.1 ∧ y = P.2 ∧ 
    (m^2 + n^2) * x^2 - 2 * n * (l + m) * x * y + (l^2 + n^2) * y^2 = (l * m - n^2)^2 ∧
    4 * n^2 * (l + m)^2 - 4 * (m^2 + n^2) * (l^2 + n^2) ≤ 0

theorem trajectory_of_point_inside_square_is_conic_or_degenerates
  (a : ℝ) (P : ℝ × ℝ)
  (h1 : 0 < P.1) (h2 : P.1 < 2 * a)
  (h3 : 0 < P.2) (h4 : P.2 < 2 * a)
  : is_conic_section a P :=
sorry

end trajectory_of_point_inside_square_is_conic_or_degenerates_l2335_233597


namespace min_value_inequality_l2335_233551

theorem min_value_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) ≥ 3 / 2 :=
sorry

end min_value_inequality_l2335_233551


namespace optimal_order_l2335_233560

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end optimal_order_l2335_233560


namespace willie_final_stickers_l2335_233548

-- Conditions
def willie_start_stickers : ℝ := 36.0
def emily_gives_willie : ℝ := 7.0

-- Theorem
theorem willie_final_stickers : willie_start_stickers + emily_gives_willie = 43.0 :=
by
  sorry

end willie_final_stickers_l2335_233548


namespace find_removed_number_l2335_233596

def list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def target_average : ℝ := 8.2

theorem find_removed_number (n : ℕ) (h : n ∈ list) :
  (list.sum - n) / (list.length - 1) = target_average -> n = 5 := by
  sorry

end find_removed_number_l2335_233596


namespace ratio_m_of_q_l2335_233521

theorem ratio_m_of_q
  (m n p q : ℚ)
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 12) :
  m / q = 3 / 4 := 
sorry

end ratio_m_of_q_l2335_233521


namespace solve_equation_l2335_233562

theorem solve_equation :
  ∃ x : ℝ, (x + 2) / 4 - (2 * x - 3) / 6 = 2 ∧ x = -12 :=
by
  sorry

end solve_equation_l2335_233562


namespace sell_price_equal_percentage_l2335_233542

theorem sell_price_equal_percentage (SP : ℝ) (CP : ℝ) :
  (SP - CP) / CP * 100 = (CP - 1280) / CP * 100 → 
  (1937.5 = CP + 0.25 * CP) → 
  SP = 1820 :=
by 
  -- Note: skip proof with sorry
  apply sorry

end sell_price_equal_percentage_l2335_233542


namespace celina_total_cost_l2335_233569

def hoodieCost : ℝ := 80
def hoodieTaxRate : ℝ := 0.05

def flashlightCost := 0.20 * hoodieCost
def flashlightTaxRate : ℝ := 0.10

def bootsInitialCost : ℝ := 110
def bootsDiscountRate : ℝ := 0.10
def bootsTaxRate : ℝ := 0.05

def waterFilterCost : ℝ := 65
def waterFilterDiscountRate : ℝ := 0.25
def waterFilterTaxRate : ℝ := 0.08

def campingMatCost : ℝ := 45
def campingMatDiscountRate : ℝ := 0.15
def campingMatTaxRate : ℝ := 0.08

def backpackCost : ℝ := 105
def backpackTaxRate : ℝ := 0.08

def totalCost : ℝ := 
  let hoodieTotal := (hoodieCost * (1 + hoodieTaxRate))
  let flashlightTotal := (flashlightCost * (1 + flashlightTaxRate))
  let bootsTotal := ((bootsInitialCost * (1 - bootsDiscountRate)) * (1 + bootsTaxRate))
  let waterFilterTotal := ((waterFilterCost * (1 - waterFilterDiscountRate)) * (1 + waterFilterTaxRate))
  let campingMatTotal := ((campingMatCost * (1 - campingMatDiscountRate)) * (1 + campingMatTaxRate))
  let backpackTotal := (backpackCost * (1 + backpackTaxRate))
  hoodieTotal + flashlightTotal + bootsTotal + waterFilterTotal + campingMatTotal + backpackTotal

theorem celina_total_cost: totalCost = 413.91 := by
  sorry

end celina_total_cost_l2335_233569


namespace negative_expression_l2335_233529

theorem negative_expression :
  -(-1) ≠ -1 ∧ (-1)^2 ≠ -1 ∧ |(-1)| ≠ -1 ∧ -|(-1)| = -1 :=
by
  sorry

end negative_expression_l2335_233529


namespace inequality_solution_l2335_233533

theorem inequality_solution (x : ℝ) : 3 * x^2 + 7 * x < 6 ↔ -3 < x ∧ x < 2 / 3 := 
sorry

end inequality_solution_l2335_233533


namespace f_4_1981_l2335_233524

-- Define the function f with its properties
axiom f : ℕ → ℕ → ℕ

axiom f_0_y (y : ℕ) : f 0 y = y + 1
axiom f_x1_0 (x : ℕ) : f (x + 1) 0 = f x 1
axiom f_x1_y1 (x y : ℕ) : f (x + 1) (y + 1) = f x (f (x + 1) y)

theorem f_4_1981 : f 4 1981 = 2 ^ 3964 - 3 :=
sorry

end f_4_1981_l2335_233524


namespace sum_of_coefficients_is_60_l2335_233532

theorem sum_of_coefficients_is_60 :
  ∀ (a b c d e : ℤ), (∀ x : ℤ, 512 * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e)) →
  a + b + c + d + e = 60 :=
by
  intros a b c d e h
  sorry

end sum_of_coefficients_is_60_l2335_233532


namespace K_time_expression_l2335_233530

variable (x : ℝ) 

theorem K_time_expression
  (hyp : (45 / (x - 2 / 5) - 45 / x = 3 / 4)) :
  45 / (x : ℝ) = 45 / x :=
sorry

end K_time_expression_l2335_233530


namespace rotate180_of_point_A_l2335_233572

-- Define the point A and the transformation
def point_A : ℝ × ℝ := (-3, 2)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement for the problem
theorem rotate180_of_point_A :
  rotate180 point_A = (3, -2) :=
sorry

end rotate180_of_point_A_l2335_233572


namespace Vasya_not_11_more_than_Kolya_l2335_233583

def is_L_shaped (n : ℕ) : Prop :=
  n % 2 = 1

def total_cells : ℕ :=
  14400

theorem Vasya_not_11_more_than_Kolya (k v : ℕ) :
  (is_L_shaped k) → (is_L_shaped v) → (k + v = total_cells) → (k % 2 = 0) → (v % 2 = 0) → (v - k ≠ 11) := 
by
  sorry

end Vasya_not_11_more_than_Kolya_l2335_233583


namespace find_a14_l2335_233536

-- Define the arithmetic sequence properties
def sum_of_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

def nth_term (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

theorem find_a14 (a1 d : ℤ) (S11 : sum_of_first_n_terms a1 d 11 = 55)
  (a10 : nth_term a1 d 10 = 9) : nth_term a1 d 14 = 13 :=
sorry

end find_a14_l2335_233536


namespace power_mod_equality_l2335_233563

theorem power_mod_equality (n : ℕ) : 
  (47 % 8 = 7) → (23 % 8 = 7) → (47 ^ 2500 - 23 ^ 2500) % 8 = 0 := 
by
  intro h1 h2
  sorry

end power_mod_equality_l2335_233563


namespace part1_part2_l2335_233517

def f (x : ℝ) : ℝ := x^2 - 1
def g (x a : ℝ) : ℝ := a * |x - 1|

theorem part1 (a : ℝ) : (∀ x : ℝ, |f x| = g x a → x = 1) → a < 0 :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ g x a) → a ≤ -2 :=
sorry

end part1_part2_l2335_233517


namespace bus_stop_minutes_per_hour_l2335_233543

/-- Given the average speed of a bus excluding stoppages is 60 km/hr
and including stoppages is 15 km/hr, prove that the bus stops for 45 minutes per hour. -/
theorem bus_stop_minutes_per_hour
  (speed_no_stops : ℝ := 60)
  (speed_with_stops : ℝ := 15) :
  ∃ t : ℝ, t = 45 :=
by
  sorry

end bus_stop_minutes_per_hour_l2335_233543


namespace min_cost_at_100_l2335_233593

noncomputable def cost_function (v : ℝ) : ℝ :=
if (0 < v ∧ v ≤ 50) then (123000 / v + 690)
else if (v > 50) then (3 * v^2 / 50 + 120000 / v + 600)
else 0

theorem min_cost_at_100 : ∃ v : ℝ, v = 100 ∧ cost_function v = 2400 :=
by
  -- We are not proving but stating the theorem here
  sorry

end min_cost_at_100_l2335_233593


namespace no_solutions_988_1991_l2335_233506

theorem no_solutions_988_1991 :
    ¬ ∃ (m n : ℤ),
      (988 ≤ m ∧ m ≤ 1991) ∧
      (988 ≤ n ∧ n ≤ 1991) ∧
      m ≠ n ∧
      ∃ (a b : ℤ), (mn + n = a^2 ∧ mn + m = b^2) := sorry

end no_solutions_988_1991_l2335_233506


namespace translated_function_is_correct_l2335_233501

-- Define the original function
def f (x : ℝ) : ℝ := (x - 2) ^ 2 + 2

-- Define the translated function after moving 1 unit to the left
def g (x : ℝ) : ℝ := f (x + 1)

-- Define the final function after moving 1 unit upward
def h (x : ℝ) : ℝ := g x + 1

-- The statement to be proved
theorem translated_function_is_correct :
  ∀ x : ℝ, h x = (x - 1) ^ 2 + 3 :=
by
  -- Proof goes here
  sorry

end translated_function_is_correct_l2335_233501


namespace psychologist_charge_difference_l2335_233589

variables (F A : ℝ)

theorem psychologist_charge_difference
  (h1 : F + 4 * A = 375)
  (h2 : F + A = 174) :
  (F - A) = 40 :=
by sorry

end psychologist_charge_difference_l2335_233589


namespace problem_l2335_233581

noncomputable def a : ℝ := Real.exp 1 - 2
noncomputable def b : ℝ := 1 - Real.log 2
noncomputable def c : ℝ := Real.exp (Real.exp 1) - Real.exp 2

theorem problem (a_def : a = Real.exp 1 - 2) 
                (b_def : b = 1 - Real.log 2) 
                (c_def : c = Real.exp (Real.exp 1) - Real.exp 2) : 
                c > a ∧ a > b := 
by 
  rw [a_def, b_def, c_def]
  sorry

end problem_l2335_233581


namespace abs_add_lt_abs_add_l2335_233522

open Real

theorem abs_add_lt_abs_add {a b : ℝ} (h : a * b < 0) : abs (a + b) < abs a + abs b := 
  sorry

end abs_add_lt_abs_add_l2335_233522


namespace troy_needs_additional_money_l2335_233586

-- Defining the initial conditions
def price_of_new_computer : ℕ := 80
def initial_savings : ℕ := 50
def money_from_selling_old_computer : ℕ := 20

-- Defining the question and expected answer
def required_additional_money : ℕ :=
  price_of_new_computer - (initial_savings + money_from_selling_old_computer)

-- The proof statement
theorem troy_needs_additional_money : required_additional_money = 10 := by
  sorry

end troy_needs_additional_money_l2335_233586


namespace probability_of_color_change_l2335_233559

def traffic_light_cycle := 90
def green_duration := 45
def yellow_duration := 5
def red_duration := 40
def green_to_yellow := green_duration
def yellow_to_red := green_duration + yellow_duration
def red_to_green := traffic_light_cycle
def observation_interval := 4
def valid_intervals := [green_to_yellow - observation_interval + 1, green_to_yellow, 
                        yellow_to_red - observation_interval + 1, yellow_to_red, 
                        red_to_green - observation_interval + 1, red_to_green]
def total_valid_intervals := valid_intervals.length * observation_interval

theorem probability_of_color_change : 
  (total_valid_intervals : ℚ) / traffic_light_cycle = 2 / 15 := 
by
  sorry

end probability_of_color_change_l2335_233559


namespace positive_integer_k_l2335_233520

theorem positive_integer_k (k x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = k * x * y * z) :
  k = 1 ∨ k = 3 :=
sorry

end positive_integer_k_l2335_233520


namespace Jessie_l2335_233554

theorem Jessie's_friends (total_muffins : ℕ) (muffins_per_person : ℕ) (num_people : ℕ) :
  total_muffins = 20 → muffins_per_person = 4 → num_people = total_muffins / muffins_per_person → num_people - 1 = 4 :=
by
  intros h1 h2 h3
  sorry

end Jessie_l2335_233554


namespace coordinates_after_5_seconds_l2335_233557

-- Define the initial coordinates of point P
def initial_coordinates : ℚ × ℚ := (-10, 10)

-- Define the velocity vector of point P
def velocity_vector : ℚ × ℚ := (4, -3)

-- Asserting the coordinates of point P after 5 seconds
theorem coordinates_after_5_seconds : 
   initial_coordinates + 5 • velocity_vector = (10, -5) :=
by 
  sorry

end coordinates_after_5_seconds_l2335_233557
