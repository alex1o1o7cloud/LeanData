import Mathlib

namespace max_area_of_rect_l19_19131

theorem max_area_of_rect (x y : ℝ) (h1 : x + y = 10) : 
  x * y ≤ 25 :=
by 
  sorry

end max_area_of_rect_l19_19131


namespace total_marbles_correct_l19_19603

-- Define the number of marbles Mary has
def MaryYellowMarbles := 9
def MaryBlueMarbles := 7
def MaryGreenMarbles := 6

-- Define the number of marbles Joan has
def JoanYellowMarbles := 3
def JoanBlueMarbles := 5
def JoanGreenMarbles := 4

-- Define the total number of marbles for Mary and Joan combined
def TotalMarbles := MaryYellowMarbles + MaryBlueMarbles + MaryGreenMarbles + JoanYellowMarbles + JoanBlueMarbles + JoanGreenMarbles

-- We want to prove that the total number of marbles is 34
theorem total_marbles_correct : TotalMarbles = 34 := by
  -- The proof is skipped with sorry
  sorry

end total_marbles_correct_l19_19603


namespace probability_green_then_blue_l19_19955

theorem probability_green_then_blue :
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := green_marbles / total_marbles
  let prob_second_blue := blue_marbles / (total_marbles - 1)
  prob_first_green * prob_second_blue = 4 / 15 :=
sorry

end probability_green_then_blue_l19_19955


namespace remainder_of_3_pow_2023_mod_5_l19_19832

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end remainder_of_3_pow_2023_mod_5_l19_19832


namespace max_tan_B_l19_19301

theorem max_tan_B (A B : ℝ) (C : Prop) 
  (sin_pos_A : 0 < Real.sin A) 
  (sin_pos_B : 0 < Real.sin B) 
  (angle_condition : Real.sin B / Real.sin A = Real.cos (A + B)) :
  Real.tan B ≤ Real.sqrt 2 / 4 :=
by
  sorry

end max_tan_B_l19_19301


namespace extremum_value_and_min_on_interval_l19_19017

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem extremum_value_and_min_on_interval
  (a b c : ℝ)
  (h1_eq : 12 * a + b = 0)
  (h2_eq : 4 * a + b = -8)
  (h_max : 16 + c = 28) :
  min (min (f a b c (-3)) (f a b c 3)) (f a b c 2) = -4 :=
by sorry

end extremum_value_and_min_on_interval_l19_19017


namespace simplify_expression_l19_19771

theorem simplify_expression (y : ℝ) : 
  2 * y * (4 * y^2 - 3 * y + 1) - 6 * (y^2 - 3 * y + 4) = 8 * y^3 - 12 * y^2 + 20 * y - 24 := 
by
  sorry

end simplify_expression_l19_19771


namespace bottles_not_placed_in_crate_l19_19649

-- Defining the constants based on the conditions
def bottles_per_crate : Nat := 12
def total_bottles : Nat := 130
def crates : Nat := 10

-- Theorem statement based on the question and the correct answer
theorem bottles_not_placed_in_crate :
  total_bottles - (bottles_per_crate * crates) = 10 :=
by
  -- Proof will be here
  sorry

end bottles_not_placed_in_crate_l19_19649


namespace solve_for_y_l19_19233

theorem solve_for_y (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 :=
  sorry

end solve_for_y_l19_19233


namespace dugu_team_prob_l19_19808

def game_prob (prob_win_first : ℝ) (prob_increase : ℝ) (prob_decrease : ℝ) : ℝ :=
  let p1 := prob_win_first
  let p2 := prob_win_first + prob_increase
  let p3 := prob_win_first + 2 * prob_increase
  let p4 := prob_win_first + 3 * prob_increase
  let p5 := prob_win_first + 4 * prob_increase
  let win_in_3 := p1 * p2 * p3
  let lose_first := (1 - prob_win_first)
  let win_then := prob_win_first
  let win_in_4a := lose_first * (prob_win_first - prob_decrease) * 
    prob_win_first * p2 * p3
  let win_in_4b := win_then * (1 - (prob_win_first + prob_increase)) *
    p2 * p3
  let win_in_4c := win_then * p2 * (1 - prob_win_first + prob_increase - 
    prob_decrease) * p4

  win_in_3 + win_in_4a + win_in_4b + win_in_4c

theorem dugu_team_prob : 
  game_prob 0.4 0.1 0.1 = 0.236 :=
by
  sorry

end dugu_team_prob_l19_19808


namespace cos_of_theta_l19_19608

theorem cos_of_theta
  (A : ℝ) (a : ℝ) (m : ℝ) (θ : ℝ) 
  (hA : A = 40) 
  (ha : a = 12) 
  (hm : m = 10) 
  (h_area: A = (1/2) * a * m * Real.sin θ) 
  : Real.cos θ = (Real.sqrt 5) / 3 :=
by
  sorry

end cos_of_theta_l19_19608


namespace find_q_value_l19_19936

theorem find_q_value 
  (p q r : ℕ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (hr : 0 < r) 
  (h : p + 1 / (q + 1 / r : ℚ) = 25 / 19) : 
  q = 3 :=
by 
  sorry

end find_q_value_l19_19936


namespace distance_foci_of_hyperbola_l19_19826

noncomputable def distance_between_foci : ℝ :=
  8 * Real.sqrt 5

theorem distance_foci_of_hyperbola :
  ∃ A B : ℝ, (9 * A^2 - 36 * A - B^2 + 4 * B = 40) → distance_between_foci = 8 * Real.sqrt 5 :=
sorry

end distance_foci_of_hyperbola_l19_19826


namespace find_monic_polynomial_of_shifted_roots_l19_19401

theorem find_monic_polynomial_of_shifted_roots (a b c : ℝ) (h : ∀ x : ℝ, (x - a) * (x - b) * (x - c) = x^3 - 5 * x + 7) : 
  (x : ℝ) → (x - (a - 3)) * (x - (b - 3)) * (x - (c - 3)) = x^3 + 9 * x^2 + 22 * x + 19 :=
by
  -- Proof will be provided here.
  sorry

end find_monic_polynomial_of_shifted_roots_l19_19401


namespace sqrt_product_l19_19599

theorem sqrt_product : (Real.sqrt 121) * (Real.sqrt 49) * (Real.sqrt 11) = 77 * (Real.sqrt 11) := by
  -- This is just the theorem statement as requested.
  sorry

end sqrt_product_l19_19599


namespace find_b_l19_19948

theorem find_b (a b c : ℝ) (A B C : ℝ) (h1 : a = 10) (h2 : c = 20) (h3 : B = 120) :
  b = 10 * Real.sqrt 7 :=
sorry

end find_b_l19_19948


namespace Nina_second_distance_l19_19911

theorem Nina_second_distance 
  (total_distance : ℝ) 
  (first_run : ℝ) 
  (second_same_run : ℝ)
  (run_twice : first_run = 0.08 ∧ second_same_run = 0.08)
  (total : total_distance = 0.83)
  : (total_distance - (first_run + second_same_run)) = 0.67 := by
  sorry

end Nina_second_distance_l19_19911


namespace cubs_more_home_runs_than_cardinals_l19_19986

theorem cubs_more_home_runs_than_cardinals 
(h1 : 2 + 1 + 2 = 5) 
(h2 : 1 + 1 = 2) : 
5 - 2 = 3 :=
by sorry

end cubs_more_home_runs_than_cardinals_l19_19986


namespace math_question_l19_19872

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l19_19872


namespace brittany_first_test_grade_l19_19084

theorem brittany_first_test_grade (x : ℤ) (h1 : (x + 84) / 2 = 81) : x = 78 :=
by
  sorry

end brittany_first_test_grade_l19_19084


namespace probability_of_sum_six_two_dice_l19_19231

noncomputable def probability_sum_six : ℚ := 5 / 36

theorem probability_of_sum_six_two_dice (dice_faces : ℕ := 6) : 
  ∃ (p : ℚ), p = probability_sum_six :=
by
  sorry

end probability_of_sum_six_two_dice_l19_19231


namespace smallest_triangle_perimeter_consecutive_even_l19_19700

theorem smallest_triangle_perimeter_consecutive_even :
  ∃ (a b c : ℕ), a = 2 ∧ b = 4 ∧ c = 6 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a + b + c = 12) :=
by {
  sorry
}

end smallest_triangle_perimeter_consecutive_even_l19_19700


namespace p_implies_q_l19_19834

def p (x : ℝ) := 0 < x ∧ x < 5
def q (x : ℝ) := -5 < x - 2 ∧ x - 2 < 5

theorem p_implies_q (x : ℝ) (h : p x) : q x :=
  by sorry

end p_implies_q_l19_19834


namespace campers_morning_count_l19_19514

theorem campers_morning_count (afternoon_count : ℕ) (additional_morning : ℕ) (h1 : afternoon_count = 39) (h2 : additional_morning = 5) :
  afternoon_count + additional_morning = 44 :=
by
  sorry

end campers_morning_count_l19_19514


namespace abcd_solution_l19_19690

-- Define the problem statement
theorem abcd_solution (a b c d : ℤ) (h1 : a + c = -2) (h2 : a * c + b + d = 3) (h3 : a * d + b * c = 4) (h4 : b * d = -10) : 
  a + b + c + d = 1 := by 
  sorry

end abcd_solution_l19_19690


namespace inequality_solution_l19_19032

theorem inequality_solution (x : ℝ) : 1 - (2 * x - 2) / 5 < (3 - 4 * x) / 2 → x < 1 / 16 := by
  sorry

end inequality_solution_l19_19032


namespace N_eq_M_union_P_l19_19375

open Set

def M : Set ℝ := { x | ∃ n : ℤ, x = n }
def N : Set ℝ := { x | ∃ n : ℤ, x = n / 2 }
def P : Set ℝ := { x | ∃ n : ℤ, x = n + 1/2 }

theorem N_eq_M_union_P : N = M ∪ P := 
sorry

end N_eq_M_union_P_l19_19375


namespace simplify_fraction_l19_19989

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l19_19989


namespace annual_interest_rate_l19_19090

theorem annual_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) 
  (hP : P = 700) 
  (hA : A = 771.75) 
  (hn : n = 2) 
  (ht : t = 1) 
  (h : A = P * (1 + r / n) ^ (n * t)) : 
  r = 0.10 := 
by 
  -- Proof steps go here
  sorry

end annual_interest_rate_l19_19090


namespace sqrt_abs_eq_zero_imp_power_eq_neg_one_l19_19429

theorem sqrt_abs_eq_zero_imp_power_eq_neg_one (m n : ℤ) (h : (Real.sqrt (m - 2) + abs (n + 3) = 0)) : (m + n) ^ 2023 = -1 := by
  sorry

end sqrt_abs_eq_zero_imp_power_eq_neg_one_l19_19429


namespace xiaohong_test_number_l19_19047

theorem xiaohong_test_number (x : ℕ) :
  (88 * x - 85 * (x - 1) = 100) → x = 5 :=
by
  intro h
  sorry

end xiaohong_test_number_l19_19047


namespace find_value_of_k_l19_19188

theorem find_value_of_k (k : ℤ) : 
  (2 + 3 * k * -1/3 = -7 * 4) → k = 30 := 
by
  sorry

end find_value_of_k_l19_19188


namespace depreciation_rate_l19_19214

theorem depreciation_rate (initial_value final_value : ℝ) (years : ℕ) (r : ℝ)
  (h_initial : initial_value = 128000)
  (h_final : final_value = 54000)
  (h_years : years = 3)
  (h_equation : final_value = initial_value * (1 - r) ^ years) :
  r = 0.247 :=
sorry

end depreciation_rate_l19_19214


namespace domain_all_real_iff_l19_19343

theorem domain_all_real_iff (k : ℝ) :
  (∀ x : ℝ, -3 * x ^ 2 - x + k ≠ 0 ) ↔ k < -1 / 12 :=
by
  sorry

end domain_all_real_iff_l19_19343


namespace hyperbola_asymptote_value_l19_19044

theorem hyperbola_asymptote_value {b : ℝ} (h : b > 0) 
  (asymptote_eq : ∀ x : ℝ, y = x * (1 / 2) ∨ y = -x * (1 / 2)) :
  b = 1 :=
sorry

end hyperbola_asymptote_value_l19_19044


namespace system_solution_is_unique_l19_19895

theorem system_solution_is_unique
  (a b : ℝ)
  (h1 : 2 - a * 5 = -1)
  (h2 : b + 3 * 5 = 8) :
  (∃ m n : ℝ, 2 * (m + n) - a * (m - n) = -1 ∧ b * (m + n) + 3 * (m - n) = 8 ∧ m = 3 ∧ n = -2) :=
by
  sorry

end system_solution_is_unique_l19_19895


namespace limit_of_R_l19_19314

noncomputable def R (m b : ℝ) : ℝ :=
  let x := ((-b) + Real.sqrt (b^2 + 4 * m)) / 2
  m * x + 3 

theorem limit_of_R (b : ℝ) (hb : b ≠ 0) : 
  (∀ m : ℝ, m < 3) → 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 0) < δ → abs ((R x (-b) - R x b) / x - b) < ε) :=
by
  sorry

end limit_of_R_l19_19314


namespace third_side_length_l19_19842

noncomputable def calc_third_side (a b : ℕ) (hypotenuse : Bool) : ℝ :=
if hypotenuse then
  Real.sqrt (a^2 + b^2)
else
  Real.sqrt (abs (a^2 - b^2))

theorem third_side_length (a b : ℕ) (h_right_triangle : (a = 8 ∧ b = 15)) :
  calc_third_side a b true = 17 ∨ calc_third_side 15 8 false = Real.sqrt 161 :=
by {
  sorry
}

end third_side_length_l19_19842


namespace parabola_equation_l19_19933

noncomputable def parabola_vertex_form (x y a : ℝ) : Prop := y = a * (x - 3)^2 + 5

noncomputable def parabola_standard_form (x y : ℝ) : Prop := y = -3 * x^2 + 18 * x - 22

theorem parabola_equation (a : ℝ) (h_vertex : parabola_vertex_form 3 5 a) (h_point : parabola_vertex_form 2 2 a) :
  ∃ x y, parabola_standard_form x y :=
by
  sorry

end parabola_equation_l19_19933


namespace no_solutions_l19_19021

theorem no_solutions : ¬ ∃ x : ℝ, (6 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 4) := by
  sorry

end no_solutions_l19_19021


namespace triangle_area_eq_l19_19369

noncomputable def areaOfTriangle (a b c A B C: ℝ): ℝ :=
1 / 2 * a * c * (Real.sin A)

theorem triangle_area_eq
  (a b c A B C : ℝ)
  (h1 : a = 2)
  (h2 : A = Real.pi / 3)
  (h3 : Real.sqrt 3 / 2 - Real.sin (B - C) = Real.sin (2 * B)) :
  areaOfTriangle a b c A B C = Real.sqrt 3 ∨ areaOfTriangle a b c A B C = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end triangle_area_eq_l19_19369


namespace fraction_female_attendees_on_time_l19_19841

theorem fraction_female_attendees_on_time (A : ℝ) (h1 : A > 0) :
  let males_fraction := 3/5
  let males_on_time := 7/8
  let not_on_time := 0.155
  let total_on_time_fraction := 1 - not_on_time
  let males := males_fraction * A
  let males_arrived_on_time := males_on_time * males
  let females := (1 - males_fraction) * A
  let females_arrived_on_time_fraction := (total_on_time_fraction * A - males_arrived_on_time) / females
  females_arrived_on_time_fraction = 4/5 :=
by
  sorry

end fraction_female_attendees_on_time_l19_19841


namespace find_optimal_price_and_units_l19_19556

noncomputable def price_and_units (x : ℝ) : Prop := 
  let cost_price := 40
  let initial_units := 500
  let profit_goal := 8000
  50 ≤ x ∧ x ≤ 70 ∧ (x - cost_price) * (initial_units - 10 * (x - 50)) = profit_goal

theorem find_optimal_price_and_units : 
  ∃ x units, price_and_units x ∧ units = 500 - 10 * (x - 50) ∧ x = 60 ∧ units = 400 := 
sorry

end find_optimal_price_and_units_l19_19556


namespace total_elephants_in_two_parks_is_280_l19_19973

def number_of_elephants_we_preserve_for_future : ℕ := 70
def multiple_factor : ℕ := 3

def number_of_elephants_gestures_for_good : ℕ := multiple_factor * number_of_elephants_we_preserve_for_future

def total_number_of_elephants : ℕ := number_of_elephants_we_preserve_for_future + number_of_elephants_gestures_for_good

theorem total_elephants_in_two_parks_is_280 : total_number_of_elephants = 280 :=
by
  sorry

end total_elephants_in_two_parks_is_280_l19_19973


namespace student_arrangement_l19_19061

theorem student_arrangement (students : Fin 6 → Prop)
  (A : (students 0) ∨ (students 5) → False)
  (females_adj : ∃ (i : Fin 6), i < 5 ∧ students i → students (i + 1))
  : ∃! n, n = 96 := by
  sorry

end student_arrangement_l19_19061


namespace gcd_lcm_mul_l19_19194

theorem gcd_lcm_mul (a b : ℤ) : (Int.gcd a b) * (Int.lcm a b) = a * b := by
  sorry

end gcd_lcm_mul_l19_19194


namespace john_tanks_needed_l19_19328

theorem john_tanks_needed 
  (num_balloons : ℕ) 
  (volume_per_balloon : ℕ) 
  (volume_per_tank : ℕ) 
  (H1 : num_balloons = 1000) 
  (H2 : volume_per_balloon = 10) 
  (H3 : volume_per_tank = 500) 
: (num_balloons * volume_per_balloon) / volume_per_tank = 20 := 
by 
  sorry

end john_tanks_needed_l19_19328


namespace arrangement_valid_l19_19894

def unique_digits (a b c d e f : Nat) : Prop :=
  (a = 4) ∧ (b = 1) ∧ (c = 2) ∧ (d = 5) ∧ (e = 6) ∧ (f = 3)

def sum_15 (x y z : Nat) : Prop :=
  x + y + z = 15

theorem arrangement_valid :
  ∃ a b c d e f : Nat, unique_digits a b c d e f ∧
  sum_15 a d e ∧
  sum_15 d b f ∧
  sum_15 f e c ∧
  sum_15 a b c ∧
  sum_15 a e f ∧
  sum_15 b d c :=
sorry

end arrangement_valid_l19_19894


namespace circle_diameter_mn_origin_l19_19664

-- Definitions based on conditions in (a)
def circle_equation (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y + m = 0
def line_equation (x y : ℝ) : Prop := x + 2 * y - 4 = 0
def orthogonal (x1 x2 y1 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem to prove (based on conditions and correct answer in (b))
theorem circle_diameter_mn_origin 
  (m : ℝ) 
  (x1 y1 x2 y2 : ℝ)
  (h1: circle_equation m x1 y1) 
  (h2: circle_equation m x2 y2)
  (h3: line_equation x1 y1)
  (h4: line_equation x2 y2)
  (h5: orthogonal x1 x2 y1 y2) :
  m = 8 / 5 := 
sorry

end circle_diameter_mn_origin_l19_19664


namespace range_of_c_l19_19023

noncomputable def p (c : ℝ) : Prop := ∀ x : ℝ, (2 * c - 1) ^ x = (2 * c - 1) ^ x

def q (c : ℝ) : Prop := ∀ x : ℝ, x + |x - 2 * c| > 1

theorem range_of_c (c : ℝ) (h1 : c > 0)
  (h2 : p c ∨ q c) (h3 : ¬ (p c ∧ q c)) : c ≥ 1 :=
sorry

end range_of_c_l19_19023


namespace equidistant_divisors_multiple_of_6_l19_19981

open Nat

theorem equidistant_divisors_multiple_of_6 (n : ℕ) :
  (∃ a b : ℕ, a ≠ b ∧ a ∣ n ∧ b ∣ n ∧ 
    (a + b = 2 * (n / 3))) → 
  (∃ k : ℕ, n = 6 * k) := 
by
  sorry

end equidistant_divisors_multiple_of_6_l19_19981


namespace R2_area_is_160_l19_19947

-- Define the initial conditions.
structure Rectangle :=
(width : ℝ)
(height : ℝ)

def R1 : Rectangle := { width := 4, height := 8 }

def similar (r1 r2 : Rectangle) : Prop :=
  r2.width / r2.height = r1.width / r1.height

def R2_diagonal := 20

-- Proving that the area of R2 is 160 square inches
theorem R2_area_is_160 (R2 : Rectangle)
  (h_similar : similar R1 R2)
  (h_diagonal : R2.width^2 + R2.height^2 = R2_diagonal^2) :
  R2.width * R2.height = 160 :=
  sorry

end R2_area_is_160_l19_19947


namespace compare_abc_l19_19761

noncomputable def a : ℝ := Real.exp (Real.sqrt Real.pi)
noncomputable def b : ℝ := Real.sqrt Real.pi + 1
noncomputable def c : ℝ := (Real.log Real.pi) / Real.exp 1 + 2

theorem compare_abc : c < b ∧ b < a := by
  sorry

end compare_abc_l19_19761


namespace percentage_above_wholesale_correct_l19_19527

variable (wholesale_cost retail_cost employee_payment : ℝ)
variable (employee_discount percentage_above_wholesale : ℝ)

theorem percentage_above_wholesale_correct :
  wholesale_cost = 200 → 
  employee_discount = 0.25 → 
  employee_payment = 180 → 
  retail_cost = wholesale_cost + (percentage_above_wholesale / 100) * wholesale_cost →
  employee_payment = (1 - employee_discount) * retail_cost →
  percentage_above_wholesale = 20 :=
by
  intros
  sorry

end percentage_above_wholesale_correct_l19_19527


namespace average_is_700_l19_19663

-- Define the list of known numbers
def numbers_without_x : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]

-- Define the value of x
def x : ℕ := 755

-- Define the list of all numbers including x
def all_numbers : List ℕ := numbers_without_x.append [x]

-- Define the total length of the list containing x
def n : ℕ := all_numbers.length

-- Define the sum of the numbers in the list including x
noncomputable def sum_all_numbers : ℕ := all_numbers.sum

-- Define the average formula
noncomputable def average : ℕ := sum_all_numbers / n

-- State the theorem
theorem average_is_700 : average = 700 := by
  sorry

end average_is_700_l19_19663


namespace necessary_but_not_sufficient_l19_19971

variable (p q : Prop)

theorem necessary_but_not_sufficient (h : ¬p → q) (h1 : ¬ (q → ¬p)) : ¬q → p := 
by
  sorry

end necessary_but_not_sufficient_l19_19971


namespace max_area_of_garden_l19_19586

theorem max_area_of_garden (p : ℝ) (h : p = 36) : 
  ∃ A : ℝ, (∀ l w : ℝ, l + l + w + w = p → l * w ≤ A) ∧ A = 81 :=
by
  sorry

end max_area_of_garden_l19_19586


namespace percentage_of_tip_l19_19720

-- Given conditions
def steak_cost : ℝ := 20
def drink_cost : ℝ := 5
def total_cost_before_tip : ℝ := 2 * (steak_cost + drink_cost)
def billy_tip_payment : ℝ := 8
def billy_tip_coverage : ℝ := 0.80

-- Required to prove
theorem percentage_of_tip : ∃ P : ℝ, (P = (billy_tip_payment / (billy_tip_coverage * total_cost_before_tip)) * 100) ∧ P = 20 := 
by {
  sorry
}

end percentage_of_tip_l19_19720


namespace average_score_all_students_l19_19103

theorem average_score_all_students 
  (n1 n2 : Nat) 
  (avg1 avg2 : Nat) 
  (h1 : n1 = 20) 
  (h2 : avg1 = 80) 
  (h3 : n2 = 30) 
  (h4 : avg2 = 70) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 74 := 
by
  sorry

end average_score_all_students_l19_19103


namespace temperature_rise_result_l19_19857

def initial_temperature : ℤ := -2
def rise : ℤ := 3

theorem temperature_rise_result : initial_temperature + rise = 1 := 
by 
  sorry

end temperature_rise_result_l19_19857


namespace nth_derivative_ln_correct_l19_19561

noncomputable def nth_derivative_ln (n : ℕ) : ℝ → ℝ
| x => (-1)^(n-1) * (Nat.factorial (n-1)) / (1 + x) ^ n

theorem nth_derivative_ln_correct (n : ℕ) (x : ℝ) :
  deriv^[n] (λ x => Real.log (1 + x)) x = nth_derivative_ln n x := 
by
  sorry

end nth_derivative_ln_correct_l19_19561


namespace shape_with_congruent_views_is_sphere_l19_19466

def is_congruent_views (shape : Type) : Prop :=
  ∀ (front_view left_view top_view : shape), 
  (front_view = left_view) ∧ (left_view = top_view) ∧ (front_view = top_view)

noncomputable def is_sphere (shape : Type) : Prop := 
  ∀ (s : shape), true -- Placeholder definition for a sphere, as recognizing a sphere is outside Lean's scope

theorem shape_with_congruent_views_is_sphere (shape : Type) :
  is_congruent_views shape → is_sphere shape :=
by
  intro h
  sorry

end shape_with_congruent_views_is_sphere_l19_19466


namespace polynomial_coefficient_product_identity_l19_19443

theorem polynomial_coefficient_product_identity (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h1 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 0)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = 32) :
  (a_0 + a_2 + a_4) * (a_1 + a_3 + a_5) = -256 := 
by {
  sorry
}

end polynomial_coefficient_product_identity_l19_19443


namespace image_of_3_5_pre_image_of_3_5_l19_19625

def f (x y : ℤ) : ℤ × ℤ := (x - y, x + y)

theorem image_of_3_5 : f 3 5 = (-2, 8) :=
by
  sorry

theorem pre_image_of_3_5 : ∃ (x y : ℤ), f x y = (3, 5) ∧ x = 4 ∧ y = 1 :=
by
  sorry

end image_of_3_5_pre_image_of_3_5_l19_19625


namespace DiagonalsOfShapesBisectEachOther_l19_19007

structure Shape where
  bisect_diagonals : Prop

def is_parallelogram (s : Shape) : Prop := s.bisect_diagonals
def is_rectangle (s : Shape) : Prop := s.bisect_diagonals
def is_rhombus (s : Shape) : Prop := s.bisect_diagonals
def is_square (s : Shape) : Prop := s.bisect_diagonals

theorem DiagonalsOfShapesBisectEachOther (s : Shape) :
  is_parallelogram s ∨ is_rectangle s ∨ is_rhombus s ∨ is_square s → s.bisect_diagonals := by
  sorry

end DiagonalsOfShapesBisectEachOther_l19_19007


namespace quadratic_solution_l19_19769

theorem quadratic_solution (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 → (x = 1 / 2 ∨ x = 1) :=
by sorry

end quadratic_solution_l19_19769


namespace max_bk_at_k_l19_19569
open Nat Real

theorem max_bk_at_k :
  let B_k (k : ℕ) := (choose 2000 k) * (0.1 : ℝ) ^ k
  ∃ k : ℕ, (k = 181) ∧ (∀ m : ℕ, B_k m ≤ B_k k) :=
sorry

end max_bk_at_k_l19_19569


namespace part1_part2_l19_19968

namespace MathProofProblem

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part1 (x : ℝ) : f 2 * x ≤ f (x + 1) ↔ 0 ≤ x ∧ x ≤ 1 := 
by
  sorry

theorem part2 (a b : ℝ) (h₀ : a + b = 2) : f (a ^ 2) + f (b ^ 2) = 2 :=
by
  sorry

end MathProofProblem

end part1_part2_l19_19968


namespace heptagon_diagonals_l19_19510

-- Define the number of sides of the polygon
def heptagon_sides : ℕ := 7

-- Define the formula for the number of diagonals of an n-gon
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem we want to prove, i.e., the number of diagonals in a convex heptagon is 14
theorem heptagon_diagonals : diagonals heptagon_sides = 14 := by
  sorry

end heptagon_diagonals_l19_19510


namespace terminal_side_quadrant_l19_19792

theorem terminal_side_quadrant (α : ℝ) (h : α = 2) : 
  90 < α * (180 / Real.pi) ∧ α * (180 / Real.pi) < 180 := 
by
  sorry

end terminal_side_quadrant_l19_19792


namespace polynomial_coefficient_sum_l19_19331

theorem polynomial_coefficient_sum :
  ∀ (a0 a1 a2 a3 a4 a5 : ℤ), 
  (3 - 2 * x)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 → 
  a0 + a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 = 233 :=
by
  sorry

end polynomial_coefficient_sum_l19_19331


namespace rectangle_area_l19_19738

theorem rectangle_area
  (x y : ℝ) -- sides of the rectangle
  (h1 : 2 * x + 2 * y = 12)  -- perimeter
  (h2 : x^2 + y^2 = 25)  -- diagonal
  : x * y = 5.5 :=
sorry

end rectangle_area_l19_19738


namespace prime_sum_of_composites_l19_19507

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ m * k = n
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def can_be_expressed_as_sum_of_two_composites (p : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b

theorem prime_sum_of_composites :
  can_be_expressed_as_sum_of_two_composites 13 ∧ 
  ∀ p : ℕ, is_prime p ∧ p > 13 → can_be_expressed_as_sum_of_two_composites p :=
by 
  sorry

end prime_sum_of_composites_l19_19507


namespace J_3_3_4_l19_19896

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_3_3_4 : J 3 (3 / 4) 4 = 259 / 48 := 
by {
    -- We would normally include proof steps here, but according to the instruction, we use 'sorry'.
    sorry
}

end J_3_3_4_l19_19896


namespace area_of_triangle_COD_l19_19898

theorem area_of_triangle_COD (x p : ℕ) (hx : 0 < x) (hx' : x < 12) (hp : 0 < p) :
  (∃ A : ℚ, A = (x * p : ℚ) / 2) :=
sorry

end area_of_triangle_COD_l19_19898


namespace exist_elements_inequality_l19_19186

open Set

theorem exist_elements_inequality (A : Set ℝ) (a_1 a_2 a_3 a_4 : ℝ)
(hA : A = {a_1, a_2, a_3, a_4})
(h_ineq1 : 0 < a_1 )
(h_ineq2 : a_1 < a_2 )
(h_ineq3 : a_2 < a_3 )
(h_ineq4 : a_3 < a_4 ) :
∃ (x y : ℝ), x ∈ A ∧ y ∈ A ∧ (2 + Real.sqrt 3) * |x - y| < (x + 1) * (y + 1) + x * y := 
sorry

end exist_elements_inequality_l19_19186


namespace cost_of_largest_pot_l19_19450

theorem cost_of_largest_pot
  (total_cost : ℝ)
  (n : ℕ)
  (a b : ℝ)
  (h_total_cost : total_cost = 7.80)
  (h_n : n = 6)
  (h_b : b = 0.25)
  (h_small_cost : ∃ x : ℝ, ∃ is_odd : ℤ → Prop, (∃ c: ℤ, x = c / 100 ∧ is_odd c) ∧
                  total_cost = x + (x + b) + (x + 2 * b) + (x + 3 * b) + (x + 4 * b) + (x + 5 * b)) :
  ∃ y, y = (x + 5*b) ∧ y = 1.92 :=
  sorry

end cost_of_largest_pot_l19_19450


namespace multiplication_verification_l19_19838

-- Define the variables
variables (P Q R S T U : ℕ)

-- Define the known digits in the numbers
def multiplicand := 60000 + 1000 * P + 100 * Q + 10 * R
def multiplier := 5000000 + 10000 * S + 1000 * T + 100 * U + 5

-- Define the proof statement
theorem multiplication_verification : 
  (multiplicand P Q R) * (multiplier S T U) = 20213 * 732575 :=
  sorry

end multiplication_verification_l19_19838


namespace part_I_part_II_l19_19916

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
x^2 / a^2 + y^2 / b^2 = 1

theorem part_I (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (eccentricity : ℝ := c / a) (h3 : eccentricity = Real.sqrt 2 / 2) (vertex : ℝ × ℝ := (0, 1)) (h4 : vertex = (0, b)) 
  : ellipse_equation (Real.sqrt 2) 1 (0:ℝ) 1 :=
sorry

theorem part_II (a b k : ℝ) (x y : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 1)
  (line_eq : ℝ → ℝ := fun x => k * x + 1) 
  (h3 : (1 + 2 * k^2) * x^2 + 4 * k * x = 0) 
  (distance_AB : ℝ := Real.sqrt 2 * 4 / 3) 
  (h4 : Real.sqrt (1 + k^2) * abs ((-4 * k) / (2 * k^2 + 1)) = distance_AB) 
  : (x, y) = (4/3, -1/3) ∨ (x, y) = (-4/3, -1/3) :=
sorry

end part_I_part_II_l19_19916


namespace cats_combined_weight_l19_19712

theorem cats_combined_weight :
  let cat1 := 2
  let cat2 := 7
  let cat3 := 4
  cat1 + cat2 + cat3 = 13 := 
by
  let cat1 := 2
  let cat2 := 7
  let cat3 := 4
  sorry

end cats_combined_weight_l19_19712


namespace elixir_concentration_l19_19998

theorem elixir_concentration (x a : ℝ) 
  (h1 : (x * 100) / (100 + a) = 9) 
  (h2 : (x * 100 + a * 100) / (100 + 2 * a) = 23) : 
  x = 11 :=
by 
  sorry

end elixir_concentration_l19_19998


namespace largest_tile_side_length_l19_19752

theorem largest_tile_side_length (w h : ℕ) (hw : w = 17) (hh : h = 23) : Nat.gcd w h = 1 := by
  -- Proof goes here
  sorry

end largest_tile_side_length_l19_19752


namespace michael_needs_more_money_l19_19294

-- Define the initial conditions
def michael_money : ℝ := 50
def cake_cost : ℝ := 20
def bouquet_cost : ℝ := 36
def balloons_cost : ℝ := 5
def perfume_gbp : ℝ := 30
def gbp_to_usd : ℝ := 1.4
def perfume_cost : ℝ := perfume_gbp * gbp_to_usd
def photo_album_eur : ℝ := 25
def eur_to_usd : ℝ := 1.2
def photo_album_cost : ℝ := photo_album_eur * eur_to_usd

-- Sum the costs
def total_cost : ℝ := cake_cost + bouquet_cost + balloons_cost + perfume_cost + photo_album_cost

-- Define the required amount
def additional_money_needed : ℝ := total_cost - michael_money

-- The theorem statement
theorem michael_needs_more_money : additional_money_needed = 83 := by
  sorry

end michael_needs_more_money_l19_19294


namespace work_done_by_force_l19_19244

def F (x : ℝ) := 4 * x - 1

theorem work_done_by_force :
  let a := 1
  let b := 3
  (∫ x in a..b, F x) = 14 := by
  sorry

end work_done_by_force_l19_19244


namespace find_f_of_3_l19_19388

theorem find_f_of_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f_of_3_l19_19388


namespace arithmetic_expression_l19_19425

theorem arithmetic_expression : (4 + 6 + 4) / 3 - 4 / 3 = 10 / 3 := by
  sorry

end arithmetic_expression_l19_19425


namespace find_digit_B_l19_19106

theorem find_digit_B (B : ℕ) (h1 : B < 10) : 3 ∣ (5 + 2 + B + 6) → B = 2 :=
by
  sorry

end find_digit_B_l19_19106


namespace valid_lineups_count_l19_19541

-- Definitions of the problem conditions
def num_players : ℕ := 18
def quadruplets : Finset ℕ := {0, 1, 2, 3} -- Indices of Benjamin, Brenda, Brittany, Bryan
def total_starters : ℕ := 8

-- Function to count lineups based on given constraints
noncomputable def count_valid_lineups : ℕ :=
  let others := num_players - quadruplets.card
  Nat.choose others total_starters + quadruplets.card * Nat.choose others (total_starters - 1)

-- The theorem to prove the count of valid lineups
theorem valid_lineups_count : count_valid_lineups = 16731 := by
  -- Placeholder for the actual proof
  sorry

end valid_lineups_count_l19_19541


namespace quotient_of_division_l19_19656

theorem quotient_of_division (L S Q : ℕ) (h1 : L - S = 2500) (h2 : L = 2982) (h3 : L = Q * S + 15) : Q = 6 := 
sorry

end quotient_of_division_l19_19656


namespace exists_disjoint_subsets_for_prime_products_l19_19100

theorem exists_disjoint_subsets_for_prime_products :
  ∃ (A : Fin 100 → Set ℕ), (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
    (∀ S : Set ℕ, Infinite S → (∃ m : ℕ, ∃ (a : Fin 100 → ℕ),
      (∀ i, a i ∈ A i) ∧ (∀ i, ∃ p : Fin m → ℕ, (∀ k, p k ∈ S) ∧ a i = (List.prod (List.ofFn p))))) :=
sorry

end exists_disjoint_subsets_for_prime_products_l19_19100


namespace relationship_of_a_and_b_l19_19128

theorem relationship_of_a_and_b (a b : ℝ) (h_b_nonzero: b ≠ 0)
  (m n : ℤ) (h_intersection : ∃ (m n : ℤ), n = m^3 - a * m^2 - b * m ∧ n = a * m + b) :
  2 * a - b + 8 = 0 :=
  sorry

end relationship_of_a_and_b_l19_19128


namespace draw_at_least_one_red_card_l19_19251

-- Define the deck and properties
def total_cards := 52
def red_cards := 26
def black_cards := 26

-- Define the calculation for drawing three cards sequentially
def total_ways_draw3 := total_cards * (total_cards - 1) * (total_cards - 2)
def black_only_ways_draw3 := black_cards * (black_cards - 1) * (black_cards - 2)

-- Define the main proof statement
theorem draw_at_least_one_red_card : 
    total_ways_draw3 - black_only_ways_draw3 = 117000 := by
    -- Proof is omitted
    sorry

end draw_at_least_one_red_card_l19_19251


namespace prove_tirzah_handbags_l19_19937
noncomputable def tirzah_has_24_handbags (H : ℕ) : Prop :=
  let P := 26 -- number of purses
  let fakeP := P / 2 -- half of the purses are fake
  let authP := P - fakeP -- number of authentic purses
  let fakeH := H / 4 -- one quarter of the handbags are fake
  let authH := H - fakeH -- number of authentic handbags
  authP + authH = 31 -- total number of authentic items
  → H = 24 -- prove the number of handbags is 24

theorem prove_tirzah_handbags : ∃ H : ℕ, tirzah_has_24_handbags H :=
  by
    use 24
    -- Proof goes here
    sorry

end prove_tirzah_handbags_l19_19937


namespace num_passed_candidates_l19_19559

theorem num_passed_candidates
  (total_candidates : ℕ)
  (avg_passed_marks : ℕ)
  (avg_failed_marks : ℕ)
  (overall_avg_marks : ℕ)
  (h1 : total_candidates = 120)
  (h2 : avg_passed_marks = 39)
  (h3 : avg_failed_marks = 15)
  (h4 : overall_avg_marks = 35) :
  ∃ (P : ℕ), P = 100 :=
by
  sorry

end num_passed_candidates_l19_19559


namespace find_x_plus_y_l19_19145

theorem find_x_plus_y (x y : ℝ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : x - y > 0) : x + y = 8 ∨ x + y = 2 :=
by
  sorry

end find_x_plus_y_l19_19145


namespace jeans_original_price_l19_19518

theorem jeans_original_price 
  (discount : ℝ -> ℝ)
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (final_price : ℝ) 
  (customer_payment : ℝ) : 
  discount_percentage = 0.10 -> 
  discount x = x * (1 - discount_percentage) -> 
  final_price = discount (2 * original_price) + original_price -> 
  customer_payment = 112 -> 
  final_price = 112 -> 
  original_price = 40 := 
by
  intros
  sorry

end jeans_original_price_l19_19518


namespace sum_of_x_y_l19_19856

theorem sum_of_x_y (x y : ℕ) (h1 : 10 * x + y = 75) (h2 : 10 * y + x = 57) : x + y = 12 :=
sorry

end sum_of_x_y_l19_19856


namespace jersey_to_shoes_ratio_l19_19264

theorem jersey_to_shoes_ratio
  (pairs_shoes: ℕ) (jerseys: ℕ) (total_cost: ℝ) (total_cost_shoes: ℝ) 
  (shoes: pairs_shoes = 6) (jer: jerseys = 4) (total: total_cost = 560) (cost_sh: total_cost_shoes = 480) :
  ((total_cost - total_cost_shoes) / jerseys) / (total_cost_shoes / pairs_shoes) = 1 / 4 := 
by 
  sorry

end jersey_to_shoes_ratio_l19_19264


namespace billy_tickets_used_l19_19381

-- Definitions for the number of rides and cost per ride
def ferris_wheel_rides : Nat := 7
def bumper_car_rides : Nat := 3
def ticket_per_ride : Nat := 5

-- Total number of rides
def total_rides : Nat := ferris_wheel_rides + bumper_car_rides

-- Total tickets used
def total_tickets : Nat := total_rides * ticket_per_ride

-- Theorem stating the number of tickets Billy used in total
theorem billy_tickets_used : total_tickets = 50 := by
  sorry

end billy_tickets_used_l19_19381


namespace periodic_odd_function_value_at_7_l19_19340

noncomputable def f : ℝ → ℝ := sorry -- Need to define f appropriately, skipped for brevity

theorem periodic_odd_function_value_at_7
    (f_odd : ∀ x : ℝ, f (-x) = -f x)
    (f_periodic : ∀ x : ℝ, f (x + 4) = f x)
    (f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) :
    f 7 = -1 := sorry

end periodic_odd_function_value_at_7_l19_19340


namespace no_three_consecutive_geo_prog_l19_19151

theorem no_three_consecutive_geo_prog (n k m: ℕ) (h: n ≠ k ∧ n ≠ m ∧ k ≠ m) :
  ¬(∃ a b c: ℕ, 
    (a = 2^n + 1 ∧ b = 2^k + 1 ∧ c = 2^m + 1) ∧ 
    (b^2 = a * c)) :=
by sorry

end no_three_consecutive_geo_prog_l19_19151


namespace sofa_price_is_correct_l19_19451

def price_sofa (invoice_total armchair_price table_price : ℕ) (armchair_count : ℕ) : ℕ :=
  invoice_total - (armchair_price * armchair_count + table_price)

theorem sofa_price_is_correct
  (invoice_total : ℕ)
  (armchair_price : ℕ)
  (table_price : ℕ)
  (armchair_count : ℕ)
  (sofa_price : ℕ)
  (h_invoice : invoice_total = 2430)
  (h_armchair_price : armchair_price = 425)
  (h_table_price : table_price = 330)
  (h_armchair_count : armchair_count = 2)
  (h_sofa_price : sofa_price = 1250) :
  price_sofa invoice_total armchair_price table_price armchair_count = sofa_price :=
by
  sorry

end sofa_price_is_correct_l19_19451


namespace area_of_set_R_is_1006point5_l19_19168

-- Define the set of points R as described in the problem
def isPointInSetR (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + y ≤ 2013 ∧ ⌈x⌉ * ⌊y⌋ = ⌊x⌋ * ⌈y⌉

noncomputable def computeAreaOfSetR : ℝ :=
  1006.5

theorem area_of_set_R_is_1006point5 :
  (∃ x y : ℝ, isPointInSetR x y) → computeAreaOfSetR = 1006.5 := by
  sorry

end area_of_set_R_is_1006point5_l19_19168


namespace fuel_tank_capacity_l19_19297

theorem fuel_tank_capacity (C : ℝ) 
  (h1 : 0.12 * 98 + 0.16 * (C - 98) = 30) : 
  C = 212 :=
by
  sorry

end fuel_tank_capacity_l19_19297


namespace number_of_space_diagonals_l19_19904

theorem number_of_space_diagonals
  (V E F T Q : ℕ)
  (hV : V = 30)
  (hE : E = 70)
  (hF : F = 42)
  (hT : T = 30)
  (hQ : Q = 12):
  (V * (V - 1) / 2 - E - 2 * Q) = 341 :=
by
  sorry

end number_of_space_diagonals_l19_19904


namespace least_value_of_b_l19_19644

variable {x y b : ℝ}

noncomputable def condition_inequality (x y b : ℝ) : Prop :=
  (x^2 + y^2)^2 ≤ b * (x^4 + y^4)

theorem least_value_of_b (h : ∀ x y : ℝ, condition_inequality x y b) : b ≥ 2 := 
sorry

end least_value_of_b_l19_19644


namespace max_chips_with_constraints_l19_19351

theorem max_chips_with_constraints (n : ℕ) (h1 : n > 0) 
  (h2 : ∀ i j : ℕ, (i < n) → (j = i + 10 ∨ j = i + 15) → ((i % 25) = 0 ∨ (j % 25) = 0)) :
  n ≤ 25 := 
sorry

end max_chips_with_constraints_l19_19351


namespace local_minimum_f_eval_integral_part_f_l19_19199

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sin x * Real.sqrt (1 - Real.cos x))

theorem local_minimum_f :
  (0 < x) -> (x < π) -> f x >= 1 :=
  by sorry

theorem eval_integral_part_f :
  ∫ x in (↑(π / 2))..(↑(2 * π / 3)), f x = sorry :=
  by sorry

end local_minimum_f_eval_integral_part_f_l19_19199


namespace VasyaSlowerWalkingFullWayHome_l19_19081

namespace FishingTrip

-- Define the variables involved
variables (x v S : ℝ)   -- x is the speed of Vasya and Petya, v is the speed of Kolya on the bicycle, S is the distance from the house to the lake

-- Conditions derived from the problem statement:
-- Condition 1: When Kolya meets Vasya then Petya starts
-- Condition 2: Given: Petya’s travel time is \( \frac{5}{4} \times \) Vasya's travel time.

theorem VasyaSlowerWalkingFullWayHome (h1 : v = 3 * x) :
  2 * (S / x + v) = (5 / 2) * (S / x) :=
sorry

end FishingTrip

end VasyaSlowerWalkingFullWayHome_l19_19081


namespace determine_f_zero_l19_19915

variable (f : ℝ → ℝ)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 + y) + 4 * (f x) * y

theorem determine_f_zero (h1: functional_equation f)
    (h2 : f 2 = 4) : f 0 = 0 := 
sorry

end determine_f_zero_l19_19915


namespace minimum_value_squared_sum_minimum_value_squared_sum_equality_l19_19034

theorem minimum_value_squared_sum (a b c t : ℝ) (h : a + b + c = t) : 
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

theorem minimum_value_squared_sum_equality (a b c t : ℝ) (h : a + b + c = t) 
  (ha : a = t / 3) (hb : b = t / 3) (hc : c = t / 3) : 
  a^2 + b^2 + c^2 = t^2 / 3 := by
  sorry

end minimum_value_squared_sum_minimum_value_squared_sum_equality_l19_19034


namespace decreasing_interval_f_l19_19617

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Statement to prove that the interval where f is monotonically decreasing is [2, +∞)
theorem decreasing_interval_f : (∀ x₁ x₂ : ℝ, 2 ≤ x₁ ∧ x₁ ≤ x₂ → f x₁ ≥ f x₂) :=
by
  sorry

end decreasing_interval_f_l19_19617


namespace sum_of_exterior_angles_of_convex_quadrilateral_l19_19345

theorem sum_of_exterior_angles_of_convex_quadrilateral:
  ∀ (α β γ δ : ℝ),
  (α + β + γ + δ = 360) → 
  (∀ (θ₁ θ₂ θ₃ θ₄ : ℝ),
    (θ₁ = 180 - α ∧ θ₂ = 180 - β ∧ θ₃ = 180 - γ ∧ θ₄ = 180 - δ) → 
    θ₁ + θ₂ + θ₃ + θ₄ = 360) := 
by 
  intros α β γ δ h1 θ₁ θ₂ θ₃ θ₄ h2
  rcases h2 with ⟨hα, hβ, hγ, hδ⟩
  rw [hα, hβ, hγ, hδ]
  linarith

end sum_of_exterior_angles_of_convex_quadrilateral_l19_19345


namespace similar_triangles_height_l19_19170

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end similar_triangles_height_l19_19170


namespace find_abc_digits_l19_19813

theorem find_abc_digits (N : ℕ) (abcd : ℕ) (a b c d : ℕ) (hN : N % 10000 = abcd) (hNsq : N^2 % 10000 = abcd)
  (ha_ne_zero : a ≠ 0) (hb_ne_six : b ≠ 6) (hc_ne_six : c ≠ 6) : (a * 100 + b * 10 + c) = 106 :=
by
  -- The proof is omitted.
  sorry

end find_abc_digits_l19_19813


namespace smallest_b_for_factorization_l19_19488

theorem smallest_b_for_factorization : ∃ (b : ℕ), (∀ p q : ℤ, (x^2 + (b * x) + 2352) = (x + p) * (x + q) → p + q = b ∧ p * q = 2352) ∧ b = 112 := 
sorry

end smallest_b_for_factorization_l19_19488


namespace diff_is_multiple_of_9_l19_19082

-- Definitions
def orig_num (a b : ℕ) : ℕ := 10 * a + b
def new_num (a b : ℕ) : ℕ := 10 * b + a

-- Statement of the mathematical proof problem
theorem diff_is_multiple_of_9 (a b : ℕ) : 
  9 ∣ (new_num a b - orig_num a b) :=
by
  sorry

end diff_is_multiple_of_9_l19_19082


namespace kanul_raw_material_expense_l19_19521

theorem kanul_raw_material_expense
  (total_amount : ℝ)
  (machinery_cost : ℝ)
  (raw_materials_cost : ℝ)
  (cash_fraction : ℝ)
  (h_total_amount : total_amount = 137500)
  (h_machinery_cost : machinery_cost = 30000)
  (h_cash_fraction: cash_fraction = 0.20)
  (h_eq : total_amount = raw_materials_cost + machinery_cost + cash_fraction * total_amount) :
  raw_materials_cost = 80000 :=
by
  rw [h_total_amount, h_machinery_cost, h_cash_fraction] at h_eq
  sorry

end kanul_raw_material_expense_l19_19521


namespace satisfactory_fraction_l19_19565

theorem satisfactory_fraction :
  let num_students_A := 8
  let num_students_B := 7
  let num_students_C := 6
  let num_students_D := 5
  let num_students_F := 4
  let satisfactory_grades := num_students_A + num_students_B + num_students_C
  let total_students := num_students_A + num_students_B + num_students_C + num_students_D + num_students_F
  satisfactory_grades / total_students = 7 / 10 :=
by
  let num_students_A := 8
  let num_students_B := 7
  let num_students_C := 6
  let num_students_D := 5
  let num_students_F := 4
  let satisfactory_grades := num_students_A + num_students_B + num_students_C
  let total_students := num_students_A + num_students_B + num_students_C + num_students_D + num_students_F
  have h1: satisfactory_grades = 21 := by sorry
  have h2: total_students = 30 := by sorry
  have fraction := (satisfactory_grades: ℚ) / total_students
  have simplified_fraction := fraction = 7 / 10
  exact sorry

end satisfactory_fraction_l19_19565


namespace drug_price_reduction_l19_19980

theorem drug_price_reduction :
  ∃ x : ℝ, 56 * (1 - x)^2 = 31.5 :=
by
  sorry

end drug_price_reduction_l19_19980


namespace Kath_payment_l19_19643

noncomputable def reducedPrice (standardPrice discount : ℝ) : ℝ :=
  standardPrice - discount

noncomputable def totalCost (numPeople price : ℝ) : ℝ :=
  numPeople * price

theorem Kath_payment :
  let standardPrice := 8
  let discount := 3
  let numPeople := 6
  let movieTime := 16 -- 4 P.M. in 24-hour format
  let reduced := reducedPrice standardPrice discount
  totalCost numPeople reduced = 30 :=
by
  sorry

end Kath_payment_l19_19643


namespace brenda_cakes_l19_19202

theorem brenda_cakes : 
  let cakes_per_day := 20
  let days := 9
  let total_cakes := cakes_per_day * days
  let sold_cakes := total_cakes / 2
  total_cakes - sold_cakes = 90 :=
by 
  sorry

end brenda_cakes_l19_19202


namespace complement_union_l19_19713

theorem complement_union (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {2, 3, 5})
  (hN : N = {4, 5}) :
  U \ (M ∪ N) = {1, 6} :=
by
  sorry

end complement_union_l19_19713


namespace min_value_sum_reciprocal_l19_19921

open Real

theorem min_value_sum_reciprocal (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
    (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
    1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 3 / 4 :=
by
  sorry

end min_value_sum_reciprocal_l19_19921


namespace desired_depth_proof_l19_19932

-- Definitions based on the conditions in Step a)
def initial_men : ℕ := 9
def initial_hours : ℕ := 8
def initial_depth : ℕ := 30
def extra_men : ℕ := 11
def total_men : ℕ := initial_men + extra_men
def new_hours : ℕ := 6

-- Total man-hours for initial setup
def initial_man_hours (days : ℕ) : ℕ := initial_men * initial_hours * days

-- Total man-hours for new setup to achieve desired depth
def new_man_hours (desired_depth : ℕ) (days : ℕ) : ℕ := total_men * new_hours * days

-- Proportional relationship between initial setup and desired depth
theorem desired_depth_proof (days : ℕ) (desired_depth : ℕ) :
  initial_man_hours days / initial_depth = new_man_hours desired_depth days / desired_depth → desired_depth = 18 :=
by
  sorry

end desired_depth_proof_l19_19932


namespace page_added_twice_l19_19317

theorem page_added_twice (n k : ℕ) (h1 : (n * (n + 1)) / 2 + k = 1986) : k = 33 :=
sorry

end page_added_twice_l19_19317


namespace geun_bae_fourth_day_jumps_l19_19829

-- Define a function for number of jump ropes Geun-bae does on each day
def jump_ropes (n : ℕ) : ℕ :=
  match n with
  | 0     => 15
  | n + 1 => 2 * jump_ropes n

-- Theorem stating the number of jump ropes Geun-bae does on the fourth day
theorem geun_bae_fourth_day_jumps : jump_ropes 3 = 120 := 
by {
  sorry
}

end geun_bae_fourth_day_jumps_l19_19829


namespace acetic_acid_molecular_weight_is_correct_l19_19705

def molecular_weight_acetic_acid : ℝ :=
  let carbon_weight := 12.01
  let hydrogen_weight := 1.008
  let oxygen_weight := 16.00
  let num_carbons := 2
  let num_hydrogens := 4
  let num_oxygens := 2
  num_carbons * carbon_weight + num_hydrogens * hydrogen_weight + num_oxygens * oxygen_weight

theorem acetic_acid_molecular_weight_is_correct : molecular_weight_acetic_acid = 60.052 :=
by 
  unfold molecular_weight_acetic_acid
  sorry

end acetic_acid_molecular_weight_is_correct_l19_19705


namespace determine_a2016_l19_19086

noncomputable def a_n (n : ℕ) : ℤ := sorry
noncomputable def S_n (n : ℕ) : ℤ := sorry

axiom S1 : S_n 1 = 6
axiom S2 : S_n 2 = 4
axiom S_pos (n : ℕ) : S_n n > 0
axiom geom_progression (n : ℕ) : (S_n (2 * n - 1))^2 = S_n (2 * n) * S_n (2 * n + 2)
axiom arith_progression (n : ℕ) : 2 * S_n (2 * n + 2) = S_n (2 * n - 1) + S_n (2 * n + 1)

theorem determine_a2016 : a_n 2016 = -1009 :=
by sorry

end determine_a2016_l19_19086


namespace angleina_speed_from_grocery_to_gym_l19_19745

variable (v : ℝ) (h1 : 720 / v - 40 = 240 / v)

theorem angleina_speed_from_grocery_to_gym : 2 * v = 24 :=
by
  sorry

end angleina_speed_from_grocery_to_gym_l19_19745


namespace total_population_of_city_l19_19253

theorem total_population_of_city (P : ℝ) (h : 0.85 * P = 85000) : P = 100000 :=
  by
  sorry

end total_population_of_city_l19_19253


namespace ratio_of_red_to_blue_beads_l19_19972

theorem ratio_of_red_to_blue_beads (red_beads blue_beads : ℕ) (h1 : red_beads = 30) (h2 : blue_beads = 20) :
    (red_beads / Nat.gcd red_beads blue_beads) = 3 ∧ (blue_beads / Nat.gcd red_beads blue_beads) = 2 := 
by 
    -- Proof will go here
    sorry

end ratio_of_red_to_blue_beads_l19_19972


namespace tan_half_angle_sum_identity_l19_19815

theorem tan_half_angle_sum_identity
  (α β γ : ℝ)
  (h : Real.sin α + Real.sin γ = 2 * Real.sin β) :
  Real.tan ((α + β) / 2) + Real.tan ((β + γ) / 2) = 2 * Real.tan ((γ + α) / 2) :=
sorry

end tan_half_angle_sum_identity_l19_19815


namespace temperature_on_tuesday_l19_19336

theorem temperature_on_tuesday 
  (T W Th F : ℝ)
  (H1 : (T + W + Th) / 3 = 45)
  (H2 : (W + Th + F) / 3 = 50)
  (H3 : F = 53) :
  T = 38 :=
by 
  sorry

end temperature_on_tuesday_l19_19336


namespace isosceles_triangle_base_length_l19_19036

theorem isosceles_triangle_base_length
  (a b : ℝ) (h₁ : a = 4) (h₂ : b = 8) (h₃ : a ≠ b)
  (triangle_inequality : ∀ x y z : ℝ, x + y > z) :
  ∃ base : ℝ, base = 8 := by
  sorry

end isosceles_triangle_base_length_l19_19036


namespace spherical_coordinates_equivalence_l19_19098

theorem spherical_coordinates_equivalence :
  ∀ (ρ θ φ : ℝ), 
        ρ = 3 → θ = (2 * Real.pi / 7) → φ = (8 * Real.pi / 5) →
        (0 < ρ) → 
        (0 ≤ (2 * Real.pi / 7) ∧ (2 * Real.pi / 7) < 2 * Real.pi) →
        (0 ≤ (8 * Real.pi / 5) ∧ (8 * Real.pi / 5) ≤ Real.pi) →
      ∃ (ρ' θ' φ' : ℝ), 
        ρ' = ρ ∧ θ' = (9 * Real.pi / 7) ∧ φ' = (2 * Real.pi / 5) :=
by
    sorry

end spherical_coordinates_equivalence_l19_19098


namespace purchase_price_eq_360_l19_19845

theorem purchase_price_eq_360 (P : ℝ) (M : ℝ) (H1 : M = 30) (H2 : M = 0.05 * P + 12) : P = 360 :=
by
  sorry

end purchase_price_eq_360_l19_19845


namespace find_number_of_white_balls_l19_19627

theorem find_number_of_white_balls (n : ℕ) (h : 6 / (6 + n) = 2 / 5) : n = 9 :=
sorry

end find_number_of_white_balls_l19_19627


namespace determine_k_l19_19442

theorem determine_k (a b c k : ℝ) (h : a + b + c = 1) (h_eq : k * (a + bc) = (a + b) * (a + c)) : k = 1 :=
sorry

end determine_k_l19_19442


namespace roots_of_x2_eq_x_l19_19994

theorem roots_of_x2_eq_x : ∀ x : ℝ, x^2 = x ↔ (x = 0 ∨ x = 1) := 
by
  sorry

end roots_of_x2_eq_x_l19_19994


namespace problem_inequality_solution_l19_19641

theorem problem_inequality_solution (x : ℝ) :
  5 ≤ (x - 1) / (3 * x - 7) ∧ (x - 1) / (3 * x - 7) < 10 ↔ (69 / 29) < x ∧ x ≤ (17 / 7) :=
by sorry

end problem_inequality_solution_l19_19641


namespace sally_students_are_30_l19_19275

-- Define the conditions given in the problem
def school_money : ℕ := 320
def book_cost : ℕ := 12
def sally_money : ℕ := 40
def total_students : ℕ := 30

-- Define the total amount Sally can spend on books
def total_amount_available : ℕ := school_money + sally_money

-- The total cost of books for S students
def total_cost (S : ℕ) : ℕ := book_cost * S

-- The main theorem stating that S students will cost the same as the amount Sally can spend
theorem sally_students_are_30 : total_cost 30 = total_amount_available :=
by
  sorry

end sally_students_are_30_l19_19275


namespace solve_x_l19_19444

theorem solve_x (x : ℝ) (h : (30 * x + 15)^(1/3) = 15) : x = 112 := by
  sorry

end solve_x_l19_19444


namespace letter_at_position_in_pattern_l19_19173

/-- Determine the 150th letter in the repeating pattern XYZ is "Z"  -/
theorem letter_at_position_in_pattern :
  ∀ (pattern : List Char) (position : ℕ), pattern = ['X', 'Y', 'Z'] → position = 150 → pattern.get! ((position - 1) % pattern.length) = 'Z' :=
by
  intros pattern position
  intro hPattern hPosition
  rw [hPattern, hPosition]
  -- pattern = ['X', 'Y', 'Z'] and position = 150
  sorry

end letter_at_position_in_pattern_l19_19173


namespace problem_solve_l19_19292

theorem problem_solve (n : ℕ) (h_pos : 0 < n) 
    (h_eq : Real.sin (Real.pi / (3 * n)) + Real.cos (Real.pi / (3 * n)) = Real.sqrt (2 * n) / 3) : 
    n = 6 := 
  sorry

end problem_solve_l19_19292


namespace y_satisfies_quadratic_l19_19737

theorem y_satisfies_quadratic (x y : ℝ) 
  (h1 : 2 * x^2 + 6 * x + 5 * y + 1 = 0)
  (h2 : 2 * x + y + 3 = 0) : y^2 + 10 * y - 7 = 0 := 
sorry

end y_satisfies_quadratic_l19_19737


namespace sum_h_k_a_b_l19_19200

def h : ℤ := 3
def k : ℤ := -5
def a : ℤ := 7
def b : ℤ := 4

theorem sum_h_k_a_b : h + k + a + b = 9 := by
  sorry

end sum_h_k_a_b_l19_19200


namespace marble_distribution_l19_19918

theorem marble_distribution (x : ℚ) :
    (2 * x + 2) + (3 * x) + (x + 4) = 56 ↔ x = 25 / 3 := by
  sorry

end marble_distribution_l19_19918


namespace airplane_shot_down_l19_19928

def P_A : ℝ := 0.4
def P_B : ℝ := 0.5
def P_C : ℝ := 0.8

def P_one_hit : ℝ := 0.4
def P_two_hit : ℝ := 0.7
def P_three_hit : ℝ := 1

def P_one : ℝ := (P_A * (1 - P_B) * (1 - P_C)) + ((1 - P_A) * P_B * (1 - P_C)) + ((1 - P_A) * (1 - P_B) * P_C)
def P_two : ℝ := (P_A * P_B * (1 - P_C)) + (P_A * (1 - P_B) * P_C) + ((1 - P_A) * P_B * P_C)
def P_three : ℝ := P_A * P_B * P_C

def total_probability := (P_one * P_one_hit) + (P_two * P_two_hit) + (P_three * P_three_hit)

theorem airplane_shot_down : total_probability = 0.604 := by
  sorry

end airplane_shot_down_l19_19928


namespace ratio_of_side_length_to_radius_l19_19855

theorem ratio_of_side_length_to_radius (r s : ℝ) (c d : ℝ) 
  (h1 : s = 2 * r)
  (h2 : s^2 = (c / d) * (s^2 - π * r^2)) : 
  (s / r) = (Real.sqrt (c * π) / Real.sqrt (d - c)) := by
  sorry

end ratio_of_side_length_to_radius_l19_19855


namespace binomial_expansion_fifth_term_constant_l19_19120

open Classical -- Allows the use of classical logic

noncomputable def binomial_term (n r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (x ^ (n - r) / (x ^ r * (2 ^ r / x ^ r)))

theorem binomial_expansion_fifth_term_constant (n : ℕ) :
  (binomial_term n 4 x = (x ^ (n - 3 * 4) * (-2) ^ 4)) → n = 12 := by
  intro h
  sorry

end binomial_expansion_fifth_term_constant_l19_19120


namespace tangent_point_at_slope_one_l19_19958

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 - 3 * x

-- Define the derivative of the curve
def derivative (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem proof problem
theorem tangent_point_at_slope_one : ∃ x : ℝ, derivative x = 1 ∧ x = 2 :=
by
  sorry

end tangent_point_at_slope_one_l19_19958


namespace Emmy_money_l19_19849

theorem Emmy_money {Gerry_money cost_per_apple number_of_apples Emmy_money : ℕ} 
    (h1 : Gerry_money = 100)
    (h2 : cost_per_apple = 2) 
    (h3 : number_of_apples = 150) 
    (h4 : number_of_apples * cost_per_apple = Gerry_money + Emmy_money) :
    Emmy_money = 200 :=
by
   sorry

end Emmy_money_l19_19849


namespace largest_difference_l19_19341

noncomputable def A := 3 * (2010: ℕ) ^ 2011
noncomputable def B := (2010: ℕ) ^ 2011
noncomputable def C := 2009 * (2010: ℕ) ^ 2010
noncomputable def D := 3 * (2010: ℕ) ^ 2010
noncomputable def E := (2010: ℕ) ^ 2010
noncomputable def F := (2010: ℕ) ^ 2009

theorem largest_difference :
  (A - B) > (B - C) ∧ (A - B) > (C - D) ∧ (A - B) > (D - E) ∧ (A - B) > (E - F) :=
by
  sorry

end largest_difference_l19_19341


namespace domain_f_log_l19_19203

noncomputable def domain_f (u : Real) : u ∈ Set.Icc (1 : Real) 2 := sorry

theorem domain_f_log (x : Real) : (x ∈ Set.Icc (4 : Real) 16) :=
by
  have h : ∀ x, (1 : Real) ≤ 2^x ∧ 2^x ≤ 2
  { intro x
    sorry }
  have h_log : ∀ x, 2 ≤ x ∧ x ≤ 4 
  { intro x
    sorry }
  have h_domain : ∀ x, 4 ≤ x ∧ x ≤ 16
  { intro x
    sorry }
  exact sorry

end domain_f_log_l19_19203


namespace evaluate_sqrt_sum_l19_19284

theorem evaluate_sqrt_sum : (Real.sqrt 1 + Real.sqrt 9) = 4 := by
  sorry

end evaluate_sqrt_sum_l19_19284


namespace find_y_l19_19879

theorem find_y (a b y : ℝ) (h1 : s = (3 * a) ^ (2 * b)) (h2 : s = 5 * (a ^ b) * (y ^ b))
  (h3 : 0 < a) (h4 : 0 < b) : 
  y = 9 * a / 5 := by
  sorry

end find_y_l19_19879


namespace range_of_a_l19_19226

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 2| + |x + 2| ≤ a^2 - 3 * a) ↔ (a ≥ 4 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l19_19226


namespace determine_a_l19_19987

theorem determine_a (x : ℝ) (n : ℕ) (h : x > 0) (h_ineq : x + a / x^n ≥ n + 1) : a = n^n := by
  sorry

end determine_a_l19_19987


namespace find_E_equals_2023_l19_19493

noncomputable def proof : Prop :=
  ∃ a b c : ℝ, a ≠ b ∧ (a^2 * (b + c) = 2023) ∧ (b^2 * (c + a) = 2023) ∧ (c^2 * (a + b) = 2023)

theorem find_E_equals_2023 : proof :=
by
  sorry

end find_E_equals_2023_l19_19493


namespace Grisha_owes_correct_l19_19156

noncomputable def Grisha_owes (dish_cost : ℝ) : ℝ × ℝ :=
  let misha_paid := 3 * dish_cost
  let sasha_paid := 2 * dish_cost
  let friends_contribution := 50
  let equal_payment := 50 / 2
  (misha_paid - equal_payment, sasha_paid - equal_payment)

theorem Grisha_owes_correct :
  ∀ (dish_cost : ℝ), (dish_cost = 30) → Grisha_owes dish_cost = (40, 10) :=
by
  intro dish_cost h
  rw [h]
  unfold Grisha_owes
  simp
  sorry

end Grisha_owes_correct_l19_19156


namespace letters_by_30_typists_in_1_hour_l19_19287

-- Definitions from the conditions
def lettersTypedByOneTypistIn20Minutes := 44 / 20

def lettersTypedBy30TypistsIn20Minutes := 30 * (lettersTypedByOneTypistIn20Minutes)

def conversionToHours := 3

-- Theorem statement
theorem letters_by_30_typists_in_1_hour : lettersTypedBy30TypistsIn20Minutes * conversionToHours = 198 := by
  sorry

end letters_by_30_typists_in_1_hour_l19_19287


namespace wrench_force_inv_proportional_l19_19165

theorem wrench_force_inv_proportional (F₁ : ℝ) (L₁ : ℝ) (F₂ : ℝ) (L₂ : ℝ) (k : ℝ)
  (h₁ : F₁ * L₁ = k) (h₂ : L₁ = 12) (h₃ : F₁ = 300) (h₄ : L₂ = 18) :
  F₂ = 200 :=
by
  sorry

end wrench_force_inv_proportional_l19_19165


namespace cos_A_value_l19_19930

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
-- a, b, c are the sides opposite to angles A, B, and C respectively.
-- Assumption 1: b - c = (1/4) * a
def condition1 := b - c = (1/4) * a
-- Assumption 2: 2 * sin B = 3 * sin C
def condition2 := 2 * Real.sin B = 3 * Real.sin C

-- The theorem statement: Under these conditions, prove that cos A = -1/4.
theorem cos_A_value (h1 : condition1 a b c) (h2 : condition2 B C) : 
    Real.cos A = -1/4 :=
sorry -- placeholder for the proof

end cos_A_value_l19_19930


namespace find_purchase_price_l19_19101

noncomputable def purchase_price (a : ℝ) : ℝ := a
def retail_price : ℝ := 1100
def discount_rate : ℝ := 0.8
def profit_rate : ℝ := 0.1

theorem find_purchase_price (a : ℝ) (h : purchase_price a * (1 + profit_rate) = retail_price * discount_rate) : a = 800 := by
  sorry

end find_purchase_price_l19_19101


namespace find_number_l19_19143

theorem find_number (n : ℝ) (h : 1 / 2 * n + 7 = 17) : n = 20 :=
by
  sorry

end find_number_l19_19143


namespace divide_numbers_into_consecutive_products_l19_19295

theorem divide_numbers_into_consecutive_products :
  ∃ (A B : Finset ℕ), A ∪ B = {2, 3, 5, 7, 11, 13, 17} ∧ A ∩ B = ∅ ∧ 
  (A.prod id = 714 ∧ B.prod id = 715 ∨ A.prod id = 715 ∧ B.prod id = 714) :=
sorry

end divide_numbers_into_consecutive_products_l19_19295


namespace floor_x_mul_x_eq_54_l19_19583

def positive_real (x : ℝ) : Prop := x > 0

theorem floor_x_mul_x_eq_54 (x : ℝ) (h_pos : positive_real x) : ⌊x⌋ * x = 54 ↔ x = 54 / 7 :=
by
  sorry

end floor_x_mul_x_eq_54_l19_19583


namespace base8_to_base10_sum_l19_19953

theorem base8_to_base10_sum (a b : ℕ) (h₁ : a = 1 * 8^3 + 4 * 8^2 + 5 * 8^1 + 3 * 8^0)
                            (h₂ : b = 5 * 8^2 + 6 * 8^1 + 7 * 8^0) :
                            ((a + b) = 2 * 8^3 + 1 * 8^2 + 4 * 8^1 + 4 * 8^0) →
                            (2 * 8^3 + 1 * 8^2 + 4 * 8^1 + 4 * 8^0 = 1124) :=
by {
  sorry
}

end base8_to_base10_sum_l19_19953


namespace alyssa_spent_on_grapes_l19_19215

theorem alyssa_spent_on_grapes (t c g : ℝ) (h1 : t = 21.93) (h2 : c = 9.85) (h3 : t = g + c) : g = 12.08 :=
by
  sorry

end alyssa_spent_on_grapes_l19_19215


namespace anna_final_stamp_count_l19_19635

theorem anna_final_stamp_count (anna_initial : ℕ) (alison_initial : ℕ) (jeff_initial : ℕ)
  (anna_receive_from_alison : ℕ) (anna_give_jeff : ℕ) (anna_receive_jeff : ℕ) :
  anna_initial = 37 →
  alison_initial = 28 →
  jeff_initial = 31 →
  anna_receive_from_alison = alison_initial / 2 →
  anna_give_jeff = 2 →
  anna_receive_jeff = 1 →
  ∃ result : ℕ, result = 50 :=
by
  intros
  sorry

end anna_final_stamp_count_l19_19635


namespace celine_change_l19_19945

theorem celine_change :
  let laptop_price := 600
  let smartphone_price := 400
  let tablet_price := 250
  let headphone_price := 100
  let laptops_purchased := 2
  let smartphones_purchased := 4
  let tablets_purchased := 3
  let headphones_purchased := 5
  let discount_rate := 0.10
  let sales_tax_rate := 0.05
  let initial_amount := 5000
  let laptop_total := laptops_purchased * laptop_price
  let smartphone_total := smartphones_purchased * smartphone_price
  let tablet_total := tablets_purchased * tablet_price
  let headphone_total := headphones_purchased * headphone_price
  let discount := discount_rate * (laptop_total + tablet_total)
  let total_before_discount := laptop_total + smartphone_total + tablet_total + headphone_total
  let total_after_discount := total_before_discount - discount
  let sales_tax := sales_tax_rate * total_after_discount
  let final_price := total_after_discount + sales_tax
  let change := initial_amount - final_price
  change = 952.25 :=
  sorry

end celine_change_l19_19945


namespace hilda_loan_compounding_difference_l19_19695

noncomputable def difference_due_to_compounding (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  let A_monthly := P * (1 + r / 12)^(12 * t)
  let A_annually := P * (1 + r)^t
  A_monthly - A_annually

theorem hilda_loan_compounding_difference :
  difference_due_to_compounding 8000 0.10 5 = 376.04 :=
sorry

end hilda_loan_compounding_difference_l19_19695


namespace probability_statements_l19_19130

-- Assigning probabilities
def p_hit := 0.9
def p_miss := 1 - p_hit

-- Definitions based on the problem conditions
def shoot_4_times (shots : List Bool) : Bool :=
  shots.length = 4 ∧ ∀ (s : Bool), s ∈ shots → (s = true → s ≠ false) ∧ (s = false → s ≠ true ∧ s ≠ 0)

-- Statements derived from the conditions
def prob_shot_3 := p_hit

def prob_exact_3_out_of_4 := 
  let binom_4_3 := 4
  binom_4_3 * (p_hit^3) * (p_miss^1)

def prob_at_least_1_out_of_4 := 1 - (p_miss^4)

-- The equivalence proof
theorem probability_statements : 
  (prob_shot_3 = 0.9) ∧ 
  (prob_exact_3_out_of_4 = 0.2916) ∧ 
  (prob_at_least_1_out_of_4 = 0.9999) := 
by 
  sorry

end probability_statements_l19_19130


namespace students_in_class_l19_19290

theorem students_in_class (S : ℕ) 
  (h1 : chess_students = S / 3)
  (h2 : tournament_students = chess_students / 2)
  (h3 : tournament_students = 4) : 
  S = 24 :=
by
  sorry

end students_in_class_l19_19290


namespace interval_of_increase_inequality_for_large_x_l19_19999

open Real

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + log x

theorem interval_of_increase :
  ∀ x > 0, ∀ y > x, f y > f x :=
by
  sorry

theorem inequality_for_large_x (x : ℝ) (hx : x > 1) :
  (1/2) * x^2 + log x < (2/3) * x^3 :=
by
  sorry

end interval_of_increase_inequality_for_large_x_l19_19999


namespace platform_length_calc_l19_19001

noncomputable def length_of_platform (V : ℝ) (T : ℝ) (L_train : ℝ) : ℝ :=
  (V * 1000 / 3600) * T - L_train

theorem platform_length_calc (speed : ℝ) (time : ℝ) (length_train : ℝ):
  speed = 72 →
  time = 26 →
  length_train = 280.0416 →
  length_of_platform speed time length_train = 239.9584 := by
  intros
  unfold length_of_platform
  sorry

end platform_length_calc_l19_19001


namespace find_number_l19_19875

theorem find_number (n : ℕ) :
  (n % 12 = 11) ∧
  (n % 11 = 10) ∧
  (n % 10 = 9) ∧
  (n % 9 = 8) ∧
  (n % 8 = 7) ∧
  (n % 7 = 6) ∧
  (n % 6 = 5) ∧
  (n % 5 = 4) ∧
  (n % 4 = 3) ∧
  (n % 3 = 2) ∧
  (n % 2 = 1)
  → n = 27719 := 
sorry

end find_number_l19_19875


namespace intersection_A_B_l19_19273

def A : Set ℝ := { x | x + 1 > 0 }
def B : Set ℝ := { x | x < 0 }

theorem intersection_A_B :
  A ∩ B = { x | -1 < x ∧ x < 0 } :=
sorry

end intersection_A_B_l19_19273


namespace james_prom_cost_l19_19869

def total_cost (ticket_cost dinner_cost tip_percent limo_cost_per_hour limo_hours tuxedo_cost persons : ℕ) : ℕ :=
  (ticket_cost * persons) +
  ((dinner_cost * persons) + (tip_percent * dinner_cost * persons) / 100) +
  (limo_cost_per_hour * limo_hours) + tuxedo_cost

theorem james_prom_cost :
  total_cost 100 120 30 80 8 150 4 = 1814 :=
by
  sorry

end james_prom_cost_l19_19869


namespace not_obtain_other_than_given_set_l19_19319

theorem not_obtain_other_than_given_set : 
  ∀ (x : ℝ), x = 1 → 
  ∃ (n : ℕ → ℝ), (n 0 = 1) ∧ 
  (∀ k, n (k + 1) = n k + 1 ∨ n (k + 1) = -1 / n k) ∧
  (x = -2 ∨ x = 1/2 ∨ x = 5/3 ∨ x = 7) → 
  ∃ k, x = n k :=
sorry

end not_obtain_other_than_given_set_l19_19319


namespace sufficient_not_necessary_condition_l19_19710

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x > 4 → x^2 - 4 * x > 0) ∧ ¬ (x^2 - 4 * x > 0 → x > 4) :=
sorry

end sufficient_not_necessary_condition_l19_19710


namespace sara_total_quarters_l19_19923

def initial_quarters : ℝ := 783.0
def given_quarters : ℝ := 271.0

theorem sara_total_quarters : initial_quarters + given_quarters = 1054.0 := 
by
  sorry

end sara_total_quarters_l19_19923


namespace find_interval_n_l19_19445

theorem find_interval_n 
  (n : ℕ) 
  (h1 : n < 500)
  (h2 : (∃ abcde : ℕ, 0 < abcde ∧ abcde < 99999 ∧ n * abcde = 99999))
  (h3 : (∃ uvw : ℕ, 0 < uvw ∧ uvw < 999 ∧ (n + 3) * uvw = 999)) 
  : 201 ≤ n ∧ n ≤ 300 := 
sorry

end find_interval_n_l19_19445


namespace wooden_block_length_l19_19942

-- Define the problem conditions
def meters_to_centimeters (m : ℕ) : ℕ := m * 100
def additional_length_cm (length_cm : ℕ) (additional_cm : ℕ) : ℕ := length_cm + additional_cm

-- Formalization of the problem
theorem wooden_block_length :
  let length_in_meters := 31
  let additional_cm := 30
  additional_length_cm (meters_to_centimeters length_in_meters) additional_cm = 3130 :=
by
  sorry

end wooden_block_length_l19_19942


namespace simplify_fraction_l19_19378

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) : (x^2 / (x - 2) - 4 / (x - 2)) = x + 2 := by
  sorry

end simplify_fraction_l19_19378


namespace hypotenuse_length_l19_19427

theorem hypotenuse_length
  (a b c : ℝ)
  (h1 : a + b + c = 40)
  (h2 : (1 / 2) * a * b = 24)
  (h3 : a^2 + b^2 = c^2) :
  c = 18.8 :=
by sorry

end hypotenuse_length_l19_19427


namespace exponential_function_range_l19_19166

noncomputable def exponential_function (a x : ℝ) : ℝ := a^x

theorem exponential_function_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : exponential_function a (-2) < exponential_function a (-3)) : 
  0 < a ∧ a < 1 :=
by
  sorry

end exponential_function_range_l19_19166


namespace eq_of_op_star_l19_19500

theorem eq_of_op_star (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^b^2)^n = a^(bn)^2 ↔ n = 1 := by
sorry

end eq_of_op_star_l19_19500


namespace intersection_A_B_l19_19670

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - x)}

theorem intersection_A_B : (A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_A_B_l19_19670


namespace shaded_area_of_octagon_l19_19359

noncomputable def areaOfShadedRegion (s : ℝ) (r : ℝ) (theta : ℝ) : ℝ :=
  let n := 8
  let octagonArea := n * 0.5 * s^2 * (Real.sin (Real.pi/n) / Real.sin (Real.pi/(2 * n)))
  let sectorArea := n * 0.5 * r^2 * (theta / (2 * Real.pi))
  octagonArea - sectorArea

theorem shaded_area_of_octagon (h_s : 5 = 5) (h_r : 3 = 3) (h_theta : 45 = 45) :
  areaOfShadedRegion 5 3 (45 * (Real.pi / 180)) = 100 - 9 * Real.pi := by
  sorry

end shaded_area_of_octagon_l19_19359


namespace abc_le_sqrt2_div_4_l19_19600

variable {a b c : ℝ}
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variable (h : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) = 1)

theorem abc_le_sqrt2_div_4 (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) = 1) :
  a * b * c ≤ (Real.sqrt 2) / 4 := 
sorry

end abc_le_sqrt2_div_4_l19_19600


namespace find_principal_l19_19759

theorem find_principal (CI SI : ℝ) (hCI : CI = 11730) (hSI : SI = 10200)
  (P R : ℝ)
  (hSI_form : SI = P * R * 2 / 100)
  (hCI_form : CI = P * (1 + R / 100)^2 - P) :
  P = 34000 := by
  sorry

end find_principal_l19_19759


namespace find_m_l19_19383

theorem find_m (m : ℝ) (h : (4 * (-1)^3 + 3 * m * (-1)^2 + 6 * (-1) = 2)) :
  m = 4 :=
by
  sorry

end find_m_l19_19383


namespace winnie_keeps_balloons_l19_19647

theorem winnie_keeps_balloons :
  let blueBalloons := 15
  let yellowBalloons := 40
  let purpleBalloons := 70
  let orangeBalloons := 90
  let friends := 9
  let totalBalloons := blueBalloons + yellowBalloons + purpleBalloons + orangeBalloons
  (totalBalloons % friends) = 8 := 
by 
  -- Definitions
  let blueBalloons := 15
  let yellowBalloons := 40
  let purpleBalloons := 70
  let orangeBalloons := 90
  let friends := 9
  let totalBalloons := blueBalloons + yellowBalloons + purpleBalloons + orangeBalloons
  -- Conclusion
  show totalBalloons % friends = 8
  sorry

end winnie_keeps_balloons_l19_19647


namespace solve_for_x_l19_19692

theorem solve_for_x (x : ℚ) (h : (x + 10) / (x - 4) = (x + 3) / (x - 6)) : x = 48 / 5 :=
sorry

end solve_for_x_l19_19692


namespace sum_of_five_consecutive_integers_l19_19308

theorem sum_of_five_consecutive_integers : ∀ (n : ℤ), (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 5 * n + 20 := 
by
  -- This would be where the proof goes
  sorry

end sum_of_five_consecutive_integers_l19_19308


namespace frequency_group_5_l19_19749

theorem frequency_group_5 (total_students : ℕ) (freq1 freq2 freq3 freq4 : ℕ)
  (h1 : total_students = 45)
  (h2 : freq1 = 12)
  (h3 : freq2 = 11)
  (h4 : freq3 = 9)
  (h5 : freq4 = 4) :
  ((total_students - (freq1 + freq2 + freq3 + freq4)) / total_students : ℚ) = 0.2 := 
sorry

end frequency_group_5_l19_19749


namespace molecular_weight_of_compound_l19_19473

theorem molecular_weight_of_compound :
  let Cu_atoms := 2
  let C_atoms := 3
  let O_atoms := 5
  let N_atoms := 1
  let atomic_weight_Cu := 63.546
  let atomic_weight_C := 12.011
  let atomic_weight_O := 15.999
  let atomic_weight_N := 14.007
  Cu_atoms * atomic_weight_Cu +
  C_atoms * atomic_weight_C +
  O_atoms * atomic_weight_O +
  N_atoms * atomic_weight_N = 257.127 :=
by
  sorry

end molecular_weight_of_compound_l19_19473


namespace term_100_is_981_l19_19283

def sequence_term (n : ℕ) : ℕ :=
  if n = 100 then 981 else sorry

theorem term_100_is_981 : sequence_term 100 = 981 := by
  rfl

end term_100_is_981_l19_19283


namespace solve_system_l19_19211

variable {R : Type*} [CommRing R]

-- Given conditions
variables (a b c x y z : R)

-- Assuming the given system of equations
axiom eq1 : x + a*y + a^2*z + a^3 = 0
axiom eq2 : x + b*y + b^2*z + b^3 = 0
axiom eq3 : x + c*y + c^2*z + c^3 = 0

-- The goal is to prove the mathematical equivalence
theorem solve_system : x = -a*b*c ∧ y = a*b + b*c + c*a ∧ z = -(a + b + c) :=
by
  sorry

end solve_system_l19_19211


namespace cube_of_square_is_15625_l19_19728

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l19_19728


namespace valid_passwords_count_l19_19861

-- Define the total number of unrestricted passwords (each digit can be 0-9)
def total_passwords := 10^5

-- Define the number of restricted passwords (those starting with the sequence 8,3,2)
def restricted_passwords := 10^2

-- State the main theorem to be proved
theorem valid_passwords_count : total_passwords - restricted_passwords = 99900 := by
  sorry

end valid_passwords_count_l19_19861


namespace product_greater_than_sum_l19_19407

variable {a b : ℝ}

theorem product_greater_than_sum (ha : a > 2) (hb : b > 2) : a * b > a + b := 
  sorry

end product_greater_than_sum_l19_19407


namespace expression_in_terms_of_p_q_l19_19019

variables {α β γ δ p q : ℝ}

-- Let α and β be the roots of x^2 - 2px + 1 = 0
axiom root_α_β : ∀ x, (x - α) * (x - β) = x^2 - 2 * p * x + 1

-- Let γ and δ be the roots of x^2 + qx + 2 = 0
axiom root_γ_δ : ∀ x, (x - γ) * (x - δ) = x^2 + q * x + 2

-- Expression to be proved
theorem expression_in_terms_of_p_q :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = 2 * (p - q) ^ 2 :=
sorry

end expression_in_terms_of_p_q_l19_19019


namespace right_triangle_third_side_l19_19642

/-- In a right triangle, given the lengths of two sides are 4 and 5, prove that the length of the
third side is either sqrt 41 or 3. -/
theorem right_triangle_third_side (a b : ℕ) (h1 : a = 4 ∨ a = 5) (h2 : b = 4 ∨ b = 5) (h3 : a ≠ b) :
  ∃ c, c = Real.sqrt 41 ∨ c = 3 :=
by
  sorry

end right_triangle_third_side_l19_19642


namespace number_of_points_l19_19065

theorem number_of_points (x y : ℕ) (h : y = (2 * x + 2018) / (x - 1)) 
  (h2 : x > y) (h3 : 0 < x) (h4 : 0 < y) : 
  ∃! (x y : ℕ), y = (2 * x + 2018) / (x - 1) ∧ x > y ∧ 0 < x ∧ 0 < y :=
sorry

end number_of_points_l19_19065


namespace pizza_problem_l19_19221

theorem pizza_problem (m d : ℕ) :
  (7 * m + 2 * d > 36) ∧ (8 * m + 4 * d < 48) ↔ (m = 5) ∧ (d = 1) := by
  sorry

end pizza_problem_l19_19221


namespace average_transformation_l19_19637

theorem average_transformation (a b c : ℝ) (h : (a + b + c) / 3 = 12) : ((2 * a + 1) + (2 * b + 2) + (2 * c + 3) + 2) / 4 = 20 :=
by
  sorry

end average_transformation_l19_19637


namespace unique_function_solution_l19_19594

theorem unique_function_solution :
  ∀ f : ℕ+ → ℕ+, (∀ x y : ℕ+, f (x + y * f x) = x * f (y + 1)) → (∀ x : ℕ+, f x = x) :=
by
  sorry

end unique_function_solution_l19_19594


namespace sufficient_not_necessary_l19_19997

variable (x : ℝ)
def p := x^2 > 4
def q := x > 2

theorem sufficient_not_necessary : (∀ x, q x -> p x) ∧ ¬ (∀ x, p x -> q x) :=
by sorry

end sufficient_not_necessary_l19_19997


namespace problem_solution_l19_19470

variable (f : ℝ → ℝ)

-- Let f be an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- f(x) = f(4 - x) for all x in ℝ
def satisfies_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (4 - x)

-- f is increasing on [0, 2]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem problem_solution :
  is_odd_function f →
  satisfies_symmetry f →
  is_increasing_on_interval f 0 2 →
  f 6 < f 4 ∧ f 4 < f 1 :=
by
  intros
  sorry

end problem_solution_l19_19470


namespace trish_walks_l19_19057

variable (n : ℕ) (M D : ℝ)
variable (d : ℕ → ℝ)
variable (H1 : d 1 = 1)
variable (H2 : ∀ k : ℕ, d (k + 1) = 2 * d k)
variable (H3 : d n > M)

theorem trish_walks (n : ℕ) (M : ℝ) (H1 : d 1 = 1) (H2 : ∀ k : ℕ, d (k + 1) = 2 * d k) (H3 : d n > M) : 2^(n-1) > M := by
  sorry

end trish_walks_l19_19057


namespace yura_catches_up_l19_19327

theorem yura_catches_up (a : ℕ) (x : ℕ) (h1 : 2 * a * x = a * (x + 5)) : x = 5 :=
by
  sorry

end yura_catches_up_l19_19327


namespace area_of_storm_eye_l19_19042

theorem area_of_storm_eye : 
  let large_quarter_circle_area := (1 / 4) * π * 5^2
  let small_circle_area := π * 2^2
  let storm_eye_area := large_quarter_circle_area - small_circle_area
  storm_eye_area = (9 * π) / 4 :=
by
  sorry

end area_of_storm_eye_l19_19042


namespace seashells_remaining_l19_19094

def initial_seashells : ℕ := 35
def given_seashells : ℕ := 18

theorem seashells_remaining : initial_seashells - given_seashells = 17 := by
  sorry

end seashells_remaining_l19_19094


namespace A_beats_B_by_63_l19_19463

variable (A B C : ℕ)

-- Condition: A beats C by 163 meters
def A_beats_C : Prop := A = 1000 - 163
-- Condition: B beats C by 100 meters
def B_beats_C (X : ℕ) : Prop := 1000 - X = 837 + 100
-- Main theorem statement
theorem A_beats_B_by_63 (X : ℕ) (h1 : A_beats_C A) (h2 : B_beats_C X): X = 63 :=
by
  sorry

end A_beats_B_by_63_l19_19463


namespace product_of_roots_l19_19279

theorem product_of_roots : ∀ (x : ℝ), (x + 3) * (x - 4) = 2 * (x + 1) → 
  let a := 1
  let b := -3
  let c := -14
  let product_of_roots := c / a
  product_of_roots = -14 :=
by
  intros x h
  let a := 1
  let b := -3
  let c := -14
  let product_of_roots := c / a
  sorry

end product_of_roots_l19_19279


namespace problem1_problem2_l19_19747

def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem problem1 (x : ℝ) : f x ≥ 4 ↔ x ≤ -4/3 ∨ x ≥ 4/3 := 
  sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, f x > a) ↔ a < 3/2 := 
  sorry

end problem1_problem2_l19_19747


namespace solve_system_of_equations_l19_19197

def system_solution : Prop := ∃ x y : ℚ, 4 * x - 6 * y = -14 ∧ 8 * x + 3 * y = -15 ∧ x = -11 / 5 ∧ y = 2.6 / 3

theorem solve_system_of_equations : system_solution := sorry

end solve_system_of_equations_l19_19197


namespace find_angle_FYD_l19_19379

noncomputable def angle_FYD (AB CD AXF FYG : ℝ) : ℝ := 180 - AXF

theorem find_angle_FYD (AB CD : ℝ) (AXF : ℝ) (FYG : ℝ) (h1 : AB = CD) (h2 : AXF = 125) (h3 : FYG = 40) :
  angle_FYD AB CD AXF FYG = 55 :=
by
  sorry

end find_angle_FYD_l19_19379


namespace percentage_of_fruits_in_good_condition_l19_19746

theorem percentage_of_fruits_in_good_condition :
  let total_oranges := 600
  let total_bananas := 400
  let rotten_oranges := (15 / 100.0) * total_oranges
  let rotten_bananas := (8 / 100.0) * total_bananas
  let good_condition_oranges := total_oranges - rotten_oranges
  let good_condition_bananas := total_bananas - rotten_bananas
  let total_fruits := total_oranges + total_bananas
  let total_fruits_in_good_condition := good_condition_oranges + good_condition_bananas
  let percentage_fruits_in_good_condition := (total_fruits_in_good_condition / total_fruits) * 100
  percentage_fruits_in_good_condition = 87.8 := sorry

end percentage_of_fruits_in_good_condition_l19_19746


namespace find_arithmetic_progression_terms_l19_19342

noncomputable def arithmetic_progression_terms (a1 a2 a3 : ℕ) (d : ℕ) 
  (condition1 : a1 + (a1 + d) = 3 * 2^2) 
  (condition2 : a1 + (a1 + d) + (a1 + 2 * d) = 3 * 3^2) : Prop := 
  a1 = 3 ∧ a2 = 9 ∧ a3 = 15

theorem find_arithmetic_progression_terms
  (a1 a2 a3 : ℕ) (d : ℕ)
  (cond1 : a1 + (a1 + d) = 3 * 2^2)
  (cond2 : a1 + (a1 + d) + (a1 + 2 * d) = 3 * 3^2) :
  arithmetic_progression_terms a1 a2 a3 d cond1 cond2 :=
sorry

end find_arithmetic_progression_terms_l19_19342


namespace quadratic_root_3_m_value_l19_19182

theorem quadratic_root_3_m_value (m : ℝ) : (∃ x : ℝ, 2*x*x - m*x + 3 = 0 ∧ x = 3) → m = 7 :=
by
  sorry

end quadratic_root_3_m_value_l19_19182


namespace student_finished_6_problems_in_class_l19_19820

theorem student_finished_6_problems_in_class (total_problems : ℕ) (x y : ℕ) (h1 : total_problems = 15) (h2 : 3 * y = 2 * x) (h3 : x + y = total_problems) : y = 6 :=
sorry

end student_finished_6_problems_in_class_l19_19820


namespace find_p1_plus_q1_l19_19435

noncomputable def p (x : ℤ) := x^4 + 14 * x^2 + 1
noncomputable def q (x : ℤ) := x^4 - 14 * x^2 + 1

theorem find_p1_plus_q1 :
  (p 1) + (q 1) = 4 :=
sorry

end find_p1_plus_q1_l19_19435


namespace city_population_distribution_l19_19562

theorem city_population_distribution :
  (20 + 35) = 55 :=
by
  sorry

end city_population_distribution_l19_19562


namespace remainder_is_zero_l19_19329

def remainder_when_multiplied_then_subtracted (a b : ℕ) : ℕ :=
  (a * b - 8) % 8

theorem remainder_is_zero : remainder_when_multiplied_then_subtracted 104 106 = 0 := by
  sorry

end remainder_is_zero_l19_19329


namespace prove_partial_fractions_identity_l19_19207

def partial_fraction_identity (x : ℚ) (A B C a b c : ℚ) : Prop :=
  a = 0 ∧ b = 1 ∧ c = -1 ∧
  (A / (x - a) + B / (x - b) + C / (x - c) = 4*x - 2 ∧ x^3 - x ≠ 0)

theorem prove_partial_fractions_identity :
  (partial_fraction_identity x 2 1 (-3) 0 1 (-1)) :=
by {
  sorry
}

end prove_partial_fractions_identity_l19_19207


namespace last_three_digits_of_2_pow_6000_l19_19266

theorem last_three_digits_of_2_pow_6000 (h : 2^200 ≡ 1 [MOD 800]) : (2^6000 ≡ 1 [MOD 800]) :=
sorry

end last_three_digits_of_2_pow_6000_l19_19266


namespace prime_triple_l19_19403

theorem prime_triple (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (h1 : p ∣ (q * r - 1)) (h2 : q ∣ (p * r - 1)) (h3 : r ∣ (p * q - 1)) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
sorry

end prime_triple_l19_19403


namespace arccos_sin_three_l19_19675

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 :=
by
  sorry

end arccos_sin_three_l19_19675


namespace simplify_expr_l19_19152

noncomputable def expr : ℝ := Real.sqrt 12 - 3 * Real.sqrt (1 / 3) + Real.sqrt 27 + (Real.pi + 1)^0

theorem simplify_expr : expr = 4 * Real.sqrt 3 + 1 := by
  sorry

end simplify_expr_l19_19152


namespace sufficient_but_not_necessary_condition_l19_19526

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (1 ≤ x ∧ x ≤ 4) ↔ (1 ≤ x^2 ∧ x^2 ≤ 16) :=
by
  sorry

end sufficient_but_not_necessary_condition_l19_19526


namespace line_passes_through_fixed_point_minimum_area_triangle_l19_19049

theorem line_passes_through_fixed_point (k : ℝ) :
  ∃ P : ℝ × ℝ, P = (2, 1) ∧ ∀ k : ℝ, (k * 2 - 1 + 1 - 2 * k = 0) :=
sorry

theorem minimum_area_triangle (k : ℝ) :
  ∀ k: ℝ, k < 0 → 1/2 * (2 - 1/k) * (1 - 2*k) ≥ 4 ∧ 
           (1/2 * (2 - 1/k) * (1 - 2*k) = 4 ↔ k = -1/2) :=
sorry

end line_passes_through_fixed_point_minimum_area_triangle_l19_19049


namespace triangle_third_side_one_third_perimeter_l19_19552

theorem triangle_third_side_one_third_perimeter
  (a b x y p c : ℝ)
  (h1 : x^2 - y^2 = a^2 - b^2)
  (h2 : p = (a + b + c) / 2)
  (h3 : x - y = 2 * (a - b)) :
  c = (a + b + c) / 3 := by
  sorry

end triangle_third_side_one_third_perimeter_l19_19552


namespace calculate_overhead_cost_l19_19862

noncomputable def overhead_cost (prod_cost revenue_cost : ℕ) (num_performances : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost - num_performances * prod_cost

theorem calculate_overhead_cost :
  overhead_cost 7000 16000 9 (9 * 16000) = 81000 :=
by
  sorry

end calculate_overhead_cost_l19_19862


namespace house_to_market_distance_l19_19497

-- Definitions of the conditions
def distance_to_school : ℕ := 50
def distance_back_home : ℕ := 50
def total_distance_walked : ℕ := 140

-- Statement of the problem
theorem house_to_market_distance :
  distance_to_market = total_distance_walked - (distance_to_school + distance_back_home) :=
by
  sorry

end house_to_market_distance_l19_19497


namespace cakes_given_away_l19_19257

theorem cakes_given_away 
  (cakes_baked : ℕ) 
  (candles_per_cake : ℕ) 
  (total_candles : ℕ) 
  (cakes_given : ℕ) 
  (cakes_left : ℕ) 
  (h1 : cakes_baked = 8) 
  (h2 : candles_per_cake = 6) 
  (h3 : total_candles = 36) 
  (h4 : total_candles = candles_per_cake * cakes_left) 
  (h5 : cakes_given = cakes_baked - cakes_left) 
  : cakes_given = 2 :=
sorry

end cakes_given_away_l19_19257


namespace minimum_trees_with_at_least_three_types_l19_19441

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l19_19441


namespace min_value_fraction_sum_l19_19212

theorem min_value_fraction_sum : 
  ∀ (n : ℕ), n > 0 → (n / 3 + 27 / n) ≥ 6 :=
by
  sorry

end min_value_fraction_sum_l19_19212


namespace distance_from_A_to_O_is_3_l19_19397

-- Define polar coordinates with the given conditions
def point_A : ℝ × ℝ := (3, -4)

-- Define the distance function in terms of polar coordinates
def distance_to_pole_O (coords : ℝ × ℝ) : ℝ := coords.1

-- The main theorem to be proved
theorem distance_from_A_to_O_is_3 : distance_to_pole_O point_A = 3 := by
  sorry

end distance_from_A_to_O_is_3_l19_19397


namespace mean_three_numbers_l19_19878

open BigOperators

theorem mean_three_numbers (a b c : ℝ) (s : Finset ℝ) (h₀ : s.card = 20)
  (h₁ : (∑ x in s, x) / 20 = 45) 
  (h₂ : (∑ x in s ∪ {a, b, c}, x) / 23 = 50) : 
  (a + b + c) / 3 = 250 / 3 :=
by
  sorry

end mean_three_numbers_l19_19878


namespace increase_average_l19_19868

variable (total_runs : ℕ) (innings : ℕ) (average : ℕ) (new_runs : ℕ) (x : ℕ)

theorem increase_average (h1 : innings = 10) 
                         (h2 : average = 30) 
                         (h3 : total_runs = average * innings) 
                         (h4 : new_runs = 74) 
                         (h5 : total_runs + new_runs = (average + x) * (innings + 1)) :
    x = 4 := 
sorry

end increase_average_l19_19868


namespace average_last_three_l19_19515

noncomputable def average (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem average_last_three (l : List ℝ) (h₁ : l.length = 7) (h₂ : average l = 62) 
  (h₃ : average (l.take 4) = 58) :
  average (l.drop 4) = 202 / 3 := 
by 
  sorry

end average_last_three_l19_19515


namespace amy_seeds_l19_19727

-- Define the conditions
def bigGardenSeeds : Nat := 47
def smallGardens : Nat := 9
def seedsPerSmallGarden : Nat := 6

-- Define the total seeds calculation
def totalSeeds := bigGardenSeeds + smallGardens * seedsPerSmallGarden

-- The theorem to be proved
theorem amy_seeds : totalSeeds = 101 := by
  sorry

end amy_seeds_l19_19727


namespace man_salary_problem_l19_19349

-- Define the problem in Lean 4
theorem man_salary_problem (S : ℝ) :
  (1/3 * S) + (1/4 * S) + (1/5 * S) + 1760 = S → 
  S = 8123.08 :=
sorry

end man_salary_problem_l19_19349


namespace meaningful_iff_x_ne_1_l19_19305

theorem meaningful_iff_x_ne_1 (x : ℝ) : (x - 1) ≠ 0 ↔ (x ≠ 1) :=
by 
  sorry

end meaningful_iff_x_ne_1_l19_19305


namespace total_games_l19_19111

/-- Definition of the number of games Alyssa went to this year -/
def games_this_year : Nat := 11

/-- Definition of the number of games Alyssa went to last year -/
def games_last_year : Nat := 13

/-- Definition of the number of games Alyssa plans to go to next year -/
def games_next_year : Nat := 15

/-- Statement to prove the total number of games Alyssa will go to in all -/
theorem total_games : games_this_year + games_last_year + games_next_year = 39 := by
  -- A sorry placeholder to skip the proof
  sorry

end total_games_l19_19111


namespace range_of_a_l19_19080

theorem range_of_a (a : ℝ) : 
  (∀ x, (x > 2 ∨ x < -1) → ¬(x^2 + 4 * x + a < 0)) → a ≥ 3 :=
by
  sorry

end range_of_a_l19_19080


namespace prob_less_than_8_prob_at_least_7_l19_19883

def prob_9_or_above : ℝ := 0.56
def prob_8 : ℝ := 0.22
def prob_7 : ℝ := 0.12

theorem prob_less_than_8 : prob_7 + (1 - prob_9_or_above - prob_8) = 0.22 := 
sorry

theorem prob_at_least_7 : prob_9_or_above + prob_8 + prob_7 = 0.9 := 
sorry

end prob_less_than_8_prob_at_least_7_l19_19883


namespace intersection_A_B_l19_19860

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | x < -1 ∨ x > 4}

theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 4 < x ∧ x < 7} :=
by
  sorry

end intersection_A_B_l19_19860


namespace find_line_equation_l19_19691

theorem find_line_equation (k x y x₁ y₁ x₂ y₂ : ℝ) (h_parabola : y ^ 2 = 2 * x) 
  (h_line_ny_eq : y = k * x + 2) (h_intersect_1 : (y₁ - (k * x₁ + 2)) = 0)
  (h_intersect_2 : (y₂ - (k * x₂ + 2)) = 0) 
  (h_y_intercept : (0,2) = (x,y))-- the line has y-intercept 2 
  (h_origin : (0,0) = (x, y)) -- origin 
  (h_orthogonal : x₁ * x₂ + y₁ * y₂ = 0): 
  y = -x + 2 :=
by {
  sorry
}

end find_line_equation_l19_19691


namespace statement_a_statement_b_statement_c_l19_19891

theorem statement_a (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  0 ≤ a ∧ a ≤ 4 := sorry

theorem statement_b (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  -1 ≤ b ∧ b ≤ 3 := sorry

theorem statement_c (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  -2 ≤ 3 * a - 2 * b ∧ 3 * a - 2 * b ≤ 10 := sorry

end statement_a_statement_b_statement_c_l19_19891


namespace arcsin_eq_solution_domain_l19_19853

open Real

theorem arcsin_eq_solution_domain (x : ℝ) (hx1 : abs (x * sqrt 5 / 3) ≤ 1)
  (hx2 : abs (x * sqrt 5 / 6) ≤ 1)
  (hx3 : abs (7 * x * sqrt 5 / 18) ≤ 1) :
  arcsin (x * sqrt 5 / 3) + arcsin (x * sqrt 5 / 6) = arcsin (7 * x * sqrt 5 / 18) ↔ 
  x = 0 ∨ x = 8 / 7 ∨ x = -8 / 7 := sorry

end arcsin_eq_solution_domain_l19_19853


namespace range_of_k_l19_19241

theorem range_of_k (k : Real) : 
  (∀ (x y : Real), x^2 + y^2 - 12 * x - 4 * y + 37 = 0)
  → ((k < -Real.sqrt 2) ∨ (k > Real.sqrt 2)) :=
by
  sorry

end range_of_k_l19_19241


namespace intersection_point_l19_19486

noncomputable def f (x : ℝ) := (x^2 - 8 * x + 7) / (2 * x - 6)

noncomputable def g (a b c : ℝ) (x : ℝ) := (a * x^2 + b * x + c) / (x - 3)

theorem intersection_point (a b c : ℝ) :
  (∀ x, 2 * x - 6 = 0 <-> x ≠ 3) →
  ∃ (k : ℝ), (g a b c x = -2 * x - 4 + k / (x - 3)) →
  (f x = g a b c x) ∧ x ≠ -3 → x = 1 ∧ f 1 = 0 :=
by
  intros
  sorry

end intersection_point_l19_19486


namespace johns_total_money_l19_19597

-- Defining the given conditions
def initial_amount : ℕ := 5
def amount_spent : ℕ := 2
def allowance : ℕ := 26

-- Constructing the proof statement
theorem johns_total_money : initial_amount - amount_spent + allowance = 29 :=
by
  sorry

end johns_total_money_l19_19597


namespace brian_total_video_length_l19_19371

theorem brian_total_video_length :
  let cat_length := 4
  let dog_length := 2 * cat_length
  let gorilla_length := cat_length ^ 2
  let elephant_length := cat_length + dog_length + gorilla_length
  let cat_dog_gorilla_elephant_sum := cat_length + dog_length + gorilla_length + elephant_length
  let penguin_length := cat_dog_gorilla_elephant_sum ^ 3
  let dolphin_length := cat_length + dog_length + gorilla_length + elephant_length + penguin_length
  let total_length := cat_length + dog_length + gorilla_length + elephant_length + penguin_length + dolphin_length
  total_length = 351344 := by
    sorry

end brian_total_video_length_l19_19371


namespace min_value_expression_l19_19006

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : y = Real.sqrt x) :
  ∃ c, c = 2 ∧ ∀ u v : ℝ, 0 < u → v = Real.sqrt u → (u^2 + v^4) / (u * v^2) = c :=
by
  sorry

end min_value_expression_l19_19006


namespace total_boys_fraction_of_girls_l19_19837

theorem total_boys_fraction_of_girls
  (n : ℕ)
  (b1 g1 b2 g2 : ℕ)
  (h_equal_students : b1 + g1 = b2 + g2)
  (h_ratio_class1 : b1 / g1 = 2 / 3)
  (h_ratio_class2: b2 / g2 = 4 / 5) :
  ((b1 + b2) / (g1 + g2) = 19 / 26) :=
by sorry

end total_boys_fraction_of_girls_l19_19837


namespace diameter_percentage_l19_19633

theorem diameter_percentage (d_R d_S : ℝ) (h : π * (d_R / 2)^2 = 0.16 * π * (d_S / 2)^2) :
  (d_R / d_S) * 100 = 40 :=
by {
  sorry
}

end diameter_percentage_l19_19633


namespace height_of_pyramid_l19_19639

theorem height_of_pyramid :
  let edge_cube := 6
  let edge_base_square_pyramid := 10
  let cube_volume := edge_cube ^ 3
  let sphere_volume := cube_volume
  let pyramid_volume := 2 * sphere_volume
  let base_area_square_pyramid := edge_base_square_pyramid ^ 2
  let height_pyramid := 12.96
  pyramid_volume = (1 / 3) * base_area_square_pyramid * height_pyramid :=
by
  sorry

end height_of_pyramid_l19_19639


namespace boys_count_at_table_l19_19920

-- Definitions from conditions
def children_count : ℕ := 13
def alternates (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- The problem to be proven in Lean:
theorem boys_count_at_table : ∃ b g : ℕ, b + g = children_count ∧ alternates b ∧ alternates g ∧ b = 7 :=
by
  sorry

end boys_count_at_table_l19_19920


namespace arithmetic_sequence_sum_19_l19_19650

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_19 (h1 : is_arithmetic_sequence a)
  (h2 : a 9 = 11) (h3 : a 11 = 9) (h4 : ∀ n, S n = n / 2 * (a 1 + a n)) :
  S 19 = 190 :=
sorry

end arithmetic_sequence_sum_19_l19_19650


namespace distance_between_neg2_and_3_l19_19031
-- Import the necessary Lean libraries

-- State the theorem to prove the distance between -2 and 3 is 5
theorem distance_between_neg2_and_3 : abs (3 - (-2)) = 5 := by
  sorry

end distance_between_neg2_and_3_l19_19031


namespace value_of_trig_expression_l19_19469

theorem value_of_trig_expression (α : Real) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -3 :=
by 
  sorry

end value_of_trig_expression_l19_19469


namespace total_boxes_count_l19_19014

theorem total_boxes_count 
    (apples_per_crate : ℕ) (apples_crates : ℕ) 
    (oranges_per_crate : ℕ) (oranges_crates : ℕ) 
    (bananas_per_crate : ℕ) (bananas_crates : ℕ) 
    (rotten_apples_percentage : ℝ) (rotten_oranges_percentage : ℝ) (rotten_bananas_percentage : ℝ)
    (apples_per_box : ℕ) (oranges_per_box : ℕ) (bananas_per_box : ℕ) :
    apples_per_crate = 42 → apples_crates = 12 → 
    oranges_per_crate = 36 → oranges_crates = 15 → 
    bananas_per_crate = 30 → bananas_crates = 18 → 
    rotten_apples_percentage = 0.08 → rotten_oranges_percentage = 0.05 → rotten_bananas_percentage = 0.02 →
    apples_per_box = 10 → oranges_per_box = 12 → bananas_per_box = 15 →
    ∃ total_boxes : ℕ, total_boxes = 126 :=
by sorry

end total_boxes_count_l19_19014


namespace elois_made_3_loaves_on_Monday_l19_19909

theorem elois_made_3_loaves_on_Monday
    (bananas_per_loaf : ℕ)
    (twice_as_many : ℕ)
    (total_bananas : ℕ) 
    (h1 : bananas_per_loaf = 4) 
    (h2 : twice_as_many = 2) 
    (h3 : total_bananas = 36)
  : ∃ L : ℕ, (4 * L + 8 * L = 36) ∧ L = 3 :=
sorry

end elois_made_3_loaves_on_Monday_l19_19909


namespace part_one_a_two_complement_union_part_one_a_two_complement_intersection_part_two_subset_l19_19227

open Set Real

def setA (a : ℝ) : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ a + 5}
def setB : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem part_one_a_two_complement_union (a : ℝ) (h : a = 2) :
  compl (setA a ∪ setB) = Iic 2 ∪ Ici 10 := sorry

theorem part_one_a_two_complement_intersection (a : ℝ) (h : a = 2) :
  compl (setA a) ∩ setB = Ioo 2 3 ∪ Ioo 7 10 := sorry

theorem part_two_subset (a : ℝ) (h : setA a ⊆ setB) :
  a < 5 := sorry

end part_one_a_two_complement_union_part_one_a_two_complement_intersection_part_two_subset_l19_19227


namespace children_exceed_bridge_limit_l19_19005

theorem children_exceed_bridge_limit :
  ∀ (Kelly_weight : ℕ) (Megan_weight : ℕ) (Mike_weight : ℕ),
  Kelly_weight = 34 ∧
  Kelly_weight = (85 * Megan_weight) / 100 ∧
  Mike_weight = Megan_weight + 5 →
  Kelly_weight + Megan_weight + Mike_weight - 100 = 19 :=
by sorry

end children_exceed_bridge_limit_l19_19005


namespace marlon_gift_card_balance_l19_19847

theorem marlon_gift_card_balance 
  (initial_amount : ℕ) 
  (spent_monday : initial_amount / 2 = 100)
  (spent_tuesday : (initial_amount / 2) / 4 = 25) 
  : (initial_amount / 2) - (initial_amount / 2 / 4) = 75 :=
by
  sorry

end marlon_gift_card_balance_l19_19847


namespace zionsDadX_l19_19689

section ZionProblem

-- Define the conditions
variables (Z : ℕ) (D : ℕ) (X : ℕ)

-- Zion's current age
def ZionAge : Prop := Z = 8

-- Zion's dad's age in terms of Zion's age and X
def DadsAge : Prop := D = 4 * Z + X

-- Zion's dad's age in 10 years compared to Zion's age in 10 years
def AgeInTenYears : Prop := D + 10 = (Z + 10) + 27

-- The theorem statement to be proved
theorem zionsDadX :
  ZionAge Z →  
  DadsAge Z D X →  
  AgeInTenYears Z D →  
  X = 3 := 
sorry

end ZionProblem

end zionsDadX_l19_19689


namespace parabola_opens_upward_l19_19176

structure QuadraticFunction :=
  (a b c : ℝ)

def quadratic_y (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

def points : List (ℝ × ℝ) :=
  [(-1, 10), (0, 5), (1, 2), (2, 1), (3, 2)]

theorem parabola_opens_upward (f : QuadraticFunction)
  (h_values : ∀ (x : ℝ), (x, quadratic_y f x) ∈ points) :
  f.a > 0 :=
sorry

end parabola_opens_upward_l19_19176


namespace jose_is_12_years_older_l19_19132

theorem jose_is_12_years_older (J M : ℕ) (h1 : M = 14) (h2 : J + M = 40) : J - M = 12 :=
by
  sorry

end jose_is_12_years_older_l19_19132


namespace sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three_l19_19777

theorem sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three : 
  ((Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) = 3) := by
  sorry

end sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three_l19_19777


namespace find_k_l19_19754

def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x + y - 2 = 0
def quadrilateral_has_circumscribed_circle (k : ℝ) : Prop :=
  ∀ x y : ℝ, line1 x y → line2 k x y →
  k = -3

theorem find_k (k : ℝ) (x y : ℝ) : 
  (line1 x y) ∧ (line2 k x y) → quadrilateral_has_circumscribed_circle k :=
by 
  sorry

end find_k_l19_19754


namespace probability_cello_viola_same_tree_l19_19391

noncomputable section

def cellos : ℕ := 800
def violas : ℕ := 600
def cello_viola_pairs_same_tree : ℕ := 100

theorem probability_cello_viola_same_tree : 
  (cello_viola_pairs_same_tree: ℝ) / ((cellos * violas : ℕ) : ℝ) = 1 / 4800 := 
by
  sorry

end probability_cello_viola_same_tree_l19_19391


namespace line_circle_separate_l19_19984

def point_inside_circle (x0 y0 a : ℝ) : Prop :=
  x0^2 + y0^2 < a^2

def not_center_of_circle (x0 y0 : ℝ) : Prop :=
  x0^2 + y0^2 ≠ 0

theorem line_circle_separate (x0 y0 a : ℝ) (h1 : point_inside_circle x0 y0 a) (h2 : a > 0) (h3 : not_center_of_circle x0 y0) :
  ∀ (x y : ℝ), ¬ (x0 * x + y0 * y = a^2 ∧ x^2 + y^2 = a^2) :=
by
  sorry

end line_circle_separate_l19_19984


namespace num_convex_pentagons_l19_19219

theorem num_convex_pentagons (n m : ℕ) (hn : n = 15) (hm : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end num_convex_pentagons_l19_19219


namespace increase_by_percentage_l19_19940

def initial_value : ℕ := 550
def percentage_increase : ℚ := 0.35
def final_value : ℚ := 742.5

theorem increase_by_percentage :
  (initial_value : ℚ) * (1 + percentage_increase) = final_value := by
  sorry

end increase_by_percentage_l19_19940


namespace bottle_caps_per_visit_l19_19418

-- Define the given conditions
def total_bottle_caps : ℕ := 25
def number_of_visits : ℕ := 5

-- The statement we want to prove
theorem bottle_caps_per_visit :
  total_bottle_caps / number_of_visits = 5 :=
sorry

end bottle_caps_per_visit_l19_19418


namespace total_books_l19_19964

variable (a : ℕ)

theorem total_books (h₁ : 5 = 5) (h₂ : a = a) : 5 + a = 5 + a :=
by
  sorry

end total_books_l19_19964


namespace least_number_to_add_l19_19366

theorem least_number_to_add {n : ℕ} (h : n = 1202) : (∃ k : ℕ, (n + k) % 4 = 0 ∧ ∀ m : ℕ, (m < k → (n + m) % 4 ≠ 0)) ∧ k = 2 := by
  sorry

end least_number_to_add_l19_19366


namespace find_last_score_l19_19758

/-- The list of scores in ascending order -/
def scores : List ℕ := [60, 65, 70, 75, 80, 85, 95]

/--
  The problem states that the average score after each entry is an integer.
  Given the scores in ascending order, determine the last score entered.
-/
theorem find_last_score (h : ∀ (n : ℕ) (hn : n < scores.length),
    (scores.take (n + 1) |>.sum : ℤ) % (n + 1) = 0) :
  scores.last' = some 80 :=
sorry

end find_last_score_l19_19758


namespace sin_subtract_of_obtuse_angle_l19_19580

open Real -- Open the Real namespace for convenience.

theorem sin_subtract_of_obtuse_angle (α : ℝ) 
  (h1 : (π / 2) < α) (h2 : α < π)
  (h3 : sin (π / 4 + α) = 3 / 4)
  : sin (π / 4 - α) = - (sqrt 7) / 4 := 
by 
  sorry -- Proof placeholder.

end sin_subtract_of_obtuse_angle_l19_19580


namespace inequality_holds_l19_19730

variables (a b c : ℝ)

theorem inequality_holds 
  (h1 : a > b) : 
  a / (c^2 + 1) > b / (c^2 + 1) :=
sorry

end inequality_holds_l19_19730


namespace remainder_abc_div9_l19_19148

theorem remainder_abc_div9 (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9) 
    (h1 : a + 2 * b + 3 * c ≡ 0 [MOD 9]) 
    (h2 : 2 * a + 3 * b + c ≡ 5 [MOD 9]) 
    (h3 : 3 * a + b + 2 * c ≡ 5 [MOD 9]) : 
    (a * b * c) % 9 = 0 := 
sorry

end remainder_abc_div9_l19_19148


namespace delta_five_three_l19_19363

def Δ (a b : ℕ) : ℕ := 4 * a - 6 * b

theorem delta_five_three :
  Δ 5 3 = 2 := by
  sorry

end delta_five_three_l19_19363


namespace functional_equation_solutions_l19_19679

theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) : 
  (∀ x : ℝ, f x = 0) ∨
  (∀ x : ℝ, f x = x - 1) ∨
  (∀ x : ℝ, f x = 1 - x) :=
sorry

end functional_equation_solutions_l19_19679


namespace total_time_outside_class_l19_19096

-- Definitions based on given conditions
def first_recess : ℕ := 15
def second_recess : ℕ := 15
def lunch : ℕ := 30
def third_recess : ℕ := 20

-- Proof problem statement
theorem total_time_outside_class : first_recess + second_recess + lunch + third_recess = 80 := 
by sorry

end total_time_outside_class_l19_19096


namespace total_cost_of_motorcycle_l19_19437

-- Definitions from conditions
def total_cost (x : ℝ) := 0.20 * x = 400

-- The theorem to prove
theorem total_cost_of_motorcycle (x : ℝ) (h : total_cost x) : x = 2000 := 
by
  sorry

end total_cost_of_motorcycle_l19_19437


namespace find_angle_y_l19_19658

open Real

theorem find_angle_y 
    (angle_ABC angle_BAC : ℝ)
    (h1 : angle_ABC = 70)
    (h2 : angle_BAC = 50)
    (triangle_sum : ∀ {A B C : ℝ}, A + B + C = 180)
    (right_triangle_sum : ∀ D E : ℝ, D + E = 90) :
    30 = 30 :=
by
    -- Given, conditions, and intermediate results (skipped)
    sorry

end find_angle_y_l19_19658


namespace intersection_of_sets_l19_19530

-- Defining the sets as given in the conditions
def setM : Set ℝ := { x | (x + 1) * (x - 3) ≤ 0 }
def setN : Set ℝ := { x | 1 < x ∧ x < 4 }

-- Statement to prove
theorem intersection_of_sets :
  { x | (x + 1) * (x - 3) ≤ 0 } ∩ { x | 1 < x ∧ x < 4 } = { x | 1 < x ∧ x ≤ 3 } := by
sorry

end intersection_of_sets_l19_19530


namespace water_added_l19_19205

-- Definitions and constants based on conditions
def initial_volume : ℝ := 80
def initial_jasmine_percentage : ℝ := 0.10
def jasmine_added : ℝ := 5
def final_jasmine_percentage : ℝ := 0.13

-- Problem statement
theorem water_added (W : ℝ) :
  (initial_volume * initial_jasmine_percentage + jasmine_added) / (initial_volume + jasmine_added + W) = final_jasmine_percentage → 
  W = 15 :=
by
  sorry

end water_added_l19_19205


namespace find_x_minus_y_l19_19886

-- Variables and conditions
variables (x y : ℝ)
def abs_x_eq_3 := abs x = 3
def y_sq_eq_one_fourth := y^2 = 1 / 4
def x_plus_y_neg := x + y < 0

-- Proof problem stating that x - y must equal one of the two possible values
theorem find_x_minus_y (h1 : abs x = 3) (h2 : y^2 = 1 / 4) (h3 : x + y < 0) : 
  x - y = -7 / 2 ∨ x - y = -5 / 2 :=
  sorry

end find_x_minus_y_l19_19886


namespace fraction_pow_rule_l19_19220

theorem fraction_pow_rule :
  (5 / 7)^4 = 625 / 2401 :=
by
  sorry

end fraction_pow_rule_l19_19220


namespace find_constant_k_l19_19809

theorem find_constant_k (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ)
  (h₁ : ∀ n, S n = 3 * 2^n + k)
  (h₂ : ∀ n, 1 ≤ n → a n = S n - S (n - 1))
  (h₃ : ∃ q, ∀ n, 1 ≤ n → a (n + 1) = a n * q ) :
  k = -3 := 
sorry

end find_constant_k_l19_19809


namespace problem1_problem2_l19_19671

-- Definitions for the conditions
variables {A B C : ℝ}
variables {a b c S : ℝ}

-- Problem 1: Proving the value of side "a" given certain conditions
theorem problem1 (h₁ : S = (1 / 2) * a * b * Real.sin C) (h₂ : a^2 = 4 * Real.sqrt 3 * S)
  (h₃ : C = Real.pi / 3) (h₄ : b = 1) : a = 3 := by
  sorry

-- Problem 2: Proving the measure of angle "A" given certain conditions
theorem problem2 (h₁ : S = (1 / 2) * a * b * Real.sin C) (h₂ : a^2 = 4 * Real.sqrt 3 * S)
  (h₃ : c / b = 2 + Real.sqrt 3) : A = Real.pi / 3 := by
  sorry

end problem1_problem2_l19_19671


namespace graph_of_conic_section_is_straight_lines_l19_19606

variable {x y : ℝ}

theorem graph_of_conic_section_is_straight_lines:
  (x^2 - 9 * y^2 = 0) ↔ (x = 3 * y ∨ x = -3 * y) := by
  sorry

end graph_of_conic_section_is_straight_lines_l19_19606


namespace add_base6_l19_19626

def base6_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  6 * d1 + d0

theorem add_base6 (a b : Nat) (ha : base6_to_base10 a = 23) (hb : base6_to_base10 b = 10) : 
  base6_to_base10 (53 : Nat) = 33 :=
by
  sorry

end add_base6_l19_19626


namespace value_range_of_log_function_l19_19519

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 2*x + 4

noncomputable def log_base_3 (x : ℝ) : ℝ :=
  Real.log x / Real.log 3

theorem value_range_of_log_function :
  ∀ x : ℝ, log_base_3 (quadratic_function x) ≥ 1 := by
  sorry

end value_range_of_log_function_l19_19519


namespace arithmetic_sequence_a18_value_l19_19547

theorem arithmetic_sequence_a18_value 
  (a : ℕ → ℕ) (d : ℕ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_sum : a 2 + a 5 + a 8 = 33)
  (h_geom : (a 5 + 1) ^ 2 = (a 2 + 1) * (a 8 + 7)) :
  a 18 = 37 :=
sorry

end arithmetic_sequence_a18_value_l19_19547


namespace number_of_recipes_needed_l19_19897

def numStudents : ℕ := 150
def avgCookiesPerStudent : ℕ := 3
def cookiesPerRecipe : ℕ := 18
def attendanceDrop : ℝ := 0.40

theorem number_of_recipes_needed (n : ℕ) (c : ℕ) (r : ℕ) (d : ℝ) : 
  n = numStudents →
  c = avgCookiesPerStudent →
  r = cookiesPerRecipe →
  d = attendanceDrop →
  ∃ (recipes : ℕ), recipes = 15 :=
by
  intros
  sorry

end number_of_recipes_needed_l19_19897


namespace only_three_A_l19_19440

def student := Type
variable (Alan Beth Carlos Diana Eliza : student)

variable (gets_A : student → Prop)

variable (H1 : gets_A Alan → gets_A Beth)
variable (H2 : gets_A Beth → gets_A Carlos)
variable (H3 : gets_A Carlos → gets_A Diana)
variable (H4 : gets_A Diana → gets_A Eliza)
variable (H5 : gets_A Eliza → gets_A Alan)
variable (H6 : ∃ a b c : student, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ gets_A a ∧ gets_A b ∧ gets_A c ∧ ∀ d : student, gets_A d → d = a ∨ d = b ∨ d = c)

theorem only_three_A : gets_A Carlos ∧ gets_A Diana ∧ gets_A Eliza :=
by
  sorry

end only_three_A_l19_19440


namespace range_of_m_l19_19232

theorem range_of_m (m : ℝ) (h1 : ∀ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0 → (-2 ≤ x → x ≤ 10))
    (h2 : ∀ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0 → (1 - m ≤ x → x ≤ 1 + m))
    (h3 : m > 0)
    (h4 : ∀ x : ℝ, ¬ ((x^2 + 1) * (x^2 - 8*x - 20) ≤ 0) → ¬ (x^2 - 2*x + 1 - m^2 ≤ 0) → (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) :
  m ≥ 9 := 
sorry

end range_of_m_l19_19232


namespace quadrant_of_angle_l19_19105

theorem quadrant_of_angle (θ : ℝ) (h1 : Real.cos θ = -3 / 5) (h2 : Real.tan θ = 4 / 3) :
    θ ∈ Set.Icc (π : ℝ) (3 * π / 2) := sorry

end quadrant_of_angle_l19_19105


namespace lines_in_n_by_n_grid_l19_19505

def num_horizontal_lines (n : ℕ) : ℕ := n + 1
def num_vertical_lines (n : ℕ) : ℕ := n + 1
def total_lines (n : ℕ) : ℕ := num_horizontal_lines n + num_vertical_lines n

theorem lines_in_n_by_n_grid (n : ℕ) :
  total_lines n = 2 * (n + 1) := by
  sorry

end lines_in_n_by_n_grid_l19_19505


namespace people_in_each_playgroup_l19_19910

theorem people_in_each_playgroup (girls boys parents playgroups : ℕ) (hg : girls = 14) (hb : boys = 11) (hp : parents = 50) (hpg : playgroups = 3) :
  (girls + boys + parents) / playgroups = 25 := by
  sorry

end people_in_each_playgroup_l19_19910


namespace b_a_range_l19_19688
open Real

-- Definitions of angles A, B, and sides a, b in an acute triangle ABC we assume that these are given.
variables {A B C a b c : ℝ}
variable {ABC_acute : A + B + C = π}
variable {angle_condition : B = 2 * A}
variable {sides : a = b * (sin A / sin B)}

theorem b_a_range (h₁ : 0 < A) (h₂ : A < π/2) (h₃ : 0 < C) (h₄ : C < π/2) :
  (∃ A, 30 * (π/180) < A ∧ A < 45 * (π/180)) → 
  (∃ b a, b / a = 2 * cos A) → 
  (∃ x : ℝ, x = b / a ∧ sqrt 2 < x ∧ x < sqrt 3) :=
sorry

end b_a_range_l19_19688


namespace older_brother_age_is_25_l19_19206

noncomputable def age_of_older_brother (father_age current_n : ℕ) (younger_brother_age : ℕ) : ℕ := 
  (father_age - current_n) / 2

theorem older_brother_age_is_25 
  (father_age : ℕ) 
  (h1 : father_age = 50) 
  (younger_brother_age : ℕ)
  (current_n : ℕ) 
  (h2 : (2 * (younger_brother_age + current_n)) = father_age + current_n) : 
  age_of_older_brother father_age current_n younger_brother_age = 25 := 
by
  sorry

end older_brother_age_is_25_l19_19206


namespace total_quartet_songs_l19_19335

/-- 
Five girls — Mary, Alina, Tina, Hanna, and Elsa — sang songs in a concert as quartets,
with one girl sitting out each time. Hanna sang 9 songs, which was more than any other girl,
and Mary sang 3 songs, which was fewer than any other girl. If the total number of songs
sung by Alina and Tina together was 16, then the total number of songs sung by these quartets is 8. -/
theorem total_quartet_songs
  (hanna_songs : ℕ) (mary_songs : ℕ) (alina_tina_songs : ℕ) (total_songs : ℕ)
  (h_hanna : hanna_songs = 9)
  (h_mary : mary_songs = 3)
  (h_alina_tina : alina_tina_songs = 16) :
  total_songs = 8 :=
sorry

end total_quartet_songs_l19_19335


namespace solve_for_x_l19_19833

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_for_x (x y : ℝ) (h : 16 * (3:ℝ) ^ x = (7:ℝ) ^ (y + 4)) (hy : y = -4) :
  x = -4 * log 3 2 := by
  sorry

end solve_for_x_l19_19833


namespace three_digit_integers_sum_to_7_l19_19901

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l19_19901


namespace ratio_of_x_to_y_l19_19931

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 3 / 5) : x / y = 16 / 15 :=
sorry

end ratio_of_x_to_y_l19_19931


namespace ratio_of_functions_l19_19134

def f (x : ℕ) : ℕ := 3 * x + 4
def g (x : ℕ) : ℕ := 4 * x - 3

theorem ratio_of_functions :
  f (g (f 3)) * 121 = 151 * g (f (g 3)) :=
by
  sorry

end ratio_of_functions_l19_19134


namespace part1_part2_l19_19095

variable {α : Type*} [LinearOrderedField α]

-- Definitions based on given problem conditions.
def arithmetic_seq(a_n : ℕ → α) := ∃ a1 d, ∀ n, a_n n = a1 + ↑(n - 1) * d

noncomputable def a10_seq := (30 : α)
noncomputable def a20_seq := (50 : α)

-- Theorem statements to prove:
theorem part1 {a_n : ℕ → α} (h : arithmetic_seq a_n) (h10 : a_n 10 = a10_seq) (h20 : a_n 20 = a20_seq) :
  ∀ n, a_n n = 2 * ↑n + 10 := sorry

theorem part2 {a_n : ℕ → α} (h : arithmetic_seq a_n) (h10 : a_n 10 = a10_seq) (h20 : a_n 20 = a20_seq)
  (Sn : α) (hSn : Sn = 242) :
  ∃ n, Sn = (↑n / 2) * (2 * 12 + (↑n - 1) * 2) ∧ n = 11 := sorry

end part1_part2_l19_19095


namespace problem1_problem2_problem3_l19_19180

-- Problem 1
theorem problem1 (x : ℝ) (h : x^2 - 3 * x = 2) : 1 + 2 * x^2 - 6 * x = 5 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : x^2 - 3 * x - 4 = 0) : 1 + 3 * x - x^2 = -3 :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) (p q : ℝ) (h1 : x = 1 → p * x^3 + q * x + 1 = 5) (h2 : p + q = 4) (hx : x = -1) : p * x^3 + q * x + 1 = -3 :=
by
  sorry

end problem1_problem2_problem3_l19_19180


namespace prob_not_lose_when_A_plays_l19_19066

def appearance_prob_center_forward : ℝ := 0.3
def appearance_prob_winger : ℝ := 0.5
def appearance_prob_attacking_midfielder : ℝ := 0.2

def lose_prob_center_forward : ℝ := 0.3
def lose_prob_winger : ℝ := 0.2
def lose_prob_attacking_midfielder : ℝ := 0.2

theorem prob_not_lose_when_A_plays : 
    (appearance_prob_center_forward * (1 - lose_prob_center_forward) + 
    appearance_prob_winger * (1 - lose_prob_winger) + 
    appearance_prob_attacking_midfielder * (1 - lose_prob_attacking_midfielder)) = 0.77 := 
by
  sorry

end prob_not_lose_when_A_plays_l19_19066


namespace min_students_in_class_l19_19969

-- Define the conditions
variables (b g : ℕ) -- number of boys and girls
variable (h1 : 3 * b = 4 * (2 * g)) -- Equal number of boys and girls passed the test

-- Define the desired minimum number of students
def min_students : ℕ := 17

-- The theorem which asserts that the total number of students in the class is at least 17
theorem min_students_in_class (b g : ℕ) (h1 : 3 * b = 4 * (2 * g)) : (b + g) ≥ min_students := 
sorry

end min_students_in_class_l19_19969


namespace not_monotonic_on_interval_l19_19846

noncomputable def f (x : ℝ) : ℝ := (x^2 / 2) - Real.log x

theorem not_monotonic_on_interval (m : ℝ) : 
  (∃ x y : ℝ, m < x ∧ x < m + 1/2 ∧ m < y ∧ y < m + 1/2 ∧ (x ≠ y) ∧ f x ≠ f y ) ↔ (1/2 < m ∧ m < 1) :=
sorry

end not_monotonic_on_interval_l19_19846


namespace area_of_sector_AOB_l19_19010

-- Definitions for the conditions
def circumference_sector_AOB : Real := 6 -- Circumference of sector AOB
def central_angle_AOB : Real := 1 -- Central angle of sector AOB

-- Theorem stating the area of the sector is 2 cm²
theorem area_of_sector_AOB (C : Real) (θ : Real) (hC : C = circumference_sector_AOB) (hθ : θ = central_angle_AOB) : 
    ∃ S : Real, S = 2 :=
by
  sorry

end area_of_sector_AOB_l19_19010


namespace quadratic_equation_single_solution_l19_19102

theorem quadratic_equation_single_solution (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + a * x + 1 = 0) ∧ (∀ x1 x2 : ℝ, a * x1^2 + a * x1 + 1 = 0 → a * x2^2 + a * x2 + 1 = 0 → x1 = x2) → a = 4 :=
by sorry

end quadratic_equation_single_solution_l19_19102


namespace last_four_digits_5_pow_2015_l19_19659

theorem last_four_digits_5_pow_2015 :
  (5^2015) % 10000 = 8125 :=
by
  sorry

end last_four_digits_5_pow_2015_l19_19659


namespace isosceles_triangle_perimeter_l19_19882

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 7) :
  ∃ (c : ℝ), (a = b ∧ 7 = c ∨ a = c ∧ 7 = b) ∧ a + b + c = 17 :=
by
  use 17
  sorry

end isosceles_triangle_perimeter_l19_19882


namespace number_of_solutions_depends_on_a_l19_19039

theorem number_of_solutions_depends_on_a (a : ℝ) : 
  (∀ x : ℝ, 2^(3 * x) + 4 * a * 2^(2 * x) + a^2 * 2^x - 6 * a^3 = 0) → 
  (if a = 0 then 0 else if a > 0 then 1 else 2) = 
  (if a = 0 then 0 else if a > 0 then 1 else 2) :=
by 
  sorry

end number_of_solutions_depends_on_a_l19_19039


namespace total_students_l19_19651

-- Given conditions
variable (A B : ℕ)
noncomputable def M_A := 80 * A
noncomputable def M_B := 70 * B

axiom classA_condition1 : M_A - 160 = 90 * (A - 8)
axiom classB_condition1 : M_B - 180 = 85 * (B - 6)

-- Required proof in Lean 4 statement
theorem total_students : A + B = 78 :=
by
  sorry

end total_students_l19_19651


namespace shuai_shuai_total_words_l19_19970

-- Conditions
def words (a : ℕ) (n : ℕ) : ℕ := a + n

-- Total words memorized in 7 days
def total_memorized (a : ℕ) : ℕ := 
  (words a 0) + (words a 1) + (words a 2) + (words a 3) + (words a 4) + (words a 5) + (words a 6)

-- Condition: Sum of words memorized in the first 4 days equals sum of words in the last 3 days
def condition (a : ℕ) : Prop := 
  (words a 0) + (words a 1) + (words a 2) + (words a 3) = (words a 4) + (words a 5) + (words a 6)

-- Theorem: If condition is satisfied, then the total number of words memorized is 84.
theorem shuai_shuai_total_words : 
  ∀ a : ℕ, condition a → total_memorized a = 84 :=
by
  intro a h
  sorry

end shuai_shuai_total_words_l19_19970


namespace rate_of_Y_l19_19412

noncomputable def rate_X : ℝ := 2
noncomputable def time_to_cross : ℝ := 0.5

theorem rate_of_Y (rate_Y : ℝ) : rate_X * time_to_cross = 1 → rate_Y * time_to_cross = 1 → rate_Y = rate_X :=
by
    intros h_rate_X h_rate_Y
    sorry

end rate_of_Y_l19_19412


namespace find_vertex_angle_of_cone_l19_19260

noncomputable def vertexAngleCone (r1 r2 : ℝ) (O1 O2 : ℝ) (touching : Prop) (Ctable : Prop) (equalAngles : Prop) : Prop :=
  -- The given conditions:
  -- r1, r2 are the radii of the spheres, where r1 = 4 and r2 = 1.
  -- O1, O2 are the centers of the spheres.
  -- touching indicates the spheres touch externally.
  -- Ctable indicates that vertex C of the cone is on the segment connecting the points where the spheres touch the table.
  -- equalAngles indicates that the rays CO1 and CO2 form equal angles with the table.
  touching → 
  Ctable → 
  equalAngles →
  -- The target to prove:
  ∃ α : ℝ, 2 * α = 2 * Real.arctan (2 / 5)

theorem find_vertex_angle_of_cone (r1 r2 : ℝ) (O1 O2 : ℝ) :
  let touching : Prop := (r1 = 4 ∧ r2 = 1 ∧ abs (O1 - O2) = r1 + r2)
  let Ctable : Prop := (True)  -- Provided by problem conditions, details can be expanded
  let equalAngles : Prop := (True)  
  vertexAngleCone r1 r2 O1 O2 touching Ctable equalAngles := 
by
  sorry

end find_vertex_angle_of_cone_l19_19260


namespace kids_go_to_camp_l19_19142

theorem kids_go_to_camp (total_kids : ℕ) (kids_stay_home : ℕ) (h1 : total_kids = 898051) (h2 : kids_stay_home = 268627) : total_kids - kids_stay_home = 629424 :=
by
  sorry

end kids_go_to_camp_l19_19142


namespace min_value_objective_l19_19598

variable (x y : ℝ)

def constraints : Prop :=
  3 * x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

def objective (x y : ℝ) : ℝ := y - 2 * x

theorem min_value_objective :
  constraints x y → ∃ x y, objective x y = -7 :=
by
  sorry

end min_value_objective_l19_19598


namespace f_m_minus_1_pos_l19_19917

variable {R : Type*} [LinearOrderedField R]

def quadratic_function (x a : R) : R :=
  x^2 - x + a

theorem f_m_minus_1_pos {a m : R} (h_pos : 0 < a) (h_fm : quadratic_function m a < 0) :
  quadratic_function (m - 1 : R) a > 0 :=
sorry

end f_m_minus_1_pos_l19_19917


namespace sphere_to_cube_volume_ratio_l19_19430

noncomputable def volume_ratio (s : ℝ) : ℝ :=
  let r := s / 4
  let V_s := (4/3:ℝ) * Real.pi * r^3 
  let V_c := s^3
  V_s / V_c

theorem sphere_to_cube_volume_ratio (s : ℝ) (h : s > 0) : volume_ratio s = Real.pi / 48 := by
  sorry

end sphere_to_cube_volume_ratio_l19_19430


namespace homework_time_decrease_l19_19796

variable (x : ℝ)
variable (initial_time final_time : ℝ)
variable (adjustments : ℕ)

def rate_of_decrease (initial_time final_time : ℝ) (adjustments : ℕ) (x : ℝ) := 
  initial_time * (1 - x)^adjustments = final_time

theorem homework_time_decrease 
  (h_initial : initial_time = 100) 
  (h_final : final_time = 70)
  (h_adjustments : adjustments = 2)
  (h_decrease : rate_of_decrease initial_time final_time adjustments x) : 
  100 * (1 - x)^2 = 70 :=
by
  sorry

end homework_time_decrease_l19_19796


namespace record_cost_calculation_l19_19653

theorem record_cost_calculation :
  ∀ (books_owned book_price records_bought money_left total_selling_price money_spent_per_record record_cost : ℕ),
  books_owned = 200 →
  book_price = 3 / 2 →
  records_bought = 75 →
  money_left = 75 →
  total_selling_price = books_owned * book_price →
  money_spent_per_record = total_selling_price - money_left →
  record_cost = money_spent_per_record / records_bought →
  record_cost = 3 :=
by
  intros books_owned book_price records_bought money_left total_selling_price money_spent_per_record record_cost
  sorry

end record_cost_calculation_l19_19653


namespace number_of_divisors_not_multiples_of_14_l19_19464

theorem number_of_divisors_not_multiples_of_14 
  (n : ℕ)
  (h1: ∃ k : ℕ, n = 2 * k * k)
  (h2: ∃ k : ℕ, n = 3 * k * k * k)
  (h3: ∃ k : ℕ, n = 5 * k * k * k * k * k)
  (h4: ∃ k : ℕ, n = 7 * k * k * k * k * k * k * k)
  : 
  ∃ num_divisors : ℕ, num_divisors = 19005 ∧ (∀ d : ℕ, d ∣ n → ¬(14 ∣ d)) := sorry

end number_of_divisors_not_multiples_of_14_l19_19464


namespace angles_at_point_l19_19729

theorem angles_at_point (x y : ℝ) 
  (h1 : x + y + 120 = 360) 
  (h2 : x = 2 * y) : 
  x = 160 ∧ y = 80 :=
by
  sorry

end angles_at_point_l19_19729


namespace solve_absolute_value_equation_l19_19316

theorem solve_absolute_value_equation (y : ℝ) :
  (|y - 8| + 3 * y = 11) → (y = 1.5) :=
by
  sorry

end solve_absolute_value_equation_l19_19316


namespace apples_total_l19_19193

theorem apples_total :
  ∀ (Marin David Amanda : ℕ),
  Marin = 6 →
  David = 2 * Marin →
  Amanda = David + 5 →
  Marin + David + Amanda = 35 :=
by
  intros Marin David Amanda hMarin hDavid hAmanda
  sorry

end apples_total_l19_19193


namespace fraction_simplifies_l19_19827

-- Define the integers
def a : ℤ := 1632
def b : ℤ := 1625
def c : ℤ := 1645
def d : ℤ := 1612

-- Define the theorem to prove
theorem fraction_simplifies :
  (a^2 - b^2) / (c^2 - d^2) = 7 / 33 := by
  sorry

end fraction_simplifies_l19_19827


namespace number_of_roses_l19_19610

theorem number_of_roses 
  (R L T : ℕ)
  (h1 : R + L + T = 100)
  (h2 : R = L + 22)
  (h3 : R = T - 20) : R = 34 := 
sorry

end number_of_roses_l19_19610


namespace Marcus_ate_more_than_John_l19_19150

theorem Marcus_ate_more_than_John:
  let John_eaten := 28
  let Marcus_eaten := 40
  Marcus_eaten - John_eaten = 12 :=
by
  sorry

end Marcus_ate_more_than_John_l19_19150


namespace range_of_a_l19_19913

-- Given conditions
def p (x : ℝ) : Prop := abs (4 - x) ≤ 6
def q (x : ℝ) (a : ℝ) : Prop := (x - 1)^2 - a^2 ≥ 0

-- The statement to prove
theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : ∀ x, ¬p x → q x a) : 
  0 < a ∧ a ≤ 3 :=
by
  sorry -- Proof placeholder

end range_of_a_l19_19913


namespace jenna_age_l19_19324

theorem jenna_age (D J : ℕ) (h1 : J = D + 5) (h2 : J + D = 21) (h3 : D = 8) : J = 13 :=
by
  sorry

end jenna_age_l19_19324


namespace ellipse_hyperbola_foci_l19_19424

theorem ellipse_hyperbola_foci (a b : ℝ) 
  (h1 : ∃ (a b : ℝ), b^2 - a^2 = 25 ∧ a^2 + b^2 = 64) : 
  |a * b| = (Real.sqrt 3471) / 2 :=
by
  sorry

end ellipse_hyperbola_foci_l19_19424


namespace pineapple_rings_per_pineapple_l19_19674

def pineapples_purchased : Nat := 6
def cost_per_pineapple : Nat := 3
def rings_sold_per_set : Nat := 4
def price_per_set_of_4_rings : Nat := 5
def profit_made : Nat := 72

theorem pineapple_rings_per_pineapple : (90 / 5 * 4 / 6) = 12 := 
by 
  sorry

end pineapple_rings_per_pineapple_l19_19674


namespace value_of_f_at_pi_over_12_l19_19053

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 12)

theorem value_of_f_at_pi_over_12 : f (Real.pi / 12) = Real.sqrt 2 / 2 :=
by
  sorry

end value_of_f_at_pi_over_12_l19_19053


namespace regular_polygon_sides_l19_19825

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l19_19825


namespace range_of_a_for_monotonicity_l19_19024

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a_for_monotonicity (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 2 < a ∧ a ≤ 3 :=
by sorry

end range_of_a_for_monotonicity_l19_19024


namespace value_of_f_neg_2_l19_19405

section
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_pos : ∀ x : ℝ, 0 < x → f x = 2 ^ x + 1)

theorem value_of_f_neg_2 (h_odd : ∀ x, f (-x) = -f x) (h_pos : ∀ x, 0 < x → f x = 2^x + 1) :
  f (-2) = -5 :=
by
  sorry
end

end value_of_f_neg_2_l19_19405


namespace martys_journey_length_l19_19935

theorem martys_journey_length (x : ℝ) (h1 : x / 4 + 30 + x / 3 = x) : x = 72 :=
sorry

end martys_journey_length_l19_19935


namespace ratio_20_to_10_exists_l19_19099

theorem ratio_20_to_10_exists (x : ℕ) (h : x = 20 * 10) : x = 200 :=
by sorry

end ratio_20_to_10_exists_l19_19099


namespace height_difference_of_packings_l19_19962

theorem height_difference_of_packings :
  (let d := 12
   let n := 180
   let rowsA := n / 10
   let heightA := rowsA * d
   let height_of_hex_gap := (6 * Real.sqrt 3 : ℝ)
   let gaps := rowsA - 1
   let heightB := gaps * height_of_hex_gap + 2 * (d / 2)
   heightA - heightB) = 204 - 102 * Real.sqrt 3 :=
  sorry

end height_difference_of_packings_l19_19962


namespace minimum_value_y_l19_19648

variable {x y : ℝ}

theorem minimum_value_y (h : y * Real.log y = Real.exp (2 * x) - y * Real.log (2 * x)) : y ≥ Real.exp 1 :=
sorry

end minimum_value_y_l19_19648


namespace problem_expression_eval_l19_19718

theorem problem_expression_eval : (1 + 2 + 3) * (1 + 1/2 + 1/3) = 11 := by
  sorry

end problem_expression_eval_l19_19718


namespace solution1_solution2_solution3_l19_19201

noncomputable def problem1 : Real :=
3.5 * 101

noncomputable def problem2 : Real :=
11 * 5.9 - 5.9

noncomputable def problem3 : Real :=
88 - 17.5 - 12.5

theorem solution1 : problem1 = 353.5 :=
by
  sorry

theorem solution2 : problem2 = 59 :=
by
  sorry

theorem solution3 : problem3 = 58 :=
by
  sorry

end solution1_solution2_solution3_l19_19201


namespace solution_set_of_inequality_l19_19628

theorem solution_set_of_inequality : { x : ℝ | x^2 - 2 * x + 1 ≤ 0 } = {1} :=
sorry

end solution_set_of_inequality_l19_19628


namespace evaluate_expression_l19_19177

theorem evaluate_expression : (4 - 3) * 2 = 2 := by
  sorry

end evaluate_expression_l19_19177


namespace sum_interior_ninth_row_l19_19784

-- Define Pascal's Triangle and the specific conditions
def pascal_sum (n : ℕ) : ℕ := 2^(n - 1)

def sum_interior_numbers (n : ℕ) : ℕ := pascal_sum n - 2

theorem sum_interior_ninth_row :
  sum_interior_numbers 9 = 254 := 
by {
  sorry
}

end sum_interior_ninth_row_l19_19784


namespace sum_a_b_l19_19568

theorem sum_a_b (a b : ℕ) (h1 : 2 + 2 / 3 = 2^2 * (2 / 3))
(h2: 3 + 3 / 8 = 3^2 * (3 / 8)) 
(h3: 4 + 4 / 15 = 4^2 * (4 / 15)) 
(h_n : ∀ n, n + n / (n^2 - 1) = n^2 * (n / (n^2 - 1)) → 
(a = 9^2 - 1) ∧ (b = 9)) : 
a + b = 89 := 
sorry

end sum_a_b_l19_19568


namespace octahedron_tetrahedron_surface_area_ratio_l19_19835

theorem octahedron_tetrahedron_surface_area_ratio 
  (s : ℝ) 
  (h₁ : s = 1)
  (A_octahedron : ℝ := 2 * Real.sqrt 3)
  (A_tetrahedron : ℝ := Real.sqrt 3)
  (h₂ : A_octahedron = 2 * Real.sqrt 3 * s^2 / 2 * Real.sqrt 3 * (1/4) * s^2) 
  (h₃ : A_tetrahedron = Real.sqrt 3 * s^2 / 4)
  :
  A_octahedron / A_tetrahedron = 2 := 
by
  sorry

end octahedron_tetrahedron_surface_area_ratio_l19_19835


namespace frank_cookies_l19_19571

theorem frank_cookies :
  ∀ (F M M_i L : ℕ),
    (F = M / 2 - 3) →
    (M = 3 * M_i) →
    (M_i = 2 * L) →
    (L = 5) →
    F = 12 :=
by
  intros F M M_i L h1 h2 h3 h4
  rw [h4] at h3
  rw [h3] at h2
  rw [h2] at h1
  sorry

end frank_cookies_l19_19571


namespace original_price_of_dish_l19_19553

theorem original_price_of_dish :
  let P : ℝ := 40
  (0.9 * P + 0.15 * P) - (0.9 * P + 0.15 * 0.9 * P) = 0.60 → P = 40 := by
  intros P h
  sorry

end original_price_of_dish_l19_19553


namespace trapezoid_proof_l19_19533

variables {Point : Type} [MetricSpace Point]

-- Definitions of the points and segments as given conditions.
variables (A B C D E : Point)

-- Definitions representing the trapezoid and point E's property.
def is_trapezoid (ABCD : (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)) : Prop :=
  (A ≠ B) ∧ (C ≠ D)

def on_segment (E : Point) (A D : Point) : Prop :=
  -- This definition will encompass the fact that E is on segment AD.
  -- Representing the notion that E lies between A and D.
  dist A E + dist E D = dist A D

def equal_perimeters (E : Point) (A B C D : Point) : Prop :=
  let p1 := (dist A B + dist B E + dist E A)
  let p2 := (dist B C + dist C E + dist E B)
  let p3 := (dist C D + dist D E + dist E C)
  p1 = p2 ∧ p2 = p3

-- The theorem we need to prove.
theorem trapezoid_proof (ABCD : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) (onSeg : on_segment E A D) (eqPerim : equal_perimeters E A B C D) : 
  dist B C = dist A D / 2 :=
sorry

end trapezoid_proof_l19_19533


namespace expression_value_zero_l19_19062

variables (a b c A B C : ℝ)

theorem expression_value_zero
  (h1 : a + b + c = 0)
  (h2 : A + B + C = 0)
  (h3 : a / A + b / B + c / C = 0) :
  a * A^2 + b * B^2 + c * C^2 = 0 :=
by
  sorry

end expression_value_zero_l19_19062


namespace server_multiplications_in_half_hour_l19_19899

theorem server_multiplications_in_half_hour : 
  let rate := 5000
  let seconds_in_half_hour := 1800
  rate * seconds_in_half_hour = 9000000 := by
  sorry

end server_multiplications_in_half_hour_l19_19899


namespace parabola_focus_distance_l19_19217

theorem parabola_focus_distance (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1) (h_dist_y_axis : |P.1| = 4) : 
  dist P (4, 0) = 5 :=
sorry

end parabola_focus_distance_l19_19217


namespace minValueExpr_ge_9_l19_19957

noncomputable def minValueExpr (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minValueExpr_ge_9 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  minValueExpr x y z ≥ 9 :=
by sorry

end minValueExpr_ge_9_l19_19957


namespace joshua_additional_cents_needed_l19_19196

def cost_of_pen_cents : ℕ := 600
def money_joshua_has_cents : ℕ := 500
def money_borrowed_cents : ℕ := 68

def additional_cents_needed (cost money has borrowed : ℕ) : ℕ :=
  cost - (has + borrowed)

theorem joshua_additional_cents_needed :
  additional_cents_needed cost_of_pen_cents money_joshua_has_cents money_borrowed_cents = 32 :=
by
  sorry

end joshua_additional_cents_needed_l19_19196


namespace find_x_l19_19876

theorem find_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 :=
by
  sorry

end find_x_l19_19876


namespace smallest_possible_b_l19_19988

-- Definition of the polynomial Q(x)
def Q (x : ℤ) : ℤ := sorry -- Polynomial with integer coefficients

-- Initial conditions for b and Q
variable (b : ℤ) (hb : b > 0)
variable (hQ1 : Q 2 = b)
variable (hQ2 : Q 4 = b)
variable (hQ3 : Q 6 = b)
variable (hQ4 : Q 8 = b)
variable (hQ5 : Q 1 = -b)
variable (hQ6 : Q 3 = -b)
variable (hQ7 : Q 5 = -b)
variable (hQ8 : Q 7 = -b)

theorem smallest_possible_b : b = 315 :=
by
  sorry

end smallest_possible_b_l19_19988


namespace shortest_distance_phenomena_explained_l19_19811

def condition1 : Prop :=
  ∀ (a b : ℕ), (exists nail1 : ℕ, exists nail2 : ℕ, nail1 ≠ nail2) → (exists wall : ℕ, wall = a + b)

def condition2 : Prop :=
  ∀ (tree1 tree2 tree3 : ℕ), tree1 ≠ tree2 → tree2 ≠ tree3 → (tree1 + tree2 + tree3) / 3 = tree2

def condition3 : Prop :=
  ∀ (A B : ℕ), ∃ (C : ℕ), C = (B - A) → (A = B - (B - A))

def condition4 : Prop :=
  ∀ (dist : ℕ), dist = 0 → exists shortest : ℕ, shortest < dist

-- The following theorem needs to be proven to match our mathematical problem
theorem shortest_distance_phenomena_explained :
  condition3 ∧ condition4 :=
by
  sorry

end shortest_distance_phenomena_explained_l19_19811


namespace constant_k_value_l19_19828

theorem constant_k_value 
  (S : ℕ → ℕ)
  (h : ∀ n : ℕ, S n = 4 * 3^(n + 1) - k) :
  k = 12 :=
sorry

end constant_k_value_l19_19828


namespace find_pairs_l19_19293

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem find_pairs (a n : ℕ) (h1 : a ≥ n) (h2 : is_power_of_two ((a + 1)^n + a - 1)) :
  (a = 4 ∧ n = 3) ∨ (∃ k : ℕ, a = 2^k ∧ n = 1) :=
by
  sorry

end find_pairs_l19_19293


namespace union_intersection_l19_19127

variable (a : ℝ)

def setA (a : ℝ) : Set ℝ := { x | (x - 3) * (x - a) = 0 }
def setB : Set ℝ := {1, 4}

theorem union_intersection (a : ℝ) :
  (if a = 3 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = ∅ else 
   if a = 1 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = {1} else
   if a = 4 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = {4} else
   setA a ∪ setB = {1, 3, 4, a} ∧ setA a ∩ setB = ∅) := sorry

end union_intersection_l19_19127


namespace perimeter_of_park_is_66_l19_19208

-- Given width and length of the flower bed
variables (w l : ℝ)
-- Given that the length is four times the width
variable (h1 : l = 4 * w)
-- Given the area of the flower bed
variable (h2 : l * w = 100)
-- Given the width of the walkway
variable (walkway_width : ℝ := 2)

-- The total width and length of the park, including the walkway
def w_park := w + 2 * walkway_width
def l_park := l + 2 * walkway_width

-- The proof statement: perimeter of the park equals 66 meters
theorem perimeter_of_park_is_66 :
  2 * (l_park + w_park) = 66 :=
by
  -- The full proof can be filled in here
  sorry

end perimeter_of_park_is_66_l19_19208


namespace total_area_correct_l19_19765

noncomputable def total_area (b l: ℝ) (h1: l = 3 * b) (h2: l * b = 588) : ℝ :=
  let rect_area := 588 -- Area of the rectangle
  let semi_circle_area := 24.5 * Real.pi -- Area of the semi-circle based on given diameter
  rect_area + semi_circle_area

theorem total_area_correct (b l: ℝ) (h1: l = 3 * b) (h2: l * b = 588) : 
  total_area b l h1 h2 = 588 + 24.5 * Real.pi :=
by
  sorry

end total_area_correct_l19_19765


namespace fraction_meaningful_l19_19660

theorem fraction_meaningful (x : ℝ) : (∃ y, y = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end fraction_meaningful_l19_19660


namespace percentage_calculation_l19_19026

theorem percentage_calculation
  (x : ℝ)
  (hx : x = 16)
  (h : 0.15 * 40 - (P * x) = 2) :
  P = 0.25 := by
  sorry

end percentage_calculation_l19_19026


namespace rods_in_one_mile_l19_19538

theorem rods_in_one_mile :
  (1 * 80 * 4 = 320) :=
sorry

end rods_in_one_mile_l19_19538


namespace solve_quadratic_l19_19725

theorem solve_quadratic (x : ℝ) : (x^2 + 2*x = 0) ↔ (x = 0 ∨ x = -2) :=
by
  sorry

end solve_quadratic_l19_19725


namespace tim_pays_300_l19_19133

def mri_cost : ℕ := 1200
def doctor_rate_per_hour : ℕ := 300
def examination_time_in_hours : ℕ := 1 / 2
def consultation_fee : ℕ := 150
def insurance_coverage : ℚ := 0.8

def examination_cost : ℕ := doctor_rate_per_hour * examination_time_in_hours
def total_cost_before_insurance : ℕ := mri_cost + examination_cost + consultation_fee
def insurance_coverage_amount : ℚ := total_cost_before_insurance * insurance_coverage
def amount_tim_pays : ℚ := total_cost_before_insurance - insurance_coverage_amount

theorem tim_pays_300 : amount_tim_pays = 300 := 
by
  -- proof goes here
  sorry

end tim_pays_300_l19_19133


namespace tangent_line_equation_l19_19978

theorem tangent_line_equation :
  ∃ (P : ℝ × ℝ) (m : ℝ), 
  P = (-2, 15) ∧ m = 2 ∧ 
  (∀ (x y : ℝ), (y = x^3 - 10 * x + 3) → (y - 15 = 2 * (x + 2))) :=
sorry

end tangent_line_equation_l19_19978


namespace probability_all_boys_probability_one_girl_probability_at_least_one_girl_l19_19775

-- Assumptions and Definitions
def total_outcomes := Nat.choose 5 3
def all_boys_outcomes := Nat.choose 3 3
def one_girl_outcomes := Nat.choose 3 2 * Nat.choose 2 1
def at_least_one_girl_outcomes := one_girl_outcomes + Nat.choose 3 1 * Nat.choose 2 2

-- The probability calculation proofs
theorem probability_all_boys : all_boys_outcomes / total_outcomes = 1 / 10 := by 
  sorry

theorem probability_one_girl : one_girl_outcomes / total_outcomes = 6 / 10 := by 
  sorry

theorem probability_at_least_one_girl : at_least_one_girl_outcomes / total_outcomes = 9 / 10 := by 
  sorry

end probability_all_boys_probability_one_girl_probability_at_least_one_girl_l19_19775


namespace email_sending_ways_l19_19288

theorem email_sending_ways (n k : ℕ) (hn : n = 3) (hk : k = 5) : n^k = 243 := 
by
  sorry

end email_sending_ways_l19_19288


namespace xiao_ming_polygon_l19_19585

theorem xiao_ming_polygon (n : ℕ) (h : (n - 2) * 180 = 2185) : n = 14 :=
by sorry

end xiao_ming_polygon_l19_19585


namespace tshirts_per_package_l19_19289

def number_of_packages := 28
def total_white_tshirts := 56
def white_tshirts_per_package : Nat :=
  total_white_tshirts / number_of_packages

theorem tshirts_per_package :
  white_tshirts_per_package = 2 :=
by
  -- Assuming the definitions and the proven facts
  sorry

end tshirts_per_package_l19_19289


namespace value_of_f_3_and_f_neg_7_point_5_l19_19686

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 1) = -f x
axiom definition_f : ∀ x : ℝ, -1 < x → x < 1 → f x = x

theorem value_of_f_3_and_f_neg_7_point_5 :
  f 3 + f (-7.5) = 0.5 :=
sorry

end value_of_f_3_and_f_neg_7_point_5_l19_19686


namespace continuous_stripe_probability_l19_19286

def cube_stripe_probability : ℚ :=
  let stripe_combinations_per_face := 8
  let total_combinations := stripe_combinations_per_face ^ 6
  let valid_combinations := 4 * 3 * 8 * 64
  let probability := valid_combinations / total_combinations
  probability

theorem continuous_stripe_probability :
  cube_stripe_probability = 3 / 128 := by
  sorry

end continuous_stripe_probability_l19_19286


namespace range_f_l19_19402

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by 
  sorry

end range_f_l19_19402


namespace reading_homework_pages_eq_three_l19_19893

-- Define the conditions
def pages_of_math_homework : ℕ := 7
def difference : ℕ := 4

-- Define what we need to prove
theorem reading_homework_pages_eq_three (x : ℕ) (h : x + difference = pages_of_math_homework) : x = 3 := by
  sorry

end reading_homework_pages_eq_three_l19_19893


namespace find_k_and_f_min_total_cost_l19_19242

-- Define the conditions
def construction_cost (x : ℝ) : ℝ := 60 * x
def energy_consumption_cost (x : ℝ) : ℝ := 40 - 4 * x
def total_cost (x : ℝ) : ℝ := construction_cost x + 20 * energy_consumption_cost x

theorem find_k_and_f :
  (∀ x, 0 ≤ x ∧ x ≤ 10 → energy_consumption_cost 0 = 8 → energy_consumption_cost x = 40 - 4 * x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 10 → total_cost x = 800 - 74 * x) :=
by
  sorry

theorem min_total_cost :
  (∀ x, 0 ≤ x ∧ x ≤ 10 → 800 - 74 * x ≥ 70) ∧
  total_cost 5 = 70 :=
by
  sorry

end find_k_and_f_min_total_cost_l19_19242


namespace find_a_3_l19_19673

noncomputable def a_n (n : ℕ) : ℤ := 2 + (n - 1)  -- Definition of the arithmetic sequence

theorem find_a_3 (d : ℤ) (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 5 + a 7 = 2 * a 4 + 4) : a 3 = 4 :=
by 
  sorry

end find_a_3_l19_19673


namespace maximize_area_minimize_length_l19_19050

-- Problem 1: Prove maximum area of the enclosure
theorem maximize_area (x y : ℝ) (h : x + 2 * y = 36) : 18 * 9 = 162 :=
by
  sorry

-- Problem 2: Prove the minimum length of steel wire mesh
theorem minimize_length (x y : ℝ) (h1 : x * y = 32) : 8 + 2 * 4 = 16 :=
by
  sorry

end maximize_area_minimize_length_l19_19050


namespace percentage_increase_biographies_l19_19479

variable (B b n : ℝ)
variable (h1 : b = 0.20 * B)
variable (h2 : b + n = 0.32 * (B + n))

theorem percentage_increase_biographies (B b n : ℝ) (h1 : b = 0.20 * B) (h2 : b + n = 0.32 * (B + n)) :
  n / b * 100 = 88.24 := by
  sorry

end percentage_increase_biographies_l19_19479


namespace haley_seeds_l19_19905

theorem haley_seeds (total_seeds seeds_big_garden total_small_gardens seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : seeds_big_garden = 35)
  (h3 : total_small_gardens = 7)
  (h4 : total_seeds - seeds_big_garden = 21)
  (h5 : 21 / total_small_gardens = seeds_per_small_garden) :
  seeds_per_small_garden = 3 :=
by sorry

end haley_seeds_l19_19905


namespace solve_inequality_a_eq_2_solve_inequality_a_in_R_l19_19056

theorem solve_inequality_a_eq_2 :
  {x : ℝ | x > 2 ∨ x < 1} = {x : ℝ | x^2 - 3*x + 2 > 0} :=
sorry

theorem solve_inequality_a_in_R (a : ℝ) :
  {x : ℝ | 
    (a > 1 ∧ (x > a ∨ x < 1)) ∨ 
    (a = 1 ∧ x ≠ 1) ∨ 
    (a < 1 ∧ (x > 1 ∨ x < a))
  } = 
  {x : ℝ | x^2 - (1 + a)*x + a > 0} :=
sorry

end solve_inequality_a_eq_2_solve_inequality_a_in_R_l19_19056


namespace inequality_of_abc_l19_19954

variable (a b c : ℝ)

theorem inequality_of_abc (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c ≥ a * b * c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c :=
sorry

end inequality_of_abc_l19_19954


namespace correct_sum_is_132_l19_19204

-- Let's define the conditions:
-- The ones digit B is mistakenly taken as 1 (when it should be 7)
-- The tens digit C is mistakenly taken as 6 (when it should be 4)
-- The incorrect sum is 146

def correct_ones_digit (mistaken_ones_digit : Nat) : Nat :=
  -- B was mistaken for 1, so B should be 7
  if mistaken_ones_digit = 1 then 7 else mistaken_ones_digit

def correct_tens_digit (mistaken_tens_digit : Nat) : Nat :=
  -- C was mistaken for 6, so C should be 4
  if mistaken_tens_digit = 6 then 4 else mistaken_tens_digit

def correct_sum (incorrect_sum : Nat) : Nat :=
  -- Correcting the sum based on the mistakes
  incorrect_sum + 6 - 20 -- 6 to correct ones mistake, minus 20 to correct tens mistake

theorem correct_sum_is_132 : correct_sum 146 = 132 :=
  by
    -- The theorem is here to check that the corrected sum equals 132
    sorry

end correct_sum_is_132_l19_19204


namespace cos_210_eq_neg_sqrt3_div_2_l19_19436

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (Real.pi + Real.pi / 6) = -Real.sqrt 3 / 2 := sorry

end cos_210_eq_neg_sqrt3_div_2_l19_19436


namespace initial_quantity_of_milk_l19_19390

theorem initial_quantity_of_milk (A B C : ℝ) 
    (h1 : B = 0.375 * A)
    (h2 : C = 0.625 * A)
    (h3 : B + 148 = C - 148) : A = 1184 :=
by
  sorry

end initial_quantity_of_milk_l19_19390


namespace find_d_h_l19_19593

theorem find_d_h (a b c d g h : ℂ) (h1 : b = 4) (h2 : g = -a - c) (h3 : a + c + g = 0) (h4 : b + d + h = 3) : 
  d + h = -1 := 
by
  sorry

end find_d_h_l19_19593


namespace find_number_l19_19858

theorem find_number (x : ℝ) (h : (5/4) * x = 40) : x = 32 := 
sorry

end find_number_l19_19858


namespace solve_congruence_l19_19944

theorem solve_congruence (n : ℤ) : 15 * n ≡ 9 [ZMOD 47] → n ≡ 18 [ZMOD 47] :=
by
  sorry

end solve_congruence_l19_19944


namespace coefficient_x_is_five_l19_19457

theorem coefficient_x_is_five (x y a : ℤ) (h1 : a * x + y = 19) (h2 : x + 3 * y = 1) (h3 : 3 * x + 2 * y = 10) : a = 5 :=
by sorry

end coefficient_x_is_five_l19_19457


namespace rains_at_least_once_l19_19185

noncomputable def prob_rains_on_weekend : ℝ :=
  let prob_rain_saturday := 0.60
  let prob_rain_sunday := 0.70
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let independent_events := prob_no_rain_saturday * prob_no_rain_sunday
  1 - independent_events

theorem rains_at_least_once :
  prob_rains_on_weekend = 0.88 :=
by sorry

end rains_at_least_once_l19_19185


namespace math_problem_l19_19611

theorem math_problem (a b c k : ℝ) (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h2 : a + b + c = 0) (h3 : a^2 = k * b^2) (hk : k ≠ 0) :
  (a^2 * b^2) / ((a^2 - b * c) * (b^2 - a * c)) + (a^2 * c^2) / ((a^2 - b * c) * (c^2 - a * b)) + (b^2 * c^2) / ((b^2 - a * c) * (c^2 - a * b)) = 1 :=
by
  sorry

end math_problem_l19_19611


namespace perpendicular_line_through_circle_center_l19_19149

theorem perpendicular_line_through_circle_center :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (y = m * x + b) → (x = -1 ∧ y = 0) ) ∧ m = 1 ∧ b = 1 ∧ (∀ (x y : ℝ), (y = x + 1) → (x - y + 1 = 0)) :=
sorry

end perpendicular_line_through_circle_center_l19_19149


namespace train_length_l19_19790

theorem train_length
  (S : ℝ)
  (L : ℝ)
  (h1 : L + 140 = S * 15)
  (h2 : L + 250 = S * 20) :
  L = 190 :=
by
  -- Proof to be provided here
  sorry

end train_length_l19_19790


namespace polygon_sides_l19_19716

theorem polygon_sides (n : ℕ) : 
  (180 * (n - 2) / 360 = 5 / 2) → n = 7 :=
by
  sorry

end polygon_sides_l19_19716


namespace range_of_a_l19_19575

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (x^2 + (a + 2) * x + 1) * ((3 - 2 * a) * x^2 + 5 * x + (3 - 2 * a)) ≥ 0) : a ∈ Set.Icc (-4 : ℝ) 0 := sorry

end range_of_a_l19_19575


namespace total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges_l19_19804

theorem total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges :
  ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ a * b * c = 1001 ∧ 2 * (a * b + b * c + c * a) = 622 :=
by
  sorry

end total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges_l19_19804


namespace inequality_proof_l19_19137

theorem inequality_proof (x y : ℝ) (h : x * y < 0) : abs (x + y) < abs (x - y) :=
sorry

end inequality_proof_l19_19137


namespace road_length_l19_19708

theorem road_length (n : ℕ) (d : ℕ) (trees : ℕ) (intervals : ℕ) (L : ℕ) 
  (h1 : n = 10) 
  (h2 : d = 10) 
  (h3 : trees = 10) 
  (h4 : intervals = trees - 1) 
  (h5 : L = intervals * d) : 
  L = 90 :=
by
  sorry

end road_length_l19_19708


namespace spiders_loose_l19_19717

noncomputable def initial_birds : ℕ := 12
noncomputable def initial_puppies : ℕ := 9
noncomputable def initial_cats : ℕ := 5
noncomputable def initial_spiders : ℕ := 15
noncomputable def birds_sold : ℕ := initial_birds / 2
noncomputable def puppies_adopted : ℕ := 3
noncomputable def remaining_puppies : ℕ := initial_puppies - puppies_adopted
noncomputable def remaining_cats : ℕ := initial_cats
noncomputable def total_remaining_animals_except_spiders : ℕ := birds_sold + remaining_puppies + remaining_cats
noncomputable def total_animals_left : ℕ := 25
noncomputable def remaining_spiders : ℕ := total_animals_left - total_remaining_animals_except_spiders
noncomputable def spiders_went_loose : ℕ := initial_spiders - remaining_spiders

theorem spiders_loose : spiders_went_loose = 7 := by
  sorry

end spiders_loose_l19_19717


namespace union_sets_l19_19377

-- Definitions of sets A and B
def set_A : Set ℝ := {x | x / (x - 1) < 0}
def set_B : Set ℝ := {x | abs (1 - x) > 1 / 2}

-- The problem: prove that the union of sets A and B is (-∞, 1) ∪ (3/2, ∞)
theorem union_sets :
  set_A ∪ set_B = {x | x < 1} ∪ {x | x > 3 / 2} :=
by
  sorry

end union_sets_l19_19377


namespace celia_time_correct_lexie_time_correct_nik_time_correct_l19_19126

noncomputable def lexie_time_per_mile : ℝ := 20
noncomputable def celia_time_per_mile : ℝ := lexie_time_per_mile / 2
noncomputable def nik_time_per_mile : ℝ := lexie_time_per_mile / 1.5

noncomputable def total_distance : ℝ := 30

-- Calculate the baseline running time without obstacles
noncomputable def lexie_baseline_time : ℝ := lexie_time_per_mile * total_distance
noncomputable def celia_baseline_time : ℝ := celia_time_per_mile * total_distance
noncomputable def nik_baseline_time : ℝ := nik_time_per_mile * total_distance

-- Additional time due to obstacles
noncomputable def celia_muddy_extra_time : ℝ := 2 * (celia_time_per_mile * 1.25 - celia_time_per_mile)
noncomputable def lexie_bee_extra_time : ℝ := 2 * 10
noncomputable def nik_detour_extra_time : ℝ := 0.5 * nik_time_per_mile

-- Total time taken including obstacles
noncomputable def celia_total_time : ℝ := celia_baseline_time + celia_muddy_extra_time
noncomputable def lexie_total_time : ℝ := lexie_baseline_time + lexie_bee_extra_time
noncomputable def nik_total_time : ℝ := nik_baseline_time + nik_detour_extra_time

theorem celia_time_correct : celia_total_time = 305 := by sorry
theorem lexie_time_correct : lexie_total_time = 620 := by sorry
theorem nik_time_correct : nik_total_time = 406.565 := by sorry

end celia_time_correct_lexie_time_correct_nik_time_correct_l19_19126


namespace meet_at_starting_line_l19_19907

theorem meet_at_starting_line (henry_time margo_time : ℕ) (h_henry : henry_time = 7) (h_margo : margo_time = 12) : Nat.lcm henry_time margo_time = 84 :=
by
  rw [h_henry, h_margo]
  sorry

end meet_at_starting_line_l19_19907


namespace not_in_M_4n2_l19_19298

def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

theorem not_in_M_4n2 (n : ℤ) : ¬ (4 * n + 2 ∈ M) :=
by
sorry

end not_in_M_4n2_l19_19298


namespace sum_abs_coeffs_l19_19234

theorem sum_abs_coeffs (a : ℝ → ℝ) :
  (∀ x, (1 - 3 * x)^9 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9) →
  |a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| = 4^9 := by
  sorry

end sum_abs_coeffs_l19_19234


namespace sqrt_of_9_neg_sqrt_of_0_49_pm_sqrt_of_64_div_81_l19_19504

-- Definition and proof of sqrt(9) = 3
theorem sqrt_of_9 : Real.sqrt 9 = 3 := by
  sorry

-- Definition and proof of -sqrt(0.49) = -0.7
theorem neg_sqrt_of_0_49 : -Real.sqrt 0.49 = -0.7 := by
  sorry

-- Definition and proof of ±sqrt(64/81) = ±(8/9)
theorem pm_sqrt_of_64_div_81 : (Real.sqrt (64 / 81) = 8 / 9) ∧ (Real.sqrt (64 / 81) = -8 / 9) := by
  sorry

end sqrt_of_9_neg_sqrt_of_0_49_pm_sqrt_of_64_div_81_l19_19504


namespace fliers_left_l19_19995

theorem fliers_left (initial_fliers : ℕ) (fraction_morning : ℕ) (fraction_afternoon : ℕ) :
  initial_fliers = 2000 → 
  fraction_morning = 1 / 10 → 
  fraction_afternoon = 1 / 4 → 
  (initial_fliers - initial_fliers * fraction_morning - 
  (initial_fliers - initial_fliers * fraction_morning) * fraction_afternoon) = 1350 := by
  intros initial_fliers_eq fraction_morning_eq fraction_afternoon_eq
  sorry

end fliers_left_l19_19995


namespace Bruce_Anne_combined_cleaning_time_l19_19459

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end Bruce_Anne_combined_cleaning_time_l19_19459


namespace base8_addition_l19_19584

theorem base8_addition (X Y : ℕ) 
  (h1 : 5 * 8 + X + Y + 3 * 8 + 2 = 6 * 64 + 4 * 8 + X) :
  X + Y = 16 := by
  sorry

end base8_addition_l19_19584


namespace chris_and_fiona_weight_l19_19564

theorem chris_and_fiona_weight (c d e f : ℕ) (h1 : c + d = 330) (h2 : d + e = 290) (h3 : e + f = 310) : c + f = 350 :=
by
  sorry

end chris_and_fiona_weight_l19_19564


namespace incorrect_statement_l19_19310

def vector_mult (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

theorem incorrect_statement (a b : ℝ × ℝ) : vector_mult a b ≠ vector_mult b a :=
by
  sorry

end incorrect_statement_l19_19310


namespace area_of_circular_platform_l19_19141

theorem area_of_circular_platform (d : ℝ) (h : d = 2) : ∃ (A : ℝ), A = Real.pi ∧ A = π *(d / 2)^2 := by
  sorry

end area_of_circular_platform_l19_19141


namespace men_wages_eq_13_5_l19_19501

-- Definitions based on problem conditions
def wages (men women boys : ℕ) : ℝ :=
  if 9 * men + women + 7 * boys = 216 then
    men
  else 
    0

def equivalent_wage (men_wage women_wage boy_wage : ℝ) : Prop :=
  9 * men_wage = women_wage ∧
  women_wage = 7 * boy_wage

def total_earning (men_wage women_wage boy_wage : ℝ) : Prop :=
  9 * men_wage + 7 * boy_wage = 216

-- Theorem statement
theorem men_wages_eq_13_5 (M_wage W_wage B_wage : ℝ) :
  equivalent_wage M_wage W_wage B_wage →
  total_earning M_wage W_wage B_wage →
  M_wage = 13.5 :=
by 
  intros h_equiv h_total
  sorry

end men_wages_eq_13_5_l19_19501


namespace detergent_per_pound_l19_19338

theorem detergent_per_pound (detergent clothes_per_det: ℝ) (h: detergent = 18 ∧ clothes_per_det = 9) :
  detergent / clothes_per_det = 2 :=
by
  sorry

end detergent_per_pound_l19_19338


namespace xiao_ming_total_score_l19_19537

-- Definitions for the given conditions
def score_regular : ℝ := 70
def score_midterm : ℝ := 80
def score_final : ℝ := 85

def weight_regular : ℝ := 0.3
def weight_midterm : ℝ := 0.3
def weight_final : ℝ := 0.4

-- The statement that we need to prove
theorem xiao_ming_total_score : 
  (score_regular * weight_regular) + (score_midterm * weight_midterm) + (score_final * weight_final) = 79 := 
by
  sorry

end xiao_ming_total_score_l19_19537


namespace equation_of_line_through_point_with_equal_intercepts_l19_19236

open LinearAlgebra

theorem equation_of_line_through_point_with_equal_intercepts :
  ∃ (a b c : ℝ), (a * 1 + b * 2 + c = 0) ∧ (a * b < 0) ∧ ∀ x y : ℝ, 
  (a * x + b * y + c = 0 ↔ (2 * x - y = 0 ∨ x + y - 3 = 0)) :=
sorry

end equation_of_line_through_point_with_equal_intercepts_l19_19236


namespace cab_base_price_l19_19052

theorem cab_base_price (base_price : ℝ) (total_cost : ℝ) (cost_per_mile : ℝ) (distance : ℝ) 
  (H1 : total_cost = 23) 
  (H2 : cost_per_mile = 4) 
  (H3 : distance = 5) 
  (H4 : base_price = total_cost - cost_per_mile * distance) : 
  base_price = 3 :=
by 
  sorry

end cab_base_price_l19_19052


namespace smaller_cube_volume_is_correct_l19_19805

noncomputable def inscribed_smaller_cube_volume 
  (edge_length_outer_cube : ℝ)
  (h : edge_length_outer_cube = 12) : ℝ := 
  let diameter_sphere := edge_length_outer_cube
  let radius_sphere := diameter_sphere / 2
  let space_diagonal_smaller_cube := diameter_sphere
  let side_length_smaller_cube := space_diagonal_smaller_cube / (Real.sqrt 3)
  let volume_smaller_cube := side_length_smaller_cube ^ 3
  volume_smaller_cube

theorem smaller_cube_volume_is_correct 
  (h : 12 = 12) : inscribed_smaller_cube_volume 12 h = 192 * Real.sqrt 3 :=
by
  sorry

end smaller_cube_volume_is_correct_l19_19805


namespace separate_curves_l19_19572

variable {A : Type} [CommRing A]

def crossing_characteristic (ε : A → ℤ) (A1 A2 A3 A4 : A) : Prop :=
  ε A1 + ε A2 + ε A3 + ε A4 = 0

theorem separate_curves {A : Type} [CommRing A]
  {ε : A → ℤ} {A1 A2 A3 A4 : A} 
  (h : ε A1 + ε A2 + ε A3 + ε A4 = 0)
  (h1 : ε A1 = 1 ∨ ε A1 = -1)
  (h2 : ε A2 = 1 ∨ ε A2 = -1)
  (h3 : ε A3 = 1 ∨ ε A3 = -1)
  (h4 : ε A4 = 1 ∨ ε A4 = -1) :
  (∃ B1 B2 : A, B1 ≠ B2 ∧  ∀ (A : A), ((ε A = 1) → (A = B1)) ∨ ((ε A = -1) → (A = B2))) :=
  sorry

end separate_curves_l19_19572


namespace largest_n_unique_k_l19_19581

theorem largest_n_unique_k : 
  ∃ n : ℕ, (∀ k : ℤ, (5 / 12 : ℚ) < n / (n + k) ∧ n / (n + k) < (4 / 9 : ℚ) → k = 9) ∧ n = 7 :=
by
  sorry

end largest_n_unique_k_l19_19581


namespace time_to_cut_womans_hair_l19_19409

theorem time_to_cut_womans_hair 
  (WL : ℕ) (WM : ℕ) (WK : ℕ) (total_time : ℕ) 
  (num_women : ℕ) (num_men : ℕ) (num_kids : ℕ) 
  (men_haircut_time : ℕ) (kids_haircut_time : ℕ) 
  (overall_time : ℕ) :
  men_haircut_time = 15 →
  kids_haircut_time = 25 →
  num_women = 3 →
  num_men = 2 →
  num_kids = 3 →
  overall_time = 255 →
  overall_time = (num_women * WL + num_men * men_haircut_time + num_kids * kids_haircut_time) →
  WL = 50 :=
by
  sorry

end time_to_cut_womans_hair_l19_19409


namespace altitude_of_isosceles_triangle_l19_19616

noncomputable def radius_X (C : ℝ) := C / (2 * Real.pi)
noncomputable def radius_Y (radius_X : ℝ) := radius_X
noncomputable def a (radius_Y : ℝ) := radius_Y / 2

-- Define the theorem to be proven
theorem altitude_of_isosceles_triangle (C : ℝ) (h_C : C = 14 * Real.pi) (radius_X := radius_X C) (radius_Y := radius_Y radius_X) (a := a radius_Y) :
  ∃ h : ℝ, h = a * Real.sqrt 3 :=
sorry

end altitude_of_isosceles_triangle_l19_19616


namespace fraction_irreducible_l19_19864

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 1) = 1 :=
sorry

end fraction_irreducible_l19_19864


namespace calculate_hidden_dots_l19_19859

def sum_faces_of_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6

def number_of_dice : ℕ := 4
def total_sum_of_dots : ℕ := number_of_dice * sum_faces_of_die

def visible_faces : List (ℕ × String) :=
  [(1, "red"), (1, "none"), (2, "none"), (2, "blue"),
   (3, "none"), (4, "none"), (5, "none"), (6, "none")]

def adjust_face_value (value : ℕ) (color : String) : ℕ :=
  match color with
  | "red" => 2 * value
  | "blue" => 2 * value
  | _ => value

def visible_sum : ℕ :=
  visible_faces.foldl (fun acc (face) => acc + adjust_face_value face.1 face.2) 0

theorem calculate_hidden_dots :
  (total_sum_of_dots - visible_sum) = 57 :=
sorry

end calculate_hidden_dots_l19_19859


namespace smallest_positive_value_l19_19370

noncomputable def exprA := 30 - 4 * Real.sqrt 14
noncomputable def exprB := 4 * Real.sqrt 14 - 30
noncomputable def exprC := 25 - 6 * Real.sqrt 15
noncomputable def exprD := 75 - 15 * Real.sqrt 30
noncomputable def exprE := 15 * Real.sqrt 30 - 75

theorem smallest_positive_value :
  exprC = 25 - 6 * Real.sqrt 15 ∧
  exprC < exprA ∧
  exprC < exprB ∧
  exprC < exprD ∧
  exprC < exprE ∧
  exprC > 0 :=
by sorry

end smallest_positive_value_l19_19370


namespace solve_equation_l19_19347

theorem solve_equation (x : ℝ) : x * (x + 5)^3 * (5 - x) = 0 ↔ x = 0 ∨ x = -5 ∨ x = 5 := by
  sorry

end solve_equation_l19_19347


namespace rectangle_area_l19_19060

theorem rectangle_area (x y : ℝ) (hx : x ≠ 0) (h : x * y = 10) : y = 10 / x :=
sorry

end rectangle_area_l19_19060


namespace ball_box_arrangement_l19_19414

-- Given n distinguishable balls and m distinguishable boxes,
-- prove that the number of ways to place the n balls into the m boxes is m^n.
-- Specifically for n = 6 and m = 3.

theorem ball_box_arrangement : (3^6 = 729) :=
by
  sorry

end ball_box_arrangement_l19_19414


namespace thomas_total_blocks_l19_19785

def stack1 := 7
def stack2 := stack1 + 3
def stack3 := stack2 - 6
def stack4 := stack3 + 10
def stack5 := stack2 * 2

theorem thomas_total_blocks : stack1 + stack2 + stack3 + stack4 + stack5 = 55 := by
  sorry

end thomas_total_blocks_l19_19785


namespace max_sum_first_n_terms_l19_19655

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem max_sum_first_n_terms (a1 : ℝ) (h1 : a1 > 0)
  (h2 : 5 * a_n a1 d 8 = 8 * a_n a1 d 13) :
  ∃ n : ℕ, n = 21 ∧ ∀ m : ℕ, S_n a1 d m ≤ S_n a1 d n :=
by
  sorry

end max_sum_first_n_terms_l19_19655


namespace transportation_tax_correct_l19_19781

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def total_months_in_year : ℕ := 12

def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / total_months_in_year

theorem transportation_tax_correct :
  adjusted_tax = 3125 := by
  sorry

end transportation_tax_correct_l19_19781


namespace calculate_expression_l19_19195

theorem calculate_expression : ((-3: ℤ) ^ 3 + (5: ℤ) ^ 2 - ((-2: ℤ) ^ 2)) = -6 := by
  sorry

end calculate_expression_l19_19195


namespace M_lies_in_third_quadrant_l19_19245

noncomputable def harmonious_point (a b : ℝ) : Prop :=
  3 * a = 2 * b + 5

noncomputable def point_M_harmonious (m : ℝ) : Prop :=
  harmonious_point (m - 1) (3 * m + 2)

theorem M_lies_in_third_quadrant (m : ℝ) (hM : point_M_harmonious m) : 
  (m - 1 < 0 ∧ 3 * m + 2 < 0) :=
by {
  sorry
}

end M_lies_in_third_quadrant_l19_19245


namespace min_value_inequality_l19_19929

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z ≥ 36) ∧ 
  ((1 / x + 4 / y + 9 / z = 36) ↔ (x = 1 / 6 ∧ y = 1 / 3 ∧ z = 1 / 2)) :=
by
  sorry

end min_value_inequality_l19_19929


namespace meals_second_restaurant_l19_19719

theorem meals_second_restaurant (r1 r2 r3 total_weekly_meals : ℕ) 
    (H1 : r1 = 20) 
    (H3 : r3 = 50) 
    (H_total : total_weekly_meals = 770) : 
    (7 * r2) = 280 := 
by 
    sorry

example (r2 : ℕ) : (40 = r2) :=
    by sorry

end meals_second_restaurant_l19_19719


namespace solve_fraction_equation_l19_19446

theorem solve_fraction_equation (x : ℝ) (h : (x + 5) / (x - 3) = 4) : x = 17 / 3 :=
by
  sorry

end solve_fraction_equation_l19_19446


namespace smallest_k_l19_19216

theorem smallest_k (M : Finset ℕ) (H : ∀ (a b c d : ℕ), a ∈ M → b ∈ M → c ∈ M → d ∈ M → a ≠ b → b ≠ c → c ≠ d → d ≠ a → 20 ∣ (a - b + c - d)) :
  ∃ k, k = 7 ∧ ∀ (M' : Finset ℕ), M'.card = k → ∀ (a b c d : ℕ), a ∈ M' → b ∈ M' → c ∈ M' → d ∈ M' → a ≠ b → b ≠ c → c ≠ d → d ≠ a → 20 ∣ (a - b + c - d) :=
sorry

end smallest_k_l19_19216


namespace median_of_consecutive_integers_l19_19041

def sum_of_consecutive_integers (n : ℕ) (a : ℤ) : ℤ :=
  n * (2*a + (n - 1)) / 2

theorem median_of_consecutive_integers (a : ℤ) : 
  (sum_of_consecutive_integers 25 a = 5^5) -> 
  (a + 12 = 125) := 
by
  sorry

end median_of_consecutive_integers_l19_19041


namespace find_a_value_l19_19577

/-- Given the distribution of the random variable ξ as p(ξ = k) = a (1/3)^k for k = 1, 2, 3, 
    prove that the value of a that satisfies the probabilities summing to 1 is 27/13. -/
theorem find_a_value (a : ℝ) :
  (a * (1 / 3) + a * (1 / 3)^2 + a * (1 / 3)^3 = 1) → a = 27 / 13 :=
by 
  intro h
  sorry

end find_a_value_l19_19577


namespace highest_throw_is_37_feet_l19_19268

theorem highest_throw_is_37_feet :
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  max (max C1 (max C2 C3)) (max J1 (max J2 J3)) = 37 := by
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  sorry

end highest_throw_is_37_feet_l19_19268


namespace remaining_amount_after_shopping_l19_19763

theorem remaining_amount_after_shopping (initial_amount spent_percentage remaining_amount : ℝ)
  (h_initial : initial_amount = 4000)
  (h_spent : spent_percentage = 0.30)
  (h_remaining : remaining_amount = 2800) :
  initial_amount - (spent_percentage * initial_amount) = remaining_amount :=
by
  sorry

end remaining_amount_after_shopping_l19_19763


namespace recurring_decimal_of_division_l19_19697

theorem recurring_decimal_of_division (a b : ℤ) (h1 : a = 60) (h2 : b = 55) : (a : ℝ) / (b : ℝ) = 1.09090909090909090909090909090909 :=
by
  -- Import the necessary definitions and facts
  sorry

end recurring_decimal_of_division_l19_19697


namespace max_fraction_diagonals_sides_cyclic_pentagon_l19_19267

theorem max_fraction_diagonals_sides_cyclic_pentagon (a b c d e A B C D E : ℝ)
  (h1 : b * e + a * A = C * D)
  (h2 : c * a + b * B = D * E)
  (h3 : d * b + c * C = E * A)
  (h4 : e * c + d * D = A * B)
  (h5 : a * d + e * E = B * C) :
  (a * b * c * d * e) / (A * B * C * D * E) ≤ (5 * Real.sqrt 5 - 11) / 2 :=
sorry

end max_fraction_diagonals_sides_cyclic_pentagon_l19_19267


namespace product_divisible_by_eight_l19_19455

theorem product_divisible_by_eight (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 96) : 
  8 ∣ n * (n + 1) * (n + 2) := 
sorry

end product_divisible_by_eight_l19_19455


namespace fraction_susan_can_eat_l19_19830

theorem fraction_susan_can_eat
  (v t n nf : ℕ)
  (h₁ : v = 6)
  (h₂ : n = 4)
  (h₃ : 1/3 * t = v)
  (h₄ : nf = v - n) :
  nf / t = 1 / 9 :=
sorry

end fraction_susan_can_eat_l19_19830


namespace find_third_number_in_second_set_l19_19588

theorem find_third_number_in_second_set (x y: ℕ) 
    (h1 : (28 + x + 42 + 78 + 104) / 5 = 90) 
    (h2 : (128 + 255 + y + 1023 + x) / 5 = 423) 
: y = 511 := 
sorry

end find_third_number_in_second_set_l19_19588


namespace wall_height_l19_19817

noncomputable def brick_volume : ℝ := 25 * 11.25 * 6

noncomputable def total_brick_volume : ℝ := brick_volume * 6400

noncomputable def wall_length : ℝ := 800

noncomputable def wall_width : ℝ := 600

theorem wall_height :
  ∀ (wall_volume : ℝ), 
  wall_volume = total_brick_volume → 
  wall_volume = wall_length * wall_width * 22.48 :=
by
  sorry

end wall_height_l19_19817


namespace jerry_daughters_games_l19_19271

theorem jerry_daughters_games (x y : ℕ) (h : 4 * x + 2 * x + 4 * y + 2 * y = 96) (hx : x = y) :
  x = 8 ∧ y = 8 :=
by
  have h1 : 6 * x + 6 * y = 96 := by linarith
  have h2 : x = y := hx
  sorry

end jerry_daughters_games_l19_19271


namespace area_enclosed_by_graph_l19_19975

noncomputable def enclosed_area (x y : ℝ) : ℝ := 
  if h : (|5 * x| + |3 * y| = 15) then
    30 -- The area enclosed by the graph
  else
    0 -- Default case for definition completeness

theorem area_enclosed_by_graph : ∀ (x y : ℝ), (|5 * x| + |3 * y| = 15) → enclosed_area x y = 30 :=
by
  sorry

end area_enclosed_by_graph_l19_19975


namespace coleen_sprinkles_l19_19330

theorem coleen_sprinkles : 
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  remaining_sprinkles = 3 :=
by
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  sorry

end coleen_sprinkles_l19_19330


namespace minimum_teachers_to_cover_all_subjects_l19_19138

/- Define the problem conditions -/
def maths_teachers := 7
def physics_teachers := 6
def chemistry_teachers := 5
def max_subjects_per_teacher := 3

/- The proof statement -/
theorem minimum_teachers_to_cover_all_subjects : 
  (maths_teachers + physics_teachers + chemistry_teachers) / max_subjects_per_teacher = 7 :=
sorry

end minimum_teachers_to_cover_all_subjects_l19_19138


namespace tangent_line_passes_through_origin_l19_19013

noncomputable def curve (α : ℝ) (x : ℝ) : ℝ := x^α + 1

theorem tangent_line_passes_through_origin (α : ℝ)
  (h_tangent : ∀ (x : ℝ), curve α 1 + (α * (x - 1)) - 2 = curve α x) :
  α = 2 :=
sorry

end tangent_line_passes_through_origin_l19_19013


namespace find_a_l19_19063

-- Points A and B on the x-axis
def point_A (a : ℝ) : (ℝ × ℝ) := (a, 0)
def point_B : (ℝ × ℝ) := (-3, 0)

-- Distance condition
def distance_condition (a : ℝ) : Prop := abs (a + 3) = 5

-- The proof problem: find a such that distance condition holds
theorem find_a (a : ℝ) : distance_condition a ↔ (a = -8 ∨ a = 2) :=
by
  sorry

end find_a_l19_19063


namespace remainder_division_l19_19522

theorem remainder_division (β : ℂ) 
  (h1 : β^6 + β^5 + β^4 + β^3 + β^2 + β + 1 = 0) 
  (h2 : β^7 = 1) : (β^100 + β^75 + β^50 + β^25 + 1) % (β^6 + β^5 + β^4 + β^3 + β^2 + β + 1) = -1 :=
by
  sorry

end remainder_division_l19_19522


namespace sin_cos_identity_l19_19054

variables (α : ℝ)

def tan_pi_add_alpha (α : ℝ) : Prop := Real.tan (Real.pi + α) = 3

theorem sin_cos_identity (h : tan_pi_add_alpha α) : 
  Real.sin (-α) * Real.cos (Real.pi - α) = 3 / 10 :=
sorry

end sin_cos_identity_l19_19054


namespace correct_calculation_A_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_correct_answer_is_A_l19_19358

theorem correct_calculation_A : (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) :=
by { sorry }

theorem incorrect_calculation_B : (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) :=
by { sorry }

theorem incorrect_calculation_C : ((Real.sqrt 2)^2 ≠ 2 * Real.sqrt 2) :=
by { sorry }

theorem incorrect_calculation_D : (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) :=
by { sorry }

theorem correct_answer_is_A :
  (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) ∧
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) ∧
  ((Real.sqrt 2)^2 ≠ 2 * Real.sqrt 2) ∧
  (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) :=
by {
  exact ⟨correct_calculation_A, incorrect_calculation_B, incorrect_calculation_C, incorrect_calculation_D⟩
}

end correct_calculation_A_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_correct_answer_is_A_l19_19358


namespace black_cars_count_l19_19684

theorem black_cars_count
    (r b : ℕ)
    (r_ratio : r = 33)
    (ratio_condition : r / b = 3 / 8) :
    b = 88 :=
by 
  sorry

end black_cars_count_l19_19684


namespace find_m_for_eccentric_ellipse_l19_19739

theorem find_m_for_eccentric_ellipse (m : ℝ) : 
  (∀ x y : ℝ, (x^2)/5 + (y^2)/m = 1) ∧
  (∀ e : ℝ, e = (Real.sqrt 10)/5) → 
  (m = 25/3 ∨ m = 3) := sorry

end find_m_for_eccentric_ellipse_l19_19739


namespace integer_root_abs_sum_l19_19346

noncomputable def solve_abs_sum (p q r : ℤ) : ℤ := |p| + |q| + |r|

theorem integer_root_abs_sum (p q r m : ℤ) 
  (h1 : p + q + r = 0)
  (h2 : p * q + q * r + r * p = -2024)
  (h3 : ∃ m, ∀ x, x^3 - 2024 * x + m = (x - p) * (x - q) * (x - r)) :
  solve_abs_sum p q r = 104 :=
by sorry

end integer_root_abs_sum_l19_19346


namespace orange_juice_production_l19_19760

theorem orange_juice_production :
  let total_oranges := 8 -- in million tons
  let exported_oranges := total_oranges * 0.25
  let remaining_oranges := total_oranges - exported_oranges
  let juice_oranges_ratio := 0.60
  let juice_oranges := remaining_oranges * juice_oranges_ratio
  juice_oranges = 3.6  :=
by
  sorry

end orange_juice_production_l19_19760


namespace cost_price_article_l19_19384

variable (SP : ℝ := 21000)
variable (d : ℝ := 0.10)
variable (p : ℝ := 0.08)

theorem cost_price_article : (SP * (1 - d)) / (1 + p) = 17500 := by
  sorry

end cost_price_article_l19_19384


namespace opposite_of_number_l19_19517

-- Define the original number
def original_number : ℚ := -1 / 6

-- Statement to prove
theorem opposite_of_number : -original_number = 1 / 6 := by
  -- This is where the proof would go
  sorry

end opposite_of_number_l19_19517


namespace girl_walked_distance_l19_19124

-- Define the conditions
def speed : ℝ := 5 -- speed in kmph
def time : ℝ := 6 -- time in hours

-- Define the distance calculation
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- The proof statement that we need to show
theorem girl_walked_distance :
  distance speed time = 30 := by
  sorry

end girl_walked_distance_l19_19124


namespace prime_sum_l19_19164

theorem prime_sum (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h : 2 * p + 3 * q = 6 * r) : 
  p + q + r = 7 := 
sorry

end prime_sum_l19_19164


namespace problem1_problem2_problem3_problem4_problem5_l19_19385

-- Definitions and conditions
variable (a : ℝ) (b : ℝ) (ha : a > 0) (hb : b > 0) (hineq : a - 2 * Real.sqrt b > 0)

-- Problem 1: √(a - 2√b) = √m - √n
theorem problem1 (h₁ : a = 5) (h₂ : b = 6) : Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 := sorry

-- Problem 2: √(a + 2√b) = √m + √n
theorem problem2 (h₁ : a = 12) (h₂ : b = 35) : Real.sqrt (12 + 2 * Real.sqrt 35) = Real.sqrt 7 + Real.sqrt 5 := sorry

-- Problem 3: √(a + 6√b) = √m + √n
theorem problem3 (h₁ : a = 9) (h₂ : b = 6) : Real.sqrt (9 + 6 * Real.sqrt 2) = Real.sqrt 6 + Real.sqrt 3 := sorry

-- Problem 4: √(a - 4√b) = √m - √n
theorem problem4 (h₁ : a = 16) (h₂ : b = 60) : Real.sqrt (16 - 4 * Real.sqrt 15) = Real.sqrt 10 - Real.sqrt 6 := sorry

-- Problem 5: √(a - √b) + √(c + √d)
theorem problem5 (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 2) (h₄ : d = 3) 
  : Real.sqrt (3 - Real.sqrt 5) + Real.sqrt (2 + Real.sqrt 3) = (Real.sqrt 10 + Real.sqrt 6) / 2 := sorry

end problem1_problem2_problem3_problem4_problem5_l19_19385


namespace calculate_initial_money_l19_19434

noncomputable def initial_money (remaining_money: ℝ) (spent_percent: ℝ) : ℝ :=
  remaining_money / (1 - spent_percent)

theorem calculate_initial_money :
  initial_money 3500 0.30 = 5000 := 
by
  rw [initial_money]
  sorry

end calculate_initial_money_l19_19434


namespace max_value_of_f_l19_19703

open Real

noncomputable def f (x : ℝ) : ℝ := -x - 9 / x + 18

theorem max_value_of_f : ∀ x > 0, f x ≤ 12 :=
by
  sorry

end max_value_of_f_l19_19703


namespace exists_quadratic_function_l19_19353

theorem exists_quadratic_function :
  (∃ (a b c : ℝ), ∀ (k : ℕ), k > 0 → (a * (5 / 9 * (10^k - 1))^2 + b * (5 / 9 * (10^k - 1)) + c = 5/9 * (10^(2*k) - 1))) :=
by
  have a := 9 / 5
  have b := 2
  have c := 0
  use a, b, c
  intros k hk
  sorry

end exists_quadratic_function_l19_19353


namespace minimal_period_of_sum_l19_19960

theorem minimal_period_of_sum (A B : ℝ)
  (hA : ∃ p : ℕ, p = 6 ∧ (∃ (x : ℝ) (l : ℕ), A = x / (10 ^ l * (10 ^ p - 1))))
  (hB : ∃ p : ℕ, p = 12 ∧ (∃ (y : ℝ) (m : ℕ), B = y / (10 ^ m * (10 ^ p - 1)))) :
  ∃ p : ℕ, p = 12 ∧ (∃ (z : ℝ) (n : ℕ), A + B = z / (10 ^ n * (10 ^ p - 1))) :=
sorry

end minimal_period_of_sum_l19_19960


namespace non_drinkers_count_l19_19157

-- Define the total number of businessmen and the sets of businessmen drinking each type of beverage.
def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 12
def soda_drinkers : ℕ := 8
def coffee_tea_drinkers : ℕ := 7
def tea_soda_drinkers : ℕ := 3
def coffee_soda_drinkers : ℕ := 2
def all_three_drinkers : ℕ := 1

-- Statement to prove:
theorem non_drinkers_count :
  total_businessmen - (coffee_drinkers + tea_drinkers + soda_drinkers - coffee_tea_drinkers - tea_soda_drinkers - coffee_soda_drinkers + all_three_drinkers) = 6 :=
by
  -- Skip the proof for now.
  sorry

end non_drinkers_count_l19_19157


namespace solution_exists_l19_19791

def divide_sum_of_squares_and_quotient_eq_seventy_two (x : ℝ) : Prop :=
  (10 - x)^2 + x^2 + (10 - x) / x = 72

theorem solution_exists (x : ℝ) : divide_sum_of_squares_and_quotient_eq_seventy_two x → x = 2 := sorry

end solution_exists_l19_19791


namespace average_of_r_s_t_l19_19158

theorem average_of_r_s_t
  (r s t : ℝ)
  (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 :=
by
  sorry

end average_of_r_s_t_l19_19158


namespace rectangle_same_color_exists_l19_19943

theorem rectangle_same_color_exists (color : ℝ × ℝ → Prop) (red blue : Prop) (h : ∀ p : ℝ × ℝ, color p = red ∨ color p = blue) :
  ∃ (a b c d : ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  (color a = color b ∧ color b = color c ∧ color c = color d) :=
sorry

end rectangle_same_color_exists_l19_19943


namespace find_possible_values_l19_19016

noncomputable def possible_values (a b : ℝ) : Set ℝ :=
  { x | ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1/a + 1/b) }

theorem find_possible_values :
  (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 2 → (1 / a + 1 / b) ∈ Set.Ici 2) ∧
  (∀ y, y ∈ Set.Ici 2 → ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ y = (1 / a + 1 / b)) :=
by
  sorry

end find_possible_values_l19_19016


namespace third_number_sixth_row_l19_19045

/-- Define the arithmetic sequence and related properties. -/
def sequence (n : ℕ) : ℕ := 2 * n - 1

/-- Define sum of first k terms in a series where each row length doubles the previous row length. -/
def sum_of_rows (k : ℕ) : ℕ :=
  2^k - 1

/-- Statement of the problem: Prove that the third number in the sixth row is 67. -/
theorem third_number_sixth_row : sequence (sum_of_rows 5 + 3) = 67 := by
  sorry

end third_number_sixth_row_l19_19045


namespace arithmetic_sequence_general_term_and_sum_l19_19773

theorem arithmetic_sequence_general_term_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 2 = 2) →
  (a 4 = 4) →
  (∀ n, a n = n) →
  (∀ n, b n = 2 ^ (a n)) →
  (∀ n, S n = 2 * (2 ^ n - 1)) :=
by
  intros h1 h2 h3 h4
  -- Proof part is skipped
  sorry

end arithmetic_sequence_general_term_and_sum_l19_19773


namespace race_positions_l19_19615

theorem race_positions
  (positions : Fin 15 → String) 
  (h_quinn_lucas : ∃ n : Fin 15, positions n = "Quinn" ∧ positions (n + 4) = "Lucas")
  (h_oliver_quinn : ∃ n : Fin 15, positions (n - 1) = "Oliver" ∧ positions n = "Quinn")
  (h_naomi_oliver : ∃ n : Fin 15, positions n = "Naomi" ∧ positions (n + 3) = "Oliver")
  (h_emma_lucas : ∃ n : Fin 15, positions n = "Lucas" ∧ positions (n + 1) = "Emma")
  (h_sara_naomi : ∃ n : Fin 15, positions n = "Naomi" ∧ positions (n + 1) = "Sara")
  (h_naomi_4th : ∃ n : Fin 15, n = 3 ∧ positions n = "Naomi") :
  positions 6 = "Oliver" :=
by
  sorry

end race_positions_l19_19615


namespace find_k_l19_19350

theorem find_k (a b c k : ℝ) 
  (h : ∀ x : ℝ, 
    (a * x^2 + b * x + c + b * x^2 + a * x - 7 + k * x^2 + c * x + 3) / (x^2 - 2 * x - 5) = (x^2 - 2*x - 5)) :
  k = 2 :=
by
  sorry

end find_k_l19_19350


namespace roots_quadratic_l19_19966

theorem roots_quadratic (a b : ℝ) 
  (h1: a^2 + 3 * a - 2010 = 0) 
  (h2: b^2 + 3 * b - 2010 = 0)
  (h_roots: a + b = -3 ∧ a * b = -2010):
  a^2 - a - 4 * b = 2022 :=
by
  sorry

end roots_quadratic_l19_19966


namespace find_abc_triplet_l19_19711

theorem find_abc_triplet (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_order : a < b ∧ b < c) 
  (h_eqn : (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) = (a + b + c) / 2) :
  ∃ d : ℕ, d > 0 ∧ ((a = d ∧ b = 2 * d ∧ c = 3 * d) ∨ (a = d ∧ b = 3 * d ∧ c = 6 * d)) :=
  sorry

end find_abc_triplet_l19_19711


namespace required_cups_of_sugar_l19_19494

-- Define the original ratios
def original_flour_water_sugar_ratio : Rat := 10 / 6 / 3
def new_flour_water_ratio : Rat := 2 * (10 / 6)
def new_flour_sugar_ratio : Rat := (1 / 2) * (10 / 3)

-- Given conditions
def cups_of_water : Rat := 2

-- Problem statement: prove the amount of sugar required
theorem required_cups_of_sugar : ∀ (sugar_cups : Rat),
  original_flour_water_sugar_ratio = 10 / 6 / 3 ∧
  new_flour_water_ratio = 2 * (10 / 6) ∧
  new_flour_sugar_ratio = (1 / 2) * (10 / 3) ∧
  cups_of_water = 2 ∧
  (6 / 12) = (2 / sugar_cups) → sugar_cups = 4 := by
  intro sugar_cups
  sorry

end required_cups_of_sugar_l19_19494


namespace convex_polygon_diagonals_25_convex_polygon_triangles_25_l19_19634

-- Define a convex polygon with 25 sides
def convex_polygon_sides : ℕ := 25

-- Define the number of diagonals in a convex polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Define the number of triangles that can be formed by choosing any three vertices from n vertices
def number_of_triangles (n : ℕ) : ℕ := n.choose 3

-- Theorem to prove the number of diagonals is 275 for a convex polygon with 25 sides
theorem convex_polygon_diagonals_25 : number_of_diagonals convex_polygon_sides = 275 :=
by sorry

-- Theorem to prove the number of triangles is 2300 for a convex polygon with 25 sides
theorem convex_polygon_triangles_25 : number_of_triangles convex_polygon_sides = 2300 :=
by sorry

end convex_polygon_diagonals_25_convex_polygon_triangles_25_l19_19634


namespace cos_diff_identity_l19_19089

variable {α : ℝ}

def sin_alpha := -3 / 5

def alpha_interval (α : ℝ) : Prop :=
  (3 * Real.pi / 2 < α) ∧ (α < 2 * Real.pi)

theorem cos_diff_identity (h1 : Real.sin α = sin_alpha) (h2 : alpha_interval α) :
  Real.cos (Real.pi / 4 - α) = Real.sqrt 2 / 10 :=
  sorry

end cos_diff_identity_l19_19089


namespace length_increase_100_l19_19613

theorem length_increase_100 (n : ℕ) (h : (n + 2) / 2 = 100) : n = 198 :=
sorry

end length_increase_100_l19_19613


namespace charity_event_equation_l19_19919

variable (x : ℕ)

theorem charity_event_equation : x + 5 * (12 - x) = 48 :=
sorry

end charity_event_equation_l19_19919


namespace original_price_l19_19657

theorem original_price (P : ℝ) (h1 : P + 0.10 * P = 330) : P = 300 := 
by
  sorry

end original_price_l19_19657


namespace absolute_value_inequality_l19_19248

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 4) ↔ (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7) := 
by sorry

end absolute_value_inequality_l19_19248


namespace roots_quadratic_equation_l19_19075

theorem roots_quadratic_equation (x1 x2 : ℝ) (h1 : x1^2 - x1 - 1 = 0) (h2 : x2^2 - x2 - 1 = 0) :
  (x2 / x1) + (x1 / x2) = -3 :=
by
  sorry

end roots_quadratic_equation_l19_19075


namespace least_candies_to_remove_for_equal_distribution_l19_19304

theorem least_candies_to_remove_for_equal_distribution :
  ∃ k : ℕ, k = 4 ∧ ∀ n : ℕ, 24 - k = 5 * n :=
sorry

end least_candies_to_remove_for_equal_distribution_l19_19304


namespace range_of_x_l19_19467

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) :
  (2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
   abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2)
  ↔ (Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4) :=
by
  sorry

end range_of_x_l19_19467


namespace sum_six_smallest_multiples_of_12_is_252_l19_19428

-- Define the six smallest positive distinct multiples of 12
def six_smallest_multiples_of_12 := [12, 24, 36, 48, 60, 72]

-- Define the sum problem
def sum_of_six_smallest_multiples_of_12 : Nat :=
  six_smallest_multiples_of_12.foldr (· + ·) 0

-- Main proof statement
theorem sum_six_smallest_multiples_of_12_is_252 :
  sum_of_six_smallest_multiples_of_12 = 252 :=
by
  sorry

end sum_six_smallest_multiples_of_12_is_252_l19_19428


namespace average_production_l19_19269

theorem average_production (n : ℕ) (P : ℕ) (h1 : P = 60 * n) (h2 : (P + 90) / (n + 1) = 62) : n = 14 :=
  sorry

end average_production_l19_19269


namespace garrett_granola_bars_l19_19179

theorem garrett_granola_bars :
  ∀ (oatmeal_raisin peanut total : ℕ),
  peanut = 8 →
  total = 14 →
  oatmeal_raisin + peanut = total →
  oatmeal_raisin = 6 :=
by
  intros oatmeal_raisin peanut total h_peanut h_total h_sum
  sorry

end garrett_granola_bars_l19_19179


namespace ellipse_eccentricity_l19_19906

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l19_19906


namespace max_peaceful_clients_kept_l19_19259

-- Defining the types for knights, liars, and troublemakers
def Person : Type := ℕ

noncomputable def isKnight : Person → Prop := sorry
noncomputable def isLiar : Person → Prop := sorry
noncomputable def isTroublemaker : Person → Prop := sorry

-- Total number of people in the bar
def totalPeople : ℕ := 30

-- Number of knights, liars, and troublemakers
def numberKnights : ℕ := 10
def numberLiars : ℕ := 10
def numberTroublemakers : ℕ := 10

-- The bartender's goal: get rid of all troublemakers and keep as many peaceful clients as possible
def maxPeacefulClients (total: ℕ) (knights: ℕ) (liars: ℕ) (troublemakers: ℕ): ℕ :=
  total - troublemakers

-- Statement to be proved
theorem max_peaceful_clients_kept (total: ℕ) (knights: ℕ) (liars: ℕ) (troublemakers: ℕ)
  (h_total : total = 30)
  (h_knights : knights = 10)
  (h_liars : liars = 10)
  (h_troublemakers : troublemakers = 10) :
  maxPeacefulClients total knights liars troublemakers = 19 :=
by
  -- Proof steps go here
  sorry

end max_peaceful_clients_kept_l19_19259


namespace johns_cookies_left_l19_19067

def dozens_to_cookies (d : ℕ) : ℕ := d * 12 -- Definition to convert dozens to actual cookie count

def cookies_left (initial_cookies : ℕ) (eaten_cookies : ℕ) : ℕ := initial_cookies - eaten_cookies -- Definition to calculate remaining cookies

theorem johns_cookies_left : cookies_left (dozens_to_cookies 2) 3 = 21 :=
by
  -- Given that John buys 2 dozen cookies
  -- And he eats 3 cookies
  -- We need to prove that he has 21 cookies left
  sorry  -- Proof is omitted as per instructions

end johns_cookies_left_l19_19067


namespace csc_square_value_l19_19121

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 ∨ x = 1 then 0 -- provision for the illegal inputs as defined in the question
else 1/(x / (x - 1))

theorem csc_square_value (t : ℝ) (ht : 0 ≤ t ∧ t ≤ π / 2) :
  f (1 / (Real.sin t)^2) = (Real.cos t)^2 :=
by
  sorry

end csc_square_value_l19_19121


namespace part1_part2_l19_19362

def A : Set ℝ := {x | x^2 + x - 12 < 0}
def B : Set ℝ := {x | 4 / (x + 3) ≤ 1}
def C (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x + m^2 - 1 ≤ 0}

theorem part1 : A ∩ B = {x | -4 < x ∧ x < -3 ∨ 1 ≤ x ∧ x < 3} := sorry

theorem part2 (m : ℝ) : (-3 < m ∧ m < 2) ↔ ∀ x, (x ∈ A → x ∈ C m) ∧ ∃ x, x ∈ C m ∧ x ∉ A := sorry

end part1_part2_l19_19362


namespace cost_of_gas_l19_19083

def hoursDriven1 : ℕ := 2
def speed1 : ℕ := 60
def hoursDriven2 : ℕ := 3
def speed2 : ℕ := 50
def milesPerGallon : ℕ := 30
def costPerGallon : ℕ := 2

def totalDistance : ℕ := (hoursDriven1 * speed1) + (hoursDriven2 * speed2)
def gallonsUsed : ℕ := totalDistance / milesPerGallon
def totalCost : ℕ := gallonsUsed * costPerGallon

theorem cost_of_gas : totalCost = 18 := by
  -- You should fill in the proof steps here.
  sorry

end cost_of_gas_l19_19083


namespace compound_proposition_l19_19924

theorem compound_proposition (Sn P Q : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → Sn n = 2 * n^2 + 3 * n + 1) →
  (∀ n : ℕ, n > 0 → Sn n = 2 * P n + 1) →
  (¬(∀ n, n > 0 → ∃ d, (P (n + 1) - P n) = d)) ∧ (∀ n, n > 0 → P n = Q (n - 1)) :=
by
  sorry

end compound_proposition_l19_19924


namespace stock_worth_l19_19456

theorem stock_worth (profit_part loss_part total_loss : ℝ) 
  (h1 : profit_part = 0.10) 
  (h2 : loss_part = 0.90) 
  (h3 : total_loss = 400) 
  (profit_rate : ℝ := 0.20) 
  (loss_rate : ℝ := 0.05)
  (profit_value := profit_rate * profit_part)
  (loss_value := loss_rate * loss_part)
  (overall_loss := total_loss)
  (h4 : loss_value - profit_value = overall_loss) :
  ∃ X : ℝ, X = 16000 :=
by
  sorry

end stock_worth_l19_19456


namespace complex_expression_evaluation_l19_19508

-- Defining the imaginary unit
def i : ℂ := Complex.I

-- Defining the complex number z
def z : ℂ := 1 - i

-- Stating the theorem to prove
theorem complex_expression_evaluation : z^2 + (2 / z) = 1 - i := by
  sorry

end complex_expression_evaluation_l19_19508


namespace sunset_time_l19_19991

theorem sunset_time (length_of_daylight : Nat := 11 * 60 + 18) -- length of daylight in minutes
    (sunrise : Nat := 6 * 60 + 32) -- sunrise time in minutes after midnight
    : (sunrise + length_of_daylight) % (24 * 60) = 17 * 60 + 50 := -- sunset time calculation
by
  sorry

end sunset_time_l19_19991


namespace students_like_both_l19_19254

-- Definitions based on given conditions
def total_students : ℕ := 500
def students_like_mountains : ℕ := 289
def students_like_sea : ℕ := 337
def students_like_neither : ℕ := 56

-- Statement to prove
theorem students_like_both : 
  students_like_mountains + students_like_sea - 182 + students_like_neither = total_students := 
by
  sorry

end students_like_both_l19_19254


namespace ratio_of_numbers_l19_19262

theorem ratio_of_numbers
  (greater less : ℕ)
  (h1 : greater = 64)
  (h2 : less = 32)
  (h3 : greater + less = 96)
  (h4 : ∃ k : ℕ, greater = k * less) :
  greater / less = 2 := by
  sorry

end ratio_of_numbers_l19_19262


namespace problem_statement_l19_19332

theorem problem_statement (x y : ℝ) (h : |x + 1| + |y + 2 * x| = 0) : (x + y) ^ 2004 = 1 := by
  sorry

end problem_statement_l19_19332


namespace fraction_study_only_japanese_l19_19715

variable (J : ℕ)

def seniors := 2 * J
def sophomores := (3 / 4) * J

def seniors_study_japanese := (3 / 8) * seniors J
def juniors_study_japanese := (1 / 4) * J
def sophomores_study_japanese := (2 / 5) * sophomores J

def seniors_study_both := (1 / 6) * seniors J
def juniors_study_both := (1 / 12) * J
def sophomores_study_both := (1 / 10) * sophomores J

def seniors_study_only_japanese := seniors_study_japanese J - seniors_study_both J
def juniors_study_only_japanese := juniors_study_japanese J - juniors_study_both J
def sophomores_study_only_japanese := sophomores_study_japanese J - sophomores_study_both J

def total_study_only_japanese := seniors_study_only_japanese J + juniors_study_only_japanese J + sophomores_study_only_japanese J
def total_students := J + seniors J + sophomores J

theorem fraction_study_only_japanese :
  (total_study_only_japanese J) / (total_students J) = 97 / 450 :=
by sorry

end fraction_study_only_japanese_l19_19715


namespace sprinkler_days_needed_l19_19307

-- Definitions based on the conditions
def morning_water : ℕ := 4
def evening_water : ℕ := 6
def daily_water : ℕ := morning_water + evening_water
def total_water_needed : ℕ := 50

-- The proof statement
theorem sprinkler_days_needed : total_water_needed / daily_water = 5 := by
  sorry

end sprinkler_days_needed_l19_19307


namespace perfect_square_of_seq_l19_19255

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n ≥ 3, a n = 7 * a (n - 1) - a (n - 2)

theorem perfect_square_of_seq (a : ℕ → ℤ) (h : seq a) (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ, k * k = a n + 2 + a (n + 1) :=
sorry

end perfect_square_of_seq_l19_19255


namespace max_intersections_arith_geo_seq_l19_19087

def arithmetic_sequence (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d

def geometric_sequence (n : ℕ) (q : ℝ) : ℝ := q ^ (n - 1)

theorem max_intersections_arith_geo_seq (d : ℝ) (q : ℝ) (h_d : d ≠ 0) (h_q_pos : q > 0) (h_q_neq1 : q ≠ 1) :
  (∃ n : ℕ, arithmetic_sequence n d = geometric_sequence n q) → ∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ (arithmetic_sequence n₁ d = geometric_sequence n₁ q) ∧ (arithmetic_sequence n₂ d = geometric_sequence n₂ q) :=
sorry

end max_intersections_arith_geo_seq_l19_19087


namespace vasya_tolya_badges_l19_19309

theorem vasya_tolya_badges (x y : ℤ)
    (h1 : y = x + 5) -- Vasya initially had 5 more badges than Tolya
    (h2 : (y - (6 * (y / 25) / 25) + (4 * x) / 25) = (x - (4 * x) / 5 + 6 * (y / 25) / 5 - 1)) : -- equation balancing after exchange
    x = 45 ∧ y = 50 := 
sorry

end vasya_tolya_badges_l19_19309


namespace greatest_integer_function_of_pi_plus_3_l19_19246

noncomputable def pi_plus_3 : Real := Real.pi + 3

theorem greatest_integer_function_of_pi_plus_3 : Int.floor pi_plus_3 = 6 := 
by
  -- sorry is used to skip the proof
  sorry

end greatest_integer_function_of_pi_plus_3_l19_19246


namespace total_cakes_correct_l19_19015

-- Define the initial number of full-size cakes
def initial_cakes : ℕ := 350

-- Define the number of additional full-size cakes made
def additional_cakes : ℕ := 125

-- Define the number of half-cakes made
def half_cakes : ℕ := 75

-- Convert half-cakes to full-size cakes, considering only whole cakes
def half_to_full_cakes := (half_cakes / 2)

-- Total full-size cakes calculation
def total_cakes :=
  initial_cakes + additional_cakes + half_to_full_cakes

-- Prove the total number of full-size cakes
theorem total_cakes_correct : total_cakes = 512 :=
by
  -- Skip the proof
  sorry

end total_cakes_correct_l19_19015


namespace fourth_power_nested_sqrt_l19_19621

noncomputable def nested_sqrt := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_nested_sqrt :
  (nested_sqrt ^ 4) = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) :=
sorry

end fourth_power_nested_sqrt_l19_19621


namespace total_people_veg_l19_19990

-- Definitions based on the conditions
def people_only_veg : ℕ := 13
def people_both_veg_nonveg : ℕ := 6

-- The statement we need to prove
theorem total_people_veg : people_only_veg + people_both_veg_nonveg = 19 :=
by
  sorry

end total_people_veg_l19_19990


namespace difference_not_divisible_by_1976_l19_19750

theorem difference_not_divisible_by_1976 (A B : ℕ) (hA : 100 ≤ A) (hA' : A < 1000) (hB : 100 ≤ B) (hB' : B < 1000) (h : A ≠ B) :
  ¬ (1976 ∣ (1000 * A + B - (1000 * B + A))) :=
by
  sorry

end difference_not_divisible_by_1976_l19_19750


namespace pedro_more_squares_l19_19774

theorem pedro_more_squares (jesus_squares : ℕ) (linden_squares : ℕ) (pedro_squares : ℕ)
  (h1 : jesus_squares = 60) (h2 : linden_squares = 75) (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 :=
by
  sorry

end pedro_more_squares_l19_19774


namespace simplify_fraction_l19_19069

theorem simplify_fraction (n : Nat) : (2^(n+4) - 3 * 2^n) / (2 * 2^(n+3)) = 13 / 16 :=
by
  sorry

end simplify_fraction_l19_19069


namespace value_of_b_minus_d_squared_l19_19574

variable {a b c d : ℤ}

theorem value_of_b_minus_d_squared (h1 : a - b - c + d = 13) (h2 : a + b - c - d = 3) : (b - d) ^ 2 = 25 := 
by
  sorry

end value_of_b_minus_d_squared_l19_19574


namespace center_circle_is_correct_l19_19814

noncomputable def find_center_of_circle : ℝ × ℝ :=
  let line1 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = 20
  let line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -40
  let center_line : ℝ → ℝ → Prop := λ x y => x - 3 * y = 15
  let mid_line : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -10
  (-18, -11)

theorem center_circle_is_correct (x y : ℝ) :
  (let line1 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = 20
   let line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -40
   let center_line : ℝ → ℝ → Prop := λ x y => x - 3 * y = 15
   let mid_line : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -10
   (x, y) = find_center_of_circle) :=
  sorry

end center_circle_is_correct_l19_19814


namespace no_solution_inequalities_l19_19181

theorem no_solution_inequalities (a : ℝ) : (¬ ∃ x : ℝ, 2 * x - 4 > 0 ∧ x - a < 0) → a ≤ 2 := 
by 
  sorry

end no_solution_inequalities_l19_19181


namespace power_multiplication_l19_19483

variable (p : ℝ)  -- Assuming p is a real number

theorem power_multiplication :
  (-p)^2 * (-p)^3 = -p^5 :=
sorry

end power_multiplication_l19_19483


namespace isosceles_trapezoid_ratio_l19_19323

theorem isosceles_trapezoid_ratio (a b d : ℝ) (h1 : b = 2 * d) (h2 : a = d) : a / b = 1 / 2 :=
by
  sorry

end isosceles_trapezoid_ratio_l19_19323


namespace binomial_coefficient_third_term_l19_19011

theorem binomial_coefficient_third_term (x a : ℝ) (h : 10 * a^3 * x = 80) : a = 2 :=
by
  sorry

end binomial_coefficient_third_term_l19_19011


namespace pyramid_volume_l19_19419

-- Define the given conditions
def regular_octagon (A B C D E F G H : Point) : Prop := sorry
def right_pyramid (P A B C D E F G H : Point) : Prop := sorry
def equilateral_triangle (P A D : Point) (side_length : ℝ) : Prop := sorry

-- Define the specific pyramid problem with all the given conditions
noncomputable def volume_pyramid (P A B C D E F G H : Point) (height : ℝ) (base_area : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- The main theorem to prove the volume of the pyramid
theorem pyramid_volume (A B C D E F G H P : Point) 
(h1 : regular_octagon A B C D E F G H)
(h2 : right_pyramid P A B C D E F G H)
(h3 : equilateral_triangle P A D 10) :
  volume_pyramid P A B C D E F G H (5 * Real.sqrt 3) (50 * Real.sqrt 3) = 250 := 
sorry

end pyramid_volume_l19_19419


namespace c_share_l19_19645

theorem c_share (x y z a b c : ℝ) 
  (H1 : b = (65/100) * a)
  (H2 : c = (40/100) * a)
  (H3 : a + b + c = 328) : 
  c = 64 := 
sorry

end c_share_l19_19645


namespace relay_race_time_l19_19395

theorem relay_race_time (R S D : ℕ) (h1 : S = R + 2) (h2 : D = R - 3) (h3 : R + S + D = 71) : R = 24 :=
by
  sorry

end relay_race_time_l19_19395


namespace ivanov_entitled_to_12_million_rubles_l19_19638

def equal_contributions (x : ℝ) : Prop :=
  let ivanov_contribution := 70 * x
  let petrov_contribution := 40 * x
  let sidorov_contribution := 44
  ivanov_contribution = 44 ∧ petrov_contribution = 44 ∧ (ivanov_contribution + petrov_contribution + sidorov_contribution) / 3 = 44

def money_ivanov_receives (x : ℝ) : ℝ :=
  let ivanov_contribution := 70 * x
  ivanov_contribution - 44

theorem ivanov_entitled_to_12_million_rubles :
  ∃ x : ℝ, equal_contributions x → money_ivanov_receives x = 12 :=
sorry

end ivanov_entitled_to_12_million_rubles_l19_19638


namespace sausage_thickness_correct_l19_19258

noncomputable def earth_radius := 6000 -- in km
noncomputable def distance_to_sun := 150000000 -- in km
noncomputable def sausage_thickness := 44 -- in km

theorem sausage_thickness_correct :
  let R := earth_radius
  let L := distance_to_sun
  let r := Real.sqrt ((4 * R^3) / (3 * L))
  abs (r - sausage_thickness) < 10 * sausage_thickness :=
by
  sorry

end sausage_thickness_correct_l19_19258


namespace smallest_n_l19_19311

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 3^n = k^4) (h2 : ∃ l : ℕ, 2^n = l^6) : n = 12 :=
by
  sorry

end smallest_n_l19_19311


namespace files_deleted_is_3_l19_19224

-- Define the initial number of files
def initial_files : Nat := 24

-- Define the remaining number of files
def remaining_files : Nat := 21

-- Define the number of files deleted
def files_deleted : Nat := initial_files - remaining_files

-- Prove that the number of files deleted is 3
theorem files_deleted_is_3 : files_deleted = 3 :=
by
  sorry

end files_deleted_is_3_l19_19224


namespace trains_crossing_time_l19_19732

theorem trains_crossing_time
  (L : ℕ) (t1 t2 : ℕ)
  (h_length : L = 120)
  (h_t1 : t1 = 10)
  (h_t2 : t2 = 15) :
  let V1 := L / t1
  let V2 := L / t2
  let V_relative := V1 + V2
  let D := L + L
  (D / V_relative) = 12 :=
by
  sorry

end trains_crossing_time_l19_19732


namespace additional_spending_required_l19_19977

def cost_of_chicken : ℝ := 1.5 * 6.00
def cost_of_lettuce : ℝ := 3.00
def cost_of_cherry_tomatoes : ℝ := 2.50
def cost_of_sweet_potatoes : ℝ := 4 * 0.75
def cost_of_broccoli : ℝ := 2 * 2.00
def cost_of_brussel_sprouts : ℝ := 2.50
def total_cost : ℝ := cost_of_chicken + cost_of_lettuce + cost_of_cherry_tomatoes + cost_of_sweet_potatoes + cost_of_broccoli + cost_of_brussel_sprouts
def minimum_spending_for_free_delivery : ℝ := 35.00
def additional_amount_needed : ℝ := minimum_spending_for_free_delivery - total_cost

theorem additional_spending_required : additional_amount_needed = 11.00 := by
  sorry

end additional_spending_required_l19_19977


namespace value_of_a_plus_b_l19_19300

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) :
  a + b = -1 ∨ a + b = -3 :=
sorry

end value_of_a_plus_b_l19_19300


namespace min_x_plus_y_l19_19230

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y ≥ 9 := by
  sorry

end min_x_plus_y_l19_19230


namespace georgia_black_buttons_l19_19372

theorem georgia_black_buttons : 
  ∀ (B : ℕ), 
  (4 + B + 3 = 9) → 
  B = 2 :=
by
  introv h
  linarith

end georgia_black_buttons_l19_19372


namespace Robert_older_than_Elizabeth_l19_19587

-- Define the conditions
def Patrick_half_Robert (Patrick Robert : ℕ) : Prop := Patrick = Robert / 2
def Robert_turn_30_in_2_years (Robert : ℕ) : Prop := Robert + 2 = 30
def Elizabeth_4_years_younger_than_Patrick (Elizabeth Patrick : ℕ) : Prop := Elizabeth = Patrick - 4

-- The theorem we need to prove
theorem Robert_older_than_Elizabeth
  (Patrick Robert Elizabeth : ℕ)
  (h1 : Patrick_half_Robert Patrick Robert)
  (h2 : Robert_turn_30_in_2_years Robert)
  (h3 : Elizabeth_4_years_younger_than_Patrick Elizabeth Patrick) :
  Robert - Elizabeth = 18 :=
sorry

end Robert_older_than_Elizabeth_l19_19587


namespace Sara_pears_left_l19_19734

def Sara_has_left (initial_pears : ℕ) (given_to_Dan : ℕ) (given_to_Monica : ℕ) (given_to_Jenny : ℕ) : ℕ :=
  initial_pears - given_to_Dan - given_to_Monica - given_to_Jenny

theorem Sara_pears_left :
  Sara_has_left 35 28 4 1 = 2 :=
by
  sorry

end Sara_pears_left_l19_19734


namespace average_score_l19_19570

variable (T : ℝ) -- Total number of students
variable (M : ℝ) -- Number of male students
variable (F : ℝ) -- Number of female students

variable (avgM : ℝ) -- Average score for male students
variable (avgF : ℝ) -- Average score for female students

-- Conditions
def M_condition : Prop := M = 0.4 * T
def F_condition : Prop := F = 0.6 * T
def avgM_condition : Prop := avgM = 75
def avgF_condition : Prop := avgF = 80

theorem average_score (h1 : M_condition T M) (h2 : F_condition T F) 
    (h3 : avgM_condition avgM) (h4 : avgF_condition avgF) :
    (75 * M + 80 * F) / T = 78 := by
  sorry

end average_score_l19_19570


namespace arithmetic_sequence_n_value_l19_19607

def arithmetic_seq_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_n_value :
  ∀ (a1 d n an : ℕ), a1 = 3 → d = 2 → an = 25 → arithmetic_seq_nth_term a1 d n = an → n = 12 :=
by
  intros a1 d n an ha1 hd han h
  sorry

end arithmetic_sequence_n_value_l19_19607


namespace intersection_complement_eq_l19_19367

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - 5 * x + 4 < 0}

theorem intersection_complement_eq :
  A ∩ {x | x ≤ 1 ∨ x ≥ 4} = {0, 1} := by
  sorry

end intersection_complement_eq_l19_19367


namespace exponent_calculation_l19_19682

theorem exponent_calculation (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : 
  a^(2 * m - 3 * n) = 9 / 8 := 
by
  sorry

end exponent_calculation_l19_19682


namespace children_multiple_of_four_l19_19866

theorem children_multiple_of_four (C : ℕ) 
  (h_event : ∃ (A : ℕ) (T : ℕ), A = 12 ∧ T = 4 ∧ 12 % T = 0 ∧ C % T = 0) : ∃ k : ℕ, C = 4 * k :=
by
  obtain ⟨A, T, hA, hT, hA_div, hC_div⟩ := h_event
  rw [hA, hT] at *
  sorry

end children_multiple_of_four_l19_19866


namespace quarters_count_l19_19210

noncomputable def num_coins := 12
noncomputable def total_value := 166 -- in cents
noncomputable def min_value := 1 + 5 + 10 + 25 + 50 -- minimum value from one of each type
noncomputable def remaining_value := total_value - min_value
noncomputable def remaining_coins := num_coins - 5

theorem quarters_count :
  ∀ (p n d q h : ℕ), 
  p + n + d + q + h = num_coins ∧
  p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧ h ≥ 1 ∧
  (p + 5*n + 10*d + 25*q + 50*h = total_value) → 
  q = 3 := 
by 
  sorry

end quarters_count_l19_19210


namespace ceil_neg_sqrt_frac_l19_19976

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end ceil_neg_sqrt_frac_l19_19976


namespace smallest_value_expression_geq_three_l19_19668

theorem smallest_value_expression_geq_three :
  ∀ (x y : ℝ), 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2 ≥ 3 := 
by
  sorry

end smallest_value_expression_geq_three_l19_19668


namespace negation_of_exists_l19_19516

theorem negation_of_exists (x : ℝ) : ¬(∃ x_0 : ℝ, |x_0| + x_0^2 < 0) ↔ ∀ x : ℝ, |x| + x^2 ≥ 0 :=
by
  sorry

end negation_of_exists_l19_19516


namespace coordinates_of_P_l19_19844

variable (a : ℝ)

def y_coord (a : ℝ) : ℝ :=
  3 * a + 9

def x_coord (a : ℝ) : ℝ :=
  4 - a

theorem coordinates_of_P :
  (∃ a : ℝ, y_coord a = 0) → ∃ a : ℝ, (x_coord a, y_coord a) = (7, 0) :=
by
  -- The proof goes here
  sorry

end coordinates_of_P_l19_19844


namespace ratio_of_length_to_breadth_l19_19793

theorem ratio_of_length_to_breadth (b l k : ℕ) (h1 : b = 15) (h2 : l = k * b) (h3 : l * b = 675) : l / b = 3 :=
by
  sorry

end ratio_of_length_to_breadth_l19_19793


namespace vertex_is_correct_l19_19755

-- Define the equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 10 * y + 4 * x + 9 = 0

-- The vertex of the parabola
def vertex_of_parabola : ℝ × ℝ := (4, -5)

-- The theorem stating that the given vertex satisfies the parabola equation
theorem vertex_is_correct : 
  parabola_equation vertex_of_parabola.1 vertex_of_parabola.2 :=
sorry

end vertex_is_correct_l19_19755


namespace candy_factory_days_l19_19191

noncomputable def candies_per_hour := 50
noncomputable def total_candies := 4000
noncomputable def working_hours_per_day := 10
noncomputable def total_hours_needed := total_candies / candies_per_hour
noncomputable def total_days_needed := total_hours_needed / working_hours_per_day

theorem candy_factory_days :
  total_days_needed = 8 := 
by
  -- (Proof steps will be filled here)
  sorry

end candy_factory_days_l19_19191


namespace vova_last_grades_l19_19723

theorem vova_last_grades (grades : Fin 19 → ℕ) 
  (first_four_2s : ∀ i : Fin 4, grades i = 2)
  (all_combinations_once : ∀ comb : Fin 4 → ℕ, 
    (∃ (start : Fin (19-3)), ∀ j : Fin 4, grades (start + j) = comb j) ∧
    (∀ i j : Fin (19-3), 
      (∀ k : Fin 4, grades (i + k) = grades (j + k)) → i = j)) :
  ∀ i : Fin 4, grades (15 + i) = if i = 0 then 3 else 2 :=
by
  sorry

end vova_last_grades_l19_19723


namespace beijing_olympics_problem_l19_19912

theorem beijing_olympics_problem
  (M T J D: Type)
  (sports: M → Type)
  (swimming gymnastics athletics volleyball: M → Prop)
  (athlete_sits: M → M → Prop)
  (Maria Tania Juan David: M)
  (woman: M → Prop)
  (left right front next_to: M → M → Prop)
  (h1: ∀ x, swimming x → left x Maria)
  (h2: ∀ x, gymnastics x → front x Juan)
  (h3: next_to Tania David)
  (h4: ∀ x, volleyball x → ∃ y, woman y ∧ next_to y x) :
  athletics David := 
sorry

end beijing_olympics_problem_l19_19912


namespace find_range_of_x_l19_19249

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 ^ x else 2 ^ (-x)

theorem find_range_of_x (x : ℝ) : 
  f (1 - 2 * x) < f 3 ↔ (-1 < x ∧ x < 2) := 
sorry

end find_range_of_x_l19_19249


namespace gcd_polynomial_l19_19801

theorem gcd_polynomial (b : ℤ) (h : 570 ∣ b) :
  Int.gcd (5 * b^4 + 2 * b^3 + 5 * b^2 + 9 * b + 95) b = 95 :=
sorry

end gcd_polynomial_l19_19801


namespace train_length_is_1400_l19_19496

theorem train_length_is_1400
  (L : ℝ) 
  (h1 : ∃ speed, speed = L / 100) 
  (h2 : ∃ speed, speed = (L + 700) / 150) :
  L = 1400 :=
by sorry

end train_length_is_1400_l19_19496


namespace find_ages_l19_19636

-- Definitions of the conditions
def cond1 (D S : ℕ) : Prop := D = 3 * S
def cond2 (D S : ℕ) : Prop := D + 5 = 2 * (S + 5)

-- Theorem statement
theorem find_ages (D S : ℕ) 
  (h1 : cond1 D S) 
  (h2 : cond2 D S) : 
  D = 15 ∧ S = 5 :=
by 
  sorry

end find_ages_l19_19636


namespace kelly_initial_games_l19_19038

theorem kelly_initial_games (games_given_away : ℕ) (games_left : ℕ)
  (h1 : games_given_away = 91) (h2 : games_left = 92) : 
  games_given_away + games_left = 183 :=
by {
  sorry
}

end kelly_initial_games_l19_19038


namespace largest_integer_solution_l19_19070

theorem largest_integer_solution (m : ℤ) (h : 2 * m + 7 ≤ 3) : m ≤ -2 :=
sorry

end largest_integer_solution_l19_19070


namespace placemat_length_l19_19529

noncomputable def calculate_placemat_length
    (R : ℝ)
    (num_mats : ℕ)
    (mat_width : ℝ)
    (overlap_ratio : ℝ) : ℝ := 
    let circumference := 2 * Real.pi * R
    let arc_length := circumference / num_mats
    let angle := 2 * Real.pi / num_mats
    let chord_length := 2 * R * Real.sin (angle / 2)
    let effective_mat_length := chord_length / (1 - overlap_ratio * 2)
    effective_mat_length

theorem placemat_length (R : ℝ) (num_mats : ℕ) (mat_width : ℝ) (overlap_ratio : ℝ): 
    R = 5 ∧ num_mats = 8 ∧ mat_width = 2 ∧ overlap_ratio = (1 / 4)
    → calculate_placemat_length R num_mats mat_width overlap_ratio = 7.654 :=
by
  sorry

end placemat_length_l19_19529


namespace neither_odd_nor_even_and_min_value_at_one_l19_19803

def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem neither_odd_nor_even_and_min_value_at_one :
  (∀ x, f (-x) ≠ f x ∧ f (-x) ≠ - f x) ∧ ∃ x, x = 1 ∧ ∀ y, f y ≥ f x :=
by
  sorry

end neither_odd_nor_even_and_min_value_at_one_l19_19803


namespace number_of_friends_l19_19595

def has14_pokemon_cards (x : String) : Prop :=
  x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith"

theorem number_of_friends :
  ∃ n, n = 4 ∧
        ∀ x, has14_pokemon_cards x ↔ x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith" :=
by
  sorry

end number_of_friends_l19_19595


namespace problem_statement_l19_19491

def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) : ℝ := 3 * x - 2

theorem problem_statement : f (g (f (g 2))) = 7189058 := by
  sorry

end problem_statement_l19_19491


namespace power_of_two_with_nines_l19_19485

theorem power_of_two_with_nines (k : ℕ) (h : k > 1) :
  ∃ (n : ℕ), (2^n % 10^k) / 10^((10 * 5^k + k + 2 - k) / 2) = 9 :=
sorry

end power_of_two_with_nines_l19_19485


namespace inequality_proof_l19_19000

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≥ b) (h5 : b ≥ c) :
  a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ∧
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ (a^3 / (b * c)) + (b^3 / (c * a)) + (c^3 / (a * b)) :=
by
  sorry

end inequality_proof_l19_19000


namespace inequality_solution_l19_19091

theorem inequality_solution :
  {x : ℝ | (x^2 + 5 * x) / ((x - 3) ^ 2) ≥ 0} = {x | x < -5} ∪ {x | 0 ≤ x ∧ x < 3} ∪ {x | x > 3} :=
by
  sorry

end inequality_solution_l19_19091


namespace volume_range_l19_19439

theorem volume_range (a b c : ℝ) (h1 : a + b + c = 9)
  (h2 : a * b + b * c + a * c = 24) : 16 ≤ a * b * c ∧ a * b * c ≤ 20 :=
by {
  -- Proof would go here
  sorry
}

end volume_range_l19_19439


namespace students_did_not_eat_2_l19_19348

-- Define the given conditions
def total_students : ℕ := 20
def total_crackers_eaten : ℕ := 180
def crackers_per_pack : ℕ := 10

-- Calculate the number of packs eaten
def packs_eaten : ℕ := total_crackers_eaten / crackers_per_pack

-- Calculate the number of students who did not eat their animal crackers
def students_who_did_not_eat : ℕ := total_students - packs_eaten

-- Prove that the number of students who did not eat their animal crackers is 2
theorem students_did_not_eat_2 :
  students_who_did_not_eat = 2 :=
  by
    sorry

end students_did_not_eat_2_l19_19348


namespace total_distance_walked_l19_19629

variables
  (distance1 : ℝ := 1.2)
  (distance2 : ℝ := 0.8)
  (distance3 : ℝ := 1.5)
  (distance4 : ℝ := 0.6)
  (distance5 : ℝ := 2)

theorem total_distance_walked :
  distance1 + distance2 + distance3 + distance4 + distance5 = 6.1 :=
sorry

end total_distance_walked_l19_19629


namespace last_digit_2_to_2010_l19_19619

theorem last_digit_2_to_2010 : (2 ^ 2010) % 10 = 4 := 
by
  -- proofs and lemmas go here
  sorry

end last_digit_2_to_2010_l19_19619


namespace function_increment_l19_19110

theorem function_increment (x₁ x₂ : ℝ) (f : ℝ → ℝ) (h₁ : x₁ = 2) 
                           (h₂ : x₂ = 2.5) (h₃ : ∀ x, f x = x ^ 2) :
  f x₂ - f x₁ = 2.25 :=
by
  sorry

end function_increment_l19_19110


namespace analytic_expression_on_1_2_l19_19974

noncomputable def f : ℝ → ℝ :=
  sorry

theorem analytic_expression_on_1_2 (x : ℝ) (h1 : 1 < x) (h2 : x < 2) :
  f x = Real.logb (1 / 2) (x - 1) :=
sorry

end analytic_expression_on_1_2_l19_19974


namespace integer_values_of_f_l19_19938

noncomputable def f (x : ℝ) : ℝ := (1 + x)^(1/3) + (3 - x)^(1/3)

theorem integer_values_of_f : 
  {x : ℝ | ∃ k : ℤ, f x = k} = {1 + Real.sqrt 5, 1 - Real.sqrt 5, 1 + (10/9) * Real.sqrt 3, 1 - (10/9) * Real.sqrt 3} :=
by
  sorry

end integer_values_of_f_l19_19938


namespace product_eq_5832_l19_19768

theorem product_eq_5832 (P Q R S : ℕ) 
(h1 : P + Q + R + S = 48)
(h2 : P + 3 = Q - 3)
(h3 : Q - 3 = R * 3)
(h4 : R * 3 = S / 3) :
P * Q * R * S = 5832 := sorry

end product_eq_5832_l19_19768


namespace x0_range_l19_19874

noncomputable def f (x : ℝ) := (1 / 2) ^ x - Real.log x

theorem x0_range (x0 : ℝ) (h : f x0 > 1 / 2) : 0 < x0 ∧ x0 < 1 :=
by
  sorry

end x0_range_l19_19874


namespace find_number_l19_19282

theorem find_number (n : ℝ) (x : ℕ) (h1 : x = 4) (h2 : n^(2*x) = 3^(12-x)) : n = 3 := by
  sorry

end find_number_l19_19282


namespace speed_ratio_l19_19558

theorem speed_ratio (va vb : ℝ) (L : ℝ) (h : va = vb * k) (head_start : vb * (L - 0.05 * L) = vb * L) : 
    (va / vb) = (1 / 0.95) :=
by
  sorry

end speed_ratio_l19_19558


namespace q_minus_p_897_l19_19652

def smallest_three_digit_integer_congruent_7_mod_13 := ∃ p : ℕ, p ≥ 100 ∧ p < 1000 ∧ p % 13 = 7
def smallest_four_digit_integer_congruent_7_mod_13 := ∃ q : ℕ, q ≥ 1000 ∧ q < 10000 ∧ q % 13 = 7

theorem q_minus_p_897 : 
  (∃ p : ℕ, p ≥ 100 ∧ p < 1000 ∧ p % 13 = 7) → 
  (∃ q : ℕ, q ≥ 1000 ∧ q < 10000 ∧ q % 13 = 7) → 
  ∀ p q : ℕ, 
    (p = 8*13+7) → 
    (q = 77*13+7) → 
    q - p = 897 :=
by
  intros h1 h2 p q hp hq
  sorry

end q_minus_p_897_l19_19652


namespace midpoint_of_hyperbola_l19_19184

theorem midpoint_of_hyperbola (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) → 
  M = (-1, -4) :=
by
  sorry

end midpoint_of_hyperbola_l19_19184


namespace dog_food_duration_l19_19511

-- Definitions for the given conditions
def number_of_dogs : ℕ := 4
def meals_per_day : ℕ := 2
def grams_per_meal : ℕ := 250
def sacks_of_food : ℕ := 2
def kilograms_per_sack : ℝ := 50
def grams_per_kilogram : ℝ := 1000

-- Lean statement to prove the correct answer
theorem dog_food_duration : 
  ((number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) * sacks_of_food * kilograms_per_sack) / 
  (number_of_dogs * meals_per_day * grams_per_meal / grams_per_kilogram) = 50 :=
by 
  simp only [number_of_dogs, meals_per_day, grams_per_meal, sacks_of_food, kilograms_per_sack, grams_per_kilogram]
  norm_num
  sorry

end dog_food_duration_l19_19511


namespace negation_example_l19_19925

theorem negation_example :
  (¬ (∀ x : ℝ, abs (x - 2) + abs (x - 4) > 3)) ↔ (∃ x : ℝ, abs (x - 2) + abs (x - 4) ≤ 3) :=
by
  sorry

end negation_example_l19_19925


namespace line_properties_l19_19726

theorem line_properties (m x_intercept : ℝ) (y_intercept point_on_line : ℝ × ℝ) :
  m = -4 → x_intercept = -3 → y_intercept = (0, -12) → point_on_line = (2, -20) → 
    (∀ x y, y = -4 * x - 12 → (y_intercept = (0, y) ∧ point_on_line = (x, y))) := 
by
  sorry

end line_properties_l19_19726


namespace regular_polygon_sides_l19_19404

theorem regular_polygon_sides (exterior_angle : ℕ) (h : exterior_angle = 30) : (360 / exterior_angle) = 12 := by
  sorry

end regular_polygon_sides_l19_19404


namespace problem_solution_l19_19543

theorem problem_solution : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 10) / (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) = 360 :=
by
  sorry

end problem_solution_l19_19543


namespace polyhedron_faces_l19_19680

theorem polyhedron_faces (V E : ℕ) (F T P : ℕ) (h1 : F = 40) (h2 : V - E + F = 2) (h3 : T + P = 40) 
  (h4 : E = (3 * T + 4 * P) / 2) (h5 : V = (160 - T) / 2 - 38) (h6 : P = 3) (h7 : T = 1) :
  100 * P + 10 * T + V = 351 :=
by
  sorry

end polyhedron_faces_l19_19680


namespace work_done_l19_19398

theorem work_done (m : ℕ) : 18 * 30 = m * 36 → m = 15 :=
by
  intro h  -- assume the equality condition
  have h1 : m = 15 := by
    -- We would solve for m here similarly to the solution given to derive 15
    sorry
  exact h1

end work_done_l19_19398


namespace carpet_length_l19_19890

-- Define the conditions as hypotheses
def width_of_carpet : ℝ := 4
def area_of_living_room : ℝ := 60

-- Formalize the corresponding proof problem
theorem carpet_length (h : 60 = width_of_carpet * length) : length = 15 :=
sorry

end carpet_length_l19_19890


namespace least_product_of_distinct_primes_greater_than_50_l19_19952

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  p ≠ q ∧ is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 : 
  ∃ p q, distinct_primes_greater_than_50 p q ∧ p * q = 3127 := 
sorry

end least_product_of_distinct_primes_greater_than_50_l19_19952


namespace solve_arithmetic_seq_l19_19787

theorem solve_arithmetic_seq (x : ℝ) (h : x > 0) (hx : x^2 = (4 + 16) / 2) : x = Real.sqrt 10 :=
sorry

end solve_arithmetic_seq_l19_19787


namespace angle_of_inclination_l19_19458

/--
Given the direction vector of line l as (-sqrt(3), 3),
prove that the angle of inclination α of line l is 120 degrees.
-/
theorem angle_of_inclination (α : ℝ) :
  let direction_vector : Real × Real := (-Real.sqrt 3, 3)
  let slope := direction_vector.2 / direction_vector.1
  slope = -Real.sqrt 3 → α = 120 :=
by
  sorry

end angle_of_inclination_l19_19458


namespace initial_average_is_100_l19_19125

-- Definitions based on the conditions from step a)
def students : ℕ := 10
def wrong_mark : ℕ := 90
def correct_mark : ℕ := 10
def correct_average : ℝ := 92

-- Initial average marks before correcting the error
def initial_average_marks (A : ℝ) : Prop :=
  10 * A = (students * correct_average) + (wrong_mark - correct_mark)

theorem initial_average_is_100 :
  ∃ A : ℝ, initial_average_marks A ∧ A = 100 :=
by {
  -- We are defining the placeholder for the actual proof.
  sorry
}

end initial_average_is_100_l19_19125


namespace Jessica_cut_roses_l19_19465

variable (initial_roses final_roses added_roses : Nat)

theorem Jessica_cut_roses
  (h_initial : initial_roses = 10)
  (h_final : final_roses = 18)
  (h_added : final_roses = initial_roses + added_roses) :
  added_roses = 8 := by
  sorry

end Jessica_cut_roses_l19_19465


namespace simplify_expression_l19_19209

section
variable (a b : ℚ) (h_a : a = -1) (h_b : b = 1/4)

theorem simplify_expression : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry
end

end simplify_expression_l19_19209


namespace closest_fraction_to_medals_won_l19_19074

theorem closest_fraction_to_medals_won :
  let gamma_fraction := (13:ℚ) / 80
  let fraction_1_4 := (1:ℚ) / 4
  let fraction_1_5 := (1:ℚ) / 5
  let fraction_1_6 := (1:ℚ) / 6
  let fraction_1_7 := (1:ℚ) / 7
  let fraction_1_8 := (1:ℚ) / 8
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_4) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_5) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_7) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_8) := by
  sorry

end closest_fraction_to_medals_won_l19_19074


namespace correctFractions_equivalence_l19_19871

def correctFractions: List (ℕ × ℕ) := [(26, 65), (16, 64), (19, 95), (49, 98)]

def isValidCancellation (num den: ℕ): Prop :=
  ∃ n₁ n₂ n₃ d₁ d₂ d₃: ℕ, 
    num = 10 * n₁ + n₂ ∧
    den = 10 * d₁ + d₂ ∧
    ((n₁ = d₁ ∧ n₂ = d₂) ∨ (n₁ = d₃ ∧ n₃ = d₂)) ∧
    n₁ ≠ 0 ∧ n₂ ≠ 0 ∧ d₁ ≠ 0 ∧ d₂ ≠ 0

theorem correctFractions_equivalence : 
  ∀ (frac : ℕ × ℕ), frac ∈ correctFractions → 
    ∃ a b: ℕ, correctFractions = [(a, b)] ∧ 
      isValidCancellation a b := sorry

end correctFractions_equivalence_l19_19871


namespace difference_in_ages_is_54_l19_19512

theorem difference_in_ages_is_54 (c d : ℕ) (h1 : 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100) 
    (h2 : 10 * c + d - (10 * d + c) = 9 * (c - d)) 
    (h3 : 10 * c + d + 10 = 3 * (10 * d + c + 10)) : 
    10 * c + d - (10 * d + c) = 54 :=
by
sorry

end difference_in_ages_is_54_l19_19512


namespace greatest_k_divides_n_l19_19484

theorem greatest_k_divides_n (n : ℕ) (h_pos : 0 < n) (h_divisors_n : Nat.totient n = 72) (h_divisors_5n : Nat.totient (5 * n) = 90) : ∃ k : ℕ, ∀ m : ℕ, (5^k ∣ n) → (5^(k+1) ∣ n) → k = 3 :=
by
  sorry

end greatest_k_divides_n_l19_19484


namespace num_five_digit_numbers_is_correct_l19_19536

-- Define the set of digits and their repetition as given in the conditions
def digits : Multiset ℕ := {1, 3, 3, 5, 8}

-- Calculate the permutation with repetitions
noncomputable def num_five_digit_numbers : ℕ := (digits.card.factorial) / 
  (Multiset.count 1 digits).factorial / 
  (Multiset.count 3 digits).factorial / 
  (Multiset.count 5 digits).factorial / 
  (Multiset.count 8 digits).factorial

-- Theorem stating the final result
theorem num_five_digit_numbers_is_correct : num_five_digit_numbers = 60 :=
by
  -- Proof is omitted
  sorry

end num_five_digit_numbers_is_correct_l19_19536


namespace gcd_problem_l19_19040

theorem gcd_problem : 
  let a := 690
  let b := 875
  let r1 := 10
  let r2 := 25
  let n1 := a - r1
  let n2 := b - r2
  gcd n1 n2 = 170 :=
by
  sorry

end gcd_problem_l19_19040


namespace swans_after_10_years_l19_19114

-- Defining the initial conditions
def initial_swans : ℕ := 15

-- Condition that the number of swans doubles every 2 years
def double_every_two_years (n t : ℕ) : ℕ := n * (2 ^ (t / 2))

-- Prove that after 10 years, the number of swans will be 480
theorem swans_after_10_years : double_every_two_years initial_swans 10 = 480 :=
by
  sorry

end swans_after_10_years_l19_19114


namespace pentagon_right_angles_l19_19360

theorem pentagon_right_angles (angles : Finset ℕ) :
  angles = {0, 1, 2, 3} ↔ ∀ (k : ℕ), k ∈ angles ↔ ∃ (a b c d e : ℕ), 
  a + b + c + d + e = 540 ∧ (a = 90 ∨ b = 90 ∨ c = 90 ∨ d = 90 ∨ e = 90) 
  ∧ Finset.card (Finset.filter (λ x => x = 90) {a, b, c, d, e}) = k := 
sorry

end pentagon_right_angles_l19_19360


namespace number_of_boys_l19_19189

theorem number_of_boys (x g : ℕ) (h1 : x + g = 100) (h2 : g = x) : x = 50 := by
  sorry

end number_of_boys_l19_19189


namespace regular_decagon_triangle_probability_l19_19550

theorem regular_decagon_triangle_probability :
  let total_triangles := Nat.choose 10 3
  let favorable_triangles := 10
  let probability := favorable_triangles / total_triangles
  probability = (1 : ℚ) / 12 :=
by
  sorry

end regular_decagon_triangle_probability_l19_19550


namespace at_least_5_limit_ups_needed_l19_19503

-- Let's denote the necessary conditions in Lean
variable (a : ℝ) -- the buying price of stock A

-- Initial price after 4 consecutive limit downs
def price_after_limit_downs (a : ℝ) : ℝ := a * (1 - 0.1) ^ 4

-- Condition of no loss after certain limit ups
def no_loss_after_limit_ups (a : ℝ) (x : ℕ) : Prop := 
  price_after_limit_downs a * (1 + 0.1)^x ≥ a
  
theorem at_least_5_limit_ups_needed (a : ℝ) : ∃ x, no_loss_after_limit_ups a x ∧ x ≥ 5 :=
by
  -- We are required to find such x and prove the condition, which has been shown in the mathematical solution
  sorry

end at_least_5_limit_ups_needed_l19_19503


namespace amit_work_days_l19_19313

variable (x : ℕ)

theorem amit_work_days
  (ananthu_rate : ℚ := 1/30) -- Ananthu's work rate is 1/30
  (amit_days : ℕ := 3) -- Amit worked for 3 days
  (ananthu_days : ℕ := 24) -- Ananthu worked for remaining 24 days
  (total_days : ℕ := 27) -- Total work completed in 27 days
  (amit_work: ℚ := amit_days * 1/x) -- Amit's work rate
  (ananthu_work: ℚ := ananthu_days * ananthu_rate) -- Ananthu's work rate
  (total_work : ℚ := 1) -- Total work completed  
  : 3 * (1/x) + 24 * (1/30) = 1 ↔ x = 15 := 
by
  sorry

end amit_work_days_l19_19313


namespace correct_option_C_l19_19447

def number_of_stamps : String := "the number of the stamps"
def number_of_people : String := "a number of people"

def is_singular (subject : String) : Prop := subject = number_of_stamps
def is_plural (subject : String) : Prop := subject = number_of_people

def correct_sentence (verb1 verb2 : String) : Prop :=
  verb1 = "is" ∧ verb2 = "want"

theorem correct_option_C : correct_sentence "is" "want" :=
by
  show correct_sentence "is" "want"
  -- Proof is omitted
  sorry

end correct_option_C_l19_19447


namespace prob_red_or_blue_l19_19544

-- Total marbles and given probabilities
def total_marbles : ℕ := 120
def prob_white : ℚ := 1 / 4
def prob_green : ℚ := 1 / 3

-- Problem statement
theorem prob_red_or_blue : (1 - (prob_white + prob_green)) = 5 / 12 :=
by
  sorry

end prob_red_or_blue_l19_19544


namespace diophantine_solution_unique_l19_19187

theorem diophantine_solution_unique (k x y : ℕ) (hk : k > 0) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 = k * x * y - 1 ↔ k = 3 :=
by sorry

end diophantine_solution_unique_l19_19187


namespace trips_to_collect_all_trays_l19_19654

-- Definition of conditions
def trays_at_once : ℕ := 7
def trays_one_table : ℕ := 23
def trays_other_table : ℕ := 5

-- Theorem statement
theorem trips_to_collect_all_trays : 
  (trays_one_table / trays_at_once) + (if trays_one_table % trays_at_once = 0 then 0 else 1) + 
  (trays_other_table / trays_at_once) + (if trays_other_table % trays_at_once = 0 then 0 else 1) = 5 := 
by
  sorry

end trips_to_collect_all_trays_l19_19654


namespace fraction_of_liars_l19_19337

theorem fraction_of_liars (n : ℕ) (villagers : Fin n → Prop) (right_neighbor : ∀ i, villagers i ↔ ∀ j : Fin n, j = (i + 1) % n → villagers j) :
  ∃ (x : ℚ), x = 1 / 2 :=
by 
  sorry

end fraction_of_liars_l19_19337


namespace total_weight_of_fruits_l19_19400

/-- Define the given conditions in Lean -/
def weight_of_orange_bags (n : ℕ) : ℝ :=
  if n = 12 then 24 else 0

def weight_of_apple_bags (n : ℕ) : ℝ :=
  if n = 8 then 30 else 0

/-- Prove that the total weight of 5 bags of oranges and 4 bags of apples is 25 pounds given the conditions -/
theorem total_weight_of_fruits :
  weight_of_orange_bags 12 / 12 * 5 + weight_of_apple_bags 8 / 8 * 4 = 25 :=
by sorry

end total_weight_of_fruits_l19_19400


namespace students_with_all_three_pets_l19_19303

variables (TotalStudents HaveDogs HaveCats HaveOtherPets NoPets x y z w : ℕ)

theorem students_with_all_three_pets :
  TotalStudents = 40 →
  HaveDogs = 20 →
  HaveCats = 16 →
  HaveOtherPets = 8 →
  NoPets = 7 →
  x = 12 →
  y = 3 →
  z = 11 →
  TotalStudents - NoPets = 33 →
  x + y + w = HaveDogs →
  z + w = HaveCats →
  y + w = HaveOtherPets →
  x + y + z + w = 33 →
  w = 5 :=
by
  intros h1 h2 h3 h4 h5 hx hy hz h6 h7 h8 h9
  sorry

end students_with_all_three_pets_l19_19303


namespace candy_count_l19_19043

def initial_candy : ℕ := 47
def eaten_candy : ℕ := 25
def sister_candy : ℕ := 40
def final_candy : ℕ := 62

theorem candy_count : initial_candy - eaten_candy + sister_candy = final_candy := 
by
  sorry

end candy_count_l19_19043


namespace input_value_of_x_l19_19035

theorem input_value_of_x (x y : ℤ) (h₁ : (x < 0 → y = (x + 1) * (x + 1)) ∧ (¬(x < 0) → y = (x - 1) * (x - 1)))
  (h₂ : y = 16) : x = 5 ∨ x = -5 :=
sorry

end input_value_of_x_l19_19035


namespace isosceles_triangle_problem_l19_19037

theorem isosceles_triangle_problem
  (BT CT : Real) (BC : Real) (BZ CZ TZ : Real) :
  BT = 20 →
  CT = 20 →
  BC = 24 →
  TZ^2 + 2 * BZ * CZ = 478 →
  BZ = CZ →
  BZ * CZ = 144 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end isosceles_triangle_problem_l19_19037


namespace algorithm_output_l19_19374

noncomputable def algorithm (x : ℝ) : ℝ :=
if x < 0 then x + 1 else -x^2

theorem algorithm_output :
  algorithm (-2) = -1 ∧ algorithm 3 = -9 :=
by
  -- proof omitted using sorry
  sorry

end algorithm_output_l19_19374


namespace problem_solution_l19_19630

theorem problem_solution
  (k : ℝ)
  (y : ℝ → ℝ)
  (quadratic_fn : ∀ x, y x = (k + 2) * x^(k^2 + k - 4))
  (increase_for_neg_x : ∀ x : ℝ, x < 0 → y (x + 1) > y x) :
  k = -3 ∧ (∀ m n : ℝ, -2 ≤ m ∧ m ≤ 1 → y m = n → -4 ≤ n ∧ n ≤ 0) := 
sorry

end problem_solution_l19_19630


namespace average_weight_of_all_girls_l19_19413

theorem average_weight_of_all_girls (avg1 : ℝ) (n1 : ℕ) (avg2 : ℝ) (n2 : ℕ) :
  avg1 = 50.25 → n1 = 16 → avg2 = 45.15 → n2 = 8 → 
  ((n1 * avg1 + n2 * avg2) / (n1 + n2)) = 48.55 := 
by
  intros h1 h2 h3 h4
  sorry

end average_weight_of_all_girls_l19_19413


namespace compute_fraction_at_six_l19_19757

theorem compute_fraction_at_six (x : ℕ) (h : x = 6) : (x^6 - 16 * x^3 + 64) / (x^3 - 8) = 208 := by
  sorry

end compute_fraction_at_six_l19_19757


namespace domain_of_function_l19_19079

theorem domain_of_function :
  (∀ x : ℝ, (2 * Real.sin x - 1 > 0) ∧ (1 - 2 * Real.cos x ≥ 0) ↔
    ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 3 ≤ x ∧ x < 2 * k * Real.pi + 5 * Real.pi / 6) :=
sorry

end domain_of_function_l19_19079


namespace evaluate_expression_l19_19706

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem evaluate_expression :
  (4 / log_base 5 (2500^3) + 2 / log_base 2 (2500^3) = 1 / 3) := by
  sorry

end evaluate_expression_l19_19706


namespace like_term_l19_19471

theorem like_term (a : ℝ) : ∃ (a : ℝ), a * x ^ 5 * y ^ 3 = a * x ^ 5 * y ^ 3 :=
by sorry

end like_term_l19_19471


namespace cuboid_height_l19_19175

-- Given conditions
def volume_cuboid : ℝ := 1380 -- cubic meters
def base_area_cuboid : ℝ := 115 -- square meters

-- Prove that the height of the cuboid is 12 meters
theorem cuboid_height : volume_cuboid / base_area_cuboid = 12 := by
  sorry

end cuboid_height_l19_19175


namespace tan_pi_add_theta_l19_19822

theorem tan_pi_add_theta (θ : ℝ) (h : Real.tan (Real.pi + θ) = 2) : 
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + 2 * Real.cos θ) = 3 / 4 :=
by
  sorry

end tan_pi_add_theta_l19_19822


namespace concentrate_amount_l19_19509

def parts_concentrate : ℤ := 1
def parts_water : ℤ := 5
def part_ratio : ℤ := parts_concentrate + parts_water -- Total parts
def servings : ℤ := 375
def volume_per_serving : ℤ := 150
def total_volume : ℤ := servings * volume_per_serving -- Total volume of orange juice
def volume_per_part : ℤ := total_volume / part_ratio -- Volume per part of mixture

theorem concentrate_amount :
  volume_per_part = 9375 :=
by
  sorry

end concentrate_amount_l19_19509


namespace mass_percentage_of_S_in_Al2S3_l19_19046

theorem mass_percentage_of_S_in_Al2S3 :
  let molar_mass_Al : ℝ := 26.98
  let molar_mass_S : ℝ := 32.06
  let formula_of_Al2S3: (ℕ × ℕ) := (2, 3)
  let molar_mass_Al2S3 : ℝ := (2 * molar_mass_Al) + (3 * molar_mass_S)
  let total_mass_S_in_Al2S3 : ℝ := 3 * molar_mass_S
  (total_mass_S_in_Al2S3 / molar_mass_Al2S3) * 100 = 64.07 :=
by
  sorry

end mass_percentage_of_S_in_Al2S3_l19_19046


namespace inequlity_for_k_one_smallest_k_l19_19735

noncomputable def triangle_sides (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

theorem inequlity_for_k_one (a b c : ℝ) (h : triangle_sides a b c) :
  a^3 + b^3 + c^3 < (a + b + c) * (a * b + b * c + c * a) :=
sorry

theorem smallest_k (a b c k : ℝ) (h : triangle_sides a b c) (hk : k = 1) :
  a^3 + b^3 + c^3 < k * (a + b + c) * (a * b + b * c + c * a) :=
sorry

end inequlity_for_k_one_smallest_k_l19_19735


namespace Bruce_bought_8_kg_of_grapes_l19_19821

-- Defining the conditions
def rate_grapes := 70
def rate_mangoes := 55
def weight_mangoes := 11
def total_paid := 1165

-- Result to be proven
def cost_mangoes := rate_mangoes * weight_mangoes
def total_cost_grapes (G : ℕ) := rate_grapes * G
def total_cost (G : ℕ) := (total_cost_grapes G) + cost_mangoes

theorem Bruce_bought_8_kg_of_grapes (G : ℕ) (h : total_cost G = total_paid) : G = 8 :=
by
  sorry  -- Proof omitted

end Bruce_bought_8_kg_of_grapes_l19_19821


namespace quadratic_solution_exists_l19_19250

theorem quadratic_solution_exists (a b : ℝ) : ∃ (x : ℝ), (a^2 - b^2) * x^2 + 2 * (a^3 - b^3) * x + (a^4 - b^4) = 0 :=
by
  sorry

end quadratic_solution_exists_l19_19250


namespace age_of_15th_student_l19_19058

theorem age_of_15th_student 
  (total_age_15_students : ℕ)
  (total_age_3_students : ℕ)
  (total_age_11_students : ℕ)
  (h1 : total_age_15_students = 225)
  (h2 : total_age_3_students = 42)
  (h3 : total_age_11_students = 176) :
  total_age_15_students - (total_age_3_students + total_age_11_students) = 7 :=
by
  sorry

end age_of_15th_student_l19_19058


namespace find_f_at_3_l19_19532

noncomputable def f (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * (x^32 + 1) - 1) / (x^(2^6 - 1) - 1)

theorem find_f_at_3 : f 3 = 3 :=
by
  sorry

end find_f_at_3_l19_19532


namespace DansAgeCalculation_l19_19475

theorem DansAgeCalculation (D x : ℕ) (h1 : D = 8) (h2 : D + 20 = 7 * (D - x)) : x = 4 :=
by
  sorry

end DansAgeCalculation_l19_19475


namespace missing_angle_measure_l19_19731

theorem missing_angle_measure (n : ℕ) (h : 180 * (n - 2) = 3240 + 2 * (180 * (n - 2)) / n) : 
  (180 * (n - 2)) / n = 166 := 
by 
  sorry

end missing_angle_measure_l19_19731


namespace certain_value_of_101n_squared_l19_19135

theorem certain_value_of_101n_squared 
  (n : ℤ) 
  (h : ∀ (n : ℤ), 101 * n^2 ≤ 4979 → n ≤ 7) : 
  4979 = 101 * 7^2 :=
by {
  /- proof goes here -/
  sorry
}

end certain_value_of_101n_squared_l19_19135


namespace value_of_first_equation_l19_19174

theorem value_of_first_equation (x y : ℚ) 
  (h1 : 5 * x + 6 * y = 7) 
  (h2 : 3 * x + 5 * y = 6) : 
  x + 4 * y = 5 :=
sorry

end value_of_first_equation_l19_19174


namespace unique_painted_cube_l19_19495

/-- Determine the number of distinct ways to paint a cube where:
  - One side is yellow,
  - Two sides are purple,
  - Three sides are orange.
  Taking into account that two cubes are considered identical if they can be rotated to match. -/
theorem unique_painted_cube :
  ∃ unique n : ℕ, n = 1 ∧
    (∃ (c : Fin 6 → Fin 3), 
      (∃ (i : Fin 6), c i = 0) ∧ 
      (∃ (j k : Fin 6), j ≠ k ∧ c j = 1 ∧ c k = 1) ∧ 
      (∃ (m p q : Fin 6), m ≠ p ∧ m ≠ q ∧ p ≠ q ∧ c m = 2 ∧ c p = 2 ∧ c q = 2)
    ) :=
sorry

end unique_painted_cube_l19_19495


namespace number_of_sheets_in_stack_l19_19499

theorem number_of_sheets_in_stack (n : ℕ) (h1 : 2 * n + 2 = 74) : n / 4 = 9 := 
by
  sorry

end number_of_sheets_in_stack_l19_19499


namespace focus_of_parabola_l19_19914

theorem focus_of_parabola (x y : ℝ) (h : y^2 + 4 * x = 0) : (x, y) = (-1, 0) := sorry

end focus_of_parabola_l19_19914


namespace min_value_y_l19_19836

theorem min_value_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 4 / b = 2) : 4 * a + b ≥ 8 :=
sorry

end min_value_y_l19_19836


namespace james_hours_worked_l19_19025

variable (x : ℝ) (y : ℝ)

theorem james_hours_worked (h1: 18 * x + 16 * (1.5 * x) = 40 * x + (y - 40) * (2 * x)) : y = 41 :=
by
  sorry

end james_hours_worked_l19_19025


namespace jelly_bean_count_l19_19624

variable (b c : ℕ)
variable (h1 : b = 3 * c)
variable (h2 : b - 5 = 5 * (c - 15))

theorem jelly_bean_count : b = 105 := by
  sorry

end jelly_bean_count_l19_19624


namespace octahedron_common_sum_is_39_l19_19736

-- Define the vertices of the regular octahedron with numbers from 1 to 12
def vertices : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the property that the sum of four numbers at the vertices of each triangle face is the same
def common_sum (faces : List (List ℕ)) (k : ℕ) : Prop :=
  ∀ face ∈ faces, face.sum = k

-- Define the faces of the regular octahedron
def faces : List (List ℕ) := [
  [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 5, 9, 6],
  [2, 6, 10, 7], [3, 7, 11, 8], [4, 8, 12, 5], [1, 9, 2, 10]
]

-- Prove that the common sum is 39
theorem octahedron_common_sum_is_39 : common_sum faces 39 :=
  sorry

end octahedron_common_sum_is_39_l19_19736


namespace remainder_of_difference_divided_by_prime_l19_19364

def largest_three_digit_number : ℕ := 999
def smallest_five_digit_number : ℕ := 10000
def smallest_prime_greater_than_1000 : ℕ := 1009

theorem remainder_of_difference_divided_by_prime :
  (smallest_five_digit_number - largest_three_digit_number) % smallest_prime_greater_than_1000 = 945 :=
by
  -- The proof will be filled in here
  sorry

end remainder_of_difference_divided_by_prime_l19_19364


namespace smallest_m_div_18_l19_19818

noncomputable def smallest_multiple_18 : ℕ :=
  900

theorem smallest_m_div_18 : (∃ m: ℕ, (m % 18 = 0) ∧ (∀ d ∈ m.digits 10, d = 9 ∨ d = 0) ∧ ∀ k: ℕ, k % 18 = 0 → (∀ d ∈ k.digits 10, d = 9 ∨ d = 0) → m ≤ k) → 900 / 18 = 50 :=
by
  intro h
  sorry

end smallest_m_div_18_l19_19818


namespace Abby_sits_in_seat_3_l19_19354

theorem Abby_sits_in_seat_3:
  ∃ (positions : Fin 5 → String),
  (positions 3 = "Abby") ∧
  (positions 4 = "Bret") ∧
  ¬ ((positions 3 = "Dana") ∨ (positions 5 = "Dana")) ∧
  ¬ ((positions 2 = "Erin") ∧ (positions 3 = "Carl") ∨
    (positions 3 = "Erin") ∧ (positions 5 = "Carl")) :=
  sorry

end Abby_sits_in_seat_3_l19_19354


namespace manuscript_fee_l19_19122

noncomputable def tax (x : ℝ) : ℝ :=
  if x ≤ 800 then 0
  else if x <= 4000 then 0.14 * (x - 800)
  else 0.11 * x

theorem manuscript_fee (x : ℝ) (h₁ : tax x = 420)
  (h₂ : 800 < x ∧ x ≤ 4000 ∨ x > 4000) :
  x = 3800 :=
sorry

end manuscript_fee_l19_19122


namespace bob_final_amount_l19_19302

noncomputable def final_amount (start: ℝ) : ℝ :=
  let day1 := start - (3/5) * start
  let day2 := day1 - (7/12) * day1
  let day3 := day2 - (2/3) * day2
  let day4 := day3 - (1/6) * day3
  let day5 := day4 - (5/8) * day4
  let day6 := day5 - (3/5) * day5
  day6

theorem bob_final_amount : final_amount 500 = 3.47 := by
  sorry

end bob_final_amount_l19_19302


namespace minimum_dot_product_l19_19487

-- Define point coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define points A, B, C, D according to the given problem statement
def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨1, 2⟩
def D : Point := ⟨0, 2⟩

-- Define the condition for points E and F on the sides BC and CD respectively.
def isOnBC (E : Point) : Prop := E.x = 1 ∧ 0 ≤ E.y ∧ E.y ≤ 2
def isOnCD (F : Point) : Prop := F.y = 2 ∧ 0 ≤ F.x ∧ F.x ≤ 1

-- Define the distance constraint for |EF| = 1
def distEF (E F : Point) : Prop :=
  (F.x - E.x)^2 + (F.y - E.y)^2 = 1

-- Define the dot product between vectors AE and AF
def dotProductAEAF (E F : Point) : ℝ :=
  2 * E.y + F.x

-- Main theorem to prove the minimum dot product value
theorem minimum_dot_product (E F : Point) (hE : isOnBC E) (hF : isOnCD F) (hDistEF : distEF E F) :
  dotProductAEAF E F = 5 - Real.sqrt 5 :=
  sorry

end minimum_dot_product_l19_19487


namespace number_of_distinct_stackings_l19_19908

-- Defining the conditions
def cubes : ℕ := 8
def edge_length : ℕ := 1
def valid_stackings (n : ℕ) : Prop := 
  n = 8 -- Stating that we are working with 8 cubes

-- The theorem stating the problem and expected solution
theorem number_of_distinct_stackings : 
  cubes = 8 ∧ edge_length = 1 ∧ valid_stackings cubes → ∃ (count : ℕ), count = 10 :=
by 
  sorry

end number_of_distinct_stackings_l19_19908


namespace abs_sum_lt_abs_sum_of_neg_product_l19_19417

theorem abs_sum_lt_abs_sum_of_neg_product 
  (a b : ℝ) : ab < 0 ↔ |a + b| < |a| + |b| := 
by 
  sorry

end abs_sum_lt_abs_sum_of_neg_product_l19_19417


namespace charley_pencils_final_count_l19_19939

def charley_initial_pencils := 50
def lost_pencils_while_moving := 8
def misplaced_fraction_first_week := 1 / 3
def lost_fraction_second_week := 1 / 4

theorem charley_pencils_final_count:
  let initial := charley_initial_pencils
  let after_moving := initial - lost_pencils_while_moving
  let misplaced_first_week := misplaced_fraction_first_week * after_moving
  let remaining_after_first_week := after_moving - misplaced_first_week
  let lost_second_week := lost_fraction_second_week * remaining_after_first_week
  let final_pencils := remaining_after_first_week - lost_second_week
  final_pencils = 21 := 
sorry

end charley_pencils_final_count_l19_19939


namespace perpendicular_vectors_l19_19667

open scoped BigOperators

noncomputable def i : ℝ × ℝ := (1, 0)
noncomputable def j : ℝ × ℝ := (0, 1)
noncomputable def u : ℝ × ℝ := (1, 3)
noncomputable def v : ℝ × ℝ := (3, -1)

theorem perpendicular_vectors :
  (u.1 * v.1 + u.2 * v.2) = 0 :=
by
  have hi : i = (1, 0) := rfl
  have hj : j = (0, 1) := rfl
  have hu : u = (1, 3) := rfl
  have hv : v = (3, -1) := rfl
  -- using the dot product definition for perpendicularity
  sorry

end perpendicular_vectors_l19_19667


namespace passes_through_origin_l19_19155

def parabola_A (x : ℝ) : ℝ := x^2 + 1
def parabola_B (x : ℝ) : ℝ := (x + 1)^2
def parabola_C (x : ℝ) : ℝ := x^2 + 2 * x
def parabola_D (x : ℝ) : ℝ := x^2 - x + 1

theorem passes_through_origin : 
  (parabola_A 0 ≠ 0) ∧
  (parabola_B 0 ≠ 0) ∧
  (parabola_C 0 = 0) ∧
  (parabola_D 0 ≠ 0) := 
by 
  sorry

end passes_through_origin_l19_19155


namespace sum_of_cubes_equality_l19_19172

theorem sum_of_cubes_equality (a b p n : ℕ) (hp : Nat.Prime p) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^3 + b^3 = p^n) ↔ 
  (∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
  (∃ k : ℕ, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
  (∃ k : ℕ, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) := sorry

end sum_of_cubes_equality_l19_19172


namespace bob_speed_l19_19453

theorem bob_speed (j_speed : ℝ) (b_headstart : ℝ) (t : ℝ) (j_catches_up : t = 20 / 60 ∧ j_speed = 9 ∧ b_headstart = 1) : 
  ∃ b_speed : ℝ, b_speed = 6 := 
by
  sorry

end bob_speed_l19_19453


namespace initial_tickets_l19_19513

theorem initial_tickets (X : ℕ) (h : (X - 22) + 15 = 18) : X = 25 :=
by
  sorry

end initial_tickets_l19_19513


namespace hours_spent_writing_l19_19702

-- Define the rates at which Jacob and Nathan write
def Nathan_rate : ℕ := 25        -- Nathan writes 25 letters per hour
def Jacob_rate : ℕ := 2 * Nathan_rate  -- Jacob writes twice as fast as Nathan

-- Define the combined rate
def combined_rate : ℕ := Nathan_rate + Jacob_rate

-- Define the total letters written and the hours spent
def total_letters : ℕ := 750
def hours_spent : ℕ := total_letters / combined_rate

-- The theorem to prove
theorem hours_spent_writing : hours_spent = 10 :=
by 
  -- Placeholder for the proof
  sorry

end hours_spent_writing_l19_19702


namespace sqrt_ceil_eq_one_range_of_x_l19_19296

/-- Given $[m]$ represents the largest integer not greater than $m$, prove $[\sqrt{2}] = 1$. -/
theorem sqrt_ceil_eq_one (floor : ℝ → ℤ) 
  (h_floor : ∀ m : ℝ, (floor m : ℝ) ≤ m ∧ ∀ z : ℤ, (z : ℝ) ≤ m → z ≤ floor m) :
  floor (Real.sqrt 2) = 1 :=
sorry

/-- Given $[m]$ represents the largest integer not greater than $m$ and $[3 + \sqrt{x}] = 6$, 
  prove $9 \leq x < 16$. -/
theorem range_of_x (floor : ℝ → ℤ) 
  (h_floor : ∀ m : ℝ, (floor m : ℝ) ≤ m ∧ ∀ z : ℤ, (z : ℝ) ≤ m → z ≤ floor m) 
  (x : ℝ) (h : floor (3 + Real.sqrt x) = 6) :
  9 ≤ x ∧ x < 16 :=
sorry

end sqrt_ceil_eq_one_range_of_x_l19_19296


namespace sampling_interval_l19_19478

theorem sampling_interval 
  (total_population : ℕ) 
  (individuals_removed : ℕ) 
  (population_after_removal : ℕ)
  (sampling_interval : ℕ) :
  total_population = 102 →
  individuals_removed = 2 →
  population_after_removal = total_population - individuals_removed →
  population_after_removal = 100 →
  ∃ s : ℕ, population_after_removal % s = 0 ∧ s = 10 := 
by
  sorry

end sampling_interval_l19_19478


namespace math_problem_l19_19085

theorem math_problem 
  (a b c : ℝ) 
  (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) 
  (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : a^2 + b^2 = c^2 + ab) : 
  c^2 + ab < a*c + b*c := 
sorry

end math_problem_l19_19085


namespace area_of_regular_octagon_l19_19171

-- Define a regular octagon with given diagonals
structure RegularOctagon where
  d_max : ℝ  -- length of the longest diagonal
  d_min : ℝ  -- length of the shortest diagonal

-- Theorem stating that the area of the regular octagon
-- is the product of its longest and shortest diagonals
theorem area_of_regular_octagon (O : RegularOctagon) : 
  let A := O.d_max * O.d_min
  A = O.d_max * O.d_min :=
by
  -- Proof to be filled in
  sorry

end area_of_regular_octagon_l19_19171


namespace inverse_h_l19_19423

-- Definitions from the problem conditions
def f (x : ℝ) : ℝ := 4 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 5
def h (x : ℝ) : ℝ := f (g x)

-- Statement of the theorem for the inverse of h
theorem inverse_h : ∀ x : ℝ, h⁻¹ x = (x + 18) / 12 :=
sorry

end inverse_h_l19_19423


namespace constant_expression_l19_19276

variable {x y m n : ℝ}

theorem constant_expression (hx : x^2 = 25) (hy : ∀ y : ℝ, (x + y) * (x - 2 * y) - m * y * (n * x - y) = 25) :
  m = 2 ∧ n = -1/2 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end constant_expression_l19_19276


namespace acute_angle_parallel_vectors_l19_19154

theorem acute_angle_parallel_vectors (x : ℝ) (a b : ℝ × ℝ)
    (h₁ : a = (Real.sin x, 1))
    (h₂ : b = (1 / 2, Real.cos x))
    (h₃ : ∃ k : ℝ, a = k • b ∧ k ≠ 0) :
    x = Real.pi / 4 :=
by
  sorry

end acute_angle_parallel_vectors_l19_19154


namespace usual_time_28_l19_19762

theorem usual_time_28 (R T : ℝ) (h1 : ∀ (d : ℝ), d = R * T)
  (h2 : ∀ (d : ℝ), d = (6/7) * R * (T - 4)) : T = 28 :=
by
  -- Variables:
  -- R : Usual rate of the boy
  -- T : Usual time to reach the school
  -- h1 : Expressing distance in terms of usual rate and time
  -- h2 : Expressing distance in terms of reduced rate and time minus 4
  sorry

end usual_time_28_l19_19762


namespace perpendicular_chords_square_sum_l19_19243

theorem perpendicular_chords_square_sum (d : ℝ) (r : ℝ) (x y : ℝ) 
  (h1 : r = d / 2)
  (h2 : x = r)
  (h3 : y = r) 
  : (x^2 + y^2) + (x^2 + y^2) = d^2 :=
by
  sorry

end perpendicular_chords_square_sum_l19_19243


namespace arc_length_correct_l19_19386

noncomputable def radius : ℝ :=
  5

noncomputable def area_of_sector : ℝ :=
  8.75

noncomputable def arc_length (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * 2 * Real.pi * r

theorem arc_length_correct :
  ∃ θ, arc_length θ radius = 3.5 ∧ (θ / 360) * Real.pi * radius^2 = area_of_sector :=
by
  sorry

end arc_length_correct_l19_19386


namespace coplanar_vectors_m_value_l19_19117

variable (m : ℝ)
variable (α β : ℝ)
def a := (5, 9, m)
def b := (1, -1, 2)
def c := (2, 5, 1)

theorem coplanar_vectors_m_value :
  ∃ (α β : ℝ), (5 = α + 2 * β) ∧ (9 = -α + 5 * β) ∧ (m = 2 * α + β) → m = 4 :=
by
  sorry

end coplanar_vectors_m_value_l19_19117


namespace linear_coefficient_is_one_l19_19482

-- Define the given equation and the coefficient of the linear term
variables {x m : ℝ}
def equation := (m - 3) * x + 4 * m^2 - 2 * m - 1 - m * x + 6

-- State the main theorem: the coefficient of the linear term in the equation is 1 given the conditions
theorem linear_coefficient_is_one (m : ℝ) (hm_neq_3 : m ≠ 3) :
  (m - 3) - m = 1 :=
by sorry

end linear_coefficient_is_one_l19_19482


namespace largest_lcm_among_pairs_is_45_l19_19477

theorem largest_lcm_among_pairs_is_45 :
  max (max (max (max (max (Nat.lcm 15 3) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10)) (Nat.lcm 15 15) = 45 :=
by
  sorry

end largest_lcm_among_pairs_is_45_l19_19477


namespace value_of_f_at_3_l19_19376

def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem value_of_f_at_3 : f 3 = 15 :=
by
  -- This proof needs to be filled in
  sorry

end value_of_f_at_3_l19_19376


namespace sin_is_odd_and_has_zero_point_l19_19160

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def has_zero_point (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

theorem sin_is_odd_and_has_zero_point :
  is_odd_function sin ∧ has_zero_point sin := 
  by sorry

end sin_is_odd_and_has_zero_point_l19_19160


namespace math_problem_l19_19109

noncomputable def ellipse_standard_equation (a b : ℝ) : Prop :=
  a = 2 ∧ b = Real.sqrt 3 ∧ (∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def constant_slope_sum (T R S : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  T = (4, 0) ∧ l (1, 0) ∧ 
  (∀ TR TS : ℝ, (TR = (R.2 / (R.1 - 4)) ∧ TS = (S.2 / (S.1 - 4)) ∧ 
  (TR + TS = 0)))

theorem math_problem 
  {a b : ℝ} {T R S : ℝ × ℝ} {l : ℝ × ℝ → Prop} : 
  ellipse_standard_equation a b ∧ constant_slope_sum T R S l :=
by
  sorry

end math_problem_l19_19109


namespace part1_solution_set_part2_range_of_a_l19_19178

-- Define the function f for part 1 
def f_part1 (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

-- Define the function f for part 2 
def f_part2 (x a : ℝ) : ℝ := |2*x + 1| + |a*x - 1|

-- Theorem for part 1
theorem part1_solution_set (x : ℝ) : 
  (f_part1 x) ≥ 3 ↔ x ∈ (Set.Iic (-3/4) ∪ Set.Ici (3/4)) :=
sorry

-- Theorem for part 2
theorem part2_range_of_a (a : ℝ) : 
  (a > 0) → (∃ x : ℝ, f_part2 x a < (a / 2) + 1) ↔ (a ∈ Set.Ioi 2) :=
sorry

end part1_solution_set_part2_range_of_a_l19_19178


namespace scientific_notation_of_10900_l19_19794

theorem scientific_notation_of_10900 : ∃ (x : ℝ) (n : ℤ), 10900 = x * 10^n ∧ x = 1.09 ∧ n = 4 := by
  use 1.09
  use 4
  sorry

end scientific_notation_of_10900_l19_19794


namespace find_expression_value_l19_19003

variable (x y z : ℚ)
variable (h1 : x - y + 2 * z = 1)
variable (h2 : x + y + 4 * z = 3)

theorem find_expression_value : x + 2 * y + 5 * z = 4 := 
by {
  sorry
}

end find_expression_value_l19_19003


namespace pizza_promotion_savings_l19_19118

theorem pizza_promotion_savings :
  let regular_price : ℕ := 18
  let promo_price : ℕ := 5
  let num_pizzas : ℕ := 3
  let total_regular_price := num_pizzas * regular_price
  let total_promo_price := num_pizzas * promo_price
  let total_savings := total_regular_price - total_promo_price
  total_savings = 39 :=
by
  sorry

end pizza_promotion_savings_l19_19118


namespace sum_arithmetic_sequence_ge_four_l19_19387

theorem sum_arithmetic_sequence_ge_four
  (a_n : ℕ → ℚ) -- arithmetic sequence
  (S : ℕ → ℚ) -- sum of the first n terms of the sequence
  (h_arith_seq : ∀ n, S n = (n * a_n 1) + (n * (n - 1) / 2) * (a_n 2 - a_n 1))
  (p q : ℕ)
  (hpq_ne : p ≠ q)
  (h_sp : S p = p / q)
  (h_sq : S q = q / p) :
  S (p + q) ≥ 4 :=
by
  sorry

end sum_arithmetic_sequence_ge_four_l19_19387


namespace complement_intersection_l19_19800

-- Definitions for the sets
def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2, 3}

-- Statement to be proved
theorem complement_intersection (hU : U = {0, 1, 2, 3}) (hA : A = {0, 1}) (hB : B = {1, 2, 3}) :
  ((U \ A) ∩ B) = {2, 3} :=
by
  -- Greek delta: skip proof details
  sorry

end complement_intersection_l19_19800


namespace ones_digit_7_pow_35_l19_19661

theorem ones_digit_7_pow_35 : (7^35) % 10 = 3 := 
by
  sorry

end ones_digit_7_pow_35_l19_19661


namespace joan_games_l19_19474

theorem joan_games (games_this_year games_total games_last_year : ℕ) 
  (h1 : games_this_year = 4) 
  (h2 : games_total = 9) 
  (h3 : games_total = games_this_year + games_last_year) :
  games_last_year = 5 :=
by {
  -- The proof goes here
  sorry
}

end joan_games_l19_19474


namespace intersection_M_N_l19_19566

-- Define sets M and N
def M := { x : ℝ | ∃ t : ℝ, x = 2^(-t) }
def N := { y : ℝ | ∃ x : ℝ, y = Real.sin x }

-- Prove the intersection of M and N
theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y ≤ 1 } :=
by sorry

end intersection_M_N_l19_19566


namespace solve_quadratic_equation_l19_19502

theorem solve_quadratic_equation (x : ℝ) : x^2 = 100 → x = -10 ∨ x = 10 :=
by
  intro h
  sorry

end solve_quadratic_equation_l19_19502


namespace highest_mean_possible_l19_19867

def max_arithmetic_mean (g : Matrix (Fin 3) (Fin 3) ℕ) : ℚ := 
  let mean (a b c d : ℕ) : ℚ := (a + b + c + d : ℚ) / 4
  let circles := [
    mean (g 0 0) (g 0 1) (g 1 0) (g 1 1),
    mean (g 0 1) (g 0 2) (g 1 1) (g 1 2),
    mean (g 1 0) (g 1 1) (g 2 0) (g 2 1),
    mean (g 1 1) (g 1 2) (g 2 1) (g 2 2)
  ]
  (circles.sum / 4)

theorem highest_mean_possible :
  ∃ g : Matrix (Fin 3) (Fin 3) ℕ, 
  (∀ i j, 1 ≤ g i j ∧ g i j ≤ 9) ∧ 
  max_arithmetic_mean g = 6.125 :=
by
  sorry

end highest_mean_possible_l19_19867


namespace value_of_each_baseball_card_l19_19677

theorem value_of_each_baseball_card (x : ℝ) (h : 2 * x + 3 = 15) : x = 6 := by
  sorry

end value_of_each_baseball_card_l19_19677


namespace unique_right_triangle_construction_l19_19213

noncomputable def right_triangle_condition (c f : ℝ) : Prop :=
  f < c / 2

theorem unique_right_triangle_construction (c f : ℝ) (h_c : 0 < c) (h_f : 0 < f) :
  right_triangle_condition c f :=
  sorry

end unique_right_triangle_construction_l19_19213


namespace symmetric_circle_eq_l19_19462

theorem symmetric_circle_eq (x y : ℝ) :
  (x + 1)^2 + (y - 1)^2 = 1 → x - y = 1 → (x - 2)^2 + (y + 2)^2 = 1 :=
by
  sorry

end symmetric_circle_eq_l19_19462


namespace trains_meet_at_distance_360_km_l19_19756

-- Define the speeds of the trains
def speed_A : ℕ := 30 -- speed of train A in kmph
def speed_B : ℕ := 40 -- speed of train B in kmph
def speed_C : ℕ := 60 -- speed of train C in kmph

-- Define the head starts in hours for trains A and B
def head_start_A : ℕ := 9 -- head start for train A in hours
def head_start_B : ℕ := 3 -- head start for train B in hours

-- Define the distances traveled by trains A and B by the time train C starts at 6 p.m.
def distance_A_start : ℕ := speed_A * head_start_A -- distance traveled by train A by 6 p.m.
def distance_B_start : ℕ := speed_B * head_start_B -- distance traveled by train B by 6 p.m.

-- The formula to calculate the distance after t hours from 6 p.m. for each train
def distance_A (t : ℕ) : ℕ := distance_A_start + speed_A * t
def distance_B (t : ℕ) : ℕ := distance_B_start + speed_B * t
def distance_C (t : ℕ) : ℕ := speed_C * t

-- Problem statement to prove the point where all three trains meet
theorem trains_meet_at_distance_360_km : ∃ t : ℕ, distance_A t = 360 ∧ distance_B t = 360 ∧ distance_C t = 360 := by
  sorry

end trains_meet_at_distance_360_km_l19_19756


namespace distance_fall_l19_19029

-- Given conditions as definitions
def velocity (g : ℝ) (t : ℝ) := g * t

-- The theorem stating the relationship between time t0 and distance S
theorem distance_fall (g : ℝ) (t0 : ℝ) : 
  (∫ t in (0 : ℝ)..t0, velocity g t) = (1/2) * g * t0^2 :=
by 
  sorry

end distance_fall_l19_19029


namespace number_division_l19_19420

theorem number_division (m k n : ℤ) (h : n = m * k + 1) : n = m * k + 1 :=
by
  exact h

end number_division_l19_19420


namespace total_blue_balloons_l19_19852

def joan_blue_balloons : ℕ := 60
def melanie_blue_balloons : ℕ := 85
def alex_blue_balloons : ℕ := 37
def gary_blue_balloons : ℕ := 48

theorem total_blue_balloons :
  joan_blue_balloons + melanie_blue_balloons + alex_blue_balloons + gary_blue_balloons = 230 :=
by simp [joan_blue_balloons, melanie_blue_balloons, alex_blue_balloons, gary_blue_balloons]

end total_blue_balloons_l19_19852


namespace tangent_line_extreme_values_l19_19018

-- Define the function f and its conditions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- Given conditions
def cond1 (a b : ℝ) : Prop := 3 * a * 2^2 + b = 0
def cond2 (a b : ℝ) : Prop := a * 2^3 + b * 2 + 2 = -14

-- Part 1: Tangent line equation at (1, f(1))
theorem tangent_line (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) : 
  ∃ m c : ℝ, m = -9 ∧ c = 9 ∧ (∀ y : ℝ, y = f a b 1 → 9 * 1 + y = 0) :=
sorry

-- Part 2: Extreme values on [-3, 3]
-- Define critical points and endpoints
def f_value_at (a b x : ℝ) := f a b x

theorem extreme_values (a b : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) :
  ∃ min max : ℝ, min = -14 ∧ max = 18 ∧ 
  f_value_at a b 2 = min ∧ f_value_at a b (-2) = max :=
sorry

end tangent_line_extreme_values_l19_19018


namespace evaluate_square_of_sum_l19_19582

theorem evaluate_square_of_sum (x y : ℕ) (h1 : x + y = 20) (h2 : 2 * x + y = 27) : (x + y) ^ 2 = 400 :=
by
  sorry

end evaluate_square_of_sum_l19_19582


namespace coupon_savings_difference_l19_19819

-- Definitions based on conditions
def P (p : ℝ) := 120 + p
def savings_coupon_A (p : ℝ) := 24 + 0.20 * p
def savings_coupon_B := 35
def savings_coupon_C (p : ℝ) := 0.30 * p

-- Conditions
def condition_A_saves_at_least_B (p : ℝ) := savings_coupon_A p ≥ savings_coupon_B
def condition_A_saves_at_least_C (p : ℝ) := savings_coupon_A p ≥ savings_coupon_C p

-- Proof problem
theorem coupon_savings_difference :
  ∀ (p : ℝ), 55 ≤ p ∧ p ≤ 240 → (P 240 - P 55) = 185 :=
by
  sorry

end coupon_savings_difference_l19_19819


namespace least_possible_value_of_d_l19_19766

theorem least_possible_value_of_d
  (x y z : ℤ)
  (h1 : x < y)
  (h2 : y < z)
  (h3 : y - x > 5)
  (hx_even : x % 2 = 0)
  (hy_odd : y % 2 = 1)
  (hz_odd : z % 2 = 1) :
  (z - x) = 9 := 
sorry

end least_possible_value_of_d_l19_19766


namespace work_done_in_a_day_l19_19432

noncomputable def A : ℕ := sorry
noncomputable def B_days : ℕ := A / 2

theorem work_done_in_a_day (h : 1 / A + 2 / A = 1 / 6) : A = 18 := 
by 
  -- skipping the proof as instructed
  sorry

end work_done_in_a_day_l19_19432


namespace algebra_inequality_l19_19783

theorem algebra_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a^3 + b^3 + c^3 = 3) : 
  1 / (a^2 + a + 1) + 1 / (b^2 + b + 1) + 1 / (c^2 + c + 1) ≥ 1 := 
by 
  sorry

end algebra_inequality_l19_19783


namespace quincy_monthly_payment_l19_19415

-- Definitions based on the conditions:
def car_price : ℕ := 20000
def down_payment : ℕ := 5000
def loan_years : ℕ := 5
def months_in_year : ℕ := 12

-- The mathematical problem to be proven:
theorem quincy_monthly_payment :
  let amount_to_finance := car_price - down_payment
  let total_months := loan_years * months_in_year
  amount_to_finance / total_months = 250 := by
  sorry

end quincy_monthly_payment_l19_19415


namespace factors_of_P_factorization_of_P_factorize_expression_l19_19320

noncomputable def P (a b c : ℝ) : ℝ :=
  a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b)

theorem factors_of_P (a b c : ℝ) :
  (a - b ∣ P a b c) ∧ (b - c ∣ P a b c) ∧ (c - a ∣ P a b c) :=
sorry

theorem factorization_of_P (a b c : ℝ) :
  P a b c = -(a - b) * (b - c) * (c - a) :=
sorry

theorem factorize_expression (x y z : ℝ) :
  (x + y + z)^3 - x^3 - y^3 - z^3 = 3 * (x + y) * (y + z) * (z + x) :=
sorry

end factors_of_P_factorization_of_P_factorize_expression_l19_19320


namespace simplify_fraction_l19_19563

-- Define the fraction and the GCD condition
def fraction_numerator : ℕ := 66
def fraction_denominator : ℕ := 4356
def gcd_condition : ℕ := Nat.gcd fraction_numerator fraction_denominator

-- State the theorem that the fraction simplifies to 1/66 given the GCD condition
theorem simplify_fraction (h : gcd_condition = 66) : (fraction_numerator / fraction_denominator = 1 / 66) :=
  sorry

end simplify_fraction_l19_19563


namespace total_snakes_owned_l19_19551

theorem total_snakes_owned 
  (total_people : ℕ)
  (only_dogs only_cats only_birds only_snakes : ℕ)
  (cats_and_dogs birds_and_dogs birds_and_cats snakes_and_dogs snakes_and_cats snakes_and_birds : ℕ)
  (cats_dogs_snakes cats_dogs_birds cats_birds_snakes dogs_birds_snakes all_four_pets : ℕ)
  (h1 : total_people = 150)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_birds = 10)
  (h5 : only_snakes = 7)
  (h6 : cats_and_dogs = 15)
  (h7 : birds_and_dogs = 12)
  (h8 : birds_and_cats = 8)
  (h9 : snakes_and_dogs = 3)
  (h10 : snakes_and_cats = 4)
  (h11 : snakes_and_birds = 2)
  (h12 : cats_dogs_snakes = 5)
  (h13 : cats_dogs_birds = 4)
  (h14 : cats_birds_snakes = 6)
  (h15 : dogs_birds_snakes = 9)
  (h16 : all_four_pets = 10) : 
  7 + 3 + 4 + 2 + 5 + 6 + 9 + 10 = 46 := 
sorry

end total_snakes_owned_l19_19551


namespace find_sum_of_squares_l19_19361

theorem find_sum_of_squares (x y : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x * y + x + y = 35) (h4 : x^2 * y + x * y^2 = 210) : x^2 + y^2 = 154 :=
sorry

end find_sum_of_squares_l19_19361


namespace find_d_l19_19112

noncomputable def d : ℝ := 3.44

theorem find_d :
  (∃ x : ℝ, (3 * x^2 + 19 * x - 84 = 0) ∧ x = ⌊d⌋) ∧
  (∃ y : ℝ, (5 * y^2 - 26 * y + 12 = 0) ∧ y = d - ⌊d⌋) →
  d = 3.44 :=
by
  sorry

end find_d_l19_19112


namespace dasha_strip_dimensions_l19_19393

theorem dasha_strip_dimensions (a b c : ℕ) (h1 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) : 
  (a = 1 ∧ (b + c = 22)) ∨ (a = 22 ∧ (b + c = 1)) :=
by sorry

end dasha_strip_dimensions_l19_19393


namespace total_distance_traveled_l19_19772

theorem total_distance_traveled :
  let car_speed1 := 90
  let car_time1 := 2
  let car_speed2 := 60
  let car_time2 := 1
  let train_speed := 100
  let train_time := 2.5
  let distance_car1 := car_speed1 * car_time1
  let distance_car2 := car_speed2 * car_time2
  let distance_train := train_speed * train_time
  distance_car1 + distance_car2 + distance_train = 490 := by
  sorry

end total_distance_traveled_l19_19772


namespace union_of_sets_l19_19798

open Set

theorem union_of_sets :
  ∀ (P Q : Set ℕ), P = {1, 2} → Q = {2, 3} → P ∪ Q = {1, 2, 3} :=
by
  intros P Q hP hQ
  rw [hP, hQ]
  exact sorry

end union_of_sets_l19_19798


namespace second_cart_travel_distance_l19_19881

-- Given definitions:
def first_cart_first_term : ℕ := 6
def first_cart_common_difference : ℕ := 8
def second_cart_first_term : ℕ := 7
def second_cart_common_difference : ℕ := 9

-- Given times:
def time_first_cart : ℕ := 35
def time_second_cart : ℕ := 33

-- Arithmetic series sum formula
def arithmetic_series_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Total distance traveled by the second cart
noncomputable def distance_second_cart : ℕ :=
  arithmetic_series_sum second_cart_first_term second_cart_common_difference time_second_cart

-- Theorem to prove the distance traveled by the second cart
theorem second_cart_travel_distance : distance_second_cart = 4983 :=
  sorry

end second_cart_travel_distance_l19_19881


namespace perpendicular_vectors_x_value_l19_19093

-- Define the vectors a and b
def a : ℝ × ℝ := (3, -1)
def b (x : ℝ) : ℝ × ℝ := (1, x)

-- Define the dot product function for vectors in ℝ^2
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

-- The mathematical statement to prove
theorem perpendicular_vectors_x_value (x : ℝ) (h : dot_product a (b x) = 0) : x = 3 :=
by
  sorry

end perpendicular_vectors_x_value_l19_19093


namespace find_a_value_l19_19694

noncomputable def A (a : ℝ) : Set ℝ := {x | x = a}
noncomputable def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else {x | a * x = 1}

theorem find_a_value (a : ℝ) :
  (A a ∩ B a = B a) → (a = 1 ∨ a = -1 ∨ a = 0) :=
by
  intro h
  sorry

end find_a_value_l19_19694


namespace Daniela_is_12_years_old_l19_19722

noncomputable def auntClaraAge : Nat := 60

noncomputable def evelinaAge : Nat := auntClaraAge / 3

noncomputable def fidelAge : Nat := evelinaAge - 6

noncomputable def caitlinAge : Nat := fidelAge / 2

noncomputable def danielaAge : Nat := evelinaAge - 8

theorem Daniela_is_12_years_old (h_auntClaraAge : auntClaraAge = 60)
                                (h_evelinaAge : evelinaAge = 60 / 3)
                                (h_fidelAge : fidelAge = (60 / 3) - 6)
                                (h_caitlinAge : caitlinAge = ((60 / 3) - 6) / 2)
                                (h_danielaAge : danielaAge = (60 / 3) - 8) :
  danielaAge = 12 := 
  sorry

end Daniela_is_12_years_old_l19_19722


namespace reduced_price_l19_19963

-- Definitions based on given conditions
def original_price (P : ℝ) : Prop := P > 0

def condition1 (P X : ℝ) : Prop := P * X = 700

def condition2 (P X : ℝ) : Prop := 0.7 * P * (X + 3) = 700

-- Main theorem to prove the reduced price per kg is 70
theorem reduced_price (P X : ℝ) (h1 : original_price P) (h2 : condition1 P X) (h3 : condition2 P X) : 
  0.7 * P = 70 := sorry

end reduced_price_l19_19963


namespace smallest_angle_of_isosceles_trapezoid_l19_19333

def is_isosceles_trapezoid (a b c d : ℝ) : Prop :=
  a = c ∧ b = d ∧ a + b + c + d = 360 ∧ a + 3 * b = 150

theorem smallest_angle_of_isosceles_trapezoid (a b : ℝ) (h1 : is_isosceles_trapezoid a b a (a + 2 * b))
  : a = 47 :=
sorry

end smallest_angle_of_isosceles_trapezoid_l19_19333


namespace find_m_and_c_l19_19265

-- Definitions & conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 3 }
def B (m : ℝ) : Point := { x := -6, y := m }

def line (c : ℝ) (p : Point) : Prop := p.x + p.y + c = 0

-- Theorem statement
theorem find_m_and_c (m : ℝ) (c : ℝ) (hc : line c A) (hcB : line c (B m)) :
  m = 3 ∧ c = -2 :=
  by
  sorry

end find_m_and_c_l19_19265


namespace maximum_weight_truck_can_carry_l19_19885

-- Definitions for the conditions.
def weight_boxes : Nat := 100 * 100
def weight_crates : Nat := 10 * 60
def weight_sacks : Nat := 50 * 50
def weight_additional_bags : Nat := 10 * 40

-- Summing up all the weights.
def total_weight : Nat :=
  weight_boxes + weight_crates + weight_sacks + weight_additional_bags

-- The theorem stating the maximum weight.
theorem maximum_weight_truck_can_carry : total_weight = 13500 := by
  sorry

end maximum_weight_truck_can_carry_l19_19885


namespace math_problem_l19_19724

theorem math_problem :
  (2^8 + 4^5) * (1^3 - (-1)^3)^2 = 5120 :=
by
  sorry

end math_problem_l19_19724


namespace total_tickets_sold_l19_19751

-- Definitions and conditions
def orchestra_ticket_price : ℕ := 12
def balcony_ticket_price : ℕ := 8
def total_revenue : ℕ := 3320
def ticket_difference : ℕ := 190

-- Variables
variables (x y : ℕ) -- x is the number of orchestra tickets, y is the number of balcony tickets

-- Statements of conditions
def revenue_eq : Prop := orchestra_ticket_price * x + balcony_ticket_price * y = total_revenue
def tickets_relation : Prop := y = x + ticket_difference

-- The proof problem statement
theorem total_tickets_sold (h1 : revenue_eq x y) (h2 : tickets_relation x y) : x + y = 370 :=
by
  sorry

end total_tickets_sold_l19_19751


namespace cost_to_fly_A_to_B_l19_19799

noncomputable def flight_cost (distance : ℕ) : ℕ := (distance * 10 / 100) + 100

theorem cost_to_fly_A_to_B :
  flight_cost 3250 = 425 :=
by
  sorry

end cost_to_fly_A_to_B_l19_19799


namespace die_face_never_touches_board_l19_19742

theorem die_face_never_touches_board : 
  ∃ (cube : Type) (roll : cube → cube) (occupied : Fin 8 × Fin 8 → cube → Prop),
    (∀ p : Fin 8 × Fin 8, ∃ c : cube, occupied p c) ∧ 
    (∃ f : cube, ¬ (∃ p : Fin 8 × Fin 8, occupied p f)) :=
by sorry

end die_face_never_touches_board_l19_19742


namespace add_pure_chocolate_to_achieve_percentage_l19_19807

/--
Given:
    Initial amount of chocolate topping: 620 ounces.
    Initial chocolate percentage: 10%.
    Desired total weight of the final mixture: 1000 ounces.
    Desired chocolate percentage in the final mixture: 70%.
Prove:
    The amount of pure chocolate to be added to achieve the desired mixture is 638 ounces.
-/
theorem add_pure_chocolate_to_achieve_percentage :
  ∃ x : ℝ,
    0.10 * 620 + x = 0.70 * 1000 ∧
    x = 638 :=
by
  sorry

end add_pure_chocolate_to_achieve_percentage_l19_19807


namespace evaluate_neg_64_exp_4_over_3_l19_19108

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l19_19108


namespace not_necessarily_periodic_l19_19810

-- Define the conditions of the problem
noncomputable def a : ℕ → ℕ := sorry
noncomputable def t : ℕ → ℕ := sorry
axiom h_t : ∀ k : ℕ, ∃ t_k : ℕ, ∀ n : ℕ, a (k + n * t_k) = a k

-- The theorem stating that the sequence is not necessarily periodic
theorem not_necessarily_periodic : ¬ ∃ T : ℕ, ∀ k : ℕ, a (k + T) = a k := sorry

end not_necessarily_periodic_l19_19810


namespace solve_y_l19_19567

theorem solve_y (y : ℚ) (h : (3 * y) / 7 = 14) : y = 98 / 3 := 
by sorry

end solve_y_l19_19567


namespace daughter_age_l19_19961

theorem daughter_age (m d : ℕ) (h1 : m + d = 60) (h2 : m - 10 = 7 * (d - 10)) : d = 15 :=
sorry

end daughter_age_l19_19961


namespace largest_four_digit_number_with_property_l19_19696

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end largest_four_digit_number_with_property_l19_19696


namespace maria_zoo_ticket_discount_percentage_l19_19767

theorem maria_zoo_ticket_discount_percentage 
  (regular_price : ℝ) (paid_price : ℝ) (discount_percentage : ℝ)
  (h1 : regular_price = 15) (h2 : paid_price = 9) :
  discount_percentage = 40 :=
by
  sorry

end maria_zoo_ticket_discount_percentage_l19_19767


namespace value_range_of_f_in_interval_l19_19107

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

theorem value_range_of_f_in_interval : 
  ∀ x, (2 ≤ x ∧ x ≤ 4) → (1/2 ≤ f x ∧ f x ≤ 2/3) := 
by
  sorry

end value_range_of_f_in_interval_l19_19107


namespace smallest_n_condition_l19_19850

def pow_mod (a b m : ℕ) : ℕ := a^(b % m)

def n (r s : ℕ) : ℕ := 2^r - 16^s

def r_condition (r : ℕ) : Prop := ∃ k : ℕ, r = 3 * k + 1

def s_condition (s : ℕ) : Prop := ∃ h : ℕ, s = 3 * h + 2

theorem smallest_n_condition (r s : ℕ) (hr : r_condition r) (hs : s_condition s) :
  (n r s) % 7 = 5 → (n r s) = 768 := sorry

end smallest_n_condition_l19_19850


namespace hyperbola_range_m_l19_19887

-- Define the condition that the equation represents a hyperbola
def isHyperbola (m : ℝ) : Prop := (2 + m) * (m + 1) < 0

-- The theorem stating the range of m given the condition
theorem hyperbola_range_m (m : ℝ) : isHyperbola m → -2 < m ∧ m < -1 := by
  sorry

end hyperbola_range_m_l19_19887


namespace bill_bought_60_rats_l19_19159

def chihuahuas_and_rats (C R : ℕ) : Prop :=
  C + R = 70 ∧ R = 6 * C

theorem bill_bought_60_rats (C R : ℕ) (h : chihuahuas_and_rats C R) : R = 60 :=
by
  sorry

end bill_bought_60_rats_l19_19159


namespace students_from_other_communities_l19_19261

noncomputable def percentageMuslims : ℝ := 0.41
noncomputable def percentageHindus : ℝ := 0.32
noncomputable def percentageSikhs : ℝ := 0.12
noncomputable def totalStudents : ℝ := 1520

theorem students_from_other_communities : 
  totalStudents * (1 - (percentageMuslims + percentageHindus + percentageSikhs)) = 228 := 
by 
  sorry

end students_from_other_communities_l19_19261


namespace problem_statement_l19_19839

theorem problem_statement (x : ℤ) (y : ℝ) (h : y = 0.5) : 
  (⌈x + y⌉ - ⌊x + y⌋ = 1) ∧ (⌈x + y⌉ - (x + y) = 0.5) := 
by 
  sorry

end problem_statement_l19_19839


namespace jacob_additional_money_needed_l19_19733

/-- Jacob's total trip cost -/
def trip_cost : ℕ := 5000

/-- Jacob's hourly wage -/
def hourly_wage : ℕ := 20

/-- Jacob's working hours -/
def working_hours : ℕ := 10

/-- Income from job -/
def job_income : ℕ := hourly_wage * working_hours

/-- Price per cookie -/
def cookie_price : ℕ := 4

/-- Number of cookies sold -/
def cookies_sold : ℕ := 24

/-- Income from cookies -/
def cookie_income : ℕ := cookie_price * cookies_sold

/-- Lottery ticket cost -/
def lottery_ticket_cost : ℕ := 10

/-- Lottery win amount -/
def lottery_win : ℕ := 500

/-- Money received from each sister -/
def sister_gift : ℕ := 500

/-- Total income from job and cookies -/
def income_without_expenses : ℕ := job_income + cookie_income

/-- Income after lottery ticket purchase -/
def income_after_ticket : ℕ := income_without_expenses - lottery_ticket_cost

/-- Total income after lottery win -/
def income_with_lottery : ℕ := income_after_ticket + lottery_win

/-- Total gift from sisters -/
def total_sisters_gift : ℕ := 2 * sister_gift

/-- Total money Jacob has -/
def total_money : ℕ := income_with_lottery + total_sisters_gift

/-- Additional amount needed by Jacob -/
def additional_needed : ℕ := trip_cost - total_money

theorem jacob_additional_money_needed : additional_needed = 3214 := by
  sorry

end jacob_additional_money_needed_l19_19733


namespace convert_decimal_to_vulgar_fraction_l19_19548

theorem convert_decimal_to_vulgar_fraction : (32 : ℝ) / 100 = (8 : ℝ) / 25 :=
by
  sorry

end convert_decimal_to_vulgar_fraction_l19_19548


namespace shift_gives_f_l19_19421

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem shift_gives_f :
  (∀ x, f x = g (x + Real.pi / 3)) :=
  by
  sorry

end shift_gives_f_l19_19421


namespace periodic_sequence_not_constant_l19_19959

theorem periodic_sequence_not_constant :
  ∃ (x : ℕ → ℤ), (∀ n : ℕ, x (n+1) = 2 * x n + 3 * x (n-1)) ∧ (∃ T > 0, ∀ n : ℕ, x (n+T) = x n) ∧ (∃ n m : ℕ, n ≠ m ∧ x n ≠ x m) :=
sorry

end periodic_sequence_not_constant_l19_19959


namespace new_energy_vehicle_price_l19_19411

theorem new_energy_vehicle_price (x : ℝ) :
  (5000 / (x + 1)) = (5000 * (1 - 0.2)) / x :=
sorry

end new_energy_vehicle_price_l19_19411


namespace quadratic_real_roots_l19_19315

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
  sorry

end quadratic_real_roots_l19_19315


namespace total_coins_l19_19352

theorem total_coins (total_value : ℕ) (value_2_coins : ℕ) (num_2_coins : ℕ) (num_1_coins : ℕ) : 
  total_value = 402 ∧ value_2_coins = 2 * num_2_coins ∧ num_2_coins = 148 ∧ total_value = value_2_coins + num_1_coins →
  num_1_coins + num_2_coins = 254 :=
by
  intros h
  sorry

end total_coins_l19_19352


namespace g_of_f_at_3_eq_1902_l19_19408

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3 * x^2 + x + 2

theorem g_of_f_at_3_eq_1902 : g (f 3) = 1902 := by
  sorry

end g_of_f_at_3_eq_1902_l19_19408


namespace total_earrings_l19_19946

-- Definitions based on the given conditions
def bella_earrings : ℕ := 10
def monica_earrings : ℕ := 4 * bella_earrings
def rachel_earrings : ℕ := monica_earrings / 2
def olivia_earrings : ℕ := bella_earrings + monica_earrings + rachel_earrings + 5

-- The theorem to prove the total number of earrings
theorem total_earrings : bella_earrings + monica_earrings + rachel_earrings + olivia_earrings = 145 := by
  sorry

end total_earrings_l19_19946


namespace solve_for_a_l19_19877

theorem solve_for_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) (h3 : (51^2012 + a) % 13 = 0) : a = 12 :=
by 
  sorry

end solve_for_a_l19_19877


namespace joy_reading_rate_l19_19222

theorem joy_reading_rate
  (h1 : ∀ t: ℕ, t = 20 → ∀ p: ℕ, p = 8 → ∀ t': ℕ, t' = 60 → ∃ p': ℕ, p' = (p * t') / t)
  (h2 : ∀ t: ℕ, t = 5 * 60 → ∀ p: ℕ, p = 120):
  ∃ r: ℕ, r = 24 :=
by
  sorry

end joy_reading_rate_l19_19222


namespace gcd_12345_6789_l19_19274

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l19_19274


namespace minimum_green_sticks_l19_19481

def natasha_sticks (m n : ℕ) : ℕ :=
  if (m = 3 ∧ n = 3) then 5 else 0

theorem minimum_green_sticks (m n : ℕ) (grid : m = 3 ∧ n = 3) :
  natasha_sticks m n = 5 :=
by
  sorry

end minimum_green_sticks_l19_19481


namespace price_decrease_proof_l19_19438

-- Definitions based on the conditions
def original_price (C : ℝ) : ℝ := C
def new_price (C : ℝ) : ℝ := 0.76 * C

theorem price_decrease_proof (C : ℝ) : new_price C = 421.05263157894734 :=
by
  sorry

end price_decrease_proof_l19_19438


namespace arithmetic_sequence_tenth_term_l19_19865

noncomputable def prove_tenth_term (a d: ℤ) (h1: a + 2*d = 10) (h2: a + 7*d = 30) : Prop :=
  a + 9*d = 38

theorem arithmetic_sequence_tenth_term (a d: ℤ) (h1: a + 2*d = 10) (h2: a + 7*d = 30) : prove_tenth_term a d h1 h2 :=
by
  sorry

end arithmetic_sequence_tenth_term_l19_19865


namespace jordan_oreos_l19_19008

def oreos (james jordan total : ℕ) : Prop :=
  james = 2 * jordan + 3 ∧
  jordan + james = total

theorem jordan_oreos (J : ℕ) (h : oreos (2 * J + 3) J 36) : J = 11 :=
by
  sorry

end jordan_oreos_l19_19008


namespace number_of_friends_gave_money_l19_19549

-- Definition of given data in conditions
def amount_per_friend : ℕ := 6
def total_amount : ℕ := 30

-- Theorem to be proved
theorem number_of_friends_gave_money : total_amount / amount_per_friend = 5 :=
by
  sorry

end number_of_friends_gave_money_l19_19549


namespace Irja_wins_probability_l19_19169

noncomputable def probability_irja_wins : ℚ :=
  let X0 : ℚ := 4 / 7
  X0

theorem Irja_wins_probability :
  probability_irja_wins = 4 / 7 :=
sorry

end Irja_wins_probability_l19_19169


namespace customers_who_didnt_tip_l19_19072

def initial_customers : ℕ := 39
def added_customers : ℕ := 12
def customers_who_tipped : ℕ := 2

theorem customers_who_didnt_tip : initial_customers + added_customers - customers_who_tipped = 49 := by
  sorry

end customers_who_didnt_tip_l19_19072


namespace calculate_expression_l19_19604

theorem calculate_expression (x : ℕ) (h : x = 3) : x + x * x^(x - 1) = 30 := by
  rw [h]
  -- Proof steps would go here but we are including only the statement
  sorry

end calculate_expression_l19_19604


namespace female_students_transfer_l19_19782

theorem female_students_transfer (x y z : ℕ) 
  (h1 : ∀ B : ℕ, B = x - 4) 
  (h2 : ∀ C : ℕ, C = x - 5)
  (h3 : ∀ B' : ℕ, B' = x - 4 + y - z)
  (h4 : ∀ C' : ℕ, C' = x + z - 7) 
  (h5 : x - y + 2 = x - 4 + y - z)
  (h6 : x - 4 + y - z = x + z - 7) 
  (h7 : 2 = 2) :
  y = 3 ∧ z = 4 := 
by 
  sorry

end female_students_transfer_l19_19782


namespace sheep_remain_l19_19951

theorem sheep_remain : ∀ (total_sheep sister_share brother_share : ℕ),
  total_sheep = 400 →
  sister_share = total_sheep / 4 →
  brother_share = (total_sheep - sister_share) / 2 →
  (total_sheep - sister_share - brother_share) = 150 :=
by
  intros total_sheep sister_share brother_share h_total h_sister h_brother
  rw [h_total, h_sister, h_brother]
  sorry

end sheep_remain_l19_19951


namespace smallest_four_digit_divisible_by_primes_l19_19321

theorem smallest_four_digit_divisible_by_primes :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ n = 1050 :=
by
  sorry

end smallest_four_digit_divisible_by_primes_l19_19321


namespace car_highway_miles_per_tankful_l19_19795

-- Condition definitions
def city_miles_per_tankful : ℕ := 336
def miles_per_gallon_city : ℕ := 24
def city_to_highway_diff : ℕ := 9

-- Calculation from conditions
def miles_per_gallon_highway : ℕ := miles_per_gallon_city + city_to_highway_diff
def tank_size : ℤ := city_miles_per_tankful / miles_per_gallon_city

-- Desired result
def highway_miles_per_tankful : ℤ := miles_per_gallon_highway * tank_size

-- Proof statement
theorem car_highway_miles_per_tankful :
  highway_miles_per_tankful = 462 := by
  unfold highway_miles_per_tankful
  unfold miles_per_gallon_highway
  unfold tank_size
  -- Sorry here to skip the detailed proof steps
  sorry

end car_highway_miles_per_tankful_l19_19795


namespace proof_problem_l19_19873

variables (Books : Type) (Available : Books -> Prop)

def all_books_available : Prop := ∀ b : Books, Available b
def some_books_not_available : Prop := ∃ b : Books, ¬ Available b
def not_all_books_available : Prop := ¬ all_books_available Books Available

theorem proof_problem (h : ¬ all_books_available Books Available) : 
  some_books_not_available Books Available ∧ not_all_books_available Books Available :=
by 
  sorry

end proof_problem_l19_19873


namespace general_term_of_arithmetic_sequence_l19_19468

theorem general_term_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a3 : a 3 = -2)
  (h_a7 : a 7 = -10) :
  ∀ n : ℕ, a n = 4 - 2 * n :=
sorry

end general_term_of_arithmetic_sequence_l19_19468


namespace third_consecutive_even_number_l19_19721

theorem third_consecutive_even_number (n : ℕ) (h : n % 2 = 0) (sum_eq : n + (n + 2) + (n + 4) = 246) : (n + 4) = 84 :=
by
  -- This statement sets up the conditions and the goal of the proof.
  sorry

end third_consecutive_even_number_l19_19721


namespace inequality_of_factorials_and_polynomials_l19_19788

open Nat

theorem inequality_of_factorials_and_polynomials (m n : ℕ) (hm : m ≥ n) :
  2^n * n! ≤ (m+n)! / (m-n)! ∧ (m+n)! / (m-n)! ≤ (m^2 + m)^n :=
by
  sorry

end inequality_of_factorials_and_polynomials_l19_19788


namespace part1a_part1b_part2_part3_l19_19228

-- Definitions for the sequences in columns ①, ②, and ③
def col1 (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (2 * n - 1)
def col2 (n : ℕ) : ℤ := ((-1 : ℤ) ^ n * (2 * n - 1)) - 2
def col3 (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (2 * n - 1) * 3

-- Problem statements
theorem part1a : col1 10 = 19 :=
sorry

theorem part1b : col2 15 = -31 :=
sorry

theorem part2 : ¬ ∃ n : ℕ, col2 (n - 1) + col2 n + col2 (n + 1) = 1001 :=
sorry

theorem part3 : ∃ k : ℕ, col1 k + col2 k + col3 k = 599 ∧ k = 301 :=
sorry

end part1a_part1b_part2_part3_l19_19228


namespace not_perfect_square_infinitely_many_l19_19640

theorem not_perfect_square_infinitely_many (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_gt : b > a) (h_prime : Prime (b - a)) :
  ∃ᶠ n in at_top, ¬ IsSquare ((a ^ n + a + 1) * (b ^ n + b + 1)) :=
sorry

end not_perfect_square_infinitely_many_l19_19640


namespace hyperbola_midpoint_exists_l19_19365

theorem hyperbola_midpoint_exists :
  ∃ A B : ℝ × ℝ, 
    (A.1 ^ 2 - (A.2 ^ 2) / 9 = 1) ∧
    (B.1 ^ 2 - (B.2 ^ 2) / 9 = 1) ∧
    ((A.1 + B.1) / 2 = -1) ∧ 
    ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_exists_l19_19365


namespace find_p_q_d_l19_19161

def f (p q d : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then p * x + 4
  else if x = 0 then p * q
  else q * x + d

theorem find_p_q_d :
  ∃ p q d : ℕ, f p q d 3 = 7 ∧ f p q d 0 = 6 ∧ f p q d (-3) = -12 ∧ (p + q + d = 13) :=
by
  sorry

end find_p_q_d_l19_19161


namespace x_add_y_add_one_is_composite_l19_19631

theorem x_add_y_add_one_is_composite (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (k : ℕ) (h : x^2 + x * y - y = k^2) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (x + y + 1 = a * b) :=
by
  sorry

end x_add_y_add_one_is_composite_l19_19631


namespace lucas_seq_mod_50_l19_19535

def lucas_seq : ℕ → ℕ
| 0       => 2
| 1       => 5
| (n + 2) => lucas_seq n + lucas_seq (n + 1)

theorem lucas_seq_mod_50 : lucas_seq 49 % 5 = 0 := 
by
  sorry

end lucas_seq_mod_50_l19_19535


namespace tan_theta_eq_neg_2sqrt2_to_expression_l19_19147

theorem tan_theta_eq_neg_2sqrt2_to_expression (θ : ℝ) (h : Real.tan θ = -2 * Real.sqrt 2) :
  (2 * (Real.cos (θ / 2)) ^ 2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) = 1 :=
by
  sorry

end tan_theta_eq_neg_2sqrt2_to_expression_l19_19147


namespace smaller_part_volume_l19_19433

noncomputable def volume_of_smaller_part (a : ℝ) : ℝ :=
  (25 / 144) * (a^3)

theorem smaller_part_volume (a : ℝ) (h_pos : 0 < a) :
  ∃ v : ℝ, v = volume_of_smaller_part a :=
  sorry

end smaller_part_volume_l19_19433


namespace smallest_n_common_factor_l19_19889

theorem smallest_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ (∀ d : ℕ, d > 1 → d ∣ (11 * n - 4) → d ∣ (8 * n - 5)) ∧ n = 15 :=
by {
  -- Define the conditions as given in the problem
  sorry
}

end smallest_n_common_factor_l19_19889


namespace intersection_complement_A_U_B_l19_19252

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def set_A : Set ℕ := {2, 4, 6}
def set_B : Set ℕ := {1, 3, 5, 7}

theorem intersection_complement_A_U_B :
  set_A ∩ (universal_set \ set_B) = {2, 4, 6} :=
by {
  sorry
}

end intersection_complement_A_U_B_l19_19252


namespace length_of_bridge_correct_l19_19748

open Real

noncomputable def length_of_bridge (length_of_train : ℝ) (time_to_cross : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed := speed_kmph * (1000 / 3600)
  let total_distance := speed * time_to_cross
  total_distance - length_of_train

theorem length_of_bridge_correct :
  length_of_bridge 200 34.997200223982084 36 = 149.97200223982084 := by
  sorry

end length_of_bridge_correct_l19_19748


namespace scout_hours_worked_l19_19452

variable (h : ℕ) -- number of hours worked on Saturday
variable (base_pay : ℕ) -- base pay per hour
variable (tip_per_customer : ℕ) -- tip per customer
variable (saturday_customers : ℕ) -- customers served on Saturday
variable (sunday_hours : ℕ) -- hours worked on Sunday
variable (sunday_customers : ℕ) -- customers served on Sunday
variable (total_earnings : ℕ) -- total earnings over the weekend

theorem scout_hours_worked {h : ℕ} (base_pay : ℕ) (tip_per_customer : ℕ) (saturday_customers : ℕ) (sunday_hours : ℕ) (sunday_customers : ℕ) (total_earnings : ℕ) :
  base_pay = 10 → 
  tip_per_customer = 5 → 
  saturday_customers = 5 → 
  sunday_hours = 5 → 
  sunday_customers = 8 → 
  total_earnings = 155 → 
  10 * h + 5 * 5 + 10 * 5 + 5 * 8 = 155 → 
  h = 4 :=
by
  intros
  sorry

end scout_hours_worked_l19_19452


namespace gcd_of_repeated_three_digit_integers_is_1001001_l19_19356

theorem gcd_of_repeated_three_digit_integers_is_1001001 :
  ∀ (n : ℕ), (100 ≤ n ∧ n <= 999) →
  ∃ d : ℕ, d = 1001001 ∧
    (∀ m : ℕ, m = n * 1001001 →
      ∃ k : ℕ, m = k * d) :=
by
  sorry

end gcd_of_repeated_three_digit_integers_is_1001001_l19_19356


namespace solve_eq_l19_19357

theorem solve_eq :
  { x : ℝ | (14 * x - x^2) / (x + 2) * (x + (14 - x) / (x + 2)) = 48 } =
  {4, (1 + Real.sqrt 193) / 2, (1 - Real.sqrt 193) / 2} :=
by
  sorry

end solve_eq_l19_19357


namespace trader_profit_percentage_l19_19698

theorem trader_profit_percentage (P : ℝ) (hP : 0 < P) :
  let bought_price := 0.90 * P
  let sold_price := 1.80 * bought_price
  let profit := sold_price - P
  let profit_percentage := (profit / P) * 100
  profit_percentage = 62 := 
by
  let bought_price := 0.90 * P
  let sold_price := 1.80 * bought_price
  let profit := sold_price - P
  let profit_percentage := (profit / P) * 100
  sorry

end trader_profit_percentage_l19_19698


namespace abdul_largest_number_l19_19097

theorem abdul_largest_number {a b c d : ℕ} 
  (h1 : a + (b + c + d) / 3 = 17)
  (h2 : b + (a + c + d) / 3 = 21)
  (h3 : c + (a + b + d) / 3 = 23)
  (h4 : d + (a + b + c) / 3 = 29) :
  d = 21 :=
by sorry

end abdul_largest_number_l19_19097


namespace triangle_area_is_correct_l19_19789

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_is_correct : 
  area_of_triangle (1, 3) (5, -2) (8, 6) = 23.5 := 
by
  sorry

end triangle_area_is_correct_l19_19789


namespace overall_sale_price_per_kg_l19_19163

-- Defining the quantities and prices
def tea_A_quantity : ℝ := 80
def tea_A_cost_per_kg : ℝ := 15
def tea_B_quantity : ℝ := 20
def tea_B_cost_per_kg : ℝ := 20
def tea_C_quantity : ℝ := 50
def tea_C_cost_per_kg : ℝ := 25
def tea_D_quantity : ℝ := 40
def tea_D_cost_per_kg : ℝ := 30

-- Defining the profit percentages
def tea_A_profit_percentage : ℝ := 0.30
def tea_B_profit_percentage : ℝ := 0.25
def tea_C_profit_percentage : ℝ := 0.20
def tea_D_profit_percentage : ℝ := 0.15

-- Desired sale price per kg
theorem overall_sale_price_per_kg : 
  (tea_A_quantity * tea_A_cost_per_kg * (1 + tea_A_profit_percentage) +
   tea_B_quantity * tea_B_cost_per_kg * (1 + tea_B_profit_percentage) +
   tea_C_quantity * tea_C_cost_per_kg * (1 + tea_C_profit_percentage) +
   tea_D_quantity * tea_D_cost_per_kg * (1 + tea_D_profit_percentage)) / 
  (tea_A_quantity + tea_B_quantity + tea_C_quantity + tea_D_quantity) = 26 := 
by
  sorry

end overall_sale_price_per_kg_l19_19163


namespace badminton_members_count_l19_19123

-- Definitions of the conditions
def total_members : ℕ := 40
def tennis_players : ℕ := 18
def neither_sport : ℕ := 5
def both_sports : ℕ := 3
def badminton_players : ℕ := 20 -- The answer we need to prove

-- The proof statement
theorem badminton_members_count :
  total_members = (badminton_players + tennis_players - both_sports) + neither_sport :=
by
  -- The proof is outlined here
  sorry

end badminton_members_count_l19_19123


namespace triangle_area_l19_19140

/-- Define the area of a triangle with one side of length 13, an opposite angle of 60 degrees, and side ratio 4:3. -/
theorem triangle_area (a b c : ℝ) (A : ℝ) (S : ℝ) 
  (h_a : a = 13)
  (h_A : A = Real.pi / 3)
  (h_bc_ratio : b / c = 4 / 3)
  (h_cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h_area : S = 1 / 2 * b * c * Real.sin A) :
  S = 39 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l19_19140


namespace remainder_317_l19_19902

theorem remainder_317 (y : ℤ)
  (h1 : 4 + y ≡ 9 [ZMOD 16])
  (h2 : 6 + y ≡ 8 [ZMOD 81])
  (h3 : 8 + y ≡ 49 [ZMOD 625]) :
  y ≡ 317 [ZMOD 360] := 
sorry

end remainder_317_l19_19902


namespace find_triangle_l19_19531

theorem find_triangle (q : ℝ) (triangle : ℝ) (h1 : 3 * triangle * q = 63) (h2 : 7 * (triangle + q) = 161) : triangle = 1 :=
sorry

end find_triangle_l19_19531


namespace chess_tournament_winner_l19_19996

theorem chess_tournament_winner :
  ∀ (x : ℕ) (P₉ P₁₀ : ℕ),
  (x > 0) →
  (9 * x) = 4 * P₃ →
  P₉ = (x * (x - 1)) / 2 + 9 * x^2 →
  P₁₀ = (9 * x * (9 * x - 1)) / 2 →
  (9 * x^2 - x) * 2 ≥ 81 * x^2 - 9 * x →
  x = 1 →
  P₃ = 9 :=
by
  sorry

end chess_tournament_winner_l19_19996


namespace find_ages_l19_19520

theorem find_ages (J sister cousin : ℝ)
  (h1 : J + 9 = 3 * (J - 11))
  (h2 : sister = 2 * J)
  (h3 : cousin = (J + sister) / 2) :
  J = 21 ∧ sister = 42 ∧ cousin = 31.5 :=
by
  sorry

end find_ages_l19_19520


namespace speed_of_current_l19_19949

theorem speed_of_current (h_start: ∀ t: ℝ, t ≥ 0 → u ≥ 0) 
  (boat1_turn_2pm: ∀ t: ℝ, t >= 1 → t < 2 → boat1_turn_13_14) 
  (boat2_turn_3pm: ∀ t: ℝ, t >= 2 → t < 3 → boat2_turn_14_15) 
  (boats_meet: ∀ x: ℝ, x = 7.5) :
  v = 2.5 := 
sorry

end speed_of_current_l19_19949


namespace gcd_360_504_l19_19064

theorem gcd_360_504 : Nat.gcd 360 504 = 72 :=
by sorry

end gcd_360_504_l19_19064


namespace pyramid_height_is_6_l19_19523

-- Define the conditions for the problem
def square_side_length : ℝ := 18
def pyramid_base_side_length (s : ℝ) : Prop := s * s = (square_side_length / 2) * (square_side_length / 2)
def pyramid_slant_height (s l : ℝ) : Prop := 2 * s * l = square_side_length * square_side_length

-- State the main theorem
theorem pyramid_height_is_6 (s l h : ℝ) (hs : pyramid_base_side_length s) (hl : pyramid_slant_height s l) : h = 6 := 
sorry

end pyramid_height_is_6_l19_19523


namespace angie_pretzels_l19_19888

theorem angie_pretzels (Barry_Shelly: ℕ) (Shelly_Angie: ℕ) :
  (Barry_Shelly = 12 / 2) → (Shelly_Angie = 3 * Barry_Shelly) → (Barry_Shelly = 6) → (Shelly_Angie = 18) :=
by
  intro h1 h2 h3
  sorry

end angie_pretzels_l19_19888


namespace p_work_alone_time_l19_19422

variable (Wp Wq : ℝ)
variable (x : ℝ)

-- Conditions
axiom h1 : Wp = 1.5 * Wq
axiom h2 : (1 / x) + (Wq / Wp) * (1 / x) = 1 / 15

-- Proof of the question (p alone can complete the work in x days)
theorem p_work_alone_time : x = 25 :=
by
  -- Add your proof here
  sorry

end p_work_alone_time_l19_19422


namespace beavers_swimming_correct_l19_19764

variable (initial_beavers remaining_beavers beavers_swimming : ℕ)

def beavers_problem : Prop :=
  initial_beavers = 2 ∧
  remaining_beavers = 1 ∧
  beavers_swimming = initial_beavers - remaining_beavers

theorem beavers_swimming_correct :
  beavers_problem initial_beavers remaining_beavers beavers_swimming → beavers_swimming = 1 :=
by
  sorry

end beavers_swimming_correct_l19_19764


namespace no_first_quadrant_l19_19490

theorem no_first_quadrant (a b : ℝ) (h_a : a < 0) (h_b : b < 0) (h_am : (a - b) < 0) :
  ¬∃ x : ℝ, (a - b) * x + b > 0 ∧ x > 0 :=
sorry

end no_first_quadrant_l19_19490


namespace pyramid_height_is_correct_l19_19498

noncomputable def pyramid_height (perimeter : ℝ) (apex_distance : ℝ) : ℝ :=
  let side_length := perimeter / 4
  let half_diagonal := side_length * Real.sqrt 2 / 2
  Real.sqrt (apex_distance ^ 2 - half_diagonal ^ 2)

theorem pyramid_height_is_correct :
  pyramid_height 40 15 = 5 * Real.sqrt 7 :=
by
  sorry

end pyramid_height_is_correct_l19_19498


namespace factorization_correctness_l19_19009

theorem factorization_correctness :
  (∀ x, x^2 + 2 * x + 1 = (x + 1)^2) ∧
  ¬ (∀ x, x * (x + 1) = x^2 + x) ∧
  ¬ (∀ x y, x^2 + x * y - 3 = x * (x + y) - 3) ∧
  ¬ (∀ x, x^2 + 6 * x + 4 = (x + 3)^2 - 5) :=
by
  sorry

end factorization_correctness_l19_19009


namespace line_tangent_to_circle_l19_19460

open Real

theorem line_tangent_to_circle :
    ∃ (x y : ℝ), (3 * x - 4 * y - 5 = 0) ∧ ((x - 1)^2 + (y + 3)^2 - 4 = 0) ∧ 
    (∃ (t r : ℝ), (t = 0 ∧ r ≠ 0) ∧ 
     (3 * t - 4 * (r + t * 3 / 4) - 5 = 0) ∧ ((r + t * 3 / 4 - 1)^2 + (3 * (-1) + t - 3)^2 = 0)) 
  :=
sorry

end line_tangent_to_circle_l19_19460


namespace day_after_75_days_l19_19256

theorem day_after_75_days (day_of_week : ℕ → String) (h : day_of_week 0 = "Tuesday") :
  day_of_week 75 = "Sunday" :=
sorry

end day_after_75_days_l19_19256


namespace compare_abc_l19_19334

noncomputable def a : ℝ := 2 + (1 / 5) * Real.log 2
noncomputable def b : ℝ := 1 + Real.exp (0.2 * Real.log 2)
noncomputable def c : ℝ := Real.exp (1.1 * Real.log 2)

theorem compare_abc : a < c ∧ c < b := by
  sorry

end compare_abc_l19_19334


namespace camille_saw_31_birds_l19_19239

def num_cardinals : ℕ := 3
def num_robins : ℕ := 4 * num_cardinals
def num_blue_jays : ℕ := 2 * num_cardinals
def num_sparrows : ℕ := 3 * num_cardinals + 1
def total_birds : ℕ := num_cardinals + num_robins + num_blue_jays + num_sparrows

theorem camille_saw_31_birds : total_birds = 31 := by
  sorry

end camille_saw_31_birds_l19_19239


namespace syllogism_major_minor_premise_l19_19325

theorem syllogism_major_minor_premise
(people_of_Yaan_strong_unyielding : Prop)
(people_of_Yaan_Chinese : Prop)
(all_Chinese_strong_unyielding : Prop) :
  all_Chinese_strong_unyielding ∧ people_of_Yaan_Chinese → (all_Chinese_strong_unyielding = all_Chinese_strong_unyielding ∧ people_of_Yaan_Chinese = people_of_Yaan_Chinese) :=
by
  intros h
  exact ⟨rfl, rfl⟩

end syllogism_major_minor_premise_l19_19325


namespace triangle_angle_condition_l19_19525

theorem triangle_angle_condition (a b h_3 : ℝ) (A C : ℝ) 
  (h : 1/(h_3^2) = 1/(a^2) + 1/(b^2)) :
  C = 90 ∨ |A - C| = 90 := 
sorry

end triangle_angle_condition_l19_19525


namespace general_term_sequence_l19_19272

-- Definition of the sequence conditions
def seq (n : ℕ) : ℤ :=
  (-1)^(n+1) * (2*n + 1)

-- The main statement to be proved
theorem general_term_sequence (n : ℕ) : seq n = (-1)^(n+1) * (2 * n + 1) :=
sorry

end general_term_sequence_l19_19272


namespace metallic_sheet_width_l19_19707

-- Defining the conditions
def sheet_length := 48
def cut_square_side := 8
def box_volume := 5632

-- Main theorem statement
theorem metallic_sheet_width 
    (L : ℕ := sheet_length)
    (s : ℕ := cut_square_side)
    (V : ℕ := box_volume) :
    (32 * (w - 2 * s) * s = V) → (w = 38) := by
  intros h1
  sorry

end metallic_sheet_width_l19_19707


namespace find_height_l19_19431

-- Definitions from the problem conditions
def Area : ℕ := 442
def width : ℕ := 7
def length : ℕ := 8

-- The statement to prove
theorem find_height (h : ℕ) (H : 2 * length * width + 2 * length * h + 2 * width * h = Area) : h = 11 := 
by
  sorry

end find_height_l19_19431


namespace find_A_time_l19_19704

noncomputable def work_rate_equations (W : ℝ) (A B C : ℝ) : Prop :=
  B + C = W / 2 ∧ A + B = W / 2 ∧ C = W / 3

theorem find_A_time {W A B C : ℝ} (h : work_rate_equations W A B C) :
  W / A = 3 :=
sorry

end find_A_time_l19_19704


namespace solve_fractional_eq_l19_19291

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) : (x / (x + 1) - 1 = 3 / (x - 1)) → x = -1 / 2 :=
by
  sorry

end solve_fractional_eq_l19_19291


namespace tickets_bought_l19_19506

theorem tickets_bought
  (olivia_money : ℕ) (nigel_money : ℕ) (ticket_cost : ℕ) (leftover_money : ℕ)
  (total_money : ℕ) (money_spent : ℕ) 
  (h1 : olivia_money = 112) 
  (h2 : nigel_money = 139) 
  (h3 : ticket_cost = 28) 
  (h4 : leftover_money = 83)
  (h5 : total_money = olivia_money + nigel_money)
  (h6 : total_money = 251)
  (h7 : money_spent = total_money - leftover_money)
  (h8 : money_spent = 168)
  : money_spent / ticket_cost = 6 := 
by
  sorry

end tickets_bought_l19_19506


namespace sin_A_is_eight_ninths_l19_19554

variable (AB AC : ℝ) (A : ℝ)

-- Given conditions
def area_triangle := 1 / 2 * AB * AC * Real.sin A = 100
def geometric_mean := Real.sqrt (AB * AC) = 15

-- Proof statement
theorem sin_A_is_eight_ninths (h1 : area_triangle AB AC A) (h2 : geometric_mean AB AC) :
  Real.sin A = 8 / 9 := sorry

end sin_A_is_eight_ninths_l19_19554


namespace alice_numbers_l19_19027

theorem alice_numbers (x y : ℝ) (h1 : x * y = 12) (h2 : x + y = 7) : (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) :=
by
  sorry

end alice_numbers_l19_19027


namespace nathaniel_tickets_l19_19115

theorem nathaniel_tickets :
  ∀ (B S : ℕ),
  (7 * B + 4 * S + 11 = 128) →
  (B + S = 20) :=
by
  intros B S h
  sorry

end nathaniel_tickets_l19_19115


namespace find_a10_l19_19238

def seq (a : ℕ → ℝ) : Prop :=
∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p + a q

theorem find_a10 (a : ℕ → ℝ) (h_seq : seq a) (h_a2 : a 2 = -6) : a 10 = -30 :=
by
  sorry

end find_a10_l19_19238


namespace journeymen_percentage_after_layoff_l19_19662

noncomputable def total_employees : ℝ := 20210
noncomputable def fraction_journeymen : ℝ := 2 / 7
noncomputable def total_journeymen : ℝ := total_employees * fraction_journeymen
noncomputable def laid_off_journeymen : ℝ := total_journeymen / 2
noncomputable def remaining_journeymen : ℝ := total_journeymen / 2
noncomputable def remaining_employees : ℝ := total_employees - laid_off_journeymen
noncomputable def journeymen_percentage : ℝ := (remaining_journeymen / remaining_employees) * 100

theorem journeymen_percentage_after_layoff : journeymen_percentage = 16.62 := by
  sorry

end journeymen_percentage_after_layoff_l19_19662


namespace solve_inequality_l19_19740

theorem solve_inequality : {x : ℝ | |x - 2| * (x - 1) < 2} = {x : ℝ | x < 3} :=
by
  sorry

end solve_inequality_l19_19740


namespace total_balls_in_box_l19_19714

theorem total_balls_in_box (red blue yellow total : ℕ) 
  (h1 : 2 * blue = 3 * red)
  (h2 : 3 * yellow = 4 * red) 
  (h3 : yellow = 40)
  (h4 : red + blue + yellow = total) : total = 90 :=
sorry

end total_balls_in_box_l19_19714


namespace compute_value_l19_19322

theorem compute_value (a b c : ℕ) (h : a = 262 ∧ b = 258 ∧ c = 150) : 
  (a^2 - b^2) + c = 2230 := 
by
  sorry

end compute_value_l19_19322


namespace joe_speed_l19_19380

theorem joe_speed (P : ℝ) (J : ℝ) (h1 : J = 2 * P) (h2 : 2 * P * (2 / 3) + P * (2 / 3) = 16) : J = 16 := 
by
  sorry

end joe_speed_l19_19380


namespace data_transmission_time_l19_19225

def chunks_per_block : ℕ := 1024
def blocks : ℕ := 30
def transmission_rate : ℕ := 256
def seconds_in_minute : ℕ := 60

theorem data_transmission_time :
  (blocks * chunks_per_block) / transmission_rate / seconds_in_minute = 2 :=
by
  sorry

end data_transmission_time_l19_19225


namespace rectangular_solid_surface_area_l19_19983

theorem rectangular_solid_surface_area
  (a b c : ℕ)
  (h_prime_a : Prime a)
  (h_prime_b : Prime b)
  (h_prime_c : Prime c)
  (h_volume : a * b * c = 143) :
  2 * (a * b + b * c + c * a) = 382 := by
  sorry

end rectangular_solid_surface_area_l19_19983


namespace largest_integer_y_l19_19880

theorem largest_integer_y (y : ℤ) : (y / 4 + 3 / 7 : ℝ) < 9 / 4 → y ≤ 7 := by
  intros h
  sorry -- Proof needed

end largest_integer_y_l19_19880


namespace sam_memorized_digits_l19_19153

theorem sam_memorized_digits (c s m : ℕ) 
  (h1 : s = c + 6) 
  (h2 : m = 6 * c)
  (h3 : m = 24) : 
  s = 10 :=
by
  sorry

end sam_memorized_digits_l19_19153


namespace exist_circle_tangent_to_three_circles_l19_19590

variable (h1 k1 r1 h2 k2 r2 h3 k3 r3 h k r : ℝ)

def condition1 : Prop := (h - h1)^2 + (k - k1)^2 = (r + r1)^2
def condition2 : Prop := (h - h2)^2 + (k - k2)^2 = (r + r2)^2
def condition3 : Prop := (h - h3)^2 + (k - k3)^2 = (r + r3)^2

theorem exist_circle_tangent_to_three_circles : 
  ∃ (h k r : ℝ), condition1 h1 k1 r1 h k r ∧ condition2 h2 k2 r2 h k r ∧ condition3 h3 k3 r3 h k r :=
by
  sorry

end exist_circle_tangent_to_three_circles_l19_19590


namespace fraction_sum_identity_l19_19676

variable (a b c : ℝ)

theorem fraction_sum_identity (h1 : a + b + c = 0) (h2 : a / b + b / c + c / a = 100) : 
  b / a + c / b + a / c = -103 :=
by {
  -- Proof goes here
  sorry
}

end fraction_sum_identity_l19_19676


namespace quadratic_eq_positive_integer_roots_l19_19055

theorem quadratic_eq_positive_integer_roots (k p : ℕ) 
  (h1 : k > 0)
  (h2 : ∃ x1 x2 : ℕ, x1 > 0 ∧ x2 > 0 ∧ (k-1) * x1^2 - p * x1 + k = 0 ∧ (k-1) * x2^2 - p * x2 + k = 0) :
  k ^ (k * p) * (p ^ p + k ^ k) + (p + k) = 1989 :=
by
  sorry

end quadratic_eq_positive_integer_roots_l19_19055


namespace sum_of_ages_l19_19002

-- Definitions based on conditions
variables (J S : ℝ) -- J and S are real numbers

-- First condition: Jane is five years older than Sarah
def jane_older_than_sarah := J = S + 5

-- Second condition: Nine years from now, Jane will be three times as old as Sarah was three years ago
def future_condition := J + 9 = 3 * (S - 3)

-- Conclusion to prove
theorem sum_of_ages (h1 : jane_older_than_sarah J S) (h2 : future_condition J S) : J + S = 28 :=
by
  sorry

end sum_of_ages_l19_19002


namespace mod_inverse_identity_l19_19797

theorem mod_inverse_identity : 
  (1 / 5 + 1 / 5^2) % 31 = 26 :=
by
  sorry

end mod_inverse_identity_l19_19797


namespace find_cement_used_lexi_l19_19618

def cement_used_total : ℝ := 15.1
def cement_used_tess : ℝ := 5.1
def cement_used_lexi : ℝ := cement_used_total - cement_used_tess

theorem find_cement_used_lexi : cement_used_lexi = 10 := by
  sorry

end find_cement_used_lexi_l19_19618


namespace go_game_prob_l19_19033

theorem go_game_prob :
  ∀ (pA pB : ℝ),
    (pA = 0.6) →
    (pB = 0.4) →
    ((pA ^ 2) + (pB ^ 2) = 0.52) :=
by
  intros pA pB hA hB
  rw [hA, hB]
  sorry

end go_game_prob_l19_19033


namespace simplify_expression_l19_19396

variable (a b : ℤ)

theorem simplify_expression :
  (30 * a + 45 * b) + (15 * a + 40 * b) - (20 * a + 55 * b) + (5 * a - 10 * b) = 30 * a + 20 * b :=
by
  sorry

end simplify_expression_l19_19396


namespace solve_quadratic_eq_l19_19078

theorem solve_quadratic_eq (x : ℝ) (h : x > 0) (eq : 4 * x^2 + 8 * x - 20 = 0) : 
  x = Real.sqrt 6 - 1 :=
sorry

end solve_quadratic_eq_l19_19078


namespace max_popsicles_l19_19534

theorem max_popsicles (total_money : ℝ) (cost_per_popsicle : ℝ) (h_money : total_money = 19.23) (h_cost : cost_per_popsicle = 1.60) : 
  ∃ (x : ℕ), x = ⌊total_money / cost_per_popsicle⌋ ∧ x = 12 :=
by
    sorry

end max_popsicles_l19_19534


namespace base_k_to_decimal_l19_19088

theorem base_k_to_decimal (k : ℕ) (h : 0 < k ∧ k < 10) : 
  1 * k^2 + 7 * k + 5 = 125 → k = 8 := 
by
  sorry

end base_k_to_decimal_l19_19088


namespace average_age_of_John_Mary_Tonya_is_35_l19_19546

-- Define the ages of the individuals
variable (John Mary Tonya : ℕ)

-- Conditions given in the problem
def John_is_twice_as_old_as_Mary : Prop := John = 2 * Mary
def John_is_half_as_old_as_Tonya : Prop := John = Tonya / 2
def Tonya_is_60 : Prop := Tonya = 60

-- The average age calculation
def average_age (a b c : ℕ) : ℕ := (a + b + c) / 3

-- The statement we need to prove
theorem average_age_of_John_Mary_Tonya_is_35 :
  John_is_twice_as_old_as_Mary John Mary →
  John_is_half_as_old_as_Tonya John Tonya →
  Tonya_is_60 Tonya →
  average_age John Mary Tonya = 35 :=
by
  sorry

end average_age_of_John_Mary_Tonya_is_35_l19_19546


namespace least_possible_value_of_x_minus_y_plus_z_l19_19950

theorem least_possible_value_of_x_minus_y_plus_z : 
  ∃ (x y z : ℕ), 3 * x = 4 * y ∧ 4 * y = 7 * z ∧ x - y + z = 19 :=
by
  sorry

end least_possible_value_of_x_minus_y_plus_z_l19_19950


namespace sqrt5_times_sqrt6_minus_1_over_sqrt5_bound_l19_19542

theorem sqrt5_times_sqrt6_minus_1_over_sqrt5_bound :
  4 < (Real.sqrt 5) * ((Real.sqrt 6) - 1 / (Real.sqrt 5)) ∧ (Real.sqrt 5) * ((Real.sqrt 6) - 1 / (Real.sqrt 5)) < 5 :=
by
  sorry

end sqrt5_times_sqrt6_minus_1_over_sqrt5_bound_l19_19542


namespace general_formula_and_arithmetic_sequence_l19_19992

noncomputable def S_n (n : ℕ) : ℕ := 3 * n ^ 2 - 2 * n
noncomputable def a_n (n : ℕ) : ℕ := S_n n - S_n (n - 1)

theorem general_formula_and_arithmetic_sequence :
  (∀ n : ℕ, a_n n = 6 * n - 5) ∧
  (∀ n : ℕ, (n ≥ 2 → a_n n - a_n (n - 1) = 6) ∧ (a_n 1 = 1)) :=
by
  sorry

end general_formula_and_arithmetic_sequence_l19_19992


namespace bob_homework_time_l19_19591

variable (T_Alice T_Bob : ℕ)

theorem bob_homework_time (h_Alice : T_Alice = 40) (h_Bob : T_Bob = (3 * T_Alice) / 8) : T_Bob = 15 :=
by
  rw [h_Alice] at h_Bob
  norm_num at h_Bob
  exact h_Bob

-- Assuming T_Alice represents the time taken by Alice to complete her homework
-- and T_Bob represents the time taken by Bob to complete his homework,
-- we prove that T_Bob is 15 minutes given the conditions.

end bob_homework_time_l19_19591


namespace ceil_sqrt_196_eq_14_l19_19831

theorem ceil_sqrt_196_eq_14 : ⌈Real.sqrt 196⌉ = 14 := 
by 
  sorry

end ceil_sqrt_196_eq_14_l19_19831


namespace male_teacher_classes_per_month_l19_19816

theorem male_teacher_classes_per_month (x y a : ℕ) :
  (15 * x = 6 * (x + y)) ∧ (a * y = 6 * (x + y)) → a = 10 :=
by
  sorry

end male_teacher_classes_per_month_l19_19816


namespace radio_price_position_l19_19183

theorem radio_price_position (n : ℕ) (h₁ : n = 42)
  (h₂ : ∃ m : ℕ, m = 18 ∧ 
    (∀ k : ℕ, k < m → (∃ x : ℕ, x > k))) : 
    ∃ m : ℕ, m = 24 :=
by
  sorry

end radio_price_position_l19_19183


namespace probability_of_selecting_A_l19_19068

noncomputable def total_students : ℕ := 4
noncomputable def selected_student_A : ℕ := 1

theorem probability_of_selecting_A : 
  (selected_student_A : ℝ) / (total_students : ℝ) = 1 / 4 :=
by
  sorry

end probability_of_selecting_A_l19_19068


namespace sample_size_is_15_l19_19848

-- Define the given conditions as constants and assumptions within the Lean environment.
def total_employees := 750
def young_workers := 350
def middle_aged_workers := 250
def elderly_workers := 150
def sample_young_workers := 7

-- Define the proposition that given these conditions, the sample size is 15.
theorem sample_size_is_15 : ∃ n : ℕ, (7 / n = 350 / 750) ∧ n = 15 := by
  sorry

end sample_size_is_15_l19_19848


namespace exists_collinear_B_points_l19_19480

noncomputable def intersection (A B C D : Point) : Point :=
sorry

noncomputable def collinearity (P Q R S T : Point) : Prop :=
sorry

def convex_pentagon (A1 A2 A3 A4 A5 : Point) : Prop :=
-- Condition ensuring A1, A2, A3, A4, A5 form a convex pentagon, to be precisely defined
sorry

theorem exists_collinear_B_points :
  ∃ (A1 A2 A3 A4 A5 : Point),
    convex_pentagon A1 A2 A3 A4 A5 ∧
    collinearity
      (intersection A1 A4 A2 A3)
      (intersection A2 A5 A3 A4)
      (intersection A3 A1 A4 A5)
      (intersection A4 A2 A5 A1)
      (intersection A5 A3 A1 A2) :=
sorry

end exists_collinear_B_points_l19_19480


namespace sum_first_32_terms_bn_l19_19934

noncomputable def a_n (n : ℕ) : ℝ := 3 * n + 1

noncomputable def b_n (n : ℕ) : ℝ :=
  1 / ((a_n n) * Real.sqrt (a_n (n + 1)) + (a_n (n + 1)) * Real.sqrt (a_n n))

noncomputable def sum_bn (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) b_n

theorem sum_first_32_terms_bn : sum_bn 32 = 2 / 15 := 
sorry

end sum_first_32_terms_bn_l19_19934


namespace polynomial_identity_l19_19612

theorem polynomial_identity (x y : ℝ) (h : x - y = 1) : 
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 := 
  sorry

end polynomial_identity_l19_19612


namespace n_fifth_plus_4n_mod_5_l19_19218

theorem n_fifth_plus_4n_mod_5 (n : ℕ) : (n^5 + 4 * n) % 5 = 0 := 
by
  sorry

end n_fifth_plus_4n_mod_5_l19_19218


namespace find_x_l19_19646

theorem find_x 
  (x : ℕ)
  (h : 3^x = 3^(20) * 3^(20) * 3^(18) + 3^(19) * 3^(20) * 3^(19) + 3^(18) * 3^(21) * 3^(19)) :
  x = 59 :=
sorry

end find_x_l19_19646


namespace problem_rewrite_equation_l19_19285

theorem problem_rewrite_equation :
  ∃ a b c : ℤ, a > 0 ∧ (64*(x^2) + 96*x - 81 = 0) → ((a*x + b)^2 = c) ∧ (a + b + c = 131) :=
sorry

end problem_rewrite_equation_l19_19285


namespace trigonometric_identity_solution_l19_19903

open Real

noncomputable def x_sol1 (k : ℤ) : ℝ := (π / 2) * (4 * k - 1)
noncomputable def x_sol2 (l : ℤ) : ℝ := (π / 3) * (6 * l + 1)
noncomputable def x_sol2_neg (l : ℤ) : ℝ := (π / 3) * (6 * l - 1)

theorem trigonometric_identity_solution (x : ℝ) :
    (3 * sin (x / 2) ^ 2 * cos (3 * π / 2 + x / 2) +
    3 * sin (x / 2) ^ 2 * cos (x / 2) -
    sin (x / 2) * cos (x / 2) ^ 2 =
    sin (π / 2 + x / 2) ^ 2 * cos (x / 2)) →
    (∃ k : ℤ, x = x_sol1 k) ∨
    (∃ l : ℤ, x = x_sol2 l ∨ x = x_sol2_neg l) :=
by
  sorry

end trigonometric_identity_solution_l19_19903


namespace find_T_b_plus_T_neg_b_l19_19985

noncomputable def T (r : ℝ) : ℝ := 15 / (1 - r)

theorem find_T_b_plus_T_neg_b (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 3600) :
  T b + T (-b) = 480 :=
sorry

end find_T_b_plus_T_neg_b_l19_19985


namespace exists_n_for_pow_lt_e_l19_19741

theorem exists_n_for_pow_lt_e {p e : ℝ} (hp : 0 < p ∧ p < 1) (he : 0 < e) :
  ∃ n : ℕ, (1 - p) ^ n < e :=
sorry

end exists_n_for_pow_lt_e_l19_19741


namespace exists_zero_point_in_interval_l19_19373

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x - 2 * x

theorem exists_zero_point_in_interval :
  ∃ c ∈ Set.Ioo 1 (Real.pi / 2), f c = 0 := 
sorry

end exists_zero_point_in_interval_l19_19373


namespace alex_jellybeans_l19_19743

theorem alex_jellybeans (n : ℕ) (h1 : n ≥ 200) (h2 : n % 17 = 15) : n = 202 :=
sorry

end alex_jellybeans_l19_19743


namespace value_of_f_at_minus_point_two_l19_19632

noncomputable def f (x : ℝ) : ℝ := 1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem value_of_f_at_minus_point_two : f (-0.2) = 0.81873 :=
by {
  sorry
}

end value_of_f_at_minus_point_two_l19_19632


namespace problem1_problem2_l19_19770

-- Define the conditions: f is an odd and decreasing function on [-1, 1]
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_decreasing : ∀ x y, x ≤ y → f y ≤ f x)

-- The domain of interest is [-1, 1]
variable (x1 x2 : ℝ)
variable (h_x1 : x1 ∈ Set.Icc (-1 : ℝ) 1)
variable (h_x2 : x2 ∈ Set.Icc (-1 : ℝ) 1)

-- Proof Problem 1
theorem problem1 : (f x1 + f x2) * (x1 + x2) ≤ 0 := by
  sorry

-- Assume condition for Problem 2
variable (a : ℝ)
variable (h_ineq : f (1 - a) + f (1 - a ^ 2) < 0)
variable (h_dom : ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → x ∈ Set.Icc (-1 : ℝ) 1)

-- Proof Problem 2
theorem problem2 : 0 < a ∧ a < 1 := by
  sorry

end problem1_problem2_l19_19770


namespace problem_statement_l19_19802

theorem problem_statement (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 5) : 
  -a - m * c * d - b = -5 ∨ -a - m * c * d - b = 5 := 
  sorry

end problem_statement_l19_19802


namespace cone_base_area_l19_19540

theorem cone_base_area (r l : ℝ) (h1 : (1/2) * π * l^2 = 2 * π) (h2 : 2 * π * r = 2 * π) :
  π * r^2 = π :=
by 
  sorry

end cone_base_area_l19_19540


namespace logan_gas_expense_l19_19701

-- Definitions based on conditions:
def annual_salary := 65000
def rent_expense := 20000
def grocery_expense := 5000
def desired_savings := 42000
def new_income_target := annual_salary + 10000

-- The property to be proved:
theorem logan_gas_expense : 
  ∀ (gas_expense : ℕ), 
  new_income_target - desired_savings = rent_expense + grocery_expense + gas_expense → 
  gas_expense = 8000 := 
by 
  sorry

end logan_gas_expense_l19_19701


namespace mary_balloons_correct_l19_19669

-- Define the number of black balloons Nancy has
def nancy_balloons : ℕ := 7

-- Define the multiplier that represents how many times more balloons Mary has compared to Nancy
def multiplier : ℕ := 4

-- Define the number of black balloons Mary has in terms of Nancy's balloons and the multiplier
def mary_balloons : ℕ := nancy_balloons * multiplier

-- The statement we want to prove
theorem mary_balloons_correct : mary_balloons = 28 :=
by
  sorry

end mary_balloons_correct_l19_19669


namespace alex_has_more_pens_than_jane_l19_19884

-- Definitions based on the conditions
def starting_pens_alex : ℕ := 4
def pens_jane_after_month : ℕ := 16

-- Alex's pen count after each week
def pens_alex_after_week (w : ℕ) : ℕ :=
  starting_pens_alex * 2 ^ w

-- Proof statement
theorem alex_has_more_pens_than_jane :
  pens_alex_after_week 4 - pens_jane_after_month = 16 := by
  sorry

end alex_has_more_pens_than_jane_l19_19884


namespace flammable_ice_storage_capacity_l19_19560

theorem flammable_ice_storage_capacity (billion : ℕ) (h : billion = 10^9) : (800 * billion = 8 * 10^11) :=
by
  sorry

end flammable_ice_storage_capacity_l19_19560


namespace sequence_conjecture_l19_19020

theorem sequence_conjecture (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n / (a n + 1)) :
  ∀ n : ℕ, 0 < n → a n = 1 / n := by
  sorry

end sequence_conjecture_l19_19020


namespace allan_balloons_count_l19_19113

-- Definition of the conditions
def Total_balloons : ℕ := 3
def Jake_balloons : ℕ := 1

-- The theorem that corresponds to the problem statement
theorem allan_balloons_count (Allan_balloons : ℕ) (h : Allan_balloons + Jake_balloons = Total_balloons) : Allan_balloons = 2 := 
by
  sorry

end allan_balloons_count_l19_19113


namespace simplify_expression_frac_l19_19620

theorem simplify_expression_frac (a b k : ℤ) (h : (6*k + 12) / 6 = a * k + b) : a = 1 ∧ b = 2 → a / b = 1 / 2 := by
  sorry

end simplify_expression_frac_l19_19620


namespace minimum_number_is_correct_l19_19389

-- Define the operations and conditions on the digits
def transform (n : ℕ) : ℕ :=
if 2 ≤ n then n - 2 + 1 else n

noncomputable def minimum_transformed_number (l : List ℕ) : List ℕ :=
l.map transform

def initial_number : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def expected_number : List ℕ := [1, 0, 1, 0, 1, 0, 1, 0, 1]

theorem minimum_number_is_correct :
  minimum_transformed_number initial_number = expected_number := 
by
  -- sorry is a placeholder for the proof
  sorry

end minimum_number_is_correct_l19_19389


namespace VivianMailApril_l19_19685

variable (piecesMailApril piecesMailMay piecesMailJune piecesMailJuly piecesMailAugust : ℕ)

-- Conditions
def condition_double_monthly (a b : ℕ) : Prop := b = 2 * a

axiom May : piecesMailMay = 10
axiom June : piecesMailJune = 20
axiom July : piecesMailJuly = 40
axiom August : piecesMailAugust = 80

axiom patternMay : condition_double_monthly piecesMailApril piecesMailMay
axiom patternJune : condition_double_monthly piecesMailMay piecesMailJune
axiom patternJuly : condition_double_monthly piecesMailJune piecesMailJuly
axiom patternAugust : condition_double_monthly piecesMailJuly piecesMailAugust

-- Statement to prove
theorem VivianMailApril :
  piecesMailApril = 5 :=
by
  sorry

end VivianMailApril_l19_19685


namespace odd_f_neg1_l19_19392

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 
  if 0 ≤ x 
  then 2^x + 2 * x + b 
  else - (2^(-x) + 2 * (-x) + b)

theorem odd_f_neg1 (b : ℝ) (h : f 0 b = 0) : f (-1) b = -3 :=
by
  sorry

end odd_f_neg1_l19_19392


namespace find_m_find_A_inter_CUB_l19_19022

-- Definitions of sets A and B given m
def A (m : ℤ) : Set ℤ := {-4, 2 * m - 1, m ^ 2}
def B (m : ℤ) : Set ℤ := {9, m - 5, 1 - m}

-- Define the universal set U
def U : Set ℤ := Set.univ

-- First part: Prove that m = -3
theorem find_m (m : ℤ) : A m ∩ B m = {9} → m = -3 := sorry

-- Condition that m = -3 is true
def m_val : ℤ := -3

-- Second part: Prove A ∩ C_U B = {-4, -7}
theorem find_A_inter_CUB: A m_val ∩ (U \ B m_val) = {-4, -7} := sorry

end find_m_find_A_inter_CUB_l19_19022


namespace mike_remaining_cards_l19_19602

def initial_cards (mike_cards : ℕ) : ℕ := 87
def sam_cards (sam_bought : ℕ) : ℕ := 13
def alex_cards (alex_bought : ℕ) : ℕ := 15

theorem mike_remaining_cards (mike_cards sam_bought alex_bought : ℕ) :
  mike_cards - (sam_bought + alex_bought) = 59 :=
by
  let mike_cards := initial_cards 87
  let sam_cards := sam_bought
  let alex_cards := alex_bought
  sorry

end mike_remaining_cards_l19_19602


namespace arithmetic_progression_first_three_terms_l19_19926

theorem arithmetic_progression_first_three_terms 
  (S_n : ℤ) (d a_1 a_2 a_3 a_5 : ℤ)
  (h1 : S_n = 112) 
  (h2 : (a_1 + d) * d = 30)
  (h3 : (a_1 + 2 * d) + (a_1 + 4 * d) = 32) 
  (h4 : ∀ (n : ℕ), S_n = (n * (2 * a_1 + (n - 1) * d)) / 2) : 
  ((a_1 = 7 ∧ a_2 = 10 ∧ a_3 = 13) ∨ (a_1 = 1 ∧ a_2 = 6 ∧ a_3 = 11)) :=
sorry

end arithmetic_progression_first_three_terms_l19_19926


namespace store_total_income_l19_19299

def pencil_with_eraser_cost : ℝ := 0.8
def regular_pencil_cost : ℝ := 0.5
def short_pencil_cost : ℝ := 0.4

def pencils_with_eraser_sold : ℕ := 200
def regular_pencils_sold : ℕ := 40
def short_pencils_sold : ℕ := 35

noncomputable def total_money_made : ℝ :=
  (pencil_with_eraser_cost * pencils_with_eraser_sold) +
  (regular_pencil_cost * regular_pencils_sold) +
  (short_pencil_cost * short_pencils_sold)

theorem store_total_income : total_money_made = 194 := by
  sorry

end store_total_income_l19_19299


namespace arithmetic_series_first_term_l19_19136

theorem arithmetic_series_first_term :
  ∃ a d : ℚ, 
    (30 * (2 * a + 59 * d) = 240) ∧
    (30 * (2 * a + 179 * d) = 3240) ∧
    a = - (247 / 12) :=
by
  sorry

end arithmetic_series_first_term_l19_19136


namespace find_p_q_r_sum_l19_19753

theorem find_p_q_r_sum (p q r : ℕ) (hpq_rel_prime : Nat.gcd p q = 1) (hq_nonzero : q ≠ 0) 
  (h1 : ∃ t, (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4) 
  (h2 : ∃ t, (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r) : 
  p + q + r = 7 :=
sorry

end find_p_q_r_sum_l19_19753


namespace largest_common_term_arith_progressions_l19_19454

theorem largest_common_term_arith_progressions (a : ℕ) : 
  (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 3 + 9 * m ∧ a < 1000) → a = 984 := by
  -- Proof is not required, so we add sorry.
  sorry

end largest_common_term_arith_progressions_l19_19454


namespace appropriate_line_chart_for_temperature_l19_19683

-- Define the assumption that line charts are effective in displaying changes in data over time
axiom effective_line_chart_display (changes_over_time : Prop) : Prop

-- Define the statement to be proved, using the assumption above
theorem appropriate_line_chart_for_temperature (changes_over_time : Prop) 
  (line_charts_effective : effective_line_chart_display changes_over_time) : Prop :=
  sorry

end appropriate_line_chart_for_temperature_l19_19683


namespace fraction_arithmetic_l19_19967

theorem fraction_arithmetic : ( (4 / 5 - 1 / 10) / (2 / 5) ) = 7 / 4 :=
  sorry

end fraction_arithmetic_l19_19967


namespace minimum_ab_bc_ca_l19_19579

theorem minimum_ab_bc_ca {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = a^3) (h5 : a * b * c = a^3) : 
  ab + bc + ca ≥ 9 :=
sorry

end minimum_ab_bc_ca_l19_19579


namespace ball_bounce_height_l19_19778

theorem ball_bounce_height
  (k : ℕ) 
  (h1 : 20 * (2 / 3 : ℝ)^k < 2) : 
  k = 7 :=
sorry

end ball_bounce_height_l19_19778


namespace line_AC_eqn_l19_19614

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

-- Define point A
def A : Point := { x := 3, y := 1 }

-- Define point B
def B : Point := { x := -1, y := 2 }

-- Define the line equation y = x + 1
def line_eq (p : Point) : Prop := p.y = p.x + 1

-- Define the bisector being on line y=x+1 as a condition
axiom bisector_on_line (C : Point) : 
  line_eq C → (∃ k : ℝ, (C.y - B.y) = k * (C.x - B.x))

-- Define the final goal to prove the equation of line AC
theorem line_AC_eqn (C : Point) :
  line_eq C → ((A.x - C.x) * (B.y - C.y) = (B.x - C.x) * (A.y - C.y)) → C.x = -3 ∧ C.y = -2 → 
  (A.x - 2 * A.y = 1) := sorry

end line_AC_eqn_l19_19614


namespace condition_1_valid_for_n_condition_2_valid_for_n_l19_19030

-- Definitions from the conditions
def is_cube_root_of_unity (ω : ℂ) : Prop := ω^3 = 1

def roots_of_polynomial (ω : ℂ) (ω2 : ℂ) : Prop :=
  ω^2 + ω + 1 = 0 ∧ is_cube_root_of_unity ω ∧ is_cube_root_of_unity ω2

-- Problem statements
theorem condition_1_valid_for_n (n : ℕ) (ω : ℂ) (ω2 : ℂ) (h : roots_of_polynomial ω ω2) :
  (x^2 + x + 1) ∣ (x+1)^n - x^n - 1 ↔ ∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k - 1 := sorry

theorem condition_2_valid_for_n (n : ℕ) (ω : ℂ) (ω2 : ℂ) (h : roots_of_polynomial ω ω2) :
  (x^2 + x + 1) ∣ (x+1)^n + x^n + 1 ↔ ∃ k : ℕ, n = 6 * k + 2 ∨ n = 6 * k - 2 := sorry

end condition_1_valid_for_n_condition_2_valid_for_n_l19_19030


namespace find_smallest_n_modulo_l19_19824

theorem find_smallest_n_modulo :
  ∃ n : ℕ, n > 0 ∧ (2007 * n) % 1000 = 837 ∧ n = 691 :=
by
  sorry

end find_smallest_n_modulo_l19_19824


namespace sum_of_products_l19_19476

variable (a b c : ℝ)

theorem sum_of_products (h1 : a^2 + b^2 + c^2 = 250) (h2 : a + b + c = 16) : 
  ab + bc + ca = 3 :=
sorry

end sum_of_products_l19_19476


namespace cucumbers_for_20_apples_l19_19843

-- Definitions for all conditions
def apples := ℕ
def bananas := ℕ
def cucumbers := ℕ

def cost_equivalence_apples_bananas (a b : ℕ) : Prop := 10 * a = 5 * b
def cost_equivalence_bananas_cucumbers (b c : ℕ) : Prop := 3 * b = 4 * c

-- Main theorem statement
theorem cucumbers_for_20_apples :
  ∀ (a b c : ℕ),
    cost_equivalence_apples_bananas a b →
    cost_equivalence_bananas_cucumbers b c →
    ∃ k : ℕ, k = 13 :=
by
  intros
  sorry

end cucumbers_for_20_apples_l19_19843


namespace lines_intersect_l19_19979

noncomputable def line1 (t : ℚ) : ℚ × ℚ :=
(1 + 2 * t, 2 - 3 * t)

noncomputable def line2 (u : ℚ) : ℚ × ℚ :=
(-1 + 3 * u, 4 + u)

theorem lines_intersect :
  ∃ t u : ℚ, line1 t = (-5 / 11, 46 / 11) ∧ line2 u = (-5 / 11, 46 / 11) :=
sorry

end lines_intersect_l19_19979


namespace question1_question2_question3_l19_19129

open Set

-- Define sets A and B
def A := { x : ℝ | x^2 + 6 * x + 5 < 0 }
def B := { x : ℝ | -1 ≤ x ∧ x < 1 }

-- Universal set U is implicitly ℝ in Lean

-- Question 1: Prove A ∩ B = ∅
theorem question1 : A ∩ B = ∅ := 
sorry

-- Question 2: Prove complement of A ∪ B in ℝ is (-∞, -5] ∪ [1, ∞)
theorem question2 : compl (A ∪ B) = { x : ℝ | x ≤ -5 } ∪ { x : ℝ | x ≥ 1 } := 
sorry

-- Define set C which depends on parameter a
def C (a: ℝ) := { x : ℝ | x < a }

-- Question 3: Prove if B ∩ C = B, then a ≥ 1
theorem question3 (a : ℝ) (h : B ∩ C a = B) : a ≥ 1 := 
sorry

end question1_question2_question3_l19_19129


namespace problem1_problem2_l19_19489

-- Problem 1
theorem problem1 : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := sorry

-- Problem 2
theorem problem2 (a : ℝ) (ha : a ≠ 1) : (1 - 1 / a) / ((a^2 - 2 * a + 1) / a) = 1 / (a - 1) := sorry

end problem1_problem2_l19_19489


namespace arithmetic_seq_15th_term_is_53_l19_19116

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Original terms given
def a₁ : ℤ := -3
def d : ℤ := 4
def n : ℕ := 15

-- Prove that the 15th term is 53
theorem arithmetic_seq_15th_term_is_53 :
  arithmetic_seq a₁ d n = 53 :=
by
  sorry

end arithmetic_seq_15th_term_is_53_l19_19116


namespace interior_edges_sum_l19_19448

-- Definitions based on conditions
def frame_width : ℕ := 2
def frame_area : ℕ := 32
def outer_edge_length : ℕ := 8

-- Mathematically equivalent proof problem
theorem interior_edges_sum :
  ∃ (y : ℕ),  (frame_width * 2) * (y - frame_width * 2) = 32 ∧ (outer_edge_length * y - (outer_edge_length - 2 * frame_width) * (y - 2 * frame_width)) = 32 -> 4 + 4 + 0 + 0 = 8 :=
sorry

end interior_edges_sum_l19_19448


namespace minimum_value_of_f_l19_19678

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x + 1)

theorem minimum_value_of_f (x : ℝ) (h : x > -1) : f x = 1 ↔ x = 0 :=
by
  sorry

end minimum_value_of_f_l19_19678


namespace expand_expression_l19_19270

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l19_19270


namespace find_y_value_l19_19589

variable (x y z k : ℝ)

-- Conditions
def inverse_relation_y (x y : ℝ) (k : ℝ) : Prop := 5 * y = k / (x^2)
def direct_relation_z (x z : ℝ) : Prop := 3 * z = x

-- Constant from conditions
def k_constant := 500

-- Problem statement
theorem find_y_value (h1 : inverse_relation_y 2 25 k_constant) (h2 : direct_relation_z 4 6) :
  y = 6.25 :=
by
  sorry

-- Auxiliary instance to fulfill the proof requirement
noncomputable def y_value : ℝ := 6.25

end find_y_value_l19_19589


namespace resulting_expression_l19_19786

def x : ℕ := 1000
def y : ℕ := 10

theorem resulting_expression : 
  (x + 2 * y) + x + 3 * y + x + 4 * y + x + y = 4 * x + 10 * y :=
by
  sorry

end resulting_expression_l19_19786


namespace a7_arithmetic_sequence_l19_19394

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a1 : ℝ := 2
def a4 : ℝ := 5

theorem a7_arithmetic_sequence : ∃ d : ℝ, is_arithmetic_sequence a d ∧ a 1 = a1 ∧ a 4 = a4 → a 7 = 8 :=
by
  sorry

end a7_arithmetic_sequence_l19_19394


namespace bisect_area_of_trapezoid_l19_19806

-- Define the vertices of the quadrilateral
structure Point :=
  (x : ℤ)
  (y : ℤ)

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 16, y := 0 }
def C : Point := { x := 8, y := 8 }
def D : Point := { x := 0, y := 8 }

-- Define the equation of a line
structure Line :=
  (slope : ℚ)
  (intercept : ℚ)

-- Define the condition for parallel lines
def parallel (L1 L2 : Line) : Prop :=
  L1.slope = L2.slope

-- Define the diagonal AC and the required line
def AC : Line := { slope := 1, intercept := 0 }
def bisecting_line : Line := { slope := 1, intercept := -4 }

-- The area of trapezoid
def trapezoid_area : ℚ := (8 * (16 + 8)) / 2

-- Proof that the required line is parallel to AC and bisects the area of the trapezoid
theorem bisect_area_of_trapezoid :
  parallel bisecting_line AC ∧ 
  (1 / 2) * (8 * (16 + bisecting_line.intercept)) = trapezoid_area / 2 :=
by
  sorry

end bisect_area_of_trapezoid_l19_19806


namespace combined_age_of_Jane_and_John_in_future_l19_19449

def Justin_age : ℕ := 26
def Jessica_age_when_Justin_born : ℕ := 6
def James_older_than_Jessica : ℕ := 7
def Julia_younger_than_Justin : ℕ := 8
def Jane_older_than_James : ℕ := 25
def John_older_than_Jane : ℕ := 3
def years_later : ℕ := 12

theorem combined_age_of_Jane_and_John_in_future :
  let Jessica_age := Justin_age + Jessica_age_when_Justin_born
  let James_age := Jessica_age + James_older_than_Jessica
  let Julia_age := Justin_age - Julia_younger_than_Justin
  let Jane_age := James_age + Jane_older_than_James
  let John_age := Jane_age + John_older_than_Jane
  let Jane_age_after_years := Jane_age + years_later
  let John_age_after_years := John_age + years_later
  Jane_age_after_years + John_age_after_years = 155 :=
by
  sorry

end combined_age_of_Jane_and_John_in_future_l19_19449


namespace compute_complex_power_l19_19851

theorem compute_complex_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 :=
by
  sorry

end compute_complex_power_l19_19851


namespace red_light_at_A_prob_calc_l19_19144

-- Defining the conditions
def count_total_permutations : ℕ := Nat.factorial 4 / Nat.factorial 1
def count_favorable_permutations : ℕ := Nat.factorial 3 / Nat.factorial 1

-- Calculating the probability
def probability_red_at_A : ℚ := count_favorable_permutations / count_total_permutations

-- Statement to be proved
theorem red_light_at_A_prob_calc : probability_red_at_A = 1 / 4 :=
by
  sorry

end red_light_at_A_prob_calc_l19_19144


namespace debby_drinking_days_l19_19247

def starting_bottles := 264
def daily_consumption := 15
def bottles_left := 99

theorem debby_drinking_days : (starting_bottles - bottles_left) / daily_consumption = 11 :=
by
  -- proof steps will go here
  sorry

end debby_drinking_days_l19_19247


namespace calculate_expression_l19_19965

variables {a b c : ℤ}
variable (h1 : 5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c) -- a, b, c are multiples of 5
variable (h2 : a < b ∧ b < c) -- a < b < c
variable (h3 : c = a + 10) -- c = a + 10

theorem calculate_expression :
  (a - b) * (a - c) / (b - c) = -10 :=
by
  sorry

end calculate_expression_l19_19965


namespace solution_pairs_l19_19071

open Int

theorem solution_pairs (a b : ℝ) (h : ∀ n : ℕ, n > 0 → a * ⌊b * n⌋ = b * ⌊a * n⌋) :
  a = 0 ∨ b = 0 ∨ a = b ∨ (∃ (a_int b_int : ℤ), a = a_int ∧ b = b_int) :=
by sorry

end solution_pairs_l19_19071


namespace solve_equation_l19_19592

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x = -4/3) ↔ (x^2 + 2 * x + 2) / (x + 2) = x + 3 :=
by
  sorry

end solve_equation_l19_19592


namespace quadratic_inequality_solution_l19_19146

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 + 7 * x + 6 < 0) ↔ (-6 < x ∧ x < -1) :=
sorry

end quadratic_inequality_solution_l19_19146


namespace neg_alpha_quadrant_l19_19051

theorem neg_alpha_quadrant (α : ℝ) (k : ℤ) 
    (h1 : k * 360 + 180 < α)
    (h2 : α < k * 360 + 270) :
    k * 360 + 90 < -α ∧ -α < k * 360 + 180 :=
by
  sorry

end neg_alpha_quadrant_l19_19051


namespace factorize_quadratic_l19_19399

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end factorize_quadratic_l19_19399


namespace base9_subtraction_multiple_of_seven_l19_19892

theorem base9_subtraction_multiple_of_seven (b : ℕ) (h1 : 0 ≤ b ∧ b ≤ 9) 
(h2 : (3 * 9^6 + 1 * 9^5 + 5 * 9^4 + 4 * 9^3 + 6 * 9^2 + 7 * 9^1 + 2 * 9^0) - b % 7 = 0) : b = 0 :=
sorry

end base9_subtraction_multiple_of_seven_l19_19892


namespace investment_in_real_estate_l19_19573

def total_investment : ℝ := 200000
def ratio_real_estate_to_mutual_funds : ℝ := 7

theorem investment_in_real_estate (mutual_funds_investment real_estate_investment: ℝ) 
  (h1 : mutual_funds_investment + real_estate_investment = total_investment)
  (h2 : real_estate_investment = ratio_real_estate_to_mutual_funds * mutual_funds_investment) :
  real_estate_investment = 175000 := sorry

end investment_in_real_estate_l19_19573


namespace determinant_zero_implies_sum_l19_19312

open Matrix

noncomputable def matrix_example (a b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![2, 5, 8],
    ![4, a, b],
    ![4, b, a]
  ]

theorem determinant_zero_implies_sum (a b : ℝ) (h : a ≠ b) (h_det : det (matrix_example a b) = 0) : a + b = 26 :=
by
  sorry

end determinant_zero_implies_sum_l19_19312


namespace negation_proposition_l19_19280

open Classical

variable (x : ℝ)

def proposition (x : ℝ) : Prop := ∀ x > 1, Real.log x / Real.log 2 > 0

theorem negation_proposition (h : ¬ proposition x) : 
  ∃ x > 1, Real.log x / Real.log 2 ≤ 0 := by
  sorry

end negation_proposition_l19_19280


namespace count_integer_solutions_less_than_zero_l19_19278

theorem count_integer_solutions_less_than_zero : 
  ∃ k : ℕ, k = 4 ∧ (∀ n : ℤ, n^4 - n^3 - 3 * n^2 - 3 * n - 17 < 0 → k = 4) :=
by
  sorry

end count_integer_solutions_less_than_zero_l19_19278


namespace base_conversion_and_addition_l19_19623

theorem base_conversion_and_addition :
  let a₈ : ℕ := 3 * 8^2 + 5 * 8^1 + 6 * 8^0
  let c₁₄ : ℕ := 4 * 14^2 + 12 * 14^1 + 3 * 14^0
  a₈ + c₁₄ = 1193 :=
by
  sorry

end base_conversion_and_addition_l19_19623


namespace net_progress_l19_19779

-- Define the conditions as properties
def lost_yards : ℕ := 5
def gained_yards : ℕ := 10

-- Prove that the team's net progress is 5 yards
theorem net_progress : (gained_yards - lost_yards) = 5 :=
by
  sorry

end net_progress_l19_19779


namespace probability_floor_sqrt_even_l19_19229

/-- Suppose x and y are chosen randomly and uniformly from (0,1). The probability that
    ⌊√(x/y)⌋ is even is 1 - π²/24. -/
theorem probability_floor_sqrt_even (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  (1 - Real.pi ^ 2 / 24) = sorry :=
sorry

end probability_floor_sqrt_even_l19_19229


namespace doris_needs_weeks_l19_19900

noncomputable def average_weeks_to_cover_expenses (weekly_babysit_hours: ℝ) (saturday_hours: ℝ) : ℝ := 
  let weekday_income := weekly_babysit_hours * 20
  let saturday_income := saturday_hours * (if weekly_babysit_hours > 15 then 15 else 20)
  let teaching_income := 100
  let total_weekly_income := weekday_income + saturday_income + teaching_income
  let monthly_income_before_tax := total_weekly_income * 4
  let monthly_income_after_tax := monthly_income_before_tax * 0.85
  monthly_income_after_tax / 4 / 1200

theorem doris_needs_weeks (weekly_babysit_hours: ℝ) (saturday_hours: ℝ) :
  1200 ≤ (average_weeks_to_cover_expenses weekly_babysit_hours saturday_hours) * 4 * 1200 :=
  by
    sorry

end doris_needs_weeks_l19_19900


namespace Sara_spent_on_hotdog_l19_19744

-- Define the given constants
def totalCost : ℝ := 10.46
def costSalad : ℝ := 5.10

-- Define the value we need to prove
def costHotdog : ℝ := 5.36

-- Statement to prove
theorem Sara_spent_on_hotdog : totalCost - costSalad = costHotdog := by
  sorry

end Sara_spent_on_hotdog_l19_19744


namespace cylindrical_to_rectangular_point_l19_19472

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_point :
  cylindrical_to_rectangular (Real.sqrt 2) (Real.pi / 4) 1 = (1, 1, 1) :=
by
  sorry

end cylindrical_to_rectangular_point_l19_19472


namespace number_of_zookeepers_12_l19_19622

theorem number_of_zookeepers_12 :
  let P := 30 -- number of penguins
  let Zr := 22 -- number of zebras
  let T := 8 -- number of tigers
  let A_heads := P + Zr + T -- total number of animal heads
  let A_feet := (2 * P) + (4 * Zr) + (4 * T) -- total number of animal feet
  ∃ Z : ℕ, -- number of zookeepers
  (A_heads + Z) + 132 = A_feet + (2 * Z) → Z = 12 :=
by
  sorry

end number_of_zookeepers_12_l19_19622


namespace fair_tickets_sold_l19_19281

theorem fair_tickets_sold (F : ℕ) (number_of_baseball_game_tickets : ℕ) 
  (h1 : F = 2 * number_of_baseball_game_tickets + 6) (h2 : number_of_baseball_game_tickets = 56) :
  F = 118 :=
by
  sorry

end fair_tickets_sold_l19_19281


namespace hot_sauce_container_size_l19_19192

theorem hot_sauce_container_size :
  let serving_size := 0.5
  let servings_per_day := 3
  let days := 20
  let total_consumed := servings_per_day * serving_size * days
  let one_quart := 32
  one_quart - total_consumed = 2 :=
by
  sorry

end hot_sauce_container_size_l19_19192


namespace sine_gamma_half_leq_c_over_a_plus_b_l19_19812

variable (a b c : ℝ) (γ : ℝ)

-- Consider a triangle with sides a, b, c, and angle γ opposite to side c.
-- We need to prove that sin(γ / 2) ≤ c / (a + b).
theorem sine_gamma_half_leq_c_over_a_plus_b (h_c_pos : 0 < c) 
  (h_g_angle : 0 < γ ∧ γ < 2 * π) : 
  Real.sin (γ / 2) ≤ c / (a + b) := 
  sorry

end sine_gamma_half_leq_c_over_a_plus_b_l19_19812


namespace line_through_point_l19_19059

theorem line_through_point (k : ℝ) : (2 - k * 3 = -4 * (-2)) → k = -2 := by
  sorry

end line_through_point_l19_19059


namespace probability_product_zero_probability_product_negative_l19_19028

def given_set : List ℤ := [-3, -2, -1, 0, 5, 6, 7]

def num_pairs : ℕ := 21

theorem probability_product_zero :
  (6 : ℚ) / num_pairs = 2 / 7 := sorry

theorem probability_product_negative :
  (9 : ℚ) / num_pairs = 3 / 7 := sorry

end probability_product_zero_probability_product_negative_l19_19028


namespace shirt_cost_l19_19870

def cost_of_jeans_and_shirts (J S : ℝ) : Prop := (3 * J + 2 * S = 69) ∧ (2 * J + 3 * S = 81)

theorem shirt_cost (J S : ℝ) (h : cost_of_jeans_and_shirts J S) : S = 21 :=
by {
  sorry
}

end shirt_cost_l19_19870


namespace find_four_digit_number_l19_19687

def is_four_digit_number (k : ℕ) : Prop :=
  1000 ≤ k ∧ k < 10000

def appended_number (k : ℕ) : ℕ :=
  4000000 + k

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_four_digit_number (k : ℕ) (hk : is_four_digit_number k) :
  is_perfect_square (appended_number k) ↔ k = 4001 ∨ k = 8004 :=
sorry

end find_four_digit_number_l19_19687


namespace angle_D_measure_l19_19681

theorem angle_D_measure (E D F : ℝ) (h1 : E + D + F = 180) (h2 : E = 30) (h3 : D = 2 * F) : D = 100 :=
by
  -- The proof is not required, only the statement
  sorry

end angle_D_measure_l19_19681


namespace harmonic_sum_base_case_l19_19555

theorem harmonic_sum_base_case : 1 + 1/2 + 1/3 < 2 := 
sorry

end harmonic_sum_base_case_l19_19555


namespace initial_people_count_l19_19167

theorem initial_people_count (x : ℕ) (h : (x - 2) + 2 = 10) : x = 10 :=
by
  sorry

end initial_people_count_l19_19167


namespace no_solution_iff_k_nonnegative_l19_19073

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then k * x + 2 else (1 / 2) ^ x

theorem no_solution_iff_k_nonnegative (k : ℝ) :
  (¬ ∃ x : ℝ, f k (f k x) = 3 / 2) ↔ k ≥ 0 :=
  sorry

end no_solution_iff_k_nonnegative_l19_19073


namespace jewelry_store_gross_profit_l19_19426

theorem jewelry_store_gross_profit (purchase_price selling_price new_selling_price gross_profit : ℝ)
    (h1 : purchase_price = 240)
    (h2 : markup = 0.25 * selling_price)
    (h3 : selling_price = purchase_price + markup)
    (h4 : decrease = 0.20 * selling_price)
    (h5 : new_selling_price = selling_price - decrease)
    (h6 : gross_profit = new_selling_price - purchase_price) :
    gross_profit = 16 :=
by
    sorry

end jewelry_store_gross_profit_l19_19426


namespace points_opposite_sides_l19_19545

theorem points_opposite_sides (x y : ℝ) (h : (3 * x + 2 * y - 8) * (-1) < 0) : 3 * x + 2 * y > 8 := 
by
  sorry

end points_opposite_sides_l19_19545


namespace sum_derivatives_positive_l19_19524

noncomputable def f (x : ℝ) : ℝ := -x^2 - x^4 - x^6
noncomputable def f' (x : ℝ) : ℝ := -2*x - 4*x^3 - 6*x^5

theorem sum_derivatives_positive (x1 x2 x3 : ℝ) (h1 : x1 + x2 < 0) (h2 : x2 + x3 < 0) (h3 : x3 + x1 < 0) :
  f' x1 + f' x2 + f' x3 > 0 := 
sorry

end sum_derivatives_positive_l19_19524


namespace inequality_implies_range_of_a_l19_19119

theorem inequality_implies_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |1 + x| ≥ a^2 - 2 * a) → (-1 ≤ a ∧ a ≤ 3) :=
sorry

end inequality_implies_range_of_a_l19_19119


namespace units_digit_fraction_l19_19092

theorem units_digit_fraction : (2^3 * 31 * 33 * 17 * 7) % 10 = 6 := by
  sorry

end units_digit_fraction_l19_19092


namespace a10_plus_b10_l19_19927

noncomputable def a : ℝ := sorry -- a will be a real number satisfying the conditions
noncomputable def b : ℝ := sorry -- b will be a real number satisfying the conditions

axiom ab_condition1 : a + b = 1
axiom ab_condition2 : a^2 + b^2 = 3
axiom ab_condition3 : a^3 + b^3 = 4
axiom ab_condition4 : a^4 + b^4 = 7
axiom ab_condition5 : a^5 + b^5 = 11

theorem a10_plus_b10 : a^10 + b^10 = 123 :=
by 
  sorry

end a10_plus_b10_l19_19927


namespace correct_option_c_l19_19223

theorem correct_option_c (x : ℝ) : -2 * (x + 1) = -2 * x - 2 :=
  by
  -- Proof can be omitted
  sorry

end correct_option_c_l19_19223


namespace carol_age_difference_l19_19492

theorem carol_age_difference (bob_age carol_age : ℕ) (h1 : bob_age + carol_age = 66)
  (h2 : carol_age = 3 * bob_age + 2) (h3 : bob_age = 16) (h4 : carol_age = 50) :
  carol_age - 3 * bob_age = 2 :=
by
  sorry

end carol_age_difference_l19_19492


namespace ab_minus_a_inv_b_l19_19162

theorem ab_minus_a_inv_b (a : ℝ) (b : ℚ) (h1 : a > 1) (h2 : 0 < (b : ℝ)) (h3 : (a ^ (b : ℝ)) + (a ^ (-(b : ℝ))) = 2 * Real.sqrt 2) :
  (a ^ (b : ℝ)) - (a ^ (-(b : ℝ))) = 2 := 
sorry

end ab_minus_a_inv_b_l19_19162


namespace units_digit_of_2_to_the_10_l19_19672

theorem units_digit_of_2_to_the_10 : ∃ d : ℕ, (d < 10) ∧ (2^10 % 10 = d) ∧ (d == 4) :=
by {
  -- sorry to skip the proof
  sorry
}

end units_digit_of_2_to_the_10_l19_19672


namespace area_of_region_R_l19_19840

open Real

noncomputable def area_of_strip (width : ℝ) (height : ℝ) : ℝ :=
  width * height

noncomputable def area_of_triangle (leg : ℝ) : ℝ :=
  1 / 2 * leg * leg

theorem area_of_region_R :
  let unit_square_area := 1
  let AE_BE := 1 / sqrt 2
  let area_triangle_ABE := area_of_triangle AE_BE
  let strip_width := 1 / 4
  let strip_height := 1
  let area_strip := area_of_strip strip_width strip_height
  let overlap_area := area_triangle_ABE / 2
  let area_R := area_strip - overlap_area
  area_R = 1 / 8 :=
by
  sorry

end area_of_region_R_l19_19840


namespace maximum_of_f_attain_maximum_of_f_l19_19557

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 4

theorem maximum_of_f : ∀ x : ℝ, f x ≤ 0 :=
sorry

theorem attain_maximum_of_f : ∃ x : ℝ, f x = 0 :=
sorry

end maximum_of_f_attain_maximum_of_f_l19_19557


namespace least_gumballs_to_get_four_same_color_l19_19665

theorem least_gumballs_to_get_four_same_color
  (R W B : ℕ)
  (hR : R = 9)
  (hW : W = 7)
  (hB : B = 8) : 
  ∃ n, n = 10 ∧ (∀ m < n, ∀ r w b : ℕ, r + w + b = m → r < 4 ∧ w < 4 ∧ b < 4) ∧ 
  (∀ r w b : ℕ, r + w + b = n → r = 4 ∨ w = 4 ∨ b = 4) :=
sorry

end least_gumballs_to_get_four_same_color_l19_19665


namespace symmetry_axis_of_transformed_function_l19_19982

theorem symmetry_axis_of_transformed_function :
  let initial_func (x : ℝ) := Real.sin (4 * x - π / 6)
  let stretched_func (x : ℝ) := Real.sin (8 * x - π / 3)
  let transformed_func (x : ℝ) := Real.sin (8 * (x + π / 4) - π / 3)
  let ω := 8
  let φ := 5 * π / 3
  x = π / 12 :=
  sorry

end symmetry_axis_of_transformed_function_l19_19982


namespace arithmetic_expression_evaluation_l19_19104

theorem arithmetic_expression_evaluation :
  2 + 8 * 3 - 4 + 7 * 6 / 3 = 36 := by
  sorry

end arithmetic_expression_evaluation_l19_19104


namespace combined_bus_capacity_l19_19306

-- Define conditions
def train_capacity : ℕ := 120
def bus_capacity : ℕ := train_capacity / 6
def number_of_buses : ℕ := 2

-- Define theorem for the combined capacity of two buses
theorem combined_bus_capacity : number_of_buses * bus_capacity = 40 := by
  -- We declare that the proof is skipped here
  sorry

end combined_bus_capacity_l19_19306


namespace total_gifts_l19_19355

theorem total_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end total_gifts_l19_19355


namespace sum_of_remainders_mod_53_l19_19941

theorem sum_of_remainders_mod_53 (x y z : ℕ) (hx : x % 53 = 36) (hy : y % 53 = 15) (hz : z % 53 = 7) : 
  (x + y + z) % 53 = 5 :=
by
  sorry

end sum_of_remainders_mod_53_l19_19941


namespace axis_of_symmetry_r_minus_2s_zero_l19_19406

/-- 
Prove that if y = x is an axis of symmetry for the curve 
y = (2 * p * x + q) / (r * x - 2 * s) with p, q, r, s nonzero, 
then r - 2s = 0. 
-/
theorem axis_of_symmetry_r_minus_2s_zero
  (p q r s : ℝ) (h_p : p ≠ 0) (h_q : q ≠ 0) (h_r : r ≠ 0) (h_s : s ≠ 0) 
  (h_sym : ∀ (a b : ℝ), (b = (2 * p * a + q) / (r * a - 2 * s)) ↔ (a = (2 * p * b + q) / (r * b - 2 * s))) :
  r - 2 * s = 0 :=
sorry

end axis_of_symmetry_r_minus_2s_zero_l19_19406


namespace inequality_proof_l19_19854

theorem inequality_proof (x y z : ℝ) (hx : 2 < x) (hx4 : x < 4) (hy : 2 < y) (hy4 : y < 4) (hz : 2 < z) (hz4 : z < 4) :
  (x / (y^2 - z) + y / (z^2 - x) + z / (x^2 - y)) > 1 :=
by
  sorry

end inequality_proof_l19_19854


namespace closest_weight_total_shortfall_total_selling_price_l19_19076

-- Definitions
def standard_weight : ℝ := 25
def weights : List ℝ := [1.5, -3, 2, -0.5, 1, -2, -2.5, -2]
def price_per_kg : ℝ := 2.6

-- Assertions
theorem closest_weight : ∃ w ∈ weights, abs w = 0.5 ∧ 25 + w = 24.5 :=
by sorry

theorem total_shortfall : (weights.sum = -5.5) :=
by sorry

theorem total_selling_price : (8 * standard_weight + weights.sum) * price_per_kg = 505.7 :=
by sorry

end closest_weight_total_shortfall_total_selling_price_l19_19076


namespace hammer_nail_cost_l19_19578

variable (h n : ℝ)

theorem hammer_nail_cost (h n : ℝ)
    (h1 : 4 * h + 5 * n = 10.45)
    (h2 : 3 * h + 9 * n = 12.87) :
  20 * h + 25 * n = 52.25 :=
sorry

end hammer_nail_cost_l19_19578


namespace domain_of_f_l19_19863

noncomputable def f (t : ℝ) : ℝ :=  1 / ((abs (t - 1))^2 + (abs (t + 1))^2)

theorem domain_of_f : ∀ t : ℝ, (abs (t - 1))^2 + (abs (t + 1))^2 ≠ 0 :=
by
  intro t
  sorry

end domain_of_f_l19_19863


namespace shara_monthly_payment_l19_19528

theorem shara_monthly_payment : 
  ∀ (T M : ℕ), 
  (T / 2 = 6 * M) → 
  (T / 2 - 4 * M = 20) → 
  M = 10 :=
by
  intros T M h1 h2
  sorry

end shara_monthly_payment_l19_19528


namespace incorrect_statement_C_l19_19596

theorem incorrect_statement_C (x : ℝ) (h : x > -2) : (6 / x) > -3 :=
sorry

end incorrect_statement_C_l19_19596


namespace combined_salaries_l19_19012

theorem combined_salaries (A B C D E : ℝ) 
  (hC : C = 11000) 
  (hAverage : (A + B + C + D + E) / 5 = 8200) : 
  A + B + D + E = 30000 := 
by 
  sorry

end combined_salaries_l19_19012


namespace no_solution_exists_l19_19198

def product_of_digits (x : ℕ) : ℕ :=
  if x < 10 then x else (x / 10) * (x % 10)

theorem no_solution_exists :
  ¬ ∃ x : ℕ, product_of_digits x = x^2 - 10 * x - 22 :=
by
  sorry

end no_solution_exists_l19_19198


namespace oranges_sold_l19_19823

def bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def rotten_oranges : ℕ := 50
def oranges_for_juice : ℕ := 30

theorem oranges_sold : (bags * oranges_per_bag) - rotten_oranges - oranges_for_juice = 220 := by
  sorry

end oranges_sold_l19_19823


namespace new_function_expression_l19_19539

def initial_function (x : ℝ) : ℝ := -2 * x ^ 2

def shifted_function (x : ℝ) : ℝ := -2 * (x + 1) ^ 2 - 3

theorem new_function_expression :
  (∀ x : ℝ, (initial_function (x + 1) - 3) = shifted_function x) :=
by
  sorry

end new_function_expression_l19_19539


namespace at_least_one_wins_l19_19922

def probability_A := 1 / 2
def probability_B := 1 / 4

def probability_at_least_one (pA pB : ℚ) : ℚ := 
  1 - ((1 - pA) * (1 - pB))

theorem at_least_one_wins :
  probability_at_least_one probability_A probability_B = 5 / 8 := 
by
  sorry

end at_least_one_wins_l19_19922


namespace number_multiplied_by_9_l19_19368

theorem number_multiplied_by_9 (x : ℕ) (h : 50 = x + 26) : 9 * x = 216 := by
  sorry

end number_multiplied_by_9_l19_19368


namespace largest_fraction_l19_19605

theorem largest_fraction
  (a b c d : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d) :
  (c + d) / (a + b) ≥ (a + b) / (c + d)
  ∧ (c + d) / (a + b) ≥ (a + d) / (b + c)
  ∧ (c + d) / (a + b) ≥ (b + c) / (a + d)
  ∧ (c + d) / (a + b) ≥ (b + d) / (a + c) :=
by
  sorry

end largest_fraction_l19_19605


namespace material_left_eq_l19_19576

theorem material_left_eq :
  let a := (4 / 17 : ℚ)
  let b := (3 / 10 : ℚ)
  let total_bought := a + b
  let used := (0.23529411764705882 : ℚ)
  total_bought - used = (51 / 170 : ℚ) :=
by
  let a := (4 / 17 : ℚ)
  let b := (3 / 10 : ℚ)
  let total_bought := a + b
  let used := (0.23529411764705882 : ℚ)
  show total_bought - used = (51 / 170)
  sorry

end material_left_eq_l19_19576


namespace largest_and_smallest_value_of_expression_l19_19993

theorem largest_and_smallest_value_of_expression
  (w x y z : ℝ)
  (h1 : w + x + y + z = 0)
  (h2 : w^7 + x^7 + y^7 + z^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 :=
sorry

end largest_and_smallest_value_of_expression_l19_19993


namespace sum_xy_22_l19_19699

theorem sum_xy_22 (x y : ℕ) (h1 : 0 < x) (h2 : x < 25) (h3 : 0 < y) (h4 : y < 25) 
  (h5 : x + y + x * y = 118) : x + y = 22 :=
sorry

end sum_xy_22_l19_19699


namespace maximum_value_of_x_minus_y_l19_19004

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l19_19004


namespace find_y_value_l19_19780

theorem find_y_value : 
  (15^2 * 8^3) / y = 450 → y = 256 :=
by
  sorry

end find_y_value_l19_19780


namespace train_cars_estimate_l19_19339

noncomputable def train_cars_count (total_time_secs : ℕ) (delay_secs : ℕ) (cars_counted : ℕ) (count_time_secs : ℕ): ℕ := 
  let rate_per_sec := cars_counted / count_time_secs
  let cars_missed := delay_secs * rate_per_sec
  let cars_in_remaining_time := rate_per_sec * (total_time_secs - delay_secs)
  cars_missed + cars_in_remaining_time

theorem train_cars_estimate :
  train_cars_count 210 15 8 20 = 120 :=
sorry

end train_cars_estimate_l19_19339


namespace problem_proof_l19_19277

variable (a b c : ℝ)

-- Given conditions
def conditions (a b c : ℝ) : Prop :=
  (0 < a ∧ 0 < b ∧ 0 < c) ∧ ((a + 1) * (b + 1) * (c + 1) = 8)

-- The proof problem
theorem problem_proof (h : conditions a b c) : a + b + c ≥ 3 ∧ a * b * c ≤ 1 :=
  sorry

end problem_proof_l19_19277


namespace problem1_problem2_problem3_problem4_l19_19263

theorem problem1 : 23 + (-16) - (-7) = 14 := by
  sorry

theorem problem2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by
  sorry

theorem problem3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -(10/3) := by
  sorry

theorem problem4 : -1 ^ 4 - (1 - 0.5) * (1/3) * (2 - (-3) ^ 2) = 1/6 := by 
  sorry

end problem1_problem2_problem3_problem4_l19_19263


namespace simplify_expression_l19_19956

theorem simplify_expression (a b c d : ℝ) (h₁ : a + b + c + d = 0) (h₂ : a ≠ 0) (h₃ : b ≠ 0) (h₄ : c ≠ 0) (h₅ : d ≠ 0) :
  (1 / (b^2 + c^2 + d^2 - a^2) + 
   1 / (a^2 + c^2 + d^2 - b^2) + 
   1 / (a^2 + b^2 + d^2 - c^2) + 
   1 / (a^2 + b^2 + c^2 - d^2)) = 4 / d^2 := 
sorry

end simplify_expression_l19_19956


namespace real_roots_range_l19_19461

theorem real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k*x^2 - 6*x + 9 = 0) ↔ k ≤ 1 :=
sorry

end real_roots_range_l19_19461


namespace magnitude_of_z_l19_19077

noncomputable def z : ℂ := Complex.I * (3 + 4 * Complex.I)

theorem magnitude_of_z : Complex.abs z = 5 := by
  sorry

end magnitude_of_z_l19_19077


namespace find_b_l19_19048

theorem find_b
  (a b c : ℚ)
  (h1 : (4 : ℚ) * a = 12)
  (h2 : (4 * (4 * b) = - (14:ℚ) + 3 * a)) :
  b = -(7:ℚ) / 2 :=
by sorry

end find_b_l19_19048


namespace infinite_n_dividing_a_pow_n_plus_1_l19_19666

theorem infinite_n_dividing_a_pow_n_plus_1 (a : ℕ) (h1 : 1 < a) (h2 : a % 2 = 0) :
  ∃ (S : Set ℕ), S.Infinite ∧ ∀ n ∈ S, n ∣ a^n + 1 := 
sorry

end infinite_n_dividing_a_pow_n_plus_1_l19_19666


namespace teal_bluish_count_l19_19410

theorem teal_bluish_count (n G Bg N B : ℕ) (h1 : n = 120) (h2 : G = 80) (h3 : Bg = 35) (h4 : N = 20) :
  B = 55 :=
by
  sorry

end teal_bluish_count_l19_19410


namespace smallest_sum_xyz_l19_19139

theorem smallest_sum_xyz (x y z : ℕ) (h : x * y * z = 40320) : x + y + z ≥ 103 :=
sorry

end smallest_sum_xyz_l19_19139


namespace device_failure_probability_l19_19609

noncomputable def probability_fail_device (p1 p2 p3 : ℝ) (p_one p_two p_three : ℝ) : ℝ :=
  0.006 * p3 + 0.092 * p_two + 0.398 * p_one

theorem device_failure_probability
  (p1 p2 p3 : ℝ) (p_one p_two p_three : ℝ)
  (h1 : p1 = 0.1)
  (h2 : p2 = 0.2)
  (h3 : p3 = 0.3)
  (h4 : p_one = 0.25)
  (h5 : p_two = 0.6)
  (h6 : p_three = 0.9) :
  probability_fail_device p1 p2 p3 p_one p_two p_three = 0.1601 :=
by
  sorry

end device_failure_probability_l19_19609


namespace BD_range_l19_19190

noncomputable def quadrilateral_BD (AB BC CD DA : ℕ) (BD : ℤ) :=
  AB = 7 ∧ BC = 15 ∧ CD = 7 ∧ DA = 11 ∧ (9 ≤ BD ∧ BD ≤ 17)

theorem BD_range : 
  ∀ (AB BC CD DA : ℕ) (BD : ℤ),
  quadrilateral_BD AB BC CD DA BD → 
  9 ≤ BD ∧ BD ≤ 17 :=
by
  intros AB BC CD DA BD h
  cases h
  -- We would then prove the conditions
  sorry

end BD_range_l19_19190


namespace intersection_is_integer_for_m_l19_19693

noncomputable def intersects_at_integer_point (m : ℤ) : Prop :=
∃ x y : ℤ, y = x - 4 ∧ y = m * x + 2 * m

theorem intersection_is_integer_for_m :
  intersects_at_integer_point 8 :=
by
  -- The proof would go here
  sorry

end intersection_is_integer_for_m_l19_19693


namespace min_value_of_z_l19_19601

theorem min_value_of_z : ∀ (x : ℝ), ∃ z : ℝ, z = 5 * x^2 - 20 * x + 45 ∧ z ≥ 25 :=
by sorry

end min_value_of_z_l19_19601


namespace bill_take_home_salary_l19_19326

-- Define the parameters
def property_taxes : ℝ := 2000
def sales_taxes : ℝ := 3000
def gross_salary : ℝ := 50000
def income_tax_rate : ℝ := 0.10

-- Define income tax calculation
def income_tax : ℝ := income_tax_rate * gross_salary

-- Define total taxes calculation
def total_taxes : ℝ := property_taxes + sales_taxes + income_tax

-- Define the take-home salary calculation
def take_home_salary : ℝ := gross_salary - total_taxes

-- Statement of the theorem
theorem bill_take_home_salary : take_home_salary = 40000 := by
  -- Sorry is used to skip the proof.
  sorry

end bill_take_home_salary_l19_19326


namespace whale_length_l19_19344

theorem whale_length
  (velocity_fast : ℕ)
  (velocity_slow : ℕ)
  (time : ℕ)
  (h1 : velocity_fast = 18)
  (h2 : velocity_slow = 15)
  (h3 : time = 15) :
  (velocity_fast - velocity_slow) * time = 45 := 
by
  sorry

end whale_length_l19_19344


namespace xunzi_statement_l19_19318

/-- 
Given the conditions:
  "If not accumulating small steps, then not reaching a thousand miles."
  Which can be represented as: ¬P → ¬q.
Prove that accumulating small steps (P) is a necessary but not sufficient condition for
reaching a thousand miles (q).
-/
theorem xunzi_statement (P q : Prop) (h : ¬P → ¬q) : (q → P) ∧ ¬(P → q) :=
by sorry

end xunzi_statement_l19_19318


namespace vectors_parallel_l19_19709

def are_parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, b = k • a

theorem vectors_parallel :
  let a := (1, 2, -2)
  let b := (-2, -4, 4)
  are_parallel a b :=
by
  let a := (1, 2, -2)
  let b := (-2, -4, 4)
  -- Proof omitted
  sorry

end vectors_parallel_l19_19709


namespace jake_weight_l19_19776

theorem jake_weight (J S B : ℝ) (h1 : J - 8 = 2 * S)
                            (h2 : B = 2 * J + 6)
                            (h3 : J + S + B = 480)
                            (h4 : B = 1.25 * S) :
  J = 230 :=
by
  sorry

end jake_weight_l19_19776


namespace find_x_l19_19237

variable (N x : ℕ)
variable (h1 : N = 500 * x + 20)
variable (h2 : 4 * 500 + 20 = 2020)

theorem find_x : x = 4 := by
  -- The proof code will go here
  sorry

end find_x_l19_19237


namespace orchard_produce_l19_19240

theorem orchard_produce (num_apple_trees num_orange_trees apple_baskets_per_tree apples_per_basket orange_baskets_per_tree oranges_per_basket : ℕ) 
  (h1 : num_apple_trees = 50) 
  (h2 : num_orange_trees = 30) 
  (h3 : apple_baskets_per_tree = 25) 
  (h4 : apples_per_basket = 18)
  (h5 : orange_baskets_per_tree = 15) 
  (h6 : oranges_per_basket = 12) 
: (num_apple_trees * (apple_baskets_per_tree * apples_per_basket) = 22500) ∧ 
  (num_orange_trees * (orange_baskets_per_tree * oranges_per_basket) = 5400) :=
  by 
  sorry

end orchard_produce_l19_19240


namespace average_rainfall_virginia_l19_19235

noncomputable def average_rainfall : ℝ :=
  (3.79 + 4.5 + 3.95 + 3.09 + 4.67) / 5

theorem average_rainfall_virginia : average_rainfall = 4 :=
by
  sorry

end average_rainfall_virginia_l19_19235


namespace area_increase_l19_19382

theorem area_increase (original_length original_width new_length : ℝ)
  (h1 : original_length = 20)
  (h2 : original_width = 5)
  (h3 : new_length = original_length + 10) :
  (new_length * original_width - original_length * original_width) = 50 := by
  sorry

end area_increase_l19_19382


namespace ratio_expression_value_l19_19416

variable {A B C : ℚ}

theorem ratio_expression_value (h : A / B = 3 / 2 ∧ A / C = 3 / 6) : (4 * A - 3 * B) / (5 * C + 2 * A) = 1 / 4 := 
sorry

end ratio_expression_value_l19_19416
