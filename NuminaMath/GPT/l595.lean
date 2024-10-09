import Mathlib

namespace students_walk_fraction_l595_59599

theorem students_walk_fraction
  (school_bus_fraction : ℚ := 1/3)
  (car_fraction : ℚ := 1/5)
  (bicycle_fraction : ℚ := 1/8) :
  (1 - (school_bus_fraction + car_fraction + bicycle_fraction) = 41/120) :=
by
  sorry

end students_walk_fraction_l595_59599


namespace simplify_expression_l595_59564

theorem simplify_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 + 2 * x) / ((x - 1) * (x + 3)) - (5 * x + 3) / ((x - 1) * (x + 3))
  = 3 * (x^2 - x - 1) / ((x - 1) * (x + 3)) :=
by
  sorry

end simplify_expression_l595_59564


namespace range_of_a_l595_59534

open Real

theorem range_of_a
  (a : ℝ)
  (curve : ∀ θ : ℝ, ∃ p : ℝ × ℝ, p = (a + 2 * cos θ, a + 2 * sin θ))
  (distance_two_points : ∀ θ : ℝ, dist (0,0) (a + 2 * cos θ, a + 2 * sin θ) = 2) :
  (-2 * sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < 2 * sqrt 2) :=
sorry

end range_of_a_l595_59534


namespace remainder_of_789987_div_8_l595_59549

theorem remainder_of_789987_div_8 : (789987 % 8) = 3 := by
  sorry

end remainder_of_789987_div_8_l595_59549


namespace arithmetic_square_root_16_l595_59551

theorem arithmetic_square_root_16 : ∀ x : ℝ, x ≥ 0 → x^2 = 16 → x = 4 :=
by
  intro x hx h
  sorry

end arithmetic_square_root_16_l595_59551


namespace ratio_x_y_z_l595_59507

variables (x y z : ℝ)

theorem ratio_x_y_z (h1 : 0.60 * x = 0.30 * y) 
                    (h2 : 0.80 * z = 0.40 * x) 
                    (h3 : z = 2 * y) : 
                    x / y = 4 ∧ y / y = 1 ∧ z / y = 2 :=
by
  sorry

end ratio_x_y_z_l595_59507


namespace neg_sin_prop_iff_l595_59517

theorem neg_sin_prop_iff :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by sorry

end neg_sin_prop_iff_l595_59517


namespace find_father_age_l595_59574

variable (M F : ℕ)

noncomputable def age_relation_1 : Prop := M = (2 / 5) * F
noncomputable def age_relation_2 : Prop := M + 5 = (1 / 2) * (F + 5)

theorem find_father_age (h1 : age_relation_1 M F) (h2 : age_relation_2 M F) : F = 25 := by
  sorry

end find_father_age_l595_59574


namespace percentage_of_students_receiving_certificates_l595_59547

theorem percentage_of_students_receiving_certificates
  (boys girls : ℕ)
  (pct_boys pct_girls : ℕ)
  (h_boys : boys = 30)
  (h_girls : girls = 20)
  (h_pct_boys : pct_boys = 30)
  (h_pct_girls : pct_girls = 40)
  :
  (pct_boys * boys + pct_girls * girls) / (100 * (boys + girls)) * 100 = 34 :=
by
  sorry

end percentage_of_students_receiving_certificates_l595_59547


namespace minimal_odd_sum_is_1683_l595_59510

/-!
# Proof Problem:
Prove that the minimal odd sum of two three-digit numbers and one four-digit number 
formed using the digits 0 through 9 exactly once is 1683.
-/
theorem minimal_odd_sum_is_1683 :
  ∃ (a b : ℕ) (c : ℕ), 
    100 ≤ a ∧ a < 1000 ∧ 
    100 ≤ b ∧ b < 1000 ∧ 
    1000 ≤ c ∧ c < 10000 ∧ 
    a + b + c % 2 = 1 ∧ 
    (∀ d e f : ℕ, 
      100 ≤ d ∧ d < 1000 ∧ 
      100 ≤ e ∧ e < 1000 ∧ 
      1000 ≤ f ∧ f < 10000 ∧ 
      d + e + f % 2 = 1 → a + b + c ≤ d + e + f) ∧ 
    a + b + c = 1683 := 
sorry

end minimal_odd_sum_is_1683_l595_59510


namespace set_B_forms_triangle_l595_59535

theorem set_B_forms_triangle (a b c : ℝ) (h1 : a = 25) (h2 : b = 24) (h3 : c = 7):
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end set_B_forms_triangle_l595_59535


namespace Kayla_points_on_first_level_l595_59537

theorem Kayla_points_on_first_level
(points_2 : ℕ) (points_3 : ℕ) (points_4 : ℕ) (points_5 : ℕ) (points_6 : ℕ)
(h2 : points_2 = 3) (h3 : points_3 = 5) (h4 : points_4 = 8) (h5 : points_5 = 12) (h6 : points_6 = 17) :
  ∃ (points_1 : ℕ), 
    (points_3 - points_2 = 2) ∧ 
    (points_4 - points_3 = 3) ∧ 
    (points_5 - points_4 = 4) ∧ 
    (points_6 - points_5 = 5) ∧ 
    (points_2 - points_1 = 1) ∧ 
    points_1 = 2 :=
by
  use 2
  repeat { split }
  sorry

end Kayla_points_on_first_level_l595_59537


namespace find_g_zero_l595_59541

noncomputable def g (x : ℝ) : ℝ := sorry  -- fourth-degree polynomial

-- Conditions
axiom cond1 : |g 1| = 16
axiom cond2 : |g 3| = 16
axiom cond3 : |g 4| = 16
axiom cond4 : |g 5| = 16
axiom cond5 : |g 6| = 16
axiom cond6 : |g 7| = 16

-- statement to prove
theorem find_g_zero : |g 0| = 54 := 
by sorry

end find_g_zero_l595_59541


namespace divides_polynomial_difference_l595_59543

def P (a b c d x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem divides_polynomial_difference (a b c d x y : ℤ) (hxneqy : x ≠ y) :
  (x - y) ∣ (P a b c d x - P a b c d y) :=
by
  sorry

end divides_polynomial_difference_l595_59543


namespace contractor_absent_days_l595_59518

variable (x y : ℝ) -- x for the number of days worked, y for the number of days absent

-- Conditions
def eng_days := x + y = 30
def total_money := 25 * x - 7.5 * y = 425

-- Theorem
theorem contractor_absent_days (x y : ℝ) (h1 : eng_days x y) (h2 : total_money x y) : y = 10 := 
sorry

end contractor_absent_days_l595_59518


namespace num_positive_divisors_36_l595_59583

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l595_59583


namespace math_equivalence_problem_l595_59540

theorem math_equivalence_problem :
  (2^2 + 92 * 3^2) * (4^2 + 92 * 5^2) = 1388^2 + 92 * 2^2 :=
by
  sorry

end math_equivalence_problem_l595_59540


namespace hyperbola_equation_l595_59567

-- Definitions of the conditions
def is_asymptote_1 (y x : ℝ) : Prop :=
  y = 2 * x

def is_asymptote_2 (y x : ℝ) : Prop :=
  y = -2 * x

def passes_through_focus (x y : ℝ) : Prop :=
  x = 1 ∧ y = 0

-- The statement to be proved
theorem hyperbola_equation :
  (∀ x y : ℝ, passes_through_focus x y → x^2 - (y^2 / 4) = 1) :=
sorry

end hyperbola_equation_l595_59567


namespace calc_expression_is_24_l595_59516

def calc_expression : ℕ := (30 / (8 + 2 - 5)) * 4

theorem calc_expression_is_24 : calc_expression = 24 :=
by
  sorry

end calc_expression_is_24_l595_59516


namespace new_ratio_l595_59545

theorem new_ratio (J: ℝ) (F: ℝ) (F_new: ℝ): 
  J = 59.99999999999997 → 
  F / J = 3 / 2 → 
  F_new = F + 10 → 
  F_new / J = 5 / 3 :=
by
  intros hJ hF hF_new
  sorry

end new_ratio_l595_59545


namespace problem_inequality_sol1_problem_inequality_sol2_l595_59548

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x - (2 * a + 2)

theorem problem_inequality_sol1 (a x : ℝ) :
  (a > -3 / 2 ∧ (x > 2 * a + 2 ∨ x < -1)) ∨
  (a = -3 / 2 ∧ x ≠ -1) ∨
  (a < -3 / 2 ∧ (x > -1 ∨ x < 2 * a + 2)) ↔
  f x a > x :=
sorry

theorem problem_inequality_sol2 (a : ℝ) :
  (∀ x : ℝ, x > -1 → f x a + 3 ≥ 0) ↔
  a ≤ Real.sqrt 2 - 1 :=
sorry

end problem_inequality_sol1_problem_inequality_sol2_l595_59548


namespace prove_a_eq_b_l595_59501

theorem prove_a_eq_b (a b : ℝ) (h : 1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b)) : a = b :=
sorry

end prove_a_eq_b_l595_59501


namespace range_of_m_necessary_condition_range_of_m_not_sufficient_condition_l595_59585

-- Problem I Statement
theorem range_of_m_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8 * x - 20 ≤ 0) → (1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) →
  (-Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
by sorry

-- Problem II Statement
theorem range_of_m_not_sufficient_condition (m : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 8 * x - 20 ≤ 0) → ¬(1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) →
  (m ≤ -3 ∨ m ≥ 3) :=
by sorry

end range_of_m_necessary_condition_range_of_m_not_sufficient_condition_l595_59585


namespace find_a_b_c_l595_59533

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_a_b_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hp1 : is_prime (a + b * c))
  (hp2 : is_prime (b + a * c))
  (hp3 : is_prime (c + a * b))
  (hdiv1 : (a + b * c) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)))
  (hdiv2 : (b + a * c) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)))
  (hdiv3 : (c + a * b) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1))) :
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end find_a_b_c_l595_59533


namespace workers_and_days_l595_59511

theorem workers_and_days (x y : ℕ) (h1 : x * y = (x - 20) * (y + 5)) (h2 : x * y = (x + 15) * (y - 2)) :
  x = 60 ∧ y = 10 := 
by {
  sorry
}

end workers_and_days_l595_59511


namespace arith_seq_ratio_l595_59579

variable {S T : ℕ → ℚ}

-- Conditions
def is_arith_seq_sum (S : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n, S n = n * (2 * a 1 + (n - 1) * a n) / 2

def ratio_condition (S T : ℕ → ℚ) :=
  ∀ n, S n / T n = (2 * n - 1) / (3 * n + 2)

-- Main theorem
theorem arith_seq_ratio
  (a b : ℕ → ℚ)
  (h1 : is_arith_seq_sum S a)
  (h2 : is_arith_seq_sum T b)
  (h3 : ratio_condition S T)
  : a 7 / b 7 = 25 / 41 :=
sorry

end arith_seq_ratio_l595_59579


namespace find_a_l595_59558

variable (U : Set ℝ) (A : Set ℝ) (a : ℝ)

theorem find_a (hU_def : U = {2, 3, a^2 - a - 1})
               (hA_def : A = {2, 3})
               (h_compl : U \ A = {1}) :
  a = -1 ∨ a = 2 := 
sorry

end find_a_l595_59558


namespace paint_cost_of_cube_l595_59538

theorem paint_cost_of_cube (cost_per_kg : ℕ) (coverage_per_kg : ℕ) (side_length : ℕ) (total_cost : ℕ) 
  (h1 : cost_per_kg = 20)
  (h2 : coverage_per_kg = 15)
  (h3 : side_length = 5)
  (h4 : total_cost = 200) : 
  (6 * side_length^2 / coverage_per_kg) * cost_per_kg = total_cost :=
by
  sorry

end paint_cost_of_cube_l595_59538


namespace fraction_shaded_area_l595_59546

theorem fraction_shaded_area (l w : ℕ) (h_l : l = 15) (h_w : w = 20)
  (h_qtr : (1 / 4: ℝ) * (l * w) = 75) (h_shaded : (1 / 5: ℝ) * 75 = 15) :
  (15 / (l * w): ℝ) = 1 / 20 :=
by
  sorry

end fraction_shaded_area_l595_59546


namespace solution_for_m_exactly_one_solution_l595_59562

theorem solution_for_m_exactly_one_solution (m : ℚ) : 
  (∀ x : ℚ, (x - 3) / (m * x + 4) = 2 * x → 
            (2 * m * x^2 + 7 * x + 3 = 0)) →
  (49 - 24 * m = 0) → 
  m = 49 / 24 :=
by
  intro h1 h2
  sorry

end solution_for_m_exactly_one_solution_l595_59562


namespace sum_of_roots_of_equation_l595_59581

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end sum_of_roots_of_equation_l595_59581


namespace total_apples_correct_l595_59530

variable (X : ℕ)

def Sarah_apples : ℕ := X

def Jackie_apples : ℕ := 2 * Sarah_apples X

def Adam_apples : ℕ := Jackie_apples X + 5

def total_apples : ℕ := Sarah_apples X + Jackie_apples X + Adam_apples X

theorem total_apples_correct : total_apples X = 5 * X + 5 := by
  sorry

end total_apples_correct_l595_59530


namespace ratio_doctors_to_lawyers_l595_59531

-- Definitions based on conditions
def average_age_doctors := 35
def average_age_lawyers := 50
def combined_average_age := 40

-- Define variables
variables (d l : ℕ) -- d is number of doctors, l is number of lawyers

-- Hypothesis based on the problem statement
axiom h : (average_age_doctors * d + average_age_lawyers * l) = combined_average_age * (d + l)

-- The theorem we need to prove is the ratio of doctors to lawyers is 2:1
theorem ratio_doctors_to_lawyers : d = 2 * l :=
by sorry

end ratio_doctors_to_lawyers_l595_59531


namespace triangle_inequality_l595_59598

noncomputable def area_triangle (a b c : ℝ) : ℝ := sorry -- Definition of area, but implementation is not required.

theorem triangle_inequality (a b c : ℝ) (S_triangle : ℝ):
  1 - (8 * ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / (a + b + c) ^ 2)
  ≤ 432 * S_triangle ^ 2 / (a + b + c) ^ 4
  ∧ 432 * S_triangle ^ 2 / (a + b + c) ^ 4
  ≤ 1 - (2 * ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / (a + b + c) ^ 2) :=
sorry -- Proof is omitted

end triangle_inequality_l595_59598


namespace triangle_is_isosceles_l595_59552

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h_triangle : A + B + C = π)
  (h_condition : 2 * Real.cos B * Real.sin C = Real.sin A) :
  B = C :=
sorry

end triangle_is_isosceles_l595_59552


namespace proof_problem_l595_59521

noncomputable def log2 : ℝ := Real.log 3 / Real.log 2
noncomputable def log5 : ℝ := Real.log 3 / Real.log 5

variables {x y : ℝ}

theorem proof_problem
  (h1 : log2 > 1)
  (h2 : 0 < log5 ∧ log5 < 1)
  (h3 : (log2^x - log5^x) ≥ (log2^(-y) - log5^(-y))) :
  x + y ≥ 0 :=
sorry

end proof_problem_l595_59521


namespace petya_vasya_sum_equality_l595_59524

theorem petya_vasya_sum_equality : ∃ (k m : ℕ), 2^(k+1) * 1023 = m * (m + 1) :=
by
  sorry

end petya_vasya_sum_equality_l595_59524


namespace polynomial_remainder_l595_59505

theorem polynomial_remainder (x : ℤ) : (x + 1) ∣ (x^15 + 1) ↔ x = -1 := sorry

end polynomial_remainder_l595_59505


namespace greatest_possible_value_of_median_l595_59560

-- Given conditions as definitions
variables (k m r s t : ℕ)

-- condition 1: The average (arithmetic mean) of the 5 integers is 10
def avg_is_10 : Prop := k + m + r + s + t = 50

-- condition 2: The integers are in a strictly increasing order
def increasing_order : Prop := k < m ∧ m < r ∧ r < s ∧ s < t

-- condition 3: t is 20
def t_is_20 : Prop := t = 20

-- The main statement to prove
theorem greatest_possible_value_of_median : 
  avg_is_10 k m r s t → 
  increasing_order k m r s t → 
  t_is_20 t → 
  r = 13 :=
by
  intros
  sorry

end greatest_possible_value_of_median_l595_59560


namespace range_of_m_real_roots_l595_59561

theorem range_of_m_real_roots (m : ℝ) : 
  (∀ x : ℝ, ∃ k l : ℝ, k = 2*x ∧ l = m - x^2 ∧ k^2 - 4*l ≥ 0) ↔ m ≤ 1 := 
sorry

end range_of_m_real_roots_l595_59561


namespace base_length_of_parallelogram_l595_59582

theorem base_length_of_parallelogram (area : ℝ) (base altitude : ℝ)
  (h1 : area = 98)
  (h2 : altitude = 2 * base) :
  base = 7 :=
by
  sorry

end base_length_of_parallelogram_l595_59582


namespace range_of_x_l595_59588

theorem range_of_x (x : ℝ) (h1 : 2 ≤ |x - 5|) (h2 : |x - 5| ≤ 10) (h3 : 0 < x) : 
  (0 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 15) := 
sorry

end range_of_x_l595_59588


namespace basketball_team_win_requirement_l595_59519

noncomputable def basketball_win_percentage_goal (games_played_so_far games_won_so_far games_remaining win_percentage_goal : ℕ) : ℕ :=
  let total_games := games_played_so_far + games_remaining
  let required_wins := (win_percentage_goal * total_games) / 100
  required_wins - games_won_so_far

theorem basketball_team_win_requirement :
  basketball_win_percentage_goal 60 45 50 75 = 38 := 
by
  sorry

end basketball_team_win_requirement_l595_59519


namespace total_amount_shared_l595_59584

theorem total_amount_shared (a b c : ℝ)
  (h1 : a = 1/3 * (b + c))
  (h2 : b = 2/7 * (a + c))
  (h3 : a = b + 20) : 
  a + b + c = 720 :=
by
  sorry

end total_amount_shared_l595_59584


namespace correct_operation_l595_59509

theorem correct_operation :
  (∀ a : ℕ, a ^ 3 * a ^ 2 = a ^ 5) ∧
  (∀ a : ℕ, a + a ^ 2 ≠ a ^ 3) ∧
  (∀ a : ℕ, 6 * a ^ 2 / (2 * a ^ 2) = 3) ∧
  (∀ a : ℕ, (3 * a ^ 2) ^ 3 ≠ 9 * a ^ 6) :=
by
  sorry

end correct_operation_l595_59509


namespace first_train_cross_time_is_10_seconds_l595_59590

-- Definitions based on conditions
def length_of_train := 120 -- meters
def time_second_train_cross_telegraph_post := 15 -- seconds
def distance_cross_each_other := 240 -- meters
def time_cross_each_other := 12 -- seconds

-- The speed of the second train
def speed_second_train := length_of_train / time_second_train_cross_telegraph_post -- m/s

-- The relative speed of both trains when crossing each other
def relative_speed := distance_cross_each_other / time_cross_each_other -- m/s

-- The speed of the first train
def speed_first_train := relative_speed - speed_second_train -- m/s

-- The time taken by the first train to cross the telegraph post
def time_first_train_cross_telegraph_post := length_of_train / speed_first_train -- seconds

-- Proof statement
theorem first_train_cross_time_is_10_seconds :
  time_first_train_cross_telegraph_post = 10 := by
  sorry

end first_train_cross_time_is_10_seconds_l595_59590


namespace w_janous_conjecture_l595_59587

theorem w_janous_conjecture (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (z^2 - x^2) / (x + y) + (x^2 - y^2) / (y + z) + (y^2 - z^2) / (z + x) ≥ 0 :=
by
  sorry

end w_janous_conjecture_l595_59587


namespace molecular_weight_boric_acid_l595_59513

theorem molecular_weight_boric_acid :
  let H := 1.008  -- atomic weight of Hydrogen in g/mol
  let B := 10.81  -- atomic weight of Boron in g/mol
  let O := 16.00  -- atomic weight of Oxygen in g/mol
  let H3BO3 := 3 * H + B + 3 * O  -- molecular weight of H3BO3
  H3BO3 = 61.834 :=  -- correct molecular weight of H3BO3
by
  sorry

end molecular_weight_boric_acid_l595_59513


namespace dog_roaming_area_comparison_l595_59503

theorem dog_roaming_area_comparison :
  let r := 10
  let a1 := (1/2) * Real.pi * r^2
  let a2 := (3/4) * Real.pi * r^2 - (1/4) * Real.pi * 6^2 
  a2 > a1 ∧ a2 - a1 = 16 * Real.pi :=
by
  sorry

end dog_roaming_area_comparison_l595_59503


namespace triangle_perimeter_l595_59514

/-- Given a triangle with two sides of lengths 2 and 5, and the third side being a root of the equation
    x^2 - 8x + 12 = 0, the perimeter of the triangle is 13. --/
theorem triangle_perimeter
  (a b : ℕ) 
  (ha : a = 2) 
  (hb : b = 5)
  (c : ℕ)
  (h_c_root : c * c - 8 * c + 12 = 0)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 13 := 
sorry

end triangle_perimeter_l595_59514


namespace number_of_pupils_in_class_l595_59550

-- Defining the conditions
def wrongMark : ℕ := 79
def correctMark : ℕ := 45
def averageIncreasedByHalf : ℕ := 2  -- Condition representing average increased by half

-- The goal is to prove the number of pupils is 68
theorem number_of_pupils_in_class (n S : ℕ) (h1 : wrongMark = 79) (h2 : correctMark = 45)
(h3 : averageIncreasedByHalf = 2) 
(h4 : S + (wrongMark - correctMark) = (3 / 2) * S) :
  n = 68 :=
  sorry

end number_of_pupils_in_class_l595_59550


namespace moores_law_transistors_l595_59563

-- Define the initial conditions
def initial_transistors : ℕ := 500000
def doubling_period : ℕ := 2 -- in years
def transistors_doubling (n : ℕ) : ℕ := initial_transistors * 2^n

-- Calculate the number of doubling events from 1995 to 2010
def years_spanned : ℕ := 15
def number_of_doublings : ℕ := years_spanned / doubling_period

-- Expected number of transistors in 2010
def expected_transistors_in_2010 : ℕ := 64000000

theorem moores_law_transistors :
  transistors_doubling number_of_doublings = expected_transistors_in_2010 :=
sorry

end moores_law_transistors_l595_59563


namespace trigonometric_identity_l595_59593

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α + Real.pi / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * Real.pi / 3) = -7 / 9 :=
  sorry

end trigonometric_identity_l595_59593


namespace rank_matA_l595_59577

def matA : Matrix (Fin 4) (Fin 5) ℤ :=
  ![![5, 7, 12, 48, -14],
    ![9, 16, 24, 98, -31],
    ![14, 24, 25, 146, -45],
    ![11, 12, 24, 94, -25]]

theorem rank_matA : Matrix.rank matA = 3 :=
by
  sorry

end rank_matA_l595_59577


namespace math_problem_common_factors_and_multiples_l595_59506

-- Definitions
def a : ℕ := 180
def b : ℕ := 300

-- The Lean statement to be proved
theorem math_problem_common_factors_and_multiples :
    Nat.lcm a b = 900 ∧
    Nat.gcd a b = 60 ∧
    {d | d ∣ a ∧ d ∣ b} = {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60} :=
by
  sorry

end math_problem_common_factors_and_multiples_l595_59506


namespace seats_needed_l595_59544

-- Definitions based on the problem's condition
def children : ℕ := 58
def children_per_seat : ℕ := 2

-- Theorem statement to prove
theorem seats_needed : children / children_per_seat = 29 :=
by
  sorry

end seats_needed_l595_59544


namespace octagon_has_20_diagonals_l595_59522

-- Define the number of sides for an octagon.
def octagon_sides : ℕ := 8

-- Define the formula for the number of diagonals in an n-sided polygon.
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove the number of diagonals in an octagon equals 20.
theorem octagon_has_20_diagonals : diagonals octagon_sides = 20 := by
  sorry

end octagon_has_20_diagonals_l595_59522


namespace acres_used_for_corn_l595_59528

-- Define the conditions in the problem:
def total_land : ℕ := 1034
def ratio_beans : ℕ := 5
def ratio_wheat : ℕ := 2
def ratio_corn : ℕ := 4
def total_ratio : ℕ := ratio_beans + ratio_wheat + ratio_corn

-- Proof problem statement: Prove the number of acres used for corn is 376 acres
theorem acres_used_for_corn : total_land * ratio_corn / total_ratio = 376 := by
  -- Proof goes here
  sorry

end acres_used_for_corn_l595_59528


namespace max_value_x_div_y_l595_59591

variables {x y a b : ℝ}

theorem max_value_x_div_y (h1 : x ≥ y) (h2 : y > 0) (h3 : 0 ≤ a) (h4 : a ≤ x) (h5 : 0 ≤ b) (h6 : b ≤ y) 
  (h7 : (x - a)^2 + (y - b)^2 = x^2 + b^2) (h8 : x^2 + b^2 = y^2 + a^2) :
  x / y ≤ (2 * Real.sqrt 3) / 3 :=
sorry

end max_value_x_div_y_l595_59591


namespace max_value_sum_l595_59520

variable (n : ℕ) (x : Fin n → ℝ)

theorem max_value_sum 
  (h1 : ∀ i, 0 ≤ x i)
  (h2 : 2 ≤ n)
  (h3 : (Finset.univ : Finset (Fin n)).sum x = 1) :
  ∃ max_val, max_val = (1 / 4) :=
sorry

end max_value_sum_l595_59520


namespace coefficient_of_x_l595_59594

theorem coefficient_of_x : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
  ∃ (a b c : ℝ), expr = a * x^2 + b * x + c ∧ b = 5 := by
    let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
    exact sorry

end coefficient_of_x_l595_59594


namespace exists_n_consecutive_non_prime_or_prime_power_l595_59569

theorem exists_n_consecutive_non_prime_or_prime_power (n : ℕ) (h : n > 0) :
  ∃ (seq : Fin n → ℕ), (∀ i, ¬ (Nat.Prime (seq i)) ∧ ¬ (∃ p k : ℕ, p.Prime ∧ k > 1 ∧ seq i = p ^ k)) :=
by
  sorry

end exists_n_consecutive_non_prime_or_prime_power_l595_59569


namespace robinson_family_children_count_l595_59556

theorem robinson_family_children_count 
  (m : ℕ) -- mother's age
  (f : ℕ) (f_age : f = 50) -- father's age is 50
  (x : ℕ) -- number of children
  (y : ℕ) -- average age of children
  (h1 : (m + 50 + x * y) / (2 + x) = 22)
  (h2 : (m + x * y) / (1 + x) = 18) :
  x = 6 := 
sorry

end robinson_family_children_count_l595_59556


namespace probability_Cecilia_rolls_4_given_win_l595_59565

noncomputable def P_roll_Cecilia_4_given_win : ℚ :=
  let P_C1_4 := 1/6
  let P_W_C := 1/5
  let P_W_C_given_C1_4 := (4/6)^4
  let P_C1_4_and_W_C := P_C1_4 * P_W_C_given_C1_4
  let P_C1_4_given_W_C := P_C1_4_and_W_C / P_W_C
  P_C1_4_given_W_C

theorem probability_Cecilia_rolls_4_given_win :
  P_roll_Cecilia_4_given_win = 256 / 1555 :=
by 
  -- Here the proof would go, but we include sorry for now.
  sorry

end probability_Cecilia_rolls_4_given_win_l595_59565


namespace units_digit_of_33_pow_33_mul_22_pow_22_l595_59566

theorem units_digit_of_33_pow_33_mul_22_pow_22 :
  (33 ^ (33 * (22 ^ 22))) % 10 = 1 :=
sorry

end units_digit_of_33_pow_33_mul_22_pow_22_l595_59566


namespace smallest_integer_y_l595_59536

theorem smallest_integer_y (y : ℤ) (h : 7 - 3 * y < 20) : ∃ (y : ℤ), y = -4 :=
by
  sorry

end smallest_integer_y_l595_59536


namespace complex_number_is_purely_imaginary_l595_59572

theorem complex_number_is_purely_imaginary (a : ℂ) : 
  (a^2 - a - 2 = 0) ∧ (a^2 - 3*a + 2 ≠ 0) ↔ a = -1 :=
by 
  sorry

end complex_number_is_purely_imaginary_l595_59572


namespace temperature_on_friday_l595_59554

theorem temperature_on_friday 
  (M T W Th F : ℤ) 
  (h1 : (M + T + W + Th) / 4 = 48) 
  (h2 : (T + W + Th + F) / 4 = 46) 
  (h3 : M = 43) : 
  F = 35 := 
by
  sorry

end temperature_on_friday_l595_59554


namespace rahul_meena_work_together_l595_59576

theorem rahul_meena_work_together (days_rahul : ℚ) (days_meena : ℚ) (combined_days : ℚ) :
  days_rahul = 5 ∧ days_meena = 10 → combined_days = 10 / 3 :=
by
  intros h
  sorry

end rahul_meena_work_together_l595_59576


namespace find_angle_BAC_l595_59578

-- Definitions and Hypotheses
variables (A B C P : Type) (AP PC AB AC : Real) (angle_BPC : Real)

-- Hypotheses
-- AP = PC
-- AB = AC
-- angle BPC = 120 
axiom AP_eq_PC : AP = PC
axiom AB_eq_AC : AB = AC
axiom angle_BPC_eq_120 : angle_BPC = 120

-- Theorem
theorem find_angle_BAC (AP_eq_PC : AP = PC) (AB_eq_AC : AB = AC) (angle_BPC_eq_120 : angle_BPC = 120) : angle_BAC = 60 :=
sorry

end find_angle_BAC_l595_59578


namespace triangle_geometric_sequence_sine_rule_l595_59539

noncomputable def sin60 : Real := Real.sqrt 3 / 2

theorem triangle_geometric_sequence_sine_rule 
  {a b c : Real} 
  {A B C : Real} 
  (h1 : a / b = b / c) 
  (h2 : A = 60 * Real.pi / 180) :
  b * Real.sin B / c = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_geometric_sequence_sine_rule_l595_59539


namespace inequality_not_hold_l595_59504

theorem inequality_not_hold (a b : ℝ) (h : a < b ∧ b < 0) : (1 / (a - b) < 1 / a) :=
by
  sorry

end inequality_not_hold_l595_59504


namespace centered_hexagonal_seq_l595_59542

def is_centered_hexagonal (a : ℕ) : Prop :=
  ∃ n : ℕ, a = 3 * n^2 - 3 * n + 1

def are_sequences (a b c d : ℕ) : Prop :=
  (b = 2 * a - 1) ∧ (d = c^2) ∧ (a + b = c + d)

theorem centered_hexagonal_seq (a : ℕ) :
  (∃ b c d, are_sequences a b c d) ↔ is_centered_hexagonal a :=
sorry

end centered_hexagonal_seq_l595_59542


namespace remainder_2519_div_6_l595_59597

theorem remainder_2519_div_6 : ∃ q r, 2519 = 6 * q + r ∧ 0 ≤ r ∧ r < 6 ∧ r = 5 := 
by
  sorry

end remainder_2519_div_6_l595_59597


namespace least_whole_number_subtracted_l595_59523

theorem least_whole_number_subtracted {x : ℕ} (h : 6 > x ∧ 7 > x) :
  (6 - x) / (7 - x : ℝ) < 16 / 21 -> x = 3 :=
by
  intros
  sorry

end least_whole_number_subtracted_l595_59523


namespace range_of_a_l595_59532

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 3) : -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l595_59532


namespace find_missing_number_l595_59502

theorem find_missing_number (n : ℝ) :
  (0.0088 * 4.5) / (0.05 * 0.1 * n) = 990 → n = 0.008 :=
by
  intro h
  sorry

end find_missing_number_l595_59502


namespace crayons_problem_l595_59515

theorem crayons_problem 
  (total_crayons : ℕ)
  (red_crayons : ℕ)
  (blue_crayons : ℕ)
  (green_crayons : ℕ)
  (pink_crayons : ℕ)
  (h1 : total_crayons = 24)
  (h2 : red_crayons = 8)
  (h3 : blue_crayons = 6)
  (h4 : green_crayons = 2 / 3 * blue_crayons)
  (h5 : pink_crayons = total_crayons - red_crayons - blue_crayons - green_crayons) :
  pink_crayons = 6 :=
by
  sorry

end crayons_problem_l595_59515


namespace find_x_l595_59571

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x^n

theorem find_x (x : ℝ) (H : series_sum x = 16) : 
  x = (33 - Real.sqrt 129) / 32 :=
by
  sorry

end find_x_l595_59571


namespace alpha_value_l595_59525

theorem alpha_value (b : ℝ) : (∀ x : ℝ, (|2 * x - 3| < 2) ↔ (x^2 + -3 * x + b < 0)) :=
by
  sorry

end alpha_value_l595_59525


namespace find_a_l595_59527

theorem find_a (a b c : ℕ) (h1 : (18 ^ a) * (9 ^ (3 * a - 1)) * (c ^ (2 * a - 3)) = (2 ^ 7) * (3 ^ b)) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : a = 7 :=
by
  sorry

end find_a_l595_59527


namespace find_q_value_l595_59555

theorem find_q_value (q : ℚ) (x y : ℚ) (hx : x = 5 - q) (hy : y = 3*q - 1) : x = 3*y → q = 4/5 :=
by
  sorry

end find_q_value_l595_59555


namespace smallest_nonfactor_product_of_factors_of_48_l595_59529

theorem smallest_nonfactor_product_of_factors_of_48 :
  ∃ a b : ℕ, a ≠ b ∧ a ∣ 48 ∧ b ∣ 48 ∧ ¬ (a * b ∣ 48) ∧ (∀ c d : ℕ, c ≠ d ∧ c ∣ 48 ∧ d ∣ 48 ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ a * b = 18 :=
sorry

end smallest_nonfactor_product_of_factors_of_48_l595_59529


namespace calculate_fraction_l595_59512

variable (a b : ℝ)

theorem calculate_fraction (h : a ≠ b) : (2 * a / (a - b)) + (2 * b / (b - a)) = 2 := by
  sorry

end calculate_fraction_l595_59512


namespace correct_judgment_l595_59573

def P := Real.pi < 2
def Q := Real.pi > 3

theorem correct_judgment : (P ∨ Q) ∧ ¬P := by
  sorry

end correct_judgment_l595_59573


namespace number_of_real_solutions_l595_59575

theorem number_of_real_solutions (x : ℝ) (n : ℤ) : 
  (3 : ℝ) * x^2 - 27 * (n : ℝ) + 29 = 0 → n = ⌊x⌋ →  ∃! x, (3 : ℝ) * x^2 - 27 * (⌊x⌋ : ℝ) + 29 = 0 := 
sorry

end number_of_real_solutions_l595_59575


namespace parabolas_intersect_at_points_l595_59570

theorem parabolas_intersect_at_points :
  ∃ (x y : ℝ), (y = 3 * x^2 - 5 * x + 1 ∧ y = 4 * x^2 + 3 * x + 1) ↔ ((x = 0 ∧ y = 1) ∨ (x = -8 ∧ y = 233)) := 
sorry

end parabolas_intersect_at_points_l595_59570


namespace problem_statement_l595_59553

theorem problem_statement (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) 
  (h_eq : x + y + z = 1/x + 1/y + 1/z) : 
  x + y + z ≥ Real.sqrt ((x * y + 1) / 2) + Real.sqrt ((y * z + 1) / 2) + Real.sqrt ((z * x + 1) / 2) :=
by
  sorry

end problem_statement_l595_59553


namespace min_packs_needed_l595_59592

theorem min_packs_needed (P8 P15 P30 : ℕ) (h: P8 * 8 + P15 * 15 + P30 * 30 = 120) : P8 + P15 + P30 = 4 :=
by
  sorry

end min_packs_needed_l595_59592


namespace bacteria_growth_final_count_l595_59596

theorem bacteria_growth_final_count (initial_count : ℕ) (t : ℕ) 
(h1 : initial_count = 10) 
(h2 : t = 7) 
(h3 : ∀ n : ℕ, (n * 60) = t * 60 → 2 ^ n = 128) : 
(initial_count * 2 ^ t) = 1280 := 
by
  sorry

end bacteria_growth_final_count_l595_59596


namespace time_to_traverse_nth_mile_l595_59526

theorem time_to_traverse_nth_mile (n : ℕ) (n_pos : n > 1) :
  let k := (1 / 2 : ℝ)
  let s_n := k / ((n-1) * (2 ^ (n-2)))
  let t_n := 1 / s_n
  t_n = 2 * (n-1) * 2^(n-2) := 
by sorry

end time_to_traverse_nth_mile_l595_59526


namespace probability_one_red_ball_distribution_of_X_l595_59589

-- Definitions of probabilities
def C (n k : ℕ) : ℕ := Nat.choose n k

def P_one_red_ball : ℚ := (C 2 1 * C 3 2 : ℚ) / C 5 3

#check (1 : ℚ)
#check (3 : ℚ)
#check (5 : ℚ)
def X_distribution (i : ℕ) : ℚ :=
  if i = 0 then (C 3 3 : ℚ) / C 5 3
  else if i = 1 then (C 2 1 * C 3 2 : ℚ) / C 5 3
  else if i = 2 then (C 2 2 * C 3 1 : ℚ) / C 5 3
  else 0

-- Statement to prove
theorem probability_one_red_ball : 
  P_one_red_ball = 3 / 5 := 
sorry

theorem distribution_of_X :
  Π i, (i = 0 → X_distribution i = 1 / 10) ∧
       (i = 1 → X_distribution i = 3 / 5) ∧
       (i = 2 → X_distribution i = 3 / 10) :=
sorry

end probability_one_red_ball_distribution_of_X_l595_59589


namespace no_valid_placement_of_prisms_l595_59508

-- Definitions: Rectangular prism with edges parallel to OX, OY, and OZ axes.
structure RectPrism :=
  (x_interval : Set ℝ)
  (y_interval : Set ℝ)
  (z_interval : Set ℝ)

-- Function to determine if two rectangular prisms intersect.
def intersects (P Q : RectPrism) : Prop :=
  ¬ Disjoint P.x_interval Q.x_interval ∧
  ¬ Disjoint P.y_interval Q.y_interval ∧
  ¬ Disjoint P.z_interval Q.z_interval

-- Definition of the 12 rectangular prisms
def prisms := Fin 12 → RectPrism

-- Conditions for intersection:
def intersection_condition (prisms : prisms) : Prop :=
  ∀ i : Fin 12, ∀ j : Fin 12,
    (j = (i + 1) % 12) ∨ (j = (i - 1 + 12) % 12) ∨ intersects (prisms i) (prisms j)

theorem no_valid_placement_of_prisms :
  ¬ ∃ (prisms : prisms), intersection_condition prisms :=
sorry

end no_valid_placement_of_prisms_l595_59508


namespace euro_operation_example_l595_59568

def euro_operation (x y : ℕ) : ℕ := 3 * x * y

theorem euro_operation_example : euro_operation 3 (euro_operation 4 5) = 540 :=
by sorry

end euro_operation_example_l595_59568


namespace speed_of_car_in_second_hour_l595_59586

noncomputable def speed_in_first_hour : ℝ := 90
noncomputable def average_speed : ℝ := 82.5
noncomputable def total_time : ℝ := 2

theorem speed_of_car_in_second_hour : 
  ∃ (speed_in_second_hour : ℝ), 
  (speed_in_first_hour + speed_in_second_hour) / total_time = average_speed ∧ 
  speed_in_first_hour = 90 ∧ 
  average_speed = 82.5 → 
  speed_in_second_hour = 75 :=
by 
  sorry

end speed_of_car_in_second_hour_l595_59586


namespace sum_digits_increment_l595_59595

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_increment (n : ℕ) (h : sum_digits n = 1365) : 
  sum_digits (n + 1) = 1360 :=
by
  sorry

end sum_digits_increment_l595_59595


namespace lcm_of_numbers_l595_59580

-- Define the conditions given in the problem
def ratio (a b : ℕ) : Prop := 7 * b = 13 * a
def hcf_23 (a b : ℕ) : Prop := Nat.gcd a b = 23

-- Main statement to prove
theorem lcm_of_numbers (a b : ℕ) (h_ratio : ratio a b) (h_hcf : hcf_23 a b) : Nat.lcm a b = 2093 := by
  sorry

end lcm_of_numbers_l595_59580


namespace son_age_l595_59557

theorem son_age (S M : ℕ) (h1 : M = S + 30) (h2 : M + 2 = 2 * (S + 2)) : S = 28 := 
by
  -- The proof can be filled in here.
  sorry

end son_age_l595_59557


namespace sequence_geometric_l595_59559

theorem sequence_geometric (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 = 1)
  (h_geom : ∀ k : ℕ, a (k + 1) - a k = (1 / 3) ^ k) :
  a n = (3 / 2) * (1 - (1 / 3) ^ n) :=
by
  sorry

end sequence_geometric_l595_59559


namespace compare_neg_fractions_l595_59500

theorem compare_neg_fractions : 
  (- (8:ℚ) / 21) > - (3 / 7) :=
by sorry

end compare_neg_fractions_l595_59500
