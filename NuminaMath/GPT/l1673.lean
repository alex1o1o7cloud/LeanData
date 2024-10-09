import Mathlib

namespace number_of_terms_in_arithmetic_sequence_is_39_l1673_167351

theorem number_of_terms_in_arithmetic_sequence_is_39 :
  ∀ (a d l : ℤ), 
  d ≠ 0 → 
  a = 128 → 
  d = -3 → 
  l = 14 → 
  ∃ n : ℕ, (a + (↑n - 1) * d = l) ∧ (n = 39) :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_is_39_l1673_167351


namespace speed_of_second_train_l1673_167307

theorem speed_of_second_train
  (distance : ℝ)
  (speed_fast : ℝ)
  (time_difference : ℝ)
  (v : ℝ)
  (h_distance : distance = 425.80645161290323)
  (h_speed_fast : speed_fast = 75)
  (h_time_difference : time_difference = 4)
  (h_v : v = distance / (distance / speed_fast + time_difference)) :
  v = 44 := 
sorry

end speed_of_second_train_l1673_167307


namespace quadratic_fixed_points_l1673_167315

noncomputable def quadratic_function (a x : ℝ) : ℝ :=
  a * x^2 + (3 * a - 1) * x - (10 * a + 3)

theorem quadratic_fixed_points (a : ℝ) (h : a ≠ 0) :
  quadratic_function a 2 = -5 ∧ quadratic_function a (-5) = 2 :=
by sorry

end quadratic_fixed_points_l1673_167315


namespace find_a_given_solution_l1673_167388

theorem find_a_given_solution (a : ℝ) (x : ℝ) (h : x = 1) (eqn : a * (x + 1) = 2 * (2 * x - a)) : a = 1 := 
by
  sorry

end find_a_given_solution_l1673_167388


namespace count_three_digit_numbers_with_identical_digits_l1673_167347

/-!
# Problem Statement:
Prove that the number of three-digit numbers with at least two identical digits is 252,
given that three-digit numbers range from 100 to 999.

## Definitions:
- Three-digit numbers are those in the range 100 to 999.

## Theorem:
The number of three-digit numbers with at least two identical digits is 252.
-/
theorem count_three_digit_numbers_with_identical_digits : 
    (∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ 
    ∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ (d1 = d2 ∨ d1 = d3 ∨ d2 = d3)) :=
sorry

end count_three_digit_numbers_with_identical_digits_l1673_167347


namespace percentage_of_ducks_among_non_heron_l1673_167365

def birds_percentage (geese swans herons ducks total_birds : ℕ) : ℕ :=
  let non_heron_birds := total_birds - herons
  let duck_percentage := (ducks * 100) / non_heron_birds
  duck_percentage

theorem percentage_of_ducks_among_non_heron : 
  birds_percentage 28 20 15 32 100 = 37 :=   /- 37 approximates 37.6 -/
sorry

end percentage_of_ducks_among_non_heron_l1673_167365


namespace probability_not_late_probability_late_and_misses_bus_l1673_167346

variable (P_Sam_late : ℚ)
variable (P_miss_bus_given_late : ℚ)

theorem probability_not_late (h1 : P_Sam_late = 5/9) :
  1 - P_Sam_late = 4/9 := by
  rw [h1]
  norm_num

theorem probability_late_and_misses_bus (h1 : P_Sam_late = 5/9) (h2 : P_miss_bus_given_late = 1/3) :
  P_Sam_late * P_miss_bus_given_late = 5/27 := by
  rw [h1, h2]
  norm_num

#check probability_not_late
#check probability_late_and_misses_bus

end probability_not_late_probability_late_and_misses_bus_l1673_167346


namespace union_complement_inter_l1673_167326

noncomputable def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x ≥ 2 }
def N : Set ℝ := { x | -1 ≤ x ∧ x < 5 }

def C_U_M : Set ℝ := U \ M
def M_inter_N : Set ℝ := { x | x ≥ 2 ∧ x < 5 }

theorem union_complement_inter (C_U_M M_inter_N : Set ℝ) :
  C_U_M ∪ M_inter_N = { x | x < 5 } :=
by
  sorry

end union_complement_inter_l1673_167326


namespace intersection_correct_l1673_167321

def A : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x : ℝ | 2 < x ∧ x < 4 }

theorem intersection_correct : A ∩ B = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_correct_l1673_167321


namespace last_number_remaining_l1673_167360

theorem last_number_remaining :
  (∃ f : ℕ → ℕ, ∃ n : ℕ, (∀ k < n, f (2 * k) = 2 * k + 2 ∧
                         ∀ k < n, f (2 * k + 1) = 2 * k + 1 + 2^(k+1)) ∧ 
                         n = 200 ∧ f (2 * n) = 128) :=
sorry

end last_number_remaining_l1673_167360


namespace woman_work_time_l1673_167359

theorem woman_work_time :
  ∀ (M W B : ℝ), (M = 1/6) → (B = 1/12) → (M + W + B = 1/3) → (W = 1/12) → (1 / W = 12) :=
by
  intros M W B hM hB h_combined hW
  sorry

end woman_work_time_l1673_167359


namespace four_b_is_222_22_percent_of_a_l1673_167382

-- noncomputable is necessary because Lean does not handle decimal numbers directly
noncomputable def a (b : ℝ) : ℝ := 1.8 * b
noncomputable def four_b (b : ℝ) : ℝ := 4 * b

theorem four_b_is_222_22_percent_of_a (b : ℝ) : four_b b = 2.2222 * a b := 
by
  sorry

end four_b_is_222_22_percent_of_a_l1673_167382


namespace max_k_divides_expression_l1673_167332

theorem max_k_divides_expression : ∃ k, (∀ n : ℕ, n > 0 → 2^k ∣ (3^(2*n + 3) + 40*n - 27)) ∧ k = 6 :=
sorry

end max_k_divides_expression_l1673_167332


namespace initial_fund_is_890_l1673_167337

-- Given Conditions
def initial_fund (n : ℕ) : ℝ := 60 * n - 10
def bonus_given (n : ℕ) : ℝ := 50 * n
def remaining_fund (initial : ℝ) (bonus : ℝ) : ℝ := initial - bonus

-- Proof problem: Prove that the initial amount equals $890 under the given constraints
theorem initial_fund_is_890 :
  ∃ n : ℕ, 
    initial_fund n = 890 ∧ 
    initial_fund n - bonus_given n = 140 :=
by
  sorry

end initial_fund_is_890_l1673_167337


namespace inequality_am_gm_l1673_167356

theorem inequality_am_gm (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/2) :
  (1 - a) * (1 - b) ≤ 9 / 16 := 
by
  sorry

end inequality_am_gm_l1673_167356


namespace completing_square_solution_l1673_167341

theorem completing_square_solution (x : ℝ) : x^2 + 4 * x - 1 = 0 → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_solution_l1673_167341


namespace stateA_selection_percentage_l1673_167305

theorem stateA_selection_percentage :
  ∀ (P : ℕ), (∀ (n : ℕ), n = 8000) → (7 * 8000 / 100 = P * 8000 / 100 + 80) → P = 6 := by
  -- The proof steps go here
  sorry

end stateA_selection_percentage_l1673_167305


namespace isosceles_right_triangle_quotient_l1673_167311

theorem isosceles_right_triangle_quotient (a : ℝ) (h : a > 0) :
  (2 * a) / (Real.sqrt (a^2 + a^2)) = Real.sqrt 2 :=
sorry

end isosceles_right_triangle_quotient_l1673_167311


namespace no_real_solutions_l1673_167334

theorem no_real_solutions :
  ∀ (x : ℝ), (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) ≠ 1 / 8) :=
by
  intro x
  sorry

end no_real_solutions_l1673_167334


namespace vertex_of_parabola_l1673_167309

-- Definition of the parabola
def parabola (x : ℝ) : ℝ := -2 * (x - 3)^2 - 2

-- The theorem stating the vertex of the parabola
theorem vertex_of_parabola : ∃ h k : ℝ, (h, k) = (2, -5) :=
by
  sorry

end vertex_of_parabola_l1673_167309


namespace vasya_numbers_l1673_167358

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l1673_167358


namespace one_cow_eating_one_bag_in_12_days_l1673_167342

def average_days_to_eat_one_bag (total_bags : ℕ) (total_days : ℕ) (number_of_cows : ℕ) : ℕ :=
  total_days / (total_bags / number_of_cows)

theorem one_cow_eating_one_bag_in_12_days (total_bags : ℕ) (total_days : ℕ) (number_of_cows : ℕ) (h_total_bags : total_bags = 50) (h_total_days : total_days = 20) (h_number_of_cows : number_of_cows = 30) : 
  average_days_to_eat_one_bag total_bags total_days number_of_cows = 12 := by
  sorry

end one_cow_eating_one_bag_in_12_days_l1673_167342


namespace parallel_lines_direction_vector_l1673_167314

theorem parallel_lines_direction_vector (k : ℝ) :
  (∃ c : ℝ, (5, -3) = (c * -2, c * k)) ↔ k = 6 / 5 :=
by sorry

end parallel_lines_direction_vector_l1673_167314


namespace range_of_m_l1673_167374

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < m^2 - m) ↔ m < -1 ∨ m > 2 := 
by
  sorry

end range_of_m_l1673_167374


namespace f_periodic_4_l1673_167383

noncomputable def f : ℝ → ℝ := sorry -- f is some function ℝ → ℝ

theorem f_periodic_4 (h : ∀ x, f x = -f (x + 2)) : f 100 = f 4 := 
by
  sorry

end f_periodic_4_l1673_167383


namespace permutation_problem_l1673_167390

noncomputable def permutation (n r : ℕ) : ℕ := (n.factorial) / ( (n - r).factorial)

theorem permutation_problem : 5 * permutation 5 3 + 4 * permutation 4 2 = 348 := by
  sorry

end permutation_problem_l1673_167390


namespace det_M_pow_three_eq_twenty_seven_l1673_167367

-- Define a matrix M
variables (M : Matrix (Fin n) (Fin n) ℝ)

-- Given condition: det M = 3
axiom det_M_eq_3 : Matrix.det M = 3

-- State the theorem we aim to prove
theorem det_M_pow_three_eq_twenty_seven : Matrix.det (M^3) = 27 :=
by
  sorry

end det_M_pow_three_eq_twenty_seven_l1673_167367


namespace complement_of_B_in_A_l1673_167399

def complement (A B : Set Int) := { x ∈ A | x ∉ B }

theorem complement_of_B_in_A (A B : Set Int) (a : Int) (h1 : A = {2, 3, 4}) (h2 : B = {a + 2, a}) (h3 : A ∩ B = B)
: complement A B = {3} :=
  sorry

end complement_of_B_in_A_l1673_167399


namespace mixture_cost_in_july_l1673_167366

theorem mixture_cost_in_july :
  (∀ C : ℝ, C > 0 → 
    (cost_green_tea_july : ℝ) = 0.1 → 
    (cost_green_tea_july = 0.1 * C) →
    (equal_quantities_mixture:  ℝ) = 1.5 →
    (cost_coffee_july: ℝ) = 2 * C →
    (total_mixture_cost: ℝ) = equal_quantities_mixture * cost_green_tea_july + equal_quantities_mixture * cost_coffee_july →
    total_mixture_cost = 3.15) :=
by
  sorry

end mixture_cost_in_july_l1673_167366


namespace quadratic_inequality_solution_l1673_167339

theorem quadratic_inequality_solution (a b c : ℝ) (h : a < 0) 
  (h_sol : ∀ x, ax^2 + bx + c > 0 ↔ x > -2 ∧ x < 1) :
  ∀ x, ax^2 + (a + b) * x + c - a < 0 ↔ x < -3 ∨ x > 1 := 
sorry

end quadratic_inequality_solution_l1673_167339


namespace sampling_method_is_systematic_l1673_167320

def conveyor_belt_sampling (interval: ℕ) (product_picking: ℕ → ℕ) : Prop :=
  ∀ (n: ℕ), product_picking n = n * interval

theorem sampling_method_is_systematic
  (interval: ℕ)
  (product_picking: ℕ → ℕ)
  (h: conveyor_belt_sampling interval product_picking) :
  interval = 30 → product_picking = systematic_sampling := 
sorry

end sampling_method_is_systematic_l1673_167320


namespace min_value_arithmetic_sequence_l1673_167344

theorem min_value_arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h_arith_seq : a n = 1 + (n - 1) * 1)
  (h_sum : S n = n * (1 + n) / 2) :
  ∃ n, (S n + 8) / a n = 9 / 2 :=
by
  sorry

end min_value_arithmetic_sequence_l1673_167344


namespace sandwich_cost_l1673_167328

theorem sandwich_cost (total_cost soda_cost sandwich_count soda_count : ℝ) :
  total_cost = 8.38 → soda_cost = 0.87 → sandwich_count = 2 → soda_count = 4 → 
  (∀ S, sandwich_count * S + soda_count * soda_cost = total_cost → S = 2.45) :=
by
  intros h_total h_soda h_sandwich_count h_soda_count S h_eqn
  sorry

end sandwich_cost_l1673_167328


namespace greatest_divisor_of_arithmetic_sequence_l1673_167350

theorem greatest_divisor_of_arithmetic_sequence (x c : ℕ) : ∃ d, d = 15 ∧ ∀ S, S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_arithmetic_sequence_l1673_167350


namespace evaluate_g_at_neg2_l1673_167338

def g (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 35 * x^2 - 28 * x - 84

theorem evaluate_g_at_neg2 : g (-2) = 320 := by
  sorry

end evaluate_g_at_neg2_l1673_167338


namespace misha_is_older_l1673_167303

-- Definitions for the conditions
def tanya_age_19_months_ago : ℕ := 16
def months_ago_for_tanya : ℕ := 19
def misha_age_in_16_months : ℕ := 19
def months_ahead_for_misha : ℕ := 16

-- Convert months to years and residual months
def months_to_years_months (m : ℕ) : ℕ × ℕ := (m / 12, m % 12)

-- Computation for Tanya's current age
def tanya_age_now : ℕ × ℕ :=
  let (years, months) := months_to_years_months months_ago_for_tanya
  (tanya_age_19_months_ago + years, months)

-- Computation for Misha's current age
def misha_age_now : ℕ × ℕ :=
  let (years, months) := months_to_years_months months_ahead_for_misha
  (misha_age_in_16_months - years, months)

-- Proof statement
theorem misha_is_older : misha_age_now > tanya_age_now := by
  sorry

end misha_is_older_l1673_167303


namespace find_150th_letter_l1673_167369

theorem find_150th_letter (n : ℕ) (pattern : ℕ → Char) (h : ∀ m, pattern (m % 3) = if m % 3 = 0 then 'A' else if m % 3 = 1 then 'B' else 'C') :
  pattern 149 = 'C' :=
by
  sorry

end find_150th_letter_l1673_167369


namespace parametric_to_standard_l1673_167325

theorem parametric_to_standard (theta : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h1 : x = 1 + 2 * Real.cos theta)
  (h2 : y = -2 + 2 * Real.sin theta) :
  (x - 1)^2 + (y + 2)^2 = 4 :=
sorry

end parametric_to_standard_l1673_167325


namespace sum_p_q_r_l1673_167355

theorem sum_p_q_r :
  ∃ (p q r : ℤ), 
    (∀ x : ℤ, x ^ 2 + 20 * x + 96 = (x + p) * (x + q)) ∧ 
    (∀ x : ℤ, x ^ 2 - 22 * x + 120 = (x - q) * (x - r)) ∧ 
    p + q + r = 30 :=
by 
  sorry

end sum_p_q_r_l1673_167355


namespace real_part_sum_l1673_167361

-- Definitions of a and b as real numbers and i as the imaginary unit
variables (a b : ℝ)
def i := Complex.I

-- Condition given in the problem
def given_condition : Prop := (a + b * i) / (2 - i) = 3 + i

-- Statement to prove
theorem real_part_sum : given_condition a b → a + b = 20 := by
  sorry

end real_part_sum_l1673_167361


namespace triangle_perimeter_l1673_167378

/-- The lengths of two sides of a triangle are 3 and 5 respectively. The third side is a root of the equation x^2 - 7x + 12 = 0. Find the perimeter of the triangle. -/
theorem triangle_perimeter :
  let side1 := 3
  let side2 := 5
  let third_side1 := 3
  let third_side2 := 4
  (third_side1 * third_side1 - 7 * third_side1 + 12 = 0) ∧
  (third_side2 * third_side2 - 7 * third_side2 + 12 = 0) →
  (side1 + side2 + third_side1 = 11 ∨ side1 + side2 + third_side2 = 12) :=
by
  sorry

end triangle_perimeter_l1673_167378


namespace root_sum_product_eq_l1673_167352

theorem root_sum_product_eq (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) :
  p + q = 69 :=
by 
  sorry

end root_sum_product_eq_l1673_167352


namespace buckets_required_l1673_167357

variable (C : ℝ) (N : ℝ)

theorem buckets_required (h : N * C = 105 * (2 / 5) * C) : N = 42 := 
  sorry

end buckets_required_l1673_167357


namespace ounces_per_bowl_l1673_167327

theorem ounces_per_bowl (oz_per_gallon : ℕ) (gallons : ℕ) (bowls_per_minute : ℕ) (minutes : ℕ) (total_ounces : ℕ) (total_bowls : ℕ) (oz_per_bowl : ℕ) : 
  oz_per_gallon = 128 → 
  gallons = 6 →
  bowls_per_minute = 5 →
  minutes = 15 →
  total_ounces = oz_per_gallon * gallons →
  total_bowls = bowls_per_minute * minutes →
  oz_per_bowl = total_ounces / total_bowls →
  round (oz_per_bowl : ℚ) = 10 :=
by
  sorry

end ounces_per_bowl_l1673_167327


namespace polynomial_relation_l1673_167348

def M (m : ℚ) : ℚ := 5 * m^2 - 8 * m + 1
def N (m : ℚ) : ℚ := 4 * m^2 - 8 * m - 1

theorem polynomial_relation (m : ℚ) : M m > N m := by
  sorry

end polynomial_relation_l1673_167348


namespace distinct_integers_integer_expression_l1673_167394

theorem distinct_integers_integer_expression 
  (x y z : ℤ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (n : ℕ) : 
  ∃ k : ℤ, k = (x^n / ((x - y) * (x - z)) + y^n / ((y - x) * (y - z)) + z^n / ((z - x) * (z - y))) := 
sorry

end distinct_integers_integer_expression_l1673_167394


namespace hot_dogs_leftover_l1673_167391

theorem hot_dogs_leftover :
  36159782 % 6 = 2 :=
by
  sorry

end hot_dogs_leftover_l1673_167391


namespace percentage_difference_l1673_167380

theorem percentage_difference (x y : ℝ) (h : x = 12 * y) : (1 - y / x) * 100 = 91.67 :=
by {
  sorry
}

end percentage_difference_l1673_167380


namespace Shinyoung_ate_most_of_cake_l1673_167395

noncomputable def Shinyoung_portion := (1 : ℚ) / 3
noncomputable def Seokgi_portion := (1 : ℚ) / 4
noncomputable def Woong_portion := (1 : ℚ) / 5

theorem Shinyoung_ate_most_of_cake :
  Shinyoung_portion > Seokgi_portion ∧ Shinyoung_portion > Woong_portion := by
  sorry

end Shinyoung_ate_most_of_cake_l1673_167395


namespace common_root_l1673_167306

variable (m x : ℝ)
variable (h₁ : m * x - 1000 = 1021)
variable (h₂ : 1021 * x = m - 1000 * x)

theorem common_root (hx : m * x - 1000 = 1021 ∧ 1021 * x = m - 1000 * x) : m = 2021 ∨ m = -2021 := sorry

end common_root_l1673_167306


namespace expand_polynomial_l1673_167312

theorem expand_polynomial (x : ℂ) : 
  (1 + x^4) * (1 - x^5) * (1 + x^7) = 1 + x^4 - x^5 + x^7 + x^11 - x^9 - x^12 - x^16 := 
sorry

end expand_polynomial_l1673_167312


namespace students_chemistry_or_physics_not_both_l1673_167385

variables (total_chemistry total_both total_physics_only : ℕ)

theorem students_chemistry_or_physics_not_both
  (h1 : total_chemistry = 30)
  (h2 : total_both = 15)
  (h3 : total_physics_only = 18) :
  total_chemistry - total_both + total_physics_only = 33 :=
by
  sorry

end students_chemistry_or_physics_not_both_l1673_167385


namespace quadratic_to_vertex_form_l1673_167330

theorem quadratic_to_vertex_form :
  ∀ (x : ℝ), (x^2 - 2*x + 3 = (x-1)^2 + 2) :=
by intro x; sorry

end quadratic_to_vertex_form_l1673_167330


namespace find_xyz_l1673_167387

theorem find_xyz
  (x y z : ℝ)
  (h1 : x + y + z = 38)
  (h2 : x * y * z = 2002)
  (h3 : 0 < x ∧ x ≤ 11)
  (h4 : z ≥ 14) :
  x = 11 ∧ y = 13 ∧ z = 14 :=
sorry

end find_xyz_l1673_167387


namespace find_x_plus_y_l1673_167353

theorem find_x_plus_y (x y : ℚ) (h1 : |x| + x + y = 12) (h2 : x + |y| - y = 10) : x + y = 26/5 :=
sorry

end find_x_plus_y_l1673_167353


namespace greatest_x_lcm_l1673_167398

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l1673_167398


namespace intersection_A_B_l1673_167349

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2 * x > 0}

-- Prove the intersection of A and B
theorem intersection_A_B :
  (A ∩ B) = {x | x < (3 / 2)} := sorry

end intersection_A_B_l1673_167349


namespace time_comparison_l1673_167379

variable (s : ℝ) (h_pos : s > 0)

noncomputable def t1 : ℝ := 120 / s
noncomputable def t2 : ℝ := 480 / (4 * s)

theorem time_comparison : t1 s = t2 s := by
  rw [t1, t2]
  field_simp [h_pos]
  norm_num
  sorry

end time_comparison_l1673_167379


namespace count_of_valid_four_digit_numbers_l1673_167317

def is_four_digit_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9

def digits_sum_to_twelve (a b c d : ℕ) : Prop :=
  a + b + c + d = 12

def divisible_by_eleven (a b c d : ℕ) : Prop :=
  (a + c - (b + d)) % 11 = 0

theorem count_of_valid_four_digit_numbers : ∃ n : ℕ, n = 20 ∧
  (∀ a b c d : ℕ, is_four_digit_number a b c d →
  digits_sum_to_twelve a b c d →
  divisible_by_eleven a b c d →
  true) :=
sorry

end count_of_valid_four_digit_numbers_l1673_167317


namespace jason_seashells_l1673_167363

theorem jason_seashells (initial_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) 
(h1 : initial_seashells = 49) (h2 : given_seashells = 13) :
remaining_seashells = initial_seashells - given_seashells := by
  sorry

end jason_seashells_l1673_167363


namespace smallest_sum_a_b_l1673_167392

theorem smallest_sum_a_b (a b: ℕ) (h₀: 0 < a) (h₁: 0 < b) (h₂: a ≠ b) (h₃: 1 / (a: ℝ) + 1 / (b: ℝ) = 1 / 15) : a + b = 64 :=
sorry

end smallest_sum_a_b_l1673_167392


namespace angle_bisector_slope_l1673_167371

/-
Given conditions:
1. line1: y = 2x
2. line2: y = 4x
Prove:
k = (sqrt(21) - 6) / 7
-/

theorem angle_bisector_slope :
  let m1 := 2
  let m2 := 4
  let k := (Real.sqrt 21 - 6) / 7
  (1 - m1 * m2) ≠ 0 →
  k = (m1 + m2 - Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)
:=
sorry

end angle_bisector_slope_l1673_167371


namespace range_of_x_for_f_lt_0_l1673_167345

noncomputable def f (x : ℝ) : ℝ := x^2 - x^(1/2)

theorem range_of_x_for_f_lt_0 :
  {x : ℝ | f x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end range_of_x_for_f_lt_0_l1673_167345


namespace existence_of_b_l1673_167354

theorem existence_of_b's (n m : ℕ) (h1 : 1 < n) (h2 : 1 < m) 
  (a : Fin m → ℕ) (h3 : ∀ i, 0 < a i ∧ a i ≤ n^m) :
  ∃ b : Fin m → ℕ, (∀ i, 0 < b i ∧ b i ≤ n) ∧ (∀ i, a i + b i < n) :=
by
  sorry

end existence_of_b_l1673_167354


namespace exists_x_y_not_divisible_by_3_l1673_167333

theorem exists_x_y_not_divisible_by_3 (k : ℕ) (h_pos : 0 < k) :
  ∃ x y : ℤ, (x^2 + 2 * y^2 = 3^k) ∧ (x % 3 ≠ 0) ∧ (y % 3 ≠ 0) := 
sorry

end exists_x_y_not_divisible_by_3_l1673_167333


namespace quadratic_has_real_solution_l1673_167364

theorem quadratic_has_real_solution (a b c : ℝ) : 
  ∃ x : ℝ, x^2 + (a - b) * x + (b - c) = 0 ∨ 
           x^2 + (b - c) * x + (c - a) = 0 ∨ 
           x^2 + (c - a) * x + (a - b) = 0 :=
  sorry

end quadratic_has_real_solution_l1673_167364


namespace solve_m_n_l1673_167377

theorem solve_m_n (m n : ℤ) :
  (m * 1 + n * 1 = 6) ∧ (m * 2 + n * -1 = 6) → (m = 4) ∧ (n = 2) := by
  sorry

end solve_m_n_l1673_167377


namespace find_f_2_l1673_167375

noncomputable def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 - x + 2

theorem find_f_2 (a b : ℝ)
  (h : f a b (-2) = 5) : f a b 2 = -1 :=
by 
  sorry

end find_f_2_l1673_167375


namespace sum_of_cubes_equals_square_l1673_167396

theorem sum_of_cubes_equals_square :
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 := 
by 
  sorry

end sum_of_cubes_equals_square_l1673_167396


namespace pipe_fills_cistern_l1673_167318

theorem pipe_fills_cistern (t : ℕ) (h : t = 5) : 11 * t = 55 :=
by
  sorry

end pipe_fills_cistern_l1673_167318


namespace dining_bill_split_l1673_167310

theorem dining_bill_split (original_bill : ℝ) (num_people : ℕ) (tip_percent : ℝ) (total_bill_with_tip : ℝ) (amount_per_person : ℝ)
  (h1 : original_bill = 139.00)
  (h2 : num_people = 3)
  (h3 : tip_percent = 0.10)
  (h4 : total_bill_with_tip = original_bill + (tip_percent * original_bill))
  (h5 : amount_per_person = total_bill_with_tip / num_people) :
  amount_per_person = 50.97 :=
by 
  sorry

end dining_bill_split_l1673_167310


namespace part1_part2_l1673_167373

-- Problem Part 1
theorem part1 : (-((-8)^(1/3)) - |(3^(1/2) - 2)| + ((-3)^2)^(1/2) + -3^(1/2) = 3) :=
by {
  sorry
}

-- Problem Part 2
theorem part2 (x : ℤ) : (2 * x + 5 ≤ 3 * (x + 2) ∧ 2 * x - (1 + 3 * x) / 2 < 1) ↔ (x = 0 ∨ x = 1 ∨ x = 2) :=
by {
  sorry
}

end part1_part2_l1673_167373


namespace dad_borrowed_nickels_l1673_167372

-- Definitions for the initial and remaining nickels
def initial_nickels : ℕ := 31
def remaining_nickels : ℕ := 11

-- Statement of the problem in Lean
theorem dad_borrowed_nickels : initial_nickels - remaining_nickels = 20 := by
  -- Proof goes here
  sorry

end dad_borrowed_nickels_l1673_167372


namespace min_value_x_plus_9_div_x_l1673_167362

theorem min_value_x_plus_9_div_x (x : ℝ) (hx : x > 0) : x + 9 / x ≥ 6 := by
  -- sorry indicates that the proof is omitted.
  sorry

end min_value_x_plus_9_div_x_l1673_167362


namespace juice_difference_proof_l1673_167381

def barrel_initial_A := 10
def barrel_initial_B := 8
def transfer_amount := 3

def barrel_final_A := barrel_initial_A + transfer_amount
def barrel_final_B := barrel_initial_B - transfer_amount

def juice_difference := barrel_final_A - barrel_final_B

theorem juice_difference_proof : juice_difference = 8 := by
  sorry

end juice_difference_proof_l1673_167381


namespace bus_stop_time_l1673_167304

theorem bus_stop_time (speed_without_stoppages speed_with_stoppages : ℝ) (h1: speed_without_stoppages = 48) (h2: speed_with_stoppages = 24) :
  ∃ (minutes_stopped_per_hour : ℝ), minutes_stopped_per_hour = 30 :=
by
  sorry

end bus_stop_time_l1673_167304


namespace buying_ways_l1673_167393

theorem buying_ways (students : ℕ) (choices : ℕ) (at_least_one_pencil : ℕ) : 
  students = 4 ∧ choices = 2 ∧ at_least_one_pencil = 1 → 
  (choices^students - 1) = 15 :=
by
  sorry

end buying_ways_l1673_167393


namespace total_weight_of_8_bags_total_sales_amount_of_qualified_products_l1673_167336

-- Definitions
def deviations : List ℤ := [-6, -3, -2, 0, 1, 4, 5, -1]
def standard_weight_per_bag : ℤ := 450
def threshold : ℤ := 4
def price_per_bag : ℤ := 3

-- Part 1: Total weight of the 8 bags of laundry detergent
theorem total_weight_of_8_bags : 
  8 * standard_weight_per_bag + deviations.sum = 3598 := 
by
  sorry

-- Part 2: Total sales amount of qualified products
theorem total_sales_amount_of_qualified_products : 
  price_per_bag * (deviations.filter (fun x => abs x ≤ threshold)).length = 18 := 
by
  sorry

end total_weight_of_8_bags_total_sales_amount_of_qualified_products_l1673_167336


namespace line_perpendicular_to_plane_l1673_167319

open Classical

-- Define the context of lines and planes.
variables {Line : Type} {Plane : Type}

-- Define the perpendicular and parallel relations.
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l₁ l₂ : Line) : Prop := sorry

-- Declare the distinct lines and non-overlapping planes.
variable {m n : Line}
variable {α β : Plane}

-- State the theorem.
theorem line_perpendicular_to_plane (h1 : parallel m n) (h2 : perpendicular n β) : perpendicular m β :=
sorry

end line_perpendicular_to_plane_l1673_167319


namespace max_min_y_l1673_167397

noncomputable def y (x : ℝ) : ℝ := (Real.sin x)^(2:ℝ) + 2 * (Real.sin x) * (Real.cos x) + 3 * (Real.cos x)^(2:ℝ)

theorem max_min_y : 
  ∀ x : ℝ, 
  2 - Real.sqrt 2 ≤ y x ∧ y x ≤ 2 + Real.sqrt 2 :=
by sorry

end max_min_y_l1673_167397


namespace factor_expression_l1673_167308

theorem factor_expression (c : ℝ) : 270 * c^2 + 45 * c - 15 = 15 * c * (18 * c + 2) :=
by
  sorry

end factor_expression_l1673_167308


namespace min_fencing_l1673_167386

variable (w l : ℝ)

noncomputable def area := w * l

noncomputable def length := 2 * w

theorem min_fencing (h1 : area w l ≥ 500) (h2 : l = length w) : 
  w = 5 * Real.sqrt 10 ∧ l = 10 * Real.sqrt 10 :=
  sorry

end min_fencing_l1673_167386


namespace part_a_part_b_part_c_l1673_167340

-- Part (a)
theorem part_a (x y : ℕ) (h : (2 * x + 11 * y) = 3 * x + 4 * y) : x = 7 * y := by
  sorry

-- Part (b)
theorem part_b (u v : ℚ) : ∃ (x y : ℚ), (x + y) / 2 = (u.num * v.den + v.num * u.den) / (2 * u.den * v.den) := by
  sorry

-- Part (c)
theorem part_c (u v : ℚ) (h : u < v) : ∀ (m : ℚ), (m.num = u.num + v.num) ∧ (m.den = u.den + v.den) → u < m ∧ m < v := by
  sorry

end part_a_part_b_part_c_l1673_167340


namespace tetrahedron_condition_proof_l1673_167335

/-- Define the conditions for the necessary and sufficient condition for each k -/
def tetrahedron_condition (a : ℝ) (k : ℕ) : Prop :=
  match k with
  | 1 => a < Real.sqrt 3
  | 2 => Real.sqrt (2 - Real.sqrt 3) < a ∧ a < Real.sqrt (2 + Real.sqrt 3)
  | 3 => a < Real.sqrt 3
  | 4 => a > Real.sqrt (2 - Real.sqrt 3)
  | 5 => a > 1 / Real.sqrt 3
  | _ => False -- not applicable for other values of k

/-- Prove that the condition is valid for given a and k -/
theorem tetrahedron_condition_proof (a : ℝ) (k : ℕ) : tetrahedron_condition a k := 
  by
  sorry

end tetrahedron_condition_proof_l1673_167335


namespace program_output_l1673_167370

theorem program_output (x : ℤ) : 
  (if x < 0 then -1 else if x = 0 then 0 else 1) = 1 ↔ x = 3 :=
by
  sorry

end program_output_l1673_167370


namespace A_E_not_third_l1673_167316

-- Define the runners and their respective positions.
inductive Runner
| A : Runner
| B : Runner
| C : Runner
| D : Runner
| E : Runner
open Runner

variable (position : Runner → Nat)

-- Conditions
axiom A_beats_B : position A < position B
axiom C_beats_D : position C < position D
axiom B_beats_E : position B < position E
axiom D_after_A_before_B : position A < position D ∧ position D < position B

-- Prove that A and E cannot be in third place.
theorem A_E_not_third : position A ≠ 3 ∧ position E ≠ 3 :=
sorry

end A_E_not_third_l1673_167316


namespace probability_two_white_balls_l1673_167343

def bagA := [1, 1]
def bagB := [2, 1]

def total_outcomes := 6
def favorable_outcomes := 2

theorem probability_two_white_balls : (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 :=
by
  sorry

end probability_two_white_balls_l1673_167343


namespace price_of_turban_correct_l1673_167331

noncomputable def initial_yearly_salary : ℝ := 90
noncomputable def initial_monthly_salary : ℝ := initial_yearly_salary / 12
noncomputable def raise : ℝ := 0.05 * initial_monthly_salary

noncomputable def first_3_months_salary : ℝ := 3 * initial_monthly_salary
noncomputable def second_3_months_salary : ℝ := 3 * (initial_monthly_salary + raise)
noncomputable def third_3_months_salary : ℝ := 3 * (initial_monthly_salary + 2 * raise)

noncomputable def total_cash_salary : ℝ := first_3_months_salary + second_3_months_salary + third_3_months_salary
noncomputable def actual_cash_received : ℝ := 80
noncomputable def price_of_turban : ℝ := actual_cash_received - total_cash_salary

theorem price_of_turban_correct : price_of_turban = 9.125 :=
by
  sorry

end price_of_turban_correct_l1673_167331


namespace slope_of_regression_line_l1673_167323

variable (h : ℝ)
variable (t1 T1 t2 T2 t3 T3 : ℝ)

-- Given conditions.
axiom t2_is_equally_spaced : t2 = t1 + h
axiom t3_is_equally_spaced : t3 = t1 + 2 * h

theorem slope_of_regression_line :
  t2 = t1 + h →
  t3 = t1 + 2 * h →
  (T3 - T1) / (t3 - t1) = (T3 - T1) / ((t1 + 2 * h) - t1) := 
by
  sorry

end slope_of_regression_line_l1673_167323


namespace number_of_sixes_l1673_167313

theorem number_of_sixes
  (total_runs : ℕ)
  (boundaries : ℕ)
  (percent_runs_by_running : ℚ)
  (runs_per_boundary : ℕ)
  (runs_per_six : ℕ)
  (runs_by_running : ℚ)
  (runs_by_boundaries : ℕ)
  (runs_by_sixes : ℕ)
  (number_of_sixes : ℕ)
  (h1 : total_runs = 120)
  (h2 : boundaries = 6)
  (h3 : percent_runs_by_running = 0.6)
  (h4 : runs_per_boundary = 4)
  (h5 : runs_per_six = 6)
  (h6 : runs_by_running = percent_runs_by_running * total_runs)
  (h7 : runs_by_boundaries = boundaries * runs_per_boundary)
  (h8 : runs_by_sixes = total_runs - (runs_by_running + runs_by_boundaries))
  (h9 : number_of_sixes = runs_by_sixes / runs_per_six)
  : number_of_sixes = 4 :=
by
  sorry

end number_of_sixes_l1673_167313


namespace central_angle_of_sector_l1673_167384

theorem central_angle_of_sector (r : ℝ) (θ : ℝ) (h_perimeter: 2 * r + θ * r = π * r / 2) : θ = π - 2 :=
sorry

end central_angle_of_sector_l1673_167384


namespace percent_of_x_is_y_l1673_167389

variable (x y : ℝ)

theorem percent_of_x_is_y
  (h : 0.5 * (x - y) = 0.2 * (x + y)) :
  y = 0.4286 * x := by
  sorry

end percent_of_x_is_y_l1673_167389


namespace sum_of_roots_l1673_167376

variable (x1 x2 k m : ℝ)
variable (h1 : x1 ≠ x2)
variable (h2 : 4 * x1^2 - k * x1 = m)
variable (h3 : 4 * x2^2 - k * x2 = m)

theorem sum_of_roots (x1 x2 k m : ℝ) (h1 : x1 ≠ x2)
  (h2 : 4 * x1 ^ 2 - k * x1 = m) (h3 : 4 * x2 ^ 2 - k * x2 = m) :
  x1 + x2 = k / 4 := sorry

end sum_of_roots_l1673_167376


namespace evaluate_expression_to_zero_l1673_167300

-- Assuming 'm' is an integer with specific constraints and providing a proof that the expression evaluates to 0 when m = -1
theorem evaluate_expression_to_zero (m : ℤ) (h1 : -2 ≤ m) (h2 : m ≤ 2) (h3 : m ≠ 0) (h4 : m ≠ 1) (h5 : m ≠ 2) (h6 : m ≠ -2) : 
  (m = -1) → ((m / (m - 2) - 4 / (m ^ 2 - 2 * m)) / (m + 2) / (m ^ 2 - m)) = 0 := 
by
  intro hm_eq_neg1
  sorry

end evaluate_expression_to_zero_l1673_167300


namespace puppy_weight_l1673_167324

theorem puppy_weight (a b c : ℕ) 
  (h1 : a + b + c = 24) 
  (h2 : a + c = 2 * b) 
  (h3 : a + b = c) : 
  a = 4 :=
sorry

end puppy_weight_l1673_167324


namespace parallelogram_properties_l1673_167368

noncomputable def length_adjacent_side_and_area (base height : ℝ) (angle : ℕ) : ℝ × ℝ :=
  let hypotenuse := height / Real.sin (angle * Real.pi / 180)
  let area := base * height
  (hypotenuse, area)

theorem parallelogram_properties :
  ∀ (base height : ℝ) (angle : ℕ),
  base = 12 → height = 6 → angle = 30 →
  length_adjacent_side_and_area base height angle = (12, 72) :=
by
  intros
  sorry

end parallelogram_properties_l1673_167368


namespace number_of_zeros_f_l1673_167322

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^2 - x - 1

-- The theorem statement that proves the function has exactly two zeros
theorem number_of_zeros_f : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end number_of_zeros_f_l1673_167322


namespace longer_diagonal_of_rhombus_l1673_167301

theorem longer_diagonal_of_rhombus {a b d1 : ℕ} (h1 : a = b) (h2 : a = 65) (h3 : d1 = 60) : 
  ∃ d2, (d2^2) = (2 * (a^2) - (d1^2)) ∧ d2 = 110 :=
by
  sorry

end longer_diagonal_of_rhombus_l1673_167301


namespace g_at_2_eq_9_l1673_167302

def g (x : ℝ) : ℝ := x^2 + 3 * x - 1

theorem g_at_2_eq_9 : g 2 = 9 := by
  sorry

end g_at_2_eq_9_l1673_167302


namespace max_difference_second_largest_second_smallest_l1673_167329

theorem max_difference_second_largest_second_smallest :
  ∀ (a b c d e f g h : ℕ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧ g < h ∧
  a + b + c = 27 ∧
  a + b + c + d + e + f + g + h = 152 ∧
  f + g + h = 87 →
  g - b = 26 :=
by
  intros;
  sorry

end max_difference_second_largest_second_smallest_l1673_167329
