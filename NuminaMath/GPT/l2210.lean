import Mathlib

namespace even_function_has_zero_coefficient_l2210_221057

theorem even_function_has_zero_coefficient (a : ℝ) :
  (∀ x : ℝ, (x^2 + a*x) = (x^2 + a*(-x))) → a = 0 :=
by
  intro h
  -- the proof part is omitted as requested
  sorry

end even_function_has_zero_coefficient_l2210_221057


namespace lemon_heads_each_person_l2210_221080

-- Define the constants used in the problem
def totalLemonHeads : Nat := 72
def numberOfFriends : Nat := 6

-- The theorem stating the problem and the correct answer
theorem lemon_heads_each_person :
  totalLemonHeads / numberOfFriends = 12 := 
by
  sorry

end lemon_heads_each_person_l2210_221080


namespace point_D_is_on_y_axis_l2210_221016

def is_on_y_axis (p : ℝ × ℝ) : Prop := p.fst = 0

def point_A : ℝ × ℝ := (3, 0)
def point_B : ℝ × ℝ := (1, 2)
def point_C : ℝ × ℝ := (2, 1)
def point_D : ℝ × ℝ := (0, -3)

theorem point_D_is_on_y_axis : is_on_y_axis point_D :=
by
  sorry

end point_D_is_on_y_axis_l2210_221016


namespace factorization_identity_l2210_221017

noncomputable def factor_expression (a b c : ℝ) : ℝ :=
  ((a ^ 2 + 1 - (b ^ 2 + 1)) ^ 3 + ((b ^ 2 + 1) - (c ^ 2 + 1)) ^ 3 + ((c ^ 2 + 1) - (a ^ 2 + 1)) ^ 3) /
  ((a - b) ^ 3 + (b - c) ^ 3 + (c - a) ^ 3)

theorem factorization_identity (a b c : ℝ) : 
  factor_expression a b c = (a + b) * (b + c) * (c + a) := 
by 
  sorry

end factorization_identity_l2210_221017


namespace percent_problem_l2210_221046

theorem percent_problem (x : ℝ) (h : 0.35 * 400 = 0.20 * x) : x = 700 :=
by sorry

end percent_problem_l2210_221046


namespace algebraic_expression_l2210_221042

-- Define a variable x
variable (x : ℝ)

-- State the theorem
theorem algebraic_expression : (5 * x - 3) = 5 * x - 3 :=
by
  sorry

end algebraic_expression_l2210_221042


namespace find_annual_interest_rate_l2210_221001

variable (r : ℝ) -- The annual interest rate we want to prove

-- Define the conditions based on the problem statement
variable (I : ℝ := 300) -- interest earned
variable (P : ℝ := 10000) -- principal amount
variable (t : ℝ := 9 / 12) -- time in years

-- Define the simple interest formula condition
def simple_interest_formula : Prop :=
  I = P * r * t

-- The statement to prove
theorem find_annual_interest_rate : simple_interest_formula r ↔ r = 0.04 :=
  by
    unfold simple_interest_formula
    simp
    sorry

end find_annual_interest_rate_l2210_221001


namespace find_f_13_l2210_221088

noncomputable def f : ℕ → ℕ :=
  sorry

axiom condition1 (x : ℕ) : f (x + f x) = 3 * f x
axiom condition2 : f 1 = 3

theorem find_f_13 : f 13 = 27 :=
  sorry

end find_f_13_l2210_221088


namespace proof_problem_l2210_221059

-- Define the propositions and conditions
def p : Prop := ∀ x > 0, 3^x > 1
def neg_p : Prop := ∃ x > 0, 3^x ≤ 1
def q (a : ℝ) : Prop := a < -2
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

-- The condition that q is a sufficient condition for f(x) to have a zero in [-1,2]
def has_zero_in_interval (a : ℝ) : Prop := 
  (-a + 3) * (2 * a + 3) ≤ 0

-- The proof problem statement
theorem proof_problem (a : ℝ) (P : p) (Q : has_zero_in_interval a) : ¬ p ∧ q a :=
by
  sorry

end proof_problem_l2210_221059


namespace total_distance_to_run_l2210_221092

theorem total_distance_to_run
  (track_length : ℕ)
  (initial_laps : ℕ)
  (additional_laps : ℕ)
  (total_laps := initial_laps + additional_laps) :
  track_length = 150 →
  initial_laps = 6 →
  additional_laps = 4 →
  total_laps * track_length = 1500 := by
  sorry

end total_distance_to_run_l2210_221092


namespace cakes_difference_l2210_221076

theorem cakes_difference (cakes_bought cakes_sold : ℕ) (h1 : cakes_bought = 139) (h2 : cakes_sold = 145) : cakes_sold - cakes_bought = 6 :=
by
  sorry

end cakes_difference_l2210_221076


namespace infinitely_many_n_divide_2n_plus_1_l2210_221064

theorem infinitely_many_n_divide_2n_plus_1 :
    ∃ (S : Set ℕ), (∀ n ∈ S, n > 0 ∧ n ∣ (2 * n + 1)) ∧ Set.Infinite S :=
by
  sorry

end infinitely_many_n_divide_2n_plus_1_l2210_221064


namespace total_weight_moved_l2210_221009

-- Define the given conditions as Lean definitions
def weight_per_rep : ℕ := 15
def number_of_reps : ℕ := 10
def number_of_sets : ℕ := 3

-- Define the theorem to prove total weight moved is 450 pounds
theorem total_weight_moved : weight_per_rep * number_of_reps * number_of_sets = 450 := by
  sorry

end total_weight_moved_l2210_221009


namespace value_of_a_minus_b_l2210_221021

theorem value_of_a_minus_b (a b : ℝ) :
  (∀ x, - (1 / 2 : ℝ) < x ∧ x < (1 / 3 : ℝ) → ax^2 + bx + 2 > 0) → a - b = -10 := by
sorry

end value_of_a_minus_b_l2210_221021


namespace scientific_notation_l2210_221096

theorem scientific_notation (h : 0.0000046 = 4.6 * 10^(-6)) : True :=
by 
  sorry

end scientific_notation_l2210_221096


namespace quadratic_equation_with_roots_sum_and_difference_l2210_221020

theorem quadratic_equation_with_roots_sum_and_difference (p q : ℚ)
  (h1 : p + q = 10)
  (h2 : abs (p - q) = 2) :
  (Polynomial.eval₂ (RingHom.id ℚ) p (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C 24) = 0) ∧
  (Polynomial.eval₂ (RingHom.id ℚ) q (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C 24) = 0) :=
by sorry

end quadratic_equation_with_roots_sum_and_difference_l2210_221020


namespace train_crossing_time_l2210_221074

theorem train_crossing_time
  (length_of_train : ℝ)
  (speed_in_kmh : ℝ)
  (speed_in_mps : ℝ)
  (conversion_factor : ℝ)
  (time : ℝ)
  (h1 : length_of_train = 160)
  (h2 : speed_in_kmh = 36)
  (h3 : conversion_factor = 1 / 3.6)
  (h4 : speed_in_mps = speed_in_kmh * conversion_factor)
  (h5 : time = length_of_train / speed_in_mps) : time = 16 :=
by
  sorry

end train_crossing_time_l2210_221074


namespace systematic_sampling_result_l2210_221004

-- Define the set of bags numbered from 1 to 30
def bags : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the systematic sampling function
def systematic_sampling (n k interval : ℕ) : List ℕ :=
  List.range k |> List.map (λ i => n + i * interval)

-- Specific parameters for the problem
def number_of_bags := 30
def bags_drawn := 6
def interval := 5
def expected_samples := [2, 7, 12, 17, 22, 27]

-- Statement of the theorem
theorem systematic_sampling_result : 
  systematic_sampling 2 bags_drawn interval = expected_samples :=
by
  sorry

end systematic_sampling_result_l2210_221004


namespace quadratic_ineq_solution_set_l2210_221082

theorem quadratic_ineq_solution_set {m : ℝ} :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 :=
sorry

end quadratic_ineq_solution_set_l2210_221082


namespace temperature_at_night_is_minus_two_l2210_221079

theorem temperature_at_night_is_minus_two (temperature_noon temperature_afternoon temperature_drop_by_night temperature_night : ℤ) : 
  temperature_noon = 5 → temperature_afternoon = 7 → temperature_drop_by_night = 9 → 
  temperature_night = temperature_afternoon - temperature_drop_by_night → 
  temperature_night = -2 := 
by
  intros h1 h2 h3 h4
  rw [h2, h3] at h4
  exact h4


end temperature_at_night_is_minus_two_l2210_221079


namespace problem_solution_l2210_221061

theorem problem_solution (a b : ℝ) (h1 : b > a) (h2 : a > 0) :
  a^2 < b^2 ∧ ab < b^2 :=
sorry

end problem_solution_l2210_221061


namespace find_m_n_l2210_221062

theorem find_m_n (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_sol : (m + Real.sqrt n)^2 - 10 * (m + Real.sqrt n) + 1 = Real.sqrt (m + Real.sqrt n) * (m + Real.sqrt n + 1)) : m + n = 55 :=
sorry

end find_m_n_l2210_221062


namespace sum_of_coefficients_l2210_221002

-- Define the polynomial P(x)
def P (x : ℤ) : ℤ := (2 * x^2021 - x^2020 + x^2019)^11 - 29

-- State the theorem we intend to prove
theorem sum_of_coefficients : P 1 = 2019 :=
by
  -- Proof omitted
  sorry

end sum_of_coefficients_l2210_221002


namespace sum_abcd_l2210_221045

variable {a b c d : ℚ}

theorem sum_abcd 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 4) : 
  a + b + c + d = -2/3 :=
by sorry

end sum_abcd_l2210_221045


namespace simplify_expression_l2210_221048

noncomputable def simplify_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) ^ Real.sqrt 3

theorem simplify_expression :
  (Real.sqrt 2 - 1) ^ (2 - Real.sqrt 3) / (Real.sqrt 2 + 1) ^ (2 + Real.sqrt 3) = simplify_expr :=
by
  sorry

end simplify_expression_l2210_221048


namespace distinct_real_numbers_inequality_l2210_221041

theorem distinct_real_numbers_inequality
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  ( (2 * a - b) / (a - b) )^2 + ( (2 * b - c) / (b - c) )^2 + ( (2 * c - a) / (c - a) )^2 ≥ 5 :=
by {
    sorry
}

end distinct_real_numbers_inequality_l2210_221041


namespace k_n_sum_l2210_221040

theorem k_n_sum (k n : ℕ) (x y : ℕ):
  2 * x^k * y^(k+2) + 3 * x^2 * y^n = 5 * x^2 * y^n → k + n = 6 :=
by sorry

end k_n_sum_l2210_221040


namespace values_of_quadratic_expression_l2210_221011

variable {x : ℝ}

theorem values_of_quadratic_expression (h : x^2 - 4 * x + 3 < 0) : 
  (8 < x^2 + 4 * x + 3) ∧ (x^2 + 4 * x + 3 < 24) :=
sorry

end values_of_quadratic_expression_l2210_221011


namespace complement_intersection_l2210_221039

open Set

theorem complement_intersection {x : ℝ} :
  (x ∉ {x | -2 ≤ x ∧ x ≤ 2}) ∧ (x < 1) ↔ (x < -2) := 
by
  sorry

end complement_intersection_l2210_221039


namespace parallel_lines_a_eq_neg1_l2210_221035

theorem parallel_lines_a_eq_neg1 (a : ℝ) :
  ∀ (x y : ℝ), 
    (x + a * y + 6 = 0) ∧ ((a - 2) * x + 3 * y + 2 * a = 0) →
    (-1 / a = - (a - 2) / 3) → 
    a = -1 :=
by
  sorry

end parallel_lines_a_eq_neg1_l2210_221035


namespace inequality_system_solution_l2210_221049

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l2210_221049


namespace fruit_shop_apples_l2210_221027

-- Given conditions
def morning_fraction : ℚ := 3 / 10
def afternoon_fraction : ℚ := 4 / 10
def total_sold : ℕ := 140

-- Define the total number of apples and the resulting condition
def total_fraction_sold : ℚ := morning_fraction + afternoon_fraction

theorem fruit_shop_apples (A : ℕ) (h : total_fraction_sold * A = total_sold) : A = 200 := 
by sorry

end fruit_shop_apples_l2210_221027


namespace orange_ratio_l2210_221006

theorem orange_ratio (total_oranges : ℕ) (brother_fraction : ℚ) (friend_receives : ℕ)
  (H1 : total_oranges = 12)
  (H2 : friend_receives = 2)
  (H3 : 1 / 4 * ((1 - brother_fraction) * total_oranges) = friend_receives) :
  brother_fraction * total_oranges / total_oranges = 1 / 3 :=
by
  sorry

end orange_ratio_l2210_221006


namespace expressions_equal_iff_l2210_221056

variable (a b c : ℝ)

theorem expressions_equal_iff :
  a^2 + b*c = (a - b)*(a - c) ↔ a = 0 ∨ b + c = 0 :=
by
  sorry

end expressions_equal_iff_l2210_221056


namespace domain_correct_l2210_221037

def domain_of_function (x : ℝ) : Prop :=
  (∃ y : ℝ, y = 2 / Real.sqrt (x + 1)) ∧ Real.sqrt (x + 1) ≠ 0

theorem domain_correct (x : ℝ) : domain_of_function x ↔ (x > -1) := by
  sorry

end domain_correct_l2210_221037


namespace product_divisible_by_3_l2210_221023

noncomputable def dice_prob_divisible_by_3 (n : ℕ) (faces : List ℕ) : ℚ := 
  let probability_div_3 := (1 / 3 : ℚ)
  let probability_not_div_3 := (2 / 3 : ℚ)
  1 - probability_not_div_3 ^ n

theorem product_divisible_by_3 (faces : List ℕ) (h_faces : faces = [1, 2, 3, 4, 5, 6]) :
  dice_prob_divisible_by_3 6 faces = 665 / 729 := 
  by 
    sorry

end product_divisible_by_3_l2210_221023


namespace num_solutions_l2210_221014

-- Let x be a real number
variable (x : ℝ)

-- Define the given equation
def equation := (x^2 - 4) * (x^2 - 1) = (x^2 + 3*x + 2) * (x^2 - 8*x + 7)

-- Theorem: The number of values of x that satisfy the equation is 3
theorem num_solutions : ∃ (S : Finset ℝ), (∀ x, x ∈ S ↔ equation x) ∧ S.card = 3 := 
by
  sorry

end num_solutions_l2210_221014


namespace fish_worth_apples_l2210_221085

-- Defining the variables
variables (f l r a : ℝ)

-- Conditions based on the problem
def condition1 : Prop := 5 * f = 3 * l
def condition2 : Prop := l = 6 * r
def condition3 : Prop := 3 * r = 2 * a

-- The statement of the problem
theorem fish_worth_apples (h1 : condition1 f l) (h2 : condition2 l r) (h3 : condition3 r a) : f = 12 / 5 * a :=
by
  sorry

end fish_worth_apples_l2210_221085


namespace inequality_conditions_l2210_221065

theorem inequality_conditions (x y z : ℝ) (h1 : y - x < 1.5 * abs x) (h2 : z = 2 * (y + x)) : 
  (x ≥ 0 → z < 7 * x) ∧ (x < 0 → z < 0) :=
by
  sorry

end inequality_conditions_l2210_221065


namespace Meadowood_problem_l2210_221066

theorem Meadowood_problem (s h : ℕ) : ¬(26 * s + 3 * h = 58) :=
sorry

end Meadowood_problem_l2210_221066


namespace exists_third_degree_poly_with_positive_and_negative_roots_l2210_221081

theorem exists_third_degree_poly_with_positive_and_negative_roots :
  ∃ (P : ℝ → ℝ), (∃ x : ℝ, P x = 0 ∧ x > 0) ∧ (∃ y : ℝ, (deriv P) y = 0 ∧ y < 0) :=
sorry

end exists_third_degree_poly_with_positive_and_negative_roots_l2210_221081


namespace sum_SHE_equals_6_l2210_221018

-- Definitions for conditions
variables {S H E : ℕ}

-- Conditions as stated in the problem
def distinct_non_zero_digits (S H E : ℕ) : Prop :=
  S ≠ H ∧ H ≠ E ∧ S ≠ E ∧ 1 ≤ S ∧ S < 8 ∧ 1 ≤ H ∧ H < 8 ∧ 1 ≤ E ∧ E < 8

-- Base 8 addition problem
def addition_holds_in_base8 (S H E : ℕ) : Prop :=
  (E + H + (S + E + H) / 8) % 8 = S ∧    -- First column carry
  (H + S + (E + H + S) / 8) % 8 = E ∧    -- Second column carry
  (S + E + (H + S + E) / 8) % 8 = H      -- Third column carry

-- Final statement
theorem sum_SHE_equals_6 :
  distinct_non_zero_digits S H E → addition_holds_in_base8 S H E → S + H + E = 6 :=
by sorry

end sum_SHE_equals_6_l2210_221018


namespace total_cost_of_apples_and_bananas_l2210_221043

variable (a b : ℝ)

theorem total_cost_of_apples_and_bananas (a b : ℝ) : 2 * a + 3 * b = 2 * a + 3 * b :=
by
  sorry

end total_cost_of_apples_and_bananas_l2210_221043


namespace arithmetic_sum_l2210_221090

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n * d)

def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sum :
  ∀ (a d : ℕ),
  arithmetic_sequence a d 2 + arithmetic_sequence a d 3 + arithmetic_sequence a d 4 = 12 →
  sum_first_n_terms a d 7 = 28 :=
by
  sorry

end arithmetic_sum_l2210_221090


namespace min_value_of_quartic_function_l2210_221073

theorem min_value_of_quartic_function : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ (∀ y : ℝ, (0 ≤ y ∧ y ≤ 1) → x^4 + (1 - x)^4 ≤ y^4 + (1 - y)^4) ∧ (x^4 + (1 - x)^4 = 1 / 8) :=
by
  sorry

end min_value_of_quartic_function_l2210_221073


namespace average_value_of_x_l2210_221000

theorem average_value_of_x
  (x : ℝ)
  (h : (5 + 5 + x + 6 + 8) / 5 = 6) :
  x = 6 :=
sorry

end average_value_of_x_l2210_221000


namespace triangle_ratio_perimeter_l2210_221015

theorem triangle_ratio_perimeter (AC BC : ℝ) (CD : ℝ) (AB : ℝ) (m n : ℕ) :
  AC = 15 → BC = 20 → AB = 25 → CD = 10 * Real.sqrt 3 →
  gcd m n = 1 → (2 * Real.sqrt ((AC * BC) / AB) + AB) / AB = m / n → m + n = 7 :=
by
  intros hAC hBC hAB hCD hmn hratio
  sorry

end triangle_ratio_perimeter_l2210_221015


namespace wheel_rpm_is_approximately_5000_23_l2210_221054

noncomputable def bus_wheel_rpm (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let speed_cm_per_min := (speed * 1000 * 100) / 60
  speed_cm_per_min / circumference

-- Conditions
def radius := 35
def speed := 66

-- Question (to be proved)
theorem wheel_rpm_is_approximately_5000_23 : 
  abs (bus_wheel_rpm radius speed - 5000.23) < 0.01 :=
by
  sorry

end wheel_rpm_is_approximately_5000_23_l2210_221054


namespace surface_area_of_cylinder_l2210_221036

noncomputable def cylinder_surface_area
    (r : ℝ) (V : ℝ) (S : ℝ) : Prop :=
    r = 1 ∧ V = 2 * Real.pi ∧ S = 6 * Real.pi

theorem surface_area_of_cylinder
    (r : ℝ) (V : ℝ) : ∃ S : ℝ, cylinder_surface_area r V S :=
by
  use 6 * Real.pi
  sorry

end surface_area_of_cylinder_l2210_221036


namespace solve_for_x_l2210_221071

def F (x y z : ℝ) : ℝ := x * y^3 + z^2

theorem solve_for_x :
  F x 3 2 = F x 2 5 → x = 21/19 :=
  by
  sorry

end solve_for_x_l2210_221071


namespace repeating_decimal_to_fraction_l2210_221026

theorem repeating_decimal_to_fraction 
  (h : ∀ {x : ℝ}, (0.01 : ℝ) = 1 / 99 → x = 1.06 → (0.06 : ℝ) = 6 * 1 / 99): 
  1.06 = 35 / 33 :=
by sorry

end repeating_decimal_to_fraction_l2210_221026


namespace circle_area_increase_l2210_221060

theorem circle_area_increase (r : ℝ) :
  let A_initial := Real.pi * r^2
  let A_new := Real.pi * (2*r)^2
  let delta_A := A_new - A_initial
  let percentage_increase := (delta_A / A_initial) * 100
  percentage_increase = 300 := by
  sorry

end circle_area_increase_l2210_221060


namespace largest_multiple_of_7_less_than_neg_100_l2210_221068

theorem largest_multiple_of_7_less_than_neg_100 : 
  ∃ (x : ℤ), (∃ n : ℤ, x = 7 * n) ∧ x < -100 ∧ ∀ y : ℤ, (∃ m : ℤ, y = 7 * m) ∧ y < -100 → y ≤ x :=
by
  sorry

end largest_multiple_of_7_less_than_neg_100_l2210_221068


namespace ohara_triple_example_l2210_221084

noncomputable def is_ohara_triple (a b x : ℕ) : Prop := 
  (Real.sqrt a + Real.sqrt b = x)

theorem ohara_triple_example : 
  is_ohara_triple 49 16 11 ∧ 11 ≠ 100 / 5 := 
by
  sorry

end ohara_triple_example_l2210_221084


namespace carries_jellybeans_l2210_221034

/-- Bert's box holds 150 jellybeans. --/
def bert_jellybeans : ℕ := 150

/-- Carrie's box is three times as high, three times as wide, and three times as long as Bert's box. --/
def volume_ratio : ℕ := 27

/-- Given that Carrie's box dimensions are three times those of Bert's and Bert's box holds 150 jellybeans, 
    we need to prove that Carrie's box holds 4050 jellybeans. --/
theorem carries_jellybeans : bert_jellybeans * volume_ratio = 4050 := 
by sorry

end carries_jellybeans_l2210_221034


namespace base_five_to_base_ten_modulo_seven_l2210_221008

-- Define the base five number 21014_5 as the corresponding base ten conversion
def base_five_number : ℕ := 2 * 5^4 + 1 * 5^3 + 0 * 5^2 + 1 * 5^1 + 4 * 5^0

-- The equivalent base ten result
def base_ten_number : ℕ := 1384

-- Verify the base ten equivalent of 21014_5
theorem base_five_to_base_ten : base_five_number = base_ten_number :=
by
  -- The expected proof should compute the value of base_five_number
  -- and check that it equals 1384
  sorry

-- Find the modulo operation result of 1384 % 7
def modulo_seven_result : ℕ := 6

-- Verify 1384 % 7 gives 6
theorem modulo_seven : base_ten_number % 7 = modulo_seven_result :=
by
  -- The expected proof should compute 1384 % 7
  -- and check that it equals 6
  sorry

end base_five_to_base_ten_modulo_seven_l2210_221008


namespace length_of_train_l2210_221086

theorem length_of_train (speed : ℝ) (time : ℝ) (h1: speed = 48 * (1000 / 3600) * (1 / 1)) (h2: time = 9) : 
  (speed * time) = 119.97 :=
by
  sorry

end length_of_train_l2210_221086


namespace probability_of_first_hearts_and_second_clubs_l2210_221094

noncomputable def probability_first_hearts_second_clubs : ℚ :=
  let total_cards := 52
  let hearts_count := 13
  let clubs_count := 13
  let probability_first_hearts := hearts_count / total_cards
  let probability_second_clubs_given_first_hearts := clubs_count / (total_cards - 1)
  probability_first_hearts * probability_second_clubs_given_first_hearts

theorem probability_of_first_hearts_and_second_clubs :
  probability_first_hearts_second_clubs = 13 / 204 :=
by
  sorry

end probability_of_first_hearts_and_second_clubs_l2210_221094


namespace reciprocals_of_roots_l2210_221055

variable (a b c k : ℝ)

theorem reciprocals_of_roots (kr ks : ℝ) (h_eq : a * kr^2 + k * c * kr + b = 0) (h_eq2 : a * ks^2 + k * c * ks + b = 0) :
  (1 / (kr^2)) + (1 / (ks^2)) = (k^2 * c^2 - 2 * a * b) / (b^2) :=
by
  sorry

end reciprocals_of_roots_l2210_221055


namespace complex_repair_cost_l2210_221089

theorem complex_repair_cost
  (charge_tire : ℕ)
  (cost_part_tire : ℕ)
  (num_tires : ℕ)
  (charge_complex : ℕ)
  (num_complex : ℕ)
  (profit_retail : ℕ)
  (fixed_expenses : ℕ)
  (total_profit : ℕ)
  (profit_tire : ℕ := charge_tire - cost_part_tire)
  (total_profit_tire : ℕ := num_tires * profit_tire)
  (total_revenue_complex : ℕ := num_complex * charge_complex)
  (initial_profit : ℕ :=
    total_profit_tire + profit_retail - fixed_expenses)
  (needed_profit_complex : ℕ := total_profit - initial_profit) :
  needed_profit_complex = 100 / num_complex :=
by
  sorry

end complex_repair_cost_l2210_221089


namespace largest_prime_divisor_for_primality_check_l2210_221038

theorem largest_prime_divisor_for_primality_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) : 
  ∃ p, Prime p ∧ p ≤ Int.sqrt 1050 ∧ ∀ q, Prime q → q ≤ Int.sqrt n → q ≤ p := sorry

end largest_prime_divisor_for_primality_check_l2210_221038


namespace probability_red_ball_l2210_221067

-- Let P_red be the probability of drawing a red ball.
-- Let P_white be the probability of drawing a white ball.
-- Let P_black be the probability of drawing a black ball.
-- Let P_red_or_white be the probability of drawing a red or white ball.
-- Let P_red_or_black be the probability of drawing a red or black ball.

variable (P_red P_white P_black : ℝ)
variable (P_red_or_white P_red_or_black : ℝ)

-- Given conditions
axiom P_red_or_white_condition : P_red_or_white = 0.58
axiom P_red_or_black_condition : P_red_or_black = 0.62

-- The total probability must sum to 1.
axiom total_probability_condition : P_red + P_white + P_black = 1

-- Prove that the probability of drawing a red ball is 0.2.
theorem probability_red_ball : P_red = 0.2 :=
by
  -- To be proven
  sorry

end probability_red_ball_l2210_221067


namespace range_of_a_l2210_221024

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) a, (x^2 - 4 * x) ∈ Set.Icc (-4 : ℝ) 32) →
  2 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l2210_221024


namespace number_of_people_per_taxi_l2210_221097

def num_people_in_each_taxi (x : ℕ) (cars taxis vans total : ℕ) : Prop :=
  (cars = 3 * 4) ∧ (vans = 2 * 5) ∧ (total = 58) ∧ (taxis = 6 * x) ∧ (cars + vans + taxis = total)

theorem number_of_people_per_taxi
  (x cars taxis vans total : ℕ)
  (h1 : cars = 3 * 4)
  (h2 : vans = 2 * 5)
  (h3 : total = 58)
  (h4 : taxis = 6 * x)
  (h5 : cars + vans + taxis = total) :
  x = 6 :=
by
  sorry

end number_of_people_per_taxi_l2210_221097


namespace max_value_of_m_l2210_221032

-- Define the function f(x)
def f (x : ℝ) := x^2 + 2 * x

-- Define the property of t and m such that the condition holds for all x in [1, m]
def valid_t_m (t m : ℝ) : Prop :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) ≤ 3 * x

-- The proof statement ensuring the maximum value of m is 8
theorem max_value_of_m 
  (t : ℝ) (m : ℝ) 
  (ht : ∃ x : ℝ, valid_t_m t x ∧ x = 8) : 
  ∀ m, valid_t_m t m → m ≤ 8 :=
  sorry

end max_value_of_m_l2210_221032


namespace six_times_eightx_plus_tenpi_eq_fourP_l2210_221077

variable {x : ℝ} {π P : ℝ}

theorem six_times_eightx_plus_tenpi_eq_fourP (h : 3 * (4 * x + 5 * π) = P) : 
    6 * (8 * x + 10 * π) = 4 * P :=
sorry

end six_times_eightx_plus_tenpi_eq_fourP_l2210_221077


namespace silverware_probability_l2210_221083

-- Define the contents of the drawer
def forks := 6
def spoons := 6
def knives := 6

-- Total number of pieces of silverware
def total_silverware := forks + spoons + knives

-- Combinations formula for choosing r items out of n
def choose (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Total number of ways to choose 3 pieces out of 18
def total_ways := choose total_silverware 3

-- Number of ways to choose 1 fork, 1 spoon, and 1 knife
def specific_ways := forks * spoons * knives

-- Calculated probability
def probability := specific_ways / total_ways

theorem silverware_probability : probability = 9 / 34 := 
  sorry
 
end silverware_probability_l2210_221083


namespace sum_of_vars_l2210_221030

theorem sum_of_vars (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 2 * y) : x + y + z = 7 * x := 
by 
  sorry

end sum_of_vars_l2210_221030


namespace hyperbola_ratio_l2210_221093

theorem hyperbola_ratio (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_foci_distance : c^2 = a^2 + b^2)
  (h_midpoint_on_hyperbola : ∀ x y, 
    (x, y) = (-(c / 2), c / 2) → ∃ (k l : ℝ), (k^2 / a^2) - (l^2 / b^2) = 1) :
  c / a = (Real.sqrt 10 + Real.sqrt 2) / 2 := 
sorry

end hyperbola_ratio_l2210_221093


namespace pages_left_to_read_l2210_221007

-- Define the given conditions
def total_pages : ℕ := 563
def pages_read : ℕ := 147

-- Define the proof statement
theorem pages_left_to_read : total_pages - pages_read = 416 :=
by
  -- The proof will be given here
  sorry

end pages_left_to_read_l2210_221007


namespace non_negative_real_inequality_l2210_221050

theorem non_negative_real_inequality
  {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
by
  sorry

end non_negative_real_inequality_l2210_221050


namespace pipe_fill_rate_l2210_221010

theorem pipe_fill_rate 
  (C : ℝ) (t : ℝ) (capacity : C = 4000) (time_to_fill : t = 300) :
  (3/4 * C / t) = 10 := 
by 
  sorry

end pipe_fill_rate_l2210_221010


namespace monotonic_f_deriv_nonneg_l2210_221098

theorem monotonic_f_deriv_nonneg (k : ℝ) :
  (∀ x : ℝ, (1 / 2) < x → k - 1 / x ≥ 0) ↔ k ≥ 2 :=
by sorry

end monotonic_f_deriv_nonneg_l2210_221098


namespace inscribed_square_product_l2210_221072

theorem inscribed_square_product (a b : ℝ)
  (h1 : a + b = 2 * Real.sqrt 5)
  (h2 : Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 2) :
  a * b = -6 := 
by
  sorry

end inscribed_square_product_l2210_221072


namespace actual_distance_between_towns_l2210_221069

theorem actual_distance_between_towns
  (d_map : ℕ) (scale1 : ℕ) (scale2 : ℕ) (distance1 : ℕ) (distance2 : ℕ) (remaining_distance : ℕ) :
  d_map = 9 →
  scale1 = 10 →
  distance1 = 5 →
  scale2 = 8 →
  remaining_distance = d_map - distance1 →
  d_map = distance1 + remaining_distance →
  (distance1 * scale1 + remaining_distance * scale2 = 82) := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end actual_distance_between_towns_l2210_221069


namespace workers_complete_job_together_in_time_l2210_221099

theorem workers_complete_job_together_in_time :
  let work_rate_A := 1 / 10 
  let work_rate_B := 1 / 15
  let work_rate_C := 1 / 20
  let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
  let time := 1 / combined_work_rate
  time = 60 / 13 :=
by
  let work_rate_A := 1 / 10
  let work_rate_B := 1 / 15
  let work_rate_C := 1 / 20
  let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
  let time := 1 / combined_work_rate
  sorry

end workers_complete_job_together_in_time_l2210_221099


namespace distance_with_wind_l2210_221031

-- Define constants
def distance_against_wind : ℝ := 320
def speed_wind : ℝ := 20
def speed_plane_still_air : ℝ := 180

-- Calculate effective speeds
def effective_speed_with_wind : ℝ := speed_plane_still_air + speed_wind
def effective_speed_against_wind : ℝ := speed_plane_still_air - speed_wind

-- Define the proof statement
theorem distance_with_wind :
  ∃ (D : ℝ), (D / effective_speed_with_wind) = (distance_against_wind / effective_speed_against_wind) ∧ D = 400 :=
by
  sorry

end distance_with_wind_l2210_221031


namespace parabola_solution_unique_l2210_221053

theorem parabola_solution_unique (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 4 * a + 2 * b + c = -1) (h3 : 4 * a + b = 1) :
  a = 3 ∧ b = -11 ∧ c = 9 := 
  by sorry

end parabola_solution_unique_l2210_221053


namespace max_value_of_xyz_l2210_221033

noncomputable def max_product (x y z : ℝ) : ℝ :=
  x * y * z

theorem max_value_of_xyz (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x = y) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) (h6 : x ≤ z) (h7 : z ≤ 2 * x) :
  max_product x y z ≤ (1 / 27) := 
by
  sorry

end max_value_of_xyz_l2210_221033


namespace max_volume_is_16_l2210_221051

noncomputable def max_volume (width : ℝ) (material : ℝ) : ℝ :=
  let l := (material - 2 * width) / (2 + 2 * width)
  let h := (material - 2 * l) / (2 * width + 2 * l)
  l * width * h

theorem max_volume_is_16 :
  max_volume 2 32 = 16 :=
by
  sorry

end max_volume_is_16_l2210_221051


namespace total_flowers_tuesday_l2210_221078

def ginger_flower_shop (lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday: ℕ) := 
  let lilacs_tuesday := lilacs_monday + lilacs_monday * 5 / 100
  let roses_tuesday := roses_monday - roses_monday * 4 / 100
  let tulips_tuesday := tulips_monday - tulips_monday * 7 / 100
  let gardenias_tuesday := gardenias_monday
  let orchids_tuesday := orchids_monday
  lilacs_tuesday + roses_tuesday + tulips_tuesday + gardenias_tuesday + orchids_tuesday

theorem total_flowers_tuesday (lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday: ℕ) 
  (h1: lilacs_monday = 15)
  (h2: roses_monday = 3 * lilacs_monday)
  (h3: gardenias_monday = lilacs_monday / 2)
  (h4: tulips_monday = 2 * (roses_monday + gardenias_monday))
  (h5: orchids_monday = (roses_monday + gardenias_monday + tulips_monday) / 3):
  ginger_flower_shop lilacs_monday roses_monday gardenias_monday tulips_monday orchids_monday = 214 :=
by
  sorry

end total_flowers_tuesday_l2210_221078


namespace average_snowfall_per_hour_l2210_221091

theorem average_snowfall_per_hour (total_snowfall : ℕ) (hours_per_week : ℕ) (total_snowfall_eq : total_snowfall = 210) (hours_per_week_eq : hours_per_week = 7 * 24) : 
  total_snowfall / hours_per_week = 5 / 4 :=
by
  -- skip the proof
  sorry

end average_snowfall_per_hour_l2210_221091


namespace optionA_is_square_difference_l2210_221095

theorem optionA_is_square_difference (x y : ℝ) : 
  (-x + y) * (x + y) = -(x + y) * (x - y) :=
by sorry

end optionA_is_square_difference_l2210_221095


namespace total_marbles_l2210_221063

/--
Some marbles in a bag are red and the rest are blue.
If one red marble is removed, then one-seventh of the remaining marbles are red.
If two blue marbles are removed instead of one red, then one-fifth of the remaining marbles are red.
Prove that the total number of marbles in the bag originally is 22.
-/
theorem total_marbles (r b : ℕ) (h1 : (r - 1) / (r + b - 1) = 1 / 7) (h2 : r / (r + b - 2) = 1 / 5) :
  r + b = 22 := by
  sorry

end total_marbles_l2210_221063


namespace expression_equals_neg_one_l2210_221012

theorem expression_equals_neg_one (b y : ℝ) (hb : b ≠ 0) (h₁ : y ≠ b) (h₂ : y ≠ -b) :
  ( (b / (b + y) + y / (b - y)) / (y / (b + y) - b / (b - y)) ) = -1 :=
sorry

end expression_equals_neg_one_l2210_221012


namespace reflection_of_point_l2210_221019

def reflect_across_y_neg_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (y, x)

theorem reflection_of_point
  (P : ℝ × ℝ)
  (h : P = (8, -3)) :
  reflect_across_y_neg_x P = (3, -8) :=
by
  rw [h]
  sorry

end reflection_of_point_l2210_221019


namespace n_squared_divisible_by_144_l2210_221005

-- Definitions based on the conditions
variables (n k : ℕ)
def is_positive (n : ℕ) : Prop := n > 0
def largest_divisor_of_n_is_twelve (n : ℕ) : Prop := ∃ k, n = 12 * k
def divisible_by (m n : ℕ) : Prop := ∃ k, m = n * k

theorem n_squared_divisible_by_144
  (h1 : is_positive n)
  (h2 : largest_divisor_of_n_is_twelve n) :
  divisible_by (n * n) 144 :=
sorry

end n_squared_divisible_by_144_l2210_221005


namespace max_min_f_m1_possible_ns_l2210_221003

noncomputable def f (a b : ℝ) (x : ℝ) (m : ℝ) : ℝ :=
  let a := (Real.sqrt 2 * Real.sin (Real.pi / 4 + m * x), -Real.sqrt 3)
  let b := (Real.sqrt 2 * Real.sin (Real.pi / 4 + m * x), Real.cos (2 * m * x))
  a.1 * b.1 + a.2 * b.2

theorem max_min_f_m1 (x : ℝ) (h₁ : x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) :
  2 ≤ f (Real.sqrt 2) 1 x 1 ∧ f (Real.sqrt 2) 1 x 1 ≤ 3 :=
by
  sorry

theorem possible_ns (n : ℤ) (h₂ : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 2017) ∧ f (Real.sqrt 2) ((n * Real.pi) / 2) x ((n * Real.pi) / 2) = 0) :
  n = 1 ∨ n = -1 :=
by
  sorry

end max_min_f_m1_possible_ns_l2210_221003


namespace monotonicity_of_f_solve_inequality_l2210_221022

noncomputable def f (x : ℝ) : ℝ := sorry

def f_defined : ∀ x > 0, ∃ y, f y = f x := sorry

axiom functional_eq : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y 

axiom f_gt_zero : ∀ x, x > 1 → f x > 0

theorem monotonicity_of_f : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 < f x2 :=
sorry

theorem solve_inequality (x : ℝ) (h1 : f 2 = 1) (h2 : 0 < x) : 
  f x + f (x - 3) ≤ 2 ↔ 3 < x ∧ x ≤ 4 :=
sorry

end monotonicity_of_f_solve_inequality_l2210_221022


namespace general_formula_a_sum_sn_l2210_221047

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ :=
  if n = 0 then 2 else 2 * n

-- Define the sequence {b_n}
def b (n : ℕ) : ℕ :=
  a n + 2 ^ (a n)

-- Define the sum of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem general_formula_a :
  ∀ n, a n = 2 * n :=
sorry

theorem sum_sn :
  ∀ n, S n = n * (n + 1) + (4^(n + 1) - 4) / 3 :=
sorry

end general_formula_a_sum_sn_l2210_221047


namespace statement_c_correct_l2210_221087

theorem statement_c_correct (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b :=
by sorry

end statement_c_correct_l2210_221087


namespace increasing_condition_l2210_221052

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) + a * (Real.exp (-x))

theorem increasing_condition (a : ℝ) : (∀ x : ℝ, 0 ≤ (Real.exp (2 * x) - a) / (Real.exp x)) ↔ a ≤ 0 :=
by
  sorry

end increasing_condition_l2210_221052


namespace sum_mod_16_l2210_221028

theorem sum_mod_16 :
  (70 + 71 + 72 + 73 + 74 + 75 + 76 + 77) % 16 = 0 := 
by
  sorry

end sum_mod_16_l2210_221028


namespace percentage_increase_from_March_to_January_l2210_221044

variable {F J M : ℝ}

def JanuaryCondition (F J : ℝ) : Prop :=
  J = 0.90 * F

def MarchCondition (F M : ℝ) : Prop :=
  M = 0.75 * F

theorem percentage_increase_from_March_to_January (F J M : ℝ) (h1 : JanuaryCondition F J) (h2 : MarchCondition F M) :
  (J / M) = 1.20 := by 
  sorry

end percentage_increase_from_March_to_January_l2210_221044


namespace subset_of_inter_eq_self_l2210_221013

variable {α : Type*}
variables (M N : Set α)

theorem subset_of_inter_eq_self (h : M ∩ N = M) : M ⊆ N :=
sorry

end subset_of_inter_eq_self_l2210_221013


namespace face_opposite_A_l2210_221075
noncomputable def cube_faces : List String := ["A", "B", "C", "D", "E", "F"]

theorem face_opposite_A (cube_faces : List String) 
  (h1 : cube_faces.length = 6)
  (h2 : "A" ∈ cube_faces) 
  (h3 : "B" ∈ cube_faces)
  (h4 : "C" ∈ cube_faces) 
  (h5 : "D" ∈ cube_faces)
  (h6 : "E" ∈ cube_faces) 
  (h7 : "F" ∈ cube_faces)
  : ("D" ≠ "A") := 
by
  sorry

end face_opposite_A_l2210_221075


namespace red_stripe_area_l2210_221058

theorem red_stripe_area (diameter height stripe_width : ℝ) (num_revolutions : ℕ) 
  (diam_pos : 0 < diameter) (height_pos : 0 < height) (width_pos : 0 < stripe_width) (height_eq_80 : height = 80)
  (width_eq_3 : stripe_width = 3) (revolutions_eq_2 : num_revolutions = 2) :
  240 = stripe_width * height := 
by
  sorry

end red_stripe_area_l2210_221058


namespace find_n_square_divides_exponential_plus_one_l2210_221029

theorem find_n_square_divides_exponential_plus_one :
  ∀ n : ℕ, (n^2 ∣ 2^n + 1) → (n = 1) :=
by
  sorry

end find_n_square_divides_exponential_plus_one_l2210_221029


namespace intersection_A_B_l2210_221025

def A : Set ℤ := {-2, -1, 0, 1, 2, 3}
def B : Set ℤ := {x | x^2 - 2 * x - 3 < 0}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l2210_221025


namespace volume_of_tetrahedron_l2210_221070

-- Define the setup of tetrahedron D-ABC
def tetrahedron_volume (V : ℝ) : Prop :=
  ∃ (DA : ℝ) (A B C D : ℝ × ℝ × ℝ), 
  A = (0, 0, 0) ∧ 
  B = (2, 0, 0) ∧ 
  C = (1, Real.sqrt 3, 0) ∧
  D = (1, Real.sqrt 3/3, DA) ∧
  DA = 2 * Real.sqrt 3 ∧
  ∃ tan_dihedral : ℝ, tan_dihedral = 2 ∧
  V = 2

-- The statement to prove the volume is indeed 2 given the conditions.
theorem volume_of_tetrahedron : ∃ V, tetrahedron_volume V :=
by 
  sorry

end volume_of_tetrahedron_l2210_221070
