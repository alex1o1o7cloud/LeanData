import Mathlib

namespace fraction_to_terminating_decimal_l938_93801

theorem fraction_to_terminating_decimal :
  (47 / (2^3 * 5^4) : ℝ) = 0.0094 := by
  sorry

end fraction_to_terminating_decimal_l938_93801


namespace least_number_to_add_l938_93898

theorem least_number_to_add (n : ℕ) (sum_digits : ℕ) (next_multiple : ℕ) 
  (h1 : n = 51234) 
  (h2 : sum_digits = 5 + 1 + 2 + 3 + 4) 
  (h3 : next_multiple = 18) :
  ∃ k, (k = next_multiple - sum_digits) ∧ (n + k) % 9 = 0 :=
sorry

end least_number_to_add_l938_93898


namespace circle_equation_l938_93880

-- Definitions of the conditions
def passes_through (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (r : ℝ) : Prop :=
  (c - a) ^ 2 + (d - b) ^ 2 = r ^ 2

def center_on_line (a : ℝ) (b : ℝ) : Prop :=
  a - b - 4 = 0

-- Statement of the problem to be proved
theorem circle_equation 
  (a b r : ℝ) 
  (h1 : passes_through a b (-1) (-4) r)
  (h2 : passes_through a b 6 3 r)
  (h3 : center_on_line a b) :
  -- Equation of the circle
  (a = 3 ∧ b = -1 ∧ r = 5) → ∀ x y : ℝ, 
    (x - 3)^2 + (y + 1)^2 = 25 :=
sorry

end circle_equation_l938_93880


namespace pure_imaginary_a_zero_l938_93850

theorem pure_imaginary_a_zero (a : ℝ) (i : ℂ) (hi : i^2 = -1) :
  (z = (1 - (a:ℝ)^2 * i) / i) ∧ (∀ (z : ℂ), z.re = 0 → z = (0 : ℂ)) → a = 0 :=
by
  sorry

end pure_imaginary_a_zero_l938_93850


namespace tangent_line_eq_l938_93844

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3 * x + 1

def point : ℝ × ℝ := (2, 5)

theorem tangent_line_eq : ∀ (x y : ℝ), 
  (y = x^2 + 3 * x + 1) ∧ (x = 2 ∧ y = 5) →
  7 * x - y = 9 :=
by
  intros x y h
  sorry

end tangent_line_eq_l938_93844


namespace solve_inequality_l938_93832

-- Define the inequality problem.
noncomputable def inequality_problem (x : ℝ) : Prop :=
(x^2 + 2 * x - 15) / (x + 5) < 0

-- Define the solution set.
def solution_set (x : ℝ) : Prop :=
-5 < x ∧ x < 3

-- State the equivalence theorem.
theorem solve_inequality (x : ℝ) (h : x ≠ -5) : 
  inequality_problem x ↔ solution_set x :=
sorry

end solve_inequality_l938_93832


namespace fifth_friend_contribution_l938_93881

variables (a b c d e : ℕ)

theorem fifth_friend_contribution:
  a + b + c + d + e = 120 ∧
  a = 2 * b ∧
  b = (c + d) / 3 ∧
  c = 2 * e →
  e = 12 :=
sorry

end fifth_friend_contribution_l938_93881


namespace participants_begin_competition_l938_93863

theorem participants_begin_competition (x : ℝ) 
  (h1 : 0.4 * x * (1 / 4) = 16) : 
  x = 160 := 
by
  sorry

end participants_begin_competition_l938_93863


namespace robotics_club_non_participants_l938_93858

theorem robotics_club_non_participants (club_students electronics_students programming_students both_students : ℕ) 
  (h1 : club_students = 80) 
  (h2 : electronics_students = 45) 
  (h3 : programming_students = 50) 
  (h4 : both_students = 30) : 
  club_students - (electronics_students - both_students + programming_students - both_students + both_students) = 15 :=
by
  -- The proof would be here
  sorry

end robotics_club_non_participants_l938_93858


namespace sqrt_pi_decimal_expansion_l938_93815

-- Statement of the problem: Compute the first 23 digits of the decimal expansion of sqrt(pi)
theorem sqrt_pi_decimal_expansion : 
  ( ∀ n, n ≤ 22 → 
    (digits : List ℕ) = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23] →
      (d1 = 1 ∧ d2 = 7 ∧ d3 = 7 ∧ d4 = 2 ∧ d5 = 4 ∧ d6 = 5 ∧ d7 = 3 ∧ d8 = 8 ∧ d9 = 5 ∧ d10 = 0 ∧ d11 = 9 ∧ d12 = 0 ∧ d13 = 5 ∧ d14 = 5 ∧ d15 = 1 ∧ d16 = 6 ∧ d17 = 0 ∧ d18 = 2 ∧ d19 = 7 ∧ d20 = 2 ∧ d21 = 9 ∧ d22 = 8 ∧ d23 = 1)) → 
  True :=
by
  sorry
  -- Actual proof to be filled, this is just the statement showing that we expected the digits 
  -- of the decimal expansion of sqrt(pi) match the specified values up to the 23rd place.

end sqrt_pi_decimal_expansion_l938_93815


namespace three_digit_numbers_mod_1000_l938_93810

theorem three_digit_numbers_mod_1000 (n : ℕ) (h_lower : 100 ≤ n) (h_upper : n ≤ 999) : 
  (n^2 ≡ n [MOD 1000]) ↔ (n = 376 ∨ n = 625) :=
by sorry

end three_digit_numbers_mod_1000_l938_93810


namespace compare_neg_frac1_l938_93816

theorem compare_neg_frac1 : (-3 / 7 : ℝ) < (-8 / 21 : ℝ) :=
sorry

end compare_neg_frac1_l938_93816


namespace cos_beta_value_l938_93804

theorem cos_beta_value
  (α β : ℝ)
  (hαβ : 0 < α ∧ α < π ∧ 0 < β ∧ β < π)
  (h1 : Real.sin (α + β) = 5 / 13)
  (h2 : Real.tan (α / 2) = 1 / 2) :
  Real.cos β = -16 / 65 := 
by 
  sorry

end cos_beta_value_l938_93804


namespace lottery_winning_situations_l938_93817

theorem lottery_winning_situations :
  let num_tickets := 8
  let first_prize := 1
  let second_prize := 1
  let third_prize := 1
  let non_winning := 5
  let customers := 4
  let tickets_per_customer := 2
  let total_ways := 24 + 36
  total_ways = 60 :=
by
  let num_tickets := 8
  let first_prize := 1
  let second_prize := 1
  let third_prize := 1
  let non_winning := 5
  let customers := 4
  let tickets_per_customer := 2
  let total_ways := 24 + 36

  -- Skipping proof steps
  sorry

end lottery_winning_situations_l938_93817


namespace quadratic_roots_conditions_l938_93827

-- Definitions of the given conditions.
variables (a b c : ℝ)  -- Coefficients of the quadratic trinomial
variable (h : b^2 - 4 * a * c ≥ 0)  -- Given condition that the discriminant is non-negative

-- Statement to prove:
theorem quadratic_roots_conditions (a b c : ℝ) (h : b^2 - 4 * a * c ≥ 0) :
  ¬(∀ x : ℝ, a^2 * x^2 + b^2 * x + c^2 = 0) ∧ (∀ x : ℝ, a^3 * x^2 + b^3 * x + c^3 = 0 → b^6 - 4 * a^3 * c^3 ≥ 0) :=
by
  sorry

end quadratic_roots_conditions_l938_93827


namespace largest_integer_of_four_l938_93814

theorem largest_integer_of_four (a b c d : ℤ) 
  (h1 : a + b + c = 160) 
  (h2 : a + b + d = 185) 
  (h3 : a + c + d = 205) 
  (h4 : b + c + d = 230) : 
  max (max a (max b c)) d = 100 := 
by
  sorry

end largest_integer_of_four_l938_93814


namespace probability_single_trial_l938_93842

theorem probability_single_trial 
  (p : ℝ) 
  (h₁ : ∀ n : ℕ, 1 ≤ n → ∃ x : ℝ, x = (1 - (1 - p) ^ n)) 
  (h₂ : 1 - (1 - p) ^ 4 = 65 / 81) : 
  p = 1 / 3 :=
by 
  sorry

end probability_single_trial_l938_93842


namespace performance_attendance_l938_93845

theorem performance_attendance (A C : ℕ) (hC : C = 18) (hTickets : 16 * A + 9 * C = 258) : A + C = 24 :=
by
  sorry

end performance_attendance_l938_93845


namespace quadrilateral_diagonal_areas_relation_l938_93819

-- Defining the areas of the four triangles and the quadrilateral
variables (A B C D Q : ℝ)

-- Stating the property to be proven
theorem quadrilateral_diagonal_areas_relation 
  (H1 : Q = A + B + C + D) :
  A * B * C * D = ((A + B) * (B + C) * (C + D) * (D + A))^2 / Q^4 :=
by sorry

end quadrilateral_diagonal_areas_relation_l938_93819


namespace gcd_221_195_l938_93836

-- Define the two numbers
def a := 221
def b := 195

-- Statement of the problem: the gcd of a and b is 13
theorem gcd_221_195 : Nat.gcd a b = 13 := 
by
  sorry

end gcd_221_195_l938_93836


namespace anne_speed_ratio_l938_93855

variable (B A A' : ℝ)

theorem anne_speed_ratio (h1 : A = 1 / 12)
                        (h2 : B + A = 1 / 4)
                        (h3 : B + A' = 1 / 3) : 
                        A' / A = 2 := 
by
  -- Proof is omitted
  sorry

end anne_speed_ratio_l938_93855


namespace power_difference_divisible_by_35_l938_93876

theorem power_difference_divisible_by_35 (n : ℕ) : (3^(6*n) - 2^(6*n)) % 35 = 0 := 
by sorry

end power_difference_divisible_by_35_l938_93876


namespace area_of_rectangle_l938_93896

theorem area_of_rectangle (l w : ℝ) (h_perimeter : 2 * (l + w) = 126) (h_difference : l - w = 37) : l * w = 650 :=
sorry

end area_of_rectangle_l938_93896


namespace solve_a_b_l938_93822

theorem solve_a_b (a b : ℕ) (h₀ : 2 * a^2 = 3 * b^3) : ∃ k : ℕ, a = 18 * k^3 ∧ b = 6 * k^2 := 
sorry

end solve_a_b_l938_93822


namespace gcd_9125_4277_l938_93889

theorem gcd_9125_4277 : Nat.gcd 9125 4277 = 1 :=
by
  -- proof by Euclidean algorithm steps
  sorry

end gcd_9125_4277_l938_93889


namespace sum_of_values_satisfying_equation_l938_93859

noncomputable def sum_of_roots_of_quadratic (a b c : ℝ) : ℝ := -b / a

theorem sum_of_values_satisfying_equation :
  (∃ x : ℝ, (x^2 - 5 * x + 7 = 9)) →
  sum_of_roots_of_quadratic 1 (-5) (-2) = 5 :=
by
  sorry

end sum_of_values_satisfying_equation_l938_93859


namespace digit_sum_eq_21_l938_93806

theorem digit_sum_eq_21 (A B C D: ℕ) (h1: A ≠ 0) 
    (h2: (A * 10 + B) * 100 + (C * 10 + D) = (C * 10 + D)^2 - (A * 10 + B)^2) 
    (hA: A < 10) (hB: B < 10) (hC: C < 10) (hD: D < 10) : 
    A + B + C + D = 21 :=
by 
  sorry

end digit_sum_eq_21_l938_93806


namespace contradiction_proof_l938_93807

theorem contradiction_proof (a b : ℕ) (h : a + b ≥ 3) : (a ≥ 2) ∨ (b ≥ 2) :=
sorry

end contradiction_proof_l938_93807


namespace truck_capacity_l938_93895

theorem truck_capacity
  (x y : ℝ)
  (h1 : 2 * x + 3 * y = 15.5)
  (h2 : 5 * x + 6 * y = 35) :
  3 * x + 5 * y = 24.5 :=
sorry

end truck_capacity_l938_93895


namespace fraction_dutch_americans_has_window_l938_93826

variable (P D DA : ℕ)
variable (f_P_d d_P_w : ℚ)
variable (DA_w : ℕ)

-- Total number of people on the bus P 
-- Fraction of people who were Dutch f_P_d
-- Fraction of Dutch Americans who got window seats d_P_w
-- Number of Dutch Americans who sat at windows DA_w
-- Define the assumptions
def total_people_on_bus := P = 90
def fraction_dutch := f_P_d = 3 / 5
def fraction_dutch_americans_window := d_P_w = 1 / 3
def dutch_americans_window := DA_w = 9

-- Prove that fraction of Dutch people who were also American is 1/2
theorem fraction_dutch_americans_has_window (P D DA DA_w : ℕ) (f_P_d d_P_w : ℚ) :
  total_people_on_bus P ∧ fraction_dutch f_P_d ∧
  fraction_dutch_americans_window d_P_w ∧ dutch_americans_window DA_w →
  (DA: ℚ) / D = 1 / 2 :=
by
  sorry

end fraction_dutch_americans_has_window_l938_93826


namespace simplify_and_rationalize_l938_93818

theorem simplify_and_rationalize :
  let x := (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (Real.sqrt 13 / Real.sqrt 17)
  x = 3 * Real.sqrt 84885 / 1309 := sorry

end simplify_and_rationalize_l938_93818


namespace find_other_endpoint_l938_93848

theorem find_other_endpoint (x_m y_m x_1 y_1 x_2 y_2 : ℝ)
  (h_midpoint_x : x_m = (x_1 + x_2) / 2)
  (h_midpoint_y : y_m = (y_1 + y_2) / 2)
  (h_given_midpoint : x_m = 3 ∧ y_m = 0)
  (h_given_endpoint1 : x_1 = 7 ∧ y_1 = -4) :
  x_2 = -1 ∧ y_2 = 4 :=
sorry

end find_other_endpoint_l938_93848


namespace value_of_x_l938_93853

theorem value_of_x (p q r x : ℝ)
  (h1 : p = 72)
  (h2 : q = 18)
  (h3 : r = 108)
  (h4 : x = 180 - (q + r)) : 
  x = 54 := by
  sorry

end value_of_x_l938_93853


namespace triangle_inequality_l938_93856

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
sorry

end triangle_inequality_l938_93856


namespace factorize_expression_l938_93869

variable (a b : ℝ)

theorem factorize_expression : (a - b)^2 + 6 * (b - a) + 9 = (a - b - 3)^2 :=
by
  sorry

end factorize_expression_l938_93869


namespace arithmetic_mean_difference_l938_93840

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := 
sorry

end arithmetic_mean_difference_l938_93840


namespace hyperbola_equation_l938_93899

-- Definitions based on the conditions:
def hyperbola (x y a b : ℝ) : Prop := (y^2 / a^2) - (x^2 / b^2) = 1

def point_on_hyperbola (a b : ℝ) : Prop := hyperbola 2 (-2) a b

def asymptotes (a b : ℝ) : Prop := a / b = (Real.sqrt 2) / 2

-- Prove the equation of the hyperbola
theorem hyperbola_equation :
  ∃ a b, a = Real.sqrt 2 ∧ b = 2 ∧ hyperbola y x (Real.sqrt 2) 2 :=
by
  -- Placeholder for the actual proof
  sorry

end hyperbola_equation_l938_93899


namespace cyclic_quadrilateral_tangency_l938_93857

theorem cyclic_quadrilateral_tangency (a b c d x y : ℝ) (h_cyclic : a = 80 ∧ b = 100 ∧ c = 140 ∧ d = 120) 
  (h_tangency: x + y = 140) : |x - y| = 5 := 
sorry

end cyclic_quadrilateral_tangency_l938_93857


namespace age_equivalence_l938_93813

variable (x : ℕ)

theorem age_equivalence : ∃ x : ℕ, 60 + x = 35 + x + 11 + x ∧ x = 14 :=
by
  sorry

end age_equivalence_l938_93813


namespace x_equals_y_squared_plus_2y_minus_1_l938_93802

theorem x_equals_y_squared_plus_2y_minus_1 (x y : ℝ) (h : x / (x - 1) = (y^2 + 2 * y - 1) / (y^2 + 2 * y - 2)) : 
  x = y^2 + 2 * y - 1 :=
sorry

end x_equals_y_squared_plus_2y_minus_1_l938_93802


namespace total_computers_needed_l938_93829

theorem total_computers_needed
    (initial_students : ℕ)
    (students_per_computer : ℕ)
    (additional_students : ℕ)
    (initial_computers : ℕ := initial_students / students_per_computer)
    (total_computers : ℕ := initial_computers + (additional_students / students_per_computer))
    (h1 : initial_students = 82)
    (h2 : students_per_computer = 2)
    (h3 : additional_students = 16) :
    total_computers = 49 :=
by
  -- The proof would normally go here
  sorry

end total_computers_needed_l938_93829


namespace inequality_add_l938_93871

theorem inequality_add {a b c : ℝ} (h : a > b) : a + c > b + c :=
sorry

end inequality_add_l938_93871


namespace Jackie_has_more_apples_l938_93823

def Adam_apples : Nat := 9
def Jackie_apples : Nat := 10

theorem Jackie_has_more_apples : Jackie_apples - Adam_apples = 1 := by
  sorry

end Jackie_has_more_apples_l938_93823


namespace arun_brother_weight_upper_limit_l938_93892

theorem arun_brother_weight_upper_limit (w : ℝ) (X : ℝ) 
  (h1 : 61 < w ∧ w < 72)
  (h2 : 60 < w ∧ w < X)
  (h3 : w ≤ 64)
  (h4 : ((62 + 63 + 64) / 3) = 63) :
  X = 64 :=
by
  sorry

end arun_brother_weight_upper_limit_l938_93892


namespace factorize_expression_l938_93879

theorem factorize_expression (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by sorry

end factorize_expression_l938_93879


namespace projectile_reaches_height_at_first_l938_93809

noncomputable def reach_height (t : ℝ) : ℝ :=
-16 * t^2 + 80 * t

theorem projectile_reaches_height_at_first (t : ℝ) :
  reach_height t = 36 → t = 0.5 :=
by
  -- The proof can be provided here
  sorry

end projectile_reaches_height_at_first_l938_93809


namespace word_limit_correct_l938_93894

-- Definition for the conditions
def saturday_words : ℕ := 450
def sunday_words : ℕ := 650
def exceeded_amount : ℕ := 100

-- The total words written
def total_words : ℕ := saturday_words + sunday_words

-- The word limit which we need to prove
def word_limit : ℕ := total_words - exceeded_amount

theorem word_limit_correct : word_limit = 1000 := by
  unfold word_limit total_words saturday_words sunday_words exceeded_amount
  sorry

end word_limit_correct_l938_93894


namespace total_price_increase_percentage_l938_93888

theorem total_price_increase_percentage 
    (P : ℝ) 
    (h1 : P > 0) 
    (P_after_first_increase : ℝ := P * 1.2) 
    (P_after_second_increase : ℝ := P_after_first_increase * 1.15) :
    ((P_after_second_increase - P) / P) * 100 = 38 :=
by
  sorry

end total_price_increase_percentage_l938_93888


namespace find_x_modulo_l938_93851

theorem find_x_modulo (k : ℤ) : ∃ x : ℤ, x = 18 + 31 * k ∧ ((37 * x) % 31 = 15) := by
  sorry

end find_x_modulo_l938_93851


namespace numerator_of_fraction_l938_93862

theorem numerator_of_fraction (x : ℤ) (h : (x : ℚ) / (4 * x - 5) = 3 / 7) : x = 3 := 
sorry

end numerator_of_fraction_l938_93862


namespace problem_l938_93839

-- Definitions
variables {a b : ℝ}
def is_root (p : ℝ → ℝ) (x : ℝ) : Prop := p x = 0

-- Root condition using the given equation
def quadratic_eq (x : ℝ) : ℝ := (x - 3) * (2 * x + 7) - (x^2 - 11 * x + 28)

-- Statement to prove
theorem problem (ha : is_root quadratic_eq a) (hb : is_root quadratic_eq b) (h_distinct : a ≠ b):
  (a + 2) * (b + 2) = -66 :=
sorry

end problem_l938_93839


namespace original_number_l938_93800

/-- Proof that the original three-digit number abc equals 118 under the given conditions. -/
theorem original_number (N : ℕ) (hN : N = 4332) (a b c : ℕ)
  (h : 100 * a + 10 * b + c = 118) :
  100 * a + 10 * b + c = 118 :=
by
  sorry

end original_number_l938_93800


namespace beth_red_pill_cost_l938_93837

noncomputable def red_pill_cost (blue_pill_cost : ℝ) : ℝ := blue_pill_cost + 3

theorem beth_red_pill_cost :
  ∃ (blue_pill_cost : ℝ), 
  (21 * (red_pill_cost blue_pill_cost + blue_pill_cost) = 966) 
  → 
  red_pill_cost blue_pill_cost = 24.5 :=
by
  sorry

end beth_red_pill_cost_l938_93837


namespace find_g9_l938_93887

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g3_value : g 3 = 4

theorem find_g9 : g 9 = 64 := sorry

end find_g9_l938_93887


namespace opposite_of_neg_four_l938_93874

-- Define the condition: the opposite of a number is the number that, when added to the original number, results in zero.
def is_opposite (a b : Int) : Prop := a + b = 0

-- The specific theorem we want to prove
theorem opposite_of_neg_four : is_opposite (-4) 4 := by
  -- Placeholder for the proof
  sorry

end opposite_of_neg_four_l938_93874


namespace kamal_twice_age_in_future_l938_93808

theorem kamal_twice_age_in_future :
  ∃ x : ℕ, (K = 40) ∧ (K - 8 = 4 * (S - 8)) ∧ (K + x = 2 * (S + x)) :=
by {
  sorry 
}

end kamal_twice_age_in_future_l938_93808


namespace silk_diameter_scientific_notation_l938_93830

-- Definition of the given condition
def silk_diameter := 0.000014 

-- The goal to be proved
theorem silk_diameter_scientific_notation : silk_diameter = 1.4 * 10^(-5) := 
by 
  sorry

end silk_diameter_scientific_notation_l938_93830


namespace students_who_like_yellow_l938_93866

theorem students_who_like_yellow (total_students girls students_like_green girls_like_pink students_like_yellow : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_green = total_students / 2)
  (h3 : girls_like_pink = girls / 3)
  (h4 : girls = 18)
  (h5 : students_like_yellow = total_students - (students_like_green + girls_like_pink)) :
  students_like_yellow = 9 :=
by
  sorry

end students_who_like_yellow_l938_93866


namespace find_smaller_number_l938_93805

theorem find_smaller_number (x y : ℤ) (h1 : x + y = 15) (h2 : 3 * x = 5 * y - 11) : x = 8 :=
by
  sorry

end find_smaller_number_l938_93805


namespace five_times_x_plus_four_l938_93843

theorem five_times_x_plus_four (x : ℝ) (h : 4 * x - 3 = 13 * x + 12) : 5 * (x + 4) = 35 / 3 := 
by
  sorry

end five_times_x_plus_four_l938_93843


namespace james_profit_correct_l938_93882

noncomputable def jamesProfit : ℝ :=
  let tickets_bought := 200
  let cost_per_ticket := 2
  let winning_ticket_percentage := 0.20
  let percentage_one_dollar := 0.50
  let percentage_three_dollars := 0.30
  let percentage_four_dollars := 0.20
  let percentage_five_dollars := 0.80
  let grand_prize_ticket_count := 1
  let average_remaining_winner := 15
  let tax_percentage := 0.10
  let total_cost := tickets_bought * cost_per_ticket
  let winning_tickets := tickets_bought * winning_ticket_percentage
  let tickets_five_dollars := winning_tickets * percentage_five_dollars
  let other_winning_tickets := winning_tickets - tickets_five_dollars - grand_prize_ticket_count
  let total_winnings_before_tax := (tickets_five_dollars * 5) + (grand_prize_ticket_count * 5000) + (other_winning_tickets * average_remaining_winner)
  let total_tax := total_winnings_before_tax * tax_percentage
  let total_winnings_after_tax := total_winnings_before_tax - total_tax
  total_winnings_after_tax - total_cost

theorem james_profit_correct : jamesProfit = 4338.50 := by
  sorry

end james_profit_correct_l938_93882


namespace time_saved_1200_miles_l938_93833

theorem time_saved_1200_miles
  (distance : ℕ)
  (speed1 speed2 : ℕ)
  (h_distance : distance = 1200)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 50) :
  (distance / speed2) - (distance / speed1) = 4 :=
by
  sorry

end time_saved_1200_miles_l938_93833


namespace smallest_positive_integer_divisible_12_15_16_exists_l938_93852

theorem smallest_positive_integer_divisible_12_15_16_exists :
  ∃ x : ℕ, x > 0 ∧ 12 ∣ x ∧ 15 ∣ x ∧ 16 ∣ x ∧ x = 240 :=
by sorry

end smallest_positive_integer_divisible_12_15_16_exists_l938_93852


namespace not_divisible_l938_93821

theorem not_divisible {x y : ℕ} (hx : x > 0) (hy : y > 2) : ¬ (2^y - 1) ∣ (2^x + 1) := sorry

end not_divisible_l938_93821


namespace fraction_addition_l938_93820

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l938_93820


namespace min_value_expression_l938_93831

noncomputable 
def min_value_condition (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : ℝ :=
  (a + 1) * (b + 1) * (c + 1)

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : 
  min_value_condition a b c h_pos h_abc = 8 :=
sorry

end min_value_expression_l938_93831


namespace find_m_p_pairs_l938_93838

theorem find_m_p_pairs (m p : ℕ) (h_prime : Nat.Prime p) (h_eq : ∃ (x : ℕ), 2^m * p^2 + 27 = x^3) :
  (m, p) = (1, 7) :=
sorry

end find_m_p_pairs_l938_93838


namespace complex_ab_value_l938_93846

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h_i : i = Complex.I) (h_z : a + b * i = (4 + 3 * i) * i) : a * b = -12 :=
by {
  sorry
}

end complex_ab_value_l938_93846


namespace inequality_solution_set_l938_93891

theorem inequality_solution_set (a b : ℝ) (h1 : a = -2) (h2 : b = 1) :
  {x : ℝ | |2 * x + a| + |x - b| < 6} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end inequality_solution_set_l938_93891


namespace zero_function_l938_93828

noncomputable def f : ℝ → ℝ := sorry -- Let it be a placeholder for now.

theorem zero_function (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b))
  (h_int : ∀ n : ℕ, ∫ x in a..b, (x : ℝ)^n * f x = 0) :
  ∀ x ∈ Set.Icc a b, f x = 0 :=
by
  sorry -- placeholder for the proof

end zero_function_l938_93828


namespace natives_cannot_obtain_910_rupees_with_50_coins_l938_93867

theorem natives_cannot_obtain_910_rupees_with_50_coins (x y z : ℤ) : 
  x + y + z = 50 → 
  10 * x + 34 * y + 62 * z = 910 → 
  false :=
by
  sorry

end natives_cannot_obtain_910_rupees_with_50_coins_l938_93867


namespace min_value_of_E_l938_93854

noncomputable def E : ℝ := sorry

theorem min_value_of_E :
  (∀ x : ℝ, |E| + |x + 7| + |x - 5| ≥ 12) →
  (∃ x : ℝ, |x + 7| + |x - 5| = 12 → |E| = 0) :=
sorry

end min_value_of_E_l938_93854


namespace total_coins_constant_l938_93875

-- Definitions based on the conditions
def stack1 := 12
def stack2 := 17
def stack3 := 23
def stack4 := 8

def totalCoins := stack1 + stack2 + stack3 + stack4 -- 60 coins
def is_divisor (x: ℕ) := x ∣ totalCoins

-- The theorem statement
theorem total_coins_constant {x: ℕ} (h: is_divisor x) : totalCoins = 60 :=
by
  -- skip the proof steps
  sorry

end total_coins_constant_l938_93875


namespace circumscribed_sphere_radius_is_3_l938_93878

noncomputable def radius_of_circumscribed_sphere (SA SB SC : ℝ) : ℝ :=
  let space_diagonal := Real.sqrt (SA^2 + SB^2 + SC^2)
  space_diagonal / 2

theorem circumscribed_sphere_radius_is_3 : radius_of_circumscribed_sphere 2 4 4 = 3 :=
by
  unfold radius_of_circumscribed_sphere
  simp
  apply sorry

end circumscribed_sphere_radius_is_3_l938_93878


namespace arithmetic_sequence_sum_l938_93841

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by
  sorry

end arithmetic_sequence_sum_l938_93841


namespace scientific_notation_0_056_l938_93803

theorem scientific_notation_0_056 :
  (0.056 = 5.6 * 10^(-2)) :=
by
  sorry

end scientific_notation_0_056_l938_93803


namespace solve_equation_l938_93870

theorem solve_equation (x : ℝ) : 2 * (x - 2)^2 = 6 - 3 * x ↔ (x = 2 ∨ x = 1 / 2) :=
by
  sorry

end solve_equation_l938_93870


namespace eighth_term_of_geometric_sequence_l938_93893

def geometric_sequence_term (a₁ r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem eighth_term_of_geometric_sequence : 
  geometric_sequence_term 12 (1 / 3) 8 = 4 / 729 :=
by 
  sorry

end eighth_term_of_geometric_sequence_l938_93893


namespace intersect_A_B_when_a_1_subset_A_B_range_a_l938_93861

def poly_eqn (x : ℝ) : Prop := -x ^ 2 - 2 * x + 8 = 0

def sol_set_A : Set ℝ := {x | poly_eqn x}

def inequality (a x : ℝ) : Prop := a * x - 1 ≤ 0

def sol_set_B (a : ℝ) : Set ℝ := {x | inequality a x}

theorem intersect_A_B_when_a_1 :
  sol_set_A ∩ sol_set_B 1 = { -4 } :=
sorry

theorem subset_A_B_range_a (a : ℝ) :
  sol_set_A ⊆ sol_set_B a ↔ (-1 / 4 : ℝ) ≤ a ∧ a ≤ 1 / 2 :=
sorry
 
end intersect_A_B_when_a_1_subset_A_B_range_a_l938_93861


namespace machine_working_time_l938_93872

theorem machine_working_time (shirts_per_minute : ℕ) (total_shirts : ℕ) (h1 : shirts_per_minute = 3) (h2 : total_shirts = 6) :
  (total_shirts / shirts_per_minute) = 2 :=
by
  -- Begin the proof
  sorry

end machine_working_time_l938_93872


namespace prob_neq_zero_l938_93825

noncomputable def probability_no_one (a b c d : ℕ) : ℚ :=
  if 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 
  then (5/6)^4 
  else 0

theorem prob_neq_zero (a b c d : ℕ) :
  (1 ≤ a) ∧ (a ≤ 6) ∧ (1 ≤ b) ∧ (b ≤ 6) ∧ (1 ≤ c) ∧ (c ≤ 6) ∧ (1 ≤ d) ∧ (d ≤ 6) →
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  probability_no_one a b c d = 625/1296 :=
by
  sorry

end prob_neq_zero_l938_93825


namespace wood_rope_length_equivalence_l938_93897

variable (x y : ℝ)

theorem wood_rope_length_equivalence :
  (x - y = 4.5) ∧ (y = (1 / 2) * x + 1) :=
  sorry

end wood_rope_length_equivalence_l938_93897


namespace cannot_form_right_triangle_l938_93824

theorem cannot_form_right_triangle (a b c : ℕ) (h_a : a = 3) (h_b : b = 5) (h_c : c = 7) : 
  a^2 + b^2 ≠ c^2 :=
by 
  rw [h_a, h_b, h_c]
  sorry

end cannot_form_right_triangle_l938_93824


namespace sum_of_possible_values_l938_93877

theorem sum_of_possible_values (x y : ℝ)
  (h : x * y - (2 * x) / (y ^ 3) - (2 * y) / (x ^ 3) = 5) :
  ∃ s : ℝ, s = (x - 2) * (y - 2) ∧ (s = -3 ∨ s = 9) :=
sorry

end sum_of_possible_values_l938_93877


namespace supreme_sports_package_channels_l938_93864

theorem supreme_sports_package_channels (c_start : ℕ) (c_removed1 : ℕ) (c_added1 : ℕ)
                                         (c_removed2 : ℕ) (c_added2 : ℕ)
                                         (c_final : ℕ)
                                         (net1 : ℕ) (net2 : ℕ) (c_mid : ℕ) :
  c_start = 150 →
  c_removed1 = 20 →
  c_added1 = 12 →
  c_removed2 = 10 →
  c_added2 = 8 →
  c_final = 147 →
  net1 = c_removed1 - c_added1 →
  net2 = c_removed2 - c_added2 →
  c_mid = c_start - net1 - net2 →
  c_final - c_mid = 7 :=
by
  intros
  sorry

end supreme_sports_package_channels_l938_93864


namespace diamond_4_3_l938_93865

def diamond (a b : ℤ) : ℤ := 4 * a + 3 * b - 2 * a * b

theorem diamond_4_3 : diamond 4 3 = 1 :=
by
  -- The proof will go here.
  sorry

end diamond_4_3_l938_93865


namespace adjacent_number_in_grid_l938_93812

def adjacent_triangle_number (k n: ℕ) : ℕ :=
  if k % 2 = 1 then n - k else n + k

theorem adjacent_number_in_grid (n : ℕ) (bound: n = 350) :
  let k := Nat.ceil (Real.sqrt n)
  let m := (k * k) - n
  k = 19 ∧ m = 19 →
  adjacent_triangle_number k n = 314 :=
by
  sorry

end adjacent_number_in_grid_l938_93812


namespace product_increase_false_l938_93834

theorem product_increase_false (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ¬ (a * b = a * (10 * b) / 10 ∧ a * (10 * b) / 10 = 10 * (a * b)) :=
by 
  sorry

end product_increase_false_l938_93834


namespace vertex_of_parabola_point_symmetry_on_parabola_range_of_m_l938_93868

open Real

-- Problem 1: Prove the vertex of the parabola is at (1, -a)
theorem vertex_of_parabola (a : ℝ) (h : a ≠ 0) : 
  ∀ x : ℝ, y = a * x^2 - 2 * a * x → (1, -a) = ((1 : ℝ), - a) := 
sorry

-- Problem 2: Prove x_0 = 3 if m = n for given points on the parabola
theorem point_symmetry_on_parabola (a : ℝ) (h : a ≠ 0) (m n : ℝ) :
  m = n → ∀ (x0 : ℝ), y = a * x0 ^ 2 - 2 * a * x0 → x0 = 3 :=
sorry

-- Problem 3: Prove the conditions for y1 < y2 ≤ -a and the range of m
theorem range_of_m (a : ℝ) (h : a < 0) : 
  ∀ (m y1 y2 : ℝ), (y1 < y2) ∧ (y2 ≤ -a) → m < (1 / 2) := 
sorry

end vertex_of_parabola_point_symmetry_on_parabola_range_of_m_l938_93868


namespace product_of_inverses_l938_93890

theorem product_of_inverses : 
  ((1 - 1 / (3^2)) * (1 - 1 / (5^2)) * (1 - 1 / (7^2)) * (1 - 1 / (11^2)) * (1 - 1 / (13^2)) * (1 - 1 / (17^2))) = 210 / 221 := 
by {
  sorry
}

end product_of_inverses_l938_93890


namespace lemon_count_l938_93884

theorem lemon_count {total_fruits mangoes pears pawpaws : ℕ} (kiwi lemon : ℕ) :
  total_fruits = 58 ∧ 
  mangoes = 18 ∧ 
  pears = 10 ∧ 
  pawpaws = 12 ∧ 
  (kiwi = lemon) →
  lemon = 9 :=
by 
  sorry

end lemon_count_l938_93884


namespace value_of_N_l938_93860

theorem value_of_N : ∃ N : ℕ, (32^5 * 16^4 / 8^7) = 2^N ∧ N = 20 := by
  use 20
  sorry

end value_of_N_l938_93860


namespace odd_difference_even_odd_l938_93885

theorem odd_difference_even_odd (a b : ℤ) (ha : a % 2 = 0) (hb : b % 2 = 1) : (a - b) % 2 = 1 :=
sorry

end odd_difference_even_odd_l938_93885


namespace abs_h_eq_1_div_2_l938_93811

theorem abs_h_eq_1_div_2 {h : ℝ} 
  (h_sum_sq_roots : ∀ (r s : ℝ), (r + s) = 4 * h ∧ (r * s) = -8 → (r ^ 2 + s ^ 2) = 20) : 
  |h| = 1 / 2 :=
sorry

end abs_h_eq_1_div_2_l938_93811


namespace last_digit_base5_of_M_l938_93847

theorem last_digit_base5_of_M (d e f : ℕ) (hd : d < 5) (he : e < 5) (hf : f < 5)
  (h : 25 * d + 5 * e + f = 64 * f + 8 * e + d) : f = 0 :=
by
  sorry

end last_digit_base5_of_M_l938_93847


namespace ending_number_of_multiples_l938_93886

theorem ending_number_of_multiples (n : ℤ) (h : 991 = (n - 100) / 10 + 1) : n = 10000 :=
by
  sorry

end ending_number_of_multiples_l938_93886


namespace work_rate_b_l938_93835

theorem work_rate_b (W : ℝ) (A B C : ℝ) :
  (A = W / 11) → 
  (C = W / 55) →
  (8 * A + 4 * B + 4 * C = W) →
  B = W / (2420 / 341) :=
by
  intros hA hC hWork
  -- We start with the given assumptions and work towards showing B = W / (2420 / 341)
  sorry

end work_rate_b_l938_93835


namespace daniela_total_spent_l938_93849

-- Step d) Rewrite the math proof problem
theorem daniela_total_spent
    (shoe_price : ℤ) (dress_price : ℤ) (shoe_discount : ℤ) (dress_discount : ℤ)
    (shoe_count : ℤ)
    (shoe_original_price : shoe_price = 50)
    (dress_original_price : dress_price = 100)
    (shoe_discount_rate : shoe_discount = 40)
    (dress_discount_rate : dress_discount = 20)
    (shoe_total_count : shoe_count = 2)
    : shoe_count * (shoe_price - (shoe_price * shoe_discount / 100)) + (dress_price - (dress_price * dress_discount / 100)) = 140 := by 
    sorry

end daniela_total_spent_l938_93849


namespace elapsed_time_l938_93883

variable (totalDistance : ℕ) (runningSpeed : ℕ) (distanceRemaining : ℕ)

theorem elapsed_time (h1 : totalDistance = 120) (h2 : runningSpeed = 4) (h3 : distanceRemaining = 20) :
  (totalDistance - distanceRemaining) / runningSpeed = 25 := by
sorry

end elapsed_time_l938_93883


namespace fraction_of_white_surface_area_is_11_16_l938_93873

theorem fraction_of_white_surface_area_is_11_16 :
  let cube_surface_area := 6 * 4^2
  let total_surface_faces := 96
  let corner_black_faces := 8 * 3
  let center_black_faces := 6 * 1
  let total_black_faces := corner_black_faces + center_black_faces
  let white_faces := total_surface_faces - total_black_faces
  (white_faces : ℚ) / total_surface_faces = 11 / 16 := 
by sorry

end fraction_of_white_surface_area_is_11_16_l938_93873
