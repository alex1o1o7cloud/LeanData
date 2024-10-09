import Mathlib

namespace find_numbers_l2137_213731

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

noncomputable def number1 := 986
noncomputable def number2 := 689

theorem find_numbers :
  is_three_digit_number number1 ∧ is_three_digit_number number2 ∧
  hundreds_digit number1 = units_digit number2 ∧ hundreds_digit number2 = units_digit number1 ∧
  number1 - number2 = 297 ∧ (hundreds_digit number2 + tens_digit number2 + units_digit number2) = 23 :=
by
  sorry

end find_numbers_l2137_213731


namespace num_dimes_l2137_213714

/--
Given eleven coins consisting of pennies, nickels, dimes, quarters, and half-dollars,
having a total value of $1.43, with at least one coin of each type,
prove that there must be exactly 4 dimes.
-/
theorem num_dimes (p n d q h : ℕ) :
  1 ≤ p ∧ 1 ≤ n ∧ 1 ≤ d ∧ 1 ≤ q ∧ 1 ≤ h ∧ 
  p + n + d + q + h = 11 ∧ 
  (1 * p + 5 * n + 10 * d + 25 * q + 50 * h) = 143
  → d = 4 :=
by
  sorry

end num_dimes_l2137_213714


namespace infinitenat_not_sum_square_prime_l2137_213765

theorem infinitenat_not_sum_square_prime : ∀ k : ℕ, ¬ ∃ (n : ℕ) (p : ℕ), Prime p ∧ (3 * k + 2) ^ 2 = n ^ 2 + p :=
by
  intro k
  sorry

end infinitenat_not_sum_square_prime_l2137_213765


namespace work_together_time_l2137_213797

theorem work_together_time (man_days : ℝ) (son_days : ℝ)
  (h_man : man_days = 5) (h_son : son_days = 7.5) :
  (1 / (1 / man_days + 1 / son_days)) = 3 :=
by
  -- Given the constraints, prove the result
  rw [h_man, h_son]
  sorry

end work_together_time_l2137_213797


namespace sum_of_first_17_terms_l2137_213710

variable {α : Type*} [LinearOrderedField α] 

-- conditions
def arithmetic_sequence (a : ℕ → α) : Prop := 
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

variable {a : ℕ → α}
variable {S : ℕ → α}

-- main theorem
theorem sum_of_first_17_terms (h_arith : arithmetic_sequence a)
  (h_S : sum_of_first_n_terms a S)
  (h_condition : a 7 + a 12 = 12 - a 8) :
  S 17 = 68 := sorry

end sum_of_first_17_terms_l2137_213710


namespace c_minus_b_seven_l2137_213741

theorem c_minus_b_seven {a b c d : ℕ} (ha : a^6 = b^5) (hb : c^4 = d^3) (hc : c - a = 31) : c - b = 7 :=
sorry

end c_minus_b_seven_l2137_213741


namespace at_least_one_of_p_or_q_true_l2137_213795

variable (p q : Prop)

theorem at_least_one_of_p_or_q_true (h : ¬(p ∨ q) = false) : p ∨ q :=
by 
  sorry

end at_least_one_of_p_or_q_true_l2137_213795


namespace triangle_perimeter_l2137_213749

theorem triangle_perimeter (a b c : ℕ) (ha : a = 14) (hb : b = 8) (hc : c = 9) : a + b + c = 31 := 
by
  sorry

end triangle_perimeter_l2137_213749


namespace sqrt_fraction_identity_l2137_213799

theorem sqrt_fraction_identity (n : ℕ) (h : n > 0) : 
    Real.sqrt ((1 : ℝ) / n - (1 : ℝ) / (n * n)) = Real.sqrt (n - 1) / n :=
by
  sorry

end sqrt_fraction_identity_l2137_213799


namespace find_ABC_l2137_213746

noncomputable def g (x : ℝ) (A B C : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem find_ABC : 
  ∀ (A B C : ℝ),
  (∀ (x : ℝ), x > 2 → g x A B C > 0.3) →
  (∃ (A : ℤ), A = 4) →
  (∃ (B : ℤ), ∃ (C : ℤ), A = 4 ∧ B = 8 ∧ C = -12) →
  A + B + C = 0 :=
by
  intros A B C h1 h2 h3
  rcases h2 with ⟨intA, h2'⟩
  rcases h3 with ⟨intB, ⟨intC, h3'⟩⟩
  simp [h2', h3']
  sorry -- proof skipped

end find_ABC_l2137_213746


namespace solve_for_x_l2137_213777

theorem solve_for_x (x : ℝ) (h : (x / 6) / 3 = 9 / (x / 3)) : x = 9 * Real.sqrt 6 ∨ x = - (9 * Real.sqrt 6) :=
by
  sorry

end solve_for_x_l2137_213777


namespace Lin_finishes_reading_on_Monday_l2137_213722

theorem Lin_finishes_reading_on_Monday :
  let start_day := "Tuesday"
  let book_days : ℕ → ℕ := fun n => n
  let total_books := 10
  let total_days := (total_books * (total_books + 1)) / 2
  let days_in_a_week := 7
  let finish_day_offset := total_days % days_in_a_week
  let day_names := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  (day_names.indexOf start_day + finish_day_offset) % days_in_a_week = day_names.indexOf "Monday" :=
by
  sorry

end Lin_finishes_reading_on_Monday_l2137_213722


namespace weekly_earnings_l2137_213740

theorem weekly_earnings (total_earnings : ℕ) (weeks : ℕ) (h1 : total_earnings = 133) (h2 : weeks = 19) : 
  round (total_earnings / weeks : ℝ) = 7 := 
by 
  sorry

end weekly_earnings_l2137_213740


namespace sarah_age_ratio_l2137_213762

theorem sarah_age_ratio 
  (S M : ℕ) 
  (h1 : S = 3 * (S / 3))
  (h2 : S - M = 5 * (S / 3 - 2 * M)) : 
  S / M = 27 / 2 := 
sorry

end sarah_age_ratio_l2137_213762


namespace nth_term_arithmetic_sequence_l2137_213783

variable (n r : ℕ)

def S (n : ℕ) : ℕ := 4 * n + 5 * n^2

theorem nth_term_arithmetic_sequence :
  (S r) - (S (r-1)) = 10 * r - 1 :=
by
  sorry

end nth_term_arithmetic_sequence_l2137_213783


namespace karlson_max_eat_chocolates_l2137_213794

noncomputable def maximum_chocolates_eaten : ℕ :=
  34 * (34 - 1) / 2

theorem karlson_max_eat_chocolates : maximum_chocolates_eaten = 561 := by
  sorry

end karlson_max_eat_chocolates_l2137_213794


namespace guide_is_knight_l2137_213730

-- Definitions
def knight (p : Prop) : Prop := p
def liar (p : Prop) : Prop := ¬p

-- Conditions
variable (GuideClaimsKnight : Prop)
variable (SecondResidentClaimsKnight : Prop)
variable (GuideReportsAccurately : Prop)

-- Proof problem
theorem guide_is_knight
  (GuideClaimsKnight : Prop)
  (SecondResidentClaimsKnight : Prop)
  (GuideReportsAccurately : (GuideClaimsKnight ↔ SecondResidentClaimsKnight)) :
  GuideClaimsKnight := 
sorry

end guide_is_knight_l2137_213730


namespace find_percentage_reduction_l2137_213782

-- Given the conditions of the problem.
def original_price : ℝ := 7500
def current_price: ℝ := 4800
def percentage_reduction (x : ℝ) : Prop := (original_price * (1 - x)^2 = current_price)

-- The statement we need to prove:
theorem find_percentage_reduction (x : ℝ) (h : percentage_reduction x) : x = 0.2 :=
by
  sorry

end find_percentage_reduction_l2137_213782


namespace initial_group_machines_l2137_213755

-- Define the number of bags produced by n machines in one minute and 150 machines in one minute
def bags_produced (machines : ℕ) (bags_per_minute : ℕ) : Prop :=
  machines * bags_per_minute = 45

def bags_produced_150 (bags_produced_in_8_mins : ℕ) : Prop :=
  150 * (bags_produced_in_8_mins / 8) = 450

-- Given the conditions, prove that the number of machines in the initial group is 15
theorem initial_group_machines (n : ℕ) (bags_produced_in_8_mins : ℕ) :
  bags_produced n 45 → bags_produced_150 bags_produced_in_8_mins → n = 15 :=
by
  intro h1 h2
  -- use the conditions to derive the result
  sorry

end initial_group_machines_l2137_213755


namespace each_parent_suitcases_l2137_213789

namespace SuitcaseProblem

-- Definitions based on conditions
def siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def total_suitcases : Nat := 14

-- Theorem statement corresponding to the question and correct answer
theorem each_parent_suitcases (suitcases_per_parent : Nat) :
  (siblings * suitcases_per_sibling + 2 * suitcases_per_parent = total_suitcases) →
  suitcases_per_parent = 3 := by
  intro h
  sorry

end SuitcaseProblem

end each_parent_suitcases_l2137_213789


namespace point_coordinates_in_second_quadrant_l2137_213737

theorem point_coordinates_in_second_quadrant
    (P : ℝ × ℝ)
    (h1 : P.1 < 0)
    (h2 : P.2 > 0)
    (h3 : |P.2| = 4)
    (h4 : |P.1| = 5) :
    P = (-5, 4) :=
sorry

end point_coordinates_in_second_quadrant_l2137_213737


namespace diff_of_squares_l2137_213771

theorem diff_of_squares (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 :=
by
  sorry

end diff_of_squares_l2137_213771


namespace remove_five_yields_average_10_5_l2137_213761

def numberList : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def averageRemaining (l : List ℕ) : ℚ :=
  (List.sum l : ℚ) / l.length

theorem remove_five_yields_average_10_5 :
  averageRemaining (numberList.erase 5) = 10.5 :=
sorry

end remove_five_yields_average_10_5_l2137_213761


namespace total_length_XYZ_l2137_213744

theorem total_length_XYZ :
  let straight_segments := 7
  let slanted_segments := 7 * Real.sqrt 2
  straight_segments + slanted_segments = 7 + 7 * Real.sqrt 2 :=
by
  sorry

end total_length_XYZ_l2137_213744


namespace slope_of_line_l2137_213716

theorem slope_of_line (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 3)) :
  (Q.snd - P.snd) / (Q.fst - P.fst) = 1 / 3 := by
  sorry

end slope_of_line_l2137_213716


namespace distance_from_hyperbola_focus_to_line_l2137_213745

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l2137_213745


namespace area_of_rectangle_l2137_213751

theorem area_of_rectangle (length width : ℝ) (h1 : length = 15) (h2 : width = length * 0.9) : length * width = 202.5 := by
  sorry

end area_of_rectangle_l2137_213751


namespace linear_regression_eq_l2137_213743

noncomputable def x_vals : List ℝ := [3, 7, 11]
noncomputable def y_vals : List ℝ := [10, 20, 24]

theorem linear_regression_eq :
  ∃ a b : ℝ, (a = 5.75) ∧ (b = 1.75) ∧ (∀ x, ∃ y, y = a + b * x) := sorry

end linear_regression_eq_l2137_213743


namespace length_of_ae_l2137_213735

-- Define the given consecutive points
variables (a b c d e : ℝ)

-- Conditions from the problem
-- 1. Points a, b, c, d, e are 5 consecutive points on a straight line - implicitly assumed on the same line
-- 2. bc = 2 * cd
-- 3. de = 4
-- 4. ab = 5
-- 5. ac = 11

theorem length_of_ae 
  (h1 : b - a = 5) -- ab = 5
  (h2 : c - a = 11) -- ac = 11
  (h3 : c - b = 2 * (d - c)) -- bc = 2 * cd
  (h4 : e - d = 4) -- de = 4
  : (e - a) = 18 := sorry

end length_of_ae_l2137_213735


namespace factorial_equation_solution_l2137_213715

theorem factorial_equation_solution (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  a.factorial * b.factorial = a.factorial + b.factorial + c.factorial → a = 3 ∧ b = 3 ∧ c = 4 := by
  sorry

end factorial_equation_solution_l2137_213715


namespace solve_inequality_l2137_213713

open Set Real

noncomputable def inequality_solution_set : Set ℝ :=
  {x | -2 ≤ x ∧ x ≤ 2} ∪ {6}

theorem solve_inequality (x : ℝ) :
  (x ^ 2 - 4) * (x - 6) ^ 2 ≤ 0 ↔ x ∈ inequality_solution_set := 
sorry

end solve_inequality_l2137_213713


namespace max_difference_primes_l2137_213705

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def even_integer : ℕ := 138

theorem max_difference_primes (p q : ℕ) :
  is_prime p ∧ is_prime q ∧ p + q = even_integer ∧ p ≠ q →
  (q - p) = 124 :=
by
  sorry

end max_difference_primes_l2137_213705


namespace inequality_is_linear_l2137_213775

theorem inequality_is_linear (k : ℝ) (h1 : (|k| - 1) = 1) (h2 : (k + 2) ≠ 0) : k = 2 :=
sorry

end inequality_is_linear_l2137_213775


namespace false_prop_range_of_a_l2137_213768

theorem false_prop_range_of_a (a : ℝ) :
  (¬ ∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ (a < -2 * Real.sqrt 2 ∨ a > 2 * Real.sqrt 2) :=
by
  sorry

end false_prop_range_of_a_l2137_213768


namespace circle_possible_values_l2137_213723

theorem circle_possible_values (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a * x + 2 * a * y + 2 * a^2 + a - 1 = 0 → -2 < a ∧ a < 2/3) := sorry

end circle_possible_values_l2137_213723


namespace circles_equal_or_tangent_l2137_213747

theorem circles_equal_or_tangent (a b c : ℝ) 
  (h : (2 * a)^2 - 4 * (b^2 - c * (b - a)) = 0) : 
  a = b ∨ c = a + b :=
by
  -- Will fill the proof later
  sorry

end circles_equal_or_tangent_l2137_213747


namespace original_price_l2137_213704

variable (P : ℝ)
variable (S : ℝ := 140)
variable (discount : ℝ := 0.60)

theorem original_price :
  (S = P * (1 - discount)) → (P = 350) :=
by
  sorry

end original_price_l2137_213704


namespace problem_1_problem_2_problem_3_l2137_213787

theorem problem_1 (avg_daily_production : ℕ) (deviation_wed : ℤ) :
  avg_daily_production = 3000 →
  deviation_wed = -15 →
  avg_daily_production + deviation_wed = 2985 :=
by intros; sorry

theorem problem_2 (avg_daily_production : ℕ) (deviation_sat : ℤ) (deviation_fri : ℤ) :
  avg_daily_production = 3000 →
  deviation_sat = 68 →
  deviation_fri = -20 →
  (avg_daily_production + deviation_sat) - (avg_daily_production + deviation_fri) = 88 :=
by intros; sorry

theorem problem_3 (planned_weekly_production : ℕ) (deviations : List ℤ) :
  planned_weekly_production = 21000 →
  deviations = [35, -12, -15, 30, -20, 68, -9] →
  planned_weekly_production + deviations.sum = 21077 :=
by intros; sorry

end problem_1_problem_2_problem_3_l2137_213787


namespace simplify_expression_evaluate_l2137_213727

theorem simplify_expression_evaluate : 
  let x := 1
  let y := 2
  (2 * x - y) * (y + 2 * x) - (2 * y + x) * (2 * y - x) = -15 :=
by
  sorry

end simplify_expression_evaluate_l2137_213727


namespace probability_exactly_half_red_balls_l2137_213700

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k))) * p^k * (1 - p)^(n - k)

theorem probability_exactly_half_red_balls :
  binomial_probability 8 4 (1/2) = 35/128 :=
by
  sorry

end probability_exactly_half_red_balls_l2137_213700


namespace not_perfect_square_2023_l2137_213719

theorem not_perfect_square_2023 : ¬ (∃ x : ℤ, x^2 = 5^2023) := 
sorry

end not_perfect_square_2023_l2137_213719


namespace positive_integer_solutions_l2137_213796

theorem positive_integer_solutions (n m : ℕ) (h : n > 0 ∧ m > 0) : 
  (n + 1) * m = n! + 1 ↔ (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 4 ∧ m = 5) := by
  sorry

end positive_integer_solutions_l2137_213796


namespace g_domain_l2137_213708

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arccos (x^3))

theorem g_domain : { x : ℝ | -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0 } = (Set.Icc (-1) 0 ∪ Set.Icc 0 1) \ {0} :=
by
  sorry

end g_domain_l2137_213708


namespace union_A_B_intersection_complement_A_B_l2137_213725

open Set Real

noncomputable def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}
noncomputable def B : Set ℝ := {x : ℝ | abs (2 * x + 1) ≤ 1}

theorem union_A_B : A ∪ B = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by
  sorry

theorem intersection_complement_A_B : (Aᶜ) ∩ (Bᶜ) = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

end union_A_B_intersection_complement_A_B_l2137_213725


namespace polynomial_inequality_l2137_213752

theorem polynomial_inequality
  (x1 x2 x3 a b c : ℝ)
  (h1 : x1 > 0) 
  (h2 : x2 > 0) 
  (h3 : x3 > 0)
  (h4 : x1 + x2 + x3 ≤ 1)
  (h5 : x1^3 + a * x1^2 + b * x1 + c = 0)
  (h6 : x2^3 + a * x2^2 + b * x2 + c = 0)
  (h7 : x3^3 + a * x3^2 + b * x3 + c = 0) :
  a^3 * (1 + a + b) - 9 * c * (3 + 3 * a + a^2) ≤ 0 :=
sorry

end polynomial_inequality_l2137_213752


namespace parabola_standard_eq_l2137_213756

theorem parabola_standard_eq (p p' : ℝ) (h₁ : p > 0) (h₂ : p' > 0) :
  (∀ (x y : ℝ), (x^2 = 2 * p * y ∨ y^2 = -2 * p' * x) → 
  (x = -2 ∧ y = 4 → (x^2 = y ∨ y^2 = -8 * x))) :=
by
  sorry

end parabola_standard_eq_l2137_213756


namespace vaclav_multiplication_correct_l2137_213772

-- Definitions of the involved numbers and their multiplication consistency.
def a : ℕ := 452
def b : ℕ := 125
def result : ℕ := 56500

-- The main theorem statement proving the correctness of the multiplication.
theorem vaclav_multiplication_correct : a * b = result :=
by sorry

end vaclav_multiplication_correct_l2137_213772


namespace minimum_time_to_cook_3_pancakes_l2137_213766

theorem minimum_time_to_cook_3_pancakes (can_fry_two_pancakes_at_a_time : Prop) 
   (time_to_fully_cook_one_pancake : ℕ) (time_to_cook_one_side : ℕ) :
  can_fry_two_pancakes_at_a_time →
  time_to_fully_cook_one_pancake = 2 →
  time_to_cook_one_side = 1 →
  3 = 3 := 
by
  intros
  sorry

end minimum_time_to_cook_3_pancakes_l2137_213766


namespace jaylen_has_2_cucumbers_l2137_213786

-- Definitions based on given conditions
def carrots_jaylen := 5
def bell_peppers_kristin := 2
def green_beans_kristin := 20
def total_vegetables_jaylen := 18

def bell_peppers_jaylen := 2 * bell_peppers_kristin
def green_beans_jaylen := (green_beans_kristin / 2) - 3

def known_vegetables_jaylen := carrots_jaylen + bell_peppers_jaylen + green_beans_jaylen
def cucumbers_jaylen := total_vegetables_jaylen - known_vegetables_jaylen

-- The theorem to prove
theorem jaylen_has_2_cucumbers : cucumbers_jaylen = 2 :=
by
  -- We'll place the proof here
  sorry

end jaylen_has_2_cucumbers_l2137_213786


namespace functional_equation_solution_l2137_213759

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ a : ℝ, ∀ x : ℝ, f x = x - a :=
by
  intro h
  sorry

end functional_equation_solution_l2137_213759


namespace divisibility_by_cube_greater_than_1_l2137_213733

theorem divisibility_by_cube_greater_than_1 (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b)
  (hdiv : (a + b^3) % (a^2 + 3 * a * b + 3 * b^2 - 1) = 0) :
  ∃ k : ℕ, 1 < k ∧ k^3 ∣ a^2 + 3 * a * b + 3 * b^2 - 1 := 
by {
  sorry
}

end divisibility_by_cube_greater_than_1_l2137_213733


namespace find_x_positive_multiple_of_8_l2137_213784

theorem find_x_positive_multiple_of_8 (x : ℕ) 
  (h1 : ∃ k, x = 8 * k) 
  (h2 : x^2 > 100) 
  (h3 : x < 20) : x = 16 :=
by
  sorry

end find_x_positive_multiple_of_8_l2137_213784


namespace no_sum_of_19_l2137_213769

theorem no_sum_of_19 (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6)
  (hprod : a * b * c * d = 180) : a + b + c + d ≠ 19 :=
sorry

end no_sum_of_19_l2137_213769


namespace g_at_5_l2137_213778

def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ (x : ℝ), g x + 2 * g (1 - x) = x^2 + 2 * x

theorem g_at_5 : g 5 = -19 / 3 :=
by {
  sorry
}

end g_at_5_l2137_213778


namespace shop_width_l2137_213729

theorem shop_width 
  (monthly_rent : ℝ) 
  (shop_length : ℝ) 
  (annual_rent_per_sqft : ℝ) 
  (width : ℝ) 
  (monthly_rent_eq : monthly_rent = 2244) 
  (shop_length_eq : shop_length = 22) 
  (annual_rent_per_sqft_eq : annual_rent_per_sqft = 68) 
  (width_eq : width = 18) : 
  (12 * monthly_rent) / annual_rent_per_sqft / shop_length = width := 
by 
  sorry

end shop_width_l2137_213729


namespace joan_paid_230_l2137_213748

theorem joan_paid_230 (J K : ℝ) (h1 : J + K = 600) (h2 : 2 * J = K + 90) : J = 230 :=
sorry

end joan_paid_230_l2137_213748


namespace triangle_inequality_for_min_segments_l2137_213717

theorem triangle_inequality_for_min_segments
  (a b c d : ℝ)
  (a1 b1 c1 : ℝ)
  (h1 : a1 = min a d)
  (h2 : b1 = min b d)
  (h3 : c1 = min c d)
  (h_triangle : c < a + b) :
  a1 + b1 > c1 ∧ a1 + c1 > b1 ∧ b1 + c1 > a1 := sorry

end triangle_inequality_for_min_segments_l2137_213717


namespace inscribed_sphere_radius_eq_l2137_213750

noncomputable def inscribed_sphere_radius (b α : ℝ) : ℝ :=
  b * (Real.sin α) / (4 * (Real.cos (α / 4))^2)

theorem inscribed_sphere_radius_eq
  (b α : ℝ) 
  (h1 : 0 < b)
  (h2 : 0 < α ∧ α < Real.pi) 
  : inscribed_sphere_radius b α = b * (Real.sin α) / (4 * (Real.cos (α / 4))^2) :=
sorry

end inscribed_sphere_radius_eq_l2137_213750


namespace problem_l2137_213758

theorem problem : (1 * (2 + 3) * 4 * 5) = 100 := by
  sorry

end problem_l2137_213758


namespace distance_PF_l2137_213701

-- Definitions for the given conditions
structure Rectangle :=
  (EF GH: ℝ)
  (interior_point : ℝ × ℝ)
  (PE : ℝ)
  (PH : ℝ)
  (PG : ℝ)

-- The theorem to prove PF equals 12 under the given conditions
theorem distance_PF 
  (r : Rectangle)
  (hPE : r.PE = 5)
  (hPH : r.PH = 12)
  (hPG : r.PG = 13) :
  ∃ PF, PF = 12 := 
sorry

end distance_PF_l2137_213701


namespace constant_function_odd_iff_zero_l2137_213707

theorem constant_function_odd_iff_zero (k : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = k) 
  (h2 : ∀ x, f (-x) = -f x) : 
  k = 0 :=
sorry

end constant_function_odd_iff_zero_l2137_213707


namespace inequality_proof_l2137_213712

theorem inequality_proof (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  a^3 * b + b^3 * c + c^3 * a ≥ a^2 * b * c + b^2 * c * a + c^2 * a * b :=
by {
  sorry
}

end inequality_proof_l2137_213712


namespace find_a_l2137_213776

-- Define the slopes of the lines and the condition that they are perpendicular.
def slope1 (a : ℝ) : ℝ := a
def slope2 (a : ℝ) : ℝ := a + 2

-- The main statement of our problem.
theorem find_a (a : ℝ) (h : slope1 a * slope2 a = -1) : a = -1 :=
sorry

end find_a_l2137_213776


namespace find_initial_mice_l2137_213702

theorem find_initial_mice : 
  ∃ x : ℕ, (∀ (h1 : ∀ (m : ℕ), m * 2 = m + m), (35 * x = 280) → x = 8) :=
by
  existsi 8
  intro h1 h2
  sorry

end find_initial_mice_l2137_213702


namespace sum_of_reciprocals_l2137_213779

variables {a b : ℕ}

def HCF (m n : ℕ) : ℕ := m.gcd n
def LCM (m n : ℕ) : ℕ := m.lcm n

theorem sum_of_reciprocals (h_sum : a + b = 55)
                           (h_hcf : HCF a b = 5)
                           (h_lcm : LCM a b = 120) :
  (1 / a : ℚ) + (1 / b) = 11 / 120 :=
sorry

end sum_of_reciprocals_l2137_213779


namespace number_of_four_digit_numbers_l2137_213780

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l2137_213780


namespace quadrilateral_area_is_8_l2137_213791

noncomputable section
open Real

def f1 : ℝ × ℝ := (-2, 0)
def f2 : ℝ × ℝ := (2, 0)

def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

def origin_symmetric (P Q : ℝ × ℝ) : Prop := P.1 = -Q.1 ∧ P.2 = -Q.2

def distance (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def is_quadrilateral (P Q F1 F2 : ℝ × ℝ) : Prop :=
  ∃ a b c d, a = P ∧ b = F1 ∧ c = Q ∧ d = F2

def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1*B.2 + B.1*C.2 + C.1*D.2 + D.1*A.2 - (B.1*A.2 + C.1*B.2 + D.1*C.2 + A.1*D.2))

theorem quadrilateral_area_is_8 (P Q : ℝ × ℝ) :
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  origin_symmetric P Q →
  distance P Q = distance f1 f2 →
  is_quadrilateral P Q f1 f2 →
  area_of_quadrilateral P f1 Q f2 = 8 := 
by
  sorry

end quadrilateral_area_is_8_l2137_213791


namespace acres_used_for_corn_l2137_213757

-- Define the conditions given in the problem
def total_land : ℕ := 1034
def ratio_beans : ℕ := 5
def ratio_wheat : ℕ := 2
def ratio_corn : ℕ := 4
def total_ratio_parts : ℕ := ratio_beans + ratio_wheat + ratio_corn
def part_size : ℕ := total_land / total_ratio_parts

-- State the theorem to prove that the land used for corn is 376 acres
theorem acres_used_for_corn : (part_size * ratio_corn = 376) :=
  sorry

end acres_used_for_corn_l2137_213757


namespace monotonic_increasing_interval_l2137_213726

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x + 6)

theorem monotonic_increasing_interval : 
  ∀ x y : ℝ, x < y → y < 1 → f x < f y :=
by
  sorry

end monotonic_increasing_interval_l2137_213726


namespace correct_equation_l2137_213770

theorem correct_equation : ∃a : ℝ, (-3 * a) ^ 2 = 9 * a ^ 2 :=
by
  use 1
  sorry

end correct_equation_l2137_213770


namespace eight_sharp_two_equals_six_thousand_l2137_213734

def new_operation (a b : ℕ) : ℕ :=
  (a + b) ^ 3 * (a - b)

theorem eight_sharp_two_equals_six_thousand : new_operation 8 2 = 6000 := 
  by
    sorry

end eight_sharp_two_equals_six_thousand_l2137_213734


namespace max_n_positive_l2137_213703

theorem max_n_positive (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : S 15 > 0)
  (h2 : S 16 < 0)
  (hs1 : S 15 = 15 * (a 8))
  (hs2 : S 16 = 8 * (a 8 + a 9)) :
  (∀ n, a n > 0 → n ≤ 8) :=
by {
    sorry
}

end max_n_positive_l2137_213703


namespace inradius_triangle_l2137_213788

theorem inradius_triangle (p A : ℝ) (h1 : p = 39) (h2 : A = 29.25) :
  ∃ r : ℝ, A = (1 / 2) * r * p ∧ r = 1.5 := by
  sorry

end inradius_triangle_l2137_213788


namespace central_angle_of_sector_in_unit_circle_with_area_1_is_2_l2137_213767

theorem central_angle_of_sector_in_unit_circle_with_area_1_is_2 :
  ∀ (θ : ℝ), (∀ (r : ℝ), (r = 1) → (1 / 2 * r^2 * θ = 1) → θ = 2) :=
by
  intros θ r hr h
  sorry

end central_angle_of_sector_in_unit_circle_with_area_1_is_2_l2137_213767


namespace set_union_l2137_213785

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem set_union : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

end set_union_l2137_213785


namespace new_price_of_sugar_l2137_213721

theorem new_price_of_sugar (C : ℝ) (H : 10 * C = P * (0.7692307692307693 * C)) : P = 13 := by
  sorry

end new_price_of_sugar_l2137_213721


namespace quadrilateral_ABCD_is_rectangle_l2137_213742

noncomputable def point := (ℤ × ℤ)

def A : point := (-2, 0)
def B : point := (1, 6)
def C : point := (5, 4)
def D : point := (2, -2)

def vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def dot_product (v1 v2 : point) : ℤ := (v1.1 * v2.1) + (v1.2 * v2.2)

def is_perpendicular (v1 v2 : point) : Prop := dot_product v1 v2 = 0

def is_rectangle (A B C D : point) :=
  vector A B = vector C D ∧ is_perpendicular (vector A B) (vector A D)

theorem quadrilateral_ABCD_is_rectangle : is_rectangle A B C D :=
by
  sorry

end quadrilateral_ABCD_is_rectangle_l2137_213742


namespace arrangement_count_l2137_213754

-- Define the problem conditions: 3 male students and 2 female students.
def male_students : ℕ := 3
def female_students : ℕ := 2
def total_students : ℕ := male_students + female_students

-- Define the condition that female students do not stand at either end.
def valid_positions_for_female : Finset ℕ := {1, 2, 3}
def valid_positions_for_male : Finset ℕ := {0, 4}

-- Theorem statement: the total number of valid arrangements is 36.
theorem arrangement_count : ∃ (n : ℕ), n = 36 := sorry

end arrangement_count_l2137_213754


namespace intersect_once_l2137_213709

theorem intersect_once (x : ℝ) : 
  (∀ y, y = 3 * Real.log x ↔ y = Real.log (3 * x)) → (∃! x, 3 * Real.log x = Real.log (3 * x)) :=
by 
  sorry

end intersect_once_l2137_213709


namespace negation_of_existential_l2137_213732

theorem negation_of_existential :
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * x_0 - 3 > 0) = (∀ x : ℝ, x^2 + 2 * x - 3 ≤ 0) := 
by
  sorry

end negation_of_existential_l2137_213732


namespace range_of_m_l2137_213738

theorem range_of_m (m : ℝ) (H : ∀ x, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) : m < -1 / 2 :=
sorry

end range_of_m_l2137_213738


namespace jane_cycling_time_difference_l2137_213773

theorem jane_cycling_time_difference :
  (3 * 5 / 6.5 - (5 / 10 + 5 / 5 + 5 / 8)) * 60 = 11 :=
by sorry

end jane_cycling_time_difference_l2137_213773


namespace arithmetic_sequence_sum_l2137_213718

theorem arithmetic_sequence_sum {a : ℕ → ℝ}
  (h1 : a 1 + a 5 = 6) 
  (h2 : a 2 + a 14 = 26) :
  (10 / 2) * (a 1 + a 10) = 80 :=
by sorry

end arithmetic_sequence_sum_l2137_213718


namespace minimum_n_for_candy_purchases_l2137_213774

theorem minimum_n_for_candy_purchases' {o s p : ℕ} (h1 : 9 * o = 10 * s) (h2 : 9 * o = 20 * p) : 
  ∃ n : ℕ, 30 * n = 180 ∧ ∀ m : ℕ, (30 * m = 9 * o) → n ≤ m :=
by sorry

end minimum_n_for_candy_purchases_l2137_213774


namespace eccentricity_of_ellipse_l2137_213711

open Real

def ellipse_eq (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def foci_dist_eq (a c : ℝ) : Prop :=
  2 * c / (2 * a) = sqrt 6 / 2

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

theorem eccentricity_of_ellipse (a b x y c : ℝ)
  (h1 : ellipse_eq a b x y)
  (h2 : foci_dist_eq a c) :
  eccentricity c a = sqrt 6 / 3 :=
sorry

end eccentricity_of_ellipse_l2137_213711


namespace initial_books_in_library_l2137_213706

theorem initial_books_in_library
  (books_out_tuesday : ℕ)
  (books_in_thursday : ℕ)
  (books_out_friday : ℕ)
  (final_books : ℕ)
  (h1 : books_out_tuesday = 227)
  (h2 : books_in_thursday = 56)
  (h3 : books_out_friday = 35)
  (h4 : final_books = 29) : 
  initial_books = 235 :=
by
  sorry

end initial_books_in_library_l2137_213706


namespace arithmetic_sqrt_sqrt_16_eq_2_l2137_213753

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l2137_213753


namespace fill_tub_together_time_l2137_213793

theorem fill_tub_together_time :
  let rate1 := 1 / 4
  let rate2 := 1 / 4
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  combined_rate ≠ 0 → (1 / combined_rate = 12 / 7) :=
by
  let rate1 := 1 / 4
  let rate2 := 1 / 4
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  sorry

end fill_tub_together_time_l2137_213793


namespace vector_condition_l2137_213728

def vec_a : ℝ × ℝ := (5, 2)
def vec_b : ℝ × ℝ := (-4, -3)
def vec_c : ℝ × ℝ := (-23, -12)

theorem vector_condition : 3 • (vec_a.1, vec_a.2) - 2 • (vec_b.1, vec_b.2) + vec_c = (0, 0) :=
by
  sorry

end vector_condition_l2137_213728


namespace calculate_negative_subtraction_l2137_213724

theorem calculate_negative_subtraction : -2 - (-3) = 1 :=
by sorry

end calculate_negative_subtraction_l2137_213724


namespace find_intersection_l2137_213763

noncomputable def A : Set ℝ := { x | -4 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x | x ≤ 2 }

theorem find_intersection : A ∩ B = { x | -4 < x ∧ x ≤ 2 } := sorry

end find_intersection_l2137_213763


namespace impossible_path_2018_grid_l2137_213736

theorem impossible_path_2018_grid :
  ¬((∃ (path : Finset (Fin 2018 × Fin 2018)), 
    (0, 0) ∈ path ∧ (2017, 2017) ∈ path ∧ 
    (∀ {x y}, (x, y) ∈ path → (x + 1, y) ∈ path ∨ (x, y + 1) ∈ path ∨ (x - 1, y) ∈ path ∨ (x, y - 1) ∈ path) ∧ 
    (∀ {x y}, (x, y) ∈ path → (Finset.card path = 2018 * 2018)))) :=
by 
  sorry

end impossible_path_2018_grid_l2137_213736


namespace q1_q2_l2137_213764

variable (a b : ℝ)

-- Definition of the conditions
def conditions : Prop := a + b = 7 ∧ a * b = 6

-- Statement of the first question
theorem q1 (h : conditions a b) : a^2 + b^2 = 37 := sorry

-- Statement of the second question
theorem q2 (h : conditions a b) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = 150 := sorry

end q1_q2_l2137_213764


namespace professionals_work_days_l2137_213739

theorem professionals_work_days (cost_per_hour_1 cost_per_hour_2 hours_per_day total_cost : ℝ) (h_cost1: cost_per_hour_1 = 15) (h_cost2: cost_per_hour_2 = 15) (h_hours: hours_per_day = 6) (h_total: total_cost = 1260) : (∃ d : ℝ, total_cost = d * hours_per_day * (cost_per_hour_1 + cost_per_hour_2) ∧ d = 7) :=
by
  use 7
  rw [h_cost1, h_cost2, h_hours, h_total]
  simp
  sorry

end professionals_work_days_l2137_213739


namespace maximum_area_of_rectangle_with_given_perimeter_l2137_213720

noncomputable def perimeter : ℝ := 30
noncomputable def area (length width : ℝ) : ℝ := length * width
noncomputable def max_area : ℝ := 56.25

theorem maximum_area_of_rectangle_with_given_perimeter :
  ∃ length width : ℝ, 2 * length + 2 * width = perimeter ∧ area length width = max_area :=
sorry

end maximum_area_of_rectangle_with_given_perimeter_l2137_213720


namespace polynomial_identity_l2137_213781

open Polynomial

-- Definition of the non-zero polynomial of interest
noncomputable def p (a : ℝ) : Polynomial ℝ := Polynomial.C a * (Polynomial.X ^ 3 - Polynomial.X)

-- Theorem stating that, for all x, the given equation holds for the polynomial p
theorem polynomial_identity (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, (x - 1) * (p a).eval (x + 1) - (x + 2) * (p a).eval x = 0 :=
by
  sorry

end polynomial_identity_l2137_213781


namespace sequence_a_5_l2137_213790

noncomputable section

-- Definition of the sequence
def a : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => a (n + 1) + a n

-- Statement to prove that a 4 = 8 (in Lean, the sequence is zero-indexed, so a 4 is a_5)
theorem sequence_a_5 : a 4 = 8 :=
  by
    sorry

end sequence_a_5_l2137_213790


namespace water_pumping_problem_l2137_213760

theorem water_pumping_problem :
  let pumpA_rate := 300 -- gallons per hour
  let pumpB_rate := 500 -- gallons per hour
  let combined_rate := pumpA_rate + pumpB_rate -- Combined rate per hour
  let time_duration := 1 / 2 -- Time in hours (30 minutes)
  combined_rate * time_duration = 400 := -- Total volume in gallons
by
  -- Lean proof would go here
  sorry

end water_pumping_problem_l2137_213760


namespace range_of_m_l2137_213798

theorem range_of_m (m : ℝ) (P : ℝ × ℝ) (h : P = (m + 3, m - 5)) (quadrant4 : P.1 > 0 ∧ P.2 < 0) : -3 < m ∧ m < 5 :=
by
  sorry

end range_of_m_l2137_213798


namespace circle_inscribed_in_square_area_l2137_213792

theorem circle_inscribed_in_square_area :
  ∀ (x y : ℝ) (h : 2 * x^2 + 2 * y^2 - 8 * x - 12 * y + 24 = 0),
  ∃ side : ℝ, 4 * (side^2) = 16 :=
by
  sorry

end circle_inscribed_in_square_area_l2137_213792
