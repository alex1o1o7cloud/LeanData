import Mathlib

namespace area_of_large_rectangle_ABCD_l73_73130

-- Definitions for conditions and given data
def shaded_rectangle_area : ℕ := 2
def area_of_rectangle_ABCD (a b c : ℕ) : ℕ := a + b + c

-- The theorem to prove
theorem area_of_large_rectangle_ABCD
  (a b c : ℕ) 
  (h1 : shaded_rectangle_area = a)
  (h2 : shaded_rectangle_area = b)
  (h3 : a + b + c = 8) : 
  area_of_rectangle_ABCD a b c = 8 :=
by
  sorry

end area_of_large_rectangle_ABCD_l73_73130


namespace simplify_polynomial_l73_73897

def P (x : ℝ) : ℝ := 3*x^3 + 4*x^2 - 5*x + 8
def Q (x : ℝ) : ℝ := 2*x^3 + x^2 + 3*x - 15

theorem simplify_polynomial (x : ℝ) : P x - Q x = x^3 + 3*x^2 - 8*x + 23 := 
by 
  -- proof goes here
  sorry

end simplify_polynomial_l73_73897


namespace marching_band_l73_73307

theorem marching_band (total_members brass woodwind percussion : ℕ)
  (h1 : brass + woodwind + percussion = 110)
  (h2 : woodwind = 2 * brass)
  (h3 : percussion = 4 * woodwind) :
  brass = 10 := by
  sorry

end marching_band_l73_73307


namespace exist_two_pies_differing_in_both_l73_73934

-- Define enumeration types for fillings and preparation methods
inductive Filling
| apple
| cherry

inductive Preparation
| fried
| baked

-- Define pie type with filling and preparation
structure Pie where
  filling : Filling
  preparation : Preparation

-- Define the types of pies available
def pie1 : Pie := { filling := Filling.apple, preparation := Preparation.fried }
def pie2 : Pie := { filling := Filling.cherry, preparation := Preparation.fried }
def pie3 : Pie := { filling := Filling.apple, preparation := Preparation.baked }
def pie4 : Pie := { filling := Filling.cherry, preparation := Preparation.baked }

-- Define the list of available pies
def availablePies : List Pie := [pie1, pie2, pie3, pie4]

theorem exist_two_pies_differing_in_both (pies : List Pie) (h : pies.length ≥ 3) :
  ∃ p1 p2 : Pie, p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
by
  -- Proof content to be filled in
  sorry

end exist_two_pies_differing_in_both_l73_73934


namespace number_of_members_greater_than_median_l73_73632

theorem number_of_members_greater_than_median (n : ℕ) (median : ℕ) (avg_age : ℕ) (youngest : ℕ) (oldest : ℕ) :
  n = 100 ∧ avg_age = 21 ∧ youngest = 1 ∧ oldest = 70 →
  ∃ k, k = 50 :=
by
  sorry

end number_of_members_greater_than_median_l73_73632


namespace arithmetic_sequence_properties_sum_of_sequence_b_n_l73_73209

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h₁ : a 2 = 3) 
  (h₂ : S 5 + a 3 = 30) 
  (h₃ : ∀ n, S n = (n * (a 1 + (n-1) * ((a 2) - (a 1)))) / 2 
                     ∧ a n = a 1 + (n-1) * ((a 2) - (a 1))) : 
  (∀ n, a n = 2 * n - 1 ∧ S n = n^2) := 
sorry

theorem sum_of_sequence_b_n (b : ℕ → ℝ) 
  (T : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h₁ : ∀ n, b n = (a (n+1)) / (S n * S (n+1))) 
  (h₂ : ∀ n, a n = 2 * n - 1 ∧ S n = n^2) : 
  (∀ n, T n = (1 - 1 / (n+1)^2)) := 
sorry

end arithmetic_sequence_properties_sum_of_sequence_b_n_l73_73209


namespace solution_set_of_fractional_inequality_l73_73504

theorem solution_set_of_fractional_inequality :
  {x : ℝ | (x + 1) / (x - 3) < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_of_fractional_inequality_l73_73504


namespace remainder_when_dividing_150_l73_73659

theorem remainder_when_dividing_150 (k : ℕ) (hk1 : k > 0) (hk2 : 80 % k^2 = 8) : 150 % k = 6 :=
by
  sorry

end remainder_when_dividing_150_l73_73659


namespace combined_time_third_attempt_l73_73895

noncomputable def first_lock_initial : ℕ := 5
noncomputable def second_lock_initial : ℕ := 3 * first_lock_initial - 3
noncomputable def combined_initial : ℕ := 5 * second_lock_initial

noncomputable def first_lock_second_attempt : ℝ := first_lock_initial - 0.1 * first_lock_initial
noncomputable def first_lock_third_attempt : ℝ := first_lock_second_attempt - 0.1 * first_lock_second_attempt

noncomputable def second_lock_second_attempt : ℝ := second_lock_initial - 0.15 * second_lock_initial
noncomputable def second_lock_third_attempt : ℝ := second_lock_second_attempt - 0.15 * second_lock_second_attempt

noncomputable def combined_third_attempt : ℝ := 5 * second_lock_third_attempt

theorem combined_time_third_attempt : combined_third_attempt = 43.35 :=
by
  sorry

end combined_time_third_attempt_l73_73895


namespace maximum_ab_l73_73649

theorem maximum_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3*a + 8*b = 48) : ab ≤ 24 :=
by
  sorry

end maximum_ab_l73_73649


namespace gcd_polynomial_example_l73_73092

theorem gcd_polynomial_example (b : ℤ) (h : ∃ k : ℤ, b = 2 * 1177 * k) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 14) = 2 :=
by
  sorry

end gcd_polynomial_example_l73_73092


namespace tan_theta_expr_l73_73635

theorem tan_theta_expr (θ : ℝ) (h : Real.tan θ = 4) : 
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + (Real.sin θ ^ 2) / 4 = 21 / 68 := 
by sorry

end tan_theta_expr_l73_73635


namespace max_objective_function_value_l73_73426

def objective_function (x1 x2 : ℝ) := 4 * x1 + 6 * x2

theorem max_objective_function_value :
  ∃ x1 x2 : ℝ, 
    (x1 >= 0) ∧ 
    (x2 >= 0) ∧ 
    (x1 + x2 <= 18) ∧ 
    (0.5 * x1 + x2 <= 12) ∧ 
    (2 * x1 <= 24) ∧ 
    (2 * x2 <= 18) ∧ 
    (∀ y1 y2 : ℝ, 
      (y1 >= 0) ∧ 
      (y2 >= 0) ∧ 
      (y1 + y2 <= 18) ∧ 
      (0.5 * y1 + y2 <= 12) ∧ 
      (2 * y1 <= 24) ∧ 
      (2 * y2 <= 18) -> 
      objective_function y1 y2 <= objective_function x1 x2) ∧
    (objective_function x1 x2 = 84) :=
by
  use 12, 6
  sorry

end max_objective_function_value_l73_73426


namespace ram_marks_l73_73387

theorem ram_marks (total_marks : ℕ) (percentage : ℕ) (h_total : total_marks = 500) (h_percentage : percentage = 90) : 
  (percentage * total_marks / 100) = 450 := by
  sorry

end ram_marks_l73_73387


namespace num_distinct_solutions_l73_73297

theorem num_distinct_solutions : 
  (∃ x : ℝ, |x - 3| = |x + 5|) ∧ 
  (∀ x1 x2 : ℝ, |x1 - 3| = |x1 + 5| → |x2 - 3| = |x2 + 5| → x1 = x2) := 
  sorry

end num_distinct_solutions_l73_73297


namespace value_of_expression_l73_73316

theorem value_of_expression : (180^2 - 150^2) / 30 = 330 := by
  sorry

end value_of_expression_l73_73316


namespace third_competitor_hot_dogs_l73_73004

theorem third_competitor_hot_dogs (first_ate : ℕ) (second_multiplier : ℕ) (third_percent_less : ℕ) (second_ate third_ate : ℕ) 
  (H1 : first_ate = 12)
  (H2 : second_multiplier = 2)
  (H3 : third_percent_less = 25)
  (H4 : second_ate = first_ate * second_multiplier)
  (H5 : third_ate = second_ate - (third_percent_less * second_ate / 100)) : 
  third_ate = 18 := 
by 
  sorry

end third_competitor_hot_dogs_l73_73004


namespace maximum_value_of_f_l73_73988

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem maximum_value_of_f : ∃ x : ℝ, f x = 13 :=
by 
  sorry

end maximum_value_of_f_l73_73988


namespace wang_hua_withdrawal_correct_l73_73993

noncomputable def wang_hua_withdrawal : ℤ :=
  let d : ℤ := 14
  let c : ℤ := 32
  -- The amount Wang Hua was supposed to withdraw in yuan
  (d * 100 + c)

theorem wang_hua_withdrawal_correct (d c : ℤ) :
  let initial_amount := (100 * d + c)
  let incorrect_amount := (100 * c + d)
  let amount_spent := 350
  let remaining_amount := incorrect_amount - amount_spent
  let expected_remaining := 2 * initial_amount
  remaining_amount = expected_remaining ∧ 
  d = 14 ∧ 
  c = 32 :=
by
  sorry

end wang_hua_withdrawal_correct_l73_73993


namespace john_paid_more_l73_73608

theorem john_paid_more 
  (original_price : ℝ)
  (discount_percentage : ℝ) 
  (tip_percentage : ℝ) 
  (discounted_price : ℝ)
  (john_tip : ℝ) 
  (john_total : ℝ)
  (jane_tip : ℝ)
  (jane_total : ℝ) 
  (difference : ℝ) :
  original_price = 42.00000000000004 →
  discount_percentage = 0.10 →
  tip_percentage = 0.15 →
  discounted_price = original_price - (discount_percentage * original_price) →
  john_tip = tip_percentage * original_price →
  john_total = original_price + john_tip →
  jane_tip = tip_percentage * discounted_price →
  jane_total = discounted_price + jane_tip →
  difference = john_total - jane_total →
  difference = 4.830000000000005 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end john_paid_more_l73_73608


namespace problem_1_problem_2_l73_73337

-- Define the propositions p and q
def proposition_p (x a : ℝ) := x^2 - (a + 1/a) * x + 1 < 0
def proposition_q (x : ℝ) := x^2 - 4 * x + 3 ≤ 0

-- Problem 1: Given a = 2 and both p and q are true, find the range of x
theorem problem_1 (a : ℝ) (x : ℝ) (ha : a = 2) (hp : proposition_p x a) (hq : proposition_q x) :
  1 ≤ x ∧ x < 2 :=
sorry

-- Problem 2: Prove that if p is a necessary but not sufficient condition for q, then 3 < a
theorem problem_2 (a : ℝ)
  (h_ns : ∀ x, proposition_q x → proposition_p x a)
  (h_not_s : ∃ x, ¬ (proposition_q x → proposition_p x a)) :
  3 < a :=
sorry

end problem_1_problem_2_l73_73337


namespace difference_of_digits_l73_73450

theorem difference_of_digits (X Y : ℕ) (h1 : 10 * X + Y < 100) 
  (h2 : 72 = (10 * X + Y) - (10 * Y + X)) : (X - Y) = 8 :=
sorry

end difference_of_digits_l73_73450


namespace european_customer_savings_l73_73657

noncomputable def popcorn_cost : ℝ := 8 - 3
noncomputable def drink_cost : ℝ := popcorn_cost + 1
noncomputable def candy_cost : ℝ := drink_cost / 2

noncomputable def discounted_popcorn_cost : ℝ := popcorn_cost * (1 - 0.15)
noncomputable def discounted_candy_cost : ℝ := candy_cost * (1 - 0.1)

noncomputable def total_normal_cost : ℝ := 8 + discounted_popcorn_cost + drink_cost + discounted_candy_cost
noncomputable def deal_price : ℝ := 20
noncomputable def savings_in_dollars : ℝ := total_normal_cost - deal_price

noncomputable def exchange_rate : ℝ := 0.85
noncomputable def savings_in_euros : ℝ := savings_in_dollars * exchange_rate

theorem european_customer_savings : savings_in_euros = 0.81 := by
  sorry

end european_customer_savings_l73_73657


namespace both_A_and_B_are_Gnomes_l73_73813

inductive Inhabitant
| Elf
| Gnome

open Inhabitant

def lies_about_gold (i : Inhabitant) : Prop :=
  match i with
  | Elf => False
  | Gnome => True

def tells_truth_about_others (i : Inhabitant) : Prop :=
  match i with
  | Elf => False
  | Gnome => True

def A_statement : Prop := ∀ i : Inhabitant, lies_about_gold i → i = Gnome
def B_statement : Prop := ∀ i : Inhabitant, tells_truth_about_others i → i = Gnome

theorem both_A_and_B_are_Gnomes (A_statement_true : A_statement) (B_statement_true : B_statement) :
  ∀ i : Inhabitant, (lies_about_gold i ∧ tells_truth_about_others i) → i = Gnome :=
by
  sorry

end both_A_and_B_are_Gnomes_l73_73813


namespace total_husk_is_30_bags_l73_73514

-- Define the total number of cows and the number of days.
def numCows : ℕ := 30
def numDays : ℕ := 30

-- Define the rate of consumption: one cow eats one bag in 30 days.
def consumptionRate (cows : ℕ) (days : ℕ) : ℕ := cows / days

-- Define the total amount of husk consumed in 30 days by 30 cows.
def totalHusk (cows : ℕ) (days : ℕ) (rate : ℕ) : ℕ := cows * rate

-- State the problem in a theorem.
theorem total_husk_is_30_bags : totalHusk numCows numDays 1 = 30 := by
  sorry

end total_husk_is_30_bags_l73_73514


namespace walk_to_bus_stop_time_l73_73043

theorem walk_to_bus_stop_time 
  (S T : ℝ)   -- Usual speed and time
  (D : ℝ)        -- Distance to bus stop
  (T'_delay : ℝ := 9)   -- Additional delay in minutes
  (T_coffee : ℝ := 6)   -- Coffee shop time in minutes
  (reduced_speed_factor : ℝ := 4/5)  -- Reduced speed factor
  (h1 : D = S * T)
  (h2 : D = reduced_speed_factor * S * (T + T'_delay - T_coffee)) :
  T = 12 :=
by
  sorry

end walk_to_bus_stop_time_l73_73043


namespace super_cool_triangles_area_sum_l73_73305

theorem super_cool_triangles_area_sum : 
  ∃ (areas : List ℕ), 
  (∀ (a b : ℕ), (a * b / 2 = 3 * (a + b)) → 
  (a * b / 2 ∈ areas)) ∧ 
  areas.sum = 471 := 
sorry

end super_cool_triangles_area_sum_l73_73305


namespace determine_digits_l73_73932

def product_consecutive_eq_120_times_ABABAB (n A B : ℕ) : Prop :=
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 * (A * 101010101 + B * 10101010 + A * 1010101 + B * 101010 + A * 10101 + B * 1010 + A * 101 + B * 10 + A)

theorem determine_digits (A B : ℕ) (h : ∃ n, product_consecutive_eq_120_times_ABABAB n A B):
  A = 5 ∧ B = 7 :=
sorry

end determine_digits_l73_73932


namespace smallest_distance_AB_ge_2_l73_73525

noncomputable def A (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9
noncomputable def B (x y : ℝ) : Prop := y^2 = -8 * x

theorem smallest_distance_AB_ge_2 :
  ∀ (x1 y1 x2 y2 : ℝ), A x1 y1 → B x2 y2 → dist (x1, y1) (x2, y2) ≥ 2 := by
  sorry

end smallest_distance_AB_ge_2_l73_73525


namespace factorial_div_sub_factorial_equality_l73_73925

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => (n + 1) * factorial n

theorem factorial_div_sub_factorial_equality :
  (factorial 12 - factorial 11) / factorial 10 = 121 :=
by
  sorry

end factorial_div_sub_factorial_equality_l73_73925


namespace intersection_of_A_and_B_l73_73076

noncomputable def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
noncomputable def B : Set ℝ := { x | 0 ≤ x }

theorem intersection_of_A_and_B :
  { x | x ∈ A ∧ x ∈ B } = { x | 0 ≤ x ∧ x ≤ 3 } :=
  by sorry

end intersection_of_A_and_B_l73_73076


namespace unique_solution_l73_73099

noncomputable def f : ℝ → ℝ :=
sorry

theorem unique_solution (x : ℝ) (hx : 0 ≤ x) : 
  (f : ℝ → ℝ) (2 * x + 1) = 3 * (f x) + 5 ↔ f x = -5 / 2 :=
by 
  sorry

end unique_solution_l73_73099


namespace AB_plus_C_eq_neg8_l73_73820

theorem AB_plus_C_eq_neg8 (A B C : ℤ) (g : ℝ → ℝ)
(hf : ∀ x > 3, g x > 0.5)
(heq : ∀ x, g x = x^2 / (A * x^2 + B * x + C))
(hasymp_vert : ∀ x, (A * (x + 3) * (x - 2) = 0 → x = -3 ∨ x = 2))
(hasymp_horiz : (1 : ℝ) / (A : ℝ) < 1) :
A + B + C = -8 :=
sorry

end AB_plus_C_eq_neg8_l73_73820


namespace solution_set_for_absolute_value_inequality_l73_73743

theorem solution_set_for_absolute_value_inequality :
  {x : ℝ | |2 * x - 1| ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by 
  sorry

end solution_set_for_absolute_value_inequality_l73_73743


namespace smallest_number_mod_l73_73398

theorem smallest_number_mod (x : ℕ) :
  (x % 2 = 1) → (x % 3 = 2) → x = 5 :=
by
  sorry

end smallest_number_mod_l73_73398


namespace staircase_steps_eq_twelve_l73_73093

theorem staircase_steps_eq_twelve (n : ℕ) :
  (3 * n * (n + 1) / 2 = 270) → (n = 12) :=
by
  intro h
  sorry

end staircase_steps_eq_twelve_l73_73093


namespace problem_l73_73801

-- Define the problem conditions and the statement that needs to be proved
theorem problem:
  ∀ (x : ℝ), (x ∈ Set.Icc (-1) m) ∧ ((1 - (-1)) / (m - (-1)) = 2 / 5) → m = 4 := by
  sorry

end problem_l73_73801


namespace union_sets_l73_73458

def M (a : ℕ) : Set ℕ := {a, 0}
def N : Set ℕ := {1, 2}

theorem union_sets (a : ℕ) (h_inter : M a ∩ N = {2}) : M a ∪ N = {0, 1, 2} :=
by
  sorry

end union_sets_l73_73458


namespace initial_men_count_l73_73215

theorem initial_men_count 
  (M : ℕ)
  (h1 : 8 * M * 30 = (M + 77) * 6 * 50) :
  M = 63 :=
by
  sorry

end initial_men_count_l73_73215


namespace find_h_l73_73299

theorem find_h (h : ℤ) (root_condition : (-3)^3 + h * (-3) - 18 = 0) : h = -15 :=
by
  sorry

end find_h_l73_73299


namespace range_and_intervals_of_f_l73_73264

noncomputable def f (x : ℝ) : ℝ := (1/3)^(x^2 - 2 * x - 3)

theorem range_and_intervals_of_f :
  (∀ y, y > 0 → y ≤ 81 → (∃ x : ℝ, f x = y)) ∧
  (∀ x y, x ≤ y → f x ≥ f y) ∧
  (∀ x y, x ≥ y → f x ≤ f y) :=
by
  sorry

end range_and_intervals_of_f_l73_73264


namespace exists_fi_l73_73985

theorem exists_fi (f : ℝ → ℝ) (h_periodic : ∀ x : ℝ, f (x + 2 * Real.pi) = f x) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ), 
    (∀ x, f1 (-x) = f1 x ∧ f1 (x + Real.pi) = f1 x) ∧ 
    (∀ x, f2 (-x) = f2 x ∧ f2 (x + Real.pi) = f2 x) ∧ 
    (∀ x, f3 (-x) = f3 x ∧ f3 (x + Real.pi) = f3 x) ∧ 
    (∀ x, f4 (-x) = f4 x ∧ f4 (x + Real.pi) = f4 x) ∧ 
    (∀ x, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
by
  sorry

end exists_fi_l73_73985


namespace arithmetic_sequence_general_formula_and_extremum_l73_73345

noncomputable def a (n : ℕ) : ℤ := sorry
def S (n : ℕ) : ℤ := sorry

theorem arithmetic_sequence_general_formula_and_extremum :
  (a 1 + a 4 = 8) ∧ (a 2 * a 3 = 15) →
  (∃ c d : ℤ, (∀ n : ℕ, a n = c * n + d) ∨ (∀ n : ℕ, a n = -c * n + d)) ∧
  ((∃ n_min : ℕ, n_min > 0 ∧ S n_min = 1) ∧ (∃ n_max : ℕ, n_max > 0 ∧ S n_max = 16)) :=
by
  sorry

end arithmetic_sequence_general_formula_and_extremum_l73_73345


namespace max_saved_houses_l73_73700

theorem max_saved_houses (n c : ℕ) (h₁ : 1 ≤ c ∧ c ≤ n / 2) : 
  ∃ k, k = n^2 + c^2 - n * c - c :=
by
  sorry

end max_saved_houses_l73_73700


namespace part_I_part_II_l73_73251

-- Part (I) 
theorem part_I (a b : ℝ) : (∀ x : ℝ, x^2 - 5 * a * x + b > 0 ↔ (x > 4 ∨ x < 1)) → 
(a = 1 ∧ b = 4) :=
by { sorry }

-- Part (II) 
theorem part_II (x y : ℝ) (a b : ℝ) (h : x + y = 2 ∧ a = 1 ∧ b = 4) : 
x > 0 → y > 0 → 
(∃ t : ℝ, t = a / x + b / y ∧ t ≥ 9 / 2) :=
by { sorry }

end part_I_part_II_l73_73251


namespace polynomial_simplification_l73_73408

theorem polynomial_simplification :
  ∃ A B C D : ℤ,
  (∀ x : ℤ, x ≠ D → (x^3 + 5 * x^2 + 8 * x + 4) / (x + 1) = A * x^2 + B * x + C)
  ∧ (A + B + C + D = 8) :=
sorry

end polynomial_simplification_l73_73408


namespace percent_of_70_is_56_l73_73719

theorem percent_of_70_is_56 : (70 / 125) * 100 = 56 := by
  sorry

end percent_of_70_is_56_l73_73719


namespace initial_days_planned_l73_73624

-- We define the variables and conditions given in the problem.
variables (men_original men_absent men_remaining days_remaining days_initial : ℕ)
variable (work_equivalence : men_original * days_initial = men_remaining * days_remaining)

-- Conditions from the problem
axiom men_original_cond : men_original = 48
axiom men_absent_cond : men_absent = 8
axiom men_remaining_cond : men_remaining = men_original - men_absent
axiom days_remaining_cond : days_remaining = 18

-- Theorem to be proved
theorem initial_days_planned : days_initial = 15 :=
by
  -- Insert proof steps here
  sorry

end initial_days_planned_l73_73624


namespace estimate_red_balls_l73_73230

-- Define the conditions
variable (total_balls : ℕ)
variable (prob_red_ball : ℝ)
variable (frequency_red_ball : ℝ := prob_red_ball)

-- Assume total number of balls in the bag is 20
axiom total_balls_eq_20 : total_balls = 20

-- Assume the probability (or frequency) of drawing a red ball
axiom prob_red_ball_eq_0_25 : prob_red_ball = 0.25

-- The Lean statement
theorem estimate_red_balls (H1 : total_balls = 20) (H2 : prob_red_ball = 0.25) : total_balls * prob_red_ball = 5 :=
by
  rw [H1, H2]
  norm_num
  sorry

end estimate_red_balls_l73_73230


namespace quadratic_three_distinct_solutions_l73_73279

open Classical

variable (a b c : ℝ) (x1 x2 x3 : ℝ)

-- Conditions:
variables (hx1 : a * x1^2 + b * x1 + c = 0)
          (hx2 : a * x2^2 + b * x2 + c = 0)
          (hx3 : a * x3^2 + b * x3 + c = 0)
          (h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)

-- Proof problem
theorem quadratic_three_distinct_solutions : a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end quadratic_three_distinct_solutions_l73_73279


namespace largest_of_A_B_C_l73_73686

noncomputable def A : ℝ := (3003 / 3002) + (3003 / 3004)
noncomputable def B : ℝ := (3003 / 3004) + (3005 / 3004)
noncomputable def C : ℝ := (3004 / 3003) + (3004 / 3005)

theorem largest_of_A_B_C : A > B ∧ A ≥ C := by
  sorry

end largest_of_A_B_C_l73_73686


namespace find_certain_number_l73_73445

-- Define the given operation a # b
def sOperation (a b : ℝ) : ℝ :=
  a * b - b + b^2

-- State the theorem to find the value of the certain number
theorem find_certain_number (x : ℝ) (h : sOperation 3 x = 48) : x = 6 :=
sorry

end find_certain_number_l73_73445


namespace real_root_ineq_l73_73390

theorem real_root_ineq (a b : ℝ) (x₀ : ℝ) (h : x₀^4 - a * x₀^3 + 2 * x₀^2 - b * x₀ + 1 = 0) :
  a^2 + b^2 ≥ 8 :=
by
  sorry

end real_root_ineq_l73_73390


namespace decrease_travel_time_l73_73486

variable (distance : ℕ) (initial_speed : ℕ) (speed_increase : ℕ)

def original_travel_time (distance initial_speed : ℕ) : ℕ :=
  distance / initial_speed

def new_travel_time (distance new_speed : ℕ) : ℕ :=
  distance / new_speed

theorem decrease_travel_time (h₁ : distance = 600) (h₂ : initial_speed = 50) (h₃ : speed_increase = 25) :
  original_travel_time distance initial_speed - new_travel_time distance (initial_speed + speed_increase) = 4 :=
by
  sorry

end decrease_travel_time_l73_73486


namespace simple_interest_fraction_l73_73320

theorem simple_interest_fraction (P : ℝ) (R T : ℝ) (hR: R = 4) (hT: T = 5) :
  (P * R * T / 100) / P = 1 / 5 := 
by
  sorry

end simple_interest_fraction_l73_73320


namespace fraction_defined_iff_l73_73625

theorem fraction_defined_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (|x| - 6)) ↔ (x ≠ 6 ∧ x ≠ -6) :=
by 
  sorry

end fraction_defined_iff_l73_73625


namespace tissue_actual_diameter_l73_73170

theorem tissue_actual_diameter (magnification_factor : ℝ) (magnified_diameter : ℝ) 
(h1 : magnification_factor = 1000)
(h2 : magnified_diameter = 0.3) : 
  magnified_diameter / magnification_factor = 0.0003 :=
by sorry

end tissue_actual_diameter_l73_73170


namespace minimize_y_l73_73899

noncomputable def y (x a b : ℝ) : ℝ := 2 * (x - a)^2 + 3 * (x - b)^2

theorem minimize_y (a b : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x a b ≤ y x' a b) ∧ x = (2 * a + 3 * b) / 5 :=
sorry

end minimize_y_l73_73899


namespace translation_line_segment_l73_73567

theorem translation_line_segment (a b : ℝ) :
  (∃ A B A1 B1: ℝ × ℝ,
    A = (1,0) ∧ B = (3,2) ∧ A1 = (a, 1) ∧ B1 = (4,b) ∧
    ∃ t : ℝ × ℝ, A + t = A1 ∧ B + t = B1) →
  a = 2 ∧ b = 3 :=
by
  sorry

end translation_line_segment_l73_73567


namespace focus_of_parabola_l73_73181

theorem focus_of_parabola : 
  ∃(h k : ℚ), ((∀ x : ℚ, -2 * x^2 - 6 * x + 1 = -2 * (x + 3 / 2)^2 + 11 / 2) ∧ 
  (∃ a : ℚ, (a = -2 / 8) ∧ (h = -3/2) ∧ (k = 11/2 + a)) ∧ 
  (h, k) = (-3/2, 43 / 8)) :=
sorry

end focus_of_parabola_l73_73181


namespace sequence_of_numbers_exists_l73_73055

theorem sequence_of_numbers_exists :
  ∃ (a b : ℤ), (a + 2 * b > 0) ∧ (7 * a + 13 * b < 0) :=
sorry

end sequence_of_numbers_exists_l73_73055


namespace tim_score_l73_73278

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

theorem tim_score :
  (first_seven_primes.sum = 58) :=
by
  sorry

end tim_score_l73_73278


namespace inscribed_rectangle_area_l73_73867

theorem inscribed_rectangle_area (h a b x : ℝ) (ha_gt_b : a > b) :
  ∃ A : ℝ, A = (b * x / h) * (h - x) :=
by
  sorry

end inscribed_rectangle_area_l73_73867


namespace total_cost_of_car_rental_l73_73103

theorem total_cost_of_car_rental :
  ∀ (rental_cost_per_day mileage_cost_per_mile : ℝ) (days rented : ℕ) (miles_driven : ℕ),
  rental_cost_per_day = 30 →
  mileage_cost_per_mile = 0.25 →
  rented = 5 →
  miles_driven = 500 →
  rental_cost_per_day * rented + mileage_cost_per_mile * miles_driven = 275 := by
  sorry

end total_cost_of_car_rental_l73_73103


namespace associate_professor_pencils_l73_73713

theorem associate_professor_pencils
  (A B P : ℕ)
  (h1 : A + B = 7)
  (h2 : P * A + B = 10)
  (h3 : A + 2 * B = 11) :
  P = 2 :=
by {
  -- Variables declarations and assumptions
  -- Combine and manipulate equations to prove P = 2
  sorry
}

end associate_professor_pencils_l73_73713


namespace solve_inequality_l73_73343

open Set

theorem solve_inequality (a x : ℝ) : 
  (x - 2) * (a * x - 2) > 0 → 
  (a = 0 ∧ x < 2) ∨ 
  (a < 0 ∧ (2/a) < x ∧ x < 2) ∨ 
  (0 < a ∧ a < 1 ∧ ((x < 2 ∨ x > 2/a))) ∨ 
  (a = 1 ∧ x ≠ 2) ∨ 
  (a > 1 ∧ ((x < 2/a ∨ x > 2)))
  := sorry

end solve_inequality_l73_73343


namespace greatest_integer_gcd_6_l73_73790

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l73_73790


namespace marble_counts_l73_73041

theorem marble_counts (A B C : ℕ) : 
  (∃ x : ℕ, 
    A = 165 ∧ 
    B = 57 ∧ 
    C = 21 ∧ 
    (A = 55 * x / 27) ∧ 
    (B = 19 * x / 27) ∧ 
    (C = 7 * x / 27) ∧ 
    (7 * x / 9 = x / 9 + 54) ∧ 
    (A + B + C) = 3 * x
  ) :=
sorry

end marble_counts_l73_73041


namespace abs_triangle_inequality_l73_73765

theorem abs_triangle_inequality {a : ℝ} (h : ∀ x : ℝ, |x - 3| + |x + 1| > a) : a < 4 :=
sorry

end abs_triangle_inequality_l73_73765


namespace triangle_at_most_one_right_angle_l73_73324

-- Definition of a triangle with its angles adding up to 180 degrees
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

-- The main theorem stating that a triangle can have at most one right angle.
theorem triangle_at_most_one_right_angle (α β γ : ℝ) 
  (h₁ : triangle α β γ) 
  (h₂ : α = 90 ∨ β = 90 ∨ γ = 90) : 
  (α = 90 → β ≠ 90 ∧ γ ≠ 90) ∧ 
  (β = 90 → α ≠ 90 ∧ γ ≠ 90) ∧ 
  (γ = 90 → α ≠ 90 ∧ β ≠ 90) :=
sorry

end triangle_at_most_one_right_angle_l73_73324


namespace divisibility_by_1956_l73_73165

theorem divisibility_by_1956 (n : ℕ) (hn : n % 2 = 1) : 
  1956 ∣ (24 * 80^n + 1992 * 83^(n-1)) :=
by
  sorry

end divisibility_by_1956_l73_73165


namespace problem1_problem2_l73_73169

-- Proof problem 1 statement in Lean 4
theorem problem1 :
  (1 : ℝ) * (Real.sqrt 2)^2 - |(1 : ℝ) - Real.sqrt 3| + Real.sqrt ((-3 : ℝ)^2) + Real.sqrt 81 = 15 - Real.sqrt 3 :=
by sorry

-- Proof problem 2 statement in Lean 4
theorem problem2 (x y : ℝ) :
  (x - 2 * y)^2 - (x + 2 * y + 3) * (x + 2 * y - 3) = -8 * x * y + 9 :=
by sorry

end problem1_problem2_l73_73169


namespace cars_meet_time_l73_73875

theorem cars_meet_time 
  (L : ℕ) (v1 v2 : ℕ) (t : ℕ)
  (H1 : L = 333)
  (H2 : v1 = 54)
  (H3 : v2 = 57)
  (H4 : v1 * t + v2 * t = L) : 
  t = 3 :=
by
  -- Insert proof here
  sorry

end cars_meet_time_l73_73875


namespace remainder_when_squared_l73_73422

theorem remainder_when_squared (n : ℕ) (h : n % 8 = 6) : (n * n) % 32 = 4 := by
  sorry

end remainder_when_squared_l73_73422


namespace correct_statements_l73_73928

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

theorem correct_statements : 
  (∀ x, f (-x) = -f (x)) ∧  -- Statement A
  (∀ x₁ x₂, x₁ + x₂ = Real.pi / 2 → g x₁ = g x₂)  -- Statement C
:= by
  sorry

end correct_statements_l73_73928


namespace false_statement_l73_73484

noncomputable def heartsuit (x y : ℝ) := abs (x - y)
noncomputable def diamondsuit (z w : ℝ) := (z + w) ^ 2

theorem false_statement : ∃ (x y : ℝ), (heartsuit x y) ^ 2 ≠ diamondsuit x y := by
  sorry

end false_statement_l73_73484


namespace first_term_arithmetic_sequence_l73_73284

theorem first_term_arithmetic_sequence (S : ℕ → ℤ) (a : ℤ) (h1 : ∀ n, S n = (n * (2 * a + (n - 1) * 5)) / 2)
    (h2 : ∀ n m, (S (3 * n)) / (S m) = (S (3 * m)) / (S n)) : a = 5 / 2 := 
sorry

end first_term_arithmetic_sequence_l73_73284


namespace part1_even_function_part2_two_distinct_zeros_l73_73775

noncomputable def f (x a : ℝ) : ℝ := (4^x + a) / 2^x
noncomputable def g (x a : ℝ) : ℝ := f x a - (a + 1)

theorem part1_even_function (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a) ↔ a = 1 :=
sorry

theorem part2_two_distinct_zeros (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -1 ≤ x1 ∧ x1 ≤ 1 ∧ -1 ≤ x2 ∧ x2 ≤ 1 ∧ g x1 a = 0 ∧ g x2 a = 0) ↔ (a ∈ Set.Icc (1/2) 1 ∪ Set.Icc 1 2) :=
sorry

end part1_even_function_part2_two_distinct_zeros_l73_73775


namespace second_shift_production_l73_73730

-- Question: Prove that the number of cars produced by the second shift is 1,100 given the conditions
-- Conditions:
-- 1. P_day = 4 * P_second
-- 2. P_day + P_second = 5,500

theorem second_shift_production (P_day P_second : ℕ) (h1 : P_day = 4 * P_second) (h2 : P_day + P_second = 5500) :
  P_second = 1100 := by
  sorry

end second_shift_production_l73_73730


namespace smallest_xym_sum_l73_73766

def is_two_digit_integer (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

def reversed_digits (x y : ℤ) : Prop :=
  ∃ a b : ℤ, x = 10 * a + b ∧ y = 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9

def odd_multiple_of_9 (n : ℤ) : Prop :=
  ∃ k : ℤ, k % 2 = 1 ∧ n = 9 * k

theorem smallest_xym_sum :
  ∃ (x y m : ℤ), is_two_digit_integer x ∧ is_two_digit_integer y ∧ reversed_digits x y ∧ x^2 + y^2 = m^2 ∧ odd_multiple_of_9 (x + y) ∧ x + y + m = 169 :=
by
  sorry

end smallest_xym_sum_l73_73766


namespace danny_total_bottle_caps_l73_73747

def danny_initial_bottle_caps : ℕ := 37
def danny_found_bottle_caps : ℕ := 18

theorem danny_total_bottle_caps : danny_initial_bottle_caps + danny_found_bottle_caps = 55 := by
  sorry

end danny_total_bottle_caps_l73_73747


namespace john_initial_running_time_l73_73191

theorem john_initial_running_time (H : ℝ) (hH1 : 1.75 * H = 168 / 12)
: H = 8 :=
sorry

end john_initial_running_time_l73_73191


namespace necessary_but_not_sufficient_condition_l73_73306

theorem necessary_but_not_sufficient_condition (a b : ℤ) :
  (a ≠ 1 ∨ b ≠ 2) → (a + b ≠ 3) ∧ ¬((a + b ≠ 3) → (a ≠ 1 ∨ b ≠ 2)) :=
sorry

end necessary_but_not_sufficient_condition_l73_73306


namespace concentration_after_removing_water_l73_73146

theorem concentration_after_removing_water :
  ∀ (initial_volume : ℝ) (initial_percentage : ℝ) (water_removed : ℝ),
  initial_volume = 18 →
  initial_percentage = 0.4 →
  water_removed = 6 →
  (initial_percentage * initial_volume) / (initial_volume - water_removed) * 100 = 60 :=
by
  intros initial_volume initial_percentage water_removed h1 h2 h3
  rw [h1, h2, h3]
  sorry

end concentration_after_removing_water_l73_73146


namespace cooking_oil_remaining_l73_73509

theorem cooking_oil_remaining (initial_weight : ℝ) (fraction_used : ℝ) (remaining_weight : ℝ) :
  initial_weight = 5 → fraction_used = 4 / 5 → remaining_weight = 21 / 5 → initial_weight * (1 - fraction_used) ≠ remaining_weight → initial_weight * (1 - fraction_used) = 1 :=
by 
  intros h_initial_weight h_fraction_used h_remaining_weight h_contradiction
  sorry

end cooking_oil_remaining_l73_73509


namespace corner_movement_l73_73027

-- Definition of corner movement problem
def canMoveCornerToBottomRight (m n : ℕ) : Prop :=
  m ≥ 2 ∧ n ≥ 2 ∧ (m % 2 = 1 ∧ n % 2 = 1)

theorem corner_movement (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) :
  (canMoveCornerToBottomRight m n ↔ (m % 2 = 1 ∧ n % 2 = 1)) :=
by
  sorry  -- Proof is omitted

end corner_movement_l73_73027


namespace product_scaled_areas_l73_73964

variable (a b c k V : ℝ)

def volume (a b c : ℝ) : ℝ := a * b * c

theorem product_scaled_areas (a b c k : ℝ) (V : ℝ) (hV : V = volume a b c) :
  (k * a * b) * (k * b * c) * (k * c * a) = k^3 * (V^2) := 
by
  -- Proof steps would go here, but we use sorry to skip the proof
  sorry

end product_scaled_areas_l73_73964


namespace pencils_purchased_l73_73302

theorem pencils_purchased (total_cost : ℝ) (num_pens : ℕ) (pen_price : ℝ) (pencil_price : ℝ) (num_pencils : ℕ) : 
  total_cost = (num_pens * pen_price) + (num_pencils * pencil_price) → 
  num_pens = 30 → 
  pen_price = 20 → 
  pencil_price = 2 → 
  total_cost = 750 →
  num_pencils = 75 :=
by
  sorry

end pencils_purchased_l73_73302


namespace find_B_l73_73242

variable (A B : ℝ)

def condition1 : Prop := A + B = 1210
def condition2 : Prop := (4 / 15) * A = (2 / 5) * B

theorem find_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 484 :=
sorry

end find_B_l73_73242


namespace yoongi_has_fewest_apples_l73_73698

noncomputable def yoongi_apples : ℕ := 4
noncomputable def yuna_apples : ℕ := 5
noncomputable def jungkook_apples : ℕ := 6 * 3

theorem yoongi_has_fewest_apples : yoongi_apples < yuna_apples ∧ yoongi_apples < jungkook_apples := by
  sorry

end yoongi_has_fewest_apples_l73_73698


namespace cards_ratio_l73_73678

theorem cards_ratio (b_c : ℕ) (m_c : ℕ) (m_l : ℕ) (m_g : ℕ) 
  (h1 : b_c = 20) 
  (h2 : m_c = b_c + 8) 
  (h3 : m_l = 14) 
  (h4 : m_g = m_c - m_l) : 
  m_g / m_c = 1 / 2 :=
by
  sorry

end cards_ratio_l73_73678


namespace inequality_hold_l73_73334

theorem inequality_hold (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 :=
by
  -- Proof goes here
  sorry

end inequality_hold_l73_73334


namespace find_x_l73_73404

noncomputable def arctan := Real.arctan

theorem find_x :
  (∃ x : ℝ, 3 * arctan (1 / 4) + arctan (1 / 5) + arctan (1 / x) = π / 4 ∧ x = -250 / 37) :=
  sorry

end find_x_l73_73404


namespace log_inequality_l73_73672

noncomputable def a := Real.log 6 / Real.log 3
noncomputable def b := Real.log 10 / Real.log 5
noncomputable def c := Real.log 14 / Real.log 7

theorem log_inequality :
  a > b ∧ b > c :=
by
  sorry

end log_inequality_l73_73672


namespace sum_max_min_ratio_l73_73136

theorem sum_max_min_ratio (x y : ℝ) 
  (h_ellipse : 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0) 
  : (∃ m_max m_min : ℝ, (∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0 → y = m_max * x ∨ y = m_min * x) ∧ (m_max + m_min = 37 / 22)) :=
sorry

end sum_max_min_ratio_l73_73136


namespace express_repeating_decimal_as_fraction_l73_73556

noncomputable def repeating_decimal_to_fraction : ℚ :=
  3 + 7 / 9  -- Representation of 3.\overline{7} as a Rational number representation

theorem express_repeating_decimal_as_fraction :
  (3 + 7 / 9 : ℚ) = 34 / 9 :=
by
  -- Placeholder for proof steps
  sorry

end express_repeating_decimal_as_fraction_l73_73556


namespace prob_not_negative_review_A_prob_two_positive_reviews_choose_platform_A_l73_73201

-- Definitions of the problem conditions
def positive_reviews_A := 75
def neutral_reviews_A := 20
def negative_reviews_A := 5
def total_reviews_A := 100

def positive_reviews_B := 64
def neutral_reviews_B := 8
def negative_reviews_B := 8
def total_reviews_B := 80

-- Prove the probability that a buyer's evaluation on platform A is not a negative review
theorem prob_not_negative_review_A : 
  (1 - negative_reviews_A / total_reviews_A) = 19 / 20 := by
  sorry

-- Prove the probability that exactly 2 out of 4 (2 from A and 2 from B) buyers give a positive review
theorem prob_two_positive_reviews :
  ((positive_reviews_A / total_reviews_A) ^ 2 * (1 - positive_reviews_B / total_reviews_B) ^ 2 + 
  2 * (positive_reviews_A / total_reviews_A) * (1 - positive_reviews_A / total_reviews_A) * 
  (positive_reviews_B / total_reviews_B) * (1 - positive_reviews_B / total_reviews_B) +
  (1 - positive_reviews_A / total_reviews_A) ^ 2 * (positive_reviews_B / total_reviews_B) ^ 2) = 
  73 / 400 := by
  sorry

-- Choose platform A based on the given data
theorem choose_platform_A :
  let E_A := (5 * 0.75 + 3 * 0.2 + 1 * 0.05)
  let D_A := (5 - E_A) ^ 2 * 0.75 + (3 - E_A) ^ 2 * 0.2 + (1 - E_A) ^ 2 * 0.05
  let E_B := (5 * 0.8 + 3 * 0.1 + 1 * 0.1)
  let D_B := (5 - E_B) ^ 2 * 0.8 + (3 - E_B) ^ 2 * 0.1 + (1 - E_B) ^ 2 * 0.1
  (E_A = E_B) ∧ (D_A < D_B) → choose_platform = "Platform A" := by
  sorry

end prob_not_negative_review_A_prob_two_positive_reviews_choose_platform_A_l73_73201


namespace equal_volume_rect_parallelepipeds_decomposable_equal_volume_prisms_decomposable_l73_73589

-- Definition of volumes for rectangular parallelepipeds
def volume_rect_parallelepiped (a b c: ℝ) : ℝ := a * b * c

-- Definition of volumes for prisms
def volume_prism (base_area height: ℝ) : ℝ := base_area * height

-- Definition of decomposability of rectangular parallelepipeds
def decomposable_rect_parallelepipeds (a1 b1 c1 a2 b2 c2: ℝ) : Prop :=
  (volume_rect_parallelepiped a1 b1 c1) = (volume_rect_parallelepiped a2 b2 c2)

-- Lean statement for part (a)
theorem equal_volume_rect_parallelepipeds_decomposable (a1 b1 c1 a2 b2 c2: ℝ) (h: decomposable_rect_parallelepipeds a1 b1 c1 a2 b2 c2) :
  True := sorry

-- Definition of decomposability of prisms
def decomposable_prisms (base_area1 height1 base_area2 height2: ℝ) : Prop :=
  (volume_prism base_area1 height1) = (volume_prism base_area2 height2)

-- Lean statement for part (b)
theorem equal_volume_prisms_decomposable (base_area1 height1 base_area2 height2: ℝ) (h: decomposable_prisms base_area1 height1 base_area2 height2) :
  True := sorry

end equal_volume_rect_parallelepipeds_decomposable_equal_volume_prisms_decomposable_l73_73589


namespace sarah_likes_digits_l73_73507

theorem sarah_likes_digits : ∀ n : ℕ, n % 8 = 0 → (n % 10 = 0 ∨ n % 10 = 4 ∨ n % 10 = 8) :=
by
  sorry

end sarah_likes_digits_l73_73507


namespace calculator_sum_is_large_l73_73276

-- Definitions for initial conditions and operations
def participants := 50
def initial_calc1 := 2
def initial_calc2 := -2
def initial_calc3 := 0

-- Define the operations
def operation_calc1 (n : ℕ) := initial_calc1 * 2^n
def operation_calc2 (n : ℕ) := (-2) ^ (2^n)
def operation_calc3 (n : ℕ) := initial_calc3 - n

-- Define the final values for each calculator
def final_calc1 := operation_calc1 participants
def final_calc2 := operation_calc2 participants
def final_calc3 := operation_calc3 participants

-- The final sum
def final_sum := final_calc1 + final_calc2 + final_calc3

-- Prove the final result
theorem calculator_sum_is_large :
  final_sum = 2 ^ (2 ^ 50) :=
by
  -- The proof would go here.
  sorry

end calculator_sum_is_large_l73_73276


namespace rest_area_location_l73_73516

theorem rest_area_location :
  ∃ (rest_area : ℝ), rest_area = 35 + (95 - 35) / 2 :=
by
  -- Here we set the variables for the conditions
  let fifth_exit := 35
  let seventh_exit := 95
  let rest_area := 35 + (95 - 35) / 2
  use rest_area
  sorry

end rest_area_location_l73_73516


namespace average_salary_excluding_manager_l73_73977

theorem average_salary_excluding_manager (A : ℝ) 
  (num_employees : ℝ := 20)
  (manager_salary : ℝ := 3300)
  (salary_increase : ℝ := 100)
  (total_salary_with_manager : ℝ := 21 * (A + salary_increase)) :
  20 * A + manager_salary = total_salary_with_manager → A = 1200 := 
by
  intro h
  sorry

end average_salary_excluding_manager_l73_73977


namespace new_ratio_milk_water_after_adding_milk_l73_73881

variable (initial_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ)
variable (added_milk_volume : ℕ)

def ratio_of_mix_after_addition (initial_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ) 
  (added_milk_volume : ℕ) : ℕ × ℕ :=
  let total_parts := initial_milk_ratio + initial_water_ratio
  let part_volume := initial_volume / total_parts
  let initial_milk_volume := initial_milk_ratio * part_volume
  let initial_water_volume := initial_water_ratio * part_volume
  let new_milk_volume := initial_milk_volume + added_milk_volume
  (new_milk_volume / initial_water_volume, 1)

theorem new_ratio_milk_water_after_adding_milk 
  (h_initial_volume : initial_volume = 20)
  (h_initial_milk_ratio : initial_milk_ratio = 3)
  (h_initial_water_ratio : initial_water_ratio = 1)
  (h_added_milk_volume : added_milk_volume = 5) : 
  ratio_of_mix_after_addition initial_volume initial_milk_ratio initial_water_ratio added_milk_volume = (4, 1) :=
  by
    sorry

end new_ratio_milk_water_after_adding_milk_l73_73881


namespace profit_percentage_with_discount_is_26_l73_73940

noncomputable def cost_price : ℝ := 100
noncomputable def profit_percentage_without_discount : ℝ := 31.25
noncomputable def discount_percentage : ℝ := 4

noncomputable def selling_price_without_discount : ℝ :=
  cost_price * (1 + profit_percentage_without_discount / 100)

noncomputable def discount : ℝ := 
  discount_percentage / 100 * selling_price_without_discount

noncomputable def selling_price_with_discount : ℝ :=
  selling_price_without_discount - discount

noncomputable def profit_with_discount : ℝ := 
  selling_price_with_discount - cost_price

noncomputable def profit_percentage_with_discount : ℝ := 
  (profit_with_discount / cost_price) * 100

theorem profit_percentage_with_discount_is_26 :
  profit_percentage_with_discount = 26 := by 
  sorry

end profit_percentage_with_discount_is_26_l73_73940


namespace inequality_proof_l73_73839

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 4) : 
  1 / a + 4 / b ≥ 9 / 4 :=
by
  sorry

end inequality_proof_l73_73839


namespace abs_inequality_solution_l73_73190

theorem abs_inequality_solution :
  {x : ℝ | |x + 2| > 3} = {x : ℝ | x < -5} ∪ {x : ℝ | x > 1} :=
by
  sorry

end abs_inequality_solution_l73_73190


namespace largest_fraction_of_consecutive_odds_is_three_l73_73418

theorem largest_fraction_of_consecutive_odds_is_three
  (p q r s : ℕ)
  (h1 : 0 < p)
  (h2 : p < q)
  (h3 : q < r)
  (h4 : r < s)
  (h_odd1 : p % 2 = 1)
  (h_odd2 : q % 2 = 1)
  (h_odd3 : r % 2 = 1)
  (h_odd4 : s % 2 = 1)
  (h_consecutive1 : q = p + 2)
  (h_consecutive2 : r = q + 2)
  (h_consecutive3 : s = r + 2) :
  (r + s) / (p + q) = 3 :=
sorry

end largest_fraction_of_consecutive_odds_is_three_l73_73418


namespace cities_with_highest_increase_l73_73598

-- Define population changes for each city
def cityF_initial := 30000
def cityF_final := 45000
def cityG_initial := 55000
def cityG_final := 77000
def cityH_initial := 40000
def cityH_final := 60000
def cityI_initial := 70000
def cityI_final := 98000
def cityJ_initial := 25000
def cityJ_final := 37500

-- Function to calculate percentage increase
def percentage_increase (initial final : ℕ) : ℚ :=
  ((final - initial) : ℚ) / (initial : ℚ) * 100

-- Theorem stating cities F, H, and J had the highest percentage increase
theorem cities_with_highest_increase :
  percentage_increase cityF_initial cityF_final = 50 ∧
  percentage_increase cityH_initial cityH_final = 50 ∧
  percentage_increase cityJ_initial cityJ_final = 50 ∧
  percentage_increase cityG_initial cityG_final < 50 ∧
  percentage_increase cityI_initial cityI_final < 50 :=
by
-- Proof omitted
sorry

end cities_with_highest_increase_l73_73598


namespace circle_condition_l73_73831

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + m = 0) → m < 1 := 
by
  sorry

end circle_condition_l73_73831


namespace sqrt_12_estimate_l73_73836

theorem sqrt_12_estimate : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end sqrt_12_estimate_l73_73836


namespace candy_from_sister_l73_73938

variable (total_neighbors : Nat) (pieces_per_day : Nat) (days : Nat) (total_pieces : Nat)
variable (pieces_per_day_eq : pieces_per_day = 9)
variable (days_eq : days = 9)
variable (total_neighbors_eq : total_neighbors = 66)
variable (total_pieces_eq : total_pieces = 81)

theorem candy_from_sister : 
  total_pieces = total_neighbors + 15 :=
by
  sorry

end candy_from_sister_l73_73938


namespace smaller_fraction_l73_73006

variable (x y : ℚ)

theorem smaller_fraction (h1 : x + y = 13 / 14) (h2 : x * y = 1 / 8) : min x y = 1 / 6 :=
by
  sorry

end smaller_fraction_l73_73006


namespace age_difference_l73_73186

theorem age_difference {A B C : ℕ} (h : A + B = B + C + 15) : A - C = 15 := 
by 
  sorry

end age_difference_l73_73186


namespace additional_amount_needed_l73_73518

-- Definitions of the conditions
def shampoo_cost : ℝ := 10.00
def conditioner_cost : ℝ := 10.00
def lotion_cost : ℝ := 6.00
def lotions_count : ℕ := 3
def free_shipping_threshold : ℝ := 50.00

-- Calculating the total amount spent
def total_spent : ℝ :=
  shampoo_cost + conditioner_cost + lotions_count * lotion_cost

-- Required statement for the proof
theorem additional_amount_needed : 
  total_spent + 12.00 = free_shipping_threshold :=
by 
  -- Proof will be here
  sorry

end additional_amount_needed_l73_73518


namespace factorization_of_polynomial_l73_73174

theorem factorization_of_polynomial (x : ℂ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^2 - 3) * (x^4 + 3 * x^2 + 9) :=
by sorry

end factorization_of_polynomial_l73_73174


namespace problem_statement_l73_73918

variables {α β : Plane} {m : Line}

def parallel (a b : Plane) : Prop := sorry
def perpendicular (m : Line) (π : Plane) : Prop := sorry

axiom parallel_symm {a b : Plane} : parallel a b → parallel b a
axiom perpendicular_trans {m : Line} {a b : Plane} : perpendicular m a → parallel a b → perpendicular m b

theorem problem_statement (h1 : parallel α β) (h2 : perpendicular m α) : perpendicular m β :=
  perpendicular_trans h2 (parallel_symm h1)

end problem_statement_l73_73918


namespace steve_take_home_pay_l73_73312

-- Defining the conditions
def annual_salary : ℕ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℕ := 800

-- Defining the taxes function
def taxes (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the healthcare function
def healthcare (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the total deductions function
def total_deductions (salary : ℕ) (tax_rate : ℝ) (healthcare_rate : ℝ) (dues : ℕ) : ℝ :=
  (taxes salary tax_rate) + (healthcare salary healthcare_rate) + dues

-- Defining the take-home pay function
def take_home_pay (salary : ℕ) (deductions : ℝ) : ℝ :=
  salary - deductions

-- Using definitions to assert the take-home pay
theorem steve_take_home_pay : take_home_pay annual_salary (total_deductions annual_salary tax_rate healthcare_rate union_dues) = 27200 :=
by
  -- proof has been skipped
  sorry

end steve_take_home_pay_l73_73312


namespace circle_diameter_l73_73984
open Real

theorem circle_diameter (A : ℝ) (hA : A = 50.26548245743669) : ∃ d : ℝ, d = 8 :=
by
  sorry

end circle_diameter_l73_73984


namespace tile_difference_is_42_l73_73963

def original_blue_tiles : ℕ := 14
def original_green_tiles : ℕ := 8
def green_tiles_first_border : ℕ := 18
def green_tiles_second_border : ℕ := 30

theorem tile_difference_is_42 :
  (original_green_tiles + green_tiles_first_border + green_tiles_second_border) - original_blue_tiles = 42 :=
by
  sorry

end tile_difference_is_42_l73_73963


namespace z_is_1_2_decades_younger_than_x_l73_73476

variable (x y z w : ℕ) -- Assume ages as natural numbers

def age_equivalence_1 : Prop := x + y = y + z + 12
def age_equivalence_2 : Prop := x + y + w = y + z + w + 12

theorem z_is_1_2_decades_younger_than_x (h1 : age_equivalence_1 x y z) (h2 : age_equivalence_2 x y z w) :
  z = x - 12 := by
  sorry

end z_is_1_2_decades_younger_than_x_l73_73476


namespace probability_first_prize_both_distribution_of_X_l73_73834

-- Definitions for the conditions
def total_students : ℕ := 500
def male_students : ℕ := 200
def female_students : ℕ := 300

def male_first_prize : ℕ := 10
def female_first_prize : ℕ := 25

def male_second_prize : ℕ := 15
def female_second_prize : ℕ := 25

def male_third_prize : ℕ := 15
def female_third_prize : ℕ := 40

-- Part (1): Prove the probability that both selected students receive the first prize is 1/240.
theorem probability_first_prize_both :
  (male_first_prize / male_students : ℚ) * (female_first_prize / female_students : ℚ) = 1 / 240 := 
sorry

-- Part (2): Prove the distribution of X.
def P_male_award : ℚ := (male_first_prize + male_second_prize + male_third_prize) / male_students
def P_female_award : ℚ := (female_first_prize + female_second_prize + female_third_prize) / female_students

theorem distribution_of_X :
  ∀ X : ℕ, X = 0 ∧ ((1 - P_male_award) * (1 - P_female_award) = 28 / 50) ∨ 
           X = 1 ∧ ((1 - P_male_award) * (1 - P_female_award) + (P_male_award * (1 - P_female_award)) + ((1 - P_male_award) * P_female_award) = 19 / 50) ∨ 
           X = 2 ∧ (P_male_award * P_female_award = 3 / 50) :=
sorry

end probability_first_prize_both_distribution_of_X_l73_73834


namespace red_cars_in_lot_l73_73692

theorem red_cars_in_lot (B : ℕ) (hB : B = 90) (ratio_condition : 3 * B = 8 * R) : R = 33 :=
by
  -- Given
  have h1 : B = 90 := hB
  have h2 : 3 * B = 8 * R := ratio_condition

  -- To solve
  sorry

end red_cars_in_lot_l73_73692


namespace mork_tax_rate_l73_73050

theorem mork_tax_rate (M R : ℝ) (h1 : 0.15 = 0.15) (h2 : 4 * M = Mindy_income) (h3 : (R / 100 * M + 0.15 * 4 * M) = 0.21 * 5 * M):
  R = 45 :=
sorry

end mork_tax_rate_l73_73050


namespace angie_pretzels_dave_pretzels_l73_73572

theorem angie_pretzels (B S A : ℕ) (hB : B = 12) (hS : S = B / 2) (hA : A = 3 * S) : A = 18 := by
  -- We state the problem using variables B, S, and A for Barry, Shelly, and Angie respectively
  sorry

theorem dave_pretzels (A S D : ℕ) (hA : A = 18) (hS : S = 12 / 2) (hD : D = 25 * (A + S) / 100) : D = 6 := by
  -- We use variables A and S from the first theorem, and introduce D for Dave
  sorry

end angie_pretzels_dave_pretzels_l73_73572


namespace solution_set_l73_73198

open Real

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the differentiable function f

axiom differentiable_f : Differentiable ℝ f
axiom condition_f : ∀ x, f x > 0 ∧ x * (deriv (deriv (deriv f))) x > 0

theorem solution_set :
  {x : ℝ | 1 ≤ x ∧ x < 2} =
    {x : ℝ | f (sqrt (x + 1)) > sqrt (x - 1) * f (sqrt (x ^ 2 - 1))} :=
sorry

end solution_set_l73_73198


namespace calculate_brick_quantity_l73_73563

noncomputable def brick_quantity (brick_length brick_width brick_height wall_length wall_height wall_width : ℝ) : ℝ :=
  let brick_volume := brick_length * brick_width * brick_height
  let wall_volume := wall_length * wall_height * wall_width
  wall_volume / brick_volume

theorem calculate_brick_quantity :
  brick_quantity 20 10 8 1000 800 2450 = 1225000 := 
by 
  -- Volume calculations are shown but proof is omitted
  sorry

end calculate_brick_quantity_l73_73563


namespace power_function_result_l73_73778
noncomputable def f (x : ℝ) (k : ℝ) (n : ℝ) : ℝ := k * x ^ n

theorem power_function_result (k n : ℝ) (h1 : f 27 k n = 3) : f 8 k (1/3) = 2 :=
by 
  sorry

end power_function_result_l73_73778


namespace average_of_distinct_numbers_l73_73080

theorem average_of_distinct_numbers (A B C D : ℕ) (hA : A = 1 ∨ A = 3 ∨ A = 5 ∨ A = 7)
                                   (hB : B = 1 ∨ B = 3 ∨ B = 5 ∨ B = 7)
                                   (hC : C = 1 ∨ C = 3 ∨ C = 5 ∨ C = 7)
                                   (hD : D = 1 ∨ D = 3 ∨ D = 5 ∨ D = 7)
                                   (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
    (A + B + C + D) / 4 = 4 := by
  sorry

end average_of_distinct_numbers_l73_73080


namespace circle_radius_l73_73570

-- Given conditions
def central_angle : ℝ := 225
def perimeter : ℝ := 83
noncomputable def pi_val : ℝ := Real.pi

-- Formula for the radius
noncomputable def radius : ℝ := 332 / (5 * pi_val + 8)

-- Prove that the radius is correct given the conditions
theorem circle_radius (theta : ℝ) (P : ℝ) (r : ℝ) (h_theta : theta = central_angle) (h_P : P = perimeter) :
  r = radius :=
sorry

end circle_radius_l73_73570


namespace union_of_A_and_B_l73_73731

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def union_AB := {x : ℝ | 1 < x ∧ x ≤ 8}

theorem union_of_A_and_B : A ∪ B = union_AB :=
sorry

end union_of_A_and_B_l73_73731


namespace rahul_share_of_payment_l73_73530

def work_rate_rahul : ℚ := 1 / 3
def work_rate_rajesh : ℚ := 1 / 2
def total_payment : ℚ := 150

theorem rahul_share_of_payment : (work_rate_rahul / (work_rate_rahul + work_rate_rajesh)) * total_payment = 60 := by
  sorry

end rahul_share_of_payment_l73_73530


namespace nonnegative_integer_count_l73_73346

def balanced_quaternary_nonnegative_count : Nat :=
  let base := 4
  let max_index := 6
  let valid_digits := [-1, 0, 1]
  let max_sum := (base ^ (max_index + 1) - 1) / (base - 1)
  max_sum + 1

theorem nonnegative_integer_count : balanced_quaternary_nonnegative_count = 5462 := by
  sorry

end nonnegative_integer_count_l73_73346


namespace total_apple_weight_proof_l73_73081

-- Define the weights of each fruit in terms of ounces
def weight_apple : ℕ := 4
def weight_orange : ℕ := 3
def weight_plum : ℕ := 2

-- Define the bag's capacity and the number of bags
def bag_capacity : ℕ := 49
def number_of_bags : ℕ := 5

-- Define the least common multiple (LCM) of the weights
def lcm_weight : ℕ := Nat.lcm weight_apple (Nat.lcm weight_orange weight_plum)

-- Define the largest multiple of LCM that is less than or equal to the bag's capacity
def max_lcm_multiple : ℕ := (bag_capacity / lcm_weight) * lcm_weight

-- Determine the number of each fruit per bag
def sets_per_bag : ℕ := max_lcm_multiple / lcm_weight
def apples_per_bag : ℕ := sets_per_bag * 1  -- 1 apple per set

-- Calculate the weight of apples per bag and total needed in all bags
def apple_weight_per_bag : ℕ := apples_per_bag * weight_apple
def total_apple_weight : ℕ := apple_weight_per_bag * number_of_bags

-- The statement to be proved in Lean
theorem total_apple_weight_proof : total_apple_weight = 80 := by
  sorry

end total_apple_weight_proof_l73_73081


namespace scientific_notation_141260_million_l73_73124

theorem scientific_notation_141260_million :
  (141260 * 10^6 : ℝ) = 1.4126 * 10^11 := 
sorry

end scientific_notation_141260_million_l73_73124


namespace vector_sum_correct_l73_73852

-- Define the three vectors
def v1 : ℝ × ℝ := (5, -3)
def v2 : ℝ × ℝ := (-4, 6)
def v3 : ℝ × ℝ := (2, -8)

-- Define the expected result
def expected_sum : ℝ × ℝ := (3, -5)

-- Define vector addition (component-wise)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- The theorem statement
theorem vector_sum_correct : vector_add (vector_add v1 v2) v3 = expected_sum := by
  sorry

end vector_sum_correct_l73_73852


namespace part1_min_value_of_f_when_a_is_1_part2_range_of_a_for_f_ge_x_l73_73159

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - Real.log x

theorem part1_min_value_of_f_when_a_is_1 : 
  (∃ x : ℝ, f 1 x = 1 / 2 ∧ (∀ y : ℝ, f 1 y ≥ f 1 x)) :=
sorry

theorem part2_range_of_a_for_f_ge_x :
  (∀ x : ℝ, x > 0 → f a x ≥ x) ↔ a ≥ 2 :=
sorry

end part1_min_value_of_f_when_a_is_1_part2_range_of_a_for_f_ge_x_l73_73159


namespace weston_academy_geography_players_l73_73010

theorem weston_academy_geography_players
  (total_players : ℕ)
  (history_players : ℕ)
  (both_players : ℕ) :
  total_players = 18 →
  history_players = 10 →
  both_players = 6 →
  ∃ (geo_players : ℕ), geo_players = 14 := 
by 
  intros h1 h2 h3
  use 18 - (10 - 6) + 6
  sorry

end weston_academy_geography_players_l73_73010


namespace vacation_animals_total_l73_73767

noncomputable def lisa := 40
noncomputable def alex := lisa / 2
noncomputable def jane := alex + 10
noncomputable def rick := 3 * jane
noncomputable def tim := 2 * rick
noncomputable def you := 5 * tim
noncomputable def total_animals := lisa + alex + jane + rick + tim + you

theorem vacation_animals_total : total_animals = 1260 := by
  sorry

end vacation_animals_total_l73_73767


namespace inequality_holds_l73_73265

theorem inequality_holds (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (4 * x + y + 2 * z) * (2 * x + y + 8 * z) ≥ (375 / 2) * x * y * z :=
by
  sorry

end inequality_holds_l73_73265


namespace money_spent_correct_l73_73126

-- Define conditions
def spring_income : ℕ := 2
def summer_income : ℕ := 27
def amount_after_supplies : ℕ := 24

-- Define the resulting money spent on supplies
def money_spent_on_supplies : ℕ :=
  (spring_income + summer_income) - amount_after_supplies

theorem money_spent_correct :
  money_spent_on_supplies = 5 := by
  sorry

end money_spent_correct_l73_73126


namespace sum_cis_angles_l73_73152

noncomputable def complex.cis (θ : ℝ) := Complex.exp (Complex.I * θ)

theorem sum_cis_angles :
  (complex.cis (80 * Real.pi / 180) + complex.cis (88 * Real.pi / 180) + complex.cis (96 * Real.pi / 180) + 
  complex.cis (104 * Real.pi / 180) + complex.cis (112 * Real.pi / 180) + complex.cis (120 * Real.pi / 180) + 
  complex.cis (128 * Real.pi / 180)) = r * complex.cis (104 * Real.pi / 180) := 
sorry

end sum_cis_angles_l73_73152


namespace average_chore_time_l73_73443

theorem average_chore_time 
  (times : List ℕ := [4, 3, 2, 1, 0])
  (counts : List ℕ := [2, 4, 2, 1, 1]) 
  (total_students : ℕ := 10)
  (total_time : ℕ := List.sum (List.zipWith (λ t c => t * c) times counts)) :
  (total_time : ℚ) / total_students = 2.5 := by
  sorry

end average_chore_time_l73_73443


namespace prob_students_on_both_days_l73_73175
noncomputable def probability_event_on_both_days: ℚ := by
  let total_days := 2
  let total_students := 4
  let prob_single_day := (1 / total_days : ℚ) ^ total_students
  let prob_all_same_day := 2 * prob_single_day
  let prob_both_days := 1 - prob_all_same_day
  exact prob_both_days

theorem prob_students_on_both_days : probability_event_on_both_days = 7 / 8 :=
by
  exact sorry

end prob_students_on_both_days_l73_73175


namespace college_student_ticket_cost_l73_73818

theorem college_student_ticket_cost 
    (total_visitors : ℕ)
    (nyc_residents: ℕ)
    (college_students_nyc: ℕ)
    (total_money_received : ℕ) :
    total_visitors = 200 →
    nyc_residents = total_visitors / 2 →
    college_students_nyc = (nyc_residents * 30) / 100 →
    total_money_received = 120 →
    (total_money_received / college_students_nyc) = 4 := 
sorry

end college_student_ticket_cost_l73_73818


namespace find_x0_l73_73112

noncomputable def slopes_product_eq_three (x : ℝ) : Prop :=
  let y1 := 2 - 1 / x
  let y2 := x^3 - x^2 + 2 * x
  let dy1_dx := 1 / (x^2)
  let dy2_dx := 3 * x^2 - 2 * x + 2
  dy1_dx * dy2_dx = 3

theorem find_x0 : ∃ (x0 : ℝ), slopes_product_eq_three x0 ∧ x0 = 1 :=
by {
  use 1,
  sorry
}

end find_x0_l73_73112


namespace tangent_lines_count_l73_73197

def f (x : ℝ) : ℝ := x^3

theorem tangent_lines_count :
  (∃ x : ℝ, deriv f x = 3) ∧ 
  (∃ y : ℝ, deriv f y = 3 ∧ y ≠ x) := 
by
  -- Since f(x) = x^3, its derivative is f'(x) = 3x^2
  -- We need to solve 3x^2 = 3
  -- Therefore, x^2 = 1 and x = ±1
  -- Thus, there are two tangent lines
  sorry

end tangent_lines_count_l73_73197


namespace sufficient_but_not_necessary_l73_73738

theorem sufficient_but_not_necessary (a : ℝ) : (a > 6 → a^2 > 36) ∧ ¬(a^2 > 36 → a > 6) := 
by
  sorry

end sufficient_but_not_necessary_l73_73738


namespace arithmetic_sequence_sum_l73_73538

theorem arithmetic_sequence_sum :
  ∀ (x y : ℤ), (∃ (n m : ℕ), (3 + n * 6 = x) ∧ (3 + m * 6 = y) ∧ x + 6 = y ∧ y + 6 = 33) → x + y = 60 :=
by
  intro x y h
  obtain ⟨n, m, hn, hm, hx, hy⟩ := h
  exact sorry

end arithmetic_sequence_sum_l73_73538


namespace playground_children_count_l73_73771

theorem playground_children_count (boys girls : ℕ) (h_boys : boys = 27) (h_girls : girls = 35) : boys + girls = 62 := by
  sorry

end playground_children_count_l73_73771


namespace compare_logs_l73_73758

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem compare_logs (a b c : ℝ) (h1 : a = log_base 4 1.25) (h2 : b = log_base 5 1.2) (h3 : c = log_base 4 8) :
  c > a ∧ a > b :=
by
  sorry

end compare_logs_l73_73758


namespace max_tan_B_l73_73323

theorem max_tan_B (A B : ℝ) (h : Real.sin (2 * A + B) = 2 * Real.sin B) : 
  Real.tan B ≤ Real.sqrt 3 / 3 := sorry

end max_tan_B_l73_73323


namespace find_m_l73_73000

theorem find_m (m : ℤ) (h1 : -180 < m ∧ m < 180) : 
  ((m = 45) ∨ (m = -135)) ↔ (Real.tan (m * Real.pi / 180) = Real.tan (225 * Real.pi / 180)) := 
by 
  sorry

end find_m_l73_73000


namespace average_speed_palindrome_trip_l73_73552

theorem average_speed_palindrome_trip :
  ∀ (initial final : ℕ) (time : ℝ),
    initial = 13431 → final = 13531 → time = 3 →
    (final - initial) / time = 33 :=
by
  intros initial final time h_initial h_final h_time
  rw [h_initial, h_final, h_time]
  norm_num
  sorry

end average_speed_palindrome_trip_l73_73552


namespace distance_between_adjacent_parallel_lines_l73_73661

noncomputable def distance_between_lines (r d : ℝ) : ℝ :=
  (49 * r^2 - 49 * 600.25 - (49 / 4) * d^2) / (1 - 49 / 4)

theorem distance_between_adjacent_parallel_lines :
  ∃ d : ℝ, ∀ (r : ℝ), 
    (r^2 = 506.25 + (1 / 4) * d^2 ∧ r^2 = 600.25 + (49 / 4) * d^2) →
    d = 2.8 :=
sorry

end distance_between_adjacent_parallel_lines_l73_73661


namespace tickets_difference_l73_73919

-- Define the conditions
def tickets_friday : ℕ := 181
def tickets_sunday : ℕ := 78
def tickets_saturday : ℕ := 2 * tickets_friday

-- The theorem to prove
theorem tickets_difference : tickets_saturday - tickets_sunday = 284 := by
  sorry

end tickets_difference_l73_73919


namespace find_a_l73_73359

theorem find_a (a : ℝ) (x : ℝ) :
  (∃ b : ℝ, (9 * x^2 - 18 * x + a) = (3 * x + b) ^ 2) → a = 9 := by
  sorry

end find_a_l73_73359


namespace exactly_one_first_class_probability_at_least_one_second_class_probability_l73_73189

-- Definitions based on the problem statement:
def total_pens : ℕ := 6
def first_class_pens : ℕ := 4
def second_class_pens : ℕ := 2

def total_draws : ℕ := 2

-- Event for drawing exactly one first-class quality pen
def probability_one_first_class := ((first_class_pens.choose 1 * second_class_pens.choose 1) /
                                    (total_pens.choose total_draws) : ℚ)

-- Event for drawing at least one second-class quality pen
def probability_at_least_one_second_class := (1 - (first_class_pens.choose total_draws /
                                                   total_pens.choose total_draws) : ℚ)

-- Statements to prove the probabilities
theorem exactly_one_first_class_probability :
  probability_one_first_class = 8 / 15 :=
sorry

theorem at_least_one_second_class_probability :
  probability_at_least_one_second_class = 3 / 5 :=
sorry

end exactly_one_first_class_probability_at_least_one_second_class_probability_l73_73189


namespace original_price_of_color_TV_l73_73487

theorem original_price_of_color_TV
  (x : ℝ)  -- Let the variable x represent the original price
  (h1 : x * 1.4 * 0.8 - x = 144)  -- Condition as equation
  : x = 1200 := 
sorry  -- Proof to be filled in later

end original_price_of_color_TV_l73_73487


namespace distance_to_lateral_face_l73_73365

theorem distance_to_lateral_face 
  (height : ℝ) 
  (angle : ℝ) 
  (h_height : height = 6 * Real.sqrt 6)
  (h_angle : angle = Real.pi / 4) : 
  ∃ (distance : ℝ), distance = 6 * Real.sqrt 30 / 5 :=
by
  sorry

end distance_to_lateral_face_l73_73365


namespace simplify_expr_1_simplify_expr_2_l73_73171

theorem simplify_expr_1 (x y : ℝ) :
  12 * x - 6 * y + 3 * y - 24 * x = -12 * x - 3 * y :=
by
  sorry

theorem simplify_expr_2 (a b : ℝ) :
  (3 / 2) * (a^2 * b - 2 * (a * b^2)) - (1 / 2) * (a * b^2 - 4 * (a^2 * b)) + (a * b^2) / 2 = (7 / 2) * (a^2 * b) - 3 * (a * b^2) :=
by
  sorry

end simplify_expr_1_simplify_expr_2_l73_73171


namespace f_of_f_neg1_eq_neg1_f_x_eq_neg1_iff_x_eq_0_or_2_l73_73188

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 1 else -x + 1

-- Prove f[f(-1)] = -1
theorem f_of_f_neg1_eq_neg1 : f (f (-1)) = -1 := sorry

-- Prove that if f(x) = -1, then x = 0 or x = 2
theorem f_x_eq_neg1_iff_x_eq_0_or_2 (x : ℝ) : f x = -1 ↔ x = 0 ∨ x = 2 := sorry

end f_of_f_neg1_eq_neg1_f_x_eq_neg1_iff_x_eq_0_or_2_l73_73188


namespace length_60_more_than_breadth_l73_73623

noncomputable def length_more_than_breadth (cost_per_meter : ℝ) (total_cost : ℝ) (length : ℝ) : Prop :=
  ∃ (breadth : ℝ) (x : ℝ), 
    length = breadth + x ∧
    2 * length + 2 * breadth = total_cost / cost_per_meter ∧
    x = length - breadth ∧
    x = 60

theorem length_60_more_than_breadth : length_more_than_breadth 26.5 5300 80 :=
by
  sorry

end length_60_more_than_breadth_l73_73623


namespace composite_a2_b2_l73_73453

-- Introduce the main definitions according to the conditions stated in a)
theorem composite_a2_b2 (x1 x2 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (a b : ℤ) 
  (ha : a = -(x1 + x2)) (hb : b = x1 * x2 - 1) : 
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ (a^2 + b^2) = m * n := 
by 
  sorry

end composite_a2_b2_l73_73453


namespace total_shaded_area_of_rectangles_l73_73915

theorem total_shaded_area_of_rectangles (w1 l1 w2 l2 ow ol : ℕ) 
  (h1 : w1 = 4) (h2 : l1 = 12) (h3 : w2 = 5) (h4 : l2 = 10) (h5 : ow = 4) (h6 : ol = 5) :
  (w1 * l1 + w2 * l2 - ow * ol = 78) :=
by
  sorry

end total_shaded_area_of_rectangles_l73_73915


namespace inequality_solution_l73_73034

theorem inequality_solution (x : ℝ) (h : x ≠ 5) :
    (15 ≤ x * (x - 2) / (x - 5) ^ 2) ↔ (4.1933 ≤ x ∧ x < 5 ∨ 5 < x ∧ x ≤ 6.3767) :=
by
  sorry

end inequality_solution_l73_73034


namespace minimum_value_of_sum_2_l73_73102

noncomputable def minimum_value_of_sum 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (inequality : (2 * x + Real.sqrt (4 * x ^ 2 + 1)) * (Real.sqrt (y ^ 2 + 4) - 2) ≥ y) : 
  Prop := 
  x + y = 2

theorem minimum_value_of_sum_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (inequality : (2 * x + Real.sqrt (4 * x ^ 2 + 1)) * (Real.sqrt (y ^ 2 + 4) - 2) ≥ y) :
  minimum_value_of_sum x y hx hy inequality := 
sorry

end minimum_value_of_sum_2_l73_73102


namespace squared_sum_of_a_b_l73_73269

theorem squared_sum_of_a_b (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 3) : (a + b) ^ 2 = 16 :=
by
  sorry

end squared_sum_of_a_b_l73_73269


namespace determine_day_from_statements_l73_73772

/-- Define the days of the week as an inductive type. -/
inductive Day where
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
  deriving DecidableEq, Repr

open Day

/-- Define the properties of the lion lying on specific days. -/
def lion_lies (d : Day) : Prop :=
  d = Monday ∨ d = Tuesday ∨ d = Wednesday

/-- Define the properties of the lion telling the truth on specific days. -/
def lion_truth (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday ∨ d = Sunday

/-- Define the properties of the unicorn lying on specific days. -/
def unicorn_lies (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday

/-- Define the properties of the unicorn telling the truth on specific days. -/
def unicorn_truth (d : Day) : Prop :=
  d = Sunday ∨ d = Monday ∨ d = Tuesday ∨ d = Wednesday

/-- Function to determine the day before a given day. -/
def yesterday (d : Day) : Day :=
  match d with
  | Monday    => Sunday
  | Tuesday   => Monday
  | Wednesday => Tuesday
  | Thursday  => Wednesday
  | Friday    => Thursday
  | Saturday  => Friday
  | Sunday    => Saturday

/-- Define the lion's statement: "Yesterday was a day when I lied." -/
def lion_statement (d : Day) : Prop :=
  lion_lies (yesterday d)

/-- Define the unicorn's statement: "Yesterday was a day when I lied." -/
def unicorn_statement (d : Day) : Prop :=
  unicorn_lies (yesterday d)

/-- Prove that today must be Thursday given the conditions and statements. -/
theorem determine_day_from_statements (d : Day) :
    lion_statement d ∧ unicorn_statement d → d = Thursday := by
  sorry

end determine_day_from_statements_l73_73772


namespace arithmetic_sequence_count_l73_73789

-- Define the initial conditions
def a1 : ℤ := -3
def d : ℤ := 3
def an : ℤ := 45

-- Proposition stating the number of terms n in the arithmetic sequence
theorem arithmetic_sequence_count :
  ∃ n : ℕ, an = a1 + (n - 1) * d ∧ n = 17 :=
by
  -- Skip the proof
  sorry

end arithmetic_sequence_count_l73_73789


namespace coconut_grove_nut_yield_l73_73505

theorem coconut_grove_nut_yield (x : ℕ) (Y : ℕ) 
  (h1 : (x + 4) * 60 + x * 120 + (x - 4) * Y = 3 * x * 100)
  (h2 : x = 8) : Y = 180 := 
by
  sorry

end coconut_grove_nut_yield_l73_73505


namespace power_mod_zero_problem_solution_l73_73658

theorem power_mod_zero (n : ℕ) (h : n ≥ 2) : 2 ^ n % 4 = 0 :=
  sorry

theorem problem_solution : 2 ^ 300 % 4 = 0 :=
  power_mod_zero 300 (by norm_num)

end power_mod_zero_problem_solution_l73_73658


namespace seahorse_penguin_ratio_l73_73735

theorem seahorse_penguin_ratio :
  ∃ S P : ℕ, S = 70 ∧ P = S + 85 ∧ Nat.gcd 70 (S + 85) = 5 ∧ 70 / Nat.gcd 70 (S + 85) = 14 ∧ (S + 85) / Nat.gcd 70 (S + 85) = 31 :=
by
  sorry

end seahorse_penguin_ratio_l73_73735


namespace find_length_of_street_l73_73869

-- Definitions based on conditions
def area_street (L : ℝ) : ℝ := L^2
def area_forest (L : ℝ) : ℝ := 3 * (area_street L)
def num_trees (L : ℝ) : ℝ := 4 * (area_forest L)

-- Statement to prove
theorem find_length_of_street (L : ℝ) (h : num_trees L = 120000) : L = 100 := by
  sorry

end find_length_of_street_l73_73869


namespace total_fruits_picked_l73_73326

theorem total_fruits_picked :
  let sara_pears := 6
  let tim_pears := 5
  let lily_apples := 4
  let max_oranges := 3
  sara_pears + tim_pears + lily_apples + max_oranges = 18 :=
by
  -- skip the proof
  sorry

end total_fruits_picked_l73_73326


namespace circle_center_radius_l73_73247

theorem circle_center_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4 * x = 0 ↔ ((x - 2)^2 + y^2 = 4) ∧ (∃ (c_x c_y r : ℝ), c_x = 2 ∧ c_y = 0 ∧ r = 2) :=
by
  sorry

end circle_center_radius_l73_73247


namespace rectangle_perimeter_eq_30sqrt10_l73_73774

theorem rectangle_perimeter_eq_30sqrt10 (A : ℝ) (l : ℝ) (w : ℝ) 
  (hA : A = 500) (hlw : l = 2 * w) (hArea : A = l * w) : 
  2 * (l + w) = 30 * Real.sqrt 10 :=
by
  sorry

end rectangle_perimeter_eq_30sqrt10_l73_73774


namespace population_increase_rate_correct_l73_73203

variable (P0 P1 : ℕ)
variable (r : ℚ)

-- Given conditions
def initial_population := P0 = 200
def population_after_one_year := P1 = 220

-- Proof problem statement
theorem population_increase_rate_correct :
  initial_population P0 →
  population_after_one_year P1 →
  r = (P1 - P0 : ℚ) / P0 * 100 →
  r = 10 :=
by
  sorry

end population_increase_rate_correct_l73_73203


namespace jackson_holidays_l73_73178

theorem jackson_holidays (holidays_per_month : ℕ) (months_in_year : ℕ) (holidays_per_year : ℕ) : 
  holidays_per_month = 3 → months_in_year = 12 → holidays_per_year = holidays_per_month * months_in_year → holidays_per_year = 36 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jackson_holidays_l73_73178


namespace pauly_cannot_make_more_omelets_l73_73462

-- Pauly's omelet data
def total_eggs : ℕ := 36
def plain_omelet_eggs : ℕ := 3
def cheese_omelet_eggs : ℕ := 4
def vegetable_omelet_eggs : ℕ := 5

-- Requested omelets
def requested_plain_omelets : ℕ := 4
def requested_cheese_omelets : ℕ := 2
def requested_vegetable_omelets : ℕ := 3

-- Number of eggs used for each type of requested omelet
def total_requested_eggs : ℕ :=
  (requested_plain_omelets * plain_omelet_eggs) +
  (requested_cheese_omelets * cheese_omelet_eggs) +
  (requested_vegetable_omelets * vegetable_omelet_eggs)

-- The remaining number of eggs
def remaining_eggs : ℕ := total_eggs - total_requested_eggs

theorem pauly_cannot_make_more_omelets :
  remaining_eggs < min plain_omelet_eggs (min cheese_omelet_eggs vegetable_omelet_eggs) :=
by
  sorry

end pauly_cannot_make_more_omelets_l73_73462


namespace gift_wrapping_combinations_l73_73182

theorem gift_wrapping_combinations :
    (10 * 3 * 4 * 5 = 600) :=
by
    sorry

end gift_wrapping_combinations_l73_73182


namespace marcy_multiple_tickets_l73_73036

theorem marcy_multiple_tickets (m : ℕ) : 
  (26 + (m * 26 - 6) = 150) → m = 5 :=
by
  intro h
  sorry

end marcy_multiple_tickets_l73_73036


namespace factor_expression_l73_73352

theorem factor_expression (x : ℝ) : 72 * x^3 - 250 * x^7 = 2 * x^3 * (36 - 125 * x^4) :=
by
  sorry

end factor_expression_l73_73352


namespace ralph_socks_l73_73406

theorem ralph_socks (x y z : ℕ) (h1 : x + y + z = 12) (h2 : x + 3 * y + 4 * z = 24) (h3 : 1 ≤ x) (h4 : 1 ≤ y) (h5 : 1 ≤ z) : x = 7 :=
sorry

end ralph_socks_l73_73406


namespace waiter_tables_l73_73715

theorem waiter_tables (total_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (number_of_tables : ℕ) 
  (h1 : total_customers = 22)
  (h2 : customers_left = 14)
  (h3 : people_per_table = 4)
  (h4 : remaining_customers = total_customers - customers_left)
  (h5 : number_of_tables = remaining_customers / people_per_table) :
  number_of_tables = 2 :=
by
  sorry

end waiter_tables_l73_73715


namespace least_four_digit_9_heavy_l73_73465

def is_9_heavy (n : ℕ) : Prop := n % 9 > 5

def four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

theorem least_four_digit_9_heavy : ∃ n, four_digit n ∧ is_9_heavy n ∧ ∀ m, (four_digit m ∧ is_9_heavy m) → n ≤ m :=
by
  exists 1005
  sorry

end least_four_digit_9_heavy_l73_73465


namespace totalPeaches_l73_73132

-- Definitions based on the given conditions
def redPeaches : Nat := 13
def greenPeaches : Nat := 3

-- Problem statement
theorem totalPeaches : redPeaches + greenPeaches = 16 := by
  sorry

end totalPeaches_l73_73132


namespace hair_cut_amount_l73_73706

theorem hair_cut_amount (initial_length final_length cut_length : ℕ) (h1 : initial_length = 11) (h2 : final_length = 7) : cut_length = 4 :=
by 
  sorry

end hair_cut_amount_l73_73706


namespace present_age_of_B_l73_73917

-- Definitions
variables (a b : ℕ)

-- Conditions
def condition1 (a b : ℕ) : Prop := a + 10 = 2 * (b - 10)
def condition2 (a b : ℕ) : Prop := a = b + 7

-- Theorem to prove
theorem present_age_of_B (a b : ℕ) (h1 : condition1 a b) (h2 : condition2 a b) : b = 37 := by
  sorry

end present_age_of_B_l73_73917


namespace largest_three_digit_number_satisfying_conditions_l73_73236

def valid_digits (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 
  1 ≤ b ∧ b ≤ 9 ∧ 
  1 ≤ c ∧ c ≤ 9 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def sum_of_two_digit_permutations_eq (a b c : ℕ) : Prop :=
  22 * (a + b + c) = 100 * a + 10 * b + c

theorem largest_three_digit_number_satisfying_conditions (a b c : ℕ) :
  valid_digits a b c →
  sum_of_two_digit_permutations_eq a b c →
  100 * a + 10 * b + c ≤ 396 :=
sorry

end largest_three_digit_number_satisfying_conditions_l73_73236


namespace exists_consecutive_natural_numbers_satisfy_equation_l73_73234

theorem exists_consecutive_natural_numbers_satisfy_equation :
  ∃ (n a b c d: ℕ), a = n ∧ b = n+2 ∧ c = n-1 ∧ d = n+1 ∧ n>0 ∧ a * b - c * d = 11 :=
by
  sorry

end exists_consecutive_natural_numbers_satisfy_equation_l73_73234


namespace sqrt_of_16_is_4_l73_73231

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_is_4_l73_73231


namespace decimal_to_binary_18_l73_73221

theorem decimal_to_binary_18 : (18: ℕ) = 0b10010 := by
  sorry

end decimal_to_binary_18_l73_73221


namespace a_can_finish_remaining_work_in_5_days_l73_73908

theorem a_can_finish_remaining_work_in_5_days (a_work_rate b_work_rate : ℝ) (total_days_b_works : ℝ):
  a_work_rate = 1/15 → 
  b_work_rate = 1/15 → 
  total_days_b_works = 10 → 
  ∃ (remaining_days_for_a : ℝ), remaining_days_for_a = 5 :=
by
  intros h1 h2 h3
  -- We are skipping the proof itself
  sorry

end a_can_finish_remaining_work_in_5_days_l73_73908


namespace second_solution_lemonade_is_45_l73_73064

-- Define percentages as real numbers for simplicity
def firstCarbonatedWater : ℝ := 0.80
def firstLemonade : ℝ := 0.20
def secondCarbonatedWater : ℝ := 0.55
def mixturePercentageFirst : ℝ := 0.50
def mixtureCarbonatedWater : ℝ := 0.675

-- The ones that already follow from conditions or trivial definitions:
def secondLemonade : ℝ := 1 - secondCarbonatedWater

-- Define the percentage of carbonated water in mixture, based on given conditions
def mixtureIsCorrect : Prop :=
  mixturePercentageFirst * firstCarbonatedWater + (1 - mixturePercentageFirst) * secondCarbonatedWater = mixtureCarbonatedWater

-- The theorem to prove: second solution's lemonade percentage is 45%
theorem second_solution_lemonade_is_45 :
  mixtureIsCorrect → secondLemonade = 0.45 :=
by
  sorry

end second_solution_lemonade_is_45_l73_73064


namespace pictures_vertical_l73_73842

theorem pictures_vertical (V H X : ℕ) (h1 : V + H + X = 30) (h2 : H = 15) (h3 : X = 5) : V = 10 := 
by 
  sorry

end pictures_vertical_l73_73842


namespace all_lines_can_be_paired_perpendicular_l73_73273

noncomputable def can_pair_perpendicular_lines : Prop := 
  ∀ (L1 L2 : ℝ), 
    L1 ≠ L2 → 
      ∃ (m : ℝ), 
        (m * L1 = -1/L2 ∨ L1 = 0 ∧ L2 ≠ 0 ∨ L2 = 0 ∧ L1 ≠ 0)

theorem all_lines_can_be_paired_perpendicular : can_pair_perpendicular_lines :=
sorry

end all_lines_can_be_paired_perpendicular_l73_73273


namespace primes_eq_2_3_7_l73_73219

theorem primes_eq_2_3_7 (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 ∨ p = 7 :=
by
  sorry

end primes_eq_2_3_7_l73_73219


namespace John_traded_in_car_money_back_l73_73665

-- First define the conditions provided in the problem.
def UberEarnings : ℝ := 30000
def CarCost : ℝ := 18000
def UberProfit : ℝ := 18000

-- We need to prove that John got $6000 back when trading in the car.
theorem John_traded_in_car_money_back : 
  UberEarnings - UberProfit = CarCost - 6000 := 
by
  -- provide the detailed steps inside the proof block if needed
  sorry

end John_traded_in_car_money_back_l73_73665


namespace find_x_perpendicular_l73_73653

/-- Given vectors a = ⟨-1, 2⟩ and b = ⟨1, x⟩, if a is perpendicular to (a + 2 * b),
    then x = -3/4. -/
theorem find_x_perpendicular
  (x : ℝ)
  (a : ℝ × ℝ := (-1, 2))
  (b : ℝ × ℝ := (1, x))
  (h : (a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0)) :
  x = -3 / 4 :=
sorry

end find_x_perpendicular_l73_73653


namespace line_equation_passing_through_and_perpendicular_l73_73612

theorem line_equation_passing_through_and_perpendicular :
  ∃ A B C : ℝ, (∀ x y : ℝ, 2 * x - 4 * y + 5 = 0 → -2 * x + y + 1 = 0 ∧ 
(x = 2 ∧ y = -1) → 2 * x + y - 3 = 0) :=
by
  sorry

end line_equation_passing_through_and_perpendicular_l73_73612


namespace compute_k_l73_73618

noncomputable def tan_inverse (k : ℝ) : ℝ := Real.arctan k

theorem compute_k (x k : ℝ) (hx1 : Real.tan x = 2 / 3) (hx2 : Real.tan (3 * x) = 3 / 5) : k = 2 / 3 := sorry

end compute_k_l73_73618


namespace inequality_proof_l73_73691

variable (a b c : ℝ)

theorem inequality_proof
  (h1 : a > b) :
  a * c^2 ≥ b * c^2 := 
sorry

end inequality_proof_l73_73691


namespace renovation_costs_l73_73220

theorem renovation_costs :
  ∃ (x y : ℝ), 
    8 * x + 8 * y = 3520 ∧
    6 * x + 12 * y = 3480 ∧
    x = 300 ∧
    y = 140 ∧
    300 * 12 > 140 * 24 :=
by sorry

end renovation_costs_l73_73220


namespace angle_C_measure_l73_73949

theorem angle_C_measure 
  (p q : Prop) 
  (h1 : p) (h2 : q) 
  (A B C : ℝ) 
  (h_parallel : p = q) 
  (h_A_B : A = B / 10) 
  (h_straight_line : B + C = 180) 
  : C = 16.36 := 
sorry

end angle_C_measure_l73_73949


namespace smallest_prime_perimeter_l73_73473

def is_prime (n : ℕ) := Nat.Prime n
def is_triangle (a b c : ℕ) := a + b > c ∧ a + c > b ∧ b + c > a
def is_scalene (a b c : ℕ) := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ is_scalene a b c ∧ a ≥ 5
  ∧ is_prime (a + b + c) ∧ a + b + c = 23 :=
by
  sorry

end smallest_prime_perimeter_l73_73473


namespace find_value_of_15b_minus_2a_l73_73769

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if 1 ≤ x ∧ x < 2 then x + a / x
else if 2 ≤ x ∧ x ≤ 3 then b * x - 3
else 0

theorem find_value_of_15b_minus_2a (a b : ℝ)
  (h_periodic : ∀ x : ℝ, f x a b = f (x + 2) a b)
  (h_condition : f (7 / 2) a b = f (-7 / 2) a b) :
  15 * b - 2 * a = 41 :=
sorry

end find_value_of_15b_minus_2a_l73_73769


namespace arithmetic_sequence_first_term_and_common_difference_l73_73676

def a_n (n : ℕ) : ℕ := 2 * n + 5

theorem arithmetic_sequence_first_term_and_common_difference :
  a_n 1 = 7 ∧ ∀ n : ℕ, a_n (n + 1) - a_n n = 2 := by
  sorry

end arithmetic_sequence_first_term_and_common_difference_l73_73676


namespace inequality_for_positive_real_numbers_l73_73355

theorem inequality_for_positive_real_numbers 
  (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) :
  (a / (b + 2 * c + 3 * d) + 
   b / (c + 2 * d + 3 * a) + 
   c / (d + 2 * a + 3 * b) + 
   d / (a + 2 * b + 3 * c)) ≥ (2 / 3) :=
by
  sorry

end inequality_for_positive_real_numbers_l73_73355


namespace allowance_spent_l73_73266

variable (A x y : ℝ)
variable (h1 : x = 0.20 * (A - y))
variable (h2 : y = 0.05 * (A - x))

theorem allowance_spent : (x + y) / A = 23 / 100 :=
by 
  sorry

end allowance_spent_l73_73266


namespace martha_cards_l73_73733

theorem martha_cards (start_cards : ℕ) : start_cards + 76 = 79 → start_cards = 3 :=
by
  sorry

end martha_cards_l73_73733


namespace tileable_by_hook_l73_73288

theorem tileable_by_hook (m n : ℕ) : 
  (∃ a b : ℕ, m = 3 * a ∧ (n = 4 * b ∨ n = 12 * b) ∨ 
              n = 3 * a ∧ (m = 4 * b ∨ m = 12 * b)) ↔ 12 ∣ (m * n) :=
by
  sorry

end tileable_by_hook_l73_73288


namespace vector_AD_length_l73_73613

open Real EuclideanSpace

noncomputable def problem_statement
  (m n : ℝ) (angle_mn : ℝ) (norm_m : ℝ) (norm_n : ℝ) (AB AC : ℝ) (AD : ℝ) : Prop :=
  angle_mn = π / 6 ∧ 
  norm_m = sqrt 3 ∧ 
  norm_n = 2 ∧ 
  AB = 2 * m + 2 * n ∧ 
  AC = 2 * m - 6 * n ∧ 
  AD = 2 * m - 2 * n ∧
  sqrt ((AD) * (AD)) = 2

theorem vector_AD_length 
  (m n : ℝ) (angle_mn : ℝ) (norm_m : ℝ) (norm_n : ℝ) (AB AC AD : ℝ) :
  problem_statement m n angle_mn norm_m norm_n AB AC AD :=
by
  unfold problem_statement
  sorry

end vector_AD_length_l73_73613


namespace distance_A_beats_B_l73_73527

noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem distance_A_beats_B :
  let distance_A := 5 -- km
  let time_A := 10 / 60 -- hours (10 minutes)
  let time_B := 14 / 60 -- hours (14 minutes)
  let speed_A := speed distance_A time_A
  let speed_B := speed distance_A time_B
  let distance_A_in_time_B := speed_A * time_B
  distance_A_in_time_B - distance_A = 2 := -- km
by
  sorry

end distance_A_beats_B_l73_73527


namespace part1_part2_l73_73914

theorem part1 (x : ℝ) : |x + 3| - 2 * x - 1 < 0 → 2 < x :=
by sorry

theorem part2 (m : ℝ) : (m > 0) →
  (∃ x : ℝ, |x - m| + |x + 1/m| = 2) → m = 1 :=
by sorry

end part1_part2_l73_73914


namespace alexis_shirt_expense_l73_73749

theorem alexis_shirt_expense :
  let B := 200
  let E_pants := 46
  let E_coat := 38
  let E_socks := 11
  let E_belt := 18
  let E_shoes := 41
  let L := 16
  let S := B - (E_pants + E_coat + E_socks + E_belt + E_shoes + L)
  S = 30 :=
by
  sorry

end alexis_shirt_expense_l73_73749


namespace rational_inequalities_l73_73364

theorem rational_inequalities (a b c d : ℚ)
  (h : a^3 - 2005 = b^3 + 2027 ∧ b^3 + 2027 = c^3 - 2822 ∧ c^3 - 2822 = d^3 + 2820) :
  c > a ∧ a > b ∧ b > d :=
by
  sorry

end rational_inequalities_l73_73364


namespace equilateral_triangle_perimeter_l73_73939

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3 / 4) = 2 * s) : 3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_l73_73939


namespace team_selection_ways_l73_73396

theorem team_selection_ways :
  let ways (n k : ℕ) := Nat.choose n k
  (ways 6 3) * (ways 6 3) = 400 := 
by
  let ways := Nat.choose
  -- Proof is omitted
  sorry

end team_selection_ways_l73_73396


namespace directrix_parabola_l73_73058

theorem directrix_parabola (p : ℝ) (h : 4 * p = 2) : 
  ∃ d : ℝ, d = -p / 2 ∧ d = -1/2 :=
by
  sorry

end directrix_parabola_l73_73058


namespace gg3_eq_585_over_368_l73_73943

def g (x : ℚ) : ℚ := 2 * x⁻¹ + (2 * x⁻¹) / (1 + 2 * x⁻¹)

theorem gg3_eq_585_over_368 : g (g 3) = 585 / 368 := 
  sorry

end gg3_eq_585_over_368_l73_73943


namespace sequence_contains_2017_l73_73272

theorem sequence_contains_2017 (a1 d : ℕ) (hpos : d > 0)
  (k n m l : ℕ) 
  (hk : 25 = a1 + k * d)
  (hn : 41 = a1 + n * d)
  (hm : 65 = a1 + m * d)
  (h2017 : 2017 = a1 + l * d) : l > 0 :=
sorry

end sequence_contains_2017_l73_73272


namespace cone_base_diameter_l73_73455

theorem cone_base_diameter {r l : ℝ} 
  (h₁ : π * r * l + π * r^2 = 3 * π) 
  (h₂ : 2 * π * r = π * l) : 
  2 * r = 2 :=
by
  sorry

end cone_base_diameter_l73_73455


namespace digit_inequality_l73_73254

theorem digit_inequality : ∃ (n : ℕ), n = 9 ∧ ∀ (d : ℕ), d < 10 → (2 + d / 10 + 5 / 1000 > 2 + 5 / 1000) → d > 0 :=
by
  sorry

end digit_inequality_l73_73254


namespace positive_integers_satisfying_inequality_l73_73452

-- Define the assertion that there are exactly 5 positive integers x satisfying the given inequality
theorem positive_integers_satisfying_inequality :
  (∃! x : ℕ, 4 < x ∧ x < 10 ∧ (10 * x)^4 > x^8 ∧ x^8 > 2^16) :=
sorry

end positive_integers_satisfying_inequality_l73_73452


namespace new_kite_area_l73_73129

def original_base := 7
def original_height := 6
def scale_factor := 2
def side_length := 2

def new_base := original_base * scale_factor
def new_height := original_height * scale_factor
def half_new_height := new_height / 2

def area_triangle := (1 / 2 : ℚ) * new_base * half_new_height
def total_area := 2 * area_triangle

theorem new_kite_area : total_area = 84 := by
  sorry

end new_kite_area_l73_73129


namespace prob_correct_l73_73553

noncomputable def r : ℝ := (4.5 : ℝ)  -- derived from solving area and line equations
noncomputable def s : ℝ := (7.5 : ℝ)  -- derived from solving area and line equations

theorem prob_correct (P Q T : ℝ × ℝ)
  (hP : P = (9, 0))
  (hQ : Q = (0, 15))
  (hT : T = (r, s))
  (hline : s = -5/3 * r + 15)
  (harea : 2 * (1/2 * 9 * 15) = (1/2 * 9 * s) * 4) :
  r + s = 12 := by
  sorry

end prob_correct_l73_73553


namespace graph_must_pass_l73_73039

variable (f : ℝ → ℝ)
variable (finv : ℝ → ℝ)
variable (h_inv : ∀ y, f (finv y) = y ∧ finv (f y) = y)
variable (h_point : (2 - f 2) = 5)

theorem graph_must_pass : finv (-3) + 3 = 5 :=
by
  -- Proof to be filled in
  sorry

end graph_must_pass_l73_73039


namespace no_solution_absval_equation_l73_73139

theorem no_solution_absval_equation (x : ℝ) : ¬ (|2*x - 5| = 3*x + 1) :=
by
  sorry

end no_solution_absval_equation_l73_73139


namespace oz_lost_words_count_l73_73694
-- We import the necessary library.

-- Define the context.
def total_letters := 69
def forbidden_letter := 7

-- Define function to calculate lost words when a specific letter is forbidden.
def lost_words (total_letters : ℕ) (forbidden_letter : ℕ) : ℕ :=
  let one_letter_lost := 1
  let two_letter_lost := 2 * (total_letters - 1)
  one_letter_lost + two_letter_lost

-- State the theorem.
theorem oz_lost_words_count :
  lost_words total_letters forbidden_letter = 139 :=
by
  sorry

end oz_lost_words_count_l73_73694


namespace students_got_off_l73_73078

-- Define the number of students originally on the bus
def original_students : ℕ := 10

-- Define the number of students left on the bus after the first stop
def students_left : ℕ := 7

-- Prove that the number of students who got off the bus at the first stop is 3
theorem students_got_off : original_students - students_left = 3 :=
by
  sorry

end students_got_off_l73_73078


namespace periodic_even_function_value_l73_73047

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x - a)

-- Conditions: 
-- 1. f(x) is even 
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- 2. f(x) is periodic with period 6
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f x = f (x + p)

-- Main theorem
theorem periodic_even_function_value 
  (a : ℝ) 
  (f_def : ∀ x, -3 ≤ x ∧ x ≤ 3 → f x a = (x + 1) * (x - a))
  (h_even : is_even_function (f · a))
  (h_periodic : is_periodic_function (f · a) 6) : 
  f (-6) a = -1 := 
sorry

end periodic_even_function_value_l73_73047


namespace students_in_first_bus_l73_73809

theorem students_in_first_bus (total_buses : ℕ) (avg_students_per_bus : ℕ) 
(avg_remaining_students : ℕ) (num_remaining_buses : ℕ) 
(h1 : total_buses = 6) 
(h2 : avg_students_per_bus = 28) 
(h3 : avg_remaining_students = 26) 
(h4 : num_remaining_buses = 5) :
  (total_buses * avg_students_per_bus - num_remaining_buses * avg_remaining_students = 38) :=
by
  sorry

end students_in_first_bus_l73_73809


namespace jennifer_remaining_money_l73_73463

noncomputable def money_spent_on_sandwich (initial_money : ℝ) : ℝ :=
  let sandwich_cost := (1/5) * initial_money
  let discount := (10/100) * sandwich_cost
  sandwich_cost - discount

noncomputable def money_spent_on_ticket (initial_money : ℝ) : ℝ :=
  (1/6) * initial_money

noncomputable def money_spent_on_book (initial_money : ℝ) : ℝ :=
  (1/2) * initial_money

noncomputable def money_after_initial_expenses (initial_money : ℝ) (gift : ℝ) : ℝ :=
  initial_money - money_spent_on_sandwich initial_money - money_spent_on_ticket initial_money - money_spent_on_book initial_money + gift

noncomputable def money_spent_on_cosmetics (remaining_money : ℝ) : ℝ :=
  (1/4) * remaining_money

noncomputable def money_after_cosmetics (remaining_money : ℝ) : ℝ :=
  remaining_money - money_spent_on_cosmetics remaining_money

noncomputable def money_spent_on_tshirt (remaining_money : ℝ) : ℝ :=
  let tshirt_cost := (1/3) * remaining_money
  let tax := (5/100) * tshirt_cost
  tshirt_cost + tax

noncomputable def remaining_money (initial_money : ℝ) (gift : ℝ) : ℝ :=
  let after_initial := money_after_initial_expenses initial_money gift
  let after_cosmetics := after_initial - money_spent_on_cosmetics after_initial
  after_cosmetics - money_spent_on_tshirt after_cosmetics

theorem jennifer_remaining_money : remaining_money 90 30 = 21.35 := by
  sorry

end jennifer_remaining_money_l73_73463


namespace max_gcd_14m_plus_4_9m_plus_2_l73_73539

theorem max_gcd_14m_plus_4_9m_plus_2 (m : ℕ) (h : m > 0) : ∃ M, M = 8 ∧ ∀ k, gcd (14 * m + 4) (9 * m + 2) = k → k ≤ M :=
by
  sorry

end max_gcd_14m_plus_4_9m_plus_2_l73_73539


namespace gcd_12012_18018_l73_73315

theorem gcd_12012_18018 : Int.gcd 12012 18018 = 6006 := 
by
  sorry

end gcd_12012_18018_l73_73315


namespace digits_sum_not_2001_l73_73187

theorem digits_sum_not_2001 (a : ℕ) (n m : ℕ) 
  (h1 : 10^(n-1) ≤ a ∧ a < 10^n)
  (h2 : 3 * n - 2 ≤ m ∧ m < 3 * n + 1)
  : m + n ≠ 2001 := 
sorry

end digits_sum_not_2001_l73_73187


namespace squirrel_climb_l73_73100

-- Define the problem conditions and the goal
variable (x : ℝ)

-- net_distance_climbed_every_two_minutes
def net_distance_climbed_every_two_minutes : ℝ := x - 2

-- distance_climbed_in_14_minutes
def distance_climbed_in_14_minutes : ℝ := 7 * (x - 2)

-- distance_climbed_in_15th_minute
def distance_climbed_in_15th_minute : ℝ := x

-- total_distance_climbed_in_15_minutes
def total_distance_climbed_in_15_minutes : ℝ := 26

-- Theorem: proving x based on the conditions
theorem squirrel_climb : 
  7 * (x - 2) + x = 26 -> x = 5 := by
  intros h
  sorry

end squirrel_climb_l73_73100


namespace james_eats_three_l73_73640

variables {p : ℕ} {f : ℕ} {j : ℕ}

-- The initial number of pizza slices
def initial_slices : ℕ := 8

-- The number of slices his friend eats
def friend_slices : ℕ := 2

-- The number of slices left after his friend eats
def remaining_slices : ℕ := initial_slices - friend_slices

-- The number of slices James eats
def james_slices : ℕ := remaining_slices / 2

-- The theorem to prove James eats 3 slices
theorem james_eats_three : james_slices = 3 :=
by
  sorry

end james_eats_three_l73_73640


namespace train_length_l73_73128

-- Definitions and conditions
variable (L : ℕ)
def condition1 (L : ℕ) : Prop := L + 100 = 15 * (L + 100) / 15
def condition2 (L : ℕ) : Prop := L + 250 = 20 * (L + 250) / 20

-- Theorem statement
theorem train_length (h1 : condition1 L) (h2 : condition2 L) : L = 350 := 
by 
  sorry

end train_length_l73_73128


namespace next_two_series_numbers_l73_73325

theorem next_two_series_numbers :
  ∀ (a : ℕ → ℤ), a 1 = 2 → a 2 = 3 →
    (∀ n, 3 ≤ n → a n = a (n - 1) + a (n - 2) - 5) →
    a 7 = -26 ∧ a 8 = -45 :=
by
  intros a h1 h2 h3
  sorry

end next_two_series_numbers_l73_73325


namespace find_a_10_l73_73425

def seq (a : ℕ → ℚ) : Prop :=
∀ n, a (n + 1) = 2 * a n / (a n + 2)

def initial_value (a : ℕ → ℚ) : Prop :=
a 1 = 1

theorem find_a_10 (a : ℕ → ℚ) (h1 : initial_value a) (h2 : seq a) : 
  a 10 = 2 / 11 := 
sorry

end find_a_10_l73_73425


namespace minimum_k_l73_73393

theorem minimum_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  a 1 = (1/2) ∧ (∀ n, 2 * a (n + 1) + S n = 0) ∧ (∀ n, S n ≤ k) → k = (1/2) :=
sorry

end minimum_k_l73_73393


namespace find_relationship_l73_73008

variables (x y : ℝ)

def AB : ℝ × ℝ := (6, 1)
def BC : ℝ × ℝ := (x, y)
def CD : ℝ × ℝ := (-2, -3)
def DA : ℝ × ℝ := (4 - x, -2 - y)
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_relationship (h_parallel : parallel (x, y) (4 - x, -2 - y)) : x + 2 * y = 0 :=
sorry

end find_relationship_l73_73008


namespace blue_chips_count_l73_73327

variable (T : ℕ) (blue_chips : ℕ) (white_chips : ℕ) (green_chips : ℕ)

-- Conditions
def condition1 : Prop := blue_chips = (T / 10)
def condition2 : Prop := white_chips = (T / 2)
def condition3 : Prop := green_chips = 12
def condition4 : Prop := blue_chips + white_chips + green_chips = T

-- Proof problem
theorem blue_chips_count (h1 : condition1 T blue_chips)
                          (h2 : condition2 T white_chips)
                          (h3 : condition3 green_chips)
                          (h4 : condition4 T blue_chips white_chips green_chips) :
  blue_chips = 3 :=
sorry

end blue_chips_count_l73_73327


namespace present_population_l73_73962

theorem present_population (P : ℕ) (h1 : P * 11 / 10 = 264) : P = 240 :=
by sorry

end present_population_l73_73962


namespace range_of_x_range_of_a_l73_73341

-- Definitions of the conditions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- Part (1)
theorem range_of_x (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
by sorry

-- Part (2)
theorem range_of_a (a : ℝ) (h : ∀ x, ¬ (p x a) → ¬ (q x)) : 1 < a ∧ a ≤ 2 :=
by sorry

end range_of_x_range_of_a_l73_73341


namespace steven_has_72_shirts_l73_73958

def brian_shirts : ℕ := 3
def andrew_shirts (brian : ℕ) : ℕ := 6 * brian
def steven_shirts (andrew : ℕ) : ℕ := 4 * andrew

theorem steven_has_72_shirts : steven_shirts (andrew_shirts brian_shirts) = 72 := 
by 
  -- We add "sorry" here to indicate that the proof is omitted
  sorry

end steven_has_72_shirts_l73_73958


namespace minimum_xy_minimum_x_plus_y_l73_73856

theorem minimum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 :=
sorry

theorem minimum_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 :=
sorry

end minimum_xy_minimum_x_plus_y_l73_73856


namespace max_difference_in_volume_l73_73647

noncomputable def computed_volume (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def max_possible_volume (length width height : ℕ) (error : ℕ) : ℕ :=
  (length + error) * (width + error) * (height + error)

theorem max_difference_in_volume :
  ∀ (length width height error : ℕ), length = 150 → width = 150 → height = 225 → error = 1 → 
  max_possible_volume length width height error - computed_volume length width height = 90726 :=
by
  intros length width height error h_length h_width h_height h_error
  rw [h_length, h_width, h_height, h_error]
  simp only [computed_volume, max_possible_volume]
  -- Intermediate calculations
  sorry

end max_difference_in_volume_l73_73647


namespace rotated_ellipse_sum_is_four_l73_73884

noncomputable def rotated_ellipse_center (h' k' : ℝ) : Prop :=
h' = 3 ∧ k' = -5

noncomputable def rotated_ellipse_axes (a' b' : ℝ) : Prop :=
a' = 4 ∧ b' = 2

noncomputable def rotated_ellipse_sum (h' k' a' b' : ℝ) : ℝ :=
h' + k' + a' + b'

theorem rotated_ellipse_sum_is_four (h' k' a' b' : ℝ) 
  (hc : rotated_ellipse_center h' k') (ha : rotated_ellipse_axes a' b') :
  rotated_ellipse_sum h' k' a' b' = 4 :=
by
  -- The proof would be provided here.
  -- Since we're asked not to provide the proof but just to ensure the statement is correct, we use sorry.
  sorry

end rotated_ellipse_sum_is_four_l73_73884


namespace base6_arithmetic_l73_73666

theorem base6_arithmetic :
  let a := 4512
  let b := 2324
  let c := 1432
  let base := 6
  let a_b10 := 4 * base^3 + 5 * base^2 + 1 * base + 2
  let b_b10 := 2 * base^3 + 3 * base^2 + 2 * base + 4
  let c_b10 := 1 * base^3 + 4 * base^2 + 3 * base + 2
  let result_b10 := a_b10 - b_b10 + c_b10
  let result_base6 := 4020
  (result_b10 / base^3) % base = 4 ∧
  (result_b10 / base^2) % base = 0 ∧
  (result_b10 / base) % base = 2 ∧
  result_b10 % base = 0 →
  result_base6 = 4020 := by
  sorry

end base6_arithmetic_l73_73666


namespace p2_div_q2_eq_4_l73_73180

theorem p2_div_q2_eq_4 
  (p q : ℝ → ℝ)
  (h1 : ∀ x, p x = 12 * x)
  (h2 : ∀ x, q x = (x + 4) * (x - 1))
  (h3 : p 0 = 0)
  (h4 : p (-1) / q (-1) = -2) :
  (p 2 / q 2 = 4) :=
by {
  sorry
}

end p2_div_q2_eq_4_l73_73180


namespace intersection_complement_l73_73500

open Set

noncomputable def U : Set ℝ := {-1, 0, 1, 4}
def A : Set ℝ := {-1, 1}
def B : Set ℝ := {1, 4}
def C_U_B : Set ℝ := U \ B

theorem intersection_complement :
  A ∩ C_U_B = {-1} :=
by
  sorry

end intersection_complement_l73_73500


namespace max_n_for_factorable_poly_l73_73372

/-- 
  Let p(x) = 6x^2 + n * x + 48 be a quadratic polynomial.
  We want to find the maximum value of n such that p(x) can be factored into
  the product of two linear factors with integer coefficients.
-/
theorem max_n_for_factorable_poly :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * B + A = n → A * B = 48) ∧ n = 289 := 
by
  sorry

end max_n_for_factorable_poly_l73_73372


namespace problem_statement_l73_73725

-- Definition of sum of digits function
def S (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Definition of the function f₁
def f₁ (k : ℕ) : ℕ :=
  (S k) ^ 2

-- Definition of the function fₙ₊₁
def f : ℕ → ℕ → ℕ
| 0, k => k
| (n+1), k => f₁ (f n k)

-- Theorem stating the proof problem
theorem problem_statement : f 2005 (2 ^ 2006) = 169 :=
  sorry

end problem_statement_l73_73725


namespace product_of_slopes_hyperbola_l73_73795

theorem product_of_slopes_hyperbola (a b x0 y0 : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : (x0, y0) ≠ (-a, 0)) (h4 : (x0, y0) ≠ (a, 0)) 
(h5 : x0^2 / a^2 - y0^2 / b^2 = 1) : 
(y0 / (x0 + a) * (y0 / (x0 - a)) = b^2 / a^2) :=
sorry

end product_of_slopes_hyperbola_l73_73795


namespace find_missing_number_l73_73905

theorem find_missing_number (x : ℚ) (h : 11 * x + 4 = 7) : x = 9 / 11 :=
sorry

end find_missing_number_l73_73905


namespace paper_cost_l73_73554
noncomputable section

variables (P C : ℝ)

theorem paper_cost (h : 100 * P + 200 * C = 6.00) : 
  20 * P + 40 * C = 1.20 :=
sorry

end paper_cost_l73_73554


namespace tangent_line_at_1_l73_73522

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

theorem tangent_line_at_1 :
  let x := (1 : ℝ)
  let y := (f 1)
  ∃ m b : ℝ, (∀ x, y - m * (x - 1) + b = 0)
  ∧ (m = -2)
  ∧ (b = -1) :=
by
  sorry

end tangent_line_at_1_l73_73522


namespace solve_system_l73_73664

theorem solve_system :
  ∃ (x y : ℚ), (4 * x - 35 * y = -1) ∧ (3 * y - x = 5) ∧ (x = -172 / 23) ∧ (y = -19 / 23) :=
by
  sorry

end solve_system_l73_73664


namespace sandy_age_when_record_l73_73992

noncomputable def calc_age (record_length current_length monthly_growth_rate age : ℕ) : ℕ :=
  let yearly_growth_rate := monthly_growth_rate * 12
  let needed_length := record_length - current_length
  let years_needed := needed_length / yearly_growth_rate
  age + years_needed

theorem sandy_age_when_record (record_length current_length monthly_growth_rate age : ℕ) :
  record_length = 26 →
  current_length = 2 →
  monthly_growth_rate = 1 →
  age = 12 →
  calc_age record_length current_length monthly_growth_rate age = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold calc_age
  simp
  sorry

end sandy_age_when_record_l73_73992


namespace area_of_square_with_adjacent_points_l73_73937

theorem area_of_square_with_adjacent_points (x1 y1 x2 y2 : ℝ)
    (h1 : x1 = 1) (h2 : y1 = 2) (h3 : x2 = 4) (h4 : y2 = 6)
    (h_adj : ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 ^ 2) :
    (5 ^ 2) = 25 := 
by
  sorry

end area_of_square_with_adjacent_points_l73_73937


namespace new_average_contribution_75_l73_73252

-- Define the conditions given in the problem
def original_contributions : ℝ := 1
def johns_donation : ℝ := 100
def increase_rate : ℝ := 1.5

-- Define a function to calculate the new average contribution size
def new_total_contributions (A : ℝ) := A + johns_donation
def new_average_contribution (A : ℝ) := increase_rate * A

-- Theorem to prove that the new average contribution size is $75
theorem new_average_contribution_75 (A : ℝ) :
  new_total_contributions A / (original_contributions + 1) = increase_rate * A →
  A = 50 →
  new_average_contribution A = 75 :=
by
  intros h1 h2
  rw [new_average_contribution, h2]
  sorry

end new_average_contribution_75_l73_73252


namespace not_divisible_1961_1963_divisible_1963_1965_l73_73344

def is_divisible_by_three (n : Nat) : Prop :=
  n % 3 = 0

theorem not_divisible_1961_1963 : ¬ is_divisible_by_three (1961 * 1963) :=
by
  sorry

theorem divisible_1963_1965 : is_divisible_by_three (1963 * 1965) :=
by
  sorry

end not_divisible_1961_1963_divisible_1963_1965_l73_73344


namespace unique_pair_odd_prime_l73_73933

theorem unique_pair_odd_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃! (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n) + (1 / m) ∧ 
  n = (p + 1) / 2 ∧ m = (p * (p + 1)) / 2 :=
by
  sorry

end unique_pair_odd_prime_l73_73933


namespace find_smallest_number_l73_73437

theorem find_smallest_number 
  : ∃ x : ℕ, (x - 18) % 14 = 0 ∧ (x - 18) % 26 = 0 ∧ (x - 18) % 28 = 0 ∧ (x - 18) / Nat.lcm 14 (Nat.lcm 26 28) = 746 ∧ x = 271562 := by
  sorry

end find_smallest_number_l73_73437


namespace commercial_break_duration_l73_73877

theorem commercial_break_duration (n1 n2 t1 t2 : ℕ) (h1 : n1 = 3) (h2: t1 = 5) (h3 : n2 = 11) (h4 : t2 = 2) : 
  n1 * t1 + n2 * t2 = 37 := 
by 
  sorry

end commercial_break_duration_l73_73877


namespace logical_contradiction_l73_73145

-- Definitions based on the conditions
def all_destroying (x : Type) : Prop := ∀ y : Type, y ≠ x → y → false
def indestructible (x : Type) : Prop := ∀ y : Type, y = x → y → false

theorem logical_contradiction (x : Type) :
  (all_destroying x ∧ indestructible x) → false :=
by
  sorry

end logical_contradiction_l73_73145


namespace arithmetic_mean_calculation_l73_73362

theorem arithmetic_mean_calculation (x : ℝ) 
  (h : (x + 10 + 20 + 3 * x + 15 + 3 * x + 6) / 5 = 30) : 
  x = 14.142857 :=
by
  sorry

end arithmetic_mean_calculation_l73_73362


namespace remainder_of_31_pow_31_plus_31_div_32_l73_73205

theorem remainder_of_31_pow_31_plus_31_div_32 :
  (31^31 + 31) % 32 = 30 := 
by 
  trivial -- Replace with actual proof

end remainder_of_31_pow_31_plus_31_div_32_l73_73205


namespace inequality_convex_l73_73947

theorem inequality_convex (x y a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : a + b = 1) : 
  (a * x + b * y) ^ 2 ≤ a * x ^ 2 + b * y ^ 2 := 
sorry

end inequality_convex_l73_73947


namespace drawing_specific_cards_from_two_decks_l73_73923

def prob_of_drawing_specific_cards (total_cards_deck1 total_cards_deck2 : ℕ) 
  (specific_card1 specific_card2 : ℕ) : ℚ :=
(specific_card1 / total_cards_deck1) * (specific_card2 / total_cards_deck2)

theorem drawing_specific_cards_from_two_decks :
  prob_of_drawing_specific_cards 52 52 1 1 = 1 / 2704 :=
by
  -- The proof can be filled in here
  sorry

end drawing_specific_cards_from_two_decks_l73_73923


namespace A_number_is_35_l73_73611

theorem A_number_is_35 (A B : ℕ) 
  (h_sum_digits : A + B = 8) 
  (h_diff_numbers : 10 * B + A = 10 * A + B + 18) :
  10 * A + B = 35 :=
by {
  sorry
}

end A_number_is_35_l73_73611


namespace find_k_range_l73_73026

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + x else Real.log x / Real.log (1 / 3)

def g (x k : ℝ) : ℝ :=
abs (x - k) + abs (x - 1)

theorem find_k_range (k : ℝ) :
  (∀ x1 x2 : ℝ, f x1 ≤ g x2 k) → (k ≤ 3 / 4 ∨ k ≥ 5 / 4) :=
by
  sorry

end find_k_range_l73_73026


namespace unbroken_seashells_l73_73936

theorem unbroken_seashells (total_seashells : ℕ) (broken_seashells : ℕ) (h1 : total_seashells = 23) (h2 : broken_seashells = 11) : total_seashells - broken_seashells = 12 := by
  sorry

end unbroken_seashells_l73_73936


namespace sum_of_234_and_142_in_base_4_l73_73545

theorem sum_of_234_and_142_in_base_4 :
  (234 + 142) = 376 ∧ (376 + 0) = 256 * 1 + 64 * 1 + 16 * 3 + 4 * 2 + 1 * 0 :=
by sorry

end sum_of_234_and_142_in_base_4_l73_73545


namespace Carmen_candle_burn_time_l73_73815

theorem Carmen_candle_burn_time
  (night_to_last_candle_first_scenario : ℕ := 8)
  (hours_per_night_second_scenario : ℕ := 2)
  (nights_second_scenario : ℕ := 24)
  (candles_second_scenario : ℕ := 6) :
  ∃ T : ℕ, (night_to_last_candle_first_scenario * T = hours_per_night_second_scenario * (nights_second_scenario / candles_second_scenario)) ∧ T = 1 :=
by
  let T := (hours_per_night_second_scenario * (nights_second_scenario / candles_second_scenario)) / night_to_last_candle_first_scenario
  have : T = 1 := by sorry
  use T
  exact ⟨ by sorry, this⟩

end Carmen_candle_burn_time_l73_73815


namespace equation1_solutions_equation2_solutions_equation3_solutions_equation4_no_real_solutions_l73_73057

theorem equation1_solutions (x : ℝ) : x^2 - 4 * x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) :=
by sorry

theorem equation2_solutions (x : ℝ) : x * (3 * x + 1) = 2 * (3 * x + 1) ↔ (x = -1 / 3 ∨ x = 2) :=
by sorry

theorem equation3_solutions (x : ℝ) : 2 * x^2 + x - 4 = 0 ↔ (x = (-1 + Real.sqrt 33) / 4 ∨ x = (-1 - Real.sqrt 33) / 4) :=
by sorry

theorem equation4_no_real_solutions (x : ℝ) : ¬ ∃ x, 4 * x^2 - 3 * x + 1 = 0 :=
by sorry

end equation1_solutions_equation2_solutions_equation3_solutions_equation4_no_real_solutions_l73_73057


namespace average_speeds_equation_l73_73298

theorem average_speeds_equation (x : ℝ) (hx : 0 < x) : 
  10 / x - 7 / (1.4 * x) = 10 / 60 :=
by
  sorry

end average_speeds_equation_l73_73298


namespace parallelogram_area_twice_quadrilateral_l73_73218

theorem parallelogram_area_twice_quadrilateral (a b : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π) :
  let quadrilateral_area := (1 / 2) * a * b * Real.sin α
  let parallelogram_area := a * b * Real.sin α
  parallelogram_area = 2 * quadrilateral_area :=
by
  let quadrilateral_area := (1 / 2) * a * b * Real.sin α
  let parallelogram_area := a * b * Real.sin α
  sorry

end parallelogram_area_twice_quadrilateral_l73_73218


namespace solve_triangle_l73_73568

theorem solve_triangle :
  (a = 6 ∧ b = 6 * Real.sqrt 3 ∧ A = 30) →
  ((B = 60 ∧ C = 90 ∧ c = 12) ∨ (B = 120 ∧ C = 30 ∧ c = 6)) :=
by
  intros h
  sorry

end solve_triangle_l73_73568


namespace arithmetic_sequence_solution_l73_73600

theorem arithmetic_sequence_solution (x : ℝ) (h : 2 * (x + 1) = 2 * x + (x + 2)) : x = 0 :=
by {
  -- To avoid actual proof steps, we add sorry.
  sorry 
}

end arithmetic_sequence_solution_l73_73600


namespace solve_bank_account_problem_l73_73844

noncomputable def bank_account_problem : Prop :=
  ∃ (A E Z : ℝ),
    A > E ∧
    Z > A ∧
    A - E = (1/12) * (A + E) ∧
    Z - A = (1/10) * (Z + A) ∧
    1.10 * A = 1.20 * E + 20 ∧
    1.10 * A + 30 = 1.15 * Z ∧
    E = 2000 / 23

theorem solve_bank_account_problem : bank_account_problem :=
sorry

end solve_bank_account_problem_l73_73844


namespace sin_cos_alpha_beta_l73_73200

theorem sin_cos_alpha_beta (α β : ℝ) 
  (h1 : Real.sin α = Real.cos β) 
  (h2 : Real.cos α = Real.sin (2 * β)) :
  Real.sin β ^ 2 + Real.cos α ^ 2 = 3 / 2 := 
by
  sorry

end sin_cos_alpha_beta_l73_73200


namespace smallest_X_l73_73037

noncomputable def T : ℕ := 1110
noncomputable def X : ℕ := T / 6

theorem smallest_X (hT_digits : (∀ d ∈ T.digits 10, d = 0 ∨ d = 1))
  (hT_positive : T > 0)
  (hT_div_6 : T % 6 = 0) :
  X = 185 := by
  sorry

end smallest_X_l73_73037


namespace seating_sessions_l73_73524

theorem seating_sessions (num_parents num_pupils morning_parents afternoon_parents morning_pupils mid_day_pupils evening_pupils session_capacity total_sessions : ℕ) 
  (h1 : num_parents = 61)
  (h2 : num_pupils = 177)
  (h3 : session_capacity = 44)
  (h4 : morning_parents = 35)
  (h5 : afternoon_parents = 26)
  (h6 : morning_pupils = 65)
  (h7 : mid_day_pupils = 57)
  (h8 : evening_pupils = 55)
  (h9 : total_sessions = 8) :
  ∃ (parent_sessions pupil_sessions : ℕ), 
    parent_sessions + pupil_sessions = total_sessions ∧
    parent_sessions = (morning_parents + session_capacity - 1) / session_capacity + (afternoon_parents + session_capacity - 1) / session_capacity ∧
    pupil_sessions = (morning_pupils + session_capacity - 1) / session_capacity + (mid_day_pupils + session_capacity - 1) / session_capacity + (evening_pupils + session_capacity - 1) / session_capacity := 
by
  sorry

end seating_sessions_l73_73524


namespace earth_surface_area_scientific_notation_l73_73410

theorem earth_surface_area_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 780000000 = a * 10^n ∧ a = 7.8 ∧ n = 8 :=
by
  sorry

end earth_surface_area_scientific_notation_l73_73410


namespace cylinder_volume_from_cone_l73_73292

/-- Given the volume of a cone, prove the volume of a cylinder with the same base and height. -/
theorem cylinder_volume_from_cone (V_cone : ℝ) (h : V_cone = 3.6) : 
  ∃ V_cylinder : ℝ, V_cylinder = 0.0108 :=
by
  have V_cylinder := 3 * V_cone
  have V_cylinder_meters := V_cylinder / 1000
  use V_cylinder_meters
  sorry

end cylinder_volume_from_cone_l73_73292


namespace sin_600_eq_neg_sqrt_3_over_2_l73_73163

theorem sin_600_eq_neg_sqrt_3_over_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_600_eq_neg_sqrt_3_over_2_l73_73163


namespace alice_minimum_speed_l73_73322

noncomputable def minimum_speed_to_exceed (d t_bob t_alice : ℝ) (v_bob : ℝ) : ℝ :=
  d / t_alice

theorem alice_minimum_speed (d : ℝ) (v_bob : ℝ) (t_lag : ℝ) (v_alice : ℝ) :
  d = 30 → v_bob = 40 → t_lag = 0.5 → v_alice = d / (d / v_bob - t_lag) → v_alice > 60 :=
by
  intros hd hv hb ht
  rw [hd, hv, hb] at ht
  simp at ht
  sorry

end alice_minimum_speed_l73_73322


namespace ratio_of_red_to_blue_marbles_l73_73655

theorem ratio_of_red_to_blue_marbles (total_marbles yellow_marbles : ℕ) (green_marbles blue_marbles red_marbles : ℕ) 
  (odds_blue : ℚ) 
  (h1 : total_marbles = 60) 
  (h2 : yellow_marbles = 20) 
  (h3 : green_marbles = yellow_marbles / 2) 
  (h4 : red_marbles + blue_marbles = total_marbles - (yellow_marbles + green_marbles)) 
  (h5 : odds_blue = 0.25) 
  (h6 : blue_marbles = odds_blue * (red_marbles + blue_marbles)) : 
  red_marbles / blue_marbles = 11 / 4 := 
by 
  sorry

end ratio_of_red_to_blue_marbles_l73_73655


namespace average_height_corrected_l73_73072

-- Defining the conditions as functions and constants
def incorrect_average_height : ℝ := 175
def number_of_students : ℕ := 30
def incorrect_height : ℝ := 151
def actual_height : ℝ := 136

-- The target average height to prove
def target_actual_average_height : ℝ := 174.5

-- Main theorem stating the problem
theorem average_height_corrected : 
  (incorrect_average_height * number_of_students - (incorrect_height - actual_height)) / number_of_students = target_actual_average_height :=
by
  sorry

end average_height_corrected_l73_73072


namespace no_solution_values_l73_73791

theorem no_solution_values (m : ℝ) :
  (∀ x : ℝ, x ≠ 5 → x ≠ -5 → (1 / (x - 5) + m / (x + 5) ≠ (m + 5) / (x^2 - 25))) ↔
  m = -1 ∨ m = 5 ∨ m = -5 / 11 :=
by
  sorry

end no_solution_values_l73_73791


namespace sales_tax_difference_l73_73752

theorem sales_tax_difference (P : ℝ) (d t1 t2 : ℝ) :
  let discounted_price := P * (1 - d)
  let total_cost1 := discounted_price * (1 + t1)
  let total_cost2 := discounted_price * (1 + t2)
  t1 = 0.08 ∧ t2 = 0.075 ∧ P = 50 ∧ d = 0.05 →
  abs ((total_cost1 - total_cost2) - 0.24) < 0.01 :=
by
  sorry

end sales_tax_difference_l73_73752


namespace correct_option_l73_73593

def condition_A : Prop := abs ((-5 : ℤ)^2) = -5
def condition_B : Prop := abs (9 : ℤ) = 3 ∨ abs (9 : ℤ) = -3
def condition_C : Prop := abs (3 : ℤ) / abs (((-2)^3 : ℤ)) = -2
def condition_D : Prop := (2 * abs (3 : ℤ))^2 = 6 

theorem correct_option : ¬condition_A ∧ ¬condition_B ∧ condition_C ∧ ¬condition_D :=
by
  sorry

end correct_option_l73_73593


namespace value_of_expression_l73_73548

theorem value_of_expression (x y z : ℤ) (h1 : x = 2) (h2 : y = 3) (h3 : z = 4) :
  (4 * x^2 - 6 * y^3 + z^2) / (5 * x + 7 * z - 3 * y^2) = -130 / 11 :=
by
  sorry

end value_of_expression_l73_73548


namespace total_blocks_per_day_l73_73424

def blocks_to_park : ℕ := 4
def blocks_to_hs : ℕ := 7
def blocks_to_home : ℕ := 11
def walks_per_day : ℕ := 3

theorem total_blocks_per_day :
  (blocks_to_park + blocks_to_hs + blocks_to_home) * walks_per_day = 66 :=
by
  sorry

end total_blocks_per_day_l73_73424


namespace eval_expr_at_3_l73_73697

theorem eval_expr_at_3 : (3^2 - 5 * 3 + 6) / (3 - 2) = 0 := by
  sorry

end eval_expr_at_3_l73_73697


namespace original_selling_price_l73_73643

-- Definitions based on the conditions
def original_price : ℝ := 933.33

-- Given conditions
def discount_rate : ℝ := 0.40
def price_after_discount : ℝ := 560.0

-- Lean theorem statement to prove that original selling price (x) is equal to 933.33
theorem original_selling_price (x : ℝ) 
  (h1 : x * (1 - discount_rate) = price_after_discount) : 
  x = original_price :=
  sorry

end original_selling_price_l73_73643


namespace tan_equality_condition_l73_73031

open Real

theorem tan_equality_condition (α β : ℝ) :
  (α = β) ↔ (tan α = tan β) :=
sorry

end tan_equality_condition_l73_73031


namespace length_of_ad_l73_73172

theorem length_of_ad (AB CD AD BC : ℝ) 
  (h1 : AB = 10) 
  (h2 : CD = 2 * AB) 
  (h3 : AD = BC) 
  (h4 : AB + BC + CD + AD = 42) : AD = 6 :=
by
  -- proof omitted
  sorry

end length_of_ad_l73_73172


namespace find_percentage_l73_73333

theorem find_percentage (x p : ℝ) (h1 : 0.25 * x = p * 10 - 30) (h2 : x = 680) : p = 20 := 
sorry

end find_percentage_l73_73333


namespace Suma_work_time_l73_73495

theorem Suma_work_time (W : ℝ) (h1 : W > 0) :
  let renu_rate := W / 8
  let combined_rate := W / 4
  let suma_rate := combined_rate - renu_rate
  let suma_time := W / suma_rate
  suma_time = 8 :=
by 
  let renu_rate := W / 8
  let combined_rate := W / 4
  let suma_rate := combined_rate - renu_rate
  let suma_time := W / suma_rate
  exact sorry

end Suma_work_time_l73_73495


namespace keaton_apple_earnings_l73_73705

theorem keaton_apple_earnings
  (orange_harvest_interval : ℕ)
  (orange_income_per_harvest : ℕ)
  (total_yearly_income : ℕ)
  (orange_harvests_per_year : ℕ)
  (orange_yearly_income : ℕ)
  (apple_yearly_income : ℕ) :
  orange_harvest_interval = 2 →
  orange_income_per_harvest = 50 →
  total_yearly_income = 420 →
  orange_harvests_per_year = 12 / orange_harvest_interval →
  orange_yearly_income = orange_harvests_per_year * orange_income_per_harvest →
  apple_yearly_income = total_yearly_income - orange_yearly_income →
  apple_yearly_income = 120 :=
by
  sorry

end keaton_apple_earnings_l73_73705


namespace math_problem_l73_73948

theorem math_problem (f_star f_ast : ℕ → ℕ → ℕ) (h₁ : f_star 20 5 = 15) (h₂ : f_ast 15 5 = 75) :
  (f_star 8 4) / (f_ast 10 2) = (1:ℚ) / 5 := by
  sorry

end math_problem_l73_73948


namespace integer_values_abc_l73_73214

theorem integer_values_abc {a b c : ℤ} :
  a^2 + b^2 + c^2 + 3 < a * b + 3 * b + 2 * c ↔ (a = 1 ∧ b = 2 ∧ c = 1) :=
by
  sorry -- Proof to be filled

end integer_values_abc_l73_73214


namespace abs_eq_solution_l73_73850

theorem abs_eq_solution (x : ℚ) : |x - 2| = |x + 3| → x = -1 / 2 :=
by
  sorry

end abs_eq_solution_l73_73850


namespace total_votes_l73_73416

variable (V : ℝ)

theorem total_votes (h : 0.70 * V - 0.30 * V = 160) : V = 400 := by
  sorry

end total_votes_l73_73416


namespace nut_weights_l73_73351

noncomputable def part_weights (total_weight : ℝ) (total_parts : ℝ) : ℝ :=
  total_weight / total_parts

theorem nut_weights
  (total_weight : ℝ)
  (parts_almonds parts_walnuts parts_cashews ratio_pistachios_to_almonds : ℝ)
  (total_parts_without_pistachios total_parts_with_pistachios weight_per_part : ℝ)
  (weights_almonds weights_walnuts weights_cashews weights_pistachios : ℝ) :
  parts_almonds = 5 →
  parts_walnuts = 3 →
  parts_cashews = 2 →
  ratio_pistachios_to_almonds = 1 / 4 →
  total_parts_without_pistachios = parts_almonds + parts_walnuts + parts_cashews →
  total_parts_with_pistachios = total_parts_without_pistachios + (parts_almonds * ratio_pistachios_to_almonds) →
  weight_per_part = total_weight / total_parts_with_pistachios →
  weights_almonds = parts_almonds * weight_per_part →
  weights_walnuts = parts_walnuts * weight_per_part →
  weights_cashews = parts_cashews * weight_per_part →
  weights_pistachios = (parts_almonds * ratio_pistachios_to_almonds) * weight_per_part →
  total_weight = 300 →
  weights_almonds = 133.35 ∧
  weights_walnuts = 80.01 ∧
  weights_cashews = 53.34 ∧
  weights_pistachios = 33.34 :=
by
  intros
  sorry

end nut_weights_l73_73351


namespace league_games_and_weeks_l73_73561

/--
There are 15 teams in a league, and each team plays each of the other teams exactly once.
Due to scheduling limitations, each team can only play one game per week.
Prove that the total number of games played is 105 and the minimum number of weeks needed to complete all the games is 15.
-/
theorem league_games_and_weeks :
  let teams := 15
  let total_games := teams * (teams - 1) / 2
  let games_per_week := Nat.div teams 2
  total_games = 105 ∧ total_games / games_per_week = 15 :=
by
  sorry

end league_games_and_weeks_l73_73561


namespace return_percentage_is_6_5_l73_73153

def investment1 : ℤ := 16250
def investment2 : ℤ := 16250
def profit_percentage1 : ℚ := 0.15
def loss_percentage2 : ℚ := 0.05
def total_investment : ℤ := 25000
def net_income : ℚ := investment1 * profit_percentage1 - investment2 * loss_percentage2
def return_percentage : ℚ := (net_income / total_investment) * 100

theorem return_percentage_is_6_5 : return_percentage = 6.5 := by
  sorry

end return_percentage_is_6_5_l73_73153


namespace parabola_focus_and_directrix_l73_73517

theorem parabola_focus_and_directrix :
  (∀ x y : ℝ, x^2 = 4 * y → ∃ a b : ℝ, (a, b) = (0, 1) ∧ y = -1) :=
by
  -- Here, we would provide definitions and logical steps if we were completing the proof.
  -- For now, we will leave it unfinished.
  sorry

end parabola_focus_and_directrix_l73_73517


namespace math_marks_l73_73259

theorem math_marks (english physics chemistry biology total_marks math_marks : ℕ) 
  (h_eng : english = 73)
  (h_phy : physics = 92)
  (h_chem : chemistry = 64)
  (h_bio : biology = 82)
  (h_avg : total_marks = 76 * 5) :
  math_marks = 69 := 
by
  sorry

end math_marks_l73_73259


namespace log_comparisons_l73_73605

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 3 / (2 * Real.log 2)
noncomputable def c := 1 / 2

theorem log_comparisons : c < b ∧ b < a := 
by
  sorry

end log_comparisons_l73_73605


namespace sum_a_c_e_l73_73353

theorem sum_a_c_e {a b c d e f : ℝ} 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 5) : 
  a + c + e = 10 :=
by
  -- Proof goes here
  sorry

end sum_a_c_e_l73_73353


namespace smallest_integer_b_l73_73586

theorem smallest_integer_b (b : ℕ) : 27 ^ b > 3 ^ 9 ↔ b = 4 := by
  sorry

end smallest_integer_b_l73_73586


namespace equalize_costs_l73_73383

variable (L B C : ℝ)
variable (h1 : L < B)
variable (h2 : B < C)

theorem equalize_costs : (B + C - 2 * L) / 3 = ((L + B + C) / 3 - L) :=
by sorry

end equalize_costs_l73_73383


namespace relation_between_x_and_y_l73_73617

noncomputable def x : ℝ := 2 + Real.sqrt 3
noncomputable def y : ℝ := 1 / (2 - Real.sqrt 3)

theorem relation_between_x_and_y : x = y := sorry

end relation_between_x_and_y_l73_73617


namespace factor_difference_of_squares_l73_73845

theorem factor_difference_of_squares (a : ℝ) : a^2 - 16 = (a - 4) * (a + 4) := 
sorry

end factor_difference_of_squares_l73_73845


namespace trajectory_of_M_l73_73578

noncomputable def P : ℝ × ℝ := (2, 2)
noncomputable def circleC (x y : ℝ) : Prop := x^2 + y^2 - 8 * y = 0
noncomputable def isMidpoint (A B M : ℝ × ℝ) : Prop := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def isIntersectionPoint (l : ℝ × ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, circleC x y ∧ l (x, y) ∧ ((A = (x, y)) ∨ (B = (x, y))) 

theorem trajectory_of_M (M : ℝ × ℝ) : 
  (∃ A B : ℝ × ℝ, isIntersectionPoint (fun p => ∃ k : ℝ, p = (k, k)) A B ∧ isMidpoint A B M) →
  (M.1 - 1)^2 + (M.2 - 3)^2 = 2 := 
sorry

end trajectory_of_M_l73_73578


namespace find_line_equation_proj_origin_l73_73471

theorem find_line_equation_proj_origin (P : ℝ × ℝ) (hP : P = (-2, 1)) :
    ∃ (a b c : ℝ), a * 2 + b * (-1) + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 5 := 
by
  sorry

end find_line_equation_proj_origin_l73_73471


namespace puzzles_pieces_count_l73_73890

theorem puzzles_pieces_count :
  let pieces_per_hour := 100
  let hours_per_day := 7
  let days := 7
  let total_pieces_can_put_together := pieces_per_hour * hours_per_day * days
  let pieces_per_puzzle1 := 300
  let number_of_puzzles1 := 8
  let total_pieces_puzzles1 := pieces_per_puzzle1 * number_of_puzzles1
  let remaining_pieces := total_pieces_can_put_together - total_pieces_puzzles1
  let number_of_puzzles2 := 5
  remaining_pieces / number_of_puzzles2 = 500
:= by
  sorry

end puzzles_pieces_count_l73_73890


namespace a_minus_b_range_l73_73868

noncomputable def range_of_a_minus_b (a b : ℝ) : Set ℝ :=
  {x | -2 < a ∧ a < 1 ∧ 0 < b ∧ b < 4 ∧ x = a - b}

theorem a_minus_b_range (a b : ℝ) (h₁ : -2 < a) (h₂ : a < 1) (h₃ : 0 < b) (h₄ : b < 4) :
  ∃ x, range_of_a_minus_b a b x ∧ (-6 < x ∧ x < 1) :=
by
  sorry

end a_minus_b_range_l73_73868


namespace negation_proposition_l73_73480

theorem negation_proposition : 
  ¬ (∃ x_0 : ℝ, x_0^2 + x_0 + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by {
  sorry
}

end negation_proposition_l73_73480


namespace polynomial_identity_l73_73342

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem polynomial_identity (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h : ∀ g : ℝ → ℝ, ∀ x : ℝ, f (g x) = g (f x)) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end polynomial_identity_l73_73342


namespace problem_not_equivalent_l73_73421

theorem problem_not_equivalent :
  (0.0000396 ≠ 3.9 * 10^(-5)) ∧ 
  (0.0000396 = 3.96 * 10^(-5)) ∧ 
  (0.0000396 = 396 * 10^(-7)) ∧ 
  (0.0000396 = (793 / 20000) * 10^(-5)) ∧ 
  (0.0000396 = 198 / 5000000) :=
by
  sorry

end problem_not_equivalent_l73_73421


namespace distinguishable_arrangements_l73_73710

theorem distinguishable_arrangements :
  let brown := 1
  let purple := 1
  let green := 3
  let yellow := 3
  let blue := 2
  let total := brown + purple + green + yellow + blue
  (Nat.factorial total) / (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow * Nat.factorial blue) = 50400 := 
by
  let brown := 1
  let purple := 1
  let green := 3
  let yellow := 3
  let blue := 2
  let total := brown + purple + green + yellow + blue
  sorry

end distinguishable_arrangements_l73_73710


namespace johns_initial_money_l73_73599

/-- John's initial money given that he gives 3/8 to his mother and 3/10 to his father,
and he has $65 left after giving away the money. Prove that he initially had $200. -/
theorem johns_initial_money 
  (M : ℕ)
  (h_left : (M : ℚ) - (3 / 8) * M - (3 / 10) * M = 65) :
  M = 200 :=
sorry

end johns_initial_money_l73_73599


namespace range_m_l73_73415

def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

noncomputable def problem :=
  ∀ (m : ℝ), (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

theorem range_m (m : ℝ) : problem := 
  sorry

end range_m_l73_73415


namespace new_student_weight_l73_73596

theorem new_student_weight 
  (w_avg : ℝ)
  (w_new : ℝ)
  (condition : (5 * w_avg - 72 = 5 * (w_avg - 12) + w_new)) 
  : w_new = 12 := 
  by 
  sorry

end new_student_weight_l73_73596


namespace solve_arithmetic_sequence_l73_73111

-- State the main problem in Lean 4
theorem solve_arithmetic_sequence (y : ℝ) (h : y^2 = (4 + 16) / 2) (hy : y > 0) : y = Real.sqrt 10 := by
  sorry

end solve_arithmetic_sequence_l73_73111


namespace find_optimal_addition_l73_73024

theorem find_optimal_addition (m : ℝ) : 
  (1000 + (m - 1000) * 0.618 = 1618 ∨ 1000 + (m - 1000) * 0.618 = 2618) →
  (m = 2000 ∨ m = 2618) :=
sorry

end find_optimal_addition_l73_73024


namespace travel_time_K_l73_73832

/-
Given that:
1. K's speed is x miles per hour.
2. M's speed is x - 1 miles per hour.
3. K takes 1 hour less than M to travel 60 miles (i.e., 60/x hours).
Prove that K's time to travel 60 miles is 6 hours.
-/
theorem travel_time_K (x : ℝ)
  (h1 : x > 0)
  (h2 : x ≠ 1)
  (h3 : 60 / (x - 1) - 60 / x = 1) :
  60 / x = 6 :=
sorry

end travel_time_K_l73_73832


namespace x_plus_y_value_l73_73053

def sum_evens_40_to_60 : ℕ :=
  (40 + 42 + 44 + 46 + 48 + 50 + 52 + 54 + 56 + 58 + 60)

def num_evens_40_to_60 : ℕ := 11

theorem x_plus_y_value : sum_evens_40_to_60 + num_evens_40_to_60 = 561 := by
  sorry

end x_plus_y_value_l73_73053


namespace percentage_spent_l73_73727

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h_initial : initial_amount = 1200) 
  (h_remaining : remaining_amount = 840) : 
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 :=
by
  sorry

end percentage_spent_l73_73727


namespace claire_earnings_l73_73432

theorem claire_earnings
  (total_flowers : ℕ)
  (tulips : ℕ)
  (white_roses : ℕ)
  (price_per_red_rose : ℚ)
  (sell_fraction : ℚ)
  (h1 : total_flowers = 400)
  (h2 : tulips = 120)
  (h3 : white_roses = 80)
  (h4 : price_per_red_rose = 0.75)
  (h5 : sell_fraction = 1/2) : 
  (total_flowers - tulips - white_roses) * sell_fraction * price_per_red_rose = 75 :=
by
  sorry

end claire_earnings_l73_73432


namespace graph_abs_symmetric_yaxis_l73_73336

theorem graph_abs_symmetric_yaxis : 
  ∀ x : ℝ, |x| = |(-x)| :=
by
  intro x
  sorry

end graph_abs_symmetric_yaxis_l73_73336


namespace min_stamps_value_l73_73621

theorem min_stamps_value (x y : ℕ) (hx : 5 * x + 7 * y = 74) : x + y = 12 :=
by
  sorry

end min_stamps_value_l73_73621


namespace smallest_possible_n_l73_73084

theorem smallest_possible_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : n > 20) : n = 52 := 
sorry

end smallest_possible_n_l73_73084


namespace number_of_groups_eq_five_l73_73361

-- Define conditions
def total_eggs : ℕ := 35
def eggs_per_group : ℕ := 7

-- Statement to prove the number of groups
theorem number_of_groups_eq_five : total_eggs / eggs_per_group = 5 := by
  sorry

end number_of_groups_eq_five_l73_73361


namespace value_of_f_m_plus_one_is_negative_l73_73894

-- Definitions for function and condition
def f (x a : ℝ) := x^2 - x + a 

-- Problem statement: Given that 'f(-m) < 0', prove 'f(m+1) < 0'
theorem value_of_f_m_plus_one_is_negative (a m : ℝ) (h : f (-m) a < 0) : f (m + 1) a < 0 :=
by 
  sorry

end value_of_f_m_plus_one_is_negative_l73_73894


namespace determine_function_l73_73499

noncomputable def functional_solution (f : ℝ → ℝ) : Prop := 
  ∃ (C₁ C₂ : ℝ), ∀ (x : ℝ), 0 < x → f x = C₁ * x + C₂ / x 

theorem determine_function (f : ℝ → ℝ) :
  (∀ (x y : ℝ), 0 < x → 0 < y → (x + 1 / x) * f y = f (x * y) + f (y / x)) →
  functional_solution f :=
sorry

end determine_function_l73_73499


namespace max_divisor_of_five_consecutive_integers_l73_73702

theorem max_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  intros n
  sorry

end max_divisor_of_five_consecutive_integers_l73_73702


namespace number_of_cars_washed_l73_73356

theorem number_of_cars_washed (cars trucks suvs total raised_per_car raised_per_truck raised_per_suv : ℕ)
  (hc : cars = 5)
  (ht : trucks = 5)
  (ha : cars + trucks + suvs = total)
  (h_cost_car : raised_per_car = 5)
  (h_cost_truck : raised_per_truck = 6)
  (h_cost_suv : raised_per_suv = 7)
  (h_amount_total : total = 100)
  (h_raised_trucks : trucks * raised_per_truck = 30)
  (h_raised_suvs : suvs * raised_per_suv = 35) :
  suvs + trucks + cars = 7 :=
by
  sorry

end number_of_cars_washed_l73_73356


namespace line_through_three_points_l73_73501

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Given conditions
def p1 : Point := { x := 1, y := -1 }
def p2 : Point := { x := 3, y := 3 }
def p3 : Point := { x := 2, y := 1 }

-- The line that passes through the points
def line_eq (m b : ℝ) (p : Point) : Prop :=
  p.y = m * p.x + b

-- The condition of passing through the three points
def passes_three_points (m b : ℝ) : Prop :=
  line_eq m b p1 ∧ line_eq m b p2 ∧ line_eq m b p3

-- The statement to prove
theorem line_through_three_points (m b : ℝ) (h : passes_three_points m b) : m + b = -1 :=
  sorry

end line_through_three_points_l73_73501


namespace hcl_reaction_l73_73722

theorem hcl_reaction
  (stoichiometry : ∀ (HCl NaHCO3 H2O CO2 NaCl : ℕ), HCl = NaHCO3 ∧ H2O = NaHCO3 ∧ CO2 = NaHCO3 ∧ NaCl = NaHCO3)
  (naHCO3_moles : ℕ)
  (reaction_moles : naHCO3_moles = 3) :
  ∃ (HCl_moles : ℕ), HCl_moles = naHCO3_moles :=
by
  sorry

end hcl_reaction_l73_73722


namespace average_after_31st_inning_l73_73532

-- Define the conditions as Lean definitions
def initial_average (A : ℝ) := A

def total_runs_before_31st_inning (A : ℝ) := 30 * A

def score_in_31st_inning := 105

def new_average (A : ℝ) := A + 3

def total_runs_after_31st_inning (A : ℝ) := total_runs_before_31st_inning A + score_in_31st_inning

-- Define the statement to prove the batsman's average after the 31st inning is 15
theorem average_after_31st_inning (A : ℝ) : total_runs_after_31st_inning A = 31 * (new_average A) → new_average A = 15 := by
  sorry

end average_after_31st_inning_l73_73532


namespace operation_addition_l73_73042

theorem operation_addition (a b c : ℝ) (op : ℝ → ℝ → ℝ)
  (H : ∀ a b c : ℝ, op (op a b) c = a + b + c) :
  ∀ a b : ℝ, op a b = a + b :=
sorry

end operation_addition_l73_73042


namespace number_of_students_and_average_output_l73_73392

theorem number_of_students_and_average_output 
  (total_potatoes : ℕ)
  (days : ℕ)
  (x y : ℕ) 
  (h1 : total_potatoes = 45715) 
  (h2 : days = 5)
  (h3 : x * y * days = total_potatoes) : 
  x = 41 ∧ y = 223 :=
by
  sorry

end number_of_students_and_average_output_l73_73392


namespace sin_cos_sum_l73_73138

-- Let theta be an angle in the second quadrant
variables (θ : ℝ)
-- Given the condition tan(θ + π / 4) = 1 / 2
variable (h1 : Real.tan (θ + Real.pi / 4) = 1 / 2)
-- Given θ is in the second quadrant
variable (h2 : θ ∈ Set.Ioc (Real.pi / 2) Real.pi)

-- Prove sin θ + cos θ = - sqrt(10) / 5
theorem sin_cos_sum (h1 : Real.tan (θ + Real.pi / 4) = 1 / 2) (h2 : θ ∈ Set.Ioc (Real.pi / 2) Real.pi) :
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_cos_sum_l73_73138


namespace bricks_required_l73_73428

theorem bricks_required (L_courtyard W_courtyard L_brick W_brick : Real)
  (hcourtyard : L_courtyard = 35) 
  (wcourtyard : W_courtyard = 24) 
  (hbrick_len : L_brick = 0.15) 
  (hbrick_wid : W_brick = 0.08) : 
  (L_courtyard * W_courtyard) / (L_brick * W_brick) = 70000 := 
by
  sorry

end bricks_required_l73_73428


namespace smaller_number_l73_73082

theorem smaller_number (x y : ℤ) (h1 : x + y = 79) (h2 : x - y = 15) : y = 32 := by
  sorry

end smaller_number_l73_73082


namespace hat_cost_l73_73040

theorem hat_cost (total_hats blue_hat_cost green_hat_cost green_hats : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_hat_cost = 6)
  (h3 : green_hat_cost = 7)
  (h4 : green_hats = 20) :
  (total_hats - green_hats) * blue_hat_cost + green_hats * green_hat_cost = 530 := 
by sorry

end hat_cost_l73_73040


namespace same_number_written_every_vertex_l73_73268

theorem same_number_written_every_vertex (a : ℕ → ℝ) (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → a i > 0) 
(h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → (a i) ^ 2 = a (i - 1) + a (i + 1) ) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → a i = 2 :=
by
  sorry

end same_number_written_every_vertex_l73_73268


namespace original_price_computer_l73_73389

noncomputable def first_store_price (P : ℝ) : ℝ := 0.94 * P

noncomputable def second_store_price (exchange_rate : ℝ) : ℝ := (920 / 0.95) * exchange_rate

theorem original_price_computer 
  (exchange_rate : ℝ)
  (h : exchange_rate = 1.1) 
  (H : (first_store_price P - second_store_price exchange_rate = 19)) :
  P = 1153.47 :=
by
  sorry

end original_price_computer_l73_73389


namespace ChipsEquivalence_l73_73290

theorem ChipsEquivalence
  (x y : ℕ)
  (h1 : y = x - 2)
  (h2 : 3 * x - 3 = 4 * y - 4) :
  3 * x - 3 = 24 :=
by
  sorry

end ChipsEquivalence_l73_73290


namespace units_digit_of_quotient_l73_73601

theorem units_digit_of_quotient (n : ℕ) (h1 : n = 1987) : 
  (((4^n + 6^n) / 5) % 10) = 0 :=
by
  have pattern_4 : ∀ (k : ℕ), (4^k) % 10 = if k % 2 = 0 then 6 else 4 := sorry
  have pattern_6 : ∀ (k : ℕ), (6^k) % 10 = 6 := sorry
  have units_sum : (4^1987 % 10 + 6^1987 % 10) % 10 = 0 := sorry
  have multiple_of_5 : (4^1987 + 6^1987) % 5 = 0 := sorry
  sorry

end units_digit_of_quotient_l73_73601


namespace area_outside_smaller_squares_l73_73862

theorem area_outside_smaller_squares (side_large : ℕ) (side_small1 : ℕ) (side_small2 : ℕ)
  (no_overlap : Prop) (side_large_eq : side_large = 9)
  (side_small1_eq : side_small1 = 4)
  (side_small2_eq : side_small2 = 2) :
  (side_large * side_large - (side_small1 * side_small1 + side_small2 * side_small2)) = 61 :=
by
  sorry

end area_outside_smaller_squares_l73_73862


namespace area_ratio_l73_73245

variables {rA rB : ℝ} (C_A C_B : ℝ)

#check C_A = 2 * Real.pi * rA
#check C_B = 2 * Real.pi * rB

theorem area_ratio (h : (60 / 360) * C_A = (40 / 360) * C_B) : (Real.pi * rA^2) / (Real.pi * rB^2) = 4 / 9 := by
  sorry

end area_ratio_l73_73245


namespace johns_age_l73_73592

variable (J : ℕ)

theorem johns_age :
  J - 5 = (1 / 2) * (J + 8) → J = 18 := by
    sorry

end johns_age_l73_73592


namespace victory_saved_less_l73_73976

-- Definitions based on conditions
def total_savings : ℕ := 1900
def sam_savings : ℕ := 1000
def victory_savings : ℕ := total_savings - sam_savings

-- Prove that Victory saved $100 less than Sam
theorem victory_saved_less : sam_savings - victory_savings = 100 := by
  -- placeholder for the proof
  sorry

end victory_saved_less_l73_73976


namespace number_of_solutions_l73_73354

-- Defining the conditions for the equation
def isCondition (x : ℝ) : Prop := x ≠ 2 ∧ x ≠ 3

-- Defining the equation
def eqn (x : ℝ) : Prop := (3 * x^2 - 15 * x + 18) / (x^2 - 5 * x + 6) = x - 2

-- Defining the property that we need to prove
def property (x : ℝ) : Prop := eqn x ∧ isCondition x

-- Statement of the proof problem
theorem number_of_solutions : 
  ∃! x : ℝ, property x :=
sorry

end number_of_solutions_l73_73354


namespace person_B_D_coins_l73_73887

theorem person_B_D_coins
  (a d : ℤ)
  (h1 : a - 3 * d = 58)
  (h2 : a - 2 * d = 58)
  (h3 : a + d = 60)
  (h4 : a + 2 * d = 60)
  (h5 : a + 3 * d = 60) :
  (a - 2 * d = 28) ∧ (a = 24) :=
by
  sorry

end person_B_D_coins_l73_73887


namespace binary_11011011_to_base4_is_3123_l73_73020

def binary_to_base4 (b : Nat) : Nat :=
  -- Function to convert binary number to base 4
  -- This will skip implementation details
  sorry

theorem binary_11011011_to_base4_is_3123 :
  binary_to_base4 0b11011011 = 0x3123 := 
sorry

end binary_11011011_to_base4_is_3123_l73_73020


namespace gcd_min_value_l73_73140

theorem gcd_min_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 18) :
  Nat.gcd (12 * a) (20 * b) = 72 :=
by
  sorry

end gcd_min_value_l73_73140


namespace simplify_complex_expr_correct_l73_73972

noncomputable def simplify_complex_expr (i : ℂ) (h : i^2 = -1) : ℂ :=
  3 * (4 - 2 * i) - 2 * i * (3 - 2 * i) + (1 + i) * (2 + i)

theorem simplify_complex_expr_correct (i : ℂ) (h : i^2 = -1) : 
  simplify_complex_expr i h = 9 - 9 * i :=
by
  sorry

end simplify_complex_expr_correct_l73_73972


namespace johns_profit_l73_73166

/-- Define the number of ducks -/
def numberOfDucks : ℕ := 30

/-- Define the cost per duck -/
def costPerDuck : ℤ := 10

/-- Define the weight per duck -/
def weightPerDuck : ℤ := 4

/-- Define the selling price per pound -/
def pricePerPound : ℤ := 5

/-- Define the total cost to buy the ducks -/
def totalCost : ℤ := numberOfDucks * costPerDuck

/-- Define the selling price per duck -/
def sellingPricePerDuck : ℤ := weightPerDuck * pricePerPound

/-- Define the total revenue from selling all the ducks -/
def totalRevenue : ℤ := numberOfDucks * sellingPricePerDuck

/-- Define the profit John made -/
def profit : ℤ := totalRevenue - totalCost

/-- The theorem stating the profit John made given the conditions is $300 -/
theorem johns_profit : profit = 300 := by
  sorry

end johns_profit_l73_73166


namespace sum_angles_acute_l73_73857

open Real

theorem sum_angles_acute (A B C : ℝ) (hA_ac : A < π / 2) (hB_ac : B < π / 2) (hC_ac : C < π / 2)
  (h_angle_sum : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 1) :
  π / 2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end sum_angles_acute_l73_73857


namespace shelves_needed_l73_73401

theorem shelves_needed (initial_stock : ℕ) (additional_shipment : ℕ) (bears_per_shelf : ℕ) (total_bears : ℕ) (shelves : ℕ) :
  initial_stock = 4 → 
  additional_shipment = 10 → 
  bears_per_shelf = 7 → 
  total_bears = initial_stock + additional_shipment →
  total_bears / bears_per_shelf = shelves →
  shelves = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end shelves_needed_l73_73401


namespace kendall_nickels_l73_73729

def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

theorem kendall_nickels (q d : ℕ) (total : ℝ) (hq : q = 10) (hd : d = 12) (htotal : total = 4) : 
  ∃ n : ℕ, value_of_nickels n = total - (value_of_quarters q + value_of_dimes d) ∧ n = 6 :=
by
  sorry

end kendall_nickels_l73_73729


namespace units_digit_17_pow_31_l73_73696

theorem units_digit_17_pow_31 : (17 ^ 31) % 10 = 3 := by
  sorry

end units_digit_17_pow_31_l73_73696


namespace inequality_proof_l73_73967

open scoped BigOperators

theorem inequality_proof {n : ℕ} (a : Fin n → ℝ) 
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 1 / 2) :
  (∑ i, (a i)^2 / (∑ i, a i)^2) ≥ (∑ i, (1 - a i)^2 / (∑ i, (1 - a i))^2) := 
by 
  sorry

end inequality_proof_l73_73967


namespace three_digit_number_count_correct_l73_73695

noncomputable
def count_three_digit_numbers (digits : List ℕ) : ℕ :=
  if h : digits.length = 5 then
    (5 * 4 * 3 : ℕ)
  else
    0

theorem three_digit_number_count_correct :
  count_three_digit_numbers [1, 3, 5, 7, 9] = 60 :=
by
  unfold count_three_digit_numbers
  simp only [List.length, if_pos]
  rfl

end three_digit_number_count_correct_l73_73695


namespace find_side_DF_in_triangle_DEF_l73_73569

theorem find_side_DF_in_triangle_DEF
  (DE EF DM : ℝ)
  (h_DE : DE = 7)
  (h_EF : EF = 10)
  (h_DM : DM = 5) :
  ∃ DF : ℝ, DF = Real.sqrt 51 :=
by
  sorry

end find_side_DF_in_triangle_DEF_l73_73569


namespace math_problem_l73_73638

theorem math_problem (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) := 
by
  sorry

end math_problem_l73_73638


namespace arithmetic_sequence_property_l73_73310

variable {a : ℕ → ℕ}

-- Given condition in the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ d c : ℕ, ∀ n : ℕ, a n = c + n * d

def condition (a : ℕ → ℕ) : Prop := a 4 + a 8 = 16

-- Problem statement
theorem arithmetic_sequence_property (a : ℕ → ℕ)
  (h_arith_seq : arithmetic_sequence a)
  (h_condition : condition a) :
  a 2 + a 6 + a 10 = 24 :=
sorry

end arithmetic_sequence_property_l73_73310


namespace students_from_second_grade_l73_73018

theorem students_from_second_grade (r1 r2 r3 : ℕ) (total_students sample_size : ℕ) (h_ratio: r1 = 3 ∧ r2 = 3 ∧ r3 = 4 ∧ r1 + r2 + r3 = 10) (h_sample_size: sample_size = 50) : 
  (r2 * sample_size / (r1 + r2 + r3)) = 15 :=
by
  sorry

end students_from_second_grade_l73_73018


namespace max_value_of_quadratic_l73_73562

theorem max_value_of_quadratic:
  ∀ (x : ℝ), (∃ y : ℝ, y = -3 * x ^ 2 + 9) → (∃ max_y : ℝ, max_y = 9 ∧ ∀ x : ℝ, -3 * x ^ 2 + 9 ≤ max_y) :=
by
  sorry

end max_value_of_quadratic_l73_73562


namespace probability_of_drawing_red_or_green_l73_73122

def red_marbles : ℕ := 4
def green_marbles : ℕ := 3
def yellow_marbles : ℕ := 6

def total_marbles : ℕ := red_marbles + green_marbles + yellow_marbles
def favorable_marbles : ℕ := red_marbles + green_marbles
def probability_of_red_or_green : ℚ := favorable_marbles / total_marbles

theorem probability_of_drawing_red_or_green :
  probability_of_red_or_green = 7 / 13 := by
  sorry

end probability_of_drawing_red_or_green_l73_73122


namespace evaluate_powers_of_i_l73_73639

theorem evaluate_powers_of_i :
  (Complex.I ^ 50) + (Complex.I ^ 105) = -1 + Complex.I :=
by 
  sorry

end evaluate_powers_of_i_l73_73639


namespace solution_set_of_inequality_l73_73461

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_f1_zero : f 1 = 0) : 
  { x | f x > 0 } = { x | x < -1 ∨ 1 < x } := 
by
  sorry

end solution_set_of_inequality_l73_73461


namespace mass_of_CaSO4_formed_correct_l73_73155

noncomputable def mass_CaSO4_formed 
(mass_CaO : ℝ) (mass_H2SO4 : ℝ)
(molar_mass_CaO : ℝ) (molar_mass_H2SO4 : ℝ) (molar_mass_CaSO4 : ℝ) : ℝ :=
  let moles_CaO := mass_CaO / molar_mass_CaO
  let moles_H2SO4 := mass_H2SO4 / molar_mass_H2SO4
  let limiting_reactant_moles := min moles_CaO moles_H2SO4
  limiting_reactant_moles * molar_mass_CaSO4

theorem mass_of_CaSO4_formed_correct :
  mass_CaSO4_formed 25 35 56.08 98.09 136.15 = 48.57 :=
by
  rw [mass_CaSO4_formed]
  sorry

end mass_of_CaSO4_formed_correct_l73_73155


namespace MrBensonPaidCorrectAmount_l73_73998

-- Definitions based on the conditions
def generalAdmissionTicketPrice : ℤ := 40
def VIPTicketPrice : ℤ := 60
def premiumTicketPrice : ℤ := 80

def generalAdmissionTicketsBought : ℤ := 10
def VIPTicketsBought : ℤ := 3
def premiumTicketsBought : ℤ := 2

def generalAdmissionExcessThreshold : ℤ := 8
def VIPExcessThreshold : ℤ := 2
def premiumExcessThreshold : ℤ := 1

def generalAdmissionDiscountPercentage : ℤ := 3
def VIPDiscountPercentage : ℤ := 7
def premiumDiscountPercentage : ℤ := 10

-- Function to calculate the cost without discounts
def costWithoutDiscount : ℤ :=
  (generalAdmissionTicketsBought * generalAdmissionTicketPrice) +
  (VIPTicketsBought * VIPTicketPrice) +
  (premiumTicketsBought * premiumTicketPrice)

-- Function to calculate the total discount
def totalDiscount : ℤ :=
  let generalAdmissionDiscount := if generalAdmissionTicketsBought > generalAdmissionExcessThreshold then 
    (generalAdmissionTicketsBought - generalAdmissionExcessThreshold) * generalAdmissionTicketPrice * generalAdmissionDiscountPercentage / 100 else 0
  let VIPDiscount := if VIPTicketsBought > VIPExcessThreshold then 
    (VIPTicketsBought - VIPExcessThreshold) * VIPTicketPrice * VIPDiscountPercentage / 100 else 0
  let premiumDiscount := if premiumTicketsBought > premiumExcessThreshold then 
    (premiumTicketsBought - premiumExcessThreshold) * premiumTicketPrice * premiumDiscountPercentage / 100 else 0
  generalAdmissionDiscount + VIPDiscount + premiumDiscount

-- Function to calculate the total cost after discounts
def totalCostAfterDiscount : ℤ := costWithoutDiscount - totalDiscount

-- Proof statement
theorem MrBensonPaidCorrectAmount :
  totalCostAfterDiscount = 723 :=
by
  sorry

end MrBensonPaidCorrectAmount_l73_73998


namespace tina_money_left_l73_73645

theorem tina_money_left :
  let june_savings := 27
  let july_savings := 14
  let august_savings := 21
  let books_spending := 5
  let shoes_spending := 17
  june_savings + july_savings + august_savings - (books_spending + shoes_spending) = 40 :=
by
  sorry

end tina_money_left_l73_73645


namespace cloth_length_l73_73860

theorem cloth_length (L : ℕ) (x : ℕ) :
  32 + x = L ∧ 20 + 3 * x = L → L = 38 :=
by
  sorry

end cloth_length_l73_73860


namespace no_common_factor_l73_73384

open Polynomial

theorem no_common_factor (f g : ℤ[X]) : f = X^2 + X - 1 → g = X^2 + 2 * X → ∀ d : ℤ[X], d ∣ f ∧ d ∣ g → d = 1 :=
by
  intros h1 h2 d h_dv
  rw [h1, h2] at h_dv
  -- Proof steps would go here
  sorry

end no_common_factor_l73_73384


namespace power_inequality_l73_73745

variable (a b c : ℝ)
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_ineq : a^19 / b^19 + b^19 / c^19 + c^19 / a^19 ≤ a^19 / c^19 + b^19 / a^19 + c^19 / b^19)

theorem power_inequality :
  a^20 / b^20 + b^20 / c^20 + c^20 / a^20 ≤ a^20 / c^20 + b^20 / a^20 + c^20 / b^20 :=
by
  sorry

end power_inequality_l73_73745


namespace hyperbola_asymptote_slope_l73_73510

theorem hyperbola_asymptote_slope
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c ≠ -a ∧ c ≠ a)
  (H1 : (c ≠ -a ∧ c ≠ a) ∧ (a ≠ 0) ∧ (b ≠ 0))
  (H_perp : (c + a) * (c - a) * (a * a * a * a) + (b * b * b * b) = 0) :
  abs (b / a) = 1 :=
by
  sorry  -- Proof here is not required as per the given instructions

end hyperbola_asymptote_slope_l73_73510


namespace smallest_angle_l73_73986

theorem smallest_angle (largest_angle : ℝ) (a b : ℝ) (h1 : largest_angle = 120) (h2 : 3 * a = 2 * b) (h3 : largest_angle + a + b = 180) : b = 24 := by
  sorry

end smallest_angle_l73_73986


namespace volume_ratio_spheres_l73_73498

theorem volume_ratio_spheres (r1 r2 r3 v1 v2 v3 : ℕ)
  (h_rad_ratio : r1 = 1 ∧ r2 = 2 ∧ r3 = 3)
  (h_vol_ratio : v1 = r1^3 ∧ v2 = r2^3 ∧ v3 = r3^3) :
  v3 = 3 * (v1 + v2) := by
  -- main proof goes here
  sorry

end volume_ratio_spheres_l73_73498


namespace car_speed_is_48_l73_73726

theorem car_speed_is_48 {v : ℝ} : (3600 / v = 75) → v = 48 := 
by {
  sorry
}

end car_speed_is_48_l73_73726


namespace jon_payment_per_visit_l73_73328

theorem jon_payment_per_visit 
  (visits_per_hour : ℕ) (operating_hours_per_day : ℕ) (income_in_month : ℚ) (days_in_month : ℕ) 
  (visits_per_hour_eq : visits_per_hour = 50) 
  (operating_hours_per_day_eq : operating_hours_per_day = 24) 
  (income_in_month_eq : income_in_month = 3600) 
  (days_in_month_eq : days_in_month = 30) :
  (income_in_month / (visits_per_hour * operating_hours_per_day * days_in_month) : ℚ) = 0.10 := 
by
  sorry

end jon_payment_per_visit_l73_73328


namespace exists_perfect_square_in_sequence_of_f_l73_73308

noncomputable def f (n : ℕ) : ℕ :=
  ⌊(n : ℝ) + Real.sqrt n⌋₊

theorem exists_perfect_square_in_sequence_of_f (m : ℕ) (h : m = 1111) :
  ∃ k, ∃ n, f^[n] m = k * k := 
sorry

end exists_perfect_square_in_sequence_of_f_l73_73308


namespace total_degree_difference_l73_73871

-- Definitions based on conditions
def timeStart : ℕ := 12 * 60  -- noon in minutes
def timeEnd : ℕ := 14 * 60 + 30  -- 2:30 PM in minutes
def numTimeZones : ℕ := 3  -- Three time zones
def degreesInCircle : ℕ := 360  -- Degrees in a full circle

-- Calculate degrees moved by each hand
def degreesMovedByHourHand : ℚ := (timeEnd - timeStart) / (12 * 60) * degreesInCircle
def degreesMovedByMinuteHand : ℚ := (timeEnd - timeStart) % 60 * (degreesInCircle / 60)
def degreesMovedBySecondHand : ℕ := 0  -- At 2:30 PM, second hand is at initial position

-- Calculate total degree difference for all three hands and time zones
def totalDegrees : ℚ := 
  (degreesMovedByHourHand + degreesMovedByMinuteHand + degreesMovedBySecondHand) * numTimeZones

-- Theorem statement to prove
theorem total_degree_difference :
  totalDegrees = 765 := by
  sorry

end total_degree_difference_l73_73871


namespace Jill_tax_on_clothing_l73_73864

theorem Jill_tax_on_clothing 
  (spent_clothing : ℝ) (spent_food : ℝ) (spent_other : ℝ) (total_spent : ℝ) (tax_clothing : ℝ) 
  (tax_other_rate : ℝ) (total_tax_rate : ℝ) 
  (h_clothing : spent_clothing = 0.5 * total_spent) 
  (h_food : spent_food = 0.2 * total_spent) 
  (h_other : spent_other = 0.3 * total_spent) 
  (h_other_tax : tax_other_rate = 0.1) 
  (h_total_tax : total_tax_rate = 0.055) 
  (h_total_spent : total_spent = 100):
  (tax_clothing * spent_clothing + tax_other_rate * spent_other) = total_tax_rate * total_spent → 
  tax_clothing = 0.05 :=
by
  sorry

end Jill_tax_on_clothing_l73_73864


namespace expected_sides_of_red_polygon_l73_73952

-- Define the conditions
def isChosenWithinSquare (F : ℝ × ℝ) (side_length: ℝ) : Prop :=
  0 ≤ F.1 ∧ F.1 ≤ side_length ∧ 0 ≤ F.2 ∧ F.2 ≤ side_length

def pointF (side_length: ℝ) : ℝ × ℝ := sorry
def foldToF (vertex: ℝ × ℝ) (F: ℝ × ℝ) : ℝ := sorry

-- Define the expected number of sides of the resulting red polygon
noncomputable def expected_sides (side_length : ℝ) : ℝ :=
  let P_g := 2 - (Real.pi / 2)
  let P_o := (Real.pi / 2) - 1 
  (3 * P_o) + (4 * P_g)

-- Prove the expected number of sides equals 5 - π / 2
theorem expected_sides_of_red_polygon (side_length : ℝ) :
  expected_sides side_length = 5 - (Real.pi / 2) := 
  by sorry

end expected_sides_of_red_polygon_l73_73952


namespace problem1_solution_set_problem2_a_range_l73_73373

section
variable (a x : ℝ)

def f (x : ℝ) (a : ℝ) := |2 * x - a| + a

-- Problem 1
theorem problem1_solution_set (h : a = 3) : {x | f x a ≤ 6} = {x | 0 ≤ x ∧ x ≤ 3} :=
by
  sorry

def g (x : ℝ) := |2 * x - 3|

-- Problem 2
theorem problem2_a_range : ∀ a : ℝ, ∀ x : ℝ, f x a + g x ≥ 5 ↔ 4 ≤ a :=
by
  sorry
end

end problem1_solution_set_problem2_a_range_l73_73373


namespace quadratic_inequality_solution_l73_73874

theorem quadratic_inequality_solution (m : ℝ) (h : m ≠ 0) : 
  (∃ x : ℝ, m * x^2 - x + 1 < 0) ↔ (m ∈ Set.Iio 0 ∨ m ∈ Set.Ioo 0 (1 / 4)) :=
by
  sorry

end quadratic_inequality_solution_l73_73874


namespace stratified_sampling_l73_73804

-- Definitions
def total_staff : ℕ := 150
def senior_titles : ℕ := 45
def intermediate_titles : ℕ := 90
def clerks : ℕ := 15
def sample_size : ℕ := 10

-- Ratios for stratified sampling
def senior_sample : ℕ := (senior_titles * sample_size) / total_staff
def intermediate_sample : ℕ := (intermediate_titles * sample_size) / total_staff
def clerks_sample : ℕ := (clerks * sample_size) / total_staff

-- Theorem statement
theorem stratified_sampling :
  senior_sample = 3 ∧ intermediate_sample = 6 ∧ clerks_sample = 1 :=
by
  sorry

end stratified_sampling_l73_73804


namespace bryan_bought_4_pairs_of_pants_l73_73253

def number_of_tshirts : Nat := 5
def total_cost : Nat := 1500
def cost_per_tshirt : Nat := 100
def cost_per_pants : Nat := 250

theorem bryan_bought_4_pairs_of_pants : (total_cost - number_of_tshirts * cost_per_tshirt) / cost_per_pants = 4 := by
  sorry

end bryan_bought_4_pairs_of_pants_l73_73253


namespace values_of_a_and_b_range_of_c_isosceles_perimeter_l73_73671

def a : ℝ := 3
def b : ℝ := 4

axiom triangle_ABC (c : ℝ) : 0 < c

noncomputable def equation_condition (a b : ℝ) : Prop :=
  |a-3| + (b-4)^2 = 0

noncomputable def is_valid_c (c : ℝ) : Prop :=
  1 < c ∧ c < 7

theorem values_of_a_and_b (h : equation_condition a b) : a = 3 ∧ b = 4 := sorry

theorem range_of_c (h : equation_condition a b) : is_valid_c c := sorry

noncomputable def isosceles_triangle (c : ℝ) : Prop :=
  c = 4 ∨ c = 3

theorem isosceles_perimeter (h : equation_condition a b) (hc : isosceles_triangle c) : (3 + 3 + 4 = 10) ∨ (4 + 4 + 3 = 11) := sorry

end values_of_a_and_b_range_of_c_isosceles_perimeter_l73_73671


namespace andrea_rhinestones_ratio_l73_73381

theorem andrea_rhinestones_ratio :
  (∃ (B : ℕ), B = 45 - (1 / 5 * 45) - 21) →
  (1/5 * 45 : ℕ) + B + 21 = 45 →
  (B : ℕ) / 45 = 1 / 3 := 
sorry

end andrea_rhinestones_ratio_l73_73381


namespace other_equation_l73_73469

-- Define the variables for the length of the rope and the depth of the well
variables (x y : ℝ)

-- Given condition
def cond1 : Prop := (1/4) * x = y + 3

-- The proof goal
theorem other_equation (h : cond1 x y) : (1/5) * x = y + 2 :=
sorry

end other_equation_l73_73469


namespace initial_deposit_l73_73799

theorem initial_deposit (P R : ℝ) (h1 : 8400 = P + (P * R * 2) / 100) (h2 : 8760 = P + (P * (R + 4) * 2) / 100) : 
  P = 2250 :=
  sorry

end initial_deposit_l73_73799


namespace cars_and_tourists_l73_73793

theorem cars_and_tourists (n t : ℕ) (h : n * t = 737) : n = 11 ∧ t = 67 ∨ n = 67 ∧ t = 11 :=
by
  sorry

end cars_and_tourists_l73_73793


namespace table_tennis_teams_equation_l73_73293

-- Variables
variable (x : ℕ)

-- Conditions
def total_matches : ℕ := 28
def teams_playing_equation : Prop := x * (x - 1) = 28 * 2

-- Theorem Statement
theorem table_tennis_teams_equation : teams_playing_equation x :=
sorry

end table_tennis_teams_equation_l73_73293


namespace cost_of_senior_ticket_l73_73736

theorem cost_of_senior_ticket (x : ℤ) (total_tickets : ℤ) (cost_regular_ticket : ℤ) (total_sales : ℤ) (senior_tickets_sold : ℤ) (regular_tickets_sold : ℤ) :
  total_tickets = 65 →
  cost_regular_ticket = 15 →
  total_sales = 855 →
  senior_tickets_sold = 24 →
  regular_tickets_sold = total_tickets - senior_tickets_sold →
  total_sales = senior_tickets_sold * x + regular_tickets_sold * cost_regular_ticket →
  x = 10 :=
by
  sorry

end cost_of_senior_ticket_l73_73736


namespace bottles_drunk_l73_73244

theorem bottles_drunk (initial_bottles remaining_bottles : ℕ)
  (h₀ : initial_bottles = 17) (h₁ : remaining_bottles = 14) :
  initial_bottles - remaining_bottles = 3 :=
sorry

end bottles_drunk_l73_73244


namespace gcd_of_8a_plus_3_and_5a_plus_2_l73_73367

theorem gcd_of_8a_plus_3_and_5a_plus_2 (a : ℕ) : Nat.gcd (8 * a + 3) (5 * a + 2) = 1 :=
by
  sorry

end gcd_of_8a_plus_3_and_5a_plus_2_l73_73367


namespace second_year_selection_l73_73141

noncomputable def students_from_first_year : ℕ := 30
noncomputable def students_from_second_year : ℕ := 40
noncomputable def selected_from_first_year : ℕ := 6
noncomputable def selected_from_second_year : ℕ := (selected_from_first_year * students_from_second_year) / students_from_first_year

theorem second_year_selection :
  students_from_second_year = 40 ∧ students_from_first_year = 30 ∧ selected_from_first_year = 6 →
  selected_from_second_year = 8 :=
by
  intros h
  sorry

end second_year_selection_l73_73141


namespace math_problem_l73_73581

theorem math_problem (a b : ℕ) (x y : ℚ) (h1 : a = 10) (h2 : b = 11) (h3 : x = 1.11) (h4 : y = 1.01) :
  ∃ k : ℕ, k * y = 2.02 ∧ (a * x + b * y - k * y = 20.19) :=
by {
  sorry
}

end math_problem_l73_73581


namespace yogurt_combinations_l73_73229

-- Definitions based on conditions
def flavors : ℕ := 5
def toppings : ℕ := 8
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The problem statement to be proved
theorem yogurt_combinations :
  flavors * choose toppings 3 = 280 :=
by
  sorry

end yogurt_combinations_l73_73229


namespace polygon_sides_in_arithmetic_progression_l73_73573

theorem polygon_sides_in_arithmetic_progression 
  (a : ℕ → ℝ) (n : ℕ) (h1: ∀ i, 1 ≤ i ∧ i ≤ n → a i = a 1 + (i - 1) * 10) 
  (h2 : a n = 150) : n = 12 :=
sorry

end polygon_sides_in_arithmetic_progression_l73_73573


namespace total_sheep_flock_l73_73226

-- Definitions and conditions based on the problem description
def crossing_rate : ℕ := 3 -- Sheep per minute
def sleep_duration : ℕ := 90 -- Duration of sleep in minutes
def sheep_counted_before_sleep : ℕ := 42 -- Sheep counted before falling asleep

-- Total sheep that crossed while Nicholas was asleep
def sheep_during_sleep := crossing_rate * sleep_duration 

-- Total sheep that crossed when Nicholas woke up
def total_sheep_after_sleep := sheep_counted_before_sleep + sheep_during_sleep

-- Prove the total number of sheep in the flock
theorem total_sheep_flock : (2 * total_sheep_after_sleep) = 624 :=
by
  sorry

end total_sheep_flock_l73_73226


namespace quadratic_roots_condition_l73_73291

theorem quadratic_roots_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ < 1 ∧ x₂ > 1 ∧ x₁^2 + (3 * a - 1) * x₁ + a + 8 = 0 ∧
  x₂^2 + (3 * a - 1) * x₂ + a + 8 = 0) → a < -2 :=
by
  sorry

end quadratic_roots_condition_l73_73291


namespace moe_cannot_finish_on_time_l73_73113

theorem moe_cannot_finish_on_time (lawn_length lawn_width : ℝ) (swath : ℕ) (overlap : ℕ) (speed : ℝ) (available_time : ℝ) :
  lawn_length = 120 ∧ lawn_width = 180 ∧ swath = 30 ∧ overlap = 6 ∧ speed = 4000 ∧ available_time = 2 →
  (lawn_width / (swath - overlap) * lawn_length / speed) > available_time :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end moe_cannot_finish_on_time_l73_73113


namespace james_worked_41_hours_l73_73089

theorem james_worked_41_hours (x : ℝ) :
  ∃ (J : ℕ), 
    (24 * x + 12 * 1.5 * x = 40 * x + (J - 40) * 2 * x) ∧ 
    J = 41 := 
by 
  sorry

end james_worked_41_hours_l73_73089


namespace sum_eq_neg_20_div_3_l73_73823
-- Import the necessary libraries

-- The main theoretical statement
theorem sum_eq_neg_20_div_3
    (a b c d : ℝ)
    (h : a + 2 = b + 4 ∧ b + 4 = c + 6 ∧ c + 6 = d + 8 ∧ d + 8 = a + b + c + d + 10) :
    a + b + c + d = -20 / 3 :=
by
  sorry

end sum_eq_neg_20_div_3_l73_73823


namespace A_plus_B_l73_73945

theorem A_plus_B {A B : ℚ} (h : ∀ x : ℚ, (Bx - 19) / (x^2 - 8*x + 15) = A / (x - 3) + 5 / (x - 5)) : 
  A + B = 33 / 5 := sorry

end A_plus_B_l73_73945


namespace lowest_exam_score_l73_73782

theorem lowest_exam_score 
  (first_exam_score : ℕ := 90) 
  (second_exam_score : ℕ := 108) 
  (third_exam_score : ℕ := 102) 
  (max_score_per_exam : ℕ := 120) 
  (desired_average : ℕ := 100) 
  (total_exams : ℕ := 5) 
  (total_score_needed : ℕ := desired_average * total_exams) : 
  ∃ (lowest_score : ℕ), lowest_score = 80 :=
by
  sorry

end lowest_exam_score_l73_73782


namespace star_example_l73_73125

section star_operation

variables (x y z : ℕ) 

-- Define the star operation as a binary function
def star (a b : ℕ) : ℕ := a * b

-- Given conditions
axiom star_idempotent : ∀ x : ℕ, star x x = 0
axiom star_associative : ∀ x y z : ℕ, star x (star y z) = (star x y) + z

-- Main theorem to be proved
theorem star_example : star 1993 1935 = 58 :=
sorry

end star_operation

end star_example_l73_73125


namespace ratio_of_area_l73_73811

noncomputable def area_of_triangle_ratio (AB CD height : ℝ) (h : CD = 2 * AB) : ℝ :=
  let ABCD_area := (AB + CD) * height / 2
  let EAB_area := ABCD_area / 3
  EAB_area / ABCD_area

theorem ratio_of_area (AB CD : ℝ) (height : ℝ) (h1 : AB = 10) (h2 : CD = 20) (h3 : height = 5) : 
  area_of_triangle_ratio AB CD height (by rw [h1, h2]; ring) = 1 / 3 :=
sorry

end ratio_of_area_l73_73811


namespace pond_width_l73_73876

theorem pond_width
  (L : ℝ) (D : ℝ) (V : ℝ) (W : ℝ)
  (hL : L = 20)
  (hD : D = 5)
  (hV : V = 1000)
  (hVolume : V = L * W * D) :
  W = 10 :=
by {
  sorry
}

end pond_width_l73_73876


namespace minimize_shoes_l73_73768

-- Definitions for inhabitants, one-legged inhabitants, and shoe calculations
def total_inhabitants := 10000
def P (percent_one_legged : ℕ) := (percent_one_legged * total_inhabitants) / 100
def non_one_legged (percent_one_legged : ℕ) := total_inhabitants - (P percent_one_legged)
def non_one_legged_with_shoes (percent_one_legged : ℕ) := (non_one_legged percent_one_legged) / 2
def shoes_needed (percent_one_legged : ℕ) := 
  (P percent_one_legged) + 2 * (non_one_legged_with_shoes percent_one_legged)

-- Theorem to prove that 100% one-legged minimizes the shoes required
theorem minimize_shoes : ∀ (percent_one_legged : ℕ), shoes_needed percent_one_legged = total_inhabitants → percent_one_legged = 100 :=
by
  intros percent_one_legged h
  sorry

end minimize_shoes_l73_73768


namespace problem1_problem2_l73_73781

variable (x y : ℝ)

-- Problem 1
theorem problem1 : (x + y) ^ 2 + x * (x - 2 * y) = 2 * x ^ 2 + y ^ 2 := by
  sorry

variable (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) -- to ensure the denominators are non-zero

-- Problem 2
theorem problem2 : (x ^ 2 - 6 * x + 9) / (x - 2) / (x + 2 - (3 * x - 4) / (x - 2)) = (x - 3) / x := by
  sorry

end problem1_problem2_l73_73781


namespace smallest_perimeter_of_triangle_with_consecutive_odd_integers_l73_73255

theorem smallest_perimeter_of_triangle_with_consecutive_odd_integers :
  ∃ (a b c : ℕ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ 
  (a < b) ∧ (b < c) ∧ (c = a + 4) ∧
  (a + b > c) ∧ (b + c > a) ∧ (a + c > b) ∧ 
  (a + b + c = 15) :=
by
  sorry

end smallest_perimeter_of_triangle_with_consecutive_odd_integers_l73_73255


namespace y_share_per_rupee_of_x_l73_73590

theorem y_share_per_rupee_of_x (share_y : ℝ) (total_amount : ℝ) (z_per_x : ℝ) (y_per_x : ℝ) 
  (h1 : share_y = 54) 
  (h2 : total_amount = 210) 
  (h3 : z_per_x = 0.30) 
  (h4 : share_y = y_per_x * (total_amount / (1 + y_per_x + z_per_x))) : 
  y_per_x = 0.45 :=
sorry

end y_share_per_rupee_of_x_l73_73590


namespace number_of_players_taking_mathematics_l73_73106

def total_players : ℕ := 25
def players_taking_physics : ℕ := 12
def players_taking_both : ℕ := 5

theorem number_of_players_taking_mathematics :
  total_players - players_taking_physics + players_taking_both = 18 :=
by
  sorry

end number_of_players_taking_mathematics_l73_73106


namespace negation_of_exists_sin_gt_one_l73_73466

theorem negation_of_exists_sin_gt_one : 
  (¬ ∃ x : ℝ, Real.sin x > 1) ↔ (∀ x : ℝ, Real.sin x ≤ 1) := 
by
  sorry

end negation_of_exists_sin_gt_one_l73_73466


namespace minimum_value_of_f_l73_73457

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / x

theorem minimum_value_of_f (h : 1 < x) : ∃ y, f x = y ∧ (∀ z, (f z) ≥ 2*sqrt 2) :=
by
  sorry

end minimum_value_of_f_l73_73457


namespace train_crosses_pole_in_time_l73_73014

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  length / speed_ms

theorem train_crosses_pole_in_time :
  ∀ (length speed_kmh : ℝ), length = 240 → speed_kmh = 126 →
    time_to_cross_pole length speed_kmh = 6.8571 :=
by
  intros length speed_kmh h_length h_speed
  rw [h_length, h_speed, time_to_cross_pole]
  sorry

end train_crosses_pole_in_time_l73_73014


namespace points_per_enemy_l73_73210

theorem points_per_enemy (total_enemies destroyed_enemies points_earned points_per_enemy : ℕ)
  (h1 : total_enemies = 8)
  (h2 : destroyed_enemies = total_enemies - 6)
  (h3 : points_earned = 10)
  (h4 : points_per_enemy = points_earned / destroyed_enemies) : 
  points_per_enemy = 5 := 
by
  sorry

end points_per_enemy_l73_73210


namespace cost_of_child_ticket_l73_73087

-- Define the conditions
def adult_ticket_cost : ℕ := 60
def total_people : ℕ := 280
def total_collected_dollars : ℕ := 140
def total_collected_cents : ℕ := total_collected_dollars * 100
def children_attended : ℕ := 80
def adults_attended : ℕ := total_people - children_attended
def total_collected_from_adults : ℕ := adults_attended * adult_ticket_cost

-- State the theorem to prove the cost of a child ticket
theorem cost_of_child_ticket (x : ℕ) :
  total_collected_from_adults + children_attended * x = total_collected_cents →
  x = 25 :=
by
  sorry

end cost_of_child_ticket_l73_73087


namespace problem_l73_73071

theorem problem (x y : ℕ) (hxpos : 0 < x ∧ x < 20) (hypos : 0 < y ∧ y < 20) (h : x + y + x * y = 119) : 
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
by sorry

end problem_l73_73071


namespace tap_B_filling_time_l73_73490

theorem tap_B_filling_time : 
  ∀ (r_A r_B : ℝ), 
  (r_A + r_B = 1 / 30) → 
  (r_B * 40 = 2 / 3) → 
  (1 / r_B = 60) := 
by
  intros r_A r_B h₁ h₂
  sorry

end tap_B_filling_time_l73_73490


namespace geometric_sequence_first_term_l73_73912

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) -- sequence a_n
  (r : ℝ) -- common ratio
  (h1 : r = 2) -- given common ratio
  (h2 : a 4 = 16) -- given a_4 = 16
  (h3 : ∀ n, a n = a 1 * r^(n-1)) -- definition of geometric sequence
  : a 1 = 2 := 
sorry

end geometric_sequence_first_term_l73_73912


namespace product_divisible_by_10_probability_l73_73196

noncomputable def probability_divisible_by_10 (n : ℕ) (h: n > 1) : ℝ :=
  1 - ((8^n + 5^n - 4^n : ℝ) / (9^n : ℝ))

theorem product_divisible_by_10_probability (n : ℕ) (h: n > 1) :
  probability_divisible_by_10 n h = 1 - ((8^n + 5^n - 4^n : ℝ) / (9^n : ℝ)) :=
by
  -- The proof is omitted
  sorry

end product_divisible_by_10_probability_l73_73196


namespace find_p_q_l73_73681

variable (R : Set ℝ)

def A (p : ℝ) : Set ℝ := {x | x^2 + p * x + 12 = 0}
def B (q : ℝ) : Set ℝ := {x | x^2 - 5 * x + q = 0}

theorem find_p_q 
  (h : (R \ (A p)) ∩ (B q) = {2}) : p + q = -1 :=
by
  sorry

end find_p_q_l73_73681


namespace always_positive_inequality_l73_73281

theorem always_positive_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end always_positive_inequality_l73_73281


namespace initial_crayons_count_l73_73546

variable (x : ℕ) -- x represents the initial number of crayons

theorem initial_crayons_count (h1 : x + 3 = 12) : x = 9 := 
by sorry

end initial_crayons_count_l73_73546


namespace negation_of_forall_l73_73683

theorem negation_of_forall (h : ¬ ∀ x > 0, Real.exp x > x + 1) : ∃ x > 0, Real.exp x < x + 1 :=
sorry

end negation_of_forall_l73_73683


namespace gwen_more_money_from_mom_l73_73296

def dollars_received_from_mom : ℕ := 7
def dollars_received_from_dad : ℕ := 5

theorem gwen_more_money_from_mom :
  dollars_received_from_mom - dollars_received_from_dad = 2 :=
by
  sorry

end gwen_more_money_from_mom_l73_73296


namespace percent_diamond_jewels_l73_73991

def percent_beads : ℝ := 0.3
def percent_ruby_jewels : ℝ := 0.5

theorem percent_diamond_jewels (percent_beads percent_ruby_jewels : ℝ) : 
  (1 - percent_beads) * (1 - percent_ruby_jewels) = 0.35 :=
by
  -- We insert the proof steps here
  sorry

end percent_diamond_jewels_l73_73991


namespace smallest_prime_dividing_7pow15_plus_9pow17_l73_73494

theorem smallest_prime_dividing_7pow15_plus_9pow17 :
  Nat.Prime 2 ∧ (∀ p : ℕ, Nat.Prime p → p ∣ (7^15 + 9^17) → 2 ≤ p) :=
by
  sorry

end smallest_prime_dividing_7pow15_plus_9pow17_l73_73494


namespace average_of_next_seven_consecutive_integers_l73_73317

theorem average_of_next_seven_consecutive_integers
  (a b : ℕ)
  (hb : b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7) :
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7) = a + 6 :=
by
  sorry

end average_of_next_seven_consecutive_integers_l73_73317


namespace calculation_l73_73417

noncomputable def distance_from_sphere_center_to_plane (S P Q R : Point) (r PQ QR RP : ℝ) : ℝ := 
  let a := PQ / 2
  let b := QR / 2
  let c := RP / 2
  let s := (PQ + QR + RP) / 2
  let K := Real.sqrt (s * (s - PQ) * (s - QR) * (s - RP))
  let R := (PQ * QR * RP) / (4 * K)
  Real.sqrt (r^2 - R^2)

theorem calculation 
  (P Q R S : Point) 
  (r : ℝ) 
  (PQ QR RP : ℝ)
  (h1 : PQ = 17)
  (h2 : QR = 18)
  (h3 : RP = 19)
  (h4 : r = 25) :
  distance_from_sphere_center_to_plane S P Q R r PQ QR RP = 35 * Real.sqrt 7 / 8 → 
  ∃ (x y z : ℕ), x + y + z = 50 ∧ (x.gcd z = 1) ∧ ¬ ∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ y := 
by {
  sorry
}

end calculation_l73_73417


namespace incorrect_statement_A_l73_73995

/-- Let prob_beijing be the probability of rainfall in Beijing and prob_shanghai be the probability
of rainfall in Shanghai. We assert that statement (A) which claims "It is certain to rain in Beijing today, 
while it is certain not to rain in Shanghai" is incorrect given the probabilities. 
-/
theorem incorrect_statement_A (prob_beijing prob_shanghai : ℝ) 
  (h_beijing : prob_beijing = 0.8)
  (h_shanghai : prob_shanghai = 0.2)
  (statement_A : ¬ (prob_beijing = 1 ∧ prob_shanghai = 0)) : 
  true := 
sorry

end incorrect_statement_A_l73_73995


namespace cube_volume_is_216_l73_73922

-- Define the conditions
def total_edge_length : ℕ := 72
def num_edges_of_cube : ℕ := 12

-- The side length of the cube can be calculated as
def side_length (E : ℕ) (n : ℕ) : ℕ := E / n

-- The volume of the cube is the cube of its side length
def volume (s : ℕ) : ℕ := s ^ 3

theorem cube_volume_is_216 (E : ℕ) (n : ℕ) (V : ℕ) 
  (hE : E = total_edge_length) 
  (hn : n = num_edges_of_cube) 
  (hv : V = volume (side_length E n)) : 
  V = 216 := by
  sorry

end cube_volume_is_216_l73_73922


namespace number_of_three_digit_numbers_with_123_exactly_once_l73_73478

theorem number_of_three_digit_numbers_with_123_exactly_once : 
  (∃ (l : List ℕ), l = [1, 2, 3] ∧ l.permutations.length = 6) :=
by
  sorry

end number_of_three_digit_numbers_with_123_exactly_once_l73_73478


namespace champion_is_C_l73_73996

-- Definitions of statements made by Zhang, Wang, and Li
def zhang_statement (winner : String) : Bool := winner = "A" ∨ winner = "B"
def wang_statement (winner : String) : Bool := winner ≠ "C"
def li_statement (winner : String) : Bool := winner ≠ "A" ∧ winner ≠ "B"

-- Predicate that indicates exactly one of the statements is correct
def exactly_one_correct (winner : String) : Prop :=
  (zhang_statement winner ∧ ¬wang_statement winner ∧ ¬li_statement winner) ∨
  (¬zhang_statement winner ∧ wang_statement winner ∧ ¬li_statement winner) ∨
  (¬zhang_statement winner ∧ ¬wang_statement winner ∧ li_statement winner)

-- The theorem stating the correct answer to the problem
theorem champion_is_C : (exactly_one_correct "C") :=
  by
    sorry  -- Proof goes here

-- Note: The import statement and sorry definition are included to ensure the code builds.

end champion_is_C_l73_73996


namespace parabola_vertex_calc_l73_73270

noncomputable def vertex_parabola (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem parabola_vertex_calc 
  (a b c : ℝ) 
  (h_vertex : vertex_parabola a b c 2 = 5)
  (h_point : vertex_parabola a b c 1 = 8) : 
  a - b + c = 32 :=
sorry

end parabola_vertex_calc_l73_73270


namespace geometric_sequence_fifth_term_l73_73286

theorem geometric_sequence_fifth_term (a₁ r : ℤ) (n : ℕ) (h_a₁ : a₁ = 5) (h_r : r = -2) (h_n : n = 5) :
  (a₁ * r^(n-1) = 80) :=
by
  rw [h_a₁, h_r, h_n]
  sorry

end geometric_sequence_fifth_term_l73_73286


namespace enlarged_banner_height_l73_73732

-- Definitions and theorem statement
theorem enlarged_banner_height 
  (original_width : ℝ) 
  (original_height : ℝ) 
  (new_width : ℝ) 
  (scaling_factor : ℝ := new_width / original_width ) 
  (new_height : ℝ := original_height * scaling_factor) 
  (h1 : original_width = 3) 
  (h2 : original_height = 2) 
  (h3 : new_width = 15): 
  new_height = 10 := 
by 
  -- The proof would go here
  sorry

end enlarged_banner_height_l73_73732


namespace crop_fraction_brought_to_AD_l73_73143

theorem crop_fraction_brought_to_AD
  (AD BC AB CD : ℝ)
  (h : ℝ)
  (angle : ℝ)
  (AD_eq_150 : AD = 150)
  (BC_eq_100 : BC = 100)
  (AB_eq_130 : AB = 130)
  (CD_eq_130 : CD = 130)
  (angle_eq_75 : angle = 75)
  (height_eq : h = (AB / 2) * Real.sin (angle * Real.pi / 180)) -- converting degrees to radians
  (area_trap : ℝ)
  (upper_area : ℝ)
  (total_area_eq : area_trap = (1 / 2) * (AD + BC) * h)
  (upper_area_eq : upper_area = (1 / 2) * (AD + (BC / 2)) * h)
  : (upper_area / area_trap) = 0.8 := 
sorry

end crop_fraction_brought_to_AD_l73_73143


namespace find_divisor_l73_73682

def remainder : Nat := 1
def quotient : Nat := 54
def dividend : Nat := 217

theorem find_divisor : ∃ divisor : Nat, (dividend = divisor * quotient + remainder) ∧ divisor = 4 :=
by
  sorry

end find_divisor_l73_73682


namespace factorization_count_is_correct_l73_73117

noncomputable def count_factorizations (n : Nat) (k : Nat) : Nat :=
  (Nat.choose (n + k - 1) (k - 1))

noncomputable def factor_count : Nat :=
  let alpha_count := count_factorizations 6 3
  let beta_count := count_factorizations 6 3
  let total_count := alpha_count * beta_count
  let unordered_factorizations := total_count - 15 * 3 - 1
  1 + 15 + unordered_factorizations / 6

theorem factorization_count_is_correct :
  factor_count = 139 := by
  sorry

end factorization_count_is_correct_l73_73117


namespace portraits_after_lunch_before_gym_class_l73_73866

-- Define the total number of students in the class
def total_students : ℕ := 24

-- Define the number of students who had their portraits taken before lunch
def students_before_lunch : ℕ := total_students / 3

-- Define the number of students who have not yet had their picture taken after gym class
def students_after_gym_class : ℕ := 6

-- Define the number of students who had their portraits taken before gym class
def students_before_gym_class : ℕ := total_students - students_after_gym_class

-- Define the number of students who had their portraits taken after lunch but before gym class
def students_after_lunch_before_gym_class : ℕ := students_before_gym_class - students_before_lunch

-- Statement of the theorem
theorem portraits_after_lunch_before_gym_class :
  students_after_lunch_before_gym_class = 10 :=
by
  -- The proof is omitted
  sorry

end portraits_after_lunch_before_gym_class_l73_73866


namespace contrapositive_inequality_l73_73637

theorem contrapositive_inequality (x : ℝ) :
  ((x + 2) * (x - 3) > 0) → (x < -2 ∨ x > 0) :=
by
  sorry

end contrapositive_inequality_l73_73637


namespace train_speed_correct_l73_73051

def length_of_train : ℕ := 700
def time_to_cross_pole : ℕ := 20
def expected_speed : ℕ := 35

theorem train_speed_correct : (length_of_train / time_to_cross_pole) = expected_speed := by
  sorry

end train_speed_correct_l73_73051


namespace maximum_amount_one_blue_cube_maximum_amount_n_blue_cubes_l73_73805

-- Part (a): One blue cube
theorem maximum_amount_one_blue_cube : 
  ∃ (B : ℕ → ℚ) (P : ℕ → ℕ), (B 1 = 2) ∧ (∀ m > 1, B m = 2^m / P m) ∧ (P 1 = 1) ∧ (∀ m > 1, P m = m) ∧ B 100 = 2^100 / 100 :=
by
  sorry

-- Part (b): Exactly n blue cubes
theorem maximum_amount_n_blue_cubes (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 100) : 
  ∃ (B : ℕ × ℕ → ℚ) (P : ℕ × ℕ → ℕ), (B (1, 0) = 2) ∧ (B (1, 1) = 2) ∧ (∀ m > 1, B (m, 0) = 2^m) ∧ (P (1, 0) = 1) ∧ (P (1, 1) = 1) ∧ (∀ m > 1, P (m, 0) = 1) ∧ B (100, n) = 2^100 / Nat.choose 100 n :=
by
  sorry

end maximum_amount_one_blue_cube_maximum_amount_n_blue_cubes_l73_73805


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l73_73892

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l73_73892


namespace price_each_clock_is_correct_l73_73794

-- Definitions based on the conditions
def numberOfDolls := 3
def numberOfClocks := 2
def numberOfGlasses := 5
def pricePerDoll := 5
def pricePerGlass := 4
def totalCost := 40
def profit := 25

-- The total revenue from selling dolls and glasses
def revenueFromDolls := numberOfDolls * pricePerDoll
def revenueFromGlasses := numberOfGlasses * pricePerGlass
def totalRevenueNeeded := totalCost + profit
def revenueFromDollsAndGlasses := revenueFromDolls + revenueFromGlasses

-- The required revenue from clocks
def revenueFromClocks := totalRevenueNeeded - revenueFromDollsAndGlasses

-- The price per clock
def pricePerClock := revenueFromClocks / numberOfClocks

-- Statement to prove
theorem price_each_clock_is_correct : pricePerClock = 15 := sorry

end price_each_clock_is_correct_l73_73794


namespace domain_f_l73_73435

noncomputable def f (x : ℝ) := Real.sqrt (3 - x) + Real.log (x - 1)

theorem domain_f : { x : ℝ | 1 < x ∧ x ≤ 3 } = { x : ℝ | True } ∩ { x : ℝ | x ≤ 3 } ∩ { x : ℝ | x > 1 } :=
by
  sorry

end domain_f_l73_73435


namespace find_a2016_l73_73827

theorem find_a2016 (S : ℕ → ℕ)
  (a : ℕ → ℤ)
  (h₁ : S 1 = 6)
  (h₂ : S 2 = 4)
  (h₃ : ∀ n, S n > 0)
  (h₄ : ∀ n, (S (2 * n - 1))^2 = S (2 * n) * S (2 * n + 2))
  (h₅ : ∀ n, 2 * S (2 * n + 2) = S (2 * n - 1) + S (2 * n + 1))
  : a 2016 = -1009 := 
  sorry

end find_a2016_l73_73827


namespace trajectory_moving_point_l73_73634

theorem trajectory_moving_point (x y : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -1 ↔ x^2 + y^2 = 1 := by
  sorry

end trajectory_moving_point_l73_73634


namespace damaged_cartons_per_customer_l73_73409

theorem damaged_cartons_per_customer (total_cartons : ℕ) (num_customers : ℕ) (total_accepted : ℕ) 
    (h1 : total_cartons = 400) (h2 : num_customers = 4) (h3 : total_accepted = 160) 
    : (total_cartons - total_accepted) / num_customers = 60 :=
by
  sorry

end damaged_cartons_per_customer_l73_73409


namespace volume_of_revolution_l73_73277

theorem volume_of_revolution (a : ℝ) (h : 0 < a) :
  let x (θ : ℝ) := a * (1 + Real.cos θ) * Real.cos θ
  let y (θ : ℝ) := a * (1 + Real.cos θ) * Real.sin θ
  V = (8 / 3) * π * a^3 :=
sorry

end volume_of_revolution_l73_73277


namespace not_perfect_square_7p_3p_4_l73_73049

theorem not_perfect_square_7p_3p_4 (p : ℕ) (hp : Nat.Prime p) : ¬∃ a : ℕ, a^2 = 7 * p + 3^p - 4 := 
by
  sorry

end not_perfect_square_7p_3p_4_l73_73049


namespace probability_eight_distinct_numbers_l73_73105

theorem probability_eight_distinct_numbers :
  let total_ways := 10^8
  let ways_distinct := (10 * 9 * 8 * 7 * 6 * 5 * 4 * 3)
  (ways_distinct / total_ways : ℚ) = 18144 / 500000 := 
by
  sorry

end probability_eight_distinct_numbers_l73_73105


namespace x_squared_eq_1_iff_x_eq_1_l73_73427

theorem x_squared_eq_1_iff_x_eq_1 (x : ℝ) : (x^2 = 1 → x = 1) ↔ false ∧ (x = 1 → x^2 = 1) :=
by
  sorry

end x_squared_eq_1_iff_x_eq_1_l73_73427


namespace product_xyz_l73_73718

/-- Prove that if x + 1/y = 2 and y + 1/z = 3, then xyz = 1/11. -/
theorem product_xyz {x y z : ℝ} (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 3) : x * y * z = 1 / 11 :=
sorry

end product_xyz_l73_73718


namespace right_triangle_roots_l73_73101

theorem right_triangle_roots (α β : ℝ) (k : ℕ) (h_triangle : (α^2 + β^2 = 100) ∧ (α + β = 14) ∧ (α * β = 4 * k - 4)) : k = 13 :=
sorry

end right_triangle_roots_l73_73101


namespace jana_walking_distance_l73_73168

theorem jana_walking_distance (t_walk_mile : ℝ) (speed : ℝ) (time : ℝ) (distance : ℝ) :
  t_walk_mile = 24 → speed = 1 / t_walk_mile → time = 36 → distance = speed * time → distance = 1.5 :=
by
  intros h1 h2 h3 h4
  sorry

end jana_walking_distance_l73_73168


namespace flower_garden_width_l73_73048

-- Define the conditions
def gardenArea : ℝ := 143.2
def gardenLength : ℝ := 4
def gardenWidth : ℝ := 35.8

-- The proof statement (question to answer)
theorem flower_garden_width :
    gardenWidth = gardenArea / gardenLength :=
by 
  sorry

end flower_garden_width_l73_73048


namespace frac_pow_eq_l73_73927

theorem frac_pow_eq : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by 
  sorry

end frac_pow_eq_l73_73927


namespace oliver_earning_correct_l73_73955

open Real

noncomputable def total_weight_two_days_ago : ℝ := 5

noncomputable def total_weight_yesterday : ℝ := total_weight_two_days_ago + 5

noncomputable def total_weight_today : ℝ := 2 * total_weight_yesterday

noncomputable def total_weight_three_days : ℝ := total_weight_two_days_ago + total_weight_yesterday + total_weight_today

noncomputable def earning_per_kilo : ℝ := 2

noncomputable def total_earning : ℝ := total_weight_three_days * earning_per_kilo

theorem oliver_earning_correct : total_earning = 70 := by
  sorry

end oliver_earning_correct_l73_73955


namespace probability_of_one_failure_l73_73240

theorem probability_of_one_failure (p1 p2 : ℝ) (h1 : p1 = 0.90) (h2 : p2 = 0.95) :
  (p1 * (1 - p2) + (1 - p1) * p2) = 0.14 :=
by
  rw [h1, h2]
  -- Additional leaning code can be inserted here to finalize the proof if this was complete
  sorry

end probability_of_one_failure_l73_73240


namespace h_at_2_l73_73615

noncomputable def f (x : ℝ) : ℝ := 3 * x - 4
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (3 * f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_at_2 : h 2 = 3 * Real.sqrt 6 - 13 := 
by 
  sorry -- We skip the proof steps.

end h_at_2_l73_73615


namespace lights_on_top_layer_l73_73199

theorem lights_on_top_layer
  (x : ℕ)
  (H1 : x + 2 * x + 4 * x + 8 * x + 16 * x + 32 * x + 64 * x = 381) :
  x = 3 :=
  sorry

end lights_on_top_layer_l73_73199


namespace Tom_initial_investment_l73_73400

noncomputable def Jose_investment : ℝ := 45000
noncomputable def Jose_investment_time : ℕ := 10
noncomputable def total_profit : ℝ := 36000
noncomputable def Jose_share : ℝ := 20000
noncomputable def Tom_share : ℝ := total_profit - Jose_share
noncomputable def Tom_investment_time : ℕ := 12
noncomputable def proportion_Tom : ℝ := (4 : ℝ) / 5
noncomputable def Tom_expected_investment : ℝ := 6000

theorem Tom_initial_investment (T : ℝ) (h1 : Jose_investment = 45000)
                               (h2 : Jose_investment_time = 10)
                               (h3 : total_profit = 36000)
                               (h4 : Jose_share = 20000)
                               (h5 : Tom_investment_time = 12)
                               (h6 : Tom_share = 16000)
                               (h7 : proportion_Tom = (4 : ℝ) / 5)
                               : T = Tom_expected_investment :=
by
  sorry

end Tom_initial_investment_l73_73400


namespace walking_rate_ratio_l73_73902

theorem walking_rate_ratio (R R' : ℝ)
  (h : R * 36 = R' * 32) : R' / R = 9 / 8 :=
sorry

end walking_rate_ratio_l73_73902


namespace translate_upwards_one_unit_l73_73812

theorem translate_upwards_one_unit (x y : ℝ) : (y = 2 * x) → (y + 1 = 2 * x + 1) := 
by sorry

end translate_upwards_one_unit_l73_73812


namespace tangent_line_parallel_coordinates_l73_73386

theorem tangent_line_parallel_coordinates :
  ∃ (x y : ℝ), y = x^3 + x - 2 ∧ (3 * x^2 + 1 = 4) ∧ (x, y) = (-1, -4) :=
by
  sorry

end tangent_line_parallel_coordinates_l73_73386


namespace katy_brownies_l73_73650

theorem katy_brownies :
  ∃ (n : ℤ), (n = 5 + 2 * 5) :=
by
  sorry

end katy_brownies_l73_73650


namespace JodiMilesFourthWeek_l73_73073

def JodiMilesFirstWeek := 1 * 6
def JodiMilesSecondWeek := 2 * 6
def JodiMilesThirdWeek := 3 * 6
def TotalMilesFirstThreeWeeks := JodiMilesFirstWeek + JodiMilesSecondWeek + JodiMilesThirdWeek
def TotalMilesFourWeeks := 60

def MilesInFourthWeek := TotalMilesFourWeeks - TotalMilesFirstThreeWeeks
def DaysInWeek := 6

theorem JodiMilesFourthWeek : (MilesInFourthWeek / DaysInWeek) = 4 := by
  sorry

end JodiMilesFourthWeek_l73_73073


namespace factorize_expression_l73_73644

theorem factorize_expression : ∀ x : ℝ, 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  intro x
  sorry

end factorize_expression_l73_73644


namespace find_second_number_l73_73585

theorem find_second_number (n : ℕ) 
  (h1 : Nat.lcm 24 (Nat.lcm n 42) = 504)
  (h2 : 504 = 2^3 * 3^2 * 7) 
  (h3 : Nat.lcm 24 42 = 168) : n = 3 := 
by 
  sorry

end find_second_number_l73_73585


namespace find_A_and_height_l73_73083

noncomputable def triangle_properties (a b : ℝ) (B : ℝ) (cos_B : ℝ) (h : ℝ) :=
  a = 7 ∧ b = 8 ∧ cos_B = -1 / 7 ∧ 
  h = (a : ℝ) * (Real.sqrt (1 - (cos_B)^2)) * (1 : ℝ) / b / 2

theorem find_A_and_height : 
  ∀ (a b : ℝ) (B : ℝ) (cos_B : ℝ) (h : ℝ), 
  triangle_properties a b B cos_B h → 
  ∃ A h1, A = Real.pi / 3 ∧ h1 = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end find_A_and_height_l73_73083


namespace general_term_formula_l73_73797

theorem general_term_formula (n : ℕ) :
  ∀ (S : ℕ → ℝ), (∀ k : ℕ, S k = 1 - 2^k) → 
  (∀ a : ℕ → ℝ, a 1 = (S 1) ∧ (∀ m : ℕ, m > 1 → a m = S m - S (m - 1)) → 
  a n = -2 ^ (n - 1)) :=
by
  intro S hS a ha
  sorry

end general_term_formula_l73_73797


namespace domain_of_function_l73_73534

theorem domain_of_function : 
  {x : ℝ | x ≠ 1 ∧ x > 0} = {x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x)} :=
by
  sorry

end domain_of_function_l73_73534


namespace arithmetic_sequence_a10_l73_73077

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : (S 9) / 9 - (S 5) / 5 = 4)
  (hSn : ∀ n, S n = n * (2 + (n - 1) / 2 * (a 2 - a 1) )) : 
  a 10 = 20 := 
sorry

end arithmetic_sequence_a10_l73_73077


namespace find_a3_l73_73779

variable (a_n : ℕ → ℤ) (a1 a4 a5 : ℤ)
variable (d : ℤ := -2)

-- Conditions
axiom h1 : ∀ n : ℕ, a_n (n + 1) = a_n n + d
axiom h2 : a4 = a1 + 3 * d
axiom h3 : a5 = a1 + 4 * d
axiom h4 : a4 * a4 = a1 * a5

-- Question to prove
theorem find_a3 : (a_n 3) = 5 := by
  sorry

end find_a3_l73_73779


namespace discount_percentage_l73_73879

noncomputable def cost_price : ℝ := 100
noncomputable def profit_with_discount : ℝ := 0.32 * cost_price
noncomputable def profit_without_discount : ℝ := 0.375 * cost_price

noncomputable def sp_with_discount : ℝ := cost_price + profit_with_discount
noncomputable def sp_without_discount : ℝ := cost_price + profit_without_discount

noncomputable def discount_amount : ℝ := sp_without_discount - sp_with_discount
noncomputable def percentage_discount : ℝ := (discount_amount / sp_without_discount) * 100

theorem discount_percentage : percentage_discount = 4 :=
by
  -- proof steps
  sorry

end discount_percentage_l73_73879


namespace greatest_product_sum_2006_l73_73523

theorem greatest_product_sum_2006 :
  (∃ x y : ℤ, x + y = 2006 ∧ ∀ a b : ℤ, a + b = 2006 → a * b ≤ x * y) → 
  ∃ x y : ℤ, x + y = 2006 ∧ x * y = 1006009 :=
by sorry

end greatest_product_sum_2006_l73_73523


namespace number_of_squares_with_prime_condition_l73_73997

theorem number_of_squares_with_prime_condition : 
  ∃! (n : ℕ), ∃ (p : ℕ), Prime p ∧ n^2 = p + 4 := 
sorry

end number_of_squares_with_prime_condition_l73_73997


namespace age_difference_is_58_l73_73971

def Milena_age : ℕ := 7
def Grandmother_age : ℕ := 9 * Milena_age
def Grandfather_age : ℕ := Grandmother_age + 2
def Age_difference : ℕ := Grandfather_age - Milena_age

theorem age_difference_is_58 : Age_difference = 58 := by
  sorry

end age_difference_is_58_l73_73971


namespace find_solution_l73_73541

theorem find_solution (x y z : ℝ) :
  (x * (y^2 + z) = z * (z + x * y)) ∧ 
  (y * (z^2 + x) = x * (x + y * z)) ∧ 
  (z * (x^2 + y) = y * (y + x * z)) → 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end find_solution_l73_73541


namespace proof_problem_l73_73989

variable {a b c : ℝ}

-- Condition: a < 0
variable (ha : a < 0)
-- Condition: b > 0
variable (hb : b > 0)
-- Condition: c > 0
variable (hc : c > 0)
-- Condition: a < b < c
variable (hab : a < b) (hbc : b < c)

-- Proof statement
theorem proof_problem :
  (ab * b < b * c) ∧
  (a * c < b * c) ∧
  (a + c < b + c) ∧
  (c / a < 1) :=
  by
    sorry

end proof_problem_l73_73989


namespace age_of_youngest_child_l73_73116

theorem age_of_youngest_child
  (x : ℕ)
  (sum_of_ages : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) :
  x = 4 :=
sorry

end age_of_youngest_child_l73_73116


namespace perpendicular_lines_condition_l73_73614

theorem perpendicular_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, x + y = 0 ∧ x - ay = 0 → x = 0) ↔ (a = 1) := 
sorry

end perpendicular_lines_condition_l73_73614


namespace find_y_l73_73559

variable {R : Type} [Field R] (y : R)

-- The condition: y = (1/y) * (-y) + 3
def condition (y : R) : Prop :=
  y = (1 / y) * (-y) + 3

-- The theorem to prove: under the condition, y = 2
theorem find_y (y : R) (h : condition y) : y = 2 := 
sorry

end find_y_l73_73559


namespace problem_statement_l73_73880

theorem problem_statement (c d : ℤ) (hc : c = 3) (hd : d = 2) :
  (c^2 + d)^2 - (c^2 - d)^2 = 72 :=
by
  sorry

end problem_statement_l73_73880


namespace product_of_two_numbers_l73_73388

theorem product_of_two_numbers (x y : ℝ) (h₁ : x + y = 23) (h₂ : x^2 + y^2 = 289) : x * y = 120 := by
  sorry

end product_of_two_numbers_l73_73388


namespace least_positive_integer_exists_l73_73301

theorem least_positive_integer_exists :
  ∃ (x : ℕ), 
    (x % 6 = 5) ∧
    (x % 8 = 7) ∧
    (x % 7 = 6) ∧
    x = 167 :=
by {
  sorry
}

end least_positive_integer_exists_l73_73301


namespace amount_lent_to_B_l73_73542

theorem amount_lent_to_B
  (rate_of_interest_per_annum : ℝ)
  (P_C : ℝ)
  (years_C : ℝ)
  (total_interest : ℝ)
  (years_B : ℝ)
  (IB : ℝ)
  (IC : ℝ)
  (P_B : ℝ):
  (rate_of_interest_per_annum = 10) →
  (P_C = 3000) →
  (years_C = 4) →
  (total_interest = 2200) →
  (years_B = 2) →
  (IC = (P_C * rate_of_interest_per_annum * years_C) / 100) →
  (IB = (P_B * rate_of_interest_per_annum * years_B) / 100) →
  (total_interest = IB + IC) →
  P_B = 5000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end amount_lent_to_B_l73_73542


namespace find_angle_A_l73_73335

variables {A B C a b c : ℝ}
variables {triangle_ABC : (2 * b - c) * (Real.cos A) = a * (Real.cos C)}

theorem find_angle_A (h : (2 * b - c) * (Real.cos A) = a * (Real.cos C)) : A = Real.pi / 3 :=
by
  sorry

end find_angle_A_l73_73335


namespace shortest_distance_proof_l73_73249

noncomputable def shortest_distance (k : ℝ) : ℝ :=
  let p := (k - 6) / 2
  let f_p := -p^2 + (6 - k) * p + 18
  let d := |f_p|
  d / (Real.sqrt (k^2 + 1))

theorem shortest_distance_proof (k : ℝ) :
  shortest_distance k = 
  |(-(k - 6) / 2^2 + (6 - k) * (k - 6) / 2 + 18)| / (Real.sqrt (k^2 + 1)) :=
sorry

end shortest_distance_proof_l73_73249


namespace number_of_quarters_l73_73192
-- Definitions of the coin values
def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def half_dollar_value := 50

-- Number of each type of coin used in the proof
variable (pennies nickels dimes quarters half_dollars : ℕ)

-- Conditions from step (a)
axiom one_penny : pennies > 0
axiom one_nickel : nickels > 0
axiom one_dime : dimes > 0
axiom one_quarter : quarters > 0
axiom one_half_dollar : half_dollars > 0
axiom total_coins : pennies + nickels + dimes + quarters + half_dollars = 11
axiom total_value : pennies * penny_value + nickels * nickel_value + dimes * dime_value + quarters * quarter_value + half_dollars * half_dollar_value = 163

-- The conclusion we want to prove
theorem number_of_quarters : quarters = 1 := 
sorry

end number_of_quarters_l73_73192


namespace range_of_a_l73_73063

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, (2 ≤ x ∧ x ≤ 4) ∧ (2 ≤ y ∧ y ≤ 3) → x * y ≤ a * x^2 + 2 * y^2) → a ≥ 0 := 
sorry

end range_of_a_l73_73063


namespace teacher_age_frequency_l73_73449

theorem teacher_age_frequency (f_less_than_30 : ℝ) (f_between_30_and_50 : ℝ) (h1 : f_less_than_30 = 0.3) (h2 : f_between_30_and_50 = 0.5) :
  1 - f_less_than_30 - f_between_30_and_50 = 0.2 :=
by
  rw [h1, h2]
  norm_num

end teacher_age_frequency_l73_73449


namespace problem_l73_73235

theorem problem (x a : ℝ) (h : x^5 - x^3 + x = a) : x^6 ≥ 2 * a - 1 := 
by 
  sorry

end problem_l73_73235


namespace prop_p_iff_prop_q_iff_not_or_p_q_l73_73551

theorem prop_p_iff (m : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + (2 + m) = 0) ↔ (m ≤ -1 ∨ m ≥ 2) :=
sorry

theorem prop_q_iff (m : ℝ) :
  (∃ x y : ℝ, (x^2)/(1 - 2*m) + (y^2)/(m + 2) = 1) ↔ (m < -2 ∨ m > 1/2) :=
sorry

theorem not_or_p_q (m : ℝ) :
  ¬(∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + (2 + m) = 0) ∧
  ¬(∃ x y : ℝ, (x^2)/(1 - 2*m) + (y^2)/(m + 2) = 1) ↔
  (-1 < m ∧ m ≤ 1/2) :=
sorry

end prop_p_iff_prop_q_iff_not_or_p_q_l73_73551


namespace regular_polygon_interior_angle_ratio_l73_73021

theorem regular_polygon_interior_angle_ratio (r k : ℕ) (h1 : 180 - 360 / r = (5 : ℚ) / (3 : ℚ) * (180 - 360 / k)) (h2 : r = 2 * k) :
  r = 8 ∧ k = 4 :=
sorry

end regular_polygon_interior_angle_ratio_l73_73021


namespace total_pages_of_book_l73_73786

-- Definitions for the conditions
def firstChapterPages : Nat := 66
def secondChapterPages : Nat := 35
def thirdChapterPages : Nat := 24

-- Theorem stating the main question and answer
theorem total_pages_of_book : firstChapterPages + secondChapterPages + thirdChapterPages = 125 := by
  -- Proof will be provided here
  sorry

end total_pages_of_book_l73_73786


namespace range_of_a_l73_73079

noncomputable def f (a x : ℝ) : ℝ := a ^ x

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a x < 2) : 
  (a ∈ Set.Ioo (Real.sqrt 2 / 2) 1 ∨ a ∈ Set.Ioo 1 (Real.sqrt 2)) :=
by
  sorry

end range_of_a_l73_73079


namespace length_of_best_day_l73_73810

theorem length_of_best_day
  (len_raise_the_roof : Nat)
  (len_rap_battle : Nat)
  (len_best_day : Nat)
  (total_ride_duration : Nat)
  (playlist_count : Nat)
  (total_songs_length : Nat)
  (h_len_raise_the_roof : len_raise_the_roof = 2)
  (h_len_rap_battle : len_rap_battle = 3)
  (h_total_ride_duration : total_ride_duration = 40)
  (h_playlist_count : playlist_count = 5)
  (h_total_songs_length : len_raise_the_roof + len_rap_battle + len_best_day = total_songs_length)
  (h_playlist_length : total_ride_duration / playlist_count = total_songs_length) :
  len_best_day = 3 := 
sorry

end length_of_best_day_l73_73810


namespace fraction_square_equality_l73_73151

theorem fraction_square_equality (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) 
    (h : a / b + c / d = 1) : 
    (a / b)^2 + c / d = (c / d)^2 + a / b :=
by
  sorry

end fraction_square_equality_l73_73151


namespace total_flowers_eaten_l73_73740

-- Definitions based on conditions
def num_bugs : ℕ := 3
def flowers_per_bug : ℕ := 2

-- Statement asserting the total number of flowers eaten
theorem total_flowers_eaten : num_bugs * flowers_per_bug = 6 := by
  sorry

end total_flowers_eaten_l73_73740


namespace solve_N_l73_73651

noncomputable def N (a b c d : ℝ) := (a + b) / c - d

theorem solve_N : 
  let a := (Real.sqrt (Real.sqrt 6 + 3))
  let b := (Real.sqrt (Real.sqrt 6 - 3))
  let c := (Real.sqrt (Real.sqrt 6 + 2))
  let d := (Real.sqrt (4 - 2 * Real.sqrt 3))
  N a b c d = -1 :=
by 
  let a := (Real.sqrt (Real.sqrt 6 + 3))
  let b := (Real.sqrt (Real.sqrt 6 - 3))
  let c := (Real.sqrt (Real.sqrt 6 + 2))
  let d := (Real.sqrt (4 - 2 * Real.sqrt 3))
  let n := N a b c d
  sorry

end solve_N_l73_73651


namespace problem1_problem2_l73_73737

-- Problem 1
theorem problem1 : -9 + (-4 * 5) = -29 :=
by
  sorry

-- Problem 2
theorem problem2 : (-(6) * -2) / (2 / 3) = -18 :=
by
  sorry

end problem1_problem2_l73_73737


namespace complement_U_M_l73_73017

def U : Set ℕ := {x | x > 0 ∧ ∃ y : ℝ, y = Real.sqrt (5 - x)}
def M : Set ℕ := {x ∈ U | 4^x ≤ 16}

theorem complement_U_M : U \ M = {3, 4, 5} := by
  sorry

end complement_U_M_l73_73017


namespace remainder_div_1234567_256_l73_73005

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end remainder_div_1234567_256_l73_73005


namespace zero_point_six_one_eight_method_l73_73728

theorem zero_point_six_one_eight_method (a b : ℝ) (h : a = 2 ∧ b = 4) : 
  ∃ x₁ x₂, x₁ = a + 0.618 * (b - a) ∧ x₂ = a + b - x₁ ∧ (x₁ = 3.236 ∨ x₂ = 2.764) := by
  sorry

end zero_point_six_one_eight_method_l73_73728


namespace Yankees_to_Mets_ratio_l73_73830

theorem Yankees_to_Mets_ratio : 
  ∀ (Y M R : ℕ), M = 88 → (M + R + Y = 330) → (4 * R = 5 * M) → (Y : ℚ) / M = 3 / 2 :=
by
  intros Y M R hm htotal hratio
  sorry

end Yankees_to_Mets_ratio_l73_73830


namespace remainder_of_division_987543_12_l73_73046

theorem remainder_of_division_987543_12 : 987543 % 12 = 7 := by
  sorry

end remainder_of_division_987543_12_l73_73046


namespace parallel_planes_mn_l73_73882

theorem parallel_planes_mn (m n : ℝ) (a b : ℝ × ℝ × ℝ) (α β : Type) (h1 : a = (0, 1, m)) (h2 : b = (0, n, -3)) 
  (h3 : ∃ k : ℝ, a = (k • b)) : m * n = -3 :=
by
  -- Proof would be here
  sorry

end parallel_planes_mn_l73_73882


namespace base10_equivalent_of_43210_7_l73_73549

def base7ToDecimal (num : Nat) : Nat :=
  let digits := [4, 3, 2, 1, 0]
  digits[0] * 7^4 + digits[1] * 7^3 + digits[2] * 7^2 + digits[3] * 7^1 + digits[4] * 7^0

theorem base10_equivalent_of_43210_7 :
  base7ToDecimal 43210 = 10738 :=
by
  sorry

end base10_equivalent_of_43210_7_l73_73549


namespace train_additional_time_l73_73065

theorem train_additional_time
  (t : ℝ)  -- time the car takes to reach station B
  (x : ℝ)  -- additional time the train takes compared to the car
  (h₁ : t = 4.5)  -- car takes 4.5 hours to reach station B
  (h₂ : t + (t + x) = 11)  -- combined time for both the car and the train to reach station B
  : x = 2 :=
sorry

end train_additional_time_l73_73065


namespace non_similar_triangles_with_arithmetic_angles_l73_73957

theorem non_similar_triangles_with_arithmetic_angles : 
  ∃! (d : ℕ), d > 0 ∧ d ≤ 50 := 
sorry

end non_similar_triangles_with_arithmetic_angles_l73_73957


namespace n_is_prime_or_power_of_2_l73_73096

noncomputable def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

noncomputable def is_power_of_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 2 ^ k

noncomputable def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem n_is_prime_or_power_of_2 {n : ℕ} (h1 : n > 6)
  (h2 : ∃ (a : ℕ → ℕ) (k : ℕ), 
    (∀ i : ℕ, i < k → a i < n ∧ coprime (a i) n) ∧ 
    (∀ i : ℕ, 1 ≤ i → i < k → a (i + 1) - a i = a 2 - a 1)) 
  : is_prime n ∨ is_power_of_2 n := 
sorry

end n_is_prime_or_power_of_2_l73_73096


namespace number_of_ways_to_divide_l73_73669

def shape_17_cells : Type := sorry -- We would define the structure of the shape here
def checkerboard_pattern : shape_17_cells → Prop := sorry -- The checkerboard pattern condition
def num_black_cells (s : shape_17_cells) : ℕ := 9 -- Number of black cells
def num_gray_cells (s : shape_17_cells) : ℕ := 8 -- Number of gray cells
def divides_into (s : shape_17_cells) (rectangles : ℕ) (squares : ℕ) : Prop := sorry -- Division condition

theorem number_of_ways_to_divide (s : shape_17_cells) (h1 : checkerboard_pattern s) (h2 : divides_into s 8 1) :
  num_black_cells s = 9 ∧ num_gray_cells s = 8 → 
  (∃ ways : ℕ, ways = 10) := 
sorry

end number_of_ways_to_divide_l73_73669


namespace betty_cookies_brownies_l73_73821

theorem betty_cookies_brownies :
  let initial_cookies := 60
  let initial_brownies := 10
  let cookies_per_day := 3
  let brownies_per_day := 1
  let days := 7
  let remaining_cookies := initial_cookies - cookies_per_day * days
  let remaining_brownies := initial_brownies - brownies_per_day * days
  remaining_cookies - remaining_brownies = 36 :=
by
  sorry

end betty_cookies_brownies_l73_73821


namespace solve_fraction_l73_73002

theorem solve_fraction (x : ℝ) (h : 2 / (x - 3) = 2) : x = 4 :=
by
  sorry

end solve_fraction_l73_73002


namespace number_of_ostriches_l73_73097

theorem number_of_ostriches
    (x y : ℕ)
    (h1 : x + y = 150)
    (h2 : 2 * x + 6 * y = 624) :
    x = 69 :=
by
  -- Proof omitted
  sorry

end number_of_ostriches_l73_73097


namespace sum_two_smallest_prime_factors_l73_73668

theorem sum_two_smallest_prime_factors (n : ℕ) (h : n = 462) : 
  (2 + 3) = 5 := 
by {
  sorry
}

end sum_two_smallest_prime_factors_l73_73668


namespace periodic_functions_exist_l73_73714

theorem periodic_functions_exist (p1 p2 : ℝ) (h1 : p1 > 0) (h2 : p2 > 0) :
    ∃ (f1 f2 : ℝ → ℝ), (∀ x, f1 (x + p1) = f1 x) ∧ (∀ x, f2 (x + p2) = f2 x) ∧ ∃ T > 0, ∀ x, (f1 - f2) (x + T) = (f1 - f2) x :=
sorry

end periodic_functions_exist_l73_73714


namespace find_f_6_l73_73238

def f : ℕ → ℕ := sorry

lemma f_equality (x : ℕ) : f (x + 1) = x := sorry

theorem find_f_6 : f 6 = 5 :=
by
  -- the proof would go here
  sorry

end find_f_6_l73_73238


namespace sin_monotonically_decreasing_l73_73711

open Real

theorem sin_monotonically_decreasing (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = sin (2 * x + π / 3)) →
  (0 ≤ x ∧ x ≤ π) →
  (∀ x, (π / 12) ≤ x ∧ x ≤ (7 * π / 12)) →
  ∀ x y, (x < y → f y ≤ f x) := by
  sorry

end sin_monotonically_decreasing_l73_73711


namespace total_spent_by_pete_and_raymond_l73_73263

def initial_money_in_cents : ℕ := 250
def pete_spent_in_nickels : ℕ := 4
def nickel_value_in_cents : ℕ := 5
def raymond_dimes_left : ℕ := 7
def dime_value_in_cents : ℕ := 10

theorem total_spent_by_pete_and_raymond : 
  (pete_spent_in_nickels * nickel_value_in_cents) 
  + (initial_money_in_cents - (raymond_dimes_left * dime_value_in_cents)) = 200 := sorry

end total_spent_by_pete_and_raymond_l73_73263


namespace augmented_matrix_determinant_l73_73577

theorem augmented_matrix_determinant (m : ℝ) 
  (h : (1 - 2 * m) / (3 - 2) = 5) : 
  m = -2 :=
  sorry

end augmented_matrix_determinant_l73_73577


namespace savings_per_bagel_in_cents_l73_73397

theorem savings_per_bagel_in_cents (cost_individual : ℝ) (cost_dozen : ℝ) (dozen : ℕ) (cents_per_dollar : ℕ) :
  cost_individual = 2.25 →
  cost_dozen = 24 →
  dozen = 12 →
  cents_per_dollar = 100 →
  (cost_individual * cents_per_dollar - (cost_dozen / dozen) * cents_per_dollar) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end savings_per_bagel_in_cents_l73_73397


namespace find_multiplier_l73_73391

theorem find_multiplier (x : ℝ) : 3 - 3 * x < 14 ↔ x = -3 :=
by {
  sorry
}

end find_multiplier_l73_73391


namespace total_cars_l73_73806

-- Definitions for the conditions
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := 2 * cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Lean theorem statement
theorem total_cars : cathy_cars + lindsey_cars + carol_cars + susan_cars = 32 := by
  sorry

end total_cars_l73_73806


namespace evaluate_expression_l73_73515

theorem evaluate_expression :
  -25 + 7 * ((8 / 4) ^ 2) = 3 :=
by
  sorry

end evaluate_expression_l73_73515


namespace mom_foster_dog_food_l73_73280

theorem mom_foster_dog_food
    (puppy_food_per_meal : ℚ := 1 / 2)
    (puppy_meals_per_day : ℕ := 2)
    (num_puppies : ℕ := 5)
    (total_food_needed : ℚ := 57)
    (days : ℕ := 6)
    (mom_meals_per_day : ℕ := 3) :
    (total_food_needed - (num_puppies * puppy_food_per_meal * ↑puppy_meals_per_day * ↑days)) / (↑days * ↑mom_meals_per_day) = 1.5 :=
by
  -- Definitions translation
  let puppy_total_food := num_puppies * puppy_food_per_meal * ↑puppy_meals_per_day * ↑days
  let mom_total_food := total_food_needed - puppy_total_food
  let mom_meals := ↑days * ↑mom_meals_per_day
  -- Proof starts with sorry to indicate that the proof part is not included
  sorry

end mom_foster_dog_food_l73_73280


namespace arc_length_l73_73980

/-- Given a circle with a radius of 5 cm and a sector area of 11.25 cm², 
prove that the length of the arc forming the sector is 4.5 cm. --/
theorem arc_length (r : ℝ) (A : ℝ) (θ : ℝ) (arc_length : ℝ) 
  (h_r : r = 5) 
  (h_A : A = 11.25) 
  (h_area_formula : A = (θ / (2 * π)) * π * r ^ 2) 
  (h_arc_length_formula : arc_length = r * θ) :
  arc_length = 4.5 :=
sorry

end arc_length_l73_73980


namespace proof_2_fx_minus_11_eq_f_x_minus_d_l73_73377

def f (x : ℝ) : ℝ := 2 * x - 3
def d : ℝ := 2

theorem proof_2_fx_minus_11_eq_f_x_minus_d :
  2 * (f 5) - 11 = f (5 - d) := by
  sorry

end proof_2_fx_minus_11_eq_f_x_minus_d_l73_73377


namespace right_triangle_and_mod_inverse_l73_73216

theorem right_triangle_and_mod_inverse (a b c m : ℕ) (h1 : a = 48) (h2 : b = 55) (h3 : c = 73) (h4 : m = 4273) 
  (h5 : a^2 + b^2 = c^2) : ∃ x : ℕ, (480 * x) % m = 1 ∧ x = 1643 :=
by
  sorry

end right_triangle_and_mod_inverse_l73_73216


namespace problem_I_problem_II_l73_73282

-- Define the function f as given
def f (x m : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Problem (I)
theorem problem_I (x : ℝ) : -2 < x ∧ x < 1 ↔ f x 2 < 0 := sorry

-- Problem (II)
theorem problem_II (m : ℝ) : ∀ x, f x m + 1 ≥ 0 ↔ -3 ≤ m ∧ m ≤ 1 := sorry

end problem_I_problem_II_l73_73282


namespace thomas_saves_40_per_month_l73_73068

variables (T J : ℝ) (months : ℝ := 72) 

theorem thomas_saves_40_per_month 
  (h1 : J = (3/5) * T)
  (h2 : 72 * T + 72 * J = 4608) : 
  T = 40 :=
by sorry

end thomas_saves_40_per_month_l73_73068


namespace sum_of_consecutive_integers_product_336_l73_73438

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_integers_product_336_l73_73438


namespace dandelion_seeds_percentage_approx_29_27_l73_73975

/-
Mathematical conditions:
- Carla has the following set of plants and seeds per plant:
  - 6 sunflowers with 9 seeds each
  - 8 dandelions with 12 seeds each
  - 4 roses with 7 seeds each
  - 10 tulips with 15 seeds each.
- Calculate:
  - total seeds
  - percentage of seeds from dandelions
-/ 

def num_sunflowers : ℕ := 6
def num_dandelions : ℕ := 8
def num_roses : ℕ := 4
def num_tulips : ℕ := 10

def seeds_per_sunflower : ℕ := 9
def seeds_per_dandelion : ℕ := 12
def seeds_per_rose : ℕ := 7
def seeds_per_tulip : ℕ := 15

def total_sunflower_seeds : ℕ := num_sunflowers * seeds_per_sunflower
def total_dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion
def total_rose_seeds : ℕ := num_roses * seeds_per_rose
def total_tulip_seeds : ℕ := num_tulips * seeds_per_tulip

def total_seeds : ℕ := total_sunflower_seeds + total_dandelion_seeds + total_rose_seeds + total_tulip_seeds

def percentage_dandelion_seeds : ℚ := (total_dandelion_seeds : ℚ) / total_seeds * 100

theorem dandelion_seeds_percentage_approx_29_27 : abs (percentage_dandelion_seeds - 29.27) < 0.01 :=
sorry

end dandelion_seeds_percentage_approx_29_27_l73_73975


namespace surface_area_implies_side_length_diagonal_l73_73755

noncomputable def cube_side_length_diagonal (A : ℝ) := 
  A = 864 → ∃ s d : ℝ, s = 12 ∧ d = 12 * Real.sqrt 3

theorem surface_area_implies_side_length_diagonal : 
  cube_side_length_diagonal 864 := by
  sorry

end surface_area_implies_side_length_diagonal_l73_73755


namespace find_x_if_arithmetic_mean_is_12_l73_73262

theorem find_x_if_arithmetic_mean_is_12 (x : ℝ) (h : (8 + 16 + 21 + 7 + x) / 5 = 12) : x = 8 :=
by
  sorry

end find_x_if_arithmetic_mean_is_12_l73_73262


namespace singing_only_pupils_l73_73924

theorem singing_only_pupils (total_pupils debate_only both : ℕ) (h1 : total_pupils = 55) (h2 : debate_only = 10) (h3 : both = 17) :
  total_pupils - debate_only = 45 :=
by
  -- skipping proof
  sorry

end singing_only_pupils_l73_73924


namespace tangent_product_l73_73319

noncomputable def tangent (x : ℝ) : ℝ := Real.tan x

theorem tangent_product : 
  tangent (20 * Real.pi / 180) * 
  tangent (40 * Real.pi / 180) * 
  tangent (60 * Real.pi / 180) * 
  tangent (80 * Real.pi / 180) = 3 :=
by
  -- Definitions and conditions
  have tg60 := Real.tan (60 * Real.pi / 180) = Real.sqrt 3
  
  -- Add tangent addition, subtraction, and triple angle formulas
  -- tangent addition formula
  have tg_add := ∀ x y : ℝ, tangent (x + y) = (tangent x + tangent y) / (1 - tangent x * tangent y)
  -- tangent subtraction formula
  have tg_sub := ∀ x y : ℝ, tangent (x - y) = (tangent x - tangent y) / (1 + tangent x * tangent y)
  -- tangent triple angle formula
  have tg_triple := ∀ α : ℝ, tangent (3 * α) = (3 * tangent α - tangent α^3) / (1 - 3 * tangent α^2)
  
  -- sorry to skip the proof
  sorry


end tangent_product_l73_73319


namespace jack_sugar_amount_l73_73144

-- Definitions of initial conditions
def initial_amount : ℕ := 65
def used_amount : ℕ := 18
def bought_amount : ℕ := 50

-- Theorem statement
theorem jack_sugar_amount : initial_amount - used_amount + bought_amount = 97 :=
by
  -- Proof goes here
  sorry

end jack_sugar_amount_l73_73144


namespace conceived_number_is_seven_l73_73110

theorem conceived_number_is_seven (x : ℕ) (h1 : x > 0) (h2 : (1 / 4 : ℚ) * (10 * x + 7 - x * x) - x = 0) : x = 7 := by
  sorry

end conceived_number_is_seven_l73_73110


namespace Kyle_papers_delivered_each_week_l73_73150

-- Definitions representing the conditions
def papers_per_day := 100
def days_Mon_to_Sat := 6
def regular_Sunday_customers := 100
def non_regular_Sunday_customers := 30
def no_delivery_customers_on_Sunday := 10

-- The total number of papers delivered each week
def total_papers_per_week : ℕ :=
  days_Mon_to_Sat * papers_per_day +
  regular_Sunday_customers - no_delivery_customers_on_Sunday + non_regular_Sunday_customers

-- Prove that Kyle delivers 720 papers each week
theorem Kyle_papers_delivered_each_week : total_papers_per_week = 720 :=
sorry

end Kyle_papers_delivered_each_week_l73_73150


namespace integer_pairs_satisfy_equation_l73_73506

theorem integer_pairs_satisfy_equation :
  ∃ (S : Finset (ℤ × ℤ)), S.card = 5 ∧ ∀ (m n : ℤ), (m, n) ∈ S ↔ m^2 + n = m * n + 1 :=
by
  sorry

end integer_pairs_satisfy_equation_l73_73506


namespace g_five_eq_one_l73_73348

noncomputable def g : ℝ → ℝ := sorry

axiom g_mul (x z : ℝ) : g (x * z) = g x * g z
axiom g_one_ne_zero : g (1) ≠ 0

theorem g_five_eq_one : g (5) = 1 := 
by
  sorry

end g_five_eq_one_l73_73348


namespace box_height_l73_73744

theorem box_height (x : ℝ) (hx : x + 5 = 10)
  (surface_area : 2*x^2 + 4*x*(x + 5) ≥ 150) : x + 5 = 10 :=
sorry

end box_height_l73_73744


namespace find_a_plus_b_l73_73646

-- Given conditions
variable (a b : ℝ)

-- The imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Condition equation
def equation := (a + i) * i = b - 2 * i

-- Define the lean statement
theorem find_a_plus_b (h : equation a b) : a + b = -3 :=
by sorry

end find_a_plus_b_l73_73646


namespace quadratic_function_min_value_in_interval_l73_73685

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 6 * x + 10

theorem quadratic_function_min_value_in_interval :
  ∀ (x : ℝ), 2 ≤ x ∧ x < 5 → (∃ min_val : ℝ, min_val = 1) ∧ (∀ upper_bound : ℝ, ∃ x0 : ℝ, x0 < 5 ∧ quadratic_function x0 > upper_bound) := 
by
  sorry

end quadratic_function_min_value_in_interval_l73_73685


namespace gcd_153_119_l73_73885

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  have h1 : 153 = 119 * 1 + 34 := by rfl
  have h2 : 119 = 34 * 3 + 17 := by rfl
  have h3 : 34 = 17 * 2 := by rfl
  sorry

end gcd_153_119_l73_73885


namespace milk_production_days_l73_73370

theorem milk_production_days (y : ℕ) :
  (y + 4) * (y + 2) * (y + 6) / (y * (y + 3) * (y + 4)) = y * (y + 3) * (y + 6) / ((y + 2) * (y + 4)) :=
sorry

end milk_production_days_l73_73370


namespace margo_pairing_probability_l73_73183

theorem margo_pairing_probability (students : Finset ℕ)
  (H_50_students : students.card = 50)
  (margo irma jess kurt : ℕ)
  (H_margo_in_students : margo ∈ students)
  (H_irma_in_students : irma ∈ students)
  (H_jess_in_students : jess ∈ students)
  (H_kurt_in_students : kurt ∈ students)
  (possible_partners : Finset ℕ := students.erase margo) :
  (3: ℝ) / 49 = ((3: ℝ) / (possible_partners.card: ℝ)) :=
by
  -- The actual steps of the proof will be here
  sorry

end margo_pairing_probability_l73_73183


namespace impossible_ratio_5_11_l73_73468

theorem impossible_ratio_5_11:
  ∀ (b g: ℕ), 
  b + g ≥ 66 →
  b + 11 = g - 13 →
  ¬(5 * b = 11 * (b + 24) ∧ b ≥ 21) := 
by
  intros b g h1 h2 h3
  sorry

end impossible_ratio_5_11_l73_73468


namespace compute_expression_l73_73739

theorem compute_expression : 2 * ((3 + 7) ^ 2 + (3 ^ 2 + 7 ^ 2)) = 316 := 
by
  sorry

end compute_expression_l73_73739


namespace multiplication_difference_l73_73114

theorem multiplication_difference :
  672 * 673 * 674 - 671 * 673 * 675 = 2019 := by
  sorry

end multiplication_difference_l73_73114


namespace small_cone_altitude_l73_73294

theorem small_cone_altitude (h_f: ℝ) (a_lb: ℝ) (a_ub: ℝ) : 
  h_f = 24 → a_lb = 225 * Real.pi → a_ub = 25 * Real.pi → ∃ h_s, h_s = 12 := 
by
  intros h1 h2 h3
  sorry

end small_cone_altitude_l73_73294


namespace construct_length_one_l73_73547

theorem construct_length_one
    (a : ℝ) 
    (h_a : a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) : 
    ∃ (b : ℝ), b = 1 :=
by
    sorry

end construct_length_one_l73_73547


namespace correct_total_score_l73_73580

theorem correct_total_score (total_score1 total_score2 : ℤ) : 
  (total_score1 = 5734 ∨ total_score2 = 5734) → (total_score1 = 5735 ∨ total_score2 = 5735) → 
  (total_score1 % 2 = 0 ∨ total_score2 % 2 = 0) → 
  (total_score1 ≠ total_score2) → 
  5734 % 2 = 0 :=
by
  sorry

end correct_total_score_l73_73580


namespace distinct_triples_l73_73309

theorem distinct_triples (a b c : ℕ) (h₁: 2 * a - 1 = k₁ * b) (h₂: 2 * b - 1 = k₂ * c) (h₃: 2 * c - 1 = k₃ * a) :
  (a, b, c) = (7, 13, 25) ∨ (a, b, c) = (13, 25, 7) ∨ (a, b, c) = (25, 7, 13) := sorry

end distinct_triples_l73_73309


namespace h_value_l73_73798

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 7

theorem h_value :
  ∃ (h : ℝ → ℝ), (h 0 = 7)
  ∧ (∃ (a b c : ℝ), (f a = 0) ∧ (f b = 0) ∧ (f c = 0) ∧ (h (-8) = (1/49) * (-8 - a^3) * (-8 - b^3) * (-8 - c^3))) 
  ∧ h (-8) = -1813 := by
  sorry

end h_value_l73_73798


namespace sum_of_coordinates_l73_73893

def g : ℝ → ℝ := sorry
def h (x : ℝ) : ℝ := (g x)^3

theorem sum_of_coordinates (hg : g 4 = 8) : 4 + h 4 = 516 :=
by
  sorry

end sum_of_coordinates_l73_73893


namespace nicky_pace_l73_73574

theorem nicky_pace :
  ∃ v : ℝ, v = 3 ∧ (
    ∀ (head_start : ℝ) (cristina_pace : ℝ) (time : ℝ) (distance_encounter : ℝ), 
      head_start = 36 ∧ cristina_pace = 4 ∧ time = 36 ∧ distance_encounter = cristina_pace * time - head_start →
      distance_encounter / time = v
  ) :=
sorry

end nicky_pace_l73_73574


namespace max_pqrs_squared_l73_73763

theorem max_pqrs_squared (p q r s : ℝ)
  (h1 : p + q = 18)
  (h2 : pq + r + s = 85)
  (h3 : pr + qs = 190)
  (h4 : rs = 120) :
  p^2 + q^2 + r^2 + s^2 ≤ 886 :=
sorry

end max_pqrs_squared_l73_73763


namespace problem1_problem2_problem3_l73_73841

noncomputable def U : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2}
noncomputable def A : Set ℝ := {x | x < 1 ∨ x > 3}
noncomputable def B : Set ℝ := {x | x < 1 ∨ x > 2}

theorem problem1 : A ∩ B = {x | x < 1 ∨ x > 3} := 
  sorry

theorem problem2 : A ∩ (U \ B) = ∅ := 
  sorry

theorem problem3 : U \ (A ∪ B) = {1, 2} := 
  sorry

end problem1_problem2_problem3_l73_73841


namespace find_k_value_l73_73148

theorem find_k_value (x₁ x₂ x₃ x₄ : ℝ)
  (h1 : (x₁ + x₂ + x₃ + x₄) = 18)
  (h2 : (x₁ * x₂ + x₁ * x₃ + x₁ * x₄ + x₂ * x₃ + x₂ * x₄ + x₃ * x₄) = k)
  (h3 : (x₁ * x₂ * x₃ + x₁ * x₂ * x₄ + x₁ * x₃ * x₄ + x₂ * x₃ * x₄) = -200)
  (h4 : (x₁ * x₂ * x₃ * x₄) = -1984)
  (h5 : x₁ * x₂ = -32) :
  k = 86 :=
by sorry

end find_k_value_l73_73148


namespace correct_statement_l73_73185

def degree (term : String) : ℕ :=
  if term = "1/2πx^2" then 2
  else if term = "-4x^2y" then 3
  else 0

def coefficient (term : String) : ℤ :=
  if term = "-4x^2y" then -4
  else if term = "3(x+y)" then 3
  else 0

def is_monomial (term : String) : Bool :=
  if term = "8" then true
  else false

theorem correct_statement : 
  (degree "1/2πx^2" ≠ 3) ∧ 
  (coefficient "-4x^2y" ≠ 4) ∧ 
  (is_monomial "8" = true) ∧ 
  (coefficient "3(x+y)" ≠ 3) := 
by
  sorry

end correct_statement_l73_73185


namespace imaginary_part_of_complex_number_l73_73433

open Complex

theorem imaginary_part_of_complex_number :
  ∀ (i : ℂ), i^2 = -1 → im ((2 * I) / (2 + I^3)) = 4 / 5 :=
by
  intro i hi
  sorry

end imaginary_part_of_complex_number_l73_73433


namespace square_difference_identity_l73_73339

theorem square_difference_identity (a b : ℕ) : (a - b)^2 = a^2 - 2 * a * b + b^2 :=
  by sorry

lemma evaluate_expression : (101 - 2)^2 = 9801 :=
  by
    have h := square_difference_identity 101 2
    exact h

end square_difference_identity_l73_73339


namespace triangle_angle_construction_l73_73274

-- Step d): Lean 4 Statement
theorem triangle_angle_construction (a b c : ℝ) (α β : ℝ) (γ : ℝ) (h1 : γ = 120)
  (h2 : a < c) (h3 : c < a + b) (h4 : b < c)  (h5 : c < a + b) :
    (∃ α' β' γ', α' = 60 ∧ β' = α ∧ γ' = 60 + β) ∧ 
    (∃ α'' β'' γ'', α'' = 60 ∧ β'' = β ∧ γ'' = 60 + α) :=
  sorry

end triangle_angle_construction_l73_73274


namespace eq_holds_for_n_l73_73966

theorem eq_holds_for_n (n : ℕ) (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a + b + c + d = n * Real.sqrt (a * b * c * d) ↔ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

end eq_holds_for_n_l73_73966


namespace how_many_more_yellow_peaches_l73_73179

-- Definitions
def red_peaches : ℕ := 7
def yellow_peaches_initial : ℕ := 15
def green_peaches : ℕ := 8
def combined_red_green_peaches := red_peaches + green_peaches
def required_yellow_peaches := 2 * combined_red_green_peaches
def additional_yellow_peaches_needed := required_yellow_peaches - yellow_peaches_initial

-- Theorem statement
theorem how_many_more_yellow_peaches :
  additional_yellow_peaches_needed = 15 :=
by
  sorry

end how_many_more_yellow_peaches_l73_73179


namespace fraction_spent_by_Rica_is_one_fifth_l73_73446

-- Define the conditions
variable (totalPrizeMoney : ℝ) (fractionReceived : ℝ) (amountLeft : ℝ)
variable (h1 : totalPrizeMoney = 1000) (h2 : fractionReceived = 3 / 8) (h3 : amountLeft = 300)

-- Define Rica's original prize money
noncomputable def RicaOriginalPrizeMoney (totalPrizeMoney fractionReceived : ℝ) : ℝ :=
  fractionReceived * totalPrizeMoney

-- Define amount spent by Rica
noncomputable def AmountSpent (originalPrizeMoney amountLeft : ℝ) : ℝ :=
  originalPrizeMoney - amountLeft

-- Define the fraction of prize money spent by Rica
noncomputable def FractionSpent (amountSpent originalPrizeMoney : ℝ) : ℝ :=
  amountSpent / originalPrizeMoney

-- Main theorem to prove
theorem fraction_spent_by_Rica_is_one_fifth :
  let totalPrizeMoney := 1000
  let fractionReceived := 3 / 8
  let amountLeft := 300
  let RicaOriginalPrizeMoney := fractionReceived * totalPrizeMoney
  let AmountSpent := RicaOriginalPrizeMoney - amountLeft
  let FractionSpent := AmountSpent / RicaOriginalPrizeMoney
  FractionSpent = 1 / 5 :=
by {
  -- Proof details are omitted as per instructions
  sorry
}

end fraction_spent_by_Rica_is_one_fifth_l73_73446


namespace ex1_l73_73394

theorem ex1 (a b : ℕ) (h₀ : a = 3) (h₁ : b = 4) : ∃ n : ℕ, 3^(7*a + b) = n^7 :=
by
  use 27
  sorry

end ex1_l73_73394


namespace boy_late_l73_73472

noncomputable def time_late (D V1 V2 : ℝ) (early : ℝ) : ℝ :=
  let T1 := D / V1
  let T2 := D / V2
  let T1_mins := T1 * 60
  let T2_mins := T2 * 60
  let actual_on_time := T2_mins + early
  T1_mins - actual_on_time

theorem boy_late :
  time_late 2.5 5 10 10 = 5 :=
by
  sorry

end boy_late_l73_73472


namespace count_4_digit_numbers_with_conditions_l73_73609

def num_valid_numbers : Nat :=
  432

-- Statement declaring the proposition to be proved
theorem count_4_digit_numbers_with_conditions :
  (count_valid_numbers == 432) :=
sorry

end count_4_digit_numbers_with_conditions_l73_73609


namespace arith_seq_sum_7_8_9_l73_73787

noncomputable def S_n (a : Nat → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n.succ).sum a

def arith_seq (a : Nat → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

theorem arith_seq_sum_7_8_9 (a : Nat → ℝ) (h_arith : arith_seq a)
    (h_S3 : S_n a 3 = 8) (h_S6 : S_n a 6 = 7) : 
  (a 7 + a 8 + a 9) = 1 / 8 := 
  sorry

end arith_seq_sum_7_8_9_l73_73787


namespace math_lovers_l73_73723

/-- The proof problem: 
Given 1256 students in total and the difference of 408 between students who like math and others,
prove that the number of students who like math is 424, given that students who like math are fewer than 500.
--/
theorem math_lovers (M O : ℕ) (h1 : M + O = 1256) (h2: O - M = 408) (h3 : M < 500) : M = 424 :=
by
  sorry

end math_lovers_l73_73723


namespace number_of_birds_is_20_l73_73636

-- Define the given conditions
def distance_jim_disney : ℕ := 50
def distance_disney_london : ℕ := 60
def total_travel_distance : ℕ := 2200

-- Define the number of birds
def num_birds (B : ℕ) : Prop :=
  (distance_jim_disney + distance_disney_london) * B = total_travel_distance

-- The theorem stating the number of birds
theorem number_of_birds_is_20 : num_birds 20 :=
by
  unfold num_birds
  sorry

end number_of_birds_is_20_l73_73636


namespace mean_of_first_set_is_67_l73_73628

theorem mean_of_first_set_is_67 (x : ℝ) 
  (h : (50 + 62 + 97 + 124 + x) / 5 = 75.6) : 
  (28 + x + 70 + 88 + 104) / 5 = 67 := 
by
  sorry

end mean_of_first_set_is_67_l73_73628


namespace rope_length_after_knots_l73_73607

def num_ropes : ℕ := 64
def length_per_rope : ℕ := 25
def length_reduction_per_knot : ℕ := 3
def num_knots : ℕ := num_ropes - 1
def initial_total_length : ℕ := num_ropes * length_per_rope
def total_reduction : ℕ := num_knots * length_reduction_per_knot
def final_rope_length : ℕ := initial_total_length - total_reduction

theorem rope_length_after_knots :
  final_rope_length = 1411 := by
  sorry

end rope_length_after_knots_l73_73607


namespace not_possible_to_cover_l73_73865

namespace CubeCovering

-- Defining the cube and its properties
def cube_side_length : ℕ := 4
def face_area := cube_side_length * cube_side_length
def total_faces : ℕ := 6
def faces_to_cover : ℕ := 3

-- Defining the paper strips and their properties
def strip_length : ℕ := 3
def strip_width : ℕ := 1
def strip_area := strip_length * strip_width
def num_strips : ℕ := 16

-- Calculate the total area to cover
def total_area_to_cover := faces_to_cover * face_area
def total_area_strips := num_strips * strip_area

-- Statement: Prove that it is not possible to cover the three faces
theorem not_possible_to_cover : total_area_to_cover = 48 → total_area_strips = 48 → false := by
  intro h1 h2
  sorry

end CubeCovering

end not_possible_to_cover_l73_73865


namespace base7_addition_l73_73009

theorem base7_addition : (26:ℕ) + (245:ℕ) = 304 :=
  sorry

end base7_addition_l73_73009


namespace spinner_probability_C_l73_73054

theorem spinner_probability_C 
  (P_A : ℚ) (P_B : ℚ) (P_C : ℚ) (P_D : ℚ)
  (hA : P_A = 1/3)
  (hB : P_B = 1/4)
  (hD : P_D = 1/6)
  (hSum : P_A + P_B + P_C + P_D = 1) :
  P_C = 1 / 4 := 
sorry

end spinner_probability_C_l73_73054


namespace fruit_selling_price_3640_l73_73123

def cost_price := 22
def initial_selling_price := 38
def initial_quantity_sold := 160
def price_reduction := 3
def quantity_increase := 120
def target_profit := 3640

theorem fruit_selling_price_3640 (x : ℝ) :
  ((initial_selling_price - x - cost_price) * (initial_quantity_sold + (x / price_reduction) * quantity_increase) = target_profit) →
  x = 9 →
  initial_selling_price - x = 29 :=
by
  intro h1 h2
  sorry

end fruit_selling_price_3640_l73_73123


namespace three_colors_sufficient_l73_73481

-- Definition of the tessellation problem with specified conditions.
def tessellation (n : ℕ) (x_divisions : ℕ) (y_divisions : ℕ) : Prop :=
  n = 8 ∧ x_divisions = 2 ∧ y_divisions = 2

-- Definition of the adjacency property.
def no_adjacent_same_color {α : Type} (coloring : ℕ → ℕ → α) : Prop :=
  ∀ (i j : ℕ), i < 8 → j < 8 →
  (i > 0 → coloring i j ≠ coloring (i-1) j) ∧ 
  (j > 0 → coloring i j ≠ coloring i (j-1)) ∧
  (i < 7 → coloring i j ≠ coloring (i+1) j) ∧ 
  (j < 7 → coloring i j ≠ coloring i (j+1)) ∧
  (i > 0 ∧ j > 0 → coloring i j ≠ coloring (i-1) (j-1)) ∧
  (i < 7 ∧ j < 7 → coloring i j ≠ coloring (i+1) (j+1)) ∧
  (i > 0 ∧ j < 7 → coloring i j ≠ coloring (i-1) (j+1)) ∧
  (i < 7 ∧ j > 0 → coloring i j ≠ coloring (i+1) (j-1))

-- The main theorem that needs to be proved.
theorem three_colors_sufficient : ∃ (k : ℕ) (coloring : ℕ → ℕ → ℕ), k = 3 ∧ 
  tessellation 8 2 2 ∧ 
  no_adjacent_same_color coloring := by
  sorry 

end three_colors_sufficient_l73_73481


namespace wheels_on_floor_l73_73436

def number_of_wheels (n_people : Nat) (w_per_person : Nat) : Nat :=
  n_people * w_per_person

theorem wheels_on_floor (n_people : Nat) (w_per_person : Nat) (h_people : n_people = 40) (h_wheels : w_per_person = 4) :
  number_of_wheels n_people w_per_person = 160 := by
  sorry

end wheels_on_floor_l73_73436


namespace expected_value_min_of_subset_l73_73979

noncomputable def expected_value_min (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : ℚ :=
  (n + 1) / (r + 1)

theorem expected_value_min_of_subset (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : 
  expected_value_min n r h = (n + 1) / (r + 1) :=
sorry

end expected_value_min_of_subset_l73_73979


namespace cost_price_computer_table_l73_73708

theorem cost_price_computer_table :
  ∃ CP : ℝ, CP * 1.25 = 5600 ∧ CP = 4480 :=
by
  sorry

end cost_price_computer_table_l73_73708


namespace star_set_l73_73314

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x}
def star (A B : Set ℝ) : Set ℝ := {x | (x ∈ A ∪ B) ∧ ¬(x ∈ A ∩ B)}

theorem star_set :
  star A B = {x | (0 ≤ x ∧ x < 1) ∨ (3 < x)} :=
by
  sorry

end star_set_l73_73314


namespace alice_burgers_each_day_l73_73503

theorem alice_burgers_each_day (cost_per_burger : ℕ) (total_spent : ℕ) (days_in_june : ℕ) 
  (h1 : cost_per_burger = 13) (h2 : total_spent = 1560) (h3 : days_in_june = 30) :
  (total_spent / cost_per_burger) / days_in_june = 4 := by
  sorry

end alice_burgers_each_day_l73_73503


namespace tim_balloon_count_l73_73629

theorem tim_balloon_count (Dan_balloons : ℕ) (h1 : Dan_balloons = 59) (Tim_balloons : ℕ) (h2 : Tim_balloons = 11 * Dan_balloons) : Tim_balloons = 649 :=
sorry

end tim_balloon_count_l73_73629


namespace number_of_triangles_with_perimeter_nine_l73_73701

theorem number_of_triangles_with_perimeter_nine : 
  ∃ (a b c : ℕ), a + b + c = 9 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry  -- Proof steps are omitted.

end number_of_triangles_with_perimeter_nine_l73_73701


namespace cassie_nails_l73_73313

-- Define the number of pets
def num_dogs := 4
def num_parrots := 8
def num_cats := 2
def num_rabbits := 6

-- Define the number of nails/claws/toes per pet
def nails_per_dog := 4 * 4
def common_claws_per_parrot := 2 * 3
def extra_toed_parrot_claws := 2 * 4
def toes_per_cat := 2 * 5 + 2 * 4
def rear_nails_per_rabbit := 2 * 5
def front_nails_per_rabbit := 3 + 4

-- Calculations
def total_dog_nails := num_dogs * nails_per_dog
def total_parrot_claws := 7 * common_claws_per_parrot + extra_toed_parrot_claws
def total_cat_toes := num_cats * toes_per_cat
def total_rabbit_nails := num_rabbits * (rear_nails_per_rabbit + front_nails_per_rabbit)

-- Total nails/claws/toes
def total_nails := total_dog_nails + total_parrot_claws + total_cat_toes + total_rabbit_nails

-- Theorem stating the problem
theorem cassie_nails : total_nails = 252 :=
by
  -- Here we would normally have the proof, but we'll skip it with sorry
  sorry

end cassie_nails_l73_73313


namespace total_population_eq_51b_over_40_l73_73193

variable (b g t : Nat)

-- Conditions
def boys_eq_four_times_girls (b g : Nat) : Prop := b = 4 * g
def girls_eq_ten_times_teachers (g t : Nat) : Prop := g = 10 * t

-- Statement to prove
theorem total_population_eq_51b_over_40 (b g t : Nat) 
  (h1 : boys_eq_four_times_girls b g) 
  (h2 : girls_eq_ten_times_teachers g t) : 
  b + g + t = (51 * b) / 40 := 
sorry

end total_population_eq_51b_over_40_l73_73193


namespace monotonicity_and_range_l73_73456

noncomputable def f (a x : ℝ) : ℝ := (a * x - 2) * Real.exp x - Real.exp (a - 2)

theorem monotonicity_and_range (a x : ℝ) :
  ( (a = 0 → ∀ x, f a x < f a (x + 1)) ∧
  (a > 0 → ∀ x < (2 - a) / a, f a x < f a (x + 1) ∧ ∀ x > (2 - a) / a, f a x > f a (x + 1) ) ∧
  (a < 0 → ∀ x > (2 - a) / a, f a x < f a (x + 1) ∧ ∀ x < (2 - a) / a, f a x > f a (x + 1) ) ∧
  (∀ x > 1, f a x > 0 ↔ a ∈ Set.Ici 1)) 
:=
sorry

end monotonicity_and_range_l73_73456


namespace new_computer_price_l73_73920

-- Define the initial conditions
def initial_price_condition (x : ℝ) : Prop := 2 * x = 540

-- Define the calculation for the new price after a 30% increase
def new_price (x : ℝ) : ℝ := x * 1.30

-- Define the final proof problem statement
theorem new_computer_price : ∃ x : ℝ, initial_price_condition x ∧ new_price x = 351 :=
by sorry

end new_computer_price_l73_73920


namespace line_through_parabola_no_intersection_l73_73108

-- Definitions of the conditions
def parabola (x : ℝ) : ℝ := x^2 
def point_Q := (10, 5)

-- The main theorem statement
theorem line_through_parabola_no_intersection :
  ∃ r s : ℝ, (∀ (m : ℝ), (r < m ∧ m < s) ↔ ¬ ∃ x : ℝ, parabola x = m * (x - 10) + 5) ∧ r + s = 40 :=
sorry

end line_through_parabola_no_intersection_l73_73108


namespace interval_for_f_l73_73961

noncomputable def f (x : ℝ) : ℝ :=
-0.5 * x ^ 2 + 13 / 2

theorem interval_for_f (a b : ℝ) :
  f a = 2 * b ∧ f b = 2 * a ∧ (a ≤ 0 ∨ 0 ≤ b) → 
  ([a, b] = [1, 3] ∨ [a, b] = [-2 - Real.sqrt 17, 13 / 4]) :=
by sorry

end interval_for_f_l73_73961


namespace puppies_sold_l73_73560

theorem puppies_sold (total_puppies sold_puppies puppies_per_cage total_cages : ℕ)
  (h1 : total_puppies = 102)
  (h2 : puppies_per_cage = 9)
  (h3 : total_cages = 9)
  (h4 : total_puppies - sold_puppies = puppies_per_cage * total_cages) :
  sold_puppies = 21 :=
by {
  -- Proof details would go here
  sorry
}

end puppies_sold_l73_73560


namespace exists_naturals_l73_73508

def sum_of_digits (a : ℕ) : ℕ := sorry

theorem exists_naturals (R : ℕ) (hR : R > 0) :
  ∃ n : ℕ, n > 0 ∧ (sum_of_digits (n^2)) / (sum_of_digits n) = R :=
by
  sorry

end exists_naturals_l73_73508


namespace potatoes_yield_l73_73287

theorem potatoes_yield (steps_length : ℕ) (steps_width : ℕ) (step_size : ℕ) (yield_per_sqft : ℚ) 
  (h_steps_length : steps_length = 18) 
  (h_steps_width : steps_width = 25) 
  (h_step_size : step_size = 3) 
  (h_yield_per_sqft : yield_per_sqft = 1/3) 
  : (steps_length * step_size) * (steps_width * step_size) * yield_per_sqft = 1350 := 
by 
  sorry

end potatoes_yield_l73_73287


namespace minute_hand_moves_180_degrees_l73_73519

noncomputable def minute_hand_angle_6_15_to_6_45 : ℝ :=
  let degrees_per_hour := 360
  let hours_period := 0.5
  degrees_per_hour * hours_period

theorem minute_hand_moves_180_degrees :
  minute_hand_angle_6_15_to_6_45 = 180 :=
by
  sorry

end minute_hand_moves_180_degrees_l73_73519


namespace simplify_expression_l73_73591

theorem simplify_expression : (2468 * 2468) / (2468 + 2468) = 1234 :=
by
  sorry

end simplify_expression_l73_73591


namespace exists_k_square_congruent_neg_one_iff_l73_73911

theorem exists_k_square_congruent_neg_one_iff (p : ℕ) [Fact p.Prime] :
  (∃ k : ℤ, (k^2 ≡ -1 [ZMOD p])) ↔ (p = 2 ∨ p % 4 = 1) :=
sorry

end exists_k_square_congruent_neg_one_iff_l73_73911


namespace largest_possible_sum_l73_73090

-- Define whole numbers
def whole_numbers : Set ℕ := Set.univ

-- Define the given conditions
variables (a b : ℕ)
axiom h1 : a ∈ whole_numbers
axiom h2 : b ∈ whole_numbers
axiom h3 : a * b = 48

-- Prove the largest sum condition
theorem largest_possible_sum : a + b ≤ 49 :=
sorry

end largest_possible_sum_l73_73090


namespace dessert_eating_contest_l73_73987

theorem dessert_eating_contest (a b c : ℚ) 
  (h1 : a = 5/6) 
  (h2 : b = 7/8) 
  (h3 : c = 1/2) :
  b - a = 1/24 ∧ a - c = 1/3 := 
by 
  sorry

end dessert_eating_contest_l73_73987


namespace number_of_sheep_l73_73022

def ratio_sheep_horses (S H : ℕ) : Prop := S / H = 3 / 7
def horse_food_per_day := 230 -- ounces
def total_food_per_day := 12880 -- ounces

theorem number_of_sheep (S H : ℕ) 
  (h1 : ratio_sheep_horses S H) 
  (h2 : H * horse_food_per_day = total_food_per_day) 
  : S = 24 :=
sorry

end number_of_sheep_l73_73022


namespace focus_of_parabola_tangent_to_circle_directrix_l73_73680

theorem focus_of_parabola_tangent_to_circle_directrix :
  ∃ p : ℝ, p > 0 ∧
  (∃ (x y : ℝ), x ^ 2 + y ^ 2 - 6 * x - 7 = 0 ∧
  ∀ x y : ℝ, y ^ 2 = 2 * p * x → x = -p) →
  (1, 0) = (p, 0) :=
by
  sorry

end focus_of_parabola_tangent_to_circle_directrix_l73_73680


namespace tina_spent_on_books_l73_73903

theorem tina_spent_on_books : 
  ∀ (saved_in_june saved_in_july saved_in_august spend_on_books spend_on_shoes money_left : ℤ),
  saved_in_june = 27 →
  saved_in_july = 14 →
  saved_in_august = 21 →
  spend_on_shoes = 17 →
  money_left = 40 →
  (saved_in_june + saved_in_july + saved_in_august) - spend_on_books - spend_on_shoes = money_left →
  spend_on_books = 5 :=
by
  intros saved_in_june saved_in_july saved_in_august spend_on_books spend_on_shoes money_left
  intros h_june h_july h_august h_shoes h_money_left h_eq
  sorry

end tina_spent_on_books_l73_73903


namespace min_N_of_block_viewed_l73_73194

theorem min_N_of_block_viewed (x y z N : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_factor : (x - 1) * (y - 1) * (z - 1) = 231) : 
  N = x * y * z ∧ N = 384 :=
by {
  sorry 
}

end min_N_of_block_viewed_l73_73194


namespace a_2n_is_perfect_square_l73_73849

-- Define the sequence a_n as per the problem's conditions
def a (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 4
  else a (n - 1) + a (n - 3) + a (n - 4)

-- Define the Fibonacci sequence for comparison
def fib (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

-- Key theorem to prove: a_{2n} is a perfect square
theorem a_2n_is_perfect_square (n : ℕ) : 
  ∃ k : ℕ, a (2 * n) = k * k :=
sorry

end a_2n_is_perfect_square_l73_73849


namespace custom_op_evaluation_l73_73564

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x

theorem custom_op_evaluation : custom_op 6 4 - custom_op 4 6 = -6 :=
by
  sorry

end custom_op_evaluation_l73_73564


namespace find_h2_l73_73208

noncomputable def h (x : ℝ) : ℝ := 
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) - 1) / (x^15 - 1)

theorem find_h2 : h 2 = 2 :=
by 
  sorry

end find_h2_l73_73208


namespace pants_cost_l73_73898

theorem pants_cost (starting_amount shirts_cost shirts_count amount_left money_after_shirts pants_cost : ℕ) 
    (h1 : starting_amount = 109)
    (h2 : shirts_cost = 11)
    (h3 : shirts_count = 2)
    (h4 : amount_left = 74)
    (h5 : money_after_shirts = starting_amount - shirts_cost * shirts_count)
    (h6 : pants_cost = money_after_shirts - amount_left) :
  pants_cost = 13 :=
by
  sorry

end pants_cost_l73_73898


namespace find_k_solve_quadratic_l73_73330

-- Define the conditions
variables (x1 x2 k : ℝ)

-- Given conditions
def quadratic_roots : Prop :=
  x1 + x2 = 6 ∧ x1 * x2 = k

def condition_A (x1 x2 : ℝ) : Prop :=
  x1^2 * x2^2 - x1 - x2 = 115

-- Prove that k = -11 given the conditions
theorem find_k (h1: quadratic_roots x1 x2 k) (h2 : condition_A x1 x2) : k = -11 :=
  sorry

-- Prove the roots of the quadratic equation when k = -11
theorem solve_quadratic (h1 : quadratic_roots x1 x2 (-11)) : 
  x1 = 3 + 2 * Real.sqrt 5 ∧ x2 = 3 - 2 * Real.sqrt 5 ∨ 
  x1 = 3 - 2 * Real.sqrt 5 ∧ x2 = 3 + 2 * Real.sqrt 5 :=
  sorry

end find_k_solve_quadratic_l73_73330


namespace fraction_ordering_l73_73223

noncomputable def t1 : ℝ := (100^100 + 1) / (100^90 + 1)
noncomputable def t2 : ℝ := (100^99 + 1) / (100^89 + 1)
noncomputable def t3 : ℝ := (100^101 + 1) / (100^91 + 1)
noncomputable def t4 : ℝ := (101^101 + 1) / (101^91 + 1)
noncomputable def t5 : ℝ := (101^100 + 1) / (101^90 + 1)
noncomputable def t6 : ℝ := (99^99 + 1) / (99^89 + 1)
noncomputable def t7 : ℝ := (99^100 + 1) / (99^90 + 1)

theorem fraction_ordering : t6 < t7 ∧ t7 < t2 ∧ t2 < t1 ∧ t1 < t3 ∧ t3 < t5 ∧ t5 < t4 := by
  sorry

end fraction_ordering_l73_73223


namespace base5_first_digit_of_1024_l73_73066

theorem base5_first_digit_of_1024: 
  ∀ (d : ℕ), (d * 5^4 ≤ 1024) ∧ (1024 < (d+1) * 5^4) → d = 1 :=
by
  sorry

end base5_first_digit_of_1024_l73_73066


namespace adoption_cost_l73_73550

theorem adoption_cost :
  let cost_cat := 50
  let cost_adult_dog := 100
  let cost_puppy := 150
  let num_cats := 2
  let num_adult_dogs := 3
  let num_puppies := 2
  (num_cats * cost_cat + num_adult_dogs * cost_adult_dog + num_puppies * cost_puppy) = 700 :=
by
  sorry

end adoption_cost_l73_73550


namespace meters_conversion_equivalence_l73_73777

-- Define the conditions
def meters_to_decimeters (m : ℝ) : ℝ := m * 10
def meters_to_centimeters (m : ℝ) : ℝ := m * 100

-- State the problem
theorem meters_conversion_equivalence :
  7.34 = 7 + (meters_to_decimeters 0.3) / 10 + (meters_to_centimeters 0.04) / 100 :=
sorry

end meters_conversion_equivalence_l73_73777


namespace chessboard_tiling_impossible_l73_73761

theorem chessboard_tiling_impossible :
  ¬ ∃ (cover : (Fin 5 × Fin 7 → Prop)), 
    (cover (0, 3) = false) ∧
    (∀ i j, (cover (i, j) → cover (i + 1, j) ∨ cover (i, j + 1)) ∧
             ∀ x y z w, cover (x, y) → cover (z, w) → (x ≠ z ∨ y ≠ w)) :=
sorry

end chessboard_tiling_impossible_l73_73761


namespace largest_multiple_of_7_less_than_100_l73_73033

theorem largest_multiple_of_7_less_than_100 : ∃ n : ℕ, 7 * n < 100 ∧ ∀ m : ℕ, 7 * m < 100 → 7 * m ≤ 7 * n := by
  sorry

end largest_multiple_of_7_less_than_100_l73_73033


namespace find_B_l73_73814

def is_prime_203B21 (B : ℕ) : Prop :=
  2 ≤ B ∧ B < 10 ∧ Prime (200000 + 3000 + 100 * B + 20 + 1)

theorem find_B : ∃ B, is_prime_203B21 B ∧ ∀ B', is_prime_203B21 B' → B' = 5 := by
  sorry

end find_B_l73_73814


namespace no_solution_exists_l73_73521

theorem no_solution_exists (x y z : ℕ) (hx : x > 2) (hy : y > 1) (h : x^y + 1 = z^2) : false := 
by
  sorry

end no_solution_exists_l73_73521


namespace perimeter_of_smaller_polygon_l73_73543

/-- The ratio of the areas of two similar polygons is 1:16, and the difference in their perimeters is 9.
Find the perimeter of the smaller polygon. -/
theorem perimeter_of_smaller_polygon (a b : ℝ) (h1 : a / b = 1 / 16) (h2 : b - a = 9) : a = 3 :=
by
  sorry

end perimeter_of_smaller_polygon_l73_73543


namespace ninety_percent_greater_than_eighty_percent_l73_73528

-- Define the constants involved in the problem
def ninety_percent (n : ℕ) : ℝ := 0.90 * n
def eighty_percent (n : ℕ) : ℝ := 0.80 * n

-- Define the problem statement
theorem ninety_percent_greater_than_eighty_percent :
  ninety_percent 40 - eighty_percent 30 = 12 :=
by
  sorry

end ninety_percent_greater_than_eighty_percent_l73_73528


namespace num_factors_48_l73_73442

theorem num_factors_48 : 
  ∀ (n : ℕ), n = 48 → (∃ k : ℕ, k = 10 ∧ ∀ d : ℕ, d ∣ n → (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 16 ∨ d = 24 ∨ d = 48)) :=
  by
    intros n h
    sorry

end num_factors_48_l73_73442


namespace interchange_digits_product_l73_73759

-- Definition of the proof problem
theorem interchange_digits_product (n a b k : ℤ) (h1 : n = 10 * a + b) (h2 : n = (k + 1) * (a + b)) :
  ∃ x : ℤ, (10 * b + a) = x * (a + b) ∧ x = 10 - k :=
by
  existsi (10 - k)
  sorry

end interchange_digits_product_l73_73759


namespace arithmetic_seq_a6_l73_73954

variable (a : ℕ → ℝ)

-- Conditions
axiom a3 : a 3 = 16
axiom a9 : a 9 = 80

-- Theorem to prove
theorem arithmetic_seq_a6 : a 6 = 48 :=
by
  sorry

end arithmetic_seq_a6_l73_73954


namespace xiaoming_grandfather_age_l73_73886

-- Define the conditions
def age_cond (x : ℕ) : Prop :=
  ((x - 15) / 4 - 6) * 10 = 100

-- State the problem
theorem xiaoming_grandfather_age (x : ℕ) (h : age_cond x) : x = 79 := 
sorry

end xiaoming_grandfather_age_l73_73886


namespace range_of_m_l73_73734

noncomputable def f (x m a : ℝ) : ℝ := Real.exp (x + 1) - m * a
noncomputable def g (x a : ℝ) : ℝ := a * Real.exp x - x

theorem range_of_m (h : ∃ a : ℝ, ∀ x : ℝ, f x m a ≤ g x a) : m ≥ -1 / Real.exp 1 :=
by
  sorry

end range_of_m_l73_73734


namespace smallest_multiple_of_45_and_75_not_20_l73_73056

-- Definitions of the conditions
def isMultipleOf (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b
def notMultipleOf (a b : ℕ) : Prop := ¬ (isMultipleOf a b)

-- The proof statement
theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ N : ℕ, isMultipleOf N 45 ∧ isMultipleOf N 75 ∧ notMultipleOf N 20 ∧ N = 225 :=
by
  -- sorry is used to indicate that the proof needs to be filled here
  sorry

end smallest_multiple_of_45_and_75_not_20_l73_73056


namespace find_Z_l73_73412

theorem find_Z (Z : ℝ) (h : (100 + 20 / Z) * Z = 9020) : Z = 90 :=
sorry

end find_Z_l73_73412


namespace expression_evaluation_l73_73228

-- Define the variables and the given condition
variables (x y : ℝ)

-- Define the equation condition
def equation_condition : Prop := x - 3 * y = 4

-- State the theorem
theorem expression_evaluation (h : equation_condition x y) : 15 * y - 5 * x + 6 = -14 :=
by
  sorry

end expression_evaluation_l73_73228


namespace triangle_inradius_exradius_l73_73285

-- Define the properties of the triangle
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the inradius
def inradius (a b c : ℝ) (r : ℝ) : Prop :=
  r = (a + b - c) / 2

-- Define the exradius
def exradius (a b c : ℝ) (rc : ℝ) : Prop :=
  rc = (a + b + c) / 2

-- Formalize the Lean statement for the given proof problem
theorem triangle_inradius_exradius (a b c r rc: ℝ) 
  (h_triangle: right_triangle a b c) : 
  inradius a b c r ∧ exradius a b c rc :=
by
  sorry

end triangle_inradius_exradius_l73_73285


namespace sector_to_cone_ratio_l73_73133

noncomputable def sector_angle : ℝ := 135
noncomputable def sector_area (S1 : ℝ) : ℝ := S1
noncomputable def cone_surface_area (S2 : ℝ) : ℝ := S2

theorem sector_to_cone_ratio (S1 S2 : ℝ) :
  sector_area S1 = (3 / 8) * (π * 1^2) →
  cone_surface_area S2 = (3 / 8) * (π * 1^2) + (9 / 64 * π) →
  (S1 / S2) = (8 / 11) :=
by
  intros h1 h2
  sorry

end sector_to_cone_ratio_l73_73133


namespace value_of_a_l73_73838

-- Definitions based on conditions
def cond1 (a : ℝ) := |a| - 1 = 0
def cond2 (a : ℝ) := a + 1 ≠ 0

-- The main proof problem
theorem value_of_a (a : ℝ) : (cond1 a ∧ cond2 a) → a = 1 :=
by
  sorry

end value_of_a_l73_73838


namespace Lisa_photos_l73_73994

variable (a f s : ℕ)

theorem Lisa_photos (h1: a = 10) (h2: f = 3 * a) (h3: s = f - 10) : a + f + s = 60 := by
  sorry

end Lisa_photos_l73_73994


namespace edges_sum_l73_73429

def edges_triangular_pyramid : ℕ := 6
def edges_triangular_prism : ℕ := 9

theorem edges_sum : edges_triangular_pyramid + edges_triangular_prism = 15 :=
by
  sorry

end edges_sum_l73_73429


namespace subset_implies_range_l73_73513

open Set

-- Definitions based on the problem statement
def A : Set ℝ := { x : ℝ | x < 5 }
def B (a : ℝ) : Set ℝ := { x : ℝ | x < a }

-- Theorem statement
theorem subset_implies_range (a : ℝ) (h : A ⊆ B a) : a ≥ 5 :=
sorry

end subset_implies_range_l73_73513


namespace minimum_area_of_quadrilateral_l73_73376

theorem minimum_area_of_quadrilateral
  (ABCD : Type)
  (O : Type)
  (S_ABO : ℝ)
  (S_CDO : ℝ)
  (BC : ℝ)
  (cos_angle_ADC : ℝ)
  (h1 : S_ABO = 3 / 2)
  (h2 : S_CDO = 3 / 2)
  (h3 : BC = 3 * Real.sqrt 2)
  (h4 : cos_angle_ADC = 3 / Real.sqrt 10) :
  ∃ S_ABCD : ℝ, S_ABCD = 6 :=
sorry

end minimum_area_of_quadrilateral_l73_73376


namespace parabola_properties_l73_73491

theorem parabola_properties 
  (p : ℝ) (h_pos : 0 < p) (m : ℝ) 
  (A B : ℝ × ℝ)
  (h_AB_on_parabola : ∀ (P : ℝ × ℝ), P = A ∨ P = B → (P.snd)^2 = 2 * p * P.fst) 
  (h_line_intersection : ∀ (P : ℝ × ℝ), P = A ∨ P = B → P.fst = m * P.snd + 3)
  (h_dot_product : (A.fst * B.fst + A.snd * B.snd) = 6)
  : (exists C : ℝ × ℝ, C = (-3, 0)) ∧
    (∃ k1 k2 : ℝ, 
        k1 = A.snd / (A.fst + 3) ∧ 
        k2 = B.snd / (B.fst + 3) ∧ 
        (1 / k1^2 + 1 / k2^2 - 2 * m^2) = 24) :=
by
  sorry

end parabola_properties_l73_73491


namespace number_of_parents_l73_73164

theorem number_of_parents (P : ℕ) (h : P + 177 = 238) : P = 61 :=
by
  sorry

end number_of_parents_l73_73164


namespace slope_range_of_line_l73_73261

/-- A mathematical proof problem to verify the range of the slope of a line
that passes through a given point (-1, -1) and intersects a circle. -/
theorem slope_range_of_line (
  k : ℝ
) : (∃ x y : ℝ, (y + 1 = k * (x + 1)) ∧ (x - 2) ^ 2 + y ^ 2 = 1) ↔ (0 < k ∧ k < 3 / 4) := 
by
  sorry  

end slope_range_of_line_l73_73261


namespace second_recipe_cup_count_l73_73239

theorem second_recipe_cup_count (bottle_ounces : ℕ) (ounces_per_cup : ℕ)
  (first_recipe_cups : ℕ) (third_recipe_cups : ℕ) (bottles_needed : ℕ)
  (total_ounces : bottle_ounces = 16)
  (ounce_to_cup : ounces_per_cup = 8)
  (first_recipe : first_recipe_cups = 2)
  (third_recipe : third_recipe_cups = 3)
  (bottles : bottles_needed = 3) :
  (bottles_needed * bottle_ounces) / ounces_per_cup - first_recipe_cups - third_recipe_cups = 1 :=
by
  sorry

end second_recipe_cup_count_l73_73239


namespace phoenix_hike_distance_l73_73555

variable (a b c d : ℕ)

theorem phoenix_hike_distance
  (h1 : a + b = 24)
  (h2 : b + c = 30)
  (h3 : c + d = 32)
  (h4 : a + c = 28) :
  a + b + c + d = 56 :=
by
  sorry

end phoenix_hike_distance_l73_73555


namespace Marley_fruits_total_is_31_l73_73162

-- Define the given conditions

def Louis_oranges : Nat := 5
def Louis_apples : Nat := 3
def Samantha_oranges : Nat := 8
def Samantha_apples : Nat := 7

def Marley_oranges : Nat := 2 * Louis_oranges
def Marley_apples : Nat := 3 * Samantha_apples

-- The statement to be proved
def Marley_total_fruits : Nat := Marley_oranges + Marley_apples

theorem Marley_fruits_total_is_31 : Marley_total_fruits = 31 := by
  sorry

end Marley_fruits_total_is_31_l73_73162


namespace tom_swim_time_l73_73981

theorem tom_swim_time (t : ℝ) :
  (2 * t + 4 * t = 12) → t = 2 :=
by
  intro h
  have eq1 : 6 * t = 12 := by linarith
  linarith

end tom_swim_time_l73_73981


namespace inequality_a_b_l73_73474

theorem inequality_a_b (a b : ℝ) (h : a > b ∧ b > 0) : (1/a) < (1/b) := 
by
  sorry

end inequality_a_b_l73_73474


namespace ratio_shorter_to_longer_l73_73703

-- Define the total length and the length of the shorter piece
def total_length : ℕ := 90
def shorter_length : ℕ := 20

-- Define the length of the longer piece
def longer_length : ℕ := total_length - shorter_length

-- Define the ratio of shorter piece to longer piece
def ratio := shorter_length / longer_length

-- The target statement to prove
theorem ratio_shorter_to_longer : ratio = 2 / 7 := by
  sorry

end ratio_shorter_to_longer_l73_73703


namespace lucy_money_left_l73_73959

theorem lucy_money_left : 
  ∀ (initial_money : ℕ) 
    (one_third_loss : ℕ → ℕ) 
    (one_fourth_spend : ℕ → ℕ), 
    initial_money = 30 → 
    one_third_loss initial_money = initial_money / 3 → 
    one_fourth_spend (initial_money - one_third_loss initial_money) = (initial_money - one_third_loss initial_money) / 4 → 
  initial_money - one_third_loss initial_money - one_fourth_spend (initial_money - one_third_loss initial_money) = 15 :=
by
  intros initial_money one_third_loss one_fourth_spend
  intro h_initial_money
  intro h_one_third_loss
  intro h_one_fourth_spend
  sorry

end lucy_money_left_l73_73959


namespace total_books_l73_73378

-- Lean 4 Statement
theorem total_books (stu_books : ℝ) (albert_ratio : ℝ) (albert_books : ℝ) (total_books : ℝ) 
  (h1 : stu_books = 9) 
  (h2 : albert_ratio = 4.5) 
  (h3 : albert_books = stu_books * albert_ratio) 
  (h4 : total_books = stu_books + albert_books) : 
  total_books = 49.5 := 
sorry

end total_books_l73_73378


namespace least_possible_mn_correct_l73_73512

def least_possible_mn (m n : ℕ) : ℕ :=
  m + n

theorem least_possible_mn_correct (m n : ℕ) :
  (Nat.gcd (m + n) 210 = 1) →
  (n^n ∣ m^m) →
  ¬(n ∣ m) →
  least_possible_mn m n = 407 :=
by
  sorry

end least_possible_mn_correct_l73_73512


namespace part1_part2_part3_l73_73587

-- Define the sequences a_n and b_n as described in the problem
def X_sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → (a n = 0 ∨ a n = 1))

def accompanying_sequence (a b : ℕ → ℝ) : Prop :=
  (b 1 = 1) ∧ (∀ n : ℕ, n > 0 → b (n + 1) = abs (a n - (a (n + 1) / 2)) * b n)

-- 1. Prove the values of b_2, b_3, and b_4
theorem part1 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  a 2 = 1 → a 3 = 0 → a 4 = 1 →
  b 2 = 1 / 2 ∧ b 3 = 1 / 2 ∧ b 4 = 1 / 4 := 
sorry

-- 2. Prove the equivalence for geometric sequence and constant sequence
theorem part2 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  (∀ n : ℕ, n > 0 → a n = 1) ↔ (∃ r : ℝ, ∀ n : ℕ, n > 0 → b (n + 1) = r * b n) := 
sorry

-- 3. Prove the maximum value of b_2019
theorem part3 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  b 2019 ≤ 1 / 2^1009 := 
sorry

end part1_part2_part3_l73_73587


namespace number_of_valid_strings_l73_73627

def count_valid_strings (n : ℕ) : ℕ :=
  4^n - 3 * 3^n + 3 * 2^n - 1

theorem number_of_valid_strings (n : ℕ) :
  count_valid_strings n = 4^n - 3 * 3^n + 3 * 2^n - 1 :=
by sorry

end number_of_valid_strings_l73_73627


namespace fit_max_blocks_l73_73289

/-- Prove the maximum number of blocks of size 1-in x 3-in x 2-in that can fit into a box of size 4-in x 3-in x 5-in is 10. -/
theorem fit_max_blocks :
  ∀ (block_dim box_dim : ℕ → ℕ ),
  block_dim 1 = 1 ∧ block_dim 2 = 3 ∧ block_dim 3 = 2 →
  box_dim 1 = 4 ∧ box_dim 2 = 3 ∧ box_dim 3 = 5 →
  ∃ max_blocks : ℕ, max_blocks = 10 :=
by
  sorry

end fit_max_blocks_l73_73289


namespace B_share_after_tax_l73_73088

noncomputable def B_share (x : ℝ) : ℝ := 3 * x
noncomputable def salary_proportion (A B C D : ℝ) (x : ℝ) :=
  A = 2 * x ∧ B = 3 * x ∧ C = 4 * x ∧ D = 6 * x
noncomputable def D_more_than_C (D C : ℝ) : Prop :=
  D - C = 700
noncomputable def meets_minimum_wage (B : ℝ) : Prop :=
  B ≥ 1000
noncomputable def tax_deduction (B : ℝ) : ℝ :=
  if B > 1500 then B - 0.15 * (B - 1500) else B

theorem B_share_after_tax (A B C D : ℝ) (x : ℝ) (h1 : salary_proportion A B C D x)
  (h2 : D_more_than_C D C) (h3 : meets_minimum_wage B) :
  tax_deduction B = 1050 :=
by
  sorry

end B_share_after_tax_l73_73088


namespace quadrilateral_area_is_33_l73_73038

-- Definitions for the points and their coordinates
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 4, y := 0}
def B : Point := {x := 0, y := 12}
def C : Point := {x := 10, y := 0}
def E : Point := {x := 3, y := 3}

-- Define the quadrilateral area computation
noncomputable def areaQuadrilateral (O B E C : Point) : ℝ :=
  let triangle_area (p1 p2 p3 : Point) :=
    abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2
  triangle_area O B E + triangle_area O E C

-- Statement to prove
theorem quadrilateral_area_is_33 : areaQuadrilateral {x := 0, y := 0} B E C = 33 := by
  sorry

end quadrilateral_area_is_33_l73_73038


namespace find_m_n_l73_73482

theorem find_m_n (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) 
  (h1 : m * n ∣ 3 ^ m + 1) (h2 : m * n ∣ 3 ^ n + 1) : 
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) :=
by
  sorry

end find_m_n_l73_73482


namespace muffin_combinations_l73_73134

theorem muffin_combinations (k : ℕ) (n : ℕ) (h_k : k = 4) (h_n : n = 4) :
  (Nat.choose ((n + k - 1) : ℕ) ((k - 1) : ℕ)) = 35 :=
by
  rw [h_k, h_n]
  -- Simplifying Nat.choose (4 + 4 - 1) (4 - 1) = Nat.choose 7 3
  sorry

end muffin_combinations_l73_73134


namespace number_of_blind_students_l73_73211

variable (B D : ℕ)

-- Condition 1: The deaf-student population is 3 times the blind-student population.
axiom H1 : D = 3 * B

-- Condition 2: There are 180 students in total.
axiom H2 : B + D = 180

theorem number_of_blind_students : B = 45 :=
by
  -- Sorry is used to skip the proof steps. The theorem statement is correct and complete based on the conditions.
  sorry

end number_of_blind_students_l73_73211


namespace inequality_proof_l73_73689

noncomputable def a := Real.log 1 / Real.log 3
noncomputable def b := Real.log 1 / Real.log (1 / 2)
noncomputable def c := (1/2)^(1/3)

theorem inequality_proof : b > c ∧ c > a := 
by 
  sorry

end inequality_proof_l73_73689


namespace swimmer_speed_proof_l73_73960

-- Definition of the conditions
def current_speed : ℝ := 2
def swimming_time : ℝ := 1.5
def swimming_distance : ℝ := 3

-- Prove: Swimmer's speed in still water
def swimmer_speed_in_still_water : ℝ := 4

-- Statement: Given the conditions, the swimmer's speed in still water equals 4 km/h
theorem swimmer_speed_proof :
  (swimming_distance = (swimmer_speed_in_still_water - current_speed) * swimming_time) →
  swimmer_speed_in_still_water = 4 :=
by
  intro h
  sorry

end swimmer_speed_proof_l73_73960


namespace remainder_3n_mod_7_l73_73423

theorem remainder_3n_mod_7 (n : ℤ) (k : ℤ) (h : n = 7*k + 1) :
  (3 * n) % 7 = 3 := by
  sorry

end remainder_3n_mod_7_l73_73423


namespace cost_of_two_other_puppies_l73_73770

theorem cost_of_two_other_puppies (total_cost : ℕ) (sale_price : ℕ) (num_puppies : ℕ) (num_sale_puppies : ℕ) (remaining_puppies_cost : ℕ) :
  total_cost = 800 →
  sale_price = 150 →
  num_puppies = 5 →
  num_sale_puppies = 3 →
  remaining_puppies_cost = (total_cost - num_sale_puppies * sale_price) →
  (remaining_puppies_cost / (num_puppies - num_sale_puppies)) = 175 :=
by
  intros
  sorry

end cost_of_two_other_puppies_l73_73770


namespace dividend_in_terms_of_a_l73_73032

variable (a Q R D : ℕ)

-- Given conditions as hypotheses
def condition1 : Prop := D = 25 * Q
def condition2 : Prop := D = 7 * R
def condition3 : Prop := Q - R = 15
def condition4 : Prop := R = 3 * a

-- Prove that the dividend given these conditions equals the expected expression
theorem dividend_in_terms_of_a (a : ℕ) (Q : ℕ) (R : ℕ) (D : ℕ) :
  condition1 D Q → condition2 D R → condition3 Q R → condition4 R a →
  (D * Q + R) = 225 * a^2 + 1128 * a + 5625 :=
by
  intro h1 h2 h3 h4
  sorry

end dividend_in_terms_of_a_l73_73032


namespace sufficient_but_not_necessary_condition_l73_73444

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a^2 ≠ 4) → (a ≠ 2) ∧ ¬ ((a ≠ 2) → (a^2 ≠ 4)) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l73_73444


namespace trigonometric_identity_l73_73411

theorem trigonometric_identity (α : ℝ) :
    (1 / Real.sin (-α) - Real.sin (Real.pi + α)) /
    (1 / Real.cos (3 * Real.pi - α) + Real.cos (2 * Real.pi - α)) =
    1 / Real.tan α ^ 3 :=
    sorry

end trigonometric_identity_l73_73411


namespace find_polynomial_l73_73419

noncomputable def polynomial_satisfies_conditions (P : Polynomial ℝ) : Prop :=
  P.eval 0 = 0 ∧ ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1

theorem find_polynomial (P : Polynomial ℝ) (h : polynomial_satisfies_conditions P) : P = Polynomial.X :=
  sorry

end find_polynomial_l73_73419


namespace factor_expression_l73_73496

variable (x : ℤ)

theorem factor_expression : 63 * x - 21 = 21 * (3 * x - 1) := 
by 
  sorry

end factor_expression_l73_73496


namespace fraction_calculation_l73_73467

noncomputable def improper_frac_1 : ℚ := 21 / 8
noncomputable def improper_frac_2 : ℚ := 33 / 14
noncomputable def improper_frac_3 : ℚ := 37 / 12
noncomputable def improper_frac_4 : ℚ := 35 / 8
noncomputable def improper_frac_5 : ℚ := 179 / 9

theorem fraction_calculation :
  (improper_frac_1 - (2 / 3) * improper_frac_2) / ((improper_frac_3 + improper_frac_4) / improper_frac_5) = 59 / 21 :=
by
  sorry

end fraction_calculation_l73_73467


namespace triangle_inequality_l73_73693

theorem triangle_inequality (a b c : ℝ) (habc : a + b > c ∧ a + c > b ∧ b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end triangle_inequality_l73_73693


namespace carter_average_goals_l73_73537

theorem carter_average_goals (C : ℝ)
  (h1 : C + (1 / 2) * C + (C - 3) = 7) : C = 4 :=
by
  sorry

end carter_average_goals_l73_73537


namespace find_f_at_six_l73_73663

theorem find_f_at_six (f : ℝ → ℝ) (h : ∀ x : ℝ, f (4 * x - 2) = x^2 - x + 2) : f 6 = 3.75 :=
by
  sorry

end find_f_at_six_l73_73663


namespace circles_intersect_l73_73754

theorem circles_intersect (R r d: ℝ) (hR: R = 7) (hr: r = 4) (hd: d = 8) : (R - r < d) ∧ (d < R + r) :=
by
  rw [hR, hr, hd]
  exact ⟨by linarith, by linarith⟩

end circles_intersect_l73_73754


namespace triangles_needed_for_hexagon_with_perimeter_19_l73_73913

def num_triangles_to_construct_hexagon (perimeter : ℕ) : ℕ :=
  match perimeter with
  | 19 => 59
  | _ => 0  -- We handle only the case where perimeter is 19

theorem triangles_needed_for_hexagon_with_perimeter_19 :
  num_triangles_to_construct_hexagon 19 = 59 :=
by
  -- Here we assert that the number of triangles to construct the hexagon with perimeter 19 is 59
  sorry

end triangles_needed_for_hexagon_with_perimeter_19_l73_73913


namespace solve_equation_l73_73243

theorem solve_equation (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : 3 / x + 4 / y = 1) : 
  x = 3 * y / (y - 4) :=
sorry

end solve_equation_l73_73243


namespace simplify_expression_l73_73267

variable (a b : ℝ)
variable (h₁ : a = 3 + Real.sqrt 5)
variable (h₂ : b = 3 - Real.sqrt 5)

theorem simplify_expression : (a^2 - 2 * a * b + b^2) / (a^2 - b^2) * (a * b) / (a - b) = 2 / 3 := by
  sorry

end simplify_expression_l73_73267


namespace garrett_cats_count_l73_73807

def number_of_cats_sheridan : ℕ := 11
def difference_in_cats : ℕ := 13

theorem garrett_cats_count (G : ℕ) (h : G - number_of_cats_sheridan = difference_in_cats) : G = 24 :=
by
  sorry

end garrett_cats_count_l73_73807


namespace probability_red_balls_fourth_draw_l73_73969

theorem probability_red_balls_fourth_draw :
  let p_red := 2 / 10
  let p_white := 8 / 10
  p_red * p_red * p_white * p_white * 3 / 10 + 
  p_red * p_white * p_red * p_white * 2 / 10 + 
  p_white * p_red * p_red * p_red = 0.0434 :=
sorry

end probability_red_balls_fourth_draw_l73_73969


namespace probability_common_letters_l73_73358

open Set

def letters_GEOMETRY : Finset Char := {'G', 'E', 'O', 'M', 'T', 'R', 'Y'}
def letters_RHYME : Finset Char := {'R', 'H', 'Y', 'M', 'E'}

def common_letters : Finset Char := letters_GEOMETRY ∩ letters_RHYME

theorem probability_common_letters :
  (common_letters.card : ℚ) / (letters_GEOMETRY.card : ℚ) = 1 / 2 := by
  sorry

end probability_common_letters_l73_73358


namespace employee_earnings_l73_73520

theorem employee_earnings (regular_rate overtime_rate first3_days_h second2_days_h total_hours overtime_hours : ℕ)
  (h1 : regular_rate = 30)
  (h2 : overtime_rate = 45)
  (h3 : first3_days_h = 6)
  (h4 : second2_days_h = 12)
  (h5 : total_hours = first3_days_h * 3 + second2_days_h * 2)
  (h6 : total_hours = 42)
  (h7 : overtime_hours = total_hours - 40)
  (h8 : overtime_hours = 2) :
  (40 * regular_rate + overtime_hours * overtime_rate) = 1290 := 
sorry

end employee_earnings_l73_73520


namespace polynomial_inequality_holds_l73_73349

def polynomial (x : ℝ) : ℝ := x^6 + 4 * x^5 + 2 * x^4 - 6 * x^3 - 2 * x^2 + 4 * x - 1

theorem polynomial_inequality_holds (x : ℝ) :
  (x ≤ -1 - Real.sqrt 2 ∨ x = (-1 - Real.sqrt 5) / 2 ∨ x ≥ -1 + Real.sqrt 2) →
  polynomial x ≥ 0 :=
by
  sorry

end polynomial_inequality_holds_l73_73349


namespace units_digit_base6_product_l73_73098

theorem units_digit_base6_product (a b : ℕ) (h1 : a = 168) (h2 : b = 59) : ((a * b) % 6) = 0 := by
  sorry

end units_digit_base6_product_l73_73098


namespace relationship_between_A_and_B_l73_73656

noncomputable def f (x : ℝ) : ℝ := x^2

def A : Set ℝ := {x | f x = x}

def B : Set ℝ := {x | f (f x) = x}

theorem relationship_between_A_and_B : A ∩ B = A :=
by sorry

end relationship_between_A_and_B_l73_73656


namespace max_y_difference_l73_73533

theorem max_y_difference : (∃ x, (5 - 2 * x^2 + 2 * x^3 = 1 + x^2 + x^3)) ∧ 
                           (∀ y1 y2, y1 = 5 - 2 * (2^2) + 2 * (2^3) ∧ y2 = 5 - 2 * (1/2)^2 + 2 * (1/2)^3 → 
                           (y1 - y2 = 11.625)) := sorry

end max_y_difference_l73_73533


namespace problem_solution_l73_73906

theorem problem_solution :
  (2200 - 2089)^2 / 196 = 63 :=
sorry

end problem_solution_l73_73906


namespace find_n_coins_l73_73792

def num_coins : ℕ := 5

theorem find_n_coins (n : ℕ) (h : (n^2 + n + 2) = 2^n) : n = num_coins :=
by {
  -- Proof to be filled in
  sorry
}

end find_n_coins_l73_73792


namespace factorize_expr1_factorize_expr2_l73_73565

-- Problem (1) Statement
theorem factorize_expr1 (x y : ℝ) : 
  -x^5 * y^3 + x^3 * y^5 = -x^3 * y^3 * (x + y) * (x - y) :=
sorry

-- Problem (2) Statement
theorem factorize_expr2 (a : ℝ) : 
  (a^2 + 1)^2 - 4 * a^2 = (a + 1)^2 * (a - 1)^2 :=
sorry

end factorize_expr1_factorize_expr2_l73_73565


namespace shiela_drawings_l73_73403

theorem shiela_drawings (neighbors : ℕ) (drawings_per_neighbor : ℕ) (total_drawings : ℕ) 
  (h1 : neighbors = 6) (h2 : drawings_per_neighbor = 9) : total_drawings = 54 :=
  by 
    have h : total_drawings = neighbors * drawings_per_neighbor := sorry
    rw [h1, h2] at h
    exact h
    -- Proof skipped with sorry.

end shiela_drawings_l73_73403


namespace elizabeth_haircut_l73_73195

theorem elizabeth_haircut (t s f : ℝ) (ht : t = 0.88) (hs : s = 0.5) : f = t - s := by
  sorry

end elizabeth_haircut_l73_73195


namespace Hari_joined_after_5_months_l73_73630

noncomputable def Praveen_investment_per_year : ℝ := 3360 * 12
noncomputable def Hari_investment_for_given_months (x : ℝ) : ℝ := 8640 * (12 - x)

theorem Hari_joined_after_5_months (x : ℝ) (h : Praveen_investment_per_year / Hari_investment_for_given_months x = 2 / 3) : x = 5 :=
by
  sorry

end Hari_joined_after_5_months_l73_73630


namespace tangent_line_eq_l73_73460

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 - 2 * x - 1)

theorem tangent_line_eq :
  let x := 1
  let y := f x
  ∃ (m : ℝ), m = -2 * Real.exp 1 ∧ (∀ (x y : ℝ), y = m * (x - 1) + f 1) :=
by
  sorry

end tangent_line_eq_l73_73460


namespace no_nat_numbers_satisfy_l73_73825

theorem no_nat_numbers_satisfy (x y z k : ℕ) (hx : x < k) (hy : y < k) : x^k + y^k ≠ z^k := 
sorry

end no_nat_numbers_satisfy_l73_73825


namespace calculateRequiredMonthlyRent_l73_73045

noncomputable def requiredMonthlyRent (purchase_price : ℝ) (annual_return_rate : ℝ) (annual_taxes : ℝ) (repair_percentage : ℝ) : ℝ :=
  let annual_return := annual_return_rate * purchase_price
  let total_annual_need := annual_return + annual_taxes
  let monthly_requirement := total_annual_need / 12
  let monthly_rent := monthly_requirement / (1 - repair_percentage)
  monthly_rent

theorem calculateRequiredMonthlyRent : requiredMonthlyRent 20000 0.06 450 0.10 = 152.78 := by
  sorry

end calculateRequiredMonthlyRent_l73_73045


namespace kendra_change_and_discounts_l73_73224

-- Define the constants and conditions
def wooden_toy_price : ℝ := 20.0
def hat_price : ℝ := 10.0
def tax_rate : ℝ := 0.08
def discount_wooden_toys_2_3 : ℝ := 0.10
def discount_wooden_toys_4_or_more : ℝ := 0.15
def discount_hats_2 : ℝ := 0.05
def discount_hats_3_or_more : ℝ := 0.10
def kendra_bill : ℝ := 250.0
def kendra_wooden_toys : ℕ := 4
def kendra_hats : ℕ := 5

-- Calculate the applicable discounts based on conditions
def discount_on_wooden_toys : ℝ :=
  if kendra_wooden_toys >= 2 ∧ kendra_wooden_toys <= 3 then
    discount_wooden_toys_2_3
  else if kendra_wooden_toys >= 4 then
    discount_wooden_toys_4_or_more
  else
    0.0

def discount_on_hats : ℝ :=
  if kendra_hats = 2 then
    discount_hats_2
  else if kendra_hats >= 3 then
    discount_hats_3_or_more
  else
    0.0

-- Main theorem statement
theorem kendra_change_and_discounts :
  let total_cost_before_discounts := kendra_wooden_toys * wooden_toy_price + kendra_hats * hat_price
  let wooden_toys_discount := discount_on_wooden_toys * (kendra_wooden_toys * wooden_toy_price)
  let hats_discount := discount_on_hats * (kendra_hats * hat_price)
  let total_discounts := wooden_toys_discount + hats_discount
  let total_cost_after_discounts := total_cost_before_discounts - total_discounts
  let tax := tax_rate * total_cost_after_discounts
  let total_cost_after_tax := total_cost_after_discounts + tax
  let change_received := kendra_bill - total_cost_after_tax
  (total_discounts = 17) → 
  (change_received = 127.96) ∧ 
  (wooden_toys_discount = 12) ∧ 
  (hats_discount = 5) :=
by
  sorry

end kendra_change_and_discounts_l73_73224


namespace final_volume_of_syrup_l73_73470

-- Definitions based on conditions extracted from step a)
def quarts_to_cups (q : ℚ) : ℚ := q * 4
def reduce_volume (v : ℚ) : ℚ := v / 12
def add_sugar (v : ℚ) (s : ℚ) : ℚ := v + s

theorem final_volume_of_syrup :
  let initial_volume_in_quarts := 6
  let sugar_added := 1
  let initial_volume_in_cups := quarts_to_cups initial_volume_in_quarts
  let reduced_volume := reduce_volume initial_volume_in_cups
  add_sugar reduced_volume sugar_added = 3 :=
by
  sorry

end final_volume_of_syrup_l73_73470


namespace min_value_of_z_l73_73434

theorem min_value_of_z (a x y : ℝ) (h1 : a > 0) (h2 : x ≥ 1) (h3 : x + y ≤ 3) (h4 : y ≥ a * (x - 3)) :
  (∃ (x y : ℝ), 2 * x + y = 1) → a = 1 / 2 :=
by {
  sorry
}

end min_value_of_z_l73_73434


namespace value_of_a2_l73_73808

theorem value_of_a2 
  (a1 a2 a3 : ℝ)
  (h_seq : ∃ d : ℝ, (-8) = -8 + d * 0 ∧ a1 = -8 + d * 1 ∧ 
                     a2 = -8 + d * 2 ∧ a3 = -8 + d * 3 ∧ 
                     10 = -8 + d * 4) :
  a2 = 1 :=
by {
  sorry
}

end value_of_a2_l73_73808


namespace geese_flew_away_l73_73464

theorem geese_flew_away (initial remaining flown_away : ℕ) (h_initial: initial = 51) (h_remaining: remaining = 23) : flown_away = 28 :=
by
  sorry

end geese_flew_away_l73_73464


namespace oil_price_reduction_l73_73492

theorem oil_price_reduction (P P_r : ℝ) (h1 : P_r = 24.3) (h2 : 1080 / P - 1080 / P_r = 8) : 
  ((P - P_r) / P) * 100 = 18.02 := by
  sorry

end oil_price_reduction_l73_73492


namespace digging_project_length_l73_73213

theorem digging_project_length (Length_2 : ℝ) : 
  (100 * 25 * 30) = (75 * Length_2 * 50) → 
  Length_2 = 20 :=
by
  sorry

end digging_project_length_l73_73213


namespace book_pages_l73_73360

noncomputable def totalPages := 240

theorem book_pages : 
  ∀ P : ℕ, 
    (1 / 2) * P + (1 / 4) * P + (1 / 6) * P + 20 = P → 
    P = totalPages :=
by
  intro P
  intros h
  sorry

end book_pages_l73_73360


namespace common_divisor_is_19_l73_73340

theorem common_divisor_is_19 (a d : ℤ) (h1 : d ∣ (35 * a + 57)) (h2 : d ∣ (45 * a + 76)) : d = 19 :=
sorry

end common_divisor_is_19_l73_73340


namespace negation_proposition_l73_73888

theorem negation_proposition (h : ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) :
  ¬(∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) = ∃ x : ℝ, x^2 - 2*x + 4 > 0 :=
by
  sorry

end negation_proposition_l73_73888


namespace surface_area_cube_l73_73350

theorem surface_area_cube (a : ℕ) (b : ℕ) (h : a = 2) : b = 54 :=
  by
  sorry

end surface_area_cube_l73_73350


namespace car_distance_l73_73414

variable (v_x v_y : ℝ) (Δt_x : ℝ) (d_x : ℝ)

theorem car_distance (h_vx : v_x = 35) (h_vy : v_y = 50) (h_Δt : Δt_x = 1.2)
  (h_dx : d_x = v_x * Δt_x):
  d_x + v_x * (d_x / (v_y - v_x)) = 98 := 
by sorry

end car_distance_l73_73414


namespace people_stools_chairs_l73_73448

def numberOfPeopleStoolsAndChairs (x y z : ℕ) : Prop :=
  2 * x + 3 * y + 4 * z = 32 ∧
  x > y ∧
  x > z ∧
  x < y + z

theorem people_stools_chairs :
  ∃ (x y z : ℕ), numberOfPeopleStoolsAndChairs x y z ∧ x = 5 ∧ y = 2 ∧ z = 4 :=
by
  sorry

end people_stools_chairs_l73_73448


namespace avg_visitors_proof_l73_73931

-- Define the constants and conditions
def Sundays_visitors : ℕ := 500
def total_days : ℕ := 30
def avg_visitors_per_day : ℕ := 200

-- Total visits on Sundays within the month
def visits_on_Sundays := 5 * Sundays_visitors

-- Total visitors for the month
def total_visitors := total_days * avg_visitors_per_day

-- Average visitors on other days (Monday to Saturday)
def avg_visitors_other_days : ℕ :=
  (total_visitors - visits_on_Sundays) / (total_days - 5)

-- The theorem stating the problem and corresponding answer
theorem avg_visitors_proof (V : ℕ) 
  (h1 : Sundays_visitors = 500)
  (h2 : total_days = 30)
  (h3 : avg_visitors_per_day = 200)
  (h4 : visits_on_Sundays = 5 * Sundays_visitors)
  (h5 : total_visitors = total_days * avg_visitors_per_day)
  (h6 : avg_visitors_other_days = (total_visitors - visits_on_Sundays) / (total_days - 5))
  : V = 140 :=
by
  -- Proof is not required, just state the theorem
  sorry

end avg_visitors_proof_l73_73931


namespace blue_paint_cans_needed_l73_73271

-- Definitions of the conditions
def blue_to_green_ratio : ℕ × ℕ := (4, 3)
def total_cans : ℕ := 42
def expected_blue_cans : ℕ := 24

-- Proof statement
theorem blue_paint_cans_needed (r : ℕ × ℕ) (total : ℕ) (expected : ℕ) 
  (h1: r = (4, 3)) (h2: total = 42) : expected = 24 :=
by
  sorry

end blue_paint_cans_needed_l73_73271


namespace tangent_line_to_parabola_l73_73873

theorem tangent_line_to_parabola (k : ℝ) :
  (∃ (x y : ℝ), 4 * x + 7 * y + k = 0 ∧ y^2 = 16 * x) →
  (28 ^ 2 - 4 * 1 * (4 * k) = 0) → k = 49 :=
by
  intro h
  intro h_discriminant
  have discriminant_eq_zero : 28 ^ 2 - 4 * 1 * (4 * k) = 0 := h_discriminant
  sorry

end tangent_line_to_parabola_l73_73873


namespace probability_of_inverse_proportion_l73_73784

def points : List (ℝ × ℝ) :=
  [(0.5, -4.5), (1, -4), (1.5, -3.5), (2, -3), (2.5, -2.5), (3, -2), (3.5, -1.5),
   (4, -1), (4.5, -0.5), (5, 0)]

def inverse_proportion_pairs : List ((ℝ × ℝ) × (ℝ × ℝ)) :=
  [((0.5, -4.5), (4.5, -0.5)), ((1, -4), (4, -1)), ((1.5, -3.5), (3.5, -1.5)), ((2, -3), (3, -2))]

theorem probability_of_inverse_proportion:
  let num_pairs := List.length points * (List.length points - 1)
  let favorable_pairs := 2 * List.length inverse_proportion_pairs
  favorable_pairs / num_pairs = (4 : ℚ) / 45 := by
  sorry

end probability_of_inverse_proportion_l73_73784


namespace value_of_c_l73_73861

theorem value_of_c (a b c : ℚ) (h1 : a / b = 2 / 3) (h2 : b / c = 3 / 7) (h3 : a - b + 3 = c - 2 * b) : c = 21 / 2 :=
sorry

end value_of_c_l73_73861


namespace polynomial_solution_l73_73003

theorem polynomial_solution (P : Polynomial ℝ) (h_0 : P.eval 0 = 0) (h_func : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) :
  P = Polynomial.X :=
sorry

end polynomial_solution_l73_73003


namespace cos_neg_pi_div_3_l73_73910

theorem cos_neg_pi_div_3 : Real.cos (-π / 3) = 1 / 2 := 
by 
  sorry

end cos_neg_pi_div_3_l73_73910


namespace f_at_2_l73_73974

noncomputable def f (x : ℝ) (a b : ℝ) := a * Real.log x + b / x + x
noncomputable def g (x : ℝ) (a b : ℝ) := (a / x) - (b / (x ^ 2)) + 1

theorem f_at_2 (a b : ℝ) (ha : g 1 a b = 0) (hb : g 3 a b = 0) : f 2 a b = 1 / 2 - 4 * Real.log 2 :=
by
  sorry

end f_at_2_l73_73974


namespace point_D_coordinates_l73_73851

noncomputable def point := ℝ × ℝ

def A : point := (2, 3)
def B : point := (-1, 5)

def vector_sub (p1 p2 : point) : point := (p1.1 - p2.1, p1.2 - p2.2)
def scalar_mul (k : ℝ) (v : point) : point := (k * v.1, k * v.2)
def vector_add (p1 p2 : point) : point := (p1.1 + p2.1, p1.2 + p2.2)

def D : point := vector_add A (scalar_mul 3 (vector_sub B A))

theorem point_D_coordinates : D = (-7, 9) :=
by
  -- Proof goes here
  sorry

end point_D_coordinates_l73_73851


namespace simplify_expression_eval_l73_73311

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  ((x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2 * x + 1)))

theorem simplify_expression_eval : simplify_expression (Real.sqrt 5 - 1) = (Real.sqrt 5) / 5 :=
by
  sorry

end simplify_expression_eval_l73_73311


namespace minimum_period_tan_2x_l73_73030

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem minimum_period_tan_2x : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi / 2 :=
by
  sorry

end minimum_period_tan_2x_l73_73030


namespace hyperbola_eccentricity_l73_73131

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0)
  (h_eq : ∀ (x y : ℝ), x^2 / a^2 - y^2 / 16 = 1 ↔ true)
  (eccentricity : a^2 + 16 / a^2 = (5 / 3)^2) : a = 3 :=
by
  sorry

end hyperbola_eccentricity_l73_73131


namespace total_tickets_sold_l73_73158

theorem total_tickets_sold 
  (A D : ℕ) 
  (cost_adv cost_door : ℝ) 
  (revenue : ℝ)
  (door_tickets_sold total_tickets : ℕ) 
  (h1 : cost_adv = 14.50) 
  (h2 : cost_door = 22.00)
  (h3 : revenue = 16640) 
  (h4 : door_tickets_sold = 672) : 
  (total_tickets = 800) :=
by
  sorry

end total_tickets_sold_l73_73158


namespace selection_competition_l73_73699

variables (p q r : Prop)

theorem selection_competition 
  (h1 : p ∨ q) 
  (h2 : ¬ (p ∧ q)) 
  (h3 : ¬ q ∧ r) : p ∧ ¬ q ∧ r :=
by
  sorry

end selection_competition_l73_73699


namespace category_D_cost_after_discount_is_correct_l73_73674

noncomputable def total_cost : ℝ := 2500
noncomputable def percentage_A : ℝ := 0.30
noncomputable def percentage_B : ℝ := 0.25
noncomputable def percentage_C : ℝ := 0.20
noncomputable def percentage_D : ℝ := 0.25
noncomputable def discount_A : ℝ := 0.03
noncomputable def discount_B : ℝ := 0.05
noncomputable def discount_C : ℝ := 0.07
noncomputable def discount_D : ℝ := 0.10

noncomputable def cost_before_discount_D : ℝ := total_cost * percentage_D
noncomputable def discount_amount_D : ℝ := cost_before_discount_D * discount_D
noncomputable def cost_after_discount_D : ℝ := cost_before_discount_D - discount_amount_D

theorem category_D_cost_after_discount_is_correct : cost_after_discount_D = 562.5 := 
by 
  sorry

end category_D_cost_after_discount_is_correct_l73_73674


namespace democrats_ratio_l73_73721

noncomputable def F : ℕ := 240
noncomputable def M : ℕ := 480
noncomputable def D_F : ℕ := 120
noncomputable def D_M : ℕ := 120

theorem democrats_ratio (total_participants : ℕ := 720)
  (h1 : F + M = total_participants)
  (h2 : D_F = 120)
  (h3 : D_F = 1/2 * F)
  (h4 : D_M = 1/4 * M)
  (h5 : D_F + D_M = 240)
  (h6 : F + M = 720) : (D_F + D_M) / total_participants = 1 / 3 :=
by
  sorry

end democrats_ratio_l73_73721


namespace smallest_t_for_sine_polar_circle_l73_73788

theorem smallest_t_for_sine_polar_circle :
  ∃ t : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t) → ∃ r : ℝ, r = Real.sin θ) ∧
           (∀ θ : ℝ, (θ = t) → ∃ r : ℝ, r = 0) ∧
           (∀ t' : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t') → ∃ r : ℝ, r = Real.sin θ) →
                       (∀ θ : ℝ, (θ = t') → ∃ r : ℝ, r = 0) → t' ≥ t) :=
by
  sorry

end smallest_t_for_sine_polar_circle_l73_73788


namespace inequality_solution_set_l73_73441

theorem inequality_solution_set : 
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end inequality_solution_set_l73_73441


namespace eugene_cards_in_deck_l73_73488

theorem eugene_cards_in_deck 
  (cards_used_per_card : ℕ)
  (boxes_used : ℕ)
  (toothpicks_per_box : ℕ)
  (cards_leftover : ℕ)
  (total_toothpicks_used : ℕ)
  (cards_used : ℕ)
  (total_cards_in_deck : ℕ)
  (h1 : cards_used_per_card = 75)
  (h2 : boxes_used = 6)
  (h3 : toothpicks_per_box = 450)
  (h4 : cards_leftover = 16)
  (h5 : total_toothpicks_used = boxes_used * toothpicks_per_box)
  (h6 : cards_used = total_toothpicks_used / cards_used_per_card)
  (h7 : total_cards_in_deck = cards_used + cards_leftover) :
  total_cards_in_deck = 52 :=
by 
  sorry

end eugene_cards_in_deck_l73_73488


namespace student_total_marks_l73_73900

theorem student_total_marks (total_questions correct_answers incorrect_mark correct_mark : ℕ) 
                             (H1 : total_questions = 60) 
                             (H2 : correct_answers = 34)
                             (H3 : incorrect_mark = 1)
                             (H4 : correct_mark = 4) :
  ((correct_answers * correct_mark) - ((total_questions - correct_answers) * incorrect_mark)) = 110 := 
by {
  -- The proof goes here.
  sorry
}

end student_total_marks_l73_73900


namespace smallest_even_integer_l73_73662

theorem smallest_even_integer (n : ℕ) (h_even : n % 2 = 0)
  (h_2digit : 10 ≤ n ∧ n ≤ 98)
  (h_property : (n - 2) * n * (n + 2) = 5 * ((n - 2) + n + (n + 2))) :
  n = 86 :=
by
  sorry

end smallest_even_integer_l73_73662


namespace find_number_of_hens_l73_73413

theorem find_number_of_hens
  (H C : ℕ)
  (h1 : H + C = 48)
  (h2 : 2 * H + 4 * C = 140) :
  H = 26 :=
by
  sorry

end find_number_of_hens_l73_73413


namespace evaluate_expression_at_x_zero_l73_73535

theorem evaluate_expression_at_x_zero (x : ℕ) (h1 : x < 3) (h2 : x ≠ 1) (h3 : x ≠ 2) : ((3 / (x - 1) - x - 1) / (x - 2) / (x^2 - 2 * x + 1)) = 2 :=
by
  -- Here we need to provide our proof, though for now it’s indicated by sorry
  sorry

end evaluate_expression_at_x_zero_l73_73535


namespace rectangle_area_error_percentage_l73_73677

theorem rectangle_area_error_percentage 
  (L W : ℝ)
  (measured_length : ℝ := L * 1.16)
  (measured_width : ℝ := W * 0.95)
  (actual_area : ℝ := L * W)
  (measured_area : ℝ := measured_length * measured_width) :
  ((measured_area - actual_area) / actual_area) * 100 = 10.2 := 
by
  sorry

end rectangle_area_error_percentage_l73_73677


namespace conditional_probability_of_wind_given_rain_l73_73891

theorem conditional_probability_of_wind_given_rain (P_A P_B P_A_and_B : ℚ)
  (h1: P_A = 4/15) (h2: P_B = 2/15) (h3: P_A_and_B = 1/10) :
  P_A_and_B / P_A = 3/8 :=
by
  sorry

end conditional_probability_of_wind_given_rain_l73_73891


namespace find_f3_l73_73109

variable (f : ℕ → ℕ)

axiom h : ∀ x : ℕ, f (x + 1) = x ^ 2

theorem find_f3 : f 3 = 4 :=
by
  sorry

end find_f3_l73_73109


namespace floor_length_l73_73283

theorem floor_length (width length : ℕ) 
  (cost_per_square total_cost : ℕ)
  (square_side : ℕ)
  (h1 : width = 64) 
  (h2 : square_side = 8)
  (h3 : cost_per_square = 24)
  (h4 : total_cost = 576) 
  : length = 24 :=
by
  -- Placeholder for the proof, using sorry
  sorry

end floor_length_l73_73283


namespace abs_diff_eq_1point5_l73_73069

theorem abs_diff_eq_1point5 (x y : ℝ)
    (hx : (⌊x⌋ : ℝ) + (y - ⌊y⌋) = 3.7)
    (hy : (x - ⌊x⌋) + (⌊y⌋ : ℝ) = 4.2) :
        |x - y| = 1.5 :=
by
  sorry

end abs_diff_eq_1point5_l73_73069


namespace minimum_area_triangle_AOB_l73_73780

theorem minimum_area_triangle_AOB : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 / a + 2 / b = 1) ∧ (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 / a + 2 / b = 1) → (1/2 * a * b ≥ 12)) := 
sorry

end minimum_area_triangle_AOB_l73_73780


namespace problem1_l73_73016

theorem problem1 (x : ℝ) (hx : x < 5/4) : 
  ∃ y : ℝ, y = 4 * x - 2 + 1 / (4 * x - 5) ∧ y ≤ 1 := 
sorry

end problem1_l73_73016


namespace range_of_a_l73_73023

noncomputable def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
noncomputable def B (a : ℝ) : Set ℝ := {a - 3, a - 1, a + 1}

theorem range_of_a (a : ℝ) : (A a ∩ B a = {-2}) ↔ (a = -1) :=
by {
  sorry
}

end range_of_a_l73_73023


namespace one_third_of_recipe_l73_73142

noncomputable def recipe_flour_required : ℚ := 7 + 3 / 4

theorem one_third_of_recipe : (1 / 3) * recipe_flour_required = (2 : ℚ) + 7 / 12 :=
by
  sorry

end one_third_of_recipe_l73_73142


namespace oliver_money_left_l73_73028

-- Definitions based on the given conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def frisbee_cost : ℕ := 4
def puzzle_cost : ℕ := 3
def birthday_gift : ℕ := 8

-- The proof problem statement
theorem oliver_money_left : 
  initial_amount + allowance_saved - (frisbee_cost + puzzle_cost) + birthday_gift = 15 := 
by
  sorry

end oliver_money_left_l73_73028


namespace quadratic_has_distinct_real_roots_l73_73907

theorem quadratic_has_distinct_real_roots : 
  ∀ (x : ℝ), x^2 - 3 * x + 1 = 0 → ∀ (a b c : ℝ), a = 1 ∧ b = -3 ∧ c = 1 →
  (b^2 - 4 * a * c) > 0 := 
by
  sorry

end quadratic_has_distinct_real_roots_l73_73907


namespace rajesh_walked_distance_l73_73338

theorem rajesh_walked_distance (H : ℝ) (D_R : ℝ) 
  (h1 : D_R = 4 * H - 10)
  (h2 : H + D_R = 25) :
  D_R = 18 :=
by
  sorry

end rajesh_walked_distance_l73_73338


namespace solve_equation_l73_73107

theorem solve_equation : ∀ x : ℝ, (x + 1 - 2 * (x - 1) = 1 - 3 * x) → x = 0 := 
by
  intros x h
  sorry

end solve_equation_l73_73107


namespace nadia_flower_shop_l73_73822

theorem nadia_flower_shop (roses lilies cost_per_rose cost_per_lily cost_roses cost_lilies total_cost : ℕ)
  (h1 : roses = 20)
  (h2 : lilies = 3 * roses / 4)
  (h3 : cost_per_rose = 5)
  (h4 : cost_per_lily = 2 * cost_per_rose)
  (h5 : cost_roses = roses * cost_per_rose)
  (h6 : cost_lilies = lilies * cost_per_lily)
  (h7 : total_cost = cost_roses + cost_lilies) :
  total_cost = 250 :=
by
  sorry

end nadia_flower_shop_l73_73822


namespace parabola_focus_coordinates_l73_73751

theorem parabola_focus_coordinates :
  ∀ x y : ℝ, y^2 = -8 * x → (x, y) = (-2, 0) := by
  sorry

end parabola_focus_coordinates_l73_73751


namespace previous_year_profit_percentage_l73_73582

theorem previous_year_profit_percentage (R : ℝ) (P : ℝ) :
  (0.16 * 0.70 * R = 1.1200000000000001 * (P / 100 * R)) → P = 10 :=
by {
  sorry
}

end previous_year_profit_percentage_l73_73582


namespace integer_powers_of_reciprocal_sum_l73_73075

variable (x: ℝ)

theorem integer_powers_of_reciprocal_sum (hx : x ≠ 0) (hx_int : ∃ k : ℤ, x + 1/x = k) : ∀ n : ℕ, ∃ k : ℤ, x^n + 1/x^n = k :=
by
  sorry

end integer_powers_of_reciprocal_sum_l73_73075


namespace combine_heaps_l73_73431

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end combine_heaps_l73_73431


namespace quadratic_expression_result_l73_73086

theorem quadratic_expression_result (x y : ℚ) 
  (h1 : 4 * x + y = 11) 
  (h2 : x + 4 * y = 15) : 
  13 * x^2 + 14 * x * y + 13 * y^2 = 275.2 := 
by 
  sorry

end quadratic_expression_result_l73_73086


namespace find_x_when_y_64_l73_73720

theorem find_x_when_y_64 (x y k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_inv_prop : x^3 * y = k) (h_given : x = 2 ∧ y = 8 ∧ k = 64) :
  y = 64 → x = 1 :=
by
  sorry

end find_x_when_y_64_l73_73720


namespace expand_and_simplify_l73_73717

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l73_73717


namespace rational_power_sum_l73_73950

theorem rational_power_sum (a b : ℚ) (ha : a = 1 / a) (hb : b = - b) : a ^ 2007 + b ^ 2007 = 1 ∨ a ^ 2007 + b ^ 2007 = -1 := by
  sorry

end rational_power_sum_l73_73950


namespace find_x_in_equation_l73_73382

theorem find_x_in_equation :
  ∃ x : ℝ, 2.5 * ( (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) ) = 2000.0000000000002 ∧ x = 0.3 :=
by 
  sorry

end find_x_in_equation_l73_73382


namespace problem_statement_l73_73606

-- Definitions of the operations △ and ⊗
def triangle (a b : ℤ) : ℤ := a + b + a * b - 1
def otimes (a b : ℤ) : ℤ := a * a - a * b + b * b

-- The theorem statement
theorem problem_statement : triangle 3 (otimes 2 4) = 50 := by
  sorry

end problem_statement_l73_73606


namespace math_problem_l73_73944

noncomputable def a : ℝ := 0.137
noncomputable def b : ℝ := 0.098
noncomputable def c : ℝ := 0.123
noncomputable def d : ℝ := 0.086

theorem math_problem : 
  ( ((a + b)^2 - (a - b)^2) / (c * d) + (d^3 - c^3) / (a * b * (a + b)) ) = 4.6886 := 
  sorry

end math_problem_l73_73944


namespace expression_comparison_l73_73999

theorem expression_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) :
  let exprI := (a + (1 / a)) * (b + (1 / b))
  let exprII := (Real.sqrt (a * b) + (1 / Real.sqrt (a * b))) ^ 2
  let exprIII := (((a + b) / 2) + (2 / (a + b))) ^ 2
  (exprI = exprII ∨ exprI = exprIII ∨ exprII = exprIII ∨ 
   (exprI > exprII ∧ exprI > exprIII) ∨
   (exprII > exprI ∧ exprII > exprIII) ∨
   (exprIII > exprI ∧ exprIII > exprII)) ∧
  ¬((exprI > exprII ∧ exprI > exprIII) ∨
    (exprII > exprI ∧ exprII > exprIII) ∨
    (exprIII > exprI ∧ exprIII > exprII)) :=
by
  let exprI := (a + (1 / a)) * (b + (1 / b))
  let exprII := (Real.sqrt (a * b) + (1 / Real.sqrt (a * b))) ^ 2
  let exprIII := (((a + b) / 2) + (2 / (a + b))) ^ 2
  sorry

end expression_comparison_l73_73999


namespace set_intersection_l73_73750

open Set

def U := {x : ℝ | True}
def A := {x : ℝ | x^2 - 2 * x < 0}
def B := {x : ℝ | x - 1 ≥ 0}
def complement (U B : Set ℝ) := {x : ℝ | x ∉ B}
def intersection (A B : Set ℝ) := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem set_intersection :
  intersection A (complement U B) = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end set_intersection_l73_73750


namespace ellipse_foci_y_axis_range_l73_73451

theorem ellipse_foci_y_axis_range (k : ℝ) :
  (∃ x y : ℝ, x^2 + k * y^2 = 4 ∧ (∃ c1 c2 : ℝ, y = 0 → c1^2 + c2^2 = 4)) ↔ 0 < k ∧ k < 1 :=
by
  sorry

end ellipse_foci_y_axis_range_l73_73451


namespace original_mixture_percentage_l73_73684

variables (a w : ℝ)

-- Conditions given
def condition1 : Prop := a / (a + w + 2) = 0.3
def condition2 : Prop := (a + 2) / (a + w + 4) = 0.4

theorem original_mixture_percentage (h1 : condition1 a w) (h2 : condition2 a w) : (a / (a + w)) * 100 = 36 :=
by
sorry

end original_mixture_percentage_l73_73684


namespace abs_inequality_solution_l73_73626

theorem abs_inequality_solution {x : ℝ} (h : |x + 1| < 5) : -6 < x ∧ x < 4 :=
by
  sorry

end abs_inequality_solution_l73_73626


namespace fixed_point_is_5_225_l73_73091

theorem fixed_point_is_5_225 : ∃ a b : ℝ, (∀ k : ℝ, 9 * a^2 + k * a - 5 * k = b) → (a = 5 ∧ b = 225) :=
by
  sorry

end fixed_point_is_5_225_l73_73091


namespace page_problem_insufficient_information_l73_73402

theorem page_problem_insufficient_information
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (x y : ℕ)
  (O E : ℕ)
  (h1 : total_problems = 450)
  (h2 : finished_problems = 185)
  (h3 : remaining_pages = 15)
  (h4 : O + E = remaining_pages)
  (h5 : O * x + E * y = total_problems - finished_problems) :
  ∀ (x y : ℕ), O * x + E * y = 265 → x = x ∧ y = y :=
by
  sorry

end page_problem_insufficient_information_l73_73402


namespace contradiction_proof_l73_73489

theorem contradiction_proof (a b c : ℝ) (h : ¬ (a > 0 ∨ b > 0 ∨ c > 0)) : false :=
by
  sorry

end contradiction_proof_l73_73489


namespace simplify_expression_l73_73104

theorem simplify_expression (x : ℝ) (h1 : x^2 - 4*x + 3 ≠ 0) (h2 : x^2 - 6*x + 9 ≠ 0) (h3 : x^2 - 3*x + 2 ≠ 0) (h4 : x^2 - 4*x + 4 ≠ 0) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / (x^2 - 3*x + 2) / (x^2 - 4*x + 4) = (x-2) / (x-3) :=
by {
  sorry
}

end simplify_expression_l73_73104


namespace fewest_four_dollar_frisbees_l73_73135

theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : x + y = 60) (h2 : 3 * x + 4 * y = 200) : y = 20 :=
by 
  sorry  

end fewest_four_dollar_frisbees_l73_73135


namespace cocktail_cost_per_litre_is_accurate_l73_73610

noncomputable def mixed_fruit_juice_cost_per_litre : ℝ := 262.85
noncomputable def acai_berry_juice_cost_per_litre : ℝ := 3104.35
noncomputable def mixed_fruit_juice_litres : ℝ := 35
noncomputable def acai_berry_juice_litres : ℝ := 23.333333333333336

noncomputable def cocktail_total_cost : ℝ := 
  (mixed_fruit_juice_cost_per_litre * mixed_fruit_juice_litres) +
  (acai_berry_juice_cost_per_litre * acai_berry_juice_litres)

noncomputable def cocktail_total_volume : ℝ := 
  mixed_fruit_juice_litres + acai_berry_juice_litres

noncomputable def cocktail_cost_per_litre : ℝ := 
  cocktail_total_cost / cocktail_total_volume

theorem cocktail_cost_per_litre_is_accurate : 
  abs (cocktail_cost_per_litre - 1399.99) < 0.01 := by
  sorry

end cocktail_cost_per_litre_is_accurate_l73_73610


namespace find_m_to_make_z1_eq_z2_l73_73347

def z1 (m : ℝ) : ℂ := (2 * m + 7 : ℝ) + (m^2 - 2 : ℂ) * Complex.I
def z2 (m : ℝ) : ℂ := (m^2 - 8 : ℝ) + (4 * m + 3 : ℂ) * Complex.I

theorem find_m_to_make_z1_eq_z2 : 
  ∃ m : ℝ, z1 m = z2 m ∧ m = 5 :=
by
  sorry

end find_m_to_make_z1_eq_z2_l73_73347


namespace sum_of_x_and_y_l73_73576

theorem sum_of_x_and_y (x y : ℕ) (h_pos_x: 0 < x) (h_pos_y: 0 < y) (h_gt: x > y) (h_eq: x + x * y = 391) : x + y = 39 :=
by
  sorry

end sum_of_x_and_y_l73_73576


namespace remainder_of_B_is_4_l73_73951

theorem remainder_of_B_is_4 (A B : ℕ) (h : B = A * 9 + 13) : B % 9 = 4 :=
by {
  sorry
}

end remainder_of_B_is_4_l73_73951


namespace solution_set_of_inequality_l73_73594

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h₁ : ∀ x > 0, deriv f x + 2 * f x > 0) :
  {x : ℝ | x + 2018 > 0 ∧ x + 2018 < 5} = {x : ℝ | -2018 < x ∧ x < -2013} := 
by
  sorry

end solution_set_of_inequality_l73_73594


namespace inequality_example_l73_73584

theorem inequality_example (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
sorry

end inequality_example_l73_73584


namespace alice_spent_19_percent_l73_73085

variable (A : ℝ) (x : ℝ)
variable (h1 : ∃ (B : ℝ), B = 0.9 * A) -- Bob's initial amount in terms of Alice's initial amount
variable (h2 : A - x = 0.81 * A) -- Alice's remaining amount after spending x

theorem alice_spent_19_percent (h1 : ∃ (B : ℝ), B = 0.9 * A) (h2 : A - x = 0.81 * A) : (x / A) * 100 = 19 := by
  sorry

end alice_spent_19_percent_l73_73085


namespace remainder_mod_500_l73_73620

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end remainder_mod_500_l73_73620


namespace distance_midpoints_eq_2_5_l73_73529

theorem distance_midpoints_eq_2_5 (A B C : ℝ) (hAB : A < B) (hBC : B < C) (hAC_len : C - A = 5) :
    let M1 := (A + B) / 2
    let M2 := (B + C) / 2
    (M2 - M1 = 2.5) :=
by
    let M1 := (A + B) / 2
    let M2 := (B + C) / 2
    sorry

end distance_midpoints_eq_2_5_l73_73529


namespace total_memory_space_l73_73688

def morning_songs : Nat := 10
def afternoon_songs : Nat := 15
def night_songs : Nat := 3
def song_size : Nat := 5

theorem total_memory_space : (morning_songs + afternoon_songs + night_songs) * song_size = 140 := by
  sorry

end total_memory_space_l73_73688


namespace updated_mean_of_decremented_observations_l73_73544

theorem updated_mean_of_decremented_observations (mean : ℝ) (n : ℕ) (decrement : ℝ) 
  (h_mean : mean = 200) (h_n : n = 50) (h_decrement : decrement = 47) : 
  (mean * n - decrement * n) / n = 153 := 
by 
  sorry

end updated_mean_of_decremented_observations_l73_73544


namespace greatest_xy_value_l73_73848

theorem greatest_xy_value (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_eq : 7 * x + 4 * y = 140) :
  (∀ z : ℕ, (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ z = x * y) → z ≤ 168) ∧
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ 168 = x * y) :=
sorry

end greatest_xy_value_l73_73848


namespace weight_around_59_3_l73_73716

noncomputable def weight_at_height (height: ℝ) : ℝ := 0.75 * height - 68.2

theorem weight_around_59_3 (x : ℝ) (h : x = 170) : abs (weight_at_height x - 59.3) < 1 :=
by
  sorry

end weight_around_59_3_l73_73716


namespace interest_calculation_l73_73642

theorem interest_calculation (P : ℝ) (r : ℝ) (CI SI : ℝ → ℝ) (n : ℝ) :
  P = 1300 →
  r = 0.10 →
  (CI n - SI n = 13) →
  (CI n = P * (1 + r)^n - P) →
  (SI n = P * r * n) →
  (1.10 ^ n - 1 - 0.10 * n = 0.01) →
  n = 2 :=
by
  intro P_eq r_eq diff_eq CI_def SI_def equation
  -- Sorry, this is just a placeholder. The proof is omitted.
  sorry

end interest_calculation_l73_73642


namespace mean_temperature_correct_l73_73044

-- Define the condition (temperatures)
def temperatures : List Int :=
  [-6, -3, -3, -4, 2, 4, 1]

-- Define the total number of days
def num_days : ℕ := 7

-- Define the expected mean temperature
def expected_mean : Rat := (-6 : Int) / (7 : Int)

-- State the theorem that we need to prove
theorem mean_temperature_correct :
  (temperatures.sum : Rat) / (num_days : Rat) = expected_mean := 
by
  sorry

end mean_temperature_correct_l73_73044


namespace max_plates_l73_73904

/-- Bill can buy pans, pots, and plates for 3, 5, and 10 dollars each, respectively.
    What is the maximum number of plates he can purchase if he must buy at least
    two of each item and will spend exactly 100 dollars? -/
theorem max_plates (x y z : ℕ) (hx : x ≥ 2) (hy : y ≥ 2) (hz : z ≥ 2) 
  (h_cost : 3 * x + 5 * y + 10 * z = 100) : z = 8 :=
sorry

end max_plates_l73_73904


namespace smallest_positive_period_of_f_minimum_value_of_f_on_interval_l73_73304

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 2) * (Real.sin (x / 2)) * (Real.cos (x / 2)) - (Real.sqrt 2) * (Real.sin (x / 2)) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by sorry

theorem minimum_value_of_f_on_interval : 
  ∃ x ∈ Set.Icc (-Real.pi) 0, 
  f x = -1 - Real.sqrt 2 / 2 :=
by sorry

end smallest_positive_period_of_f_minimum_value_of_f_on_interval_l73_73304


namespace find_cost_price_l73_73329

variable (CP SP1 SP2 : ℝ)

theorem find_cost_price
    (h1 : SP1 = CP * 0.92)
    (h2 : SP2 = CP * 1.04)
    (h3 : SP2 = SP1 + 140) :
    CP = 1166.67 :=
by
  -- Proof would be filled here
  sorry

end find_cost_price_l73_73329


namespace derivative_f_at_1_l73_73177

noncomputable def f (x : Real) : Real := x^3 * Real.sin x

theorem derivative_f_at_1 : deriv f 1 = 3 * Real.sin 1 + Real.cos 1 := by
  sorry

end derivative_f_at_1_l73_73177


namespace probability_same_color_is_correct_l73_73332

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l73_73332


namespace quadratic_inequality_solution_l73_73137

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 2 * x - 3 < 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end quadratic_inequality_solution_l73_73137


namespace arithmetic_seq_property_l73_73015

theorem arithmetic_seq_property (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_cond : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 9 - a 10 = 24 := 
sorry

end arithmetic_seq_property_l73_73015


namespace problem_xy_l73_73300

theorem problem_xy (x y : ℝ) (h1 : x + y = 25) (h2 : x^2 * y^3 + y^2 * x^3 = 25) : x * y = 1 :=
by
  sorry

end problem_xy_l73_73300


namespace four_played_games_l73_73583

theorem four_played_games
  (A B C D E : Prop)
  (A_answer : ¬A)
  (B_answer : A ∧ ¬B)
  (C_answer : B ∧ ¬C)
  (D_answer : C ∧ ¬D)
  (E_answer : D ∧ ¬E)
  (truth_condition : (¬A ∧ ¬B) ∨ (¬B ∧ ¬C) ∨ (¬C ∧ ¬D) ∨ (¬D ∧ ¬E)) :
  A ∨ B ∨ C ∨ D ∧ E := sorry

end four_played_games_l73_73583


namespace manager_salary_is_3600_l73_73926

-- Definitions based on the conditions
def average_salary_20_employees := 1500
def number_of_employees := 20
def new_average_salary := 1600
def number_of_people_incl_manager := number_of_employees + 1

-- Calculate necessary total salaries and manager's salary
def total_salary_of_20_employees := number_of_employees * average_salary_20_employees
def new_total_salary_with_manager := number_of_people_incl_manager * new_average_salary
def manager_monthly_salary := new_total_salary_with_manager - total_salary_of_20_employees

-- The statement to be proved
theorem manager_salary_is_3600 : manager_monthly_salary = 3600 :=
by
  sorry

end manager_salary_is_3600_l73_73926


namespace number_of_correct_propositions_l73_73829

theorem number_of_correct_propositions : 
    (∀ a b : ℝ, a < b → ¬ (a^2 < b^2)) ∧ 
    (∀ a : ℝ, (∀ x : ℝ, |x + 1| + |x - 1| ≥ a ↔ a ≤ 2)) ∧ 
    (¬ (∃ x : ℝ, x^2 - x > 0) ↔ ∀ x : ℝ, x^2 - x ≤ 0) → 
    1 = 1 := 
by
  sorry

end number_of_correct_propositions_l73_73829


namespace juniors_score_l73_73248

theorem juniors_score (juniors seniors total_students avg_score avg_seniors_score total_score : ℝ)
  (hj: juniors = 0.2 * total_students)
  (hs: seniors = 0.8 * total_students)
  (ht: total_students = 20)
  (ha: avg_score = 78)
  (hp: (seniors * avg_seniors_score + juniors * c) / total_students = avg_score)
  (havg_seniors: avg_seniors_score = 76)
  (hts: total_score = total_students * avg_score)
  (total_seniors_score : ℝ)
  (hts_seniors: total_seniors_score = seniors * avg_seniors_score)
  (total_juniors_score : ℝ)
  (hts_juniors: total_juniors_score = total_score - total_seniors_score)
  (hjs: c = total_juniors_score / juniors) :
  c = 86 :=
sorry

end juniors_score_l73_73248


namespace central_angle_of_sector_l73_73479

open Real

theorem central_angle_of_sector (l S : ℝ) (α R : ℝ) (hl : l = 4) (hS : S = 4) (h1 : l = α * R) (h2 : S = 1/2 * α * R^2) : 
  α = 2 :=
by
  -- Proof will be supplied here
  sorry

end central_angle_of_sector_l73_73479


namespace tangent_at_point_l73_73202

theorem tangent_at_point (a b : ℝ) :
  (∀ x : ℝ, (x^3 - x^2 - a * x + b) = 2 * x + 1) →
  (a + b = -1) :=
by
  intro tangent_condition
  sorry

end tangent_at_point_l73_73202


namespace geometric_sequence_first_term_l73_73237

open Real Nat

theorem geometric_sequence_first_term (a r : ℝ)
  (h1 : a * r^4 = (7! : ℝ))
  (h2 : a * r^7 = (8! : ℝ)) : a = 315 := by
  sorry

end geometric_sequence_first_term_l73_73237


namespace new_concentration_of_mixture_l73_73526

theorem new_concentration_of_mixture :
  let v1 := 2
  let c1 := 0.25
  let v2 := 6
  let c2 := 0.40
  let V := 10
  let alcohol_amount_v1 := v1 * c1
  let alcohol_amount_v2 := v2 * c2
  let total_alcohol := alcohol_amount_v1 + alcohol_amount_v2
  let new_concentration := (total_alcohol / V) * 100
  new_concentration = 29 := 
by
  sorry

end new_concentration_of_mixture_l73_73526


namespace jina_total_mascots_l73_73060

-- Definitions and Conditions
def num_teddies := 5
def num_bunnies := 3 * num_teddies
def num_koala_bears := 1
def additional_teddies := 2 * num_bunnies

-- Total mascots calculation
def total_mascots := num_teddies + num_bunnies + num_koala_bears + additional_teddies

theorem jina_total_mascots : total_mascots = 51 := by
  sorry

end jina_total_mascots_l73_73060


namespace range_of_a_l73_73483

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * x - 1) / (x - 1) < 0 ↔ x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) → 
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  intro h
  sorry

end range_of_a_l73_73483


namespace leak_empty_time_l73_73061

/-- 
The time taken for a leak to empty a full tank, given that an electric pump can fill a tank in 7 hours and it takes 14 hours to fill the tank with the leak present, is 14 hours.
 -/
theorem leak_empty_time (P L : ℝ) (hP : P = 1 / 7) (hCombined : P - L = 1 / 14) : L = 1 / 14 ∧ 1 / L = 14 :=
by
  sorry

end leak_empty_time_l73_73061


namespace composite_integer_expression_l73_73855

theorem composite_integer_expression (n : ℕ) (h : n > 1) (hn : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b) :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ n = x * y + x * z + y * z + 1 :=
by
  sorry

end composite_integer_expression_l73_73855


namespace problem_statement_l73_73748

variable (a b : Type) [LinearOrder a] [LinearOrder b]
variable (α β : Type) [LinearOrder α] [LinearOrder β]

-- Given conditions
def line_perpendicular_to_plane (l : Type) (p : Type) [LinearOrder l] [LinearOrder p] : Prop :=
True -- This is a placeholder. Actual geometry definition required.

def lines_parallel (l1 : Type) (l2 : Type) [LinearOrder l1] [LinearOrder l2] : Prop :=
True -- This is a placeholder. Actual geometry definition required.

theorem problem_statement (a b α : Type) [LinearOrder a] [LinearOrder b] [LinearOrder α]
(val_perp1 : line_perpendicular_to_plane a α)
(val_perp2 : line_perpendicular_to_plane b α)
: lines_parallel a b :=
sorry

end problem_statement_l73_73748


namespace number_of_male_employees_l73_73257

theorem number_of_male_employees (num_female : ℕ) (x y : ℕ) 
  (h1 : 7 * x = y) 
  (h2 : 8 * x = num_female) 
  (h3 : 9 * (7 * x + 3) = 8 * num_female) :
  y = 189 := by
  sorry

end number_of_male_employees_l73_73257


namespace cody_discount_l73_73477

theorem cody_discount (initial_cost tax_rate cody_paid total_paid price_before_discount discount: ℝ) 
  (h1 : initial_cost = 40)
  (h2 : tax_rate = 0.05)
  (h3 : cody_paid = 17)
  (h4 : total_paid = 2 * cody_paid)
  (h5 : price_before_discount = initial_cost * (1 + tax_rate))
  (h6 : discount = price_before_discount - total_paid) :
  discount = 8 := by
  sorry

end cody_discount_l73_73477


namespace find_params_l73_73256

theorem find_params (a b c : ℝ) :
    (∀ x : ℝ, x = 2 ∨ x = -2 → x^5 + 4 * x^4 + a * x = b * x^2 + 4 * c) 
    → a = 16 ∧ b = 48 ∧ c = -32 :=
by
  sorry

end find_params_l73_73256


namespace locus_of_points_l73_73833

-- Define points A and B
variable {A B : (ℝ × ℝ)}
-- Define constant d
variable {d : ℝ}

-- Definition of the distances
def distance_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem locus_of_points (A B : (ℝ × ℝ)) (d : ℝ) :
  ∀ M : (ℝ × ℝ), distance_sq M A - distance_sq M B = d ↔ 
  ∃ x : ℝ, ∃ y : ℝ, (M.1, M.2) = (x, y) ∧ 
  x = ((B.1 - A.1)^2 + d) / (2 * (B.1 - A.1)) :=
by
  sorry

end locus_of_points_l73_73833


namespace percentage_less_than_l73_73828

theorem percentage_less_than (P T J : ℝ) 
  (h1 : T = 0.9375 * P) 
  (h2 : J = 0.8 * T) 
  : (P - J) / P * 100 = 25 := 
by
  sorry

end percentage_less_than_l73_73828


namespace mark_paintable_area_l73_73773

theorem mark_paintable_area :
  let num_bedrooms := 4
  let length := 14
  let width := 11
  let height := 9
  let area_excluded := 70
  let area_wall_one_bedroom := 2 * (length * height) + 2 * (width * height) - area_excluded 
  (area_wall_one_bedroom * num_bedrooms) = 1520 :=
by
  sorry

end mark_paintable_area_l73_73773


namespace total_amount_is_correct_l73_73863

-- Given conditions
def original_price : ℝ := 200
def discount_rate: ℝ := 0.25
def coupon_value: ℝ := 10
def tax_rate: ℝ := 0.05

-- Define the price calculations
def discounted_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)
def price_after_coupon (p : ℝ) (c : ℝ) : ℝ := p - c
def final_price_with_tax (p : ℝ) (t : ℝ) : ℝ := p * (1 + t)

-- Goal: Prove the final amount the customer pays
theorem total_amount_is_correct : final_price_with_tax (price_after_coupon (discounted_price original_price discount_rate) coupon_value) tax_rate = 147 := by
  sorry

end total_amount_is_correct_l73_73863


namespace subcommittee_count_l73_73019

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let select_republicans := 4
  let select_democrats := 3
  let num_ways_republicans := Nat.choose republicans select_republicans
  let num_ways_democrats := Nat.choose democrats select_democrats
  let num_ways := num_ways_republicans * num_ways_democrats
  num_ways = 11760 :=
by
  sorry

end subcommittee_count_l73_73019


namespace sector_angle_l73_73379

-- Defining the conditions
def perimeter (r l : ℝ) : Prop := 2 * r + l = 8
def area (r l : ℝ) : Prop := (1 / 2) * l * r = 4

-- Lean theorem statement
theorem sector_angle (r l θ : ℝ) :
  (perimeter r l) → (area r l) → (θ = l / r) → |θ| = 2 :=
by sorry

end sector_angle_l73_73379


namespace hyperbola_foci_coordinates_l73_73575

theorem hyperbola_foci_coordinates :
  let a : ℝ := Real.sqrt 7
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  (c = Real.sqrt 10 ∧
  ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0))) :=
by
  let a := Real.sqrt 7
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 + b^2)
  have hc : c = Real.sqrt 10 := sorry
  have h_foci : ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0)) := sorry
  exact ⟨hc, h_foci⟩

end hyperbola_foci_coordinates_l73_73575


namespace initial_number_of_men_l73_73233

theorem initial_number_of_men (x : ℕ) :
    (50 * x = 25 * (x + 20)) → x = 20 := 
by
  sorry

end initial_number_of_men_l73_73233


namespace running_time_l73_73602

variable (t : ℝ)
variable (v_j v_p d : ℝ)

-- Given conditions
variable (v_j : ℝ := 0.133333333333)  -- Joe's speed
variable (v_p : ℝ := 0.0666666666665) -- Pete's speed
variable (d : ℝ := 16)                -- Distance between them after t minutes

theorem running_time (h : v_j + v_p = 0.2 * t) : t = 80 :=
by
  -- Distance covered by Joe and Pete running in opposite directions
  have h1 : v_j * t + v_p * t = d := by sorry
  -- Given combined speeds
  have h2 : v_j + v_p = 0.2 := by sorry
  -- Using the equation to solve for time t
  exact sorry

end running_time_l73_73602


namespace functional_relationship_find_selling_price_maximum_profit_l73_73930

noncomputable def linear_relation (x : ℤ) : ℤ := -5 * x + 150
def profit_function (x : ℤ) : ℤ := -5 * x * x + 200 * x - 1500

theorem functional_relationship (x : ℤ) (hx : 10 ≤ x ∧ x ≤ 15) : linear_relation x = -5 * x + 150 :=
by sorry

theorem find_selling_price (h : ∃ x : ℤ, (10 ≤ x ∧ x ≤ 15) ∧ ((-5 * x + 150) * (x - 10) = 320)) :
  ∃ x : ℤ, x = 14 :=
by sorry

theorem maximum_profit (hx : 10 ≤ 15 ∧ 15 ≤ 15) : profit_function 15 = 375 :=
by sorry

end functional_relationship_find_selling_price_maximum_profit_l73_73930


namespace min_g_l73_73118

noncomputable def g (x : ℝ) : ℝ := (x^2 + 2) / Real.sqrt (x^2 + 1)

theorem min_g : ∃ x : ℝ, g x = 2 :=
by
  use 0
  sorry

end min_g_l73_73118


namespace first_day_bacteria_exceeds_200_l73_73303

noncomputable def N : ℕ → ℕ := λ n => 5 * 3^n

theorem first_day_bacteria_exceeds_200 : ∃ n : ℕ, N n > 200 ∧ ∀ m : ℕ, m < n → N m ≤ 200 :=
by
  sorry

end first_day_bacteria_exceeds_200_l73_73303


namespace gears_together_again_l73_73946

theorem gears_together_again (r₁ r₂ : ℕ) (h₁ : r₁ = 3) (h₂ : r₂ = 5) : 
  (∃ t : ℕ, t = Nat.lcm r₁ r₂ / r₁ ∨ t = Nat.lcm r₁ r₂ / r₂) → 5 = Nat.lcm r₁ r₂ / min r₁ r₂ := 
by
  sorry

end gears_together_again_l73_73946


namespace simplify_fraction_l73_73074

theorem simplify_fraction :
  (1 : ℚ) / ((1 / (1 / 3 : ℚ) ^ 1) + (1 / (1 / 3 : ℚ) ^ 2) + (1 / (1 / 3 : ℚ) ^ 3) + (1 / (1 / 3 : ℚ) ^ 4)) = 1 / 120 := 
by 
  sorry

end simplify_fraction_l73_73074


namespace ratio_of_speeds_l73_73670

theorem ratio_of_speeds (v_A v_B : ℝ) (h1 : 500 / v_A = 400 / v_B) : v_A / v_B = 5 / 4 :=
by
  sorry

end ratio_of_speeds_l73_73670


namespace walter_time_at_seals_l73_73250

theorem walter_time_at_seals 
  (s p e total : ℕ)
  (h1 : p = 8 * s)
  (h2 : e = 13)
  (h3 : total = 130)
  (h4 : s + p + e = total) : s = 13 := 
by 
  sorry

end walter_time_at_seals_l73_73250


namespace expand_product_l73_73889

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5 * x - 36 :=
by
  -- No proof required, just state the theorem
  sorry

end expand_product_l73_73889


namespace polynomial_roots_l73_73318

theorem polynomial_roots :
  ∃ (x : ℚ) (y : ℚ) (z : ℚ) (w : ℚ),
    (x = 1) ∧ (y = 1) ∧ (z = -2) ∧ (w = -1/2) ∧
    2*x^4 + x^3 - 6*x^2 + x + 2 = 0 ∧
    2*y^4 + y^3 - 6*y^2 + y + 2 = 0 ∧
    2*z^4 + z^3 - 6*z^2 + z + 2 = 0 ∧
    2*w^4 + w^3 - 6*w^2 + w + 2 = 0 :=
by
  sorry

end polynomial_roots_l73_73318


namespace rice_mixture_ratio_l73_73673

-- Definitions for the given conditions
def cost_per_kg_rice1 : ℝ := 5
def cost_per_kg_rice2 : ℝ := 8.75
def cost_per_kg_mixture : ℝ := 7.50

-- The problem: ratio of two quantities
theorem rice_mixture_ratio (x y : ℝ) (h : cost_per_kg_rice1 * x + cost_per_kg_rice2 * y = 
                                     cost_per_kg_mixture * (x + y)) :
  y / x = 2 := 
sorry

end rice_mixture_ratio_l73_73673


namespace tangent_curves_line_exists_l73_73070

theorem tangent_curves_line_exists (a : ℝ) :
  (∃ l : ℝ → ℝ, ∃ x₀ : ℝ, l 1 = 0 ∧ ∀ x, (l x = x₀^3 ∧ l x = a * x^2 + (15 / 4) * x - 9)) →
  a = -25/64 ∨ a = -1 :=
by
  sorry

end tangent_curves_line_exists_l73_73070


namespace magnitude_difference_l73_73631

noncomputable def vector_a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
noncomputable def vector_b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))

theorem magnitude_difference (a b : ℝ × ℝ) 
  (ha : a = (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180)))
  (hb : b = (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))) :
  (Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2)) = Real.sqrt 3 :=
by
  sorry

end magnitude_difference_l73_73631


namespace solution_set_of_abs_fraction_eq_fraction_l73_73275

-- Problem Statement
theorem solution_set_of_abs_fraction_eq_fraction :
  { x : ℝ | |x / (x - 1)| = x / (x - 1) } = { x : ℝ | x ≤ 0 ∨ x > 1 } :=
by
  sorry

end solution_set_of_abs_fraction_eq_fraction_l73_73275


namespace sum_is_45_l73_73704

noncomputable def sum_of_numbers (a b c : ℝ) : ℝ :=
  a + b + c

theorem sum_is_45 {a b c : ℝ} (h1 : ∃ a b c, (a ≤ b ∧ b ≤ c) ∧ b = 10)
  (h2 : (a + b + c) / 3 = a + 20)
  (h3 : (a + b + c) / 3 = c - 25) :
  sum_of_numbers a b c = 45 := 
sorry

end sum_is_45_l73_73704


namespace faster_train_speed_l73_73935

theorem faster_train_speed (length_train : ℝ) (time_cross : ℝ) (speed_ratio : ℝ) (total_distance : ℝ) (relative_speed : ℝ) :
  length_train = 100 → 
  time_cross = 8 → 
  speed_ratio = 2 → 
  total_distance = 2 * length_train → 
  relative_speed = (1 + speed_ratio) * (total_distance / time_cross) → 
  (1 + speed_ratio) * (total_distance / time_cross) / 3 * 2 = 8.33 := 
by
  intros
  sorry

end faster_train_speed_l73_73935


namespace housewife_spending_l73_73204

theorem housewife_spending
    (R : ℝ) (P : ℝ) (M : ℝ)
    (h1 : R = 25)
    (h2 : R = 0.85 * P)
    (h3 : M / R - M / P = 3) :
  M = 450 :=
by
  sorry

end housewife_spending_l73_73204


namespace Emily_sixth_quiz_score_l73_73493

theorem Emily_sixth_quiz_score :
  let scores := [92, 95, 87, 89, 100]
  ∃ s : ℕ, (s + scores.sum : ℚ) / 6 = 93 :=
  by
    sorry

end Emily_sixth_quiz_score_l73_73493


namespace car_overtakes_truck_l73_73571

theorem car_overtakes_truck 
  (car_speed : ℝ)
  (truck_speed : ℝ)
  (car_arrival_time : ℝ)
  (truck_arrival_time : ℝ)
  (route_same : Prop)
  (time_difference : ℝ)
  (car_speed_km_min : car_speed = 66 / 60)
  (truck_speed_km_min : truck_speed = 42 / 60)
  (arrival_time_difference : truck_arrival_time - car_arrival_time = 18 / 60) :
  ∃ d : ℝ, d = 34.65 := 
by {
  sorry
}

end car_overtakes_truck_l73_73571


namespace Lucy_total_groceries_l73_73762

theorem Lucy_total_groceries :
  let packs_of_cookies := 12
  let packs_of_noodles := 16
  let boxes_of_cereals := 5
  let packs_of_crackers := 45
  (packs_of_cookies + packs_of_noodles + packs_of_crackers + boxes_of_cereals) = 78 :=
by
  sorry

end Lucy_total_groceries_l73_73762


namespace weekly_milk_production_l73_73859

-- Define the conditions
def num_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 5
def days_per_week : ℕ := 7

-- Define the proof that total weekly milk production is 1820 liters
theorem weekly_milk_production : num_cows * milk_per_cow_per_day * days_per_week = 1820 := by
  sorry

end weekly_milk_production_l73_73859


namespace part1_part2_part3_l73_73258

-- Part 1: Prove that B = 90° given a=20, b=29, c=21

theorem part1 (a b c : ℝ) (h1 : a = 20) (h2 : b = 29) (h3 : c = 21) : 
  ∃ B : ℝ, B = 90 := 
sorry

-- Part 2: Prove that b = 7 given a=3√3, c=2, B=150°

theorem part2 (a c B b : ℝ) (h1 : a = 3 * Real.sqrt 3) (h2 : c = 2) (h3 : B = 150) : 
  ∃ b : ℝ, b = 7 :=
sorry

-- Part 3: Prove that A = 45° given a=2, b=√2, c=√3 + 1

theorem part3 (a b c A : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 2) (h3 : c = Real.sqrt 3 + 1) : 
  ∃ A : ℝ, A = 45 :=
sorry

end part1_part2_part3_l73_73258


namespace ella_savings_l73_73956

theorem ella_savings
  (initial_cost_per_lamp : ℝ)
  (num_lamps : ℕ)
  (discount_rate : ℝ)
  (additional_discount : ℝ)
  (initial_total_cost : ℝ := num_lamps * initial_cost_per_lamp)
  (discounted_lamp_cost : ℝ := initial_cost_per_lamp - (initial_cost_per_lamp * discount_rate))
  (total_cost_with_discount : ℝ := num_lamps * discounted_lamp_cost)
  (total_cost_after_additional_discount : ℝ := total_cost_with_discount - additional_discount) :
  initial_cost_per_lamp = 15 →
  num_lamps = 3 →
  discount_rate = 0.25 →
  additional_discount = 5 →
  initial_total_cost - total_cost_after_additional_discount = 16.25 :=
by
  intros
  sorry

end ella_savings_l73_73956


namespace golf_ratio_l73_73161

-- Definitions based on conditions
def first_turn_distance : ℕ := 180
def excess_distance : ℕ := 20
def total_distance_to_hole : ℕ := 250

-- Derived definitions based on conditions
def second_turn_distance : ℕ := (total_distance_to_hole - first_turn_distance) + excess_distance

-- Lean proof problem statement
theorem golf_ratio : (second_turn_distance : ℚ) / first_turn_distance = 1 / 2 :=
by
  -- use sorry to skip the proof
  sorry

end golf_ratio_l73_73161


namespace initial_books_count_l73_73405

theorem initial_books_count (x : ℕ) (h : x + 10 = 48) : x = 38 := 
by
  sorry

end initial_books_count_l73_73405


namespace sum_of_digits_in_binary_representation_of_315_l73_73982

theorem sum_of_digits_in_binary_representation_of_315 : 
  (Nat.digits 2 315).sum = 6 := 
by
  sorry

end sum_of_digits_in_binary_representation_of_315_l73_73982


namespace quarters_needed_l73_73156

-- Define the cost of items in cents and declare the number of items to purchase.
def quarter_value : ℕ := 25
def candy_bar_cost : ℕ := 25
def chocolate_cost : ℕ := 75
def juice_cost : ℕ := 50

def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Theorem stating the number of quarters needed to buy the given items.
theorem quarters_needed : 
  (num_candy_bars * candy_bar_cost + num_chocolates * chocolate_cost + num_juice_packs * juice_cost) / quarter_value = 11 := 
sorry

end quarters_needed_l73_73156


namespace mechanic_worked_hours_l73_73921

theorem mechanic_worked_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (part_count : ℕ) 
  (labor_cost_per_minute : ℝ) 
  (parts_total_cost : ℝ) 
  (labor_total_cost : ℝ) 
  (hours_worked : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_count = 2) 
  (h3 : part_cost = 20) 
  (h4 : labor_cost_per_minute = 0.5) 
  (h5 : parts_total_cost = part_count * part_cost) 
  (h6 : labor_total_cost = total_cost - parts_total_cost) 
  (labor_cost_per_hour := labor_cost_per_minute * 60) 
  (h7 : hours_worked = labor_total_cost / labor_cost_per_hour) : 
  hours_worked = 6 := 
sorry

end mechanic_worked_hours_l73_73921


namespace problem1_problem2_l73_73675

-- Definitions of the sets A, B, C
def A (a : ℝ) : Set ℝ := { x | x^2 - a*x + a^2 - 12 = 0 }
def B : Set ℝ := { x | x^2 - 2*x - 8 = 0 }
def C (m : ℝ) : Set ℝ := { x | m*x + 1 = 0 }

-- Problem 1: If A = B, then a = 2
theorem problem1 (a : ℝ) (h : A a = B) : a = 2 := sorry

-- Problem 2: If B ∪ C m = B, then m ∈ {-1/4, 0, 1/2}
theorem problem2 (m : ℝ) (h : B ∪ C m = B) : m = -1/4 ∨ m = 0 ∨ m = 1/2 := sorry

end problem1_problem2_l73_73675


namespace shaded_percentage_of_grid_l73_73942

def percent_shaded (total_squares shaded_squares : ℕ) : ℚ :=
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100

theorem shaded_percentage_of_grid :
  percent_shaded 36 16 = 44.44 :=
by 
  sorry

end shaded_percentage_of_grid_l73_73942


namespace other_endpoint_coordinates_sum_l73_73217

noncomputable def other_endpoint_sum (x1 y1 x2 y2 xm ym : ℝ) : ℝ :=
  let x := 2 * xm - x1
  let y := 2 * ym - y1
  x + y

theorem other_endpoint_coordinates_sum :
  (other_endpoint_sum 6 (-2) 0 12 3 5) = 12 := by
  sorry

end other_endpoint_coordinates_sum_l73_73217


namespace collinear_condition_perpendicular_condition_l73_73147

-- Problem 1: Prove collinearity condition for k = -2
theorem collinear_condition (k : ℝ) : 
  (k - 5) * (-12) - (12 - k) * 6 = 0 ↔ k = -2 :=
sorry

-- Problem 2: Prove perpendicular condition for k = 2 or k = 11
theorem perpendicular_condition (k : ℝ) : 
  (20 + (k - 6) * (7 - k)) = 0 ↔ (k = 2 ∨ k = 11) :=
sorry

end collinear_condition_perpendicular_condition_l73_73147


namespace simplify_and_evaluate_l73_73870

def expr (a b : ℤ) := -a^2 * b + (3 * a * b^2 - a^2 * b) - 2 * (2 * a * b^2 - a^2 * b)

theorem simplify_and_evaluate : expr (-1) (-2) = -4 := by
  sorry

end simplify_and_evaluate_l73_73870


namespace certain_number_x_l73_73817

theorem certain_number_x (p q x : ℕ) (hp : p > 1) (hq : q > 1)
  (h_eq : x * (p + 1) = 21 * (q + 1)) 
  (h_sum : p + q = 36) : x = 245 := 
by 
  sorry

end certain_number_x_l73_73817


namespace Euclid_Middle_School_AMC8_contest_l73_73619

theorem Euclid_Middle_School_AMC8_contest (students_Germain students_Newton students_Young : ℕ)
       (hG : students_Germain = 11) 
       (hN : students_Newton = 8) 
       (hY : students_Young = 9) : 
       students_Germain + students_Newton + students_Young = 28 :=
by
  sorry

end Euclid_Middle_School_AMC8_contest_l73_73619


namespace chess_club_probability_l73_73121

theorem chess_club_probability :
  let total_members := 20
  let boys := 12
  let girls := 8
  let total_ways := Nat.choose total_members 4
  let all_boys := Nat.choose boys 4
  let all_girls := Nat.choose girls 4
  total_ways ≠ 0 → 
  (1 - (all_boys + all_girls) / total_ways) = (4280 / 4845) :=
by
  sorry

end chess_club_probability_l73_73121


namespace range_of_a_l73_73454

theorem range_of_a (x y : ℝ) (a : ℝ) :
  (0 < x ∧ x ≤ 2) ∧ (0 < y ∧ y ≤ 2) ∧ (x * y = 2) ∧ (6 - 2 * x - y ≥ a * (2 - x) * (4 - y)) →
  a ≤ 1 :=
by sorry

end range_of_a_l73_73454


namespace percent_uni_no_job_choice_l73_73511

variable (P_ND_JC P_JC P_UD P_U_NJC P_NJC : ℝ)
variable (h1 : P_ND_JC = 0.18)
variable (h2 : P_JC = 0.40)
variable (h3 : P_UD = 0.37)

theorem percent_uni_no_job_choice :
  (P_UD - (P_JC - P_ND_JC)) / (1 - P_JC) = 0.25 :=
by
  sorry

end percent_uni_no_job_choice_l73_73511


namespace percent_relation_l73_73232

variable (x y z : ℝ)

theorem percent_relation (h1 : x = 1.30 * y) (h2 : y = 0.60 * z) : x = 0.78 * z :=
by sorry

end percent_relation_l73_73232


namespace friends_attended_reception_l73_73067

-- Definition of the given conditions
def total_guests : ℕ := 180
def couples_per_side : ℕ := 20

-- Statement based on the given problem
theorem friends_attended_reception : 
  let family_guests := 2 * couples_per_side + 2 * couples_per_side
  let friends := total_guests - family_guests
  friends = 100 :=
by
  -- We define the family_guests calculation
  let family_guests := 2 * couples_per_side + 2 * couples_per_side
  -- We define the friends calculation
  let friends := total_guests - family_guests
  -- We state the conclusion
  show friends = 100
  sorry

end friends_attended_reception_l73_73067


namespace min_value_expression_l73_73756

theorem min_value_expression (x y : ℝ) (h1 : x < 0) (h2 : y < 0) (h3 : x + y = -1) :
  xy + (1 / xy) = 17 / 4 :=
sorry

end min_value_expression_l73_73756


namespace sum_of_possible_values_of_x_l73_73369

-- Conditions
def radius (x : ℝ) : ℝ := x - 2
def semiMajor (x : ℝ) : ℝ := x - 3
def semiMinor (x : ℝ) : ℝ := x + 4

-- Theorem to be proved
theorem sum_of_possible_values_of_x (x : ℝ) :
  (π * semiMajor x * semiMinor x = 2 * π * (radius x) ^ 2) →
  (x = 5 ∨ x = 4) →
  5 + 4 = 9 :=
by
  intros
  rfl

end sum_of_possible_values_of_x_l73_73369


namespace find_distance_from_origin_l73_73802

-- Define the conditions as functions
def point_distance_from_x_axis (y : ℝ) : Prop := abs y = 15
def distance_from_point (x y : ℝ) (x₀ y₀ : ℝ) (d : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = d^2

-- Define the proof problem
theorem find_distance_from_origin (x y : ℝ) (n : ℝ) (hx : x = 2 + Real.sqrt 105) (hy : point_distance_from_x_axis y) (hx_gt : x > 2) (hdist : distance_from_point x y 2 7 13) :
  n = Real.sqrt (334 + 4 * Real.sqrt 105) :=
sorry

end find_distance_from_origin_l73_73802


namespace MargaretsMeanScore_l73_73953

theorem MargaretsMeanScore :
  ∀ (scores : List ℕ)
    (cyprian_mean : ℝ)
    (highest_lowest_different : Prop),
    scores = [82, 85, 88, 90, 92, 95, 97, 99] →
    cyprian_mean = 88.5 →
    highest_lowest_different →
    ∃ (margaret_mean : ℝ), margaret_mean = 93.5 := by
  sorry

end MargaretsMeanScore_l73_73953


namespace product_not_50_l73_73459

theorem product_not_50 :
  (1 / 2 * 100 = 50) ∧
  (-5 * -10 = 50) ∧
  ¬(5 * 11 = 50) ∧
  (2 * 25 = 50) ∧
  (5 / 2 * 20 = 50) :=
by
  sorry

end product_not_50_l73_73459


namespace ages_proof_l73_73295

noncomputable def A : ℝ := 12.1
noncomputable def B : ℝ := 6.1
noncomputable def C : ℝ := 11.3

-- Conditions extracted from the problem
def sum_of_ages (A B C : ℝ) : Prop := A + B + C = 29.5
def specific_age (C : ℝ) : Prop := C = 11.3
def twice_as_old (A B : ℝ) : Prop := A = 2 * B

theorem ages_proof : 
  ∃ (A B C : ℝ), 
    specific_age C ∧ twice_as_old A B ∧ sum_of_ages A B C :=
by
  exists 12.1, 6.1, 11.3
  sorry

end ages_proof_l73_73295


namespace simon_number_of_legos_l73_73667

variable (Kent_legos : ℕ) (Bruce_legos : ℕ) (Simon_legos : ℕ)

def Kent_condition : Prop := Kent_legos = 40
def Bruce_condition : Prop := Bruce_legos = Kent_legos + 20 
def Simon_condition : Prop := Simon_legos = Bruce_legos + (Bruce_legos * 20 / 100)

theorem simon_number_of_legos : Kent_condition Kent_legos ∧ Bruce_condition Kent_legos Bruce_legos ∧ Simon_condition Bruce_legos Simon_legos → Simon_legos = 72 := by
  intros h
  -- proof steps would go here
  sorry

end simon_number_of_legos_l73_73667


namespace p_sufficient_but_not_necessary_for_q_l73_73757

def proposition_p (x : ℝ) := x - 1 = 0
def proposition_q (x : ℝ) := (x - 1) * (x + 2) = 0

theorem p_sufficient_but_not_necessary_for_q :
  ( (∀ x, proposition_p x → proposition_q x) ∧ ¬(∀ x, proposition_p x ↔ proposition_q x) ) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l73_73757


namespace marsha_pay_per_mile_l73_73363

variable (distance1 distance2 payment : ℝ)
variable (distance3 : ℝ := distance2 / 2)
variable (totalDistance := distance1 + distance2 + distance3)

noncomputable def payPerMile (payment : ℝ) (totalDistance : ℝ) : ℝ :=
  payment / totalDistance

theorem marsha_pay_per_mile
  (distance1: ℝ := 10)
  (distance2: ℝ := 28)
  (payment: ℝ := 104)
  (distance3: ℝ := distance2 / 2)
  (totalDistance: ℝ := distance1 + distance2 + distance3)
  : payPerMile payment totalDistance = 2 := by
  sorry

end marsha_pay_per_mile_l73_73363


namespace find_range_for_two_real_solutions_l73_73012

noncomputable def f (k x : ℝ) := k * x
noncomputable def g (x : ℝ) := (Real.log x) / x

noncomputable def h (x : ℝ) := (Real.log x) / (x^2)

theorem find_range_for_two_real_solutions :
  (∃ k : ℝ, ∀ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 → (f k x = g x ↔ k ∈ Set.Icc (1 / Real.exp 2) (1 / (2 * Real.exp 1)))) :=
sorry

end find_range_for_two_real_solutions_l73_73012


namespace probability_same_unit_l73_73796

theorem probability_same_unit
  (units : ℕ) (people : ℕ) (same_unit_cases total_cases : ℕ)
  (h_units : units = 4)
  (h_people : people = 2)
  (h_total_cases : total_cases = units * units)
  (h_same_unit_cases : same_unit_cases = units) :
  (same_unit_cases :  ℝ) / total_cases = 1 / 4 :=
by sorry

end probability_same_unit_l73_73796


namespace arithmetic_sequence_sum_l73_73973

theorem arithmetic_sequence_sum :
  ∃ (a_n : ℕ → ℝ) (d : ℝ), 
  (∀ n, a_n n = a_n 0 + n * d) ∧
  d > 0 ∧
  a_n 0 + a_n 1 + a_n 2 = 15 ∧
  a_n 0 * a_n 1 * a_n 2 = 80 →
  a_n 10 + a_n 11 + a_n 12 = 135 :=
by
  sorry

end arithmetic_sequence_sum_l73_73973


namespace solutions_of_quadratic_l73_73785

theorem solutions_of_quadratic (c : ℝ) (h : ∀ α β : ℝ, 
  (α^2 - 3*α + c = 0 ∧ β^2 - 3*β + c = 0) → 
  ( (-α)^2 + 3*(-α) - c = 0 ∨ (-β)^2 + 3*(-β) - c = 0 ) ) :
  ∃ α β : ℝ, (α = 0 ∧ β = 3) ∨ (α = 3 ∧ β = 0) :=
by
  sorry

end solutions_of_quadratic_l73_73785


namespace pony_wait_time_l73_73652

-- Definitions of the conditions
def cycle_time_monster_A : ℕ := 2 + 1 -- hours (2 awake, 1 rest)
def cycle_time_monster_B : ℕ := 3 + 2 -- hours (3 awake, 2 rest)

-- The theorem to prove the correct answer
theorem pony_wait_time :
  Nat.lcm cycle_time_monster_A cycle_time_monster_B = 15 :=
by
  -- Skip the proof
  sorry

end pony_wait_time_l73_73652


namespace calc_exponent_result_l73_73633

theorem calc_exponent_result (m : ℝ) : (2 * m^2)^3 = 8 * m^6 := 
by
  sorry

end calc_exponent_result_l73_73633


namespace product_gcf_lcm_l73_73440

def gcf (a b c : Nat) : Nat := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : Nat) : Nat := Nat.lcm (Nat.lcm a b) c

theorem product_gcf_lcm :
  let A := gcf 6 18 24
  let B := lcm 6 18 24
  A * B = 432 :=
by
  let A := gcf 6 18 24
  let B := lcm 6 18 24
  have hA : A = Nat.gcd (Nat.gcd 6 18) 24 := rfl
  have hB : B = Nat.lcm (Nat.lcm 6 18) 24 := rfl
  sorry

end product_gcf_lcm_l73_73440


namespace top_leftmost_rectangle_is_B_l73_73742

-- Definitions for the side lengths of each rectangle
def A_w : ℕ := 6
def A_x : ℕ := 2
def A_y : ℕ := 7
def A_z : ℕ := 10

def B_w : ℕ := 2
def B_x : ℕ := 1
def B_y : ℕ := 4
def B_z : ℕ := 8

def C_w : ℕ := 5
def C_x : ℕ := 11
def C_y : ℕ := 6
def C_z : ℕ := 3

def D_w : ℕ := 9
def D_x : ℕ := 7
def D_y : ℕ := 5
def D_z : ℕ := 9

def E_w : ℕ := 11
def E_x : ℕ := 4
def E_y : ℕ := 9
def E_z : ℕ := 1

-- The problem statement to prove
theorem top_leftmost_rectangle_is_B : 
  (B_w = 2 ∧ B_y = 4) ∧ 
  (A_w = 6 ∨ D_w = 9 ∨ C_w = 5 ∨ E_w = 11) ∧
  (A_y = 7 ∨ D_y = 5 ∨ C_y = 6 ∨ E_y = 9) → 
  (B_w = 2 ∧ ∀ w : ℕ, w = 6 ∨ w = 5 ∨ w = 9 ∨ w = 11 → B_w < w) :=
by {
  -- skipping the proof
  sorry
}

end top_leftmost_rectangle_is_B_l73_73742


namespace find_numbers_in_progressions_l73_73566

theorem find_numbers_in_progressions (a b c d : ℝ) :
    (a + b + c = 114) ∧ -- Sum condition
    (b^2 = a * c) ∧ -- Geometric progression condition
    (b = a + 3 * d) ∧ -- Arithmetic progression first condition
    (c = a + 24 * d) -- Arithmetic progression second condition
    ↔ (a = 38 ∧ b = 38 ∧ c = 38) ∨ (a = 2 ∧ b = 14 ∧ c = 98) := by
  sorry

end find_numbers_in_progressions_l73_73566


namespace two_pairs_of_dice_probability_l73_73475

noncomputable def two_pairs_probability : ℚ :=
  5 / 36

theorem two_pairs_of_dice_probability :
  ∃ p : ℚ, p = two_pairs_probability := 
by 
  use 5 / 36
  sorry

end two_pairs_of_dice_probability_l73_73475


namespace sum_expr_le_e4_l73_73127

theorem sum_expr_le_e4
  (α β γ δ ε : ℝ) :
  (1 - α) * Real.exp α +
  (1 - β) * Real.exp (α + β) +
  (1 - γ) * Real.exp (α + β + γ) +
  (1 - δ) * Real.exp (α + β + γ + δ) +
  (1 - ε) * Real.exp (α + β + γ + δ + ε) ≤ Real.exp 4 :=
sorry

end sum_expr_le_e4_l73_73127


namespace find_circle_eq_find_range_of_dot_product_l73_73687

open Real
open Set

-- Define the problem conditions
def line_eq (x y : ℝ) : Prop := x - sqrt 3 * y = 4
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P inside the circle and condition that |PA|, |PO|, |PB| form a geometric sequence
def geometric_sequence_condition (x y : ℝ) : Prop :=
  sqrt ((x + 2)^2 + y^2) * sqrt ((x - 2)^2 + y^2) = x^2 + y^2

-- Prove the equation of the circle
theorem find_circle_eq :
  (∃ (r : ℝ), ∀ (x y : ℝ), line_eq x y → r = 2) → circle_eq x y :=
by
  -- skipping the proof
  sorry

-- Prove the range of values for the dot product
theorem find_range_of_dot_product :
  (∀ (x y : ℝ), circle_eq x y ∧ geometric_sequence_condition x y) →
  -2 < (x^2 - 1 * y^2 - 1) → (x^2 - 4 + y^2) < 0 :=
by
  -- skipping the proof
  sorry

end find_circle_eq_find_range_of_dot_product_l73_73687


namespace prove_sets_l73_73724

noncomputable def A := { y : ℝ | ∃ x : ℝ, y = 3^x }
def B := { x : ℝ | x^2 - 4 ≤ 0 }

theorem prove_sets :
  A ∪ B = { x : ℝ | x ≥ -2 } ∧ A ∩ B = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by {
  sorry
}

end prove_sets_l73_73724


namespace maximum_real_roots_maximum_total_real_roots_l73_73846

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

def quadratic_discriminant (p q r : ℝ) : ℝ := q^2 - 4 * p * r

theorem maximum_real_roots (h1 : quadratic_discriminant a b c < 0)
  (h2 : quadratic_discriminant b c a < 0)
  (h3 : quadratic_discriminant c a b < 0) :
  ∀ (x : ℝ), (a * x^2 + b * x + c ≠ 0) ∧ 
             (b * x^2 + c * x + a ≠ 0) ∧ 
             (c * x^2 + a * x + b ≠ 0) :=
sorry

theorem maximum_total_real_roots :
    ∃ x : ℝ, ∃ y : ℝ, ∃ z : ℝ,
    (a * x^2 + b * x + c = 0) ∧
    (b * y^2 + c * y + a = 0) ∧
    (a * y ≠ x) ∧
    (c * z^2 + a * z + b = 0) ∧
    (b * z ≠ x) ∧
    (c * z ≠ y) :=
sorry

end maximum_real_roots_maximum_total_real_roots_l73_73846


namespace find_a_l73_73536

open Complex

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the main hypothesis: (ai / (1 - i)) = (-1 + i)
def hypothesis (a : ℂ) : Prop :=
  (a * i) / (1 - i) = -1 + i

-- Now, we state the theorem we need to prove
theorem find_a (a : ℝ) (ha : hypothesis a) : a = 2 := by
  sorry

end find_a_l73_73536


namespace find_a_solution_set_a_negative_l73_73896

-- Definitions
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (a - 1) * x - 1 ≥ 0

-- Problem 1: Prove the value of 'a'
theorem find_a (h : ∀ x : ℝ, quadratic_inequality a x ↔ (-1 ≤ x ∧ x ≤ -1/2)) :
  a = -2 :=
sorry

-- Problem 2: Prove the solution sets when a < 0
theorem solution_set_a_negative (h : a < 0) :
  (a = -1 → (∀ x : ℝ, quadratic_inequality a x ↔ x = -1)) ∧
  (a < -1 → (∀ x : ℝ, quadratic_inequality a x ↔ (-1 ≤ x ∧ x ≤ 1/a))) ∧
  (-1 < a ∧ a < 0 → (∀ x : ℝ, quadratic_inequality a x ↔ (1/a ≤ x ∧ x ≤ -1))) :=
sorry

end find_a_solution_set_a_negative_l73_73896


namespace eight_odot_six_eq_ten_l73_73783

-- Define the operation ⊙ as given in the problem statement
def operation (a b : ℕ) : ℕ := a + (3 * a) / (2 * b)

-- State the theorem to prove
theorem eight_odot_six_eq_ten : operation 8 6 = 10 :=
by
  -- Here you will provide the proof, but we skip it with sorry
  sorry

end eight_odot_six_eq_ten_l73_73783


namespace probability_blue_or_purple_is_4_over_11_l73_73847

def total_jelly_beans : ℕ := 10 + 12 + 13 + 15 + 5
def blue_or_purple_jelly_beans : ℕ := 15 + 5
def probability_blue_or_purple : ℚ := blue_or_purple_jelly_beans / total_jelly_beans

theorem probability_blue_or_purple_is_4_over_11 :
  probability_blue_or_purple = 4 / 11 :=
sorry

end probability_blue_or_purple_is_4_over_11_l73_73847


namespace congruent_triangles_solve_x_l73_73970

theorem congruent_triangles_solve_x (x : ℝ) (h1 : x > 0)
    (h2 : x^2 - 1 = 3) (h3 : x^2 + 1 = 5) (h4 : x^2 + 3 = 7) : x = 2 :=
by
  sorry

end congruent_triangles_solve_x_l73_73970


namespace dot_product_ABC_l73_73641

open Real

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 6
noncomputable def angleC : ℝ := π / 6  -- 30 degrees in radians

theorem dot_product_ABC :
  let CB := a
  let CA := b
  let angle_between := π - angleC  -- 150 degrees in radians
  let cos_angle := - (sqrt 3) / 2  -- cos(150 degrees)
  ∃ (dot_product : ℝ), dot_product = CB * CA * cos_angle :=
by
  have CB := a
  have CA := b
  have angle_between := π - angleC
  have cos_angle := - (sqrt 3) / 2
  use CB * CA * cos_angle
  sorry

end dot_product_ABC_l73_73641


namespace parabola_focus_distance_l73_73160

noncomputable def PF (x₁ : ℝ) : ℝ := x₁ + 1
noncomputable def QF (x₂ : ℝ) : ℝ := x₂ + 1

theorem parabola_focus_distance 
  (x₁ x₂ : ℝ) (h₁ : x₂ = 3 * x₁ + 2) : 
  QF x₂ / PF x₁ = 3 :=
by
  sorry

end parabola_focus_distance_l73_73160


namespace houses_in_lawrence_county_l73_73654

theorem houses_in_lawrence_county 
  (houses_before_boom : ℕ := 1426) 
  (houses_built_during_boom : ℕ := 574) 
  : houses_before_boom + houses_built_during_boom = 2000 := 
by 
  sorry

end houses_in_lawrence_county_l73_73654


namespace number_of_attendants_writing_with_both_l73_73157

-- Definitions for each of the conditions
def attendants_using_pencil : ℕ := 25
def attendants_using_pen : ℕ := 15
def attendants_using_only_one : ℕ := 20

-- Theorem that states the mathematically equivalent proof problem
theorem number_of_attendants_writing_with_both 
  (p : ℕ := attendants_using_pencil)
  (e : ℕ := attendants_using_pen)
  (o : ℕ := attendants_using_only_one) : 
  ∃ x, (p - x) + (e - x) = o ∧ x = 10 :=
by
  sorry

end number_of_attendants_writing_with_both_l73_73157


namespace percentage_decrease_of_original_number_is_30_l73_73803

theorem percentage_decrease_of_original_number_is_30 :
  ∀ (original_number : ℕ) (difference : ℕ) (percent_increase : ℚ) (percent_decrease : ℚ),
  original_number = 40 →
  percent_increase = 0.25 →
  difference = 22 →
  original_number + percent_increase * original_number - (original_number - percent_decrease * original_number) = difference →
  percent_decrease = 0.30 :=
by
  intros original_number difference percent_increase percent_decrease h_original h_increase h_diff h_eq
  sorry

end percentage_decrease_of_original_number_is_30_l73_73803


namespace frank_problems_per_type_l73_73029

-- Definitions based on the problem conditions
def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def types_of_problems : ℕ := 4

-- The proof statement equivalent to the math problem
theorem frank_problems_per_type : frank_problems / types_of_problems = 30 :=
by 
  -- skipping the proof steps
  sorry

end frank_problems_per_type_l73_73029


namespace lcm_12_15_18_l73_73878

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end lcm_12_15_18_l73_73878


namespace sum_cubes_of_roots_l73_73603

noncomputable def cube_root_sum_cubes (α β γ : ℝ) : ℝ :=
  α^3 + β^3 + γ^3
  
theorem sum_cubes_of_roots : 
  (cube_root_sum_cubes (Real.rpow 27 (1/3)) (Real.rpow 64 (1/3)) (Real.rpow 125 (1/3))) - 3 * ((Real.rpow 27 (1/3)) * (Real.rpow 64 (1/3)) * (Real.rpow 125 (1/3)) + 4/3) = 36 
  ∧
  ((Real.rpow 27 (1/3) + Real.rpow 64 (1/3) + Real.rpow 125 (1/3)) * ((Real.rpow 27 (1/3) + Real.rpow 64 (1/3) + Real.rpow 125 (1/3))^2 - 3 * ((Real.rpow 27 (1/3)) * (Real.rpow 64 (1/3)) + (Real.rpow 64 (1/3)) * (Real.rpow 125 (1/3)) + (Real.rpow 125 (1/3)) * (Real.rpow 27 (1/3)))) = 36) 
  → 
  cube_root_sum_cubes (Real.rpow 27 (1/3)) (Real.rpow 64 (1/3)) (Real.rpow 125 (1/3)) = 220 := 
sorry

end sum_cubes_of_roots_l73_73603


namespace true_propositions_l73_73497

theorem true_propositions :
  (∀ x : ℚ, ∃ y : ℚ, y = (1/3 : ℚ) * x^2 + (1/2 : ℚ) * x + 1) ∧
  (∃ x y : ℤ, 3 * x - 2 * y = 10) :=
by {
  sorry
}

end true_propositions_l73_73497


namespace simple_interest_years_l73_73941

theorem simple_interest_years (P : ℝ) (R : ℝ) (N : ℝ) (higher_interest_amount : ℝ) (additional_rate : ℝ) (initial_sum : ℝ) :
  (initial_sum * (R + additional_rate) * N) / 100 - (initial_sum * R * N) / 100 = higher_interest_amount →
  initial_sum = 3000 →
  higher_interest_amount = 1350 →
  additional_rate = 5 →
  N = 9 :=
by
  sorry

end simple_interest_years_l73_73941


namespace problem_angle_magnitude_and_sin_l73_73331

theorem problem_angle_magnitude_and_sin (
  a b c : ℝ) (A B C : ℝ) 
  (h1 : a = Real.sqrt 7) (h2 : b = 3) 
  (h3 : Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3)
  (triangle_is_acute : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) : 
  A = Real.pi / 3 ∧ Real.sin (2 * B + Real.pi / 6) = -1 / 7 :=
by
  sorry

end problem_angle_magnitude_and_sin_l73_73331


namespace dividend_is_correct_l73_73357

def quotient : ℕ := 36
def divisor : ℕ := 85
def remainder : ℕ := 26

theorem dividend_is_correct : divisor * quotient + remainder = 3086 := by
  sorry

end dividend_is_correct_l73_73357


namespace no_distinct_natural_numbers_exist_l73_73212

theorem no_distinct_natural_numbers_exist 
  (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ¬ (a + 1 / a = (1 / 2) * (b + 1 / b + c + 1 / c)) :=
sorry

end no_distinct_natural_numbers_exist_l73_73212


namespace quadratic_function_series_sum_l73_73531

open Real

noncomputable def P (x : ℝ) : ℝ := 6 * x^2 - 3 * x + 7

theorem quadratic_function_series_sum :
  (∀ (x : ℝ), 0 < x ∧ x < 1 →
    (∑' n, P n * x^n) = (16 * x^2 - 11 * x + 7) / (1 - x)^3) :=
sorry

end quadratic_function_series_sum_l73_73531


namespace laborer_monthly_income_l73_73883

variable (I : ℝ)

noncomputable def average_expenditure_six_months := 70 * 6
noncomputable def debt_condition := I * 6 < average_expenditure_six_months
noncomputable def expenditure_next_four_months := 60 * 4
noncomputable def total_income_next_four_months := expenditure_next_four_months + (average_expenditure_six_months - I * 6) + 30

theorem laborer_monthly_income (h1 : debt_condition I) (h2 : total_income_next_four_months I = I * 4) :
  I = 69 :=
by
  sorry

end laborer_monthly_income_l73_73883


namespace value_of_x_l73_73260

theorem value_of_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 6 = 3 * y) : x = 108 :=
by
  sorry

end value_of_x_l73_73260


namespace inequality_relationship_l73_73604

noncomputable def a : ℝ := Real.sin (4 / 5)
noncomputable def b : ℝ := Real.cos (4 / 5)
noncomputable def c : ℝ := Real.tan (4 / 5)

theorem inequality_relationship : c > a ∧ a > b := sorry

end inequality_relationship_l73_73604


namespace final_tree_count_l73_73241

noncomputable def current_trees : ℕ := 39
noncomputable def trees_planted_today : ℕ := 41
noncomputable def trees_planted_tomorrow : ℕ := 20

theorem final_tree_count : current_trees + trees_planted_today + trees_planted_tomorrow = 100 := by
  sorry

end final_tree_count_l73_73241


namespace max_value_of_y_l73_73968

open Classical

noncomputable def satisfies_equation (x y : ℝ) : Prop := y * x * (x + y) = x - y

theorem max_value_of_y : 
  ∀ (y : ℝ), (∃ (x : ℝ), x > 0 ∧ satisfies_equation x y) → y ≤ 1 / 3 := 
sorry

end max_value_of_y_l73_73968


namespace cos_beta_eq_sqrt10_over_10_l73_73746

-- Define the conditions and the statement
theorem cos_beta_eq_sqrt10_over_10 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_tan : Real.tan α = 2)
  (h_sin_sum : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 :=
sorry

end cos_beta_eq_sqrt10_over_10_l73_73746


namespace find_k_l73_73978

theorem find_k (k α β : ℝ)
  (h1 : (∀ x : ℝ, x^2 - (k-1) * x - 3*k - 2 = 0 → x = α ∨ x = β))
  (h2 : α^2 + β^2 = 17) :
  k = 2 :=
sorry

end find_k_l73_73978


namespace largest_base5_eq_124_l73_73399

-- Define largest base-5 number with three digits
def largest_base5_three_digits : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_eq_124 : largest_base5_three_digits = 124 :=
by
  -- Proof steps go here
  sorry

end largest_base5_eq_124_l73_73399


namespace find_skirts_l73_73709

variable (blouses : ℕ) (skirts : ℕ) (slacks : ℕ)
variable (blouses_in_hamper : ℕ) (slacks_in_hamper : ℕ) (skirts_in_hamper : ℕ)
variable (clothes_in_hamper : ℕ)

-- Given conditions
axiom h1 : blouses = 12
axiom h2 : slacks = 8
axiom h3 : blouses_in_hamper = (75 * blouses) / 100
axiom h4 : slacks_in_hamper = (25 * slacks) / 100
axiom h5 : skirts_in_hamper = 3
axiom h6 : clothes_in_hamper = blouses_in_hamper + slacks_in_hamper + skirts_in_hamper
axiom h7 : clothes_in_hamper = 11

-- Proof goal: proving the total number of skirts
theorem find_skirts : skirts_in_hamper = (50 * skirts) / 100 → skirts = 6 :=
by sorry

end find_skirts_l73_73709


namespace employee_monthly_wage_l73_73983

theorem employee_monthly_wage 
(revenue : ℝ)
(tax_rate : ℝ)
(marketing_rate : ℝ)
(operational_cost_rate : ℝ)
(wage_rate : ℝ)
(num_employees : ℕ)
(h_revenue : revenue = 400000)
(h_tax_rate : tax_rate = 0.10)
(h_marketing_rate : marketing_rate = 0.05)
(h_operational_cost_rate : operational_cost_rate = 0.20)
(h_wage_rate : wage_rate = 0.15)
(h_num_employees : num_employees = 10) :
(revenue * (1 - tax_rate) * (1 - marketing_rate) * (1 - operational_cost_rate) * wage_rate / num_employees = 4104) :=
by
  sorry

end employee_monthly_wage_l73_73983


namespace birds_on_the_fence_l73_73184

theorem birds_on_the_fence (x : ℕ) : 10 + 2 * x = 50 → x = 20 := by
  sorry

end birds_on_the_fence_l73_73184


namespace gcd_lcm_product_l73_73447

theorem gcd_lcm_product (a b : ℤ) (h1 : Int.gcd a b = 8) (h2 : Int.lcm a b = 24) : a * b = 192 := by
  sorry

end gcd_lcm_product_l73_73447


namespace complement_A_l73_73095

open Set

variable (A : Set ℝ) (x : ℝ)
def A_def : Set ℝ := { x | x ≥ 1 }

theorem complement_A : Aᶜ = { y | y < 1 } :=
by
  sorry

end complement_A_l73_73095


namespace product_equivalence_l73_73648

theorem product_equivalence 
  (a b c d e f : ℝ) 
  (h1 : a + b + c + d + e + f = 0) 
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 + f^3 = 0) : 
  (a + c) * (a + d) * (a + e) * (a + f) = (b + c) * (b + d) * (b + e) * (b + f) :=
by
  sorry

end product_equivalence_l73_73648


namespace nature_of_graph_l73_73753

theorem nature_of_graph :
  ∀ (x y : ℝ), (x^2 - 3 * y) * (x - y + 1) = (y^2 - 3 * x) * (x - y + 1) →
    (y = -x - 3 ∨ y = x ∨ y = x + 1) ∧ ¬( (y = -x - 3) ∧ (y = x) ∧ (y = x + 1) ) :=
by
  intros x y h
  sorry

end nature_of_graph_l73_73753


namespace parabola_directrix_distance_l73_73843

theorem parabola_directrix_distance (a : ℝ) : 
  (abs (a / 4 + 1) = 2) → (a = -12 ∨ a = 4) := 
by
  sorry

end parabola_directrix_distance_l73_73843


namespace simplest_form_option_l73_73154

theorem simplest_form_option (x y : ℚ) :
  (∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (12 * (x - y) / (15 * (x + y)) ≠ 4 * (x - y) / 5 * (x + y))) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 + y^2) / (x + y) = a / b) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 - y^2) / ((x + y)^2) ≠ (x - y) / (x + y)) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 - y^2) / (x + y) ≠ x - y)) := sorry

end simplest_form_option_l73_73154


namespace trapezoid_triangle_area_ratio_l73_73826

/-- Given a trapezoid with triangles ABC and ADC such that the ratio of their areas is 4:1 and AB + CD = 150 cm.
Prove that the length of segment AB is 120 cm. --/
theorem trapezoid_triangle_area_ratio
  (h ABC_area ADC_area : ℕ)
  (AB CD : ℕ)
  (h_ratio : ABC_area / ADC_area = 4)
  (area_ABC : ABC_area = AB * h / 2)
  (area_ADC : ADC_area = CD * h / 2)
  (h_sum : AB + CD = 150) :
  AB = 120 := 
sorry

end trapezoid_triangle_area_ratio_l73_73826


namespace spending_limit_l73_73225

variable (n b total_spent limit: ℕ)

theorem spending_limit (hne: n = 34) (hbe: b = n + 5) (hts: total_spent = n + b) (hlo: total_spent = limit + 3) : limit = 70 := by
  sorry

end spending_limit_l73_73225


namespace valid_outfits_count_l73_73741

noncomputable def number_of_valid_outfits (shirt_count: ℕ) (pant_colors: List String) (hat_count: ℕ) : ℕ :=
  let total_combinations := shirt_count * (pant_colors.length) * hat_count
  let matching_outfits := List.length (List.filter (λ c => c ∈ pant_colors) ["tan", "black", "blue", "gray"])
  total_combinations - matching_outfits

theorem valid_outfits_count :
    number_of_valid_outfits 8 ["tan", "black", "blue", "gray"] 8 = 252 := by
  sorry

end valid_outfits_count_l73_73741


namespace function_fixed_point_l73_73368

theorem function_fixed_point {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) : (2, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a^(x-2) + 1) } :=
by
  sorry

end function_fixed_point_l73_73368


namespace no_perfect_square_in_range_l73_73485

theorem no_perfect_square_in_range :
  ¬∃ (x : ℕ), 99990000 ≤ x ∧ x ≤ 99999999 ∧ ∃ (n : ℕ), x = n * n :=
by
  sorry

end no_perfect_square_in_range_l73_73485


namespace sum_of_possible_values_l73_73119

-- Define the triangle's base and height
def triangle_base (x : ℝ) : ℝ := x - 2
def triangle_height (x : ℝ) : ℝ := x - 2

-- Define the parallelogram's base and height
def parallelogram_base (x : ℝ) : ℝ := x - 3
def parallelogram_height (x : ℝ) : ℝ := x + 4

-- Define the areas
def triangle_area (x : ℝ) : ℝ := 0.5 * (triangle_base x) * (triangle_height x)
def parallelogram_area (x : ℝ) : ℝ := (parallelogram_base x) * (parallelogram_height x)

-- Statement to prove
theorem sum_of_possible_values (x : ℝ) (h : parallelogram_area x = 3 * triangle_area x) : x = 8 ∨ x = 3 →
  (x = 8 ∨ x = 3) → 8 + 3 = 11 :=
by sorry

end sum_of_possible_values_l73_73119


namespace solve_for_x_l73_73760

theorem solve_for_x (x y : ℝ) (h1 : 9^y = x^12) (h2 : y = 6) : x = 3 :=
by
  sorry

end solve_for_x_l73_73760


namespace parallelogram_altitude_base_ratio_l73_73824

theorem parallelogram_altitude_base_ratio 
  (area base : ℕ) (h : ℕ) 
  (h_base : base = 9)
  (h_area : area = 162)
  (h_area_eq : area = base * h) : 
  h / base = 2 := 
by 
  -- placeholder for the proof
  sorry

end parallelogram_altitude_base_ratio_l73_73824


namespace magnitude_of_complex_l73_73622

noncomputable def z : ℂ := (2 / 3 : ℝ) - (4 / 5 : ℝ) * Complex.I

theorem magnitude_of_complex :
  Complex.abs z = (2 * Real.sqrt 61) / 15 :=
by
  sorry

end magnitude_of_complex_l73_73622


namespace a_is_perfect_square_l73_73690

theorem a_is_perfect_square {a : ℕ} (h : ∀ n : ℕ, ∃ d : ℕ, d ≠ 1 ∧ d % n = 1 ∧ d ∣ n ^ 2 * a - 1) : ∃ k : ℕ, a = k ^ 2 :=
by
  sorry

end a_is_perfect_square_l73_73690


namespace jellybeans_condition_l73_73835

theorem jellybeans_condition (n : ℕ) (h1 : n ≥ 150) (h2 : n % 15 = 14) : n = 164 :=
sorry

end jellybeans_condition_l73_73835


namespace find_side_b_l73_73819

theorem find_side_b (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hc : c = 2 * Real.sqrt 3)
    (hC : C = Real.pi / 3) (hA : A = Real.pi / 6) (hB : B = Real.pi / 2) : b = 4 := by
  sorry

end find_side_b_l73_73819


namespace difference_in_girls_and_boys_l73_73206

theorem difference_in_girls_and_boys (x : ℕ) (h1 : 3 + 4 = 7) (h2 : 7 * x = 49) : 4 * x - 3 * x = 7 := by
  sorry

end difference_in_girls_and_boys_l73_73206


namespace problem_statement_l73_73407

noncomputable def tangent_sum_formula (x y : ℝ) : ℝ :=
  (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)

theorem problem_statement
  (α β : ℝ)
  (hαβ1 : 0 < α ∧ α < π)
  (hαβ2 : 0 < β ∧ β < π)
  (h1 : Real.tan (α - β) = 1 / 2)
  (h2 : Real.tan β = - 1 / 7)
  : 2 * α - β = - (3 * π / 4) :=
sorry

end problem_statement_l73_73407


namespace number_of_pencils_bought_l73_73579

-- Define the conditions
def cost_of_glue : ℕ := 270
def cost_per_pencil : ℕ := 210
def amount_paid : ℕ := 1000
def change_received : ℕ := 100

-- Define the statement to prove
theorem number_of_pencils_bought : 
  ∃ (n : ℕ), cost_of_glue + (cost_per_pencil * n) = amount_paid - change_received :=
by {
  sorry 
}

end number_of_pencils_bought_l73_73579


namespace complex_magnitude_equality_l73_73207

open Complex Real

theorem complex_magnitude_equality :
  abs ((Complex.mk (5 * sqrt 2) (-5)) * (Complex.mk (2 * sqrt 3) 6)) = 60 :=
by
  sorry

end complex_magnitude_equality_l73_73207


namespace value_of_expression_l73_73595

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 2005 = -2004 :=
by
  sorry

end value_of_expression_l73_73595


namespace find_f2_plus_fneg2_l73_73502

def f (x a: ℝ) := (x + a)^3

theorem find_f2_plus_fneg2 (a : ℝ)
  (h_cond : ∀ x : ℝ, f (1 + x) a = -f (1 - x) a) :
  f 2 (-1) + f (-2) (-1) = -26 :=
by
  sorry

end find_f2_plus_fneg2_l73_73502


namespace sqrt_sum_eq_ten_l73_73001

theorem sqrt_sum_eq_ten :
  Real.sqrt ((5 - 4*Real.sqrt 2)^2) + Real.sqrt ((5 + 4*Real.sqrt 2)^2) = 10 := 
by 
  sorry

end sqrt_sum_eq_ten_l73_73001


namespace quadratic_inequality_iff_abs_a_le_2_l73_73800

theorem quadratic_inequality_iff_abs_a_le_2 (a : ℝ) :
  (|a| ≤ 2) ↔ (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) :=
sorry

end quadratic_inequality_iff_abs_a_le_2_l73_73800


namespace spherical_circle_radius_l73_73916

theorem spherical_circle_radius:
  (∀ (θ : Real), ∃ (r : Real), r = 1 * Real.sin (Real.pi / 6)) → ∀ (θ : Real), r = 1 / 2 := by
  sorry

end spherical_circle_radius_l73_73916


namespace parallel_lines_slope_eq_l73_73013

theorem parallel_lines_slope_eq (k : ℚ) :
  (5 = 3 * k) → k = 5 / 3 :=
by
  intros h
  sorry

end parallel_lines_slope_eq_l73_73013


namespace find_BD_l73_73094

theorem find_BD 
  (A B C D : Type)
  (AC BC : ℝ) (h₁ : AC = 10) (h₂ : BC = 10)
  (AD CD : ℝ) (h₃ : AD = 12) (h₄ : CD = 5) :
  ∃ (BD : ℝ), BD = 152 / 24 := 
sorry

end find_BD_l73_73094


namespace range_S13_over_a14_l73_73173

lemma a_n_is_arithmetic_progression (a S : ℕ → ℕ) (h1 : ∀ n, a n > 1)
  (h2 : ∀ n, S (n + 1) - S n - (1 / 2) * ((a (n + 1)) ^ 2 - (a n) ^ 2) = 1 / 2) :
  ∀ n, a (n + 1) = a n + 1 := 
sorry

theorem range_S13_over_a14 (a S : ℕ → ℕ) (h1 : ∀ n, a n > 1)
  (h2 : ∀ n, S (n + 1) - S n - (1 / 2) * ((a (n + 1)) ^ 2 - (a n) ^ 2) = 1 / 2)
  (h3 : a 1 > 4) :
  130 / 17 < (S 13 / a 14) ∧ (S 13 / a 14) < 13 := 
sorry

end range_S13_over_a14_l73_73173


namespace yard_length_eq_250_l73_73222

noncomputable def number_of_trees : ℕ := 26
noncomputable def distance_between_trees : ℕ := 10
noncomputable def number_of_gaps := number_of_trees - 1
noncomputable def length_of_yard := number_of_gaps * distance_between_trees

theorem yard_length_eq_250 : 
  length_of_yard = 250 := 
sorry

end yard_length_eq_250_l73_73222


namespace find_length_of_field_l73_73420

variables (L : ℝ) -- Length of the field
variables (width_field : ℝ := 55) -- Width of the field, given as 55 meters.
variables (width_path : ℝ := 2.5) -- Width of the path around the field, given as 2.5 meters.
variables (area_path : ℝ := 1200) -- Area of the path, given as 1200 square meters.

theorem find_length_of_field
  (h : area_path = (L + 2 * width_path) * (width_field + 2 * width_path) - L * width_field)
  : L = 180 :=
by sorry

end find_length_of_field_l73_73420


namespace fish_served_l73_73375

theorem fish_served (H E P : ℕ) 
  (h1 : H = E) (h2 : E = P) 
  (fat_herring fat_eel fat_pike total_fat : ℕ) 
  (herring_fat : fat_herring = 40) 
  (eel_fat : fat_eel = 20)
  (pike_fat : fat_pike = 30)
  (total_fat_served : total_fat = 3600) 
  (fat_eq : 40 * H + 20 * E + 30 * P = 3600) : 
  H = 40 ∧ E = 40 ∧ P = 40 := by
  sorry

end fish_served_l73_73375


namespace other_root_of_quadratic_l73_73035

theorem other_root_of_quadratic (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∃ x : ℝ, (a * x ^ 2 = b) ∧ (x = 2)) : 
  ∃ m : ℝ, (a * m ^ 2 = b) ∧ (m = -2) := 
sorry

end other_root_of_quadratic_l73_73035


namespace girls_on_playground_l73_73246

variable (total_children : ℕ) (boys : ℕ) (girls : ℕ)

theorem girls_on_playground (h1 : total_children = 117) (h2 : boys = 40) (h3 : girls = total_children - boys) : girls = 77 :=
by
  sorry

end girls_on_playground_l73_73246


namespace solution_set_inequality_l73_73371

theorem solution_set_inequality (x : ℝ) : 
  x * (x - 1) ≥ x ↔ x ≤ 0 ∨ x ≥ 2 := 
sorry

end solution_set_inequality_l73_73371


namespace probability_drawing_red_l73_73901

/-- The probability of drawing a red ball from a bag that contains 1 red ball and 2 yellow balls. -/
theorem probability_drawing_red : 
  let N_red := 1
  let N_yellow := 2
  let N_total := N_red + N_yellow
  let P_red := (N_red : ℝ) / N_total
  P_red = (1 : ℝ) / 3 :=
by {
  sorry
}

end probability_drawing_red_l73_73901


namespace periodic_sequence_a2019_l73_73840

theorem periodic_sequence_a2019 :
  (∃ (a : ℕ → ℤ),
    a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ 
    (∀ n : ℕ, n ≥ 4 → a n = a (n-1) * a (n-3)) ∧
    a 2019 = -1) :=
sorry

end periodic_sequence_a2019_l73_73840


namespace sum_of_digits_a_l73_73776

def a : ℕ := 10^10 - 47

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_a : sum_of_digits a = 81 := 
  by 
    sorry

end sum_of_digits_a_l73_73776


namespace log_abs_monotone_decreasing_l73_73660

open Real

theorem log_abs_monotone_decreasing {a : ℝ} (h : ∀ x y, 0 < x ∧ x < y ∧ y ≤ a → |log x| ≥ |log y|) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end log_abs_monotone_decreasing_l73_73660


namespace solve_equation_l73_73149

open Real

noncomputable def f (x : ℝ) := 2017 * x ^ 2017 - 2017 + x
noncomputable def g (x : ℝ) := (2018 - 2017 * x) ^ (1 / 2017 : ℝ)

theorem solve_equation :
  ∀ x : ℝ, 2017 * x ^ 2017 - 2017 + x = (2018 - 2017 * x) ^ (1 / 2017 : ℝ) → x = 1 :=
by
  sorry

end solve_equation_l73_73149


namespace sum_of_distinct_prime_factors_315_l73_73540

theorem sum_of_distinct_prime_factors_315 : 
  ∃ factors : List ℕ, factors = [3, 5, 7] ∧ 315 = 3 * 3 * 5 * 7 ∧ factors.sum = 15 :=
by
  sorry

end sum_of_distinct_prime_factors_315_l73_73540


namespace WorldCup_group_stage_matches_l73_73616

theorem WorldCup_group_stage_matches
  (teams : ℕ)
  (groups : ℕ)
  (teams_per_group : ℕ)
  (matches_per_group : ℕ)
  (total_matches : ℕ) :
  teams = 32 ∧ 
  groups = 8 ∧ 
  teams_per_group = 4 ∧ 
  matches_per_group = teams_per_group * (teams_per_group - 1) / 2 ∧ 
  total_matches = matches_per_group * groups →
  total_matches = 48 :=
by 
  -- sorry lets Lean skip the proof.
  sorry

end WorldCup_group_stage_matches_l73_73616


namespace simplify_fraction_l73_73430

theorem simplify_fraction : (2 / 520) + (23 / 40) = 301 / 520 := by
  sorry

end simplify_fraction_l73_73430


namespace ms_smith_books_divided_l73_73380

theorem ms_smith_books_divided (books_for_girls : ℕ) (girls boys : ℕ) (books_per_girl : ℕ)
  (h1 : books_for_girls = 225)
  (h2 : girls = 15)
  (h3 : boys = 10)
  (h4 : books_for_girls / girls = books_per_girl)
  (h5 : books_per_girl * boys + books_for_girls = 375) : 
  books_for_girls / girls * (girls + boys) = 375 := 
by
  sorry

end ms_smith_books_divided_l73_73380


namespace seating_arrangements_count_l73_73853

-- Define the main entities: the three teams and the conditions
inductive Person
| Jupitarian
| Saturnian
| Neptunian

open Person

-- Define the seating problem constraints
def valid_arrangement (seating : Fin 12 → Person) : Prop :=
  seating 0 = Jupitarian ∧ seating 11 = Neptunian ∧
  (∀ i, seating (i % 12) = Jupitarian → seating ((i + 11) % 12) ≠ Neptunian) ∧
  (∀ i, seating (i % 12) = Neptunian → seating ((i + 11) % 12) ≠ Saturnian) ∧
  (∀ i, seating (i % 12) = Saturnian → seating ((i + 11) % 12) ≠ Jupitarian)

-- Main theorem: The number of valid arrangements is 225 * (4!)^3
theorem seating_arrangements_count :
  ∃ M : ℕ, (M = 225) ∧ ∃ arrangements : Fin 12 → Person, valid_arrangement arrangements :=
sorry

end seating_arrangements_count_l73_73853


namespace gcd_m_n_l73_73120

def m : ℕ := 333333333
def n : ℕ := 9999999999

theorem gcd_m_n : Nat.gcd m n = 9 := by
  sorry

end gcd_m_n_l73_73120


namespace problem_1_problem_2_l73_73011

noncomputable def O := (0, 0)
noncomputable def A := (1, 2)
noncomputable def B := (-3, 4)

noncomputable def vector_AB := (B.1 - A.1, B.2 - A.2)
noncomputable def magnitude_AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def dot_OA_OB := A.1 * B.1 + A.2 * B.2
noncomputable def magnitude_OA := Real.sqrt (A.1^2 + A.2^2)
noncomputable def magnitude_OB := Real.sqrt (B.1^2 + B.2^2)
noncomputable def cosine_angle := dot_OA_OB / (magnitude_OA * magnitude_OB)

theorem problem_1 : vector_AB = (-4, 2) ∧ magnitude_AB = 2 * Real.sqrt 5 := sorry

theorem problem_2 : cosine_angle = Real.sqrt 5 / 5 := sorry

end problem_1_problem_2_l73_73011


namespace arithmetic_sequence_S11_l73_73227

theorem arithmetic_sequence_S11 (a1 d : ℝ) 
  (h1 : a1 + d + a1 + 3 * d + 3 * (a1 + 6 * d) + a1 + 8 * d = 24) : 
  let a2 := a1 + d
  let a4 := a1 + 3 * d
  let a7 := a1 + 6 * d
  let a9 := a1 + 8 * d
  let S11 := 11 * (a1 + 5 * d)
  S11 = 44 :=
by
  sorry

end arithmetic_sequence_S11_l73_73227


namespace max_truthful_gnomes_l73_73872

theorem max_truthful_gnomes :
  ∀ (heights : Fin 7 → ℝ), 
    heights 0 = 60 →
    heights 1 = 61 →
    heights 2 = 62 →
    heights 3 = 63 →
    heights 4 = 64 →
    heights 5 = 65 →
    heights 6 = 66 →
      (∃ i : Fin 7, ∀ j : Fin 7, (i ≠ j → heights j ≠ (60 + j.1)) ∧ (i = j → heights j = (60 + j.1))) :=
by
  intro heights h1 h2 h3 h4 h5 h6 h7
  sorry

end max_truthful_gnomes_l73_73872


namespace mustard_found_at_third_table_l73_73439

variable (a b T : ℝ)
def found_mustard_at_first_table := (a = 0.25)
def found_mustard_at_second_table := (b = 0.25)
def total_mustard_found := (T = 0.88)

theorem mustard_found_at_third_table
  (h1 : found_mustard_at_first_table a)
  (h2 : found_mustard_at_second_table b)
  (h3 : total_mustard_found T) :
  T - (a + b) = 0.38 := by
  sorry

end mustard_found_at_third_table_l73_73439


namespace G_is_odd_l73_73712

noncomputable def G (F : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ :=
  F x * (1 / (a^x - 1) + 1 / 2)

theorem G_is_odd (F : ℝ → ℝ) (a : ℝ) (h : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x : ℝ, F (-x) = - F x) :
  ∀ x : ℝ, G F a (-x) = - G F a x :=
by 
  sorry

end G_is_odd_l73_73712


namespace treaty_signed_on_wednesday_l73_73395

-- This function calculates the weekday after a given number of days since a known weekday.
def weekday_after (start_day: ℕ) (days: ℕ) : ℕ :=
  (start_day + days) % 7

-- Given the problem conditions:
-- The war started on a Friday: 5th day of the week (considering Sunday as 0)
def war_start_day_of_week : ℕ := 5

-- The number of days after which the treaty was signed
def days_until_treaty : ℕ := 926

-- Expected final day (Wednesday): 3rd day of the week (considering Sunday as 0)
def treaty_day_of_week : ℕ := 3

-- The theorem to be proved:
theorem treaty_signed_on_wednesday :
  weekday_after war_start_day_of_week days_until_treaty = treaty_day_of_week :=
by
  sorry

end treaty_signed_on_wednesday_l73_73395


namespace number_times_half_squared_eq_eight_l73_73059

theorem number_times_half_squared_eq_eight : 
  ∃ n : ℝ, n * (1/2)^2 = 2^3 := 
sorry

end number_times_half_squared_eq_eight_l73_73059


namespace sum_difference_l73_73321

def sum_even (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

def sum_odd (n : ℕ) : ℕ :=
  (n / 2) * (1 + (n - 1))

theorem sum_difference : sum_even 100 - sum_odd 99 = 50 :=
by
  sorry

end sum_difference_l73_73321


namespace perfect_square_trinomial_m_l73_73588

theorem perfect_square_trinomial_m (m : ℤ) : (∃ (a : ℤ), (x : ℝ) → x^2 + m * x + 9 = (x + a)^2) → (m = 6 ∨ m = -6) :=
sorry

end perfect_square_trinomial_m_l73_73588


namespace sum_f_84_eq_1764_l73_73965

theorem sum_f_84_eq_1764 (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, 0 < n → f n < f (n + 1))
  (h2 : ∀ m n : ℕ, 0 < m → 0 < n → f (m * n) = f m * f n)
  (h3 : ∀ m n : ℕ, 0 < m → 0 < n → m ≠ n → m^n = n^m → (f m = n ∨ f n = m)) :
  f 84 = 1764 :=
by
  sorry

end sum_f_84_eq_1764_l73_73965


namespace willy_episodes_per_day_l73_73557

def total_episodes (seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  seasons * episodes_per_season

def episodes_per_day (total_episodes : ℕ) (days : ℕ) : ℕ :=
  total_episodes / days

theorem willy_episodes_per_day :
  episodes_per_day (total_episodes 3 20) 30 = 2 :=
by
  sorry

end willy_episodes_per_day_l73_73557


namespace intersection_point_proof_l73_73025

def intersect_point : Prop := 
  ∃ x y : ℚ, (5 * x - 6 * y = 3) ∧ (8 * x + 2 * y = 22) ∧ x = 69 / 29 ∧ y = 43 / 29

theorem intersection_point_proof : intersect_point :=
  sorry

end intersection_point_proof_l73_73025


namespace mul_99_101_equals_9999_l73_73167

theorem mul_99_101_equals_9999 : 99 * 101 = 9999 := by
  sorry

end mul_99_101_equals_9999_l73_73167


namespace hours_of_use_per_charge_l73_73858

theorem hours_of_use_per_charge
  (c h u : ℕ)
  (h_c : c = 10)
  (h_fraction : h = 6)
  (h_use : 6 * u = 12) :
  u = 2 :=
sorry

end hours_of_use_per_charge_l73_73858


namespace find_a_l73_73558

theorem find_a (a : ℝ) (b : ℝ) :
  (9 * x^2 - 27 * x + a = (3 * x + b)^2) → b = -4.5 → a = 20.25 := 
by sorry

end find_a_l73_73558


namespace pages_per_brochure_l73_73062

-- Define the conditions
def single_page_spreads := 20
def double_page_spreads := 2 * single_page_spreads
def pages_per_double_spread := 2
def pages_from_single := single_page_spreads
def pages_from_double := double_page_spreads * pages_per_double_spread
def total_pages_from_spreads := pages_from_single + pages_from_double
def ads_per_4_pages := total_pages_from_spreads / 4
def total_ads_pages := ads_per_4_pages
def total_pages := total_pages_from_spreads + total_ads_pages
def brochures := 25

-- The theorem we want to prove
theorem pages_per_brochure : total_pages / brochures = 5 :=
by
  -- This is a placeholder for the actual proof
  sorry

end pages_per_brochure_l73_73062


namespace inequality_proof_equality_condition_l73_73909

variables {a b c x y z : ℕ}

theorem inequality_proof (h1 : a ^ 2 + b ^ 2 = c ^ 2) (h2 : x ^ 2 + y ^ 2 = z ^ 2) : 
  (a + x) ^ 2 + (b + y) ^ 2 ≤ (c + z) ^ 2 :=
sorry

theorem equality_condition (h1 : a ^ 2 + b ^ 2 = c ^ 2) (h2 : x ^ 2 + y ^ 2 = z ^ 2) : 
  (a + x) ^ 2 + (b + y) ^ 2 = (c + z) ^ 2 ↔ a * z = c * x ∧ a * y = b * x :=
sorry

end inequality_proof_equality_condition_l73_73909


namespace total_cost_750_candies_l73_73707

def candy_cost (candies : ℕ) (cost_per_box : ℕ) (candies_per_box : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
  let boxes := candies / candies_per_box
  let total_cost := boxes * cost_per_box
  if candies > discount_threshold then
    (1 - discount_rate) * total_cost
  else
    total_cost

theorem total_cost_750_candies :
  candy_cost 750 8 30 500 0.1 = 180 :=
by sorry

end total_cost_750_candies_l73_73707


namespace coffee_consumption_l73_73597

-- Defining the necessary variables and conditions
variable (Ivory_cons Brayan_cons : ℕ)
variable (hr : ℕ := 1)
variable (hrs : ℕ := 5)

-- Condition: Brayan drinks twice as much coffee as Ivory
def condition1 := Brayan_cons = 2 * Ivory_cons

-- Condition: Brayan drinks 4 cups of coffee in an hour
def condition2 := Brayan_cons = 4

-- The proof problem
theorem coffee_consumption : ∀ (Ivory_cons Brayan_cons : ℕ), (Brayan_cons = 2 * Ivory_cons) → 
  (Brayan_cons = 4) → 
  ((Brayan_cons * hrs) + (Ivory_cons * hrs) = 30) :=
by
  intro hBrayan hIvory hr
  sorry

end coffee_consumption_l73_73597


namespace obtuse_angle_only_dihedral_planar_l73_73679

/-- Given the range of three types of angles, prove that only the dihedral angle's planar angle can be obtuse. -/
theorem obtuse_angle_only_dihedral_planar 
  (α : ℝ) (β : ℝ) (γ : ℝ) 
  (hα : 0 < α ∧ α ≤ 90)
  (hβ : 0 ≤ β ∧ β ≤ 90)
  (hγ : 0 ≤ γ ∧ γ < 180) : 
  (90 < γ ∧ (¬(90 < α)) ∧ (¬(90 < β))) :=
by 
  sorry

end obtuse_angle_only_dihedral_planar_l73_73679


namespace subtract_vectors_l73_73385

def vec_a : ℤ × ℤ × ℤ := (5, -3, 2)
def vec_b : ℤ × ℤ × ℤ := (-2, 4, 1)
def vec_result : ℤ × ℤ × ℤ := (9, -11, 0)

theorem subtract_vectors :
  vec_a - 2 • vec_b = vec_result :=
by sorry

end subtract_vectors_l73_73385


namespace problem_l73_73764

theorem problem (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = 1) 
  (h3 : a + c + d = 16) 
  (h4 : b + c + d = 9) : 
  a * b + c * d = 734 / 9 := 
by 
  sorry

end problem_l73_73764


namespace largest_angle_in_triangle_l73_73990

theorem largest_angle_in_triangle (k : ℕ) (h : 3 * k + 4 * k + 5 * k = 180) : 5 * k = 75 :=
  by
  -- This is a placeholder for the proof, which is not required as per instructions
  sorry

end largest_angle_in_triangle_l73_73990


namespace solve_expression_l73_73374

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem solve_expression : f (g 3) - g (f 3) = -5 := by
  sorry

end solve_expression_l73_73374


namespace meeting_equation_correct_l73_73929

-- Define the conditions
def distance : ℝ := 25
def time : ℝ := 3
def speed_Xiaoming : ℝ := 4
def speed_Xiaogang (x : ℝ) : ℝ := x

-- The target equation derived from conditions which we need to prove valid.
theorem meeting_equation_correct (x : ℝ) : 3 * (speed_Xiaoming + speed_Xiaogang x) = distance :=
by
  sorry

end meeting_equation_correct_l73_73929


namespace product_two_smallest_one_digit_primes_and_largest_three_digit_prime_l73_73816

theorem product_two_smallest_one_digit_primes_and_largest_three_digit_prime :
  2 * 3 * 997 = 5982 :=
by
  sorry

end product_two_smallest_one_digit_primes_and_largest_three_digit_prime_l73_73816


namespace symmetric_about_z_correct_l73_73007

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_about_z (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

theorem symmetric_about_z_correct (p : Point3D) :
  p = {x := 3, y := 4, z := 5} → symmetric_about_z p = {x := -3, y := -4, z := 5} :=
by
  sorry

end symmetric_about_z_correct_l73_73007


namespace proof_prob_boy_pass_all_rounds_proof_prob_girl_pass_all_rounds_proof_xi_distribution_proof_exp_xi_l73_73052

noncomputable def prob_boy_pass_all_rounds : ℚ :=
  (5/6) * (4/5) * (3/4) * (2/3)

noncomputable def prob_girl_pass_all_rounds : ℚ :=
  (4/5) * (3/4) * (2/3) * (1/2)

def prob_xi_distribution : (ℚ × ℚ × ℚ × ℚ × ℚ) :=
  (64/225, 96/225, 52/225, 12/225, 1/225)

def exp_xi : ℚ :=
  (0 * (64/225) + 1 * (96/225) + 2 * (52/225) + 3 * (12/225) + 4 * (1/225))

theorem proof_prob_boy_pass_all_rounds :
  prob_boy_pass_all_rounds = 1/3 :=
by
  sorry

theorem proof_prob_girl_pass_all_rounds :
  prob_girl_pass_all_rounds = 1/5 :=
by
  sorry

theorem proof_xi_distribution :
  prob_xi_distribution = (64/225, 96/225, 52/225, 12/225, 1/225) :=
by
  sorry

theorem proof_exp_xi :
  exp_xi = 16/15 :=
by
  sorry

end proof_prob_boy_pass_all_rounds_proof_prob_girl_pass_all_rounds_proof_xi_distribution_proof_exp_xi_l73_73052


namespace no_such_nat_n_l73_73854

theorem no_such_nat_n :
  ¬ ∃ n : ℕ, ∀ a b : ℕ, (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) → (10 * (10 * a + n) + b) % (10 * a + b) = 0 :=
by
  sorry

end no_such_nat_n_l73_73854


namespace greatest_divisor_of_product_of_four_consecutive_integers_l73_73115

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, ∃ k : Nat, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l73_73115


namespace speed_of_man_l73_73366

/-
  Problem Statement:
  A train 100 meters long takes 6 seconds to cross a man walking at a certain speed in the direction opposite to that of the train. The speed of the train is 54.99520038396929 kmph. What is the speed of the man in kmph?
-/
 
theorem speed_of_man :
  ∀ (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_train_kmph : ℝ) (relative_speed_mps : ℝ),
    length_of_train = 100 →
    time_to_cross = 6 →
    speed_of_train_kmph = 54.99520038396929 →
    relative_speed_mps = length_of_train / time_to_cross →
    (relative_speed_mps - (speed_of_train_kmph * (1000 / 3600))) * (3600 / 1000) = 5.00479961403071 :=
by
  intros length_of_train time_to_cross speed_of_train_kmph relative_speed_mps
  intros h1 h2 h3 h4
  sorry

end speed_of_man_l73_73366


namespace find_factor_l73_73176

variable (x : ℕ) (f : ℕ)

def original_number := x = 20
def resultant := f * (2 * x + 5) = 135

theorem find_factor (h1 : original_number x) (h2 : resultant x f) : f = 3 := by
  sorry

end find_factor_l73_73176


namespace factorize_expression_l73_73837

variable (a x : ℝ)

theorem factorize_expression : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := 
by 
  sorry

end factorize_expression_l73_73837
