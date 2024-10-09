import Mathlib

namespace tree_ratio_l2145_214585

theorem tree_ratio (native_trees : ℕ) (total_planted : ℕ) (M : ℕ) 
  (h1 : native_trees = 30) 
  (h2 : total_planted = 80) 
  (h3 : total_planted = M + M / 3) :
  (native_trees + M) / native_trees = 3 :=
sorry

end tree_ratio_l2145_214585


namespace remainder_div_180_l2145_214527

theorem remainder_div_180 {j : ℕ} (h1 : 0 < j) (h2 : 120 % (j^2) = 12) : 180 % j = 0 :=
by
  sorry

end remainder_div_180_l2145_214527


namespace find_leak_rate_l2145_214559

-- Conditions in Lean 4
def pool_capacity : ℝ := 60
def hose_rate : ℝ := 1.6
def fill_time : ℝ := 40

-- Define the leak rate calculation
def leak_rate (L : ℝ) : Prop :=
  pool_capacity = (hose_rate - L) * fill_time

-- The main theorem we want to prove
theorem find_leak_rate : ∃ L, leak_rate L ∧ L = 0.1 := by
  sorry

end find_leak_rate_l2145_214559


namespace Marcia_wardrobe_cost_l2145_214532

-- Definitions from the problem
def skirt_price : ℝ := 20
def blouse_price : ℝ := 15
def pant_price : ℝ := 30

def num_skirts : ℕ := 3
def num_blouses : ℕ := 5
def num_pants : ℕ := 2

-- The main theorem statement
theorem Marcia_wardrobe_cost :
  (num_skirts * skirt_price) + (num_blouses * blouse_price) + (pant_price + (pant_price / 2)) = 180 :=
by
  sorry

end Marcia_wardrobe_cost_l2145_214532


namespace initial_apples_l2145_214514

-- Define the number of initial fruits
def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def fruits_given : ℕ := 40
def fruits_left : ℕ := 15

-- Define the equation for the initial number of fruits
def initial_total_fruits (A : ℕ) : Prop :=
  initial_plums + initial_guavas + A = fruits_left + fruits_given

-- Define the proof problem to find the number of apples
theorem initial_apples : ∃ A : ℕ, initial_total_fruits A ∧ A = 21 :=
  by
    sorry

end initial_apples_l2145_214514


namespace birth_year_1957_l2145_214537

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem birth_year_1957 (y x : ℕ) (h : y = 2023) (h1 : sum_of_digits x = y - x) : x = 1957 :=
by
  sorry

end birth_year_1957_l2145_214537


namespace Brady_average_hours_l2145_214567

-- Definitions based on conditions
def hours_per_day_April : ℕ := 6
def hours_per_day_June : ℕ := 5
def hours_per_day_September : ℕ := 8
def days_in_April : ℕ := 30
def days_in_June : ℕ := 30
def days_in_September : ℕ := 30

-- Definition to prove
def average_hours_per_month : ℕ := 190

-- Theorem statement
theorem Brady_average_hours :
  (hours_per_day_April * days_in_April + hours_per_day_June * days_in_June + hours_per_day_September * days_in_September) / 3 = average_hours_per_month :=
sorry

end Brady_average_hours_l2145_214567


namespace evaluate_expression_l2145_214589

theorem evaluate_expression (a b c : ℚ) (h1 : c = b - 8) (h2 : b = a + 3) (h3 : a = 2) 
  (h4 : a + 1 ≠ 0) (h5 : b - 3 ≠ 0) (h6 : c + 5 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 7) / (c + 5) = 20 / 3 := by
  sorry

end evaluate_expression_l2145_214589


namespace schedule_courses_l2145_214595

-- Define the number of courses and periods
def num_courses : Nat := 4
def num_periods : Nat := 8

-- Define the total number of ways to schedule courses without restrictions
def unrestricted_schedules : Nat := Nat.choose num_periods num_courses * Nat.factorial num_courses

-- Define the number of invalid schedules using PIE (approximate value given in problem)
def invalid_schedules : Nat := 1008 + 180 + 120

-- Define the number of valid schedules
def valid_schedules : Nat := unrestricted_schedules - invalid_schedules

theorem schedule_courses : valid_schedules = 372 := sorry

end schedule_courses_l2145_214595


namespace average_pastries_per_day_l2145_214548

-- Conditions
def pastries_on_monday := 2

def pastries_on_day (n : ℕ) : ℕ :=
  pastries_on_monday + n

def total_pastries_in_week : ℕ :=
  List.sum (List.map pastries_on_day (List.range 7))

def number_of_days_in_week : ℕ := 7

-- Theorem to prove
theorem average_pastries_per_day : (total_pastries_in_week / number_of_days_in_week) = 5 :=
by
  sorry

end average_pastries_per_day_l2145_214548


namespace probability_odd_product_l2145_214520

-- Given conditions
def numbers : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the proof problem
theorem probability_odd_product (h: choices = 15 ∧ odd_choices = 3) :
  (odd_choices : ℚ) / choices = 1 / 5 :=
by sorry

end probability_odd_product_l2145_214520


namespace john_has_500_dollars_l2145_214536

-- Define the initial amount and the condition
def initial_amount : ℝ := 1600
def condition (spent : ℝ) : Prop := (1600 - spent) = (spent - 600)

-- The final amount of money John still has
def final_amount (spent : ℝ) : ℝ := initial_amount - spent

-- The main theorem statement
theorem john_has_500_dollars : ∃ (spent : ℝ), condition spent ∧ final_amount spent = 500 :=
by
  sorry

end john_has_500_dollars_l2145_214536


namespace jason_two_weeks_eggs_l2145_214508

-- Definitions of given conditions
def eggs_per_omelet := 3
def days_per_week := 7
def weeks := 2

-- Statement to prove
theorem jason_two_weeks_eggs : (eggs_per_omelet * (days_per_week * weeks)) = 42 := by
  sorry

end jason_two_weeks_eggs_l2145_214508


namespace arithmetic_sequence_common_difference_l2145_214593

theorem arithmetic_sequence_common_difference {a : ℕ → ℝ} (h₁ : a 1 = 2) (h₂ : a 2 + a 4 = a 6) : ∃ d : ℝ, d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l2145_214593


namespace triangle_inequality_harmonic_mean_l2145_214561

theorem triangle_inequality_harmonic_mean (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ DP DQ : ℝ, DP + DQ ≤ (2 * a * b) / (a + b) :=
by
  sorry

end triangle_inequality_harmonic_mean_l2145_214561


namespace arithmetic_sequence_a12_l2145_214522

def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) (a1 d : ℤ) (h : arithmetic_sequence a a1 d) :
  a 11 = 23 :=
by
  -- condtions
  let a1_val := 1
  let d_val := 2
  have ha1 : a1 = a1_val := sorry
  have hd : d = d_val := sorry
  
  -- proof
  rw [ha1, hd] at h
  
  sorry

end arithmetic_sequence_a12_l2145_214522


namespace a_81_eq_640_l2145_214565

noncomputable def sequence_a (n : ℕ) : ℕ :=
if n = 0 then 0 -- auxiliary value since sequence begins from n=1
else if n = 1 then 1
else (2 * n - 1) ^ 2 - (2 * n - 3) ^ 2

theorem a_81_eq_640 : sequence_a 81 = 640 :=
by
  sorry

end a_81_eq_640_l2145_214565


namespace find_a1_l2145_214570

theorem find_a1 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 1 / (1 - a n))
  (h2 : a 2 = 2)
  : a 1 = 1 / 2 :=
sorry

end find_a1_l2145_214570


namespace no_prime_quadruple_l2145_214529

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_quadruple 
    (a b c d : ℕ)
    (ha_prime : is_prime a) 
    (hb_prime : is_prime b)
    (hc_prime : is_prime c)
    (hd_prime : is_prime d)
    (h_order : a < b ∧ b < c ∧ c < d) :
    (1 / a + 1 / d ≠ 1 / b + 1 / c) := 
by 
  sorry

end no_prime_quadruple_l2145_214529


namespace airplane_speeds_l2145_214524

theorem airplane_speeds (v : ℝ) 
  (h1 : 2.5 * v + 2.5 * 250 = 1625) : 
  v = 400 := 
sorry

end airplane_speeds_l2145_214524


namespace problem_l2145_214510

theorem problem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 := 
sorry

end problem_l2145_214510


namespace find_y_l2145_214523

theorem find_y (x k m y : ℤ) 
  (h1 : x = 82 * k + 5) 
  (h2 : x + y = 41 * m + 12) : 
  y = 7 := 
sorry

end find_y_l2145_214523


namespace length_rest_of_body_l2145_214500

theorem length_rest_of_body (height legs head arms rest_of_body : ℝ) 
  (hlegs : legs = (1/3) * height)
  (hhead : head = (1/4) * height)
  (harms : arms = (1/5) * height)
  (htotal : height = 180)
  (hr: rest_of_body = height - (legs + head + arms)) : 
  rest_of_body = 39 :=
by
  -- proof is not required
  sorry

end length_rest_of_body_l2145_214500


namespace max_min_y_l2145_214592

def g (t : ℝ) : ℝ := 80 - 2 * t

def f (t : ℝ) : ℝ := 20 - |t - 10|

def y (t : ℝ) : ℝ := g t * f t

theorem max_min_y (t : ℝ) (h : 0 ≤ t ∧ t ≤ 20) :
  (y t = 1200 → t = 10) ∧ (y t = 400 → t = 20) :=
by
  sorry

end max_min_y_l2145_214592


namespace verify_incorrect_option_l2145_214555

variable (a : ℕ → ℝ) -- The sequence a_n
variable (S : ℕ → ℝ) -- The sum of the first n terms S_n

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def condition_1 (S : ℕ → ℝ) : Prop := S 5 < S 6

def condition_2 (S : ℕ → ℝ) : Prop := S 6 = S 7 ∧ S 7 > S 8

theorem verify_incorrect_option (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond1 : condition_1 S)
  (h_cond2 : condition_2 S) :
  S 9 ≤ S 5 :=
sorry

end verify_incorrect_option_l2145_214555


namespace parabola_intersects_once_compare_y_values_l2145_214599

noncomputable def parabola (x : ℝ) (m : ℝ) : ℝ := -2 * x^2 + 4 * x + m

theorem parabola_intersects_once (m : ℝ) : 
  ∃ x, parabola x m = 0 ↔ m = -2 := 
by 
  sorry

theorem compare_y_values (x1 x2 m : ℝ) (h1 : x1 > x2) (h2 : x2 > 2) : 
  parabola x1 m < parabola x2 m :=
by 
  sorry

end parabola_intersects_once_compare_y_values_l2145_214599


namespace min_value_inverse_sum_l2145_214541

theorem min_value_inverse_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) :
  (1 / x + 1 / y + 1 / z) ≥ 9 :=
  sorry

end min_value_inverse_sum_l2145_214541


namespace quadratic_equation_with_means_l2145_214525

theorem quadratic_equation_with_means (α β : ℝ) 
  (h_am : (α + β) / 2 = 8) 
  (h_gm : Real.sqrt (α * β) = 15) : 
  (Polynomial.X^2 - Polynomial.C (α + β) * Polynomial.X + Polynomial.C (α * β) = 0) := 
by
  have h1 : α + β = 16 := by linarith
  have h2 : α * β = 225 := by sorry
  rw [h1, h2]
  sorry

end quadratic_equation_with_means_l2145_214525


namespace trigonometric_identity_1_l2145_214582

theorem trigonometric_identity_1 :
  ( (Real.sqrt 3 * Real.sin (-1200 * Real.pi / 180)) / (Real.tan (11 * Real.pi / 3)) 
  - Real.cos (585 * Real.pi / 180) * Real.tan (-37 * Real.pi / 4) = (Real.sqrt 3 / 2) - (Real.sqrt 2 / 2) ) :=
by
  sorry

end trigonometric_identity_1_l2145_214582


namespace spherical_coordinate_conversion_l2145_214563

theorem spherical_coordinate_conversion (ρ θ φ : ℝ) 
  (h_ρ : ρ > 0) 
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h_φ : 0 ≤ φ): 
  (ρ, θ, φ - 2 * Real.pi * ⌊φ / (2 * Real.pi)⌋) = (5, 3 * Real.pi / 4, Real.pi / 4) :=
  by 
  sorry

end spherical_coordinate_conversion_l2145_214563


namespace P_union_Q_eq_Q_l2145_214516

noncomputable def P : Set ℝ := {x : ℝ | x > 1}
noncomputable def Q : Set ℝ := {x : ℝ | x^2 - x > 0}

theorem P_union_Q_eq_Q : P ∪ Q = Q := by
  sorry

end P_union_Q_eq_Q_l2145_214516


namespace eggs_distributed_equally_l2145_214533

-- Define the total number of eggs
def total_eggs : ℕ := 8484

-- Define the number of baskets
def baskets : ℕ := 303

-- Define the expected number of eggs per basket
def eggs_per_basket : ℕ := 28

-- State the theorem
theorem eggs_distributed_equally :
  total_eggs / baskets = eggs_per_basket := sorry

end eggs_distributed_equally_l2145_214533


namespace Jessica_paid_1000_for_rent_each_month_last_year_l2145_214504

/--
Jessica paid $200 for food each month last year.
Jessica paid $100 for car insurance each month last year.
This year her rent goes up by 30%.
This year food costs increase by 50%.
This year the cost of her car insurance triples.
Jessica pays $7200 more for her expenses over the whole year compared to last year.
-/
theorem Jessica_paid_1000_for_rent_each_month_last_year
  (R : ℝ) -- monthly rent last year
  (h1 : 12 * (0.30 * R + 100 + 200) = 7200) :
  R = 1000 :=
sorry

end Jessica_paid_1000_for_rent_each_month_last_year_l2145_214504


namespace ab_sum_eq_2_l2145_214573

theorem ab_sum_eq_2 (a b : ℝ) (M : Set ℝ) (N : Set ℝ) (f : ℝ → ℝ) 
  (hM : M = {b / a, 1})
  (hN : N = {a, 0})
  (hf : ∀ x ∈ M, f x ∈ N)
  (f_def : ∀ x, f x = 2 * x) :
  a + b = 2 :=
by
  -- proof goes here.
  sorry

end ab_sum_eq_2_l2145_214573


namespace remaining_budget_after_purchases_l2145_214540

theorem remaining_budget_after_purchases :
  let budget := 80
  let fried_chicken_cost := 12
  let beef_cost_per_pound := 3
  let beef_quantity := 4.5
  let soup_cost_per_can := 2
  let soup_quantity := 3
  let milk_original_price := 4
  let milk_discount := 0.10
  let beef_cost := beef_quantity * beef_cost_per_pound
  let paid_soup_quantity := soup_quantity / 2
  let milk_discounted_price := milk_original_price * (1 - milk_discount)
  let total_cost := fried_chicken_cost + beef_cost + (paid_soup_quantity * soup_cost_per_can) + milk_discounted_price
  let remaining_budget := budget - total_cost
  remaining_budget = 47.90 :=
by
  sorry

end remaining_budget_after_purchases_l2145_214540


namespace banknotes_combination_l2145_214553

theorem banknotes_combination (a b c d : ℕ) (h : a + b + c + d = 10) (h_val : 2000 * a + 1000 * b + 500 * c + 200 * d = 5000) :
  (a = 0 ∧ b = 0 ∧ c = 10 ∧ d = 0) ∨ 
  (a = 1 ∧ b = 0 ∧ c = 4 ∧ d = 5) ∨ 
  (a = 0 ∧ b = 3 ∧ c = 2 ∧ d = 5) :=
by
  sorry

end banknotes_combination_l2145_214553


namespace reciprocal_sum_greater_l2145_214569

theorem reciprocal_sum_greater (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
    (1 / a + 1 / b) > 1 / (a + b) :=
sorry

end reciprocal_sum_greater_l2145_214569


namespace jake_has_3_peaches_l2145_214597

-- Define the number of peaches Steven has.
def steven_peaches : ℕ := 13

-- Define the number of peaches Jake has based on the condition.
def jake_peaches (P_S : ℕ) : ℕ := P_S - 10

-- The theorem that states Jake has 3 peaches.
theorem jake_has_3_peaches : jake_peaches steven_peaches = 3 := sorry

end jake_has_3_peaches_l2145_214597


namespace negation_of_exists_l2145_214521

theorem negation_of_exists (h : ¬ ∃ x : ℝ, x^2 + x + 1 < 0) : ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l2145_214521


namespace find_f2_l2145_214515

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f (f x) = (x ^ 2 - x) / 2 * f x + 2 - x

theorem find_f2 : f 2 = 2 :=
by
  sorry

end find_f2_l2145_214515


namespace p_sufficient_but_not_necessary_for_q_l2145_214511

def condition_p (x : ℝ) : Prop := abs (x - 1) < 2
def condition_q (x : ℝ) : Prop := x^2 - 5 * x - 6 < 0

theorem p_sufficient_but_not_necessary_for_q : 
  (∀ x, condition_p x → condition_q x) ∧ 
  ¬ (∀ x, condition_q x → condition_p x) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l2145_214511


namespace expression_value_l2145_214588

open Real

theorem expression_value :
  3 + sqrt 3 + 1 / (3 + sqrt 3) + 1 / (sqrt 3 - 3) = 3 + 2 * sqrt 3 / 3 := 
sorry

end expression_value_l2145_214588


namespace find_geometric_ratio_l2145_214545

-- Definitions for the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 1 - a 0)

def geometric_sequence (a1 a3 a4 : ℝ) (q : ℝ) : Prop :=
  a3 * a3 = a1 * a4 ∧ a3 = a1 * q ∧ a4 = a3 * q

-- Definition for the proof statement
theorem find_geometric_ratio (a : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hnz : ∀ n, a n ≠ 0)
  (hq : ∃ (q : ℝ), geometric_sequence (a 0) (a 2) (a 3) q) :
  ∃ q, q = 1 ∨ q = 1 / 2 := sorry

end find_geometric_ratio_l2145_214545


namespace exists_colored_triangle_l2145_214580

structure Point := (x : ℝ) (y : ℝ)
inductive Color
| red
| blue

def collinear (a b c : Point) : Prop :=
  (b.y - a.y) * (c.x - a.x) = (c.y - a.y) * (b.x - a.x)
  
def same_color_triangle_exists (S : Finset Point) (color : Point → Color) : Prop :=
  ∃ (A B C : Point), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧
                    (color A = color B ∧ color B = color C) ∧
                    ¬ collinear A B C ∧
                    (∃ (X Y Z : Point), 
                      ((X ∈ S ∧ color X ≠ color A ∧ (X ≠ A ∧ X ≠ B ∧ X ≠ C)) ∧ 
                       (Y ∈ S ∧ color Y ≠ color A ∧ (Y ≠ A ∧ Y ≠ B ∧ Y ≠ C)) ∧
                       (Z ∈ S ∧ color Z ≠ color A ∧ (Z ≠ A ∧ Z ≠ B ∧ Z ≠ C)) → 
                       False))

theorem exists_colored_triangle 
  (S : Finset Point) (h1 : 5 ≤ S.card) (color : Point → Color) 
  (h2 : ∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → (color A = color B ∧ color B = color C) → ¬ collinear A B C) 
  : same_color_triangle_exists S color :=
sorry

end exists_colored_triangle_l2145_214580


namespace solve_x_squared_plus_15_eq_y_squared_l2145_214587

theorem solve_x_squared_plus_15_eq_y_squared (x y : ℤ) : x^2 + 15 = y^2 → x = 7 ∨ x = -7 ∨ x = 1 ∨ x = -1 := by
  sorry

end solve_x_squared_plus_15_eq_y_squared_l2145_214587


namespace system_of_equations_solution_l2145_214506

theorem system_of_equations_solution (x y : ℝ) (h1 : |y - x| - (|x| / x) + 1 = 0) (h2 : |2 * x - y| + |x + y - 1| + |x - y| + y - 1 = 0) (hx : x ≠ 0) :
  (0 < x ∧ x ≤ 0.5 ∧ y = x) :=
by
  sorry

end system_of_equations_solution_l2145_214506


namespace ratio_of_socks_l2145_214572

variable (b : ℕ)            -- the number of pairs of blue socks
variable (x : ℝ)            -- the price of blue socks per pair

def original_cost : ℝ := 5 * 3 * x + b * x
def interchanged_cost : ℝ := b * 3 * x + 5 * x

theorem ratio_of_socks :
  (5 : ℝ) / b = 5 / 14 :=
by
  sorry

end ratio_of_socks_l2145_214572


namespace sum_of_reciprocals_l2145_214534

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 24) : 
  (1 / x + 1 / y = 1 / 2) :=
by 
  sorry

end sum_of_reciprocals_l2145_214534


namespace compute_value_l2145_214544

theorem compute_value : 302^2 - 298^2 = 2400 :=
by
  sorry

end compute_value_l2145_214544


namespace water_to_add_l2145_214528

theorem water_to_add (x : ℚ) (alcohol water : ℚ) (ratio : ℚ) :
  alcohol = 4 → water = 4 →
  (3 : ℚ) / (3 + 5) = (3 : ℚ) / 8 →
  (5 : ℚ) / (3 + 5) = (5 : ℚ) / 8 →
  ratio = 5 / 8 →
  (4 + x) / (8 + x) = ratio →
  x = 8 / 3 :=
by
  intros
  sorry

end water_to_add_l2145_214528


namespace smallest_fraction_numerator_l2145_214584

theorem smallest_fraction_numerator (a b : ℕ) (h1 : 10 ≤ a) (h2 : a ≤ 99) (h3 : 10 ≤ b) (h4 : b ≤ 99) (h5 : 9 * a > 4 * b) (smallest : ∀ c d, 10 ≤ c → c ≤ 99 → 10 ≤ d → d ≤ 99 → 9 * c > 4 * d → (a * d ≤ b * c) → a * d ≤ 41 * 92) :
  a = 41 :=
by
  sorry

end smallest_fraction_numerator_l2145_214584


namespace Kelly_needs_to_give_away_l2145_214557

variable (n k : Nat)

theorem Kelly_needs_to_give_away (h_n : n = 20) (h_k : k = 12) : n - k = 8 := 
by
  sorry

end Kelly_needs_to_give_away_l2145_214557


namespace total_amount_spent_l2145_214531

variables (D B : ℝ)

-- Conditions
def condition1 : Prop := B = 1.5 * D
def condition2 : Prop := D = B - 15

-- Question: Prove that the total amount they spent together is 75.00
theorem total_amount_spent (h1 : condition1 D B) (h2 : condition2 D B) : B + D = 75 :=
sorry

end total_amount_spent_l2145_214531


namespace max_value_of_2a_plus_b_l2145_214579

variable (a b : ℝ)

def cond1 := 4 * a + 3 * b ≤ 10
def cond2 := 3 * a + 5 * b ≤ 11

theorem max_value_of_2a_plus_b : 
  cond1 a b → 
  cond2 a b → 
  2 * a + b ≤ 48 / 11 := 
by 
  sorry

end max_value_of_2a_plus_b_l2145_214579


namespace min_value_of_expression_l2145_214583

theorem min_value_of_expression (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 3) : 
  ∃ k : ℝ, k = 4 + 2 * Real.sqrt 3 ∧ ∀ z, (z = (1 / (x - 1) + 3 / (y - 1))) → z ≥ k :=
sorry

end min_value_of_expression_l2145_214583


namespace findLastNames_l2145_214591

noncomputable def peachProblem : Prop :=
  ∃ (a b c d : ℕ),
    2 * a + 3 * b + 4 * c + 5 * d = 32 ∧
    a + b + c + d = 10 ∧
    (a = 3 ∧ b = 4 ∧ c = 1 ∧ d = 2)

theorem findLastNames :
  peachProblem :=
sorry

end findLastNames_l2145_214591


namespace right_triangle_sides_l2145_214517

theorem right_triangle_sides (a b c : ℝ)
    (h1 : a + b + c = 30)
    (h2 : a^2 + b^2 = c^2)
    (h3 : ∃ r, a = (5 * r) / 2 ∧ a + b = 5 * r ∧ ∀ x y, x / y = 2 / 3) :
  a = 5 ∧ b = 12 ∧ c = 13 :=
sorry

end right_triangle_sides_l2145_214517


namespace find_integers_l2145_214513

theorem find_integers 
  (A k : ℕ) 
  (h_sum : A + A * k + A * k^2 = 93) 
  (h_product : A * (A * k) * (A * k^2) = 3375) : 
  (A, A * k, A * k^2) = (3, 15, 75) := 
by 
  sorry

end find_integers_l2145_214513


namespace daniel_original_noodles_l2145_214586

-- Define the total number of noodles Daniel had originally
def original_noodles : ℕ := 81

-- Define the remaining noodles after giving 1/3 to William
def remaining_noodles (n : ℕ) : ℕ := (2 * n) / 3

-- State the theorem
theorem daniel_original_noodles (n : ℕ) (h : remaining_noodles n = 54) : n = original_noodles := by sorry

end daniel_original_noodles_l2145_214586


namespace largest_n_l2145_214549

theorem largest_n {x y z n : ℕ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (n:ℤ)^2 = (x:ℤ)^2 + (y:ℤ)^2 + (z:ℤ)^2 + 2*(x:ℤ)*(y:ℤ) + 2*(y:ℤ)*(z:ℤ) + 2*(z:ℤ)*(x:ℤ) + 6*(x:ℤ) + 6*(y:ℤ) + 6*(z:ℤ) - 12
  → n = 13 :=
sorry

end largest_n_l2145_214549


namespace participants_with_exactly_five_problems_l2145_214566

theorem participants_with_exactly_five_problems (n : ℕ) 
  (p : Fin 6 → Fin 6 → ℕ)
  (h1 : ∀ i j : Fin 6, i ≠ j → p i j > 2 * n / 5)
  (h2 : ¬ ∃ i : Fin 6, ∀ j : Fin 6, j ≠ i → p i j = n)
  : ∃ k1 k2 : Fin n, k1 ≠ k2 ∧ (∀ i : Fin 6, (p i k1 = 5) ∧ (p i k2 = 5)) :=
sorry

end participants_with_exactly_five_problems_l2145_214566


namespace problem_statement_l2145_214576

theorem problem_statement (x y : ℝ) (h₁ : 2.5 * x = 0.75 * y) (h₂ : x = 20) : y = 200 / 3 := by
  sorry

end problem_statement_l2145_214576


namespace correct_answer_l2145_214505

noncomputable def original_number (y : ℝ) :=
  (y - 14) / 2 = 50

theorem correct_answer (y : ℝ) (h : original_number y) :
  (y - 5) / 7 = 15 :=
by
  sorry

end correct_answer_l2145_214505


namespace sum_of_fractions_l2145_214554

theorem sum_of_fractions :
  (1 / 3) + (1 / 2) + (-5 / 6) + (1 / 5) + (1 / 4) + (-9 / 20) + (-9 / 20) = -9 / 20 := 
by
  sorry

end sum_of_fractions_l2145_214554


namespace train_speed_l2145_214568

theorem train_speed (x : ℝ) (v : ℝ) 
  (h1 : (x / 50) + (2 * x / v) = 3 * x / 25) : v = 20 :=
by
  sorry

end train_speed_l2145_214568


namespace winning_percentage_votes_l2145_214526

theorem winning_percentage_votes (P : ℝ) (votes_total : ℝ) (majority_votes : ℝ) (winning_votes : ℝ) : 
  votes_total = 4500 → majority_votes = 900 → 
  winning_votes = (P / 100) * votes_total → 
  majority_votes = winning_votes - ((100 - P) / 100) * votes_total → P = 60 := 
by
  intros h_total h_majority h_winning_votes h_majority_eq
  sorry

end winning_percentage_votes_l2145_214526


namespace digit_agreement_l2145_214558

theorem digit_agreement (N : ℕ) (abcd : ℕ) (h1 : N % 10000 = abcd) (h2 : N ^ 2 % 10000 = abcd) (h3 : ∃ a b c d, abcd = a * 1000 + b * 100 + c * 10 + d ∧ a ≠ 0) : abcd / 10 = 937 := sorry

end digit_agreement_l2145_214558


namespace find_y_value_l2145_214598

theorem find_y_value :
  ∀ (y : ℝ), (dist (1, 3) (7, y) = 13) ∧ (y > 0) → y = 3 + Real.sqrt 133 :=
by
  sorry

end find_y_value_l2145_214598


namespace f_specification_l2145_214590

open Function

def f : ℕ → ℕ := sorry -- define function f here

axiom f_involution (n : ℕ) : f (f n) = n

axiom f_functional_property (n : ℕ) : f (f n + 1) = if n % 2 = 0 then n - 1 else n + 3

axiom f_bijective : Bijective f

axiom f_not_two (n : ℕ) : f (f n + 1) ≠ 2

axiom f_one_eq_two : f 1 = 2

theorem f_specification (n : ℕ) : 
  f n = if n % 2 = 1 then n + 1 else n - 1 :=
sorry

end f_specification_l2145_214590


namespace problem_expression_value_l2145_214556

variable (m n p q : ℝ)
variable (h1 : m + n = 0) (h2 : m / n = -1)
variable (h3 : p * q = 1) (h4 : m ≠ n)

theorem problem_expression_value : 
  (m + n) / m + 2 * p * q - m / n = 3 :=
by sorry

end problem_expression_value_l2145_214556


namespace geometric_sequence_common_ratio_l2145_214575

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : a 5 = 16)
  (h_pos : ∀ n : ℕ, 0 < a n) :
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l2145_214575


namespace complete_consoles_production_rate_l2145_214578

-- Define the production rates of each chip
def production_rate_A := 467
def production_rate_B := 413
def production_rate_C := 532
def production_rate_D := 356
def production_rate_E := 494

-- Define the maximum number of consoles that can be produced per day
def max_complete_consoles (A B C D E : ℕ) := min (min (min (min A B) C) D) E

-- Statement
theorem complete_consoles_production_rate :
  max_complete_consoles production_rate_A production_rate_B production_rate_C production_rate_D production_rate_E = 356 :=
by
  sorry

end complete_consoles_production_rate_l2145_214578


namespace coins_with_specific_probabilities_impossible_l2145_214535

theorem coins_with_specific_probabilities_impossible 
  (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (eq1 : (1 - p1) * (1 - p2) = p1 * p2) 
  (eq2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : 
  false :=
by
  sorry

end coins_with_specific_probabilities_impossible_l2145_214535


namespace rectangular_field_area_l2145_214546

theorem rectangular_field_area (L B : ℝ) (h1 : B = 0.6 * L) (h2 : 2 * L + 2 * B = 800) : L * B = 37500 :=
by
  -- Proof will go here
  sorry

end rectangular_field_area_l2145_214546


namespace largest_angle_of_pentagon_l2145_214538

-- Define the angles of the pentagon and the conditions on them.
def is_angle_of_pentagon (A B C D E : ℝ) :=
  A = 108 ∧ B = 72 ∧ C = D ∧ E = 3 * C ∧
  A + B + C + D + E = 540

-- Prove the largest angle in the pentagon is 216
theorem largest_angle_of_pentagon (A B C D E : ℝ) (h : is_angle_of_pentagon A B C D E) :
  max (max (max (max A B) C) D) E = 216 :=
by
  sorry

end largest_angle_of_pentagon_l2145_214538


namespace parameterization_theorem_l2145_214543

theorem parameterization_theorem (a b c d : ℝ) (h1 : b = 1) (h2 : d = -3) (h3 : a + b = 4) (h4 : c + d = 5) :
  a^2 + b^2 + c^2 + d^2 = 83 :=
by
  sorry

end parameterization_theorem_l2145_214543


namespace decimal_to_base7_l2145_214552

theorem decimal_to_base7 :
    ∃ k₀ k₁ k₂ k₃ k₄, 1987 = k₀ * 7^4 + k₁ * 7^3 + k₂ * 7^2 + k₃ * 7^1 + k₄ * 7^0 ∧
    k₀ = 0 ∧
    k₁ = 5 ∧
    k₂ = 3 ∧
    k₃ = 5 ∧
    k₄ = 6 :=
by
  sorry

end decimal_to_base7_l2145_214552


namespace trigonometric_comparison_l2145_214594

open Real

theorem trigonometric_comparison :
  let a := 2 * sin (1 / 2)
  let b := 3 * sin (1 / 3)
  let c := 3 * cos (1 / 3)
  a < b ∧ b < c := 
by
  let a := 2 * sin (1 / 2)
  let b := 3 * sin (1 / 3)
  let c := 3 * cos (1 / 3)
  sorry

end trigonometric_comparison_l2145_214594


namespace average_salary_without_manager_l2145_214509

theorem average_salary_without_manager (A : ℝ) (H : 15 * A + 4200 = 16 * (A + 150)) : A = 1800 :=
by {
  sorry
}

end average_salary_without_manager_l2145_214509


namespace find_root_of_quadratic_equation_l2145_214596

theorem find_root_of_quadratic_equation
  (a b c : ℝ)
  (h1 : 3 * a * (2 * b - 3 * c) ≠ 0)
  (h2 : 2 * b * (3 * c - 2 * a) ≠ 0)
  (h3 : 5 * c * (2 * a - 3 * b) ≠ 0)
  (r : ℝ)
  (h_roots : (r = -2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c))) ∨ (r = (-2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c))) * 2)) :
  r = -2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c)) :=
by
  sorry

end find_root_of_quadratic_equation_l2145_214596


namespace minimum_value_of_expr_l2145_214507

noncomputable def expr (x y : ℝ) : ℝ := 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4

theorem minimum_value_of_expr : ∃ x y : ℝ, expr x y = -1 ∧ ∀ (a b : ℝ), expr a b ≥ -1 := 
by
  sorry

end minimum_value_of_expr_l2145_214507


namespace solution_to_fraction_problem_l2145_214574

noncomputable def fraction_problem : Prop :=
  ∀ (a b : Nat), a > 0 -> b > 0 -> Nat.gcd a b = 1 ->
    ((a + 12) * b = 3 * a * (b + 12)) -> a = 2 ∧ b = 9

theorem solution_to_fraction_problem : fraction_problem :=
sorry

end solution_to_fraction_problem_l2145_214574


namespace mixed_oil_rate_is_correct_l2145_214501

def rate_of_mixed_oil (volume1 : ℕ) (price1 : ℕ) (volume2 : ℕ) (price2 : ℕ) : ℕ :=
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2)

theorem mixed_oil_rate_is_correct :
  rate_of_mixed_oil 10 50 5 68 = 56 := by
  sorry

end mixed_oil_rate_is_correct_l2145_214501


namespace no_even_is_prime_equiv_l2145_214530

def even (x : ℕ) : Prop := x % 2 = 0
def prime (x : ℕ) : Prop := x > 1 ∧ ∀ d : ℕ, d ∣ x → (d = 1 ∨ d = x)

theorem no_even_is_prime_equiv 
  (H : ¬ ∃ x : ℕ, even x ∧ prime x) :
  ∀ x : ℕ, even x → ¬ prime x :=
by
  sorry

end no_even_is_prime_equiv_l2145_214530


namespace intersection_M_N_l2145_214518

def M : Set ℝ := { x : ℝ | Real.log x / Real.log 2 < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l2145_214518


namespace remaining_water_at_end_of_hike_l2145_214539

-- Define conditions
def initial_water : ℝ := 9
def hike_length : ℝ := 7
def hike_duration : ℝ := 2
def leak_rate : ℝ := 1
def drink_rate_6_miles : ℝ := 0.6666666666666666
def drink_last_mile : ℝ := 2

-- Define the question and answer
def remaining_water (initial: ℝ) (duration: ℝ) (leak: ℝ) (drink6: ℝ) (drink_last: ℝ) : ℝ :=
  initial - ((drink6 * 6) + drink_last + (leak * duration))

-- Theorem stating the proof problem 
theorem remaining_water_at_end_of_hike :
  remaining_water initial_water hike_duration leak_rate drink_rate_6_miles drink_last_mile = 1 :=
by
  sorry

end remaining_water_at_end_of_hike_l2145_214539


namespace arithmetic_sequence_ninth_term_l2145_214560

theorem arithmetic_sequence_ninth_term (a d : ℤ) 
    (h5 : a + 4 * d = 23) (h7 : a + 6 * d = 37) : 
    a + 8 * d = 51 := 
by 
  sorry

end arithmetic_sequence_ninth_term_l2145_214560


namespace percentage_students_below_8_years_l2145_214571

theorem percentage_students_below_8_years :
  ∀ (n8 : ℕ) (n_gt8 : ℕ) (n_total : ℕ),
  n8 = 24 →
  n_gt8 = 2 * n8 / 3 →
  n_total = 50 →
  (n_total - (n8 + n_gt8)) * 100 / n_total = 20 :=
by
  intros n8 n_gt8 n_total h1 h2 h3
  sorry

end percentage_students_below_8_years_l2145_214571


namespace no_unique_y_exists_l2145_214502

theorem no_unique_y_exists (x y : ℕ) (k m : ℤ) 
  (h1 : x % 82 = 5)
  (h2 : (x + 7) % y = 12) :
  ¬ ∃! y, (∃ k m : ℤ, x = 82 * k + 5 ∧ (x + 7) = y * m + 12) :=
by
  sorry

end no_unique_y_exists_l2145_214502


namespace find_coordinates_l2145_214551

def point_in_fourth_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 < 0
def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop := |P.2| = d
def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop := |P.1| = d

theorem find_coordinates :
  ∃ P : ℝ × ℝ, point_in_fourth_quadrant P ∧ distance_to_x_axis P 2 ∧ distance_to_y_axis P 5 ∧ P = (5, -2) :=
by
  sorry

end find_coordinates_l2145_214551


namespace smallest_period_cos_l2145_214542

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem smallest_period_cos (x : ℝ) : 
  smallest_positive_period (λ x => 2 * (Real.cos x)^2 + 1) Real.pi := 
by 
  sorry

end smallest_period_cos_l2145_214542


namespace find_loan_amount_l2145_214550

-- Define the conditions
def rate_of_interest : ℝ := 0.06
def time_period : ℝ := 6
def interest_paid : ℝ := 432

-- Define the simple interest formula
def simple_interest (P r t : ℝ) : ℝ := P * r * t

-- State the theorem to prove the loan amount
theorem find_loan_amount (P : ℝ) (h1 : rate_of_interest = 0.06) (h2 : time_period = 6) (h3 : interest_paid = 432) (h4 : simple_interest P rate_of_interest time_period = interest_paid) : P = 1200 :=
by
  -- Here should be the proof, but it's omitted for now
  sorry

end find_loan_amount_l2145_214550


namespace length_of_segments_equal_d_l2145_214564

noncomputable def d_eq (AB BC AC : ℝ) (h : AB = 550 ∧ BC = 580 ∧ AC = 620) : ℝ :=
  if h_eq : AB = 550 ∧ BC = 580 ∧ AC = 620 then 342 else 0

theorem length_of_segments_equal_d (AB BC AC : ℝ) (h : AB = 550 ∧ BC = 580 ∧ AC = 620) :
  d_eq AB BC AC h = 342 :=
by
  sorry

end length_of_segments_equal_d_l2145_214564


namespace percentage_of_female_officers_on_duty_l2145_214581

theorem percentage_of_female_officers_on_duty :
  ∀ (total_on_duty : ℕ) (half_on_duty : ℕ) (total_female_officers : ℕ), 
  total_on_duty = 204 → half_on_duty = total_on_duty / 2 → total_female_officers = 600 → 
  ((half_on_duty: ℚ) / total_female_officers) * 100 = 17 :=
by
  intro total_on_duty half_on_duty total_female_officers
  intros h1 h2 h3
  sorry

end percentage_of_female_officers_on_duty_l2145_214581


namespace total_number_of_boys_l2145_214562

-- Define the circular arrangement and the opposite positions
variable (n : ℕ)

theorem total_number_of_boys (h : (40 ≠ 10 ∧ (40 - 10) * 2 = n - 2)) : n = 62 := 
sorry

end total_number_of_boys_l2145_214562


namespace cube_painted_probability_l2145_214547

theorem cube_painted_probability :
  let total_cubes := 125
  let cubes_with_3_faces := 1
  let cubes_with_no_faces := 76
  let total_ways := Nat.choose total_cubes 2
  let favorable_ways := cubes_with_3_faces * cubes_with_no_faces
  let probability := (favorable_ways : ℚ) / total_ways
  probability = (2 : ℚ) / 205 :=
by
  sorry

end cube_painted_probability_l2145_214547


namespace max_gcd_dn_l2145_214503

def a (n : ℕ) := 101 + n^2

def d (n : ℕ) := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_dn : ∃ n : ℕ, ∀ m : ℕ, d m ≤ 3 := sorry

end max_gcd_dn_l2145_214503


namespace pants_cost_is_250_l2145_214512

-- Define the cost of a T-shirt
def tshirt_cost := 100

-- Define the total amount spent
def total_amount := 1500

-- Define the number of T-shirts bought
def num_tshirts := 5

-- Define the number of pants bought
def num_pants := 4

-- Define the total cost of T-shirts
def total_tshirt_cost := tshirt_cost * num_tshirts

-- Define the total cost of pants
def total_pants_cost := total_amount - total_tshirt_cost

-- Define the cost per pair of pants
def pants_cost_per_pair := total_pants_cost / num_pants

-- Proving that the cost per pair of pants is $250
theorem pants_cost_is_250 : pants_cost_per_pair = 250 := by
  sorry

end pants_cost_is_250_l2145_214512


namespace factor_theorem_l2145_214577

noncomputable def polynomial_to_factor : Prop :=
  ∀ x : ℝ, x^4 - 4 * x^2 + 4 = (x^2 - 2)^2

theorem factor_theorem : polynomial_to_factor :=
by
  sorry

end factor_theorem_l2145_214577


namespace number_of_groups_of_three_books_l2145_214519

-- Define the given conditions in terms of Lean
def books : ℕ := 15
def chosen_books : ℕ := 3

-- The combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem we need to prove
theorem number_of_groups_of_three_books : combination books chosen_books = 455 := by
  -- Our proof will go here, but we omit it for now
  sorry

end number_of_groups_of_three_books_l2145_214519
