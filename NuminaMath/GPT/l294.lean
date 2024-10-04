import Mathlib

namespace problem_a_problem_b_l294_294204

theorem problem_a (p : ℕ) (hp : Nat.Prime p) : 
  (∃ x : ℕ, (7^(p-1) - 1) = p * x^2) ↔ p = 3 := 
by
  sorry

theorem problem_b (p : ℕ) (hp : Nat.Prime p) : 
  ¬ ∃ x : ℕ, (11^(p-1) - 1) = p * x^2 := 
by
  sorry

end problem_a_problem_b_l294_294204


namespace unknown_number_value_l294_294648

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end unknown_number_value_l294_294648


namespace power_function_general_form_l294_294841

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_general_form (α : ℝ) :
  ∃ y : ℝ, ∃ α : ℝ, f 3 α = y ∧ ∀ x : ℝ, f x α = x ^ α :=
by
  sorry

end power_function_general_form_l294_294841


namespace find_heaviest_or_lightest_l294_294973

theorem find_heaviest_or_lightest (stones : Fin 10 → ℝ)
  (h_distinct: ∀ i j : Fin 10, i ≠ j → stones i ≠ stones j)
  (h_pairwise_sums_distinct : ∀ i j k l : Fin 10, 
    i ≠ j → k ≠ l → stones i + stones j ≠ stones k + stones l) :
  (∃ i : Fin 10, ∀ j : Fin 10, stones i ≥ stones j) ∨ 
  (∃ i : Fin 10, ∀ j : Fin 10, stones i ≤ stones j) :=
sorry

end find_heaviest_or_lightest_l294_294973


namespace average_marks_l294_294497

theorem average_marks (num_students : ℕ) (marks1 marks2 marks3 : ℕ) (num1 num2 num3 : ℕ) (h1 : num_students = 50)
  (h2 : marks1 = 90) (h3 : num1 = 10) (h4 : marks2 = marks1 - 10) (h5 : num2 = 15) (h6 : marks3 = 60) 
  (h7 : num1 + num2 + num3 = 50) (h8 : num3 = num_students - (num1 + num2)) (total_marks : ℕ) 
  (h9 : total_marks = (num1 * marks1) + (num2 * marks2) + (num3 * marks3)) : 
  (total_marks / num_students = 72) :=
by
  sorry

end average_marks_l294_294497


namespace ellipse_equation_line_equation_of_area_l294_294421

open Real

-- Defining constants and variables
variables (a b c x y m : ℝ)
variables (ellipse_eq line_eq : ℝ → ℝ → Prop)

-- Conditions
def conditions : Prop :=
  (a > b ∧ b > 0) ∧
  (2 * a * 1 / 2 = 2 * b) ∧
  (a - c = 2 - sqrt 3) ∧
  (a^2 = b^2 + c^2)

-- The equation of the ellipse
def ellipse_eq := (x : ℝ) (y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)

-- The equation of the line
def line_eq := (x : ℝ) (y : ℝ) : Prop := (y = x + m)

-- Proof of the ellipse equation
theorem ellipse_equation (h : conditions a b c) : ellipse_eq x y :=
by
  sorry

-- Proof of the line equation given the area of the triangle is 1
theorem line_equation_of_area (h : conditions a b c)
  (ha : ellipse_eq x y)
  (area_eq : ∃ A B, (A ≠ B ∧ ∀ m, y = x + m ∧ 
  abs ((1 / 2) * (A.1 * B.2 - A.2 * B.1)) = 1)) : 
  ∃ m, (y = x + m ∨ y = x - m) :=
by
  sorry

end ellipse_equation_line_equation_of_area_l294_294421


namespace breakable_iff_composite_l294_294214

-- Definitions directly from the problem conditions
def is_breakable (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ), a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ a + b = n ∧ (x / a : ℚ) + (y / b : ℚ) = 1

def is_composite (n : ℕ) : Prop :=
  ∃ (s t : ℕ), s > 1 ∧ t > 1 ∧ n = s * t

-- The proof statement
theorem breakable_iff_composite (n : ℕ) : is_breakable n ↔ is_composite n := sorry

end breakable_iff_composite_l294_294214


namespace max_gold_coins_l294_294199

theorem max_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 110) : n ≤ 107 :=
by
  sorry

end max_gold_coins_l294_294199


namespace trigonometric_problem_l294_294243

open Real

noncomputable def problem1 (α : ℝ) : Prop :=
  2 * sin α = 2 * (sin (α / 2))^2 - 1

noncomputable def problem2 (β : ℝ) : Prop :=
  3 * (tan β)^2 - 2 * tan β = 1

theorem trigonometric_problem (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : π / 2 < β ∧ β < π)
  (h1 : problem1 α) (h2 : problem2 β) :
  sin (2 * α) + cos (2 * α) = -1 / 5 ∧ α + β = 7 * π / 4 :=
  sorry

end trigonometric_problem_l294_294243


namespace smallest_six_factors_l294_294724

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end smallest_six_factors_l294_294724


namespace Bert_total_profit_is_14_90_l294_294588

-- Define the sales price for each item
def sales_price_barrel : ℝ := 90
def sales_price_tools : ℝ := 50
def sales_price_fertilizer : ℝ := 30

-- Define the tax rates for each item
def tax_rate_barrel : ℝ := 0.10
def tax_rate_tools : ℝ := 0.05
def tax_rate_fertilizer : ℝ := 0.12

-- Define the profit added per item
def profit_per_item : ℝ := 10

-- Define the tax amount for each item
def tax_barrel : ℝ := tax_rate_barrel * sales_price_barrel
def tax_tools : ℝ := tax_rate_tools * sales_price_tools
def tax_fertilizer : ℝ := tax_rate_fertilizer * sales_price_fertilizer

-- Define the cost price for each item
def cost_price_barrel : ℝ := sales_price_barrel - profit_per_item
def cost_price_tools : ℝ := sales_price_tools - profit_per_item
def cost_price_fertilizer : ℝ := sales_price_fertilizer - profit_per_item

-- Define the profit for each item
def profit_barrel : ℝ := sales_price_barrel - tax_barrel - cost_price_barrel
def profit_tools : ℝ := sales_price_tools - tax_tools - cost_price_tools
def profit_fertilizer : ℝ := sales_price_fertilizer - tax_fertilizer - cost_price_fertilizer

-- Define the total profit
def total_profit : ℝ := profit_barrel + profit_tools + profit_fertilizer

-- Assert the total profit is $14.90
theorem Bert_total_profit_is_14_90 : total_profit = 14.90 :=
by
  -- Omitted proof
  sorry

end Bert_total_profit_is_14_90_l294_294588


namespace find_supplementary_angle_l294_294222

def A := 45
def supplementary_angle (A S : ℕ) := A + S = 180
def complementary_angle (A C : ℕ) := A + C = 90
def thrice_complementary (S C : ℕ) := S = 3 * C

theorem find_supplementary_angle : 
  ∀ (A S C : ℕ), 
    A = 45 → 
    supplementary_angle A S →
    complementary_angle A C →
    thrice_complementary S C → 
    S = 135 :=
by
  intros A S C hA hSupp hComp hThrice
  have h1 : A = 45 := by assumption
  have h2 : A + S = 180 := by assumption
  have h3 : A + C = 90 := by assumption
  have h4 : S = 3 * C := by assumption
  sorry

end find_supplementary_angle_l294_294222


namespace arithmetic_sequence_problem_l294_294667

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h1 : a 2 + a 3 = 4)
  (h2 : a 4 + a 5 = 6) :
  a 9 + a 10 = 11 :=
sorry

end arithmetic_sequence_problem_l294_294667


namespace discount_double_time_l294_294434

theorem discount_double_time (TD FV : ℝ) (h1 : TD = 10) (h2 : FV = 110) : 
  2 * TD = 20 :=
by
  sorry

end discount_double_time_l294_294434


namespace number_of_pumps_l294_294694

theorem number_of_pumps (P : ℕ) : 
  (P * 8 * 2 = 8 * 6) → P = 3 :=
by
  intro h
  sorry

end number_of_pumps_l294_294694


namespace u_2023_is_4_l294_294790

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 2
  | 4 => 1
  | 5 => 4
  | _ => 0  -- f is only defined for x in {1, 2, 3, 4, 5}

def u : ℕ → ℕ
| 0 => 5
| (n + 1) => f (u n)

theorem u_2023_is_4 : u 2023 = 4 := by
  sorry

end u_2023_is_4_l294_294790


namespace cylindrical_tank_depth_l294_294864

theorem cylindrical_tank_depth (V : ℝ) (d h : ℝ) (π : ℝ) : 
  V = 1848 ∧ d = 14 ∧ π = Real.pi → h = 12 :=
by
  sorry

end cylindrical_tank_depth_l294_294864


namespace find_weight_of_b_l294_294697

variable (a b c d : ℝ)

def average_weight_of_four : Prop := (a + b + c + d) / 4 = 45

def average_weight_of_a_and_b : Prop := (a + b) / 2 = 42

def average_weight_of_b_and_c : Prop := (b + c) / 2 = 43

def ratio_of_d_to_a : Prop := d / a = 3 / 4

theorem find_weight_of_b (h1 : average_weight_of_four a b c d)
                        (h2 : average_weight_of_a_and_b a b)
                        (h3 : average_weight_of_b_and_c b c)
                        (h4 : ratio_of_d_to_a a d) :
    b = 29.43 :=
  by sorry

end find_weight_of_b_l294_294697


namespace probability_suitable_joint_given_physique_l294_294107

noncomputable def total_children : ℕ := 20
noncomputable def suitable_physique : ℕ := 4
noncomputable def suitable_joint_structure : ℕ := 5
noncomputable def both_physique_and_joint : ℕ := 2

noncomputable def P (n m : ℕ) : ℚ := n / m

theorem probability_suitable_joint_given_physique :
  P both_physique_and_joint total_children / P suitable_physique total_children = 1 / 2 :=
by
  sorry

end probability_suitable_joint_given_physique_l294_294107


namespace spinner_probabilities_l294_294070

noncomputable def prob_A : ℚ := 1 / 3
noncomputable def prob_B : ℚ := 1 / 4
noncomputable def prob_C : ℚ := 5 / 18
noncomputable def prob_D : ℚ := 5 / 36

theorem spinner_probabilities :
  prob_A + prob_B + prob_C + prob_D = 1 ∧
  prob_C = 2 * prob_D :=
by {
  -- The statement of the theorem matches the given conditions and the correct answers.
  -- Proof will be provided later.
  sorry
}

end spinner_probabilities_l294_294070


namespace platform_length_l294_294868

theorem platform_length
  (train_length : ℤ)
  (speed_kmph : ℤ)
  (time_sec : ℤ)
  (speed_mps : speed_kmph * 1000 / 3600 = 20)
  (distance_eq : (train_length + 220) = (20 * time_sec))
  (train_length_val : train_length = 180)
  (time_sec_val : time_sec = 20) :
  220 = 220 := by
  sorry

end platform_length_l294_294868


namespace absent_minded_scientist_mistake_l294_294484

theorem absent_minded_scientist_mistake (ξ η : ℝ) (h₁ : E ξ = 3) (h₂ : E η = 5) (h₃ : E (min ξ η) = 3 + 2/3) : false :=
by
  sorry

end absent_minded_scientist_mistake_l294_294484


namespace real_roots_exactly_three_l294_294778

theorem real_roots_exactly_three (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * |x| + 2 = m) → (∃ a b c : ℝ, 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (a^2 - 2 * |a| + 2 = m) ∧ 
  (b^2 - 2 * |b| + 2 = m) ∧ 
  (c^2 - 2 * |c| + 2 = m)) → 
  m = 2 := 
sorry

end real_roots_exactly_three_l294_294778


namespace min_value_of_u_l294_294610

theorem min_value_of_u (x y : ℝ) (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) (hxy : x * y = -1) :
  (∀ u, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) → u ≥ (12 / 5)) :=
by
  sorry

end min_value_of_u_l294_294610


namespace coffee_price_decrease_is_37_5_l294_294939

-- Define the initial and new prices
def initial_price_per_packet := 12 / 3
def new_price_per_packet := 10 / 4

-- Define the calculation of the percent decrease
def percent_decrease (initial_price : ℚ) (new_price : ℚ) : ℚ :=
  ((initial_price - new_price) / initial_price) * 100

-- The theorem statement
theorem coffee_price_decrease_is_37_5 :
  percent_decrease initial_price_per_packet new_price_per_packet = 37.5 := by
  sorry

end coffee_price_decrease_is_37_5_l294_294939


namespace parallel_lines_iff_a_eq_2_l294_294428

-- Define line equations
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - a + 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y - 2 = 0

-- Prove that a = 2 is necessary and sufficient for the lines to be parallel.
theorem parallel_lines_iff_a_eq_2 (a : ℝ) :
  (∀ x y : ℝ, l1 a x y → ∃ u v : ℝ, l2 a u v → x = u ∧ y = v) ↔ (a = 2) :=
by {
  sorry
}

end parallel_lines_iff_a_eq_2_l294_294428


namespace an_gt_bn_l294_294312

theorem an_gt_bn (a b : ℕ → ℕ) (h₁ : a 1 = 2013) (h₂ : ∀ n, a (n + 1) = 2013^(a n))
                            (h₃ : b 1 = 1) (h₄ : ∀ n, b (n + 1) = 2013^(2012 * (b n))) :
  ∀ n, a n > b n := 
sorry

end an_gt_bn_l294_294312


namespace interest_rate_proof_l294_294033

variable (P : ℝ) (n : ℕ) (CI SI : ℝ → ℝ → ℕ → ℝ) (diff : ℝ → ℝ → ℝ)

def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n
def simple_interest (P r : ℝ) (n : ℕ) : ℝ := P * r * n

theorem interest_rate_proof (r : ℝ) :
  diff (compound_interest 5400 r 2) (simple_interest 5400 r 2) = 216 → r = 0.2 :=
by sorry

end interest_rate_proof_l294_294033


namespace largest_fraction_l294_294259

theorem largest_fraction (x y z w : ℝ) (hx : 0 < x) (hxy : x < y) (hyz : y < z) (hzw : z < w) :
  max (max (max (max ((x + y) / (z + w)) ((x + w) / (y + z))) ((y + z) / (x + w))) ((y + w) / (x + z))) ((z + w) / (x + y)) = (z + w) / (x + y) :=
by sorry

end largest_fraction_l294_294259


namespace total_trash_cans_paid_for_l294_294748

-- Definitions based on conditions
def trash_cans_on_streets : ℕ := 14
def trash_cans_back_of_stores : ℕ := 2 * trash_cans_on_streets

-- Theorem to prove
theorem total_trash_cans_paid_for : trash_cans_on_streets + trash_cans_back_of_stores = 42 := 
by
  -- proof would go here, but we use sorry since proof is not required
  sorry

end total_trash_cans_paid_for_l294_294748


namespace thirtieth_change_month_is_february_l294_294767

def months_in_year := 12

def months_per_change := 7

def first_change_month := 3 -- March (if we assume January = 1, February = 2, etc.)

def nth_change_month (n : ℕ) : ℕ :=
  (first_change_month + months_per_change * (n - 1)) % months_in_year

theorem thirtieth_change_month_is_february :
  nth_change_month 30 = 2 := -- February (if we assume January = 1, February = 2, etc.)
by 
  sorry

end thirtieth_change_month_is_february_l294_294767


namespace distinct_factors_count_l294_294763

-- Given conditions
def eight_squared : ℕ := 8^2
def nine_cubed : ℕ := 9^3
def seven_fifth : ℕ := 7^5
def number : ℕ := eight_squared * nine_cubed * seven_fifth

-- Proving the number of natural-number factors of the given number
theorem distinct_factors_count : 
  (number.factors.count 1 = 294) := sorry

end distinct_factors_count_l294_294763


namespace find_prices_l294_294975

def price_system_of_equations (x y : ℕ) : Prop :=
  3 * x + 2 * y = 474 ∧ x - y = 8

theorem find_prices (x y : ℕ) :
  price_system_of_equations x y :=
by
  sorry

end find_prices_l294_294975


namespace savings_wednesday_l294_294592

variable (m t s w : ℕ)

theorem savings_wednesday :
  m = 15 → t = 28 → s = 28 → 2 * s = 56 → 
  m + t + w = 56 → w = 13 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end savings_wednesday_l294_294592


namespace largest_n_for_factorable_polynomial_l294_294238

theorem largest_n_for_factorable_polynomial : ∃ n, 
  (∀ A B : ℤ, (6 * B + A = n) → (A * B = 144)) ∧ 
  (∀ n', (∀ A B : ℤ, (6 * B + A = n') → (A * B = 144)) → n' ≤ n) ∧ 
  (n = 865) :=
by
  sorry

end largest_n_for_factorable_polynomial_l294_294238


namespace find_percentage_decrease_l294_294036

noncomputable def initialPrice : ℝ := 100
noncomputable def priceAfterJanuary : ℝ := initialPrice * 1.30
noncomputable def priceAfterFebruary : ℝ := priceAfterJanuary * 0.85
noncomputable def priceAfterMarch : ℝ := priceAfterFebruary * 1.10

theorem find_percentage_decrease :
  ∃ (y : ℝ), (priceAfterMarch * (1 - y / 100) = initialPrice) ∧ abs (y - 18) < 1 := 
sorry

end find_percentage_decrease_l294_294036


namespace expand_binom_l294_294407

theorem expand_binom (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 :=
by
  sorry

end expand_binom_l294_294407


namespace equation_is_point_l294_294884

-- Definition of the condition in the problem
def equation (x y : ℝ) := x^2 + 36*y^2 - 12*x - 72*y + 36 = 0

-- The theorem stating the equivalence to the point (6, 1)
theorem equation_is_point :
  ∀ (x y : ℝ), equation x y → (x = 6 ∧ y = 1) :=
by
  intros x y h
  -- The proof steps would go here
  sorry

end equation_is_point_l294_294884


namespace variance_N_l294_294453

-- Define the independent random variables ε_i
noncomputable def epsilon (i : ℕ) : ℕ → ℤ := sorry

-- Define S_k as the sum of the ε_i's up to k
noncomputable def S (k : ℕ) : ℕ → ℤ :=
  λ n, (1 to k).sum_fun (εpsilon n)

-- Define N_{2n} as the count of integers k in [2, 2n] such that S_k > 0 or S_k = 0 and S_{k-1} > 0
noncomputable def N (n : ℕ) : ℕ → ℤ :=
  λ n, (2 to 2 * n).count_fun (λ k, (S k n > 0) ∨ ((S k n = 0) ∧ (S (k - 1) n > 0)))

-- Prove the variance of N_{2n} is 3(2n-1)/16
theorem variance_N (n : ℕ) : var (N (2 * n)) = 3 * (2 * n - 1) / 16 := by
  sorry

end variance_N_l294_294453


namespace range_of_x_l294_294038

theorem range_of_x (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x - 3 ≠ 0) ↔ (x ≥ -1) ∧ (x ≠ 3) :=
by sorry

end range_of_x_l294_294038


namespace track_length_is_450_l294_294878

theorem track_length_is_450 (x : ℝ) (d₁ : ℝ) (d₂ : ℝ)
  (h₁ : d₁ = 150)
  (h₂ : x - d₁ = 120)
  (h₃ : d₂ = 200)
  (h₄ : ∀ (d₁ d₂ : ℝ) (t₁ t₂ : ℝ), t₁ / t₂ = d₁ / d₂)
  : x = 450 := by
  sorry

end track_length_is_450_l294_294878


namespace ball_box_problem_l294_294268

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l294_294268


namespace library_books_l294_294353

theorem library_books (N x y : ℕ) (h1 : x = N / 17) (h2 : y = x + 2000)
    (h3 : y = (N - 2 * 2000) / 15 + (14 * (N - 2000) / 17)): 
  N = 544000 := 
sorry

end library_books_l294_294353


namespace correct_statement_among_given_l294_294198

theorem correct_statement_among_given (
  (cond_A : ∀ Q : Type, ∀ (q : quadrilateral Q), (one_pair_parallel_sides q → one_pair_equal_sides q → parallelogram q)),
  (cond_B : ∀ P : Type, ∀ (p : parallelogram P), (complementary_diagonals p)),
  (cond_C : ∀ Q : Type, ∀ (q : quadrilateral Q), (two_pairs_equal_angles q → parallelogram q)),
  (cond_D : ∀ P : Type, ∀ (p : parallelogram P), (diagonals_bisect_opposite_angles p))
) : 
  ∃ S, S = cond_C := sorry

end correct_statement_among_given_l294_294198


namespace largest_angle_sine_of_C_l294_294663

-- Given conditions
def side_a : ℝ := 7
def side_b : ℝ := 3
def side_c : ℝ := 5

-- 1. Prove the largest angle
theorem largest_angle (a b c : ℝ) (h₁ : a = 7) (h₂ : b = 3) (h₃ : c = 5) : 
  ∃ A : ℝ, A = 120 :=
by
  sorry

-- 2. Prove the sine value of angle C
theorem sine_of_C (a b c A : ℝ) (h₁ : a = 7) (h₂ : b = 3) (h₃ : c = 5) (h₄ : A = 120) : 
  ∃ sinC : ℝ, sinC = 5 * (Real.sqrt 3) / 14 :=
by
  sorry

end largest_angle_sine_of_C_l294_294663


namespace average_marks_l294_294494

theorem average_marks (total_students : ℕ) (first_group : ℕ) (first_group_marks : ℕ)
                      (second_group : ℕ) (second_group_marks_diff : ℕ) (third_group_marks : ℕ)
                      (total_marks : ℕ) (class_average : ℕ) :
  total_students = 50 → 
  first_group = 10 → 
  first_group_marks = 90 → 
  second_group = 15 → 
  second_group_marks_diff = 10 → 
  third_group_marks = 60 →
  total_marks = (first_group * first_group_marks) + (second_group * (first_group_marks - second_group_marks_diff)) + ((total_students - (first_group + second_group)) * third_group_marks) →
  class_average = total_marks / total_students →
  class_average = 72 :=
by
  intros
  sorry

end average_marks_l294_294494


namespace triangle_area_proof_l294_294692

noncomputable def area_of_triangle_ABC : ℝ :=
  8 * Real.sqrt 2

theorem triangle_area_proof (A B C C1 A1 : ℝ) 
  (h1 : is_height_of_triangle A B C C1)
  (h2 : right_angle_at A C1 C)
  (h3 : is_median A A1)
  (h4: isosceles_triangle A B C)
  (h5 : right_triangle B C C1)
  (h6 : median_equals A1 C1 2)
  (h7 : length BC = 4)
  (h8 : length AB = 6)
  (h9 : length AC = 6)
  (h10 : semi_perimeter = (length AB + length BC + length AC) / 2)
  (h: Herons_formula semi_perimeter (length AB) (length BC) (length AC) = 8 * Real.sqrt 2):
  area_of_triangle_ABC = Herons_formula semi_perimeter (length AB) (length BC) (length AC) := 
  by sorry


end triangle_area_proof_l294_294692


namespace slow_speed_distance_l294_294292

theorem slow_speed_distance (D : ℝ) (h : (D + 20) / 14 = D / 10) : D = 50 := by
  sorry

end slow_speed_distance_l294_294292


namespace reaction2_follows_markovnikov_l294_294562

-- Define Markovnikov's rule - applying to case with protic acid (HX) to an alkene.
def follows_markovnikov_rule (HX : String) (initial_molecule final_product : String) : Prop :=
  initial_molecule = "CH3-CH=CH2 + HBr" ∧ final_product = "CH3-CHBr-CH3"

-- Example reaction data
def reaction1_initial : String := "CH2=CH2 + Br2"
def reaction1_final : String := "CH2Br-CH2Br"

def reaction2_initial : String := "CH3-CH=CH2 + HBr"
def reaction2_final : String := "CH3-CHBr-CH3"

def reaction3_initial : String := "CH4 + Cl2"
def reaction3_final : String := "CH3Cl + HCl"

def reaction4_initial : String := "CH ≡ CH + HOH"
def reaction4_final : String := "CH3''-C-H"

-- Proof statement
theorem reaction2_follows_markovnikov : follows_markovnikov_rule "HBr" reaction2_initial reaction2_final := by
  sorry

end reaction2_follows_markovnikov_l294_294562


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_152_l294_294681

-- Conditions
variable (x : ℝ) (h₀ : 0 < x) (h₁ : x + 1 / x = 150)

-- Statement to prove
theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_152 : (Real.sqrt x + Real.sqrt (1 / x) = Real.sqrt 152) := 
sorry -- Proof not needed, skip with sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_152_l294_294681


namespace Apollonian_Circle_Range_l294_294345

def range_of_m := Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 21 / 2)

theorem Apollonian_Circle_Range :
  ∃ P : ℝ × ℝ, ∃ m > 0, ((P.1 - 2) ^ 2 + (P.2 - m) ^ 2 = 1 / 4) ∧ 
            (Real.sqrt ((P.1 + 1) ^ 2 + P.2 ^ 2) = 2 * Real.sqrt ((P.1 - 2) ^ 2 + P.2 ^ 2)) →
            m ∈ range_of_m :=
  sorry

end Apollonian_Circle_Range_l294_294345


namespace min_value_l294_294901

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 
  ∃ c : ℝ, c = 4 ∧ 
  ∀ x y : ℝ, (x = 1 / (a - 1) ∧ y = 4 / (b - 1)) → (x + y ≥ c) :=
sorry

end min_value_l294_294901


namespace solve_for_x_l294_294173

theorem solve_for_x (x : ℚ) : 
  5*x + 9*x = 450 - 10*(x - 5) -> x = 125/6 :=
by
  sorry

end solve_for_x_l294_294173


namespace probability_sum_less_than_product_l294_294520

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l294_294520


namespace walkway_area_l294_294074

theorem walkway_area (l w : ℕ) (walkway_width : ℕ) (total_length total_width pool_area walkway_area : ℕ)
  (hl : l = 20) 
  (hw : w = 8)
  (hww : walkway_width = 1)
  (htl : total_length = l + 2 * walkway_width)
  (htw : total_width = w + 2 * walkway_width)
  (hpa : pool_area = l * w)
  (hta : (total_length * total_width) = pool_area + walkway_area) :
  walkway_area = 60 := 
  sorry

end walkway_area_l294_294074


namespace sin_2theta_in_third_quadrant_l294_294120

open Real

variables (θ : ℝ)

/-- \theta is an angle in the third quadrant.
Given that \(\sin^{4}\theta + \cos^{4}\theta = \frac{5}{9}\), 
prove that \(\sin 2\theta = \frac{2\sqrt{2}}{3}\). --/
theorem sin_2theta_in_third_quadrant (h_theta_third_quadrant : π < θ ∧ θ < 3 * π / 2)
(h_cond : sin θ ^ 4 + cos θ ^ 4 = 5 / 9) : sin (2 * θ) = 2 * sqrt 2 / 3 :=
sorry

end sin_2theta_in_third_quadrant_l294_294120


namespace geometric_series_sum_correct_l294_294757

-- Given conditions
def a : ℤ := 3
def r : ℤ := -2
def n : ℤ := 10

-- Sum of the geometric series formula
def geometric_series_sum (a r n : ℤ) : ℤ := 
  a * (r^n - 1) / (r - 1)

-- Goal: Prove that the sum of the series is -1023
theorem geometric_series_sum_correct : 
  geometric_series_sum a r n = -1023 := 
by
  sorry

end geometric_series_sum_correct_l294_294757


namespace ac_bd_leq_8_l294_294253

theorem ac_bd_leq_8 (a b c d : ℝ) (h1 : a^2 + b^2 = 4) (h2 : c^2 + d^2 = 16) : ac + bd ≤ 8 :=
sorry

end ac_bd_leq_8_l294_294253


namespace magician_trick_success_l294_294741

theorem magician_trick_success {n : ℕ} (T_pos : ℕ) (deck_size : ℕ := 52) (discard_count : ℕ := 51):
  (T_pos = 1 ∨ T_pos = deck_size) → ∃ strategy : Type, ∀ spectator_choice : ℕ, (spectator_choice ≤ deck_size) → 
                          ((T_pos = 1 → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)
                          ∧ (T_pos = deck_size → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)) :=
sorry

end magician_trick_success_l294_294741


namespace B_pow_16_eq_I_l294_294815

noncomputable def B : Matrix (Fin 4) (Fin 4) ℝ := 
  ![
    ![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4), 0 , 0],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4), 0 , 0],
    ![0, 0, Real.cos (Real.pi / 4), Real.sin (Real.pi / 4)],
    ![0, 0, -Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]
  ]

theorem B_pow_16_eq_I : B^16 = 1 := by
  sorry

end B_pow_16_eq_I_l294_294815


namespace problem_l294_294192

theorem problem (a b c d e : ℝ) (h0 : a ≠ 0)
  (h1 : 625 * a + 125 * b + 25 * c + 5 * d + e = 0)
  (h2 : -81 * a + 27 * b - 9 * c + 3 * d + e = 0)
  (h3 : 16 * a + 8 * b + 4 * c + 2 * d + e = 0) :
  (b + c + d) / a = -6 :=
by
  sorry

end problem_l294_294192


namespace probability_without_replacement_probability_with_replacement_l294_294067

-- Definition for without replacement context
def without_replacement_total_outcomes : ℕ := 6
def without_replacement_favorable_outcomes : ℕ := 3
def without_replacement_prob : ℚ :=
  without_replacement_favorable_outcomes / without_replacement_total_outcomes

-- Theorem stating that the probability of selecting two consecutive integers without replacement is 1/2
theorem probability_without_replacement : 
  without_replacement_prob = 1 / 2 := by
  sorry

-- Definition for with replacement context
def with_replacement_total_outcomes : ℕ := 16
def with_replacement_favorable_outcomes : ℕ := 3
def with_replacement_prob : ℚ :=
  with_replacement_favorable_outcomes / with_replacement_total_outcomes

-- Theorem stating that the probability of selecting two consecutive integers with replacement is 3/16
theorem probability_with_replacement : 
  with_replacement_prob = 3 / 16 := by
  sorry

end probability_without_replacement_probability_with_replacement_l294_294067


namespace total_opponent_scores_is_45_l294_294081

-- Definitions based on the conditions
def games : Fin 10 := Fin.mk 10 sorry

def team_scores : Fin 10 → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 2
| ⟨2, _⟩ => 3
| ⟨3, _⟩ => 4
| ⟨4, _⟩ => 5
| ⟨5, _⟩ => 6
| ⟨6, _⟩ => 7
| ⟨7, _⟩ => 8
| ⟨8, _⟩ => 9
| ⟨9, _⟩ => 10
| _ => 0  -- Placeholder for out-of-bounds, should not be used

def lost_games : Fin 5 → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 3
| ⟨2, _⟩ => 5
| ⟨3, _⟩ => 7
| ⟨4, _⟩ => 9

def opponent_score_lost : ℕ → ℕ := λ s => s + 1

def won_games : Fin 5 → ℕ
| ⟨0, _⟩ => 2
| ⟨1, _⟩ => 4
| ⟨2, _⟩ => 6
| ⟨3, _⟩ => 8
| ⟨4, _⟩ => 10

def opponent_score_won : ℕ → ℕ := λ s => s / 2

-- Main statement to prove total opponent scores
theorem total_opponent_scores_is_45 :
  let total_lost_scores := (lost_games 0 :: lost_games 1 :: lost_games 2 :: lost_games 3 :: lost_games 4 :: []).map opponent_score_lost
  let total_won_scores  := (won_games 0 :: won_games 1 :: won_games 2 :: won_games 3 :: won_games 4 :: []).map opponent_score_won
  total_lost_scores.sum + total_won_scores.sum = 45 :=
by sorry

end total_opponent_scores_is_45_l294_294081


namespace range_of_p_l294_294249

theorem range_of_p 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, S n = (-1 : ℝ)^n * a n + 1/(2^n) + n - 3)
  (h2 : ∀ n : ℕ, (a (n + 1) - p) * (a n - p) < 0) :
  -3/4 < p ∧ p < 11/4 :=
sorry

end range_of_p_l294_294249


namespace subset_P_Q_l294_294324

-- Definitions of the sets P and Q
def P : Set ℝ := {x | x^2 - 3 * x + 2 < 0}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- Statement to prove P ⊆ Q
theorem subset_P_Q : P ⊆ Q :=
sorry

end subset_P_Q_l294_294324


namespace probability_sum_less_than_product_l294_294544

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l294_294544


namespace gotham_street_termite_ridden_not_collapsing_l294_294329

def fraction_termite_ridden := 1 / 3
def fraction_collapsing_given_termite_ridden := 4 / 7
def fraction_not_collapsing := 3 / 21

theorem gotham_street_termite_ridden_not_collapsing
  (h1: fraction_termite_ridden = 1 / 3)
  (h2: fraction_collapsing_given_termite_ridden = 4 / 7) :
  fraction_termite_ridden * (1 - fraction_collapsing_given_termite_ridden) = fraction_not_collapsing :=
sorry

end gotham_street_termite_ridden_not_collapsing_l294_294329


namespace incorrect_calculation_l294_294482

theorem incorrect_calculation
  (ξ η : ℝ)
  (Eξ : ℝ)
  (Eη : ℝ)
  (E_min : ℝ)
  (hEξ : Eξ = 3)
  (hEη : Eη = 5)
  (hE_min : E_min = 3.67) :
  E_min > Eξ :=
by
  sorry

end incorrect_calculation_l294_294482


namespace count_two_digit_integers_remainder_3_l294_294626

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l294_294626


namespace minimum_people_correct_answer_l294_294381

theorem minimum_people_correct_answer (people questions : ℕ) (common_correct : ℕ) (h_people : people = 21) (h_questions : questions = 15) (h_common_correct : ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ people → 1 ≤ b ∧ b ≤ people → a ≠ b → common_correct ≥ 1) :
  ∃ (min_correct : ℕ), min_correct = 7 := 
sorry

end minimum_people_correct_answer_l294_294381


namespace lines_intersect_l294_294873

structure Point where
  x : ℝ
  y : ℝ

def line1 (t : ℝ) : Point :=
  ⟨1 + 2 * t, 4 - 3 * t⟩

def line2 (u : ℝ) : Point :=
  ⟨5 + 4 * u, -2 - 5 * u⟩

theorem lines_intersect (x y t u : ℝ) 
  (h1 : x = 1 + 2 * t)
  (h2 : y = 4 - 3 * t)
  (h3 : x = 5 + 4 * u)
  (h4 : y = -2 - 5 * u) :
  x = 5 ∧ y = -2 := 
sorry

end lines_intersect_l294_294873


namespace protective_additive_increase_l294_294058

def percentIncrease (old_val new_val : ℕ) : ℚ :=
  (new_val - old_val) / old_val * 100

theorem protective_additive_increase :
  percentIncrease 45 60 = 33.33 := 
sorry

end protective_additive_increase_l294_294058


namespace sum_of_variables_l294_294174

theorem sum_of_variables (x y z w : ℤ) 
(h1 : x - y + z = 7) 
(h2 : y - z + w = 8) 
(h3 : z - w + x = 4) 
(h4 : w - x + y = 3) : 
x + y + z + w = 11 := 
sorry

end sum_of_variables_l294_294174


namespace balls_in_boxes_l294_294267

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l294_294267


namespace Anhui_mountains_arrangement_l294_294847

/-!
The terrain in Anhui Province includes plains, plateaus (hills), hills, mountains, and other types. 
The hilly areas account for a large proportion, so there are many mountains. Some famous mountains include 
Huangshan, Jiuhuashan, and Tianzhushan. A school has organized a study tour course and plans to send 
5 outstanding students to these three places for study tours. Each mountain must have at least 
one student participating. Prove the number of different arrangement options is 150.
-/

open Nat Function

def countArrangements (n k : Nat) : Nat :=
  factorial n / factorial (n - k)

theorem Anhui_mountains_arrangement :
  let students := 5
  let mountains := 3
  ∃ arrangements options, 
  arrangements = countArrangements 5 3 * 6 ∧
  options = 150 ∧
  arrangements = options
:=
  sorry

end Anhui_mountains_arrangement_l294_294847


namespace fred_seashells_l294_294854

-- Definitions based on conditions
def tom_seashells : Nat := 15
def total_seashells : Nat := 58

-- The theorem we want to prove
theorem fred_seashells : (15 + F = 58) → F = 43 := 
by
  intro h
  have h1 : F = 58 - 15 := by linarith
  exact h1

end fred_seashells_l294_294854


namespace ways_to_distribute_balls_in_boxes_l294_294279

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l294_294279


namespace find_y_l294_294834

theorem find_y (t y : ℝ) (h1 : -3 = 2 - t) (h2 : y = 4 * t + 7) : y = 27 :=
sorry

end find_y_l294_294834


namespace donation_student_amount_l294_294136

theorem donation_student_amount (a : ℕ) : 
  let total_amount := 3150
  let teachers_count := 5
  let donation_teachers := teachers_count * a 
  let donation_students := total_amount - donation_teachers
  donation_students = 3150 - 5 * a :=
by
  sorry

end donation_student_amount_l294_294136


namespace sum_of_all_n_l294_294551

-- Definitions based on the problem statement
def is_integer_fraction (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def is_odd_divisor (a b : ℤ) : Prop := b % 2 = 1 ∧ ∃ k : ℤ, a = b * k

-- Problem Statement
theorem sum_of_all_n (S : ℤ) :
  (S = ∑ n in {n : ℤ | is_integer_fraction 36 (2 * n - 1)}, n) →
  S = 8 :=
by
  sorry

end sum_of_all_n_l294_294551


namespace width_of_boxes_l294_294211

theorem width_of_boxes
  (total_volume : ℝ)
  (total_payment : ℝ)
  (cost_per_box : ℝ)
  (h1 : total_volume = 1.08 * 10^6)
  (h2 : total_payment = 120)
  (h3 : cost_per_box = 0.2) :
  (∃ w : ℝ, w = (total_volume / (total_payment / cost_per_box))^(1/3)) :=
by {
  sorry
}

end width_of_boxes_l294_294211


namespace problem_1_problem_2_l294_294251

/-- Given an ellipse defined by (x^2)/4 + y^2 = 1, a line parallel to the x-axis intersects the ellipse at points A and B, and ∠AOB = 90°. Then the area of △AOB is 4/5. -/
theorem problem_1 (O A B : Real × Real) (hO : O = (0, 0)) 
  (hA : ∃ x1 y1 : Real, A = (x1, y1) ∧ (x1 < 0) ∧ (x1 ≠ 0) ∧ (y1 > 0) ∧ (x1 ^ 2) / 4 + y1 ^ 2 = 1)
  (hB : ∃ x2 y2 : Real, B = (x2, y2) ∧ x2 = -x1 ∧ y2 = y1)
  (hangle : (x1, y1) • (-x1, y1) = 0) :
  let △OAB := 1 / 2 * |(O.1 * (A.2 - B.2) + A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2))| in
  △OAB = 4 / 5 := sorry

/-- Given an ellipse defined by (x^2)/4 + y^2 = 1, and a line always tangent to the circle of radius r, the value of r is 2√5/5. -/
theorem problem_2 (r : Real) 
  (hr : ∃ k m : Real, ∀ x y : Real, x ^ 2 + y ^ 2 = r ^ 2 ∧ (4 * k ^ 2 + 1) * x ^ 2 + 8 * k * m * x + 4 * m ^ 2 - 4 = 0 
  ∧ (5 * m ^ 2 - 4 * k ^ 2 - 4 = 0) ∧ r > 0 ∧ (4 * k ^ 2 = 5 * m ^ 2 - 4) 
  ∧ m ^ 2 ≥ 4 / 5 ∧ m ^ 2 > 3 / 4) :
  r = 2 * real.sqrt(5.0) / 5 := sorry

end problem_1_problem_2_l294_294251


namespace maximum_area_l294_294576

-- Define necessary variables and conditions
variables (x y : ℝ)
variable (A : ℝ)
variable (peri : ℝ := 30)

-- Provide the premise that defines the perimeter condition
axiom perimeter_condition : 2 * x + 2 * y = peri

-- Define y in terms of x based on the perimeter condition
def y_in_terms_of_x (x : ℝ) : ℝ := 15 - x

-- Define the area of the rectangle in terms of x
def area (x : ℝ) : ℝ := x * (y_in_terms_of_x x)

-- The statement that needs to be proved
theorem maximum_area : A = 56.25 :=
by sorry

end maximum_area_l294_294576


namespace find_m_pure_imaginary_l294_294919

noncomputable def find_m (m : ℝ) : ℝ := m

theorem find_m_pure_imaginary (m : ℝ) (h : (m^2 - 5 * m + 6 : ℂ) = 0) :
  find_m m = 2 :=
by
  sorry

end find_m_pure_imaginary_l294_294919


namespace count_two_digit_integers_with_remainder_3_l294_294635

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l294_294635


namespace paint_amount_third_day_l294_294078

theorem paint_amount_third_day : 
  let initial_paint := 80
  let first_day_usage := initial_paint / 2
  let paint_after_first_day := initial_paint - first_day_usage
  let added_paint := 20
  let new_total_paint := paint_after_first_day + added_paint
  let second_day_usage := new_total_paint / 2
  let paint_after_second_day := new_total_paint - second_day_usage
  paint_after_second_day = 30 :=
by
  sorry

end paint_amount_third_day_l294_294078


namespace gcf_360_180_l294_294858

theorem gcf_360_180 : Nat.gcd 360 180 = 180 :=
by
  sorry

end gcf_360_180_l294_294858


namespace exists_points_B_and_C_on_circle_l294_294602

open Classical

noncomputable theory

-- Definitions
def Circle := { p : ℝ × ℝ // p.1^2 + p.2^2 = 1 }

variable (A O : ℝ × ℝ) [Circle A]

-- Conditions
def is_incenter (O A B C : ℝ × ℝ) : Prop :=
  ∃ (I : ℝ × ℝ) (r : ℝ), I = O ∧ 
  dist I (line_segment ℝ A B) = r ∧ 
  dist I (line_segment ℝ B C) = r ∧ 
  dist I (line_segment ℝ C A) = r

-- Proof statement
theorem exists_points_B_and_C_on_circle
  (A O : ℝ × ℝ) (hA : A.1^2 + A.2^2 = 1) (hO_inside_circle : ∃ r, 0 < r ∧ dist O (0,0) < r) :
  ∃ (B C : ℝ × ℝ), B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧ is_incenter O A B C :=
sorry

end exists_points_B_and_C_on_circle_l294_294602


namespace Margie_distance_on_25_dollars_l294_294327

theorem Margie_distance_on_25_dollars
  (miles_per_gallon : ℝ)
  (cost_per_gallon : ℝ)
  (amount_spent : ℝ) :
  miles_per_gallon = 40 →
  cost_per_gallon = 5 →
  amount_spent = 25 →
  (amount_spent / cost_per_gallon) * miles_per_gallon = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end Margie_distance_on_25_dollars_l294_294327


namespace quadrilateral_with_equal_angles_is_parallelogram_l294_294197

axiom Quadrilateral (a b c d : Type) : Prop
axiom Parallelogram (a b c d : Type) : Prop
axiom equal_angles (a b c d : Type) : Prop

theorem quadrilateral_with_equal_angles_is_parallelogram 
  (a b c d : Type) 
  (q : Quadrilateral a b c d)
  (h : equal_angles a b c d) : Parallelogram a b c d := 
sorry

end quadrilateral_with_equal_angles_is_parallelogram_l294_294197


namespace prob_xi_ge_2_eq_one_third_l294_294161

noncomputable def pmf (c k : ℝ) : ℝ := c / (k * (k + 1))

theorem prob_xi_ge_2_eq_one_third 
  (c : ℝ) 
  (h₁ : pmf c 1 + pmf c 2 + pmf c 3 = 1) :
  pmf c 2 + pmf c 3 = 1 / 3 :=
by
  sorry

end prob_xi_ge_2_eq_one_third_l294_294161


namespace triangle_inequality_for_f_l294_294110

noncomputable def f (x m : ℝ) : ℝ :=
  x^3 - 3 * x + m

theorem triangle_inequality_for_f (a b c m : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 2) (h₂ : 0 ≤ b) (h₃ : b ≤ 2) (h₄ : 0 ≤ c) (h₅ : c ≤ 2) 
(h₆ : 6 < m) :
  ∃ u v w, u = f a m ∧ v = f b m ∧ w = f c m ∧ u + v > w ∧ u + w > v ∧ v + w > u := 
sorry

end triangle_inequality_for_f_l294_294110


namespace range_of_a_l294_294568

noncomputable def has_root_in_R (f : ℝ → ℝ) : Prop :=
∃ x : ℝ, f x = 0

theorem range_of_a (a : ℝ) (h : has_root_in_R (λ x => 4 * x + a * 2^x + a + 1)) : a ≤ 0 :=
sorry

end range_of_a_l294_294568


namespace units_digit_of_product_l294_294365

theorem units_digit_of_product : 
  (3 ^ 401 * 7 ^ 402 * 23 ^ 403) % 10 = 9 := 
by
  sorry

end units_digit_of_product_l294_294365


namespace Tom_spends_375_dollars_l294_294046

noncomputable def totalCost (numBricks : ℕ) (halfDiscount : ℚ) (fullPrice : ℚ) : ℚ :=
  let halfBricks := numBricks / 2
  let discountedPrice := fullPrice * halfDiscount
  (halfBricks * discountedPrice) + (halfBricks * fullPrice)

theorem Tom_spends_375_dollars : 
  ∀ (numBricks : ℕ) (halfDiscount fullPrice : ℚ), 
  numBricks = 1000 → halfDiscount = 0.5 → fullPrice = 0.5 → totalCost numBricks halfDiscount fullPrice = 375 := 
by
  intros numBricks halfDiscount fullPrice hnumBricks hhalfDiscount hfullPrice
  rw [hnumBricks, hhalfDiscount, hfullPrice]
  sorry

end Tom_spends_375_dollars_l294_294046


namespace part1_part2_part3_l294_294437

variable {a b c : ℝ}

-- Part (1)
theorem part1 (a b c : ℝ) : a * (b - c) ^ 2 + b * (c - a) ^ 2 + c * (a - b) ^ 2 + 4 * a * b * c > a ^ 3 + b ^ 3 + c ^ 3 :=
sorry

-- Part (2)
theorem part2 (a b c : ℝ) : 2 * a ^ 2 * b ^ 2 + 2 * b ^ 2 * c ^ 2 + 2 * c ^ 2 * a ^ 2 > a ^ 4 + b ^ 4 + c ^ 4 :=
sorry

-- Part (3)
theorem part3 (a b c : ℝ) : 2 * a * b + 2 * b * c + 2 * c * a > a ^ 2 + b ^ 2 + c ^ 2 :=
sorry

end part1_part2_part3_l294_294437


namespace count_interesting_numbers_l294_294630

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l294_294630


namespace same_grades_percentage_l294_294808

theorem same_grades_percentage (total_students same_grades_A same_grades_B same_grades_C same_grades_D : ℕ) 
  (total_eq : total_students = 50) 
  (same_A : same_grades_A = 3) 
  (same_B : same_grades_B = 6) 
  (same_C : same_grades_C = 7) 
  (same_D : same_grades_D = 2) : 
  (same_grades_A + same_grades_B + same_grades_C + same_grades_D) * 100 / total_students = 36 := 
by
  sorry

end same_grades_percentage_l294_294808


namespace finding_b_for_infinite_solutions_l294_294779

theorem finding_b_for_infinite_solutions :
  ∀ b : ℝ, (∀ x : ℝ, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 :=
by
  sorry

end finding_b_for_infinite_solutions_l294_294779


namespace distribute_balls_in_boxes_l294_294273

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l294_294273


namespace largest_smallest_divisible_by_99_l294_294660

-- Definitions for distinct digits 3, 7, 9
def largest_number (x y z : Nat) : Nat := 100 * x + 10 * y + z
def smallest_number (x y z : Nat) : Nat := 100 * z + 10 * y + x

-- Proof problem statement
theorem largest_smallest_divisible_by_99 
  (a b c : Nat) (h : a > b ∧ b > c ∧ c > 0) : 
  ∃ (x y z : Nat), 
    (x = 9 ∧ y = 7 ∧ z = 3 ∧ largest_number x y z = 973 ∧ smallest_number x y z = 379) ∧
    99 ∣ (largest_number a b c - smallest_number a b c) :=
by
  sorry

end largest_smallest_divisible_by_99_l294_294660


namespace original_avg_is_40_l294_294451

noncomputable def original_average (A : ℝ) := (15 : ℝ) * A

noncomputable def new_sum (A : ℝ) := (15 : ℝ) * A + 15 * (15 : ℝ)

theorem original_avg_is_40 (A : ℝ) (h : new_sum A / 15 = 55) :
  A = 40 :=
by sorry

end original_avg_is_40_l294_294451


namespace meeting_day_correct_l294_294031

noncomputable def smallest_meeting_day :=
  ∀ (players courts : ℕ)
    (initial_reimu_court initial_marisa_court : ℕ),
    players = 2016 →
    courts = 1008 →
    initial_reimu_court = 123 →
    initial_marisa_court = 876 →
    ∀ (winner_moves_to court : ℕ → ℕ),
      (∀ (i : ℕ), 2 ≤ i ∧ i ≤ courts → winner_moves_to i = i - 1) →
      (winner_moves_to 1 = 1) →
      ∀ (loser_moves_to court : ℕ → ℕ),
        (∀ (j : ℕ), 1 ≤ j ∧ j ≤ courts - 1 → loser_moves_to j = j + 1) →
        (loser_moves_to courts = courts) →
        ∃ (n : ℕ), n = 1139

theorem meeting_day_correct : smallest_meeting_day :=
  sorry

end meeting_day_correct_l294_294031


namespace sum_of_integer_n_l294_294550

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end sum_of_integer_n_l294_294550


namespace coins_after_five_hours_l294_294510

-- Definitions of the conditions
def first_hour : ℕ := 20
def next_two_hours : ℕ := 2 * 30
def fourth_hour : ℕ := 40
def fifth_hour : ℕ := -20

-- The total number of coins calculation
def total_coins : ℕ := first_hour + next_two_hours + fourth_hour + fifth_hour

-- The theorem to be proved
theorem coins_after_five_hours : total_coins = 100 :=
by
  sorry

end coins_after_five_hours_l294_294510


namespace rectangle_diagonal_length_l294_294443

theorem rectangle_diagonal_length
    (PQ QR : ℝ) (RT RU ST : ℝ) (Area_RST : ℝ)
    (hPQ : PQ = 8) (hQR : QR = 10)
    (hRT_RU : RT = RU)
    (hArea_RST: Area_RST = (1/5) * (PQ * QR)) :
    ST = 8 :=
by
  sorry

end rectangle_diagonal_length_l294_294443


namespace earnings_difference_l294_294977

-- We define the price per bottle for each company and the number of bottles sold by each company.
def priceA : ℝ := 4
def priceB : ℝ := 3.5
def quantityA : ℕ := 300
def quantityB : ℕ := 350

-- We define the earnings for each company based on the provided conditions.
def earningsA : ℝ := priceA * quantityA
def earningsB : ℝ := priceB * quantityB

-- We state the theorem that the difference in earnings is $25.
theorem earnings_difference : (earningsB - earningsA) = 25 := by
  -- Proof omitted.
  sorry

end earnings_difference_l294_294977


namespace geometry_progressions_not_exhaust_nat_l294_294713

theorem geometry_progressions_not_exhaust_nat :
  ∃ (g : Fin 1975 → ℕ → ℕ), 
  (∀ i : Fin 1975, ∃ (a r : ℤ), ∀ n : ℕ, g i n = (a * r^n)) ∧
  (∃ m : ℕ, ∀ i : Fin 1975, ∀ n : ℕ, m ≠ g i n) :=
sorry

end geometry_progressions_not_exhaust_nat_l294_294713


namespace bottle_caps_per_visit_l294_294144

-- Define the given conditions
def total_bottle_caps : ℕ := 25
def number_of_visits : ℕ := 5

-- The statement we want to prove
theorem bottle_caps_per_visit :
  total_bottle_caps / number_of_visits = 5 :=
sorry

end bottle_caps_per_visit_l294_294144


namespace integer_solutions_count_l294_294777

theorem integer_solutions_count :
  let circle_eq : ∀ (x y : ℤ), x^2 + y^2 = 65 → Prop := 
    λ x y h, true 
  let line_eq : ∀ (a b x y : ℤ), ax + by = 2 → Prop :=
    λ a b x y h, true 
  (∃ (a b : ℤ), ∃ (x y : ℤ), line_eq a b x y (2) ∧ circle_eq x y 65) → 
  ∃ (k : ℕ), k = 128 := sorry

end integer_solutions_count_l294_294777


namespace polynomial_sum_l294_294947

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l294_294947


namespace maximum_value_of_N_l294_294596

-- Define J_k based on the conditions given
def J (k : ℕ) : ℕ := 10^(k+3) + 128

-- Define the number of factors of 2 in the prime factorization of J_k
def N (k : ℕ) : ℕ := Nat.factorization (J k) 2

-- The proposition to be proved
theorem maximum_value_of_N (k : ℕ) (hk : k > 0) : N 4 = 7 :=
by
  sorry

end maximum_value_of_N_l294_294596


namespace nancy_shoes_l294_294465

theorem nancy_shoes (boots slippers heels : ℕ) 
  (h₀ : boots = 6)
  (h₁ : slippers = boots + 9)
  (h₂ : heels = 3 * (boots + slippers)) :
  2 * (boots + slippers + heels) = 168 := by
  sorry

end nancy_shoes_l294_294465


namespace minimize_expression_l294_294118

theorem minimize_expression (x : ℝ) (h : 0 < x) : 
  x = 9 ↔ (∀ y : ℝ, 0 < y → x + 81 / x ≤ y + 81 / y) :=
sorry

end minimize_expression_l294_294118


namespace two_digit_integers_leaving_remainder_3_div_7_count_l294_294623

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l294_294623


namespace minimum_value_of_polynomial_l294_294299

def polynomial (a b : ℝ) : ℝ := 2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999

theorem minimum_value_of_polynomial : ∃ (a b : ℝ), polynomial a b = 1947 :=
by
  sorry

end minimum_value_of_polynomial_l294_294299


namespace max_mn_value_l294_294659

-- Define the function f(x)
def f (m n : ℝ) (x : ℝ) : ℝ := (1/2) * (2 - m) * x^2 + (n - 8) * x + 1

-- Define the condition on m
axiom m_pos (m: ℝ) : m > 2

-- Define the condition of monotonic decreasing
axiom f_mon_decreasing (m n : ℝ) (m_pos : m > 2) : 
  (∀ x ∈ set.Icc (-2 : ℝ) (-1 : ℝ), deriv (f m n) x ≤ 0)

-- The theorem we aim to prove: the maximum value of mn is 18 given the conditions
theorem max_mn_value (m n : ℝ) (h₁ : m > 2) (h₂ : ∀ x ∈ set.Icc (-2 : ℝ) (-1 : ℝ), deriv (f m n) x ≤ 0) :
  mn ≤ 18 :=
by sorry

end max_mn_value_l294_294659


namespace distribute_balls_in_boxes_l294_294272

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l294_294272


namespace probability_sum_less_than_product_l294_294536

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l294_294536


namespace sum_of_valid_n_l294_294560

theorem sum_of_valid_n : 
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)] in
  (n_values.sum = 3) :=
by
  -- Define the values of n according to the problem's conditions
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)],
  -- Proof will be filled in here
  sorry

end sum_of_valid_n_l294_294560


namespace flowers_bloom_l294_294042

theorem flowers_bloom (num_unicorns : ℕ) (flowers_per_step : ℕ) (distance_km : ℕ) (step_length_m : ℕ) 
  (h1 : num_unicorns = 6) (h2 : flowers_per_step = 4) (h3 : distance_km = 9) (h4 : step_length_m = 3) : 
  num_unicorns * (distance_km * 1000 / step_length_m) * flowers_per_step = 72000 :=
by
  sorry

end flowers_bloom_l294_294042


namespace twin_primes_solution_l294_294764

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def are_twin_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ (p = q + 2 ∨ q = p + 2)

theorem twin_primes_solution (p q : ℕ) :
  are_twin_primes p q ∧ is_prime (p^2 - p * q + q^2) ↔ (p, q) = (5, 3) ∨ (p, q) = (3, 5) := by
  sorry

end twin_primes_solution_l294_294764


namespace polynomial_sum_l294_294159

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3
def g (x : ℝ) : ℝ := -3 * x^2 + 7 * x - 6
def h (x : ℝ) : ℝ := 3 * x^2 - 3 * x + 2
def j (x : ℝ) : ℝ := x^2 + x - 1

theorem polynomial_sum (x : ℝ) : f x + g x + h x + j x = 3 * x^2 + x - 2 := by
  sorry

end polynomial_sum_l294_294159


namespace prime_square_pairs_l294_294771

theorem prime_square_pairs (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    ∃ n : Nat, p^2 + 5 * p * q + 4 * q^2 = n^2 ↔ (p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11) ∨ (p = 3 ∧ q = 13) ∨ (p = 5 ∧ q = 7) ∨ (p = 11 ∧ q = 5) :=
by
  sorry

end prime_square_pairs_l294_294771


namespace last_integer_in_sequence_is_21853_l294_294707

def is_divisible_by (n m : ℕ) : Prop := 
  ∃ k : ℕ, n = m * k

-- Conditions
def starts_with : ℕ := 590049
def divides_previous (a b : ℕ) : Prop := b = a / 3

-- The target hypothesis to prove
theorem last_integer_in_sequence_is_21853 :
  ∀ (a b c d : ℕ),
    a = starts_with →
    divides_previous a b →
    divides_previous b c →
    divides_previous c d →
    ¬ is_divisible_by d 3 →
    d = 21853 :=
by
  intros a b c d ha hb hc hd hnd
  sorry

end last_integer_in_sequence_is_21853_l294_294707


namespace intersection_unique_point_x_coordinate_l294_294297

theorem intersection_unique_point_x_coordinate (a b : ℝ) (h : a ≠ b) : 
  (∃ x y : ℝ, y = x^2 + 2*a*x + 6*b ∧ y = x^2 + 2*b*x + 6*a) → ∃ x : ℝ, x = 3 :=
by
  sorry

end intersection_unique_point_x_coordinate_l294_294297


namespace bounded_sequence_exists_l294_294063

noncomputable def positive_sequence := ℕ → ℝ

variables {a : positive_sequence}

axiom positive_sequence_pos (n : ℕ) : 0 < a n

axiom sequence_condition (k n m l : ℕ) (h : k + n = m + l) : 
  (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)

theorem bounded_sequence_exists 
  (a : positive_sequence) 
  (h_pos : ∀ n, 0 < a n)
  (h_cond : ∀ (k n m l : ℕ), k + n = m + l → 
              (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ (b c : ℝ), (0 < b) ∧ (0 < c) ∧ (∀ n, b ≤ a n ∧ a n ≤ c) :=
sorry

end bounded_sequence_exists_l294_294063


namespace polynomial_sum_l294_294949

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l294_294949


namespace number_of_boys_in_other_communities_l294_294372

-- Definitions from conditions
def total_boys : ℕ := 700
def percentage_muslims : ℕ := 44
def percentage_hindus : ℕ := 28
def percentage_sikhs : ℕ := 10

-- Proof statement
theorem number_of_boys_in_other_communities : 
  (700 * (100 - (44 + 28 + 10)) / 100) = 126 := 
by
  sorry

end number_of_boys_in_other_communities_l294_294372


namespace problem_solution_l294_294678

theorem problem_solution :
  ∀ p q : ℝ, (3 * p ^ 2 - 5 * p - 21 = 0) → (3 * q ^ 2 - 5 * q - 21 = 0) →
  (9 * p ^ 3 - 9 * q ^ 3) * (p - q)⁻¹ = 88 :=
by 
  sorry

end problem_solution_l294_294678


namespace remainder_of_product_l294_294714

theorem remainder_of_product (a b c : ℕ) (h₁ : a % 7 = 3) (h₂ : b % 7 = 4) (h₃ : c % 7 = 5) :
  (a * b * c) % 7 = 4 :=
by
  sorry

end remainder_of_product_l294_294714


namespace units_digit_G_n_for_n_eq_3_l294_294366

def G (n : ℕ) : ℕ := 2 ^ 2 ^ 2 ^ n + 1

theorem units_digit_G_n_for_n_eq_3 : (G 3) % 10 = 7 := 
by 
  sorry

end units_digit_G_n_for_n_eq_3_l294_294366


namespace sum_of_squares_l294_294609

theorem sum_of_squares (x y z w a b c d : ℝ) (h1: x * y = a) (h2: x * z = b) (h3: y * z = c) (h4: x * w = d) :
  x^2 + y^2 + z^2 + w^2 = (ab + bd + da)^2 / abd := 
by
  sorry

end sum_of_squares_l294_294609


namespace find_a_b_l294_294123

noncomputable def f (x : ℝ) (a b : ℝ) := x^3 + a * x + b

theorem find_a_b 
  (a b : ℝ) 
  (h_tangent : ∀ x y, y = 2 * x - 5 → y = f 1 a b - 3) 
  : a = -1 ∧ b = -3 :=
by 
{
  sorry
}

end find_a_b_l294_294123


namespace arithmetic_sequence_range_of_k_l294_294122

def T (a_n : ℕ → ℝ) (n : ℕ) : ℝ := 1 - a_n n
def c (a_n : ℕ → ℝ) (n : ℕ) : ℝ := 1 / (T a_n n)

theorem arithmetic_sequence (a_n : ℕ → ℝ) (n : ℕ) :
  (∀ n, T a_n n = 1 - a_n n) → (∀ n, c a_n n = n + 1) ∧ 
  (∀ n, a_n n = n / (n + 1)) :=
sorry

def b (n : ℕ) : ℝ := 1 / (2^n)
def T' (n : ℕ) : ℝ := 1 / (n + 1)

theorem range_of_k (k : ℝ) :
  (∀ n, n ∈ ℕ → T' n * (n * b n + n - 2) ≤ k * n) → k ≥ 11 / 96 :=
sorry

end arithmetic_sequence_range_of_k_l294_294122


namespace elder_person_age_l294_294836

-- Definitions based on conditions
variables (y e : ℕ) 

-- Given conditions
def condition1 : Prop := e = y + 20
def condition2 : Prop := e - 5 = 5 * (y - 5)

-- Theorem stating the required proof problem
theorem elder_person_age (h1 : condition1 y e) (h2 : condition2 y e) : e = 30 :=
by
  sorry

end elder_person_age_l294_294836


namespace probability_sum_less_than_product_l294_294519

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l294_294519


namespace trapezoid_ratio_l294_294196

noncomputable theory

-- Definitions based on the given conditions
variables {EF FG GH HE : ℕ}
variable {Q : (fin 2) → ℝ} -- point Q has coordinates in 2D space
variables {x : ℝ} -- let EQ = x
variables {p q : ℕ} -- relatively prime positive integers p and q

-- Assume the conditions
def trapezoid_conditions : Prop :=
  EF = 86 ∧ FG = 38 ∧ GH = 23 ∧ HE = 66 ∧ (∃ Q : (fin 2) → ℝ, are_parallel EF GH ∧ tangent_to Q FG ∧ tangent_to Q HE)

-- Prove the ratio EQ:QF and p + q = 2879
theorem trapezoid_ratio : trapezoid_conditions → ∃ p q : ℕ, (p / q) = (2832 / 47) ∧ p + q = 2879 :=
by
  intros h
  sorry

end trapezoid_ratio_l294_294196


namespace david_age_l294_294057

theorem david_age (x : ℕ) (y : ℕ) (h1 : y = x + 7) (h2 : y = 2 * x) : x = 7 :=
by
  sorry

end david_age_l294_294057


namespace trains_cross_time_l294_294202

noncomputable def timeToCrossEachOther (L : ℝ) (T1 : ℝ) (T2 : ℝ) : ℝ :=
  let V1 := L / T1
  let V2 := L / T2
  let Vr := V1 + V2
  let totalDistance := L + L
  totalDistance / Vr

theorem trains_cross_time (L T1 T2 : ℝ) (hL : L = 120) (hT1 : T1 = 10) (hT2 : T2 = 15) :
  timeToCrossEachOther L T1 T2 = 12 :=
by
  simp [timeToCrossEachOther, hL, hT1, hT2]
  sorry

end trains_cross_time_l294_294202


namespace p_sufficient_not_necessary_l294_294416

theorem p_sufficient_not_necessary:
  (∀ a b : ℝ, a > b ∧ b > 0 → (1 / a^2 < 1 / b^2)) ∧ 
  (∃ a b : ℝ, (1 / a^2 < 1 / b^2) ∧ ¬ (a > b ∧ b > 0)) :=
sorry

end p_sufficient_not_necessary_l294_294416


namespace combined_area_of_triangles_l294_294189

noncomputable def area_of_rectangle (length width : ℝ) : ℝ :=
  length * width

noncomputable def first_triangle_area (x : ℝ) : ℝ :=
  5 * x

noncomputable def second_triangle_area (base height : ℝ) : ℝ :=
  (base * height) / 2

theorem combined_area_of_triangles (length width x base height : ℝ)
  (h1 : area_of_rectangle length width / first_triangle_area x = 2 / 5)
  (h2 : base + height = 20)
  (h3 : second_triangle_area base height / first_triangle_area x = 3 / 5)
  (length_value : length = 6)
  (width_value : width = 4)
  (base_value : base = 8) :
  first_triangle_area x + second_triangle_area base height = 108 := 
by
  sorry

end combined_area_of_triangles_l294_294189


namespace calculator_display_after_50_presses_l294_294835

theorem calculator_display_after_50_presses :
  let initial_display := 3
  let operation (x : ℚ) := 1 / (1 - x)
  (Nat.iterate operation 50 initial_display) = 2 / 3 :=
by
  sorry

end calculator_display_after_50_presses_l294_294835


namespace opposite_of_two_is_negative_two_l294_294035

theorem opposite_of_two_is_negative_two : -2 = -2 :=
by
  sorry

end opposite_of_two_is_negative_two_l294_294035


namespace sum_f_values_l294_294245

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2 / x) + 1

theorem sum_f_values : 
  f (-7) + f (-5) + f (-3) + f (-1) + f (3) + f (5) + f (7) + f (9) = 8 := 
by
  sorry

end sum_f_values_l294_294245


namespace conference_games_l294_294342

/-- 
Two divisions of 8 teams each, where each team plays 21 games within its division 
and 8 games against the teams of the other division. 
Prove total number of scheduled conference games is 232.
-/
theorem conference_games (div_teams : ℕ) (intra_div_games : ℕ) (inter_div_games : ℕ) (total_teams : ℕ) :
  div_teams = 8 →
  intra_div_games = 21 →
  inter_div_games = 8 →
  total_teams = 2 * div_teams →
  (total_teams * (intra_div_games + inter_div_games)) / 2 = 232 :=
by
  intros
  sorry


end conference_games_l294_294342


namespace manufacturing_percentage_l294_294343

theorem manufacturing_percentage (a b : ℕ) (h1 : a = 108) (h2 : b = 360) : (a / b : ℚ) * 100 = 30 :=
by
  sorry

end manufacturing_percentage_l294_294343


namespace find_unknown_number_l294_294656

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end find_unknown_number_l294_294656


namespace intersection_eq_l294_294784

-- Definitions of sets A and B
def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

-- The theorem statement
theorem intersection_eq : A ∩ B = {1} :=
by
  unfold A B
  sorry

end intersection_eq_l294_294784


namespace total_savings_l294_294016

theorem total_savings :
  let J := 0.25 in
  let D_J := 24 in
  let L := 0.50 in
  let D_L := 20 in
  let M := 2 * L in
  let D_M := 12 in
  J * D_J + L * D_L + M * D_M = 28.00 :=
by 
  sorry

end total_savings_l294_294016


namespace num_ordered_pairs_eq_1728_l294_294037

theorem num_ordered_pairs_eq_1728 (x y : ℕ) (h1 : 1728 = 2^6 * 3^3) (h2 : x * y = 1728) : 
  ∃ (n : ℕ), n = 28 := 
sorry

end num_ordered_pairs_eq_1728_l294_294037


namespace polynomial_sum_l294_294946

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l294_294946


namespace total_sum_of_rupees_l294_294852

theorem total_sum_of_rupees :
  ∃ (total_coins : ℕ) (paise20_coins : ℕ) (paise25_coins : ℕ),
    total_coins = 344 ∧ paise20_coins = 300 ∧ paise25_coins = total_coins - paise20_coins ∧
    (60 + (44 * 0.25)) = 71 :=
by
  sorry

end total_sum_of_rupees_l294_294852


namespace inequality_sqrt_a_b_c_l294_294960

noncomputable def sqrt (x : ℝ) := x ^ (1 / 2)

theorem inequality_sqrt_a_b_c (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  sqrt (a ^ (1 - a) * b ^ (1 - b) * c ^ (1 - c)) ≤ 1 / 3 := 
sorry

end inequality_sqrt_a_b_c_l294_294960


namespace maximize_parabola_area_l294_294900

variable {a b : ℝ}

/--
The parabola y = ax^2 + bx is tangent to the line x + y = 4 within the first quadrant. 
Prove that the values of a and b that maximize the area S enclosed by this parabola and 
the x-axis are a = -1 and b = 3, and that the maximum value of S is 9/2.
-/
theorem maximize_parabola_area (hab_tangent : ∃ x y, y = a * x^2 + b * x ∧ y = 4 - x ∧ x > 0 ∧ y > 0) 
  (area_eqn : S = 1/6 * (b^3 / a^2)) : 
  a = -1 ∧ b = 3 ∧ S = 9/2 := 
sorry

end maximize_parabola_area_l294_294900


namespace average_marks_for_class_l294_294491

theorem average_marks_for_class (total_students : ℕ) (marks_group1 marks_group2 marks_group3 : ℕ) (num_students_group1 num_students_group2 num_students_group3 : ℕ) 
  (h1 : total_students = 50) 
  (h2 : num_students_group1 = 10) 
  (h3 : marks_group1 = 90) 
  (h4 : num_students_group2 = 15) 
  (h5 : marks_group2 = 80) 
  (h6 : num_students_group3 = total_students - num_students_group1 - num_students_group2) 
  (h7 : marks_group3 = 60) : 
  (10 * 90 + 15 * 80 + (total_students - 10 - 15) * 60) / total_students = 72 := 
by
  sorry

end average_marks_for_class_l294_294491


namespace compute_expression_l294_294229

theorem compute_expression : 7^2 - 2 * 5 + 4^2 / 2 = 47 := by
  sorry

end compute_expression_l294_294229


namespace multiplier_for_ab_to_equal_1800_l294_294643

variable (a b m : ℝ)
variable (h1 : 4 * a = 30)
variable (h2 : 5 * b = 30)
variable (h3 : a * b = 45)
variable (h4 : m * (a * b) = 1800)

theorem multiplier_for_ab_to_equal_1800 (h1 : 4 * a = 30) (h2 : 5 * b = 30) (h3 : a * b = 45) (h4 : m * (a * b) = 1800) :
  m = 40 :=
sorry

end multiplier_for_ab_to_equal_1800_l294_294643


namespace abs_inequality_solution_set_l294_294190

theorem abs_inequality_solution_set (x : ℝ) :
  |x| + |x - 1| < 2 ↔ - (1 / 2) < x ∧ x < (3 / 2) :=
by
  sorry

end abs_inequality_solution_set_l294_294190


namespace total_bowling_balls_l294_294180

theorem total_bowling_balls (red_balls : ℕ) (green_balls : ℕ) (h1 : red_balls = 30) (h2 : green_balls = red_balls + 6) : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l294_294180


namespace bags_sold_in_first_week_l294_294068

def total_bags_sold : ℕ := 100
def bags_sold_week1 (X : ℕ) : ℕ := X
def bags_sold_week2 (X : ℕ) : ℕ := 3 * X
def bags_sold_week3_4 : ℕ := 40

theorem bags_sold_in_first_week (X : ℕ) (h : total_bags_sold = bags_sold_week1 X + bags_sold_week2 X + bags_sold_week3_4) : X = 15 :=
by
  sorry

end bags_sold_in_first_week_l294_294068


namespace polynomial_sum_l294_294952

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l294_294952


namespace greatest_possible_median_l294_294566

theorem greatest_possible_median : 
  ∀ (k m r s t : ℕ),
    k < m → m < r → r < s → s < t →
    (k + m + r + s + t = 90) →
    (t = 40) →
    (r = 23) :=
by
  intros k m r s t h1 h2 h3 h4 h_sum h_t
  sorry

end greatest_possible_median_l294_294566


namespace sum_of_n_l294_294558

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end sum_of_n_l294_294558


namespace sum_series_eq_eight_l294_294092

noncomputable def sum_series : ℝ := ∑' n : ℕ, (3 * (n + 1) + 2) / 2^(n + 1)

theorem sum_series_eq_eight : sum_series = 8 := 
 by
  sorry

end sum_series_eq_eight_l294_294092


namespace simplify_and_evaluate_l294_294172

theorem simplify_and_evaluate : 
  ∀ (x y : ℚ), x = 1 / 2 → y = 2 / 3 →
  ((x - 2 * y)^2 + (x - 2 * y) * (x + 2 * y) - 3 * x * (2 * x - y)) / (2 * x) = -4 / 3 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_l294_294172


namespace repeating_decimal_value_l294_294435

def repeating_decimal : ℝ := 0.0000253253325333 -- Using repeating decimal as given in the conditions

theorem repeating_decimal_value :
  (10^7 - 10^5) * repeating_decimal = 253 / 990 :=
sorry

end repeating_decimal_value_l294_294435


namespace charles_cleaning_time_l294_294396

theorem charles_cleaning_time :
  let Alice_time := 20
  let Bob_time := (3/4) * Alice_time
  let Charles_time := (2/3) * Bob_time
  Charles_time = 10 :=
by
  sorry

end charles_cleaning_time_l294_294396


namespace twin_brothers_age_l294_294200

theorem twin_brothers_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 17) : x = 8 := 
  sorry

end twin_brothers_age_l294_294200


namespace geometric_series_sum_l294_294755

theorem geometric_series_sum :
  ∀ (a r : ℤ) (n : ℕ),
  a = 3 → r = -2 → n = 10 →
  (a * ((r ^ n - 1) / (r - 1))) = -1024 :=
by
  intros a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l294_294755


namespace monotonic_increasing_interval_l294_294351

noncomputable def log_base_1_div_3 (t : ℝ) := Real.log t / Real.log (1/3)

def quadratic (x : ℝ) := 4 + 3 * x - x^2

theorem monotonic_increasing_interval :
  ∃ (a b : ℝ), (∀ x, a < x ∧ x < b → (log_base_1_div_3 (quadratic x)) < (log_base_1_div_3 (quadratic (x + ε))) ∧
               ((-1 : ℝ) < x ∧ x < 4) ∧ (quadratic x > 0)) ↔ (a, b) = (3 / 2, 4) :=
by
  sorry

end monotonic_increasing_interval_l294_294351


namespace find_f_prime_2_l294_294258

theorem find_f_prime_2 (a : ℝ) (f' : ℝ → ℝ) 
    (h1 : f' 1 = -5)
    (h2 : ∀ x, f' x = 3 * a * x^2 + 2 * f' 2 * x) : f' 2 = -4 := by
    sorry

end find_f_prime_2_l294_294258


namespace nancy_shoes_l294_294461

theorem nancy_shoes (boots_slippers_relation : ∀ (boots slippers : ℕ), slippers = boots + 9)
                    (heels_relation : ∀ (boots slippers heels : ℕ), heels = 3 * (boots + slippers)) :
                    ∃ (total_individual_shoes : ℕ), total_individual_shoes = 168 :=
by
  let boots := 6
  let slippers := boots + 9
  let total_pairs := boots + slippers
  let heels := 3 * total_pairs
  let total_pairs_shoes := boots + slippers + heels
  let total_individual_shoes := 2 * total_pairs_shoes
  use total_individual_shoes
  exact sorry

end nancy_shoes_l294_294461


namespace tangent_line_at_0_l294_294603

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x + Real.sin x

theorem tangent_line_at_0 :
  ∃ (m b : ℝ), ∀ (x : ℝ), f 0 = 1 ∧ (f' : ℝ → ℝ) 0 = 1 ∧ (f' x = Real.exp x + 2 * x - 1 + Real.cos x) ∧ 
  (m = 1) ∧ (b = (m * 0 + 1)) ∧ (∀ x : ℝ, y = m * x + b) :=
by
  sorry

end tangent_line_at_0_l294_294603


namespace largest_area_of_rotating_triangle_l294_294001

def Point := (ℝ × ℝ)

def A : Point := (0, 0)
def B : Point := (13, 0)
def C : Point := (21, 0)

def line (P : Point) (slope : ℝ) (x : ℝ) : ℝ := P.2 + slope * (x - P.1)

def l_A (x : ℝ) : ℝ := line A 1 x
def l_B (x : ℝ) : ℝ := x
def l_C (x : ℝ) : ℝ := line C (-1) x

def rotating_triangle_max_area (l_A l_B l_C : ℝ → ℝ) : ℝ := 116.5

theorem largest_area_of_rotating_triangle :
  rotating_triangle_max_area l_A l_B l_C = 116.5 :=
sorry

end largest_area_of_rotating_triangle_l294_294001


namespace exists_four_distinct_numbers_with_equal_half_sum_l294_294247

theorem exists_four_distinct_numbers_with_equal_half_sum (S : Finset ℕ) (h_card : S.card = 10) (h_range : ∀ x ∈ S, x ≤ 23) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S) ∧ (a + b = c + d) :=
by
  sorry

end exists_four_distinct_numbers_with_equal_half_sum_l294_294247


namespace find_expression_for_f_l294_294899

theorem find_expression_for_f (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2 + 6 * x) :
  ∀ x, f x = x^2 + 8 * x + 7 :=
by
  sorry

end find_expression_for_f_l294_294899


namespace distance_between_city_centers_l294_294840

theorem distance_between_city_centers :
  let distance_on_map_cm := 55
  let scale_cm_to_km := 30
  let km_to_m := 1000
  (distance_on_map_cm * scale_cm_to_km * km_to_m) = 1650000 :=
by
  sorry

end distance_between_city_centers_l294_294840


namespace find_s_for_g3_eq_0_l294_294957

def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + x^3 - 4 * x^2 + 5 * x + s

theorem find_s_for_g3_eq_0 : (g 3 s = 0) ↔ (s = -573) :=
by
  sorry

end find_s_for_g3_eq_0_l294_294957


namespace complex_modulus_z_l294_294422

-- Define the complex number z with given conditions
noncomputable def z : ℂ := (2 + Complex.I) / Complex.I + Complex.I

-- State the theorem to be proven
theorem complex_modulus_z : Complex.abs z = Real.sqrt 2 := 
sorry

end complex_modulus_z_l294_294422


namespace compare_three_and_negfour_l294_294754

theorem compare_three_and_negfour : 3 > -4 := by
  sorry

end compare_three_and_negfour_l294_294754


namespace bus_speed_excluding_stoppages_l294_294768

noncomputable def average_speed_excluding_stoppages
  (speed_including_stoppages : ℝ)
  (stoppage_time_ratio : ℝ) : ℝ :=
  (speed_including_stoppages * 1) / (1 - stoppage_time_ratio)

theorem bus_speed_excluding_stoppages :
  average_speed_excluding_stoppages 15 (3/4) = 60 := 
by
  sorry

end bus_speed_excluding_stoppages_l294_294768


namespace prob_diff_colors_with_replacement_expectation_variance_white_balls_without_replacement_l294_294507

-- (I) Probability of drawing two balls of different colors with replacement
theorem prob_diff_colors_with_replacement 
  (white_balls black_balls : ℕ) 
  (h_white : white_balls = 2) 
  (h_black : black_balls = 3) :
  let total_balls := white_balls + black_balls in
  let prob_white := (white_balls : ℝ) / total_balls in
  let prob_black := (black_balls : ℝ) / total_balls in
  (prob_white * prob_black + prob_black * prob_white = 12 / 25) :=
begin
  sorry
end

-- (II) Expectation and variance of the number of white balls drawn without replacement
theorem expectation_variance_white_balls_without_replacement
  (white_balls black_balls : ℕ) 
  (h_white : white_balls = 2) 
  (h_black : black_balls = 3) :
  let total_balls := white_balls + black_balls in
  let prob_0_white := (black_balls : ℝ) / total_balls * (black_balls - 1) / (total_balls - 1) in
  let prob_1_white := (black_balls : ℝ) / total_balls * white_balls / (total_balls - 1) + 
                      (white_balls : ℝ) / total_balls * black_balls / (total_balls - 1) in
  let prob_2_white := (white_balls : ℝ) / total_balls * (white_balls - 1) / (total_balls - 1) in
  let E_xi := 0 * prob_0_white + 1 * prob_1_white + 2 * prob_2_white in
  let D_xi := (0 - E_xi)^2 * prob_0_white + (1 - E_xi)^2 * prob_1_white + (2 - E_xi)^2 * prob_2_white in
  (E_xi = 4 / 5) ∧ (D_xi = 9 / 25) :=
begin
  sorry
end

end prob_diff_colors_with_replacement_expectation_variance_white_balls_without_replacement_l294_294507


namespace slope_angle_of_line_x_equal_one_l294_294805

noncomputable def slope_angle_of_vertical_line : ℝ := 90

theorem slope_angle_of_line_x_equal_one : slope_angle_of_vertical_line = 90 := by
  sorry

end slope_angle_of_line_x_equal_one_l294_294805


namespace mobile_price_two_years_ago_l294_294857

-- Definitions and conditions
def price_now : ℝ := 1000
def decrease_rate : ℝ := 0.2
def years_ago : ℝ := 2

-- Main statement
theorem mobile_price_two_years_ago :
  ∃ (a : ℝ), a * (1 - decrease_rate)^years_ago = price_now :=
sorry

end mobile_price_two_years_ago_l294_294857


namespace solve_inequality_system_l294_294743

theorem solve_inequality_system (y : ℝ) :
  (2 * (y + 1) < 5 * y - 7) ∧ ((y + 2) / 2 < 5) ↔ (3 < y) ∧ (y < 8) := 
by
  sorry

end solve_inequality_system_l294_294743


namespace ratio_clara_alice_pens_l294_294082

def alice_pens := 60
def alice_age := 20
def clara_future_age := 61

def clara_age := clara_future_age - 5
def age_difference := clara_age - alice_age

def clara_pens := alice_pens - age_difference

theorem ratio_clara_alice_pens :
  clara_pens / alice_pens = 2 / 5 :=
by
  -- The addressed proof will be here.
  sorry

end ratio_clara_alice_pens_l294_294082


namespace largest_m_dividing_factorial_l294_294891

theorem largest_m_dividing_factorial :
  (∃ m : ℕ, (∀ n : ℕ, (18^n ∣ 30!) ↔ n ≤ m) ∧ m = 7) :=
by
  sorry

end largest_m_dividing_factorial_l294_294891


namespace find_number_l294_294649

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end find_number_l294_294649


namespace find_k_when_lines_perpendicular_l294_294910

theorem find_k_when_lines_perpendicular (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (3-k) * y + 1 = 0 → ∀ x y : ℝ, 2 * (k-3) * x - 2 * y + 3 = 0 → -((k-3)/(3-k)) * (k-3) = -1) → 
  k = 2 :=
by
  sorry

end find_k_when_lines_perpendicular_l294_294910


namespace avg_difference_l294_294346

theorem avg_difference : 
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 80 + 15) / 3
  avg1 - avg2 = 5 :=
by
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 80 + 15) / 3
  show avg1 - avg2 = 5
  sorry

end avg_difference_l294_294346


namespace inequality_holds_for_all_x_l294_294893

theorem inequality_holds_for_all_x (m : ℝ) (h : ∀ x : ℝ, |x + 5| ≥ m + 2) : m ≤ -2 :=
sorry

end inequality_holds_for_all_x_l294_294893


namespace problem_circumscribing_sphere_surface_area_l294_294114

noncomputable def surface_area_of_circumscribing_sphere (a b c : ℕ) :=
  let R := (Real.sqrt (a^2 + b^2 + c^2)) / 2
  4 * Real.pi * R^2

theorem problem_circumscribing_sphere_surface_area
  (a b c : ℕ)
  (ha : (1 / 2 : ℝ) * a * b = 4)
  (hb : (1 / 2 : ℝ) * b * c = 6)
  (hc : (1 / 2: ℝ) * a * c = 12) : 
  surface_area_of_circumscribing_sphere a b c = 56 * Real.pi := 
sorry

end problem_circumscribing_sphere_surface_area_l294_294114


namespace first_chinese_supercomputer_is_milkyway_l294_294664

-- Define the names of the computers
inductive ComputerName
| Universe
| Taihu
| MilkyWay
| Dawn

-- Define a structure to hold the properties of the computer
structure Computer :=
  (name : ComputerName)
  (introduction_year : Nat)
  (calculations_per_second : Nat)

-- Define the properties of the specific computer in the problem
def first_chinese_supercomputer := 
  Computer.mk ComputerName.MilkyWay 1983 100000000

-- The theorem to be proven
theorem first_chinese_supercomputer_is_milkyway :
  first_chinese_supercomputer.name = ComputerName.MilkyWay :=
by
  -- Provide the conditions that lead to the conclusion (proof steps will be added here)
  sorry

end first_chinese_supercomputer_is_milkyway_l294_294664


namespace nancy_shoes_l294_294462

theorem nancy_shoes (boots_slippers_relation : ∀ (boots slippers : ℕ), slippers = boots + 9)
                    (heels_relation : ∀ (boots slippers heels : ℕ), heels = 3 * (boots + slippers)) :
                    ∃ (total_individual_shoes : ℕ), total_individual_shoes = 168 :=
by
  let boots := 6
  let slippers := boots + 9
  let total_pairs := boots + slippers
  let heels := 3 * total_pairs
  let total_pairs_shoes := boots + slippers + heels
  let total_individual_shoes := 2 * total_pairs_shoes
  use total_individual_shoes
  exact sorry

end nancy_shoes_l294_294462


namespace jack_needs_more_money_l294_294929

variable (cost_per_pair_of_socks : ℝ := 9.50)
variable (number_of_pairs_of_socks : ℕ := 2)
variable (cost_of_soccer_shoes : ℝ := 92.00)
variable (jack_money : ℝ := 40.00)

theorem jack_needs_more_money :
  let total_cost := number_of_pairs_of_socks * cost_per_pair_of_socks + cost_of_soccer_shoes
  let money_needed := total_cost - jack_money
  money_needed = 71.00 := by
  sorry

end jack_needs_more_money_l294_294929


namespace import_tax_l294_294367

theorem import_tax (total_value : ℝ) (tax_rate : ℝ) (excess_limit : ℝ) (correct_tax : ℝ)
  (h1 : total_value = 2560) (h2 : tax_rate = 0.07) (h3 : excess_limit = 1000) : 
  correct_tax = tax_rate * (total_value - excess_limit) :=
by
  sorry

end import_tax_l294_294367


namespace geometric_series_sum_l294_294759

theorem geometric_series_sum :
  let a := 3
  let r := -2
  let n := 10
  let S := a * ((r^n - 1) / (r - 1))
  S = -1023 :=
by 
  -- Sorry allows us to omit the proof details
  sorry

end geometric_series_sum_l294_294759


namespace probability_sum_less_than_product_l294_294521

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l294_294521


namespace lamp_probability_l294_294715

theorem lamp_probability (rope_length : ℝ) (pole_distance : ℝ) (h_pole_distance : pole_distance = 8) :
  let lamp_range := 2
  let favorable_segment_length := 4
  let total_rope_length := rope_length
  let probability := (favorable_segment_length / total_rope_length)
  rope_length = 8 → probability = 1 / 2 :=
by
  intros
  sorry

end lamp_probability_l294_294715


namespace cindy_correct_answer_l294_294881

/-- 
Cindy accidentally first subtracted 9 from a number, then multiplied the result 
by 2 before dividing by 6, resulting in an answer of 36. 
Following these steps, she was actually supposed to subtract 12 from the 
number and then divide by 8. What would her answer have been had she worked the 
problem correctly?
-/
theorem cindy_correct_answer :
  ∀ (x : ℝ), (2 * (x - 9) / 6 = 36) → ((x - 12) / 8 = 13.125) :=
by
  intro x
  sorry

end cindy_correct_answer_l294_294881


namespace invitational_tournament_l294_294512

theorem invitational_tournament (x : ℕ) (h : 2 * (x * (x - 1) / 2) = 56) : x = 8 :=
by
  sorry

end invitational_tournament_l294_294512


namespace geometric_series_sum_l294_294401

def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  sum_geometric_series (1/4) (1/4) 7 = 4/3 :=
by
  -- Proof is omitted
  sorry

end geometric_series_sum_l294_294401


namespace total_cupcakes_l294_294205

-- Definitions of initial conditions
def cupcakes_initial : ℕ := 42
def cupcakes_sold : ℕ := 22
def cupcakes_made_after : ℕ := 39

-- Proof statement: Total number of cupcakes Robin would have
theorem total_cupcakes : 
  (cupcakes_initial - cupcakes_sold + cupcakes_made_after) = 59 := by
    sorry

end total_cupcakes_l294_294205


namespace probability_even_sum_l294_294480

def p_even_first_wheel : ℚ := 1 / 3
def p_odd_first_wheel : ℚ := 2 / 3
def p_even_second_wheel : ℚ := 3 / 5
def p_odd_second_wheel : ℚ := 2 / 5

theorem probability_even_sum : 
  (p_even_first_wheel * p_even_second_wheel) + (p_odd_first_wheel * p_odd_second_wheel) = 7 / 15 :=
by
  sorry

end probability_even_sum_l294_294480


namespace ratio_of_cars_to_trucks_l294_294673

-- Definitions based on conditions
def total_vehicles : ℕ := 60
def trucks : ℕ := 20
def cars : ℕ := total_vehicles - trucks

-- Theorem to prove
theorem ratio_of_cars_to_trucks : (cars / trucks : ℚ) = 2 := by
  sorry

end ratio_of_cars_to_trucks_l294_294673


namespace inequality_proof_l294_294025

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a + b + c ≤ (a^2 + b^2) / (2 * c) + (a^2 + c^2) / (2 * b) + (b^2 + c^2) / (2 * a) ∧ 
    (a^2 + b^2) / (2 * c) + (a^2 + c^2) / (2 * b) + (b^2 + c^2) / (2 * a) ≤ (a^3 / (b * c)) + (b^3 / (a * c)) + (c^3 / (a * b)) := 
by
  sorry

end inequality_proof_l294_294025


namespace janet_total_distance_l294_294141

-- Define the distances covered in each week for each activity
def week1_running := 8 * 5
def week1_cycling := 7 * 3

def week2_running := 10 * 4
def week2_swimming := 2 * 2

def week3_running := 6 * 5
def week3_hiking := 3 * 2

-- Total distances for each activity
def total_running := week1_running + week2_running + week3_running
def total_cycling := week1_cycling
def total_swimming := week2_swimming
def total_hiking := week3_hiking

-- Total distance covered
def total_distance := total_running + total_cycling + total_swimming + total_hiking

-- Prove that the total distance is 141 miles
theorem janet_total_distance : total_distance = 141 := by
  sorry

end janet_total_distance_l294_294141


namespace average_marks_for_class_l294_294492

theorem average_marks_for_class (total_students : ℕ) (marks_group1 marks_group2 marks_group3 : ℕ) (num_students_group1 num_students_group2 num_students_group3 : ℕ) 
  (h1 : total_students = 50) 
  (h2 : num_students_group1 = 10) 
  (h3 : marks_group1 = 90) 
  (h4 : num_students_group2 = 15) 
  (h5 : marks_group2 = 80) 
  (h6 : num_students_group3 = total_students - num_students_group1 - num_students_group2) 
  (h7 : marks_group3 = 60) : 
  (10 * 90 + 15 * 80 + (total_students - 10 - 15) * 60) / total_students = 72 := 
by
  sorry

end average_marks_for_class_l294_294492


namespace shaded_area_calculation_l294_294701

-- Define the dimensions of the grid and the size of each square
def gridWidth : ℕ := 9
def gridHeight : ℕ := 7
def squareSize : ℕ := 2

-- Define the number of 2x2 squares horizontally and vertically
def numSquaresHorizontally : ℕ := gridWidth / squareSize
def numSquaresVertically : ℕ := gridHeight / squareSize

-- Define the area of one 2x2 square and one shaded triangle within it
def squareArea : ℕ := squareSize * squareSize
def shadedTriangleArea : ℕ := squareArea / 2

-- Define the total number of 2x2 squares
def totalNumSquares : ℕ := numSquaresHorizontally * numSquaresVertically

-- Define the total area of shaded regions
def totalShadedArea : ℕ := totalNumSquares * shadedTriangleArea

-- The theorem to be proved
theorem shaded_area_calculation : totalShadedArea = 24 := by
  sorry    -- Placeholder for the proof

end shaded_area_calculation_l294_294701


namespace sandy_initial_carrots_l294_294169

-- Defining the conditions
def sam_took : ℕ := 3
def sandy_left : ℕ := 3

-- The statement to be proven
theorem sandy_initial_carrots :
  (sandy_left + sam_took = 6) :=
by
  sorry

end sandy_initial_carrots_l294_294169


namespace parallelogram_area_example_l294_294051

noncomputable def area_parallelogram (A B C D : (ℝ × ℝ)) : ℝ := 
  0.5 * |(A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * A.1)|

theorem parallelogram_area_example : 
  let A := (0, 0)
  let B := (20, 0)
  let C := (25, 7)
  let D := (5, 7)
  area_parallelogram A B C D = 140 := 
by
  sorry

end parallelogram_area_example_l294_294051


namespace robin_photo_count_l294_294690

theorem robin_photo_count (photos_per_page : ℕ) (full_pages : ℕ) 
  (h1 : photos_per_page = 6) (h2 : full_pages = 122) :
  photos_per_page * full_pages = 732 :=
by
  sorry

end robin_photo_count_l294_294690


namespace log_base_2_iff_l294_294206

open Function

theorem log_base_2_iff (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : a > b ↔ log 2 a > log 2 b :=
by

-- The proof would be filled here

sorry

end log_base_2_iff_l294_294206


namespace february_sales_increase_l294_294393

theorem february_sales_increase (Slast : ℝ) (r : ℝ) (Sthis : ℝ) 
  (h_last_year_sales : Slast = 320) 
  (h_percent_increase : r = 0.25) : 
  Sthis = 400 :=
by
  have h1 : Sthis = Slast * (1 + r) := sorry
  sorry

end february_sales_increase_l294_294393


namespace probability_sum_less_than_product_l294_294532

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l294_294532


namespace balls_in_boxes_l294_294265

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l294_294265


namespace unknown_number_value_l294_294646

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end unknown_number_value_l294_294646


namespace integer_inequality_l294_294469

theorem integer_inequality (x y : ℤ) : x * (x + 1) ≠ 2 * (5 * y + 2) := 
  sorry

end integer_inequality_l294_294469


namespace max_sum_of_lengths_l294_294894

def length_of_integer (k : ℤ) (hk : k > 1) : ℤ := sorry

theorem max_sum_of_lengths (x y : ℤ) (hx : x > 1) (hy : y > 1) (h : x + 3 * y < 920) :
  length_of_integer x hx + length_of_integer y hy = 15 :=
sorry

end max_sum_of_lengths_l294_294894


namespace expand_binom_l294_294408

theorem expand_binom (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 :=
by
  sorry

end expand_binom_l294_294408


namespace find_fourth_vertex_l294_294604

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_parallelogram (A B C D : Point) : Prop :=
  is_midpoint ({x := 0, y := -9}) A C ∧ is_midpoint ({x := 2, y := 6}) B D ∧
  is_midpoint ({x := 4, y := 5}) C D ∧ is_midpoint ({x := 0, y := -9}) A D

theorem find_fourth_vertex :
  ∃ D : Point,
    (is_parallelogram ({x := 0, y := -9}) ({x := 2, y := 6}) ({x := 4, y := 5}) D)
    ∧ ((D = {x := 2, y := -10}) ∨ (D = {x := -2, y := -8}) ∨ (D = {x := 6, y := 20})) :=
sorry

end find_fourth_vertex_l294_294604


namespace count_two_digit_integers_remainder_3_div_7_l294_294615

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l294_294615


namespace another_divisor_l294_294502

theorem another_divisor (n : ℕ) (h1 : n = 44402) (h2 : ∀ d ∈ [12, 48, 74, 100], (n + 2) % d = 0) : 
  199 ∣ (n + 2) := 
by 
  sorry

end another_divisor_l294_294502


namespace problem_statement_l294_294821

open Real

theorem problem_statement (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (a : ℝ := x + x⁻¹) (b : ℝ := y + y⁻¹) (c : ℝ := z + z⁻¹) :
  a > 2 ∧ b > 2 ∧ c > 2 :=
by sorry

end problem_statement_l294_294821


namespace count_two_digit_integers_remainder_3_l294_294620

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l294_294620


namespace find_y_value_l294_294303
-- Import the necessary Lean library

-- Define the conditions and the target theorem
theorem find_y_value (h : 6 * y + 3 * y + y + 4 * y = 360) : y = 180 / 7 :=
by
  sorry

end find_y_value_l294_294303


namespace sum_of_abc_l294_294917

theorem sum_of_abc (a b c : ℝ) (h : (a - 5)^2 + (b - 6)^2 + (c - 7)^2 = 0) :
  a + b + c = 18 :=
sorry

end sum_of_abc_l294_294917


namespace component_probability_l294_294359

theorem component_probability (p : ℝ) 
  (h : (1 - p)^3 = 0.001) : 
  p = 0.9 :=
sorry

end component_probability_l294_294359


namespace correct_judgments_about_f_l294_294094

-- Define the function f with its properties
variable {f : ℝ → ℝ} 

-- f is an even function
axiom even_function : ∀ x, f (-x) = f x

-- f satisfies f(x + 1) = -f(x)
axiom function_property : ∀ x, f (x + 1) = -f x

-- f is increasing on [-1, 0]
axiom increasing_on_interval : ∀ x y, -1 ≤ x → x ≤ y → y ≤ 0 → f x ≤ f y

theorem correct_judgments_about_f :
  (∀ x, f x = f (x + 2)) ∧
  (∀ x, f x = f (-x + 2)) ∧
  (f 2 = f 0) :=
by 
  sorry

end correct_judgments_about_f_l294_294094


namespace ratio_of_surface_areas_of_spheres_l294_294661

theorem ratio_of_surface_areas_of_spheres (V1 V2 S1 S2 : ℝ) 
(h : V1 / V2 = 8 / 27) 
(h1 : S1 = 4 * π * (V1^(2/3)) / (2 * π)^(2/3))
(h2 : S2 = 4 * π * (V2^(2/3)) / (3 * π)^(2/3)) :
S1 / S2 = 4 / 9 :=
sorry

end ratio_of_surface_areas_of_spheres_l294_294661


namespace jack_needs_more_money_l294_294933

-- Definitions based on given conditions
def cost_per_sock_pair : ℝ := 9.50
def num_sock_pairs : ℕ := 2
def cost_per_shoe : ℝ := 92
def jack_money : ℝ := 40

-- Theorem statement
theorem jack_needs_more_money (cost_per_sock_pair num_sock_pairs cost_per_shoe jack_money : ℝ) : 
  ((cost_per_sock_pair * num_sock_pairs) + cost_per_shoe) - jack_money = 71 := by
  sorry

end jack_needs_more_money_l294_294933


namespace find_value_l294_294126

-- Define the variables and given conditions
variables (x y z : ℚ)
variables (h1 : 2 * x - y = 4)
variables (h2 : 3 * x + z = 7)
variables (h3 : y = 2 * z)

-- Define the goal to prove
theorem find_value : 6 * x - 3 * y + 3 * z = 51 / 4 := by 
  sorry

end find_value_l294_294126


namespace spots_combined_l294_294797

def Rover : ℕ := 46
def Cisco : ℕ := Rover / 2 - 5
def Granger : ℕ := 5 * Cisco

theorem spots_combined : Granger + Cisco = 108 := by
  sorry

end spots_combined_l294_294797


namespace justin_current_age_l294_294583

theorem justin_current_age
  (angelina_older : ∀ (j a : ℕ), a = j + 4)
  (angelina_future_age : ∀ (a : ℕ), a + 5 = 40) :
  ∃ (justin_current_age : ℕ), justin_current_age = 31 := 
by
  sorry

end justin_current_age_l294_294583


namespace compute_expression_l294_294883

theorem compute_expression : 12 + 5 * (4 - 9)^2 - 3 = 134 := by
  sorry

end compute_expression_l294_294883


namespace determine_a_value_l294_294793

theorem determine_a_value (a : ℤ) (h : ∀ x : ℝ, x^2 + 2 * (a:ℝ) * x + 1 > 0) : a = 0 := 
sorry

end determine_a_value_l294_294793


namespace probability_sum_less_than_product_l294_294535

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l294_294535


namespace slow_speed_distance_l294_294293

theorem slow_speed_distance (D : ℝ) (h : (D + 20) / 14 = D / 10) : D = 50 := by
  sorry

end slow_speed_distance_l294_294293


namespace part_a_part_b_part_c_part_d_part_e_part_f_l294_294565

-- Part (a)
theorem part_a (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^2 = 5 * k + 1 ∨ n^2 = 5 * k - 1 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^4 - 1 = 5 * k := 
sorry

-- Part (c)
theorem part_c (n : ℤ) : n^5 % 10 = n % 10 := 
sorry

-- Part (d)
theorem part_d (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k := 
sorry

-- Part (e)
theorem part_e (k n : ℤ) (h1 : ¬ ∃ j : ℤ, k = 5 * j) (h2 : ¬ ∃ j : ℤ, n = 5 * j) : ∃ j : ℤ, k^4 - n^4 = 5 * j := 
sorry

-- Part (f)
theorem part_f (k m n : ℤ) (h : k^2 + m^2 = n^2) : ∃ j : ℤ, k = 5 * j ∨ ∃ r : ℤ, m = 5 * r ∨ ∃ s : ℤ, n = 5 * s := 
sorry

end part_a_part_b_part_c_part_d_part_e_part_f_l294_294565


namespace total_nap_duration_l294_294152

def nap1 : ℚ := 1 / 5
def nap2 : ℚ := 1 / 4
def nap3 : ℚ := 1 / 6
def hour_to_minutes : ℚ := 60

theorem total_nap_duration :
  (nap1 + nap2 + nap3) * hour_to_minutes = 37 := by
  sorry

end total_nap_duration_l294_294152


namespace maximum_reduced_price_l294_294350

theorem maximum_reduced_price (marked_price : ℝ) (cost_price : ℝ) (reduced_price : ℝ) 
    (h1 : marked_price = 240) 
    (h2 : marked_price = cost_price * 1.6) 
    (h3 : reduced_price - cost_price ≥ cost_price * 0.1) : 
    reduced_price ≤ 165 :=
sorry

end maximum_reduced_price_l294_294350


namespace gemstones_needed_l294_294472

noncomputable def magnets_per_earring : ℕ := 2

noncomputable def buttons_per_earring : ℕ := magnets_per_earring / 2

noncomputable def gemstones_per_earring : ℕ := 3 * buttons_per_earring

noncomputable def sets_of_earrings : ℕ := 4

noncomputable def earrings_per_set : ℕ := 2

noncomputable def total_gemstones : ℕ := sets_of_earrings * earrings_per_set * gemstones_per_earring

theorem gemstones_needed :
  total_gemstones = 24 :=
  by
    sorry

end gemstones_needed_l294_294472


namespace derivative_at_zero_l294_294111

def f (x : ℝ) : ℝ := x^3

theorem derivative_at_zero : deriv f 0 = 0 :=
by
  sorry

end derivative_at_zero_l294_294111


namespace no_solution_in_A_l294_294454

def A : Set ℕ := 
  {n | ∃ k : ℤ, abs (n * Real.sqrt 2022 - 1 / 3 - k) ≤ 1 / 2022}

theorem no_solution_in_A (x y z : ℕ) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A) : 
  20 * x + 21 * y ≠ 22 * z := 
sorry

end no_solution_in_A_l294_294454


namespace inequality_solution_l294_294772

theorem inequality_solution :
  {x : ℝ | (3 * x - 8) * (x - 4) / (x - 1) ≥ 0 } = { x : ℝ | x < 1 } ∪ { x : ℝ | x ≥ 4 } :=
by {
  sorry
}

end inequality_solution_l294_294772


namespace perimeter_of_similar_triangle_l294_294501

theorem perimeter_of_similar_triangle (a b c d : ℕ) (h_iso : (a = 12) ∧ (b = 24) ∧ (c = 24)) (h_sim : d = 30) 
  : (d + 2 * b) = 150 := by
  sorry

end perimeter_of_similar_triangle_l294_294501


namespace company_sales_difference_l294_294983

theorem company_sales_difference:
  let price_A := 4
  let quantity_A := 300
  let total_sales_A := price_A * quantity_A
  let price_B := 3.5
  let quantity_B := 350
  let total_sales_B := price_B * quantity_B
  total_sales_B - total_sales_A = 25 :=
by
  sorry

end company_sales_difference_l294_294983


namespace radius_of_bicycle_wheel_is_13_l294_294548

-- Define the problem conditions
def diameter_cm : ℕ := 26

-- Define the function to calculate radius from diameter
def radius (d : ℕ) : ℕ := d / 2

-- Prove that the radius is 13 cm when diameter is 26 cm
theorem radius_of_bicycle_wheel_is_13 :
  radius diameter_cm = 13 := 
sorry

end radius_of_bicycle_wheel_is_13_l294_294548


namespace complement_intersection_in_U_l294_294163

universe u

variables {α : Type u} (U A B : Set α)

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_in_U : U \ (A ∩ B) = {1, 4, 5} :=
by
  sorry

end complement_intersection_in_U_l294_294163


namespace part1_part2_l294_294426

variable {a x y : ℝ} 

-- Conditions
def condition_1 (a x y : ℝ) := x - y = 1 + 3 * a
def condition_2 (a x y : ℝ) := x + y = -7 - a
def condition_3 (x : ℝ) := x ≤ 0
def condition_4 (y : ℝ) := y < 0

-- Part 1: Range for a
theorem part1 (a : ℝ) : 
  (∀ x y, condition_1 a x y ∧ condition_2 a x y ∧ condition_3 x ∧ condition_4 y → (-2 < a ∧ a ≤ 3)) :=
sorry

-- Part 2: Specific integer value for a
theorem part2 (a : ℝ) :
  (-2 < a ∧ a ≤ 3 → (∃ (x : ℝ), (2 * a + 1) * x > 2 * a + 1 ∧ x < 1) → a = -1) :=
sorry

end part1_part2_l294_294426


namespace rectangle_area_l294_294773

theorem rectangle_area (P l w : ℝ) (h1 : P = 60) (h2 : l / w = 3 / 2) (h3 : P = 2 * l + 2 * w) : l * w = 216 :=
by
  sorry

end rectangle_area_l294_294773


namespace pupils_in_program_l294_294509

theorem pupils_in_program {total_people parents : ℕ} (h1 : total_people = 238) (h2 : parents = 61) :
  total_people - parents = 177 := by
  sorry

end pupils_in_program_l294_294509


namespace cost_of_each_skirt_l294_294100

-- Problem definitions based on conditions
def cost_of_art_supplies : ℕ := 20
def total_expenditure : ℕ := 50
def number_of_skirts : ℕ := 2

-- Proving the cost of each skirt
theorem cost_of_each_skirt (cost_of_each_skirt : ℕ) : 
  number_of_skirts * cost_of_each_skirt + cost_of_art_supplies = total_expenditure → 
  cost_of_each_skirt = 15 := 
by 
  sorry

end cost_of_each_skirt_l294_294100


namespace petunia_fertilizer_problem_l294_294028

theorem petunia_fertilizer_problem
  (P : ℕ)
  (h1 : 4 * P * 8 + 3 * 6 * 3 + 2 * 2 = 314) :
  P = 8 :=
by
  sorry

end petunia_fertilizer_problem_l294_294028


namespace balls_in_boxes_l294_294284

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l294_294284


namespace books_checked_out_on_Thursday_l294_294859

theorem books_checked_out_on_Thursday (initial_books : ℕ) (wednesday_checked_out : ℕ) 
                                      (thursday_returned : ℕ) (friday_returned : ℕ) (final_books : ℕ) 
                                      (thursday_checked_out : ℕ) : 
  (initial_books = 98) → 
  (wednesday_checked_out = 43) → 
  (thursday_returned = 23) → 
  (friday_returned = 7) → 
  (final_books = 80) → 
  (initial_books - wednesday_checked_out + thursday_returned - thursday_checked_out + friday_returned = final_books) → 
  (thursday_checked_out = 5) :=
by
  intros
  sorry

end books_checked_out_on_Thursday_l294_294859


namespace fifth_house_number_is_13_l294_294708

theorem fifth_house_number_is_13 (n : ℕ) (a₁ : ℕ) (h₀ : n ≥ 5) (h₁ : (a₁ + n - 1) * n = 117) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n -> (a₁ + 2 * (i - 1)) = 2*(i-1) + a₁) : 
  (a₁ + 2 * (5 - 1)) = 13 :=
by
  sorry

end fifth_house_number_is_13_l294_294708


namespace kenneth_left_with_amount_l294_294813

theorem kenneth_left_with_amount (total_earnings : ℝ) (percentage_spent : ℝ) (amount_left : ℝ) 
    (h_total_earnings : total_earnings = 450) (h_percentage_spent : percentage_spent = 0.10) 
    (h_spent_amount : total_earnings * percentage_spent = 45) : 
    amount_left = total_earnings - total_earnings * percentage_spent :=
by sorry

end kenneth_left_with_amount_l294_294813


namespace average_correct_l294_294547

theorem average_correct :
  (12 + 13 + 14 + 510 + 520 + 530 + 1115 + 1120 + 1252140 + 2345) / 10 = 125831.9 := 
sorry

end average_correct_l294_294547


namespace book_pairs_count_l294_294430

theorem book_pairs_count :
  let mystery_count := 3
  let fantasy_count := 4
  let biography_count := 3
  mystery_count * fantasy_count + mystery_count * biography_count + fantasy_count * biography_count = 33 :=
by 
  sorry

end book_pairs_count_l294_294430


namespace sum_consecutive_integers_150_l294_294095

theorem sum_consecutive_integers_150 (n : ℕ) (a : ℕ) (hn : n ≥ 3) (hdiv : 300 % n = 0) :
  n * (2 * a + n - 1) = 300 ↔ (a > 0) → n = 3 ∨ n = 5 ∨ n = 15 :=
by sorry

end sum_consecutive_integers_150_l294_294095


namespace determine_a_l294_294130

theorem determine_a
  (h : ∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a * x - 2) ≥ 0) : 
  a = 1 :=
sorry

end determine_a_l294_294130


namespace num_ways_to_put_5_balls_into_4_boxes_l294_294282

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l294_294282


namespace liars_at_table_l294_294332

open Set

noncomputable def number_of_liars : Set ℕ :=
  {n | ∃ (knights, liars : ℕ), knights + liars = 450 ∧
                                (∀ i : ℕ, i < 450 → (liars + ((i + 1) % 450) + ((i + 2) % 450) = 1)) }

theorem liars_at_table : number_of_liars = {150, 450} := 
  sorry

end liars_at_table_l294_294332


namespace F_at_2_eq_minus_22_l294_294109

variable (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x^7 + b * x^5 + c * x^3 + d * x

def F (x : ℝ) : ℝ := f a b c d x - 6

theorem F_at_2_eq_minus_22 (h : F a b c d (-2) = 10) : F a b c d 2 = -22 :=
by
  sorry

end F_at_2_eq_minus_22_l294_294109


namespace abc_inequality_l294_294736

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a * b + b * c + c * a ≥ a * Real.sqrt (b * c) + b * Real.sqrt (a * c) + c * Real.sqrt (a * b) :=
sorry

end abc_inequality_l294_294736


namespace tenth_term_geom_seq_l294_294404

theorem tenth_term_geom_seq :
  let a := (5 : ℚ)
  let r := (4 / 3 : ℚ)
  let n := 10
  (a * r^(n - 1)) = (1310720 / 19683 : ℚ) :=
by
  sorry

end tenth_term_geom_seq_l294_294404


namespace company_sales_difference_l294_294982

theorem company_sales_difference:
  let price_A := 4
  let quantity_A := 300
  let total_sales_A := price_A * quantity_A
  let price_B := 3.5
  let quantity_B := 350
  let total_sales_B := price_B * quantity_B
  total_sales_B - total_sales_A = 25 :=
by
  sorry

end company_sales_difference_l294_294982


namespace pascals_triangle_row_20_fifth_element_l294_294879

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.div (Nat.factorial n) ((Nat.factorial k) * (Nat.factorial (n - k)))

-- State the theorem about Row 20, fifth element in Pascal's triangle
theorem pascals_triangle_row_20_fifth_element :
  binomial 20 4 = 4845 := 
by
  sorry

end pascals_triangle_row_20_fifth_element_l294_294879


namespace num_ways_to_put_5_balls_into_4_boxes_l294_294281

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l294_294281


namespace cost_to_paint_cube_l294_294567

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) 
  (h1 : cost_per_kg = 40) 
  (h2 : coverage_per_kg = 20) 
  (h3 : side_length = 10) 
  : (6 * side_length^2 / coverage_per_kg) * cost_per_kg = 1200 :=
by
  sorry

end cost_to_paint_cube_l294_294567


namespace gemstones_needed_l294_294473

noncomputable def magnets_per_earring : ℕ := 2

noncomputable def buttons_per_earring : ℕ := magnets_per_earring / 2

noncomputable def gemstones_per_earring : ℕ := 3 * buttons_per_earring

noncomputable def sets_of_earrings : ℕ := 4

noncomputable def earrings_per_set : ℕ := 2

noncomputable def total_gemstones : ℕ := sets_of_earrings * earrings_per_set * gemstones_per_earring

theorem gemstones_needed :
  total_gemstones = 24 :=
  by
    sorry

end gemstones_needed_l294_294473


namespace radius_of_scrap_cookie_l294_294766

theorem radius_of_scrap_cookie :
  ∀ (r : ℝ),
    (∃ (r_dough r_cookie : ℝ),
      r_dough = 6 ∧  -- Radius of the large dough
      r_cookie = 2 ∧  -- Radius of each cookie
      8 * (π * r_cookie^2) ≤ π * r_dough^2 ∧  -- Total area of cookies is less than or equal to area of large dough
      (π * r_dough^2) - (8 * (π * r_cookie^2)) = π * r^2  -- Area of scrap dough forms a circle of radius r
    ) → r = 2 := by
  sorry

end radius_of_scrap_cookie_l294_294766


namespace positive_two_digit_integers_remainder_3_l294_294628

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l294_294628


namespace ralph_socks_l294_294828

theorem ralph_socks (x y z : ℕ) (h1 : x + y + z = 12) (h2 : x + 3 * y + 4 * z = 24) (h3 : 1 ≤ x) (h4 : 1 ≤ y) (h5 : 1 ≤ z) : x = 7 :=
sorry

end ralph_socks_l294_294828


namespace determine_age_l294_294055

def David_age (D Y : ℕ) : Prop := Y = 2 * D ∧ Y = D + 7

theorem determine_age (D : ℕ) (h : David_age D (D + 7)) : D = 7 :=
by
  sorry

end determine_age_l294_294055


namespace count_two_digit_integers_remainder_3_l294_294619

theorem count_two_digit_integers_remainder_3 :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 13 ∧ (∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ x % 7 = 3) ∧ (nat.num_digits 10 (7 * n + 3) = 2) :=
sorry

end count_two_digit_integers_remainder_3_l294_294619


namespace koi_fish_in_pond_l294_294209

theorem koi_fish_in_pond:
  ∃ k : ℕ, 2 * k - 14 = 64 ∧ k = 39 := sorry

end koi_fish_in_pond_l294_294209


namespace solution_l294_294317

-- Given conditions in the problem
def F (x : ℤ) : ℤ := sorry -- Placeholder for the polynomial with integer coefficients
variables (a : ℕ → ℤ) (m : ℕ)

-- Given that: ∀ n, ∃ k, F(n) is divisible by a(k) for some k in {1, 2, ..., m}
axiom forall_n_exists_k : ∀ n : ℤ, ∃ k : ℕ, k < m ∧ a k ∣ F n

-- Desired conclusion: ∃ k, ∀ n, F(n) is divisible by a(k)
theorem solution : ∃ k : ℕ, k < m ∧ (∀ n : ℤ, a k ∣ F n) :=
sorry

end solution_l294_294317


namespace find_triangle_sides_l294_294004

-- Define the variables and conditions
noncomputable def k := 5
noncomputable def c := 12
noncomputable def d := 10

-- Assume the perimeters of the figures
def P1 : ℕ := 74
def P2 : ℕ := 84
def P3 : ℕ := 82

-- Define the equations based on the perimeters
def Equation1 := P2 = P1 + 2 * k
def Equation2 := P3 = P1 + 6 * c - 2 * k

-- The lean theorem proving that the sides of the triangle are as given
theorem find_triangle_sides : 
  (Equation1 ∧ Equation2) →
  (k = 5 ∧ c = 12 ∧ d = 10) :=
by
  sorry

end find_triangle_sides_l294_294004


namespace krystiana_monthly_income_l294_294147

theorem krystiana_monthly_income :
  let first_floor_income := 3 * 15
  let second_floor_income := 3 * 20
  let third_floor_income := 2 * (2 * 15)
  first_floor_income + second_floor_income + third_floor_income = 165 :=
by
  let first_floor_income := 3 * 15
  let second_floor_income := 3 * 20
  let third_floor_income := 2 * (2 * 15)
  have h1: first_floor_income = 45 := by simp [first_floor_income]
  have h2: second_floor_income = 60 := by simp [second_floor_income]
  have h3: third_floor_income = 60 := by simp [third_floor_income]
  rw [h1, h2, h3]
  simp
  done

end krystiana_monthly_income_l294_294147


namespace largest_multiple_of_8_less_than_100_l294_294992

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l294_294992


namespace incorrect_calculation_l294_294485

noncomputable def ξ : ℝ := 3 -- Expected lifetime of the sensor
noncomputable def η : ℝ := 5 -- Expected lifetime of the transmitter
noncomputable def T (ξ η : ℝ) : ℝ := min ξ η -- Lifetime of the entire device

theorem incorrect_calculation (h1 : E ξ = 3) (h2 : E η = 5) (h3 : E (min ξ η ) = 3.67) : False :=
by
  have h4 : E (min ξ η ) ≤ 3 := sorry -- Based on properties of expectation and min
  have h5 : 3.67 > 3 := by linarith -- Known inequality
  sorry

end incorrect_calculation_l294_294485


namespace total_votes_cast_l294_294061

theorem total_votes_cast (F A T : ℕ) (h1 : F = A + 70) (h2 : A = 2 * T / 5) (h3 : T = F + A) : T = 350 :=
by
  sorry

end total_votes_cast_l294_294061


namespace value_of_a_is_negative_one_l294_294818

-- Conditions
def I (a : ℤ) : Set ℤ := {2, 4, a^2 - a - 3}
def A (a : ℤ) : Set ℤ := {4, 1 - a}
def complement_I_A (a : ℤ) : Set ℤ := {x ∈ I a | x ∉ A a}

-- Theorem statement
theorem value_of_a_is_negative_one (a : ℤ) (h : complement_I_A a = {-1}) : a = -1 :=
by
  sorry

end value_of_a_is_negative_one_l294_294818


namespace sequence_general_formula_l294_294244

/--
A sequence a_n is defined such that the first term a_1 = 3 and the recursive formula 
a_{n+1} = (3 * a_n - 4) / (a_n - 2).

We aim to prove that the general term of the sequence is given by:
a_n = ( (-2)^(n+2) - 1 ) / ( (-2)^n - 1 )
-/
theorem sequence_general_formula (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 3)
  (hr : ∀ n, a (n + 1) = (3 * a n - 4) / (a n - 2)) :
  a n = ( (-2:ℝ)^(n+2) - 1 ) / ( (-2:ℝ)^n - 1) :=
sorry

end sequence_general_formula_l294_294244


namespace fish_total_after_transfer_l294_294683

-- Definitions of the initial conditions
def lilly_initial : ℕ := 10
def rosy_initial : ℕ := 9
def jack_initial : ℕ := 15
def fish_transferred : ℕ := 2

-- Total fish after Lilly transfers 2 fish to Jack
theorem fish_total_after_transfer : (lilly_initial - fish_transferred) + rosy_initial + (jack_initial + fish_transferred) = 34 := by
  sorry

end fish_total_after_transfer_l294_294683


namespace complement_inter_of_A_and_B_l294_294125

open Set

variable (U A B : Set ℕ)

theorem complement_inter_of_A_and_B:
  U = {1, 2, 3, 4, 5}
  ∧ A = {1, 2, 3}
  ∧ B = {2, 3, 4} 
  → U \ (A ∩ B) = {1, 4, 5} :=
by
  sorry

end complement_inter_of_A_and_B_l294_294125


namespace joan_has_10_books_l294_294010

def toms_books := 38
def together_books := 48
def joans_books := together_books - toms_books

theorem joan_has_10_books : joans_books = 10 :=
by
  -- The proof goes here, but we'll add "sorry" to indicate it's a placeholder.
  sorry

end joan_has_10_books_l294_294010


namespace find_f_of_3_l294_294786

theorem find_f_of_3 (a b c : ℝ) (f : ℝ → ℝ) (h1 : f 1 = 7) (h2 : f 2 = 12) (h3 : ∀ x, f x = ax + bx + c) : f 3 = 17 :=
by
  sorry

end find_f_of_3_l294_294786


namespace sum_faces_of_cube_l294_294961

theorem sum_faces_of_cube (p u q v r w : ℕ) (hp : 0 < p) (hu : 0 < u) (hq : 0 < q) (hv : 0 < v)
    (hr : 0 < r) (hw : 0 < w)
    (h_sum_vertices : p * q * r + p * v * r + p * q * w + p * v * w 
        + u * q * r + u * v * r + u * q * w + u * v * w = 2310) : 
    p + u + q + v + r + w = 40 := 
sorry

end sum_faces_of_cube_l294_294961


namespace find_plaid_shirts_l294_294341

def total_shirts : ℕ := 5
def total_pants : ℕ := 24
def total_items : ℕ := total_shirts + total_pants
def neither_plaid_nor_purple : ℕ := 21
def total_plaid_or_purple : ℕ := total_items - neither_plaid_nor_purple
def purple_pants : ℕ := 5
def plaid_shirts (p : ℕ) : Prop := total_plaid_or_purple - purple_pants = p

theorem find_plaid_shirts : plaid_shirts 3 := by
  unfold plaid_shirts
  repeat { sorry }

end find_plaid_shirts_l294_294341


namespace jack_needs_more_money_l294_294935

-- Definitions based on given conditions
def cost_per_sock_pair : ℝ := 9.50
def num_sock_pairs : ℕ := 2
def cost_per_shoe : ℝ := 92
def jack_money : ℝ := 40

-- Theorem statement
theorem jack_needs_more_money (cost_per_sock_pair num_sock_pairs cost_per_shoe jack_money : ℝ) : 
  ((cost_per_sock_pair * num_sock_pairs) + cost_per_shoe) - jack_money = 71 := by
  sorry

end jack_needs_more_money_l294_294935


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l294_294633

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l294_294633


namespace common_ratio_l294_294447

variable {G : Type} [LinearOrderedField G]

-- Definitions based on conditions
def geometric_seq (a₁ q : G) (n : ℕ) : G := a₁ * q^(n-1)
def sum_geometric_seq (a₁ q : G) (n : ℕ) : G :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

-- Hypotheses from conditions
variable {a₁ q : G}
variable (h1 : sum_geometric_seq a₁ q 3 = 7)
variable (h2 : sum_geometric_seq a₁ q 6 = 63)

theorem common_ratio (a₁ q : G) (h1 : sum_geometric_seq a₁ q 3 = 7)
  (h2 : sum_geometric_seq a₁ q 6 = 63) : q = 2 :=
by
  -- Proof to be completed
  sorry

end common_ratio_l294_294447


namespace savings_percentage_l294_294564

theorem savings_percentage (I S : ℝ) (h1 : I > 0) (h2 : S > 0) (h3 : S ≤ I) 
  (h4 : 1.25 * I - 2 * S + I - S = 2 * (I - S)) :
  (S / I) * 100 = 25 :=
by
  sorry

end savings_percentage_l294_294564


namespace jerry_won_47_tickets_l294_294088

open Nat

-- Define the initial number of tickets
def initial_tickets : Nat := 4

-- Define the number of tickets spent on the beanie
def tickets_spent_on_beanie : Nat := 2

-- Define the current total number of tickets Jerry has
def current_tickets : Nat := 49

-- Define the number of tickets Jerry won later
def tickets_won_later : Nat := current_tickets - (initial_tickets - tickets_spent_on_beanie)

-- The theorem to prove
theorem jerry_won_47_tickets :
  tickets_won_later = 47 :=
by sorry

end jerry_won_47_tickets_l294_294088


namespace ways_to_distribute_balls_in_boxes_l294_294278

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l294_294278


namespace find_triples_l294_294314

theorem find_triples (k : ℕ) (hk : 0 < k) :
  ∃ (a b c : ℕ), 
    (0 < a ∧ 0 < b ∧ 0 < c) ∧
    (a + b + c = 3 * k + 1) ∧ 
    (a * b + b * c + c * a = 3 * k^2 + 2 * k) ∧ 
    (a = k + 1 ∧ b = k ∧ c = k) :=
by
  sorry

end find_triples_l294_294314


namespace unique_solution_l294_294770

noncomputable def f : ℝ → ℝ :=
sorry

theorem unique_solution (x : ℝ) (hx : 0 ≤ x) : 
  (f : ℝ → ℝ) (2 * x + 1) = 3 * (f x) + 5 ↔ f x = -5 / 2 :=
by 
  sorry

end unique_solution_l294_294770


namespace cost_of_dinner_l294_294011

theorem cost_of_dinner (x : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (total_cost : ℝ) : 
  tax_rate = 0.09 → tip_rate = 0.18 → total_cost = 36.90 → 
  1.27 * x = 36.90 → x = 29 :=
by
  intros htr htt htc heq
  rw [←heq] at htc
  sorry

end cost_of_dinner_l294_294011


namespace at_least_one_triangle_l294_294456

theorem at_least_one_triangle {n : ℕ} (h1 : n ≥ 2) (points : Finset ℕ) (segments : Finset (ℕ × ℕ)) : 
(points.card = 2 * n) ∧ (segments.card = n^2 + 1) → 
∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ ((a, b) ∈ segments ∨ (b, a) ∈ segments) ∧ ((b, c) ∈ segments ∨ (c, b) ∈ segments) ∧ ((c, a) ∈ segments ∨ (a, c) ∈ segments) := 
by 
  sorry

end at_least_one_triangle_l294_294456


namespace polynomial_sum_l294_294950

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l294_294950


namespace work_done_by_gas_l294_294468

def gas_constant : ℝ := 8.31 -- J/(mol·K)
def temperature_change : ℝ := 100 -- K (since 100°C increase is equivalent to 100 K in Kelvin)
def moles_of_gas : ℝ := 1 -- one mole of gas

theorem work_done_by_gas :
  (1/2) * gas_constant * temperature_change = 415.5 :=
by sorry

end work_done_by_gas_l294_294468


namespace range_of_m_l294_294905

theorem range_of_m (m : ℝ) :
  let p := (2 < m ∧ m < 4)
  let q := (m > 1 ∧ 4 - 4 * m < 0)
  (¬ (p ∧ q) ∧ (p ∨ q)) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 4) :=
by intros p q h
   let p := 2 < m ∧ m < 4
   let q := m > 1 ∧ 4 - 4 * m < 0
   sorry

end range_of_m_l294_294905


namespace obtuse_angle_half_in_first_quadrant_l294_294912

-- Define α to be an obtuse angle
variable {α : ℝ}

-- The main theorem we want to prove
theorem obtuse_angle_half_in_first_quadrant (h_obtuse : (π / 2) < α ∧ α < π) :
  0 < α / 2 ∧ α / 2 < π / 2 :=
  sorry

end obtuse_angle_half_in_first_quadrant_l294_294912


namespace cos_330_eq_sqrt3_over_2_l294_294885

theorem cos_330_eq_sqrt3_over_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_over_2_l294_294885


namespace total_bees_in_hive_at_end_of_7_days_l294_294569

-- Definitions of given conditions
def daily_hatch : Nat := 3000
def daily_loss : Nat := 900
def initial_bees : Nat := 12500
def days : Nat := 7
def queen_count : Nat := 1

-- Statement to prove
theorem total_bees_in_hive_at_end_of_7_days :
  initial_bees + daily_hatch * days - daily_loss * days + queen_count = 27201 := by
  sorry

end total_bees_in_hive_at_end_of_7_days_l294_294569


namespace Angie_necessities_amount_l294_294223

noncomputable def Angie_salary : ℕ := 80
noncomputable def Angie_left_over : ℕ := 18
noncomputable def Angie_taxes : ℕ := 20
noncomputable def Angie_expenses : ℕ := Angie_salary - Angie_left_over
noncomputable def Angie_necessities : ℕ := Angie_expenses - Angie_taxes

theorem Angie_necessities_amount :
  Angie_necessities = 42 :=
by
  unfold Angie_necessities
  unfold Angie_expenses
  sorry

end Angie_necessities_amount_l294_294223


namespace sum_of_valid_n_l294_294559

theorem sum_of_valid_n : 
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)] in
  (n_values.sum = 3) :=
by
  -- Define the values of n according to the problem's conditions
  let n_values := 
    [n | ∃ d : ℤ, (d ∣ 36) ∧ (2 * n - 1 = d) ∧ (d % 2 ≠ 0)],
  -- Proof will be filled in here
  sorry

end sum_of_valid_n_l294_294559


namespace range_of_a_l294_294662

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (1 ≤ x) ∧ (∀ a : ℝ, (1 + 1 / x) ^ (x + a) ≥ Real.exp 1 → a ≥ 1 / Real.log 2 - 1)

theorem range_of_a : problem_statement :=
sorry

end range_of_a_l294_294662


namespace most_economical_speed_and_cost_l294_294611

open Real

theorem most_economical_speed_and_cost :
  ∀ (x : ℝ),
  (120:ℝ) / x * 36 + (120:ℝ) / x * 6 * (4 + x^2 / 360) = ((7200:ℝ) / x) + 2 * x → 
  50 ≤ x ∧ x ≤ 100 → 
  (∀ v : ℝ, (50 ≤ v ∧ v ≤ 100) → 
  (120 / v * 36 + 120 / v * 6 * (4 + v^2 / 360) ≤ 120 / x * 36 + 120 / x * 6 * (4 + x^2 / 360)) ) → 
  x = 60 → 
  (120 / x * 36 + 120 / x * 6 * (4 + x^2 / 360) = 240) :=
by
  intros x hx bounds min_cost opt_speed
  sorry

end most_economical_speed_and_cost_l294_294611


namespace find_x_l294_294261

-- Definitions based directly on conditions
def vec_a : ℝ × ℝ := (2, 4)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 3)
def vec_c (x : ℝ) : ℝ × ℝ := (2 - x, 1)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The mathematically equivalent proof problem statement
theorem find_x (x : ℝ) : dot_product (vec_c x) (vec_b x) = 0 → (x = -1 ∨ x = 3) :=
by
  -- Placeholder for the proof
  sorry

end find_x_l294_294261


namespace differences_occur_10_times_l294_294313

variable (a : Fin 45 → Nat)

theorem differences_occur_10_times 
    (h : ∀ i j : Fin 44, i < j → a i < a j)
    (h_lt_125 : ∀ i : Fin 44, a i < 125) :
    ∃ i : Fin 43, ∃ j : Fin 43, i ≠ j ∧ (a (i + 1) - a i) = (a (j + 1) - a j) ∧ 
    (∃ k : Nat, k ≥ 10 ∧ (a (j + 1) - a j) = (a (k + 1) - a k)) :=
sorry

end differences_occur_10_times_l294_294313


namespace tangent_line_parallel_to_x_axis_l294_294488

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def f_derivative (x : ℝ) : ℝ := (1 - Real.log x) / (x^2)

theorem tangent_line_parallel_to_x_axis :
  ∀ x₀ : ℝ, 
  f_derivative x₀ = 0 → 
  f x₀ = 1 / Real.exp 1 :=
by
  intro x₀ h_deriv_zero
  sorry

end tangent_line_parallel_to_x_axis_l294_294488


namespace largest_multiple_of_8_less_than_100_l294_294988

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l294_294988


namespace line_through_A_and_B_l294_294419

variables (x y x₁ y₁ x₂ y₂ : ℝ)

-- Conditions
def condition1 : Prop := 3 * x₁ - 4 * y₁ - 2 = 0
def condition2 : Prop := 3 * x₂ - 4 * y₂ - 2 = 0

-- Proof that the line passing through A(x₁, y₁) and B(x₂, y₂) is 3x - 4y - 2 = 0
theorem line_through_A_and_B (h1 : condition1 x₁ y₁) (h2 : condition2 x₂ y₂) :
    ∀ (x y : ℝ), (∃ k : ℝ, x = x₁ + k * (x₂ - x₁) ∧ y = y₁ + k * (y₂ - y₁)) → 3 * x - 4 * y - 2 = 0 :=
sorry

end line_through_A_and_B_l294_294419


namespace tournament_participants_l294_294666

theorem tournament_participants (n : ℕ) (h₁ : 2 * (n * (n - 1) / 2 + 4) - (n - 2) * (n - 3) - 16 = 124) : n = 13 :=
sorry

end tournament_participants_l294_294666


namespace find_multiple_l294_294875

variables (total_questions correct_answers score : ℕ)
variable (m : ℕ)
variable (incorrect_answers : ℕ := total_questions - correct_answers)

-- Given conditions
axiom total_questions_eq : total_questions = 100
axiom correct_answers_eq : correct_answers = 92
axiom score_eq : score = 76

-- Define the scoring method
def score_formula : ℕ := correct_answers - m * incorrect_answers

-- Statement to prove
theorem find_multiple : score = 76 → correct_answers = 92 → total_questions = 100 → score_formula total_questions correct_answers m = score → m = 2 := by
  intros h1 h2 h3 h4
  sorry

end find_multiple_l294_294875


namespace k_h_5_eq_148_l294_294639

def h (x : ℤ) : ℤ := 4 * x + 6
def k (x : ℤ) : ℤ := 6 * x - 8

theorem k_h_5_eq_148 : k (h 5) = 148 := by
  sorry

end k_h_5_eq_148_l294_294639


namespace star_operation_possible_l294_294670

noncomputable def star_operation_exists : Prop := 
  ∃ (star : ℤ → ℤ → ℤ), 
  (∀ (a b c : ℤ), star (star a b) c = star a (star b c)) ∧ 
  (∀ (x y : ℤ), star (star x x) y = y ∧ star y (star x x) = y)

theorem star_operation_possible : star_operation_exists :=
sorry

end star_operation_possible_l294_294670


namespace balls_in_boxes_l294_294287

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l294_294287


namespace min_distance_line_curve_l294_294682

/-- 
  Given line l with parametric equations:
    x = 1 + t * cos α,
    y = t * sin α,
  and curve C with the polar equation:
    ρ * sin^2 θ = 4 * cos θ,
  prove:
    1. The Cartesian coordinate equation of C is y^2 = 4x.
    2. The minimum value of the distance |AB|, where line l intersects curve C, is 4.
-/
theorem min_distance_line_curve {t α θ ρ x y : ℝ} 
  (h_line_x: x = 1 + t * Real.cos α)
  (h_line_y: y = t * Real.sin α)
  (h_curve_polar: ρ * (Real.sin θ)^2 = 4 * Real.cos θ)
  (h_alpha_range: 0 < α ∧ α < Real.pi) : 
  (∀ {x y}, y^2 = 4 * x) ∧ (min_value_of_AB = 4) :=
sorry

end min_distance_line_curve_l294_294682


namespace sum_lt_prod_probability_l294_294525

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l294_294525


namespace number_of_real_b_l294_294105

noncomputable def count_integer_roots_of_quadratic_eq_b : ℕ :=
  let pairs := [(1, 64), (2, 32), (4, 16), (8, 8), (-1, -64), (-2, -32), (-4, -16), (-8, -8)]
  pairs.length

theorem number_of_real_b : count_integer_roots_of_quadratic_eq_b = 8 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end number_of_real_b_l294_294105


namespace simplify_tangent_expression_l294_294170

open Real

theorem simplify_tangent_expression :
  (tan (π / 6) + tan (2 * π / 9) + tan (5 * π / 18) + tan (π / 3)) / cos (π / 9) = 8 * sqrt 3 / 3 :=
by sorry

end simplify_tangent_expression_l294_294170


namespace krystiana_monthly_earnings_l294_294146

-- Definitions based on the conditions
def first_floor_cost : ℕ := 15
def second_floor_cost : ℕ := 20
def third_floor_cost : ℕ := 2 * first_floor_cost
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms_occupied : ℕ := 2

-- Statement to prove Krystiana's total monthly earnings are $165
theorem krystiana_monthly_earnings : 
  first_floor_cost * first_floor_rooms + 
  second_floor_cost * second_floor_rooms + 
  third_floor_cost * third_floor_rooms_occupied = 165 :=
by admit

end krystiana_monthly_earnings_l294_294146


namespace right_triangle_hypotenuse_l294_294176

theorem right_triangle_hypotenuse (A : ℝ) (h height : ℝ) :
  A = 320 ∧ height = 16 →
  ∃ c : ℝ, c = 4 * Real.sqrt 116 :=
by
  intro h
  sorry

end right_triangle_hypotenuse_l294_294176


namespace find_missing_number_l294_294374

theorem find_missing_number (x : ℚ) (h : (476 + 424) * 2 - x * 476 * 424 = 2704) : 
  x = -1 / 223 :=
by
  sorry

end find_missing_number_l294_294374


namespace number_of_members_greater_than_median_l294_294440

theorem number_of_members_greater_than_median (n : ℕ) (median : ℕ) (avg_age : ℕ) (youngest : ℕ) (oldest : ℕ) :
  n = 100 ∧ avg_age = 21 ∧ youngest = 1 ∧ oldest = 70 →
  ∃ k, k = 50 :=
by
  sorry

end number_of_members_greater_than_median_l294_294440


namespace number_of_liars_l294_294334

constant islanders : Type
constant knight : islanders → Prop
constant liar : islanders → Prop
constant sits_at_table : islanders → Prop
constant right_of : islanders → islanders

axiom A1 : ∀ x : islanders, sits_at_table x → (knight x ∨ liar x)
axiom A2 : (∃ n : ℕ, n = 450 ∧ (λ x, sits_at_table x))
axiom A3 : ∀ x : islanders, sits_at_table x →
  (liar (right_of x) ∧ ¬ liar (right_of (right_of x))) ∨ 
  (¬ liar (right_of x) ∧ liar (right_of (right_of x)))

theorem number_of_liars : 
  (∃ n, ∃ m, (n = 450) ∨ (m = 150)) :=
sorry

end number_of_liars_l294_294334


namespace cos_equation_solution_l294_294436

open Real

theorem cos_equation_solution (m : ℝ) :
  (∀ x : ℝ, 4 * cos x - cos x^2 + m - 3 = 0) ↔ (0 ≤ m ∧ m ≤ 8) := by
  sorry

end cos_equation_solution_l294_294436


namespace find_f_neg4_l294_294791

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg4 (a b : ℝ) (h : f a b 4 = 0) : f a b (-4) = 2 := by
  -- sorry to skip the proof
  sorry

end find_f_neg4_l294_294791


namespace students_total_l294_294357

def num_girls : ℕ := 11
def num_boys : ℕ := num_girls + 5

theorem students_total : num_girls + num_boys = 27 := by
  sorry

end students_total_l294_294357


namespace polynomial_sum_l294_294951

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l294_294951


namespace overall_loss_is_450_l294_294080

noncomputable def total_worth_stock : ℝ := 22499.999999999996

noncomputable def selling_price_20_percent_stock (W : ℝ) : ℝ :=
    0.20 * W * 1.10

noncomputable def selling_price_80_percent_stock (W : ℝ) : ℝ :=
    0.80 * W * 0.95

noncomputable def total_selling_price (W : ℝ) : ℝ :=
    selling_price_20_percent_stock W + selling_price_80_percent_stock W

noncomputable def overall_loss (W : ℝ) : ℝ :=
    W - total_selling_price W

theorem overall_loss_is_450 :
  overall_loss total_worth_stock = 450 := by
  sorry

end overall_loss_is_450_l294_294080


namespace probability_crossing_river_l294_294733

noncomputable def probability_of_crossing (jump_distance river_width : ℝ) : ℝ :=
  let successful_region := (4 - 2) in
  let total_region := river_width in
  successful_region / total_region

theorem probability_crossing_river :
  probability_of_crossing 4 6 = 1 / 3 :=
by
  unfold probability_of_crossing
  sorry

end probability_crossing_river_l294_294733


namespace cost_per_rose_l294_294307

theorem cost_per_rose (P : ℝ) (h1 : 5 * 12 = 60) (h2 : 0.8 * 60 * P = 288) : P = 6 :=
by
  -- Proof goes here
  sorry

end cost_per_rose_l294_294307


namespace complement_intersection_l294_294162

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection :
  U \ (A ∩ B) = {1, 4, 5} := by
    sorry

end complement_intersection_l294_294162


namespace gcd_lcm_sum_eq_90_l294_294316

def gcd_three (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem gcd_lcm_sum_eq_90 : 
  let A := gcd_three 18 36 72
  let B := lcm_three 18 36 72
  A + B = 90 :=
by
  let A := gcd_three 18 36 72
  let B := lcm_three 18 36 72
  sorry

end gcd_lcm_sum_eq_90_l294_294316


namespace johns_total_profit_l294_294143

theorem johns_total_profit
  (cost_price : ℝ) (selling_price : ℝ) (bags_sold : ℕ)
  (h_cost : cost_price = 4) (h_sell : selling_price = 8) (h_bags : bags_sold = 30) :
  (selling_price - cost_price) * bags_sold = 120 := by
    sorry

end johns_total_profit_l294_294143


namespace distance_from_point_to_y_axis_l294_294116

/-- Proof that the distance from point P(-4, 3) to the y-axis is 4. -/
theorem distance_from_point_to_y_axis {P : ℝ × ℝ} (hP : P = (-4, 3)) : |P.1| = 4 :=
by {
   -- The proof will depend on the properties of absolute value
   -- and the given condition about the coordinates of P.
   sorry
}

end distance_from_point_to_y_axis_l294_294116


namespace gemstones_needed_l294_294475

-- Define the initial quantities and relationships
def magnets_per_earring := 2
def buttons_per_magnet := 1 / 2
def gemstones_per_button := 3
def earrings_per_set := 2
def sets_of_earrings := 4

-- Define the total gemstones needed
theorem gemstones_needed : 
    let earrings := sets_of_earrings * earrings_per_set in
    let total_magnets := earrings * magnets_per_earring in
    let total_buttons := total_magnets * buttons_per_magnet in
    let total_gemstones := total_buttons * gemstones_per_button in
    total_gemstones = 24 :=
by
    have earrings := 2 * 4
    have total_magnets := earrings * 2
    have total_buttons := total_magnets / 2
    have total_gemstones := total_buttons * 3
    exact eq.refl 24

end gemstones_needed_l294_294475


namespace ways_to_distribute_balls_in_boxes_l294_294276

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l294_294276


namespace cos_alpha_plus_pi_over_3_l294_294607

theorem cos_alpha_plus_pi_over_3 (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) : Real.cos (α + π / 3) = -1 / 3 :=
  sorry

end cos_alpha_plus_pi_over_3_l294_294607


namespace marbles_problem_a_marbles_problem_b_l294_294676

-- Define the problem as Lean statements.

-- Part (a): m = 2004, n = 2006
theorem marbles_problem_a (m n : ℕ) (h_m : m = 2004) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) := 
sorry

-- Part (b): m = 2005, n = 2006
theorem marbles_problem_b (m n : ℕ) (h_m : m = 2005) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) → false := 
sorry

end marbles_problem_a_marbles_problem_b_l294_294676


namespace min_distance_between_curves_l294_294907

noncomputable def distance_between_intersections : ℝ :=
  let f (x : ℝ) := (2 * x + 1) - (x + Real.log x)
  let f' (x : ℝ) := 1 - 1 / x
  let minimum_distance :=
    if hs : 1 < 1 then 2 else
    if hs : 1 > 1 then 2 else
    2
  minimum_distance

theorem min_distance_between_curves : distance_between_intersections = 2 :=
by
  sorry

end min_distance_between_curves_l294_294907


namespace band_members_minimum_n_l294_294066

theorem band_members_minimum_n 
  (n : ℕ) 
  (h1 : n % 6 = 3) 
  (h2 : n % 8 = 5) 
  (h3 : n % 9 = 7) : 
  n ≥ 165 := 
sorry

end band_members_minimum_n_l294_294066


namespace value_of_x_l294_294433

theorem value_of_x (x : ℤ) (h : x + 3 = 4 ∨ x + 3 = -4) : x = 1 ∨ x = -7 := sorry

end value_of_x_l294_294433


namespace find_a_and_x_l294_294216

theorem find_a_and_x (a x : ℝ) (ha1 : x = (2 * a - 1)^2) (ha2 : x = (-a + 2)^2) : a = -1 ∧ x = 9 := 
by
  sorry

end find_a_and_x_l294_294216


namespace hens_count_l294_294071

theorem hens_count (H C : ℕ) (heads_eq : H + C = 44) (feet_eq : 2 * H + 4 * C = 140) : H = 18 := by
  sorry

end hens_count_l294_294071


namespace theater_total_bills_l294_294165

theorem theater_total_bills (tickets : ℕ) (price : ℕ) (x : ℕ) (number_of_5_bills : ℕ) (number_of_10_bills : ℕ) (number_of_20_bills : ℕ) :
  tickets = 300 →
  price = 40 →
  number_of_20_bills = x →
  number_of_10_bills = 2 * x →
  number_of_5_bills = 2 * x + 20 →
  20 * x + 10 * (2 * x) + 5 * (2 * x + 20) = tickets * price →
  number_of_5_bills + number_of_10_bills + number_of_20_bills = 1210 := by
    intro h_tickets h_price h_20_bills h_10_bills h_5_bills h_total
    sorry

end theater_total_bills_l294_294165


namespace angles_cosine_condition_l294_294003

theorem angles_cosine_condition {A B : ℝ} (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  (A > B) ↔ (Real.cos A < Real.cos B) :=
by
sorry

end angles_cosine_condition_l294_294003


namespace inscribe_circle_tangent_l294_294716

variables {α : Type*} [plane_geometry α]
variables (A B C O : α) (R : ℝ) (hR : R > 0)

theorem inscribe_circle_tangent
  (h_angle : is_angle BAC) 
  (h_given_circle : is_circle O R)
  : ∃ O1 r, is_circle O1 r ∧ is_tangent O1 r O R ∧ is_inscribed O1 r BAC :=
begin
  sorry
end

end inscribe_circle_tangent_l294_294716


namespace circles_non_intersecting_l294_294096

def circle1_equation (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4
def circle2_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem circles_non_intersecting :
    (∀ (x y : ℝ), ¬(circle1_equation x y ∧ circle2_equation x y)) :=
by
  sorry

end circles_non_intersecting_l294_294096


namespace num_ways_to_put_5_balls_into_4_boxes_l294_294283

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l294_294283


namespace factorize_polynomial_l294_294887

def p (a b : ℝ) : ℝ := a^2 - b^2 + 2 * a + 1

theorem factorize_polynomial (a b : ℝ) : 
  p a b = (a + 1 + b) * (a + 1 - b) :=
by
  sorry

end factorize_polynomial_l294_294887


namespace probability_ab_gt_a_add_b_l294_294516

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l294_294516


namespace number_of_people_in_group_l294_294032

/-- The number of people in the group N is such that when one of the people weighing 65 kg is replaced
by a new person weighing 100 kg, the average weight of the group increases by 3.5 kg. -/
theorem number_of_people_in_group (N : ℕ) (W : ℝ) 
  (h1 : (W + 35) / N = W / N + 3.5) 
  (h2 : W + 35 = W - 65 + 100) : 
  N = 10 :=
sorry

end number_of_people_in_group_l294_294032


namespace not_perfect_square_l294_294166

theorem not_perfect_square (n : ℤ) (hn : n > 4) : ¬ (∃ k : ℕ, n^2 - 3*n = k^2) :=
sorry

end not_perfect_square_l294_294166


namespace distance_to_hospital_l294_294718

theorem distance_to_hospital {total_paid base_price price_per_mile : ℝ} (h1 : total_paid = 23) (h2 : base_price = 3) (h3 : price_per_mile = 4) : (total_paid - base_price) / price_per_mile = 5 :=
by
  sorry

end distance_to_hospital_l294_294718


namespace company_match_percentage_l294_294263

theorem company_match_percentage (total_contribution : ℝ) (holly_contribution_per_paycheck : ℝ) (total_paychecks : ℕ) (total_contribution_one_year : ℝ) : 
  let holly_contribution := holly_contribution_per_paycheck * total_paychecks
  let company_contribution := total_contribution_one_year - holly_contribution
  (company_contribution / holly_contribution) * 100 = 6 :=
by
  let holly_contribution := holly_contribution_per_paycheck * total_paychecks
  let company_contribution := total_contribution_one_year - holly_contribution
  have h : holly_contribution = 2600 := by sorry
  have c : company_contribution = 156 := by sorry
  exact sorry

end company_match_percentage_l294_294263


namespace speed_of_train_l294_294059

-- Define the conditions
def length_of_train : ℕ := 240
def length_of_bridge : ℕ := 150
def time_to_cross : ℕ := 20

-- Compute the expected speed of the train
def expected_speed : ℝ := 19.5

-- The statement that needs to be proven
theorem speed_of_train : (length_of_train + length_of_bridge) / time_to_cross = expected_speed := by
  -- sorry is used to skip the actual proof
  sorry

end speed_of_train_l294_294059


namespace fg_of_neg3_eq_3_l294_294680

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_of_neg3_eq_3 : f (g (-3)) = 3 :=
by
  sorry

end fg_of_neg3_eq_3_l294_294680


namespace appropriate_word_count_l294_294021

-- Define the conditions of the problem
def min_minutes := 40
def max_minutes := 55
def words_per_minute := 120

-- Define the bounds for the number of words
def min_words := min_minutes * words_per_minute
def max_words := max_minutes * words_per_minute

-- Define the appropriate number of words
def appropriate_words (words : ℕ) : Prop :=
  words >= min_words ∧ words <= max_words

-- The specific numbers to test
def words1 := 5000
def words2 := 6200

-- The main proof statement
theorem appropriate_word_count : 
  appropriate_words words1 ∧ appropriate_words words2 :=
by
  -- We do not need to provide the proof steps, just state the theorem
  sorry

end appropriate_word_count_l294_294021


namespace sum_lt_prod_probability_l294_294523

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l294_294523


namespace sample_size_eq_36_l294_294573

def total_population := 27 + 54 + 81
def ratio_elderly_total := 27 / total_population
def selected_elderly := 6
def sample_size := 36

theorem sample_size_eq_36 : 
  (selected_elderly : ℚ) / (sample_size : ℚ) = ratio_elderly_total → 
  sample_size = 36 := 
by 
sorry

end sample_size_eq_36_l294_294573


namespace problem_statement_l294_294220

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

def candidate_function (x : ℝ) : ℝ :=
  x * |x|

theorem problem_statement : is_odd_function candidate_function ∧ is_increasing_function candidate_function :=
by
  sorry

end problem_statement_l294_294220


namespace largest_multiple_of_8_less_than_100_l294_294995

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l294_294995


namespace vector_identity_l294_294794

-- Definitions of the vectors
variables {V : Type*} [AddGroup V]

-- Conditions as Lean definitions
def cond1 (AB BO AO : V) : Prop := AB + BO = AO
def cond2 (AO OM AM : V) : Prop := AO + OM = AM
def cond3 (AM MB AB : V) : Prop := AM + MB = AB

-- The main statement to be proved
theorem vector_identity (AB MB BO BC OM AO AM AC : V) 
  (h1 : cond1 AB BO AO) 
  (h2 : cond2 AO OM AM) 
  (h3 : cond3 AM MB AB) 
  : (AB + MB) + (BO + BC) + OM = AC :=
sorry

end vector_identity_l294_294794


namespace preimage_exists_l294_294908

-- Define the mapping function f
def f (x y : ℚ) : ℚ × ℚ :=
  (x + 2 * y, 2 * x - y)

-- Define the statement
theorem preimage_exists (x y : ℚ) :
  f x y = (3, 1) → (x, y) = (-1/3, 5/3) :=
by
  sorry

end preimage_exists_l294_294908


namespace ball_box_problem_l294_294269

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l294_294269


namespace trigonometric_inequality_l294_294688

theorem trigonometric_inequality (x : ℝ) (n m : ℕ) 
  (hx : 0 < x ∧ x < (Real.pi / 2))
  (hnm : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤
  3 * |Real.sin x ^ m - Real.cos x ^ m| := 
by 
  sorry

end trigonometric_inequality_l294_294688


namespace find_xy_l294_294242

theorem find_xy (x y : ℝ) (k : ℤ) :
  3 * Real.sin x - 4 * Real.cos x = 4 * y^2 + 4 * y + 6 ↔
  (x = -Real.arccos (-4/5) + (2 * k + 1) * Real.pi ∧ y = -1/2) := by
  sorry

end find_xy_l294_294242


namespace mean_of_solutions_l294_294892

open Polynomial Rat

theorem mean_of_solutions (f : ℚ[X]) (h_f : f = X^3 + 5 * X^2 - 14 * X) :
  mean_of_solutions f = -5/3 :=
by
  sorry

noncomputable def mean_of_solutions (f : Polynomial ℚ) : ℚ :=
if h_solutions : (roots f).toList.length > 0 then
  (roots f).toList.sum / (roots f).toList.length
else 0

end mean_of_solutions_l294_294892


namespace quadratic_pairs_square_diff_exists_l294_294073

open Nat Polynomial

theorem quadratic_pairs_square_diff_exists (P : Polynomial ℤ) (u v w a b n : ℤ) (n_pos : 0 < n)
    (hp : ∃ (u v w : ℤ), P = C u * X ^ 2 + C v * X + C w)
    (h_ab : P.eval a - P.eval b = n^2) : ∃ k > 10^6, ∃ m : ℕ, ∃ c d : ℤ, (c - d = a - b + 2 * k) ∧ 
    (P.eval c - P.eval d = n^2 * m ^ 2) :=
by
  sorry

end quadratic_pairs_square_diff_exists_l294_294073


namespace incorrect_calculation_l294_294481

noncomputable def ξ : ℝ := 3
noncomputable def η : ℝ := 5

def T (ξ η : ℝ) : ℝ := min ξ η

theorem incorrect_calculation : E[T(ξ, η)] ≤ 3 := by
  sorry

end incorrect_calculation_l294_294481


namespace probability_ab_gt_a_add_b_l294_294517

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l294_294517


namespace tan_neg_585_eq_neg_1_l294_294402

theorem tan_neg_585_eq_neg_1 : Real.tan (-585 * Real.pi / 180) = -1 := by
  sorry

end tan_neg_585_eq_neg_1_l294_294402


namespace grid_to_black_probability_l294_294571

theorem grid_to_black_probability :
  let n := 16
  let p_black_after_rotation := 3 / 4
  (p_black_after_rotation ^ n) = (3 / 4) ^ 16 :=
by
  -- Proof goes here
  sorry

end grid_to_black_probability_l294_294571


namespace find_k_l294_294429

-- Define vector a and vector b
def vec_a : (ℝ × ℝ) := (1, 1)
def vec_b : (ℝ × ℝ) := (-3, 1)

-- Define the expression for k * vec_a - vec_b
def k_vec_a_minus_vec_b (k : ℝ) : ℝ × ℝ :=
  (k * vec_a.1 - vec_b.1, k * vec_a.2 - vec_b.2)

-- Define the dot product condition for perpendicular vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved: k = -1 is the value that makes the dot product zero
theorem find_k : ∃ k : ℝ, dot_product (k_vec_a_minus_vec_b k) vec_a = 0 :=
by
  use -1
  sorry

end find_k_l294_294429


namespace pen_cost_is_4_l294_294215

variable (penCost pencilCost : ℝ)

-- Conditions
def totalCost := penCost + pencilCost = 6
def costRelation := penCost = 2 * pencilCost

-- Theorem to be proved
theorem pen_cost_is_4 (h1 : totalCost) (h2 : costRelation) : penCost = 4 :=
by
  rw [totalCost, costRelation] at h1
  sorry

end pen_cost_is_4_l294_294215


namespace cloth_sold_worth_l294_294734

-- Define the commission rate and commission received
def commission_rate := 0.05
def commission_received := 12.50

-- State the theorem to be proved
theorem cloth_sold_worth : commission_received / commission_rate = 250 :=
by
  sorry

end cloth_sold_worth_l294_294734


namespace geoff_tuesday_multiple_l294_294599

variable (monday_spending : ℝ) (tuesday_multiple : ℝ) (total_spending : ℝ)

-- Given conditions
def geoff_conditions (monday_spending tuesday_multiple total_spending : ℝ) : Prop :=
  monday_spending = 60 ∧
  (tuesday_multiple * monday_spending) + (5 * monday_spending) + monday_spending = total_spending ∧
  total_spending = 600

-- Proof goal
theorem geoff_tuesday_multiple (monday_spending tuesday_multiple total_spending : ℝ)
  (h : geoff_conditions monday_spending tuesday_multiple total_spending) : 
  tuesday_multiple = 4 :=
by
  sorry

end geoff_tuesday_multiple_l294_294599


namespace bogan_maggots_l294_294399

theorem bogan_maggots (x : ℕ) (total_maggots : ℕ) (eaten_first : ℕ) (eaten_second : ℕ) (thrown_out : ℕ) 
  (h1 : eaten_first = 1) (h2 : eaten_second = 3) (h3 : total_maggots = 20) (h4 : thrown_out = total_maggots - eaten_first - eaten_second) 
  (h5 : x + eaten_first = thrown_out) : x = 15 :=
by
  -- Use the given conditions
  sorry

end bogan_maggots_l294_294399


namespace solve_for_nabla_l294_294799

theorem solve_for_nabla (nabla mu : ℤ) (h1 : 5 * (-3) = nabla + mu - 3) (h2 : mu = 4) : 
  nabla = -16 := 
by
  sorry

end solve_for_nabla_l294_294799


namespace beth_lost_red_marbles_l294_294752

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

end beth_lost_red_marbles_l294_294752


namespace inclination_angle_range_l294_294612

theorem inclination_angle_range :
  let Γ := fun x y : ℝ => x * abs x + y * abs y = 1
  let line (m : ℝ) := fun x y : ℝ => y = m * (x - 1)
  ∀ m : ℝ,
  (∃ p1 p2 p3 : ℝ × ℝ, 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    line m p1.1 p1.2 ∧ Γ p1.1 p1.2 ∧ 
    line m p2.1 p2.2 ∧ Γ p2.1 p2.2 ∧ 
    line m p3.1 p3.2 ∧ Γ p3.1 p3.2) →
  (∃ θ : ℝ, θ ∈ (Set.Ioo (Real.pi / 2) (3 * Real.pi / 4) ∪ 
                  Set.Ioo (3 * Real.pi / 4) (Real.pi - Real.arctan (Real.sqrt 2 / 2)))) :=
sorry

end inclination_angle_range_l294_294612


namespace person_speed_in_kmph_l294_294369

-- Define the distance in meters
def distance_meters : ℕ := 300

-- Define the time in minutes
def time_minutes : ℕ := 4

-- Function to convert distance from meters to kilometers
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000

-- Function to convert time from minutes to hours
def minutes_to_hours (min : ℕ) : ℚ := min / 60

-- Define the expected speed in km/h
def expected_speed : ℚ := 4.5

-- Proof statement
theorem person_speed_in_kmph : 
  meters_to_kilometers distance_meters / minutes_to_hours time_minutes = expected_speed :=
by 
  -- This is where the steps to verify the theorem would be located, currently omitted for the sake of the statement.
  sorry

end person_speed_in_kmph_l294_294369


namespace remaining_dogs_eq_200_l294_294711

def initial_dogs : ℕ := 200
def additional_dogs : ℕ := 100
def first_adoption : ℕ := 40
def second_adoption : ℕ := 60

def total_dogs_after_adoption : ℕ :=
  initial_dogs + additional_dogs - first_adoption - second_adoption

theorem remaining_dogs_eq_200 : total_dogs_after_adoption = 200 :=
by
  -- Omitted the proof as requested
  sorry

end remaining_dogs_eq_200_l294_294711


namespace fraction_ordering_l294_294478

noncomputable def t1 : ℝ := (100^100 + 1) / (100^90 + 1)
noncomputable def t2 : ℝ := (100^99 + 1) / (100^89 + 1)
noncomputable def t3 : ℝ := (100^101 + 1) / (100^91 + 1)
noncomputable def t4 : ℝ := (101^101 + 1) / (101^91 + 1)
noncomputable def t5 : ℝ := (101^100 + 1) / (101^90 + 1)
noncomputable def t6 : ℝ := (99^99 + 1) / (99^89 + 1)
noncomputable def t7 : ℝ := (99^100 + 1) / (99^90 + 1)

theorem fraction_ordering : t6 < t7 ∧ t7 < t2 ∧ t2 < t1 ∧ t1 < t3 ∧ t3 < t5 ∧ t5 < t4 := by
  sorry

end fraction_ordering_l294_294478


namespace average_marks_l294_294493

theorem average_marks (total_students : ℕ) (first_group : ℕ) (first_group_marks : ℕ)
                      (second_group : ℕ) (second_group_marks_diff : ℕ) (third_group_marks : ℕ)
                      (total_marks : ℕ) (class_average : ℕ) :
  total_students = 50 → 
  first_group = 10 → 
  first_group_marks = 90 → 
  second_group = 15 → 
  second_group_marks_diff = 10 → 
  third_group_marks = 60 →
  total_marks = (first_group * first_group_marks) + (second_group * (first_group_marks - second_group_marks_diff)) + ((total_students - (first_group + second_group)) * third_group_marks) →
  class_average = total_marks / total_students →
  class_average = 72 :=
by
  intros
  sorry

end average_marks_l294_294493


namespace koi_fish_in_pond_l294_294208

theorem koi_fish_in_pond : ∃ k : ℤ, 2 * k - 14 = 64 ∧ k = 39 := by
  use 39
  split
  · sorry
  · rfl

end koi_fish_in_pond_l294_294208


namespace ivan_years_l294_294695

theorem ivan_years (years months weeks days hours : ℕ) (h1 : years = 48) (h2 : months = 48)
    (h3 : weeks = 48) (h4 : days = 48) (h5 : hours = 48) :
    (53 : ℕ) = (years + (months / 12) + ((weeks * 7 + days) / 365) + ((hours / 24) / 365)) := by
  sorry

end ivan_years_l294_294695


namespace area_enclosed_by_lines_and_curve_cylindrical_to_cartesian_coordinates_complex_number_evaluation_area_of_triangle_AOB_l294_294867

-- 1. Prove that the area enclosed by x = π/2, x = 3π/2, y = 0 and y = cos x is 2
theorem area_enclosed_by_lines_and_curve : 
  ∫ (x : ℝ) in (Real.pi / 2)..(3 * Real.pi / 2), (-Real.cos x) = 2 := sorry

-- 2. Prove that the cylindrical coordinates (sqrt(2), π/4, 1) correspond to Cartesian coordinates (1, 1, 1)
theorem cylindrical_to_cartesian_coordinates :
  let r := Real.sqrt 2
  let θ := Real.pi / 4
  let z := 1
  (r * Real.cos θ, r * Real.sin θ, z) = (1, 1, 1) := sorry

-- 3. Prove that (3 + 2i) / (2 - 3i) - (3 - 2i) / (2 + 3i) = 2i
theorem complex_number_evaluation : 
  ((3 + 2 * Complex.I) / (2 - 3 * Complex.I)) - ((3 - 2 * Complex.I) / (2 + 3 * Complex.I)) = 2 * Complex.I := sorry

-- 4. Prove that the area of triangle AOB with given polar coordinates is 2
theorem area_of_triangle_AOB :
  let A := (2, Real.pi / 6)
  let B := (4, Real.pi / 3)
  let area := 1 / 2 * (2 * 4 * Real.sin (Real.pi / 3 - Real.pi / 6))
  area = 2 := sorry

end area_enclosed_by_lines_and_curve_cylindrical_to_cartesian_coordinates_complex_number_evaluation_area_of_triangle_AOB_l294_294867


namespace ryan_bread_slices_l294_294338

theorem ryan_bread_slices 
  (num_pb_people : ℕ)
  (pb_sandwiches_per_person : ℕ)
  (num_tuna_people : ℕ)
  (tuna_sandwiches_per_person : ℕ)
  (num_turkey_people : ℕ)
  (turkey_sandwiches_per_person : ℕ)
  (slices_per_pb_sandwich : ℕ)
  (slices_per_tuna_sandwich : ℕ)
  (slices_per_turkey_sandwich : ℝ)
  (h1 : num_pb_people = 4)
  (h2 : pb_sandwiches_per_person = 2)
  (h3 : num_tuna_people = 3)
  (h4 : tuna_sandwiches_per_person = 3)
  (h5 : num_turkey_people = 2)
  (h6 : turkey_sandwiches_per_person = 1)
  (h7 : slices_per_pb_sandwich = 2)
  (h8 : slices_per_tuna_sandwich = 3)
  (h9 : slices_per_turkey_sandwich = 1.5) : 
  (num_pb_people * pb_sandwiches_per_person * slices_per_pb_sandwich 
  + num_tuna_people * tuna_sandwiches_per_person * slices_per_tuna_sandwich 
  + (num_turkey_people * turkey_sandwiches_per_person : ℝ) * slices_per_turkey_sandwich) = 46 :=
by
  sorry

end ryan_bread_slices_l294_294338


namespace perpendicular_line_eqn_l294_294306

noncomputable def line_intersection_with_x_axis (a b c : ℝ) (h : b ≠ 0) : ℝ := (-c / b)

theorem perpendicular_line_eqn (x y : ℝ) (h : 2 * x - y - 4 = 0) :
  x + 2 * y - 2 = 0 :=
by {
  have slope_l : ℝ := 2,
  have slope_perpendicular : ℝ := -1 / 2,
  have M_x_coord : ℝ := line_intersection_with_x_axis 2 (-1) (-4) (ne_of_lt (by norm_num)),
  have M : ℝ × ℝ := (M_x_coord, 0),
  sorry
}

end perpendicular_line_eqn_l294_294306


namespace sum_of_integers_from_neg15_to_5_l294_294590

-- defining the conditions
def first_term : ℤ := -15
def last_term : ℤ := 5

-- sum of integers from first_term to last_term
def sum_arithmetic_series (a l : ℤ) : ℤ :=
  let n := l - a + 1
  (n * (a + l)) / 2

-- the statement we need to prove
theorem sum_of_integers_from_neg15_to_5 : sum_arithmetic_series first_term last_term = -105 := by
  sorry

end sum_of_integers_from_neg15_to_5_l294_294590


namespace no_triangle_possible_l294_294850

-- Define the lengths of the sticks
def stick_lengths : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

-- The theorem stating the impossibility of forming a triangle with any combination of these lengths
theorem no_triangle_possible : ¬ ∃ (a b c : ℕ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  sorry

end no_triangle_possible_l294_294850


namespace items_purchased_total_profit_l294_294212

-- Definitions based on conditions given in part (a)
def total_cost := 6000
def cost_A := 22
def cost_B := 30
def sell_A := 29
def sell_B := 40

-- Proven answers from the solution (part (b))
def items_A := 150
def items_B := 90
def profit := 1950

-- Lean theorem statements (problems to be proved)
theorem items_purchased : (22 * items_A + 30 * (items_A / 2 + 15) = total_cost) → 
                          (items_A = 150) ∧ (items_B = 90) := sorry

theorem total_profit : (items_A = 150) → (items_B = 90) → 
                       ((items_A * (sell_A - cost_A) + items_B * (sell_B - cost_B)) = profit) := sorry

end items_purchased_total_profit_l294_294212


namespace only_1996_is_leap_l294_294224

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0))

def is_leap_year_1996 := is_leap_year 1996
def is_leap_year_1998 := is_leap_year 1998
def is_leap_year_2010 := is_leap_year 2010
def is_leap_year_2100 := is_leap_year 2100

theorem only_1996_is_leap : 
  is_leap_year_1996 ∧ ¬is_leap_year_1998 ∧ ¬is_leap_year_2010 ∧ ¬is_leap_year_2100 :=
by 
  -- proof will be added here later
  sorry

end only_1996_is_leap_l294_294224


namespace number_of_shirts_is_20_l294_294442

/-- Given the conditions:
1. The total price for some shirts is 360,
2. The total price for 45 sweaters is 900,
3. The average price of a sweater exceeds that of a shirt by 2,
prove that the number of shirts is 20. -/

theorem number_of_shirts_is_20
  (S : ℕ) (P_shirt P_sweater : ℝ)
  (h1 : S * P_shirt = 360)
  (h2 : 45 * P_sweater = 900)
  (h3 : P_sweater = P_shirt + 2) :
  S = 20 :=
by
  sorry

end number_of_shirts_is_20_l294_294442


namespace geometric_sequence_sum_5_l294_294668

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ i j : ℕ, ∃ r : ℝ, a (i + 1) = a i * r ∧ a (j + 1) = a j * r

theorem geometric_sequence_sum_5
  (a : ℕ → ℝ)
  (h : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : a 2 * a 6 + 2 * a 4 * a 5 + (a 5) ^ 2 = 25) :
  a 4 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_5_l294_294668


namespace balls_in_boxes_l294_294264

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l294_294264


namespace prove_intersection_points_l294_294781

noncomputable def sqrt5 := Real.sqrt 5

def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 5 / 2
def curve2 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1
def curve3 (x y : ℝ) : Prop := x^2 + y^2 / 4 = 1
def curve4 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def line (x y : ℝ) : Prop := x + y = sqrt5

theorem prove_intersection_points :
  (∃! (x y : ℝ), curve1 x y ∧ line x y) ∧
  (∃! (x y : ℝ), curve3 x y ∧ line x y) ∧
  (∃! (x y : ℝ), curve4 x y ∧ line x y) :=
by
  sorry

end prove_intersection_points_l294_294781


namespace teacher_allocation_l294_294413

theorem teacher_allocation :
  ∃ n : ℕ, n = 150 ∧ 
  (∀ t1 t2 t3 t4 t5 : Prop, -- represent the five teachers
    ∃ s1 s2 s3 : Prop, -- represent the three schools
      s1 ∧ s2 ∧ s3 ∧ -- each school receives at least one teacher
        ((t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5) ∧ -- allocation condition
         (t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5) ∧
         (t1 ∨ t2 ∨ t3 ∨ t4 ∨ t5))) := sorry

end teacher_allocation_l294_294413


namespace tom_needs_44000_pounds_salt_l294_294513

theorem tom_needs_44000_pounds_salt 
  (flour_needed : ℕ)
  (flour_bag_weight : ℕ)
  (flour_bag_cost : ℕ)
  (salt_cost_per_pound : ℝ)
  (promotion_cost : ℕ)
  (ticket_price : ℕ)
  (tickets_sold : ℕ)
  (total_revenue : ℕ) 
  (expected_salt_cost : ℝ) 
  (S : ℝ) : 
  flour_needed = 500 → 
  flour_bag_weight = 50 → 
  flour_bag_cost = 20 → 
  salt_cost_per_pound = 0.2 → 
  promotion_cost = 1000 → 
  ticket_price = 20 → 
  tickets_sold = 500 → 
  total_revenue = 8798 → 
  0.2 * S = (500 * 20) - (500 / 50) * 20 - 1000 →
  S = 44000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end tom_needs_44000_pounds_salt_l294_294513


namespace probability_two_slate_rocks_l294_294041

theorem probability_two_slate_rocks :
  let slate := 12
  let pumice := 16
  let granite := 8
  let total := slate + pumice + granite
  (slate / total) * ((slate - 1) / (total - 1)) = 11 / 105 :=
by 
  sorry

end probability_two_slate_rocks_l294_294041


namespace contrapositive_statement_l294_294839

-- Definitions derived from conditions
def Triangle (ABC : Type) : Prop := 
  ∃ a b c : ABC, true

def IsIsosceles (ABC : Type) : Prop :=
  ∃ a b c : ABC, a = b ∨ b = c ∨ a = c

def InteriorAnglesNotEqual (ABC : Type) : Prop :=
  ∀ a b : ABC, a ≠ b

-- The contrapositive implication we need to prove
theorem contrapositive_statement (ABC : Type) (h : Triangle ABC) 
  (h_not_isosceles_implies_not_equal : ¬IsIsosceles ABC → InteriorAnglesNotEqual ABC) :
  (∃ a b c : ABC, a = b → IsIsosceles ABC) := 
sorry

end contrapositive_statement_l294_294839


namespace parking_lot_wheels_l294_294406

-- Define the total number of wheels for each type of vehicle
def car_wheels (n : ℕ) : ℕ := n * 4
def motorcycle_wheels (n : ℕ) : ℕ := n * 2
def truck_wheels (n : ℕ) : ℕ := n * 6
def van_wheels (n : ℕ) : ℕ := n * 4

-- Number of each type of guests' vehicles
def num_cars : ℕ := 5
def num_motorcycles : ℕ := 4
def num_trucks : ℕ := 3
def num_vans : ℕ := 2

-- Number of parents' vehicles and their wheels
def parents_car_wheels : ℕ := 4
def parents_jeep_wheels : ℕ := 4

-- Summing up all the wheels
def total_wheels : ℕ :=
  car_wheels num_cars +
  motorcycle_wheels num_motorcycles +
  truck_wheels num_trucks +
  van_wheels num_vans +
  parents_car_wheels +
  parents_jeep_wheels

theorem parking_lot_wheels : total_wheels = 62 := by
  sorry

end parking_lot_wheels_l294_294406


namespace polynomial_sum_l294_294943

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l294_294943


namespace interest_earned_l294_294151

-- Define the principal, interest rate, and number of years
def principal : ℝ := 1200
def annualInterestRate : ℝ := 0.12
def numberOfYears : ℕ := 4

-- Define the compound interest formula
def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Define the total interest earned
def totalInterest (P A : ℝ) : ℝ :=
  A - P

-- State the theorem
theorem interest_earned :
  totalInterest principal (compoundInterest principal annualInterestRate numberOfYears) = 688.224 :=
by
  sorry

end interest_earned_l294_294151


namespace ticket_sales_revenue_l294_294580

theorem ticket_sales_revenue (total_tickets advance_tickets same_day_tickets price_advance price_same_day: ℕ) 
    (h1: total_tickets = 60) 
    (h2: price_advance = 20) 
    (h3: price_same_day = 30) 
    (h4: advance_tickets = 20) 
    (h5: same_day_tickets = total_tickets - advance_tickets):
    advance_tickets * price_advance + same_day_tickets * price_same_day = 1600 := 
by
  sorry

end ticket_sales_revenue_l294_294580


namespace geom_series_min_q_l294_294157

theorem geom_series_min_q (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h_geom : ∃ k : ℝ, q = p * k ∧ r = q * k)
  (hpqr : p * q * r = 216) : q = 6 :=
sorry

end geom_series_min_q_l294_294157


namespace range_of_function_l294_294364

theorem range_of_function :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x + 2)) ↔ (y ∈ Set.Iio 1 ∨ y ∈ Set.Ioi 1) := 
sorry

end range_of_function_l294_294364


namespace inradii_sum_l294_294026

theorem inradii_sum (ABCD : Type) (r_a r_b r_c r_d : ℝ) 
  (inscribed_quadrilateral : Prop) 
  (inradius_BCD : Prop) 
  (inradius_ACD : Prop) 
  (inradius_ABD : Prop) 
  (inradius_ABC : Prop) 
  (Tebo_theorem : Prop) :
  r_a + r_c = r_b + r_d := 
by
  sorry

end inradii_sum_l294_294026


namespace negation_of_proposition_l294_294489

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) :=
sorry

end negation_of_proposition_l294_294489


namespace sum_of_all_n_l294_294552

-- Definitions based on the problem statement
def is_integer_fraction (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def is_odd_divisor (a b : ℤ) : Prop := b % 2 = 1 ∧ ∃ k : ℤ, a = b * k

-- Problem Statement
theorem sum_of_all_n (S : ℤ) :
  (S = ∑ n in {n : ℤ | is_integer_fraction 36 (2 * n - 1)}, n) →
  S = 8 :=
by
  sorry

end sum_of_all_n_l294_294552


namespace alien_saturday_sequence_l294_294084

def a_1 : String := "A"
def a_2 : String := "AY"
def a_3 : String := "AYYA"
def a_4 : String := "AYYAYAAY"

noncomputable def a_5 : String := a_4 ++ "YAAYAYYA"
noncomputable def a_6 : String := a_5 ++ "YAAYAYYAAAYAYAAY"

theorem alien_saturday_sequence : 
  a_6 = "AYYAYAAYYAAYAYYAYAAYAYYAAAYAYAAY" :=
sorry

end alien_saturday_sequence_l294_294084


namespace carlos_laundry_time_l294_294880

def washing_time1 := 30
def washing_time2 := 45
def washing_time3 := 40
def washing_time4 := 50
def washing_time5 := 35
def drying_time1 := 85
def drying_time2 := 95

def total_laundry_time := washing_time1 + washing_time2 + washing_time3 + washing_time4 + washing_time5 + drying_time1 + drying_time2

theorem carlos_laundry_time : total_laundry_time = 380 :=
by
  sorry

end carlos_laundry_time_l294_294880


namespace traveler_drank_32_ounces_l294_294394

-- Definition of the given condition
def total_gallons : ℕ := 2
def ounces_per_gallon : ℕ := 128
def total_ounces := total_gallons * ounces_per_gallon
def camel_multiple : ℕ := 7
def traveler_ounces (T : ℕ) := T
def camel_ounces (T : ℕ) := camel_multiple * T
def total_drunk (T : ℕ) := traveler_ounces T + camel_ounces T

-- Theorem to prove
theorem traveler_drank_32_ounces :
  ∃ T : ℕ, total_drunk T = total_ounces ∧ T = 32 :=
by 
  sorry

end traveler_drank_32_ounces_l294_294394


namespace tiles_needed_to_cover_floor_l294_294450

-- Definitions of the conditions
def room_length : ℕ := 2
def room_width : ℕ := 12
def tile_area : ℕ := 4

-- The proof statement: calculate the number of tiles needed to cover the entire floor
theorem tiles_needed_to_cover_floor : 
  (room_length * room_width) / tile_area = 6 := 
by 
  sorry

end tiles_needed_to_cover_floor_l294_294450


namespace largest_multiple_of_8_less_than_100_l294_294990

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l294_294990


namespace min_value_of_expression_l294_294432

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 1) :
  (2 / x + 5 / y) ≥ 2 := sorry

end min_value_of_expression_l294_294432


namespace value_of_y_l294_294985

variable {x y : ℝ}

theorem value_of_y (h1 : x > 2) (h2 : y > 2) (h3 : 1/x + 1/y = 3/4) (h4 : x * y = 8) : y = 4 :=
sorry

end value_of_y_l294_294985


namespace find_ratios_sum_l294_294085
noncomputable def Ana_biking_rate : ℝ := 8.6
noncomputable def Bob_biking_rate : ℝ := 6.2
noncomputable def CAO_biking_rate : ℝ := 5

variable (a b c : ℝ)

-- Conditions  
def Ana_distance := 2 * a + b + c = Ana_biking_rate
def Bob_distance := b + c = Bob_biking_rate
def Cao_distance := Real.sqrt (b^2 + c^2) = CAO_biking_rate

-- Main statement
theorem find_ratios_sum : 
  Ana_distance a b c ∧ 
  Bob_distance b c ∧ 
  Cao_distance b c →
  ∃ (p q r : ℕ), p + q + r = 37 ∧ Nat.gcd p q = 1 ∧ ((a / c) = p / r) ∧ ((b / c) = q / r) ∧ ((a / b) = p / q) :=
sorry

end find_ratios_sum_l294_294085


namespace value_of_a_l294_294379

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_a (a : ℝ) (f_symmetric : ∀ x y : ℝ, y = f x ↔ -y = 2^(-x + a)) (sum_f_condition : f (-2) + f (-4) = 1) :
  a = 2 :=
sorry

end value_of_a_l294_294379


namespace balls_into_boxes_problem_l294_294598

theorem balls_into_boxes_problem :
  ∃ (n : ℕ), n = 144 ∧ ∃ (balls : Fin 4 → ℕ), 
  (∃ (boxes : Fin 4 → Fin 4), 
    (∀ (b : Fin 4), boxes b < 4 ∧ boxes b ≠ b) ∧ 
    (∃! (empty_box : Fin 4), ∀ (b : Fin 4), (boxes b = empty_box) → false)) := 
by
  sorry

end balls_into_boxes_problem_l294_294598


namespace solve_equation_l294_294030

theorem solve_equation (x : ℝ) : (x + 4)^2 = 5 * (x + 4) ↔ (x = -4 ∨ x = 1) :=
by sorry

end solve_equation_l294_294030


namespace number_of_passed_candidates_l294_294177

variables (P F : ℕ) (h1 : P + F = 100)
          (h2 : P * 70 + F * 20 = 100 * 50)
          (h3 : ∀ p, p = P → 70 * p = 70 * P)
          (h4 : ∀ f, f = F → 20 * f = 20 * F)

theorem number_of_passed_candidates (P F : ℕ) (h1 : P + F = 100) 
                                    (h2 : P * 70 + F * 20 = 100 * 50) 
                                    (h3 : ∀ p, p = P → 70 * p = 70 * P) 
                                    (h4 : ∀ f, f = F → 20 * f = 20 * F) : 
  P = 60 :=
sorry

end number_of_passed_candidates_l294_294177


namespace find_value_of_k_l294_294138

def line_equation_holds (m n : ℤ) : Prop := m = 2 * n + 5
def second_point_condition (m n k : ℤ) : Prop := m + 4 = 2 * (n + k) + 5

theorem find_value_of_k (m n k : ℤ) 
  (h1 : line_equation_holds m n) 
  (h2 : second_point_condition m n k) : 
  k = 2 :=
by sorry

end find_value_of_k_l294_294138


namespace difference_of_squares_l294_294124

theorem difference_of_squares (x : ℤ) (h : x^2 = 1521) : (x + 1) * (x - 1) = 1520 := by
  sorry

end difference_of_squares_l294_294124


namespace arithmetic_sequence_problem_l294_294810

noncomputable def a_n (n : ℕ) (a d : ℝ) : ℝ := a + (n - 1) * d

theorem arithmetic_sequence_problem (a d : ℝ) 
  (h : a_n 1 a d - a_n 4 a d - a_n 8 a d - a_n 12 a d + a_n 15 a d = 2) :
  a_n 3 a d + a_n 13 a d = -4 :=
by
  sorry

end arithmetic_sequence_problem_l294_294810


namespace remainder_x_squared_l294_294248

theorem remainder_x_squared (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 7 * x ≡ 14 [ZMOD 20]) : 
  (x^2 ≡ 4 [ZMOD 20]) :=
sorry

end remainder_x_squared_l294_294248


namespace find_unknown_number_l294_294655

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end find_unknown_number_l294_294655


namespace total_bowling_balls_l294_294184

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l294_294184


namespace perimeter_ratio_l294_294175

def original_paper : ℕ × ℕ := (12, 8)
def folded_paper : ℕ × ℕ := (original_paper.1, original_paper.2 / 2)
def small_rectangle : ℕ × ℕ := (folded_paper.1 / 2, folded_paper.2)

def perimeter (rect : ℕ × ℕ) : ℕ :=
  2 * (rect.1 + rect.2)

theorem perimeter_ratio :
  perimeter small_rectangle = 1 / 2 * perimeter original_paper :=
by
  sorry

end perimeter_ratio_l294_294175


namespace water_fraction_after_replacements_l294_294210

-- Initially given conditions
def radiator_capacity : ℚ := 20
def initial_water_fraction : ℚ := 1
def antifreeze_quarts : ℚ := 5
def replacements : ℕ := 5

-- Derived condition
def water_remain_fraction : ℚ := 3 / 4

-- Statement of the problem
theorem water_fraction_after_replacements :
  (water_remain_fraction ^ replacements) = 243 / 1024 :=
by
  -- Proof goes here
  sorry

end water_fraction_after_replacements_l294_294210


namespace prob_three_students_exactly_two_absent_l294_294438

def prob_absent : ℚ := 1 / 30
def prob_present : ℚ := 29 / 30

theorem prob_three_students_exactly_two_absent :
  (prob_absent * prob_absent * prob_present) * 3 = 29 / 9000 := by
  sorry

end prob_three_students_exactly_two_absent_l294_294438


namespace probability_ab_gt_a_add_b_l294_294514

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l294_294514


namespace exists_a_b_not_multiple_p_l294_294418

theorem exists_a_b_not_multiple_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ a b : ℤ, ∀ m : ℤ, ¬ (m^3 + 2017 * a * m + b) ∣ (p : ℤ) :=
sorry

end exists_a_b_not_multiple_p_l294_294418


namespace triangle_ratio_l294_294855

theorem triangle_ratio (a b c : ℝ) (P Q : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c)
  (h₄ : P > 0) (h₅ : Q > P) (h₆ : Q < c) (h₇ : P = 21) (h₈ : Q - P = 35) (h₉ : c - Q = 100)
  (h₁₀ : P + (Q - P) + (c - Q) = c)
  (angle_trisect : ∃ x y : ℝ, x ≠ y ∧ x = a / b ∧ y = 7 / 45) :
  ∃ p q r : ℕ, p + q + r = 92 ∧ p.gcd r = 1 ∧ ¬ ∃ k : ℕ, k^2 ∣ q := sorry

end triangle_ratio_l294_294855


namespace largest_integer_m_dividing_30_factorial_l294_294890

theorem largest_integer_m_dividing_30_factorial :
  ∃ (m : ℕ), (∀ (k : ℕ), (18^k ∣ Nat.factorial 30) ↔ k ≤ m) ∧ m = 7 := by
  sorry

end largest_integer_m_dividing_30_factorial_l294_294890


namespace jack_money_proof_l294_294931

variable (sock_cost : ℝ) (number_of_socks : ℕ) (shoe_cost : ℝ) (available_money : ℝ)

def jack_needs_more_money : Prop := 
  (number_of_socks * sock_cost + shoe_cost) - available_money = 71

theorem jack_money_proof (h1 : sock_cost = 9.50)
                         (h2 : number_of_socks = 2)
                         (h3 : shoe_cost = 92)
                         (h4 : available_money = 40) :
  jack_needs_more_money sock_cost number_of_socks shoe_cost available_money :=
by
  unfold jack_needs_more_money
  simp [h1, h2, h3, h4]
  norm_num
  done

end jack_money_proof_l294_294931


namespace total_amount_is_correct_l294_294388

-- Definitions based on the conditions
def share_a (x : ℕ) : ℕ := 2 * x
def share_b (x : ℕ) : ℕ := 4 * x
def share_c (x : ℕ) : ℕ := 5 * x
def share_d (x : ℕ) : ℕ := 4 * x

-- Condition: combined share of a and b is 1800
def combined_share_of_ab (x : ℕ) : Prop := share_a x + share_b x = 1800

-- Theorem we want to prove: Total amount given to all children is $4500
theorem total_amount_is_correct (x : ℕ) (h : combined_share_of_ab x) : 
  share_a x + share_b x + share_c x + share_d x = 4500 := sorry

end total_amount_is_correct_l294_294388


namespace gemstones_needed_l294_294474

-- Define the initial quantities and relationships
def magnets_per_earring := 2
def buttons_per_magnet := 1 / 2
def gemstones_per_button := 3
def earrings_per_set := 2
def sets_of_earrings := 4

-- Define the total gemstones needed
theorem gemstones_needed : 
    let earrings := sets_of_earrings * earrings_per_set in
    let total_magnets := earrings * magnets_per_earring in
    let total_buttons := total_magnets * buttons_per_magnet in
    let total_gemstones := total_buttons * gemstones_per_button in
    total_gemstones = 24 :=
by
    have earrings := 2 * 4
    have total_magnets := earrings * 2
    have total_buttons := total_magnets / 2
    have total_gemstones := total_buttons * 3
    exact eq.refl 24

end gemstones_needed_l294_294474


namespace estimated_value_of_n_l294_294300

-- Definitions from the conditions of the problem
def total_balls (n : ℕ) : ℕ := n + 18 + 9
def probability_of_yellow (n : ℕ) : ℚ := 18 / total_balls n

-- The theorem stating what we need to prove
theorem estimated_value_of_n : ∃ n : ℕ, probability_of_yellow n = 0.30 ∧ n = 42 :=
by {
  sorry
}

end estimated_value_of_n_l294_294300


namespace area_of_hexagon_correct_l294_294153

variable (α β γ : ℝ) (S : ℝ) (r R : ℝ)
variable (AB BC AC : ℝ)
variable (A' B' C' : ℝ)

noncomputable def area_of_hexagon (AB BC AC : ℝ) (R : ℝ) (S : ℝ) (r : ℝ) : ℝ :=
  2 * (S / (r * r))

theorem area_of_hexagon_correct
  (hAB : AB = 13) (hBC : BC = 14) (hAC : AC = 15)
  (hR : R = 65 / 8) (hS : S = 1344 / 65) :
  area_of_hexagon AB BC AC R S r = 2 * (S / (r * r)) :=
sorry

end area_of_hexagon_correct_l294_294153


namespace quadratic_inequality_solution_l294_294127

variable (a x : ℝ)

theorem quadratic_inequality_solution (h : 0 < a ∧ a < 1) : (x - a) * (x - (1 / a)) > 0 ↔ (x < a ∨ x > 1 / a) :=
sorry

end quadratic_inequality_solution_l294_294127


namespace triangle_proof_l294_294325

noncomputable def triangle_math_proof (A B C : ℝ) (AA1 BB1 CC1 : ℝ) : Prop :=
  AA1 = 2 * Real.sin (B + A / 2) ∧
  BB1 = 2 * Real.sin (C + B / 2) ∧
  CC1 = 2 * Real.sin (A + C / 2) ∧
  (Real.sin A + Real.sin B + Real.sin C) ≠ 0 ∧
  ∀ x, x = (AA1 * Real.cos (A / 2) + BB1 * Real.cos (B / 2) + CC1 * Real.cos (C / 2)) / (Real.sin A + Real.sin B + Real.sin C) → x = 2

theorem triangle_proof (A B C AA1 BB1 CC1 : ℝ) (h : triangle_math_proof A B C AA1 BB1 CC1) :
  (AA1 * Real.cos (A / 2) + BB1 * Real.cos (B / 2) + CC1 * Real.cos (C / 2)) /
  (Real.sin A + Real.sin B + Real.sin C) = 2 := by
  sorry

end triangle_proof_l294_294325


namespace complement_of_A_in_U_l294_294423

def U : Set ℤ := {-2, -1, 1, 3, 5}
def A : Set ℤ := {-1, 3}
def CU_A : Set ℤ := {x ∈ U | x ∉ A}

theorem complement_of_A_in_U :
  CU_A = {-2, 1, 5} :=
by
  -- Proof goes here
  sorry

end complement_of_A_in_U_l294_294423


namespace fraction_nonneg_if_x_ge_m8_l294_294693

noncomputable def denominator (x : ℝ) : ℝ := x^2 + 4*x + 13
noncomputable def numerator (x : ℝ) : ℝ := x + 8

theorem fraction_nonneg_if_x_ge_m8 (x : ℝ) (hx : x ≥ -8) : numerator x / denominator x ≥ 0 :=
by sorry

end fraction_nonneg_if_x_ge_m8_l294_294693


namespace mary_turnips_grown_l294_294168

variable (sally_turnips : ℕ)
variable (total_turnips : ℕ)
variable (mary_turnips : ℕ)

theorem mary_turnips_grown (h_sally : sally_turnips = 113)
                          (h_total : total_turnips = 242) :
                          mary_turnips = total_turnips - sally_turnips := by
  sorry

end mary_turnips_grown_l294_294168


namespace distinct_solutions_square_difference_l294_294785

theorem distinct_solutions_square_difference 
  (Φ φ : ℝ) (h1 : Φ^2 = Φ + 2) (h2 : φ^2 = φ + 2) (h_distinct : Φ ≠ φ) :
  (Φ - φ)^2 = 9 :=
  sorry

end distinct_solutions_square_difference_l294_294785


namespace candy_bar_split_l294_294356
noncomputable def split (total: ℝ) (people: ℝ): ℝ := total / people

theorem candy_bar_split: split 5.0 3.0 = 1.67 :=
by
  sorry

end candy_bar_split_l294_294356


namespace phi_value_l294_294256

noncomputable def f (x φ : ℝ) := Real.sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h1 : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|) (h2 : f (π / 3) φ > f (π / 2) φ) : φ = π / 6 :=
by
  sorry

end phi_value_l294_294256


namespace total_people_hired_l294_294876

theorem total_people_hired (H L : ℕ) (hL : L = 1) (payroll : ℕ) (hPayroll : 129 * H + 82 * L = 3952) : H + L = 31 := by
  sorry

end total_people_hired_l294_294876


namespace functional_eq_solution_l294_294318

theorem functional_eq_solution (f : ℤ → ℤ) (h : ∀ x y : ℤ, x ≠ 0 →
  x * f (2 * f y - x) + y^2 * f (2 * x - f y) = (f x ^ 2) / x + f (y * f y)) :
  (∀ x: ℤ, f x = 0) ∨ (∀ x : ℤ, f x = x^2) :=
sorry

end functional_eq_solution_l294_294318


namespace meaningful_fraction_l294_294658

theorem meaningful_fraction {a : ℝ} : 2 * a - 1 ≠ 0 ↔ a ≠ 1 / 2 :=
by sorry

end meaningful_fraction_l294_294658


namespace sum_of_solutions_l294_294097

theorem sum_of_solutions (x : ℝ) : 
  (x^2 - 5*x - 26 = 4*x + 21) → 
  (∃ S, S = 9 ∧ ∀ x1 x2, x1 + x2 = S) := by
  intros h
  sorry

end sum_of_solutions_l294_294097


namespace initial_garrison_men_l294_294574

theorem initial_garrison_men (M : ℕ) (H1 : ∃ provisions : ℕ, provisions = M * 60)
  (H2 : ∃ provisions_15 : ℕ, provisions_15 = M * 45)
  (H3 : ∀ provisions_15 (new_provisions: ℕ), (provisions_15 = M * 45 ∧ new_provisions = 20 * (M + 1250)) → provisions_15 = new_provisions) :
  M = 1000 :=
by
  sorry

end initial_garrison_men_l294_294574


namespace equation_solutions_l294_294914

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃ x : ℝ, ax + b = 0) ∨ (∃ x : ℝ, ∀ y : ℝ, ax + b = 0 → x = y) :=
sorry

end equation_solutions_l294_294914


namespace molecular_weight_of_1_mole_l294_294363

theorem molecular_weight_of_1_mole (W_5 : ℝ) (W_1 : ℝ) (h : 5 * W_1 = W_5) (hW5 : W_5 = 490) : W_1 = 490 :=
by
  sorry

end molecular_weight_of_1_mole_l294_294363


namespace common_tangents_l294_294825

theorem common_tangents (r1 r2 d : ℝ) (h_r1 : r1 = 10) (h_r2 : r2 = 4) : 
  ∀ (n : ℕ), (n = 1) → ¬ (∃ (d : ℝ), 
    (6 < d ∧ d < 14 ∧ n = 2) ∨ 
    (d = 14 ∧ n = 3) ∨ 
    (d < 6 ∧ n = 0) ∨ 
    (d > 14 ∧ n = 4)) :=
by
  intro n h
  sorry

end common_tangents_l294_294825


namespace distance_between_A_and_B_l294_294049

-- Let d be the unknown distance we need to find
variable (d : ℚ)

-- Condition when Jia reaches the midpoint of AB
def jia_midpoint : Prop := d / 2 + (d / 2 - 5) = d - 5

-- Condition when Yi reaches the midpoint of AB
def yi_midpoint : Prop := d - (d / 2 - 45 / 8) = 45 / 8

-- The theorem stating that under given conditions, the distance d is 90 km
theorem distance_between_A_and_B :
  jia_midpoint d ∧ yi_midpoint d → d = 90 := 
sorry -- Proof is omitted

end distance_between_A_and_B_l294_294049


namespace probability_sum_less_than_product_l294_294530

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l294_294530


namespace remaining_water_in_bathtub_l294_294139

theorem remaining_water_in_bathtub : 
  ∀ (dripping_rate : ℕ) (evaporation_rate : ℕ) (duration_hr : ℕ) (dumped_out_liters : ℕ), 
    dripping_rate = 40 →
    evaporation_rate = 200 →
    duration_hr = 9 →
    dumped_out_liters = 12 →
    let total_dripped_in_ml := dripping_rate * 60 * duration_hr in
    let total_evaporated_in_ml := evaporation_rate * duration_hr in
    let net_water_in_ml := total_dripped_in_ml - total_evaporated_in_ml in
    let dumped_out_in_ml := dumped_out_liters * 1000 in
    net_water_in_ml - dumped_out_in_ml = 7800 :=
by
  intros dripping_rate evaporation_rate duration_hr dumped_out_liters
  intros rate_eq evap_eq duration_eq dump_eq
  simp [rate_eq, evap_eq, duration_eq, dump_eq]
  let total_dripped_in_ml := 40 * 60 * 9
  let total_evaporated_in_ml := 200 * 9
  let net_water_in_ml := total_dripped_in_ml - total_evaporated_in_ml
  let dumped_out_in_ml := 12 * 1000
  simp [net_water_in_ml, dumped_out_in_ml]
  sorry

end remaining_water_in_bathtub_l294_294139


namespace bill_amount_each_person_shared_l294_294709

noncomputable def total_bill : ℝ := 139.00
noncomputable def tip_percentage : ℝ := 0.10
noncomputable def num_people : ℝ := 7.00

noncomputable def tip : ℝ := tip_percentage * total_bill
noncomputable def total_bill_with_tip : ℝ := total_bill + tip
noncomputable def amount_each_person_pays : ℝ := total_bill_with_tip / num_people

theorem bill_amount_each_person_shared :
  amount_each_person_pays = 21.84 := by
  -- proof goes here
  sorry

end bill_amount_each_person_shared_l294_294709


namespace average_marks_l294_294495

theorem average_marks (total_students : ℕ) (first_group : ℕ) (first_group_marks : ℕ)
                      (second_group : ℕ) (second_group_marks_diff : ℕ) (third_group_marks : ℕ)
                      (total_marks : ℕ) (class_average : ℕ) :
  total_students = 50 → 
  first_group = 10 → 
  first_group_marks = 90 → 
  second_group = 15 → 
  second_group_marks_diff = 10 → 
  third_group_marks = 60 →
  total_marks = (first_group * first_group_marks) + (second_group * (first_group_marks - second_group_marks_diff)) + ((total_students - (first_group + second_group)) * third_group_marks) →
  class_average = total_marks / total_students →
  class_average = 72 :=
by
  intros
  sorry

end average_marks_l294_294495


namespace distribute_balls_in_boxes_l294_294275

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l294_294275


namespace arith_seq_a4a6_equals_4_l294_294301

variable (a : ℕ → ℝ) (d : ℝ)
variable (h2 : a 2 = a 1 + d)
variable (h4 : a 4 = a 1 + 3 * d)
variable (h6 : a 6 = a 1 + 5 * d)
variable (h8 : a 8 = a 1 + 7 * d)
variable (h10 : a 10 = a 1 + 9 * d)
variable (condition : (a 2)^2 + 2 * a 2 * a 8 + a 6 * a 10 = 16)

theorem arith_seq_a4a6_equals_4 : a 4 * a 6 = 4 := by
  sorry

end arith_seq_a4a6_equals_4_l294_294301


namespace johns_number_l294_294309

theorem johns_number (n : ℕ) 
  (h1 : 125 ∣ n) 
  (h2 : 30 ∣ n) 
  (h3 : 800 ≤ n ∧ n ≤ 2000) : 
  n = 1500 :=
sorry

end johns_number_l294_294309


namespace total_bowling_balls_l294_294186

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l294_294186


namespace complex_number_simplification_l294_294958

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) : 
  (↑(1 : ℂ) - i) / (↑(1 : ℂ) + i) ^ 2017 = -i :=
sorry

end complex_number_simplification_l294_294958


namespace find_x_plus_y_l294_294128

theorem find_x_plus_y (x y : ℝ) (h1 : |x| + x + y = 16) (h2 : x + |y| - y = 18) : x + y = 6 := 
sorry

end find_x_plus_y_l294_294128


namespace correct_statement_l294_294730

theorem correct_statement :
  (Real.sqrt (9 / 16) = 3 / 4) :=
by
  sorry

end correct_statement_l294_294730


namespace velocity_at_t_10_time_to_reach_max_height_max_height_l294_294348

-- Define the height function H(t)
def H (t : ℝ) : ℝ := 200 * t - 4.9 * t^2

-- Define the velocity function v(t) as the derivative of H(t)
def v (t : ℝ) : ℝ := 200 - 9.8 * t

-- Theorem: The velocity of the body at t = 10 seconds
theorem velocity_at_t_10 : v 10 = 102 := by
  sorry

-- Theorem: The time to reach maximum height
theorem time_to_reach_max_height : (∃ t : ℝ, v t = 0 ∧ t = 200 / 9.8) := by
  sorry

-- Theorem: The maximum height the body will reach
theorem max_height : H (200 / 9.8) = 2040.425 := by
  sorry

end velocity_at_t_10_time_to_reach_max_height_max_height_l294_294348


namespace krystiana_monthly_earnings_l294_294145

-- Definitions based on the conditions
def first_floor_cost : ℕ := 15
def second_floor_cost : ℕ := 20
def third_floor_cost : ℕ := 2 * first_floor_cost
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms_occupied : ℕ := 2

-- Statement to prove Krystiana's total monthly earnings are $165
theorem krystiana_monthly_earnings : 
  first_floor_cost * first_floor_rooms + 
  second_floor_cost * second_floor_rooms + 
  third_floor_cost * third_floor_rooms_occupied = 165 :=
by admit

end krystiana_monthly_earnings_l294_294145


namespace tina_coins_after_five_hours_l294_294511

theorem tina_coins_after_five_hours :
  let coins_in_first_hour := 20
  let coins_in_second_hour := 30
  let coins_in_third_hour := 30
  let coins_in_fourth_hour := 40
  let coins_taken_out_in_fifth_hour := 20
  let total_coins_after_five_hours := coins_in_first_hour + coins_in_second_hour + coins_in_third_hour + coins_in_fourth_hour - coins_taken_out_in_fifth_hour
  total_coins_after_five_hours = 100 :=
by {
  sorry
}

end tina_coins_after_five_hours_l294_294511


namespace nicholas_bottle_caps_l294_294328

theorem nicholas_bottle_caps (N : ℕ) (h : N + 85 = 93) : N = 8 :=
by
  sorry

end nicholas_bottle_caps_l294_294328


namespace perfect_square_condition_l294_294315

theorem perfect_square_condition (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
    (gcd_xyz : Nat.gcd (Nat.gcd x y) z = 1)
    (hx_dvd : x ∣ y * z * (x + y + z))
    (hy_dvd : y ∣ x * z * (x + y + z))
    (hz_dvd : z ∣ x * y * (x + y + z))
    (sum_dvd : x + y + z ∣ x * y * z) :
  ∃ m : ℕ, m * m = x * y * z * (x + y + z) := sorry

end perfect_square_condition_l294_294315


namespace go_to_yolka_together_l294_294086

noncomputable def anya_will_not_wait : Prop := true
noncomputable def boris_wait_time : ℕ := 10 -- in minutes
noncomputable def vasya_wait_time : ℕ := 15 -- in minutes
noncomputable def meeting_time_window : ℕ := 60 -- total available time in minutes

noncomputable def probability_all_go_together : ℝ :=
  (1 / 3) * (3500 / 3600)

theorem go_to_yolka_together :
  anya_will_not_wait ∧
  boris_wait_time = 10 ∧
  vasya_wait_time = 15 ∧
  meeting_time_window = 60 →
  probability_all_go_together = 0.324 :=
by
  intros
  sorry

end go_to_yolka_together_l294_294086


namespace even_blue_faces_cubes_correct_l294_294577

/-- A rectangular wooden block is 6 inches long, 3 inches wide, and 2 inches high.
    The block is painted blue on all six sides and then cut into 1 inch cubes.
    This function determines the number of 1-inch cubes that have a total number
    of blue faces that is an even number (in this case, 2 blue faces). -/
def count_even_blue_faces_cubes : Nat :=
  let length := 6
  let width := 3
  let height := 2
  let total_cubes := length * width * height
  
  -- Calculate corner cubes
  let corners := 8

  -- Calculate edges but not corners cubes
  let edge_not_corners := 
    (4 * (length - 2)) + 
    (4 * (width - 2)) + 
    (4 * (height - 2))

  -- Calculate even number of blue faces cubes 
  let even_number_blue_faces := edge_not_corners

  even_number_blue_faces

theorem even_blue_faces_cubes_correct : count_even_blue_faces_cubes = 20 := by
  -- Place your proof here.
  sorry

end even_blue_faces_cubes_correct_l294_294577


namespace direction_vector_l1_l294_294796

theorem direction_vector_l1
  (m : ℝ)
  (l₁ : ∀ x y : ℝ, (m + 3) * x + 4 * y + 3 * m - 5 = 0)
  (l₂ : ∀ x y : ℝ, 2 * x + (m + 6) * y - 8 = 0)
  (h_perp : ((m + 3) * 2 = -4 * (m + 6)))
  : ∃ v : ℝ × ℝ, v = (-1, -1/2) :=
by
  sorry

end direction_vector_l1_l294_294796


namespace probability_sum_less_than_product_l294_294540

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l294_294540


namespace max_n_for_expected_samples_l294_294737

-- Conditions in Lean
def blueberry_distribution : ProbabilityMassFunction ℝ :=
  ProbabilityMassFunction.normal 15 3

def premium_fruit (Z : ℝ) : Prop :=
  Z > 18

def premium_prob : ℝ :=
  ProbabilityMassFunction.prob_of blueberry_distribution premium_fruit

-- Maximum n such that the expected number of samples is at most 3
theorem max_n_for_expected_samples :
  let E (n : ℕ) := 5 * (1 - (0.8 : ℝ)^n)
  ∃ n : ℕ, E n ≤ 3 ∧ ∀ m : ℕ, m > n → E m > 3 :=
sorry

end max_n_for_expected_samples_l294_294737


namespace exists_v_mod_eq_l294_294941

noncomputable def v (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n+1).log2, n / 2^(k + 1)

theorem exists_v_mod_eq (a m : ℕ) (ha : 0 < a) (hm : 0 < m) : 
  ∃ n > 1, v(n) % m = a % m :=
by 
  sorry

end exists_v_mod_eq_l294_294941


namespace find_savings_l294_294842

def income : ℕ := 15000
def expenditure (I : ℕ) : ℕ := 4 * I / 5
def savings (I E : ℕ) : ℕ := I - E

theorem find_savings : savings income (expenditure income) = 3000 := 
by
  sorry

end find_savings_l294_294842


namespace justin_current_age_l294_294582

theorem justin_current_age
  (angelina_older : ∀ (j a : ℕ), a = j + 4)
  (angelina_future_age : ∀ (a : ℕ), a + 5 = 40) :
  ∃ (justin_current_age : ℕ), justin_current_age = 31 := 
by
  sorry

end justin_current_age_l294_294582


namespace sqrt_11_bounds_l294_294920

theorem sqrt_11_bounds : ∃ a : ℤ, a < Real.sqrt 11 ∧ Real.sqrt 11 < a + 1 ∧ a = 3 := 
by
  sorry

end sqrt_11_bounds_l294_294920


namespace fabric_cost_equation_l294_294090

theorem fabric_cost_equation (x : ℝ) :
  (3 * x + 5 * (138 - x) = 540) :=
sorry

end fabric_cost_equation_l294_294090


namespace revenue_per_investment_l294_294824

theorem revenue_per_investment (Banks_investments : ℕ) (Elizabeth_investments : ℕ) (Elizabeth_revenue_per_investment : ℕ) (revenue_difference : ℕ) :
  Banks_investments = 8 →
  Elizabeth_investments = 5 →
  Elizabeth_revenue_per_investment = 900 →
  revenue_difference = 500 →
  ∃ (R : ℤ), R = (5 * 900 - 500) / 8 :=
by
  intros h1 h2 h3 h4
  let T_elizabeth := 5 * Elizabeth_revenue_per_investment
  let T_banks := T_elizabeth - revenue_difference
  let R := T_banks / 8
  use R
  sorry

end revenue_per_investment_l294_294824


namespace tan_theta_eq_1_over_3_l294_294219

noncomputable def unit_circle_point (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := Real.sin θ
  (x^2 + y^2 = 1) ∧ (θ = Real.arccos ((4*x + 3*y) / 5))

theorem tan_theta_eq_1_over_3 (θ : ℝ) (h : unit_circle_point θ) : Real.tan θ = 1 / 3 := 
by
  sorry

end tan_theta_eq_1_over_3_l294_294219


namespace contrapositive_false_1_negation_false_1_l294_294731

theorem contrapositive_false_1 (m : ℝ) : ¬ (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

theorem negation_false_1 (m : ℝ) : ¬ ((m > 0) → ¬ (∃ x : ℝ, x^2 + x - m = 0)) :=
sorry

end contrapositive_false_1_negation_false_1_l294_294731


namespace find_a1_plus_a2_l294_294108

theorem find_a1_plus_a2 (x : ℝ) (a0 a1 a2 a3 : ℝ) 
  (h : (1 - 2/x)^3 = a0 + a1 * (1/x) + a2 * (1/x)^2 + a3 * (1/x)^3) : 
  a1 + a2 = 6 :=
by
  sorry

end find_a1_plus_a2_l294_294108


namespace earnings_difference_l294_294979

-- Define the prices and quantities.
def price_A : ℝ := 4
def price_B : ℝ := 3.5
def quantity_A : ℕ := 300
def quantity_B : ℕ := 350

-- Define the earnings for both companies.
def earnings_A := price_A * quantity_A
def earnings_B := price_B * quantity_B

-- State the theorem we intend to prove.
theorem earnings_difference :
  earnings_B - earnings_A = 25 := by
  sorry

end earnings_difference_l294_294979


namespace total_profit_is_64000_l294_294065

-- Definitions for investments and periods
variables (IB IA TB TA Profit_B Profit_A Total_Profit : ℕ)

-- Conditions from the problem
def condition1 := IA = 5 * IB
def condition2 := TA = 3 * TB
def condition3 := Profit_B = 4000
def condition4 := Profit_A / Profit_B = (IA * TA) / (IB * TB)

-- Target statement to be proved
theorem total_profit_is_64000 (IB IA TB TA Profit_B Profit_A Total_Profit : ℕ) :
  condition1 IA IB → condition2 TA TB → condition3 Profit_B → condition4 IA TA IB TB Profit_A Profit_B → 
  Total_Profit = Profit_A + Profit_B → Total_Profit = 64000 :=
by {
  sorry
}

end total_profit_is_64000_l294_294065


namespace laura_owes_amount_l294_294060

noncomputable def calculate_amount_owed (P R T : ℝ) : ℝ :=
  let I := P * R * T 
  P + I

theorem laura_owes_amount (P : ℝ) (R : ℝ) (T : ℝ) (hP : P = 35) (hR : R = 0.09) (hT : T = 1) :
  calculate_amount_owed P R T = 38.15 := by
  -- Prove that the total amount owed calculated by the formula matches the correct answer
  sorry

end laura_owes_amount_l294_294060


namespace sum_arithmetic_sequence_l294_294605

variables {a1 d : ℝ} {n : ℕ}

def arithmetic_seq (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

def Sn (a1 d : ℝ) (n : ℕ) : ℝ := n/2 * (2 * a1 + (n - 1) * d)

def circle (x : ℝ) (y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

def intersects (a1 : ℝ) : set (ℝ × ℝ) := {p : ℝ × ℝ | line a1 p.1 = p.2 ∧ circle p.1 p.2}

def symmetric (p1 p2 : ℝ × ℝ) (d : ℝ) : Prop := p1.1 + p1.2 + d = 0 ∧ p2.1 + p2.2 + d = 0

theorem sum_arithmetic_sequence (h : ∃ p1 p2 : ℝ × ℝ, p1 ∈ intersects a1 ∧ p2 ∈ intersects a1 ∧ symmetric p1 p2 d) :
  Sn a1 d n = -n^2 + 2 * n :=
sorry

end sum_arithmetic_sequence_l294_294605


namespace distance_to_angle_bisector_l294_294296

theorem distance_to_angle_bisector 
  (P : ℝ × ℝ) 
  (h_hyperbola : P.1^2 - P.2^2 = 9) 
  (h_distance_to_line_neg_x : abs (P.1 + P.2) = 2016 * Real.sqrt 2) : 
  abs (P.1 - P.2) / Real.sqrt 2 = 448 :=
sorry

end distance_to_angle_bisector_l294_294296


namespace number_of_women_l294_294087

theorem number_of_women (n_men n_women n_dances men_partners women_partners : ℕ) 
  (h_men_partners : men_partners = 4)
  (h_women_partners : women_partners = 3)
  (h_n_men : n_men = 15)
  (h_total_dances : n_dances = n_men * men_partners)
  (h_women_calc : n_women = n_dances / women_partners) :
  n_women = 20 :=
sorry

end number_of_women_l294_294087


namespace probability_sum_less_than_product_l294_294541

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l294_294541


namespace coefficient_x_squared_in_binomial_expansion_l294_294304

theorem coefficient_x_squared_in_binomial_expansion :
  let C := Nat.choose in
  ((x : ℚ) - (1 / (4 * x)))^6 = ∑ r in Finset.range 7, (-(1 / 4))^r * (C 6 r) * x^(6 - 2 * r) →
  ∑ r in Finset.range 3, (-(1 / 4))^r * (C 6 r) * x^(6 - 2 * r) = 15 / 16 * x^2 ∧
  ∀ r ∈ Finset.range 6, x^(6-2*r) = 2 → r = 2 ∧
  (-(1 / 4))^2 * (C 6 2) * x^2 = 1 / 16 * 15 * x^2 : by sorry

end coefficient_x_squared_in_binomial_expansion_l294_294304


namespace arithmetic_expression_value_l294_294226

def mixed_to_frac (a b c : ℕ) : ℚ := a + b / c

theorem arithmetic_expression_value :
  ( ( (mixed_to_frac 5 4 45 - mixed_to_frac 4 1 6) / mixed_to_frac 5 8 15 ) / 
    ( (mixed_to_frac 4 2 3 + 3 / 4) * mixed_to_frac 3 9 13 ) * mixed_to_frac 34 2 7 + 
    (3 / 10 / (1 / 100) / 70) + 2 / 7 ) = 1 :=
by
  -- We need to convert the mixed numbers to fractions using mixed_to_frac
  -- Then, we simplify step-by-step as in the problem solution, but for now we just use sorry
  sorry

end arithmetic_expression_value_l294_294226


namespace tap_b_fill_time_l294_294048

theorem tap_b_fill_time (t : ℝ) (h1 : t > 0) : 
  (∀ (A_fill B_fill together_fill : ℝ), 
    A_fill = 1/45 ∧ 
    B_fill = 1/t ∧ 
    together_fill = A_fill + B_fill ∧ 
    (9 * A_fill) + (23 * B_fill) = 1) → 
    t = 115 / 4 :=
by
  sorry

end tap_b_fill_time_l294_294048


namespace composite_infinitely_many_l294_294689

theorem composite_infinitely_many (t : ℕ) (ht : t ≥ 2) :
  ∃ n : ℕ, n = 3 ^ (2 ^ t) - 2 ^ (2 ^ t) ∧ (3 ^ (n - 1) - 2 ^ (n - 1)) % n = 0 :=
by
  use 3 ^ (2 ^ t) - 2 ^ (2 ^ t)
  sorry 

end composite_infinitely_many_l294_294689


namespace domain_of_f_symmetry_of_f_l294_294476

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 * x^2 - x^4)) / (abs (x - 2) - 2)

theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

theorem symmetry_of_f :
  ∀ x : ℝ, f (x + 1) + 1 = f (-(x + 1)) + 1 :=
by
  sorry

end domain_of_f_symmetry_of_f_l294_294476


namespace jack_needs_more_money_l294_294928

variable (cost_per_pair_of_socks : ℝ := 9.50)
variable (number_of_pairs_of_socks : ℕ := 2)
variable (cost_of_soccer_shoes : ℝ := 92.00)
variable (jack_money : ℝ := 40.00)

theorem jack_needs_more_money :
  let total_cost := number_of_pairs_of_socks * cost_per_pair_of_socks + cost_of_soccer_shoes
  let money_needed := total_cost - jack_money
  money_needed = 71.00 := by
  sorry

end jack_needs_more_money_l294_294928


namespace find_bk_l294_294337

theorem find_bk
  (A B C D : ℝ)
  (BC : ℝ) (hBC : BC = 3)
  (AB CD : ℝ) (hAB_CD : AB = 2 * CD)
  (BK : ℝ) (hBK : BK = 2) :
  ∃ x a : ℝ, (x = BK) ∧ (AB = 2 * CD) ∧ ((2 * a + x) * (3 - x) = x * (a + 3 - x)) :=
by
  sorry

end find_bk_l294_294337


namespace probability_larry_wins_l294_294311

noncomputable def P_larry_wins_game : ℝ :=
  let p_hit := (1 : ℝ) / 3
  let p_miss := (2 : ℝ) / 3
  let r := p_miss^3
  (p_hit / (1 - r))

theorem probability_larry_wins :
  P_larry_wins_game = 9 / 19 :=
by
  -- Proof is omitted, but the outline and logic are given in the problem statement
  sorry

end probability_larry_wins_l294_294311


namespace calculate_savings_l294_294019

theorem calculate_savings :
  let plane_cost : ℕ := 600
  let boat_cost : ℕ := 254
  plane_cost - boat_cost = 346 := by
    let plane_cost : ℕ := 600
    let boat_cost : ℕ := 254
    sorry

end calculate_savings_l294_294019


namespace incorrect_lifetime_calculation_l294_294483

-- Define expectation function
noncomputable def expectation (X : ℝ) : ℝ := sorry

-- We define the lifespans
variables (xi eta : ℝ)
-- Expected lifespan of the sensor and transmitter
axiom exp_xi : expectation xi = 3
axiom exp_eta : expectation eta = 5

-- Define the lifetime of the device
noncomputable def T := min xi eta

-- Given conditions
theorem incorrect_lifetime_calculation :
  expectation T ≤ 3 → 3 + (2 / 3) > 3 → false := 
sorry

end incorrect_lifetime_calculation_l294_294483


namespace lucas_fraction_to_emma_l294_294684

variable (n : ℕ)

-- Define initial stickers
def noah_stickers := n
def emma_stickers := 3 * n
def lucas_stickers := 12 * n

-- Define the final state where each has the same number of stickers
def final_stickers_per_person := (16 * n) / 3

-- Lucas gives some stickers to Emma. Calculate the fraction of Lucas's stickers given to Emma
theorem lucas_fraction_to_emma :
  (7 * n / 3) / (12 * n) = 7 / 36 := by
  sorry

end lucas_fraction_to_emma_l294_294684


namespace solution_set_f_leq_g_range_of_a_l294_294160

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := x + 2

theorem solution_set_f_leq_g (x : ℝ) : f x 1 ≤ g x ↔ (0 ≤ x ∧ x ≤ 2 / 3) := by
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ g x) : 2 ≤ a := by
  sorry

end solution_set_f_leq_g_range_of_a_l294_294160


namespace find_number_of_elements_l294_294696

theorem find_number_of_elements (n S : ℕ) (h1 : S + 26 = 19 * n) (h2 : S + 76 = 24 * n) : n = 10 := 
sorry

end find_number_of_elements_l294_294696


namespace arrow_hits_apple_l294_294326

noncomputable def time_to_hit (L V0 : ℝ) (α β : ℝ) : ℝ :=
  (L / V0) * (Real.sin β / Real.sin (α + β))

theorem arrow_hits_apple (g : ℝ) (L V0 : ℝ) (α β : ℝ) (h : (L / V0) * (Real.sin β / Real.sin (α + β)) = 3 / 4) 
  : time_to_hit L V0 α β = 3 / 4 := 
  by
  sorry

end arrow_hits_apple_l294_294326


namespace differential_equation_solution_l294_294833

def C1 : ℝ := sorry
def C2 : ℝ := sorry

noncomputable def y (x : ℝ) : ℝ := C1 * Real.cos x + C2 * Real.sin x
noncomputable def z (x : ℝ) : ℝ := -C1 * Real.sin x + C2 * Real.cos x

theorem differential_equation_solution : 
  (∀ x : ℝ, deriv y x = z x) ∧ 
  (∀ x : ℝ, deriv z x = -y x) :=
by
  sorry

end differential_equation_solution_l294_294833


namespace average_weight_of_class_l294_294735

theorem average_weight_of_class (students_a students_b : ℕ) (avg_weight_a avg_weight_b : ℝ)
  (h_students_a : students_a = 24)
  (h_students_b : students_b = 16)
  (h_avg_weight_a : avg_weight_a = 40)
  (h_avg_weight_b : avg_weight_b = 35) :
  ((students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b)) = 38 := 
by
  sorry

end average_weight_of_class_l294_294735


namespace intersection_x_sum_l294_294262

theorem intersection_x_sum :
  ∃ x : ℤ, (0 ≤ x ∧ x < 17) ∧ (4 * x + 3 ≡ 13 * x + 14 [ZMOD 17]) ∧ x = 5 :=
by
  sorry

end intersection_x_sum_l294_294262


namespace remainder_relation_l294_294860

theorem remainder_relation (P P' D R R' : ℕ) (hP : P > P') (h1 : P % D = R) (h2 : P' % D = R') :
  ∃ C : ℕ, ((P + C) * P') % D ≠ (P * P') % D ∧ ∃ C : ℕ, ((P + C) * P') % D = (P * P') % D :=
by sorry

end remainder_relation_l294_294860


namespace unknown_number_value_l294_294645

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end unknown_number_value_l294_294645


namespace goals_even_more_likely_l294_294870

theorem goals_even_more_likely (p_1 : ℝ) (q_1 : ℝ) (h1 : p_1 + q_1 = 1) :
  let p := p_1^2 + q_1^2 
  let q := 2 * p_1 * q_1
  p ≥ q := by
    sorry

end goals_even_more_likely_l294_294870


namespace geometric_sequence_sum_l294_294321

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h1 : a 1 + a 2 + a 3 = 7) 
  (h2 : a 2 + a 3 + a 4 = 14) 
  (geom_seq : ∃ q, ∀ n, a (n + 1) = q * a n ∧ q = 2) :
  a 4 + a 5 + a 6 = 56 := 
by
  sorry

end geometric_sequence_sum_l294_294321


namespace distance_traveled_l294_294290

theorem distance_traveled
  (D : ℝ) (T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 14 * T)
  : D = 50 := sorry

end distance_traveled_l294_294290


namespace sasha_skated_distance_l294_294375

theorem sasha_skated_distance (d total_distance v : ℝ)
  (h1 : total_distance = 3300)
  (h2 : v > 0)
  (h3 : d = 3 * v * (total_distance / (3 * v + 2 * v))) :
  d = 1100 :=
by
  sorry

end sasha_skated_distance_l294_294375


namespace simplify_scientific_notation_l294_294339

theorem simplify_scientific_notation :
  (12 * 10^10) / (6 * 10^2) = 2 * 10^8 := 
sorry

end simplify_scientific_notation_l294_294339


namespace problem_solution_l294_294417

noncomputable def vector_magnitudes_and_angle 
  (a b : ℝ) (angle_ab : ℝ) (norma normb : ℝ) (k : ℝ) : Prop :=
(a = 4 ∧ b = 8 ∧ angle_ab = 2 * Real.pi / 3 ∧ norma = 4 ∧ normb = 8) →
((norma^2 + normb^2 + 2 * norma * normb * Real.cos angle_ab = 48) ∧
  (16 * k - 32 * k + 16 - 128 = 0))

theorem problem_solution : vector_magnitudes_and_angle 4 8 (2 * Real.pi / 3) 4 8 (-7) := 
by 
  sorry

end problem_solution_l294_294417


namespace number_of_dogs_l294_294874

theorem number_of_dogs (total_animals cats : ℕ) (probability : ℚ) (h1 : total_animals = 7) (h2 : cats = 2) (h3 : probability = 2 / 7) :
  total_animals - cats = 5 := 
by
  sorry

end number_of_dogs_l294_294874


namespace shorter_side_length_l294_294740

variables (x y : ℝ)
variables (h1 : 2 * x + 2 * y = 60)
variables (h2 : x * y = 200)

theorem shorter_side_length :
  min x y = 10 :=
by
  sorry

end shorter_side_length_l294_294740


namespace max_earth_to_sun_distance_l294_294671

-- Define the semi-major axis a and semi-focal distance c
def semi_major_axis : ℝ := 1.5 * 10^8
def semi_focal_distance : ℝ := 3 * 10^6

-- Define the maximum distance from the Earth to the Sun
def max_distance (a c : ℝ) : ℝ := a + c

-- Define the Lean statement to be proved
theorem max_earth_to_sun_distance :
  max_distance semi_major_axis semi_focal_distance = 1.53 * 10^8 :=
by
  -- skipping the proof for now
  sorry

end max_earth_to_sun_distance_l294_294671


namespace shift_right_inverse_exp_eq_ln_l294_294830

variable (f : ℝ → ℝ)

theorem shift_right_inverse_exp_eq_ln :
  (∀ x, f (x - 1) = Real.log x) → ∀ x, f x = Real.log (x + 1) :=
by
  sorry

end shift_right_inverse_exp_eq_ln_l294_294830


namespace larger_number_is_391_l294_294373

-- Define the H.C.F and factors
def HCF := 23
def factor1 := 13
def factor2 := 17
def LCM := HCF * factor1 * factor2

-- Define the two numbers based on the factors
def number1 := HCF * factor1
def number2 := HCF * factor2

-- Theorem statement
theorem larger_number_is_391 : max number1 number2 = 391 := 
by
  sorry

end larger_number_is_391_l294_294373


namespace knights_minimum_count_l294_294866

/-- There are 1001 people sitting around a round table, each of whom is either a knight (always tells the truth) or a liar (always lies).
Next to each knight, there is exactly one liar, and next to each liar, there is exactly one knight.
Prove that the minimum number of knights that can be sitting at the table is 502. -/
theorem knights_minimum_count (n : ℕ) (h : n = 1001) (N : ℕ) (L : ℕ) 
  (h1 : N + L = n) (h2 : ∀ i, (i < n) → 
    ((is_knight i ∧ is_liar ((i + 1) % n)) ∨ (is_liar i ∧ is_knight ((i + 1) % n)))) 
  : N = 502 :=
sorry

end knights_minimum_count_l294_294866


namespace new_temperature_l294_294039

-- Define the initial temperature
variable (t : ℝ)

-- Define the temperature drop
def temperature_drop : ℝ := 2

-- State the theorem
theorem new_temperature (t : ℝ) (temperature_drop : ℝ) : t - temperature_drop = t - 2 :=
by
  sorry

end new_temperature_l294_294039


namespace least_positive_integer_with_six_factors_is_18_l294_294728

-- Define the least positive integer with exactly six distinct positive factors.
def least_positive_with_six_factors (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∣ n → d > 0) ∧ (finset.card (finset.filter (λ d, d ∣ n) (finset.range (n + 1)))) = 6

-- Prove that the least positive integer with exactly six distinct positive factors is 18.
theorem least_positive_integer_with_six_factors_is_18 : (∃ n : ℕ, least_positive_with_six_factors n ∧ n = 18) :=
sorry


end least_positive_integer_with_six_factors_is_18_l294_294728


namespace count_two_digit_integers_remainder_3_div_7_l294_294616

theorem count_two_digit_integers_remainder_3_div_7 :
  ∃ (n : ℕ) (hn : 1 ≤ n ∧ n < 14), (finset.range 14).filter (λ n, 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100).card = 13 :=
sorry

end count_two_digit_integers_remainder_3_div_7_l294_294616


namespace tangent_slope_at_one_l294_294354

def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem tangent_slope_at_one : f' 1 = 2 := by
  sorry

end tangent_slope_at_one_l294_294354


namespace number_of_valid_consecutive_sum_sets_l294_294240

-- Definition of what it means to be a set of consecutive integers summing to 225
def sum_of_consecutive_integers (n a : ℕ) : Prop :=
  ∃ k : ℕ, (k = (n * (2 * a + n - 1)) / 2) ∧ (k = 225)

-- Prove that there are exactly 4 sets of two or more consecutive positive integers that sum to 225
theorem number_of_valid_consecutive_sum_sets : 
  ∃ (sets : Finset (ℕ × ℕ)), 
    (∀ (n a : ℕ), (n, a) ∈ sets ↔ sum_of_consecutive_integers n a) ∧ 
    (2 ≤ n) ∧ 
    sets.card = 4 := sorry

end number_of_valid_consecutive_sum_sets_l294_294240


namespace math_problem_l294_294888

theorem math_problem (m n : ℕ) (hm : m > 0) (hn : n > 0):
  ((2^(2^n) + 1) * (2^(2^m) + 1)) % (m * n) = 0 →
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1) :=
by
  sorry

end math_problem_l294_294888


namespace days_to_use_up_one_bag_l294_294225

def rice_kg : ℕ := 11410
def bags : ℕ := 3260
def rice_per_day : ℚ := 0.25
def rice_per_bag : ℚ := rice_kg / bags

theorem days_to_use_up_one_bag : (rice_per_bag / rice_per_day) = 14 := by
  sorry

end days_to_use_up_one_bag_l294_294225


namespace max_trains_ratio_l294_294164

theorem max_trains_ratio (years : ℕ) 
    (birthday_trains : ℕ) 
    (christmas_trains : ℕ) 
    (total_trains : ℕ)
    (parents_multiple : ℕ) 
    (h_years : years = 5)
    (h_birthday_trains : birthday_trains = 1)
    (h_christmas_trains : christmas_trains = 2)
    (h_total_trains : total_trains = 45)
    (h_parents_multiple : parents_multiple = 2) :
  let trains_received_in_years := years * (birthday_trains + 2 * christmas_trains)
  let trains_given_by_parents := total_trains - trains_received_in_years
  let trains_before_gift := total_trains - trains_given_by_parents
  trains_given_by_parents / trains_before_gift = parents_multiple := by
  sorry

end max_trains_ratio_l294_294164


namespace polynomial_sum_l294_294953

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l294_294953


namespace contrapositive_example_l294_294838

theorem contrapositive_example (x : ℝ) : 
  (x = 1 ∨ x = 2) → (x^2 - 3 * x + 2 ≤ 0) :=
by
  sorry

end contrapositive_example_l294_294838


namespace students_without_scholarships_l294_294023

theorem students_without_scholarships :
  let total_students := 300
  let full_merit_percent := 0.05
  let half_merit_percent := 0.10
  let sports_percent := 0.03
  let need_based_percent := 0.07
  let full_merit_and_sports_percent := 0.01
  let half_merit_and_need_based_percent := 0.02
  let full_merit := full_merit_percent * total_students
  let half_merit := half_merit_percent * total_students
  let sports := sports_percent * total_students
  let need_based := need_based_percent * total_students
  let full_merit_and_sports := full_merit_and_sports_percent * total_students
  let half_merit_and_need_based := half_merit_and_need_based_percent * total_students
  let total_with_scholarships := (full_merit + half_merit + sports + need_based) - (full_merit_and_sports + half_merit_and_need_based)
  let students_without_scholarships := total_students - total_with_scholarships
  students_without_scholarships = 234 := 
by
  sorry

end students_without_scholarships_l294_294023


namespace probability_sum_less_than_product_l294_294533

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l294_294533


namespace least_pos_int_with_six_factors_l294_294725

theorem least_pos_int_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, (number_of_factors m = 6 → m ≥ n)) ∧ n = 12 := 
sorry

end least_pos_int_with_six_factors_l294_294725


namespace factor_polynomial_l294_294904

open Polynomial

noncomputable def polynomial1 := (2 : ℚ) * X^4 + X^3 - 16 * X^2 + 3 * X + 16 + 3 - 1
noncomputable def polynomial2 := X^2 + X - 6

theorem factor_polynomial : polynomial2 ∣ polynomial1 :=
by
  -- Here we should find the quotient and check it exactly divides, but proof is omitted.
  sorry

end factor_polynomial_l294_294904


namespace volume_ratio_remainder_520_l294_294075

noncomputable def simplex_ratio_mod : Nat :=
  let m := 2 ^ 2015 - 2016
  let n := 2 ^ 2015
  (m + n) % 1000

theorem volume_ratio_remainder_520 :
  let m := 2 ^ 2015 - 2016
  let n := 2 ^ 2015
  (m + n) % 1000 = 520 :=
by 
  sorry

end volume_ratio_remainder_520_l294_294075


namespace intersection_of_sets_l294_294918

def A := { x : ℝ | 0 ≤ x ∧ x ≤ 2 }
def B := { x : ℝ | x^2 > 1 }
def C := { x : ℝ | 1 < x ∧ x ≤ 2 }

theorem intersection_of_sets : 
  (A ∩ B) = C := 
by sorry

end intersection_of_sets_l294_294918


namespace find_distance_l294_294720

-- Conditions: total cost, base price, cost per mile
variables (total_cost base_price cost_per_mile : ℕ)

-- Definition of the distance as per the problem
def distance_from_home_to_hospital (total_cost base_price cost_per_mile : ℕ) : ℕ :=
  (total_cost - base_price) / cost_per_mile

-- Given values:
def total_cost_value : ℕ := 23
def base_price_value : ℕ := 3
def cost_per_mile_value : ℕ := 4

-- The theorem that encapsulates the problem statement
theorem find_distance :
  distance_from_home_to_hospital total_cost_value base_price_value cost_per_mile_value = 5 :=
by
  -- Placeholder for the proof
  sorry

end find_distance_l294_294720


namespace measure_of_angle_C_l294_294856

-- Definitions of the angles
def angles (A B C : ℝ) : Prop :=
  -- Conditions: measure of angle A is 1/4 of measure of angle B
  A = (1 / 4) * B ∧
  -- Lines p and q are parallel so alternate interior angles are equal
  C = A ∧
  -- Since angles B and C are supplementary
  B + C = 180

-- The problem in Lean 4 statement: Prove that C = 36 given the conditions
theorem measure_of_angle_C (A B C : ℝ) (h : angles A B C) : C = 36 := sorry

end measure_of_angle_C_l294_294856


namespace least_positive_integer_with_six_distinct_factors_l294_294727

theorem least_positive_integer_with_six_distinct_factors : ∃ n : ℕ, (∀ k : ℕ, (number_of_factors k = 6) → (n ≤ k)) ∧ (number_of_factors n = 6) ∧ (n = 12) :=
by
  sorry

end least_positive_integer_with_six_distinct_factors_l294_294727


namespace greatest_integer_less_than_or_equal_to_frac_l294_294361

theorem greatest_integer_less_than_or_equal_to_frac (a b c d : ℝ)
  (ha : a = 4^100) (hb : b = 3^100) (hc : c = 4^95) (hd : d = 3^95) :
  ⌊(a + b) / (c + d)⌋ = 1023 := 
by
  sorry

end greatest_integer_less_than_or_equal_to_frac_l294_294361


namespace total_number_of_students_l294_294457

theorem total_number_of_students (b h p s : ℕ) 
  (h1 : b = 30)
  (h2 : b = 2 * h)
  (h3 : p = h + 5)
  (h4 : s = 3 * p) :
  b + h + p + s = 125 :=
by sorry

end total_number_of_students_l294_294457


namespace hyperbola_eccentricity_l294_294613

open Real

/-- Given the hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 2 - x^2 / 8 = 1

/-- Prove the eccentricity of the given hyperbola -/
theorem hyperbola_eccentricity (x y : ℝ) (h : hyperbola_equation x y) : 
  ∃ e : ℝ, e = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l294_294613


namespace problem_equiv_l294_294637

theorem problem_equiv (a b : ℝ) (h : a ≠ b) : 
  (a^2 - 4 * a + 5 > 0) ∧ (a^2 + b^2 ≥ 2 * (a - b - 1)) :=
by {
  sorry
}

end problem_equiv_l294_294637


namespace x_gt_y_necessary_not_sufficient_for_x_gt_abs_y_l294_294380

variable {x : ℝ}
variable {y : ℝ}

theorem x_gt_y_necessary_not_sufficient_for_x_gt_abs_y
  (hx : x > 0) :
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|) := by
  sorry

end x_gt_y_necessary_not_sufficient_for_x_gt_abs_y_l294_294380


namespace amount_invested_l294_294657

variables (P y : ℝ)

-- Conditions
def condition1 : Prop := 800 = P * (2 * y) / 100
def condition2 : Prop := 820 = P * ((1 + y / 100) ^ 2 - 1)

-- The proof we seek
theorem amount_invested (h1 : condition1 P y) (h2 : condition2 P y) : P = 8000 :=
by
  -- Place the proof here
  sorry

end amount_invested_l294_294657


namespace percentage_of_percentage_l294_294062

theorem percentage_of_percentage (a b : ℝ) (h_a : a = 0.03) (h_b : b = 0.05) : (a / b) * 100 = 60 :=
by
  sorry

end percentage_of_percentage_l294_294062


namespace find_y_l294_294072

theorem find_y (x : ℝ) (h1 : x = 1.3333333333333333) (h2 : (x * y) / 3 = x^2) : y = 4 :=
by 
  sorry

end find_y_l294_294072


namespace baker_cakes_total_l294_294587

def initial_cakes : ℕ := 110
def cakes_sold : ℕ := 75
def additional_cakes : ℕ := 76

theorem baker_cakes_total : 
  (initial_cakes - cakes_sold) + additional_cakes = 111 := by
  sorry

end baker_cakes_total_l294_294587


namespace min_value_of_squares_l294_294320

theorem min_value_of_squares (a b t : ℝ) (h : a + b = t) : (a^2 + b^2) ≥ t^2 / 2 := 
by
  sorry

end min_value_of_squares_l294_294320


namespace geometric_progression_fourth_term_l294_294761

theorem geometric_progression_fourth_term :
  ∀ (a₁ a₂ a₃ a₄ : ℝ), a₁ = 2^(1/2) ∧ a₂ = 2^(1/4) ∧ a₃ = 2^(1/6) ∧ (a₂ / a₁ = r) ∧ (a₃ = a₂ * r⁻¹) ∧ (a₄ = a₃ * r) → a₄ = 2^(1/8) := by
intro a₁ a₂ a₃ a₄
intro h
sorry

end geometric_progression_fourth_term_l294_294761


namespace probability_sum_less_than_product_l294_294538

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l294_294538


namespace james_paid_amount_l294_294008

def total_stickers (packs : ℕ) (stickers_per_pack : ℕ) : ℕ :=
  packs * stickers_per_pack

def total_cost (num_stickers : ℕ) (cost_per_sticker : ℕ) : ℕ :=
  num_stickers * cost_per_sticker

def half_cost (total_cost : ℕ) : ℕ :=
  total_cost / 2

theorem james_paid_amount :
  let packs : ℕ := 4,
      stickers_per_pack : ℕ := 30,
      cost_per_sticker : ℕ := 10,  -- Using cents for simplicity to avoid decimals
      friend_share : ℕ := 2,
      num_stickers := total_stickers packs stickers_per_pack,
      total_amt := total_cost num_stickers cost_per_sticker,
      james_amt := half_cost total_amt
  in
  james_amt = 600 :=
by
  sorry

end james_paid_amount_l294_294008


namespace k_h_5_eq_148_l294_294638

def h (x : ℤ) : ℤ := 4 * x + 6
def k (x : ℤ) : ℤ := 6 * x - 8

theorem k_h_5_eq_148 : k (h 5) = 148 := by
  sorry

end k_h_5_eq_148_l294_294638


namespace bisecting_line_of_circle_l294_294034

theorem bisecting_line_of_circle : 
  (∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y + 1 = 0 → x - y + 1 = 0) := 
sorry

end bisecting_line_of_circle_l294_294034


namespace average_marks_l294_294496

theorem average_marks (num_students : ℕ) (marks1 marks2 marks3 : ℕ) (num1 num2 num3 : ℕ) (h1 : num_students = 50)
  (h2 : marks1 = 90) (h3 : num1 = 10) (h4 : marks2 = marks1 - 10) (h5 : num2 = 15) (h6 : marks3 = 60) 
  (h7 : num1 + num2 + num3 = 50) (h8 : num3 = num_students - (num1 + num2)) (total_marks : ℕ) 
  (h9 : total_marks = (num1 * marks1) + (num2 * marks2) + (num3 * marks3)) : 
  (total_marks / num_students = 72) :=
by
  sorry

end average_marks_l294_294496


namespace average_speed_is_70_l294_294503

theorem average_speed_is_70 
  (distance1 distance2 : ℕ) (time1 time2 : ℕ)
  (h1 : distance1 = 80) (h2 : distance2 = 60)
  (h3 : time1 = 1) (h4 : time2 = 1) :
  (distance1 + distance2) / (time1 + time2) = 70 := 
by 
  sorry

end average_speed_is_70_l294_294503


namespace g_eq_g_inv_l294_294233

-- Define the function g
def g (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

-- Define the inverse function of g
noncomputable def g_inv (y : ℝ) : ℝ := (5 + Real.sqrt (1 + 8 * y)) / 4 -- simplified to handle the principal value

theorem g_eq_g_inv (x : ℝ) : g x = g_inv x → x = 1 := by
  -- Placeholder for proof
  sorry

end g_eq_g_inv_l294_294233


namespace sum_of_n_l294_294557

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end sum_of_n_l294_294557


namespace find_function_expression_point_on_function_graph_l294_294005

-- Problem setup
def y_minus_2_is_directly_proportional_to_x (y x : ℝ) : Prop :=
  ∃ k : ℝ, y - 2 = k * x

-- Conditions
def specific_condition : Prop :=
  y_minus_2_is_directly_proportional_to_x 6 1

-- Function expression derivation
theorem find_function_expression : ∃ k, ∀ x, 6 - 2 = k * 1 ∧ ∀ y, y = k * x + 2 :=
sorry

-- Given point P belongs to the function graph
theorem point_on_function_graph (a : ℝ) : (∀ x y, y = 4 * x + 2) → ∃ a, 4 * a + 2 = -1 :=
sorry

end find_function_expression_point_on_function_graph_l294_294005


namespace even_integers_between_sqrt_10_and_sqrt_100_l294_294798

theorem even_integers_between_sqrt_10_and_sqrt_100 : 
  ∃ (n : ℕ), n = 4 ∧ (∀ (a : ℕ), (∃ k, (2 * k = a ∧ a > Real.sqrt 10 ∧ a < Real.sqrt 100)) ↔ 
  (a = 4 ∨ a = 6 ∨ a = 8 ∨ a = 10)) := 
by 
  sorry

end even_integers_between_sqrt_10_and_sqrt_100_l294_294798


namespace timothy_tea_cups_l294_294194

theorem timothy_tea_cups (t : ℕ) (h : 6 * t + 60 = 120) : t + 12 = 22 :=
by
  sorry

end timothy_tea_cups_l294_294194


namespace sum_of_n_values_such_that_fraction_is_integer_l294_294553

theorem sum_of_n_values_such_that_fraction_is_integer : 
  let is_odd (d : ℤ) : Prop := d % 2 ≠ 0
  let divisors (n : ℤ) := ∃ d : ℤ, d ∣ n
  let a_values := { n : ℤ | ∃ (d : ℤ), divisors 36 ∧ is_odd d ∧ 2 * n - 1 = d }
  let a_sum := ∑ n in a_values, n
  a_sum = 8 := 
by
  sorry

end sum_of_n_values_such_that_fraction_is_integer_l294_294553


namespace coupon1_best_discount_l294_294386

noncomputable def listed_prices : List ℝ := [159.95, 179.95, 199.95, 219.95, 239.95]

theorem coupon1_best_discount (x : ℝ) (h₁ : x ∈ listed_prices) (h₂ : x > 120) :
  0.15 * x > 25 ∧ 0.15 * x > 0.20 * (x - 120) ↔ 
  x = 179.95 ∨ x = 199.95 ∨ x = 219.95 ∨ x = 239.95 :=
sorry

end coupon1_best_discount_l294_294386


namespace quadratic_eqns_mod_7_l294_294832

/-- Proving the solutions for quadratic equations in arithmetic modulo 7. -/
theorem quadratic_eqns_mod_7 :
  (¬ ∃ x : ℤ, (5 * x^2 + 3 * x + 1) % 7 = 0) ∧
  (∃! x : ℤ, (x^2 + 3 * x + 4) % 7 = 0 ∧ x % 7 = 2) ∧
  (∃ x1 x2 : ℤ, (x1 ^ 2 - 2 * x1 - 3) % 7 = 0 ∧ (x2 ^ 2 - 2 * x2 - 3) % 7 = 0 ∧ 
              x1 % 7 = 3 ∧ x2 % 7 = 6) :=
by
  sorry

end quadratic_eqns_mod_7_l294_294832


namespace solve_for_x_l294_294231

def custom_mul (a b : ℤ) : ℤ := a * b + a + b

theorem solve_for_x (x : ℤ) :
  custom_mul 3 (3 * x - 1) = 27 → x = 7 / 3 := by
sorry

end solve_for_x_l294_294231


namespace fraction_calculation_correct_l294_294400

noncomputable def calculate_fraction : ℚ :=
  let numerator := (1 / 2) - (1 / 3)
  let denominator := (3 / 4) + (1 / 8)
  numerator / denominator

theorem fraction_calculation_correct : calculate_fraction = 4 / 21 := 
  by
    sorry

end fraction_calculation_correct_l294_294400


namespace monthly_earnings_is_correct_l294_294149

-- Conditions as definitions

def first_floor_cost_per_room : ℕ := 15
def second_floor_cost_per_room : ℕ := 20
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms : ℕ := 3
def occupied_third_floor_rooms : ℕ := 2

-- Calculated values from conditions
def third_floor_cost_per_room : ℕ := 2 * first_floor_cost_per_room

-- Total earnings on each floor
def first_floor_earnings : ℕ := first_floor_cost_per_room * first_floor_rooms
def second_floor_earnings : ℕ := second_floor_cost_per_room * second_floor_rooms
def third_floor_earnings : ℕ := third_floor_cost_per_room * occupied_third_floor_rooms

-- Total monthly earnings
def total_monthly_earnings : ℕ :=
  first_floor_earnings + second_floor_earnings + third_floor_earnings

theorem monthly_earnings_is_correct : total_monthly_earnings = 165 := by
  -- proof omitted
  sorry

end monthly_earnings_is_correct_l294_294149


namespace linear_inequality_solution_l294_294255

theorem linear_inequality_solution (a b : ℝ)
  (h₁ : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ (x < -3 ∨ x > 1)) :
  ∀ x : ℝ, a * x + b < 0 ↔ x < 3 / 2 :=
by
  sorry

end linear_inequality_solution_l294_294255


namespace pencils_initial_count_l294_294101

theorem pencils_initial_count (pencils_initially: ℕ) :
  (∀ n, n > 0 → n < 36 → 36 % n = 1) →
  pencils_initially + 30 = 36 → 
  pencils_initially = 6 :=
by
  intro h hn
  sorry

end pencils_initial_count_l294_294101


namespace range_of_a_l294_294260

noncomputable def proposition_p (a : ℝ) : Prop :=
∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

noncomputable def proposition_q (a : ℝ) : Prop :=
∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0

theorem range_of_a (a : ℝ) (h : proposition_p a ∧ proposition_q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l294_294260


namespace maximize_binomial_probability_l294_294665

open Nat

theorem maximize_binomial_probability :
  ∀ k : ℕ, (k ≤ 5) ∧
  (P_formula k = (choose 5 k) * (1/4 : ℚ)^k * (3/4 : ℚ)^(5 - k)) →
  P_formula 1 = max (P_formula 0)
    (max (P_formula 1)
      (max (P_formula 2)
        (max (P_formula 3)
          (max (P_formula 4) (P_formula 5))))) :=
  by -- structure and conditions of the proof go here
  sorry

def P_formula (k : ℕ) : ℚ :=
  if k ≤ 5 then (choose 5 k) * (1/4 : ℚ)^k * (3/4 : ℚ)^(5 - k)
  else 0

end maximize_binomial_probability_l294_294665


namespace ball_box_problem_l294_294270

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l294_294270


namespace jose_bottle_caps_proof_l294_294452

def jose_bottle_caps_initial : Nat := 7
def rebecca_bottle_caps : Nat := 2
def jose_bottle_caps_final : Nat := 9

theorem jose_bottle_caps_proof : jose_bottle_caps_initial + rebecca_bottle_caps = jose_bottle_caps_final := by
  sorry

end jose_bottle_caps_proof_l294_294452


namespace reflection_slope_intercept_l294_294575

noncomputable def reflect_line_slope_intercept (k : ℝ) (hk1 : k ≠ 0) (hk2 : k ≠ -1) : ℝ × ℝ :=
  let slope := (1 : ℝ) / k
  let intercept := (k - 1) / k
  (slope, intercept)

theorem reflection_slope_intercept {k : ℝ} (hk1 : k ≠ 0) (hk2 : k ≠ -1) :
  reflect_line_slope_intercept k hk1 hk2 = (1/k, (k-1)/k) := by
  sorry

end reflection_slope_intercept_l294_294575


namespace expression_value_l294_294561

theorem expression_value : (1 * 3 * 5 * 7) / (1^2 + 2^2 + 3^2 + 4^2) = 7 / 2 := by
  sorry

end expression_value_l294_294561


namespace num_regular_soda_l294_294871

theorem num_regular_soda (t d r : ℕ) (h₁ : t = 17) (h₂ : d = 8) (h₃ : r = t - d) : r = 9 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end num_regular_soda_l294_294871


namespace Marie_finish_time_l294_294685

def Time := Nat × Nat -- Represents time as (hours, minutes)

def start_time : Time := (9, 0)
def finish_two_tasks_time : Time := (11, 20)
def total_tasks : Nat := 4

def minutes_since_start (t : Time) : Nat :=
  let (h, m) := t
  (h - 9) * 60 + m

def calculate_finish_time (start: Time) (two_tasks_finish: Time) (total_tasks: Nat) : Time :=
  let duration_two_tasks := minutes_since_start two_tasks_finish
  let duration_each_task := duration_two_tasks / 2
  let total_time := duration_each_task * total_tasks
  let total_minutes_after_start := total_time + minutes_since_start start
  let finish_hour := 9 + total_minutes_after_start / 60
  let finish_minute := total_minutes_after_start % 60
  (finish_hour, finish_minute)

theorem Marie_finish_time :
  calculate_finish_time start_time finish_two_tasks_time total_tasks = (13, 40) :=
by
  sorry

end Marie_finish_time_l294_294685


namespace find_number_l294_294801

theorem find_number (y : ℝ) (h : 0.25 * 820 = 0.15 * y - 20) : y = 1500 :=
by
  sorry

end find_number_l294_294801


namespace book_club_meeting_days_l294_294009

theorem book_club_meeting_days :
  Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := 
by sorry

end book_club_meeting_days_l294_294009


namespace geometric_series_sum_correct_l294_294758

-- Given conditions
def a : ℤ := 3
def r : ℤ := -2
def n : ℤ := 10

-- Sum of the geometric series formula
def geometric_series_sum (a r n : ℤ) : ℤ := 
  a * (r^n - 1) / (r - 1)

-- Goal: Prove that the sum of the series is -1023
theorem geometric_series_sum_correct : 
  geometric_series_sum a r n = -1023 := 
by
  sorry

end geometric_series_sum_correct_l294_294758


namespace inequality_sum_l294_294817

theorem inequality_sum (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
  (a / (a ^ 3 + b * c) + b / (b ^ 3 + c * a) + c / (c ^ 3 + a * b)) > 3 :=
by
  sorry

end inequality_sum_l294_294817


namespace triangle_congruent_reciprocal_circumcenter_orthocenter_l294_294819

-- Definitions and Conditions
variables {V : Type _} [InnerProductSpace ℝ V] [EuclideanSpace V] (A B C M A1 B1 C1 : V)

-- Existing conditions
-- M is the orthocenter of triangle ABC
def isOrthocenter_of_triangle : Prop :=
  isOrthocenter ℝ (triangle.mk A B C) M

-- Centers of the circumcircles of triangles BCM, CAM, and ABM
def circumcenter_BCM : Prop :=
  A1 = circumcenter ℝ ∠B C M
def circumcenter_CAM : Prop :=
  B1 = circumcenter ℝ ∠C A M
def circumcenter_ABM : Prop :=
  C1 = circumcenter ℝ ∠A B M

-- Proof Statements
theorem triangle_congruent :
  isOrthocenter_of_triangle ℝ A B C M →
  circumcenter_BCM ℝ A B C M A1 →
  circumcenter_CAM ℝ A B C M B1 →
  circumcenter_ABM ℝ A B C M C1 →
  triangle.congruent ℝ (triangle.mk A B C) (triangle.mk A1 B1 C1)
:= by sorry

theorem reciprocal_circumcenter_orthocenter :
  isOrthocenter_of_triangle ℝ A B C M →
  circumcenter_BCM ℝ A B C M A1 →
  circumcenter_CAM ℝ A B C M B1 →
  circumcenter_ABM ℝ A B C M C1 →
  isCircumcenter ℝ (triangle.mk A1 B1 C1) M ∧
  isOrthocenter ℝ (triangle.mk A1 B1 C1) (circumcenter ℝ ∠A B C)
:= by sorry

end triangle_congruent_reciprocal_circumcenter_orthocenter_l294_294819


namespace circles_common_point_l294_294191

theorem circles_common_point {n : ℕ} (hn : n ≥ 5) (circles : Fin n → Set Point)
  (hcommon : ∀ (a b c : Fin n), (circles a ∩ circles b ∩ circles c).Nonempty) :
  ∃ p : Point, ∀ i : Fin n, p ∈ circles i :=
sorry

end circles_common_point_l294_294191


namespace largest_m_divides_30_fact_l294_294889

theorem largest_m_divides_30_fact : 
  let pow2_in_fact := 15 + 7 + 3 + 1,
      pow3_in_fact := 10 + 3 + 1,
      max_m_from_2 := pow2_in_fact,
      max_m_from_3 := pow3_in_fact / 2
  in max_m_from_2 >= 7 ∧ max_m_from_3 >= 7 → 7 = 7 :=
by
  sorry

end largest_m_divides_30_fact_l294_294889


namespace xinxin_nights_at_seaside_l294_294053

-- Definitions from conditions
def arrival_day : ℕ := 30
def may_days : ℕ := 31
def departure_day : ℕ := 4
def nights_spent : ℕ := (departure_day + (may_days - arrival_day))

-- Theorem to prove the number of nights spent
theorem xinxin_nights_at_seaside : nights_spent = 5 := 
by
  -- Include proof steps here in actual Lean proof
  sorry

end xinxin_nights_at_seaside_l294_294053


namespace sum_angles_acute_l294_294137

open Real

theorem sum_angles_acute (A B C : ℝ) (hA_ac : A < π / 2) (hB_ac : B < π / 2) (hC_ac : C < π / 2)
  (h_angle_sum : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 1) :
  π / 2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end sum_angles_acute_l294_294137


namespace water_left_in_bathtub_l294_294140

theorem water_left_in_bathtub :
  (40 * 60 * 9 - 200 * 9 - 12000 = 7800) :=
by
  -- Dripping rate per minute * number of minutes in an hour * number of hours
  let inflow_rate := 40 * 60
  let total_inflow := inflow_rate * 9
  -- Evaporation rate per hour * number of hours
  let total_evaporation := 200 * 9
  -- Water dumped out
  let water_dumped := 12000
  -- Final amount of water
  let final_amount := total_inflow - total_evaporation - water_dumped
  have h : final_amount = 7800 := by
    sorry
  exact h

end water_left_in_bathtub_l294_294140


namespace sum_of_valid_n_l294_294556

theorem sum_of_valid_n :
  (∑ n in {n : ℤ | (∃ d ∈ ({1, 3, 9} : Finset ℤ), 2 * n - 1 = d)}, n) = 8 := by
sorry

end sum_of_valid_n_l294_294556


namespace solve_inequalities_l294_294234

theorem solve_inequalities (x : ℝ) :
  (3 * x^2 - x > 4) ∧ (x < 3) ↔ (1 < x ∧ x < 3) := 
by 
  sorry

end solve_inequalities_l294_294234


namespace shauna_min_test_score_l294_294441

theorem shauna_min_test_score (score1 score2 score3 : ℕ) (h1 : score1 = 82) (h2 : score2 = 88) (h3 : score3 = 95) 
  (max_score : ℕ) (h4 : max_score = 100) (desired_avg : ℕ) (h5 : desired_avg = 85) :
  ∃ (score4 score5 : ℕ), score4 ≥ 75 ∧ score5 ≥ 75 ∧ (score1 + score2 + score3 + score4 + score5) / 5 = desired_avg :=
by
  -- proof here
  sorry

end shauna_min_test_score_l294_294441


namespace number_of_cars_in_second_box_is_31_l294_294675

-- Define the total number of toy cars, and the number of toy cars in the first and third boxes
def total_toy_cars : ℕ := 71
def cars_in_first_box : ℕ := 21
def cars_in_third_box : ℕ := 19

-- Define the number of toy cars in the second box
def cars_in_second_box : ℕ := total_toy_cars - cars_in_first_box - cars_in_third_box

-- Theorem stating that the number of toy cars in the second box is 31
theorem number_of_cars_in_second_box_is_31 : cars_in_second_box = 31 :=
by
  sorry

end number_of_cars_in_second_box_is_31_l294_294675


namespace cost_of_flowers_cost_function_minimum_cost_l294_294863

-- Define the costs in terms of yuan
variables (n m : ℕ) -- n is the cost of one lily, m is the cost of one carnation.

-- Define the conditions
axiom cost_condition1 : 2 * n + m = 14
axiom cost_condition2 : 3 * m = 2 * n + 2

-- Prove the cost of one carnation and one lily
theorem cost_of_flowers : n = 5 ∧ m = 4 :=
by {
  sorry
}

-- Variables for the second part
variables (w x : ℕ) -- w is the total cost, x is the number of carnations.

-- Define the conditions
axiom total_condition : 11 = 2 + x + (11 - x)
axiom min_lilies_condition : 11 - x ≥ 2

-- State the relationship between w and x
theorem cost_function : w = 55 - x :=
by {
  sorry
}

-- Prove the minimum cost
theorem minimum_cost : ∃ x, (x ≤ 9 ∧  w = 46) :=
by {
  sorry
}

end cost_of_flowers_cost_function_minimum_cost_l294_294863


namespace largest_multiple_of_8_less_than_100_l294_294993

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l294_294993


namespace total_pupils_correct_l294_294448

-- Definitions of the number of girls and boys in each school
def girlsA := 542
def boysA := 387
def girlsB := 713
def boysB := 489
def girlsC := 628
def boysC := 361

-- Total pupils in each school
def pupilsA := girlsA + boysA
def pupilsB := girlsB + boysB
def pupilsC := girlsC + boysC

-- Total pupils across all schools
def total_pupils := pupilsA + pupilsB + pupilsC

-- The proof statement (no proof provided, hence sorry)
theorem total_pupils_correct : total_pupils = 3120 := by sorry

end total_pupils_correct_l294_294448


namespace cos_alpha_minus_pi_over_2_l294_294811

theorem cos_alpha_minus_pi_over_2 (α : ℝ) 
  (h1 : ∃ k : ℤ, α = k * (2 * Real.pi) ∨ α = k * (2 * Real.pi) + Real.pi / 2 ∨ α = k * (2 * Real.pi) + Real.pi ∨ α = k * (2 * Real.pi) + 3 * Real.pi / 2)
  (h2 : Real.cos α = 4 / 5)
  (h3 : Real.sin α = -3 / 5) : 
  Real.cos (α - Real.pi / 2) = -3 / 5 := 
by 
  sorry

end cos_alpha_minus_pi_over_2_l294_294811


namespace probability_sum_less_than_product_l294_294543

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l294_294543


namespace value_computation_l294_294642

theorem value_computation (N : ℝ) (h1 : 1.20 * N = 2400) : 0.20 * N = 400 := 
by
  sorry

end value_computation_l294_294642


namespace count_two_digit_remainders_l294_294618

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l294_294618


namespace f_is_increasing_exists_ratio_two_range_k_for_negative_two_ratio_l294_294782

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (k x : ℝ) : ℝ := x^2 + k * x
noncomputable def a (x1 x2 : ℝ) : ℝ := (f x1 - f x2) / (x1 - x2)
noncomputable def b (z1 z2 k : ℝ) : ℝ := (g k z1 - g k z2) / (z1 - z2)

theorem f_is_increasing (x1 x2 : ℝ) (h : x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0) : a x1 x2 > 0 := by
  sorry

theorem exists_ratio_two (k : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a x1 x2 ≠ 0 ∧ b x1 x2 k = 2 * a x1 x2 := by
  sorry

theorem range_k_for_negative_two_ratio (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a x1 x2 ≠ 0 ∧ b x1 x2 k = -2 * a x1 x2) → k < -4 := by
  sorry

end f_is_increasing_exists_ratio_two_range_k_for_negative_two_ratio_l294_294782


namespace minimum_S_l294_294119

theorem minimum_S (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  S = (a + 1/a)^2 + (b + 1/b)^2 → S ≥ 8 :=
by
  sorry

end minimum_S_l294_294119


namespace bug_traverses_36_tiles_l294_294230

-- Define the dimensions of the rectangle and the bug's problem setup
def width : ℕ := 12
def length : ℕ := 25

-- Define the function to calculate the number of tiles traversed by the bug
def tiles_traversed (w l : ℕ) : ℕ :=
  w + l - Nat.gcd w l

-- Prove the number of tiles traversed by the bug is 36
theorem bug_traverses_36_tiles : tiles_traversed width length = 36 :=
by
  -- This part will be proven; currently, we add sorry
  sorry

end bug_traverses_36_tiles_l294_294230


namespace least_positive_integer_with_six_factors_l294_294723

theorem least_positive_integer_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → m < n → (count_factors m ≠ 6)) ∧ count_factors n = 6 ∧ n = 18 :=
sorry

noncomputable def count_factors (n : ℕ) : ℕ :=
sorry

end least_positive_integer_with_six_factors_l294_294723


namespace largest_trifecta_sum_l294_294382

def trifecta (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a ∣ b ∧ b ∣ c ∧ c ∣ (a * b) ∧ (100 ≤ a) ∧ (a < 1000) ∧ (100 ≤ b) ∧ (b < 1000) ∧ (100 ≤ c) ∧ (c < 1000)

theorem largest_trifecta_sum : ∃ (a b c : ℕ), trifecta a b c ∧ a + b + c = 700 :=
sorry

end largest_trifecta_sum_l294_294382


namespace smallest_positive_n_l294_294479

theorem smallest_positive_n
  (a x y : ℤ)
  (h1 : x ≡ a [ZMOD 9])
  (h2 : y ≡ -a [ZMOD 9]) :
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n) % 9 = 0 ∧ n = 6 :=
by
  sorry

end smallest_positive_n_l294_294479


namespace tan_alpha_equiv_l294_294415

theorem tan_alpha_equiv (α : ℝ) (h : Real.tan α = 2) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 := 
by 
  sorry

end tan_alpha_equiv_l294_294415


namespace plane_parallel_l294_294439

-- Definitions for planes and lines within a plane
variable (Plane : Type) (Line : Type)
variables (lines_in_plane1 : Set Line)
variables (parallel_to_plane2 : Line → Prop)
variables (Plane1 Plane2 : Plane)

-- Conditions
axiom infinite_lines_in_plane1_parallel_to_plane2 : ∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l
axiom planes_are_parallel : ∀ (P1 P2 : Plane), (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l) → P1 = Plane1 → P2 = Plane2 → (Plane1 ≠ Plane2 ∧ (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l))

-- The proof that Plane 1 and Plane 2 are parallel based on the conditions
theorem plane_parallel : Plane1 ≠ Plane2 → ∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l → (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l) := 
by
  sorry

end plane_parallel_l294_294439


namespace infinite_divisible_269_l294_294323

theorem infinite_divisible_269 (a : ℕ → ℤ) (h₀ : a 0 = 2) (h₁ : a 1 = 15) 
  (h_recur : ∀ n : ℕ, a (n + 2) = 15 * a (n + 1) + 16 * a n) :
  ∃ infinitely_many k: ℕ, 269 ∣ a k :=
by
  sorry

end infinite_divisible_269_l294_294323


namespace pauls_weekly_spending_l294_294827

def mowing_lawns : ℕ := 3
def weed_eating : ℕ := 3
def total_weeks : ℕ := 2
def total_money : ℕ := mowing_lawns + weed_eating
def spending_per_week : ℕ := total_money / total_weeks

theorem pauls_weekly_spending : spending_per_week = 3 := by
  sorry

end pauls_weekly_spending_l294_294827


namespace find_tangent_line_l294_294969

def is_perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

def is_tangent_to_circle (a b c : ℝ) : Prop :=
  let d := abs c / (Real.sqrt (a^2 + b^2))
  d = 1

def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem find_tangent_line :
  ∀ (k b : ℝ),
    is_perpendicular k 1 →
    is_tangent_to_circle 1 1 b →
    ∃ (x y : ℝ), in_first_quadrant x y ∧ x + y - b = 0 →
    b = Real.sqrt 2 := sorry

end find_tangent_line_l294_294969


namespace arithmetic_problem_l294_294228

theorem arithmetic_problem : 72 * 1313 - 32 * 1313 = 52520 := by
  sorry

end arithmetic_problem_l294_294228


namespace number_of_adult_tickets_l294_294188

-- Let's define our conditions and the theorem to prove.
theorem number_of_adult_tickets (A C : ℕ) (h₁ : A + C = 522) (h₂ : 15 * A + 8 * C = 5086) : A = 131 :=
by
  sorry

end number_of_adult_tickets_l294_294188


namespace total_bowling_balls_l294_294179

theorem total_bowling_balls (red_balls : ℕ) (green_balls : ℕ) (h1 : red_balls = 30) (h2 : green_balls = red_balls + 6) : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l294_294179


namespace rex_cards_remaining_l294_294686

theorem rex_cards_remaining
  (nicole_cards : ℕ)
  (cindy_cards : ℕ)
  (rex_cards : ℕ)
  (cards_per_person : ℕ)
  (h1 : nicole_cards = 400)
  (h2 : cindy_cards = 2 * nicole_cards)
  (h3 : rex_cards = (nicole_cards + cindy_cards) / 2)
  (h4 : cards_per_person = rex_cards / 4) :
  cards_per_person = 150 :=
by
  sorry

end rex_cards_remaining_l294_294686


namespace usual_time_to_school_l294_294546

theorem usual_time_to_school (R T : ℝ) (h : (R * T = (6/5) * R * (T - 4))) : T = 24 :=
by 
  sorry

end usual_time_to_school_l294_294546


namespace two_digit_integers_leaving_remainder_3_div_7_count_l294_294624

noncomputable def count_two_digit_integers_leaving_remainder_3_div_7 : ℕ :=
  (finset.Ico 1 14).card

theorem two_digit_integers_leaving_remainder_3_div_7_count :
  count_two_digit_integers_leaving_remainder_3_div_7 = 13 :=
  sorry

end two_digit_integers_leaving_remainder_3_div_7_count_l294_294624


namespace correct_operations_result_l294_294024

theorem correct_operations_result {n : ℕ} (h₁ : n / 8 - 20 = 12) :
  (n * 8 + 20) = 2068 ∧ 1800 < 2068 ∧ 2068 < 2200 :=
by
  sorry

end correct_operations_result_l294_294024


namespace largest_multiple_of_8_less_than_100_l294_294989

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l294_294989


namespace range_of_m_l294_294597

theorem range_of_m (m x : ℝ) : 
  (2 / (x - 3) + (x + m) / (3 - x) = 2) 
  ∧ (x ≥ 0) →
  (m ≤ 8 ∧ m ≠ -1) :=
by 
  sorry

end range_of_m_l294_294597


namespace total_ticket_sales_is_48_l294_294853

noncomputable def ticket_sales (total_revenue : ℕ) (price_per_ticket : ℕ) (discount_1 : ℕ) (discount_2 : ℕ) : ℕ :=
  let number_first_batch := 10
  let number_second_batch := 20
  let revenue_first_batch := number_first_batch * (price_per_ticket - (price_per_ticket * discount_1 / 100))
  let revenue_second_batch := number_second_batch * (price_per_ticket - (price_per_ticket * discount_2 / 100))
  let revenue_full_price := total_revenue - (revenue_first_batch + revenue_second_batch)
  let number_full_price_tickets := revenue_full_price / price_per_ticket
  number_first_batch + number_second_batch + number_full_price_tickets

theorem total_ticket_sales_is_48 : ticket_sales 820 20 40 15 = 48 :=
by
  sorry

end total_ticket_sales_is_48_l294_294853


namespace segment_length_of_points_A_l294_294389

-- Define the basic setup
variable (d BA CA : ℝ)
variable {A B C : Point} -- Assume we have a type Point for the geometric points

-- Establish some conditions: A right triangle with given lengths
def is_right_triangle (A B C : Point) : Prop := sorry -- Placeholder for definition

def distance (P Q : Point) : ℝ := sorry -- Placeholder for the distance function

-- Conditions
variables (h_right_triangle : is_right_triangle A B C)
variables (h_hypotenuse : distance B C = d)
variables (h_smallest_leg : min (distance B A) (distance C A) = min BA CA)

-- The theorem statement
theorem segment_length_of_points_A (h_right_triangle : is_right_triangle A B C)
                                    (h_hypotenuse : distance B C = d)
                                    (h_smallest_leg : min (distance B A) (distance C A) = min BA CA) :
  ∃ A, (∀ t : ℝ, distance O A = d - min BA CA) :=
sorry -- Proof to be provided

end segment_length_of_points_A_l294_294389


namespace plane_equation_passing_through_point_and_parallel_l294_294102

-- Define the point and the plane parameters
def point : ℝ × ℝ × ℝ := (2, 3, 1)
def normal_vector : ℝ × ℝ × ℝ := (2, -1, 3)
def plane (A B C D : ℝ) (x y z : ℝ) : Prop := A * x + B * y + C * z + D = 0

-- Main theorem statement
theorem plane_equation_passing_through_point_and_parallel :
  ∃ D : ℝ, plane 2 (-1) 3 D 2 3 1 ∧ plane 2 (-1) 3 D 0 0 0 :=
sorry

end plane_equation_passing_through_point_and_parallel_l294_294102


namespace trig_identity_sin_cos_l294_294896

theorem trig_identity_sin_cos
  (a : ℝ)
  (h : Real.sin (Real.pi / 3 - a) = 1 / 3) :
  Real.cos (5 * Real.pi / 6 - a) = -1 / 3 :=
by
  sorry

end trig_identity_sin_cos_l294_294896


namespace evaluate_expression_l294_294848

theorem evaluate_expression : 
  (-2 : ℤ)^2004 + 3 * (-2)^2003 = (-2)^2003 :=
by
  sorry

end evaluate_expression_l294_294848


namespace last_nonzero_digit_aperiodic_l294_294822

/-- Definition of the last nonzero digit of n! --/
def last_nonzero_digit (n : ℕ) : ℕ := -- implementation, e.g., taking n! modulo 10 after removing trailing zeros
  sorry

/-- The main theorem stating that the sequence of last nonzero digits of n! is aperiodic --/
theorem last_nonzero_digit_aperiodic :
  ¬ ∃ (T n₀ : ℕ), ∀ (n : ℕ), n ≥ n₀ → last_nonzero_digit (n + T) = last_nonzero_digit n :=
sorry

end last_nonzero_digit_aperiodic_l294_294822


namespace blue_pill_cost_is_25_l294_294882

variable (blue_pill_cost red_pill_cost : ℕ)

-- Clara takes one blue pill and one red pill each day for 10 days.
-- A blue pill costs $2 more than a red pill.
def pill_cost_condition (blue_pill_cost red_pill_cost : ℕ) : Prop :=
  blue_pill_cost = red_pill_cost + 2 ∧
  10 * blue_pill_cost + 10 * red_pill_cost = 480

-- Prove that the cost of one blue pill is $25.
theorem blue_pill_cost_is_25 (h : pill_cost_condition blue_pill_cost red_pill_cost) : blue_pill_cost = 25 :=
  sorry

end blue_pill_cost_is_25_l294_294882


namespace star_comm_star_assoc_star_id_exists_star_not_dist_add_l294_294232

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Statement 1: Commutativity
theorem star_comm : ∀ x y : ℝ, star x y = star y x := 
by sorry

-- Statement 2: Associativity
theorem star_assoc : ∀ x y z : ℝ, star (star x y) z = star x (star y z) := 
by sorry

-- Statement 3: Identity Element
theorem star_id_exists : ∃ e : ℝ, ∀ x : ℝ, star x e = x := 
by sorry

-- Statement 4: Distributivity Over Addition
theorem star_not_dist_add : ∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z := 
by sorry

end star_comm_star_assoc_star_id_exists_star_not_dist_add_l294_294232


namespace possible_values_of_angle_F_l294_294002

-- Define angle F conditions in a triangle DEF
def triangle_angle_F_conditions (D E : ℝ) : Prop :=
  5 * Real.sin D + 2 * Real.cos E = 8 ∧ 3 * Real.sin E + 5 * Real.cos D = 2

-- The main statement: proving the possible values of ∠F
theorem possible_values_of_angle_F (D E : ℝ) (h : triangle_angle_F_conditions D E) : 
  ∃ F : ℝ, F = Real.arcsin (43 / 50) ∨ F = 180 - Real.arcsin (43 / 50) :=
by
  sorry

end possible_values_of_angle_F_l294_294002


namespace f_monotonicity_g_min_l294_294820

-- Definitions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * a ^ x - 2 * a ^ (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a ^ (2 * x) + a ^ (-2 * x) - 2 * f x a

-- Conditions
variable {a : ℝ} 
variable (a_pos : 0 < a) (a_ne_one : a ≠ 1) (f_one : f 1 a = 3) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3)

-- Monotonicity of f(x)
theorem f_monotonicity : 
  (∀ x y, x < y → f x a < f y a) ∨ (∀ x y, x < y → f y a < f x a) :=
sorry

-- Minimum value of g(x)
theorem g_min : ∃ x' : ℝ, 0 ≤ x' ∧ x' ≤ 3 ∧ g x' a = -2 :=
sorry

end f_monotonicity_g_min_l294_294820


namespace find_integer_n_l294_294499

theorem find_integer_n (a b : ℕ) (n : ℕ)
  (h1 : n = 2^a * 3^b)
  (h2 : (2^(a+1) - 1) * ((3^(b+1) - 1) / (3 - 1)) = 1815) : n = 648 :=
  sorry

end find_integer_n_l294_294499


namespace positive_two_digit_integers_remainder_3_l294_294627

theorem positive_two_digit_integers_remainder_3 :
  {x : ℕ | x % 7 = 3 ∧ 10 ≤ x ∧ x < 100}.finite.card = 13 := 
by
  sorry

end positive_two_digit_integers_remainder_3_l294_294627


namespace last_two_nonzero_digits_of_factorial_100_l294_294843

theorem last_two_nonzero_digits_of_factorial_100 : let n := last_two_nonzero_digits (Nat.factorial 100) in n = 76 :=
sorry

end last_two_nonzero_digits_of_factorial_100_l294_294843


namespace entire_function_constant_l294_294816

open Complex

theorem entire_function_constant
  (f : ℂ → ℂ) (h_entire : ∀ z, differentiable_at ℂ f z)
  (ω1 ω2 : ℂ) (h_irrational : ω1 / ω2 ∉ ℚ)
  (h_periodic : ∀ z, f z = f (z + ω1) ∧ f z = f (z + ω2)) :
  ∃ c : ℂ, ∀ z, f z = c :=
by
  sorry

end entire_function_constant_l294_294816


namespace total_savings_correct_l294_294015

-- Definitions of savings per day and days saved for Josiah, Leah, and Megan
def josiah_saving_per_day : ℝ := 0.25
def josiah_days : ℕ := 24

def leah_saving_per_day : ℝ := 0.50
def leah_days : ℕ := 20

def megan_saving_per_day : ℝ := 1.00
def megan_days : ℕ := 12

-- Definition to calculate total savings for each child
def total_saving (saving_per_day : ℝ) (days : ℕ) : ℝ :=
  saving_per_day * days

-- Total amount saved by Josiah, Leah, and Megan
def total_savings : ℝ :=
  total_saving josiah_saving_per_day josiah_days +
  total_saving leah_saving_per_day leah_days +
  total_saving megan_saving_per_day megan_days

-- Theorem to prove the total savings is $28
theorem total_savings_correct : total_savings = 28 := by
  sorry

end total_savings_correct_l294_294015


namespace min_value_expr_l294_294595

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) : (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 24 :=
sorry

end min_value_expr_l294_294595


namespace slope_angle_l294_294187

theorem slope_angle (A B : ℝ × ℝ) (θ : ℝ) (hA : A = (-1, 3)) (hB : B = (1, 1)) (hθ : θ ∈ Set.Ico 0 Real.pi)
  (hslope : Real.tan θ = (B.2 - A.2) / (B.1 - A.1)) :
  θ = (3 / 4) * Real.pi :=
by
  cases hA
  cases hB
  simp at hslope
  sorry

end slope_angle_l294_294187


namespace cone_volume_l294_294506

theorem cone_volume (r h : ℝ) (h_cylinder_vol : π * r^2 * h = 72 * π) : 
  (1 / 3) * π * r^2 * (h / 2) = 12 * π := by
  sorry

end cone_volume_l294_294506


namespace unique_a_values_l294_294104

theorem unique_a_values :
  ∃ a_values : Finset ℝ,
    (∀ a ∈ a_values, ∃ r s : ℤ, (r + s = -a) ∧ (r * s = 8 * a)) ∧ a_values.card = 4 :=
by
  sorry

end unique_a_values_l294_294104


namespace sector_area_l294_294966

noncomputable def area_of_sector (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) : ℝ :=
  1 / 2 * arc_length * radius

theorem sector_area (R : ℝ)
  (arc_length : ℝ) (central_angle : ℝ)
  (h_arc : arc_length = 4 * Real.pi)
  (h_angle : central_angle = Real.pi / 3)
  (h_radius : arc_length = central_angle * R) :
  area_of_sector arc_length central_angle 12 = 24 * Real.pi :=
by
  -- Proof skipped
  sorry

#check sector_area

end sector_area_l294_294966


namespace problem1_problem2_l294_294029

-- Proof of Problem 1
theorem problem1 (x y : ℤ) (h1 : x = -2) (h2 : y = -3) : (6 * x - 5 * y + 3 * y - 2 * x) = -2 :=
by
  sorry

-- Proof of Problem 2
theorem problem2 (a : ℚ) (h : a = -1 / 2) : (1 / 4 * (-4 * a^2 + 2 * a - 8) - (1 / 2 * a - 2)) = -1 / 4 :=
by
  sorry

end problem1_problem2_l294_294029


namespace sample_size_correct_l294_294780

-- Definitions following the conditions in the problem
def total_products : Nat := 80
def sample_products : Nat := 10

-- Statement of the proof problem
theorem sample_size_correct : sample_products = 10 :=
by
  -- The proof is replaced with a placeholder sorry to skip the proof step
  sorry

end sample_size_correct_l294_294780


namespace benzene_carbon_mass_percentage_l294_294412

noncomputable def carbon_mass_percentage_in_benzene 
  (carbon_atomic_mass : ℝ) (hydrogen_atomic_mass : ℝ) 
  (benzene_formula_ratio : (ℕ × ℕ)) : ℝ := 
    let (num_carbon_atoms, num_hydrogen_atoms) := benzene_formula_ratio
    let total_carbon_mass := num_carbon_atoms * carbon_atomic_mass
    let total_hydrogen_mass := num_hydrogen_atoms * hydrogen_atomic_mass
    let total_mass := total_carbon_mass + total_hydrogen_mass
    100 * (total_carbon_mass / total_mass)

theorem benzene_carbon_mass_percentage 
  (carbon_atomic_mass : ℝ := 12.01) 
  (hydrogen_atomic_mass : ℝ := 1.008) 
  (benzene_formula_ratio : (ℕ × ℕ) := (6, 6)) : 
    carbon_mass_percentage_in_benzene carbon_atomic_mass hydrogen_atomic_mass benzene_formula_ratio = 92.23 :=
by 
  unfold carbon_mass_percentage_in_benzene
  sorry

end benzene_carbon_mass_percentage_l294_294412


namespace rectangle_same_color_exists_l294_294235

def color := ℕ -- We use ℕ as a stand-in for three colors {0, 1, 2}

def same_color_rectangle_exists (coloring : (Fin 4) → (Fin 82) → color) : Prop :=
  ∃ (i j : Fin 4) (k l : Fin 82), i ≠ j ∧ k ≠ l ∧
    coloring i k = coloring i l ∧
    coloring j k = coloring j l ∧
    coloring i k = coloring j k

theorem rectangle_same_color_exists :
  ∀ (coloring : (Fin 4) → (Fin 82) → color),
  same_color_rectangle_exists coloring :=
by
  sorry

end rectangle_same_color_exists_l294_294235


namespace count_two_digit_integers_with_remainder_3_when_divided_by_7_l294_294634

theorem count_two_digit_integers_with_remainder_3_when_divided_by_7 :
  {n : ℕ // 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 := by
sorry

end count_two_digit_integers_with_remainder_3_when_divided_by_7_l294_294634


namespace total_savings_l294_294017

theorem total_savings :
  let J := 0.25 in
  let D_J := 24 in
  let L := 0.50 in
  let D_L := 20 in
  let M := 2 * L in
  let D_M := 12 in
  J * D_J + L * D_L + M * D_M = 28.00 :=
by 
  sorry

end total_savings_l294_294017


namespace lcm_of_two_numbers_l294_294965

theorem lcm_of_two_numbers (a b : ℕ) (h_ratio : a * 3 = b * 2) (h_sum : a + b = 30) : Nat.lcm a b = 18 :=
  sorry

end lcm_of_two_numbers_l294_294965


namespace point_on_x_axis_l294_294294

theorem point_on_x_axis (m : ℝ) (h : (m, m - 1).snd = 0) : m = 1 :=
by
  sorry

end point_on_x_axis_l294_294294


namespace root_expression_of_cubic_l294_294155

theorem root_expression_of_cubic :
  ∀ a b c : ℝ, (a^3 - 2*a - 2 = 0) ∧ (b^3 - 2*b - 2 = 0) ∧ (c^3 - 2*c - 2 = 0)
    → a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = -6 := 
by 
  sorry

end root_expression_of_cubic_l294_294155


namespace sum_of_valid_n_l294_294555

theorem sum_of_valid_n :
  (∑ n in {n : ℤ | (∃ d ∈ ({1, 3, 9} : Finset ℤ), 2 * n - 1 = d)}, n) = 8 := by
sorry

end sum_of_valid_n_l294_294555


namespace largest_multiple_of_8_less_than_100_l294_294999

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l294_294999


namespace paint_weight_correct_l294_294706

def weight_of_paint (total_weight : ℕ) (half_paint_weight : ℕ) : ℕ :=
  2 * (total_weight - half_paint_weight)

theorem paint_weight_correct :
  weight_of_paint 24 14 = 20 := by 
  sorry

end paint_weight_correct_l294_294706


namespace rectangle_parallelepiped_angles_l294_294987

theorem rectangle_parallelepiped_angles 
  (a b c d : ℝ) 
  (α β : ℝ) 
  (h_a : a = d * Real.sin β)
  (h_b : b = d * Real.sin α)
  (h_d : d^2 = (d * Real.sin β)^2 + c^2 + (d * Real.sin α)^2) :
  (α > 0 ∧ β > 0 ∧ α + β < 90) := sorry

end rectangle_parallelepiped_angles_l294_294987


namespace polynomial_sum_l294_294944

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l294_294944


namespace faster_car_distance_l294_294047

theorem faster_car_distance (d v : ℝ) (h_dist: d + 2 * d = 4) (h_faster: 2 * v = 2 * (d / v)) : 
  d = 4 / 3 → 2 * d = 8 / 3 :=
by sorry

end faster_car_distance_l294_294047


namespace pentagon_triangle_area_percentage_l294_294745

def is_equilateral_triangle (s : ℝ) (area : ℝ) : Prop :=
  area = (s^2 * Real.sqrt 3) / 4

def is_square (s : ℝ) (area : ℝ) : Prop :=
  area = s^2

def pentagon_area (square_area triangle_area : ℝ) : ℝ :=
  square_area + triangle_area

noncomputable def percentage (triangle_area pentagon_area : ℝ) : ℝ :=
  (triangle_area / pentagon_area) * 100

theorem pentagon_triangle_area_percentage (s : ℝ) (h₁ : s > 0) :
  let square_area := s^2
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let pentagon_total_area := pentagon_area square_area triangle_area
  let triangle_percentage := percentage triangle_area pentagon_total_area
  triangle_percentage = (100 * (4 * Real.sqrt 3 - 3) / 13) :=
by
  sorry

end pentagon_triangle_area_percentage_l294_294745


namespace distance_to_hospital_l294_294717

theorem distance_to_hospital {total_paid base_price price_per_mile : ℝ} (h1 : total_paid = 23) (h2 : base_price = 3) (h3 : price_per_mile = 4) : (total_paid - base_price) / price_per_mile = 5 :=
by
  sorry

end distance_to_hospital_l294_294717


namespace similarity_coefficient_interval_l294_294411

-- Definitions
def similarTriangles (x y z p : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ x = k * y ∧ y = k * z ∧ z = k * p

-- Theorem statement
theorem similarity_coefficient_interval (x y z p k : ℝ) (h_sim : similarTriangles x y z p) :
  0 ≤ k ∧ k ≤ 2 :=
sorry

end similarity_coefficient_interval_l294_294411


namespace units_digit_sum_is_9_l294_294052

-- Define the units function
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions
def x := 42 ^ 2
def y := 25 ^ 3

-- Define variables for the units digits of x and y
def units_digit_x := units_digit x
def units_digit_y := units_digit y

-- Define the problem statement to be proven
theorem units_digit_sum_is_9 : units_digit (x + y) = 9 :=
by sorry

end units_digit_sum_is_9_l294_294052


namespace circle_radius_l294_294762

theorem circle_radius :
  ∃ radius : ℝ, (∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 16 → (x - 2)^2 + (y - 1)^2 = radius^2)
  ∧ radius = 4 :=
sorry

end circle_radius_l294_294762


namespace volume_ratio_cone_prism_l294_294077

variables (r h : ℝ) (π : ℝ)

def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

def volume_prism (r h : ℝ) : ℝ := 6 * r^2 * h

theorem volume_ratio_cone_prism (r h : ℝ) (π_pos : π > 0) (r_pos : r > 0) (h_pos : h > 0) :
  (volume_cone π r h) / (volume_prism r h) = π / 18 :=
by
  sorry

end volume_ratio_cone_prism_l294_294077


namespace jack_needs_more_money_l294_294934

-- Definitions based on given conditions
def cost_per_sock_pair : ℝ := 9.50
def num_sock_pairs : ℕ := 2
def cost_per_shoe : ℝ := 92
def jack_money : ℝ := 40

-- Theorem statement
theorem jack_needs_more_money (cost_per_sock_pair num_sock_pairs cost_per_shoe jack_money : ℝ) : 
  ((cost_per_sock_pair * num_sock_pairs) + cost_per_shoe) - jack_money = 71 := by
  sorry

end jack_needs_more_money_l294_294934


namespace attendees_not_from_A_B_C_D_l294_294385

theorem attendees_not_from_A_B_C_D
  (num_A : ℕ) (num_B : ℕ) (num_C : ℕ) (num_D : ℕ) (total_attendees : ℕ)
  (hA : num_A = 30)
  (hB : num_B = 2 * num_A)
  (hC : num_C = num_A + 10)
  (hD : num_D = num_C - 5)
  (hTotal : total_attendees = 185)
  : total_attendees - (num_A + num_B + num_C + num_D) = 20 := by
  sorry

end attendees_not_from_A_B_C_D_l294_294385


namespace coordinates_of_P_l294_294789

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

end coordinates_of_P_l294_294789


namespace find_angle_CDE_l294_294446

-- Definition of the angles and their properties
variables {A B C D E : Type}

-- Hypotheses
def angleA_is_right (angleA: ℝ) : Prop := angleA = 90
def angleB_is_right (angleB: ℝ) : Prop := angleB = 90
def angleC_is_right (angleC: ℝ) : Prop := angleC = 90
def angleAEB_value (angleAEB : ℝ) : Prop := angleAEB = 40
def angleBED_eq_angleBDE (angleBED angleBDE : ℝ) : Prop := angleBED = angleBDE

-- The theorem to be proved
theorem find_angle_CDE 
  (angleA : ℝ) (angleB : ℝ) (angleC : ℝ) (angleAEB : ℝ) (angleBED angleBDE : ℝ) (angleCDE : ℝ) :
  angleA_is_right angleA → 
  angleB_is_right angleB → 
  angleC_is_right angleC → 
  angleAEB_value angleAEB → 
  angleBED_eq_angleBDE angleBED angleBDE →
  angleBED = 45 →
  angleCDE = 95 :=
by
  intros
  sorry


end find_angle_CDE_l294_294446


namespace find_A_and_evaluate_A_minus_B_l294_294732

-- Given definitions
def B (x y : ℝ) : ℝ := 4 * x ^ 2 - 3 * y - 1
def result (x y : ℝ) : ℝ := 6 * x ^ 2 - y

-- Defining the polynomial A based on the first condition
def A (x y : ℝ) : ℝ := 2 * x ^ 2 + 2 * y + 1

-- The main theorem to be proven
theorem find_A_and_evaluate_A_minus_B :
  (∀ x y : ℝ, B x y + A x y = result x y) →
  (∀ x y : ℝ, |x - 1| * (y + 1) ^ 2 = 0 → A x y - B x y = -5) :=
by
  intro h1 h2
  sorry

end find_A_and_evaluate_A_minus_B_l294_294732


namespace number_of_members_l294_294018

theorem number_of_members
  (headband_cost : ℕ := 3)
  (jersey_cost : ℕ := 10)
  (total_cost : ℕ := 2700)
  (cost_per_member : ℕ := 26) :
  total_cost / cost_per_member = 103 := by
  sorry

end number_of_members_l294_294018


namespace find_unknown_number_l294_294653

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end find_unknown_number_l294_294653


namespace log2_bounds_l294_294986

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_bounds (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) 
  (h3 : 2^10 = 1024) (h4 : 2^11 = 2048) (h5 : 2^12 = 4096) 
  (h6 : 2^13 = 8192) (h7 : 2^14 = 16384) :
  (3 : ℝ) / 10 < log2 10 ∧ log2 10 < (2 : ℝ) / 7 :=
by
  sorry

end log2_bounds_l294_294986


namespace average_marks_l294_294498

theorem average_marks (num_students : ℕ) (marks1 marks2 marks3 : ℕ) (num1 num2 num3 : ℕ) (h1 : num_students = 50)
  (h2 : marks1 = 90) (h3 : num1 = 10) (h4 : marks2 = marks1 - 10) (h5 : num2 = 15) (h6 : marks3 = 60) 
  (h7 : num1 + num2 + num3 = 50) (h8 : num3 = num_students - (num1 + num2)) (total_marks : ℕ) 
  (h9 : total_marks = (num1 * marks1) + (num2 * marks2) + (num3 * marks3)) : 
  (total_marks / num_students = 72) :=
by
  sorry

end average_marks_l294_294498


namespace shortest_fence_length_l294_294769

open Real

noncomputable def area_of_garden (length width : ℝ) : ℝ := length * width

theorem shortest_fence_length (length width : ℝ) (h : area_of_garden length width = 64) :
  4 * sqrt 64 = 32 :=
by
  -- The statement sets up the condition that the area is 64 and asks to prove minimum perimeter (fence length = perimeter).
  sorry

end shortest_fence_length_l294_294769


namespace max_value_of_z_l294_294103

theorem max_value_of_z (x y z : ℝ) (h_add : x + y + z = 5) (h_mult : x * y + y * z + z * x = 3) : z ≤ 13 / 3 :=
sorry

end max_value_of_z_l294_294103


namespace find_f_l294_294600

noncomputable def func_satisfies_eq (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = x * f x - y * f y

theorem find_f (f : ℝ → ℝ) (h : func_satisfies_eq f) : ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end find_f_l294_294600


namespace number_of_non_officers_l294_294837

theorem number_of_non_officers 
  (avg_salary_employees: ℝ) (avg_salary_officers: ℝ) (avg_salary_nonofficers: ℝ) 
  (num_officers: ℕ) (num_nonofficers: ℕ):
  avg_salary_employees = 120 ∧ avg_salary_officers = 440 ∧ avg_salary_nonofficers = 110 ∧
  num_officers = 15 ∧ 
  (15 * 440 + num_nonofficers * 110 = (15 + num_nonofficers) * 120)  → 
  num_nonofficers = 480 := 
by 
sorry

end number_of_non_officers_l294_294837


namespace nancy_shoes_l294_294466

theorem nancy_shoes (boots slippers heels : ℕ) 
  (h₀ : boots = 6)
  (h₁ : slippers = boots + 9)
  (h₂ : heels = 3 * (boots + slippers)) :
  2 * (boots + slippers + heels) = 168 := by
  sorry

end nancy_shoes_l294_294466


namespace sculpture_cost_in_INR_l294_294467

def USD_per_NAD := 1 / 5
def INR_per_USD := 8
def cost_in_NAD := 200
noncomputable def cost_in_INR := (cost_in_NAD * USD_per_NAD) * INR_per_USD

theorem sculpture_cost_in_INR :
  cost_in_INR = 320 := by
  sorry

end sculpture_cost_in_INR_l294_294467


namespace jack_needs_more_money_l294_294927

variable (cost_per_pair_of_socks : ℝ := 9.50)
variable (number_of_pairs_of_socks : ℕ := 2)
variable (cost_of_soccer_shoes : ℝ := 92.00)
variable (jack_money : ℝ := 40.00)

theorem jack_needs_more_money :
  let total_cost := number_of_pairs_of_socks * cost_per_pair_of_socks + cost_of_soccer_shoes
  let money_needed := total_cost - jack_money
  money_needed = 71.00 := by
  sorry

end jack_needs_more_money_l294_294927


namespace count_two_digit_integers_remainder_3_l294_294625

theorem count_two_digit_integers_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
sorry

end count_two_digit_integers_remainder_3_l294_294625


namespace solve_for_x_l294_294563

theorem solve_for_x (x : ℝ) (h₁: 0.45 * x = 0.15 * (1 + x)) : x = 0.5 :=
by sorry

end solve_for_x_l294_294563


namespace area_increase_l294_294298

theorem area_increase (l w : ℝ) :
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_original := l * w
  let A_new := l_new * w_new
  ((A_new - A_original) / A_original) * 100 = 56 := 
by
  sorry

end area_increase_l294_294298


namespace earnings_difference_l294_294980

-- Define the prices and quantities.
def price_A : ℝ := 4
def price_B : ℝ := 3.5
def quantity_A : ℕ := 300
def quantity_B : ℕ := 350

-- Define the earnings for both companies.
def earnings_A := price_A * quantity_A
def earnings_B := price_B * quantity_B

-- State the theorem we intend to prove.
theorem earnings_difference :
  earnings_B - earnings_A = 25 := by
  sorry

end earnings_difference_l294_294980


namespace graph_does_not_pass_first_quadrant_l294_294704

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2

theorem graph_does_not_pass_first_quadrant :
  ¬ ∃ x > 0, f x > 0 := by
sorry

end graph_does_not_pass_first_quadrant_l294_294704


namespace eggs_from_Martha_is_2_l294_294449

def eggs_from_Gertrude : ℕ := 4
def eggs_from_Blanche : ℕ := 3
def eggs_from_Nancy : ℕ := 2
def total_eggs_left : ℕ := 9
def eggs_dropped : ℕ := 2

def total_eggs_before_dropping (eggs_from_Martha : ℕ) :=
  eggs_from_Gertrude + eggs_from_Blanche + eggs_from_Nancy + eggs_from_Martha - eggs_dropped = total_eggs_left

-- The theorem stating the eggs collected from Martha.
theorem eggs_from_Martha_is_2 : ∃ (m : ℕ), total_eggs_before_dropping m ∧ m = 2 :=
by
  use 2
  sorry

end eggs_from_Martha_is_2_l294_294449


namespace customers_in_other_countries_l294_294738

def total_customers : ℕ := 7422
def us_customers : ℕ := 723
def other_customers : ℕ := total_customers - us_customers

theorem customers_in_other_countries : other_customers = 6699 := by
  sorry

end customers_in_other_countries_l294_294738


namespace children_total_savings_l294_294012

theorem children_total_savings :
  let josiah_savings := 0.25 * 24
  let leah_savings := 0.50 * 20
  let megan_savings := (2 * 0.50) * 12
  josiah_savings + leah_savings + megan_savings = 28 := by
{
  -- lean proof goes here
  sorry
}

end children_total_savings_l294_294012


namespace f_2011_equals_1_l294_294971

-- Define odd function property
def is_odd_function (f : ℤ → ℤ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Define function with period property
def has_period_3 (f : ℤ → ℤ) : Prop :=
  ∀ x, f (x + 3) = f (x)

-- Define main problem statement
theorem f_2011_equals_1 
  (f : ℤ → ℤ)
  (h1 : is_odd_function f)
  (h2 : has_period_3 f)
  (h3 : f (-1) = -1) 
  : f 2011 = 1 :=
sorry

end f_2011_equals_1_l294_294971


namespace parallel_line_slope_l294_294591

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 9) : ∃ (m : ℝ), m = 1 / 2 := 
sorry

end parallel_line_slope_l294_294591


namespace no_such_reals_exist_l294_294765

-- Define the existence of distinct real numbers such that the given condition holds
theorem no_such_reals_exist :
  ¬ ∃ x y z : ℝ, (x ≠ y) ∧ (y ≠ z) ∧ (z ≠ x) ∧ 
  (1 / (x^2 - y^2) + 1 / (y^2 - z^2) + 1 / (z^2 - x^2) = 0) :=
by
  -- Placeholder for proof
  sorry

end no_such_reals_exist_l294_294765


namespace remainder_sum_mod_15_l294_294823

variable (k j : ℤ) -- these represent any integers

def p := 60 * k + 53
def q := 75 * j + 24

theorem remainder_sum_mod_15 :
  (p k + q j) % 15 = 2 :=  
by 
  sorry

end remainder_sum_mod_15_l294_294823


namespace find_exponent_l294_294241

theorem find_exponent (y : ℝ) (exponent : ℝ) :
  (12^1 * 6^exponent / 432 = y) → (y = 36) → (exponent = 3) :=
by 
  intros h₁ h₂ 
  sorry

end find_exponent_l294_294241


namespace area_of_side_face_of_box_l294_294201

theorem area_of_side_face_of_box:
  ∃ (l w h : ℝ), (w * h = (1/2) * (l * w)) ∧
                 (l * w = 1.5 * (l * h)) ∧
                 (l * w * h = 3000) ∧
                 ((l * h) = 200) :=
sorry

end area_of_side_face_of_box_l294_294201


namespace largest_multiple_of_8_less_than_100_l294_294991

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l294_294991


namespace theater_ticket_sales_l294_294193

theorem theater_ticket_sales (A K : ℕ) (h1 : A + K = 275) (h2 :  12 * A + 5 * K = 2150) : K = 164 := by
  sorry

end theater_ticket_sales_l294_294193


namespace two_digit_integers_remainder_3_count_l294_294632

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l294_294632


namespace pizzas_needed_l294_294974

theorem pizzas_needed (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) (h_people : people = 18) (h_slices_per_person : slices_per_person = 3) (h_slices_per_pizza : slices_per_pizza = 9) :
  people * slices_per_person / slices_per_pizza = 6 :=
by
  sorry

end pizzas_needed_l294_294974


namespace jessica_deposited_fraction_l294_294674

-- Definitions based on conditions
def original_balance (B : ℝ) : Prop :=
  B * (3 / 5) = B - 200

def final_balance (B : ℝ) (F : ℝ) : Prop :=
  ((3 / 5) * B) + (F * ((3 / 5) * B)) = 360

-- Theorem statement proving that the fraction deposited is 1/5
theorem jessica_deposited_fraction (B : ℝ) (F : ℝ) (h1 : original_balance B) (h2 : final_balance B F) : F = 1 / 5 :=
  sorry

end jessica_deposited_fraction_l294_294674


namespace total_bowling_balls_is_66_l294_294183

-- Define the given conditions
def red_bowling_balls := 30
def difference_green_red := 6
def green_bowling_balls := red_bowling_balls + difference_green_red

-- The statement to prove
theorem total_bowling_balls_is_66 :
  red_bowling_balls + green_bowling_balls = 66 := by
  sorry

end total_bowling_balls_is_66_l294_294183


namespace krystiana_monthly_income_l294_294148

theorem krystiana_monthly_income :
  let first_floor_income := 3 * 15
  let second_floor_income := 3 * 20
  let third_floor_income := 2 * (2 * 15)
  first_floor_income + second_floor_income + third_floor_income = 165 :=
by
  let first_floor_income := 3 * 15
  let second_floor_income := 3 * 20
  let third_floor_income := 2 * (2 * 15)
  have h1: first_floor_income = 45 := by simp [first_floor_income]
  have h2: second_floor_income = 60 := by simp [second_floor_income]
  have h3: third_floor_income = 60 := by simp [third_floor_income]
  rw [h1, h2, h3]
  simp
  done

end krystiana_monthly_income_l294_294148


namespace monthly_earnings_is_correct_l294_294150

-- Conditions as definitions

def first_floor_cost_per_room : ℕ := 15
def second_floor_cost_per_room : ℕ := 20
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms : ℕ := 3
def occupied_third_floor_rooms : ℕ := 2

-- Calculated values from conditions
def third_floor_cost_per_room : ℕ := 2 * first_floor_cost_per_room

-- Total earnings on each floor
def first_floor_earnings : ℕ := first_floor_cost_per_room * first_floor_rooms
def second_floor_earnings : ℕ := second_floor_cost_per_room * second_floor_rooms
def third_floor_earnings : ℕ := third_floor_cost_per_room * occupied_third_floor_rooms

-- Total monthly earnings
def total_monthly_earnings : ℕ :=
  first_floor_earnings + second_floor_earnings + third_floor_earnings

theorem monthly_earnings_is_correct : total_monthly_earnings = 165 := by
  -- proof omitted
  sorry

end monthly_earnings_is_correct_l294_294150


namespace sticker_price_l294_294911

theorem sticker_price (x : ℝ) (h1 : 0.8 * x - 100 = 0.7 * x - 25) : x = 750 :=
by
  sorry

end sticker_price_l294_294911


namespace general_term_of_sequence_l294_294669

theorem general_term_of_sequence (a : Nat → ℚ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) / (2 + a (n + 1))) :
  ∀ n : ℕ, a (n + 1) = 2 / (n + 2) := 
sorry

end general_term_of_sequence_l294_294669


namespace islanders_liars_l294_294333

theorem islanders_liars (n : ℕ) (h : n = 450) : (∃ L : ℕ, (L = 150 ∨ L = 450)) :=
sorry

end islanders_liars_l294_294333


namespace largest_multiple_of_8_less_than_100_l294_294997

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l294_294997


namespace triangle_area_dodecagon_l294_294846

noncomputable def least_possible_triangle_area : ℂ := 
  let r := 2 * real.sqrt 3
  let vertices := λ k, r * complex.exp (2 * real.pi * I * k / 12)
  let D := vertices 0
  let E := vertices 1
  let F := vertices 2
  let base := complex.abs (E - F)
  let height := complex.abs (D - (E + F) / 2)
  (1 / 2) * base * height

theorem triangle_area_dodecagon :
  least_possible_triangle_area = 3 * real.sqrt 3 := by
  sorry

end triangle_area_dodecagon_l294_294846


namespace max_f_and_sin_alpha_l294_294246

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * Real.cos x

theorem max_f_and_sin_alpha :
  (∀ x : ℝ, f x ≤ Real.sqrt 5) ∧ (∃ α : ℝ, (α + Real.arccos (1 / Real.sqrt 5) = π / 2 + 2 * π * some_integer) ∧ (f α = Real.sqrt 5) ∧ (Real.sin α = 1 / Real.sqrt 5)) :=
by
  sorry

end max_f_and_sin_alpha_l294_294246


namespace log_inequality_l294_294898

theorem log_inequality (a x y : ℝ) (ha : 0 < a) (ha_lt_1 : a < 1) 
(h : x^2 + y = 0) : 
  Real.log (a^x + a^y) / Real.log a ≤ Real.log 2 / Real.log a + 1 / 8 :=
sorry

end log_inequality_l294_294898


namespace exists_positive_integers_for_equation_l294_294586

theorem exists_positive_integers_for_equation :
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a^4 = b^3 + c^2 :=
by
  sorry

end exists_positive_integers_for_equation_l294_294586


namespace tire_miles_used_l294_294712

theorem tire_miles_used (total_miles : ℕ) (number_of_tires : ℕ) (tires_in_use : ℕ)
  (h_total_miles : total_miles = 40000) (h_number_of_tires : number_of_tires = 6)
  (h_tires_in_use : tires_in_use = 4) : 
  (total_miles * tires_in_use) / number_of_tires = 26667 := 
by 
  sorry

end tire_miles_used_l294_294712


namespace largest_multiple_of_8_less_than_100_l294_294996

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l294_294996


namespace total_students_in_class_l294_294132

-- Definitions of the conditions
def E : ℕ := 55
def T : ℕ := 85
def N : ℕ := 30
def B : ℕ := 20

-- Statement of the theorem to prove the total number of students
theorem total_students_in_class : (E + T - B) + N = 150 := by
  -- Proof is omitted
  sorry

end total_students_in_class_l294_294132


namespace four_digit_number_8802_l294_294237

theorem four_digit_number_8802 (x : ℕ) (a b c d : ℕ) (h1 : 1000 ≤ x ∧ x ≤ 9999)
  (h2 : x = 1000 * a + 100 * b + 10 * c + d)
  (h3 : a ≠ 0)  -- since a 4-digit number cannot start with 0
  (h4 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) : 
  x + 8802 = 1099 + 8802 :=
by
  sorry

end four_digit_number_8802_l294_294237


namespace probability_ab_gt_a_add_b_l294_294515

theorem probability_ab_gt_a_add_b :
  let S := {1, 2, 3, 4, 5, 6}
  let all_pairs := S.product S
  let valid_pairs := { p : ℕ × ℕ | p.1 * p.2 > p.1 + p.2 ∧ p.1 ∈ S ∧ p.2 ∈ S }
  (all_pairs.card > 0) →
  (all_pairs ≠ ∅) →
  (all_pairs.card = 36) →
  (2 * valid_pairs.card = 46) →
  valid_pairs.card / all_pairs.card = (23 : ℚ) / 36 := sorry

end probability_ab_gt_a_add_b_l294_294515


namespace sum_of_integers_from_neg15_to_5_l294_294589

theorem sum_of_integers_from_neg15_to_5 : 
  (∑ x in Finset.Icc (-15 : ℤ) 5, x) = -105 := 
by
  sorry

end sum_of_integers_from_neg15_to_5_l294_294589


namespace train_length_is_199_95_l294_294579

noncomputable def convert_speed_to_m_s (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000) / 3600

noncomputable def length_of_train (bridge_length : ℝ) (time_seconds : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := convert_speed_to_m_s speed_kmh
  speed_ms * time_seconds - bridge_length

theorem train_length_is_199_95 :
  length_of_train 300 45 40 = 199.95 := by
  sorry

end train_length_is_199_95_l294_294579


namespace expand_binom_l294_294409

theorem expand_binom (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 :=
by
  sorry

end expand_binom_l294_294409


namespace how_fast_is_a_l294_294289

variable (a b : ℝ) (k : ℝ)

theorem how_fast_is_a (h1 : a = k * b) (h2 : a + b = 1 / 30) (h3 : a = 1 / 40) : k = 3 := sorry

end how_fast_is_a_l294_294289


namespace ratio_senior_junior_l294_294749

theorem ratio_senior_junior
  (J S : ℕ)
  (h1 : ∃ k : ℕ, S = k * J)
  (h2 : (3 / 8) * S + (1 / 4) * J = (1 / 3) * (S + J)) :
  S = 2 * J :=
by
  -- The proof is to be provided
  sorry

end ratio_senior_junior_l294_294749


namespace tv_weight_calculations_l294_294089

theorem tv_weight_calculations
    (w1 h1 r1 : ℕ) -- Represents Bill's TV dimensions and weight ratio
    (w2 h2 r2 : ℕ) -- Represents Bob's TV dimensions and weight ratio
    (w3 h3 r3 : ℕ) -- Represents Steve's TV dimensions and weight ratio
    (ounce_to_pound: ℕ) -- Represents the conversion factor from ounces to pounds
    (bill_tv_weight bob_tv_weight steve_tv_weight : ℕ) -- Computed weights in pounds
    (weight_diff: ℕ):
  (w1 * h1 * r1) / ounce_to_pound = bill_tv_weight → -- Bill's TV weight calculation
  (w2 * h2 * r2) / ounce_to_pound = bob_tv_weight → -- Bob's TV weight calculation
  (w3 * h3 * r3) / ounce_to_pound = steve_tv_weight → -- Steve's TV weight calculation
  steve_tv_weight > (bill_tv_weight + bob_tv_weight) → -- Steve's TV is the heaviest
  steve_tv_weight - (bill_tv_weight + bob_tv_weight) = weight_diff → -- weight difference calculation
  True := sorry

end tv_weight_calculations_l294_294089


namespace hyperbola_asymptote_b_l294_294295

theorem hyperbola_asymptote_b {b : ℝ} (hb : b > 0) :
  (∀ x y : ℝ, x^2 - (y^2 / b^2) = 1 → (y = 2 * x)) → b = 2 := by
  sorry

end hyperbola_asymptote_b_l294_294295


namespace remaining_dogs_after_adoptions_l294_294710

theorem remaining_dogs_after_adoptions 
  (initial_dogs : ℕ)
  (additional_dogs : ℕ)
  (adopted_week1 : ℕ)
  (adopted_week4 : ℕ) :
  initial_dogs = 200 →
  additional_dogs = 100 →
  adopted_week1 = 40 →
  adopted_week4 = 60 →
  initial_dogs + additional_dogs - adopted_week1 - adopted_week4 = 200 :=
by
  intros h_init h_add h_adopt1 h_adopt2
  rw [h_init, h_add, h_adopt1, h_adopt2]
  rfl

end remaining_dogs_after_adoptions_l294_294710


namespace slices_per_pizza_l294_294721

theorem slices_per_pizza (total_slices number_of_pizzas slices_per_pizza : ℕ) 
  (h_total_slices : total_slices = 168) 
  (h_number_of_pizzas : number_of_pizzas = 21) 
  (h_division : total_slices / number_of_pizzas = slices_per_pizza) : 
  slices_per_pizza = 8 :=
sorry

end slices_per_pizza_l294_294721


namespace judge_guilty_cases_l294_294213

theorem judge_guilty_cases :
  let total_cases := 27
  let dismissed_cases := 3
  let remaining_cases := total_cases - dismissed_cases
  let innocent_cases := 3 * remaining_cases / 4
  let delayed_rulings := 2
  remaining_cases - innocent_cases - delayed_rulings = 4 :=
by
  let total_cases := 27
  let dismissed_cases := 3
  let remaining_cases := total_cases - dismissed_cases
  let innocent_cases := 3 * remaining_cases / 4
  let delayed_rulings := 2
  show remaining_cases - innocent_cases - delayed_rulings = 4
  sorry

end judge_guilty_cases_l294_294213


namespace more_students_suggested_bacon_than_mashed_potatoes_l294_294751

-- Define the number of students suggesting each type of food
def students_suggesting_mashed_potatoes := 479
def students_suggesting_bacon := 489

-- State the theorem that needs to be proven
theorem more_students_suggested_bacon_than_mashed_potatoes :
  students_suggesting_bacon - students_suggesting_mashed_potatoes = 10 := 
  by
  sorry

end more_students_suggested_bacon_than_mashed_potatoes_l294_294751


namespace simplify_evaluate_expression_l294_294171

theorem simplify_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6)) = Real.sqrt 2 :=
by
  sorry

end simplify_evaluate_expression_l294_294171


namespace compare_magnitudes_l294_294156

noncomputable def A : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))
noncomputable def B : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def C : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def D : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))

theorem compare_magnitudes : B < C ∧ C < A ∧ A < D :=
by
  sorry

end compare_magnitudes_l294_294156


namespace pictures_on_front_l294_294310

-- Conditions
variable (total_pictures : ℕ)
variable (pictures_on_back : ℕ)

-- Proof obligation
theorem pictures_on_front (h1 : total_pictures = 15) (h2 : pictures_on_back = 9) : total_pictures - pictures_on_back = 6 :=
sorry

end pictures_on_front_l294_294310


namespace problem1_l294_294378

theorem problem1 (a : ℝ) (m n : ℕ) (h1 : a^m = 10) (h2 : a^n = 2) : a^(m - 2 * n) = 2.5 := by
  sorry

end problem1_l294_294378


namespace justin_current_age_l294_294585

theorem justin_current_age (angelina_future_age : ℕ) (years_until_future : ℕ) (age_difference : ℕ)
  (h_future_age : angelina_future_age = 40) (h_years_until_future : years_until_future = 5)
  (h_age_difference : age_difference = 4) : 
  (angelina_future_age - years_until_future) - age_difference = 31 :=
by
  -- This is where the proof would go.
  sorry

end justin_current_age_l294_294585


namespace jack_money_proof_l294_294930

variable (sock_cost : ℝ) (number_of_socks : ℕ) (shoe_cost : ℝ) (available_money : ℝ)

def jack_needs_more_money : Prop := 
  (number_of_socks * sock_cost + shoe_cost) - available_money = 71

theorem jack_money_proof (h1 : sock_cost = 9.50)
                         (h2 : number_of_socks = 2)
                         (h3 : shoe_cost = 92)
                         (h4 : available_money = 40) :
  jack_needs_more_money sock_cost number_of_socks shoe_cost available_money :=
by
  unfold jack_needs_more_money
  simp [h1, h2, h3, h4]
  norm_num
  done

end jack_money_proof_l294_294930


namespace crows_and_trees_l294_294877

variable (x y : ℕ)

theorem crows_and_trees (h1 : x = 3 * y + 5) (h2 : x = 5 * (y - 1)) : 
  (x - 5) / 3 = y ∧ x / 5 = y - 1 :=
by
  sorry

end crows_and_trees_l294_294877


namespace sum_lt_prod_probability_l294_294522

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l294_294522


namespace range_f_l294_294788

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.sqrt (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) + 
  Real.sqrt (a * Real.sin x ^ 2 + b * Real.cos x ^ 2)

theorem range_f (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  Set.range (f a b) = Set.Icc (Real.sqrt a + Real.sqrt b) (Real.sqrt (2 * (a + b))) :=
sorry

end range_f_l294_294788


namespace total_shoes_l294_294464

variable boots : ℕ
variable slippers : ℕ
variable heels : ℕ

-- Condition: Nancy has six pairs of boots
def boots_pairs : boots = 6 := rfl

-- Condition: Nancy has nine more pairs of slippers than boots
def slippers_pairs : slippers = boots + 9 := rfl

-- Condition: Nancy has a number of pairs of heels equal to three times the combined number of slippers and boots
def heels_pairs : heels = 3 * (boots + slippers) := by
  rw [boots_pairs, slippers_pairs]
  sorry  -- assuming the correctness of the consequent computation as rfl

-- Goal: Total number of individual shoes is 168
theorem total_shoes : (boots * 2) + (slippers * 2) + (heels * 2) = 168 := by
  rw [boots_pairs, slippers_pairs, heels_pairs]
  sorry  -- verifying the summing up to 168 as a proof

end total_shoes_l294_294464


namespace number_of_tables_l294_294806

noncomputable def stools_per_table : ℕ := 7
noncomputable def legs_per_stool : ℕ := 4
noncomputable def legs_per_table : ℕ := 5
noncomputable def total_legs : ℕ := 658

theorem number_of_tables : 
  ∃ t : ℕ, 
  (∃ s : ℕ, s = stools_per_table * t ∧ legs_per_stool * s + legs_per_table * t = total_legs) ∧ t = 20 :=
by {
  sorry
}

end number_of_tables_l294_294806


namespace car_a_has_higher_avg_speed_l294_294227

-- Definitions of the conditions for Car A
def distance_car_a : ℕ := 120
def speed_segment_1_car_a : ℕ := 60
def distance_segment_1_car_a : ℕ := 40
def speed_segment_2_car_a : ℕ := 40
def distance_segment_2_car_a : ℕ := 40
def speed_segment_3_car_a : ℕ := 80
def distance_segment_3_car_a : ℕ := distance_car_a - distance_segment_1_car_a - distance_segment_2_car_a

-- Definitions of the conditions for Car B
def distance_car_b : ℕ := 120
def time_segment_1_car_b : ℕ := 1
def speed_segment_1_car_b : ℕ := 60
def time_segment_2_car_b : ℕ := 1
def speed_segment_2_car_b : ℕ := 40
def total_time_car_b : ℕ := 3
def distance_segment_1_car_b := speed_segment_1_car_b * time_segment_1_car_b
def distance_segment_2_car_b := speed_segment_2_car_b * time_segment_2_car_b
def time_segment_3_car_b := total_time_car_b - time_segment_1_car_b - time_segment_2_car_b
def distance_segment_3_car_b := distance_car_b - distance_segment_1_car_b - distance_segment_2_car_b
def speed_segment_3_car_b := distance_segment_3_car_b / time_segment_3_car_b

-- Total Time for Car A
def time_car_a := distance_segment_1_car_a / speed_segment_1_car_a
                + distance_segment_2_car_a / speed_segment_2_car_a
                + distance_segment_3_car_a / speed_segment_3_car_a

-- Average Speed for Car A
def avg_speed_car_a := distance_car_a / time_car_a

-- Total Time for Car B
def time_car_b := total_time_car_b

-- Average Speed for Car B
def avg_speed_car_b := distance_car_b / time_car_b

-- Proof that Car A has a higher average speed than Car B
theorem car_a_has_higher_avg_speed : avg_speed_car_a > avg_speed_car_b := by sorry

end car_a_has_higher_avg_speed_l294_294227


namespace island_liars_l294_294331

theorem island_liars (n : ℕ) (h₁ : n = 450) (h₂ : ∀ (i : ℕ), i < 450 → 
  ∃ (a : bool),  (if a then (i + 1) % 450 else (i + 2) % 450) = "liar"):
    (n = 150 ∨ n = 450) :=
sorry

end island_liars_l294_294331


namespace find_other_digits_l294_294352

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem find_other_digits (n : ℕ) (h : ℕ) :
  tens_digit n = h →
  h = 1 →
  is_divisible_by_9 n →
  ∃ m : ℕ, m < 9 ∧ n = 10 * ((n / 10) / 10) * 10 + h * 10 + m ∧ (∃ k : ℕ, k * 9 = h + m + (n / 100)) :=
sorry

end find_other_digits_l294_294352


namespace number_of_ordered_pairs_l294_294776

noncomputable def count_valid_ordered_pairs (a b: ℝ) : Prop :=
  ∃ (x y : ℤ), a * (x : ℝ) + b * (y : ℝ) = 2 ∧ x^2 + y^2 = 65

theorem number_of_ordered_pairs : ∃ s : Finset (ℝ × ℝ), s.card = 128 ∧ ∀ (p : ℝ × ℝ), p ∈ s ↔ count_valid_ordered_pairs p.1 p.2 :=
by
  sorry

end number_of_ordered_pairs_l294_294776


namespace chicken_coop_problem_l294_294384

-- Definitions of conditions
def available_area : ℝ := 240
def area_per_chicken : ℝ := 4
def area_per_chick : ℝ := 2
def max_daily_feed : ℝ := 8000
def feed_per_chicken : ℝ := 160
def feed_per_chick : ℝ := 40

-- Variables representing the number of chickens and chicks
variables (x y : ℕ)

-- Condition expressions
def space_condition (x y : ℕ) : Prop := 
  (2 * x + y = (available_area / area_per_chick))

def feed_condition (x y : ℕ) : Prop := 
  ((4 * x + y) * feed_per_chick <= max_daily_feed / feed_per_chick)

-- Given conditions and queries proof problem
theorem chicken_coop_problem : 
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 20 ∧ y = 80)) 
  ∧
  (¬ ∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 30 ∧ y = 100))
  ∧
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 40 ∧ y = 40))
  ∧
  (∃ x y : ℕ, space_condition x y ∧ feed_condition x y ∧ (x = 0 ∧ y = 120)) :=
by
  sorry  -- The proof will be provided here.


end chicken_coop_problem_l294_294384


namespace triangle_DEF_all_acute_l294_294809

theorem triangle_DEF_all_acute
  (α : ℝ)
  (hα : 0 < α ∧ α < 90)
  (DEF : Type)
  (D : DEF) (E : DEF) (F : DEF)
  (angle_DFE : DEF → DEF → DEF → ℝ) 
  (angle_FED : DEF → DEF → DEF → ℝ) 
  (angle_EFD : DEF → DEF → DEF → ℝ)
  (h1 : angle_DFE D F E = 45)
  (h2 : angle_FED F E D = 90 - α / 2)
  (h3 : angle_EFD E D F = 45 + α / 2) :
  (0 < angle_DFE D F E ∧ angle_DFE D F E < 90) ∧ 
  (0 < angle_FED F E D ∧ angle_FED F E D < 90) ∧ 
  (0 < angle_EFD E D F ∧ angle_EFD E D F < 90) := by
  sorry

end triangle_DEF_all_acute_l294_294809


namespace john_profit_l294_294142

theorem john_profit (cost_per_bag selling_price : ℕ) (number_of_bags : ℕ) (profit_per_bag total_profit : ℕ) :
  cost_per_bag = 4 →
  selling_price = 8 →
  number_of_bags = 30 →
  profit_per_bag = selling_price - cost_per_bag →
  total_profit = number_of_bags * profit_per_bag →
  total_profit = 120 :=
by
  intro h_cost h_sell h_num_bags h_profit_per_bag h_total_profit
  rw [h_profit_per_bag, h_cost, h_sell] at h_profit_per_bag
  rw [h_total_profit, h_num_bags, h_profit_per_bag]
  norm_num

end john_profit_l294_294142


namespace second_smallest_integer_l294_294775

theorem second_smallest_integer (x y z w v : ℤ) (h_avg : (x + y + z + w + v) / 5 = 69)
  (h_median : z = 83) (h_mode : w = 85 ∧ v = 85) (h_range : 85 - x = 70) :
  y = 77 :=
by
  sorry

end second_smallest_integer_l294_294775


namespace probability_sum_less_than_product_is_5_div_9_l294_294528

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l294_294528


namespace mean_median_difference_is_correct_l294_294330

noncomputable def mean_median_difference (scores : List ℕ) (percentages : List ℚ) : ℚ := sorry

theorem mean_median_difference_is_correct :
  mean_median_difference [60, 75, 85, 90, 100] [15/100, 20/100, 25/100, 30/100, 10/100] = 2.75 :=
sorry

end mean_median_difference_is_correct_l294_294330


namespace non_coincident_angles_l294_294221

theorem non_coincident_angles : ¬ ∃ k : ℤ, 1050 - (-300) = k * 360 := by
  sorry

end non_coincident_angles_l294_294221


namespace value_of_m_minus_n_l294_294606

theorem value_of_m_minus_n (m n : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (m : ℂ) / (1 + i) = 1 - n * i) : m - n = 1 :=
sorry

end value_of_m_minus_n_l294_294606


namespace regular_21_gon_symmetries_and_angle_sum_l294_294076

theorem regular_21_gon_symmetries_and_angle_sum :
  let L' := 21
  let R' := 360 / 21
  L' + R' = 38.142857 := by
    sorry

end regular_21_gon_symmetries_and_angle_sum_l294_294076


namespace find_k_and_other_root_l294_294687

def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem find_k_and_other_root (k β : ℝ) (h1 : quadratic_eq 4 k 2 (-0.5)) (h2 : 4 * (-0.5) ^ 2 + k * (-0.5) + 2 = 0) : 
  k = 6 ∧ β = -1 ∧ quadratic_eq 4 k 2 β := 
by 
  sorry

end find_k_and_other_root_l294_294687


namespace domain_of_f_l294_294117

def f (x : ℝ) : ℝ := 2 / Real.logb (1 / 2) (2 * x + 1)

theorem domain_of_f :
  {x : ℝ | 2 * x + 1 > 0 ∧ Real.logb (1 / 2) (2 * x + 1) ≠ 0} =
  {x : ℝ | x > -1/2 ∧ x ≠ 0} :=
by
  ext x
  simp only [set.mem_set_of_eq]
  split
  { intro h
    cases h with h1 h2
    split
    { linarith only [h1] }
    { rw [ne, ←Real.logb_eq_zero_iff] at h2
      simp at h2
      exact h2 }
  }
  { intro h
    cases h with h1 h2
    split
    { linarith only [h1] }
    { intro hlog
      rw Real.logb_eq_zero_iff at hlog
      simp at hlog
      contradiction }
  }

lemma domain_of_f_correct :
  {x : ℝ | x > -1/2 ∧ x ≠ 0} = set.Ioo (-1/2) 0 ∪ set.Ioi 0 :=
by
  ext x
  simp only [set.mem_set_of_eq, set.mem_Ioo, set.mem_Ioi, set.mem_union_eq]
  split
  { rintro ⟨hl,h⟩
    cases (lt_trichotomy x 0) with hlt heqgt
    { left
      exact ⟨hl,hlt⟩ }
    { cases heqgt 
      { exfalso
        exact h heqgt }
      { right
        exact heqgt }
    }
  }
  { intro h
    cases h
    { cases h with h₁ h₂
      exact ⟨h₁, h₂.2⟩ }
    { exact ⟨lt_of_le_of_ne (le_of_lt h) h.symm⟩ }
  }

example : {x : ℝ | 2 * x + 1 > 0 ∧ Real.logb (1 / 2) (2 * x + 1) ≠ 0} =
    set.Ioo (-1/2) 0 ∪ set.Ioi 0 :=
by
  rw [domain_of_f, domain_of_f_correct]
  sorry

end domain_of_f_l294_294117


namespace charges_equal_at_x_4_cost_effectiveness_l294_294500

-- Defining the conditions
def full_price : ℕ := 240

def yA (x : ℕ) : ℕ := 120 * x + 240
def yB (x : ℕ) : ℕ := 144 * x + 144

-- (Ⅰ) Establishing the expressions for the charges is already encapsulated in the definitions.

-- (Ⅱ) Proving the equivalence of the two charges for a specific number of students x.
theorem charges_equal_at_x_4 : ∀ x : ℕ, yA x = yB x ↔ x = 4 := 
by {
  sorry
}

-- (Ⅲ) Discussing which travel agency is more cost-effective based on the number of students x.
theorem cost_effectiveness (x : ℕ) :
  (x < 4 → yA x > yB x) ∧ (x > 4 → yA x < yB x) :=
by {
  sorry
}

end charges_equal_at_x_4_cost_effectiveness_l294_294500


namespace find_number_l294_294651

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end find_number_l294_294651


namespace union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_in_U_l294_294424

open Set

noncomputable def U : Set ℤ := {x | -2 < x ∧ x < 2}
def A : Set ℤ := {x | x^2 - 5 * x - 6 = 0}
def B : Set ℤ := {x | x^2 = 1}

theorem union_of_A_and_B : A ∪ B = {-1, 1, 6} :=
by
  sorry

theorem intersection_of_A_and_B : A ∩ B = {-1} :=
by
  sorry

theorem complement_of_intersection_in_U : U \ (A ∩ B) = {0, 1} :=
by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_in_U_l294_294424


namespace product_equals_permutation_l294_294207

-- Definitions and conditions
def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Given product sequence
def product_seq (n k : ℕ) : ℕ :=
  (List.range' (n - k + 1) k).foldr (λ x y => x * y) 1

-- Problem statement: The product of numbers from 18 to 9 is equivalent to A_{18}^{10}
theorem product_equals_permutation :
  product_seq 18 10 = perm 18 10 :=
by
  sorry

end product_equals_permutation_l294_294207


namespace number_of_white_balls_l294_294925

theorem number_of_white_balls (x : ℕ) (h1 : 3 + x ≠ 0) (h2 : (3 : ℚ) / (3 + x) = 1 / 5) : x = 12 :=
sorry

end number_of_white_balls_l294_294925


namespace find_geometric_progression_l294_294045

theorem find_geometric_progression (a b c : ℚ)
  (h1 : a * c = b * b)
  (h2 : a + c = 2 * (b + 8))
  (h3 : a * (c + 64) = (b + 8) * (b + 8)) :
  (a = 4/9 ∧ b = -20/9 ∧ c = 100/9) ∨ (a = 4 ∧ b = 12 ∧ c = 36) :=
sorry

end find_geometric_progression_l294_294045


namespace ball_hits_ground_at_5_over_2_l294_294968

noncomputable def ball_height (t : ℝ) : ℝ := -16 * t^2 + 40 * t + 60

theorem ball_hits_ground_at_5_over_2 :
  ∃ t : ℝ, t = 5 / 2 ∧ ball_height t = 0 :=
sorry

end ball_hits_ground_at_5_over_2_l294_294968


namespace B_and_C_mutually_exclusive_but_not_complementary_l294_294395

-- Define the sample space of the cube
def faces : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define events based on conditions
def event_A (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_B (n : ℕ) : Prop := n = 1 ∨ n = 2
def event_C (n : ℕ) : Prop := n = 4 ∨ n = 5 ∨ n = 6

-- Define mutually exclusive events
def mutually_exclusive (A B : ℕ → Prop) : Prop := ∀ n, A n → ¬ B n

-- Define complementary events (for events over finite sample spaces like faces)
-- Events A and B are complementary if they partition the sample space faces
def complementary (A B : ℕ → Prop) : Prop := (∀ n, n ∈ faces → A n ∨ B n) ∧ (∀ n, A n → ¬ B n) ∧ (∀ n, B n → ¬ A n)

theorem B_and_C_mutually_exclusive_but_not_complementary :
  mutually_exclusive event_B event_C ∧ ¬ complementary event_B event_C := 
by
  sorry

end B_and_C_mutually_exclusive_but_not_complementary_l294_294395


namespace justin_current_age_l294_294584

theorem justin_current_age (angelina_future_age : ℕ) (years_until_future : ℕ) (age_difference : ℕ)
  (h_future_age : angelina_future_age = 40) (h_years_until_future : years_until_future = 5)
  (h_age_difference : age_difference = 4) : 
  (angelina_future_age - years_until_future) - age_difference = 31 :=
by
  -- This is where the proof would go.
  sorry

end justin_current_age_l294_294584


namespace decreasing_interval_l294_294792

def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative function
def f_prime (x : ℝ) : ℝ := 3*x^2 - 3

theorem decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x < 0 :=
by
  intro x h
  have h1: x^2 < 1 := by
    sorry
  have h2: 3*x^2 < 3 := by
    sorry
  have h3: 3*x^2 - 3 < 0 := by
    sorry
  exact h3

end decreasing_interval_l294_294792


namespace triangle_angles_l294_294924

theorem triangle_angles
  (h_a a h_b b : ℝ)
  (h_a_ge_a : h_a ≥ a)
  (h_b_ge_b : h_b ≥ b)
  (a_ge_h_b : a ≥ h_b)
  (b_ge_h_a : b ≥ h_a) : 
  a = b ∧ 
  (a = h_a ∧ b = h_b) → 
  ∃ A B C : ℝ, Set.toFinset ({A, B, C} : Set ℝ) = {90, 45, 45} := 
by 
  sorry

end triangle_angles_l294_294924


namespace car_mass_nearest_pound_l294_294572

def mass_of_car_kg : ℝ := 1500
def kg_to_pounds : ℝ := 0.4536

theorem car_mass_nearest_pound :
  (↑(Int.floor ((mass_of_car_kg / kg_to_pounds) + 0.5))) = 3307 :=
by
  sorry

end car_mass_nearest_pound_l294_294572


namespace probability_sum_less_than_product_is_5_div_9_l294_294527

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l294_294527


namespace jack_money_proof_l294_294932

variable (sock_cost : ℝ) (number_of_socks : ℕ) (shoe_cost : ℝ) (available_money : ℝ)

def jack_needs_more_money : Prop := 
  (number_of_socks * sock_cost + shoe_cost) - available_money = 71

theorem jack_money_proof (h1 : sock_cost = 9.50)
                         (h2 : number_of_socks = 2)
                         (h3 : shoe_cost = 92)
                         (h4 : available_money = 40) :
  jack_needs_more_money sock_cost number_of_socks shoe_cost available_money :=
by
  unfold jack_needs_more_money
  simp [h1, h2, h3, h4]
  norm_num
  done

end jack_money_proof_l294_294932


namespace total_revenue_l294_294672

theorem total_revenue (chips_sold : ℕ) (chips_price : ℝ) (hotdogs_sold : ℕ) (hotdogs_price : ℝ)
(drinks_sold : ℕ) (drinks_price : ℝ) (sodas_sold : ℕ) (lemonades_sold : ℕ) (sodas_ratio : ℕ)
(lemonades_ratio : ℕ) (h1 : chips_sold = 27) (h2 : chips_price = 1.50) (h3 : hotdogs_sold = chips_sold - 8)
(h4 : hotdogs_price = 3.00) (h5 : drinks_sold = hotdogs_sold + 12) (h6 : drinks_price = 2.00)
(h7 : sodas_ratio = 2) (h8 : lemonades_ratio = 3) (h9 : sodas_sold = (sodas_ratio * drinks_sold) / (sodas_ratio + lemonades_ratio))
(h10 : lemonades_sold = drinks_sold - sodas_sold) :
chips_sold * chips_price + hotdogs_sold * hotdogs_price + drinks_sold * drinks_price = 159.50 := 
by
  -- Proof is left as an exercise for the reader
  sorry

end total_revenue_l294_294672


namespace infinite_sorted_subsequence_l294_294050

theorem infinite_sorted_subsequence : 
  ∀ (warriors : ℕ → ℕ), (∀ n, ∃ m, m > n ∧ warriors m < warriors n) 
  ∨ (∃ k, warriors k = 0) → 
  ∃ (remaining : ℕ → ℕ), (∀ i j, i < j → remaining i > remaining j) :=
by
  intros warriors h
  sorry

end infinite_sorted_subsequence_l294_294050


namespace find_apartment_number_l294_294581

open Nat

def is_apartment_number (x a b : ℕ) : Prop :=
  x = 10 * a + b ∧ x = 17 * b

theorem find_apartment_number : ∃ x a b : ℕ, is_apartment_number x a b ∧ x = 85 :=
by
  sorry

end find_apartment_number_l294_294581


namespace living_room_size_l294_294458

theorem living_room_size :
  let length := 16
  let width := 10
  let total_rooms := 6
  let total_area := length * width
  let unit_size := total_area / total_rooms
  let living_room_size := 3 * unit_size
  living_room_size = 80 := by
    sorry

end living_room_size_l294_294458


namespace find_height_l294_294444

namespace RightTriangleProblem

variables {x h : ℝ}

-- Given the conditions described in the problem
def right_triangle_proportional (a b c : ℝ) : Prop :=
  ∃ (x : ℝ), a = 3 * x ∧ b = 4 * x ∧ c = 5 * x

def hypotenuse (c : ℝ) : Prop := 
  c = 25

def leg (b : ℝ) : Prop :=
  b = 20

-- The theorem stating that the height h of the triangle is 12
theorem find_height (a b c : ℝ) (h : ℝ)
  (H1 : right_triangle_proportional a b c)
  (H2 : hypotenuse c)
  (H3 : leg b) :
  h = 12 :=
by
  sorry

end RightTriangleProblem

end find_height_l294_294444


namespace alex_blueberry_pies_l294_294218

-- Definitions based on given conditions:
def total_pies : ℕ := 30
def ratio (a b c : ℕ) : Prop := (a : ℚ) / b = 2 / 3 ∧ (b : ℚ) / c = 3 / 5

-- Statement to prove the number of blueberry pies
theorem alex_blueberry_pies :
  ∃ (a b c : ℕ), ratio a b c ∧ a + b + c = total_pies ∧ b = 9 :=
by
  sorry

end alex_blueberry_pies_l294_294218


namespace range_of_a_l294_294702

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def increasing_on_negative (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → y ≤ 0 → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (ha : even_function f) (hb : increasing_on_negative f) 
  (hc : ∀ a : ℝ, f a ≤ f (2 - a)) : ∀ a : ℝ, a < 1 → false :=
by
  sorry

end range_of_a_l294_294702


namespace balls_in_boxes_l294_294266

theorem balls_in_boxes (n m : ℕ) (h1 : n = 5) (h2 : m = 4) :
  m^n = 1024 :=
by
  rw [h1, h2]
  exact Nat.pow 4 5 sorry

end balls_in_boxes_l294_294266


namespace min_n_1014_dominoes_l294_294134

theorem min_n_1014_dominoes (n : ℕ) :
  (n + 1) ^ 2 ≥ 6084 → n ≥ 77 :=
sorry

end min_n_1014_dominoes_l294_294134


namespace apples_sold_fresh_l294_294486

-- Definitions per problem conditions
def total_production : Float := 8.0
def initial_percentage_mixed : Float := 0.30
def percentage_increase_per_million : Float := 0.05
def percentage_for_apple_juice : Float := 0.60
def percentage_sold_fresh : Float := 0.40

-- We need to prove that given the conditions, the amount of apples sold fresh is 2.24 million tons
theorem apples_sold_fresh :
  ( (total_production - (initial_percentage_mixed * total_production)) * percentage_sold_fresh = 2.24 ) :=
by
  sorry

end apples_sold_fresh_l294_294486


namespace find_number_l294_294650

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end find_number_l294_294650


namespace earnings_difference_l294_294976

-- We define the price per bottle for each company and the number of bottles sold by each company.
def priceA : ℝ := 4
def priceB : ℝ := 3.5
def quantityA : ℕ := 300
def quantityB : ℕ := 350

-- We define the earnings for each company based on the provided conditions.
def earningsA : ℝ := priceA * quantityA
def earningsB : ℝ := priceB * quantityB

-- We state the theorem that the difference in earnings is $25.
theorem earnings_difference : (earningsB - earningsA) = 25 := by
  -- Proof omitted.
  sorry

end earnings_difference_l294_294976


namespace night_crew_fraction_of_day_l294_294397

variable (D : ℕ) -- Number of workers in the day crew
variable (N : ℕ) -- Number of workers in the night crew
variable (total_boxes : ℕ) -- Total number of boxes loaded by both crews

-- Given conditions
axiom day_fraction : D > 0 ∧ N > 0 ∧ total_boxes > 0
axiom night_workers_fraction : N = (4 * D) / 5
axiom day_crew_boxes_fraction : (5 * total_boxes) / 7 = (5 * D)
axiom night_crew_boxes_fraction : (2 * total_boxes) / 7 = (2 * N)

-- To prove
theorem night_crew_fraction_of_day : 
  let F_d := (5 : ℚ) / (7 * D)
  let F_n := (2 : ℚ) / (7 * N)
  F_n = (5 / 14) * F_d :=
by
  sorry

end night_crew_fraction_of_day_l294_294397


namespace gcd_of_ten_digit_repeated_l294_294747

theorem gcd_of_ten_digit_repeated :
  ∃ k, (∀ n : ℕ, 10000 ≤ n ∧ n < 100000 → Nat.gcd (100001 * n) k = k) ∧ k = 100001 :=
by {
  sorry
}

end gcd_of_ten_digit_repeated_l294_294747


namespace sum_of_integer_n_l294_294549

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end sum_of_integer_n_l294_294549


namespace fuel_tank_capacity_l294_294069

theorem fuel_tank_capacity (x : ℝ) 
  (h1 : (5 / 6) * x - (2 / 3) * x = 15) : x = 90 :=
sorry

end fuel_tank_capacity_l294_294069


namespace sum_of_consecutive_integers_product_2730_eq_42_l294_294844

theorem sum_of_consecutive_integers_product_2730_eq_42 :
  ∃ x : ℤ, x * (x + 1) * (x + 2) = 2730 ∧ x + (x + 1) + (x + 2) = 42 :=
by
  sorry

end sum_of_consecutive_integers_product_2730_eq_42_l294_294844


namespace min_max_of_quadratic_l294_294594

theorem min_max_of_quadratic 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 - 6 * x + 1)
  (h2 : ∀ x, -1 ≤ x ∧ x ≤ 1) : 
  (∃ xmin, ∃ xmax, f xmin = -3 ∧ f xmax = 9 ∧ -1 ≤ xmin ∧ xmin ≤ 1 ∧ -1 ≤ xmax ∧ xmax ≤ 1) :=
sorry

end min_max_of_quadratic_l294_294594


namespace scientific_notation_example_l294_294923

theorem scientific_notation_example : 10500 = 1.05 * 10^4 :=
by
  sorry

end scientific_notation_example_l294_294923


namespace angle_F_after_decrease_l294_294926

theorem angle_F_after_decrease (D E F : ℝ) (h1 : D = 60) (h2 : E = 60) (h3 : F = 60) (h4 : E = D) :
  F - 20 = 40 := by
  simp [h3]
  sorry

end angle_F_after_decrease_l294_294926


namespace count_two_digit_integers_with_remainder_3_l294_294636

theorem count_two_digit_integers_with_remainder_3 :
  {n : ℕ | 10 ≤ 7 * n + 3 ∧ 7 * n + 3 < 100}.card = 13 :=
by
  sorry

end count_two_digit_integers_with_remainder_3_l294_294636


namespace selection_methods_correct_l294_294829

open Finset

-- Defining the set consisting of 10 village officials
def officials := range 10

-- Conditions as given in the problem
def A := 0
def B := 1
def C := 2

-- We are selecting from the remaining 9 since B is not chosen
def officials_without_B := (officials \ {B})

-- Counting the number of ways to choose 3 officials where at least one is A or C
def select_3_with_condition : ℕ :=
  (officials_without_B.choose 3).filter (λ s, A ∈ s ∨ C ∈ s).card

-- The expected answer given the conditions
def expected_answer : ℕ := 49

-- Prove that the counting is correct as per the solution
theorem selection_methods_correct :
  select_3_with_condition = expected_answer :=
sorry

end selection_methods_correct_l294_294829


namespace find_q_l294_294826

noncomputable def p : ℝ := -(5 / 6)
noncomputable def g (x : ℝ) : ℝ := p * x^2 + (5 / 6) * x + 5

theorem find_q :
  (∀ x : ℝ, g x = p * x^2 + q * x + r) ∧ 
  (g (-2) = 0) ∧ 
  (g 3 = 0) ∧ 
  (g 1 = 5) 
  → q = 5 / 6 :=
sorry

end find_q_l294_294826


namespace probability_sum_less_than_product_l294_294542

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l294_294542


namespace complement_of_A_in_U_l294_294427

noncomputable def U : Set ℤ := {x : ℤ | x^2 ≤ 2*x + 3}
def A : Set ℤ := {0, 1, 2}

theorem complement_of_A_in_U : (U \ A) = {-1, 3} :=
by
  sorry

end complement_of_A_in_U_l294_294427


namespace find_y_l294_294203

theorem find_y (x y : ℕ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : ∃ q : ℕ, x = q * y + 9) (h₃ : x / y = 96 + 3 / 20) : y = 60 :=
sorry

end find_y_l294_294203


namespace factor_expression_l294_294236

variable (x : ℤ)

theorem factor_expression : 63 * x - 21 = 21 * (3 * x - 1) := 
by 
  sorry

end factor_expression_l294_294236


namespace basic_computer_price_l294_294040

theorem basic_computer_price :
  ∃ C P : ℝ,
    C + P = 2500 ∧
    (C + 800) + (1 / 5) * (C + 800 + P) = 2500 ∧
    (C + 1100) + (1 / 8) * (C + 1100 + P) = 2500 ∧
    (C + 1500) + (1 / 10) * (C + 1500 + P) = 2500 ∧
    C = 1040 :=
by
  sorry

end basic_computer_price_l294_294040


namespace Union_A_B_eq_l294_294783

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
noncomputable def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem Union_A_B_eq : A ∪ B = {x | -2 < x ∧ x ≤ 4} :=
by
  sorry

end Union_A_B_eq_l294_294783


namespace find_f_1998_l294_294703

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function

theorem find_f_1998 (x : ℝ) (h1 : ∀ x, f (x +1) = f x - 1) (h2 : f 1 = 3997) : f 1998 = 2000 :=
  sorry

end find_f_1998_l294_294703


namespace sin_pi_over_six_l294_294355

theorem sin_pi_over_six : Real.sin (Real.pi / 6) = 1 / 2 := 
by 
  sorry

end sin_pi_over_six_l294_294355


namespace percentage_problem_l294_294886

theorem percentage_problem (X : ℝ) (h : 0.28 * X + 0.45 * 250 = 224.5) : X = 400 :=
sorry

end percentage_problem_l294_294886


namespace part1_part2_l294_294955

-- Given conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Proof statements to be demonstrated
theorem part1 (a : ℝ) : a = 1 := sorry

theorem part2 (f_inv : ℝ → ℝ) : 
  (∀ x : ℝ, x > -1 ∧ x < 1 → f (f_inv x) = x ∧ f_inv (f x) = x) :=
sorry

end part1_part2_l294_294955


namespace children_total_savings_l294_294013

theorem children_total_savings :
  let josiah_savings := 0.25 * 24
  let leah_savings := 0.50 * 20
  let megan_savings := (2 * 0.50) * 12
  josiah_savings + leah_savings + megan_savings = 28 := by
{
  -- lean proof goes here
  sorry
}

end children_total_savings_l294_294013


namespace find_value_l294_294677

noncomputable def roots_of_equation (a b c : ℝ) : Prop :=
  10 * a^3 + 502 * a + 3010 = 0 ∧
  10 * b^3 + 502 * b + 3010 = 0 ∧
  10 * c^3 + 502 * c + 3010 = 0

theorem find_value (a b c : ℝ)
  (h : roots_of_equation a b c) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 903 :=
by
  sorry

end find_value_l294_294677


namespace value_range_for_positive_roots_l294_294849

theorem value_range_for_positive_roots (a : ℝ) :
  (∀ x : ℝ, x > 0 → a * |x| + |x + a| = 0) ↔ (-1 < a ∧ a < 0) :=
by
  sorry

end value_range_for_positive_roots_l294_294849


namespace pairs_condition_l294_294593

theorem pairs_condition (a b : ℕ) (prime_p : ∃ p, p = a^2 + b + 1 ∧ Nat.Prime p)
    (divides : ∀ p, p = a^2 + b + 1 → p ∣ (b^2 - a^3 - 1))
    (not_divides : ∀ p, p = a^2 + b + 1 → ¬ p ∣ (a + b - 1)^2) :
  ∃ x, x ≥ 2 ∧ a = 2 ^ x ∧ b = 2 ^ (2 * x) - 1 := sorry

end pairs_condition_l294_294593


namespace distance_traveled_l294_294291

theorem distance_traveled
  (D : ℝ) (T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 14 * T)
  : D = 50 := sorry

end distance_traveled_l294_294291


namespace cos_value_l294_294913

theorem cos_value (A : ℝ) (h : Real.sin (π + A) = 1/2) : Real.cos (3*π/2 - A) = 1/2 :=
sorry

end cos_value_l294_294913


namespace successfully_served_pizzas_l294_294578

-- Defining the conditions
def total_pizzas_served : ℕ := 9
def pizzas_returned : ℕ := 6

-- Stating the theorem
theorem successfully_served_pizzas :
  total_pizzas_served - pizzas_returned = 3 :=
by
  -- Since this is only the statement, the proof is omitted using sorry
  sorry

end successfully_served_pizzas_l294_294578


namespace max_sum_of_arithmetic_sequence_l294_294698

theorem max_sum_of_arithmetic_sequence 
  (d : ℤ) (a₁ a₃ a₅ a₁₅ : ℤ) (S : ℕ → ℤ)
  (h₁ : d ≠ 0)
  (h₂ : a₃ = a₁ + 2 * d)
  (h₃ : a₅ = a₃ + 2 * d)
  (h₄ : a₁₅ = a₅ + 10 * d)
  (h_geom : a₃ * a₃ = a₅ * a₁₅)
  (h_a₁ : a₁ = 3)
  (h_S : ∀ n, S n = n * a₁ + (n * (n - 1) / 2) * d) :
  ∃ n, S n = 4 :=
by
  sorry

end max_sum_of_arithmetic_sequence_l294_294698


namespace problem_1_problem_2_l294_294753

-- Problem 1 Lean statement
theorem problem_1 :
  (1 - 1^4 - (1/2) * (3 - (-3)^2)) = 2 :=
by sorry

-- Problem 2 Lean statement
theorem problem_2 :
  ((3/8 - 1/6 - 3/4) * 24) = -13 :=
by sorry

end problem_1_problem_2_l294_294753


namespace max_n_for_factorable_polynomial_l294_294239

theorem max_n_for_factorable_polynomial :
  ∃ A B : ℤ, AB = 144 ∧ (A + 6 * B = 865) :=
begin
  sorry
end

end max_n_for_factorable_polynomial_l294_294239


namespace f_at_3_l294_294121

variable {R : Type} [LinearOrderedField R]

-- Define odd function
def is_odd_function (f : R → R) := ∀ x : R, f (-x) = -f x

-- Define the given function f and its properties
variables (f : R → R)
  (h_odd : is_odd_function f)
  (h_domain : ∀ x : R, true) -- domain is R implicitly
  (h_eq : ∀ x : R, f x + f (2 - x) = 4)

-- Prove that f(3) = 6
theorem f_at_3 : f 3 = 6 :=
  sorry

end f_at_3_l294_294121


namespace total_cost_paper_plates_and_cups_l294_294505

theorem total_cost_paper_plates_and_cups :
  ∀ (P C : ℝ), (20 * P + 40 * C = 1.20) → (100 * P + 200 * C = 6.00) := by
  intros P C h
  sorry

end total_cost_paper_plates_and_cups_l294_294505


namespace probability_sum_less_than_product_l294_294545

theorem probability_sum_less_than_product:
  let S := {x | x ∈ Finset.range 7 ∧ x ≠ 0} in
  (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a*b > a+b) →
  (Finset.card (Finset.filter (λ x : ℕ × ℕ, (x.1 * x.2 > x.1 + x.2)) (Finset.product S S))) =
  18 →
  Finset.card (Finset.product S S) = 36 →
  18 / 36 = 1 / 2 :=
by
  sorry

end probability_sum_less_than_product_l294_294545


namespace chromosomal_variations_l294_294861

-- Define the conditions
def condition1 := "Plants grown from anther culture in vitro."
def condition2 := "Addition or deletion of DNA base pairs on chromosomes."
def condition3 := "Free combination of non-homologous chromosomes."
def condition4 := "Crossing over between non-sister chromatids in a tetrad."
def condition5 := "Cells of a patient with Down syndrome have three copies of chromosome 21."

-- Define a concept of belonging to chromosomal variations
def belongs_to_chromosomal_variations (condition: String) : Prop :=
  condition = condition1 ∨ condition = condition5

-- State the theorem
theorem chromosomal_variations :
  belongs_to_chromosomal_variations condition1 ∧ 
  belongs_to_chromosomal_variations condition5 ∧ 
  ¬ (belongs_to_chromosomal_variations condition2 ∨ 
     belongs_to_chromosomal_variations condition3 ∨ 
     belongs_to_chromosomal_variations condition4) :=
by
  sorry

end chromosomal_variations_l294_294861


namespace find_triples_l294_294410

theorem find_triples (x n p : ℕ) (hp : Nat.Prime p) 
  (hx_pos : x > 0) (hn_pos : n > 0) : 
  x^3 + 3 * x + 14 = 2 * p^n → (x = 1 ∧ n = 2 ∧ p = 3) ∨ (x = 3 ∧ n = 2 ∧ p = 5) :=
by 
  sorry

end find_triples_l294_294410


namespace s_mores_graham_crackers_l294_294006

def graham_crackers_per_smore (total_graham_crackers total_marshmallows : ℕ) : ℕ :=
total_graham_crackers / total_marshmallows

theorem s_mores_graham_crackers :
  let total_graham_crackers := 48
  let available_marshmallows := 6
  let additional_marshmallows := 18
  let total_marshmallows := available_marshmallows + additional_marshmallows
  graham_crackers_per_smore total_graham_crackers total_marshallows = 2 := sorry

end s_mores_graham_crackers_l294_294006


namespace sum_of_ages_l294_294487

theorem sum_of_ages (y : ℕ) 
  (h_diff : 38 - y = 2) : y + 38 = 74 := 
by {
  sorry
}

end sum_of_ages_l294_294487


namespace min_value_l294_294915

theorem min_value (x : ℝ) (h : x > 1) : x + 4 / (x - 1) ≥ 5 :=
sorry

end min_value_l294_294915


namespace workbooks_needed_l294_294079

theorem workbooks_needed (classes : ℕ) (workbooks_per_class : ℕ) (spare_workbooks : ℕ) (total_workbooks : ℕ) :
  classes = 25 → workbooks_per_class = 144 → spare_workbooks = 80 → total_workbooks = 25 * 144 + 80 → 
  total_workbooks = classes * workbooks_per_class + spare_workbooks :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  exact h4

end workbooks_needed_l294_294079


namespace jessica_blueberry_pies_l294_294938

theorem jessica_blueberry_pies 
  (total_pies : ℕ)
  (ratio_apple : ℕ)
  (ratio_blueberry : ℕ)
  (ratio_cherry : ℕ)
  (h_total : total_pies = 36)
  (h_ratios : ratio_apple = 2)
  (h_ratios_b : ratio_blueberry = 5)
  (h_ratios_c : ratio_cherry = 3) : 
  total_pies * ratio_blueberry / (ratio_apple + ratio_blueberry + ratio_cherry) = 18 := 
by
  sorry

end jessica_blueberry_pies_l294_294938


namespace abcd_product_l294_294154

noncomputable def A := (Real.sqrt 3000 + Real.sqrt 3001)
noncomputable def B := (-Real.sqrt 3000 - Real.sqrt 3001)
noncomputable def C := (Real.sqrt 3000 - Real.sqrt 3001)
noncomputable def D := (Real.sqrt 3001 - Real.sqrt 3000)

theorem abcd_product :
  A * B * C * D = -1 :=
by
  sorry

end abcd_product_l294_294154


namespace total_bowling_balls_is_66_l294_294182

-- Define the given conditions
def red_bowling_balls := 30
def difference_green_red := 6
def green_bowling_balls := red_bowling_balls + difference_green_red

-- The statement to prove
theorem total_bowling_balls_is_66 :
  red_bowling_balls + green_bowling_balls = 66 := by
  sorry

end total_bowling_balls_is_66_l294_294182


namespace num_rem_three_by_seven_l294_294621

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l294_294621


namespace count_interesting_numbers_l294_294629

-- Define the condition of leaving a remainder of 3 when divided by 7
def leaves_remainder_3 (n : ℕ) : Prop :=
  n % 7 = 3

-- Define the condition of being a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Combine the conditions to define the numbers we are interested in
def interesting_numbers (n : ℕ) : Prop :=
  is_two_digit n ∧ leaves_remainder_3 n

-- The main theorem: the count of such numbers
theorem count_interesting_numbers : 
  (finset.filter interesting_numbers (finset.Icc 10 99)).card = 13 :=
by
  sorry

end count_interesting_numbers_l294_294629


namespace class_average_weight_l294_294851

theorem class_average_weight (n_A n_B : ℕ) (w_A w_B : ℝ) (h1 : n_A = 50) (h2 : n_B = 40) (h3 : w_A = 50) (h4 : w_B = 70) :
  (n_A * w_A + n_B * w_B) / (n_A + n_B) = 58.89 :=
by
  sorry

end class_average_weight_l294_294851


namespace minimum_value_ineq_l294_294158

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  (1 : ℝ) ≤ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by {
  sorry
}

end minimum_value_ineq_l294_294158


namespace num_square_tiles_is_zero_l294_294869

def triangular_tiles : ℕ := sorry
def square_tiles : ℕ := sorry
def hexagonal_tiles : ℕ := sorry

axiom tile_count_eq : triangular_tiles + square_tiles + hexagonal_tiles = 30
axiom edge_count_eq : 3 * triangular_tiles + 4 * square_tiles + 6 * hexagonal_tiles = 120

theorem num_square_tiles_is_zero : square_tiles = 0 :=
by
  sorry

end num_square_tiles_is_zero_l294_294869


namespace pages_needed_l294_294377

def new_cards : ℕ := 2
def old_cards : ℕ := 10
def cards_per_page : ℕ := 3
def total_cards : ℕ := new_cards + old_cards

theorem pages_needed : total_cards / cards_per_page = 4 := by
  sorry

end pages_needed_l294_294377


namespace football_team_total_players_l294_294044

variable (P : ℕ)
variable (throwers : ℕ := 52)
variable (total_right_handed : ℕ := 64)
variable (remaining := P - throwers)
variable (left_handed := remaining / 3)
variable (right_handed_non_throwers := 2 * remaining / 3)

theorem football_team_total_players:
  right_handed_non_throwers + throwers = total_right_handed →
  P = 70 :=
by
  sorry

end football_team_total_players_l294_294044


namespace ways_to_distribute_balls_in_boxes_l294_294277

theorem ways_to_distribute_balls_in_boxes :
  ∃ (num_ways : ℕ), num_ways = 4 ^ 5 := sorry

end ways_to_distribute_balls_in_boxes_l294_294277


namespace inequality_solution_set_l294_294340

theorem inequality_solution_set (a : ℝ) : 
    (a = 0 → (∃ x : ℝ, x > 1 ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a < 0 → (∃ x : ℝ, (x < 2/a ∨ x > 1) ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (0 < a ∧ a < 2 → (∃ x : ℝ, (1 < x ∧ x < 2/a) ∧ ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a = 2 → ¬(∃ x : ℝ, ax^2 - (a + 2) * x + 2 < 0)) ∧
    (a > 2 → (∃ x : ℝ, (2/a < x ∧ x < 1) ∧ ax^2 - (a + 2) * x + 2 < 0)) :=
by sorry

end inequality_solution_set_l294_294340


namespace find_number_l294_294652

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end find_number_l294_294652


namespace probability_sum_less_than_product_is_5_div_9_l294_294529

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l294_294529


namespace angle_A_is_60_degrees_value_of_b_plus_c_l294_294922

noncomputable def triangleABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  let area := (3 * Real.sqrt 3) / 2
  c + 2 * a * Real.cos C = 2 * b ∧
  1/2 * b * c * Real.sin A = area 

theorem angle_A_is_60_degrees (A B C : ℝ) (a b c : ℝ) :
  triangleABC A B C a b c →
  Real.cos A = 1 / 2 → 
  A = 60 :=
by
  intros h1 h2 
  sorry

theorem value_of_b_plus_c (A B C : ℝ) (b c : ℝ) :
  triangleABC A B C (Real.sqrt 7) b c →
  b * c = 6 →
  (b + c) = 5 :=
by 
  intros h1 h2 
  sorry

end angle_A_is_60_degrees_value_of_b_plus_c_l294_294922


namespace bianca_initial_cupcakes_l294_294414

theorem bianca_initial_cupcakes (X : ℕ) (h : X - 6 + 17 = 25) : X = 14 := by
  sorry

end bianca_initial_cupcakes_l294_294414


namespace largest_multiple_of_8_less_than_100_l294_294998

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, n < 100 ∧ (∃ k : ℕ, n = 8 * k) ∧ ∀ m : ℕ, m < 100 ∧ (∃ k : ℕ, m = 8 * k) → m ≤ 96 := 
by
  sorry

end largest_multiple_of_8_less_than_100_l294_294998


namespace find_a_l294_294800

theorem find_a (a : ℝ) (b : ℝ) :
  (9 * x^2 - 27 * x + a = (3 * x + b)^2) → b = -4.5 → a = 20.25 := 
by sorry

end find_a_l294_294800


namespace intervals_union_l294_294940

open Set

noncomputable def I (a b : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < b}

theorem intervals_union {I1 I2 I3 : Set ℝ} (h1 : ∃ (a1 b1 : ℝ), I1 = I a1 b1)
  (h2 : ∃ (a2 b2 : ℝ), I2 = I a2 b2) (h3 : ∃ (a3 b3 : ℝ), I3 = I a3 b3)
  (h_non_empty : (I1 ∩ I2 ∩ I3).Nonempty) (h_not_contained : ¬ (I1 ⊆ I2) ∧ ¬ (I1 ⊆ I3) ∧ ¬ (I2 ⊆ I1) ∧ ¬ (I2 ⊆ I3) ∧ ¬ (I3 ⊆ I1) ∧ ¬ (I3 ⊆ I2)) :
  I1 ⊆ (I2 ∪ I3) ∨ I2 ⊆ (I1 ∪ I3) ∨ I3 ⊆ (I1 ∪ I2) :=
sorry

end intervals_union_l294_294940


namespace find_distance_l294_294719

-- Conditions: total cost, base price, cost per mile
variables (total_cost base_price cost_per_mile : ℕ)

-- Definition of the distance as per the problem
def distance_from_home_to_hospital (total_cost base_price cost_per_mile : ℕ) : ℕ :=
  (total_cost - base_price) / cost_per_mile

-- Given values:
def total_cost_value : ℕ := 23
def base_price_value : ℕ := 3
def cost_per_mile_value : ℕ := 4

-- The theorem that encapsulates the problem statement
theorem find_distance :
  distance_from_home_to_hospital total_cost_value base_price_value cost_per_mile_value = 5 :=
by
  -- Placeholder for the proof
  sorry

end find_distance_l294_294719


namespace solve_equation_real_l294_294831

theorem solve_equation_real (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 2) * (x - 1) / ((x - 4) * (x - 2) * (x - 1)) = 1 ↔
  x = (9 + Real.sqrt 5) / 2 ∨ x = (9 - Real.sqrt 5) / 2 :=
by  
  sorry

end solve_equation_real_l294_294831


namespace quadratic_inequality_solution_set_l294_294774

theorem quadratic_inequality_solution_set (a b c : ℝ) : 
  (∀ x : ℝ, - (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) := 
by
  sorry

end quadratic_inequality_solution_set_l294_294774


namespace tan_arccos_eq_2y_l294_294508

noncomputable def y_squared : ℝ :=
  (-1 + Real.sqrt 17) / 8

theorem tan_arccos_eq_2y (y : ℝ) (hy : 0 < y) (htan : Real.tan (Real.arccos y) = 2 * y) :
  y^2 = y_squared := sorry

end tan_arccos_eq_2y_l294_294508


namespace p_lim_p20_minus_p15_l294_294964

noncomputable def p : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 2) * p n + (1 / 2) * p (n - 1)

theorem p_lim (n : ℕ) : filter.tendsto p filter.at_top (nhds 1) :=
sorry

theorem p20_minus_p15 : p 20 - p 15 = 0 :=
begin
  have h_lim := p_lim,
  sorry -- Proof that p 20 - p 15 = 0
end

end p_lim_p20_minus_p15_l294_294964


namespace James_pays_6_dollars_l294_294007

-- Defining the conditions
def packs : ℕ := 4
def stickers_per_pack : ℕ := 30
def cost_per_sticker : ℚ := 0.10
def friend_share : ℚ := 0.5

-- Total number of stickers
def total_stickers : ℕ := packs * stickers_per_pack

-- Total cost calculation
def total_cost : ℚ := total_stickers * cost_per_sticker

-- James' payment calculation
def james_payment : ℚ := total_cost * friend_share

-- Theorem statement to be proven
theorem James_pays_6_dollars : james_payment = 6 := by
  sorry

end James_pays_6_dollars_l294_294007


namespace rebecca_gemstones_needed_l294_294471

-- Definitions for the conditions
def magnets_per_earring : Nat := 2
def buttons_per_magnet : Nat := 1 / 2
def gemstones_per_button : Nat := 3
def earrings_per_set : Nat := 2
def sets : Nat := 4

-- Statement to be proved
theorem rebecca_gemstones_needed : 
  gemstones_per_button * (buttons_per_magnet * (magnets_per_earring * (earrings_per_set * sets))) = 24 :=
by
  sorry

end rebecca_gemstones_needed_l294_294471


namespace problem1_max_min_value_problem2_min_value_exists_l294_294906

noncomputable section

-- Helper definitions to express the functions involved
def f1 (x : ℝ) : ℝ := -x^2 + 3 * x - Real.log x
def f2 (b : ℝ) (x : ℝ) : ℝ := b * x - Real.log x

-- Problem 1
theorem problem1_max_min_value :
  (∀ x ∈ Set.Icc (1/2) 2, x > 0) ∧
  (∀ x ∈ Set.Icc (1/2) 2, f1 x ≤ 2) ∧
  (2 = f1 1) ∧
  (Real.log 2 + 5/4 = f1 (1/2)) := by
  sorry

-- Problem 2
theorem problem2_min_value_exists :
  ∃ (b : ℝ) (hb : 0 < b), b = Real.exp 2 ∧
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.exp 1), f2 b x ≥ 3) ∧
  (f2 b (Real.exp 1) = 3) := by
  sorry

end problem1_max_min_value_problem2_min_value_exists_l294_294906


namespace cube_splitting_height_l294_294921

/-- If we split a cube with an edge of 1 meter into small cubes with an edge of 1 millimeter,
what will be the height of a column formed by stacking all the small cubes one on top of another? -/
theorem cube_splitting_height :
  let edge_meter := 1
  let edge_mm := 1000
  let num_cubes := (edge_meter * edge_mm) ^ 3
  let height_mm := num_cubes * edge_mm
  let height_km := height_mm / (1000 * 1000 * 1000)
  height_km = 1000 :=
by
  sorry

end cube_splitting_height_l294_294921


namespace decrease_of_negative_five_l294_294302

-- Definition: Positive and negative numbers as explained
def increase (n: ℤ) : Prop := n > 0
def decrease (n: ℤ) : Prop := n < 0

-- Conditions
def condition : Prop := increase 17

-- Theorem stating the solution
theorem decrease_of_negative_five (h : condition) : decrease (-5) ∧ -5 = -5 :=
by
  sorry

end decrease_of_negative_five_l294_294302


namespace geometric_sequence_sum_l294_294420

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h₀ : q > 1)
  (h₁ : ∀ n : ℕ, a (n + 1) = a n * q)
  (h₂ : ∀ x : ℝ, 4 * x^2 - 8 * x + 3 = 0 → (x = a 2005 ∨ x = a 2006)) : 
  a 2007 + a 2008 = 18 := 
sorry

end geometric_sequence_sum_l294_294420


namespace count_perf_squares_less_than_20000_l294_294405

theorem count_perf_squares_less_than_20000 :
  ∃ (n : ℕ), (n = 35) ∧
  ∀ (a : ℕ), (∃ (b : ℕ), a^2 = (2 * b + 2)^2 - (2 * b)^2) → a^2 < 20000 :=
begin
  use 35,
  split,
  { refl },
  intros a h,
  obtain ⟨b, h1⟩ := h,
  sorry
end

end count_perf_squares_less_than_20000_l294_294405


namespace intersection_M_S_l294_294425

def M := {x : ℕ | 0 < x ∧ x < 4 }

def S : Set ℕ := {2, 3, 5}

theorem intersection_M_S : (M ∩ S) = {2, 3} := by
  sorry

end intersection_M_S_l294_294425


namespace black_king_eventually_in_check_l294_294570

theorem black_king_eventually_in_check 
  (n : ℕ) (h1 : n = 1000) (r : ℕ) (h2 : r = 499)
  (rooks : Fin r → (ℕ × ℕ)) (king : ℕ × ℕ)
  (take_not_allowed : ∀ rk : Fin r, rooks rk ≠ king) :
  ∃ m : ℕ, m ≤ 1000 ∧ (∃ t : Fin r, rooks t = king) :=
by
  sorry

end black_king_eventually_in_check_l294_294570


namespace total_bowling_balls_is_66_l294_294181

-- Define the given conditions
def red_bowling_balls := 30
def difference_green_red := 6
def green_bowling_balls := red_bowling_balls + difference_green_red

-- The statement to prove
theorem total_bowling_balls_is_66 :
  red_bowling_balls + green_bowling_balls = 66 := by
  sorry

end total_bowling_balls_is_66_l294_294181


namespace comparison_of_large_exponents_l294_294862

theorem comparison_of_large_exponents : 2^1997 > 5^850 := sorry

end comparison_of_large_exponents_l294_294862


namespace lines_perpendicular_to_same_plane_are_parallel_l294_294252

variables {Point Line Plane : Type*}
variables [MetricSpace Point] [LinearOrder Line]

def line_parallel_to_plane (a : Line) (M : Plane) : Prop := sorry -- Define the formal condition
def line_perpendicular_to_plane (a : Line) (M : Plane) : Prop := sorry -- Define the formal condition
def lines_parallel (a b : Line) : Prop := sorry -- Define the formal condition

theorem lines_perpendicular_to_same_plane_are_parallel 
  (a b : Line) (M : Plane) 
  (h₁ : line_perpendicular_to_plane a M) 
  (h₂ : line_perpendicular_to_plane b M) : 
  lines_parallel a b :=
sorry

end lines_perpendicular_to_same_plane_are_parallel_l294_294252


namespace cost_of_each_skirt_l294_294099

-- Problem definitions based on conditions
def cost_of_art_supplies : ℕ := 20
def total_expenditure : ℕ := 50
def number_of_skirts : ℕ := 2

-- Proving the cost of each skirt
theorem cost_of_each_skirt (cost_of_each_skirt : ℕ) : 
  number_of_skirts * cost_of_each_skirt + cost_of_art_supplies = total_expenditure → 
  cost_of_each_skirt = 15 := 
by 
  sorry

end cost_of_each_skirt_l294_294099


namespace magnitude_of_complex_l294_294802

variable (z : ℂ)
variable (h : Complex.I * z = 3 - 4 * Complex.I)

theorem magnitude_of_complex :
  Complex.abs z = 5 :=
by
  sorry

end magnitude_of_complex_l294_294802


namespace rod_length_l294_294895

/--
Prove that given the number of pieces that can be cut from the rod is 40 and the length of each piece is 85 cm, the length of the rod is 3400 cm.
-/
theorem rod_length (number_of_pieces : ℕ) (length_of_each_piece : ℕ) (h_pieces : number_of_pieces = 40) (h_length_piece : length_of_each_piece = 85) : number_of_pieces * length_of_each_piece = 3400 := 
by
  -- We need to prove that 40 * 85 = 3400
  sorry

end rod_length_l294_294895


namespace bush_height_at_2_years_l294_294383

theorem bush_height_at_2_years (H: ℕ → ℕ) 
  (quadruple_height: ∀ (n: ℕ), H (n+1) = 4 * H n)
  (H_4: H 4 = 64) : H 2 = 4 :=
by
  sorry

end bush_height_at_2_years_l294_294383


namespace log_101600_l294_294370

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_101600 (h : log_base_10 102 = 0.3010) : log_base_10 101600 = 2.3010 :=
by
  sorry

end log_101600_l294_294370


namespace molecular_weight_one_mole_l294_294362

theorem molecular_weight_one_mole {compound : Type} (moles : ℕ) (total_weight : ℝ) 
  (h_moles : moles = 5) (h_total_weight : total_weight = 490) :
  total_weight / moles = 98 := 
by {
    rw [h_moles, h_total_weight],
    norm_num,
    sorry
  }

end molecular_weight_one_mole_l294_294362


namespace find_m_l294_294787

variable (m x1 x2 : ℝ)

def quadratic_eqn (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m * x + 2 * m - 1 = 0

def roots_condition (m x1 x2 : ℝ) : Prop :=
  x1^2 + x2^2 = 23 ∧
  x1 + x2 = m ∧
  x1 * x2 = 2 * m - 1

theorem find_m (m x1 x2 : ℝ) : 
  quadratic_eqn m → 
  roots_condition m x1 x2 → 
  m = -3 :=
by
  intro hQ hR
  sorry

end find_m_l294_294787


namespace triangle_lattice_points_l294_294027

-- Given lengths of the legs of the right triangle
def DE : Nat := 15
def EF : Nat := 20

-- Calculate the hypotenuse using the Pythagorean theorem
def DF : Nat := Nat.sqrt (DE ^ 2 + EF ^ 2)

-- Calculate the area of the triangle
def Area : Nat := (DE * EF) / 2

-- Calculate the number of boundary points
def B : Nat :=
  let points_DE := DE + 1
  let points_EF := EF + 1
  let points_DF := DF + 1
  points_DE + points_EF + points_DF - 3

-- Calculate the number of interior points using Pick's Theorem
def I : Int := Area - (B / 2 - 1)

-- Calculate the total number of lattice points
def total_lattice_points : Int := I + Int.ofNat B

-- The theorem statement
theorem triangle_lattice_points : total_lattice_points = 181 := by
  -- The actual proof goes here
  sorry

end triangle_lattice_points_l294_294027


namespace determinant_matrix_example_l294_294091

open Matrix

def matrix_example : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -2], ![-3, 6]]

noncomputable def compute_det_and_add_5 : ℤ := (matrix_example.det) + 5

theorem determinant_matrix_example :
  compute_det_and_add_5 = 41 := by
  sorry

end determinant_matrix_example_l294_294091


namespace total_number_of_athletes_l294_294398

theorem total_number_of_athletes (M F x : ℕ) (r1 r2 r3 : ℕ×ℕ) (H1 : r1 = (19, 12)) (H2 : r2 = (20, 13)) (H3 : r3 = (30, 19))
  (initial_males : M = 380 * x) (initial_females : F = 240 * x)
  (males_after_gym : M' = 390 * x) (females_after_gym : F' = 247 * x)
  (conditions : (M' - M) - (F' - F) = 30) : M' + F' = 6370 :=
by
  sorry

end total_number_of_athletes_l294_294398


namespace problem_statement_l294_294897

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (x : ℝ) : ℝ := (2 ^ x) / 2 - 2 / (2 ^ x) - x + 1

theorem problem_statement (a : ℝ) (x₁ x₂ : ℝ) (h₀ : x₁ < x₂)
  (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) : g x₁ + g x₂ > 0 :=
sorry

end problem_statement_l294_294897


namespace pigeons_among_non_sparrows_l294_294750

theorem pigeons_among_non_sparrows (P_total P_parrots P_peacocks P_sparrows : ℝ)
    (h1 : P_total = 20)
    (h2 : P_parrots = 30)
    (h3 : P_peacocks = 15)
    (h4 : P_sparrows = 35) :
    (P_total / (100 - P_sparrows)) * 100 = 30.77 :=
by
  -- Proof will be provided here
  sorry

end pigeons_among_non_sparrows_l294_294750


namespace expression_evaluation_l294_294729

theorem expression_evaluation : 
  3 / 5 * ((2 / 3 + 3 / 8) / 2) - 1 / 16 = 1 / 4 := 
by
  sorry

end expression_evaluation_l294_294729


namespace solve_quadratic_eq_l294_294845

theorem solve_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := sorry

end solve_quadratic_eq_l294_294845


namespace balls_in_boxes_l294_294285

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l294_294285


namespace prob_draw_correct_l294_294254

-- Given conditions
def prob_A_wins : ℝ := 0.40
def prob_A_not_lose : ℝ := 0.90

-- Definition to be proved
def prob_draw : ℝ := prob_A_not_lose - prob_A_wins

theorem prob_draw_correct : prob_draw = 0.50 := by
  sorry

end prob_draw_correct_l294_294254


namespace smallest_angle_in_scalene_triangle_l294_294391

theorem smallest_angle_in_scalene_triangle :
  ∃ (triangle : Type) (a b c : ℝ),
    ∀ (A B C : triangle),
      a = 162 ∧
      b / c = 3 / 4 ∧
      a + b + c = 180 ∧
      a ≠ b ∧ a ≠ c ∧ b ≠ c ->
        min b c = 7.7 :=
sorry

end smallest_angle_in_scalene_triangle_l294_294391


namespace mean_height_calc_l294_294972

/-- Heights of players on the soccer team -/
def heights : List ℕ := [47, 48, 50, 50, 54, 55, 57, 59, 63, 63, 64, 65]

/-- Total number of players -/
def total_players : ℕ := heights.length

/-- Sum of heights of players -/
def sum_heights : ℕ := heights.sum

/-- Mean height of players on the soccer team -/
def mean_height : ℚ := sum_heights / total_players

/-- Proof that the mean height is correct -/
theorem mean_height_calc : mean_height = 56.25 := by
  sorry

end mean_height_calc_l294_294972


namespace probability_sum_less_than_product_l294_294534

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l294_294534


namespace rob_final_value_in_euros_l294_294167

noncomputable def initial_value_in_usd : ℝ := 
  (7 * 0.25) + (3 * 0.10) + (5 * 0.05) + (12 * 0.01) + (3 * 0.50) + (2 * 1.00)

noncomputable def value_after_losing_coins : ℝ := 
  (6 * 0.25) + (2 * 0.10) + (4 * 0.05) + (11 * 0.01) + (2 * 0.50) + (1 * 1.00)

noncomputable def value_after_first_exchange : ℝ :=
  (6 * 0.25) + (4 * 0.10) + (1 * 0.05) + (11 * 0.01) + (2 * 0.50) + (1 * 1.00)

noncomputable def value_after_second_exchange : ℝ :=
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (11 * 0.01) + (1 * 0.50) + (1 * 1.00)

noncomputable def value_after_third_exchange : ℝ :=
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (61 * 0.01) + (1 * 0.50)

noncomputable def final_value_in_usd : ℝ := 
  (7 * 0.25) + (6 * 0.10) + (1 * 0.05) + (61 * 0.01) + (1 * 0.50)

noncomputable def exchange_rate_usd_to_eur : ℝ := 0.85

noncomputable def final_value_in_eur : ℝ :=
  final_value_in_usd * exchange_rate_usd_to_eur

theorem rob_final_value_in_euros : final_value_in_eur = 2.9835 := by
  sorry

end rob_final_value_in_euros_l294_294167


namespace quadratic_two_distinct_real_roots_l294_294106

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (k - 1 ≠ 0 ∧ 8 - 4 * k > 0) ↔ (k < 2 ∧ k ≠ 1) := 
by
  sorry

end quadratic_two_distinct_real_roots_l294_294106


namespace no_person_has_fewer_than_6_cards_l294_294129

-- Definition of the problem and conditions
def cards := 60
def people := 10
def cards_per_person := cards / people

-- Lean statement of the proof problem
theorem no_person_has_fewer_than_6_cards
  (cards_dealt : cards = 60)
  (people_count : people = 10)
  (even_distribution : cards % people = 0) :
  ∀ person, person < people → cards_per_person = 6 ∧ person < people → person = 0 := 
by 
  sorry

end no_person_has_fewer_than_6_cards_l294_294129


namespace bisection_method_example_l294_294368

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 4

theorem bisection_method_example :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0) →
  (∃ x : ℝ, (1 / 2) < x ∧ x < 1 ∧ f x = 0) :=
by
  sorry

end bisection_method_example_l294_294368


namespace earnings_difference_l294_294981

-- Define the prices and quantities.
def price_A : ℝ := 4
def price_B : ℝ := 3.5
def quantity_A : ℕ := 300
def quantity_B : ℕ := 350

-- Define the earnings for both companies.
def earnings_A := price_A * quantity_A
def earnings_B := price_B * quantity_B

-- State the theorem we intend to prove.
theorem earnings_difference :
  earnings_B - earnings_A = 25 := by
  sorry

end earnings_difference_l294_294981


namespace max_sum_of_factors_of_1764_l294_294431

theorem max_sum_of_factors_of_1764 :
  ∃ (a b : ℕ), a * b = 1764 ∧ a + b = 884 :=
by
  sorry

end max_sum_of_factors_of_1764_l294_294431


namespace sum_of_n_values_such_that_fraction_is_integer_l294_294554

theorem sum_of_n_values_such_that_fraction_is_integer : 
  let is_odd (d : ℤ) : Prop := d % 2 ≠ 0
  let divisors (n : ℤ) := ∃ d : ℤ, d ∣ n
  let a_values := { n : ℤ | ∃ (d : ℤ), divisors 36 ∧ is_odd d ∧ 2 * n - 1 = d }
  let a_sum := ∑ n in a_values, n
  a_sum = 8 := 
by
  sorry

end sum_of_n_values_such_that_fraction_is_integer_l294_294554


namespace distance_from_yz_plane_l294_294700

theorem distance_from_yz_plane (x z : ℝ) : 
  (abs (-6) = (abs x) / 2) → abs x = 12 :=
by
  sorry

end distance_from_yz_plane_l294_294700


namespace probability_sum_less_than_product_l294_294518

theorem probability_sum_less_than_product :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      conditioned_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 * p.2 > p.1 + p.2},
      total_pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S} in
  (conditioned_pairs.to_finset.card : ℚ) / total_pairs.to_finset.card = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l294_294518


namespace smallest_X_divisible_by_60_l294_294942

/-
  Let \( T \) be a positive integer consisting solely of 0s and 1s.
  If \( X = \frac{T}{60} \) and \( X \) is an integer, prove that the smallest possible value of \( X \) is 185.
-/
theorem smallest_X_divisible_by_60 (T X : ℕ) 
  (hT_digit : ∀ d, d ∈ T.digits 10 → d = 0 ∨ d = 1) 
  (h1 : X = T / 60) 
  (h2 : T % 60 = 0) : 
  X = 185 :=
sorry

end smallest_X_divisible_by_60_l294_294942


namespace jesse_bananas_l294_294937

def number_of_bananas_shared (friends : ℕ) (bananas_per_friend : ℕ) : ℕ :=
  friends * bananas_per_friend

theorem jesse_bananas :
  number_of_bananas_shared 3 7 = 21 :=
by
  sorry

end jesse_bananas_l294_294937


namespace mika_saucer_surface_area_l294_294459

noncomputable def surface_area_saucer (r h rim_thickness : ℝ) : ℝ :=
  let A_cap := 2 * Real.pi * r * h  -- Surface area of the spherical cap
  let R_outer := r
  let R_inner := r - rim_thickness
  let A_rim := Real.pi * (R_outer^2 - R_inner^2)  -- Area of the rim
  A_cap + A_rim

theorem mika_saucer_surface_area :
  surface_area_saucer 3 1.5 1 = 14 * Real.pi :=
sorry

end mika_saucer_surface_area_l294_294459


namespace rod_total_length_l294_294390

theorem rod_total_length
  (n : ℕ) (l : ℝ)
  (h₁ : n = 50)
  (h₂ : l = 0.85) :
  n * l = 42.5 := by
  sorry

end rod_total_length_l294_294390


namespace average_rainfall_virginia_l294_294504

noncomputable def average_rainfall : ℝ :=
  (3.79 + 4.5 + 3.95 + 3.09 + 4.67) / 5

theorem average_rainfall_virginia : average_rainfall = 4 :=
by
  sorry

end average_rainfall_virginia_l294_294504


namespace max_sum_a_b_c_d_e_f_g_l294_294349

theorem max_sum_a_b_c_d_e_f_g (a b c d e f g : ℕ)
  (h1 : a + b + c = 2)
  (h2 : b + c + d = 2)
  (h3 : c + d + e = 2)
  (h4 : d + e + f = 2)
  (h5 : e + f + g = 2) :
  a + b + c + d + e + f + g ≤ 6 := 
sorry

end max_sum_a_b_c_d_e_f_g_l294_294349


namespace mountaineering_team_problem_l294_294336

structure Climber :=
  (total_students : ℕ)
  (advanced_climbers : ℕ)
  (intermediate_climbers : ℕ)
  (beginners : ℕ)

structure Experience :=
  (advanced_points : ℕ)
  (intermediate_points : ℕ)
  (beginner_points : ℕ)

structure TeamComposition :=
  (advanced_needed : ℕ)
  (intermediate_needed : ℕ)
  (beginners_needed : ℕ)
  (max_experience : ℕ)

def team_count (students : Climber) (xp : Experience) (comp : TeamComposition) : ℕ :=
  let total_experience := comp.advanced_needed * xp.advanced_points +
                          comp.intermediate_needed * xp.intermediate_points +
                          comp.beginners_needed * xp.beginner_points
  let max_teams_from_advanced := students.advanced_climbers / comp.advanced_needed
  let max_teams_from_intermediate := students.intermediate_climbers / comp.intermediate_needed
  let max_teams_from_beginners := students.beginners / comp.beginners_needed
  if total_experience ≤ comp.max_experience then
    min (max_teams_from_advanced) $ min (max_teams_from_intermediate) (max_teams_from_beginners)
  else 0

def problem : Prop :=
  team_count
    ⟨172, 45, 70, 57⟩
    ⟨80, 50, 30⟩
    ⟨5, 8, 5, 1000⟩ = 8

-- Let's declare the theorem now:
theorem mountaineering_team_problem : problem := sorry

end mountaineering_team_problem_l294_294336


namespace timeAfter2687Minutes_l294_294699

-- We define a structure for representing time in hours and minutes.
structure Time :=
  (hour : Nat)
  (minute : Nat)

-- Define the current time
def currentTime : Time := {hour := 7, minute := 0}

-- Define a function that computes the time after adding a given number of minutes to a given time
noncomputable def addMinutes (t : Time) (minutesToAdd : Nat) : Time :=
  let totalMinutes := t.minute + minutesToAdd
  let extraHours := totalMinutes / 60
  let remainingMinutes := totalMinutes % 60
  let totalHours := t.hour + extraHours
  let effectiveHours := totalHours % 24
  {hour := effectiveHours, minute := remainingMinutes}

-- The theorem to state that 2687 minutes after 7:00 a.m. is 3:47 a.m.
theorem timeAfter2687Minutes : addMinutes currentTime 2687 = { hour := 3, minute := 47 } :=
  sorry

end timeAfter2687Minutes_l294_294699


namespace count_multiples_5_or_7_but_not_both_l294_294614

-- Definitions based on the given problem conditions
def multiples_of_five (n : Nat) : Nat :=
  (n - 1) / 5

def multiples_of_seven (n : Nat) : Nat :=
  (n - 1) / 7

def multiples_of_thirty_five (n : Nat) : Nat :=
  (n - 1) / 35

def count_multiples (n : Nat) : Nat :=
  (multiples_of_five n) + (multiples_of_seven n) - 2 * (multiples_of_thirty_five n)

-- The main statement to be proved
theorem count_multiples_5_or_7_but_not_both : count_multiples 101 = 30 :=
by
  sorry

end count_multiples_5_or_7_but_not_both_l294_294614


namespace probability_sum_less_than_product_l294_294537

theorem probability_sum_less_than_product :
  let S := {1, 2, 3, 4, 5, 6}
  in (∃ N : ℕ, N = 6) ∧
     (∃ S' : finset ℕ, S' = finset.Icc 1 N) ∧
     (S = {1, 2, 3, 4, 5, 6}) ∧
     (∀ (a b : ℕ), a ∈ S → b ∈ S →
      (∃ (c d : ℕ), c ∈ S ∧ d ∈ S ∧ (c + d) < (c * d) →
      ∑ S' [set.matrix_card _ (finset ℕ) --> set_prob.select c] = 24 / 36) :=
begin
  let S := {1, 2, 3, 4, 5, 6},
  have hS : S = {1, 2, 3, 4, 5, 6} := rfl,
  let N := 6,
  have hN : N = 6 := rfl,
  let S' := finset.Icc 1 N,
  have hS' : S' = finset.Icc 1 N := rfl,
  sorry
end

end probability_sum_less_than_product_l294_294537


namespace probability_sum_less_than_product_is_5_div_9_l294_294526

-- Define the set of positive integers less than or equal to 6
def ℤ₆ := {n : ℤ | 1 ≤ n ∧ n ≤ 6}

-- Define the probability space on set ℤ₆ x ℤ₆
noncomputable def probability_space : ProbabilitySpace (ℤ₆ × ℤ₆) :=
sorry

-- Event where the sum of two numbers is less than their product
def event_sum_less_than_product (a b : ℤ) : Prop := a + b < a * b

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
Pr[probability_space] {p : ℤ₆ × ℤ₆ | event_sum_less_than_product p.1 p.2}

-- The theorem to prove the probability is 5/9
theorem probability_sum_less_than_product_is_5_div_9 :
  probability_event = 5 / 9 :=
sorry

end probability_sum_less_than_product_is_5_div_9_l294_294526


namespace distribute_balls_in_boxes_l294_294274

theorem distribute_balls_in_boxes :
  let balls := 5
  let boxes := 4
  (4 ^ balls) = 1024 :=
by
  sorry

end distribute_balls_in_boxes_l294_294274


namespace largest_multiple_of_8_less_than_100_l294_294994

theorem largest_multiple_of_8_less_than_100 : ∃ x, x < 100 ∧ 8 ∣ x ∧ ∀ y, y < 100 ∧ 8 ∣ y → y ≤ x :=
by sorry

end largest_multiple_of_8_less_than_100_l294_294994


namespace sqrt_fraction_subtraction_l294_294403

theorem sqrt_fraction_subtraction :
  (Real.sqrt (9 / 2) - Real.sqrt (2 / 9)) = (7 * Real.sqrt 2 / 6) :=
by sorry

end sqrt_fraction_subtraction_l294_294403


namespace intersection_union_complement_l294_294909

open Set

variable (U : Set ℝ)
variable (A B : Set ℝ)

def universal_set := U = univ
def set_A := A = {x : ℝ | -1 ≤ x ∧ x < 2}
def set_B := B = {x : ℝ | 1 < x ∧ x ≤ 3}

theorem intersection (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) (hB : B = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := sorry

theorem union (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) (hB : B = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := sorry

theorem complement (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) :
  U \ A = {x : ℝ | x < -1 ∨ 2 ≤ x} := sorry

end intersection_union_complement_l294_294909


namespace num_students_other_color_shirts_num_students_other_types_pants_num_students_other_color_shoes_l294_294133

def total_students : ℕ := 800

def percentage_blue_shirts : ℕ := 45
def percentage_red_shirts : ℕ := 23
def percentage_green_shirts : ℕ := 15

def percentage_black_pants : ℕ := 30
def percentage_khaki_pants : ℕ := 25
def percentage_jeans_pants : ℕ := 10

def percentage_white_shoes : ℕ := 40
def percentage_black_shoes : ℕ := 20
def percentage_brown_shoes : ℕ := 15

def students_other_color_shirts : ℕ :=
  total_students * (100 - (percentage_blue_shirts + percentage_red_shirts + percentage_green_shirts)) / 100

def students_other_types_pants : ℕ :=
  total_students * (100 - (percentage_black_pants + percentage_khaki_pants + percentage_jeans_pants)) / 100

def students_other_color_shoes : ℕ :=
  total_students * (100 - (percentage_white_shoes + percentage_black_shoes + percentage_brown_shoes)) / 100

theorem num_students_other_color_shirts : students_other_color_shirts = 136 := by
  sorry

theorem num_students_other_types_pants : students_other_types_pants = 280 := by
  sorry

theorem num_students_other_color_shoes : students_other_color_shoes = 200 := by
  sorry

end num_students_other_color_shirts_num_students_other_types_pants_num_students_other_color_shoes_l294_294133


namespace find_x_of_total_area_l294_294445

theorem find_x_of_total_area 
  (x : Real)
  (h_triangle : (1/2) * (4 * x) * (3 * x) = 6 * x^2)
  (h_square1 : (3 * x)^2 = 9 * x^2)
  (h_square2 : (6 * x)^2 = 36 * x^2)
  (h_total : 6 * x^2 + 9 * x^2 + 36 * x^2 = 700) :
  x = Real.sqrt (700 / 51) :=
by {
  sorry
}

end find_x_of_total_area_l294_294445


namespace percentage_difference_wages_l294_294814

variables (W1 W2 : ℝ)
variables (h1 : W1 > 0) (h2 : W2 > 0)
variables (h3 : 0.40 * W2 = 1.60 * 0.20 * W1)

theorem percentage_difference_wages (W1 W2 : ℝ) (h1 : W1 > 0) (h2 : W2 > 0) (h3 : 0.40 * W2 = 1.60 * 0.20 * W1) :
  (W1 - W2) / W1 = 0.20 :=
by
  sorry

end percentage_difference_wages_l294_294814


namespace equal_lengths_imply_equal_segments_l294_294376

theorem equal_lengths_imply_equal_segments 
  (a₁ a₂ b₁ b₂ x y : ℝ) 
  (h₁ : a₁ = a₂) 
  (h₂ : b₁ = b₂) : 
  x = y := 
sorry

end equal_lengths_imply_equal_segments_l294_294376


namespace laptop_total_selling_price_l294_294872

-- Define the original price of the laptop
def originalPrice : ℝ := 1200

-- Define the discount rate
def discountRate : ℝ := 0.30

-- Define the redemption coupon amount
def coupon : ℝ := 50

-- Define the tax rate
def taxRate : ℝ := 0.15

-- Calculate the discount amount
def discountAmount : ℝ := originalPrice * discountRate

-- Calculate the sale price after discount
def salePrice : ℝ := originalPrice - discountAmount

-- Calculate the new sale price after applying the coupon
def newSalePrice : ℝ := salePrice - coupon

-- Calculate the tax amount
def taxAmount : ℝ := newSalePrice * taxRate

-- Calculate the total selling price after tax
def totalSellingPrice : ℝ := newSalePrice + taxAmount

-- Prove that the total selling price is 908.5 dollars
theorem laptop_total_selling_price : totalSellingPrice = 908.5 := by
  unfold totalSellingPrice newSalePrice taxAmount salePrice discountAmount
  norm_num
  sorry

end laptop_total_selling_price_l294_294872


namespace geometric_series_sum_l294_294760

theorem geometric_series_sum :
  let a := 3
  let r := -2
  let n := 10
  let S := a * ((r^n - 1) / (r - 1))
  S = -1023 :=
by 
  -- Sorry allows us to omit the proof details
  sorry

end geometric_series_sum_l294_294760


namespace total_bowling_balls_l294_294178

theorem total_bowling_balls (red_balls : ℕ) (green_balls : ℕ) (h1 : red_balls = 30) (h2 : green_balls = red_balls + 6) : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l294_294178


namespace max_min_product_l294_294679

theorem max_min_product (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x + y + z = 15) (h2 : x * y + y * z + z * x = 45) :
    ∃ m : ℝ, m = min (x * y) (min (y * z) (z * x)) ∧ m ≤ 17.5 :=
by
  sorry

end max_min_product_l294_294679


namespace convex_quadrilateral_lower_bound_l294_294000

theorem convex_quadrilateral_lower_bound
  (n : ℕ)
  (h_n_gt_four : n > 4)
  (no_three_collinear : ∀ p1 p2 p3 : ℝ × ℝ, 
    (p1 ≠ p2) → (p2 ≠ p3) → (p1 ≠ p3) → ¬ collinear ℝ {p1, p2, p3}) :
  ∃ k, k ≥ binom (n - 3) 2 ∧ (∃ convex_quadrilaterals : set (set (ℝ × ℝ)),
    convex_quadrilaterals ⊆ (powerset_univ n).filter(λ s, s.card = 4 ∧ convex_hull_convex s) ∧
    convex_quadrilaterals.card = k) :=
sorry

end convex_quadrilateral_lower_bound_l294_294000


namespace sum_lt_prod_probability_l294_294524

def probability_product_greater_than_sum : ℚ :=
  23 / 36

theorem sum_lt_prod_probability :
  ∃ a b : ℤ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧
  (∑ i in finset.Icc 1 6, ∑ j in finset.Icc 1 6, 
    if (a, b) = (i, j) ∧ (a - 1) * (b - 1) > 1 
    then 1 else 0) / 36 = probability_product_greater_than_sum := by
  sorry

end sum_lt_prod_probability_l294_294524


namespace rebecca_gemstones_needed_l294_294470

-- Definitions for the conditions
def magnets_per_earring : Nat := 2
def buttons_per_magnet : Nat := 1 / 2
def gemstones_per_button : Nat := 3
def earrings_per_set : Nat := 2
def sets : Nat := 4

-- Statement to be proved
theorem rebecca_gemstones_needed : 
  gemstones_per_button * (buttons_per_magnet * (magnets_per_earring * (earrings_per_set * sets))) = 24 :=
by
  sorry

end rebecca_gemstones_needed_l294_294470


namespace polynomial_sum_l294_294954

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l294_294954


namespace flowers_bloom_l294_294043

theorem flowers_bloom (num_unicorns : ℕ) (flowers_per_step : ℕ) (distance_km : ℕ) (step_length_m : ℕ) 
  (h1 : num_unicorns = 6) (h2 : flowers_per_step = 4) (h3 : distance_km = 9) (h4 : step_length_m = 3) : 
  num_unicorns * (distance_km * 1000 / step_length_m) * flowers_per_step = 72000 :=
by
  sorry

end flowers_bloom_l294_294043


namespace arithmetic_sequence_ratio_l294_294601

theorem arithmetic_sequence_ratio (x y a₁ a₂ a₃ b₁ b₂ b₃ b₄ : ℝ) (h₁ : x ≠ y)
    (h₂ : a₁ = x + d) (h₃ : a₂ = x + 2 * d) (h₄ : a₃ = x + 3 * d) (h₅ : y = x + 4 * d)
    (h₆ : b₁ = x - d') (h₇ : b₂ = x + d') (h₈ : b₃ = x + 2 * d') (h₉ : y = x + 3 * d') (h₁₀ : b₄ = x + 4 * d') :
    (b₄ - b₃) / (a₂ - a₁) = 8 / 3 :=
by
  sorry

end arithmetic_sequence_ratio_l294_294601


namespace sequence_properties_l294_294250

theorem sequence_properties (S : ℕ → ℝ) (a : ℕ → ℝ) :
  S 2 = 4 →
  (∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1) →
  a 1 = 1 ∧ S 5 = 121 :=
by
  intros hS2 ha
  sorry

end sequence_properties_l294_294250


namespace age_twice_in_years_l294_294195

theorem age_twice_in_years (x : ℕ) : (40 + x = 2 * (12 + x)) → x = 16 :=
by {
  sorry
}

end age_twice_in_years_l294_294195


namespace odd_number_expression_l294_294319

theorem odd_number_expression (o n : ℤ) (ho : o % 2 = 1) : (o^2 + n * o + 1) % 2 = 1 ↔ n % 2 = 1 := by
  sorry

end odd_number_expression_l294_294319


namespace part1_part2_l294_294959

noncomputable def f (a : ℝ) (x : ℝ) := (a * x - 1) * (x - 1)

theorem part1 (h : ∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 2) : a = 1/2 :=
  sorry

theorem part2 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 1/a) ∨
  (a = 1 → ∀ x : ℝ, ¬(f a x < 0)) ∨
  (∀ x : ℝ, f a x < 0 ↔ 1/a < x ∧ x < 1) :=
  sorry

end part1_part2_l294_294959


namespace sin_double_alpha_zero_l294_294322

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem sin_double_alpha_zero (α : ℝ) (h : f α = 1) : Real.sin (2 * α) = 0 :=
by 
  -- Proof would go here, but we're using sorry
  sorry

end sin_double_alpha_zero_l294_294322


namespace complex_problem_l294_294112

open Complex

noncomputable def z : ℂ := (1 + I) / Real.sqrt 2

theorem complex_problem :
  1 + z^50 + z^100 = I := 
by
  -- Subproofs or transformations will be here.
  sorry

end complex_problem_l294_294112


namespace company_sales_difference_l294_294984

theorem company_sales_difference:
  let price_A := 4
  let quantity_A := 300
  let total_sales_A := price_A * quantity_A
  let price_B := 3.5
  let quantity_B := 350
  let total_sales_B := price_B * quantity_B
  total_sales_B - total_sales_A = 25 :=
by
  sorry

end company_sales_difference_l294_294984


namespace distinct_pairwise_products_l294_294902

theorem distinct_pairwise_products
  (n a b c d : ℕ) (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_bounds: n^2 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2) :
  (a * b ≠ a * c ∧ a * b ≠ a * d ∧ a * b ≠ b * c ∧ a * b ≠ b * d ∧ a * b ≠ c * d) ∧
  (a * c ≠ a * d ∧ a * c ≠ b * c ∧ a * c ≠ b * d ∧ a * c ≠ c * d) ∧
  (a * d ≠ b * c ∧ a * d ≠ b * d ∧ a * d ≠ c * d) ∧
  (b * c ≠ b * d ∧ b * c ≠ c * d) ∧
  (b * d ≠ c * d) :=
sorry

end distinct_pairwise_products_l294_294902


namespace unoccupied_volume_l294_294358

/--
Given:
1. Three congruent cones, each with a radius of 8 cm and a height of 8 cm.
2. The cones are enclosed within a cylinder such that the bases of two cones are at each base of the cylinder, and one cone is inverted in the middle touching the other two cones at their vertices.
3. The height of the cylinder is 16 cm.

Prove:
The volume of the cylinder not occupied by the cones is 512π cubic cm.
-/
theorem unoccupied_volume 
  (r h : ℝ) 
  (hr : r = 8) 
  (hh_cone : h = 8) 
  (hh_cyl : h_cyl = 16) 
  : (π * r^2 * h_cyl) - (3 * (1/3 * π * r^2 * h)) = 512 * π := 
by 
  sorry

end unoccupied_volume_l294_294358


namespace sheilas_hours_mwf_is_24_l294_294477

-- Define Sheila's earning conditions and working hours
def sheilas_hours_mwf (H : ℕ) : Prop :=
  let hours_tu_th := 6 * 2
  let earnings_tu_th := hours_tu_th * 14
  let earnings_mwf := 504 - earnings_tu_th
  H = earnings_mwf / 14

-- The theorem to state that Sheila works 24 hours on Monday, Wednesday, and Friday
theorem sheilas_hours_mwf_is_24 : sheilas_hours_mwf 24 :=
by
  -- Proof is omitted
  sorry

end sheilas_hours_mwf_is_24_l294_294477


namespace average_marks_for_class_l294_294490

theorem average_marks_for_class (total_students : ℕ) (marks_group1 marks_group2 marks_group3 : ℕ) (num_students_group1 num_students_group2 num_students_group3 : ℕ) 
  (h1 : total_students = 50) 
  (h2 : num_students_group1 = 10) 
  (h3 : marks_group1 = 90) 
  (h4 : num_students_group2 = 15) 
  (h5 : marks_group2 = 80) 
  (h6 : num_students_group3 = total_students - num_students_group1 - num_students_group2) 
  (h7 : marks_group3 = 60) : 
  (10 * 90 + 15 * 80 + (total_students - 10 - 15) * 60) / total_students = 72 := 
by
  sorry

end average_marks_for_class_l294_294490


namespace probability_sum_less_than_product_l294_294539

noncomputable def probability_condition_met : ℚ :=
  let S : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6);
  let pairs_meeting_condition : Finset (ℕ × ℕ) := S.filter (λ p, (p.1 + 1) * (p.2 + 1) > (p.1 + 1) + (p.2 + 1));
  pairs_meeting_condition.card.to_rat / S.card

theorem probability_sum_less_than_product :
  probability_condition_met = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l294_294539


namespace parallelogram_angle_l294_294371

theorem parallelogram_angle (a b : ℝ) (h1 : a + b = 180) (h2 : a = b + 50) : b = 65 :=
by
  -- Proof would go here, but we're adding a placeholder
  sorry

end parallelogram_angle_l294_294371


namespace balls_in_boxes_l294_294286

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l294_294286


namespace probability_sum_less_than_product_l294_294531

theorem probability_sum_less_than_product :
  let s := Finset.Icc 1 6
  let pairs := s.product s
  let valid_pairs := pairs.filter (fun (a, b) => (a - 1) * (b - 1) > 1)
  (valid_pairs.card : ℚ) / pairs.card = 4 / 9 := by
  sorry

end probability_sum_less_than_product_l294_294531


namespace compressor_stations_distances_l294_294064

theorem compressor_stations_distances 
    (x y z a : ℝ) 
    (h1 : x + y = 2 * z)
    (h2 : z + y = x + a)
    (h3 : x + z = 75)
    (h4 : 0 ≤ x)
    (h5 : 0 ≤ y)
    (h6 : 0 ≤ z)
    (h7 : 0 < a)
    (h8 : a < 100) :
  (a = 15 → x = 42 ∧ y = 24 ∧ z = 33) :=
by 
  intro ha_eq_15
  sorry

end compressor_stations_distances_l294_294064


namespace solve_system_correct_l294_294963

noncomputable def solve_system (a b c d e : ℝ) : Prop :=
  3 * a = (b + c + d) ^ 3 ∧ 
  3 * b = (c + d + e) ^ 3 ∧ 
  3 * c = (d + e + a) ^ 3 ∧ 
  3 * d = (e + a + b) ^ 3 ∧ 
  3 * e = (a + b + c) ^ 3

theorem solve_system_correct :
  ∀ (a b c d e : ℝ), solve_system a b c d e → 
    (a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3 ∧ e = 1/3) ∨ 
    (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0) ∨ 
    (a = -1/3 ∧ b = -1/3 ∧ c = -1/3 ∧ d = -1/3 ∧ e = -1/3) :=
by
  sorry

end solve_system_correct_l294_294963


namespace average_age_of_10_students_l294_294967

theorem average_age_of_10_students
  (avg_age_25_students : ℕ)
  (num_students_25 : ℕ)
  (avg_age_14_students : ℕ)
  (num_students_14 : ℕ)
  (age_25th_student : ℕ)
  (avg_age_10_students : ℕ)
  (h_avg_age_25 : avg_age_25_students = 25)
  (h_num_students_25 : num_students_25 = 25)
  (h_avg_age_14 : avg_age_14_students = 28)
  (h_num_students_14 : num_students_14 = 14)
  (h_age_25th : age_25th_student = 13)
  : avg_age_10_students = 22 :=
by
  sorry

end average_age_of_10_students_l294_294967


namespace find_m_of_odd_function_l294_294803

theorem find_m_of_odd_function (m : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = ((x + 3) * (x + m)) / x)
  (h₂ : ∀ x, f (-x) = -f x) : m = -3 :=
sorry

end find_m_of_odd_function_l294_294803


namespace find_unknown_number_l294_294654

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end find_unknown_number_l294_294654


namespace cone_base_area_l294_294804

theorem cone_base_area (r l : ℝ) (h1 : (1/2) * π * l^2 = 2 * π) (h2 : 2 * π * r = 2 * π) :
  π * r^2 = π :=
by 
  sorry

end cone_base_area_l294_294804


namespace polynomial_sum_l294_294945

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l294_294945


namespace range_of_z_l294_294608

theorem range_of_z (x y : ℝ) (h : x^2 + 2 * x * y + 4 * y^2 = 6) :
  4 ≤ x^2 + 4 * y^2 ∧ x^2 + 4 * y^2 ≤ 12 :=
by
  sorry

end range_of_z_l294_294608


namespace solve_equation_l294_294098

theorem solve_equation : ∀ x : ℝ, (10 - x) ^ 2 = 4 * x ^ 2 ↔ x = 10 / 3 ∨ x = -10 :=
by
  intros x
  sorry

end solve_equation_l294_294098


namespace total_profit_percentage_l294_294746

theorem total_profit_percentage (total_apples : ℕ) (percent_sold_10 : ℝ) (percent_sold_30 : ℝ) (profit_10 : ℝ) (profit_30 : ℝ) : 
  total_apples = 280 → 
  percent_sold_10 = 0.40 → 
  percent_sold_30 = 0.60 → 
  profit_10 = 0.10 → 
  profit_30 = 0.30 → 
  ((percent_sold_10 * total_apples * (1 + profit_10) + percent_sold_30 * total_apples * (1 + profit_30) - total_apples) / total_apples * 100) = 22 := 
by 
  intros; sorry

end total_profit_percentage_l294_294746


namespace least_positive_integer_with_six_factors_l294_294726

-- Define what it means for a number to have exactly six distinct positive factors
def hasExactlySixFactors (n : ℕ) : Prop :=
  (n.factorization.support.card = 2 ∧ (n.factorization.values' = [2, 1])) ∨
  (n.factorization.support.card = 1 ∧ (n.factorization.values' = [5]))

-- The main theorem statement
theorem least_positive_integer_with_six_factors : ∃ n : ℕ, hasExactlySixFactors n ∧ ∀ m : ℕ, (hasExactlySixFactors m → n ≤ m) :=
  exists.intro 12 (and.intro
    (show hasExactlySixFactors 12, by sorry)
    (show ∀ m : ℕ, hasExactlySixFactors m → 12 ≤ m, by sorry))

end least_positive_integer_with_six_factors_l294_294726


namespace determine_age_l294_294054

def David_age (D Y : ℕ) : Prop := Y = 2 * D ∧ Y = D + 7

theorem determine_age (D : ℕ) (h : David_age D (D + 7)) : D = 7 :=
by
  sorry

end determine_age_l294_294054


namespace tan_double_angle_identity_l294_294288

theorem tan_double_angle_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) : 
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := 
  sorry

end tan_double_angle_identity_l294_294288


namespace count_two_digit_remainders_l294_294617

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l294_294617


namespace geometric_series_sum_l294_294756

theorem geometric_series_sum :
  ∀ (a r : ℤ) (n : ℕ),
  a = 3 → r = -2 → n = 10 →
  (a * ((r ^ n - 1) / (r - 1))) = -1024 :=
by
  intros a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l294_294756


namespace ring_stack_distance_l294_294392

noncomputable def vertical_distance (rings : Nat) : Nat :=
  let diameters := List.range rings |>.map (λ i => 15 - 2 * i)
  let thickness := 1 * rings
  thickness

theorem ring_stack_distance :
  vertical_distance 7 = 58 :=
by 
  sorry

end ring_stack_distance_l294_294392


namespace round_trip_time_l294_294022

theorem round_trip_time 
  (d1 d2 d3 : ℝ) 
  (s1 s2 s3 t : ℝ) 
  (h1 : d1 = 18) 
  (h2 : d2 = 18) 
  (h3 : d3 = 36) 
  (h4 : s1 = 12) 
  (h5 : s2 = 10) 
  (h6 : s3 = 9) 
  (h7 : t = (d1 / s1) + (d2 / s2) + (d3 / s3)) :
  t = 7.3 :=
by
  sorry

end round_trip_time_l294_294022


namespace func4_same_domain_range_as_func1_l294_294083

noncomputable def func1_domain : Set ℝ := {x | 0 < x}
noncomputable def func1_range : Set ℝ := {y | 0 < y}

noncomputable def func4_domain : Set ℝ := {x | 0 < x}
noncomputable def func4_range : Set ℝ := {y | 0 < y}

theorem func4_same_domain_range_as_func1 :
  (func4_domain = func1_domain) ∧ (func4_range = func1_range) :=
sorry

end func4_same_domain_range_as_func1_l294_294083


namespace ball_box_problem_l294_294271

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end ball_box_problem_l294_294271


namespace fifth_term_arithmetic_sequence_l294_294970

-- Conditions provided
def first_term (x y : ℝ) := x + y^2
def second_term (x y : ℝ) := x - y^2
def third_term (x y : ℝ) := x - 3*y^2
def fourth_term (x y : ℝ) := x - 5*y^2

-- Proof to determine the fifth term
theorem fifth_term_arithmetic_sequence (x y : ℝ) :
  (fourth_term x y) - (third_term x y) = -2*y^2 →
  (x - 5 * y^2) - 2 * y^2 = x - 7 * y^2 :=
by sorry

end fifth_term_arithmetic_sequence_l294_294970


namespace find_a_find_inverse_function_l294_294956

section proof_problem

variables (f : ℝ → ℝ) (a : ℝ)

-- Condition that f is an odd function
def is_odd_function : Prop := ∀ x, f (-x) = -f (x)

-- Condition from the problem
def condition_1 : Prop := (a - 1) * (2 ^ (1:ℝ) + 1) = 0
def function_definition : Prop := f = λ x, 2^x - 1

-- Question 1: Find the value of a
theorem find_a
  (h1 : is_odd_function f)
  (h2 : condition_1 a)
  (h3 : function_definition f) : a = 1 :=
sorry

-- Question 2: Find the inverse function f⁻¹(x)
noncomputable def inverse_function (y : ℝ) : ℝ := real.log (y + 1) / real.log 2

theorem find_inverse_function
  (h3 : function_definition f) : 
  ∀ x, f (inverse_function x) = x :=
sorry

end proof_problem

end find_a_find_inverse_function_l294_294956


namespace total_shoes_l294_294463

variable boots : ℕ
variable slippers : ℕ
variable heels : ℕ

-- Condition: Nancy has six pairs of boots
def boots_pairs : boots = 6 := rfl

-- Condition: Nancy has nine more pairs of slippers than boots
def slippers_pairs : slippers = boots + 9 := rfl

-- Condition: Nancy has a number of pairs of heels equal to three times the combined number of slippers and boots
def heels_pairs : heels = 3 * (boots + slippers) := by
  rw [boots_pairs, slippers_pairs]
  sorry  -- assuming the correctness of the consequent computation as rfl

-- Goal: Total number of individual shoes is 168
theorem total_shoes : (boots * 2) + (slippers * 2) + (heels * 2) = 168 := by
  rw [boots_pairs, slippers_pairs, heels_pairs]
  sorry  -- verifying the summing up to 168 as a proof

end total_shoes_l294_294463


namespace first_grade_sample_count_l294_294387

-- Defining the total number of students and their ratio in grades 1, 2, and 3.
def total_students : ℕ := 2400
def ratio_grade1 : ℕ := 5
def ratio_grade2 : ℕ := 4
def ratio_grade3 : ℕ := 3
def total_ratio := ratio_grade1 + ratio_grade2 + ratio_grade3

-- Defining the sample size
def sample_size : ℕ := 120

-- Proving that the number of first-grade students sampled should be 50.
theorem first_grade_sample_count : 
  (sample_size * ratio_grade1) / total_ratio = 50 :=
by
  -- sorry is added here to skip the proof
  sorry

end first_grade_sample_count_l294_294387


namespace arithmetic_seq_sum_a4_a6_l294_294135

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_sum_a4_a6 (a : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_root1 : a 3 ^ 2 - 3 * a 3 + 1 = 0)
  (h_root2 : a 7 ^ 2 - 3 * a 7 + 1 = 0) :
  a 4 + a 6 = 3 :=
sorry

end arithmetic_seq_sum_a4_a6_l294_294135


namespace smallest_integer_with_six_distinct_factors_l294_294722

noncomputable def least_pos_integer_with_six_factors : ℕ :=
  12

theorem smallest_integer_with_six_distinct_factors 
  (n : ℕ)
  (p q : ℕ)
  (a b : ℕ)
  (hp : prime p)
  (hq : prime q)
  (h_diff : p ≠ q)
  (h_n : n = p ^ a * q ^ b)
  (h_factors : (a + 1) * (b + 1) = 6) :
  n = least_pos_integer_with_six_factors :=
by
  sorry

end smallest_integer_with_six_distinct_factors_l294_294722


namespace num_ways_to_put_5_balls_into_4_boxes_l294_294280

theorem num_ways_to_put_5_balls_into_4_boxes : 
  ∃ n : ℕ, n = 4^5 ∧ n = 1024 :=
by
  use 4^5
  split
  · rfl
  · norm_num

end num_ways_to_put_5_balls_into_4_boxes_l294_294280


namespace fraction_of_quarters_from_1800_to_1809_l294_294936

def num_total_quarters := 26
def num_states_1800s := 8

theorem fraction_of_quarters_from_1800_to_1809 : 
  (num_states_1800s / num_total_quarters : ℚ) = 4 / 13 :=
by
  sorry

end fraction_of_quarters_from_1800_to_1809_l294_294936


namespace find_number_l294_294744

theorem find_number (x : ℝ) (h : x / 0.025 = 40) : x = 1 := 
by sorry

end find_number_l294_294744


namespace tangent_line_tangent_to_circle_min_2_pow_a_2_pow_b_l294_294257

open Real

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := -(1 / sqrt b) * exp (sqrt a * x)

-- Condition that tangent line to graph at x = 0 is tangent to the circle x^2 + y^2 = 1
theorem tangent_line_tangent_to_circle_min_2_pow_a_2_pow_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (tangent_condition : a + b = 1) : (2:ℝ)^a + (2:ℝ)^b = 2 * sqrt 2 :=
by
  sorry

end tangent_line_tangent_to_circle_min_2_pow_a_2_pow_b_l294_294257


namespace div_by_3_pow_101_l294_294691

theorem div_by_3_pow_101 : ∀ (n : ℕ), (∀ k : ℕ, (3^(k+1)) ∣ (2^(3^k) + 1)) → 3^101 ∣ 2^(3^100) + 1 :=
by
  sorry

end div_by_3_pow_101_l294_294691


namespace josh_ribbon_shortfall_l294_294812

-- Define the total amount of ribbon Josh has
def total_ribbon : ℝ := 18

-- Define the number of gifts
def num_gifts : ℕ := 6

-- Define the ribbon requirements for each gift
def ribbon_per_gift_wrapping : ℝ := 2
def ribbon_per_bow : ℝ := 1.5
def ribbon_per_tag : ℝ := 0.25
def ribbon_per_trim : ℝ := 0.5

-- Calculate the total ribbon required for all the tasks
def total_ribbon_needed : ℝ :=
  (ribbon_per_gift_wrapping * num_gifts) +
  (ribbon_per_bow * num_gifts) +
  (ribbon_per_tag * num_gifts) +
  (ribbon_per_trim * num_gifts)

-- Calculate the ribbon shortfall
def ribbon_shortfall : ℝ :=
  total_ribbon_needed - total_ribbon

-- Prove that Josh will be short by 7.5 yards of ribbon
theorem josh_ribbon_shortfall : ribbon_shortfall = 7.5 := by
  sorry

end josh_ribbon_shortfall_l294_294812


namespace angle_A_measure_find_a_l294_294305

theorem angle_A_measure (a b c : ℝ) (A B C : ℝ) (h1 : (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) :
  A = π / 3 :=
by
  -- proof steps are omitted
  sorry

theorem find_a (a b c : ℝ) (A : ℝ) (h2 : 2 * c = 3 * b) (area : ℝ) (h3 : area = 6 * Real.sqrt 3)
  (h4 : A = π / 3) :
  a = 2 * Real.sqrt 21 / 3 :=
by
  -- proof steps are omitted
  sorry

end angle_A_measure_find_a_l294_294305


namespace parallelogram_slope_l294_294093

theorem parallelogram_slope (a b c d : ℚ) :
    a = 35 + c ∧ b = 125 - c ∧ 875 - 25 * c = 280 + 8 * c ∧ (a, 8) = (b, 25)
    → ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ (∃ h : 8 * 33 * a + 595 = 2350, (m, n) = (25, 4)) :=
by
  sorry

end parallelogram_slope_l294_294093


namespace negation_of_universal_prop_l294_294705

theorem negation_of_universal_prop : 
  (¬ (∀ (x : ℝ), x ^ 2 ≥ 0)) ↔ (∃ (x : ℝ), x ^ 2 < 0) :=
by sorry

end negation_of_universal_prop_l294_294705


namespace solve_z_for_complex_eq_l294_294962

theorem solve_z_for_complex_eq (i : ℂ) (h : i^2 = -1) : ∀ (z : ℂ), 3 - 2 * i * z = -4 + 5 * i * z → z = -i :=
by
  intro z
  intro eqn
  -- The proof would go here
  sorry

end solve_z_for_complex_eq_l294_294962


namespace find_x_plus_y_l294_294113

theorem find_x_plus_y (x y : ℝ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : x - y > 0) : x + y = 8 ∨ x + y = 2 :=
by
  sorry

end find_x_plus_y_l294_294113


namespace value_of_k_h_5_l294_294640

def h (x : ℝ) : ℝ := 4 * x + 6
def k (x : ℝ) : ℝ := 6 * x - 8

theorem value_of_k_h_5 : k (h 5) = 148 :=
by
  have h5 : h 5 = 4 * 5 + 6 := rfl
  simp [h5, h, k]
  sorry

end value_of_k_h_5_l294_294640


namespace two_digit_integers_remainder_3_count_l294_294631

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end two_digit_integers_remainder_3_count_l294_294631


namespace sum_expr_le_e4_l294_294455

theorem sum_expr_le_e4
  (α β γ δ ε : ℝ) :
  (1 - α) * Real.exp α +
  (1 - β) * Real.exp (α + β) +
  (1 - γ) * Real.exp (α + β + γ) +
  (1 - δ) * Real.exp (α + β + γ + δ) +
  (1 - ε) * Real.exp (α + β + γ + δ + ε) ≤ Real.exp 4 :=
sorry

end sum_expr_le_e4_l294_294455


namespace unknown_number_value_l294_294647

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end unknown_number_value_l294_294647


namespace total_savings_correct_l294_294014

-- Definitions of savings per day and days saved for Josiah, Leah, and Megan
def josiah_saving_per_day : ℝ := 0.25
def josiah_days : ℕ := 24

def leah_saving_per_day : ℝ := 0.50
def leah_days : ℕ := 20

def megan_saving_per_day : ℝ := 1.00
def megan_days : ℕ := 12

-- Definition to calculate total savings for each child
def total_saving (saving_per_day : ℝ) (days : ℕ) : ℝ :=
  saving_per_day * days

-- Total amount saved by Josiah, Leah, and Megan
def total_savings : ℝ :=
  total_saving josiah_saving_per_day josiah_days +
  total_saving leah_saving_per_day leah_days +
  total_saving megan_saving_per_day megan_days

-- Theorem to prove the total savings is $28
theorem total_savings_correct : total_savings = 28 := by
  sorry

end total_savings_correct_l294_294014


namespace david_age_l294_294056

theorem david_age (x : ℕ) (y : ℕ) (h1 : y = x + 7) (h2 : y = 2 * x) : x = 7 :=
by
  sorry

end david_age_l294_294056


namespace vector_addition_example_l294_294903

noncomputable def OA : ℝ × ℝ := (-2, 3)
noncomputable def AB : ℝ × ℝ := (-1, -4)
noncomputable def OB : ℝ × ℝ := (OA.1 + AB.1, OA.2 + AB.2)

theorem vector_addition_example :
  OB = (-3, -1) :=
by
  sorry

end vector_addition_example_l294_294903


namespace line_through_intersection_of_circles_l294_294795

theorem line_through_intersection_of_circles 
  (x y : ℝ)
  (C1 : x^2 + y^2 = 10)
  (C2 : (x-1)^2 + (y-3)^2 = 20) :
  x + 3 * y = 0 :=
sorry

end line_through_intersection_of_circles_l294_294795


namespace num_rem_three_by_seven_l294_294622

theorem num_rem_three_by_seven : 
  ∃ (n : ℕ → ℕ), 13 = cardinality {m : ℕ | 10 ≤ m ∧ m < 100 ∧ ∃ k, m = 7 * k + 3}.count sorry

end num_rem_three_by_seven_l294_294622


namespace sufficient_but_not_necessary_condition_l294_294115

-- Define the conditions as predicates
def p (x : ℝ) : Prop := x^2 - 3 * x - 4 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 6 * x + 9 - m^2 ≤ 0

-- Range for m where p is sufficient but not necessary for q
def m_range (m : ℝ) : Prop := m ≤ -4 ∨ m ≥ 4

-- The main goal to be proven
theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x, p x → q x m) ∧ ¬(∀ x, q x m → p x) ↔ m_range m :=
sorry

end sufficient_but_not_necessary_condition_l294_294115


namespace find_difference_l294_294916

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  sorry

end find_difference_l294_294916


namespace value_of_k_h_5_l294_294641

def h (x : ℝ) : ℝ := 4 * x + 6
def k (x : ℝ) : ℝ := 6 * x - 8

theorem value_of_k_h_5 : k (h 5) = 148 :=
by
  have h5 : h 5 = 4 * 5 + 6 := rfl
  simp [h5, h, k]
  sorry

end value_of_k_h_5_l294_294641


namespace triangle_area_proof_l294_294347

noncomputable def cos_fun1 (x : ℝ) : ℝ := 2 * Real.cos (3 * x) + 1
noncomputable def cos_fun2 (x : ℝ) : ℝ := - Real.cos (2 * x)

theorem triangle_area_proof :
  let P := (5 * Real.pi, cos_fun1 (5 * Real.pi))
  let Q := (9 * Real.pi / 2, cos_fun2 (9 * Real.pi / 2))
  let m := (Q.snd - P.snd) / (Q.fst - P.fst)
  let y_intercept := P.snd - m * P.fst
  let y_intercept_point := (0, y_intercept)
  let x_intercept := -y_intercept / m
  let x_intercept_point := (x_intercept, 0)
  let base := x_intercept
  let height := y_intercept
  17 * Real.pi / 4 ≤ P.fst ∧ P.fst ≤ 21 * Real.pi / 4 ∧
  17 * Real.pi / 4 ≤ Q.fst ∧ Q.fst ≤ 21 * Real.pi / 4 ∧
  (P.fst = 5 * Real.pi ∧ Q.fst = 9 * Real.pi / 2) →
  1/2 * base * height = 361 * Real.pi / 8 :=
by
  sorry

end triangle_area_proof_l294_294347


namespace Jimin_addition_l294_294308

theorem Jimin_addition (x : ℕ) (h : 96 / x = 6) : 34 + x = 50 := 
by
  sorry

end Jimin_addition_l294_294308


namespace percentage_selected_B_l294_294807

-- Definitions for the given conditions
def candidates := 7900
def selected_A := (6 / 100) * candidates
def selected_B := selected_A + 79

-- The question to be answered
def P_B := (selected_B / candidates) * 100

-- Proof statement
theorem percentage_selected_B : P_B = 7 := 
by
  -- Canonical statement placeholder 
  sorry

end percentage_selected_B_l294_294807


namespace min_knights_proof_l294_294865

-- Noncomputable theory as we are dealing with existence proofs
noncomputable def min_knights (n : ℕ) : ℕ :=
  -- Given the table contains 1001 people
  if n = 1001 then 502 else 0

-- The proof problem statement, we need to ensure that minimum number of knights is 502
theorem min_knights_proof : min_knights 1001 = 502 := 
  by
    -- Sketch of proof: Deriving that the minimum number of knights must be 502 based on the problem constraints
    sorry

end min_knights_proof_l294_294865


namespace total_onions_grown_l294_294020

-- Given conditions
def onions_grown_by_Nancy : ℕ := 2
def onions_grown_by_Dan : ℕ := 9
def onions_grown_by_Mike : ℕ := 4
def days_worked : ℕ := 6

-- Statement we need to prove
theorem total_onions_grown : onions_grown_by_Nancy + onions_grown_by_Dan + onions_grown_by_Mike = 15 :=
by sorry

end total_onions_grown_l294_294020


namespace magician_trick_l294_294742

theorem magician_trick (T : ℕ) (cards : fin 52) (edge_choice : fin 52 → bool) :
  ∀ init_position, (init_position = 0 ∨ init_position = 51) →
  (∃ remaining_position, remaining_position = init_position ∧ (∀ k, cards k = true → k ≠ remaining_position)) :=
sorry

end magician_trick_l294_294742


namespace Mishas_fathers_speed_Mishas_fathers_speed_in_kmh_l294_294460

theorem Mishas_fathers_speed (d : ℝ) (t : ℝ) (V : ℝ) 
  (h1 : d = 5) 
  (h2 : t = 10) 
  (h3 : 2 * (d / V) = t) :
  V = 1 :=
by
  sorry

theorem Mishas_fathers_speed_in_kmh (d : ℝ) (t : ℝ) (V : ℝ) (V_kmh : ℝ)
  (h1 : d = 5) 
  (h2 : t = 10) 
  (h3 : 2 * (d / V) = t) 
  (h4 : V_kmh = V * 60):
  V_kmh = 60 :=
by
  sorry

end Mishas_fathers_speed_Mishas_fathers_speed_in_kmh_l294_294460


namespace eds_weight_l294_294217

variable (Al Ben Carl Ed : ℕ)

def weight_conditions : Prop :=
  Carl = 175 ∧ Ben = Carl - 16 ∧ Al = Ben + 25 ∧ Ed = Al - 38

theorem eds_weight (h : weight_conditions Al Ben Carl Ed) : Ed = 146 :=
by
  -- Conditions
  have h1 : Carl = 175    := h.1
  have h2 : Ben = Carl - 16 := h.2.1
  have h3 : Al = Ben + 25   := h.2.2.1
  have h4 : Ed = Al - 38    := h.2.2.2
  -- Proof itself is omitted, sorry placeholder
  sorry

end eds_weight_l294_294217


namespace earnings_difference_l294_294978

-- We define the price per bottle for each company and the number of bottles sold by each company.
def priceA : ℝ := 4
def priceB : ℝ := 3.5
def quantityA : ℕ := 300
def quantityB : ℕ := 350

-- We define the earnings for each company based on the provided conditions.
def earningsA : ℝ := priceA * quantityA
def earningsB : ℝ := priceB * quantityB

-- We state the theorem that the difference in earnings is $25.
theorem earnings_difference : (earningsB - earningsA) = 25 := by
  -- Proof omitted.
  sorry

end earnings_difference_l294_294978


namespace pure_imaginary_a_l294_294131

theorem pure_imaginary_a (a : ℝ) :
  (a^2 - 4 = 0) ∧ (a - 2 ≠ 0) ↔ a = -2 :=
by
  sorry

end pure_imaginary_a_l294_294131


namespace ashley_friends_ages_correct_sum_l294_294344

noncomputable def ashley_friends_ages_sum : Prop :=
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
                   (1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9) ∧
                   (a * b = 36) ∧ (c * d = 30) ∧ (a + b + c + d = 24)

theorem ashley_friends_ages_correct_sum : ashley_friends_ages_sum := sorry

end ashley_friends_ages_correct_sum_l294_294344


namespace product_divisible_by_60_l294_294335

open Nat

theorem product_divisible_by_60 (S : Finset ℕ) (h_card : S.card = 10) (h_sum : S.sum id = 62) :
  60 ∣ S.prod id :=
  sorry

end product_divisible_by_60_l294_294335


namespace horizontal_asymptote_value_l294_294644

theorem horizontal_asymptote_value :
  ∀ (x : ℝ),
  ((8 * x^4 + 6 * x^3 + 7 * x^2 + 2 * x + 4) / 
  (2 * x^4 + 5 * x^3 + 3 * x^2 + x + 6)) = (4 : ℝ) :=
by sorry

end horizontal_asymptote_value_l294_294644


namespace fraction_product_eq_six_l294_294360

theorem fraction_product_eq_six : (2/5) * (3/4) * (1/6) * (120 : ℚ) = 6 := by
  sorry

end fraction_product_eq_six_l294_294360


namespace total_bowling_balls_l294_294185

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 :=
by
  sorry

end total_bowling_balls_l294_294185


namespace polynomial_sum_l294_294948

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + h x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l294_294948


namespace limit_hours_overtime_l294_294739

theorem limit_hours_overtime (R O : ℝ) (earnings total_hours : ℕ) (L : ℕ) 
    (hR : R = 16)
    (hO : O = R + 0.75 * R)
    (h_earnings : earnings = 864)
    (h_total_hours : total_hours = 48)
    (calc_earnings : earnings = L * R + (total_hours - L) * O) :
    L = 40 := by
  sorry

end limit_hours_overtime_l294_294739
