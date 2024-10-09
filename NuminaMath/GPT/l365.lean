import Mathlib

namespace sum_of_a_for_unique_solution_l365_36577

theorem sum_of_a_for_unique_solution (a : ℝ) (x : ℝ) :
  (∃ (a : ℝ), 3 * x ^ 2 + a * x + 6 * x + 7 = 0 ∧ (a + 6) ^ 2 - 4 * 3 * 7 = 0) →
  (-6 + 2 * Real.sqrt 21 + -6 - 2 * Real.sqrt 21 = -12) :=
by
  sorry

end sum_of_a_for_unique_solution_l365_36577


namespace value_of_x_squared_minus_y_squared_l365_36562

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end value_of_x_squared_minus_y_squared_l365_36562


namespace parallel_lines_constant_l365_36534

theorem parallel_lines_constant (a : ℝ) : 
  (∀ x y : ℝ, (a - 1) * x + 2 * y + 3 = 0 → x + a * y + 3 = 0) → a = -1 :=
by sorry

end parallel_lines_constant_l365_36534


namespace car_distance_and_velocity_l365_36504

def acceleration : ℝ := 12 -- constant acceleration in m/s^2
def time : ℝ := 36 -- time in seconds
def conversion_factor : ℝ := 3.6 -- conversion factor from m/s to km/h

theorem car_distance_and_velocity :
  (1/2 * acceleration * time^2 = 7776) ∧ (acceleration * time * conversion_factor = 1555.2) :=
by
  sorry

end car_distance_and_velocity_l365_36504


namespace eval_expression_l365_36558

def f (x : ℤ) : ℤ := 3 * x^2 - 6 * x + 10

theorem eval_expression : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end eval_expression_l365_36558


namespace polynomial_value_at_minus_2_l365_36529

variable (a b : ℝ)

def polynomial (x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem polynomial_value_at_minus_2 :
  (polynomial a b (-2) = -21) :=
  sorry

end polynomial_value_at_minus_2_l365_36529


namespace ratio_of_width_to_length_l365_36564

-- Definitions of length, width, perimeter
def l : ℕ := 10
def P : ℕ := 30

-- Define the condition for the width
def width_from_perimeter (l P : ℕ) : ℕ :=
  (P - 2 * l) / 2

-- Calculate the width using the given length and perimeter
def w : ℕ := width_from_perimeter l P

-- Theorem stating the ratio of width to length
theorem ratio_of_width_to_length : (w : ℚ) / l = 1 / 2 := by
  -- Proof steps will go here
  sorry

end ratio_of_width_to_length_l365_36564


namespace frankie_pets_total_l365_36555

noncomputable def total_pets (c : ℕ) : ℕ :=
  let dogs := 2
  let cats := c
  let snakes := c + 5
  let parrots := c - 1
  dogs + cats + snakes + parrots

theorem frankie_pets_total (c : ℕ) (hc : 2 + 4 + (c + 1) + (c - 1) = 19) : total_pets c = 19 := by
  sorry

end frankie_pets_total_l365_36555


namespace initial_cards_l365_36532

variable (x : ℕ)
variable (h1 : x - 3 = 2)

theorem initial_cards (x : ℕ) (h1 : x - 3 = 2) : x = 5 := by
  sorry

end initial_cards_l365_36532


namespace fraction_of_pizza_peter_ate_l365_36519

theorem fraction_of_pizza_peter_ate (total_slices : ℕ) (peter_slices : ℕ) (shared_slices : ℚ) 
  (pizza_fraction : ℚ) : 
  total_slices = 16 → 
  peter_slices = 2 → 
  shared_slices = 1/3 → 
  pizza_fraction = peter_slices / total_slices + (1 / 2) * shared_slices / total_slices → 
  pizza_fraction = 13 / 96 :=
by 
  intros h1 h2 h3 h4
  -- to be proved later
  sorry

end fraction_of_pizza_peter_ate_l365_36519


namespace correct_operation_l365_36582

theorem correct_operation :
  (∀ a : ℝ, a^4 * a^3 = a^7)
  ∧ (∀ a : ℝ, (a^2)^3 ≠ a^5)
  ∧ (∀ a : ℝ, 3 * a^2 - a^2 ≠ 2)
  ∧ (∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2) :=
by {
  sorry
}

end correct_operation_l365_36582


namespace melanie_plums_count_l365_36581

theorem melanie_plums_count (dan_plums sally_plums total_plums melanie_plums : ℕ)
    (h1 : dan_plums = 9)
    (h2 : sally_plums = 3)
    (h3 : total_plums = 16)
    (h4 : melanie_plums = total_plums - (dan_plums + sally_plums)) :
    melanie_plums = 4 := by
  -- Proof will be filled here
  sorry

end melanie_plums_count_l365_36581


namespace compare_a_b_c_l365_36569

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l365_36569


namespace range_of_x_l365_36537

-- Define the necessary properties and functions.
variable (f : ℝ → ℝ)
variable (hf_even : ∀ x : ℝ, f (-x) = f x)
variable (hf_monotonic : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y)

-- Define the statement to be proved.
theorem range_of_x (f : ℝ → ℝ) (hf_even : ∀ x, f (-x) = f x) (hf_monotonic : ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  { x : ℝ | f (2 * x - 1) ≤ f 3 } = { x | -1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end range_of_x_l365_36537


namespace mary_flour_amount_l365_36505

noncomputable def cups_of_flour_already_put_in
    (total_flour_needed : ℕ)
    (total_sugar_needed : ℕ)
    (extra_flour_needed : ℕ)
    (flour_to_be_added : ℕ) : ℕ :=
total_flour_needed - (total_sugar_needed + extra_flour_needed)

theorem mary_flour_amount
    (total_flour_needed : ℕ := 9)
    (total_sugar_needed : ℕ := 6)
    (extra_flour_needed : ℕ := 1) :
    cups_of_flour_already_put_in total_flour_needed total_sugar_needed extra_flour_needed (total_sugar_needed + extra_flour_needed) = 2 := by
  sorry

end mary_flour_amount_l365_36505


namespace solution_of_inequality_l365_36571

theorem solution_of_inequality : 
  {x : ℝ | x^2 - x - 2 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 2 < x} :=
by
  sorry

end solution_of_inequality_l365_36571


namespace smallest_interesting_rectangle_area_l365_36597

/-- 
  A rectangle is interesting if both its side lengths are integers and 
  it contains exactly four lattice points strictly in its interior.
  Prove that the area of the smallest such interesting rectangle is 10.
-/
theorem smallest_interesting_rectangle_area :
  ∃ (a b : ℕ), (a - 1) * (b - 1) = 4 ∧ a * b = 10 :=
by
  sorry

end smallest_interesting_rectangle_area_l365_36597


namespace remainder_of_polynomial_l365_36574

noncomputable def P (x : ℝ) := 3 * x^5 - 2 * x^3 + 5 * x^2 - 8
noncomputable def D (x : ℝ) := x^2 + 3 * x + 2
noncomputable def R (x : ℝ) := 64 * x + 60

theorem remainder_of_polynomial :
  ∀ x : ℝ, P x % D x = R x :=
sorry

end remainder_of_polynomial_l365_36574


namespace length_of_PQ_l365_36512

theorem length_of_PQ (p : ℝ) (h : p > 0) (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hx1x2 : x1 + x2 = 3 * p) (hy1 : y1^2 = 2 * p * x1) (hy2 : y2^2 = 2 * p * x2) 
  (focus : ¬ (y1 = 0)) : (abs (x1 - x2 + 2 * p) = 4 * p) := 
sorry

end length_of_PQ_l365_36512


namespace simplify_expression_l365_36545

theorem simplify_expression (x : ℝ) : (3 * x)^4 + 3 * x * x^3 + 2 * x^5 = 84 * x^4 + 2 * x^5 := by
    sorry

end simplify_expression_l365_36545


namespace number_of_terminating_decimals_l365_36596

theorem number_of_terminating_decimals : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 299 → (∃ k : ℕ, n = 9 * k) → 
  ∃ count : ℕ, count = 33 := 
sorry

end number_of_terminating_decimals_l365_36596


namespace average_income_of_other_40_customers_l365_36528

/-
Given:
1. The average income of 50 customers is $45,000.
2. The average income of the wealthiest 10 customers is $55,000.

Prove:
1. The average income of the other 40 customers is $42,500.
-/

theorem average_income_of_other_40_customers 
  (avg_income_50 : ℝ)
  (wealthiest_10_avg : ℝ) 
  (total_customers : ℕ)
  (wealthiest_customers : ℕ)
  (remaining_customers : ℕ)
  (h1 : avg_income_50 = 45000)
  (h2 : wealthiest_10_avg = 55000)
  (h3 : total_customers = 50)
  (h4 : wealthiest_customers = 10)
  (h5 : remaining_customers = 40) :
  let total_income_50 := total_customers * avg_income_50
  let total_income_wealthiest_10 := wealthiest_customers * wealthiest_10_avg
  let income_remaining_customers := total_income_50 - total_income_wealthiest_10
  let avg_income_remaining := income_remaining_customers / remaining_customers
  avg_income_remaining = 42500 := 
sorry

end average_income_of_other_40_customers_l365_36528


namespace sequence_term_20_l365_36595

theorem sequence_term_20 :
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (∀ n, a (n+1) = a n + 2) → (a 20 = 39) := by
  intros a h1 h2
  sorry

end sequence_term_20_l365_36595


namespace coefficient_of_x_neg_2_in_binomial_expansion_l365_36554

theorem coefficient_of_x_neg_2_in_binomial_expansion :
  let x := (x : ℚ)
  let term := (x^3 - (2 / x))^6
  (coeff_of_term : Int) ->
  (coeff_of_term = -192) :=
by
  -- Placeholder for the proof
  sorry

end coefficient_of_x_neg_2_in_binomial_expansion_l365_36554


namespace excluded_number_is_35_l365_36510

theorem excluded_number_is_35 (numbers : List ℝ) 
  (h_len : numbers.length = 5)
  (h_avg1 : (numbers.sum / 5) = 27)
  (h_len_excl : (numbers.length - 1) = 4)
  (avg_remaining : ℝ)
  (remaining_numbers : List ℝ)
  (remaining_condition : remaining_numbers.length = 4)
  (h_avg2 : (remaining_numbers.sum / 4) = 25) :
  numbers.sum - remaining_numbers.sum = 35 :=
by sorry

end excluded_number_is_35_l365_36510


namespace find_a_plus_b_l365_36567

noncomputable def real_part (z : ℂ) : ℝ := z.re
noncomputable def imag_part (z : ℂ) : ℝ := z.im

theorem find_a_plus_b (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (1 + i) * (2 + i) = a + b * i) : a + b = 4 :=
by sorry

end find_a_plus_b_l365_36567


namespace not_axiom_l365_36540

theorem not_axiom (P Q R S : Prop)
  (B : P -> Q -> R -> S)
  (C : P -> Q)
  (D : P -> R)
  : ¬ (P -> Q -> S) :=
sorry

end not_axiom_l365_36540


namespace product_price_reduction_l365_36548

theorem product_price_reduction (z : ℝ) (x : ℝ) (hp1 : z > 0) (hp2 : 0.85 * 0.85 * z = z * (1 - x / 100)) : x = 27.75 := by
  sorry

end product_price_reduction_l365_36548


namespace simpl_eval_l365_36514

variable (a b : ℚ)

theorem simpl_eval (h_a : a = 1/2) (h_b : b = -1/3) :
    5 * (3 * a ^ 2 * b - a * b ^ 2) - 4 * (- a * b ^ 2 + 3 * a ^ 2 * b) = -11 / 36 := by
  sorry

end simpl_eval_l365_36514


namespace number_of_terms_in_expansion_l365_36594

theorem number_of_terms_in_expansion (A B : Finset ℕ) (h1 : A.card = 4) (h2 : B.card = 5) :
  (A.product B).card = 20 :=
by
  sorry

end number_of_terms_in_expansion_l365_36594


namespace convert_speed_kmph_to_mps_l365_36538

def kilometers_to_meters := 1000
def hours_to_seconds := 3600
def speed_kmph := 18
def expected_speed_mps := 5

theorem convert_speed_kmph_to_mps :
  speed_kmph * (kilometers_to_meters / hours_to_seconds) = expected_speed_mps :=
by
  sorry

end convert_speed_kmph_to_mps_l365_36538


namespace value_range_l365_36526

-- Step to ensure proofs about sine and real numbers are within scope
open Real

noncomputable def y (x : ℝ) : ℝ := 2 * sin x * cos x - 1

theorem value_range (x : ℝ) : -2 ≤ y x ∧ y x ≤ 0 :=
by sorry

end value_range_l365_36526


namespace simplified_expression_at_3_l365_36544

noncomputable def simplify_and_evaluate (x : ℝ) : ℝ :=
  (3 * x ^ 2 + 8 * x - 6) - (2 * x ^ 2 + 4 * x - 15)

theorem simplified_expression_at_3 : simplify_and_evaluate 3 = 30 :=
by
  sorry

end simplified_expression_at_3_l365_36544


namespace inequality_solution_set_l365_36527

theorem inequality_solution_set :
  {x : ℝ | (x^2 - x - 6) / (x - 1) > 0} = {x : ℝ | (-2 < x ∧ x < 1) ∨ (3 < x)} := by
  sorry

end inequality_solution_set_l365_36527


namespace min_value_eq_l365_36546

open Real
open Classical

noncomputable def min_value (x y : ℝ) : ℝ := x + 4 * y

theorem min_value_eq :
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (1 / x + 1 / (2 * y) = 1) → (min_value x y) = 3 + 2 * sqrt 2 :=
by
  sorry

end min_value_eq_l365_36546


namespace find_xyz_l365_36542

open Complex

-- Definitions of the variables and conditions
variables {a b c x y z : ℂ} (h_a_ne_zero : a ≠ 0) (h_b_ne_zero : b ≠ 0) (h_c_ne_zero : c ≠ 0)
  (h_x_ne_zero : x ≠ 0) (h_y_ne_zero : y ≠ 0) (h_z_ne_zero : z ≠ 0)
  (h1 : a = (b - c) * (x + 2))
  (h2 : b = (a - c) * (y + 2))
  (h3 : c = (a - b) * (z + 2))
  (h4 : x * y + x * z + y * z = 12)
  (h5 : x + y + z = 6)

-- Statement of the theorem
theorem find_xyz : x * y * z = 7 := 
by
  -- Proof steps to be filled in
  sorry

end find_xyz_l365_36542


namespace carla_highest_final_number_l365_36509

def alice_final_number (initial : ℕ) : ℕ :=
  let step1 := initial * 2
  let step2 := step1 - 3
  let step3 := step2 / 3
  step3 + 4

def bob_final_number (initial : ℕ) : ℕ :=
  let step1 := initial + 5
  let step2 := step1 * 2
  let step3 := step2 - 4
  step3 / 2

def carla_final_number (initial : ℕ) : ℕ :=
  let step1 := initial - 2
  let step2 := step1 * 2
  let step3 := step2 + 3
  step3 * 2

theorem carla_highest_final_number : carla_final_number 12 > bob_final_number 12 ∧ carla_final_number 12 > alice_final_number 12 :=
  by
  have h_alice : alice_final_number 12 = 11 := by rfl
  have h_bob : bob_final_number 12 = 15 := by rfl
  have h_carla : carla_final_number 12 = 46 := by rfl
  sorry

end carla_highest_final_number_l365_36509


namespace a_n_sequence_term2015_l365_36583

theorem a_n_sequence_term2015 :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ a 2 = 1/2 ∧ (∀ n ≥ 2, a n * (a (n-1) + a (n+1)) = 2 * a (n+1) * a (n-1)) ∧ a 2015 = 1/2015 :=
sorry

end a_n_sequence_term2015_l365_36583


namespace sum_first_10_terms_arithmetic_sequence_l365_36508

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l365_36508


namespace solve_first_l365_36573

theorem solve_first (x y : ℝ) (C : ℝ) :
  (1 + y^2) * (deriv id x) - (1 + x^2) * y * (deriv id y) = 0 →
  Real.arctan x = 1/2 * Real.log (1 + y^2) + Real.log C := 
sorry

end solve_first_l365_36573


namespace solve_for_q_l365_36565

noncomputable def is_arithmetic_SUM_seq (a₁ q: ℝ) (n: ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem solve_for_q (a₁ q S3 S6 S9: ℝ) (hq: q ≠ 1) (hS3: S3 = is_arithmetic_SUM_seq a₁ q 3) 
(hS6: S6 = is_arithmetic_SUM_seq a₁ q 6) (hS9: S9 = is_arithmetic_SUM_seq a₁ q 9) 
(h_arith: 2 * S9 = S3 + S6) : q^3 = 3 / 2 :=
sorry

end solve_for_q_l365_36565


namespace find_line_equation_l365_36587

noncomputable def y_line (m b x : ℝ) : ℝ := m * x + b
noncomputable def quadratic_y (x : ℝ) : ℝ := x ^ 2 + 8 * x + 7

noncomputable def equation_of_the_line : Prop :=
  ∃ (m b k : ℝ),
    (quadratic_y k = y_line m b k + 6 ∨ quadratic_y k = y_line m b k - 6) ∧
    (y_line m b 2 = 7) ∧ 
    b ≠ 0 ∧
    y_line 19.5 (-32) = y_line m b

theorem find_line_equation : equation_of_the_line :=
sorry

end find_line_equation_l365_36587


namespace valid_y_values_for_triangle_l365_36580

-- Define the triangle inequality conditions for sides 8, 11, and y^2
theorem valid_y_values_for_triangle (y : ℕ) (h_pos : y > 0) :
  (8 + 11 > y^2) ∧ (8 + y^2 > 11) ∧ (11 + y^2 > 8) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by
  sorry

end valid_y_values_for_triangle_l365_36580


namespace jon_found_marbles_l365_36547

-- Definitions based on the conditions
variables (M J B : ℕ)

-- Prove that Jon found 110 marbles
theorem jon_found_marbles
  (h1 : M + J = 66)
  (h2 : M = 2 * J)
  (h3 : J + B = 3 * M) :
  B = 110 :=
by
  sorry -- proof to be completed

end jon_found_marbles_l365_36547


namespace negation_proposition_p_l365_36593

open Classical

variable (n : ℕ)

def proposition_p : Prop := ∃ n : ℕ, 2^n > 100

theorem negation_proposition_p : ¬ proposition_p ↔ ∀ n : ℕ, 2^n ≤ 100 := 
by sorry

end negation_proposition_p_l365_36593


namespace hyperbola_eccentricity_is_sqrt2_l365_36506

noncomputable def eccentricity_of_hyperbola {a b : ℝ} (h : a ≠ 0) (hb : b = a) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  (c / a)

theorem hyperbola_eccentricity_is_sqrt2 {a : ℝ} (h : a ≠ 0) :
  eccentricity_of_hyperbola h (rfl) = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_is_sqrt2_l365_36506


namespace michael_final_revenue_l365_36598

noncomputable def total_revenue_before_discount : ℝ :=
  (3 * 45) + (5 * 22) + (7 * 16) + (8 * 10) + (10 * 5)

noncomputable def discount : ℝ := 0.10 * total_revenue_before_discount

noncomputable def discounted_revenue : ℝ := total_revenue_before_discount - discount

noncomputable def sales_tax : ℝ := 0.06 * discounted_revenue

noncomputable def final_revenue : ℝ := discounted_revenue + sales_tax

theorem michael_final_revenue : final_revenue = 464.60 :=
by
  sorry

end michael_final_revenue_l365_36598


namespace carson_gardening_time_l365_36592

-- Definitions of the problem conditions
def lines_to_mow : ℕ := 40
def minutes_per_line : ℕ := 2
def rows_of_flowers : ℕ := 8
def flowers_per_row : ℕ := 7
def minutes_per_flower : ℚ := 0.5

-- Total time calculation for the proof 
theorem carson_gardening_time : 
  (lines_to_mow * minutes_per_line) + (rows_of_flowers * flowers_per_row * minutes_per_flower) = 108 := 
by 
  sorry

end carson_gardening_time_l365_36592


namespace ramu_repair_cost_l365_36578

theorem ramu_repair_cost
  (initial_cost : ℝ)
  (selling_price : ℝ)
  (profit_percent : ℝ)
  (repair_cost : ℝ)
  (h1 : initial_cost = 42000)
  (h2 : selling_price = 64900)
  (h3 : profit_percent = 13.859649122807017 / 100)
  (h4 : selling_price = initial_cost + repair_cost + profit_percent * (initial_cost + repair_cost)) :
  repair_cost = 15000 :=
by
  sorry

end ramu_repair_cost_l365_36578


namespace area_ratio_gt_two_ninths_l365_36599

variables {A B C P Q R : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited R]

def divides_perimeter_eq (A B C : Type*) (P Q R : Type*) : Prop :=
-- Definition that P, Q, and R divide the perimeter into three equal parts
sorry

def is_on_side_AB (A B C P Q : Type*) : Prop :=
-- Definition that points P and Q are on side AB
sorry

theorem area_ratio_gt_two_ninths (A B C P Q R : Type*)
  (H1 : divides_perimeter_eq A B C P Q R)
  (H2 : is_on_side_AB A B C P Q) :
  -- Statement to prove that the area ratio is greater than 2/9
  (S_ΔPQR / S_ΔABC) > (2 / 9) :=
sorry

end area_ratio_gt_two_ninths_l365_36599


namespace find_x_l365_36531

theorem find_x (x : ℝ) (h : 45 * x = 0.60 * 900) : x = 12 :=
by
  sorry

end find_x_l365_36531


namespace all_solutions_of_diophantine_eq_l365_36543

theorem all_solutions_of_diophantine_eq
  (a b c x0 y0 : ℤ) (h_gcd : Int.gcd a b = 1)
  (h_sol : a * x0 + b * y0 = c) :
  ∀ x y : ℤ, (a * x + b * y = c) →
  ∃ t : ℤ, x = x0 + b * t ∧ y = y0 - a * t :=
by
  sorry

end all_solutions_of_diophantine_eq_l365_36543


namespace max_min_values_of_f_l365_36500

noncomputable def f (x : ℝ) : ℝ := x^2

theorem max_min_values_of_f : 
  (∀ x, -3 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 9) ∧ (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = 9) ∧ (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = 0) :=
by
  sorry

end max_min_values_of_f_l365_36500


namespace chocolate_cost_first_store_l365_36541

def cost_first_store (x : ℕ) : ℕ := x
def chocolate_promotion_store : ℕ := 2
def savings_in_three_weeks : ℕ := 6
def number_of_chocolates (weeks : ℕ) : ℕ := 2 * weeks

theorem chocolate_cost_first_store :
  ∀ (weeks : ℕ) (x : ℕ), 
    number_of_chocolates weeks = 6 →
    chocolate_promotion_store * number_of_chocolates weeks + savings_in_three_weeks = cost_first_store x * number_of_chocolates weeks →
    cost_first_store x = 3 :=
by
  intros weeks x h1 h2
  sorry

end chocolate_cost_first_store_l365_36541


namespace find_f_1789_l365_36572

def f : ℕ → ℕ := sorry

axiom f_1 : f 1 = 5
axiom f_f_n : ∀ n, f (f n) = 4 * n + 9
axiom f_2n : ∀ n, f (2 * n) = (2 * n) + 1 + 3

theorem find_f_1789 : f 1789 = 3581 :=
by
  sorry

end find_f_1789_l365_36572


namespace isosceles_triangle_area_l365_36524

theorem isosceles_triangle_area (p x : ℝ) 
  (h1 : 2 * p = 6 * x) 
  (h2 : 0 < p) 
  (h3 : 0 < x) :
  (1 / 2) * (2 * x) * (Real.sqrt (8 * p^2 / 9)) = (Real.sqrt 8 * p^2) / 3 :=
by
  sorry

end isosceles_triangle_area_l365_36524


namespace cos_squared_value_l365_36507

theorem cos_squared_value (α : ℝ) (h : Real.tan (α + π/4) = 3/4) : Real.cos (π/4 - α) ^ 2 = 9 / 25 :=
sorry

end cos_squared_value_l365_36507


namespace smaller_number_l365_36589

theorem smaller_number (x y : ℤ) (h1 : x + y = 12) (h2 : x - y = 20) : y = -4 := 
by 
  sorry

end smaller_number_l365_36589


namespace work_done_resistive_force_l365_36568

noncomputable def mass : ℝ := 0.01  -- 10 grams converted to kilograms
noncomputable def v1 : ℝ := 400.0  -- initial speed in m/s
noncomputable def v2 : ℝ := 100.0  -- final speed in m/s

noncomputable def kinetic_energy (m v : ℝ) : ℝ := 0.5 * m * v^2

theorem work_done_resistive_force :
  let KE1 := kinetic_energy mass v1
  let KE2 := kinetic_energy mass v2
  KE1 - KE2 = 750 :=
by
  sorry

end work_done_resistive_force_l365_36568


namespace salary_after_cuts_l365_36588

noncomputable def finalSalary (init_salary : ℝ) (cuts : List ℝ) : ℝ :=
  cuts.foldl (λ salary cut => salary * (1 - cut)) init_salary

theorem salary_after_cuts :
  finalSalary 5000 [0.0525, 0.0975, 0.146, 0.128] = 3183.63 :=
by
  sorry

end salary_after_cuts_l365_36588


namespace find_multiple_l365_36579

theorem find_multiple:
  let number := 220025
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := number / sum
  let remainder := number % sum
  (remainder = 25) → (quotient = 220) → (quotient / diff = 2) :=
by
  intros number sum diff quotient remainder h1 h2
  sorry

end find_multiple_l365_36579


namespace part1_part2_l365_36550

noncomputable def f (x : ℝ) := Real.exp x

theorem part1 (x : ℝ) (h : x ≥ 0) (m : ℝ) : 
  (x - 1) * f x ≥ m * x^2 - 1 ↔ m ≤ 1 / 2 :=
sorry

theorem part2 (x : ℝ) (h : x > 0) : 
  f x > 4 * Real.log x + 8 - 8 * Real.log 2 :=
sorry

end part1_part2_l365_36550


namespace find_digit_A_l365_36591

-- Define the six-digit number for any digit A
def six_digit_number (A : ℕ) : ℕ := 103200 + A * 10 + 4
-- Define the condition that a number is prime
def is_prime (n : ℕ) : Prop := (2 ≤ n) ∧ ∀ m : ℕ, 2 ≤ m → m * m ≤ n → ¬ (m ∣ n)

-- The main theorem stating that A must equal 1 for the number to be prime
theorem find_digit_A (A : ℕ) : A = 1 ↔ is_prime (six_digit_number A) :=
by
  sorry -- Proof to be filled in


end find_digit_A_l365_36591


namespace part1_l365_36518

theorem part1 (a b c : ℚ) (h1 : a^2 = 9) (h2 : |b| = 4) (h3 : c^3 = 27) (h4 : a * b < 0) (h5 : b * c > 0) : 
  a * b - b * c + c * a = -33 := by
  sorry

end part1_l365_36518


namespace parking_lot_problem_l365_36552

variable (M S : Nat)

theorem parking_lot_problem (h1 : M + S = 30) (h2 : 15 * M + 8 * S = 324) :
  M = 12 ∧ S = 18 :=
by
  -- proof omitted
  sorry

end parking_lot_problem_l365_36552


namespace couscous_dishes_l365_36561

def dishes (a b c d : ℕ) : ℕ := (a + b + c) / d

theorem couscous_dishes :
  dishes 7 13 45 5 = 13 :=
by
  unfold dishes
  sorry

end couscous_dishes_l365_36561


namespace payments_option1_option2_option1_more_effective_combined_option_cost_l365_36560

variable {x : ℕ}

-- Condition 1: Prices and discount options
def badminton_rackets_price : ℕ := 40
def shuttlecocks_price : ℕ := 10
def discount_option1_free_shuttlecocks (pairs : ℕ): ℕ := pairs
def discount_option2_price (price : ℕ) : ℕ := price * 9 / 10

-- Condition 2: Buying requirements
def pairs_needed : ℕ := 10
def shuttlecocks_needed (n : ℕ) : ℕ := n
axiom x_gt_10 : x > 10

-- Proof Problem 1: Payment calculations
theorem payments_option1_option2 (x : ℕ) (h : x > 10) :
  (shuttlecocks_price * (shuttlecocks_needed x - discount_option1_free_shuttlecocks pairs_needed) + badminton_rackets_price * pairs_needed =
    10 * x + 300) ∧
  (discount_option2_price (shuttlecocks_price * shuttlecocks_needed x + badminton_rackets_price * pairs_needed) =
    9 * x + 360) :=
sorry

-- Proof Problem 2: More cost-effective option when x=30
theorem option1_more_effective (x : ℕ) (h : x = 30) :
  (10 * x + 300 < 9 * x + 360) :=
sorry

-- Proof Problem 3: Another cost-effective method when x=30
theorem combined_option_cost (x : ℕ) (h : x = 30) :
  (badminton_rackets_price * pairs_needed + discount_option2_price (shuttlecocks_price * (shuttlecocks_needed x - 10)) = 580) :=
sorry

end payments_option1_option2_option1_more_effective_combined_option_cost_l365_36560


namespace average_lecture_minutes_l365_36549

theorem average_lecture_minutes
  (lecture_duration : ℕ)
  (total_audience : ℕ)
  (percent_entire : ℝ)
  (percent_missed : ℝ)
  (percent_half : ℝ)
  (average_minutes : ℝ) :
  lecture_duration = 90 →
  total_audience = 200 →
  percent_entire = 0.30 →
  percent_missed = 0.20 →
  percent_half = 0.40 →
  average_minutes = 56.25 :=
by
  sorry

end average_lecture_minutes_l365_36549


namespace minimum_dwarfs_to_prevent_empty_chair_sitting_l365_36557

theorem minimum_dwarfs_to_prevent_empty_chair_sitting :
  ∀ (C : Fin 30 → Bool), (∀ i, C i ∨ C ((i + 1) % 30) ∨ C ((i + 2) % 30)) ↔ (∃ n, n = 10) :=
by
  sorry

end minimum_dwarfs_to_prevent_empty_chair_sitting_l365_36557


namespace proof_solution_l365_36566

variable (U : Set ℝ) (A : Set ℝ) (C_U_A : Set ℝ)
variables (a b : ℝ)

noncomputable def proof_problem : Prop :=
  (U = Set.univ) →
  (A = {x | a ≤ x ∧ x ≤ b}) →
  (C_U_A = {x | x > 4 ∨ x < 3}) →
  A = {x | 3 ≤ x ∧ x ≤ 4} ∧ a = 3 ∧ b = 4

theorem proof_solution : proof_problem U A C_U_A a b :=
by
  intro hU hA hCUA
  have hA_eq : A = {x | 3 ≤ x ∧ x ≤ 4} :=
    by { sorry }
  have ha : a = 3 :=
    by { sorry }
  have hb : b = 4 :=
    by { sorry }
  exact ⟨hA_eq, ha, hb⟩

end proof_solution_l365_36566


namespace trains_meet_in_32_seconds_l365_36539

noncomputable def train_meeting_time
  (length_train1 : ℕ)
  (length_train2 : ℕ)
  (initial_distance : ℕ)
  (speed_train1_kmph : ℕ)
  (speed_train2_kmph : ℕ)
  : ℕ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600
  let relative_speed := speed_train1_mps + speed_train2_mps
  let total_distance := length_train1 + length_train2 + initial_distance
  total_distance / relative_speed

theorem trains_meet_in_32_seconds :
  train_meeting_time 400 200 200 54 36 = 32 := 
by
  sorry

end trains_meet_in_32_seconds_l365_36539


namespace part1_solution_set_part2_range_of_m_l365_36535

def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

theorem part1_solution_set (x : ℝ) : (f x 3 >= 6) ↔ (x ≤ -2 ∨ x ≥ 4) :=
by sorry

theorem part2_range_of_m (m : ℝ) (x : ℝ) : 
 (∀ x : ℝ, f x m ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7) :=
by sorry

end part1_solution_set_part2_range_of_m_l365_36535


namespace find_smaller_number_l365_36584

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 := by
  sorry

end find_smaller_number_l365_36584


namespace distance_between_Jay_and_Paul_l365_36522

theorem distance_between_Jay_and_Paul
  (initial_distance : ℕ)
  (jay_speed : ℕ)
  (paul_speed : ℕ)
  (time : ℕ)
  (jay_distance_walked : ℕ)
  (paul_distance_walked : ℕ) :
  initial_distance = 3 →
  jay_speed = 1 / 20 →
  paul_speed = 3 / 40 →
  time = 120 →
  jay_distance_walked = jay_speed * time →
  paul_distance_walked = paul_speed * time →
  initial_distance + jay_distance_walked + paul_distance_walked = 18 := by
  sorry

end distance_between_Jay_and_Paul_l365_36522


namespace book_arrangement_count_l365_36511

theorem book_arrangement_count :
  let total_books := 6
  let identical_science_books := 3
  let unique_other_books := total_books - identical_science_books
  (total_books! / (identical_science_books! * unique_other_books!)) = 120 := by
  sorry

end book_arrangement_count_l365_36511


namespace multiple_of_son_age_last_year_l365_36556

theorem multiple_of_son_age_last_year
  (G : ℕ) (S : ℕ) (M : ℕ)
  (h1 : G = 42 - 1)
  (h2 : S = 16 - 1)
  (h3 : G = M * S - 4) :
  M = 3 := by
  sorry

end multiple_of_son_age_last_year_l365_36556


namespace postcard_cost_l365_36570

theorem postcard_cost (x : ℕ) (h₁ : 9 * x < 1000) (h₂ : 10 * x > 1100) : x = 111 :=
by
  sorry

end postcard_cost_l365_36570


namespace monday_rainfall_l365_36586

theorem monday_rainfall (tuesday_rainfall monday_rainfall: ℝ) 
(less_rain: ℝ) (h1: tuesday_rainfall = 0.2) 
(h2: less_rain = 0.7) 
(h3: tuesday_rainfall = monday_rainfall - less_rain): 
monday_rainfall = 0.9 :=
by sorry

end monday_rainfall_l365_36586


namespace altitude_division_l365_36575

variables {A B C D E : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E]

theorem altitude_division 
  (AD DC CE EB y : ℝ)
  (hAD : AD = 6)
  (hDC : DC = 4)
  (hCE : CE = 3)
  (hEB : EB = y)
  (h_similarity : CE / DC = (AD + DC) / (y + CE)) : 
  y = 31 / 3 :=
by
  sorry

end altitude_division_l365_36575


namespace side_length_of_square_l365_36521

theorem side_length_of_square (r : ℝ) (A : ℝ) (s : ℝ) 
  (h1 : π * r^2 = 36 * π) 
  (h2 : s = 2 * r) : 
  s = 12 :=
by 
  sorry

end side_length_of_square_l365_36521


namespace fractional_part_painted_correct_l365_36516

noncomputable def fractional_part_painted (time_fence : ℕ) (time_hole : ℕ) : ℚ :=
  (time_hole : ℚ) / time_fence

theorem fractional_part_painted_correct : fractional_part_painted 60 40 = 2 / 3 := by
  sorry

end fractional_part_painted_correct_l365_36516


namespace probability_multiple_of_4_l365_36513

def prob_at_least_one_multiple_of_4 : ℚ :=
  1 - (38/50)^3

theorem probability_multiple_of_4 (n : ℕ) (h : n = 3) : 
  prob_at_least_one_multiple_of_4 = 28051 / 50000 :=
by
  rw [prob_at_least_one_multiple_of_4, ← h]
  sorry

end probability_multiple_of_4_l365_36513


namespace parallel_lines_m_value_l365_36585

/-- Given two lines x + m * y + 6 = 0 and (m - 2) * x + 3 * y + 2 * m = 0 are parallel,
    prove that the value of the real number m that makes the lines parallel is -1. -/
theorem parallel_lines_m_value (m : ℝ) : 
  (x + m * y + 6 = 0 ∧ (m - 2) * x + 3 * y + 2 * m = 0 → 
  (m = -1)) :=
by
  sorry

end parallel_lines_m_value_l365_36585


namespace inequality_proof_l365_36553

theorem inequality_proof 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 > 0) 
  (h2 : a2 > 0) 
  (h3 : a3 > 0)
  (h4 : a4 > 0):
  (a1 + a3) / (a1 + a2) + 
  (a2 + a4) / (a2 + a3) + 
  (a3 + a1) / (a3 + a4) + 
  (a4 + a2) / (a4 + a1) ≥ 4 :=
by
  sorry

end inequality_proof_l365_36553


namespace surface_area_of_solid_l365_36533

theorem surface_area_of_solid (num_unit_cubes : ℕ) (top_layer_cubes : ℕ) 
(bottom_layer_cubes : ℕ) (side_layer_cubes : ℕ) 
(front_and_back_cubes : ℕ) (left_and_right_cubes : ℕ) :
  num_unit_cubes = 15 →
  top_layer_cubes = 5 →
  bottom_layer_cubes = 5 →
  side_layer_cubes = 3 →
  front_and_back_cubes = 5 →
  left_and_right_cubes = 3 →
  let top_and_bottom_surface := top_layer_cubes + bottom_layer_cubes
  let front_and_back_surface := 2 * front_and_back_cubes
  let left_and_right_surface := 2 * left_and_right_cubes
  let total_surface := top_and_bottom_surface + front_and_back_surface + left_and_right_surface
  total_surface = 26 :=
by
  intros h_n h_t h_b h_s h_f h_lr
  let top_and_bottom_surface := top_layer_cubes + bottom_layer_cubes
  let front_and_back_surface := 2 * front_and_back_cubes
  let left_and_right_surface := 2 * left_and_right_cubes
  let total_surface := top_and_bottom_surface + front_and_back_surface + left_and_right_surface
  sorry

end surface_area_of_solid_l365_36533


namespace largest_a_mul_b_l365_36530

-- Given conditions and proof statement
theorem largest_a_mul_b {m k q a b : ℕ} (hm : m = 720 * k + 83)
  (ha : m = a * q + b) (h_b_lt_a: b < a): a * b = 5112 :=
sorry

end largest_a_mul_b_l365_36530


namespace right_triangle_hypotenuse_l365_36501

theorem right_triangle_hypotenuse (a h : ℝ) (r : ℝ) (h1 : r = 8) (h2 : h = a * Real.sqrt 2)
  (h3 : r = (a - h) / 2) : h = 16 * (Real.sqrt 2 + 1) := 
by
  sorry

end right_triangle_hypotenuse_l365_36501


namespace min_sum_complementary_events_l365_36525

theorem min_sum_complementary_events (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hP : (1 / y) + (4 / x) = 1) : x + y ≥ 9 :=
sorry

end min_sum_complementary_events_l365_36525


namespace find_k_l365_36520

theorem find_k
  (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : ∃ (x y : ℝ), (x - k * y - 5 = 0 ∧ x^2 + y^2 = 10 ∧ (A = (x, y) ∨ B = (x, y))))
  (h2 : (A.fst^2 + A.snd^2 = 10) ∧ (B.fst^2 + B.snd^2 = 10))
  (h3 : (A.fst - k * A.snd - 5 = 0) ∧ (B.fst - k * B.snd - 5 = 0))
  (h4 : A.fst * B.fst + A.snd * B.snd = 0) :
  k = 2 ∨ k = -2 :=
by
  sorry

end find_k_l365_36520


namespace Reema_loan_problem_l365_36517

-- Define problem parameters
def Principal : ℝ := 150000
def Interest : ℝ := 42000
def ProfitRate : ℝ := 0.1
def Profit : ℝ := 25000

-- State the problem as a Lean 4 theorem
theorem Reema_loan_problem (R : ℝ) (Investment : ℝ) : 
  Principal * (R / 100) * R = Interest ∧ 
  Profit = Investment * ProfitRate * R ∧ 
  R = 5 ∧ 
  Investment = 50000 :=
by
  sorry

end Reema_loan_problem_l365_36517


namespace proof_problem_l365_36563

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, f x + x * g x = x ^ 2 - 1
axiom condition2 : f 1 = 1

theorem proof_problem : deriv f 1 + deriv g 1 = 3 :=
by
  sorry

end proof_problem_l365_36563


namespace relation_between_a_b_l365_36590

variables {x y a b : ℝ}

theorem relation_between_a_b 
  (h1 : a = (x^2 + y^2) * (x - y))
  (h2 : b = (x^2 - y^2) * (x + y))
  (h3 : x < y) 
  (h4 : y < 0) : 
  a > b := 
by sorry

end relation_between_a_b_l365_36590


namespace impossible_to_reduce_time_l365_36523

def current_speed := 60 -- speed in km/h
def time_per_km (v : ℕ) : ℕ := 60 / v -- 60 minutes divided by speed in km/h gives time per km in minutes

theorem impossible_to_reduce_time (v : ℕ) (h : v = current_speed) : time_per_km v = 1 → ¬(time_per_km v - 1 = 0) :=
by
  intros h1 h2
  sorry

end impossible_to_reduce_time_l365_36523


namespace ab_equals_one_l365_36503

theorem ab_equals_one (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (f : ℝ → ℝ) (h3 : f = abs ∘ log) (h4 : f a = f b) : a * b = 1 :=
by
  sorry

end ab_equals_one_l365_36503


namespace wrapping_paper_area_correct_l365_36502

-- Conditions as given in the problem
variables (l w h : ℝ)
variable (hlw : l > w)

-- Definition of the area of the wrapping paper
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  (l + 2 * h) * (w + 2 * h)

-- Proof statement
theorem wrapping_paper_area_correct (hlw : l > w) : 
  wrapping_paper_area l w h = l * w + 2 * l * h + 2 * w * h + 4 * h^2 :=
by
  sorry

end wrapping_paper_area_correct_l365_36502


namespace summer_discount_percentage_l365_36536

/--
Given:
1. The original cost of the jeans (original_price) is $49.
2. On Wednesdays, there is an additional $10.00 off on all jeans after the summer discount is applied.
3. Before the sales tax applies, the cost of a pair of jeans (final_price) is $14.50.

Prove:
The summer discount percentage (D) is 50%.
-/
theorem summer_discount_percentage (original_price final_price : ℝ) (D : ℝ) :
  original_price = 49 → 
  final_price = 14.50 → 
  (original_price - (original_price * D / 100) - 10 = final_price) → 
  D = 50 :=
by intros h_original h_final h_discount; sorry

end summer_discount_percentage_l365_36536


namespace apples_to_grapes_proof_l365_36515

theorem apples_to_grapes_proof :
  (3 / 4 * 12 = 9) → (1 / 3 * 9 = 3) :=
by
  sorry

end apples_to_grapes_proof_l365_36515


namespace inscribed_circle_radius_l365_36551

theorem inscribed_circle_radius 
  (A : ℝ) -- Area of the triangle
  (p : ℝ) -- Perimeter of the triangle
  (r : ℝ) -- Radius of the inscribed circle
  (s : ℝ) -- Semiperimeter of the triangle
  (h1 : A = 2 * p) -- Condition: Area is numerically equal to twice the perimeter
  (h2 : p = 2 * s) -- Perimeter is twice the semiperimeter
  (h3 : A = r * s) -- Formula: Area in terms of inradius and semiperimeter
  (h4 : s ≠ 0) -- Semiperimeter is non-zero
  : r = 4 := 
sorry

end inscribed_circle_radius_l365_36551


namespace det_matrix_A_l365_36576

noncomputable def matrix_A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![z, x, y], ![y, z, x]]

theorem det_matrix_A (x y z : ℝ) : 
  Matrix.det (matrix_A x y z) = x^3 + y^3 + z^3 - 3*x*y*z := by
  sorry

end det_matrix_A_l365_36576


namespace coefficient_square_sum_l365_36559

theorem coefficient_square_sum (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1728 * x ^ 3 + 64 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by
  sorry

end coefficient_square_sum_l365_36559
