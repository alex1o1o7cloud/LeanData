import Mathlib

namespace vector_magnitude_sum_l975_97504

noncomputable def magnitude_sum (a b : ℝ) (θ : ℝ) := by
  let dot_product := a * b * Real.cos θ
  let a_square := a ^ 2
  let b_square := b ^ 2
  let magnitude := Real.sqrt (a_square + 2 * dot_product + b_square)
  exact magnitude

theorem vector_magnitude_sum (a b : ℝ) (θ : ℝ)
  (ha : a = 2) (hb : b = 1) (hθ : θ = Real.pi / 4) :
  magnitude_sum a b θ = Real.sqrt (5 + 2 * Real.sqrt 2) := by
  rw [ha, hb, hθ, magnitude_sum]
  sorry

end vector_magnitude_sum_l975_97504


namespace average_speed_round_trip_l975_97562

theorem average_speed_round_trip (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (2 * m * n) / (m + n) = (2 * (m * n)) / (m + n) :=
  sorry

end average_speed_round_trip_l975_97562


namespace deanna_initial_speed_l975_97516

namespace TripSpeed

variables (v : ℝ) (h : v > 0)

def speed_equation (v : ℝ) : Prop :=
  (1/2 * v) + (1/2 * (v + 20)) = 100

theorem deanna_initial_speed (v : ℝ) (h : speed_equation v) : v = 90 := sorry

end TripSpeed

end deanna_initial_speed_l975_97516


namespace sum_of_remainders_l975_97532

theorem sum_of_remainders (n : ℤ) (h : n % 15 = 7) : 
  (n % 3) + (n % 5) = 3 := 
by
  -- the proof will go here
  sorry

end sum_of_remainders_l975_97532


namespace abcdeq_five_l975_97555

theorem abcdeq_five (a b c d : ℝ) 
    (h1 : a + b + c + d = 20) 
    (h2 : ab + ac + ad + bc + bd + cd = 150) : 
    a = 5 ∧ b = 5 ∧ c = 5 ∧ d = 5 := 
  by
  sorry

end abcdeq_five_l975_97555


namespace rate_of_first_batch_l975_97500

theorem rate_of_first_batch (x : ℝ) 
  (cost_second_batch : ℝ := 20 * 14.25)
  (total_cost : ℝ := 30 * x + 285)
  (weight_mixture : ℝ := 30 + 20)
  (selling_price_per_kg : ℝ := 15.12) :
  (total_cost * 1.20 / weight_mixture = selling_price_per_kg) → x = 11.50 :=
by
  sorry

end rate_of_first_batch_l975_97500


namespace transitiveSim_l975_97543

def isGreat (f : ℕ × ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, f (m + 1, n + 1) * f (m, n) - f (m + 1, n) * f (m, n + 1) = 1

def seqSim (A B : ℕ → ℤ) : Prop :=
  ∃ f : ℕ × ℕ → ℤ, isGreat f ∧ (∀ n, f (n, 0) = A n) ∧ (∀ n, f (0, n) = B n)

theorem transitiveSim (A B C D : ℕ → ℤ)
  (h1 : seqSim A B)
  (h2 : seqSim B C)
  (h3 : seqSim C D) : seqSim D A :=
sorry

end transitiveSim_l975_97543


namespace calculate_expression_l975_97536

theorem calculate_expression :
  5 * 6 - 2 * 3 + 7 * 4 + 9 * 2 = 70 := by
  sorry

end calculate_expression_l975_97536


namespace range_of_a_l975_97550

theorem range_of_a (m : ℝ) (a : ℝ) (hx : ∃ x : ℝ, mx^2 + x - m - a = 0) : -1 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l975_97550


namespace find_m_l975_97553

open Real

noncomputable def x_values : List ℝ := [1, 3, 4, 5, 7]
noncomputable def y_values (m : ℝ) : List ℝ := [1, m, 2 * m + 1, 2 * m + 3, 10]

noncomputable def mean (l : List ℝ) : ℝ :=
l.sum / l.length

theorem find_m (m : ℝ) :
  mean x_values = 4 →
  mean (y_values m) = m + 3 →
  (1.3 * 4 + 0.8 = m + 3) →
  m = 3 :=
by
  intros h1 h2 h3
  sorry

end find_m_l975_97553


namespace quadratic_complex_inequality_solution_l975_97535
noncomputable def quadratic_inequality_solution (x : ℝ) : Prop :=
  (x^2 / (x + 2) ≥ 3 / (x - 2) + 7/4) ↔ -2 < x ∧ x < 2 ∨ 3 ≤ x

theorem quadratic_complex_inequality_solution (x : ℝ) (hx : x ≠ -2 ∧ x ≠ 2):
  quadratic_inequality_solution x :=
  sorry

end quadratic_complex_inequality_solution_l975_97535


namespace price_per_glass_first_day_l975_97509

theorem price_per_glass_first_day 
(O G : ℝ) (H : 2 * O * G * P₁ = 3 * O * G * 0.5466666666666666 ) : 
  P₁ = 0.82 :=
by
  sorry

end price_per_glass_first_day_l975_97509


namespace fixed_point_of_inverse_l975_97548

-- Define an odd function f on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f (x)

-- Define the transformed function g
def g (f : ℝ → ℝ) (x : ℝ) := f (x + 1) - 2

-- Define the condition for a point to be on the inverse of a function
def inv_contains (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = f p.1

-- The theorem statement
theorem fixed_point_of_inverse (f : ℝ → ℝ) 
  (Hf_odd : odd_function f) :
  inv_contains (λ y => g f (y)) (-2, -1) :=
sorry

end fixed_point_of_inverse_l975_97548


namespace simplify_expression_l975_97501

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end simplify_expression_l975_97501


namespace max_integer_k_l975_97540

-- First, define the sequence a_n
def a (n : ℕ) : ℕ := n + 5

-- Define the sequence b_n given the recurrence relation and initial condition
def b (n : ℕ) : ℕ := 3 * n + 2

-- Define the sequence c_n
def c (n : ℕ) : ℚ := 3 / ((2 * a n - 11) * (2 * b n - 1))

-- Define the sum T_n of the first n terms of the sequence c_n
def T (n : ℕ) : ℚ := (1 / 2) * (1 - (1 / (2 * n + 1)))

-- The theorem to prove
theorem max_integer_k :
  ∃ k : ℕ, ∀ n : ℕ, n > 0 → T n > (k : ℚ) / 57 ∧ k = 18 :=
by
  sorry

end max_integer_k_l975_97540


namespace find_a_b_l975_97517

theorem find_a_b (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a * x^2 + b) 
  (h2 : ∀ x, f' x = 3 * x^2 + 2 * a * x) 
  (h3 : f' 1 = -3) 
  (h4 : f 1 = 0) : 
  a = -3 ∧ b = 2 := 
by
  sorry

end find_a_b_l975_97517


namespace find_number_l975_97582

noncomputable def N : ℕ :=
  76

theorem find_number :
  (N % 13 = 11) ∧ (N % 17 = 9) :=
by
  -- These are the conditions translated to Lean 4, as stated:
  have h1 : N % 13 = 11 := by sorry
  have h2 : N % 17 = 9 := by sorry
  exact ⟨h1, h2⟩

end find_number_l975_97582


namespace largest_coefficient_term_in_expansion_l975_97594

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_coefficient_term_in_expansion :
  ∃ (T : ℕ × ℤ × ℕ), 
  (2 : ℤ) ^ (14 - 1) = 8192 ∧ 
  T = (binom 14 4, 2 ^ 10, 4) ∧ 
  ∀ (k : ℕ), 
    (binom 14 k * (2 ^ (14 - k))) ≤ (binom 14 4 * 2 ^ 10) :=
sorry

end largest_coefficient_term_in_expansion_l975_97594


namespace Bill_composes_20_problems_l975_97557

theorem Bill_composes_20_problems :
  ∀ (B : ℕ), (∀ R : ℕ, R = 2 * B) →
    (∀ F : ℕ, F = 3 * R) →
    (∀ T : ℕ, T = 4) →
    (∀ P : ℕ, P = 30) →
    (∀ F : ℕ, F = T * P) →
    (∃ B : ℕ, B = 20) :=
by sorry

end Bill_composes_20_problems_l975_97557


namespace perimeter_of_square_is_160_cm_l975_97596

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_of_square (area_of_rectangle : ℝ) : ℝ := 5 * area_of_rectangle

noncomputable def side_length_of_square (area_of_square : ℝ) : ℝ := Real.sqrt area_of_square

noncomputable def perimeter_of_square (side_length : ℝ) : ℝ := 4 * side_length

theorem perimeter_of_square_is_160_cm :
  perimeter_of_square (side_length_of_square (area_of_square (area_of_rectangle 32 10))) = 160 :=
by
  sorry

end perimeter_of_square_is_160_cm_l975_97596


namespace trigonometric_values_l975_97595

variable (α : ℝ)

theorem trigonometric_values (h : Real.cos (3 * Real.pi + α) = 3 / 5) :
  Real.cos α = -3 / 5 ∧
  Real.cos (Real.pi + α) = 3 / 5 ∧
  Real.sin (3 * Real.pi / 2 - α) = -3 / 5 :=
by
  sorry

end trigonometric_values_l975_97595


namespace common_divisors_count_l975_97574

def prime_exponents (n : Nat) : List (Nat × Nat) :=
  if n = 9240 then [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
  else if n = 10800 then [(2, 4), (3, 3), (5, 2)]
  else []

def gcd_prime_exponents (exps1 exps2 : List (Nat × Nat)) : List (Nat × Nat) :=
  exps1.filterMap (fun (p1, e1) =>
    match exps2.find? (fun (p2, _) => p1 = p2) with
    | some (p2, e2) => if e1 ≤ e2 then some (p1, e1) else some (p1, e2)
    | none => none
  )

def count_divisors (exps : List (Nat × Nat)) : Nat :=
  exps.foldl (fun acc (_, e) => acc * (e + 1)) 1

theorem common_divisors_count :
  count_divisors (gcd_prime_exponents (prime_exponents 9240) (prime_exponents 10800)) = 16 :=
by
  sorry

end common_divisors_count_l975_97574


namespace polynomial_equivalence_l975_97506

-- Define the polynomial 'A' according to the conditions provided
def polynomial_A (x : ℝ) : ℝ := x^2 - 2*x

-- Define the given equation with polynomial A
def given_equation (x : ℝ) (A : ℝ) : Prop :=
  (x / (x + 2)) = (A / (x^2 - 4))

-- Prove that for the given equation, the polynomial 'A' is 'x^2 - 2x'
theorem polynomial_equivalence (x : ℝ) : given_equation x (polynomial_A x) :=
  by
    sorry -- Proof is skipped

end polynomial_equivalence_l975_97506


namespace rectangle_length_eq_fifty_l975_97546

theorem rectangle_length_eq_fifty (x : ℝ) :
  (∃ w : ℝ, 6 * x * w = 6000 ∧ w = (2 / 5) * x) → x = 50 :=
by
  sorry

end rectangle_length_eq_fifty_l975_97546


namespace time_for_c_to_finish_alone_l975_97572

variable (A B C : ℚ) -- A, B, and C are the work rates

theorem time_for_c_to_finish_alone :
  (A + B = 1/3) →
  (B + C = 1/4) →
  (C + A = 1/6) →
  1/C = 24 := 
by
  intros h1 h2 h3
  sorry

end time_for_c_to_finish_alone_l975_97572


namespace line_m_eq_line_n_eq_l975_97513
-- Definitions for conditions
def point_A : ℝ × ℝ := (-2, 1)
def line_l (x y : ℝ) := 2 * x - y - 3 = 0

-- Proof statement for part (1)
theorem line_m_eq :
  ∃ (m : ℝ → ℝ → Prop), (∀ x y, m x y ↔ (2 * x - y + 5 = 0)) ∧
    (∀ x y, line_l x y → m (-2) 1 → True) :=
sorry

-- Proof statement for part (2)
theorem line_n_eq :
  ∃ (n : ℝ → ℝ → Prop), (∀ x y, n x y ↔ (x + 2 * y = 0)) ∧
    (∀ x y, line_l x y → n (-2) 1 → True) :=
sorry

end line_m_eq_line_n_eq_l975_97513


namespace min_value_expression_l975_97588

theorem min_value_expression (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a + b + c = 1) :
  9 ≤ (1 / (a^2 + 2 * b^2)) + (1 / (b^2 + 2 * c^2)) + (1 / (c^2 + 2 * a^2)) :=
by
  sorry

end min_value_expression_l975_97588


namespace sin_cos_quad_ineq_l975_97561

open Real

theorem sin_cos_quad_ineq (x : ℝ) : 
  2 * (sin x) ^ 4 + 3 * (sin x) ^ 2 * (cos x) ^ 2 + 5 * (cos x) ^ 4 ≤ 5 :=
by
  sorry

end sin_cos_quad_ineq_l975_97561


namespace olivia_nigel_remaining_money_l975_97547

theorem olivia_nigel_remaining_money :
  let olivia_money := 112
  let nigel_money := 139
  let ticket_count := 6
  let ticket_price := 28
  let total_money := olivia_money + nigel_money
  let total_cost := ticket_count * ticket_price
  total_money - total_cost = 83 := 
by 
  sorry

end olivia_nigel_remaining_money_l975_97547


namespace real_values_x_l975_97549

theorem real_values_x (x y : ℝ) :
  (3 * y^2 + 5 * x * y + x + 7 = 0) →
  (5 * x + 6) * (5 * x - 14) ≥ 0 →
  x ≤ -6 / 5 ∨ x ≥ 14 / 5 :=
by
  sorry

end real_values_x_l975_97549


namespace commission_rate_change_amount_l975_97510

theorem commission_rate_change_amount :
  ∃ X : ℝ, (∀ S : ℝ, ∀ commission : ℝ, S = 15885.42 → commission = (S - 15000) →
  commission = 0.10 * X + 0.05 * (S - X) → X = 1822.98) :=
sorry

end commission_rate_change_amount_l975_97510


namespace zachary_pushups_l975_97579

variable (Zachary David John : ℕ)
variable (h1 : David = Zachary + 39)
variable (h2 : John = David - 13)
variable (h3 : David = 58)

theorem zachary_pushups : Zachary = 19 :=
by
  -- Proof goes here
  sorry

end zachary_pushups_l975_97579


namespace product_without_zero_digits_l975_97537

def no_zero_digits (n : ℕ) : Prop :=
  ¬ ∃ d : ℕ, d ∈ n.digits 10 ∧ d = 0

theorem product_without_zero_digits :
  ∃ a b : ℕ, a * b = 1000000000 ∧ no_zero_digits a ∧ no_zero_digits b :=
by
  sorry

end product_without_zero_digits_l975_97537


namespace handshakes_at_convention_l975_97556

theorem handshakes_at_convention :
  let gremlins := 30
  let imps := 15
  let handshakes_among_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_between_imps_gremlins := imps * (gremlins / 2)
  handshakes_among_gremlins + handshakes_between_imps_gremlins = 660 :=
by
  let gremlins := 30
  let imps := 15
  let handshakes_among_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_between_imps_gremlins := imps * (gremlins / 2)
  show handshakes_among_gremlins + handshakes_between_imps_gremlins = 660
  sorry

end handshakes_at_convention_l975_97556


namespace handshake_remainder_l975_97580

noncomputable def handshakes (n : ℕ) (k : ℕ) : ℕ := sorry

theorem handshake_remainder :
  handshakes 12 3 % 1000 = 850 :=
sorry

end handshake_remainder_l975_97580


namespace smaller_number_is_476_l975_97566

theorem smaller_number_is_476 (x y : ℕ) 
  (h1 : y - x = 2395) 
  (h2 : y = 6 * x + 15) : 
  x = 476 := 
by 
  sorry

end smaller_number_is_476_l975_97566


namespace tilde_tilde_tilde_47_l975_97599

def tilde (N : ℝ) : ℝ := 0.4 * N + 2

theorem tilde_tilde_tilde_47 : tilde (tilde (tilde 47)) = 6.128 := 
by
  sorry

end tilde_tilde_tilde_47_l975_97599


namespace caleb_spent_more_on_ice_cream_l975_97570

theorem caleb_spent_more_on_ice_cream :
  let num_ic_cream := 10
  let cost_ic_cream := 4
  let num_frozen_yog := 4
  let cost_frozen_yog := 1
  (num_ic_cream * cost_ic_cream - num_frozen_yog * cost_frozen_yog) = 36 := 
by
  sorry

end caleb_spent_more_on_ice_cream_l975_97570


namespace no_a_for_x4_l975_97515

theorem no_a_for_x4 : ∃ a : ℝ, (1 / (4 + a) + 1 / (4 - a) = 1 / (4 - a)) → false :=
  by sorry

end no_a_for_x4_l975_97515


namespace expression_is_integer_l975_97560

theorem expression_is_integer (n : ℕ) : 
    ∃ k : ℤ, (n^5 : ℤ) / 5 + (n^3 : ℤ) / 3 + (7 * n : ℤ) / 15 = k :=
by
  sorry

end expression_is_integer_l975_97560


namespace train_length_l975_97551

noncomputable def length_of_each_train (L : ℝ) : Prop :=
  let v1 := 46 -- speed of faster train in km/hr
  let v2 := 36 -- speed of slower train in km/hr
  let relative_speed := (v1 - v2) * (5/18) -- converting relative speed to m/s
  let time := 72 -- time in seconds
  2 * L = relative_speed * time -- distance equation

theorem train_length : ∃ (L : ℝ), length_of_each_train L ∧ L = 100 :=
by
  use 100
  unfold length_of_each_train
  sorry

end train_length_l975_97551


namespace simplify_trig_expression_l975_97502

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem simplify_trig_expression :
  (tan (20 * Real.pi / 180) + tan (30 * Real.pi / 180) + tan (60 * Real.pi / 180) + tan (70 * Real.pi / 180)) / Real.sin (10 * Real.pi / 180) =
  1 / (2 * Real.sin (10 * Real.pi / 180) ^ 2 * Real.cos (20 * Real.pi / 180)) + 4 / (Real.sqrt 3 * Real.sin (10 * Real.pi / 180)) :=
by
  sorry

end simplify_trig_expression_l975_97502


namespace tan_alpha_l975_97539

theorem tan_alpha (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 1 / 5) : Real.tan α = -2 / 3 :=
by
  sorry

end tan_alpha_l975_97539


namespace max_n_is_4024_l975_97598

noncomputable def max_n_for_positive_sum (a : ℕ → ℝ) (d : ℝ) (h1 : d < 0) (h2 : a 1 > 0) (h3 : a 2013 * (a 2012 + a 2013) < 0) : ℕ :=
  4024

theorem max_n_is_4024 (a : ℕ → ℝ) (d : ℝ) (h1 : d < 0) (h2 : a 1 > 0) (h3 : a 2013 * (a 2012 + a 2013) < 0) :
  max_n_for_positive_sum a d h1 h2 h3 = 4024 :=
by
  sorry

end max_n_is_4024_l975_97598


namespace boxes_with_neither_l975_97507

-- Definitions for conditions
def total_boxes := 15
def boxes_with_crayons := 9
def boxes_with_markers := 5
def boxes_with_both := 4

-- Theorem statement
theorem boxes_with_neither :
  total_boxes - (boxes_with_crayons + boxes_with_markers - boxes_with_both) = 5 :=
by
  sorry

end boxes_with_neither_l975_97507


namespace knights_divisible_by_4_l975_97591

-- Define the conditions: Assume n is the total number of knights (n > 0).
-- Condition 1: Knights from two opposing clans A and B
-- Condition 2: Number of knights with an enemy to the right equals number of knights with a friend to the right.

open Nat

theorem knights_divisible_by_4 (n : ℕ) (h1 : 0 < n)
  (h2 : ∃k : ℕ, 2 * k = n ∧ ∀ (i : ℕ), (i < n → ((i % 2 = 0 → (i+1) % 2 = 1) ∧ (i % 2 = 1 → (i+1) % 2 = 0)))) :
  n % 4 = 0 :=
sorry

end knights_divisible_by_4_l975_97591


namespace jackson_investment_ratio_l975_97505

theorem jackson_investment_ratio:
  ∀ (B J: ℝ), B = 0.20 * 500 → J = B + 1900 → (J / 500) = 4 :=
by
  intros B J hB hJ
  sorry

end jackson_investment_ratio_l975_97505


namespace units_digit_difference_l975_97544

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_difference :
  units_digit (72^3) - units_digit (24^3) = 4 :=
by
  sorry

end units_digit_difference_l975_97544


namespace xy_product_solution_l975_97573

theorem xy_product_solution (x y : ℝ)
  (h1 : x / (x^2 * y^2 - 1) - 1 / x = 4)
  (h2 : (x^2 * y) / (x^2 * y^2 - 1) + y = 2) :
  x * y = 1 / Real.sqrt 2 ∨ x * y = -1 / Real.sqrt 2 :=
sorry

end xy_product_solution_l975_97573


namespace money_distribution_l975_97533

theorem money_distribution (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 310) (h3 : C = 10) : A + B + C = 500 :=
by
  sorry

end money_distribution_l975_97533


namespace extreme_value_f_range_of_a_l975_97581

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x^2 + a * x - 3
noncomputable def h (x : ℝ) : ℝ := 2 * Real.log x + x + 3 / x

theorem extreme_value_f : ∃ x, f x = -1 / Real.exp 1 :=
by sorry

theorem range_of_a (a : ℝ) : (∀ x > 0, 2 * f x ≥ g x a) → a ≤ 4 :=
by sorry

end extreme_value_f_range_of_a_l975_97581


namespace sin_cos_fraction_eq_two_l975_97541

theorem sin_cos_fraction_eq_two (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 :=
sorry

end sin_cos_fraction_eq_two_l975_97541


namespace prove_values_l975_97589

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 1/x + b

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem prove_values (a b : ℝ) (h1 : a > 0) (h2 : is_integer b) :
  (f a b (Real.log a) = 6 ∧ f a b (Real.log (1 / a)) = 2) ∨
  (f a b (Real.log a) = -2 ∧ f a b (Real.log (1 / a)) = 2) :=
sorry

end prove_values_l975_97589


namespace ellipse_parameters_sum_l975_97590

def ellipse_sum (h k a b : ℝ) : ℝ :=
  h + k + a + b

theorem ellipse_parameters_sum :
  let h := 5
  let k := -3
  let a := 7
  let b := 4
  ellipse_sum h k a b = 13 := by
  sorry

end ellipse_parameters_sum_l975_97590


namespace cut_into_two_pieces_is_possible_cut_into_three_pieces_is_impossible_cut_into_four_pieces_is_possible_cut_into_five_pieces_is_impossible_l975_97520

-- Definitions based on the conditions:
-- 1. Folded napkin structure
structure Napkin where
  folded_in_two: Bool -- A napkin folded in half once along one axis 
  folded_in_four: Bool -- A napkin folded in half twice to form a smaller square

-- 2. Cutting through a folded napkin
def single_cut_through_folded_napkin (n: Nat) (napkin: Napkin) : Bool :=
  if (n = 2 ∨ n = 4) then
    true
  else
    false

-- Main theorem statements 
-- If the napkin can be cut into 2 pieces
theorem cut_into_two_pieces_is_possible (napkin: Napkin) : single_cut_through_folded_napkin 2 napkin = true := by
  sorry

-- If the napkin can be cut into 3 pieces
theorem cut_into_three_pieces_is_impossible (napkin: Napkin) : single_cut_through_folded_napkin 3 napkin = false := by
  sorry

-- If the napkin can be cut into 4 pieces
theorem cut_into_four_pieces_is_possible (napkin: Napkin) : single_cut_through_folded_napkin 4 napkin = true := by
  sorry

-- If the napkin can be cut into 5 pieces
theorem cut_into_five_pieces_is_impossible (napkin: Napkin) : single_cut_through_folded_napkin 5 napkin = false := by
  sorry

end cut_into_two_pieces_is_possible_cut_into_three_pieces_is_impossible_cut_into_four_pieces_is_possible_cut_into_five_pieces_is_impossible_l975_97520


namespace find_x_l975_97565

-- Let \( x \) be a real number such that 
-- \( x = 2 \left( \frac{1}{x} \cdot (-x) \right) - 5 \).
-- Prove \( x = -7 \).

theorem find_x (x : ℝ) (h : x = 2 * (1 / x * (-x)) - 5) : x = -7 :=
by
  sorry

end find_x_l975_97565


namespace pool_water_after_45_days_l975_97554

-- Defining the initial conditions and the problem statement in Lean
noncomputable def initial_amount : ℝ := 500
noncomputable def evaporation_rate : ℝ := 0.7
noncomputable def addition_rate : ℝ := 5
noncomputable def total_days : ℕ := 45

noncomputable def final_amount : ℝ :=
  initial_amount - (evaporation_rate * total_days) +
  (addition_rate * (total_days / 3))

theorem pool_water_after_45_days : final_amount = 543.5 :=
by
  -- Inserting the proof is not required here
  sorry

end pool_water_after_45_days_l975_97554


namespace proof_triangle_tangent_l975_97559

open Real

def isCongruentAngles (ω : ℝ) := 
  let a := 15
  let b := 18
  let c := 21
  ∃ (x y z : ℝ), 
  (y^2 = x^2 + a^2 - 2 * a * x * cos ω) 
  ∧ (z^2 = y^2 + b^2 - 2 * b * y * cos ω)
  ∧ (x^2 = z^2 + c^2 - 2 * c * z * cos ω)

def isTriangleABCWithSides (AB BC CA : ℝ) (ω : ℝ) (tan_ω : ℝ) : Prop := 
  (AB = 15) ∧ (BC = 18) ∧ (CA = 21) ∧ isCongruentAngles ω 
  ∧ tan ω = tan_ω

theorem proof_triangle_tangent : isTriangleABCWithSides 15 18 21 ω (88/165) := 
by
  sorry

end proof_triangle_tangent_l975_97559


namespace total_weight_loss_l975_97542

theorem total_weight_loss (S J V : ℝ) 
  (hS : S = 17.5) 
  (hJ : J = 3 * S) 
  (hV : V = S + 1.5) : 
  S + J + V = 89 := 
by 
  sorry

end total_weight_loss_l975_97542


namespace half_angle_quadrant_l975_97597

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end half_angle_quadrant_l975_97597


namespace radius_of_inscribed_circle_in_COD_l975_97587

theorem radius_of_inscribed_circle_in_COD
  (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ)
  (H1 : r1 = 6)
  (H2 : r2 = 2)
  (H3 : r3 = 1.5)
  (H4 : 1/r1 + 1/r3 = 1/r2 + 1/r4) :
  r4 = 3 :=
by
  sorry

end radius_of_inscribed_circle_in_COD_l975_97587


namespace tan_C_in_triangle_l975_97521

theorem tan_C_in_triangle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : Real.tan A = 1) (h₃ : Real.tan B = 2) :
  Real.tan C = 3 :=
sorry

end tan_C_in_triangle_l975_97521


namespace hyperbola_equation_l975_97563

theorem hyperbola_equation (c : ℝ) (b a : ℝ) 
  (h₁ : c = 2 * Real.sqrt 5) 
  (h₂ : a^2 + b^2 = c^2) 
  (h₃ : b / a = 1 / 2) : 
  (x y : ℝ) → (x^2 / 16) - (y^2 / 4) = 1 :=
by
  sorry

end hyperbola_equation_l975_97563


namespace sum_of_excluded_values_l975_97534

theorem sum_of_excluded_values (C D : ℝ) (h₁ : 2 * C^2 - 8 * C + 6 = 0)
    (h₂ : 2 * D^2 - 8 * D + 6 = 0) (h₃ : C ≠ D) :
    C + D = 4 :=
sorry

end sum_of_excluded_values_l975_97534


namespace cost_of_eight_memory_cards_l975_97523

theorem cost_of_eight_memory_cards (total_cost_of_three: ℕ) (h: total_cost_of_three = 45) : 8 * (total_cost_of_three / 3) = 120 := by
  sorry

end cost_of_eight_memory_cards_l975_97523


namespace segments_not_arrangeable_l975_97503

theorem segments_not_arrangeable :
  ¬∃ (segments : ℕ → (ℝ × ℝ) × (ℝ × ℝ)), 
    (∀ i, 0 ≤ i → i < 1000 → 
      ∃ j, 0 ≤ j → j < 1000 → 
        i ≠ j ∧
        (segments i).fst.1 > (segments j).fst.1 ∧
        (segments i).fst.2 < (segments j).snd.2 ∧
        (segments i).snd.1 > (segments j).fst.1 ∧
        (segments i).snd.2 < (segments j).snd.2) :=
by
  sorry

end segments_not_arrangeable_l975_97503


namespace jack_turn_in_correct_amount_l975_97519

-- Definition of the conditions
def exchange_rate_euro : ℝ := 1.18
def exchange_rate_pound : ℝ := 1.39

def till_usd_total : ℝ := (2 * 100) + (1 * 50) + (5 * 20) + (3 * 10) + (7 * 5) + (27 * 1) + (42 * 0.25) + (19 * 0.1) + (36 * 0.05) + (47 * 0.01)
def till_euro_total : ℝ := 20 * 5
def till_pound_total : ℝ := 25 * 10

def till_usd : ℝ := till_usd_total + (till_euro_total * exchange_rate_euro) + (till_pound_total * exchange_rate_pound)

def leave_in_till_notes : ℝ := 300
def leave_in_till_coins : ℝ := (42 * 0.25) + (19 * 0.1) + (36 * 0.05) + (47 * 0.01)
def leave_in_till_total : ℝ := leave_in_till_notes + leave_in_till_coins

def turn_in_to_office : ℝ := till_usd - leave_in_till_total

theorem jack_turn_in_correct_amount : turn_in_to_office = 607.50 := by
  sorry

end jack_turn_in_correct_amount_l975_97519


namespace company_pays_240_per_month_l975_97571

-- Conditions as definitions
def box_length : ℕ := 15
def box_width : ℕ := 12
def box_height : ℕ := 10
def total_volume : ℕ := 1080000      -- 1.08 million cubic inches
def price_per_box_per_month : ℚ := 0.4

-- The volume of one box
def box_volume : ℕ := box_length * box_width * box_height

-- Calculate the number of boxes
def number_of_boxes : ℕ := total_volume / box_volume

-- Total amount paid per month for record storage
def total_amount_paid_per_month : ℚ := number_of_boxes * price_per_box_per_month

-- Theorem statement to prove
theorem company_pays_240_per_month : total_amount_paid_per_month = 240 := 
by 
  sorry

end company_pays_240_per_month_l975_97571


namespace Nicky_pace_5_mps_l975_97545

/-- Given the conditions:
  - Cristina runs at a pace of 5 meters per second.
  - Nicky runs for 30 seconds before Cristina catches up to him.
  Prove that Nicky’s pace is 5 meters per second. -/
theorem Nicky_pace_5_mps
  (Cristina_pace : ℝ)
  (time_Nicky : ℝ)
  (catchup : Cristina_pace * time_Nicky = 150)
  (def_Cristina_pace : Cristina_pace = 5)
  (def_time_Nicky : time_Nicky = 30) :
  (150 / 30) = 5 :=
by
  sorry

end Nicky_pace_5_mps_l975_97545


namespace sally_found_more_balloons_l975_97538

def sally_original_balloons : ℝ := 9.0
def sally_new_balloons : ℝ := 11.0

theorem sally_found_more_balloons :
  sally_new_balloons - sally_original_balloons = 2.0 :=
by
  -- math proof goes here
  sorry

end sally_found_more_balloons_l975_97538


namespace train_length_correct_l975_97527

noncomputable def train_length (speed_kmph: ℝ) (time_sec: ℝ) : ℝ :=
  let speed_mps := speed_kmph * (5 / 18)
  speed_mps * time_sec

theorem train_length_correct : train_length 250 12 = 833.28 := by
  sorry

end train_length_correct_l975_97527


namespace children_count_l975_97518

theorem children_count (C A : ℕ) (h1 : 15 * A + 8 * C = 720) (h2 : A = C + 25) : C = 15 := 
by
  sorry

end children_count_l975_97518


namespace original_number_l975_97586

theorem original_number (sum_orig : ℕ) (sum_new : ℕ) (changed_value : ℕ) (avg_orig : ℕ) (avg_new : ℕ) (n : ℕ) :
    sum_orig = n * avg_orig →
    sum_new = sum_orig - changed_value + 9 →
    avg_new = 8 →
    avg_orig = 7 →
    n = 7 →
    sum_new = n * avg_new →
    changed_value = 2 := 
by
  sorry

end original_number_l975_97586


namespace card_probability_l975_97575

-- Define the total number of cards
def total_cards : ℕ := 52

-- Define the number of Kings in the deck
def kings_in_deck : ℕ := 4

-- Define the number of Aces in the deck
def aces_in_deck : ℕ := 4

-- Define the probability of the top card being a King
def prob_top_king : ℚ := kings_in_deck / total_cards

-- Define the probability of the second card being an Ace given the first card is a King
def prob_second_ace_given_king : ℚ := aces_in_deck / (total_cards - 1)

-- Define the combined probability of both events happening in sequence
def combined_probability : ℚ := prob_top_king * prob_second_ace_given_king

-- Theorem statement that the combined probability is equal to 4/663
theorem card_probability : combined_probability = 4 / 663 := by
  -- Proof to be filled in
  sorry

end card_probability_l975_97575


namespace simplify_expression_l975_97524

theorem simplify_expression (x y : ℝ) : 2 - (3 - (2 + (5 - (3 * y - x)))) = 6 - 3 * y + x :=
by
  sorry

end simplify_expression_l975_97524


namespace zero_of_fn_exists_between_2_and_3_l975_97525

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 3 * x - 9

theorem zero_of_fn_exists_between_2_and_3 :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
sorry

end zero_of_fn_exists_between_2_and_3_l975_97525


namespace bond_value_after_8_years_l975_97529

theorem bond_value_after_8_years (r t1 t2 : ℕ) (A1 A2 P : ℚ) :
  r = 4 / 100 ∧ t1 = 3 ∧ t2 = 8 ∧ A1 = 560 ∧ A1 = P * (1 + r * t1) 
  → A2 = P * (1 + r * t2) ∧ A2 = 660 :=
by
  intro h
  obtain ⟨hr, ht1, ht2, hA1, hA1eq⟩ := h
  -- Proof needs to be filled in here
  sorry

end bond_value_after_8_years_l975_97529


namespace sparrow_pecks_seeds_l975_97522

theorem sparrow_pecks_seeds (x : ℕ) (h1 : 9 * x < 1001) (h2 : 10 * x > 1100) : x = 111 :=
by
  sorry

end sparrow_pecks_seeds_l975_97522


namespace annual_population_increase_l975_97592

theorem annual_population_increase 
  (P : ℕ) (A : ℕ) (t : ℕ) (r : ℚ)
  (hP : P = 10000)
  (hA : A = 14400)
  (ht : t = 2)
  (h_eq : A = P * (1 + r)^t) :
  r = 0.2 :=
by
  sorry

end annual_population_increase_l975_97592


namespace solve_speed_of_second_train_l975_97568

open Real

noncomputable def speed_of_second_train
  (L1 : ℝ) (L2 : ℝ) (S1 : ℝ) (T : ℝ) : ℝ :=
  let D := (L1 + L2) / 1000   -- Total distance in kilometers
  let H := T / 3600           -- Time in hours
  let relative_speed := D / H -- Relative speed in km/h
  relative_speed - S1         -- Speed of the second train

theorem solve_speed_of_second_train :
  speed_of_second_train 100 220 42 15.99872010239181 = 30 := by
  sorry

end solve_speed_of_second_train_l975_97568


namespace abs_eq_linear_eq_l975_97569

theorem abs_eq_linear_eq (x : ℝ) : (|x - 5| = 3 * x + 1) ↔ x = 1 := by
  sorry

end abs_eq_linear_eq_l975_97569


namespace initial_lives_l975_97526

theorem initial_lives (L : ℕ) (h1 : L - 6 + 37 = 41) : L = 10 :=
by
  sorry

end initial_lives_l975_97526


namespace power_sum_int_l975_97528

theorem power_sum_int {x : ℝ} (hx : ∃ k : ℤ, x + 1/x = k) : ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by
  sorry

end power_sum_int_l975_97528


namespace appears_in_31st_equation_l975_97593

theorem appears_in_31st_equation : 
  ∃ n : ℕ, 2016 ∈ {x | 2*x^2 ≤ 2016 ∧ 2016 < 2*(x+1)^2} ∧ n = 31 :=
by
  sorry

end appears_in_31st_equation_l975_97593


namespace geometric_sequence_common_ratio_l975_97531

theorem geometric_sequence_common_ratio
  (a₁ a₂ a₃ : ℝ) (q : ℝ) 
  (h₀ : 0 < a₁) 
  (h₁ : a₂ = a₁ * q) 
  (h₂ : a₃ = a₁ * q^2) 
  (h₃ : 2 * a₁ + a₂ = 2 * (1 / 2 * a₃)) 
  : q = 2 := 
sorry

end geometric_sequence_common_ratio_l975_97531


namespace alice_has_ball_after_three_turns_l975_97577

def alice_keeps_ball (prob_Alice_to_Bob: ℚ) (prob_Bob_to_Alice: ℚ): ℚ := 
  let prob_Alice_keeps := 1 - prob_Alice_to_Bob
  let prob_Bob_keeps := 1 - prob_Bob_to_Alice
  let path1 := prob_Alice_to_Bob * prob_Bob_to_Alice * prob_Alice_keeps
  let path2 := prob_Alice_keeps * prob_Alice_keeps * prob_Alice_keeps
  path1 + path2

theorem alice_has_ball_after_three_turns:
  alice_keeps_ball (1/2) (1/3) = 5/24 := 
by
  sorry

end alice_has_ball_after_three_turns_l975_97577


namespace alpha_plus_2beta_eq_45_l975_97512

theorem alpha_plus_2beta_eq_45 
  (α β : ℝ) 
  (hα_pos : 0 < α ∧ α < π / 2) 
  (hβ_pos : 0 < β ∧ β < π / 2) 
  (tan_alpha : Real.tan α = 1 / 7) 
  (sin_beta : Real.sin β = 1 / Real.sqrt 10)
  : α + 2 * β = π / 4 :=
sorry

end alpha_plus_2beta_eq_45_l975_97512


namespace moment_goal_equality_l975_97508

theorem moment_goal_equality (total_goals_russia total_goals_tunisia : ℕ) (T : total_goals_russia = 9) (T2 : total_goals_tunisia = 5) :
  ∃ n, n ≤ 9 ∧ (9 - n) = total_goals_tunisia :=
by
  sorry

end moment_goal_equality_l975_97508


namespace sum_of_reciprocals_of_roots_l975_97584

theorem sum_of_reciprocals_of_roots : 
  ∀ {r1 r2 : ℝ}, (r1 + r2 = 14) → (r1 * r2 = 6) → (1 / r1 + 1 / r2 = 7 / 3) :=
by
  intros r1 r2 h_sum h_product
  sorry

end sum_of_reciprocals_of_roots_l975_97584


namespace class_average_l975_97576

theorem class_average (p1 p2 p3 avg1 avg2 avg3 overall_avg : ℕ) 
  (h1 : p1 = 45) 
  (h2 : p2 = 50) 
  (h3 : p3 = 100 - p1 - p2) 
  (havg1 : avg1 = 95) 
  (havg2 : avg2 = 78) 
  (havg3 : avg3 = 60) 
  (hoverall : overall_avg = (p1 * avg1 + p2 * avg2 + p3 * avg3) / 100) : 
  overall_avg = 85 :=
by
  sorry

end class_average_l975_97576


namespace smallest_three_digit_number_satisfying_conditions_l975_97567

theorem smallest_three_digit_number_satisfying_conditions :
  ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n + 6) % 9 = 0 ∧ (n - 4) % 6 = 0 ∧ n = 112 :=
by
  -- Proof goes here
  sorry

end smallest_three_digit_number_satisfying_conditions_l975_97567


namespace min_value_is_1_5_l975_97514

noncomputable def min_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) : ℝ :=
  (1 : ℝ) / (a + b) + 
  (1 : ℝ) / (b + c) + 
  (1 : ℝ) / (c + a)

theorem min_value_is_1_5 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  min_value a b c h1 h2 h3 h4 = 1.5 :=
sorry

end min_value_is_1_5_l975_97514


namespace subtract_three_from_binary_l975_97552

theorem subtract_three_from_binary (M : ℕ) (M_binary: M = 0b10110000) : (M - 3) = 0b10101101 := by
  sorry

end subtract_three_from_binary_l975_97552


namespace complete_the_square_sum_l975_97585

theorem complete_the_square_sum :
  ∃ p q : ℝ, (∀ x : ℝ, 15 * x^2 - 30 * x - 60 = 0 → (x + p)^2 = q) ∧ p + q = 1 :=
by 
  sorry

end complete_the_square_sum_l975_97585


namespace total_loss_is_correct_l975_97558

-- Definitions for each item's purchase conditions
def paintings_cost : ℕ := 18 * 75
def toys_cost : ℕ := 25 * 30
def hats_cost : ℕ := 12 * 20
def wallets_cost : ℕ := 10 * 50
def mugs_cost : ℕ := 35 * 10

def paintings_loss_percentage : ℝ := 0.22
def toys_loss_percentage : ℝ := 0.27
def hats_loss_percentage : ℝ := 0.15
def wallets_loss_percentage : ℝ := 0.05
def mugs_loss_percentage : ℝ := 0.12

-- Calculation of loss on each item
def paintings_loss : ℝ := paintings_cost * paintings_loss_percentage
def toys_loss : ℝ := toys_cost * toys_loss_percentage
def hats_loss : ℝ := hats_cost * hats_loss_percentage
def wallets_loss : ℝ := wallets_cost * wallets_loss_percentage
def mugs_loss : ℝ := mugs_cost * mugs_loss_percentage

-- Total loss calculation
def total_loss : ℝ := paintings_loss + toys_loss + hats_loss + wallets_loss + mugs_loss

-- Lean statement to verify the total loss
theorem total_loss_is_correct : total_loss = 602.50 := by
  sorry

end total_loss_is_correct_l975_97558


namespace buses_trips_product_l975_97511

theorem buses_trips_product :
  ∃ (n k : ℕ), n > 3 ∧ n * (n - 1) * (2 * k - 1) = 600 ∧ (n * k = 52 ∨ n * k = 40) := 
by
  sorry

end buses_trips_product_l975_97511


namespace rectangle_difference_l975_97583

theorem rectangle_difference (L B D : ℝ)
  (h1 : L - B = D)
  (h2 : 2 * (L + B) = 186)
  (h3 : L * B = 2030) :
  D = 23 :=
by
  sorry

end rectangle_difference_l975_97583


namespace probability_xi_eq_1_l975_97564

-- Definitions based on conditions
def white_balls_bag_A := 8
def red_balls_bag_A := 4
def white_balls_bag_B := 6
def red_balls_bag_B := 6

-- Combinatorics function for choosing k items from n items
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Definition for probability P(ξ = 1)
def P_xi_eq_1 := 
  (C white_balls_bag_A 1 * C white_balls_bag_B 1 + C red_balls_bag_A 1 * C white_balls_bag_B 1) /
  (C (white_balls_bag_A + red_balls_bag_A) 1 * C (white_balls_bag_B + red_balls_bag_B) 1)

theorem probability_xi_eq_1 :
  P_xi_eq_1 = (C 8 1 * C 6 1 + C 4 1 * C 6 1) / (C 12 1 * C 12 1) :=
by
  sorry

end probability_xi_eq_1_l975_97564


namespace rectangle_area_increase_l975_97530

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_new := l_new * w_new
  let A := l * w
  let increase := A_new - A
  let percent_increase := (increase / A) * 100
  percent_increase = 56 := sorry

end rectangle_area_increase_l975_97530


namespace rectangle_area_ratio_l975_97578

noncomputable def area_ratio (s : ℝ) : ℝ :=
  let area_square := s^2
  let longer_side := 1.15 * s
  let shorter_side := 0.95 * s
  let area_rectangle := longer_side * shorter_side
  area_rectangle / area_square

theorem rectangle_area_ratio (s : ℝ) : area_ratio s = 109.25 / 100 := by
  sorry

end rectangle_area_ratio_l975_97578
