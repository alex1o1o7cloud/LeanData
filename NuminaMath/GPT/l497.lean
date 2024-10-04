import Mathlib

namespace fraction_to_three_decimal_places_l497_497862

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497862


namespace digits_to_right_of_decimal_point_l497_497734

theorem digits_to_right_of_decimal_point :
  let expr := (4 ^ 7) / (8 ^ 5 * 1250)
  in to_digits_right_of_decimal_point expr = 3 :=
by
  let expr := (4 ^ 7) / (8 ^ 5 * 1250)
  -- expected implementation of to_digits_right_of_decimal_point
  sorry

end digits_to_right_of_decimal_point_l497_497734


namespace rounded_to_three_decimal_places_l497_497900

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497900


namespace is_rectangle_iff_measure_three_right_angles_l497_497308

def is_rectangle (q : Quadrilateral) : Prop :=
  ∃ A B C D : Point, Quadrilateral.mk A B C D = q ∧
  angle A B C = π / 2 ∧ angle B C D = π / 2 ∧ angle C D A = π / 2

def measure_three_right_angles (q : Quadrilateral) : Prop :=
  ∃ A B C D : Point, Quadrilateral.mk A B C D = q ∧
  angle A B C = π / 2 ∧ angle B C D = π / 2 ∧ angle C D A = π / 2

theorem is_rectangle_iff_measure_three_right_angles (q : Quadrilateral) :
  is_rectangle q ↔ measure_three_right_angles q :=
sorry

end is_rectangle_iff_measure_three_right_angles_l497_497308


namespace parabola_no_real_intersection_l497_497722

theorem parabola_no_real_intersection (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -4) (h₃ : c = 5) :
  ∀ (x : ℝ), ¬ (a * x^2 + b * x + c = 0) :=
by
  sorry

end parabola_no_real_intersection_l497_497722


namespace terminating_decimal_multiples_l497_497213

theorem terminating_decimal_multiples :
  {m : ℕ | 1 ≤ m ∧ m ≤ 180 ∧ m % 3 = 0}.to_finset.card = 60 := 
sorry

end terminating_decimal_multiples_l497_497213


namespace final_mixture_percentage_l497_497484

-- Definitions based on given conditions
def total_volume : ℝ := 55
def pure_antifreeze_100 : ℝ := 6.11
def pure_rate_10 : ℝ := 0.10

-- The total amount of pure antifreeze in a 10% mixture V
def total_pure_volume (V : ℝ) : ℝ := pure_antifreeze_100 + pure_rate_10 * V

-- The volume of the 10% mixture required to fill 55 gallons
def remaining_volume : ℝ := total_volume - pure_antifreeze_100

-- Proof that remaining volume when combined with given antifreeze results in approximately 20% antifreeze
theorem final_mixture_percentage : 
  (total_pure_volume remaining_volume / total_volume) * 100 ≈ 20 := sorry

end final_mixture_percentage_l497_497484


namespace discount_percentage_l497_497512

variable {P P_b P_s : ℝ}
variable {D : ℝ}

theorem discount_percentage (P_s_eq_bought : P_s = 1.60 * P_b)
  (P_s_eq_original : P_s = 1.52 * P)
  (P_b_eq_discount : P_b = P * (1 - D)) :
  D = 0.05 := by
sorry

end discount_percentage_l497_497512


namespace proof_problem_l497_497237

-- Definitions for the first condition
def is_linear_eqn (a b : ℤ) : Prop :=
(a + b) = 0 ∧ (1/3 * a + 2) = 1

-- Definitions for the second condition
def eq_condition (x m : ℤ) : Prop :=
((x + 2) / 6 - (x - 1) / 2 + 3 = x - (2 * x - m) / 6)

-- Proof problem statement
theorem proof_problem :
  ∃ (a b m : ℤ),
    is_linear_eqn a b ∧
    eq_condition a m ∧
    |a - b| - |b - m| = -32 := 
by {
  -- Given conditions and translated proof statements
  use [-3, 3, 41],
  repeat { split },
  -- Proving the first condition is_linear_eqn for a and b
  { unfold is_linear_eqn, split,
    { exact rfl, }, 
    { norm_num, }, },
  -- Proving the second condition eq_condition for a and m
  { unfold eq_condition, calculate, norm_num, },
  -- Proving the final condition on |a - b| and |b - m|
  { norm_num, },
  sorry
}

end proof_problem_l497_497237


namespace repeating_decimals_count_l497_497218

theorem repeating_decimals_count :
  ∃ (count : ℕ), count = 18 ∧
  ∀ n ∈ finset.range (21), -- Since we need to go up to 20 inclusive
    (¬ (∃ k, n = 9 * k) ↔ (n / 18 ∉ finset.range (2))) :=
by
  sorry

end repeating_decimals_count_l497_497218


namespace fraction_to_three_decimal_places_l497_497856

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497856


namespace a_2_is_neg3_abs_sum_first_10_l497_497409

-- Definitions based on given conditions
def S (n : ℕ) : ℝ := n ^ 2 - 6 * n

def a (n : ℕ) : ℝ := 
  if n = 1 then S 1
  else S n - S (n - 1)

-- Prove the required equivalences

theorem a_2_is_neg3 : a 2 = -3 := by sorry

theorem abs_sum_first_10 : (∑ i in Finset.range 10, |a (i + 1)|) = 58 := by sorry

end a_2_is_neg3_abs_sum_first_10_l497_497409


namespace three_digit_sum_remainder_l497_497799

theorem three_digit_sum_remainder : 
  let T := (∑ x in finset.range 10, x) * 100 * 100 + (∑ x in finset.range 10, x) * 100 * 10 + (∑ x in finset.range 10, x) * 100
  in T % 1000 = 500 := by
  let T := (∑ x in finset.range 10, x) * 100 * 100 + (∑ x in finset.range 10, x) * 100 * 10 + (∑ x in finset.range 10, x) * 100
  have h1: ∑ x in finset.range 10, x = 45 := 
    finset.sum_range_succ 10
      (by
        -- We'll skip the intermediary proof of sum range with sorry
        -- The sum will actually be calculated as (n * (n-1)) / 2 with n = 10
        sorry)

  rw [h1]
  have h2: T = 450000 + 45000 + 4500 := by
    -- Simplify to find T based on the sum value h1
    sorry
  have h3: 450000 + 45000 + 4500 = 499500 := by 
    -- Combine values
    sorry
  have h4 : 499500 % 1000 = 500 := by
    -- Compute the remainder
    sorry
  rw [h3] at h2
  rw [h4]
  trivial

end three_digit_sum_remainder_l497_497799


namespace august_8th_is_saturday_l497_497756

-- Defining the conditions
def august_has_31_days : Prop := true

def august_has_5_mondays : Prop := true

def august_has_4_tuesdays : Prop := true

-- Statement of the theorem
theorem august_8th_is_saturday (h1 : august_has_31_days) (h2 : august_has_5_mondays) (h3 : august_has_4_tuesdays) : ∃ d : ℕ, d = 6 :=
by
  -- Translate the correct answer "August 8th is a Saturday" into the equivalent proposition
  -- Saturday is represented by 6 if we assume 0 = Sunday, 1 = Monday, ..., 6 = Saturday.
  sorry

end august_8th_is_saturday_l497_497756


namespace pure_imaginary_a_zero_l497_497750

theorem pure_imaginary_a_zero (a : ℝ) (h : ∃ b : ℝ, (i : ℂ) * (1 + (a : ℂ) * i) = (b : ℂ) * i) : a = 0 :=
by
  sorry

end pure_imaginary_a_zero_l497_497750


namespace fraction_rounded_to_decimal_l497_497939

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497939


namespace midpoints_form_regular_hexagon_l497_497344

-- Define the structure of a cube and its properties
structure Cube (k : ℝ) :=
(vertices : set (ℝ × ℝ × ℝ))
(is_cube : ∀ (v ∈ vertices), ∃ (x y z : ℝ), v = (x, y, z))
(edge_length : ∀ v w ∈ vertices, edge(v, w) -> distance(v, w) = 2 * k)

-- Define the diagonally opposite vertices A and B
def A := (0.0, 0.0, 0.0) : ℝ × ℝ × ℝ
def B := (2 * k, 2 * k, 2 * k) : ℝ × ℝ × ℝ

-- Define the midpoint of an edge
def midpoint (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2, (v1.3 + v2.3) / 2)

-- Define the set of midpoints that do not include A or B
def midpoints_without_A_B (cube : Cube k) : set (ℝ × ℝ × ℝ) :=
  {m | ∃ v1 v2 ∈ cube.vertices, ¬ (v1 = A ∨ v1 = B ∨ v2 = A ∨ v2 = B) ∧ m = midpoint v1 v2}

-- Define a regular hexagon property
def is_regular_hexagon (points : set (ℝ × ℝ × ℝ)) : Prop :=
  (∃ plane : ℝ × ℝ × ℝ → Prop, ∀ p ∈ points, plane p) ∧
  (∀ p1 p2 ∈ points, p1 ≠ p2 → distance p1 p2 = constant_distance) ∧
  (∃ center : ℝ × ℝ × ℝ, ∀ p ∈ points, distance center p = constant_radius)

-- The theorem we want to prove
theorem midpoints_form_regular_hexagon (k : ℝ) (cube : Cube k) :
  is_regular_hexagon (midpoints_without_A_B cube) :=
sorry

end midpoints_form_regular_hexagon_l497_497344


namespace division_rounded_l497_497849

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497849


namespace distance_from_point_to_circle_center_l497_497780

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def circle_center : ℝ × ℝ := (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_from_point_to_circle_center :
  distance (polar_to_rect 2 (Real.pi / 3)) circle_center = Real.sqrt 3 := sorry

end distance_from_point_to_circle_center_l497_497780


namespace incorrect_calculation_l497_497448

noncomputable def condition_A : Prop := sqrt 2 + sqrt 3 = sqrt 5
noncomputable def condition_B : Prop := 5 * sqrt 3 - 2 * sqrt 3 = 3 * sqrt 3
noncomputable def condition_C : Prop := sqrt 6 / sqrt 2 = sqrt 3
noncomputable def condition_D : Prop := (-sqrt 2)^2 = 2

theorem incorrect_calculation : ¬condition_A := by
  sorry

end incorrect_calculation_l497_497448


namespace integer_solutions_for_exponential_equation_l497_497203

theorem integer_solutions_for_exponential_equation :
  ∃ (a b c : ℕ), 
  2 ^ a * 3 ^ b + 9 = c ^ 2 ∧ 
  (a = 4 ∧ b = 0 ∧ c = 5) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 9) ∨ 
  (a = 4 ∧ b = 3 ∧ c = 21) ∨ 
  (a = 3 ∧ b = 3 ∧ c = 15) ∨ 
  (a = 4 ∧ b = 5 ∧ c = 51) :=
by {
  -- This is where the proof would go.
  sorry
}

end integer_solutions_for_exponential_equation_l497_497203


namespace range_of_set_of_three_numbers_l497_497134

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497134


namespace range_of_numbers_is_six_l497_497110

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497110


namespace GCF_seven_eight_factorial_l497_497655

-- Given conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Calculating 7! and 8!
def seven_factorial := factorial 7
def eight_factorial := factorial 8

-- Proof statement
theorem GCF_seven_eight_factorial : ∃ g, g = seven_factorial ∧ g = Nat.gcd seven_factorial eight_factorial ∧ g = 5040 :=
by sorry

end GCF_seven_eight_factorial_l497_497655


namespace four_digit_count_is_300_four_digit_odd_count_is_192_four_digit_even_count_is_108_natural_number_count_is_1631_l497_497704

-- Definitions of the digit set and helper functions
def digits := {0, 1, 2, 3, 5, 9}
def is_odd (n : Nat) : Prop := n % 2 = 1
def is_even (n : Nat) : Prop := n % 2 = 0
def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n < 10000

-- 1. Prove that the number of four-digit numbers that can be formed is 300
theorem four_digit_count_is_300 : 
  (∃ valid_numbers : Finset ℕ, (∀ n ∈ valid_numbers, is_four_digit n ∧ ∀ d ∈ Int.toFinset n, d ∈ digits) ∧ valid_numbers.card = 300) := sorry

-- 2. Prove that the number of four-digit odd numbers that can be formed is 192
theorem four_digit_odd_count_is_192 : 
  (∃ valid_numbers : Finset ℕ, (∀ n ∈ valid_numbers, is_four_digit n ∧ is_odd n ∧ ∀ d ∈ Int.toFinset n, d ∈ digits) ∧ valid_numbers.card = 192) := sorry

-- 3. Prove that the number of four-digit even numbers that can be formed is 108
theorem four_digit_even_count_is_108 : 
  (∃ valid_numbers : Finset ℕ, (∀ n ∈ valid_numbers, is_four_digit n ∧ is_even n ∧ ∀ d ∈ Int.toFinset n, d ∈ digits) ∧ valid_numbers.card = 108) := sorry

-- 4. Prove that the number of natural numbers that can be formed is 1631
theorem natural_number_count_is_1631 : 
  (∃ valid_numbers : Finset ℕ, (∀ n ∈ valid_numbers, ∀ d ∈ Int.toFinset n, d ∈ digits) ∧ valid_numbers.card = 1631) := sorry

end four_digit_count_is_300_four_digit_odd_count_is_192_four_digit_even_count_is_108_natural_number_count_is_1631_l497_497704


namespace fraction_rounded_equals_expected_l497_497919

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497919


namespace no_member_of_T_is_divisible_by_4_l497_497805

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

theorem no_member_of_T_is_divisible_by_4 : ∀ n : ℤ, ¬ (sum_of_squares_of_four_consecutive_integers n % 4 = 0) := by
  intro n
  sorry

end no_member_of_T_is_divisible_by_4_l497_497805


namespace find_c_value_l497_497038

theorem find_c_value (b c : ℝ) 
  (h1 : 1 + b + c = 4) 
  (h2 : 25 + 5 * b + c = 4) : 
  c = 9 :=
by
  sorry

end find_c_value_l497_497038


namespace sequence_a_2017_l497_497265

theorem sequence_a_2017 :
  (∃ (a : ℕ → ℚ), (a 1 = 1) ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 2016 * a n / (2014 * a n + 2016)) → a 2017 = 1008 / (1007 * 2017 + 1)) :=
by
  sorry

end sequence_a_2017_l497_497265


namespace factor_polynomial_l497_497620

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497620


namespace sin_series_bound_l497_497681

theorem sin_series_bound (n : ℕ) (θ : ℝ) (h_n : n ≥ 2) (h_θ1 : 0 < θ) (h_θ2 : θ < π) : 
  (sin (θ / 2)) * (∑ k in Finset.range (n + 1), (sin (k * θ)) / k) < 1 := 
begin
  sorry
end

end sin_series_bound_l497_497681


namespace range_of_set_l497_497071

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497071


namespace unique_ordered_triple_l497_497738

def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem unique_ordered_triple : 
  ∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  lcm x y = 180 ∧ 
  lcm y z = 1260 ∧ 
  lcm x z = 840 :=
sorry

end unique_ordered_triple_l497_497738


namespace zach_cookies_total_l497_497453

theorem zach_cookies_total :
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  cookies_monday + cookies_tuesday + cookies_wednesday = 92 :=
by
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  sorry

end zach_cookies_total_l497_497453


namespace num_factors_of_a_l497_497516

variable (a : ℕ) (f : ℕ → ℕ)
hypothesis (h_pos : a > 1)
hypothesis (h_prod : (a^(f(a) / 2)) = (a^3))

theorem num_factors_of_a : f(a) = 6 := sorry

end num_factors_of_a_l497_497516


namespace polynomial_factorization_l497_497562

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497562


namespace peaches_sold_to_friends_l497_497815

theorem peaches_sold_to_friends (x : ℕ) (total_peaches : ℕ) (peaches_to_relatives : ℕ) (peach_price_friend : ℕ) (peach_price_relative : ℝ) (total_earnings : ℝ) (peaches_left : ℕ) (total_peaches_sold : ℕ) 
  (h1 : total_peaches = 15) 
  (h2 : peaches_to_relatives = 4) 
  (h3 : peach_price_relative = 1.25) 
  (h4 : total_earnings = 25) 
  (h5 : peaches_left = 1)
  (h6 : total_peaches_sold = 14)
  (h7 : total_earnings = peach_price_friend * x + peach_price_relative * peaches_to_relatives)
  (h8 : total_peaches_sold = total_peaches - peaches_left) :
  x = 10 := 
sorry

end peaches_sold_to_friends_l497_497815


namespace staff_price_correct_l497_497456

variable (d : ℝ)

-- Define the conditions
def initial_discount := 0.15 * d
def discounted_price := d - initial_discount
def staff_discount := 0.10 * discounted_price
def final_price := discounted_price - staff_discount

-- Prove that final price is equal to 0.765 * d
theorem staff_price_correct : final_price = 0.765 * d :=
by
  -- This is a placeholder. Proof should be provided here.
  sorry

end staff_price_correct_l497_497456


namespace jonathan_fourth_task_completion_l497_497788

-- Conditions
def start_time : Nat := 9 * 60 -- 9:00 AM in minutes
def third_task_completion_time : Nat := 11 * 60 + 30 -- 11:30 AM in minutes
def number_of_tasks : Nat := 4
def number_of_completed_tasks : Nat := 3

-- Calculation of time duration
def total_time_first_three_tasks : Nat :=
  third_task_completion_time - start_time

def duration_of_one_task : Nat :=
  total_time_first_three_tasks / number_of_completed_tasks
  
-- Statement to prove
theorem jonathan_fourth_task_completion :
  (third_task_completion_time + duration_of_one_task) = (12 * 60 + 20) :=
  by
    -- We do not need to provide the proof steps as per instructions
    sorry

end jonathan_fourth_task_completion_l497_497788


namespace fraction_rounded_to_decimal_l497_497935

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497935


namespace fraction_rounding_l497_497890

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497890


namespace rotate_blocks_to_top_l497_497416

-- Definitions and conditions
def has_one_red_face (block : ℕ → ℕ → bool) : Prop :=
∀ i j, ∃ k : Fin 6, (∀ m : Fin 6, m ≠ k → block i j (m.val) = false) ∧ block i j k.val = true

def rotate_row (blocks : ℕ → ℕ → ℕ → bool) (r : ℕ) : (ℕ → ℕ → ℕ → bool) :=
fun i j k => if k = 0 then blocks r j 4 else if k = 1 then blocks r j 5 else if k = 2 then blocks r j 2 else if k = 3 then blocks r j 0 else if k = 4 then blocks r j 1 else blocks r j 3

def rotate_col (blocks : ℕ → ℕ → ℕ → bool) (c : ℕ) : (ℕ → ℕ → ℕ → bool) :=
fun i j k => if k = 0 then blocks i c 4 else if k = 1 then blocks i c 5 else if k = 2 then blocks i c 2 else if k = 3 then blocks i c 0 else if k = 4 then blocks i c 1 else blocks i c 3

-- Main statement
theorem rotate_blocks_to_top :
  ∃ (n : ℕ) (blocks : ℕ → ℕ → ℕ → bool), 
    n = 8 ∧ 
    (∀ i j : ℕ, i < 8 → j < 8 → has_one_red_face blocks) ∧
    (∃ (rotate_row : (ℕ → ℕ → ℕ → bool) → ℕ → (ℕ → ℕ → ℕ → bool)) 
       (rotate_col : (ℕ → ℕ → ℕ → bool) → ℕ → (ℕ → ℕ → ℕ → bool)), 
      ∀ i j : ℕ, i < 8 → j < 8 → (rotate_row blocks j) i j 0 = true) :=
sorry

end rotate_blocks_to_top_l497_497416


namespace range_of_set_is_six_l497_497081

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497081


namespace range_of_set_of_three_numbers_l497_497131

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497131


namespace ratio_of_red_to_blue_marbles_l497_497534

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

end ratio_of_red_to_blue_marbles_l497_497534


namespace max_investment_at_7_percent_l497_497007

theorem max_investment_at_7_percent 
  (x y : ℝ)
  (h₁ : x + y = 25000)
  (h₂ : 0.07 * x + 0.12 * y ≥ 2450) : 
  x ≤ 11000 :=
begin
  -- The proof will be here
  sorry
end

end max_investment_at_7_percent_l497_497007


namespace part1_part2_part3_l497_497341

open Real

def f (x : ℝ) (n : ℕ) : ℝ := (log x) / (x^n)
def g (x : ℝ) (n : ℕ) : ℝ := (exp x) / (x^n)

theorem part1 : 
  (f 1 1) has exactly one zero in (0, +∞) := 
sorry

theorem part2 (n : ℕ) (h1 : f (1 : ℝ) n < 1) (h2 : g (1 : ℝ) n > 1) : 
  n = 1 ∨ n = 2 :=
sorry

theorem part3 (n : ℕ) (h : n = 1 ∨ n = 2) (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2): 
  abs (f x1 n - g x2 n) = \frac{e^2}{4} - \frac{1}{2e} :=
sorry

end part1_part2_part3_l497_497341


namespace increasing_sequences_mod_1000_l497_497736

theorem increasing_sequences_mod_1000 :
  (Fintype.card {s : Fin 12 → ℤ // (∀ i, s i ≤ 2010) ∧ (∀ i, s i - i.val % 2 = 1) ∧ ∀ i j, i ≤ j → s i ≤ s j}) % 1000 = 16 :=
sorry

end increasing_sequences_mod_1000_l497_497736


namespace f_monotonic_intervals_and_extremum_g_always_above_x_axis_l497_497709

def f (x : ℝ) : ℝ := x - Real.log x

noncomputable def g (x : ℝ) : ℝ := x^3 + x^2 * f x - 16 * x + 20

theorem f_monotonic_intervals_and_extremum :
  (∀ x, x > 1 → f' x > 0) ∧ (∀ x, 0 < x < 1 → f' x < 0) ∧ (f 1 = 1) :=
sorry

theorem g_always_above_x_axis :
  ∀ x > 0, g x > 0 :=
sorry

end f_monotonic_intervals_and_extremum_g_always_above_x_axis_l497_497709


namespace factorization_identity_l497_497612

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497612


namespace problem_statement_l497_497222

theorem problem_statement (x y : ℝ) (h : -x + 2 * y = 5) :
  5 * (x - 2 * y) ^ 2 - 3 * (x - 2 * y) - 60 = 80 :=
by
  sorry

end problem_statement_l497_497222


namespace solution_set_I_range_of_m_II_l497_497718

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem solution_set_I : {x : ℝ | 0 ≤ x ∧ x ≤ 3} = {x : ℝ | f x ≤ 3} :=
sorry

theorem range_of_m_II (x : ℝ) (hx : x > 0) : ∃ m : ℝ, ∀ (x : ℝ), f x ≤ m - x - 4 / x → m ≥ 5 :=
sorry

end solution_set_I_range_of_m_II_l497_497718


namespace proof_a_lt_b_lt_c_l497_497678

def a : ℝ := Real.sqrt ((1 - Real.cos (110 * Real.pi / 180)) / 2)
def b : ℝ := (Real.sqrt 2 / 2) * (Real.sin (20 * Real.pi / 180) + Real.cos (20 * Real.pi / 180))
def c : ℝ := (1 + Real.tan (20 * Real.pi / 180)) / (1 - Real.tan (20 * Real.pi / 180))

theorem proof_a_lt_b_lt_c : a < b ∧ b < c := by
  sorry

end proof_a_lt_b_lt_c_l497_497678


namespace area_of_BEFC_l497_497012

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A B C : Point
  hAB : (A.x - B.x)^2 + (A.y - B.y)^2 = 4 -- AB = 2^2
  hAC : (A.x - C.x)^2 + (A.y - C.y)^2 = 4 -- AC = 2^2
  hBC : (B.x - C.x)^2 + (B.y - C.y)^2 = 9 -- BC = 3^2

structure Line where
  p1 p2 : Point
  slope : ℝ
  h_slope : slope = (p2.y - p1.y) / (p2.x - p1.x)

noncomputable def area_triangle (P Q R : Point) : ℝ := 
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

noncomputable def midpoint (P Q : Point) : Point := {
  x := (P.x + Q.x) / 2,
  y := (P.y + Q.y) / 2
}

def intersects (L1 L2 : Line) : Point := sorry

theorem area_of_BEFC (A B C D E F : Point)
  (h_triangle : Triangle A B C)
  (h_D : D.x = C.x + 1.5 ∧ D.y = 0) -- CD = BC/2
  (h_E : E = midpoint A B)
  (h_IE : Line E D)
  (h_IAC : Line A C)
  (h_F : F = intersects (Line.mk E D h_IE.slope h_IE.h_slope) (Line.mk A C h_IAC.slope h_IAC.h_slope)) :
  ∃ area_BEFC : ℝ, area_BEFC = area_triangle B F C + area_triangle E F C := sorry

end area_of_BEFC_l497_497012


namespace round_8_div_11_l497_497872

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497872


namespace max_truthful_dwarfs_l497_497962

theorem max_truthful_dwarfs :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  (∃ n, n = 1 ∧ ∀ i, i < 7 → i ≠ n → heights[i] ≠ (heights[i] + 1)) :=
by
  let heights := [60, 61, 62, 63, 64, 65, 66]
  existsi 1
  intro i
  intro h
  intro hi
  sorry

end max_truthful_dwarfs_l497_497962


namespace percentage_of_men_attended_l497_497755

variable (E : ℝ) (M W : ℝ)
variable (TotalAttended WomenAttended : ℝ)

-- Given Conditions
def men_percentage (E : ℝ) : ℝ := 0.55 * E
def women_percentage (E : ℝ) : ℝ := 0.45 * E
def women_attended (W : ℝ) : ℝ := 0.4 * W
def total_attended (E : ℝ) : ℝ := 0.29 * E

-- Proof statement
theorem percentage_of_men_attended : 
  ∀ (E : ℝ), let M := men_percentage E in
  let W := women_percentage E in
  let WomenAttended := women_attended W in 
  let TotalAttended := total_attended E in
  (TotalAttended - WomenAttended) / M * 100 = 20 :=
by
  intros
  let M := men_percentage E
  let W := women_percentage E
  let WomenAttended := women_attended W
  let TotalAttended := total_attended E
  sorry

end percentage_of_men_attended_l497_497755


namespace simson_lines_concurrent_l497_497373

open EuclideanGeometry

theorem simson_lines_concurrent 
  {A B C D O : Point}
  (h_circle : IsCyclicQuadrilateral A B C D)
  (lₐ : SimsonLine A B C D)
  (l_b : SimsonLine B A C D)
  (l_c : SimsonLine C A B D)
  (l_d : SimsonLine D A B C) :
  ∃ H : Point, ConcurrentLines lₐ l_b l_c l_d H := 
sorry

end simson_lines_concurrent_l497_497373


namespace xyz_inequality_proof_l497_497349

noncomputable def xyz_inequality (x y z : ℝ) : Prop :=
  (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) →
  ( ( ( (x * y + y * z + z * x) / 3 ) ^ 3 ) ≤ 
    ( (x^2 - x * y + y^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2) ) ∧
    ( (x^2 - x * y + y^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2) ) ≤ 
    ( ( (x^2 + y^2 + z^2) / 2 ) ^ 3 ) )

theorem xyz_inequality_proof (x y z : ℝ) : xyz_inequality x y z :=
begin
  sorry
end

end xyz_inequality_proof_l497_497349


namespace range_of_set_of_three_numbers_l497_497132

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497132


namespace length_AD_is_1_l497_497164

theorem length_AD_is_1 (A B C D : Type)
  [Isosceles : Triangle.is_isosceles ⟨A, B, C⟩ (AB = AC)]
  (angle_A : ∠ A = 120) (BD DC : ℝ) (BD_eq_1 : BD = 1) (DC_eq_2 : DC = 2) :
  length AD = 1 :=
sorry

end length_AD_is_1_l497_497164


namespace sqrt_difference_l497_497171

theorem sqrt_difference : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := 
by 
  sorry

end sqrt_difference_l497_497171


namespace fraction_to_three_decimal_places_l497_497859

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497859


namespace factor_poly_eq_factored_form_l497_497604

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497604


namespace triangle_trigonometric_identities_l497_497834

theorem triangle_trigonometric_identities
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : a / sin α = b / sin β)
  (h2 : b / sin β = c / sin γ)
  (h3 : a = c * sin α / sin γ)
  (h4 : b = c * sin β / sin γ) :
  (a + b) / c = cos ((α - β) / 2) / sin (γ / 2) ∧
  (a - b) / c = sin ((α - β) / 2) / cos (γ / 2) :=
sorry

end triangle_trigonometric_identities_l497_497834


namespace polynomial_factorization_l497_497570

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497570


namespace floor_sum_inequality_l497_497807

noncomputable def floor_sum_ineq (x : ℚ) (n : ℕ) : Prop :=
  (x > 0) → (n > 0) → (∑ k in Finset.range n, (⌊(k + 1) * x⌋ / (k + 1))) ≤ ⌊n * x⌋

-- Proof (to be completed)
theorem floor_sum_inequality (x : ℚ) (n : ℕ) : floor_sum_ineq x n :=
by
  sorry

end floor_sum_inequality_l497_497807


namespace count_elements_arithmetic_seq_13_l497_497189

theorem count_elements_arithmetic_seq_13 :
  ∃ n : ℕ, 2.5 + (n - 1) * 5 = 62.5 ∧ n = 13 :=
by sorry

end count_elements_arithmetic_seq_13_l497_497189


namespace integration_minimum_value_l497_497796

noncomputable def f (a b : ℝ) (c : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin x + b * Real.cos x + c

-- Conditions
theorem integration_minimum_value {a b : ℝ}
  (Hf : ∀ x : ℝ, (-π ≤ x ∧ x ≤ π) → f a b (∫ t in (-π)..π, f a b (∫ t in (-π)..π, f a b (∫ t in (-π)..π, (Real.cos t) ∂t) ∂t) ∂t) x = 2π)
  (Hc_eq : ∫ t in (-π)..π, f a b (∫ t in (-π)..π, f a b (∫ t in (-π)..π, f a b (∫ t in (-π)..π, (Real.cos t) ∂t) ∂t) ∂t) (t) * (Real.cos (t)) ∂t = b * π / 2) :
  ∫ x in (-π)..π, (f a b (b * π / 2) x)^2 ∂x = 52 * π^3 / 9 :=
  sorry

end integration_minimum_value_l497_497796


namespace complex_z_modulus_l497_497249

noncomputable def i : ℂ := Complex.I

theorem complex_z_modulus (z : ℂ) (h : (1 + i) * z = 2 * i) : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end complex_z_modulus_l497_497249


namespace division_rounded_l497_497852

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497852


namespace divide_towns_and_cities_l497_497481

noncomputable def divide_into_republics (n : ℕ) (cities : Fin n → String) (towns : List String)
  (road : (String × String) → Option ℕ) : Prop :=
  ∃ (republics : Fin n → List String), 
  ∀ (i : Fin n), 
    (cities i ∈ republics i) ∧
    ∀ (town : String), 
    town ∈ republics i →
      (∃ (G : List (String × String)), 
        List.head G = Some (town, cities i) ∧ 
        List.last G = Some (town, cities i) ∧ 
        (∀ (j : Fin n), (town ∈ republics j → i = j)) ∧
        (∀ (path : List (String × String)), road.cyclic path = false →
          (List.head path = Some (town, cities i) ∧ List.length path < List.length G) → 
          List.head path = Some (town, cities j) → i = j))

axiom problem_setup (n : ℕ) (cities : Fin n → String) (towns : List String) 
  (road : (String × String) → Option ℕ) :
  (∀ (t1 t2 : String), (t1 ∈ towns) → (t2 ∈ towns) → road (t1, t2) = none ∨ road (t1, t2) = some 1) ∧
  (∀ (c : Fin n) (t : String), (t ∈ towns) → road (cities c, t) = none ∨ road (cities c, t) = some 1) ∧
  (∀ (t : String), (t ∈ towns) → ∃ (c : Fin n), road (cities c, t) ≠ none) ∧
  (∀ path, List.length path > 0 → (road.cyclic path = false) → 
    (∃ (t : String), (t ∈ towns) → ¬ ∀ (c : Fin n), road (cities c, t) = none))

theorem divide_towns_and_cities (n : ℕ) (cities : Fin n → String) (towns : List String) 
  (road : (String × String) → Option ℕ) 
  (h : problem_setup n cities towns road) : 
  divide_into_republics n cities towns road :=
sorry

end divide_towns_and_cities_l497_497481


namespace range_of_set_l497_497058

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497058


namespace factor_polynomial_l497_497590

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497590


namespace students_not_in_biology_l497_497289

theorem students_not_in_biology (total_students : ℕ) (percent_enrolled : ℝ) 
  (hn : percent_enrolled = 32.5) (ht : total_students = 880) : 
  total_students - nat.floor ((percent_enrolled / 100) * total_students) = 594 := by
  sorry

end students_not_in_biology_l497_497289


namespace least_four_digit_palindrome_divisible_by_5_l497_497369

-- Defining a four-digit palindrome and the condition of being divisible by 5
def is_palindrome (n : ℕ) : Prop :=
  n / 1000 = n % 10 ∧ (n / 100) % 10 = (n / 10) % 10

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- The main theorem to be proved
theorem least_four_digit_palindrome_divisible_by_5 :
  ∃ n, is_four_digit n ∧ is_palindrome n ∧ is_divisible_by_5 n ∧
       ∀ m, is_four_digit m ∧ is_palindrome m ∧ is_divisible_by_5 m → n ≤ m :=
  exists.intro 5005 (by
    repeat {split};
    try { sorry }) 

end least_four_digit_palindrome_divisible_by_5_l497_497369


namespace fraction_to_three_decimal_places_l497_497865

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497865


namespace green_more_than_blue_l497_497008

theorem green_more_than_blue (B Y G : Nat) (h1 : B + Y + G = 108) (h2 : B * 7 = Y * 3) (h3 : B * 8 = G * 3) : G - B = 30 := by
  sorry

end green_more_than_blue_l497_497008


namespace max_truthful_dwarfs_le_one_l497_497985

def dwarf_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

theorem max_truthful_dwarfs_le_one (heights : List ℕ) (h_lengths : heights.length = 7) (h_sorted : heights.sorted (· > ·)) :
  ∃ n, (n ≤ 1) ∧ ∀ i, i ∈ (Finset.range 7).to_list → (heights[i] ≠ dwarf_height_claims[i] → (1 = n)) :=
  sorry

end max_truthful_dwarfs_le_one_l497_497985


namespace standard_equation_of_ellipse_max_area_triangle_PAB_l497_497233

variables {a b : ℝ} (P A B : ℝ × ℝ)
def ellipse := { (x, y) : ℝ × ℝ // (x^2 / a^2) + (y^2 / b^2) = 1 }
def P := (2, 1)
def l (m : ℝ) := { (x, y) : ℝ × ℝ // y = (1/2) * x + m }

theorem standard_equation_of_ellipse (h1 : a > b) (h2 : b > 0) (h3 : a^2 = 4 * b^2) (h4 : P ∈ ellipse) :
 (∃ a b : ℝ, a^2 = 8 ∧ b^2 = 2 ∧ ∀ x y, x^2/8 + y^2/2 = 1) :=
sorry

theorem max_area_triangle_PAB (h1 : a > b) (h2 : b > 0) (h3 : P ∈ ellipse) (h4 : ∀ m, ∃ A B : ellipse, A ∈ l m ∧ B ∈ l m) :
  (∃ S, S = 2) :=
sorry

end standard_equation_of_ellipse_max_area_triangle_PAB_l497_497233


namespace smallest_number_divisible_l497_497443

theorem smallest_number_divisible (n : ℤ) : 
  (n + 7) % 25 = 0 ∧
  (n + 7) % 49 = 0 ∧
  (n + 7) % 15 = 0 ∧
  (n + 7) % 21 = 0 ↔ n = 3668 :=
by 
 sorry

end smallest_number_divisible_l497_497443


namespace max_one_truthful_dwarf_l497_497989

def dwarf_height_claims : List (ℕ × ℕ) := [
  (1, 60),  -- First dwarf claims 60 cm
  (2, 61),  -- Second dwarf claims 61 cm
  (3, 62),  -- Third dwarf claims 62 cm
  (4, 63),  -- Fourth dwarf claims 63 cm
  (5, 64),  -- Fifth dwarf claims 64 cm
  (6, 65),  -- Sixth dwarf claims 65 cm
  (7, 66)   -- Seventh dwarf claims 66 cm
]

-- Proof problem statement: Prove that at most one dwarf is telling the truth about their height
theorem max_one_truthful_dwarf : ∀ (heights : List ℕ), heights.length = 7 → 
  (∀ i, i < heights.length → dwarf_height_claims.get? i ≠ none → (heights.get? i = dwarf_height_claims.get? i) → 
    ∑ i in finset.range heights.length, bool.to_num (heights.get? i = dwarf_height_claims.get? i) ≤ 1) :=
by
  intros heights len_heights valid_claims idx idx_in_range some_claim match_claim
  -- Add proof here 
  sorry

end max_one_truthful_dwarf_l497_497989


namespace family_age_partition_l497_497433

open List

theorem family_age_partition (ages : List ℕ) (h : ages = [2, 3, 4, 5, 6, 8]) :
  (∃ family1 family2 : List ℕ, family1.sum = 14 ∧ family2.sum = 14 ∧ family1 ++ family2 ~ ages) :=
by
  have total_sum : ages.sum = 28 := by
  { rw h,
    norm_num }

  have sum_each_family : 28 / 2 = 14 := by norm_num

  use [8, 6], [2, 3, 4, 5]
  split
  { norm_num }
  split
  { norm_num }
  { rw [concat_cons, h] }
  { sorry }

  use [8, 4, 2], [3, 5, 6]
  split
  { norm_num }
  split
  { norm_num }
  { rw [concat_cons, h] }
  { sorry }

end family_age_partition_l497_497433


namespace range_of_set_l497_497120

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497120


namespace find_AC_l497_497777

noncomputable def BD := real.sqrt (13^2 - 12^2)
noncomputable def BC := real.sqrt (15^2 - BD^2)
noncomputable def DE := BC
noncomputable def CE := BD
noncomputable def AE := 12 + DE
noncomputable def AC := real.sqrt (AE^2 + CE^2)

theorem find_AC :
  let AB := 13
  let DC := 15
  let AD := 12
  |real.to_nnreal AC - 24.1| < 0.1 := 
by
  sorry

end find_AC_l497_497777


namespace number_of_cows_l497_497307

def problem_statement (cows chickens ducks : ℕ) : Prop :=
  let total_legs := 4 * cows + 2 * chickens + 2 * ducks in
  let total_heads := cows + chickens + 2 * ducks in
  total_legs = 18 + 2 * total_heads

theorem number_of_cows (c : ℕ) : problem_statement c 8 3 → c = 9 :=
sorry

end number_of_cows_l497_497307


namespace range_of_a_l497_497243

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

noncomputable def monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ {x y}, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_mono : monotonically_increasing f {x | 0 < x})
  (h_cond : ∀ x : ℝ, f (Real.log2 a) ≤ f (x^2 - 2*x + 2)) :
  ∃ a, a ∈ Icc (1/2 : ℝ) (2 : ℝ) :=
sorry

end range_of_a_l497_497243


namespace white_marble_fraction_after_tripling_l497_497758

theorem white_marble_fraction_after_tripling (
  (green_frac blue_frac : ℚ) (total_marbles : ℚ) 
  (h_green_frac : green_frac = 4 / 7) 
  (h_blue_frac : blue_frac = 1 / 7)
) : 
  let white_frac := 1 - (green_frac + blue_frac),
  let initial_white := white_frac * total_marbles,
  let tripled_white := 3 * initial_white,
  let new_total_marbles := total_marbles + 2 * initial_white 

  in tripled_white / new_total_marbles = 6 / 11 := 
by {
  sorry
}

end white_marble_fraction_after_tripling_l497_497758


namespace fraction_of_fifth_set_l497_497792

theorem fraction_of_fifth_set :
  let total_match_duration := 11 * 60 + 5
  let fifth_set_duration := 8 * 60 + 11
  (fifth_set_duration : ℚ) / total_match_duration = 3 / 4 := 
sorry

end fraction_of_fifth_set_l497_497792


namespace derivative_pattern_cos_l497_497225

noncomputable def f : ℕ → (ℝ → ℝ)
| 0       := sin
| (n + 1) := (f n)' -- the derivative function

theorem derivative_pattern_cos :
  f 2005 = cos :=
by sorry

end derivative_pattern_cos_l497_497225


namespace pets_percentage_of_cats_l497_497361

theorem pets_percentage_of_cats :
  ∀ (total_pets dogs as_percentage bunnies cats_percentage : ℕ),
    total_pets = 36 →
    dogs = total_pets * as_percentage / 100 →
    as_percentage = 25 →
    bunnies = 9 →
    cats_percentage = (total_pets - (dogs + bunnies)) * 100 / total_pets →
    cats_percentage = 50 :=
by
  intros total_pets dogs as_percentage bunnies cats_percentage
  sorry

end pets_percentage_of_cats_l497_497361


namespace factor_polynomial_l497_497585

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497585


namespace factorization_correct_l497_497572

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497572


namespace radius_of_circle_through_A_B_tangent_to_other_side_l497_497827

variables {a b : ℝ}
variables {O A B M : Type*}

-- Definitions for point O, distance OA, and OB
def distance_OA := a
def distance_OB := b

-- Theorem statement
theorem radius_of_circle_through_A_B_tangent_to_other_side :
  ∃ (R : ℝ), 
  R = (a + b) / 2 := 
begin
  use (a + b) / 2,
  sorry
end

end radius_of_circle_through_A_B_tangent_to_other_side_l497_497827


namespace meaningful_sqrt_range_l497_497424

theorem meaningful_sqrt_range (x : ℝ) : (∃ y : ℝ, y = sqrt (2 - x)) → x ≤ 2 :=
by {
  sorry
}

end meaningful_sqrt_range_l497_497424


namespace common_chord_length_l497_497247

theorem common_chord_length 
    (O1_eqn : ∀ (ρ θ : ℝ), ρ = 2) 
    (O2_eqn : ∀ (ρ θ : ℝ), 
        ρ^2 - 2 * √2 * ρ * Real.cos (θ - π / 4) = 2) 
    : (∃ A B : ℝ → ℝ × ℝ, 
        let x := A ∘ so_sqrt
        let y := B ∘ so_sqrt 
        (x 1 = y 1) → 
        (|AB| = √2 * 2 + √14) :=
by
    sorry

end common_chord_length_l497_497247


namespace monotonic_range_of_a_l497_497698

noncomputable def f (a x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1
noncomputable def f' (a x : ℝ) : ℝ := -3*x^2 + 2*a*x - 1

theorem monotonic_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f' a x ≤ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by 
  sorry

end monotonic_range_of_a_l497_497698


namespace max_one_truthful_dwarf_l497_497988

def dwarf_height_claims : List (ℕ × ℕ) := [
  (1, 60),  -- First dwarf claims 60 cm
  (2, 61),  -- Second dwarf claims 61 cm
  (3, 62),  -- Third dwarf claims 62 cm
  (4, 63),  -- Fourth dwarf claims 63 cm
  (5, 64),  -- Fifth dwarf claims 64 cm
  (6, 65),  -- Sixth dwarf claims 65 cm
  (7, 66)   -- Seventh dwarf claims 66 cm
]

-- Proof problem statement: Prove that at most one dwarf is telling the truth about their height
theorem max_one_truthful_dwarf : ∀ (heights : List ℕ), heights.length = 7 → 
  (∀ i, i < heights.length → dwarf_height_claims.get? i ≠ none → (heights.get? i = dwarf_height_claims.get? i) → 
    ∑ i in finset.range heights.length, bool.to_num (heights.get? i = dwarf_height_claims.get? i) ≤ 1) :=
by
  intros heights len_heights valid_claims idx idx_in_range some_claim match_claim
  -- Add proof here 
  sorry

end max_one_truthful_dwarf_l497_497988


namespace even_sum_probability_l497_497422

theorem even_sum_probability :
  let p_even_w1 := 3 / 4
  let p_even_w2 := 1 / 2
  let p_even_w3 := 1 / 4
  let p_odd_w1 := 1 - p_even_w1
  let p_odd_w2 := 1 - p_even_w2
  let p_odd_w3 := 1 - p_even_w3
  (p_even_w1 * p_even_w2 * p_even_w3) +
  (p_odd_w1 * p_odd_w2 * p_even_w3) +
  (p_odd_w1 * p_even_w2 * p_odd_w3) +
  (p_even_w1 * p_odd_w2 * p_odd_w3) = 1 / 2 := by
    sorry

end even_sum_probability_l497_497422


namespace rectangle_area_eq_l497_497500

-- Conditions
def triangle_sides := (10 : ℝ, 12 : ℝ, 15 : ℝ)
def rectangle_side1 := (12 : ℝ)
def triangle_perimeter := triangle_sides.1 + triangle_sides.2 + triangle_sides.3
def rectangle_perimeter := triangle_perimeter

-- Theorem stating the area of the rectangle
theorem rectangle_area_eq :
  ∃ (side2 : ℝ), 2 * (rectangle_side1 + side2) = rectangle_perimeter ∧ (rectangle_side1 * side2 = 78) :=
by
  -- Recall necessary conditions
  let side2 : ℝ := 6.5 
  have side2_eqn : 2 * (rectangle_side1 + side2) = rectangle_perimeter := by 
    calc
      2 * (rectangle_side1 + side2) = 2 * (12 + 6.5) : by sorry 
      ... = 2 * 18.5 : by sorry 
      ... = 37 : by sorry --  2 * 18.5 = 37, which is the perimeter
  have area_eq : rectangle_side1 * side2 = 78 := by 
    calc
      12 * 6.5 = 78 : by sorry 
  -- Conclusion
  exact ⟨side2, side2_eqn, area_eq⟩

end rectangle_area_eq_l497_497500


namespace positive_numbers_inequality_l497_497837

variable {n : ℕ}
variable {x : Fin n → ℝ}

theorem positive_numbers_inequality (h_pos : ∀ i, 0 < x i) (h_le_one : ∀ i, x i ≤ 1) :
  (∏ i : Fin n, (1 + x i).pow (1 / x ((i + 1) % n))) > 2 ^ n :=
by
  sorry

end positive_numbers_inequality_l497_497837


namespace find_y_value_l497_497488

theorem find_y_value :
  ∃ y : ℝ, (sqrt ((-6 - 2)^2 + (y - 5)^2) = 10) ∧ (y > 0) ∧ (y = 11) := 
by
  sorry

end find_y_value_l497_497488


namespace max_pq_rs_l497_497197

theorem max_pq_rs (p q r s : ℕ) 
  (h_distinct : {p, q, r, s} = {1, 2, 3, 4}) :
  p ^ q + r ^ s ≤ 83 :=
by
  sorry

end max_pq_rs_l497_497197


namespace R_l497_497803

variable (a d n : ℕ)

def arith_sum (k : ℕ) : ℕ :=
  k * (a + (k - 1) * d / 2)

def s1 := arith_sum n
def s2 := arith_sum (3 * n)
def s3 := arith_sum (5 * n)
def s4 := arith_sum (7 * n)

def R' := s4 - s3 - s2

theorem R'_depends_on_d_n : 
  R' = 2 * d * n^2 := 
by 
  sorry

end R_l497_497803


namespace max_dot_product_of_ellipse_l497_497292

/--
  Let \( O \) be the center and \( F \) be the left focus of the ellipse \( \frac{x^2}{4} + \frac{y^2}{3} = 1 \).
  Let \( P(x_0, y_0) \) be any point on the ellipse.
  Then the maximum value of the dot product of vectors \( \overrightarrow{OP} \) and \( \overrightarrow{FP} \) is \( 6 \).
-/
theorem max_dot_product_of_ellipse (x_0 y_0 : ℝ) (hP : (x_0^2) / 4 + (y_0^2) / 3 = 1) : 
  let F : ℝ × ℝ := (-1, 0)
  let O : ℝ × ℝ := (0, 0)
  let OP : ℝ × ℝ := (x_0, y_0)
  let FP : ℝ × ℝ := (x_0 + 1, y_0)
  in 
  real_inner OP FP ≤ 6 :=
by
  sorry

end max_dot_product_of_ellipse_l497_497292


namespace factorization_correct_l497_497575

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497575


namespace simplify_sqrt_fraction_sum_l497_497378

theorem simplify_sqrt_fraction_sum :
  (∃ (a b c d : ℝ), a = 15 * Real.sqrt 2 ∧ b = 10 * Real.sqrt 2 ∧ c = 7 * Real.sqrt 2 ∧ d = 4 * Real.sqrt.2 ∧ 
    (a / b + c / d = 13 / 4)) :=
begin
  let a := sqrt 450,
  let b := sqrt 200,
  let c := sqrt 98,
  let d := sqrt 56,
  have ha : a = 15 * sqrt 2 := sorry,
  have hb : b = 10 * sqrt 2 := sorry,
  have hc : c = 7 * sqrt 2 := sorry,
  have hd : d = 4 * sqrt.2 := sorry,
  use [a, b, c, d],
  split,
  { exact ha },
  split,
  { exact hb },
  split,
  { exact hc },
  split,
  { exact hd },
  have hab : a / b = 3 / 2 := sorry,
  have hcd : c / d = 7 / 4 := sorry,
  have hs : (a / b + c / d = 13 / 4),
  { rw [hab, hcd],
    norm_num },
  exact hs,
end

end simplify_sqrt_fraction_sum_l497_497378


namespace bubble_sort_probability_l497_497028

-- A sequence of 40 distinct real numbers can be sorted by bubble sort.
-- We need to show the probability given that the specific number r20 ends up at r30 after one pass is such that p + q = 931.
theorem bubble_sort_probability :
  let n := 40,
      p := 1,
      q := 930,
      sequence : Fin n → ℝ := sorry,     -- sequence of 40 distinct real numbers
      is_distinct : ∀ i j, i ≠ j → sequence i ≠ sequence j := sorry,   -- all elements are distinct
      r₃₀ := sequence 29, -- one-indexed as given problem, zero-indexed for Lean
      r₂₀ := sequence 19 in
  (p / q: ℝ) = (1/930: ℝ) ∧ p + q = 931 :=
by { 
  have : (p / q: ℝ) = (1 / 930: ℝ), sorry, 
  exact ⟨this, rfl⟩
}

end bubble_sort_probability_l497_497028


namespace find_c_k_l497_497406

theorem find_c_k (d r k : ℕ) (hn : k ≥ 3)
  (a_n := λ n, 1 + (n - 1) * d)
  (b_n := λ n, r ^ (n - 1))
  (c_n := λ n, a_n n + b_n n)
  (h1 : c_n (k - 1) = 200)
  (h2 : c_n (k + 1) = 1200) :
  c_n k = 423 :=
sorry

end find_c_k_l497_497406


namespace min_players_team_l497_497510

theorem min_players_team : Nat.lcm (Nat.lcm (Nat.lcm 8 9) 10) 11 = 7920 := 
by 
  -- The proof will be filled here.
  sorry

end min_players_team_l497_497510


namespace income_spent_on_food_l497_497035

theorem income_spent_on_food (F : ℚ) (income : ℚ) : 
  (income = 100) →
  ((income - 0.25 * income) = 75) →
  (let remaining_after_food := 75 - (F / 100) * 75 in 
    let remaining_after_rent := 0.2 * remaining_after_food in 
      remaining_after_rent = 8) →
  F = 46.67 :=
by 
  intros h1 h2 h3 
  sorry

end income_spent_on_food_l497_497035


namespace line_symmetric_circles_l497_497317

theorem line_symmetric_circles 
(a : ℝ) (x y : ℝ) 
(l : ℝ → ℝ → Prop)
(hC1 : ∀ (x y : ℝ), x^2 + y^2 = a) 
(hC2 : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 2 * a * y + 3 = 0) 
(h_sym : SymmetricCircles hC1 hC2)
: l := fun x y => 2 * x - 4 * y + 5 = 0
sorry

end line_symmetric_circles_l497_497317


namespace gnomes_telling_the_truth_l497_497943

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l497_497943


namespace propositionA_propositionB_propositionC_propositionD_correct_proposition_l497_497664

theorem propositionA (a b : ℝ) : a^2 < b^2 → a < b := sorry

theorem propositionB (a b c d : ℝ) : a < b → c < d → a * c < b * d := sorry

theorem propositionC (a b c : ℝ) : a < b → b > c → a > c := sorry

theorem propositionD (a b c d : ℝ) : a < b → c < d → a + c < b + d :=
by intros h1 h2; exact add_lt_add h1 h2

theorem correct_proposition (a b c d : ℝ) :
  (∃ a b, ¬ propositionA a b)
  ∧ (∃ a b c d, ¬ propositionB a b c d)
  ∧ (∃ a b c, ¬ propositionC a b c)
  ∧ ∀ (a b c d : ℝ), propositionD a b c d := 
by sorry

end propositionA_propositionB_propositionC_propositionD_correct_proposition_l497_497664


namespace factor_polynomial_l497_497636

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497636


namespace max_truthful_dwarfs_l497_497955

theorem max_truthful_dwarfs :
  ∃ n : ℕ, n = 1 ∧ 
    ∀ heights : Fin 7 → ℕ, 
      (heights 0 = 60 ∨ heights 1 = 61 ∨ heights 2 = 62 ∨ heights 3 = 63 ∨ heights 4 = 64 ∨ heights 5 = 65 ∨ heights 6 = 66) → 
      (∀ i j : Fin 7, i < j → heights i > heights j) → 
      ( ∃ t : Fin 7 → bool, 
        ( ∀ i j : Fin 7, t i = tt → t j = tt → i = j) ∧ 
        ( ∀ i : Fin 7, t i = tt → heights i = 60 + i ) ∧
        #(t.to_list.count tt) = n ) :=
by
  use 1
  sorry

end max_truthful_dwarfs_l497_497955


namespace total_points_needed_l497_497410

def num_students : ℕ := 25
def num_weeks : ℕ := 2
def vegetables_per_student_per_week : ℕ := 2
def points_per_vegetable : ℕ := 2

theorem total_points_needed : 
  (num_students * (vegetables_per_student_per_week * num_weeks) * points_per_vegetable) = 200 := by
  sorry

end total_points_needed_l497_497410


namespace at_most_one_dwarf_tells_truth_l497_497976

noncomputable def dwarfs_tell_truth : Prop :=
  ∀ heights: Fin 7 → ℕ,
  heights = fun i => i + 60 →
  ∀ k ∈ Finset.range 7, (heights k) ≠ (60 + k) → (∃ i ∈ Finset.range 7, i ≠ k)

theorem at_most_one_dwarf_tells_truth :
  dwarfs_tell_truth = 1 := 
sorry

end at_most_one_dwarf_tells_truth_l497_497976


namespace find_side_c_find_cos_B_minus_C_l497_497304

noncomputable def triangle_conditions (a b c : ℝ) (C : ℝ) : Prop :=
  C = π / 3 ∧ b = 8 ∧ (1 / 2 * a * b * Real.sin C) = 10 * Real.sqrt 3

theorem find_side_c (a c : ℝ) (h : triangle_conditions a 8 c (π / 3)) : c = 7 := 
by
  sorry

theorem find_cos_B_minus_C (a c B C: ℝ) (h : triangle_conditions a 8 c C) (hC: C = π / 3) : 
  Real.cos (B - C) = 13 / 14 := 
by
  sorry

end find_side_c_find_cos_B_minus_C_l497_497304


namespace area_of_quadrilateral_l497_497372

theorem area_of_quadrilateral (
  {A B C D E : Type}
  (angle_ABC_eq_90 : ∠ABC = 90)
  (angle_ACD_eq_90 : ∠ACD = 90)
  (AC_eq_24 : AC = 24)
  (CD_eq_18 : CD = 18)
  (AE_eq_6 : AE = 6)
  (radius_inscribed_circle_6 : ∃ r, r = 6 ∧ ∀ P, P ∈ circle_inscribed_in_triangle_ABD → distance(P, AB) = r ∧ distance(P, BD) = r ∧ distance(P, AD) = r)) :
  area(ABCD) = 216 + 72 * real.sqrt(2) :=
begin
  sorry
end

end area_of_quadrilateral_l497_497372


namespace range_of_f_l497_497296

def f (x : ℤ) : ℤ := x^2 + 1

theorem range_of_f : 
  (∀ x ∈ ({-1, 0, 1} : set ℤ), f x = 1 ∨ f x = 2) ∧ 
  (∀ y, (∃ x ∈ ({-1, 0, 1} : set ℤ), f x = y) → y ∈ ({1, 2} : set ℤ)) :=
by {
  sorry
}

end range_of_f_l497_497296


namespace range_of_set_l497_497126

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497126


namespace range_of_set_l497_497064

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497064


namespace elective_schemes_count_l497_497507

-- Define the conditions
def total_courses : ℕ := 10
def courses_clash {A B C : Type} (h : A = B ∧ B = C) : Prop := true

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k
noncomputable def number_of_schemes (n k : ℕ) : ℕ := ((choose n k) + (choose 3 1 * choose (n - 3) (k - 1)))

-- Prove the number of different elective schemes available for each student is 98 given the conditions
theorem elective_schemes_count :
  ∀ (A B C : Type) (h : courses_clash ⟨A, B, C⟩), 
  number_of_schemes total_courses 3 = 98 :=
by
  sorry

end elective_schemes_count_l497_497507


namespace factorization_correct_l497_497576

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497576


namespace rectangle_ratios_l497_497311

theorem rectangle_ratios (A B C D E F P Q : Type) 
    [is_rect : rectangle ABCD] (hAB : AB = 8) (hBC : BC = 4)
    (E_between_BC : E ∈ segment B C) (F_between_EC : F ∈ segment E C)
    (hBE : BE = 2 * EF) (hEF : EF = FC) 
    (AE_intersect_BD_at_P : AE ∩ BD = P)
    (AF_intersect_BD_at_Q : AF ∩ BD = Q) :
    let r := ratio BP PQ QD,
    r.1 + r.2 + r.3 = 55 := by 
  sorry

end rectangle_ratios_l497_497311


namespace range_of_set_l497_497060

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497060


namespace triangle_vertex_difference_l497_497429

open Set Function

/-- 
  Let D, E, and F be points in the Cartesian plane with coordinates:
  D(0,10), E(4,0), and F(10,0). 
  Define a vertical line intersecting DF at P and EF at Q
  such that the area of triangle PQF is 16.
  Prove that the positive difference of the x and y coordinates of point P is 8 * sqrt 2 - 10.
-/
theorem triangle_vertex_difference :
  let D := (0 : ℝ, 10 : ℝ)
  let E := (4 : ℝ, 0 : ℝ)
  let F := (10 : ℝ, 0 : ℝ)
  let P := (10 - 4 * real.sqrt 2 : ℝ, 4 * real.sqrt 2 : ℝ)
  in |P.1 - P.2| = 8 * real.sqrt 2 - 10 :=
by
  sorry

end triangle_vertex_difference_l497_497429


namespace range_of_set_l497_497096

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497096


namespace round_8_div_11_l497_497874

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497874


namespace value_of_expression_l497_497412

theorem value_of_expression : 30 - 5^2 = 5 := by
  sorry

end value_of_expression_l497_497412


namespace david_wins_2011th_even_l497_497395

theorem david_wins_2011th_even :
  ∃ n : ℕ, (∃ k : ℕ, k = 2011 ∧ n = 2 * k) ∧ (∀ a b : ℕ, a < b → a + b < b * a) ∧ (n % 2 = 0) := 
sorry

end david_wins_2011th_even_l497_497395


namespace sequence_a113_l497_497781

theorem sequence_a113 {a : ℕ → ℝ} 
  (h1 : ∀ n, a n > 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n, (a (n+1))^2 + (a n)^2 = 2 * n * ((a (n+1))^2 - (a n)^2)) :
  a 113 = 15 :=
sorry

end sequence_a113_l497_497781


namespace at_most_one_dwarf_tells_truth_l497_497972

noncomputable def dwarfs_tell_truth : Prop :=
  ∀ heights: Fin 7 → ℕ,
  heights = fun i => i + 60 →
  ∀ k ∈ Finset.range 7, (heights k) ≠ (60 + k) → (∃ i ∈ Finset.range 7, i ≠ k)

theorem at_most_one_dwarf_tells_truth :
  dwarfs_tell_truth = 1 := 
sorry

end at_most_one_dwarf_tells_truth_l497_497972


namespace line_plane_intersection_l497_497266

-- The definition of the line and plane setup
variables {Line Plane : Type} [HasSubset Line Plane] [HasIntersection Line Line] [HasIntersection Plane Plane]
variables {a b : Line} {α β : Plane}

-- Conditions
axiom line_in_plane (a : Line) (α : Plane) : a ⊆ α
axiom line_in_plane2 (b : Line) (β : Plane) : b ⊆ β

-- Theorem statement
theorem line_plane_intersection (a_inter_b : a ∩ b ≠ ∅) : α ∩ β ≠ ∅ :=
begin
  sorry
end

end line_plane_intersection_l497_497266


namespace factor_polynomial_l497_497630

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497630


namespace smallest_number_l497_497162

theorem smallest_number : 
  ∀ (a b c d : ℝ), 
    a = 0 → b = -3 → c = -real.pi → d = -real.sqrt 3 → 
    c < b ∧ b < d ∧ d < a → 
    ∀ x ∈ {a, b, c, d}, c ≤ x := 
begin
  intros a b c d ha hb hc hd h,
  cases h with hcb h,
  cases h with hbd hda,
  intros x hx,
  rw [ha, hb, hc, hd] at *,
  cases hx,
  { rw hx, exact hcb,},
  { cases hx,
    { rw hx, exact le_of_lt (lt_trans hcb hbd), },
    { cases hx,
      { rw hx, exact le_refl _,},
      { rw hx, exact lt_trans (lt_trans hcb hbd) hda }, } }
end

end smallest_number_l497_497162


namespace range_of_numbers_is_six_l497_497151

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497151


namespace proof_tan_problem_l497_497682

noncomputable def tan_addition_formula (α β : ℝ) : ℝ :=
  (Real.tan α + Real.tan β) / (1 - (Real.tan α * Real.tan β))

theorem proof_tan_problem (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) 
  (h_tan_roots : (Polynomial.roots (Polynomial.X^2 - 5 * Polynomial.X + 6)).mfilter (λ x, Real.tan α = x ∨ Real.tan β = x) = [Real.tan α, Real.tan β]) :
  α + β = 3 * Real.pi / 4 ∧ ¬(∃ x, Real.tan (2 * (α + β)) = x) :=
by
  sorry

end proof_tan_problem_l497_497682


namespace cut_out_pieces_border_each_other_l497_497823

theorem cut_out_pieces_border_each_other :
  ∀ (n m : ℕ) (unit_squares l_pieces : ℕ),
  n = 55 → m = 55 →
  unit_squares = 500 → l_pieces = 400 →
  n * m = 55 * 55 →
  unit_squares + 3 * l_pieces > (n * m) / 2 →
  ∃ (i j : ℕ), (i < n) ∧ (j < m) ∧
  (unit_squares * (i, j) + 3 * l_pieces * (i, j) > 2 * 55) →
  ∃ (a b : ℕ), (a < n - 1) ∧ (b < m - 1) ∧
    adjacent ((a, b), (a + 1, b)) ∧ adjacent ((a, b), (a, b + 1)) :=
by
  sorry

end cut_out_pieces_border_each_other_l497_497823


namespace magnitude_of_vector_sum_l497_497676

-- Define vector a
def a : ℝ × ℝ := (1, 0)

-- Define vector b
def b : ℝ × ℝ := (2, 1)

-- Define the scalar multiplication of a vector
def smul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

-- Define vector addition
def vadd (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- State the theorem
theorem magnitude_of_vector_sum : magnitude (vadd a (smul 3 b)) = Real.sqrt 58 := 
by sorry

end magnitude_of_vector_sum_l497_497676


namespace sqrt_a_minus_2_meaningful_l497_497744

theorem sqrt_a_minus_2_meaningful (a : ℝ) (h : 0 ≤ a - 2) : 2 ≤ a :=
by
  sorry

end sqrt_a_minus_2_meaningful_l497_497744


namespace set_intersection_complement_l497_497674

open Set

theorem set_intersection_complement {x : ℝ} :
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | 2^x > 1}
  let C := {x : ℝ | -2 < x ∧ x ≤ 0}
  A ∩ (Bᶜ) = C :=
by
  sorry

end set_intersection_complement_l497_497674


namespace min_le_one_fourth_sum_max_ge_four_ninths_sum_l497_497723

variable (a b c : ℝ)

theorem min_le_one_fourth_sum
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots : b^2 - 4 * a * c ≥ 0) :
  min a (min b c) ≤ 1 / 4 * (a + b + c) :=
sorry

theorem max_ge_four_ninths_sum
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots : b^2 - 4 * a * c ≥ 0) :
  max a (max b c) ≥ 4 / 9 * (a + b + c) :=
sorry

end min_le_one_fourth_sum_max_ge_four_ninths_sum_l497_497723


namespace find_m_l497_497489

def ListOfIntegers := List ℤ

noncomputable def problem_statement (L : ListOfIntegers) (m : ℤ) : Prop :=
  -- Condition 1: Mode is 30 and mean is 25
  (30 ∈ L.mode) ∧ (L.sum / L.length = 25) ∧ 
  -- Condition 2: Smallest number is 15
  (L.minimum = 15) ∧
  -- Condition 3: Median is m and is a member of the list
  (L.median = some m) ∧ (m ∈ L) ∧
  -- Condition 4: Replacement of m by m+12 results in mean 27 and median m+12
  let L1 := (L.remove m) in
  let L1 := (L1.insert (m + 12)) in
  (L1.sum / L1.length = 27) ∧ (L1.median = some (m + 12)) ∧
  -- Condition 5: Replacement of m by m−10 results in median m−5
  let L2 := (L.remove m) in
  let L2 := (L2.insert (m - 10)) in
  (L2.median = some (m-5))

theorem find_m : ∃ (L : ListOfIntegers) (m : ℤ), problem_statement L m ∧ m = 30 := 
sorry

end find_m_l497_497489


namespace _l497_497667

noncomputable theorem SL_square_iff (L : ℕ) : 
  (∃ k : ℕ, (∑ n in finset.range (L + 1), n / 2) = k^2) ↔ L = 1 ∨ (∃ k : ℕ, L = 2 * k) := by
  sorry

end _l497_497667


namespace division_rounded_l497_497854

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497854


namespace at_most_one_dwarf_tells_truth_l497_497973

noncomputable def dwarfs_tell_truth : Prop :=
  ∀ heights: Fin 7 → ℕ,
  heights = fun i => i + 60 →
  ∀ k ∈ Finset.range 7, (heights k) ≠ (60 + k) → (∃ i ∈ Finset.range 7, i ≠ k)

theorem at_most_one_dwarf_tells_truth :
  dwarfs_tell_truth = 1 := 
sorry

end at_most_one_dwarf_tells_truth_l497_497973


namespace no_non_square_number_with_triple_product_divisors_l497_497464

theorem no_non_square_number_with_triple_product_divisors (N : ℕ) (h_non_square : ∀ k : ℕ, k * k ≠ N) : 
  ¬ (∃ t : ℕ, ∃ d : Finset (Finset ℕ), (∀ s ∈ d, s.card = 3) ∧ (∀ s ∈ d, s.prod id = t)) := 
sorry

end no_non_square_number_with_triple_product_divisors_l497_497464


namespace ellipse_slope_ratio_l497_497776

theorem ellipse_slope_ratio (a b : ℝ) (k1 k2 : ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a > 2)
  (h4 : k2 = k1 * (a^2 + 5) / (a^2 - 1)) : 
  1 < (k2 / k1) ∧ (k2 / k1) < 3 :=
by
  sorry

end ellipse_slope_ratio_l497_497776


namespace fraction_to_three_decimal_places_l497_497867

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497867


namespace max_irreducible_fractions_in_interval_l497_497689

open Real

-- The main theorem statement
theorem max_irreducible_fractions_in_interval (n : ℕ) (h_pos : n > 0) : 
  ∀ I : set ℝ, 
    (∃ a b : ℝ, I = {x | a < x ∧ x < b} ∧ b - a = 1 / (n : ℝ)) → 
    ∃ s : finset (ℚ), s.card ≤ (n + 1) / 2 ∧ ∀ q ∈ s, (q.den ∈ (set.Icc 1 n)) ∧ ((q : ℝ) ∈ I) :=
sorry

end max_irreducible_fractions_in_interval_l497_497689


namespace range_of_numbers_is_six_l497_497113

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497113


namespace find_a_of_inequality_solution_l497_497297

theorem find_a_of_inequality_solution (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 ↔ x^2 - a * x < 0) → a = 1 := 
by 
  sorry

end find_a_of_inequality_solution_l497_497297


namespace plane_centroid_l497_497808

theorem plane_centroid (a b : ℝ) (h : 1 / a ^ 2 + 1 / b ^ 2 + 1 / 25 = 1 / 4) :
  let p := a / 3
  let q := b / 3
  let r := 5 / 3
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 369 / 400 :=
by
  sorry

end plane_centroid_l497_497808


namespace range_of_numbers_is_six_l497_497147

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497147


namespace calculate_difference_square_l497_497277

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end calculate_difference_square_l497_497277


namespace max_edges_intersected_by_plane_l497_497440

theorem max_edges_intersected_by_plane (p : ℕ) (h_pos : p > 0) : ℕ :=
  let vertices := 2 * p
  let base_edges := p
  let lateral_edges := p
  let total_edges := 3 * p
  total_edges

end max_edges_intersected_by_plane_l497_497440


namespace range_of_set_l497_497076

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497076


namespace range_of_numbers_is_six_l497_497109

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497109


namespace hexagon_area_half_triangle_area_l497_497772

open EuclideanGeometry

theorem hexagon_area_half_triangle_area
  (A B C K L M Q S T : Point)
  (hacute : acute_triangle A B C)
  (hK_mid : midpoint K A B)
  (hL_mid : midpoint L B C)
  (hM_mid : midpoint M C A)
  (hQ_perp : perpendicular K BC)
  (hS_perp : perpendicular L CA)
  (hT_perp : perpendicular M AB)
  (hHexagon : hexagon K Q L S M T)
  : area_hexagon K Q L S M T = (1 / 2) * area_triangle A B C :=
begin
  sorry
end

end hexagon_area_half_triangle_area_l497_497772


namespace tangent_parallel_to_BC_l497_497327

theorem tangent_parallel_to_BC {A B C D M X H P : Point} (h_isosceles : A = C) (h_BC_gt_AB : dist B C > dist A B)
  (h_midpoints_BCM : midpoint B C = D) (h_midpoints_AB : midpoint A B = M) 
  (h_BX_perp_AC : ⊥ X (line_through B C)) (h_XD_parallel_AB : parallel X D (line_through A B))
  (h_BX_inter_AD_H : intersect B X A D = H) (h_P_circumcircle_AHX : is_on_circumcircle P (triangle A H X)) :
  parallel (tangent_at A (circumcircle (triangle A M P))) (line_through B C) :=
by
  sorry

end tangent_parallel_to_BC_l497_497327


namespace prove_n_eq_1_l497_497332

-- Definitions of the given conditions
def is_prime (x : ℕ) : Prop := Nat.Prime x

variable {p q r n : ℕ}
variable (hp : is_prime p) (hq : is_prime q) (hr : is_prime r)
variable (hn_pos : n > 0)
variable (h_eq : p^n + q^n = r^2)

-- Statement to prove
theorem prove_n_eq_1 : n = 1 :=
  sorry

end prove_n_eq_1_l497_497332


namespace pqrs_l497_497746

theorem pqrs(p q r s t u : ℤ) :
  (729 * (x : ℤ) * x * x + 64 = (p * x * x + q * x + r) * (s * x * x + t * x + u)) →
  p = 9 → q = 4 → r = 0 → s = 81 → t = -36 → u = 16 →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 := by
  intros h1 hp hq hr hs ht hu
  sorry

end pqrs_l497_497746


namespace coin_weight_problem_l497_497017

-- Define the weight of one 5-jiao coin in grams
def weight_of_coin : ℕ := 3

-- Define the conversion factors
def grams_per_kilogram : ℕ := 1000
def grams_per_ton : ℕ := 1000000

-- Prove the stated questions
theorem coin_weight_problem :
  (weight_of_coin * 10000 = 30000) ∧ (30000 / grams_per_kilogram = 30) ∧
  (weight_of_coin * 10000000 = 30000000) ∧ (30000000 / grams_per_ton = 30) ∧
  (weight_of_coin * 200000000 = 600000000) ∧ (600000000 / grams_per_ton = 600) :=
by {
  -- All intermediate calculations can be skipped, we only state the proof goals
  sorry,
}

end coin_weight_problem_l497_497017


namespace constant_angle_between_chords_l497_497423

-- Define the basic entities: points, lines, circles, and angles.

structure Point := (x : ℝ) (y : ℝ)
structure Circle := (center : Point) (radius : ℝ)
structure Line := (p1 : Point) (p2 : Point)

-- Define the intersection point and arbitrary secants.
variables {A C C1 D D1 O O1 : Point}
variables {circle1 circle2 : Circle}

-- Conditions provided: Intersection of two circles.
axiom intersection_circles : circle1.center = A ∧ circle2.center = A

-- Definitions of points intersected by arbitrary secants.
axiom secant1 : Line
axiom secant2 : Line
axiom points_on_circles : secant1.p1 = C ∧ secant1.p2 = C1 ∧ secant2.p1 = D ∧ secant2.p2 = D1

-- Define that the two circles' centers are O and O1.
axiom centers : circle1.center = O ∧ circle2.center = O1 

-- Define that the angle between two chords is constant.
theorem constant_angle_between_chords :
  ∀ (circle1 circle2 : Circle) (O O1 A C C1 D D1 : Point), 
    (circle1.center = O) ∧ (circle2.center = O1) ∧
    (O ≠ O1) ∧
    (∃ A, (A ∈ circle1) ∧ (A ∈ circle2)) ∧
    (∃ secant1 secant2 : Line, 
      (secant1.p1 = C ∧ secant1.p2 = C1) ∧ 
      (secant2.p1 = D ∧ secant2.p2 = D1)) 
    → 
    ∃ (angle_COD1 : ℝ) (angle_D1C1D : ℝ), 
      (angle_COD1 + angle_D1C1D = 180) :=
sorry

end constant_angle_between_chords_l497_497423


namespace range_of_set_is_six_l497_497086

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497086


namespace calculate_L_l497_497557

theorem calculate_L (T H K : ℝ) (hT : T = 2 * Real.sqrt 5) (hH : H = 10) (hK : K = 2) :
  L = 100 :=
by
  let L := 50 * T^4 / (H^2 * K)
  have : T = 2 * Real.sqrt 5 := hT
  have : H = 10 := hH
  have : K = 2 := hK
  sorry

end calculate_L_l497_497557


namespace find_common_ratio_l497_497245

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

noncomputable def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

noncomputable def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a 0 else a 0 * (1 - q ^ (n + 1)) / (1 - q)

noncomputable def integral (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in[a, b], f x

theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  increasing_sequence a →
  a 3 = 8 →
  geometric_sum a q 2 = integral (λ x, 4 * x + 3) 0 2 →
  q = 2 := by
  sorry

end find_common_ratio_l497_497245


namespace max_truthful_gnomes_l497_497996

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

end max_truthful_gnomes_l497_497996


namespace chip_placement_l497_497740

def red_chips : ℕ := 4
def blue_chips : ℕ := 4
def green_chips : ℕ := 4

def grid_size : ℕ := 4

def no_adjacent_same_color (placement : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, ∀ color, 
  (i < grid_size ∧ j < grid_size) →
  (∀ Δi Δj, (Δi = 1 ∨ Δi = -1 ∨ Δi = 0) → (Δj = 1 ∨ Δj = -1 ∨ Δj = 0) → 
    (Δi ≠ 0 ∨ Δj ≠ 0) → 
    (i + Δi < grid_size) → (j + Δj < grid_size) → 
    placement i j ≠ placement (i + Δi) (j + Δj))

def exactly_one_color_per_row_column (placement : ℕ → ℕ → ℕ) : Prop :=
  ∀ i, ∃! j, (placement i j = 1 ∧ placement i j = 2 ∧ placement i j = 3) ∧
  ∀ j, ∃! i, (placement i j = 1 ∧ placement i j = 2 ∧ placement i j = 3)

theorem chip_placement (placement : ℕ → ℕ → ℕ) :
  (no_adjacent_same_color placement ∧ exactly_one_color_per_row_column placement) →
  number_of_ways = 4 :=
sorry

end chip_placement_l497_497740


namespace range_of_numbers_is_six_l497_497148

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497148


namespace range_of_set_l497_497045

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497045


namespace range_of_set_l497_497092

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497092


namespace gnomes_telling_the_truth_l497_497945

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l497_497945


namespace weights_sum_l497_497556

theorem weights_sum (e f g h : ℕ) (h₁ : e + f = 280) (h₂ : f + g = 230) (h₃ : e + h = 300) : g + h = 250 := 
by 
  sorry

end weights_sum_l497_497556


namespace ranking_of_ABC_l497_497198

-- Define the ranking type
inductive Rank
| first
| second
| third

-- Define types for people
inductive Person
| A
| B
| C

open Rank Person

-- Alias for ranking of each person
def ranking := Person → Rank

-- Define the conditions
def A_statement (r : ranking) : Prop := r A ≠ first
def B_statement (r : ranking) : Prop := A_statement r ≠ false
def C_statement (r : ranking) : Prop := r C ≠ third

def B_lied : Prop := true
def C_told_truth : Prop := true

-- The equivalent problem, asked to prove the final result
theorem ranking_of_ABC (r : ranking) : 
  (B_lied ∧ C_told_truth ∧ B_statement r = false ∧ C_statement r = true) → 
  (r A = first ∧ r B = third ∧ r C = second) :=
sorry

end ranking_of_ABC_l497_497198


namespace factor_polynomial_l497_497632

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497632


namespace definite_integral_l497_497169

-- Definitions based on the conditions in part a)
def integrand (x : ℝ) : ℝ := (3 - 7 * x^2) * Real.cos (2 * x)

-- The statement to be proved
theorem definite_integral : 
  ∫ x in 0..2 * Real.pi, integrand x = -7 * Real.pi :=
by
  sorry

end definite_integral_l497_497169


namespace integral_square_geq_integral_cube_l497_497454

noncomputable theory
open Real

variable {f : ℝ → ℝ}

-- Define the conditions
def cond1 : Prop := ∀ x ∈ Icc (0 : ℝ) 1, ContinuousAt f x
def cond2 : Prop := f 0 = 0
def cond3 : Prop := ∀ x ∈ Icc (0 : ℝ) 1, fderiv ℝ f x ∈ Icc (0 : ℝ) 1

-- Define the theorem to be proved
theorem integral_square_geq_integral_cube 
  (h1 : cond1) 
  (h2 : cond2) 
  (h3 : cond3) : 
  ( ∫ x in (0 : ℝ)..1, f x )^2 ≥ ∫ x in (0 : ℝ)..1, f x^3 := 
by
sorrry

end integral_square_geq_integral_cube_l497_497454


namespace round_fraction_to_three_decimal_l497_497913

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497913


namespace fraction_rounded_equals_expected_l497_497917

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497917


namespace checkerboard_29_black_squares_l497_497537

axiom alternating_colors (m n : ℕ) : bool

def is_color (i j : ℕ) (color : bool) : bool :=
  (alternating_colors i j) = color

def is_black (i j : ℕ) : bool :=
  is_color i j true

def count_squares (n : ℕ) : ℕ :=
  (n * n) / 2 + (if n % 2 == 1 then (n / 2) + 1 + (n / 2) else 0)

def black_squares_29 : ℕ :=
  count_squares 29

theorem checkerboard_29_black_squares : black_squares_29 = 420 := 
  by
  sorry

end checkerboard_29_black_squares_l497_497537


namespace y_intercept_line_a_l497_497357

open Real

-- Define line a which is parallel to the line y = 2x + 4
def line_a (x : ℝ) : ℝ := 2 * x + b

-- Define the condition that line a passes through the point (2, 5)
def passes_through (line : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  line p.1 = p.2

-- Proof statement
theorem y_intercept_line_a :
  ∃ b : ℝ, passes_through (line_a b) (2, 5) ∧ b = 1 :=
sorry

end y_intercept_line_a_l497_497357


namespace area_of_quadrilateral_xyza_l497_497775

def trisect (a b c : ℝ) : Prop := b - a = c - b

noncomputable def area_of_xyza : ℝ :=
let PS := 6
let PR := 3
let P := (0, 0)
let S := (0, PS)
let Q := (PR, PS)
let R := (PR, 0)
let T := (P.1, PS / 3)
let U := (P.1, 2 * PS / 3)
let V := (R.1, PS / 3)
let W := (R.1, 2 * PS / 3)

let X := ((P.1 + T.1) / 2, (P.2 + T.2) / 2)
let Y := ((T.1 + U.1) / 2, (T.2 + U.2) / 2)
let Z := ((U.1 + S.1) / 2, (U.2 + S.2) / 2)
let A := ((P.1 + S.1) / 2, (P.2 + S.2) / 2)

(X.1 - Y.1) * (X.2 - Z.2)

theorem area_of_quadrilateral_xyza : area_of_xyza = 3 :=
by {
  have h₁ : trisect P.2 T.2 U.2 := by simp [trisect, P, T, U],
  have h₂ : trisect R.2 V.2 W.2 := by simp [trisect, R, V, W],
  have h₃ : X.1 = Y.1 := by simp [X, Y, T, U, P],
  have h₄ : X.1 = Z.1 := by simp [X, Z, U, S, P],
  have h₅ : X.1 * Z.2 = 3 := by simp [X, Z, P, S, T, U],
  sorry
}

end area_of_quadrilateral_xyza_l497_497775


namespace vehicle_capacity_rental_plans_l497_497774

variables (a b x y : ℕ)

/-- Conditions -/
axiom cond1 : 2*x + y = 11
axiom cond2 : x + 2*y = 13

/-- Resulting capacities for each vehicle type -/
theorem vehicle_capacity : 
  x = 3 ∧ y = 5 :=
by
  sorry

/-- Rental plans for transporting 33 tons of drugs -/
theorem rental_plans :
  3*a + 5*b = 33 ∧ ((a = 6 ∧ b = 3) ∨ (a = 1 ∧ b = 6)) :=
by
  sorry

end vehicle_capacity_rental_plans_l497_497774


namespace max_truthful_dwarfs_l497_497959

theorem max_truthful_dwarfs :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  (∃ n, n = 1 ∧ ∀ i, i < 7 → i ≠ n → heights[i] ≠ (heights[i] + 1)) :=
by
  let heights := [60, 61, 62, 63, 64, 65, 66]
  existsi 1
  intro i
  intro h
  intro hi
  sorry

end max_truthful_dwarfs_l497_497959


namespace at_most_one_dwarf_tells_truth_l497_497977

noncomputable def dwarfs_tell_truth : Prop :=
  ∀ heights: Fin 7 → ℕ,
  heights = fun i => i + 60 →
  ∀ k ∈ Finset.range 7, (heights k) ≠ (60 + k) → (∃ i ∈ Finset.range 7, i ≠ k)

theorem at_most_one_dwarf_tells_truth :
  dwarfs_tell_truth = 1 := 
sorry

end at_most_one_dwarf_tells_truth_l497_497977


namespace remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12_l497_497209

theorem remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12 :
  (7 * 11 ^ 24 + 2 ^ 24) % 12 = 11 := by
sorry

end remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12_l497_497209


namespace grid_points_l497_497824

theorem grid_points (points : set (ℕ × ℕ)) (H : points.card = 100) :
  ∃ A B ∈ points, let { A = (x₁, y₁), B = (x₂, y₂) } in 
  card { p ∈ points | x₁ ≤ p.1 ∧ p.1 ≤ x₂ ∧ y₁ ≤ p.2 ∧ p.2 ≤ y₂ } ≥ 20 :=
begin
  sorry
end

end grid_points_l497_497824


namespace complement_of_angle_is_15424_l497_497520

-- Define the given angle
def given_angle : ℝ := 25 + 36 / 60

-- Define the formula for the complement
def complement (angle : ℝ) : ℝ := 180 - angle

-- State the theorem claiming that the complement of the given angle is 154°24'
theorem complement_of_angle_is_15424 : complement given_angle = 154 + 24 / 60 := 
sorry

end complement_of_angle_is_15424_l497_497520


namespace order_of_p_q_r_l497_497226

theorem order_of_p_q_r (p q r : ℝ) (h1 : p = Real.sqrt 2) (h2 : q = Real.sqrt 7 - Real.sqrt 3) (h3 : r = Real.sqrt 6 - Real.sqrt 2) :
  p > r ∧ r > q :=
by
  sorry

end order_of_p_q_r_l497_497226


namespace math_problem_l497_497291

theorem math_problem 
  (x y z : ℚ)
  (h1 : 4 * x - 5 * y - z = 0)
  (h2 : x + 5 * y - 18 * z = 0)
  (hz : z ≠ 0) :
  (x^2 + 4 * x * y) / (y^2 + z^2) = 3622 / 9256 := 
sorry

end math_problem_l497_497291


namespace triangles_APQ_ADB_similar_l497_497768

-- Define the points and triangles according to the conditions.
variable (P Q A B C D: Type)

-- Given conditions.
axiom right_triangle_ABC : ∠ C A B = 90
axiom D_on_BC : D ∈ line_segment B C
axiom CP_altitude : is_altitude C P (triangle A B C)
axiom CQ_altitude : is_altitude C Q (triangle A D C)

-- Now state what needs to be proven.
theorem triangles_APQ_ADB_similar :
  ∆ A P Q ∼ ∆ A D B :=
sorry

end triangles_APQ_ADB_similar_l497_497768


namespace factor_polynomial_l497_497631

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497631


namespace count_f_eq_n_up_to_1988_l497_497026

def f : ℕ → ℕ
| 1       := 1
| 3       := 3
| (2 * n) := f n
| (4 * n + 1) := 2 * f (2 * n + 1) - f n
| (4 * n + 3) := 3 * f (2 * n + 1) - 2 * f n
| _        := 0 -- necessary to handle the non-pattern matching cases

theorem count_f_eq_n_up_to_1988 :
  {n : ℕ | n ≤ 1988 ∧ f n = n}.toFinset.card = 26 := 
begin
  sorry
end

end count_f_eq_n_up_to_1988_l497_497026


namespace range_of_set_l497_497073

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497073


namespace max_truthful_dwarfs_le_one_l497_497983

def dwarf_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

theorem max_truthful_dwarfs_le_one (heights : List ℕ) (h_lengths : heights.length = 7) (h_sorted : heights.sorted (· > ·)) :
  ∃ n, (n ≤ 1) ∧ ∀ i, i ∈ (Finset.range 7).to_list → (heights[i] ≠ dwarf_height_claims[i] → (1 = n)) :=
  sorry

end max_truthful_dwarfs_le_one_l497_497983


namespace distinct_pairs_count_l497_497551

theorem distinct_pairs_count :
  ∃ (n : ℕ), n = 2 ∧ 
  ∀ (x y : ℝ), (x = 3 * x^2 + y^2) ∧ (y = 3 * x * y) → 
    ((x = 0 ∧ y = 0) ∨ (x = 1 / 3 ∧ y = 0)) :=
by
  sorry

end distinct_pairs_count_l497_497551


namespace percentage_chinese_stamps_l497_497821

-- Definitions
def total_stamps : ℕ := 100
def us_stamps : ℕ := 20% * total_stamps
def japanese_stamps : ℕ := 45
def chinese_stamps : ℕ := (total_stamps - (us_stamps + japanese_stamps)).to_nat

-- Theorem Statement
theorem percentage_chinese_stamps : (chinese_stamps / total_stamps) * 100 = 35 := sorry

end percentage_chinese_stamps_l497_497821


namespace fraction_rounding_l497_497882

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497882


namespace volunteer_comprehensive_score_is_92_l497_497476

noncomputable def written_score : ℝ := 90
noncomputable def trial_lecture_score : ℝ := 94
noncomputable def interview_score : ℝ := 90

noncomputable def written_weight : ℝ := 0.3
noncomputable def trial_lecture_weight : ℝ := 0.5
noncomputable def interview_weight : ℝ := 0.2

noncomputable def comprehensive_score : ℝ :=
  written_score * written_weight +
  trial_lecture_score * trial_lecture_weight +
  interview_score * interview_weight

theorem volunteer_comprehensive_score_is_92 :
  comprehensive_score = 92 := by
  sorry

end volunteer_comprehensive_score_is_92_l497_497476


namespace rohit_distance_from_start_l497_497374

-- Define Rohit's movements
def rohit_walked_south (d: ℕ) : ℕ := d
def rohit_turned_left_walked_east (d: ℕ) : ℕ := d
def rohit_turned_left_walked_north (d: ℕ) : ℕ := d
def rohit_turned_right_walked_east (d: ℕ) : ℕ := d

-- Rohit's total movement in east direction
def total_distance_moved_east (d1 d2 : ℕ) : ℕ :=
  rohit_turned_left_walked_east d1 + rohit_turned_right_walked_east d2

-- Prove the distance from the starting point is 35 meters
theorem rohit_distance_from_start : 
  total_distance_moved_east 20 15 = 35 :=
by
  sorry

end rohit_distance_from_start_l497_497374


namespace cost_prices_of_products_l497_497020

-- Define the variables and conditions from the problem
variables (x y : ℝ)

-- Theorem statement
theorem cost_prices_of_products (h1 : 20 * x + 15 * y = 380) (h2 : 15 * x + 10 * y = 280) : 
  x = 16 ∧ y = 4 :=
by
  -- Proof omitted
  sorry

end cost_prices_of_products_l497_497020


namespace fraction_rounding_l497_497891

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497891


namespace PR2_plus_QR2_constant_l497_497319

theorem PR2_plus_QR2_constant {P Q R S : Type*} [metric_space P]
  (PQ QR PR : ℝ) (PS : Segment P Q)
  (hPQ : PQ = 10)
  (hPS : PS.length = 7)
  (midpoint_S : S = (Q + R) / 2)
  (N : ℝ)
  (n : ℝ)
  (hN : N = PR^2 + QR^2)
  (hn : n = PR^2 + QR^2) :
  N - n = 0 :=
by
  sorry

end PR2_plus_QR2_constant_l497_497319


namespace range_of_set_is_six_l497_497090

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497090


namespace range_of_set_l497_497095

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497095


namespace trading_organization_increase_price_l497_497513

theorem trading_organization_increase_price 
  (initial_moisture_content : ℝ)
  (final_moisture_content : ℝ)
  (solid_mass : ℝ)
  (initial_total_mass final_total_mass : ℝ) :
  initial_moisture_content = 0.99 → 
  final_moisture_content = 0.98 →
  initial_total_mass = 100 →
  solid_mass = initial_total_mass * (1 - initial_moisture_content) →
  final_total_mass = solid_mass / (1 - final_moisture_content) →
  (final_total_mass / initial_total_mass) = 0.5 →
  100 * (1 - (final_total_mass / initial_total_mass)) = 100 :=
by sorry

end trading_organization_increase_price_l497_497513


namespace sequence_never_2016_l497_497356

noncomputable def a : ℕ → ℤ
| 0       := 1
| 1       := 2
| (n + 2) := if (a (n + 1) * a n) % 2 = 0 then 5 * a (n + 1) - 3 * a n else a (n + 1) - a n

theorem sequence_never_2016 (n : ℕ) : a n ≠ 2016 :=
sorry -- Proof goes here

end sequence_never_2016_l497_497356


namespace ellipse_equation_angle_AFB_constant_l497_497687

noncomputable def ellipse : Prop := 
  ∃ (a b c : ℝ), 
    a > b ∧ b > 0 ∧ 
    2 * a = 4 ∧ 
    c = a * (√3 / 2) ∧ 
    b = √(a^2 - c^2) ∧ 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_equation :
  ellipse → ∀ x y : ℝ, x^2 / 4 + y^2 = 1 := 
by {
  assume h,
  rcases h with ⟨a, b, c, ha, hb, h1, h2, h3, hₑq⟩,
  rw h1 at h1 ⊢,
  rw h2,
  exact hₑq,
  sorry
}

theorem angle_AFB_constant :
  ellipse → ∀ P : ℝ × ℝ, ¬ (P.fst = 2 ∨ P.fst = -2) →
  ∀ A B F : ℝ × ℝ, 
    A = (2, (1 - (P.fst / 2)) / P.snd) →
    B = (-2, (1 + (P.fst / 2)) / P.snd) →
    F = (√3, 0) →
    (atan ((A.snd - F.snd) / (A.fst - F.fst)) -
     atan ((B.snd - F.snd) / (B.fst - F.fst))
     ) = π / 2 := 
by {
  assume h P hP A B F hA hB hF,
  sorry
}

end ellipse_equation_angle_AFB_constant_l497_497687


namespace select_four_blocks_l497_497042

open Finset

theorem select_four_blocks : 
  let n : ℕ := 6
  let k : ℕ := 4
  let combinations_r_c : ℕ := choose n k * choose n k
  let assign_blocks : ℕ := fact k
  let answer : ℕ := combinations_r_c * assign_blocks
  in 
  answer = 5400 :=
by
  let n := 6
  let k := 4
  let combinations_r_c := choose n k * choose n k
  let assign_blocks := fact k
  let answer := combinations_r_c * assign_blocks
  show answer = 5400
  sorry

end select_four_blocks_l497_497042


namespace minimum_distance_between_parallel_lines_l497_497695

-- Define the two lines l1 and l2
def l1 : Prop := ∀ x y : ℝ, x + 3 * y = 9
def l2 : Prop := ∀ x y : ℝ, x + 3 * y = -1

-- Define the distance formula between two lines
noncomputable def distance_between_lines : ℝ := 
  (|-9 - 1|: ℝ) / Math.sqrt (1 + 9)

-- The final goal is to prove that this distance is equal to sqrt(10)
theorem minimum_distance_between_parallel_lines : distance_between_lines = Real.sqrt 10 :=
  by sorry

end minimum_distance_between_parallel_lines_l497_497695


namespace continuous_function_solutions_l497_497650

noncomputable def continuous_solutions (f : ℝ → ℝ) : Prop :=
  continuous f ∧ ∀ x y : ℝ, f (x^2 - y^2) = f x ^ 2 + f y ^ 2

theorem continuous_function_solutions :
  ∀ f : ℝ → ℝ,
  continuous_solutions f →
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1 / 2) :=
by
  intro f
  intro h
  sorry

end continuous_function_solutions_l497_497650


namespace jessica_games_attended_l497_497419

def total_games : ℕ := 6
def games_missed_by_jessica : ℕ := 4

theorem jessica_games_attended : total_games - games_missed_by_jessica = 2 := by
  sorry

end jessica_games_attended_l497_497419


namespace Mitzi_score_l497_497270

-- Definitions based on the conditions
def Gretchen_score : ℕ := 120
def Beth_score : ℕ := 85
def average_score (total_score : ℕ) (num_bowlers : ℕ) : ℕ := total_score / num_bowlers

-- Theorem stating that Mitzi's bowling score is 113
theorem Mitzi_score (m : ℕ) (h : average_score (Gretchen_score + m + Beth_score) 3 = 106) :
  m = 113 :=
by sorry

end Mitzi_score_l497_497270


namespace largest_sum_of_ABC_l497_497783

-- Define the variables and the conditions
def A := 533
def B := 5
def C := 1

-- Define the product condition
def product_condition : Prop := (A * B * C = 2665)

-- Define the distinct positive integers condition
def distinct_positive_integers_condition : Prop := (A > 0 ∧ B > 0 ∧ C > 0 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C)

-- State the theorem
theorem largest_sum_of_ABC : product_condition → distinct_positive_integers_condition → A + B + C = 539 := by
  intros _ _
  sorry

end largest_sum_of_ABC_l497_497783


namespace first_guinea_pig_food_l497_497842

theorem first_guinea_pig_food (x : ℕ) (h1 : ∃ x : ℕ, R = x + 2 * x + (2 * x + 3)) (hp : 13 = x + 2 * x + (2 * x + 3)) : x = 2 :=
by
  sorry

end first_guinea_pig_food_l497_497842


namespace min_distance_circle_to_line_l497_497720

def line (x y : ℝ) : Prop := x - y + 4 = 0

def circle (P : ℝ × ℝ) : Prop := (P.1 - 1)^2 + (P.2 - 1)^2 = 4

theorem min_distance_circle_to_line :
  ∀ (P : ℝ × ℝ), circle P → ∃ (min_d : ℝ), min_d = 2 * sqrt 2 - 2 ∧
  (∀ (Q : ℝ × ℝ), circle Q → line Q.1 Q.2 → dist P Q >= min_d) :=
sorry

end min_distance_circle_to_line_l497_497720


namespace factor_polynomial_l497_497621

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497621


namespace difference_of_squares_l497_497286

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end difference_of_squares_l497_497286


namespace fraction_to_three_decimal_places_l497_497858

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497858


namespace range_of_set_l497_497061

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497061


namespace numbers_are_perfect_squares_l497_497371

/-- Prove that the numbers 49, 4489, 444889, ... obtained by inserting 48 into the 
middle of the previous number are perfect squares. -/
theorem numbers_are_perfect_squares :
  ∀ n : ℕ, ∃ k : ℕ, (k ^ 2) = (Int.ofNat ((20 * (10 : ℕ) ^ n + 1) / 3)) :=
by
  sorry

end numbers_are_perfect_squares_l497_497371


namespace range_of_set_l497_497069

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497069


namespace rounding_effect_l497_497707

/-- Given positive integers x, y, and z, and rounding scenarios, the
  approximation of x/y - z is necessarily less than its exact value
  when z is rounded up and x and y are rounded down. -/
theorem rounding_effect (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
(RoundXDown RoundYDown RoundZUp : ℕ → ℕ) 
(HRoundXDown : ∀ a, RoundXDown a ≤ a)
(HRoundYDown : ∀ a, RoundYDown a ≤ a)
(HRoundZUp : ∀ a, a ≤ RoundZUp a) :
  (RoundXDown x) / (RoundYDown y) - (RoundZUp z) < x / y - z :=
sorry

end rounding_effect_l497_497707


namespace share_of_a_120_l497_497468

theorem share_of_a_120 (A B C : ℝ) 
  (h1 : A = (2 / 3) * (B + C)) 
  (h2 : B = (6 / 9) * (A + C)) 
  (h3 : A + B + C = 300) : 
  A = 120 := 
by 
  sorry

end share_of_a_120_l497_497468


namespace modified_game_probability_l497_497221

theorem modified_game_probability :
  let players := ["Alice", "Bob", "Cindy", "Dave"]
  let initial_money_distribution := [2, 2, 2, 2]
  let rings := 5
  let possible_outcomes_per_ring := 81
  in (4 / possible_outcomes_per_ring) ^ rings = (4 / 81) ^ 5 :=
by
  sorry

end modified_game_probability_l497_497221


namespace shrink_ray_reduction_l497_497033

theorem shrink_ray_reduction (initial_cups : ℕ) (initial_ounces_per_cup : ℕ) (final_ounces : ℕ)
  (h_initial_cups : initial_cups = 5) (h_initial_ounces_per_cup : initial_ounces_per_cup = 8)
  (h_final_ounces : final_ounces = 20) :
  let total_initial_ounces := initial_cups * initial_ounces_per_cup in
  let total_final_ounces := final_ounces in
  let percentage_reduction := ((total_initial_ounces - total_final_ounces) / total_initial_ounces : ℚ) * 100 in
  percentage_reduction = 50 :=
by {
  sorry
}

end shrink_ray_reduction_l497_497033


namespace quadratic_equation_correct_l497_497517

theorem quadratic_equation_correct : 
  (3 * (x + 1)^2 = 2 * (x + 1) ⟶ ∃ a b c, a ≠ 0 ∧ 3 * x^2 + 4 * x + 1 = a * x^2 + b * x + c) 
  ∧ ¬(∃ a b c, a ≠ 0 ∧ 1/x^2 + 1/x - 2 = a * x^2 + b * x + c)
  ∧ ¬(∃ a b c, a ≠ 0 ∧ (-7 * x) = a * x^2 + b * x + c) 
  ∧ (a ≠ 0 ⟶ (a * x^2 + b * x + c = a * x^2 + b * x + c)) := 
by
  sorry

end quadratic_equation_correct_l497_497517


namespace ants_in_field_approx_173_million_l497_497502

/-
Given:
- The width of the rectangular field in feet w_ft = 500
- The length of the rectangular field in feet l_ft = 600
- There are 4 ants per square inch
- 12 inches = 1 foot 

We want to prove that the number of ants in the field is approximately 173,000,000
-/

theorem ants_in_field_approx_173_million :
  let w_ft := 500
  let l_ft := 600
  let ants_per_square_inch := 4
  let inches_per_foot := 12
  let w_in := w_ft * inches_per_foot
  let l_in := l_ft * inches_per_foot
  let area_sq_in := w_in * l_in
  let total_ants := ants_per_square_inch * area_sq_in
  172_800_000 ≤ total_ants ∧ total_ants ≤ 173_200_000 :=
by
  let w_ft := 500
  let l_ft := 600
  let ants_per_square_inch := 4
  let inches_per_foot := 12
  let w_in := w_ft * inches_per_foot
  let l_in := l_ft * inches_per_foot
  let area_sq_in := w_in * l_in
  let total_ants := ants_per_square_inch * area_sq_in
  show 172_800_000 ≤ total_ants ∧ total_ants ≤ 173_200_000
  sorry

end ants_in_field_approx_173_million_l497_497502


namespace mean_score_of_seniors_l497_497829

-- Conditions
variables (s n : ℕ) (m_s m_n : ℝ)
hypothesis (total_students : s + n = 120)
hypothesis (non_seniors_twice : n = 2 * s)
hypothesis (mean_seniors_higher : m_s = 1.6 * m_n)
hypothesis (overall_mean : (40 * m_s + 80 * m_n) = 12000)

-- The theorem to prove
theorem mean_score_of_seniors :
  m_s = 133.33 :=
by
  sorry

end mean_score_of_seniors_l497_497829


namespace problem_solution_l497_497280

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end problem_solution_l497_497280


namespace isosceles_triangle_length_l497_497779

structure Triangle :=
(A B C D M E : Point)
(AC_eq_BC : A.distanceTo C = B.distanceTo C)
(foot_of_altitude : D ∈ line (C, B) ∧ ∀ p : Point, p ∈ line (C, B) → ∀ q : Point, q ∈ line (B, A) → p ⊥ q)
(midpoint_CD : M.coords = (C.coords + D.coords) / 2)
(intersection_BM_AC : E.coords ∈ line (B, M) ∧ E.coords ∈ line (A, C))

theorem isosceles_triangle_length (T : Triangle) : 
  T.A.distanceTo T.C = 3 * T.E.distanceTo T.C :=
sorry

end isosceles_triangle_length_l497_497779


namespace lighthouse_lights_total_l497_497031

theorem lighthouse_lights_total:
  ∃ (n : ℕ), (n = 7) →
  ∃ (a_1 q : ℕ), (a_1 = 1) ∧ (q = 2) →
  ∑ (i : ℕ) in finset.range n, a_1 * q^i = 127 :=
by
  sorry

end lighthouse_lights_total_l497_497031


namespace sin_sum_l497_497703

variables {α β : ℝ}

-- Given the cosine formula for the sum of two angles
axiom cos_sum_formula (α β : ℝ) : cos (α - β) = cos α * cos β - sin α * sin β

-- Conditions on α and β
axiom cos_alpha (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) : cos α = -4 / 5
axiom tan_beta (β : ℝ) (hβ : π / 2 < β ∧ β < π) : tan β = -1 / 3

-- The proof statement
theorem sin_sum (α β : ℝ) (hα : π < α ∧ α < 3 * π / 2) (hβ : π / 2 < β ∧ β < π) :
  sin (α + β) = sqrt 10 / 10 :=
by
  sorry

end sin_sum_l497_497703


namespace krista_savings_exceeds_500_on_saturday_l497_497791

def geometric_sum (a r n : ℕ) : ℕ := a * (r^n - 1) / (r - 1)

def exceeds_amount (a r target : ℕ) : ℕ :=
  Nat.find (λ n, geometric_sum a r n > target)

-- Krista starts with 2 cents and triples each day; we want to know when it exceeds 500 cents
def day_exceeds_500 : ℕ := exceeds_amount 2 3 500

theorem krista_savings_exceeds_500_on_saturday :
  day_exceeds_500 = 6 := sorry

end krista_savings_exceeds_500_on_saturday_l497_497791


namespace coefficients_identity_l497_497386

def coefficients_of_quadratic (a b c : ℤ) (x : ℤ) : Prop :=
  a * x^2 + b * x + c = 0

theorem coefficients_identity : ∀ x : ℤ,
  coefficients_of_quadratic 3 (-4) 1 x :=
by
  sorry

end coefficients_identity_l497_497386


namespace quadratic_solution_sum_l497_497382

theorem quadratic_solution_sum :
  ∃ a b : ℕ, (∃ x : ℝ, x^2 + 16 * x = 96 ∧ x = real.sqrt a - b) ∧ a + b = 168 :=
sorry

end quadratic_solution_sum_l497_497382


namespace log5_243_between_3_and_4_l497_497546

open Real

theorem log5_243_between_3_and_4 : ∃ (a b : ℕ), (a < log 243 / log 5 ∧ log 243 / log 5 < b) ∧ a + b = 7 :=
by
  let a := 3
  let b := 4
  use a, b
  have h1 : a < log 243 / log 5 := by norm_num
  have h2 : log 243 / log 5 < b := by norm_num
  exact ⟨⟨h1, h2⟩, rfl⟩

end log5_243_between_3_and_4_l497_497546


namespace determine_a_l497_497690

noncomputable def z1 (a : ℝ) : ℂ := 1 + a * complex.I
noncomputable def z2 : ℂ := 3 + 2 * complex.I

theorem determine_a (a : ℝ) (h : (z1 a * z2).im = 0) : a = -2/3 := 
by
  sorry

end determine_a_l497_497690


namespace B_paisa_per_Rs_A_l497_497478

def C_share : ℝ := 32
def total_sum : ℝ := 164
def conversion_rate : ℝ := 40 / 100  -- 40 paisa in rupees

theorem B_paisa_per_Rs_A :
  ∀ (A B C : ℝ),
  C = C_share →
  A + B / 100 + C = total_sum →
  conversion_rate * A = C * 100 →
  B / A = 65 :=
by
  sorry

end B_paisa_per_Rs_A_l497_497478


namespace unit_digit_25_pow_2010_sub_3_pow_2012_l497_497445

theorem unit_digit_25_pow_2010_sub_3_pow_2012 :
  (25^2010 - 3^2012) % 10 = 4 :=
by 
  sorry

end unit_digit_25_pow_2010_sub_3_pow_2012_l497_497445


namespace range_of_numbers_is_six_l497_497145

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497145


namespace meal_total_cost_including_tax_and_tip_l497_497159

theorem meal_total_cost_including_tax_and_tip:
  let per_person_cost := 90
  let number_of_people := 5
  let sales_tax_percentage := 0.0825
  let tip_percentages := [0.15, 0.18, 0.20, 0.22, 0.25]
  let total_cost_before_tax := per_person_cost * number_of_people
  let sales_tax := sales_tax_percentage * total_cost_before_tax
  let total_cost_before_tip := total_cost_before_tax + Real.round (100 * sales_tax) / 100
  let average_tip_percentage := List.sum tip_percentages / tip_percentages.length
  let tip_amount := average_tip_percentage * total_cost_before_tip
  in total_cost_before_tip + Real.round (100 * tip_amount) / 100 = 974.26 := sorry

end meal_total_cost_including_tax_and_tip_l497_497159


namespace regular_polygon_is_octagon_l497_497041

theorem regular_polygon_is_octagon (n : ℕ) (interior_angle exterior_angle : ℝ) :
  interior_angle = 3 * exterior_angle ∧ interior_angle + exterior_angle = 180 → n = 8 :=
by
  intros h
  sorry

end regular_polygon_is_octagon_l497_497041


namespace solve_system_solve_inequality_compute_solution_l497_497469

-- Problem 1
theorem solve_system (x y : ℝ) (h1 : y = x + 1) (h2 : x^2 + 4 * y^2 = 4) : 
(x = 0 ∧ y = 1) ∨ (x = -8/5 ∧ y = -3/5) := 
sorry

-- Problem 2
theorem solve_inequality (x t : ℝ) (h : x^2 - 2 * t * x + 1 > 0) : 
  (t > -1 ∧ t < 1 → true) ∧ 
  (t = 1 → x ≠ 1) ∧ 
  (t = -1 → x ≠ -1) ∧ 
  (t < -1 ∨ t > 1 → x < t - real.sqrt (t^2 - 1) ∨ x > t + real.sqrt (t^2 - 1)) := 
sorry

-- Problem 3
theorem compute_solution (a b c : ℝ) (h1 : ∀ x, (1 < x ∧ x < 2) → a * x^2 + b * x + c > 0) 
(h2 : b = -3 * a) (h3 : c = 2 * a) : 
  ∀ x, (x < -1 ∨ x > -1/2) ↔ a * (2 * x^2 + 3 * x + 1) < 0 := 
sorry

end solve_system_solve_inequality_compute_solution_l497_497469


namespace rounded_to_three_decimal_places_l497_497896

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497896


namespace book_arrangement_count_l497_497773

theorem book_arrangement_count :
  let count_math_books := 4
  let count_history_books := 6
  let count_valid_arrangements := 362880
  ∃ (math_ends: list (ℕ × ℕ)), 
  (∀ (m1 m2: ℕ), m1 ≠ m2) →
  (list.length math_ends = 12) →
  ( ∀ (l: list ℕ),  
    list.permutations [1,2,3,4,5,6,7,8] = list.length l → 
    ∀ (H1 H2: ℕ), 
    (H1 ≠ H2) → 
    l ≠ (list.cons H1 (list.cons H2 l)) → 
    list.length (list.permutations l).filter (λ x, x.count_p (λ v, 29) = 362880>):=
begin
  sorry  -- Proof to be provided
end

end book_arrangement_count_l497_497773


namespace factor_poly_eq_factored_form_l497_497602

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497602


namespace problem_statement_l497_497239

theorem problem_statement
  (a b m n c : ℝ)
  (h1 : a = -b)
  (h2 : m * n = 1)
  (h3 : |c| = 3)
  : a + b + m * n - |c| = -2 := by
  sorry

end problem_statement_l497_497239


namespace no_leopards_in_circus_l497_497531

theorem no_leopards_in_circus (L T : ℕ) (N : ℕ) (h₁ : L = N / 5) (h₂ : T = 5 * (N - T)) : 
  ∀ A, A = L + N → A = T + (N - T) → ¬ ∃ x, x ≠ L ∧ x ≠ T ∧ x ≠ (N - L - T) :=
by
  sorry

end no_leopards_in_circus_l497_497531


namespace largest_n_factorial_as_consecutive_product_l497_497383

theorem largest_n_factorial_as_consecutive_product :
  ∃ (n : ℕ), (∃ (k : ℕ), k = n - 4 ∧ n! = ∏ i in finset.range (k + 1) + 7, (i + 1)) ∧ n = 119 :=
by
  sorry

end largest_n_factorial_as_consecutive_product_l497_497383


namespace michael_students_l497_497362

theorem michael_students (M N : ℕ) (h1 : M = 5 * N) (h2 : M + N + 300 = 3500) : M = 2667 := 
by 
  -- This to be filled later
  sorry

end michael_students_l497_497362


namespace variance_scaled_data_l497_497263

theorem variance_scaled_data (x : ℕ → ℝ) (s : ℝ) (h : s^2 = 3) :
  let x' := λ n, 2 * x n in
  (variance x' = 12) :=
by
  sorry

end variance_scaled_data_l497_497263


namespace projection_is_correct_non_basis_vectors_are_collinear_l497_497253

-- Define vectors
def vec_a : ℝ × ℝ × ℝ := (9, 4, -4)
def vec_b : ℝ × ℝ × ℝ := (1, 2, 2)

-- Define projection function
def proj (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let b_norm_sq := b1 * b1 + b2 * b2 + b3 * b3
  let dot_product := a1 * b1 + a2 * b2 + a3 * b3
  let scalar := dot_product / b_norm_sq
  (scalar * b1, scalar * b2, scalar * b3)

-- Theorem 1: projection of vec_a onto vec_b is (1,2,2)
theorem projection_is_correct : proj vec_a vec_b = (1,2,2) := by
  unfold proj
  -- proof steps (omitted)
  sorry

-- Theorem 2: non-zero vectors that can't form a basis with any other vector are collinear
theorem non_basis_vectors_are_collinear (u v : ℝ × ℝ × ℝ) (hu : u ≠ (0,0,0)) (hv : v ≠ (0,0,0)) :
  (∀ w : ℝ × ℝ × ℝ, ¬(set.range ![u, v, w].linear_independent)) → (∃ k : ℝ, u = (k * v.1, k * v.2, k * v.3)) := by
  -- proof steps (omitted)
  sorry

end projection_is_correct_non_basis_vectors_are_collinear_l497_497253


namespace contrapositive_proposition_l497_497387

theorem contrapositive_proposition (α : ℝ) :
  (α = π / 4 → Real.tan α = 1) ↔ (Real.tan α ≠ 1 → α ≠ π / 4) :=
by
  sorry

end contrapositive_proposition_l497_497387


namespace chord_length_from_diameter_l497_497403

open Real

noncomputable def calculateChordLength (R : ℝ) : ℝ :=
  (4 * R * sqrt 5) / 5

theorem chord_length_from_diameter (R : ℝ) (H : 0 < R) :
  let O : ℝ × ℝ := (0, 0),
      A : ℝ × ℝ := (-R, 0),
      B : ℝ × ℝ := (R, 0),
      K : ℝ × ℝ := (0, R),
      M : ℝ × ℝ := (0, R / 2),
      AC : ℝ := dist A M in
  AC = calculateChordLength R :=
sorry

end chord_length_from_diameter_l497_497403


namespace george_boxes_of_eggs_l497_497671

theorem george_boxes_of_eggs (boxes_eggs : Nat) (h1 : ∀ (eggs_per_box : Nat), eggs_per_box = 3 → boxes_eggs * eggs_per_box = 15) :
  boxes_eggs = 5 :=
by
  sorry

end george_boxes_of_eggs_l497_497671


namespace factorization_identity_l497_497610

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497610


namespace bucket_A_contains_13_l497_497005

variable (A B : ℚ)

-- The problem conditions in lean 4
def condition1 (A B : ℚ) : Prop := A - 6 = (1 / 3) * (B + 6)
def condition2 (A B : ℚ) : Prop := B - 6 = (1 / 2) * (A + 6)

theorem bucket_A_contains_13.2 : (∃ A B : ℚ, condition1 A B ∧ condition2 A B ∧ A = 13.2) :=
by
  sorry

end bucket_A_contains_13_l497_497005


namespace factor_polynomial_l497_497625

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497625


namespace xiaoxian_mistake_xiaoxuan_difference_l497_497706

-- Define the initial expressions and conditions
def original_expr := (-9) * 3 - 5
def xiaoxian_expr (x : Int) := (-9) * 3 - x
def xiaoxuan_expr := (-9) / 3 - 5

-- Given conditions
variable (result_xiaoxian : Int)
variable (result_original : Int)

-- Proof statement
theorem xiaoxian_mistake (hx : xiaoxian_expr 2 = -29) : 
  xiaoxian_expr 5 = result_xiaoxian := sorry

theorem xiaoxuan_difference : 
  abs (xiaoxuan_expr - original_expr) = 24 := sorry

end xiaoxian_mistake_xiaoxuan_difference_l497_497706


namespace angle_of_inclination_l497_497685

theorem angle_of_inclination {P : ℝ × ℝ} (hP : P = (1, -3 / 2)) (f : ℝ → ℝ)
  (hf : ∀ x, f x = 1 / 2 * x^2 - 2) (hderiv : ∀ x, deriv f x = x) :
  ∃ θ : ℝ, θ = Real.arctan 1 ∧ θ = Real.pi / 4 :=
by
  -- Definitions from the conditions
  let k := 1
  have h1 : k = deriv f 1;
  -- Slope of the tangent line at point P should equal to the derivative evaluated at x = 1.
  rw [hderiv 1]
  exact θ = Real.pi / 4
  -- Angle of inclination from the slope k.
  exact ⟨Real.arctan k, by exact (Real.arctan_eq_pi_div_four) (ne_of_gt (Real.pi_div_four_gt_zero)), by exact Real.arctan_eq_pi_div_four _⟩
sorry

end angle_of_inclination_l497_497685


namespace zach_cookies_total_l497_497452

theorem zach_cookies_total :
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  cookies_monday + cookies_tuesday + cookies_wednesday = 92 :=
by
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  sorry

end zach_cookies_total_l497_497452


namespace magician_earned_4_dollars_l497_497034

-- Define conditions
def price_per_deck := 2
def initial_decks := 5
def decks_left := 3

-- Define the number of decks sold
def decks_sold := initial_decks - decks_left

-- Define the total money earned
def money_earned := decks_sold * price_per_deck

-- Theorem to prove the money earned is 4 dollars
theorem magician_earned_4_dollars : money_earned = 4 := by
  sorry

end magician_earned_4_dollars_l497_497034


namespace division_rounded_l497_497846

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497846


namespace range_of_set_is_six_l497_497087

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497087


namespace find_unique_n_l497_497202

open Complex

noncomputable def is_equilateral (a b c : ℂ) : Prop :=
  (a - b = (b - c) * Complex.exp (2 * π * Complex.I / 3) ∨ 
   a - b = (b - c) * Complex.exp (-2 * π * Complex.I / 3))

theorem find_unique_n (n : ℕ) (hn : n ≥ 4) :
  (∀ a b c : ℂ, a ≠ b → b ≠ c → c ≠ a → a ≠ 0 → b ≠ 0 → c ≠ 0 → 
      (a - b)^n + (b - c)^n + (c - a)^n = 0 → is_equilateral a b c) ↔ n = 4 :=
begin
  sorry
end

end find_unique_n_l497_497202


namespace deal_or_no_deal_minimum_boxes_to_eliminate_l497_497765

noncomputable def boxes : list ℕ :=
  [1, 10, 20, 50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 5000, 7500, 10000,
   15000, 20000, 25000, 30000, 40000, 50000, 75000, 100000, 200000, 300000, 400000, 
   500000, 750000, 1000000]

theorem deal_or_no_deal_minimum_boxes_to_eliminate :
  ∃ (n : ℕ), n = 12 ∧ (∃ (remaining_boxes : list ℕ), 
  remaining_boxes.length = 18 ∧ (∃ (high_value_boxes : list ℕ), 
  high_value_boxes.length = 9 ∧
  (∀ x ∈ high_value_boxes, x >= 50000) ∧ 
  (∀ y ∈ (remaining_boxes \ high_value_boxes), y < 50000))) :=
by
  sorry

end deal_or_no_deal_minimum_boxes_to_eliminate_l497_497765


namespace sudomilé_count_lichomilé_count_l497_497157

def is_sudomilé (n : ℕ) : Prop :=
  let digits := List.ofDigits 10 n in
  List.length digits = 3 ∧
  List.count (λ d, d = 1) digits = 1 ∧
  List.count (λ d, d % 2 = 0) digits = 2

def is_lichomilé (n : ℕ) : Prop :=
  let digits := List.ofDigits 10 n in
  List.length digits = 3 ∧
  List.count (λ d, d = 2) digits = 1 ∧
  List.count (λ d, d % 2 = 1) digits = 2

theorem sudomilé_count : (List.range 900).count (λ n, is_sudomilé (n + 100)) = 65 := by
  sorry

theorem lichomilé_count : (List.range 900).count (λ n, is_lichomilé (n + 100)) = 75 := by
  sorry

end sudomilé_count_lichomilé_count_l497_497157


namespace factor_polynomial_l497_497629

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497629


namespace chessboard_not_divisible_by_10_l497_497817

theorem chessboard_not_divisible_by_10 :
  ∀ (B : ℕ × ℕ → ℕ), 
  (∀ x y, B (x, y) < 10) ∧ 
  (∀ x y, x ≥ 0 ∧ x < 8 ∧ y ≥ 0 ∧ y < 8) →
  ¬ ( ∃ k : ℕ, ∀ x y, (B (x, y) + k) % 10 = 0 ) :=
by
  intros
  sorry

end chessboard_not_divisible_by_10_l497_497817


namespace problem_quadrilateral_l497_497759

/-- Given a convex quadrilateral ABCD with specific angles and equal sides, 
extend CB to meet the circumcircle of triangle DAC at E, then prove that CE = BD. 
-/
theorem problem_quadrilateral (A B C D E : Type)
  [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint D] [IsPoint E]
  (convex_ABCD : IsConvexQuadrilateral ABCD)
  (angle_ABD : ∡ ABD = 30°)
  (angle_BCA : ∡ BCA = 75°)
  (angle_ACD : ∡ ACD = 25°)
  (side_CD_CB : CD = CB)
  (circumcircle_DAC : OnCircumcircle (triangle DAC) E) :
  CE = BD :=
  sorry

end problem_quadrilateral_l497_497759


namespace find_m_l497_497729

theorem find_m (a b m : ℤ) (h1 : a - b = 6) (h2 : a + b = 0) : 2 * a + b = m → m = 3 :=
by
  sorry

end find_m_l497_497729


namespace round_fraction_to_three_decimal_l497_497907

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497907


namespace b_and_c_together_work_rate_l497_497455

-- Definitions based on conditions
def total_work : ℝ := 1
def rate_a_plus_b : ℝ := 1 / 16
def rate_a : ℝ := 1 / 32
def rate_c : ℝ := 1 / 24

-- Result statement
theorem b_and_c_together_work_rate :
  (rate_a_plus_b - rate_a) + rate_c = 1 / (96 / 7) :=
by 
  -- Work rate of b
  let rate_b := rate_a_plus_b - rate_a
  -- Combined work rate of b and c
  have rate_bc : ℝ := rate_b + rate_c
  -- Simplification to show that the combined work rate matches the expected value
  calc
    rate_bc = (1 / 16 - 1 / 32) + (1 / 24) : by sorry
      ...   = 7 / 96                  : by sorry
      ...   = 1 / (96 / 7)            : by sorry

end b_and_c_together_work_rate_l497_497455


namespace range_of_set_l497_497077

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497077


namespace area_of_rectangle_l497_497013

noncomputable def area_proof : ℝ :=
  let a := 294
  let b := 147
  let c := 3
  a + b * Real.sqrt c

theorem area_of_rectangle (ABCD : ℝ × ℝ) (E : ℝ) (F : ℝ) (BE : ℝ) (AB' : ℝ) : 
  BE = 21 ∧ BE = 2 * CF → AB' = 7 → 
  (ABCD.1 * ABCD.2 = 294 + 147 * Real.sqrt 3 ∧ (294 + 147 + 3 = 444)) :=
sorry

end area_of_rectangle_l497_497013


namespace factor_polynomial_l497_497592

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497592


namespace rectangle_width_l497_497401

theorem rectangle_width (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * (w + l) = w * l) : w = 3 :=
by sorry

end rectangle_width_l497_497401


namespace barney_no_clean_towels_days_l497_497529

theorem barney_no_clean_towels_days
  (wash_cycle_weeks : ℕ := 1)
  (total_towels : ℕ := 18)
  (towels_per_day : ℕ := 2)
  (days_per_week : ℕ := 7)
  (missed_laundry_weeks : ℕ := 1) :
  (days_per_week - (total_towels - (days_per_week * towels_per_day * missed_laundry_weeks)) / towels_per_day) = 5 :=
by
  sorry

end barney_no_clean_towels_days_l497_497529


namespace max_truthful_gnomes_l497_497997

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

end max_truthful_gnomes_l497_497997


namespace exists_linear_forms_sum_pow_2017_l497_497032

noncomputable def linear_form (a : Fin 2017 → ℝ) (x : Fin 2017 → ℝ) : ℝ :=
  ∑ i in Finset.univ, a i * x i

theorem exists_linear_forms_sum_pow_2017 :
  ∃ n : ℕ, ∃ (P : Fin n → (Fin 2017 → ℝ) → ℝ),
    (∀ x : Fin 2017 → ℝ, 
      (∏ i in Finset.univ, x i) = 
      ∑ i in Finset.univ, (P i x)^2017) :=
sorry

end exists_linear_forms_sum_pow_2017_l497_497032


namespace round_fraction_to_three_decimal_l497_497910

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497910


namespace number_of_lines_l497_497766

-- Define a plane with two points P and Q that are 8 units apart
variables {P Q : Type} [metric_space P] [inhabited P] [inhabited Q]
variables [metric_space Q] [inhabited P] {dist : P → Q → ℝ}

-- Conditions
axiom dist_PQ : dist P Q = 8
axiom radius_P : ℝ := 4
axiom radius_Q : ℝ := 6

-- a theorem to prove the number of such lines
theorem number_of_lines (P Q : P) (dist_PQ : dist P Q = 8) (radius_P = 4) (radius_Q = 6) : 
    ∃ (l : list (line ℝ P Q)), l.length = 2 := 
by sorry

end number_of_lines_l497_497766


namespace krystian_total_books_borrowed_l497_497325

/-
Conditions:
1. Krystian starts on Monday by borrowing 40 books.
2. Each day from Tuesday to Thursday, he borrows 5% more books than he did the previous day.
3. On Friday, his number of borrowed books is 40% higher than on Thursday.
4. During weekends, Krystian borrows books for his friends, and he borrows 2 additional books for every 10 books borrowed during the weekdays.

Theorem: Given these conditions, Krystian borrows a total of 283 books from Monday to Sunday.
-/
theorem krystian_total_books_borrowed : 
  let mon := 40
  let tue := mon + (5 * mon / 100)
  let wed := tue + (5 * tue / 100)
  let thu := wed + (5 * wed / 100)
  let fri := thu + (40 * thu / 100)
  let weekday_total := mon + tue + wed + thu + fri
  let weekend := 2 * (weekday_total / 10)
  weekday_total + weekend = 283 := 
by
  sorry

end krystian_total_books_borrowed_l497_497325


namespace factor_polynomial_l497_497646

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497646


namespace complement_of_A_with_respect_to_U_l497_497731

-- Definitions
def U : Set ℤ := {-2, -1, 1, 3, 5}
def A : Set ℤ := {-1, 3}

-- Statement of the problem
theorem complement_of_A_with_respect_to_U :
  (U \ A) = {-2, 1, 5} := 
by
  sorry

end complement_of_A_with_respect_to_U_l497_497731


namespace division_rounded_l497_497845

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497845


namespace max_truthful_dwarfs_le_one_l497_497984

def dwarf_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

theorem max_truthful_dwarfs_le_one (heights : List ℕ) (h_lengths : heights.length = 7) (h_sorted : heights.sorted (· > ·)) :
  ∃ n, (n ≤ 1) ∧ ∀ i, i ∈ (Finset.range 7).to_list → (heights[i] ≠ dwarf_height_claims[i] → (1 = n)) :=
  sorry

end max_truthful_dwarfs_le_one_l497_497984


namespace arithmetic_geometric_harmonic_inequality_l497_497814

theorem arithmetic_geometric_harmonic_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1.01 * real.sqrt (x * y) > (x + y) / 2 ∧ (x + y) / 2 > 100 * (2 * x * y / (x + y)) :=
by
  sorry

end arithmetic_geometric_harmonic_inequality_l497_497814


namespace meet_at_boundary_A_and_D_l497_497368

noncomputable def alice_bob_meet_at_boundary : Prop :=
  let track_length := 60 in
  let total_distance_alice := 7200 in
  let laps_alice := total_distance_alice / track_length in
  let alice_position := (laps_alice * track_length) % track_length in
  let bob_start := track_length / 2 in
  let bob_position := (laps_alice * -track_length) % track_length + bob_start in
  (alice_position = 0 ∨ alice_position = track_length) ∧ bob_position = 0

theorem meet_at_boundary_A_and_D : alice_bob_meet_at_boundary :=
sorry

end meet_at_boundary_A_and_D_l497_497368


namespace range_of_set_l497_497068

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497068


namespace miles_per_day_for_last_two_islands_l497_497402

theorem miles_per_day_for_last_two_islands:
  ∀ (x : ℕ), (2 * 20 * 1.5 + 2 * x * 1.5 = 135) → x = 25 := by
  sorry

end miles_per_day_for_last_two_islands_l497_497402


namespace stack_height_of_five_pipes_l497_497661

open Real

theorem stack_height_of_five_pipes :
  ∀ (r : ℝ), (r = 4) → (h : ℝ), h = 12 + 4 * sqrt 3 := by
  intro r hr h
  sorry

end stack_height_of_five_pipes_l497_497661


namespace original_price_of_cycle_l497_497491

theorem original_price_of_cycle (P : ℝ) (h1 : P * 0.85 = 1190) : P = 1400 :=
by
  sorry

end original_price_of_cycle_l497_497491


namespace gnomes_telling_the_truth_l497_497942

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l497_497942


namespace polynomial_coefficient_sum_l497_497274

theorem polynomial_coefficient_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (2 * x - 3) ^ 5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  sorry

end polynomial_coefficient_sum_l497_497274


namespace solution_set_f_l497_497710

def f (x : ℝ) : ℝ := 2022 ^ (x - 3) + (x - 3) ^ 3 - 2022 ^ (3 - x) + 2 * x

theorem solution_set_f :
  { x : ℝ | f (x^2 - 4) + f (2 - 2*x) ≤ 12 } = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
by
  sorry

end solution_set_f_l497_497710


namespace problem1_problem2_problem3_l497_497300

-- Given conditions for the sequence
axiom pos_seq {a : ℕ → ℝ} : (∀ n : ℕ, 0 < a n)
axiom relation1 {a : ℕ → ℝ} (t : ℝ) : (∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
axiom relation2 {a : ℕ → ℝ} : 2 * (a 3) = (a 2) + (a 4)

-- Proof Requirements

-- (1) Find the value of (a1 + a3) / a2
theorem problem1 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) :
  (a 1 + a 3) / a 2 = 2 :=
sorry

-- (2) Prove that the sequence is an arithmetic sequence
theorem problem2 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) :
  ∀ n : ℕ, a (n+2) - a (n+1) = a (n+1) - a n :=
sorry

-- (3) Show p and r such that (1/a_k), (1/a_p), (1/a_r) form an arithmetic sequence
theorem problem3 {a : ℕ → ℝ} (t : ℝ) (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, (a (n+1))^2 = (a n) * (a (n+2)) + t)
  (h3 : 2 * (a 3) = (a 2) + (a 4)) (k : ℕ) (hk : k ≠ 0) :
  (k = 1 → ∀ p r : ℕ, ¬((k < p ∧ p < r) ∧ (1 / a k) + (1 / a r) = 2 * (1 / a p))) ∧ 
  (k ≥ 2 → ∃ p r : ℕ, (k < p ∧ p < r) ∧ (1 / a k) + (1 / a r) = 2 * (1 / a p) ∧ p = 2 * k - 1 ∧ r = k * (2 * k - 1)) :=
sorry

end problem1_problem2_problem3_l497_497300


namespace trevor_eggs_l497_497830

theorem trevor_eggs :
  let gertrude := 4
  let blanche := 3
  let nancy := 2
  let martha := 2
  let ophelia := 5
  let penelope := 1
  let quinny := 3
  let dropped := 2
  let gifted := 3
  let total_collected := gertrude + blanche + nancy + martha + ophelia + penelope + quinny
  let remaining_after_drop := total_collected - dropped
  let final_eggs := remaining_after_drop - gifted
  final_eggs = 15 := by
    sorry

end trevor_eggs_l497_497830


namespace range_of_t_circle_largest_area_eq_point_P_inside_circle_l497_497705

open Real

-- Defining the given equation representing the trajectory of a point on a circle
def circle_eq (x y t : ℝ) : Prop :=
  x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16 * t^4 + 9 = 0

-- Problem 1: Proving the range of t
theorem range_of_t : 
  ∀ t : ℝ, (∃ x y : ℝ, circle_eq x y t) → -1/7 < t ∧ t < 1 :=
sorry

-- Problem 2: Proving the equation of the circle with the largest area
theorem circle_largest_area_eq : 
  ∃ t : ℝ, t = 3/7 ∧ (∀ x y : ℝ, circle_eq x y (3/7)) → 
  ∀ x y : ℝ, (x - 24/7)^2 + (y + 13/49)^2 = 16/7 :=
sorry

-- Problem 3: Proving the range of t for point P to be inside the circle
theorem point_P_inside_circle : 
  ∀ t : ℝ, (∃ x y : ℝ, circle_eq x y t) → 
  (0 < t ∧ t < 3/4) :=
sorry

end range_of_t_circle_largest_area_eq_point_P_inside_circle_l497_497705


namespace more_freshmen_than_sophomores_l497_497770

-- Define the total number of students
def total_students : ℕ := 800

-- Define the percentage of juniors
def percentage_juniors : ℝ := 0.26

-- Define the percentage of sophomores
def percentage_sophomores : ℝ := 0.25

-- Define the number of seniors
def number_seniors : ℕ := 160

-- The lean statement proving the number of more freshmen than sophomores
theorem more_freshmen_than_sophomores : 232 - 200 = 32 :=
by
  -- Calculation steps (included as part of the conditions here for proof)
  have juniors := (percentage_juniors * total_students)
  have sophomores := (percentage_sophomores * total_students)
  have freshmen := total_students - juniors - sophomores - number_seniors
  have result := freshmen - sophomores
  show result = 32
  sorry -- Proof goes here

end more_freshmen_than_sophomores_l497_497770


namespace range_of_numbers_is_six_l497_497144

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497144


namespace fraction_rounded_equals_expected_l497_497921

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497921


namespace smallest_x_l497_497742

theorem smallest_x (y : ℕ) (x : ℕ) (h : 0.75 = y / (240 + x)) (hx : x > 0) (hy : y > 0) : x = 4 :=
sorry

end smallest_x_l497_497742


namespace range_of_set_l497_497125

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497125


namespace factorization_identity_l497_497607

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497607


namespace exists_non_monotonic_function_l497_497194

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ [0,1] ∧ is_rat(x) then x else 1 - x

theorem exists_non_monotonic_function :
  (∀ x, x ∈ [0,1] → ∀ y, y ∈ [0,1] → (f x = f y) ↔ x = y) ∧
  (∀ a b, a < b → ∃ x y, x ∈ [a, b] ∧ y ∈ [a, b] ∧ x < y ∧ f x > f y ∧ f x < f y) := sorry

end exists_non_monotonic_function_l497_497194


namespace ellipse_equation_standard_midpoint_slope_fixed_point_l497_497684

-- Definitions for given conditions
def ellipse_focus_left : ℝ × ℝ := (-1, 0)
def ellipse_point_E : ℝ × ℝ := (1, (2 * Real.sqrt 3) / 3)
def point_P : ℝ × ℝ := (1, 1)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Theorem statements
theorem ellipse_equation_standard :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, (x, y) ∈ ((λ x y => x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) :=
-- conditions
(h1 : ellipse_focus_left = (-1, 0))
(h2 : ∃ xf yf, xf = 1 ∧ yf = 0 ∧ (1, (2 * Real.sqrt 3) / 3) ∈ (λ x y => x^2/xf^2 + y^2/(yf^2 - 1)) = 1)
-- to prove
(z : a = √3 ∧ b = √2)

theorem midpoint_slope :
  ∀ (A B : ℝ × ℝ), midpoint A B = point_P → (∃ x y : ℝ, 
    (x^2 / 3 + y^2 / 2 = 1) → k1 = -(2 / 3))
-- conditions
(h3 : ∃ k1 : ℝ, (∀ x1 y1 x2 y2 : ℝ, (point_P = midpoint ( (x1, y1), (x2, y2) ) ) → (k1 = (y1 - y2)/(x1 - x2)))
-- to prove
(k1 = - 2 / 3)

theorem fixed_point :
  ∀ (k1 k2 : ℝ), k1 + k2 = 1 → ∃ p : ℝ × ℝ, 
    (let M := midpoint (AB, CD), N := midpoint (CD, CD), MN_slope := ( N.2 - M.2 ) / ( N.1 - M.1) 
      (MN_line : ℝ × ℝ := (a, b)) → (MN_line = fixed_point)) 
-- conditions
(h4 : ∀ k1 k2 : ℝ, k1 + k2 = 1)
-- to prove
(fixed_point := (0, - 2 / 3 ))

end ellipse_equation_standard_midpoint_slope_fixed_point_l497_497684


namespace factor_polynomial_l497_497626

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497626


namespace range_of_numbers_is_six_l497_497140

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497140


namespace range_of_set_l497_497093

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497093


namespace round_fraction_to_three_decimal_l497_497911

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497911


namespace factorize_polynomial_l497_497201

variable (x : ℝ)

theorem factorize_polynomial : 4 * x^3 - 8 * x^2 + 4 * x = 4 * x * (x - 1)^2 := 
by 
  sorry

end factorize_polynomial_l497_497201


namespace theater_seating_capacity_l497_497166

-- Definitions for the conditions
variables (a : ℝ) -- number of seats in the theater
variables (c4 : ℝ) (c6 : ℝ) (c : ℝ) (t : ℝ) 

-- Condition Definitions
def condition1 := c4 = 0.65 * a
def condition2 := c6 = 0.50 * a
def condition3 := t = 2 * a - 40
def condition4 := c = (0.65 * a) + (0.50 * a) - 28
def condition5 := 0.57 * t = c

-- The proof statement
theorem theater_seating_capacity (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) : 
  a = 520 :=
by 
  sorry

end theater_seating_capacity_l497_497166


namespace range_of_numbers_is_six_l497_497105

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497105


namespace factor_poly_eq_factored_form_l497_497601

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497601


namespace meals_second_restaurant_l497_497269

theorem meals_second_restaurant (r1 r2 r3 total_weekly_meals : ℕ) 
    (H1 : r1 = 20) 
    (H3 : r3 = 50) 
    (H_total : total_weekly_meals = 770) : 
    (7 * r2) = 280 := 
by 
    sorry

example (r2 : ℕ) : (40 = r2) :=
    by sorry

end meals_second_restaurant_l497_497269


namespace cube_painting_l497_497514

theorem cube_painting (n : ℕ) (h₁ : n > 4) 
  (h₂ : (2 * (n - 2)) = (n^2 - 2*n + 1)) : n = 5 :=
sorry

end cube_painting_l497_497514


namespace ladder_slide_l497_497029

theorem ladder_slide (L d Δh : ℝ) (hL : L = 25) (hd : d = 7) (hΔh : Δh = 4) :
  let initial_height := real.sqrt (L^2 - d^2)
  let new_height := initial_height - Δh
  let new_distance := real.sqrt (L^2 - new_height^2)
  new_distance - d = 8 :=
by
  sorry

end ladder_slide_l497_497029


namespace no_valid_n_for_three_digit_conditions_l497_497665

theorem no_valid_n_for_three_digit_conditions :
  ∃ (n : ℕ) (h₁ : 100 ≤ n / 4 ∧ n / 4 ≤ 999) (h₂ : 100 ≤ 4 * n ∧ 4 * n ≤ 999), false :=
by sorry

end no_valid_n_for_three_digit_conditions_l497_497665


namespace monotonicity_and_tangent_pass_through_point_l497_497716

noncomputable def f (x : ℝ) (m : ℝ) := -x^3 + m*x^2 - m

def is_monotone_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y)

theorem monotonicity_and_tangent_pass_through_point
  (m : ℝ)
  (h0m : 0 < m)
  (h1m : m < 20) 
  (a : ℝ)
  (ha : 1 ≤ a) : 
  ((9 ≤ m) ∧ (m < 20) →
    is_monotone_in_interval (λ x, f x m) 2 6) ∧
  (3 < m ∧ m < 9 →
    ∃ c, (2 < c ∧ c < 6 ∧ is_monotone_in_interval (λ x, f x m) 2 c ∧ is_monotone_in_interval (λ x, f x m) c 6)) ∧
  ((0 < m) ∧ (m ≤ 3) →
    ¬is_monotone_in_interval (λ x, f x m) 2 6) ∧
  (∃ x1 x2, x1 ≠ x2 ∧ ∀ x, f x1 m - (f x m - f x2 m) * (x1 - x2) = log (1/a) →
    (0 < m ∧ m ≤ 8/3) ∨ (9 + 3 * Real.sqrt 6 ≤ m ∧ m < 20)) :=
by sorry

end monotonicity_and_tangent_pass_through_point_l497_497716


namespace air_quality_I_prob_avg_exercise_l497_497155

-- Define conditions for problem
def air_quality_days : ℕ -> ℕ := λ n,
  match n with
  | 1 => 3 + 18 + 28
  | 2 => 5 + 9 + 11
  | 3 => 5 + 6 + 6
  | 4 => 7 + 2 + 0
  | _ => 0
  end

-- Define the probability of air quality level Ⅰ being 0.49
theorem air_quality_I_prob :
  air_quality_days 1 / 100 = 0.49 := sorry

-- Define mid-values
def mid_values : ℕ -> ℕ := λ n,
  match n with
  | 1 => 100
  | 2 => 300
  | 3 => 500
  | _ => 0
  end

-- Define total days of each exercise group
def exercise_days : ℕ -> ℕ := λ n,
  match n with
  | 1 => 3 + 5 + 5 + 7
  | 2 => 18 + 9 + 6 + 2
  | 3 => 28 + 11 + 6 + 0
  | _ => 0
  end

-- Define the estimated average number of people exercising to be 355
theorem avg_exercise :
  let weighted_avg := (mid_values 1 * exercise_days 1 + mid_values 2 * exercise_days 2 + mid_values 3 * exercise_days 3) / 100
  in weighted_avg = 355 := sorry

end air_quality_I_prob_avg_exercise_l497_497155


namespace number_of_children_on_bus_l497_497016

theorem number_of_children_on_bus (initial_children : ℕ) (additional_children : ℕ) (total_children : ℕ) 
  (h1 : initial_children = 26) (h2 : additional_children = 38) : total_children = 64 :=
by
  sorry

end number_of_children_on_bus_l497_497016


namespace perfect_square_and_integer_c_exists_l497_497342

-- Define the problem statement and conditions
theorem perfect_square_and_integer_c_exists (q a b : ℕ) (h : a ^ 2 - q * a * b + b ^ 2 = q) : 
    (∃ c : ℤ, c ≠ a ∧ c ^ 2 - q * b * c + b ^ 2 = q) ∧ ∃ k : ℕ, q = k ^ 2 := 
by
  sorry

end perfect_square_and_integer_c_exists_l497_497342


namespace find_discount_percentage_l497_497511

-- Define conditions:
def price_paid : ℝ := 184
def savings : ℝ := 16

-- Define the original price based on conditions:
def original_price : ℝ := price_paid + savings

-- Define the discount percentage:
def discount_percentage : ℝ := (savings / original_price) * 100

-- State the main theorem:
theorem find_discount_percentage : discount_percentage = 8 :=
by
  -- Proof skipped
  sorry

end find_discount_percentage_l497_497511


namespace Ram_money_l497_497404

theorem Ram_money (R G K : ℕ) (h1 : R = 7 * G / 17) (h2 : G = 7 * K / 17) (h3 : K = 4046) : R = 686 := by
  sorry

end Ram_money_l497_497404


namespace count_special_integers_l497_497272

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

theorem count_special_integers : 
  {n : ℕ // n < 700 ∧ n = 7 * sum_of_digits n}.to_finset.card = 1 :=
by 
  sorry

end count_special_integers_l497_497272


namespace angle_obtuse_not_parallel_l497_497267

theorem angle_obtuse_not_parallel (x : ℝ) :
  let a := (x, 2)
  let b := (2, -5)
  (x * 2 + 2 * (-5)) < 0 ∧ x ≠ -4/5 → x < 5 ∧ x ≠ -4/5 := by
sry

end angle_obtuse_not_parallel_l497_497267


namespace concyclic_ABCD_l497_497804

open EuclideanGeometry

variables (A B C D M : Point)
variables (convex_ABCD : convex_quadrilateral A B C D)
variables (midpoint_M : is_midpoint M A C)
variables (angle_BCD : angle_at B C D = α)
variables (angle_BMA : angle_at B M A = α)
variables (angle_AMD : angle_at A M D = α)

theorem concyclic_ABCD : is_concyclic A B C D :=
by
  sorry

end concyclic_ABCD_l497_497804


namespace gnomes_telling_the_truth_l497_497946

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l497_497946


namespace point_P_in_triangle_A_l497_497014

-- Define the basic geometric entities and conditions
variables (A B C P L M N B' C' A' : Type)
variables [is_triangle ABC : triangle A B C]
variables [is_inside P ABC : P_inside_triangle P A B C]
variables [is_perpendicular PL PBC : perpendicular P L B C]
variables [is_perpendicular PM PCA : perpendicular P M C A]
variables [is_perpendicular PN PAB : perpendicular P N A B]
variables [feet_of_perpendicular LBC : feet L B C P PL]
variables [feet_of_perpendicular MCA : feet M C A P PM]
variables [feet_of_perpendicular NAB: feet N A B P PN]

-- Conditions for forming a triangle by segments PL, PM, and PN
variables [triangle_inequality : PL + PM > PN]

-- Constructs of the angle bisectors
variables [angle_bisector BB' : bisector A C B B']
variables [angle_bisector CC' : bisector A B C C']
variables [angle_bisector AA' : bisector B C A A']

-- Conclusion
theorem point_P_in_triangle_A'B'C' (h : P := interior_triangle B' C' A') : 
  ∃ A B C P L M N B' C' A', 
    is_triangle A B C ∧ 
    is_inside P A B C ∧ 
    perpendicular P L B C ∧ 
    perpendicular P M C A ∧ 
    perpendicular P N A B ∧ 
    (L = feet_of_perpendicular P B C) ∧ 
    (M = feet_of_perpendicular P C A) ∧ 
    (N = feet_of_perpendicular P A B) ∧ 
    (PL + PM > PN) ∧ 
    bisector B B' A C ∧ 
    bisector C C' A B ∧ 
    bisector A A' B C ∧ 
    (P ∈ triangle_boundaries A' B' C').
Proof := sorry

end point_P_in_triangle_A_l497_497014


namespace min_value_of_function_l497_497250

theorem min_value_of_function (f : ℝ → ℝ) (a : ℝ) (h_f : ∀ x, f x = a * Real.exp x + Real.exp (-x))
  (h_perpendicular : ∀ k, k = f' 0 → k * (-1 / 3) = -1) : ∃ x_min, f x_min = 4 :=
by
  -- Definitions and conditions
  let f := λ x, 4 * Real.exp x + Real.exp (-x)
  have hf_deriv : ∀ x, deriv f x = 4 * Real.exp x - Real.exp (-x) := sorry
  use -Real.log 2
  sorry

end min_value_of_function_l497_497250


namespace find_second_dimension_l497_497503

variable (l h w : ℕ)
variable (cost_per_sqft total_cost : ℕ)
variable (surface_area : ℕ)

def insulation_problem_conditions (l : ℕ) (h : ℕ) (cost_per_sqft : ℕ) (total_cost : ℕ) (w : ℕ) (surface_area : ℕ) : Prop :=
  l = 4 ∧ h = 3 ∧ cost_per_sqft = 20 ∧ total_cost = 1880 ∧ surface_area = (2 * l * w + 2 * l * h + 2 * w * h)

theorem find_second_dimension (l h w : ℕ) (cost_per_sqft total_cost surface_area : ℕ) :
  insulation_problem_conditions l h cost_per_sqft total_cost w surface_area →
  surface_area = 94 →
  w = 5 :=
by
  intros
  simp [insulation_problem_conditions] at *
  sorry

end find_second_dimension_l497_497503


namespace center_is_endpoint_l497_497199

-- Define the size of the grid
def grid_size := 2017

-- Define a type for cells
structure Cell :=
  (x : ℕ)
  (y : ℕ)
  (valid : x < grid_size ∧ y < grid_size)

-- Predicate to determine if a cell shares a side with another cell of the same color
def shares_side (c1 c2 : Cell) : Prop :=
  (c1.x = c2.x ∧ (c1.y = c2.y + 1 ∨ c1.y + 1 = c2.y)) ∨
  (c1.y = c2.y ∧ (c1.x = c2.x + 1 ∨ c1.x + 1 = c2.x))

-- Define the center cell
def center : Cell :=
  { x := grid_size / 2,
    y := grid_size / 2,
    valid := ⟨Nat.div_lt_of_lt_mul (Nat.odd_iff_not_even.1 dec_trivial).symm, Nat.div_lt_of_lt_mul (Nat.odd_iff_not_even.1 dec_trivial).symm⟩ }

-- Define the graph G_i as set of cells that form a path
structure Graph :=
  (endpoints : set Cell) 
  (is_path : ∀ c1 c2 ∈ endpoints, (c1 ≠ c2 → ¬shares_side c1 c2))

-- Define the grids G1 and G2
def G1 : Graph := sorry
def G2 : Graph := sorry

-- Proposition to be proved
theorem center_is_endpoint : center ∈ G1.endpoints ∨ center ∈ G2.endpoints := sorry

end center_is_endpoint_l497_497199


namespace max_one_truthful_dwarf_l497_497995

def dwarf_height_claims : List (ℕ × ℕ) := [
  (1, 60),  -- First dwarf claims 60 cm
  (2, 61),  -- Second dwarf claims 61 cm
  (3, 62),  -- Third dwarf claims 62 cm
  (4, 63),  -- Fourth dwarf claims 63 cm
  (5, 64),  -- Fifth dwarf claims 64 cm
  (6, 65),  -- Sixth dwarf claims 65 cm
  (7, 66)   -- Seventh dwarf claims 66 cm
]

-- Proof problem statement: Prove that at most one dwarf is telling the truth about their height
theorem max_one_truthful_dwarf : ∀ (heights : List ℕ), heights.length = 7 → 
  (∀ i, i < heights.length → dwarf_height_claims.get? i ≠ none → (heights.get? i = dwarf_height_claims.get? i) → 
    ∑ i in finset.range heights.length, bool.to_num (heights.get? i = dwarf_height_claims.get? i) ≤ 1) :=
by
  intros heights len_heights valid_claims idx idx_in_range some_claim match_claim
  -- Add proof here 
  sorry

end max_one_truthful_dwarf_l497_497995


namespace max_truthful_dwarfs_le_one_l497_497982

def dwarf_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

theorem max_truthful_dwarfs_le_one (heights : List ℕ) (h_lengths : heights.length = 7) (h_sorted : heights.sorted (· > ·)) :
  ∃ n, (n ≤ 1) ∧ ∀ i, i ∈ (Finset.range 7).to_list → (heights[i] ≠ dwarf_height_claims[i] → (1 = n)) :=
  sorry

end max_truthful_dwarfs_le_one_l497_497982


namespace time_to_clear_l497_497434

-- Define the lengths of the trains
def length_train1 : ℝ := 235
def length_train2 : ℝ := 379

-- Define the speeds of the trains in km/h and convert to m/s
def speed_train1_kmph : ℝ := 75
def speed_train2_kmph : ℝ := 120

def speed_train1_mps : ℝ := speed_train1_kmph * (1000 / 3600)
def speed_train2_mps : ℝ := speed_train2_kmph * (1000 / 3600)

-- Define the total distance to be covered and the relative speed
def total_distance : ℝ := length_train1 + length_train2
def relative_speed : ℝ := speed_train1_mps + speed_train2_mps

-- Prove the time for the trains to be completely clear of each other
theorem time_to_clear : total_distance / relative_speed ≈ 11.34 := sorry

end time_to_clear_l497_497434


namespace dwarfs_truth_claims_l497_497970

/-- Seven dwarfs lined up by height, starting with the tallest.
Each dwarf claims a specific height as follows.
Verify that the largest number of dwarfs telling the truth is exactly 1. -/
theorem dwarfs_truth_claims :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  ∀ (actual_height: Fin 7 → ℕ), 
    (∀ i, actual_height i > actual_height (Fin.mk i (by linarith))) →
    ∀ claim_truth: Fin 7 → Prop, 
    (∀ i, claim_truth i ↔ actual_height i = heights[i]) →
    (∃ k, ∀ j, claim_truth j → j = k) :=
by
  intro heights actual_height h_order claim_truth h_claim_match
  existsi Fin.mk 0 (by linarith) -- Claim that only one can be telling the truth (arbitrary choice)
  sorry  -- Proof steps are omitted

end dwarfs_truth_claims_l497_497970


namespace factor_poly_eq_factored_form_l497_497596

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497596


namespace find_difference_of_segments_l497_497498

theorem find_difference_of_segments 
  (a b c d x y : ℝ)
  (h1 : a + b = 70)
  (h2 : b + c = 90)
  (h3 : c + d = 130)
  (h4 : a + d = 110)
  (hx_y_sum : x + y = 130)
  (hx_c : x = c)
  (hy_d : y = d) : 
  |x - y| = 13 :=
sorry

end find_difference_of_segments_l497_497498


namespace combined_average_mark_l497_497309

theorem combined_average_mark 
  (n_A n_B n_C n_D n_E : ℕ) 
  (avg_A avg_B avg_C avg_D avg_E : ℕ)
  (students_A : n_A = 22) (students_B : n_B = 28)
  (students_C : n_C = 15) (students_D : n_D = 35)
  (students_E : n_E = 25)
  (avg_marks_A : avg_A = 40) (avg_marks_B : avg_B = 60)
  (avg_marks_C : avg_C = 55) (avg_marks_D : avg_D = 75)
  (avg_marks_E : avg_E = 50) : 
  (22 * 40 + 28 * 60 + 15 * 55 + 35 * 75 + 25 * 50) / (22 + 28 + 15 + 35 + 25) = 58.08 := 
  by 
    sorry

end combined_average_mark_l497_497309


namespace dwarfs_truth_claims_l497_497964

/-- Seven dwarfs lined up by height, starting with the tallest.
Each dwarf claims a specific height as follows.
Verify that the largest number of dwarfs telling the truth is exactly 1. -/
theorem dwarfs_truth_claims :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  ∀ (actual_height: Fin 7 → ℕ), 
    (∀ i, actual_height i > actual_height (Fin.mk i (by linarith))) →
    ∀ claim_truth: Fin 7 → Prop, 
    (∀ i, claim_truth i ↔ actual_height i = heights[i]) →
    (∃ k, ∀ j, claim_truth j → j = k) :=
by
  intro heights actual_height h_order claim_truth h_claim_match
  existsi Fin.mk 0 (by linarith) -- Claim that only one can be telling the truth (arbitrary choice)
  sorry  -- Proof steps are omitted

end dwarfs_truth_claims_l497_497964


namespace ratio_of_adults_to_children_l497_497158

-- Definitions based on conditions
def adult_ticket_price : ℝ := 5.50
def child_ticket_price : ℝ := 2.50
def total_receipts : ℝ := 1026
def number_of_adults : ℝ := 152

-- Main theorem to prove ratio of adults to children is 2:1
theorem ratio_of_adults_to_children : 
  ∃ (number_of_children : ℝ), adult_ticket_price * number_of_adults + child_ticket_price * number_of_children = total_receipts ∧ 
  number_of_adults / number_of_children = 2 :=
by
  sorry

end ratio_of_adults_to_children_l497_497158


namespace intersection_range_l497_497702

-- Define the parametric equations for the circle and the line
def circle (θ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos θ, Real.sin θ)

def line (t α : ℝ) : ℝ × ℝ :=
  (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

-- Define the discriminant requirement for intersection
def intersection_discriminant_nonneg (α : ℝ) : Prop :=
  4 * (Real.sqrt 3 * Real.sin α + Real.cos α)^2 - 12 ≥ 0

-- Rewrite the problem as a Lean statement to prove the range of α
theorem intersection_range (α : ℝ) (hα : 0 ≤ α ∧ α ≤ Real.pi) :
  intersection_discriminant_nonneg α ↔ (Real.pi / 6 ≤ α ∧ α ≤ Real.pi / 2) := 
sorry

end intersection_range_l497_497702


namespace max_truthful_dwarfs_l497_497961

theorem max_truthful_dwarfs :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  (∃ n, n = 1 ∧ ∀ i, i < 7 → i ≠ n → heights[i] ≠ (heights[i] + 1)) :=
by
  let heights := [60, 61, 62, 63, 64, 65, 66]
  existsi 1
  intro i
  intro h
  intro hi
  sorry

end max_truthful_dwarfs_l497_497961


namespace odd_integers_equality_l497_497346

-- Definitions
def is_odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

def divides (d n : ℤ) := ∃ k : ℤ, n = d * k

-- Main statement
theorem odd_integers_equality (a b : ℤ) (ha_pos : 0 < a) (hb_pos : 0 < b)
 (ha_odd : is_odd a) (hb_odd : is_odd b)
 (h_div : divides (2 * a * b + 1) (a^2 + b^2 + 1))
 : a = b :=
by 
  sorry

end odd_integers_equality_l497_497346


namespace total_fruits_is_78_l497_497359

def oranges_louis : Nat := 5
def apples_louis : Nat := 3

def oranges_samantha : Nat := 8
def apples_samantha : Nat := 7

def oranges_marley : Nat := 2 * oranges_louis
def apples_marley : Nat := 3 * apples_samantha

def oranges_edward : Nat := 3 * oranges_louis
def apples_edward : Nat := 3 * apples_louis

def total_fruits_louis : Nat := oranges_louis + apples_louis
def total_fruits_samantha : Nat := oranges_samantha + apples_samantha
def total_fruits_marley : Nat := oranges_marley + apples_marley
def total_fruits_edward : Nat := oranges_edward + apples_edward

def total_fruits_all : Nat :=
  total_fruits_louis + total_fruits_samantha + total_fruits_marley + total_fruits_edward

theorem total_fruits_is_78 : total_fruits_all = 78 := by
  sorry

end total_fruits_is_78_l497_497359


namespace inequality_proof_inverse_inequality_proof_l497_497236

theorem inequality_proof (n : ℕ) (a x : Fin n → ℝ)
  (h_sum_nonneg: ∀ i j, 0 ≤ a i + a j)
  (h_x_nonneg : ∀ i, 0 ≤ x i)
  (h_sum_x : (Finset.univ.sum (λ i, x i) = 1)) :
  (Finset.univ.sum (λ i, a i * x i)) ≥ (Finset.univ.sum (λ i, a i * (x i)^2)) :=
sorry

theorem inverse_inequality_proof (n : ℕ) (a x : Fin n → ℝ)
  (h_sum_nonneg: ∀ i j, 0 ≤ a i + a j)
  (h_x_nonneg : ∀ i, 0 ≤ x i)
  (h_sum_x : (Finset.univ.sum (λ i, x i) = 1))
  (h_inequality: (Finset.univ.sum (λ i, a i * x i)) ≥ (Finset.univ.sum (λ i, a i * (x i)^2))) :
  (∀ i j, 0 ≤ a i + a j)
:=
sorry

end inequality_proof_inverse_inequality_proof_l497_497236


namespace no_integer_solutions_l497_497190

theorem no_integer_solutions :
  ∀ x y : ℤ, (2 : ℤ)^(x + 2) - (3 : ℤ)^(y + 1) ≠ 41 :=
by
  sorry

end no_integer_solutions_l497_497190


namespace range_of_numbers_is_six_l497_497112

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497112


namespace angle_APB_eq_95_l497_497312

-- Definitions from the conditions of the problem
variables (PA SAR PB RBT SRT : Type) [Tangent PA SAR] [Tangent PB RBT]
variables {line_SRT : SRT}

-- Define the arcs and their given measures
variables (arc_AS : angle) (arc_BT : angle)
variables (A B P S R T O1 O2 : Type)
variables (is_straight_SRT : is_straight S R T)
variables (angle_AO1R_eq : angle_AO1R = 180 - arc_AS)
variables (angle_BO2R_eq : angle_BO2R = 180 - arc_BT)

-- Question to prove
theorem angle_APB_eq_95 :
  ∀ PA SAR PB RBT SRT, 
    Tangent PA SAR → Tangent PB RBT →
    ∀ line_SRT, SRT → 
    ∀ arc_AS arc_BT A B P S R T O1 O2, 
      is_straight S R T → 
      arc_AS = 58 → arc_BT = 37 →
      ∃ angle_APB, angle_APB = 95 :=
  sorry

end angle_APB_eq_95_l497_497312


namespace range_of_set_of_three_numbers_l497_497133

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497133


namespace fondant_area_l497_497022

variable (r h : ℝ)

-- Given conditions: 
-- 1. r is the radius of the cylindrical cake
-- 2. h is the height of the cylindrical cake
-- 3. The radius of the fondant is 2 * r

theorem fondant_area (r : ℝ) (h : ℝ) : 
  let fondant_radius := 2 * r in 
  let fondant_area := π * fondant_radius^2 in 
  fondant_area = 4 * π * r^2 :=
by sorry

end fondant_area_l497_497022


namespace round_fraction_to_three_decimal_l497_497915

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497915


namespace find_circle_radius_l497_497771

noncomputable def radius_of_circle (a h : ℝ) : ℝ := (a^2 + 4 * h^2) / (8 * h)

theorem find_circle_radius (a h : ℝ) (h_pos : 0 < h) (a_pos : 0 < a) : 
    let R := radius_of_circle a h in R = (a^2 + 4 * h^2) / (8 * h) :=
by
    let R := radius_of_circle a h
    have : R = radius_of_circle a h := rfl
    rw [this]
    sorry

end find_circle_radius_l497_497771


namespace fraction_rounded_equals_expected_l497_497927

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497927


namespace range_of_set_l497_497051

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497051


namespace range_of_set_is_six_l497_497082

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497082


namespace range_of_set_l497_497121

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497121


namespace range_of_set_l497_497053

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497053


namespace factor_polynomial_l497_497641

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497641


namespace fraction_rounded_to_decimal_l497_497936

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497936


namespace find_k_l497_497210

theorem find_k (k : ℕ) :
  (∑' n : ℕ, (5 + n * k) / 5 ^ n) = 12 → k = 90 :=
by
  sorry

end find_k_l497_497210


namespace dwarfs_truth_claims_l497_497967

/-- Seven dwarfs lined up by height, starting with the tallest.
Each dwarf claims a specific height as follows.
Verify that the largest number of dwarfs telling the truth is exactly 1. -/
theorem dwarfs_truth_claims :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  ∀ (actual_height: Fin 7 → ℕ), 
    (∀ i, actual_height i > actual_height (Fin.mk i (by linarith))) →
    ∀ claim_truth: Fin 7 → Prop, 
    (∀ i, claim_truth i ↔ actual_height i = heights[i]) →
    (∃ k, ∀ j, claim_truth j → j = k) :=
by
  intro heights actual_height h_order claim_truth h_claim_match
  existsi Fin.mk 0 (by linarith) -- Claim that only one can be telling the truth (arbitrary choice)
  sorry  -- Proof steps are omitted

end dwarfs_truth_claims_l497_497967


namespace num_solutions_naturals_num_solutions_nonneg_ints_l497_497739

noncomputable def num_sol_nat := (999.choose 2)
noncomputable def num_sol_nonneg := (1002.choose 2)

theorem num_solutions_naturals : 
  ∃ (x_1 x_2 x_3: ℕ), x_1 + x_2 + x_3 = 1000 ∧ (nat.card {x | ∃ (x_1 x_2 x_3: ℕ),(x_1 + x_2 + x_3 = 1000)}) = num_sol_nat := 
sorry

theorem num_solutions_nonneg_ints : 
  ∃ (x_1 x_2 x_3: ℕ), x_1 + x_2 + x_3 = 1000 ∧ (nat.card {x | ∃ (x_1 x_2 x_3: ℕ),(x_1 + x_2 + x_3 = 1000)}) = num_sol_nonneg := 
sorry

end num_solutions_naturals_num_solutions_nonneg_ints_l497_497739


namespace lindy_speed_10_ft_per_sec_l497_497787

variable (Jack Christina Lindy : Type)
variable (distance_apart : ℕ) (distance_lindy : ℕ)
variable (speed_jack speed_christina : ℕ)

def time_to_meet (distance_apart speed_jack speed_christina : ℕ) : ℕ :=
  (distance_apart / (speed_jack + speed_christina))

def lindy's_speed (distance_lindy time : ℕ) : ℕ :=
  distance_lindy / time

theorem lindy_speed_10_ft_per_sec
  (distance_apart = 240)
  (speed_jack = 3)
  (speed_christina = 3)
  (distance_lindy = 400) :
  lindy's_speed distance_lindy (time_to_meet distance_apart speed_jack speed_christina) = 10 :=
by
  sorry

end lindy_speed_10_ft_per_sec_l497_497787


namespace average_x1_x2_x3_x4_x5_eq_3_l497_497293

variable (x1 x2 x3 x4 x5 : ℝ)

def average (xs : List ℝ) : ℝ := xs.sum / xs.length

theorem average_x1_x2_x3_x4_x5_eq_3
  (h : average [x1, x2, x3, x4, x5, 3] = 3) :
  average [x1, x2, x3, x4, x5] = 3 := by
  sorry

end average_x1_x2_x3_x4_x5_eq_3_l497_497293


namespace max_truthful_dwarfs_l497_497956

theorem max_truthful_dwarfs :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  (∃ n, n = 1 ∧ ∀ i, i < 7 → i ≠ n → heights[i] ≠ (heights[i] + 1)) :=
by
  let heights := [60, 61, 62, 63, 64, 65, 66]
  existsi 1
  intro i
  intro h
  intro hi
  sorry

end max_truthful_dwarfs_l497_497956


namespace gnomes_telling_the_truth_l497_497940

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l497_497940


namespace fraction_rounded_equals_expected_l497_497920

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497920


namespace problem_solution_l497_497281

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end problem_solution_l497_497281


namespace inequality_xyz_geq_3_l497_497836

theorem inequality_xyz_geq_3
  (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_not_all_zero : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :
  (2 * x^2 - x + y + z) / (x + y^2 + z^2) +
  (2 * y^2 + x - y + z) / (x^2 + y + z^2) +
  (2 * z^2 + x + y - z) / (x^2 + y^2 + z) ≥ 3 := 
sorry

end inequality_xyz_geq_3_l497_497836


namespace polynomial_factorization_l497_497566

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497566


namespace range_of_a_l497_497696

-- Definition: f is an odd function and for x >= 0, f(x) = x^2 + 2x
def f (x : ℝ) : ℝ := if x >= 0 then x^2 + 2*x else -(x^2 + 2*(-x))

-- Statement to prove: 2 - a^2 > a implies a ∈ (-2, 1)
theorem range_of_a (a : ℝ) (h : f (2 - a^2) > f a) : -2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l497_497696


namespace prob_chocolate_milk_5_of_6_days_l497_497822

theorem prob_chocolate_milk_5_of_6_days :
  let days_visited := 6
  let prob_chocolate_milk := (2/3 : ℝ)
  ((nat.choose days_visited 5) * prob_chocolate_milk^5 * (1 - prob_chocolate_milk)^1) = (64/243 : ℝ) :=
by sorry

end prob_chocolate_milk_5_of_6_days_l497_497822


namespace range_of_set_is_six_l497_497083

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497083


namespace triangle_equality_l497_497523

open Function

-- Definitions of equilateral triangle, inscribed circle, and points X, A, B, C
variables {A B C X : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X]
variables [EquilateralTriangle A B C] [InscribedInCircle A B C]
variables [PointOnArc X (arc A M C)]

theorem triangle_equality
  (hABC : EquilateralTriangle A B C)
  (hInscribed : InscribedInCircle A B C)
  (hX : PointOnArc X (arc A M C)) :
  dist A X + dist X C = dist B X :=
sorry

end triangle_equality_l497_497523


namespace rounded_to_three_decimal_places_l497_497895

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497895


namespace find_unique_positive_integers_l497_497651

theorem find_unique_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  3 ^ x + 7 = 2 ^ y → x = 2 ∧ y = 4 :=
by
  -- Proof will go here
  sorry

end find_unique_positive_integers_l497_497651


namespace fraction_rounded_to_decimal_l497_497938

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497938


namespace fake_square_polynomial_l497_497186

-- Define what it means for a positive integer to be a fake square
def is_fake_square (n : ℕ) : Prop :=
  n = 1 ∨ (∃ (f : ℕ → ℕ), (∀ i, prime (f i)) ∧ list.even (list.length (list.filter (λ x, f x ≠ 0) (list.map f (list.range n)))) ∧ (list.prod (list.filter (λ x, f x ≠ 0) (list.map f (list.range n))) = n))

-- Main theorem statement
theorem fake_square_polynomial (k : ℕ) (h : even k ∧ k ≥ 2) :
  ∃ (a : fin k → ℕ), pairwise (≠) (finset.univ.image a) ∧ (∀ x ∈ (finset.range 2024), is_fake_square (list.prod (list.map (λ i, (x + a i)) (list.fin_range k)))) :=
sorry

end fake_square_polynomial_l497_497186


namespace shaded_area_l497_497168

/-- Prove that the shaded area of a shape formed by removing four right triangles of legs 2 from each corner of a 6 × 6 square is equal to 28 square units -/
theorem shaded_area (a b c d : ℕ) (square_side_length : ℕ) (triangle_leg_length : ℕ)
  (h1 : square_side_length = 6)
  (h2 : triangle_leg_length = 2)
  (h3 : a = 1)
  (h4 : b = 2)
  (h5 : c = b)
  (h6 : d = 4*a) : 
  a * square_side_length * square_side_length - d * (b * b / 2) = 28 := 
sorry

end shaded_area_l497_497168


namespace calculator_unit_prices_and_min_cost_l497_497019

-- Definitions for conditions
def unit_price_type_A (x : ℕ) : Prop :=
  ∀ y : ℕ, (y = x + 10) → (550 / x = 600 / y)

def purchase_constraint (a : ℕ) : Prop :=
  25 ≤ a ∧ a ≤ 100

def total_cost (a : ℕ) (x y : ℕ) : ℕ :=
  110 * a + 120 * (100 - a)

-- Statement to prove
theorem calculator_unit_prices_and_min_cost :
  ∃ x y, unit_price_type_A x ∧ unit_price_type_A x ∧ total_cost 100 x y = 11000 :=
by
  sorry

end calculator_unit_prices_and_min_cost_l497_497019


namespace initial_savings_amount_l497_497490

theorem initial_savings_amount (A : ℝ) (P : ℝ) (r1 r2 t1 t2 : ℝ) (hA : A = 2247.50) (hr1 : r1 = 0.08) (hr2 : r2 = 0.04) (ht1 : t1 = 0.25) (ht2 : t2 = 0.25) :
  P = 2181 :=
by
  sorry

end initial_savings_amount_l497_497490


namespace part1_part2_l497_497811

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : A ∪ B a = B a ↔ a = 1 :=
by
  sorry

theorem part2 (a : ℝ) : A ∩ B a = B a ↔ a ≤ -1 ∨ a = 1 :=
by
  sorry

end part1_part2_l497_497811


namespace factor_polynomial_l497_497587

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497587


namespace range_of_set_l497_497098

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497098


namespace v_2002_eq_1_l497_497392

def g : ℕ → ℕ := λ x, match x with
  | 1 => 2
  | 2 => 3
  | 3 => 1
  | 4 => 4
  | 5 => 5
  | _ => 1 -- default case for safety (though it won't be used as per the problem's domain)

def v : ℕ → ℕ
| 0 => 2
| (n+1) => g (v n)

theorem v_2002_eq_1 : v 2002 = 1 :=
  sorry

end v_2002_eq_1_l497_497392


namespace round_fraction_to_three_decimal_l497_497905

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497905


namespace at_most_one_dwarf_tells_truth_l497_497979

noncomputable def dwarfs_tell_truth : Prop :=
  ∀ heights: Fin 7 → ℕ,
  heights = fun i => i + 60 →
  ∀ k ∈ Finset.range 7, (heights k) ≠ (60 + k) → (∃ i ∈ Finset.range 7, i ≠ k)

theorem at_most_one_dwarf_tells_truth :
  dwarfs_tell_truth = 1 := 
sorry

end at_most_one_dwarf_tells_truth_l497_497979


namespace motorcycle_speed_comparison_l497_497036

variables (v1 v2 : ℝ) (A B C : ℝ)

-- Define the conditions
def speeds_relation : Prop := v2 = 7 * v1

def pedestrian_total_time : Prop := (2 / (v1 + v2) = (1 / 4) * (1 / v1))

def motorcyclist_total_time : ℝ := (22 / 56) * (1 / v1)
def motorcyclist_direct_time : ℝ := (1 / 7) * (1 / v1)

def faster_by : ℝ := motorcyclist_total_time v1 - motorcyclist_direct_time v1

-- Define the proof problem
theorem motorcycle_speed_comparison (h1 : speeds_relation v1 v2) (h2 : pedestrian_total_time v1 v2) :
  faster_by v1 = 2.75 :=
sorry

end motorcycle_speed_comparison_l497_497036


namespace range_of_set_l497_497100

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497100


namespace range_of_set_l497_497122

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497122


namespace quadrilateral_area_l497_497313

noncomputable theory
open_locale big_operators

def AB := 15
def BC := 4
def CD := 3
def AD := 14
def angle_BCD := 90

theorem quadrilateral_area :
  let BD := Real.sqrt (BC^2 + CD^2),
  let s := (AB + BD + AD) / 2 in
  let area_BCD := (1/2) * BC * CD,
  let area_ABD := Real.sqrt (s * (s - AB) * (s - BD) * (s - AD)) in
  area_BCD + area_ABD = 41 :=
by
  let BD := Real.sqrt (BC^2 + CD^2),
  let s := (AB + BD + AD) / 2,
  let area_BCD := (1/2) * BC * CD,
  let area_ABD := Real.sqrt (s * (s - AB) * (s - BD) * (s - AD)),
  show area_BCD + area_ABD = 41, from sorry

end quadrilateral_area_l497_497313


namespace sequence_a113_l497_497782

theorem sequence_a113 {a : ℕ → ℝ} 
  (h1 : ∀ n, a n > 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n, (a (n+1))^2 + (a n)^2 = 2 * n * ((a (n+1))^2 - (a n)^2)) :
  a 113 = 15 :=
sorry

end sequence_a113_l497_497782


namespace range_of_numbers_is_six_l497_497141

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497141


namespace magnitude_of_angle_B_value_of_k_l497_497305

-- Define the conditions and corresponding proofs

variable {a b c : ℝ}
variable {A B C : ℝ} -- Angles in the triangle
variable (k : ℝ) -- Define k
variable (h1 : (2 * a - c) * Real.cos B = b * Real.cos C) -- Given condition for part 1
variable (h2 : (A + B + C) = Real.pi) -- Angle sum in triangle
variable (h3 : k > 1) -- Condition for part 2
variable (m_dot_n_max : ∀ (t : ℝ), 4 * k * t + Real.cos (2 * Real.arcsin t) = 5) -- Given condition for part 2

-- Proofs Required

theorem magnitude_of_angle_B (hA : 0 < A ∧ A < Real.pi) : B = Real.pi / 3 :=
by 
  sorry -- proof to be completed

theorem value_of_k : k = 3 / 2 :=
by 
  sorry -- proof to be completed

end magnitude_of_angle_B_value_of_k_l497_497305


namespace polynomial_factorization_l497_497563

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497563


namespace round_8_div_11_l497_497870

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497870


namespace volume_of_polyhedron_l497_497509

open Real

-- Define the conditions
def square_side : ℝ := 100  -- in cm, equivalent to 1 meter
def rectangle_length : ℝ := 40  -- in cm
def rectangle_width : ℝ := 20  -- in cm
def trapezoid_leg_length : ℝ := 130  -- in cm

-- Define the question as a theorem statement
theorem volume_of_polyhedron :
  ∃ V : ℝ, V = 552 :=
sorry

end volume_of_polyhedron_l497_497509


namespace factorization_correct_l497_497574

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497574


namespace odometer_problem_l497_497185

theorem odometer_problem (a b c : ℕ) (h₀ : a + b + c = 7) (h₁ : 1 ≤ a)
  (h₂ : a < 10) (h₃ : b < 10) (h₄ : c < 10) (h₅ : (c - a) % 20 = 0) : a^2 + b^2 + c^2 = 37 := 
  sorry

end odometer_problem_l497_497185


namespace range_of_set_l497_497044

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497044


namespace fraction_to_three_decimal_places_l497_497866

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497866


namespace fraction_rounded_to_decimal_l497_497934

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497934


namespace coeff_sum_eq_l497_497348

noncomputable def sum_coeffs (n : ℕ) : ℕ :=
  let a := (λ (i : ℕ), if i % 3 = 0 then 3^(n-1) else 0)
  in (finset.range (2*n+1)).sum a

theorem coeff_sum_eq (n : ℕ) (h : 0 < n) :
  sum_coeffs n = 3^(n-1) :=
sorry

end coeff_sum_eq_l497_497348


namespace max_truthful_dwarfs_l497_497954

theorem max_truthful_dwarfs :
  ∃ n : ℕ, n = 1 ∧ 
    ∀ heights : Fin 7 → ℕ, 
      (heights 0 = 60 ∨ heights 1 = 61 ∨ heights 2 = 62 ∨ heights 3 = 63 ∨ heights 4 = 64 ∨ heights 5 = 65 ∨ heights 6 = 66) → 
      (∀ i j : Fin 7, i < j → heights i > heights j) → 
      ( ∃ t : Fin 7 → bool, 
        ( ∀ i j : Fin 7, t i = tt → t j = tt → i = j) ∧ 
        ( ∀ i : Fin 7, t i = tt → heights i = 60 + i ) ∧
        #(t.to_list.count tt) = n ) :=
by
  use 1
  sorry

end max_truthful_dwarfs_l497_497954


namespace sum_of_probability_fractions_l497_497485

def total_tree_count := 15
def non_birch_count := 9
def birch_count := 6
def total_arrangements := Nat.choose 15 6
def non_adjacent_birch_arrangements := Nat.choose 10 6
def birch_probability := non_adjacent_birch_arrangements / total_arrangements
def simplified_probability_numerator := 6
def simplified_probability_denominator := 143
def answer := simplified_probability_numerator + simplified_probability_denominator

theorem sum_of_probability_fractions :
  answer = 149 := by
  sorry

end sum_of_probability_fractions_l497_497485


namespace net_rate_of_pay_25_dollars_per_hour_l497_497024

noncomputable def calculate_net_rate_of_pay(
  travel_time_hours: ℕ,
  speed_mph: ℕ,
  fuel_efficiency_mpg: ℕ,
  earnings_per_mile: ℝ,
  cost_per_gallon: ℝ
): ℝ :=
  (earnings_per_mile * (speed_mph * travel_time_hours) - cost_per_gallon * ((speed_mph * travel_time_hours) / fuel_efficiency_mpg)) / travel_time_hours

theorem net_rate_of_pay_25_dollars_per_hour:
  ∀ (travel_time_hours speed_mph fuel_efficiency_mpg: ℕ) (earnings_per_mile cost_per_gallon: ℝ),
    travel_time_hours = 3 →
    speed_mph = 50 →
    fuel_efficiency_mpg = 25 →
    earnings_per_mile = 0.60 →
    cost_per_gallon = 2.50 →
    calculate_net_rate_of_pay travel_time_hours speed_mph fuel_efficiency_mpg earnings_per_mile cost_per_gallon = 25 :=
by
  intros travel_time_hours speed_mph fuel_efficiency_mpg earnings_per_mile cost_per_gallon
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end net_rate_of_pay_25_dollars_per_hour_l497_497024


namespace factor_polynomial_l497_497584

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497584


namespace total_cookies_after_three_days_l497_497451

-- Define the initial conditions
def cookies_baked_monday : ℕ := 32
def cookies_baked_tuesday : ℕ := cookies_baked_monday / 2
def cookies_baked_wednesday_before : ℕ := cookies_baked_tuesday * 3
def brother_ate : ℕ := 4

-- Define the total cookies before brother ate any
def total_cookies_before : ℕ := cookies_baked_monday + cookies_baked_tuesday + cookies_baked_wednesday_before

-- Define the total cookies after brother ate some
def total_cookies_after : ℕ := total_cookies_before - brother_ate

-- The proof statement
theorem total_cookies_after_three_days : total_cookies_after = 92 := by
  -- Here, we would provide the proof, but we add sorry for now to compile successfully.
  sorry

end total_cookies_after_three_days_l497_497451


namespace chessboard_fraction_sum_l497_497479

theorem chessboard_fraction_sum (r s m n : ℕ) (h_r : r = 1296) (h_s : s = 204) (h_frac : (17 : ℚ) / 108 = (s : ℕ) / (r : ℕ)) : m + n = 125 :=
sorry

end chessboard_fraction_sum_l497_497479


namespace max_truthful_dwarfs_le_one_l497_497986

def dwarf_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

theorem max_truthful_dwarfs_le_one (heights : List ℕ) (h_lengths : heights.length = 7) (h_sorted : heights.sorted (· > ·)) :
  ∃ n, (n ≤ 1) ∧ ∀ i, i ∈ (Finset.range 7).to_list → (heights[i] ≠ dwarf_height_claims[i] → (1 = n)) :=
  sorry

end max_truthful_dwarfs_le_one_l497_497986


namespace range_of_set_l497_497078

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497078


namespace factorization_identity_l497_497613

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497613


namespace cube_edge_length_eq_six_l497_497400

theorem cube_edge_length_eq_six {s : ℝ} (h : s^3 = 6 * s^2) : s = 6 :=
sorry

end cube_edge_length_eq_six_l497_497400


namespace factor_polynomial_l497_497583

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497583


namespace children_on_playground_l497_497418

theorem children_on_playground (boys_soccer girls_soccer boys_swings girls_swings boys_snacks girls_snacks : ℕ)
(h1 : boys_soccer = 27) (h2 : girls_soccer = 35)
(h3 : boys_swings = 15) (h4 : girls_swings = 20)
(h5 : boys_snacks = 10) (h6 : girls_snacks = 5) :
boys_soccer + girls_soccer + boys_swings + girls_swings + boys_snacks + girls_snacks = 112 := by
  sorry

end children_on_playground_l497_497418


namespace range_of_set_l497_497101

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497101


namespace factor_polynomial_l497_497619

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497619


namespace hexagon_area_l497_497334

/-- Given a regular hexagon $ABCDEF$ with points $J$, $K$, and $L$ dividing sides
$AB$, $CD$, and $EF$ in the ratio $1:3$ from vertices $A$, $C$, and $E$ respectively,
and the area of $\triangle JKL$ is $81$, we aim to prove that the area of hexagon $ABCDEF$
is 1296. -/
theorem hexagon_area (A B C D E F J K L : Type) 
  (h_hexagon : regular_hexagon A B C D E F)
  (h1 : point_on_side_ratio A B J (1:3))
  (h2 : point_on_side_ratio C D K (1:3))
  (h3 : point_on_side_ratio E F L (1:3))
  (h_area_JKL : area_of_triangle J K L = 81) :
  area_of_hexagon A B C D E F = 1296 := 
sorry

end hexagon_area_l497_497334


namespace polynomial_factorization_l497_497568

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497568


namespace largest_positive_integer_n_l497_497290

def binary_operation (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_n (x : ℤ) (h : x = -15) : 
  ∃ (n : ℤ), n > 0 ∧ binary_operation n < x ∧ ∀ m > 0, binary_operation m < x → m ≤ n :=
by
  sorry

end largest_positive_integer_n_l497_497290


namespace factor_polynomial_l497_497616

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497616


namespace represent_all_integers_from_1_to_2019_l497_497835

theorem represent_all_integers_from_1_to_2019 :
  ∀ n : ℕ, 1 ≤ n → n ≤ 2019 → ∃ (expr : String), is_valid_expression expr 2 17 n :=
sorry

-- Additional definitions required for the theorem
def is_valid_expression (expr : String) (symbol : ℕ) (max_count : ℕ) (result : ℕ) : Prop :=
  -- Assuming a valid expression using the given symbol, within the count limit, that evaluates to result
  sorry

end represent_all_integers_from_1_to_2019_l497_497835


namespace probability_above_line_l497_497798

noncomputable def unit_square := Set.Icc (0 : ℝ) 1 ×ˢ Set.Icc (0 : ℝ) 1

noncomputable def point := (3/4 : ℝ, 1/4 : ℝ)

noncomputable def slope_condition (Q : ℝ × ℝ) : Prop :=
  (Q.2 - 1/4 : ℝ) / (Q.1 - 3/4 : ℝ) ≥ 2

noncomputable def probability_of_slope_condition : ℝ :=
  Classical.cardinality {Q : ℝ × ℝ | Q ∈ unit_square ∧ slope_condition Q}.toReal 
    / Classical.cardinality unit_square.toReal

theorem probability_above_line 
  (a b : ℕ) (h_rel_prime : Nat.gcd a b = 1) (h_eq : probability_of_slope_condition = ↑a / ↑b) :
  a = 1 ∧ b = 64 ∧ a + b = 65 :=
by
  sorry

end probability_above_line_l497_497798


namespace cos_comp_l497_497541

open Real

theorem cos_comp {a b c : ℝ} (h1 : a = cos (3 / 2)) (h2 : b = -cos (7 / 4)) (h3 : c = sin (1 / 10)) : 
  a < c ∧ c < b := 
by
  -- Assume the hypotheses
  sorry

end cos_comp_l497_497541


namespace find_d_l497_497397

noncomputable def distance (p q : ℝ × ℝ) := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem find_d (d : ℝ × ℝ)
  (h1 : ∀ t : ℝ, t = distance ⟨(4 + t * d.1), ((5 * (4 + t * d.1) - 9) / 6)⟩ ⟨4, 2⟩) :
  d = ⟨6 / real.sqrt 61, 5 / real.sqrt 61⟩ :=
sorry

end find_d_l497_497397


namespace range_of_b_div_a_l497_497694

theorem range_of_b_div_a 
  (a b : ℝ)
  (h1 : 0 < a) 
  (h2 : a ≤ 2)
  (h3 : b ≥ 1)
  (h4 : b ≤ a^2) : 
  (1 / 2) ≤ b / a ∧ b / a ≤ 2 := 
sorry

end range_of_b_div_a_l497_497694


namespace factor_polynomial_l497_497622

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497622


namespace spatial_analogy_l497_497336

open Vector

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]

noncomputable def planar_relationship
  (S₁ S₂ S₃ : ℝ) (a b c : Point) : Prop :=
  S₁ • a + S₂ • b + S₃ • c = 0

noncomputable def spatial_relationship
  (V₁ V₂ V₃ V₄ : ℝ) (a b c d : Point) : Prop :=
  V₁ • a + V₂ • b + V₃ • c + V₄ • d = 0

theorem spatial_analogy
  {A B C D O : Point}
  (S₁ S₂ S₃ : ℝ)
  (V₁ V₂ V₃ V₄ : ℝ)
  (h_planar : planar_relationship S₁ S₂ S₃ (A - O) (B - O) (C - O))
  (h_point_within_tetra : ¬ (collinear (D::(A::B::C)::nil)) ∧ D ∉ affine_span ℝ {A, B, C})
  : spatial_relationship V₁ V₂ V₃ V₄ (A - O) (B - O) (C - O) (D - O) :=
sorry

end spatial_analogy_l497_497336


namespace range_of_set_l497_497097

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497097


namespace convex_quadrilateral_two_sides_shorter_than_diagonal_l497_497006

theorem convex_quadrilateral_two_sides_shorter_than_diagonal {A B C D : Point} 
  (h: convex_quadrilateral A B C D) 
  (hAC_longest: is_longest_diagonal A C B D) 
  : exists (X Y : Point), X ≠ Y ∧ (X = A ∨ X = B ∨ X = C ∨ X = D) ∧ (Y = A ∨ Y = B ∨ Y = C ∨ Y = D) ∧ 
    (dist X Y < dist A C) :=
sorry

end convex_quadrilateral_two_sides_shorter_than_diagonal_l497_497006


namespace Tommy_accessible_area_l497_497426

def shed_width : ℝ := 4
def shed_length : ℝ := 3
def leash_length : ℝ := 4
def tree_distance : ℝ := 1
def tree_thickness : ℝ := 1

theorem Tommy_accessible_area : 
  let sector_area := (3 / 4) * Real.pi * leash_length^2 in
  let accessible_area := sector_area in
  accessible_area = 12 * Real.pi :=
by
  sorry

end Tommy_accessible_area_l497_497426


namespace area_triangle_agb_l497_497318

theorem area_triangle_agb
  (A B C D E G H: ℝ)
  (AC: ℝ) (AD: ℝ) (BE: ℝ) (tangentFromD: ℝ)
  (midpoint_D: D = (A + C) / 2)
  (midpoint_E: E = (A + B) / 2)
  (AH: H = (A + 1/3 * B))
  (length_AC_eq_30: AC = 30)
  (length_AD_eq_24: AD = 24)
  (length_BE_eq_30: BE = 30)
  (tangent_D_eq_15: tangentFromD = 15) :
  (areaOfTriangle A G B = 420) := sorry

end area_triangle_agb_l497_497318


namespace gamma_max_two_day_success_ratio_l497_497161

theorem gamma_max_two_day_success_ratio :
  ∃ (e g f h : ℕ), 0 < e ∧ 0 < g ∧
  e + g = 335 ∧ 
  e < f ∧ g < h ∧ 
  f + h = 600 ∧ 
  (e : ℚ) / f < (180 : ℚ) / 360 ∧ 
  (g : ℚ) / h < (150 : ℚ) / 240 ∧ 
  (e + g) / 600 = 67 / 120 :=
by
  sorry

end gamma_max_two_day_success_ratio_l497_497161


namespace total_number_of_employees_l497_497522

theorem total_number_of_employees (n : ℕ) (hm : ℕ) (hd : ℕ) 
  (h_ratio : 4 * hd = hm)
  (h_diff : hm = hd + 72) : n = 120 :=
by
  -- proof steps would go here
  sorry

end total_number_of_employees_l497_497522


namespace rounded_to_three_decimal_places_l497_497892

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497892


namespace terminating_decimal_multiples_l497_497216

theorem terminating_decimal_multiples :
  (∃ n : ℕ, 20 = n ∧ ∀ m, 1 ≤ m ∧ m ≤ 180 → 
  (∃ k : ℕ, m = 9 * k)) :=
by
  sorry

end terminating_decimal_multiples_l497_497216


namespace restore_original_l497_497411

def rearranged_message := "8,9ТекстМИМОПРАСТЕТИРАСИСПДАИСАФЕИИБ ОЕТКЖРГЛЕОЛОИШИСАННС Й С А О О Л Т Л Е Я Т У И Ц В Ы И П И Я ДПИЩ ПЬ П С Е Ю Я Я"
def substituted_message := "ТекстУ Щ Ф М ШП Д Р Е Ц Ч Е Ш Ю Ч Д А К Е Ч М Д В К Ш Б Е Е Ч ДФ Э П Й ЩГШ ФЩ ЦЕ ЮЩФ П МЕ Ч П М Р Р М Е О Ч Х ЕШ Р Т Г И Ф Р С ЯЫ Л К ДФ Ф Е Е"
noncomputable def original_message := "ШЕСТАЯОЛИМПИАДАПОКРИПТОГРАФИИПОСВЯЩЕНАСЕМИДЕСЯТИПЯТИЛЕТИЮСПЕЦИАЛЬНОЙСЛУЖБЫРОССИИ"

theorem restore_original (C1 C2 : string) 
  (h1 : C1 = rearranged_message) 
  (h2 : C2 = substituted_message) : 
  ∃ M, M = original_message :=
by
  -- statement without proof
  sorry

end restore_original_l497_497411


namespace probability_less_than_8_rings_l497_497310

def P_10_ring : ℝ := 0.20
def P_9_ring : ℝ := 0.30
def P_8_ring : ℝ := 0.10

theorem probability_less_than_8_rings : 
  (1 - (P_10_ring + P_9_ring + P_8_ring)) = 0.40 :=
by
  sorry

end probability_less_than_8_rings_l497_497310


namespace radar_placement_problem_l497_497212

noncomputable def max_distance (n : ℕ) (coverage_radius : ℝ) (central_angle : ℝ) : ℝ :=
  coverage_radius / Real.sin (central_angle / 2)

noncomputable def ring_area (inner_radius : ℝ) (outer_radius : ℝ) : ℝ :=
  Real.pi * (outer_radius ^ 2 - inner_radius ^ 2)

theorem radar_placement_problem (r : ℝ := 13) (n : ℕ := 5) (width : ℝ := 10) :
  let angle := 2 * Real.pi / n
  let max_dist := max_distance n r angle
  let inner_radius := (r ^ 2 - (r - width) ^ 2) / Real.tan (angle / 2)
  let outer_radius := inner_radius + width
  max_dist = 12 / Real.sin (angle / 2) ∧
  ring_area inner_radius outer_radius = 240 * Real.pi / Real.tan (angle / 2) :=
by
  sorry

end radar_placement_problem_l497_497212


namespace alice_coin_value_alice_coin_percentage_l497_497515

theorem alice_coin_value (p n d h T: ℕ) (hp: p = 2 * 1) (hn: n = 3 * 5) (hd: d = 4 * 10) (hh: h = 1 * 50) (hT: T = p + n + d + h) : T = 107 := 
by 
  simp [hp, hn, hd, hh] at hT 
  exact hT.symm
  
theorem alice_coin_percentage (p n d h T: ℕ) (hp: p = 2 * 1) (hn: n = 3 * 5) (hd: d = 4 * 10) (hh: h = 1 * 50) (hT: T = p + n + d + h) : (T / 100) * 100 = 107 :=
by 
  have hT_eq: T = 107 := alice_coin_value p n d h T hp hn hd hh hT
  simp [hT_eq]
  sorry

end alice_coin_value_alice_coin_percentage_l497_497515


namespace monotonic_intervals_minimum_tangent_l497_497714

open Real

def f (x : ℝ) : ℝ := log x - 1 / x - 2 * x

-- Part (1): Monotonic intervals of f(x)
theorem monotonic_intervals :
  (∀ x ∈ Ioo 0 1, deriv f x > 0) ∧ (∀ x ∈ Ioi 1, deriv f x < 0) := sorry

-- Part (2): Minimum value of a + b for the tangent line
def F (x : ℝ) : ℝ := f x + 2 * x
def h (x : ℝ) : ℝ := log x - 1 / x + 1 / (x * x) - 1

theorem minimum_tangent :
  (∃ (a b : ℝ) (x₀ : ℝ), a = deriv f x₀ ∧ b = F x₀ - a * x₀ ∧ a + b = h x₀ ∧ ∀ x > 0, h x ≥ -1)
    → ∃ a b, a + b = -1 := sorry

end monotonic_intervals_minimum_tangent_l497_497714


namespace find_d_radius_l497_497495

theorem find_d_radius (d : ℝ) : 
  let area_within_distance := λ (d : ℝ), 1000 * 2000 * (π * d^2) / 4,
      prob_condition := 3 / 4 in
  (area_within_distance d) / (1000 * 2000) = prob_condition → d = 0.5 :=
by
  intros
  -- Args and intermediate steps omitted for brevity
  sorry

end find_d_radius_l497_497495


namespace line_circle_intersect_two_points_shortest_chord_length_l497_497721

noncomputable def line (k : ℝ) : ℝ → ℝ :=
λ x, k * x + 1

noncomputable def circle (x y : ℝ) : Prop :=
(x - 1)^2 + (y + 1)^2 = 12

theorem line_circle_intersect_two_points (k : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ ∃ (y₁ y₂ : ℝ), y₁ = line k x₁ ∧ y₂ = line k x₂ ∧ circle x₁ y₁ ∧ circle x₂ y₂ :=
by
  sorry

theorem shortest_chord_length : 
  2 * real.sqrt(7) = real.sqrt(11 - (4 * real.exp 1 + 3) / (real.exp 1^2 + 1)) :=
by
  sorry

end line_circle_intersect_two_points_shortest_chord_length_l497_497721


namespace parabola_lambda_mu_sum_l497_497260

theorem parabola_lambda_mu_sum (p m x1 y1 x2 y2 x0 y0 : ℝ) (λ μ : ℝ) 
  (hp : p > 0) (hm : m ≠ 0)
  (hM : y1^2 = 2 * p * x1)
  (hN : y2^2 = 2 * p * x2)
  (hP : (x0, y0) = (0, y0))
  (hPM : (x1 - x0, y1 - y0) = λ * (m - x1, -y1))
  (hPN : (x2 - x0, y2 - y0) = μ * (m - x2, -y2)) :
  λ + μ = -1 :=
by
  sorry

end parabola_lambda_mu_sum_l497_497260


namespace same_function_l497_497518

def func_f (x : ℝ) (n : ℤ) : ℝ := real.root (x ^ (2 * n + 1)) (2 * n + 1).toNat
def func_g (x : ℝ) (n : ℤ) : ℝ := (real.root x (2 * n - 1).toNat) ^ (2 * n - 1)

theorem same_function (n : ℤ) : ∀ x : ℝ, func_f x n = func_g x n :=
by {
  intros,
  sorry
}

end same_function_l497_497518


namespace terminating_decimal_multiples_l497_497214

theorem terminating_decimal_multiples :
  {m : ℕ | 1 ≤ m ∧ m ≤ 180 ∧ m % 3 = 0}.to_finset.card = 60 := 
sorry

end terminating_decimal_multiples_l497_497214


namespace find_m_l497_497673

variable (m : ℝ)
def A := {-1, 3, m}
def B := {3, 4}

theorem find_m (h : B ∩ A = B) : m = 4 := by
  sorry

end find_m_l497_497673


namespace fraction_rounding_l497_497887

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497887


namespace factor_poly_eq_factored_form_l497_497603

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497603


namespace geometric_sum_problem_l497_497234

variable {a : ℕ → ℚ} (q : ℚ)

def is_geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
  ∃ c, ∀ n, a (n + 1) = a n * q

theorem geometric_sum_problem 
  (h_geo : is_geometric_sequence a q) 
  (h_a2 : a 2 = 1 / 4)
  (h_cond : a 2 * a 8 = 4 * (a 5 - 1)) :
  a 4 + a 5 + a 6 + a 7 + a 8 = 31 :=
sorry

end geometric_sum_problem_l497_497234


namespace pat_kate_mark_ratio_l497_497370

variables (P K M r : ℚ) 

theorem pat_kate_mark_ratio (h1 : P + K + M = 189) 
                            (h2 : P = r * K) 
                            (h3 : P = (1 / 3) * M) 
                            (h4 : M = K + 105) :
  r = 4 / 3 :=
sorry

end pat_kate_mark_ratio_l497_497370


namespace work_done_is_48_l497_497472

-- Define the force function F(x)
def F (x : ℝ) : ℝ := 5 * x + 2

-- Define the work done according to the integral from x = 0 to x = 4
noncomputable def work_done : ℝ := ∫ x in 0..4, F x

-- State the theorem
theorem work_done_is_48 : work_done = 48 :=
by
  sorry

end work_done_is_48_l497_497472


namespace angle_BPC_is_110_degrees_l497_497353

theorem angle_BPC_is_110_degrees 
  (A B C D E P : Point)
  (h1 : AB = AD)
  (h2 : angle A B C = 20)
  (h3 : midpoint E B C)
  (h4 : DE = EC)
  (h5 : BP = PC) :
  angle B P C = 110 := 
sorry

end angle_BPC_is_110_degrees_l497_497353


namespace range_of_set_l497_497119

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497119


namespace round_8_div_11_l497_497878

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497878


namespace dot_product_eq_neg29_l497_497180

def v := (3, -2)
def w := (-5, 7)

theorem dot_product_eq_neg29 : (v.1 * w.1 + v.2 * w.2) = -29 := 
by 
  -- this is where the detailed proof will occur
  sorry

end dot_product_eq_neg29_l497_497180


namespace minimum_sum_l497_497550

theorem minimum_sum (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) (hsum : ∑ i, x i = 1) :
    ∑ i j in Finset.filter (λ p, p.1 < p.2) (Finset.univ.product Finset.univ), x i.1 * x i.2 * (x i.1 + x i.2) = 0 := 
    sorry

end minimum_sum_l497_497550


namespace pizza_problem_solution_l497_497554

variable (L : ℕ)
variable (total_slices remaining_slices_after_lunch remaining_slices_after_dinner : ℕ)

def pizza_problem_conditions (L total_slices remaining_slices_after_lunch remaining_slices_after_dinner : ℕ) :=
  total_slices = 12 ∧
  remaining_slices_after_lunch = total_slices - L ∧
  remaining_slices_after_dinner = remaining_slices_after_lunch - (1/3) * remaining_slices_after_lunch ∧
  remaining_slices_after_dinner = 4

theorem pizza_problem_solution (L : ℕ) (total_slices remaining_slices_after_lunch remaining_slices_after_dinner : ℕ) 
  (h : pizza_problem_conditions L total_slices remaining_slices_after_lunch remaining_slices_after_dinner) :
  L / total_slices = 1 / 2 :=
begin
  sorry
end

end pizza_problem_solution_l497_497554


namespace smallest_positive_period_f_intervals_where_f_increasing_range_f_on_interval_l497_497255

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * sin (x + π / 3) - sqrt 3 * (sin x)^2 + sin x * cos x

-- (1) Prove smallest positive period is π
theorem smallest_positive_period_f : ∃ T > 0, ∀ x, f(x) = f(x + T) ∧ ∀ T' > 0, (∀ x, f(x) = f(x + T')) → T ≤ T' := by
  sorry

-- (2) Prove intervals where f is increasing
theorem intervals_where_f_increasing :
  ∀ k : ℤ, ∀ x, (k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12) →
  (∃ δ > 0, ∀ y, x ≤ y ∧ y ≤ x + δ → f y ≥ f x) := by
  sorry

-- (3) Prove the range of f when x ∈ [0, π/4] is [1, 2]
theorem range_f_on_interval : 
  ∀ x, 0 ≤ x ∧ x ≤ π / 4 → 1 ≤ f x ∧ f x ≤ 2 := by
  sorry

end smallest_positive_period_f_intervals_where_f_increasing_range_f_on_interval_l497_497255


namespace max_one_truthful_dwarf_l497_497990

def dwarf_height_claims : List (ℕ × ℕ) := [
  (1, 60),  -- First dwarf claims 60 cm
  (2, 61),  -- Second dwarf claims 61 cm
  (3, 62),  -- Third dwarf claims 62 cm
  (4, 63),  -- Fourth dwarf claims 63 cm
  (5, 64),  -- Fifth dwarf claims 64 cm
  (6, 65),  -- Sixth dwarf claims 65 cm
  (7, 66)   -- Seventh dwarf claims 66 cm
]

-- Proof problem statement: Prove that at most one dwarf is telling the truth about their height
theorem max_one_truthful_dwarf : ∀ (heights : List ℕ), heights.length = 7 → 
  (∀ i, i < heights.length → dwarf_height_claims.get? i ≠ none → (heights.get? i = dwarf_height_claims.get? i) → 
    ∑ i in finset.range heights.length, bool.to_num (heights.get? i = dwarf_height_claims.get? i) ≤ 1) :=
by
  intros heights len_heights valid_claims idx idx_in_range some_claim match_claim
  -- Add proof here 
  sorry

end max_one_truthful_dwarf_l497_497990


namespace ratio_water_heater_to_oven_l497_497789

-- Given conditions
def joseph_expenses (W O : ℕ) : Prop :=
  W + 3 * W + O = 1500 ∧ O = 500

-- The theorem to prove
theorem ratio_water_heater_to_oven (W O : ℕ) (h : joseph_expenses W O) : W / O = 1 / 2 :=
  by 
    unfold joseph_expenses at h
    cases h with h₁ h₂
    sorry

end ratio_water_heater_to_oven_l497_497789


namespace inequality_solution_l497_497438

theorem inequality_solution (x : ℝ) :
  (∀ y : ℝ, (0 < y) → (4 * (x^2 * y^2 + x * y^3 + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y + y)) ↔ (1 < x) :=
by
  sorry

end inequality_solution_l497_497438


namespace minimum_value_f_range_of_m_l497_497259

noncomputable def f (x m : ℝ) := x - m * Real.log x - (m - 1) / x
noncomputable def g (x : ℝ) := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem minimum_value_f (m : ℝ) (x : ℝ) (hx : 1 ≤ x ∧ x ≤ Real.exp 1) : 
  if m ≤ 2 then f x m = 2 - m 
  else if m ≥ Real.exp 1 + 1 then f x m = Real.exp 1 - m - (m - 1) / Real.exp 1 
  else f x m = m - 2 - m * Real.log (m - 1) :=
sorry

theorem range_of_m (m : ℝ) :
  (m ≤ 2 ∧ ∀ x2 ∈ [-2, 0], ∃ x1 ∈ [Real.exp 1, Real.exp 2], f x1 m ≤ g x2) ↔
  (m ∈ [ (Real.exp 2 - Real.exp 1 + 1) / (Real.exp 1 + 1), 2 ]) :=
sorry

end minimum_value_f_range_of_m_l497_497259


namespace f_2018_of_8_l497_497801

def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_string.to_list.map (λ c, c.to_nat - '0'.to_nat)).sum

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_k (k : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate f k n

theorem f_2018_of_8 : f_k 2018 8 = 5 := by
  sorry

end f_2018_of_8_l497_497801


namespace log_exp_identity_l497_497748

theorem log_exp_identity (x : ℝ) : x * Real.log 5 / Real.log 4 = 1 → 5 ^ x = 4 :=
by
  sorry

end log_exp_identity_l497_497748


namespace range_of_set_l497_497074

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497074


namespace factorization_correct_l497_497582

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497582


namespace prime_divisibility_infinitely_many_l497_497649

open Nat

theorem prime_divisibility_infinitely_many (p : ℕ) (hp : Prime p) (h := ne_of_gt (prime.gt_one hp)) :
  p ≠ 2 → ∃ᶠ n in at_top, p ∣ n^(n+1) + (n+1)^n :=
by
  sorry

end prime_divisibility_infinitely_many_l497_497649


namespace distance_with_wind_l497_497494

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

end distance_with_wind_l497_497494


namespace find_seashells_yesterday_l497_497425

-- Define the number of seashells found yesterday
def seashells_yesterday (y : ℕ) : Prop :=
  y + 4 = 11

theorem find_seashells_yesterday : ∃ y : ℕ, seashells_yesterday y ∧ y = 7 :=
begin
  sorry
end

end find_seashells_yesterday_l497_497425


namespace number_of_females_l497_497384

theorem number_of_females (total_people : ℕ) (avg_age_total : ℕ) 
  (avg_age_males : ℕ) (avg_age_females : ℕ) (females : ℕ) :
  total_people = 140 → avg_age_total = 24 →
  avg_age_males = 21 → avg_age_females = 28 → 
  females = 60 :=
by
  intros h1 h2 h3 h4
  -- Using the given conditions
  sorry

end number_of_females_l497_497384


namespace total_cookies_after_three_days_l497_497450

-- Define the initial conditions
def cookies_baked_monday : ℕ := 32
def cookies_baked_tuesday : ℕ := cookies_baked_monday / 2
def cookies_baked_wednesday_before : ℕ := cookies_baked_tuesday * 3
def brother_ate : ℕ := 4

-- Define the total cookies before brother ate any
def total_cookies_before : ℕ := cookies_baked_monday + cookies_baked_tuesday + cookies_baked_wednesday_before

-- Define the total cookies after brother ate some
def total_cookies_after : ℕ := total_cookies_before - brother_ate

-- The proof statement
theorem total_cookies_after_three_days : total_cookies_after = 92 := by
  -- Here, we would provide the proof, but we add sorry for now to compile successfully.
  sorry

end total_cookies_after_three_days_l497_497450


namespace factorization_identity_l497_497608

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497608


namespace family_ages_l497_497320

theorem family_ages:
  (∀ (Peter Harriet Jane Emily father: ℕ),
  ((Peter + 12 = 2 * (Harriet + 12)) ∧
   (Jane = Emily + 10) ∧
   (Peter = 60 / 3) ∧
   (Peter = Jane + 5) ∧
   (Aunt_Lucy = 52) ∧
   (Aunt_Lucy = 4 + Peter_Jane_mother) ∧
   (father - 20 = Aunt_Lucy)) →
  (Harriet = 4) ∧ (Peter = 20) ∧ (Jane = 15) ∧ (Emily = 5) ∧ (father = 72)) :=
sorry

end family_ages_l497_497320


namespace number_of_small_triangles_l497_497321

theorem number_of_small_triangles (n : ℕ) (A B C : ℝ) (points : Finset ℝ) 
  (h_collinear : ¬(AreCollinear points)) : points.card = 2008 → 
  (∀ (t : Finset ℝ), t ⊆ points ∪ {A, B, C} → (IsTriangle t)) → 
  ∃ (k : ℕ), k = 4017 :=
by
  intros h_points h_triangle
  sorry

end number_of_small_triangles_l497_497321


namespace calculate_difference_square_l497_497278

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end calculate_difference_square_l497_497278


namespace difference_of_squares_l497_497285

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end difference_of_squares_l497_497285


namespace prob_min_score_guaranteeing_payoff_l497_497306

-- Definitions
def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
def prob_single_throw (n : ℕ) : ℚ := if n ∈ dice_faces then 1 / 6 else 0
def two_throws_event : Event (Finset (ℕ × ℕ)) := 
  {e | e.1 ∈ dice_faces ∧ e.2 ∈ dice_faces}

-- Mathematical Statement
theorem prob_min_score_guaranteeing_payoff : 
  Probability (Event.filter two_throws_event (λ x : ℕ × ℕ, x.1 + x.2 = 12)) = 1 / 36 :=
by
  sorry

end prob_min_score_guaranteeing_payoff_l497_497306


namespace correct_conclusions_l497_497715

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) := Real.sin (ω * x + φ)
noncomputable def g (x : ℝ) (ω : ℝ) (φ : ℝ) := f (x - π / 12) ω φ

variables (ω : ℝ) (φ : ℝ)
-- given conditions
axiom ω_pos (h0 : ω > 0) : True
axiom φ_range (h1 : |φ| < π / 2) : True
axiom x1_x2_distance (h2 : ∀ x1 x2, abs (x1 - x2) = π / 2 → x1 ≠ x2 → ∃ T, T = π / ω) : True
axiom symmetry_axis (h3 : ∀ x, x = π / 3 → f x ω φ = f (π - x) ω φ) : True

theorem correct_conclusions : 
  ω = 2 ∧
  φ = -π / 6 ∧
  (∀ x, x ∈ Icc (-π / 6) 0 → monotone_on (λ x, f x ω φ) (Icc (-π / 6) 0)) ∧
  (∀ x, g x ω φ = g (π / 6 - x) ω φ) :=
by
  sorry

end correct_conclusions_l497_497715


namespace total_money_in_wallet_l497_497160

-- Definitions of conditions
def initial_five_dollar_bills := 7
def initial_ten_dollar_bills := 1
def initial_twenty_dollar_bills := 3
def initial_fifty_dollar_bills := 1
def initial_one_dollar_coins := 8

def spent_groceries := 65
def paid_fifty_dollar_bill := 1
def paid_twenty_dollar_bill := 1
def received_five_dollar_bill_change := 1
def received_one_dollar_coin_change := 5

def received_twenty_dollar_bills_from_friend := 2
def received_one_dollar_bills_from_friend := 2

-- Proving total amount of money
theorem total_money_in_wallet : 
  initial_five_dollar_bills * 5 + 
  initial_ten_dollar_bills * 10 + 
  initial_twenty_dollar_bills * 20 + 
  initial_fifty_dollar_bills * 50 + 
  initial_one_dollar_coins * 1 - 
  spent_groceries + 
  received_five_dollar_bill_change * 5 + 
  received_one_dollar_coin_change * 1 + 
  received_twenty_dollar_bills_from_friend * 20 + 
  received_one_dollar_bills_from_friend * 1 
  = 150 := 
by
  -- This is where the proof would be located
  sorry

end total_money_in_wallet_l497_497160


namespace factor_polynomial_l497_497635

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497635


namespace noncongruent_triangles_count_l497_497832

noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def P : Point := midpoint A B
noncomputable def Q : Point := midpoint B C
noncomputable def R : Point := midpoint A C

theorem noncongruent_triangles_count :
  is_right_triangle A B C ∧
  dist A B = 3 ∧
  dist B C = 4 ∧
  dist A C = 5 →
  (∃ S, S = {A, B, C, P, Q, R} ∧
  (triangle_count S = 4) :
  sorry

end noncongruent_triangles_count_l497_497832


namespace factorization_identity_l497_497615

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497615


namespace max_one_truthful_dwarf_l497_497993

def dwarf_height_claims : List (ℕ × ℕ) := [
  (1, 60),  -- First dwarf claims 60 cm
  (2, 61),  -- Second dwarf claims 61 cm
  (3, 62),  -- Third dwarf claims 62 cm
  (4, 63),  -- Fourth dwarf claims 63 cm
  (5, 64),  -- Fifth dwarf claims 64 cm
  (6, 65),  -- Sixth dwarf claims 65 cm
  (7, 66)   -- Seventh dwarf claims 66 cm
]

-- Proof problem statement: Prove that at most one dwarf is telling the truth about their height
theorem max_one_truthful_dwarf : ∀ (heights : List ℕ), heights.length = 7 → 
  (∀ i, i < heights.length → dwarf_height_claims.get? i ≠ none → (heights.get? i = dwarf_height_claims.get? i) → 
    ∑ i in finset.range heights.length, bool.to_num (heights.get? i = dwarf_height_claims.get? i) ≤ 1) :=
by
  intros heights len_heights valid_claims idx idx_in_range some_claim match_claim
  -- Add proof here 
  sorry

end max_one_truthful_dwarf_l497_497993


namespace fraction_rounding_l497_497889

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497889


namespace at_most_one_dwarf_tells_truth_l497_497975

noncomputable def dwarfs_tell_truth : Prop :=
  ∀ heights: Fin 7 → ℕ,
  heights = fun i => i + 60 →
  ∀ k ∈ Finset.range 7, (heights k) ≠ (60 + k) → (∃ i ∈ Finset.range 7, i ≠ k)

theorem at_most_one_dwarf_tells_truth :
  dwarfs_tell_truth = 1 := 
sorry

end at_most_one_dwarf_tells_truth_l497_497975


namespace find_x_for_f_eq_neg_one_l497_497708

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 1
else if x > 1 then 4 - x^2
else 0  -- defining in the otherwise case to make f total

theorem find_x_for_f_eq_neg_one (x : ℝ) : 
  f(x) = -1 ↔ (x = -2 ∨ x = Real.sqrt 5) := 
sorry

end find_x_for_f_eq_neg_one_l497_497708


namespace conic_section_is_ellipse_l497_497446

theorem conic_section_is_ellipse :
  ∀ (x y : ℝ),
    (sqrt (x^2 + (y - 2)^2) + sqrt ((x - 6)^2 + (y + 4)^2) = 12) →
    ∃ (a b : ℝ), a^2 > 0 ∧ b^2 > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) :=
by
  sorry

end conic_section_is_ellipse_l497_497446


namespace sum_first_three_coeffs_eq_29_l497_497447

-- Define the expression
def expr (b : ℝ) : ℝ := (1 - (2 / b)) ^ 7

-- Expansion using Binomial Theorem
def binomialExpansionFirstThreeCoeffs (b : ℝ) : List ℝ :=
  [b^7, -7 * b^6 * (2 / b), (7 * 6 / 2) * b^5 * (2 / b)^2]

-- Calculate the sum of coefficients
def sumOfFirstThreeCoeffs (b : ℝ) : ℝ :=
  sum (binomialExpansionFirstThreeCoeffs b)

-- The theorem to verify the sum of the first three coefficients
theorem sum_first_three_coeffs_eq_29 (b : ℝ) : sumOfFirstThreeCoeffs b = 29 := by
  sorry

end sum_first_three_coeffs_eq_29_l497_497447


namespace max_d_n_value_l497_497809

-- Definition of the set X of positive integers with distinct digits
def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.nodup

def X : set ℕ := {n | has_distinct_digits n}

-- Definition of A_n as the set of numbers formed by permuting the digits of n
def A_n (n : ℕ) : set ℕ :=
  {m | m ∈ permutations (n.digits 10) ∧ has_distinct_digits m}

-- Definition of d_n as the GCD of the elements of A_n
def d_n (n : ℕ) : ℕ :=
  gcd (A_n n)

-- The proof statement that the maximum value of d_n is 81
theorem max_d_n_value : ∀ n ∈ X, d_n n ≤ 81 := sorry

end max_d_n_value_l497_497809


namespace find_N_on_prism_l497_497527

-- Define the regular triangular prism structure
structure TriangularPrism where
  A B C A1 B1 C1 : ℝ^3 -- Vertices of the prism
  lateral_edge_len base_edge_len : ℝ
  lateral_edge_len_eq : lateral_edge_len = 2
  base_edge_len_eq : base_edge_len = 1
  midpoint_BC (M : ℝ^3) : M = (B + C) / 2

-- Define a point N on line OC1 such that MN is perpendicular to AB1
noncomputable def find_N 
  (prism : TriangularPrism)
  (N : ℝ^3)
  (MN_perp_AB1 : (N - (prism.B + prism.C) / 2) ⬝ (prism.B1 - prism.A) = 0) 
  : Prop :=
  N = (0, 0, 1)

-- Translate the math problem statement to a Lean theorem
theorem find_N_on_prism 
  (prism : TriangularPrism)
  (M : ℝ^3)
  (hM : M = (prism.B + prism.C) / 2)
  : ∃ N : ℝ^3, N = (0, 0, 1) ∧
    (N - M) ⬝ (prism.B1 - prism.A) = 0 :=
sorry

end find_N_on_prism_l497_497527


namespace find_M_lambda_l497_497230

theorem find_M_lambda (λ : ℝ) (n : ℕ) (hλ : λ > 0) (hn : n ≥ 2) :
  ∃ (M : ℝ), M > 0 ∧ 
    (∀ (x : Fin n → ℝ), (∀ i, 0 ≤ x i) → 
      M * ((Finset.univ.sum x) ^ n) ≥ (Finset.univ.sum (λ i, (x i)^n) + λ * (Finset.univ.prod x))) ∧
    M = max 1 ((n + λ) / (n^n)) :=
sorry

end find_M_lambda_l497_497230


namespace amount_of_flour_per_new_bread_roll_l497_497420

theorem amount_of_flour_per_new_bread_roll :
  (24 * (1 / 8) = 3) → (16 * f = 3) → (f = 3 / 16) :=
by
  intro h1 h2
  sorry

end amount_of_flour_per_new_bread_roll_l497_497420


namespace sum_of_possible_base2_digits_l497_497475

theorem sum_of_possible_base2_digits :
  ∀ n : ℕ, (n.to_digits 6).length = 4 →
  (∃ d : ℕ, (n.to_digits 2).length = d ∧ (d = 8 ∨ d = 9 ∨ d = 10 ∨ d = 11)) →
  (finset.sum (finset.filter (λ d, d = 8 ∨ d = 9 ∨ d = 10 ∨ d = 11) (finset.range 12)) id) = 38 :=
by
  sorry

end sum_of_possible_base2_digits_l497_497475


namespace max_distance_Q_to_plane_BCD_l497_497691

noncomputable def maximum_distance_Q_to_plane_BCD 
    (A B C D Q : ℝ × ℝ × ℝ)
    (h_AB : dist B A = 1)
    (h_AC : dist C A = 1)
    (h_AD : dist D A = 1)
    (h_perp_AB_AC : dot_product (B - A) (C - A) = 0)
    (h_perp_AB_AD : dot_product (B - A) (D - A) = 0)
    (h_perp_AC_AD : dot_product (C - A) (D - A) = 0)
    (h_on_sphere : dist Q ((1/2, 1/2, 1/2)) = sqrt 3 / 2) : 
    ℝ :=
    dist Q (1/3, 1/3, 1/3) + sqrt 3 / 2

theorem max_distance_Q_to_plane_BCD 
    (A B C D Q : ℝ × ℝ × ℝ)
    (h_AB : dist B A = 1)
    (h_AC : dist C A = 1)
    (h_AD : dist D A = 1)
    (h_perp_AB_AC : dot_product (B - A) (C - A) = 0)
    (h_perp_AB_AD : dot_product (B - A) (D - A) = 0)
    (h_perp_AC_AD : dot_product (C - A) (D - A) = 0)
    (h_on_sphere : dist Q ((1/2, 1/2, 1/2)) = sqrt 3 / 2) : 
    maximum_distance_Q_to_plane_BCD A B C D Q h_AB h_AC h_AD h_perp_AB_AC h_perp_AB_AD h_perp_AC_AD h_on_sphere = 
    2 * sqrt 3 / 3 :=
  sorry

end max_distance_Q_to_plane_BCD_l497_497691


namespace clarify_photos_needed_l497_497184

-- Definitions based on the conditions
def photos_by_Cristina : ℕ := 7
def photos_by_John : ℕ := 10
def photos_by_Sarah : ℕ := 9
def total_slots : ℕ := 40

-- Proof statement using the conditions
theorem clarify_photos_needed :
  photos_by_Cristina + photos_by_John + photos_by_Sarah + 14 = total_slots :=
begin
  -- Justify assuming the proof steps are correct
  sorry
end

end clarify_photos_needed_l497_497184


namespace angle_man_is_45_degree_l497_497329

theorem angle_man_is_45_degree
  (s : ℝ) (B C D M N : ℝ × ℝ)
  (BM DN NC : ℝ)
  (h_square : B ≠ C ∧ C ≠ D ∧ D ≠ B)
  (h_BC : BM = dist B C)
  (h_CD : DN = dist C D)
  (h_BM : dist B M = 21)
  (h_DN : dist D N = 4)
  (h_NC : dist N C = 24) :
  ∠(B, M, N) = 45 :=
sorry

end angle_man_is_45_degree_l497_497329


namespace range_of_set_l497_497063

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497063


namespace difference_of_squares_l497_497284

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end difference_of_squares_l497_497284


namespace distribute_a_eq_l497_497172

variable (a b c : ℝ)

theorem distribute_a_eq : a * (a + b - c) = a^2 + a * b - a * c := 
sorry

end distribute_a_eq_l497_497172


namespace factor_polynomial_l497_497634

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497634


namespace celsius_fahrenheit_idempotence_l497_497390

/-- 
For how many integer Celsius temperatures C between -10 and 50 inclusive does the original
temperature equal the final temperature after converting to Fahrenheit, rounding to the 
nearest tenth, converting back to Celsius, and rounding to the nearest integer?
-/
theorem celsius_fahrenheit_idempotence:
  (∑ C in Finset.range 61, if let real_F := (9/5) * (C - 10) + 32
     let rounded_F := ( (real_F * 10).round / 10)
     let C' := (5/9) * (rounded_F - 32)
     let rounded_C := C'.round
     in rounded_C = C - 10 then 1 else 0) = 61 :=
sorry

end celsius_fahrenheit_idempotence_l497_497390


namespace round_fraction_to_three_decimal_l497_497904

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497904


namespace four_digit_integers_with_remainders_l497_497735

theorem four_digit_integers_with_remainders :
  ∃ (n : ℕ) (z : ℕ), (1000 ≤ n ∧ n < 10000) ∧
    (n % 7 = 3) ∧
    (n % 10 = 6) ∧
    (n % 13 = 9) ∧ 
    (77 ≤ z ∧ z < 769) ∧
    (∃ (p : ℕ), (2 ≤ p ∧ p < 11) ∧ z = 70 * p - 1) ∧
    ∃! (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ (n = 13 * z + 9) ∧ (9 ∃! z, 77 ≤ z ∧ z < 769 ∧ (n = 13 * z + 9)) := sorry

end four_digit_integers_with_remainders_l497_497735


namespace max_one_truthful_dwarf_l497_497992

def dwarf_height_claims : List (ℕ × ℕ) := [
  (1, 60),  -- First dwarf claims 60 cm
  (2, 61),  -- Second dwarf claims 61 cm
  (3, 62),  -- Third dwarf claims 62 cm
  (4, 63),  -- Fourth dwarf claims 63 cm
  (5, 64),  -- Fifth dwarf claims 64 cm
  (6, 65),  -- Sixth dwarf claims 65 cm
  (7, 66)   -- Seventh dwarf claims 66 cm
]

-- Proof problem statement: Prove that at most one dwarf is telling the truth about their height
theorem max_one_truthful_dwarf : ∀ (heights : List ℕ), heights.length = 7 → 
  (∀ i, i < heights.length → dwarf_height_claims.get? i ≠ none → (heights.get? i = dwarf_height_claims.get? i) → 
    ∑ i in finset.range heights.length, bool.to_num (heights.get? i = dwarf_height_claims.get? i) ≤ 1) :=
by
  intros heights len_heights valid_claims idx idx_in_range some_claim match_claim
  -- Add proof here 
  sorry

end max_one_truthful_dwarf_l497_497992


namespace convert_rectangular_to_polar_l497_497183

theorem convert_rectangular_to_polar (x y : ℝ) (h₁ : x = -2) (h₂ : y = -2) : 
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (2 * Real.sqrt 2, 5 * Real.pi / 4) := by
  sorry

end convert_rectangular_to_polar_l497_497183


namespace interval_length_difference_l497_497187

theorem interval_length_difference (a b : ℝ) (h1 : a < b)
  (h2 : ∀ x, a ≤ x ∧ x ≤ b ↔ 1 ≤ 2^|x| ∧ 2^|x| ≤ 2)
  : (b - a) = 1 :=
by
  sorry

end interval_length_difference_l497_497187


namespace fraction_rounding_l497_497880

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497880


namespace binomial_sum_mod_eq_l497_497483

open Nat

theorem binomial_sum_mod_eq :
  (∑ k in range 11, (Nat.choose 11 k) * 6^(11 - k)) % 8 = 5 := 
by
  sorry

end binomial_sum_mod_eq_l497_497483


namespace range_of_set_l497_497065

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497065


namespace range_of_set_of_three_numbers_l497_497138

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497138


namespace a_2016_eq_neg_3_l497_497315

noncomputable def a : ℕ → ℤ
| 1 := -1
| 2 := 2
| n + 1 := (a n) + (a (n + 2))

theorem a_2016_eq_neg_3 : a 2016 = -3 := 
by
  sorry

end a_2016_eq_neg_3_l497_497315


namespace multiple_of_E_by_D_l497_497195

-- Define E and D based on given conditions
def E (n : ℕ) : ℕ := (List.range n).map (λ i, 2 * (i + 1)).product
def D (n : ℕ) : ℕ := (List.range n).map (λ i, 2 * i + 1).product

-- The proof statement translated to Lean 4
theorem multiple_of_E_by_D (n : ℕ) : ∃ m k : ℕ, k * E n = D n * 2^m ∧ m = n :=
by
  sorry

end multiple_of_E_by_D_l497_497195


namespace total_weight_correct_l497_497444

open Real

-- defining the molar mass of Chromic acid (H₂CrO₄)
def molar_mass_chromic_acid : ℝ := 118.02

-- defining the molar mass of Sodium hydroxide (NaOH)
def molar_mass_sodium_hydroxide : ℝ := 40.00

-- defining the number of moles of Chromic acid
def moles_chromic_acid : ℝ := 17.3

-- defining the number of moles of Sodium hydroxide
def moles_sodium_hydroxide : ℝ := 8.5

-- calculating the total weight
noncomputable def total_weight : ℝ := 
  (moles_chromic_acid * molar_mass_chromic_acid) + (moles_sodium_hydroxide * molar_mass_sodium_hydroxide)

theorem total_weight_correct : total_weight = 2381.746 :=
by
  unfold total_weight
  rw [mul_comm, mul_comm, mul_assoc, mul_assoc]
  sorry

end total_weight_correct_l497_497444


namespace dwarfs_truth_claims_l497_497971

/-- Seven dwarfs lined up by height, starting with the tallest.
Each dwarf claims a specific height as follows.
Verify that the largest number of dwarfs telling the truth is exactly 1. -/
theorem dwarfs_truth_claims :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  ∀ (actual_height: Fin 7 → ℕ), 
    (∀ i, actual_height i > actual_height (Fin.mk i (by linarith))) →
    ∀ claim_truth: Fin 7 → Prop, 
    (∀ i, claim_truth i ↔ actual_height i = heights[i]) →
    (∃ k, ∀ j, claim_truth j → j = k) :=
by
  intro heights actual_height h_order claim_truth h_claim_match
  existsi Fin.mk 0 (by linarith) -- Claim that only one can be telling the truth (arbitrary choice)
  sorry  -- Proof steps are omitted

end dwarfs_truth_claims_l497_497971


namespace factorization_correct_l497_497577

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497577


namespace factor_poly_eq_factored_form_l497_497599

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497599


namespace triangle_is_isosceles_l497_497015

/-- Given triangle ABC with angles A, B, and C, where C = π - (A + B),
    if 2 * sin A * cos B = sin C, then triangle ABC is an isosceles triangle -/
theorem triangle_is_isosceles
  (A B C : ℝ)
  (hC : C = π - (A + B))
  (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  A = B :=
by
  sorry

end triangle_is_isosceles_l497_497015


namespace triangle_area_computation_l497_497428

theorem triangle_area_computation :
  ∃ (p q r : ℕ), gcd p r = 1 ∧ (∀ (q_pos : q > 0), ¬ (∃ (k : ℕ), k * k ∣ q)) ∧
  let cosE := 7 / 10 in
  let sinE := 3 * real.sqrt 51 / 10 in
  let e := 7.5 * real.sqrt 51 in
  let area := (10 * (225 * real.sqrt 51 / 2)) / 5 in
  (area = 225 * real.sqrt 51 / 5 ∧ p + q + r = 281) := 
begin
  sorry
end

end triangle_area_computation_l497_497428


namespace range_of_numbers_is_six_l497_497104

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497104


namespace round_8_div_11_l497_497877

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497877


namespace kyle_age_l497_497377

theorem kyle_age :
  ∃ (kyle shelley julian frederick tyson casey : ℕ),
    shelley = kyle - 3 ∧ 
    shelley = julian + 4 ∧
    julian = frederick - 20 ∧
    frederick = 2 * tyson ∧
    tyson = 2 * casey ∧
    casey = 15 ∧ 
    kyle = 47 :=
by
  sorry

end kyle_age_l497_497377


namespace range_of_a_l497_497256

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*x + a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → f x a > 0) ↔ a > -3 := 
by sorry

end range_of_a_l497_497256


namespace loom_weaving_time_l497_497018

theorem loom_weaving_time 
    (meters_per_sec : ℝ := 0.132) 
    (total_meters : ℝ := 15) : 
    (total_meters / meters_per_sec).round = 114 := 
by
  sorry

end loom_weaving_time_l497_497018


namespace rounded_to_three_decimal_places_l497_497902

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497902


namespace lines_region_division_l497_497767

theorem lines_region_division (f : ℕ → ℕ) (k : ℕ) (h : k ≥ 2) : 
  (∀ m, f m = m * (m + 1) / 2 + 1) → f (k + 1) = f k + (k + 1) :=
by
  intro h_f
  have h_base : f 1 = 2 := by sorry
  have h_ih : ∀ n, n ≥ 2 → f (n + 1) = f n + (n + 1) := by sorry
  exact h_ih k h

end lines_region_division_l497_497767


namespace sequence_behavior_l497_497264

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0     => x
  | n + 1 => (sequence x n)^3 + 3 * (sequence x n) / (3 * (sequence x n)^2 + 1)

theorem sequence_behavior (x1 : ℝ) :
  (x1 < -1 → ∀ n, sequence x1 n < sequence x1 (n + 1) ∧ tendsto (λ n, sequence x1 n) at_top (nhds (-1))) ∧
  (-1 < x1 ∧ x1 < 0 → ∀ n, sequence x1 (n + 1) < sequence x1 n ∧ tendsto (λ n, sequence x1 n) at_top (nhds (-1))) ∧
  (0 < x1 ∧ x1 < 1 → ∀ n, sequence x1 n < sequence x1 (n + 1) ∧ tendsto (λ n, sequence x1 n) at_top (nhds 1)) ∧
  (1 < x1 → ∀ n, sequence x1 (n + 1) < sequence x1 n ∧ tendsto (λ n, sequence x1 n) at_top (nhds 1)) ∧
  (x1 = -1 ∨ x1 = 0 ∨ x1 = 1 → ∀ n, sequence x1 n = x1) :=
begin
  sorry
end

end sequence_behavior_l497_497264


namespace routes_A_to_B_l497_497733

theorem routes_A_to_B : (number_of_routes : ℕ) (two_moves_down two_moves_right: ℕ) 
  (total_moves: ℕ) (fact: ∀ n, (n + 1) * (n !).succ = (n + 1) !) 
  (binomial: (a : ℕ) → (b : ℕ) → (ℕ) := (nat.factorial (a + b) ) / ((nat.factorial a) * (nat.factorial b)))
  (correct_answer: binomial(2, 2) = 6) : 
  (routes_num : ℕ) := 
begin 
  have two_moves_down := 2, 
  have two_moves_right := 2, 
  have total_moves := two_moves_down + two_moves_right, 
  have num_routes := nat.binomial total_moves two_moves_down, 
  exact num_routes
end

#print routes_A_to_B

end routes_A_to_B_l497_497733


namespace find_k_l497_497701

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V) (k : ℝ)

-- Conditions
def not_collinear (a b : V) : Prop := ¬ ∃ (m : ℝ), b = m • a
def collinear (u v : V) : Prop := ∃ (m : ℝ), u = m • v

theorem find_k (h1 : not_collinear a b) (h2 : collinear (2 • a + k • b) (a - b)) : k = -2 :=
by
  sorry

end find_k_l497_497701


namespace Peggy_needs_to_add_stamps_l497_497167

theorem Peggy_needs_to_add_stamps :
  ∀ (Peggy_stamps Bert_stamps Ernie_stamps : ℕ),
  Peggy_stamps = 75 →
  Ernie_stamps = 3 * Peggy_stamps →
  Bert_stamps = 4 * Ernie_stamps →
  Bert_stamps - Peggy_stamps = 825 :=
by
  intros Peggy_stamps Bert_stamps Ernie_stamps hPeggy hErnie hBert
  sorry

end Peggy_needs_to_add_stamps_l497_497167


namespace domain_h_l497_497439

def h (x : ℝ) : ℝ := (5 * x - 2) / (2 * x - 10)

theorem domain_h :
  ∀ x : ℝ, x ∈ set.univ \ {5} ↔ x ∈ (set.Ioo (⊤ : ℝ) 5 ∪ set.Ioo 5 ⊥) := sorry

end domain_h_l497_497439


namespace imaginary_part_of_i_times_1_plus_i_is_1_l497_497745

-- Definitions based on the conditions
def imaginary_unit : ℂ := complex.I

-- Main theorem statement
theorem imaginary_part_of_i_times_1_plus_i_is_1 : 
  complex.im (imaginary_unit * (1 + imaginary_unit)) = 1 := 
sorry

end imaginary_part_of_i_times_1_plus_i_is_1_l497_497745


namespace fraction_rounded_equals_expected_l497_497918

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497918


namespace find_original_price_of_petrol_l497_497457

noncomputable def original_price_of_petrol (P : ℝ) : Prop :=
  280 / (0.9 * P) - 280 / P  = 5

theorem find_original_price_of_petrol : ∃ P : ℝ, original_price_of_petrol P ∧ P ≈ 6.22 :=
by
  existsi 6.22
  split
  · sorry
  · apply Real.approx 6.22


end find_original_price_of_petrol_l497_497457


namespace division_rounded_l497_497844

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497844


namespace factor_polynomial_l497_497639

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497639


namespace factor_polynomial_l497_497647

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497647


namespace school_days_in_year_l497_497417

variable (days_in_year : ℕ)
variable (aliyah_packs : ℕ)
variable (becky_packs : ℕ)
variable (half : ℕ → ℕ := (λ n, n / 2))

axiom aliyah_packs_half_time : aliyah_packs = half days_in_year
axiom becky_packs_half_aliyah : becky_packs = half aliyah_packs
axiom becky_packs_45_days : becky_packs = 45

theorem school_days_in_year : days_in_year = 180 :=
by
  sorry

end school_days_in_year_l497_497417


namespace find_m_l497_497223

theorem find_m (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 1/2) : 
  m = 100 := 
by
  sorry

end find_m_l497_497223


namespace isosceles_trapezoid_l497_497345

-- Define a type for geometric points
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define structures for geometric properties
structure Trapezoid :=
  (A B C D M N : Point)
  (is_midpoint_M : 2 * M.x = A.x + B.x ∧ 2 * M.y = A.y + B.y)
  (is_midpoint_N : 2 * N.x = C.x + D.x ∧ 2 * N.y = C.y + D.y)
  (AB_parallel_CD : (B.y - A.y) * (D.x - C.x) = (B.x - A.x) * (D.y - C.y)) -- AB || CD
  (MN_perpendicular_AB_CD : (N.y - M.y) * (B.y - A.y) + (N.x - M.x) * (B.x - A.x) = 0 ∧
                            (N.y - M.y) * (D.y - C.y) + (N.x - M.x) * (D.x - C.x) = 0) -- MN ⊥ AB && MN ⊥ CD

-- The isosceles condition
def is_isosceles (T : Trapezoid) : Prop :=
  ((T.A.x - T.D.x) ^ 2 + (T.A.y - T.D.y) ^ 2) = ((T.B.x - T.C.x) ^ 2 + (T.B.y - T.C.y) ^ 2)

-- The theorem statement
theorem isosceles_trapezoid (T : Trapezoid) : is_isosceles T :=
by
  sorry

end isosceles_trapezoid_l497_497345


namespace range_of_set_l497_497103

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497103


namespace find_k_l497_497677

def e1 : ℝ × ℝ := (1, 0)
def e2 : ℝ × ℝ := (0, 1)

def a : ℝ × ℝ := (e1.1, e1.2) - (2 * e2.1, 2 * e2.2)  -- vector subtraction
def b (k : ℝ) : ℝ × ℝ := (k * e1.1 + e2.1, k * e1.2 + e2.2)  -- vector addition and scalar multiplication

def vectors_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (λ : ℝ), (v1 = (λ * v2.1, λ * v2.2))

theorem find_k (k : ℝ) (h : vectors_parallel a (b k)) : k = -1 / 2 :=
sorry

end find_k_l497_497677


namespace fraction_rounded_to_decimal_l497_497929

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497929


namespace dwarfs_truth_claims_l497_497966

/-- Seven dwarfs lined up by height, starting with the tallest.
Each dwarf claims a specific height as follows.
Verify that the largest number of dwarfs telling the truth is exactly 1. -/
theorem dwarfs_truth_claims :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  ∀ (actual_height: Fin 7 → ℕ), 
    (∀ i, actual_height i > actual_height (Fin.mk i (by linarith))) →
    ∀ claim_truth: Fin 7 → Prop, 
    (∀ i, claim_truth i ↔ actual_height i = heights[i]) →
    (∃ k, ∀ j, claim_truth j → j = k) :=
by
  intro heights actual_height h_order claim_truth h_claim_match
  existsi Fin.mk 0 (by linarith) -- Claim that only one can be telling the truth (arbitrary choice)
  sorry  -- Proof steps are omitted

end dwarfs_truth_claims_l497_497966


namespace range_of_set_l497_497052

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497052


namespace intersection_complement_B_and_A_l497_497727

open Set Real

def A : Set ℝ := { x | x^2 - 4 * x + 3 < 0 }
def B : Set ℝ := { x | x > 2 }
def CR_B : Set ℝ := { x | x ≤ 2 }

theorem intersection_complement_B_and_A : CR_B ∩ A = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_complement_B_and_A_l497_497727


namespace sequence_limit_l497_497463

theorem sequence_limit :
    ∀ (a b : ℕ → ℝ), (∀ n, a n = (2 * (n:ℝ) ^ 2 + 2 * n + 3) / (2 * n ^ 2 - 2 * n + 5)) →
    ∀ n, b n = 5 - 7 * n →
    ∀ n, (λ n, (a n) ^ (b n)) ⟶ (λ n, e ^ (-14)) :=
begin
  assume (a b : ℕ → ℝ) (hn_a : ∀ n, a n = (2 * (n:ℝ) ^ 2 + 2 * n + 3) / (2 * n ^ 2 - 2 * n + 5))
         (hn_b : ∀ n, b n = (5 - 7 * n)),
  sorry
end

end sequence_limit_l497_497463


namespace smaller_number_is_15_l497_497660

theorem smaller_number_is_15 :
  ∀ (L S : ℤ), (L - S = 28) ∧ (L + 13 = 2 * (S + 13)) → S = 15 :=
by
  intros L S h,
  cases h with h1 h2,
  have h3 : L = S + 28 := h1,
  rw [h3] at h2,
  linarith,
sorry

end smaller_number_is_15_l497_497660


namespace percent_volume_occupied_is_correct_l497_497501

-- Define the dimensions of the box and cube
def box_length := 8
def box_width := 6
def box_height := 12
def cube_side := 4

-- Define the volumes
def volume_of_box := box_length * box_width * box_height
def volume_of_cube := cube_side ^ 3

-- Define the number of cubes that fit in each dimension
def cubes_in_length := Nat.div box_length cube_side
def cubes_in_width := Nat.div box_width cube_side
def cubes_in_height := Nat.div box_height cube_side

-- Define the total number of cubes
def total_cubes := cubes_in_length * cubes_in_width * cubes_in_height

-- Define the total volume occupied by the cubes
def volume_occupied_by_cubes := total_cubes * volume_of_cube

-- Define the percentage of the volume occupied
def percent_volume_occupied := (volume_occupied_by_cubes / volume_of_box : ℚ) * 100

-- State the theorem
theorem percent_volume_occupied_is_correct : 
  percent_volume_occupied ≈ 66.67 := 
by
  sorry

end percent_volume_occupied_is_correct_l497_497501


namespace max_truthful_dwarfs_l497_497958

theorem max_truthful_dwarfs :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  (∃ n, n = 1 ∧ ∀ i, i < 7 → i ≠ n → heights[i] ≠ (heights[i] + 1)) :=
by
  let heights := [60, 61, 62, 63, 64, 65, 66]
  existsi 1
  intro i
  intro h
  intro hi
  sorry

end max_truthful_dwarfs_l497_497958


namespace max_truthful_dwarfs_l497_497951

theorem max_truthful_dwarfs :
  ∃ n : ℕ, n = 1 ∧ 
    ∀ heights : Fin 7 → ℕ, 
      (heights 0 = 60 ∨ heights 1 = 61 ∨ heights 2 = 62 ∨ heights 3 = 63 ∨ heights 4 = 64 ∨ heights 5 = 65 ∨ heights 6 = 66) → 
      (∀ i j : Fin 7, i < j → heights i > heights j) → 
      ( ∃ t : Fin 7 → bool, 
        ( ∀ i j : Fin 7, t i = tt → t j = tt → i = j) ∧ 
        ( ∀ i : Fin 7, t i = tt → heights i = 60 + i ) ∧
        #(t.to_list.count tt) = n ) :=
by
  use 1
  sorry

end max_truthful_dwarfs_l497_497951


namespace polynomial_factorization_l497_497569

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497569


namespace flowers_given_l497_497843

theorem flowers_given (initial_flowers total_flowers flowers_given : ℝ)
  (h1 : initial_flowers = 67)
  (h2 : total_flowers = 157)
  (h3 : total_flowers = initial_flowers + flowers_given) :
  flowers_given = 90 :=
sorry

end flowers_given_l497_497843


namespace fraction_rounding_l497_497881

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497881


namespace sum_of_coefficients_of_expansion_l497_497176

theorem sum_of_coefficients_of_expansion :
  (let expr := (2 * x^2 + 3 * x * y - 4 * y^2)^5 in
   let sum_of_coeffs := expr.eval (fun _ => 1) in
   sum_of_coeffs = 1) := sorry

end sum_of_coefficients_of_expansion_l497_497176


namespace range_of_numbers_is_six_l497_497150

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497150


namespace range_of_set_l497_497055

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497055


namespace triangle_area_tangent_line_at_x_eq_1_l497_497713

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x + m * x ^ 2
noncomputable def tangent_line (n : ℝ) : LinearMap ℝ ℝ := LinearMap.mk (λ x : ℝ, n * x + (2 : ℝ))
noncomputable def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

theorem triangle_area_tangent_line_at_x_eq_1 {m n : ℝ} 
  (h_slope : (1 + 2 * m) = - n) 
  (h_point : f 1 m = 2 - n) : 
  triangle_area (2 / 5) 2 = 2 / 5 := 
sorry

end triangle_area_tangent_line_at_x_eq_1_l497_497713


namespace sin_exp_eqn_solutions_intervals_l497_497208

theorem sin_exp_eqn_solutions_intervals : 
  let sin_eq_exp (x : ℝ) := Real.sin x = (1 / 3) ^ x
  let interval_1 := set.Ioc 0 (150 * Real.pi)
  ∃ N ≥ 75, ∀ I ∈ (set.Ico 0 (75 : ℕ)).image (λ n, set.Ioc (2 * n * Real.pi) (2 * n * Real.pi + Real.pi)), 
    (∃ x ∈ I, sin_eq_exp x) → N = 75 := sorry

end sin_exp_eqn_solutions_intervals_l497_497208


namespace midpoint_product_is_six_l497_497191

-- Definitions of the problem conditions
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (1, 6)

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def coord_product (P : ℝ × ℝ) : ℝ :=
  P.1 * P.2

-- Statement of the problem
theorem midpoint_product_is_six : coord_product (midpoint A B) = 6 :=
by
  sorry

end midpoint_product_is_six_l497_497191


namespace range_of_set_l497_497067

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497067


namespace number_of_n_l497_497658

theorem number_of_n (n : ℕ) : 
  (∃ (n : ℕ), n ≤ 500 ∧ (∃ (k : ℕ), 30 * n = k^2)) → 
  (finset.filter (λ n, n ≤ 500 ∧ (∃ (k : ℕ), 30 * n = k^2)) (finset.range 501)).card = 4 :=
by
  sorry

end number_of_n_l497_497658


namespace find_smaller_number_l497_497813

theorem find_smaller_number
  (x y : ℝ) (m : ℝ)
  (h1 : x - y = 9) 
  (h2 : x + y = 46)
  (h3 : x = m * y) : 
  min x y = 18.5 :=
by 
  sorry

end find_smaller_number_l497_497813


namespace find_lambda_l497_497268

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (2, 1)
def vec_b (λ : ℝ) : ℝ × ℝ := (3, λ)

-- State the condition that the vectors are perpendicular
def are_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Theorem to prove that λ = -6
theorem find_lambda (λ : ℝ) (h : are_perpendicular vec_a (vec_b λ)) : λ = -6 :=
by
  sorry

end find_lambda_l497_497268


namespace count_primes_in_sequence_eq_one_l497_497737

def sequence (n : ℕ) : ℕ :=
  17 * ∑ k in (Finset.range (2 * n + 1)), 10 ^ k

theorem count_primes_in_sequence_eq_one :
  (Finset.filter (λ n, Nat.Prime (sequence n)) (Finset.range 100)).card = 1 :=
sorry

end count_primes_in_sequence_eq_one_l497_497737


namespace is_isosceles_triangle_l497_497303

variable (A B C : ℝ) -- Angles A, B, and C in triangle ABC.

-- Definition of the condition
def condition (A B C : ℝ) : Prop :=
  sin (A / 2) = cos ((A + B) / 2)

-- Statement we want to prove
theorem is_isosceles_triangle (A B C : ℝ) (h : condition A B C) (h_sum : A + B + C = π) : 
  A = C ∨ B = C ∨ A = B := 
sorry

end is_isosceles_triangle_l497_497303


namespace positive_integer_solutions_l497_497547

theorem positive_integer_solutions (a b : ℕ) (h_pos_ab : 0 < a ∧ 0 < b) :
  (∃ k : ℕ, k = a^2 / (2 * a * b^2 - b^3 + 1) ∧ 0 < k) ↔
  ∃ n : ℕ, (a = 2 * n ∧ b = 1) ∨ (a = n ∧ b = 2 * n) ∨ (a = 8 * n^4 - n ∧ b = 2 * n) :=
by
  sorry

end positive_integer_solutions_l497_497547


namespace range_of_set_l497_497059

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497059


namespace max_truthful_dwarfs_l497_497952

theorem max_truthful_dwarfs :
  ∃ n : ℕ, n = 1 ∧ 
    ∀ heights : Fin 7 → ℕ, 
      (heights 0 = 60 ∨ heights 1 = 61 ∨ heights 2 = 62 ∨ heights 3 = 63 ∨ heights 4 = 64 ∨ heights 5 = 65 ∨ heights 6 = 66) → 
      (∀ i j : Fin 7, i < j → heights i > heights j) → 
      ( ∃ t : Fin 7 → bool, 
        ( ∀ i j : Fin 7, t i = tt → t j = tt → i = j) ∧ 
        ( ∀ i : Fin 7, t i = tt → heights i = 60 + i ) ∧
        #(t.to_list.count tt) = n ) :=
by
  use 1
  sorry

end max_truthful_dwarfs_l497_497952


namespace factor_poly_eq_factored_form_l497_497597

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497597


namespace factor_polynomial_l497_497638

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497638


namespace original_data_properties_l497_497043

variables (n : ℕ) (x : ℕ → ℝ)

open scoped real

-- Given conditions
def new_data (x : ℕ → ℝ) := λ i, 2 * x i - 80
def average (f : ℕ → ℝ) : ℝ := (∑ i in range n, f i) / n
def variance (f : ℕ → ℝ) : ℝ := (∑ i in range n, (f i - average f)^2) / n

-- Prove the average and variance of the original set of data
theorem original_data_properties (h_avg_new : average (new_data x) = 1.2)
    (h_var_new : variance (new_data x) = 4.4) :
  average x = 40.6 ∧ variance x = 1.1 :=
by
  sorry

end original_data_properties_l497_497043


namespace factor_poly_eq_factored_form_l497_497595

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497595


namespace walk_fraction_correct_l497_497165

def bus_fraction := 1/3
def automobile_fraction := 1/5
def bicycle_fraction := 1/8
def metro_fraction := 1/15

def total_transport_fraction := bus_fraction + automobile_fraction + bicycle_fraction + metro_fraction

def walk_fraction := 1 - total_transport_fraction

theorem walk_fraction_correct : walk_fraction = 11/40 := by
  sorry

end walk_fraction_correct_l497_497165


namespace minimum_students_in_both_clubs_l497_497754

def ClassSize := 33
def MinimumClubSize := Int.ceil (0.7 * ClassSize)

theorem minimum_students_in_both_clubs (A B : Finset ℕ) (hA : A.card ≥ MinimumClubSize) (hB : B.card ≥ MinimumClubSize) :
  (A ∪ B).card ≤ ClassSize → (A ∩ B).card ≥ 15 :=
by
  sorry

end minimum_students_in_both_clubs_l497_497754


namespace range_of_set_l497_497116

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497116


namespace find_coefficients_find_extreme_values_l497_497711

noncomputable def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1
noncomputable def f' (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b
noncomputable def f'' (a : ℝ) (x : ℝ) : ℝ := 12 * x + 2 * a

theorem find_coefficients (a b : ℝ) (h1 : ¬ a / 6 = 1 / 2) (h2 : f' a b 1 = 0) : 
  a = 3 ∧ b = -12 :=
sorry

noncomputable def f_extreme_points :=
  let a := 3 in
  let b := -12 in
  (f a b (-2), f a b 1)

theorem find_extreme_values : 
  f_extreme_points = (21, -6) :=
sorry

end find_coefficients_find_extreme_values_l497_497711


namespace max_one_truthful_dwarf_l497_497994

def dwarf_height_claims : List (ℕ × ℕ) := [
  (1, 60),  -- First dwarf claims 60 cm
  (2, 61),  -- Second dwarf claims 61 cm
  (3, 62),  -- Third dwarf claims 62 cm
  (4, 63),  -- Fourth dwarf claims 63 cm
  (5, 64),  -- Fifth dwarf claims 64 cm
  (6, 65),  -- Sixth dwarf claims 65 cm
  (7, 66)   -- Seventh dwarf claims 66 cm
]

-- Proof problem statement: Prove that at most one dwarf is telling the truth about their height
theorem max_one_truthful_dwarf : ∀ (heights : List ℕ), heights.length = 7 → 
  (∀ i, i < heights.length → dwarf_height_claims.get? i ≠ none → (heights.get? i = dwarf_height_claims.get? i) → 
    ∑ i in finset.range heights.length, bool.to_num (heights.get? i = dwarf_height_claims.get? i) ≤ 1) :=
by
  intros heights len_heights valid_claims idx idx_in_range some_claim match_claim
  -- Add proof here 
  sorry

end max_one_truthful_dwarf_l497_497994


namespace fraction_rounded_to_decimal_l497_497930

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497930


namespace range_of_numbers_is_six_l497_497149

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497149


namespace range_of_set_of_three_numbers_l497_497129

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497129


namespace calculate_value_pa_pb_l497_497314

theorem calculate_value_pa_pb :
  let P := (2 : ℝ, 2 : ℝ)
  let α := real.pi / 3
  let l_parametric_eqns (t : ℝ) := (2 + 1/2 * t, 2 + (real.sqrt 3)/2 * t) -- Parametric equations of the line through P with inclination π/3
  let C_eqn (x y : ℝ) := x^2 + y^2 = 2 * x -- Rectangular equation of the circle from its polar form
  let intersection_pts := {t : ℝ | C_eqn (2 + 1/2 * t) (2 + (real.sqrt 3)/2 * t)} -- Intersection points
  let t1 t2 := classical.some (exists_pair this intersection_pts) -- Let t1 and t2 be the roots of the equation from the intersections
  in
  (1 / |t1| + 1 / |t2|) = (2 * real.sqrt 3 + 1) / 4 :=
sorry

end calculate_value_pa_pb_l497_497314


namespace division_rounded_l497_497855

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497855


namespace problem1_problem2_l497_497675

def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x | x > 1 ∨ x < -6}

theorem problem1 (a : ℝ) : (setA a ∩ setB = ∅) → (-6 ≤ a ∧ a ≤ -2) := by
  intro h
  sorry

theorem problem2 (a : ℝ) : (setA a ∪ setB = setB) → (a < -9 ∨ a > 1) := by
  intro h
  sorry

end problem1_problem2_l497_497675


namespace fraction_rounded_to_decimal_l497_497937

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497937


namespace range_of_set_l497_497056

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497056


namespace geometric_progression_fourth_term_l497_497389

theorem geometric_progression_fourth_term :
  let a1 := 3^(1/2)
  let a2 := 3^(1/3)
  let a3 := 3^(1/6)
  let r  := a3 / a2    -- Common ratio of the geometric sequence
  let a4 := a3 * r     -- Fourth term in the geometric sequence
  a4 = 1 := by
  sorry

end geometric_progression_fourth_term_l497_497389


namespace distinct_complex_roots_A_eq_neg7_l497_497548

theorem distinct_complex_roots_A_eq_neg7 (x₁ x₂ : ℂ) (A : ℝ) (hx1: x₁ ≠ x₂)
  (h1 : x₁ * (x₁ + 1) = A)
  (h2 : x₂ * (x₂ + 1) = A)
  (h3 : x₁^4 + 3 * x₁^3 + 5 * x₁ = x₂^4 + 3 * x₂^3 + 5 * x₂) : A = -7 := 
sorry

end distinct_complex_roots_A_eq_neg7_l497_497548


namespace imaginary_part_is_one_l497_497680

def i : ℂ := complex.I
def z_conjugate (z : ℂ) := conj z
def condition : Prop := ∀ z : ℂ, (2 - i) * z_conjugate z = 3 - 4 * i

theorem imaginary_part_is_one (z : ℂ) (h : condition z) : z.im = 1 := by
  sorry

end imaginary_part_is_one_l497_497680


namespace rhombus_area_l497_497653

theorem rhombus_area (s : ℝ) (θ : ℝ) (hθ : θ = 55) (h_sin : Real.sin θ = 0.8192) :
  s = 8 →
  (s ^ 2 * Real.sin θ ≈ 52.4288) :=
by 
  intros h_s
  rw [h_s, hθ, h_sin]
  sorry

end rhombus_area_l497_497653


namespace product_of_distinct_nonzero_real_numbers_l497_497241

variable {x y : ℝ}

theorem product_of_distinct_nonzero_real_numbers (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 4 / x = y + 4 / y) : x * y = 4 := 
sorry

end product_of_distinct_nonzero_real_numbers_l497_497241


namespace quadratic_function_opens_downwards_l497_497752

theorem quadratic_function_opens_downwards (m : ℤ) (h1 : |m| = 2) (h2 : m + 1 < 0) : m = -2 := by
  sorry

end quadratic_function_opens_downwards_l497_497752


namespace power_sum_l497_497532

theorem power_sum : (-2) ^ 2007 + (-2) ^ 2008 = 2 ^ 2007 := by
  sorry

end power_sum_l497_497532


namespace factor_polynomial_l497_497586

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497586


namespace factor_polynomial_l497_497640

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497640


namespace fraction_rounding_l497_497886

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497886


namespace length_of_AC_isosceles_triangle_l497_497828

theorem length_of_AC_isosceles_triangle (A B C M : Type) 
  [IsoscelesTriangle A B C] (h1 : OnLineSegment M A C) 
  (h2 : AM = 7) (h3 : MB = 3) (h4 : ∠ B M C = 60) : 
  AC = 17 := 
sorry

end length_of_AC_isosceles_triangle_l497_497828


namespace at_most_one_dwarf_tells_truth_l497_497978

noncomputable def dwarfs_tell_truth : Prop :=
  ∀ heights: Fin 7 → ℕ,
  heights = fun i => i + 60 →
  ∀ k ∈ Finset.range 7, (heights k) ≠ (60 + k) → (∃ i ∈ Finset.range 7, i ≠ k)

theorem at_most_one_dwarf_tells_truth :
  dwarfs_tell_truth = 1 := 
sorry

end at_most_one_dwarf_tells_truth_l497_497978


namespace division_rounded_l497_497847

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497847


namespace rounded_to_three_decimal_places_l497_497893

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497893


namespace ellipse_equation_distance_between_O_AB_l497_497521

noncomputable def eccentricity := 1 / 2
noncomputable def distance := sqrt 21 / 7
noncomputable def a := 2
noncomputable def b := sqrt 3

theorem ellipse_equation : 
  (∃ (a b : ℝ), (0 < b) ∧ (b < a) ∧ (eccentricity = 1 / 2) ∧ 
  (distance = sqrt 21 / 7) ∧ (a = 2) ∧ (b = sqrt 3)) → 
  (∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1) :=
by
  intro h
  sorry

theorem distance_between_O_AB :
  (∃ (m k : ℝ), OA_perpendicular_ob OB OA OB (eq_of_line_AB (distance = sqrt 21 / 7))) → 
  (∃ (d : ℝ), d = 2 * sqrt 21 / 7) :=
by
  intro h
  sorry 

end ellipse_equation_distance_between_O_AB_l497_497521


namespace overall_profit_percentage_l497_497508

def W1 : ℕ := 900
def W2 : ℕ := 960
def W3 : ℕ := 980

def R1 : ℕ := 2
def R2 : ℕ := 3
def R3 : ℕ := 5

def total_ratio : ℕ := R1 + R2 + R3

def weighted_average_weight : ℚ := 
  (W1 * R1 + W2 * R2 + W3 * R3) / total_ratio

def profit_grams : ℕ := 1000 - weighted_average_weight.to_nat

def profit_percentage : ℚ := (profit_grams.to_rat / weighted_average_weight) * 100

theorem overall_profit_percentage : profit_percentage ≈ 4.38 :=
by sorry

end overall_profit_percentage_l497_497508


namespace workers_problem_l497_497322

theorem workers_problem
    (n : ℕ)
    (total_workers : ℕ)
    (c_choose_2 : ℕ → ℕ)
    (probability_jack_jill : ℚ) :
    total_workers = n + 2 →
    c_choose_2 total_workers = (total_workers * (total_workers - 1)) / 2 →
    probability_jack_jill = 1 / (c_choose_2 total_workers) →
    probability_jack_jill = 1 / 6 →
    n = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1] at h2
  sorry

end workers_problem_l497_497322


namespace leftover_space_is_one_meter_l497_497163

theorem leftover_space_is_one_meter
  (total_length : ℝ)
  (desk_length : ℝ)
  (bookcase_length : ℝ)
  (num_desks : ℕ)
  (num_bookcases : ℕ)
  (h1 : total_length = 15)
  (h2 : desk_length = 2)
  (h3 : bookcase_length = 1.5)
  (h4 : num_desks = num_bookcases)
  (h5 : 3.5 * num_desks ≤ 15) :
  15 - (3.5 * ↑num_desks) = 1 := 
sorry

end leftover_space_is_one_meter_l497_497163


namespace range_of_set_is_six_l497_497088

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497088


namespace range_of_set_of_three_numbers_l497_497135

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497135


namespace age_of_teacher_l497_497010

/-- Given that the average age of 23 students is 22 years, and the average age increases
by 1 year when the teacher's age is included, prove that the teacher's age is 46 years. -/
theorem age_of_teacher (n : ℕ) (s_avg : ℕ) (new_avg : ℕ) (teacher_age : ℕ) :
  n = 23 →
  s_avg = 22 →
  new_avg = s_avg + 1 →
  teacher_age = new_avg * (n + 1) - s_avg * n →
  teacher_age = 46 :=
by
  intros h_n h_s_avg h_new_avg h_teacher_age
  sorry

end age_of_teacher_l497_497010


namespace polynomial_factorization_l497_497567

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497567


namespace max_diff_mean_division_num_ordered_pairs_max_diff_l497_497812

noncomputable def mean (s : Finset ℕ) (k : ℕ) : ℚ :=
  (s.sum id : ℚ) / k

-- Main theorem statement
theorem max_diff_mean_division (n : ℕ) (A B : Finset ℕ)
  (hA : A.nonempty) (hB : B.nonempty) (h_disjoint : A ∩ B = ∅) (h_union : A ∪ B = Finset.range n) :
  |mean A A.card - mean B B.card| ≤ n / 2 := sorry

theorem num_ordered_pairs_max_diff (n : ℕ) :
  ∃! (A B : Finset ℕ), |mean A A.card - mean B B.card| = n / 2 ∧ A ∩ B = ∅ ∧ A ∪ B = Finset.range n := sorry

end max_diff_mean_division_num_ordered_pairs_max_diff_l497_497812


namespace true_vs_claimed_discount_difference_l497_497492

theorem true_vs_claimed_discount_difference:
  let initial_discount := 30 / 100
  let additional_discount := 20 / 100
  let claimed_discount := 50 / 100
  let original_price := 1 in
  let price_after_first_discount := original_price * (1 - initial_discount) in
  let price_after_second_discount := price_after_first_discount * (1 - additional_discount) in
  let true_discount := 1 - price_after_second_discount in
  let difference := claimed_discount - true_discount in
  abs difference = 6 / 100 :=
by
  sorry

end true_vs_claimed_discount_difference_l497_497492


namespace carol_rectangle_length_l497_497535

theorem carol_rectangle_length :
  ∃ L : ℝ, (L * 24 = 4 * 30) ∧ (L = 5) :=
by
  use 5
  split
  · sorry
  · sorry

end carol_rectangle_length_l497_497535


namespace product_divisibility_l497_497235

def sequence (a b : ℕ) : ℕ → ℕ
| 0     := 0
| 1     := 1
| (2*n) := a * (sequence a b (2*n - 1)) - (sequence a b (2*n - 2))
| (2*n+1) := b * (sequence a b (2*n)) - (sequence a b (2*n - 1))

theorem product_divisibility (a b : ℕ) (h_a : a > 1) (h_b : b > 1) (m n : ℕ) (pos_m : m > 0) (pos_n : n > 0) : 
  ∃ k : ℕ, (k * (sequence a b (n+m)) * (sequence a b (n+m-1)) * ... * (sequence a b (n+1)) = 
  (sequence a b m) * (sequence a b (m-1)) * ... * (sequence a b 1)) :=
sorry

end product_divisibility_l497_497235


namespace max_truthful_dwarfs_l497_497948

theorem max_truthful_dwarfs :
  ∃ n : ℕ, n = 1 ∧ 
    ∀ heights : Fin 7 → ℕ, 
      (heights 0 = 60 ∨ heights 1 = 61 ∨ heights 2 = 62 ∨ heights 3 = 63 ∨ heights 4 = 64 ∨ heights 5 = 65 ∨ heights 6 = 66) → 
      (∀ i j : Fin 7, i < j → heights i > heights j) → 
      ( ∃ t : Fin 7 → bool, 
        ( ∀ i j : Fin 7, t i = tt → t j = tt → i = j) ∧ 
        ( ∀ i : Fin 7, t i = tt → heights i = 60 + i ) ∧
        #(t.to_list.count tt) = n ) :=
by
  use 1
  sorry

end max_truthful_dwarfs_l497_497948


namespace factor_polynomial_l497_497637

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497637


namespace train_crossing_time_l497_497533

def train_length := 120 -- meters
def bridge_length := 160 -- meters
def train_speed_kmh := 40 -- km/hour

noncomputable def total_distance := train_length + bridge_length -- meters
noncomputable def train_speed_ms := train_speed_kmh * (1000 / 3600) -- meters/second

theorem train_crossing_time :
  (total_distance / train_speed_ms) ≈ 25.20 :=
sorry

end train_crossing_time_l497_497533


namespace number_of_planes_l497_497330

theorem number_of_planes (a b : ℕ) (S : Set Point) (hS_card : S.card = a + b + 3) 
  (hS_no_coplanar : ∀ (p1 p2 p3 p4 : Point), p1 ∈ S → p2 ∈ S → p3 ∈ S → p4 ∈ S → 
                   ¬ coplanar {p1, p2, p3, p4}) :
  let planes_count := if a = b then (a + 1) * (b + 1) else 2 * (a + 1) * (b + 1) in
  planes_count = number_of_planes_through_three_points_separating S a b :=
sorry

end number_of_planes_l497_497330


namespace range_of_set_l497_497099

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497099


namespace asymptotes_of_hyperbola_l497_497751

-- Defining the problem conditions
def hyperbola (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

def eccentricity (a b : ℝ) (e : ℝ) : Prop := 
  e = Real.sqrt 3 ∧ b^2 = 2 * a^2

-- The main theorem to prove the answer
theorem asymptotes_of_hyperbola (a b : ℝ) (e : ℝ) 
  (h : eccentricity a b e) : 
  ∀ (x : ℝ), y = Math.sqrt 2 * x ∨ y = - Math.sqrt 2 * x :=
sorry

end asymptotes_of_hyperbola_l497_497751


namespace pyramid_coloring_l497_497170

-- Define the set of colors
def colors := Finset (Fin 5)

-- A function that checks if adjacent faces of the pyramid have the same color
def valid_coloring (colors : Fin 5 → Fin 5) : Prop :=
  colors 0 ≠ colors 1 ∧ colors 0 ≠ colors 2 ∧ colors 0 ≠ colors 3 ∧ colors 0 ≠ colors 4 ∧
  colors 1 ≠ colors 2 ∧ colors 2 ≠ colors 3 ∧ colors 3 ≠ colors 4 ∧ colors 4 ≠ colors 1 ∧
  colors 1 ≠ colors 3 ∧ colors 2 ≠ colors 4

-- Statement of the theorem
theorem pyramid_coloring : (Finset.univ.filter valid_coloring).card = 1200 := sorry

end pyramid_coloring_l497_497170


namespace max_truthful_dwarfs_le_one_l497_497987

def dwarf_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

theorem max_truthful_dwarfs_le_one (heights : List ℕ) (h_lengths : heights.length = 7) (h_sorted : heights.sorted (· > ·)) :
  ∃ n, (n ≤ 1) ∧ ∀ i, i ∈ (Finset.range 7).to_list → (heights[i] ≠ dwarf_height_claims[i] → (1 = n)) :=
  sorry

end max_truthful_dwarfs_le_one_l497_497987


namespace james_new_weight_is_correct_l497_497323

-- Define initial conditions
def initial_weight : ℝ := 120
def percentage_muscle_gain : ℝ := 0.20
def muscle_gain (initial_weight : ℝ) (percentage_muscle_gain : ℝ) : ℝ := initial_weight * percentage_muscle_gain
def fat_gain (muscle_gain : ℝ) : ℝ := muscle_gain / 4

-- Calculate new weight
def new_weight (initial_weight : ℝ) (muscle_gain : ℝ) (fat_gain : ℝ) : ℝ :=
  initial_weight + muscle_gain + fat_gain

-- Theorem statement
theorem james_new_weight_is_correct :
  new_weight initial_weight (muscle_gain initial_weight percentage_muscle_gain) (fat_gain (muscle_gain initial_weight percentage_muscle_gain)) = 150 :=
by
  sorry

end james_new_weight_is_correct_l497_497323


namespace gardener_bushes_needed_l497_497486

noncomputable def bushes_needed_around_ellipse
  (a b : ℝ) (spacing : ℝ) : ℝ :=
  π * (3 * (a + b) - real.sqrt ((3 * a + b) * (a + 3 * b)))

theorem gardener_bushes_needed :
  let a := 15
  let b := 10
  let spacing := 1
  abs (bushes_needed_around_ellipse a b spacing - 79) < 1 :=
by
  let a := 15
  let b := 10
  let spacing := 1
  have C := bushes_needed_around_ellipse a b spacing
  have hC_approx : abs (C - 79) < 1, sorry
  exact hC_approx

end gardener_bushes_needed_l497_497486


namespace max_truthful_dwarfs_le_one_l497_497981

def dwarf_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

theorem max_truthful_dwarfs_le_one (heights : List ℕ) (h_lengths : heights.length = 7) (h_sorted : heights.sorted (· > ·)) :
  ∃ n, (n ≤ 1) ∧ ∀ i, i ∈ (Finset.range 7).to_list → (heights[i] ≠ dwarf_height_claims[i] → (1 = n)) :=
  sorry

end max_truthful_dwarfs_le_one_l497_497981


namespace sum_two_digit_factors_290224_l497_497363

def is_factor (n d : ℕ) : Prop :=
  d ∣ n

def two_digit_factors (n : ℕ) : List ℕ :=
  List.filter (λ x, x ≥ 10 ∧ x < 100) (List.range n.succ).tail.filter (is_factor n)

def sum_two_digit_factors (n : ℕ) : ℕ :=
  (two_digit_factors n).sum

theorem sum_two_digit_factors_290224 : sum_two_digit_factors 290224 = 397 := sorry

end sum_two_digit_factors_290224_l497_497363


namespace minimum_fence_length_protect_cabbages_l497_497482

-- Define the side length of each triangular section
def side_length : ℝ := 50

-- Define the total additional fence length needed
def min_total_fence_length : ℝ := 650

-- Formalize the proof problem
theorem minimum_fence_length_protect_cabbages :
  ∃ (f : ℝ), f = min_total_fence_length ∧
  (∀ f', f' < f → ¬(well_enclosed f')) :=
sorry

-- Predicate to determine if the cabbages are well enclosed with a given fence length
def well_enclosed(f: ℝ) : Prop :=
  -- Implementation-specific details would go here
  sorry

end minimum_fence_length_protect_cabbages_l497_497482


namespace range_of_numbers_is_six_l497_497111

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497111


namespace range_of_set_l497_497118

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497118


namespace identify_worst_player_l497_497761

-- Define the participants
inductive Participant
| father
| sister
| son
| daughter

open Participant

-- Conditions
def participants : List Participant :=
  [father, sister, son, daughter]

def twins (p1 p2 : Participant) : Prop := 
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father) ∨
  (p1 = son ∧ p2 = daughter) ∨
  (p1 = daughter ∧ p2 = son)

def not_same_sex (p1 p2 : Participant) : Prop :=
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father) ∨
  (p1 = son ∧ p2 = daughter) ∨
  (p1 = daughter ∧ p2 = son)

def older_by_one_year (p1 p2 : Participant) : Prop :=
  (p1 = father ∧ p2 = sister) ∨
  (p1 = sister ∧ p2 = father)

-- Question: who is the worst player?
def worst_player : Participant := sister

-- Proof statement
theorem identify_worst_player
  (h_twins : ∃ p1 p2, twins p1 p2)
  (h_not_same_sex : ∀ p1 p2, twins p1 p2 → not_same_sex p1 p2)
  (h_age_diff : ∀ p1 p2, twins p1 p2 → older_by_one_year p1 p2) :
  worst_player = sister :=
sorry

end identify_worst_player_l497_497761


namespace only_solution_for_perfect_number_l497_497496

def is_perfect (m : ℕ) : Prop :=
  (∑ d in (Finset.divisors m), d) = 2 * m

theorem only_solution_for_perfect_number : ∀ n : ℕ, n > 0 → (is_perfect (n^n + 1) ↔ n = 3) :=
by
  sorry

end only_solution_for_perfect_number_l497_497496


namespace round_fraction_to_three_decimal_l497_497906

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497906


namespace range_of_numbers_is_six_l497_497106

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497106


namespace factor_polynomial_l497_497618

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497618


namespace range_of_set_of_three_numbers_l497_497128

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497128


namespace problem_solution_l497_497279

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end problem_solution_l497_497279


namespace tangent_line_at_point_l497_497388

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2

theorem tangent_line_at_point {x y : ℝ} (hx : x = 1) (hy : y = 2) :
  let m := (deriv f) 1 in y = m * x + -1 := by
    sorry

end tangent_line_at_point_l497_497388


namespace relationship_of_a_b_c_l497_497244

section
variables {f : ℝ → ℝ}

-- Defining the conditions
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g(x) = g(-x)

def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ ∈ s → x₂ ∈ s → x₁ < x₂ → f x₁ < f x₂

-- Given conditions
variables (h1 : is_even (λ x, f (x - 1)))
variables (h2 : ∀ (x₁ x₂ : ℝ), -1 < x₁ → x₁ < x₂ → [f x₂ - f x₁] * (x₂ - x₁) < 0)

-- The resulting proof goal
theorem relationship_of_a_b_c :
  let a := f (-2), b := f 1, c := f 2 in a < b ∧ b < c :=
sorry
end

end relationship_of_a_b_c_l497_497244


namespace leak_drain_time_l497_497040

theorem leak_drain_time (P L : ℝ) (h1 : P = 0.5) (h2 : (P - L) = (6 / 13)) :
    (1 / L) = 26 := by
  sorry

end leak_drain_time_l497_497040


namespace probability_of_odd_sum_is_one_third_l497_497324

def outcomes_P := {1, 2, 3}
def outcomes_Q := {2, 4, 5}
def outcomes_R := {1, 3, 5}
def outcomes_S := {2, 4, 6}

noncomputable def probability_sum_is_odd : ℚ :=
  let prob_P_odd := 1 / 3
  let prob_P_even := 2 / 3
  let prob_Q_odd := 1 / 3
  let prob_Q_even := 2 / 3
  let prob_R_odd := 1
  let prob_R_even := 0
  let prob_S_odd := 0
  let prob_S_even := 1
  -- Odd + Odd + Odd + Even
  let scenario1 := prob_P_odd * prob_Q_odd * prob_R_odd * prob_S_even
  -- Even + Even + Even + Odd
  let scenario2 := prob_P_even * prob_Q_even * prob_R_odd * prob_S_even +
                   prob_P_even * prob_Q_odd * prob_R_odd * prob_S_even
  -- Total probability for odd sum
  scenario1 + scenario2

theorem probability_of_odd_sum_is_one_third :
  probability_sum_is_odd = 1 / 3 :=
sorry

end probability_of_odd_sum_is_one_third_l497_497324


namespace problem_solution_l497_497282

variable (x y : ℝ)

theorem problem_solution
  (h1 : (x + y)^2 = 64)
  (h2 : x * y = 15) :
  (x - y)^2 = 4 := 
by
  sorry

end problem_solution_l497_497282


namespace range_of_set_l497_497102

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497102


namespace boys_play_football_l497_497763

theorem boys_play_football (total_boys basketball_players neither_players both_players : ℕ)
    (h_total : total_boys = 22)
    (h_basketball : basketball_players = 13)
    (h_neither : neither_players = 3)
    (h_both : both_players = 18) : total_boys - neither_players - both_players + (both_players - basketball_players) = 19 :=
by
  sorry

end boys_play_football_l497_497763


namespace division_rounded_l497_497853

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497853


namespace range_g_minus_x_l497_497181

def g (x : ℝ) : ℝ := 
  if -3 ≤ x ∧ x < -1 then -x - 1
  else if -1 ≤ x ∧ x ≤ 2 then x
  else if 2 < x ∧ x ≤ 3 then 5 - x
  else 0  -- This case shouldn't actually happen within our given range

noncomputable def h (x : ℝ) : ℝ := g x - x

theorem range_g_minus_x : 
  Set.range h = Set.Icc (-3) 5 :=
by
  sorry

end range_g_minus_x_l497_497181


namespace range_of_numbers_is_six_l497_497143

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497143


namespace expression_equality_l497_497175

theorem expression_equality (a b c : ℝ) : a * (a + b - c) = a^2 + a * b - a * c :=
by
  sorry

end expression_equality_l497_497175


namespace num_solutions_tan_cos_eq_cot_sin_num_solutions_tan_cos_eq_cot_sin_three_l497_497273

def tan_cos_eq_cot_sin (x : ℝ) : Prop :=
  Real.tan ((Real.pi / 2) * Real.cos x) = Real.cot ((Real.pi / 2) * Real.sin x)

theorem num_solutions_tan_cos_eq_cot_sin :
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ tan_cos_eq_cot_sin x}.finite :=
begin
  sorry,
end

theorem num_solutions_tan_cos_eq_cot_sin_three :
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ tan_cos_eq_cot_sin x}.to_finset.card = 3 :=
begin
  sorry,
end

end num_solutions_tan_cos_eq_cot_sin_num_solutions_tan_cos_eq_cot_sin_three_l497_497273


namespace evaluate_expression_l497_497560

theorem evaluate_expression : (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))) = (8 / 21) :=
by
  sorry

end evaluate_expression_l497_497560


namespace function_symmetric_about_point_l497_497393

theorem function_symmetric_about_point :
  ∃ x₀ y₀, (x₀, y₀) = (Real.pi / 3, 0) ∧ ∀ x y, y = Real.sin (2 * x + Real.pi / 3) →
    (Real.sin (2 * (2 * x₀ - x) + Real.pi / 3) = y) :=
sorry

end function_symmetric_about_point_l497_497393


namespace fraction_rounded_equals_expected_l497_497925

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497925


namespace factor_polynomial_l497_497642

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497642


namespace max_truthful_dwarfs_l497_497963

theorem max_truthful_dwarfs :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  (∃ n, n = 1 ∧ ∀ i, i < 7 → i ≠ n → heights[i] ≠ (heights[i] + 1)) :=
by
  let heights := [60, 61, 62, 63, 64, 65, 66]
  existsi 1
  intro i
  intro h
  intro hi
  sorry

end max_truthful_dwarfs_l497_497963


namespace ellipse_and_circle_equation_l497_497251

noncomputable def ellipse_eq (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
  a > b ∧ b > 0 ∧ 0 < a ∧ ∃ (x y : ℝ), (x = 0 ∧ y = sqrt 2) ∧
  ((x^2 / a^2) + (y^2 / b^2) = 1) ∧ 
  ( sqrt(1 - (b^2 / a^2)) = sqrt(2)/2)

noncomputable def circle_eq : Prop :=
∃ (k m : ℝ), 
k^2 > 0 ∧ 4 * (k^2 + 1) / 3 = m^2 ∧
((m^2 * (4 + 12*k^2 - m^2))/(1 + 2*k^2)) > 0 ∧
(m/sqrt(1 + k^2) = 2/sqrt(3)) ∧ 
( ∀ θ radius, radius = sqrt(4 / 3) → ∘ = 0)


theorem ellipse_and_circle_equation : Prop :=
∃ a b : ℝ, ellipse_eq a b True True ∧ circle_eq

#print ellipse_and_circle_equation

end ellipse_and_circle_equation_l497_497251


namespace find_third_median_l497_497399

-- Define the context of the triangle with specific properties
variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variables (triangle ∆ : A × B × C)
variables (side_length : ℝ)
variables (median1_length : ℝ)
variables (median2_length : ℝ)
variables (third_median_length : ℝ)

-- Define specific conditions for the problem
variables (side_length_10 : side_length = 10)
variables (median1_length_9 : median1_length = 9)
variables (median1_perpendicular_median2 : ⊥ median1 → ⊥ median2)

-- Statement of the proof problem
theorem find_third_median (h1 : side_length_10) (h2 : median1_length_9) (h3 : median1_perpendicular_median2) :
  third_median_length = 3 * real.sqrt 13 :=
sorry

end find_third_median_l497_497399


namespace bags_needed_l497_497415

theorem bags_needed (total_sand : ℝ) (bag_capacity : ℝ) : ℕ :=
if H : total_sand / bag_capacity = ⌊total_sand / bag_capacity⌋.to_real then
  ⌊total_sand / bag_capacity⌋.nat_abs
else
  ⌊total_sand / bag_capacity⌋.nat_abs + 1

example : bags_needed 15723.56 785.25 = 21 := by
  sorry

end bags_needed_l497_497415


namespace DMV_waiting_times_l497_497364

theorem DMV_waiting_times :
  let w := 25 in
  let license_wait := 3 * w + 12 in
  let registration_wait := 1.5 * w + 14 in
  let record_wait := 2 * w + 10 in
  let total_wait := license_wait + registration_wait + record_wait in
  license_wait = 87 ∧
  registration_wait = 52 ∧
  record_wait = 60 ∧
  total_wait = 199 ∧
  license_wait >= registration_wait ∧
  license_wait >= record_wait :=
by
  sorry

end DMV_waiting_times_l497_497364


namespace trajectory_of_midpoint_l497_497229

open Real

theorem trajectory_of_midpoint (A : ℝ × ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A = (-2, 0))
    (hP_on_curve : P.1 = 2 * P.2 ^ 2)
    (hM_midpoint : M = ((A.1 + P.1) / 2, (A.2 + P.2) / 2)) :
    M.1 = 4 * M.2 ^ 2 - 1 :=
sorry

end trajectory_of_midpoint_l497_497229


namespace largest_inscribed_triangle_area_l497_497178

-- Definition of the conditions
def radius : ℝ := 10
def diameter : ℝ := 2 * radius

-- The theorem to be proven
theorem largest_inscribed_triangle_area (r : ℝ) (D : ℝ) (h : D = 2 * r) : 
  ∃ (A : ℝ), A = 100 := by
  have base := D
  have height := r
  have area := (1 / 2) * base * height
  use area
  sorry

end largest_inscribed_triangle_area_l497_497178


namespace part1_part2_l497_497683

noncomputable def f (x a : ℝ) : ℝ := (4^x / (4^x + 2)) + a

-- Given condition
axiom h_f_lg2_lg5 : f (log 2) a + f (log 5) a = 3

-- Prove: the value of a
theorem part1 (a : ℝ) : a = 1 := sorry

-- Prove: the range of m
theorem part2 (a m : ℝ) (h1 : a = 1) (h2 : ∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), f x a ≥ 4^x + m) : m ≤ -7 / 3 := sorry

end part1_part2_l497_497683


namespace intersection_eq_l497_497753

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x < 2}

theorem intersection_eq : M ∩ N = {-1, 0, 1} :=
by
  sorry

end intersection_eq_l497_497753


namespace car_order_correct_l497_497367

noncomputable def car_order := ["Yellow truck", "Blue passenger car", "Green van", "Red bus"]

structure CarArrangement :=
  (positions : list string)
  (unique_colors : ∀ c, c ∈ ["Red", "Blue", "Yellow", "Green"] → ∃! p, p ∈ positions ∧ c ∈ p)
  (unique_types : ∀ t, t ∈ ["sedan", "bus", "truck", "van"] → ∃! p, p ∈ positions ∧ t ∈ p)
  (blue_between_red_and_green : ∃ left middle right,
    left ∈ positions ∧ middle ∈ positions ∧ right ∈ positions ∧
    "Red" ∈ left ∧ "Blue" ∈ middle ∧ "Green" ∈ right ∧ positions.indexOf middle = positions.indexOf left + 1 ∧ positions.indexOf right = positions.indexOf middle + 1)
  (van_right_of_yellow : ∃ y v, y ∈ positions ∧ v ∈ positions ∧ "Yellow" ∈ y ∧ "van" ∈ v ∧ positions.indexOf v > positions.indexOf y)
  (bus_right_of_sedan_and_van : ∃ s v b, s ∈ positions ∧ v ∈ positions ∧ b ∈ positions ∧
    "sedan" ∈ s ∧ "van" ∈ v ∧ "bus" ∈ b ∧ positions.indexOf b > positions.indexOf s ∧ positions.indexOf b > positions.indexOf v)
  (sedan_not_at_edge : ∃ s, s ∈ positions ∧ "sedan" ∈ s ∧ ∃! p, p ∈ ["2nd", "3rd"] ∧ s = p)
  (red_not_next_to_yellow : ∀ p1 p2, p1 ∈ positions ∧ p2 ∈ positions → "Red" ∈ p1 ∧ "Yellow" ∈ p2 → abs(positions.indexOf p1 - positions.indexOf p2) ≠ 1)

theorem car_order_correct (arr : CarArrangement) : arr.positions = car_order :=
  sorry

end car_order_correct_l497_497367


namespace line_passes_through_fixed_point_l497_497396

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k + 1) * (-1) - (2 * k - 1) * (1) + 3 * k = 0 :=
by
  intro k
  sorry

end line_passes_through_fixed_point_l497_497396


namespace problem_statement_l497_497347

def f (x : ℕ) : ℕ := x^2 + x + 4
def g (x : ℕ) : ℕ := 3 * x^3 + 2

theorem problem_statement : g (f 3) = 12290 := by
  sorry

end problem_statement_l497_497347


namespace arrangement_count_l497_497470

-- Definitions from the conditions
def people : Nat := 5
def valid_positions_for_A : Finset Nat := Finset.range 5 \ {0, 4}

-- The theorem that states the question equals the correct answer given the conditions
theorem arrangement_count (A_positions : Finset Nat := valid_positions_for_A) : 
  ∃ (total_arrangements : Nat), total_arrangements = 72 :=
by
  -- Placeholder for the proof
  sorry

end arrangement_count_l497_497470


namespace slope_of_l_through_origin_is_13_l497_497350

noncomputable def slope_of_tangent_line : ℝ :=
  let t : ℝ := 2 in
  3 * t^2 + 1

theorem slope_of_l_through_origin_is_13 :
  let l : (ℝ × ℝ) → ℝ := λ p, (3 * p.1^2 + 1) in
  let curve : ℝ → ℝ := λ x, x^3 + x + 16 in
  let tangent_points : ℝ := 2 in
  l (0, 0) = 13 :=
by
  let t := tangent_points
  have h_slope_t := 3 * t^2 + 1
  rw h_slope_t
  norm_num
  sorry

end slope_of_l_through_origin_is_13_l497_497350


namespace increasing_function_range_of_a_l497_497712

def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 1 then (2 - a) * x + 2 else 2^x - 5 * a

theorem increasing_function_range_of_a :
  (∀ (x y : ℝ), x < y → f x a ≤ f y a) ↔ -1/2 ≤ a ∧ a < 2 :=
sorry

end increasing_function_range_of_a_l497_497712


namespace factorization_correct_l497_497578

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497578


namespace population_std_dev_l497_497436

theorem population_std_dev (x : ℝ) (h_avg : (8 + x + 10 + 11 + 9) / 5 = 10) : 
  let variance := (1 / 5) * ((8 - 10)^2 + (x - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2)
  in sqrt variance = sqrt 2 :=
by
  sorry

end population_std_dev_l497_497436


namespace circumcenter_under_homothety_l497_497427

open EuclideanGeometry

def similar_triangles (A B C : Point) (A0 B0 C0 : Point) : Prop :=
∃ (f : Point → Point),
  (∀ P Q R, (A0 = f(A) ∧ B0 = f(B) ∧ C0 = f(C)) ↔ similar (P, Q, R) (A0, B0, C0))

def circumcenter (A B C : Point) : Point :=
sorry -- Placeholder for the actual definition of circumcenter

theorem circumcenter_under_homothety (A B C A0 B0 C0 : Point)
  (h1 : ∃ (f : Point → Point), homothety f A0 B0 C0 A B C)
  (h : similar_triangles A B C A0 B0 C0) :
  ∃ (f : Point → Point), circumcenter A B C = f (circumcenter A0 B0 C0) := sorry

end circumcenter_under_homothety_l497_497427


namespace fraction_rounding_l497_497884

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497884


namespace range_of_set_of_three_numbers_l497_497137

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497137


namespace part1_part2_l497_497257

def f (x : ℝ) : ℝ := (1 + x) / Real.exp x
def g (x : ℝ) (a : ℝ) : ℝ := 1 - a * x^2

theorem part1 (a : ℝ) : 
  ∀ x, (1 + x) / Real.exp x = 1 - a * x^2 -> a = 1 / (2 * Real.exp 1) :=
by
  sorry

theorem part2 (a : ℝ) :
  ∀ x ∈ Set.Icc 0 1, f x ≤ g x a → a ≤ (Real.exp 1 - 3) / Real.exp 1 :=
by
  sorry

end part1_part2_l497_497257


namespace probability_not_next_to_each_other_l497_497413

theorem probability_not_next_to_each_other :
  let num_chairs := 10
  let valid_chairs := num_chairs - 1
  let total_ways := Nat.choose valid_chairs 2
  let next_to_ways := valid_chairs - 1
  let p_next_to := next_to_ways.toReal / total_ways.toReal
  let p_not_next_to := 1 - p_next_to
  p_not_next_to = (7 : ℚ) / 9 := 
by {
  -- Definitions for conditions
  let num_chairs := 10
  let valid_chairs := num_chairs - 1
  let total_ways := Nat.choose valid_chairs 2
  let next_to_ways := valid_chairs - 1
  let p_next_to := next_to_ways.toReal / total_ways.toReal

  -- Calculations
  have h1 : total_ways = 36 := by sorry
  have h2 : next_to_ways = 8 := by sorry
  have h3 : p_next_to = (2 : ℚ) / 9 := by sorry
  have h4 : p_not_next_to = 1 - (2 : ℚ) / 9 := by sorry

  -- Conclusion
  show p_not_next_to = (7 : ℚ) / 9 from by sorry
}  

end probability_not_next_to_each_other_l497_497413


namespace distribute_a_eq_l497_497173

variable (a b c : ℝ)

theorem distribute_a_eq : a * (a + b - c) = a^2 + a * b - a * c := 
sorry

end distribute_a_eq_l497_497173


namespace locus_of_centers_of_tangent_circles_l497_497656

variables {R r : ℝ} -- Define R and r as real numbers

-- Define the geometric locus problem
theorem locus_of_centers_of_tangent_circles (R r : ℝ) :
  ∃ C₁ C₂ : set (ℝ × ℝ), 
  (∀ p : ℝ × ℝ, p ∈ C₁ ↔ dist (0,0) p = R + r) ∧
  (∀ p : ℝ × ℝ, p ∈ C₂ ↔ dist (0,0) p = R - r) :=
sorry

end locus_of_centers_of_tangent_circles_l497_497656


namespace function_increasing_on_interval_l497_497391

noncomputable def f : ℝ → ℝ := λ x, 1 + x - sin x

theorem function_increasing_on_interval :
  ∀ x ∈ (Ioo 0 (2 * real.pi)), deriv f x > 0 :=
begin
  intros x hx,
  have h_deriv : deriv f x = 1 - cos x,
  { sorry }, -- Derivation steps here
  have h_cos : -1 ≤ cos x ∧ cos x < 1,
  { sorry }, -- Range analysis here
  linarith,
end

end function_increasing_on_interval_l497_497391


namespace largest_possible_red_socks_l497_497023

theorem largest_possible_red_socks (r b : ℕ) (h1 : 0 < r) (h2 : 0 < b)
  (h3 : r + b ≤ 2500) (h4 : r > b) :
  r * (r - 1) + b * (b - 1) = 3/5 * (r + b) * (r + b - 1) → r ≤ 1164 :=
by sorry

end largest_possible_red_socks_l497_497023


namespace find_x_value_l497_497287

-- Let's define the conditions
def equation (x y : ℝ) : Prop := x^2 - 4 * x + y = 0
def y_value : ℝ := 4

-- Define the theorem which states that x = 2 satisfies the conditions
theorem find_x_value (x : ℝ) (h : equation x y_value) : x = 2 :=
by
  sorry

end find_x_value_l497_497287


namespace fraction_rounding_l497_497888

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497888


namespace fraction_rounded_equals_expected_l497_497924

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497924


namespace area_triangle_CEF_l497_497528

-- Condition definitions:
constant A B C D E F : Type
constants (BD CD : ℕ) (length_eq : BD = 8) (width_eq : CD = 6)
constant on_segment : E ∈ segment B C
constant fold : B → F = ref_on_AC A E B

-- Question formalization:
theorem area_triangle_CEF : area (triangle C E F) = 12 := 
sorry

end area_triangle_CEF_l497_497528


namespace problem_statement_l497_497730

theorem problem_statement (x y m a b : ℝ) :
  (x - y ≤ 0) → (x - 2y + 2 ≥ 0) → (x ≥ m) → 
  f(x) = (1/3)*x^3 - (a-1)*x^2 + b^2*x :=
sorry

end problem_statement_l497_497730


namespace mixing_ratios_indeterminacy_l497_497003

theorem mixing_ratios_indeterminacy (x : ℝ) (a b : ℝ) (h1 : a + b = 50) (h2 : 0.40 * a + (x / 100) * b = 25) : False :=
sorry

end mixing_ratios_indeterminacy_l497_497003


namespace square_area_16_l497_497154

-- Define the vertices with their y-coordinates as given by the problem
def is_square (A B C D : ℝ × ℝ) : Prop :=
  A.2 = 2 ∧ B.2 = 3 ∧ C.2 = 7 ∧ D.2 = 6

-- Define the property to be proved, i.e., the area of this square
def square_area (A B C D : ℝ × ℝ) : ℝ :=
  let side_length := (B - A).dist in side_length ^ 2

-- Prove that given the specific y-coordinates, the area is 16
theorem square_area_16 (A B C D : ℝ × ℝ) (h : is_square A B C D) : square_area A B C D = 16 :=
by sorry

end square_area_16_l497_497154


namespace sum_outer_equal_sum_inner_l497_497816

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

theorem sum_outer_equal_sum_inner (M N : ℕ) (a b c d : ℕ) 
  (h1 : is_four_digit M)
  (h2 : M = 1000 * a + 100 * b + 10 * c + d) 
  (h3 : N = reverse_digits M) 
  (h4 : M + N % 101 = 0) 
  (h5 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) : 
  a + d = b + c :=
  sorry

end sum_outer_equal_sum_inner_l497_497816


namespace quad_eq1_solution_quad_eq2_solution_quad_eq3_solution_l497_497381

theorem quad_eq1_solution (x : ℝ) : x^2 - 4 * x + 1 = 0 → x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by
  sorry

theorem quad_eq2_solution (x : ℝ) : 2 * x^2 - 7 * x + 5 = 0 → x = 5 / 2 ∨ x = 1 :=
by
  sorry

theorem quad_eq3_solution (x : ℝ) : (x + 3)^2 - 2 * (x + 3) = 0 → x = -3 ∨ x = -1 :=
by
  sorry

end quad_eq1_solution_quad_eq2_solution_quad_eq3_solution_l497_497381


namespace find_x_l497_497204

noncomputable def is_solution (x : ℝ) : Prop :=
   (⌊x * ⌊x⌋⌋ = 29)

theorem find_x (x : ℝ) (h : is_solution x) : 5.8 ≤ x ∧ x < 6 :=
sorry

end find_x_l497_497204


namespace range_of_numbers_is_six_l497_497107

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497107


namespace min_value_fraction_l497_497238

theorem min_value_fraction (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 9^a * 27^b = 9) : 
  ∃ c : ℝ, (c = 12) ∧ (∀ x, x = (3/a + 2/b) -> x >= c) :=
begin
  sorry
end

end min_value_fraction_l497_497238


namespace diagonal_of_quadrilateral_is_correct_l497_497152

noncomputable def diagonal_of_resulting_quadrilateral (s : ℝ) (a : ℝ) : ℝ :=
  let side_length := s
  let area_triangle := a
  let leg_length := real.sqrt (2 * area_triangle)
  real.sqrt (2 * leg_length^2)

theorem diagonal_of_quadrilateral_is_correct :
  diagonal_of_resulting_quadrilateral 20 50 = 10 * real.sqrt 2 :=
by
  sorry

end diagonal_of_quadrilateral_is_correct_l497_497152


namespace complex_number_solution_l497_497549

-- Define complex numbers and their properties
variable {z : Complex}

-- The main theorem to be proved
theorem complex_number_solution (z : Complex) (h : 3 * z - 4 * Complex.i * Complex.conj z = -8 + 6 * Complex.i) :
  z = 2 * Complex.i :=
by sorry

end complex_number_solution_l497_497549


namespace find_value_of_c_l497_497302

def triangle_value_c (a b area c : ℝ) : Prop :=
  (a = 1) ∧ (b = sqrt 7) ∧ (area = sqrt 3 / 2) ∧ 
  ((c = 2) ∨ (c = 2 * sqrt 3))

theorem find_value_of_c :
  ∃ (c : ℝ), triangle_value_c 1 (sqrt 7) (sqrt 3 / 2) c :=
sorry

end find_value_of_c_l497_497302


namespace expression_equality_l497_497174

theorem expression_equality (a b c : ℝ) : a * (a + b - c) = a^2 + a * b - a * c :=
by
  sorry

end expression_equality_l497_497174


namespace centroid_of_homogeneous_plate_l497_497205

structure Rectangle :=
  (w : ℝ) (h : ℝ)

def center (r : Rectangle) : (ℝ × ℝ) := 
  (r.w / 2, r.h / 2)

def combined_centroid (r1 r2 : Rectangle) : (ℝ × ℝ) :=
  let c1 := center r1
  let c2 := (r2.w / 2, r1.h + r2.h / 2)
  ((c1.1 + c2.1) / 2, (c1.2 + c2.2) / 2)

noncomputable def centroid := sorry

theorem centroid_of_homogeneous_plate (r1 r2 : Rectangle) : 
  centroid r1 r2 = combined_centroid r1 r2 :=
  sorry

end centroid_of_homogeneous_plate_l497_497205


namespace geometric_sequence_log_l497_497762

theorem geometric_sequence_log (a : ℕ → ℝ) (a1 : ℝ)
    (r : ℝ) (log_base_a1 : ℝ) :
  (∀ n, a n = a1 * r^(n-1)) →
  a 9 = 13 →
  a 13 = 1 →
  log_base_a1 = Real.log 13 / Real.log (a1) →
  log_base_a1 = 1/3 :=
by 
  intros h geom_seq a_9_eq_13 a_13_eq_1 log_base_def
  sorry

end geometric_sequence_log_l497_497762


namespace correct_transformation_l497_497001

theorem correct_transformation (a b : ℝ) (h : a ≠ 0) : (b / a = (a * b) / (a ^ 2)) :=
begin
  sorry
end

end correct_transformation_l497_497001


namespace distance_between_A_and_B_l497_497449

theorem distance_between_A_and_B :
  (let speed_xiaodong := 50
       speed_xiaorong := 60
       time := 10
       total_distance := (speed_xiaodong + speed_xiaorong) * time
       distance_A_B := total_distance / 2
   in distance_A_B) = 550 :=
by
  sorry

end distance_between_A_and_B_l497_497449


namespace dwarfs_truth_claims_l497_497965

/-- Seven dwarfs lined up by height, starting with the tallest.
Each dwarf claims a specific height as follows.
Verify that the largest number of dwarfs telling the truth is exactly 1. -/
theorem dwarfs_truth_claims :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  ∀ (actual_height: Fin 7 → ℕ), 
    (∀ i, actual_height i > actual_height (Fin.mk i (by linarith))) →
    ∀ claim_truth: Fin 7 → Prop, 
    (∀ i, claim_truth i ↔ actual_height i = heights[i]) →
    (∃ k, ∀ j, claim_truth j → j = k) :=
by
  intro heights actual_height h_order claim_truth h_claim_match
  existsi Fin.mk 0 (by linarith) -- Claim that only one can be telling the truth (arbitrary choice)
  sorry  -- Proof steps are omitted

end dwarfs_truth_claims_l497_497965


namespace common_ratio_is_2_or_neg3_l497_497688

open_locale big_operators

noncomputable def common_ratio_of_geometric_sequence (a q : ℝ) : Prop :=
  let S3 := a * (1 + q + q^2) in
  S3 = 7 * a

theorem common_ratio_is_2_or_neg3 (a q : ℝ) (h : common_ratio_of_geometric_sequence a q) : q = 2 ∨ q = -3 :=
by {
  simp [common_ratio_of_geometric_sequence] at h, 
  have : a * (1 + q + q^2) = 7 * a := h,
  sorry
}

end common_ratio_is_2_or_neg3_l497_497688


namespace max_truthful_gnomes_l497_497999

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

end max_truthful_gnomes_l497_497999


namespace polynomial_factorization_l497_497561

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497561


namespace division_rounded_l497_497848

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497848


namespace factor_polynomial_l497_497623

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497623


namespace trigonometric_propositions_l497_497254

theorem trigonometric_propositions :
  (∀ x, sin (5 * Real.pi / 2 - 2 * x) = cos (2 * x) ∧
  ∀ x ∈ Set.Icc(-Real.pi / 2, Real.pi / 2), ¬MonotoneOn (λ x, sin (x + Real.pi / 4)) (Set.Icc (-Real.pi / 2) (Real.pi / 2)) ∧
  (2 * Real.pi / 8 + 5 * Real.pi / 4) = π → sin (2 * π / 8 + 5 * π / 4) = -1 → x = Real.pi / 8) ∧ 
  ( Cosine (2x -Real.pi /3).Translating Left byReal.pi /3 units = Cosine(2x)):
  {
satisfy sorry,
}

end trigonometric_propositions_l497_497254


namespace sandy_initial_puppies_l497_497376

theorem sandy_initial_puppies (x : ℕ) (h3_spots: 3 > 0) (h_given: 4 > 0) (h_left: 4 > 0):
  x - 4 = 4 → x = 8 :=
by
  intro h
  rw [← h]
  sorry

end sandy_initial_puppies_l497_497376


namespace MeganMarkers_l497_497360

def initialMarkers : Nat := 217
def additionalMarkers : Nat := 109
def totalMarkers : Nat := initialMarkers + additionalMarkers

theorem MeganMarkers : totalMarkers = 326 := by
    sorry

end MeganMarkers_l497_497360


namespace largest_integer_is_59_l497_497193

theorem largest_integer_is_59 
  {w x y z : ℤ} 
  (h₁ : (w + x + y) / 3 = 32)
  (h₂ : (w + x + z) / 3 = 39)
  (h₃ : (w + y + z) / 3 = 40)
  (h₄ : (x + y + z) / 3 = 44) :
  max (max w x) (max y z) = 59 :=
by {
  sorry
}

end largest_integer_is_59_l497_497193


namespace emily_journey_l497_497542

noncomputable def travel_ways : ℕ := 6

theorem emily_journey :
  ∃ (n : ℕ), n = travel_ways ∧
    (∀ (roads : ℕ), roads = 20 →
      ∃ (cities : ℕ → ℕ → Prop),
        (∃ (B M G : ℕ), 
          B ≠ M ∧ M ≠ G ∧ B ≠ G ∧
          (∃ (path : list ℕ),
            path.head = B ∧ path.last = M ∧
            (∀ r ∈ path.zip path.tail, roads r) ∧
            path.count G ≥ 1 ∧
            path.nodup = tt ∧ path.length = 16))) :=
sorry

end emily_journey_l497_497542


namespace problem_statement_l497_497793

-- Definitions reflecting the conditions in the problem statement.
structure Triangle (α : Type) :=
(a b c : α) -- sides of the triangle

noncomputable def semiperimeter {α : Type} [Add α] [Div α] (T : Triangle α) : α := 
  (T.a + T.b + T.c) / 2

noncomputable def inradius {α : Type} [Add α] [Div α] (T : Triangle α) (A : α) : α := 
  sorry -- to be defined based on some area A of the triangle

def circle_tangent_to_semicircles {α : Type} [LT α] [LE α] [Add α] (s r t : α) : Prop :=
  (s / 2 < t) ∧ (t ≤ s / 2 + (1 - sqrt 3 / 2) * r)

-- Triangle ABC with inradius r, semiperimeter s, and a circle t tangent to the semicircles of diameters BC, CA, and AB
theorem problem_statement {α : Type} [LinearOrderedField α] --
  (a b c : α) (r t : α):
  let T := Triangle.mk a b c in
  let s := semiperimeter T in
  circle_tangent_to_semicircles s r t := -- Proving this relationship
sorry

end problem_statement_l497_497793


namespace complex_number_line_condition_l497_497339

theorem complex_number_line_condition (a : ℝ) (i : ℂ) (h_i : i = complex.I) :
  ((a / i) + (1 - i) / 2).re + ((a / i) + (1 - i) / 2).im = 0 → a = 0 :=
by sorry

end complex_number_line_condition_l497_497339


namespace parabola_problem_l497_497670

noncomputable def point :=
  (ℝ × ℝ)

def parabola (p : ℝ) (h : p > 0) :=
  { P : point | P.2^2 = 2 * p * P.1 }

def focus (p : ℝ) : point :=
  (p / 2, 0)

def point_P (y1 y2 p : ℝ) (h : p > 0) : point :=
  (y1 * y2 / (2 * p), (y1 + y2) / 2)

def point_M_N (x y p : ℝ) (h : p > 0) : point :=
  (x, y)

def distance (A B : point) : ℝ :=
  ( (A.1 - B.1)^2 + (A.2 - B.2)^2 )^0.5

def distance_squared (A B : point) : ℝ :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2

theorem parabola_problem
    (p : ℝ) (hp : p > 0)
    (y1 y2 : ℝ)
    (P : point := point_P y1 y2 p hp)
    (M : point := point_M_N (y1^2 / (2 * p)) y1 p hp)
    (N : point := point_M_N (y2^2 / (2 * p)) y2 p hp)
    (F : point := focus p) :
  distance_squared P F = distance_squared M F * distance_squared N F ∧
  ∠PMF = ∠FPN := 
sorry

end parabola_problem_l497_497670


namespace sum_of_leading_digits_l497_497335

-- Definition of M which is a 400-digit number with each digit 8
def M : ℕ := 10^399 * 8 + (10^399 - 1) / 9 * 8

-- Definition of g(r) as the leading digit of the r-th root of M
def leading_digit (r : ℕ) : ℕ :=
  Nat.digits 10 (Real.floor (Real.root r (M : ℝ))) |> List.head? |>.getD 0

-- Mathematical equivalent proof problem statement
theorem sum_of_leading_digits : 
  leading_digit 2 + leading_digit 3 + leading_digit 4 + leading_digit 5 + leading_digit 6 + leading_digit 7 + leading_digit 8 = 14 :=
by
  sorry -- Proof goes here

end sum_of_leading_digits_l497_497335


namespace polynomial_factorization_l497_497565

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497565


namespace rectangle_triangle_height_l497_497231

theorem rectangle_triangle_height (l : ℝ) (h : ℝ) (w : ℝ) (d : ℝ) 
  (hw : w = Real.sqrt 2 * l)
  (hd : d = Real.sqrt (l^2 + w^2))
  (A_triangle : (1 / 2) * d * h = l * w) :
  h = (2 * l * Real.sqrt 6) / 3 := by
  sorry

end rectangle_triangle_height_l497_497231


namespace fraction_rounded_to_decimal_l497_497933

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497933


namespace transformation_correct_l497_497747

variables {x y : ℝ}

theorem transformation_correct (h : x = y) : x - 2 = y - 2 := by
  sorry

end transformation_correct_l497_497747


namespace round_8_div_11_l497_497876

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497876


namespace incorrect_statements_l497_497002

open Function

theorem incorrect_statements (a : ℝ) (x y x₁ y₁ x₂ y₂ k : ℝ) : 
  ¬ (a = -1 ↔ (∀ x y, a^2 * x - y + 1 = 0 ∧ x - a * y - 2 = 0 → (a = -1 ∨ a = 0))) ∧ 
  ¬ (∀ x y (x₁ y₁ x₂ y₂ : ℝ), (∃ (m : ℝ), (y - y₁) = m * (x - x₁) ∧ (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)) → 
    ((y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁))) :=
sorry

end incorrect_statements_l497_497002


namespace max_value_of_x_plus_2y_l497_497724

theorem max_value_of_x_plus_2y (x y : ℝ) (h : 2 ^ x + 4 ^ y = 1) : x + 2 * y ≤ -2 :=
by sorry

end max_value_of_x_plus_2y_l497_497724


namespace greatest_integer_half_solution_l497_497806

theorem greatest_integer_half_solution :
  let n := 200th_smallest_solution \ (x - (π / 2) = tan x) in
  ⌊n / 2⌋ = 314 :=
by
  sorry

end greatest_integer_half_solution_l497_497806


namespace lisa_total_eggs_l497_497358

def eggs_per_day (c : ℕ) (h : ℕ) (l : ℕ) (mult : ℕ → ℕ) : ℕ :=
  (c * mult 0) + h + l

def monday_eggs := eggs_per_day 3 3 2 (λ i, if i = 0 then 2 else 3)
def tuesday_eggs := eggs_per_day 3 3 2 (λ i, if i = 0 then 2 else if i = 1 then 4 else 3)
def wednesday_eggs := eggs_per_day 4 4 3 (λ i, if i = 0 then 3 else 3)
def thursday_eggs := eggs_per_day 3 2 1 (λ i, if i = 0 then 1 else 2)
def friday_eggs := eggs_per_day 4 3 2 (λ i, if i = 0 then 2 else 3)
def holiday_eggs := eggs_per_day 4 2 2 (λ i, 2)

def total_eggs_per_year : ℕ :=
  (monday_eggs * 52) + (tuesday_eggs * 52) + (wednesday_eggs * 52) +
  (thursday_eggs * 52) + (friday_eggs * 52) + (holiday_eggs * 8)

theorem lisa_total_eggs : total_eggs_per_year = 3320 :=
  by { 
    -- the proof process is omitted as per the instructions
    sorry
  }

end lisa_total_eggs_l497_497358


namespace factorization_identity_l497_497609

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497609


namespace plants_under_lamps_l497_497840

-- Definitions based on the conditions
def rachel_plants := {cacti : ℕ // cacti = 3} × {rose : ℕ // rose = 1}
def rachel_lamps := {yellow : ℕ // yellow = 3} × {blue : ℕ // blue = 2}

-- The theorem stating the number of ways to distribute the plants under the lamps
theorem plants_under_lamps :
  let plants : rachel_plants := ⟨3, rfl⟩ in
  let lamps : rachel_lamps := ⟨3, rfl⟩ × ⟨2, rfl⟩ in
  (number_of_distribution_ways plants lamps) = 22 :=
sorry  -- Proof is omitted

end plants_under_lamps_l497_497840


namespace rounded_to_three_decimal_places_l497_497898

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497898


namespace factor_polynomial_l497_497644

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497644


namespace repeating_decimals_count_l497_497217

theorem repeating_decimals_count :
  ∃ (count : ℕ), count = 18 ∧
  ∀ n ∈ finset.range (21), -- Since we need to go up to 20 inclusive
    (¬ (∃ k, n = 9 * k) ↔ (n / 18 ∉ finset.range (2))) :=
by
  sorry

end repeating_decimals_count_l497_497217


namespace unaccounted_bottles_l497_497545

theorem unaccounted_bottles :
  let total_bottles := 254
  let football_bottles := 11 * 6
  let soccer_bottles := 53
  let lacrosse_bottles := football_bottles + 12
  let rugby_bottles := 49
  let team_bottles := football_bottles + soccer_bottles + lacrosse_bottles + rugby_bottles
  total_bottles - team_bottles = 8 :=
by
  rfl

end unaccounted_bottles_l497_497545


namespace factor_polynomial_l497_497643

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497643


namespace new_supervisor_salary_l497_497460

-- Given conditions
def avg_salary_old (W S1 : ℝ) (n : ℝ := 9) := (W + S1) / n
def avg_salary_new (W S2 : ℝ) (n : ℝ := 9) := (W + S2) / n

theorem new_supervisor_salary
  (W : ℝ) (S1 : ℝ) (S2 : ℝ) 
  (H1 : (W + S1) / 9 = 430) 
  (H2 : S1 = 870)
  (H3 : (W + S2) / 9 = 430) :
  S2 = 870 :=
by
  sorry

end new_supervisor_salary_l497_497460


namespace range_of_set_is_six_l497_497080

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497080


namespace minimum_value_of_a_squared_plus_b_squared_l497_497261

def quadratic (a b x : ℝ) : ℝ := a * x^2 + (2 * b + 1) * x - a - 2

theorem minimum_value_of_a_squared_plus_b_squared (a b : ℝ) (hab : a ≠ 0)
  (hroot : ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 4 ∧ quadratic a b x = 0) :
  a^2 + b^2 = 1 / 100 :=
sorry

end minimum_value_of_a_squared_plus_b_squared_l497_497261


namespace gnomes_telling_the_truth_l497_497941

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l497_497941


namespace largest_angle_correct_l497_497398

-- Define the conditions of the problem
def measure_angle_1 := 60
def measure_angle_2 := 70
def measure_angle_3 := 15 + measure_angle_2

-- Sum of interior angles in a triangle
def sum_of_angles := measure_angle_1 + measure_angle_2 + measure_angle_3

-- The proof problem statement: verifying if the largest angle degree measure is 85
theorem largest_angle_correct :
  measure_angle_3 = 85 :=
by
  unfold measure_angle_3
  unfold measure_angle_2
  rw [Nat.add_comm, Nat.add_assoc]
  exact rfl

end largest_angle_correct_l497_497398


namespace factor_polynomial_l497_497624

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497624


namespace inequality_sqrt_sum_l497_497693

theorem inequality_sqrt_sum (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
  sqrt (4 * a + 1) + sqrt (4 * b + 1) + sqrt (4 * c + 1) > 2 + sqrt 5 := 
begin
  sorry
end

end inequality_sqrt_sum_l497_497693


namespace lines_parallel_if_perpendicular_to_same_plane_l497_497831

theorem lines_parallel_if_perpendicular_to_same_plane
  (a b : ℝ) (P : set ℝ) : perpendicular a P ∧ perpendicular b P → parallel a b :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l497_497831


namespace max_divisors_in_range_1_to_20_l497_497182

def count_divisors (n : ℕ) : ℕ := (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem max_divisors_in_range_1_to_20 :
  ∃ (n : ℕ), n ∈ {12, 18, 20} ∧ ∀ m ∈ finset.range 21, count_divisors m ≤ 6 ∧ (count_divisors m = 6 → m ∈ {12, 18, 20}) :=
by
  sorry

end max_divisors_in_range_1_to_20_l497_497182


namespace find_y_value_l497_497749

theorem find_y_value (k : ℝ) (x y : ℝ) (h1 : y = k * x^(1/5)) (h2 : y = 4) (h3 : x = 32) :
  y = 6 := by
  sorry

end find_y_value_l497_497749


namespace correct_transformation_l497_497000

theorem correct_transformation (a b : ℝ) (h : a ≠ 0) : (b / a = (a * b) / (a ^ 2)) :=
begin
  sorry
end

end correct_transformation_l497_497000


namespace coefficient_x2_expansion_l497_497385

def binomial_coefficient (n k : ℕ) := Nat.choose n k

def expansion_term (n r : ℕ) (x : ℕ) : ℤ :=
  binomial_coefficient n r * (x ^ (n - r)) * ((-2) ^ r)

def coefficient_of_x_squared (expr : ℕ → ℤ) : ℤ :=
  expr 2

theorem coefficient_x2_expansion :
  coefficient_of_x_squared (expansion_term 3) = 18 :=
  sorry

end coefficient_x2_expansion_l497_497385


namespace Nils_has_300_geese_l497_497365

variables (A x k n : ℕ)

def condition1 (A x k n : ℕ) : Prop :=
  A = k * x * n

def condition2 (A x k n : ℕ) : Prop :=
  A = (k + 20) * x * (n - 50)

def condition3 (A x k n : ℕ) : Prop :=
  A = (k - 10) * x * (n + 100)

theorem Nils_has_300_geese (A x k n : ℕ) :
  condition1 A x k n →
  condition2 A x k n →
  condition3 A x k n →
  n = 300 :=
by
  intros h1 h2 h3
  sorry

end Nils_has_300_geese_l497_497365


namespace sqrt_x_roots_l497_497298

theorem sqrt_x_roots (a x : ℝ) (h1 : sqrt x = 3 * a - 4) (h2 : sqrt x = 1 - 6 * a) : a = -1 ∧ x = 49 :=
by
  sorry

end sqrt_x_roots_l497_497298


namespace candle_height_problem_l497_497431

/-- Define the height functions of the two candles. -/
def h1 (t : ℚ) : ℚ := 1 - t / 5
def h2 (t : ℚ) : ℚ := 1 - t / 4

/-- The main theorem stating the time t when the first candle is three times the height of the second candle. -/
theorem candle_height_problem : 
  (∀ t : ℚ, h1 t = 3 * h2 t) → t = (40 : ℚ) / 11 :=
by
  sorry

end candle_height_problem_l497_497431


namespace factorization_correct_l497_497573

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497573


namespace factor_poly_eq_factored_form_l497_497594

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497594


namespace least_sum_of_exponents_l497_497009

theorem least_sum_of_exponents (a b c d e : ℕ) (h : ℕ) (h_divisors : 225 ∣ h ∧ 216 ∣ h ∧ 847 ∣ h)
  (h_form : h = (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) * (11 ^ e)) : 
  a + b + c + d + e = 10 :=
sorry

end least_sum_of_exponents_l497_497009


namespace min_tgA_tgB_right_triangle_l497_497784

noncomputable theory

-- Define the problem in Lean
theorem min_tgA_tgB_right_triangle
  (A B C : Real)
  (h1 : A + B + C = Real.pi)
  (h2 : A < Real.pi / 2)
  (h3 : B < Real.pi / 2)
  (h4 : C ≤ Real.pi / 2)
  (h5 : ∀ A B, A < Real.pi / 2 → B < Real.pi / 2 → (∃ C, A + B + C = Real.pi → C ≤ Real.pi / 2))
  (h_min : ∀ tgA tgB : Real, tgA * tgB ≥ 1) :
  C = Real.pi / 2 :=
sorry

end min_tgA_tgB_right_triangle_l497_497784


namespace range_of_set_l497_497049

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497049


namespace range_of_set_l497_497057

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497057


namespace cost_of_cookies_l497_497553

theorem cost_of_cookies (diane_has : ℕ) (needs_more : ℕ) (cost : ℕ) :
  diane_has = 27 → needs_more = 38 → cost = 65 :=
by
  sorry

end cost_of_cookies_l497_497553


namespace range_of_set_l497_497048

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497048


namespace range_of_set_is_six_l497_497091

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497091


namespace fraction_rounded_to_decimal_l497_497931

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497931


namespace problem_B_and_D_l497_497719

variables {ℝ : Type*} [linear_ordered_field ℝ]

-- Define the functions f and g
variables (f g : ℝ → ℝ)

-- Define the conditions
def periodic (h : ℝ → ℝ) : Prop := ∃ T > 0, ∀ x, h(x + T) = h(x)
def monotonically_increasing_on (h : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → h(x) ≤ h(y)

-- Assume f and g are periodic and monotonically increasing on [-1, 1]
variables (T_f T_g : ℝ) (hf_periodic : periodic f) (hg_periodic : periodic g)
variables (hf_inc : monotonically_increasing_on f (-1) 1) (hg_inc : monotonically_increasing_on g (-1) 1)

-- Lean statement to prove B and D:

theorem problem_B_and_D :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f(x) + g(x) - |f(x) - g(x)| = 2 * min (f(x)) (g(x)) ∧ 2 * min (f(x)) (g(x)) is strictly increasing on [-1, 1])
  ∧
  ∀ x (hx : True), f(x) + g(x) + f(-x) + g(-x) = f(-x) + g(-x) + f(x) + g(x) :=
begin
  sorry
end

end problem_B_and_D_l497_497719


namespace num_satisfying_values_of_a_l497_497666

theorem num_satisfying_values_of_a :
  (∃! (x : ℕ) (H₁ : x > 0), 3 * x > 4 * x - 4 ∧ ∀ (a : ℕ), 4 * x - a > -8) ∧
  ∀ (x : ℕ) (a : ℕ), (3 * x > 4 * x - 4 ∧ 4 * x - a > -8) →
  (x < 4 ∧ (x = 1 → a < 12) ∧ (x = 2 → a < 16)) →
  16 ≤ a ∧ a < 20 :=
begin
  sorry,
end

end num_satisfying_values_of_a_l497_497666


namespace range_of_set_l497_497050

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497050


namespace product_of_solutions_eq_neg_ten_l497_497442

theorem product_of_solutions_eq_neg_ten :
  (∃ x₁ x₂, -20 = -2 * x₁^2 - 6 * x₁ ∧ -20 = -2 * x₂^2 - 6 * x₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -10) :=
by
  sorry

end product_of_solutions_eq_neg_ten_l497_497442


namespace sum_2017_terms_l497_497686

noncomputable def sequence_a (n : ℕ) : ℝ := sorry
noncomputable def sequence_S (n : ℕ) : ℝ := ∑ i in Finset.range (n+1), sequence_a i

axiom a1 : sequence_a 1 = 1
axiom recurrence_relation (n : ℕ) : sequence_a (n+1) + 2 * sequence_S n * sequence_S (n+1) = 0

theorem sum_2017_terms : sequence_S 2017 = 1 / 4033 := sorry

end sum_2017_terms_l497_497686


namespace count_repeating_decimals_between_1_and_20_l497_497220

def is_repeating_decimal (n : ℕ) : Prop := 
  ∀ (q r : ℕ), q * 18 + r = n → (r ≠ 0 ∧ r % 2 ≠ 0 ∧ r % 5 ≠ 0)

theorem count_repeating_decimals_between_1_and_20 : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 20 ∧ is_repeating_decimal n}.card = 14 :=
sorry

end count_repeating_decimals_between_1_and_20_l497_497220


namespace students_with_both_uncool_parents_l497_497757

theorem students_with_both_uncool_parents :
  let total_students := 35
  let cool_dads := 18
  let cool_moms := 22
  let both_cool := 11
  total_students - (cool_dads + cool_moms - both_cool) = 6 := by
sorry

end students_with_both_uncool_parents_l497_497757


namespace factor_poly_eq_factored_form_l497_497598

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497598


namespace max_truthful_gnomes_l497_497998

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

end max_truthful_gnomes_l497_497998


namespace area_of_sheet_is_correct_l497_497669

noncomputable def area_of_rolled_sheet (length width height thickness : ℝ) : ℝ :=
  (length * width * height) / thickness

theorem area_of_sheet_is_correct :
  area_of_rolled_sheet 80 20 5 0.1 = 80000 :=
by
  -- The proof is omitted (sorry).
  sorry

end area_of_sheet_is_correct_l497_497669


namespace chessboard_not_divisible_by_10_l497_497819

theorem chessboard_not_divisible_by_10 (board : Fin 8 × Fin 8 → ℕ)
    (operation : (Fin 8 × Fin 8) → bool) -- an abstract representation of selecting a 3x3 or 4x4 square
    (increase : (Fin 8 × Fin 8) → (Fin 8 × Fin 8 → ℕ) → (Fin 8 × Fin 8 → ℕ)) -- an abstract representation of the increase operation
    (goal : (Fin 8 × Fin 8 → ℕ) → Prop) -- a representation of the goal of having all numbers divisible by 10
    : ¬(∃ op_seq : List (Fin 8 × Fin 8), (∀ op ∈ op_seq, operation op) ∧ goal (op_seq.foldl (λ b op, increase op b) board)) :=
by
  -- The proof will go here
  sorry

end chessboard_not_divisible_by_10_l497_497819


namespace sequence_divisible_1986_infinite_l497_497794

theorem sequence_divisible_1986_infinite :
  ∃ k p : ℕ, ∀ m : ℕ,
    let u : ℕ → ℤ := 
      λ n, if n = 0 then 39 else 
        if n = 1 then 45 else 
          u (n - 1) ^ 2 - u (n - 2)
    in (u (k + 2) % 1986 = 0) ∧ (u (k + 2 + m * p) % 1986 = 0) := sorry

end sequence_divisible_1986_infinite_l497_497794


namespace rationalize_denominator_l497_497841
open Nat

theorem rationalize_denominator :
  ∃ A B C D : ℤ, A = -7 ∧ B = 5 ∧ C = 21 ∧ D = 4 ∧
    D > 0 ∧
    ∀ (p : Nat), prime p → ¬(p * p ∣ B) ∧
    gcd A (gcd C D) = 1 ∧
    A + B + C + D = 23 :=
by
  use [-7, 5, 21, 4]
  simp
  sorry

end rationalize_denominator_l497_497841


namespace main_theorem_l497_497188

-- Reframe part (a)
def part_a : Prop :=
  let P2020 := ∏ i in finset.range 2021, nat.factorial i
  ∃ m : ℕ, m = 1010 ∧ is_square (P2020 / nat.factorial m)

-- Reframe part (b)
def part_b : Prop :=
  ∀ k : ℕ, k > 0 →
    let n := 8 * k^2
    let Pn := ∏ i in finset.range (n + 1), nat.factorial i
    ∃ m₁ m₂ : ℕ, m₁ ≠ m₂ ∧ is_square (Pn / nat.factorial m₁) ∧ is_square (Pn / nat.factorial m₂)

-- The main theorem to combine both parts
theorem main_theorem : part_a ∧ part_b :=
by
  split
  -- Prove part_a
  sorry,
  -- Prove part_b
  sorry

end main_theorem_l497_497188


namespace sum_abs_arithmetic_sequence_l497_497248

variable (n : ℕ)

def S_n (n : ℕ) : ℚ :=
  - ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n

def T_n (n : ℕ) : ℚ :=
  if n ≤ 34 then
    -((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n
  else
    ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 - ((205 : ℕ) / (2 : ℕ) : ℚ) * n + 3502

theorem sum_abs_arithmetic_sequence :
  T_n n = (if n ≤ 34 then -((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n
           else ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 - ((205 : ℕ) / (2 : ℕ) : ℚ) * n + 3502) :=
by sorry

end sum_abs_arithmetic_sequence_l497_497248


namespace general_term_aₙ_sum_first_n_terms_l497_497232

-- Definitions of the sequences and conditions
def a₄ : ℕ := 14
def S₁₀ : ℕ := 185
def b₁ : ℕ := 1
def r : ℕ := 2

-- Statement for the general term formula aₙ
theorem general_term_aₙ (n : ℕ) : 
  ∃ a₁ d, aₙ = a₁ + (n - 1) * d ∧ a₁ + 3 * d = a₄ ∧ 10 * a₁ + 45 * d = S₁₀ :=
sorry

-- The general term formula for the geometric sequence bₙ
def bₙ (n : ℕ) : ℕ := 2^(n - 1)

-- Statement for the sum of the first n terms of the sequence {aₙ + bₙ}
theorem sum_first_n_terms (n : ℕ) : 
  ∃ aₙ, (S_n : ℕ → ℕ) (S_n (λ n, aₙ n + bₙ n) = (3 * n^2 + 7 * n) / 2 + 2^n - 1) :=
sorry

end general_term_aₙ_sum_first_n_terms_l497_497232


namespace range_of_set_of_three_numbers_l497_497139

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497139


namespace range_of_a_l497_497025

def f (x : ℝ) : ℝ :=
  if 2 ≤ x ∧ x ≤ 3 then -x^2 + 4 * x
  else if 3 < x ∧ x < 4 then (x^2 + 2) / x
  else 0 -- Placeholder for other cases

def g (a : ℝ) (x : ℝ) : ℝ := a * x + 1

theorem range_of_a (a : ℝ) : 
  (∃ (x_1 : ℝ) (H : x_1 ∈ Ico (-2 : ℝ) 0), 
     ∃ (x_2 : ℝ) (H : x_2 ∈ Icc (-2 : ℝ) 1), 
       g a x_2 = f x_1) ↔ (a ∈ Iic (-1/4) ∨ a ∈ Ici (1/8)) :=
sorry

end range_of_a_l497_497025


namespace factorization_correct_l497_497580

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497580


namespace round_fraction_to_three_decimal_l497_497909

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497909


namespace fraction_rounded_equals_expected_l497_497926

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497926


namespace fraction_to_three_decimal_places_l497_497863

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497863


namespace find_horses_l497_497458

theorem find_horses {x : ℕ} :
  (841 : ℝ) = 8 * (x : ℝ) + 16 * 9 + 18 * 6 → 348 = 16 * 9 →
  x = 73 :=
by
  intros h₁ h₂
  sorry

end find_horses_l497_497458


namespace equilateral_triangle_intersections_l497_497555

-- Define the main theorem based on the conditions

theorem equilateral_triangle_intersections :
  let a_1 := (6 - 1) * (7 - 1) / 2
  let a_2 := (6 - 2) * (7 - 2) / 2
  let a_3 := (6 - 3) * (7 - 3) / 2
  let a_4 := (6 - 4) * (7 - 4) / 2
  let a_5 := (6 - 5) * (7 - 5) / 2
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 70 := by
  sorry

end equilateral_triangle_intersections_l497_497555


namespace fraction_to_three_decimal_places_l497_497864

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497864


namespace distance_between_cities_l497_497432

theorem distance_between_cities (T : ℝ) (hT : 6 * T = 24) : 
  let D := 78 * T in
  D = 312 :=
by
  admit

end distance_between_cities_l497_497432


namespace gasoline_tank_capacity_l497_497027

theorem gasoline_tank_capacity (x : ℝ)
  (h1 : (7 / 8) * x - (1 / 2) * x = 12) : x = 32 := 
sorry

end gasoline_tank_capacity_l497_497027


namespace min_tables_42_l497_497021

def min_tables_needed (total_people : ℕ) (table_sizes : List ℕ) : ℕ :=
  sorry

theorem min_tables_42 :
  min_tables_needed 42 [4, 6, 8] = 6 :=
sorry

end min_tables_42_l497_497021


namespace tax_percentage_l497_497530

theorem tax_percentage (C T : ℝ) (h1 : C + 10 = 90) (h2 : 1 = 90 - C - T * 90) : T = 0.1 := 
by 
  -- We provide the conditions using sorry to indicate the steps would go here
  sorry

end tax_percentage_l497_497530


namespace B_age_l497_497011

-- Define the conditions
variables (x y : ℕ)
variable (current_year : ℕ)
axiom h1 : 10 * x + y + 4 = 43
axiom reference_year : current_year = 1955

-- Define the relationship between the digit equation and the year
def birth_year (x y : ℕ) : ℕ := 1900 + 10 * x + y

-- Birth year calculation
def age (current_year birth_year : ℕ) : ℕ := current_year - birth_year

-- Final theorem: Age of B
theorem B_age (x y : ℕ) (current_year : ℕ) (h1 : 10 * x  + y + 4 = 43) (reference_year : current_year = 1955) :
  age current_year (birth_year x y) = 16 :=
by
  sorry

end B_age_l497_497011


namespace trig_condition_necessary_but_not_sufficient_l497_497467

theorem trig_condition_necessary_but_not_sufficient :
  (∀ α : ℝ, (sin (2 * α) - real.sqrt 3 * cos (2 * α) = 1) → α = real.pi / 4 ∨ α = 7 * real.pi / 12) →
  (∃ α : ℝ, sin (2 * α) - real.sqrt 3 * cos (2 * α) = 1) →
  (α = real.pi / 4 → sin (2 * α) - real.sqrt 3 * cos (2 * α) = 1) ∧
  (sin (2 * α) - real.sqrt 3 * cos (2 * α) = 1 → α = real.pi / 4 ∨ α = 7 * real.pi / 12) :=
sorry

end trig_condition_necessary_but_not_sufficient_l497_497467


namespace problem1_problem2_l497_497262

theorem problem1 (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
    (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := sorry

theorem problem2 (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
    (A + C) / (2 * B + A) = 9 / 5 := sorry

end problem1_problem2_l497_497262


namespace line_intersects_circle_and_angle_conditions_l497_497246

noncomputable def line_circle_intersection_condition (k : ℝ) : Prop :=
  - (Real.sqrt 3) / 3 ≤ k ∧ k ≤ (Real.sqrt 3) / 3

noncomputable def inclination_angle_condition (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

theorem line_intersects_circle_and_angle_conditions (k θ : ℝ) :
  line_circle_intersection_condition k →
  inclination_angle_condition θ →
  ∃ x y : ℝ, (y = k * (x + 1)) ∧ ((x - 1)^2 + y^2 = 1) :=
by
  sorry

end line_intersects_circle_and_angle_conditions_l497_497246


namespace ending_number_divisible_by_3_l497_497414

theorem ending_number_divisible_by_3 : 
∃ n : ℕ, (∀ k : ℕ, (10 + k * 3) ≤ n → (10 + k * 3) % 3 = 0) ∧ 
       (∃ c : ℕ, c = 12 ∧ (n - 10) / 3 + 1 = c) ∧ 
       n = 45 := 
sorry

end ending_number_divisible_by_3_l497_497414


namespace eulers_formula_l497_497810

noncomputable def exp_complex (a b : ℝ) : ℂ :=
  lim (λ n : ℕ, (1 + complex.mk a b / n) ^ n)

theorem eulers_formula (a b : ℝ) : 
  exp_complex a b = complex.exp a * (complex.cos b + complex.sin b * complex.I) := 
sorry

end eulers_formula_l497_497810


namespace factor_polynomial_l497_497645

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497645


namespace polynomial_inequality_l497_497355

theorem polynomial_inequality 
  (n : ℕ) (a : Fin n → ℝ) 
  (h_n : 2 ≤ n)
  (h_roots : ∃ x : Fin n → ℝ, (∀ i, 0 ≤ x i) ∧ ∑ i, x i = 1 ∧ 
  (∀ x, (Polynomial.sum_degree n a).eval x = 0)) :
  0 ≤ ∑ i : Fin n, 2^(i + 2) * a i ∧ 
  ∑ i : Fin n, 2^(i + 2) * a i ≤ ( ((n - 2) / n) ^ n) + 1 := 
sorry

end polynomial_inequality_l497_497355


namespace factor_polynomial_l497_497617

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l497_497617


namespace fraction_rounded_to_decimal_l497_497928

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497928


namespace fraction_to_three_decimal_places_l497_497860

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497860


namespace calculate_difference_square_l497_497276

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end calculate_difference_square_l497_497276


namespace factor_polynomial_l497_497589

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497589


namespace transformed_in_region_l497_497504

noncomputable def in_region (x y : ℝ) : Prop := 
  -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2

noncomputable def transformed_z (x y : ℝ) : ℂ := 
  (1 / 2 + 1 / 2 * complex.I) * (x + y * complex.I)

theorem transformed_in_region (x y : ℝ) (hx : in_region x y) :
  in_region ((x - y) / 2) ((x + y) / 2) :=
begin
  obtain ⟨hx1, hx2, hy1, hy2⟩ := hx,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { linarith },
  { linarith },
end

end transformed_in_region_l497_497504


namespace sum_f_eq_3sqrt2_l497_497679

def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem sum_f_eq_3sqrt2 : 
  (f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6) = 3 * Real.sqrt 2 := 
by 
  sorry

end sum_f_eq_3sqrt2_l497_497679


namespace division_rounded_l497_497850

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497850


namespace common_ratio_l497_497337

-- Define the geometric sequence sum function
def sum_geom (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a1
  else a1 * (1 - q^n) / (1 - q)

-- Given conditions
variables (a1 q : ℝ) (S2 S4 : ℝ)

-- Hypotheses based on conditions
axiom h1 : S2 = sum_geom a1 q 2
axiom h2 : S4 = sum_geom a1 q 4
axiom h3 : S4 = 5 * S2

-- Question to solve
theorem common_ratio (a1 : ℝ) (q S2 S4 : ℝ)
  (h1 : S2 = sum_geom a1 q 2)
  (h2 : S4 = sum_geom a1 q 4)
  (h3 : S4 = 5 * S2) :
  q = -1 ∨ q = 2 ∨ q = -2 :=
begin
  sorry
end

end common_ratio_l497_497337


namespace ComplementaryIsMutuallyExclusiveButNotNecessarilyConversely_l497_497179

-- Definitions based on the conditions provided in the problem
def MutuallyExclusive (A B : Set Ω) : Prop :=
  A ∩ B = ∅

def Complementary (A : Set Ω) (Ω : Set Ω) : Set Ω :=
  Ω \ A

-- Proposition that we need to prove
theorem ComplementaryIsMutuallyExclusiveButNotNecessarilyConversely {Ω : Set Ω} (A B : Set Ω) :
  (∀ A, MutuallyExclusive A (Complementary A Ω)) ∧ (MutuallyExclusive A B → ¬(Complementary A Ω = B)) := 
sorry

end ComplementaryIsMutuallyExclusiveButNotNecessarilyConversely_l497_497179


namespace rounded_to_three_decimal_places_l497_497899

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497899


namespace sum_first_20_terms_sequence_a_l497_497525

noncomputable def sum_of_natural_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def sum_of_odds (m : ℕ) : ℕ :=
  m * m

def sequence_a (n : ℕ) : ℕ :=
  let m := sum_of_natural_numbers n in
  sum_of_odds m - sum_of_odds (m - n)

theorem sum_first_20_terms_sequence_a : (Finset.range 20).sum sequence_a = 44100 :=
  sorry

end sum_first_20_terms_sequence_a_l497_497525


namespace distance_between_points_l497_497654

theorem distance_between_points : 
  (dist (0, 8) (6, 0) = 10) :=
by
  -- All conditions defined directly as given in the problem
  let x1 : ℝ := 0
  let y1 : ℝ := 8
  let x2 : ℝ := 6
  let y2 : ℝ := 0
  -- Using the distance formula definition
  let d := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  -- The proof to be added here is not necessary
  -- because we are just setting up the statement framework
  have h1 : (x2 - x1)^2 = 36 := by norm_num
  have h2 : (y2 - y1)^2 = 64 := by norm_num
  have h3 : 36 + 64 = 100 := by norm_num
  have h4 : Real.sqrt 100 = 10 := by norm_num
  show d = 10 from 
    calc
      d = Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) : rfl
      ... = Real.sqrt (36 + 64) : by rw [h1, h2]
      ... = Real.sqrt 100 : by rw [h3]
      ... = 10 : h4

end distance_between_points_l497_497654


namespace round_8_div_11_l497_497879

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497879


namespace range_of_set_l497_497127

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497127


namespace probability_of_cut_l497_497506

-- Given: A rope of length 4 meters
-- Cutting point C is uniformly distributed between 0 and 4 meters
-- Question: Prove that the probability that one of the pieces is at least 3 times as long as the other
--           and the shorter piece is at least 0.5 meters is 0.25

noncomputable def ropeProbability : ℝ :=
  let p := pmf.uniform (closedInterval (0, 4))
  p.toReal ((set.Icc (0.5, 1)).union (set.Icc (3, 3.5)))

theorem probability_of_cut :
  ropeProbability = 0.25 :=
by
  sorry

end probability_of_cut_l497_497506


namespace factorization_correct_l497_497581

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497581


namespace minimal_moves_to_sort_tokens_l497_497741

-- Conditions and Definitions
def initial_positions : List ℕ := [7, 24, 10, 19, 3, 12, 20, 8, 22, 9, 15, 6, 25, 13, 17, 14, 4, 11, 16, 18, 21, 1, 2, 23, 5]
def target_positions : List ℕ := List.range' 1 26

def swaps : List (ℕ × ℕ) :=
  [(1, 7), (7, 20), (20, 16), (16, 11), (11, 2), (2, 24),
   (3, 10), (10, 23), (23, 14), (14, 18), (18, 5),
   (14, 19), (19, 9), (9, 22),
   (6, 12), (12, 15), (15, 13), (13, 25),
   (17, 21)]

-- Proof Problem Statement
theorem minimal_moves_to_sort_tokens :
  ∃ moves : List (ℕ × ℕ), moves.length = 19 ∧
  (apply_swaps initial_positions swaps) = target_positions := sorry

end minimal_moves_to_sort_tokens_l497_497741


namespace range_of_set_is_six_l497_497084

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497084


namespace rounded_to_three_decimal_places_l497_497894

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497894


namespace age_difference_l497_497379

variables (F S M B : ℕ)

theorem age_difference:
  (F - S = 38) → (M - B = 36) → (F - M = 6) → (S - B = 4) :=
by
  intros h1 h2 h3
  -- Use the conditions to derive that S - B = 4
  sorry

end age_difference_l497_497379


namespace range_of_numbers_is_six_l497_497114

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497114


namespace find_g_neg2_l497_497802

variable (f g : ℝ → ℝ)

axiom f_def : ∀ x : ℝ, f(x) = 4 * x + 6
axiom g_f_def : ∀ x : ℝ, g(f(x)) = 3 * x ^ 2 - 4 * x + 7

theorem find_g_neg2 : g(-2) = 27 := by
  have h1 : f(-2) = -2 := by
    calc
      f(-2) = 4 * (-2) + 6 := by rw [f_def]
      _ = -8 + 6 := by ring
      _ = -2 := by ring
  
  have h2 : g(f(-2)) = g(-2) := by rw [h1]
  
  have h3 : g(f(-2)) = 27 := by
    calc
      g(f(-2)) = 3 * (-2) ^ 2 - 4 * (-2) + 7 := by rw [g_f_def]
      _ = 3 * 4 + 8 + 7 := by ring
      _ = 12 + 8 + 7 := by ring
      _ = 27 := by ring
  
  rw [h2] at h3
  exact h3

end find_g_neg2_l497_497802


namespace solve_for_x_l497_497459

theorem solve_for_x (x y : ℤ) (h1 : 3 ^ x * 4 ^ y = 19683) (h2 : x - y = 9) : x = 9 := by
  sorry

end solve_for_x_l497_497459


namespace fraction_rounded_equals_expected_l497_497923

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497923


namespace factorization_correct_l497_497579

noncomputable def polynomial_expr := 
  λ x : ℝ => (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8)

noncomputable def factored_expr := 
  λ x : ℝ => (x^2 + 6 * x + 9) * (x^2 + 6 * x + 1)

theorem factorization_correct : 
  ∀ x : ℝ, polynomial_expr x = factored_expr x :=
by
  intro x
  sorry

end factorization_correct_l497_497579


namespace dwarfs_truth_claims_l497_497968

/-- Seven dwarfs lined up by height, starting with the tallest.
Each dwarf claims a specific height as follows.
Verify that the largest number of dwarfs telling the truth is exactly 1. -/
theorem dwarfs_truth_claims :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  ∀ (actual_height: Fin 7 → ℕ), 
    (∀ i, actual_height i > actual_height (Fin.mk i (by linarith))) →
    ∀ claim_truth: Fin 7 → Prop, 
    (∀ i, claim_truth i ↔ actual_height i = heights[i]) →
    (∃ k, ∀ j, claim_truth j → j = k) :=
by
  intro heights actual_height h_order claim_truth h_claim_match
  existsi Fin.mk 0 (by linarith) -- Claim that only one can be telling the truth (arbitrary choice)
  sorry  -- Proof steps are omitted

end dwarfs_truth_claims_l497_497968


namespace boy_speed_l497_497473

def convert_miles_to_kilometers (miles : ℝ) : ℝ :=
  miles * 1.60934

def convert_minutes_to_hours (minutes : ℝ) : ℝ :=
  minutes / 60

def calculate_speed (distance_km : ℝ) (time_hr : ℝ) : ℝ :=
  distance_km / time_hr

theorem boy_speed :
  let distance_miles := 4.8
  let time_minutes := 33
  let distance_km := convert_miles_to_kilometers distance_miles
  let time_hr := convert_minutes_to_hours time_minutes
  calculate_speed distance_km time_hr = 14.05 :=
by
  sorry

end boy_speed_l497_497473


namespace cersei_initial_candy_count_l497_497536

theorem cersei_initial_candy_count (C: ℤ) 
  (h1: ∃ B: ℤ, B = 5)
  (h2: ∃ S: ℤ, S = 5)
  (h3: ∃ E: ℤ, E = 12)
  (h4: ∃ R: ℤ, R = 18)
  : (3 / 4 * (C - 10) - 12 = 18) → C = 40 :=
begin
  sorry
end

end cersei_initial_candy_count_l497_497536


namespace annual_interest_rate_l497_497493

-- Define the initial conditions
def P : ℝ := 5600
def A : ℝ := 6384
def t : ℝ := 2
def n : ℝ := 1

-- The theorem statement:
theorem annual_interest_rate : ∃ (r : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ r = 0.067 :=
by 
  sorry -- proof goes here

end annual_interest_rate_l497_497493


namespace minimum_perimeter_of_polygons_l497_497543

theorem minimum_perimeter_of_polygons (n m : ℕ) (h1 : 3 ≤ n ∧ 3 ≤ m) (h2 : n ≥ 4) :
    let perimeter := 2 * n + 2 * n + 2 * m - 6 in
    perimeter = 16 →
    16 ≤ perimeter
:= by
   sorry

end minimum_perimeter_of_polygons_l497_497543


namespace rounded_to_three_decimal_places_l497_497901

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497901


namespace sequence_a2016_l497_497725

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 1 ∧ 
  ∀ n, a (n + 2) = a (n + 1) - a n

theorem sequence_a2016 : ∃ a : ℕ → ℤ, sequence a ∧ a 2016 = 0 := 
by
  sorry

end sequence_a2016_l497_497725


namespace impossible_to_obtain_vertex_l497_497662

open Set

noncomputable def initial_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)}

def mirror (p q : ℝ × ℝ × ℝ) : (ℝ × ℝ × ℝ) :=
  (2 * q.1 - p.1, 2 * q.2 - p.2, 2 * q.3 - p.3)

theorem impossible_to_obtain_vertex :
  ∀ S, initial_set ⊆ S →
  (∀ p q ∈ S, mirror p q ∈ S) →
  (1, 1, 1) ∉ S :=
by
  intro S hS hmir
  sorry

end impossible_to_obtain_vertex_l497_497662


namespace range_of_set_l497_497054

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497054


namespace range_of_set_l497_497047

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497047


namespace collinear_proof_l497_497526

noncomputable def points_are_collinear 
  (A B C D Q P : Point) 
  (h1: is_diameter AB)
  (h2: is_tangent PA A)
  (h3: is_tangent PC C)
  (h4: perp CD AB D)
  (h5: is_midpoint Q CD) : Prop := 
  collinear P Q B

-- The theorem to be proven
theorem collinear_proof
  (A B C D Q P : Point)
  (h1: is_diameter AB)
  (h2: is_tangent PA A)
  (h3: is_tangent PC C)
  (h4: perp CD AB D)
  (h5: is_midpoint Q CD) : collinear P Q B := 
sorry

end collinear_proof_l497_497526


namespace geom_progression_identity_l497_497351

-- Definition of the geometric progression terms and sums.
def a_n (a q : ℝ) (n : ℕ) : ℝ := a * q^(n-1)
def S_n (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)
def S_n_prime (a q : ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, a * q^(-i)

-- Proof statement.
theorem geom_progression_identity (a q : ℝ) (n : ℕ) (h_q_ne_1 : q ≠ 1) : 
  a * S_n a q n = a_n a q n * S_n_prime a q n := 
sorry

end geom_progression_identity_l497_497351


namespace range_of_numbers_is_six_l497_497142

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497142


namespace triangle_d_not_right_l497_497519

noncomputable def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_d_not_right :
  ¬is_right_triangle 7 8 13 :=
by sorry

end triangle_d_not_right_l497_497519


namespace gnomes_telling_the_truth_l497_497947

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l497_497947


namespace linear_func_passing_point_l497_497699

theorem linear_func_passing_point :
  ∃ k : ℝ, ∀ x y : ℝ, (y = k * x + 1) → (x = -1 ∧ y = 0) → k = 1 :=
by
  sorry

end linear_func_passing_point_l497_497699


namespace solve_for_Q_l497_497380

theorem solve_for_Q (Q : ℝ) : 
  (sqrt (2 * Q^3) = 64 * (32)^(1/16)) → 
  Q = 2^(123/24) :=
by 
  intros h
  sorry

end solve_for_Q_l497_497380


namespace relationship_among_abc_l497_497240

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := (2 / 3) ^ (2 / 5)
noncomputable def c : ℝ := Real.log (1 / 5) / Real.log (1 / 3)

theorem relationship_among_abc : c > b ∧ b > a :=
by
  have h1 : a = (1 / 3) ^ (2 / 5) := rfl
  have h2 : b = (2 / 3) ^ (2 / 5) := rfl
  have h3 : c = Real.log (1 / 5) / Real.log (1 / 3) := rfl
  sorry

end relationship_among_abc_l497_497240


namespace max_truthful_dwarfs_l497_497957

theorem max_truthful_dwarfs :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  (∃ n, n = 1 ∧ ∀ i, i < 7 → i ≠ n → heights[i] ≠ (heights[i] + 1)) :=
by
  let heights := [60, 61, 62, 63, 64, 65, 66]
  existsi 1
  intro i
  intro h
  intro hi
  sorry

end max_truthful_dwarfs_l497_497957


namespace certain_number_value_l497_497301

theorem certain_number_value (x : ℕ) (p n : ℕ) (hp : Nat.Prime p) (hx : x = 44) (h : x / (n * p) = 2) : n = 2 := 
by
  sorry

end certain_number_value_l497_497301


namespace probability_only_one_solves_l497_497764

theorem probability_only_one_solves :
  (1/2) * (1 - 1/3) * (1 - 1/4) + (1/3) * (1 - 1/2) * (1 - 1/4) + (1/4) * (1 - 1/2) * (1 - 1/3) = 11/24 :=
by
  rw [show 1 - 1/3 = 2/3, by norm_num, show 1 - 1/4 = 3/4, by norm_num]
  rw [show 1 - 1/2 = 1/2, by norm_num, show 1 - 1/4 = 3/4, by norm_num]
  rw [show 1 - 1/2 = 1/2, by norm_num, show 1 - 1/3 = 2/3, by norm_num]
  norm_num

end probability_only_one_solves_l497_497764


namespace problem_statement_l497_497672

noncomputable def a := Real.logBase (1/3) 2
noncomputable def b := (1/3)^2
noncomputable def c := 2^(1/3)

theorem problem_statement : a < b ∧ b < c := by
  have h0 : Real.logBase (1/3) 2 < Real.logBase (1/3) 1 := sorry,
  have h1 : Real.logBase (1/3) 1 = 0 := sorry,
  have h2 : 0 < (1/3)^2 := sorry,
  have h3 : (1/3)^2 < 1 := sorry,
  have h4 : 2^(1/3) > 1 := sorry,
  sorry

end problem_statement_l497_497672


namespace find_other_endpoint_diameter_l497_497538

-- Define the given conditions
def center : ℝ × ℝ := (1, 2)
def endpoint_A : ℝ × ℝ := (4, 6)

-- Define a function to find the other endpoint
def other_endpoint (center endpoint_A : ℝ × ℝ) : ℝ × ℝ := 
  let vector_CA := (center.1 - endpoint_A.1, center.2 - endpoint_A.2)
  let vector_CB := (-vector_CA.1, -vector_CA.2)
  (center.1 + vector_CB.1, center.2 + vector_CB.2)

-- State the theorem
theorem find_other_endpoint_diameter : 
  ∀ center endpoint_A, other_endpoint center endpoint_A = (4, 6) :=
by
  intro center endpoint_A
  -- Proof would go here
  sorry

end find_other_endpoint_diameter_l497_497538


namespace find_first_parrot_weight_l497_497177

def cats_weights := [7, 10, 13, 15]
def cats_sum := List.sum cats_weights
def dog1 := cats_sum - 2
def dog2 := cats_sum + 7
def dog3 := (dog1 + dog2) / 2
def dogs_sum := dog1 + dog2 + dog3
def total_parrots_weight := 2 / 3 * dogs_sum

noncomputable def parrot1 := 2 / 5 * total_parrots_weight
noncomputable def parrot2 := 3 / 5 * total_parrots_weight

theorem find_first_parrot_weight : parrot1 = 38 :=
by
  sorry

end find_first_parrot_weight_l497_497177


namespace factor_polynomial_l497_497591

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497591


namespace term_30_is_132_l497_497405

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ (n.digits 10) -- n.digits 10 returns the list of digits of n in base 10

def satisfies_conditions (n : ℕ) : Prop := is_multiple_of_4 n ∧ contains_digit n 2

def nth_term_of_sequence (n : ℕ) : ℕ :=
  (List.filter satisfies_conditions (List.range (4 * 35 + 1))).nth (n - 1)

theorem term_30_is_132 :
    nth_term_of_sequence 30 = 132 :=
by
  sorry

end term_30_is_132_l497_497405


namespace find_BP_l497_497328

noncomputable theory
open Real

namespace Triangle

def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def segment_ratio (A M P : ℝ × ℝ) (r : ℝ) : Prop :=
  dist A P = r * dist P M

theorem find_BP 
  (A B C M P : ℝ × ℝ)
  (h1 : is_equilateral A B C)
  (h2 : dist A B = 1)
  (h3 : M = midpoint B C)
  (h4 : segment_ratio A M P 4) :
  dist B P = sqrt 7 / 5 :=
sorry

end Triangle

end find_BP_l497_497328


namespace minimum_value_of_expression_l497_497352

theorem minimum_value_of_expression (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (hxyz : x * y * z = 27) : 
  (3 * x + 2 * y + z) ≥ 18 * 2 ^ (1 / 3) := 
sorry

end minimum_value_of_expression_l497_497352


namespace coefficient_of_x_squared_in_expansion_sum_of_coefficients_in_expansion_l497_497778

theorem coefficient_of_x_squared_in_expansion : 
  (∃ n : ℕ, n = 24 ∧ (2 * x + 1)^4.expand_coeff x 2 n) := 
sorry

theorem sum_of_coefficients_in_expansion : 
  (∀ x : ℝ, (2 * x + 1) ^ 4) = 81 := 
sorry

end coefficient_of_x_squared_in_expansion_sum_of_coefficients_in_expansion_l497_497778


namespace juan_distance_l497_497790

def time : ℝ := 80.0
def speed : ℝ := 10.0
def distance (t : ℝ) (s : ℝ) : ℝ := t * s

theorem juan_distance : distance time speed = 800.0 := by
  sorry

end juan_distance_l497_497790


namespace angle_c_greater_than_60_l497_497316

theorem angle_c_greater_than_60 (ABC : Type) [triangle ABC] (A B C : ABC)
  (h : (dist A B) ^ 3 = (dist A C) ^ 3 + (dist B C) ^ 3) :
  ∠ C > 60 := 
sorry

end angle_c_greater_than_60_l497_497316


namespace max_one_truthful_dwarf_l497_497991

def dwarf_height_claims : List (ℕ × ℕ) := [
  (1, 60),  -- First dwarf claims 60 cm
  (2, 61),  -- Second dwarf claims 61 cm
  (3, 62),  -- Third dwarf claims 62 cm
  (4, 63),  -- Fourth dwarf claims 63 cm
  (5, 64),  -- Fifth dwarf claims 64 cm
  (6, 65),  -- Sixth dwarf claims 65 cm
  (7, 66)   -- Seventh dwarf claims 66 cm
]

-- Proof problem statement: Prove that at most one dwarf is telling the truth about their height
theorem max_one_truthful_dwarf : ∀ (heights : List ℕ), heights.length = 7 → 
  (∀ i, i < heights.length → dwarf_height_claims.get? i ≠ none → (heights.get? i = dwarf_height_claims.get? i) → 
    ∑ i in finset.range heights.length, bool.to_num (heights.get? i = dwarf_height_claims.get? i) ≤ 1) :=
by
  intros heights len_heights valid_claims idx idx_in_range some_claim match_claim
  -- Add proof here 
  sorry

end max_one_truthful_dwarf_l497_497991


namespace intersection_points_count_l497_497258

noncomputable def f (x : ℝ) : ℝ := Real.ln x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp 1

theorem intersection_points_count : 
  let h (x : ℝ) := Real.ln x - x / Real.exp 1 in
  ∃! (x : ℝ), h x = 0 :=
sorry

end intersection_points_count_l497_497258


namespace trains_cross_time_l497_497435
noncomputable def time_to_cross (length : ℝ) (t1 t2 : ℝ) : ℝ :=
  let s1 := length / t1
  let s2 := length / t2
  let relative_speed := s1 + s2
  let total_distance := 2 * length
  total_distance / relative_speed

theorem trains_cross_time :
  time_to_cross 120 10 12 ≈ 10.91 := sorry

end trains_cross_time_l497_497435


namespace min_area_part_bounded_by_C_l_m_l497_497326

noncomputable def k : ℝ := sorry
noncomputable def r (k : ℝ) : ℝ := (Real.sqrt (k^2 + 4)) / (2 * k)
noncomputable def alpha (k : ℝ) : ℝ := r(k) + 1/2
noncomputable def beta (k : ℝ) : ℝ := r(k) - 1/2

theorem min_area_part_bounded_by_C_l_m (k : ℝ) (hk : k > 0) : 
  let A (k : ℝ) := (k / 6) + (1 / k) in
  A (Real.sqrt 6) = Real.sqrt (2 / 3) :=
sorry

end min_area_part_bounded_by_C_l_m_l497_497326


namespace range_of_a_l497_497242

noncomputable def f (x a : ℝ) : ℝ := (log (2^x + 4 / (2^x) - a))⁻¹

theorem range_of_a {a : ℝ} 
  (h1 : ∀ x : ℝ, 2^x + 4 * (2^(-x)) - a > 0)
  (h2 : ∀ x : ℝ, 2^x + 4 * (2^(-x)) - a ≠ 1) :
  a < 3 :=
sorry

end range_of_a_l497_497242


namespace fraction_to_three_decimal_places_l497_497857

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497857


namespace theta_value_l497_497743

theorem theta_value (θ : ℝ) (h1 : 0 < θ ∧ θ < 90) : θ = 30 :=
by
  have h2 : sqrt 5 * sin (15 * π / 180) = cos (θ * π / 180) + sin (θ * π / 180),
    from sorry,
  sorry

end theta_value_l497_497743


namespace median_2035_5_of_list_l497_497441

def is_sorted (l : List ℕ) : Prop :=
  l = l.sorted

theorem median_2035_5_of_list :
  let list_n := (List.range 2080).map (λ x => x + 1)
  let list_n2 := (List.range 2080).map (λ x => (x + 1) ^ 2)
  let combined_list := list_n ++ list_n2
  is_sorted combined_list ->
  (combined_list.nth 2079).getD 0 + (combined_list.nth 2080).getD 0 = 4071 ∧
  2 * 2035.5 = 4071 :=
by
  sorry

end median_2035_5_of_list_l497_497441


namespace round_8_div_11_l497_497868

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497868


namespace find_y_l497_497700

theorem find_y (y : ℝ) (A B : ℝ × ℝ)
  (hA : A = (3, y))
  (hB : B = (2, -3))
  (h_slope : Real.tan (135 * Real.pi / 180) = -1) :
  y = -4 :=
by sorry

end find_y_l497_497700


namespace transform_equation_l497_497288

variable (x y : ℝ)

theorem transform_equation 
  (h : y = x + 1/x) 
  (hx : x^4 + x^3 - 5 * x^2 + x + 1 = 0):
  x^2 * (y^2 + y - 7) = 0 := by
  sorry

end transform_equation_l497_497288


namespace chord_length_of_conic_intersection_l497_497692

theorem chord_length_of_conic_intersection :
  let C := { p : ℝ × ℝ | ∃ (θ : ℝ), p.1 = 2 * Real.cos θ ∧ p.2 = Real.sin θ }
  let A := (0 : ℝ, -Real.sqrt 3)
  let l := { p : ℝ × ℝ | p.1 + p.2 + Real.sqrt 3 = 0 }
  ∃ (E F : ℝ × ℝ), E ∈ C ∧ F ∈ C ∧ E ∈ l ∧ F ∈ l ∧ 
  ( ( (E.1 - F.1)^2 + (E.2 - F.2)^2 ) = (8 / 5)^2 ) :=
sorry

end chord_length_of_conic_intersection_l497_497692


namespace range_of_set_l497_497123

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497123


namespace no_valid_prime_angles_l497_497769

def is_prime (n : ℕ) : Prop := Prime n

theorem no_valid_prime_angles :
  ∀ (x : ℕ), (x < 30) ∧ is_prime x ∧ is_prime (3 * x) → False :=
by sorry

end no_valid_prime_angles_l497_497769


namespace train_max_power_and_speed_correct_l497_497499

noncomputable def train_max_power_and_speed  (m : ℕ) (v_kmh : ℕ) (epsilon : ℝ) (sin_alpha : ℝ) (cos_alpha : ℝ) : ℝ × ℝ :=
  let v := (v_kmh : ℝ) * 1000 / 3600 in -- Convert km/h to m/s
  let P1 := m * epsilon in -- Force overcoming resistance on horizontal track
  let Hp := P1 * v / 75 in -- Power on horizontal track in horsepower
  let P2 := m * sin_alpha in -- Force due to gravity along the incline
  let P := P1 * cos_alpha + P2 in -- Total force on inclined plane
  let max_speed_m_s := Hp * 75 / P in -- Maximum speed on incline in m/s
  let max_speed_kmh := max_speed_m_s * 3600 / 1000 in -- Convert m/s to km/h
  (Hp, max_speed_kmh)

theorem train_max_power_and_speed_correct :
  train_max_power_and_speed 300000 35 0.005 0.01 1 = (194.4, 11.6) := sorry

end train_max_power_and_speed_correct_l497_497499


namespace ratio_square_pentagon_l497_497153

theorem ratio_square_pentagon (P_sq P_pent : ℕ) 
  (h_sq : P_sq = 60) (h_pent : P_pent = 60) :
  (P_sq / 4) / (P_pent / 5) = 5 / 4 :=
by 
  sorry

end ratio_square_pentagon_l497_497153


namespace find_xy_ratio_l497_497343

noncomputable def xy_ratio (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x / y + y / x = 4) : ℝ :=
  xy / (x^2 - y^2)

theorem find_xy_ratio (x y : ℝ) (h: x > 0 ∧ y > 0 ∧ x / y + y / x = 4) : xy_ratio x y h = Real.sqrt 3 / 3 := by
  sorry

end find_xy_ratio_l497_497343


namespace factorization_identity_l497_497605

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497605


namespace slope_divides_rectangle_l497_497462

def slope_of_dividing_line (x1 y1 x2 y2 : ℝ) (origin center : ℝ × ℝ): ℝ :=
  (center.snd - origin.snd) / (center.fst - origin.fst)

def center_of_rectangle (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

theorem slope_divides_rectangle :
  let vertices := ((1 : ℝ), (0 : ℝ)), ((9 : ℝ), (0 : ℝ)), ((1 : ℝ), (2 : ℝ)), ((9 : ℝ), (2 : ℝ))
  let origin := (0 : ℝ, 0 : ℝ)
  let center := center_of_rectangle 1 0 9 2
  slope_of_dividing_line 1 0 9 2 origin center = (1/5 : ℝ) :=
by sorry

end slope_divides_rectangle_l497_497462


namespace cubic_and_quintic_values_l497_497795

theorem cubic_and_quintic_values (a : ℝ) (h : (a + 1/a)^2 = 11) : 
    (a^3 + 1/a^3 = 8 * Real.sqrt 11 ∧ a^5 + 1/a^5 = 71 * Real.sqrt 11) ∨ 
    (a^3 + 1/a^3 = -8 * Real.sqrt 11 ∧ a^5 + 1/a^5 = -71 * Real.sqrt 11) :=
by
  sorry

end cubic_and_quintic_values_l497_497795


namespace at_most_one_dwarf_tells_truth_l497_497974

noncomputable def dwarfs_tell_truth : Prop :=
  ∀ heights: Fin 7 → ℕ,
  heights = fun i => i + 60 →
  ∀ k ∈ Finset.range 7, (heights k) ≠ (60 + k) → (∃ i ∈ Finset.range 7, i ≠ k)

theorem at_most_one_dwarf_tells_truth :
  dwarfs_tell_truth = 1 := 
sorry

end at_most_one_dwarf_tells_truth_l497_497974


namespace factor_polynomial_l497_497627

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497627


namespace product_of_chord_lengths_l497_497797

theorem product_of_chord_lengths :
  ∀ (P Q : ℂ) (r : ℝ) (D₁ D₂ D₃ D₄ : ℂ),
  (abs P = r) →
  (abs Q = r) →
  (abs (P + Q) = 2 * r) →
  (D₁ = r * exp(2 * π * I / 10)) →
  (D₂ = r * exp(4 * π * I / 10)) →
  (D₃ = r * exp(6 * π * I / 10)) →
  (D₄ = r * exp(8 * π * I / 10)) →
  (∏ i in [1, 2, 3, 4], abs (P - D i) * abs (Q - D i) = 98415) :=
begin
  -- placeholder for proof
  sorry,
end

end product_of_chord_lengths_l497_497797


namespace max_sum_condition_l497_497466

theorem max_sum_condition (a b : ℕ) (h1 : 10 ≤ a ∧ a ≤ 99) (h2 : 10 ≤ b ∧ b ≤ 99) (h3 : Nat.gcd a b = 6) : a + b ≤ 186 :=
sorry

end max_sum_condition_l497_497466


namespace factor_polynomial_l497_497588

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497588


namespace range_of_set_l497_497117

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497117


namespace count_repeating_decimals_between_1_and_20_l497_497219

def is_repeating_decimal (n : ℕ) : Prop := 
  ∀ (q r : ℕ), q * 18 + r = n → (r ≠ 0 ∧ r % 2 ≠ 0 ∧ r % 5 ≠ 0)

theorem count_repeating_decimals_between_1_and_20 : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 20 ∧ is_repeating_decimal n}.card = 14 :=
sorry

end count_repeating_decimals_between_1_and_20_l497_497219


namespace factor_polynomial_l497_497593

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l497_497593


namespace tenth_student_problems_l497_497480

variable (p n : ℕ)

@[spec]
axiom class_size (students : ℕ) : students = 10

@[spec]
axiom solved_by_7 (problems : ℕ) : ∀ problem, problem < problems → ∃ (count : ℕ), count = 7

@[spec]
axiom nine_students_solved_4_each (solved : ℕ) : solved = 4

theorem tenth_student_problems (p n : ℕ) 
    (class_size : ∀ students, students = 10)
    (solved_by_7 : ∀ (problem: ℕ) , problem < p → ∃ (count : ℕ), count = 7)
    (nine_students_solved_4_each : ∀ solved, solved = 4) : n = 6 :=
by
  -- Proof goes here
  sorry

end tenth_student_problems_l497_497480


namespace white_lambs_count_l497_497196

theorem white_lambs_count : ∀ (total_lambs black_lambs : ℕ), total_lambs = 6048 → black_lambs = 5855 → total_lambs - black_lambs = 193 :=
by 
  intros total_lambs black_lambs h1 h2
  rw [h1, h2]
  norm_num

end white_lambs_count_l497_497196


namespace max_truthful_dwarfs_l497_497949

theorem max_truthful_dwarfs :
  ∃ n : ℕ, n = 1 ∧ 
    ∀ heights : Fin 7 → ℕ, 
      (heights 0 = 60 ∨ heights 1 = 61 ∨ heights 2 = 62 ∨ heights 3 = 63 ∨ heights 4 = 64 ∨ heights 5 = 65 ∨ heights 6 = 66) → 
      (∀ i j : Fin 7, i < j → heights i > heights j) → 
      ( ∃ t : Fin 7 → bool, 
        ( ∀ i j : Fin 7, t i = tt → t j = tt → i = j) ∧ 
        ( ∀ i : Fin 7, t i = tt → heights i = 60 + i ) ∧
        #(t.to_list.count tt) = n ) :=
by
  use 1
  sorry

end max_truthful_dwarfs_l497_497949


namespace round_8_div_11_l497_497869

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497869


namespace bug_moves_to_starting_vertex_on_twelfth_move_l497_497474

noncomputable def probability_relation : ℕ → ℚ 
| 0 => 1
| n + 1 => 1 / 2 * (1 - probability_relation n)

theorem bug_moves_to_starting_vertex_on_twelfth_move :
  let P := probability_relation 12
  let m := (P.num : ℕ)  -- Numerator of P in reduced form
  let n := (P.den : ℕ)  -- Denominator of P in reduced form
  (nat.coprime m n) 
  ∧ m + n = 2731 :=
by
  -- P_12 and additional required calculations according to Lean's norms
  sorry

end bug_moves_to_starting_vertex_on_twelfth_move_l497_497474


namespace max_real_roots_poly_l497_497657

theorem max_real_roots_poly (n : ℕ) (hn : 0 < n) :
  (∀ (P : polynomial ℝ), P = polynomial.sum (range (2 * n + 1)) (λ k, polynomial.C 1 * X^k) → 
  P.roots.count_root 1 = 0 ∧ P.roots.count_root (-1) = 1) → 
  ∃ (f : polynomial ℝ), 
    (∀ (z : ℝ), f.eval z = 0 → z = -1 ∨ z = 1) := sorry

end max_real_roots_poly_l497_497657


namespace range_of_numbers_is_six_l497_497146

noncomputable def mean (s : Finset ℝ) : ℝ :=
  (s.sum id) / (s.card : ℝ)

theorem range_of_numbers_is_six (a b c : ℝ) (h1 : mean ({a, b, c} : Finset ℝ) = 5) 
  (h2 : b = 5) (h3 : a = 2 ∨ b = 2 ∨ c = 2) (h4 : {a, b, c}.card = 3) : 
  (Finset.max {a, b, c} - Finset.min {a, b, c}) = 6 :=
sorry

end range_of_numbers_is_six_l497_497146


namespace sum_is_two_l497_497331

-- Definitions of the given conditions
variables {a1 a2 a3 a4 a5 : ℝ}

-- Condition 1
def cond1 := 2 * a1 + a2 + a3 + a4 + a5 = 1 + (1 / 8) * a4

-- Condition 2
def cond2 := 2 * a2 + a3 + a4 + a5 = 2 + (1 / 4) * a3

-- Condition 3
def cond3 := 2 * a3 + a4 + a5 = 4 + (1 / 2) * a2

-- Condition 4
def cond4 := 2 * a4 + a5 = 6 + a1

theorem sum_is_two : cond1 ∧ cond2 ∧ cond3 ∧ cond4 → a1 + a2 + a3 + a4 + a5 = 2 :=
by
  intros h
  exact sorry

end sum_is_two_l497_497331


namespace fraction_to_three_decimal_places_l497_497861

theorem fraction_to_three_decimal_places :
  (Real.toRat 0.727 = 727 / 1000) ∧ (8 / 11 : Real) ≈ 0.727 :=
by
  have h1: (Real.toRat 0.727 = 727 / 1000), from sorry
  have h2: ((8 / 11) ≈ 0.727), from sorry
  exact ⟨ h1, h2 ⟩

end fraction_to_three_decimal_places_l497_497861


namespace part_one_part_two_l497_497785

variables {A B C M : Type*} [real A] [real B] [real C] [real M]
variables (a b c : ℝ) (angle_A angle_B angle_C : ℝ) 
  (AM BM CM : ℝ)
  (cos_B : cosine angle_B)
  (cos_A : cosine angle_A)
  (cos_C : cosine angle_C)

theorem part_one 
  (b : ℝ) (hb : b = 5)
  (c : ℝ) (hc : c = real.sqrt 10)
  (h : a * cos_B + b * cos_A = real.sqrt 2 * c * cos_B) :
  a = 3 * real.sqrt 5 :=
  sorry

theorem part_two 
  (tan_ ∠_AMB : ℝ) (hAMB : tan_ ∠_AMB = 3 / 4) 
  (tan_ ∠_MAC : ℝ) :
  tan_ ∠_MAC = 2 / 11 :=
  sorry

end part_one_part_two_l497_497785


namespace rational_x_implies_rational_y_l497_497211

-- Definitions based on the problem statement
def has_digits (x : ℚ) (f : ℕ → ℕ) := λ n, let d := f n in x.num.nth_digit d

def specific_function (n : ℕ) : ℕ := Nat.pow 2 n

-- Main theorem statement translating the problem 1-1
theorem rational_x_implies_rational_y (x y : ℚ) (hx : 0 < x ∧ x < 1)
                                        (hy : 0 < y ∧ y < 1)
                                        (h_digits : ∀ n, y.num.nth_digit n = x.num.nth_digit (specific_function n))
                                        (hr : x.is_rational) : y.is_rational :=
sorry

end rational_x_implies_rational_y_l497_497211


namespace find_C_l497_497786

def A : ℝ × ℝ := (2, 8)
def M : ℝ × ℝ := (4, 11)
def L : ℝ × ℝ := (6, 6)

theorem find_C (C : ℝ × ℝ) (B : ℝ × ℝ) :
  -- Median condition: M is the midpoint of A and B
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  -- Given coordinates for A, M, L
  A = (2, 8) → M = (4, 11) → L = (6, 6) →
  -- Correct answer
  C = (14, 2) :=
by
  intros hmedian hA hM hL
  sorry

end find_C_l497_497786


namespace FriedChickenDinner_orders_count_l497_497437

-- Defining the number of pieces of chicken used by each type of order
def piecesChickenPasta := 2
def piecesBarbecueChicken := 3
def piecesFriedChickenDinner := 8

-- Defining the number of orders for Chicken Pasta and Barbecue Chicken
def numChickenPastaOrders := 6
def numBarbecueChickenOrders := 3

-- Defining the total pieces of chicken needed for all orders
def totalPiecesOfChickenNeeded := 37

-- Defining the number of pieces of chicken needed for Chicken Pasta and Barbecue orders
def piecesNeededChickenPasta : Nat := piecesChickenPasta * numChickenPastaOrders
def piecesNeededBarbecueChicken : Nat := piecesBarbecueChicken * numBarbecueChickenOrders

-- Defining the total pieces of chicken needed for Chicken Pasta and Barbecue orders
def piecesNeededChickenPastaAndBarbecue : Nat := piecesNeededChickenPasta + piecesNeededBarbecueChicken

-- Calculating the pieces of chicken needed for Fried Chicken Dinner orders
def piecesNeededFriedChickenDinner : Nat := totalPiecesOfChickenNeeded - piecesNeededChickenPastaAndBarbecue

-- Defining the number of Fried Chicken Dinner orders
def numFriedChickenDinnerOrders : Nat := piecesNeededFriedChickenDinner / piecesFriedChickenDinner

-- Proving Victor has 2 Fried Chicken Dinner orders
theorem FriedChickenDinner_orders_count : numFriedChickenDinnerOrders = 2 := by
  unfold numFriedChickenDinnerOrders
  unfold piecesNeededFriedChickenDinner
  unfold piecesNeededChickenPastaAndBarbecue
  unfold piecesNeededBarbecueChicken
  unfold piecesNeededChickenPasta
  unfold totalPiecesOfChickenNeeded
  unfold numBarbecueChickenOrders
  unfold piecesBarbecueChicken
  unfold numChickenPastaOrders
  unfold piecesChickenPasta
  sorry

end FriedChickenDinner_orders_count_l497_497437


namespace stratified_sampling_third_year_students_l497_497487

theorem stratified_sampling_third_year_students 
  (N : ℕ) (N_1 : ℕ) (P_sophomore : ℝ) (n : ℕ) (N_2 : ℕ) :
  N = 2000 →
  N_1 = 760 →
  P_sophomore = 0.37 →
  n = 20 →
  N_2 = Nat.ceil (N - N_1 - P_sophomore * N) →
  Nat.floor ((n : ℝ) / (N : ℝ) * (N_2 : ℝ)) = 5 :=
by
  sorry

end stratified_sampling_third_year_students_l497_497487


namespace area_of_quadrilateral_is_correct_l497_497497

def point := ℝ × ℝ

def quadrilateral (A B C D : point) : Prop :=
  A = (0, 0) ∧ B = (0, 2) ∧ C = (4, 2) ∧ D = (8, 0)

theorem area_of_quadrilateral_is_correct (A B C D : point) (h : quadrilateral A B C D) :
  let area := 4 + 2 * Real.sqrt 5 in
  area = 4 + 2 * Real.sqrt 5 :=
by
  sorry

end area_of_quadrilateral_is_correct_l497_497497


namespace rounded_to_three_decimal_places_l497_497903

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497903


namespace calculate_difference_square_l497_497275

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end calculate_difference_square_l497_497275


namespace sniper_max_five_hits_l497_497156

theorem sniper_max_five_hits (sniper_shots : Finset (Fin 100 → Fin 4)) (five_hits : ℕ) : 
  (∃ (triangles : Finset (Fin 100)), ∀ t ∈ triangles, sniper_shots.filter (λ s, s t = 5).card = five_hits ∧ triangles.card = 25) :=
sorry

end sniper_max_five_hits_l497_497156


namespace find_n_mod_16_l497_497206

theorem find_n_mod_16 (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n ≤ 15) (h₂ : n ≡ 15893 [MOD 16]) : n = 5 := 
by sorry

end find_n_mod_16_l497_497206


namespace range_of_set_is_six_l497_497085

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497085


namespace range_of_numbers_is_six_l497_497108

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497108


namespace square_roots_of_positive_number_l497_497299

theorem square_roots_of_positive_number (a : ℝ) (h1 : ∃ b : ℝ, b > 0 ∧ (a + 3)^2 = b ∧ (2a + 3)^2 = b) : a = -2 :=
by 
  sorry

end square_roots_of_positive_number_l497_497299


namespace fraction_rounding_l497_497885

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497885


namespace factorization_identity_l497_497611

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497611


namespace division_rounded_l497_497851

theorem division_rounded (x y : ℚ) (h1 : x = 8) (h2 : y = 11) :
  (x / y).round_to 3 = 0.727 := sorry

end division_rounded_l497_497851


namespace a_n_is_integer_l497_497800

-- Define the initial condition and recurrence relation
def recurrence (a : ℕ → ℤ) (k : ℕ) (n : ℕ) :=
  a(n+1) = (k + 1) * a(n) + k * (a(n) + 1) + 2 * Int.sqrt (k * (k + 1) * a(n) * (a(n) + 1))

-- State the main theorem
theorem a_n_is_integer (k : ℕ) : ∀ n : ℕ, ∃ a : ℕ → ℤ, a 0 = 0 ∧ (∀ m : ℕ, recurrence a k m) ∧ ∀ n : ℕ, a n ∈ ℤ :=
by
  sorry

end a_n_is_integer_l497_497800


namespace fraction_rounded_equals_expected_l497_497922

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497922


namespace red_triangles_exist_l497_497825

theorem red_triangles_exist (n : ℕ) (h_n : 1 < n) (points : Finset (EuclideanSpace 2 ℝ))
  (h_card : points.card = 2 * n)
  (no_three_collinear : ∀ (p q r : points), ¬Collinear ({p, q, r} : Set points))
  (red_segments : Finset (Set points))
  (h_red_segments_card : red_segments.card = n^2 + 1) :
  ∃ (triangle_set : Finset (Finset (Set points))), 
  (triangle_set.card = n) ∧ 
  (∀ (triangle : Finset (Set points)) (h_mem : triangle ∈ triangle_set), (∀ (seg : Set points) (h_seg_mem : seg ∈ triangle), seg ∈ red_segments)) :=
sorry

end red_triangles_exist_l497_497825


namespace affine_transformation_maps_polygon_to_regular_polygon_l497_497465

theorem affine_transformation_maps_polygon_to_regular_polygon
    {n : ℕ} (h : n ≥ 3)
    (O : ℝ × ℝ)
    (A : Fin n → ℝ × ℝ)
    (I : ∀ i : Fin n, 0 ≤ i + 2)
    (h1 : ∀ i : Fin n, (A 0 + A (2 + (i : ℝ) % n)) = 2 * (Real.cos (2 * Real.pi / n)) * (A (1 + (i : ℝ) % n))) 
    (h2 : ∀ i : Fin n, (A 1 + A (3 + (i : ℝ) % n)) = 2 * (Real.cos (2 * Real.pi / n)) * (A (2 + (i : ℝ) % n))) 
    (h_eq : ∀ i : Fin n, (A (i - 1) + A (i + 1)) = 2 * (Real.cos (2 * Real.pi / n)) * (A i)) :
  ∃ (f : ℝ → ℝ) (b : ℝ), True :=
sorry

end affine_transformation_maps_polygon_to_regular_polygon_l497_497465


namespace smallest_D_inequality_l497_497192

theorem smallest_D_inequality (D : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + 4 ≤ D * (x + y + z)) ↔ D ≤ 2 * real.sqrt 2 := 
begin
  split,
  { intro h,
    sorry },  -- here we skip the proof
  { intro hD,
    sorry }   -- here we skip the proof
end

end smallest_D_inequality_l497_497192


namespace minimum_value_of_u_minimum_value_of_u_l497_497732

noncomputable def vec_a : ℝ × ℝ := (Real.cos (40 * Real.pi / 180), Real.sin (40 * Real.pi / 180))
noncomputable def vec_b : ℝ × ℝ := (Real.sin (20 * Real.pi / 180), Real.cos (20 * Real.pi / 180))
noncomputable def vec_u (λ : ℝ) : ℝ × ℝ := 
  (sqrt 3 * vec_a.1 + λ * vec_b.1, sqrt 3 * vec_a.2 + λ * vec_b.2)

def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem minimum_value_of_u (λ : ℝ) :
  (λ : ℝ) → norm (vec_u λ) >= (norm (vec_u (λ := -3 / 2)) ) := sorry

theorem minimum_value_of_u' : norm (vec_u (-3 / 2)) = sqrt 3 / 2 := sorry

end minimum_value_of_u_minimum_value_of_u_l497_497732


namespace geometric_series_sum_l497_497668

theorem geometric_series_sum (n : ℕ) : 
  let a₁ := 2
  let q := 2
  let S_n := a₁ * (1 - q^n) / (1 - q)
  S_n = 2 - 2^(n + 1) := 
by
  sorry

end geometric_series_sum_l497_497668


namespace ellipse_condition_l497_497295

theorem ellipse_condition (m : ℝ) :
  (m > 0) ∧ (2 * m - 1 > 0) ∧ (m ≠ 2 * m - 1) ↔ (m > 1/2) ∧ (m ≠ 1) :=
by
  sorry

end ellipse_condition_l497_497295


namespace chessboard_not_divisible_by_10_l497_497818

theorem chessboard_not_divisible_by_10 :
  ∀ (B : ℕ × ℕ → ℕ), 
  (∀ x y, B (x, y) < 10) ∧ 
  (∀ x y, x ≥ 0 ∧ x < 8 ∧ y ≥ 0 ∧ y < 8) →
  ¬ ( ∃ k : ℕ, ∀ x y, (B (x, y) + k) % 10 = 0 ) :=
by
  intros
  sorry

end chessboard_not_divisible_by_10_l497_497818


namespace round_8_div_11_l497_497871

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497871


namespace approx_value_func_at_x_2_004_l497_497652

theorem approx_value_func_at_x_2_004 :
  let f (x: ℝ) := (x^2 + 2 * x)^(1 / 3: ℝ)
  let x0 := 2
  let Δx := 0.004
  let df_dx := λ x : ℝ, (1 / 3: ℝ) * (x^2 + 2 * x)^(-2 / 3: ℝ) * (2 * x + 2)
  let dy := df_dx x0 * Δx
  let y_approx := f x0 + dy
  y_approx = 2.002
  sorry

end approx_value_func_at_x_2_004_l497_497652


namespace triangle_two_right_angles_impossible_l497_497839

theorem triangle_two_right_angles_impossible
  (A B C : ℝ)
  (h_triangle : A + B + C = 180)
  (h_rightA : A = 90)
  (h_rightB : B = 90) :
  false :=
by
  -- Calculate the sum of angles assuming A and B are 90 degrees
  have h_sum : A + B + C = 90 + 90 + C := by rw [h_rightA, h_rightB]
  -- Prove the contradiction
  have h_contradiction : 180 < 90 + 90 + C := by linarith
  rw h_triangle at h_contradiction
  exact lt_irrefl 180 h_contradiction

end triangle_two_right_angles_impossible_l497_497839


namespace round_8_div_11_l497_497873

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497873


namespace difference_of_squares_l497_497283

theorem difference_of_squares (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by
  sorry

end difference_of_squares_l497_497283


namespace determine_n_l497_497552

theorem determine_n (n : ℕ) : 5^n = 5 * 25^3 * 625^2 → n = 15 := by
  intro h
  have p1 : 25 = 5^2 := by norm_num
  have p2 : 625 = 5^4 := by norm_num
  rw [p1, p2] at h
  simp at h
  have : 5^n = 5^(1 + 6 + 8) := by simpa using h
  norm_num at this
  exact this

end determine_n_l497_497552


namespace probability_two_one_ball_colors_l497_497471

theorem probability_two_one_ball_colors :
  let total_black := 10
  let total_white := 5
  let after_removal_white := total_white - 1
  let after_removal_total := total_black + after_removal_white
  let draw := 3

  -- Probability calculation
  let c := λ n k, Nat.choose n k
  let total_ways := c after_removal_total draw
  let favorable_ways := (c total_black 2 * c after_removal_white 1) + (c total_black 1 * c after_removal_white 2)
  let probability := favorable_ways / total_ways

  -- Expected answer
  probability = 60 / 91 :=
by
  sorry

end probability_two_one_ball_colors_l497_497471


namespace min_value_of_f_max_value_of_f_exist_min_f_exist_max_f_l497_497227

variables {m : ℕ} (r : Fin m → ℚ)
def f (n : ℕ) : ℤ := n - ∑ i, ⌊(r i * n : ℚ)⌋

noncomputable def r_valid := (∀ i, r i > 0) ∧ (∑ i, r i = 1)

theorem min_value_of_f (h : r_valid r) (n : ℕ) (hn : n > 0) : 0 ≤ f r n := 
sorry

theorem max_value_of_f (h : r_valid r) (n : ℕ) (hn : n > 0) : f r n < ↑m :=
sorry

theorem exist_min_f (h : r_valid r) : ∃ n, n > 0 ∧ f r n = 0 :=
sorry

theorem exist_max_f (h : r_valid r) : ∃ n, n > 0 ∧ f r n = m - 1 :=
sorry

end min_value_of_f_max_value_of_f_exist_min_f_exist_max_f_l497_497227


namespace find_other_endpoint_diameter_l497_497539

-- Define the given conditions
def center : ℝ × ℝ := (1, 2)
def endpoint_A : ℝ × ℝ := (4, 6)

-- Define a function to find the other endpoint
def other_endpoint (center endpoint_A : ℝ × ℝ) : ℝ × ℝ := 
  let vector_CA := (center.1 - endpoint_A.1, center.2 - endpoint_A.2)
  let vector_CB := (-vector_CA.1, -vector_CA.2)
  (center.1 + vector_CB.1, center.2 + vector_CB.2)

-- State the theorem
theorem find_other_endpoint_diameter : 
  ∀ center endpoint_A, other_endpoint center endpoint_A = (4, 6) :=
by
  intro center endpoint_A
  -- Proof would go here
  sorry

end find_other_endpoint_diameter_l497_497539


namespace polynomial_factorization_l497_497571

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497571


namespace correct_conclusions_l497_497354

noncomputable def M : Set ℝ := sorry

axiom non_empty : Nonempty M
axiom mem_2 : (2 : ℝ) ∈ M
axiom closed_under_sub : ∀ {x y : ℝ}, x ∈ M → y ∈ M → (x - y) ∈ M
axiom closed_under_div : ∀ {x : ℝ}, x ∈ M → x ≠ 0 → (1 / x) ∈ M

theorem correct_conclusions :
  (0 : ℝ) ∈ M ∧
  (∀ x y : ℝ, x ∈ M → y ∈ M → (x + y) ∈ M) ∧
  (∀ x y : ℝ, x ∈ M → y ∈ M → (x * y) ∈ M) ∧
  ¬ (1 ∉ M) := sorry

end correct_conclusions_l497_497354


namespace solution_set_16_sin_pi_x_cos_pi_x_l497_497407

theorem solution_set_16_sin_pi_x_cos_pi_x (x : ℝ) :
  (x = 1 / 4 ∨ x = -1 / 4) ↔ 16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x :=
sorry

end solution_set_16_sin_pi_x_cos_pi_x_l497_497407


namespace range_of_set_l497_497124

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l497_497124


namespace range_of_set_l497_497070

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497070


namespace possible_values_of_a_l497_497728

-- Declare the sets M and N based on given conditions.
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define a proof where the set of possible values for a is {-1, 0, 2/3}
theorem possible_values_of_a : 
  {a : ℝ | N a ⊆ M} = {-1, 0, 2 / 3} := 
by 
  sorry

end possible_values_of_a_l497_497728


namespace range_of_set_of_three_numbers_l497_497130

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497130


namespace max_truthful_dwarfs_le_one_l497_497980

def dwarf_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

theorem max_truthful_dwarfs_le_one (heights : List ℕ) (h_lengths : heights.length = 7) (h_sorted : heights.sorted (· > ·)) :
  ∃ n, (n ≤ 1) ∧ ∀ i, i ∈ (Finset.range 7).to_list → (heights[i] ≠ dwarf_height_claims[i] → (1 = n)) :=
  sorry

end max_truthful_dwarfs_le_one_l497_497980


namespace find_c_l497_497659

theorem find_c (c : ℝ) (h : ∀ x, 2 < x ∧ x < 6 → -x^2 + c * x + 8 > 0) : c = 8 := 
by
  sorry

end find_c_l497_497659


namespace problem_1_problem_2_problem_3_l497_497697

-- Define the given conditions
def circle_passes_through_points (C : ℝ × ℝ → Prop) (M N : ℝ × ℝ) : Prop :=
  C M ∧ C N

def center_on_line (center : ℝ × ℝ) : Prop :=
  center.1 + 2 * center.2 + 1 = 0

-- Define the circle
def circle_eq (x y : ℝ) (D E F : ℝ) : Prop :=
  x^2 + y^2 + D * x + E * y + F = 0

-- Problem statements
theorem problem_1
  (C : ℝ × ℝ → Prop)
  (M N : ℝ × ℝ)
  (center : ℝ × ℝ)
  (H1 : circle_passes_through_points C M N)
  (H2 : center_on_line center) :
  ∃ D E F, circle_eq (fst M) (snd M) D E F ∧
           circle_eq (fst N) (snd N) D E F ∧
           ∀ x y, C (x, y) ↔ circle_eq x y (-6) 4 4 := 
sorry

theorem problem_2
  (C : ℝ × ℝ → Prop)
  (center : ℝ × ℝ)
  (H1 : ∀ x y, C (x, y) ↔ (x - 3)^2 + (y + 2)^2 = 9)
  (P : ℝ × ℝ)
  (H2 : P = (6, 3)) :
  ∃ k : ℝ, (P.2 - 3 = k * (P.1 - 6) ∨ P.1 = 6) ∧
           (8 * P.1 - 15 * P.2 - 3 = 0 ∨ P.1 = 6) :=
sorry

theorem problem_3
  (C : ℝ × ℝ → Prop)
  (center : ℝ × ℝ)
  (H : ∀ x y, C (x, y) ↔ x^2 + y^2 - 6 * x + 4 * y + 4 = 0)
  (m : ℝ) :
  (∃ l : ℝ → ℝ, l = (λ x, x + m)) ∧ ∀ l, 
  (
    exists A B : ℝ × ℝ,
    C A ∧ C B ∧ 
    (A.2 = A.1 + m ∧ B.2 = B.1 + m) ∧
    ((A.1 * B.1) + (A.2 * B.2) = 0) ∧ 
    ∃ C1 : ℝ × ℝ → Prop, 
    ∀ x y, C1 (x, y) ↔ (x = 0 ∧ y = 0)
  ) → (m = -1 ∨ m = -4) :=
sorry

end problem_1_problem_2_problem_3_l497_497697


namespace polynomial_factorization_l497_497564

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 1) * (x^2 + 6*x + 37) :=
  sorry

end polynomial_factorization_l497_497564


namespace inclusion_property_l497_497726

-- Define the sets A and B based on given conditions
def set_A (r : ℝ) : set (ℝ × ℝ) :=
  { p | ∃ x y, p = (x, y) ∧ x * (x - 1) + y * (y - 1) ≤ r }

def set_B (r : ℝ) : set (ℝ × ℝ) :=
  { p | ∃ x y, p = (x, y) ∧ x^2 + y^2 ≤ r^2 }

-- Define the inclusion property
theorem inclusion_property (r : ℝ) :
  (set_A r ⊆ set_B r) ↔ r ≥ sqrt 2 + 1 :=
sorry

end inclusion_property_l497_497726


namespace find_divisor_l497_497826

theorem find_divisor (n d k : ℤ) (h1 : n = k * d + 3) (h2 : n^2 % d = 4) : d = 5 :=
by
  sorry

end find_divisor_l497_497826


namespace solution_of_inequality_l497_497408

-- Let us define the inequality and the solution set
def inequality (x : ℝ) := (x - 1)^2023 - 2^2023 * x^2023 ≤ x + 1
def solution_set (x : ℝ) := x ≥ -1

-- The theorem statement to prove that the solution set matches the inequality
theorem solution_of_inequality :
  {x : ℝ | inequality x} = {x : ℝ | solution_set x} := sorry

end solution_of_inequality_l497_497408


namespace range_of_alpha_plus_3beta_l497_497224

variable (α β : ℝ)

axiom cond1 : -1 ≤ α + β ∧ α + β ≤ 1
axiom cond2 : 1 ≤ α + 2β ∧ α + 2β ≤ 3

theorem range_of_alpha_plus_3beta (h₁ : -1 ≤ α + β) (h₂ : α + β ≤ 1) 
                                  (h₃ : 1 ≤ α + 2β) (h₄ : α + 2β ≤ 3) :
  1 ≤ α + 3β ∧ α + 3β ≤ 7 :=
sorry

end range_of_alpha_plus_3beta_l497_497224


namespace fraction_rounded_equals_expected_l497_497916

-- Define the fraction and the rounding precision
def fraction : ℚ := 8 / 11
def precision : ℕ := 3

-- Define the expected rounded result
def expected_result : ℚ := 0.727

-- State the theorem to be proved
theorem fraction_rounded_equals_expected : 
  Real.floor (10^precision * fraction) / 10^precision = expected_result :=
sorry

end fraction_rounded_equals_expected_l497_497916


namespace sum_of_repeating_decimals_l497_497559

noncomputable def repeating_decimal_one_third := 0.333333...
noncomputable def repeating_decimal_two_thirds := 0.666666...
noncomputable def repeating_decimal_one := 0.999999...

/-- Prove that the sum of the repeating decimals 0.\overline{3}, 0.\overline{6}, and 0.\overline{9} is equal to 2
    given the conditions x = 0.\overline{3}, y = 0.\overline{6}, and z = 0.\overline{9}. 
--/
theorem sum_of_repeating_decimals :
  let x := repeating_decimal_one_third,
      y := repeating_decimal_two_thirds,
      z := repeating_decimal_one in
  x + y + z = 2 :=
by
  sorry

end sum_of_repeating_decimals_l497_497559


namespace log_evaluation_l497_497558

theorem log_evaluation :
  let a := Real.log 625 / Real.log 5
      b := Real.log 25 / Real.log 5
      c := Real.log (1 / 125) / Real.log 5
  in a - b + c = -1 := by
  let a := Real.log 625 / Real.log 5
  let b := Real.log 25 / Real.log 5
  let c := Real.log (1 / 125) / Real.log 5
  have ha : a = 4 := sorry
  have hb : b = 2 := sorry
  have hc : c = -3 := sorry
  calc
    a - b + c = 4 - 2 + (-3) := by rw [ha, hb, hc]
    ... = -1           := by norm_num

end log_evaluation_l497_497558


namespace exists_perfect_square_with_sequence_l497_497838

-- Defining a sequence of digits as an integer
def given_sequence (s : Nat) : Nat := s

-- Defining N1 as the given sequence of digits followed by 3k zeros
def N1 (s k : Nat) : Nat := s * 10^(3*k)

-- Defining N2 as the given sequence of digits followed by 3^k nines
def N2 (s k : Nat) : Nat := s * 10^(3*k) + 10^(3*k) - 1

-- Defining the condition that n^2 ≤ N1
def is_largest_square (n : Nat) (N1 : Nat) : Prop := n^2 ≤ N1

-- Defining the condition that n < 10^(2*k)
def bound_on_n (n k : Nat) : Prop := n < 10^(2*k)

-- The main theorem statement
theorem exists_perfect_square_with_sequence (s k : Nat) (n : Nat) 
  (h1 : is_largest_square n (N1 s k))
  (h2 : bound_on_n n k) : 
  ∃ m : Nat, (m * m > N1 s k) ∧ (m * m < N2 s k) ∧ (toString (N1 s k)).take (s.toString.length) = (toString (m * m)).take (s.toString.length) :=
by
  sorry

end exists_perfect_square_with_sequence_l497_497838


namespace unique_lines_count_l497_497252

theorem unique_lines_count :
  let S := {0, 1, 2, 3, 5}
  in ∃ (A B : ℤ), A ≠ B ∧ A ∈ S ∧ B ∈ S ∧ (S.card.choose 2 + 2 * (S.card - 1) = 14) :=
by
  let S := {0, 1, 2, 3, 5}
  have h1 : S.card = 5 := by simp [S]
  have h2 : S.card.choose 2 = 10 := by simp [nat.choose_eq, S]
  sorry  -- Proof omitted

end unique_lines_count_l497_497252


namespace congruent_triangles_in_100gon_l497_497505

theorem congruent_triangles_in_100gon :
  ∀ (P : Polygon) (parallelograms : set Parallelogram) (T1 T2 : Triangle), 
    P.is_regular 100 →
    (P.divide_into_parallelograms_and_triangles parallelograms {T1, T2}) →
    equal_blue_red_segments (P ∪ parallelograms ∪ {T1, T2}) →
    T1 ≅ T2 :=
by {
  -- Proof omitted
  sorry
}

end congruent_triangles_in_100gon_l497_497505


namespace tomato_acres_l497_497366

def total_land : ℝ := 3999.9999999999995
def cleared_percent : ℝ := 0.90
def grape_percent : ℝ := 0.60
def potato_percent : ℝ := 0.30
def cleared_land : ℝ := cleared_percent * total_land
def grape_land : ℝ := grape_percent * cleared_land
def potato_land : ℝ := potato_percent * cleared_land
def tomato_land : ℝ := cleared_land - (grape_land + potato_land)

theorem tomato_acres : tomato_land = 360 :=
by
  unfold cleared_land
  unfold grape_land
  unfold potato_land
  unfold tomato_land
  sorry

end tomato_acres_l497_497366


namespace find_length_of_AE_l497_497760

noncomputable def length_of_AE (AB CD AC : ℝ) (equal_areas : Prop) : ℝ :=
let AE := 8 in
AE

theorem find_length_of_AE (AB CD AC : ℝ) (equal_areas : Prop) :
  AB = 12 ∧ CD = 15 ∧ AC = 18 ∧ equal_areas → length_of_AE AB CD AC equal_areas = 8 :=
by {
  intro h,
  have AB_12 := h.1,
  have CD_15 := h.2.1,
  have AC_18 := h.2.2.1,
  have areas_equal := h.2.2.2,
  sorry
}

end find_length_of_AE_l497_497760


namespace round_fraction_to_three_decimal_l497_497912

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497912


namespace range_of_set_l497_497046

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l497_497046


namespace maximum_value_l497_497333

theorem maximum_value (a b c : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 2) (h3 : 0 ≤ b) (h4 : b ≤ 2) (h5 : 0 ≤ c) (h6 : c ≤ 2) :
  sqrt (a^2 * b^2 * c^2) + sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 8 :=
sorry

end maximum_value_l497_497333


namespace find_AD_l497_497833

noncomputable def A := 0
noncomputable def C := 3
noncomputable def B (x : ℝ) := C - x
noncomputable def D (x : ℝ) := A + 3 + x

-- conditions
def AC := 3
def BD := 4
def ratio_condition (x : ℝ) := (A + C - x - (A + 3)) / x = (A + 3 + x) / x

-- theorem statement
theorem find_AD (x : ℝ) (h1 : AC = 3) (h2 : BD = 4) (h3 : ratio_condition x) :
  D x = 6 :=
sorry

end find_AD_l497_497833


namespace range_of_set_l497_497062

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497062


namespace repeating_decimal_in_lowest_terms_l497_497200

def repeating_decimal_to_fraction (x : ℚ) (h : x = 6 + 182 / 999) : x = 6176 / 999 :=
by sorry

theorem repeating_decimal_in_lowest_terms : (6176, 999).gcd = 1 :=
by sorry

end repeating_decimal_in_lowest_terms_l497_497200


namespace max_truthful_dwarfs_l497_497953

theorem max_truthful_dwarfs :
  ∃ n : ℕ, n = 1 ∧ 
    ∀ heights : Fin 7 → ℕ, 
      (heights 0 = 60 ∨ heights 1 = 61 ∨ heights 2 = 62 ∨ heights 3 = 63 ∨ heights 4 = 64 ∨ heights 5 = 65 ∨ heights 6 = 66) → 
      (∀ i j : Fin 7, i < j → heights i > heights j) → 
      ( ∃ t : Fin 7 → bool, 
        ( ∀ i j : Fin 7, t i = tt → t j = tt → i = j) ∧ 
        ( ∀ i : Fin 7, t i = tt → heights i = 60 + i ) ∧
        #(t.to_list.count tt) = n ) :=
by
  use 1
  sorry

end max_truthful_dwarfs_l497_497953


namespace range_of_set_l497_497066

theorem range_of_set (nums : Finset ℝ) (mean_med_eq : (mean nums == 5) ∧ (median nums == 5)) (h : nums.min' (by { sorry }) = 2) : range nums = 6 :=
sorry

end range_of_set_l497_497066


namespace chessboard_not_divisible_by_10_l497_497820

theorem chessboard_not_divisible_by_10 (board : Fin 8 × Fin 8 → ℕ)
    (operation : (Fin 8 × Fin 8) → bool) -- an abstract representation of selecting a 3x3 or 4x4 square
    (increase : (Fin 8 × Fin 8) → (Fin 8 × Fin 8 → ℕ) → (Fin 8 × Fin 8 → ℕ)) -- an abstract representation of the increase operation
    (goal : (Fin 8 × Fin 8 → ℕ) → Prop) -- a representation of the goal of having all numbers divisible by 10
    : ¬(∃ op_seq : List (Fin 8 × Fin 8), (∀ op ∈ op_seq, operation op) ∧ goal (op_seq.foldl (λ b op, increase op b) board)) :=
by
  -- The proof will go here
  sorry

end chessboard_not_divisible_by_10_l497_497820


namespace factor_polynomial_l497_497648

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l497_497648


namespace factor_poly_eq_factored_form_l497_497600

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l497_497600


namespace gnomes_telling_the_truth_l497_497944

-- Conditions
def gnome_height_claims : List ℕ := [60, 61, 62, 63, 64, 65, 66]

-- Question and Proof Problem in Lean 4
theorem gnomes_telling_the_truth :
  (∀ (actual_heights : List ℕ), 
    actual_heights.length = 7 →
    (∀ i, i < 7 → i > 0 → 
    actual_heights.get! i > actual_heights.get! (i - 1) → gnome_height_claims.get! i ≠ actual_heights.get! i)) →
  -- conclusion
  (∃ count, count = 1) :=
by
  sorry

end gnomes_telling_the_truth_l497_497944


namespace factorization_identity_l497_497614

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497614


namespace multiple_of_C_share_l497_497375

noncomputable def find_multiple (A B C : ℕ) (total : ℕ) (mult : ℕ) (h1 : 4 * A = mult * C) (h2 : 5 * B = mult * C) (h3 : A + B + C = total) : ℕ :=
  mult

theorem multiple_of_C_share (A B : ℕ) (h1 : 4 * A = 10 * 160) (h2 : 5 * B = 10 * 160) (h3 : A + B + 160 = 880) : find_multiple A B 160 880 10 h1 h2 h3 = 10 :=
by
  sorry

end multiple_of_C_share_l497_497375


namespace f_property_f_one_sum_f_0_to_2017_l497_497228

noncomputable def f : ℤ → ℝ := sorry

theorem f_property (x y : ℤ) : f(x + y) + f(x - y) = 4 * f(x) * f(y) := sorry

theorem f_one : f 1 = 1 / 4 := sorry

theorem sum_f_0_to_2017 : (∑ i in finset.range 2018, f i) = 3 / 4 :=
by
  sorry

end f_property_f_one_sum_f_0_to_2017_l497_497228


namespace round_8_div_11_l497_497875

def fraction_to_real (n d : ℕ) : ℝ :=
  n / d

def round_to_n_decimals (x : ℝ) (n : ℕ) : ℝ :=
  Float.round (x * 10^n) / 10^n

theorem round_8_div_11 :
  round_to_n_decimals (fraction_to_real 8 11) 3 = 0.727 := by
  sorry

end round_8_div_11_l497_497875


namespace calculate_bankers_discount_l497_497461

variables (FV BG PV BD : ℝ)
variables (r t : ℝ)

-- Given conditions
def banker's_gain := 540
def rate_of_interest := 0.12
def time_period := 6
def face_value := 1073.56
def present_value := face_value / (1 + rate_of_interest)^time_period

-- Question to be proved (i.e., calculate BD)
def bankers_discount : ℝ := (face_value * rate_of_interest * time_period) / 100

-- The statement to be proved
theorem calculate_bankers_discount : bankers_discount ≈ 128.83 :=
begin
  sorry -- The proof is to be filled in
end

end calculate_bankers_discount_l497_497461


namespace range_of_set_l497_497079

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497079


namespace terminating_decimal_multiples_l497_497215

theorem terminating_decimal_multiples :
  (∃ n : ℕ, 20 = n ∧ ∀ m, 1 ≤ m ∧ m ≤ 180 → 
  (∃ k : ℕ, m = 9 * k)) :=
by
  sorry

end terminating_decimal_multiples_l497_497215


namespace polygon_to_triangle_l497_497039

theorem polygon_to_triangle {n : ℕ} (h : n > 4) :
  ∃ (a b c : ℕ), (a + b > c ∧ a + c > b ∧ b + c > a) :=
sorry

end polygon_to_triangle_l497_497039


namespace min_value_of_reciprocal_sum_l497_497294

theorem min_value_of_reciprocal_sum {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + 1 = 0 → False)
  (line_eq : ∀ (x y : ℝ), 2 * a * x - b * y + 2 = 0 → False)
  (chord_length : 2 * (a^2 + b^2)^0.5 = 4) :
  ∀ (c d : ℝ), a + b = 1 → (1 / a + 1 / b) = 4 :=
sorry

end min_value_of_reciprocal_sum_l497_497294


namespace vertex_locus_parabola_l497_497207

theorem vertex_locus_parabola (a c : ℝ) (ha : 0 < a) (hc : 0 < c) :
  ∃ S : set (ℝ × ℝ), (∀ (z : ℝ), (z, -a * (z / (2 * a))^2 + c) ∈ S) ∧
  (∀ (x_z y_z : ℝ), (x_z, y_z) ∈ S ↔ y_z = -a * x_z^2 + c) :=
by 
  sorry

end vertex_locus_parabola_l497_497207


namespace rounded_to_three_decimal_places_l497_497897

def rounded_fraction_eq : Prop :=
  let div_result := (8 : ℚ) / 11
  let rounded_result := Real.round (div_result * 1000) / 1000
  rounded_result = 0.727

theorem rounded_to_three_decimal_places : rounded_fraction_eq := by
  -- The proof would be inserted here
  sorry

end rounded_to_three_decimal_places_l497_497897


namespace range_of_set_of_three_numbers_l497_497136

noncomputable def range_of_three_numbers (a b c : ℝ) : ℝ :=
c - a

theorem range_of_set_of_three_numbers
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : a = 2)
  (h4 : b = 5)
  (h5 : (a + b + c) / 3 = 5) : 
  range_of_three_numbers a b c = 6 :=
by
  sorry

end range_of_set_of_three_numbers_l497_497136


namespace fraction_rounding_l497_497883

theorem fraction_rounding :
  (Real.ceil ((8 : ℝ) / 11 * 1000 - 0.5) / 1000 = 0.727) :=
by
  sorry

end fraction_rounding_l497_497883


namespace combined_area_of_removed_triangles_l497_497524

/-- Given a square with side length 12 units, if an isosceles right triangle is removed from each corner, 
    the combined area of the four removed triangles is 72 square units. -/
theorem combined_area_of_removed_triangles :
  let a := 12 in 
  let r (a b : ℝ) := a*a + b*b - 2*a*b = a*a - 2*a*b + b*b in
  let s (a b : ℝ) := a*a + b*b + 2*a*b = a*a + 2*a*b + b*b in
  (s 12 0 + r 12 0) / 2 = 72 := 
by sorry

end combined_area_of_removed_triangles_l497_497524


namespace round_fraction_to_three_decimal_l497_497908

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497908


namespace num_with_consecutive_ones_in_10_digits_l497_497271

open Nat

noncomputable def num_good_digits_without_consecutive_ones : ℕ → ℕ
| 0     => 1
| 1     => 2
| n + 2 => num_good_digits_without_consecutive_ones (n + 1) + num_good_digits_without_consecutive_ones n

theorem num_with_consecutive_ones_in_10_digits : 
  let total_digits := 2^10,
      good_digits := num_good_digits_without_consecutive_ones 10
  in total_digits - good_digits = 880 := 
by
  let total_digits := 2^10
  let good_digits := num_good_digits_without_consecutive_ones 10
  have good_digits_def : good_digits = 144 := by sorry
  rw good_digits_def
  calc
  1024 - 144 = 880 : by rfl

end num_with_consecutive_ones_in_10_digits_l497_497271


namespace factor_polynomial_l497_497628

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497628


namespace maximal_digit_sum_ratio_l497_497663

def digit_sum (n : ℕ) : ℕ := n.digits.sum

theorem maximal_digit_sum_ratio :
  ∃ n : ℕ, n > 0 ∧ (digit_sum n) / (digit_sum (16 * n)) = 13 ∧
    ∀ m : ℕ, m > 0 → (digit_sum m) / (digit_sum (16 * m)) ≤ 13 :=
by 
  sorry

end maximal_digit_sum_ratio_l497_497663


namespace find_factor_l497_497037

theorem find_factor (x : ℕ) (f : ℕ) (h1 : x = 9)
  (h2 : (2 * x + 6) * f = 72) : f = 3 := by
  sorry

end find_factor_l497_497037


namespace round_fraction_to_three_decimal_l497_497914

noncomputable def fraction := 8 / 11

def rounded_to_three_decimal_places (x : ℝ) : ℝ :=
  Real.floor (x * 1000) / 1000

theorem round_fraction_to_three_decimal :
  rounded_to_three_decimal_places fraction = 0.727 :=
by
  sorry

end round_fraction_to_three_decimal_l497_497914


namespace factorization_identity_l497_497606

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l497_497606


namespace range_of_numbers_is_six_l497_497115

-- Definitions for the problem conditions
def mean_of_three (a b c : ℝ) : ℝ := (a + b + c) / 3
def median_of_three (a b c : ℝ) : ℝ := b -- assuming a ≤ b ≤ c

-- Problem statement
theorem range_of_numbers_is_six (x : ℝ) (h1 : mean_of_three 2 5 x = 5) (h2 : median_of_three 2 5 x = 5) : (x - 2) = 6 :=
by sorry

end range_of_numbers_is_six_l497_497115


namespace range_of_set_is_six_l497_497089

noncomputable def problem_statement : Prop :=
  ∃ (S : Set ℕ), S.card = 3 ∧ S.mean = 5 ∧ S.median = 5 ∧ (2 ∈ S) ∧ (range S = 6)

theorem range_of_set_is_six : problem_statement :=
  sorry

end range_of_set_is_six_l497_497089


namespace range_of_set_l497_497094

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l497_497094


namespace find_rotation_u_v_l497_497430

def is_rotated (D E F D' E' F' : ℝ × ℝ)
  (u v : ℝ) (n : ℝ) : Prop :=
  let rotate (p : ℝ × ℝ) : ℝ × ℝ :=
    ((p.1 - u) * real.cos n - (p.2 - v) * real.sin n + u,
     (p.1 - u) * real.sin n + (p.2 - v) * real.cos n + v)
  in rotate D = D' ∧ rotate E = E' ∧ rotate F = F'

def vertices_DEF := ((0, 0), (0, 20), (30, 0))
def vertices_D'E'F' := ((30, 50), (60, 50), (30, 10))

theorem find_rotation_u_v :
  ∃ (n u v : ℝ), 0 < n ∧ n < 180 ∧ is_rotated (vertices_DEF.1) (vertices_DEF.2.1) (vertices_DEF.2.2)
                           (vertices_D'E'F'.1) (vertices_D'E'F'.2.1) (vertices_D'E'F'.2.2) u v n ∧ n + u + v = 140 :=
by
  sorry

end find_rotation_u_v_l497_497430


namespace range_of_set_l497_497072

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497072


namespace quadratic_function_count_l497_497717

theorem quadratic_function_count :
  let S := {0, 1, 2, 3, 4}
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ 0 ∧
  (∃ (fs : Finset (ℕ × ℕ × ℕ)), ∀ (x : ℕ × ℕ × ℕ), x ∈ fs ↔
  (∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ 0 ∧ x = (a, b, c))) ∧
  fs.card = 100 :=
by
  sorry

end quadratic_function_count_l497_497717


namespace kruskal_katona_l497_497338

variables {X : Type*} {A : set (set X)}

def family_of_subsets (A : set (set X)) (l : ℕ) :=
  ∀ a ∈ A, a.card = l

def binomial_representation (t l : ℕ) (a : ℕ → ℕ) : Prop :=
  t = nat.choose (a t) l + nat.choose (a (t - 1)) (t - 1) + 
      nat.choose (a (t - 2)) (t - 2) + 
      ... + 
      nat.choose (a m) m

theorem kruskal_katona (A : set (set X)) (l m : ℕ) (a : ℕ → ℕ) (t : ℕ) 
  (hA : family_of_subsets A l)
  (hRep : binomial_representation t l a) :
  (|A| ≥ ∑ i, nat.choose (a i) (i - 1)) :=
sorry

end kruskal_katona_l497_497338


namespace dining_hall_distance_equidistant_492_45_rods_l497_497030

theorem dining_hall_distance_equidistant_492_45_rods :
  ∀ (L G D : ℝ) (r : ℝ), r = 0 ∧
  (L = 400) ∧
  (sqrt (L ^ 2 + r ^ 2) = 700) ∧
  (D = sqrt ((L / 2) ^ 2 + (r / 2) ^ 2))
  → D = 492.45 :=
by sorry

end dining_hall_distance_equidistant_492_45_rods_l497_497030


namespace find_phi_and_increasing_interval_l497_497340

section MathProblem

-- Define the function f and the variables
variable (f : ℝ → ℝ) (φ : ℝ)
variable (φ_cond : -π < φ ∧ φ < 0)
variable (symmetry_condition : ∀ x, f x = f (-(x + π/12)))

noncomputable def fx : (ℝ → ℝ) := λ x, sin (2 * x + φ)

-- Statement of the proof problem
theorem find_phi_and_increasing_interval :
  (∀ x, f x = sin (2 * x + φ)) →
  symmetry_condition → 
  φ = -(π / 3) ∧ ∀ k : ℤ, ∀ x, [k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12] → exists I: set ℝ, ∀ x ∈ I, fx x < fx (x + 1) :=
begin
  intros fx_eq_symmetry_symetrics symmetry_cond,
  
  -- We skip the proof
  sorry
end

end MathProblem

end find_phi_and_increasing_interval_l497_497340


namespace factor_polynomial_l497_497633

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l497_497633


namespace dwarfs_truth_claims_l497_497969

/-- Seven dwarfs lined up by height, starting with the tallest.
Each dwarf claims a specific height as follows.
Verify that the largest number of dwarfs telling the truth is exactly 1. -/
theorem dwarfs_truth_claims :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  ∀ (actual_height: Fin 7 → ℕ), 
    (∀ i, actual_height i > actual_height (Fin.mk i (by linarith))) →
    ∀ claim_truth: Fin 7 → Prop, 
    (∀ i, claim_truth i ↔ actual_height i = heights[i]) →
    (∃ k, ∀ j, claim_truth j → j = k) :=
by
  intro heights actual_height h_order claim_truth h_claim_match
  existsi Fin.mk 0 (by linarith) -- Claim that only one can be telling the truth (arbitrary choice)
  sorry  -- Proof steps are omitted

end dwarfs_truth_claims_l497_497969


namespace sports_preference_related_to_gender_l497_497477

theorem sports_preference_related_to_gender :
  let a := 40
  let b := 20
  let c := 20
  let d := 30
  let n := 110
  let K_sq := n * ((a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  K_sq ≈ 7.82 →
  (K_sq > 6.635 ∧ K_sq < 7.879) →
  True :=
  by
    intros hK hInterval
    sorry

end sports_preference_related_to_gender_l497_497477


namespace blue_red_area_ratio_l497_497540

theorem blue_red_area_ratio (d1 d2 : ℝ) (h1 : d1 = 2) (h2 : d2 = 6) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let a_red := π * r1^2
  let a_large := π * r2^2
  let a_blue := a_large - a_red
  a_blue / a_red = 8 :=
by
  have r1 := d1 / 2
  have r2 := d2 / 2
  have a_red := π * r1^2
  have a_large := π * r2^2
  have a_blue := a_large - a_red
  sorry

end blue_red_area_ratio_l497_497540


namespace max_truthful_dwarfs_l497_497960

theorem max_truthful_dwarfs :
  let heights := [60, 61, 62, 63, 64, 65, 66] in
  (∃ n, n = 1 ∧ ∀ i, i < 7 → i ≠ n → heights[i] ≠ (heights[i] + 1)) :=
by
  let heights := [60, 61, 62, 63, 64, 65, 66]
  existsi 1
  intro i
  intro h
  intro hi
  sorry

end max_truthful_dwarfs_l497_497960


namespace fraction_rounded_to_decimal_l497_497932

theorem fraction_rounded_to_decimal :
  (Real.rat_of_near_rat 8 11).round_to_prec 3 11 = 0.727 := sorry

end fraction_rounded_to_decimal_l497_497932


namespace sequence_b_1_eq_252_l497_497544

theorem sequence_b_1_eq_252 (b : ℕ → ℕ) (h : ∀ n, 2 ≤ n → (∑ k in finset.range (n+1), b k) = (n^3 - n) * b n) (hb63 : b 63 = 63) : b 1 = 252 := 
sorry

end sequence_b_1_eq_252_l497_497544


namespace max_truthful_dwarfs_l497_497950

theorem max_truthful_dwarfs :
  ∃ n : ℕ, n = 1 ∧ 
    ∀ heights : Fin 7 → ℕ, 
      (heights 0 = 60 ∨ heights 1 = 61 ∨ heights 2 = 62 ∨ heights 3 = 63 ∨ heights 4 = 64 ∨ heights 5 = 65 ∨ heights 6 = 66) → 
      (∀ i j : Fin 7, i < j → heights i > heights j) → 
      ( ∃ t : Fin 7 → bool, 
        ( ∀ i j : Fin 7, t i = tt → t j = tt → i = j) ∧ 
        ( ∀ i : Fin 7, t i = tt → heights i = 60 + i ) ∧
        #(t.to_list.count tt) = n ) :=
by
  use 1
  sorry

end max_truthful_dwarfs_l497_497950


namespace skipping_ropes_l497_497004

theorem skipping_ropes (length1 length2 : ℕ) (h1 : length1 = 18) (h2 : length2 = 24) :
  ∃ (max_length : ℕ) (num_ropes : ℕ),
    max_length = Nat.gcd length1 length2 ∧
    max_length = 6 ∧
    num_ropes = length1 / max_length + length2 / max_length ∧
    num_ropes = 7 :=
by
  have max_length : ℕ := Nat.gcd length1 length2
  have num_ropes : ℕ := length1 / max_length + length2 / max_length
  use max_length, num_ropes
  sorry

end skipping_ropes_l497_497004


namespace find_coordinates_Q_l497_497421

-- Define the coordinates of the points as variables
variables (M N P Q : ℝ × ℝ × ℝ)

-- Define the points as given in the condition
def M : ℝ × ℝ × ℝ := (-2, 4, 1)
def N : ℝ × ℝ × ℝ := (0, -1, 3)
def P : ℝ × ℝ × ℝ := (4, 3, -2)

-- Assume the last point Q is some (x, y, z)
def Q := (2, 8, -4)

-- Prove that Q is correct given M, N, and P form a parallelogram
theorem find_coordinates_Q (Q_correct : Q = (2, 8, -4)) : 
  let midpoint_MP := ( (M.1 + P.1) / 2, (M.2 + P.2) / 2, (M.3 + P.3) / 2 )
  let midpoint_NQ := ( (N.1 + Q.1) / 2, (N.2 + Q.2) / 2, (N.3 + Q.3) / 2 )
  midpoint_MP = midpoint_NQ := by
  -- Here goes the logic of proof which we skip
  sorry

end find_coordinates_Q_l497_497421


namespace bisect_sides_of_triangle_l497_497394

theorem bisect_sides_of_triangle
  (A B C M N K D E : Point)
  (ABC_incircle_touch : IncircleTouches ABC M N K)
  (line_AD_parallel_NK : Parallel (Through A A) (Line NK))
  (line_AE_parallel_MN : Parallel (Through A A) (Line MN))
  (AD_intersect_MN_D : Intersect (Line AD) (Line MN) D)
  (AE_intersect_NK_E : Intersect (Line AE) (Line NK) E)
  : Bisect (Line DE) (Seg AB) ∧ Bisect (Line DE) (Seg AC) :=
by
  sorry

end bisect_sides_of_triangle_l497_497394


namespace range_of_set_l497_497075

theorem range_of_set (a b c : ℕ) (mean median smallest : ℕ) :
  mean = 5 ∧ median = 5 ∧ smallest = 2 ∧ (a + b + c) / 3 = mean ∧ 
  (a = smallest ∨ b = smallest ∨ c = smallest) ∧ 
  (((a ≤ b ∧ b ≤ c) ∨ (b ≤ a ∧ a ≤ c) ∨ (a ≤ c ∧ c ≤ b))) 
  → (max a (max b c) - min a (min b c)) = 6 :=
sorry

end range_of_set_l497_497075
