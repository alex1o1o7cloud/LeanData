import Mathlib

namespace NUMINAMATH_GPT_polynomial_value_l2029_202978

theorem polynomial_value (y : ℝ) (h : 4 * y^2 - 2 * y + 5 = 7) : 2 * y^2 - y + 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l2029_202978


namespace NUMINAMATH_GPT_min_value_problem_l2029_202949

noncomputable def min_value (a b c d e f : ℝ) := (2 / a) + (3 / b) + (9 / c) + (16 / d) + (25 / e) + (36 / f)

theorem min_value_problem 
  (a b c d e f : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) 
  (h_sum : a + b + c + d + e + f = 10) : 
  min_value a b c d e f >= (329 + 38 * Real.sqrt 6) / 10 := 
sorry

end NUMINAMATH_GPT_min_value_problem_l2029_202949


namespace NUMINAMATH_GPT_inequality_reciprocal_l2029_202957

theorem inequality_reciprocal (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 1 / (b - c) > 1 / (a - c) :=
sorry

end NUMINAMATH_GPT_inequality_reciprocal_l2029_202957


namespace NUMINAMATH_GPT_two_divides_a_squared_minus_a_three_divides_a_cubed_minus_a_l2029_202998

theorem two_divides_a_squared_minus_a (a : ℤ) : ∃ k₁ : ℤ, a^2 - a = 2 * k₁ :=
sorry

theorem three_divides_a_cubed_minus_a (a : ℤ) : ∃ k₂ : ℤ, a^3 - a = 3 * k₂ :=
sorry

end NUMINAMATH_GPT_two_divides_a_squared_minus_a_three_divides_a_cubed_minus_a_l2029_202998


namespace NUMINAMATH_GPT_solve_for_x_l2029_202923

-- Define the conditions
def percentage15_of_25 : ℝ := 0.15 * 25
def percentage12 (x : ℝ) : ℝ := 0.12 * x
def condition (x : ℝ) : Prop := percentage15_of_25 + percentage12 x = 9.15

-- The target statement to prove
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 45 :=
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_solve_for_x_l2029_202923


namespace NUMINAMATH_GPT_base_equivalence_l2029_202930

theorem base_equivalence :
  let n_7 := 4 * 7 + 3  -- 43 in base 7 expressed in base 10.
  ∃ d : ℕ, (3 * d + 4 = n_7) → d = 9 :=
by
  let n_7 := 31
  sorry

end NUMINAMATH_GPT_base_equivalence_l2029_202930


namespace NUMINAMATH_GPT_coefficient_of_pi_x_over_5_l2029_202915

-- Definition of the function where we find the coefficient
def coefficient_of_fraction (expr : ℝ) : ℝ := sorry

-- Statement with proof obligation
theorem coefficient_of_pi_x_over_5 :
  coefficient_of_fraction (π * x / 5) = π / 5 :=
sorry

end NUMINAMATH_GPT_coefficient_of_pi_x_over_5_l2029_202915


namespace NUMINAMATH_GPT_three_digit_integers_product_36_l2029_202922

theorem three_digit_integers_product_36 : 
  ∃ (num_digits : ℕ), num_digits = 21 ∧ 
    ∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (1 ≤ b ∧ b ≤ 9) ∧ 
      (1 ≤ c ∧ c ≤ 9) ∧ 
      (a * b * c = 36) → 
      num_digits = 21 :=
sorry

end NUMINAMATH_GPT_three_digit_integers_product_36_l2029_202922


namespace NUMINAMATH_GPT_length_of_square_cut_off_l2029_202925

theorem length_of_square_cut_off 
  (x : ℝ) 
  (h_eq : (48 - 2 * x) * (36 - 2 * x) * x = 5120) : 
  x = 8 := 
sorry

end NUMINAMATH_GPT_length_of_square_cut_off_l2029_202925


namespace NUMINAMATH_GPT_lenny_has_39_left_l2029_202960

/-- Define the initial amount Lenny has -/
def initial_amount : ℕ := 84

/-- Define the amount Lenny spent on video games -/
def spent_on_video_games : ℕ := 24

/-- Define the amount Lenny spent at the grocery store -/
def spent_on_groceries : ℕ := 21

/-- Define the total amount Lenny spent -/
def total_spent : ℕ := spent_on_video_games + spent_on_groceries

/-- Calculate the amount Lenny has left -/
def amount_left (initial amount_spent : ℕ) : ℕ :=
  initial - amount_spent

/-- The statement of our mathematical equivalent proof problem
  Prove that Lenny has $39 left given the initial amount,
  and the amounts spent on video games and groceries.
-/
theorem lenny_has_39_left :
  amount_left initial_amount total_spent = 39 :=
by
  -- Leave the proof as 'sorry' for now
  sorry

end NUMINAMATH_GPT_lenny_has_39_left_l2029_202960


namespace NUMINAMATH_GPT_coordinates_C_l2029_202953

theorem coordinates_C 
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) 
  (hA : A = (-1, 3)) 
  (hB : B = (11, 7))
  (hBC_AB : (C.1 - B.1, C.2 - B.2) = (2 / 3) • (B.1 - A.1, B.2 - A.2)) :
  C = (19, 29 / 3) :=
sorry

end NUMINAMATH_GPT_coordinates_C_l2029_202953


namespace NUMINAMATH_GPT_largest_n_for_perfect_square_l2029_202988

theorem largest_n_for_perfect_square :
  ∃ n : ℕ, 4 ^ 27 + 4 ^ 500 + 4 ^ n = k ^ 2 ∧ ∀ m : ℕ, 4 ^ 27 + 4 ^ 500 + 4 ^ m = l ^ 2 → m ≤ n  → n = 972 :=
sorry

end NUMINAMATH_GPT_largest_n_for_perfect_square_l2029_202988


namespace NUMINAMATH_GPT_find_x0_l2029_202979

-- Define a function f with domain [0, 3] and its inverse
variable {f : ℝ → ℝ}

-- Assume conditions for the inverse function
axiom f_inv_1 : ∀ x, 0 ≤ x ∧ x < 1 → 1 ≤ f x ∧ f x < 2
axiom f_inv_2 : ∀ x, 2 < x ∧ x ≤ 4 → 0 ≤ f x ∧ f x < 1

-- Domain condition
variables (x : ℝ) (hf_domain : 0 ≤ x ∧ x ≤ 3)

-- The main theorem
theorem find_x0 : (∃ x0: ℝ, f x0 = x0) → x = 2 :=
  sorry

end NUMINAMATH_GPT_find_x0_l2029_202979


namespace NUMINAMATH_GPT_roots_of_varying_signs_l2029_202980

theorem roots_of_varying_signs :
  (∃ x : ℝ, (4 * x^2 - 8 = 40 ∧ x != 0) ∧
           (∃ y : ℝ, (3 * y - 2)^2 = (y + 2)^2 ∧ y != 0) ∧
           (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ (z1 = 0 ∨ z2 = 0) ∧ x^3 - 8 * x^2 + 13 * x + 10 = 0)) :=
sorry

end NUMINAMATH_GPT_roots_of_varying_signs_l2029_202980


namespace NUMINAMATH_GPT_total_population_l2029_202963

variables (b g t : ℕ)

theorem total_population (h1 : b = 4 * g) (h2 : g = 5 * t) : b + g + t = 26 * t :=
sorry

end NUMINAMATH_GPT_total_population_l2029_202963


namespace NUMINAMATH_GPT_student_weekly_allowance_l2029_202951

theorem student_weekly_allowance (A : ℝ) (h1 : (4 / 15) * A = 1) : A = 3.75 :=
by
  sorry

end NUMINAMATH_GPT_student_weekly_allowance_l2029_202951


namespace NUMINAMATH_GPT_fraction_of_primes_is_prime_l2029_202905

theorem fraction_of_primes_is_prime
  (p q r : ℕ) 
  (hp : Nat.Prime p)
  (hq : Nat.Prime q)
  (hr : Nat.Prime r)
  (h : ∃ k : ℕ, p * q * r = k * (p + q + r)) :
  Nat.Prime (p * q * r / (p + q + r)) := 
sorry

end NUMINAMATH_GPT_fraction_of_primes_is_prime_l2029_202905


namespace NUMINAMATH_GPT_num_integers_for_polynomial_negative_l2029_202946

open Int

theorem num_integers_for_polynomial_negative :
  ∃ (set_x : Finset ℤ), set_x.card = 12 ∧ ∀ x ∈ set_x, (x^4 - 65 * x^2 + 64) < 0 :=
by
  sorry

end NUMINAMATH_GPT_num_integers_for_polynomial_negative_l2029_202946


namespace NUMINAMATH_GPT_simplify_expression_l2029_202942

theorem simplify_expression (x y : ℝ) (h : (x + 2)^2 + abs (y - 1/2) = 0) :
  (x - 2*y)*(x + 2*y) - (x - 2*y)^2 = -6 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_simplify_expression_l2029_202942


namespace NUMINAMATH_GPT_share_a_is_240_l2029_202935

def total_profit : ℕ := 630

def initial_investment_a : ℕ := 3000
def initial_investment_b : ℕ := 4000

def months_a1 : ℕ := 8
def months_a2 : ℕ := 4
def investment_a1 : ℕ := initial_investment_a * months_a1
def investment_a2 : ℕ := (initial_investment_a - 1000) * months_a2
def total_investment_a : ℕ := investment_a1 + investment_a2

def months_b1 : ℕ := 8
def months_b2 : ℕ := 4
def investment_b1 : ℕ := initial_investment_b * months_b1
def investment_b2 : ℕ := (initial_investment_b + 1000) * months_b2
def total_investment_b : ℕ := investment_b1 + investment_b2

def ratio_a : ℕ := 8
def ratio_b : ℕ := 13
def total_ratio : ℕ := ratio_a + ratio_b

noncomputable def share_a (total_profit : ℕ) (ratio_a ratio_total : ℕ) : ℕ :=
  (ratio_a * total_profit) / ratio_total

theorem share_a_is_240 :
  share_a total_profit ratio_a total_ratio = 240 :=
by
  sorry

end NUMINAMATH_GPT_share_a_is_240_l2029_202935


namespace NUMINAMATH_GPT_xy_solutions_l2029_202964

theorem xy_solutions : 
  ∀ (x y : ℕ), 0 < x → 0 < y →
  (xy ^ 2 + 7) ∣ (x^2 * y + x) →
  (x, y) = (7, 1) ∨ (x, y) = (14, 1) ∨ (x, y) = (35, 1) ∨ (x, y) = (7, 2) ∨ (∃ k : ℕ, x = 7 * k ∧ y = 7) :=
by
  sorry

end NUMINAMATH_GPT_xy_solutions_l2029_202964


namespace NUMINAMATH_GPT_total_students_left_l2029_202908

-- Definitions for given conditions
def initialBoys := 14
def initialGirls := 10
def boysDropOut := 4
def girlsDropOut := 3

-- The proof problem statement
theorem total_students_left : 
  initialBoys - boysDropOut + (initialGirls - girlsDropOut) = 17 := 
by
  sorry

end NUMINAMATH_GPT_total_students_left_l2029_202908


namespace NUMINAMATH_GPT_pear_weight_l2029_202910

theorem pear_weight
  (w_apple : ℕ)
  (p_weight_relation : 12 * w_apple = 8 * P + 5400)
  (apple_weight : w_apple = 530) :
  P = 120 :=
by
  -- sorry, proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_pear_weight_l2029_202910


namespace NUMINAMATH_GPT_always_positive_iff_k_gt_half_l2029_202939

theorem always_positive_iff_k_gt_half (k : ℝ) :
  (∀ x : ℝ, k * x^2 + x + k > 0) ↔ k > 0.5 :=
sorry

end NUMINAMATH_GPT_always_positive_iff_k_gt_half_l2029_202939


namespace NUMINAMATH_GPT_find_a_mul_b_l2029_202937

theorem find_a_mul_b (x y z a b : ℝ)
  (h1 : a = x)
  (h2 : b = y)
  (h3 : x + x = y * x)
  (h4 : b = z)
  (h5 : x + x = z * z)
  (h6 : y = 3)
  : a * b = 4 := by
  sorry

end NUMINAMATH_GPT_find_a_mul_b_l2029_202937


namespace NUMINAMATH_GPT_joe_time_to_school_l2029_202929

theorem joe_time_to_school
    (r_w : ℝ) -- Joe's walking speed
    (t_w : ℝ) -- Time to walk halfway
    (t_stop : ℝ) -- Time stopped at the store
    (r_running_factor : ℝ) -- Factor by which running speed is faster than walking speed
    (initial_walk_time_halfway : t_w = 10)
    (store_stop_time : t_stop = 3)
    (running_speed_factor : r_running_factor = 4) :
    t_w + t_stop + t_w / r_running_factor = 15.5 :=
by
    -- Implementation skipped, just verifying statement is correctly captured
    sorry

end NUMINAMATH_GPT_joe_time_to_school_l2029_202929


namespace NUMINAMATH_GPT_volume_of_displaced_water_square_of_displaced_water_volume_l2029_202968

-- Definitions for the conditions
def cube_side_length : ℝ := 10
def displaced_water_volume : ℝ := cube_side_length ^ 3
def displaced_water_volume_squared : ℝ := displaced_water_volume ^ 2

-- The Lean theorem statements proving the equivalence
theorem volume_of_displaced_water : displaced_water_volume = 1000 := by
  sorry

theorem square_of_displaced_water_volume : displaced_water_volume_squared = 1000000 := by
  sorry

end NUMINAMATH_GPT_volume_of_displaced_water_square_of_displaced_water_volume_l2029_202968


namespace NUMINAMATH_GPT_Julie_can_print_complete_newspapers_l2029_202906

def sheets_in_box_A : ℕ := 4 * 200
def sheets_in_box_B : ℕ := 3 * 350
def total_sheets : ℕ := sheets_in_box_A + sheets_in_box_B

def front_section_sheets : ℕ := 10
def sports_section_sheets : ℕ := 7
def arts_section_sheets : ℕ := 5
def events_section_sheets : ℕ := 3

def sheets_per_newspaper : ℕ := front_section_sheets + sports_section_sheets + arts_section_sheets + events_section_sheets

theorem Julie_can_print_complete_newspapers : total_sheets / sheets_per_newspaper = 74 := by
  sorry

end NUMINAMATH_GPT_Julie_can_print_complete_newspapers_l2029_202906


namespace NUMINAMATH_GPT_blind_box_problem_l2029_202934

theorem blind_box_problem (x y : ℕ) :
  x + y = 135 ∧ 2 * x = 3 * y :=
sorry

end NUMINAMATH_GPT_blind_box_problem_l2029_202934


namespace NUMINAMATH_GPT_probability_event_occurring_exactly_once_l2029_202941

theorem probability_event_occurring_exactly_once
  (P : ℝ)
  (h1 : ∀ n : ℕ, P ≥ 0 ∧ P ≤ 1) -- Probabilities are valid for all trials
  (h2 : (1 - (1 - P)^3) = 63 / 64) : -- Given condition for at least once
  (3 * P * (1 - P)^2 = 9 / 64) := 
by
  -- Here you would provide the proof steps using the conditions given.
  sorry

end NUMINAMATH_GPT_probability_event_occurring_exactly_once_l2029_202941


namespace NUMINAMATH_GPT_values_of_m_zero_rain_l2029_202970

def f (x y : ℝ) : ℝ := abs (x^3 + 2*x^2*y - 5*x*y^2 - 6*y^3)

theorem values_of_m_zero_rain :
  {m : ℝ | ∀ x : ℝ, f x (m * x) = 0} = {-1, 1/2, -1/3} :=
sorry

end NUMINAMATH_GPT_values_of_m_zero_rain_l2029_202970


namespace NUMINAMATH_GPT_f_for_negative_x_l2029_202983

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x * abs (x - 2) else 0  -- only assume the given case for x > 0

theorem f_for_negative_x (x : ℝ) (h : x < 0) : 
  f x = x * abs (x + 2) :=
by
  -- Sorry block to bypass the proof
  sorry

end NUMINAMATH_GPT_f_for_negative_x_l2029_202983


namespace NUMINAMATH_GPT_find_slope_of_line_l_l2029_202921

-- Define the vectors OA and OB
def OA : ℝ × ℝ := (4, 1)
def OB : ℝ × ℝ := (2, -3)

-- The slope k is such that the lengths of projections of OA and OB on line l are equal
theorem find_slope_of_line_l (k : ℝ) :
  (|4 + k| = |2 - 3 * k|) → (k = 3 ∨ k = -1/2) :=
by
  -- Intentionally leave the proof out
  sorry

end NUMINAMATH_GPT_find_slope_of_line_l_l2029_202921


namespace NUMINAMATH_GPT_bird_wings_l2029_202987

theorem bird_wings (P Pi C : ℕ) (h_total_money : 4 * 50 = 200)
  (h_total_cost : 30 * P + 20 * Pi + 15 * C = 200)
  (h_P_ge : P ≥ 1) (h_Pi_ge : Pi ≥ 1) (h_C_ge : C ≥ 1) :
  2 * (P + Pi + C) = 24 :=
sorry

end NUMINAMATH_GPT_bird_wings_l2029_202987


namespace NUMINAMATH_GPT_otimes_identity_l2029_202974

-- Define the operation ⊗
def otimes (k l : ℝ) : ℝ := k^2 - l^2

-- The goal is to show k ⊗ (k ⊗ k) = k^2 for any real number k
theorem otimes_identity (k : ℝ) : otimes k (otimes k k) = k^2 :=
by sorry

end NUMINAMATH_GPT_otimes_identity_l2029_202974


namespace NUMINAMATH_GPT_find_y_l2029_202994

theorem find_y : ∃ y : ℝ, 1.5 * y - 10 = 35 ∧ y = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2029_202994


namespace NUMINAMATH_GPT_cost_difference_l2029_202926

/-- The selling price and cost of pants -/
def selling_price : ℕ := 34
def store_cost : ℕ := 26

/-- The proof that the store paid 8 dollars less than the selling price -/
theorem cost_difference : selling_price - store_cost = 8 := by
  sorry

end NUMINAMATH_GPT_cost_difference_l2029_202926


namespace NUMINAMATH_GPT_gcd_sum_equality_l2029_202948

theorem gcd_sum_equality (n : ℕ) : 
  (Nat.gcd 6 n + Nat.gcd 8 (2 * n) = 10) ↔ 
  (∃ t : ℤ, n = 12 * t + 4 ∨ n = 12 * t + 6 ∨ n = 12 * t + 8) :=
by
  sorry

end NUMINAMATH_GPT_gcd_sum_equality_l2029_202948


namespace NUMINAMATH_GPT_F_shaped_to_cube_l2029_202928

-- Define the problem context in Lean 4
structure F_shaped_figure :=
  (squares : Finset (Fin 5) )

structure additional_squares :=
  (label : String )

def is_valid_configuration (f : F_shaped_figure) (s : additional_squares) : Prop :=
  -- This function should encapsulate the logic for checking the validity of a configuration
  sorry -- Implementation of validity check is omitted (replacing it with sorry)

-- The main theorem statement
theorem F_shaped_to_cube (f : F_shaped_figure) (squares: Finset additional_squares) : 
  ∃ valid_squares : Finset additional_squares, valid_squares.card = 3 ∧ 
    ∀ s ∈ valid_squares, is_valid_configuration f s := 
sorry

end NUMINAMATH_GPT_F_shaped_to_cube_l2029_202928


namespace NUMINAMATH_GPT_carrie_first_day_miles_l2029_202967

theorem carrie_first_day_miles
  (x : ℕ)
  (h1 : ∀ y : ℕ, y = x + 124) -- Second day
  (h2 : ∀ y : ℕ, y = 159) -- Third day
  (h3 : ∀ y : ℕ, y = 189) -- Fourth day
  (h4 : ∀ z : ℕ, z = 106) -- Phone charge interval
  (h5 : ∀ n : ℕ, n = 7) -- Number of charges
  (h_total : 106 * 7 = x + (x + 124) + 159 + 189)
  : x = 135 :=
by sorry

end NUMINAMATH_GPT_carrie_first_day_miles_l2029_202967


namespace NUMINAMATH_GPT_prove_k_range_l2029_202985

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x - b * Real.log x

theorem prove_k_range (a b k : ℝ) (h1 : a - b = 1) (h2 : f 1 a b = 2) :
  (∀ x ≥ 1, f x a b ≥ k * x) → k ≤ 2 - 1 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_prove_k_range_l2029_202985


namespace NUMINAMATH_GPT_pat_stickers_l2029_202938

theorem pat_stickers (stickers_given_away stickers_left : ℝ) 
(h_given_away : stickers_given_away = 22.0)
(h_left : stickers_left = 17.0) : 
(stickers_given_away + stickers_left = 39) :=
by
  sorry

end NUMINAMATH_GPT_pat_stickers_l2029_202938


namespace NUMINAMATH_GPT_focus_parabola_l2029_202973

theorem focus_parabola (f : ℝ) (d : ℝ) (y : ℝ) :
  (∀ y, ((- (1 / 8) * y^2 - f) ^ 2 + y^2 = (- (1 / 8) * y^2 - d) ^ 2)) → 
  (d - f = 4) → 
  (f^2 = d^2) → 
  f = -2 :=
by
  sorry

end NUMINAMATH_GPT_focus_parabola_l2029_202973


namespace NUMINAMATH_GPT_smallest_int_neither_prime_nor_square_no_prime_lt_70_l2029_202975

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬ p ∣ n

theorem smallest_int_neither_prime_nor_square_no_prime_lt_70
  (n : ℕ) : 
  n = 5183 ∧ ¬ is_prime n ∧ ¬ is_square n ∧ has_no_prime_factor_less_than n 70 ∧
  (∀ m : ℕ, 0 < m → m < 5183 →
    ¬ (¬ is_prime m ∧ ¬ is_square m ∧ has_no_prime_factor_less_than m 70)) :=
by sorry

end NUMINAMATH_GPT_smallest_int_neither_prime_nor_square_no_prime_lt_70_l2029_202975


namespace NUMINAMATH_GPT_min_value_of_quadratic_function_min_attained_at_negative_two_l2029_202966

def quadratic_function (x : ℝ) : ℝ := 3 * (x + 2)^2 - 5

theorem min_value_of_quadratic_function : ∀ x : ℝ, quadratic_function x ≥ -5 :=
by
  sorry

theorem min_attained_at_negative_two : quadratic_function (-2) = -5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_function_min_attained_at_negative_two_l2029_202966


namespace NUMINAMATH_GPT_bounded_region_area_l2029_202969

theorem bounded_region_area : 
  (∀ x y : ℝ, (y^2 + 4*x*y + 50*|x| = 500) → (x ≥ 0 ∧ y = 25 - 4*x) ∨ (x ≤ 0 ∧ y = -12.5 - 4*x)) →
  ∃ (A : ℝ), A = 156.25 :=
by
  sorry

end NUMINAMATH_GPT_bounded_region_area_l2029_202969


namespace NUMINAMATH_GPT_like_terms_monomials_l2029_202999

theorem like_terms_monomials (a b : ℕ) (x y : ℝ) (c : ℝ) (H1 : x^(a+1) * y^3 = c * y^b * x^2) : a = 1 ∧ b = 3 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_like_terms_monomials_l2029_202999


namespace NUMINAMATH_GPT_ratio_of_perimeters_l2029_202911

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_l2029_202911


namespace NUMINAMATH_GPT_least_upper_bound_neg_expression_l2029_202902

noncomputable def least_upper_bound : ℝ :=
  - (9 / 2)

theorem least_upper_bound_neg_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  ∃ M, M = least_upper_bound ∧
  ∀ x, (∀ a b, 0 < a → 0 < b → a + b = 1 → x ≤ - (1 / (2 * a)) - (2 / b)) ↔ x ≤ M :=
sorry

end NUMINAMATH_GPT_least_upper_bound_neg_expression_l2029_202902


namespace NUMINAMATH_GPT_number_greater_by_l2029_202984

def question (a b : Int) : Int := a + b

theorem number_greater_by (a b : Int) : question a b = -11 :=
  by
    sorry

-- Use specific values from the provided problem:
example : question -5 -6 = -11 :=
  by
    sorry

end NUMINAMATH_GPT_number_greater_by_l2029_202984


namespace NUMINAMATH_GPT_quadratic_real_equal_roots_l2029_202944

theorem quadratic_real_equal_roots (k : ℝ) :
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 15 = 0 ∧ ∀ y : ℝ, (3 * y^2 - k * y + 2 * y + 15 = 0 → y = x)) ↔
  (k = 6 * Real.sqrt 5 + 2 ∨ k = -6 * Real.sqrt 5 + 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_equal_roots_l2029_202944


namespace NUMINAMATH_GPT_a_alone_can_finish_job_l2029_202927

def work_in_one_day (A B : ℕ) : Prop := 1/A + 1/B = 1/40

theorem a_alone_can_finish_job (A B : ℕ)
  (work_rate : work_in_one_day A B) 
  (together_10_days : 10 * (1/A + 1/B) = 1/4) 
  (a_21_days : 21 * (1/A) = 3/4) : 
  A = 28 := 
sorry

end NUMINAMATH_GPT_a_alone_can_finish_job_l2029_202927


namespace NUMINAMATH_GPT_more_red_flowers_than_white_l2029_202924

-- Definitions based on given conditions
def yellow_and_white := 13
def red_and_yellow := 17
def red_and_white := 14
def blue_and_yellow := 16

-- Definitions based on the requirements of the problem
def red_flowers := red_and_yellow + red_and_white
def white_flowers := yellow_and_white + red_and_white

-- Theorem to prove the number of more flowers containing red than white
theorem more_red_flowers_than_white : red_flowers - white_flowers = 4 := by
  sorry

end NUMINAMATH_GPT_more_red_flowers_than_white_l2029_202924


namespace NUMINAMATH_GPT_original_cost_price_l2029_202936

theorem original_cost_price 
  (C SP SP_new C_new : ℝ)
  (h1 : SP = 1.05 * C)
  (h2 : C_new = 0.95 * C)
  (h3 : SP_new = SP - 8)
  (h4 : SP_new = 1.045 * C_new) :
  C = 1600 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_price_l2029_202936


namespace NUMINAMATH_GPT_additional_flour_minus_salt_l2029_202940

structure CakeRecipe where
  flour    : ℕ
  sugar    : ℕ
  salt     : ℕ

def MaryHasAdded (cups_flour : ℕ) (cups_sugar : ℕ) (cups_salt : ℕ) : Prop :=
  cups_flour = 2 ∧ cups_sugar = 0 ∧ cups_salt = 0

variable (r : CakeRecipe)

theorem additional_flour_minus_salt (H : MaryHasAdded 2 0 0) : 
  (r.flour - 2) - r.salt = 3 :=
sorry

end NUMINAMATH_GPT_additional_flour_minus_salt_l2029_202940


namespace NUMINAMATH_GPT_find_n_l2029_202909

variable {a_n : ℕ → ℤ}
variable (a2 : ℤ) (an : ℤ) (d : ℤ) (n : ℕ)

def arithmetic_sequence (a2 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a2 + (n - 2) * d

theorem find_n 
  (h1 : a2 = 12)
  (h2 : an = -20)
  (h3 : d = -2)
  : n = 18 := by
  sorry

end NUMINAMATH_GPT_find_n_l2029_202909


namespace NUMINAMATH_GPT_statement_C_correct_l2029_202919

theorem statement_C_correct (a b : ℝ) (h1 : a < b) (h2 : a * b ≠ 0) : (1 / a) > (1 / b) :=
sorry

end NUMINAMATH_GPT_statement_C_correct_l2029_202919


namespace NUMINAMATH_GPT_gibi_percentage_is_59_l2029_202996

-- Define the conditions
def max_score := 700
def avg_score := 490
def jigi_percent := 55
def mike_percent := 99
def lizzy_percent := 67

def jigi_score := (jigi_percent * max_score) / 100
def mike_score := (mike_percent * max_score) / 100
def lizzy_score := (lizzy_percent * max_score) / 100

def total_score := 4 * avg_score
def gibi_score := total_score - (jigi_score + mike_score + lizzy_score)

def gibi_percent := (gibi_score * 100) / max_score

-- The proof goal
theorem gibi_percentage_is_59 : gibi_percent = 59 := by
  sorry

end NUMINAMATH_GPT_gibi_percentage_is_59_l2029_202996


namespace NUMINAMATH_GPT_frank_money_l2029_202903

theorem frank_money (X : ℝ) (h1 : (3/4) * (4/5) * X = 360) : X = 600 :=
sorry

end NUMINAMATH_GPT_frank_money_l2029_202903


namespace NUMINAMATH_GPT_system_nonzero_solution_l2029_202989

-- Definition of the game setup and conditions
def initial_equations (a b c : ℤ) (x y z : ℤ) : Prop :=
  (a * x + b * y + c * z = 0) ∧
  (a * x + b * y + c * z = 0) ∧
  (a * x + b * y + c * z = 0)

-- The main proposition statement in Lean
theorem system_nonzero_solution :
  ∀ (a b c : ℤ), ∃ (x y z : ℤ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∧ initial_equations a b c x y z :=
by
  sorry

end NUMINAMATH_GPT_system_nonzero_solution_l2029_202989


namespace NUMINAMATH_GPT_eq_x_in_terms_of_y_l2029_202993

theorem eq_x_in_terms_of_y (x y : ℝ) (h : 2 * x + y = 5) : x = (5 - y) / 2 := by
  sorry

end NUMINAMATH_GPT_eq_x_in_terms_of_y_l2029_202993


namespace NUMINAMATH_GPT_sum_of_powers_l2029_202976

theorem sum_of_powers : (-1: ℤ) ^ 2006 - (-1) ^ 2007 + 1 ^ 2008 + 1 ^ 2009 - 1 ^ 2010 = 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_l2029_202976


namespace NUMINAMATH_GPT_arithmetic_sum_of_11_terms_l2029_202982

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α) (d : α)

def arithmetic_sequence (a : ℕ → α) (a₁ : α) (d : α) : Prop :=
∀ n, a n = a₁ + n * d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
(n + 1) * (a 0 + a n) / 2

theorem arithmetic_sum_of_11_terms
  (a₁ d : α)
  (a : ℕ → α)
  (h_seq : arithmetic_sequence a a₁ d)
  (h_cond : a 8 = (1 / 2) * a 11 + 3) :
  sum_first_n_terms a 10 = 66 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_of_11_terms_l2029_202982


namespace NUMINAMATH_GPT_find_f2_l2029_202912

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder function definition

theorem find_f2 (h : ∀ x : ℝ, f x + 2 * f (1 - x) = x^3 + 1) : f 2 = -3 :=
by
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_find_f2_l2029_202912


namespace NUMINAMATH_GPT_solve_quadratic_equation_l2029_202920

theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 - 4 * x = 6 - 3 * x) ↔ 
  (x = -3/2 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l2029_202920


namespace NUMINAMATH_GPT_investment_calculation_l2029_202933

theorem investment_calculation
    (R Trishul Vishal Alok Harshit : ℝ)
    (hTrishul : Trishul = 0.9 * R)
    (hVishal : Vishal = 0.99 * R)
    (hAlok : Alok = 1.035 * Trishul)
    (hHarshit : Harshit = 0.95 * Vishal)
    (hTotal : R + Trishul + Vishal + Alok + Harshit = 22000) :
  R = 22000 / 3.8655 ∧
  Trishul = 0.9 * R ∧
  Vishal = 0.99 * R ∧
  Alok = 1.035 * Trishul ∧
  Harshit = 0.95 * Vishal ∧
  R + Trishul + Vishal + Alok + Harshit = 22000 :=
sorry

end NUMINAMATH_GPT_investment_calculation_l2029_202933


namespace NUMINAMATH_GPT_part1_part2_l2029_202947

section
variable (k : ℝ)

/-- Part 1: Range of k -/
def discriminant_eqn (k : ℝ) := (2 * k - 1) ^ 2 - 4 * (k ^ 2 - 1)

theorem part1 (h : discriminant_eqn k ≥ 0) : k ≤ 5 / 4 :=
by sorry

/-- Part 2: Value of k when x₁ and x₂ satisfy the given condition -/
def x1_x2_eqn (k x1 x2 : ℝ) := x1 ^ 2 + x2 ^ 2 = 16 + x1 * x2

def vieta (k : ℝ) (x1 x2 : ℝ) :=
  x1 + x2 = 1 - 2 * k ∧ x1 * x2 = k ^ 2 - 1

theorem part2 (x1 x2 : ℝ) (h1 : vieta k x1 x2) (h2 : x1_x2_eqn k x1 x2) : k = -2 :=
by sorry

end

end NUMINAMATH_GPT_part1_part2_l2029_202947


namespace NUMINAMATH_GPT_power_division_l2029_202986

theorem power_division (a : ℝ) (h : a ≠ 0) : ((-a)^6) / (a^3) = a^3 := by
  sorry

end NUMINAMATH_GPT_power_division_l2029_202986


namespace NUMINAMATH_GPT_pages_to_read_tomorrow_l2029_202914

-- Define the conditions
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Lean statement for the proof problem
theorem pages_to_read_tomorrow : (total_pages - (pages_yesterday + pages_today) = 35) :=
by
  let yesterday := pages_yesterday
  let today := pages_today
  let read_so_far := yesterday + today
  have read_so_far_eq : yesterday + today = 65 := by sorry
  have total_eq : total_pages - read_so_far = 35 := by sorry
  exact total_eq

end NUMINAMATH_GPT_pages_to_read_tomorrow_l2029_202914


namespace NUMINAMATH_GPT_longest_segment_is_CD_l2029_202932

-- Define points A, B, C, D
def A := (-3, 0)
def B := (0, 2)
def C := (3, 0)
def D := (0, -1)

-- Angles in triangle ABD
def angle_ABD := 35
def angle_BAD := 95
def angle_ADB := 50

-- Angles in triangle BCD
def angle_BCD := 55
def angle_BDC := 60
def angle_CBD := 65

-- Length comparison conclusion from triangle ABD
axiom compare_lengths_ABD : ∀ (AD AB BD : ℝ), AD < AB ∧ AB < BD

-- Length comparison conclusion from triangle BCD
axiom compare_lengths_BCD : ∀ (BC BD CD : ℝ), BC < BD ∧ BD < CD

-- Combine results
theorem longest_segment_is_CD : ∀ (AD AB BD BC CD : ℝ), AD < AB → AB < BD → BC < BD → BD < CD → CD ≥ AD ∧ CD ≥ AB ∧ CD ≥ BD ∧ CD ≥ BC :=
by
  intros AD AB BD BC CD h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_longest_segment_is_CD_l2029_202932


namespace NUMINAMATH_GPT_fraction_inequality_l2029_202990

-- Given the conditions
variables {c x y : ℝ} (h1 : c > x) (h2 : x > y) (h3 : y > 0)

-- Prove that \frac{x}{c-x} > \frac{y}{c-y}
theorem fraction_inequality (h4 : c > 0) : (x / (c - x)) > (y / (c - y)) :=
by {
  sorry  -- Proof to be completed
}

end NUMINAMATH_GPT_fraction_inequality_l2029_202990


namespace NUMINAMATH_GPT_choosing_one_student_is_50_l2029_202962

-- Define the number of male students and female students
def num_male_students : Nat := 26
def num_female_students : Nat := 24

-- Define the total number of ways to choose one student
def total_ways_to_choose_one_student : Nat := num_male_students + num_female_students

-- Theorem statement proving the total number of ways to choose one student is 50
theorem choosing_one_student_is_50 : total_ways_to_choose_one_student = 50 := by
  sorry

end NUMINAMATH_GPT_choosing_one_student_is_50_l2029_202962


namespace NUMINAMATH_GPT_points_on_opposite_sides_l2029_202945

theorem points_on_opposite_sides (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by sorry

end NUMINAMATH_GPT_points_on_opposite_sides_l2029_202945


namespace NUMINAMATH_GPT_average_percent_increase_per_year_l2029_202952

-- Definitions and conditions
def initialPopulation : ℕ := 175000
def finalPopulation : ℕ := 297500
def numberOfYears : ℕ := 10

-- Statement to prove
theorem average_percent_increase_per_year : 
  ((finalPopulation - initialPopulation) / numberOfYears : ℚ) / initialPopulation * 100 = 7 := by
  sorry

end NUMINAMATH_GPT_average_percent_increase_per_year_l2029_202952


namespace NUMINAMATH_GPT_quadratic_rewrite_correct_a_b_c_l2029_202931

noncomputable def quadratic_rewrite (x : ℝ) : ℝ := -6*x^2 + 36*x + 216

theorem quadratic_rewrite_correct_a_b_c :
  ∃ a b c : ℝ, quadratic_rewrite x = a * (x + b)^2 + c ∧ a + b + c = 261 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_rewrite_correct_a_b_c_l2029_202931


namespace NUMINAMATH_GPT_students_more_than_pets_l2029_202991

-- Definition of given conditions
def num_students_per_classroom := 20
def num_rabbits_per_classroom := 2
def num_goldfish_per_classroom := 3
def num_classrooms := 5

-- Theorem stating the proof problem
theorem students_more_than_pets :
  let total_students := num_students_per_classroom * num_classrooms
  let total_pets := (num_rabbits_per_classroom + num_goldfish_per_classroom) * num_classrooms
  total_students - total_pets = 75 := by
  sorry

end NUMINAMATH_GPT_students_more_than_pets_l2029_202991


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2029_202900

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 3) * (1 - x) ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2029_202900


namespace NUMINAMATH_GPT_beads_per_necklace_correct_l2029_202954
-- Importing the necessary library.

-- Defining the given number of necklaces and total beads.
def number_of_necklaces : ℕ := 11
def total_beads : ℕ := 308

-- Stating the proof goal as a theorem.
theorem beads_per_necklace_correct : (total_beads / number_of_necklaces) = 28 := 
by
  sorry

end NUMINAMATH_GPT_beads_per_necklace_correct_l2029_202954


namespace NUMINAMATH_GPT_derivative_of_reciprocal_at_one_l2029_202950

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_of_reciprocal_at_one : (deriv f 1) = -1 :=
by {
    sorry
}

end NUMINAMATH_GPT_derivative_of_reciprocal_at_one_l2029_202950


namespace NUMINAMATH_GPT_radhika_total_games_l2029_202943

-- Define the conditions
def giftsOnChristmas := 12
def giftsOnBirthday := 8
def alreadyOwned := (giftsOnChristmas + giftsOnBirthday) / 2
def totalGifts := giftsOnChristmas + giftsOnBirthday
def expectedTotalGames := totalGifts + alreadyOwned

-- Define the proof statement
theorem radhika_total_games : 
  giftsOnChristmas = 12 ∧ giftsOnBirthday = 8 ∧ alreadyOwned = 10 
  ∧ totalGifts = 20 ∧ expectedTotalGames = 30 :=
by 
  sorry

end NUMINAMATH_GPT_radhika_total_games_l2029_202943


namespace NUMINAMATH_GPT_cakes_baker_made_initially_l2029_202918

theorem cakes_baker_made_initially (x : ℕ) (h1 : x - 75 + 76 = 111) : x = 110 :=
by
  sorry

end NUMINAMATH_GPT_cakes_baker_made_initially_l2029_202918


namespace NUMINAMATH_GPT_find_a5_l2029_202961

variable {a : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n+1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a n < a (n+1)

def condition1 (a : ℕ → ℝ) : Prop :=
  a 5 ^ 2 = a 10

def condition2 (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * (a n + a (n+2)) = 5 * a (n+1)

theorem find_a5 (h1 : is_geometric_sequence a q) (h2 : is_increasing_sequence a) (h3 : condition1 a) (h4 : condition2 a) : 
  a 5 = 32 :=
sorry

end NUMINAMATH_GPT_find_a5_l2029_202961


namespace NUMINAMATH_GPT_monthly_rent_l2029_202913

-- Definitions based on the given conditions
def length_ft : ℕ := 360
def width_ft : ℕ := 1210
def sq_feet_per_acre : ℕ := 43560
def cost_per_acre_per_month : ℕ := 60

-- Statement of the problem
theorem monthly_rent : (length_ft * width_ft / sq_feet_per_acre) * cost_per_acre_per_month = 600 := sorry

end NUMINAMATH_GPT_monthly_rent_l2029_202913


namespace NUMINAMATH_GPT_smallest_value_floor_l2029_202907

theorem smallest_value_floor (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  (⌊(2 * a + b) / c⌋ + ⌊(b + 2 * c) / a⌋ + ⌊(c + 2 * a) / b⌋) = 9 :=
sorry

end NUMINAMATH_GPT_smallest_value_floor_l2029_202907


namespace NUMINAMATH_GPT_bus_speed_calculation_l2029_202955

noncomputable def bus_speed_excluding_stoppages : ℝ :=
  let effective_speed_with_stoppages := 50 -- kmph
  let stoppage_time_in_minutes := 13.125 -- minutes per hour
  let stoppage_time_in_hours := stoppage_time_in_minutes / 60 -- convert to hours
  let effective_moving_time := 1 - stoppage_time_in_hours -- effective moving time in one hour
  let bus_speed := (effective_speed_with_stoppages * 60) / (60 - stoppage_time_in_minutes) -- calculate bus speed
  bus_speed

theorem bus_speed_calculation : bus_speed_excluding_stoppages = 64 := by
  sorry

end NUMINAMATH_GPT_bus_speed_calculation_l2029_202955


namespace NUMINAMATH_GPT_volume_region_between_concentric_spheres_l2029_202904

open Real

theorem volume_region_between_concentric_spheres (r1 r2 : ℝ) (h_r1 : r1 = 4) (h_r2 : r2 = 8) :
  (4 / 3 * π * r2^3 - 4 / 3 * π * r1^3) = 1792 / 3 * π :=
by
  sorry

end NUMINAMATH_GPT_volume_region_between_concentric_spheres_l2029_202904


namespace NUMINAMATH_GPT_problem_solution_inf_problem_solution_prime_l2029_202956

-- Definitions based on the given conditions and problem statement
def is_solution_inf (m : ℕ) : Prop := 3^m ∣ 2^(3^m) + 1

def is_solution_prime (n : ℕ) : Prop := n.Prime ∧ n ∣ 2^n + 1

-- Lean statement for the math proof problem
theorem problem_solution_inf : ∀ m : ℕ, m ≥ 0 → is_solution_inf m := sorry

theorem problem_solution_prime : ∀ n : ℕ, n.Prime → is_solution_prime n → n = 3 := sorry

end NUMINAMATH_GPT_problem_solution_inf_problem_solution_prime_l2029_202956


namespace NUMINAMATH_GPT_faculty_after_reduction_is_correct_l2029_202977

-- Define the original number of faculty members
def original_faculty : ℝ := 253.25

-- Define the reduction percentage as a decimal
def reduction_percentage : ℝ := 0.23

-- Calculate the reduction amount
def reduction_amount : ℝ := original_faculty * reduction_percentage

-- Define the rounded reduction amount
def rounded_reduction_amount : ℝ := 58.25

-- Calculate the number of professors after the reduction
def professors_after_reduction : ℝ := original_faculty - rounded_reduction_amount

-- Statement to be proven: the number of professors after the reduction is 195
theorem faculty_after_reduction_is_correct : professors_after_reduction = 195 := by
  sorry

end NUMINAMATH_GPT_faculty_after_reduction_is_correct_l2029_202977


namespace NUMINAMATH_GPT_fraction_eq_l2029_202965

def at_op (a b : ℝ) : ℝ := a * b - a * b^2
def hash_op (a b : ℝ) : ℝ := a^2 + b - a^2 * b

theorem fraction_eq :
  (at_op 8 3) / (hash_op 8 3) = 48 / 125 :=
by sorry

end NUMINAMATH_GPT_fraction_eq_l2029_202965


namespace NUMINAMATH_GPT_sum_of_squares_eq_2_l2029_202997

theorem sum_of_squares_eq_2 (a b : ℝ) 
  (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_eq_2_l2029_202997


namespace NUMINAMATH_GPT_circle_radius_l2029_202972

theorem circle_radius (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l2029_202972


namespace NUMINAMATH_GPT_preferred_apples_percentage_l2029_202959

theorem preferred_apples_percentage (A B C O G : ℕ) (total freq_apples : ℕ)
  (hA : A = 70) (hB : B = 50) (hC: C = 30) (hO: O = 50) (hG: G = 40)
  (htotal : total = A + B + C + O + G)
  (hfa : freq_apples = A) :
  (freq_apples / total : ℚ) * 100 = 29 :=
by sorry

end NUMINAMATH_GPT_preferred_apples_percentage_l2029_202959


namespace NUMINAMATH_GPT_solution_set_inequality_l2029_202992

   theorem solution_set_inequality (a : ℝ) : (∀ x : ℝ, x^2 - 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) :=
   sorry
   
end NUMINAMATH_GPT_solution_set_inequality_l2029_202992


namespace NUMINAMATH_GPT_max_product_of_two_integers_whose_sum_is_300_l2029_202995

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end NUMINAMATH_GPT_max_product_of_two_integers_whose_sum_is_300_l2029_202995


namespace NUMINAMATH_GPT_integer_solution_pair_l2029_202901

theorem integer_solution_pair (x y : ℤ) (h : x^2 + x * y = y^2) : (x = 0 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_pair_l2029_202901


namespace NUMINAMATH_GPT_find_X_l2029_202916

-- Define the variables for income, tax, and the variable X
def income := 58000
def tax := 8000

-- Define the tax formula as per the problem
def tax_formula (X : ℝ) : ℝ :=
  0.11 * X + 0.20 * (income - X)

-- The theorem we want to prove
theorem find_X :
  ∃ X : ℝ, tax_formula X = tax ∧ X = 40000 :=
sorry

end NUMINAMATH_GPT_find_X_l2029_202916


namespace NUMINAMATH_GPT_extracurricular_hours_l2029_202917

theorem extracurricular_hours :
  let soccer_hours_per_day := 2
  let soccer_days := 3
  let band_hours_per_day := 1.5
  let band_days := 2
  let total_soccer_hours := soccer_hours_per_day * soccer_days
  let total_band_hours := band_hours_per_day * band_days
  total_soccer_hours + total_band_hours = 9 := 
by
  -- The proof steps go here.
  sorry

end NUMINAMATH_GPT_extracurricular_hours_l2029_202917


namespace NUMINAMATH_GPT_edward_lives_left_l2029_202971

theorem edward_lives_left : 
  let initial_lives := 50
  let stage1_loss := 18
  let stage1_gain := 7
  let stage2_loss := 10
  let stage2_gain := 5
  let stage3_loss := 13
  let stage3_gain := 2
  let final_lives := initial_lives - stage1_loss + stage1_gain - stage2_loss + stage2_gain - stage3_loss + stage3_gain
  final_lives = 23 :=
by
  sorry

end NUMINAMATH_GPT_edward_lives_left_l2029_202971


namespace NUMINAMATH_GPT_different_colors_of_roads_leading_out_l2029_202981

-- Define the city with intersections and streets
variables (n : ℕ) -- number of intersections
variables (c₁ c₂ c₃ : ℕ) -- number of external roads of each color

-- Conditions
axiom intersections_have_three_streets : ∀ (i : ℕ), i < n → (∀ (color : ℕ), color < 3 → exists (s : ℕ → ℕ), s color < n ∧ s color ≠ s ((color + 1) % 3) ∧ s color ≠ s ((color + 2) % 3))
axiom streets_colored_differently : ∀ (i : ℕ), i < n → (∀ (color1 color2 : ℕ), color1 < 3 → color2 < 3 → color1 ≠ color2 → exists (s1 s2 : ℕ → ℕ), s1 color1 < n ∧ s2 color2 < n ∧ s1 color1 ≠ s2 color2)

-- Problem Statement
theorem different_colors_of_roads_leading_out (h₁ : n % 2 = 0) (h₂ : c₁ + c₂ + c₃ = 3) : c₁ = 1 ∧ c₂ = 1 ∧ c₃ = 1 :=
by sorry

end NUMINAMATH_GPT_different_colors_of_roads_leading_out_l2029_202981


namespace NUMINAMATH_GPT_quadratic_integer_roots_l2029_202958

theorem quadratic_integer_roots (a b x : ℤ) :
  (∀ x₁ x₂ : ℤ, x₁ + x₂ = -b / a ∧ x₁ * x₂ = b / a → (x₁ = x₂ ∧ x₁ = -2 ∧ b = 4 * a) ∨ (x = -1 ∧ a = 0 ∧ b ≠ 0) ∨ (x = 0 ∧ a ≠ 0 ∧ b = 0)) :=
sorry

end NUMINAMATH_GPT_quadratic_integer_roots_l2029_202958
