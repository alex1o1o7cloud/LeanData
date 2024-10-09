import Mathlib

namespace max_points_of_intersection_l2331_233179

-- Define the conditions
variable {α : Type*} [DecidableEq α]
variable (L : Fin 100 → α → α → Prop) -- Representation of the lines

-- Define property of being parallel
variable (are_parallel : ∀ {n : ℕ}, L (5 * n) = L (5 * n + 5))

-- Define property of passing through point B
variable (passes_through_B : ∀ {n : ℕ}, ∃ P B, L (5 * n - 4) P B)

-- Prove the stated result
theorem max_points_of_intersection : 
  ∃ max_intersections, max_intersections = 4571 :=
by {
  sorry
}

end max_points_of_intersection_l2331_233179


namespace salary_reduction_percentage_l2331_233131

theorem salary_reduction_percentage
  (S : ℝ) 
  (h : S * (1 - R / 100) = S / 1.388888888888889): R = 28 :=
sorry

end salary_reduction_percentage_l2331_233131


namespace lines_from_equation_l2331_233135

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end lines_from_equation_l2331_233135


namespace smallest_value_x_l2331_233158

theorem smallest_value_x : 
  (∃ x : ℝ, ((5*x - 20)/(4*x - 5))^2 + ((5*x - 20)/(4*x - 5)) = 6 ∧ 
  (∀ y : ℝ, ((5*y - 20)/(4*y - 5))^2 + ((5*y - 20)/(4*y - 5)) = 6 → x ≤ y)) → 
  x = 35 / 17 :=
by 
  sorry

end smallest_value_x_l2331_233158


namespace arithmetic_sequence_5_7_9_l2331_233152

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_5_7_9 (h : 13 * (a 7) = 39) : a 5 + a 7 + a 9 = 9 := 
sorry

end arithmetic_sequence_5_7_9_l2331_233152


namespace find_a_pow_b_l2331_233145

theorem find_a_pow_b (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : a^b = 1 / 2 := 
sorry

end find_a_pow_b_l2331_233145


namespace smallest_prime_dividing_large_sum_is_5_l2331_233178

-- Definitions based on the conditions
def large_sum : ℕ := 4^15 + 7^12

-- Prime number checking function
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Check for the smallest prime number dividing the sum
def smallest_prime_dividing_sum (n : ℕ) : ℕ := 
  if n % 2 = 0 then 2 
  else if n % 3 = 0 then 3 
  else if n % 5 = 0 then 5 
  else 2 -- Since 2 is a placeholder, theoretical logic checks can replace this branch

-- Final theorem to prove
theorem smallest_prime_dividing_large_sum_is_5 : smallest_prime_dividing_sum large_sum = 5 := 
  sorry

end smallest_prime_dividing_large_sum_is_5_l2331_233178


namespace nesbitt_inequality_nesbitt_inequality_eq_l2331_233139

variable {a b c : ℝ}

theorem nesbitt_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ (3 / 2) :=
sorry

theorem nesbitt_inequality_eq (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ((a / (b + c)) + (b / (a + c)) + (c / (a + b)) = (3 / 2)) ↔ (a = b ∧ b = c) :=
sorry

end nesbitt_inequality_nesbitt_inequality_eq_l2331_233139


namespace hyperbola_foci_distance_l2331_233130

theorem hyperbola_foci_distance :
  let a := Real.sqrt 25
  let b := Real.sqrt 9
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let distance := 2 * c
  distance = 2 * Real.sqrt 34 :=
by
  let a := Real.sqrt 25
  let b := Real.sqrt 9
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let distance := 2 * c
  exact sorry

end hyperbola_foci_distance_l2331_233130


namespace complement_P_inter_Q_l2331_233197

def P : Set ℝ := {x | x^2 - 2 * x ≥ 0}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}
def complement_P : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_P_inter_Q : (complement_P ∩ Q) = {x | 1 < x ∧ x < 2} := by
  sorry

end complement_P_inter_Q_l2331_233197


namespace car_y_speed_l2331_233103

noncomputable def carY_average_speed (vX : ℝ) (tY : ℝ) (d : ℝ) : ℝ :=
  d / tY

theorem car_y_speed (vX : ℝ := 35) (tY_min : ℝ := 72) (dX_after_Y : ℝ := 245) :
  carY_average_speed vX (dX_after_Y / vX) dX_after_Y = 35 := 
by
  sorry

end car_y_speed_l2331_233103


namespace inequality_hold_l2331_233151

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end inequality_hold_l2331_233151


namespace slide_vs_slip_l2331_233129

noncomputable def ladder : Type := sorry

def slide_distance (ladder : ladder) : ℝ := sorry
def slip_distance (ladder : ladder) : ℝ := sorry
def is_right_triangle (ladder : ladder) : Prop := sorry

theorem slide_vs_slip (l : ladder) (h : is_right_triangle l) : slip_distance l > slide_distance l :=
sorry

end slide_vs_slip_l2331_233129


namespace gcd_min_val_l2331_233196

theorem gcd_min_val (p q r : ℕ) (hpq : Nat.gcd p q = 210) (hpr : Nat.gcd p r = 1155) : ∃ (g : ℕ), g = Nat.gcd q r ∧ g = 105 :=
by
  sorry

end gcd_min_val_l2331_233196


namespace transformed_expression_value_l2331_233114

theorem transformed_expression_value :
  (240 / 80) * 60 / 40 + 10 = 14.5 :=
by
  sorry

end transformed_expression_value_l2331_233114


namespace original_total_cost_l2331_233120

-- Definitions based on the conditions
def price_jeans : ℝ := 14.50
def price_shirt : ℝ := 9.50
def price_jacket : ℝ := 21.00

def jeans_count : ℕ := 2
def shirts_count : ℕ := 4
def jackets_count : ℕ := 1

-- The proof statement
theorem original_total_cost :
  (jeans_count * price_jeans) + (shirts_count * price_shirt) + (jackets_count * price_jacket) = 88 := 
by
  sorry

end original_total_cost_l2331_233120


namespace unpainted_cubes_count_l2331_233141

/- Definitions of the conditions -/
def total_cubes : ℕ := 6 * 6 * 6
def painted_faces_per_face : ℕ := 4
def total_faces : ℕ := 6
def painted_faces : ℕ := painted_faces_per_face * total_faces
def overlapped_painted_faces : ℕ := 4 -- Each center four squares on one face corresponds to a center square on the opposite face.
def unique_painted_cubes : ℕ := painted_faces / 2

/- Lean Theorem statement that corresponds to proving the question asked in the problem -/
theorem unpainted_cubes_count : 
  total_cubes - unique_painted_cubes = 208 :=
  by
    sorry

end unpainted_cubes_count_l2331_233141


namespace remainder_a25_div_26_l2331_233168

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Placeholder function for concatenating numbers from 1 to n
  sorry

theorem remainder_a25_div_26 :
  let a_25 := concatenate_numbers 25
  a_25 % 26 = 13 :=
by sorry

end remainder_a25_div_26_l2331_233168


namespace correct_polynomial_l2331_233128

noncomputable def p : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X^6 - Polynomial.C 8 * Polynomial.X^4 - Polynomial.C 2 * Polynomial.X^3 + Polynomial.C 13 * Polynomial.X^2 - Polynomial.C 10 * Polynomial.X - Polynomial.C 1

theorem correct_polynomial (r t : ℝ) :
  (r^3 - r - 1 = 0) → (t = r + Real.sqrt 2) → Polynomial.aeval t p = 0 :=
by
  sorry

end correct_polynomial_l2331_233128


namespace line_through_point_and_intersects_circle_with_chord_length_8_l2331_233136

theorem line_through_point_and_intersects_circle_with_chord_length_8 :
  ∃ (l : ℝ → ℝ), (∀ (x : ℝ), l x = 0 ↔ x = 5) ∨ 
  (∀ (x y : ℝ), 7 * x + 24 * y = 35) ↔ 
  (∃ (x : ℝ), x = 5) ∨ 
  (∀ (x y : ℝ), 7 * x + 24 * y = 35) := 
by
  sorry

end line_through_point_and_intersects_circle_with_chord_length_8_l2331_233136


namespace closest_fraction_l2331_233116

theorem closest_fraction (won : ℚ) (options : List ℚ) (closest : ℚ) 
  (h_won : won = 25 / 120) 
  (h_options : options = [1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8]) 
  (h_closest : closest = 1 / 5) :
  ∃ x ∈ options, abs (won - x) = abs (won - closest) := 
sorry

end closest_fraction_l2331_233116


namespace find_N_l2331_233163

theorem find_N (a b c : ℤ) (N : ℤ)
  (h1 : a + b + c = 105)
  (h2 : a - 5 = N)
  (h3 : b + 10 = N)
  (h4 : 5 * c = N) : 
  N = 50 :=
by
  sorry

end find_N_l2331_233163


namespace price_per_book_sold_l2331_233127

-- Definitions based on the given conditions
def total_books_before_sale : ℕ := 3 * 50
def books_sold : ℕ := 2 * 50
def total_amount_received : ℕ := 500

-- Target statement to be proved
theorem price_per_book_sold :
  (total_amount_received : ℚ) / books_sold = 5 :=
sorry

end price_per_book_sold_l2331_233127


namespace vincent_earnings_after_5_days_l2331_233132

def fantasy_book_price : ℕ := 4
def daily_fantasy_books_sold : ℕ := 5
def literature_book_price : ℕ := fantasy_book_price / 2
def daily_literature_books_sold : ℕ := 8
def days : ℕ := 5

def daily_earnings : ℕ :=
  (fantasy_book_price * daily_fantasy_books_sold) +
  (literature_book_price * daily_literature_books_sold)

def total_earnings (d : ℕ) : ℕ :=
  daily_earnings * d

theorem vincent_earnings_after_5_days : total_earnings days = 180 := by
  sorry

end vincent_earnings_after_5_days_l2331_233132


namespace margo_paired_with_irma_probability_l2331_233143

theorem margo_paired_with_irma_probability :
  let n := 15
  let total_outcomes := n
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_outcomes
  probability = (1 / 15) :=
by
  let n := 15
  let total_outcomes := n
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_outcomes
  have h : probability = 1 / 15 := by
    -- skipping the proof details as per instructions
    sorry
  exact h

end margo_paired_with_irma_probability_l2331_233143


namespace volume_in_cubic_meters_l2331_233119

noncomputable def mass_condition : ℝ := 100 -- mass in kg
noncomputable def volume_per_gram : ℝ := 10 -- volume in cubic centimeters per gram
noncomputable def volume_per_kg : ℝ := volume_per_gram * 1000 -- volume in cubic centimeters per kg
noncomputable def mass_in_kg : ℝ := mass_condition

theorem volume_in_cubic_meters (h : mass_in_kg = 100)
    (v_per_kg : volume_per_kg = volume_per_gram * 1000) :
  (mass_in_kg * volume_per_kg) / 1000000 = 1 := by
  sorry

end volume_in_cubic_meters_l2331_233119


namespace polynomial_remainder_l2331_233157

theorem polynomial_remainder (x : ℤ) :
  let dividend := 3*x^3 - 2*x^2 - 23*x + 60
  let divisor := x - 4
  let quotient := 3*x^2 + 10*x + 17
  let remainder := 128
  dividend = divisor * quotient + remainder :=
by 
  -- proof steps would go here, but we use "sorry" as instructed
  sorry

end polynomial_remainder_l2331_233157


namespace parallel_lines_slope_condition_l2331_233113

-- Define the first line equation and the slope
def line1 (x : ℝ) : ℝ := 6 * x + 5
def slope1 : ℝ := 6

-- Define the second line equation and the slope
def line2 (x c : ℝ) : ℝ := (3 * c) * x - 7
def slope2 (c : ℝ) : ℝ := 3 * c

-- Theorem stating that if the lines are parallel, the value of c is 2
theorem parallel_lines_slope_condition (c : ℝ) : 
  (slope1 = slope2 c) → c = 2 := 
  by
    sorry -- Proof

end parallel_lines_slope_condition_l2331_233113


namespace only_one_way_to_center_l2331_233172

def is_center {n : ℕ} (grid_size n : ℕ) (coord : ℕ × ℕ) : Prop :=
  coord = (grid_size / 2 + 1, grid_size / 2 + 1)

def count_ways_to_center : ℕ :=
  if h : (1 <= 3 ∧ 3 <= 5) then 1 else 0

theorem only_one_way_to_center : count_ways_to_center = 1 := by
  sorry

end only_one_way_to_center_l2331_233172


namespace multiplication_of_positive_and_negative_l2331_233159

theorem multiplication_of_positive_and_negative :
  9 * (-3) = -27 := by
  sorry

end multiplication_of_positive_and_negative_l2331_233159


namespace integer_coefficient_equation_calculate_expression_l2331_233125

noncomputable def a : ℝ := (Real.sqrt 5 - 1) / 2

theorem integer_coefficient_equation :
  a ^ 2 + a - 1 = 0 :=
sorry

theorem calculate_expression :
  a ^ 3 - 2 * a + 2015 = 2014 :=
sorry

end integer_coefficient_equation_calculate_expression_l2331_233125


namespace robot_swap_eventually_non_swappable_l2331_233177

theorem robot_swap_eventually_non_swappable (n : ℕ) (a : Fin n → ℕ) :
  ∃ t : ℕ, ∀ i : Fin (n - 1), ¬ (a (⟨i, sorry⟩ : Fin n) > a (⟨i + 1, sorry⟩ : Fin n)) ↔ n > 1 :=
sorry

end robot_swap_eventually_non_swappable_l2331_233177


namespace mod_2021_2022_2023_2024_eq_zero_mod_7_l2331_233153

theorem mod_2021_2022_2023_2024_eq_zero_mod_7 :
  (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end mod_2021_2022_2023_2024_eq_zero_mod_7_l2331_233153


namespace rectangle_dimensions_l2331_233189

theorem rectangle_dimensions (x : ℝ) (h : 4 * x * x = 120) : x = Real.sqrt 30 ∧ 4 * x = 4 * Real.sqrt 30 :=
by
  sorry

end rectangle_dimensions_l2331_233189


namespace solveRealInequality_l2331_233195

theorem solveRealInequality (x : ℝ) (hx : 0 < x) : x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x^3) ≥ 18 → x = 3 :=
by
  sorry -- proof to be filled in

end solveRealInequality_l2331_233195


namespace find_sum_of_a_and_b_l2331_233194

theorem find_sum_of_a_and_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, (abs (x^2 - 2 * a * x + b) = 8) → (x = a ∨ x = a + 4 ∨ x = a - 4))
  (h2 : a^2 + (a - 4)^2 = (a + 4)^2) :
  a + b = 264 :=
by
  sorry

end find_sum_of_a_and_b_l2331_233194


namespace parabola_x_intercepts_l2331_233148

theorem parabola_x_intercepts : 
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = -3 * y ^ 2 + 2 * y + 3 :=
by 
  sorry

end parabola_x_intercepts_l2331_233148


namespace value_of_f2_l2331_233144

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b * x + 3

theorem value_of_f2 (a b : ℝ) (h1 : f 1 a b = 7) (h2 : f 3 a b = 15) : f 2 a b = 11 :=
by
  sorry

end value_of_f2_l2331_233144


namespace minimum_value_l2331_233106

noncomputable def function_y (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 1450

theorem minimum_value : ∀ x : ℝ, function_y x ≥ 1438 :=
by 
  intro x
  sorry

end minimum_value_l2331_233106


namespace sheila_initial_savings_l2331_233117

noncomputable def initial_savings (monthly_savings : ℕ) (years : ℕ) (family_addition : ℕ) (total_amount : ℕ) : ℕ :=
  total_amount - (monthly_savings * 12 * years + family_addition)

def sheila_initial_savings_proof : Prop :=
  initial_savings 276 4 7000 23248 = 3000

theorem sheila_initial_savings : sheila_initial_savings_proof :=
  by
    -- Proof goes here
    sorry

end sheila_initial_savings_l2331_233117


namespace baker_usual_pastries_l2331_233192

variable (P : ℕ)

theorem baker_usual_pastries
  (h1 : 2 * 14 + 4 * 25 - (2 * P + 4 * 10) = 48) : P = 20 :=
by
  sorry

end baker_usual_pastries_l2331_233192


namespace selecting_elements_l2331_233198

theorem selecting_elements (P Q S : ℕ) (a : ℕ) 
    (h1 : P = Nat.choose 17 (2 * a - 1))
    (h2 : Q = Nat.choose 17 (2 * a))
    (h3 : S = Nat.choose 18 12) :
    P + Q = S → (a = 3 ∨ a = 6) :=
by
  sorry

end selecting_elements_l2331_233198


namespace rectangle_width_is_pi_l2331_233161

theorem rectangle_width_is_pi (w : ℝ) (h1 : real_w ≠ 0)
    (h2 : ∀ w, ∃ length, length = 2 * w)
    (h3 : ∀ w, 2 * (length + w) = 6 * w)
    (h4 : 2 * (2 * w + w) = 6 * π) : 
    w = π :=
by {
  sorry -- The proof would go here.
}

end rectangle_width_is_pi_l2331_233161


namespace find_m_l2331_233137

theorem find_m (
  x : ℚ 
) (m : ℚ) 
  (h1 : 4 * x + 2 * m = 3 * x + 1) 
  (h2 : 3 * x + 2 * m = 6 * x + 1) 
: m = 1/2 := 
  sorry

end find_m_l2331_233137


namespace num_perfect_square_factors_of_450_l2331_233164

theorem num_perfect_square_factors_of_450 :
  ∃ n : ℕ, n = 4 ∧ ∀ d : ℕ, d ∣ 450 → (∃ k : ℕ, d = k * k) → d = 1 ∨ d = 25 ∨ d = 9 ∨ d = 225 :=
by
  sorry

end num_perfect_square_factors_of_450_l2331_233164


namespace trigonometric_identity_l2331_233175

open Real

theorem trigonometric_identity :
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_18 := (sqrt 5 - 1) / 4
  4 * cos_18 ^ 2 - 1 = 1 / (4 * sin_18 ^ 2) :=
by
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_18 := (sqrt 5 - 1) / 4
  sorry

end trigonometric_identity_l2331_233175


namespace fraction_to_decimal_l2331_233184

theorem fraction_to_decimal : (5 / 50) = 0.10 := 
by
  sorry

end fraction_to_decimal_l2331_233184


namespace johns_videos_weekly_minutes_l2331_233190

theorem johns_videos_weekly_minutes (daily_minutes weekly_minutes : ℕ) (short_video_length long_factor: ℕ) (short_videos_per_day long_videos_per_day days : ℕ)
  (h1 : daily_minutes = short_videos_per_day * short_video_length + long_videos_per_day * (long_factor * short_video_length))
  (h2 : weekly_minutes = daily_minutes * days)
  (h_short_videos_per_day : short_videos_per_day = 2)
  (h_long_videos_per_day : long_videos_per_day = 1)
  (h_short_video_length : short_video_length = 2)
  (h_long_factor : long_factor = 6)
  (h_weekly_minutes : weekly_minutes = 112):
  days = 7 :=
by
  sorry

end johns_videos_weekly_minutes_l2331_233190


namespace correct_operation_l2331_233199

variable (a b : ℝ)

theorem correct_operation : 
  ¬ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∧
  ¬ ((a^3) ^ 2 = a ^ 5) ∧
  (a ^ 5 / a ^ 3 = a ^ 2) ∧
  ¬ (a ^ 3 + a ^ 2 = a ^ 5) :=
by
  sorry

end correct_operation_l2331_233199


namespace geometric_sequence_analogy_l2331_233173

variables {a_n b_n : ℕ → ℕ} {S T : ℕ → ℕ}

-- Conditions for the arithmetic sequence
def is_arithmetic_sequence_sum (S : ℕ → ℕ) :=
  S 8 - S 4 = 2 * (S 4) ∧ S 12 - S 8 = 2 * (S 8 - S 4)

-- Conditions for the geometric sequence
def is_geometric_sequence_product (T : ℕ → ℕ) :=
  (T 8 / T 4) = (T 4) ∧ (T 12 / T 8) = (T 8 / T 4)

-- Statement of the proof problem
theorem geometric_sequence_analogy
  (h_arithmetic : is_arithmetic_sequence_sum S)
  (h_geometric_nil : is_geometric_sequence_product T) :
  T 4 / T 4 = 1 ∧
  (T 8 / T 4) / (T 8 / T 4) = 1 ∧
  (T 12 / T 8) / (T 12 / T 8) = 1 := 
by
  sorry

end geometric_sequence_analogy_l2331_233173


namespace amy_tips_calculation_l2331_233122

theorem amy_tips_calculation 
  (hourly_wage : ℝ) (hours_worked : ℝ) (total_earnings : ℝ) 
  (h_wage : hourly_wage = 2)
  (h_hours : hours_worked = 7)
  (h_total : total_earnings = 23) : 
  total_earnings - (hourly_wage * hours_worked) = 9 := 
sorry

end amy_tips_calculation_l2331_233122


namespace min_value_frac_l2331_233169

theorem min_value_frac (x y : ℝ) (h₁ : x + y = 1) (h₂ : x > 0) (h₃ : y > 0) : 
  ∃ c, (∀ (a b : ℝ), (a + b = 1) → (a > 0) → (b > 0) → (1/a + 4/b) ≥ c) ∧ c = 9 :=
by
  sorry

end min_value_frac_l2331_233169


namespace value_of_a1_l2331_233140

def seq (a : ℕ → ℚ) (a_8 : ℚ) : Prop :=
  ∀ n : ℕ, (a (n + 1) = 1 / (1 - a n)) ∧ a 8 = 2

theorem value_of_a1 (a : ℕ → ℚ) (h : seq a 2) : a 1 = 1 / 2 :=
  sorry

end value_of_a1_l2331_233140


namespace quadratic_equal_roots_relation_l2331_233101

theorem quadratic_equal_roots_relation (a b c : ℝ) (h₁ : b ≠ c) 
  (h₂ : ∀ x : ℝ, (b - c) * x^2 + (a - b) * x + (c - a) = 0 → 
          (a - b)^2 - 4 * (b - c) * (c - a) = 0) : 
  c = (a + b) / 2 := sorry

end quadratic_equal_roots_relation_l2331_233101


namespace runner_speed_ratio_l2331_233160

theorem runner_speed_ratio (d s u v_f v_s : ℝ) (hs : s ≠ 0) (hu : u ≠ 0)
  (H1 : (v_f + v_s) * s = d) (H2 : (v_f - v_s) * u = v_s * u) :
  v_f / v_s = 2 :=
by
  sorry

end runner_speed_ratio_l2331_233160


namespace grade_A_probability_l2331_233126

theorem grade_A_probability
  (P_B : ℝ) (P_C : ℝ)
  (hB : P_B = 0.05)
  (hC : P_C = 0.03) :
  1 - P_B - P_C = 0.92 :=
by
  sorry

end grade_A_probability_l2331_233126


namespace arithmetic_seq_a11_l2331_233107

theorem arithmetic_seq_a11 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 21 = 105) : a 11 = 5 :=
sorry

end arithmetic_seq_a11_l2331_233107


namespace profit_percentage_is_20_l2331_233171

noncomputable def selling_price : ℝ := 200
noncomputable def cost_price : ℝ := 166.67
noncomputable def profit : ℝ := selling_price - cost_price

theorem profit_percentage_is_20 :
  (profit / cost_price) * 100 = 20 := by
  sorry

end profit_percentage_is_20_l2331_233171


namespace max_marks_set_for_test_l2331_233150

-- Define the conditions according to the problem statement
def passing_percentage : ℝ := 0.70
def student_marks : ℝ := 120
def marks_needed_to_pass : ℝ := 150
def passing_threshold (M : ℝ) : ℝ := passing_percentage * M

-- The maximum marks set for the test
theorem max_marks_set_for_test (M : ℝ) : M = 386 :=
by
  -- Given the conditions
  have h : passing_threshold M = student_marks + marks_needed_to_pass := sorry
  -- Solving for M
  sorry

end max_marks_set_for_test_l2331_233150


namespace tens_digit_6_pow_18_l2331_233104

/--
To find the tens digit of \(6^{18}\), we look at the powers of 6 and determine their tens digits. 
We note the pattern in tens digits (3, 1, 9, 7, 6) which repeats every 5 powers. 
Since \(6^{18}\) corresponds to the 3rd position in the repeating cycle, we claim the tens digit is 1.
--/
theorem tens_digit_6_pow_18 : (6^18 / 10) % 10 = 1 :=
by sorry

end tens_digit_6_pow_18_l2331_233104


namespace find_number_divided_by_6_l2331_233118

theorem find_number_divided_by_6 (x : ℤ) (h : (x + 17) / 5 = 25) : x / 6 = 18 :=
by
  sorry

end find_number_divided_by_6_l2331_233118


namespace smallest_possible_a_plus_b_l2331_233123

theorem smallest_possible_a_plus_b :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ gcd (a + b) 330 = 1 ∧ (b ^ b ∣ a ^ a) ∧ ¬ (b ∣ a) ∧ (a + b = 507) := 
sorry

end smallest_possible_a_plus_b_l2331_233123


namespace circle_equation_tangent_to_line_l2331_233110

def circle_center : (ℝ × ℝ) := (3, -1)
def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- The equation of the circle with center at (3, -1) and tangent to the line 3x + 4y = 0 is (x - 3)^2 + (y + 1)^2 = 1 -/
theorem circle_equation_tangent_to_line : 
  ∃ r, ∀ x y: ℝ, ((x - 3)^2 + (y + 1)^2 = r^2) ∧ (∀ (cx cy: ℝ), cx = 3 → cy = -1 → (tangent_line cx cy → r = 1)) :=
by
  sorry

end circle_equation_tangent_to_line_l2331_233110


namespace solution_set_l2331_233187

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ) -- Function for the derivative of f

axiom f_deriv : ∀ x, f' x = (deriv f) x

axiom f_condition1 : ∀ x, f x > 1 - f' x
axiom f_condition2 : f 0 = 0
  
theorem solution_set (x : ℝ) : (e^x * f x > e^x - 1) ↔ (x > 0) := 
  sorry

end solution_set_l2331_233187


namespace remainder_correct_l2331_233182

def dividend : ℕ := 165
def divisor : ℕ := 18
def quotient : ℕ := 9
def remainder : ℕ := 3

theorem remainder_correct {d q r : ℕ} (h1 : d = dividend) (h2 : q = quotient) (h3 : r = divisor * q) : d = 165 → q = 9 → 165 = 162 + remainder :=
by { sorry }

end remainder_correct_l2331_233182


namespace compound_interest_rate_l2331_233166

theorem compound_interest_rate
  (P : ℝ) (r : ℝ) :
  (3000 = P * (1 + r / 100)^3) →
  (3600 = P * (1 + r / 100)^4) →
  r = 20 :=
by
  sorry

end compound_interest_rate_l2331_233166


namespace profit_without_discount_l2331_233142

theorem profit_without_discount (CP SP_discount SP_without_discount : ℝ) (profit_discount profit_without_discount percent_discount : ℝ)
  (h1 : CP = 100) 
  (h2 : percent_discount = 0.05) 
  (h3 : profit_discount = 0.425) 
  (h4 : SP_discount = CP + profit_discount * CP) 
  (h5 : SP_discount = 142.5)
  (h6 : SP_without_discount = SP_discount / (1 - percent_discount)) : 
  profit_without_discount = ((SP_without_discount - CP) / CP) * 100 := 
by
  sorry

end profit_without_discount_l2331_233142


namespace order_xyz_l2331_233165

theorem order_xyz (x : ℝ) (h1 : 0.8 < x) (h2 : x < 0.9) :
  let y := x^x
  let z := x^(x^x)
  x < z ∧ z < y :=
by
  sorry

end order_xyz_l2331_233165


namespace a9_value_l2331_233121

-- Define the sequence
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n+1) = 1 - (1 / a n)

-- State the theorem
theorem a9_value : ∃ a : ℕ → ℚ, seq a ∧ a 9 = -1/2 :=
by
  sorry

end a9_value_l2331_233121


namespace total_tissues_brought_l2331_233115

def number_of_students (group1 group2 group3 : Nat) : Nat :=
  group1 + group2 + group3

def number_of_tissues_per_student (tissues_per_box : Nat) (total_students : Nat) : Nat :=
  tissues_per_box * total_students

theorem total_tissues_brought :
  let group1 := 9
  let group2 := 10
  let group3 := 11
  let tissues_per_box := 40
  let total_students := number_of_students group1 group2 group3
  number_of_tissues_per_student tissues_per_box total_students = 1200 :=
by
  sorry

end total_tissues_brought_l2331_233115


namespace two_by_three_grid_count_l2331_233124

noncomputable def valid2x3Grids : Nat :=
  let valid_grids : Nat := 9
  valid_grids

theorem two_by_three_grid_count : valid2x3Grids = 9 := by
  -- Skipping the proof steps, but stating the theorem.
  sorry

end two_by_three_grid_count_l2331_233124


namespace find_t_when_perpendicular_l2331_233185

variable {t : ℝ}

def vector_m (t : ℝ) : ℝ × ℝ := (t + 1, 1)
def vector_n (t : ℝ) : ℝ × ℝ := (t + 2, 2)
def add_vectors (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def sub_vectors (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

theorem find_t_when_perpendicular : 
  (dot_product (add_vectors (vector_m t) (vector_n t)) (sub_vectors (vector_m t) (vector_n t)) = 0) ↔ t = -3 := by
  sorry

end find_t_when_perpendicular_l2331_233185


namespace total_net_loss_l2331_233102

theorem total_net_loss 
  (P_x P_y : ℝ)
  (h1 : 1.2 * P_x = 25000)
  (h2 : 0.8 * P_y = 25000) :
  (25000 - P_x) - (P_y - 25000) = -2083.33 :=
by 
  sorry

end total_net_loss_l2331_233102


namespace ellipse_non_degenerate_l2331_233167

noncomputable def non_degenerate_ellipse_condition (b : ℝ) : Prop := b > -13

theorem ellipse_non_degenerate (b : ℝ) :
  (∃ x y : ℝ, 4*x^2 + 9*y^2 - 16*x + 18*y + 12 = b) → non_degenerate_ellipse_condition b :=
by
  sorry

end ellipse_non_degenerate_l2331_233167


namespace exists_p_q_for_integer_roots_l2331_233109

theorem exists_p_q_for_integer_roots : 
  ∃ (p q : ℤ), ∀ k (hk : k ∈ (Finset.range 10)), 
    ∃ (r1 r2 : ℤ), (r1 + r2 = -(p + k)) ∧ (r1 * r2 = (q + k)) :=
sorry

end exists_p_q_for_integer_roots_l2331_233109


namespace rational_division_example_l2331_233134

theorem rational_division_example : (3 / 7) / 5 = 3 / 35 := by
  sorry

end rational_division_example_l2331_233134


namespace simplify_expression_l2331_233112

theorem simplify_expression (a : ℤ) (ha : a = -2) : 
  3 * a^2 + (a^2 + (5 * a^2 - 2 * a) - 3 * (a^2 - 3 * a)) = 10 := 
by 
  sorry

end simplify_expression_l2331_233112


namespace value_of_y_l2331_233186

variable (x y : ℤ)

-- Define the conditions
def condition1 : Prop := 3 * (x^2 + x + 1) = y - 6
def condition2 : Prop := x = -3

-- Theorem to prove
theorem value_of_y (h1 : condition1 x y) (h2 : condition2 x) : y = 27 := by
  sorry

end value_of_y_l2331_233186


namespace tracy_initial_candies_l2331_233162

theorem tracy_initial_candies (x : ℕ) (consumed_candies : ℕ) (remaining_candies_given_rachel : ℕ) (remaining_candies_given_monica : ℕ) (candies_eaten_by_tracy : ℕ) (candies_eaten_by_mom : ℕ) 
  (brother_candies_taken : ℕ) (final_candies : ℕ) (h_consume : consumed_candies = 2 / 5 * x) (h_remaining1 : remaining_candies_given_rachel = 1 / 3 * (3 / 5 * x)) 
  (h_remaining2 : remaining_candies_given_monica = 1 / 6 * (3 / 5 * x)) (h_left_after_friends : 3 / 5 * x - (remaining_candies_given_rachel + remaining_candies_given_monica) = 3 / 10 * x)
  (h_candies_left : 3 / 10 * x - (candies_eaten_by_tracy + candies_eaten_by_mom) = final_candies + brother_candies_taken) (h_eaten_tracy : candies_eaten_by_tracy = 10)
  (h_eaten_mom : candies_eaten_by_mom = 10) (h_final : final_candies = 6) (h_brother_bound : 2 ≤ brother_candies_taken ∧ brother_candies_taken ≤ 6) : x = 100 := 
by 
  sorry

end tracy_initial_candies_l2331_233162


namespace odd_number_diff_squares_unique_l2331_233170

theorem odd_number_diff_squares_unique (n : ℕ) (h : 0 < n) : 
  ∃! (x y : ℤ), (2 * n + 1) = x^2 - y^2 :=
by {
  sorry
}

end odd_number_diff_squares_unique_l2331_233170


namespace problem_1_1_and_2_problem_1_2_l2331_233183

section Sequence

variables (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom a_1 : a 1 = 3
axiom a_n_recurr : ∀ n ≥ 2, a n = 2 * a (n - 1) + (n - 2)

-- Prove that {a_n + n} is a geometric sequence and find the general term formula for {a_n}
theorem problem_1_1_and_2 :
  (∀ n ≥ 2, (a (n - 1) + (n - 1) ≠ 0)) ∧ ((a 1 + 1) * 2^(n - 1) = a n + n) ∧
  (∀ n, a n = 2^(n + 1) - n) :=
sorry

-- Find the sum of the first n terms, S_n, of the sequence {a_n}
theorem problem_1_2 (n : ℕ) : S n = 2^(n + 2) - 4 - (n^2 + n) / 2 :=
sorry

end Sequence

end problem_1_1_and_2_problem_1_2_l2331_233183


namespace bob_corn_calc_l2331_233111

noncomputable def bob_corn_left (initial_bushels : ℕ) (ears_per_bushel : ℕ) (bushels_taken_by_terry : ℕ) (bushels_taken_by_jerry : ℕ) (bushels_taken_by_linda : ℕ) (ears_taken_by_stacy : ℕ) : ℕ :=
  let initial_ears := initial_bushels * ears_per_bushel
  let ears_given_away := (bushels_taken_by_terry + bushels_taken_by_jerry + bushels_taken_by_linda) * ears_per_bushel + ears_taken_by_stacy
  initial_ears - ears_given_away

theorem bob_corn_calc :
  bob_corn_left 50 14 8 3 12 21 = 357 :=
by
  sorry

end bob_corn_calc_l2331_233111


namespace fraction_identity_l2331_233100

-- Definitions for conditions
variables (a b : ℚ)

-- The main statement to prove
theorem fraction_identity (h : a/b = 2/5) : (a + b) / b = 7 / 5 :=
by
  sorry

end fraction_identity_l2331_233100


namespace fraction_eq_l2331_233176

theorem fraction_eq : (15.5 / (-0.75) : ℝ) = (-62 / 3) := 
by {
  sorry
}

end fraction_eq_l2331_233176


namespace binary_101_to_decimal_l2331_233146

theorem binary_101_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 5 := by
  sorry

end binary_101_to_decimal_l2331_233146


namespace sum_infinite_series_eq_l2331_233133

theorem sum_infinite_series_eq : 
  (∑' n : ℕ, if n > 0 then ((3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3))) else 0) = (7 / 12) :=
by
  sorry

end sum_infinite_series_eq_l2331_233133


namespace smallest_perimeter_of_scalene_triangle_with_conditions_l2331_233188

def is_odd_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 2 = 1

-- Define a scalene triangle
structure ScaleneTriangle :=
  (a b c : ℕ)
  (a_ne_b : a ≠ b)
  (a_ne_c : a ≠ c)
  (b_ne_c : b ≠ c)
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a)

-- Define the problem conditions
def problem_conditions (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧
  a < b ∧ b < c ∧
  Nat.Prime (a + b + c) ∧
  (∃ (t : ScaleneTriangle), t.a = a ∧ t.b = b ∧ t.c = c)

-- Define the proposition
theorem smallest_perimeter_of_scalene_triangle_with_conditions :
  ∃ (a b c : ℕ), problem_conditions a b c ∧ a + b + c = 23 :=
sorry

end smallest_perimeter_of_scalene_triangle_with_conditions_l2331_233188


namespace hyperbola_center_l2331_233154

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (f1 : x1 = 3) (f2 : y1 = -2) (f3 : x2 = 11) (f4 : y2 = 6) :
    (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 2 :=
by
  sorry

end hyperbola_center_l2331_233154


namespace total_weight_correct_total_money_earned_correct_l2331_233193

variable (records : List Int) (std_weight : Int)

-- Conditions
def deviation_sum (records : List Int) : Int := records.foldl (· + ·) 0

def batch_weight (std_weight : Int) (n : Int) (deviation_sum : Int) : Int :=
  deviation_sum + std_weight * n

def first_day_sales (total_weight : Int) (price_per_kg : Int) : Int :=
  price_per_kg * (total_weight / 2)

def second_day_sales (total_weight : Int) (first_day_sales_weight : Int) (discounted_price_per_kg : Int) : Int :=
  discounted_price_per_kg * (total_weight - first_day_sales_weight)

def total_earnings (first_day_sales : Int) (second_day_sales : Int) : Int :=
  first_day_sales + second_day_sales

-- Proof statements
theorem total_weight_correct : 
  deviation_sum records = 4 ∧ std_weight = 30 ∧ records.length = 8 → 
  batch_weight std_weight records.length (deviation_sum records) = 244 :=
by
  intro h
  sorry

theorem total_money_earned_correct :
  first_day_sales (batch_weight std_weight records.length (deviation_sum records)) 10 = 1220 ∧
  second_day_sales (batch_weight std_weight records.length (deviation_sum records)) (batch_weight std_weight records.length (deviation_sum records) / 2) (10 * 9 / 10) = 1098 →
  total_earnings 1220 1098 = 2318 :=
by
  intro h
  sorry

end total_weight_correct_total_money_earned_correct_l2331_233193


namespace at_least_one_A_or_B_selected_prob_l2331_233105

theorem at_least_one_A_or_B_selected_prob :
  let students := ['A', 'B', 'C', 'D']
  let total_pairs := 6
  let complementary_event_prob := 1 / total_pairs
  let at_least_one_A_or_B_prob := 1 - complementary_event_prob
  at_least_one_A_or_B_prob = 5 / 6 :=
by
  let students := ['A', 'B', 'C', 'D']
  let total_pairs := 6
  let complementary_event_prob := 1 / total_pairs
  let at_least_one_A_or_B_prob := 1 - complementary_event_prob
  sorry

end at_least_one_A_or_B_selected_prob_l2331_233105


namespace smallest_positive_x_for_palindrome_l2331_233138

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem smallest_positive_x_for_palindrome :
  ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 1234) ∧ (∀ y : ℕ, y > 0 → is_palindrome (y + 1234) → x ≤ y) ∧ x = 97 := 
sorry

end smallest_positive_x_for_palindrome_l2331_233138


namespace plane_divides_pyramid_l2331_233147

noncomputable def volume_of_parts (a h KL KK1: ℝ): ℝ × ℝ :=
  -- Define the pyramid and prism structure and the conditions
  let volume_total := (1/3) * (a^2) * h
  let volume_part1 := 512/15
  let volume_part2 := volume_total - volume_part1
  (⟨volume_part1, volume_part2⟩ : ℝ × ℝ)

theorem plane_divides_pyramid (a h KL KK1: ℝ) 
  (h₁ : a = 8 * Real.sqrt 2) 
  (h₂ : h = 4) 
  (h₃ : KL = 2) 
  (h₄ : KK1 = 1):
  volume_of_parts a h KL KK1 = (512/15, 2048/15) := 
by 
  sorry

end plane_divides_pyramid_l2331_233147


namespace boat_speed_still_water_l2331_233180

variable (V_b V_s t : ℝ)

-- Conditions given in the problem
axiom speedOfStream : V_s = 13
axiom timeRelation : ∀ t, (V_b + V_s) * t = 2 * (V_b - V_s) * t

-- The statement to be proved
theorem boat_speed_still_water : V_b = 39 :=
by
  sorry

end boat_speed_still_water_l2331_233180


namespace arithmetic_geom_sequence_ratio_l2331_233108

theorem arithmetic_geom_sequence_ratio (a : ℕ → ℝ) (d a1 : ℝ) (h1 : d ≠ 0) 
(h2 : ∀ n, a (n+1) = a n + d)
(h3 : (a 0 + 2 * d)^2 = a 0 * (a 0 + 8 * d)):
  (a 0 + a 2 + a 8) / (a 1 + a 3 + a 9) = 13 / 16 := 
by sorry

end arithmetic_geom_sequence_ratio_l2331_233108


namespace fraction_half_way_l2331_233156

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l2331_233156


namespace dora_knows_coin_position_l2331_233181

-- Definitions
def R_is_dime_or_nickel (R : ℕ) (L : ℕ) : Prop := 
  (R = 10 ∧ L = 5) ∨ (R = 5 ∧ L = 10)

-- Theorem statement
theorem dora_knows_coin_position (R : ℕ) (L : ℕ) 
  (h : R_is_dime_or_nickel R L) :
  (3 * R + 2 * L) % 2 = 0 ↔ (R = 10 ∧ L = 5) :=
by
  sorry

end dora_knows_coin_position_l2331_233181


namespace find_k_l2331_233191

-- Define the problem's conditions and constants
variables (S x y : ℝ)

-- Define the main theorem to prove k = 8 given the conditions
theorem find_k (h1 : 0.75 * x + ((S - 0.75 * x) * x) / (x + y) - (S * x) / (x + y) = 18) :
  (x * y / 3) / (x + y) = 8 := by 
  sorry

end find_k_l2331_233191


namespace field_width_calculation_l2331_233174

theorem field_width_calculation (w : ℝ) (h_length : length = 24) (h_length_width_relation : length = 2 * w - 3) : w = 13.5 :=
by 
  sorry

end field_width_calculation_l2331_233174


namespace not_sum_three_nonzero_squares_l2331_233149

-- To state that 8n - 1 is not the sum of three non-zero squares
theorem not_sum_three_nonzero_squares (n : ℕ) :
  ¬ (∃ a b c : ℕ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 8 * n - 1 = a^2 + b^2 + c^2) := by
  sorry

end not_sum_three_nonzero_squares_l2331_233149


namespace estimation_correct_l2331_233155

-- Definitions corresponding to conditions.
def total_population : ℕ := 10000
def surveyed_population : ℕ := 200
def aware_surveyed : ℕ := 125

-- The proportion step: 125/200 = x/10000
def proportion (aware surveyed total_pop : ℕ) : ℕ :=
  (aware * total_pop) / surveyed

-- Using this to define our main proof goal
def estimated_aware := proportion aware_surveyed surveyed_population total_population

-- Final proof statement
theorem estimation_correct :
  estimated_aware = 6250 :=
sorry

end estimation_correct_l2331_233155
