import Mathlib

namespace NUMINAMATH_GPT_find_original_number_l1099_109938

def original_number (x : ℝ) : Prop :=
  let step1 := 1.20 * x
  let step2 := step1 * 0.85
  let final_value := step2 * 1.30
  final_value = 1080

theorem find_original_number : ∃ x : ℝ, original_number x :=
by
  use 1080 / (1.20 * 0.85 * 1.30)
  sorry

end NUMINAMATH_GPT_find_original_number_l1099_109938


namespace NUMINAMATH_GPT_find_original_sum_of_money_l1099_109973

theorem find_original_sum_of_money
  (R : ℝ)
  (P : ℝ)
  (h1 : 3 * P * (R + 1) / 100 - 3 * P * R / 100 = 63) :
  P = 2100 :=
sorry

end NUMINAMATH_GPT_find_original_sum_of_money_l1099_109973


namespace NUMINAMATH_GPT_no_real_solution_l1099_109983

theorem no_real_solution (x : ℝ) : 
  x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 → 
  ¬ (
    (1 / ((x - 1) * (x - 3))) + (1 / ((x - 3) * (x - 5))) + (1 / ((x - 5) * (x - 7))) = 1 / 4
  ) :=
by sorry

end NUMINAMATH_GPT_no_real_solution_l1099_109983


namespace NUMINAMATH_GPT_overall_percentage_of_favor_l1099_109932

theorem overall_percentage_of_favor
    (n_starting : ℕ)
    (n_experienced : ℕ)
    (perc_starting_favor : ℝ)
    (perc_experienced_favor : ℝ)
    (in_favor_from_starting : ℕ)
    (in_favor_from_experienced : ℕ)
    (total_surveyed : ℕ)
    (total_in_favor : ℕ)
    (overall_percentage : ℝ) :
    n_starting = 300 →
    n_experienced = 500 →
    perc_starting_favor = 0.40 →
    perc_experienced_favor = 0.70 →
    in_favor_from_starting = 120 →
    in_favor_from_experienced = 350 →
    total_surveyed = 800 →
    total_in_favor = 470 →
    overall_percentage = (470 / 800) * 100 →
    overall_percentage = 58.75 :=
by
  sorry

end NUMINAMATH_GPT_overall_percentage_of_favor_l1099_109932


namespace NUMINAMATH_GPT_stipulated_percentage_l1099_109977

theorem stipulated_percentage
  (A B C : ℝ)
  (P : ℝ)
  (hA : A = 20000)
  (h_range : B - C = 10000)
  (hB : B = A + (P / 100) * A)
  (hC : C = A - (P / 100) * A) :
  P = 25 :=
sorry

end NUMINAMATH_GPT_stipulated_percentage_l1099_109977


namespace NUMINAMATH_GPT_toothpaste_duration_l1099_109928

theorem toothpaste_duration 
  (toothpaste_grams : ℕ)
  (dad_usage_per_brushing : ℕ) 
  (mom_usage_per_brushing : ℕ) 
  (anne_usage_per_brushing : ℕ) 
  (brother_usage_per_brushing : ℕ) 
  (brushes_per_day : ℕ) 
  (total_usage : ℕ) 
  (days : ℕ) 
  (h1 : toothpaste_grams = 105) 
  (h2 : dad_usage_per_brushing = 3) 
  (h3 : mom_usage_per_brushing = 2) 
  (h4 : anne_usage_per_brushing = 1) 
  (h5 : brother_usage_per_brushing = 1) 
  (h6 : brushes_per_day = 3)
  (h7 : total_usage = (3 * brushes_per_day) + (2 * brushes_per_day) + (1 * brushes_per_day) + (1 * brushes_per_day)) 
  (h8 : days = toothpaste_grams / total_usage) : 
  days = 5 :=
  sorry

end NUMINAMATH_GPT_toothpaste_duration_l1099_109928


namespace NUMINAMATH_GPT_sqrt_equation_l1099_109924

theorem sqrt_equation (n : ℕ) (h : 0 < n) : 
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / (n * (n + 1) : ℝ) :=
sorry

end NUMINAMATH_GPT_sqrt_equation_l1099_109924


namespace NUMINAMATH_GPT_root_expression_equals_181_div_9_l1099_109951

noncomputable def polynomial_root_sum (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22) 
  (h3 : a*b*c = 8) : ℝ :=
  (a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)) 

theorem root_expression_equals_181_div_9
  (a b c : ℝ)
  (h1 : a + b + c = 15)
  (h2 : a*b + b*c + c*a = 22)
  (h3 : a*b*c = 8) :
  polynomial_root_sum a b c h1 h2 h3 = 181 / 9 := by 
  sorry

end NUMINAMATH_GPT_root_expression_equals_181_div_9_l1099_109951


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1099_109937

variable (a b : ℝ)

theorem problem_part1 (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 4) :
  9 / a + 1 / b ≥ 4 :=
sorry

theorem problem_part2 (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 4) :
  ∃ a b, (a + 3 / b) * (b + 3 / a) = 12 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1099_109937


namespace NUMINAMATH_GPT_solve_for_x_l1099_109997

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = 14.4 / x) : x = 0.0144 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1099_109997


namespace NUMINAMATH_GPT_correct_exponentiation_l1099_109903

theorem correct_exponentiation (x : ℝ) : x^2 * x^3 = x^5 :=
by sorry

end NUMINAMATH_GPT_correct_exponentiation_l1099_109903


namespace NUMINAMATH_GPT_line_equation_passing_through_P_and_equal_intercepts_l1099_109991

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition: line passes through point P(1, 3)
def passes_through_P (P : Point) (line_eq : ℝ → ℝ → ℝ) : Prop :=
  line_eq 1 3 = 0

-- Define the condition: equal intercepts on the x-axis and y-axis
def has_equal_intercepts (line_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ (∀ x y, line_eq x y = 0 ↔ x / a + y / a = 1)

-- Define the specific lines x + y - 4 = 0 and 3x - y = 0
def specific_line1 (x y : ℝ) : ℝ := x + y - 4
def specific_line2 (x y : ℝ) : ℝ := 3 * x - y

-- Define the point P(1, 3)
def P := Point.mk 1 3

theorem line_equation_passing_through_P_and_equal_intercepts :
  (passes_through_P P specific_line1 ∧ has_equal_intercepts specific_line1) ∨
  (passes_through_P P specific_line2 ∧ has_equal_intercepts specific_line2) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_passing_through_P_and_equal_intercepts_l1099_109991


namespace NUMINAMATH_GPT_find_pairs_l1099_109976

theorem find_pairs (a b : ℕ) (h1: a > 0) (h2: b > 0) (q r : ℕ)
  (h3: a^2 + b^2 = q * (a + b) + r) (h4: q^2 + r = 1977) : 
  (a = 50 ∧ b = 37) ∨ (a = 37 ∧ b = 50) :=
sorry

end NUMINAMATH_GPT_find_pairs_l1099_109976


namespace NUMINAMATH_GPT_blue_to_yellow_ratio_is_half_l1099_109948

noncomputable section

def yellow_fish := 12
def blue_fish : ℕ := by 
  have total_fish := 42
  have green_fish := 2 * yellow_fish
  exact total_fish - (yellow_fish + green_fish)
def fish_ratio (x y : ℕ) := x / y

theorem blue_to_yellow_ratio_is_half : fish_ratio blue_fish yellow_fish = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_blue_to_yellow_ratio_is_half_l1099_109948


namespace NUMINAMATH_GPT_cubic_identity_l1099_109908

theorem cubic_identity (x : ℝ) (hx : x + 1/x = -5) : x^3 + 1/x^3 = -110 := by
  sorry

end NUMINAMATH_GPT_cubic_identity_l1099_109908


namespace NUMINAMATH_GPT_gcd_102_238_l1099_109949

theorem gcd_102_238 : Int.gcd 102 238 = 34 :=
by
  sorry

end NUMINAMATH_GPT_gcd_102_238_l1099_109949


namespace NUMINAMATH_GPT_jim_less_than_anthony_l1099_109988

-- Definitions for the conditions
def scott_shoes : ℕ := 7

def anthony_shoes : ℕ := 3 * scott_shoes

def jim_shoes : ℕ := anthony_shoes - 2

-- Lean statement to prove the problem
theorem jim_less_than_anthony : anthony_shoes - jim_shoes = 2 := by
  sorry

end NUMINAMATH_GPT_jim_less_than_anthony_l1099_109988


namespace NUMINAMATH_GPT_percent_of_x_is_z_l1099_109960

-- Defining the conditions as constants in the Lean environment
variables (x y z : ℝ)

-- Given conditions
def cond1 : Prop := 0.45 * z = 0.90 * y
def cond2 : Prop := y = 0.75 * x

-- The statement of the problem proving z = 1.5 * x under given conditions
theorem percent_of_x_is_z
  (h1 : cond1 z y)
  (h2 : cond2 y x) :
  z = 1.5 * x :=
sorry

end NUMINAMATH_GPT_percent_of_x_is_z_l1099_109960


namespace NUMINAMATH_GPT_angle_C_of_triangle_l1099_109981

theorem angle_C_of_triangle (A B C : ℝ) (h1 : A + B = 110) (h2 : A + B + C = 180) : C = 70 := 
by
  sorry

end NUMINAMATH_GPT_angle_C_of_triangle_l1099_109981


namespace NUMINAMATH_GPT_mary_spent_on_jacket_l1099_109970

def shirt_cost : ℝ := 13.04
def total_cost : ℝ := 25.31
def jacket_cost : ℝ := total_cost - shirt_cost

theorem mary_spent_on_jacket :
  jacket_cost = 12.27 := by
  sorry

end NUMINAMATH_GPT_mary_spent_on_jacket_l1099_109970


namespace NUMINAMATH_GPT_rounding_strategy_correct_l1099_109986

-- Definitions of rounding functions
def round_down (n : ℕ) : ℕ := n - 1  -- Assuming n is a large integer, for simplicity
def round_up (n : ℕ) : ℕ := n + 1

-- Definitions for conditions
def cond1 (p q r : ℕ) : ℕ := round_down p / round_down q + round_down r
def cond2 (p q r : ℕ) : ℕ := round_up p / round_down q + round_down r
def cond3 (p q r : ℕ) : ℕ := round_down p / round_up q + round_down r
def cond4 (p q r : ℕ) : ℕ := round_down p / round_down q + round_up r
def cond5 (p q r : ℕ) : ℕ := round_up p / round_up q + round_down r

-- Theorem stating the correct condition
theorem rounding_strategy_correct (p q r : ℕ) (hp : 1 ≤ p) (hq : 1 ≤ q) (hr : 1 ≤ r) :
  cond3 p q r < p / q + r :=
sorry

end NUMINAMATH_GPT_rounding_strategy_correct_l1099_109986


namespace NUMINAMATH_GPT_unique_solution_iff_t_eq_quarter_l1099_109906

variable {x y t : ℝ}

theorem unique_solution_iff_t_eq_quarter : (∃! (x y : ℝ), (x ≥ y^2 + t * y ∧ y^2 + t * y ≥ x^2 + t)) ↔ t = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_iff_t_eq_quarter_l1099_109906


namespace NUMINAMATH_GPT_polynomials_with_three_different_roots_count_l1099_109917

theorem polynomials_with_three_different_roots_count :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6: ℕ), 
    a_0 = 0 ∧ 
    (a_6 = 0 ∨ a_6 = 1) ∧
    (a_5 = 0 ∨ a_5 = 1) ∧
    (a_4 = 0 ∨ a_4 = 1) ∧
    (a_3 = 0 ∨ a_3 = 1) ∧
    (a_2 = 0 ∨ a_2 = 1) ∧
    (a_1 = 0 ∨ a_1 = 1) ∧
    (1 + a_6 + a_5 + a_4 + a_3 + a_2 + a_1) % 2 = 0 ∧
    (1 - a_6 + a_5 - a_4 + a_3 - a_2 + a_1) % 2 = 0) -> 
  ∃ (n : ℕ), n = 8 :=
sorry

end NUMINAMATH_GPT_polynomials_with_three_different_roots_count_l1099_109917


namespace NUMINAMATH_GPT_concentration_sequences_and_min_operations_l1099_109931

theorem concentration_sequences_and_min_operations :
  (a_1 = 1.55 ∧ b_1 = 0.65) ∧
  (∀ n ≥ 1, a_n - b_n = 0.9 * (1 / 2)^(n - 1)) ∧
  (∃ n, 0.9 * (1 / 2)^(n - 1) < 0.01 ∧ n = 8) :=
by
  sorry

end NUMINAMATH_GPT_concentration_sequences_and_min_operations_l1099_109931


namespace NUMINAMATH_GPT_find_a_l1099_109929

theorem find_a (a b : ℝ) (h1 : a * (a - 4) = 5) (h2 : b * (b - 4) = 5) (h3 : a ≠ b) (h4 : a + b = 4) : a = -1 :=
by 
sorry

end NUMINAMATH_GPT_find_a_l1099_109929


namespace NUMINAMATH_GPT_album_photos_proof_l1099_109980

def photos_per_page := 4

-- Conditions
def position_81st_photo (n: ℕ) (x: ℕ) :=
  4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20

def position_171st_photo (n: ℕ) (y: ℕ) :=
  4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12

noncomputable def album_photos := 32

theorem album_photos_proof :
  ∃ n x y, position_81st_photo n x ∧ position_171st_photo n y ∧ 4 * n = album_photos :=
by
  sorry

end NUMINAMATH_GPT_album_photos_proof_l1099_109980


namespace NUMINAMATH_GPT_range_of_a_exists_distinct_x1_x2_eq_f_l1099_109900

noncomputable
def f (a x : ℝ) : ℝ :=
  if x < 1 then a * x + 1 - 4 * a else x^2 - 3 * a * x

theorem range_of_a_exists_distinct_x1_x2_eq_f :
  { a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2 } = 
  { a : ℝ | (a > (2 / 3)) ∨ (a ≤ 0) } :=
sorry

end NUMINAMATH_GPT_range_of_a_exists_distinct_x1_x2_eq_f_l1099_109900


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l1099_109907

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^3 = 8) (h2 : x * y = 5) : 
  x^2 + y^2 = -6 := by
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l1099_109907


namespace NUMINAMATH_GPT_probability_three_primes_l1099_109945

def primes : List ℕ := [2, 3, 5, 7]

def is_prime (n : ℕ) : Prop := n ∈ primes

noncomputable def probability_prime : ℚ := 4/10
noncomputable def probability_non_prime : ℚ := 1 - probability_prime

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def calculation :
  ℚ := (choose 5 3) * (probability_prime ^ 3) * (probability_non_prime ^ 2)

theorem probability_three_primes :
  calculation = 720 / 3125 := by
  sorry

end NUMINAMATH_GPT_probability_three_primes_l1099_109945


namespace NUMINAMATH_GPT_trigonometric_identity_l1099_109953

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / 
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := 
by
  -- proof steps are omitted, using sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1099_109953


namespace NUMINAMATH_GPT_plane_equation_through_point_and_parallel_l1099_109936

theorem plane_equation_through_point_and_parallel (P : ℝ × ℝ × ℝ) (D : ℝ)
  (normal_vector : ℝ × ℝ × ℝ) (A B C : ℝ)
  (h1 : normal_vector = (2, -1, 3))
  (h2 : P = (2, 3, -1))
  (h3 : A = 2) (h4 : B = -1) (h5 : C = 3)
  (hD : A * 2 + B * 3 + C * -1 + D = 0) :
  A * x + B * y + C * z + D = 0 :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_through_point_and_parallel_l1099_109936


namespace NUMINAMATH_GPT_percentage_increase_l1099_109901

def x (y: ℝ) : ℝ := 1.25 * y
def z : ℝ := 250
def total_amount (x y z : ℝ) : ℝ := x + y + z

theorem percentage_increase (y: ℝ) : (total_amount (x y) y z = 925) → ((y - z) / z) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1099_109901


namespace NUMINAMATH_GPT_theta_interval_l1099_109926

noncomputable def f (x θ: ℝ) : ℝ := x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ

theorem theta_interval (θ: ℝ) (k: ℤ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x θ > 0) → 
  (2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12) := 
by
  sorry

end NUMINAMATH_GPT_theta_interval_l1099_109926


namespace NUMINAMATH_GPT_number_of_clown_mobiles_l1099_109910

def num_clown_mobiles (total_clowns clowns_per_mobile : ℕ) : ℕ :=
  total_clowns / clowns_per_mobile

theorem number_of_clown_mobiles :
  num_clown_mobiles 140 28 = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_clown_mobiles_l1099_109910


namespace NUMINAMATH_GPT_power_function_convex_upwards_l1099_109914

noncomputable def f (x : ℝ) : ℝ :=
  x ^ (4 / 5)

theorem power_function_convex_upwards (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  f ((x1 + x2) / 2) > (f x1 + f x2) / 2 :=
sorry

end NUMINAMATH_GPT_power_function_convex_upwards_l1099_109914


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l1099_109968

theorem repeating_decimal_as_fraction :
  ∃ x : ℚ, x = 6 / 10 + 7 / 90 ∧ x = 61 / 90 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l1099_109968


namespace NUMINAMATH_GPT_total_employee_costs_in_February_l1099_109905

def weekly_earnings (hours_per_week : ℕ) (rate_per_hour : ℕ) : ℕ :=
  hours_per_week * rate_per_hour

def monthly_earnings 
  (hours_per_week : ℕ) 
  (rate_per_hour : ℕ) 
  (weeks_worked : ℕ) 
  (bonus_deduction : ℕ := 0) 
  : ℕ :=
  weeks_worked * weekly_earnings hours_per_week rate_per_hour + bonus_deduction

theorem total_employee_costs_in_February 
  (hours_Fiona : ℕ := 40) (rate_Fiona : ℕ := 20) (weeks_worked_Fiona : ℕ := 3)
  (hours_John : ℕ := 30) (rate_John : ℕ := 22) (overtime_hours_John : ℕ := 10)
  (hours_Jeremy : ℕ := 25) (rate_Jeremy : ℕ := 18) (bonus_Jeremy : ℕ := 200)
  (hours_Katie : ℕ := 35) (rate_Katie : ℕ := 21) (deduction_Katie : ℕ := 150)
  (hours_Matt : ℕ := 28) (rate_Matt : ℕ := 19) : 
  monthly_earnings hours_Fiona rate_Fiona weeks_worked_Fiona 
  + monthly_earnings hours_John rate_John 4 
    + overtime_hours_John * (rate_John * 3 / 2)
  + monthly_earnings hours_Jeremy rate_Jeremy 4 bonus_Jeremy
  + monthly_earnings hours_Katie rate_Katie 4 - deduction_Katie
  + monthly_earnings hours_Matt rate_Matt 4 = 13278 := 
by sorry

end NUMINAMATH_GPT_total_employee_costs_in_February_l1099_109905


namespace NUMINAMATH_GPT_complete_the_square_example_l1099_109969

theorem complete_the_square_example (x : ℝ) : 
  ∃ c d : ℝ, (x^2 - 6 * x + 5 = 0) ∧ ((x + c)^2 = d) ∧ (d = 4) :=
sorry

end NUMINAMATH_GPT_complete_the_square_example_l1099_109969


namespace NUMINAMATH_GPT_certain_number_minus_two_l1099_109964

theorem certain_number_minus_two (x : ℝ) (h : 6 - x = 2) : x - 2 = 2 := 
sorry

end NUMINAMATH_GPT_certain_number_minus_two_l1099_109964


namespace NUMINAMATH_GPT_coins_donated_l1099_109992

theorem coins_donated (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (coins_left : ℕ) : 
  pennies = 42 ∧ nickels = 36 ∧ dimes = 15 ∧ coins_left = 27 → (pennies + nickels + dimes - coins_left) = 66 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_coins_donated_l1099_109992


namespace NUMINAMATH_GPT_max_value_ab_bc_cd_l1099_109996

theorem max_value_ab_bc_cd (a b c d : ℝ) (h1 : 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) (h4: 0 ≤ d) (h_sum : a + b + c + d = 200) :
  ab + bc + cd ≤ 2500 :=
by
  sorry

end NUMINAMATH_GPT_max_value_ab_bc_cd_l1099_109996


namespace NUMINAMATH_GPT_alley_width_l1099_109911

theorem alley_width (ℓ : ℝ) (m : ℝ) (n : ℝ): ℓ * (1 / 2 + Real.cos (70 * Real.pi / 180)) = ℓ * (Real.cos (60 * Real.pi / 180)) + ℓ * (Real.cos (70 * Real.pi / 180)) := by
  sorry

end NUMINAMATH_GPT_alley_width_l1099_109911


namespace NUMINAMATH_GPT_perpendicular_vectors_m_eq_half_l1099_109995

theorem perpendicular_vectors_m_eq_half (m : ℝ) (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (-1, m)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : m = 1 / 2 :=
sorry

end NUMINAMATH_GPT_perpendicular_vectors_m_eq_half_l1099_109995


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l1099_109919

-- The sum of the squares of three consecutive positive integers equals 770.
-- We aim to prove that the largest integer among them is 17.
theorem sum_of_squares_of_consecutive_integers (n : ℕ) (h_pos : n > 0) 
    (h_sum : (n-1)^2 + n^2 + (n+1)^2 = 770) : n + 1 = 17 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l1099_109919


namespace NUMINAMATH_GPT_triangle_area_eq_l1099_109955

variable (a b c: ℝ) (A B C : ℝ)
variable (h_cosC : Real.cos C = 1/4)
variable (h_c : c = 3)
variable (h_ratio : a / Real.cos A = b / Real.cos B)

theorem triangle_area_eq : (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 15 / 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_eq_l1099_109955


namespace NUMINAMATH_GPT_integers_less_than_2019_divisible_by_18_or_21_but_not_both_l1099_109942

theorem integers_less_than_2019_divisible_by_18_or_21_but_not_both :
  ∃ (N : ℕ), (∀ (n : ℕ), (n < 2019 → (n % 18 = 0 ∨ n % 21 = 0) → n % (18 * 21 / gcd 18 21) ≠ 0) ↔ (∀ (m : ℕ), m < N)) ∧ N = 176 :=
by
  sorry

end NUMINAMATH_GPT_integers_less_than_2019_divisible_by_18_or_21_but_not_both_l1099_109942


namespace NUMINAMATH_GPT_Mary_cut_10_roses_l1099_109933

-- Defining the initial and final number of roses
def initial_roses := 6
def final_roses := 16

-- Calculating the number of roses cut by Mary
def roses_cut := final_roses - initial_roses

-- The proof problem: Prove that the number of roses cut is 10
theorem Mary_cut_10_roses : roses_cut = 10 := by
  sorry

end NUMINAMATH_GPT_Mary_cut_10_roses_l1099_109933


namespace NUMINAMATH_GPT_stable_performance_l1099_109990

theorem stable_performance 
  (X_A_mean : ℝ) (X_B_mean : ℝ) (S_A_var : ℝ) (S_B_var : ℝ)
  (h1 : X_A_mean = 82) (h2 : X_B_mean = 82)
  (h3 : S_A_var = 245) (h4 : S_B_var = 190) : S_B_var < S_A_var :=
by {
  sorry
}

end NUMINAMATH_GPT_stable_performance_l1099_109990


namespace NUMINAMATH_GPT_total_messages_equation_l1099_109975

theorem total_messages_equation (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  exact h

end NUMINAMATH_GPT_total_messages_equation_l1099_109975


namespace NUMINAMATH_GPT_power_of_2_l1099_109998

theorem power_of_2 (n : ℕ) (h1 : n ≥ 1) (h2 : ∃ m : ℕ, m ≥ 1 ∧ (2^n - 1) ∣ (m^2 + 9)) : ∃ k : ℕ, n = 2^k := 
sorry

end NUMINAMATH_GPT_power_of_2_l1099_109998


namespace NUMINAMATH_GPT_slope_parallel_to_original_line_l1099_109985

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end NUMINAMATH_GPT_slope_parallel_to_original_line_l1099_109985


namespace NUMINAMATH_GPT_range_of_a_l1099_109922

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + 1/2 * x^2

theorem range_of_a (a : ℝ)
  (H : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a (x1 + a) - f a (x2 + a)) / (x1 - x2) ≥ 3) :
  a ≥ 9 / 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1099_109922


namespace NUMINAMATH_GPT_coeff_of_term_equal_three_l1099_109956

theorem coeff_of_term_equal_three (x : ℕ) (h : x = 13) : 
    2^x - 2^(x - 2) = 3 * 2^(11) :=
by
    rw [h]
    sorry

end NUMINAMATH_GPT_coeff_of_term_equal_three_l1099_109956


namespace NUMINAMATH_GPT_potato_difference_l1099_109993

def x := 8 * 13
def k := (67 - 13) / 2
def z := 20 * k
def d := z - x

theorem potato_difference : d = 436 :=
by
  sorry

end NUMINAMATH_GPT_potato_difference_l1099_109993


namespace NUMINAMATH_GPT_max_sum_of_triplet_product_60_l1099_109923

theorem max_sum_of_triplet_product_60 : 
  ∃ a b c : ℕ, a * b * c = 60 ∧ a + b + c = 62 :=
sorry

end NUMINAMATH_GPT_max_sum_of_triplet_product_60_l1099_109923


namespace NUMINAMATH_GPT_inequality_interval_l1099_109912

def differentiable_on_R (f : ℝ → ℝ) : Prop := Differentiable ℝ f
def strictly_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ x ∈ I, ∀ y ∈ I, x < y → f x > f y

theorem inequality_interval (f : ℝ → ℝ)
  (h_diff : differentiable_on_R f)
  (h_cond : ∀ x : ℝ, f x > deriv f x)
  (h_init : f 0 = 1) :
  ∀ x : ℝ, (x > 0) ↔ (f x / Real.exp x < 1) := 
by
  sorry

end NUMINAMATH_GPT_inequality_interval_l1099_109912


namespace NUMINAMATH_GPT_negation_of_prop_l1099_109957

theorem negation_of_prop :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_prop_l1099_109957


namespace NUMINAMATH_GPT_max_n_value_l1099_109909

-- Define the arithmetic sequence
variable {a : ℕ → ℤ} (d : ℤ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)

-- Given conditions
variable (h1 : a 1 + a 3 + a 5 = 105)
variable (h2 : a 2 + a 4 + a 6 = 99)

-- Goal: Prove the maximum integer value of n is 10
theorem max_n_value (n : ℕ) (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 3 + a 5 = 105) (h2 : a 2 + a 4 + a 6 = 99) : n ≤ 10 → 
  (∀ m, (0 < m ∧ m ≤ n) → a (2 * m) ≥ 0) → n = 10 := 
sorry

end NUMINAMATH_GPT_max_n_value_l1099_109909


namespace NUMINAMATH_GPT_tan_sum_product_l1099_109941

theorem tan_sum_product (A B C : ℝ) (h_eq: Real.log (Real.tan A) + Real.log (Real.tan C) = 2 * Real.log (Real.tan B)) :
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := by
  sorry

end NUMINAMATH_GPT_tan_sum_product_l1099_109941


namespace NUMINAMATH_GPT_find_intersection_l1099_109946

def intersection_point (x y : ℚ) : Prop :=
  3 * x + 4 * y = 12 ∧ 7 * x - 2 * y = 14

theorem find_intersection :
  intersection_point (40 / 17) (21 / 17) :=
by
  sorry

end NUMINAMATH_GPT_find_intersection_l1099_109946


namespace NUMINAMATH_GPT_sam_coins_and_value_l1099_109916

-- Define initial conditions
def initial_dimes := 9
def initial_nickels := 5
def initial_pennies := 12

def dimes_from_dad := 7
def nickels_taken_by_dad := 3

def pennies_exchanged := 12
def dimes_from_exchange := 2
def pennies_from_exchange := 2

-- Define final counts of coins after transactions
def final_dimes := initial_dimes + dimes_from_dad + dimes_from_exchange
def final_nickels := initial_nickels - nickels_taken_by_dad
def final_pennies := initial_pennies - pennies_exchanged + pennies_from_exchange

-- Define the total count of coins
def total_coins := final_dimes + final_nickels + final_pennies

-- Define the total value in cents
def value_dimes := final_dimes * 10
def value_nickels := final_nickels * 5
def value_pennies := final_pennies * 1

def total_value := value_dimes + value_nickels + value_pennies

-- Proof statement
theorem sam_coins_and_value :
  total_coins = 22 ∧ total_value = 192 := by
  -- Proof details would go here
  sorry

end NUMINAMATH_GPT_sam_coins_and_value_l1099_109916


namespace NUMINAMATH_GPT_percentage_cd_only_l1099_109954

noncomputable def percentage_power_windows : ℝ := 0.60
noncomputable def percentage_anti_lock_brakes : ℝ := 0.40
noncomputable def percentage_cd_player : ℝ := 0.75
noncomputable def percentage_gps_system : ℝ := 0.50
noncomputable def percentage_pw_and_abs : ℝ := 0.10
noncomputable def percentage_abs_and_cd : ℝ := 0.15
noncomputable def percentage_pw_and_cd : ℝ := 0.20
noncomputable def percentage_gps_and_abs : ℝ := 0.12
noncomputable def percentage_gps_and_cd : ℝ := 0.18
noncomputable def percentage_pw_and_gps : ℝ := 0.25

theorem percentage_cd_only : 
  percentage_cd_player - (percentage_abs_and_cd + percentage_pw_and_cd + percentage_gps_and_cd) = 0.22 := 
by
  sorry

end NUMINAMATH_GPT_percentage_cd_only_l1099_109954


namespace NUMINAMATH_GPT_three_f_x_eq_l1099_109961

theorem three_f_x_eq (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 2 / (3 + x)) (x : ℝ) (hx : x > 0) : 
  3 * f x = 18 / (9 + x) := sorry

end NUMINAMATH_GPT_three_f_x_eq_l1099_109961


namespace NUMINAMATH_GPT_max_average_speed_palindromic_journey_l1099_109944

theorem max_average_speed_palindromic_journey
  (initial_odometer : ℕ)
  (final_odometer : ℕ)
  (trip_duration : ℕ)
  (max_speed : ℕ)
  (palindromic : ℕ → Prop)
  (initial_palindrome : palindromic initial_odometer)
  (final_palindrome : palindromic final_odometer)
  (max_speed_constraint : ∀ t, t ≤ trip_duration → t * max_speed ≤ final_odometer - initial_odometer)
  (trip_duration_eq : trip_duration = 5)
  (max_speed_eq : max_speed = 85)
  (initial_odometer_eq : initial_odometer = 69696)
  (final_odometer_max : final_odometer ≤ initial_odometer + max_speed * trip_duration) :
  (max_speed * (final_odometer - initial_odometer) / trip_duration : ℚ) = 82.2 :=
by sorry

end NUMINAMATH_GPT_max_average_speed_palindromic_journey_l1099_109944


namespace NUMINAMATH_GPT_positive_integral_solution_l1099_109920

theorem positive_integral_solution (n : ℕ) (hn : 0 < n) 
  (h : (n : ℚ) / (n + 1) = 125 / 126) : n = 125 := sorry

end NUMINAMATH_GPT_positive_integral_solution_l1099_109920


namespace NUMINAMATH_GPT_combined_salaries_of_A_B_C_D_l1099_109934

theorem combined_salaries_of_A_B_C_D (salaryE : ℕ) (avg_salary : ℕ) (num_people : ℕ)
    (h1 : salaryE = 9000) (h2 : avg_salary = 8800) (h3 : num_people = 5) :
    (avg_salary * num_people) - salaryE = 35000 :=
by
  sorry

end NUMINAMATH_GPT_combined_salaries_of_A_B_C_D_l1099_109934


namespace NUMINAMATH_GPT_evaluation_expression_l1099_109989

theorem evaluation_expression (a b c d : ℝ) 
  (h1 : a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h2 : b = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h3 : c = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h4 : d = -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6) :
  (1/a + 1/b + 1/c + 1/d)^2 = (16 * (11 + 2 * Real.sqrt 30)) / ((11 + 2 * Real.sqrt 30 - 3 * Real.sqrt 6)^2) :=
sorry

end NUMINAMATH_GPT_evaluation_expression_l1099_109989


namespace NUMINAMATH_GPT_find_k_l1099_109971

-- Define the vectors
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 1)
def c : ℝ × ℝ := (-5, 1)

-- Define the condition for parallel vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Define the statement to prove
theorem find_k : parallel (a.1 + k * b.1, a.2 + k * b.2) c → k = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1099_109971


namespace NUMINAMATH_GPT_ratio_equal_one_of_log_conditions_l1099_109999

noncomputable def logBase (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem ratio_equal_one_of_log_conditions
  (p q : ℝ)
  (hp : 0 < p)
  (hq : 0 < q)
  (h : logBase 8 p = logBase 18 q ∧ logBase 18 q = logBase 24 (p + 2 * q)) :
  q / p = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_equal_one_of_log_conditions_l1099_109999


namespace NUMINAMATH_GPT_sphere_surface_area_l1099_109987

theorem sphere_surface_area (edge_length : ℝ) (diameter_eq_edge_length : (diameter : ℝ) = edge_length) :
  (edge_length = 2) → (diameter = 2) → (surface_area : ℝ) = 8 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l1099_109987


namespace NUMINAMATH_GPT_remainder_sum_mod_15_l1099_109940

variable (k j : ℤ) -- these represent any integers

def p := 60 * k + 53
def q := 75 * j + 24

theorem remainder_sum_mod_15 :
  (p k + q j) % 15 = 2 :=  
by 
  sorry

end NUMINAMATH_GPT_remainder_sum_mod_15_l1099_109940


namespace NUMINAMATH_GPT_pyramid_lateral_edge_ratio_l1099_109947

variable (h x : ℝ)

-- We state the conditions as hypotheses
axiom pyramid_intersected_by_plane_parallel_to_base (h : ℝ) (S S' : ℝ) :
  S' = S / 2 → (S' / S = (x / h) ^ 2) → (x = h / Real.sqrt 2)

-- The theorem we need to prove
theorem pyramid_lateral_edge_ratio (h x : ℝ) (S S' : ℝ)
  (cond1 : S' = S / 2)
  (cond2 : S' / S = (x / h) ^ 2) :
  x / h = 1 / Real.sqrt 2 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_pyramid_lateral_edge_ratio_l1099_109947


namespace NUMINAMATH_GPT_janet_dresses_l1099_109921

theorem janet_dresses : 
  ∃ D : ℕ, 
    (D / 2) * (2 / 3) + (D / 2) * (6 / 3) = 32 → D = 24 := 
by {
  sorry
}

end NUMINAMATH_GPT_janet_dresses_l1099_109921


namespace NUMINAMATH_GPT_molecular_weight_of_aluminum_part_in_Al2_CO3_3_l1099_109994

def total_molecular_weight_Al2_CO3_3 : ℝ := 234
def atomic_weight_Al : ℝ := 26.98
def num_atoms_Al_in_Al2_CO3_3 : ℕ := 2

theorem molecular_weight_of_aluminum_part_in_Al2_CO3_3 :
  num_atoms_Al_in_Al2_CO3_3 * atomic_weight_Al = 53.96 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_aluminum_part_in_Al2_CO3_3_l1099_109994


namespace NUMINAMATH_GPT_sum_of_solutions_l1099_109930

theorem sum_of_solutions (S : Set ℝ) (h : ∀ y ∈ S, y + 16 / y = 12) :
  ∃ t : ℝ, (∀ y ∈ S, y = 8 ∨ y = 4) ∧ t = 12 := by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1099_109930


namespace NUMINAMATH_GPT_ferry_tourists_total_l1099_109959

theorem ferry_tourists_total 
  (n : ℕ)
  (a d : ℕ)
  (sum_arithmetic_series : ℕ → ℕ → ℕ → ℕ)
  (trip_count : n = 5)
  (first_term : a = 85)
  (common_difference : d = 3) :
  sum_arithmetic_series n a d = 455 :=
by
  sorry

end NUMINAMATH_GPT_ferry_tourists_total_l1099_109959


namespace NUMINAMATH_GPT_option_transformations_incorrect_l1099_109925

variable {a b x : ℝ}

theorem option_transformations_incorrect (h : a < b) :
  ¬ (3 - a < 3 - b) := by
  -- Here, we would show the incorrectness of the transformation in Option B
  sorry

end NUMINAMATH_GPT_option_transformations_incorrect_l1099_109925


namespace NUMINAMATH_GPT_smallest_perfect_square_divisible_by_3_and_5_l1099_109918

theorem smallest_perfect_square_divisible_by_3_and_5 : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ n = 225 :=
by
  sorry

end NUMINAMATH_GPT_smallest_perfect_square_divisible_by_3_and_5_l1099_109918


namespace NUMINAMATH_GPT_passing_marks_l1099_109927

variable (T P : ℝ)

-- condition 1: 0.30T = P - 30
def condition1 : Prop := 0.30 * T = P - 30

-- condition 2: 0.45T = P + 15
def condition2 : Prop := 0.45 * T = P + 15

-- Proof Statement: P = 120 (passing marks)
theorem passing_marks (T P : ℝ) (h1 : condition1 T P) (h2 : condition2 T P) : P = 120 := 
  sorry

end NUMINAMATH_GPT_passing_marks_l1099_109927


namespace NUMINAMATH_GPT_multiplication_identity_l1099_109972

theorem multiplication_identity (x y z w : ℝ) (h1 : x = 2000) (h2 : y = 2992) (h3 : z = 0.2992) (h4 : w = 20) : 
  x * y * z * w = 4 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_identity_l1099_109972


namespace NUMINAMATH_GPT_travel_probability_l1099_109950

theorem travel_probability (P_A P_B P_C : ℝ) (hA : P_A = 1/3) (hB : P_B = 1/4) (hC : P_C = 1/5) :
  let P_none_travel := (1 - P_A) * (1 - P_B) * (1 - P_C)
  ∃ (P_at_least_one : ℝ), P_at_least_one = 1 - P_none_travel ∧ P_at_least_one = 3/5 :=
by {
  sorry
}

end NUMINAMATH_GPT_travel_probability_l1099_109950


namespace NUMINAMATH_GPT_polynomial_division_quotient_l1099_109978

noncomputable def P (x : ℝ) := 8 * x^3 + 5 * x^2 - 4 * x - 7
noncomputable def D (x : ℝ) := x + 3

theorem polynomial_division_quotient :
  ∀ x : ℝ, (P x) / (D x) = 8 * x^2 - 19 * x + 53 := sorry

end NUMINAMATH_GPT_polynomial_division_quotient_l1099_109978


namespace NUMINAMATH_GPT_circle_value_a_l1099_109913

noncomputable def represents_circle (a : ℝ) (x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0

theorem circle_value_a {a : ℝ} (h : ∀ x y : ℝ, represents_circle a x y) :
  a = -1 :=
by
  sorry

end NUMINAMATH_GPT_circle_value_a_l1099_109913


namespace NUMINAMATH_GPT_calculate_markup_percentage_l1099_109965

noncomputable def cost_price : ℝ := 225
noncomputable def profit_percentage : ℝ := 0.25
noncomputable def discount1_percentage : ℝ := 0.10
noncomputable def discount2_percentage : ℝ := 0.15
noncomputable def selling_price : ℝ := cost_price * (1 + profit_percentage)
noncomputable def markup_percentage : ℝ := 63.54

theorem calculate_markup_percentage :
  let marked_price := selling_price / ((1 - discount1_percentage) * (1 - discount2_percentage))
  let calculated_markup_percentage := ((marked_price - cost_price) / cost_price) * 100
  abs (calculated_markup_percentage - markup_percentage) < 0.01 :=
sorry

end NUMINAMATH_GPT_calculate_markup_percentage_l1099_109965


namespace NUMINAMATH_GPT_eight_pow_91_gt_seven_pow_92_l1099_109915

theorem eight_pow_91_gt_seven_pow_92 : 8^91 > 7^92 :=
  sorry

end NUMINAMATH_GPT_eight_pow_91_gt_seven_pow_92_l1099_109915


namespace NUMINAMATH_GPT_complete_square_k_value_l1099_109967

theorem complete_square_k_value : 
  ∃ k : ℝ, ∀ x : ℝ, (x^2 - 8*x = (x - 4)^2 + k) ∧ k = -16 :=
by
  use -16
  intro x
  sorry

end NUMINAMATH_GPT_complete_square_k_value_l1099_109967


namespace NUMINAMATH_GPT_four_consecutive_integers_divisible_by_24_l1099_109904

noncomputable def product_of_consecutive_integers (n : ℤ) : ℤ :=
  n * (n + 1) * (n + 2) * (n + 3)

theorem four_consecutive_integers_divisible_by_24 (n : ℤ) :
  24 ∣ product_of_consecutive_integers n :=
by
  sorry

end NUMINAMATH_GPT_four_consecutive_integers_divisible_by_24_l1099_109904


namespace NUMINAMATH_GPT_find_a_for_symmetry_l1099_109952

theorem find_a_for_symmetry :
  ∃ a : ℝ, (∀ x : ℝ, a * Real.sin x + Real.cos (x + π / 6) = 
                    a * Real.sin (π / 3 - x) + Real.cos (π / 3 - x + π / 6)) 
           ↔ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_symmetry_l1099_109952


namespace NUMINAMATH_GPT_sqrt_three_cubes_l1099_109902

theorem sqrt_three_cubes : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := 
  sorry

end NUMINAMATH_GPT_sqrt_three_cubes_l1099_109902


namespace NUMINAMATH_GPT_fraction_expression_as_common_fraction_l1099_109943

theorem fraction_expression_as_common_fraction :
  ((3 / 7 + 5 / 8) / (5 / 12 + 2 / 15)) = (295 / 154) := 
by
  sorry

end NUMINAMATH_GPT_fraction_expression_as_common_fraction_l1099_109943


namespace NUMINAMATH_GPT_polynomial_divisibility_l1099_109935

theorem polynomial_divisibility (A B : ℝ) 
    (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^(205 : ℕ) + A * x + B = 0) : 
    A + B = -1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1099_109935


namespace NUMINAMATH_GPT_ice_cream_total_sum_l1099_109958

noncomputable def totalIceCream (friday saturday sunday monday tuesday : ℝ) : ℝ :=
  friday + saturday + sunday + monday + tuesday

theorem ice_cream_total_sum : 
  let friday := 3.25
  let saturday := 2.5
  let sunday := 1.75
  let monday := 0.5
  let tuesday := 2 * monday
  totalIceCream friday saturday sunday monday tuesday = 9 := by
    sorry

end NUMINAMATH_GPT_ice_cream_total_sum_l1099_109958


namespace NUMINAMATH_GPT_find_detergent_volume_l1099_109962

variable (B D W : ℕ)
variable (B' D' W': ℕ)
variable (water_volume: unit)
variable (detergent_volume: unit)

def original_ratio (B D W : ℕ) : Prop := B = 2 * W / 100 ∧ D = 40 * W / 100

def altered_ratio (B' D' W' B D W : ℕ) : Prop :=
  B' = 3 * B ∧ D' = D / 2 ∧ W' = W ∧ W' = 300

theorem find_detergent_volume {B D W B' D' W'} (h₀ : original_ratio B D W) (h₁ : altered_ratio B' D' W' B D W) :
  D' = 120 :=
sorry

end NUMINAMATH_GPT_find_detergent_volume_l1099_109962


namespace NUMINAMATH_GPT_molly_takes_180_minutes_longer_l1099_109939

noncomputable def time_for_Xanthia (pages_per_hour : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / pages_per_hour

noncomputable def time_for_Molly (pages_per_hour : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / pages_per_hour

theorem molly_takes_180_minutes_longer (pages : ℕ) (Xanthia_speed : ℕ) (Molly_speed : ℕ) :
  (time_for_Molly Molly_speed pages - time_for_Xanthia Xanthia_speed pages) * 60 = 180 :=
by
  -- Definitions specific to problem conditions
  let pages := 360
  let Xanthia_speed := 120
  let Molly_speed := 60

  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_molly_takes_180_minutes_longer_l1099_109939


namespace NUMINAMATH_GPT_increasing_function_l1099_109979

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_function (a : ℝ) :
  (∀ x y, x ≤ y → f a x ≤ f a y) ↔ (a ∈ Set.Ici (1 : ℝ)) := by
  sorry

end NUMINAMATH_GPT_increasing_function_l1099_109979


namespace NUMINAMATH_GPT_math_score_computation_l1099_109963

def comprehensive_score 
  (reg_score : ℕ) (mid_score : ℕ) (fin_score : ℕ) 
  (reg_weight : ℕ) (mid_weight : ℕ) (fin_weight : ℕ) 
  : ℕ :=
  (reg_score * reg_weight + mid_score * mid_weight + fin_score * fin_weight) 
  / (reg_weight + mid_weight + fin_weight)

theorem math_score_computation :
  comprehensive_score 80 80 85 3 3 4 = 82 := by
sorry

end NUMINAMATH_GPT_math_score_computation_l1099_109963


namespace NUMINAMATH_GPT_total_dogs_is_28_l1099_109984

def number_of_boxes : ℕ := 7
def dogs_per_box : ℕ := 4
def total_dogs (boxes : ℕ) (dogs_in_each : ℕ) : ℕ := boxes * dogs_in_each

theorem total_dogs_is_28 : total_dogs number_of_boxes dogs_per_box = 28 :=
by
  sorry

end NUMINAMATH_GPT_total_dogs_is_28_l1099_109984


namespace NUMINAMATH_GPT_normal_vector_to_line_l1099_109982

theorem normal_vector_to_line : 
  ∀ (x y : ℝ), x - 3 * y + 6 = 0 → (1, -3) = (1, -3) :=
by
  intros x y h_line
  sorry

end NUMINAMATH_GPT_normal_vector_to_line_l1099_109982


namespace NUMINAMATH_GPT_graph_does_not_pass_through_third_quadrant_l1099_109974

theorem graph_does_not_pass_through_third_quadrant (k x y : ℝ) (hk : k < 0) :
  y = k * x - k → (¬ (x < 0 ∧ y < 0)) :=
by
  sorry

end NUMINAMATH_GPT_graph_does_not_pass_through_third_quadrant_l1099_109974


namespace NUMINAMATH_GPT_unique_trivial_solution_of_linear_system_l1099_109966

variable {R : Type*} [Field R]

theorem unique_trivial_solution_of_linear_system (a b c x y z : R)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_system : x + a * y + a^2 * z = 0 ∧ x + b * y + b^2 * z = 0 ∧ x + c * y + c^2 * z = 0) :
  x = 0 ∧ y = 0 ∧ z = 0 := sorry

end NUMINAMATH_GPT_unique_trivial_solution_of_linear_system_l1099_109966
