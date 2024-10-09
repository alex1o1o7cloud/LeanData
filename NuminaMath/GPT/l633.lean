import Mathlib

namespace fraction_blue_after_doubling_l633_63352

theorem fraction_blue_after_doubling (x : ℕ) (h1 : ∃ x, (2 : ℚ) / 3 * x + (1 : ℚ) / 3 * x = x) :
  ((2 * (2 / 3 * x)) / ((2 / 3 * x) + (1 / 3 * x))) = (4 / 5) := by
  sorry

end fraction_blue_after_doubling_l633_63352


namespace valid_P_values_l633_63359

/-- 
Construct a 3x3 grid of distinct natural numbers where the product of the numbers 
in each row and each column is equal. Verify the valid values of P among the given set.
-/
theorem valid_P_values (P : ℕ) :
  (∃ (a b c d e f g h i : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ 
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ 
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ 
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ 
    g ≠ h ∧ g ≠ i ∧ 
    h ≠ i ∧ 
    a * b * c = P ∧ 
    d * e * f = P ∧ 
    g * h * i = P ∧ 
    a * d * g = P ∧ 
    b * e * h = P ∧ 
    c * f * i = P ∧ 
    P = (Nat.sqrt ((1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9)) )) ↔ P = 1998 ∨ P = 2000 :=
sorry

end valid_P_values_l633_63359


namespace distinct_divisors_in_set_l633_63332

theorem distinct_divisors_in_set (p : ℕ) (hp : Nat.Prime p) (hp5 : 5 < p) :
  ∃ (x y : ℕ), x ∈ {p - n^2 | n : ℕ} ∧ y ∈ {p - n^2 | n : ℕ} ∧ x ≠ y ∧ x ≠ 1 ∧ x ∣ y :=
by
  sorry

end distinct_divisors_in_set_l633_63332


namespace rope_total_length_is_54m_l633_63312

noncomputable def totalRopeLength : ℝ :=
  let horizontalDistance : ℝ := 16
  let heightAB : ℝ := 18
  let heightCD : ℝ := 30
  let ropeBC := Real.sqrt (horizontalDistance^2 + (heightCD - heightAB)^2)
  let ropeAC := Real.sqrt (horizontalDistance^2 + heightCD^2)
  ropeBC + ropeAC

theorem rope_total_length_is_54m : totalRopeLength = 54 := sorry

end rope_total_length_is_54m_l633_63312


namespace coconut_grove_l633_63384

theorem coconut_grove (x N : ℕ) (h1 : (x + 4) * 60 + x * N + (x - 4) * 180 = 3 * x * 100) (hx : x = 8) : N = 120 := 
by
  subst hx
  sorry

end coconut_grove_l633_63384


namespace remainder_correct_l633_63355

open Polynomial

noncomputable def polynomial_remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  p % q

theorem remainder_correct : polynomial_remainder (X^6 - 2*X^5 + X^4 - X^2 - 2*X + 1)
                                                  ((X^2 - 1)*(X - 2)*(X + 2))
                                                = 2*X^3 - 9*X^2 + 3*X + 2 :=
by
  sorry

end remainder_correct_l633_63355


namespace shaded_area_of_rotated_semicircle_l633_63362

noncomputable def area_of_shaded_region (R : ℝ) (α : ℝ) : ℝ :=
  (1 / 2) * (2 * R) ^ 2 * (α / (2 * Real.pi))

theorem shaded_area_of_rotated_semicircle (R : ℝ) (α : ℝ) (h : α = Real.pi / 9) :
  area_of_shaded_region R α = 2 * Real.pi * R ^ 2 / 9 :=
by
  sorry

end shaded_area_of_rotated_semicircle_l633_63362


namespace part_I_part_II_l633_63325

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - x * Real.log x

theorem part_I (a : ℝ) :
  (∀ x > 0, 0 ≤ a * Real.exp x - (1 + Real.log x)) ↔ a ≥ 1 / Real.exp 1 :=
sorry

theorem part_II (a : ℝ) (h : a ≥ 2 / Real.exp 2) (x : ℝ) (hx : x > 0) :
  f a x > 0 :=
sorry

end part_I_part_II_l633_63325


namespace range_of_a_l633_63316

noncomputable def f (x : ℝ) := x * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ -x^2 + a*x - 6) → a ≤ 5 + Real.log 2 :=
by
  sorry

end range_of_a_l633_63316


namespace mia_socks_problem_l633_63357

theorem mia_socks_problem (x y z w : ℕ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hw : 1 ≤ w)
  (h1 : x + y + z + w = 16) (h2 : x + 2*y + 3*z + 4*w = 36) : x = 3 :=
sorry

end mia_socks_problem_l633_63357


namespace silver_excess_in_third_chest_l633_63360

theorem silver_excess_in_third_chest :
  ∀ (x1 y1 x2 y2 x3 y3 : ℕ),
    x1 + x2 + x3 = 40 →
    y1 + y2 + y3 = 40 →
    x1 = y1 + 7 →
    y2 = x2 - 15 →
    y3 = x3 + 22 :=
by
  intros x1 y1 x2 y2 x3 y3 h1 h2 h3 h4
  sorry

end silver_excess_in_third_chest_l633_63360


namespace box_dimensions_l633_63382

theorem box_dimensions (x : ℝ) (bow_length_top bow_length_side : ℝ)
  (h1 : bow_length_top = 156 - 6 * x)
  (h2 : bow_length_side = 178 - 7 * x)
  (h_eq : bow_length_top = bow_length_side) :
  x = 22 :=
by sorry

end box_dimensions_l633_63382


namespace find_number_l633_63321

theorem find_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 14) : 0.40 * N = 168 :=
sorry

end find_number_l633_63321


namespace simplify_expression_l633_63372

theorem simplify_expression : (225 / 10125) * 45 = 1 := by
  sorry

end simplify_expression_l633_63372


namespace energy_conservation_l633_63394

-- Define the conditions
variables (m : ℝ) (v_train v_ball : ℝ)
-- The speed of the train and the ball, converted to m/s
variables (v := 60 * 1000 / 3600) -- 60 km/h in m/s
variables (E_initial : ℝ := 0.5 * m * (v ^ 2))

-- Kinetic energy of the ball when thrown in the same direction
variables (E_same_direction : ℝ := 0.5 * m * (2 * v)^2)

-- Kinetic energy of the ball when thrown in the opposite direction
variables (E_opposite_direction : ℝ := 0.5 * m * (0)^2)

-- Prove energy conservation
theorem energy_conservation : 
  (E_same_direction - E_initial) + (E_opposite_direction - E_initial) = 0 :=
sorry

end energy_conservation_l633_63394


namespace basketball_games_count_l633_63300

noncomputable def tokens_per_game : ℕ := 3
noncomputable def total_tokens : ℕ := 18
noncomputable def air_hockey_games : ℕ := 2
noncomputable def air_hockey_tokens := air_hockey_games * tokens_per_game
noncomputable def remaining_tokens := total_tokens - air_hockey_tokens

theorem basketball_games_count :
  (remaining_tokens / tokens_per_game) = 4 := by
  sorry

end basketball_games_count_l633_63300


namespace problem1_problem2_l633_63310

-- Define conditions for Problem 1
def problem1_cond (x : ℝ) : Prop :=
  x ≠ 0 ∧ 2 * x ≠ 1

-- Statement for Problem 1
theorem problem1 (x : ℝ) (h : problem1_cond x) :
  (2 / x = 3 / (2 * x - 1)) ↔ x = 2 := by
  sorry

-- Define conditions for Problem 2
def problem2_cond (x : ℝ) : Prop :=
  x ≠ 2 

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h : problem2_cond x) :
  ((x - 3) / (x - 2) + 1 = 3 / (2 - x)) ↔ x = 1 := by
  sorry

end problem1_problem2_l633_63310


namespace trig_expression_simplification_l633_63311

theorem trig_expression_simplification (α : Real) :
  Real.cos (3/2 * Real.pi + 4 * α)
  + Real.sin (3 * Real.pi - 8 * α)
  - Real.sin (4 * Real.pi - 12 * α)
  = 4 * Real.cos (2 * α) * Real.cos (4 * α) * Real.sin (6 * α) :=
sorry

end trig_expression_simplification_l633_63311


namespace factorize_x_cubic_l633_63344

-- Define the function and the condition
def factorize (x : ℝ) : Prop := x^3 - 9 * x = x * (x + 3) * (x - 3)

-- Prove the factorization property
theorem factorize_x_cubic (x : ℝ) : factorize x :=
by
  sorry

end factorize_x_cubic_l633_63344


namespace initial_ducks_l633_63337

theorem initial_ducks (D : ℕ) (h1 : D + 20 = 33) : D = 13 :=
by sorry

end initial_ducks_l633_63337


namespace tangent_expression_l633_63339

open Real

theorem tangent_expression
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (geom_seq : ∀ n m, a (n + m) = a n * a m) 
  (arith_seq : ∀ n, b (n + 1) = b n + (b 2 - b 1))
  (cond1 : a 1 * a 6 * a 11 = -3 * sqrt 3)
  (cond2 : b 1 + b 6 + b 11 = 7 * pi) :
  tan ( (b 3 + b 9) / (1 - a 4 * a 8) ) = -sqrt 3 :=
sorry

end tangent_expression_l633_63339


namespace curved_surface_area_of_sphere_l633_63358

theorem curved_surface_area_of_sphere (r : ℝ) (h : r = 4) : 4 * π * r^2 = 64 * π :=
by
  rw [h, sq]
  norm_num
  sorry

end curved_surface_area_of_sphere_l633_63358


namespace original_savings_calculation_l633_63368

theorem original_savings_calculation (S : ℝ) (F : ℝ) (T : ℝ) 
  (h1 : 0.8 * F = (3 / 4) * S)
  (h2 : 1.1 * T = 150)
  (h3 : (1 / 4) * S = T) :
  S = 545.44 :=
by
  sorry

end original_savings_calculation_l633_63368


namespace sum_of_primes_less_than_20_is_77_l633_63369

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l633_63369


namespace total_ages_l633_63390

-- Definitions of the conditions
variables (A B : ℕ) (x : ℕ)

-- Condition 1: 10 years ago, A was half of B in age.
def condition1 : Prop := A - 10 = 1/2 * (B - 10)

-- Condition 2: The ratio of their present ages is 3:4.
def condition2 : Prop := A = 3 * x ∧ B = 4 * x

-- Main theorem to prove
theorem total_ages (A B : ℕ) (x : ℕ) (h1 : condition1 A B) (h2 : condition2 A B x) : A + B = 35 := 
by
  sorry

end total_ages_l633_63390


namespace P_is_subtract_0_set_P_is_not_subtract_1_set_no_subtract_2_set_exists_all_subtract_1_sets_l633_63324

def is_subtract_set (T : Set ℕ) (i : ℕ) := T ⊆ Set.univ ∧ T ≠ {1} ∧ (∀ {x y : ℕ}, x ∈ Set.univ → y ∈ Set.univ → x + y ∈ T → x * y - i ∈ T)

theorem P_is_subtract_0_set : is_subtract_set {1, 2} 0 := sorry

theorem P_is_not_subtract_1_set : ¬ is_subtract_set {1, 2} 1 := sorry

theorem no_subtract_2_set_exists : ¬∃ T : Set ℕ, is_subtract_set T 2 := sorry

theorem all_subtract_1_sets : ∀ T : Set ℕ, is_subtract_set T 1 ↔ T = {1, 3} ∨ T = {1, 3, 5} := sorry

end P_is_subtract_0_set_P_is_not_subtract_1_set_no_subtract_2_set_exists_all_subtract_1_sets_l633_63324


namespace simplification_of_expression_l633_63377

variable {a b : ℚ}

theorem simplification_of_expression (h1a : a ≠ 0) (h1b : b ≠ 0) (h2 : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ( (3 * a)⁻¹ - (b / 3)⁻¹ ) = -(a * b)⁻¹ := 
sorry

end simplification_of_expression_l633_63377


namespace Quentin_chickens_l633_63375

variable (C S Q : ℕ)

theorem Quentin_chickens (h1 : C = 37)
    (h2 : S = 3 * C - 4)
    (h3 : Q + S + C = 383) :
    (Q = 2 * S + 32) :=
by
  sorry

end Quentin_chickens_l633_63375


namespace correctness_statement_l633_63319

-- Given points A, B, C are on the specific parabola
variable (a c x1 x2 x3 y1 y2 y3 : ℝ)
variable (ha : a < 0) -- a < 0 since the parabola opens upwards
variable (hA : y1 = - (a / 4) * x1^2 + a * x1 + c)
variable (hB : y2 = a + c) -- B is the vertex
variable (hC : y3 = - (a / 4) * x3^2 + a * x3 + c)
variable (hOrder : y1 > y3 ∧ y3 ≥ y2)

theorem correctness_statement : abs (x1 - x2) > abs (x3 - x2) :=
sorry

end correctness_statement_l633_63319


namespace number_of_children_l633_63371
-- Import the entirety of the Mathlib library

-- Define the conditions and the theorem to be proven
theorem number_of_children (C n : ℕ) 
  (h1 : C = 8 * n + 4) 
  (h2 : C = 11 * (n - 1)) : 
  n = 5 :=
by sorry

end number_of_children_l633_63371


namespace average_of_a_and_b_l633_63381

theorem average_of_a_and_b (a b c M : ℝ)
  (h1 : (a + b) / 2 = M)
  (h2 : (b + c) / 2 = 180)
  (h3 : a - c = 200) : 
  M = 280 :=
sorry

end average_of_a_and_b_l633_63381


namespace quadratic_inequality_always_positive_l633_63326

theorem quadratic_inequality_always_positive (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
by sorry

end quadratic_inequality_always_positive_l633_63326


namespace max_value_sqrt_sum_l633_63330

open Real

noncomputable def max_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 7) : ℝ :=
  sqrt (3 * x + 1) + sqrt (3 * y + 1) + sqrt (3 * z + 1)

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 7) :
  max_sqrt_sum x y z h1 h2 h3 h_sum ≤ 3 * sqrt 8 :=
sorry

end max_value_sqrt_sum_l633_63330


namespace initial_oranges_l633_63335

theorem initial_oranges (X : ℕ) : 
  (X - 9 + 38 = 60) → X = 31 :=
sorry

end initial_oranges_l633_63335


namespace factorized_sum_is_33_l633_63315

theorem factorized_sum_is_33 (p q r : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 21 * x + 110 = (x + p) * (x + q))
  (h2 : ∀ x : ℤ, x^2 - 23 * x + 132 = (x - q) * (x - r)) : 
  p + q + r = 33 := by
  sorry

end factorized_sum_is_33_l633_63315


namespace problem_solution_l633_63317

def f (x m : ℝ) : ℝ :=
  3 * x ^ 2 + m * (m - 6) * x + 5

theorem problem_solution (m n : ℝ) :
  (f 1 m > 0) ∧ (∀ x : ℝ, -1 < x ∧ x < 4 → f x m < n) ↔ (m = 3 ∧ n = 17) :=
by sorry

end problem_solution_l633_63317


namespace food_waste_in_scientific_notation_l633_63385

-- Given condition that 1 billion equals 10^9
def billion : ℕ := 10 ^ 9

-- Problem statement: expressing 530 billion kilograms in scientific notation
theorem food_waste_in_scientific_notation :
  (530 * billion : ℝ) = 5.3 * 10^10 := 
  sorry

end food_waste_in_scientific_notation_l633_63385


namespace gcd_102_238_l633_63333

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l633_63333


namespace original_price_l633_63318

theorem original_price (a b x : ℝ) (h : (x - a) * 0.60 = b) : x = (5 / 3 * b) + a :=
  sorry

end original_price_l633_63318


namespace quadratic_inequality_solutions_l633_63322

theorem quadratic_inequality_solutions {k : ℝ} (h1 : 0 < k) (h2 : k < 16) :
  ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end quadratic_inequality_solutions_l633_63322


namespace balance_scale_with_blue_balls_l633_63395

variables (G Y W B : ℝ)

-- Conditions
def green_to_blue := 4 * G = 8 * B
def yellow_to_blue := 3 * Y = 8 * B
def white_to_blue := 5 * B = 3 * W

-- Proof problem statement
theorem balance_scale_with_blue_balls (h1 : green_to_blue G B) (h2 : yellow_to_blue Y B) (h3 : white_to_blue W B) : 
  3 * G + 3 * Y + 3 * W = 19 * B :=
by sorry

end balance_scale_with_blue_balls_l633_63395


namespace minimum_value_y_l633_63347

theorem minimum_value_y (x y : ℕ) (h1 : x + y = 64) (h2 : 3 * x + 4 * y = 200) : y = 8 :=
by
  sorry

end minimum_value_y_l633_63347


namespace height_at_15_inches_l633_63304

-- Define the conditions
def parabolic_eq (a x : ℝ) : ℝ := a * x^2 + 24
noncomputable def a : ℝ := -2 / 75
def x : ℝ := 15
def expected_y : ℝ := 18

-- Lean 4 statement
theorem height_at_15_inches :
  parabolic_eq a x = expected_y :=
by
  sorry

end height_at_15_inches_l633_63304


namespace problem_1_simplified_problem_2_simplified_l633_63351

noncomputable def problem_1 : ℝ :=
  2 * Real.sqrt 18 - Real.sqrt 50 + (1/2) * Real.sqrt 32

theorem problem_1_simplified : problem_1 = 3 * Real.sqrt 2 :=
  sorry

noncomputable def problem_2 : ℝ :=
  (Real.sqrt 5 + Real.sqrt 6) * (Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - 1)^2

theorem problem_2_simplified : problem_2 = -7 + 2 * Real.sqrt 5 :=
  sorry

end problem_1_simplified_problem_2_simplified_l633_63351


namespace melanie_total_dimes_l633_63354

-- Definitions based on the problem conditions
def initial_dimes : ℕ := 7
def dad_dimes : ℕ := 8
def mom_dimes : ℕ := 4

def total_dimes : ℕ := initial_dimes + dad_dimes + mom_dimes

-- Proof statement based on the correct answer
theorem melanie_total_dimes : total_dimes = 19 := by 
  -- Proof here is omitted as per instructions
  sorry

end melanie_total_dimes_l633_63354


namespace terminating_decimal_representation_l633_63353

-- Definitions derived from conditions
def given_fraction : ℚ := 53 / (2^2 * 5^3)

-- The theorem we aim to state that expresses the question and correct answer
theorem terminating_decimal_representation : given_fraction = 0.106 :=
by
  sorry  -- proof goes here

end terminating_decimal_representation_l633_63353


namespace percentage_increase_in_weight_l633_63365

theorem percentage_increase_in_weight :
  ∀ (num_plates : ℕ) (weight_per_plate lowered_weight : ℝ),
    num_plates = 10 →
    weight_per_plate = 30 →
    lowered_weight = 360 →
    ((lowered_weight - num_plates * weight_per_plate) / (num_plates * weight_per_plate)) * 100 = 20 :=
by
  intros num_plates weight_per_plate lowered_weight h_num_plates h_weight_per_plate h_lowered_weight
  sorry

end percentage_increase_in_weight_l633_63365


namespace beetles_eaten_per_day_l633_63334
-- Import the Mathlib library

-- Declare the conditions as constants
def bird_eats_beetles_per_day : Nat := 12
def snake_eats_birds_per_day : Nat := 3
def jaguar_eats_snakes_per_day : Nat := 5
def number_of_jaguars : Nat := 6

-- Define the theorem and provide the expected proof
theorem beetles_eaten_per_day :
  12 * (3 * (5 * 6)) = 1080 := by
  sorry

end beetles_eaten_per_day_l633_63334


namespace one_third_greater_than_333_l633_63331

theorem one_third_greater_than_333 :
  (1 : ℝ) / 3 > (333 : ℝ) / 1000 - 1 / 3000 :=
sorry

end one_third_greater_than_333_l633_63331


namespace range_of_a_l633_63328

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ), |x + 3| - |x - 1| ≤ a ^ 2 - 3 * a) ↔ a ≤ -1 ∨ a ≥ 4 :=
by
  sorry

end range_of_a_l633_63328


namespace min_additional_games_l633_63366

-- Definitions of parameters
def initial_total_games : ℕ := 5
def initial_falcon_wins : ℕ := 2
def win_percentage_threshold : ℚ := 91 / 100

-- Theorem stating the minimum value for N
theorem min_additional_games (N : ℕ) : (initial_falcon_wins + N : ℚ) / (initial_total_games + N : ℚ) ≥ win_percentage_threshold → N ≥ 29 :=
by
  sorry

end min_additional_games_l633_63366


namespace symmetry_axis_g_l633_63314

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 3) - Real.pi / 3)

theorem symmetry_axis_g :
  ∃ k : ℤ, (x = k * Real.pi / 2 + Real.pi / 4) := sorry

end symmetry_axis_g_l633_63314


namespace original_ratio_l633_63336

theorem original_ratio (x y : ℤ) (h1 : y = 24) (h2 : (x + 6) / y = 1 / 2) : x / y = 1 / 4 := by
  sorry

end original_ratio_l633_63336


namespace coordinates_of_B_l633_63349

theorem coordinates_of_B (a : ℝ) (h : a - 2 = 0) : (a + 2, a - 1) = (4, 1) :=
by
  sorry

end coordinates_of_B_l633_63349


namespace scalene_triangle_angle_obtuse_l633_63323

theorem scalene_triangle_angle_obtuse (a b c : ℝ) 
  (h_scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_longest : a > b ∧ a > c)
  (h_obtuse_angle : a^2 > b^2 + c^2) : 
  ∃ A : ℝ, A = (Real.pi / 2) ∧ (b^2 + c^2 - a^2) / (2 * b * c) < 0 := 
sorry

end scalene_triangle_angle_obtuse_l633_63323


namespace sin_90_eq_one_l633_63380

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l633_63380


namespace smallest_natural_number_condition_l633_63327

theorem smallest_natural_number_condition (N : ℕ) : 
  (∀ k : ℕ, (10^6 - 1) * k = (10^54 - 1) / 9 → k < N) →
  N = 111112 :=
by
  sorry

end smallest_natural_number_condition_l633_63327


namespace floor_width_is_120_l633_63350

def tile_length := 25 -- cm
def tile_width := 16 -- cm
def floor_length := 180 -- cm
def max_tiles := 54

theorem floor_width_is_120 :
  ∃ (W : ℝ), W = 120 ∧ (floor_length / tile_width) * W = max_tiles * (tile_length * tile_width) := 
sorry

end floor_width_is_120_l633_63350


namespace equal_areas_of_shapes_l633_63376

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def semicircle_area (r : ℝ) : ℝ :=
  (Real.pi * r^2) / 2

noncomputable def sector_area (theta : ℝ) (r : ℝ) : ℝ :=
  (theta / (2 * Real.pi)) * Real.pi * r^2

noncomputable def shape1_area (s : ℝ) : ℝ :=
  hexagon_area s + 3 * semicircle_area (s / 4) - 6 * sector_area (Real.pi / 3) (s / 4)

noncomputable def shape2_area (s : ℝ) : ℝ :=
  hexagon_area s + 6 * sector_area (2 * Real.pi / 3) (s / 4) - 3 * semicircle_area (s / 4)

theorem equal_areas_of_shapes (s : ℝ) : shape1_area s = shape2_area s :=
by {
  sorry
}

end equal_areas_of_shapes_l633_63376


namespace investment_worth_l633_63348

noncomputable def initial_investment (total_earning : ℤ) : ℤ := total_earning / 2

noncomputable def current_worth (initial_investment total_earning : ℤ) : ℤ :=
  initial_investment + total_earning

theorem investment_worth (monthly_earning : ℤ) (months : ℤ) (earnings : ℤ)
  (h1 : monthly_earning * months = earnings)
  (h2 : earnings = 2 * initial_investment earnings) :
  current_worth (initial_investment earnings) earnings = 90 := 
by
  -- We proceed to show the current worth is $90
  -- Proof will be constructed here
  sorry
  
end investment_worth_l633_63348


namespace find_y_l633_63364

variables (y : ℝ)

def rectangle_vertices (A B C D : (ℝ × ℝ)) : Prop :=
  (A = (-2, y)) ∧ (B = (10, y)) ∧ (C = (-2, 1)) ∧ (D = (10, 1))

def rectangle_area (length height : ℝ) : Prop :=
  length * height = 108

def positive_value (x : ℝ) : Prop :=
  0 < x

theorem find_y (A B C D : (ℝ × ℝ)) (hV : rectangle_vertices y A B C D) (hA : rectangle_area 12 (y - 1)) (hP : positive_value y) :
  y = 10 :=
sorry

end find_y_l633_63364


namespace principal_amount_l633_63329

theorem principal_amount
(SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
(h₀ : SI = 800)
(h₁ : R = 0.08)
(h₂ : T = 1)
(h₃ : SI = P * R * T) : P = 10000 :=
by
  sorry

end principal_amount_l633_63329


namespace intersection_A_B_l633_63398

def A : Set ℝ := { x | |x - 1| < 2 }
def B : Set ℝ := { x | Real.log x / Real.log 2 ≤ 1 }

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x ≤ 2} := 
sorry

end intersection_A_B_l633_63398


namespace prod_of_three_consec_ints_l633_63343

theorem prod_of_three_consec_ints (a : ℤ) (h : a + (a + 1) + (a + 2) = 27) :
  a * (a + 1) * (a + 2) = 720 :=
by
  sorry

end prod_of_three_consec_ints_l633_63343


namespace minimum_problems_45_l633_63378

-- Define the types for problems and their corresponding points
structure Problem :=
(points : ℕ)

def isValidScore (s : ℕ) : Prop :=
  ∃ x y z : ℕ, 3 * x + 8 * y + 10 * z = s

def minimumProblems (s : ℕ) (min_problems : ℕ) : Prop :=
  ∃ x y z : ℕ, 3 * x + 8 * y + 10 * z = s ∧ x + y + z = min_problems

-- Main statement
theorem minimum_problems_45 : minimumProblems 45 6 :=
by 
  sorry

end minimum_problems_45_l633_63378


namespace angles_MAB_NAC_l633_63370

/-- Given equal chords AB and AC, and a tangent MAN, with arc BC's measure (excluding point A) being 200 degrees,
prove that the angles MAB and NAC are either 40 degrees or 140 degrees. -/
theorem angles_MAB_NAC (AB AC : ℝ) (tangent_MAN : Prop)
    (arc_BC_measure : ∀ A : ℝ , A = 200) : 
    ∃ θ : ℝ, (θ = 40 ∨ θ = 140) :=
by
  sorry

end angles_MAB_NAC_l633_63370


namespace max_ways_to_ascend_descend_l633_63303

theorem max_ways_to_ascend_descend :
  let east_paths := 2
  let west_paths := 1
  let south_paths := 3
  let north_paths := 4

  let descend_from_east := west_paths + south_paths + north_paths
  let descend_from_west := east_paths + south_paths + north_paths
  let descend_from_south := east_paths + west_paths + north_paths
  let descend_from_north := east_paths + west_paths + south_paths

  let ways_from_east := east_paths * descend_from_east
  let ways_from_west := west_paths * descend_from_west
  let ways_from_south := south_paths * descend_from_south
  let ways_from_north := north_paths * descend_from_north

  max ways_from_east (max ways_from_west (max ways_from_south ways_from_north)) = 24 := 
by
  -- Insert the proof here
  sorry

end max_ways_to_ascend_descend_l633_63303


namespace g_ge_one_l633_63367

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + Real.log x + 4

noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

theorem g_ge_one (x : ℝ) (h : 0 < x) : g x ≥ 1 :=
sorry

end g_ge_one_l633_63367


namespace length_of_AB_l633_63308

theorem length_of_AB :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}
  let focus := (Real.sqrt 3, 0)
  let line := {p : ℝ × ℝ | p.2 = p.1 - Real.sqrt 3}
  ∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line ∧ B ∈ line ∧
  (dist A B = 8 / 5) :=
by
  sorry

end length_of_AB_l633_63308


namespace sharks_at_other_beach_is_12_l633_63396

-- Define the conditions
def cape_may_sharks := 32
def sharks_other_beach (S : ℕ) := 2 * S + 8

-- Statement to prove
theorem sharks_at_other_beach_is_12 (S : ℕ) (h : cape_may_sharks = sharks_other_beach S) : S = 12 :=
by
  -- Sorry statement to skip the proof part
  sorry

end sharks_at_other_beach_is_12_l633_63396


namespace find_line_equation_through_two_points_find_circle_equation_tangent_to_x_axis_l633_63309

open Real

-- Given conditions
def line_passes_through (x1 y1 x2 y2 : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l x1 y1 ∧ l x2 y2

def circle_tangent_to_x_axis (center_x center_y : ℝ) (r : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  C center_x center_y ∧ center_y = r

-- We want to prove:
-- 1. The equation of line l is x - 2y = 0
theorem find_line_equation_through_two_points:
  ∃ l : ℝ → ℝ → Prop, line_passes_through 2 1 6 3 l ∧ (∀ x y, l x y ↔ x - 2 * y = 0) :=
  sorry

-- 2. The equation of circle C is (x - 2)^2 + (y - 1)^2 = 1
theorem find_circle_equation_tangent_to_x_axis:
  ∃ C : ℝ → ℝ → Prop, circle_tangent_to_x_axis 2 1 1 C ∧ (∀ x y, C x y ↔ (x - 2)^2 + (y - 1)^2 = 1) :=
  sorry

end find_line_equation_through_two_points_find_circle_equation_tangent_to_x_axis_l633_63309


namespace trapezoid_EFBA_area_l633_63361

theorem trapezoid_EFBA_area {a : ℚ} (AE BF : ℚ) (area_ABCD : ℚ) (column_areas : List ℚ)
  (h_grid : column_areas = [a, 2 * a, 4 * a, 8 * a])
  (h_total_area : 3 * (a + 2 * a + 4 * a + 8 * a) = 48)
  (h_AE : AE = 2)
  (h_BF : BF = 4) :
  let AFGB_area := 15 * a
  let triangle_EF_area := 7 * a
  let total_trapezoid_area := AFGB_area + (triangle_EF_area / 2)
  total_trapezoid_area = 352 / 15 :=
by
  sorry

end trapezoid_EFBA_area_l633_63361


namespace power_of_product_l633_63393

variable (x y : ℝ)

theorem power_of_product (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 :=
  sorry

end power_of_product_l633_63393


namespace cos_Z_value_l633_63301

-- The conditions given in the problem
def sin_X := 4 / 5
def cos_Y := 3 / 5

-- The theorem we want to prove
theorem cos_Z_value (sin_X : ℝ) (cos_Y : ℝ) (hX : sin_X = 4/5) (hY : cos_Y = 3/5) : 
  ∃ cos_Z : ℝ, cos_Z = 7 / 25 :=
by
  -- Attach all conditions and solve
  sorry

end cos_Z_value_l633_63301


namespace divisor_greater_than_8_l633_63379

-- Define the condition that remainder is 8
def remainder_is_8 (n m : ℕ) : Prop :=
  n % m = 8

-- Theorem: If n divided by m has remainder 8, then m must be greater than 8
theorem divisor_greater_than_8 (m : ℕ) (hm : m ≤ 8) : ¬ exists n, remainder_is_8 n m :=
by
  sorry

end divisor_greater_than_8_l633_63379


namespace find_x_l633_63306

theorem find_x (x y : ℕ) (h1 : y = 30) (h2 : x / y = 5 / 2) : x = 75 := by
  sorry

end find_x_l633_63306


namespace compare_f_values_l633_63346

noncomputable def f (x : ℝ) : ℝ := Real.sin x - x

theorem compare_f_values : 
  f (-π / 4) > f 1 ∧ f 1 > f (π / 3) := 
sorry

end compare_f_values_l633_63346


namespace evaluate_expression_l633_63313

theorem evaluate_expression : (723 * 723) - (722 * 724) = 1 :=
by
  sorry

end evaluate_expression_l633_63313


namespace factor_polynomial_l633_63320

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := 
by sorry

end factor_polynomial_l633_63320


namespace find_u_plus_v_l633_63374

theorem find_u_plus_v (u v : ℚ) (h1: 5 * u - 3 * v = 26) (h2: 3 * u + 5 * v = -19) :
  u + v = -101 / 34 :=
sorry

end find_u_plus_v_l633_63374


namespace cosine_inequality_l633_63399

theorem cosine_inequality
  (x y z : ℝ)
  (hx : 0 < x ∧ x < π / 2)
  (hy : 0 < y ∧ y < π / 2)
  (hz : 0 < z ∧ z < π / 2) :
  (x * Real.cos x + y * Real.cos y + z * Real.cos z) / (x + y + z) ≤
  (Real.cos x + Real.cos y + Real.cos z) / 3 := sorry

end cosine_inequality_l633_63399


namespace price_of_pants_l633_63383

theorem price_of_pants (S P H : ℝ) (h1 : 0.8 * S + P + H = 340) (h2 : S = (3 / 4) * P) (h3 : H = P + 10) : P = 91.67 :=
by sorry

end price_of_pants_l633_63383


namespace evaluate_expression_l633_63373

theorem evaluate_expression : 6 + 4 / 2 = 8 :=
by
  sorry

end evaluate_expression_l633_63373


namespace ways_to_change_12_dollars_into_nickels_and_quarters_l633_63363

theorem ways_to_change_12_dollars_into_nickels_and_quarters :
  ∃ n q : ℕ, 5 * n + 25 * q = 1200 ∧ n > 0 ∧ q > 0 ∧ ∀ q', (q' ≥ 1 ∧ q' ≤ 47) ↔ (n = 240 - 5 * q') :=
by
  sorry

end ways_to_change_12_dollars_into_nickels_and_quarters_l633_63363


namespace determine_M_l633_63340

noncomputable def M : Set ℤ :=
  {a | ∃ k : ℕ, k > 0 ∧ 6 = k * (5 - a)}

theorem determine_M : M = {-1, 2, 3, 4} :=
  sorry

end determine_M_l633_63340


namespace rectangle_perimeter_l633_63391

noncomputable def perimeter_rectangle (x y : ℝ) : ℝ := 2 * (x + y)

theorem rectangle_perimeter
  (x y a b : ℝ)
  (H1 : x * y = 2006)
  (H2 : x + y = 2 * a)
  (H3 : x^2 + y^2 = 4 * (a^2 - b^2))
  (b_val : b = Real.sqrt 1003)
  (a_val : a = 2 * Real.sqrt 1003) :
  perimeter_rectangle x y = 8 * Real.sqrt 1003 := by
  sorry

end rectangle_perimeter_l633_63391


namespace solve_for_x_l633_63387

theorem solve_for_x : 
  (∀ x : ℝ, x ≠ -2 → (x^2 - x - 2) / (x + 2) = x - 1 ↔ x = 0) := 
by 
  sorry

end solve_for_x_l633_63387


namespace set_diff_N_M_l633_63389

universe u

def set_difference {α : Type u} (A B : Set α) : Set α :=
  { x | x ∈ A ∧ x ∉ B }

def M : Set ℕ := { 1, 2, 3, 4, 5 }
def N : Set ℕ := { 1, 2, 3, 7 }

theorem set_diff_N_M : set_difference N M = { 7 } :=
  by
    sorry

end set_diff_N_M_l633_63389


namespace Sarahs_score_l633_63342

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end Sarahs_score_l633_63342


namespace beyonce_total_songs_l633_63338

theorem beyonce_total_songs :
  let singles := 12
  let albums := 4
  let songs_per_album := 18 + 14
  let total_album_songs := albums * songs_per_album
  let total_songs := total_album_songs + singles
  total_songs = 140 := by
  let singles := 12
  let albums := 4
  let songs_per_album := 18 + 14
  let total_album_songs := albums * songs_per_album
  let total_songs := total_album_songs + singles
  sorry

end beyonce_total_songs_l633_63338


namespace eel_count_l633_63305

theorem eel_count 
  (x y z : ℕ)
  (h1 : y + z = 12)
  (h2 : x + z = 14)
  (h3 : x + y = 16) : 
  x + y + z = 21 := 
by 
  sorry

end eel_count_l633_63305


namespace students_failed_l633_63341

theorem students_failed (Q : ℕ) (x : ℕ) (h1 : 4 * Q < 56) (h2 : x = Nat.lcm 3 (Nat.lcm 7 2)) (h3 : x < 56) :
  let R := x - (x / 3 + x / 7 + x / 2) 
  R = 1 := 
by
  sorry

end students_failed_l633_63341


namespace problem_part1_problem_part2_problem_part3_l633_63345

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem problem_part1 : f 1 = 5 / 2 ∧ f 2 = 17 / 4 := 
by
  sorry

theorem problem_part2 : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem problem_part3 : ∀ x1 x2 : ℝ, x1 < x2 → x1 < 0 → x2 < 0 → f x1 > f x2 :=
by
  sorry

end problem_part1_problem_part2_problem_part3_l633_63345


namespace simplify_expression_l633_63388

theorem simplify_expression : 5 * (18 / -9) * (24 / 36) = -(20 / 3) :=
by
  sorry

end simplify_expression_l633_63388


namespace num_valid_colorings_l633_63302

namespace ColoringGrid

-- Definition of the grid and the constraint.
-- It's easier to represent with simply 9 nodes and adjacent constraints, however,
-- we will declare the conditions and result as discussed.

def Grid := Fin 3 × Fin 3
def Colors := Fin 2

-- Define adjacency relationship
def adjacent (a b : Grid) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

-- Condition stating no two adjacent squares can share the same color
def valid_coloring (f : Grid → Colors) : Prop :=
  ∀ a b : Grid, adjacent a b → f a ≠ f b

-- The main theorem stating the number of valid colorings
theorem num_valid_colorings : ∃ (n : ℕ), n = 2 ∧ ∀ (f : Grid → Colors), valid_coloring f → n = 2 :=
by sorry

end ColoringGrid

end num_valid_colorings_l633_63302


namespace rain_total_duration_l633_63397

theorem rain_total_duration : 
  let first_day_hours := 17 - 7
  let second_day_hours := first_day_hours + 2
  let third_day_hours := 2 * second_day_hours
  first_day_hours + second_day_hours + third_day_hours = 46 :=
by
  sorry

end rain_total_duration_l633_63397


namespace a_greater_than_b_for_n_ge_2_l633_63386

theorem a_greater_than_b_for_n_ge_2 
  (n : ℕ) 
  (hn : n ≥ 2) 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a^n = a + 1) 
  (h2 : b^(2 * n) = b + 3 * a) : 
  a > b := 
  sorry

end a_greater_than_b_for_n_ge_2_l633_63386


namespace value_range_for_positive_roots_l633_63392

theorem value_range_for_positive_roots (a : ℝ) :
  (∀ x : ℝ, x > 0 → a * |x| + |x + a| = 0) ↔ (-1 < a ∧ a < 0) :=
by
  sorry

end value_range_for_positive_roots_l633_63392


namespace bacteria_initial_count_l633_63307

theorem bacteria_initial_count (n : ℕ) :
  (∀ t : ℕ, t % 30 = 0 → n * 2^(t / 30) = 262144 → t = 240) → n = 1024 :=
by sorry

end bacteria_initial_count_l633_63307


namespace value_of_a_for_perfect_square_trinomial_l633_63356

theorem value_of_a_for_perfect_square_trinomial (a : ℝ) (x y : ℝ) :
  (∃ b : ℝ, (x + b * y) ^ 2 = x^2 + a * x * y + y^2) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end value_of_a_for_perfect_square_trinomial_l633_63356
