import Mathlib

namespace NUMINAMATH_GPT_max_discount_l2419_241929

theorem max_discount (C : ℝ) (x : ℝ) (h1 : 1.8 * C = 360) (h2 : ∀ y, y ≥ 1.3 * C → 360 - x ≥ y) : x ≤ 100 :=
by
  have hC : C = 360 / 1.8 := by sorry
  have hMinPrice : 1.3 * C = 1.3 * (360 / 1.8) := by sorry
  have hDiscount : 360 - x ≥ 1.3 * (360 / 1.8) := by sorry
  sorry

end NUMINAMATH_GPT_max_discount_l2419_241929


namespace NUMINAMATH_GPT_determine_x_l2419_241919

-- Definitions for given conditions
variables (x y z a b c : ℝ)
variables (h₁ : xy / (x - y) = a) (h₂ : xz / (x - z) = b) (h₃ : yz / (y - z) = c)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- Main statement to prove
theorem determine_x :
  x = (2 * a * b * c) / (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_GPT_determine_x_l2419_241919


namespace NUMINAMATH_GPT_difference_is_four_l2419_241966

open Nat

-- Assume we have a 5x5x5 cube
def cube_side_length : ℕ := 5
def total_unit_cubes : ℕ := cube_side_length ^ 3

-- Define the two configurations
def painted_cubes_config1 : ℕ := 65  -- Two opposite faces and one additional face
def painted_cubes_config2 : ℕ := 61  -- Three adjacent faces

-- The difference in the number of unit cubes with at least one painted face
def painted_difference : ℕ := painted_cubes_config1 - painted_cubes_config2

theorem difference_is_four :
    painted_difference = 4 := by
  sorry

end NUMINAMATH_GPT_difference_is_four_l2419_241966


namespace NUMINAMATH_GPT_fraction_difference_l2419_241974

variable (a b : ℝ)

theorem fraction_difference (h : 1/a - 1/b = 1/(a + b)) : 
  1/a^2 - 1/b^2 = 1/(a * b) := 
  sorry

end NUMINAMATH_GPT_fraction_difference_l2419_241974


namespace NUMINAMATH_GPT_binomial_510_510_l2419_241949

theorem binomial_510_510 : Nat.choose 510 510 = 1 :=
by
  sorry

end NUMINAMATH_GPT_binomial_510_510_l2419_241949


namespace NUMINAMATH_GPT_max_value_min_expression_l2419_241910

def f (x y : ℝ) : ℝ :=
  x^3 + (y-4)*x^2 + (y^2-4*y+4)*x + (y^3-4*y^2+4*y)

theorem max_value_min_expression (a b c : ℝ) (h₁: a ≠ b) (h₂: b ≠ c) (h₃: c ≠ a)
  (hab : f a b = f b c) (hbc : f b c = f c a) :
  (max (min (a^4 - 4*a^3 + 4*a^2) (min (b^4 - 4*b^3 + 4*b^2) (c^4 - 4*c^3 + 4*c^2))) 1) = 1 :=
sorry

end NUMINAMATH_GPT_max_value_min_expression_l2419_241910


namespace NUMINAMATH_GPT_smallest_y_l2419_241932

theorem smallest_y (y : ℕ) :
  (y > 0 ∧ 800 ∣ (540 * y)) ↔ (y = 40) :=
by
  sorry

end NUMINAMATH_GPT_smallest_y_l2419_241932


namespace NUMINAMATH_GPT_find_m_sum_terms_l2419_241950

theorem find_m (a : ℕ → ℤ) (d : ℤ) (h1 : d ≠ 0) 
  (h2 : a 3 + a 6 + a 10 + a 13 = 32) (hm : a m = 8) : m = 8 :=
sorry

theorem sum_terms (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) (hS3 : S 3 = 9) (hS6 : S 6 = 36) 
  (a_def : ∀ n, S n = n * (a 1 + a n) / 2) : a 7 + a 8 + a 9 = 45 :=
sorry

end NUMINAMATH_GPT_find_m_sum_terms_l2419_241950


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2419_241911

theorem sufficient_but_not_necessary_condition (x y : ℝ) : 
  (x > 3 ∧ y > 3 → x + y > 6) ∧ ¬(x + y > 6 → x > 3 ∧ y > 3) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2419_241911


namespace NUMINAMATH_GPT_linear_regression_increase_l2419_241931

-- Define the linear regression function
def linear_regression (x : ℝ) : ℝ :=
  1.6 * x + 2

-- Prove that y increases by 1.6 when x increases by 1
theorem linear_regression_increase (x : ℝ) :
  linear_regression (x + 1) - linear_regression x = 1.6 :=
by sorry

end NUMINAMATH_GPT_linear_regression_increase_l2419_241931


namespace NUMINAMATH_GPT_valid_license_plates_count_l2419_241922

-- Define the number of choices for letters and digits
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the total number of valid license plates
def num_valid_license_plates : ℕ := num_letters^3 * num_digits^3

-- Theorem stating that the number of valid license plates is 17,576,000
theorem valid_license_plates_count :
  num_valid_license_plates = 17576000 :=
by
  sorry

end NUMINAMATH_GPT_valid_license_plates_count_l2419_241922


namespace NUMINAMATH_GPT_solve_problem_l2419_241925

def bracket (a b c : ℕ) : ℕ := (a + b) / c

theorem solve_problem :
  bracket (bracket 50 50 100) (bracket 3 6 9) (bracket 20 30 50) = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l2419_241925


namespace NUMINAMATH_GPT_find_expression_value_l2419_241964

theorem find_expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) : (7 * x + 8 * y) / (x - 2 * y) = 29 := by
  sorry

end NUMINAMATH_GPT_find_expression_value_l2419_241964


namespace NUMINAMATH_GPT_inequality_f_n_l2419_241980

theorem inequality_f_n {f : ℕ → ℕ} {k : ℕ} (strict_mono_f : ∀ {a b : ℕ}, a < b → f a < f b)
  (h_f : ∀ n : ℕ, f (f n) = k * n) : ∀ n : ℕ, 
  (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ ((k + 1) * n) / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_f_n_l2419_241980


namespace NUMINAMATH_GPT_smallest_b_factors_l2419_241969

theorem smallest_b_factors (p q b : ℤ) (hpq : p * q = 1764) (hb : b = p + q) (hposp : p > 0) (hposq : q > 0) :
  b = 84 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_factors_l2419_241969


namespace NUMINAMATH_GPT_point_below_parabola_l2419_241976

theorem point_below_parabola (a b c : ℝ) (h : 2 < a + b + c) : 
  2 < c + b + a :=
by
  sorry

end NUMINAMATH_GPT_point_below_parabola_l2419_241976


namespace NUMINAMATH_GPT_arith_seq_s14_gt_0_l2419_241961

variable {S : ℕ → ℝ} -- S_n is the sum of the first n terms of an arithmetic sequence
variable {a : ℕ → ℝ} -- a_n is the nth term of the arithmetic sequence
variable {d : ℝ} -- d is the common difference of the arithmetic sequence

-- Conditions
variable (a_7_lt_0 : a 7 < 0)
variable (a_5_plus_a_10_gt_0 : a 5 + a 10 > 0)

-- Assertion
theorem arith_seq_s14_gt_0 (a_7_lt_0 : a 7 < 0) (a_5_plus_a_10_gt_0 : a 5 + a 10 > 0) : S 14 > 0 := by
  sorry

end NUMINAMATH_GPT_arith_seq_s14_gt_0_l2419_241961


namespace NUMINAMATH_GPT_smallest_positive_integer_ending_in_9_divisible_by_13_l2419_241943

theorem smallest_positive_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, (n % 10 = 9) ∧ (n % 13 = 0) ∧ (∀ m : ℕ, (m % 10 = 9) ∧ (m % 13 = 0) → m ≥ n) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_ending_in_9_divisible_by_13_l2419_241943


namespace NUMINAMATH_GPT_gcd_of_72_and_90_l2419_241914

theorem gcd_of_72_and_90 :
  Int.gcd 72 90 = 18 := 
sorry

end NUMINAMATH_GPT_gcd_of_72_and_90_l2419_241914


namespace NUMINAMATH_GPT_cost_of_baseball_cards_l2419_241907

variables (cost_football cost_pokemon total_spent cost_baseball : ℝ)
variable (h1 : cost_football = 2 * 2.73)
variable (h2 : cost_pokemon = 4.01)
variable (h3 : total_spent = 18.42)
variable (total_cost_football_pokemon : ℝ)
variable (h4 : total_cost_football_pokemon = cost_football + cost_pokemon)

theorem cost_of_baseball_cards
  (h : cost_baseball = total_spent - total_cost_football_pokemon) : 
  cost_baseball = 8.95 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_baseball_cards_l2419_241907


namespace NUMINAMATH_GPT_virginia_avg_rainfall_l2419_241933

theorem virginia_avg_rainfall:
  let march := 3.79
  let april := 4.5
  let may := 3.95
  let june := 3.09
  let july := 4.67
  let total_rainfall := march + april + may + june + july
  let avg_rainfall := total_rainfall / 5
  avg_rainfall = 4 := by sorry

end NUMINAMATH_GPT_virginia_avg_rainfall_l2419_241933


namespace NUMINAMATH_GPT_position_of_z_l2419_241951

theorem position_of_z (total_distance : ℕ) (total_steps : ℕ) (steps_taken : ℕ) (distance_covered : ℕ) (h1 : total_distance = 30) (h2 : total_steps = 6) (h3 : steps_taken = 4) (h4 : distance_covered = total_distance / total_steps) : 
  steps_taken * distance_covered = 20 :=
by
  sorry

end NUMINAMATH_GPT_position_of_z_l2419_241951


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_transformed_roots_l2419_241960

theorem sum_of_reciprocals_of_transformed_roots :
  ∀ (a b c : ℂ), (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  (1/(a+1) + 1/(b+1) + 1/(c+1) = -2) :=
by
  intros a b c ha hb hc habc
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_transformed_roots_l2419_241960


namespace NUMINAMATH_GPT_bottles_in_one_bag_l2419_241920

theorem bottles_in_one_bag (total_bottles : ℕ) (cartons bags_per_carton : ℕ)
  (h1 : total_bottles = 180)
  (h2 : cartons = 3)
  (h3 : bags_per_carton = 4) :
  total_bottles / cartons / bags_per_carton = 15 :=
by sorry

end NUMINAMATH_GPT_bottles_in_one_bag_l2419_241920


namespace NUMINAMATH_GPT_shares_difference_l2419_241962

theorem shares_difference (x : ℝ) (hp : ℝ) (hq : ℝ) (hr : ℝ)
  (hx : hp = 3 * x) (hqx : hq = 7 * x) (hrx : hr = 12 * x) 
  (hqr_diff : hr - hq = 3500) : (hq - hp = 2800) :=
by
  -- The proof would be done here, but the problem statement requires only the theorem statement
  sorry

end NUMINAMATH_GPT_shares_difference_l2419_241962


namespace NUMINAMATH_GPT_total_workers_calculation_l2419_241908

theorem total_workers_calculation :
  ∀ (N : ℕ), 
  (∀ (total_avg_salary : ℕ) (techs_salary : ℕ) (nontech_avg_salary : ℕ),
    total_avg_salary = 8000 → 
    techs_salary = 7 * 20000 → 
    nontech_avg_salary = 6000 →
    8000 * (7 + N) = 7 * 20000 + N * 6000 →
    (7 + N) = 49) :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_workers_calculation_l2419_241908


namespace NUMINAMATH_GPT_polynomial_root_p_value_l2419_241948

theorem polynomial_root_p_value (p : ℝ) : (3 : ℝ) ^ 3 + p * (3 : ℝ) - 18 = 0 → p = -3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_root_p_value_l2419_241948


namespace NUMINAMATH_GPT_distance_center_is_12_l2419_241904

-- Define the side length of the square and the radius of the circle
def side_length_square : ℝ := 5
def radius_circle : ℝ := 1

-- The center path forms a smaller square inside the original square
-- with side length 3 units
def side_length_smaller_square : ℝ := side_length_square - 2 * radius_circle

-- The perimeter of the smaller square, which is the path length that
-- the center of the circle travels
def distance_center_travel : ℝ := 4 * side_length_smaller_square

-- Prove that the distance traveled by the center of the circle is 12 units
theorem distance_center_is_12 : distance_center_travel = 12 := by
  -- the proof is skipped
  sorry

end NUMINAMATH_GPT_distance_center_is_12_l2419_241904


namespace NUMINAMATH_GPT_intersection_P_Q_l2419_241994

def P : Set ℤ := {-4, -2, 0, 2, 4}
def Q : Set ℤ := {x : ℤ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l2419_241994


namespace NUMINAMATH_GPT_height_of_wall_l2419_241954

theorem height_of_wall (length_brick width_brick height_brick : ℝ)
                        (length_wall width_wall number_of_bricks : ℝ)
                        (volume_of_bricks : ℝ) :
  (length_brick, width_brick, height_brick) = (125, 11.25, 6) →
  (length_wall, width_wall) = (800, 22.5) →
  number_of_bricks = 1280 →
  volume_of_bricks = length_brick * width_brick * height_brick * number_of_bricks →
  volume_of_bricks = length_wall * width_wall * 600 := 
by
  intros h1 h2 h3 h4
  -- proof skipped
  sorry

end NUMINAMATH_GPT_height_of_wall_l2419_241954


namespace NUMINAMATH_GPT_largest_multiple_of_7_gt_neg_150_l2419_241936

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_7_gt_neg_150_l2419_241936


namespace NUMINAMATH_GPT_total_payment_correct_l2419_241965

-- Define the prices of different apples.
def price_small_apple : ℝ := 1.5
def price_medium_apple : ℝ := 2.0
def price_big_apple : ℝ := 3.0

-- Define the quantities of apples bought by Donny.
def quantity_small_apples : ℕ := 6
def quantity_medium_apples : ℕ := 6
def quantity_big_apples : ℕ := 8

-- Define the conditions.
def discount_medium_apples_threshold : ℕ := 5
def discount_medium_apples_rate : ℝ := 0.20
def tax_rate : ℝ := 0.10
def big_apple_special_offer_count : ℕ := 3
def big_apple_special_offer_discount_rate : ℝ := 0.50

-- Step function to calculate discount and total cost.
noncomputable def total_cost : ℝ :=
  let cost_small := quantity_small_apples * price_small_apple
  let cost_medium := quantity_medium_apples * price_medium_apple
  let discount_medium := if quantity_medium_apples > discount_medium_apples_threshold 
                         then cost_medium * discount_medium_apples_rate else 0
  let cost_medium_after_discount := cost_medium - discount_medium
  let cost_big := quantity_big_apples * price_big_apple
  let discount_big := (quantity_big_apples / big_apple_special_offer_count) * 
                       (price_big_apple * big_apple_special_offer_discount_rate)
  let cost_big_after_discount := cost_big - discount_big
  let total_cost_before_tax := cost_small + cost_medium_after_discount + cost_big_after_discount
  let tax := total_cost_before_tax * tax_rate
  total_cost_before_tax + tax

-- Define the expected total payment.
def expected_total_payment : ℝ := 43.56

-- The theorem statement: Prove that total_cost equals the expected total payment.
theorem total_payment_correct : total_cost = expected_total_payment := sorry

end NUMINAMATH_GPT_total_payment_correct_l2419_241965


namespace NUMINAMATH_GPT_find_middle_number_l2419_241942

theorem find_middle_number (a b c d x e f g : ℝ) 
  (h1 : (a + b + c + d + x + e + f + g) / 8 = 7)
  (h2 : (a + b + c + d + x) / 5 = 6)
  (h3 : (x + e + f + g + d) / 5 = 9) :
  x = 9.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_middle_number_l2419_241942


namespace NUMINAMATH_GPT_vector_addition_correct_l2419_241989

variables (a b : ℝ × ℝ)
def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-1, 2)

theorem vector_addition_correct : vector_a + vector_b = (1, 5) :=
by
  -- Assume a and b are vectors in 2D space
  have a := vector_a
  have b := vector_b
  -- By definition of vector addition
  sorry

end NUMINAMATH_GPT_vector_addition_correct_l2419_241989


namespace NUMINAMATH_GPT_no_perfect_squares_in_seq_l2419_241900

def seq (x : ℕ → ℤ) : Prop :=
  x 0 = 1 ∧ x 1 = 3 ∧ ∀ n : ℕ, 0 < n → x (n + 1) = 6 * x n - x (n - 1)

theorem no_perfect_squares_in_seq (x : ℕ → ℤ) (n : ℕ) (h_seq : seq x) :
  ¬ ∃ k : ℤ, k * k = x (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_no_perfect_squares_in_seq_l2419_241900


namespace NUMINAMATH_GPT_population_net_increase_l2419_241941

-- Definitions of conditions
def birth_rate := 7 / 2 -- 7 people every 2 seconds
def death_rate := 1 / 2 -- 1 person every 2 seconds
def seconds_in_a_day := 86400 -- Number of seconds in one day

-- Definition of the total births in one day
def total_births_per_day := birth_rate * seconds_in_a_day

-- Definition of the total deaths in one day
def total_deaths_per_day := death_rate * seconds_in_a_day

-- Proposition to prove the net population increase in one day
theorem population_net_increase : total_births_per_day - total_deaths_per_day = 259200 := by
  sorry

end NUMINAMATH_GPT_population_net_increase_l2419_241941


namespace NUMINAMATH_GPT_real_inequality_l2419_241903

theorem real_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + a * c + b * c := by
  sorry

end NUMINAMATH_GPT_real_inequality_l2419_241903


namespace NUMINAMATH_GPT_percentage_increase_variable_cost_l2419_241923

noncomputable def variable_cost_first_year : ℝ := 26000
noncomputable def fixed_cost : ℝ := 40000
noncomputable def total_breeding_cost_third_year : ℝ := 71460

theorem percentage_increase_variable_cost (x : ℝ) 
  (h : 40000 + 26000 * (1 + x) ^ 2 = 71460) : 
  x = 0.1 := 
by sorry

end NUMINAMATH_GPT_percentage_increase_variable_cost_l2419_241923


namespace NUMINAMATH_GPT_profit_percentage_l2419_241963

theorem profit_percentage (SP CP : ℝ) (h₁ : SP = 300) (h₂ : CP = 250) : ((SP - CP) / CP) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_l2419_241963


namespace NUMINAMATH_GPT_smallest_even_sum_equals_200_l2419_241987

theorem smallest_even_sum_equals_200 :
  ∃ (x : ℤ), (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 200) ∧ (x = 36) :=
by
  sorry

end NUMINAMATH_GPT_smallest_even_sum_equals_200_l2419_241987


namespace NUMINAMATH_GPT_smallest_n_for_2007_l2419_241939

/-- The smallest number of positive integers \( n \) such that their product is 2007 and their sum is 2007.
Given that \( n > 1 \), we need to show 1337 is the smallest such \( n \).
-/
theorem smallest_n_for_2007 (n : ℕ) (H : n > 1) :
  (∃ s : Finset ℕ, (s.sum id = 2007) ∧ (s.prod id = 2007) ∧ (s.card = n)) → (n = 1337) :=
sorry

end NUMINAMATH_GPT_smallest_n_for_2007_l2419_241939


namespace NUMINAMATH_GPT_vector_dot_product_l2419_241988

open Real

variables (a b : ℝ × ℝ)

def condition1 : Prop := (a.1 + b.1 = 1 ∧ a.2 + b.2 = -3)
def condition2 : Prop := (a.1 - b.1 = 3 ∧ a.2 - b.2 = 7)
def dot_product : ℝ := a.1 * b.1 + a.2 * b.2

theorem vector_dot_product :
  condition1 a b ∧ condition2 a b → dot_product a b = -12 := by
  sorry

end NUMINAMATH_GPT_vector_dot_product_l2419_241988


namespace NUMINAMATH_GPT_bailey_points_final_game_l2419_241952

def chandra_points (a: ℕ) := 2 * a
def akiko_points (m: ℕ) := m + 4
def michiko_points (b: ℕ) := b / 2
def team_total_points (b c a m: ℕ) := b + c + a + m

theorem bailey_points_final_game (B: ℕ) 
  (M : ℕ := michiko_points B)
  (A : ℕ := akiko_points M)
  (C : ℕ := chandra_points A)
  (H : team_total_points B C A M = 54): B = 14 :=
by 
  sorry

end NUMINAMATH_GPT_bailey_points_final_game_l2419_241952


namespace NUMINAMATH_GPT_regression_prediction_l2419_241992

-- Define the linear regression model as a function
def linear_regression (x : ℝ) : ℝ :=
  7.19 * x + 73.93

-- State that using this model, the predicted height at age 10 is approximately 145.83
theorem regression_prediction :
  abs (linear_regression 10 - 145.83) < 0.01 :=
by 
  sorry

end NUMINAMATH_GPT_regression_prediction_l2419_241992


namespace NUMINAMATH_GPT_maximum_area_rectangular_backyard_l2419_241901

theorem maximum_area_rectangular_backyard (x : ℕ) (y : ℕ) (h_perimeter : 2 * (x + y) = 100) : 
  x * y ≤ 625 :=
by
  sorry

end NUMINAMATH_GPT_maximum_area_rectangular_backyard_l2419_241901


namespace NUMINAMATH_GPT_find_greatest_divisor_l2419_241947

def greatest_divisor_leaving_remainders (n₁ n₁_r n₂ n₂_r d : ℕ) : Prop :=
  (n₁ % d = n₁_r) ∧ (n₂ % d = n₂_r) 

theorem find_greatest_divisor :
  greatest_divisor_leaving_remainders 1657 10 2037 7 1 :=
by
  sorry

end NUMINAMATH_GPT_find_greatest_divisor_l2419_241947


namespace NUMINAMATH_GPT_problem1_l2419_241946

variable (α : ℝ)

theorem problem1 (h : Real.tan α = -3/4) : 
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = -3/4 := 
sorry

end NUMINAMATH_GPT_problem1_l2419_241946


namespace NUMINAMATH_GPT_simplify_expression_l2419_241985

-- Define the fractions involved
def frac1 : ℚ := 1 / 2
def frac2 : ℚ := 1 / 3
def frac3 : ℚ := 1 / 5
def frac4 : ℚ := 1 / 7

-- Define the expression to be simplified
def expr : ℚ := (frac1 - frac2 + frac3) / (frac2 - frac1 + frac4)

-- The goal is to show that the expression simplifies to -77 / 5
theorem simplify_expression : expr = -77 / 5 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2419_241985


namespace NUMINAMATH_GPT_boxes_needed_l2419_241977

def initial_games : ℕ := 76
def games_sold : ℕ := 46
def games_per_box : ℕ := 5

theorem boxes_needed : (initial_games - games_sold) / games_per_box = 6 := by
  sorry

end NUMINAMATH_GPT_boxes_needed_l2419_241977


namespace NUMINAMATH_GPT_part_a_l2419_241935

theorem part_a (m n : ℕ) (hm : m > 1) : n ∣ Nat.totient (m^n - 1) :=
sorry

end NUMINAMATH_GPT_part_a_l2419_241935


namespace NUMINAMATH_GPT_polynomial_value_l2419_241983

theorem polynomial_value (a b : ℝ) (h₁ : a * b = 7) (h₂ : a + b = 2) : a^2 * b + a * b^2 - 20 = -6 :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_value_l2419_241983


namespace NUMINAMATH_GPT_share_ratio_l2419_241934

theorem share_ratio (A B C : ℕ) (hA : A = (2 * B) / 3) (hA_val : A = 372) (hB_val : B = 93) (hC_val : C = 62) : B / C = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_share_ratio_l2419_241934


namespace NUMINAMATH_GPT_jed_speeding_l2419_241921

-- Define the constants used in the conditions
def F := 16
def T := 256
def S := 50

theorem jed_speeding : (T / F) + S = 66 := 
by sorry

end NUMINAMATH_GPT_jed_speeding_l2419_241921


namespace NUMINAMATH_GPT_difference_place_values_l2419_241975

def place_value (digit : Char) (position : String) : Real :=
  match digit, position with
  | '1', "hundreds" => 100
  | '1', "tenths" => 0.1
  | _, _ => 0 -- for any other cases (not required in this problem)

theorem difference_place_values :
  (place_value '1' "hundreds" - place_value '1' "tenths" = 99.9) :=
by
  sorry

end NUMINAMATH_GPT_difference_place_values_l2419_241975


namespace NUMINAMATH_GPT_second_integer_value_l2419_241937

theorem second_integer_value (n : ℚ) (h : (n - 1) + (n + 1) + (n + 2) = 175) : n = 57 + 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_second_integer_value_l2419_241937


namespace NUMINAMATH_GPT_negative_x_y_l2419_241928

theorem negative_x_y (x y : ℝ) (h1 : x - y > x) (h2 : x + y < y) : x < 0 ∧ y < 0 :=
by
  sorry

end NUMINAMATH_GPT_negative_x_y_l2419_241928


namespace NUMINAMATH_GPT_probability_of_six_being_largest_l2419_241999

noncomputable def probability_six_is_largest : ℚ := sorry

theorem probability_of_six_being_largest (cards : Finset ℕ) (selected_cards : Finset ℕ) :
  cards = {1, 2, 3, 4, 5, 6, 7} →
  selected_cards ⊆ cards →
  selected_cards.card = 4 →
  (probability_six_is_largest = 2 / 7) := sorry

end NUMINAMATH_GPT_probability_of_six_being_largest_l2419_241999


namespace NUMINAMATH_GPT_olivia_not_sold_bars_l2419_241955

theorem olivia_not_sold_bars (cost_per_bar : ℕ) (total_bars : ℕ) (total_money_made : ℕ) :
  cost_per_bar = 3 →
  total_bars = 7 →
  total_money_made = 9 →
  total_bars - (total_money_made / cost_per_bar) = 4 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_olivia_not_sold_bars_l2419_241955


namespace NUMINAMATH_GPT_proof_y_times_1_minus_g_eq_1_l2419_241990
noncomputable def y : ℝ := (3 + Real.sqrt 8) ^ 100
noncomputable def m : ℤ := Int.floor y
noncomputable def g : ℝ := y - m

theorem proof_y_times_1_minus_g_eq_1 :
  y * (1 - g) = 1 := 
sorry

end NUMINAMATH_GPT_proof_y_times_1_minus_g_eq_1_l2419_241990


namespace NUMINAMATH_GPT_chosen_number_is_5_l2419_241906

theorem chosen_number_is_5 (x : ℕ) (h_pos : x > 0)
  (h_eq : ((10 * x + 5 - x^2) / x) - x = 1) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_chosen_number_is_5_l2419_241906


namespace NUMINAMATH_GPT_a3_eq_5_l2419_241986

variable {a_n : ℕ → Real} (S : ℕ → Real)
variable (a1 d : Real)

-- Define arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → Real) (a1 d : Real) : Prop :=
  ∀ n : ℕ, n > 0 → a_n n = a1 + (n - 1) * d

-- Define sum of first n terms
def sum_of_arithmetic (S : ℕ → Real) (a_n : ℕ → Real) : Prop :=
  ∀ n : ℕ, S n = n / 2 * (a_n 1 + a_n n)

-- Given conditions: S_5 = 25
def S_5_eq_25 (S : ℕ → Real) : Prop :=
  S 5 = 25

-- Goal: prove a_3 = 5
theorem a3_eq_5 (h_arith : is_arithmetic_sequence a_n a1 d)
                (h_sum : sum_of_arithmetic S a_n)
                (h_S5 : S_5_eq_25 S) : a_n 3 = 5 :=
  sorry

end NUMINAMATH_GPT_a3_eq_5_l2419_241986


namespace NUMINAMATH_GPT_katie_earnings_l2419_241958

-- Define the constants for the problem
def bead_necklaces : Nat := 4
def gemstone_necklaces : Nat := 3
def cost_per_necklace : Nat := 3

-- Define the total earnings calculation
def total_necklaces : Nat := bead_necklaces + gemstone_necklaces
def total_earnings : Nat := total_necklaces * cost_per_necklace

-- Statement of the proof problem
theorem katie_earnings : total_earnings = 21 := by
  sorry

end NUMINAMATH_GPT_katie_earnings_l2419_241958


namespace NUMINAMATH_GPT_focus_of_parabola_l2419_241978

theorem focus_of_parabola (x y : ℝ) : 
  (∃ x y : ℝ, x^2 = -2 * y) → (0, -1/2) = (0, -1/2) :=
sorry

end NUMINAMATH_GPT_focus_of_parabola_l2419_241978


namespace NUMINAMATH_GPT_parabola_vertex_coordinates_l2419_241959

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = -(x - 1) ^ 2 + 3 → (1, 3) = (1, 3) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_parabola_vertex_coordinates_l2419_241959


namespace NUMINAMATH_GPT_boa_constrictors_in_park_l2419_241913

theorem boa_constrictors_in_park :
  ∃ (B : ℕ), (∃ (p : ℕ), p = 3 * B) ∧ (B + 3 * B + 40 = 200) ∧ B = 40 :=
by
  sorry

end NUMINAMATH_GPT_boa_constrictors_in_park_l2419_241913


namespace NUMINAMATH_GPT_range_of_independent_variable_l2419_241981

theorem range_of_independent_variable (x : ℝ) :
  (x + 2 >= 0) → (x - 1 ≠ 0) → (x ≥ -2 ∧ x ≠ 1) :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_range_of_independent_variable_l2419_241981


namespace NUMINAMATH_GPT_intersection_complement_eq_l2419_241927

open Set

-- Definitions from the problem conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | x ≥ 0}
def C_U_N : Set ℝ := {x | x < 0}

-- Statement of the proof problem
theorem intersection_complement_eq : M ∩ C_U_N = {x | -1 ≤ x ∧ x < 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l2419_241927


namespace NUMINAMATH_GPT_maximize_profit_marginal_profit_monotonic_decreasing_l2419_241970

-- Definition of revenue function R
def R (x : ℕ) : ℤ := 3700 * x + 45 * x^2 - 10 * x^3

-- Definition of cost function C
def C (x : ℕ) : ℤ := 460 * x + 500

-- Definition of profit function p
def p (x : ℕ) : ℤ := R x - C x

-- Lemma for the solution
theorem maximize_profit (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 20) : 
  p x = -10 * x^3 + 45 * x^2 + 3240 * x - 500 ∧ 
  (∀ y, 1 ≤ y ∧ y ≤ 20 → p y ≤ p 12) :=
by
  sorry

-- Definition of marginal profit function Mp
def Mp (x : ℕ) : ℤ := p (x + 1) - p x

-- Lemma showing Mp is monotonically decreasing
theorem marginal_profit_monotonic_decreasing (x : ℕ) (h2 : 1 ≤ x ∧ x ≤ 19) : 
  Mp x = -30 * x^2 + 60 * x + 3275 ∧ 
  ∀ y, 1 ≤ y ∧ y ≤ 19 → (Mp y ≥ Mp (y + 1)) :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_marginal_profit_monotonic_decreasing_l2419_241970


namespace NUMINAMATH_GPT_even_func_monotonic_on_negative_interval_l2419_241902

variable {α : Type*} [LinearOrderedField α]
variable {f : α → α}

theorem even_func_monotonic_on_negative_interval 
  (h_even : ∀ x : α, f (-x) = f x)
  (h_mon_incr : ∀ x y : α, x < y → (x < 0 ∧ y ≤ 0) → f x < f y) :
  f 2 < f (-3 / 2) :=
sorry

end NUMINAMATH_GPT_even_func_monotonic_on_negative_interval_l2419_241902


namespace NUMINAMATH_GPT_comparison_of_abc_l2419_241972

noncomputable def a : ℝ := 24 / 7
noncomputable def b : ℝ := Real.log 7
noncomputable def c : ℝ := Real.log (7 / Real.exp 1) / Real.log 3 + 1

theorem comparison_of_abc :
  (a = 24 / 7) →
  (b * Real.exp b = 7 * Real.log 7) →
  (3 ^ (c - 1) = 7 / Real.exp 1) →
  a > b ∧ b > c :=
by
  intros ha hb hc
  sorry

end NUMINAMATH_GPT_comparison_of_abc_l2419_241972


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l2419_241905

theorem problem_1 : 42.67 - (12.67 - 2.87) = 32.87 :=
by sorry

theorem problem_2 : (4.8 - 4.8 * (3.2 - 2.7)) / 0.24 = 10 :=
by sorry

theorem problem_3 : 4.31 * 0.57 + 0.43 * 4.31 - 4.31 = 0 :=
by sorry

theorem problem_4 : 9.99 * 222 + 3.33 * 334 = 3330 :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l2419_241905


namespace NUMINAMATH_GPT_disputed_piece_weight_l2419_241938

theorem disputed_piece_weight (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := by
  sorry

end NUMINAMATH_GPT_disputed_piece_weight_l2419_241938


namespace NUMINAMATH_GPT_find_real_roots_l2419_241998

theorem find_real_roots : 
  {x : ℝ | x^9 + (9 / 8) * x^6 + (27 / 64) * x^3 - x + (219 / 512) = 0} =
  {1 / 2, (-1 + Real.sqrt 13) / 4, (-1 - Real.sqrt 13) / 4} :=
by
  sorry

end NUMINAMATH_GPT_find_real_roots_l2419_241998


namespace NUMINAMATH_GPT_window_area_l2419_241917

def meter_to_feet : ℝ := 3.28084
def length_in_meters : ℝ := 2
def width_in_feet : ℝ := 15

def length_in_feet := length_in_meters * meter_to_feet
def area_in_square_feet := length_in_feet * width_in_feet

theorem window_area : area_in_square_feet = 98.4252 := 
by
  sorry

end NUMINAMATH_GPT_window_area_l2419_241917


namespace NUMINAMATH_GPT_min_value_3x_plus_4y_l2419_241993

theorem min_value_3x_plus_4y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x + 3*y = 5*x*y) :
  ∃ (c : ℝ), (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y → 3 * x + 4 * y ≥ c) ∧ c = 5 :=
sorry

end NUMINAMATH_GPT_min_value_3x_plus_4y_l2419_241993


namespace NUMINAMATH_GPT_custom_dollar_five_neg3_l2419_241984

-- Define the custom operation
def custom_dollar (a b : Int) : Int :=
  a * (b - 1) + a * b

-- State the theorem
theorem custom_dollar_five_neg3 : custom_dollar 5 (-3) = -35 := by
  sorry

end NUMINAMATH_GPT_custom_dollar_five_neg3_l2419_241984


namespace NUMINAMATH_GPT_max_soap_boxes_in_carton_l2419_241916

theorem max_soap_boxes_in_carton
  (L_carton W_carton H_carton : ℕ)
  (L_soap_box W_soap_box H_soap_box : ℕ)
  (vol_carton := L_carton * W_carton * H_carton)
  (vol_soap_box := L_soap_box * W_soap_box * H_soap_box)
  (max_soap_boxes := vol_carton / vol_soap_box) :
  L_carton = 25 → W_carton = 42 → H_carton = 60 →
  L_soap_box = 7 → W_soap_box = 6 → H_soap_box = 5 →
  max_soap_boxes = 300 :=
by
  intros hL hW hH hLs hWs hHs
  sorry

end NUMINAMATH_GPT_max_soap_boxes_in_carton_l2419_241916


namespace NUMINAMATH_GPT_cone_height_l2419_241982

theorem cone_height (R : ℝ) (r h l : ℝ)
  (volume_sphere : ∀ R,  V_sphere = (4 / 3) * π * R^3)
  (volume_cone : ∀ r h,  V_cone = (1 / 3) * π * r^2 * h)
  (lateral_surface_area : ∀ r l, A_lateral = π * r * l)
  (area_base : ∀ r, A_base = π * r^2)
  (vol_eq : (1/3) * π * r^2 * h = (4/3) * π * R^3)
  (lat_eq : π * r * l = 3 * π * r^2) 
  (pyth_rel : l^2 = r^2 + h^2) :
  h = 4 * R * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_cone_height_l2419_241982


namespace NUMINAMATH_GPT_lock_and_key_requirements_l2419_241930

/-- There are 7 scientists each with a key to an electronic lock which requires at least 4 scientists to open.
    - Prove that the minimum number of unique features (locks) the electronic lock must have is 35.
    - Prove that each scientist's key should have at least 20 features.
--/
theorem lock_and_key_requirements :
  ∃ (locks : ℕ) (features_per_key : ℕ), 
    locks = 35 ∧ features_per_key = 20 ∧
    (∀ (n_present : ℕ), n_present ≥ 4 → 7 - n_present ≤ 3) ∧
    (∀ (n_absent : ℕ), n_absent ≤ 3 → 7 - n_absent ≥ 4)
:= sorry

end NUMINAMATH_GPT_lock_and_key_requirements_l2419_241930


namespace NUMINAMATH_GPT_time_to_pass_platform_l2419_241953

-- Definitions for the given conditions
def train_length := 1200 -- length of the train in meters
def tree_crossing_time := 120 -- time taken to cross a tree in seconds
def platform_length := 1200 -- length of the platform in meters

-- Calculation of speed of the train and distance to be covered
def train_speed := train_length / tree_crossing_time -- speed in meters per second
def total_distance_to_cover := train_length + platform_length -- total distance in meters

-- Proof statement that given the above conditions, the time to pass the platform is 240 seconds
theorem time_to_pass_platform : 
  total_distance_to_cover / train_speed = 240 :=
  by sorry

end NUMINAMATH_GPT_time_to_pass_platform_l2419_241953


namespace NUMINAMATH_GPT_jasmine_percentage_new_solution_l2419_241967

-- Define the initial conditions
def initial_volume : ℝ := 80
def initial_jasmine_percent : ℝ := 0.10
def added_jasmine : ℝ := 5
def added_water : ℝ := 15

-- Define the correct answer
theorem jasmine_percentage_new_solution :
  let initial_jasmine := initial_jasmine_percent * initial_volume
  let new_jasmine := initial_jasmine + added_jasmine
  let total_new_volume := initial_volume + added_jasmine + added_water
  (new_jasmine / total_new_volume) * 100 = 13 := 
by 
  sorry

end NUMINAMATH_GPT_jasmine_percentage_new_solution_l2419_241967


namespace NUMINAMATH_GPT_solve_for_z_l2419_241957

variable (z : ℂ) (i : ℂ)

theorem solve_for_z
  (h1 : 3 - 2*i*z = 7 + 4*i*z)
  (h2 : i^2 = -1) :
  z = 2*i / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_z_l2419_241957


namespace NUMINAMATH_GPT_total_charge_correct_l2419_241968

def boxwoodTrimCost (numBoxwoods : Nat) (trimCost : Nat) : Nat :=
  numBoxwoods * trimCost

def boxwoodShapeCost (numBoxwoods : Nat) (shapeCost : Nat) : Nat :=
  numBoxwoods * shapeCost

theorem total_charge_correct :
  let numBoxwoodsTrimmed := 30
  let trimCost := 5
  let numBoxwoodsShaped := 4
  let shapeCost := 15
  let totalTrimCost := boxwoodTrimCost numBoxwoodsTrimmed trimCost
  let totalShapeCost := boxwoodShapeCost numBoxwoodsShaped shapeCost
  let totalCharge := totalTrimCost + totalShapeCost
  totalCharge = 210 :=
by sorry

end NUMINAMATH_GPT_total_charge_correct_l2419_241968


namespace NUMINAMATH_GPT_student_ticket_cost_l2419_241940

def general_admission_ticket_cost : ℕ := 6
def total_tickets_sold : ℕ := 525
def total_revenue : ℕ := 2876
def general_admission_tickets_sold : ℕ := 388

def number_of_student_tickets_sold : ℕ := total_tickets_sold - general_admission_tickets_sold
def revenue_from_general_admission : ℕ := general_admission_tickets_sold * general_admission_ticket_cost

theorem student_ticket_cost : ∃ S : ℕ, number_of_student_tickets_sold * S + revenue_from_general_admission = total_revenue ∧ S = 4 :=
by
  sorry

end NUMINAMATH_GPT_student_ticket_cost_l2419_241940


namespace NUMINAMATH_GPT_number_is_0_point_5_l2419_241997

theorem number_is_0_point_5 (x : ℝ) (h : x = 1/6 + 0.33333333333333337) : x = 0.5 := 
by
  -- The actual proof would go here.
  sorry

end NUMINAMATH_GPT_number_is_0_point_5_l2419_241997


namespace NUMINAMATH_GPT_no_n_satisfies_mod_5_l2419_241944

theorem no_n_satisfies_mod_5 (n : ℤ) : (n^3 + 2*n - 1) % 5 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_n_satisfies_mod_5_l2419_241944


namespace NUMINAMATH_GPT_luke_games_l2419_241956

variables (F G : ℕ)

theorem luke_games (G_eq_2 : G = 2) (total_games : F + G - 2 = 2) : F = 2 :=
by
  rw [G_eq_2] at total_games
  simp at total_games
  exact total_games

-- sorry

end NUMINAMATH_GPT_luke_games_l2419_241956


namespace NUMINAMATH_GPT_find_D_coordinates_l2419_241991

theorem find_D_coordinates:
  ∀ (A B C : (ℝ × ℝ)), 
  A = (-2, 5) ∧ C = (3, 7) ∧ B = (-3, 0) →
  ∃ D : (ℝ × ℝ), D = (2, 2) :=
by
  sorry

end NUMINAMATH_GPT_find_D_coordinates_l2419_241991


namespace NUMINAMATH_GPT_book_transaction_difference_l2419_241971

def number_of_books : ℕ := 15
def cost_per_book : ℕ := 11
def selling_price_per_book : ℕ := 25

theorem book_transaction_difference :
  number_of_books * selling_price_per_book - number_of_books * cost_per_book = 210 :=
by
  sorry

end NUMINAMATH_GPT_book_transaction_difference_l2419_241971


namespace NUMINAMATH_GPT_deal_or_no_deal_min_eliminations_l2419_241918

theorem deal_or_no_deal_min_eliminations (n_boxes : ℕ) (n_high_value : ℕ) 
    (initial_count : n_boxes = 26)
    (high_value_count : n_high_value = 9) :
  ∃ (min_eliminations : ℕ), min_eliminations = 8 ∧
    ((n_boxes - min_eliminations - 1) / 2) ≥ n_high_value :=
sorry

end NUMINAMATH_GPT_deal_or_no_deal_min_eliminations_l2419_241918


namespace NUMINAMATH_GPT_inequality_C_incorrect_l2419_241924

theorem inequality_C_incorrect (x : ℝ) (h : x ≠ 0) : ¬(e^x < 1 + x) → (e^1 ≥ 1 + 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_C_incorrect_l2419_241924


namespace NUMINAMATH_GPT_afternoon_snack_calories_l2419_241979

def ellen_daily_calories : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def dinner_remaining_calories : ℕ := 832

theorem afternoon_snack_calories :
  ellen_daily_calories - (breakfast_calories + lunch_calories + dinner_remaining_calories) = 130 :=
by sorry

end NUMINAMATH_GPT_afternoon_snack_calories_l2419_241979


namespace NUMINAMATH_GPT_find_length_of_room_l2419_241915

def length_of_room (L : ℕ) (width verandah_width verandah_area : ℕ) : Prop :=
  (L + 2 * verandah_width) * (width + 2 * verandah_width) - (L * width) = verandah_area

theorem find_length_of_room : length_of_room 15 12 2 124 :=
by
  -- We state the proof here, which is not requested in this exercise
  sorry

end NUMINAMATH_GPT_find_length_of_room_l2419_241915


namespace NUMINAMATH_GPT_count_ones_digits_of_numbers_divisible_by_4_and_3_l2419_241973

theorem count_ones_digits_of_numbers_divisible_by_4_and_3 :
  let eligible_numbers := { n : ℕ | n < 100 ∧ n % 4 = 0 ∧ n % 3 = 0 }
  ∃ (digits : Finset ℕ), 
    (∀ n ∈ eligible_numbers, n % 10 ∈ digits) ∧
    digits.card = 5 :=
by
  sorry

end NUMINAMATH_GPT_count_ones_digits_of_numbers_divisible_by_4_and_3_l2419_241973


namespace NUMINAMATH_GPT_S_n_formula_l2419_241926

def P (n : ℕ) : Type := sorry -- The type representing the nth polygon, not fully defined here.
def S : ℕ → ℝ := sorry -- The sequence S_n defined recursively.

-- Recursive definition of S_n given
axiom S_0 : S 0 = 1

-- This axiom represents the recursive step mentioned in the problem.
axiom S_rec : ∀ (k : ℕ), S (k + 1) = S k + (4^k / 3^(2*k + 2))

-- The main theorem we need to prove
theorem S_n_formula (n : ℕ) : 
  S n = (8 / 5) - (3 / 5) * (4 / 9)^n := sorry

end NUMINAMATH_GPT_S_n_formula_l2419_241926


namespace NUMINAMATH_GPT_tangent_line_at_five_l2419_241909

variable {f : ℝ → ℝ}

theorem tangent_line_at_five 
  (h_tangent : ∀ x, f x = -x + 8)
  (h_tangent_deriv : deriv f 5 = -1) :
  f 5 = 3 ∧ deriv f 5 = -1 :=
by sorry

end NUMINAMATH_GPT_tangent_line_at_five_l2419_241909


namespace NUMINAMATH_GPT_probability_of_exactly_nine_correct_placements_is_zero_l2419_241945

-- Define the number of letters and envelopes
def num_letters : ℕ := 10

-- Define the condition of letters being randomly inserted into envelopes
def random_insertion (n : ℕ) : Prop := true

-- Prove that the probability of exactly nine letters being correctly placed is zero
theorem probability_of_exactly_nine_correct_placements_is_zero
  (h : random_insertion num_letters) : 
  (∃ p : ℝ, p = 0) := 
sorry

end NUMINAMATH_GPT_probability_of_exactly_nine_correct_placements_is_zero_l2419_241945


namespace NUMINAMATH_GPT_zero_of_f_l2419_241995

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ↔ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_zero_of_f_l2419_241995


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2419_241912

theorem quadratic_inequality_solution :
  {x : ℝ | 2 * x ^ 2 - x - 3 > 0} = {x : ℝ | x > 3 / 2 ∨ x < -1} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2419_241912


namespace NUMINAMATH_GPT_max_area_of_garden_l2419_241996

theorem max_area_of_garden (L : ℝ) (hL : 0 ≤ L) :
  ∃ x y : ℝ, x + 2 * y = L ∧ x ≥ 0 ∧ y ≥ 0 ∧ x * y = L^2 / 8 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_garden_l2419_241996
