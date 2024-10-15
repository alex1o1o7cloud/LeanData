import Mathlib

namespace NUMINAMATH_GPT_perfect_square_solution_l73_7303

theorem perfect_square_solution (m n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (∃ k : ℕ, (5 ^ m + 2 ^ n * p) / (5 ^ m - 2 ^ n * p) = k ^ 2)
  ↔ (m = 1 ∧ n = 1 ∧ p = 2 ∨ m = 3 ∧ n = 2 ∧ p = 3 ∨ m = 2 ∧ n = 2 ∧ p = 5) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_solution_l73_7303


namespace NUMINAMATH_GPT_area_of_triangle_A2B2C2_l73_7397

noncomputable def area_DA1B1 : ℝ := 15 / 4
noncomputable def area_DA1C1 : ℝ := 10
noncomputable def area_DB1C1 : ℝ := 6
noncomputable def area_DA2B2 : ℝ := 40
noncomputable def area_DA2C2 : ℝ := 30
noncomputable def area_DB2C2 : ℝ := 50

theorem area_of_triangle_A2B2C2 : ∃ area : ℝ, 
  area = (50 * Real.sqrt 2) ∧ 
  (area_DA1B1 = 15/4 ∧ 
  area_DA1C1 = 10 ∧ 
  area_DB1C1 = 6 ∧ 
  area_DA2B2 = 40 ∧ 
  area_DA2C2 = 30 ∧ 
  area_DB2C2 = 50) := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_A2B2C2_l73_7397


namespace NUMINAMATH_GPT_simplify_expression_l73_7345

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l73_7345


namespace NUMINAMATH_GPT_sleepySquirrelNutsPerDay_l73_7386

def twoBusySquirrelsNutsPerDay : ℕ := 2 * 30
def totalDays : ℕ := 40
def totalNuts : ℕ := 3200

theorem sleepySquirrelNutsPerDay 
  (s  : ℕ) 
  (h₁ : 2 * 30 * totalDays + s * totalDays = totalNuts) 
  : s = 20 := 
  sorry

end NUMINAMATH_GPT_sleepySquirrelNutsPerDay_l73_7386


namespace NUMINAMATH_GPT_possible_values_of_cubes_l73_7351

noncomputable def matrix_N (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

def related_conditions (x y z : ℂ) (N : Matrix (Fin 3) (Fin 3) ℂ) : Prop :=
  N^2 = -1 ∧ x * y * z = -1

theorem possible_values_of_cubes (x y z : ℂ) (N : Matrix (Fin 3) (Fin 3) ℂ)
  (hc1 : matrix_N x y z = N) (hc2 : related_conditions x y z N) :
  ∃ w : ℂ, w = x^3 + y^3 + z^3 ∧ (w = -3 + Complex.I ∨ w = -3 - Complex.I) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_cubes_l73_7351


namespace NUMINAMATH_GPT_cups_per_larger_crust_l73_7396

theorem cups_per_larger_crust
  (initial_crusts : ℕ)
  (initial_flour : ℚ)
  (new_crusts : ℕ)
  (constant_flour : ℚ)
  (h1 : initial_crusts * (initial_flour / initial_crusts) = initial_flour )
  (h2 : new_crusts * (constant_flour / new_crusts) = constant_flour )
  (h3 : initial_flour = constant_flour)
  : (constant_flour / new_crusts) = (8 / 10) :=
by 
  sorry

end NUMINAMATH_GPT_cups_per_larger_crust_l73_7396


namespace NUMINAMATH_GPT_median_possible_values_l73_7379

theorem median_possible_values (S : Finset ℤ)
  (h : S.card = 10)
  (h_contains : {5, 7, 12, 15, 18, 21} ⊆ S) :
  ∃! n : ℕ, n = 5 :=
by
   sorry

end NUMINAMATH_GPT_median_possible_values_l73_7379


namespace NUMINAMATH_GPT_simplify_trig_expression_l73_7339

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem simplify_trig_expression :
  (tan (20 * Real.pi / 180) + tan (30 * Real.pi / 180) + tan (60 * Real.pi / 180) + tan (70 * Real.pi / 180)) / Real.sin (10 * Real.pi / 180) =
  1 / (2 * Real.sin (10 * Real.pi / 180) ^ 2 * Real.cos (20 * Real.pi / 180)) + 4 / (Real.sqrt 3 * Real.sin (10 * Real.pi / 180)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_trig_expression_l73_7339


namespace NUMINAMATH_GPT_sparrow_pecks_seeds_l73_7360

theorem sparrow_pecks_seeds (x : ℕ) (h1 : 9 * x < 1001) (h2 : 10 * x > 1100) : x = 111 :=
by
  sorry

end NUMINAMATH_GPT_sparrow_pecks_seeds_l73_7360


namespace NUMINAMATH_GPT_books_read_so_far_l73_7347

/-- There are 22 different books in the 'crazy silly school' series -/
def total_books : Nat := 22

/-- You still have to read 10 more books -/
def books_left_to_read : Nat := 10

theorem books_read_so_far :
  total_books - books_left_to_read = 12 :=
by
  sorry

end NUMINAMATH_GPT_books_read_so_far_l73_7347


namespace NUMINAMATH_GPT_cows_milk_production_l73_7373

variable (p q r s t : ℕ)

theorem cows_milk_production
  (h : p * r > 0)  -- Assuming p and r are positive to avoid division by zero
  (produce : p * r * q ≠ 0) -- Additional assumption to ensure non-zero q
  (h_cows : q = p * r * (q / (p * r))) 
  : s * t * q / (p * r) = s * t * (q / (p * r)) :=
by
  sorry

end NUMINAMATH_GPT_cows_milk_production_l73_7373


namespace NUMINAMATH_GPT_gcd_2023_2048_l73_7356

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_2023_2048_l73_7356


namespace NUMINAMATH_GPT_correct_operation_l73_7300

theorem correct_operation : -5 * 3 = -15 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l73_7300


namespace NUMINAMATH_GPT_tulip_price_correct_l73_7393

-- Initial conditions
def first_day_tulips : ℕ := 30
def first_day_roses : ℕ := 20
def second_day_tulips : ℕ := 60
def second_day_roses : ℕ := 40
def third_day_tulips : ℕ := 6
def third_day_roses : ℕ := 16
def rose_price : ℝ := 3
def total_revenue : ℝ := 420

-- Question: What is the price of one tulip?
def tulip_price (T : ℝ) : ℝ :=
    first_day_tulips * T + first_day_roses * rose_price +
    second_day_tulips * T + second_day_roses * rose_price +
    third_day_tulips * T + third_day_roses * rose_price

-- Proof problem statement
theorem tulip_price_correct (T : ℝ) : tulip_price T = total_revenue → T = 2 :=
by
  sorry

end NUMINAMATH_GPT_tulip_price_correct_l73_7393


namespace NUMINAMATH_GPT_intersection_complement_l73_7389

open Set

variable (U P Q : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5, 6})
variable (H_P : P = {1, 2, 3, 4})
variable (H_Q : Q = {3, 4, 5})

theorem intersection_complement (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5}) :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l73_7389


namespace NUMINAMATH_GPT_proof_y_solves_diff_eqn_l73_7313

noncomputable def y (x : ℝ) : ℝ := Real.exp (2 * x)

theorem proof_y_solves_diff_eqn : ∀ x : ℝ, (deriv^[3] y x) - 8 * y x = 0 := by
  sorry

end NUMINAMATH_GPT_proof_y_solves_diff_eqn_l73_7313


namespace NUMINAMATH_GPT_min_value_is_1_5_l73_7323

noncomputable def min_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) : ℝ :=
  (1 : ℝ) / (a + b) + 
  (1 : ℝ) / (b + c) + 
  (1 : ℝ) / (c + a)

theorem min_value_is_1_5 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  min_value a b c h1 h2 h3 h4 = 1.5 :=
sorry

end NUMINAMATH_GPT_min_value_is_1_5_l73_7323


namespace NUMINAMATH_GPT_number_of_balls_selected_is_three_l73_7367

-- Definitions of conditions
def total_balls : ℕ := 100
def odd_balls_selected : ℕ := 2
def even_balls_selected : ℕ := 1
def probability_first_ball_odd : ℚ := 2 / 3

-- The number of balls selected
def balls_selected := odd_balls_selected + even_balls_selected

-- Statement of the proof problem
theorem number_of_balls_selected_is_three 
(h1 : total_balls = 100)
(h2 : odd_balls_selected = 2)
(h3 : even_balls_selected = 1)
(h4 : probability_first_ball_odd = 2 / 3) :
  balls_selected = 3 :=
sorry

end NUMINAMATH_GPT_number_of_balls_selected_is_three_l73_7367


namespace NUMINAMATH_GPT_circle_center_radius_l73_7343

def circle_equation (x y : ℝ) : Prop := x^2 + 4 * x + y^2 - 6 * y - 12 = 0

theorem circle_center_radius :
  ∃ (h k r : ℝ), (circle_equation (x : ℝ) (y: ℝ) -> (x + h)^2 + (y + k)^2 = r^2) ∧ h = -2 ∧ k = 3 ∧ r = 5 :=
sorry

end NUMINAMATH_GPT_circle_center_radius_l73_7343


namespace NUMINAMATH_GPT_segments_not_arrangeable_l73_7314

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

end NUMINAMATH_GPT_segments_not_arrangeable_l73_7314


namespace NUMINAMATH_GPT_find_a_b_l73_7344

theorem find_a_b (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a * x^2 + b) 
  (h2 : ∀ x, f' x = 3 * x^2 + 2 * a * x) 
  (h3 : f' 1 = -3) 
  (h4 : f 1 = 0) : 
  a = -3 ∧ b = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_b_l73_7344


namespace NUMINAMATH_GPT_moment_goal_equality_l73_7357

theorem moment_goal_equality (total_goals_russia total_goals_tunisia : ℕ) (T : total_goals_russia = 9) (T2 : total_goals_tunisia = 5) :
  ∃ n, n ≤ 9 ∧ (9 - n) = total_goals_tunisia :=
by
  sorry

end NUMINAMATH_GPT_moment_goal_equality_l73_7357


namespace NUMINAMATH_GPT_zero_of_fn_exists_between_2_and_3_l73_7337

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 3 * x - 9

theorem zero_of_fn_exists_between_2_and_3 :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
sorry

end NUMINAMATH_GPT_zero_of_fn_exists_between_2_and_3_l73_7337


namespace NUMINAMATH_GPT_fly_least_distance_l73_7304

noncomputable def least_distance_fly_crawled (radius height dist_start dist_end : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let slant_height := Real.sqrt (radius^2 + height^2)
  let angle := circumference / slant_height
  let half_angle := angle / 2
  let start_x := dist_start
  let end_x := dist_end * Real.cos half_angle
  let end_y := dist_end * Real.sin half_angle
  Real.sqrt ((end_x - start_x)^2 + end_y^2)

theorem fly_least_distance : least_distance_fly_crawled 500 (300 * Real.sqrt 3) 150 (450 * Real.sqrt 2) = 486.396 := by
  sorry

end NUMINAMATH_GPT_fly_least_distance_l73_7304


namespace NUMINAMATH_GPT_sin_lg_roots_l73_7353

theorem sin_lg_roots (f : ℝ → ℝ) (g : ℝ → ℝ) (h₁ : ∀ x, f x = Real.sin x) (h₂ : ∀ x, g x = Real.log x)
  (domain : ∀ x, x > 0 → x < 10) (h₃ : ∀ x, f x ≤ 1 ∧ g x ≤ 1) :
  ∃ x1 x2 x3, (0 < x1 ∧ x1 < 10) ∧ (f x1 = g x1) ∧
               (0 < x2 ∧ x2 < 10) ∧ (f x2 = g x2) ∧
               (0 < x3 ∧ x3 < 10) ∧ (f x3 = g x3) ∧
               x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
by
  sorry

end NUMINAMATH_GPT_sin_lg_roots_l73_7353


namespace NUMINAMATH_GPT_perimeter_triangle_APR_l73_7365

-- Define given lengths
def AB := 24
def AC := AB
def AP := 8
def AR := AP

-- Define lengths calculated from conditions 
def PB := AB - AP
def RC := AC - AR

-- Define properties from the tangent intersection at Q
def PQ := PB
def QR := RC
def PR := PQ + QR

-- Calculate the perimeter
def perimeter_APR := AP + PR + AR

-- Proof of the problem statement
theorem perimeter_triangle_APR : perimeter_APR = 48 :=
by
  -- Calculations already given in the statement
  sorry

end NUMINAMATH_GPT_perimeter_triangle_APR_l73_7365


namespace NUMINAMATH_GPT_minimum_period_f_l73_7308

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x / 2 + Real.pi / 4)

theorem minimum_period_f :
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T) :=
sorry

end NUMINAMATH_GPT_minimum_period_f_l73_7308


namespace NUMINAMATH_GPT_compare_two_and_neg_three_l73_7384

theorem compare_two_and_neg_three (h1 : 2 > 0) (h2 : -3 < 0) : 2 > -3 :=
by
  sorry

end NUMINAMATH_GPT_compare_two_and_neg_three_l73_7384


namespace NUMINAMATH_GPT_deanna_initial_speed_l73_7328

namespace TripSpeed

variables (v : ℝ) (h : v > 0)

def speed_equation (v : ℝ) : Prop :=
  (1/2 * v) + (1/2 * (v + 20)) = 100

theorem deanna_initial_speed (v : ℝ) (h : speed_equation v) : v = 90 := sorry

end TripSpeed

end NUMINAMATH_GPT_deanna_initial_speed_l73_7328


namespace NUMINAMATH_GPT_gcd_seq_finitely_many_values_l73_7327

def gcd_seq_finite_vals (A B : ℕ) (x : ℕ → ℕ) : Prop :=
  (∀ n ≥ 2, x (n + 1) = A * Nat.gcd (x n) (x (n-1)) + B) →
  ∃ N : ℕ, ∀ m n, m ≥ N → n ≥ N → x m = x n

theorem gcd_seq_finitely_many_values (A B : ℕ) (x : ℕ → ℕ) :
  gcd_seq_finite_vals A B x :=
by
  intros h
  sorry

end NUMINAMATH_GPT_gcd_seq_finitely_many_values_l73_7327


namespace NUMINAMATH_GPT_b5b9_l73_7348

-- Assuming the sequences are indexed from natural numbers starting at 1
-- a_n is an arithmetic sequence with common difference d
-- b_n is a geometric sequence
-- Given conditions
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry
def d : ℝ := sorry
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) - a n = d
axiom d_nonzero : d ≠ 0
axiom condition_arith : 2 * a 4 - a 7 ^ 2 + 2 * a 10 = 0
axiom geometric_seq : ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1
axiom b7_equals_a7 : b 7 = a 7

-- To prove
theorem b5b9 : b 5 * b 9 = 16 :=
by
  sorry

end NUMINAMATH_GPT_b5b9_l73_7348


namespace NUMINAMATH_GPT_linear_function_mask_l73_7355

theorem linear_function_mask (x : ℝ) : ∃ k, k = 0.9 ∧ ∀ x, y = k * x :=
by
  sorry

end NUMINAMATH_GPT_linear_function_mask_l73_7355


namespace NUMINAMATH_GPT_jack_turn_in_correct_amount_l73_7306

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

end NUMINAMATH_GPT_jack_turn_in_correct_amount_l73_7306


namespace NUMINAMATH_GPT_alpha_plus_2beta_eq_45_l73_7331

theorem alpha_plus_2beta_eq_45 
  (α β : ℝ) 
  (hα_pos : 0 < α ∧ α < π / 2) 
  (hβ_pos : 0 < β ∧ β < π / 2) 
  (tan_alpha : Real.tan α = 1 / 7) 
  (sin_beta : Real.sin β = 1 / Real.sqrt 10)
  : α + 2 * β = π / 4 :=
sorry

end NUMINAMATH_GPT_alpha_plus_2beta_eq_45_l73_7331


namespace NUMINAMATH_GPT_polynomial_equivalence_l73_7350

-- Define the polynomial 'A' according to the conditions provided
def polynomial_A (x : ℝ) : ℝ := x^2 - 2*x

-- Define the given equation with polynomial A
def given_equation (x : ℝ) (A : ℝ) : Prop :=
  (x / (x + 2)) = (A / (x^2 - 4))

-- Prove that for the given equation, the polynomial 'A' is 'x^2 - 2x'
theorem polynomial_equivalence (x : ℝ) : given_equation x (polynomial_A x) :=
  by
    sorry -- Proof is skipped

end NUMINAMATH_GPT_polynomial_equivalence_l73_7350


namespace NUMINAMATH_GPT_subtraction_verification_l73_7369

theorem subtraction_verification : 888888888888 - 111111111111 = 777777777777 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_verification_l73_7369


namespace NUMINAMATH_GPT_jackson_investment_ratio_l73_7316

theorem jackson_investment_ratio:
  ∀ (B J: ℝ), B = 0.20 * 500 → J = B + 1900 → (J / 500) = 4 :=
by
  intros B J hB hJ
  sorry

end NUMINAMATH_GPT_jackson_investment_ratio_l73_7316


namespace NUMINAMATH_GPT_find_positive_real_solution_l73_7341

theorem find_positive_real_solution (x : ℝ) (h : 0 < x) :
  (1 / 3) * (4 * x ^ 2 - 3) = (x ^ 2 - 75 * x - 15) * (x ^ 2 + 40 * x + 8) →
  x = (75 + Real.sqrt (75 ^ 2 + 4 * 13)) / 2 ∨ x = (-40 + Real.sqrt (40 ^ 2 - 4 * 7)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_real_solution_l73_7341


namespace NUMINAMATH_GPT_simplify_expression_l73_7302

theorem simplify_expression (x y : ℝ) : 2 - (3 - (2 + (5 - (3 * y - x)))) = 6 - 3 * y + x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l73_7302


namespace NUMINAMATH_GPT_expression_value_at_2_l73_7390

theorem expression_value_at_2 : (2^2 + 3 * 2 - 4) = 6 :=
by 
  sorry

end NUMINAMATH_GPT_expression_value_at_2_l73_7390


namespace NUMINAMATH_GPT_equivalent_annual_rate_8_percent_quarterly_is_8_24_l73_7318

noncomputable def quarterly_interest_rate (annual_rate : ℚ) := annual_rate / 4

noncomputable def growth_factor (interest_rate : ℚ) := 1 + interest_rate / 100

noncomputable def annual_growth_factor_from_quarterly (quarterly_factor : ℚ) := quarterly_factor ^ 4

noncomputable def equivalent_annual_interest_rate (annual_growth_factor : ℚ) := 
  ((annual_growth_factor - 1) * 100)

theorem equivalent_annual_rate_8_percent_quarterly_is_8_24 :
  let quarter_rate := quarterly_interest_rate 8
  let quarterly_factor := growth_factor quarter_rate
  let annual_factor := annual_growth_factor_from_quarterly quarterly_factor
  equivalent_annual_interest_rate annual_factor = 8.24 := by
  sorry

end NUMINAMATH_GPT_equivalent_annual_rate_8_percent_quarterly_is_8_24_l73_7318


namespace NUMINAMATH_GPT_exists_nat_a_b_l73_7376

theorem exists_nat_a_b (n : ℕ) (hn : 0 < n) : 
∃ a b : ℕ, 1 ≤ b ∧ b ≤ n ∧ |a - b * Real.sqrt 2| ≤ 1 / n :=
by
  -- The proof steps would be filled here.
  sorry

end NUMINAMATH_GPT_exists_nat_a_b_l73_7376


namespace NUMINAMATH_GPT_water_left_after_four_hours_l73_7395

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def water_added_hour3 : ℕ := 1
def water_added_hour4 : ℕ := 3

theorem water_left_after_four_hours :
    initial_water - 2 * 2 - 2 + water_added_hour3 - 2 + water_added_hour4 - 2 = 36 := 
by
    sorry

end NUMINAMATH_GPT_water_left_after_four_hours_l73_7395


namespace NUMINAMATH_GPT_f_eq_l73_7361

noncomputable def a (n : ℕ) : ℚ := 1 / ((n + 1) ^ 2)

noncomputable def f : ℕ → ℚ
| 0     => 1
| (n+1) => f n * (1 - a (n+1))

theorem f_eq : ∀ n : ℕ, f n = (n + 2) / (2 * (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_f_eq_l73_7361


namespace NUMINAMATH_GPT_percentage_of_water_in_nectar_l73_7370

-- Define the necessary conditions and variables
def weight_of_nectar : ℝ := 1.7 -- kg
def weight_of_honey : ℝ := 1 -- kg
def honey_water_percentage : ℝ := 0.15 -- 15%

noncomputable def water_in_honey : ℝ := weight_of_honey * honey_water_percentage -- Water content in 1 kg of honey

noncomputable def total_water_in_nectar : ℝ := water_in_honey + (weight_of_nectar - weight_of_honey) -- Total water content in nectar

-- The theorem to prove
theorem percentage_of_water_in_nectar :
    (total_water_in_nectar / weight_of_nectar) * 100 = 50 := 
by 
    -- Skipping the proof by using sorry as it is not required
    sorry

end NUMINAMATH_GPT_percentage_of_water_in_nectar_l73_7370


namespace NUMINAMATH_GPT_inequality_pqr_l73_7346

theorem inequality_pqr (p q r : ℝ) (n : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p * q * r = 1) :
  1 / (p^n + q^n + 1) + 1 / (q^n + r^n + 1) + 1 / (r^n + p^n + 1) ≤ 1 :=
sorry

end NUMINAMATH_GPT_inequality_pqr_l73_7346


namespace NUMINAMATH_GPT_exposed_surface_area_hemisphere_l73_7375

-- Given conditions
def radius : ℝ := 10
def height_above_liquid : ℝ := 5

-- The attempt to state the problem as a proposition
theorem exposed_surface_area_hemisphere : 
  (π * radius ^ 2) + (π * radius * height_above_liquid) = 200 * π :=
by
  sorry

end NUMINAMATH_GPT_exposed_surface_area_hemisphere_l73_7375


namespace NUMINAMATH_GPT_initial_quantity_of_gummy_worms_l73_7371

theorem initial_quantity_of_gummy_worms (x : ℕ) (h : x / 2^4 = 4) : x = 64 :=
sorry

end NUMINAMATH_GPT_initial_quantity_of_gummy_worms_l73_7371


namespace NUMINAMATH_GPT_find_other_root_of_quadratic_l73_7382

theorem find_other_root_of_quadratic (m x_1 x_2 : ℝ) 
  (h_root1 : x_1 = 1) (h_eqn : ∀ x, x^2 - 4 * x + m = 0) : x_2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_other_root_of_quadratic_l73_7382


namespace NUMINAMATH_GPT_cut_into_two_pieces_is_possible_cut_into_three_pieces_is_impossible_cut_into_four_pieces_is_possible_cut_into_five_pieces_is_impossible_l73_7307

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

end NUMINAMATH_GPT_cut_into_two_pieces_is_possible_cut_into_three_pieces_is_impossible_cut_into_four_pieces_is_possible_cut_into_five_pieces_is_impossible_l73_7307


namespace NUMINAMATH_GPT_tomatoes_difference_is_50_l73_7324

variable (yesterday_tomatoes today_tomatoes total_tomatoes : ℕ)

theorem tomatoes_difference_is_50 
  (h1 : yesterday_tomatoes = 120)
  (h2 : total_tomatoes = 290)
  (h3 : total_tomatoes = today_tomatoes + yesterday_tomatoes) :
  today_tomatoes - yesterday_tomatoes = 50 := sorry

end NUMINAMATH_GPT_tomatoes_difference_is_50_l73_7324


namespace NUMINAMATH_GPT_min_value_expr_l73_7372

theorem min_value_expr (m n : ℝ) (h : m - n^2 = 8) : m^2 - 3 * n^2 + m - 14 ≥ 58 :=
sorry

end NUMINAMATH_GPT_min_value_expr_l73_7372


namespace NUMINAMATH_GPT_line_does_not_pass_through_third_quadrant_l73_7311

theorem line_does_not_pass_through_third_quadrant (k : ℝ) :
  (∀ x : ℝ, ¬ (x > 0 ∧ (-3 * x + k) < 0)) ∧ (∀ x : ℝ, ¬ (x < 0 ∧ (-3 * x + k) > 0)) → k ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_line_does_not_pass_through_third_quadrant_l73_7311


namespace NUMINAMATH_GPT_tan_C_in_triangle_l73_7319

theorem tan_C_in_triangle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : Real.tan A = 1) (h₃ : Real.tan B = 2) :
  Real.tan C = 3 :=
sorry

end NUMINAMATH_GPT_tan_C_in_triangle_l73_7319


namespace NUMINAMATH_GPT_find_positive_integers_n_satisfying_equation_l73_7362

theorem find_positive_integers_n_satisfying_equation :
  ∀ x y z : ℕ,
  x > 0 → y > 0 → z > 0 →
  (x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) →
  (n = 1 ∨ n = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integers_n_satisfying_equation_l73_7362


namespace NUMINAMATH_GPT_point_value_of_other_questions_is_4_l73_7352

theorem point_value_of_other_questions_is_4
  (total_points : ℕ)
  (total_questions : ℕ)
  (points_from_2_point_questions : ℕ)
  (other_questions : ℕ)
  (points_each_2_point_question : ℕ)
  (points_from_2_point_questions_calc : ℕ)
  (remaining_points : ℕ)
  (point_value_of_other_type : ℕ)
  : total_points = 100 →
    total_questions = 40 →
    points_each_2_point_question = 2 →
    other_questions = 10 →
    points_from_2_point_questions = 30 →
    points_from_2_point_questions_calc = points_each_2_point_question * points_from_2_point_questions →
    remaining_points = total_points - points_from_2_point_questions_calc →
    remaining_points = other_questions * point_value_of_other_type →
    point_value_of_other_type = 4 := by
  sorry

end NUMINAMATH_GPT_point_value_of_other_questions_is_4_l73_7352


namespace NUMINAMATH_GPT_arithmetic_sqrt_of_9_l73_7392

theorem arithmetic_sqrt_of_9 : ∃ y : ℝ, y ^ 2 = 9 ∧ y ≥ 0 ∧ y = 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_of_9_l73_7392


namespace NUMINAMATH_GPT_find_a_plus_b_l73_7363

-- Given points A and B, where A(1, a) and B(b, -2) are symmetric with respect to the origin.
variables (a b : ℤ)

-- Definition for symmetry conditions
def symmetric_wrt_origin (x1 y1 x2 y2 : ℤ) :=
  x2 = -x1 ∧ y2 = -y1

-- The main theorem
theorem find_a_plus_b :
  symmetric_wrt_origin 1 a b (-2) → a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l73_7363


namespace NUMINAMATH_GPT_fraction_of_A_or_B_l73_7336

def fraction_A : ℝ := 0.7
def fraction_B : ℝ := 0.2

theorem fraction_of_A_or_B : fraction_A + fraction_B = 0.9 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_A_or_B_l73_7336


namespace NUMINAMATH_GPT_calories_in_200_grams_is_137_l73_7383

-- Define the grams of ingredients used.
def lemon_juice_grams := 100
def sugar_grams := 100
def water_grams := 400

-- Define the calories per 100 grams of each ingredient.
def lemon_juice_calories_per_100_grams := 25
def sugar_calories_per_100_grams := 386
def water_calories_per_100_grams := 0

-- Calculate the total calories in the entire lemonade mixture.
def total_calories : Nat :=
  (lemon_juice_grams * lemon_juice_calories_per_100_grams / 100) + 
  (sugar_grams * sugar_calories_per_100_grams / 100) +
  (water_grams * water_calories_per_100_grams / 100)

-- Calculate the total weight of the lemonade mixture.
def total_weight : Nat := lemon_juice_grams + sugar_grams + water_grams

-- Calculate the caloric density (calories per gram).
def caloric_density := total_calories / total_weight

-- Calculate the calories in 200 grams of lemonade.
def calories_in_200_grams := (caloric_density * 200)

-- The theorem to prove
theorem calories_in_200_grams_is_137 : calories_in_200_grams = 137 :=
by sorry

end NUMINAMATH_GPT_calories_in_200_grams_is_137_l73_7383


namespace NUMINAMATH_GPT_boxes_with_neither_l73_7359

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

end NUMINAMATH_GPT_boxes_with_neither_l73_7359


namespace NUMINAMATH_GPT_average_speed_is_35_l73_7366

-- Given constants
def distance : ℕ := 210
def speed_difference : ℕ := 5
def time_difference : ℕ := 1

-- Definition of time for planned speed and actual speed
def planned_time (x : ℕ) : ℚ := distance / (x - speed_difference)
def actual_time (x : ℕ) : ℚ := distance / x

-- Main theorem to be proved
theorem average_speed_is_35 (x : ℕ) (h : (planned_time x - actual_time x) = time_difference) : x = 35 :=
sorry

end NUMINAMATH_GPT_average_speed_is_35_l73_7366


namespace NUMINAMATH_GPT_buses_trips_product_l73_7335

theorem buses_trips_product :
  ∃ (n k : ℕ), n > 3 ∧ n * (n - 1) * (2 * k - 1) = 600 ∧ (n * k = 52 ∨ n * k = 40) := 
by
  sorry

end NUMINAMATH_GPT_buses_trips_product_l73_7335


namespace NUMINAMATH_GPT_lopez_family_seating_arrangement_count_l73_7330

def lopez_family_seating_arrangements : Nat := 2 * 4 * 6

theorem lopez_family_seating_arrangement_count : lopez_family_seating_arrangements = 48 :=
by 
    sorry

end NUMINAMATH_GPT_lopez_family_seating_arrangement_count_l73_7330


namespace NUMINAMATH_GPT_books_left_over_l73_7312

theorem books_left_over (boxes : ℕ) (books_per_box_initial : ℕ) (books_per_box_new: ℕ) (total_books : ℕ) :
  boxes = 1500 →
  books_per_box_initial = 45 →
  books_per_box_new = 47 →
  total_books = boxes * books_per_box_initial →
  (total_books % books_per_box_new) = 8 :=
by intros; sorry

end NUMINAMATH_GPT_books_left_over_l73_7312


namespace NUMINAMATH_GPT_train_length_l73_7325

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 60) (h2 : time_sec = 9) : length_m = 150 := by
  sorry

end NUMINAMATH_GPT_train_length_l73_7325


namespace NUMINAMATH_GPT_general_term_min_sum_Sn_l73_7332

-- (I) Prove the general term formula for the arithmetic sequence
theorem general_term (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -10) 
  (geometric_cond : (a 2 + 10) * (a 4 + 6) = (a 3 + 8) ^ 2) : 
  ∃ n : ℕ, a n = 2 * n - 12 :=
by
  sorry

-- (II) Prove the minimum value of the sum of the first n terms
theorem min_sum_Sn (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -10)
  (general_term : ∀ n, a n = 2 * n - 12) : 
  ∃ n, S n = n * n - 11 * n ∧ S n = -30 :=
by
  sorry

end NUMINAMATH_GPT_general_term_min_sum_Sn_l73_7332


namespace NUMINAMATH_GPT_vector_magnitude_sum_l73_7315

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

end NUMINAMATH_GPT_vector_magnitude_sum_l73_7315


namespace NUMINAMATH_GPT_janet_wait_time_l73_7320

theorem janet_wait_time
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (janet_time : ℝ)
  (sister_time : ℝ) :
  janet_speed = 30 →
  sister_speed = 12 →
  lake_width = 60 →
  janet_time = lake_width / janet_speed →
  sister_time = lake_width / sister_speed →
  (sister_time - janet_time = 3) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_janet_wait_time_l73_7320


namespace NUMINAMATH_GPT_count_congruent_to_5_mod_7_l73_7364

theorem count_congruent_to_5_mod_7 (n : ℕ) :
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 300 ∧ x % 7 = 5) → ∃ count : ℕ, count = 43 := by
  sorry

end NUMINAMATH_GPT_count_congruent_to_5_mod_7_l73_7364


namespace NUMINAMATH_GPT_max_value_of_m_l73_7388

theorem max_value_of_m {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 12) (h_prod : a * b + b * c + c * a = 20) :
  ∃ m, m = min (a * b) (min (b * c) (c * a)) ∧ m = 12 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_m_l73_7388


namespace NUMINAMATH_GPT_cost_of_eight_memory_cards_l73_7301

theorem cost_of_eight_memory_cards (total_cost_of_three: ℕ) (h: total_cost_of_three = 45) : 8 * (total_cost_of_three / 3) = 120 := by
  sorry

end NUMINAMATH_GPT_cost_of_eight_memory_cards_l73_7301


namespace NUMINAMATH_GPT_problem1_problem2_l73_7385

variables {a b : ℝ}

-- Given conditions
def condition1 : a + b = 2 := sorry
def condition2 : a * b = -1 := sorry

-- Proof for a^2 + b^2 = 6
theorem problem1 (h1 : a + b = 2) (h2 : a * b = -1) : a^2 + b^2 = 6 :=
by sorry

-- Proof for (a - b)^2 = 8
theorem problem2 (h1 : a + b = 2) (h2 : a * b = -1) : (a - b)^2 = 8 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l73_7385


namespace NUMINAMATH_GPT_solve_problem_l73_7380

theorem solve_problem (a b c : ℤ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c)
    (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end NUMINAMATH_GPT_solve_problem_l73_7380


namespace NUMINAMATH_GPT_mike_total_spending_is_497_50_l73_7394

def rose_bush_price : ℝ := 75
def rose_bush_count : ℕ := 6
def rose_bush_discount : ℝ := 0.10
def friend_rose_bushes : ℕ := 2
def tax_rose_bushes : ℝ := 0.05

def aloe_price : ℝ := 100
def aloe_count : ℕ := 2
def tax_aloe : ℝ := 0.07

def calculate_total_cost_for_mike : ℝ :=
  let total_rose_bush_cost := rose_bush_price * rose_bush_count
  let discount := total_rose_bush_cost * rose_bush_discount
  let cost_after_discount := total_rose_bush_cost - discount
  let sales_tax_rose_bushes := tax_rose_bushes * cost_after_discount
  let cost_rose_bushes_after_tax := cost_after_discount + sales_tax_rose_bushes

  let total_aloe_cost := aloe_price * aloe_count
  let sales_tax_aloe := tax_aloe * total_aloe_cost

  let total_cost_friend_rose_bushes := friend_rose_bushes * (rose_bush_price - (rose_bush_price * rose_bush_discount))
  let sales_tax_friend_rose_bushes := tax_rose_bushes * total_cost_friend_rose_bushes
  let total_cost_friend := total_cost_friend_rose_bushes + sales_tax_friend_rose_bushes

  let total_mike_rose_bushes := cost_rose_bushes_after_tax - total_cost_friend

  let total_cost_mike_aloe := total_aloe_cost + sales_tax_aloe

  total_mike_rose_bushes + total_cost_mike_aloe

theorem mike_total_spending_is_497_50 : calculate_total_cost_for_mike = 497.50 := by
  sorry

end NUMINAMATH_GPT_mike_total_spending_is_497_50_l73_7394


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l73_7340

variable (a : ℚ)

theorem simplify_and_evaluate_expression (h : a = -1/3) : 
  (a + 1) * (a - 1) - a * (a + 3) = 0 := 
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l73_7340


namespace NUMINAMATH_GPT_piglet_steps_count_l73_7368

theorem piglet_steps_count (u v L : ℝ) (h₁ : (L * u) / (u + v) = 66) (h₂ : (L * u) / (u - v) = 198) : L = 99 :=
sorry

end NUMINAMATH_GPT_piglet_steps_count_l73_7368


namespace NUMINAMATH_GPT_GCF_LCM_example_l73_7381

/-- Greatest Common Factor (GCF) definition -/
def GCF (a b : ℕ) : ℕ := a.gcd b

/-- Least Common Multiple (LCM) definition -/
def LCM (a b : ℕ) : ℕ := a.lcm b

/-- Main theorem statement to prove -/
theorem GCF_LCM_example : 
  GCF (LCM 9 21) (LCM 8 15) = 3 := by
  sorry

end NUMINAMATH_GPT_GCF_LCM_example_l73_7381


namespace NUMINAMATH_GPT_factorize_square_difference_l73_7398

theorem factorize_square_difference (x: ℝ):
  x^2 - 4 = (x + 2) * (x - 2) := by
  -- Using the difference of squares formula a^2 - b^2 = (a + b)(a - b)
  sorry

end NUMINAMATH_GPT_factorize_square_difference_l73_7398


namespace NUMINAMATH_GPT_sum_of_coordinates_l73_7317

noncomputable def g : ℝ → ℝ := sorry
noncomputable def h (x : ℝ) : ℝ := (g x) ^ 2

theorem sum_of_coordinates : g 3 = 6 → (3 + h 3 = 39) := by
  intro hg3
  have : h 3 = (g 3) ^ 2 := by rfl
  rw [hg3] at this
  rw [this]
  exact sorry

end NUMINAMATH_GPT_sum_of_coordinates_l73_7317


namespace NUMINAMATH_GPT_prove_a_5_l73_7378

noncomputable def a_5_proof : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ),
    (∀ n, a n > 0) → 
    (a 1 + 2 * a 2 = 4) →
    ((a 1)^2 * q^6 = 4 * a 1 * q^2 * a 1 * q^6) →
    a 5 = 1 / 8

theorem prove_a_5 : a_5_proof := sorry

end NUMINAMATH_GPT_prove_a_5_l73_7378


namespace NUMINAMATH_GPT_selling_price_l73_7349

theorem selling_price (cost_price profit_percentage : ℝ) (h1 : cost_price = 90) (h2 : profit_percentage = 100) : 
    cost_price + (profit_percentage * cost_price / 100) = 180 :=
by
  rw [h1, h2]
  norm_num
  -- sorry

end NUMINAMATH_GPT_selling_price_l73_7349


namespace NUMINAMATH_GPT_lottery_probability_correct_l73_7342

noncomputable def probability_winning_lottery : ℚ :=
  let starBall_probability := 1 / 30
  let combinations (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  let magicBalls_probability := 1 / (combinations 49 6)
  starBall_probability * magicBalls_probability

theorem lottery_probability_correct :
  probability_winning_lottery = 1 / 419514480 := by
  sorry

end NUMINAMATH_GPT_lottery_probability_correct_l73_7342


namespace NUMINAMATH_GPT_rate_of_first_batch_l73_7333

theorem rate_of_first_batch (x : ℝ) 
  (cost_second_batch : ℝ := 20 * 14.25)
  (total_cost : ℝ := 30 * x + 285)
  (weight_mixture : ℝ := 30 + 20)
  (selling_price_per_kg : ℝ := 15.12) :
  (total_cost * 1.20 / weight_mixture = selling_price_per_kg) → x = 11.50 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_first_batch_l73_7333


namespace NUMINAMATH_GPT_cone_radius_l73_7374

theorem cone_radius (CSA : ℝ) (l : ℝ) (r : ℝ) (h_CSA : CSA = 989.6016858807849) (h_l : l = 15) :
    r = 21 :=
by
  sorry

end NUMINAMATH_GPT_cone_radius_l73_7374


namespace NUMINAMATH_GPT_hypotenuse_length_l73_7358

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l73_7358


namespace NUMINAMATH_GPT_ball_fall_time_l73_7321

theorem ball_fall_time (h g : ℝ) (t : ℝ) : 
  h = 20 → g = 10 → h + 20 * (t - 2) - 5 * ((t - 2) ^ 2) = t * (20 - 10 * (t - 2)) → 
  t = Real.sqrt 8 := 
by
  intros h_eq g_eq motion_eq
  sorry

end NUMINAMATH_GPT_ball_fall_time_l73_7321


namespace NUMINAMATH_GPT_line_m_eq_line_n_eq_l73_7322
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

end NUMINAMATH_GPT_line_m_eq_line_n_eq_l73_7322


namespace NUMINAMATH_GPT_initial_lives_l73_7338

theorem initial_lives (L : ℕ) (h1 : L - 6 + 37 = 41) : L = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_lives_l73_7338


namespace NUMINAMATH_GPT_new_tv_width_l73_7326

-- Define the conditions
def first_tv_width := 24
def first_tv_height := 16
def first_tv_cost := 672
def new_tv_height := 32
def new_tv_cost := 1152
def cost_difference := 1

-- Define the question as a theorem
theorem new_tv_width : 
  let first_tv_area := first_tv_width * first_tv_height
  let first_tv_cost_per_sq_inch := first_tv_cost / first_tv_area
  let new_tv_cost_per_sq_inch := first_tv_cost_per_sq_inch - cost_difference
  let new_tv_area := new_tv_cost / new_tv_cost_per_sq_inch
  let new_tv_width := new_tv_area / new_tv_height
  new_tv_width = 48 :=
by
  -- Here, we would normally provide the proof steps, but we insert sorry as required.
  sorry

end NUMINAMATH_GPT_new_tv_width_l73_7326


namespace NUMINAMATH_GPT_commission_rate_change_amount_l73_7334

theorem commission_rate_change_amount :
  ∃ X : ℝ, (∀ S : ℝ, ∀ commission : ℝ, S = 15885.42 → commission = (S - 15000) →
  commission = 0.10 * X + 0.05 * (S - X) → X = 1822.98) :=
sorry

end NUMINAMATH_GPT_commission_rate_change_amount_l73_7334


namespace NUMINAMATH_GPT_slope_angle_of_line_l73_7387

theorem slope_angle_of_line (x y : ℝ) (θ : ℝ) : (x - y + 3 = 0) → θ = 45 := 
sorry

end NUMINAMATH_GPT_slope_angle_of_line_l73_7387


namespace NUMINAMATH_GPT_children_count_l73_7305

theorem children_count (C A : ℕ) (h1 : 15 * A + 8 * C = 720) (h2 : A = C + 25) : C = 15 := 
by
  sorry

end NUMINAMATH_GPT_children_count_l73_7305


namespace NUMINAMATH_GPT_GCD_40_48_l73_7391

theorem GCD_40_48 : Int.gcd 40 48 = 8 :=
by sorry

end NUMINAMATH_GPT_GCD_40_48_l73_7391


namespace NUMINAMATH_GPT_rectangle_area_l73_7377

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l73_7377


namespace NUMINAMATH_GPT_missing_root_l73_7329

theorem missing_root (p q r : ℝ) 
  (h : p * (q - r) ≠ 0 ∧ q * (r - p) ≠ 0 ∧ r * (p - q) ≠ 0 ∧ 
       p * (q - r) * (-1)^2 + q * (r - p) * (-1) + r * (p - q) = 0) : 
  ∃ x : ℝ, x ≠ -1 ∧ 
  p * (q - r) * x^2 + q * (r - p) * x + r * (p - q) = 0 ∧ 
  x = - (r * (p - q) / (p * (q - r))) :=
sorry

end NUMINAMATH_GPT_missing_root_l73_7329


namespace NUMINAMATH_GPT_solve_for_x_and_compute_value_l73_7399

theorem solve_for_x_and_compute_value (x : ℝ) (h : 5 * x - 3 = 15 * x + 15) : 6 * (x + 5) = 19.2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_and_compute_value_l73_7399


namespace NUMINAMATH_GPT_no_a_for_x4_l73_7309

theorem no_a_for_x4 : ∃ a : ℝ, (1 / (4 + a) + 1 / (4 - a) = 1 / (4 - a)) → false :=
  by sorry

end NUMINAMATH_GPT_no_a_for_x4_l73_7309


namespace NUMINAMATH_GPT_inequality_abc_l73_7354

theorem inequality_abc (a b c : ℝ) 
  (habc : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ ab + bc + ca = 1) :
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 5 / 2) :=
sorry

end NUMINAMATH_GPT_inequality_abc_l73_7354


namespace NUMINAMATH_GPT_price_per_glass_first_day_l73_7310

theorem price_per_glass_first_day 
(O G : ℝ) (H : 2 * O * G * P₁ = 3 * O * G * 0.5466666666666666 ) : 
  P₁ = 0.82 :=
by
  sorry

end NUMINAMATH_GPT_price_per_glass_first_day_l73_7310
