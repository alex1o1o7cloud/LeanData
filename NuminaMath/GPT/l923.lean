import Mathlib

namespace NUMINAMATH_GPT_halfway_between_one_eighth_and_one_third_is_correct_l923_92383

-- Define the fractions
def one_eighth : ℚ := 1 / 8
def one_third : ℚ := 1 / 3

-- Define the correct answer
def correct_answer : ℚ := 11 / 48

-- State the theorem to prove the halfway number is correct_answer
theorem halfway_between_one_eighth_and_one_third_is_correct : 
  (one_eighth + one_third) / 2 = correct_answer :=
sorry

end NUMINAMATH_GPT_halfway_between_one_eighth_and_one_third_is_correct_l923_92383


namespace NUMINAMATH_GPT_problem_l923_92368

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : geometric_sequence a q)
  (h1 : a 0 + a 1 = 4 / 9)
  (h2 : a 2 + a 3 + a 4 + a 5 = 40) :
  (a 6 + a 7 + a 8) / 9 = 117 :=
sorry

end NUMINAMATH_GPT_problem_l923_92368


namespace NUMINAMATH_GPT_mean_home_runs_l923_92302

theorem mean_home_runs :
  let n_5 := 3
  let n_8 := 5
  let n_9 := 3
  let n_11 := 1
  let total_home_runs := 5 * n_5 + 8 * n_8 + 9 * n_9 + 11 * n_11
  let total_players := n_5 + n_8 + n_9 + n_11
  let mean := total_home_runs / total_players
  mean = 7.75 :=
by
  sorry

end NUMINAMATH_GPT_mean_home_runs_l923_92302


namespace NUMINAMATH_GPT_customers_who_did_not_tip_l923_92327

def total_customers := 10
def total_tips := 15
def tip_per_customer := 3

theorem customers_who_did_not_tip : total_customers - (total_tips / tip_per_customer) = 5 :=
by
  sorry

end NUMINAMATH_GPT_customers_who_did_not_tip_l923_92327


namespace NUMINAMATH_GPT_distinct_ordered_pair_count_l923_92374

theorem distinct_ordered_pair_count (x y : ℕ) (h1 : x + y = 50) (h2 : 1 ≤ x) (h3 : 1 ≤ y) : 
  ∃! (x y : ℕ), x + y = 50 ∧ 1 ≤ x ∧ 1 ≤ y :=
by
  sorry

end NUMINAMATH_GPT_distinct_ordered_pair_count_l923_92374


namespace NUMINAMATH_GPT_quotient_is_four_l923_92333

theorem quotient_is_four (dividend : ℕ) (k : ℕ) (h1 : dividend = 16) (h2 : k = 4) : dividend / k = 4 :=
by
  sorry

end NUMINAMATH_GPT_quotient_is_four_l923_92333


namespace NUMINAMATH_GPT_probability_of_at_most_3_heads_l923_92372

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end NUMINAMATH_GPT_probability_of_at_most_3_heads_l923_92372


namespace NUMINAMATH_GPT_ratio_of_linear_combination_l923_92365

theorem ratio_of_linear_combination (a b x y : ℝ) (hb : b ≠ 0) 
  (h1 : 4 * x - 2 * y = a) (h2 : 5 * y - 10 * x = b) :
  a / b = -2 / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_linear_combination_l923_92365


namespace NUMINAMATH_GPT_distance_covered_at_40_kmph_l923_92314

theorem distance_covered_at_40_kmph (x : ℝ) 
  (h₁ : x / 40 + (250 - x) / 60 = 5.5) :
  x = 160 :=
sorry

end NUMINAMATH_GPT_distance_covered_at_40_kmph_l923_92314


namespace NUMINAMATH_GPT_compute_expression_l923_92334

theorem compute_expression : 2 * (Real.sqrt 144)^2 = 288 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l923_92334


namespace NUMINAMATH_GPT_max_sum_of_four_distinct_with_lcm_165_l923_92393

theorem max_sum_of_four_distinct_with_lcm_165 (a b c d : ℕ)
  (h1 : Nat.lcm a b = 165)
  (h2 : Nat.lcm a c = 165)
  (h3 : Nat.lcm a d = 165)
  (h4 : Nat.lcm b c = 165)
  (h5 : Nat.lcm b d = 165)
  (h6 : Nat.lcm c d = 165)
  (h7 : a ≠ b) (h8 : a ≠ c) (h9 : a ≠ d)
  (h10 : b ≠ c) (h11 : b ≠ d) (h12 : c ≠ d) :
  a + b + c + d ≤ 268 := sorry

end NUMINAMATH_GPT_max_sum_of_four_distinct_with_lcm_165_l923_92393


namespace NUMINAMATH_GPT_bus_speed_including_stoppages_l923_92356

theorem bus_speed_including_stoppages
  (speed_excluding_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ) :
  speed_excluding_stoppages = 64 ∧ stoppage_time_per_hour = 15 / 60 →
  (44 / 60) * speed_excluding_stoppages = 48 :=
by
  sorry

end NUMINAMATH_GPT_bus_speed_including_stoppages_l923_92356


namespace NUMINAMATH_GPT_sum_of_first_five_primes_with_units_digit_3_l923_92364

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_five_primes_with_units_digit_3_l923_92364


namespace NUMINAMATH_GPT_rayden_spent_more_l923_92355

-- Define the conditions
def lily_ducks := 20
def lily_geese := 10
def lily_chickens := 5
def lily_pigeons := 30

def rayden_ducks := 3 * lily_ducks
def rayden_geese := 4 * lily_geese
def rayden_chickens := 5 * lily_chickens
def rayden_pigeons := lily_pigeons / 2

def duck_price := 15
def geese_price := 20
def chicken_price := 10
def pigeon_price := 5

def lily_total := lily_ducks * duck_price +
                  lily_geese * geese_price +
                  lily_chickens * chicken_price +
                  lily_pigeons * pigeon_price

def rayden_total := rayden_ducks * duck_price +
                    rayden_geese * geese_price +
                    rayden_chickens * chicken_price +
                    rayden_pigeons * pigeon_price

def spending_difference := rayden_total - lily_total

theorem rayden_spent_more : spending_difference = 1325 := 
by 
  unfold spending_difference rayden_total lily_total -- to simplify the definitions
  sorry -- Proof is omitted

end NUMINAMATH_GPT_rayden_spent_more_l923_92355


namespace NUMINAMATH_GPT_no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime_l923_92391

theorem no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 10000 ∧ (n % 10 = 0) ∧ (Prime n) → False :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime_l923_92391


namespace NUMINAMATH_GPT_sqrt_7_irrational_l923_92396

theorem sqrt_7_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a: ℝ) / b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_GPT_sqrt_7_irrational_l923_92396


namespace NUMINAMATH_GPT_ratio_M_N_l923_92362

theorem ratio_M_N (P Q M N : ℝ) (h1 : M = 0.30 * Q) (h2 : Q = 0.20 * P) (h3 : N = 0.50 * P) (hP_nonzero : P ≠ 0) :
  M / N = 3 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_M_N_l923_92362


namespace NUMINAMATH_GPT_scientific_notation_example_l923_92329

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * 10^b

theorem scientific_notation_example : 
  scientific_notation 0.00519 5.19 (-3) :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_example_l923_92329


namespace NUMINAMATH_GPT_sqrt_1_0201_eq_1_01_l923_92395

theorem sqrt_1_0201_eq_1_01 (h : Real.sqrt 102.01 = 10.1) : Real.sqrt 1.0201 = 1.01 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_1_0201_eq_1_01_l923_92395


namespace NUMINAMATH_GPT_average_rate_of_change_l923_92388

noncomputable def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change :
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_rate_of_change_l923_92388


namespace NUMINAMATH_GPT_bacteria_growth_time_l923_92328

-- Define the conditions and the final proof statement
theorem bacteria_growth_time (n0 n1 : ℕ) (t : ℕ) :
  (∀ (k : ℕ), k > 0 → n1 = n0 * 3 ^ k) →
  (∀ (h : ℕ), t = 5 * h) →
  n0 = 200 →
  n1 = 145800 →
  t = 30 :=
by
  sorry

end NUMINAMATH_GPT_bacteria_growth_time_l923_92328


namespace NUMINAMATH_GPT_measure_of_angle_Q_l923_92312

theorem measure_of_angle_Q (Q R : ℝ) 
  (h1 : Q = 2 * R)
  (h2 : 130 + 90 + 110 + 115 + Q + R = 540) :
  Q = 63.33 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_Q_l923_92312


namespace NUMINAMATH_GPT_smallest_solution_correct_l923_92378

noncomputable def smallest_solution (x : ℝ) : ℝ :=
if (⌊ x^2 ⌋ - ⌊ x ⌋^2 = 17) then x else 0

theorem smallest_solution_correct :
  smallest_solution (7 * Real.sqrt 2) = 7 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_smallest_solution_correct_l923_92378


namespace NUMINAMATH_GPT_Travis_annual_cereal_cost_l923_92370

def cost_of_box_A : ℚ := 2.50
def cost_of_box_B : ℚ := 3.50
def cost_of_box_C : ℚ := 4.00
def cost_of_box_D : ℚ := 5.25
def cost_of_box_E : ℚ := 6.00

def quantity_of_box_A : ℚ := 1
def quantity_of_box_B : ℚ := 0.5
def quantity_of_box_C : ℚ := 0.25
def quantity_of_box_D : ℚ := 0.75
def quantity_of_box_E : ℚ := 1.5

def cost_week1 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  cost_of_box_B * quantity_of_box_B +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  cost_of_box_E * quantity_of_box_E

def cost_week2 : ℚ :=
  let subtotal := 
    cost_of_box_A * quantity_of_box_A +
    cost_of_box_B * quantity_of_box_B +
    cost_of_box_C * quantity_of_box_C +
    cost_of_box_D * quantity_of_box_D +
    cost_of_box_E * quantity_of_box_E
  subtotal * 0.8

def cost_week3 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  0 +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  cost_of_box_E * quantity_of_box_E

def cost_week4 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  cost_of_box_B * quantity_of_box_B +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  let discounted_box_E := cost_of_box_E * quantity_of_box_E * 0.85
  cost_of_box_A * quantity_of_box_A +
  discounted_box_E
  
def monthly_cost : ℚ :=
  cost_week1 + cost_week2 + cost_week3 + cost_week4

def annual_cost : ℚ :=
  monthly_cost * 12

theorem Travis_annual_cereal_cost :
  annual_cost = 792.24 := by
  sorry

end NUMINAMATH_GPT_Travis_annual_cereal_cost_l923_92370


namespace NUMINAMATH_GPT_minimum_value_of_K_l923_92330

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / Real.exp x

noncomputable def f_K (K x : ℝ) : ℝ :=
  if f x ≤ K then f x else K

theorem minimum_value_of_K :
  (∀ x > 0, f_K (1 / Real.exp 1) x = f x) → (∃ K : ℝ, K = 1 / Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_K_l923_92330


namespace NUMINAMATH_GPT_monotonic_increasing_quadratic_l923_92398

theorem monotonic_increasing_quadratic (b : ℝ) (c : ℝ) :
  (∀ x y : ℝ, (0 ≤ x → x ≤ y → (x^2 + b*x + c) ≤ (y^2 + b*y + c))) ↔ (b ≥ 0) :=
sorry  -- Proof is omitted

end NUMINAMATH_GPT_monotonic_increasing_quadratic_l923_92398


namespace NUMINAMATH_GPT_min_value_of_xy_cond_l923_92307

noncomputable def minValueOfXY (x y : ℝ) : ℝ :=
  if 2 * Real.cos (x + y - 1) ^ 2 = ((x + 1) ^ 2 + (y - 1) ^ 2 - 2 * x * y) / (x - y + 1) then 
    x * y
  else 
    0

theorem min_value_of_xy_cond (x y : ℝ) 
  (h : 2 * Real.cos (x + y - 1) ^ 2 = ((x + 1) ^ 2 + (y - 1) ^ 2 - 2 * x * y) / (x - y + 1)) : 
  (∃ k : ℤ, x = (k * Real.pi + 1) / 2 ∧ y = (k * Real.pi + 1) / 2) → 
  x * y = 1/4 := 
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_min_value_of_xy_cond_l923_92307


namespace NUMINAMATH_GPT_shaded_regions_area_l923_92305

/-- Given a grid of 1x1 squares with 2015 shaded regions where boundaries are either:
    - Horizontal line segments
    - Vertical line segments
    - Segments connecting the midpoints of adjacent sides of 1x1 squares
    - Diagonals of 1x1 squares

    Prove that the total area of these 2015 shaded regions is 47.5.
-/
theorem shaded_regions_area (n : ℕ) (h1 : n = 2015) : 
  ∃ (area : ℝ), area = 47.5 :=
by sorry

end NUMINAMATH_GPT_shaded_regions_area_l923_92305


namespace NUMINAMATH_GPT_nonzero_fraction_exponent_zero_l923_92358

theorem nonzero_fraction_exponent_zero (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : (a / b : ℚ)^0 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_nonzero_fraction_exponent_zero_l923_92358


namespace NUMINAMATH_GPT_total_distance_apart_l923_92380

def Jay_rate : ℕ := 1 / 15 -- Jay walks 1 mile every 15 minutes
def Paul_rate : ℕ := 3 / 30 -- Paul walks 3 miles every 30 minutes
def time_in_minutes : ℕ := 120 -- 2 hours converted to minutes

def Jay_distance (rate time : ℕ) : ℕ := rate * time / 15
def Paul_distance (rate time : ℕ) : ℕ := rate * time / 30

theorem total_distance_apart : 
  Jay_distance Jay_rate time_in_minutes + Paul_distance Paul_rate time_in_minutes = 20 :=
  by
  -- Proof here
  sorry

end NUMINAMATH_GPT_total_distance_apart_l923_92380


namespace NUMINAMATH_GPT_area_of_base_of_cone_l923_92324

theorem area_of_base_of_cone (semicircle_area : ℝ) (h1 : semicircle_area = 2 * Real.pi) : 
  ∃ (base_area : ℝ), base_area = Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_base_of_cone_l923_92324


namespace NUMINAMATH_GPT_periodicity_of_m_arith_fibonacci_l923_92336

def m_arith_fibonacci (m : ℕ) (v : ℕ → ℕ) : Prop :=
∀ n : ℕ, v (n + 2) = (v n + v (n + 1)) % m

theorem periodicity_of_m_arith_fibonacci (m : ℕ) (v : ℕ → ℕ) 
  (hv : m_arith_fibonacci m v) : 
  ∃ r : ℕ, r ≤ m^2 ∧ ∀ n : ℕ, v (n + r) = v n := 
by
  sorry

end NUMINAMATH_GPT_periodicity_of_m_arith_fibonacci_l923_92336


namespace NUMINAMATH_GPT_car_mileage_l923_92377

/-- If a car needs 3.5 gallons of gasoline to travel 140 kilometers, it gets 40 kilometers per gallon. -/
theorem car_mileage (gallons_used : ℝ) (distance_traveled : ℝ) 
  (h : gallons_used = 3.5 ∧ distance_traveled = 140) : 
  distance_traveled / gallons_used = 40 :=
by
  sorry

end NUMINAMATH_GPT_car_mileage_l923_92377


namespace NUMINAMATH_GPT_sum_x_coordinates_l923_92375

-- Define the equations of the line segments
def segment1 (x : ℝ) := 2 * x + 6
def segment2 (x : ℝ) := -0.5 * x - 1.5
def segment3 (x : ℝ) := 2 * x + 1
def segment4 (x : ℝ) := -0.5 * x + 3.5
def segment5 (x : ℝ) := 2 * x - 4

-- Definition of the problem
theorem sum_x_coordinates (h1 : segment1 (-5) = -4 ∧ segment1 (-3) = 0)
    (h2 : segment2 (-3) = 0 ∧ segment2 (-1) = -1)
    (h3 : segment3 (-1) = -1 ∧ segment3 (1) = 3)
    (h4 : segment4 (1) = 3 ∧ segment4 (3) = 2)
    (h5 : segment5 (3) = 2 ∧ segment5 (5) = 6)
    (hx1 : ∃ x1, segment3 x1 = 2.4 ∧ -1 ≤ x1 ∧ x1 ≤ 1)
    (hx2 : ∃ x2, segment4 x2 = 2.4 ∧ 1 ≤ x2 ∧ x2 ≤ 3)
    (hx3 : ∃ x3, segment5 x3 = 2.4 ∧ 3 ≤ x3 ∧ x3 ≤ 5) :
    (∃ (x1 x2 x3 : ℝ), segment3 x1 = 2.4 ∧ segment4 x2 = 2.4 ∧ segment5 x3 = 2.4 ∧ x1 = 0.7 ∧ x2 = 2.2 ∧ x3 = 3.2 ∧ x1 + x2 + x3 = 6.1) :=
sorry

end NUMINAMATH_GPT_sum_x_coordinates_l923_92375


namespace NUMINAMATH_GPT_file_size_correct_l923_92306

theorem file_size_correct:
  (∀ t1 t2 : ℕ, (60 / 5 = t1) ∧ (15 - t1 = t2) ∧ (t2 * 10 = 30) → (60 + 30 = 90)) := 
by
  sorry

end NUMINAMATH_GPT_file_size_correct_l923_92306


namespace NUMINAMATH_GPT_second_order_arithmetic_sequence_a30_l923_92385

theorem second_order_arithmetic_sequence_a30 {a : ℕ → ℝ}
  (h₁ : ∀ n, a (n + 1) - a n - (a (n + 2) - a (n + 1)) = 20)
  (h₂ : a 10 = 23)
  (h₃ : a 20 = 23) :
  a 30 = 2023 := 
sorry

end NUMINAMATH_GPT_second_order_arithmetic_sequence_a30_l923_92385


namespace NUMINAMATH_GPT_multiple_of_P_l923_92344

theorem multiple_of_P (P Q R : ℝ) (T : ℝ) (x : ℝ) (total_profit Rs900 : ℝ)
  (h1 : P = 6 * Q)
  (h2 : P = 10 * R)
  (h3 : R = T / 5.1)
  (h4 : total_profit = Rs900 + (T - R)) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_P_l923_92344


namespace NUMINAMATH_GPT_ben_has_56_marbles_l923_92347

-- We define the conditions first
variables (B : ℕ) (L : ℕ)

-- Leo has 20 more marbles than Ben
def condition1 : Prop := L = B + 20

-- Total number of marbles is 132
def condition2 : Prop := B + L = 132

-- The goal: proving the number of marbles Ben has is 56
theorem ben_has_56_marbles (h1 : condition1 B L) (h2 : condition2 B L) : B = 56 :=
by sorry

end NUMINAMATH_GPT_ben_has_56_marbles_l923_92347


namespace NUMINAMATH_GPT_solve_for_x_l923_92301

theorem solve_for_x (x : ℝ) : (2 / 7) * (1 / 4) * x - 3 = 5 → x = 112 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l923_92301


namespace NUMINAMATH_GPT_sum_of_values_l923_92309

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 5 * x + 20 else 3 * x - 21

theorem sum_of_values (h₁ : ∃ x, x < 3 ∧ f x = 4) (h₂ : ∃ x, x ≥ 3 ∧ f x = 4) :
  ∃a b : ℝ, a = -16 / 5 ∧ b = 25 / 3 ∧ (a + b = 77 / 15) :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_values_l923_92309


namespace NUMINAMATH_GPT_no_solution_system_l923_92300

theorem no_solution_system : ¬ ∃ (x y z : ℝ), 
  x^2 - 2*y + 2 = 0 ∧ 
  y^2 - 4*z + 3 = 0 ∧ 
  z^2 + 4*x + 4 = 0 := 
by
  sorry

end NUMINAMATH_GPT_no_solution_system_l923_92300


namespace NUMINAMATH_GPT_total_pushups_l923_92345

def Zachary_pushups : ℕ := 44
def David_pushups : ℕ := Zachary_pushups + 58

theorem total_pushups : Zachary_pushups + David_pushups = 146 := by
  sorry

end NUMINAMATH_GPT_total_pushups_l923_92345


namespace NUMINAMATH_GPT_cos2_add_2sin2_eq_64_over_25_l923_92381

theorem cos2_add_2sin2_eq_64_over_25 (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
sorry

end NUMINAMATH_GPT_cos2_add_2sin2_eq_64_over_25_l923_92381


namespace NUMINAMATH_GPT_maximal_product_sum_l923_92348

theorem maximal_product_sum : 
  ∃ (k m : ℕ), 
  k = 671 ∧ 
  m = 2 ∧ 
  2017 = 3 * k + 2 * m ∧ 
  ∀ a b : ℕ, a + b = 2017 ∧ (a < k ∨ b < m) → a * b ≤ 3 * k * 2 * m
:= 
sorry

end NUMINAMATH_GPT_maximal_product_sum_l923_92348


namespace NUMINAMATH_GPT_adam_has_23_tattoos_l923_92394

-- Conditions as definitions
def tattoos_on_each_of_jason_arms := 2
def number_of_jason_arms := 2
def tattoos_on_each_of_jason_legs := 3
def number_of_jason_legs := 2

def jason_total_tattoos : Nat :=
  tattoos_on_each_of_jason_arms * number_of_jason_arms + tattoos_on_each_of_jason_legs * number_of_jason_legs

def adam_tattoos (jason_tattoos : Nat) : Nat :=
  2 * jason_tattoos + 3

-- The main theorem to be proved
theorem adam_has_23_tattoos : adam_tattoos jason_total_tattoos = 23 := by
  sorry

end NUMINAMATH_GPT_adam_has_23_tattoos_l923_92394


namespace NUMINAMATH_GPT_kanul_initial_amount_l923_92353

theorem kanul_initial_amount (X Y : ℝ) (loan : ℝ) (R : ℝ) 
  (h1 : loan = 2000)
  (h2 : R = 0.20)
  (h3 : Y = 0.15 * X + loan)
  (h4 : loan = R * Y) : 
  X = 53333.33 :=
by 
  -- The proof would come here, but is not necessary for this example
sorry

end NUMINAMATH_GPT_kanul_initial_amount_l923_92353


namespace NUMINAMATH_GPT_complex_fraction_evaluation_l923_92311

theorem complex_fraction_evaluation :
  ( 
    ((3 + 1/3) / 10 + 0.175 / 0.35) / 
    (1.75 - (1 + 11/17) * (51/56)) - 
    ((11/18 - 1/15) / 1.4) / 
    ((0.5 - 1/9) * 3)
  ) = 1/2 := 
sorry

end NUMINAMATH_GPT_complex_fraction_evaluation_l923_92311


namespace NUMINAMATH_GPT_actual_cost_l923_92363

theorem actual_cost (x : ℝ) (h : 0.80 * x = 200) : x = 250 :=
sorry

end NUMINAMATH_GPT_actual_cost_l923_92363


namespace NUMINAMATH_GPT_acid_solution_mix_l923_92367

theorem acid_solution_mix (x : ℝ) (h₁ : 0.2 * x + 50 = 0.35 * (100 + x)) : x = 100 :=
by
  sorry

end NUMINAMATH_GPT_acid_solution_mix_l923_92367


namespace NUMINAMATH_GPT_fraction_addition_simplification_l923_92382

theorem fraction_addition_simplification :
  (2 / 5 : ℚ) + (3 / 15) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_addition_simplification_l923_92382


namespace NUMINAMATH_GPT_geometric_seq_arithmetic_example_l923_92335

noncomputable def a_n (n : ℕ) (q : ℝ) : ℝ :=
if n = 0 then 1 else q ^ n

theorem geometric_seq_arithmetic_example {q : ℝ} (h₀ : q ≠ 0)
    (h₁ : ∀ n : ℕ, a_n 0 q = 1)
    (h₂ : 2 * (2 * (q ^ 2)) = 3 * q) :
    (q + q^2 + (q^3)) = 14 :=
by sorry

end NUMINAMATH_GPT_geometric_seq_arithmetic_example_l923_92335


namespace NUMINAMATH_GPT_solve_for_x_l923_92326

theorem solve_for_x (x : ℝ) (h₀ : x^2 - 2 * x = 0) (h₁ : x ≠ 0) : x = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l923_92326


namespace NUMINAMATH_GPT_tangent_y_axis_circle_eq_l923_92389

theorem tangent_y_axis_circle_eq (h k r : ℝ) (hc : h = -2) (kc : k = 3) (rc : r = abs h) :
  (x + h)^2 + (y - k)^2 = r^2 ↔ (x + 2)^2 + (y - 3)^2 = 4 := by
  sorry

end NUMINAMATH_GPT_tangent_y_axis_circle_eq_l923_92389


namespace NUMINAMATH_GPT_infinite_integer_solutions_l923_92318

theorem infinite_integer_solutions 
  (a b c k D x0 y0 : ℤ) 
  (hD_pos : D = b^2 - 4 * a * c) 
  (hD_non_square : (∀ n : ℤ, D ≠ n^2)) 
  (hk_nonzero : k ≠ 0) 
  (h_initial_sol : a * x0^2 + b * x0 * y0 + c * y0^2 = k) :
  ∃ (X Y : ℤ), a * X^2 + b * X * Y + c * Y^2 = k ∧
  (∀ (m : ℕ), ∃ (Xm Ym : ℤ), a * Xm^2 + b * Xm * Ym + c * Ym^2 = k ∧
  (Xm, Ym) ≠ (x0, y0)) :=
sorry

end NUMINAMATH_GPT_infinite_integer_solutions_l923_92318


namespace NUMINAMATH_GPT_inclination_angle_tan_60_perpendicular_l923_92399

/-
The inclination angle of the line given by x = tan(60 degrees) is 90 degrees.
-/
theorem inclination_angle_tan_60_perpendicular : 
  ∀ (x : ℝ), x = Real.tan (60 *Real.pi / 180) → 
  ∃ θ : ℝ, θ = 90 :=
sorry

end NUMINAMATH_GPT_inclination_angle_tan_60_perpendicular_l923_92399


namespace NUMINAMATH_GPT_puzzle_pieces_missing_l923_92386

/-- Trevor and Joe were working together to finish a 500 piece puzzle. 
They put the border together first and that was 75 pieces. 
Trevor was able to place 105 pieces of the puzzle.
Joe was able to place three times the number of puzzle pieces as Trevor. 
Prove that the number of puzzle pieces missing is 5. -/
theorem puzzle_pieces_missing :
  let total_pieces := 500
  let border_pieces := 75
  let trevor_pieces := 105
  let joe_pieces := 3 * trevor_pieces
  let placed_pieces := trevor_pieces + joe_pieces
  let remaining_pieces := total_pieces - border_pieces
  remaining_pieces - placed_pieces = 5 :=
by
  sorry

end NUMINAMATH_GPT_puzzle_pieces_missing_l923_92386


namespace NUMINAMATH_GPT_company_b_profit_l923_92313

-- Definitions as per problem conditions
def A_profit : ℝ := 90000
def A_share : ℝ := 0.60
def B_share : ℝ := 0.40

-- Theorem statement to be proved
theorem company_b_profit : B_share * (A_profit / A_share) = 60000 :=
by
  sorry

end NUMINAMATH_GPT_company_b_profit_l923_92313


namespace NUMINAMATH_GPT_mixture_weight_l923_92373

theorem mixture_weight :
  let weight_a_per_liter := 900 -- in gm
  let weight_b_per_liter := 750 -- in gm
  let ratio_a := 3
  let ratio_b := 2
  let total_volume := 4 -- in liters
  let volume_a := (ratio_a / (ratio_a + ratio_b)) * total_volume
  let volume_b := (ratio_b / (ratio_a + ratio_b)) * total_volume
  let weight_a := volume_a * weight_a_per_liter
  let weight_b := volume_b * weight_b_per_liter
  let total_weight_gm := weight_a + weight_b
  let total_weight_kg := total_weight_gm / 1000 
  total_weight_kg = 3.36 :=
by
  sorry

end NUMINAMATH_GPT_mixture_weight_l923_92373


namespace NUMINAMATH_GPT_leap_years_count_l923_92366

def is_leap_year (y : ℕ) : Bool :=
  if y % 800 = 300 ∨ y % 800 = 600 then true else false

theorem leap_years_count : 
  { y : ℕ // 1500 ≤ y ∧ y ≤ 3500 ∧ y % 100 = 0 ∧ is_leap_year y } = {y | y = 1900 ∨ y = 2200 ∨ y = 2700 ∨ y = 3000 ∨ y = 3500} :=
by
  sorry

end NUMINAMATH_GPT_leap_years_count_l923_92366


namespace NUMINAMATH_GPT_prime_pairs_l923_92376

open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, 2 ≤ m → m ≤ n / 2 → n % m ≠ 0

theorem prime_pairs :
  ∀ (p q : ℕ), is_prime p → is_prime q →
  1 < p → p < 100 →
  1 < q → q < 100 →
  is_prime (p + 6) →
  is_prime (p + 10) →
  is_prime (q + 4) →
  is_prime (q + 10) →
  is_prime (p + q + 1) →
  (p, q) = (7, 3) ∨ (p, q) = (13, 3) ∨ (p, q) = (37, 3) ∨ (p, q) = (97, 3) :=
by
  sorry

end NUMINAMATH_GPT_prime_pairs_l923_92376


namespace NUMINAMATH_GPT_mike_passing_percentage_l923_92316

theorem mike_passing_percentage (scored shortfall max_marks : ℝ) (total_marks := scored + shortfall) :
    scored = 212 →
    shortfall = 28 →
    max_marks = 800 →
    (total_marks / max_marks) * 100 = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mike_passing_percentage_l923_92316


namespace NUMINAMATH_GPT_sum_of_digits_joey_age_l923_92359

def int.multiple (a b : ℕ) := ∃ k : ℕ, a = k * b

theorem sum_of_digits_joey_age (J C M n : ℕ) (h1 : J = C + 2) (h2 : M = 2) (h3 : ∃ k, C = k * M) (h4 : C = 12) (h5 : J + n = 26) : 
  (2 + 6 = 8) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_joey_age_l923_92359


namespace NUMINAMATH_GPT_hyperbola_eccentricity_is_2_l923_92360

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let c := 4 * a
  let e := c / a
  e

theorem hyperbola_eccentricity_is_2
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity a b ha hb = 2 := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_is_2_l923_92360


namespace NUMINAMATH_GPT_min_value_x_y_l923_92343

theorem min_value_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 19 / x + 98 / y = 1) : x + y ≥ 117 + 14 * Real.sqrt 38 := 
sorry

end NUMINAMATH_GPT_min_value_x_y_l923_92343


namespace NUMINAMATH_GPT_find_n_l923_92308

theorem find_n (n : ℕ) (h : 12^(4 * n) = (1/12)^(n - 30)) : n = 6 := 
by {
  sorry 
}

end NUMINAMATH_GPT_find_n_l923_92308


namespace NUMINAMATH_GPT_candy_crush_ratio_l923_92303

theorem candy_crush_ratio :
  ∃ m : ℕ, (400 + (400 - 70) + (400 - 70) * m = 1390) ∧ (m = 2) :=
by
  sorry

end NUMINAMATH_GPT_candy_crush_ratio_l923_92303


namespace NUMINAMATH_GPT_ellipse_parabola_intersection_l923_92349

open Real

theorem ellipse_parabola_intersection (a : ℝ) :
  (∃ (x y : ℝ), x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) ↔ (-1 ≤ a ∧ a ≤ 17 / 8) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_parabola_intersection_l923_92349


namespace NUMINAMATH_GPT_inheritance_value_l923_92342

def inheritance_proof (x : ℝ) (federal_tax_ratio : ℝ) (state_tax_ratio : ℝ) (total_tax : ℝ) : Prop :=
  let federal_taxes := federal_tax_ratio * x
  let remaining_after_federal := x - federal_taxes
  let state_taxes := state_tax_ratio * remaining_after_federal
  let total_taxes := federal_taxes + state_taxes
  total_taxes = total_tax

theorem inheritance_value :
  inheritance_proof 41379 0.25 0.15 15000 :=
by
  sorry

end NUMINAMATH_GPT_inheritance_value_l923_92342


namespace NUMINAMATH_GPT_ones_digit_of_prime_in_arithmetic_sequence_l923_92332

theorem ones_digit_of_prime_in_arithmetic_sequence (p q r : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) 
  (arithmetic_sequence : q = p + 4 ∧ r = q + 4)
  (h : p > 5) : 
    (p % 10 = 3 ∨ p % 10 = 9) :=
sorry

end NUMINAMATH_GPT_ones_digit_of_prime_in_arithmetic_sequence_l923_92332


namespace NUMINAMATH_GPT_prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l923_92369

/-- Let us define the probabilities for school A winning the events -/
def prob_A_wins_event_1 : ℝ := 0.5
def prob_A_wins_event_2 : ℝ := 0.4
def prob_A_wins_event_3 : ℝ := 0.8

/-- The total probability of school A winning the championship -/
noncomputable def prob_A_championship_wins : ℝ :=
  prob_A_wins_event_1 * prob_A_wins_event_2 * prob_A_wins_event_3 +   -- All three events
  (prob_A_wins_event_1 * prob_A_wins_event_2 * (1 - prob_A_wins_event_3) + -- First two events
   prob_A_wins_event_1 * (1 - prob_A_wins_event_2) * prob_A_wins_event_3 + -- First and third event
   (1 - prob_A_wins_event_1) * prob_A_wins_event_2 * prob_A_wins_event_3)  -- Second and third events

/-- The distribution for school B's scores -/
def score_dist_B : List (ℕ × ℝ) :=
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

/-- The expectation of X (total score of school B) -/
noncomputable def expectation_X : ℝ :=
  0 * 0.16 + 10 * 0.44 + 20 * 0.34 + 30 * 0.06

/-- The proofs for the derived results -/
theorem prob_A_championship_win_is_correct : prob_A_championship_wins = 0.6 := sorry

theorem expectation_X_is_correct : expectation_X = 13 := sorry

theorem distribution_X_is_correct :
  score_dist_B = [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)] := sorry

end NUMINAMATH_GPT_prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l923_92369


namespace NUMINAMATH_GPT_right_triangle_cos_pq_l923_92322

theorem right_triangle_cos_pq (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : c = 13) (h2 : b / c = 5/13) : a = 12 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_cos_pq_l923_92322


namespace NUMINAMATH_GPT_range_of_a_l923_92352

theorem range_of_a (a : ℝ) (h : a - 2 * 1 + 4 > 0) : a > -2 :=
by
  -- proof is not required
  sorry

end NUMINAMATH_GPT_range_of_a_l923_92352


namespace NUMINAMATH_GPT_inverse_g_167_is_2_l923_92390

def g (x : ℝ) := 5 * x^5 + 7

theorem inverse_g_167_is_2 : g⁻¹' {167} = {2} := by
  sorry

end NUMINAMATH_GPT_inverse_g_167_is_2_l923_92390


namespace NUMINAMATH_GPT_metal_sheets_per_panel_l923_92392

-- Define the given conditions
def num_panels : ℕ := 10
def rods_per_sheet : ℕ := 10
def rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def total_rods_needed : ℕ := 380

-- Question translated to Lean statement
theorem metal_sheets_per_panel (S : ℕ) (h : 10 * (10 * S + 8) = 380) : S = 3 := 
  sorry

end NUMINAMATH_GPT_metal_sheets_per_panel_l923_92392


namespace NUMINAMATH_GPT_distance_between_foci_l923_92361

-- Given problem
def hyperbola_eq (x y : ℝ) : Prop := 9 * x^2 - 18 * x - 16 * y^2 + 32 * y = 144

theorem distance_between_foci :
  ∀ (x y : ℝ),
    hyperbola_eq x y →
    2 * Real.sqrt ((137 / 9) + (137 / 16)) / 72 = 38 * Real.sqrt 7 / 72 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_distance_between_foci_l923_92361


namespace NUMINAMATH_GPT_upper_limit_of_arun_weight_l923_92397

variable (w : ℝ)

noncomputable def arun_opinion (w : ℝ) := 62 < w ∧ w < 72
noncomputable def brother_opinion (w : ℝ) := 60 < w ∧ w < 70
noncomputable def average_weight := 64

theorem upper_limit_of_arun_weight 
  (h1 : ∀ w, arun_opinion w → brother_opinion w → 64 = (62 + w) / 2 ) 
  : ∀ w, arun_opinion w ∧ brother_opinion w → w ≤ 66 :=
sorry

end NUMINAMATH_GPT_upper_limit_of_arun_weight_l923_92397


namespace NUMINAMATH_GPT_age_ratio_albert_mary_l923_92320

variable (A M B : ℕ) 

theorem age_ratio_albert_mary
    (h1 : A = 4 * B)
    (h2 : M = A - 10)
    (h3 : B = 5) :
    A = 2 * M :=
by
    sorry

end NUMINAMATH_GPT_age_ratio_albert_mary_l923_92320


namespace NUMINAMATH_GPT_gomoku_black_pieces_l923_92384

/--
Two students, A and B, are preparing to play a game of Gomoku but find that 
the box only contains a certain number of black and white pieces, each of the
same quantity, and the total does not exceed 10. Then, they find 20 more pieces 
(only black and white) and add them to the box. At this point, the ratio of 
the total number of white to black pieces is 7:8. We want to prove that the total number
of black pieces in the box after adding is 16.
-/
theorem gomoku_black_pieces (x y : ℕ) (hx : x = 15 * y - 160) (h_total : x + y ≤ 5)
  (h_ratio : 7 * (x + y) = 8 * (x + (20 - y))) : (x + y = 16) :=
by
  sorry

end NUMINAMATH_GPT_gomoku_black_pieces_l923_92384


namespace NUMINAMATH_GPT_Penelope_Candies_l923_92304

variable (M : ℕ) (S : ℕ)
variable (h1 : 5 * S = 3 * M)
variable (h2 : M = 25)

theorem Penelope_Candies : S = 15 := by
  sorry

end NUMINAMATH_GPT_Penelope_Candies_l923_92304


namespace NUMINAMATH_GPT_max_value_x_plus_y_l923_92340

theorem max_value_x_plus_y : ∀ (x y : ℝ), 
  (5 * x + 3 * y ≤ 9) → 
  (3 * x + 5 * y ≤ 11) → 
  x + y ≤ 32 / 17 :=
by
  intros x y h1 h2
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_max_value_x_plus_y_l923_92340


namespace NUMINAMATH_GPT_plane_figures_l923_92317

def polyline_two_segments : Prop := -- Definition for a polyline composed of two line segments
  sorry

def polyline_three_segments : Prop := -- Definition for a polyline composed of three line segments
  sorry

def closed_three_segments : Prop := -- Definition for a closed figure composed of three line segments
  sorry

def quadrilateral_equal_opposite_sides : Prop := -- Definition for a quadrilateral with equal opposite sides
  sorry

def trapezoid : Prop := -- Definition for a trapezoid
  sorry

def is_plane_figure (fig : Prop) : Prop :=
  sorry  -- Axiom or definition that determines whether a figure is a plane figure.

-- Translating the proof problem
theorem plane_figures :
  is_plane_figure polyline_two_segments ∧
  ¬ is_plane_figure polyline_three_segments ∧
  is_plane_figure closed_three_segments ∧
  ¬ is_plane_figure quadrilateral_equal_opposite_sides ∧
  is_plane_figure trapezoid :=
by
  sorry

end NUMINAMATH_GPT_plane_figures_l923_92317


namespace NUMINAMATH_GPT_find_fifth_score_l923_92346

-- Define the known scores
def score1 : ℕ := 90
def score2 : ℕ := 93
def score3 : ℕ := 85
def score4 : ℕ := 97

-- Define the average of all scores
def average : ℕ := 92

-- Define the total number of scores
def total_scores : ℕ := 5

-- Define the total sum of all scores using the average
def total_sum : ℕ := total_scores * average

-- Define the sum of the four known scores
def known_sum : ℕ := score1 + score2 + score3 + score4

-- Define the fifth score
def fifth_score : ℕ := 95

-- Theorem statement: The fifth score plus the known sum equals the total sum.
theorem find_fifth_score : fifth_score + known_sum = total_sum := by
  sorry

end NUMINAMATH_GPT_find_fifth_score_l923_92346


namespace NUMINAMATH_GPT_find_original_price_l923_92337

-- Definitions for the conditions mentioned in the problem
variables {P : ℝ} -- Original price per gallon in dollars

-- Proof statement assuming the given conditions
theorem find_original_price 
  (h1 : ∃ P : ℝ, P > 0) -- There exists a positive price per gallon in dollars
  (h2 : (250 / (0.9 * P)) = (250 / P + 5)) -- After a 10% price reduction, 5 gallons more can be bought for $250
  : P = 25 / 4.5 := -- The solution states the original price per gallon is approximately $5.56
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_find_original_price_l923_92337


namespace NUMINAMATH_GPT_sum_of_cubes_of_nonneg_rationals_l923_92379

theorem sum_of_cubes_of_nonneg_rationals (n : ℤ) (h1 : n > 1) (h2 : ∃ a b : ℚ, a^3 + b^3 = n) :
  ∃ c d : ℚ, c ≥ 0 ∧ d ≥ 0 ∧ c^3 + d^3 = n :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_of_nonneg_rationals_l923_92379


namespace NUMINAMATH_GPT_m_range_l923_92310

noncomputable def otimes (a b : ℝ) : ℝ := 
if a > b then a else b

theorem m_range (m : ℝ) : (otimes (2 * m - 5) 3 = 3) ↔ (m ≤ 4) := by
  sorry

end NUMINAMATH_GPT_m_range_l923_92310


namespace NUMINAMATH_GPT_race_distance_l923_92354

theorem race_distance (x : ℝ) (D : ℝ) (vA vB : ℝ) (head_start win_margin : ℝ):
  vA = 5 * x →
  vB = 4 * x →
  head_start = 100 →
  win_margin = 200 →
  (D - win_margin) / vB = (D - head_start) / vA →
  D = 600 :=
by 
  sorry

end NUMINAMATH_GPT_race_distance_l923_92354


namespace NUMINAMATH_GPT_car_mpg_in_city_l923_92325

theorem car_mpg_in_city:
  ∃ (h c T : ℝ), 
    (420 = h * T) ∧ 
    (336 = c * T) ∧ 
    (c = h - 6) ∧ 
    (c = 24) :=
by
  sorry

end NUMINAMATH_GPT_car_mpg_in_city_l923_92325


namespace NUMINAMATH_GPT_trigonometric_values_l923_92341

-- Define cos and sin terms
def cos (x : ℝ) : ℝ := sorry
def sin (x : ℝ) : ℝ := sorry

-- Define the condition given in the problem statement
def condition (x : ℝ) : Prop := cos x - 4 * sin x = 1

-- Define the result we need to prove
def result (x : ℝ) : Prop := sin x + 4 * cos x = 4 ∨ sin x + 4 * cos x = -4

-- The main statement in Lean 4 to be proved
theorem trigonometric_values (x : ℝ) : condition x → result x := by
  sorry

end NUMINAMATH_GPT_trigonometric_values_l923_92341


namespace NUMINAMATH_GPT_weeks_to_work_l923_92319

def iPhone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80

theorem weeks_to_work (iPhone_cost trade_in_value weekly_earnings : ℕ) :
  (iPhone_cost - trade_in_value) / weekly_earnings = 7 :=
by
  sorry

end NUMINAMATH_GPT_weeks_to_work_l923_92319


namespace NUMINAMATH_GPT_smallest_possible_intersections_l923_92350

theorem smallest_possible_intersections (n : ℕ) (hn : n = 2000) :
  ∃ N : ℕ, N ≥ 3997 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_intersections_l923_92350


namespace NUMINAMATH_GPT_eight_digit_numbers_count_l923_92339

theorem eight_digit_numbers_count :
  let first_digit_choices := 9
  let remaining_digits_choices := 10 ^ 7
  9 * 10^7 = 90000000 :=
by
  sorry

end NUMINAMATH_GPT_eight_digit_numbers_count_l923_92339


namespace NUMINAMATH_GPT_total_capacity_iv_bottle_l923_92321

-- Definitions of the conditions
def initial_volume : ℝ := 100 -- milliliters
def rate_of_flow : ℝ := 2.5 -- milliliters per minute
def observation_time : ℝ := 12 -- minutes
def empty_space_at_12_min : ℝ := 80 -- milliliters

-- Definition of the problem statement in Lean 4
theorem total_capacity_iv_bottle :
  initial_volume + rate_of_flow * observation_time + empty_space_at_12_min = 150 := 
by
  sorry

end NUMINAMATH_GPT_total_capacity_iv_bottle_l923_92321


namespace NUMINAMATH_GPT_simplify_and_evaluate_l923_92351

variable (a : ℝ)
variable (ha : a = Real.sqrt 3 - 1)

theorem simplify_and_evaluate : 
  (1 + 3 / (a - 2)) / ((a^2 + 2 * a + 1) / (a - 2)) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l923_92351


namespace NUMINAMATH_GPT_cost_per_pizza_l923_92323

theorem cost_per_pizza (total_amount : ℝ) (num_pizzas : ℕ) (H : total_amount = 24) (H1 : num_pizzas = 3) : 
  (total_amount / num_pizzas) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_cost_per_pizza_l923_92323


namespace NUMINAMATH_GPT_minimum_k_exists_l923_92371

theorem minimum_k_exists (k : ℕ) (h : k > 0) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ a + c > b ∧ b + c > a) ↔ k = 6 :=
sorry

end NUMINAMATH_GPT_minimum_k_exists_l923_92371


namespace NUMINAMATH_GPT_max_theater_members_l923_92338

theorem max_theater_members (N : ℕ) :
  (∃ (k : ℕ), (N = k^2 + 3)) ∧ (∃ (n : ℕ), (N = n * (n + 9))) → N ≤ 360 :=
by
  sorry

end NUMINAMATH_GPT_max_theater_members_l923_92338


namespace NUMINAMATH_GPT_vincent_back_to_A_after_5_min_p_plus_q_computation_l923_92387

def probability (n : ℕ) : ℚ :=
  if n = 0 then 1
  else 1 / 4 * (1 - probability (n - 1))

theorem vincent_back_to_A_after_5_min : 
  probability 5 = 51 / 256 :=
by sorry

theorem p_plus_q_computation :
  51 + 256 = 307 :=
by linarith

end NUMINAMATH_GPT_vincent_back_to_A_after_5_min_p_plus_q_computation_l923_92387


namespace NUMINAMATH_GPT_part_I_part_II_l923_92315

-- Part (I)
theorem part_I (a₁ : ℝ) (d : ℝ) (S : ℕ → ℝ) (k : ℕ) :
  a₁ = 3 / 2 →
  d = 1 →
  (∀ n, S n = (n / 2 : ℝ) * (n + 2)) →
  S (k ^ 2) = S k ^ 2 →
  k = 4 :=
by
  intros ha₁ hd hSn hSeq
  sorry

-- Part (II)
theorem part_II (a : ℝ) (d : ℝ) (S : ℕ → ℝ) :
  (∀ k : ℕ, S (k ^ 2) = (S k) ^ 2) →
  ( (∀ n, a = 0 ∧ d = 0 ∧ a + d * (n - 1) = 0) ∨
    (∀ n, a = 1 ∧ d = 0 ∧ a + d * (n - 1) = 1) ∨
    (∀ n, a = 1 ∧ d = 2 ∧ a + d * (n - 1) = 2 * n - 1) ) :=
by
  intros hSeq
  sorry

end NUMINAMATH_GPT_part_I_part_II_l923_92315


namespace NUMINAMATH_GPT_problem_condition_l923_92331

theorem problem_condition (a : ℝ) (x : ℝ) (h_a : -1 ≤ a ∧ a ≤ 1) :
  (x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
sorry

end NUMINAMATH_GPT_problem_condition_l923_92331


namespace NUMINAMATH_GPT_ratio_of_first_to_fourth_term_l923_92357

theorem ratio_of_first_to_fourth_term (a d : ℝ) (h1 : (a + d) + (a + 3 * d) = 6 * a) (h2 : a + 2 * d = 10) :
  a / (a + 3 * d) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_first_to_fourth_term_l923_92357
