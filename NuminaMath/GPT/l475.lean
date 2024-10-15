import Mathlib

namespace NUMINAMATH_GPT_isosceles_triangle_leg_l475_47597

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ a = c ∨ b = c)

theorem isosceles_triangle_leg
  (a b c : ℝ)
  (h1 : is_isosceles_triangle a b c)
  (h2 : a + b + c = 18)
  (h3 : a = 8 ∨ b = 8 ∨ c = 8) :
  (a = 5 ∨ b = 5 ∨ c = 5 ∨ a = 8 ∨ b = 8 ∨ c = 8) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_leg_l475_47597


namespace NUMINAMATH_GPT_solve_system_of_equations_solve_algebraic_equation_l475_47591

-- Problem 1: System of Equations
theorem solve_system_of_equations (x y : ℝ) (h1 : x + 2 * y = 3) (h2 : 2 * x - y = 1) : x = 1 ∧ y = 1 :=
sorry

-- Problem 2: Algebraic Equation
theorem solve_algebraic_equation (x : ℝ) (h : 1 / (x - 1) + 2 = 5 / (1 - x)) : x = -2 :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_solve_algebraic_equation_l475_47591


namespace NUMINAMATH_GPT_smallest_possible_value_of_c_l475_47508

theorem smallest_possible_value_of_c
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (H : ∀ x : ℝ, (a * Real.sin (b * x + c)) ≤ (a * Real.sin (b * 0 + c))) :
  c = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_c_l475_47508


namespace NUMINAMATH_GPT_num_nat_numbers_divisible_by_7_between_100_and_250_l475_47574

noncomputable def countNatNumbersDivisibleBy7InRange : ℕ :=
  let smallest := Nat.ceil (100 / 7) * 7
  let largest := Nat.floor (250 / 7) * 7
  (largest - smallest) / 7 + 1

theorem num_nat_numbers_divisible_by_7_between_100_and_250 :
  countNatNumbersDivisibleBy7InRange = 21 :=
by
  -- Placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_num_nat_numbers_divisible_by_7_between_100_and_250_l475_47574


namespace NUMINAMATH_GPT_max_A_value_l475_47599

-- Variables
variables {x1 x2 x3 y1 y2 y3 z1 z2 z3 : ℝ}

-- Assumptions
axiom pos_x1 : 0 < x1
axiom pos_x2 : 0 < x2
axiom pos_x3 : 0 < x3
axiom pos_y1 : 0 < y1
axiom pos_y2 : 0 < y2
axiom pos_y3 : 0 < y3
axiom pos_z1 : 0 < z1
axiom pos_z2 : 0 < z2
axiom pos_z3 : 0 < z3

-- Statement
theorem max_A_value :
  ∃ A : ℝ, 
    (∀ x1 x2 x3 y1 y2 y3 z1 z2 z3, 
    (0 < x1) → (0 < x2) → (0 < x3) →
    (0 < y1) → (0 < y2) → (0 < y3) →
    (0 < z1) → (0 < z2) → (0 < z3) →
    (x1^3 + x2^3 + x3^3 + 1) * (y1^3 + y2^3 + y3^3 + 1) * (z1^3 + z2^3 + z3^3 + 1) ≥
    A * (x1 + y1 + z1) * (x2 + y2 + z2) * (x3 + y3 + z3)) ∧ 
    A = 9/2 := 
by 
  exists 9/2 
  sorry

end NUMINAMATH_GPT_max_A_value_l475_47599


namespace NUMINAMATH_GPT_ratio_time_A_to_B_l475_47567

-- Definition of total examination time in minutes
def total_time : ℕ := 180

-- Definition of time spent on type A problems
def time_A : ℕ := 40

-- Definition of time spent on type B problems as total_time - time_A
def time_B : ℕ := total_time - time_A

-- Statement that we need to prove
theorem ratio_time_A_to_B : time_A * 7 = time_B * 2 :=
by
  -- Implementation of the proof will go here
  sorry

end NUMINAMATH_GPT_ratio_time_A_to_B_l475_47567


namespace NUMINAMATH_GPT_distance_to_focus_l475_47577

open Real

theorem distance_to_focus {P : ℝ × ℝ} 
  (h₁ : P.2 ^ 2 = 4 * P.1)
  (h₂ : abs (P.1 + 3) = 5) :
  dist P ⟨1, 0⟩ = 3 := 
sorry

end NUMINAMATH_GPT_distance_to_focus_l475_47577


namespace NUMINAMATH_GPT_original_paint_intensity_l475_47502

theorem original_paint_intensity
  (I : ℝ) -- Original intensity of the red paint
  (f : ℝ) -- Fraction of the original paint replaced
  (new_intensity : ℝ) -- Intensity of the new paint
  (replacement_intensity : ℝ) -- Intensity of the replacement red paint
  (hf : f = 2 / 3)
  (hreplacement_intensity : replacement_intensity = 0.30)
  (hnew_intensity : new_intensity = 0.40)
  : I = 0.60 := 
sorry

end NUMINAMATH_GPT_original_paint_intensity_l475_47502


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l475_47596

theorem neither_sufficient_nor_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬ ((a > 0 ∧ b > 0) ↔ (ab < ((a + b) / 2)^2)) :=
sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l475_47596


namespace NUMINAMATH_GPT_find_least_integer_l475_47519

theorem find_least_integer (x : ℤ) : (3 * |x| - 4 < 20) → (x ≥ -7) :=
by
  sorry

end NUMINAMATH_GPT_find_least_integer_l475_47519


namespace NUMINAMATH_GPT_diameter_of_double_area_square_l475_47503

-- Define the given conditions and the problem to be solved
theorem diameter_of_double_area_square (d₁ : ℝ) (d₁_eq : d₁ = 4 * Real.sqrt 2) :
  ∃ d₂ : ℝ, d₂ = 8 :=
by
  -- Define the conditions
  let s₁ := d₁ / Real.sqrt 2
  have s₁_sq : s₁ ^ 2 = (d₁ ^ 2) / 2 := by sorry -- Pythagorean theorem

  let A₁ := s₁ ^ 2
  have A₁_eq : A₁ = 16 := by sorry -- Given diagonal, thus area

  let A₂ := 2 * A₁
  have A₂_eq : A₂ = 32 := by sorry -- Double the area

  let s₂ := Real.sqrt A₂
  have s₂_eq : s₂ = 4 * Real.sqrt 2 := by sorry -- Side length of second square

  let d₂ := s₂ * Real.sqrt 2
  have d₂_eq : d₂ = 8 := by sorry -- Diameter of the second square

  -- Prove the theorem
  existsi d₂
  exact d₂_eq

end NUMINAMATH_GPT_diameter_of_double_area_square_l475_47503


namespace NUMINAMATH_GPT_turnip_bag_weighs_l475_47583

theorem turnip_bag_weighs (bags : List ℕ) (T : ℕ)
  (h_weights : bags = [13, 15, 16, 17, 21, 24])
  (h_turnip : T ∈ bags)
  (h_carrot_onion_relation : ∃ O C: ℕ, C = 2 * O ∧ C + O = 106 - T) :
  T = 13 ∨ T = 16 := by
  sorry

end NUMINAMATH_GPT_turnip_bag_weighs_l475_47583


namespace NUMINAMATH_GPT_time_required_painting_rooms_l475_47555

-- Definitions based on the conditions
def alice_rate := 1 / 4
def bob_rate := 1 / 6
def charlie_rate := 1 / 8
def combined_rate := 13 / 24
def required_time : ℚ := 74 / 13

-- Proof problem statement
theorem time_required_painting_rooms (t : ℚ) :
  (combined_rate) * (t - 2) = 2 ↔ t = required_time :=
by
  sorry

end NUMINAMATH_GPT_time_required_painting_rooms_l475_47555


namespace NUMINAMATH_GPT_complement_A_in_U_l475_47561

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}

theorem complement_A_in_U : (U \ A) = {x | -1 <= x ∧ x <= 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_A_in_U_l475_47561


namespace NUMINAMATH_GPT_smallest_angle_opposite_smallest_side_l475_47581

theorem smallest_angle_opposite_smallest_side 
  (a b c : ℝ) 
  (h_triangle : triangle_inequality_proof)
  (h_condition : 3 * a = b + c) :
  smallest_angle_proof :=
sorry

end NUMINAMATH_GPT_smallest_angle_opposite_smallest_side_l475_47581


namespace NUMINAMATH_GPT_first_part_lent_years_l475_47557

theorem first_part_lent_years (P P1 P2 : ℝ) (rate1 rate2 : ℝ) (years2 : ℝ) (interest1 interest2 : ℝ) (t : ℝ) 
  (h1 : P = 2717)
  (h2 : P2 = 1672)
  (h3 : P1 = P - P2)
  (h4 : rate1 = 0.03)
  (h5 : rate2 = 0.05)
  (h6 : years2 = 3)
  (h7 : interest1 = P1 * rate1 * t)
  (h8 : interest2 = P2 * rate2 * years2)
  (h9 : interest1 = interest2) :
  t = 8 :=
sorry

end NUMINAMATH_GPT_first_part_lent_years_l475_47557


namespace NUMINAMATH_GPT_final_jacket_price_is_correct_l475_47582

-- Define the initial price, the discounts, and the tax rate
def initial_price : ℝ := 120
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25
def sales_tax : ℝ := 0.05

-- Calculate the final price using the given conditions
noncomputable def price_after_first_discount := initial_price * (1 - first_discount)
noncomputable def price_after_second_discount := price_after_first_discount * (1 - second_discount)
noncomputable def final_price := price_after_second_discount * (1 + sales_tax)

-- The theorem to prove
theorem final_jacket_price_is_correct : final_price = 75.60 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_final_jacket_price_is_correct_l475_47582


namespace NUMINAMATH_GPT_pencil_length_difference_l475_47571

theorem pencil_length_difference (a b : ℝ) (h1 : a = 1) (h2 : b = 4/9) :
  a - b - b = 1/9 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_pencil_length_difference_l475_47571


namespace NUMINAMATH_GPT_ball_first_less_than_25_cm_l475_47560

theorem ball_first_less_than_25_cm (n : ℕ) :
  ∀ n, (200 : ℝ) * (3 / 4) ^ n < 25 ↔ n ≥ 6 := by sorry

end NUMINAMATH_GPT_ball_first_less_than_25_cm_l475_47560


namespace NUMINAMATH_GPT_distance_missouri_to_new_york_by_car_l475_47592

variable (d_flight d_car : ℚ)

theorem distance_missouri_to_new_york_by_car :
  d_car = 1.4 * d_flight → 
  d_car = 1400 → 
  (d_car / 2 = 700) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_distance_missouri_to_new_york_by_car_l475_47592


namespace NUMINAMATH_GPT_coeff_x3_l475_47549

noncomputable def M (n : ℕ) : ℝ := (5 * (1:ℝ) - (1:ℝ)^(1/2)) ^ n
noncomputable def N (n : ℕ) : ℝ := 2 ^ n

theorem coeff_x3 (n : ℕ) (h : M n - N n = 240) : 
  (M 3) = 150 := sorry

end NUMINAMATH_GPT_coeff_x3_l475_47549


namespace NUMINAMATH_GPT_price_reduction_for_target_profit_l475_47544
-- Import the necessary libraries

-- Define the conditions
def average_sales_per_day := 70
def initial_profit_per_item := 50
def sales_increase_per_dollar_decrease := 2

-- Define the functions for sales volume increase and profit per item
def sales_volume_increase (x : ℝ) : ℝ := 2 * x
def profit_per_item (x : ℝ) : ℝ := initial_profit_per_item - x

-- Define the function for daily profit
def daily_profit (x : ℝ) : ℝ := (profit_per_item x) * (average_sales_per_day + sales_volume_increase x)

-- State the main theorem
theorem price_reduction_for_target_profit : 
  ∃ x : ℝ, daily_profit x = 3572 ∧ x = 12 :=
sorry

end NUMINAMATH_GPT_price_reduction_for_target_profit_l475_47544


namespace NUMINAMATH_GPT_solve_abs_inequality_l475_47548

theorem solve_abs_inequality (x : ℝ) :
  2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 15 ↔ (-3 ≤ x ∧ x ≤ 4 / 3) ∨ (8 / 3 ≤ x ∧ x ≤ 7) := 
sorry

end NUMINAMATH_GPT_solve_abs_inequality_l475_47548


namespace NUMINAMATH_GPT_change_received_correct_l475_47552

-- Define the conditions
def apples := 5
def cost_per_apple_cents := 80
def paid_dollars := 10

-- Convert the cost per apple to dollars
def cost_per_apple_dollars := (cost_per_apple_cents : ℚ) / 100

-- Calculate the total cost for 5 apples
def total_cost_dollars := apples * cost_per_apple_dollars

-- Calculate the change received
def change_received := paid_dollars - total_cost_dollars

-- Prove that the change received by Margie
theorem change_received_correct : change_received = 6 := by
  sorry

end NUMINAMATH_GPT_change_received_correct_l475_47552


namespace NUMINAMATH_GPT_calculate_weight_5_moles_Al2O3_l475_47536

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def molecular_weight_Al2O3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_O)
def moles_Al2O3 : ℝ := 5
def weight_5_moles_Al2O3 : ℝ := moles_Al2O3 * molecular_weight_Al2O3

theorem calculate_weight_5_moles_Al2O3 :
  weight_5_moles_Al2O3 = 509.8 :=
by sorry

end NUMINAMATH_GPT_calculate_weight_5_moles_Al2O3_l475_47536


namespace NUMINAMATH_GPT_circle_equation_l475_47559

theorem circle_equation :
  ∃ (r : ℝ), ∀ (x y : ℝ), (x - 3) ^ 2 + (y - 1) ^ 2 = r ↔ (x = 0 ∧ y = 0) → ((x - 3) ^ 2 + (y - 1) ^ 2 = 10) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l475_47559


namespace NUMINAMATH_GPT_chicken_bucket_feeds_l475_47514

theorem chicken_bucket_feeds :
  ∀ (cost_per_bucket : ℝ) (total_cost : ℝ) (total_people : ℕ),
  cost_per_bucket = 12 →
  total_cost = 72 →
  total_people = 36 →
  (total_people / (total_cost / cost_per_bucket)) = 6 :=
by
  intros cost_per_bucket total_cost total_people h1 h2 h3
  sorry

end NUMINAMATH_GPT_chicken_bucket_feeds_l475_47514


namespace NUMINAMATH_GPT_find_line_equation_through_ellipse_midpoint_l475_47575

theorem find_line_equation_through_ellipse_midpoint {A B : ℝ × ℝ} 
  (hA : (A.fst^2 / 2) + A.snd^2 = 1) 
  (hB : (B.fst^2 / 2) + B.snd^2 = 1) 
  (h_midpoint : (A.fst + B.fst) / 2 = 1 ∧ (A.snd + B.snd) / 2 = 1 / 2) : 
  ∃ k : ℝ, (k = -1) ∧ (∀ x y : ℝ, (y - 1/2 = k * (x - 1)) → 2*x + 2*y - 3 = 0) :=
sorry

end NUMINAMATH_GPT_find_line_equation_through_ellipse_midpoint_l475_47575


namespace NUMINAMATH_GPT_jars_contain_k_balls_eventually_l475_47588

theorem jars_contain_k_balls_eventually
  (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hkp : k < 2 * p + 1) :
  ∃ n : ℕ, ∃ x y : ℕ, x + y = 2 * p + 1 ∧ (x = k ∨ y = k) :=
by
  sorry

end NUMINAMATH_GPT_jars_contain_k_balls_eventually_l475_47588


namespace NUMINAMATH_GPT_fraction_irreducible_l475_47563

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end NUMINAMATH_GPT_fraction_irreducible_l475_47563


namespace NUMINAMATH_GPT_maria_workers_problem_l475_47585

-- Define the initial conditions
def initial_days : ℕ := 40
def days_passed : ℕ := 10
def fraction_completed : ℚ := 2/5
def initial_workers : ℕ := 10

-- Define the required minimum number of workers to complete the job on time
def minimum_workers_required : ℕ := 5

-- The theorem statement
theorem maria_workers_problem 
  (initial_days : ℕ)
  (days_passed : ℕ)
  (fraction_completed : ℚ)
  (initial_workers : ℕ) :
  ( ∀ (total_days remaining_days : ℕ), 
    initial_days = 40 ∧ days_passed = 10 ∧ fraction_completed = 2/5 ∧ initial_workers = 10 → 
    remaining_days = initial_days - days_passed ∧ 
    total_days = initial_days ∧ 
    fraction_completed + (remaining_days / total_days) = 1) →
  minimum_workers_required = 5 := 
sorry

end NUMINAMATH_GPT_maria_workers_problem_l475_47585


namespace NUMINAMATH_GPT_combined_population_of_New_England_and_New_York_l475_47568

noncomputable def population_of_New_England : ℕ := 2100000

noncomputable def population_of_New_York := (2/3 : ℚ) * population_of_New_England

theorem combined_population_of_New_England_and_New_York :
  population_of_New_England + population_of_New_York = 3500000 :=
by sorry

end NUMINAMATH_GPT_combined_population_of_New_England_and_New_York_l475_47568


namespace NUMINAMATH_GPT_contrapositive_lemma_l475_47590

theorem contrapositive_lemma (a : ℝ) (h : a^2 ≤ 9) : a < 4 := 
sorry

end NUMINAMATH_GPT_contrapositive_lemma_l475_47590


namespace NUMINAMATH_GPT_geom_S4_eq_2S2_iff_abs_q_eq_1_l475_47587

variable {α : Type*} [LinearOrderedField α]

-- defining the sum of first n terms of a geometric sequence
def geom_series_sum (a q : α) (n : ℕ) :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

noncomputable def S (a q : α) (n : ℕ) := geom_series_sum a q n

theorem geom_S4_eq_2S2_iff_abs_q_eq_1 
  (a q : α) : 
  S a q 4 = 2 * S a q 2 ↔ |q| = 1 :=
sorry

end NUMINAMATH_GPT_geom_S4_eq_2S2_iff_abs_q_eq_1_l475_47587


namespace NUMINAMATH_GPT_probability_of_7_successes_l475_47532

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_7_successes_l475_47532


namespace NUMINAMATH_GPT_train_initial_speed_l475_47540

theorem train_initial_speed (x : ℝ) (h : 3 * 25 * (x / V + (2 * x / 20)) = 3 * x) : V = 50 :=
  by
  sorry

end NUMINAMATH_GPT_train_initial_speed_l475_47540


namespace NUMINAMATH_GPT_smallest_possible_value_of_b_l475_47543

theorem smallest_possible_value_of_b (a b x : ℕ) (h_pos_x : 0 < x)
  (h_gcd : Nat.gcd a b = x + 7)
  (h_lcm : Nat.lcm a b = x * (x + 7))
  (h_a : a = 56)
  (h_x : x = 21) :
  b = 294 := by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_b_l475_47543


namespace NUMINAMATH_GPT_highest_student_id_in_sample_l475_47512

variable (n : ℕ) (start : ℕ) (interval : ℕ)

theorem highest_student_id_in_sample :
  start = 5 → n = 54 → interval = 9 → 6 = n / interval → start = 5 →
  5 + (interval * (6 - 1)) = 50 :=
by
  sorry

end NUMINAMATH_GPT_highest_student_id_in_sample_l475_47512


namespace NUMINAMATH_GPT_sum_of_cubes_is_nine_l475_47524

def sum_of_cubes_of_consecutive_integers (n : ℤ) : ℤ :=
  n^3 + (n + 1)^3

theorem sum_of_cubes_is_nine :
  ∃ n : ℤ, sum_of_cubes_of_consecutive_integers n = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_is_nine_l475_47524


namespace NUMINAMATH_GPT_initial_bottles_calculation_l475_47539

theorem initial_bottles_calculation (maria_bottles : ℝ) (sister_bottles : ℝ) (left_bottles : ℝ) 
  (H₁ : maria_bottles = 14.0) (H₂ : sister_bottles = 8.0) (H₃ : left_bottles = 23.0) :
  maria_bottles + sister_bottles + left_bottles = 45.0 :=
by
  sorry

end NUMINAMATH_GPT_initial_bottles_calculation_l475_47539


namespace NUMINAMATH_GPT_percentage_female_on_duty_l475_47558

-- Definition of conditions
def on_duty_officers : ℕ := 152
def female_on_duty : ℕ := on_duty_officers / 2
def total_female_officers : ℕ := 400

-- Proof goal
theorem percentage_female_on_duty : (female_on_duty * 100) / total_female_officers = 19 := by
  -- We would complete the proof here
  sorry

end NUMINAMATH_GPT_percentage_female_on_duty_l475_47558


namespace NUMINAMATH_GPT_ratio_second_to_first_l475_47529

noncomputable def ratio_of_second_to_first (x y z : ℕ) (k : ℕ) : ℕ := sorry

theorem ratio_second_to_first
    (x y z : ℕ)
    (h1 : z = 2 * y)
    (h2 : y = k * x)
    (h3 : (x + y + z) / 3 = 78)
    (h4 : x = 18)
    (k_val : k = 4):
  ratio_of_second_to_first x y z k = 4 := sorry

end NUMINAMATH_GPT_ratio_second_to_first_l475_47529


namespace NUMINAMATH_GPT_midpoint_product_l475_47566

theorem midpoint_product (x y : ℝ) :
  (∃ B : ℝ × ℝ, B = (x, y) ∧ 
  (4, 6) = ( (2 + B.1) / 2, (9 + B.2) / 2 )) → x * y = 18 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_midpoint_product_l475_47566


namespace NUMINAMATH_GPT_red_balloon_count_l475_47565

theorem red_balloon_count (total_balloons : ℕ) (green_balloons : ℕ) (red_balloons : ℕ) :
  total_balloons = 17 →
  green_balloons = 9 →
  red_balloons = total_balloons - green_balloons →
  red_balloons = 8 := by
  sorry

end NUMINAMATH_GPT_red_balloon_count_l475_47565


namespace NUMINAMATH_GPT_problem_statement_l475_47509

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (-2^x + b) / (2^(x+1) + a)

theorem problem_statement :
  (∀ (x : ℝ), f (x) 2 1 = -f (-x) 2 1) ∧
  (∀ (t : ℝ), f (t^2 - 2*t) 2 1 + f (2*t^2 - k) 2 1 < 0 → k < -1/3) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l475_47509


namespace NUMINAMATH_GPT_smallest_bottom_right_value_l475_47594

theorem smallest_bottom_right_value :
  ∃ (grid : ℕ × ℕ × ℕ → ℕ), -- grid as a function from row/column pairs to natural numbers
    (∀ i j, 1 ≤ i ∧ i ≤ 3 → 1 ≤ j ∧ j ≤ 3 → grid (i, j) ≠ 0) ∧ -- all grid values are non-zero
    (grid (1, 1) ≠ grid (1, 2) ∧ grid (1, 1) ≠ grid (1, 3) ∧ grid (1, 2) ≠ grid (1, 3) ∧
     grid (2, 1) ≠ grid (2, 2) ∧ grid (2, 1) ≠ grid (2, 3) ∧ grid (2, 2) ≠ grid (2, 3) ∧
     grid (3, 1) ≠ grid (3, 2) ∧ grid (3, 1) ≠ grid (3, 3) ∧ grid (3, 2) ≠ grid (3, 3)) ∧ -- all grid values are distinct
    (grid (1, 1) + grid (1, 2) = grid (1, 3)) ∧ 
    (grid (2, 1) + grid (2, 2) = grid (2, 3)) ∧ 
    (grid (3, 1) + grid (3, 2) = grid (3, 3)) ∧ -- row sum conditions
    (grid (1, 1) + grid (2, 1) = grid (3, 1)) ∧ 
    (grid (1, 2) + grid (2, 2) = grid (3, 2)) ∧ 
    (grid (1, 3) + grid (2, 3) = grid (3, 3)) ∧ -- column sum conditions
    (grid (3, 3) = 12) :=
by
  sorry

end NUMINAMATH_GPT_smallest_bottom_right_value_l475_47594


namespace NUMINAMATH_GPT_Panikovsky_share_l475_47533

theorem Panikovsky_share :
  ∀ (horns hooves weight : ℕ) 
    (k δ : ℝ),
    horns = 17 →
    hooves = 2 →
    weight = 1 →
    (∀ h, h = k + δ) →
    (∀ wt, wt = k + 2 * δ) →
    (20 * k + 19 * δ) / 2 = 10 * k + 9.5 * δ →
    9 * k + 7.5 * δ = (9 * (k + δ) + 2 * k) →
    ∃ (Panikov_hearts Panikov_hooves : ℕ), 
    Panikov_hearts = 9 ∧ Panikov_hooves = 2 := 
by
  intros
  sorry

end NUMINAMATH_GPT_Panikovsky_share_l475_47533


namespace NUMINAMATH_GPT_sara_ate_16_apples_l475_47546

theorem sara_ate_16_apples (S : ℕ) (h1 : ∃ (A : ℕ), A = 4 * S ∧ S + A = 80) : S = 16 :=
by
  obtain ⟨A, h2, h3⟩ := h1
  sorry

end NUMINAMATH_GPT_sara_ate_16_apples_l475_47546


namespace NUMINAMATH_GPT_smallest_n_not_prime_l475_47564

theorem smallest_n_not_prime : ∃ n, n = 4 ∧ ∀ m : ℕ, m < 4 → Prime (2 * m + 1) ∧ ¬ Prime (2 * 4 + 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_not_prime_l475_47564


namespace NUMINAMATH_GPT_greatest_value_of_x_l475_47554

theorem greatest_value_of_x : ∀ x : ℝ, 4*x^2 + 6*x + 3 = 5 → x ≤ 1/2 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_greatest_value_of_x_l475_47554


namespace NUMINAMATH_GPT_bakery_water_requirement_l475_47593

theorem bakery_water_requirement (flour water : ℕ) (total_flour : ℕ) (h : flour = 300) (w : water = 75) (t : total_flour = 900) : 
  225 = (total_flour / flour) * water :=
by
  sorry

end NUMINAMATH_GPT_bakery_water_requirement_l475_47593


namespace NUMINAMATH_GPT_expression_is_integer_l475_47518

theorem expression_is_integer (n : ℕ) : 
  (3 ^ (2 * n) / 112 - 4 ^ (2 * n) / 63 + 5 ^ (2 * n) / 144) = (k : ℤ) :=
sorry

end NUMINAMATH_GPT_expression_is_integer_l475_47518


namespace NUMINAMATH_GPT_exists_n_consecutive_numbers_l475_47556

theorem exists_n_consecutive_numbers:
  ∃ n : ℕ, n % 5 = 0 ∧ (n + 1) % 4 = 0 ∧ (n + 2) % 3 = 0 := sorry

end NUMINAMATH_GPT_exists_n_consecutive_numbers_l475_47556


namespace NUMINAMATH_GPT_range_of_f_x_lt_1_l475_47534

theorem range_of_f_x_lt_1 (x : ℝ) (f : ℝ → ℝ) (h : f x = x^3) : f x < 1 ↔ x < 1 := by
  sorry

end NUMINAMATH_GPT_range_of_f_x_lt_1_l475_47534


namespace NUMINAMATH_GPT_math_problem_l475_47520

theorem math_problem : 1003^2 - 997^2 - 1001^2 + 999^2 = 8000 := by
  sorry

end NUMINAMATH_GPT_math_problem_l475_47520


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l475_47505

theorem necessary_and_sufficient_condition (t : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
    (∀ n, S n = n^2 + 5*n + t) →
    (t = 0 ↔ (∀ n, a n = 2*n + 4 ∧ (n > 0 → a n = S n - S (n - 1)))) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l475_47505


namespace NUMINAMATH_GPT_oak_trees_initially_in_park_l475_47511

def initialOakTrees (new_oak_trees total_oak_trees_after: ℕ) : ℕ :=
  total_oak_trees_after - new_oak_trees

theorem oak_trees_initially_in_park (new_oak_trees total_oak_trees_after initial_oak_trees : ℕ) 
  (h_new_trees : new_oak_trees = 2) 
  (h_total_after : total_oak_trees_after = 11) 
  (h_correct : initial_oak_trees = 9) : 
  initialOakTrees new_oak_trees total_oak_trees_after = initial_oak_trees := 
by 
  rw [h_new_trees, h_total_after, h_correct]
  sorry

end NUMINAMATH_GPT_oak_trees_initially_in_park_l475_47511


namespace NUMINAMATH_GPT_island_knights_liars_two_people_l475_47517

def islanders_knights_and_liars (n : ℕ) : Prop :=
  ∃ (knight liar : ℕ),
    knight + liar = n ∧
    (∀ i : ℕ, 1 ≤ i → i ≤ n → 
      ((i % i = 0 → liar > 0 ∧ knight > 0) ∧ (i % i ≠ 0 → liar > 0)))

theorem island_knights_liars_two_people :
  islanders_knights_and_liars 2 :=
sorry

end NUMINAMATH_GPT_island_knights_liars_two_people_l475_47517


namespace NUMINAMATH_GPT_circle_ways_l475_47576

noncomputable def count3ConsecutiveCircles : ℕ :=
  let longSideWays := 1 + 2 + 3 + 4 + 5 + 6
  let perpendicularWays := (4 + 4 + 4 + 3 + 2 + 1) * 2
  longSideWays + perpendicularWays

theorem circle_ways : count3ConsecutiveCircles = 57 := by
  sorry

end NUMINAMATH_GPT_circle_ways_l475_47576


namespace NUMINAMATH_GPT_blood_drops_per_liter_l475_47526

def mosquito_drops : ℕ := 20
def fatal_blood_loss_liters : ℕ := 3
def mosquitoes_to_kill : ℕ := 750

theorem blood_drops_per_liter (D : ℕ) (total_drops : ℕ) : 
  (total_drops = mosquitoes_to_kill * mosquito_drops) → 
  (fatal_blood_loss_liters * D = total_drops) → 
  D = 5000 := 
  by 
    intros h1 h2
    sorry

end NUMINAMATH_GPT_blood_drops_per_liter_l475_47526


namespace NUMINAMATH_GPT_quadratic_condition_l475_47500

theorem quadratic_condition (a b c : ℝ) : (a ≠ 0) ↔ ∃ (x : ℝ), ax^2 + bx + c = 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_condition_l475_47500


namespace NUMINAMATH_GPT_initial_apples_l475_47531

theorem initial_apples (A : ℕ) 
  (H1 : A - 2 + 4 + 5 = 14) : 
  A = 7 := 
by 
  sorry

end NUMINAMATH_GPT_initial_apples_l475_47531


namespace NUMINAMATH_GPT_correct_statements_l475_47570

-- Statement B
def statementB : Prop := 
∀ x : ℝ, x < 1/2 → (∃ y : ℝ, y = 2 * x + 1 / (2 * x - 1) ∧ y = -1)

-- Statement D
def statementD : Prop :=
∃ y : ℝ, (∀ x : ℝ, y = 1 / (Real.sin x) ^ 2 + 4 / (Real.cos x) ^ 2) ∧ y = 9

-- Combined proof problem
theorem correct_statements : statementB ∧ statementD :=
sorry

end NUMINAMATH_GPT_correct_statements_l475_47570


namespace NUMINAMATH_GPT_telephone_number_problem_l475_47535

theorem telephone_number_problem
  (digits : Finset ℕ)
  (A B C D E F G H I J : ℕ)
  (h_digits : digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h_distinct : [A, B, C, D, E, F, G, H, I, J].Nodup)
  (h_ABC : A > B ∧ B > C)
  (h_DEF : D > E ∧ E > F)
  (h_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_DEF_consecutive_odd : D = E + 2 ∧ E = F + 2 ∧ (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1))
  (h_GHIJ_consecutive_even : G = H + 2 ∧ H = I + 2 ∧ I = J + 2 ∧ (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0))
  (h_sum_ABC : A + B + C = 15) :
  A = 9 :=
by
  sorry

end NUMINAMATH_GPT_telephone_number_problem_l475_47535


namespace NUMINAMATH_GPT_domain_of_sqrt_log_function_l475_47504

def domain_of_function (x : ℝ) : Prop :=
  (1 ≤ x ∧ x < 2) ∨ (2 < x ∧ x < 3)

theorem domain_of_sqrt_log_function :
  ∀ x : ℝ, (x - 1 ≥ 0) → (x - 2 ≠ 0) → (-x^2 + 2 * x + 3 > 0) →
    domain_of_function x :=
by
  intros x h1 h2 h3
  unfold domain_of_function
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_log_function_l475_47504


namespace NUMINAMATH_GPT_rectangle_minimal_area_l475_47580

theorem rectangle_minimal_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (l + w) = 120) : l * w = 675 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_rectangle_minimal_area_l475_47580


namespace NUMINAMATH_GPT_triangle_structure_twelve_rows_l475_47525

theorem triangle_structure_twelve_rows :
  let rods n := 3 * n * (n + 1) / 2
  let connectors n := n * (n + 1) / 2
  rods 12 + connectors 13 = 325 :=
by
  let rods n := 3 * n * (n + 1) / 2
  let connectors n := n * (n + 1) / 2
  sorry

end NUMINAMATH_GPT_triangle_structure_twelve_rows_l475_47525


namespace NUMINAMATH_GPT_probability_both_selected_l475_47506

theorem probability_both_selected 
  (p_jamie : ℚ) (p_tom : ℚ) 
  (h1 : p_jamie = 2/3) 
  (h2 : p_tom = 5/7) : 
  (p_jamie * p_tom = 10/21) :=
by
  sorry

end NUMINAMATH_GPT_probability_both_selected_l475_47506


namespace NUMINAMATH_GPT_height_of_old_lamp_l475_47501

theorem height_of_old_lamp (height_new_lamp : ℝ) (height_difference : ℝ) (h : height_new_lamp = 2.33) (h_diff : height_difference = 1.33) : 
  (height_new_lamp - height_difference) = 1.00 :=
by
  have height_new : height_new_lamp = 2.33 := h
  have height_diff : height_difference = 1.33 := h_diff
  sorry

end NUMINAMATH_GPT_height_of_old_lamp_l475_47501


namespace NUMINAMATH_GPT_choir_average_age_l475_47572

-- Conditions
def women_count : ℕ := 12
def men_count : ℕ := 10
def avg_age_women : ℝ := 25.0
def avg_age_men : ℝ := 40.0

-- Expected Answer
def expected_avg_age : ℝ := 31.82

-- Proof Statement
theorem choir_average_age :
  ((women_count * avg_age_women) + (men_count * avg_age_men)) / (women_count + men_count) = expected_avg_age :=
by
  sorry

end NUMINAMATH_GPT_choir_average_age_l475_47572


namespace NUMINAMATH_GPT_find_b_l475_47538

theorem find_b (b : ℤ) :
  (∃ x : ℤ, x^2 + b * x - 36 = 0 ∧ x = -9) → b = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l475_47538


namespace NUMINAMATH_GPT_max_books_borrowed_l475_47573

theorem max_books_borrowed (total_students books_per_student : ℕ) (students_with_no_books: ℕ) (students_with_one_book students_with_two_books: ℕ) (rest_at_least_three_books students : ℕ) :
  total_students = 20 →
  books_per_student = 2 →
  students_with_no_books = 2 →
  students_with_one_book = 8 →
  students_with_two_books = 3 →
  rest_at_least_three_books = total_students - (students_with_no_books + students_with_one_book + students_with_two_books) →
  (students_with_no_books * 0 + students_with_one_book * 1 + students_with_two_books * 2 + students * books_per_student = total_students * books_per_student) →
  (students * 3 + some_student_max = 26) →
  some_student_max ≥ 8 :=
by
  introv h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_max_books_borrowed_l475_47573


namespace NUMINAMATH_GPT_function_property_l475_47553

def y (x : ℝ) : ℝ := x - 2

theorem function_property : y 1 = -1 :=
by
  -- place for proof
  sorry

end NUMINAMATH_GPT_function_property_l475_47553


namespace NUMINAMATH_GPT_sprinkles_remaining_l475_47569

theorem sprinkles_remaining (initial_cans : ℕ) (remaining_cans : ℕ) 
  (h1 : initial_cans = 12) 
  (h2 : remaining_cans = (initial_cans / 2) - 3) : 
  remaining_cans = 3 := 
by
  sorry

end NUMINAMATH_GPT_sprinkles_remaining_l475_47569


namespace NUMINAMATH_GPT_find_f_7_5_l475_47530

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom f_range_0_1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7_5 : f 7.5 = -0.5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_find_f_7_5_l475_47530


namespace NUMINAMATH_GPT_count_negative_values_correct_l475_47522

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end NUMINAMATH_GPT_count_negative_values_correct_l475_47522


namespace NUMINAMATH_GPT_custom_mul_expansion_l475_47598

variable {a b x y : ℝ}

def custom_mul (a b : ℝ) : ℝ := (a - b)^2

theorem custom_mul_expansion (x y : ℝ) : custom_mul (x^2) (y^2) = (x + y)^2 * (x - y)^2 := by
  sorry

end NUMINAMATH_GPT_custom_mul_expansion_l475_47598


namespace NUMINAMATH_GPT_multiples_of_7_between_50_and_200_l475_47586

theorem multiples_of_7_between_50_and_200 : 
  ∃ n, n = 21 ∧ ∀ k, (k ≥ 50 ∧ k ≤ 200) ↔ ∃ m, k = 7 * m := sorry

end NUMINAMATH_GPT_multiples_of_7_between_50_and_200_l475_47586


namespace NUMINAMATH_GPT_teamA_teamB_repair_eq_l475_47527

-- conditions
def teamADailyRepair (x : ℕ) := x -- represent Team A repairing x km/day
def teamBDailyRepair (x : ℕ) := x + 3 -- represent Team B repairing x + 3 km/day
def timeTaken (distance rate: ℕ) := distance / rate -- time = distance / rate

-- Proof problem statement
theorem teamA_teamB_repair_eq (x : ℕ) (hx : x > 0) (hx_plus_3 : x + 3 > 0) :
  timeTaken 6 (teamADailyRepair x) = timeTaken 8 (teamBDailyRepair x) → (6 / x = 8 / (x + 3)) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_teamA_teamB_repair_eq_l475_47527


namespace NUMINAMATH_GPT_coffee_shop_lattes_l475_47578

theorem coffee_shop_lattes (T : ℕ) (L : ℕ) (hT : T = 6) (hL : L = 4 * T + 8) : L = 32 :=
by
  sorry

end NUMINAMATH_GPT_coffee_shop_lattes_l475_47578


namespace NUMINAMATH_GPT_corey_needs_more_golf_balls_l475_47547

-- Defining the constants based on the conditions
def goal : ℕ := 48
def found_on_saturday : ℕ := 16
def found_on_sunday : ℕ := 18

-- The number of golf balls Corey has found over the weekend
def total_found : ℕ := found_on_saturday + found_on_sunday

-- The number of golf balls Corey still needs to find to reach his goal
def remaining : ℕ := goal - total_found

-- The desired theorem statement
theorem corey_needs_more_golf_balls : remaining = 14 := 
by 
  sorry

end NUMINAMATH_GPT_corey_needs_more_golf_balls_l475_47547


namespace NUMINAMATH_GPT_number_of_customers_l475_47515

-- Definitions based on conditions
def popularity (p : ℕ) (c w : ℕ) (k : ℝ) : Prop :=
  p = k * (w / c)

-- Given values
def given_values : Prop :=
  ∃ k : ℝ, popularity 15 500 1000 k

-- Problem statement
theorem number_of_customers:
  given_values →
  popularity 15 600 1200 7.5 :=
by
  intro h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_number_of_customers_l475_47515


namespace NUMINAMATH_GPT_square_diagonal_l475_47551

theorem square_diagonal (A : ℝ) (s : ℝ) (d : ℝ) (hA : A = 338) (hs : s^2 = A) (hd : d^2 = 2 * s^2) : d = 26 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_square_diagonal_l475_47551


namespace NUMINAMATH_GPT_fraction_simplification_l475_47516

theorem fraction_simplification (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) : 
  (x^2 + x) / (x^2 - 1) = x / (x - 1) :=
by
  -- Hint of expected development environment setting
  sorry

end NUMINAMATH_GPT_fraction_simplification_l475_47516


namespace NUMINAMATH_GPT_ratio_of_newspapers_l475_47528

theorem ratio_of_newspapers (C L : ℕ) (h1 : C = 42) (h2 : L = C + 23) : C / (C + 23) = 42 / 65 := by
  sorry

end NUMINAMATH_GPT_ratio_of_newspapers_l475_47528


namespace NUMINAMATH_GPT_price_of_one_liter_l475_47510

theorem price_of_one_liter
  (total_cost : ℝ) (num_bottles : ℝ) (liters_per_bottle : ℝ)
  (H : total_cost = 12 ∧ num_bottles = 6 ∧ liters_per_bottle = 2) :
  total_cost / (num_bottles * liters_per_bottle) = 1 :=
by
  sorry

end NUMINAMATH_GPT_price_of_one_liter_l475_47510


namespace NUMINAMATH_GPT_find_B_l475_47550

variables {a b c A B C : ℝ}

-- Conditions
axiom given_condition_1 : (c - b) / (c - a) = (Real.sin A) / (Real.sin C + Real.sin B)

-- Law of Sines
axiom law_of_sines_1 : (c - b) / (c - a) = a / (c + b)

-- Law of Cosines
axiom law_of_cosines_1 : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c)

-- Target
theorem find_B : B = Real.pi / 3 := 
sorry

end NUMINAMATH_GPT_find_B_l475_47550


namespace NUMINAMATH_GPT_darkCubeValidPositions_l475_47537

-- Conditions:
-- 1. The structure is made up of twelve identical cubes.
-- 2. The dark cube must be relocated to a position where the surface area remains unchanged.
-- 3. The cubes must touch each other with their entire faces.
-- 4. The positions of the light cubes cannot be changed.

-- Let's define the structure and the conditions in Lean.

structure Cube :=
  (id : ℕ) -- unique identifier for each cube

structure Position :=
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)

structure Configuration :=
  (cubes : List Cube)
  (positions : Cube → Position)

def initialCondition (config : Configuration) : Prop :=
  config.cubes.length = 12

def surfaceAreaUnchanged (config : Configuration) (darkCube : Cube) (newPos : Position) : Prop :=
  sorry -- This predicate should capture the logic that the surface area remains unchanged

def validPositions (config : Configuration) (darkCube : Cube) : List Position :=
  sorry -- This function should return the list of valid positions for the dark cube

-- Main theorem: The number of valid positions for the dark cube to maintain the surface area.
theorem darkCubeValidPositions (config : Configuration) (darkCube : Cube) :
    initialCondition config →
    (validPositions config darkCube).length = 3 :=
  by
  sorry

end NUMINAMATH_GPT_darkCubeValidPositions_l475_47537


namespace NUMINAMATH_GPT_martin_initial_spending_l475_47542

theorem martin_initial_spending :
  ∃ (x : ℝ), 
    ∀ (a b : ℝ), 
      a = x - 100 →
      b = a - 0.20 * a →
      x - b = 280 →
      x = 1000 :=
by
  sorry

end NUMINAMATH_GPT_martin_initial_spending_l475_47542


namespace NUMINAMATH_GPT_profit_percentage_l475_47513

theorem profit_percentage (initial_cost_per_pound : ℝ) (ruined_percent : ℝ) (selling_price_per_pound : ℝ) (desired_profit_percent : ℝ) : 
  initial_cost_per_pound = 0.80 ∧ ruined_percent = 0.10 ∧ selling_price_per_pound = 0.96 → desired_profit_percent = 8 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_l475_47513


namespace NUMINAMATH_GPT_average_hours_per_day_l475_47523

theorem average_hours_per_day (h : ℝ) :
  (3 * h * 12 + 2 * h * 9 = 108) → h = 2 :=
by 
  intro h_condition
  sorry

end NUMINAMATH_GPT_average_hours_per_day_l475_47523


namespace NUMINAMATH_GPT_arithmetic_sequence_conditions_l475_47507

open Nat

theorem arithmetic_sequence_conditions (S : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  d < 0 ∧ S 11 > 0 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_conditions_l475_47507


namespace NUMINAMATH_GPT_negation_of_universal_prop_l475_47579

variable (P : ∀ x : ℝ, Real.cos x ≤ 1)

theorem negation_of_universal_prop : ∃ x₀ : ℝ, Real.cos x₀ > 1 :=
sorry

end NUMINAMATH_GPT_negation_of_universal_prop_l475_47579


namespace NUMINAMATH_GPT_sine_theorem_l475_47562

theorem sine_theorem (a b c α β γ : ℝ) 
  (h1 : a / Real.sin α = b / Real.sin β) 
  (h2 : b / Real.sin β = c / Real.sin γ) 
  (h3 : α + β + γ = Real.pi) :
  a = b * Real.cos γ + c * Real.cos β ∧
  b = c * Real.cos α + a * Real.cos γ ∧
  c = a * Real.cos β + b * Real.cos α :=
by
  sorry

end NUMINAMATH_GPT_sine_theorem_l475_47562


namespace NUMINAMATH_GPT_largest_value_of_n_l475_47545

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end NUMINAMATH_GPT_largest_value_of_n_l475_47545


namespace NUMINAMATH_GPT_solve_inequality_l475_47541

theorem solve_inequality (a x : ℝ) : 
  (a = 0 ∧ x ≤ -1) ∨ 
  (a > 0 ∧ (x ≥ 2 / a ∨ x ≤ -1)) ∨ 
  (-2 < a ∧ a < 0 ∧ 2 / a ≤ x ∧ x ≤ -1) ∨ 
  (a = -2 ∧ x = -1) ∨
  (a < -2 ∧ -1 ≤ x ∧ x ≤ 2 / a) ↔ 
  a * x ^ 2 + (a - 2) * x - 2 ≥ 0 := 
sorry

end NUMINAMATH_GPT_solve_inequality_l475_47541


namespace NUMINAMATH_GPT_basketball_games_played_l475_47521

theorem basketball_games_played (G : ℕ) (H1 : 35 ≤ G) (H2 : 25 ≥ 0) (H3 : 64 = 100 * (48 / (G + 25))):
  G = 50 :=
sorry

end NUMINAMATH_GPT_basketball_games_played_l475_47521


namespace NUMINAMATH_GPT_fraction_oj_is_5_over_13_l475_47584

def capacity_first_pitcher : ℕ := 800
def capacity_second_pitcher : ℕ := 500
def fraction_oj_first_pitcher : ℚ := 1 / 4
def fraction_oj_second_pitcher : ℚ := 3 / 5

def amount_oj_first_pitcher : ℚ := capacity_first_pitcher * fraction_oj_first_pitcher
def amount_oj_second_pitcher : ℚ := capacity_second_pitcher * fraction_oj_second_pitcher

def total_amount_oj : ℚ := amount_oj_first_pitcher + amount_oj_second_pitcher
def total_capacity : ℚ := capacity_first_pitcher + capacity_second_pitcher

def fraction_oj_large_container : ℚ := total_amount_oj / total_capacity

theorem fraction_oj_is_5_over_13 : fraction_oj_large_container = (5 / 13) := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_fraction_oj_is_5_over_13_l475_47584


namespace NUMINAMATH_GPT_total_cost_of_panels_l475_47595

theorem total_cost_of_panels
    (sidewall_width : ℝ)
    (sidewall_height : ℝ)
    (triangle_base : ℝ)
    (triangle_height : ℝ)
    (panel_width : ℝ)
    (panel_height : ℝ)
    (panel_cost : ℝ)
    (total_cost : ℝ)
    (h_sidewall : sidewall_width = 9)
    (h_sidewall_height : sidewall_height = 7)
    (h_triangle_base : triangle_base = 9)
    (h_triangle_height : triangle_height = 6)
    (h_panel_width : panel_width = 10)
    (h_panel_height : panel_height = 15)
    (h_panel_cost : panel_cost = 32)
    (h_total_cost : total_cost = 32) :
    total_cost = panel_cost :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_panels_l475_47595


namespace NUMINAMATH_GPT_Robert_diff_C_l475_47589

/- Define the conditions as hypotheses -/
variables (C : ℕ) -- Assuming the number of photos Claire has taken as a natural number.

-- Lisa has taken 3 times as many photos as Claire.
def Lisa_photos := 3 * C

-- Robert has taken the same number of photos as Lisa.
def Robert_photos := Lisa_photos C -- which will be 3 * C

-- Proof of the difference.
theorem Robert_diff_C : (Robert_photos C) - C = 2 * C :=
by
  sorry

end NUMINAMATH_GPT_Robert_diff_C_l475_47589
