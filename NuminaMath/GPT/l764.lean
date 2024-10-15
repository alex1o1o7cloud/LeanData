import Mathlib

namespace NUMINAMATH_GPT_sufficient_condition_l764_76467

variable {α : Type*} (A B : Set α)

theorem sufficient_condition (h : A ⊆ B) (x : α) : x ∈ A → x ∈ B :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_l764_76467


namespace NUMINAMATH_GPT_natalie_blueberry_bushes_l764_76492

-- Definitions of the conditions
def bushes_yield_containers (bushes containers : ℕ) : Prop :=
  containers = bushes * 7

def containers_exchange_zucchinis (containers zucchinis : ℕ) : Prop :=
  zucchinis = containers * 3 / 7

-- Theorem statement
theorem natalie_blueberry_bushes (zucchinis_needed : ℕ) (zucchinis_per_trade containers_per_trade bushes_per_container : ℕ) 
  (h1 : zucchinis_per_trade = 3) (h2 : containers_per_trade = 7) (h3 : bushes_per_container = 7) 
  (h4 : zucchinis_needed = 63) : 
  ∃ bushes_needed : ℕ, bushes_needed = 21 := 
by
  sorry

end NUMINAMATH_GPT_natalie_blueberry_bushes_l764_76492


namespace NUMINAMATH_GPT_minimum_value_of_2x_plus_y_l764_76441

theorem minimum_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) : 2 * x + y ≥ 12 :=
  sorry

end NUMINAMATH_GPT_minimum_value_of_2x_plus_y_l764_76441


namespace NUMINAMATH_GPT_fraction_difference_l764_76431

theorem fraction_difference (a b : ℝ) (h : a - b = 2 * a * b) : (1 / a - 1 / b) = -2 := 
by
  sorry

end NUMINAMATH_GPT_fraction_difference_l764_76431


namespace NUMINAMATH_GPT_find_a_from_perpendicular_lines_l764_76479

theorem find_a_from_perpendicular_lines (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_from_perpendicular_lines_l764_76479


namespace NUMINAMATH_GPT_find_c_l764_76434

theorem find_c (a b c : ℝ) (h : 1/a + 1/b = 1/c) : c = (a * b) / (a + b) := 
by
  sorry

end NUMINAMATH_GPT_find_c_l764_76434


namespace NUMINAMATH_GPT_afternoon_to_morning_ratio_l764_76411

theorem afternoon_to_morning_ratio (total_kg : ℕ) (afternoon_kg : ℕ) (morning_kg : ℕ) 
  (h1 : total_kg = 390) (h2 : afternoon_kg = 260) (h3 : morning_kg = total_kg - afternoon_kg) :
  afternoon_kg / morning_kg = 2 :=
sorry

end NUMINAMATH_GPT_afternoon_to_morning_ratio_l764_76411


namespace NUMINAMATH_GPT_set_properties_proof_l764_76457

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := Icc (-2 : ℝ) 2)
variable (N : Set ℝ := Iic (1 : ℝ))

theorem set_properties_proof :
  (M ∪ N = Iic (2 : ℝ)) ∧
  (M ∩ N = Icc (-2 : ℝ) 1) ∧
  (U \ N = Ioi (1 : ℝ)) := by
  sorry

end NUMINAMATH_GPT_set_properties_proof_l764_76457


namespace NUMINAMATH_GPT_digit_matching_equalities_l764_76418

theorem digit_matching_equalities :
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ 99 → 0 ≤ b ∧ b ≤ 99 →
    ((a = 98 ∧ b = 1 ∧ (98 + 1)^2 = 100*98 + 1) ∨
     (a = 20 ∧ b = 25 ∧ (20 + 25)^2 = 100*20 + 25)) :=
by
  intros a b ha hb
  sorry

end NUMINAMATH_GPT_digit_matching_equalities_l764_76418


namespace NUMINAMATH_GPT_safe_dishes_count_l764_76456

theorem safe_dishes_count (total_dishes vegan_dishes vegan_with_nuts : ℕ) 
  (h1 : vegan_dishes = total_dishes / 3) 
  (h2 : vegan_with_nuts = 4) 
  (h3 : vegan_dishes = 6) : vegan_dishes - vegan_with_nuts = 2 :=
by
  sorry

end NUMINAMATH_GPT_safe_dishes_count_l764_76456


namespace NUMINAMATH_GPT_two_a_minus_b_l764_76435

variables (a b : ℝ × ℝ)
variables (m : ℝ)

def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

theorem two_a_minus_b 
  (ha : a = (1, -2))
  (hb : b = (m, 4))
  (h_parallel : parallel a b) :
  2 • a - b = (4, -8) :=
sorry

end NUMINAMATH_GPT_two_a_minus_b_l764_76435


namespace NUMINAMATH_GPT_lattice_intersections_l764_76412

theorem lattice_intersections (squares : ℕ) (circles : ℕ) 
        (line_segment : ℤ × ℤ → ℤ × ℤ) 
        (radius : ℚ) (side_length : ℚ) : 
        line_segment (0, 0) = (1009, 437) → 
        radius = 1/8 → side_length = 1/4 → 
        (squares + circles = 430) :=
by
  sorry

end NUMINAMATH_GPT_lattice_intersections_l764_76412


namespace NUMINAMATH_GPT_Sandy_total_marks_l764_76494

theorem Sandy_total_marks
  (correct_marks_per_sum : ℤ)
  (incorrect_marks_per_sum : ℤ)
  (total_sums : ℕ)
  (correct_sums : ℕ)
  (incorrect_sums : ℕ)
  (total_marks : ℤ) :
  correct_marks_per_sum = 3 →
  incorrect_marks_per_sum = -2 →
  total_sums = 30 →
  correct_sums = 24 →
  incorrect_sums = total_sums - correct_sums →
  total_marks = correct_marks_per_sum * correct_sums + incorrect_marks_per_sum * incorrect_sums →
  total_marks = 60 :=
by
  sorry

end NUMINAMATH_GPT_Sandy_total_marks_l764_76494


namespace NUMINAMATH_GPT_train_pass_bridge_in_50_seconds_l764_76484

noncomputable def time_to_pass_bridge (length_train length_bridge : ℕ) (speed_kmh : ℕ) : ℕ :=
  let total_distance := length_train + length_bridge
  let speed_ms := (speed_kmh * 1000) / 3600
  total_distance / speed_ms

theorem train_pass_bridge_in_50_seconds :
  time_to_pass_bridge 485 140 45 = 50 :=
by
  sorry

end NUMINAMATH_GPT_train_pass_bridge_in_50_seconds_l764_76484


namespace NUMINAMATH_GPT_simplify_complex_expression_l764_76460

theorem simplify_complex_expression (i : ℂ) (h_i : i * i = -1) : 
  (11 - 3 * i) / (1 + 2 * i) = 3 - 5 * i :=
sorry

end NUMINAMATH_GPT_simplify_complex_expression_l764_76460


namespace NUMINAMATH_GPT_boat_travel_distance_per_day_l764_76493

-- Definitions from conditions
def men : ℕ := 25
def water_daily_per_man : ℚ := 1/2
def travel_distance : ℕ := 4000
def total_water : ℕ := 250

-- Main theorem
theorem boat_travel_distance_per_day : 
  ∀ (men : ℕ) (water_daily_per_man : ℚ) (travel_distance : ℕ) (total_water : ℕ), 
  men = 25 ∧ water_daily_per_man = 1/2 ∧ travel_distance = 4000 ∧ total_water = 250 ->
  travel_distance / (total_water / (men * water_daily_per_man)) = 200 :=
by
  sorry

end NUMINAMATH_GPT_boat_travel_distance_per_day_l764_76493


namespace NUMINAMATH_GPT_isolate_urea_decomposing_bacteria_valid_option_l764_76473

variable (KH2PO4 Na2HPO4 MgSO4_7H2O urea glucose agar water : Type)
variable (urea_decomposing_bacteria : Type)
variable (CarbonSource : Type → Prop)
variable (NitrogenSource : Type → Prop)
variable (InorganicSalt : Type → Prop)
variable (bacteria_can_synthesize_urease : urea_decomposing_bacteria → Prop)

axiom KH2PO4_is_inorganic_salt : InorganicSalt KH2PO4
axiom Na2HPO4_is_inorganic_salt : InorganicSalt Na2HPO4
axiom MgSO4_7H2O_is_inorganic_salt : InorganicSalt MgSO4_7H2O
axiom urea_is_nitrogen_source : NitrogenSource urea

theorem isolate_urea_decomposing_bacteria_valid_option :
  (InorganicSalt KH2PO4) ∧
  (InorganicSalt Na2HPO4) ∧
  (InorganicSalt MgSO4_7H2O) ∧
  (NitrogenSource urea) ∧
  (CarbonSource glucose) → (∃ bacteria : urea_decomposing_bacteria, bacteria_can_synthesize_urease bacteria) := sorry

end NUMINAMATH_GPT_isolate_urea_decomposing_bacteria_valid_option_l764_76473


namespace NUMINAMATH_GPT_least_number_of_coins_l764_76436

theorem least_number_of_coins : ∃ (n : ℕ), 
  (n % 6 = 3) ∧ 
  (n % 4 = 1) ∧ 
  (n % 7 = 2) ∧ 
  (∀ m : ℕ, (m % 6 = 3) ∧ (m % 4 = 1) ∧ (m % 7 = 2) → n ≤ m) :=
by
  exists 9
  simp
  sorry

end NUMINAMATH_GPT_least_number_of_coins_l764_76436


namespace NUMINAMATH_GPT_solve_linear_system_l764_76491

theorem solve_linear_system :
  ∃ (x y : ℝ), (x + 3 * y = -1) ∧ (2 * x + y = 3) ∧ (x = 2) ∧ (y = -1) :=
  sorry

end NUMINAMATH_GPT_solve_linear_system_l764_76491


namespace NUMINAMATH_GPT_check_inequality_l764_76400

theorem check_inequality : 1.7^0.3 > 0.9^3.1 :=
sorry

end NUMINAMATH_GPT_check_inequality_l764_76400


namespace NUMINAMATH_GPT_unit_price_solution_purchase_plan_and_costs_l764_76442

-- Definitions based on the conditions (Note: numbers and relationships purely)
def unit_prices (x y : ℕ) : Prop :=
  3 * x + 2 * y = 60 ∧ x + 3 * y = 55

def prize_purchase_conditions (m n : ℕ) : Prop :=
  m + n = 100 ∧ 10 * m + 15 * n ≤ 1160 ∧ m ≤ 3 * n

-- Proving that the unit prices found match the given constraints
theorem unit_price_solution : ∃ x y : ℕ, unit_prices x y := by
  sorry

-- Proving the number of purchasing plans and minimum cost
theorem purchase_plan_and_costs : 
  (∃ (num_plans : ℕ) (min_cost : ℕ), 
    num_plans = 8 ∧ min_cost = 1125 ∧ 
    ∀ m n : ℕ, prize_purchase_conditions m n → 
      ((68 ≤ m ∧ m ≤ 75) →
      10 * m + 15 * (100 - m) = min_cost)) := by
  sorry

end NUMINAMATH_GPT_unit_price_solution_purchase_plan_and_costs_l764_76442


namespace NUMINAMATH_GPT_equation_of_line_containing_BC_l764_76465

theorem equation_of_line_containing_BC (A B C : ℝ × ℝ)
  (hA : A = (1, 2))
  (altitude_from_CA : ∀ x y : ℝ, 2 * x - 3 * y + 1 = 0)
  (altitude_from_BA : ∀ x y : ℝ, x + y = 1)
  (eq_BC : ∃ k b : ℝ, ∀ x y : ℝ, y = k * x + b) :
  ∃ k b : ℝ, (∀ x y : ℝ, y = k * x + b) ∧ 2 * x + 3 * y + 7 = 0 :=
sorry

end NUMINAMATH_GPT_equation_of_line_containing_BC_l764_76465


namespace NUMINAMATH_GPT_four_numbers_divisible_by_2310_in_4_digit_range_l764_76498

/--
There exist exactly 4 numbers within the range of 1000 to 9999 that are divisible by 2310.
-/
theorem four_numbers_divisible_by_2310_in_4_digit_range :
  ∃ n₁ n₂ n₃ n₄,
    1000 ≤ n₁ ∧ n₁ ≤ 9999 ∧ n₁ % 2310 = 0 ∧
    1000 ≤ n₂ ∧ n₂ ≤ 9999 ∧ n₂ % 2310 = 0 ∧ n₁ < n₂ ∧
    1000 ≤ n₃ ∧ n₃ ≤ 9999 ∧ n₃ % 2310 = 0 ∧ n₂ < n₃ ∧
    1000 ≤ n₄ ∧ n₄ ≤ 9999 ∧ n₄ % 2310 = 0 ∧ n₃ < n₄ ∧
    ∀ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 2310 = 0 → (n = n₁ ∨ n = n₂ ∨ n = n₃ ∨ n = n₄) :=
by
  sorry

end NUMINAMATH_GPT_four_numbers_divisible_by_2310_in_4_digit_range_l764_76498


namespace NUMINAMATH_GPT_fraction_simplify_l764_76416

theorem fraction_simplify:
  (1/5 + 1/7) / (3/8 - 1/9) = 864 / 665 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplify_l764_76416


namespace NUMINAMATH_GPT_smallest_digit_not_found_in_units_place_of_odd_number_l764_76487

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end NUMINAMATH_GPT_smallest_digit_not_found_in_units_place_of_odd_number_l764_76487


namespace NUMINAMATH_GPT_expenditure_ratio_l764_76458

def ratio_of_incomes (I1 I2 : ℕ) : Prop := I1 / I2 = 5 / 4
def savings (I E : ℕ) : ℕ := I - E
def ratio_of_expenditures (E1 E2 : ℕ) : Prop := E1 / E2 = 3 / 2

theorem expenditure_ratio (I1 I2 E1 E2 : ℕ) 
  (I1_income : I1 = 5500)
  (income_ratio : ratio_of_incomes I1 I2)
  (savings_equal : savings I1 E1 = 2200 ∧ savings I2 E2 = 2200)
  : ratio_of_expenditures E1 E2 :=
by 
  sorry

end NUMINAMATH_GPT_expenditure_ratio_l764_76458


namespace NUMINAMATH_GPT_tan_150_degrees_l764_76406

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_150_degrees_l764_76406


namespace NUMINAMATH_GPT_sum_of_numbers_eq_answer_l764_76480

open Real

noncomputable def sum_of_numbers (x y : ℝ) : ℝ := x + y

theorem sum_of_numbers_eq_answer (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) (h4 : (1 / x) = 3 * (1 / y)) :
  sum_of_numbers x y = 16 * Real.sqrt 3 / 3 := 
sorry

end NUMINAMATH_GPT_sum_of_numbers_eq_answer_l764_76480


namespace NUMINAMATH_GPT_seeds_per_flowerbed_l764_76437

theorem seeds_per_flowerbed (total_seeds : ℕ) (flowerbeds : ℕ) (seeds_per_bed : ℕ) 
  (h1 : total_seeds = 45) (h2 : flowerbeds = 9) 
  (h3 : total_seeds = flowerbeds * seeds_per_bed) : seeds_per_bed = 5 :=
by sorry

end NUMINAMATH_GPT_seeds_per_flowerbed_l764_76437


namespace NUMINAMATH_GPT_audrey_dreaming_fraction_l764_76450

theorem audrey_dreaming_fraction
  (total_asleep_time : ℕ) 
  (not_dreaming_time : ℕ)
  (dreaming_time : ℕ)
  (fraction_dreaming : ℚ)
  (h_total_asleep : total_asleep_time = 10)
  (h_not_dreaming : not_dreaming_time = 6)
  (h_dreaming : dreaming_time = total_asleep_time - not_dreaming_time)
  (h_fraction : fraction_dreaming = dreaming_time / total_asleep_time) :
  fraction_dreaming = 2 / 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_audrey_dreaming_fraction_l764_76450


namespace NUMINAMATH_GPT_solve_problem_1_solve_problem_2_l764_76439

-- Problem statement 1: Prove that the solutions to x(x-2) = x-2 are x = 1 and x = 2.
theorem solve_problem_1 (x : ℝ) : (x * (x - 2) = x - 2) ↔ (x = 1 ∨ x = 2) :=
  sorry

-- Problem statement 2: Prove that the solutions to 2x^2 + 3x - 5 = 0 are x = 1 and x = -5/2.
theorem solve_problem_2 (x : ℝ) : (2 * x^2 + 3 * x - 5 = 0) ↔ (x = 1 ∨ x = -5 / 2) :=
  sorry

end NUMINAMATH_GPT_solve_problem_1_solve_problem_2_l764_76439


namespace NUMINAMATH_GPT_sum_of_pairwise_rel_prime_integers_l764_76464

def is_pairwise_rel_prime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

theorem sum_of_pairwise_rel_prime_integers 
  (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h_prod : a * b * c = 343000) (h_rel_prime : is_pairwise_rel_prime a b c) : 
  a + b + c = 476 := 
sorry

end NUMINAMATH_GPT_sum_of_pairwise_rel_prime_integers_l764_76464


namespace NUMINAMATH_GPT_students_in_line_l764_76495

theorem students_in_line (T N : ℕ) (hT : T = 1) (h_btw : N = T + 4) (h_behind: ∃ k, k = 8) : T + (N - T) + 1 + 8 = 13 :=
by
  sorry

end NUMINAMATH_GPT_students_in_line_l764_76495


namespace NUMINAMATH_GPT_anna_baked_60_cupcakes_l764_76438

variable (C : ℕ)
variable (h1 : (1/5 : ℚ) * C - 3 = 9)

theorem anna_baked_60_cupcakes (h1 : (1/5 : ℚ) * C - 3 = 9) : C = 60 :=
sorry

end NUMINAMATH_GPT_anna_baked_60_cupcakes_l764_76438


namespace NUMINAMATH_GPT_sum_of_primes_l764_76414

theorem sum_of_primes (p1 p2 p3 : ℕ) (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3) 
    (h : p1 * p2 * p3 = 31 * (p1 + p2 + p3)) :
    p1 + p2 + p3 = 51 := by
  sorry

end NUMINAMATH_GPT_sum_of_primes_l764_76414


namespace NUMINAMATH_GPT_rectangle_area_l764_76432

def length : ℝ := 2
def width : ℝ := 4
def area := length * width

theorem rectangle_area : area = 8 := 
by
  -- Proof can be written here
  sorry

end NUMINAMATH_GPT_rectangle_area_l764_76432


namespace NUMINAMATH_GPT_find_k_l764_76476

theorem find_k (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ m n, a (m + n) = a m * a n) (hk : a (k + 1) = 1024) : k = 9 := 
sorry

end NUMINAMATH_GPT_find_k_l764_76476


namespace NUMINAMATH_GPT_total_cost_full_units_l764_76444

def total_units : Nat := 12
def cost_1_bedroom : Nat := 360
def cost_2_bedroom : Nat := 450
def num_2_bedroom : Nat := 7
def num_1_bedroom : Nat := total_units - num_2_bedroom

def total_cost : Nat := (num_1_bedroom * cost_1_bedroom) + (num_2_bedroom * cost_2_bedroom)

theorem total_cost_full_units : total_cost = 4950 := by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_total_cost_full_units_l764_76444


namespace NUMINAMATH_GPT_grid_area_l764_76426

theorem grid_area :
  let B := 10   -- Number of boundary points
  let I := 12   -- Number of interior points
  I + B / 2 - 1 = 16 :=
by
  sorry

end NUMINAMATH_GPT_grid_area_l764_76426


namespace NUMINAMATH_GPT_sequence_problems_l764_76485
open Nat

-- Define the arithmetic sequence conditions
def arith_seq_condition_1 (a : ℕ → ℤ) : Prop :=
  a 2 + a 7 = -23

def arith_seq_condition_2 (a : ℕ → ℤ) : Prop :=
  a 3 + a 8 = -29

-- Define the geometric sequence condition
def geom_seq_condition (a b : ℕ → ℤ) (c : ℤ) : Prop :=
  ∀ n, a n + b n = c^(n - 1)

-- Define the arithmetic sequence formula
def arith_seq_formula (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = -3 * n + 2

-- Define the sum of the first n terms of the sequence b_n
def sum_b_n (b : ℕ → ℤ) (S_n : ℕ → ℤ) (c : ℤ) : Prop :=
  (c = 1 → ∀ n, S_n n = (3 * n^2 + n) / 2) ∧
  (c ≠ 1 → ∀ n, S_n n = (n * (3 * n - 1)) / 2 + ((1 - c^n) / (1 - c)))

-- Define the main theorem
theorem sequence_problems (a b : ℕ → ℤ) (c : ℤ) (S_n : ℕ → ℤ) :
  arith_seq_condition_1 a →
  arith_seq_condition_2 a →
  geom_seq_condition a b c →
  arith_seq_formula a ∧ sum_b_n b S_n c :=
by
  -- Proofs for the conditions to the formula
  sorry

end NUMINAMATH_GPT_sequence_problems_l764_76485


namespace NUMINAMATH_GPT_roots_expression_value_l764_76486

theorem roots_expression_value (x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : x1 * x2 = 2) :
  2 * x1 - x1 * x2 + 2 * x2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_roots_expression_value_l764_76486


namespace NUMINAMATH_GPT_triangle_angles_sum_l764_76478

theorem triangle_angles_sum (x : ℝ) (h : 40 + 3 * x + (x + 10) = 180) : x = 32.5 := by
  sorry

end NUMINAMATH_GPT_triangle_angles_sum_l764_76478


namespace NUMINAMATH_GPT_find_positive_integers_l764_76461

theorem find_positive_integers (n : ℕ) (h_pos : n > 0) : 
  (∃ d : ℕ, ∀ k : ℕ, 6^n + 1 = d * (10^k - 1) / 9 → d = 7) → 
  n = 1 ∨ n = 5 :=
sorry

end NUMINAMATH_GPT_find_positive_integers_l764_76461


namespace NUMINAMATH_GPT_beta_gt_half_alpha_l764_76490

theorem beta_gt_half_alpha (alpha beta : ℝ) (h1 : Real.sin beta = (3/4) * Real.sin alpha) (h2 : 0 < alpha ∧ alpha ≤ 90) : beta > alpha / 2 :=
by
  sorry

end NUMINAMATH_GPT_beta_gt_half_alpha_l764_76490


namespace NUMINAMATH_GPT_selling_price_correct_l764_76447

noncomputable def selling_price (purchase_price : ℝ) (overhead_expenses : ℝ) (profit_percent : ℝ) : ℝ :=
  let total_cost_price := purchase_price + overhead_expenses
  let profit := (profit_percent / 100) * total_cost_price
  total_cost_price + profit

theorem selling_price_correct :
    selling_price 225 28 18.577075098814234 = 300 := by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l764_76447


namespace NUMINAMATH_GPT_find_fg3_l764_76466

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem find_fg3 : f (g 3) = 2 := by
  sorry

end NUMINAMATH_GPT_find_fg3_l764_76466


namespace NUMINAMATH_GPT_find_a_for_quadratic_max_l764_76417

theorem find_a_for_quadratic_max :
  ∃ a : ℝ, (∀ x : ℝ, a ≤ x ∧ x ≤ 1/2 → (x^2 + 2 * x - 2 ≤ 1)) ∧
           (∃ x : ℝ, a ≤ x ∧ x ≤ 1/2 ∧ (x^2 + 2 * x - 2 = 1)) ∧ 
           a = -3 :=
sorry

end NUMINAMATH_GPT_find_a_for_quadratic_max_l764_76417


namespace NUMINAMATH_GPT_sum_of_cubes_1998_l764_76459

theorem sum_of_cubes_1998 : 1998 = 334^3 + 332^3 + (-333)^3 + (-333)^3 := by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_1998_l764_76459


namespace NUMINAMATH_GPT_smallest_sector_angle_division_is_10_l764_76421

/-
  Prove that the smallest possible sector angle in a 15-sector division of a circle,
  where the central angles form an arithmetic sequence with integer values and the
  total sum of angles is 360 degrees, is 10 degrees.
-/
theorem smallest_sector_angle_division_is_10 :
  ∃ (a1 d : ℕ), (∀ i, i ∈ (List.range 15) → a1 + i * d > 0) ∧ (List.sum (List.map (fun i => a1 + i * d) (List.range 15)) = 360) ∧
  a1 = 10 := by
  sorry

end NUMINAMATH_GPT_smallest_sector_angle_division_is_10_l764_76421


namespace NUMINAMATH_GPT_six_digit_square_number_cases_l764_76413

theorem six_digit_square_number_cases :
  ∃ n : ℕ, 316 ≤ n ∧ n < 1000 ∧ (n^2 = 232324 ∨ n^2 = 595984 ∨ n^2 = 929296) :=
by {
  sorry
}

end NUMINAMATH_GPT_six_digit_square_number_cases_l764_76413


namespace NUMINAMATH_GPT_find_k_l764_76446

-- Definitions based on the conditions
def number := 24
def bigPart := 13
  
theorem find_k (x y k : ℕ) 
  (original_number : x + y = 24)
  (big_part : x = 13 ∨ y = 13)
  (equation : k * x + 5 * y = 146) : k = 7 := 
  sorry

end NUMINAMATH_GPT_find_k_l764_76446


namespace NUMINAMATH_GPT_distribute_pictures_l764_76470

/-
Tiffany uploaded 34 pictures from her phone, 55 from her camera,
and 12 from her tablet to Facebook. If she sorted the pics into 7 different albums
with the same amount of pics in each album, how many pictures were in each of the albums?
-/

theorem distribute_pictures :
  let phone_pics := 34
  let camera_pics := 55
  let tablet_pics := 12
  let total_pics := phone_pics + camera_pics + tablet_pics
  let albums := 7
  ∃ k r, (total_pics = k * albums + r) ∧ (r < albums) := by
  sorry

end NUMINAMATH_GPT_distribute_pictures_l764_76470


namespace NUMINAMATH_GPT_reflection_across_x_axis_l764_76469

theorem reflection_across_x_axis (x y : ℝ) : 
  (x, -y) = (-2, -4) ↔ (x, y) = (-2, 4) :=
by
  sorry

end NUMINAMATH_GPT_reflection_across_x_axis_l764_76469


namespace NUMINAMATH_GPT_modified_goldbach_2024_l764_76409

def is_prime (p : ℕ) : Prop := ∀ n : ℕ, n > 1 → n < p → ¬ (p % n = 0)

theorem modified_goldbach_2024 :
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ is_prime p1 ∧ is_prime p2 ∧ p1 + p2 = 2024 := 
sorry

end NUMINAMATH_GPT_modified_goldbach_2024_l764_76409


namespace NUMINAMATH_GPT_num_ways_placing_2015_bishops_l764_76483

-- Define the concept of placing bishops on a 2 x n chessboard without mutual attacks
def max_bishops (n : ℕ) : ℕ := n

-- Define the calculation of the number of ways to place these bishops
def num_ways_to_place_bishops (n : ℕ) : ℕ := 2 ^ n

-- The proof statement for our specific problem
theorem num_ways_placing_2015_bishops :
  num_ways_to_place_bishops 2015 = 2 ^ 2015 :=
by
  sorry

end NUMINAMATH_GPT_num_ways_placing_2015_bishops_l764_76483


namespace NUMINAMATH_GPT_triangle_b_value_triangle_area_value_l764_76482

noncomputable def triangle_b (a : ℝ) (cosA : ℝ) : ℝ :=
  let sinA := Real.sqrt (1 - cosA^2)
  let sinB := cosA
  (a * sinB) / sinA

noncomputable def triangle_area (a b c : ℝ) (sinC : ℝ) : ℝ :=
  0.5 * a * b * sinC

-- Given conditions
variable (A B : ℝ) (a : ℝ := 3) (cosA : ℝ := Real.sqrt 6 / 3) (B := A + Real.pi / 2)

-- The assertions to prove
theorem triangle_b_value :
  triangle_b a cosA = 3 * Real.sqrt 2 :=
sorry

theorem triangle_area_value :
  triangle_area 3 (3 * Real.sqrt 2) 1 (1 / 3) = (3 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_triangle_b_value_triangle_area_value_l764_76482


namespace NUMINAMATH_GPT_similar_triangles_perimeter_l764_76401

theorem similar_triangles_perimeter (P_small P_large : ℝ) 
  (h_ratio : P_small / P_large = 2 / 3) 
  (h_sum : P_small + P_large = 20) : 
  P_small = 8 := 
sorry

end NUMINAMATH_GPT_similar_triangles_perimeter_l764_76401


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l764_76408

theorem relationship_between_a_and_b : 
  ∀ (a b : ℝ), (∀ x y : ℝ, (x-a)^2 + (y-b)^2 = b^2 + 1 → (x+1)^2 + (y+1)^2 = 4 → (2 + 2*a)*x + (2 + 2*b)*y - a^2 - 1 = 0) → a^2 + 2*a + 2*b + 5 = 0 :=
by
  intros a b hyp
  sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l764_76408


namespace NUMINAMATH_GPT_length_of_bridge_correct_l764_76440

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_seconds : ℝ) : ℝ :=
  let train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
  let total_distance : ℝ := train_speed_ms * crossing_time_seconds
  total_distance - train_length

theorem length_of_bridge_correct :
  length_of_bridge 500 42 60 = 200.2 :=
by
  sorry -- Proof of the theorem

end NUMINAMATH_GPT_length_of_bridge_correct_l764_76440


namespace NUMINAMATH_GPT_point_M_coordinates_l764_76402

theorem point_M_coordinates (a : ℤ) (h : a + 3 = 0) : (a + 3, 2 * a - 2) = (0, -8) :=
by
  sorry

end NUMINAMATH_GPT_point_M_coordinates_l764_76402


namespace NUMINAMATH_GPT_point_not_in_plane_l764_76477

def is_in_plane (p0 : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) : Prop :=
  let (x0, y0, z0) := p0
  let (nx, ny, nz) := n
  let (x, y, z) := p
  (nx * (x - x0) + ny * (y - y0) + nz * (z - z0)) = 0

theorem point_not_in_plane :
  ¬ is_in_plane (1, 2, 3) (1, 1, 1) (-2, 5, 4) :=
by
  sorry

end NUMINAMATH_GPT_point_not_in_plane_l764_76477


namespace NUMINAMATH_GPT_max_m_l764_76488

noncomputable def f (x a : ℝ) : ℝ := 2 ^ |x + a|

theorem max_m (a m : ℝ) (H1 : ∀ x, f (3 + x) a = f (3 - x) a) 
(H2 : ∀ x y, x ≤ y → y ≤ m → f x a ≥ f y a) : 
  m = 3 :=
by
  sorry

end NUMINAMATH_GPT_max_m_l764_76488


namespace NUMINAMATH_GPT_x_plus_y_value_l764_76462

theorem x_plus_y_value (x y : ℕ) (h1 : 2^x = 8^(y + 1)) (h2 : 9^y = 3^(x - 9)) : x + y = 27 :=
by
  sorry

end NUMINAMATH_GPT_x_plus_y_value_l764_76462


namespace NUMINAMATH_GPT_lucas_investment_l764_76419

noncomputable def investment_amount (y : ℝ) : ℝ := 1500 - y

theorem lucas_investment :
  ∃ y : ℝ, (y * 1.04 + (investment_amount y) * 1.06 = 1584.50) ∧ y = 275 :=
by
  sorry

end NUMINAMATH_GPT_lucas_investment_l764_76419


namespace NUMINAMATH_GPT_largest_divisor_360_450_l764_76430

theorem largest_divisor_360_450 : ∃ d, (d ∣ 360 ∧ d ∣ 450) ∧ (∀ e, (e ∣ 360 ∧ e ∣ 450) → e ≤ d) ∧ d = 90 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_360_450_l764_76430


namespace NUMINAMATH_GPT_train_crosses_platform_in_39_seconds_l764_76497

-- Definitions based on the problem's conditions
def train_length : ℕ := 450
def time_to_cross_signal : ℕ := 18
def platform_length : ℕ := 525

-- The speed of the train
def train_speed : ℕ := train_length / time_to_cross_signal

-- The total distance the train has to cover
def total_distance : ℕ := train_length + platform_length

-- The time it takes for the train to cross the platform
def time_to_cross_platform : ℕ := total_distance / train_speed

-- The theorem we need to prove
theorem train_crosses_platform_in_39_seconds :
  time_to_cross_platform = 39 := by
  sorry

end NUMINAMATH_GPT_train_crosses_platform_in_39_seconds_l764_76497


namespace NUMINAMATH_GPT_cube_root_of_8_is_2_l764_76429

theorem cube_root_of_8_is_2 : ∃ x : ℝ, x ^ 3 = 8 ∧ x = 2 :=
by
  have h : (2 : ℝ) ^ 3 = 8 := by norm_num
  exact ⟨2, h, rfl⟩

end NUMINAMATH_GPT_cube_root_of_8_is_2_l764_76429


namespace NUMINAMATH_GPT_compute_100p_plus_q_l764_76489

theorem compute_100p_plus_q
  (p q : ℝ)
  (h1 : ∀ x : ℝ, (x + p) * (x + q) * (x + 15) = 0 → 
                  x ≠ -4 → x ≠ -15 → x ≠ -p → x ≠ -q)
  (h2 : ∀ x : ℝ, (x + 2 * p) * (x + 4) * (x + 9) = 0 → 
                  x ≠ -q → x ≠ -15 → (x = -4 ∨ x = -9))
  : 100 * p + q = -191 := 
sorry

end NUMINAMATH_GPT_compute_100p_plus_q_l764_76489


namespace NUMINAMATH_GPT_condition_eq_l764_76403

-- We are given a triangle ABC with sides opposite angles A, B, and C being a, b, and c respectively.
variable (A B C a b c : ℝ)

-- Conditions for the problem
def sin_eq (A B : ℝ) := Real.sin A = Real.sin B
def cos_eq (A B : ℝ) := Real.cos A = Real.cos B
def sin2_eq (A B : ℝ) := Real.sin (2 * A) = Real.sin (2 * B)
def cos2_eq (A B : ℝ) := Real.cos (2 * A) = Real.cos (2 * B)

-- The main statement we need to prove
theorem condition_eq (h1 : sin_eq A B) (h2 : cos_eq A B) (h4 : cos2_eq A B) : a = b :=
sorry

end NUMINAMATH_GPT_condition_eq_l764_76403


namespace NUMINAMATH_GPT_find_n_cosine_l764_76448

theorem find_n_cosine : ∃ (n : ℤ), -180 ≤ n ∧ n ≤ 180 ∧ (∃ m : ℤ, n = 25 + 360 * m ∨ n = -25 + 360 * m) :=
by
  sorry

end NUMINAMATH_GPT_find_n_cosine_l764_76448


namespace NUMINAMATH_GPT_inequality_solution_l764_76404

theorem inequality_solution (x : ℝ) (h : x * (x^2 + 1) > (x + 1) * (x^2 - x + 1)) : x > 1 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l764_76404


namespace NUMINAMATH_GPT_largest_int_less_than_100_remainder_4_l764_76453

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end NUMINAMATH_GPT_largest_int_less_than_100_remainder_4_l764_76453


namespace NUMINAMATH_GPT_percentage_sum_l764_76452

noncomputable def womenWithRedHairBelow30 : ℝ := 0.07
noncomputable def menWithDarkHair30OrOlder : ℝ := 0.13

theorem percentage_sum :
  womenWithRedHairBelow30 + menWithDarkHair30OrOlder = 0.20 := by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_percentage_sum_l764_76452


namespace NUMINAMATH_GPT_compute_expression_l764_76472

theorem compute_expression : 1004^2 - 996^2 - 1000^2 + 1000^2 = 16000 := 
by sorry

end NUMINAMATH_GPT_compute_expression_l764_76472


namespace NUMINAMATH_GPT_probability_two_queens_or_at_least_one_jack_l764_76433

-- Definitions
def num_jacks : ℕ := 4
def num_queens : ℕ := 4
def total_cards : ℕ := 52

-- Probability calculation for drawing either two Queens or at least one Jack
theorem probability_two_queens_or_at_least_one_jack :
  (4 / 52) * (3 / (52 - 1)) + ((4 / 52) * (48 / (52 - 1)) + (48 / 52) * (4 / (52 - 1)) + (4 / 52) * (3 / (52 - 1))) = 2 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_queens_or_at_least_one_jack_l764_76433


namespace NUMINAMATH_GPT_identical_lines_unique_pair_l764_76474

theorem identical_lines_unique_pair :
  ∃! (a b : ℚ), 2 * (0 : ℚ) + a * (0 : ℚ) + 10 = 0 ∧ b * (0 : ℚ) - 3 * (0 : ℚ) - 15 = 0 ∧ 
  (-2 / a = b / 3) ∧ (-10 / a = 5) :=
by {
  -- Given equations in slope-intercept form:
  -- y = -2 / a * x - 10 / a
  -- y = b / 3 * x + 5
  -- Slope and intercept comparison leads to equations:
  -- -2 / a = b / 3
  -- -10 / a = 5
  sorry
}

end NUMINAMATH_GPT_identical_lines_unique_pair_l764_76474


namespace NUMINAMATH_GPT_find_prob_A_l764_76445

variable (P : String → ℝ)
variable (A B : String)

-- Conditions
axiom prob_complement_twice : P B = 2 * P A
axiom prob_sum_to_one : P A + P B = 1

-- Statement to be proved
theorem find_prob_A : P A = 1 / 3 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_find_prob_A_l764_76445


namespace NUMINAMATH_GPT_negate_universal_statement_l764_76468

theorem negate_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negate_universal_statement_l764_76468


namespace NUMINAMATH_GPT_min_value_of_sum_range_of_x_l764_76499

noncomputable def ab_condition (a b : ℝ) : Prop := a + b = 1
noncomputable def ra_positive (a b : ℝ) : Prop := a > 0 ∧ b > 0

-- Problem 1: Minimum value of (1/a + 4/b)

theorem min_value_of_sum (a b : ℝ) (h_ab : ab_condition a b) (h_pos : ra_positive a b) : 
    ∃ m : ℝ, m = 9 ∧ ∀ a b, ab_condition a b → ra_positive a b → 
    (1 / a + 4 / b) ≥ m :=
by sorry

-- Problem 2: Range of x for which the inequality holds

theorem range_of_x (a b x : ℝ) (h_ab : ab_condition a b) (h_pos : ra_positive a b) : 
    (1 / a + 4 / b) ≥ |2 * x - 1| - |x + 1| → x ∈ Set.Icc (-7 : ℝ) 11 :=
by sorry

end NUMINAMATH_GPT_min_value_of_sum_range_of_x_l764_76499


namespace NUMINAMATH_GPT_max_books_borrowed_l764_76496

theorem max_books_borrowed (total_students : ℕ) (no_books : ℕ) (one_book : ℕ)
  (two_books : ℕ) (at_least_three_books : ℕ) (avg_books_per_student : ℕ) :
  total_students = 35 →
  no_books = 2 →
  one_book = 12 →
  two_books = 10 →
  avg_books_per_student = 2 →
  total_students - (no_books + one_book + two_books) = at_least_three_books →
  ∃ max_books_borrowed_by_individual, max_books_borrowed_by_individual = 8 :=
by
  intros h_total_students h_no_books h_one_book h_two_books h_avg_books_per_student h_remaining_students
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_max_books_borrowed_l764_76496


namespace NUMINAMATH_GPT_cost_per_adult_meal_l764_76443

theorem cost_per_adult_meal (total_people : ℕ) (num_kids : ℕ) (total_cost : ℕ) (cost_per_kid : ℕ) :
  total_people = 12 →
  num_kids = 7 →
  cost_per_kid = 0 →
  total_cost = 15 →
  (total_cost / (total_people - num_kids)) = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cost_per_adult_meal_l764_76443


namespace NUMINAMATH_GPT_gloria_initial_dimes_l764_76420

variable (Q D : ℕ)

theorem gloria_initial_dimes (h1 : D = 5 * Q) 
                             (h2 : (3 * Q) / 5 + D = 392) : 
                             D = 350 := 
by {
  sorry
}

end NUMINAMATH_GPT_gloria_initial_dimes_l764_76420


namespace NUMINAMATH_GPT_complement_of_angle_l764_76481

theorem complement_of_angle (x : ℝ) (h : 90 - x = 3 * x + 10) : x = 20 := by
  sorry

end NUMINAMATH_GPT_complement_of_angle_l764_76481


namespace NUMINAMATH_GPT_simplify_to_quadratic_l764_76423

noncomputable def simplify_expression (a b c x : ℝ) : ℝ := 
  (x + a)^2 / ((a - b) * (a - c)) + 
  (x + b)^2 / ((b - a) * (b - c + 2)) + 
  (x + c)^2 / ((c - a) * (c - b))

theorem simplify_to_quadratic {a b c x : ℝ} (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  simplify_expression a b c x = x^2 - (a + b + c) * x + sorry :=
sorry

end NUMINAMATH_GPT_simplify_to_quadratic_l764_76423


namespace NUMINAMATH_GPT_expected_value_proof_l764_76425

noncomputable def expected_value_xi : ℚ :=
  let p_xi_2 : ℚ := 3/5
  let p_xi_3 : ℚ := 3/10
  let p_xi_4 : ℚ := 1/10
  2 * p_xi_2 + 3 * p_xi_3 + 4 * p_xi_4

theorem expected_value_proof :
  expected_value_xi = 5/2 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_proof_l764_76425


namespace NUMINAMATH_GPT_bookstore_floor_l764_76407

theorem bookstore_floor
  (academy_floor : ℤ)
  (reading_room_floor : ℤ)
  (bookstore_floor : ℤ)
  (h1 : academy_floor = 7)
  (h2 : reading_room_floor = academy_floor + 4)
  (h3 : bookstore_floor = reading_room_floor - 9) :
  bookstore_floor = 2 :=
by
  sorry

end NUMINAMATH_GPT_bookstore_floor_l764_76407


namespace NUMINAMATH_GPT_bronze_needed_l764_76415

/-- 
The total amount of bronze Martin needs for three bells in pounds.
-/
theorem bronze_needed (w1 w2 w3 : ℕ) 
  (h1 : w1 = 50) 
  (h2 : w2 = 2 * w1) 
  (h3 : w3 = 4 * w2) 
  : (w1 + w2 + w3 = 550) := 
by { 
  sorry 
}

end NUMINAMATH_GPT_bronze_needed_l764_76415


namespace NUMINAMATH_GPT_line_passing_quadrants_l764_76427

-- Definition of the line
def line (x : ℝ) : ℝ := 5 * x - 2

-- Prove that the line passes through Quadrants I, III, and IV
theorem line_passing_quadrants :
  ∃ x_1 > 0, line x_1 > 0 ∧      -- Quadrant I
  ∃ x_3 < 0, line x_3 < 0 ∧      -- Quadrant III
  ∃ x_4 > 0, line x_4 < 0 :=     -- Quadrant IV
sorry

end NUMINAMATH_GPT_line_passing_quadrants_l764_76427


namespace NUMINAMATH_GPT_f_neg2_eq_neg1_l764_76454

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem f_neg2_eq_neg1 : f (-2) = -1 := by
  sorry

end NUMINAMATH_GPT_f_neg2_eq_neg1_l764_76454


namespace NUMINAMATH_GPT_avg_price_per_book_l764_76449

theorem avg_price_per_book (n1 n2 p1 p2 : ℕ) (h1 : n1 = 65) (h2 : n2 = 55) (h3 : p1 = 1380) (h4 : p2 = 900) :
    (p1 + p2) / (n1 + n2) = 19 := by
  sorry

end NUMINAMATH_GPT_avg_price_per_book_l764_76449


namespace NUMINAMATH_GPT_cake_cubes_with_exactly_two_faces_iced_l764_76463

theorem cake_cubes_with_exactly_two_faces_iced :
  let cake : ℕ := 3 -- cake dimension
  let total_cubes : ℕ := cake ^ 3 -- number of smaller cubes (total 27)
  let cubes_with_two_faces_icing := 4
  (∀ cake icing (smaller_cubes : ℕ), icing ≠ 0 → smaller_cubes = cake ^ 3 → 
    let top_iced := cake - 2 -- cubes with icing on top only
    let front_iced := cake - 2 -- cubes with icing on front only
    let back_iced := cake - 2 -- cubes with icing on back only
    ((top_iced * 2) = cubes_with_two_faces_icing)) :=
  sorry

end NUMINAMATH_GPT_cake_cubes_with_exactly_two_faces_iced_l764_76463


namespace NUMINAMATH_GPT_decompose_x_l764_76451

def x : ℝ × ℝ × ℝ := (6, 12, -1)
def p : ℝ × ℝ × ℝ := (1, 3, 0)
def q : ℝ × ℝ × ℝ := (2, -1, 1)
def r : ℝ × ℝ × ℝ := (0, -1, 2)

theorem decompose_x :
  x = (4 : ℝ) • p + q - r :=
sorry

end NUMINAMATH_GPT_decompose_x_l764_76451


namespace NUMINAMATH_GPT_tetrahedron_edge_length_correct_l764_76471

noncomputable def radius := Real.sqrt 2
noncomputable def center_to_center_distance := 2 * radius
noncomputable def tetrahedron_edge_length := center_to_center_distance

theorem tetrahedron_edge_length_correct :
  tetrahedron_edge_length = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_tetrahedron_edge_length_correct_l764_76471


namespace NUMINAMATH_GPT_sin_equations_solution_l764_76428

theorem sin_equations_solution {k : ℤ} (hk : k ≤ 1 ∨ k ≥ 5) : 
  (∃ x : ℝ, 2 * x = π * k ∧ x = (π * k) / 2) ∨ x = 7 * π / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_equations_solution_l764_76428


namespace NUMINAMATH_GPT_integral_solution_l764_76475

noncomputable def integral_expression : Real → Real :=
  fun x => (1 + (x ^ (3 / 4))) ^ (4 / 5) / (x ^ (47 / 20))

theorem integral_solution :
  ∫ (x : Real), integral_expression x = - (20 / 27) * ((1 + (x ^ (3 / 4)) / (x ^ (3 / 4))) ^ (9 / 5)) + C := 
by 
  sorry

end NUMINAMATH_GPT_integral_solution_l764_76475


namespace NUMINAMATH_GPT_minimum_distance_l764_76410

theorem minimum_distance (x y : ℝ) (h : x - y - 1 = 0) : (x - 2)^2 + (y - 2)^2 ≥ 1 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_distance_l764_76410


namespace NUMINAMATH_GPT_solve_fraction_equation_l764_76455

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ↔ x = -9 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_fraction_equation_l764_76455


namespace NUMINAMATH_GPT_smallest_pos_n_l764_76405

theorem smallest_pos_n (n : ℕ) (h : 435 * n % 30 = 867 * n % 30) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_pos_n_l764_76405


namespace NUMINAMATH_GPT_perimeter_of_triangle_l764_76424

theorem perimeter_of_triangle
  (P : ℝ)
  (r : ℝ := 1.5)
  (A : ℝ := 29.25)
  (h : A = r * (P / 2)) :
  P = 39 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l764_76424


namespace NUMINAMATH_GPT_traffic_light_probability_l764_76422

theorem traffic_light_probability :
  let total_cycle_time := 63
  let green_time := 30
  let yellow_time := 3
  let red_time := 30
  let observation_window := 3
  let change_intervals := 3 * 3
  ∃ (P : ℚ), P = change_intervals / total_cycle_time ∧ P = 1 / 7 := 
by
  sorry

end NUMINAMATH_GPT_traffic_light_probability_l764_76422
