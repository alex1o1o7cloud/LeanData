import Mathlib

namespace NUMINAMATH_GPT_miles_in_one_hour_eq_8_l1114_111454

-- Parameters as given in the conditions
variables (x : ℕ) (h1 : ∀ t : ℕ, t >= 6 → t % 6 = 0 ∨ t % 6 < 6)
variables (miles_in_one_hour : ℕ)
-- Given condition: The car drives 88 miles in 13 hours.
variable (miles_in_13_hours : miles_in_one_hour * 11 = 88)

-- Statement to prove: The car can drive 8 miles in one hour.
theorem miles_in_one_hour_eq_8 : miles_in_one_hour = 8 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_miles_in_one_hour_eq_8_l1114_111454


namespace NUMINAMATH_GPT_sin_transformation_l1114_111406

theorem sin_transformation (α : ℝ) (h : Real.sin (3 * Real.pi / 2 + α) = 3 / 5) :
  Real.sin (Real.pi / 2 + 2 * α) = -7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_transformation_l1114_111406


namespace NUMINAMATH_GPT_cube_painting_probability_l1114_111497

theorem cube_painting_probability :
  let total_configurations := 2^6 * 2^6
  let identical_configurations := 90
  (identical_configurations / total_configurations : ℚ) = 45 / 2048 :=
by
  sorry

end NUMINAMATH_GPT_cube_painting_probability_l1114_111497


namespace NUMINAMATH_GPT_triangle_area_from_squares_l1114_111456

noncomputable def area_of_triangle (S1 S2 : ℝ) : ℝ :=
  let side1 := Real.sqrt S1
  let side2 := Real.sqrt S2
  0.5 * side1 * side2

theorem triangle_area_from_squares
  (A1 A2 : ℝ)
  (h1 : A1 = 196)
  (h2 : A2 = 100) :
  area_of_triangle A1 A2 = 70 :=
by
  rw [h1, h2]
  unfold area_of_triangle
  rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow]
  norm_num
  sorry

end NUMINAMATH_GPT_triangle_area_from_squares_l1114_111456


namespace NUMINAMATH_GPT_simplify_expression_l1114_111400

theorem simplify_expression :
  (144 / 12) * (5 / 90) * (9 / 3) * 2 = 4 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1114_111400


namespace NUMINAMATH_GPT_log2_6_gt_2_sqrt_5_l1114_111415

theorem log2_6_gt_2_sqrt_5 : 2 + Real.logb 2 6 > 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_log2_6_gt_2_sqrt_5_l1114_111415


namespace NUMINAMATH_GPT_snowfall_difference_l1114_111414

-- Defining all conditions given in the problem
def BaldMountain_snowfall_meters : ℝ := 1.5
def BillyMountain_snowfall_meters : ℝ := 3.5
def MountPilot_snowfall_centimeters : ℝ := 126
def RockstonePeak_snowfall_millimeters : ℝ := 5250
def SunsetRidge_snowfall_meters : ℝ := 2.25

-- Conversion constants
def meters_to_centimeters : ℝ := 100
def millimeters_to_centimeters : ℝ := 0.1

-- Converting snowfall amounts to centimeters
def BaldMountain_snowfall_centimeters : ℝ := BaldMountain_snowfall_meters * meters_to_centimeters
def BillyMountain_snowfall_centimeters : ℝ := BillyMountain_snowfall_meters * meters_to_centimeters
def RockstonePeak_snowfall_centimeters : ℝ := RockstonePeak_snowfall_millimeters * millimeters_to_centimeters
def SunsetRidge_snowfall_centimeters : ℝ := SunsetRidge_snowfall_meters * meters_to_centimeters

-- Defining total combined snowfall
def combined_snowfall_centimeters : ℝ :=
  BillyMountain_snowfall_centimeters + MountPilot_snowfall_centimeters + RockstonePeak_snowfall_centimeters + SunsetRidge_snowfall_centimeters

-- Stating the proof statement
theorem snowfall_difference :
  combined_snowfall_centimeters - BaldMountain_snowfall_centimeters = 1076 := 
  by
    sorry

end NUMINAMATH_GPT_snowfall_difference_l1114_111414


namespace NUMINAMATH_GPT_investor_profits_l1114_111404

/-- Problem: Given the total contributions and profit sharing conditions, calculate the amount 
    each investor receives. -/

theorem investor_profits :
  ∀ (A_contribution B_contribution C_contribution D_contribution : ℝ) 
    (A_profit B_profit C_profit D_profit : ℝ) 
    (total_capital total_profit : ℝ),
    total_capital = 100000 → 
    A_contribution = B_contribution + 5000 →
    B_contribution = C_contribution + 10000 →
    C_contribution = D_contribution + 5000 →
    total_profit = 60000 →
    A_profit = (35 / 100) * total_profit * (1 + 10 / 100) →
    B_profit = (30 / 100) * total_profit * (1 + 8 / 100) →
    C_profit = (20 / 100) * total_profit * (1 + 5 / 100) → 
    D_profit = (15 / 100) * total_profit →
    (A_profit = 23100 ∧ B_profit = 19440 ∧ C_profit = 12600 ∧ D_profit = 9000) :=
by
  intros
  sorry

end NUMINAMATH_GPT_investor_profits_l1114_111404


namespace NUMINAMATH_GPT_value_of_expression_l1114_111426

variables (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 + b * x ^ 2 + c * x + d

theorem value_of_expression (h : f a b c d (-2) = -3) : 8 * a - 4 * b + 2 * c - d = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_expression_l1114_111426


namespace NUMINAMATH_GPT_initial_butterfly_count_l1114_111481

theorem initial_butterfly_count (n : ℕ) (h : (2 / 3 : ℚ) * n = 6) : n = 9 :=
sorry

end NUMINAMATH_GPT_initial_butterfly_count_l1114_111481


namespace NUMINAMATH_GPT_original_price_l1114_111439

variable (x : ℝ)

-- Condition 1: Selling at 60% of the original price results in a 20 yuan loss
def condition1 : Prop := 0.6 * x + 20 = x * 0.8 - 15

-- The goal is to prove that the original price is 175 yuan under the given conditions
theorem original_price (h : condition1 x) : x = 175 :=
sorry

end NUMINAMATH_GPT_original_price_l1114_111439


namespace NUMINAMATH_GPT_smallest_y_76545_l1114_111413

theorem smallest_y_76545 (y : ℕ) (h1 : ∀ z : ℕ, 0 < z → (76545 * z = k ^ 2 → (3 ∣ z ∨ 5 ∣ z) → z = y)) : y = 7 :=
sorry

end NUMINAMATH_GPT_smallest_y_76545_l1114_111413


namespace NUMINAMATH_GPT_Adam_ate_more_than_Bill_l1114_111423

-- Definitions
def Sierra_ate : ℕ := 12
def Bill_ate : ℕ := Sierra_ate / 2
def total_pies_eaten : ℕ := 27
def Sierra_and_Bill_ate : ℕ := Sierra_ate + Bill_ate
def Adam_ate : ℕ := total_pies_eaten - Sierra_and_Bill_ate
def Adam_more_than_Bill : ℕ := Adam_ate - Bill_ate

-- Statement to prove
theorem Adam_ate_more_than_Bill :
  Adam_more_than_Bill = 3 :=
by
  sorry

end NUMINAMATH_GPT_Adam_ate_more_than_Bill_l1114_111423


namespace NUMINAMATH_GPT_sqrt_mul_example_complex_expression_example_l1114_111425

theorem sqrt_mul_example : Real.sqrt 3 * Real.sqrt 27 = 9 :=
by sorry

theorem complex_expression_example : 
  (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) - (Real.sqrt 3 - 2)^2 = 4 * Real.sqrt 3 - 6 :=
by sorry

end NUMINAMATH_GPT_sqrt_mul_example_complex_expression_example_l1114_111425


namespace NUMINAMATH_GPT_part_a_exists_part_b_not_exists_l1114_111437

theorem part_a_exists :
  ∃ (a b : ℤ), (∀ x : ℝ, x^2 + a*x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + a*x + b = 0) :=
sorry

theorem part_b_not_exists :
  ¬ ∃ (a b : ℤ), (∀ x : ℝ, x^2 + 2*a*x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + 2*a*x + b = 0) :=
sorry

end NUMINAMATH_GPT_part_a_exists_part_b_not_exists_l1114_111437


namespace NUMINAMATH_GPT_difference_between_max_and_min_34_l1114_111455

theorem difference_between_max_and_min_34 
  (A B C D E: ℕ) 
  (h_avg: (A + B + C + D + E) / 5 = 50) 
  (h_max: E ≤ 58) 
  (h_distinct: A < B ∧ B < C ∧ C < D ∧ D < E) 
: E - A = 34 := 
sorry

end NUMINAMATH_GPT_difference_between_max_and_min_34_l1114_111455


namespace NUMINAMATH_GPT_solve_ineq_l1114_111448

noncomputable def f (x : ℝ) : ℝ := (2 / (x + 2)) + (4 / (x + 8)) - (7 / 3)

theorem solve_ineq (x : ℝ) : 
  (f x ≤ 0) ↔ (x ∈ Set.Ioc (-8) 4) := 
sorry

end NUMINAMATH_GPT_solve_ineq_l1114_111448


namespace NUMINAMATH_GPT_max_area_BPC_l1114_111445

noncomputable def triangle_area_max (AB BC CA : ℝ) (D : ℝ) : ℝ :=
  if h₁ : AB = 13 ∧ BC = 15 ∧ CA = 14 then
    112.5 - 56.25 * Real.sqrt 3
  else 0

theorem max_area_BPC : triangle_area_max 13 15 14 D = 112.5 - 56.25 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_max_area_BPC_l1114_111445


namespace NUMINAMATH_GPT_plane_speed_with_tailwind_l1114_111402

theorem plane_speed_with_tailwind (V : ℝ) (tailwind_speed : ℝ) (ground_speed_against_tailwind : ℝ) 
  (H1 : tailwind_speed = 75) (H2 : ground_speed_against_tailwind = 310) (H3 : V - tailwind_speed = ground_speed_against_tailwind) :
  V + tailwind_speed = 460 :=
by
  sorry

end NUMINAMATH_GPT_plane_speed_with_tailwind_l1114_111402


namespace NUMINAMATH_GPT_cheolsu_weight_l1114_111490

variable (C M : ℝ)

theorem cheolsu_weight:
  (C = (2/3) * M) →
  (C + 72 = 2 * M) →
  C = 36 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cheolsu_weight_l1114_111490


namespace NUMINAMATH_GPT_evaluate_expression_l1114_111462

theorem evaluate_expression : 
  (3^4 + 3^4 + 3^4) / (3^(-4) + 3^(-4)) = 9841.5 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1114_111462


namespace NUMINAMATH_GPT_find_function_l1114_111468

theorem find_function (f : ℝ → ℝ)
  (h₁ : ∀ x : ℝ, x * (f (x + 1) - f x) = f x)
  (h₂ : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ |k| ≤ 1 :=
sorry

end NUMINAMATH_GPT_find_function_l1114_111468


namespace NUMINAMATH_GPT_smallest_n_probability_l1114_111463

theorem smallest_n_probability (n : ℕ) : (1 / (n * (n + 1)) < 1 / 2023) → (n ≥ 45) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_probability_l1114_111463


namespace NUMINAMATH_GPT_lollipops_per_day_l1114_111492

variable (Alison_lollipops : ℕ) (Henry_lollipops : ℕ) (Diane_lollipops : ℕ) (Total_lollipops : ℕ) (Days : ℕ)

-- Conditions given in the problem
axiom condition1 : Alison_lollipops = 60
axiom condition2 : Henry_lollipops = Alison_lollipops + 30
axiom condition3 : Alison_lollipops = Diane_lollipops / 2
axiom condition4 : Total_lollipops = Alison_lollipops + Henry_lollipops + Diane_lollipops
axiom condition5 : Days = 6

-- Question to prove
theorem lollipops_per_day : (Total_lollipops / Days) = 45 := sorry

end NUMINAMATH_GPT_lollipops_per_day_l1114_111492


namespace NUMINAMATH_GPT_gcd_xyz_times_xyz_is_square_l1114_111474

theorem gcd_xyz_times_xyz_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, k^2 = Nat.gcd x (Nat.gcd y z) * x * y * z :=
by
  sorry

end NUMINAMATH_GPT_gcd_xyz_times_xyz_is_square_l1114_111474


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_simplify_expr3_l1114_111444

-- 1. Proving (1)(2x^{2})^{3}-x^{2}·x^{4} = 7x^{6}
theorem simplify_expr1 (x : ℝ) : (1 : ℝ) * (2 * x^2)^3 - x^2 * x^4 = 7 * x^6 := 
by 
  sorry

-- 2. Proving (a+b)^{2}-b(2a+b) = a^{2}
theorem simplify_expr2 (a b : ℝ) : (a + b)^2 - b * (2 * a + b) = a^2 := 
by 
  sorry

-- 3. Proving (x+1)(x-1)-x^{2} = -1
theorem simplify_expr3 (x : ℝ) : (x + 1) * (x - 1) - x^2 = -1 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_simplify_expr3_l1114_111444


namespace NUMINAMATH_GPT_popped_white_probability_l1114_111441

theorem popped_white_probability :
  let P_white := 2 / 3
  let P_yellow := 1 / 3
  let P_pop_given_white := 1 / 2
  let P_pop_given_yellow := 2 / 3

  let P_white_and_pop := P_white * P_pop_given_white
  let P_yellow_and_pop := P_yellow * P_pop_given_yellow
  let P_pop := P_white_and_pop + P_yellow_and_pop

  let P_white_given_pop := P_white_and_pop / P_pop

  P_white_given_pop = 3 / 5 := sorry

end NUMINAMATH_GPT_popped_white_probability_l1114_111441


namespace NUMINAMATH_GPT_largest_divisor_l1114_111451

theorem largest_divisor (n : ℕ) (hn : Even n) : ∃ k, ∀ n, Even n → k ∣ (n * (n+2) * (n+4) * (n+6) * (n+8)) ∧ (∀ m, (∀ n, Even n → m ∣ (n * (n+2) * (n+4) * (n+6) * (n+8))) → m ≤ k) :=
by
  use 96
  { sorry }

end NUMINAMATH_GPT_largest_divisor_l1114_111451


namespace NUMINAMATH_GPT_electronics_weight_is_9_l1114_111457

noncomputable def electronics_weight : ℕ :=
  let B : ℕ := sorry -- placeholder for the value of books weight.
  let C : ℕ := 12
  let E : ℕ := 9
  have h1 : (B : ℚ) / (C : ℚ) = 7 / 4 := sorry
  have h2 : (C : ℚ) / (E : ℚ) = 4 / 3 := sorry
  have h3 : (B : ℚ) / (C - 6 : ℚ) = 7 / 2 := sorry
  E

theorem electronics_weight_is_9 : electronics_weight = 9 :=
by
  dsimp [electronics_weight]
  repeat { sorry }

end NUMINAMATH_GPT_electronics_weight_is_9_l1114_111457


namespace NUMINAMATH_GPT_total_payment_correct_l1114_111411

def cost (n : ℕ) : ℕ :=
  if n <= 10 then n * 25
  else 10 * 25 + (n - 10) * (4 * 25 / 5)

def final_cost_with_discount (n : ℕ) : ℕ :=
  let initial_cost := cost n
  if n > 20 then initial_cost - initial_cost / 10
  else initial_cost

def orders_X := 60 * 20 / 100
def orders_Y := 60 * 25 / 100
def orders_Z := 60 * 55 / 100

def cost_X := final_cost_with_discount orders_X
def cost_Y := final_cost_with_discount orders_Y
def cost_Z := final_cost_with_discount orders_Z

theorem total_payment_correct : cost_X + cost_Y + cost_Z = 1279 := by
  sorry

end NUMINAMATH_GPT_total_payment_correct_l1114_111411


namespace NUMINAMATH_GPT_alpha_value_l1114_111436

open Complex

theorem alpha_value (α β : ℂ) (h1 : β = 2 + 3 * I) (h2 : (α + β).im = 0) (h3 : (I * (2 * α - β)).im = 0) : α = 6 + 4 * I :=
by
  sorry

end NUMINAMATH_GPT_alpha_value_l1114_111436


namespace NUMINAMATH_GPT_john_baseball_cards_l1114_111443

theorem john_baseball_cards (new_cards old_cards cards_per_page : ℕ) (h1 : new_cards = 8) (h2 : old_cards = 16) (h3 : cards_per_page = 3) :
  (new_cards + old_cards) / cards_per_page = 8 := by
  sorry

end NUMINAMATH_GPT_john_baseball_cards_l1114_111443


namespace NUMINAMATH_GPT_repairs_cost_correct_l1114_111401

variable (C : ℝ)

def cost_of_scooter : ℝ := C
def repair_cost (C : ℝ) : ℝ := 0.10 * C
def selling_price (C : ℝ) : ℝ := 1.20 * C
def profit (C : ℝ) : ℝ := 1100
def profit_percentage (C : ℝ) : ℝ := 0.20 

theorem repairs_cost_correct (C : ℝ) (h₁ : selling_price C - cost_of_scooter C = profit C) (h₂ : profit_percentage C = 0.20) : 
  repair_cost C = 550 := by
  sorry

end NUMINAMATH_GPT_repairs_cost_correct_l1114_111401


namespace NUMINAMATH_GPT_always_non_monotonic_l1114_111430

noncomputable def f (a t x : ℝ) : ℝ :=
if x ≤ t then (2*a - 1)*x + 3*a - 4 else x^3 - x

theorem always_non_monotonic (a : ℝ) (t : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → f a t x1 ≤ f a t x2 ∨ f a t x1 ≥ f a t x2) → a ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_always_non_monotonic_l1114_111430


namespace NUMINAMATH_GPT_cubic_as_diff_of_squares_l1114_111447

theorem cubic_as_diff_of_squares (n : ℕ) (h : n > 1) :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n^3 = a^2 - b^2 := 
sorry

end NUMINAMATH_GPT_cubic_as_diff_of_squares_l1114_111447


namespace NUMINAMATH_GPT_molecular_weight_of_acid_l1114_111434

theorem molecular_weight_of_acid (molecular_weight : ℕ) (n : ℕ) (h : molecular_weight = 792) (hn : n = 9) :
  molecular_weight = 792 :=
by 
  sorry

end NUMINAMATH_GPT_molecular_weight_of_acid_l1114_111434


namespace NUMINAMATH_GPT_harmonic_progression_l1114_111428

theorem harmonic_progression (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
(h_harm : 1 / (a : ℝ) + 1 / (c : ℝ) = 2 / (b : ℝ))
(h_div : c % b = 0)
(h_inc : a < b ∧ b < c) :
  a = 20 → 
  (b, c) = (30, 60) ∨ (b, c) = (35, 140) ∨ (b, c) = (36, 180) ∨ (b, c) = (38, 380) ∨ (b, c) = (39, 780) :=
by sorry

end NUMINAMATH_GPT_harmonic_progression_l1114_111428


namespace NUMINAMATH_GPT_edge_length_is_correct_l1114_111485

-- Define the given conditions
def volume_material : ℕ := 12 * 18 * 6
def edge_length : ℕ := 3
def number_cubes : ℕ := 48
def volume_cube (e : ℕ) : ℕ := e * e * e

-- Problem statement in Lean:
theorem edge_length_is_correct : volume_material = number_cubes * volume_cube edge_length → edge_length = 3 :=
by
  sorry

end NUMINAMATH_GPT_edge_length_is_correct_l1114_111485


namespace NUMINAMATH_GPT_area_of_triangle_ABC_equation_of_circumcircle_l1114_111446

-- Define points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := 1, y := 3 }
def C : Point := { x := 3, y := 6 }

-- Theorem to prove the area of triangle ABC
theorem area_of_triangle_ABC : 
  let base := |B.y - A.y|
  let height := |C.x - A.x|
  (1/2) * base * height = 1 := sorry

-- Theorem to prove the equation of the circumcircle of triangle ABC
theorem equation_of_circumcircle : 
  let D := -10
  let E := -5
  let F := 15
  ∀ (x y : ℝ), (x - 5)^2 + (y - 5/2)^2 = 65/4 ↔ 
                x^2 + y^2 + D * x + E * y + F = 0 := sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_equation_of_circumcircle_l1114_111446


namespace NUMINAMATH_GPT_inequality_proof_l1114_111408

theorem inequality_proof {x y z : ℝ} (hxy : 0 < x) (hyz : 0 < y) (hzx : 0 < z) (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (x + z) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1114_111408


namespace NUMINAMATH_GPT_fourth_quadrangle_area_l1114_111475

theorem fourth_quadrangle_area (S1 S2 S3 S4 : ℝ) (h : S1 + S4 = S2 + S3) : S4 = S2 + S3 - S1 :=
by
  sorry

end NUMINAMATH_GPT_fourth_quadrangle_area_l1114_111475


namespace NUMINAMATH_GPT_multiply_103_97_l1114_111449

theorem multiply_103_97 : 103 * 97 = 9991 := 
by
  sorry

end NUMINAMATH_GPT_multiply_103_97_l1114_111449


namespace NUMINAMATH_GPT_complex_norm_solution_l1114_111478

noncomputable def complex_norm (z : Complex) : Real :=
  Complex.abs z

theorem complex_norm_solution (w z : Complex) 
  (wz_condition : w * z = 24 - 10 * Complex.I)
  (w_norm_condition : complex_norm w = Real.sqrt 29) :
  complex_norm z = (26 * Real.sqrt 29) / 29 :=
by
  sorry

end NUMINAMATH_GPT_complex_norm_solution_l1114_111478


namespace NUMINAMATH_GPT_neg_p_iff_neg_q_l1114_111458

theorem neg_p_iff_neg_q (a : ℝ) : (¬ (a < 0)) ↔ (¬ (a^2 > a)) :=
by 
    sorry

end NUMINAMATH_GPT_neg_p_iff_neg_q_l1114_111458


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l1114_111496

theorem arithmetic_sequence_ratio (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = (1/2) * n * (2 * a 1 + (n-1) * d))
  (h2 : ∀ n, T n = (1/2) * n * (2 * b 1 + (n-1) * d'))
  (h3 : ∀ n, S n / T n = 7*n / (n + 3)): a 5 / b 5 = 21 / 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l1114_111496


namespace NUMINAMATH_GPT_bao_interest_l1114_111487

noncomputable def initial_amount : ℝ := 1000
noncomputable def interest_rate : ℝ := 0.05
noncomputable def periods : ℕ := 6
noncomputable def final_amount : ℝ := initial_amount * (1 + interest_rate) ^ periods
noncomputable def interest_earned : ℝ := final_amount - initial_amount

theorem bao_interest :
  interest_earned = 340.095 := by
  sorry

end NUMINAMATH_GPT_bao_interest_l1114_111487


namespace NUMINAMATH_GPT_solution_set_l1114_111440

theorem solution_set (x : ℝ) : (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1114_111440


namespace NUMINAMATH_GPT_Freddy_is_18_l1114_111498

-- Definitions based on the conditions
def Job_age : Nat := 5
def Stephanie_age : Nat := 4 * Job_age
def Freddy_age : Nat := Stephanie_age - 2

-- Statement to prove
theorem Freddy_is_18 : Freddy_age = 18 := by
  sorry

end NUMINAMATH_GPT_Freddy_is_18_l1114_111498


namespace NUMINAMATH_GPT_bridget_apples_l1114_111491

variable (x : ℕ)

-- Conditions as definitions
def apples_after_splitting : ℕ := x / 2
def apples_after_giving_to_cassie : ℕ := apples_after_splitting x - 5
def apples_after_finding_hidden : ℕ := apples_after_giving_to_cassie x + 2
def final_apples : ℕ := apples_after_finding_hidden x
def bridget_keeps : ℕ := 6

-- Proof statement
theorem bridget_apples : x / 2 - 5 + 2 = bridget_keeps → x = 18 := by
  intros h
  sorry

end NUMINAMATH_GPT_bridget_apples_l1114_111491


namespace NUMINAMATH_GPT_min_value_of_a_plus_2b_l1114_111486

theorem min_value_of_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b - 3) :
  a + 2 * b = 4 * Real.sqrt 2 + 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_plus_2b_l1114_111486


namespace NUMINAMATH_GPT_solution_l1114_111476

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (4 + 2 * x) / (6 + 3 * x) = (3 + 2 * x) / (5 + 3 * x) ∧ x = -2

theorem solution : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_solution_l1114_111476


namespace NUMINAMATH_GPT_sum_first_five_terms_l1114_111424

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 1, ∀ n, a (n + 1) = a n * q

theorem sum_first_five_terms (h₁ : is_geometric_sequence a) 
  (h₂ : a 1 > 0) 
  (h₃ : a 1 * a 7 = 64) 
  (h₄ : a 3 + a 5 = 20) : 
  a 1 * (1 - (2 : ℝ) ^ 5) / (1 - 2) = 31 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_five_terms_l1114_111424


namespace NUMINAMATH_GPT_malcolm_initial_white_lights_l1114_111417

theorem malcolm_initial_white_lights :
  ∀ (red blue green remaining total_initial : ℕ),
    red = 12 →
    blue = 3 * red →
    green = 6 →
    remaining = 5 →
    total_initial = red + blue + green + remaining →
    total_initial = 59 :=
by
  intros red blue green remaining total_initial h1 h2 h3 h4 h5
  -- Add details if necessary for illustration
  -- sorry typically as per instructions
  sorry

end NUMINAMATH_GPT_malcolm_initial_white_lights_l1114_111417


namespace NUMINAMATH_GPT_find_b_l1114_111471

noncomputable def general_quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_b (a c : ℝ) (y1 y2 : ℝ) :
  y1 = general_quadratic a 3 c 2 →
  y2 = general_quadratic a 3 c (-2) →
  y1 - y2 = 12 →
  3 = 3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_b_l1114_111471


namespace NUMINAMATH_GPT_condition_swap_l1114_111477

variable {p q : Prop}

theorem condition_swap (h : ¬ p → q) (nh : ¬ (¬ p ↔ q)) : (p → ¬ q) ∧ ¬ (¬ (p ↔ ¬ q)) :=
by
  sorry

end NUMINAMATH_GPT_condition_swap_l1114_111477


namespace NUMINAMATH_GPT_arithmetic_sequence_nth_term_l1114_111416

noncomputable def nth_arithmetic_term (a : ℤ) (n : ℕ) : ℤ :=
  let a1 := a - 1
  let a2 := a + 1
  let a3 := 2 * a + 3
  if 2 * (a + 1) = (a - 1) + (2 * a + 3) then
    -1 + (n - 1) * 2
  else
    sorry

theorem arithmetic_sequence_nth_term (a : ℤ) (n : ℕ) (h : 2 * (a + 1) = (a - 1) + (2 * a + 3)) :
  nth_arithmetic_term a n = 2 * (n : ℤ) - 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_nth_term_l1114_111416


namespace NUMINAMATH_GPT_square_value_l1114_111499

theorem square_value {square : ℚ} (h : 8 / 12 = square / 3) : square = 2 :=
sorry

end NUMINAMATH_GPT_square_value_l1114_111499


namespace NUMINAMATH_GPT_part_one_solution_set_part_two_range_of_a_l1114_111488

def f (x : ℝ) (a : ℝ) : ℝ := |x - a| - 2

theorem part_one_solution_set (a : ℝ) (h : a = 1) : { x : ℝ | f x a + |2 * x - 3| > 0 } = { x : ℝ | x > 2 ∨ x < 2 / 3 } := 
sorry

theorem part_two_range_of_a : (∃ x : ℝ, f x (a) > |x - 3|) ↔ (a < 1 ∨ a > 5) :=
sorry

end NUMINAMATH_GPT_part_one_solution_set_part_two_range_of_a_l1114_111488


namespace NUMINAMATH_GPT_find_dads_dimes_l1114_111493

variable (original_dimes mother_dimes total_dimes dad_dimes : ℕ)

def proof_problem (original_dimes mother_dimes total_dimes dad_dimes : ℕ) : Prop :=
  original_dimes = 7 ∧
  mother_dimes = 4 ∧
  total_dimes = 19 ∧
  total_dimes = original_dimes + mother_dimes + dad_dimes

theorem find_dads_dimes (h : proof_problem 7 4 19 8) : dad_dimes = 8 :=
sorry

end NUMINAMATH_GPT_find_dads_dimes_l1114_111493


namespace NUMINAMATH_GPT_probability_of_circle_l1114_111470

theorem probability_of_circle :
  let numCircles := 4
  let numSquares := 3
  let numTriangles := 3
  let totalFigures := numCircles + numSquares + numTriangles
  let probability := numCircles / totalFigures
  probability = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_circle_l1114_111470


namespace NUMINAMATH_GPT_inequality_proof_l1114_111495

theorem inequality_proof (a b t : ℝ) (h₀ : 0 < t) (h₁ : t < 1) (h₂ : a * b > 0) : 
  (a^2 / t^3) + (b^2 / (1 - t^3)) ≥ (a + b)^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1114_111495


namespace NUMINAMATH_GPT_determinant_problem_l1114_111467

variables {p q r s : ℝ}

theorem determinant_problem
  (h : p * s - q * r = 5) :
  p * (4 * r + 2 * s) - (4 * p + 2 * q) * r = 10 := 
sorry

end NUMINAMATH_GPT_determinant_problem_l1114_111467


namespace NUMINAMATH_GPT_lines_intersect_at_single_point_l1114_111466

theorem lines_intersect_at_single_point (m : ℚ)
    (h1 : ∃ x y : ℚ, y = 4 * x - 8 ∧ y = -3 * x + 9)
    (h2 : ∀ x y : ℚ, (y = 4 * x - 8 ∧ y = -3 * x + 9) → (y = 2 * x + m)) :
    m = -22/7 := by
  sorry

end NUMINAMATH_GPT_lines_intersect_at_single_point_l1114_111466


namespace NUMINAMATH_GPT_max_discount_rate_l1114_111407

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end NUMINAMATH_GPT_max_discount_rate_l1114_111407


namespace NUMINAMATH_GPT_train_length_l1114_111482

theorem train_length (L : ℝ) (h1 : (L + 120) / 60 = L / 20) : L = 60 := 
sorry

end NUMINAMATH_GPT_train_length_l1114_111482


namespace NUMINAMATH_GPT_sample_size_l1114_111418

theorem sample_size (w_under30 : ℕ) (w_30to40 : ℕ) (w_40plus : ℕ) (sample_40plus : ℕ) (total_sample : ℕ) :
  w_under30 = 2400 →
  w_30to40 = 3600 →
  w_40plus = 6000 →
  sample_40plus = 60 →
  total_sample = 120 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sample_size_l1114_111418


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1114_111403

theorem repeating_decimal_sum :
  let a := (2 : ℚ) / 3
  let b := (2 : ℚ) / 9
  let c := (4 : ℚ) / 9
  a + b - c = (4 : ℚ) / 9 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l1114_111403


namespace NUMINAMATH_GPT_represent_nat_as_combinations_l1114_111419

theorem represent_nat_as_combinations (n : ℕ) :
  ∃ x y z : ℕ,
  (0 ≤ x ∧ x < y ∧ y < z ∨ 0 = x ∧ x = y ∧ y < z) ∧
  (n = Nat.choose x 1 + Nat.choose y 2 + Nat.choose z 3) :=
sorry

end NUMINAMATH_GPT_represent_nat_as_combinations_l1114_111419


namespace NUMINAMATH_GPT_school_basketballs_l1114_111473

theorem school_basketballs (n_classes n_basketballs_per_class total_basketballs : ℕ)
  (h1 : n_classes = 7)
  (h2 : n_basketballs_per_class = 7)
  (h3 : total_basketballs = n_classes * n_basketballs_per_class) :
  total_basketballs = 49 :=
sorry

end NUMINAMATH_GPT_school_basketballs_l1114_111473


namespace NUMINAMATH_GPT_math_problem_proof_l1114_111489

theorem math_problem_proof : 
  ((9 - 8 + 7) ^ 2 * 6 + 5 - 4 ^ 2 * 3 + 2 ^ 3 - 1) = 347 := 
by sorry

end NUMINAMATH_GPT_math_problem_proof_l1114_111489


namespace NUMINAMATH_GPT_squared_difference_l1114_111421

theorem squared_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 10) : (x - y)^2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_squared_difference_l1114_111421


namespace NUMINAMATH_GPT_vertex_of_parabola_l1114_111432

theorem vertex_of_parabola :
  ∃ (a b c : ℝ), 
      (4 * a - 2 * b + c = 9) ∧ 
      (16 * a + 4 * b + c = 9) ∧ 
      (49 * a + 7 * b + c = 16) ∧ 
      (-b / (2 * a) = 1) :=
by {
  -- we need to provide the proof here; sorry is a placeholder
  sorry
}

end NUMINAMATH_GPT_vertex_of_parabola_l1114_111432


namespace NUMINAMATH_GPT_moles_of_CH4_l1114_111442

theorem moles_of_CH4 (moles_Be2C moles_H2O : ℕ) (balanced_equation : 1 * Be2C + 4 * H2O = 2 * CH4 + 2 * BeOH2) 
  (h_Be2C : moles_Be2C = 3) (h_H2O : moles_H2O = 12) : 
  6 = 2 * moles_Be2C :=
by
  sorry

end NUMINAMATH_GPT_moles_of_CH4_l1114_111442


namespace NUMINAMATH_GPT_log_ratios_l1114_111429

noncomputable def ratio_eq : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem log_ratios
  {a b : ℝ}
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : Real.log a / Real.log 8 = Real.log b / Real.log 18)
  (h4 : Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) :
  b / a = ratio_eq :=
sorry

end NUMINAMATH_GPT_log_ratios_l1114_111429


namespace NUMINAMATH_GPT_find_m_l1114_111422

theorem find_m (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) 
  (hS : ∀ n, S n = n^2 - 6 * n) :
  (forall m, (5 < a m ∧ a m < 8) → m = 7)
:= 
by
  sorry

end NUMINAMATH_GPT_find_m_l1114_111422


namespace NUMINAMATH_GPT_jan_more_miles_than_ian_l1114_111452

noncomputable def distance_diff (d t s : ℝ) : ℝ :=
  let han_distance := (s + 10) * (t + 2)
  let jan_distance := (s + 15) * (t + 3)
  jan_distance - (d + 100)

theorem jan_more_miles_than_ian {d t s : ℝ} (H : d = s * t) (H_han : d + 100 = (s + 10) * (t + 2)) : distance_diff d t s = 165 :=
by {
  sorry
}

end NUMINAMATH_GPT_jan_more_miles_than_ian_l1114_111452


namespace NUMINAMATH_GPT_profit_percentage_is_correct_l1114_111433

-- Definitions for the given conditions
def SP : ℝ := 850
def Profit : ℝ := 255
def CP : ℝ := SP - Profit

-- The target proof statement
theorem profit_percentage_is_correct : 
  (Profit / CP) * 100 = 42.86 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_correct_l1114_111433


namespace NUMINAMATH_GPT_cost_per_tissue_l1114_111412

-- Annalise conditions
def boxes : ℕ := 10
def packs_per_box : ℕ := 20
def tissues_per_pack : ℕ := 100
def total_spent : ℝ := 1000

-- Definition for total packs and total tissues
def total_packs : ℕ := boxes * packs_per_box
def total_tissues : ℕ := total_packs * tissues_per_pack

-- The math problem: Prove the cost per tissue
theorem cost_per_tissue : (total_spent / total_tissues) = 0.05 := by
  sorry

end NUMINAMATH_GPT_cost_per_tissue_l1114_111412


namespace NUMINAMATH_GPT_int_coeffs_square_sum_l1114_111484

theorem int_coeffs_square_sum (a b c d e f : ℤ)
  (h : ∀ x, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 767 := 
sorry

end NUMINAMATH_GPT_int_coeffs_square_sum_l1114_111484


namespace NUMINAMATH_GPT_smoothie_one_serving_ingredients_in_cups_containers_needed_l1114_111453

theorem smoothie_one_serving_ingredients_in_cups :
  (0.2 + 0.1 + 0.2 + 1 * 0.125 + 2 * 0.0625 + 0.5).round = 1.25.round := sorry

theorem containers_needed :
  (5 * 1.25 / 1.5).ceil = 5 := sorry

end NUMINAMATH_GPT_smoothie_one_serving_ingredients_in_cups_containers_needed_l1114_111453


namespace NUMINAMATH_GPT_least_possible_number_of_straight_lines_l1114_111472

theorem least_possible_number_of_straight_lines :
  ∀ (segments : Fin 31 → (Fin 2 → ℝ)), 
  (∀ i j, i ≠ j → (segments i 0 = segments j 0) ∧ (segments i 1 = segments j 1) → false) →
  ∃ (lines_count : ℕ), lines_count = 16 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_number_of_straight_lines_l1114_111472


namespace NUMINAMATH_GPT_problem1_problem2_l1114_111483

-- Proof for Problem 1
theorem problem1 : (99^2 + 202*99 + 101^2) = 40000 := 
by {
  -- proof
  sorry
}

-- Proof for Problem 2
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : ((1 / (x - 1) - 2) / ((2 * x - 3) / (x^2 - 1))) = -x - 1 :=
by {
  -- proof
  sorry
}

end NUMINAMATH_GPT_problem1_problem2_l1114_111483


namespace NUMINAMATH_GPT_students_count_l1114_111420

theorem students_count (x y : ℕ) (h1 : 3 * x + 20 = y) (h2 : 4 * x - 25 = y) : x = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_count_l1114_111420


namespace NUMINAMATH_GPT_solve_inequality_l1114_111405

theorem solve_inequality (x : ℝ) : -1/3 * x + 1 ≤ -5 → x ≥ 18 := 
  sorry

end NUMINAMATH_GPT_solve_inequality_l1114_111405


namespace NUMINAMATH_GPT_complement_of_45_is_45_l1114_111410

def angle_complement (A : Real) : Real :=
  90 - A

theorem complement_of_45_is_45:
  angle_complement 45 = 45 :=
by
  sorry

end NUMINAMATH_GPT_complement_of_45_is_45_l1114_111410


namespace NUMINAMATH_GPT_laura_charges_for_truck_l1114_111435

theorem laura_charges_for_truck : 
  ∀ (car_wash suv_wash truck_wash total_amount num_suvs num_trucks num_cars : ℕ),
  car_wash = 5 →
  suv_wash = 7 →
  num_suvs = 5 →
  num_trucks = 5 →
  num_cars = 7 →
  total_amount = 100 →
  car_wash * num_cars + suv_wash * num_suvs + truck_wash * num_trucks = total_amount →
  truck_wash = 6 :=
by
  intros car_wash suv_wash truck_wash total_amount num_suvs num_trucks num_cars h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_laura_charges_for_truck_l1114_111435


namespace NUMINAMATH_GPT_integer_k_values_l1114_111461

theorem integer_k_values (a b k : ℝ) (m : ℝ) (ha : a > 0) (hb : b > 0) (hba_int : ∃ n : ℤ, n ≠ 0 ∧ b = (n : ℝ) * a) 
  (hA : a = a * k + m) (hB : 8 * b = b * k + m) : k = 9 ∨ k = 15 := 
by
  sorry

end NUMINAMATH_GPT_integer_k_values_l1114_111461


namespace NUMINAMATH_GPT_actual_distance_travelled_l1114_111431

theorem actual_distance_travelled :
  ∃ (D : ℝ), (D / 10 = (D + 20) / 14) ∧ D = 50 :=
by
  sorry

end NUMINAMATH_GPT_actual_distance_travelled_l1114_111431


namespace NUMINAMATH_GPT_g_of_neg5_eq_651_over_16_l1114_111459

def f (x : ℝ) : ℝ := 4 * x + 6

def g (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 7

theorem g_of_neg5_eq_651_over_16 : g (-5) = 651 / 16 := by
  sorry

end NUMINAMATH_GPT_g_of_neg5_eq_651_over_16_l1114_111459


namespace NUMINAMATH_GPT_range_of_f_l1114_111460

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+1) + 3

theorem range_of_f : Set.range f = Set.Ici 2 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_f_l1114_111460


namespace NUMINAMATH_GPT_unique_triple_solution_zero_l1114_111438

theorem unique_triple_solution_zero (m n k : ℝ) :
  (∃ x : ℝ, m * x ^ 2 + n = 0) ∧
  (∃ x : ℝ, n * x ^ 2 + k = 0) ∧
  (∃ x : ℝ, k * x ^ 2 + m = 0) ↔
  (m = 0 ∧ n = 0 ∧ k = 0) := 
sorry

end NUMINAMATH_GPT_unique_triple_solution_zero_l1114_111438


namespace NUMINAMATH_GPT_general_formula_for_a_n_l1114_111409

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Defining a_n as a function of n assuming it's an arithmetic sequence.
noncomputable def a (x : ℝ) (n : ℕ) : ℝ :=
  if x = 1 then 2 * n - 4 else if x = 3 then 4 - 2 * n else 0

theorem general_formula_for_a_n (x : ℝ) (n : ℕ) (h1 : a x 1 = f (x + 1))
  (h2 : a x 2 = 0) (h3 : a x 3 = f (x - 1)) :
  (x = 1 → a x n = 2 * n - 4) ∧ (x = 3 → a x n = 4 - 2 * n) :=
by sorry

end NUMINAMATH_GPT_general_formula_for_a_n_l1114_111409


namespace NUMINAMATH_GPT_minimum_days_bacteria_count_exceeds_500_l1114_111464

theorem minimum_days_bacteria_count_exceeds_500 :
  ∃ n : ℕ, 4 * 3^n > 500 ∧ ∀ m : ℕ, m < n → 4 * 3^m ≤ 500 :=
by
  sorry

end NUMINAMATH_GPT_minimum_days_bacteria_count_exceeds_500_l1114_111464


namespace NUMINAMATH_GPT_total_tweets_correct_l1114_111494

-- Define the rates at which Polly tweets under different conditions
def happy_rate : ℕ := 18
def hungry_rate : ℕ := 4
def mirror_rate : ℕ := 45

-- Define the durations of each activity
def happy_duration : ℕ := 20
def hungry_duration : ℕ := 20
def mirror_duration : ℕ := 20

-- Compute the total number of tweets
def total_tweets : ℕ := happy_rate * happy_duration + hungry_rate * hungry_duration + mirror_rate * mirror_duration

-- Statement to prove
theorem total_tweets_correct : total_tweets = 1340 := by
  sorry

end NUMINAMATH_GPT_total_tweets_correct_l1114_111494


namespace NUMINAMATH_GPT_tank_overflow_time_l1114_111465

noncomputable def pipeARate : ℚ := 1 / 32
noncomputable def pipeBRate : ℚ := 3 * pipeARate
noncomputable def combinedRate (rateA rateB : ℚ) : ℚ := rateA + rateB

theorem tank_overflow_time : 
  combinedRate pipeARate pipeBRate = 1 / 8 ∧ (1 / combinedRate pipeARate pipeBRate = 8) :=
by
  sorry

end NUMINAMATH_GPT_tank_overflow_time_l1114_111465


namespace NUMINAMATH_GPT_fraction_of_capacity_l1114_111450

theorem fraction_of_capacity
    (bus_capacity : ℕ)
    (x : ℕ)
    (first_pickup : ℕ)
    (second_pickup : ℕ)
    (unable_to_board : ℕ)
    (bus_full : bus_capacity = x + (second_pickup - unable_to_board))
    (carry_fraction : x / bus_capacity = 3 / 5) : 
    true := 
sorry

end NUMINAMATH_GPT_fraction_of_capacity_l1114_111450


namespace NUMINAMATH_GPT_exp_problem_l1114_111427

theorem exp_problem (a b c : ℕ) (H1 : a = 1000) (H2 : b = 1000^1000) (H3 : c = 500^1000) :
  a * b / c = 2^1001 * 500 :=
sorry

end NUMINAMATH_GPT_exp_problem_l1114_111427


namespace NUMINAMATH_GPT_find_p_over_q_at_0_l1114_111480

noncomputable def p (x : ℝ) := 3 * (x - 4) * (x - 1)
noncomputable def q (x : ℝ) := (x + 3) * (x - 1) * (x - 4)

theorem find_p_over_q_at_0 : (p 0) / (q 0) = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_p_over_q_at_0_l1114_111480


namespace NUMINAMATH_GPT_distance_blown_by_storm_l1114_111469

-- Definitions based on conditions
def speed : ℤ := 30
def time_travelled : ℤ := 20
def distance_travelled := speed * time_travelled
def total_distance := 2 * distance_travelled
def fractional_distance_left := total_distance / 3

-- Final statement to prove
theorem distance_blown_by_storm : distance_travelled - fractional_distance_left = 200 := by
  sorry

end NUMINAMATH_GPT_distance_blown_by_storm_l1114_111469


namespace NUMINAMATH_GPT_percent_decrease_internet_cost_l1114_111479

theorem percent_decrease_internet_cost :
  ∀ (initial_cost final_cost : ℝ), initial_cost = 120 → final_cost = 45 → 
  ((initial_cost - final_cost) / initial_cost) * 100 = 62.5 :=
by
  intros initial_cost final_cost h_initial h_final
  sorry

end NUMINAMATH_GPT_percent_decrease_internet_cost_l1114_111479
