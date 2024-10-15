import Mathlib

namespace NUMINAMATH_GPT_expected_number_of_hits_l1784_178400

variable (W : ℝ) (n : ℕ)
def expected_hits (W : ℝ) (n : ℕ) : ℝ := W * n

theorem expected_number_of_hits :
  W = 0.75 → n = 40 → expected_hits W n = 30 :=
by
  intros hW hn
  rw [hW, hn]
  norm_num
  sorry

end NUMINAMATH_GPT_expected_number_of_hits_l1784_178400


namespace NUMINAMATH_GPT_frequency_distribution_necessary_l1784_178425

/-- Definition of the necessity to use Frequency Distribution to understand 
the proportion of first-year high school students in the city whose height 
falls within a certain range -/
def necessary_for_proportion (A B C D : Prop) : Prop := D

theorem frequency_distribution_necessary (A B C D : Prop) :
  necessary_for_proportion A B C D ↔ D :=
by
  sorry

end NUMINAMATH_GPT_frequency_distribution_necessary_l1784_178425


namespace NUMINAMATH_GPT_opposite_points_l1784_178457

theorem opposite_points (A B : ℝ) (h1 : A = -B) (h2 : A < B) (h3 : abs (A - B) = 6.4) : A = -3.2 ∧ B = 3.2 :=
by
  sorry

end NUMINAMATH_GPT_opposite_points_l1784_178457


namespace NUMINAMATH_GPT_modulus_of_z_l1784_178429

noncomputable def z : ℂ := (Complex.I / (1 + 2 * Complex.I))

theorem modulus_of_z : Complex.abs z = (Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_GPT_modulus_of_z_l1784_178429


namespace NUMINAMATH_GPT_difference_is_divisible_by_p_l1784_178409

-- Lean 4 statement equivalent to the math proof problem
theorem difference_is_divisible_by_p
  (a : ℕ → ℕ) (p : ℕ) (d : ℕ)
  (h_prime : Nat.Prime p)
  (h_prog : ∀ i j: ℕ, 1 ≤ i ∧ i ≤ p ∧ 1 ≤ j ∧ j ≤ p ∧ i < j → a j = a (i + 1) + (j - 1) * d)
  (h_a_gt_p : a 1 > p)
  (h_arith_prog_primes : ∀ i: ℕ, 1 ≤ i ∧ i ≤ p → Nat.Prime (a i)) :
  d % p = 0 := sorry

end NUMINAMATH_GPT_difference_is_divisible_by_p_l1784_178409


namespace NUMINAMATH_GPT_geometric_sequence_third_sixth_term_l1784_178427

theorem geometric_sequence_third_sixth_term (a r : ℝ) 
  (h3 : a * r^2 = 18) 
  (h6 : a * r^5 = 162) : 
  a = 2 ∧ r = 3 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_third_sixth_term_l1784_178427


namespace NUMINAMATH_GPT_valid_N_count_l1784_178430

theorem valid_N_count : 
  (∃ n : ℕ, 0 < n ∧ (49 % (n + 3) = 0) ∧ (49 / (n + 3)) % 2 = 1) → 
  (∃ count : ℕ, count = 2) :=
sorry

end NUMINAMATH_GPT_valid_N_count_l1784_178430


namespace NUMINAMATH_GPT_polynomial_operations_l1784_178471

-- Define the given options for M, N, and P
def A (x : ℝ) : ℝ := 2 * x - 6
def B (x : ℝ) : ℝ := 3 * x + 5
def C (x : ℝ) : ℝ := -5 * x - 21

-- Define the original expression and its simplified form
def original_expr (M N : ℝ → ℝ) (x : ℝ) : ℝ :=
  2 * M x - 3 * N x

-- Define the simplified target expression
def simplified_expr (x : ℝ) : ℝ := -5 * x - 21

theorem polynomial_operations :
  ∀ (M N P : ℝ → ℝ),
  (original_expr M N = simplified_expr) →
  (M = A ∨ N = B ∨ P = C)
:= by
  intros M N P H
  sorry

end NUMINAMATH_GPT_polynomial_operations_l1784_178471


namespace NUMINAMATH_GPT_area_of_region_l1784_178411

def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (abs p.1 - p.1)^2 + (abs p.2 - p.2)^2 ≤ 16 ∧ 2 * p.2 + p.1 ≤ 0}

noncomputable def area : ℝ := sorry

theorem area_of_region : area = 5 + Real.pi := by
  sorry

end NUMINAMATH_GPT_area_of_region_l1784_178411


namespace NUMINAMATH_GPT_square_of_equal_side_of_inscribed_triangle_l1784_178479

theorem square_of_equal_side_of_inscribed_triangle :
  ∀ (x y : ℝ),
  (x^2 + 9 * y^2 = 9) →
  ((x = 0) → (y = 1)) →
  ((x ≠ 0) → y = (x + 1)) →
  square_of_side = (324 / 25) :=
by
  intros x y hEllipse hVertex hSlope
  sorry

end NUMINAMATH_GPT_square_of_equal_side_of_inscribed_triangle_l1784_178479


namespace NUMINAMATH_GPT_initial_flour_amount_l1784_178451

theorem initial_flour_amount (initial_flour : ℕ) (additional_flour : ℕ) (total_flour : ℕ) 
  (h1 : additional_flour = 4) (h2 : total_flour = 16) (h3 : initial_flour + additional_flour = total_flour) :
  initial_flour = 12 := 
by 
  sorry

end NUMINAMATH_GPT_initial_flour_amount_l1784_178451


namespace NUMINAMATH_GPT_lcm_24_90_l1784_178491

theorem lcm_24_90 : lcm 24 90 = 360 :=
by 
-- lcm is the least common multiple of 24 and 90.
-- lcm 24 90 is defined as 360.
sorry

end NUMINAMATH_GPT_lcm_24_90_l1784_178491


namespace NUMINAMATH_GPT_incorrect_correlation_statement_l1784_178465

/--
  The correlation coefficient measures the degree of linear correlation between two variables. 
  The linear correlation coefficient is a quantity whose absolute value is less than 1. 
  Furthermore, the larger its absolute value, the greater the degree of correlation.

  Let r be the sample correlation coefficient.

  We want to prove that the statement "D: |r| ≥ 1, and the closer |r| is to 1, the greater the degree of correlation" 
  is incorrect.
-/
theorem incorrect_correlation_statement (r : ℝ) (h1 : |r| ≤ 1) : ¬ (|r| ≥ 1) :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_incorrect_correlation_statement_l1784_178465


namespace NUMINAMATH_GPT_min_max_area_of_CDM_l1784_178499

theorem min_max_area_of_CDM (x y z : ℕ) (h1 : 2 * x + y = 4) (h2 : 2 * y + z = 8) :
  z = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_max_area_of_CDM_l1784_178499


namespace NUMINAMATH_GPT_unique_triple_l1784_178403

theorem unique_triple (x y z : ℤ) (h₁ : x + y = z) (h₂ : y + z = x) (h₃ : z + x = y) :
  (x = 0) ∧ (y = 0) ∧ (z = 0) :=
sorry

end NUMINAMATH_GPT_unique_triple_l1784_178403


namespace NUMINAMATH_GPT_solve_porters_transportation_l1784_178473

variable (x : ℝ)

def porters_transportation_equation : Prop :=
  (5000 / x = 8000 / (x + 600))

theorem solve_porters_transportation (x : ℝ) (h₁ : 600 > 0) (h₂ : x > 0):
  porters_transportation_equation x :=
sorry

end NUMINAMATH_GPT_solve_porters_transportation_l1784_178473


namespace NUMINAMATH_GPT_restaurant_bill_l1784_178440

theorem restaurant_bill
    (t : ℝ)
    (h1 : ∀ k : ℝ, k = 9 * (t / 10 + 3)) :
    t = 270 :=
by
    sorry

end NUMINAMATH_GPT_restaurant_bill_l1784_178440


namespace NUMINAMATH_GPT_lamps_on_after_n2_minus_1_lamps_on_after_n2_minus_n_plus_1_l1784_178432

def lamps_on_again (n : ℕ) (steps : ℕ → Bool → Bool) : ∃ M : ℕ, ∀ s, (s ≥ M) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

theorem lamps_on_after_n2_minus_1 (n : ℕ) (k : ℕ) (hk : n = 2^k) (steps : ℕ → Bool → Bool) : 
∀ s, (s ≥ n^2 - 1) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

theorem lamps_on_after_n2_minus_n_plus_1 (n : ℕ) (k : ℕ) (hk : n = 2^k + 1) (steps : ℕ → Bool → Bool) : 
∀ s, (s ≥ n^2 - n + 1) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

end NUMINAMATH_GPT_lamps_on_after_n2_minus_1_lamps_on_after_n2_minus_n_plus_1_l1784_178432


namespace NUMINAMATH_GPT_cones_sold_l1784_178477

-- Define the conditions
variable (milkshakes : Nat)
variable (cones : Nat)

-- Assume the given conditions
axiom h1 : milkshakes = 82
axiom h2 : milkshakes = cones + 15

-- State the theorem to prove
theorem cones_sold : cones = 67 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cones_sold_l1784_178477


namespace NUMINAMATH_GPT_pies_can_be_made_l1784_178419

def total_apples : Nat := 51
def apples_handout : Nat := 41
def apples_per_pie : Nat := 5

theorem pies_can_be_made :
  ((total_apples - apples_handout) / apples_per_pie) = 2 := by
  sorry

end NUMINAMATH_GPT_pies_can_be_made_l1784_178419


namespace NUMINAMATH_GPT_haley_seeds_total_l1784_178439

-- Conditions
def seeds_in_big_garden : ℕ := 35
def small_gardens : ℕ := 7
def seeds_per_small_garden : ℕ := 3

-- Question rephrased as a problem with the correct answer
theorem haley_seeds_total : seeds_in_big_garden + small_gardens * seeds_per_small_garden = 56 := by
  sorry

end NUMINAMATH_GPT_haley_seeds_total_l1784_178439


namespace NUMINAMATH_GPT_daily_production_l1784_178431

theorem daily_production (x : ℕ) (hx1 : 216 / x > 4)
  (hx2 : 3 * x + (x + 8) * ((216 / x) - 4) = 232) : 
  x = 24 := by
sorry

end NUMINAMATH_GPT_daily_production_l1784_178431


namespace NUMINAMATH_GPT_rectangle_area_in_triangle_l1784_178421

theorem rectangle_area_in_triangle (c k y : ℝ) (h1 : c > 0) (h2 : k > 0) (h3 : 0 < y) (h4 : y < k) : 
  ∃ A : ℝ, A = y * ((c * (k - y)) / k) := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_in_triangle_l1784_178421


namespace NUMINAMATH_GPT_luke_piles_of_quarters_l1784_178448

theorem luke_piles_of_quarters (Q D : ℕ) 
  (h1 : Q = D) -- number of piles of quarters equals number of piles of dimes
  (h2 : 3 * Q + 3 * D = 30) -- total number of coins is 30
  : Q = 5 :=
by
  sorry

end NUMINAMATH_GPT_luke_piles_of_quarters_l1784_178448


namespace NUMINAMATH_GPT_total_distance_karl_drove_l1784_178458

theorem total_distance_karl_drove :
  ∀ (consumption_rate miles_per_gallon : ℕ) 
    (tank_capacity : ℕ) 
    (initial_gas : ℕ) 
    (distance_leg1 : ℕ) 
    (purchased_gas : ℕ) 
    (remaining_gas : ℕ)
    (final_gas : ℕ),
  consumption_rate = 25 → 
  tank_capacity = 18 →
  initial_gas = 12 →
  distance_leg1 = 250 →
  purchased_gas = 10 →
  remaining_gas = initial_gas - distance_leg1 / consumption_rate + purchased_gas →
  final_gas = remaining_gas - distance_leg2 / consumption_rate →
  remaining_gas - distance_leg2 / consumption_rate = final_gas →
  distance_leg2 = (initial_gas - remaining_gas + purchased_gas - final_gas) * miles_per_gallon →
  miles_per_gallon = 25 →
  distance_leg2 + distance_leg1 = 475 :=
sorry

end NUMINAMATH_GPT_total_distance_karl_drove_l1784_178458


namespace NUMINAMATH_GPT_computation_of_sqrt_expr_l1784_178459

theorem computation_of_sqrt_expr : 
  (Real.sqrt ((52 : ℝ) * 51 * 50 * 49 + 1) = 2549) := 
by
  sorry

end NUMINAMATH_GPT_computation_of_sqrt_expr_l1784_178459


namespace NUMINAMATH_GPT_word_value_at_l1784_178484

def letter_value (c : Char) : ℕ :=
  if 'A' ≤ c ∧ c ≤ 'Z' then c.toNat - 'A'.toNat + 1 else 0

def word_value (s : String) : ℕ :=
  let sum_values := s.toList.map letter_value |>.sum
  sum_values * s.length

theorem word_value_at : word_value "at" = 42 := by
  sorry

end NUMINAMATH_GPT_word_value_at_l1784_178484


namespace NUMINAMATH_GPT_calculate_c_l1784_178435

-- Define the given equation as a hypothesis
theorem calculate_c (a b k c : ℝ) (h : (1 / (k * a) - 1 / (k * b) = 1 / c)) :
  c = k * a * b / (b - a) :=
by
  sorry

end NUMINAMATH_GPT_calculate_c_l1784_178435


namespace NUMINAMATH_GPT_circular_park_diameter_factor_l1784_178433

theorem circular_park_diameter_factor (r : ℝ) :
  (π * (3 * r)^2) / (π * r^2) = 9 ∧ (2 * π * (3 * r)) / (2 * π * r) = 3 :=
by
  sorry

end NUMINAMATH_GPT_circular_park_diameter_factor_l1784_178433


namespace NUMINAMATH_GPT_cannot_fit_rectangle_l1784_178416

theorem cannot_fit_rectangle 
  (w1 h1 : ℕ) (w2 h2 : ℕ) 
  (h1_pos : 0 < h1) (w1_pos : 0 < w1)
  (h2_pos : 0 < h2) (w2_pos : 0 < w2) :
  w1 = 5 → h1 = 6 → w2 = 3 → h2 = 8 →
  ¬(w2 ≤ w1 ∧ h2 ≤ h1) :=
by
  intros H1 W1 H2 W2
  sorry

end NUMINAMATH_GPT_cannot_fit_rectangle_l1784_178416


namespace NUMINAMATH_GPT_symmetric_line_proof_l1784_178401

-- Define the given lines
def line_l (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0
def axis_of_symmetry (x y : ℝ) : Prop := x + y = 0

-- Define the final symmetric line to be proved
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0

-- State the theorem
theorem symmetric_line_proof (x y : ℝ) : 
  (line_l (-y) (-x)) → 
  axis_of_symmetry x y → 
  symmetric_line x y := 
sorry

end NUMINAMATH_GPT_symmetric_line_proof_l1784_178401


namespace NUMINAMATH_GPT_correct_statement_l1784_178418

-- Definitions as per conditions
def P1 : Prop := ∃ x : ℝ, x^2 = 64 ∧ abs x ^ 3 = 2
def P2 : Prop := ∀ x : ℝ, x = 0 → (¬∃ y, y * x = 1 ∧ -x = y)
def P3 : Prop := ∀ x y : ℝ, x + y = 0 → abs x / abs y = -1
def P4 : Prop := ∀ x a : ℝ, abs x + x = a → a > 0

-- The proof problem
theorem correct_statement : P1 ∧ ¬P2 ∧ ¬P3 ∧ ¬P4 := by
  sorry

end NUMINAMATH_GPT_correct_statement_l1784_178418


namespace NUMINAMATH_GPT_no_solution_inequality_l1784_178463

theorem no_solution_inequality (a : ℝ) : (¬ ∃ x : ℝ, |x - 5| + |x + 3| < a) ↔ a ≤ 8 := 
sorry

end NUMINAMATH_GPT_no_solution_inequality_l1784_178463


namespace NUMINAMATH_GPT_simplify_fraction_l1784_178447

open Complex

theorem simplify_fraction :
  (3 + 3 * I) / (-1 + 3 * I) = -1.2 - 1.2 * I :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1784_178447


namespace NUMINAMATH_GPT_candidate_votes_percentage_l1784_178424

-- Conditions
variables {P : ℝ} 
variables (totalVotes : ℝ := 8000)
variables (differenceVotes : ℝ := 2400)

-- Proof Problem
theorem candidate_votes_percentage (h : ((P / 100) * totalVotes + ((P / 100) * totalVotes + differenceVotes) = totalVotes)) : P = 35 :=
by
  sorry

end NUMINAMATH_GPT_candidate_votes_percentage_l1784_178424


namespace NUMINAMATH_GPT_avg_amount_lost_per_loot_box_l1784_178468

-- Define the conditions
def cost_per_loot_box : ℝ := 5
def avg_value_of_items : ℝ := 3.5
def total_amount_spent : ℝ := 40

-- Define the goal
theorem avg_amount_lost_per_loot_box : 
  (total_amount_spent / cost_per_loot_box) * (cost_per_loot_box - avg_value_of_items) / (total_amount_spent / cost_per_loot_box) = 1.5 := 
by 
  sorry

end NUMINAMATH_GPT_avg_amount_lost_per_loot_box_l1784_178468


namespace NUMINAMATH_GPT_composite_expr_l1784_178498

open Nat

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

theorem composite_expr (n : ℕ) : n ≥ 2 ↔ is_composite (3^(2*n + 1) - 2^(2*n + 1) - 6^n) :=
sorry

end NUMINAMATH_GPT_composite_expr_l1784_178498


namespace NUMINAMATH_GPT_discount_difference_l1784_178483

theorem discount_difference (x : ℝ) (h1 : x = 8000) : 
  (x * 0.7) - ((x * 0.8) * 0.9) = 160 :=
by
  rw [h1]
  sorry

end NUMINAMATH_GPT_discount_difference_l1784_178483


namespace NUMINAMATH_GPT_range_of_2x_plus_y_l1784_178497

-- Given that positive numbers x and y satisfy this equation:
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x + y + 4 * x * y = 15 / 2

-- Define the range for 2x + y
def range_2x_plus_y (x y : ℝ) : ℝ :=
  2 * x + y

-- State the theorem.
theorem range_of_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : satisfies_equation x y) :
  3 ≤ range_2x_plus_y x y :=
by
  sorry

end NUMINAMATH_GPT_range_of_2x_plus_y_l1784_178497


namespace NUMINAMATH_GPT_triangle_inequalities_l1784_178413

-- Definitions of the variables
variables {ABC : Triangle} {r : ℝ} {R : ℝ} {ρ_a ρ_b ρ_c : ℝ} {P_a P_b P_c : ℝ}

-- Problem statement based on given conditions and proof requirement
theorem triangle_inequalities (ABC : Triangle) (r : ℝ) (R : ℝ) (ρ_a ρ_b ρ_c : ℝ) (P_a P_b P_c : ℝ) :
  (3/2) * r ≤ ρ_a + ρ_b + ρ_c ∧ ρ_a + ρ_b + ρ_c ≤ (3/4) * R ∧ 4 * r ≤ P_a + P_b + P_c ∧ P_a + P_b + P_c ≤ 2 * R :=
  sorry

end NUMINAMATH_GPT_triangle_inequalities_l1784_178413


namespace NUMINAMATH_GPT_henry_final_money_l1784_178456

def initial_money : ℝ := 11.75
def received_from_relatives : ℝ := 18.50
def found_in_card : ℝ := 5.25
def spent_on_game : ℝ := 10.60
def donated_to_charity : ℝ := 3.15

theorem henry_final_money :
  initial_money + received_from_relatives + found_in_card - spent_on_game - donated_to_charity = 21.75 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_henry_final_money_l1784_178456


namespace NUMINAMATH_GPT_dana_jellybeans_l1784_178488

noncomputable def jellybeans_in_dana_box (alex_capacity : ℝ) (mul_factor : ℝ) : ℝ :=
  let alex_volume := 1 * 1 * 1.5
  let dana_volume := mul_factor * mul_factor * (mul_factor * 1.5)
  let volume_ratio := dana_volume / alex_volume
  volume_ratio * alex_capacity

theorem dana_jellybeans
  (alex_capacity : ℝ := 150)
  (mul_factor : ℝ := 3) :
  jellybeans_in_dana_box alex_capacity mul_factor = 4050 :=
by
  rw [jellybeans_in_dana_box]
  simp
  sorry

end NUMINAMATH_GPT_dana_jellybeans_l1784_178488


namespace NUMINAMATH_GPT_triangle_inequality_range_l1784_178443

theorem triangle_inequality_range {a b c : ℝ} (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  1 ≤ (a^2 + b^2 + c^2) / (a * b + b * c + c * a) ∧ (a^2 + b^2 + c^2) / (a * b + b * c + c * a) < 2 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_inequality_range_l1784_178443


namespace NUMINAMATH_GPT_one_is_sum_of_others_l1784_178467

theorem one_is_sum_of_others {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : |a - b| ≥ c) (h2 : |b - c| ≥ a) (h3 : |c - a| ≥ b) :
    a = b + c ∨ b = a + c ∨ c = a + b :=
sorry

end NUMINAMATH_GPT_one_is_sum_of_others_l1784_178467


namespace NUMINAMATH_GPT_bobs_improvement_percentage_l1784_178460

-- Define the conditions
def bobs_time_minutes := 10
def bobs_time_seconds := 40
def sisters_time_minutes := 10
def sisters_time_seconds := 8

-- Convert minutes and seconds to total seconds
def bobs_total_time_seconds := bobs_time_minutes * 60 + bobs_time_seconds
def sisters_total_time_seconds := sisters_time_minutes * 60 + sisters_time_seconds

-- Define the improvement needed and calculate the percentage improvement
def improvement_needed := bobs_total_time_seconds - sisters_total_time_seconds
def percentage_improvement := (improvement_needed / bobs_total_time_seconds) * 100

-- The lean statement to prove
theorem bobs_improvement_percentage : percentage_improvement = 5 := by
  sorry

end NUMINAMATH_GPT_bobs_improvement_percentage_l1784_178460


namespace NUMINAMATH_GPT_part1_part2_part3_l1784_178422

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 1

theorem part1 (a : ℝ) (x : ℝ) (h : 0 < x) :
  (a ≤ 0 → (∀ x > 0, f a x < 0)) ∧
  (a > 0 → (∀ x ∈ Set.Ioo 0 a, f a x > 0) ∧ (∀ x ∈ Set.Ioi a, f a x < 0)) :=
sorry

theorem part2 {a : ℝ} : (∀ x > 0, f a x ≤ 0) → a = 1 :=
sorry

theorem part3 (n : ℕ) (h : 0 < n) :
  (1 + 1 / n : ℝ)^n < Real.exp 1 ∧ Real.exp 1 < (1 + 1 / n : ℝ)^(n + 1) :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1784_178422


namespace NUMINAMATH_GPT_age_ratio_l1784_178472

theorem age_ratio (s a : ℕ) (h1 : s - 3 = 2 * (a - 3)) (h2 : s - 7 = 3 * (a - 7)) :
  ∃ x : ℕ, (x = 23) ∧ (s + x) / (a + x) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l1784_178472


namespace NUMINAMATH_GPT_spencer_walk_distance_l1784_178446

theorem spencer_walk_distance :
  let distance_house_library := 0.3
  let distance_library_post_office := 0.1
  let total_distance := 0.8
  (total_distance - (distance_house_library + distance_library_post_office)) = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_spencer_walk_distance_l1784_178446


namespace NUMINAMATH_GPT_ball_count_l1784_178478

theorem ball_count (r b y : ℕ) 
  (h1 : b + y = 9) 
  (h2 : r + y = 5) 
  (h3 : r + b = 6) : 
  r + b + y = 10 := 
  sorry

end NUMINAMATH_GPT_ball_count_l1784_178478


namespace NUMINAMATH_GPT_work_completed_by_a_l1784_178423

theorem work_completed_by_a (a b : ℕ) (work_in_30_days : a + b = 4 * 30) (a_eq_3b : a = 3 * b) : (120 / a) = 40 :=
by
  -- Given a + b = 120 and a = 3 * b, prove that 120 / a = 40
  sorry

end NUMINAMATH_GPT_work_completed_by_a_l1784_178423


namespace NUMINAMATH_GPT_Eldora_total_cost_l1784_178455

-- Conditions
def paper_clip_cost : ℝ := 1.85
def index_card_cost : ℝ := 3.95 -- from Finn's purchase calculation
def total_cost (clips : ℝ) (cards : ℝ) (clip_price : ℝ) (card_price : ℝ) : ℝ :=
  (clips * clip_price) + (cards * card_price)

theorem Eldora_total_cost :
  total_cost 15 7 paper_clip_cost index_card_cost = 55.40 :=
by
  sorry

end NUMINAMATH_GPT_Eldora_total_cost_l1784_178455


namespace NUMINAMATH_GPT_find_some_number_l1784_178410

theorem find_some_number (a : ℕ) (some_number : ℕ)
  (h1 : a = 105)
  (h2 : a ^ 3 = 21 * 35 * some_number * 35) :
  some_number = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_some_number_l1784_178410


namespace NUMINAMATH_GPT_max_S_n_l1784_178485

/-- Arithmetic sequence proof problem -/
theorem max_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 + a 3 + a 5 = 15)
  (h2 : a 2 + a 4 + a 6 = 0)
  (d : ℝ) (h3 : ∀ n, a (n + 1) = a n + d) :
  (∃ n, S n = 30) :=
sorry

end NUMINAMATH_GPT_max_S_n_l1784_178485


namespace NUMINAMATH_GPT_total_allocation_is_1800_l1784_178437

-- Definitions from conditions.
def part_value (amount_food : ℕ) (ratio_food : ℕ) : ℕ :=
  amount_food / ratio_food

def total_parts (ratio_household : ℕ) (ratio_food : ℕ) (ratio_misc : ℕ) : ℕ :=
  ratio_household + ratio_food + ratio_misc

def total_amount (part_value : ℕ) (total_parts : ℕ) : ℕ :=
  part_value * total_parts

-- Given conditions
def ratio_household := 5
def ratio_food := 4
def ratio_misc := 1
def amount_food := 720

-- Prove the total allocation
theorem total_allocation_is_1800 
  (amount_food : ℕ := 720) 
  (ratio_household : ℕ := 5) 
  (ratio_food : ℕ := 4) 
  (ratio_misc : ℕ := 1) : 
  total_amount (part_value amount_food ratio_food) (total_parts ratio_household ratio_food ratio_misc) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_total_allocation_is_1800_l1784_178437


namespace NUMINAMATH_GPT_solve_system_l1784_178415

theorem solve_system (a b c : ℝ)
  (h1 : b + c = 10 - 4 * a)
  (h2 : a + c = -16 - 4 * b)
  (h3 : a + b = 9 - 4 * c) :
  2 * a + 2 * b + 2 * c = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1784_178415


namespace NUMINAMATH_GPT_car_speed_is_80_l1784_178407

theorem car_speed_is_80 : ∃ v : ℝ, (1 / v * 3600 = 45) ∧ (v = 80) :=
by
  sorry

end NUMINAMATH_GPT_car_speed_is_80_l1784_178407


namespace NUMINAMATH_GPT_negation_of_proposition_l1784_178453

theorem negation_of_proposition (x : ℝ) (h : 2 * x + 1 ≤ 0) : ¬ (2 * x + 1 ≤ 0) ↔ 2 * x + 1 > 0 := 
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1784_178453


namespace NUMINAMATH_GPT_cost_of_500_cookies_in_dollars_l1784_178428

def cost_in_cents (cookies : Nat) (cost_per_cookie : Nat) : Nat :=
  cookies * cost_per_cookie

def cents_to_dollars (cents : Nat) : Nat :=
  cents / 100

theorem cost_of_500_cookies_in_dollars :
  cents_to_dollars (cost_in_cents 500 2) = 10
:= by
  sorry

end NUMINAMATH_GPT_cost_of_500_cookies_in_dollars_l1784_178428


namespace NUMINAMATH_GPT_sufficient_conditions_for_equation_l1784_178438

theorem sufficient_conditions_for_equation 
  (a b c : ℤ) :
  (a = b ∧ b = c + 1) ∨ (a = c ∧ b - 1 = c) →
  a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_conditions_for_equation_l1784_178438


namespace NUMINAMATH_GPT_rectangular_to_polar_coordinates_l1784_178404

theorem rectangular_to_polar_coordinates :
  ∃ r θ, (r > 0) ∧ (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ (r, θ) = (5, 7 * Real.pi / 4) :=
by
  sorry

end NUMINAMATH_GPT_rectangular_to_polar_coordinates_l1784_178404


namespace NUMINAMATH_GPT_find_m_l1784_178486

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_find_m_l1784_178486


namespace NUMINAMATH_GPT_smallest_base_b_l1784_178492

theorem smallest_base_b (k : ℕ) (hk : k = 7) : ∃ (b : ℕ), b = 64 ∧ b^k > 4^20 := by
  sorry

end NUMINAMATH_GPT_smallest_base_b_l1784_178492


namespace NUMINAMATH_GPT_miles_collection_height_l1784_178489

-- Definitions based on conditions
def pages_per_inch_miles : ℕ := 5
def pages_per_inch_daphne : ℕ := 50
def daphne_height_inches : ℕ := 25
def longest_collection_pages : ℕ := 1250

-- Theorem to prove the height of Miles's book collection.
theorem miles_collection_height :
  (longest_collection_pages / pages_per_inch_miles) = 250 := by sorry

end NUMINAMATH_GPT_miles_collection_height_l1784_178489


namespace NUMINAMATH_GPT_laura_weekly_mileage_l1784_178414

-- Define the core conditions

-- Distance to school per round trip (house <-> school)
def school_trip_distance : ℕ := 20

-- Number of trips to school per week
def school_trips_per_week : ℕ := 7

-- Distance to supermarket: 10 miles farther than school
def extra_distance_to_supermarket : ℕ := 10
def supermarket_trip_distance : ℕ := school_trip_distance + 2 * extra_distance_to_supermarket

-- Number of trips to supermarket per week
def supermarket_trips_per_week : ℕ := 2

-- Calculate the total weekly distance
def total_distance_per_week : ℕ := 
  (school_trips_per_week * school_trip_distance) +
  (supermarket_trips_per_week * supermarket_trip_distance)

-- Theorem to prove the total distance Laura drives per week
theorem laura_weekly_mileage :
  total_distance_per_week = 220 := by
  sorry

end NUMINAMATH_GPT_laura_weekly_mileage_l1784_178414


namespace NUMINAMATH_GPT_remainder_sum_l1784_178445

theorem remainder_sum (a b c d : ℕ) 
  (h_a : a % 30 = 15) 
  (h_b : b % 30 = 7) 
  (h_c : c % 30 = 22) 
  (h_d : d % 30 = 6) : 
  (a + b + c + d) % 30 = 20 := 
by
  sorry

end NUMINAMATH_GPT_remainder_sum_l1784_178445


namespace NUMINAMATH_GPT_find_abc_l1784_178495

theorem find_abc (a b c : ℝ) (ha : a + 1 / b = 5)
                             (hb : b + 1 / c = 2)
                             (hc : c + 1 / a = 3) :
    a * b * c = 10 + 3 * Real.sqrt 11 :=
sorry

end NUMINAMATH_GPT_find_abc_l1784_178495


namespace NUMINAMATH_GPT_ratio_of_ages_l1784_178490

variable (T N : ℕ)
variable (sum_ages : T = T) -- This is tautological based on the given condition; we can consider it a given sum
variable (age_condition : T - N = 3 * (T - 3 * N))

theorem ratio_of_ages (T N : ℕ) (sum_ages : T = T) (age_condition : T - N = 3 * (T - 3 * N)) : T / N = 4 :=
sorry

end NUMINAMATH_GPT_ratio_of_ages_l1784_178490


namespace NUMINAMATH_GPT_only_n_divides_2_n_minus_1_l1784_178476

theorem only_n_divides_2_n_minus_1 :
  ∀ n : ℕ, n ≥ 1 → (n ∣ (2^n - 1)) → n = 1 :=
by
  sorry

end NUMINAMATH_GPT_only_n_divides_2_n_minus_1_l1784_178476


namespace NUMINAMATH_GPT_total_fruits_is_78_l1784_178420

def oranges_louis : Nat := 5
def apples_louis : Nat := 3

def oranges_samantha : Nat := 8
def apples_samantha : Nat := 7

def oranges_marley : Nat := 2 * oranges_louis
def apples_marley : Nat := 3 * apples_samantha

def oranges_edward : Nat := 3 * oranges_louis
def apples_edward : Nat := 3 * apples_louis

def total_fruits_louis : Nat := oranges_louis + apples_louis
def total_fruits_samantha : Nat := oranges_samantha + apples_samantha
def total_fruits_marley : Nat := oranges_marley + apples_marley
def total_fruits_edward : Nat := oranges_edward + apples_edward

def total_fruits_all : Nat :=
  total_fruits_louis + total_fruits_samantha + total_fruits_marley + total_fruits_edward

theorem total_fruits_is_78 : total_fruits_all = 78 := by
  sorry

end NUMINAMATH_GPT_total_fruits_is_78_l1784_178420


namespace NUMINAMATH_GPT_percentage_increase_in_area_is_96_l1784_178442

theorem percentage_increase_in_area_is_96 :
  let r₁ := 5
  let r₃ := 7
  let A (r : ℝ) := Real.pi * r^2
  ((A r₃ - A r₁) / A r₁) * 100 = 96 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_area_is_96_l1784_178442


namespace NUMINAMATH_GPT_three_minus_pi_to_zero_l1784_178426

theorem three_minus_pi_to_zero : (3 - Real.pi) ^ 0 = 1 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_three_minus_pi_to_zero_l1784_178426


namespace NUMINAMATH_GPT_tan_315_eq_neg_one_l1784_178462

theorem tan_315_eq_neg_one : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end NUMINAMATH_GPT_tan_315_eq_neg_one_l1784_178462


namespace NUMINAMATH_GPT_min_value_f_l1784_178482

noncomputable def f (x : ℝ) : ℝ := (8^x + 5) / (2^x + 1)

theorem min_value_f : ∃ x : ℝ, f x = 3 :=
sorry

end NUMINAMATH_GPT_min_value_f_l1784_178482


namespace NUMINAMATH_GPT_letters_posting_ways_l1784_178496

theorem letters_posting_ways :
  let mailboxes := 4
  let letters := 3
  (mailboxes ^ letters) = 64 :=
by
  let mailboxes := 4
  let letters := 3
  show (mailboxes ^ letters) = 64
  sorry

end NUMINAMATH_GPT_letters_posting_ways_l1784_178496


namespace NUMINAMATH_GPT_oranges_picked_l1784_178449

theorem oranges_picked (total_oranges second_tree third_tree : ℕ) 
    (h1 : total_oranges = 260) 
    (h2 : second_tree = 60) 
    (h3 : third_tree = 120) : 
    total_oranges - (second_tree + third_tree) = 80 := by 
  sorry

end NUMINAMATH_GPT_oranges_picked_l1784_178449


namespace NUMINAMATH_GPT_nicky_cristina_race_l1784_178406

theorem nicky_cristina_race :
  ∀ (head_start t : ℕ), ∀ (cristina_speed nicky_speed time_nicky_run : ℝ),
  head_start = 12 →
  cristina_speed = 5 →
  nicky_speed = 3 →
  ((cristina_speed * t) = (nicky_speed * t + nicky_speed * head_start)) →
  time_nicky_run = head_start + t →
  time_nicky_run = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_nicky_cristina_race_l1784_178406


namespace NUMINAMATH_GPT_p_necessary_not_sufficient_for_p_and_q_l1784_178436

-- Define statements p and q as propositions
variables (p q : Prop)

-- Prove that "p is true" is a necessary but not sufficient condition for "p ∧ q is true"
theorem p_necessary_not_sufficient_for_p_and_q : (p ∧ q → p) ∧ (p → ¬ (p ∧ q)) :=
by sorry

end NUMINAMATH_GPT_p_necessary_not_sufficient_for_p_and_q_l1784_178436


namespace NUMINAMATH_GPT_min_value_frac_l1784_178470

theorem min_value_frac (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 1) :
  (1 / x + 1 / (3 * y)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_l1784_178470


namespace NUMINAMATH_GPT_different_picture_size_is_correct_l1784_178475

-- Define constants and conditions
def memory_card_picture_capacity := 3000
def single_picture_size := 8
def different_picture_capacity := 4000

-- Total memory card capacity in megabytes
def total_capacity := memory_card_picture_capacity * single_picture_size

-- The size of each different picture
def different_picture_size := total_capacity / different_picture_capacity

-- The theorem to prove
theorem different_picture_size_is_correct :
  different_picture_size = 6 := 
by
  -- We include 'sorry' here to bypass actual proof
  sorry

end NUMINAMATH_GPT_different_picture_size_is_correct_l1784_178475


namespace NUMINAMATH_GPT_quadratic_root_exists_l1784_178461

theorem quadratic_root_exists (a b c : ℝ) (ha : a ≠ 0)
  (h1 : a * (0.6 : ℝ)^2 + b * 0.6 + c = -0.04)
  (h2 : a * (0.7 : ℝ)^2 + b * 0.7 + c = 0.19) :
  ∃ x : ℝ, 0.6 < x ∧ x < 0.7 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_exists_l1784_178461


namespace NUMINAMATH_GPT_mileage_per_gallon_l1784_178408

noncomputable def car_mileage (distance: ℝ) (gasoline: ℝ) : ℝ :=
  distance / gasoline

theorem mileage_per_gallon :
  car_mileage 190 4.75 = 40 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_mileage_per_gallon_l1784_178408


namespace NUMINAMATH_GPT_floral_shop_bouquets_l1784_178493

theorem floral_shop_bouquets (T : ℕ) 
  (h1 : 12 + T + T / 3 = 60) 
  (hT : T = 36) : T / 12 = 3 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_floral_shop_bouquets_l1784_178493


namespace NUMINAMATH_GPT_find_d_l1784_178417

open Real

theorem find_d (a b c d : ℝ) 
  (h : a^3 + b^3 + c^3 + a^2 + b^2 + 1 = d^2 + d + sqrt (a + b + c - 2 * d)) : 
  d = 1 ∨ d = -(4 / 3) :=
sorry

end NUMINAMATH_GPT_find_d_l1784_178417


namespace NUMINAMATH_GPT_police_officer_can_catch_gangster_l1784_178466

theorem police_officer_can_catch_gangster
  (a : ℝ) -- length of the side of the square
  (v_police : ℝ) -- maximum speed of the police officer
  (v_gangster : ℝ) -- maximum speed of the gangster
  (h_gangster_speed : v_gangster = 2.9 * v_police) :
  ∃ (t : ℝ), t ≥ 0 ∧ (a / (2 * v_police)) = t := sorry

end NUMINAMATH_GPT_police_officer_can_catch_gangster_l1784_178466


namespace NUMINAMATH_GPT_garden_perimeter_l1784_178434

theorem garden_perimeter
  (a b : ℝ)
  (h1: a^2 + b^2 = 225)
  (h2: a * b = 54) :
  2 * (a + b) = 2 * Real.sqrt 333 :=
by
  sorry

end NUMINAMATH_GPT_garden_perimeter_l1784_178434


namespace NUMINAMATH_GPT_find_m_l1784_178454

theorem find_m (x y m : ℝ) (opp_sign: y = -x) 
  (h1 : 4 * x + 2 * y = 3 * m) 
  (h2 : 3 * x + y = m + 2) : 
  m = 1 :=
by 
  -- Placeholder for the steps to prove the theorem
  sorry

end NUMINAMATH_GPT_find_m_l1784_178454


namespace NUMINAMATH_GPT_even_natural_number_factors_count_l1784_178405

def is_valid_factor (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a ≤ 3 ∧ 
  0 ≤ b ∧ b ≤ 2 ∧ 
  0 ≤ c ∧ c ≤ 2 ∧ 
  a + b + c ≤ 4

noncomputable def count_valid_factors : ℕ :=
  Nat.card { x : ℕ × ℕ × ℕ // is_valid_factor x.1 x.2.1 x.2.2 }

theorem even_natural_number_factors_count : count_valid_factors = 15 := 
  sorry

end NUMINAMATH_GPT_even_natural_number_factors_count_l1784_178405


namespace NUMINAMATH_GPT_estimated_total_fish_l1784_178494

-- Let's define the conditions first
def total_fish_marked := 100
def second_catch_total := 200
def marked_in_second_catch := 5

-- The variable representing the total number of fish in the pond
variable (x : ℕ)

-- The theorem stating that given the conditions, the total number of fish is 4000
theorem estimated_total_fish
  (h1 : total_fish_marked = 100)
  (h2 : second_catch_total = 200)
  (h3 : marked_in_second_catch = 5)
  (h4 : (marked_in_second_catch : ℝ) / second_catch_total = (total_fish_marked : ℝ) / x) :
  x = 4000 := 
sorry

end NUMINAMATH_GPT_estimated_total_fish_l1784_178494


namespace NUMINAMATH_GPT_find_income_of_deceased_l1784_178402
noncomputable def income_of_deceased_member 
  (members_before : ℕ) (avg_income_before : ℕ) 
  (members_after : ℕ) (avg_income_after : ℕ) : ℕ :=
  (members_before * avg_income_before) - (members_after * avg_income_after)

theorem find_income_of_deceased 
  (members_before avg_income_before members_after avg_income_after : ℕ) :
  income_of_deceased_member 4 840 3 650 = 1410 :=
by
  -- Problem claims income_of_deceased_member = Income before - Income after
  sorry

end NUMINAMATH_GPT_find_income_of_deceased_l1784_178402


namespace NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_same_plane_l1784_178412

variables (m n : Line) (α : Plane)

-- Define conditions using Lean's logical constructs
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry -- This would define the condition
def parallel_lines (l1 l2 : Line) : Prop := sorry -- This would define the condition

-- The statement to prove
theorem lines_parallel_if_perpendicular_to_same_plane 
  (h1 : perpendicular_to_plane m α) 
  (h2 : perpendicular_to_plane n α) : 
  parallel_lines m n :=
sorry

end NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_same_plane_l1784_178412


namespace NUMINAMATH_GPT_both_true_sufficient_but_not_necessary_for_either_l1784_178464

variable (p q : Prop)

theorem both_true_sufficient_but_not_necessary_for_either:
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  sorry

end NUMINAMATH_GPT_both_true_sufficient_but_not_necessary_for_either_l1784_178464


namespace NUMINAMATH_GPT_probability_is_correct_l1784_178469

def num_red : ℕ := 7
def num_green : ℕ := 9
def num_yellow : ℕ := 10
def num_blue : ℕ := 5
def num_purple : ℕ := 3

def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue + num_purple

def num_blue_or_purple : ℕ := num_blue + num_purple

-- Probability of selecting a blue or purple jelly bean
def probability_blue_or_purple : ℚ := num_blue_or_purple / total_jelly_beans

theorem probability_is_correct :
  probability_blue_or_purple = 4 / 17 := sorry

end NUMINAMATH_GPT_probability_is_correct_l1784_178469


namespace NUMINAMATH_GPT_car_owners_without_motorcycles_l1784_178450

theorem car_owners_without_motorcycles (total_adults cars motorcycles no_vehicle : ℕ) 
  (h1 : total_adults = 560) (h2 : cars = 520) (h3 : motorcycles = 80) (h4 : no_vehicle = 10) : 
  cars - (total_adults - no_vehicle - cars - motorcycles) = 470 := 
by
  sorry

end NUMINAMATH_GPT_car_owners_without_motorcycles_l1784_178450


namespace NUMINAMATH_GPT_range_of_a_l1784_178481

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 + 2 * x - 3

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → x < 4 → y < 4 → f a x ≤ f a y) ↔ (- (1/4:ℝ) ≤ a ∧ a ≤ 0) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1784_178481


namespace NUMINAMATH_GPT_prod_ineq_min_value_l1784_178480

theorem prod_ineq_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) ≥ 216 := by
  sorry

end NUMINAMATH_GPT_prod_ineq_min_value_l1784_178480


namespace NUMINAMATH_GPT_find_3a_plus_4b_l1784_178487

noncomputable def g (x : ℝ) := 3 * x - 6

noncomputable def f_inverse (x : ℝ) := (3 * x - 2) / 2

noncomputable def f (x : ℝ) (a b : ℝ) := a * x + b

theorem find_3a_plus_4b (a b : ℝ) (h1 : ∀ x, g x = 2 * f_inverse x - 4) (h2 : ∀ x, f_inverse (f x a b) = x) :
  3 * a + 4 * b = 14 / 3 :=
sorry

end NUMINAMATH_GPT_find_3a_plus_4b_l1784_178487


namespace NUMINAMATH_GPT_zoo_children_count_l1784_178444

theorem zoo_children_count:
  ∀ (C : ℕ), 
  (10 * C + 16 * 10 = 220) → 
  C = 6 :=
by
  intro C
  intro h
  sorry

end NUMINAMATH_GPT_zoo_children_count_l1784_178444


namespace NUMINAMATH_GPT_martin_less_than_43_l1784_178452

variable (C K M : ℕ)

-- Conditions
def campbell_correct := C = 35
def kelsey_correct := K = C + 8
def martin_fewer := M < K

-- Conclusion we want to prove
theorem martin_less_than_43 (h1 : campbell_correct C) (h2 : kelsey_correct C K) (h3 : martin_fewer K M) : M < 43 := 
by {
  sorry
}

end NUMINAMATH_GPT_martin_less_than_43_l1784_178452


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l1784_178474

/--
Given a triangle ABC where BC is 12 cm and the height from A
perpendicular to BC is 15 cm, prove that the area of the triangle is 90 cm^2.
-/
theorem area_of_triangle_ABC (BC : ℝ) (hA : ℝ) (h_BC : BC = 12) (h_hA : hA = 15) : 
  1/2 * BC * hA = 90 := 
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l1784_178474


namespace NUMINAMATH_GPT_estimated_white_balls_l1784_178441

noncomputable def estimate_white_balls (total_balls draws white_draws : ℕ) : ℕ :=
  total_balls * white_draws / draws

theorem estimated_white_balls (total_balls draws white_draws : ℕ) (h1 : total_balls = 20)
  (h2 : draws = 100) (h3 : white_draws = 40) :
  estimate_white_balls total_balls draws white_draws = 8 := by
  sorry

end NUMINAMATH_GPT_estimated_white_balls_l1784_178441
