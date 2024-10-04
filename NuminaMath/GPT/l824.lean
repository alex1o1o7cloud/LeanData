import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Divisibility.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.LCM
import Mathlib.Algebra.Quotient
import Mathlib.Algebra.Square
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Partition
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Bitwise
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Perm
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.ArithmeticFunctions
import Mathlib.Probability.Normal
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.SolveByElim
import Mathlib.Topology.Instances.Real
import ProbabilityTheory.Basic
import mathlib

namespace length_of_shorter_train_l824_824858

noncomputable def relativeSpeedInMS (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  (speed1_kmh + speed2_kmh) * (5 / 18)

noncomputable def totalDistanceCovered (relativeSpeed_ms time_s : ℝ) : ℝ :=
  relativeSpeed_ms * time_s

noncomputable def lengthOfShorterTrain (longerTrainLength_m time_s : ℝ) (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let relativeSpeed_ms := relativeSpeedInMS speed1_kmh speed2_kmh
  let totalDistance := totalDistanceCovered relativeSpeed_ms time_s
  totalDistance - longerTrainLength_m

theorem length_of_shorter_train :
  lengthOfShorterTrain 160 10.07919366450684 60 40 = 117.8220467912412 := 
sorry

end length_of_shorter_train_l824_824858


namespace deriv_y1_is_correct_deriv_y2_is_correct_deriv_y3_is_correct_deriv_f_is_correct_l824_824959

-- Define function y1 and its derivative
def y1 (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)
def dy1_dx (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem deriv_y1_is_correct : ∀ x : ℝ, deriv y1 x = dy1_dx x := by
  sorry

-- Define function y2 and its derivative
def y2 (x : ℝ) : ℝ := x^2 * sin x
def dy2_dx (x : ℝ) : ℝ := 2 * x * sin x + x^2 * cos x

theorem deriv_y2_is_correct : ∀ x : ℝ, deriv y2 x = dy2_dx x := by
  sorry

-- Define function y3 and its derivative
def y3 (x : ℝ) : ℝ := (exp x + 1) / (exp x - 1)
def dy3_dx (x : ℝ) : ℝ := -2 * exp x / (exp x - 1)^2

theorem deriv_y3_is_correct : ∀ x : ℝ, deriv y3 x = dy3_dx x := by
  sorry

-- Define function f and its derivative
def f (x : ℝ) : ℝ := exp x / (x - 2)
def df_dx (x : ℝ) : ℝ := exp x * (x - 3) / (x - 2)^2

theorem deriv_f_is_correct : ∀ x : ℝ, deriv f x = df_dx x := by
  sorry

end deriv_y1_is_correct_deriv_y2_is_correct_deriv_y3_is_correct_deriv_f_is_correct_l824_824959


namespace AIME_F6_is_7_l824_824318

noncomputable def F : ℕ → ℕ
| 1 := 1
| 2 := 1
| 3 := 1
| (n + 1) := (F n * F (n - 1) + 1) / F (n - 2)

theorem AIME_F6_is_7 : F 6 = 7 :=
by
  sorry

end AIME_F6_is_7_l824_824318


namespace ratio_sprite_to_coke_l824_824354

theorem ratio_sprite_to_coke (total_drink : ℕ) (coke_ounces : ℕ) (mountain_dew_parts : ℕ)
  (parts_coke : ℕ) (parts_mountain_dew : ℕ) (total_parts : ℕ) :
  total_drink = 18 →
  coke_ounces = 6 →
  parts_coke = 2 →
  parts_mountain_dew = 3 →
  total_parts = parts_coke + parts_mountain_dew + ((total_drink - coke_ounces - (parts_mountain_dew * (coke_ounces / parts_coke))) / (coke_ounces / parts_coke)) →
  (total_drink - coke_ounces - (parts_mountain_dew * (coke_ounces / parts_coke))) / coke_ounces = 1 / 2 :=
by sorry

end ratio_sprite_to_coke_l824_824354


namespace find_cos_B_l824_824744

variables {α : Type*} [linear_ordered_field α] {a b c A B C : α}

-- Definitions of the variables and mathematical relationships between them
def sine_rule (a b c A B C : α) : Prop := 
  a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C

def angle_sum (A B C : α) : Prop := 
  A + B + C = 180

theorem find_cos_B (h1 : b * Real.cos C - c * Real.cos (A + C) = 3 * a * Real.cos B)
                   (h2 : sine_rule a b c A B C)
                   (h3 : angle_sum A B C) 
                   (non_zero : Real.sin A ≠ 0) : 
                   Real.cos B = 1 / 3 := 
by
  sorry

end find_cos_B_l824_824744


namespace evaluate_expression_l824_824476

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := by
  sorry

end evaluate_expression_l824_824476


namespace correct_minor_premise_l824_824867

-- Define the exponential function and the condition of it being an increasing function for a > 1
def exp_fun (a : ℝ) (x : ℝ) : ℝ := a^x

-- Condition: Exponential function y = a^x (a > 1) is increasing
def is_increasing {a : ℝ} (h : 1 < a) : Prop :=
  ∀ x y : ℝ, x < y → exp_fun a x < exp_fun a y

-- Condition: y = 2^x is an exponential function
def is_exponential (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, 1 < a ∧ ∀ x : ℝ, f x = a^x

-- Given that y = a^x (a > 1) is increasing and y = 2^x is an exponential function,
-- we will prove that y = 2^x is the correct minor premise.
theorem correct_minor_premise (a : ℝ) (h1 : 1 < a) (h2 : is_increasing h1) : 
  is_exponential (exp_fun 2) :=
sorry

end correct_minor_premise_l824_824867


namespace doughnut_machine_completion_time_l824_824506

-- Define the start time and the time when half the job is completed
def start_time := 8 * 60 -- 8:00 AM in minutes
def half_job_time := 10 * 60 + 30 -- 10:30 AM in minutes

-- Given the machine completes half of the day's job by 10:30 AM
-- Prove that the doughnut machine will complete the entire job by 1:00 PM
theorem doughnut_machine_completion_time :
  half_job_time - start_time = 150 → 
  (start_time + 2 * 150) % (24 * 60) = 13 * 60 :=
by
  sorry

end doughnut_machine_completion_time_l824_824506


namespace sqrt_500_least_integer_l824_824015

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824015


namespace p_plus_q_l824_824810

theorem p_plus_q (x : ℝ) (p q : ℕ) (hpq : p.gcd q = 1) 
  (h1 : Real.sec x + Real.tan x = 25 / 7) 
  (h2 : Real.csc x + Real.cot x = p / q) : p + q = 31 := by
  sorry

end p_plus_q_l824_824810


namespace sqrt_500_least_integer_l824_824013

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824013


namespace part1_a_part1_b_part2_a_part2_b_l824_824287

variable {x y c : ℝ}

noncomputable def f : ℝ → ℝ := sorry

axiom A1 : ∀ x y : ℝ, f (x + y) = f x + f y - 1
axiom A2 : ∀ x : ℝ, x > 0 → f x > 1
axiom A3 : f 3 = 4
axiom A4 : ∀ x y : ℝ, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom A5 : f 0 ≠ 0
axiom A6 : ∃ c : ℝ, c ≠ 0 ∧ f c = 0

theorem part1_a : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := sorry

theorem part1_b : ∀ x ∈ set.Icc 1 2, f x ≤ 3 ∧ f x ≥ 2 := sorry

theorem part2_a : ∀ x : ℝ, f x = f (-x) := sorry

theorem part2_b : ∃ p : ℝ, p = 4 * c ∧ (∀ x : ℝ, f x = f (x + p)) := sorry

end part1_a_part1_b_part2_a_part2_b_l824_824287


namespace normal_distribution_probability_l824_824262

noncomputable def xi : Type := sorry -- Define the random variable

def mean : ℝ := 2
def variance : ℝ := sorry -- placeholder for sigma^2
def P_xi_le_4 : ℝ := 0.84

-- Defining the problem: Prove that P(ξ ≤ 0) = 0.16 given the above conditions.
theorem normal_distribution_probability (xi : ℝ → ℝ) (h_norm : ∀ x, xi x = normalPDF mean variance x) :
  P_xi_le_4 = 0.84 → (xi 0) = 0.16 :=
by
  sorry

end normal_distribution_probability_l824_824262


namespace sum_of_digits_of_power_eight_2010_l824_824125

theorem sum_of_digits_of_power_eight_2010 :
  let n := 2010
  let a := 8
  let tens_digit := (a ^ n / 10) % 10
  let units_digit := a ^ n % 10
  tens_digit + units_digit = 1 :=
by
  sorry

end sum_of_digits_of_power_eight_2010_l824_824125


namespace geom_series_sum_l824_824124

theorem geom_series_sum :
  let a := (1 : ℝ) / 4,
      r := (1 : ℝ) / 4,
      n := 5 in
  (a * (1 - r^n) / (1 - r)) = 341 / 1024 :=
by
  sorry

end geom_series_sum_l824_824124


namespace speed_against_current_l824_824488

theorem speed_against_current (V_curr : ℝ) (V_man : ℝ) (V_curr_val : V_curr = 3.2) (V_man_with_curr : V_man = 15) :
    V_man - V_curr = 8.6 := 
by 
  rw [V_curr_val, V_man_with_curr]
  norm_num
  sorry

end speed_against_current_l824_824488


namespace symmetry_condition_sufficient_but_not_necessary_l824_824880

theorem symmetry_condition_sufficient_but_not_necessary (φ : ℝ) :
  (∀ x : ℝ, sin (x + φ) = sin (-x + φ)) ↔ (∃ k : ℤ, φ = π/2 + k * π) :=
sorry

end symmetry_condition_sufficient_but_not_necessary_l824_824880


namespace problem_l824_824478

theorem problem (a b : ℤ) (ha : a = 4) (hb : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := 
by
  -- Provide proof here
  sorry

end problem_l824_824478


namespace generalized_convex_inequality_l824_824377

variables {n : ℕ} {I : set ℝ} {x_i : ℕ → ℝ} {p_i : ℕ → ℝ} {f: ℝ → ℝ} {m r : ℝ}

-- Generalized convex function definition on interval I
def generalized_convex (f: ℝ → ℝ) (I: set ℝ): Prop := 
  ∀ (n: ℕ) (x_i: ℕ → ℝ) (p_i: ℕ → ℝ), 
  (∀ i < n, x_i i ∈ I) → (∀ i < n, p_i i > 0) → (∑ i in finset.range n, p_i i = 1) →
  f (∑ i in finset.range n, p_i i * x_i i) ≤ ∑ i in finset.range n, p_i i * f (x_i i)

-- Generalized mean
def generalized_mean (m: ℝ) (x_i: ℕ → ℝ) (p_i: ℕ → ℝ) (n: ℕ): ℝ :=
  (∑ i in finset.range n, p_i i * (x_i i) ^ m) ^ (1 / m)

-- The statement of the problem
theorem generalized_convex_inequality
  (h_gen_convex: generalized_convex f I)
  (hx_in_I: ∀ i < n, x_i i ∈ I)
  (hp_pos: ∀ i < n, p_i i > 0)
  (hm_geq_r: m ≥ r)
  (hp_sum_one: ∑ i in finset.range n, p_i i = 1) :
  generalized_mean m (λ i, f (x_i i)) p_i n ≥ f (generalized_mean r x_i p_i n) :=
sorry

end generalized_convex_inequality_l824_824377


namespace sum_4501st_and_4052nd_digits_l824_824899

theorem sum_4501st_and_4052nd_digits :
  (sum_of_nth_digits 4501 4052) = 13 :=
sorry

/-
Definitions used in conditions:
* sum_of_nth_digits: given a sequence where each positive integer n is repeated n times, the sum of the nth and mth digits can be defined in Lean. 
-/
-- Auxiliary definition to describe the sequence.
def sequence : ℕ → ℕ
| n := sorry -- The n-th element of the defined sequence.

-- Define the function to compute the sum of the nth and mth digits in the sequence.
def sum_of_nth_digits (n m : ℕ) : ℕ :=
sequence n + sequence m

end sum_4501st_and_4052nd_digits_l824_824899


namespace least_int_gt_sqrt_500_l824_824049

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824049


namespace least_integer_greater_than_sqrt_500_l824_824071

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824071


namespace return_trip_time_l824_824947

-- Define the given conditions
def run_time : ℕ := 20
def jog_time : ℕ := 10
def trip_time := run_time + jog_time
def multiplier: ℕ := 3

-- State the theorem
theorem return_trip_time : trip_time * multiplier = 90 := by
  sorry

end return_trip_time_l824_824947


namespace A_investment_l824_824903

variable (x : ℕ)
variable (A_share : ℕ := 3780)
variable (Total_profit : ℕ := 12600)
variable (B_invest : ℕ := 4200)
variable (C_invest : ℕ := 10500)

theorem A_investment :
  (A_share : ℝ) / (Total_profit : ℝ) = (x : ℝ) / (x + B_invest + C_invest) →
  x = 6300 :=
by
  sorry

end A_investment_l824_824903


namespace P_ξ_between_neg2_and_2_l824_824631

-- Let's assume that P is a function that gives the probability of an event.
noncomputable def P : Set ℝ → ℝ := sorry

-- Given a random variable ξ that follows a normal distribution N(0, σ^2)
axiom normal_dist (μ : ℝ) (σ : ℝ) (ξ : ℝ → ℝ) : Prop

-- Assume that ξ follows a normal distribution N(0, σ^2)
axiom ξ_is_normal (σ : ℝ) (ξ : ℝ → ℝ) : normal_dist 0 σ ξ

-- Assume that P(ξ > 2) = 0.023
axiom P_ξ_gt_2 (ξ : ℝ → ℝ) : P ({x : ℝ | ξ x > 2}) = 0.023

-- Prove that P(-2 <= ξ <= 2) = 0.954
theorem P_ξ_between_neg2_and_2 (σ : ℝ) (ξ : ℝ → ℝ) (h1 : normal_dist 0 σ ξ) (h2 : P ({x : ℝ | ξ x > 2}) = 0.023) :
  P ({x : ℝ | -2 ≤ ξ x ∧ ξ x ≤ 2}) = 0.954 :=
begin
  sorry
end

end P_ξ_between_neg2_and_2_l824_824631


namespace least_integer_greater_than_sqrt_500_l824_824069

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824069


namespace eric_return_home_time_l824_824945

-- Definitions based on conditions
def time_running_to_park : ℕ := 20
def time_jogging_to_park : ℕ := 10
def trip_to_park_time : ℕ := time_running_to_park + time_jogging_to_park
def return_time_multiplier : ℕ := 3

-- Statement of the problem
theorem eric_return_home_time : 
  return_time_multiplier * trip_to_park_time = 90 :=
by 
  -- Skipping proof steps
  sorry

end eric_return_home_time_l824_824945


namespace least_integer_greater_than_sqrt_500_l824_824089

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824089


namespace div_eq_four_l824_824695

theorem div_eq_four (x : ℝ) (h : 64 / x = 4) : x = 16 :=
sorry

end div_eq_four_l824_824695


namespace distance_between_foci_of_ellipse_l824_824969

theorem distance_between_foci_of_ellipse :
  let eq := 25 * x^2 + 100 * x + 4 * y^2 + 8 * y + 9 = 0 in
  let a² := 23.75 in
  let b² := 3.8 in
  let c := (a² - b²).sqrt in
  2 * c = 4 * Real.sqrt 5 :=
by
  sorry

end distance_between_foci_of_ellipse_l824_824969


namespace least_integer_greater_than_sqrt_500_l824_824079

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824079


namespace triangle_height_l824_824535

def width := 10
def length := 2 * width
def area_rectangle := width * length
def base_triangle := width

theorem triangle_height (h : ℝ) : (1 / 2) * base_triangle * h = area_rectangle → h = 40 :=
by
  sorry

end triangle_height_l824_824535


namespace sqrt_500_least_integer_l824_824020

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824020


namespace arithmetic_progressions_count_l824_824399

theorem arithmetic_progressions_count (d : ℕ) (h_d : d = 2) (S : ℕ) (h_S : S = 200) : 
  ∃ n : ℕ, n = 6 := sorry

end arithmetic_progressions_count_l824_824399


namespace room_height_l824_824599

theorem room_height (L B D : ℝ) (hL : L = 12) (hB : B = 8) (hD : D = 17) : ∃ H : ℝ, H = 9 ∧ (D^2 = L^2 + B^2 + H^2) :=
by
  have h₁ : L^2 = 12^2 := by rw hL; rfl;
  have h₂ : B^2 = 8^2 := by rw hB; rfl;
  have h₃ : D^2 = 17^2 := by rw hD; rfl;
  use 9
  split
  . sorry
  . rw [h₁, h₂, h₃]; norm_num; sorry

end room_height_l824_824599


namespace least_int_gt_sqrt_500_l824_824058

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824058


namespace binom_1300_2_eq_l824_824553

theorem binom_1300_2_eq : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_eq_l824_824553


namespace balls_in_boxes_l824_824312

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 7 → boxes = 4 → 
  (number_of_unique_distributions balls boxes = 11) := by
  intros balls boxes hb hc
  subst hb
  subst hc
  sorry

end balls_in_boxes_l824_824312


namespace spinsters_count_l824_824137

theorem spinsters_count (S C : ℕ) (h_ratio : S / C = 2 / 7) (h_diff : C = S + 55) : S = 22 :=
by
  sorry

end spinsters_count_l824_824137


namespace product_of_possible_N_l824_824916

theorem product_of_possible_N (N : ℕ) (M L : ℕ) :
  (M = L + N) →
  (M - 5 = L + N - 5) →
  (L + 3 = L + 3) →
  |(L + N - 5) - (L + 3)| = 2 →
  (10 * 6 = 60) :=
by
  sorry

end product_of_possible_N_l824_824916


namespace sum_2016_div_2016_l824_824264

noncomputable def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) + (-1) ^ n * a n = 2 * n)

def sum_first_n (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in finset.range (n + 1), a i

theorem sum_2016_div_2016 (a : ℕ → ℕ) 
  (h : seq a) :
  (sum_first_n a 2016 / 2016 = 1009) :=
sorry

end sum_2016_div_2016_l824_824264


namespace perpendicular_lines_l824_824273

variables (a : ℝ) 

def line_1 (a : ℝ) (x y : ℝ) : ℝ := a * x + y - 1
def line_2 (a : ℝ) (x y : ℝ) : ℝ := x + a * y - 1

theorem perpendicular_lines (a : ℝ) : 
  (line_1 a 0 1 = 0 ∧ line_2 a 1 0 = 0) ↔ a = 0 := 
by 
  sorry

end perpendicular_lines_l824_824273


namespace similar_triangles_l824_824674

open EuclideanGeometry

variables {Point : Type*} [EuclideanSpace Point]

/-- Given two segments AB and A'B' on a plane, construct a point O such that triangles AOB and A'OB' are similar. -/
def find_point_O (A B A' B' : Point) : Point :=
  let AB_ratio := dist A B / dist A' B'
  if AB_ratio = 1 then sorry else -- case when AB and A'B' are equal, special handling or different theorem may be required
  let GMT_A := {O : Point | dist A O / dist A' O = AB_ratio}
  let GMT_B := {O : Point | dist B O / dist B' O = AB_ratio}
  (GMT_A ∩ GMT_B).nonempty.some -- Intersection of GMT lines
  
theorem similar_triangles (A B A' B' O : Point) (h1 : dist A O / dist A' O = dist A B / dist A' B') (h2 : dist B O / dist B' O = dist A B / dist A' B') 
: ∃ O : Point, similar_triangle A B O A' B' O :=
begin
  use O,
  sorry
end

end similar_triangles_l824_824674


namespace parallel_lines_m_values_l824_824304

theorem parallel_lines_m_values (m : ℝ) :
  (∀ (x y : ℝ), (m - 2) * x - y + 5 = 0) ∧ 
  (∀ (x y : ℝ), (m - 2) * x + (3 - m) * y + 2 = 0) → 
  (m = 2 ∨ m = 4) :=
sorry

end parallel_lines_m_values_l824_824304


namespace intersection_and_value_l824_824622

-- Definitions of lines l1 and l2
def l1 (ρ θ : ℝ) : Prop := ρ * sin (θ - π/3) = sqrt 3

def l2 (x y : ℝ) : Prop := ∃ t : ℝ, x = -t ∧ y = sqrt 3 * t

-- Definition of the ellipse and angle conditions
def on_ellipse (A : ℝ × ℝ) : Prop := (A.1)^2 / 4 + (A.2)^2 = 1

def angle_condition (A B C : ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  (angle A O B = 2 * π / 3) ∧ (angle B O C = 2 * π / 3) ∧ (angle C O A = 2 * π / 3)

-- The proof problem
theorem intersection_and_value :
  (∃ (ρ θ : ℝ), l1 ρ θ ∧ l2 ρ θ ∧ (ρ = 2) ∧ (θ = 2 * π / 3)) ∧
  (∀ (A B C O : ℝ × ℝ), on_ellipse A ∧ on_ellipse B ∧ on_ellipse C ∧
   angle_condition A B C O →
  (1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) + 1 / (C.1^2 + C.2^2) = 15 / 8)) :=
by
  sorry

end intersection_and_value_l824_824622


namespace balls_in_boxes_wrong_positions_l824_824790

-- Define the total number of ways to place the balls in the boxes
def numberOfWays (n m : ℕ) : ℕ :=
  Nat.choose n m * 9

-- Prove that the total number of ways to place 9 balls into 9 boxes 
-- such that exactly 4 balls do not match the numbers of their respective boxes is 1134
theorem balls_in_boxes_wrong_positions :
  numberOfWays 9 5 = 1134 :=
by
  -- (5 balls placed correctly means exactly 4 balls placed incorrectly)
  unfold numberOfWays
  sorry

end balls_in_boxes_wrong_positions_l824_824790


namespace find_f_neg_m_l824_824260

variable {α : Type*} [Field α]

noncomputable def f (a b x : α) : α := a*x^3 + b*sin x + 2

theorem find_f_neg_m (a b m : α) (hf : f a b m = -5) : f a b (-m) = 9 := by
  sorry

end find_f_neg_m_l824_824260


namespace exponentiation_rule_l824_824252

theorem exponentiation_rule (a m : ℕ) (h : (a^2)^m = a^6) : m = 3 :=
by
  sorry

end exponentiation_rule_l824_824252


namespace permutation_count_A_before_B_l824_824546

theorem permutation_count_A_before_B :
  let speakers := ['A', 'B', 'C', 'D', 'E', 'F'],
      total_permutations := speakers.permutations.length in
  total_permutations / 2 = 360 := by
  let speakers := ['A', 'B', 'C', 'D', 'E', 'F']
  let total_permutations := speakers.permutations.length
  have total_permutations_eq_6_factorial : total_permutations = 6! := by sorry
  have half_of_permutations_eq_360 : total_permutations / 2 = 360 := by
    rw [total_permutations_eq_6_factorial]
    exact nat.factorial_succ 5
  exact half_of_permutations_eq_360

end permutation_count_A_before_B_l824_824546


namespace meet_third_and_fourth_l824_824331

-- Definitions for conditions a1 to a5
variables (L1 L2 L3 L4 : set (ℝ × ℝ)) -- Four lines in a plane
variables (P1 P2 P3 P4 : ℝ × ℝ → ℝ) -- Pedestrians moving along those lines

-- Non-parallel lines and no three lines intersect at a point
axiom a1 : ¬ ∃ k, ∀ x ∈ ℝ × ℝ, (x ∈ L1) ∧ (x ∈ L2) ∧ (x ∈ L3) ∧ (x ∈ L4)

-- Defining movement along lines
axiom a2 : ∀ t ∈ ℝ, (P1 t ∈ L1) ∧ (P2 t ∈ L2) ∧ (P3 t ∈ L3) ∧ (P4 t ∈ L4)

-- First pedestrian meets the second, third, and fourth pedestrian
axiom a3 : ∃ t1 t2 t3 t4 : ℝ, P1 t1 = P2 t2 ∧ P1 t1 = P3 t3 ∧ P1 t1 = P4 t4

-- Second pedestrian meets the third and the fourth pedestrian
axiom a4 : ∃ t5 t6 t7 : ℝ, P2 t5 = P3 t6 ∧ P2 t5 = P4 t7

-- Statement of the proof problem: third pedestrian meets the fourth pedestrian
theorem meet_third_and_fourth : ∃ t8 t9 : ℝ, P3 t8 = P4 t9 :=
  sorry

end meet_third_and_fourth_l824_824331


namespace sufficient_conditions_m_perp_β_l824_824366
noncomputable theory

-- Definitions for planes and lines
variables {Plane : Type} {Line : Type}

-- Predicate definitions for perpendicularity and intersections
variables [perpendicular : Plane → Plane → Prop]
variables [perpendicular_line_plane : Line → Plane → Prop]
variables [intersection_line : Plane → Plane → Line]

-- Conditions as definitions
def condition_1 (α β : Plane) (l m : Line) : Prop :=
  perpendicular α β ∧
  intersection_line α β = l ∧
  perpendicular_line_plane m l

def condition_2 (α β γ : Plane) (m : Line) : Prop :=
  intersection_line α γ = m ∧
  perpendicular α β ∧
  perpendicular γ β

def condition_3 (α β γ : Plane) (m : Line) : Prop :=
  perpendicular α γ ∧
  perpendicular β γ ∧
  perpendicular_line_plane m α

def condition_4 (α β : Plane) (m n : Line) : Prop :=
  perpendicular_line_plane n α ∧
  perpendicular_line_plane n β ∧
  perpendicular_line_plane m α

-- Theorem to prove that conditions 2 and 4 are sufficient for m ⊥ β
theorem sufficient_conditions_m_perp_β (α β γ : Plane) (l m n : Line) :
  condition_2 α β γ m ∨ condition_4 α β m n → perpendicular_line_plane m β :=
sorry

end sufficient_conditions_m_perp_β_l824_824366


namespace angle_in_third_quadrant_l824_824269

theorem angle_in_third_quadrant 
{α : ℝ} (h₁ : ∃ x y : ℝ, x = tan α ∧ y = cos α ∧ x > 0 ∧ y < 0): 
  ∃ q : ℕ, 
    (1 ≤ q ∧ q ≤ 4) ∧ 
    ((q = 3) ↔ 
      (0 <= α ∧ α < π ∧ cos α < 0 ∧ sin α > 0)
    ) := 
sorry

end angle_in_third_quadrant_l824_824269


namespace sum_of_roots_of_quadratic_eq_l824_824962

theorem sum_of_roots_of_quadratic_eq :
  ∀ x : ℝ, x^2 + 2023 * x - 2024 = 0 → 
  x = -2023 := 
sorry

end sum_of_roots_of_quadratic_eq_l824_824962


namespace domain_of_v_l824_824861

def domain_v (x : ℝ) : Prop :=
  x ≥ 2 ∧ x ≠ 5

theorem domain_of_v :
  {x : ℝ | domain_v x} = { x | 2 < x ∧ x < 5 } ∪ { x | 5 < x }
:= by
  sorry

end domain_of_v_l824_824861


namespace polynomial_remainder_l824_824605

-- Define the polynomials f and g
def f (x : ℚ) : ℚ := x^4 + 1
def g (x : ℚ) : ℚ := x^2 - 4*x + 6

-- State the theorem, that the remainder when f(x) is divided by g(x) is 16*x - 59.
theorem polynomial_remainder : 
  ∀ x : ℚ, polynomial.modBy g f = 16*x - 59 := 
by
  sorry

end polynomial_remainder_l824_824605


namespace solution_set_lg_sine_cosine_l824_824453

theorem solution_set_lg_sine_cosine (x : ℝ) :
  (∃ k : ℤ, x = 2 * k * Real.pi + 5 * Real.pi / 6) ↔
  (log (sqrt 3 * sin x) = log (-cos x)) := 
sorry

end solution_set_lg_sine_cosine_l824_824453


namespace sum_possible_students_l824_824518

theorem sum_possible_students:
  let s := (list.range 61).map (λ n, 141 + 7 * n)
  ∈ [141, 148, 155, 162, 169, 176, 183, 190, 197] 
  in
  (∀ x ∈ s, 140 ≤ x ∧ x ≤ 200 ∧ (x - 1) % 7 = 0) →
  list.sum s = 1521 :=
by
  sorry

end sum_possible_students_l824_824518


namespace evaluate_expression_l824_824213

theorem evaluate_expression :
  (- (3 / 4 : ℚ)) / 3 * (- (2 / 5 : ℚ)) = 1 / 10 := 
by
  -- Here is where the proof would go
  sorry

end evaluate_expression_l824_824213


namespace pure_imaginary_iff_a_zero_l824_824501

theorem pure_imaginary_iff_a_zero (a b : ℝ) : (a + b * complex.I).im ≠ 0 ↔ a = 0 :=
by
  sorry

end pure_imaginary_iff_a_zero_l824_824501


namespace least_integer_greater_than_sqrt_500_l824_824085

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824085


namespace distance_between_A_and_B_l824_824570

theorem distance_between_A_and_B 
    (Time_E : ℝ) (Time_F : ℝ) (D_AC : ℝ) (V_ratio : ℝ)
    (E_time : Time_E = 3) (F_time : Time_F = 4) 
    (AC_distance : D_AC = 300) (speed_ratio : V_ratio = 4) : 
    ∃ D_AB : ℝ, D_AB = 900 :=
by
  sorry

end distance_between_A_and_B_l824_824570


namespace chord_square_l824_824215

/-- 
Circles with radii 3 and 6 are externally tangent and are internally tangent to a circle with radius 9. 
The circle with radius 9 has a chord that is a common external tangent of the other two circles. Prove that 
the square of the length of this chord is 72.
-/
theorem chord_square (O₁ O₂ O₃ : Type) 
  (r₁ r₂ r₃ : ℝ) 
  (O₁_tangent_O₂ : r₁ + r₂ = 9) 
  (O₃_tangent_O₁ : r₃ - r₁ = 6) 
  (O₃_tangent_O₂ : r₃ - r₂ = 3) 
  (tangent_chord : ℝ) : 
  tangent_chord^2 = 72 :=
by sorry

end chord_square_l824_824215


namespace Eric_return_time_l824_824949

theorem Eric_return_time (t1 t2 t_return : ℕ) 
  (h1 : t1 = 20) 
  (h2 : t2 = 10) 
  (h3 : t_return = 3 * (t1 + t2)) : 
  t_return = 90 := 
by 
  sorry

end Eric_return_time_l824_824949


namespace value_of_y_l824_824682

theorem value_of_y (y : ℝ) (h : 27 ^ y = 243) : y = 5 / 3 :=
by sorry

end value_of_y_l824_824682


namespace triangle_PQT_is_isosceles_l824_824641

open Set

variables {A B C D E P Q T : Point} [h : Circle (Set.insert A (Set.insert B (Set.insert C (Set.insert D (Set.singleton E)))))] 
variables (h1 : dist A B = dist B C)
variables (h2 : dist C D = dist D E)
variables (hP : P = line_intersection (line_through A D) (line_through B E))
variables (hQ : Q = line_intersection (line_through A C) (line_through B D))
variables (hT : T = line_intersection (line_through B D) (line_through C E))

theorem triangle_PQT_is_isosceles : is_isosceles_triangle P Q T := sorry

end triangle_PQT_is_isosceles_l824_824641


namespace range_of_BD_l824_824498

-- Define the types of points and triangle
variables {α : Type*} [MetricSpace α]

-- Hypothesis: AD is the median of triangle ABC
-- Definition of lengths AB, AC, and that BD = CD.
def isMedianOnBC (A B C D : α) : Prop :=
  dist A B = 5 ∧ dist A C = 7 ∧ dist B D = dist C D

-- The theorem to be proven
theorem range_of_BD {A B C D : α} (h : isMedianOnBC A B C D) : 
  1 < dist B D ∧ dist B D < 6 :=
by
  sorry

end range_of_BD_l824_824498


namespace find_y_l824_824253

theorem find_y
  (a b c x : ℝ) (p q r y : ℝ)
  (h1 : log a / p = log b / q) 
  (h2 : log b / q = log c / r) 
  (h3 : log c / r = log x)
  (h4 : x ≠ 1) 
  (h5 : a^3 * b / c^2 = x^y) : 
  y = 3 * p + q - 2 * r :=
sorry

end find_y_l824_824253


namespace zero_point_interval_l824_824428

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_point_interval : ∃ (a b : ℝ), a < b ∧ (∀ x ∈ set.Ioo a b, f x = 0) ↔ a = 2 ∧ b = Real.exp 1 := 
sorry

end zero_point_interval_l824_824428


namespace triangle_side_length_AC_l824_824339

theorem triangle_side_length_AC
  (a b c : ℝ)
  (angle_A angle_B angle_C : ℝ)
  (area : ℝ)
  (h_obtuse : angle_A > π / 2 ∨ angle_B > π / 2 ∨ angle_C > π / 2)
  (h_area : area = 1 / 2)
  (h_AB_eq_1 : a = 1)
  (h_BC_eq_sqrt2 : c = sqrt 2)
  (h_angles_sum : angle_A + angle_B + angle_C = π) :
  b = sqrt 5 :=
  sorry

end triangle_side_length_AC_l824_824339


namespace hexagon_tiling_colors_l824_824347

-- Problem Definition
theorem hexagon_tiling_colors (k l : ℕ) (hk : 0 < k ∨ 0 < l) : 
  ∃ n: ℕ, n = k^2 + k * l + l^2 :=
by
  sorry

end hexagon_tiling_colors_l824_824347


namespace least_integer_greater_than_sqrt_500_l824_824093

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824093


namespace last_colored_cell_in_spiral_l824_824786

-- Define the dimensions of the rectangle
def width : ℕ := 200
def height : ℕ := 100

-- Define the function representing the spiral coloring pattern (informal definition for now)
def spiral_coloring (w h : ℕ) : (ℕ × ℕ) → Prop := sorry

-- Lean statement proving the final cell
theorem last_colored_cell_in_spiral :
  spiral_coloring width height (51, 50) :=
sorry

end last_colored_cell_in_spiral_l824_824786


namespace avg_speed_approx_l824_824697

theorem avg_speed_approx (D : ℝ) (hD : D > 0) : 
  let time1 := D / 240
  let time2 := D / 72
  let time3 := D / 132
  let total_time := (time1 + time2 + time3)
  let avg_speed := D / total_time
  avg_speed ≈ 38.8235 := 
by {
  sorry
}

end avg_speed_approx_l824_824697


namespace least_integer_greater_than_sqrt_500_l824_824115

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824115


namespace min_omega_l824_824321

theorem min_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : ∀ x, 0 ≤ x ∧ x ≤ 1 → sin(ω * x) ≥ 50) : 
  ω ≥ 197 * π / 2 :=
sorry

end min_omega_l824_824321


namespace smallest_a_value_l824_824370

theorem smallest_a_value (a b : ℝ) (h_a_nonneg : 0 ≤ a) (h_b_nonneg : 0 ≤ b)
  (h_sin_eq : ∀ x : ℝ, sin (a * x + b) = sin (15 * x)) :
  a = 15 :=
  sorry

end smallest_a_value_l824_824370


namespace ratio_triangle_area_to_rectangle_area_l824_824332

theorem ratio_triangle_area_to_rectangle_area 
  (ABCD : Type) [rect : Rectangle ABCD]
  (P Q : ABCD)
  (P_midpoint : P = midpoint AB)
  (Q_midpoint : Q = midpoint BC)
  (AB_length_twice_BC : ∀ (A B C D : ABCD), side_length AB = 2 * side_length BC) :
  area (triangle APQ) / area (rectangle ABCD) = 1 / 8 :=
  sorry

end ratio_triangle_area_to_rectangle_area_l824_824332


namespace ink_amount_equality_l824_824931

-- Definitions for initial amounts of ink
def initial_red_ink_in_A (m : ℝ) : ℝ := m
def initial_blue_ink_in_B (m : ℝ) : ℝ := m

-- Definition for the first transfer from A to B
def transfer_from_A_to_B (m a : ℝ) : ℝ := m + a
def concentration_red_after_first_transfer (a m : ℝ) : ℝ := a / (m + a)
def concentration_blue_after_first_transfer (m a : ℝ) : ℝ := m / (m + a)

-- Definition for the second transfer from B to A
def blue_ink_transferred_to_A (m a : ℝ) : ℝ := a * (m / (m + a))
def red_ink_transferred_to_B (a m : ℝ) : ℝ := a * (a / (m + a))

-- Definition for the amounts of ink after both transfers
def final_blue_ink_in_A (m a : ℝ) : ℝ := blue_ink_transferred_to_A m a
def final_red_ink_in_B (m a : ℝ) : ℝ := initial_red_ink_in_A m - red_ink_transferred_to_B a m

-- Theorem statement for the equivalence of amounts
theorem ink_amount_equality (m a : ℝ) (h : m > 0 ∧ a > 0) :
  final_blue_ink_in_A m a = final_red_ink_in_B m a :=
sorry

end ink_amount_equality_l824_824931


namespace least_integer_greater_than_sqrt_500_l824_824112

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824112


namespace segments_equal_and_perpendicular_l824_824850

noncomputable def square_vertices (A B C D : Point) : Prop :=
  is_square A B C D

noncomputable def lines_through_vertex (l1 l2 : Line) (A : Point) : Prop :=
  passes_through l1 A ∧ passes_through l2 A

noncomputable def perpendiculars (l1 l2 : Line) (A B D B1 B2 D1 D2 : Point) : Prop :=
  foot_of_perpendicular B l1 B1 ∧ foot_of_perpendicular B l2 B2 ∧ 
  foot_of_perpendicular D l1 D1 ∧ foot_of_perpendicular D l2 D2

theorem segments_equal_and_perpendicular 
  (A B C D B1 B2 D1 D2 : Point) (l1 l2 : Line)
  (h_square : square_vertices A B C D)
  (h_lines : lines_through_vertex l1 l2 A)
  (h_perpendiculars : perpendiculars l1 l2 A B D B1 B2 D1 D2) :
  segment_length_equal B1 B2 D1 D2 ∧ segments_perpendicular B1 B2 D1 D2 :=
begin
  sorry
end

end segments_equal_and_perpendicular_l824_824850


namespace intersection_A_B_l824_824298

-- define the set A
def A : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ 3 * x - y = 7 }

-- define the set B
def B : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ 2 * x + y = 3 }

-- Prove the intersection
theorem intersection_A_B :
  A ∩ B = { (2, -1) } :=
by
  -- We will insert the proof here
  sorry

end intersection_A_B_l824_824298


namespace find_y_given_x_eq_neg6_l824_824994

theorem find_y_given_x_eq_neg6 :
  ∀ (y : ℤ), (∃ (x : ℤ), x = -6 ∧ x^2 - x + 6 = y - 6) → y = 54 :=
by
  intros y h
  obtain ⟨x, hx1, hx2⟩ := h
  rw [hx1] at hx2
  simp at hx2
  linarith

end find_y_given_x_eq_neg6_l824_824994


namespace least_integer_greater_than_sqrt_500_l824_824032

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824032


namespace single_elimination_games_l824_824522

theorem single_elimination_games (teams : ℕ) (h : teams = 23) :
  single_elimination_games_needed teams = 22 := by
  sorry

end single_elimination_games_l824_824522


namespace min_product_OP_OQ_l824_824659

theorem min_product_OP_OQ (a b : ℝ) (h : a > b ∧ b > 0) 
  {P Q : ℝ × ℝ}
  (hP : (P.1^2 / a^2 + P.2^2 / b^2 = 1))
  (hQ : (Q.1^2 / a^2 + Q.2^2 / b^2 = 1))
  (h_perp : P.1 * Q.1 + P.2 * Q.2 = 0) :
  |sqrt(P.1^2 + P.2^2) * sqrt(Q.1^2 + Q.2^2)| ≥ 2 * a ^ 2 * b ^ 2 / (a ^ 2 + b ^ 2) := 
sorry

end min_product_OP_OQ_l824_824659


namespace no_digit_B_divisible_by_4_l824_824351

theorem no_digit_B_divisible_by_4 : 
  ∀ B : ℕ, B < 10 → ¬ (8 * 1000000 + B * 100000 + 4 * 10000 + 6 * 1000 + 3 * 100 + 5 * 10 + 1) % 4 = 0 :=
by
  intros B hB_lt_10
  sorry

end no_digit_B_divisible_by_4_l824_824351


namespace quiz_true_false_questions_l824_824907

theorem quiz_true_false_questions (n : ℕ) 
  (h1 : 2^n - 2 ≠ 0) 
  (h2 : (2^n - 2) * 16 = 224) : 
  n = 4 := 
sorry

end quiz_true_false_questions_l824_824907


namespace minimum_distance_on_ellipse_l824_824652

noncomputable def ellipse_PQ_min_distance : ℝ := 
  sqrt (6) / 3

theorem minimum_distance_on_ellipse :
  ∀ x y : ℝ, (x^2 / 4 + y^2 = 1) → 
  ∃ (min_dist : ℝ), min_dist = sqrt (6) / 3 ∧ 
  min_dist = Real.dist (x, y) (1, 0) := 
begin
  sorry
end

end minimum_distance_on_ellipse_l824_824652


namespace problem_l824_824637

-- Definitions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a (n : ℕ) : ℤ := sorry -- Define the arithmetic sequence a_n based on conditions

-- Problem statement
theorem problem : 
  (a 1 = 4) ∧
  (a 2 + a 4 = 4) →
  (∃ d : ℤ, arithmetic_sequence a d ∧ a 10 = -5) :=
by {
  sorry
}

end problem_l824_824637


namespace orange_count_in_bin_l824_824879

-- Definitions of the conditions
def initial_oranges : Nat := 5
def oranges_thrown_away : Nat := 2
def new_oranges_added : Nat := 28

-- The statement of the proof problem
theorem orange_count_in_bin : initial_oranges - oranges_thrown_away + new_oranges_added = 31 :=
by
  sorry

end orange_count_in_bin_l824_824879


namespace handshake_problem_l824_824841

theorem handshake_problem (n : ℕ) (h : n * (n - 1) / 2 = 1770) : n = 60 :=
sorry

end handshake_problem_l824_824841


namespace tan_sum_identity_l824_824211

theorem tan_sum_identity :
  Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180) + 
  Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180) = 1 :=
by sorry

end tan_sum_identity_l824_824211


namespace polynomial_is_linear_rational_l824_824582

noncomputable def polynomial_condition (P : Polynomial ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℚ, P.eval r = n

theorem polynomial_is_linear_rational 
  (P : Polynomial ℝ) 
  (h : polynomial_condition P) 
  : ∃ a b : ℚ, P = Polynomial.C a + Polynomial.X * Polynomial.C b :=
  sorry

end polynomial_is_linear_rational_l824_824582


namespace next_balanced_year_is_2048_l824_824829

def is_balanced_binary (n : Nat) : Bool :=
  let ones := n.binary_digits.filter (λ b => b = 1).length
  let zeros := n.binary_digits.filter (λ b => b = 0).length
  ones ≤ zeros

theorem next_balanced_year_is_2048 : ∃ n, n > 2017 ∧ is_balanced_binary n ∧ n = 2048 :=
by {
  -- Fill in the proof step, if required
  sorry
}

end next_balanced_year_is_2048_l824_824829


namespace constant_term_in_expansion_l824_824281

theorem constant_term_in_expansion :
  (∑ k in Finset.range (n + 1), (binomial n k : ℝ)) = 64 →
  n = 6 →
  (∃ r, (6 : ℝ) - 3 * (r : ℝ) / 2 = 0 ∧ 
  finset.choose 6 r * 5 ^ (6 - r) = 375) :=
begin
  intros h₁ h₂,
  rw [finset.sum_range_succ] at h₁,
  cases h₂ with h₂,
  use 4,
  split,
  {
    exact sorry, -- here we'd solve 6 - 3 * r / 2 = 0 to find r = 4
  },
  {
    exact sorry, -- here we'd compute the binomial coefficient and powers to get 375
  }
end

end constant_term_in_expansion_l824_824281


namespace transformation_of_vector_l824_824219

variable (U : ℝ^3 → ℝ^3)

def linearity (U : ℝ^3 → ℝ^3) : Prop :=
  ∀ (a b : ℝ) (u v : ℝ^3), 
    U (a • u + b • v) = a • U u + b • U v

def cross_product_preservation (U : ℝ^3 → ℝ^3) : Prop :=
  ∀ (u v : ℝ^3), 
    U (u × v) = U u × U v

def example_property1 (U : ℝ^3 → ℝ^3) : Prop :=
  U ⟨5, 7, 2⟩ = ⟨3, -2, 7⟩ 

def example_property2 (U : ℝ^3 → ℝ^3) : Prop :=
  U ⟨-5, 2, 7⟩ = ⟨3, 7, -2⟩

theorem transformation_of_vector
  (lin : linearity U)
  (cross_pres : cross_product_preservation U)
  (prop1 : example_property1 U)
  (prop2 : example_property2 U) :
  U ⟨2, 11, 16⟩ = ⟨61 / 13, 104 / 13, 69 / 13⟩ :=
  sorry

end transformation_of_vector_l824_824219


namespace opposite_of_23_is_neg23_l824_824832

theorem opposite_of_23_is_neg23 : ∃ b : ℤ, 23 + b = 0 ∧ b = -23 := by
  use -23
  split
  · linarith
  · rfl

end opposite_of_23_is_neg23_l824_824832


namespace tan_ratio_l824_824764

open Real

variable (x y : ℝ)

-- Conditions
def sin_add_eq : sin (x + y) = 5 / 8 := sorry
def sin_sub_eq : sin (x - y) = 1 / 4 := sorry

-- Proof problem statement
theorem tan_ratio : sin (x + y) = 5 / 8 → sin (x - y) = 1 / 4 → (tan x) / (tan y) = 2 := sorry

end tan_ratio_l824_824764


namespace binom_1300_2_eq_844350_l824_824555

theorem binom_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 :=
by
  sorry

end binom_1300_2_eq_844350_l824_824555


namespace equality_of_shaded_areas_l824_824737

theorem equality_of_shaded_areas (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 4) :
    tan (2 * θ) = 2 * θ :=
sorry

end equality_of_shaded_areas_l824_824737


namespace triangle_equality_l824_824768

-- Definitions for geometric objects
variables (A B C M Q H S : Type) 

-- Conditions specified in the problem
variables [IsTriangle ABC] [Midpoint M B C] [AngleBisector A Q B C] 
          [Altitude H A B C] [PerpendicularTo AQ A S]

-- The main theorem statement
theorem triangle_equality 
  (h_triangle : IsTriangle ABC)
  (h_midpoint : Midpoint M B C)
  (h_angle_bisector : AngleBisector A Q B C)
  (h_altitude : Altitude H A B C)
  (h_perpendicular : PerpendicularTo AQ A S) :
  MH * QS = AB * AC := 
  sorry

end triangle_equality_l824_824768


namespace least_integer_greater_than_sqrt_500_l824_824036

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824036


namespace least_integer_greater_than_sqrt_500_l824_824062

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824062


namespace least_integer_greater_than_sqrt_500_l824_824034

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824034


namespace find_b_l824_824687

theorem find_b (b : ℝ) : (∃ c : ℝ, (16 : ℝ) * x^2 + 40 * x + b = (4 * x + c)^2) → b = 25 :=
by
  sorry

end find_b_l824_824687


namespace vet_donates_correct_amount_l824_824190

-- Define the conditions
def vet_fee_dog : ℕ := 15
def vet_fee_cat : ℕ := 13
def dog_adopters : ℕ := 8
def cat_adopters : ℕ := 3

-- Define the total fee calculation
def total_fee_collected : ℕ := (dog_adopters * vet_fee_dog) + (cat_adopters * vet_fee_cat)

-- Prove that the vet donates one third of the total fee collected back to the shelter
def vet_donation : ℕ := total_fee_collected / 3

-- Statement of the proof problem
theorem vet_donates_correct_amount : vet_donation = 53 :=
by
  -- Calculate total vet fees collected
  have total_fee : ℕ := total_fee_collected
  -- Calculate the donation
  have donation : ℕ := total_fee / 3
  -- The donation should be equal to $53
  exact Eq.refl 53

  sorry -- proof to be filled

end vet_donates_correct_amount_l824_824190


namespace contains_all_integers_l824_824359

def is_closed_under_divisors (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, b ∣ a → a ∈ A → b ∈ A

def contains_product_plus_one (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, 1 < a → a < b → a ∈ A → b ∈ A → (1 + a * b) ∈ A

theorem contains_all_integers
  (A : Set ℕ)
  (h1 : is_closed_under_divisors A)
  (h2 : contains_product_plus_one A)
  (h3 : ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ 1 < a ∧ 1 < b ∧ 1 < c) :
  ∀ n : ℕ, n > 0 → n ∈ A := 
  by 
    sorry

end contains_all_integers_l824_824359


namespace trig_second_quadrant_l824_824481

theorem trig_second_quadrant (α : ℝ) (h1 : α > π / 2) (h2 : α < π) :
  (|Real.sin α| / Real.sin α) - (|Real.cos α| / Real.cos α) = 2 :=
by
  sorry

end trig_second_quadrant_l824_824481


namespace least_integer_gt_sqrt_500_l824_824102

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824102


namespace greatest_sum_consecutive_integers_l824_824470

theorem greatest_sum_consecutive_integers (n : ℤ) (h : n * (n + 1) < 360) : n + (n + 1) ≤ 37 := by
  sorry

end greatest_sum_consecutive_integers_l824_824470


namespace largest_power_of_15_dividing_factorial_30_l824_824590

theorem largest_power_of_15_dividing_factorial_30 : ∃ (n : ℕ), 
  (∀ m : ℕ, 15 ^ m ∣ Nat.factorial 30 ↔ m ≤ n) ∧ n = 7 :=
by
  have h3 : ∀ m : ℕ, 3 ^ m ∣ Nat.factorial 30 ↔ m ≤ 14 := sorry
  have h5 : ∀ m : ℕ, 5 ^ m ∣ Nat.factorial 30 ↔ m ≤ 7 := sorry
  use 7
  split
  · intro m
    split
    · intro h
      obtain ⟨k, rfl⟩ : ∃ k, m = k := Nat.exists_eq m
      have : 3 ^ k ∣ Nat.factorial 30 := by exact (15 ^ k).dvd_of_dvd_mul_left (by convert h)
      rw [h3, h5] at this
      exact Nat.le_min this.left this.right
    · intro h
      exact (Nat.min_le_iff.mpr ⟨(h3.mpr (Nat.le_of_lt_succ (Nat.sub_lt_succ (14 - 7))), h5.mpr h⟩, sorry
  exact rfl
  sorry

end largest_power_of_15_dividing_factorial_30_l824_824590


namespace successive_increases_eq_single_l824_824447

variable (P : ℝ)

def increase_by (initial : ℝ) (pct : ℝ) : ℝ := initial * (1 + pct)
def discount_by (initial : ℝ) (pct : ℝ) : ℝ := initial * (1 - pct)

theorem successive_increases_eq_single (P : ℝ) :
  increase_by (increase_by (discount_by (increase_by P 0.30) 0.10) 0.15) 0.20 = increase_by P 0.6146 :=
  sorry

end successive_increases_eq_single_l824_824447


namespace tetrahedron_median_sum_geq_16_over_3_radius_l824_824901

variables {A B C D O A_1 B_1 C_1 D_1 : Type} [geometry A B C D O A_1 B_1 C_1 D_1] {R : ℝ} 

theorem tetrahedron_median_sum_geq_16_over_3_radius (h_inscribed : is_inscribed A B C D O R)
  (h_intersection_A1 : intersects AO (opposite_face D A B C) A_1)
  (h_intersection_B1 : intersects BO (opposite_face A B C D) B_1)
  (h_intersection_C1 : intersects CO (opposite_face B C D A) C_1)
  (h_intersection_D1 : intersects DO (opposite_face C D A B) D_1) :
  AA_1 + BB_1 + CC_1 + DD_1 ≥ (16 / 3) * R :=
sorry

end tetrahedron_median_sum_geq_16_over_3_radius_l824_824901


namespace hyperbola_point_distance_l824_824706

theorem hyperbola_point_distance
  (P : ℝ × ℝ)
  (F1 : ℝ × ℝ := (-5, 0))
  (F2 : ℝ × ℝ := (5, 0))
  (hP_on_hyperbola : (P.1^2 / 9 - P.2^2 / 16 = 1))
  (hPF1 : real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) = 3) :
  real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 9 :=
sorry

end hyperbola_point_distance_l824_824706


namespace taxi_charge_first_fifth_mile_is_3_l824_824888

def charge_per_first_fifth_mile (total_charge : ℝ) (increment_charge : ℝ) (total_distance : ℝ) : Prop :=
  ∀ (charge_first_fifth : ℝ),
    total_distance = 8 ∧
    total_charge = 18.60 ∧
    increment_charge = 0.40 →
    total_charge = charge_first_fifth + ((total_distance - 1/5) * 5 * increment_charge)

theorem taxi_charge_first_fifth_mile_is_3 :
  charge_per_first_fifth_mile 18.60 0.40 8 3 :=
by
  sorry

end taxi_charge_first_fifth_mile_is_3_l824_824888


namespace least_integer_greater_than_sqrt_500_l824_824082

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824082


namespace least_integer_greater_than_sqrt_500_l824_824081

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824081


namespace farm_field_ploughing_l824_824857

theorem farm_field_ploughing (A D : ℕ) 
  (h1 : ∀ farmerA_initial_capacity: ℕ, farmerA_initial_capacity = 120)
  (h2 : ∀ farmerB_initial_capacity: ℕ, farmerB_initial_capacity = 100)
  (h3 : ∀ farmerA_adjustment: ℕ, farmerA_adjustment = 10)
  (h4 : ∀ farmerA_reduced_capacity: ℕ, farmerA_reduced_capacity = farmerA_initial_capacity - (farmerA_adjustment * farmerA_initial_capacity / 100))
  (h5 : ∀ farmerB_reduced_capacity: ℕ, farmerB_reduced_capacity = 90)
  (h6 : ∀ extra_days: ℕ, extra_days = 3)
  (h7 : ∀ remaining_hectares: ℕ, remaining_hectares = 60)
  (h8 : ∀ initial_combined_effort: ℕ, initial_combined_effort = (farmerA_initial_capacity + farmerB_initial_capacity) * D)
  (h9 : ∀ total_combined_effort: ℕ, total_combined_effort = (farmerA_reduced_capacity + farmerB_reduced_capacity) * (D + extra_days))
  (h10 : ∀ area_covered: ℕ, area_covered = total_combined_effort + remaining_hectares)
  : initial_combined_effort = A ∧ D = 30 ∧ A = 6600 :=
by
  sorry

end farm_field_ploughing_l824_824857


namespace infinite_sqrt_solution_l824_824230

noncomputable def infinite_sqrt (x : ℝ) : ℝ := Real.sqrt (20 + x)

theorem infinite_sqrt_solution : 
  ∃ x : ℝ, infinite_sqrt x = x ∧ x ≥ 0 ∧ x = 5 :=
by
  sorry

end infinite_sqrt_solution_l824_824230


namespace highest_number_of_years_of_service_l824_824831

theorem highest_number_of_years_of_service
  (years_of_service : Fin 8 → ℕ)
  (h_range : ∃ L, ∃ H, H - L = 14)
  (h_second_highest : ∃ second_highest, second_highest = 16) :
  ∃ highest, highest = 17 := by
  sorry

end highest_number_of_years_of_service_l824_824831


namespace cost_of_four_dozen_apples_l824_824538

theorem cost_of_four_dozen_apples (cost_two_dozen : ℝ) (h : cost_two_dozen = 15.60) : ∃ cost_four_dozen, cost_four_dozen = 31.20 :=
by
  have cost_per_dozen := cost_two_dozen / 2
  have cost_four_dozen := 4 * cost_per_dozen
  use cost_four_dozen
  rw h
  norm_num
  exact eq.refl 31.20

end cost_of_four_dozen_apples_l824_824538


namespace find_n_l824_824617

-- Definitions based directly on conditions
def perms (n : ℕ) := {f // ∀ i : fin n, 1 ≤ i → f i ≥ (i : ℕ) - 1}

def valid_perms (n : ℕ) := {f // ∀ i : fin n, 1 ≤ i → f i ≥ (i : ℕ) - 1 ∧ f i ≤ (i : ℕ) + 1}

-- Function to calculate the number of valid permutations
def B_n (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else sorry -- Follow recurrence relation to fill in

-- Function to calculate total permutations
def A_n (n : ℕ) : ℕ := 2^(n - 1)

-- Calculate p_n
def p_n (n : ℕ) : ℚ := (B_n n) / (A_n n)

-- Predicate to check if p_n > 1/3
def pn_greater_than_one_third (n : ℕ) : Prop := p_n n > (1 / 3)

-- Theorem statement that n ∈ {1, 2, 3, 4, 5, 6} if and only if pn_greater_than_one_third
theorem find_n (n : ℕ) : pn_greater_than_one_third n ↔ n ∈ {1, 2, 3, 4, 5, 6} :=
sorry

end find_n_l824_824617


namespace find_sine_l824_824988

-- Definitions based on given conditions
structure AngleInSecondQuadrant (α : ℝ) : Prop :=
  (hx1 : α ∈ Set.Ioc (π / 2) π)

def point_on_terminal_side (P : ℝ × ℝ) (α : ℝ) (hP : P = (x, sqrt 5)) : Prop :=
  true  -- simply indicating the point exists as given

def cosine_condition (α : ℝ) (x : ℝ) : Prop :=
  cos α = (sqrt 2 / 4) * x

-- The theorem to be proved
theorem find_sine (α x : ℝ) (hα : AngleInSecondQuadrant α)
  (hP : point_on_terminal_side (x, sqrt 5) α rfl)
  (hcos : cosine_condition α x) :
  sin α = sqrt 10 / 4 := 
sorry

end find_sine_l824_824988


namespace zeke_estimate_smaller_l824_824851

variable (x y k : ℝ)
variable (hx_pos : 0 < x)
variable (hy_pos : 0 < y)
variable (h_inequality : x > 2 * y)
variable (hk_pos : 0 < k)

theorem zeke_estimate_smaller : (x + k) - 2 * (y + k) < x - 2 * y :=
by
  sorry

end zeke_estimate_smaller_l824_824851


namespace meat_market_ratio_l824_824400

theorem meat_market_ratio :
  ∀ (F : ℕ), 
    (210 + F + 130 + 65 = 825) →
    (F = 420) →
    (F : ℚ) / 210 = 2 :=
by
  intros F h1 h2
  rw h2
  norm_num


end meat_market_ratio_l824_824400


namespace tank_capacity_correct_l824_824194

-- let's define our variables and constants
def outlet_time : ℝ := 10
def inlet_rate_per_min : ℝ := 16
def extra_time_with_inlet_open : ℝ := 8
def effective_total_time : ℝ := outlet_time + extra_time_with_inlet_open
def inlet_rate_per_hour : ℝ := inlet_rate_per_min * 60

noncomputable def tank_capacity : ℝ :=
  let outlet_rate : ℝ := tank_capacity / outlet_time
  let effective_effective_outlet_rate: ℝ := tank_capacity / effective_total_time
  let equation := outlet_rate - inlet_rate_per_hour = effective_effective_outlet_rate
  21600

theorem tank_capacity_correct : tank_capacity = 21600 :=
by
  have outlet_rate: ℝ := tank_capacity / outlet_time
  have effective_effective_outlet_rate: ℝ := tank_capacity / effective_total_time
  have equation: outlet_rate - inlet_rate_per_hour = effective_effective_outlet_rate := sorry
  show tank_capacity = 21600, from sorry

end tank_capacity_correct_l824_824194


namespace midpoint_length_l824_824404

theorem midpoint_length (X Y G H I J : Type*) (d : ℕ) 
  (midpoint_G : G = (X + Y) / 2) 
  (midpoint_H : H = (X + G) / 2)
  (midpoint_I : I = (X + H) / 2) 
  (midpoint_J : J = (X + I) / 2) 
  (length_XJ : d = 5) 
  : 16 * d = (X - Y) :=
sorry

end midpoint_length_l824_824404


namespace product_of_possible_values_of_N_l824_824918

theorem product_of_possible_values_of_N (M L N : ℝ) (h1 : M = L + N) (h2 : M - 5 = (L + N) - 5) (h3 : L + 3 = L + 3) (h4 : |(L + N - 5) - (L + 3)| = 2) : 10 * 6 = 60 := by
  sorry

end product_of_possible_values_of_N_l824_824918


namespace time_difference_l824_824877

-- Define the capacity of the tanks
def capacity : ℕ := 20

-- Define the inflow rates of tanks A and B in litres per hour
def inflow_rate_A : ℕ := 2
def inflow_rate_B : ℕ := 4

-- Define the times to fill tanks A and B
def time_A : ℕ := capacity / inflow_rate_A
def time_B : ℕ := capacity / inflow_rate_B

-- Proving the time difference between filling tanks A and B
theorem time_difference : (time_A - time_B) = 5 := by
  sorry

end time_difference_l824_824877


namespace log_equation_l824_824649

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_equation (x : ℝ) (h1 : x > 1) (h2 : (log_base_10 x)^2 - log_base_10 (x^4) = 32) :
  (log_base_10 x)^4 - log_base_10 (x^4) = 4064 :=
by
  sorry

end log_equation_l824_824649


namespace tan_ratio_l824_824766

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : (Real.tan x) / (Real.tan y) = 2 := 
by
  sorry

end tan_ratio_l824_824766


namespace train_length_approx_l824_824872

noncomputable def length_of_train (distance_km : ℝ) (time_min : ℝ) (time_sec : ℝ) : ℝ :=
  let distance_m := distance_km * 1000 -- Convert km to meters
  let time_s := time_min * 60 -- Convert min to seconds
  let speed := distance_m / time_s -- Speed in meters/second
  speed * time_sec -- Length of train in meters

theorem train_length_approx :
  length_of_train 10 15 10 = 111.1 :=
by
  sorry

end train_length_approx_l824_824872


namespace min_blue_eyes_with_lunchbox_l824_824955

theorem min_blue_eyes_with_lunchbox (B L : Finset Nat) (hB : B.card = 15) (hL : L.card = 25) (students : Finset Nat) (hst : students.card = 35)  : 
  ∃ (x : Finset Nat), x ⊆ B ∧ x ⊆ L ∧ x.card ≥ 5 :=
by
  sorry

end min_blue_eyes_with_lunchbox_l824_824955


namespace values_of_x25_l824_824139

noncomputable def system_of_equations (x : ℕ → ℝ) : Prop :=
  (x 1 * x 2 * ... * x 1962 = 1) ∧
  (x 1 - x 2 * x 3 * ... * x 1962 = 1) ∧
  ...
  (x 1 * x 2 * ... * x 1961 - x 1962 = 1)

theorem values_of_x25 (x : ℕ → ℝ) (h : system_of_equations x) :
  x 25 = 1 ∨ x 25 = - (3 + Real.sqrt 5) / 2 ∨ x 25 = - (3 - Real.sqrt 5) / 2 :=
sorry

end values_of_x25_l824_824139


namespace determine_coefficients_l824_824376

theorem determine_coefficients
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h_def : f = (λ x, a * x^3 - 3 * a * x^2 + b))
  (h_interval : ∀ x, x ∈ Set.Icc (-1 : ℝ) 2 → True)  -- This condition implicitly that x ∈ [-1, 2]
  (h_max : ∃ x, x ∈ Set.Icc (-1 : ℝ) 2 ∧ f x = 3)
  (h_min : ∃ x, x ∈ Set.Icc (-1 : ℝ) 2 ∧ f x = -21)
  (h_a_pos : 0 < a):
  a = 6 ∧ b = 3 := sorry

end determine_coefficients_l824_824376


namespace average_walking_speed_correct_l824_824166

noncomputable def average_walking_speed (total_distance : ℝ) (total_time : ℝ) 
(speed_canoe : ℝ) (distance_walked : ℝ) : ℝ :=
let distance_canoe := total_distance - distance_walked 
in let time_canoe := distance_canoe / speed_canoe 
in let time_walking := total_time - time_canoe 
in distance_walked / time_walking

theorem average_walking_speed_correct :
  average_walking_speed 43.25 5.5 12 27 ≈ 6.51 :=
by
  -- use appropriate metric for "approximately equal"
  sorry

end average_walking_speed_correct_l824_824166


namespace original_number_divisible_l824_824627

theorem original_number_divisible (N M R : ℕ) (n : ℕ) (hN : N = 1000 * M + R)
  (hDiff : (M - R) % n = 0) (hn : n = 7 ∨ n = 11 ∨ n = 13) : N % n = 0 :=
by
  sorry

end original_number_divisible_l824_824627


namespace ratio_of_perimeters_l824_824473

theorem ratio_of_perimeters (a : ℝ) : 
  let b := 4 * a in
  let c := 6 * a in
  (4 * a) / (4 * b) = 1 / 4 ∧ (4 * b) / (4 * c) = 4 / 6 :=
by
  sorry

end ratio_of_perimeters_l824_824473


namespace field_total_rent_l824_824503

theorem field_total_rent (A_cows : ℕ) (A_months : ℕ) (B_cows : ℕ) (B_months : ℕ) (C_cows : ℕ) (C_months : ℕ) (D_cows : ℕ) (D_months : ℕ) (A_rent : ℕ) :
  (A_cows = 24) ∧ (A_months = 3) ∧ 
  (B_cows = 10) ∧ (B_months = 5) ∧ 
  (C_cows = 35) ∧ (C_months = 4) ∧ 
  (D_cows = 21) ∧ (D_months = 3) ∧ 
  (A_rent = 720) →
  let A_cow_months := A_cows * A_months in
  let B_cow_months := B_cows * B_months in
  let C_cow_months := C_cows * C_months in
  let D_cow_months := D_cows * D_months in
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + D_cow_months in
  let rent_per_cow_month := A_rent / A_cow_months in
  let total_rent := rent_per_cow_month * total_cow_months in
  total_rent = 3250 :=
begin
  sorry
end

end field_total_rent_l824_824503


namespace complex_seq_sum_bound_l824_824265

noncomputable def complex_seq (n : ℕ) : ℂ := sorry

axiom complex_seq_base (z : ℕ → ℂ) : |z 1| = 1

axiom complex_seq_recurrence (z : ℕ → ℂ) (n : ℕ) : 
  4 * (z (n + 1))^2 + 2 * z n * z (n + 1) + (z n)^2 = 0

theorem complex_seq_sum_bound (z : ℕ → ℂ) (m : ℕ) (hm : 0 < m) :
  |∑ i in Finset.range m, z (i + 1)| < 2 * Real.sqrt 3 / 3 :=
by
  sorry

end complex_seq_sum_bound_l824_824265


namespace bowling_average_l824_824894

theorem bowling_average (A : ℝ) (W : ℕ) (hW : W = 145) (hW7 : W + 7 ≠ 0)
  (h : ( A * W + 26 ) / ( W + 7 ) = A - 0.4) : A = 12.4 := 
by 
  sorry

end bowling_average_l824_824894


namespace min_sum_of_abc_conditions_l824_824781

theorem min_sum_of_abc_conditions
  (a b c d : ℕ)
  (hab : a + b = 2)
  (hac : a + c = 3)
  (had : a + d = 4)
  (hbc : b + c = 5)
  (hbd : b + d = 6)
  (hcd : c + d = 7) :
  a + b + c + d = 9 :=
sorry

end min_sum_of_abc_conditions_l824_824781


namespace vector_parallel_no_intersection_l824_824929

theorem vector_parallel_no_intersection (m : ℝ) :
  (∃ t s : ℝ, 
    (⟨1, 3⟩ : ℝ × ℝ) + t • (⟨6, -2⟩ : ℝ × ℝ) = (⟨4, 1⟩ : ℝ × ℝ) + s • (⟨-3, m⟩ : ℝ × ℝ)) 
  → False ↔ m = 1 :=
begin
  split,
  {
    intro h,
    cases h with t ht,
    sorry,
  },
  {
    intro hm,
    substitute m,
    intro h,
    cases h with t ht,
    sorry,
  }
end

end vector_parallel_no_intersection_l824_824929


namespace new_person_weight_l824_824816

noncomputable def weight_of_new_person (weight_of_replaced : ℕ) (number_of_persons : ℕ) (increase_in_average : ℕ) := 
  weight_of_replaced + number_of_persons * increase_in_average

theorem new_person_weight:
  weight_of_new_person 70 8 3 = 94 :=
  by
  -- Proof omitted
  sorry

end new_person_weight_l824_824816


namespace sequence_a_5_eq_21_l824_824451

theorem sequence_a_5_eq_21 :
  (∀ n : ℕ, n > 0 → ∃ a : ℕ → ℕ, 
    a 1 = 1 ∧ 
    (∀ n : ℕ, n > 0 → a (n + 1) - a n = 2 * n)) →
  a 5 = 21 :=
  sorry

end sequence_a_5_eq_21_l824_824451


namespace find_k_l824_824323

noncomputable def line1 (t : ℝ) (k : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + k * t)
noncomputable def line2 (s : ℝ) : ℝ × ℝ := (s, 1 - 2 * s)

def correct_k (k : ℝ) : Prop :=
  let slope1 := -k / 2
  let slope2 := -2
  slope1 * slope2 = -1

theorem find_k (k : ℝ) (h_perpendicular : correct_k k) : k = -1 :=
sorry

end find_k_l824_824323


namespace least_int_gt_sqrt_500_l824_824059

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824059


namespace product_of_divisors_eq_15625_l824_824448

theorem product_of_divisors_eq_15625 (n : ℕ) (h_pos : 0 < n)
  (h_prod : ∏ d in (finset.filter (λ d, n % d = 0) (finset.range (n + 1))), d = 15625) :
  n = 3125 :=
sorry

end product_of_divisors_eq_15625_l824_824448


namespace least_integer_greater_than_sqrt_500_l824_824006

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824006


namespace bob_second_week_hours_l824_824921

theorem bob_second_week_hours (total_earnings : ℕ) (total_hours_first_week : ℕ) (regular_hours_pay : ℕ) 
  (overtime_hours_pay : ℕ) (regular_hours_max : ℕ) (total_hours_overtime_first_week : ℕ) 
  (earnings_first_week : ℕ) (earnings_second_week : ℕ) : 
  total_earnings = 472 →
  total_hours_first_week = 44 →
  regular_hours_pay = 5 →
  overtime_hours_pay = 6 →
  regular_hours_max = 40 →
  total_hours_overtime_first_week = total_hours_first_week - regular_hours_max →
  earnings_first_week = regular_hours_max * regular_hours_pay + 
                          total_hours_overtime_first_week * overtime_hours_pay →
  earnings_second_week = total_earnings - earnings_first_week → 
  ∃ h, earnings_second_week = h * regular_hours_pay ∨ 
  earnings_second_week = (regular_hours_max * regular_hours_pay + (h - regular_hours_max) * overtime_hours_pay) ∧ 
  h = 48 :=
by 
  intros 
  sorry 

end bob_second_week_hours_l824_824921


namespace sqrt_500_least_integer_l824_824012

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824012


namespace number_of_dogs_total_l824_824919

theorem number_of_dogs_total
  (A : Finset ℕ) (B : Finset ℕ) (C : Finset ℕ)
  (n_fetch : A.card = 40)
  (n_jump : B.card = 35)
  (n_playdead : C.card = 22)
  (n_fetch_jump : (A ∩ B).card = 14)
  (n_jump_playdead : (B ∩ C).card = 10)
  (n_fetch_playdead : (A ∩ C).card = 16)
  (n_all_three : (A ∩ B ∩ C).card = 6)
  (n_none : 12)
  : A.card + B.card + C.card - (A ∩ B).card - (B ∩ C).card - (A ∩ C).card + (A ∩ B ∩ C).card + n_none = 75 := by
  sorry

end number_of_dogs_total_l824_824919


namespace trigonometric_identity_l824_824130

theorem trigonometric_identity (x : ℝ) (k : ℤ)
  (h1 : cos x ≠ 0)
  (h2 : cos (x / 2) ≠ 0)
  (h3 : 1 + sin (2 * x) - cos (2 * x) ≠ 0) :
  (1 + sin (2 * x) + cos (2 * x)) / (1 + sin (2 * x) - cos (2 * x)) + sin x * (1 + tan x * tan (x / 2)) = 4 ↔ 
  ∃ (k : ℤ), x = (if k % 2 = 0 then k else -k) * (π / 12) + (π * k) / 2 :=
sorry

end trigonometric_identity_l824_824130


namespace raja_first_half_speed_l824_824797

-- Define the conditions
def total_distance : ℝ := 225
def total_time : ℝ := 10
def second_half_speed : ℝ := 24
def half_distance : ℝ := total_distance / 2

-- Prove the speed at which Raja traveled the first half of the journey
theorem raja_first_half_speed :
  let second_half_time := half_distance / second_half_speed in
  let first_half_time := total_time - second_half_time in
  let first_half_speed := half_distance / first_half_time in
  first_half_speed = 21.176470588235294 :=
by 
  -- The proof part can be filled in later
  sorry

end raja_first_half_speed_l824_824797


namespace least_integer_greater_than_sqrt_500_l824_824041

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824041


namespace number_of_true_propositions_l824_824228

theorem number_of_true_propositions (a b c : ℝ) (h : a > b) :
  let P := ∀ (a b c : ℝ), a > b → (a * c^2 > b * c^2)
  let Q := ∀ (a b c : ℝ), (a * c^2 > b * c^2) → a > b
  let R := ∀ (a b c : ℝ), ¬(a > b → a * c^2 > b * c^2)
  (∃ n : ℕ, n = 2) :=
begin
  let P := ∀ (a b c : ℝ), a > b → (a * c^2 > b * c^2),
  let Q := ∀ (a b c : ℝ), (a * c^2 > b * c^2) → a > b,
  let R := ∀ (a b c : ℝ), ¬(a > b → a * c^2 > b * c^2),
  use 2,
  sorry
end

end number_of_true_propositions_l824_824228


namespace distance_between_parallel_lines_l824_824461

theorem distance_between_parallel_lines 
  (d : ℝ) 
  (r : ℝ)
  (h1 : (42 * 21 + (d / 2) * 42 * (d / 2) = 42 * r^2))
  (h2 : (40 * 20 + (3 * d / 2) * 40 * (3 * d / 2) = 40 * r^2)) :
  d = 3 + 3 / 8 :=
  sorry

end distance_between_parallel_lines_l824_824461


namespace inverse_of_f_l824_824565

noncomputable def f (x : ℝ) : ℝ := 6 - 8 * x + x^2

noncomputable def g (x : ℝ) : ℝ := 4 + sqrt (10 + x)

theorem inverse_of_f (x : ℝ) : f (g x) = x ∧ g (f x) = x :=
by
  sorry

end inverse_of_f_l824_824565


namespace trapezoid_area_increase_correct_l824_824524

noncomputable def trapezoid_area_increase : ℝ :=
let base1 := 10 in
let base2 := 20 in
let height := 5 in
let new_base1 := base1 * 1.15 in
let new_base2 := base2 * 1.25 in
let new_height := height * 1.1 in
let original_area := (1 / 2) * (base1 + base2) * height in
let new_area := (1 / 2) * (new_base1 + new_base2) * new_height in
new_area - original_area

theorem trapezoid_area_increase_correct : trapezoid_area_increase = 25.375 := by
  sorry

end trapezoid_area_increase_correct_l824_824524


namespace tenth_chick_grains_l824_824847

noncomputable def grains_pecked : ℕ → ℕ
| 1 := 40
| 2 := 60
| (n+1) := (List.sum (List.map grains_pecked (List.range n))) / n

theorem tenth_chick_grains : grains_pecked 10 = 50 := 
by 
  sorry

end tenth_chick_grains_l824_824847


namespace isosceles_triangle_angles_l824_824731

theorem isosceles_triangle_angles
  (ABC : Triangle)
  (h_isosceles : ABC.isosceles)
  (h_A_equals_C : ABC.angle_at_vertex B = ABC.angle_at_opposite_side C)
  (incircle_ratio : ℝ) 
  (h_ratio : ABC.dist_incenter_to B / ABC.dist_incenter_to C = incircle_ratio)
  (h_ABC : ABC.opposite_sides_equal A C)
  :
  ∃ α : ℝ, α = 2 * arcsin ((sqrt(8 * (incircle_ratio)^2 + 1) - 1) / (4 * incircle_ratio))
  ∧ ABC.angle_at_vertex A = α
  ∧ ABC.angle_at_vertex C = α
  ∧ ABC.angle_at_vertex B = π - 2 * α
  ∧ incircle_ratio > 0 := sorry

end isosceles_triangle_angles_l824_824731


namespace range_of_y_l824_824692

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ set.Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end range_of_y_l824_824692


namespace least_integer_greater_than_sqrt_500_l824_824109

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824109


namespace distance_between_foci_l824_824587

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 36 * x - 16 * y^2 - 64 * y = 144

-- Define the distance between the foci
def foci_distance : ℝ :=
  (2 * Real.sqrt (1189 / 72)) / (Real.sqrt 2)

-- Problem statement to prove the distance between the foci
theorem distance_between_foci {x y : ℝ} (h : hyperbola_eq x y) :
  ∃ x y, hyperbola_eq x y ∧ (2 * Real.sqrt (1189 / 72)) / (Real.sqrt 2) = foci_distance :=
sorry

end distance_between_foci_l824_824587


namespace total_rooms_count_l824_824395

noncomputable def apartment_area : ℕ := 160
noncomputable def living_room_area : ℕ := 60
noncomputable def other_room_area : ℕ := 20

theorem total_rooms_count (A : apartment_area = 160) (L : living_room_area = 60) (O : other_room_area = 20) :
  1 + (apartment_area - living_room_area) / other_room_area = 6 :=
by
  sorry

end total_rooms_count_l824_824395


namespace surface_area_of_interior_of_box_l824_824897

-- Definitions from conditions in a)
def length : ℕ := 25
def width : ℕ := 40
def cut_side : ℕ := 4

-- The proof statement we need to prove, using the correct answer from b)
theorem surface_area_of_interior_of_box : 
  (length - 2 * cut_side) * (width - 2 * cut_side) + 2 * (cut_side * (length + width - 2 * cut_side)) = 936 :=
by
  sorry

end surface_area_of_interior_of_box_l824_824897


namespace probability_no_shaded_square_l824_824150

theorem probability_no_shaded_square :
  let n := (2006 * 2005) / 2,
      m := 1003 * 1003 in
  (n - m) / n = 1002 / 2005 :=
by 
  sorry

end probability_no_shaded_square_l824_824150


namespace common_incircles_iff_rhombus_l824_824361

theorem common_incircles_iff_rhombus 
  (A B C D : Point)
  (convex_ABCD : ConvexQuadrilateral A B C D) :
  (∃ P : Point, P ∈ Incircle A B C ∧ P ∈ Incircle B C D ∧ P ∈ Incircle C D A ∧ P ∈ Incircle D A B) ↔ Rhombus A B C D := 
sorry

end common_incircles_iff_rhombus_l824_824361


namespace incorrect_statement_among_given_options_l824_824398

theorem incorrect_statement_among_given_options :
  (∀ (b h : ℝ), 3 * (b * h) = (3 * b) * h) ∧
  (∀ (b h : ℝ), 3 * (1 / 2 * b * h) = 1 / 2 * b * (3 * h)) ∧
  (∀ (π r : ℝ), 9 * (π * r * r) ≠ (π * (3 * r) * (3 * r))) ∧
  (∀ (a b : ℝ), (3 * a) / (2 * b) ≠ a / b) ∧
  (∀ (x : ℝ), x < 0 → 3 * x < x) →
  false :=
by
  sorry

end incorrect_statement_among_given_options_l824_824398


namespace correct_statements_cube_l824_824562

-- Definitions for the statements regarding the geometric shapes that can be formed
def can_form_rectangle (vertices : set (fin 3 → bool)) : Prop := sorry
def can_form_non_rect_parallelogram (vertices : set (fin 3 → bool)) : Prop := sorry
def can_form_all_eq_triangle_faces_tetrahedron (vertices : set (fin 3 → bool)) : Prop := sorry
def can_form_all_right_triangle_faces_tetrahedron (vertices : set (fin 3 → bool)) : Prop := sorry
def can_form_special_tetrahedron (vertices : set (fin 3 → bool)) : Prop := sorry

-- Placeholders for actual proofs that need to be filled out
axiom can_form_rectangle_cube (vertices : set (fin 3 → bool)) : can_form_rectangle vertices
axiom can_form_non_rect_parallelogram_cube (vertices : set (fin 3 → bool)) : can_form_non_rect_parallelogram vertices
axiom not_can_form_all_eq_triangle_faces_tetrahedron_cube (vertices : set (fin 3 → bool)) : ¬ can_form_all_eq_triangle_faces_tetrahedron vertices
axiom can_form_all_right_triangle_faces_tetrahedron_cube (vertices : set (fin 3 → bool)) : can_form_all_right_triangle_faces_tetrahedron vertices
axiom not_can_form_special_tetrahedron_cube (vertices : set (fin 3 → bool)) : ¬ can_form_special_tetrahedron vertices

theorem correct_statements_cube : let vertices : set (fin 3 → bool) := sorry in
  ([can_form_rectangle_cube vertices,
    can_form_non_rect_parallelogram_cube vertices,
    not_can_form_all_eq_triangle_faces_tetrahedron_cube vertices,
    can_form_all_right_triangle_faces_tetrahedron_cube vertices,
    not_can_form_special_tetrahedron_cube vertices].count (λ x, x = true)) = 3 := sorry

end correct_statements_cube_l824_824562


namespace simplify_expression_l824_824484

theorem simplify_expression :
  1 + 1 / (1 + 1 / (2 + 2)) = 9 / 5 := by
  sorry

end simplify_expression_l824_824484


namespace find_sum_of_coefficients_l824_824248

theorem find_sum_of_coefficients
  (a b c d : ℤ)
  (h1 : (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + x^3 - 2 * x^2 + 17 * x - 5) :
  a + b + c + d = 5 :=
by
  sorry

end find_sum_of_coefficients_l824_824248


namespace least_integer_greater_than_sqrt_500_l824_824084

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824084


namespace correct_option_D_l824_824483

theorem correct_option_D (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (sqrt a) * (sqrt b) = sqrt (a * b) := 
sorry

end correct_option_D_l824_824483


namespace least_integer_greater_than_sqrt_500_l824_824030

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824030


namespace polygon_is_hexagon_l824_824711

theorem polygon_is_hexagon (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  have hd : (n - 2) * 180 = 720 := by rw [h]
  have hn : n - 2 = 4 := by linarith
  rw [← hd, ← hn]
  linarith

end polygon_is_hexagon_l824_824711


namespace function_properties_l824_824970

noncomputable def f (x : ℝ) : ℝ := 4^x + 2*x - 2

noncomputable def g1 (x : ℝ) : ℝ := 4*x - 1
noncomputable def g2 (x : ℝ) : ℝ := (x - 1/2)^2
noncomputable def g3 (x : ℝ) : ℝ := Real.exp x - 1
noncomputable def g4 (x : ℝ) : ℝ := Real.log (Real.pi / x - 3)

theorem function_properties : 
  let f_zero := Classical.some (exists_zero_of_continuous f)
      g1_zero := Classical.some (exists_zero_of_continuous g1)
      g2_zero := Classical.some (exists_zero_of_continuous g2)
      g3_zero := Classical.some (exists_zero_of_continuous g3)
      g4_zero := Classical.some (exists_zero_of_continuous g4)
  in 
  (|f_zero - g1_zero| ≤ 0.25 ∧ |f_zero - g2_zero| ≤ 0.25) ∧ 
  (¬ (|f_zero - g3_zero| ≤ 0.25) ∧ ¬ (|f_zero - g4_zero| ≤ 0.25)) :=
sorry

end function_properties_l824_824970


namespace limit_trig_identity_l824_824205

theorem limit_trig_identity :
  filter.tendsto (λ x : ℝ, (1 - (real.sin (2 * x))) / ((real.pi - 4 * x)^2))
    (filter.nhds_within (real.pi / 4) filter.univ) 
    (filter.nhds (1 / 8)) :=
sorry

end limit_trig_identity_l824_824205


namespace solve_y_from_exponential_eq_l824_824418

theorem solve_y_from_exponential_eq :
  ∀ y : ℚ, (1/8 : ℚ)^(3 * y + 9) = (32 : ℚ)^(3 * y + 6) ↔ y = -57 / 24 :=
by
  sorry

end solve_y_from_exponential_eq_l824_824418


namespace equivalent_statements_l824_824128

variable (P Q : Prop)

theorem equivalent_statements :
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by
  sorry

end equivalent_statements_l824_824128


namespace probability_sum_even_l824_824195

noncomputable def unfair_die_probability_sum_even : ℚ := 
  let p_odd := 1 / 3 in
  let p_even := 2 / 3 in
  let prob_three_even := (p_even ^ 3) in
  let prob_three_odd := (p_odd ^ 3) in
  let prob_two_even_one_odd := 3 * (p_odd * (p_even ^ 2)) in
  prob_three_even + prob_three_odd + prob_two_even_one_odd

theorem probability_sum_even (h: unfair_die_probability_sum_even = 7 / 9) : 
  (∃ p : ℚ, p = unfair_die_probability_sum_even) :=
by
  use 7 / 9
  exact h

end probability_sum_even_l824_824195


namespace candy_bar_price_l824_824868

theorem candy_bar_price (total_money bread_cost candy_bar_price remaining_money : ℝ) 
    (h1 : total_money = 32)
    (h2 : bread_cost = 3)
    (h3 : remaining_money = 18)
    (h4 : total_money - bread_cost - candy_bar_price - (1 / 3) * (total_money - bread_cost - candy_bar_price) = remaining_money) :
    candy_bar_price = 1.33 := 
sorry

end candy_bar_price_l824_824868


namespace parabola_has_correct_equation_l824_824588

noncomputable def parabola_equation : Prop :=
  let focus_ellipse := (-1, 0)
  let vertex_parabola := (0, 0)
  ∃ (p : ℝ), focus_ellipse = (-p / 2, 0) ∧ vertex_parabola = (0,0) ∧ (y : ℝ) * (y : ℝ) = -4 * (x : ℝ)

theorem parabola_has_correct_equation : parabola_equation :=
  sorry

end parabola_has_correct_equation_l824_824588


namespace least_integer_greater_than_sqrt_500_l824_824031

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824031


namespace ellipse_hyperbola_same_foci_l824_824821

theorem ellipse_hyperbola_same_foci (a : ℝ) :
  (sqrt (4 - a^2) = sqrt (a^2 + 2)) → (a = 1 ∨ a = -1) :=
by
  sorry

end ellipse_hyperbola_same_foci_l824_824821


namespace binom_1300_2_eq_844350_l824_824554

theorem binom_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 :=
by
  sorry

end binom_1300_2_eq_844350_l824_824554


namespace smallest_positive_integer_divisible_12_15_16_exists_l824_824243

theorem smallest_positive_integer_divisible_12_15_16_exists :
  ∃ x : ℕ, x > 0 ∧ 12 ∣ x ∧ 15 ∣ x ∧ 16 ∣ x ∧ x = 240 :=
by sorry

end smallest_positive_integer_divisible_12_15_16_exists_l824_824243


namespace hyperbola_equation_proof_eccentricity_proof_l824_824222

-- Definition of hyperbola and conditions
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Given conditions
def conditions (a b c : ℝ) : Prop :=
  (c = 2) ∧ (a = b) ∧ (c = sqrt(2) * a)

-- Additional conditions for second part
def additional_conditions (θ c x y : ℝ) : Prop :=
  (θ = π / 6) ∧ (c = 2) ∧
  (x = (sqrt(3) / 2) * c) ∧ (y = (1 / 2) * c)

-- Proving the required equations
theorem hyperbola_equation_proof : ∀ (a b c x y : ℝ), 
  conditions a b c → hyperbola a b x y →
  a = sqrt(2) ∧ b = sqrt(2) ∧ (x^2 / 2 - y^2 / 2 = 1) :=
by sorry

theorem eccentricity_proof : ∀ (a b c θ x y e : ℝ),
  conditions a b c → additional_conditions θ c x y →
  hyperbola a b x y →
  e^2 = 2 ∧ e = sqrt(2) :=
by sorry

end hyperbola_equation_proof_eccentricity_proof_l824_824222


namespace least_integer_gt_sqrt_500_l824_824107

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824107


namespace find_b_l824_824686

theorem find_b (b : ℝ) : (∃ c : ℝ, (16 : ℝ) * x^2 + 40 * x + b = (4 * x + c)^2) → b = 25 :=
by
  sorry

end find_b_l824_824686


namespace sin_60_eq_sqrt3_div_2_l824_824203

theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_60_eq_sqrt3_div_2_l824_824203


namespace train_crossing_time_l824_824131

noncomputable def train_length : ℝ := 250
noncomputable def platform_length : ℝ := 300
noncomputable def train_speed_kmph : ℝ := 55
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def total_distance : ℝ := train_length + platform_length

theorem train_crossing_time :
  let time := total_distance / train_speed_mps in
  |time - 35.98| < 1 :=
by
  sorry

end train_crossing_time_l824_824131


namespace least_integer_gt_sqrt_500_l824_824103

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824103


namespace paula_go_kart_rides_l824_824402

theorem paula_go_kart_rides
  (g : ℕ)
  (ticket_cost_go_karts : ℕ := 4 * g)
  (ticket_cost_bumper_cars : ℕ := 20)
  (total_tickets : ℕ := 24) :
  ticket_cost_go_karts + ticket_cost_bumper_cars = total_tickets → g = 1 :=
by {
  sorry
}

end paula_go_kart_rides_l824_824402


namespace hyperbola_eccentricity_l824_824669

variable {a b c x₀ : ℝ}
variable (h_a : a > 0) (h_b : b > 0)
def C (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def asymptote_eq (x : ℝ) : ℝ := (b / a) * x
def M := (x₀, asymptote_eq a x₀)

theorem hyperbola_eccentricity {F M : ℝ × ℝ} (h_point_on_asymptote : M = (x₀, b*x₀/a))
  (h_distance_O_M : |M.1| + |M.2| = a)
  (h_slope_MF : ((b*x₀/a)/(x₀ - c)) = - (b/a))
  : let e := c / a in e = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l824_824669


namespace probability_no_shaded_square_l824_824155

theorem probability_no_shaded_square (n m : ℕ) (h1 : n = (2006.choose 2)) (h2 : m = 1003^2) : 
  (1 - (m / n) = (1002 / 2005)) := 
by
  -- Number of rectangles in one row
  have hn : n = 1003 * 2005 := h1
  -- Number of rectangles in one row containing a shaded square
  have hm : m = 1003 * 1003 := h2
  sorry

end probability_no_shaded_square_l824_824155


namespace find_a_l824_824708

theorem find_a (a : ℝ) :
  {x : ℝ | (x + a) / ((x + 1) * (x + 3)) > 0} = {x : ℝ | x > -3 ∧ x ≠ -1} →
  a = 1 := 
by sorry

end find_a_l824_824708


namespace max_edges_partitioned_square_l824_824699

theorem max_edges_partitioned_square (n v e : ℕ) 
  (h : v - e + n = 1) : e ≤ 3 * n + 1 := 
sorry

end max_edges_partitioned_square_l824_824699


namespace quadratic_polynomial_with_root_and_real_coeffs_l824_824602

theorem quadratic_polynomial_with_root_and_real_coeffs :
  ∃ (p : ℝ[X]), (p.coeff 2 = 3) ∧ (p.coeff 1 = -24) ∧ (p.coeff 0 = 48) ∧ 
  (∀ z : ℂ, (z = 4 + 2*complex.i ∨ z = 4 - 2*complex.i) → (p.eval z = 0) ) :=
by
  sorry

end quadratic_polynomial_with_root_and_real_coeffs_l824_824602


namespace least_integer_greater_than_sqrt_500_l824_824042

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824042


namespace pow_sum_geq_pow_prod_l824_824407

theorem pow_sum_geq_pow_prod (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x^5 + y^5 ≥ x^4 * y + x * y^4 :=
 by sorry

end pow_sum_geq_pow_prod_l824_824407


namespace min_value_of_expression_l824_824315

theorem min_value_of_expression {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : (a + b) * (a + c) = 4) : 2 * a + b + c ≥ 4 :=
sorry

end min_value_of_expression_l824_824315


namespace least_integer_greater_than_sqrt_500_l824_824010

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824010


namespace circle_passes_through_focus_l824_824173

noncomputable def parabola := {p : ℝ × ℝ | p.2 ^ 2 = 8 * p.1}
def line := {p : ℝ × ℝ | p.1 = -2}
def focus : ℝ × ℝ := (2, 0)
def center_circle (c : ℝ × ℝ) := c ∈ parabola
def tangent_circle (c : ℝ × ℝ) [radius := dist c focus] := ∀ p : ℝ × ℝ, (dist p line = radius) 

theorem circle_passes_through_focus (c : ℝ × ℝ) (pf_center : center_circle c) (pf_tangent : tangent_circle c) : 
    dist c focus = dist focus line := sorry

end circle_passes_through_focus_l824_824173


namespace value_of_expression_l824_824941

theorem value_of_expression :
  (243 : ℝ) ^ (-(2 : ℝ) ^ -3) = (1 / (3 : ℝ) ^ (5 / 8) : ℝ) :=
by
  sorry

end value_of_expression_l824_824941


namespace probability_X_eq_Y_l824_824530

-- Define the domain and conditions
def domain : set ℝ := { x | -2 * real.pi ≤ x ∧ x ≤ 2 * real.pi }

-- Define the condition function
def condition (x y : ℝ) : Prop := real.cos (real.cos x) = real.cos (real.cos y)

-- Statement of the problem
theorem probability_X_eq_Y :
  ∀ X Y : ℝ,
  X ∈ domain → Y ∈ domain → condition X Y →
  (∑ i in [0 : 4].to_finset, 1) = 5 →
  (∑ i in [0 : 4].to_finset, ∑ j in [0 : 4].to_finset, if i = j then 1 else 0) = 5 →
  5 / (5 * 5) = 1 / 5 :=
  sorry

end probability_X_eq_Y_l824_824530


namespace vans_needed_for_trip_l824_824495

theorem vans_needed_for_trip (total_people : ℕ) (van_capacity : ℕ) (h_total_people : total_people = 24) (h_van_capacity : van_capacity = 8) : ℕ :=
  let exact_vans := total_people / van_capacity
  let vans_needed := if total_people % van_capacity = 0 then exact_vans else exact_vans + 1
  have h_exact : exact_vans = 3 := by sorry
  have h_vans_needed : vans_needed = 4 := by sorry
  vans_needed

end vans_needed_for_trip_l824_824495


namespace simson_line_is_parallel_l824_824791

noncomputable def circle_center (O : Point) (A B C P Q : Point) (α β γ : ℝ) : Prop :=
  let θ := (α + β + γ) / 2
  ∧ ∠POA = α
  ∧ ∠POB = β
  ∧ ∠POC = γ
  ∧ ∠POQ = θ

noncomputable def simson_line_parallel (O A B C P Q : Point) (α β γ : ℝ) : Prop :=
  Simson_line_parallel_to O A B C P Q (circle_center O A B C P Q α β γ)

theorem simson_line_is_parallel (O A B C P Q : Point) (α β γ : ℝ)
  (h_circle : circle_center O A B C P Q α β γ) :
  simson_line_parallel O A B C P Q α β γ :=
  sorry

end simson_line_is_parallel_l824_824791


namespace binomial_1300_2_eq_844350_l824_824557

theorem binomial_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 := 
by
  sorry

end binomial_1300_2_eq_844350_l824_824557


namespace smallest_prime_factor_in_C_l824_824755

-- Definition of the set C
def C : Set ℕ := {63, 65, 68, 71, 73}

-- Function to compute the smallest prime factor of a number
noncomputable def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then
    Nat.find_gcd (Prime.prod_of_nat_prime_factors_separable h).rem h.prime_factors_nonempty
  else 0

-- Main theorem statement
theorem smallest_prime_factor_in_C : 
  ∀ n ∈ C, 
    (smallest_prime_factor n = min (smallest_prime_factor 63) 
                                 (min (smallest_prime_factor 65) 
                                      (min (smallest_prime_factor 68) 
                                           (min (smallest_prime_factor 71) 
                                                (smallest_prime_factor 73)))))
  → n = 68 := sorry

end smallest_prime_factor_in_C_l824_824755


namespace quadratic_polynomial_with_root_and_real_coeffs_l824_824601

theorem quadratic_polynomial_with_root_and_real_coeffs :
  ∃ (p : ℝ[X]), (p.coeff 2 = 3) ∧ (p.coeff 1 = -24) ∧ (p.coeff 0 = 48) ∧ 
  (∀ z : ℂ, (z = 4 + 2*complex.i ∨ z = 4 - 2*complex.i) → (p.eval z = 0) ) :=
by
  sorry

end quadratic_polynomial_with_root_and_real_coeffs_l824_824601


namespace tangential_circles_sum_l824_824325

namespace Geometry
open EuclideanGeometry

theorem tangential_circles_sum {a b c r₁ r₂ : ℝ} 
    (h : ∃ Δ, right_triangle Δ ∧ Δ.circumcircle.exists ∧ 
    Δ.circle_one.exists ∧ Δ.circle_two.exists) :
    r₁ + r₂ = a + b - c := by
 sorry
end Geometry

end tangential_circles_sum_l824_824325


namespace apple_distribution_l824_824567

theorem apple_distribution : ∃ methods : set (fin 6 × fin 6 × fin 6), 
  (∀ t ∈ methods, t.1.val + t.2.val + t.3.val = 10) ∧
  (∀ t ∈ methods, 1 ≤ t.1.val ∧ t.1.val ≤ 5 ∧ 
                   1 ≤ t.2.val ∧ t.2.val ≤ 5 ∧ 
                   1 ≤ t.3.val ∧ t.3.val ≤ 5) ∧
  (methods.card = 4) :=
by
  sorry

end apple_distribution_l824_824567


namespace pizza_slices_remaining_l824_824714

-- Step d) Lean 4 statement
theorem pizza_slices_remaining :
  let large_pizza_slices := 8 in
  let extra_large_pizza_slices := 12 in
  let mary_eats_large := 7 in
  let mary_eats_extra_large := 3 in
  let john_eats_large := 2 in
  let john_eats_extra_large := 5 in
  let remaining_large_after_mary := large_pizza_slices - mary_eats_large in
  let remaining_extra_large_after_mary := extra_large_pizza_slices - mary_eats_extra_large in
  let remaining_large_after_john := max 0 (remaining_large_after_mary - john_eats_large) in
  let remaining_extra_large_after_john := max 0 (remaining_extra_large_after_mary - john_eats_extra_large) in
  remaining_large_after_john + remaining_extra_large_after_john = 4 := sorry

end pizza_slices_remaining_l824_824714


namespace problem_solution_l824_824986

def Point : Type := ℝ × ℝ × ℝ
def Vector := Point

-- Definitions of points A, B (with variable k), and C
def A : Point := (1, 2, -1)
def B (k : ℝ) : Point := (2, k, -3)
def C : Point := (0, 5, 1)

-- Definition of vector a
def a : Vector := (-3, 4, 5)

-- Calculate vector AB
def AB (k : ℝ) : Vector := (1, k - 2, -2)

-- Calculate vector AC
def AC : Vector := (-1, 3, 2)

-- Condition: AB ⊥ a implies their dot product is 0
def perpendicular (v1 v2 : Vector) : Prop :=
  (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3) = 0

-- Calculate the dot product of two vectors
def dot (v1 v2 : Vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Find projection of vector AC onto vector a
def proj (v1 v2 : Vector) : Vector :=
  let scale := (dot v1 v2) / (v2.1 * v2.1 + v2.2 * v2.2 + v2.3 * v2.3)
  (scale * v2.1, scale * v2.2, scale * v2.3)

-- Main theorem to be proved:
theorem problem_solution : ∃ k : ℝ, 
  perpendicular (AB k) a ∧ 
  k = 21 / 4 ∧ 
  proj AC a = (-3 / 2, 2, 5 / 2) := 
  by 
    sorry

end problem_solution_l824_824986


namespace fourth_term_of_geometric_sequence_is_320_l824_824167

theorem fourth_term_of_geometric_sequence_is_320
  (a : ℕ) (r : ℕ)
  (h_a : a = 5)
  (h_fifth_term : a * r^4 = 1280) :
  a * r^3 = 320 := 
by
  sorry

end fourth_term_of_geometric_sequence_is_320_l824_824167


namespace binomial_1300_2_eq_844350_l824_824559

theorem binomial_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 := 
by
  sorry

end binomial_1300_2_eq_844350_l824_824559


namespace least_integer_greater_than_sqrt_500_l824_824029

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824029


namespace f_125_eq_1197_l824_824823

noncomputable def f : ℤ → ℤ
  | n => if n >= 1200 then n - 4 else f (f (n + 6))

theorem f_125_eq_1197 : f 125 = 1197 := 
by
  sorry

end f_125_eq_1197_l824_824823


namespace total_cost_calculation_l824_824216

def total_transportation_cost (x : ℝ) : ℝ :=
  let cost_A_to_C := 20 * x
  let cost_A_to_D := 30 * (240 - x)
  let cost_B_to_C := 24 * (200 - x)
  let cost_B_to_D := 32 * (60 + x)
  cost_A_to_C + cost_A_to_D + cost_B_to_C + cost_B_to_D

theorem total_cost_calculation (x : ℝ) :
  total_transportation_cost x = 13920 - 2 * x := by
  sorry

end total_cost_calculation_l824_824216


namespace determine_domain_of_y_l824_824936

noncomputable def domain_of_function : Set ℝ :=
  { x : ℝ | ∃ k : ℤ, ∀ x, (2 * k * Real.pi) < x ∧ x < (Real.pi / 2 + 2 * k * Real.pi) }

theorem determine_domain_of_y :
  (∀ x : ℝ, sin x ≥ 0 ∧ cos x > 0 ∧ tan x ≠ 0 → x ∈ domain_of_function) :=
by
  intro x
  rw [Set.mem_setOf_eq]
  intro h
  sorry

end determine_domain_of_y_l824_824936


namespace polynomial_two_factors_l824_824696

theorem polynomial_two_factors (h k : ℝ)
  (h1 : ∀ x, f x = 2 * x^3 - h * x + k)
  (h2 : f (-2) = 0)
  (h3 : f 1 = 0) :
  2 * h - 3 * k = 0 :=
by
  -- The conditions as equations
  have eq1 : 2 * (-2)^3 - h * (-2) + k = 0 := by rw [h1, h2];
  have eq2 : 2 * 1^3 - h * 1 + k = 0 := by rw [h1, h3];

  -- Derive values from the equations
  sorry

end polynomial_two_factors_l824_824696


namespace sufficient_condition_l824_824842

theorem sufficient_condition (a : ℝ) : (∃ x ∈ set.Icc 1 2, x^2 ≤ a) → a ≥ 2 := by
  sorry

end sufficient_condition_l824_824842


namespace minimum_value_l824_824985

open Classical -- Allow classical logic

noncomputable def geometric_sequence {m n : ℕ} (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ k : ℕ, a k = a 0 * q ^ k) ∧
  (∀ i : ℕ, 0 < a i) ∧
  sqrt (a m * a n) = 4 * a 0 ∧
  a 6 = a 5 + 2 * a 4

theorem minimum_value (a : ℕ → ℝ) (m n : ℕ) (h : geometric_sequence a) : 
  m + n = 6 → (1 / m + 4 / n) ≥ 3 / 2 :=
by 
  sorry

end minimum_value_l824_824985


namespace range_of_m_l824_824320

noncomputable def range_m (m : ℝ) : set ℝ :=
  {x : ℝ | ∃ y : ℝ, y = x / sqrt (m * x^2 + m * x + 1)}

theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, ∃ y : ℝ, y = x / sqrt (m * x^2 + m * x + 1)) ↔ 0 ≤ m ∧ m < 4 :=
by
  intros m
  sorry

end range_of_m_l824_824320


namespace integer_points_between_A_B_l824_824172

/-- 
Prove that the number of integer coordinate points strictly between 
A(2, 3) and B(50, 80) on the line passing through A and B is c.
-/
theorem integer_points_between_A_B 
  (A B : ℤ × ℤ) (hA : A = (2, 3)) (hB : B = (50, 80)) 
  (c : ℕ) :
  ∃ (n : ℕ), n = c ∧ ∀ (x y : ℤ), (A.1 < x ∧ x < B.1) → (A.2 < y ∧ y < B.2) → 
              (y = ((A.2 - B.2) / (A.1 - B.1) * x + 3 - (A.2 - B.2) / (A.1 - B.1) * 2)) :=
by {
  sorry
}

end integer_points_between_A_B_l824_824172


namespace double_root_polynomial_l824_824179

theorem double_root_polynomial (b4 b3 b2 b1 : ℤ) (s : ℤ) :
  (Polynomial.eval s (Polynomial.C 1 * Polynomial.X^5 + Polynomial.C b4 * Polynomial.X^4 + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C 24) = 0)
  ∧ (Polynomial.eval s (Polynomial.derivative (Polynomial.C 1 * Polynomial.X^5 + Polynomial.C b4 * Polynomial.X^4 + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C 24)) = 0)
  → s = 1 ∨ s = -1 ∨ s = 2 ∨ s = -2 :=
by
  sorry

end double_root_polynomial_l824_824179


namespace minimum_area_AOB_line_eq_correct_l824_824656

variable (a b x y : Real)
variable (P : Real × Real) (l_eq : Real → Real → Prop)
variable (A B: Real × Real)

-- Define points and their coordinates
def P : Real × Real := (3, 4)
def A : Real × Real := (a, 0)
def B : Real × Real := (0, b)

-- Define the line equation passing through P and the intercept condition
def line_eq_P (x y : Real) : Prop := (y == (4/3) * x) ∨ ((2 * x) + y == 10)

-- Conditions (rewrite in Lean)
axiom (line_condition : (y == (4 / 3) * x) ∨ ((2 * x) + y == 10))
axiom (passes_through_P : line_condition 3 4)

-- Calculate area of triangle AOB
def triangle_area (A B : Real × Real) : Real := (1 / 2) * A.1 * B.2

-- Applying AM-GM inequality and defining intercepts
def min_area : Real := 24

-- Prove the minimum area condition
theorem minimum_area_AOB (a b : Real) (h₁ : b = 2 * a)
  (h₂ : 3 / a + 4 / (2 * a) = 1) : 
  (∀ x y, line_eq_P x y → triangle_area A B ≥ 24) :=
sorry

theorem line_eq_correct :
  (line_condition 3 4 → (line_eq_P 3 4)) :=
sorry

end minimum_area_AOB_line_eq_correct_l824_824656


namespace number_of_valid_paths_l824_824504

theorem number_of_valid_paths :
  let start := (-5, -5)
  let end := (5, 5)
  let steps := 20
  let boundary := λ (x y : ℤ), (x ≤ -3 ∨ x ≥ 3 ∨ y ≤ -3 ∨ y ≥ 3)
  ∃ (f : ℕ → (ℤ × ℤ)), 
    (f 0 = start) ∧ 
    (f steps = end) ∧ 
    (∀ i < steps, (f (i+1) = (f i).1 + 1 ∨ f (i+1) = (f i).2 + 1)) ∧
    (∀ i ≤ steps, boundary (f i).1 (f i).2) → 
    (nat.choose 20 10) * (nat.choose 10 2) * (nat.choose 10 2) = 4252 := 
by sorry

end number_of_valid_paths_l824_824504


namespace exists_k_eq_one_or_three_infinite_solutions_k_one_three_l824_824144

theorem exists_k_eq_one_or_three (a b c : ℕ) (h_nonzero : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ k, (a^2 + b^2 + c^2 = k * a * b * c) ↔ (k = 1 ∨ k = 3) :=
by sorry

theorem infinite_solutions_k_one_three :
  ∀ k ∈ {1, 3}, ∃ (a b c : ℕ) (n : ℕ), 
    (a^2 + b^2 + c^2 = k * a * b * c) ∧ 
    (∀ m, a * b = m^2 + n^2) ∧ 
    (a * c = m^2 + n^2) ∧ 
    (b * c = m^2 + n^2) :=
by sorry

end exists_k_eq_one_or_three_infinite_solutions_k_one_three_l824_824144


namespace find_smallest_n_l824_824360

def is_pairwise_rel_prime (nums : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ nums → b ∈ nums → a ≠ b → Nat.gcd a b = 1

theorem find_smallest_n
    (S : Set ℕ) (h_S : S = Finset.range 281 .toSet) : 
    ∃ n, (∀ (X : Finset ℕ), (X ⊆ S) → (|X| = n) → ∃ (subset : Finset ℕ), subset ⊆ X ∧ |subset| >= 5 ∧ is_pairwise_rel_prime subset) ∧ n = 217 := 
sorry

end find_smallest_n_l824_824360


namespace points_concyclic_l824_824635

-- Lean 4 statement for the translated proof problem

theorem points_concyclic 
  {A B C P Q R S : Type} [Field A B C] [Point P Q R S] 
  (circle_ABC : Circle A B) 
  (hP : circle_ABC intersects Line AC at P) 
  (hQ : circle_ABC intersects Line BC at Q)
  (hR : R ∈ Line AB ∧ QR ∥ CA)
  (hS : S ∈ Line AB ∧ PS ∥ CB) : 
  Concyclic P Q R S :=
sorry

end points_concyclic_l824_824635


namespace number_of_graphing_calculators_in_class_l824_824722

-- Define a structure for the problem
structure ClassData where
  num_boys : ℕ
  num_girls : ℕ
  num_scientific_calculators : ℕ
  num_girls_with_calculators : ℕ
  num_graphing_calculators : ℕ
  no_overlap : Prop

-- Instantiate the problem using given conditions
def mrs_anderson_class : ClassData :=
{
  num_boys := 20,
  num_girls := 18,
  num_scientific_calculators := 30,
  num_girls_with_calculators := 15,
  num_graphing_calculators := 10,
  no_overlap := true
}

-- Lean statement for the proof problem
theorem number_of_graphing_calculators_in_class (data : ClassData) :
  data.num_graphing_calculators = 10 :=
by
  sorry

end number_of_graphing_calculators_in_class_l824_824722


namespace exist_even_odd_predecessors_l824_824835

-- The set M is defined as {1, 2, ..., 2n} and is partitioned into k subsets
variable (n k : ℕ) (M : Set ℕ) [finite M]
variable (M_subsets : Fin k → Set ℕ)
-- Condition: n >= k^3 + k
variable (h : n ≥ k^3 + k)

-- Given that M = {1, 2, ..., 2n}
def M_def : Set ℕ := {x : ℕ | 1 ≤ x ∧ x ≤ 2 * n}

-- Assume that M is partitioned into non-intersecting subsets M_1, M_2, ..., M_k
axiom partition (M : Set ℕ) (M_subsets : Fin k → Set ℕ) : 
(∀ i j : Fin k, i ≠ j → M_subsets i ∩ M_subsets j = ∅) ∧ (⋃ i, M_subsets i = M)

-- Main theorem statement
theorem exist_even_odd_predecessors :
∃ (M_i M_j : Fin k) (even_numbers : Fin (k+1) → ℕ),
(even_numbers.all (λ j, 2 * j ∈ M_subsets M_i)) ∧ 
(∃ (odd_predecessors : Fin (k+1) → ℕ), 
  (odd_predecessors = (λ j, 2 * j - 1)) ∧ 
  (odd_predecessors.all (λ j, odd_predecessors j ∈ M_subsets M_j))) :=
sorry

end exist_even_odd_predecessors_l824_824835


namespace peanuts_remaining_l824_824844

theorem peanuts_remaining (initial_peanuts brock_ate bonita_ate brock_fraction : ℕ) (h_initial : initial_peanuts = 148) (h_brock_fraction : brock_fraction = 4) (h_brock_ate : brock_ate = initial_peanuts / brock_fraction) (h_bonita_ate : bonita_ate = 29) :
  (initial_peanuts - brock_ate - bonita_ate) = 82 :=
by
  sorry

end peanuts_remaining_l824_824844


namespace barbara_winning_strategy_part_a_alice_winning_strategy_part_b_l824_824529

-- Part (a)
theorem barbara_winning_strategy_part_a :
  ∃ strategy_B : (ℝ → ℝ) → (ℝ → ℝ), ∀ (strategy_A : ℝ → ℝ) (turns : ℕ), (∀ n, 0 < strategy_A n - n < 1) ∧
    (∀ n, 0 < strategy_B (strategy_A n) - n < 1) ∧ (strategy_A(2009) < 2010) ∧ 
    (strategy_B(strategy_A(2009)) = 2010) :=
sorry

-- Part (b)
theorem alice_winning_strategy_part_b :
  ∃ strategy_A : (ℝ → ℝ) → (ℝ → ℝ), ∀ (strategy_B : ℝ → ℝ) (turns : ℕ), 
  (∀ n, 0 < strategy_A n - n < 1) ∧ (∀ n, 0 < strategy_B (strategy_A n) - n < 1) ∧
    (strategy_A 2010 < 2010) ∧
    (strategy_B (strategy_A 2010) < 2010) ∧
    (strategy_A 2011 ≥ 2010) :=
sorry

end barbara_winning_strategy_part_a_alice_winning_strategy_part_b_l824_824529


namespace least_integer_greater_than_sqrt_500_l824_824074

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824074


namespace loss_percentage_on_first_book_l824_824681

variable (C1 C2 SP L : ℝ)
variable (total_cost : ℝ := 540)
variable (C1_value : ℝ := 315)
variable (gain_percentage : ℝ := 0.19)
variable (common_selling_price : ℝ := 267.75)

theorem loss_percentage_on_first_book :
  C1 = C1_value →
  C2 = total_cost - C1 →
  SP = 1.19 * C2 →
  SP = C1 - (L / 100 * C1) →
  L = 15 :=
sorry

end loss_percentage_on_first_book_l824_824681


namespace max_coin_sums_l824_824168

-- Definitions representing the coin values
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25

-- Set of coins in the purse
def coins : List ℕ := [penny, penny, penny, nickel, nickel, dime, dime, quarter, quarter]

-- Function to calculate the sum of all distinct pairs of coins drawn randomly
def coin_sums (coins : List ℕ) : Set ℕ :=
  (coins.product coins).map (λ pair, pair.1 + pair.2).toSet

-- Define the maximum number of different sums
def max_num_of_different_sums : ℕ := 10

-- Statement to be proven
theorem max_coin_sums (coins : List ℕ) (h_coins : coins = [penny, penny, penny, nickel, nickel, dime, dime, quarter, quarter]) :
  card (coin_sums coins) = max_num_of_different_sums := 
by {
  sorry
}

end max_coin_sums_l824_824168


namespace symmetrical_point_l824_824729

theorem symmetrical_point (P : ℝ × ℝ × ℝ) (h : P = (4, -1, 2)) : 
  ∃ Q : ℝ × ℝ × ℝ, Q = (-4, 1, -2) :=
by
  use (-4, 1, -2)
  sorry

end symmetrical_point_l824_824729


namespace cloth_coloring_problem_l824_824884

theorem cloth_coloring_problem (lengthOfCloth : ℕ) 
  (women_can_color_100m_in_1_day : 5 * 1 = 100) 
  (women_can_color_in_3_days : 6 * 3 = lengthOfCloth) : lengthOfCloth = 360 := 
sorry

end cloth_coloring_problem_l824_824884


namespace solve_for_x_and_2y_l824_824258

noncomputable def complex_eq_real_combination (x y : ℝ) : Prop :=
  let z := x + y * complex.I in
  abs (z - 4 * complex.I) = abs (z + 2)

theorem solve_for_x_and_2y (x y : ℝ) (h : complex_eq_real_combination x y) :
  x + 2 * y = 3 :=
by
  -- proof to be filled in
  sorry

end solve_for_x_and_2y_l824_824258


namespace lines_are_concurrent_l824_824382

-- Define the initial setup
variable (A B C A₁ B₁ C₁ : Point)
variable isAcuteTriangleABC : ∀ {ABC : Triangle}, ABC.isAcute
variable centerOfSquareOnBC : isCenterOfInscribedSquare A₁ (Triangle.mk A B C) BC
variable centerOfSquareOnAC : isCenterOfInscribedSquare B₁ (Triangle.mk A B C) AC
variable centerOfSquareOnAB : isCenterOfInscribedSquare C₁ (Triangle.mk A B C) AB

-- Theorem statement
theorem lines_are_concurrent :
  areConcurrent (Line.mk A A₁) (Line.mk B B₁) (Line.mk C C₁) :=
sorry

end lines_are_concurrent_l824_824382


namespace ratio_of_angles_l824_824911

open Real

noncomputable def ABCD_is_square (A B C D : Point) : Prop :=
is_square A B C D

noncomputable def BF_parallel_AC (B F A C : Point) : Prop :=
is_parallel B F A C

noncomputable def AECF_is_rhombus (A E C F : Point) : Prop :=
is_rhombus A E C F

noncomputable def measure_angle (P Q R : Point) : ℝ :=
angle P Q R

theorem ratio_of_angles (A B C D E F : Point)
  (h1 : ABCD_is_square A B C D)
  (h2 : BF_parallel_AC B F A C)
  (h3 : AECF_is_rhombus A E C F) :
  measure_angle A C F / measure_angle F = 5 :=
sorry

end ratio_of_angles_l824_824911


namespace digit_in_2710th_position_l824_824906

/-- Prove the digit at the 2710th position in a sequence formed by writing natural numbers from 999 to 1 consecutively and in descending order is 9. -/
theorem digit_in_2710th_position :
  let sequence := List.join (List.map (λ n : Nat => toString (999 - n)) (List.range 999))
  sequence.getD 2709 '\0' = '9' := sorry

end digit_in_2710th_position_l824_824906


namespace unique_sums_count_l824_824739

theorem unique_sums_count :
  ∃ (unique_sums : ℕ), unique_sums = 3 ∧
    ∃ (a b c d : ℕ), 
      {a, b, c, d} = {1, 2, 3, 5} ∧
      (let sums := 
        { (a * b) + (c * d) | (a b c d : ℕ), 
          {a, b, c, d} = {1, 2, 3, 5}} in
         (sums.size = unique_sums)) :=
by
  sorry

end unique_sums_count_l824_824739


namespace emma_bank_account_remains_0_32_l824_824571

/-- Emma's bank account problem -/
theorem emma_bank_account_remains_0_32 : 
  (let initial_balance : ℝ := 100 in  -- Initial balance
   let daily_spending : ℝ := 8 in  -- Daily spending
   let currency_fee_rate : ℝ := 0.03 in  -- Currency fee rate
   let days_in_week : ℕ := 7 in  -- Number of days in a week
   let flat_fee : ℝ := 2 in  -- Flat fee for getting $5 bills
   let bill_value : ℝ := 5 in  -- Value of each $5 bill
   -- Calculations:
   let daily_total_spending := daily_spending + daily_spending * currency_fee_rate in
   let weekly_spending := daily_total_spending * days_in_week in
   let post_week_balance := initial_balance - weekly_spending in
   let post_fee_balance := post_week_balance - flat_fee in
   let number_of_bills := (post_fee_balance / bill_value).floor.to_nat in
   let spent_on_bills := number_of_bills * bill_value in
   let remaining_balance := post_fee_balance - spent_on_bills in
   remaining_balance = 0.32) 
:= 
by { sorry }

/- 
Explanation of Lean 4 statement:
1. The initial balance, daily spending, fee rate, days in a week, flat fee, and value of $5 bills are defined.
2. Intermediate calculations are performed as per the conditions to determine the remaining balance.
3. The theorem statement proves that the final remaining balance is $0.32.
-/

end emma_bank_account_remains_0_32_l824_824571


namespace polka_dot_blankets_l824_824574

theorem polka_dot_blankets (total_blankets : ℕ) (initial_polka_dot_fraction : ℚ) (additional_polka_dot : ℕ) 
  (h1 : total_blankets = 24) 
  (h2 : initial_polka_dot_fraction = 1/3) 
  (h3 : additional_polka_dot = 2) : 
  let initial_polka_dots := (total_blankets : ℚ) * initial_polka_dot_fraction,
      total_polka_dots := (initial_polka_dots : ℕ) + additional_polka_dot in
  total_polka_dots = 10 :=
by sorry

end polka_dot_blankets_l824_824574


namespace least_integer_greater_than_sqrt_500_l824_824046

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824046


namespace smallest_root_l824_824419

theorem smallest_root :
  ∃ (x : ℚ), ((x - 5/6) ^ 2 + (x - 5/6) * (x - 2/3) = 0) ∧ (x ^ 2 - 2 * x + 1 ≥ 0) ∧ x = 5/6 :=
begin
  sorry
end

end smallest_root_l824_824419


namespace vet_donates_correct_amount_l824_824191

-- Define the conditions
def vet_fee_dog : ℕ := 15
def vet_fee_cat : ℕ := 13
def dog_adopters : ℕ := 8
def cat_adopters : ℕ := 3

-- Define the total fee calculation
def total_fee_collected : ℕ := (dog_adopters * vet_fee_dog) + (cat_adopters * vet_fee_cat)

-- Prove that the vet donates one third of the total fee collected back to the shelter
def vet_donation : ℕ := total_fee_collected / 3

-- Statement of the proof problem
theorem vet_donates_correct_amount : vet_donation = 53 :=
by
  -- Calculate total vet fees collected
  have total_fee : ℕ := total_fee_collected
  -- Calculate the donation
  have donation : ℕ := total_fee / 3
  -- The donation should be equal to $53
  exact Eq.refl 53

  sorry -- proof to be filled

end vet_donates_correct_amount_l824_824191


namespace sin_alpha_in_second_quadrant_l824_824989

theorem sin_alpha_in_second_quadrant (α y : ℝ)
  (h1 : P = (-√3, y))
  (h2 : cos α = -√15 / 5) 
  (h3 : α ∈ set_of_quadrant_2) : 
  sin α = √10 / 5 :=
sorry

end sin_alpha_in_second_quadrant_l824_824989


namespace base10_representation_of_n_l824_824443

theorem base10_representation_of_n (a b c n : ℕ) (ha : a > 0)
  (h14 : n = 14^2 * a + 14 * b + c)
  (h15 : n = 15^2 * a + 15 * c + b)
  (h6 : n = 6^3 * a + 6^2 * c + 6 * a + c) : n = 925 :=
by sorry

end base10_representation_of_n_l824_824443


namespace function_properties_l824_824299

-- Define the function f and the required conditions
def f (x : ℝ) : ℝ := x^λ + x^(-λ)

-- Statement of the problem
theorem function_properties (λ : ℝ) (hλ : λ > 0) : 
  (∀ x y z : ℝ, (0 < x) → (0 < y) → (0 < z) → f(x * y * z) + f(x) + f(y) + f(z) = 
     f(Real.sqrt (x * y)) * f(Real.sqrt (y * z)) * f(Real.sqrt (z * x)))
  ∧ ∀ x y : ℝ, (1 ≤ x ∧ x < y) → f(x) < f(y) :=
by
  -- Skip the proof for now with sorry
  sorry

end function_properties_l824_824299


namespace investment_period_l824_824534

theorem investment_period (P A : ℝ) (r n t : ℝ)
  (hP : P = 4000)
  (hA : A = 4840.000000000001)
  (hr : r = 0.10)
  (hn : n = 1)
  (hC : A = P * (1 + r / n) ^ (n * t)) :
  t = 2 := by
-- Adding a sorry to skip the actual proof.
sorry

end investment_period_l824_824534


namespace armand_guessing_game_difference_l824_824196

theorem armand_guessing_game_difference : 
  ∃ (diff : ℕ), 
  let n := 33 
  in let twice_51 := 2 * 51 
  in diff = twice_51 - (n * 3) ∧ diff = 3 :=
begin
  -- n is 33
  let n := 33,
  -- twice_51 is 2 * 51
  let twice_51 := 2 * 51,
  -- Let's calculate the difference
  let diff := twice_51 - (n * 3),
  -- Now we assert that the difference is 3
  use diff,
  split,
  -- We assert that diff equals to twice_51 - (n * 3)
  -- Here diff = 102 - 99 = 3
  sorry,
end

end armand_guessing_game_difference_l824_824196


namespace find_circle_equation_and_chord_length_l824_824214

-- Define the conditions
def point_A : ℝ × ℝ := (2, -1)
def line_L : ℝ × ℝ := (1, 1, -1) -- x + y = 1
def line_M (a : ℝ) : ℝ := -2 * a -- y = -2x

-- Define the center of the circle
def center_C (a : ℝ) : ℝ × ℝ := (a, line_M a)

-- Define the radius formulas
def radius_formula1 (a : ℝ) : ℝ :=
  real.sqrt ((a - 2)^2 + (line_M a + 1)^2)

def radius_formula2 (a : ℝ) : ℝ :=
  real.abs (a - 2 * a - 1) / real.sqrt 2

-- Define the circle equation and verify it
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 2

-- Define the chord length on the y-axis
def chord_length_on_y_axis : ℝ := 2

-- The problem statements to be proven
theorem find_circle_equation_and_chord_length :
  (∃ a : ℝ, radius_formula1 a = radius_formula2 a ∧
    ∀ x y : ℝ, circle_equation x y) ∧
  chord_length_on_y_axis = 2 := 
sorry

end find_circle_equation_and_chord_length_l824_824214


namespace problem_statement_l824_824771

noncomputable def E (T : Finset ℕ) (p : ℕ) : Finset (Fin (p - 1) → ℕ) :=
  {tuple | ∀ i, tuple i ∈ T ∧ (Finset.sum (Finset.univ.image (λ i, (i + 1) * tuple i)) % p = 0)}

theorem problem_statement (p : ℕ) (hp : Nat.Prime p) (h3 : 3 < p) :
  let E0 := E ({0, 1, 2} : Finset ℕ) p in
  let E1 := E ({0, 1, 3} : Finset ℕ) p in
  (E1.card ≥ E0.card) ∧ (E1.card = E0.card ↔ p = 5) :=
by
  sorry

end problem_statement_l824_824771


namespace proof_problem_l824_824267

noncomputable def arithmetic_seq (n : ℕ) : ℕ := 4 * n - 3
noncomputable def sum_seq (n : ℕ) : ℕ := 2 * n^2 - n
noncomputable def b_seq (S_n : ℕ → ℕ) (n k : ℤ) : ℤ := S_n n / (n + k)
noncomputable def sum_T_n (b_n : ℕ → ℕ) (n : ℕ) : ℚ :=
  if b_n n = 2 * n then n / (4 * (n + 1))
  else if b_n n = 2 * n - 1 then n / (2 * n + 1)
  else 0

theorem proof_problem (d a_1 : ℤ) (k : ℤ)
  (h1 : d ≠ 0)
  (h2 : a_1 + (a_1 + 3 * d) = 14)
  (h3 : (a_1 + d)^2 = a_1 * (a_1 + 6 * d)) :
  ((∀ n : ℕ, arithmetic_seq n = 4 * n - 3) ∧
   (∀ n : ℕ, sum_seq n = 2 * n^2 - n) ∧
   ((k = -1/2 ∧ ∀ n : ℕ, sum_T_n (b_seq sum_seq k) n = n / (4 * (n + 1))) ∨
    (k = 0 ∧ ∀ n : ℕ, sum_T_n (b_seq sum_seq k) n = n / (2 * n + 1))))
  := by sorry

end proof_problem_l824_824267


namespace intersection_A_B_l824_824388

def set_A : set ℝ := {x | |x| < 4}
def set_B : set ℝ := Icc (-6 : ℝ) 1

theorem intersection_A_B :
  (set_A ∩ set_B) = {x : ℝ | -4 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_B_l824_824388


namespace pastries_sold_value_l824_824199

-- Define the number of cakes sold and the relationship between cakes and pastries
def number_of_cakes_sold := 78
def pastries_sold (C : Nat) := C + 76

-- State the theorem we want to prove
theorem pastries_sold_value : pastries_sold number_of_cakes_sold = 154 := by
  sorry

end pastries_sold_value_l824_824199


namespace odd_terms_in_expansion_l824_824934

theorem odd_terms_in_expansion (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  (∑ k in Finset.range 9, if binomial 8 k % 2 = 1 then 1 else 0) = 2 :=
sorry

end odd_terms_in_expansion_l824_824934


namespace least_integer_greater_than_sqrt_500_l824_824001

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824001


namespace digits_in_base8_of_1728_l824_824678

def base8_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else 1 + (Finset.range (n + 1)).filter (λ k => 8^k ≤ n).sup id

theorem digits_in_base8_of_1728 : base8_digits 1728 = 4 := 
by
  -- proof would go here
  sorry

end digits_in_base8_of_1728_l824_824678


namespace number_of_subsets_with_mean_6_l824_824680

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

theorem number_of_subsets_with_mean_6 (S : Set ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) :
    (∃ B ⊆ S, B.card = 2 ∧
    (let remaining := S \ B in remaining.sum / remaining.card = 6) ∧
    (B.sum = 12 → True)) → B.card = 5 := sorry

end number_of_subsets_with_mean_6_l824_824680


namespace jet_flight_time_with_tail_wind_l824_824129

theorem jet_flight_time_with_tail_wind 
  (d : ℕ) (t_return : ℕ) (w_speed : ℕ) (v : ℕ) (t : ℕ) 
  (h1 : d = 2000) 
  (h2 : t_return = 5) 
  (h3 : w_speed = 50) 
  (h4 : (v - w_speed) * t_return = d) 
  (h5 : (v + w_speed) * t = d) : 
  t = 4 := 
by 
  -- Definitions and conditions
  have v_450 : v = 450, from calc
    v - 50 = 400 : by rw [h1, h2, h3, mul_right_inj' (ne_of_gt (by norm_num : 5 > 0)), nat.div_eq_of_eq_mul_left (by norm_num : 5 > 0) (by norm_num : 5 * 400 = d)]
    ...      = 450 : by linarith,
  have t_4 : t = 4, from calc
    500 * t = d : by rw [h4, h5, mul_comm d 500]
    ...      = 2000 : by rw h1
    ...      = 4 * 500 : by norm_num,
  exact t_4

end jet_flight_time_with_tail_wind_l824_824129


namespace transformed_function_l824_824854

theorem transformed_function :
  ∀ (x : ℝ), (sin (2 * (x + π / 3))) = sin (x + π / 3) :=
by
  sorry

end transformed_function_l824_824854


namespace enter_exit_ways_correct_l824_824883

-- Defining the problem conditions
def num_entrances := 4

-- Defining the problem question and answer
def enter_exit_ways (n : Nat) : Nat := n * (n - 1)

-- Statement: Prove the number of different ways to enter and exit is 12
theorem enter_exit_ways_correct : enter_exit_ways num_entrances = 12 := by
  -- Proof
  sorry

end enter_exit_ways_correct_l824_824883


namespace math_problem_l824_824348

noncomputable def cartesian_equation : Prop :=
  ∀ (ρ θ : ℝ), 
    ρ^2 - 4 * real.sqrt 2 * ρ * real.cos (θ - π / 4) + 7 = 0 ↔ 
    let x := ρ * real.cos θ;
    let y := ρ * real.sin θ 
    in x^2 + y^2 - 4 * x - 4 * y + 7 = 0

noncomputable def x_plus_sqrt3y_range : set ℝ :=
  {m : ℝ | ∃ (α : ℝ), 
    let x := 2 + real.cos α;
    let y := 2 + real.sin α 
    in m = x + real.sqrt 3 * y }

noncomputable def range_x_plus_sqrt3y : Prop :=
  x_plus_sqrt3y_range = set.Icc (2 * real.sqrt 3) (4 + 2 * real.sqrt 3)

theorem math_problem :
  cartesian_equation ∧ range_x_plus_sqrt3y :=
by
  split
  · sorry
  · sorry

end math_problem_l824_824348


namespace least_integer_greater_than_sqrt_500_l824_824065

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824065


namespace solve_for_r_l824_824417

theorem solve_for_r (r : ℤ) : 24 - 5 = 3 * r + 7 → r = 4 :=
by
  intro h
  sorry

end solve_for_r_l824_824417


namespace exponential_function_inequality_l824_824995

theorem exponential_function_inequality {a : ℝ} (h0 : 0 < a) (h1 : a < 1) :
  (a^3) * (a^2) < a^2 :=
by
  sorry

end exponential_function_inequality_l824_824995


namespace unwanted_texts_per_week_l824_824356

-- Define the conditions as constants
def messages_per_day_old : ℕ := 20
def messages_per_day_new : ℕ := 55
def days_per_week : ℕ := 7

-- Define the theorem stating the problem
theorem unwanted_texts_per_week (messages_per_day_old messages_per_day_new days_per_week 
  : ℕ) : (messages_per_day_new - messages_per_day_old) * days_per_week = 245 :=
by
  sorry

end unwanted_texts_per_week_l824_824356


namespace range_of_a_l824_824690

theorem range_of_a (x a : ℝ) (h₁ : 0 < x) (h₂ : x < 2) (h₃ : a - 1 < x) (h₄ : x ≤ a) :
  1 ≤ a ∧ a < 2 :=
by
  sorry

end range_of_a_l824_824690


namespace find_width_of_tank_l824_824184

-- Given conditions
def length := 25
def depth := 6
def cost := 558
def rate_paise := 75
def rate := rate_paise / 100  -- in rupees

theorem find_width_of_tank (W : ℝ) (H1 : cost = 558) (H2 : rate = 0.75) :
  744 = (length * W) + 2 * (length * depth) + 2 * (W * depth) → W = 12 :=
sorry

end find_width_of_tank_l824_824184


namespace integral_evaluation_l824_824953

theorem integral_evaluation : 
  ∫ x in -1..1, (1 + x + real.sqrt (1 - x^2)) = 2 + real.pi / 2 :=
by
  sorry

end integral_evaluation_l824_824953


namespace poly_less_than_zero_l824_824805

-- Define the polynomial function
def poly (x : ℝ) : ℝ := -15 * x ^ 2 + 4 * x - 6

-- Statement: Prove that the polynomial is less than zero for all real x
theorem poly_less_than_zero : ∀ x : ℝ, poly x < 0 := by 
  sorry

end poly_less_than_zero_l824_824805


namespace find_line_l_l824_824625

noncomputable def circle (m : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1 - (3 - m))^2 + (p.2 - 2*m)^2 = 9}

def passes_through (l : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  l p.1 = p.2

theorem find_line_l (l : ℝ → ℝ) :
  (∀ m : ℝ, ∃ k : ℝ, passes_through l (1, 0) ∧ ∀ p1 p2 ∈ (circle m), distance p1 p2 = constant) →
  ∃ a b c : ℝ, a * l.1 + b * l.2 + c = 0 ∧ a = 2 ∧ b = 1 ∧ c = -2 :=
sorry

end find_line_l_l824_824625


namespace prob_roll_678_before_less_than_5_l824_824165

open ProbabilityTheory

theorem prob_roll_678_before_less_than_5 :
  (fair_probability (of_real (1 / 960))) = (probability_of_condition (roll_until (< 5) => sequence_increasing [6, 7, 8])) :=
sorry

end prob_roll_678_before_less_than_5_l824_824165


namespace tangent_lines_to_circle_M_through_A_l824_824983

-- Definitions based on conditions
def circle_M (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 16
def point_A : ℝ × ℝ := (4, -2)

-- Lean statement for the proof

theorem tangent_lines_to_circle_M_through_A :
  (∀ x y : ℝ, circle_M x y → (x = 4 ∨ 7 * x - 24 * y - 76 = 0)) ∧
  (let l : ℝ × ℝ → Prop := λ p, fst p + snd p - 2 = 0 in
   (∃ (d : ℝ), d = (| 0 + 1 - 2 | / real.sqrt (1^2 + 1^2)) ∧
   (∃ (chord_length : ℝ), chord_length = 2 * real.sqrt (16 - d^2) ∧ chord_length = real.sqrt 62))) := 
sorry

end tangent_lines_to_circle_M_through_A_l824_824983


namespace sara_ate_16_apples_l824_824188

theorem sara_ate_16_apples (S : ℕ) (h1 : ∃ (A : ℕ), A = 4 * S ∧ S + A = 80) : S = 16 :=
by
  obtain ⟨A, h2, h3⟩ := h1
  sorry

end sara_ate_16_apples_l824_824188


namespace range_of_x_l824_824999

def f (x : ℝ) : ℝ := x^3 + 3 * x

theorem range_of_x (m x : ℝ) (h_m : m ∈ set.Icc (-2:ℝ) 2) (h : f (m * x - 2) + f x < 0) :
  x ∈ set.Ioo (-2:ℝ) (2/3:ℝ) :=
sorry

end range_of_x_l824_824999


namespace binom_1300_2_eq_844350_l824_824556

theorem binom_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 :=
by
  sorry

end binom_1300_2_eq_844350_l824_824556


namespace cars_without_air_conditioning_l824_824329

theorem cars_without_air_conditioning :
  ∀ (totalCars racingStripes maxAirNoStripes : ℕ),
    totalCars = 100 →
    racingStripes ≥ 53 →
    maxAirNoStripes = 47 →
    (totalCars - (racingStripes - maxAirNoStripes + maxAirNoStripes) = 47) :=
begin
  intros totalCars racingStripes maxAirNoStripes totalCars_eq racingStripes_ge maxAirNoStripes_eq,
  rw [totalCars_eq, maxAirNoStripes_eq],
  calc
    100 - ((racingStripes - 47) + 47)
        = 100 - racingStripes                   : by ring
    ... = (100 - racingStripes) + 0              : by ring
    ... = 100 - racingStripes                    : by ring
    ... = 47                                   : by { sorry },
end

end cars_without_air_conditioning_l824_824329


namespace cost_for_four_dozen_l824_824542

-- Definitions based on conditions
def cost_of_two_dozen_apples : ℝ := 15.60
def cost_of_one_dozen_apples : ℝ := cost_of_two_dozen_apples / 2
def cost_of_four_dozen_apples : ℝ := 4 * cost_of_one_dozen_apples

-- Statement to prove
theorem cost_for_four_dozen :
  cost_of_four_dozen_apples = 31.20 :=
sorry

end cost_for_four_dozen_l824_824542


namespace linear_equation_value_l824_824703

-- Define the conditions of the equation
def equation_is_linear (m : ℝ) : Prop :=
  |m| = 1 ∧ m - 1 ≠ 0

-- Prove the equivalence statement
theorem linear_equation_value (m : ℝ) (h : equation_is_linear m) : m = -1 := 
sorry

end linear_equation_value_l824_824703


namespace product_of_roots_eq_neg10_l824_824566

theorem product_of_roots_eq_neg10 :
  let p1 := λ x : ℝ, 3*x^4 - 2*x^3 + 5*x - 15
  let p2 := λ x : ℝ, 4*x^3 + 6*x^2 - 8
  polynomial.roots (p1 * p2).map polynomial.C = -10 := 
sorry

end product_of_roots_eq_neg10_l824_824566


namespace least_integer_greater_than_sqrt_500_l824_824061

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824061


namespace probability_of_multiple_of_225_l824_824324
open Set

def s : Set ℕ := {3, 5, 15, 21, 25, 30, 75}
def isMultipleOf225 (a b : ℕ) : Prop := 225 ∣ (a * b)

theorem probability_of_multiple_of_225 : 
  let pairs := (finset.unorderedPairs s).to_finset.to_set in
  let valid_pairs := { ab ∈ pairs | isMultipleOf225 ab.1 ab.2 } in
  ((valid_pairs.to_finset.card : ℚ) / (pairs.to_finset.card : ℚ)) = 2/7 :=
sorry

end probability_of_multiple_of_225_l824_824324


namespace B_pow_101_l824_824358

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ]

theorem B_pow_101 :
  B ^ 101 = ![
    ![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]
  ] :=
  sorry

end B_pow_101_l824_824358


namespace least_integer_greater_than_sqrt_500_l824_824005

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824005


namespace geometric_sequence_x_l824_824276

theorem geometric_sequence_x (x : ℝ) (h : 1 * 9 = x^2) : x = 3 ∨ x = -3 :=
by
  sorry

end geometric_sequence_x_l824_824276


namespace solar_systems_per_planet_l824_824853

theorem solar_systems_per_planet (total_systems_and_planets : ℕ) (number_of_planets : ℕ) :
  total_systems_and_planets = 200 → number_of_planets = 20 → 
  total_systems_and_planets - number_of_planets = 180 → 180 / number_of_planets = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end solar_systems_per_planet_l824_824853


namespace least_integer_greater_than_sqrt_500_l824_824024

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824024


namespace concentration_of_salt_solution_l824_824889

-- Conditions:
def total_volume : ℝ := 1 + 0.25
def concentration_of_mixture : ℝ := 0.15
def volume_of_salt_solution : ℝ := 0.25

-- Expression for the concentration of the salt solution used, $C$:
theorem concentration_of_salt_solution (C : ℝ) :
  (volume_of_salt_solution * (C / 100)) = (total_volume * concentration_of_mixture) → C = 75 := by
  sorry

end concentration_of_salt_solution_l824_824889


namespace sin_cos_eq_sqrt2_div_8_l824_824487

theorem sin_cos_eq_sqrt2_div_8 (z : ℂ) (k : ℤ) : 
  (sin z)^3 * cos z - (sin z) * (cos z)^3 = (Real.sqrt 2) / 8 ↔ 
  ∃ k : ℤ, z = (-1)^(k + 1) * Real.pi / 16 + k * (Real.pi / 4) :=
by
  sorry

end sin_cos_eq_sqrt2_div_8_l824_824487


namespace checkerboard_black_squares_count_l824_824925

namespace Checkerboard

def is_black (n : ℕ) : Bool :=
  -- Define the alternating pattern of the checkerboard
  (n % 2 = 0)

def black_square_count (n : ℕ) : ℕ :=
  -- Calculate the number of black squares in a checkerboard of size n x n
  if n % 2 = 0 then n * n / 2 else n * n / 2 + n / 2 + 1

def additional_black_squares (n : ℕ) : ℕ :=
  -- Calculate the additional black squares due to modification of every 33rd square in every third row
  ((n - 1) / 3 + 1)

def total_black_squares (n : ℕ) : ℕ :=
  -- Calculate the total black squares considering the modified hypothesis
  black_square_count n + additional_black_squares n

theorem checkerboard_black_squares_count : total_black_squares 33 = 555 := 
  by sorry

end Checkerboard

end checkerboard_black_squares_count_l824_824925


namespace max_elements_in_M_l824_824280

-- Define the sets S1 and S2
def S1 := {0, 1, 2, 3, 4}
def S2 := {0, 2, 4, 8}

-- Define the set M with given conditions
def M : Set ℕ := {x | x ∈ S1 ∧ x ∈ S2}

-- Statement of the problem
theorem max_elements_in_M : ∃ M, M ⊆ S1 ∧ M ⊆ S2 ∧ Finite M ∧ M.card = 3 := sorry

end max_elements_in_M_l824_824280


namespace least_int_gt_sqrt_500_l824_824057

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824057


namespace least_integer_greater_than_sqrt_500_l824_824086

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824086


namespace problem_T_n_l824_824283

noncomputable def a_n (n : ℕ) : ℤ := 11 - 2 * n

noncomputable def S_n (n : ℕ) : ℤ := -n^2 + 10 * n

noncomputable def T_n (n : ℕ) : ℤ :=
  if n ≤ 5 then
    -n^2 + 10 * n
  else
    n^2 - 10 * n + 50

theorem problem_T_n (n : ℕ) : 
  let T_n := if n ≤ 5 then -n^2 + 10 * n else n^2 - 10 * n + 50 in
  (T_n = if n ≤ 5 then -n^2 + 10 * n else n^2 - 10 * n + 50) := 
by
  sorry

end problem_T_n_l824_824283


namespace focus_of_parabola_l824_824586

theorem focus_of_parabola (x y : ℝ) (h : y = -4 * x^2) : 
  ∃ fy, focus (0, fy) = (0, -1 / 16) := sorry

end focus_of_parabola_l824_824586


namespace product_of_possible_N_l824_824915

theorem product_of_possible_N (N : ℕ) (M L : ℕ) :
  (M = L + N) →
  (M - 5 = L + N - 5) →
  (L + 3 = L + 3) →
  |(L + N - 5) - (L + 3)| = 2 →
  (10 * 6 = 60) :=
by
  sorry

end product_of_possible_N_l824_824915


namespace pentagon_perimeter_value_l824_824611

open Real

-- Definitions based on conditions
def radius : ℝ := 5
def diameter : ℝ := 10
def A := (0 : ℝ, 5 : ℝ)
def B := (5 * (cos (π / 3)) : ℝ, 5 * (sin (π / 3)): ℝ)
def C := (5 : ℝ, 0 : ℝ)
def D := (0 : ℝ,  -5 : ℝ)
def E := (-5 * (cos (π / 3)): ℝ, -5 * (sin (π / 3)): ℝ)

-- Assuming the above points follow the conditions set in the problem.
def pentagon_perimeter (A B C D E : (ℝ × ℝ)) : ℝ :=
  dist A B + dist B C + dist C D + dist D E + dist E A

theorem pentagon_perimeter_value :
  let perimeter := pentagon_perimeter A B C D E in
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ perimeter = m + sqrt n ∧ m + n = 95 :=
by
  sorry

end pentagon_perimeter_value_l824_824611


namespace problem_solution_l824_824976

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ (m n : ℝ), f(m + n) + f(m - n) = 2 * f(m) * f(n)
axiom condition2 : ∀ (m : ℝ), f(1 + m) = f(1 - m)
axiom condition3 : ∃ (x : ℝ), f(x) ≠ 0 ∧ (0 < x ∧ x < 1 → f(x) < 1)

theorem problem_solution :
  f(0) = 1 ∧ f(1) = -1 ∧ 
  (∀ x : ℝ, f(-x) = f(x)) ∧ 
  (∀ x : ℝ, f(x + 2) = f(x) ∧ (∑ k in finset.range 2017, f((k + 1) / 3)) = 1 / 2) :=
sorry

end problem_solution_l824_824976


namespace percentage_gain_correct_l824_824870

-- Define the number of bowls purchased and their per-bowl cost
def bowls_purchased := 115
def cost_per_bowl := 18

-- Define the number of bowls sold and their selling price per bowl
def bowls_sold := 104
def selling_price_per_bowl := 20

-- Calculate the total cost
def total_cost := bowls_purchased * cost_per_bowl  -- Rs. 2070

-- Calculate the total revenue
def total_revenue := bowls_sold * selling_price_per_bowl  -- Rs. 2080

-- Calculate the gain
def gain := total_revenue - total_cost  -- Rs. 10

-- Calculate the percentage gain
def percentage_gain := (gain.toFloat / total_cost.toFloat) * 100  -- 0.483%

-- Prove that the percentage gain is approximately 0.483%
theorem percentage_gain_correct : percentage_gain ≈ 0.483 :=
by
  sorry

end percentage_gain_correct_l824_824870


namespace integral_value_l824_824138

theorem integral_value : 
  ∫ x in 0..(2 * Real.pi / 3), (1 + Real.sin x) / (1 + Real.cos x + Real.sin x) = (Real.pi / 3) + Real.log 2 := 
by
  sorry

end integral_value_l824_824138


namespace least_integer_greater_than_sqrt_500_l824_824113

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824113


namespace list_scores_lowest_highest_l824_824409

variable (M Q S T : ℕ)

axiom Quay_thinks : Q = T
axiom Marty_thinks : M > T
axiom Shana_thinks : S < T
axiom Tana_thinks : T ≠ max M (max Q (max S T)) ∧ T ≠ min M (min Q (min S T))

theorem list_scores_lowest_highest : (S < T) ∧ (T = Q) ∧ (Q < M) ↔ (S < T) ∧ (T < M) :=
by
  sorry

end list_scores_lowest_highest_l824_824409


namespace find_constants_l824_824958

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x + c

theorem find_constants :
  ∃ a b c,
    (a = -2) ∧
    (b = 3 / (1 - π / 4)) ∧
    (c = (2 / π) * (-7 - 3 / (1 - π / 4))) ∧
    (∀ x, 5 * Real.sin x + 3 * Real.cos x + 1 + 
      ∫ t in 0..(π / 2), (Real.sin x + Real.cos t) * f a b c t = f a b c x) :=
begin
  sorry
end

end find_constants_l824_824958


namespace advance_copies_l824_824806

theorem advance_copies (total_copies : ℕ) (earnings_per_copy : ℕ) (agent_percentage : ℚ)
  (kept_amount : ℚ) (advance_copies_result : ℕ) :
  total_copies = 1000000 →
  earnings_per_copy = 2 →
  agent_percentage = 0.10 →
  kept_amount = 1620000 →
  advance_copies_result = total_copies - (kept_amount / ((1 - agent_percentage) * earnings_per_copy)).toNat →
  advance_copies_result = 100000 := by
  intros h1 h2 h3 h4 h5
  sorry

end advance_copies_l824_824806


namespace runners_adjacent_vertices_after_2013_l824_824458

def hexagon_run_probability (t : ℕ) : ℚ :=
  (2 / 3) + (1 / 3) * ((1 / 4) ^ t)

theorem runners_adjacent_vertices_after_2013 :
  hexagon_run_probability 2013 = (2 / 3) + (1 / 3) * ((1 / 4) ^ 2013) := 
by 
  sorry

end runners_adjacent_vertices_after_2013_l824_824458


namespace least_integer_greater_than_sqrt_500_l824_824039

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824039


namespace least_int_gt_sqrt_500_l824_824050

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824050


namespace seeds_in_fourth_pot_l824_824576

theorem seeds_in_fourth_pot (total_seeds : ℕ) (total_pots : ℕ) (seeds_per_pot : ℕ) (first_three_pots : ℕ)
  (h1 : total_seeds = 10) (h2 : total_pots = 4) (h3 : seeds_per_pot = 3) (h4 : first_three_pots = 3) : 
  (total_seeds - (seeds_per_pot * first_three_pots)) = 1 :=
by
  sorry

end seeds_in_fourth_pot_l824_824576


namespace least_integer_greater_than_sqrt_500_l824_824088

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824088


namespace least_integer_greater_than_sqrt_500_l824_824072

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824072


namespace arith_seq_general_formula_sum_bn_formula_l824_824730

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a n = 2 + n * d

def geometric_condition (a : ℕ → ℤ) : Prop :=
(a 2)^2 = a 1 * a 4

theorem arith_seq_general_formula:
  ∃ a : ℕ → ℤ, (arithmetic_seq a 2) ∧ (geometric_condition a) → (∀ n : ℕ, a n = 2 * n) :=
begin
  sorry
end

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(-1) ^ (n + 1) * (2 / a n + 2 / a (n + 1))

def sum_bn (a : ℕ → ℤ) (n : ℕ) : ℤ :=
∑ k in range (2 * n - 1), b_n a k

theorem sum_bn_formula:
  ∃ a : ℕ → ℤ, (arithmetic_seq a 2) ∧ (geometric_condition a) → 
  (∀ n : ℕ, sum_bn a n = 1 + 1 / (2 * n)) :=
begin
  sorry
end

end arith_seq_general_formula_sum_bn_formula_l824_824730


namespace simplify_expr_l824_824413

-- Define the given expression using square roots and exponents
def expr1 := Real.sqrt (3 * 5)
def expr2 := Real.sqrt (5^3 * 3^3)

-- Combine both expressions
def combined_expr := expr1 * expr2

-- Define the target value we want to prove the combined expression equals to
def target_value := 225

-- State the theorem
theorem simplify_expr : combined_expr = target_value := by
  sorry

end simplify_expr_l824_824413


namespace andre_total_payment_l824_824785

theorem andre_total_payment :
  let original_price_treadmill := 1350
  let discount_rate := 0.30
  let price_per_plate := 50
  let number_of_plates := 2
  let discount := original_price_treadmill * discount_rate
  let discounted_price_treadmill := original_price_treadmill - discount
  let total_plate_cost := price_per_plate * number_of_plates
  let total_payment := discounted_price_treadmill + total_plate_cost
  in total_payment = 1045 :=
by
  sorry

end andre_total_payment_l824_824785


namespace number_of_terminating_decimals_l824_824614

theorem number_of_terminating_decimals :
  ∃ (count : ℕ), count = 64 ∧ ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 449 → (∃ k : ℕ, n = 7 * k) → (∃ k : ℕ, (∃ m : ℕ, 560 = 2^m * 5^k * n)) :=
sorry

end number_of_terminating_decimals_l824_824614


namespace BE_CF_Midpoint_l824_824326

variable {A B C D E F M : Type}
variable [Metric_Space A] [Metric_Space B] [Metric_Space C] [Metric_Space D] [Metric_Space E] [Metric_Space F] [Metric_Space M]
variable {AB AC BC AD : ℝ}
variable {BE CF : ℝ}

-- Definitions based on given problem
def isMidpoint (M : Type) (B C : Type) [Metric_Space B] [Metric_Space C] :=
  dist M B = dist M C

def isAngleBisector (AD : ℝ) (ABC : Type) [Metric_Space ABC] : Prop :=
  ∀ a b c, a / b = c

def isParallel (ME AD : ℝ) (ABC : Type) [Metric_Space ABC] : Prop :=
  ∀ angle, angle M E = angle D A

-- Restate the problem to prove
theorem BE_CF_Midpoint (ABC AD ME BE CF : ℝ) (hyp : isAngleBisector AD ABC) (mid_M : isMidpoint M B C) (par_ME_AD : isParallel ME AD ABC)
  : BE = CF ∧ BE = (1/2) * (AB + AC) ∨ BE = CF ∧ BE = (1/2) * (|AB - AC|) :=
  sorry

end BE_CF_Midpoint_l824_824326


namespace odd_number_of_different_color_triangles_l824_824636

theorem odd_number_of_different_color_triangles (A B C : Point) 
    (m : ℕ) (points_interior : Fin m → Point) 
    (colors : Point → Color) 
    (triangle_divided : Fin (m + 3) → Triangle) :
    (colors A = red) ∧ (colors B = yellow) ∧ (colors C = blue) →
    (∀ i, colors (points_interior i) = red ∨ colors (points_interior i) = yellow ∨ colors (points_interior i) = blue) →
    (number_of_different_color_triangles(triangle_divided, colors) % 2 = 1) :=
begin
  sorry
end

end odd_number_of_different_color_triangles_l824_824636


namespace least_integer_greater_than_sqrt_500_l824_824008

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824008


namespace square_divisible_by_six_in_range_l824_824236

theorem square_divisible_by_six_in_range (x : ℕ) (h1 : ∃ n : ℕ, x = n^2)
  (h2 : 6 ∣ x) (h3 : 30 < x) (h4 : x < 150) : x = 36 ∨ x = 144 :=
by {
  sorry
}

end square_divisible_by_six_in_range_l824_824236


namespace explicit_formula_and_parity_solve_inequality_l824_824667

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  log m ((1 + x) / (1 - x))

theorem explicit_formula_and_parity (m : ℝ) :
  (∀ x, f (x^2 - 1) m = log m (x^2 / (2 - x^2))) →
  (∀ x, f x m = log m ((1 + x) / (1 - x))) ∧ 
  (∀ x, f (-x) m = -f x m) := by
  sorry

theorem solve_inequality (m : ℝ) (x : ℝ) :
  (∀ x, f (x^2 - 1) m = log m (x^2 / (2 - x^2))) →
  (m > 1 → (-1 < x ∧ x ≤ 0) ↔ f x m ≤ 0) ∧
  (0 < m ∧ m < 1 → (0 ≤ x ∧ x < 1) ↔ f x m ≤ 0) := by
  sorry

end explicit_formula_and_parity_solve_inequality_l824_824667


namespace find_a_b_range_f_0_pi_by_2_l824_824289

-- Given the function f(x)
def f (x a b : ℝ) : ℝ :=
  2 * a * (cos (x / 2))^2 + 2 * sqrt 3 * a * (sin (x / 2)) * (cos (x / 2)) - a + b

-- Conditions
axiom f_pi_by_3 : f (π / 3) a b = 3
axiom f_5pi_by_6 : f (5 * π / 6) a b = 1

-- Assertions to be proved
theorem find_a_b : a = 1 ∧ b = 1 :=
sorry

theorem range_f_0_pi_by_2 : set.range (λ x, f x 1 1) = set.Icc 2 3 :=
sorry

end find_a_b_range_f_0_pi_by_2_l824_824289


namespace chord_midpoint_ellipse_l824_824651

-- Define the ellipse equation and the midpoint
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 8) + (y^2 / 4) = 1

def is_midpoint (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P in
  let (qx, qy) := Q in
  let (rx, ry) := R in
  px = (qx + rx) / 2 ∧ py = (qy + ry) / 2

theorem chord_midpoint_ellipse (Q R : ℝ × ℝ) :
  is_midpoint (2, -1) Q R →
  ellipse (fst Q) (snd Q) →
  ellipse (fst R) (snd R) →
  ∃ (m b : ℝ), b = -3 ∧ m = 1 ∧ (∀ x y, y = m * x + b → (y - snd (2, -1))=((x - fst (2, -1)) * m)) :=
sorry

end chord_midpoint_ellipse_l824_824651


namespace eric_containers_l824_824573

theorem eric_containers (initial_pencils : ℕ) (additional_pencils : ℕ) (pencils_per_container : ℕ) 
  (h1 : initial_pencils = 150) (h2 : additional_pencils = 30) (h3 : pencils_per_container = 36) :
  (initial_pencils + additional_pencils) / pencils_per_container = 5 := 
by {
  sorry
}

end eric_containers_l824_824573


namespace math_problem_l824_824580

theorem math_problem :
  2537 + 240 * 3 / 60 - 347 = 2202 :=
by
  sorry

end math_problem_l824_824580


namespace sum_of_mean_median_mode_l824_824475

def numbers : List ℤ := [-3, -1, 0, 2, 2, 3, 3, 3, 4, 5]

noncomputable def mean (l : List ℤ) : ℚ :=
  (l.sum : ℚ) / l.length

noncomputable def median (l : List ℤ) : ℚ :=
  let sorted := l.qsort (· ≤ ·)
  if l.length % 2 = 0 then
    (sorted.get! (l.length / 2 - 1) + sorted.get! (l.length / 2) : ℚ) / 2
  else
    sorted.get! (l.length / 2)

def mode (l : List ℤ) : ℤ :=
  let freqs := l.foldl (fun acc x => acc.insertWith (· + ·) x 1) (Std.RBMap.ofList [])
  freqs.toList.maxBy (·.2).1

noncomputable def sum_mean_median_mode (l : List ℤ) : ℚ :=
  mean l + median l + (mode l : ℚ)

theorem sum_of_mean_median_mode : sum_mean_median_mode numbers = 7.3 := by
  sorry

end sum_of_mean_median_mode_l824_824475


namespace point_in_second_quadrant_l824_824433

def z : ℂ := complex.i - 1

theorem point_in_second_quadrant (z : ℂ) (hz : z = complex.i - 1) : 
  z.re < 0 ∧ z.im > 0 :=
by {
  rw [hz],
  -- z = i - 1
  sorry
}

end point_in_second_quadrant_l824_824433


namespace intersection_when_a_eq_1_range_for_A_inter_B_eq_A_l824_824375

noncomputable def f (x : ℝ) : ℝ := Real.log (3 - abs (x - 1))

def setA : Set ℝ := { x | 3 - abs (x - 1) > 0 }

def setB (a : ℝ) : Set ℝ := { x | x^2 - (a + 5) * x + 5 * a < 0 }

theorem intersection_when_a_eq_1 : (setA ∩ setB 1) = { x | 1 < x ∧ x < 4 } :=
by
  sorry

theorem range_for_A_inter_B_eq_A : { a | (setA ∩ setB a) = setA } = { a | a ≤ -2 } :=
by
  sorry

end intersection_when_a_eq_1_range_for_A_inter_B_eq_A_l824_824375


namespace least_integer_greater_than_sqrt_500_l824_824025

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824025


namespace angle_RPS_l824_824743

theorem angle_RPS (P Q R S : Type) 
  (collinear_QRS : PointsCollinear Q R S)
  (angle_PQS : ∠P Q S = 55)
  (angle_PSQ : ∠P S Q = 40)
  (angle_QPR : ∠Q P R = 72) : 
  ∠R P S = 13 := 
by
  sorry

end angle_RPS_l824_824743


namespace number_of_partitions_of_7_into_4_parts_l824_824309

theorem number_of_partitions_of_7_into_4_parts : 
  (finset.attach (finset.powerset (finset.range (7+1)))).filter (λ s, s.sum = 7 ∧ s.card ≤ 4)).card = 11 := 
by sorry

end number_of_partitions_of_7_into_4_parts_l824_824309


namespace volume_of_prism_l824_824848

theorem volume_of_prism :
  ∃ (a b c : ℝ), ab * bc * ac = 762 ∧ (ab = 56) ∧ (bc = 63) ∧ (ac = 72) ∧ (b = 2 * a) :=
sorry

end volume_of_prism_l824_824848


namespace evaluate_expression_l824_824231

-- Given variables x and y are non-zero
variables (x y : ℝ)

-- Condition
axiom xy_nonzero : x * y ≠ 0

-- Statement of the proof
theorem evaluate_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x * (y^3 + 2) / y + (x^3 - 2) / y * (y^3 - 2) / x) = 2 * x * y * (x^2 * y^2) + 8 / (x * y) := 
by {
  sorry
}

end evaluate_expression_l824_824231


namespace trig_identity_proof_l824_824254

-- Define the variables
variable (x : ℝ)

-- Given condition
def given_condition : Prop := sin (2 * x + π / 5) = sqrt 3 / 3

-- The statement to be proved
theorem trig_identity_proof (h : given_condition x) :
  sin (4 * π / 5 - 2 * x) + sin (3 * π / 10 - 2 * x) ^ 2 = (2 + sqrt 3) / 3 :=
sorry

end trig_identity_proof_l824_824254


namespace integer_sets_count_l824_824444

theorem integer_sets_count : 
  (∃ x y z : ℤ, (y+z)^1949 + (z+x)^1999 + (x+y)^2000 = 2) ↔ 3 :=
begin
  sorry
end

end integer_sets_count_l824_824444


namespace factor_x4_minus_64_l824_824954

theorem factor_x4_minus_64 :
  ∀ (x : ℝ), (x^4 - 64) = (x^2 - 8) * (x^2 + 8) :=
by
  intro x
  sorry

end factor_x4_minus_64_l824_824954


namespace least_integer_greater_than_sqrt_500_l824_824068

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824068


namespace sqrt_500_least_integer_l824_824021

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824021


namespace joe_eats_different_fruits_l824_824749

noncomputable def joe_probability : ℚ :=
  let single_fruit_prob := (1 / 3) ^ 4
  let all_same_fruit_prob := 3 * single_fruit_prob
  let at_least_two_diff_fruits_prob := 1 - all_same_fruit_prob
  at_least_two_diff_fruits_prob

theorem joe_eats_different_fruits :
  joe_probability = 26 / 27 :=
by
  -- The proof is omitted for this task
  sorry

end joe_eats_different_fruits_l824_824749


namespace polar_equation_of_curve_C_range_of_AB_l824_824734

-- Definition of the curve C
def curve_C (alpha : ℝ) : ℝ × ℝ :=
  (2 * Real.cos alpha, Real.sin alpha)

-- Polar equation proof statement
theorem polar_equation_of_curve_C :
  ∀ (θ : ℝ), (let ρ := 2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2) in (ρ^2 = 4 / (1 + 3 * Real.sin θ ^ 2))) :=
sorry

-- Distance AB and range proof statement
theorem range_of_AB :
  ∀ (θ : ℝ),
    (let ρ₁ := 2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2),
         ρ₂ := 2 / Real.sqrt (1 + 3 * Real.sin (θ + Real.pi / 2) ^ 2),
         dist_AB := Real.sqrt (ρ₁^2 + ρ₂^2)
     in (ρ₁ = 2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2)) → 
        (ρ₂ = 2 / Real.sqrt (1 + 3 * Real.sin (θ + Real.pi / 2) ^ 2)) → 
        (Real.sqrt (16 / (1 + 5 * Real.sin (2 * θ) ^ 2))) = dist_AB) :=
sorry

end polar_equation_of_curve_C_range_of_AB_l824_824734


namespace sqrt_500_least_integer_l824_824017

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824017


namespace probability_of_yellow_l824_824885

-- Definitions of the given conditions
def red_jelly_beans := 4
def green_jelly_beans := 8
def yellow_jelly_beans := 9
def blue_jelly_beans := 5
def total_jelly_beans := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans

-- Theorem statement
theorem probability_of_yellow :
  (yellow_jelly_beans : ℚ) / total_jelly_beans = 9 / 26 :=
by
  sorry

end probability_of_yellow_l824_824885


namespace area_of_triangle_BGD_l824_824403

-- Given conditions as definitions

-- Point G is the centroid of triangle ABC
def is_centroid {A B C G : Point} (G : Point) : Prop :=
  barycentric.G = barycentric.A / 3 + barycentric.B / 3 + barycentric.C / 3

-- Line AG is extended to intersect side BC at point D
def points_collinear {A G D : Point} : Prop :=
  collinear A G D

-- The area of triangle ABC is 6 cm^2
def triangle_area_ABC (ABC : Triangle) : real :=
  6

-- To prove: the area of triangle BGD is 1 cm^2
theorem area_of_triangle_BGD {A B C G D : Point}
  (h1 : is_centroid G)
  (h2 : points_collinear A G D)
  (area_ABC : triangle_area_ABC ⟨A, B, C⟩ = 6) :
  ∃ area_BGD : real, area_BGD = 1 :=
by
  sorry

end area_of_triangle_BGD_l824_824403


namespace perimeter_of_region_l824_824429

-- Define the condition
def area_of_region := 512 -- square centimeters
def number_of_squares := 8

-- Define the presumed perimeter
def presumed_perimeter := 144 -- the correct answer

-- Mathematical statement that needs proof
theorem perimeter_of_region (area_of_region: ℕ) (number_of_squares: ℕ) (presumed_perimeter: ℕ) : 
   area_of_region = 512 ∧ number_of_squares = 8 → presumed_perimeter = 144 :=
by 
  sorry

end perimeter_of_region_l824_824429


namespace polygon_with_given_angle_sums_is_hexagon_l824_824710

theorem polygon_with_given_angle_sums_is_hexagon
  (n : ℕ)
  (h_interior : (n - 2) * 180 = 2 * 360) :
  n = 6 :=
by
  sorry

end polygon_with_given_angle_sums_is_hexagon_l824_824710


namespace possible_values_of_a_and_b_l824_824782

theorem possible_values_of_a_and_b (a b : ℕ) : 
  (a = 22 ∨ a = 33 ∨ a = 40 ∨ a = 42) ∧ 
  (b = 21 ∨ b = 10 ∨ b = 3 ∨ b = 1) ∧ 
  (a % (b + 1) = 0) ∧ (43 % (a + b) = 0) :=
sorry

end possible_values_of_a_and_b_l824_824782


namespace a_2023_l824_824350

noncomputable def a : ℕ → ℕ
| 1 := 1
| 2 := 4
| 3 := 9
| 4 := 16
| 5 := 25
| n := if h : n > 5 then a (n - 4) + a (n - 5) - a (n - 3) else 0

lemma a_rec (n : ℕ) (h : n > 0) :
  a (n + 5) + a (n + 1) = a (n + 4) + a n :=
by {
  induction n using nat.strong_induction_on with n ih,
  cases n,
  { linarith, },
  cases n,
  { linarith, },
  cases n,
  { linarith, },
  cases n,
  { linarith, },
  { simp only [a, nat.succ_pos'], split_ifs,
    { 
      have h1 := ih (n+1-_ : n+1 > 5) (lt_of_succ_lt_succ h),
      rw ←sub_add_eq_add_sub,
      simp only [add_assoc, nat.add_sub_cancel],
      linarith,
    },
    { linarith, },}
}

lemma a_periodic (n : ℕ) :
  ∀ k, a (k * 8 + n) = a n :=
by {
  intro k,
  induction k with k ih,
  { rw zero_mul, },
  { have h1 := ih,
    replace h1 : a (k * 8 + 8 + n) = a n := by linarith,
    rw mul_succ at *,
    simp only [add_assoc] at h1,
    exact h1, }
}

theorem a_2023 : a 2023 = 17 :=
by {
  replace a_periodic := a_periodic 7 252,
  simp only [nat.mod_eq_of_lt (show 2023 % 8 = 7, from rfl)] at a_periodic,
  rw a_periodic,
  exact rfl,
  sorry
}

#eval a 2023  -- should output 17 (the final evaluated result)

end a_2023_l824_824350


namespace sum_le_sqrt_n_l824_824799

theorem sum_le_sqrt_n (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 ≤ a i) (hsq : (∑ i, (a i)^2) = 1) :
  (∑ i, a i) ≤ Real.sqrt n := by
  sorry

end sum_le_sqrt_n_l824_824799


namespace AlissaMorePresents_l824_824575

/-- Ethan has 31 presents -/
def EthanPresents : ℕ := 31

/-- Alissa has 53 presents -/
def AlissaPresents : ℕ := 53

/-- How many more presents does Alissa have than Ethan? -/
theorem AlissaMorePresents : AlissaPresents - EthanPresents = 22 := by
  -- Place the proof here
  sorry

end AlissaMorePresents_l824_824575


namespace prob_of_different_colors_l824_824422

def total_balls_A : ℕ := 4 + 5 + 6
def total_balls_B : ℕ := 7 + 6 + 2

noncomputable def prob_same_color : ℚ :=
  (4 / ↑total_balls_A * 7 / ↑total_balls_B) +
  (5 / ↑total_balls_A * 6 / ↑total_balls_B) +
  (6 / ↑total_balls_A * 2 / ↑total_balls_B)

noncomputable def prob_different_color : ℚ :=
  1 - prob_same_color

theorem prob_of_different_colors :
  prob_different_color = 31 / 45 :=
by
  sorry

end prob_of_different_colors_l824_824422


namespace japanese_turtle_crane_problem_l824_824427

theorem japanese_turtle_crane_problem (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x + y = 35 ∧ 2 * x + 4 * y = 94 :=
by
  sorry

end japanese_turtle_crane_problem_l824_824427


namespace f_is_odd_evaluate_ff2017_l824_824563

-- Define the function f and the given conditions
variable {f : ℝ → ℝ} (h1 : ∀ x y : ℝ, f(x + y + 1) = f(x - y + 1) - f(x) * f(y))
variable (h2 : f(1) = 2)

-- First goal: f is odd
theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) := by
  sorry

-- Second goal: f(f(2017)) = 0
theorem evaluate_ff2017 : f(f(2017)) = 0 := by
  sorry

end f_is_odd_evaluate_ff2017_l824_824563


namespace intersecting_segments_common_point_l824_824839

-- Definitions of the points on pentagon and midpoints of sides
variables (A B C D E M N P Q R O : Point)

-- Definitions of segments
variables (AP BQ CR DM EN : Line)

-- Conditions
axiom vertices_on_sides : (A, B, C, D, E) ∈ Pentagon
axiom mid_AB : midpoint A B M
axiom mid_BC : midpoint B C N
axiom mid_CD : midpoint C D P
axiom mid_DE : midpoint D E Q
axiom mid_EA : midpoint E A R

axiom intersect_at_O : is_collinear [AP, BQ, CR, DM]

-- Proof statement
theorem intersecting_segments_common_point (A B C D E M N P Q R O : Point) 
  (AP BQ CR DM EN : Line) 
  (vertices_on_sides : (A, B, C, D, E) ∈ Pentagon)
  (mid_AB : midpoint A B M) 
  (mid_BC : midpoint B C N) 
  (mid_CD : midpoint C D P) 
  (mid_DE : midpoint D E Q) 
  (mid_EA : midpoint E A R)
  (intersect_at_O : is_collinear [AP, BQ, CR, DM]) 
  : intersects O EN := 
sorry

end intersecting_segments_common_point_l824_824839


namespace probability_no_shaded_square_l824_824148

theorem probability_no_shaded_square :
  let n := (2006 * 2005) / 2,
      m := 1003 * 1003 in
  (n - m) / n = 1002 / 2005 :=
by 
  sorry

end probability_no_shaded_square_l824_824148


namespace determine_positive_x_l824_824229

-- Defining the problem conditions
def satisfies_logarithmic_condition (x : ℝ) : Prop :=
  (log x / log 4) * (log 9 / log x) = log 9 / log 4

def satisfies_quadratic_condition (x : ℝ) : Prop :=
  x^2 = 16

-- Combining the conditions to prove the final statement
theorem determine_positive_x (x : ℝ) (hx : x > 0) :
  satisfies_logarithmic_condition x ∧ satisfies_quadratic_condition x → x = 4 :=
by
  sorry

end determine_positive_x_l824_824229


namespace sara_ate_16_apples_l824_824187

theorem sara_ate_16_apples (S : ℕ) (h1 : ∃ (A : ℕ), A = 4 * S ∧ S + A = 80) : S = 16 :=
by
  obtain ⟨A, h2, h3⟩ := h1
  sorry

end sara_ate_16_apples_l824_824187


namespace matrix_a4_zero_a3_a2_inf_possible_l824_824368

noncomputable def num_possible_matrices_a2_and_a3 (A : Matrix (Fin 3) (Fin 3) ℝ) (h : A^4 = 0) : ℕ × ℕ :=
  (1, ℕ.Infinity)

theorem matrix_a4_zero_a3_a2_inf_possible (A : Matrix (Fin 3) (Fin 3) ℝ) (h : A^4 = 0) :
    num_possible_matrices_a2_and_a3 A h = (1, ℕ.Infinity) :=
  sorry

end matrix_a4_zero_a3_a2_inf_possible_l824_824368


namespace sqrt2_decimal_zeros_l824_824485

theorem sqrt2_decimal_zeros (k : ℕ) : 
  (∃ n : ℕ, n ≤ k ∧ ∀ i : ℕ, (1 ≤ i ∧ i ≤ k → (sqrt 2).digits (n + i) = 0)) → false := 
sorry

end sqrt2_decimal_zeros_l824_824485


namespace probability_all_females_l824_824727

theorem probability_all_females (total_people : ℕ) (females : ℕ) (males : ℕ) (selected : ℕ) 
  (total_people_eq : total_people = 8) (females_eq : females = 5) (males_eq : males = 3)
  (selected_eq : selected = 3) :
  (nat.choose females selected : ℚ) / (nat.choose total_people selected) = 5 / 28 :=
by
  sorry

end probability_all_females_l824_824727


namespace simplify_expression_l824_824415

theorem simplify_expression (θ : ℝ) : 
  ((1 + Real.sin θ ^ 2) ^ 2 - Real.cos θ ^ 4) * ((1 + Real.cos θ ^ 2) ^ 2 - Real.sin θ ^ 4) = 4 * Real.sin (2 * θ) ^ 2 :=
by 
  sorry

end simplify_expression_l824_824415


namespace cube_necklace_packing_condition_l824_824499

theorem cube_necklace_packing_condition (N : ℕ) :
  (∃ cubes_threaded : ℕ, cubes_threaded = N ^ 3 ∧  
    (drilled_diagonally cubes_threaded) ∧ 
    (tied_into_loop cubes_threaded) ∧ 
    (packed_into_cubic_box cubes_threaded N)) ↔ even N :=
sorry

end cube_necklace_packing_condition_l824_824499


namespace non_intersecting_segments_l824_824624

theorem non_intersecting_segments {n : ℕ} (h : n ≥ 1) 
  (points : fin 2n → ℝ × ℝ)
  (blue_red_partition : ∃ blue red : fin n → fin 2n, 
                         function.bijective blue ∧ function.bijective red) :
  ∃ segments : fin n → (ℝ × ℝ) × (ℝ × ℝ), 
    (∀ i j : fin n, i ≠ j → ¬∃ p : ℝ × ℝ, 
      (p ∈ line segments i.1 ∧ p ∈ line segments j.1)) :=
    sorry

end non_intersecting_segments_l824_824624


namespace new_coordinates_l824_824735

-- Define the original point P
def P : (Int × Int) := (-5, -2)

-- Define the movement
def move_left (p : Int × Int) (units : Int) : Int × Int :=
  (p.1 - units, p.2)

def move_up (p : Int × Int) (units : Int) : Int × Int :=
  (p.1, p.2 + units)

-- Define the combined movement to the left and up
def move_left_and_up (p : Int × Int) (left_units up_units : Int) : Int × Int :=
  move_up (move_left p left_units) up_units

-- The theorem to be proved
theorem new_coordinates : move_left_and_up P 3 2 = (-8, 0) := 
by
  simp [P, move_left, move_up, move_left_and_up]
  norm_num
  sorry

end new_coordinates_l824_824735


namespace hyperbola_equation_proof_l824_824512

noncomputable def hyperbola_eq : Prop :=
  let e := Real.sqrt 5 / 2 in
  let a₂ := 9 in
  let b₂ := 4 in
  let c_ellipse := Real.sqrt (a₂ - b₂) in
  let c_hyperbola := c_ellipse in
  let a_hyperbola := 2 in
  let b_hyperbola := Real.sqrt (c_hyperbola^2 - a_hyperbola^2) in
  (e = c_hyperbola / a_hyperbola) ∧ (c_hyperbola^2 = a_hyperbola^2 + b_hyperbola^2) ∧
  (a_hyperbola = 2) ∧ (b_hyperbola = 1) ∧ (c_hyperbola = Real.sqrt 5) ∧
  (e = Real.sqrt 5 / 2) →
  (∀ x y, (x^2 / a_hyperbola^2 - y^2 / b_hyperbola^2 = 1 ↔ x^2 / 4 - y^2 = 1))

theorem hyperbola_equation_proof : hyperbola_eq :=
sorry

end hyperbola_equation_proof_l824_824512


namespace trigonometric_identity_l824_824486

theorem trigonometric_identity
  (α β : ℝ) :
  1 - sin α ^ 2 - sin β ^ 2 + 2 * sin α * sin β * cos (α - β) = cos (α - β) ^ 2 :=
by {
  sorry
}

end trigonometric_identity_l824_824486


namespace order_theorems_l824_824789

theorem order_theorems : 
  ∃ a b c d e f g : String,
    (a = "H") ∧ (b = "M") ∧ (c = "P") ∧ (d = "C") ∧ 
    (e = "V") ∧ (f = "S") ∧ (g = "E") ∧
    (a = "Heron's Theorem") ∧
    (b = "Menelaus' Theorem") ∧
    (c = "Pascal's Theorem") ∧
    (d = "Ceva's Theorem") ∧
    (e = "Varignon's Theorem") ∧
    (f = "Stewart's Theorem") ∧
    (g = "Euler's Theorem") := 
  sorry

end order_theorems_l824_824789


namespace num_students_in_second_class_l824_824430

theorem num_students_in_second_class 
  (avg_marks_class1 : ℕ → ℝ) (avg_marks_class2 : ℕ → ℝ) 
  (total_students_avg_marks : ℝ) 
  (num_students_class1 : ℕ) : Prop :=
  ∀ (num_students_class2 : ℕ), 
    (avg_marks_class1 num_students_class1 = 40) →
    (avg_marks_class2 num_students_class2 = 60) →
    (total_students_avg_marks = 50.90909090909091) →
    (num_students_class1 = 25) →
    (num_students_class2 = 30) →
    (let combined_total_marks := num_students_class1 * 40 + num_students_class2 * 60 in
     let combined_total_students := num_students_class1 + num_students_class2 in
     combined_total_marks = combined_total_students * total_students_avg_marks)

/- A non-interactive statement showing there are indeed 30 students in the second class. -/
example : num_students_in_second_class
  (λ n, (if n = 25 then 40 else 0 : ℝ)) 
  (λ n, (if n = 30 then 60 else 0 : ℝ)) 
  50.90909090909091 
  25 :=
by {
  intros num_students_class2 h1 h2 h3 h4 h5,
  sorry
}

end num_students_in_second_class_l824_824430


namespace time_to_cross_is_30_seconds_l824_824185

-- Definitions of conditions from the problem statement
def train_length : ℝ := 100
def train_speed_kmh : ℝ := 45
def bridge_length : ℝ := 275

-- Conversion factor
def kmph_to_mps (speed: ℝ) : ℝ := speed * 1000 / 3600

-- Conversion of train speed to m/s
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmh

-- Total distance to be covered
def total_distance : ℝ := train_length + bridge_length

-- Time to cross the bridge
def time_to_cross : ℝ := total_distance / train_speed_mps

-- The theorem ensuring the time to cross the bridge is 30 seconds
theorem time_to_cross_is_30_seconds : time_to_cross = 30 := by
  sorry

end time_to_cross_is_30_seconds_l824_824185


namespace girls_count_l824_824335

noncomputable def num_girls (total_students : ℕ) (avg_age_boys : ℚ) (avg_age_girls : ℚ) 
  (avg_age_school : ℚ) (total_boys : ℕ) (total_girls : ℕ) : Prop :=
  total_students = 652 ∧
  avg_age_boys = 12 ∧
  avg_age_girls = 11 ∧
  avg_age_school = 11.75 ∧
  total_boys + total_girls = total_students ∧
  12 * total_boys + 11 * total_girls = avg_age_school * total_students

theorem girls_count : 
  ∃ total_girls, num_girls 652 12 11 11.75 _ total_girls ∧ total_girls = 162 :=
by
  sorry

end girls_count_l824_824335


namespace distinct_sums_count_l824_824307

open Set

def my_set : Set ℕ := {2, 5, 8, 11, 14, 17, 20, 23}

def num_distinct_sums (s : Set ℕ) : ℕ :=
  (finset.image (λ t : Finset ℕ, (∑ x in t, x)) (finset.powersetLen 4 s.to_finset)).card

theorem distinct_sums_count : num_distinct_sums my_set = 49 :=
  sorry

end distinct_sums_count_l824_824307


namespace days_missed_difference_l824_824250

/-- Mean and median calculation of days missed by students -/

def median_days (histogram : List Nat) (n : Nat) : Nat :=
  let sorted_histogram := histogram.sort
  sorted_histogram.get (n / 2)

def mean_days (histogram : List (Nat × Nat)) (n : Nat) : Rat :=
  (histogram.foldr (λ (pair : Nat × Nat) acc, acc + pair.1 * pair.2) 0) / n

noncomputable def days_difference (histogram : List (Nat × Nat)) (students : Nat) : Rat :=
  let flat_histogram := List.join (histogram.map (λ (h : Nat × Nat), List.replicate h.2 h.1))
  mean_days histogram students - median_days flat_histogram students

theorem days_missed_difference : 
  days_difference [(0, 4), (1, 2), (2, 5), (3, 3), (4, 3), (5, 2), (6, 1)] 20 = 9 / 20 := by
  sorry

end days_missed_difference_l824_824250


namespace parallelogram_angle_l824_824389

theorem parallelogram_angle (λ : ℝ) (h : λ > 1) :
  ∃ φ : ℝ, φ = 2 * real.arccot λ := sorry

end parallelogram_angle_l824_824389


namespace least_b_value_l824_824808

def has_n_factors (k n : ℕ) : Prop := (finset.range (k + 1)).filter (λ d, d ∣ k).card = n
def is_divisible_by (x y : ℕ) : Prop := y ∣ x

theorem least_b_value (a b : ℕ) (h1 : 0 < a) (h2 : has_n_factors a 4) (h3 : has_n_factors b a) (h4 : is_divisible_by b a) : b = 24 :=
by sorry

end least_b_value_l824_824808


namespace solution_set_of_inequality_l824_824836

variable (m x : ℝ)

-- Defining the condition
def inequality (m x : ℝ) := x^2 - (2 * m - 1) * x + m^2 - m > 0

-- Problem statement
theorem solution_set_of_inequality (h : inequality m x) : x < m-1 ∨ x > m :=
  sorry

end solution_set_of_inequality_l824_824836


namespace part1_part2_l824_824263

-- Define the increasing geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = q * b n

-- Define the conditions for b1 and b3
def conditions (b : ℕ → ℝ) : Prop :=
  b 1 + b 3 = 5 ∧ b 1 * b 3 = 4

-- Define the sequence a_n
def a_n (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  Real.logb 2 (b n) + 3

-- Define the sequence c_n
def c_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  1 / (a n * a (n + 1))

-- Define the sum S_n of the sequence c_n
def S_n (c : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum c

-- Define the Lean 4 Theorems
theorem part1 (b : ℕ → ℝ) (h₁ : is_geometric_sequence b) (h₂ : conditions b) :
  ∀ n : ℕ, ∃ d, ∀ m : ℕ, a_n b (m + 1) = a_n b m + d := by
  sorry

theorem part2 (b : ℕ → ℝ) (h₁ : is_geometric_sequence b) (h₂ : conditions b) (n : ℕ)
  (h₃ : ∀ n : ℕ, a_n b n = n + 2) :
  S_n (c_n (a_n b)) n = n / (3 * (n + 3)) := by
  sorry

end part1_part2_l824_824263


namespace range_of_g_l824_824374

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := f (f (f (f (f x))))

theorem range_of_g :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 1 ≤ g x ∧ g x ≤ 2049) :=
begin
  sorry
end

end range_of_g_l824_824374


namespace intersection_of_A_B_find_a_b_l824_824775

-- Lean 4 definitions based on the given conditions
def setA (x : ℝ) : Prop := 4 - x^2 > 0
def setB (x : ℝ) (y : ℝ) : Prop := y = Real.log (-x^2 + 2*x + 3) ∧ -x^2 + 2*x + 3 > 0

-- Prove the intersection of sets A and B
theorem intersection_of_A_B :
  {x : ℝ | setA x} ∩ {x : ℝ | ∃ y : ℝ, setB x y} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

-- On the roots of the quadratic equation and solution interval of inequality
theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, 2 * x^2 + a * x + b < 0 ↔ -3 < x ∧ x < 1) →
  a = 4 ∧ b = -6 :=
by
  sorry

end intersection_of_A_B_find_a_b_l824_824775


namespace proposition_induction_l824_824391

theorem proposition_induction {P : ℕ → Prop} (h : ∀ n, P n → P (n + 1)) (hn : ¬ P 7) : ¬ P 6 :=
by
  sorry

end proposition_induction_l824_824391


namespace coefficient_of_term_containing_1_over_x_value_of_a_l824_824972

theorem coefficient_of_term_containing_1_over_x :
  coefficient_of_term_containing (λ x => (2 * x - 1 / x^(1/2))) 5 (-1) = 10 := 
sorry

theorem value_of_a (a : ℝ) :
  let M := binomial_coefficient_sum (hold 5 2) 5 3
  let N := (1 + a)^6
  4 * M = N → a = 1 ∨ a = -3 :=
sorry

end coefficient_of_term_containing_1_over_x_value_of_a_l824_824972


namespace find_a_l824_824819

noncomputable theory

variables (a x b : ℝ)

theorem find_a (h1 : (a + x) / 2 = 18) (h2 : b = -2) : a = 6 :=
sorry

end find_a_l824_824819


namespace find_f_107_5_l824_824776

theorem find_f_107_5 
  (f : ℝ → ℝ) 
  (h_even : ∀ x, f(-x) = f(x)) 
  (h_func_eq : ∀ x, f(x + 3) = -1 / f(x))
  (h_def_neg : ∀ x, x < 0 → f(x) = 4 * x ) :
  f 107.5 = 1 / 10 := 
sorry

end find_f_107_5_l824_824776


namespace cost_of_four_dozen_apples_l824_824537

theorem cost_of_four_dozen_apples (cost_two_dozen : ℝ) (h : cost_two_dozen = 15.60) : ∃ cost_four_dozen, cost_four_dozen = 31.20 :=
by
  have cost_per_dozen := cost_two_dozen / 2
  have cost_four_dozen := 4 * cost_per_dozen
  use cost_four_dozen
  rw h
  norm_num
  exact eq.refl 31.20

end cost_of_four_dozen_apples_l824_824537


namespace exists_point_Q_l824_824629

-- Define the polygon as a set of points in the plane
structure Polygon :=
  (vertices : set (ℝ × ℝ))
  (is_non_self_intersecting : Prop)
  (is_closed : Prop)

-- Define the enclosing polygon with given conditions
def encloses (S : Polygon) (A B C : ℝ × ℝ) : Prop :=
  ∃ (P : ℝ × ℝ), P ∈ S.vertices ∧ segment P A ⊆ S.vertices ∧ segment P B ⊆ S.vertices ∧ segment P C ⊆ S.vertices

-- Define the problem statement
theorem exists_point_Q (S : Polygon)
  (h1 : S.is_non_self_intersecting)
  (h2 : S.is_closed)
  (h3 : ∀ (A B C : ℝ × ℝ), A ∈ S.vertices → B ∈ S.vertices → C ∈ S.vertices → encloses S A B C) :
  ∃ (Q : ℝ × ℝ), Q ∈ S.vertices ∧ ∀ (D : ℝ × ℝ), D ∈ S.vertices → segment Q D ⊆ S.vertices :=
sorry

end exists_point_Q_l824_824629


namespace sum_sequence_l824_824665

-- Definitions from the conditions
def f (x : ℕ) : ℝ := 1 / (↑x + 1)
def A (n : ℕ) (h : n > 0) : ℝ × ℝ := (n, f n)
def θ (n : ℕ) (h : n > 0) : ℝ := 
  let (x, y) := A n h in 
  real.angle.of_real (y / x)
def cosθ (n : ℕ) (h : n > 0) : ℝ := real.cos (θ n h)
def sinθ (n : ℕ) (h : n > 0) : ℝ := real.sin (θ n h)
def sequence_term (n : ℕ) (h : n > 0) : ℝ := |cosθ n h / sinθ n h|

-- Theorem statement
theorem sum_sequence (s : ℕ → ℝ) (N : ℕ) (h : N > 0) 
  (hs : ∀ n, s n = |cosθ n (nat.succ_pos _) / sinθ n (nat.succ_pos _)|)
  : (finset.sum (finset.range (N + 1)) (λ n, s (nat.succ n))) = 2016 / 2017 :=
sorry

end sum_sequence_l824_824665


namespace triangle_area_ratio_l824_824337

-- We define the given conditions.
variable {α : ℝ} (a : ℝ)

-- Define the main statement as a theorem in Lean
theorem triangle_area_ratio (hα : α > 0) :
  let DE := a * Real.tan α
  let BD := a * Real.cot (α / 2)
  (DE * a / 2) / (BD * a / 2) = 1 / 2 * Real.tan α * Real.tan (α / 2) :=
by 
  sorry

end triangle_area_ratio_l824_824337


namespace smallest_seven_star_number_greater_than_2000_l824_824965

def is_factor (a b : ℕ) : Prop := b % a = 0

def seven_star_number (N : ℕ) : Prop :=
  ∃ factors, (factors ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
             (∀ f ∈ factors, is_factor f N) ∧
             factors.card ≥ 7

theorem smallest_seven_star_number_greater_than_2000 : 
  ∃ N, N > 2000 ∧ seven_star_number N ∧ ∀ M, M > 2000 ∧ seven_star_number M → N ≤ M :=
begin
  use 2016,
  split,
  { exact 2016 > 2000 },
  split,
  { -- proof that 2016 is a seven-star number
    sorry },
  { -- proof that 2016 is the smallest such number
    sorry }
end

end smallest_seven_star_number_greater_than_2000_l824_824965


namespace derivative_at_3_l824_824987

noncomputable def f (x : ℝ) := x^2

theorem derivative_at_3 : deriv f 3 = 6 := by
  sorry

end derivative_at_3_l824_824987


namespace equation_of_line_AB_l824_824683

def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def on_circle (C : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - C.1) ^ 2 + P.2 ^ 2 = r ^ 2

theorem equation_of_line_AB : 
  ∃ A B : ℝ × ℝ, 
    is_midpoint (2, -1) A B ∧ 
    on_circle (1, 0) 5 A ∧ 
    on_circle (1, 0) 5 B ∧ 
    ∀ x y : ℝ, (x - y - 3 = 0) ∧ 
    ∃ t : ℝ, ∃ u : ℝ, (t - u - 3 = 0) := 
sorry

end equation_of_line_AB_l824_824683


namespace shaded_region_area_l824_824344

def area_of_shaded_region (grid_height grid_width triangle_base triangle_height : ℝ) : ℝ :=
  let total_area := grid_height * grid_width
  let triangle_area := 0.5 * triangle_base * triangle_height
  total_area - triangle_area

theorem shaded_region_area :
  area_of_shaded_region 3 15 5 3 = 37.5 :=
by 
  sorry

end shaded_region_area_l824_824344


namespace max_min_product_l824_824761

theorem max_min_product (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h_sum : p + q + r = 13) (h_prod_sum : p * q + q * r + r * p = 30) :
  ∃ n, n = min (p * q) (min (q * r) (r * p)) ∧ n = 10 :=
by
  sorry

end max_min_product_l824_824761


namespace remainder_of_polynomial_division_l824_824122

-- Define the polynomials
def P(x : ℝ) := 3*x^5 + 15*x^4 - 42*x^3 - 60*x^2 + 48*x - 47
def D(x : ℝ) := x^3 + 7*x^2 + 5*x - 5
def R(x : ℝ) := 3*x - 47

-- State that the remainder when dividing P(x) by D(x) is R(x)
theorem remainder_of_polynomial_division :
  ∃ Q : ℝ → ℝ, P(x) = Q(x) * D(x) + R(x) :=
sorry

end remainder_of_polynomial_division_l824_824122


namespace probability_sum_odd_l824_824849

open Classical
open Probability

noncomputable def probability_odd_sum : ℚ := sorry

theorem probability_sum_odd (n : ℕ) (h : n = 3) : 
  probability_odd_sum = 7 / 16 := by
  sorry

end probability_sum_odd_l824_824849


namespace number_of_partitions_of_7_into_4_parts_l824_824310

theorem number_of_partitions_of_7_into_4_parts : 
  (finset.attach (finset.powerset (finset.range (7+1)))).filter (λ s, s.sum = 7 ∧ s.card ≤ 4)).card = 11 := 
by sorry

end number_of_partitions_of_7_into_4_parts_l824_824310


namespace order_of_abc_l824_824974

noncomputable section

open Real

def x : ℝ := sorry -- placeholder for x in (0, π / 4)
def a : ℝ := cos (x ^ (sin (x ^ (sin x))))
def b : ℝ := sin (x ^ (cos (x ^ (π * x)) * x))
def c : ℝ := cos (x ^ (sin (x ^ (san x))))

theorem order_of_abc : 0 < x ∧ x < π / 4 → b < a ∧ a < c := 
sorry

end order_of_abc_l824_824974


namespace simplify_expr_l824_824414

-- Define the given expression using square roots and exponents
def expr1 := Real.sqrt (3 * 5)
def expr2 := Real.sqrt (5^3 * 3^3)

-- Combine both expressions
def combined_expr := expr1 * expr2

-- Define the target value we want to prove the combined expression equals to
def target_value := 225

-- State the theorem
theorem simplify_expr : combined_expr = target_value := by
  sorry

end simplify_expr_l824_824414


namespace slices_per_pizza_l824_824860

theorem slices_per_pizza (total_slices : ℕ) (total_pizzas : ℕ) (h1 : total_slices = 14) (h2 : total_pizzas = 7) : 
  (total_slices / total_pizzas = 2) :=
by
  rw [h1, h2]
  simp
sorry

end slices_per_pizza_l824_824860


namespace trefoil_is_reliable_l824_824838

structure BunkerSystem :=
  (bunkers : Type)
  (trenches : bunkers → bunkers → Prop)
  (reachable : ∀ {a b : bunkers}, (reachable a b))

def trefoil_system : BunkerSystem := sorry

def isReliable (s : BunkerSystem) : Prop :=
  ∀ cannon_strategy : ℕ → s.bunkers,
    ∃ soldier_strategy : ℕ → s.bunkers,
      ∀ n, soldier_strategy n ≠ cannon_strategy n

theorem trefoil_is_reliable : isReliable trefoil_system := sorry

end trefoil_is_reliable_l824_824838


namespace sum_first_3n_terms_l824_824456

-- Geometric Sequence: Sum of first n terms Sn, first 2n terms S2n, first 3n terms S3n.
variables {n : ℕ} {S : ℕ → ℕ}

-- Conditions
def sum_first_n_terms (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 48
def sum_first_2n_terms (S : ℕ → ℕ) (n : ℕ) : Prop := S (2 * n) = 60

-- Theorem to Prove
theorem sum_first_3n_terms {S : ℕ → ℕ} (h1 : sum_first_n_terms S n) (h2 : sum_first_2n_terms S n) :
  S (3 * n) = 63 :=
sorry

end sum_first_3n_terms_l824_824456


namespace least_integer_gt_sqrt_500_l824_824105

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824105


namespace least_integer_greater_than_sqrt_500_l824_824063

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824063


namespace smallest_hiding_number_l824_824393

def hides (A B : Nat) : Prop :=
∃ (indices : List Nat), (List.length indices = String.length (toString B)) ∧ 
  (String.ofList $ List.map (λ i, (toString A).get i) indices = toString B)

theorem smallest_hiding_number : 
  ∃ (n : Nat), n = 1201201 ∧ 
  hides n 2021 ∧ 
  hides n 2120 ∧ 
  hides n 1220 ∧ 
  hides n 1202 :=
by
  use 1201201
  split
  exact rfl
  repeat { sorry }

end smallest_hiding_number_l824_824393


namespace marching_band_members_l824_824441

theorem marching_band_members :
  ∃ (n : ℕ), 100 < n ∧ n < 200 ∧
             n % 4 = 1 ∧
             n % 5 = 2 ∧
             n % 7 = 3 :=
  by sorry

end marching_band_members_l824_824441


namespace sqrt_500_least_integer_l824_824022

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824022


namespace num_subsets_satisfy_property_M_l824_824628

-- Define property M for subsets
def property_M (A : Finset ℕ) : Prop :=
  (∃ k ∈ A, k + 1 ∈ A) ∧ ∀ k ∈ A, k - 2 ∉ A

-- The main statement - number of 3-element subsets with property M
theorem num_subsets_satisfy_property_M :
  (Finset.filter property_M { A : Finset ℕ // A.card = 3 } (Finset.powerset (Finset.range' 1 6))).card = 6 :=
sorry

end num_subsets_satisfy_property_M_l824_824628


namespace quadratic_polynomial_with_root_and_coefficient_l824_824604

theorem quadratic_polynomial_with_root_and_coefficient :
  ∃ (a b c : ℝ), (a = 3) ∧ (4 + 2 * (1 : ℝ) * complex.I = 4 + 2i) ∧
  (∀ (x : ℂ), (x = 4 + 2i) ∨ (x = 4 - 2i) → 3 * (x - 4 - 2i) * (x - 4 + 2i) = 3 * x^2 - 24 * x + 60) :=
by
  sorry

end quadratic_polynomial_with_root_and_coefficient_l824_824604


namespace choir_arrangement_l824_824893

theorem choir_arrangement (boys : ℕ) (girls : ℕ) (select_boys : ℕ) (select_girls : ℕ) : 
  boys = 4 ∧ girls = 3 ∧ select_boys = 2 ∧ select_girls = 2 → 
  ∃ arrangements : ℕ, arrangements = 216 :=
by
  intro h
  cases h with hb hg
  cases hg with hg hg'
  exists 216
  sorry

end choir_arrangement_l824_824893


namespace x1x2_lt_one_l824_824255

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  |exp x - exp 1| + exp x + a * x

theorem x1x2_lt_one (a x1 x2 : ℝ) 
  (ha : a < -exp 1) 
  (hzero1 : f a x1 = 0) 
  (hzero2 : f a x2 = 0) 
  (h_order : x1 < x2) : x1 * x2 < 1 := 
sorry

end x1x2_lt_one_l824_824255


namespace sine_addition_l824_824233

noncomputable def sin_inv_45 := Real.arcsin (4 / 5)
noncomputable def tan_inv_12 := Real.arctan (1 / 2)

theorem sine_addition :
  Real.sin (sin_inv_45 + tan_inv_12) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sine_addition_l824_824233


namespace at_least_seven_coins_of_same_denomination_l824_824887

theorem at_least_seven_coins_of_same_denomination 
  (num_coins : ℕ) (denominations : ℕ) (num_coins = 25) (denominations = 4) : 
  ∃ (d : ℕ), d ≤ denominations ∧ (denom_count : ℕ) ≥ 7 :=
by 
  sorry

end at_least_seven_coins_of_same_denomination_l824_824887


namespace BRAIN_7225_cycle_line_number_l824_824828

def BRAIN_cycle : Nat := 5
def _7225_cycle : Nat := 4

theorem BRAIN_7225_cycle_line_number : Nat.lcm BRAIN_cycle _7225_cycle = 20 :=
by
  sorry

end BRAIN_7225_cycle_line_number_l824_824828


namespace calculate_savings_l824_824133

-- Definitions of expenses
def rent := 5000
def milk := 1500
def groceries := 4500
def education := 2500
def petrol := 2000
def misc := 2500

-- Definition of total expenses
def total_expenses := rent + milk + groceries + education + petrol + misc

-- Definition of saving rate
def saving_rate := 0.10

-- Definition of monthly salary and savings
def monthly_salary (savings : ℝ) := total_expenses / (1 - saving_rate)
def savings (monthly_salary : ℝ) := saving_rate * monthly_salary

-- Proving that the savings are Rs. 2000
theorem calculate_savings : savings (monthly_salary 2000) = 2000 :=
by
  sorry

end calculate_savings_l824_824133


namespace least_integer_greater_than_sqrt_500_l824_824040

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824040


namespace probability_no_shaded_square_l824_824156

theorem probability_no_shaded_square (n m : ℕ) (h1 : n = (2006.choose 2)) (h2 : m = 1003^2) : 
  (1 - (m / n) = (1002 / 2005)) := 
by
  -- Number of rectangles in one row
  have hn : n = 1003 * 2005 := h1
  -- Number of rectangles in one row containing a shaded square
  have hm : m = 1003 * 1003 := h2
  sorry

end probability_no_shaded_square_l824_824156


namespace chord_length_is_four_l824_824171

theorem chord_length_is_four :
  let line := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 2 + 2 * t ∧ p.2 = -t}
  let circle_center : ℝ × ℝ := (2, 0)
  let circle_radius : ℝ := 2
  let circle := {p : ℝ × ℝ | (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2}
  ∀ a b : ℝ × ℝ, a ∈ line → b ∈ line → a ∈ circle → b ∈ circle → a ≠ b → dist a b = 4 :=
by
  sorry

end chord_length_is_four_l824_824171


namespace cost_for_four_dozen_l824_824544

-- Definitions based on conditions
def cost_of_two_dozen_apples : ℝ := 15.60
def cost_of_one_dozen_apples : ℝ := cost_of_two_dozen_apples / 2
def cost_of_four_dozen_apples : ℝ := 4 * cost_of_one_dozen_apples

-- Statement to prove
theorem cost_for_four_dozen :
  cost_of_four_dozen_apples = 31.20 :=
sorry

end cost_for_four_dozen_l824_824544


namespace tangent_line_at_one_max_value_t_inequality_sum_l824_824660

noncomputable def f (x : ℝ) := (Real.log x) / (x + 1)

-- Problem 1: Equation of the tangent line
theorem tangent_line_at_one :
  ∀ (x y : ℝ), x = 1 → y = f 1 →
  x - 2 * y - 1 = 0 :=
sorry

-- Problem 2(i): Maximum value of t
theorem max_value_t :
  ∀ (x t : ℝ), x > 0 → x ≠ 1 →
  (f x - t / x > Real.log x / (x - 1)) → t ≤ -1 :=
sorry

-- Problem 2(ii): Inequality involving n
theorem inequality_sum :
  ∀ (n : ℕ), 2 ≤ n →
  Real.log n < (Finset.sum (Finset.range n) (λ i, 1 / (i + 1)) - 1/2 - 1/(2 * n)) :=
sorry

end tangent_line_at_one_max_value_t_inequality_sum_l824_824660


namespace least_integer_greater_than_sqrt_500_l824_824028

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824028


namespace approximate_number_of_fish_in_pond_l824_824489

theorem approximate_number_of_fish_in_pond
  (tagged_first_catch : ℕ)
  (total_first_catch : ℕ)
  (tagged_second_catch : ℕ)
  (total_second_catch : ℕ)
  (h_tagged_first_catch : tagged_first_catch = 50)
  (h_total_first_catch : total_first_catch = 50)
  (h_tagged_second_catch : tagged_second_catch = 10)
  (h_total_second_catch : total_second_catch = 50)
  (proportion : ℚ)
  (h_proportion : proportion = (tagged_second_catch : ℚ) / (total_second_catch : ℚ)) :
  (total_tagged_fish : ℚ)
  (h_total_tagged_fish: total_tagged_fish = tagged_first_catch) :
  total_tagged_fish / proportion = 250 := 
begin
  sorry,
end

end approximate_number_of_fish_in_pond_l824_824489


namespace age_multiple_l824_824528

theorem age_multiple (ron_age_now : ℕ) (maurice_age_now : ℕ) 
  (h_ron : ron_age_now = 43) 
  (h_maurice : maurice_age_now = 7) : 
  (ron_age_now + 5) / (maurice_age_now + 5) = 4 := by
  rw [h_ron, h_maurice]
  sorry

end age_multiple_l824_824528


namespace complex_number_not_in_fourth_quadrant_l824_824432

-- Define the conditions and the question
variable (m : ℝ)
def z : ℂ := (m-1)/2 + ((m+1)/2) * complex.i

theorem complex_number_not_in_fourth_quadrant (h : (1 - complex.i) * z = m + complex.i) : ¬ ((m - 1) > 0 ∧ (m + 1) < 0) :=
sorry

end complex_number_not_in_fourth_quadrant_l824_824432


namespace find_f_lg_lg_2_l824_824345

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * (Real.sin x) + 4

theorem find_f_lg_lg_2 (a b : ℝ) (m : ℝ) 
  (h1 : f a b (Real.logb 10 2) = 5) 
  (h2 : m = Real.logb 10 2) : 
  f a b (Real.logb 2 m) = 3 :=
sorry

end find_f_lg_lg_2_l824_824345


namespace find_7c_7d_l824_824933

noncomputable def h (x : ℝ) := 7*x - 6
noncomputable def f_inv (x : ℝ) := 7*x - 1
noncomputable def f (x : ℝ) := (x + 1) / 7

theorem find_7c_7d :
  (∃ c d : ℝ, f = λ x, c * x + d ∧ f_inv = λ x, 7 * x - 1 ∧ 7 * c + 7 * d = 2) :=
by {
  use [1 / 7, 1 / 7],
  split; {
    intros x,
    split,
    { unfold f, unfold h, field_simp, ring },
    { ring_nf } },
  { ring },
  { sorry }
}

end find_7c_7d_l824_824933


namespace num_sets_satisfying_condition_l824_824830

open Set

def satisfies_condition (A : Set ℕ) : Prop := {1, 3} ∪ A = {1, 3, 5}

theorem num_sets_satisfying_condition : 
  {A : Set ℕ | satisfies_condition A}.toFinset.card = 4 := 
by
  sorry

end num_sets_satisfying_condition_l824_824830


namespace effect_on_revenue_l824_824875

-- Define the conditions using parameters and variables

variables {P Q : ℝ} -- Original price and quantity of TV sets

def new_price (P : ℝ) : ℝ := P * 1.60 -- New price after 60% increase
def new_quantity (Q : ℝ) : ℝ := Q * 0.80 -- New quantity after 20% decrease

def original_revenue (P Q : ℝ) : ℝ := P * Q -- Original revenue
def new_revenue (P Q : ℝ) : ℝ := (new_price P) * (new_quantity Q) -- New revenue

theorem effect_on_revenue
  (P Q : ℝ) :
  new_revenue P Q = original_revenue P Q * 1.28 :=
by
  sorry

end effect_on_revenue_l824_824875


namespace A1B1C1D1_is_parallelogram_area_of_A1B1C1D1_l824_824261

variables {A B C D A1 B1 C1 D1 : Type} [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ A1] [AffineSpace ℝ B1] [AffineSpace ℝ C1] [AffineSpace ℝ D1]

-- Given conditions
def is_midpoint (P Q M : Type) [AffineSpace ℝ P] [AffineSpace ℝ Q] [AffineSpace ℝ M] :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ vectorSpan ℝ {P, Q} = vectorSpan ℝ {M}

def is_parallelogram (A B C D : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] :=
  ∃ x y : ℝ, x * (vectorSpan ℝ {A, B}) + y * (vectorSpan ℝ {A, D}) = vectorSpan ℝ {A, C} ∧
             ∃ x y : ℝ, x * (vectorSpan ℝ {B, C}) + y * (vectorSpan ℝ {B, D}) = vectorSpan ℝ {B, D}

def area_parallelogram (A B C D : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] : ℝ := 
  -- This would be a mathematical function to compute the area
  sorry

def area_condition (A B C D : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] : Prop := 
  area_parallelogram A B C D = 1

-- Proof claims
theorem A1B1C1D1_is_parallelogram :
  is_parallelogram A B C D →
  is_midpoint D D1 A →
  is_midpoint A A1 B →
  is_midpoint B B1 C →
  is_midpoint C C1 D →
  is_parallelogram A1 B1 C1 D1 :=
sorry

theorem area_of_A1B1C1D1 :
  is_parallelogram A B C D →
  is_midpoint D D1 A →
  is_midpoint A A1 B →
  is_midpoint B B1 C →
  is_midpoint C C1 D →
  area_condition A B C D →
  area_parallelogram A1 B1 C1 D1 = 1 :=
sorry

end A1B1C1D1_is_parallelogram_area_of_A1B1C1D1_l824_824261


namespace largest_n_divides_30_factorial_l824_824593

theorem largest_n_divides_30_factorial : 
  ∃ (n : ℕ), (∀ (a b : ℕ), (prime a ∧ prime b ∧ a * b = 15) → prime_factors_in_factorial 30 a ≥ n ∧ prime_factors_in_factorial 30 b ≥ n) ∧ 
             (∀ m : ℕ, (∀ (a b : ℕ), (prime a ∧ prime b ∧ a * b = 15) → prime_factors_in_factorial 30 a ≥ m ∧ prime_factors_in_factorial 30 b ≥ m) → m ≤ n) ∧ n = 7 :=
by {
  sorry
}

end largest_n_divides_30_factorial_l824_824593


namespace larger_number_is_28_l824_824700

theorem larger_number_is_28
  (x y : ℕ)
  (h1 : 4 * y = 7 * x)
  (h2 : y - x = 12) : y = 28 :=
sorry

end larger_number_is_28_l824_824700


namespace systematic_sampling_second_student_is_20_l824_824718

noncomputable def total_students : ℕ := 56
noncomputable def sample_size : ℕ := 4
noncomputable def known_students : List ℕ := [6, 34, 48]

theorem systematic_sampling_second_student_is_20 :
  ∃ k (second_student : ℕ), 
    k = total_students / sample_size ∧
    second_student = 6 + k ∧
    known_students = [6, second_student + k, second_student + 2 * k] ∧ 
    second_student = 20 :=
by
  sorry

end systematic_sampling_second_student_is_20_l824_824718


namespace find_a8_l824_824638

variable (a : ℕ → ℤ)

def arithmetic_sequence : Prop :=
  ∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)

theorem find_a8 (h1 : a 7 + a 9 = 16) (h2 : arithmetic_sequence a) : a 8 = 8 := by
  -- proof would go here
  sorry

end find_a8_l824_824638


namespace isosceles_triangle_sides_l824_824626

-- Definitions based on the problem conditions
def circle_radius : ℝ := 3
def base_angle_deg : ℝ := 30
def base_angle_rad : ℝ := real.pi * base_angle_deg / 180

theorem isosceles_triangle_sides
  (r : ℝ)
  (angle_rad : ℝ)
  (h_r : r = circle_radius)
  (h_angle_rad : angle_rad = base_angle_rad)
  :
  ∃ (s1 s2 s3 : ℝ),
    s1 = 6 * real.sqrt 3 + 3 ∧
    s2 = 6 * real.sqrt 3 + 3 ∧
    s3 = 12 + 6 * real.sqrt 3 :=
sorry

end isosceles_triangle_sides_l824_824626


namespace problem_statement_l824_824141

theorem problem_statement :
  2 * Real.log 2 / Real.log 3 -
  Real.log (32 / 9) / Real.log 3 +
  Real.log 8 / Real.log 3 -
  2 * (5 ^ (Real.log 3 / Real.log 5)) +
  16 ^ 0.75 = 1 := 
sorry

end problem_statement_l824_824141


namespace sum_inequality_l824_824494

noncomputable def a (n : ℕ) : ℚ := 2^(-(2 * (n + 1) + 1))

noncomputable def c (n : ℕ) : ℚ :=
  if n = 1 then 0
  else (List.range n).drop 1.sum (λ k, log (2 : ℚ)⁻¹ (a k))

theorem sum_inequality (n : ℕ) (hn : 2 ≤ n) :
  (1 / 3 : ℚ) ≤ ∑ k in Finset.range n, if k = 0 ∨ k = 1 then 0 else (1 / c k) ∧
  (∑ k in Finset.range n, if k = 0 ∨ k = 1 then 0 else (1 / c k)) < 3 / 4 := 
sorry

end sum_inequality_l824_824494


namespace new_car_travel_distance_l824_824174

-- Define the distance traveled by the older car
def distance_older_car : ℝ := 150

-- Define the percentage increase
def percentage_increase : ℝ := 0.30

-- Define the condition for the newer car's travel distance
def distance_newer_car (d_old : ℝ) (perc_inc : ℝ) : ℝ :=
  d_old * (1 + perc_inc)

-- Prove the main statement
theorem new_car_travel_distance :
  distance_newer_car distance_older_car percentage_increase = 195 := by
  -- Skip the proof body as instructed
  sorry

end new_car_travel_distance_l824_824174


namespace cyclist_distance_l824_824164

theorem cyclist_distance
  (x t d : ℝ)
  (h1 : d = x * t)
  (h2 : d = (x + 1) * (3 * t / 4))
  (h3 : d = (x - 1) * (t + 3)) :
  d = 18 :=
by {
  sorry
}

end cyclist_distance_l824_824164


namespace log_identity_l824_824688

theorem log_identity (c b : ℝ) (h1 : c = Real.log 81 / Real.log 4) (h2 : b = Real.log 3 / Real.log 2) : c = 2 * b := by
  sorry

end log_identity_l824_824688


namespace eq_1_solution_eq_2_solution_eq_3_solution_eq_4_solution_l824_824803

-- Equation (1): 2x^2 + 2x - 1 = 0
theorem eq_1_solution (x : ℝ) :
  2 * x^2 + 2 * x - 1 = 0 ↔ (x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2) := by
  sorry

-- Equation (2): x(x-1) = 2(x-1)
theorem eq_2_solution (x : ℝ) :
  x * (x - 1) = 2 * (x - 1) ↔ (x = 1 ∨ x = 2) := by
  sorry

-- Equation (3): 4(x-2)^2 = 9(2x+1)^2
theorem eq_3_solution (x : ℝ) :
  4 * (x - 2)^2 = 9 * (2 * x + 1)^2 ↔ (x = -7 / 4 ∨ x = 1 / 8) := by
  sorry

-- Equation (4): (2x-1)^2 - 3(2x-1) = 4
theorem eq_4_solution (x : ℝ) :
  (2 * x - 1)^2 - 3 * (2 * x - 1) = 4 ↔ (x = 5 / 2 ∨ x = 0) := by
  sorry

end eq_1_solution_eq_2_solution_eq_3_solution_eq_4_solution_l824_824803


namespace systematic_sampling_student_selection_l824_824725

theorem systematic_sampling_student_selection
    (total_students : ℕ)
    (num_groups : ℕ)
    (students_per_group : ℕ)
    (third_group_selected : ℕ)
    (third_group_num : ℕ)
    (eighth_group_num : ℕ)
    (h1 : total_students = 50)
    (h2 : num_groups = 10)
    (h3 : students_per_group = total_students / num_groups)
    (h4 : students_per_group = 5)
    (h5 : 11 ≤ third_group_selected ∧ third_group_selected ≤ 15)
    (h6 : third_group_selected = 12)
    (h7 : third_group_num = 3)
    (h8 : eighth_group_num = 8) :
  eighth_group_selected = 37 :=
by
  sorry

end systematic_sampling_student_selection_l824_824725


namespace probability_of_minimal_arrangement_l824_824783

open Finset

def arrangements := {l : List ℕ | l.length = 9 ∧ l.toFinset = (finset.range 1 10)}

noncomputable def all_arrangements := arrangements.card / 2

noncomputable def minimal_arrangements := (2 ^ 6)

theorem probability_of_minimal_arrangement :
  minimal_arrangements / (all_arrangements : ℝ) = 1 / 315 := 
sorry

end probability_of_minimal_arrangement_l824_824783


namespace least_integer_greater_than_sqrt_500_l824_824110

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824110


namespace simplify_fraction_l824_824802

theorem simplify_fraction : 
  (5 / (2 * Real.sqrt 27 + 3 * Real.sqrt 12 + Real.sqrt 108)) = (5 * Real.sqrt 3 / 54) :=
by
  -- Proof will be provided here
  sorry

end simplify_fraction_l824_824802


namespace smallest_k_sum_of_squares_multiple_of_360_l824_824606

theorem smallest_k_sum_of_squares_multiple_of_360 :
    ∃ k : ℕ, (∑ i in Finset.range (k + 1), i ^ 2) % 360 = 0 ∧ k = 72 :=
sorry

end smallest_k_sum_of_squares_multiple_of_360_l824_824606


namespace c_alone_works_in_two_days_l824_824132

-- Define the conditions
def workRateAB : ℝ := 1 / 2  -- work rate of A and B together (work per day)
def workRateABC : ℝ := 1     -- work rate of A, B, and C together (work per day)

-- Theorem stating C alone can finish the work in 2 days
theorem c_alone_works_in_two_days :
  ∃ W_C : ℝ, workRateABC = workRateAB + W_C ∧ (1 / W_C = 2) :=
by 
  sorry

end c_alone_works_in_two_days_l824_824132


namespace increasing_interval_iff_sum_ln_inequality_l824_824291

noncomputable def f (a x : ℝ) : ℝ := ln (1 + x) - x - a * x^2

theorem increasing_interval_iff (a : ℝ) :
  (∃ x ∈ Set.Icc (-1/2 : ℝ) (-1/3 : ℝ), 0 < deriv (f a) x) ↔ (-1 < a) := 
sorry 

theorem sum_ln_inequality (n : ℕ) (h : 0 < n) :
  (∑ k in Finset.range n, 1 / (Real.log (k + 2))) > n / (n + 1) := 
sorry

end increasing_interval_iff_sum_ln_inequality_l824_824291


namespace arithmetic_geometric_sequence_l824_824980

theorem arithmetic_geometric_sequence : 
  ∀ (a : ℤ), (∀ n : ℤ, a_n = a + (n-1) * 2) → 
  (a + 4)^2 = a * (a + 6) → 
  (a + 10 = 2) :=
by
  sorry

end arithmetic_geometric_sequence_l824_824980


namespace find_number_l824_824480

-- Define the given conditions
def twelve_percent_of_160 : ℝ := 0.12 * 160
def thirty_eight_percent_of (x : ℝ) : ℝ := 0.38 * x
def difference (a b : ℝ) : ℝ := a - b

-- State the main theorem/proof problem
theorem find_number (y : ℝ) : difference twelve_percent_of_160 (thirty_eight_percent_of y) = 11.2 -> y ≈ 21.05 :=
by {
  -- Since we only need the statement and not the proof, we use sorry here.
  sorry
}

end find_number_l824_824480


namespace conic_section_is_parabola_l824_824939

theorem conic_section_is_parabola (x y : ℝ) : 
  |x-3| = sqrt((y+4)^2 + (x-1)^2) → conic_section_type = "P" :=
by 
  sorry

end conic_section_is_parabola_l824_824939


namespace rank_skew_symmetric_matrix_l824_824967

open Matrix

noncomputable def skew_symmetric_matrix (n : ℕ) : Matrix (Fin (2 * n + 1)) (Fin (2 * n + 1)) ℤ :=
  λ i j, if h : i + n ≥ j then 1 else -1

theorem rank_skew_symmetric_matrix (n : ℕ) :
  Matrix.rank (skew_symmetric_matrix n) = 2 * n := by
  sorry

end rank_skew_symmetric_matrix_l824_824967


namespace Connie_marbles_left_l824_824561

-- Define the initial conditions
def initial_marbles : ℕ := 776
def given_away_marbles : ℝ := 183.0

-- Define the hypothesis on how many marbles are left
theorem Connie_marbles_left : initial_marbles - given_away_marbles.toInt = 593 :=
by
  -- Proof steps will follow here
  sorry

end Connie_marbles_left_l824_824561


namespace angle_hyperbola_l824_824670

theorem angle_hyperbola (a b : ℝ) (e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (hyperbola_eq : ∀ (x y : ℝ), ((x^2)/(a^2) - (y^2)/(b^2) = 1)) 
  (eccentricity_eq : e = 2 + Real.sqrt 6 - Real.sqrt 3 - Real.sqrt 2) :
  ∃ α : ℝ, α = 15 :=
by
  sorry

end angle_hyperbola_l824_824670


namespace mathematician_reaches_treasure_l824_824545

-- Define the basics of the treasure hunt problem
noncomputable def island_radius : ℝ := 1
noncomputable def device_range : ℝ := 0.5
noncomputable def start_distance_to_center : ℝ := 1

-- Define the question
def can_reach_treasure : Prop :=
  ∃ path_length < 4, 
  ∀ device_signal_range, 
  (distance to treasure is within device range) ∧ (path_length < 4)

-- Define the conditions
def conditions : Prop := 
  island_radius = 1 ∧ 
  device_range = 0.5 ∧ 
  start_distance_to_center = 1

-- The final statement to prove
theorem mathematician_reaches_treasure : conditions → can_reach_treasure :=
begin
  sorry
end

end mathematician_reaches_treasure_l824_824545


namespace angle_between_line_and_plane_l824_824702

theorem angle_between_line_and_plane {l α : Type} 
  (angle_direction_vector_normal_vector : ℝ)
  (h : angle_direction_vector_normal_vector = 120) :
  ∃ (angle_between_line_plane : ℝ), angle_between_line_plane = 30 :=
by {
  use 30,
  sorry
}

end angle_between_line_and_plane_l824_824702


namespace distance_covered_approx_l824_824526

noncomputable def pi_approx : ℝ := 3.14159
noncomputable def diameter : ℝ := 14
noncomputable def revolutions : ℝ := 11.010009099181074
noncomputable def circumference : ℝ := pi_approx * diameter
noncomputable def distance_covered : ℝ := circumference * revolutions

theorem distance_covered_approx :
  distance_covered ≈ 484.110073 := by
  sorry

end distance_covered_approx_l824_824526


namespace number_of_selections_l824_824460

theorem number_of_selections (num_classes num_spots : ℕ) :
  num_classes = 3 → num_spots = 5 → (num_spots ^ num_classes) = 5^3 :=
by
  intros h_classes h_spots
  rw [h_classes, h_spots]
  apply rfl

end number_of_selections_l824_824460


namespace tan_ratio_l824_824767

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : (Real.tan x) / (Real.tan y) = 2 := 
by
  sorry

end tan_ratio_l824_824767


namespace domain_of_f_l824_824960

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (x^3))

theorem domain_of_f : set_of (λ x : ℝ, -1 ≤ x ∧ x ≤ 1) = set_of (λ x : ℝ, ∃ y : ℝ, f y = x) := by
  sorry

end domain_of_f_l824_824960


namespace least_integer_greater_than_sqrt_500_l824_824118

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824118


namespace least_integer_greater_than_sqrt_500_l824_824009

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824009


namespace GCD_180_252_315_l824_824468

theorem GCD_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by
  sorry

end GCD_180_252_315_l824_824468


namespace BC_value_area_ABD_l824_824745

variables {α : Type*} [LinearOrderedField α] 
variables (A B C D : α × α)
variables (angle : α) (AB AC AD BC : α)

def BAC := (2 * Real.pi) / 3

def BD_DC_ratio := 2
def area_ABC := 3 * Real.sqrt 3

-- Assumptions
def conditions :=
  (angle = BAC) ∧ 
  (AB = 3) ∧ 
  (AD = 1) ∧
  (BD / DC = BD_DC_ratio) ∧
  (1 / 2 * AB * AC * Real.sin angle = area_ABC)

-- Statement 1: Prove BC
theorem BC_value (h : conditions) : 
  BC = Real.sqrt 37 := 
sorry

-- Statement 2: Prove area of ΔABD
theorem area_ABD (h : conditions ∧ BC = Real.sqrt 37) : 
  1 / 2 * AD * BD * Real.sin (2 * Real.pi / 3) = 3 * Real.sqrt 3 / 4 := 
sorry

end BC_value_area_ABD_l824_824745


namespace sum_first_two_integers_l824_824607

/-- Prove that the sum of the first two integers n > 1 such that 3^n is divisible by n 
and 3^n - 1 is divisible by n - 1 is equal to 30. -/
theorem sum_first_two_integers (n : ℕ) (h1 : n > 1) (h2 : 3 ^ n % n = 0) (h3 : (3 ^ n - 1) % (n - 1) = 0) : 
  n = 3 ∨ n = 27 → n + 3 + 27 = 30 :=
sorry

end sum_first_two_integers_l824_824607


namespace least_integer_greater_than_sqrt_500_l824_824035

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824035


namespace angle_between_median_and_altitude_l824_824794

-- Define the triangle and its relevant geometric elements
structure triangle (A B C : Type) [plane_geometry] :=
(right_angle_at_C : is_right_angle C)
(CD_is_altitude : is_altitude C D)
(CE_is_median : is_median C E)

-- Define the angles in the triangle
variables {A B C : Type} [plane_geometry] (T : triangle A B C)
variables (angle_A angle_B : ℝ)

-- The theorem statement
theorem angle_between_median_and_altitude :
  abs (angle_B - angle_A) = ∠ DCE :=
sorry

end angle_between_median_and_altitude_l824_824794


namespace find_weight_first_dog_l824_824457

noncomputable def weight_first_dog (x : ℕ) (y : ℕ) : Prop :=
  (x + 31 + 35 + 33) / 4 = (x + 31 + 35 + 33 + y) / 5

theorem find_weight_first_dog (x : ℕ) : weight_first_dog x 31 → x = 25 := by
  sorry

end find_weight_first_dog_l824_824457


namespace fraction_evaluation_l824_824616

theorem fraction_evaluation :
  let x := 198719871987
  in (x: ℕ) / (x^2 - (x - 1) * (x + 1)) = x :=
by
  let x := 198719871987
  show (x: ℕ) / (x^2 - (x - 1) * (x + 1)) = x
  sorry

end fraction_evaluation_l824_824616


namespace average_score_l824_824426

variable (score : Fin 5 → ℤ)
variable (actual_score : ℤ)
variable (rank : Fin 5)
variable (average : ℤ)

def students_scores_conditions := 
  score 0 = 10 ∧ score 1 = -5 ∧ score 2 = 0 ∧ score 3 = 8 ∧ score 4 = -3 ∧
  actual_score = 90 ∧ rank.val = 2

theorem average_score (h : students_scores_conditions score actual_score rank) :
  average = 92 :=
sorry

end average_score_l824_824426


namespace distinct_expression_count_l824_824676

theorem distinct_expression_count (n : ℕ) (hn : n ≥ 2) : 
  (number_of_distinct_expressions n) = 2^(n - 2) :=
sorry

end distinct_expression_count_l824_824676


namespace least_integer_gt_sqrt_500_l824_824099

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824099


namespace polygon_with_given_angle_sums_is_hexagon_l824_824709

theorem polygon_with_given_angle_sums_is_hexagon
  (n : ℕ)
  (h_interior : (n - 2) * 180 = 2 * 360) :
  n = 6 :=
by
  sorry

end polygon_with_given_angle_sums_is_hexagon_l824_824709


namespace find_a_n_find_b_n_find_m_find_T_n_l824_824632

variable (a b c T : ℕ → ℕ)
variable (S H : ℕ → ℝ)

-- Problem conditions
axiom cond1 : ∀ n, S n = 2 * a n - 2
axiom cond2 : b 1 = 1
axiom cond3 : ∀ n, b n + 2 = b (n + 1)

-- Answers
theorem find_a_n : ∀ n, a n = 2 ^ n := by sorry

theorem find_b_n : ∀ n, b n = 2 * n - 1 := by sorry

def H_n : ℕ → ℝ
| 0     := 0
| (n+1) := ∑ i in Finset.range (n+1), 1 / (b i * b (i + 1) : ℝ)

theorem find_m : ∃ m, ∀ n : ℕ, H_n b n < m / 30 := 
  by have : 15 = 15 := rfl; exact ⟨15, by sorry⟩

def c_n (n : ℕ) := a n * b n

def T_n : ℕ → ℝ
| 0     := 0
| (n+1) := ∑ i in Finset.range (n+1), (c_n i : ℝ)

theorem find_T_n : ∀ n, T n = (2 * n - 3) * 2 ^ (n + 1) + 6 := by sorry

end find_a_n_find_b_n_find_m_find_T_n_l824_824632


namespace race_distance_l824_824728

noncomputable def total_distance_of_race (A_time B_time time_difference : ℕ) : ℕ :=
  let D := A_time * (time_difference / (A_time - B_time)) in
  D + (A_time * (time_difference % (A_time - B_time)) / (A_time - B_time))

theorem race_distance (A_time B_time : ℕ) (condition1 condition2 condition3 : ℕ) :
  A_time = 60 → B_time = 100 → condition1 = 60 → condition2 = 100 → condition3 = 160 →
  total_distance_of_race A_time B_time condition3 = 240 :=
by
  intro h1 h2 h3 h4 h5
  unfold total_distance_of_race 
  sorry

end race_distance_l824_824728


namespace estimate_undetected_typos_l824_824466

variables (a b c : ℕ)
-- a, b, c ≥ 0 are non-negative integers representing discovered errors by proofreader A, B, and common errors respectively.

theorem estimate_undetected_typos (h : c ≤ a ∧ c ≤ b) :
  ∃ n : ℕ, n = a * b / c - a - b + c :=
sorry

end estimate_undetected_typos_l824_824466


namespace number_written_on_card_l824_824406

-- Define the Euler's Totient function (for demonstration purposes, generally you would use a pre-defined library in real code)
def euler_totient (n : ℕ) : ℕ := sorry

-- Define the function g(n) which equals to Euler's Totient function
def g (n : ℕ) : ℕ := euler_totient n

theorem number_written_on_card : ∀ (n : ℕ), n ≤ 1968 → g(n) > 0 :=
by
  sorry

end number_written_on_card_l824_824406


namespace least_int_gt_sqrt_500_l824_824052

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824052


namespace cylinder_symmetry_l824_824425

-- Define the three types of cylinders
inductive CylinderType
| double_sided_bounded
| single_sided_bounded
| double_sided_unbounded

-- Conditions and properties for each type of cylinder
axiom symmetry_axis (c : CylinderType) : Prop
axiom single_plane_of_symmetry (c : CylinderType) : Prop
axiom bundle_of_planes_of_symmetry (c : CylinderType) : Prop
axiom center_of_symmetry (c : CylinderType) : Prop
axiom two_systems_of_planes_of_symmetry (c : CylinderType) : Prop

-- Specific properties based on the cylinder type
axiom double_sided_bounded_properties : 
  (symmetry_axis CylinderType.double_sided_bounded) ∧
  (single_plane_of_symmetry CylinderType.double_sided_bounded) ∧
  (bundle_of_planes_of_symmetry CylinderType.double_sided_bounded) ∧
  (center_of_symmetry CylinderType.double_sided_bounded)

axiom single_sided_bounded_properties :
  (symmetry_axis CylinderType.single_sided_bounded) ∧
  (bundle_of_planes_of_symmetry CylinderType.single_sided_bounded) ∧
  ¬(center_of_symmetry CylinderType.single_sided_bounded)

axiom double_sided_unbounded_properties :
  (symmetry_axis CylinderType.double_sided_unbounded) ∧
  (two_systems_of_planes_of_symmetry CylinderType.double_sided_unbounded) ∧
  ∀ p, ∃ a, (center_of_symmetry CylinderType.double_sided_unbounded)

-- Theorem stating the symmetry properties
theorem cylinder_symmetry (c : CylinderType) : 
  (c = CylinderType.double_sided_bounded → 
    symmetry_axis c ∧ 
    single_plane_of_symmetry c ∧ 
    bundle_of_planes_of_symmetry c ∧ 
    center_of_symmetry c) ∧ 
  (c = CylinderType.single_sided_bounded → 
    symmetry_axis c ∧ 
    bundle_of_planes_of_symmetry c ∧ 
    ¬center_of_symmetry c) ∧ 
  (c = CylinderType.double_sided_unbounded → 
    symmetry_axis c ∧ 
    two_systems_of_planes_of_symmetry c ∧ 
    ∀ p, ∃ a, center_of_symmetry c) := 
sorry

end cylinder_symmetry_l824_824425


namespace max_value_expression_l824_824623

theorem max_value_expression (K : ℕ) (a : ℝ) (k : ℕ) (r : ℕ) (k_i : ℕ) (ks : List ℕ)
  (hk : List.sum ks = k) (h_len : ks.length = r) (h_ki_nat : ∀ k_i ∈ ks, k_i ∈ ℕ) (h_pos_a : a > 0) (h_r : 1 ≤ r ∧ r ≤ k) :
  a^(k - r + 1) + (r - 1) * a 
  = List.sum (List.map (λ k_i, a^k_i) ks) := 
sorry

end max_value_expression_l824_824623


namespace find_f_one_l824_824705

noncomputable def f_inv (x : ℝ) : ℝ := 2^(x + 1)

theorem find_f_one : ∃ f : ℝ → ℝ, (∀ y, f (f_inv y) = y) ∧ f 1 = -1 :=
by
  sorry

end find_f_one_l824_824705


namespace exists_ab_l824_824249

def f : ℕ → ℕ 
| 1 := 1
| (n + 1) := 
  let max_m := 
    (List.range (n + 1)).filter (λ m, ∃ ap, (∀ i < ap.length, ap.get ⟨i, sorry⟩ > 0) ∧ 
      (∀ i < ap.length - 1, ap.get ⟨i, sorry⟩ < ap.get ⟨i + 1, sorry⟩) ∧ 
      ap.get ⟨ap.length - 1, sorry⟩ = n ∧ 
      (∀ i < ap.length, f (ap.get ⟨i, sorry⟩) = f (ap.get ⟨0, sorry⟩)))
    |> List.maximum |> Option.getD 1 in
  max_m.getD 1

theorem exists_ab (n : ℕ) : ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ ∀ n : ℕ, f (a * n + b) = n + 2 :=
  ∃ (a : ℕ) (b : ℕ), a = 4 ∧ b = 8 ∧ ∀ n : ℕ, f (4 * n + 8) = n + 2 := sorry 

end exists_ab_l824_824249


namespace valid_four_digit_numbers_count_l824_824679

-- Define the range and floor function properties.
def is_valid_four_digit_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ b = (a + d) / 2

-- Define the count of such numbers.
def count_valid_four_digit_numbers :=
  { count := ∑ a in Fin₀ 9,
               ∑ b in Fin₀ 10,
               ∑ c in Fin₀ 10,
               ∑ d in Fin₀ 10,
               if is_valid_four_digit_number a b c d then 1 else 0 }

theorem valid_four_digit_numbers_count :
  count_valid_four_digit_numbers = 900 :=
by sorry

end valid_four_digit_numbers_count_l824_824679


namespace charity_delivered_100_plates_l824_824161

variables (cost_rice_per_plate cost_chicken_per_plate total_amount_spent : ℝ)
variable (P : ℝ)

-- Conditions provided
def rice_cost : ℝ := 0.10
def chicken_cost : ℝ := 0.40
def total_spent : ℝ := 50
def total_cost_per_plate : ℝ := rice_cost + chicken_cost

-- Lean 4 statement to prove:
theorem charity_delivered_100_plates :
  total_spent = 50 →
  total_cost_per_plate = rice_cost + chicken_cost →
  rice_cost = 0.10 →
  chicken_cost = 0.40 →
  P = total_spent / total_cost_per_plate →
  P = 100 :=
by
  sorry

end charity_delivered_100_plates_l824_824161


namespace sequence_general_term_l824_824437

theorem sequence_general_term (n : ℕ) : 
  (∃ a_n : ℚ, a_n = (-1 : ℚ)^(n) * ↑(n^2) / (↑(2*n - 1)) ∧ 
  (if n % 2 = 0 then a_n ≥ 0 else a_n ≤ 0)) :=
sorry

end sequence_general_term_l824_824437


namespace geometric_progression_ratio_l824_824192

theorem geometric_progression_ratio (r : ℕ) (h : 4 + 4 * r + 4 * r^2 + 4 * r^3 = 60) : r = 2 :=
by
  sorry

end geometric_progression_ratio_l824_824192


namespace least_integer_greater_than_sqrt_500_l824_824117

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824117


namespace periods_same_ranges_not_same_in_general_odd_function_condition_max_value_of_product_l824_824661

noncomputable def f (ω : ℝ) (φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (ω : ℝ) (φ x : ℝ) : ℝ := ω * Real.cos (ω * x + φ)

theorem periods_same {ω φ : ℝ} (hω : ω > 0) :
  (∃ T > 0, ∀ x, f ω φ (x + T) = f ω φ x) ∧
  (∃ T > 0, ∀ x, g ω φ (x + T) = g ω φ x) := sorry

theorem ranges_not_same_in_general {ω φ : ℝ} (hω : ω > 0) :
  (set.range (f ω φ) ≠ set.range (g ω φ)) ∨ (ω = 1 ∧ set.range (f ω φ) = set.range (g ω φ)) := sorry

theorem odd_function_condition {ω φ : ℝ} (hω : ω > 0) :
  ∃ θ k : ℤ, y := f ω φ + g ω φ,
    Real.sin(ω + θ) + ω * Real.cos(ω + θ) = y  ∧ θ + φ = k * Real.pi := sorry

theorem max_value_of_product {ω φ : ℝ} (hω : ω > 0) :
  ¬ (∃ x, f ω φ x * g ω φ x = 1/2)  ∧ (ω > 0 ∧ ∀ x, f ω φ x * g ω φ x ≤ ω/2) := sorry

end periods_same_ranges_not_same_in_general_odd_function_condition_max_value_of_product_l824_824661


namespace least_integer_gt_sqrt_500_l824_824097

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824097


namespace discount_given_l824_824780

variables (initial_money : ℕ) (extra_fraction : ℕ) (additional_money_needed : ℕ)
variables (total_with_discount : ℕ) (discount_amount : ℕ)

def total_without_discount (initial_money : ℕ) (extra_fraction : ℕ) : ℕ :=
  initial_money + extra_fraction

def discount (initial_money : ℕ) (total_without_discount : ℕ) (total_with_discount : ℕ) : ℕ :=
  total_without_discount - total_with_discount

def discount_percentage (discount_amount : ℕ) (total_without_discount : ℕ) : ℚ :=
  (discount_amount : ℚ) / (total_without_discount : ℚ) * 100

theorem discount_given 
  (initial_money : ℕ := 500)
  (extra_fraction : ℕ := 200)
  (additional_money_needed : ℕ := 95)
  (total_without_discount₀ : ℕ := total_without_discount initial_money extra_fraction)
  (total_with_discount₀ : ℕ := initial_money + additional_money_needed)
  (discount_amount₀ : ℕ := discount initial_money total_without_discount₀ total_with_discount₀)
  : discount_percentage discount_amount₀ total_without_discount₀ = 15 :=
by sorry

end discount_given_l824_824780


namespace find_a_monotonicity_and_range_of_f_range_of_k_l824_824757

def f (a : ℝ) (x : ℝ) : ℝ := (1 / (2^x + a)) - (1 / 2)

theorem find_a (a : ℝ) : f a 0 = 0 ↔ a = 1 := by
  sorry

theorem monotonicity_and_range_of_f : 
  monotone (f 1) ∧ (∀ x : ℝ, f 1 x ∈ set.Ioo (-(1 / 2)) (1 / 2)) := by
  sorry

theorem range_of_k (k : ℝ) : 
  (∀ x ∈ set.Icc (1 : ℝ) 4, f 1 (k - 2 / x) + f 1 (2 - x) > 0) ↔ k < 2 * real.sqrt 2 - 2 := by
  sorry

end find_a_monotonicity_and_range_of_f_range_of_k_l824_824757


namespace binom_1300_2_eq_l824_824552

theorem binom_1300_2_eq : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_eq_l824_824552


namespace least_integer_greater_than_sqrt_500_l824_824080

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824080


namespace probability_no_shaded_square_l824_824154

theorem probability_no_shaded_square (n m : ℕ) (h1 : n = (2006.choose 2)) (h2 : m = 1003^2) : 
  (1 - (m / n) = (1002 / 2005)) := 
by
  -- Number of rectangles in one row
  have hn : n = 1003 * 2005 := h1
  -- Number of rectangles in one row containing a shaded square
  have hm : m = 1003 * 1003 := h2
  sorry

end probability_no_shaded_square_l824_824154


namespace line_plane_perpendicular_l824_824621

-- Definitions of parallel and perpendicular for lines and planes.
def line : Type := ℝ → ℝ → ℝ → Prop
def plane : Type := ℝ → ℝ → ℝ → Prop

def is_parallel (l1 l2 : ∀ x y z, Prop) : Prop :=
  ∀ r : ℝ, l1 r 0 0 → l2 r 0 0 -- This is a stub definition, parallelism needs a specific definition.

def is_perpendicular (l1 l2 : ∀ x y z, Prop) : Prop :=
  ∀ x y z : ℝ, l1 x y z → l2 y z x → false -- This is a stub definition, perpendicularity needs a specific definition.

variable (l : ∀ x y z, Prop)
variable (α β : ∀ x y z, Prop)

-- The assertion that we need to prove
theorem line_plane_perpendicular (H1 : is_parallel l α) (H2 : is_perpendicular l β) : is_perpendicular α β := 
sorry

end line_plane_perpendicular_l824_824621


namespace candies_in_each_pile_l824_824245

def number_of_candies_in_each_pile (c a p : ℕ) : ℕ :=
  (c - a) / p

theorem candies_in_each_pile (c a p : ℕ) (hc : c = 78) (ha : a = 30) (hp : p = 6) :
  number_of_candies_in_each_pile c a p = 8 :=
by
  have hc' : c = 78 := hc
  have ha' : a = 30 := ha
  have hp' : p = 6 := hp
  simp [number_of_candies_in_each_pile, hc', ha', hp']
  sorry

end candies_in_each_pile_l824_824245


namespace probability_no_shaded_square_l824_824149

theorem probability_no_shaded_square :
  let n := (2006 * 2005) / 2,
      m := 1003 * 1003 in
  (n - m) / n = 1002 / 2005 :=
by 
  sorry

end probability_no_shaded_square_l824_824149


namespace inscribed_circles_radii_sum_l824_824162

noncomputable def sum_of_radii (d : ℝ) (r1 r2 : ℝ) : Prop :=
  r1 + r2 = d / 2

theorem inscribed_circles_radii_sum (d : ℝ) (h : d = 23) (r1 r2 : ℝ) (h1 : r1 + r2 = d / 2) :
  r1 + r2 = 23 / 2 :=
by
  rw [h] at h1
  exact h1

end inscribed_circles_radii_sum_l824_824162


namespace find_a_l824_824658

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 4)

-- Define the circle radius
def circle_radius : ℝ := 2

-- Define the distance formula from a point to a line
def distance_point_to_line (a b c : ℝ) (x y : ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

-- The theorem statement
theorem find_a (a : ℝ) :
  (distance_point_to_line a 1 (-1) (1, 4) = 1) → a = -4/3 :=
by
  sorry

end find_a_l824_824658


namespace rotated_line_eq_l824_824800

theorem rotated_line_eq :
  ∀ (x y : ℝ), 
  (x - y + 4 = 0) ∨ (x - y - 4 = 0) ↔ 
  ∃ (x' y' : ℝ), (-x', -y') = (x, y) ∧ (x' - y' + 4 = 0) :=
by
  sorry

end rotated_line_eq_l824_824800


namespace point_M_is_fixed_area_range_l824_824197

noncomputable def parabola (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2

noncomputable def tangent_at (a : ℝ) (x_A : ℝ) : ℝ → ℝ :=
  λ x, 2 * a * x_A * (x - x_A) + a * x_A^2

noncomputable def point_B (a : ℝ) (x_A : ℝ) : ℝ × ℝ :=
  (0, -a * x_A^2)

noncomputable def point_M (a : ℝ) (x_A : ℝ) : ℝ × ℝ :=
  (x_A / 2, 0)

theorem point_M_is_fixed (a : ℝ) (m : ℝ) (x_A : ℝ) (h : 0 < a) (hm : m < 0) :
  (point_M a x_A).snd = 0 := sorry

noncomputable def area_triangle (p q : ℝ × ℝ) : ℝ :=
  1 / 2 * (p.1 * q.2 - q.1 * p.2).abs

theorem area_range (a : ℝ) (x_A : ℝ) (p q : ℝ × ℝ)
  (h : 0 < a) (hm : p.1 = 0 ∧ p.2 = 0)
  (hpq : ∀ x, (p = (x, a*x^2)) ∨ (q = (x, a*x^2))) :
  1 ≤ area_triangle p q ∧ area_triangle p q ≤ 2 * sqrt 2 + 1 := sorry

end point_M_is_fixed_area_range_l824_824197


namespace arithmetic_progression_of_nonnperfect_powers_l824_824238

/--
  There exists an arithmetic progression of 2016 natural numbers
  such that none of them is a perfect power, but their product is
  a perfect power.
-/
theorem arithmetic_progression_of_nonnperfect_powers :
  ∃ (a : ℕ) (n : ℕ), 
  n = 2016 ∧ 
  (∀ i : ℕ, i < n →¬ 
  ∃ k : ℕ, ∃ x : ℕ, k ≥ 2 ∧ x ≥ 2 ∧ i = x^k) ∧ 
  (∃ m : ℕ, m ≠ 1 ∧ 
  m = ∏ i in finset.range n, (a + i)) :=
sorry

end arithmetic_progression_of_nonnperfect_powers_l824_824238


namespace expand_product_l824_824579

theorem expand_product (x a : ℝ) : 2 * (x + (a + 2)) * (x + (a - 3)) = 2 * x^2 + (4 * a - 2) * x + 2 * a^2 - 2 * a - 12 :=
by
  sorry

end expand_product_l824_824579


namespace number_of_zeros_l824_824381

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi * x)

def is_zero (x₀ : ℝ) : Prop := f x₀ = 0

def is_valid (x₀ : ℝ) : Prop := |x₀| + f (x₀ + 1 / 2) < 33

theorem number_of_zeros :
  (∃ s : Set ℝ, s = {x₀ | is_zero x₀ ∧ is_valid x₀} ∧ s.to_finset.card = 65) :=
by
  sorry

end number_of_zeros_l824_824381


namespace frog_jump_problem_l824_824462

theorem frog_jump_problem (A B C : ℝ) (PA PB PC : ℝ) 
  (H1: PA' = (PB + PC) / 2)
  (H2: jump_distance_B = 60)
  (H3: jump_distance_B = 2 * abs ((PB - (PB + PC) / 2))) :
  third_jump_distance = 30 := sorry

end frog_jump_problem_l824_824462


namespace graph_with_6_vertices_contains_triangle_or_independent_set_l824_824412

def has_triangle_or_independent_set (G : SimpleGraph (Fin 6)) : Prop :=
  ∃ (A B C : Fin 6), G.Adj A B ∧ G.Adj B C ∧ G.Adj C A ∨ 
  ¬G.Adj A B ∧ ¬G.Adj B C ∧ ¬G.Adj C A

theorem graph_with_6_vertices_contains_triangle_or_independent_set (G : SimpleGraph (Fin 6)) :
  has_triangle_or_independent_set G :=
sorry

end graph_with_6_vertices_contains_triangle_or_independent_set_l824_824412


namespace regression_equation_correct_l824_824572

-- Define the data points for household incomes and savings
def n : ℕ := 100
def x : Fin n → ℝ := λ i, -- x values go here, assuming i is an integer index
sorry
def y : Fin n → ℝ := λ i, -- y values go here, assuming i is an integer index
sorry

-- Define the given sums
def sum_x : ℝ := 500
def sum_y : ℝ := 100
def sum_xy : ℝ := 1000
def sum_x2 : ℝ := 3750

-- Assertion for the sample means
def bar_x : ℝ := sum_x / n
def bar_y : ℝ := sum_y / n

-- Define the linear regression equation parameters
def b_hat : ℝ := (sum_xy - n * bar_x * bar_y) / (sum_x2 - n * bar_x^2)
def a_hat : ℝ := bar_y - b_hat * bar_x

-- The linear regression equation
def y_hat (x: ℝ) : ℝ := b_hat * x + a_hat

-- The minimum annual income required to achieve well-off life
def min_income_for_well_off : ℝ := 15

-- Prove that the computed slope, intercept, and minimum income are correct based on the given conditions
theorem regression_equation_correct :
  b_hat = 0.4 ∧ a_hat = -1 ∧ min_income_for_well_off = 15 := by
  sorry

end regression_equation_correct_l824_824572


namespace least_integer_greater_than_sqrt_500_l824_824108

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824108


namespace value_of_t_eq_3_over_4_l824_824716

-- Define the values x and y as per the conditions
def x (t : ℝ) : ℝ := 1 - 2 * t
def y (t : ℝ) : ℝ := 2 * t - 2

-- Statement only, proof is omitted using sorry
theorem value_of_t_eq_3_over_4 (t : ℝ) (h : x t = y t) : t = 3 / 4 :=
by
  sorry

end value_of_t_eq_3_over_4_l824_824716


namespace smallest_x_for_360_multiple_of_800_l824_824863

theorem smallest_x_for_360_multiple_of_800 :
  ∃ (x : ℕ), x > 0 ∧ (360 * x) % 800 = 0 ∧ (∀ y > 0, (360 * y) % 800 = 0 → y ≥ x) :=
begin
  use 40,
  split,
  { exact nat.succ_pos' 39 },
  split,
  { norm_num },
  { intros y hy₁ hy₂,
    have h₃ := mul_div_cancel' _ (nat.pos_of_ne_zero (ne_of_gt hy₁.symm)),
    rw [h₃, mul_comm 360 y, ←mul_assoc, nat.mul_div_cancel_left _ (nat.pos_of_ne_zero (ne_of_gt hy₁.symm)), ←mul_assoc] at hy₂,
    exact le_of_dvd hy₁.symm hy₂ }
end

end smallest_x_for_360_multiple_of_800_l824_824863


namespace least_int_gt_sqrt_500_l824_824048

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824048


namespace triangle_with_given_angles_l824_824909

noncomputable def describeTriangle (r : ℝ) (α β : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ), 
  let γ := 180 - (α + β) in
  interior_angle A B C = α ∧
  interior_angle B C A = β ∧
  interior_angle C A B = γ ∧
  circumscribes_circle (A, B, C) (center, r)

theorem triangle_with_given_angles (r : ℝ) (α β : ℝ) :
  ∃ (A B C : ℝ × ℝ),
  describeTriangle r α β :=
sorry

end triangle_with_given_angles_l824_824909


namespace train_length_360_l824_824523

variable (time_to_cross : ℝ) (speed_of_train : ℝ)

theorem train_length_360 (h1 : time_to_cross = 12) (h2 : speed_of_train = 30) :
  speed_of_train * time_to_cross = 360 :=
by
  rw [h1, h2]
  norm_num

end train_length_360_l824_824523


namespace remainder_a_pow_4_is_1_l824_824760

noncomputable def remainder_a_pow_4 (n : ℕ) (a : ℤ) : ℤ :=
  if (n > 0) ∧ ∃ (inv_a : ℤ), a * inv_a ≡ 1 [MOD n] then (a ^ 4) % n else sorry

theorem remainder_a_pow_4_is_1 (n : ℕ) (a : ℤ)
  (h1 : 0 < n)
  (h2 : (a * a) % n = 1 % n) :
  (a ^ 4) % n = 1 :=
by
  sorry

end remainder_a_pow_4_is_1_l824_824760


namespace question1_question2_case1_question2_case2_question2_case3_question3_l824_824645

-- Given conditions and function definition
def f (x : ℝ) (a : ℝ) : ℝ := 
  (x^2 / 2) + 2 * a * (a + 1) * Real.log x - (3 * a + 1) * x

-- Question 1: Prove that a = 3/2 given the tangent line condition
theorem question1 (a : ℝ) (h : a > 0) (h_tangent : deriv (λ x, f x a) 1 = 3) : a = 3 / 2 :=
sorry

-- Question 2: Prove the intervals where f(x) is strictly increasing based on the value of a
theorem question2_case1 (a : ℝ) (h : a > 1) : 
  ∃ (I1 I2 : Set ℝ), I1 = (Set.Ioo 0 (a + 1)) ∧ I2 = (Set.Ioi (2 * a)) ∧ 
  (∀ x ∈ I1, deriv (λ x, f x a) x > 0) ∧ 
  (∀ x ∈ I2, deriv (λ x, f x a) x > 0) := 
sorry

theorem question2_case2 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  ∃ (I1 I2 : Set ℝ), I1 = (Set.Ioo 0 (2 * a)) ∧ I2 = (Set.Ioi (a + 1)) ∧ 
  (∀ x ∈ I1, deriv (λ x, f x a) x > 0) ∧ 
  (∀ x ∈ I2, deriv (λ x, f x a) x > 0) := 
sorry

theorem question2_case3 (a : ℝ) (h : a = 1) : 
  ∀ x > 0, deriv (λ x, f x a) x ≥ 0 :=
sorry

-- Question 3: Under the condition of (1), find the set of b values for inequality
theorem question3 (b : ℝ) (a : ℝ) (h : a = 3 / 2) (h_inequality : ∀ x ∈ Set.Icc 1 2, f x a - b^2 - 6 * b ≥ 0) : 
  b ∈ Set.Icc (-5) (-1) :=
sorry

end question1_question2_case1_question2_case2_question2_case3_question3_l824_824645


namespace least_integer_greater_than_sqrt_500_l824_824011

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824011


namespace tetrahedron_plane_point_sets_l824_824912

-- Define the main problem statement
theorem tetrahedron_plane_point_sets (P : Fin 10 → ℝ × ℝ) 
    (tetrahedron_conditions : ∀ i, i < 10 → ∃ a b c d : ℝ × ℝ, 
        P 0 = a ∧ (∀ j, 1 ≤ j → j < 10 → 
        (P j = midpoint a b ∨ P j = midpoint b c ∨ P j = midpoint c d ∨ P j = midpoint d a ∨ 
        P j = midpoint a c ∨ P j = midpoint b d))) :
  ∃ sets_on_plane : Finset (Finset (Fin 10)), 
    sets_on_plane.card = 33 ∧ 
     ∀ s ∈ sets_on_plane, ∀ i j k ∈ s, i ≠ j ∧ i ≠ k ∧ j ≠ k :=
sorry

end tetrahedron_plane_point_sets_l824_824912


namespace no_solution_inequalities_l824_824713

theorem no_solution_inequalities (a : ℝ) : (¬ ∃ x : ℝ, 2 * x - 4 > 0 ∧ x - a < 0) → a ≤ 2 := 
by 
  sorry

end no_solution_inequalities_l824_824713


namespace skew_lines_common_perpendicular_distance_between_skew_lines_l824_824259

-- Define the cube with edge length a
variables {a : ℝ} (A B C D A₁ B₁ C₁ D₁ : Type)

-- Assume A, B, C, D, A₁, B₁, C₁, D₁ form a cube with edge length a
-- Define the positions of points and the lines as follows:
def is_cube (A B C D A₁ B₁ C₁ D₁ : Type) (a : ℝ) : Prop :=
  -- Define other conditions of being a cube, such as orthogonal edges of length a
  sorry

-- Define the lines A A₁ and B C
def line_AA₁ (A A₁ : Type) : Type := sorry
def line_BC (B C : Type) : Type := sorry

-- Prove that A A₁ and B C are skew
theorem skew_lines (h : is_cube A B C D A₁ B₁ C₁ D₁ a) : ¬ is_parallel (line_AA₁ A A₁) (line_BC B C) ∧ ¬ is_intersecting (line_AA₁ A A₁) (line_BC B C) :=
  sorry

-- Define the common perpendicular line and prove it
theorem common_perpendicular (h : is_cube A B C D A₁ B₁ C₁ D₁ a) : is_perpendicular (line_AA₁ A A₁) (line_AB A B) ∧ is_perpendicular (line_BC B C) (line_AB A B) :=
  sorry

-- Prove the distance between the skew lines
theorem distance_between_skew_lines (h : is_cube A B C D A₁ B₁ C₁ D₁ a) : distance (line_AA₁ A A₁) (line_BC B C) = a :=
  sorry

end skew_lines_common_perpendicular_distance_between_skew_lines_l824_824259


namespace inequality_imply_positive_a_l824_824295

theorem inequality_imply_positive_a 
  (a b d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h_d_pos : d > 0) 
  (h : a / b > -3 / (2 * d)) : a > 0 :=
sorry

end inequality_imply_positive_a_l824_824295


namespace sqrt_500_least_integer_l824_824014

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824014


namespace least_integer_gt_sqrt_500_l824_824100

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824100


namespace sum_of_solutions_eq_3_l824_824961

theorem sum_of_solutions_eq_3 (x y : ℝ) (h1 : x * y = 1) (h2 : x + y = 3) :
  x + y = 3 := sorry

end sum_of_solutions_eq_3_l824_824961


namespace sum_of_fractions_l824_824210

theorem sum_of_fractions :
  (1/15 + 2/15 + 3/15 + 4/15 + 5/15 + 6/15 + 7/15 + 8/15 + 9/15 + 46/15) = (91/15) := by
  sorry

end sum_of_fractions_l824_824210


namespace part1_part2_l824_824278

-- Define the parabola and the condition point M lies on it
structure Parabola1 (p : ℝ) where
  x2_eq_2py : ∀ (x y : ℝ), x^2 = 2 * p * y

structure Point (x y : ℝ) where
  coords : (x, y)

-- Two tangents from point P intersect this parabola at points A and B
structure Tangents (P : Point) (A B : Point) where
  on_parabola : Parabola1 2 → A.coords.1^2 = 2 * 2 * A.coords.2
  tangents_slope_product : (A.coords.1 - P.coords.1) / (A.coords.2 - P.coords.2) * (B.coords.1 - P.coords.1) / (B.coords.2 - P.coords.2) = -2

-- statement for part 1
theorem part1 (M : Point) (P : Point) (A B : Point) (h : Tangents P A B) :
  ∃ k b : ℝ, k * 0 + b = 2 :=
sorry

-- statement for part 2
theorem part2 (P : Point) (A B C D : Point) (h : Tangents P A B)  :
  ¬(∃ (A C P D : Point), concyclic A C P D) :=
sorry

end part1_part2_l824_824278


namespace range_of_f_on_interval_l824_824654

theorem range_of_f_on_interval :
  ∃ f : ℝ → ℝ, strict_mono f ∧ (∀ x > 0, f (f x - x - real.logb 2 x) = 5) ∧ 
  (∀ x ∈ set.Icc (1:ℝ) 8, f x ∈ set.Icc (3:ℝ) 13) := by
  sorry

end range_of_f_on_interval_l824_824654


namespace jason_initial_cards_l824_824748

-- Definitions based on conditions
def cards_given_away : ℕ := 4
def cards_left : ℕ := 5

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 9 :=
by sorry

end jason_initial_cards_l824_824748


namespace solve_question_l824_824549

noncomputable def question := real.sqrt 2 * (real.pow 4 (1 / 3)) * (real.root 6 32) + (real.log10 (1 / 100)) - (real.pow 3 (real.logb 3 2))

theorem solve_question : question = 0 := 
by
  sorry

end solve_question_l824_824549


namespace has_inverse_l824_824305

section
  /-- Definitions of the given conditions for each function / graph -/

  def is_linear (f : ℝ → ℝ) : Prop :=
    ∀ x y, f (x + y) = f x + f y ∧ f (x * y) = f x * f y

  def is_sinusoidal (g : ℝ → ℝ) : Prop :=
    ∃ a b, ∀ x, g x = a * sin (b * x) + 1

  def is_exponential (h : ℝ → ℝ) : Prop :=
    ∃ a b, ∀ x, h x = a * exp (b * x)

  def is_bijective_linear (i : ℝ → ℝ) : Prop :=
    is_linear i ∧ function.bijective i

  /-- Main proof problem statement: Prove that functions F, H, and I have inverses given the conditions. -/

  theorem has_inverse (F H I : ℝ → ℝ) (G : ℝ → ℝ)
    (hf : is_linear F)
    (hg : is_sinusoidal G)
    (hh : is_exponential H)
    (hi : is_bijective_linear I) :
    (∃ F_inv : ℝ → ℝ, function.left_inverse F_inv F ∧ function.right_inverse F_inv F) ∧
    (∃ H_inv : ℝ → ℝ, function.left_inverse H_inv H ∧ function.right_inverse H_inv H) ∧
    (∃ I_inv : ℝ → ℝ, function.left_inverse I_inv I ∧ function.right_inverse I_inv I) :=
  sorry

end

end has_inverse_l824_824305


namespace right_triangle_area_incircle_l824_824446

theorem right_triangle_area_incircle
  (c : ℝ)
  (h : c > 0)
  (x : ℝ)
  (hx : x = c / 13) :
  let a := 4 * x in
  let b := 9 * x in
  a * b / 2 = 36 / 169 * c^2 :=
by
  sorry

end right_triangle_area_incircle_l824_824446


namespace function_periodic_five_l824_824707

theorem function_periodic_five (f : ℝ → ℝ) (h_periodic : ∀ x, f(x + 6) = f(x)) (h_neg_one : f(-1) = 1) : f(5) = 1 :=
by
  sorry

end function_periodic_five_l824_824707


namespace average_speed_comparison_l824_824550

variables (D t : ℝ) (u v w : ℝ) (xu xv xw yu yv yw : ℝ)

-- Define the speeds for car A and car B
def car_a_speed (u v w : ℝ) : ℝ := 3 / (1/u + 1/v + 1/w)
def car_b_speed (u v w : ℝ) : ℝ := (u + v + w) / 3

-- The proof statement to be proved
theorem average_speed_comparison (u v w : ℝ) : car_a_speed u v w ≤ car_b_speed u v w :=
sorry

end average_speed_comparison_l824_824550


namespace geometric_sum_first_10_terms_l824_824741

theorem geometric_sum_first_10_terms :
  ∃ (a : ℕ → ℚ), 
  a 1 = 1 / 4 ∧ 
  (a 3) * (a 5) = 4 * (a 4 - 1) ∧
  (finset.range 10).sum (λ n, a (n + 1)) = 1023 / 4 :=
by
  sorry

end geometric_sum_first_10_terms_l824_824741


namespace least_integer_gt_sqrt_500_l824_824104

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824104


namespace least_integer_greater_than_sqrt_500_l824_824087

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824087


namespace solve_equation_l824_824420

theorem solve_equation : ∃ x : ℝ, 3 * x + 2 * (x - 2) = 6 ↔ x = 2 :=
by
  sorry

end solve_equation_l824_824420


namespace domain_of_function_range_of_a_range_of_x_l824_824998

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log (x - 3) / log 3 + log (a - x) / log 3

theorem domain_of_function (a : ℝ) (ha : a > 3) :
  set_of (λ x, 3 < x ∧ x < a) = {x | 3 < x ∧ x < a} :=
by
  sorry

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h₀ : monotone_on f {x | 3 < x ∧ x ≤ 6}) :
  a ≥ 9 :=
by
  sorry

theorem range_of_x (a : ℝ) (h₀ : a = 9) (f : ℝ → ℝ) (x : ℝ)
  (h₁ : f (2 * x - 1) > f 4) :
  5/2 < x ∧ x < 9/2 :=
by
  sorry

end domain_of_function_range_of_a_range_of_x_l824_824998


namespace least_integer_greater_than_sqrt_500_l824_824044

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824044


namespace correct_operation_l824_824866

theorem correct_operation : 
(2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5 * Real.sqrt 5) ∧
(Real.sqrt 25 ≠ ±5) ∧
(Real.sqrt 15 / Real.sqrt 5 ≠ 3) ∧
(5 ^ (-2) = 1 / 25) :=
by
  sorry

end correct_operation_l824_824866


namespace balls_in_boxes_l824_824311

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 7 → boxes = 4 → 
  (number_of_unique_distributions balls boxes = 11) := by
  intros balls boxes hb hc
  subst hb
  subst hc
  sorry

end balls_in_boxes_l824_824311


namespace andrew_paid_to_shopkeeper_l824_824134

def cost_of_grapes (rate_per_kg_g : ℕ) (weight_g : ℕ) : ℕ :=
  rate_per_kg_g * weight_g

def cost_of_mangoes (rate_per_kg_m : ℕ) (weight_m : ℕ) : ℕ :=
  rate_per_kg_m * weight_m

def total_amount_paid (rate_per_kg_g : ℕ) (weight_g : ℕ)
                      (rate_per_kg_m : ℕ) (weight_m : ℕ) : ℕ :=
  (cost_of_grapes rate_per_kg_g weight_g) + (cost_of_mangoes rate_per_kg_m weight_m)

theorem andrew_paid_to_shopkeeper :
  total_amount_paid 68 7 48 9 = 908 :=
by
  simp [total_amount_paid, cost_of_grapes, cost_of_mangoes]
  exact add_assoc 476 432 0
  simp
  sorry

end andrew_paid_to_shopkeeper_l824_824134


namespace evaluate_fraction_l824_824952

theorem evaluate_fraction : 3 / (2 - 3 / 4) = 12 / 5 := by
  sorry

end evaluate_fraction_l824_824952


namespace daragh_sisters_count_l824_824227

theorem daragh_sisters_count (initial_bears : ℕ) (favorite_bears : ℕ) (eden_initial_bears : ℕ) (eden_total_bears : ℕ) 
    (remaining_bears := initial_bears - favorite_bears)
    (eden_received_bears := eden_total_bears - eden_initial_bears)
    (bears_per_sister := eden_received_bears) :
    initial_bears = 20 → favorite_bears = 8 → eden_initial_bears = 10 → eden_total_bears = 14 → 
    remaining_bears / bears_per_sister = 3 := 
by
  sorry

end daragh_sisters_count_l824_824227


namespace ordered_triple_exists_l824_824386

theorem ordered_triple_exists (a b c : ℝ) (h₁ : 4 < a) (h₂ : 4 < b) (h₃ : 4 < c)
  (h₄ : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  (a, b, c) = (12, 10, 8) :=
sorry

end ordered_triple_exists_l824_824386


namespace length_of_parallel_at_60N_l824_824833

noncomputable def parallel_length (R : ℝ) (lat_deg : ℝ) : ℝ :=
  2 * Real.pi * R * Real.cos (Real.pi * lat_deg / 180)

theorem length_of_parallel_at_60N :
  parallel_length 20 60 = 20 * Real.pi :=
by
  sorry

end length_of_parallel_at_60N_l824_824833


namespace limit_of_diff_at_l824_824990

variable {ℝ : Type*} [DifferentiableField ℝ]

theorem limit_of_diff_at 
  (f : ℝ → ℝ) 
  (x₀ : ℝ) 
  (h_diff : DifferentiableAt ℝ f x₀) 
  : (lim (λ x, (f(x₀ + x) - f(x₀ - 3 * x)) / x) (nhds 0)) = 4 * deriv f x₀ :=
sorry

end limit_of_diff_at_l824_824990


namespace wrapping_paper_area_l824_824890

theorem wrapping_paper_area (l w : ℝ) (h : ℝ) (h_eq : h = w / 2) : 
  let area : ℝ := (7 * l * w) / 2
  in area = (7 * l * w) / 2 := 
by sorry

end wrapping_paper_area_l824_824890


namespace problem_probability_not_distinct_positive_real_roots_l824_824268

def range := [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]

noncomputable def probability_no_distinct_positive_real_roots : ℚ :=
  let total_pairs := (range.length : ℚ) * (range.length : ℚ)
  let valid_pairs := (do
    let b in range
    let c in range
    if b < 0 ∧ c > 0 ∧ b^2 > 4 * c then pure (b, c) else []
  ).length : ℚ
  let non_valid_pairs := total_pairs - valid_pairs
  non_valid_pairs / total_pairs

theorem problem_probability_not_distinct_positive_real_roots :
  probability_no_distinct_positive_real_roots = 13 / 15 :=
sorry

end problem_probability_not_distinct_positive_real_roots_l824_824268


namespace tan_ratio_l824_824765

open Real

variable (x y : ℝ)

-- Conditions
def sin_add_eq : sin (x + y) = 5 / 8 := sorry
def sin_sub_eq : sin (x - y) = 1 / 4 := sorry

-- Proof problem statement
theorem tan_ratio : sin (x + y) = 5 / 8 → sin (x - y) = 1 / 4 → (tan x) / (tan y) = 2 := sorry

end tan_ratio_l824_824765


namespace gcd_gx_x_eq_one_l824_824275

   variable (x : ℤ)
   variable (hx : ∃ k : ℤ, x = 34567 * k)

   def g (x : ℤ) : ℤ := (3 * x + 4) * (8 * x + 3) * (15 * x + 11) * (x + 15)

   theorem gcd_gx_x_eq_one : Int.gcd (g x) x = 1 :=
   by 
     sorry
   
end gcd_gx_x_eq_one_l824_824275


namespace max_combination_value_l824_824272

theorem max_combination_value (a b c d : ℚ) (h1 : a = 3) (h2 : b = -7) (h3 : c = -10) (h4 : d = 12) :
  a - b - c + d = 32 :=
by
  rw [h1, h2, h3, h4]
  simp
  norm_num
  sorry

end max_combination_value_l824_824272


namespace matrix_vector_multiplication_correct_l824_824756

variables {N : Matrix (Fin 2) (Fin 2) ℝ}
variables (v1 v2 v3 : Fin 2 → ℝ)

def vector1 := ![1, -2]
def vector2 := ![-2, 3]
def vector3 := ![4, -1]

def result1 := ![2, 1]
def result2 := ![0, -2]
def expected_result := ![-20, 4]

theorem matrix_vector_multiplication_correct 
  (h1 : N.mul_vec vector1 = result1) 
  (h2 : N.mul_vec vector2 = result2) : 
  N.mul_vec vector3 = expected_result :=
sorry

end matrix_vector_multiplication_correct_l824_824756


namespace complex_quadrant_l824_824694

theorem complex_quadrant (θ : ℝ) (h_sec_quad : θ ∈ Icc π (3 * π / 2))
  (h_sin : Real.sin θ > 0) (h_cos : Real.cos θ < 0) (h_tan : Real.tan θ < 0) :
  ((Real.sin θ - Real.cos θ) + (Real.tan θ - 2017) * Complex.i).im < 0 ∧ 
  ((Real.sin θ - Real.cos θ) + (Real.tan θ - 2017) * Complex.i).re > 0 :=
by
  sorry

end complex_quadrant_l824_824694


namespace binomial_1300_2_eq_844350_l824_824558

theorem binomial_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 := 
by
  sorry

end binomial_1300_2_eq_844350_l824_824558


namespace greatest_sum_consecutive_integers_l824_824469

theorem greatest_sum_consecutive_integers (n : ℤ) (h : n * (n + 1) < 360) : n + (n + 1) ≤ 37 := by
  sorry

end greatest_sum_consecutive_integers_l824_824469


namespace find_crossed_out_digit_l824_824142

theorem find_crossed_out_digit (n : ℕ) (h_rev : ∀ (k : ℕ), k < n → k % 9 = 0) (remaining_sum : ℕ) 
  (crossed_sum : ℕ) (h_sum : remaining_sum + crossed_sum = 27) : 
  crossed_sum = 8 :=
by
  -- We can incorporate generating the value from digit sum here.
  sorry

end find_crossed_out_digit_l824_824142


namespace can_color_board_l824_824736

theorem can_color_board : ∃ (m n : ℕ), m = 4 ∧ n = 8 ∧
  (∀ i j, ((i + j) % 2 = 0 → board_color i j = Color.white) ∧ 
          ((i + j) % 2 = 1 → board_color i j = Color.black)) :=
by
  let m := 4
  let n := 8
  use m, n
  split
  { exact rfl }
  split
  { exact rfl }
  intros i j
  split
  { intro h
    sorry }
  { intro h
    sorry }

end can_color_board_l824_824736


namespace yards_green_correct_l824_824463

-- Define the conditions
def total_yards_silk := 111421
def yards_pink := 49500

-- Define the question as a theorem statement
theorem yards_green_correct :
  (total_yards_silk - yards_pink = 61921) :=
by
  sorry

end yards_green_correct_l824_824463


namespace domain_of_g_l824_824935

noncomputable def g (x : ℝ) : ℝ := Real.cot (Real.arcsin (x ^ 3))

theorem domain_of_g :
  {x : ℝ | (x > -1) ∧ (x < 0) ∨ (x > 0) ∧ (x ≤ 1)} =
  {x : ℝ | -1 < x ∧ x < 0 ∨ 0 < x ∧ x ≤ 1} :=
by {
  sorry
}

end domain_of_g_l824_824935


namespace minimum_linear_feet_of_framing_l824_824157

theorem minimum_linear_feet_of_framing 
  (original_width : ℕ) (original_height : ℕ) 
  (enlargement_factor : ℕ) (border_width : ℕ)
  (framing_increment : ℕ) :
  original_width = 5 →
  original_height = 7 →
  enlargement_factor = 2 →
  border_width = 3 →
  framing_increment = 12 →
  let enlarged_width := original_width * enlargement_factor,
      enlarged_height := original_height * enlargement_factor,
      final_width := enlarged_width + 2 * border_width,
      final_height := enlarged_height + 2 * border_width,
      perimeter := 2 * final_width + 2 * final_height in
  perimeter / framing_increment = 6 :=
by
  -- Sorry is used here to skip the proof.
  sorry

end minimum_linear_feet_of_framing_l824_824157


namespace ratio_perimeter_l824_824341

-- Define the necessary components of the problem
variables (D E F G J : Point) -- Points definition
variable (ω' : Circle) -- Circle definition

-- Conditions given in the problem
axiom h_triangle : right_triangle D E F
axiom h_hypotenuse : hypotenuse D E F DE
axiom h_DF : DF = 9
axiom h_EF : EF = 40
axiom h_altitude : altitude G DE FG
axiom h_diameter : diameter G DE ω'
axiom h_tangents : is_tangent J ω' DJ ∧ is_tangent J ω' EJ

-- Define lengths and calculations
noncomputable def DE := sqrt (DF ^ 2 + EF ^ 2)
noncomputable def FG := (DF * EF) / DE
noncomputable def x := FG / 2

-- Prove that the ratio is as specified
theorem ratio_perimeter (h_triangle : right_triangle D E F) (h_hypotenuse : hypotenuse D E F DE) (h_DF : DF = 9) (h_EF : EF = 40) (h_altitude : altitude G DE FG) (h_diameter : diameter G DE ω') (h_tangents : is_tangent J ω' DJ ∧ is_tangent J ω' EJ) :
  ((DE + 2 * x) / DE) = (49 / 41) :=
sorry

end ratio_perimeter_l824_824341


namespace Morse_code_distinct_symbols_l824_824721

theorem Morse_code_distinct_symbols : 
  (∑ n in {1, 2, 3, 4, 5}, 2^n) = 62 :=
by
  sorry

end Morse_code_distinct_symbols_l824_824721


namespace min_n_satisfies_inequality_l824_824927

theorem min_n_satisfies_inequality :
  ∃ n : ℕ, 0 < n ∧ -3 * (n : ℤ) ^ 4 + 5 * (n : ℤ) ^ 2 - 199 < 0 ∧ (∀ m : ℕ, 0 < m ∧ -3 * (m : ℤ) ^ 4 + 5 * (m : ℤ) ^ 2 - 199 < 0 → 2 ≤ m) := 
  sorry

end min_n_satisfies_inequality_l824_824927


namespace new_car_distance_l824_824176

-- State the given conditions as a Lemma in Lean 4
theorem new_car_distance (d_old : ℝ) (d_new : ℝ) (h1 : d_old = 150) (h2 : d_new = d_old * 1.30) : d_new = 195 := 
by
  sorry

end new_car_distance_l824_824176


namespace quadratic_condition_l824_824585

theorem quadratic_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ x1^2 - x2^2 = c^2 / a^2) ↔
  b^4 - c^4 = 4 * a * b^2 * c :=
sorry

end quadratic_condition_l824_824585


namespace correct_statement_l824_824521

def heights : List ℝ := [10, 20, 30, 40, 50, 60, 70]
def times : List ℝ := [4.23, 3.00, 2.45, 2.13, 1.89, 1.71, 1.59]

theorem correct_statement :
  ∀ (h : ℝ) (i : ℕ), h ∈ heights → i < heights.length →
  let speed := heights[i] / times[i] in
  (∀ j < i, heights[j] < heights[i] ∧ times[j] > times[i] → speed > heights[j] / times[j]) :=
by
  sorry

end correct_statement_l824_824521


namespace incorrect_interpretations_count_l824_824615

theorem incorrect_interpretations_count : 
  (let not_opposite := ¬ (-( -8) = 8);
       not_product := ¬ (-( -8) = (-1) * (-8));
       not_abs_val := ¬ (-( -8) = abs (-8));
       inc_count := (if not_opposite then 1 else 0) +
                    (if not_product then 1 else 0) +
                    (if not_abs_val then 1 else 0) in
   inc_count = 1) :=
by
  sorry

end incorrect_interpretations_count_l824_824615


namespace solution_set_l824_824292

def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

theorem solution_set (x : ℝ) : (f (Real.log x) + f (-Real.log x) < 2 * f 1) ↔ x ∈ Set.Ioc (1 / Real.exp 1) (Real.exp 1) := 
sorry

end solution_set_l824_824292


namespace valid_votes_received_by_B_l824_824136

variable (V : ℕ) (A_votes B_votes : ℕ)
variable (valid_votes : ℕ := 0.80 * V)

theorem valid_votes_received_by_B (h1 : V = 5720)
    (h2 : valid_votes = 0.80 * V)
    (h3 : A_votes = B_votes + 0.15 * V)
    (h4 : A_votes + B_votes = valid_votes) : B_votes = 1859 := by
  sorry

end valid_votes_received_by_B_l824_824136


namespace subset_sum_n_l824_824840

def exists_subset_sum_to_n (n : ℕ) (k : ℕ) (weights : Fin k → ℕ) : Prop :=
    (∀ i, weights i < n) ∧ (∑ i, weights i < 2 * n) ∧ ∃ subset : Finset (Fin k), ∑ i in subset, weights i = n

theorem subset_sum_n (n : ℕ) (k : ℕ) (weights : Fin k → ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ i, weights i < n) (h3 : ∑ i, weights i < 2 * n) : 
  ∃ subset : Finset (Fin k), ∑ i in subset, weights i = n := 
begin
  sorry
end

end subset_sum_n_l824_824840


namespace product_of_terms_geometric_sequence_l824_824644

variable {a : ℕ → ℝ}
variable {q : ℝ}
noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem product_of_terms_geometric_sequence
  (ha: geometric_sequence a q)
  (h3_4: a 3 * a 4 = 6) :
  a 2 * a 5 = 6 :=
by
  sorry

end product_of_terms_geometric_sequence_l824_824644


namespace ab_necessary_but_not_sufficient_l824_824378

theorem ab_necessary_but_not_sufficient (a b : ℝ) (i : ℂ) (hi : i^2 = -1) : 
  ab < 0 → ¬ (ab >= 0) ∧ (¬ (ab <= 0)) → (z = i * (a + b * i)) ∧ a > 0 ∧ -b > 0 := 
  sorry

end ab_necessary_but_not_sufficient_l824_824378


namespace range_of_k_l824_824434

theorem range_of_k (k : ℝ) : (2 > 0) ∧ (k > 0) ∧ (k < 2) ↔ (0 < k ∧ k < 2) :=
by
  sorry

end range_of_k_l824_824434


namespace least_integer_greater_than_sqrt_500_l824_824095

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824095


namespace circle_equation_is_correct_l824_824257

def center := (4 : ℝ, 7 : ℝ)
def chord_intercepted_by_line (line : ℝ × ℝ → Prop) (C : ℝ × ℝ) (chord_length : ℝ) : Prop :=
  line (4, 7) ∧ chord_length = 8

noncomputable def equation_of_the_circle (C : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ x y, (x - C.1)^2 + (y - C.2)^2 = radius^2

theorem circle_equation_is_correct (line_eq : ℝ × ℝ → Prop) (chord_length : ℝ) :
  chord_intercepted_by_line line_eq center chord_length →
  equation_of_the_circle center 5 :=
by {
  intro h,
  sorry
}

end circle_equation_is_correct_l824_824257


namespace four_dozen_cost_l824_824540

-- Defining the conditions
def cost_of_two_dozen (c : ℝ) : Prop := c = 15.60
def cost_per_dozen (c : ℝ) : ℝ := c / 2

-- The statement that we need to prove, given the conditions.
theorem four_dozen_cost (c : ℝ) (h : cost_of_two_dozen c) : 4 * (cost_per_dozen c) = 31.20 :=
by
  sorry

end four_dozen_cost_l824_824540


namespace volleyball_team_girls_l824_824902

theorem volleyball_team_girls (B G : ℕ) (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 20) : G = 15 :=
sorry

end volleyball_team_girls_l824_824902


namespace hyperbola_s2_l824_824169

theorem hyperbola_s2 (s : ℝ) :
  ∃ (b^2 : ℝ), 
  ∀ (x y : ℝ), 
  (x = 1 ∧ y = 3 ∧ (x^2 / 25 - y^2 / b^2 = 1)) ∧ 
  (x = 5 ∧ y = 0 ∧ (x^2 / 25 - y^2 / b^2 = 1)) ∧ 
  (x = s ∧ y = -3 ∧ (x^2 / 25 - y^2 / b^2 = 1)) → 
  s^2 = 49 :=
by
  sorry

end hyperbola_s2_l824_824169


namespace coeff_x9_equals_num_ways_l824_824411

open BigOperators

def coefficient_x9 (f : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Multiset.range (n+1)).powerset.sum (λ s, if s.sum = n then 1 else 0)

theorem coeff_x9_equals_num_ways :
  coefficient_x9 (λ i, 1) 9 = 
    ∑ (s : Multiset ℕ) in (Multiset.range 12).powerset, if s.sum = 9 then 1 else 0 :=
sorry

end coeff_x9_equals_num_ways_l824_824411


namespace lying_people_count_l824_824942

-- Definitions for the statements made by A, B, C, and D
def A_statement := sorry -- Place appropriate formulas here
def B_statement := sorry -- Place appropriate formulas here
def C_statement := sorry -- Place appropriate formulas here
def D_statement := sorry -- Place appropriate formulas here

theorem lying_people_count : 
(1 lie ∧ B ∧ C ∧ D) → false :=
sorry


end lying_people_count_l824_824942


namespace new_car_travel_distance_l824_824175

-- Define the distance traveled by the older car
def distance_older_car : ℝ := 150

-- Define the percentage increase
def percentage_increase : ℝ := 0.30

-- Define the condition for the newer car's travel distance
def distance_newer_car (d_old : ℝ) (perc_inc : ℝ) : ℝ :=
  d_old * (1 + perc_inc)

-- Prove the main statement
theorem new_car_travel_distance :
  distance_newer_car distance_older_car percentage_increase = 195 := by
  -- Skip the proof body as instructed
  sorry

end new_car_travel_distance_l824_824175


namespace inequality_proof_equality_case_l824_824646

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 ≥ 6 * Real.sqrt 3 :=
begin
  sorry
end

theorem equality_case (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 = 6 * Real.sqrt 3 ↔ a = b ∧ b = c ∧ c = Real.sqrt (Real.sqrt 3) :=
begin
  sorry
end

end inequality_proof_equality_case_l824_824646


namespace Carl_avg_gift_bags_l824_824924

theorem Carl_avg_gift_bags :
  ∀ (known expected extravagant remaining : ℕ), 
  known = 50 →
  expected = 40 →
  extravagant = 10 →
  remaining = 60 →
  (known + expected) - extravagant - remaining = 30 := by
  intros
  sorry

end Carl_avg_gift_bags_l824_824924


namespace polyhedron_floating_l824_824163

theorem polyhedron_floating (P : Polyhedron) (h_convex : Convex P) (V : Volume P) (A : SurfaceArea P) :
  (exists (V_below : Volume P) (A_above : SurfaceArea P), 
    V_below = 0.9 * V ∧ A_above > 0.5 * A) :=
sorry

end polyhedron_floating_l824_824163


namespace inequality_abc_l824_824405

theorem inequality_abc (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a :=
by
  sorry

end inequality_abc_l824_824405


namespace percentage_of_Republicans_voting_for_X_l824_824328

theorem percentage_of_Republicans_voting_for_X
  (x : ℝ)
  (R : ℝ)
  (D : ℝ)
  (Rep_votes : R = 3 * x)
  (Dem_votes : D = 2 * x)
  (Dem_percent_for_X : ℝ := 15)
  (X_win_margin : ℝ := 20.000000000000004) :
  let total_population := R + D
  let expected_X_percentage := (50 + X_win_margin / 2) / 100 in
  let vote_for_X_from_Reps := R * alpha / 100 in
  let vote_for_X_from_Dems := D * Dem_percent_for_X / 100 in
  let total_votes_for_X := vote_for_X_from_Reps + vote_for_X_from_Dems in
  let expected_percentage_for_X := total_votes_for_X / total_population in
  expected_percentage_for_X = expected_X_percentage → alpha = 90
:=
sorry

end percentage_of_Republicans_voting_for_X_l824_824328


namespace no_obtuse_isosceles_triangle_l824_824511

-- Definitions of angles and properties of the quadrilateral
variables (Q : Type)
  [quadrilateral : Q]
  [angle_A : angle Q 90]
  [angle_B : angle Q 120]
  [equal_diagonals : diagonals_equal Q]
  [diagonals_right_angle : diagonals_intersect_at_right_angle Q]

-- Theorem statement
theorem no_obtuse_isosceles_triangle (quadrilateral : Q) 
  [angle_A : angle Q 90]
  [angle_B : angle Q 120]
  [equal_diagonals : diagonals_equal Q]
  [diagonals_right_angle : diagonals_intersect_at_right_angle Q]
  : ¬ exists (T : Type) [triangle T] [obtuse_isosceles T], 
    T ∈ quadrilateral_division quadrilateral :=
sorry

end no_obtuse_isosceles_triangle_l824_824511


namespace tomatoes_first_shipment_l824_824778

theorem tomatoes_first_shipment :
  ∃ X : ℕ, 
    (∀Y : ℕ, 
      (Y = 300) → -- Saturday sale
      (X - Y = X - 300) ∧
      (∀Z : ℕ, 
        (Z = 200) → -- Sunday rotting
        (X - 300 - Z = X - 500) ∧
        (∀W : ℕ, 
          (W = 2 * X) → -- Monday new shipment
          (X - 500 + W = 2500) →
          (X = 1000)
        )
      )
    ) :=
by
  sorry

end tomatoes_first_shipment_l824_824778


namespace least_integer_gt_sqrt_500_l824_824106

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824106


namespace sine_addition_l824_824235

noncomputable def sin_inv_45 := Real.arcsin (4 / 5)
noncomputable def tan_inv_12 := Real.arctan (1 / 2)

theorem sine_addition :
  Real.sin (sin_inv_45 + tan_inv_12) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sine_addition_l824_824235


namespace searchlight_probability_l824_824871

/-- 
Given that a searchlight makes 2 revolutions per minute,
we need to prove that the probability that a man appearing
near the tower will stay in the dark for at least 10 seconds
is 1/3.
-/
theorem searchlight_probability :
  (∃ r : ℝ, r = 2) → (∃ p : ℝ, p = 1/3) :=
by
  intros h
  use 1/3
  -- Proof steps omitted
  sorry

end searchlight_probability_l824_824871


namespace quadratic_polynomial_with_root_and_coefficient_l824_824603

theorem quadratic_polynomial_with_root_and_coefficient :
  ∃ (a b c : ℝ), (a = 3) ∧ (4 + 2 * (1 : ℝ) * complex.I = 4 + 2i) ∧
  (∀ (x : ℂ), (x = 4 + 2i) ∨ (x = 4 - 2i) → 3 * (x - 4 - 2i) * (x - 4 + 2i) = 3 * x^2 - 24 * x + 60) :=
by
  sorry

end quadratic_polynomial_with_root_and_coefficient_l824_824603


namespace complete_the_square_1_complete_the_square_2_complete_the_square_3_l824_824926

theorem complete_the_square_1 (x : ℝ) : 
  (x^2 - 2 * x + 3) = (x - 1)^2 + 2 :=
sorry

theorem complete_the_square_2 (x : ℝ) : 
  (3 * x^2 + 6 * x - 1) = 3 * (x + 1)^2 - 4 :=
sorry

theorem complete_the_square_3 (x : ℝ) : 
  (-2 * x^2 + 3 * x - 2) = -2 * (x - 3 / 4)^2 - 7 / 8 :=
sorry

end complete_the_square_1_complete_the_square_2_complete_the_square_3_l824_824926


namespace line_plane_relationship_l824_824371

variables {Point : Type} [affine_space V Point V]
variables {a b : set Point} -- Lines
variables {α : set Point} -- Plane

def parallel (a b : set Point) : Prop := ∀ p1 ∈ a, ∀ p2 ∈ b, ∀ d ∈ (a - (b ∩ a)), ∃ k : ℝ, p1 + k • d = p2
def perpendicular (a : set Point) (α : set Point) : Prop := ∀ p1 ∈ a, ∀ n ∈ α,  ⟪p1, n⟫ = (0 : ℝ)

theorem line_plane_relationship (a b : set Point) (α : set Point)
    (diff_lines : a ≠ b) (diff_planes : ∃ p q : Point, p ∈ α ∧ q ∈ α ∧ p ≠ q)
    (h1 : parallel a b) (h2 : perpendicular a α) : perpendicular b α := 
by 
  sorry

end line_plane_relationship_l824_824371


namespace sticker_distribution_l824_824306

theorem sticker_distribution :
  ∃! (x : ℕ → ℕ) (h : (x 1) + (x 2) + (x 3) + (x 4) + (x 5) = 10),
  (finset.range 5).sum (λ i, x i) = 10 ∧ 
  nat.choose (10 + 5 - 1) (5 - 1) = 1001 := 
by
  sorry

end sticker_distribution_l824_824306


namespace inequality_inequality_hold_l824_824362

theorem inequality_inequality_hold (k : ℕ) (x y z : ℝ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) 
  (h_sum : x + y + z = 1) : 
  (x ^ (k + 2) / (x ^ (k + 1) + y ^ k + z ^ k) 
  + y ^ (k + 2) / (y ^ (k + 1) + z ^ k + x ^ k) 
  + z ^ (k + 2) / (z ^ (k + 1) + x ^ k + y ^ k)) 
  ≥ (1 / 7) :=
sorry

end inequality_inequality_hold_l824_824362


namespace least_integer_greater_than_sqrt_500_l824_824094

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824094


namespace least_int_gt_sqrt_500_l824_824055

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824055


namespace perfect_square_probability_l824_824520

theorem perfect_square_probability :
  let outcomes := Finset.pi (Finset.range 6) (fun _ => Finset.range 6)
  let is_perfect_square (l : List Nat) := ∃ n, l.prod = n * n
  let desired_outcome_count := (Finset.filter (fun l => is_perfect_square l) outcomes).card
  let total_outcomes := 6^4
  let probability := desired_outcome_count / total_outcomes
  probability = 25 / 162 :=
by
  sorry

end perfect_square_probability_l824_824520


namespace sum_of_medians_bounds_l824_824795

theorem sum_of_medians_bounds (a b c m_a m_b m_c : ℝ) 
    (h1 : m_a < (b + c) / 2)
    (h2 : m_b < (a + c) / 2)
    (h3 : m_c < (a + b) / 2)
    (h4 : ∀a b c : ℝ, a + b > c) :
    (3 / 4) * (a + b + c) < m_a + m_b + m_c ∧ m_a + m_b + m_c < a + b + c := 
by
  sorry

end sum_of_medians_bounds_l824_824795


namespace part1_1_part1_2_part2_l824_824284
noncomputable theory

def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2

theorem part1_1 : f (-real.sqrt 2) = 8 + 5 * real.sqrt 2 := by 
  sorry

theorem part1_2 (a : ℝ) : f (a + 3) = 3 * a^2 + 13 * a + 14 := by 
  sorry

theorem part2 (x : ℝ) : f (5^x) = 4 → x = real.log 2 / real.log 5 := by 
  sorry

end part1_1_part1_2_part2_l824_824284


namespace min_valid_triples_l824_824754

noncomputable def avg (a : Fin 9 → ℝ) : ℝ := (∑ i, a i) / 9

noncomputable def num_valid_triples (a : Fin 9 → ℝ) (m : ℝ) : ℕ :=
  ∑ i j k in Finset.triples 9, if a i + a j + a k ≥ 3 * m then 1 else 0

theorem min_valid_triples (a : Fin 9 → ℝ) (m : ℝ) (h : avg a = m) :
  num_valid_triples a m ≥ 28 :=
  sorry

end min_valid_triples_l824_824754


namespace sine_addition_l824_824234

noncomputable def sin_inv_45 := Real.arcsin (4 / 5)
noncomputable def tan_inv_12 := Real.arctan (1 / 2)

theorem sine_addition :
  Real.sin (sin_inv_45 + tan_inv_12) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sine_addition_l824_824234


namespace pumpkin_weight_difference_l824_824920

/- Definitions of pumpkin weights based on the problem conditions -/
def brad_pumpkin := 54
def jessica_pumpkin := brad_pumpkin / 2
def betty_pumpkin := 4 * jessica_pumpkin
def carlos_pumpkin := 2.5 * (brad_pumpkin + jessica_pumpkin)
def emily_pumpkin := 1.5 * (betty_pumpkin - brad_pumpkin)
def dave_pumpkin := (jessica_pumpkin + betty_pumpkin) / 2 + 20

/- Total weight of all pumpkins -/
def total_weight := brad_pumpkin + jessica_pumpkin + betty_pumpkin + carlos_pumpkin + emily_pumpkin + dave_pumpkin

/- Determine the heaviest and lightest pumpkins -/
def heaviest_pumpkin := max (max (max (max (max brad_pumpkin jessica_pumpkin) betty_pumpkin) carlos_pumpkin) emily_pumpkin) dave_pumpkin
def lightest_pumpkin := min (min (min (min (min brad_pumpkin jessica_pumpkin) betty_pumpkin) carlos_pumpkin) emily_pumpkin) dave_pumpkin

/- Proof goal: The difference between the heaviest and lightest pumpkins -/
theorem pumpkin_weight_difference :
  15 ≤ brad_pumpkin ∧ 15 ≤ jessica_pumpkin ∧ 15 ≤ betty_pumpkin ∧ 15 ≤ carlos_pumpkin ∧ 15 ≤ emily_pumpkin ∧ 15 ≤ dave_pumpkin ∧
  total_weight ≤ 750 ∧
  (heaviest_pumpkin - lightest_pumpkin) = 175.5 :=
by
  sorry

end pumpkin_weight_difference_l824_824920


namespace Eric_return_time_l824_824950

theorem Eric_return_time (t1 t2 t_return : ℕ) 
  (h1 : t1 = 20) 
  (h2 : t2 = 10) 
  (h3 : t_return = 3 * (t1 + t2)) : 
  t_return = 90 := 
by 
  sorry

end Eric_return_time_l824_824950


namespace no_three_digit_num_l824_824796

theorem no_three_digit_num :
  ¬ ∃ a b c : ℕ, 
    1 ≤ a ∧ a ≤ 9 ∧ -- a is hundreds digit, a cannot be 0
    0 ≤ b ∧ b ≤ 9 ∧ -- b is tens digit
    0 ≤ c ∧ c ≤ 9 ∧ -- c is units digit
    a + b + c = 100 * a + 10 * b + c - a * b * c :=
by
  intro h
  rcases h with ⟨a, b, c, ha1, ha2, hb1, hb2, hc1, hc2, eq⟩
  have eq' : 99 * a + 9 * b = a * b * c,
  { linarith, }
  have h1 : a * b * c ≥ 99 * a,
  { rw eq', exact le_add_of_nonneg_right (mul_nonneg (mul_nonneg (nat.cast_nonneg _) (nat.cast_nonneg _)) (nat.cast_nonneg _)), }
  have h2 : a * b * c ≤ 81,
  { rw eq', linarith [mul_le_mul_right (nat.cast_pos.2 hb2)], }
  exact lt_irrefl _ (lt_of_le_of_lt h1 h2)
sorry -- proof is not included as per instructions

end no_three_digit_num_l824_824796


namespace gcd_459_357_is_51_l824_824826

-- Define the problem statement
theorem gcd_459_357_is_51 : Nat.gcd 459 357 = 51 :=
by
  -- Proof here
  sorry

end gcd_459_357_is_51_l824_824826


namespace sequence_not_integer_l824_824859

-- Define the sequence as per the problem statement
def sequence (a b : ℕ) : ℕ → ℚ
| 1       := a
| 2       := b
| (n + 3) := (sequence (n + 1) ^ 2 + sequence (n + 2) ^ 2) / (sequence (n + 1) + sequence (n + 2))

theorem sequence_not_integer 
  (a b : ℕ) 
  (ha : a > 1)
  (hb : b > 1)
  (hab : Nat.gcd a b = 1) 
  (n : ℕ) 
  (hn : n ≥ 3) :
  ¬ ∃ k : ℤ, sequence a b n = ↑k :=
by sorry

end sequence_not_integer_l824_824859


namespace haleigh_cats_l824_824675

open Nat

def total_pairs := 14
def dog_leggings := 4
def legging_per_animal := 1

theorem haleigh_cats : ∀ (dogs cats : ℕ), 
  dogs = 4 → 
  total_pairs = dogs * legging_per_animal + cats * legging_per_animal → 
  cats = 10 :=
by
  intros dogs cats h1 h2
  sorry

end haleigh_cats_l824_824675


namespace entries_multiple_of_31_count_l824_824186

-- Definitions based on conditions in part (a)
def first_row : list ℕ := list.range' 1 26 |>.map (λ n, 2 * n - 1)

def a (n k : ℕ) : ℕ := 2^(n-1) * (n + 2*k - 2)

def is_multiple_of_31 (x : ℕ) : Prop := 31 ∣ x

-- Lean 4 theorem statement based on part (c)
theorem entries_multiple_of_31_count :
  (∃ entries : list ℕ, 
    (∀ i, i ∈ entries -> is_multiple_of_31 i) ∧ 
    entries.length = 14) :=
begin
  -- Sorry added to skip the proof
  sorry
end

end entries_multiple_of_31_count_l824_824186


namespace book_total_pages_l824_824159

theorem book_total_pages (num_chapters pages_per_chapter : ℕ) (h1 : num_chapters = 31) (h2 : pages_per_chapter = 61) :
  num_chapters * pages_per_chapter = 1891 := sorry

end book_total_pages_l824_824159


namespace winning_strategy_l824_824633

/-- Given a square table n x n, two players A and B are playing the following game: 
  - At the beginning, all cells of the table are empty.
  - Player A has the first move, and in each of their moves, a player will put a coin on some cell 
    that doesn't contain a coin and is not adjacent to any of the cells that already contain a coin. 
  - The player who makes the last move wins. 

  Cells are adjacent if they share an edge.

  - If n is even, player B has the winning strategy.
  - If n is odd, player A has the winning strategy.
-/
theorem winning_strategy (n : ℕ) : (n % 2 = 0 → ∃ (B_strat : winning_strategy_for_B), True) ∧ (n % 2 = 1 → ∃ (A_strat : winning_strategy_for_A), True) :=
by {
  admit
}

end winning_strategy_l824_824633


namespace ellipse_and_line_PQ_l824_824982

def ellipse_eq (x y c : ℝ) := x^2 / 6 + y^2 / 2 = 1

def eccentricity (c : ℝ) : ℝ := 2 / sqrt (6)

theorem ellipse_and_line_PQ (c : ℝ) (k : ℝ) (x y : ℝ)
  (h1 : 0 < c)
  (h2 : 2 * ∥(10 / c - c, 0) - (c, 0)∥ = ∥(c, 0)∥)
  (hx1x2_sum : (1 + 3 * k^2) * x^2 - 18 * k^2 * x + 27 * k^2 - 6 = 0)
  (hx1x2_prod : ((27 * k^2 - 6) * (5 * k^2)) / ((1 + 3 * k^2) * (1 + 3 * k^2)) = 0)
  : (∀ x y, ellipse_eq x y c) ∧ eccentricity c = sqrt (6) / 3 ∧ 
      (hx1x2_sum = 0 → (∃ k, y = k * (x - 3))) :=
by
  sorry

end ellipse_and_line_PQ_l824_824982


namespace circle_radius_l824_824383

theorem circle_radius
  (C S : Type)
  [MetricSpace C] [MetricSpace S]
  (diameter_angle : ∀ θ : ℝ, θ = 30 * Real.pi / 180)
  (tangent_to_diameters : ∀ (A B O : C), tangent (S) A ∧ tangent (S) B)
  (tangent_to_C : ∀ (O : C), tangent (S) O)
  (radius_S : ∀ (r_S : ℝ), r_S = 1) :  
  ∃ r_C : ℝ, r_C = 1 + Real.sqrt 6 + Real.sqrt 2 := 
sorry

end circle_radius_l824_824383


namespace limit_example_l824_824792

open Real

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, (∀ x : ℝ, 0 < |x + 4| ∧ |x + 4| < δ → abs ((2 * x^2 + 6 * x - 8) / (x + 4) + 10) < ε) :=
by
  sorry

end limit_example_l824_824792


namespace sarah_initial_trucks_l824_824140

theorem sarah_initial_trucks (trucks_given : ℕ) (trucks_left : ℕ) (initial_trucks : ℕ) :
  trucks_given = 13 → trucks_left = 38 → initial_trucks = trucks_left + trucks_given → initial_trucks = 51 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sarah_initial_trucks_l824_824140


namespace move_decimal_point_one_place_right_l824_824779

theorem move_decimal_point_one_place_right (x : ℝ) (h : x = 76.08) : x * 10 = 760.8 :=
by
  rw [h]
  -- Here, you would provide proof steps, but we'll use sorry to indicate the proof is omitted.
  sorry

end move_decimal_point_one_place_right_l824_824779


namespace least_integer_greater_than_sqrt_500_l824_824076

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824076


namespace range_of_x_l824_824322

theorem range_of_x (x : ℝ) :
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → 2 * x - 1 > m * (x^2 - 1)) ↔
  ( ∃ a b : ℝ, a = (sqrt 7 - 1) / 2 ∧ b = (sqrt 3 + 1) / 2 ∧ a < x ∧ x < b ) :=
by sorry

end range_of_x_l824_824322


namespace relationship_between_a_b_c_l824_824619

-- Definitions as given in the problem
def a : ℝ := 2 ^ 0.5
def b : ℝ := Real.log 5 / Real.log 2
def c : ℝ := Real.log 10 / Real.log 4

-- Statement we need to prove
theorem relationship_between_a_b_c : a < c ∧ c < b := by
  sorry

end relationship_between_a_b_c_l824_824619


namespace janet_height_l824_824410

variable (Pablo Charlene Janet : ℝ)
variable (Ruby : ℝ := 192)
variable (h1 : Ruby + 2 = Pablo)
variable (h2 : Pablo = Charlene + 70)
variable (h3 : Charlene = 2 * Janet)

theorem janet_height : Janet = 62 := by
  have pablo_height : Pablo = 194 := by
    rw [←h1, Ruby]
    norm_num
  have charlene_height : Charlene = 124 := by
    rw [←h2, pablo_height]
    norm_num
  have janet_height : Janet = 62 := by
    rw [←h3, charlene_height]
    norm_num
  exact janet_height

end janet_height_l824_824410


namespace least_integer_greater_than_sqrt_500_l824_824067

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824067


namespace no_solution_for_Cn_eq_Dn_l824_824364

open BigOperators

def C_n (n : ℕ) : ℤ :=
  if n = 0 then 0 
  else (1024 * (1 - (1 / (2:ℤ)^n).num * ((1 - (1 / 2.den * (1 - (1 / 2.den:ℤ))).num)))

def D_n (n : ℕ) : ℤ :=
  if n = 0 then 0
  else (3072 * (1 - (1 / ((-2):ℤ)^n).num * ((1 + (1 / 2.den * 1 / 2.den.num)).den)))

theorem no_solution_for_Cn_eq_Dn : ∀ n : ℕ, 1 ≤ n → ¬ (C_n n = D_n n) :=
by
  intro n hn
  have :=
  sorry

end no_solution_for_Cn_eq_Dn_l824_824364


namespace mila_snowman_volume_l824_824396

-- Define the volume of a sphere
def sphere_volume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

-- Define the given radii for the snowballs
def radius1 : ℝ := 4
def radius2 : ℝ := 6
def radius3 : ℝ := 8
def radius4 : ℝ := 10

-- Define the volumes of the individual snowballs
def V1 : ℝ := sphere_volume radius1
def V2 : ℝ := sphere_volume radius2
def V3 : ℝ := sphere_volume radius3
def V4 : ℝ := sphere_volume radius4

-- Define the total volume of the snowballs
def total_volume : ℝ := V1 + V2 + V3 + V4

-- Theorem stating the total volume is as given
theorem mila_snowman_volume : total_volume = (7168 / 3) * Real.pi :=
by sorry

end mila_snowman_volume_l824_824396


namespace correct_statement_l824_824127

theorem correct_statement :
  let monomial := {expr : Type* | ∃ (n : ℕ) (a : ℚ), expr = a * (variable ^ n)},
      is_monomial (e : Type*) := e ∈ monomial,
      coefficient (expr : Type*) := 0, -- Placeholder computation for the coefficient
      degree (e : Type*) := 0, -- Placeholder computation for the degree of monomial
      constant_term (poly : Type*) := 0 in -- Placeholder computation for constant term
    (¬ is_monomial (1 : ℕ)) ∨
    (coefficient (2 * π * a) = 2) ∨
    (degree (x * y * z^2) ≠ 4) ∨
    (constant_term ((2 : ℕ) * x^2 - 3 * x - 1) ≠ 1) :=
sorry

end correct_statement_l824_824127


namespace new_car_distance_l824_824177

-- State the given conditions as a Lemma in Lean 4
theorem new_car_distance (d_old : ℝ) (d_new : ℝ) (h1 : d_old = 150) (h2 : d_new = d_old * 1.30) : d_new = 195 := 
by
  sorry

end new_car_distance_l824_824177


namespace shorter_piece_length_l824_824158

theorem shorter_piece_length :
  ∃ (x : ℝ), x + 2 * x = 69 ∧ x = 23 :=
by
  sorry

end shorter_piece_length_l824_824158


namespace peanuts_remaining_l824_824843

theorem peanuts_remaining (initial_peanuts brock_ate bonita_ate brock_fraction : ℕ) (h_initial : initial_peanuts = 148) (h_brock_fraction : brock_fraction = 4) (h_brock_ate : brock_ate = initial_peanuts / brock_fraction) (h_bonita_ate : bonita_ate = 29) :
  (initial_peanuts - brock_ate - bonita_ate) = 82 :=
by
  sorry

end peanuts_remaining_l824_824843


namespace min_M_value_l824_824648

theorem min_M_value (a1 a2 a3 x y : ℝ) (h_nonzero : a1 ≠ 0 ∨ a2 ≠ 0 ∨ a3 ≠ 0)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x^2 + y^2 = 2) :
  (xa1a2_ya2_3 : ℝ) :=
  (xa1a2_ya2_3 = (x * a1 * a2 + y * a2 * a3) / (a1^2 + a2^2 + a3^2)) →
  xa1a2_ya2_3 ≤ real.sqrt(2) / 2 :=
sorry

end min_M_value_l824_824648


namespace largest_power_of_15_dividing_factorial_30_l824_824592

theorem largest_power_of_15_dividing_factorial_30 : ∃ (n : ℕ), 
  (∀ m : ℕ, 15 ^ m ∣ Nat.factorial 30 ↔ m ≤ n) ∧ n = 7 :=
by
  have h3 : ∀ m : ℕ, 3 ^ m ∣ Nat.factorial 30 ↔ m ≤ 14 := sorry
  have h5 : ∀ m : ℕ, 5 ^ m ∣ Nat.factorial 30 ↔ m ≤ 7 := sorry
  use 7
  split
  · intro m
    split
    · intro h
      obtain ⟨k, rfl⟩ : ∃ k, m = k := Nat.exists_eq m
      have : 3 ^ k ∣ Nat.factorial 30 := by exact (15 ^ k).dvd_of_dvd_mul_left (by convert h)
      rw [h3, h5] at this
      exact Nat.le_min this.left this.right
    · intro h
      exact (Nat.min_le_iff.mpr ⟨(h3.mpr (Nat.le_of_lt_succ (Nat.sub_lt_succ (14 - 7))), h5.mpr h⟩, sorry
  exact rfl
  sorry

end largest_power_of_15_dividing_factorial_30_l824_824592


namespace exist_P_Q_l824_824384

variable (X : Finset α) [Fintype α]
variable (f : (Finset α) → ℝ)
variable (even_subset : Finset α → Prop) -- Predicate for even-sized subset
variable (P Q : Finset α)

-- Conditions
axiom X_finite : Finite X
axiom f_on_even_subsets: ∀ E, even_subset E → f E ∈ ℝ
axiom D_exists : ∃ D, even_subset D ∧ f D > 1990
axiom f_disjoint_union : ∀ {A B : Finset α}, even_subset A → even_subset B → Disjoint A B → f (A ∪ B) = f A + f B - 1990

-- Proof statement
theorem exist_P_Q (X : Finset α) [Fintype α] (f : (Finset α) → ℝ) 
    (even_subset : Finset α → Prop) (P Q : Finset α) :
  (∨ P Q : Finset α, ∀ (X_finite : Finite X)
  (f_on_even_subsets : ∀ E, even_subset E → f E ∈ ℝ)
  (D_exists : ∃ D, even_subset D ∧ f D > 1990)
  (f_disjoint_union : ∀ {A B : Finset α}, even_subset A → even_subset B → Disjoint A B → f (A ∪ B) = f A + f B - 1990),
  (P ∩ Q = ∅) ∧ (P ∪ Q = X) ∧
  (∀ (S : Finset α), even_subset S → S ⊆ P → S ≠ ∅ → f S > 1990) ∧
  (∀ (T : Finset α), even_subset T → T ⊆ Q → f T ≤ 1990)) :=
sorry

end exist_P_Q_l824_824384


namespace part1_part2_l824_824758

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log (x^2 + 1) + b * x
noncomputable def g (a b x : ℝ) : ℝ := b * x^2 + 2 * a * x + b

-- Conditions
variables {a b x1 x2 λ : ℝ}
variable (h₀ : a > 0)
variable (h₁ : b > 0)
variable (h₂ : ∃ x1 x2, x1 ≠ 0 ∧ x2 ≠ 0 ∧ g a b x1 = 0 ∧ g a b x2 = 0 ∧ x1 ≠ x2)

-- Proof Statements

-- (1) Prove that x1 + x2 < -2
theorem part1 : x1 + x2 < -2 :=
sorry

-- (2) If the real number λ satisfies the equation f(x1) + f(x2) + 3a - λ * b = 0,
-- prove the range of λ is λ > 2*Real.log 2 + 1
theorem part2 (h₃ : f a b x1 + f a b x2 + 3 * a - λ * b = 0) : λ > 2 * Real.log 2 + 1 :=
sorry

end part1_part2_l824_824758


namespace find_x_l824_824584

def product_of_digits (x : ℕ) : ℕ :=
  x.digits.base 10 |> List.foldr (λ a b => a * b) 1

theorem find_x (x : ℕ) (h : product_of_digits x = x^2 - 10 * x - 22) (h' : product_of_digits x ≤ x) : x = 12 := 
  by sorry

end find_x_l824_824584


namespace find_y_l824_824581

-- Define the condition
def cond (y : ℝ) := 9 ^ (Real.log y / Real.log (8 : ℝ)) = 81

-- The theorem statement
theorem find_y (y : ℝ) (h : cond y) : y = 64 :=
by
  sorry

end find_y_l824_824581


namespace clock_angle_calculation_l824_824913

-- Define the constant values
def circle_degrees : ℝ := 360
def hours_in_clock : ℝ := 12
def minutes_in_hour : ℝ := 60
def hour_deg_incr : ℝ := circle_degrees / hours_in_clock
def minute_deg_incr : ℝ := circle_degrees / minutes_in_hour

-- Define the specific time
def specific_time_hours : ℝ := 8
def specific_time_minutes : ℝ := 15

-- Calculate the minute hand angle:
def minute_hand_angle : ℝ := specific_time_minutes * minute_deg_incr

-- Calculate the hour hand angle:
def hour_hand_angle : ℝ := (specific_time_hours + specific_time_minutes / minutes_in_hour) * hour_deg_incr

-- Function to calculate the angle difference
def angle_diff (angle1 angle2 : ℝ) : ℝ := abs (angle1 - angle2)

-- Smaller angle between the minute hand and hour hand
def smaller_angle : ℝ := 
  let diff := angle_diff hour_hand_angle minute_hand_angle in
  if diff > circle_degrees / 2 then circle_degrees - diff else diff

-- The required statement to be proved
theorem clock_angle_calculation : smaller_angle = 157.5 := by
  sorry

end clock_angle_calculation_l824_824913


namespace mod_remainder_l824_824482

theorem mod_remainder (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end mod_remainder_l824_824482


namespace smallest_n_congruent_l824_824862

theorem smallest_n_congruent (n : ℕ) (h : 635 * n ≡ 1251 * n [MOD 30]) : n = 15 :=
sorry

end smallest_n_congruent_l824_824862


namespace matrix_product_zero_l824_824217

variable {R : Type*} [Ring R]

def matrix_A (d e f : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![0, 2 * d, -e], ![-2 * d, 0, 3 * f], ![e, -3 * f, 0]]

def matrix_B (d e f : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![d^2 - e^2, d * e, d * f], ![d * e, e^2 - f^2, e * f], ![d * f, e * f, f^2 - d^2]]

theorem matrix_product_zero (d e f : R) (h1 : 2 * d^2 = e * f) (h2 : 2 * d^2 = 3 * e * f) (h3 : e * d = 3 * f^2) :
  matrix_A d e f ⬝ matrix_B d e f = 0 := sorry

end matrix_product_zero_l824_824217


namespace min_possible_value_sum_squares_l824_824380

theorem min_possible_value_sum_squares :
  ∃ (p q r s t u v w : ℤ),
  {p, q, r, s, t, u, v, w}.card = 8 ∧
  {p, q, r, s, t, u, v, w} ⊆ {-9, -8, -4, -1, 1, 5, 7, 10} ∧
  p + q + r + s + t + u + v + w = 1 ∧
  (p + q + r + s)^2 + (t + u + v + w)^2 = 1 :=
by
  sorry

end min_possible_value_sum_squares_l824_824380


namespace neg_i_pow_four_l824_824940

-- Define i as the imaginary unit satisfying i^2 = -1
def i : ℂ := Complex.I

-- The proof problem: Prove (-i)^4 = 1 given i^2 = -1
theorem neg_i_pow_four : (-i)^4 = 1 :=
by
  -- sorry is used to skip proof
  sorry

end neg_i_pow_four_l824_824940


namespace total_lunch_combinations_l824_824327

theorem total_lunch_combinations :
  let meats := 4 in
  let vegetables := 7 in
  let comb_two_meats := Nat.choose meats 2 in
  let comb_one_meat := Nat.choose meats 1 in
  let comb_two_vegetables := Nat.choose vegetables 2 in
  comb_two_meats * comb_two_vegetables + comb_one_meat * comb_two_vegetables = 210 := 
by 
  sorry

end total_lunch_combinations_l824_824327


namespace range_of_a_l824_824715

theorem range_of_a (a : ℝ) :
    (∀ x : ℤ, x + 1 > 0 → 3 * x - a ≤ 0 → x = 0 ∨ x = 1 ∨ x = 2) ↔ 6 ≤ a ∧ a < 9 :=
by
  sorry

end range_of_a_l824_824715


namespace sqrt_500_least_integer_l824_824018

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824018


namespace domain_of_f_l824_824239

def domain_function (x : ℝ) : Prop :=
  x > 7

noncomputable def f (x : ℝ) : ℝ :=
  (4 * x + 2) / real.sqrt (x - 7)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f y = (4 * y + 2) / real.sqrt (y - 7)) ↔ domain_function x :=
sorry

end domain_of_f_l824_824239


namespace average_price_of_pen_l824_824502

theorem average_price_of_pen (c_total : ℝ) (n_pens n_pencils : ℕ) (p_pencil : ℝ)
  (h1 : c_total = 450) (h2 : n_pens = 30) (h3 : n_pencils = 75) (h4 : p_pencil = 2) :
  (c_total - (n_pencils * p_pencil)) / n_pens = 10 :=
by
  sorry

end average_price_of_pen_l824_824502


namespace least_int_gt_sqrt_500_l824_824053

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824053


namespace seating_arrangements_l824_824732

noncomputable def total_arrangements (n : Nat) : Nat :=
  (n - 1)!  -- Total arrangements for n people in a circle

noncomputable def invalid_arrangements : Nat :=
  6! * 2!  -- Arrangements where Alice and Bob sit next to each other

theorem seating_arrangements :
  total_arrangements 8 - invalid_arrangements = 3600 := by
  sorry

end seating_arrangements_l824_824732


namespace four_dozen_cost_l824_824539

-- Defining the conditions
def cost_of_two_dozen (c : ℝ) : Prop := c = 15.60
def cost_per_dozen (c : ℝ) : ℝ := c / 2

-- The statement that we need to prove, given the conditions.
theorem four_dozen_cost (c : ℝ) (h : cost_of_two_dozen c) : 4 * (cost_per_dozen c) = 31.20 :=
by
  sorry

end four_dozen_cost_l824_824539


namespace least_integer_greater_than_sqrt_500_l824_824003

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824003


namespace optimal_garden_dimensions_l824_824813

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), l ≥ 100 ∧ w ≥ 60 ∧ l + w = 180 ∧ l * w = 8000 := by
  sorry

end optimal_garden_dimensions_l824_824813


namespace find_preimage_of_128_l824_824256

open Function

noncomputable def f : ℝ → ℝ := sorry

theorem find_preimage_of_128 (f : ℝ → ℝ) (h₁ : f(3) = 2) (h₂ : ∀ x, f(2 * x) = 2 * f(x)) :
  f⁻¹' {128} = {192} :=
by {
  -- Proof is omitted. This is the statement only.
  sorry
}

end find_preimage_of_128_l824_824256


namespace ravi_total_money_l824_824798

theorem ravi_total_money (nickels quarters dimes half_dollars pennies : ℕ) 
  (h1 : nickels = 6)
  (h2 : quarters = nickels + 2)
  (h3 : dimes = quarters + 4)
  (h4 : half_dollars = dimes + 5)
  (h5 : pennies = 3 * half_dollars) :
  (5 * nickels + 25 * quarters + 10 * dimes + 50 * half_dollars + pennies) / 100 = 12.51 := 
by
  sorry

end ravi_total_money_l824_824798


namespace quadrilateral_midpoints_rectangle_l824_824895

theorem quadrilateral_midpoints_rectangle
  (P Q R S : ℝ^3)
  (h1 : (diagonal PQ PR).perpendicular (diagonal PR QS)) :
  is_rectangle (quadrilateral_midpoints P Q R S) :=
sorry

end quadrilateral_midpoints_rectangle_l824_824895


namespace compare_2_roses_3_carnations_l824_824657

variable (x y : ℝ)

def condition1 : Prop := 6 * x + 3 * y > 24
def condition2 : Prop := 4 * x + 5 * y < 22

theorem compare_2_roses_3_carnations (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x > 3 * y := sorry

end compare_2_roses_3_carnations_l824_824657


namespace age_of_other_man_replaced_l824_824815

theorem age_of_other_man_replaced 
  (avg_age : ℕ)             -- the average age of the 10 men initially
  (age_of_replaced_1 : ℕ)    -- the age of the first man replaced (21 years)
  (avg_age_increase : ℕ)     -- the increase in the average age (2 years)
  (total_age_new_men : ℕ)    -- the total age of the two new men (64 years)
  : ∀ (x : ℕ),  -- age of the other man who was replaced
    age_of_replaced_1 = 21 →
    avg_age_increase = 2 →
    total_age_new_men = 2 * 32 →
    21 + x + (10 * avg_age_increase) = total_age_new_men →
    x = 23 := by
  intros x h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃] at h₄
  exact sorry

end age_of_other_man_replaced_l824_824815


namespace area_of_triangle_AED_l824_824738

-- Define the properties of triangle and midpoints
def is_right_isosceles (A B C D E : Type) (AB BC : ℝ) (H : AB = 24 ∧ BC = 24 ∧ ∀ x : A = D, B = C ∧ D = E): Prop := sorry

-- Prove the area of the triangle AED given the properties
theorem area_of_triangle_AED {A B C D E : Type} {AB BC : ℝ} (h : is_right_isosceles A B C D E AB BC) :
  ∃ AED_area : ℝ, AED_area = 72 := sorry

end area_of_triangle_AED_l824_824738


namespace integral_f_l824_824204

noncomputable def f (x : ℝ) := x^2 + 3

theorem integral_f :
  ∫ (x : ℝ) in 0..6, f x = 90 :=
by
  -- Proof will be provided here
  sorry

end integral_f_l824_824204


namespace exists_constant_C_l824_824367

def floor (x : ℝ) : ℤ := int.floor x
def fractional_part (x : ℝ) : ℝ := x - floor x

noncomputable def H : Set ℤ := { i | ∃ n : ℕ, i = floor (n * Real.sqrt 2) }

theorem exists_constant_C 
    (n : ℕ) 
    (A : Set ℕ) 
    (hA : A ⊆ Finset.range n) 
    (card_hA : Finset.card A ≥ C * Real.sqrt n) :
    ∃ (C > 0), ∃ a b ∈ A, a ≠ b ∧ (a - b : ℤ) ∈ H :=
sorry

end exists_constant_C_l824_824367


namespace statues_left_l824_824740

def year1_statues := 4

def year2_fibonacci := 1 + 1 + 2
def year2_triangular := 3
def year2_broken := 2 + 2
def year2_added := year2_fibonacci + year2_triangular
def year2_statues := year1_statues + year2_added - year2_broken

def year3_fibonacci := 3
def year3_triangular := 6
def year3_broken := 3 + 4
def year3_added := year3_fibonacci + year3_triangular
def year3_statues := year2_statues + year3_added - year3_broken

def year4_planned_fibonacci := 5
def year4_planned_triangular := 10
def year4_planned_added := year4_planned_fibonacci + year4_planned_triangular
def year4_prev_broken := year2_broken + year3_broken
def year4_added := year4_planned_added - year4_prev_broken
def year4_statues := year3_statues + year4_added

theorem statues_left : year4_statues = 13 := by
  unfold year1_statues
  unfold year2_fibonacci year2_triangular year2_broken year2_added year2_statues
  unfold year3_fibonacci year3_triangular year3_broken year3_added year3_statues
  unfold year4_planned_fibonacci year4_planned_triangular year4_planned_added year4_prev_broken year4_added year4_statues
  sorry

end statues_left_l824_824740


namespace part_one_q_eq_p_part_two_sum_b_n_l824_824266

-- Defines the arithmetic sequence sum
def sum_arithmetic (p q : ℝ) (n : ℕ) : ℝ :=
  p * (n:ℝ)^2 - 2 * n + q

-- Condition that a_n = 4 log_2 b_n
def a_n (p : ℝ) (n : ℕ) : ℝ :=
  2 * p * n - p - 2

-- Prove that q = p
theorem part_one_q_eq_p (p q : ℝ) : sum_arithmetic p q 1 = p - 2 + q → q = p :=
by
  intro h1
  sorry

-- Sequence b_n definition using log with base 2
def b_n (n : ℕ) : ℝ :=
  2 ^ (n - 1)

-- Prove that sum of first n terms of b_n is S_n = 2^n - 1
theorem part_two_sum_b_n (n : ℕ) : (finset.range n).sum b_n = 2^n - 1 :=
by
  sorry

end part_one_q_eq_p_part_two_sum_b_n_l824_824266


namespace numberOfWaysToDistributeMedals_correct_l824_824509

-- Define the medals and their constraints
noncomputable def numberOfWaysToDistributeMedals : ℕ :=
  (Nat.choose (12 - 1) (3 - 1))

theorem numberOfWaysToDistributeMedals_correct :
  numberOfWaysToDistributeMedals = 55 := by
  sorry

end numberOfWaysToDistributeMedals_correct_l824_824509


namespace triangle_has_three_altitudes_l824_824677

-- Assuming a triangle in ℝ² space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Definition of an altitude in the context of Lean
def altitude (T : Triangle) (p : ℝ × ℝ) := 
  ∃ (a : ℝ) (b : ℝ), T.A.1 * p.1 + T.A.2 * p.2 = a * p.1 + b -- Placeholder, real definition of altitude may vary

-- Prove that a triangle has exactly 3 altitudes
theorem triangle_has_three_altitudes (T : Triangle) : ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
  altitude T p₁ ∧ altitude T p₂ ∧ altitude T p₃ :=
sorry

end triangle_has_three_altitudes_l824_824677


namespace eric_return_home_time_l824_824943

-- Definitions based on conditions
def time_running_to_park : ℕ := 20
def time_jogging_to_park : ℕ := 10
def trip_to_park_time : ℕ := time_running_to_park + time_jogging_to_park
def return_time_multiplier : ℕ := 3

-- Statement of the problem
theorem eric_return_home_time : 
  return_time_multiplier * trip_to_park_time = 90 :=
by 
  -- Skipping proof steps
  sorry

end eric_return_home_time_l824_824943


namespace max_sum_consecutive_integers_less_360_l824_824471

theorem max_sum_consecutive_integers_less_360 :
  ∃ n : ℤ, n * (n + 1) < 360 ∧ (n + (n + 1)) = 37 :=
by
  sorry

end max_sum_consecutive_integers_less_360_l824_824471


namespace part_I_solution_part_II_solution_l824_824997

-- Defining f(x) given parameters a and b
def f (x a b : ℝ) := |x - a| + |x + b|

-- Part (I): Given a = 1 and b = 2, solve the inequality f(x) ≤ 5
theorem part_I_solution (x : ℝ) : 
  (f x 1 2) ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
by
  sorry

-- Part (II): Given the minimum value of f(x) is 3, find min (a^2 / b + b^2 / a)
theorem part_II_solution (a b : ℝ) (h : 3 = |a| + |b|) (ha : a > 0) (hb : b > 0) : 
  (min (a^2 / b + b^2 / a)) = 3 := 
by
  sorry

end part_I_solution_part_II_solution_l824_824997


namespace hotel_charge_l824_824817

variable (R G P : ℝ)

theorem hotel_charge (h1 : P = 0.60 * R) (h2 : P = 0.90 * G) : (R - G) / G = 0.50 :=
by
  sorry

end hotel_charge_l824_824817


namespace problem_I_domain_problem_II_sin2x_l824_824290

noncomputable def f (x : ℝ) : ℝ := (cos (2 * x)) / (sin (x + π/4))

theorem problem_I_domain (x: ℝ) : 
  ¬ ∃ k: ℤ, x = k * π - π/4 :=
sorry

theorem problem_II_sin2x (x: ℝ) (h: f x = 4 / 3) : 
  sin (2 * x) = 1 / 9 :=
sorry

end problem_I_domain_problem_II_sin2x_l824_824290


namespace roots_of_quadratic_complex_l824_824834

noncomputable def complexRootsOfQuadratic : Prop :=
  ∀ x : ℂ, (x^2 + x + 2 = 0) ↔ (x = (-1 + complex.sqrt(-7))/2) ∨ (x = (-1 - complex.sqrt(-7))/2)

theorem roots_of_quadratic_complex :
  complexRootsOfQuadratic :=
sorry

end roots_of_quadratic_complex_l824_824834


namespace probability_of_six_given_sum_is_9_is_half_l824_824510

-- Define the condition of a fair six-sided dice toss summing to 9
def fair_dice := {1, 2, 3, 4, 5, 6}

def sum_is_9 (x y : ℕ) := x + y = 9

-- Define the event of at least one "6" being tossed
def at_least_one_six (x y z : ℕ) := x = 6 ∨ y = 6 ∨ z = 6

-- Probability calculation using the valid outcomes
noncomputable def probability_at_least_one_six_given_sum_is_9 : ℝ :=
  (fair_dice.to_finset * fair_dice.to_finset * fair_dice.to_finset).filter (λ (t : ℕ × ℕ × ℕ),
    sum_is_9 t.1 t.2 ∧ at_least_one_six t.1 t.2 t.3).card.to_real /
  (fair_dice.to_finset * fair_dice.to_finset).filter (λ (t : ℕ × ℕ), sum_is_9 t.1 t.2).card.to_real

theorem probability_of_six_given_sum_is_9_is_half :
  probability_at_least_one_six_given_sum_is_9 = 1 / 2 :=
sorry

end probability_of_six_given_sum_is_9_is_half_l824_824510


namespace molecular_weight_of_3_moles_CaOH2_is_correct_l824_824121

-- Define the atomic weights as given by the conditions
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Define the molecular formula contributions for Ca(OH)2
def molecular_weight_CaOH2 : ℝ :=
  atomic_weight_Ca + 2 * atomic_weight_O + 2 * atomic_weight_H

-- Define the weight of 3 moles of Ca(OH)2 based on the molecular weight
def weight_of_3_moles_CaOH2 : ℝ :=
  3 * molecular_weight_CaOH2

-- Theorem to prove the final result
theorem molecular_weight_of_3_moles_CaOH2_is_correct :
  weight_of_3_moles_CaOH2 = 222.30 := by
  sorry

end molecular_weight_of_3_moles_CaOH2_is_correct_l824_824121


namespace lean_formula_l824_824811

theorem lean_formula (x y : ℚ) (r s : ℕ) (h₁ : y = (3 / 4) * x) (h₂ : x^y = y^x) (h₃ : x + y = (r : ℚ) / (s : ℚ)) (h₄ : Nat.gcd r s = 1) :
  r + s = 529 :=
by
  sorry

end lean_formula_l824_824811


namespace ratio_of_area_CDE_to_ABE_l824_824724

-- Define the basic properties and structures of the problem
variables (circle : Type) -- The circle in which all the points lie
variables {A B C D E : circle} -- Points on the circle
variables [diameter : diameter A B] -- AB is the diameter
variables [chord : chord C D] -- CD is a chord
variables [parallel : parallel AB CD] -- CD is parallel to AB
variables [intersection : intersection (line A C) (line B D) E] -- E is the intersection of lines AC and BD

-- Define the angles at E
variables (β : ℝ)  -- Angle AED is β
variables [angle_condition : angle AED β] -- Given condition of angle AED as β
variables [sum_condition : angle_sum_condition (angle AED + angle BED) (180 - 2*β)] -- Condition: angle AED + angle BED = 180° - 2β

-- Define the proof problem
theorem ratio_of_area_CDE_to_ABE (h : ∀ CDE ABE, ratio_area CDE ABE = cos β ^ 2) : 
  ∃ (CDE ABE : Type), ratio_area CDE ABE = cos β ^ 2 := 
sorry

end ratio_of_area_CDE_to_ABE_l824_824724


namespace intersection_of_sets_l824_824301

open Set Real

theorem intersection_of_sets :
  let M := {x : ℝ | x ≤ 4}
  let N := {x : ℝ | x > 0}
  M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 4} :=
by
  sorry

end intersection_of_sets_l824_824301


namespace total_points_scored_l824_824825

theorem total_points_scored :
  let a := 7
  let b := 8
  let c := 2
  let d := 11
  let e := 6
  let f := 12
  let g := 1
  let h := 7
  a + b + c + d + e + f + g + h = 54 :=
by
  let a := 7
  let b := 8
  let c := 2
  let d := 11
  let e := 6
  let f := 12
  let g := 1
  let h := 7
  sorry

end total_points_scored_l824_824825


namespace sum_abs_of_sequence_l824_824742

def sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => -60
  | (n+1) => sequence n + 3

theorem sum_abs_of_sequence :
  let s := ∑ i in Finset.range 30, |sequence i|
  s = 765 :=
by
  sorry

end sum_abs_of_sequence_l824_824742


namespace shadow_length_l824_824357

variable (H h d : ℝ) (h_pos : h > 0) (H_pos : H > 0) (H_neq_h : H ≠ h)

theorem shadow_length (x : ℝ) (hx : x = d * h / (H - h)) :
  x = d * h / (H - h) :=
sorry

end shadow_length_l824_824357


namespace exponential_function_identity_l824_824285

-- Given conditions
def f (x : ℝ) : ℝ := 5 ^ x

-- Statement to prove
theorem exponential_function_identity (a b : ℝ) (h : f (a + b) = 3) : f a * f b = 3 :=
  sorry

end exponential_function_identity_l824_824285


namespace domain_of_f_range_of_f_in_interval_l824_824882

def f (x : ℝ) : ℝ := log x (2^(-4 * x + 5 * 2^x + 1 - 16))

theorem domain_of_f :
  {x : ℝ | 1 < x ∧ x < 3} = {x : ℝ | -4 * x + 5 * 2^x + 1 - 16 > 0} :=
by sorry

theorem range_of_f_in_interval :
  (∀ x, 2 ≤ x ∧ x ≤ log x 27 → (log x (2^(-4 * x + 5 * 2^x + 1 - 16)) ∈ [log x 5, 2 * log x 3])) :=
by sorry

end domain_of_f_range_of_f_in_interval_l824_824882


namespace angle_A0_A3_A7_l824_824181

-- Define a regular decagon in the plane
structure RegularDecagon :=
(A : Fin 10 → ℝ × ℝ) -- A function mapping each vertex to a point in the plane
(is_regular : ∀ i j : Fin 10, dist (A i) (A ((i + 1) % 10)) = dist (A j) (A ((j + 1) % 10)))

-- Define the angle computation problem
theorem angle_A0_A3_A7 (dec : RegularDecagon) : 
  let A := dec.A in
  let angle_deg (A0 A3 A7 : ℝ × ℝ) := sorry in -- a function measuring the angle in degrees
  angle_deg (A 0) (A 3) (A 7) = 54 := 
sorry

end angle_A0_A3_A7_l824_824181


namespace distributive_laws_fail_for_all_l824_824365

def has_op_hash (a b : ℝ) : ℝ := a + 2 * b

theorem distributive_laws_fail_for_all (x y z : ℝ) : 
  ¬ (∀ x y z, has_op_hash x (y + z) = has_op_hash x y + has_op_hash x z) ∧
  ¬ (∀ x y z, x + has_op_hash y z = has_op_hash (x + y) (x + z)) ∧
  ¬ (∀ x y z, has_op_hash x (has_op_hash y z) = has_op_hash (has_op_hash x y) (has_op_hash x z)) := 
sorry

end distributive_laws_fail_for_all_l824_824365


namespace four_dozen_cost_l824_824541

-- Defining the conditions
def cost_of_two_dozen (c : ℝ) : Prop := c = 15.60
def cost_per_dozen (c : ℝ) : ℝ := c / 2

-- The statement that we need to prove, given the conditions.
theorem four_dozen_cost (c : ℝ) (h : cost_of_two_dozen c) : 4 * (cost_per_dozen c) = 31.20 :=
by
  sorry

end four_dozen_cost_l824_824541


namespace problem_equivalence_l824_824978

-- Define the given circles and their properties
def E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1

-- Define the curve C as the trajectory of the center of the moving circle P
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l intersecting curve C at points A and B with midpoint M(1,1)
def M (A B : ℝ × ℝ) : Prop := (A.1 + B.1 = 2) ∧ (A.2 + B.2 = 2)
def l (x y : ℝ) : Prop := x + 4 * y - 5 = 0

theorem problem_equivalence :
  (∀ x y, E x y ∧ F x y → C x y) ∧
  (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧ M A B → (∀ x y, l x y)) :=
sorry

end problem_equivalence_l824_824978


namespace largest_n_divides_30_factorial_l824_824595

theorem largest_n_divides_30_factorial : 
  ∃ (n : ℕ), (∀ (a b : ℕ), (prime a ∧ prime b ∧ a * b = 15) → prime_factors_in_factorial 30 a ≥ n ∧ prime_factors_in_factorial 30 b ≥ n) ∧ 
             (∀ m : ℕ, (∀ (a b : ℕ), (prime a ∧ prime b ∧ a * b = 15) → prime_factors_in_factorial 30 a ≥ m ∧ prime_factors_in_factorial 30 b ≥ m) → m ≤ n) ∧ n = 7 :=
by {
  sorry
}

end largest_n_divides_30_factorial_l824_824595


namespace range_q_interval_l824_824762

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, is_prime p ∧ p ∣ n ∧ p ≠ 1 then classical.some h else n

def q (x : ℝ) : ℝ :=
  if is_prime (⌊x⌋) then x^2 - 1
  else q (smallest_prime_factor (⌊x⌋)) + (x^2 - (⌊x⌋:ℝ)^2)

theorem range_q_interval :
  set.range q = {x | 8 ≤ x ∧ x < 9} ∪ {x | 24 ≤ x ∧ x < 25} ∪ {x | 48 ≤ x ∧ x < 49} ∪
                {x | 120 ≤ x ∧ x < 121} ∪ {x | 168 ≤ x ∧ x < 169} ∪ {27} :=
sorry

end range_q_interval_l824_824762


namespace limit_trig_identity_l824_824206

theorem limit_trig_identity :
  filter.tendsto (λ x : ℝ, (1 - (real.sin (2 * x))) / ((real.pi - 4 * x)^2))
    (filter.nhds_within (real.pi / 4) filter.univ) 
    (filter.nhds (1 / 8)) :=
sorry

end limit_trig_identity_l824_824206


namespace sqrt_500_least_integer_l824_824023

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824023


namespace quadrant_of_points_l824_824693

theorem quadrant_of_points (x y : ℝ) (h : |3 * x + 2| + |2 * y - 1| = 0) : 
  ((x < 0) ∧ (y > 0) ∧ (x + 1 > 0) ∧ (y - 2 < 0)) :=
by
  sorry

end quadrant_of_points_l824_824693


namespace find_slant_height_l824_824452

namespace ConeGeometry

-- Definitions for the conditions
def radius : ℝ := 3
def CSA : ℝ := 141.3716694115407

-- Definition for slant height
def slantHeight (r CSA : ℝ) : ℝ := CSA / (Real.pi * r)

-- Main theorem statement
theorem find_slant_height : slantHeight radius CSA = 15 := by
  sorry

end ConeGeometry

end find_slant_height_l824_824452


namespace least_integer_greater_than_sqrt_500_l824_824043

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824043


namespace largest_sections_five_lines_l824_824349

-- Define the number of sections created by n lines
def num_sections : ℕ → ℕ
| 0     := 1
| (n+1) := num_sections n + (n + 1)

-- State the theorem to be proven
theorem largest_sections_five_lines : num_sections 5 = 16 :=
by
  -- Proof omitted: this is the place where the proof would be constructed
  sorry

end largest_sections_five_lines_l824_824349


namespace probability_no_shaded_square_l824_824151

theorem probability_no_shaded_square :
  let num_rects := 1003 * 2005
  let num_rects_with_shaded := 1002^2
  let probability_no_shaded := 1 - (num_rects_with_shaded / num_rects)
  probability_no_shaded = 1 / 1003 := by
  -- The proof steps go here
  sorry

end probability_no_shaded_square_l824_824151


namespace increasing_interval_f_l824_824827

noncomputable def f (x : ℝ) : ℝ := real.logb (1/2) (2 * x^2 - 3 * x + 1)

def domain_t (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 > 0

theorem increasing_interval_f :
  ∀ x : ℝ, domain_t x → (x < 1/2) → (∀ y, y < x → f y < f x) :=
sorry

end increasing_interval_f_l824_824827


namespace largest_of_five_consecutive_sum_180_l824_824527

theorem largest_of_five_consecutive_sum_180 (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 180) :
  n + 4 = 38 :=
by
  sorry

end largest_of_five_consecutive_sum_180_l824_824527


namespace selection_ways_l824_824516

/-- 
A math interest group in a vocational school consists of 4 boys and 3 girls. 
If 3 students are randomly selected from these 7 students to participate in a math competition, 
and the selection must include both boys and girls, then the number of different ways to select the 
students is 30.
-/
theorem selection_ways (B G : ℕ) (students : ℕ) (selections : ℕ) (condition_boys_girls : B = 4 ∧ G = 3)
  (condition_students : students = B + G) (condition_selections : selections = 3) :
  (B = 4 ∧ G = 3 ∧ students = 7 ∧ selections = 3) → 
  ∃ (res : ℕ), res = 30 :=
by
  sorry

end selection_ways_l824_824516


namespace distance_between_points_on_parabola_l824_824514

theorem distance_between_points_on_parabola (x1 y1 x2 y2 : ℝ) 
  (h_parabola : ∀ (x : ℝ), 4 * ((x^2)/4) = x^2) 
  (h_focus : F = (0, 1))
  (h_line : y1 = k * x1 + 1 ∧ y2 = k * x2 + 1)
  (h_intersects : x1^2 = 4 * y1 ∧ x2^2 = 4 * y2)
  (h_y_sum : y1 + y2 = 6) :
  |dist (x1, y1) (x2, y2)| = 8 := sorry

end distance_between_points_on_parabola_l824_824514


namespace fraction_power_multiplication_l824_824928

theorem fraction_power_multiplication :
  ((1 : ℝ) / 3) ^ 4 * ((1 : ℝ) / 5) = ((1 : ℝ) / 405) := by
  sorry

end fraction_power_multiplication_l824_824928


namespace eric_return_home_time_l824_824944

-- Definitions based on conditions
def time_running_to_park : ℕ := 20
def time_jogging_to_park : ℕ := 10
def trip_to_park_time : ℕ := time_running_to_park + time_jogging_to_park
def return_time_multiplier : ℕ := 3

-- Statement of the problem
theorem eric_return_home_time : 
  return_time_multiplier * trip_to_park_time = 90 :=
by 
  -- Skipping proof steps
  sorry

end eric_return_home_time_l824_824944


namespace peanuts_remaining_l824_824845

theorem peanuts_remaining (initial : ℕ) (brock_fraction : ℚ) (bonita_count : ℕ) (h_initial : initial = 148) (h_brock_fraction : brock_fraction = 1/4) (h_bonita_count : bonita_count = 29) : initial - (initial * brock_fraction).natValue - bonita_count = 82 := 
by 
  sorry

end peanuts_remaining_l824_824845


namespace find_a1_b1_l824_824182

noncomputable def sequence_relation : ℕ → ℂ
| 0 := 0
| (n + 1) := 
  let z_n := sequence_relation n in
  let a_n := z_n.re in
  let b_n := z_n.im in
  ⟨(real.sqrt 2 * a_n - b_n), (real.sqrt 2 * b_n + a_n)⟩

theorem find_a1_b1 (a_100 b_100 : ℝ) (h : (a_100, b_100) = (1, 3)) :
  let z_1 := (sequence_relation 100) in
  z_1 = (1 + 3 * complex.I) / (3^49.5 * -complex.I) →
  z_1 = (3 - complex.I) / 3^49.5 →
  a_1 + b_1 = 2 / 3^49.5 :=
by 
  assume h1 h2,
  sorry

end find_a1_b1_l824_824182


namespace jennifer_money_left_l824_824490

variable (initial_amount : ℝ) (spent_sandwich_rate : ℝ) (spent_museum_rate : ℝ) (spent_book_rate : ℝ)

def money_left := initial_amount - (spent_sandwich_rate * initial_amount + spent_museum_rate * initial_amount + spent_book_rate * initial_amount)

theorem jennifer_money_left (h_initial : initial_amount = 150)
  (h_sandwich_rate : spent_sandwich_rate = 1/5)
  (h_museum_rate : spent_museum_rate = 1/6)
  (h_book_rate : spent_book_rate = 1/2) :
  money_left initial_amount spent_sandwich_rate spent_museum_rate spent_book_rate = 20 :=
by
  sorry

end jennifer_money_left_l824_824490


namespace sum_of_solutions_eq_neg_8_div_3_l824_824387

def f (x : ℝ) : ℝ :=
if x < -3 then 3 * x + 4 else -x^2 - 2 * x + 2

theorem sum_of_solutions_eq_neg_8_div_3 :
  let solutions := {x | f x = -7}
  let sum := solutions.toFinset.sum
  ∃ s : ℝ, sum = -8 / 3 :=
sorry

end sum_of_solutions_eq_neg_8_div_3_l824_824387


namespace nursery_school_total_students_l824_824876

noncomputable def students_in_nursery : ℕ := 50

theorem nursery_school_total_students :
  (∃ S : ℕ, (1 / 10 : ℝ) * S = 5 ∧
             20 + ((1 / 10 : ℝ) * S : ℕ) = 25) →
  students_in_nursery = 50 :=
by
  intros h
  sorry

end nursery_school_total_students_l824_824876


namespace congruent_triangles_l824_824303

variables {A B C E F G H I J P1 Q1 R1 P2 Q2 R2 : Type}

-- Given triangle ABC and the constructions of squares
def is_square (a b c d : Type) : Prop := sorry

variables [triangle_ABC : triangle A B C]
          [square_ABEF : is_square A B E F]
          [square_BCGH : is_square B C G H]
          [square_CAIJ : is_square C A I J]
          [point_P1 : P1 = intersection_of_lines (line_through A H) (line_through B J)]
          [point_Q1 : Q1 = intersection_of_lines (line_through B J) (line_through C F)]
          [point_R1 : R1 = intersection_of_lines (line_through C F) (line_through A H)]
          [point_P2 : P2 = intersection_of_lines (line_through A G) (line_through C E)]
          [point_Q2 : Q2 = intersection_of_lines (line_through B I) (line_through A G)]
          [point_R2 : R2 = intersection_of_lines (line_through C E) (line_through B I)]

-- Statement to prove the congruence of triangles
theorem congruent_triangles : triangle P1 Q1 R1 ≅ triangle P2 Q2 R2 := sorry

end congruent_triangles_l824_824303


namespace range_of_dot_product_l824_824342

def point_O : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (1, 0)
def point_B (k : ℝ) : ℝ × ℝ := (1 - 4 * k / (1 + k^2), 4 / (1 + k^2))
def dot_product (u v : ℝ × ℝ) : ℝ := (u.1 * v.1) + (u.2 * v.2)

theorem range_of_dot_product (k : ℝ) : -1 ≤ dot_product point_A (point_B k) ∧ dot_product point_A (point_B k) ≤ 3 :=
  sorry

end range_of_dot_product_l824_824342


namespace largest_n_divides_30_factorial_l824_824594

theorem largest_n_divides_30_factorial : 
  ∃ (n : ℕ), (∀ (a b : ℕ), (prime a ∧ prime b ∧ a * b = 15) → prime_factors_in_factorial 30 a ≥ n ∧ prime_factors_in_factorial 30 b ≥ n) ∧ 
             (∀ m : ℕ, (∀ (a b : ℕ), (prime a ∧ prime b ∧ a * b = 15) → prime_factors_in_factorial 30 a ≥ m ∧ prime_factors_in_factorial 30 b ≥ m) → m ≤ n) ∧ n = 7 :=
by {
  sorry
}

end largest_n_divides_30_factorial_l824_824594


namespace simplify_power_expression_l824_824801

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^5 = 243 * x^20 :=
by
  sorry

end simplify_power_expression_l824_824801


namespace geometric_sequence_fourth_term_l824_824390

theorem geometric_sequence_fourth_term (x : ℚ) (r : ℚ)
  (h1 : x ≠ 0)
  (h2 : x ≠ -1)
  (h3 : 3 * x + 3 = r * x)
  (h4 : 5 * x + 5 = r * (3 * x + 3)) :
  r^3 * (5 * x + 5) = -125 / 12 :=
by
  sorry

end geometric_sequence_fourth_term_l824_824390


namespace probability_not_sitting_next_to_each_other_probability_Lila_Tom_not_next_to_each_other_l824_824333

theorem probability_not_sitting_next_to_each_other : (9.choose 2 - 9) = 36 :=
by sorry

def probability_of_not_sitting_next_to_each_other (total_ways : ℕ) (adjacent_ways : ℕ) : ℚ :=
  (total_ways - adjacent_ways) / total_ways

theorem probability_Lila_Tom_not_next_to_each_other : probability_of_not_sitting_next_to_each_other 45 9 = 4 / 5 :=
by sorry

end probability_not_sitting_next_to_each_other_probability_Lila_Tom_not_next_to_each_other_l824_824333


namespace derivative_of_f_l824_824663

def f (x : ℝ) : ℝ := (2 * x - 1) * (x ^ 2 + 3)

theorem derivative_of_f :
  (deriv f) = (λ x, 6 * x ^ 2 - 2 * x + 6) :=
sorry

end derivative_of_f_l824_824663


namespace cuberoot_eq_l824_824957

open Real

theorem cuberoot_eq (x : ℝ) (h: (5:ℝ) * x + 4 = (5:ℝ) ^ 3 / (2:ℝ) ^ 3) : x = 93 / 40 := by
  sorry

end cuberoot_eq_l824_824957


namespace find_starting_number_l824_824454

theorem find_starting_number (S : ℤ) (n : ℤ) (sum_eq : 10 = S) (consec_eq : S = (20 / 2) * (n + (n + 19))) : 
  n = -9 := 
by
  sorry

end find_starting_number_l824_824454


namespace range_of_a_l824_824643

variable {x a : ℝ}

def A : Set ℝ := {x | x ≤ -3 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

theorem range_of_a (A_union_B_R : A ∪ B a = Set.univ) : a ∈ Set.Ici 3 :=
  sorry

end range_of_a_l824_824643


namespace sequence_a_not_perfect_square_l824_824183

noncomputable def a : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := a n * a (n + 1) + 1

theorem sequence_a_not_perfect_square {n : ℕ} (h : n ≥ 2) : ¬ ∃ k : ℕ, a n = k * k :=
by
  sorry

end sequence_a_not_perfect_square_l824_824183


namespace daily_production_l824_824900

theorem daily_production (x : ℕ) (hx1 : 216 / x > 4)
  (hx2 : 3 * x + (x + 8) * ((216 / x) - 4) = 232) : 
  x = 24 := by
sorry

end daily_production_l824_824900


namespace ammonium_nitrate_formed_l824_824241

-- Defining the chemical substances and the stoichiometry
def ammonia := "NH₃"
def nitric_acid := "HNO₃"
def ammonium_nitrate := "NH₄NO₃"

-- Defining the balanced chemical equation as a condition
def balanced_chemical_equation : Prop :=
  ∀ (nNH₃ nHNO₃ nNH₄NO₃ : ℕ), 
    (nNH₃ ≥ nHNO₃) →
    (nHNO₃ = 3) →
    (nNH₄NO₃ = nHNO₃)

-- The statement asserting the proof problem
theorem ammonium_nitrate_formed : balanced_chemical_equation → ∀ (nNH₄NO₃ : ℕ), nNH₄NO₃ = 3 :=
by
  intros h1 nNH₄NO₃
  have h2: nNH₃ ≥ nHNO₃ := sorry  -- Using there is enough ammonia
  have h3: nHNO₃ = 3 := sorry  -- Given condition
  exact h1 nNH₃ nHNO₃ nNH₄NO₃ h2 h3

end ammonium_nitrate_formed_l824_824241


namespace net_effect_sale_value_l824_824135

variable {P Q : ℝ}

theorem net_effect_sale_value (hP : P > 0) (hQ : Q > 0) :
  (0.8 * P * 1.8 * Q - P * Q) / (P * Q) * 100 = 44 :=
by
  -- Simplify to match the known answer step by step
  calc
    (0.8 * P * 1.8 * Q - P * Q) / (P * Q) * 100
      = ((0.8 * 1.8 * P * Q) - (P * Q)) / (P * Q) * 100 : by ring
  ... = ((1.44 * P * Q) - (P * Q)) / (P * Q) * 100 : by simp [mul_assoc]
  ... = (0.44 * P * Q) / (P * Q) * 100 : by ring
  ... = 0.44 * 100 : by field_simp [mul_comm, ne_of_gt hP, ne_of_gt hQ]
  ... = 44 : by norm_num

end net_effect_sale_value_l824_824135


namespace tan_neg_55_over_6_pi_l824_824244

theorem tan_neg_55_over_6_pi : 
  tan (- (55 / 6) * Real.pi) = - (Real.sqrt 3 / 3) := 
sorry

end tan_neg_55_over_6_pi_l824_824244


namespace least_integer_greater_than_sqrt_500_l824_824116

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824116


namespace evaluate_expression_l824_824232

variable {z p q : ℝ}

theorem evaluate_expression (hz : z ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  ( ( (z ^ (2 / p) + z ^ (2 / q)) ^ 2 - 4 * z ^ (2 / p + 2 / q) ) /
    ( (z ^ (1 / p) - z ^ (1 / q)) ^ 2 + 4 * z ^ (1 / p + 1 / q) ) )^(1 / 2) =
  | z ^ (1 / p) - z ^ (1 / q) | :=
sorry

end evaluate_expression_l824_824232


namespace geometric_progression_ratio_l824_824993

theorem geometric_progression_ratio 
  (x y z : ℂ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (h_non_zero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h_sum_zero : x + y + z = 0) 
  (h_geom_prog : ∃ (r : ℂ), x * (y - z) = r ∧ y * (z - x) = r * y ∧ z * (x - y) = r * r * z ) :
  ∃ (r : ℂ), r = (-1 + complex.I * complex.sqrt 3) / 2 
                ∨ r = (-1 - complex.I * complex.sqrt 3) / 2 :=
by
  sorry

end geometric_progression_ratio_l824_824993


namespace distance_D_to_centroid_ABC_l824_824270

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2)

def centroid (A B C : Point3D) : Point3D :=
  { x := (A.x + B.x + C.x) / 3,
    y := (A.y + B.y + C.y) / 3,
    z := (A.z + B.z + C.z) / 3 }

noncomputable def D : Point3D := {x := -2, y := 2, z := -1}
noncomputable def A : Point3D := {x := -1, y := 2, z := 0}
noncomputable def B : Point3D := {x := 5, y := 2, z := -1}
noncomputable def C : Point3D := {x := 2, y := -1, z := 4}

theorem distance_D_to_centroid_ABC :
  distance D (centroid A B C) = Real.sqrt 21 :=
by
  sorry

end distance_D_to_centroid_ABC_l824_824270


namespace least_integer_greater_than_sqrt_500_l824_824004

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824004


namespace calculation_result_l824_824922

theorem calculation_result :
  1500 * 451 * 0.0451 * 25 = 7627537500 :=
by
  -- Simply state without proof as instructed
  sorry

end calculation_result_l824_824922


namespace height_of_peak_l824_824852

/-- Given measurements to determine the height of a mountain peak. -/
theorem height_of_peak 
  {A B C H T : ℝ} 
  (AB : ℝ) (BC : ℝ) (angleABBC : ℝ)
  (elevA : ℝ) (elevC : ℝ)
  (cot20 : ℝ) (cot22 : ℝ)
  (ht1 : ℝ) (ht2 : ℝ) :
  AB = 100 → BC = 150 → angleABBC = 130 →
  elevA = 20 → elevC = 22 →
  cot20 = Real.cot 20 → cot22 = Real.cot 22 →
  ht1 = 93.4 → ht2 = 390.9 →
  (height H T = ht1 ∨ height H T = ht2) :=
by
  sorry

end height_of_peak_l824_824852


namespace part_one_part_two_l824_824666

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2

theorem part_one (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m * n > 1) : f m >= 0 ∨ f n >= 0 :=
sorry

theorem part_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (hf : f a = f b) : a + b < 4 / 3 :=
sorry

end part_one_part_two_l824_824666


namespace david_basketball_shots_l824_824932

theorem david_basketball_shots :
  ∀ (hits_first: ℕ) (hits_total: ℕ),
  (hits_first = 12) ∧ (hits_total = 20) ∧ ((hits_total - hits_first = 8)) →
  hits_total - hits_first = 8 :=
by intros hits_first hits_total h;
  cases h with h1 h_2; 
  cases h_2 with h2 h3;
  exact h3;
  sorry

end david_basketball_shots_l824_824932


namespace line_hyperbola_intersect_exactly_once_l824_824671

noncomputable def intersection_points_count (k : ℝ) : ℕ :=
  let disc b c := b^2 - 4*c
  let eqn := disc (2 * (2 : ℝ).sqrt * k^2) ((1 - 4*k^2) / 4 * (-2*k^2 - 1))
  if (1 - 4*k^2 = 0) then 1 else
    if eqn = 0 then 1 else 0

theorem line_hyperbola_intersect_exactly_once :
  {k : ℝ // (intersection_points_count k) = 1}.to_finset.card = 4 := sorry

end line_hyperbola_intersect_exactly_once_l824_824671


namespace peter_pizza_fraction_l824_824788

theorem peter_pizza_fraction :
  let total_slices := 18
  let peter_alone := 3
  let slices_with_paul := 2 / 2
  let slice_with_mary := 1 / 2 / 18
  peter_alone / total_slices + slices_with_paul / total_slices + slice_with_mary = 11 / 36 :=
by
  let total_slices := 18
  let peter_alone := 3
  let slices_with_paul := 2 / 2
  let slice_with_mary := 1 / 2 / 18
  have h1 : peter_alone / total_slices = 1 / 6 := by sorry
  have h2 : slices_with_paul / total_slices = 1 / 9 := by sorry
  have h3 : slice_with_mary = 1 / 36 := by sorry
  calc
    1 / 6 + 1 / 9 + 1 / 36 = 6 / 36 + 4 / 36 + 1 / 36 := by sorry
    ... = 11 / 36 := by sorry

end peter_pizza_fraction_l824_824788


namespace vector_EB_l824_824277

noncomputable def is_parallelogram (A B C D : Type) (AB AC : Type) :=
by sorry  -- Assume a function to validate parallelogram characteristics

variables {A B C D E : Type}
variables {a b : Type}
variables [vector_space ℝ a] [vector_space ℝ b]

theorem vector_EB (h1 : is_parallelogram A B C D)
                  (h2 : vector A B = a)
                  (h3 : vector A C = b)
                  (h4 : midpoint E C D) : 
  vector E B = (3 / 2) • a - b :=
by sorry

end vector_EB_l824_824277


namespace measure_of_angle_C_l824_824353

-- Given conditions
variables {a b c : ℝ}  -- Sides of the triangle
variable h : a^2 + b^2 = c^2 + real.sqrt 3 * a * b  -- Given equation

-- Goal: Prove angle C = 30 degrees
theorem measure_of_angle_C {α β γ : ℝ} (htri : α + β + γ = π) : 
  a^2 + b^2 = c^2 + real.sqrt 3 * a * b → 
  γ = real.pi / 6 := 
begin
  sorry
end

end measure_of_angle_C_l824_824353


namespace median_is_4_l824_824282

def data_set : List ℕ := [1, 4, 4, 5, 1]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  sorted.get! (l.length / 2)

theorem median_is_4 : median data_set = 4 := 
by 
  sorry

end median_is_4_l824_824282


namespace smallest_pos_d_l824_824873

theorem smallest_pos_d (d : ℕ) (h : d > 0) (hd : ∃ k : ℕ, 3150 * d = k * k) : d = 14 := 
by 
  sorry

end smallest_pos_d_l824_824873


namespace alice_lawn_area_l824_824905

theorem alice_lawn_area (posts : ℕ) (distance : ℕ) (ratio : ℕ) : 
    posts = 24 → distance = 5 → ratio = 3 → 
    ∃ (short_side long_side : ℕ), 
        (2 * (short_side + long_side - 2) = posts) ∧
        (long_side = ratio * short_side) ∧
        (distance * (short_side - 1) * distance * (long_side - 1) = 825) :=
by
  intros h_posts h_distance h_ratio
  sorry

end alice_lawn_area_l824_824905


namespace cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2_l824_824578

theorem cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2 : 
  Real.cos (- (11 / 4) * Real.pi) = - Real.sqrt 2 / 2 := 
sorry

end cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2_l824_824578


namespace domain_of_f_l824_824820

noncomputable def f (x : ℝ) := 1 / Real.logb 2 (x - 1)

theorem domain_of_f :
  ∀ x : ℝ, x > 1 ∧ Real.logb 2 (x - 1) ≠ 0 ↔ x ∈ set.Ioo 1 2 ∪ set.Ioi 2 :=
by
  sorry

end domain_of_f_l824_824820


namespace BoxesMakers_l824_824733

-- Definitions for the boxes
def Box : Type := ℕ  -- Assume Box is represented by some indices, say ℕ.

def Box1 : Box := 1  -- Golden Box
def Box2 : Box := 2  -- Silver Box

def Cellini : Box → Prop := λ b, b = Box1 ∨ b = Box2
def NotCellinisSon : Box → Prop := λ b, b ≠ Box1 ∧ b ≠ Box2

def MadeBy (b : Box) (p: Box → Prop) : Prop := p b

-- Problem conditions
def GoldBoxCondition : Prop := ∀ b, (MadeBy b Cellini)
def SilverBoxCondition : Prop := ∀ b, (MadeBy b NotCellinisSon)

-- Correct answers to be proved
def GoldBoxMaker := Box1 -- Made by Cellini
def SilverBoxMaker := Box2 -- Made by Bellini

-- Main theorem to prove
theorem BoxesMakers :
    ¬GoldBoxCondition → SilverBoxCondition → MadeBy Box1 Cellini ∧ MadeBy Box2 Bellini :=
by sorry

end BoxesMakers_l824_824733


namespace correct_remainder_l824_824246

def remainder (m v : ℕ) : ℕ :=
  m % v

theorem correct_remainder (x : ℕ) (h : x = 40) :
  remainder (remainder x 33) 17 - remainder 99 (remainder 33 17) = 4 :=
by
  have h₁ : remainder 33 17 = 16 := by norm_num
  have h₂ : remainder 99 16 = 3 := by norm_num
  rw [h, h₁, h₂]
  have h₃ : remainder 40 33 = 7 := by norm_num
  rw [h₃]
  have h₄ : remainder 7 17 = 7 := by norm_num
  rw [h₄]
  norm_num
  sorry

end correct_remainder_l824_824246


namespace find_vector_d_l824_824440

noncomputable def line_param (t : ℝ) : ℝ × ℝ :=
  (4, 2) + t • (5/9, 4/9)

def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem find_vector_d : ∀ (t : ℝ), 
  t = distance (line_param t) (4, 2) →
  line_param t = (4, 2) + t • (5/9, 4/9) := 
by
  intro t
  assume h
  sorry

end find_vector_d_l824_824440


namespace product_of_real_solutions_l824_824450

theorem product_of_real_solutions :
  (∀ x : ℝ, (x + 1) / (3 * x + 3) = (3 * x + 2) / (8 * x + 2)) →
  x = -1 ∨ x = -4 →
  (-1) * (-4) = 4 := 
sorry

end product_of_real_solutions_l824_824450


namespace number_of_men_l824_824508

theorem number_of_men (M : ℕ) (h : M * 40 = 20 * 68) : M = 34 :=
by
  sorry

end number_of_men_l824_824508


namespace repetend_of_4_div_17_l824_824242

theorem repetend_of_4_div_17 :
  ∃ (r : String), (∀ (n : ℕ), (∃ (k : ℕ), (0 < k) ∧ (∃ (q : ℤ), (4 : ℤ) * 10 ^ (n + 12 * k) / 17 % 10 ^ 12 = q)) ∧ r = "235294117647") :=
sorry

end repetend_of_4_div_17_l824_824242


namespace parallel_lines_necessary_not_sufficient_l824_824673

theorem parallel_lines_necessary_not_sufficient {a : ℝ} 
  (h1 : ∀ x y : ℝ, a * x + (a + 2) * y + 1 = 0) 
  (h2 : ∀ x y : ℝ, x + a * y + 2 = 0) 
  (h3 : ∀ x y : ℝ, a * (1 * y + 2) = 1 * (a * y + 2)) : 
  (a = -1) -> (a = 2 ∨ a = -1 ∧ ¬(∀ b, a = b → a = -1)) :=
by
  -- proof goes here
  sorry

end parallel_lines_necessary_not_sufficient_l824_824673


namespace find_x_l824_824423

variable (A B x : ℝ)
variable (h1 : A > 0) (h2 : B > 0)
variable (h3 : A = (x / 100) * B)

theorem find_x : x = 100 * (A / B) :=
by
  sorry

end find_x_l824_824423


namespace least_int_gt_sqrt_500_l824_824054

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824054


namespace least_integer_greater_than_sqrt_500_l824_824092

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824092


namespace sum_of_roots_of_quadratic_eq_l824_824963

theorem sum_of_roots_of_quadratic_eq :
  ∀ x : ℝ, x^2 + 2023 * x - 2024 = 0 → 
  x = -2023 := 
sorry

end sum_of_roots_of_quadratic_eq_l824_824963


namespace min_value_of_a_plus_b_find_min_value_of_a_plus_b_l824_824640

theorem min_value_of_a_plus_b (a b : ℝ) (h : a^2 + b^2 = 1) : a + b ≥ -real.sqrt 2 := by
  sorry

theorem find_min_value_of_a_plus_b (a b : ℝ) (h : a^2 + b^2 = 1) : ∃ (x : ℝ), x = a + b ∧ x = -real.sqrt 2 := by
  exists ((-real.sqrt 2) : ℝ)
  split
  { sorry }
  { sorry }


end min_value_of_a_plus_b_find_min_value_of_a_plus_b_l824_824640


namespace prove_solution_set_correct_l824_824653

-- Conditions
variables (f : ℝ → ℝ) (h_dom : ∀ x, x ∈ ℝ) (h_f_neg1 : f (-1) = 2) (h_f_derivative : ∀ x, deriv f x > 2)

-- Definition of the proposition that the solution set of f(x) > 2x + 4 is (-1, +∞)
def solution_set_correct : Prop :=
  ∀ x, f x > 2 * x + 4 ↔ x > -1

-- Proof statement (without proof)
theorem prove_solution_set_correct : solution_set_correct f :=
by {
  sorry
}

end prove_solution_set_correct_l824_824653


namespace largest_n_for_15_divisor_of_30_l824_824597

theorem largest_n_for_15_divisor_of_30! : 
  ∃ n : ℕ, (15 ^ n ∣ nat.factorial 30) ∧ ∀ m : ℕ, (15 ^ m ∣ nat.factorial 30) → m ≤ 7 :=
begin
  sorry
end

end largest_n_for_15_divisor_of_30_l824_824597


namespace least_integer_greater_than_sqrt_500_l824_824083

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824083


namespace proof_smallest_beta_l824_824369

variables {m n p q : EuclideanSpace ℝ (fin 3)}
variables (β : ℝ)
variables [UnitVector m] [UnitVector n] [UnitVector p] [UnitVector q]
variables (hnq : dot_product n q = 0) -- q orthogonal to n
variables (hnpm : scalarTripleProduct n p m = 1/3) -- scalar triple product condition

noncomputable def smallest_beta : ℝ :=
  0.5 * real.arcsin (2/3)

theorem proof_smallest_beta :
  ∃ β : ℝ, β = smallest_beta ∧ β ≈ 20.91 :=
sorry

end proof_smallest_beta_l824_824369


namespace sum_of_reciprocals_of_roots_l824_824609

theorem sum_of_reciprocals_of_roots (p : ℤ → ℤ) (h : p = λ x, x^2 - 17 * x + 8) :
  let roots := [17, 8] in
  (roots.head + roots.tail.head) / (roots.head * roots.tail.head) = 17 / 8 := 
by {
  sorry
}

end sum_of_reciprocals_of_roots_l824_824609


namespace fish_per_black_duck_l824_824613

theorem fish_per_black_duck :
  ∀ (W_d B_d M_d : ℕ) (fish_per_W fish_per_M total_fish : ℕ),
    (fish_per_W = 5) →
    (fish_per_M = 12) →
    (W_d = 3) →
    (B_d = 7) →
    (M_d = 6) →
    (total_fish = 157) →
    (total_fish - (W_d * fish_per_W + M_d * fish_per_M)) = 70 →
    (70 / B_d) = 10 :=
by
  intros W_d B_d M_d fish_per_W fish_per_M total_fish hW hM hW_d hB_d hM_d htotal_fish hcalculation
  sorry

end fish_per_black_duck_l824_824613


namespace probability_of_alternating_colors_l824_824160

theorem probability_of_alternating_colors :
  ∃ p : ℚ, p = 1 / 126 ∧ 
  (∀ (balls : list ℕ), balls = [5, 5] → 
  (∃ seq : list (ℕ × ℕ), (∀ (ball : ℕ × ℕ), 
    ball.1 = 0 ∨ ball.1 = 1) ∧ 
    (alt_colors seq) ∧ 
    (starts_ends_with_same_color seq) → 
    (length seq = 10) ∧ (number_of (seq, 0) = 5) ∧ (number_of (seq, 1) = 5) ∧ (prob seq = p))) :=
begin
  sorry
end

end probability_of_alternating_colors_l824_824160


namespace original_cost_price_l824_824515

theorem original_cost_price (C : ℝ) (h1 : S = 1.25 * C) (h2 : C_new = 0.80 * C) 
    (h3 : S_new = 1.25 * C - 14.70) (h4 : S_new = 1.04 * C) : C = 70 := 
by {
  sorry
}

end original_cost_price_l824_824515


namespace IJ_eq_AH_l824_824769

open EuclideanGeometry

variables {A B C H G I J : Point}

-- Given conditions
axiom triangle_with_orthocenter : triangle A B C → orthocenter H A B C
axiom parallelogram_ABGH : parallelogram A B G H
axiom I_on_GH_bisected_by_AC : on_line I G H → midpoint (bisection_point AC I H)
axiom AC_intersects_circumcircle_at_J : intersects_at_circumcircle AC (circumcircle G C I) J

-- Prove statement
theorem IJ_eq_AH : 
  ∀ (A B C H G I J : Point), 
    triangle A B C → 
    orthocenter H A B C → 
    parallelogram A B G H → 
    on_line I G H → 
    midpoint (bisection_point AC I H) → 
    intersects_at_circumcircle AC (circumcircle G C I) J → 
    distance I J = distance A H := 
by 
  sorry

end IJ_eq_AH_l824_824769


namespace cost_of_four_dozen_apples_l824_824536

theorem cost_of_four_dozen_apples (cost_two_dozen : ℝ) (h : cost_two_dozen = 15.60) : ∃ cost_four_dozen, cost_four_dozen = 31.20 :=
by
  have cost_per_dozen := cost_two_dozen / 2
  have cost_four_dozen := 4 * cost_per_dozen
  use cost_four_dozen
  rw h
  norm_num
  exact eq.refl 31.20

end cost_of_four_dozen_apples_l824_824536


namespace find_point_C_l824_824642

-- Define the points A and B, and the condition that B is the midpoint of AC
def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def A : ℝ × ℝ × ℝ := point 0 2 2
def B : ℝ × ℝ × ℝ := point 1 1 1

def is_midpoint (A B C : ℝ × ℝ × ℝ) :=
  B = point ((A.1 + C.1) / 2) ((A.2 + C.2) / 2) ((A.3 + C.3) / 2)

theorem find_point_C :
  ∃ C : ℝ × ℝ × ℝ,
    is_midpoint A B C ∧ C = point 2 0 0 :=  by
  sorry

end find_point_C_l824_824642


namespace proof_equation_of_line_l824_824822
   
   -- Define the point P
   structure Point where
     x : ℝ
     y : ℝ
     
   -- Define conditions
   def passesThroughP (line : ℝ → ℝ → Prop) : Prop :=
     line 2 (-1)
     
   def interceptRelation (line : ℝ → ℝ → Prop) : Prop :=
     ∃ a : ℝ, a ≠ 0 ∧ (∀ x y, line x y ↔ (x / a + y / (2 * a) = 1))
   
   -- Define the line equation
   def line_equation (line : ℝ → ℝ → Prop) : Prop :=
     passesThroughP line ∧ interceptRelation line
     
   -- The final statement
   theorem proof_equation_of_line (line : ℝ → ℝ → Prop) :
     line_equation line →
     (∀ x y, line x y ↔ (2 * x + y = 3)) ∨ (∀ x y, line x y ↔ (x + 2 * y = 0)) :=
   by
     sorry
   
end proof_equation_of_line_l824_824822


namespace minimum_value_condition_l824_824439

variable {α : Type*} [TopologicalSpace α] [LinearOrder α] {f : α → ℝ}
variable {a b : α} [TopologicalSpace α] [h_diff : Differentiable α f]
variable [ContinuityOnIntegral (a, b) f]

theorem minimum_value_condition :
  (∃ x0 ∈ set.Ioo a b, deriv (deriv f) x0 = 0) →
  (∃ x1 ∈ set.Ioo a b, IsMinOn f (set.Icc a b) x1) :=
sorry

end minimum_value_condition_l824_824439


namespace max_temperature_range_l824_824431

noncomputable theory

def average_temperature (temps : List ℕ) : Prop := (temps.sum / temps.length) = 60
def lowest_temperature (temps : List ℕ) : Prop := temps.minimum = some 45

theorem max_temperature_range (temps : List ℕ) :
  average_temperature temps →
  lowest_temperature temps →
  (temps.maximum - temps.minimum) = some 75 :=
by
  sorry

end max_temperature_range_l824_824431


namespace max_plus_min_eq_16_l824_824293

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x + 8

theorem max_plus_min_eq_16 :
  let I : Set ℝ := Set.Icc (-3 : ℝ) (3 : ℝ)
  let M : ℝ := Sup (f '' I)
  let m : ℝ := Inf (f '' I)
  M + m = 16 :=
by
  let I : Set ℝ := Set.Icc (-3 : ℝ) (3 : ℝ)
  let M : ℝ := Sup (f '' I)
  let m : ℝ := Inf (f '' I)
  sorry

end max_plus_min_eq_16_l824_824293


namespace boats_distance_one_minute_before_collision_l824_824492

noncomputable def distance_between_boats_one_minute_before_collision
  (speed_boat1 : ℝ) (speed_boat2 : ℝ) (initial_distance : ℝ) : ℝ :=
  let relative_speed := speed_boat1 + speed_boat2
  let relative_speed_per_minute := relative_speed / 60
  let time_to_collide := initial_distance / relative_speed_per_minute
  let distance_one_minute_before := initial_distance - (relative_speed_per_minute * (time_to_collide - 1))
  distance_one_minute_before

theorem boats_distance_one_minute_before_collision :
  distance_between_boats_one_minute_before_collision 5 21 20 = 0.4333 :=
by
  -- Proof skipped
  sorry

end boats_distance_one_minute_before_collision_l824_824492


namespace value_of_1_plus_i_cubed_l824_824145

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- The main statement to verify
theorem value_of_1_plus_i_cubed : (1 + i ^ 3) = (1 - i) :=
by {  
  -- Use given conditions here if needed
  sorry
}

end value_of_1_plus_i_cubed_l824_824145


namespace cos_beta_value_l824_824317

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
    (h1 : Real.sin α = 3/5) (h2 : Real.cos (α + β) = 5/13) : 
    Real.cos β = 56/65 := 
by
  sorry

end cos_beta_value_l824_824317


namespace least_integer_greater_than_sqrt_500_l824_824064

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824064


namespace problem1_problem2_l824_824496

-- For Problem (1)
theorem problem1 : sqrt 3 - sqrt 3 * (1 - sqrt 3) = 3 := 
by 
  sorry

-- For Problem (2)
theorem problem2 : (sqrt 3 - 2) ^ 2 + sqrt 12 + 6 * sqrt (1 / 3) = 7 := 
by 
  sorry

end problem1_problem2_l824_824496


namespace infinite_pairs_m_n_l824_824408

theorem infinite_pairs_m_n :
  ∃ (f : ℕ → ℕ × ℕ), (∀ k, (f k).1 > 0 ∧ (f k).2 > 0 ∧ ((f k).1 ∣ (f k).2 ^ 2 + 1) ∧ ((f k).2 ∣ (f k).1 ^ 2 + 1)) :=
sorry

end infinite_pairs_m_n_l824_824408


namespace number_dislike_both_l824_824787

/-- Define the total number of people polled -/
def total_people := 1500

/-- Define the proportion of people who do not like radio -/
def proportion_dislike_radio := 0.40

/-- Define the proportion of people who do not like both radio and music among those who do not like radio -/
def proportion_dislike_both := 0.15

/-- The theorem stating the number of people who do not like both radio and music -/
theorem number_dislike_both :
  let num_dislike_radio := proportion_dislike_radio * total_people in
  let num_dislike_both := proportion_dislike_both * num_dislike_radio in
  num_dislike_both = 90 := 
by
  sorry

end number_dislike_both_l824_824787


namespace probability_no_shaded_square_l824_824152

theorem probability_no_shaded_square :
  let num_rects := 1003 * 2005
  let num_rects_with_shaded := 1002^2
  let probability_no_shaded := 1 - (num_rects_with_shaded / num_rects)
  probability_no_shaded = 1 / 1003 := by
  -- The proof steps go here
  sorry

end probability_no_shaded_square_l824_824152


namespace quadratic_inequality_k_quadratic_root_zero_l824_824296

open Real

variables (k x: ℝ)

theorem quadratic_inequality_k (h1 : ∀ x, x^2 + 2*(k-1)*x + k^2 - 1 = 0) :
  k < 1 :=
by {
  let Δ := (2*(k - 1))^2 - 4*(k^2 - 1),
  have hΔ_pos : Δ > 0 := by sorry,
  exact calc
    Δ = 4*k^2 - 8*k + 4 + 4 - 4*k^2 : by ring
    ... = -8*k + 8 : by ring
    ... > 0 : hΔ_pos,
}

theorem quadratic_root_zero (h1 : ∀ x, x^2 + 2*(k-1)*x + k^2 - 1 = 0) :
  0 = x ∧ k < 1 → (k = -1 ∧ x = 4) :=
by {
  intros,
  have k_eq : k = -1 ∨ k = 1,
  { have : k^2 - 1 = 0 := by sorry,
    exact eq_or_eq_of_eq_or_eq (pow_eq_zero_iff_eq_zero.mpr this) },
  cases k_eq with k_n1 k_1,
  { rw k_n1,
    replace h : x^2 - 4*x = 0 := by sorry,
    exact (sq_eq_zero_iff_eq_zero.mp this).ax },
  { exfalso,
    have : k < 1 := by sorry,
    exact not_le_of_gt this (le_of_eq k_1) },
}

end quadratic_inequality_k_quadratic_root_zero_l824_824296


namespace least_integer_greater_than_sqrt_500_l824_824000

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824000


namespace set_complement_intersection_subset_condition_l824_824300

open Set

variable {R : Type} [LinearOrder R]

theorem set_complement_intersection (A B : Set R) :
  A = {x | 3 ≤ x ∧ x < 7} →
  B = {x | 2 < x ∧ x < 10} →
  (compl A ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
sorry

theorem subset_condition (a : R) (A C : Set R) :
  A = {x | 3 ≤ x ∧ x < 7} →
  C = {x | x < a} →
  A ⊆ C →
  a ≥ 7 :=
sorry

end set_complement_intersection_subset_condition_l824_824300


namespace exists_25_consecutive_odd_numbers_l824_824568

theorem exists_25_consecutive_odd_numbers
    (seq : Fin 25 → ℤ)
    (h1 : ∀ i : Fin 25, seq i = -23 + 2 * ↑i) :
    ∃ (sum_prod_is_square : ∃ (S P : ℤ), S = (Finset.univ.sum seq) ∧ P = (Finset.univ.prod seq) ∧ S = k^2 ∧ P = m^2) :=
begin
    -- Proof is omitted
    sorry
end

end exists_25_consecutive_odd_numbers_l824_824568


namespace least_integer_gt_sqrt_500_l824_824096

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824096


namespace least_integer_greater_than_sqrt_500_l824_824073

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824073


namespace baby_guppies_l824_824189

theorem baby_guppies (x : ℕ) (h1 : 7 + x + 9 = 52) : x = 36 :=
by
  sorry

end baby_guppies_l824_824189


namespace angle_E_is_135_l824_824352

variables (EF GH : Type) [parallel: parallel EF GH]
variables (angle_E angle_H angle_G angle_F : ℝ)

-- Given conditions
axiom cond1 : angle_E = 3 * angle_H
axiom cond2 : angle_G = 2 * angle_F
axiom cond3 : angle_H = 45

theorem angle_E_is_135 
    (parallel_EF_GH : parallel EF GH) 
    (h1 : angle_E = 3 * angle_H) 
    (h2 : angle_G = 2 * angle_F) 
    (h3 : angle_H = 45) : 
    angle_E = 135 := 
by 
    sorry

end angle_E_is_135_l824_824352


namespace units_digit_G100_l824_824221

def G (n : ℕ) := 3 * 2 ^ (2 ^ n) + 2

theorem units_digit_G100 : (G 100) % 10 = 0 :=
by
  sorry

end units_digit_G100_l824_824221


namespace trigonometric_identity_l824_824274

open Real

noncomputable def tan_condition (α : ℝ) : Prop := tan (-α) = 3

theorem trigonometric_identity (α : ℝ) (h : tan_condition α) : 
  (sin α)^2 - sin (2 * α) = 8 * cos (2 * α) :=
sorry

end trigonometric_identity_l824_824274


namespace winning_candidate_percentage_l824_824532

theorem winning_candidate_percentage (total_membership: ℕ)
  (votes_cast: ℕ) (winning_percentage: ℝ) (h1: total_membership = 1600)
  (h2: votes_cast = 525) (h3: winning_percentage = 19.6875)
  : (winning_percentage / 100 * total_membership / votes_cast * 100 = 60) :=
by
  sorry

end winning_candidate_percentage_l824_824532


namespace find_fifth_month_sale_l824_824892

theorem find_fifth_month_sale (s1 s2 s3 s4 s6 A : ℝ) (h1 : s1 = 800) (h2 : s2 = 900) (h3 : s3 = 1000) (h4 : s4 = 700) (h5 : s6 = 900) (h6 : A = 850) :
  ∃ s5 : ℝ, (s1 + s2 + s3 + s4 + s5 + s6) / 6 = A ∧ s5 = 800 :=
by
  sorry

end find_fifth_month_sale_l824_824892


namespace solve_for_z_l824_824684

variable {x y z : ℝ} -- Declare variables x, y, and z as real numbers

theorem solve_for_z (h : 1/x + 1/y = 1/z) : z = x * y / (x + y) :=
by
s  
end solve_for_z_l824_824684


namespace proof_problem_l824_824971

noncomputable def f (n : ℕ) : ℝ :=
  ∑ i in finset.range n, 1 / (i + 1 : ℝ)

theorem proof_problem (n : ℕ) (h : n > 0) : f (2^(n+1)) > (n + 3) / 2 :=
  sorry

end proof_problem_l824_824971


namespace find_ellipse_eq_l824_824193

-- Define the parameters and conditions
def center_origin (x y : ℝ) : Prop := 
  x = 0 ∧ y = 0

def foci_on_x_axis (F1 F2 : ℝ × ℝ) : Prop := 
  F1.snd = 0 ∧ F2.snd = 0 ∧ F1.fst = -c ∧ F2.fst = c

def passes_through (ellipse : ℝ → ℝ → ℝ) (P : ℝ × ℝ) : Prop := 
  ellipse P.fst P.snd = 1

def perpendicular_slopes (F1 F2 P : ℝ × ℝ) : Prop := 
  let k_FP1 := (P.snd - F1.snd) / (P.fst - F1.fst)
  let k_FP2 := (P.snd - F2.snd) / (P.fst - F2.fst)
  k_FP1 * k_FP2 = -1

-- Define the ellipse equation
def ellipse (a b : ℝ) (x y : ℝ) : ℝ := 
  (x^2 / a^2) + (y^2 / b^2)

-- Define the correct equation
def correct_ellipse_eq (x y : ℝ) : ℝ := 
  (x^2 / 45) + (y^2 / 20)

-- Main theorem statement
theorem find_ellipse_eq {c : ℝ}
  (h1 : center_origin 0 0)
  (h2 : ∃ F1 F2 : ℝ × ℝ, foci_on_x_axis F1 F2 ∧ perpendicular_slopes F1 F2 (3, 4))
  (h3 : ∃ a b : ℝ, passes_through (ellipse a b) (3, 4) ∧ a > b ∧ 0 < b)
  : ∀ x y : ℝ, correct_ellipse_eq x y = 1 := 
sorry

end find_ellipse_eq_l824_824193


namespace brenda_has_8_dollars_l824_824147

-- Define the amounts of money each friend has
def emma_money : ℕ := 8
def daya_money : ℕ := emma_money + (25 * emma_money / 100) -- 25% more than Emma's money
def jeff_money : ℕ := 2 * daya_money / 5 -- Jeff has 2/5 of Daya's money
def brenda_money : ℕ := jeff_money + 4 -- Brenda has 4 more dollars than Jeff

-- The theorem stating the final question
theorem brenda_has_8_dollars : brenda_money = 8 :=
by
  sorry

end brenda_has_8_dollars_l824_824147


namespace fifteen_mn_equals_PnQm_l824_824809

-- Definitions of P and Q based on given conditions
variables (m n : ℤ)
def P : ℤ := 2^m
def Q : ℤ := 5^n

-- The statement to prove in Lean
theorem fifteen_mn_equals_PnQm : 15^(m * n) = P n * Q m :=
sorry

end fifteen_mn_equals_PnQm_l824_824809


namespace largest_m_l824_824372

noncomputable def max_min_ab_bc_ca (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a + b + c = 9) (h2 : ab + bc + ca = 27) : ℝ :=
  min (a * b) (min (b * c) (c * a))

theorem largest_m (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : a + b + c = 9) (h2 : ab + bc + ca = 27) : max_min_ab_bc_ca a b c ha hb hc h1 h2 = 6.75 :=
by
  sorry

end largest_m_l824_824372


namespace g_g_g_g_g_2_l824_824773

def g (x : ℝ) : ℝ :=
if x >= 0 then -x^3 else x + 10

theorem g_g_g_g_g_2 : g (g (g (g (g 2)))) = -8 := by
  sorry

end g_g_g_g_g_2_l824_824773


namespace Laura_presentation_non_invited_students_l824_824723

-- Conditions as definitions
def total_students : ℕ := 25
def Laura_included : Prop := 1 = 1 -- since Laura is included in the total count.
def is_friend (a b : ℕ) : Prop := sorry
def invites (laura : ℕ) : set ℕ := 
  {x | is_friend laura x ∨ (∃ y, is_friend laura y ∧ is_friend y x) ∨ 
  (∃ y z, is_friend laura y ∧ is_friend y z ∧ is_friend z x)}

-- Question to prove
theorem Laura_presentation_non_invited_students : 
  ∃ non_invited_count, invites 1 = {x | x ≠ 1} ∧ non_invited_count = 25 - (invites 1).card ∧ non_invited_count = 4 :=
sorry

end Laura_presentation_non_invited_students_l824_824723


namespace arithmetic_sequence_product_l824_824373

theorem arithmetic_sequence_product {b : ℕ → ℤ} (d : ℤ) (h1 : ∀ n, b (n + 1) = b n + d)
    (h2 : b 5 * b 6 = 21) : b 4 * b 7 = -11 :=
  sorry

end arithmetic_sequence_product_l824_824373


namespace angle_XAY_eq_angle_XYM_l824_824336

open EuclideanGeometry

/-- In an acute-angled triangle ABC, a circle with diameter BC intersects sides AB and AC 
    at points E and F respectively. Let M be the midpoint of BC. Line AM intersects EF at 
    point P. Let X be a point on the minor arc EF, and Y be the other intersection point 
    of line XP with the circle. Prove that: ∠XAY = ∠XYM. -/
theorem angle_XAY_eq_angle_XYM 
  (A B C E F M P X Y : Point)
  (h_triangle_acute : Triangle A B C) 
  (h_circle_diameter : Circle_diameter E F B C) 
  (h_midpoint : Midpoint M B C)
  (h_intersect_AM_EF : Line_intersect AM EF P)
  (h_X_minor_arc : Minor_arc_point X E F)
  (h_Y_intersection : Other_intersection_point Y XP) :
  Angle X A Y = Angle X Y M :=
sorry

end angle_XAY_eq_angle_XYM_l824_824336


namespace ratio_AD_BC_l824_824634

-- Defining the conditions of the problem
variables (A B C D M : Type*) [trapezoid A B C D] 
variables (ratioCM_MD ratioCN_NA : ℚ) 
variables (pointM : on_segment CD M)

-- Assumptions based on given conditions
axiom h1 : ratioCM_MD = 4 / 3
axiom h2 : ratioCN_NA = 4 / 3

-- Goal: prove that AD / BC = 7 / 12
theorem ratio_AD_BC : AD / BC = 7 / 12 := 
by
  sorry

end ratio_AD_BC_l824_824634


namespace frood_game_least_froods_to_exceed_eating_points_l824_824346

theorem frood_game (n : ℕ) (h : ∑ i in finset.range (n + 1), i = n * (n + 1) / 2) :
  n > 29 → ∑ i in finset.range (n + 1), i > 15 * n :=
sorry

theorem least_froods_to_exceed_eating_points : ∃ n : ℕ, n = 30 ∧ ∀ m : ℕ, m < n → ∑ i in finset.range (m + 1), i ≤ 15 * m :=
sorry

end frood_game_least_froods_to_exceed_eating_points_l824_824346


namespace product_of_possible_values_of_N_l824_824917

theorem product_of_possible_values_of_N (M L N : ℝ) (h1 : M = L + N) (h2 : M - 5 = (L + N) - 5) (h3 : L + 3 = L + 3) (h4 : |(L + N - 5) - (L + 3)| = 2) : 10 * 6 = 60 := by
  sorry

end product_of_possible_values_of_N_l824_824917


namespace plane_divided_by_lines_l824_824747

theorem plane_divided_by_lines (n : ℕ) : 
  ∀ (lines_in_general_position : ∀ i j : ℕ, i < n → j < n → i ≠ j → no_two_parallel (line i) (line j) ∧ no_three_concurrent (line i) (line j) (line k)), 
  plane_divided_regions n = 1 + (n * (n + 1)) / 2 := 
by 
  -- Proof omitted
  sorry

end plane_divided_by_lines_l824_824747


namespace relationship_between_a_b_c_l824_824620

-- Definitions as given in the problem
def a : ℝ := 2 ^ 0.5
def b : ℝ := Real.log 5 / Real.log 2
def c : ℝ := Real.log 10 / Real.log 4

-- Statement we need to prove
theorem relationship_between_a_b_c : a < c ∧ c < b := by
  sorry

end relationship_between_a_b_c_l824_824620


namespace certain_number_l824_824717

theorem certain_number (x : ℝ) (h : x - 4 = 2) : x^2 - 3 * x = 18 :=
by
  -- Proof yet to be completed
  sorry

end certain_number_l824_824717


namespace tangent_line_correct_l824_824436

noncomputable def tangent_line_eqn_at_P : Prop :=
  let f := λ x : ℝ, x^2 + 3 * x
  let P := (1 : ℝ, 4 : ℝ)
  let f' := λ x : ℝ, 2 * x + 3
  ∃ m : ℝ, f' 1 = m ∧ ∃ b : ℝ, (f P.1 = P.2) ∧ (m * P.1 + b = P.2) ∧ (m = 5) ∧ (b = -1) ∧ ∀ x y, y - P.2 = m * (x - P.1) ↔ y = 5 * x - 1

theorem tangent_line_correct : tangent_line_eqn_at_P := by
  sorry

end tangent_line_correct_l824_824436


namespace min_area_and_line_eq_l824_824513

theorem min_area_and_line_eq (a b : ℝ) (l : ℝ → ℝ → Prop)
    (h1 : l 3 2)
    (h2: ∀ x y: ℝ, l x y → (x/a + y/b = 1))
    (h3: a > 0)
    (h4: b > 0)
    : 
    a = 6 ∧ b = 4 ∧ 
    (∀ x y : ℝ, l x y ↔ (4 * x + 6 * y - 24 = 0)) ∧ 
    (∃ min_area : ℝ, min_area = 12) :=
by
  sorry

end min_area_and_line_eq_l824_824513


namespace no_prime_roots_quadratic_eq_l824_824202

theorem no_prime_roots_quadratic_eq (k : ℕ) : 
  ¬ (∃ p q : ℕ, p.prime ∧ q.prime ∧ p + q = 95 ∧ p * q = k) :=
by
  sorry

end no_prime_roots_quadratic_eq_l824_824202


namespace least_integer_gt_sqrt_500_l824_824101

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824101


namespace solve_for_n_l824_824455

theorem solve_for_n : ∃ (n : ℕ), (n^2 + 7 * n = 210) ∧ (n = 12) :=
by {
      use 12,
      split,
      { norm_num },
      sorry
    }

end solve_for_n_l824_824455


namespace lcm_1_to_5_l824_824864

theorem lcm_1_to_5 : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5 = 60 := by
  sorry

end lcm_1_to_5_l824_824864


namespace bound_b_l824_824225

variable {a : ℕ → ℝ} (h : ∀ n, 1 ≤ a n ∧ a n ≤ a (n + 1))

noncomputable def b (n : ℕ) : ℝ :=
∑ k in Finset.range n + 1, (1 - (a (k - 1) / a k)) * (1 / Real.sqrt (a k))

theorem bound_b (n : ℕ) (h : ∀ n, 1 = a 0 ∧ 1 ≤ a n ∧ a n ≤ a (n + 1)) : 
  0 ≤ b n ∧ b n ≤ 2 := by
  sorry  

end bound_b_l824_824225


namespace count_valid_placements_l824_824424

-- Define the conditions
def L_shaped_figure (figure : Type) : Prop :=
  -- Suppose the figure can be considered as a part of a cube with missing faces
  sorry 

def valid_placement (square : Type) (figure : Type) : Prop :=
  -- A square is a valid placement if it can be included in the L-shaped figure to form a topless cubical box
  sorry

-- The theorem to prove
theorem count_valid_placements : 
  ∃ (valid_squares : Finset Square), valid_squares.card = 5 :=
by
  use {A, B, D, F, H}
  sorry

end count_valid_placements_l824_824424


namespace garage_sale_items_l824_824914

-- Definition of conditions
def is_18th_highest (num_highest: ℕ) : Prop := num_highest = 17
def is_25th_lowest (num_lowest: ℕ) : Prop := num_lowest = 24

-- Theorem statement
theorem garage_sale_items (num_highest num_lowest total_items: ℕ) 
  (h1: is_18th_highest num_highest) (h2: is_25th_lowest num_lowest) :
  total_items = num_highest + num_lowest + 1 :=
by
  -- Proof omitted
  sorry

end garage_sale_items_l824_824914


namespace roots_of_quadratic_l824_824647

theorem roots_of_quadratic (a b c : ℝ) (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) :
  ¬ ∃ (x : ℝ), x^2 + (a + b + c) * x + a^2 + b^2 + c^2 = 0 :=
by
  sorry

end roots_of_quadratic_l824_824647


namespace max_value_S_n_l824_824297

theorem max_value_S_n 
  (a : ℕ → ℕ)
  (a1 : a 1 = 2)
  (S : ℕ → ℕ)
  (h : ∀ n, 6 * S n = 3 * a (n + 1) + 4 ^ n - 1) :
  ∃ n, S n = 10 := 
sorry

end max_value_S_n_l824_824297


namespace least_integer_greater_than_sqrt_500_l824_824047

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824047


namespace angle_A_of_inscribed_triangle_l824_824525

theorem angle_A_of_inscribed_triangle 
  (x : ℝ) 
  (h : (x + 50) + (2 * x + 30) + (4 * x - 20) = 360) : 
  (x = 300 / 7) → (2 * 300 / 7 + 30) = 2 * 300 / 7 + 30 → 
  ∠ABC = 405 / 7 :=
by
  sorry

end angle_A_of_inscribed_triangle_l824_824525


namespace external_common_tangent_parallel_to_BD_l824_824343

-- Definitions
variables (A B C D : Point)
variable (Ω : Circle)
variable (I1 I2 : Incircle)

-- Assumptions
axiom cyclic_quadrilateral : ∀ a b c d, inscribed_in_circle a b c d Ω
axiom angle_equality : ∀ a b c, ∠a b c = ∠a d c
axiom incircle_triangle1 : ∀ a b c, is_incircle I1 (Triangle a b c)
axiom incircle_triangle2 : ∀ a b c, is_incircle I2 (Triangle a d c)

-- To Prove
theorem external_common_tangent_parallel_to_BD :
  ∃ (tangent : Line), is_external_common_tangent I1 I2 tangent ∧ parallel tangent (line_through B D) :=
by
  sorry

end external_common_tangent_parallel_to_BD_l824_824343


namespace minimum_positive_period_of_f_axis_of_symmetry_of_f_max_value_of_f_min_value_of_f_l824_824286

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem minimum_positive_period_of_f : ∀ x, f (x + π) = f x := 
by sorry

theorem axis_of_symmetry_of_f : ∃ k : ℤ, ∀ x, f x = f (π * (k : ℝ) / 2 + π / 3) := 
by sorry

theorem max_value_of_f : ∃ x ∈ Icc 0 (π / 2), f x = 3 / 2 := 
by sorry

theorem min_value_of_f : ∃ x ∈ Icc 0 (π / 2), f x = 0 := 
by sorry

end minimum_positive_period_of_f_axis_of_symmetry_of_f_max_value_of_f_min_value_of_f_l824_824286


namespace final_answer_l824_824612

def digit_sum_base (n : ℕ) (b : ℕ) : ℕ :=
  (nat.digits b n).sum

def sum_of_squares_of_digits_base (n : ℕ) (b : ℕ) : ℕ :=
  (nat.digits b n).map (λ x => x * x).sum

def h (n : ℕ) : ℕ :=
  digit_sum_base n 6

def j (n : ℕ) : ℕ :=
  digit_sum_base (h n) 10

def k (n : ℕ) : ℕ :=
  sum_of_squares_of_digits_base (j n) 12

def M : ℕ :=
  nat.find (λ n, k(n) ≥ 10) -- Find the least n such that k(n) >= 10 (base-10 representation)

theorem final_answer : M % 500 = 31 := by
  -- This is the target theorem
  sorry

end final_answer_l824_824612


namespace no_visit_days_in_2021_l824_824397

theorem no_visit_days_in_2021 (n1 n2 n3 n4 : ℕ) (days_in_year : ℕ) :
  n1 = 6 → n2 = 8 → n3 = 9 → n4 = 12 → days_in_year = 365 →
  let visits n := days_in_year / n in
  let lcm_visits a b := days_in_year / Nat.lcm a b in
  let lcm_visits_three a b c := days_in_year / Nat.lcm (Nat.lcm a b) c in
  let lcm_visits_four a b c d := days_in_year / Nat.lcm (Nat.lcm a b) (Nat.lcm c d) in
  let total_visits := visits n1 + visits n2 + visits n3 + visits n4 -
                      (lcm_visits n1 n2 + lcm_visits n1 n3 + lcm_visits n1 n4 + lcm_visits n2 n3 + lcm_visits n2 n4 + lcm_visits n3 n4) +
                      (lcm_visits_three n1 n2 n3 + lcm_visits_three n1 n2 n4 + lcm_visits_three n1 n3 n4 + lcm_visits_three n2 n3 n4) -
                      lcm_visits_four n1 n2 n3 n4 in
  days_in_year - total_visits = 280 :=
begin
  intros,
  sorry
end

end no_visit_days_in_2021_l824_824397


namespace v_2023_eq_8442_l824_824763

def v : ℕ → ℕ
| 0 := 4
| 1 := 5
| 2 := 8
| n := if n % 3 == 0 then 4*(n/3 + 1)^2 
       else if n % 3 == 1 then v(n-1) + 4 else v(n-1) + 4

theorem v_2023_eq_8442 : v 2023 = 8442 :=
sorry

end v_2023_eq_8442_l824_824763


namespace work_together_l824_824507

/--
A can complete the work in 4 days.
B can complete the work in 12 days.
-/
variables (A B : ℝ)
variables (work : ℝ := 1)

/-- 
Proof that A and B working together will finish the work in 3 days.
-/
theorem work_together (hA : A = 1/4) (hB : B = 1/12) : 
  (1 / (A + B)) = 3 := by
  sorry

end work_together_l824_824507


namespace tom_can_go_on_three_rides_l824_824464

def rides_possible (total_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  (total_tickets - spent_tickets) / tickets_per_ride

theorem tom_can_go_on_three_rides :
  rides_possible 40 28 4 = 3 :=
by
  -- proof goes here
  sorry

end tom_can_go_on_three_rides_l824_824464


namespace problem_statement_l824_824881

-- Define the universal set U, and sets A and B
def U : Set ℕ := { n | 1 ≤ n ∧ n ≤ 10 }
def A : Set ℕ := {1, 2, 3, 5, 8}
def B : Set ℕ := {1, 3, 5, 7, 9}

-- Define the complement of set A with respect to U
def complement_U_A : Set ℕ := { n | n ∈ U ∧ n ∉ A }

-- Define the intersection of complement_U_A and B
def intersection_complement_U_A_B : Set ℕ := { n | n ∈ complement_U_A ∧ n ∈ B }

-- Prove the given statement
theorem problem_statement : intersection_complement_U_A_B = {7, 9} := by
  sorry

end problem_statement_l824_824881


namespace least_integer_greater_than_sqrt_500_l824_824045

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824045


namespace count_prime_divisor_sum_up_to_30_l824_824759

open Nat

def is_prime_divisor_sum (n : ℕ) : Prop :=
  Nat.prime (divisorSum n)

theorem count_prime_divisor_sum_up_to_30 : 
  (Finset.range 31).filter (λ n, is_prime_divisor_sum n).card = 5 := 
by {
  sorry
}

end count_prime_divisor_sum_up_to_30_l824_824759


namespace least_integer_greater_than_sqrt_500_l824_824060

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824060


namespace sqrt_500_least_integer_l824_824016

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824016


namespace inequality_solution_l824_824421

theorem inequality_solution 
  (x : ℝ) 
  (h1 : (x + 3) / 2 ≤ x + 2) 
  (h2 : 2 * (x + 4) > 4 * x + 2) : 
  -1 ≤ x ∧ x < 3 := sorry

end inequality_solution_l824_824421


namespace bhanu_income_l824_824200

theorem bhanu_income (I P : ℝ) (h1 : (P / 100) * I = 300) (h2 : (20 / 100) * (I - 300) = 140) : P = 30 := by
  sorry

end bhanu_income_l824_824200


namespace Amy_reroll_probability_l824_824531

def Amy_rolls_three_dice : Type := {x : ℕ | x > 0 ∧ x <= 6} × {x : ℕ | x > 0 ∧ x <= 6} × {x : ℕ | x > 0 ∧ x <= 6}

def reroll_strategy (d1 d2 d3 : ℕ) : Type := 
  {r : ℕ × ℕ × ℕ | (r.1.1 = d1 ∨ r.1.1 ∈ {1,2,3,4,5,6}) ∧ 
                    (r.1.2 = d2 ∨ r.1.2 ∈ {1,2,3,4,5,6}) ∧ 
                    (r.2 = d3 ∨ r.2 ∈ {1,2,3,4,5,6}) 
                    ∧ (r.1.1 + r.1.2 + r.2) > 12}

def probability_reroll_two_dice (d1 d2 d3 : ℕ) : ℝ :=
  if h : ∃ a b : ℕ, (a ∈ {1,2,3,4,5,6} ∧ b ∈ {1,2,3,4,5,6}) ∧ 
                     a + b + d1 > 12 then
    (7 / 72 : ℝ)
  else 0

theorem Amy_reroll_probability : 
  ∀ (d1 d2 d3 : ℕ), 
  (d1 ∈ {1,2,3,4,5,6}) ∧ 
  (d2 ∈ {1,2,3,4,5,6}) ∧ 
  (d3 ∈ {1,2,3,4,5,6}) ∧ 
  probability_reroll_two_dice d1 d2 d3 = (7/72 : ℝ) := 
by 
  sorry

end Amy_reroll_probability_l824_824531


namespace transform_f_to_h_l824_824438

-- Definitions of the piecewise function f
def f (x : ℝ) : ℝ :=
  if x < 1 then 1 - x
  else if x < 3 then real.sqrt (1 - (x - 2)^2)
  else 2 * (x - 3)

-- Define the transformed function h
def h (x : ℝ) : ℝ :=
  2 * f(5 - x)

-- Statement of the problem in Lean 4
theorem transform_f_to_h (x : ℝ) : h(x) = 2 * f(5 - x) :=
by
  sorry

end transform_f_to_h_l824_824438


namespace part_a_part_b_part_c_l824_824878

variables {S : Type*} [normed_space ℝ S]
variables (P : S → S) [is_polynomial P]
variables (p q : S) (ξ : ℝ)

-- We assume the Heisenberg-Weyl commutator relation
axiom commutator_relation : p * q - q * p = 1

-- Part (a) statement
theorem part_a (P' : S → S) [is_derivative P' P] :
  p * P q = P q * p + P' q :=
sorry

-- Part (b) statement
theorem part_b : P (p + q) = (λ (ξ : ℝ), P (p + q + ξ)) :=
sorry

-- Part (c) statement
theorem part_c : ¬∃ n : normed_space S, ∀ x y : S, n (x * y) ≤ n x * n y :=
sorry

end part_a_part_b_part_c_l824_824878


namespace exponent_property_l824_824497

theorem exponent_property :
  4^4 * 9^4 * 4^9 * 9^9 = 36^13 :=
by
  -- Add the proof here
  sorry

end exponent_property_l824_824497


namespace least_integer_gt_sqrt_500_l824_824098

theorem least_integer_gt_sqrt_500 : ∃ n : ℕ, n = 23 ∧ (500 < n * n) ∧ ((n - 1) * (n - 1) < 500) := by
  use 23
  split
  · rfl
  · split
    · norm_num
    · norm_num

end least_integer_gt_sqrt_500_l824_824098


namespace PropositionA_sufficient_not_necessary_PropositionB_l824_824271
open Set

-- Define points E, F, G, H in space
variables (E F G H : Point)

-- Define Proposition A: Points E, F, G, H are not coplanar
def PropositionA : Prop := ¬coplanar {E, F, G, H}

-- Define the line through two points
def line_through (P Q : Point) : Set Point := {R | ∃ a b : ℝ, a * P + b * Q = R}

-- Define Proposition B: The lines EF and GH do not intersect
def PropositionB : Prop := ∀ R : Point, R ∉ (line_through E F ∩ line_through G H)

-- The mathematical problem in Lean: Prove Proposition A is a sufficient but not necessary condition for Proposition B
theorem PropositionA_sufficient_not_necessary_PropositionB (E F G H : Point) :
  PropositionA E F G H → ¬PropositionA E F G H → PropositionB E F G H ∧ ¬ (PropositionB E F G H → PropositionA E F G H) :=
by sorry

end PropositionA_sufficient_not_necessary_PropositionB_l824_824271


namespace least_integer_greater_than_sqrt_500_l824_824078

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824078


namespace no_integer_k_sq_plus_k_plus_one_divisible_by_101_l824_824793

theorem no_integer_k_sq_plus_k_plus_one_divisible_by_101 (k : ℤ) : 
  (k^2 + k + 1) % 101 ≠ 0 := 
by
  sorry

end no_integer_k_sq_plus_k_plus_one_divisible_by_101_l824_824793


namespace complex_eq_zero_l824_824313

theorem complex_eq_zero (x y : ℝ) (h : (x + y - 3) + (x - 4) * complex.I = 0) : x = 4 ∧ y = -1 :=
by 
  have h_re : x + y - 3 = 0 := by sorry
  have h_im : x - 4 = 0 := by sorry
  have hx : x = 4 := by sorry
  have hy : y = -1 := by sorry
  exact ⟨hx, hy⟩

end complex_eq_zero_l824_824313


namespace maximum_performances_student_theater_l824_824855

theorem maximum_performances_student_theater
  (students : Finset ℕ) -- 12 students
  (P : ℕ) -- Number of performances (n)
  (in_performance : students → Finset ℕ) -- Function mapping each student to the set of performances they participate in
  (student_count : ∀ p ∈ P, (in_performance p).card = 6) -- Each performance has 6 students
  (pairwise_distinct : ∀ (p1 p2 : ℕ), p1 ≠ p2 → ((in_performance p1) ∩ (in_performance p2)).card ≤ 2) -- Pairs share at most 2 students
  (students_count : students.card = 12) -- Total 12 students
  : P ≤ 4 := sorry -- Therefore, the largest possible value of P is 4

end maximum_performances_student_theater_l824_824855


namespace least_integer_greater_than_sqrt_500_l824_824002

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824002


namespace log_translation_l824_824824

theorem log_translation (a : ℝ) (h : 1 < a) : (∃ x, (x = 2 ∧ (log a (x - 1) + 1 = 1))) :=
begin
  use 2,
  split,
  { refl, },
  { sorry, }
end

end log_translation_l824_824824


namespace distance_points_to_line_is_sqrt3_plus_1_l824_824146

noncomputable def distance_from_point_to_line : ℝ :=
  let P := (2, -Real.pi / 6)
  let line_polar := λ ρ θ, ρ * Real.sin (θ - Real.pi / 6) = 1
  let P_rectangular := (Real.sqrt 3, -1)  -- converted coordinates of P
  let line_rectangular := λ x y, x - Real.sqrt 3 * y + 2 = 0  -- converted equation of line
  let distance := (Real.abs (Real.sqrt 3 + Real.sqrt 3 + 2)) / (Real.sqrt (1 + 3))
  distance

theorem distance_points_to_line_is_sqrt3_plus_1 :
  distance_from_point_to_line = Real.sqrt 3 + 1 :=
sorry

end distance_points_to_line_is_sqrt3_plus_1_l824_824146


namespace increasing_interval_l824_824288

noncomputable def f (ω x : ℝ) : ℝ := sqrt 3 * sin (2 * ω * x) - cos (2 * ω * x)

theorem increasing_interval (ω : ℝ) (h_ω : 0 < ω ∧ ω < 1)
  (h_passes_through : f ω (π / 6) = 0) :
  ∃ (a b : ℝ), 0 ≤ a ∧ a < b ∧ b ≤ π ∧ (∀ (x : ℝ), a ≤ x ∧ x ≤ b → is_increasing (f ω x)) ∧
    (a = 0 ∧ b = 2 * π / 3) := sorry

end increasing_interval_l824_824288


namespace rhombus_area_and_perimeter_l824_824898

def d1 : ℕ := 18
def d2 : ℕ := 16
def s : ℕ := 10

theorem rhombus_area_and_perimeter :
  (d1 * d2 / 2 = 144) ∧ (4 * s = 40) :=
by
  split
  · sorry
  · sorry

end rhombus_area_and_perimeter_l824_824898


namespace odd_function_periodic_l824_824220

theorem odd_function_periodic (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
                            (h_periodic : ∀ x, f (x + 2) = f x)
                            (h_def : ∀ x, -1 < x ∧ x < 0 → f x = Real.exp (-x)) :
                            f (9 / 2) = - Real.sqrt Real.exp 1 :=
by
  sorry

end odd_function_periodic_l824_824220


namespace least_integer_greater_than_sqrt_500_l824_824066

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824066


namespace least_integer_greater_than_sqrt_500_l824_824075

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824075


namespace unique_solution_n_l824_824247

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem unique_solution_n (h : ∀ n : ℕ, (n > 0) → n^3 = 8 * (sum_digits n)^3 + 6 * (sum_digits n) * n + 1 → n = 17) : 
  n = 17 := 
by
  sorry

end unique_solution_n_l824_824247


namespace cos_330_eq_sqrt3_div_2_l824_824218

theorem cos_330_eq_sqrt3_div_2
    (h1 : ∀ θ : ℝ, Real.cos (2 * Real.pi - θ) = Real.cos θ)
    (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
    Real.cos (11 * Real.pi / 6) = Real.sqrt 3 / 2 :=
by
  -- Proof goes here
  sorry

end cos_330_eq_sqrt3_div_2_l824_824218


namespace evaluated_expression_l824_824966

def greatest_integer (x : ℝ) : ℤ := Real.floor x

theorem evaluated_expression : 
  greatest_integer 6.5 * greatest_integer (2 / 3) + greatest_integer 2 * 7.2 + greatest_integer 8.3 - 6.6 = 9.2 :=
by {
  sorry
}

end evaluated_expression_l824_824966


namespace construct_unit_segment_l824_824226

-- Definitions of the problem
variable (a b : ℝ)

-- Parabola definition
def parabola (x : ℝ) : ℝ := x^2 + a * x + b

-- Statement of the problem in Lean 4
theorem construct_unit_segment
  (h : ∃ x y : ℝ, parabola a b x = y) :
  ∃ (u v : ℝ), abs (u - v) = 1 :=
sorry

end construct_unit_segment_l824_824226


namespace determine_c_l824_824937

noncomputable def is_defined_log_exp (x : ℝ) : Prop :=
  log 2010 (log 2009 (log 2008 (log 2007 (log 2006 x)))) > 0

theorem determine_c :
  ∀ x : ℝ, is_defined_log_exp x ↔ x > 2006^(2007^2008) :=
by
  sorry

end determine_c_l824_824937


namespace least_integer_greater_than_sqrt_500_l824_824114

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824114


namespace chords_triangle_count_l824_824784

-- Defining the problem constraints as a Lean 4 statement
theorem chords_triangle_count (h₁ : ∃ points : Finset ℝ, points.card = 9)
  (h₂ : ∀ (p1 p2 p3 : ℝ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3)
  : ∃ n : ℕ, n = 315500 :=
by
  sorry

end chords_triangle_count_l824_824784


namespace range_of_a_l824_824294

def f (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (-2) a, -5 ≤ f x ∧ f x ≤ 4) ↔ (1 ≤ a ∧ a ≤ 4) :=
sorry

end range_of_a_l824_824294


namespace parallel_lines_from_skew_lines_l824_824923

structure Line3D (α : Type*) :=
(point : α × α × α)
(direction : α × α × α)

def are_parallel (l1 l2 : Line3D ℝ) : Prop :=
  ∃ t1 t2 : ℝ, l1.point + t1 • l1.direction = l2.point + t2 • l2.direction

def are_skew (l1 l2 : Line3D ℝ) : Prop :=
  ¬(are_parallel l1 l2) ∧ ∃ p1 p2 : ℝ × ℝ × ℝ, 
    p1 ∈ l1 ∧ p2 ∈ l2 ∧ p1 ≠ p2 ∧ 
    ∃ n1 n2 : ℝ × ℝ × ℝ, 
      n1 ≠ n2 ∧ 
      ∀ t : ℝ, p1 + t • n1 ∉ l2 ∧ p2 + t • n2 ∉ l1

structure Plane (α : Type*) :=
(point : α × α × α)
(normal : α × α × α)

def contains_line (π : Plane ℝ) (l : Line3D ℝ) : Prop :=
  ∃ t : ℝ, π.point + t • π.normal = l.point

theorem parallel_lines_from_skew_lines
  (l1 l2 : Line3D ℝ) (π1 π2 : Plane ℝ) (k : Line3D ℝ) :
  are_skew l1 l2 →
  contains_line π1 l1 →
  contains_line π2 l2 →
  are_parallel k l1 →
  are_parallel k l2 →
  are_parallel
    { point := l1.point, direction := l1.direction }
    { point := l2.point, direction := l2.direction } :=
sorry

end parallel_lines_from_skew_lines_l824_824923


namespace zmod_field_l824_824379

theorem zmod_field (p : ℕ) [Fact (Nat.Prime p)] : Field (ZMod p) :=
sorry

end zmod_field_l824_824379


namespace dance_team_recruitment_l824_824904

theorem dance_team_recruitment 
  (total_students choir_students track_field_students dance_students : ℕ)
  (h1 : total_students = 100)
  (h2 : choir_students = 2 * track_field_students)
  (h3 : dance_students = choir_students + 10)
  (h4 : total_students = track_field_students + choir_students + dance_students) : 
  dance_students = 46 :=
by {
  -- The proof goes here, but it is not required as per instructions
  sorry
}

end dance_team_recruitment_l824_824904


namespace sum_of_terms_in_arithmetic_sequence_l824_824981

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d : ℝ, ∀ n m : ℕ, a (n + 1) - a n = d ∧ a (n + m) = a n + m * d

theorem sum_of_terms_in_arithmetic_sequence (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 7 = ∫ x in 0..π, Real.sin x) :
  a 4 + a 6 + a 8 = 3 := 
by 
  sorry

end sum_of_terms_in_arithmetic_sequence_l824_824981


namespace calculate_total_earning_for_7_mice_l824_824178

-- Definitions based on given conditions:
def price_per_two_mice : ℝ := 5.34
def price_per_mouse : ℝ := price_per_two_mice / 2
def number_of_mice_sold : ℝ := 7

-- Theorem stating the question and answer:
theorem calculate_total_earning_for_7_mice :
  number_of_mice_sold * price_per_mouse = 18.69 :=
by
  sorry

end calculate_total_earning_for_7_mice_l824_824178


namespace solve_equation_l824_824964

noncomputable def find_x : ℚ :=
  50 / 17

theorem solve_equation : 
  ∀ x : ℚ, (x = find_x) ↔ (√(8 * x) / √(4 * (x - 2)) = 5 / 2) := 
by 
  sorry

end solve_equation_l824_824964


namespace product_of_roots_leq_l824_824750

theorem product_of_roots_leq (n : ℕ) (n_ge_1 : 1 ≤ n) (a : Fin n → ℝ) 
  (f : ℝ → ℝ) 
  (h₁ : f x = x ^ n + (∑ i in range (n - 1), a i * x ^ i) ∀ x) 
  (h₂ : |f 0| = f 1) 
  (roots : Fin n → ℝ) 
  (roots_in_interval : ∀ i, roots i ∈ Icc 0 1) 
  (h : f x = (∏ i in range n, (x - roots i)) ∀ x) : 
  (∏ i in range n, roots i) ≤ 1 / 2 ^ n :=
sorry

end product_of_roots_leq_l824_824750


namespace pencils_are_left_l824_824910

-- Define the conditions
def original_pencils : ℕ := 87
def removed_pencils : ℕ := 4

-- Define the expected outcome
def pencils_left : ℕ := original_pencils - removed_pencils

-- Prove that the number of pencils left in the jar is 83
theorem pencils_are_left : pencils_left = 83 := by
  -- Placeholder for the proof
  sorry

end pencils_are_left_l824_824910


namespace photograph_perimeter_is_23_l824_824180

noncomputable def photograph_perimeter (w h m : ℝ) : ℝ :=
if (w + 4) * (h + 4) = m ∧ (w + 8) * (h + 8) = m + 94 then 2 * (w + h) else 0

theorem photograph_perimeter_is_23 (w h m : ℝ) 
    (h₁ : (w + 4) * (h + 4) = m) 
    (h₂ : (w + 8) * (h + 8) = m + 94) : 
    photograph_perimeter w h m = 23 := 
by 
  sorry

end photograph_perimeter_is_23_l824_824180


namespace subtraction_problem_l824_824807

variable (x : ℕ) -- Let's assume x is a natural number for this problem

theorem subtraction_problem (h : x - 46 = 15) : x - 29 = 32 := 
by 
  sorry -- Proof to be filled in

end subtraction_problem_l824_824807


namespace equation_of_line_AB_l824_824314

theorem equation_of_line_AB
    (P : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (h : P = (2, -1))
    (circle_eq : ∀ (M : ℝ × ℝ), (M.1 - 1)^2 + M.2^2 = 25 → 
        line_through A B M)
    (midpoint_eq : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
    ∃ (m b : ℝ), line_eq (x, y) = m * x + b → (x - y - 3 = 0) := 
by 
    sorry

end equation_of_line_AB_l824_824314


namespace eq_zero_of_sqrt_eq_l824_824126

theorem eq_zero_of_sqrt_eq (x y z : ℝ) (h : sqrt (x - y + z) = sqrt x - sqrt y + sqrt z) : 
  (x - y) * (y - z) * (z - x) = 0 := 
by 
  sorry

end eq_zero_of_sqrt_eq_l824_824126


namespace Eric_return_time_l824_824951

theorem Eric_return_time (t1 t2 t_return : ℕ) 
  (h1 : t1 = 20) 
  (h2 : t2 = 10) 
  (h3 : t_return = 3 * (t1 + t2)) : 
  t_return = 90 := 
by 
  sorry

end Eric_return_time_l824_824951


namespace min_fence_length_l824_824467

theorem min_fence_length (length width : ℕ) (h_length : length = 32) (h_width : width = 14) : 
    2 * width + length = 60 :=
by
  rw [h_length, h_width]
  sorry

end min_fence_length_l824_824467


namespace unpainted_unit_cubes_l824_824505

theorem unpainted_unit_cubes (n : ℕ) : n = 6 → 
(∀ k : ℕ, 1 ≤ k ∧ k ≤ 6 → painted_squares_per_face = 36 - 4) →
(∀ l : ℕ, 1 ≤ l ∧ l ≤ 6 → total_painted_squares_without_overlap = 32 * 6) →
(∀ m : ℕ, 1 ≤ m ∧ m ≤ 12 → double_counted_squares = 4 * m) →
(∀ p : ℕ, painted_cubes = total_painted_squares_without_overlap - double_counted_squares) →
unpainted_unit_cubes = 216 - painted_cubes → unpainted_unit_cubes = 72 :=
by
  sorry

end unpainted_unit_cubes_l824_824505


namespace find_a_b_find_x_range_l824_824630

-- Part (I)
theorem find_a_b (a b : ℝ)
  (h1 : ∀ x : ℝ, ax^2 - (a + 1) * x + b < 0 ↔ (x < -1/2 ∨ x > 1)) : 
  a = -2 ∧ b = 1 :=
sorry

-- Part (II)
theorem find_x_range (a b : ℝ) (m : ℝ)
  (h_a : a = -2) (h_b : b = 1)
  (h2 : ∀ m : ℝ, 0 ≤ m → m ≤ 4 → ∀ x, x^2 + (m - 4) * x + 3 - m ≥ 0) : 
  (∀ x : ℝ, (x ∈ set.Iic (-1) ∨ x = 1 ∨ x ∈ set.Ici 3)) :=
sorry

end find_a_b_find_x_range_l824_824630


namespace angle_measure_l824_824120

variable (x : ℝ)

noncomputable def is_supplement (x : ℝ) : Prop := 180 - x = 3 * (90 - x) - 60

theorem angle_measure : is_supplement x → x = 15 :=
by
  sorry

end angle_measure_l824_824120


namespace ones_digit_of_largest_power_of_two_dividing_factorial_l824_824600

theorem ones_digit_of_largest_power_of_two_dividing_factorial :
  let n := 2^5 in
  let k := (2 * ((n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32))) in
  let pow_ones_digit := 2 ^ ((k : ℕ) mod 4) in
  (match pow_ones_digit % 10 with
   | 0 => 6
   | n => n
    end) = 8 := by
  sorry

end ones_digit_of_largest_power_of_two_dividing_factorial_l824_824600


namespace remove_50_arrows_l824_824338

structure Board :=
    (arrows : Array (Array Direction))
    (size : arrows.size = 10)
    (subsize : ∀ i, (arrows[i]).size = 10)

inductive Direction
| up
| down
| left
| right

def can_remove_50_without_pairs (board : Board) : Prop :=
  ∃ removed_arrows : Finset (Fin 10 × Fin 10),
  removed_arrows.card = 50 ∧
  ∀ i j : Fin 10, (i, j) ∉ removed_arrows →
    ∀ k l : Fin 10, (k, l) ∉ removed_arrows →
      (i ≠ k ∨ j ≠ l) ∧
      (
        (board.arrows[i][j] = Direction.up → board.arrows[k][l] ≠ Direction.down) ∧
        (board.arrows[i][j] = Direction.down → board.arrows[k][l] ≠ Direction.up) ∧
        (board.arrows[i][j] = Direction.left → board.arrows[k][l] ≠ Direction.right) ∧
        (board.arrows[i][j] = Direction.right → board.arrows[k][l] ≠ Direction.left)
      )

theorem remove_50_arrows : ∀ board : Board, can_remove_50_without_pairs board :=
sorry

end remove_50_arrows_l824_824338


namespace hudson_daily_burger_spending_l824_824198

-- Definitions based on conditions
def total_spent := 465
def days_in_december := 31

-- Definition of the question
def amount_spent_per_day := total_spent / days_in_december

-- The theorem to prove
theorem hudson_daily_burger_spending : amount_spent_per_day = 15 := by
  sorry

end hudson_daily_burger_spending_l824_824198


namespace binom_1300_2_eq_l824_824551

theorem binom_1300_2_eq : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_eq_l824_824551


namespace sparrows_on_fence_l824_824746

-- Define the number of sparrows initially on the fence
def initial_sparrows : ℕ := 2

-- Define the number of sparrows that joined later
def additional_sparrows : ℕ := 4

-- Define the number of sparrows that flew away
def sparrows_flew_away : ℕ := 3

-- Define the final number of sparrows on the fence
def final_sparrows : ℕ := initial_sparrows + additional_sparrows - sparrows_flew_away

-- Prove that the final number of sparrows on the fence is 3
theorem sparrows_on_fence : final_sparrows = 3 := by
  sorry

end sparrows_on_fence_l824_824746


namespace length_of_escalator_l824_824533

-- Given conditions
def escalator_speed : ℝ := 12 -- ft/sec
def person_speed : ℝ := 8 -- ft/sec
def time : ℝ := 8 -- seconds

-- Length of the escalator
def length : ℝ := 160 -- feet

-- Theorem stating the length of the escalator given the conditions
theorem length_of_escalator
  (h1 : escalator_speed = 12)
  (h2 : person_speed = 8)
  (h3 : time = 8)
  (combined_speed := escalator_speed + person_speed) :
  combined_speed * time = length :=
by
  -- Here the proof would go, but it's omitted as per instructions
  sorry

end length_of_escalator_l824_824533


namespace cone_volume_approx_l824_824449

-- Define the key terms and conditions
def radius_cm : ℝ := 3
def height_cm : ℝ := 4
def pi_approx : ℝ := 3.14159

-- Define the formula for the volume of the cone
def cone_volume (r h : ℝ) := (1/3) * π * r^2 * h

-- Prove the volume of the cone is approximately 37.69908 cubic centimeters
theorem cone_volume_approx :
  cone_volume radius_cm height_cm ≈ 37.69908 :=
  by
  -- Setup and calculations
  sorry

end cone_volume_approx_l824_824449


namespace least_integer_greater_than_sqrt_500_l824_824091

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824091


namespace symmetric_circle_l824_824837

theorem symmetric_circle (x y : ℝ) :
  let C := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1 }
  let L := { p : ℝ × ℝ | p.1 + p.2 = 1 }
  ∃ C' : ℝ × ℝ → Prop, (∀ p, C' p ↔ (p.1)^2 + (p.2)^2 = 1) :=
sorry

end symmetric_circle_l824_824837


namespace tens_digit_of_11_pow_12_pow_13_l824_824610

theorem tens_digit_of_11_pow_12_pow_13 :
  let n := 12^13
  let t := 10
  let tens_digit := (11^n % 100) / 10 % 10
  tens_digit = 2 :=
by 
  let n := 12^13
  let t := 10
  let tens_digit := (11^n % 100) / 10 % 10
  show tens_digit = 2
  sorry

end tens_digit_of_11_pow_12_pow_13_l824_824610


namespace divisors_of_3d_plus_15_l824_824812

-- Conditions: c and d are integers such that 4d = 10 - 3c
variables (c d : ℤ)
hypothesis h : 4 * d = 10 - 3 * c

-- Goal: To prove that exactly 4 out of the first five positive integers are always divisors of 3d + 15
theorem divisors_of_3d_plus_15 : ∃ (d : ℤ), 4d = 10 - 3c → 
  let n := 3 * d + 15 in (∀ (k : ℕ), k ∈ {1, 2, 3, 4} → k ∣ n) ∧ ¬(5 ∣ n) :=
begin
  -- proof will be inserted here
  sorry
end

end divisors_of_3d_plus_15_l824_824812


namespace distinct_values_of_c_l824_824770

theorem distinct_values_of_c {c p q : ℂ} 
  (h_distinct : p ≠ q) 
  (h_eq : ∀ z : ℂ, (z - p) * (z - q) = (z - c * p) * (z - c * q)) :
  (∃ c_values : ℕ, c_values = 2) :=
sorry

end distinct_values_of_c_l824_824770


namespace find_50th_element_largest_element_with_sum_2014_smallest_element_with_sum_2014_l824_824753
-- Import the necessary library

-- Define the set of palindromic integers of the form 5n + 4
def is_palindromic (x : ℕ) : Prop := 
  let s := x.to_digits 10 in
  s = s.reverse

def M : set ℕ := {y | ∃ n : ℕ, y = 5 * n + 4 ∧ is_palindromic y}

-- The 50th element in the set M in increasing order
theorem find_50th_element : ∃ x ∈ M, true := sorry

-- Define conditions for digit sum
def digit_sum (x : ℕ) : ℕ := x.to_digits 10 |>.foldl (+) 0

-- Find the largest element in M with digit sum 2014
theorem largest_element_with_sum_2014 : 
  ∃ x ∈ M, digit_sum x = 2014 ∧ ∀ y ∈ M, digit_sum y = 2014 → x ≥ y := sorry

-- Find the smallest element in M with digit sum 2014
theorem smallest_element_with_sum_2014 :
  ∃ x ∈ M, digit_sum x = 2014 ∧ ∀ y ∈ M, digit_sum y = 2014 → x ≤ y := sorry

end find_50th_element_largest_element_with_sum_2014_smallest_element_with_sum_2014_l824_824753


namespace apples_difference_l824_824777

theorem apples_difference :
  ∀ (total_apples first_day_fraction second_day_multiplier remaining_apples : ℕ),
  total_apples = 200 →
  first_day_fraction = 1 / 5 →
  second_day_multiplier = 2 →
  remaining_apples = 20 →
  let first_day_pick := total_apples * first_day_fraction,
      second_day_pick := second_day_multiplier * first_day_pick,
      total_pick_first_two_days := first_day_pick + second_day_pick,
      total_pick_by_end_third_day := total_apples - remaining_apples,
      third_day_pick := total_pick_by_end_third_day - total_pick_first_two_days,
      result := third_day_pick - first_day_pick
  in result = 20 :=
by {
  intros total_apples first_day_fraction second_day_multiplier remaining_apples htott hfraction hmultiplier hremain,
  let first_day_pick := total_apples * first_day_fraction,
  let second_day_pick := second_day_multiplier * first_day_pick,
  let total_pick_first_two_days := first_day_pick + second_day_pick,
  let total_pick_by_end_third_day := total_apples - remaining_apples,
  let third_day_pick := total_pick_by_end_third_day - total_pick_first_two_days,
  let result := third_day_pick - first_day_pick,
  sorry
}

end apples_difference_l824_824777


namespace convex_m_gon_l824_824774

open BigOperators

theorem convex_m_gon (n m : ℕ) (h_nm : n ≥ m) (h_m5 : m ≥ 5) : 
  ∃ P : Finset (Fin (2 * n + 1)), 
  (P.card = 2 * n + 1) ∧
  ∀ (k : Finset (Fin (2 * n + 1))), 
    k.card = m → 
    (∃ acute_count : ℕ, 
      acute_count ≥ 1 ∧
      acute_count ≤ 2 ∧
      (acute_count = 2 → adjacent_angles_acute k)) → 
    (2 * n + 1) * (n.choose (m - 2) - (n + 1).choose (m - 1)) = 
      ∑ (k : ℕ) in Finset.range (2 * n + 1), 
        (2 * n + 1) * (choose n (m - 2) - choose (n + 1) (m - 1)) :=
sorry

end convex_m_gon_l824_824774


namespace ellipse_equation_l824_824143

variable (F1 F2 : ℝ × ℝ)
variable (a b c : ℝ)
variable (A B : ℝ × ℝ)
variable (e : ℝ)

-- Define the conditions
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def is_chord (F2 : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := 
  A = F1 ∧ B = F2 ∧ (∃ k : ℝ, k * F2.fst = F2.snd)

def perimeter (A F1 B : ℝ × ℝ) : Prop := 
  (A.1 - F1.1)^2 + (A.2 - F1.2)^2 + (F1.1 - B.1)^2 + (F1.2 - B.2)^2 = 16

def eccentricity (e c a : ℝ) : Prop := e = c / a

-- The statement to prove
theorem ellipse_equation (h1 : a > b)
  (h2 : b > 0)
  (h3 : is_chord F2 A B)
  (h4 : perimeter A F1 B)
  (h5 : eccentricity e c a)
  (h6 : e = (√3) / 2)
  (h7 : 4 * a = 16)
  : a = 4 ∧ b^2 = 4 ∧ (ellipse a b = λ x y => x^2 / 16 + y^2 / 4 = 1) :=
by
  sorry

end ellipse_equation_l824_824143


namespace net_effect_on_sale_value_l824_824874

theorem net_effect_on_sale_value (P Q : ℝ) (hP : P > 0) (hQ : Q > 0) :
  let original_sale_value := P * Q
  let new_price := 0.82 * P
  let new_quantity := 1.88 * Q
  let new_sale_value := new_price * new_quantity
  let net_effect := (new_sale_value / original_sale_value - 1) * 100
  net_effect = 54.16 :=
by
  sorry

end net_effect_on_sale_value_l824_824874


namespace find_a_range_l824_824664

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 / 4 * x + 1 else Real.log x

theorem find_a_range : 
  {a : ℝ | ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = a * x1 ∧ f x2 = a * x2} = [1 / 4, 1 / Real.e) := 
sorry

end find_a_range_l824_824664


namespace at_least_one_hit_l824_824569

-- Introduce the predicates
variable (p q : Prop)

-- State the theorem
theorem at_least_one_hit : (¬ (¬ p ∧ ¬ q)) = (p ∨ q) :=
by
  sorry

end at_least_one_hit_l824_824569


namespace evaluate_expression_l824_824477

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := by
  sorry

end evaluate_expression_l824_824477


namespace max_value_expression_l824_824691

theorem max_value_expression (x : ℝ) (h : x > 0) : 2 - x - 4 / x ≤ -2 :=
begin
  -- "proof here"
  sorry
end

end max_value_expression_l824_824691


namespace multiple_of_669_l824_824583

theorem multiple_of_669 (k : ℕ) (h : ∃ a : ℤ, 2007 ∣ (a + k : ℤ)^3 - a^3) : 669 ∣ k :=
sorry

end multiple_of_669_l824_824583


namespace sum_five_smallest_alphas_l824_824608

noncomputable def Q (x : ℂ) : ℂ := 
  (1 + x + x^2 + ⋯ + x^20)^2 - x^15

-- Given conditions for the problem:
axiom exists_zeros_of_Q :
  ∃ (α: ℕ → ℝ), 
    (∀ k, 0 < α k ∧ α k < 1) ∧
    (∀ k < 34, Q (complex.exp (2 * real.pi * I * α k)) = 0) ∧
    (∀ k r, r > 0)

-- Given specific roots:
def alpha_values : list ℝ := [1 / 21, 1 / 15, 2 / 21, 2 / 15, 3 / 21]

axiom alpha_distinct_sorted : 
  list.sorted (≤) alpha_values ∧ (list.nodup alpha_values)

-- Prove that the sum of the first 5 smallest alpha values equals 11/35:
theorem sum_five_smallest_alphas : 
  list.sum (list.take 5 alpha_values) = 11 / 35 :=
sorry

end sum_five_smallest_alphas_l824_824608


namespace real_roots_range_of_k_l824_824704

theorem real_roots_range_of_k (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 - 2 * k * x + (k + 3) = 0) ↔ (k ≤ 3 / 2) :=
sorry

end real_roots_range_of_k_l824_824704


namespace necessary_but_not_sufficient_l824_824650

open Real

def condition1 (x : ℝ) : Prop := (1 / x < 1)
def condition2 (x : ℝ) : Prop := ((1/2)^x > 1)

theorem necessary_but_not_sufficient {x : ℝ} (h₁ : condition2 x → condition1 x) (h₂ : ∃ x, condition1 x ∧ ¬ condition2 x) :
  (condition1 x ∧ ¬ condition2 x) ∨ (¬ condition1 x ∧ condition2 x) :=
begin
  sorry
end

end necessary_but_not_sufficient_l824_824650


namespace cryptarithm_no_solution_l824_824956

theorem cryptarithm_no_solution :
  ∀ (I Y M P D : ℕ), 
  I ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  Y ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  M ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  P ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  D ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
  (I ≠ Y) ∧ (I ≠ M) ∧ (I ≠ P) ∧ (I ≠ D) ∧ 
  (Y ≠ M) ∧ (Y ≠ P) ∧ (Y ≠ D) ∧
  (M ≠ P) ∧ (M ≠ D) ∧
  (P ≠ D) →
  I * 3 * Y * M + P * Y * D ≠ 2022 :=
by
  intros _ _ _ _ _
  linarith
  sorry

end cryptarithm_no_solution_l824_824956


namespace smallest_integer_to_make_square_l824_824170

noncomputable def y : ℕ := 2^37 * 3^18 * 5^6 * 7^8

theorem smallest_integer_to_make_square : ∃ z : ℕ, z = 10 ∧ ∃ k : ℕ, (y * z) = k^2 :=
by
  sorry

end smallest_integer_to_make_square_l824_824170


namespace least_number_to_divisible_l824_824493

theorem least_number_to_divisible (x : ℕ) : 
  (∃ x, (1049 + x) % 25 = 0) ∧ (∀ y, y < x → (1049 + y) % 25 ≠ 0) ↔ x = 1 :=
by
  sorry

end least_number_to_divisible_l824_824493


namespace least_integer_greater_than_sqrt_500_l824_824038

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824038


namespace hypotenuse_of_right_triangle_l824_824589

variables {m n : ℝ} (h : m < n)

theorem hypotenuse_of_right_triangle
  (h : m < n)
  (tangent_segments : ∃ a b : ℝ, a = m ∧ b = n):
  ∃ x : ℝ, x = (m^2 + n^2) / (n - m)  :=
begin
  sorry
end

end hypotenuse_of_right_triangle_l824_824589


namespace blue_shirt_corduroy_glasses_count_l824_824334

-- Given conditions
def total_students := 1500
def percentage_blue_shirts := 0.35
def percentage_corduroy_pants_among_blue := 0.20
def percentage_glasses_among_blue_and_corduroy := 0.15

-- Statement to be proved
theorem blue_shirt_corduroy_glasses_count :
  let blue_shirt_wearers := (percentage_blue_shirts * total_students).toInt in
  let blue_and_corduroy_wearers := (percentage_corduroy_pants_among_blue * blue_shirt_wearers).toInt in
  let blue_corduroy_and_glasses_wearers := (percentage_glasses_among_blue_and_corduroy * blue_and_corduroy_wearers).toInt in
  blue_corduroy_and_glasses_wearers = 15 :=
by
  sorry

end blue_shirt_corduroy_glasses_count_l824_824334


namespace minneapolis_st_louis_temperature_l824_824547

theorem minneapolis_st_louis_temperature (N M L : ℝ) (h1 : M = L + N)
                                         (h2 : M - 7 = L + N - 7)
                                         (h3 : L + 5 = L + 5)
                                         (h4 : (M - 7) - (L + 5) = |(L + N - 7) - (L + 5)|) :
  ∃ (N1 N2 : ℝ), (|N - 12| = 4) ∧ N1 = 16 ∧ N2 = 8 ∧ N1 * N2 = 128 :=
by {
  sorry
}

end minneapolis_st_louis_temperature_l824_824547


namespace diagonals_in_polygon_with_150_sides_l824_824209

-- (a) Definitions for conditions
def sides : ℕ := 150

def diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- (c) Statement of the problem in Lean 4
theorem diagonals_in_polygon_with_150_sides :
  diagonals sides = 11025 :=
by
  sorry

end diagonals_in_polygon_with_150_sides_l824_824209


namespace necessary_but_not_sufficient_condition_l824_824560

-- Definitions of conditions
def condition_p (x : ℝ) := (x - 1) * (x + 2) ≤ 0
def condition_q (x : ℝ) := abs (x + 1) ≤ 1

-- The theorem statement
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (∀ x, condition_q x → condition_p x) ∧ ¬(∀ x, condition_p x → condition_q x) := 
by
  sorry

end necessary_but_not_sufficient_condition_l824_824560


namespace remaining_employees_earn_rate_l824_824726

theorem remaining_employees_earn_rate
  (total_employees : ℕ)
  (employees_12_per_hour : ℕ)
  (employees_14_per_hour : ℕ)
  (total_cost : ℝ)
  (hourly_rate_12 : ℝ)
  (hourly_rate_14 : ℝ)
  (shift_hours : ℝ)
  (remaining_employees : ℕ)
  (remaining_hourly_rate : ℝ) :
  total_employees = 300 →
  employees_12_per_hour = 200 →
  employees_14_per_hour = 40 →
  total_cost = 31840 →
  hourly_rate_12 = 12 →
  hourly_rate_14 = 14 →
  shift_hours = 8 →
  remaining_employees = 60 →
  remaining_hourly_rate = 
    (total_cost - (employees_12_per_hour * hourly_rate_12 * shift_hours) - 
    (employees_14_per_hour * hourly_rate_14 * shift_hours)) / 
    (remaining_employees * shift_hours) →
  remaining_hourly_rate = 17 :=
by
  sorry

end remaining_employees_earn_rate_l824_824726


namespace average_rainfall_l824_824720

theorem average_rainfall {R D H: ℕ} (hR : R = 320) (hD : D = 30) (hH: H = 24) :
  (R / (D * H) : ℚ) = 4 / 9 :=
by {
  -- start of the proof
  sorry
}

end average_rainfall_l824_824720


namespace largest_divisor_if_n_sq_div_72_l824_824701

theorem largest_divisor_if_n_sq_div_72 (n : ℕ) (h : n > 0) (h72 : 72 ∣ n^2) : ∃ m, m = 12 ∧ m ∣ n :=
by { sorry }

end largest_divisor_if_n_sq_div_72_l824_824701


namespace least_integer_greater_than_sqrt_500_l824_824033

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824033


namespace min_value_f_l824_824685

noncomputable def f (x : ℝ) : ℝ :=
  max (|3 * x|) (-x^2 + 4)

theorem min_value_f : ∃ x₀ ∈ (set.univ : set ℝ), f x₀ = 3 :=
by
  use 1
  sorry

end min_value_f_l824_824685


namespace count_special_integers_l824_824308

theorem count_special_integers : 
  let N : ℕ := {n | n < 500 ∧ 
                 (∃ (p q : ℕ), (n = p^7 ∨ n = p^3 * q ∨ n = p * q^3) ∧ 
                 nat.prime p ∧ nat.prime q ∧ 
                 (p = 1 ∨ q = 1 ∨ p ≠ q)) ∧ 
                 n.factors.length = 8}
  in  N.count == 5 :=
sorry

end count_special_integers_l824_824308


namespace smallest_three_digit_multiple_of_3_5_and_7_l824_824123

open Nat

theorem smallest_three_digit_multiple_of_3_5_and_7 :
  ∃ n : ℕ, n >= 100 ∧ n <= 999 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ n = 105 :=
by {
  sorry
}

end smallest_three_digit_multiple_of_3_5_and_7_l824_824123


namespace election_winner_percentage_l824_824459

theorem election_winner_percentage 
  (v1 v2 v3 : ℕ)
  (total_votes : ℕ)
  (winning_votes : ℕ)
  (h1: v1 = 1136)
  (h2: v2 = 8236)
  (h3: v3 = 11628)
  (h_total: total_votes = v1 + v2 + v3)
  (h_winning: winning_votes = v3):
  (winning_votes : ℝ) / (total_votes : ℝ) * 100 ≈ 55.37 := 
sorry

end election_winner_percentage_l824_824459


namespace true_proposition_B_true_proposition_C_l824_824996

-- Define the conditions for the angles α, β, and θ
variable (α β θ : ℝ)

-- Define the proposition B
def proposition_B (α β : ℝ) : Prop :=
  (sin α ≠ sin β) → (α ≠ β)

-- Proof that Proposition B is true
theorem true_proposition_B (α β : ℝ) : proposition_B α β :=
by sorry

-- Define the conditions for the angles θ
-- First quadrant: 0 < θ < π/2
def first_quadrant (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π / 2

-- Second quadrant: π/2 < θ < π
def second_quadrant (θ : ℝ) : Prop :=
  π / 2 < θ ∧ θ < π

-- Define the proposition C
def proposition_C (θ : ℝ) : Prop :=
  (first_quadrant θ ∨ second_quadrant θ) → (sin θ > 0)

-- Proof that Proposition C is true
theorem true_proposition_C (θ : ℝ) : proposition_C θ :=
by sorry

end true_proposition_B_true_proposition_C_l824_824996


namespace limit_pi_over_4_l824_824207

open Real

theorem limit_pi_over_4 : 
  (tendsto (λ x, (1 - sin (2 * x)) / (π - 4 * x)^2) 
    (𝓝 (π / 4)) 
    (𝓝 (1 / 8))) := 
sorry

end limit_pi_over_4_l824_824207


namespace animals_in_carlton_zoo_l824_824814

noncomputable def total_animals_carlton_zoo : ℕ :=
  let L_Carlton := 1 in
  let R_Bell := L_Carlton in
  let E_Bell := L_Carlton + 3 in
  let R_Carlton := L_Carlton + 3 in
  let E_Carlton := R_Carlton + 2 in
  let M_Carlton := 2 * (R_Carlton + E_Carlton + L_Carlton) in
  let P_Carlton := M_Carlton + 2 in
  let M_Bell := (2 * P_Carlton) / 3 in
  let P_Bell := M_Bell + 2 in
  let L_Bell := P_Bell / 2 in
  if R_Bell + E_Bell + L_Bell + M_Bell + P_Bell = 48 then
    R_Carlton + E_Carlton + L_Carlton + M_Carlton + P_Carlton
  else 
    sorry

theorem animals_in_carlton_zoo :
  total_animals_carlton_zoo = 57 :=
sorry

end animals_in_carlton_zoo_l824_824814


namespace number_of_tangent_lines_through_point_l824_824698

theorem number_of_tangent_lines_through_point :
  let point := (0, 6)
  let parabola := λ y, y^2 = -12 * (y / sqrt(12))^2
  ∃! (l : Line), ∃ p ∈ l, point ∈ l ∧ (∀ q : Point, q ∈ l → q ∈ parabola → q = p) → 3 :=
by
  sorry

end number_of_tangent_lines_through_point_l824_824698


namespace problem_l824_824479

theorem problem (a b : ℤ) (ha : a = 4) (hb : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := 
by
  -- Provide proof here
  sorry

end problem_l824_824479


namespace boat_speed_relative_to_river_bank_angle_minimizes_crossing_time_angle_gets_boat_directly_across_l824_824517

variables (u v : ℝ) (α : ℝ)

-- Proof problem 1: The speed of the boat relative to the river bank
theorem boat_speed_relative_to_river_bank :
  (sqrt (u^2 + v^2 + 2 * u * v * cos α) = sqrt ((u + v * cos α)^2 + (v * sin α)^2)) :=
sorry

-- Proof problem 2: The angle that minimizes the time taken to cross the river
theorem angle_minimizes_crossing_time : (α = 90) :=
sorry

-- Proof problem 3: The angle that gets the boat directly across the river
theorem angle_gets_boat_directly_across :
  (α = real.arccos (-u / v)) :=
sorry

end boat_speed_relative_to_river_bank_angle_minimizes_crossing_time_angle_gets_boat_directly_across_l824_824517


namespace hyperbola_foci_coordinates_l824_824818

-- Definition of the hyperbola equation.
def hyperbola_eq (x y : ℝ) : Prop := (x^2)/3 - (y^2)/2 = 1

-- Proves that the coordinates of the foci of the hyperbola are (±√5, 0)
theorem hyperbola_foci_coordinates : ∀ x y : ℝ, hyperbola_eq x y → (x = √5 ∧ y = 0) ∨ (x = -√5 ∧ y = 0) :=
by
  sorry

end hyperbola_foci_coordinates_l824_824818


namespace evaluate_expression_l824_824577

theorem evaluate_expression (b : ℝ) (hb : b = 3) : 
  (3 * b^(-2) + b^(-2) / 3) / b^2 = 10 / 243 :=
by
  rw hb
  sorry

end evaluate_expression_l824_824577


namespace limit_pi_over_4_l824_824208

open Real

theorem limit_pi_over_4 : 
  (tendsto (λ x, (1 - sin (2 * x)) / (π - 4 * x)^2) 
    (𝓝 (π / 4)) 
    (𝓝 (1 / 8))) := 
sorry

end limit_pi_over_4_l824_824208


namespace total_trash_picked_l824_824465

theorem total_trash_picked (trash_classrooms trash_outside : ℕ) (h_classrooms : trash_classrooms = 344) (h_outside : trash_outside = 1232) : trash_classrooms + trash_outside = 1576 :=
by
  rw [h_classrooms, h_outside]
  exact Nat.add_comm 344 1232

end total_trash_picked_l824_824465


namespace least_integer_greater_than_sqrt_500_l824_824026

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824026


namespace ratio_of_ages_l824_824355

theorem ratio_of_ages (joe_age_now james_age_now : ℕ) (h1 : joe_age_now = james_age_now + 10)
  (h2 : 2 * (joe_age_now + 8) = 3 * (james_age_now + 8)) : 
  (james_age_now + 8) / (joe_age_now + 8) = 2 / 3 := 
by
  sorry

end ratio_of_ages_l824_824355


namespace least_integer_greater_than_sqrt_500_l824_824077

theorem least_integer_greater_than_sqrt_500 : 
  ∃ x : ℤ, (22 < real.sqrt 500 ∧ real.sqrt 500 < 23) ∧ x = 23 :=
begin
  use 23,
  split,
  { split,
    { linarith [real.sqrt_lt.2 (by norm_num : 484 < 500)], },
    { linarith [real.sqrt_lt.2 (by norm_num : 500 < 529)], }, },
  refl,
end

end least_integer_greater_than_sqrt_500_l824_824077


namespace least_int_gt_sqrt_500_l824_824051

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824051


namespace value_of_expression_l824_824973

variable (a b : ℝ)

def system_of_equations : Prop :=
  (2 * a - b = 12) ∧ (a + 2 * b = 8)

theorem value_of_expression (h : system_of_equations a b) : 3 * a + b = 20 :=
  sorry

end value_of_expression_l824_824973


namespace least_integer_greater_than_sqrt_500_l824_824090

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ (∀ m : ℤ, m > real.sqrt 500 → n ≤ m) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824090


namespace eval_expr_at_2_l824_824416

-- Define the expressions involved
def expr (a : ℝ) := (1 + 4 / (a - 1)) / ((a^2 + 6 * a + 9) / (a^2 - a))

-- Define the target result after evaluation at a = 2
def target := 2 / 5

theorem eval_expr_at_2 : expr 2 = target := by
  sorry

end eval_expr_at_2_l824_824416


namespace cost_for_four_dozen_l824_824543

-- Definitions based on conditions
def cost_of_two_dozen_apples : ℝ := 15.60
def cost_of_one_dozen_apples : ℝ := cost_of_two_dozen_apples / 2
def cost_of_four_dozen_apples : ℝ := 4 * cost_of_one_dozen_apples

-- Statement to prove
theorem cost_for_four_dozen :
  cost_of_four_dozen_apples = 31.20 :=
sorry

end cost_for_four_dozen_l824_824543


namespace find_abc_l824_824500

theorem find_abc (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : c < 4)
  (h4 : a + b + c = a * b * c) : (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
                                 (a = -3 ∧ b = -2 ∧ c = -1) ∨ 
                                 (a = -1 ∧ b = 0 ∧ c = 1) ∨ 
                                 (a = -2 ∧ b = 0 ∧ c = 2) ∨ 
                                 (a = -3 ∧ b = 0 ∧ c = 3) :=
sorry

end find_abc_l824_824500


namespace largest_coefficient_term_l824_824938

theorem largest_coefficient_term :
  ∃ r, r = 5 ∧ (∀ s, s ≠ 5 → binomial (15 : Nat) s * (1 / 2)^s < binomial (15 : Nat) (5 : Nat) * (1 / 2)^5) :=
sorry

end largest_coefficient_term_l824_824938


namespace wilfred_carrots_l824_824392

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem wilfred_carrots (x y z : ℕ) (h₁ : x + y + z = 15) (h₂ : is_odd z) : 
  (z ∈ {1, 3, 5, 7, 9, 11, 13, 15} ∧ ∃ x y : ℕ, x + y = 15 - z) := 
sorry

end wilfred_carrots_l824_824392


namespace inequality_l824_824363

variables {n : ℕ} (a : Fin n → ℝ)

noncomputable def geometric_mean (a : Fin n → ℝ) : ℝ :=
  Real.root n (∏ i, a i)

noncomputable def arithmetic_mean (a : Fin n → ℝ) : Fin (n + 1) → ℝ
  | ⟨k, hk⟩ => (∑ i in Finset.range k.succ, a i) / k.succ

noncomputable def G_n (a : Fin n → ℝ) :=
  geometric_mean (λ i, arithmetic_mean a ⟨i + 1, Nat.succ_lt_succ_iff.2 i.is_lt⟩)

theorem inequality (hn : n > 1) (pos : ∀ i, 0 < a i) :
  let g_n := geometric_mean a
      A_n := arithmetic_mean a ⟨n, Nat.lt_succ_self n⟩
      G_n := G_n a in
    n * Real.root n (G_n / A_n) + (g_n / G_n) ≤ n + 1 :=
sorry

end inequality_l824_824363


namespace bjorn_cannot_prevent_vakha_l824_824401

-- Define the primary settings and objects involved
def n_points : ℕ := 99
inductive Color
| red 
| blue 

structure GameState :=
  (turn : ℕ)
  (points : Fin n_points → Option Color)

-- Define the valid states of the game where turn must be within the range of points
def valid_state (s : GameState) : Prop :=
  s.turn ≤ n_points ∧ ∀ p, s.points p ≠ none

-- Define what it means for an equilateral triangle to be monochromatically colored
def monochromatic_equilateral_triangle (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Fin n_points), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    (p1.val + (n_points/3) % n_points) = p2.val ∧
    (p2.val + (n_points/3) % n_points) = p3.val ∧
    (p3.val + (n_points/3) % n_points) = p1.val ∧
    (state.points p1 = state.points p2) ∧ 
    (state.points p2 = state.points p3)

-- Vakha's winning condition
def vakha_wins (state : GameState) : Prop := 
  monochromatic_equilateral_triangle state

-- Bjorn's winning condition prevents Vakha from winning
def bjorn_can_prevent_vakha (initial_state : GameState) : Prop :=
  ¬ vakha_wins initial_state

-- Main theorem stating Bjorn cannot prevent Vakha from winning
theorem bjorn_cannot_prevent_vakha : ∀ (initial_state : GameState),
  valid_state initial_state → ¬ bjorn_can_prevent_vakha initial_state :=
sorry

end bjorn_cannot_prevent_vakha_l824_824401


namespace margaret_time_correct_l824_824394

def margaret_time : ℕ :=
  let n := 7
  let r := 15
  (Nat.factorial n) / r

theorem margaret_time_correct : margaret_time = 336 := by
  sorry

end margaret_time_correct_l824_824394


namespace polygon_is_hexagon_l824_824712

theorem polygon_is_hexagon (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  have hd : (n - 2) * 180 = 720 := by rw [h]
  have hn : n - 2 = 4 := by linarith
  rw [← hd, ← hn]
  linarith

end polygon_is_hexagon_l824_824712


namespace parabola_focus_l824_824240

theorem parabola_focus (f : ℝ) :
  (∀ x : ℝ, 2*x^2 = x^2 + (2*x^2 - f)^2 - (2*x^2 - -f)^2) →
  f = -1/8 :=
by sorry

end parabola_focus_l824_824240


namespace probability_interval_contains_p_l824_824991

theorem probability_interval_contains_p (P_A P_B p : ℝ) 
  (hA : P_A = 5 / 6) 
  (hB : P_B = 3 / 4) 
  (hp : p = P_A + P_B - 1) : 
  (5 / 12 ≤ p ∧ p ≤ 3 / 4) :=
by
  -- The proof is skipped by sorry as per the instructions.
  sorry

end probability_interval_contains_p_l824_824991


namespace reducible_fraction_implies_divisibility_l824_824772

theorem reducible_fraction_implies_divisibility
  (a b c d l k : ℤ)
  (m n : ℤ)
  (h1 : a * l + b = k * m)
  (h2 : c * l + d = k * n)
  : k ∣ (a * d - b * c) :=
by
  sorry

end reducible_fraction_implies_divisibility_l824_824772


namespace problem_statement_l824_824752

-- Define the set A
def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the function property
def satisfies_property (f : ℕ → ℕ) : Prop :=
  ∀ x ∈ A, ∃ c ∈ A, ∀ y ∈ A, f(f(y)) = c

-- Define the number of such functions
noncomputable def N : ℕ :=
  7 * (Finset.sum (Finset.range 7) (λ k, Nat.choose 6 k * k^(6 - k)))

-- Define the remainder
def remainder : ℕ := N % 1000

-- The theorem we need to prove
theorem problem_statement : remainder = 448 := by
  sorry

end problem_statement_l824_824752


namespace largest_n_for_15_divisor_of_30_l824_824598

theorem largest_n_for_15_divisor_of_30! : 
  ∃ n : ℕ, (15 ^ n ∣ nat.factorial 30) ∧ ∀ m : ℕ, (15 ^ m ∣ nat.factorial 30) → m ≤ 7 :=
begin
  sorry
end

end largest_n_for_15_divisor_of_30_l824_824598


namespace probability_one_girl_l824_824977

theorem probability_one_girl (boys girls : ℕ) (total_pairs total_pairs_with_one_girl : ℕ) :
  boys = 4 → girls = 2 →
  total_pairs = Nat.choose 6 2 →
  total_pairs_with_one_girl = (Nat.choose 4 1) * (Nat.choose 2 1) →
  (total_pairs_with_one_girl : ℚ) / total_pairs = 8 / 15 :=
by
  intros h_boys h_girls h_total_pairs h_total_pairs_with_one_girl
  rw [h_boys, h_girls, h_total_pairs, h_total_pairs_with_one_girl]
  norm_num
  sorry

end probability_one_girl_l824_824977


namespace sequence_term_five_l824_824655

theorem sequence_term_five :
  (∀ n : ℕ, (a n = 4 * n - 3)) → a 5 = 17 :=
by
  assume h : ∀ n : ℕ, (a n = 4 * n - 3)
  have h_a5 : a 5 = 4 * 5 - 3 := by sorry
  exact h_a5

end sequence_term_five_l824_824655


namespace largest_n_for_15_divisor_of_30_l824_824596

theorem largest_n_for_15_divisor_of_30! : 
  ∃ n : ℕ, (15 ^ n ∣ nat.factorial 30) ∧ ∀ m : ℕ, (15 ^ m ∣ nat.factorial 30) → m ≤ 7 :=
begin
  sorry
end

end largest_n_for_15_divisor_of_30_l824_824596


namespace least_integer_greater_than_sqrt_500_l824_824119

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824119


namespace one_third_sugar_l824_824886

theorem one_third_sugar (sugar : ℚ) (h : sugar = 3 + 3 / 4) : sugar / 3 = 1 + 1 / 4 :=
by sorry

end one_third_sugar_l824_824886


namespace circle_alignment_l824_824672

theorem circle_alignment :
  ∀ (circleA circleB : Set ℝ) 
  (circumferenceA circumferenceB : ℝ)
  (markedPointsA : Finset ℝ) 
  (markedArcsB : Set (Set ℝ)),
  circumferenceA = 100 →
  circumferenceB = 100 →
  markedPointsA.card = 100 →
  (∑ arc in markedArcsB, arc.card) < 1 →
  ∃ θ : ℝ, ∀ p ∈ markedPointsA, ∀ arc ∈ markedArcsB, (rotate arc θ) ∉ markedArcB :=
by
  sorry

end circle_alignment_l824_824672


namespace return_trip_time_l824_824948

-- Define the given conditions
def run_time : ℕ := 20
def jog_time : ℕ := 10
def trip_time := run_time + jog_time
def multiplier: ℕ := 3

-- State the theorem
theorem return_trip_time : trip_time * multiplier = 90 := by
  sorry

end return_trip_time_l824_824948


namespace find_c_interval_l824_824237

theorem find_c_interval (c : ℚ) : 
  (c / 4 ≤ 3 + c ∧ 3 + c < -3 * (1 + c)) ↔ (-4 ≤ c ∧ c < -3 / 2) := 
by 
  sorry

end find_c_interval_l824_824237


namespace evaluate_expression_l824_824223

-- Define the operation * given by the table
def op (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1,1) => 1 | (1,2) => 2 | (1,3) => 3 | (1,4) => 4
  | (2,1) => 2 | (2,2) => 4 | (2,3) => 1 | (2,4) => 3
  | (3,1) => 3 | (3,2) => 1 | (3,3) => 4 | (3,4) => 2
  | (4,1) => 4 | (4,2) => 3 | (4,3) => 2 | (4,4) => 1
  | _ => 0  -- default to handle cases outside the defined table

-- Define the theorem to prove $(2*4)*(1*3) = 4$
theorem evaluate_expression : op (op 2 4) (op 1 3) = 4 := by
  sorry

end evaluate_expression_l824_824223


namespace sum_of_squares_l824_824689

open Int

theorem sum_of_squares (p q r s t u : ℤ) (h : ∀ x : ℤ, 343 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3506 :=
sorry

end sum_of_squares_l824_824689


namespace number_of_testing_methods_l824_824968

-- Definitions based on conditions
def num_genuine_items : ℕ := 6
def num_defective_items : ℕ := 4
def total_tests : ℕ := 5

-- Theorem stating the number of testing methods
theorem number_of_testing_methods 
    (h1 : total_tests = 5) 
    (h2 : num_genuine_items = 6) 
    (h3 : num_defective_items = 4) :
    ∃ n : ℕ, n = 576 := 
sorry

end number_of_testing_methods_l824_824968


namespace bill_and_alice_total_objects_l824_824548

variable (Ted_sticks Bill_sticks Alice_sticks Ted_rocks Bill_rocks Alice_rocks : ℕ)

def conditions (Ted_sticks = 12) (Ted_rocks = 18) : Prop :=
  Bill_sticks = Ted_sticks + 6 ∧
  Alice_sticks = Ted_sticks / 2 ∧
  Bill_rocks = Ted_rocks / 2 ∧
  Alice_rocks = 3 * Bill_rocks

theorem bill_and_alice_total_objects :
  conditions 12 18 →
  Bill_sticks + Bill_rocks + Alice_sticks + Alice_rocks = 60 := by
  sorry

end bill_and_alice_total_objects_l824_824548


namespace question_l824_824618

open Complex

-- Definitions of conditions
def condition1 (x y : ℝ) : Prop := (1 - I) * (x + y * I) = 2

-- Statement to prove the quadrants
theorem question (x y : ℝ) (h : condition1 x y) : (conj (x + y * I)).im < 0 ∧ (conj (x + y * I)).re > 0 := 
sorry

end question_l824_824618


namespace divisible_by_11_l824_824251

theorem divisible_by_11 (n : ℤ) : (11 ∣ (n^2001 - n^4)) ↔ (n % 11 = 0 ∨ n % 11 = 1) :=
by
  sorry

end divisible_by_11_l824_824251


namespace cats_not_liking_catnip_or_tuna_l824_824330

theorem cats_not_liking_catnip_or_tuna :
  ∀ (total_cats catnip_lovers tuna_lovers both_lovers : ℕ),
  total_cats = 80 →
  catnip_lovers = 15 →
  tuna_lovers = 60 →
  both_lovers = 10 →
  (total_cats - (catnip_lovers - both_lovers + both_lovers + tuna_lovers - both_lovers)) = 15 :=
by
  intros total_cats catnip_lovers tuna_lovers both_lovers ht hc ht hboth
  sorry

end cats_not_liking_catnip_or_tuna_l824_824330


namespace return_trip_time_l824_824946

-- Define the given conditions
def run_time : ℕ := 20
def jog_time : ℕ := 10
def trip_time := run_time + jog_time
def multiplier: ℕ := 3

-- State the theorem
theorem return_trip_time : trip_time * multiplier = 90 := by
  sorry

end return_trip_time_l824_824946


namespace general_term_less_than_zero_from_13_l824_824992

-- Define the arithmetic sequence and conditions
def an (n : ℕ) : ℝ := 12 - n

-- Condition: a_3 = 9
def a3_condition : Prop := an 3 = 9

-- Condition: a_9 = 3
def a9_condition : Prop := an 9 = 3

-- Prove the general term of the sequence is 12 - n
theorem general_term (n : ℕ) (h3 : a3_condition) (h9 : a9_condition) :
  an n = 12 - n := 
sorry

-- Prove that the sequence becomes less than 0 starting from the 13th term
theorem less_than_zero_from_13 (h3 : a3_condition) (h9 : a9_condition) :
  ∀ n, n ≥ 13 → an n < 0 :=
sorry

end general_term_less_than_zero_from_13_l824_824992


namespace inverse_value_l824_824316

-- Define the function g
def g(x : ℝ) : ℝ := 20 / (4 + 2 * x)

-- Define the inverse of g, g⁻¹ such that g(g⁻¹(y)) = y
noncomputable def g_inv(y : ℝ) : ℝ := (20 / y - 4) / 2

-- Define the theorem to prove [g⁻¹(4)]⁻¹ = 2
theorem inverse_value : (g_inv 4)⁻¹ = 2 :=
by sorry

end inverse_value_l824_824316


namespace sqrt_mul_eq_sqrt_correct_calculation_l824_824865

theorem sqrt_mul_eq_sqrt {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b :=
by sorry

theorem correct_calculation : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 :=
by apply sqrt_mul_eq_sqrt; exacts [le_of_lt (Real.sqrt_pos.mpr (by linarith)), le_of_lt (Real.sqrt_pos.mpr (by linarith))]

end sqrt_mul_eq_sqrt_correct_calculation_l824_824865


namespace proof_part1_proof_part2_proof_part3_l824_824302

noncomputable def point1 : ℝ × ℝ := (1, 0)
noncomputable def point2 : ℝ × ℝ := (0, 1)
noncomputable def point3 : ℝ × ℝ := (2, 5)

noncomputable def vec_AB : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)
noncomputable def vec_AC : ℝ × ℝ := (point3.1 - point1.1, point3.2 - point1.2)
noncomputable def vec_BC : ℝ × ℝ := (point3.1 - point2.1, point3.2 - point2.2)

noncomputable def vec_result1 : ℝ × ℝ := (2 * vec_AB.1 + vec_AC.1, 2 * vec_AB.2 + vec_AC.2)
noncomputable def magnitude_vec_result1 : ℝ := real.sqrt (vec_result1.1^2 + vec_result1.2^2)

noncomputable def magnitude_AB : ℝ := real.sqrt (vec_AB.1^2 + vec_AB.2^2)
noncomputable def magnitude_AC : ℝ := real.sqrt (vec_AC.1^2 + vec_AC.2^2)
noncomputable def dot_AB_AC : ℝ := vec_AB.1 * vec_AC.1 + vec_AB.2 * vec_AC.2
noncomputable def cosine_theta : ℝ := dot_AB_AC / (magnitude_AB * magnitude_AC)

theorem proof_part1 : magnitude_vec_result1 = 5 * real.sqrt 2 := 
by
  sorry

theorem proof_part2 : cosine_theta = 2 * real.sqrt 13 / 13 := 
by 
  sorry

theorem proof_part3 : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧  (vec_BC.1 * x + vec_BC.2 * y = 0) ∧ 
    ((x = 2 * real.sqrt 5 / 5 ∧ y = - real.sqrt 5 / 5) 
  ∨ (x = - 2 * real.sqrt 5 / 5 ∧ y = real.sqrt 5 / 5)) := 
by 
  sorry

end proof_part1_proof_part2_proof_part3_l824_824302


namespace total_paths_from_X_to_Z_l824_824224

variable (X Y Z : Type)
variables (f : X → Y → Z)
variables (g : X → Z)

-- Conditions
def paths_X_to_Y : ℕ := 3
def paths_Y_to_Z : ℕ := 4
def direct_paths_X_to_Z : ℕ := 1

-- Proof problem statement
theorem total_paths_from_X_to_Z : paths_X_to_Y * paths_Y_to_Z + direct_paths_X_to_Z = 13 := sorry

end total_paths_from_X_to_Z_l824_824224


namespace probability_event_l824_824662

def f (x : ℝ) : ℝ :=
  if x < π then 1 - Real.sin x else Real.log (x / π) / Real.log 2016

theorem probability_event (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3)
  (h3 : f x1 = f x2) (h4 : f x2 = f x3) :
  (Real.to_nnreal (x1 + x2) > Real.to_nnreal (4 * π - x3)) ->
  ℝ :=
sorry

end probability_event_l824_824662


namespace higher_profit_percentage_l824_824908

theorem higher_profit_percentage (P : ℝ) :
  (P / 100 * 800 = 144) ↔ (P = 18) :=
by
  sorry

end higher_profit_percentage_l824_824908


namespace probability_no_shaded_square_l824_824153

theorem probability_no_shaded_square :
  let num_rects := 1003 * 2005
  let num_rects_with_shaded := 1002^2
  let probability_no_shaded := 1 - (num_rects_with_shaded / num_rects)
  probability_no_shaded = 1 / 1003 := by
  -- The proof steps go here
  sorry

end probability_no_shaded_square_l824_824153


namespace radius_of_sphere_l824_824975

-- Define the edge length of the cube
def edge_length : ℝ := Real.sqrt 41

-- Define the midpoint function
def midpoint (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2, (a.3 + b.3) / 2)

-- Define points A, B, C, D, A1, B1, C1, D1
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (edge_length, 0, 0)
def C : ℝ × ℝ × ℝ := (edge_length, edge_length, 0)
def D : ℝ × ℝ × ℝ := (0, edge_length, 0)
def A1 : ℝ × ℝ × ℝ := (0, 0, edge_length)
def B1 : ℝ × ℝ × ℝ := (edge_length, 0, edge_length)
def C1 : ℝ × ℝ × ℝ := (edge_length, edge_length, edge_length)
def D1 : ℝ × ℝ × ℝ := (0, edge_length, edge_length)

-- Define points M and K
def M : ℝ × ℝ × ℝ := midpoint A B
def K : ℝ × ℝ × ℝ := midpoint C D

-- Prove that the radius of the sphere passing through points M, K, A1, and C1 is 11/8
theorem radius_of_sphere : 
  ∃ R : ℝ, 
    (sphere.passes_through_points R M K A1 C1) ∧
    R = 11 / 8 :=
sorry

end radius_of_sphere_l824_824975


namespace solve_system_l824_824804

theorem solve_system : ∃ s t : ℝ, (11 * s + 7 * t = 240) ∧ (s = 1 / 2 * t + 3) ∧ (t = 414 / 25) :=
by
  sorry

end solve_system_l824_824804


namespace peanuts_remaining_l824_824846

theorem peanuts_remaining (initial : ℕ) (brock_fraction : ℚ) (bonita_count : ℕ) (h_initial : initial = 148) (h_brock_fraction : brock_fraction = 1/4) (h_bonita_count : bonita_count = 29) : initial - (initial * brock_fraction).natValue - bonita_count = 82 := 
by 
  sorry

end peanuts_remaining_l824_824846


namespace least_integer_greater_than_sqrt_500_l824_824027

theorem least_integer_greater_than_sqrt_500 : 
  let sqrt_500 := Real.sqrt 500
  ∃ n : ℕ, (n > sqrt_500) ∧ (n = 23) :=
by 
  have h1: 22^2 = 484 := rfl
  have h2: 23^2 = 529 := rfl
  have h3: 484 < 500 := by norm_num
  have h4: 500 < 529 := by norm_num
  have h5: 484 < 500 < 529 := by exact ⟨h3, h4⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824027


namespace triangle_area_l824_824979

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem triangle_area :
  area_of_triangle 0 0 0 6 8 0 = 24 :=
by
  sorry

end triangle_area_l824_824979


namespace part_I_part_II_part_III_l824_824639

-- Definitions and conditions from the problem
variables {a b : ℕ} (a_pos : 0 < a) (b_pos : 0 < b)
def a_n (n : ℕ) : ℕ := a + (n - 1) * b
def b_n (n : ℕ) : ℕ := b * a^(n - 1)

theorem part_I (h : a + (n - 1) * b < b * a^(n - 1)) (h1 : a = 2) : a = 2 :=
by
  sorry

theorem part_II {m n : ℕ} (h: a_m m + 1 = b_n n) (a_eq : a = 2) (b_pos : 2 < b) : b = 3 :=
by
  sorry

theorem part_III (a_eq : a = 2) (b_eq : b = 3) :
  ∀ (n : ℕ), c_n n = (n - 3) / (2^(n - 1)) → max (c_n n) = 1 / 8 :=
by
  sorry

end part_I_part_II_part_III_l824_824639


namespace max_tetrahedron_volume_on_sphere_l824_824984

theorem max_tetrahedron_volume_on_sphere 
    (A B C D : EuclideanSpace ℝ (Fin 3)) 
    (r : ℝ)
    (h_radius : r = 2)
    (h_on_sphere : dist A (0:ℝ^3) = r ∧ dist B (0:ℝ^3) = r ∧ dist C (0:ℝ^3) = r ∧ dist D (0:ℝ^3) = r)
    (h_AB : dist A B = 2)
    (h_CD : dist C D = 2) :
    ∃ V : ℝ, V = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end max_tetrahedron_volume_on_sphere_l824_824984


namespace inequality_of_powers_l824_824751

theorem inequality_of_powers (k n : ℕ) (x : Fin k → ℝ)
  (hk1 : 1 < k)
  (hk2 : k ≤ n)
  (hx_pos : ∀ i, 0 < x i)
  (hx_product_sum : ∏ i in Finset.range k, x i = ∑ i in Finset.range k, x i) :
  ∑ i in Finset.range k, x i ^ (n - 1) ≥ k * n :=
sorry

end inequality_of_powers_l824_824751


namespace jamies_class_girls_count_l824_824719

theorem jamies_class_girls_count 
  (g b : ℕ)
  (h_ratio : 4 * g = 3 * b)
  (h_total : g + b = 35) 
  : g = 15 := 
by 
  sorry 

end jamies_class_girls_count_l824_824719


namespace evaluate_expression_l824_824930

def g (x : ℝ) : ℝ := x^2 - 2 * Real.sqrt x

theorem evaluate_expression : 2 * g 3 + g 7 = 67 - 4 * Real.sqrt 3 - 2 * Real.sqrt 7 := by
  sorry

end evaluate_expression_l824_824930


namespace least_integer_greater_than_sqrt_500_l824_824037

/-- 
If \( n^2 < x < (n+1)^2 \), then the least integer greater than \(\sqrt{x}\) is \(n+1\). 
In this problem, we prove the least integer greater than \(\sqrt{500}\) is 23 given 
that \( 22^2 < 500 < 23^2 \).
-/
theorem least_integer_greater_than_sqrt_500 
    (h1 : 22^2 < 500) 
    (h2 : 500 < 23^2) : 
    (∃ k : ℤ, k > real.sqrt 500 ∧ k = 23) :=
sorry 

end least_integer_greater_than_sqrt_500_l824_824037


namespace area_of_two_congruent_squares_l824_824856

/-- Two congruent squares $ABCD$ and $XYZW$ have side length $12$.
    The center of square $XYZW$ coincides with vertex $A$ of square $ABCD$.
    The area of the region in the plane covered by these squares is $144$. -/
theorem area_of_two_congruent_squares (A B C D X Y Z W : ℝ) (side : ℝ) (h : side = 12) :
  let abcd := side^2,
      xyzw := side^2,
      overlap := abcd,
      total_area := abcd + xyzw - overlap in
  total_area = 144 :=
by
  sorry

end area_of_two_congruent_squares_l824_824856


namespace least_integer_greater_than_sqrt_500_l824_824007

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, n^2 > 500 ∧ ∀ m : ℕ, m < n → m^2 ≤ 500 := by
  let n := 23
  have h1 : n^2 > 500 := by norm_num
  have h2 : ∀ m : ℕ, m < n → m^2 ≤ 500 := by
    intros m h
    cases m
    . norm_num
    iterate 22
    · norm_num
  exact ⟨n, h1, h2⟩
  sorry

end least_integer_greater_than_sqrt_500_l824_824007


namespace ratio_BE_ED_in_quadrilateral_l824_824340

theorem ratio_BE_ED_in_quadrilateral
  (A B C D E : Type)
  [Quadrilateral A B C D] -- Assume a type class for quadrilateral
  (h1 : AB = 5)
  (h2 : BC = 6)
  (h3 : CD = 5)
  (h4 : DA = 4)
  (h5 : Angle ABC = 90)
  (h6 : LinesIntersect AC BD E)
  : BE / ED = sqrt 3 := 
sorry

end ratio_BE_ED_in_quadrilateral_l824_824340


namespace rational_coefficient_terms_count_l824_824564

theorem rational_coefficient_terms_count : (set_of (λ k : ℕ, k % 4 = 0 ∧ k ≤ 1024)).card = 257 := by
sorry

end rational_coefficient_terms_count_l824_824564


namespace least_integer_greater_than_sqrt_500_l824_824111

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l824_824111


namespace necessary_sufficient_condition_real_l824_824442

open Complex

theorem necessary_sufficient_condition_real (a b : ℝ) :
  (∃ z : ℂ, z = a^2 + b^2 + (a + |a|) * I ∧ Im z = 0) ↔ a ≤ 0 :=
by
  sorry

end necessary_sufficient_condition_real_l824_824442


namespace total_output_approx_six_point_six_l824_824445

theorem total_output_approx_six_point_six 
  (last_year_output : ℝ)
  (growth_rate : ℝ)
  (approx : ℝ) :
  last_year_output = 1 →
  growth_rate = 1.1 →
  approx = 1.6 →
  let first_year_output := last_year_output * growth_rate in
  let second_year_output := first_year_output * growth_rate in
  let third_year_output := second_year_output * growth_rate in
  let fourth_year_output := third_year_output * growth_rate in
  let fifth_year_output := fourth_year_output * growth_rate in
  (first_year_output + second_year_output + third_year_output + fourth_year_output + fifth_year_output) ≈ 6.6 :=
by
  intros h1 h2 h3
  let first_year_output := last_year_output * growth_rate
  let second_year_output := first_year_output * growth_rate
  let third_year_output := second_year_output * growth_rate
  let fourth_year_output := third_year_output * growth_rate
  let fifth_year_output := fourth_year_output * growth_rate
  sorry

end total_output_approx_six_point_six_l824_824445


namespace mike_max_marks_l824_824491

theorem mike_max_marks (m : ℕ) (h : 30 * m = 237 * 10) : m = 790 := by
  sorry

end mike_max_marks_l824_824491


namespace zyka_expense_increase_l824_824869

theorem zyka_expense_increase (C_k C_c : ℝ) (h1 : 0.5 * C_k = 0.2 * C_c) : 
  (((1.2 * C_c) - C_c) / C_c) * 100 = 20 := by
  sorry

end zyka_expense_increase_l824_824869


namespace ship_journey_distance_graph_l824_824519

theorem ship_journey_distance_graph (r : ℝ) :
  let A := (0, r), X := (0, 0)
  let B := (r, 0)
  let C := (r, r)
  let dist (p q : ℝ × ℝ) := (p.1 - q.1)^2 + (p.2 - q.2)^2
  graph_representation : string :=
    if dist A X = r^2 ∧ dist B X = r^2 ∧ dist C X = 2 * r^2
    then "Horizontal line, ascending line, descending line"
    else "Wrong Representation"
  in graph_representation = "Horizontal line, ascending line, descending line" :=
by 
  sorry

end ship_journey_distance_graph_l824_824519


namespace domain_f_l824_824435

open Real Set

noncomputable def f (x : ℝ) : ℝ := log (x + 1) + (x - 2) ^ 0

theorem domain_f :
  (∃ x : ℝ, f x = f x) ↔ (∀ x, (x > -1 ∧ x ≠ 2) ↔ (x ∈ Ioo (-1 : ℝ) 2 ∨ x ∈ Ioi 2)) :=
by
  sorry

end domain_f_l824_824435


namespace chocolate_cost_lollipops_equiv_l824_824201

theorem chocolate_cost_lollipops_equiv :
  ∃ (lollipop_cost : ℕ) (chocolate_pack_cost : ℕ),
    lollipop_cost = 2 ∧
    chocolate_pack_cost = 8 ∧
    (chocolate_pack_cost / lollipop_cost) = 4 :=
begin
  sorry
end

end chocolate_cost_lollipops_equiv_l824_824201


namespace least_int_gt_sqrt_500_l824_824056

theorem least_int_gt_sqrt_500 : ∃ n : ℤ, n > real.sqrt 500 ∧ ∀ m : ℤ, m > real.sqrt 500 → n ≤ m :=
begin
  use 23,
  split,
  {
    -- show 23 > sqrt 500
    sorry
  },
  {
    -- show that for all m > sqrt 500, 23 <= m
    intros m hm,
    sorry,
  }
end

end least_int_gt_sqrt_500_l824_824056


namespace sqrt_500_least_integer_l824_824019

theorem sqrt_500_least_integer : ∀ (n : ℕ), n > 0 ∧ n^2 > 500 ∧ (n - 1)^2 <= 500 → n = 23 :=
by
  intros n h,
  sorry

end sqrt_500_least_integer_l824_824019


namespace part1_part2_l824_824212

-- Part (1)
theorem part1 : -6 * -2 + -5 * 16 = -68 := by
  sorry

-- Part (2)
theorem part2 : -1^4 + (1 / 4) * (2 * -6 - (-4)^2) = -8 := by
  sorry

end part1_part2_l824_824212


namespace smallest_five_digit_neg_int_congruent_to_one_mod_17_l824_824474

theorem smallest_five_digit_neg_int_congruent_to_one_mod_17 :
  ∃ (x : ℤ), x < -9999 ∧ x % 17 = 1 ∧ x = -10011 := by
  -- The proof would go here
  sorry

end smallest_five_digit_neg_int_congruent_to_one_mod_17_l824_824474


namespace game_show_possible_guesses_l824_824891

theorem game_show_possible_guesses :
  let digits := [2, 2, 2, 4, 4, 4, 4]
  ∃ (D E F : ℕ → ℕ) (p : Finset (Finset ℕ)),
  (∀ x ∈ p, ∃ D E F, (D + E + F = 7) ∧ 
    (∀ n ∈ p, list.nodup ([D, E, F]) ∧ (digits = t))) →
  list.count digits 2 = 3 ∧ 
  list.count digits 4 = 4 →
  ∃ n, n = 420 :=
begin
  sorry
end

end game_show_possible_guesses_l824_824891


namespace object_speed_approx_13_64_mph_l824_824319

theorem object_speed_approx_13_64_mph
  (distance_feet : ℕ)
  (time_seconds : ℕ) 
  (feet_per_mile : ℕ)
  (seconds_per_hour : ℕ)
  (distance_miles : ℝ)
  (time_hours : ℝ)
  (speed_mph : ℝ)
  (h1 : distance_feet = 80)
  (h2 : time_seconds = 4)
  (h3 : feet_per_mile = 5280)
  (h4 : seconds_per_hour = 3600)
  (h5 : distance_miles = (distance_feet : ℝ) / (feet_per_mile : ℝ))
  (h6 : time_hours = (time_seconds : ℝ) / (seconds_per_hour : ℝ))
  (h7 : speed_mph = distance_miles / time_hours) :
  speed_mph ≈ 13.64 :=
  sorry

end object_speed_approx_13_64_mph_l824_824319


namespace horses_more_than_ponies_l824_824896

theorem horses_more_than_ponies {P H : ℕ} :
  (3 / 10 * P).denom = 1 →         -- 3/10 of the ponies have horseshoes, implies as a whole number
  (5 / 8 * (3 / 10 * P)).denom = 1 → -- 5/8 of these ponies are from Iceland, implies as a whole number
  P + H = 164 →                   -- Combined number of horses and ponies
  H > P →                         -- Horses more than ponies
  H - P = 4 :=
by
  sorry

end horses_more_than_ponies_l824_824896


namespace largest_power_of_15_dividing_factorial_30_l824_824591

theorem largest_power_of_15_dividing_factorial_30 : ∃ (n : ℕ), 
  (∀ m : ℕ, 15 ^ m ∣ Nat.factorial 30 ↔ m ≤ n) ∧ n = 7 :=
by
  have h3 : ∀ m : ℕ, 3 ^ m ∣ Nat.factorial 30 ↔ m ≤ 14 := sorry
  have h5 : ∀ m : ℕ, 5 ^ m ∣ Nat.factorial 30 ↔ m ≤ 7 := sorry
  use 7
  split
  · intro m
    split
    · intro h
      obtain ⟨k, rfl⟩ : ∃ k, m = k := Nat.exists_eq m
      have : 3 ^ k ∣ Nat.factorial 30 := by exact (15 ^ k).dvd_of_dvd_mul_left (by convert h)
      rw [h3, h5] at this
      exact Nat.le_min this.left this.right
    · intro h
      exact (Nat.min_le_iff.mpr ⟨(h3.mpr (Nat.le_of_lt_succ (Nat.sub_lt_succ (14 - 7))), h5.mpr h⟩, sorry
  exact rfl
  sorry

end largest_power_of_15_dividing_factorial_30_l824_824591


namespace sector_central_angle_l824_824279

theorem sector_central_angle (r : ℝ) (h : r > 0) (perimeter_eq : 2 * r + (2 * r + 2 * r / π) = 3 * r) : 
  let l := r in
  let α := l / r in
  α = 1 :=
by
  sorry

end sector_central_angle_l824_824279


namespace part1_part2_l824_824668

noncomputable def g (x : ℝ) : ℝ := x / Real.log x
noncomputable def f (x a : ℝ) : ℝ := g x - a * x

theorem part1 {x : ℝ} (hx1 : x > 0 ∧ x ≠ 1) :
  (g' x > 0 ↔ x > Real.exp 1) ∧ 
  (g' x < 0 ↔ 0 < x ∧ x < 1) ∧ 
  (g' x < 0 ↔ 1 < x ∧ x < Real.exp 1) := 
sorry

theorem part2 (a : ℝ) (h : ∀ x > 1, f' x a ≤ 0) : 
  a ≥ 1 / 4 :=
sorry

end part1_part2_l824_824668


namespace number_of_valid_sequences_l824_824385

structure Point := (x : ℝ) (y : ℝ)

def T : list Point := [⟨0, 0⟩, ⟨6, 0⟩, ⟨0, 4⟩]

def rotate_120 (p : Point) : Point := ⟨-p.y, p.x + p.y⟩
def rotate_180 (p : Point) : Point := ⟨-p.x, -p.y⟩
def rotate_240 (p : Point) : Point := ⟨p.y - p.x, -p.x⟩
def reflect_y_eq_x (p : Point) : Point := ⟨p.y, p.x⟩
def reflect_y_eq_neg_x (p : Point) : Point := ⟨-p.y, -p.x⟩

def apply_transformations (trans : list (Point → Point)) (triangle : list Point) : list Point :=
triangle.map (λ p, trans.foldr (λ f acc, f acc) p)

def count_sequences_return_to_position (T : list Point) : ℕ :=
(list.permutations [rotate_120, rotate_180, rotate_240, reflect_y_eq_x, reflect_y_eq_neg_x]).count (λ seq,
  apply_transformations (seq.take 3) T = T)

theorem number_of_valid_sequences : count_sequences_return_to_position T = 18 :=
sorry

end number_of_valid_sequences_l824_824385


namespace max_sum_consecutive_integers_less_360_l824_824472

theorem max_sum_consecutive_integers_less_360 :
  ∃ n : ℤ, n * (n + 1) < 360 ∧ (n + (n + 1)) = 37 :=
by
  sorry

end max_sum_consecutive_integers_less_360_l824_824472


namespace least_integer_greater_than_sqrt_500_l824_824070

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end least_integer_greater_than_sqrt_500_l824_824070
