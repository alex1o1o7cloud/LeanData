import Data.Finset.Basic
import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Power
import Mathlib.Algebra.Log
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Sequences
import Mathlib.Algebra.Vector
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry.Euclidean.Basic
import Mathlib.Analysis.Geometry.Objects
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Combinatorial
import Mathlib.Combinatorics.CombinatorialGameTheory
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combination
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Topology.Geometry.Euclidean
import ProbabilityTheory.Independence
import Tactic

namespace S9_equals_27_l526_526942

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {d : ℤ}

-- (Condition 1) The sequence is an arithmetic sequence: a_{n+1} = a_n + d
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d

-- (Condition 2) The sum S_n is the sum of the first n terms of the sequence
axiom sum_first_n_terms : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- (Condition 3) Given a_1 = 2 * a_3 - 3
axiom given_condition : a 1 = 2 * a 3 - 3

-- Prove that S_9 = 27
theorem S9_equals_27 : S 9 = 27 :=
by
  sorry

end S9_equals_27_l526_526942


namespace find_t_l526_526354

def z1 : Complex := 3 + 4 * Complex.i
def z2 (t : ℝ) : Complex := t + Complex.i

theorem find_t (t : ℝ) (h : (z1 * Complex.conj (z2 t)).im = 0) : t = 3 / 4 :=
by
  sorry

end find_t_l526_526354


namespace fraction_difference_l526_526693

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l526_526693


namespace minimum_value_and_period_cos_2A_eq_l526_526375

noncomputable def f (x : ℝ) : ℝ := 
  √3 * (sin (x + π / 4))^2 - (cos x)^2 - (1 + √3) / 2

theorem minimum_value_and_period :
  (∃ x : ℝ, f x = -2) ∧ (∀ T : ℝ, T > 0 → (∀ x : ℝ, f (x + T) = f x) ↔ T = π) :=
sorry

theorem cos_2A_eq (A : ℝ) (hA1 : 0 < A ∧ A < π / 2)
  (hA2 : 1 + 5 * f (π / 4 - A) = 0) :
  cos (2 * A) = (4 * √3 + 3) / 10 :=
sorry

end minimum_value_and_period_cos_2A_eq_l526_526375


namespace solve_paula_painter_lunch_break_duration_l526_526512

noncomputable def paula_painter_lunch_break_duration 
  (p h : ℝ) -- Paula's rate and Helpers' combined rate respectively
  (L : ℝ)  -- Duration of lunch break in hours
  (start_time : ℝ := 8) 
  (end_monday_time : ℝ := 17) (monday_fraction : ℝ := 0.6)
  (end_tuesday_time : ℝ := 15) (tuesday_fraction : ℝ := 0.3)
  (end_wednesday_time : ℝ := 18) (wednesday_fraction : ℝ := 0.1) : ℝ :=
  let monday_hours := end_monday_time - start_time - L in
  let tuesday_hours := end_tuesday_time - start_time - L in
  let wednesday_hours := end_wednesday_time - start_time - L in
  (monday_hours * (p + h) = monday_fraction) ∧ 
  (tuesday_hours * h = tuesday_fraction) ∧
  (wednesday_hours * p = wednesday_fraction) →
  L * 60 = 40

theorem solve_paula_painter_lunch_break_duration 
  (p h L : ℝ)
  (start_time end_monday_time end_tuesday_time end_wednesday_time monday_fraction tuesday_fraction wednesday_fraction : ℝ) 
  (h1 : end_monday_time - start_time - L = 9 - L)
  (h2 : end_tuesday_time - start_time - L = 7 - L)
  (h3 : end_wednesday_time - start_time - L = 10 - L)
  (h4 : (9 - L) * (p + h) = 0.6)
  (h5 : (7 - L) * h = 0.3)
  (h6 : (10 - L) * p = 0.1) : 
  L * 60 = 40 :=
begin
  let monday_hours := end_monday_time - start_time - L,
  let tuesday_hours := end_tuesday_time - start_time - L,
  let wednesday_hours := end_wednesday_time - start_time - L,
  -- The exact proof steps would proceed from here
  sorry
end

end solve_paula_painter_lunch_break_duration_l526_526512


namespace merchant_discount_l526_526255

-- Definitions based on conditions
def original_price : ℝ := 1
def increased_price : ℝ := original_price * 1.2
def final_price : ℝ := increased_price * 0.8
def actual_discount : ℝ := original_price - final_price

-- The theorem to be proved
theorem merchant_discount : actual_discount = 0.04 :=
by
  -- Proof goes here
  sorry

end merchant_discount_l526_526255


namespace maria_payment_l526_526120

noncomputable def calculate_payment : ℝ :=
  let regular_price := 15
  let first_discount := 0.40 * regular_price
  let after_first_discount := regular_price - first_discount
  let holiday_discount := 0.10 * after_first_discount
  let after_holiday_discount := after_first_discount - holiday_discount
  after_holiday_discount + 2

theorem maria_payment : calculate_payment = 10.10 :=
by
  sorry

end maria_payment_l526_526120


namespace total_tourists_transported_l526_526251

theorem total_tourists_transported :
  let num_trips := 10,
      first_trip := 120,
      decrement := 2 in
  let tourists := λ (n : ℕ), first_trip - n * decrement in
  let total := Finset.range num_trips |>.sum tourists in
  total = 1110 :=
by
  sorry

end total_tourists_transported_l526_526251


namespace car_speed_l526_526233

variable (Distance : ℕ) (Time : ℕ)
variable (h1 : Distance = 495)
variable (h2 : Time = 5)

theorem car_speed (Distance Time : ℕ) (h1 : Distance = 495) (h2 : Time = 5) : 
  Distance / Time = 99 :=
by
  sorry

end car_speed_l526_526233


namespace vision_data_l526_526544

theorem vision_data (L V : ℝ) (approx : ℝ) (h1 : L = 5 + real.log10 V) (h2 : L = 4.9) (h3 : approx = 1.259) : 
  V = 0.8 := 
sorry

end vision_data_l526_526544


namespace number_of_distinct_tilings_l526_526231

theorem number_of_distinct_tilings : 
  let length := 10
  let tile_lengths := {n : ℕ | 1 ≤ n ∧ n ≤ 10}
  let colors := {green, yellow}
  (length = 10) → (∀ n ∈ tile_lengths, 1 ≤ n ∧ n ≤ 10) → (∃ g ∈ colors, ∃ y ∈ colors, g ≠ y) → (∀ tiles : list (ℕ × color), (tiles.foldl (+) 0 (map prod.fst tiles) = 10) ∧ (∀ i < length - 1, (tiles.nth i).2 ≠ (tiles.nth (i + 1)).2)) →
  (∃ total_tilings : ℕ, total_tilings = 1022) := 
begin
  intros,
  use 1022,
  sorry
end

end number_of_distinct_tilings_l526_526231


namespace integer_points_in_circle_l526_526332

theorem integer_points_in_circle : 
  {x : ℤ | (x - 3) ^ 2 + (2 * x + 4) ^ 2 ≤ 64}.card = 15 := 
by
  sorry

end integer_points_in_circle_l526_526332


namespace compare_a_b_c_l526_526847

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l526_526847


namespace exists_five_numbers_l526_526205

theorem exists_five_numbers :
  ∃ a1 a2 a3 a4 a5 : ℤ,
  a1 + a2 < 0 ∧
  a2 + a3 < 0 ∧
  a3 + a4 < 0 ∧
  a4 + a5 < 0 ∧
  a5 + a1 < 0 ∧
  a1 + a2 + a3 + a4 + a5 > 0 :=
by
  sorry

end exists_five_numbers_l526_526205


namespace find_f_2019_l526_526027

noncomputable def f : ℝ → ℝ 
| x := if x < 1 then 2 ^ x else (x + 1) / x * f (x - 1)

theorem find_f_2019 : f 2019 = 2020 :=
by sorry

end find_f_2019_l526_526027


namespace part1_part2_l526_526343

variable (α : ℝ)

theorem part1 (h : Real.tan α = 2) : (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 := 
by
  sorry

theorem part2 (h : Real.tan α = 2) : Real.sin α ^ 2 + Real.sin (2 * α) = 8 / 5 :=
by
  sorry

end part1_part2_l526_526343


namespace sum_possible_values_k_l526_526465

open Nat

theorem sum_possible_values_k (j k : ℕ) (h : (1 / j : ℚ) + 1 / k = 1 / 4) : 
  ∃ ks : List ℕ, (∀ k' ∈ ks, ∃ j', (1 / j' : ℚ) + 1 / k' = 1 / 4) ∧ ks.sum = 51 :=
by
  sorry

end sum_possible_values_k_l526_526465


namespace probability_heads_10_out_of_12_l526_526599

theorem probability_heads_10_out_of_12 :
  let total_outcomes := (2^12 : ℕ)
  let favorable_outcomes := nat.choose 12 10
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 66 / 4096 :=
by 
  sorry

end probability_heads_10_out_of_12_l526_526599


namespace fraction_option_C_l526_526619

def is_fraction (expr : String) : Prop := 
  expr = "fraction"

def option_C_fraction (x : ℝ) : Prop :=
  ∃ (numerator : ℝ), ∃ (denominator : ℝ), 
  numerator = 2 ∧ denominator = x + 3

theorem fraction_option_C (x : ℝ) (h : x ≠ -3) :
  is_fraction "fraction" ↔ option_C_fraction x :=
by 
  sorry

end fraction_option_C_l526_526619


namespace min_moves_equalize_coins_l526_526142

theorem min_moves_equalize_coins :
  let chests := [20, 15, 17, 18, 6, 5, 10]
  ∀ moves, (moves = 22) ↔ (∃ chests' (all_eq : ∀ x ∈ chests', x = 13), transition_by_moves chests chests' moves)
:=
sorry

/-- A helper function (or axiom) to represent the concept of moving coins between chests in specific moves --/
axiom transition_by_moves : list ℕ → list ℕ → ℕ → Prop

end min_moves_equalize_coins_l526_526142


namespace valid_wave_number_count_l526_526256

def is_wave_number (n : ℕ) : Prop :=
  let d1 := (n / 10000) % 10
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := (n / 1) % 10
  (d1 > d2 ∧ d2 > d3 ∧ d3 < d4 ∧ d4 > d5 ∨ d2 > d3 ∧ d3 < d4 ∧ d4 > d5 ∧ d4 > d1)

def is_valid_wave_number (n : ℕ) : Prop :=
  let digits := [(n / 10000) % 10, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, (n % 10)]
  digits.eraseDuplicates = digits ∧ digits.sorted = [1, 2, 3, 4, 5] ∧ is_wave_number n

noncomputable def count_valid_wave_numbers : ℕ :=
  Finset.univ.filter (λ n, 10000 ≤ n ∧ n < 100000 ∧ is_wave_number n ∧ is_valid_wave_number n).card

theorem valid_wave_number_count : count_valid_wave_numbers = 11 := by
  sorry

end valid_wave_number_count_l526_526256


namespace negation_of_implication_l526_526514

theorem negation_of_implication (a b : ℝ) :
  (¬ (a^2 + b^2 = 0 → a = 0 ∧ b = 0)) ↔ (a^2 + b^2 ≠ 0 → a ≠ 0 ∨ b ≠ 0) :=
begin
  sorry
end

end negation_of_implication_l526_526514


namespace greg_situps_calculation_greg_situps_result_l526_526792

variable (peter_situps greg_ratio : ℕ)
variable (greg_situps : ℕ)
hypothesis (ratio : greg_ratio = 4)
hypothesis (peter_situps_done : peter_situps = 24)

theorem greg_situps_calculation : greg_situps = peter_situps * greg_ratio := by
  sorry

theorem greg_situps_result : greg_situps = 96 :=
  greg_situps_calculation 24 4 rfl sorry sorry

end greg_situps_calculation_greg_situps_result_l526_526792


namespace total_cost_of_gas_l526_526797

theorem total_cost_of_gas :
  ∃ x : ℚ, (4 * (x / 4) - 4 * (x / 7) = 40) ∧ x = 280 / 3 :=
by
  sorry

end total_cost_of_gas_l526_526797


namespace real_imaginary_parts_l526_526570

theorem real_imaginary_parts : ∃ a b : ℝ, (∀ c : ℂ, c = (1 + real.sqrt 3) * complex.i → (complex.re c = a ∧ complex.im c = b)) ∧ a = 0 ∧ b = 1 + real.sqrt 3 :=
by
  existsi (0 : ℝ)
  existsi (1 + real.sqrt 3 : ℝ)
  intro c h
  rw h
  split
  · rw complex.re_mul_i
    rw complex.re_of_real
    norm_num
  · rw complex.im_mul_i
    norm_cast
    norm_num
  split
  · refl
  · refl
  sorry

end real_imaginary_parts_l526_526570


namespace smallest_integer_with_eight_prime_power_divisors_l526_526614

theorem smallest_integer_with_eight_prime_power_divisors : ∃ n : ℕ,
  (∀ d : ℕ, d ∣ n → ∃ p k : ℕ, nat.prime p ∧ d = p ^ k) ∧ 
  finset.card (finset.filter (λ d, d ∣ n) (finset.range (n + 1))) = 8 ∧ 
  ∀ m : ℕ, (∀ d : ℕ, d ∣ m → ∃ p k : ℕ, nat.prime p ∧ d = p ^ k) ∧ 
           finset.card (finset.filter (λ d, d ∣ m) (finset.range (m + 1))) = 8 → n ≤ m := 
begin
  use 24,
  split,
  { intros d hd,
    sorry
  },
  split,
  { sorry
  },
  intros m hm,
  sorry
end

end smallest_integer_with_eight_prime_power_divisors_l526_526614


namespace trapezium_area_l526_526778

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem trapezium_area :
  (1/2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
  sorry

end trapezium_area_l526_526778


namespace find_m_n_l526_526384

theorem find_m_n (m n : ℕ) 
  (h1: 2 + (m + 1) = 6) 
  (h2: 2n + (5 - m) = 6): 
  m = 3 ∧ n = 2 := 
  by 
    sorry

end find_m_n_l526_526384


namespace solution_Y_required_l526_526984

theorem solution_Y_required (V_total V_ratio_Y : ℝ) (h_total : V_total = 0.64) (h_ratio : V_ratio_Y = 3 / 8) : 
  (0.64 * (3 / 8) = 0.24) :=
by
  sorry

end solution_Y_required_l526_526984


namespace percent_increase_in_price_per_ounce_l526_526208

-- Definitions for conditions
variables (W P : ℝ) (h_weight_reduction : W > 0) (h_price_unchanged : P > 0)
def new_weight := 0.75 * W
def old_price_per_ounce := P / W
def new_price_per_ounce := P / new_weight

-- Target proof statement
theorem percent_increase_in_price_per_ounce (h_weight_reduction : W > 0) (h_price_unchanged : P > 0) :
  ((new_price_per_ounce W P h_weight_reduction h_price_unchanged) / (old_price_per_ounce W P h_weight_reduction h_price_unchanged) - 1) * 100 = 33.33 :=
by
  sorry

end percent_increase_in_price_per_ounce_l526_526208


namespace min_value_2xy_div_x_y_minus_1_l526_526062

theorem min_value_2xy_div_x_y_minus_1 (x y : ℝ) (h : x^2 + y^2 = 1) : (∃ z : ℝ, z = (2 * x * y) / (x + y - 1) ∧ z ≥ 1 - real.sqrt 2) :=
by
  sorry

end min_value_2xy_div_x_y_minus_1_l526_526062


namespace probability_of_10_heads_in_12_flips_l526_526596

open_locale big_operators

noncomputable def calculate_probability : ℕ → ℕ → ℚ := 
  λ n k, (nat.choose n k : ℚ) / (2 ^ n)

theorem probability_of_10_heads_in_12_flips :
  calculate_probability 12 10 = 66 / 4096 :=
by
  sorry

end probability_of_10_heads_in_12_flips_l526_526596


namespace probability_heads_10_out_of_12_l526_526601

theorem probability_heads_10_out_of_12 :
  let total_outcomes := (2^12 : ℕ)
  let favorable_outcomes := nat.choose 12 10
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 66 / 4096 :=
by 
  sorry

end probability_heads_10_out_of_12_l526_526601


namespace claudia_coins_l526_526294

theorem claudia_coins (x y : ℕ) (h1 : x + y = 15) (h2 : 29 - x = 26) :
  y = 12 :=
by
  sorry

end claudia_coins_l526_526294


namespace prove_a4_plus_1_div_a4_l526_526532

theorem prove_a4_plus_1_div_a4 (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/(a^4) = 7 :=
by
  sorry

end prove_a4_plus_1_div_a4_l526_526532


namespace value_of_expression_l526_526573

theorem value_of_expression : 1 + 2 + 3 - 4 + 5 + 6 + 7 - 8 + 9 + 10 + 11 - 12 = 30 :=
by
  sorry

end value_of_expression_l526_526573


namespace min_m_min_expression_l526_526763

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Part (Ⅰ)
theorem min_m (m : ℝ) (h : ∃ x₀ : ℝ, f x₀ ≤ m) : m ≥ 2 := sorry

-- Part (Ⅱ)
theorem min_expression (a b : ℝ) (h1 : 3 * a + b = 2) (h2 : 0 < a) (h3 : 0 < b) :
  (1 / (2 * a) + 1 / (a + b)) ≥ 2 := sorry

end min_m_min_expression_l526_526763


namespace ratio_AH_OD_two_to_one_l526_526362

noncomputable def point := ℝ × ℝ

variables {A B C H O D : point}

-- Conditions
variables (H_is_orthocenter : (is_orthocenter H A B C))
variables (O_is_circumcenter : (is_circumcenter O A B C))
variables (OD_perp_BC : perp O D (B, C))

-- Definition of perp: A point is perpendicular to the line 
-- segment formed by two points.

open_locale classical

theorem ratio_AH_OD_two_to_one :
  AH_ratio_OD A B C H O D H_is_orthocenter O_is_circumcenter OD_perp_BC = 2 :=
sorry

end ratio_AH_OD_two_to_one_l526_526362


namespace round_robin_tournament_matches_l526_526923

theorem round_robin_tournament_matches (n : Nat) (h : n = 10) : (n * (n - 1)) / 2 = 45 :=
by
  rw [h]
  sorry

end round_robin_tournament_matches_l526_526923


namespace sum_possible_values_k_l526_526466

open Nat

theorem sum_possible_values_k (j k : ℕ) (h : (1 / j : ℚ) + 1 / k = 1 / 4) : 
  ∃ ks : List ℕ, (∀ k' ∈ ks, ∃ j', (1 / j' : ℚ) + 1 / k' = 1 / 4) ∧ ks.sum = 51 :=
by
  sorry

end sum_possible_values_k_l526_526466


namespace inscribed_sphere_tangent_point_of_tetrahedron_l526_526799

theorem inscribed_sphere_tangent_point_of_tetrahedron
  (eq_triangle : Type)
  (eq_tetrahedron : Type)
  (inscribed_circle : eq_triangle → Type)
  (inscribed_sphere : eq_tetrahedron → Type)
  (tangent_midpoints : ∀ (t : eq_triangle) (c : inscribed_circle t), ∀ (side : t.side), c.is_tangent (t.midpoint side))
  (t : eq_tetrahedron)
  (s : inscribed_sphere t) :
  ∀ (face : t.face), ∃ (point : face.point), s.is_tangent point ∧ point.is_trisection_of_altitude :=
sorry

end inscribed_sphere_tangent_point_of_tetrahedron_l526_526799


namespace arithmetic_sequence_a1_l526_526965

theorem arithmetic_sequence_a1 :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∑ i in Finset.range 50, (a 1 + i * d) = 200) →
  (∑ i in Finset.range 50, (a 51 + i * d) = 2700) →
  a 1 = -20.5 :=
by
  intros a d h1 h2
  sorry

end arithmetic_sequence_a1_l526_526965


namespace revenue_growth_20_percent_l526_526209

noncomputable def revenue_increase (R2000 R2003 R2005 : ℝ) : ℝ :=
  ((R2005 - R2003) / R2003) * 100

theorem revenue_growth_20_percent (R2000 : ℝ) (h1 : R2003 = 1.5 * R2000) (h2 : R2005 = 1.8 * R2000) :
  revenue_increase R2000 R2003 R2005 = 20 :=
by
  sorry

end revenue_growth_20_percent_l526_526209


namespace number_of_new_terms_l526_526593

theorem number_of_new_terms (n : ℕ) (h : n > 1) :
  (2^(n+1) - 1) - (2^n - 1) + 1 = 2^n := by
sorry

end number_of_new_terms_l526_526593


namespace larry_channels_l526_526098

theorem larry_channels : 
  let initial_channels := 150
  let step1_channels := initial_channels - 20 + 12
  let step2_channels := step1_channels - 10 + 8
  let step3_channels := step2_channels + 15 - 5
  let international_overlap := step3_channels * 10 / 100
  let step4_channels := step3_channels + 25 - international_overlap
  let final_channels := step4_channels + 7 - 3
  in final_channels = 164 := by
  sorry

end larry_channels_l526_526098


namespace monotonic_intervals_extreme_points_inequality_l526_526030

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (1 / 2) * x ^ 2 + m * real.log (1 - x)

theorem monotonic_intervals (m : ℝ) :
  (if m ≤ 0 then
      decreasing_interval : set ℝ := {x | x < (1 - real.sqrt (1 - 4 * m)) / 2} ∧ 
      increasing_interval : set ℝ := {x | (1 - real.sqrt (1 - 4 * m)) / 2 < x ∧ x < 1}
   else if 0 < m ∧ m < 1 / 4 then
      decreasing_interval_1 : set ℝ := {x | x < (1 - real.sqrt (1 - 4 * m)) / 2} ∧ 
      decreasing_interval_2 : set ℝ := {x | (1 + real.sqrt (1 - 4 * m)) / 2 < x ∧ x < 1} ∧ 
      increasing_interval : set ℝ := {x | (1 - real.sqrt (1 - 4 * m)) / 2 < x ∧ x < (1 + real.sqrt (1 - 4 * m)) / 2}
   else
      decreasing_interval : set ℝ := {x | x < 1}) :=
  sorry

theorem extreme_points_inequality {x1 x2 : ℝ} (m : ℝ) (h1 : 0 < m ∧ m < 1 / 4) (h2 : x1 + x2 = 1) (h3 : x1 * x2 = m) (hx : x1 < x2) :
  f x1 m + f x2 m > 1 / 4 - 1 / 4 * real.log 4 :=
  sorry

end monotonic_intervals_extreme_points_inequality_l526_526030


namespace math_problem_l526_526147

variable {x : ℕ → ℝ} {n : ℕ}

theorem math_problem (h1 : ∀ i, 1 ≤ i → i ≤ n → 0 < x i)
                     (h2 : 2 ≤ n)
                     (h3 : ∑ i in Finset.range (n+1), x i = 1) :
  (∑ i in Finset.range (n+1), x i / real.sqrt (1 - x i)) ≥ 
  (1 / real.sqrt (n - 1)) * (∑ i in Finset.range (n+1), real.sqrt (x i)) :=
sorry

end math_problem_l526_526147


namespace distance_IN_gt_IM_l526_526973

variables {α : Type*} [MetricSpace α]
variables {A B I M A_1 B_1 N : α}

-- Definitions from the problem conditions
def midpoint (X Y : α) : α := sorry
def reflection (X L : α) : α := sorry
def incenter (A B C : α) : α := sorry

axiom midpoint_AB : M = midpoint A B
axiom incenter_ABC : I = incenter A B _
axiom reflection_A_BI : A_1 = reflection A I
axiom reflection_B_AI : B_1 = reflection B I
axiom midpoint_A1B1 : N = midpoint A_1 B_1

-- Proposition to prove
theorem distance_IN_gt_IM (h1 : midpoint_AB) (h2 : incenter_ABC) (h3 : reflection_A_BI) (h4 : reflection_B_AI) (h5 : midpoint_A1B1) : 
  dist I N > dist I M :=
sorry

end distance_IN_gt_IM_l526_526973


namespace find_a_l526_526059

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

theorem find_a (a : ℝ) (extremum_condition : ∀ x, deriv (f a) x = 0 → x = 1) : a = Real.exp 1 :=
by
  let f' := λ x, (Real.exp x - a)
  have derivative_at_1 : deriv (f a) 1 = f' 1 := sorry -- Proof that the derivative matches f'
  have zero_slope_at_1 : deriv (f a) 1 = 0 := extremum_condition 1 (by simp [derivative_at_1])
  rw [derivative_at_1, Real.exp_one] at zero_slope_at_1
  exact (eq_of_sub_eq_zero zero_slope_at_1.symm)

end find_a_l526_526059


namespace repeating_decimal_exceeds_finite_decimal_by_l526_526732

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l526_526732


namespace probability_heads_exactly_10_out_of_12_flips_l526_526603

theorem probability_heads_exactly_10_out_of_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := Nat.choose 12 10
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = (66 : ℝ) / 4096 :=
by
  let total_outcomes := 2^12
  let favorable_outcomes := Nat.choose 12 10
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  have total_outcomes_val : total_outcomes = 4096 := by norm_num
  have favorable_outcomes_val : favorable_outcomes = 66 := by norm_num
  have probability_val : probability = (66 : ℝ) / 4096 := by rw [favorable_outcomes_val, total_outcomes_val]
  exact probability_val

end probability_heads_exactly_10_out_of_12_flips_l526_526603


namespace transform_cos_to_base_form_l526_526583

theorem transform_cos_to_base_form :
  let f (x : ℝ) := Real.cos (2 * x + (Real.pi / 3))
  let g (x : ℝ) := Real.cos (2 * x)
  ∃ (shift : ℝ), shift = Real.pi / 6 ∧
    (∀ x : ℝ, f (x - shift) = g x) :=
by
  let f := λ x : ℝ => Real.cos (2 * x + (Real.pi / 3))
  let g := λ x : ℝ => Real.cos (2 * x)
  use Real.pi / 6
  sorry

end transform_cos_to_base_form_l526_526583


namespace sym_func_range_l526_526035

-- Statement of the problem in Lean 4
theorem sym_func_range (f g : ℝ → ℝ) (m : ℝ) 
  (hf : ∀ x, f x = log x - x^2)
  (hg : ∀ x, g x = (x - 2)^2 - 1 / (2 * x - 4) - m)
  (symm : ∀ x, f (2 - x) = g x) :
  ∃ m, ∀ x, g x = log (2 - x) - 1 / (2 * (2 - x) - 4) →
       m ∈ Set.Ici (1 - log 2) :=
by
  sorry -- Proof is not required

end sym_func_range_l526_526035


namespace fraction_difference_l526_526690

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l526_526690


namespace minimum_value_l526_526909

noncomputable def condition (x : ℝ) : Prop := (2 * x - 1) / 3 - 1 ≥ x - (5 - 3 * x) / 2

noncomputable def target_function (x : ℝ) : ℝ := abs (x - 1) - abs (x + 3)

theorem minimum_value :
  ∃ x : ℝ, condition x ∧ ∀ y : ℝ, condition y → target_function y ≥ target_function x :=
sorry

end minimum_value_l526_526909


namespace alpha_range_theorem_l526_526932

noncomputable def alpha_range (k : ℤ) (α : ℝ) : Prop :=
  2 * k * Real.pi - Real.pi ≤ α ∧ α ≤ 2 * k * Real.pi

theorem alpha_range_theorem (α : ℝ) (k : ℤ) (h : |Real.sin (4 * Real.pi - α)| = Real.sin (Real.pi + α)) :
  alpha_range k α :=
by
  sorry

end alpha_range_theorem_l526_526932


namespace max_jars_with_same_coins_l526_526576

theorem max_jars_with_same_coins (d : ℕ) (h1 : d > 0) : ∃ N, N ≤ 2014 ∧ ∀ jars : fin 2017 → ℕ, 
  (∀ n, ∃ fn, fn jars n) := sorry

end max_jars_with_same_coins_l526_526576


namespace square_area_l526_526677

theorem square_area (x : ℝ) 
  (h1 : 5 * x - 18 = 27 - 4 * x) 
  (side_length : ℝ := 5 * x - 18) : 
  side_length ^ 2 = 49 := 
by 
  sorry

end square_area_l526_526677


namespace correct_inequality_l526_526014

theorem correct_inequality (x : ℝ) : (1 / (x^2 + 1)) > (1 / (x^2 + 2)) :=
by {
  -- Lean proof steps would be here, but we will use 'sorry' instead to indicate the proof is omitted.
  sorry
}

end correct_inequality_l526_526014


namespace orthocenter_of_triangle_l526_526961

theorem orthocenter_of_triangle :
  ∀ (A B C H : ℝ × ℝ × ℝ),
    A = (2, 3, 4) → 
    B = (6, 4, 2) → 
    C = (4, 5, 6) → 
    H = (17/53, 152/53, 725/53) → 
    true :=
by sorry

end orthocenter_of_triangle_l526_526961


namespace four_digit_numbers_with_conditions_l526_526594

theorem four_digit_numbers_with_conditions :
  let digits := {1, 2, 3, 4, 5, 6, 7}
  let odd_digits := {1, 3, 5, 7}
  let even_digits := {2, 4, 6}
  ∃ (nums : Finset ℕ), nums.card = 4 ∧
    nums ⊆ digits ∧
    (nums.filter (λ x, x % 2 = 0)).card ≤ 1
  → (∑ nums in Finset.powersetLen 4 digits, 
           if (nums.filter (λ x, x % 2 = 0)).card ≤ 1 then 1 else 0) = 312 :=
by
  let digits := {1, 2, 3, 4, 5, 6, 7}
  let odd_digits := {1, 3, 5, 7}
  let even_digits := {2, 4, 6}
  sorry

end four_digit_numbers_with_conditions_l526_526594


namespace angle_at_630_is_15_degrees_l526_526610

-- Definitions for positions of hour and minute hands at 6:30 p.m.
def angle_per_hour : ℝ := 30
def minute_hand_position_630 : ℝ := 180
def hour_hand_position_630 : ℝ := 195

-- The angle between the hour hand and minute hand at 6:30 p.m.
def angle_between_hands_630 : ℝ := |hour_hand_position_630 - minute_hand_position_630|

-- Statement to prove
theorem angle_at_630_is_15_degrees :
  angle_between_hands_630 = 15 := by
  sorry

end angle_at_630_is_15_degrees_l526_526610


namespace profit_without_discount_l526_526211

theorem profit_without_discount (CP SP_with_discount SP_without_discount : ℝ) (h1 : CP = 100) (h2 : SP_with_discount = CP + 0.235 * CP) (h3 : SP_with_discount = 0.95 * SP_without_discount) : (SP_without_discount - CP) / CP * 100 = 30 :=
by
  sorry

end profit_without_discount_l526_526211


namespace angle_congruence_parallelogram_l526_526567

open EuclideanGeometry

theorem angle_congruence_parallelogram
  {A B C D M : Point}
  (h₁ : inside_parallelogram A B C D M) : 
  (angle A M B = angle C M B) ↔ (angle B M A = angle D M A) :=
by sorry

end angle_congruence_parallelogram_l526_526567


namespace sum_of_possible_k_l526_526437

theorem sum_of_possible_k (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  j > 0 ∧ k > 0 → 
  ∑ i in {20, 12, 8, 6, 5}.to_finset, i = 51 :=
by
  sorry

end sum_of_possible_k_l526_526437


namespace excess_common_fraction_l526_526716

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l526_526716


namespace distribution_of_tickets_l526_526577

def tickets := {1, 2, 3, 4, 5, 6, 7}
def people := {'A', 'B', 'C', 'D', 'E'}

theorem distribution_of_tickets :
  let groups := { g ⊆ tickets | g ≠ ∅ ∧ |g| ≤ 2 ∧ (∀ x y ∈ g, abs (x - y) = 1 ∨ abs (x - y) = 0) } in
  ∃ (distributions : set (set (group → people))),
      |distributions| = 1200 :=
sorry

end distribution_of_tickets_l526_526577


namespace tan_x_tan_2x_l526_526999

theorem tan_x_tan_2x (a b : ℝ) (h1 : tan x = a / b) (h2 : tan (2 * x) = b / (a + b)) :
  let k := (1 / 3 : ℝ) in atan k = x :=
sorry

end tan_x_tan_2x_l526_526999


namespace compare_constants_l526_526839

theorem compare_constants :
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  in c < a ∧ a < b :=
by
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  sorry

end compare_constants_l526_526839


namespace non_congruent_squares_in_6x6_grid_l526_526391

theorem non_congruent_squares_in_6x6_grid : 
  let grid_size := 6
  let regular_squares := (let f := λ n, (grid_size - n) ^ 2 in f 1 + f 2 + f 3 + f 4 + f 5),
  let diagonal_squares := (let f := λ n, (grid_size - (n + 1)) ^ 2 in f 1 + f 2 + f 3)
  in regular_squares + diagonal_squares = 105
:= by
  let grid_size := 6
  let f n := (grid_size - n) ^ 2
  have regular_squares : ∀ n, f 1 + f 2 + f 3 + f 4 + f 5 = 25 + 16 + 9 + 4 + 1 := sorry
  let diagonal_f n := (grid_size - (n + 1)) ^ 2
  have diagonal_squares : ∀ n, diagonal_f 1 + diagonal_f 2 + diagonal_f 3 = 25 + 16 + 9 := sorry
  have total_squares : regular_squares + diagonal_squares = 25 + 16 + 9 + 4 + 1 + 25 + 16 + 9 := sorry
  exact (regular_squares + diagonal_squares = 105) sorry

end non_congruent_squares_in_6x6_grid_l526_526391


namespace shirt_price_before_discount_l526_526631

noncomputable def original_price (final_price : ℝ) : ℝ :=
  let discount_factor : ℝ := 0.5625 in
  final_price / discount_factor

theorem shirt_price_before_discount (h : original_price 15 = 26.67) : true :=
by {
  sorry
}

end shirt_price_before_discount_l526_526631


namespace area_of_trapezium_l526_526775

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
by sorry

end area_of_trapezium_l526_526775


namespace eventual_survival_of_amoebas_l526_526951

theorem eventual_survival_of_amoebas (m n : ℤ) : 
  (∀ t : ℕ, (let (mt, nt) := (2^t * m - (2^t - 1) * n, 2^t * n - (2^t - 1) * m) 
             in (mt < 0 ∨ nt < 0)) ∨ m = n) := by
  sorry

end eventual_survival_of_amoebas_l526_526951


namespace necessary_and_sufficient_condition_for_equal_areas_l526_526425

variable (r : ℝ) (φ : ℝ)
variable (h₁ : 0 < φ) (h₂ : φ < real.pi / 2)

theorem necessary_and_sufficient_condition_for_equal_areas : 
    tan φ = 4 * φ := 
sorry

end necessary_and_sufficient_condition_for_equal_areas_l526_526425


namespace max_possible_area_quadrilateral_l526_526609

def Brahmagupta (a b c d s : ℝ) : ℝ :=
  Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))

theorem max_possible_area_quadrilateral
  {a b c d : ℝ} (h : a = 1) (h1 : b = 4) (h2 : c = 7) (h3 : d = 8) :
  let s := (a + b + c + d) / 2 
  in Brahmagupta a b c d s = 18 :=
by
  sorry

end max_possible_area_quadrilateral_l526_526609


namespace compare_abc_l526_526824
open Real

noncomputable def a : ℝ := 0.1 * exp(0.1)
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - log 0.9

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l526_526824


namespace MK_minus_ML_eq_AB_minus_CD_l526_526681

variables {A B C D M P K L : Point}
variables {AB CD MK ML : Real}

-- Conditions
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry
def circles_with_diameters_tangent (AB CD : Real) (M : Point) : Prop := sorry
def midpoint (P : Point) (A B : Point) : Prop := sorry
def concyclic_points (A M C K : Point) : Prop := sorry
def concyclic_points' (B M D L : Point) : Prop := sorry

-- Problem statement
theorem MK_minus_ML_eq_AB_minus_CD
  (h_convex : is_convex_quadrilateral A B C D)
  (h_tangent : circles_with_diameters_tangent AB CD M)
  (h_midpoint_P : midpoint P A B)
  (h_concyclic_1 : concyclic_points A M C K)
  (h_concyclic_2 : concyclic_points' B M D L) :
  |dist M K - dist M L| = |AB - CD| :=
sorry

end MK_minus_ML_eq_AB_minus_CD_l526_526681


namespace num_satisfying_a1_l526_526109

def sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if a (n-1) % 2 = 0 then a (n-1) / 2 else 3 * a (n-1) + 1

theorem num_satisfying_a1 {n : ℕ} :
  ∃ count : ℕ, count = 750 ∧ (∀ a1, a1 ≤ 3000 →
    (a1 % 4 = 3 → a1 < sequence id 2 ∧ a1 < sequence id 3 ∧ a1 < sequence id 4 ∧ a1 < sequence id 5)) →
    count = cardinal.mk {a1 ∣ a1 ∈ {1..3000} ∧ a1 % 4 = 3}
:=
sorry

end num_satisfying_a1_l526_526109


namespace sarah_copies_pages_l526_526526

/-- Sarah is in charge of making 2 copies of a contract for 9 people that will be in a meeting.
The contract is 20 pages long. Let's prove that Sarah will copy 360 pages. -/
theorem sarah_copies_pages
  (copies_per_person : ℕ)
  (people : ℕ)
  (pages_per_contract : ℕ)
  (total_copies : ℕ := copies_per_person * people)
  (total_pages : ℕ := total_copies * pages_per_contract) :
  copies_per_person = 2 →
  people = 9 →
  pages_per_contract = 20 →
  total_pages = 360 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp only [Nat.mul_eq_mul_left_iff]
  -/
  -/
  unfold total_copies
  unfold total_pages
  calc 2 * 9 * 20 = 18 * 20 : by rw Nat.one_mul
               ... = 360 : by rw Nat.mul_comm
  sorry

end sarah_copies_pages_l526_526526


namespace find_erased_integer_l526_526131

theorem find_erased_integer {n : ℤ} (h : n ≥ 0) (sum_remaining : ℤ) (h2 : sum_remaining = 153) 
  (cond : ∃ x : ℤ, x ∈ {n, n+1, n+2, n+3, n+4} ∧ (sum_remaining = (5 * n + 10) - x)) : 
  ∃ x : ℤ, x = 37 :=
by
  sorry

end find_erased_integer_l526_526131


namespace y_pow_one_div_x_neq_x_pow_y_l526_526345

theorem y_pow_one_div_x_neq_x_pow_y (t : ℝ) (ht : t > 1) : 
  let x := t ^ (2 / (t - 1))
  let y := t ^ (3 / (t - 1))
  (y ^ (1 / x) ≠ x ^ y) :=
by
  let x := t ^ (2 / (t - 1))
  let y := t ^ (3 / (t - 1))
  sorry

end y_pow_one_div_x_neq_x_pow_y_l526_526345


namespace compare_a_b_c_l526_526816

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b :=
by
  have : c < b := sorry,
  have : a < b := sorry,
  have : a > c := sorry,
  exact ⟨this.2, this.1⟩

end compare_a_b_c_l526_526816


namespace min_value_a_plus_b_l526_526488

theorem min_value_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : Real.sqrt (3^a * 3^b) = 3^((a + b) / 2)) : a + b = 4 := by
  sorry

end min_value_a_plus_b_l526_526488


namespace repaint_all_numbers_white_l526_526165

theorem repaint_all_numbers_white :
  ∃ (f : ℕ → Prop), (∀ n ∈ {1, ..., 1000000}, f n) :=
sorry

end repaint_all_numbers_white_l526_526165


namespace algebraic_expression_value_l526_526045

theorem algebraic_expression_value (a : ℝ) (h : (a^2 - 3) * (a^2 + 1) = 0) : a^2 = 3 :=
by
  sorry

end algebraic_expression_value_l526_526045


namespace compare_a_b_c_l526_526814

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b :=
by
  have : c < b := sorry,
  have : a < b := sorry,
  have : a > c := sorry,
  exact ⟨this.2, this.1⟩

end compare_a_b_c_l526_526814


namespace new_cookie_radius_eq_sqrt_two_l526_526273

noncomputable def radius_of_new_cookie (r_large r_small : ℝ) (n : ℕ) (area_large : ℝ) :=
  let area_small := n * (π * r_small ^ 2)
  let area_remaining := area_large - area_small
  let r_new := sqrt (area_remaining / π)
  r_new

-- Conditions
def r_large : ℝ := 3
def r_small : ℝ := 1
def n : ℕ := 7
def area_large := π * r_large ^ 2

-- Theorem (proof problem)
theorem new_cookie_radius_eq_sqrt_two : radius_of_new_cookie r_large r_small n area_large = sqrt 2 := by
  sorry

end new_cookie_radius_eq_sqrt_two_l526_526273


namespace most_accurate_value_l526_526074

-- Define the given constants and conditions
def D : ℝ := 3.7194
def error : ℝ := 0.00456
def D_upper : ℝ := D + error
def D_lower : ℝ := D - error

-- Stating the theorem
theorem most_accurate_value : D_upper.round 1 = 3.7 ∧ D_lower.round 1 = 3.7 :=
by
  sorry

end most_accurate_value_l526_526074


namespace function_correct_safe_entry_exit_times_l526_526167

noncomputable def tidal_function (t: ℝ) : ℝ :=
  2.5 * Real.sin (↑(Float.pi / 6) * t + Float.pi / 6) + 5

theorem function_correct :
  ∀ t : ℝ, tidal_function t = 2.5 * Real.sin (↑(Float.pi / 6) * t + Float.pi / 6) + 5 :=
by
  -- Proof to show tidal_function(t) matches given conditions
  sorry

theorem safe_entry_exit_times :
  ∀ t : ℝ, 
  (0 ≤ t ∧ t ≤ 4) ∨ (12 ≤ t ∧ t ≤ 16) -> 
  tidal_function t ≥ 6.25 :=
by
  -- Proof to show that the cargo ship can safely enter and exit within specified times
  sorry

end function_correct_safe_entry_exit_times_l526_526167


namespace student_percentage_to_pass_l526_526667

/-- A student needs to obtain 50% of the total marks to pass given the conditions:
    1. The student got 200 marks.
    2. The student failed by 20 marks.
    3. The maximum marks are 440. -/
theorem student_percentage_to_pass : 
  ∀ (student_marks : ℕ) (failed_by : ℕ) (max_marks : ℕ),
  student_marks = 200 → failed_by = 20 → max_marks = 440 →
  (student_marks + failed_by) / max_marks * 100 = 50 := 
by
  intros student_marks failed_by max_marks h1 h2 h3
  sorry

end student_percentage_to_pass_l526_526667


namespace greatest_integer_difference_l526_526630

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) : 
  ∃ d, d = y - x ∧ d = 2 := 
by
  sorry

end greatest_integer_difference_l526_526630


namespace petya_max_candies_l526_526127

-- Define initial conditions of the game
def initial_piles : List Nat := List.range 1 56  -- Generates 1 to 55 inclusive

-- Define the game and prove Petya's guaranteed candies consumption 
theorem petya_max_candies : (Petya: String) (Vasya: String) (takes_turns: ∀ Petya Vasya, true) (initial_piles = List.range 1 56) : 
    (∃ Petya_moves: List Nat, Petya_moves.head = 1) :=
by
    -- Initial step: Assume the conditions
    sorry

end petya_max_candies_l526_526127


namespace fraction_zero_implies_x_is_minus_5_l526_526065

theorem fraction_zero_implies_x_is_minus_5 (x : ℝ) (h1 : (x + 5) / (x - 2) = 0) (h2 : x ≠ 2) : x = -5 := 
by
  sorry

end fraction_zero_implies_x_is_minus_5_l526_526065


namespace vision_approximation_l526_526536

noncomputable
def five_point_to_decimal (L: ℝ) : ℝ := 10 ^ (L - 5)

theorem vision_approximation : 
  ∀ L : ℝ, L = 4.9 → (five_point_to_decimal L) ≈ 0.8 :=
by
  intros L hL
  rw hL
  sorry

end vision_approximation_l526_526536


namespace proof_problem_l526_526832

-- Define the constants a, b, c
def a := 0.1 * Real.exp 0.1
def b := 1 / 9
def c := -Real.log 0.9

-- State the theorem to prove
theorem proof_problem : c < a ∧ a < b := 
by
  sorry

end proof_problem_l526_526832


namespace smallest_expression_l526_526119

/-- Define y as the number \(0.\underbrace{0000...0000}_{2023\text{ zeros}}1\), which equals \(10^{-2024}\) -/
def y : ℝ := 10^(-2024)

/-- The main theorem stating that the smallest expression among the given options is \(\frac{y}{5}\) -/
theorem smallest_expression : 
  min (min (min (5 + y) (5 - y)) (min (5 * y) (5 / y))) (y / 5) = y / 5 :=
sorry

end smallest_expression_l526_526119


namespace triangle_inequality_sides_l526_526101

theorem triangle_inequality_sides
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  (a + b) * Real.sqrt (a * b) + (a + c) * Real.sqrt (a * c) + (b + c) * Real.sqrt (b * c) ≥ (a + b + c)^2 / 2 := 
by
  sorry

end triangle_inequality_sides_l526_526101


namespace excess_common_fraction_l526_526711

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l526_526711


namespace monotonicity_intervals_range_of_m_l526_526904

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + Real.log x) / x

theorem monotonicity_intervals (m : ℝ) (x : ℝ) (hx : x > 1):
  (m >= 1 → ∀ x' > 1, f m x' ≤ f m x) ∧
  (m < 1 → (∀ x' ∈ Set.Ioo 1 (Real.exp (1 - m)), f m x' > f m x) ∧
            (∀ x' ∈ Set.Ioi (Real.exp (1 - m)), f m x' < f m x)) := by
  sorry

theorem range_of_m (m : ℝ) :
  (∀ x > 1, f m x < m * x) ↔ m ≥ 1/2 := by
  sorry

end monotonicity_intervals_range_of_m_l526_526904


namespace investment_amounts_proof_l526_526215

noncomputable def investment_proof_statement : Prop :=
  let p_investment_first_year := 52000
  let q_investment := (5/4) * p_investment_first_year
  let r_investment := (6/4) * p_investment_first_year;
  let p_investment_second_year := p_investment_first_year + (20/100) * p_investment_first_year;
  (q_investment = 65000) ∧ (r_investment = 78000) ∧ (q_investment = 65000) ∧ (r_investment = 78000)

theorem investment_amounts_proof : investment_proof_statement :=
  by
    sorry

end investment_amounts_proof_l526_526215


namespace tangent_length_from_point_l526_526082

theorem tangent_length_from_point
  (A B C D E : ℝ)
  (hTriangle : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (hRightAngle : ∠ABC = 90)
  (hAC : dist A C = 9)
  (hBC : dist B C = 40)
  (hCD : is_altitude C D A B)
  (hCircleDiameter : is_diameter (dist D C) A E)
  (hTangent : is_tangent A E (dist D C)) :
  dist A E = sqrt 35 / 2 := 
sorry

end tangent_length_from_point_l526_526082


namespace complex_series_solution_l526_526117

noncomputable def complex_real_part_expansion (n : ℕ) : ℝ :=
  let expr := ∑ k in Finset.range (n / 2 + 1), ((-1 : ℤ) ^ k) * (3 ^ k) * (Nat.choose n (2 * k)) / (2 ^ n) in
  ↑expr -- converting ℤ to ℝ

theorem complex_series_solution (n : ℕ) (h : n = 1990) :
  complex_real_part_expansion n = -1 / 2 :=
by
  rw h
  sorry

end complex_series_solution_l526_526117


namespace n_cubed_minus_n_plus_one_is_square_l526_526330

theorem n_cubed_minus_n_plus_one_is_square (n : ℕ) (h : (n^5 + n^4 + 1).divisors.card = 6) : ∃ k : ℕ, n^3 - n + 1 = k^2 :=
sorry

end n_cubed_minus_n_plus_one_is_square_l526_526330


namespace sum_possible_values_k_l526_526462

open Nat

theorem sum_possible_values_k (j k : ℕ) (h : (1 / j : ℚ) + 1 / k = 1 / 4) : 
  ∃ ks : List ℕ, (∀ k' ∈ ks, ∃ j', (1 / j' : ℚ) + 1 / k' = 1 / 4) ∧ ks.sum = 51 :=
by
  sorry

end sum_possible_values_k_l526_526462


namespace find_eigenvalues_and_eigenvectors_l526_526320

noncomputable theory

def matrixA : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![3, 0, 0],
  ![1, 2, -1],
  ![1, -1, 2]
]

def eigenvalues (A : Matrix (Fin 3) (Fin 3) ℝ) : Fin 3 → ℝ := 
λ i, if i = 0 then 3 else if i = 1 then 3 else 1

def eigenvectors : Fin 3 → Fin 3 → ℝ
| 0, 0 := 1 | 0, 1 := 1 | 0, 2 := 0
| 1, 0 := 1 | 1, 1 := 0 | 1, 2 := 1
| 2, 0 := 0 | 2, 1 := 1 | 2, 2 := 1
| _, _ := 0

theorem find_eigenvalues_and_eigenvectors :
  ∃ (λ : Fin 3 → ℝ) (v : Fin 3 → Fin 3 → ℝ), 
  λ = eigenvalues matrixA ∧ v = eigenvectors ∧ 
  (∀ i, matrixA.mulVec (λ i • v i) = v i) :=
by
  use eigenvalues matrixA
  use eigenvectors
  split
  . exact rfl
  . split
  . exact rfl
  . intro i
    sorry

end find_eigenvalues_and_eigenvectors_l526_526320


namespace cos_one_sufficient_not_necessary_l526_526930

theorem cos_one_sufficient_not_necessary (x : ℝ) : 
  (cos x = 1 → sin x = 0) ∧ ¬(sin x = 0 → cos x = 1) :=
by 
  split
  · intro h
    rw [h, cos_one_eq]
    sorry -- proof for cos x = 1 implies sin x = 0
  · intro h
    contradiction -- proof for cos x = 1 is not necessary when sin x = 0

end cos_one_sufficient_not_necessary_l526_526930


namespace concurrency_of_trisectors_l526_526761

theorem concurrency_of_trisectors
    (A B C D E F X Y Z : Point)
    (is_triangle_ABC : Triangle A B C)
    (angle_trisected_at_each_vertex : ∀ (angle : Angle), trisected (external_angle angle))
    (trisectors_intersect_at_DEF : TrisectorsIntersect A B C D E F)
    (is_angle_bisector_AX : IsAngleBisector A X (InteriorAngle A B C))
    (is_angle_bisector_BY : IsAngleBisector B Y (InteriorAngle B A C))
    (is_angle_bisector_CZ : IsAngleBisector C Z (InteriorAngle C A B))
    (X_on_EF : OnSegment X E F)
    (Y_on_FD : OnSegment Y F D)
    (Z_on_DE : OnSegment Z D E) :
    Concurrent (D X) (E Y) (F Z) :=
sorry

end concurrency_of_trisectors_l526_526761


namespace area_of_one_trapezoid_l526_526954

theorem area_of_one_trapezoid (outer_area inner_area : ℝ)
  (h_outer : outer_area = 64)
  (h_inner : inner_area = 4)
  (trapezoids : 3) :
  (outer_area - inner_area) / trapezoids = 20 := by 
  sorry

end area_of_one_trapezoid_l526_526954


namespace problem_statement_l526_526102

theorem problem_statement (a b c : ℝ) (ha : a ^ 3 - 9 * a ^ 2 + 11 * a - 1 = 0) 
  (hb : b ^ 3 - 9 * b ^ 2 + 11 * b - 1 = 0) (hc : c ^ 3 - 9 * c ^ 2 + 11 * c - 1 = 0) :
  let s := (sqrt a + sqrt b + sqrt c) in
  s^4 - 18 * s^2 - 8 * s = -37 :=
by {
  sorry -- The proof would go here
}

end problem_statement_l526_526102


namespace time_for_investment_to_quadruple_l526_526938

-- Define the conditions in the problem.
def interest_rate : ℝ := 8
def initial_investment : ℝ := 5000
def final_investment : ℝ := 20000
def doubling_time (r : ℝ) : ℝ := 70 / r

-- Define the assertion we need to prove.
theorem time_for_investment_to_quadruple :
  let t := 2 * (doubling_time interest_rate) in
  t = 17.5 :=
by
  -- Calculate doubling time based on the interest rate.
  let double_t := doubling_time interest_rate
  -- Doubling time should be 70 / 8
  have h1 : double_t = 70 / 8 := rfl
  -- twice the doubling time should be 2 * (70 / 8)
  let t := 2 * double_t
  -- t should be 2 * (70 / 8) == 17.5
  have h2 : t = 2 * (70 / 8) := rfl
  -- Therefore, by arithmetic t should be 17.5
  have h3 : t = 17.5 := by norm_num
  exact h3

end time_for_investment_to_quadruple_l526_526938


namespace necessary_not_sufficient_geom_sum_increasing_l526_526075

-- Definition for the sum of a geometric series
def geom_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

-- The theorem stating the condition to be necessary but not sufficient
theorem necessary_not_sufficient_geom_sum_increasing (a₁ : ℝ) (q : ℝ) :
  (∀ n : ℕ, geom_sum a₁ q (n + 1) > geom_sum a₁ q n) ↔ (a₁ > 0 ∧ q > 0 ∧ q ≠ 1) :=
by sorry

end necessary_not_sufficient_geom_sum_increasing_l526_526075


namespace general_formula_for_geometric_sequence_sum_of_first_n_terms_b_sequence_l526_526470

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∃ q, (∀ n : ℕ, a (n + 1) = a 1 * q ^ n) ∧ 2 * a 2 = a 1 + (a 3 - 1)

noncomputable def arithmetic_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = 2 * n + a n

noncomputable def sum_of_first_n_terms (b : ℕ → ℝ) : ℕ → ℝ
| 0     := 0
| (n+1) := b (n + 1) + sum_of_first_n_terms b n

theorem general_formula_for_geometric_sequence :
  ∀ (a : ℕ → ℝ), geometric_sequence a → ∀ n, a n = 2^(n - 1) :=
by
  intros a ha n
  sorry

theorem sum_of_first_n_terms_b_sequence :
  ∀ (a b : ℕ → ℝ), geometric_sequence a → arithmetic_sequence b a →
  ∀ n, sum_of_first_n_terms b n = n^2 + n + 2^n - 1 :=
by
  intros a b ha hb n
  sorry

end general_formula_for_geometric_sequence_sum_of_first_n_terms_b_sequence_l526_526470


namespace derivative_at_minus_one_l526_526905

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

theorem derivative_at_minus_one :
  (deriv f 1 = 2) → (deriv f (-1) = -2) :=
begin
  intros h,
  sorry
end

end derivative_at_minus_one_l526_526905


namespace shaded_area_of_octagon_l526_526663

noncomputable def areaOfShadedRegion (s : ℝ) (r : ℝ) (theta : ℝ) : ℝ :=
  let n := 8
  let octagonArea := n * 0.5 * s^2 * (Real.sin (Real.pi/n) / Real.sin (Real.pi/(2 * n)))
  let sectorArea := n * 0.5 * r^2 * (theta / (2 * Real.pi))
  octagonArea - sectorArea

theorem shaded_area_of_octagon (h_s : 5 = 5) (h_r : 3 = 3) (h_theta : 45 = 45) :
  areaOfShadedRegion 5 3 (45 * (Real.pi / 180)) = 100 - 9 * Real.pi := by
  sorry

end shaded_area_of_octagon_l526_526663


namespace sum_of_all_possible_k_values_l526_526443

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l526_526443


namespace sum_of_possible_ks_l526_526432

theorem sum_of_possible_ks : 
  (∃ (j k : ℕ), (1 < j) ∧ (1 < k) ∧ j ≠ k ∧ ((1/j : ℝ) + (1/k : ℝ) = (1/4))) → 
  (∑ k in {20, 12, 8, 6, 5}, k) = 51 :=
begin
  sorry
end

end sum_of_possible_ks_l526_526432


namespace find_f_at_2_l526_526895

-- Given conditions
def f (a b x : ℝ) : ℝ := (a * x + b) / (x^2 + 1)
def is_odd_fun (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

-- Prove the target statement
theorem find_f_at_2 (a b : ℝ) (Hodd : is_odd_fun (f a b)) (H_val : f a b (1/2) = 2/5) : f a b 2 = 2/5 :=
by
  sorry

end find_f_at_2_l526_526895


namespace number_of_solutions_l526_526032

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ :=
if x ≤ 0 then x^2 + b*x + c else 2

theorem number_of_solutions (b c : ℝ) 
  (h1 : f (-4) b c = f 0 b c)
  (h2 : f (-2) b c = -2) :
  set.finite {x : ℝ | f x b c = x} ∧ finset.card (to_finset {x : ℝ | f x b c = x}) = 3 :=
sorry

end number_of_solutions_l526_526032


namespace value_of_a10_minus_one_fifth_a12_l526_526423

noncomputable def a (n : ℕ) : ℚ := sorry -- Definition of a_n in the arithmetic sequence

def d : ℚ := sorry -- Common difference in the arithmetic sequence

axiom arithmetic_sequence_condition (a1 : ℚ) (h : 4 * a 1 + 34 * d = 20) : 
  a 4 + a 6 + a 13 + a 15 = 20

theorem value_of_a10_minus_one_fifth_a12 (a1 : ℚ) (h_condition : arithmetic_sequence_condition a1 (4 * a1 + 34 * d = 20)) :
  a 10 - (1 / 5) * a 12 = 4 :=
by
  sorry

end value_of_a10_minus_one_fifth_a12_l526_526423


namespace number_of_values_xn_sum_of_values_xn_l526_526484

variables (n : ℕ) (x0 a b : ℝ)

-- Condition: b > 0, x0 ≠ 0, recurrence relation.
axiom b_pos : b > 0
axiom x0_ne_zero : x0 ≠ 0
axiom recurrence_relation (n : ℕ) : ∀ n, x_(n + 1)^2 = a * x_n * x_(n + 1) + b * x_n^2

-- The statement to be proved: number of different values x_n can take is n + 1.
theorem number_of_values_xn : ∀ n, ∃ f : ℕ → ℝ, ∀ k ≤ n, f k = x0 ∧ (∃ s t : ℝ, s = (a + sqrt (a^2 + 4 * b)) / 2 ∧ t = (a - sqrt (a^2 + 4 * b)) / 2 ∧ ∃ k ∈ (Icc 0 n), x_n = ((s ^ k) * (t ^ (n - k)) * x0)) :=
sorry

-- The statement to be proved: sum of all possible values of x_n.
theorem sum_of_values_xn : ∀ n, ∑ k in finset.range (n + 1), x0 * (((a + sqrt (a^2 + 4 * b)) / 2)^k * ((a - sqrt (a^2 + 4 * b)) / 2)^(n - k)) = x0 * ((a + sqrt (a^2 + 4 * b))^(n + 1) - (a - sqrt (a^2 + 4 * b))^(n + 1)) / (2 ^ (n + 2) * sqrt (a^2 + 4 * b)) :=
sorry

end number_of_values_xn_sum_of_values_xn_l526_526484


namespace find_arithmetic_progression_areas_l526_526976

noncomputable def triangle_ABC : ℝ × ℝ × ℝ := (5, 8, 7)

def area_arithmetic_progression (AB BC AC : ℝ) (AOB_area AOC_area BOC_area : ℝ) : Prop :=
  (AOC_area = 10 * Real.sqrt 3 / 3) ∧ 
  (AOB_area = 5 * (AOC_area) / 7) ∧ 
  (BOC_area = 10 * Real.sqrt 3 - AOB_area - AOC_area) ∧ 
  (AOB_area, AOC_area, BOC_area).antisymm = [5 * Real.sqrt 3 / 21, 10 * Real.sqrt 3 / 3, 15 * Real.sqrt 3 / 21]

theorem find_arithmetic_progression_areas :
  ∃ (AOB_area AOC_area BOC_area : ℝ),
    area_arithmetic_progression 5 8 7 AOB_area AOC_area BOC_area :=
begin
  -- Here we can outline the proof steps if necessary
  sorry
end

end find_arithmetic_progression_areas_l526_526976


namespace yellow_dandelions_day_before_yesterday_l526_526249

-- Define the problem in terms of conditions and conclusion
theorem yellow_dandelions_day_before_yesterday
  (yellow_yesterday : ℕ) (white_yesterday : ℕ)
  (yellow_today : ℕ) (white_today : ℕ) :
  yellow_yesterday = 20 → white_yesterday = 14 →
  yellow_today = 15 → white_today = 11 →
  (let yellow_day_before_yesterday := white_yesterday + white_today
  in yellow_day_before_yesterday = 25) :=
by
  intros h1 h2 h3 h4
  let yellow_day_before_yesterday := white_yesterday + white_today
  show yellow_day_before_yesterday = 25
  exact (by sorry)

end yellow_dandelions_day_before_yesterday_l526_526249


namespace abs_diff_odd_even_S_2500_l526_526793

def tau (n : ℕ) : ℕ := (finset.range (n + 1)).filter (λ d, n % d = 0).card

def S (n : ℕ) : ℕ := (finset.range (n + 1)).sum tau 

def count_odd_S (n : ℕ) : ℕ := (finset.range (n + 1)).count 
  (λ k, S k % 2 = 1)

def count_even_S (n : ℕ) : ℕ := (finset.range (n + 1)).count 
  (λ k, S k % 2 = 0)

def abs_diff_odd_even_S (n : ℕ) : ℕ := 
  (count_odd_S n - count_even_S n).nat_abs

theorem abs_diff_odd_even_S_2500 : abs_diff_odd_even_S 2500 = (2 * 24 + 99) :=
sorry

end abs_diff_odd_even_S_2500_l526_526793


namespace TriangleAngleBisectors_l526_526684

theorem TriangleAngleBisectors
  (ABC : Type*) [triangle ABC]
  (B D E C : ABC)
  (angle_BDE : angle ABC D E = 24 * (π / 180))
  (angle_CED : angle ABC E D = 18 * (π / 180))
  (bisector_BD : is_angle_bisector (angle ABC B D) (angle ABC D E))
  (bisector_CE : is_angle_bisector (angle ABC C E) (angle ABC E D)) :
  ∃ (A B C : ℝ),
    A = 96 ∧ B = 12 ∧ C = 72 ∧ 
    A + B + C = 180 := 
sorry

end TriangleAngleBisectors_l526_526684


namespace red_triangles_linked_with_yellow_cannot_be_2023_l526_526958

theorem red_triangles_linked_with_yellow_cannot_be_2023
  (points : Fin 43 → Prop)
  (yellow : Finset (Fin 43))
  (red : Finset (Fin 43))
  (h1 : yellow.card = 3)
  (h2 : red.card = 40)
  (h3 : ∀ (s : Finset (Fin 43)), s.card = 4 → ¬ ∃ (P : Prop), affine_span ℝ (coe s) = P) :
  ¬ ∃ (t : Finset (Fin 40)), (t.card = 2023 ∧ (∃ (ty : Finset (Fin 3)), is_linked t ty)) :=
by
  sorry

end red_triangles_linked_with_yellow_cannot_be_2023_l526_526958


namespace count_incorrect_statements_l526_526204

-- Definitions of each statement to be evaluated
def statement1 : Prop := "The height of a triangle is either inside or outside the triangle."
def statement2 : Prop := "The sum of the interior angles of a polygon must be less than the sum of its exterior angles."
def statement3 : Prop := "Two triangles with equal perimeter and area are necessarily congruent."
def statement4 : Prop := "Two equilateral triangles with equal perimeter are congruent."
def statement5 : Prop := "Two triangles with two sides and an angle respectively congruent are congruent."
def statement6 : Prop := "The line that bisects the exterior angle of the vertex of an isosceles triangle is parallel to the base of the isosceles triangle."

-- Definitions of incorrect and correct status for statements
def is_incorrect (s : Prop) : Prop := 
  s = statement1 ∨ 
  s = statement2 ∨ 
  s = statement3 ∨ 
  s = statement5

def is_correct (s : Prop) : Prop := 
  s = statement4 ∨ 
  s = statement6

-- Lean statement to prove the number of incorrect statements equals 4
theorem count_incorrect_statements : 
    (∃ n, n = 4 ∧ 
      ((statement1 ∨ ¬statement1) ∧ 
      (statement2 ∨ ¬statement2) ∧ 
      (statement3 ∨ ¬statement3) ∧ 
      (statement4 ∨ ¬statement4) ∧ 
      (statement5 ∨ ¬statement5) ∧ 
      (statement6 ∨ ¬statement6)) ∧
      (is_incorrect statement1) ∧
      (is_incorrect statement2) ∧
      (is_incorrect statement3) ∧
      (is_incorrect statement5) ∧
      (is_correct statement4) ∧
      (is_correct statement6))
    :=
begin
  sorry  -- Placeholder to indicate this requires a proof
end

end count_incorrect_statements_l526_526204


namespace tan_half_prod_eq_sqrt3_l526_526927

theorem tan_half_prod_eq_sqrt3 (a b : ℝ) (h : 7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ (xy : ℝ), xy = Real.tan (a / 2) * Real.tan (b / 2) ∧ (xy = Real.sqrt 3 ∨ xy = -Real.sqrt 3) :=
by
  sorry

end tan_half_prod_eq_sqrt3_l526_526927


namespace sqrt_domain_l526_526060

   theorem sqrt_domain (x : ℝ) : (∃ y : ℝ, y = sqrt (5 - x)) ↔ x ≤ 5 := 
   by
     sorry
   
end sqrt_domain_l526_526060


namespace compute_expression_l526_526998

theorem compute_expression (p q r : ℝ) 
  (h1 : p + q + r = 6) 
  (h2 : pq + qr + rp = 11) 
  (h3 : pqr = 12) : 
  (pq / r) + (qr / p) + (rp / q) = -23 / 12 := 
sorry

end compute_expression_l526_526998


namespace even_square_is_even_l526_526006

theorem even_square_is_even (a : ℤ) (h : Even (a^2)) : Even a :=
sorry

end even_square_is_even_l526_526006


namespace compare_constants_l526_526843

theorem compare_constants :
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  in c < a ∧ a < b :=
by
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  sorry

end compare_constants_l526_526843


namespace exists_element_x_l526_526989

open Set

theorem exists_element_x (n : ℕ) (S : Finset (Fin n)) (A : Fin n → Finset (Fin n)) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → A i ≠ A j) : 
  ∃ x ∈ S, ∀ i j : Fin n, i ≠ j → (A i \ {x}) ≠ (A j \ {x}) :=
sorry

end exists_element_x_l526_526989


namespace repeatingDecimal_exceeds_l526_526696

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l526_526696


namespace repeating_decimal_difference_l526_526705

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l526_526705


namespace option_b_not_zero_option_a_is_zero_option_c_is_zero_option_d_is_zero_l526_526268

variables {V : Type*} [AddCommGroup V] [Vector V]

-- Define vectors
variables (A B C D O N Q P M : V) 

--1  The expression sums
def OptionA := A + B + C
def OptionB := O + C + B + O
def OptionC := A - B + D - C
def OptionD := N + Q + P - M

--2  Prove that OptionB is not necessarily 0
theorem option_b_not_zero (v₁ v₂ v₃ v₄ : V) : ¬ (v₁ + v₂ + v₃ + v₄ = 0) :=
begin
  sorry
end

--3  Prove that the other expressions add up to zero
theorem option_a_is_zero (v₁ v₂ v₃ : V) : v₁ + v₂ + v₃ = 0 :=
begin
  sorry
end

theorem option_c_is_zero (v₁ v₂ v₃ v₄ : V) : v₁ - v₂ + v₃ - v₄ = 0 :=
begin
  sorry
end

theorem option_d_is_zero (v₁ v₂ v₃ v₄ : V) : v₁ + v₂ + v₃ - v₄ = 0 :=
begin
  sorry
end

end option_b_not_zero_option_a_is_zero_option_c_is_zero_option_d_is_zero_l526_526268


namespace repeating_seventy_two_exceeds_seventy_two_l526_526737

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l526_526737


namespace transformed_volume_l526_526915

noncomputable def volume_parallelepiped (a b c : ℝ^3) : ℝ := 
|a • (b × c)|

theorem transformed_volume {a b c : ℝ^3} 
(h : volume_parallelepiped a b c = 6) :
volume_parallelepiped (a - 2 • b) (2 • b + 4 • c) (c + 3 • a) = 132 := 
sorry

end transformed_volume_l526_526915


namespace magnitude_of_z_l526_526360

def imaginary_unit_i (i : ℂ) : Prop :=
  i = Complex.I

def complex_z (i : ℂ) : ℂ :=
  (1 + 2 * i) / i

theorem magnitude_of_z (i : ℂ) (hi : imaginary_unit_i i) :
  Complex.abs (complex_z i) = Real.sqrt 5 :=
by
  sorry

end magnitude_of_z_l526_526360


namespace no_simple_solution_binomial_probability_l526_526272

theorem no_simple_solution_binomial_probability 
  (p : ℝ) 
  (w : ℝ)
  (binom : ∀ n k, real(binomial n k) = real.combination n k)
  (n : ℕ := 6) 
  (k : ℕ := 4) 
  (w_val : w = 27 / 256):
  ¬ ∃ p, (15 * p^4 * (1 - p)^2 = w) :=
by
  sorry

end no_simple_solution_binomial_probability_l526_526272


namespace vision_data_approximation_l526_526542

theorem vision_data_approximation :
  ∀ (L V : ℝ), L = 4.9 → (L = 5 + log10 V) → (0.8 ≤ V ∧ V ≤ 0.8 * 1.0001) := 
by
  intros L V hL hRel
  have hlog : log10 V = -0.1 := 
    by rw [←hRel, hL]; ring
  have hV : V = 10 ^ (-0.1) :=
    by rw [←hlog]; exact (real.rpow10_log10 V)
  rw hV
  have hApprox : 10 ^ (-0.1) ≈ 0.8 := sorry -- This is the approximation step
  exact hApprox

end vision_data_approximation_l526_526542


namespace num_female_students_l526_526151

theorem num_female_students (F : ℕ) (h1: 8 * 85 + F * 92 = (8 + F) * 90) : F = 20 := 
by
  sorry

end num_female_students_l526_526151


namespace area_of_rectangle_excluding_hole_l526_526660

theorem area_of_rectangle_excluding_hole (x y : ℝ) 
  (h1 : (y + 1) = (x - 2)) 
  (h2 : (2y + 3) = (2 * (x - 3) + 3)) :
  let area_large := (2 * x + 14) * (2 * x + 10),
      area_hole := y * (2 * y + 3)
  in area_large - area_hole = 2 * x^2 + 57 * x + 131 :=
by
  sorry

end area_of_rectangle_excluding_hole_l526_526660


namespace quadratic_has_real_root_l526_526931

theorem quadratic_has_real_root {b : ℝ} :
  ∃ x : ℝ, x^2 + b*x + 25 = 0 ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_has_real_root_l526_526931


namespace part1_l526_526974

variable (a b c : ℝ) (A B : ℝ)
variable (triangle_abc : Triangle ABC)
variable (cos : ℝ → ℝ)

axiom law_of_cosines : ∀ {a b c A : ℝ}, a^2 = b^2 + c^2 - 2 * b * c * cos A

theorem part1 (h1 : b^2 + 3 * a * c * (a^2 + c^2 - b^2) / (2 * a * c) = 2 * c^2) (h2 : a = c) : A = π / 4 := 
sorry

end part1_l526_526974


namespace compare_constants_l526_526837

theorem compare_constants :
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  in c < a ∧ a < b :=
by
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  sorry

end compare_constants_l526_526837


namespace max_area_ABC_l526_526356

-- Definitions of the given conditions
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 2)
def C : {p : ℝ × ℝ // p.1^2 + p.2^2 - 2 * p.1 = 0}

-- The theorem statement
theorem max_area_ABC : 
  ∃ C : {p : ℝ × ℝ // p.1^2 + p.2^2 - 2 * p.1 = 0}, 
  (area_triangle A B C) = 3 + real.sqrt 2 :=
sorry

end max_area_ABC_l526_526356


namespace fraction_difference_is_correct_l526_526721

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l526_526721


namespace masha_comb_teeth_count_l526_526097

theorem masha_comb_teeth_count (katya_teeth : ℕ) (masha_to_katya_ratio : ℕ) 
  (katya_teeth_eq : katya_teeth = 11) 
  (masha_to_katya_ratio_eq : masha_to_katya_ratio = 5) : 
  ∃ masha_teeth : ℕ, masha_teeth = 53 :=
by
  have katya_segments := 2 * katya_teeth - 1
  have masha_segments := masha_to_katya_ratio * katya_segments
  let masha_teeth := (masha_segments + 1) / 2
  use masha_teeth
  have masha_teeth_eq := (2 * masha_teeth - 1 = 105)
  sorry

end masha_comb_teeth_count_l526_526097


namespace sum_of_possible_ks_l526_526450

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l526_526450


namespace rationalize_denominator_l526_526519

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ), B < D ∧
  (∀ n : ℤ, Prime n → ¬ (∃ m : ℤ, m^2 = B)) ∧
  (∀ n : ℤ, Prime n → ¬ (∃ m : ℤ, m^2 = D)) ∧
  A + B + C + D + E = 94 ∧
  (∀ (a b : ℤ), a + b ≠ 0 → Int.gcd a b = 1) →
  (6 * sqrt 7 - 9 * sqrt 13) / 89 = A * sqrt B + C * sqrt D / E :=
sorry

end rationalize_denominator_l526_526519


namespace smallest_integer_divisible_l526_526170

theorem smallest_integer_divisible:
  ∃ n : ℕ, n > 1 ∧ (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 6 = 1) ∧ n = 61 :=
by
  sorry

end smallest_integer_divisible_l526_526170


namespace compare_a_b_c_l526_526849

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l526_526849


namespace total_volume_of_cubes_l526_526981

theorem total_volume_of_cubes (Jim_cubes : Nat) (Jim_side_length : Nat) 
    (Laura_cubes : Nat) (Laura_side_length : Nat)
    (h1 : Jim_cubes = 7) (h2 : Jim_side_length = 3) 
    (h3 : Laura_cubes = 4) (h4 : Laura_side_length = 4) : 
    (Jim_cubes * Jim_side_length^3 + Laura_cubes * Laura_side_length^3 = 445) :=
by
  sorry

end total_volume_of_cubes_l526_526981


namespace find_ratio_l526_526091

def is_right_triangle (A B C : Type) [MetricSpace A] (a b c : A) :=
  ∃ x : ℝ, ∃ y : ℝ, x ^ 2 + y ^ 2 = (dist b c) ^ 2 ∧ y = dist c a

def triangle_with_conditions (A B C : Type) [MetricSpace A] (a b c q : A) :=
  is_right_triangle A B C a b c ∧ 
  dist a b = 5 ∧ 
  angle a q c = 3 * angle a c q ∧ 
  dist c q = 2

def ratio_AQ_BQ (A B C : Type) [MetricSpace A] [InnerProductSpace ℝ A] [NormedAddTorsor A B] (a b c q : A) : ℝ :=
  let aq := dist a q
  let bq := dist q b
  aq / bq

theorem find_ratio (A B C : Type) [MetricSpace A] [InnerProductSpace ℝ A] [NormedAddTorsor A B]
  (a b c q : A) : triangle_with_conditions A B C a b c q → ratio_AQ_BQ A B C a b c q = 7 / 3 :=
by
  sorry

end find_ratio_l526_526091


namespace factorization_of_square_difference_l526_526768

variable (t : ℝ)

theorem factorization_of_square_difference : t^2 - 144 = (t - 12) * (t + 12) := 
sorry

end factorization_of_square_difference_l526_526768


namespace compare_values_l526_526805

variable (a b c : ℝ)

noncomputable def a_def : ℝ := 0.1 * Real.exp 0.1
noncomputable def b_def : ℝ := 1 / 9
noncomputable def c_def : ℝ := - Real.log 0.9

theorem compare_values : 
  c < a ∧ a < b := 
begin
  let a := 0.1 * Real.exp 0.1,
  let b := 1 / 9,
  let c := - Real.log 0.9,
  have h1 : c < a, sorry,
  have h2 : a < b, sorry,
  exact ⟨h1, h2⟩
end

end compare_values_l526_526805


namespace ellipse_correctness_line_through_fixed_point_l526_526301

noncomputable def ellipse_equation (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_correctness (a b : ℝ) (h : a > b ∧ b > 0) :
  (∀ P : ℝ × ℝ,
    let PF₁ := -sqrt (a^2 - b^2) - P.1, PF₂ := sqrt (a^2 - b^2) - P.1, PSq :=
    (PF₁ * PF₂) + (P.2)^2 in
    max_value (: ℝ)[PSq] = 3 ∧ min_value (: ℝ)[PSq] = 2) →
  (let (xaa, ybb) := (4, 3) in (xaa = 4 ∧ ybb = 3)) :=
begin
  sorry
end

theorem line_through_fixed_point (a b k m : ℝ) :
  let A := (a, 0) in
  if ∀(x : ℝ),
    let lhs := ((x^2 / 4) + (((k * x) + m)^2 / 3) + 1) = 1,
    nextls := (A = ((x - sqrt(3)) * (x + sqrt(3))) + ((k * x + m)^2) = 0),
    midsum := let pa := (-2*k),
              fix := A in pa let := 7m + 2 k at pa := 0
end :
  fixed_point p := (2/7, 0) :=
begin
  sorry
end

end ellipse_correctness_line_through_fixed_point_l526_526301


namespace find_polynomials_g_l526_526533

-- Define functions f and proof target is g
def f (x : ℝ) : ℝ := x ^ 2

-- g is defined as an unknown polynomial with some constraints
variable (g : ℝ → ℝ)

-- The proof problem stating that if f(g(x)) = 9x^2 + 12x + 4, 
-- then g(x) = 3x + 2 or g(x) = -3x - 2
theorem find_polynomials_g (h : ∀ x : ℝ, f (g x) = 9 * x ^ 2 + 12 * x + 4) :
  (∀ x : ℝ, g x = 3 * x + 2) ∨ (∀ x : ℝ, g x = -3 * x - 2) := 
by
  sorry

end find_polynomials_g_l526_526533


namespace orthocenter_of_triangle_l526_526960

theorem orthocenter_of_triangle :
  ∀ (A B C H : ℝ × ℝ × ℝ),
    A = (2, 3, 4) → 
    B = (6, 4, 2) → 
    C = (4, 5, 6) → 
    H = (17/53, 152/53, 725/53) → 
    true :=
by sorry

end orthocenter_of_triangle_l526_526960


namespace repeating_seventy_two_exceeds_seventy_two_l526_526742

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l526_526742


namespace excess_common_fraction_l526_526714

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l526_526714


namespace minimum_airline_companies_l526_526069

-- Define the number of cities
def num_cities : ℕ := 2019

-- Define the condition of flights between any pair of cities
def flight_exists (A B : ℕ) (hA : A < num_cities) (hB : B < num_cities) : Prop :=
  A ≠ B ∧ ∃ flight, A < B ∨ B < A

-- Define the condition for different companies for flights among any three distinct cities
def distinct_companies (A B C : ℕ) (hA : A < num_cities) (hB : B < num_cities) (hC : C < num_cities) : Prop :=
  ∀ (f1 f2 f3 : ℕ), flight_exists A B hA hB → flight_exists B C hB hC → flight_exists C A hC hA → f1 ≠ f2 ∧ f2 ≠ f3 ∧ f1 ≠ f3

-- The minimum number of airline companies required
theorem minimum_airline_companies : ∃ (n : ℕ), n = 2019 ∧
  (∀ A B C : ℕ, A < num_cities → B < num_cities → C < num_cities → distinct_companies A B C sorry sorry sorry) :=
begin
  existsi 2019,
  split,
  { refl, },
  { intros A B C hA hB hC,
    sorry, }
end

end minimum_airline_companies_l526_526069


namespace selecting_8_points_l526_526758

theorem selecting_8_points (n : ℕ) (points : Fin n → Nat) (selected_points : Fin 8 → Fin n)
  (h₁ : n = 24)
  (h₂ : ∀ i j : Fin 8, i ≠ j → (points (selected_points i) - points (selected_points j) ≠ 3) ∧ 
                        (points (selected_points i) - points (selected_points j) ≠ 8)) : 
  ∃ s : Finset (Fin 24), s.card = 8 ∧ 
    ∀ (i j : Fin 8), i ≠ j → (points (s i) - points (s j) ≠ 3) ∧ 
                        (points (s i) - points (s j) ≠ 8) ∧ 
    Finset.card (filter (λ sₛ, sₛ.card = 8 ∧ 
                              ∀ (i j : Fin 8), i ≠ j → (points (sₛ i) - points (sₛ j) ≠ 3) ∧ 
                                              (points (sₛ i) - points (sₛ j) ≠ 8)) 
                               (powerset (Finset.univ : Finset (Fin 24)))) = 258 := 
  sorry

end selecting_8_points_l526_526758


namespace compare_two_sqrt_three_l526_526751

theorem compare_two_sqrt_three : 2 > Real.sqrt 3 :=
by {
  sorry
}

end compare_two_sqrt_three_l526_526751


namespace proof_problem_l526_526834

-- Define the constants a, b, c
def a := 0.1 * Real.exp 0.1
def b := 1 / 9
def c := -Real.log 0.9

-- State the theorem to prove
theorem proof_problem : c < a ∧ a < b := 
by
  sorry

end proof_problem_l526_526834


namespace line_circle_no_intersection_l526_526755

theorem line_circle_no_intersection :
  (∀ (x y : ℝ), 3 * x + 4 * y = 12 ∨ x^2 + y^2 = 4) →
  (∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4) →
  false :=
by
  sorry

end line_circle_no_intersection_l526_526755


namespace like_terms_equal_exponents_l526_526047

theorem like_terms_equal_exponents (x y : ℤ) : 3 * a^2 * b^(y-1) = 6 * a^(x+3) * b^3 → x^y = 1 := by
  intro h
  -- Since 3a^2b^(y-1) = 6a^(x+3)b^3, every term and exponent needs to be equal
  have hx : x + 3 = 2 := by sorry
  have hy : y - 1 = 3 := by sorry
  -- Solve for x and y from the equations
  have heqx : x = -1 := by sorry
  have heqy : y = 4 := by sorry
  -- Calculate x^y
  show x^y = 1
  rw [heqx, heqy]
  exact (-1)^4 = 1

end like_terms_equal_exponents_l526_526047


namespace fraction_difference_is_correct_l526_526723

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l526_526723


namespace inverse_function_correct_l526_526560

theorem inverse_function_correct :
  ( ∀ x : ℝ, (x > 1) → (∃ y : ℝ, y = 1 + Real.log (x - 1)) ↔ (∀ y : ℝ, y > 0 → (∃ x : ℝ, x = e^(y + 1) - 1))) :=
by
  sorry

end inverse_function_correct_l526_526560


namespace original_wire_length_l526_526253

theorem original_wire_length (side_square : ℝ) (diameter_circle : ℝ) (total_area_squares : ℝ) (total_area_circles : ℝ) 
    (h1 : side_square = 2.5) 
    (h2 : diameter_circle = 3) 
    (h3 : total_area_squares = 87.5) 
    (h4 : total_area_circles = 56.52) 
    : 
    total_wire_length (side_square : ℝ) (num_squares : ℝ) (num_circles : ℝ) (circumference_circle : ℝ) (total_wire_squares : ℝ) (total_wire_circles : ℝ)
    (n_squares : num_squares * (side_square ^ 2) = total_area_squares)
    (n_circles : num_circles * (( real.pi * (( diameter_circle / 2 ) ^ 2) ) = total_area_circles))
    (length_squares : total_wire_squares = 4 * side_square * num_squares)
    (length_circles : total_wire_circles = ((real.pi * diameter_circle) * num_circles))
    (original_length_wire : total_wire_squares + total_wire_circles = 215.39824):
  original_length_wire = total_wire_squares + total_wire_circles :=
  sorry

end original_wire_length_l526_526253


namespace check_propositions_count_l526_526882

-- Definitions based on given conditions
def proposition1 (A B C D : Type) (vec : A → B → C) : Prop :=
    vec A B + vec B C + vec C D + vec D A = 0

def collinear (a b : Type) : Prop :=
    ∃ (λ : ℝ), a = λ * b

def proposition2 (b : Type) (vec : Type) : Prop :=
    b ≠ vec → collinear a b

def parallel_or_coincident (a b : Type) : Prop :=  -- appropriately fix your required types
    collinear a b → (lines containing a) ∥ (lines containing b)

def coplanar (O A B C P : Type) (vec : O → A → B → C → P) 
(x y z : ℝ) : Prop := (x + y + z = 1) → P, A, B, C are coplanar

-- Main goal to check the total number of true propositions
def num_true_propositions (A B C D O P : Type) (vec : A → B → C) : Type :=
    (proposition1 A B C D vec + proposition2 b vec + 
     (¬ parallel_or_coincident a b) + coplanar O A B C P vec)

theorem check_propositions_count : 
  A B C D O P : Type,
    (num_true_propositions A B C D O P) = 3 := by sorry

end check_propositions_count_l526_526882


namespace sum_possible_k_l526_526458

theorem sum_possible_k (j k : ℕ) (hcond : 1 / j + 1 / k = 1 / 4) (hj : 0 < j) (hk : 0 < k) :
  j ≠ k → ∑ k in {k | ∃ j, 1 / j + 1 / k = 1 / 4}, k = 51 :=
by
  sorry

end sum_possible_k_l526_526458


namespace problem1_calculation_problem2_eq1_solution_problem2_eq2_solution_l526_526636

-- Problem 1 Equivalent Proof
theorem problem1_calculation : 
  sqrt 9 + abs (-2) - (-3)^2 + (pi - 100)^0 = -3 :=
by sorry

-- Problem 2 Proofs
-- Equation 1
theorem problem2_eq1_solution (x : ℝ) (h : x^2 + 1 = 5) : x = 2 ∨ x = -2 :=
by sorry

-- Equation 2
theorem problem2_eq2_solution (x : ℝ) (h : x^2 = (x - 2)^2 + 7) : x = 11 / 4 :=
by sorry

end problem1_calculation_problem2_eq1_solution_problem2_eq2_solution_l526_526636


namespace stones_required_to_pave_hall_l526_526658

def length_hall_m : ℝ := 36
def breadth_hall_m : ℝ := 15

def length_stone_dm : ℝ := 2
def breadth_stone_dm : ℝ := 5

def length_hall_dm : ℝ := length_hall_m * 10
def breadth_hall_dm : ℝ := breadth_hall_m * 10

def area_hall_dm2 : ℝ := length_hall_dm * breadth_hall_dm
def area_stone_dm2 : ℝ := length_stone_dm * breadth_stone_dm

def number_of_stones : ℝ := area_hall_dm2 / area_stone_dm2

theorem stones_required_to_pave_hall : number_of_stones = 5400 := by
  sorry

end stones_required_to_pave_hall_l526_526658


namespace total_zeros_of_odd_function_l526_526359

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder definition for f

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x  -- Definition of odd function

def has_zeros_in_interval (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) :=
  ∃ (s : finset ℝ), (∀ x ∈ s, f x = 0) ∧ (∀ x ∈ s, a < x ∧ x < b) ∧ s.card = n

theorem total_zeros_of_odd_function (h1 : is_odd f)
  (h2 : ∀ x, f x = f x)
  (h3 : has_zeros_in_interval f (-∞) 0 2012) : finset.card {x : ℝ | f x = 0}.to_finset = 4025 :=
sorry

end total_zeros_of_odd_function_l526_526359


namespace vision_approximation_l526_526537

noncomputable
def five_point_to_decimal (L: ℝ) : ℝ := 10 ^ (L - 5)

theorem vision_approximation : 
  ∀ L : ℝ, L = 4.9 → (five_point_to_decimal L) ≈ 0.8 :=
by
  intros L hL
  rw hL
  sorry

end vision_approximation_l526_526537


namespace f_half_l526_526928

def g (x : ℝ) : ℝ := 1 - x^2

noncomputable def f (y : ℝ) : ℝ := if h : y ≠ 0 then (1 - (sqrt (1 - y))) / (sqrt (1 - y)) else 0

theorem f_half : f (1/2) = 1 := by
  have h : (sqrt (1 / 2)) ≠ 0 := by linarith
  simp [f, h]
  sorry

end f_half_l526_526928


namespace general_term_indeterminate_l526_526499

-- Definition of function f and the sequence {a_n}
def f (x : ℝ) (coeffs : List ℝ) : ℝ := 
coeffs.enum.map (λ ⟨i, a⟩, a * x^i).sum

-- Using condition 
def f_zero (f : ℝ → ℝ) : ℝ :=
f 0

def f_one (f : ℝ → ℝ) : ℝ :=
f 1

def seq_cond (coeffs : List ℝ) (n : ℕ) : Prop :=
f_one (f coeffs) = n^2 * coeffs.getOrElse (n-1) 0

-- Prove the general term a_n = indeterminate
theorem general_term_indeterminate (coeffs : List ℝ) (n : ℕ) (h : f_zero (f coeffs) = coeffs.head 0)
(h' : seq_cond coeffs n) : coeffs.getOrElse (n-1) 0 = sorry := 
sorry

end general_term_indeterminate_l526_526499


namespace inequality_proof_l526_526859

-- Define constants
def a : ℝ := 0.1 * Real.exp 0.1
def b : ℝ := 1 / 9
def c : ℝ := -Real.log 0.9

-- Assert the inequalities
theorem inequality_proof : c < a ∧ a < b :=
by sorry

end inequality_proof_l526_526859


namespace coloring_scheme_sufficient_l526_526481

def point := (ℕ × ℕ × ℕ)

def V : set point := {p | ∃ x y z, p = (x, y, z) ∧ 0 ≤ x ∧ x ≤ 2008 ∧ 0 ≤ y ∧ y ≤ 2008 ∧ 0 ≤ z ∧ z ≤ 2008}

def f (p : point) : ℕ :=
  p.1 + 2 * p.2 + 3 * p.3

def distance (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2

theorem coloring_scheme_sufficient :
  ∃ (coloring : point → fin 7),
    (∀ p1 p2 ∈ V, p1 ≠ p2 →
      (distance p1 p2 = 1 ∨ distance p1 p2 = 2 ∨ distance p1 p2 = 4) →
        coloring p1 ≠ coloring p2) :=
sorry

end coloring_scheme_sufficient_l526_526481


namespace vision_approximation_l526_526539

noncomputable
def five_point_to_decimal (L: ℝ) : ℝ := 10 ^ (L - 5)

theorem vision_approximation : 
  ∀ L : ℝ, L = 4.9 → (five_point_to_decimal L) ≈ 0.8 :=
by
  intros L hL
  rw hL
  sorry

end vision_approximation_l526_526539


namespace num_two_digit_integers_no_repetition_l526_526026

theorem num_two_digit_integers_no_repetition : 
  let digits := {2, 4, 7, 9} in
  let two_digit_numbers := {d1 * 10 + d2 | d1 d2 : ℕ, d1 ≠ d2, d1 ∈ digits, d2 ∈ digits} in
  two_digit_numbers.card = 12 :=
by
  sorry

end num_two_digit_integers_no_repetition_l526_526026


namespace compare_constants_l526_526863

noncomputable def a : ℝ := 0.1 * real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - real.log 0.9

theorem compare_constants : c < a ∧ a < b :=
by sorry

end compare_constants_l526_526863


namespace factorial_mod_eq_l526_526633

theorem factorial_mod_eq :
  63! % 71 = 61! % 71 := 
sorry

end factorial_mod_eq_l526_526633


namespace compare_constants_l526_526840

theorem compare_constants :
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  in c < a ∧ a < b :=
by
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  sorry

end compare_constants_l526_526840


namespace binom_prime_division_l526_526103

-- Definitions based on conditions
def p_expansion (a : ℕ) (p : ℕ) : list ℕ :=
  if h : p > 1 then
    let rec loop (n : ℕ) (acc : list ℕ) :=
      if n = 0 then acc
      else loop (n / p) (n % p :: acc)
    loop a []
  else
    []

def m_prec_p (m n : ℕ) (p : ℕ) : Prop :=
  p > 1 ∧ m ≤ n ∧
  (∀ i ∈ (p_expansion m p), i < p) ∧
  (∀ i ∈ (p_expansion n p), i < p) ∧
  list.length (p_expansion m p) = list.length (p_expansion n p) ∧
  list.all2 (λ a_i b_i, a_i ≤ b_i) (p_expansion m p) (p_expansion n p)

-- Lean 4 statement

theorem binom_prime_division (m n p : ℕ) (hp : nat.prime p) :
  (p ∣ (nat.choose n m) ↔ ¬ m_prec_p m n p) := 
sorry

end binom_prime_division_l526_526103


namespace time_to_cross_tree_l526_526229

noncomputable def train_length : ℕ := 1400
noncomputable def platform_length : ℕ := 700
noncomputable def time_to_pass_platform : ℕ := 150

theorem time_to_cross_tree :
  ∃ t : ℕ, t = train_length / ( (train_length + platform_length) / time_to_pass_platform ) :=
begin
  use 100,
  sorry -- proof steps are not required
end

end time_to_cross_tree_l526_526229


namespace fraction_difference_l526_526691

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l526_526691


namespace positive_integer_condition_l526_526403

theorem positive_integer_condition (p : ℕ) (hp : 0 < p) : 
  (∃ k : ℤ, k > 0 ∧ 4 * p + 17 = k * (3 * p - 8)) ↔ p = 3 :=
by {
  sorry
}

end positive_integer_condition_l526_526403


namespace only_A_is_quadratic_l526_526622

def is_quadratic_equation (e : Expr) : Prop :=
  ∃ a b c, a ≠ 0 ∧ (e = a * x^2 + b * x + c = 0)

def equation_A := 2 * x^2 = 5 * x - 1
def equation_B := x + 1 / x = 2
def equation_C := (x - 3) * (x + 1) = x^2 - 5
def equation_D := 3 * x - y = 5

theorem only_A_is_quadratic:
  is_quadratic_equation equation_A ∧
  ¬is_quadratic_equation equation_B ∧
  ¬is_quadratic_equation equation_C ∧
  ¬is_quadratic_equation equation_D :=
by
  sorry

end only_A_is_quadratic_l526_526622


namespace cotangent_ratio_l526_526948

theorem cotangent_ratio (a b c : ℝ) (h : 9 * a ^ 2 + 9 * b ^ 2 - 19 * c ^ 2 = 0) :
  (Real.cot (Real.angle_cos (a ^ 2 + b ^ 2 - c ^ 2) (2 * a * b))) / 
  (Real.cot (Real.angle_cos (b ^ 2 + c ^ 2 - a ^ 2) (2 * b * c)) + 
   Real.cot (Real.angle_cos (c ^ 2 + a ^ 2 - b ^ 2) (2 * c * a))) = 5 / 9 :=
by sorry

end cotangent_ratio_l526_526948


namespace fraction_difference_is_correct_l526_526722

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l526_526722


namespace parallel_chords_l526_526002
open EuclideanGeometry

variables {circle : Type} [metric_space circle] [normed_group circle] [normed_space ℝ circle] [nontrivial circle]
variables {A B M N A1 B1 : circle}

def points_on_circle (C : circle) (A B M N : circle) : Prop :=
∃ r : ℝ, r > 0 ∧ dist C A = r ∧ dist C B = r ∧ dist C M = r ∧ dist C N = r

def perpendicular (M A1 N B : circle) : Prop :=
∀ {M A1 N B : circle}, ⟪M - A1, N - B⟫ = 0

def parallel (AA1 BB1: circle) : Prop :=
∀ (AA1 BB1 : circle), AA1 - BB1 = 0

theorem parallel_chords
  (circle : circle)
  (h1 : points_on_circle circle A B M N)
  (h2 : perpendicular M A1 N B)
  (h3 : perpendicular M B1 N A) :
  parallel AA1 BB1 :=
sorry

end parallel_chords_l526_526002


namespace proof_problem_l526_526831

-- Define the constants a, b, c
def a := 0.1 * Real.exp 0.1
def b := 1 / 9
def c := -Real.log 0.9

-- State the theorem to prove
theorem proof_problem : c < a ∧ a < b := 
by
  sorry

end proof_problem_l526_526831


namespace compare_abc_l526_526822
open Real

noncomputable def a : ℝ := 0.1 * exp(0.1)
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - log 0.9

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l526_526822


namespace sin_difference_simplification_l526_526528

theorem sin_difference_simplification : 
  ∀ (x : ℝ), x = sin 40 → sin 40 - sin 80 = x * (1 - 2 * sqrt (1 - x^2)) :=
by 
  intro x h,
  rw h,
  sorry

end sin_difference_simplification_l526_526528


namespace find_total_stock_worth_l526_526627

noncomputable def total_stock_worth (X : ℝ) : Prop :=
  let profit := 0.10 * (0.20 * X)
  let loss := 0.05 * (0.80 * X)
  loss - profit = 450

theorem find_total_stock_worth (X : ℝ) (h : total_stock_worth X) : X = 22500 :=
by
  sorry

end find_total_stock_worth_l526_526627


namespace expression_equivalence_l526_526304

theorem expression_equivalence :
  (4 + 3) * (4^2 + 3^2) * (4^4 + 3^4) * (4^8 + 3^8) * (4^16 + 3^16) * (4^32 + 3^32) * (4^64 + 3^64) = 3^128 - 4^128 :=
by
  sorry

end expression_equivalence_l526_526304


namespace possible_marking_partitions_in_7x7_l526_526502

/--
  A partition is an arbitrary side of a unit square in a 7x7 grid. 
  Partitions do not intersect if they do not share common points (including endpoints).
  The goal is to mark several non-intersecting partitions such that the projections of all 
  marked partitions on the horizontal and vertical sides of the 7x7 grid completely cover them.
-/
def non_intersecting_partitions (n : ℕ) (partitions : set (ℕ × ℕ)) : Prop :=
  ∀ (x : ℕ) (y : ℕ), (x, y) ∈ partitions → x <= n ∧ y <= n ∧
    ∀ (a b : ℕ), (x, y) ≠ (a, b) → (a, b) ∉ partitions ∨ (x ≠ a ∧ y ≠ b)

def projection_covered (n : ℕ) (partitions : set (ℕ × ℕ)) : Prop :=
  ∀ (i : ℕ), (i < n → ∃ (x : ℕ), (x, i) ∈ partitions) ∧ (i < n → ∃ (y : ℕ), (i, y) ∈ partitions)

theorem possible_marking_partitions_in_7x7:
  ∃ (partitions : set (ℕ × ℕ)), non_intersecting_partitions 7 partitions ∧ projection_covered 7 partitions :=
sorry

end possible_marking_partitions_in_7x7_l526_526502


namespace find_a_l526_526554

variable (a : ℤ) -- We assume a is an integer for simplicity

def point_on_x_axis (P : Nat × ℤ) : Prop :=
  P.snd = 0

theorem find_a (h : point_on_x_axis (4, 2 * a + 6)) : a = -3 :=
by
  sorry

end find_a_l526_526554


namespace R_has_at_least_one_side_integral_l526_526113

structure Rectangle (α : Type) :=
(NW SE : α)

def sides_parallel {α : Type} [LinearOrder α] (r1 r2 : Rectangle α) : Prop :=
  true -- Assume sides are parallel for demonstration purposes

def interiors_disjoint {α : Type} [LinearOrder α] (r1 r2 : Rectangle α) : Prop :=
  true -- Assume interiors are disjoint for demonstration purposes

def at_least_one_side_integral {α : Type} [LinearOrder α] [Add α] (r : Rectangle α) : Prop :=
  (NW r).x = floor ((SE r).x / 1) ∨ (NW r).y = floor ((SE r).y / 1)

theorem R_has_at_least_one_side_integral {α : Type} [LinearOrder α] [Add α]
  (R : Rectangle α) (R_i : fin (n : ℕ) -> Rectangle α)
  (h1 : ∀ i, sides_parallel R (R_i i))
  (h2 : ∀ i j, i ≠ j → interiors_disjoint (R_i i) (R_i j))
  (h3 : ∀ i, at_least_one_side_integral (R_i i)) :
  at_least_one_side_integral R := sorry

end R_has_at_least_one_side_integral_l526_526113


namespace no_five_consecutive_divisible_by_2025_l526_526037

def seq (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2025 : 
  ¬ ∃ (a : ℕ), (∀ (i : ℕ), i < 5 → 2025 ∣ seq (a + i)) := 
sorry

end no_five_consecutive_divisible_by_2025_l526_526037


namespace volume_solid_eq_l526_526298

noncomputable def volume_solid := 
  ∫ x in (0 : ℝ)..1, ∫ y in (x : ℝ)..1, (1 - y * sqrt y) ∂x ∂y

theorem volume_solid_eq : volume_solid = 3 / 14 :=
sorry

end volume_solid_eq_l526_526298


namespace problem1_problem2_l526_526902

noncomputable def f (x : ℝ) : ℝ := √3 * sin (2 * x) - 2 * (sin x)^2

-- Definitions for problem conditions
def P_terminal_side (x y : ℝ) (α : ℝ) : Prop :=
  x = cos α ∧ y = sin α

-- Problem 1: Evaluate f(α)
theorem problem1 (α : ℝ) (h : P_terminal_side 1 √3 α) : f α = -3 :=
by
  sorry

-- Problem 2: Range of f(x) over the specified interval
theorem problem2 (x : ℝ) (h : x ∈ Icc (-π/6) (π/3)) : -2 ≤ f x ∧ f x ≤ 1 :=
by
  sorry

end problem1_problem2_l526_526902


namespace find_initial_oranges_l526_526506

variable (O : ℕ)
variable (reserved_fraction : ℚ := 1 / 4)
variable (sold_fraction : ℚ := 3 / 7)
variable (rotten_oranges : ℕ := 4)
variable (good_oranges_today : ℕ := 32)

-- Define the total oranges before finding the rotten oranges
def oranges_before_rotten := good_oranges_today + rotten_oranges

-- Define the remaining fraction of oranges after reserving for friends and selling some
def remaining_fraction := (1 - reserved_fraction) * (1 - sold_fraction)

-- State the theorem to be proven
theorem find_initial_oranges (h : remaining_fraction * O = oranges_before_rotten) : O = 84 :=
sorry

end find_initial_oranges_l526_526506


namespace minimum_value_of_3a_plus_b_l526_526004

noncomputable def minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : Real.logBase 9 (9 * a + b) = Real.logBase 3 (Real.sqrt (a * b))) : ℝ :=
  3 * a + b

theorem minimum_value_of_3a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : Real.logBase 9 (9 * a + b) = Real.logBase 3 (Real.sqrt (a * b))) :
  minimum_value a b h1 h2 h3 = 12 + 6 * Real.sqrt 3 :=
sorry

end minimum_value_of_3a_plus_b_l526_526004


namespace probability_heads_10_out_of_12_l526_526600

theorem probability_heads_10_out_of_12 :
  let total_outcomes := (2^12 : ℕ)
  let favorable_outcomes := nat.choose 12 10
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 66 / 4096 :=
by 
  sorry

end probability_heads_10_out_of_12_l526_526600


namespace inequality_proof_l526_526852

-- Define constants
def a : ℝ := 0.1 * Real.exp 0.1
def b : ℝ := 1 / 9
def c : ℝ := -Real.log 0.9

-- Assert the inequalities
theorem inequality_proof : c < a ∧ a < b :=
by sorry

end inequality_proof_l526_526852


namespace find_m_and_period_find_smallest_positive_period_find_triangle_area_l526_526341

noncomputable def vector_a (m : ℝ) (h : m > 0) := (real.sqrt 2, m)
def vector_b (x : ℝ) := (real.sin x, real.cos x)
def f (x m : ℝ) : ℝ := real.sqrt 2 * real.sin x + m * real.cos x

theorem find_m_and_period {m : ℝ} (h : m > 0) :
  (∀ x, f x m ≤ 2) → m = real.sqrt 2 :=
sorry

theorem find_smallest_positive_period {m : ℝ} (h : m > 0) :
  m = real.sqrt 2 → ∀ x, (f (x + 2 * real.pi) m = f x m) :=
sorry

theorem find_triangle_area {A B C a b c : ℝ} (hC : C = real.pi / 3)
    (hc : c = real.sqrt 6) (h₁ : f (A - real.pi / 4) (real.sqrt 2) + f (B - real.pi / 4) (real.sqrt 2) = 12 * real.sqrt 2 * real.sin A * real.sin B)
    (h₂ : a = 2 * real.sqrt 2 * real.sin A) (h₃ : b = 2 * real.sqrt 2 * real.sin B) :
  1 / 2 * a * b * real.sin (real.pi / 3) = real.sqrt 3 / 4 :=
sorry

end find_m_and_period_find_smallest_positive_period_find_triangle_area_l526_526341


namespace cindy_same_color_probability_l526_526647

def box := ["red", "red", "red", "blue", "blue", "blue", "green", "yellow"]

def alice_draws (b : List String) : Finset (Finset String) := 
  Finset.powersetLen 3 (Finset.ofList b)

def bob_draws (a_draws b : List String) : Finset (Finset String) :=
  let remaining := b.filter (λ x, ¬ a_draws.contains x)
  Finset.powersetLen 2 (Finset.ofList remaining)

def cindy_draws (a_draws b_draws b : List String) : Finset (Finset String) :=
  let remaining := b.filter (λ x, ¬ a_draws.contains x ∧ ¬ b_draws.contains x)
  Finset.powersetLen 2 (Finset.ofList remaining)

def is_same_color (drawn : Finset String) : Bool :=
  ∀ x ∈ drawn, x = drawn.choose (by sorry)

theorem cindy_same_color_probability :
  let α := algebra_map (Fin ℕ) ℚ -- Probability ratio
  let a_draws := alice_draws box
  let favorable_ways := a_draws.sum (λ a, (bob_draws a box).sum (λ b, cindy_draws a b box).count (λ c, is_same_color c))
  let total_ways := a_draws.card * (bob_draws (a_draws.choose sorry) box).card * (cindy_draws (a_draws.choose sorry) ((bob_draws (a_draws.choose sorry) box).choose sorry) box).card
  α (favorable_ways) / α (total_ways) = 1 / 35 := 
sorry

end cindy_same_color_probability_l526_526647


namespace triangle_property_l526_526366

theorem triangle_property
  (A B C : ℝ)
  (a b c : ℝ)
  (R : ℝ)
  (hR : R = Real.sqrt 3)
  (h1 : a * Real.sin C + Real.sqrt 3 * c * Real.cos A = 0)
  (h2 : b + c = Real.sqrt 11)
  (htri : a / Real.sin A = 2 * R ∧ b / Real.sin B = 2 * R ∧ c / Real.sin C = 2 * R):
  a = 3 ∧ (1 / 2 * b * c * Real.sin A = Real.sqrt 3 / 2) := 
sorry

end triangle_property_l526_526366


namespace folding_square_problem_l526_526261

theorem folding_square_problem
  (side_len : ℝ)
  (E F : ℝ) 
  (AE CF : ℝ)
  (BD : ℝ)
  (k m : ℝ)
  (AE_eq_CF : AE = CF)
  (folding_coincide : AE = 2 - 3 * Real.sqrt 2)
  (resulting_length : AE = Real.sqrt k - m) :
  k + m = 20 :=
by
  -- Setting up the given conditions
  have h1 : side_len = 2 := rfl
  have h2 : BD = Real.sqrt 8 := rfl
  -- Assigning the given values based on the problem
  have h3 : k = 18 := rfl
  have h4 : m = 2 := rfl
  -- The length of AE is provided as AE = 2 - 3√2
  have h5 : AE = 2 - 3 * Real.sqrt 2 := folding_coincide
  -- Given resulting length equation
  have h6 : AE = Real.sqrt k - m := resulting_length
  -- Therefore we have k + m
  show k + m = 20 by
    rw [←h3, ←h4]
    exact rfl

end folding_square_problem_l526_526261


namespace max_angle_MPN_l526_526085

-- Define points M and N in 2D Cartesian coordinate system
def M := (-1 : ℝ, 2 : ℝ)
def N := (1 : ℝ, 4 : ℝ)

-- Define the predicate for point P on the X-axis
def P (x : ℝ) := (x, 0 : ℝ)

-- Statement of the problem, proving that the x-coordinate of P that maximizes the angle ∠MPN is x = 1
theorem max_angle_MPN : ∃ x : ℝ, P x ∧ (x = 1) :=
by
  sorry

end max_angle_MPN_l526_526085


namespace repeating_decimal_exceeds_finite_decimal_by_l526_526730

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l526_526730


namespace sum_of_solutions_eq_zero_l526_526325

noncomputable def f (x : ℝ) : ℝ := 2 ^ |x| + 5 * |x|

theorem sum_of_solutions_eq_zero (x : ℝ) (hx : f x = 28) :
  x + -x = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l526_526325


namespace cyclic_quadrilateral_WXYZ_l526_526494

-- Define points and tangency conditions.
variables {A B C D W Z K X Y : Point}

-- Assume ABCD is a convex quadrilateral
def is_convex_quadrilateral (A B C D : Point) : Prop := 
  convex_quadrilateral A B C D

-- Incircle tangency conditions for triangles
def incircle_tangent_to_ABD (A B D W Z K : Point) : Prop :=
  tangent_point A B D W Z K

def incircle_tangent_to_CBD (C B D X Y K : Point) : Prop :=
  tangent_point C B D X Y K

-- Main theorem stating that quadrilateral WXYZ is cyclic
theorem cyclic_quadrilateral_WXYZ 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : incircle_tangent_to_ABD A B D W Z K)
  (h3 : incircle_tangent_to_CBD C B D X Y K) :
  cyclic_quad W X Y Z :=
sorry

end cyclic_quadrilateral_WXYZ_l526_526494


namespace sum_of_all_possible_k_values_l526_526447

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l526_526447


namespace compare_a_b_c_l526_526819

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b :=
by
  have : c < b := sorry,
  have : a < b := sorry,
  have : a > c := sorry,
  exact ⟨this.2, this.1⟩

end compare_a_b_c_l526_526819


namespace transport_storage_fee_sufficiency_l526_526641

def cost_per_batch(x : ℕ) (k : ℚ) : ℚ := 4 + k * 20 * x

axiom storage_fee_proportion (x : ℕ) : cost_per_batch 4 (1/4) = 52

theorem transport_storage_fee_sufficiency : ∃ x : ℕ, 
  (0 < x ∧ x ≤ 36 ∧ cost_per_batch x (1/4) ≤ 48) :=
by {
  suffices h : ∃ x : ℕ, (0 < x ∧ x ≤ 36 ∧ cost_per_batch x (1/4) ≤ 48),
  { exact h, },
  sorry 
}

end transport_storage_fee_sufficiency_l526_526641


namespace correct_statements_count_l526_526158

def statement1 (T : Type) [Triangle T] : Prop :=
  ∀ (t : T), (is_acute t) → (altitudes_intersect t)

def statement2 (T : Type) [Triangle T] : Prop :=
  ∀ (t : T), (is_midline_correct t = false)

def statement3 (T : Type) [Triangle T] : Prop :=
  ∀ (t : T), (angles_relation t) → (is_right_triangle t)

def statement4 (T : Type) [Triangle T] : Prop :=
  ∀ (t : T), (exterior_angle_greater t)

def statement5 (T : Type) [Triangle T] : Prop :=
  ∀ (t : T), ((has_sides t 8 10) → (shortest_side_range t 2 18 = false))

theorem correct_statements_count (T : Type) [Triangle T] :
  let s1 := statement1 T,
      s2 := statement2 T,
      s3 := statement3 T,
      s4 := statement4 T,
      s5 := statement5 T in
  (count_correct [s1, s2, s3, s4, s5]) = 2 :=
by sorry

end correct_statements_count_l526_526158


namespace sum_of_possible_ks_l526_526431

theorem sum_of_possible_ks : 
  (∃ (j k : ℕ), (1 < j) ∧ (1 < k) ∧ j ≠ k ∧ ((1/j : ℝ) + (1/k : ℝ) = (1/4))) → 
  (∑ k in {20, 12, 8, 6, 5}, k) = 51 :=
begin
  sorry
end

end sum_of_possible_ks_l526_526431


namespace minimum_area_triangle_DEF_l526_526220

def is_triangle (D E F : ℝ × ℝ) : Prop :=
  ¬ collinear D E F

def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

def area (D E F : ℝ × ℝ) : ℝ :=
  let ((x1, y1), (x2, y2), (x3, y3)) := (D, E, F)
  (1 / 2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem minimum_area_triangle_DEF : 
  ∃ (F : ℝ × ℝ), F.1 ∈ ℤ ∧ F.2 ∈ ℤ ∧ is_triangle (0,0) (24,10) F ∧ area (0,0) (24,10) F = 5 :=
by sorry

end minimum_area_triangle_DEF_l526_526220


namespace initial_investment_l526_526525

theorem initial_investment (A P : ℝ) (r : ℝ) (n t : ℕ) 
  (hA : A = 16537.5)
  (hr : r = 0.10)
  (hn : n = 2)
  (ht : t = 1)
  (hA_calc : A = P * (1 + r / n) ^ (n * t)) :
  P = 15000 :=
by {
  sorry
}

end initial_investment_l526_526525


namespace sum_of_digits_of_N_l526_526264

theorem sum_of_digits_of_N :
  ∀ (N : ℕ), (N * (N + 1) / 2 = 2145) → (N = 65 ∧ (6 + 5 = 11)) :=
by
  intro N h
  have hN : N = 65 := sorry
  rw hN
  exact ⟨rfl, by norm_num⟩

end sum_of_digits_of_N_l526_526264


namespace tangent_line_at_x4_l526_526945

noncomputable def f : ℝ → ℝ := sorry  -- We're not providing the actual function f here

theorem tangent_line_at_x4 (f : ℝ → ℝ) (x : ℝ) (y : ℝ) 
(h_tangent : ∀ x, y = 3 * x + 5)
(h_slope : ∀ x, deriv f x = 3)
(h_value : ∀ x, f 4 = y) : f 4 + deriv f 4 = 20 :=
sorry

end tangent_line_at_x4_l526_526945


namespace minimum_streetlights_l526_526275

theorem minimum_streetlights 
    (AB : ℕ) (BC : ℕ) (dist : ℕ)
    (hAB : AB = 175)
    (hBC : BC = 125)
    (hGCD : Nat.gcd AB BC = dist)
    (hdist : dist = 25) :
    ∃ (N : ℕ), N = 13 :=
by
  use 13
  rw [hAB, hBC, hdist]
  have h1 : 175 / 25 = 7 := rfl
  have h2 : 125 / 25 = 5 := rfl
  have h3 : (7 + 1) + (5 + 1) - 1 = 13 := rfl
  exact h3

end minimum_streetlights_l526_526275


namespace find_k_when_root_is_1_find_k_when_x1_squared_x2_l526_526347

open Real

theorem find_k_when_root_is_1 (k : ℝ) : (k - 1) * (1 : ℝ) ^ 2 - 4 * (1 : ℝ) + 3 = 0 → k = 2 :=
sorry

theorem find_k_when_x1_squared_x2 (k : ℝ) :
  let discriminant := 16 - 4 * (k - 1) * 3 in
  (∃ x1 x2 : ℝ, (k - 1) * x1 ^ 2 - 4 * x1 + 3 = 0 ∧ (k - 1) * x2 ^ 2 - 4 * x2 + 3 = 0 ∧
  x1 ^ 2 * x2 + x1 * x2 ^ 2 = 3) ∧ discriminant ≥ 0 → k = -1 :=
sorry

end find_k_when_root_is_1_find_k_when_x1_squared_x2_l526_526347


namespace alpha_in_second_quadrant_l526_526342

theorem alpha_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 > 0 :=
by
  -- Given conditions
  have : Real.sin α > 0 := h1
  have : Real.cos α < 0 := h2
  sorry

end alpha_in_second_quadrant_l526_526342


namespace functions_with_inverses_l526_526326

-- Conditions
def GraphA := ∀ x y : ℝ, (y = x) ↔ (x = y)
def GraphB := ∀ x y : ℝ, (y = x^2) ↔ (x = sqrt(y)) -- and tests whether the given values violate bijection
def GraphC := ∀ x y : ℝ, (y = 4 - x) ↔ (x = 4 - y)
def GraphD := ∀ x y : ℝ, (y = sqrt(9 - x^2)) ↔ exists z : ℝ, (z ≠ y ∧ (z = sqrt(9 - x^2)))
def GraphE := ∀ x y : ℝ, (y = 5 * exp(-x)) ↔ (x = -log(y / 5))

-- Proof problem statement
theorem functions_with_inverses : 
  (GraphA) ∧ 
  (GraphC) ∧ 
  (GraphE) ∧ 
  ¬(GraphB) ∧ 
  ¬(GraphD) :=
by
  sorry

end functions_with_inverses_l526_526326


namespace num_planes_from_four_non_coplanar_points_l526_526164

theorem num_planes_from_four_non_coplanar_points 
    (P1 P2 P3 P4 : Point)
    (h1 : ¬ Collinear P1 P2 P3)
    (h2 : ¬ Collinear P1 P2 P4)
    (h3 : ¬ Collinear P1 P3 P4)
    (h4 : ¬ Collinear P2 P3 P4) : 
    number_of_planes {P1, P2, P3, P4} = 4 := 
sorry

end num_planes_from_four_non_coplanar_points_l526_526164


namespace area_of_trapezoid_ACDE_l526_526276

-- Define the given geometric conditions and dimensions.
variables (A B C D E F : Type)
variables [Euclidean_space A] [Linear_ordered_field B]
variables [Metric_space C] [Metric_space D] [Metric_space E] [Metric_space F]

variable AB : ℝ
variable BC : ℝ

-- Given conditions
def folded_1 := ∃ A B C F, F ∈ line A C ∧ point F = B
def folded_2 := ∃ D E, line D E ∥ line A C ∧ F ∈ line D E

-- The final goal
theorem area_of_trapezoid_ACDE :
  (AB = 5) →
  (BC = 3) →
  folded_1 →
  folded_2 →
  area_trapezoid_ACDE = 22.5 :=
by
  -- sorry is used as a placeholder for the proof
  sorry

end area_of_trapezoid_ACDE_l526_526276


namespace fraction_of_pizza_covered_by_pepperoni_l526_526271

noncomputable def pepperoni_fraction : ℝ :=
  let pizza_diameter := 18
  let number_of_pepperonis_across := 8
  let total_pepperonis := 36
  let pepperoni_diameter := pizza_diameter / number_of_pepperonis_across
  let pepperoni_radius := pepperoni_diameter / 2
  let pepperoni_area := π * (pepperoni_radius ^ 2)
  let total_pepperoni_area := total_pepperonis * pepperoni_area
  let pizza_radius := pizza_diameter / 2
  let pizza_area := π * (pizza_radius ^ 2)
  total_pepperoni_area / pizza_area

theorem fraction_of_pizza_covered_by_pepperoni : pepperoni_fraction = 9 / 16 :=
by
  -- The specific proof will be filled here.
  sorry

end fraction_of_pizza_covered_by_pepperoni_l526_526271


namespace problem_l526_526869

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x ^ 2 - x - a * Real.log (x - a)

def monotonicity_f (a : ℝ) : Prop :=
  if a = 0 then
    ∀ x : ℝ, 0 < x → (x < 1 → f x 0 < f (x + 1) 0) ∧ (x > 1 → f x 0 > f (x + 1) 0)
  else if a > 0 then
    ∀ x : ℝ, a < x → (x < a + 1 → f x a < f (x + 1) a) ∧ (x > a + 1 → f x a > f (x + 1) a)
  else if -1 < a ∧ a < 0 then
    ∀ x : ℝ, 0 < x → (x < a + 1 → f x a < f (x + 1) a) ∧ (x > a + 1 → f (x + 1) a > f x a)
  else if a = -1 then
    ∀ x : ℝ, -1 < x → f x (-1) < f (x + 1) (-1)
  else
    ∀ x : ℝ, a < x → (x < 0 → f (x + 1) a > f x a) ∧ (0 < x → f x a > f (x + 1) a)

noncomputable def g (x a : ℝ) : ℝ := f (x + a) a - a * (x + (1/2) * a - 1)

def extreme_points (x₁ x₂ a : ℝ) : Prop :=
  x₁ < x₂ ∧ ∀ x : ℝ, 0 < x → x < 1 → g x a = 0

theorem problem (a : ℝ) (x₁ x₂ : ℝ) (hx : extreme_points x₁ x₂ a) (h_dom : -1/4 < a ∧ a < 0) :
  0 < f x₁ a - f x₂ a ∧ f x₁ a - f x₂ a < 1/2 := sorry

end problem_l526_526869


namespace find_point_P_l526_526022

noncomputable def curve (x : ℝ) : ℝ := 2 * x ^ 2 + 4 * x

noncomputable def slope_at_point (x : ℝ) : ℝ := 4 * x + 4

def point_P : ℝ × ℝ := (3, curve 3)

theorem find_point_P : ∃ P : ℝ × ℝ, P = point_P ∧ slope_at_point P.1 = 16 :=
by
  sorry

end find_point_P_l526_526022


namespace Joan_spent_on_shirt_l526_526476

/-- Joan spent $15 on shorts, $14.82 on a jacket, and a total of $42.33 on clothing.
    Prove that Joan spent $12.51 on the shirt. -/
theorem Joan_spent_on_shirt (shorts jacket total: ℝ) 
                            (h1: shorts = 15)
                            (h2: jacket = 14.82)
                            (h3: total = 42.33) :
  total - (shorts + jacket) = 12.51 :=
by
  sorry

end Joan_spent_on_shirt_l526_526476


namespace triangle_circumcenters_perpendicular_l526_526132

theorem triangle_circumcenters_perpendicular 
  (A B C D E P F G O1 O2 : Point)
  (hD : D ∈ segment A B)
  (hE : E ∈ segment A C)
  (hDE_parallel_BC : parallel (line DE) (line BC))
  (hP_in_ABC : P ∈ int_triangle A B C)
  (hF : F = (line PB) ∩ (line DE))
  (hG : G = (line PC) ∩ (line DE))
  (hO1 : circumcenter (triangle P D G) = O1)
  (hO2 : circumcenter (triangle P E F) = O2) :
  perpendicular (line AP) (line O1 O2) :=
sorry

end triangle_circumcenters_perpendicular_l526_526132


namespace triangle_ADE_is_isosceles_l526_526991

-- Definitions of points, lines, circle, and intersection
variable (C : Circle) (A B C : Point) (d t : Line)
variable (E D : Point) 

-- Conditions
variable (h1 : B ∈ C)
variable (h2 : C ∈ C)
variable (h3 : A ∈ C) (h3a : A ≠ B) (h3b : A ≠ C)
variable (tangent_at_A : Tangent t C A) 
variable (d_intersects_C : LineIntersects d C B C)
variable (t_intersects_d_at_E : LineIntersects d t E)
variable (angle_bisector_AD : AngleBisector (Angle A B C) (Segment A D) (Segment D C))

-- Theorem statement: triangle ADE is isosceles
theorem triangle_ADE_is_isosceles (h : ∃ α β, Angle A C D = α ∧ Angle C A D = β ∧ Angle D A E = α + β):
  IsIsosceles (Triangle A D E) :=
sorry

end triangle_ADE_is_isosceles_l526_526991


namespace inequality_proof_l526_526855

-- Define constants
def a : ℝ := 0.1 * Real.exp 0.1
def b : ℝ := 1 / 9
def c : ℝ := -Real.log 0.9

-- Assert the inequalities
theorem inequality_proof : c < a ∧ a < b :=
by sorry

end inequality_proof_l526_526855


namespace sequence_general_formula_max_n_sum_less_2023_l526_526874

theorem sequence_general_formula (a : ℕ → ℤ) (a_recurrence : ∀ n, a (n + 1) - 2 * a n = n - 1) (a_initial : a 1 = 1) :
  ∀ n, a n = 2^n - n := sorry

theorem max_n_sum_less_2023 (a : ℕ → ℤ) (S : ℕ → ℤ) (a_recurrence : ∀ n, a (n + 1) - 2 * a n = n - 1) (a_initial : a 1 = 1)
  (S_def : ∀ n, S n = (Finset.range n).sum (λ k, a (k + 1))) :
  (∀ n, S n < 2023) → 10 = 10 := sorry

end sequence_general_formula_max_n_sum_less_2023_l526_526874


namespace sequence_bound_100_l526_526381

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = a (n - 1) + 1 / a (n - 1)

theorem sequence_bound_100 (a : ℕ → ℝ) (h : seq a) : 
  14 < a 100 ∧ a 100 < 18 := 
sorry

end sequence_bound_100_l526_526381


namespace curve_c2_equation_distance_AB_l526_526422

noncomputable def curve_C1 (α : ℝ) : ℝ × ℝ :=
  (3 * Real.cos α, 3 + 3 * Real.sin α)

noncomputable def curve_C2 (α : ℝ) : ℝ × ℝ :=
  (6 * Real.cos α, 6 + 6 * Real.sin α)

theorem curve_c2_equation :
  ∀ α : ℝ, 
  let (x1, y1) := curve_C1 α
  let (x2, y2) := curve_C2 α
  (x2, y2) = (6 * Real.cos α, 6 + 6 * Real.sin α) :=
by
  intros
  simp [curve_C1, curve_C2]

theorem distance_AB :
  ∀ A B : ℝ × ℝ, 
  A = (3 * Real.cos (π / 3), 3 + 3 * Real.sin (π / 3)) ∧ 
  B = (6 * Real.cos (π / 3), 6 + 6 * Real.sin (π / 3)) → 
  Real.dist B A = 3 * Real.sqrt 3 :=
by
  intros
  sorry

end curve_c2_equation_distance_AB_l526_526422


namespace number_of_similar_dividing_lines_l526_526474

theorem number_of_similar_dividing_lines 
  (A B C : Point)
  (h1 : ∠A > ∠B)
  (h2 : ∠B > ∠C)
  (h3 : ∠A ≠ 90)
  : ∃ n, n = 6 ∧ (∃ (lines : List Line), lines.length = n ∧ 
    ∀ l ∈ lines, divides_triangle_similar (triangle ABC) l) :=
sorry

end number_of_similar_dividing_lines_l526_526474


namespace proof_problem_l526_526369

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := x^3 * f(x)
def a := g(log 2 (1 / 5))
def b := g(1 / Real.exp 1)
def c := g(1 / 2)

theorem proof_problem :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, 0 < x → 3 * f x + x * deriv f x > 0) →
  b < c ∧ c < a := sorry

end proof_problem_l526_526369


namespace repeating_decimal_difference_l526_526707

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l526_526707


namespace proof_problem_l526_526833

-- Define the constants a, b, c
def a := 0.1 * Real.exp 0.1
def b := 1 / 9
def c := -Real.log 0.9

-- State the theorem to prove
theorem proof_problem : c < a ∧ a < b := 
by
  sorry

end proof_problem_l526_526833


namespace triangle_circumradius_sqrt3_triangle_area_l526_526365

variables {a b c : ℝ} {A B C : ℝ} {R : ℝ}
variables (triangle_ABC : a = 2 * R * sin A ∧ c = 2 * R * sin C ∧ b = 2 * R * sin B)

theorem triangle_circumradius_sqrt3 (ha : a * sin C + sqrt 3 * c * cos A = 0) (hR : R = sqrt 3)
  (hsinC_nonzero : sin C ≠ 0) : a = 3 :=
begin
  sorry
end

theorem triangle_area (ha : a = 3) (hb : b + c = sqrt 11) (hA : A = 2 * real.pi / 3) : 
  (1/2)*b*c*sin A = sqrt 3 / 2 :=
begin
  sorry
end

end triangle_circumradius_sqrt3_triangle_area_l526_526365


namespace compare_constants_l526_526867

noncomputable def a : ℝ := 0.1 * real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - real.log 0.9

theorem compare_constants : c < a ∧ a < b :=
by sorry

end compare_constants_l526_526867


namespace weight_of_8_moles_of_AlI3_l526_526611

noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_I : ℝ := 126.90
noncomputable def molecular_weight_AlI3 : ℝ := atomic_weight_Al + 3 * atomic_weight_I

theorem weight_of_8_moles_of_AlI3 : 
  (8 * molecular_weight_AlI3) = 3261.44 := by
sorry

end weight_of_8_moles_of_AlI3_l526_526611


namespace best_possible_overall_standing_l526_526073

noncomputable def N : ℕ := 100 -- number of participants
noncomputable def M : ℕ := 14  -- number of stages

-- Define a competitor finishing 93rd in each stage
def finishes_93rd_each_stage (finishes : ℕ → ℕ) : Prop :=
  ∀ i, i < M → finishes i = 93

-- Define the best possible overall standing
theorem best_possible_overall_standing
  (finishes : ℕ → ℕ) -- function representing stage finishes for the competitor
  (h : finishes_93rd_each_stage finishes) :
  ∃ k, k = 2 := 
sorry

end best_possible_overall_standing_l526_526073


namespace proof_problem_l526_526829

-- Define the constants a, b, c
def a := 0.1 * Real.exp 0.1
def b := 1 / 9
def c := -Real.log 0.9

-- State the theorem to prove
theorem proof_problem : c < a ∧ a < b := 
by
  sorry

end proof_problem_l526_526829


namespace square_field_side_length_l526_526232

theorem square_field_side_length (t : ℕ) (v : ℕ) 
  (run_time : t = 56) 
  (run_speed : v = 9) : 
  ∃ l : ℝ, l = 35 := 
sorry

end square_field_side_length_l526_526232


namespace sum_of_x_values_l526_526198

theorem sum_of_x_values (x : ℝ) (h : Real.sqrt ((x + 5)^2) = 9) :
  {x | Real.sqrt ((x + 5)^2) = 9}.sum = -10 :=
by
  sorry

end sum_of_x_values_l526_526198


namespace angle_B_is_60_degrees_l526_526946

theorem angle_B_is_60_degrees 
  {A B C : Type}
  [InnerProductSpace ℝ A]
  (a b c : A) 
  (h : dist a c * dist a c + dist b c * dist b c - dist a b * dist a b = dist a c * dist b c) :
  angle a b c = 60 :=
by
  sorry

end angle_B_is_60_degrees_l526_526946


namespace find_m_l526_526886

noncomputable def f : ℝ → ℝ := sorry

theorem find_m (h₁ : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 2) (h₂ : f 2 = m) : m = -1 / 2 :=
by
  sorry

end find_m_l526_526886


namespace city_with_no_outgoing_road_city_with_direct_reach_one_direction_change_for_connectivity_catastrophic_networks_count_l526_526072

-- 1. For any catastrophic network of 6 cities, there exists at least one city with no outgoing roads.
theorem city_with_no_outgoing_road (network : Type) [catastrophic_network : catastrophic network] : 
  ∃ city, ∀ road, ¬(outgoing road city) := sorry

-- 2. For any catastrophic network of 6 cities, there exists at least one city from which one can directly reach all other cities.
theorem city_with_direct_reach (network : Type) [catastrophic_network : catastrophic network] : 
  ∃ city, ∀ other_city, ∃ road, connects road city other_city := sorry

-- 3. For any catastrophic network of 6 cities, changing the direction of exactly one road suffices to make the network fully connected.
theorem one_direction_change_for_connectivity (network : Type) [catastrophic_network : catastrophic network] : 
  ∃ road, ∃ new_network, changes road new_network network ∧ fully_connected new_network := sorry

-- 4. The number of catastrophic networks among 6 cities is exactly 720.
theorem catastrophic_networks_count : 
  number_of_catastrophic_networks 6 = 720 := sorry

end city_with_no_outgoing_road_city_with_direct_reach_one_direction_change_for_connectivity_catastrophic_networks_count_l526_526072


namespace acute_angle_sine_l526_526005
--import Lean library

-- Define the problem conditions and statement
theorem acute_angle_sine (a : ℝ) (h1 : 0 < a) (h2 : a < π / 2) (h3 : Real.sin a = 0.6) :
  π / 6 < a ∧ a < π / 4 :=
by 
  sorry

end acute_angle_sine_l526_526005


namespace combined_yellow_ratio_is_33_percent_l526_526579

def bag_a_beans : ℕ := 24
def bag_b_beans : ℕ := 32
def bag_c_beans : ℕ := 34

def bag_a_yellow_ratio : ℚ := 40 / 100
def bag_b_yellow_ratio : ℚ := 35 / 100
def bag_c_yellow_ratio : ℚ := 25 / 100

def num_yellow_beans_in_bag (total_beans : ℕ) (yellow_ratio : ℚ) : ℕ :=
  (total_beans : ℚ * yellow_ratio).to_nat

theorem combined_yellow_ratio_is_33_percent :
  let yellow_beans_a := num_yellow_beans_in_bag bag_a_beans bag_a_yellow_ratio in
  let yellow_beans_b := num_yellow_beans_in_bag bag_b_beans bag_b_yellow_ratio in
  let yellow_beans_c := num_yellow_beans_in_bag bag_c_beans bag_c_yellow_ratio in
  let total_yellow_beans := yellow_beans_a + yellow_beans_b + yellow_beans_c in
  let total_beans := bag_a_beans + bag_b_beans + bag_c_beans in
  total_yellow_beans * 100 / total_beans = 33 :=
by
  sorry

end combined_yellow_ratio_is_33_percent_l526_526579


namespace factorization_of_difference_of_squares_l526_526770

theorem factorization_of_difference_of_squares (m : ℝ) : 
  m^2 - 16 = (m + 4) * (m - 4) := 
by 
  sorry

end factorization_of_difference_of_squares_l526_526770


namespace factor_difference_of_squares_l526_526767

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l526_526767


namespace first_pump_rate_is_180_l526_526258

-- Define the known conditions
variables (R : ℕ) -- The rate of the first pump in gallons per hour
def second_pump_rate : ℕ := 250 -- The rate of the second pump in gallons per hour
def second_pump_time : ℕ := 35 / 10 -- 3.5 hours represented as a fraction
def total_pump_time : ℕ := 60 / 10 -- 6 hours represented as a fraction
def total_volume : ℕ := 1325 -- Total volume pumped by both pumps in gallons

-- Define derived conditions from the problem
def second_pump_volume : ℕ := second_pump_rate * second_pump_time -- Volume pumped by the second pump
def first_pump_volume : ℕ := total_volume - second_pump_volume -- Volume pumped by the first pump
def first_pump_time : ℕ := total_pump_time - second_pump_time -- Time the first pump was used

-- The main theorem to prove that the rate of the first pump is 180 gallons per hour
theorem first_pump_rate_is_180 : R = 180 :=
by
  -- The proof would go here
  sorry

end first_pump_rate_is_180_l526_526258


namespace divides_segments_ratios_l526_526274

theorem divides_segments_ratios
  (A B C D M N E F : Point)
  (m n e f : ℝ)
  (hABM : ∃ (AM MB : ℝ), AM/MB = m/n)
  (hDCN : ∃ (DN NC : ℝ), DN/NC = m/n)
  (hBCE : ∃ (BE EC : ℝ), BE/EC = e/f)
  (hADF : ∃ (AF FD : ℝ), AF/FD = e/f)
  (convex : isConvexQuadrilateral A B C D)
  (hM : isOnSegment A B M)
  (hN : isOnSegment D C N)
  (hE : isOnSegment B C E)
  (hF : isOnSegment A D F) :
  (∃ O : Point, isOnSegment E F O ∧ EO/OE = m/n) ∧
  (∃ P : Point, isOnSegment M N P ∧ MP/PN = e/f) := sorry

end divides_segments_ratios_l526_526274


namespace sum_of_all_possible_k_values_l526_526445

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l526_526445


namespace number_is_square_l526_526328

theorem number_is_square (n : ℕ) (h : (n^5 + n^4 + 1).factors.length = 6) : ∃ k : ℕ, n^3 - n + 1 = k^2 :=
begin
  sorry
end

end number_is_square_l526_526328


namespace tetrahedron_volume_A1_A2_A3_A4_height_from_A4_to_A1A2A3_l526_526217

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A1 : Point3D := { x := 1, y := 5, z := -7 }
def A2 : Point3D := { x := -3, y := 6, z := 3 }
def A3 : Point3D := { x := -2, y := 7, z := 3 }
def A4 : Point3D := { x := -4, y := 8, z := -12 }

noncomputable def vector (p1 p2 : Point3D) : Point3D :=
{ x := p2.x - p1.x,
  y := p2.y - p1.y,
  z := p2.z - p1.z }

noncomputable def scalarTripleProduct (u v w : Point3D) : ℝ :=
u.x * (v.y * w.z - v.z * w.y) - u.y * (v.x * w.z - v.z * w.x) + u.z * (v.x * w.y - v.y * w.x)

noncomputable def tetrahedronVolume (p1 p2 p3 p4 : Point3D) : ℝ :=
(1 / 6) * real.abs (scalarTripleProduct (vector p1 p2) (vector p1 p3) (vector p1 p4))

noncomputable def crossProduct (u v : Point3D) : Point3D :=
{ x := u.y * v.z - u.z * v.y,
  y := u.z * v.x - u.x * v.z,
  z := u.x * v.y - u.y * v.x }

noncomputable def triangleArea (p1 p2 p3 : Point3D) : ℝ :=
(1 / 2) * real.abs (norm (crossProduct (vector p1 p2) (vector p1 p3)))

noncomputable def heightTetrahedronBase (p1 p2 p3 p4 : Point3D) : ℝ :=
2 * tetrahedronVolume p1 p2 p3 p4 / triangleArea p1 p2 p3

theorem tetrahedron_volume_A1_A2_A3_A4 : tetrahedronVolume A1 A2 A3 A4 = 17.5 :=
sorry

theorem height_from_A4_to_A1A2A3 : heightTetrahedronBase A1 A2 A3 A4 = 7 :=
sorry

end tetrahedron_volume_A1_A2_A3_A4_height_from_A4_to_A1A2A3_l526_526217


namespace max_real_part_z7_l526_526507

open Complex

-- Define provided complex numbers
def z_set := {z1 := Complex.mk (-2) 0, 
              z2 := Complex.mk (Real.sqrt 3) 1, 
              z3 := Complex.mk (Real.sqrt 2) (-Real.sqrt 2),
              z4 := Complex.mk (-1) (-Real.sqrt 3), 
              z5 := Complex.mk 0 (-2)}

-- Define the real part of z^7 function
def real_part_z7 (z : Complex) : ℝ :=
  (z^7).re

-- Main statement to prove the correct answer
theorem max_real_part_z7 : real_part_z7 z_set.z2 > real_part_z7 z_set.z1 ∧
                           real_part_z7 z_set.z2 > real_part_z7 z_set.z3 ∧
                           real_part_z7 z_set.z2 > real_part_z7 z_set.z4 ∧
                           real_part_z7 z_set.z2 > real_part_z7 z_set.z5 :=
by sorry

end max_real_part_z7_l526_526507


namespace sqrt_1708249_eq_1307_l526_526285

theorem sqrt_1708249_eq_1307 :
  ∃ (n : ℕ), n * n = 1708249 ∧ n = 1307 :=
sorry

end sqrt_1708249_eq_1307_l526_526285


namespace repeatingDecimal_exceeds_l526_526697

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l526_526697


namespace perpendicular_planes_of_perpendicular_lines_and_planes_l526_526491

-- Definitions for lines and planes
variables {L M : Type} [linear_ordered_field L] [linear_ordered_field M]

def perpendicular (a b : Type) : Prop := sorry
def parallel (a b : Type) : Prop := sorry

variables (l m : L) (α β : M)

-- Main statement
theorem perpendicular_planes_of_perpendicular_lines_and_planes
  (h_l_m : perpendicular l m)
  (h_l_α : perpendicular l α)
  (h_m_β : perpendicular m β) :
  perpendicular α β :=
sorry

end perpendicular_planes_of_perpendicular_lines_and_planes_l526_526491


namespace fraction_of_dark_tiles_in_tiled_floor_preserved_symmetry_l526_526279

theorem fraction_of_dark_tiles_in_tiled_floor_preserved_symmetry 
  (n : ℕ) (h : n = 4) 
  (corner_symmetry : ∀ (i j : ℕ), i < 2 → j < 2 → tile_color (4*i + j) = tile_color (2*i + j)) :
  let dark_tiles := 8 in
  let total_tiles := 16 in
  (dark_tiles / total_tiles : ℚ) = (1 / 2 : ℚ) :=
by {
  sorry
}

end fraction_of_dark_tiles_in_tiled_floor_preserved_symmetry_l526_526279


namespace vision_data_l526_526546

theorem vision_data (L V : ℝ) (approx : ℝ) (h1 : L = 5 + real.log10 V) (h2 : L = 4.9) (h3 : approx = 1.259) : 
  V = 0.8 := 
sorry

end vision_data_l526_526546


namespace centroid_trace_l526_526041

def Point := ℝ × ℝ
def Line := Point × Point

variables (A B C G M : Point)
variables (α β : ℝ)
noncomputable def centroid (A B C : Point) : Point := 
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

axiom fixed_base : ∀ (A B : Point), ∃ l : Line, l = (A, B)
axiom move_C_linear : ∃ (C : Point → ℝ → Point), ∀ t, C (0, t) = (t / (√2 : ℝ), t / (√2 : ℝ))
axiom vertex_movement : ∀ (C : Point), C = (α / (√2 : ℝ), α / (√2 : ℝ))

theorem centroid_trace : ∀ (A B : Point) (α : ℝ), fixed_base A B →
  move_C_linear C α → 
  vertex_movement C →
  centroid A B C = centroid A B (α / (√2 : ℝ), α / (√2 : ℝ)) :=
sorry

end centroid_trace_l526_526041


namespace sum_of_possible_k_l526_526440

theorem sum_of_possible_k (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  j > 0 ∧ k > 0 → 
  ∑ i in {20, 12, 8, 6, 5}.to_finset, i = 51 :=
by
  sorry

end sum_of_possible_k_l526_526440


namespace probability_of_10_heads_in_12_flips_l526_526597

open_locale big_operators

noncomputable def calculate_probability : ℕ → ℕ → ℚ := 
  λ n k, (nat.choose n k : ℚ) / (2 ^ n)

theorem probability_of_10_heads_in_12_flips :
  calculate_probability 12 10 = 66 / 4096 :=
by
  sorry

end probability_of_10_heads_in_12_flips_l526_526597


namespace inequality_proof_l526_526853

-- Define constants
def a : ℝ := 0.1 * Real.exp 0.1
def b : ℝ := 1 / 9
def c : ℝ := -Real.log 0.9

-- Assert the inequalities
theorem inequality_proof : c < a ∧ a < b :=
by sorry

end inequality_proof_l526_526853


namespace option_A_is_not_odd_l526_526269

def f_A (a x : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ x ≠ 0) : ℝ := (a^x + 1) * x / (a^x - 1)
def f_B (a x : ℝ) (h : a > 0 ∧ a ≠ 1) : ℝ := (a^x - a^(-x)) / 2
def f_C (x : ℝ) : ℝ := if x > 0 then 1 else -1
def f_D (a x : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ abs x < 1) : ℝ := Real.log (1 + x) / Real.log (1 - x)

theorem option_A_is_not_odd (a x : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ x ≠ 0) :
  ¬ (∀ x, f_A a x h = - f_A a (-x) h) := by
  sorry

end option_A_is_not_odd_l526_526269


namespace no_real_solution_eq_l526_526774

theorem no_real_solution_eq (x : ℝ) : 
  ¬ (∃ x : ℝ, (real.sqrt (real.sqrt x) = 20 / (9 - 2 * real.sqrt (real.sqrt x)))) :=
begin
  sorry -- Proof not required
end

end no_real_solution_eq_l526_526774


namespace binomial_constant_term_l526_526785

theorem binomial_constant_term : 
  ∃ (c : ℚ), (x : ℝ) → (x^2 + (1 / (2 * x)))^6 = c ∧ c = 15 / 16 := by
  sorry

end binomial_constant_term_l526_526785


namespace incorrect_statement_l526_526870

variable (p q : Prop)
variable (hp: p = (2 + 2 = 5))
variable (hq: q = (3 > 2))
-- Definitions reflecting the problem conditions
def condA : Prop := (p ∨ q) ∧ ¬q = false
def condB : Prop := (p ∧ q = false) ∧ ¬p = true
def condC : Prop := (p ∧ q = false) ∧ ¬p = false
def condD : Prop := (p ∧ q = false) ∧ (p ∨ q) = true

-- The statement we need to prove,
-- that condition C is incorrect given p and q
theorem incorrect_statement : condC hp hq = false :=
by
  sorry

end incorrect_statement_l526_526870


namespace calculate_sum_and_difference_l526_526286

theorem calculate_sum_and_difference : 0.5 - 0.03 + 0.007 = 0.477 := sorry

end calculate_sum_and_difference_l526_526286


namespace find_x_l526_526935

theorem find_x (x : ℝ) (h : 2^(x - 3) = 4^3) : x = 9 :=
by sorry

end find_x_l526_526935


namespace find_c_to_make_f_odd_l526_526784

def f (x c : Real) : Real := Real.arctan ((2 - 2 * x) / (1 + 4 * x)) + c

theorem find_c_to_make_f_odd : 
  ∃ c : Real, (∀ x : Real, x ∈ Ioo (-1/4) (1/4) → f x c = - f (-x) c) :=
sorry

end find_c_to_make_f_odd_l526_526784


namespace no_equilateral_triangle_subset_size_l526_526872

noncomputable def no_equilateral_triangle_subset (points : FinSet (ℝ × ℝ)) : FinSet (ℝ × ℝ) :=
sorry -- Function to find the subset. Implementation is omitted.

theorem no_equilateral_triangle_subset_size (points : FinSet (ℝ × ℝ)) (h : points.card = n) : 
  ∃ S : FinSet (ℝ × ℝ), S ⊆ points ∧ S.card ≥ ⌈Real.sqrt n⌉₊ ∧ ∀ P1 P2 P3 ∈ S, ¬is_equilateral_triangle P1 P2 P3 :=
begin
  sorry -- Proof is omitted.
end

end no_equilateral_triangle_subset_size_l526_526872


namespace compare_abc_l526_526826
open Real

noncomputable def a : ℝ := 0.1 * exp(0.1)
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - log 0.9

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l526_526826


namespace find_k_solution_l526_526489

variables (a b c k : ℝ)
variables (hc : c ≠ 0) (hb : b ≠ 0) (ha : a ≠ 0)

def f (x : ℝ) : ℝ := 1 / (a * x^2 + b * x + c)

theorem find_k_solution : 
  (∃ x : ℝ, f x = k) ↔ a * k^3 + b * k^2 + c * k - 1 = 0 := 
sorry

end find_k_solution_l526_526489


namespace gold_bars_total_worth_l526_526524

theorem gold_bars_total_worth :
  let rows := 4
  let bars_per_row := 20
  let worth_per_bar : ℕ := 20000
  let total_bars := rows * bars_per_row
  let total_worth := total_bars * worth_per_bar
  total_worth = 1600000 :=
by
  sorry

end gold_bars_total_worth_l526_526524


namespace initial_population_l526_526569

theorem initial_population (rate_decrease : ℝ) (population_after_2_years : ℝ) (P : ℝ) : 
  rate_decrease = 0.1 → 
  population_after_2_years = 8100 → 
  ((1 - rate_decrease) ^ 2) * P = population_after_2_years → 
  P = 10000 :=
by
  intros h1 h2 h3
  sorry

end initial_population_l526_526569


namespace train_speed_l526_526212

-- Defining the lengths and time
def length_train : ℕ := 100
def length_bridge : ℕ := 300
def time_crossing : ℕ := 15

-- Defining the total distance
def total_distance : ℕ := length_train + length_bridge

-- Proving the speed of the train
theorem train_speed : (total_distance / time_crossing : ℚ) = 26.67 := by
  sorry

end train_speed_l526_526212


namespace range_of_m_for_ellipse_and_hyperbola_l526_526901

theorem range_of_m_for_ellipse_and_hyperbola (m : ℝ) :
  (0 < m ∧ m < 6 ∧ m ≠ 5) ↔ (m ∈ set.Ioo 0 5 ∪ set.Ioo 5 6) :=
by {
  sorry
}

end range_of_m_for_ellipse_and_hyperbola_l526_526901


namespace fraction_difference_l526_526692

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l526_526692


namespace greatest_product_of_slopes_l526_526181

theorem greatest_product_of_slopes :
  ∃ m1 m2 : ℝ, m2 = 3 * m1 ∧ abs ((m2 - m1) / (1 + m1 * m2)) = 1 / √3 ∧
  m1 * m2 = 1 :=
sorry

end greatest_product_of_slopes_l526_526181


namespace seating_arrangements_count_l526_526643

-- Define the problem scenario
def people : Type := fin 8

-- Special positions: leader (l), vice leader (v), recorder (r)
structure SpecialPositions (l v r : people) :=
(seats : people → option people)
(between : ∀ (x y z : people), x ≠ y → y ≠ z → x ≠ z → (seats x = some y ∧ seats z = some y ∧ (y = l ∨ y = v → seats y = some r)) )

-- Define the main problem
def seating_arrangements (l v r : people) [SpecialPositions l v r] : Nat :=
sorry -- To be filled with the number of such arrangements

-- Statement to prove the number of seats is 240 given the conditions
theorem seating_arrangements_count (l v r : people) [SpecialPositions l v r] :
  seating_arrangements l v r = 240 :=
sorry

end seating_arrangements_count_l526_526643


namespace largest_sum_of_products_is_238_l526_526574

noncomputable def largest_possible_value (S : Finset ℕ) (f g h j k : ℕ) : ℕ :=
  f * g + g * h + h * j + j * k + k * f

theorem largest_sum_of_products_is_238 :
  ∀ f g h j k : ℕ,
    {f, g, h, j, k}.val = [5, 6, 7, 8, 9] →
    largest_possible_value ({5, 6, 7, 8, 9} : Finset ℕ) f g h j k = 238 :=
begin
  intros f g h j k h1,
  sorry
end

end largest_sum_of_products_is_238_l526_526574


namespace probability_of_painting_different_color_more_than_half_l526_526175

-- Define the total number of balls, possible colors, and equal probability
def total_balls : Nat := 8
def possible_colors : List String := ["red", "black", "white"]
def probability_each_color : ℚ := 1 / 3

-- Define the event's probability as specified in the problem
def favorable_probability : ℚ := 1680 / 6561

theorem probability_of_painting_different_color_more_than_half :
  ∀ (ball : Fin total_balls),
    (1 / (List.length possible_colors) ^ total_balls) * 
    (Finset.card (Finset.filter (λ conf, (Conf.card (λ c, c = "red") conf ≤ 3) ∧ 
    (Conf.card (λ c, c = "black") conf ≤ 3) ∧ (Conf.card (λ c, c = "white") conf ≤ 2)) 
    (Finset.univ : Finset (Vector String total_balls))) : ℚ) = favorable_probability := sorry

end probability_of_painting_different_color_more_than_half_l526_526175


namespace inverse_function_log_l526_526024

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x

theorem inverse_function_log (a : ℝ) (g : ℝ → ℝ) (x : ℝ) (y : ℝ) :
  (a > 0) → (a ≠ 1) → 
  (f 2 a = 4) → 
  (f y a = x) → 
  (g x = y) → 
  g x = Real.logb 2 x := 
by
  intros ha hn hfx hfy hg
  sorry

end inverse_function_log_l526_526024


namespace range_of_a_l526_526015

noncomputable def quadratic_inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  x^2 - 2 * (a - 2) * x + a > 0

theorem range_of_a :
  (∀ x : ℝ, (x < 1 ∨ x > 5) → quadratic_inequality_condition a x) ↔ (1 < a ∧ a ≤ 5) :=
by
  sorry

end range_of_a_l526_526015


namespace min_x2_y2_l526_526404

theorem min_x2_y2 (x y : ℝ) (h : x * y - x - y = 1) : x^2 + y^2 ≥ 6 - 4 * Real.sqrt 2 :=
by
  sorry

end min_x2_y2_l526_526404


namespace roses_cut_from_garden_l526_526177

-- Define the variables and conditions
variables {x : ℕ} -- x is the number of freshly cut roses

def initial_roses : ℕ := 17
def roses_thrown_away : ℕ := 8
def roses_final_vase : ℕ := 42
def roses_given_away : ℕ := 6

-- The condition that describes the total roses now
def condition (x : ℕ) : Prop :=
  initial_roses - roses_thrown_away + (1/3 : ℚ) * x = roses_final_vase

-- The verification step that checks the total roses concerning given away roses
def verification (x : ℕ) : Prop :=
  (1/3 : ℚ) * x + roses_given_away = roses_final_vase + roses_given_away

-- The main theorem to prove the number of roses cut
theorem roses_cut_from_garden (x : ℕ) (h1 : condition x) (h2 : verification x) : x = 99 :=
  sorry

end roses_cut_from_garden_l526_526177


namespace sum_possible_k_l526_526459

theorem sum_possible_k (j k : ℕ) (hcond : 1 / j + 1 / k = 1 / 4) (hj : 0 < j) (hk : 0 < k) :
  j ≠ k → ∑ k in {k | ∃ j, 1 / j + 1 / k = 1 / 4}, k = 51 :=
by
  sorry

end sum_possible_k_l526_526459


namespace f_f_neg2016_equals_zero_l526_526029

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then f (x + 2)
  else if -1 < x ∧ x < 1 then 2 * x + 2
  else 2^ x - 4

theorem f_f_neg2016_equals_zero : f (f (-2016)) = 0 := 
  by 
    -- The proof is omitted.
    sorry

end f_f_neg2016_equals_zero_l526_526029


namespace equal_real_roots_possible_values_l526_526407

theorem equal_real_roots_possible_values (a : ℝ): 
  (∀ x : ℝ, x^2 + a * x + 1 = 0) → (a = 2 ∨ a = -2) :=
by
  sorry

end equal_real_roots_possible_values_l526_526407


namespace minimum_instantaneous_rate_of_change_l526_526668

noncomputable def f (x : ℝ) := (1/3) * x^3 - x^2 + 8

theorem minimum_instantaneous_rate_of_change :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 5 ∧ (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 5 → f'(x) ≤ f'(y)) ∧ f'(x) = -1 := 
by
  sorry

end minimum_instantaneous_rate_of_change_l526_526668


namespace triangle_acute_l526_526937

theorem triangle_acute
  (A B C : ℝ)
  (h_sum : A + B + C = 180)
  (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 4) :
  A < 90 ∧ B < 90 ∧ C < 90 :=
by
  -- proof goes here
  sorry

end triangle_acute_l526_526937


namespace ratio_of_areas_l526_526107

-- Define the sets T and S
def T : Set (ℝ × ℝ × ℝ) := { t | 0 ≤ t.1 ∧ 0 ≤ t.2 ∧ 0 ≤ t.3 ∧ t.1 + t.2 + t.3 = 1}
def supports (a b c : ℝ) (t : ℝ × ℝ × ℝ) : Prop :=
  (t.1 ≥ a ∧ t.2 ≥ b) ∨ (t.1 ≥ a ∧ t.3 ≥ c) ∨ (t.2 ≥ b ∧ t.3 ≥ c)

def S : Set (ℝ × ℝ × ℝ) := {t ∈ T | supports (1/4) (1/4) (1/4) t}

-- Prove the required ratio of areas
theorem ratio_of_areas : (area S) / (area T) = 9 / 32 := 
  sorry

end ratio_of_areas_l526_526107


namespace solution_set_gx_inequality_l526_526878

theorem solution_set_gx_inequality
  (f : ℝ → ℝ)
  (hf_even : ∀ x, f (-x) = f x)
  (hf_derivative : ∀ x, deriv f x)
  (cond : ∀ x, x ≥ 0 → (x / 2) * (deriv f x) + f (-x) ≤ 0) :
  {x : ℝ | x^2 * f x < (1 - 2*x)^2 * f (1 - 2*x)} = {x : ℝ | 1/3 < x ∧ x < 1} :=
by
  sorry

end solution_set_gx_inequality_l526_526878


namespace problem_aq_am_eq_ac_ae_l526_526750

theorem problem_aq_am_eq_ac_ae
  (O : Type*) [metric_space O] [nonempty O]
  (A B M C E Q : O)
  (h_AB_diameter : dist A B = 2 * dist A O)
  (h_AM_tangent : ∀ (X : O), X ≠ A → dist A M = dist A X → dist X M^2 = dist X A^2 + dist A M^2)
  (h_CE_intersect_AM_at_Q : dist C Q + dist Q E = dist C E) :
  dist A Q * dist A M = dist A C * dist A E :=
by sorry

end problem_aq_am_eq_ac_ae_l526_526750


namespace sum_of_possible_ks_l526_526427

theorem sum_of_possible_ks : 
  (∃ (j k : ℕ), (1 < j) ∧ (1 < k) ∧ j ≠ k ∧ ((1/j : ℝ) + (1/k : ℝ) = (1/4))) → 
  (∑ k in {20, 12, 8, 6, 5}, k) = 51 :=
begin
  sorry
end

end sum_of_possible_ks_l526_526427


namespace sum_max_min_expr_l526_526929

theorem sum_max_min_expr (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) : 
    let expr := (x / |x|) + (|y| / y) - (|x * y| / (x * y))
    max (max expr (expr)) (min expr expr) = -2 :=
sorry

end sum_max_min_expr_l526_526929


namespace desired_ellipse_properties_l526_526321

def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (y^2)/(a^2) + (x^2)/(b^2) = 1

def ellipse_has_foci (a b : ℝ) (c : ℝ) : Prop :=
  c^2 = a^2 - b^2

def desired_ellipse_passes_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  is_ellipse a b P.1 P.2

def foci_of_ellipse (a b : ℝ) (c : ℝ) : Prop :=
  ellipse_has_foci a b c

axiom given_ellipse_foci : foci_of_ellipse 3 2 (Real.sqrt 5)

theorem desired_ellipse_properties :
  desired_ellipse_passes_through_point 4 (Real.sqrt 11) (0, 4) ∧
  foci_of_ellipse 4 (Real.sqrt 11) (Real.sqrt 5) :=
by
  sorry

end desired_ellipse_properties_l526_526321


namespace money_left_l526_526206

theorem money_left (total money_spent leftover_spent: ℕ) (h₁ : total = 32) (h₂ : money_spent = 5) (h₃ : leftover_spent = 1/3 * (total - money_spent)) :
    total - money_spent - (1/3 * (total - money_spent)) = 18 :=
by
  have h_leftover: total - money_spent = 27 := by
    rw [h₁, h₂]
    norm_num
  have h_turkey: leftover_spent = 9 := by
    rw [h₃, h_leftover]
    norm_num
  rw [h_leftover, h_turkey]
  norm_num
  sorry

end money_left_l526_526206


namespace evaluate_expression_l526_526765

theorem evaluate_expression : 101^3 + 3 * 101^2 * 2 + 3 * 101 * 2^2 + 2^3 = 1092727 := by
  sorry

end evaluate_expression_l526_526765


namespace vision_data_approximation_l526_526541

theorem vision_data_approximation :
  ∀ (L V : ℝ), L = 4.9 → (L = 5 + log10 V) → (0.8 ≤ V ∧ V ≤ 0.8 * 1.0001) := 
by
  intros L V hL hRel
  have hlog : log10 V = -0.1 := 
    by rw [←hRel, hL]; ring
  have hV : V = 10 ^ (-0.1) :=
    by rw [←hlog]; exact (real.rpow10_log10 V)
  rw hV
  have hApprox : 10 ^ (-0.1) ≈ 0.8 := sorry -- This is the approximation step
  exact hApprox

end vision_data_approximation_l526_526541


namespace company_team_selection_l526_526952

theorem company_team_selection :
  ∃ (teams : ℕ), teams = 600 ∧ 
    (let phd := 5 in let ms := 6 in let bs := 4 in
    nat.choose phd 2 * nat.choose ms 2 * nat.choose bs 1 = teams) :=
by
  let phd := 5
  let ms := 6
  let bs := 4
  existsi (nat.choose phd 2 * nat.choose ms 2 * nat.choose bs 1)
  split
  · sorry -- proof of teams = 600
  · rfl

end company_team_selection_l526_526952


namespace satisfies_derivative_l526_526161

-- Definitions for the given functions
def f1 : ℝ → ℝ := λ x, 1 - x
def f2 : ℝ → ℝ := λ x, x
def f3 : ℝ → ℝ := λ x, 0
def f4 : ℝ → ℝ := λ x, 1

-- The theorem to prove
theorem satisfies_derivative :
  (f3 = (λ x : ℝ, deriv f3 x)) ∧
  ¬ (f1 = (λ x : ℝ, deriv f1 x)) ∧
  ¬ (f2 = (λ x : ℝ, deriv f2 x)) ∧
  ¬ (f4 = (λ x : ℝ, deriv f4 x)) :=
sorry

end satisfies_derivative_l526_526161


namespace OC_magnitude_and_direction_l526_526087

open Real
open EuclideanGeometry
open Point2D

-- Set up the points and the bisector condition
def A : Point2D := (0, 1)
def B : Point2D := (-3, 4)
def C : Point2D := sorry /- Point on / (OA angle bisector with magnitude 2) -/

-- Statement of the problem
theorem OC_magnitude_and_direction :
  ∃ C : Point2D, 
    (C ∈ bisector (A, (0, 0)) B) ∧ 
    distance (0, 0) C = 2
→ C = (-√10/5, 3√10/5) :=
begin
  sorry
end

end OC_magnitude_and_direction_l526_526087


namespace repeating_decimal_exceeds_finite_decimal_by_l526_526729

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l526_526729


namespace compare_values_l526_526804

variable (a b c : ℝ)

noncomputable def a_def : ℝ := 0.1 * Real.exp 0.1
noncomputable def b_def : ℝ := 1 / 9
noncomputable def c_def : ℝ := - Real.log 0.9

theorem compare_values : 
  c < a ∧ a < b := 
begin
  let a := 0.1 * Real.exp 0.1,
  let b := 1 / 9,
  let c := - Real.log 0.9,
  have h1 : c < a, sorry,
  have h2 : a < b, sorry,
  exact ⟨h1, h2⟩
end

end compare_values_l526_526804


namespace quadrant_of_point_C_l526_526055

theorem quadrant_of_point_C
  (a b : ℝ)
  (h1 : -(a-2) = -1)
  (h2 : b+5 = 3) :
  a = 3 ∧ b = -2 ∧ 0 < a ∧ b < 0 :=
by {
  sorry
}

end quadrant_of_point_C_l526_526055


namespace student_vision_data_correct_l526_526550

noncomputable def student_vision_decimal_approx : ℝ :=
  let L := 4.9
  let V := 10^(-(L - 5))
  approx := 0.8
  sqrt10 := 10^(1/10)
  1 / sqrt10

theorem student_vision_data_correct :
  (L = 4.9) →
  (L = 5 + log10 V) →
  (sqrt10 ≈ 1.259) →
  (V ≈ 0.8) :=
  by
    sorry

end student_vision_data_correct_l526_526550


namespace compare_abc_l526_526820
open Real

noncomputable def a : ℝ := 0.1 * exp(0.1)
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - log 0.9

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l526_526820


namespace excess_common_fraction_l526_526712

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l526_526712


namespace value_of_f_neg_4_l526_526903

theorem value_of_f_neg_4 {f : ℝ → ℝ} (h1 : ∀ x, f(2 - x) = 2 - f(x + 2)) (h2 : function.injective f) (h3 : f 8 = 4) : f (-4) = -2 := 
sorry

end value_of_f_neg_4_l526_526903


namespace greatest_possible_n_l526_526053

theorem greatest_possible_n (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 :=
begin
  sorry
end

end greatest_possible_n_l526_526053


namespace compare_a_b_c_l526_526851

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l526_526851


namespace part_a_part_b_l526_526114

noncomputable def f (x : ℝ) : ℝ := ∑ n, x^n

noncomputable def g (x : ℝ) : ℝ :=
  ∑ n, (-1)^n * x^n

theorem part_a (x : ℝ) :
  (f x) * (f x) = ∑ n, (n + 1) * x^n :=
sorry

theorem part_b (x : ℝ) :
  (f x) * (g x) = ∑ n, if (n % 2 = 0) then x^n else 0 :=
sorry

end part_a_part_b_l526_526114


namespace temperature_on_April_15_and_19_l526_526626

/-
We define the daily temperatures as functions of the temperature on April 15 (T_15) with the given increment of 1.5 degrees each day. 
T_15 represents the temperature on April 15.
-/
theorem temperature_on_April_15_and_19 (T : ℕ → ℝ) (T_avg : ℝ) (inc : ℝ) 
  (h1 : inc = 1.5)
  (h2 : T_avg = 17.5)
  (h3 : ∀ n, T (15 + n) = T 15 + inc * n)
  (h4 : (T 15 + T 16 + T 17 + T 18 + T 19) / 5 = T_avg) :
  T 15 = 14.5 ∧ T 19 = 20.5 :=
by
  sorry

end temperature_on_April_15_and_19_l526_526626


namespace yogurt_combination_count_l526_526673

theorem yogurt_combination_count :
  let yogurt_choices := 2
  let flavor_choices := 5
  let topping_choices := choose 8 2
  in yogurt_choices * flavor_choices * topping_choices = 280 :=
by
  let yogurt_choices := 2
  let flavor_choices := 5
  let topping_choices := Nat.choose 8 2
  calc
    yogurt_choices * flavor_choices * topping_choices
      = 2 * 5 * 28 : by rw [Nat.choose_eq 8 2]
      = 280 : by norm_num

end yogurt_combination_count_l526_526673


namespace find_a_l526_526058

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

theorem find_a (a : ℝ) (extremum_condition : ∀ x, deriv (f a) x = 0 → x = 1) : a = Real.exp 1 :=
by
  let f' := λ x, (Real.exp x - a)
  have derivative_at_1 : deriv (f a) 1 = f' 1 := sorry -- Proof that the derivative matches f'
  have zero_slope_at_1 : deriv (f a) 1 = 0 := extremum_condition 1 (by simp [derivative_at_1])
  rw [derivative_at_1, Real.exp_one] at zero_slope_at_1
  exact (eq_of_sub_eq_zero zero_slope_at_1.symm)

end find_a_l526_526058


namespace curve_parametric_to_general_l526_526654

theorem curve_parametric_to_general :
  ∃ (a b c : ℚ),
  (∀ t : ℝ, let x := 3 * Real.cos t - Real.sin t
             let y := 5 * Real.sin t
             in a * x^2 + b * x * y + c * y^2 = 1) ∧ 
  a = 1/9 ∧ 
  b = 2/45 ∧ 
  c = 4/45 := 
sorry

end curve_parametric_to_general_l526_526654


namespace repeatingDecimal_exceeds_l526_526699

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l526_526699


namespace length_CD_of_inscribed_square_CD_length_is_twice_decagon_side_l526_526414

theorem length_CD_of_inscribed_square
  (circle : Type) [metric_space circle] [normed_group circle]
  (O : circle) (r : ℝ) (A B C D : circle)
  (inscribed_square : metric_ball O r) (segment_AB : ℝ) (ext_AB_to_C : ℝ) :
  (AB B C) = inscribed_square.side_length (B' O){
  sorry,
}
  (BC = AB') (OC.intersect_circle_at_D)→ 
  length_segment_CD (lemmas_point_A2 C -

  sorry
          
theorem CD_length_is_twice_decagon_side
  (AB : circle) [metric_space AB] [normed_group AB]
  (O : circle) [metric_space O] [normed_group O]
  (r : ℝ)
  (C : circle) [metric_space C] [normed_group C]
  (D : circle) [metric_space D] [normed_group D]
  (AB_is_square : AB.side_length (O)) 
  (BC_extension_to_segment_C_from_point_B: BC.eq (Bc) ) : 

  length_eq_CD_twice_side_length_decagon()
  x AB)

length_CD (segment_DC(ext BC :AB(A) O)) : 
  AB'(side_length  :A) = inscribed_square
 
end length_CD_of_inscribed_square_CD_length_is_twice_decagon_side_l526_526414


namespace proposition_2_correct_proposition_3_correct_proposition_1_incorrect_proposition_4_incorrect_l526_526110

-- Define variables for lines and planes
variables {m n : Line} {α β : Plane}

-- Define conditions and propositions
structure conditions :=
(m_ne_n : m ≠ n)
(alpha_ne_beta : α ≠ β)

def proposition_1 (h1 : m ⊆ β) (h2 : ⊥ α β) : ⊥ m α :=
sorry

def proposition_2 (h1 : α ∥ β) (h2 : m ⊆ α) : m ∥ β :=
sorry

def proposition_3 (h1 : ⊥ n α) (h2 : ⊥ n β) (h3 : ⊥ m α) : ⊥ m β :=
sorry

def proposition_4 (h1 : m ∥ α) (h2 : m ∥ β) : α ∥ β :=
sorry

-- Define the theorems corresponding to propositions 
theorem proposition_2_correct (h1 : α ∥ β) (h2 : m ⊆ α) : m ∥ β :=
begin
  -- The proof will go here
  sorry
end

theorem proposition_3_correct (h1 : ⊥ n α) (h2 : ⊥ n β) (h3 : ⊥ m α) : ⊥ m β :=
begin
  -- The proof will go here
  sorry
end

theorem proposition_1_incorrect : ¬ proposition_1 :=
begin
  -- The proof will go here demonstrating that proposition 1 is incorrect
  sorry
end

theorem proposition_4_incorrect : ¬ proposition_4 :=
begin
  -- The proof will go here demonstrating that proposition 4 is incorrect
  sorry
end

-- Assertions
example (hc : conditions) : 
  proposition_2_correct α β m ∧ proposition_3_correct α β m ∧ 
  proposition_1_incorrect ∧ proposition_4_incorrect :=
begin
  sorry
end

end proposition_2_correct_proposition_3_correct_proposition_1_incorrect_proposition_4_incorrect_l526_526110


namespace coefficient_x5_in_expansion_l526_526186

theorem coefficient_x5_in_expansion :
  (∃ c : ℤ, c = binomial 9 4 * (3 * real.sqrt 2) ^ 4 ∧ c = 40824) → true := 
by
  intro h,
  have : 40824 = 40824, from rfl,
  exact h sorry

end coefficient_x5_in_expansion_l526_526186


namespace exists_six_clique_l526_526243

noncomputable def knows (delegate : Type) := delegate → delegate → Prop

theorem exists_six_clique (delegates : Type) [Fintype delegates] (h : Fintype.card delegates = 500)
  (knows_each_other : ∀ (d1 d2 : delegates), knows delegates d1 d2 → knows delegates d2 d1)
  (deg : ∀ (d : delegates), Fintype.card { d' : delegates // knows delegates d d' } > 400) :
  ∃ (clique : Finset delegates), clique.card = 6 ∧ ∀ {d1 d2 : delegates}, d1 ∈ clique → d2 ∈ clique → knows delegates d1 d2 :=
sorry

end exists_six_clique_l526_526243


namespace evaluate_expression_l526_526312

theorem evaluate_expression : [3 - 4 * (3 - 5)⁻¹]⁻¹ = 1 / 5 := by
  sorry

end evaluate_expression_l526_526312


namespace quadratic_square_binomial_l526_526773

theorem quadratic_square_binomial (a r s : ℝ) :
  (ax^2 + 16x + 16 = (rx + s)^2) → a = 4 :=
by
  sorry -- proof to be filled in

end quadratic_square_binomial_l526_526773


namespace n_cubed_minus_n_plus_one_is_square_l526_526329

theorem n_cubed_minus_n_plus_one_is_square (n : ℕ) (h : (n^5 + n^4 + 1).divisors.card = 6) : ∃ k : ℕ, n^3 - n + 1 = k^2 :=
sorry

end n_cubed_minus_n_plus_one_is_square_l526_526329


namespace fraction_difference_l526_526688

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l526_526688


namespace find_ellipse_l526_526911

open Real

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  (x^2) / 4 + (y^2) / 2 = 1

theorem find_ellipse :
  (∃ a b : ℝ, 
    (a > b) ∧ (b > 0) ∧ 
    (∀ x y : ℝ, (x^2) / a^2 + (y^2) / b^2 = 1) ∧
    (∀ x : ℝ, y = 2 * sqrt 2 * x + 2 * sqrt 3) ∧
    ∠AOB = 90 ∧ 
    AB = 12 * sqrt 11 / 17
  ) →
  (∀ x y : ℝ, ellipse_eq x y) :=
by
  sorry

end find_ellipse_l526_526911


namespace sum_of_possible_k_l526_526439

theorem sum_of_possible_k (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  j > 0 ∧ k > 0 → 
  ∑ i in {20, 12, 8, 6, 5}.to_finset, i = 51 :=
by
  sorry

end sum_of_possible_k_l526_526439


namespace op_five_two_is_twentyfour_l526_526566

def op (x y : Int) : Int :=
  (x + y + 1) * (x - y)

theorem op_five_two_is_twentyfour : op 5 2 = 24 := by
  unfold op
  sorry

end op_five_two_is_twentyfour_l526_526566


namespace solution_set_of_g_l526_526009

variable {R : Type*} [CommRing R]

-- Given definitions
def is_even (f : R → R) := ∀ x, f x = f (-x)
def g (f : R → R) (x : R) := x^2 * f x

-- Given conditions
variable (f : R → R)
variable h_even : is_even f
variable (y : R)
variable h_deriv : ∀ x > 0, x * y > -2 * f x

-- Statement to prove
theorem solution_set_of_g (x : R) :
  g f x < g f (1 - x) → x ∈ Iio (1 / 2) :=
sorry

end solution_set_of_g_l526_526009


namespace pentagon_area_correct_l526_526257

-- Define the side lengths and the values of r, s, e
variables (a b c d e r s : ℕ)
variables (sides : set ℕ)
variables (rectangle_area triangle_area pentagon_area : ℕ)

-- The conditions given in the problem
def conditions :=
  sides = {14, 21, 22, 28, 35} ∧
  (r = 20 ∧ s = 21 ∧ e = 29) ∧
  (r^2 + s^2 = e^2) ∧
  (rectangle_area = b * d) ∧
  (triangle_area = r * s / 2) ∧
  (pentagon_area = rectangle_area - triangle_area)

-- The theorem statement proving the area of the pentagon
theorem pentagon_area_correct (h : conditions) :
  pentagon_area = 770 :=
sorry

end pentagon_area_correct_l526_526257


namespace count_isosceles_triangles_l526_526086

-- Define some helper structures and assumptions
structure Point :=
  (x : ℕ) (y : ℕ)

def A : Point := ⟨2, 2⟩
def B : Point := ⟨4, 2⟩

-- Considering a standard 6x6 geoboard but avoiding points A and B.
def remaining_points : set Point :=
  {p : Point | p.x > 0 ∧ p.x < 6 ∧ p.y > 0 ∧ p.y < 6 ∧ p ≠ A ∧ p ≠ B}

-- Define distance function
def distance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

-- Define the is_isosceles predicate
def is_isosceles (A B C : Point) : Prop :=
  distance A B = distance A C ∨
  distance A B = distance B C ∨
  distance A C = distance B C

-- The theorem statement to be proven
theorem count_isosceles_triangles :
  (remaining_points.filter (λ C, is_isosceles A B C)).card = 6 :=
sorry

end count_isosceles_triangles_l526_526086


namespace distinct_elements_in_sets_l526_526483

theorem distinct_elements_in_sets (n : ℕ) 
  (S : Fin n → Finset α) 
  (h : ∀ i, (S i).Nonempty)
  (cond : ∑ i j in Finset.offDiag (Finset.range n), 
          (S i ∩ S j).card.toReal / ((S i).card.toReal * (S j).card.toReal) < 1) 
  : ∃ (x : Fin n → α), Function.Injective x ∧ ∀ i, x i ∈ S i := 
sorry

end distinct_elements_in_sets_l526_526483


namespace color_sum_zero_l526_526309

open Function

theorem color_sum_zero {color : ℤ → bool}
  (h_diff_colors : color 2016 ≠ color 2017) :
  ∃ x y z : ℤ, color x = color y ∧ color y = color z ∧ x + y + z = 0 :=
sorry

end color_sum_zero_l526_526309


namespace sum_of_possible_ks_l526_526433

theorem sum_of_possible_ks : 
  (∃ (j k : ℕ), (1 < j) ∧ (1 < k) ∧ j ≠ k ∧ ((1/j : ℝ) + (1/k : ℝ) = (1/4))) → 
  (∑ k in {20, 12, 8, 6, 5}, k) = 51 :=
begin
  sorry
end

end sum_of_possible_ks_l526_526433


namespace find_common_difference_l526_526794

variable {a : ℕ → ℝ} -- Define a sequence of real numbers

-- Conditions
def a1 := (a 1 = 1)
def sum_first_three := (a 1 + a 2 + a 3 = 12)
def is_arithmetic_seq := ∀ n : ℕ, (a (n + 1) = a n + d)

theorem find_common_difference (h1 : a1) (h2 : sum_first_three) : 
  ∃ d : ℝ, is_arithmetic_seq ∧ (a 2 = a 1 + d) ∧ (a 3 = (a 1 + d) + d) ∧ d = 3 :=
by
  sorry

end find_common_difference_l526_526794


namespace linear_combination_set1_not_linear_combination_set2_linear_combination_log_range_t_l526_526795

-- Definitions for Problem Part (I)
def f1_set1 (x : ℝ) : ℝ := Math.sin x
def f2_set1 (x : ℝ) : ℝ := Math.cos x
def h_set1 (x : ℝ) : ℝ := Math.sin (x + Real.pi / 3)

def f1_set2 (x : ℝ) : ℝ := x^2 - x
def f2_set2 (x : ℝ) : ℝ := x^2 + x + 1
def h_set2 (x : ℝ) : ℝ := x^2 - x + 1

-- Definitions for Problem Part (II)
def f1_log (x : ℝ) : ℝ := Real.log x / Real.log 2
def f2_log (x : ℝ) : ℝ := Real.log x / Real.log (1 / 2)
def h_log (x : ℝ) : ℝ := 2 * (Real.log x / Real.log 2) + Real.log x / Real.log (1 / 2)

-- Linear Combination Problems
theorem linear_combination_set1 :
  ∃ a b : ℝ, ∀ x : ℝ, h_set1 x = a * f1_set1 x + b * f2_set1 x :=
sorry

theorem not_linear_combination_set2 :
  ¬ (∃ a b : ℝ, ∀ x : ℝ, h_set2 x = a * f1_set2 x + b * f2_set2 x) :=
sorry

-- Linear Combination for given logs and range for t
theorem linear_combination_log :
  ∀ x : ℝ, h_log x = Real.log x / Real.log 2 :=
sorry

theorem range_t (t : ℝ) : (∃ x ∈ Icc 2 4, 3 * (Real.log x / Real.log 2)^2 + 2 * (Real.log x / Real.log 2) + t < 0) ↔ t < -5 :=
sorry

end linear_combination_set1_not_linear_combination_set2_linear_combination_log_range_t_l526_526795


namespace hyperbola_asymptote_l526_526907

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x y, 3 * x + 2 * y = 0 ∨ 3 * x - 2 * y = 0) →
  (∀ x y, y * y = 9 * (x * x / (a * a) - 1)) →
  a = 2 :=
by
  intros asymptote_constr hyp
  sorry

end hyperbola_asymptote_l526_526907


namespace find_balanced_grid_pairs_l526_526568

-- Define a balanced grid condition
def is_balanced_grid (m n : ℕ) (grid : ℕ → ℕ → Prop) : Prop :=
  ∀ i j, i < m → j < n →
    (∀ k, k < m → grid i k = grid i j) ∧ (∀ l, l < n → grid l j = grid i j)

-- Main theorem statement
theorem find_balanced_grid_pairs (m n : ℕ) :
  (∃ grid, is_balanced_grid m n grid) ↔ (m = n ∨ m = n / 2 ∨ n = 2 * m) :=
by
  sorry

end find_balanced_grid_pairs_l526_526568


namespace compare_a_b_c_l526_526848

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l526_526848


namespace vision_data_approximation_l526_526543

theorem vision_data_approximation :
  ∀ (L V : ℝ), L = 4.9 → (L = 5 + log10 V) → (0.8 ≤ V ∧ V ≤ 0.8 * 1.0001) := 
by
  intros L V hL hRel
  have hlog : log10 V = -0.1 := 
    by rw [←hRel, hL]; ring
  have hV : V = 10 ^ (-0.1) :=
    by rw [←hlog]; exact (real.rpow10_log10 V)
  rw hV
  have hApprox : 10 ^ (-0.1) ≈ 0.8 := sorry -- This is the approximation step
  exact hApprox

end vision_data_approximation_l526_526543


namespace minimum_water_amount_tension_period_l526_526649

noncomputable def water_amount (t : ℝ) : ℝ := 400 + 60 * t - 120 * real.sqrt(6 * t)

theorem minimum_water_amount : 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 24 ∧ (∀ t' : ℝ, 0 ≤ t' ∧ t' ≤ 24 → water_amount t ≤ water_amount t') ∧ t = 6 :=
sorry

theorem tension_period :
  ∃ t1 t2 : ℝ, 0 ≤ t1 ∧ t1 ≤ 24 ∧ 0 ≤ t2 ∧ t2 ≤ 24 ∧ t1 < t2 ∧ 
  (∀ t : ℝ, t1 ≤ t ∧ t ≤ t2 → water_amount t ≤ 80) ∧ (∀ t : ℝ, (0 ≤ t ∧ t < t1 ∨ t2 < t ∧ t ≤ 24) → 80 < water_amount t) ∧ t2 - t1 = 8 :=
sorry

end minimum_water_amount_tension_period_l526_526649


namespace probability_same_color_l526_526926

theorem probability_same_color :
  let total_ways := Nat.choose 13 3 in
  let red_ways := Nat.choose 7 3 in
  let blue_ways := Nat.choose 6 3 in
  let successful_outcomes := red_ways + blue_ways in
  (successful_outcomes : ℚ) / total_ways = 55 / 286 :=
by
  sorry

end probability_same_color_l526_526926


namespace olaf_number_of_men_on_boat_l526_526126

def olaf_sailing_problem :=
  (half_gallon_per_day_per_man : ℝ) (boat_speed : ℝ) (total_distance : ℝ) (total_water : ℝ) 
  (total_days : ℝ := total_distance / boat_speed)
  (water_per_man : ℝ := half_gallon_per_day_per_man * total_days)
  (number_of_men : ℝ := total_water / water_per_man)
  (half_gallon_per_day_per_man = 1 / 2 ∧ boat_speed = 200 ∧ total_distance = 4000 ∧ total_water = 250) →
  number_of_men = 25

theorem olaf_number_of_men_on_boat : olaf_sailing_problem := by
  sorry

end olaf_number_of_men_on_boat_l526_526126


namespace roast_cost_l526_526983

-- Given conditions as described in the problem.
def initial_money : ℝ := 100
def cost_vegetables : ℝ := 11
def money_left : ℝ := 72
def total_spent : ℝ := initial_money - money_left

-- The cost of the roast that we need to prove. We expect it to be €17.
def cost_roast : ℝ := total_spent - cost_vegetables

-- The theorem that states the cost of the roast given the conditions.
theorem roast_cost :
  cost_roast = 100 - 72 - 11 := by
  -- skipping the proof steps with sorry
  sorry

end roast_cost_l526_526983


namespace compare_constants_l526_526836

theorem compare_constants :
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  in c < a ∧ a < b :=
by
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  sorry

end compare_constants_l526_526836


namespace cubes_sum_modified_l526_526613

theorem cubes_sum_modified 
  (S₁ : ℕ → ℕ → ℕ)
  (S₂ : ℕ → ℕ → ℕ)
  (h₁ : S₁ 50 100 = ∑ x in finset.range (101 - 50), (50 + x)^3)
  (h₂ : S₂ 50 100 = ∑ x in finset.range (101 - 50), (-(50 + x)^3 + 1000)) :
  S₁ 50 100 + S₂ 50 100 = 51000 :=
by
  sorry

end cubes_sum_modified_l526_526613


namespace trapezium_area_proof_l526_526783

-- Define the lengths of the parallel sides and the distance between them
def a : ℝ := 20
def b : ℝ := 18
def h : ℝ := 15

-- Define the area of the trapezium
def area_of_trapezium (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

-- State the theorem to be proved
theorem trapezium_area_proof : area_of_trapezium a b h = 285 := by
  sorry

end trapezium_area_proof_l526_526783


namespace count_ordered_triples_l526_526924

open Nat

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b

theorem count_ordered_triples :
  {t : ℕ × ℕ × ℕ // lcm t.1 t.2.1 = 90 ∧ lcm t.1 t.2.2 = 450 ∧ lcm t.2.1 t.2.2 = 1350}.to_finset.card = 10 :=
by {
  sorry
}

end count_ordered_triples_l526_526924


namespace George_says_242_l526_526143

def skips_middle_number (n : ℕ) : Prop :=
  n % 3 = 2

def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def is_said_by (f : ℕ → Prop) (up_to : ℕ) : list ℕ :=
  (list.range up_to).filter (λ n, ¬ skips_middle_number n ∧ ¬ divisible_by_4 n ∧ f n)

noncomputable def George_number : ℕ :=
  let Alice_said := is_said_by (λ _, true) 900
  let Barbara_said := is_said_by (λ n, ¬ n ∈ Alice_said) 900
  let Candice_said := is_said_by (λ n, ¬ n ∈ Alice_said ∧ ¬ n ∈ Barbara_said) 900
  let Debbie_said := is_said_by (λ n, ¬ n ∈ Alice_said ∧ ¬ n ∈ Barbara_said ∧ ¬ n ∈ Candice_said) 900
  let Eliza_said := is_said_by (λ n, ¬ n ∈ Alice_said ∧ ¬ n ∈ Barbara_said ∧ ¬ n ∈ Candice_said ∧ ¬ n ∈ Debbie_said) 900
  let Fatima_said := is_said_by (λ n, ¬ n ∈ Alice_said ∧ ¬ n ∈ Barbara_said ∧ ¬ n ∈ Candice_said ∧ ¬ n ∈ Debbie_said ∧ ¬ n ∈ Eliza_said) 900
  (list.range 900).filter (λ n, ¬ skips_middle_number n ∧ ¬ divisible_by_4 n ∧ ¬ n ∈ Alice_said ∧ ¬ n ∈ Barbara_said ∧ ¬ n ∈ Candice_said ∧ ¬ n ∈ Debbie_said ∧ ¬ n ∈ Eliza_said ∧ ¬ n ∈ Fatima_said).head!

theorem George_says_242 : George_number = 242 := by
  sorry

end George_says_242_l526_526143


namespace find_extremum_l526_526056

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x

theorem find_extremum (a : ℝ) (h : Deriv (f x a) 1 = 0) : a = Real.exp 1 :=
by sorry

end find_extremum_l526_526056


namespace sum_of_possible_values_l526_526480

-- Define parameters
variables {O A B C : Type}
variables (a b c : ℝ)
variables (Area_OAB Area_OBC Area_OCA Area_ABC : ℝ)

-- Given conditions
def tetrahedron_conditions (O A B C : Type) (a b c : ℝ) (Area_OAB Area_OBC Area_OCA : ℝ) : Prop :=
  (∡ A O B = 90) ∧ (∡ B O C = 90) ∧ (∡ C O A = 90) ∧ (Area_OAB = 20) ∧ (Area_OBC = 14)

theorem sum_of_possible_values [O A B C : Type] (a b c : ℝ) (Area_OAB Area_OBC Area_OCA Area_ABC : ℝ) :
  tetrahedron_conditions O A B C a b c Area_OAB Area_OBC Area_OCA →
  ∃ Area_OCA Area_ABC, Area_OAB = 20 ∧ Area_OBC = 14 ∧ 
  Area_OCA * Area_ABC = 22200 :=
sorry

end sum_of_possible_values_l526_526480


namespace sum_possible_k_l526_526460

theorem sum_possible_k (j k : ℕ) (hcond : 1 / j + 1 / k = 1 / 4) (hj : 0 < j) (hk : 0 < k) :
  j ≠ k → ∑ k in {k | ∃ j, 1 / j + 1 / k = 1 / 4}, k = 51 :=
by
  sorry

end sum_possible_k_l526_526460


namespace compare_constants_l526_526861

noncomputable def a : ℝ := 0.1 * real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - real.log 0.9

theorem compare_constants : c < a ∧ a < b :=
by sorry

end compare_constants_l526_526861


namespace partition_sum_le_152_l526_526357

theorem partition_sum_le_152 {S : ℕ} (l : List ℕ) 
  (h1 : ∀ n ∈ l, 1 ≤ n ∧ n ≤ 10) 
  (h2 : l.sum = S) : 
  (∃ l1 l2 : List ℕ, l1.sum ≤ 80 ∧ l2.sum ≤ 80 ∧ l1 ++ l2 = l) ↔ S ≤ 152 := 
by
  sorry

end partition_sum_le_152_l526_526357


namespace boys_to_girls_ratio_l526_526950

theorem boys_to_girls_ratio (T G : ℕ) (h : (1 / 2) * G = (1 / 6) * T) : (T - G) = 2 * G := by
  sorry

end boys_to_girls_ratio_l526_526950


namespace coefficient_of_x5_in_expansion_l526_526188

theorem coefficient_of_x5_in_expansion :
  let a := (x : ℝ),
      b := (3 * Real.sqrt 2),
      n := 9,
      k := 4 in
  let binom := (Nat.choose n k) in
  let term := binom * a ^ (n - k) * b ^ k in
  term = 40824 * a ^ 5 := by
    sorry

end coefficient_of_x5_in_expansion_l526_526188


namespace arithmetic_geometric_inequality_l526_526640

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) :=
by
  sorry

end arithmetic_geometric_inequality_l526_526640


namespace shaded_region_area_l526_526650

def radius_smaller_circle := 4
def radius_larger_circle := 10

theorem shaded_region_area :
  let smaller_circle_area := Float.pi * radius_smaller_circle^2
  let larger_circle_area := Float.pi * radius_larger_circle^2
  larger_circle_area - smaller_circle_area = 84 * Float.pi :=
by
  let smaller_circle_area := Float.pi * radius_smaller_circle^2
  let larger_circle_area := Float.pi * radius_larger_circle^2
  have h1 : smaller_circle_area = 16 * Float.pi := by sorry
  have h2 : larger_circle_area = 100 * Float.pi := by sorry
  show larger_circle_area - smaller_circle_area = 84 * Float.pi from sorry

end shaded_region_area_l526_526650


namespace orthocenter_of_triangle_ABC_l526_526963

open Point

-- Definition of the points A, B, and C.
def A : Point ℝ := (2, 3, 4)
def B : Point ℝ := (6, 4, 2)
def C : Point ℝ := (4, 5, 6)

-- Definition of the orthocenter H of the triangle ABC.
def H : Point ℝ := (4, 3, 2)

-- Lean statement to prove that H is the orthocenter of triangle ABC.
theorem orthocenter_of_triangle_ABC : 
  orthocenter A B C = H :=
sorry

end orthocenter_of_triangle_ABC_l526_526963


namespace back_wheel_revolutions_calculation_l526_526508

noncomputable def front_diameter : ℝ := 3 -- Diameter of the front wheel in feet
noncomputable def back_diameter : ℝ := 0.5 -- Diameter of the back wheel in feet
noncomputable def no_slippage : Prop := true -- No slippage condition
noncomputable def front_revolutions : ℕ := 150 -- Number of front wheel revolutions

theorem back_wheel_revolutions_calculation 
  (d_f : ℝ) (d_b : ℝ) (slippage : Prop) (n_f : ℕ) : 
  slippage → d_f = front_diameter → d_b = back_diameter → 
  n_f = front_revolutions → 
  ∃ n_b : ℕ, n_b = 900 := 
by
  sorry

end back_wheel_revolutions_calculation_l526_526508


namespace trapezoid_perimeter_l526_526969

variables (ABCD : Type) [trapezoid ABCD]
variables (A B C D E : Point ABCD)
variables (AB BC CD : ℝ)
variables (h_iso : is_isosceles_trapezoid A B C D)
variables (h_AB : AB = CD)
variables (h_ratio : AB = BC / Real.sqrt 2)
variables (CE : ℝ)
variables (h_CE_height : is_height C E A D)
variables (h_BE : (B - E).dist = Real.sqrt 5)
variables (h_BD : (B - D).dist = Real.sqrt 10)

noncomputable def perimeter_trapezoid : ℝ :=
  (B - C).dist + (C - D).dist + (D - A).dist + (A - B).dist

theorem trapezoid_perimeter :
  perimeter_trapezoid ABCD = 6 + 2 * Real.sqrt 2 :=
sorry

end trapezoid_perimeter_l526_526969


namespace sum_of_roots_l526_526616

theorem sum_of_roots :
  let a := 1
  let b := 10
  let c := -25
  let sum_of_roots := -b / a
  (∀ x, 25 - 10 * x - x ^ 2 = 0 ↔ x ^ 2 + 10 * x - 25 = 0) →
  sum_of_roots = -10 :=
by
  intros
  sorry

end sum_of_roots_l526_526616


namespace find_y_coordinate_of_P_l526_526090

theorem find_y_coordinate_of_P (P Q : ℝ × ℝ)
  (h1 : ∀ x, y = 0.8 * x) -- line equation
  (h2 : P.1 = 4) -- x-coordinate of P
  (h3 : P = Q) -- P and Q are equidistant from the line
  : P.2 = 3.2 := sorry

end find_y_coordinate_of_P_l526_526090


namespace pq_perpendicular_o1o2_every_point_on_pq_same_power_radical_axes_intersect_or_parallel_l526_526632

variables {P Q O1 O2 : Point}
variables {k1 k2 k : Circle}
variables {r1 r2 : ℝ}

-- Assume the conditions
axiom non_intersecting_circles (h1 : ¬intersect k1 k2) :
  -- Given the powers of points P and Q w.r.t circles k1 and k2
  (power P k1 = power P k2) ∧ (power Q k1 = power Q k2) :=
  sorry

-- Show the proof statements
theorem pq_perpendicular_o1o2 :
  PQ ⊥ O1O2 :=
  sorry

theorem every_point_on_pq_same_power (S : Point) (hS : S ∈ PQ) :
  power S k1 = power S k2 :=
  sorry

theorem radical_axes_intersect_or_parallel (h_intersect_k : intersect k k1 ∧ intersect k k2) :
  intersect (radical_axis k k1) (radical_axis k k2) PQ ∨ parallel (radical_axis k k1) PQ :=
  sorry

end pq_perpendicular_o1o2_every_point_on_pq_same_power_radical_axes_intersect_or_parallel_l526_526632


namespace color_opposite_gold_is_yellow_l526_526244

-- Define the colors as a datatype for clarity
inductive Color
| B | Y | O | K | S | G

-- Define the type for each face's color
structure CubeFaces :=
(top front right back left bottom : Color)

-- Given conditions
def first_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.Y ∧ c.right = Color.O

def second_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.K ∧ c.right = Color.O

def third_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.S ∧ c.right = Color.O

-- Problem statement
theorem color_opposite_gold_is_yellow (c : CubeFaces) :
  first_view c → second_view c → third_view c → (c.back = Color.G) → (c.front = Color.Y) :=
by
  sorry

end color_opposite_gold_is_yellow_l526_526244


namespace find_f_value_l526_526912

-- Definitions of the function properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def satisfies_periodicity (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

def value_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x ∧ x < 1 → f x = 2^x

-- Theorem to be proven
theorem find_f_value (f : ℝ → ℝ)
  (odd_f : is_odd_function f)
  (periodicity_f : satisfies_periodicity f)
  (interval_f : value_in_interval f) :
  f (7 / 2) = -real.sqrt 2 := 
sorry

end find_f_value_l526_526912


namespace repeating_seventy_two_exceeds_seventy_two_l526_526739

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l526_526739


namespace length_of_real_axis_hyperbola_l526_526378

noncomputable def hyperbola_real_axis_length
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (eccentricity : ℝ) (he : eccentricity = sqrt 5)
  (distance_to_asymptote : ℝ) (hd : distance_to_asymptote = 8)
  (h_c : c = sqrt (a^2 + b^2))
  (h_distance : abs (b * c) / (sqrt (a^2 + b^2)) = distance_to_asymptote)
  : ℝ := 2 * a

theorem length_of_real_axis_hyperbola
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (eccentricity : ℝ) (he : eccentricity = sqrt 5)
  (distance_to_asymptote : ℝ) (hd : distance_to_asymptote = 8)
  (h_c : c = sqrt (a^2 + b^2))
  (h_distance : abs (b * c) / (sqrt (a^2 + b^2)) = distance_to_asymptote) :
  hyperbola_real_axis_length a b ha hb eccentricity he distance_to_asymptote hd h_c h_distance = 8 := 
sorry

end length_of_real_axis_hyperbola_l526_526378


namespace used_computer_lifespan_l526_526095

-- Problem statement
theorem used_computer_lifespan (cost_new : ℕ) (lifespan_new : ℕ) (cost_used : ℕ) (num_used : ℕ) (savings : ℕ) :
  cost_new = 600 →
  lifespan_new = 6 →
  cost_used = 200 →
  num_used = 2 →
  savings = 200 →
  ((cost_new - savings = num_used * cost_used) → (2 * (lifespan_new / 2) = 6) → lifespan_new / 2 = 3)
:= by
  intros
  sorry

end used_computer_lifespan_l526_526095


namespace fraction_difference_is_correct_l526_526719

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l526_526719


namespace sum_of_possible_ks_l526_526451

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l526_526451


namespace trapezoid_line_bisects_segment_l526_526798

-- Define the types for points and lines in projective geometry
variables {P : Type*} [projective_geometry P]

-- Define the points and their relationships
variables (A B C D X Y E M : P)
variables (lAB lCD lAC lBD lAD lBC lXY : set P)

-- Conditions based on the problem statement
axiom trapezoid (h1 : line lAB A B) (h2 : line lCD C D) (h3 : line lAC A C) (h4 : line lBD B D)
  (h1_parallel : parallel lAB lCD)
  (h_intersection_X : X ∈ lAC ∧ X ∈ lBD)
  (h_intersection_Y : Y ∈ lAD ∧ Y ∈ lBC)

-- Define the concurrency and bisection relation
axiom concurrency (h_E_intersection : E ∈ lAB ∧ E ∈ lCD)
axiom midside_bisects (h_XY_bisects_AB : midpoint M A B)

-- Translate the statement "Show that the line (XY) bisects the segment (AB)"
theorem trapezoid_line_bisects_segment :
  trapezoid (line A B) (line C D) (line A C) (line B D) ∧
  parallel (line A B) (line C D) ∧
  X ∈ (intersection (line A C) (line B D)) ∧
  Y ∈ (intersection (line A D) (line B C)) →
  midpoint M A B :=
by
  intros h1 h2 h3 h4 h1_parallel h_intersection_X h_intersection_Y
  sorry

end trapezoid_line_bisects_segment_l526_526798


namespace inequality_proof_l526_526857

-- Define constants
def a : ℝ := 0.1 * Real.exp 0.1
def b : ℝ := 1 / 9
def c : ℝ := -Real.log 0.9

-- Assert the inequalities
theorem inequality_proof : c < a ∧ a < b :=
by sorry

end inequality_proof_l526_526857


namespace sum_of_sequence_l526_526914

theorem sum_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → a (n + 1) * a n = 2 ^ n) →
  S 2016 = (∑ k in (finset.range 1008).filter (λ n, odd n), a (2 * k + 1))
         + (∑ k in (finset.range 1008), a (2 * (k + 1))) →
  S 2016 = 3 * 2 ^ 1008 - 3 :=
sorry

end sum_of_sequence_l526_526914


namespace mother_returns_to_freezer_l526_526683

noncomputable def probability_return_to_freezer : ℝ :=
  1 - ((5 / 17) * (4 / 16) * (3 / 15) * (2 / 14) * (1 / 13))

theorem mother_returns_to_freezer :
  abs (probability_return_to_freezer - 0.99979) < 0.00001 :=
by
    sorry

end mother_returns_to_freezer_l526_526683


namespace pupils_like_burgers_total_l526_526174

theorem pupils_like_burgers_total (total_pupils pizza_lovers both_lovers : ℕ) :
  total_pupils = 200 →
  pizza_lovers = 125 →
  both_lovers = 40 →
  (pizza_lovers - both_lovers) + (total_pupils - pizza_lovers - both_lovers) + both_lovers = 115 :=
by
  intros h_total h_pizza h_both
  rw [h_total, h_pizza, h_both]
  sorry

end pupils_like_burgers_total_l526_526174


namespace sum_of_x_satisfying_sqrt_eq_9_l526_526197

theorem sum_of_x_satisfying_sqrt_eq_9 :
  (∀ x : ℝ, sqrt ((x + 5)^2) = 9 → x = 4 ∨ x = -14) →
  (4 + (-14) = -10) :=
by
  intros h
  have h1 : sqrt ((4 + 5)^2) = 9 := by norm_num
  have h2 : sqrt ((-14 + 5)^2) = 9 := by norm_num
  apply h 4 h1
  apply h (-14) h2
  norm_num
  sorry

end sum_of_x_satisfying_sqrt_eq_9_l526_526197


namespace alternating_sum_equals_subsum_l526_526515

theorem alternating_sum_equals_subsum (n : ℕ) (hn : 0 < n) :
  (Finset.range (2 * n)).sum (λ i, (-1)^i * (1 / (i + 1))) = (Finset.Ico (n + 1) (2 * n + 1)).sum (λ j, 1 / j) :=
sorry

end alternating_sum_equals_subsum_l526_526515


namespace exists_number_divisible_by_d_times_2_pow_1996_l526_526796

noncomputable def is_composed_of_digits_1_and_2 (n : ℕ) : Prop :=
  ∀ d, d ∣ n → ∀ k, k ∈ List.digits 10 n → k = 1 ∨ k = 2

theorem exists_number_divisible_by_d_times_2_pow_1996 (d : ℕ) (h_d_pos : 0 < d) :
  (∃ n, is_composed_of_digits_1_and_2 n ∧ d * 2^1996 ∣ n) ↔ ¬ (5 ∣ d) :=
by
  sorry

end exists_number_divisible_by_d_times_2_pow_1996_l526_526796


namespace count_complex_solutions_l526_526387

theorem count_complex_solutions :
  let S := {z : ℂ | complex.abs z < 5 ∧ complex.exp z = (z - 1) / (z + 1)} in
  set.card S = 4 :=
sorry

end count_complex_solutions_l526_526387


namespace find_angle_A_find_tan_B_l526_526884

noncomputable theory

-- Problem 1: Proof of angle A

theorem find_angle_A 
  (A B C : ℝ)       -- The interior angles of the acute triangle
  (h_acuter: A + B + C = π)
  (h_acuteA: 0 < A ∧ A < π / 2)
  (h_acuteB: 0 < B ∧ B < π / 2)
  (h_acuteC: 0 < C ∧ C < π / 2)
  (a : ℝ)           -- The coefficient in the equation
  (h_root1: ∀ x : ℝ, x = sqrt 3 * sin A → x^2 - x + 2 * a = 0)
  (h_root2: ∀ x : ℝ, x = -cos A → x^2 - x + 2 * a = 0) :
  A = π / 3 :=
sorry

-- Problem 2: Proof of tan B

theorem find_tan_B 
  (B : ℝ)          -- The interior angle B
  (h_upper_b: 0 < B ∧ B < π / 2)
  (h_trig_eq: (1 + 2 * sin B * cos B) / (cos B^2 - sin B^2) = -3) :
  tan B = 2 :=
sorry

end find_angle_A_find_tan_B_l526_526884


namespace normal_price_of_article_l526_526612

theorem normal_price_of_article (P : ℝ) (sale_price : ℝ) (discount1 discount2 : ℝ) :
  discount1 = 0.10 → discount2 = 0.20 → sale_price = 108 →
  P * (1 - discount1) * (1 - discount2) = sale_price → P = 150 :=
by
  intro hd1 hd2 hsp hdiscount
  -- skipping the proof for now
  sorry

end normal_price_of_article_l526_526612


namespace compare_a_b_c_l526_526815

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b :=
by
  have : c < b := sorry,
  have : a < b := sorry,
  have : a > c := sorry,
  exact ⟨this.2, this.1⟩

end compare_a_b_c_l526_526815


namespace compare_abc_l526_526825
open Real

noncomputable def a : ℝ := 0.1 * exp(0.1)
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - log 0.9

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l526_526825


namespace factorization_of_square_difference_l526_526769

variable (t : ℝ)

theorem factorization_of_square_difference : t^2 - 144 = (t - 12) * (t + 12) := 
sorry

end factorization_of_square_difference_l526_526769


namespace problem_solution_l526_526118

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def n : ℕ := primes.foldl (·*·) 1

def φ (x : ℕ) : ℕ := x.totient

def S (n : ℕ) : ℕ :=
  ∑ x y : ℕ in (Finset.divisors n).product (Finset.divisors n), if x * y ∣ n then φ x * y else 0

theorem problem_solution (n := primes.foldl (·*·) 1) (S := S n) :
  S / n = 1024 :=
by
  sorry

end problem_solution_l526_526118


namespace conjugate_coordinates_l526_526940

open Complex

theorem conjugate_coordinates (z : ℂ) (h : conj z = 1 - sqrt 3 * I) : z = 1 + sqrt 3 * I :=
by
  sorry

end conjugate_coordinates_l526_526940


namespace Robinson_age_l526_526043

theorem Robinson_age (R : ℕ)
    (brother : ℕ := R + 2)
    (sister : ℕ := R + 6)
    (mother : ℕ := R + 20)
    (avg_age_yesterday : ℕ := 39)
    (total_age_yesterday : ℕ := 156)
    (eq : (R - 1) + (brother - 1) + (sister - 1) + (mother - 1) = total_age_yesterday) :
  R = 33 :=
by
  sorry

end Robinson_age_l526_526043


namespace count_even_factors_l526_526922

theorem count_even_factors : 
  let n := 2^2 * 3^2 * 7^2 in
  ∃ k, k = 18 ∧ (set.count (λ d, d ∣ n ∧ d.even) (set.range (λ ⟨x, y, z⟩, 2^x * 3^y * 7^z) ⟨fin.range 3, fin.range 3, fin.range 3⟩)) = k :=
by
  let n := 2^2 * 3^2 * 7^2
  use 18
  have : ∀ d, d ∣ n ∧ d.even → ∃ x y z, 1 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 3 ∧ 0 ≤ z ∧ z < 3 ∧ d = 2^x * 3^y * 7^z := sorry
  have count : set.count (λ d, d ∣ n ∧ d.even) (set.range (λ ⟨x, y, z⟩, 2^x * 3^y * 7^z) ⟨fin.range 3, fin.range 3, fin.range 3⟩) = 18 := sorry
  exact ⟨18, rfl⟩

end count_even_factors_l526_526922


namespace percent_rain_monday_l526_526078

variables (M T N : ℝ)

-- Conditions
def received_rain_tuesday : Prop := T = 0.54
def no_rain_either_day : Prop := N = 0.28
def rain_both_days : Prop := M ∩ T = 0.44

theorem percent_rain_monday :
  (∃ M : ℝ, (M + T - M ∩ T + N = 1) ∧ received_rain_tuesday ∧ no_rain_either_day ∧ rain_both_days ∧ M = 0.62) :=
sorry

end percent_rain_monday_l526_526078


namespace probability_at_least_half_correct_l526_526531

-- Definitions based on conditions
def num_questions : ℕ := 20
def prob_correct : ℝ := 1/2
def min_correct : ℕ := 10

-- Main problem statement
theorem probability_at_least_half_correct :
  (∑ k in Finset.range (num_questions + 1), if k ≥ min_correct then 
    (Nat.choose num_questions k : ℝ) * (prob_correct ^ k) * ((1 - prob_correct) ^ (num_questions - k)) else 0) = 1/2 := 
by {
  sorry
}

end probability_at_least_half_correct_l526_526531


namespace probability_of_both_events_l526_526052

theorem probability_of_both_events (P : Set → ℚ) (A B : Set) 
  (hA : P A = 3 / 4)
  (hB : P B = 1 / 2)
  (hNotAandNotB : P (compl A ∩ compl B) = 0.125) : 
  P (A ∩ B) = 0.375 :=
by
  -- Proof skeletons can be provided here
  sorry

end probability_of_both_events_l526_526052


namespace proof_problem_l526_526828

-- Define the constants a, b, c
def a := 0.1 * Real.exp 0.1
def b := 1 / 9
def c := -Real.log 0.9

-- State the theorem to prove
theorem proof_problem : c < a ∧ a < b := 
by
  sorry

end proof_problem_l526_526828


namespace vision_data_l526_526545

theorem vision_data (L V : ℝ) (approx : ℝ) (h1 : L = 5 + real.log10 V) (h2 : L = 4.9) (h3 : approx = 1.259) : 
  V = 0.8 := 
sorry

end vision_data_l526_526545


namespace angle_between_vectors_l526_526400

variables {E : Type*} [inner_product_space ℝ E]

theorem angle_between_vectors (a b : E) (h : ∥a + b∥ = ∥a - b∥) : ⟪a, b⟫ = 0 :=
sorry

end angle_between_vectors_l526_526400


namespace percentage_of_male_champions_with_beard_l526_526471

theorem percentage_of_male_champions_with_beard (women_percentage : ℝ)
  (years : ℕ) (total_bearded_men : ℕ) (one_per_year : years = 25) (women_ratio : women_percentage = 0.60) 
  (bearded_men : total_bearded_men = 4) : 
  let men_percentage := 1 - women_percentage,
      total_men := men_percentage * years in 
  (total_bearded_men / total_men) * 100 = 40 := 
by 
  unfold men_percentage total_men; 
  rw [women_ratio, one_per_year, bearded_men];
  have total_men := 0.4 * 25;
  have bearded_ratio := 4 / total_men;
  have percentage := bearded_ratio * 100;
  have h := percentage = 40;
  exact h;
  sorry

end percentage_of_male_champions_with_beard_l526_526471


namespace friends_among_25_students_l526_526412

theorem friends_among_25_students :
  ∀ (students : Fin 25 → Type)
    (friendship : ∀ (a b : Fin 25), Prop)
    (h1 : ∀ (a b c : Fin 25), (friendship a b ∨ friendship b c ∨ friendship a c)),
    ∃ (x : Fin 25), (∑ i, if friendship x i then 1 else 0) ≥ 12 := 
by 
  sorry

end friends_among_25_students_l526_526412


namespace repeating_decimal_exceeds_finite_decimal_by_l526_526733

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l526_526733


namespace right_triangle_median_length_l526_526079

theorem right_triangle_median_length (X Y Z : Point)
  (right_triangle : is_right_triangle X Y Z)
  (median_XY_YZ : length (median X (segment Y Z)) = 5)
  (median_XY_XZ : length (median Y (segment X Z)) = 3 * √5)
  (area_XYZ : area (triangle X Y Z) = 30) :
  length (segment X Y) = 2 * √14 :=
  sorry

end right_triangle_median_length_l526_526079


namespace evaluate_f_at_3_l526_526868

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^7 + a * x^5 + b * x - 5

theorem evaluate_f_at_3 (a b : ℝ)
  (h : f (-3) a b = 5) : f 3 a b = -15 :=
by
  sorry

end evaluate_f_at_3_l526_526868


namespace sum_of_possible_k_l526_526435

theorem sum_of_possible_k (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  j > 0 ∧ k > 0 → 
  ∑ i in {20, 12, 8, 6, 5}.to_finset, i = 51 :=
by
  sorry

end sum_of_possible_k_l526_526435


namespace min_length_of_PT_l526_526669

-- Define the circle with the given parameters
def circle (a : ℝ) : set (ℝ × ℝ) := { p | (p.1)^2 + (p.2)^2 - 2 * a * p.2 + a^2 - 2 = 0 }

-- Define the fixed point P(2, -1)
def point_P := (2 : ℝ, -1 : ℝ)

-- Function to calculate the length of PT
noncomputable def length_PT (a : ℝ) : ℝ :=
  Real.sqrt (((a + 1)^2 + 2) - a^2)

-- Minimum length of PT function
def min_length_PT : ℝ := Real.sqrt 2

-- Theorem statement asserting the minimum length of PT
theorem min_length_of_PT (a : ℝ) (ha : a = -1) : length_PT a = min_length_PT :=
by sorry

end min_length_of_PT_l526_526669


namespace min_length_intersect_l526_526038

-- Definitions for M, N, and P following the given conditions
def M (m : ℝ) : set ℝ := {x | m ≤ x ∧ x ≤ m + 3/4}
def N (n : ℝ) : set ℝ := {x | n - 1/3 ≤ x ∧ x ≤ n}
def P : set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Define a subset condition for M and N being subsets of P
def subset_P {s : set ℝ} : Prop := ∀ x, s x → P x

-- The intersection of M and N
def intersection (m n : ℝ) : set ℝ := M m ∩ N n

-- Length of a set defined by an interval
def length (s : set ℝ) : ℝ := if ∃ a b, s = {x | a ≤ x ∧ x ≤ b} then b - a else 0

-- The statement to prove
theorem min_length_intersect (m n : ℝ) : 
  subset_P (M m) → subset_P (N n) → 
  length (intersection m n) = 1/12 :=
by 
  intros hM_P hN_P
  sorry

end min_length_intersect_l526_526038


namespace students_remain_on_bus_l526_526936

theorem students_remain_on_bus (init_students : ℕ) (one_third_leaves : ∀ (n : ℕ), n * 2 / 3)
  (students_after_first : ∀ (n : ℕ), 54 = init_students * 2 / 3)
  (students_after_second : ∀ (n : ℕ), 36 = students_after_first 54 * 2 / 3)
  (students_after_third : ∀ (n : ℕ), 24 = students_after_second 36 * 2 / 3)
  (students_after_fourth : ∀ (n : ℕ), 16 = students_after_third 24 * 2 / 3) :
  16 = students_after_fourth 24 :=
by sorry

end students_remain_on_bus_l526_526936


namespace fraction_option_C_l526_526618

def is_fraction (expr : String) : Prop := 
  expr = "fraction"

def option_C_fraction (x : ℝ) : Prop :=
  ∃ (numerator : ℝ), ∃ (denominator : ℝ), 
  numerator = 2 ∧ denominator = x + 3

theorem fraction_option_C (x : ℝ) (h : x ≠ -3) :
  is_fraction "fraction" ↔ option_C_fraction x :=
by 
  sorry

end fraction_option_C_l526_526618


namespace cone_cross_section_parabola_l526_526651

/-- A cone with a vertex angle of 90 degrees and a cross-section angle of 45 degrees 
     results in a parabolic cross-section. -/
theorem cone_cross_section_parabola :
  ∀ (cone : Type) (vertex_angle cross_section_angle : ℝ),
  vertex_angle = 90 ∧ cross_section_angle = 45 → curve_of_cross_section cone = "parabola" :=
by
  -- Assuming necessary definitions and properties
  intros
  sorry  -- Proof will go here

end cone_cross_section_parabola_l526_526651


namespace angle_between_plane_and_face_of_cube_l526_526424

theorem angle_between_plane_and_face_of_cube 
  (a : ℝ) (A B C D A1 B1 C1 D1 : ℝ^3)
  (cube : cube A B C D A1 B1 C1 D1)
  (midpoint_DD1 : midpoint (D, D1))
  (midpoint_D1C1 : midpoint (D1, C1))
  (plane : plane_through_points A midpoint_DD1 midpoint_D1C1) :
  angle_between plane.face_abc 
  = real.arctan (real.sqrt 5 / 2) := 
sorry

end angle_between_plane_and_face_of_cube_l526_526424


namespace segments_not_arrangeable_l526_526977

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

end segments_not_arrangeable_l526_526977


namespace coefficient_x5_in_expansion_l526_526191

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binomial_coefficient n k + binomial_coefficient n (k + 1)

open_locale big_operators

theorem coefficient_x5_in_expansion :
  ∀ (x : ℝ), (x + 3 * real.sqrt 2)^9.coeff 5 = 20412 :=
begin
  -- Your proof goes here.
  sorry
end

end coefficient_x5_in_expansion_l526_526191


namespace proof_problem_l526_526994

def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x^2 + 2 * x + 1

theorem proof_problem : f (g 3) - g (f 3) = -5 := by
  sorry

end proof_problem_l526_526994


namespace perpendicular_lines_condition_perpendicular_lines_sufficient_not_necessary_l526_526346

-- Mathematical definitions and theorems required for the problem
theorem perpendicular_lines_condition (m : ℝ) :
  3 * m + m * (2 * m - 1) = 0 ↔ (m = 0 ∨ m = -1) :=
by sorry

-- Translate the specific problem into Lean
theorem perpendicular_lines_sufficient_not_necessary (m : ℝ) (h : 3 * m + m * (2 * m - 1) = 0) :
  m = -1 ∨ (m ≠ -1 ∧ 3 * m + m * (2 * m - 1) = 0) :=
by sorry

end perpendicular_lines_condition_perpendicular_lines_sufficient_not_necessary_l526_526346


namespace average_price_of_pig_l526_526222

theorem average_price_of_pig :
  ∀ (total_cost total_cost_hens total_cost_pigs : ℕ) (num_hens num_pigs avg_price_hen avg_price_pig : ℕ),
  num_hens = 10 →
  num_pigs = 3 →
  total_cost = 1200 →
  avg_price_hen = 30 →
  total_cost_hens = num_hens * avg_price_hen →
  total_cost_pigs = total_cost - total_cost_hens →
  avg_price_pig = total_cost_pigs / num_pigs →
  avg_price_pig = 300 :=
by
  intros total_cost total_cost_hens total_cost_pigs num_hens num_pigs avg_price_hen avg_price_pig h_num_hens h_num_pigs h_total_cost h_avg_price_hen h_total_cost_hens h_total_cost_pigs h_avg_price_pig
  sorry

end average_price_of_pig_l526_526222


namespace compare_constants_l526_526842

theorem compare_constants :
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  in c < a ∧ a < b :=
by
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  sorry

end compare_constants_l526_526842


namespace isosceles_angle_bisector_theorem_l526_526092

/-- In triangle ABC, if b = c and BE is the angle bisector, then A = 100° ↔ AE + BE = BC -/
theorem isosceles_angle_bisector_theorem 
  {A B C E : Point} (b c : ℝ) (h : b = c) (h₁ : is_angle_bisector B E (A, B, C)) :
  (∠BAC = 100 : ℝ) ↔ (dist A E + dist B E = dist B C) :=
sorry

end isosceles_angle_bisector_theorem_l526_526092


namespace find_f_at_2_l526_526894

-- Given conditions
def f (a b x : ℝ) : ℝ := (a * x + b) / (x^2 + 1)
def is_odd_fun (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

-- Prove the target statement
theorem find_f_at_2 (a b : ℝ) (Hodd : is_odd_fun (f a b)) (H_val : f a b (1/2) = 2/5) : f a b 2 = 2/5 :=
by
  sorry

end find_f_at_2_l526_526894


namespace find_full_haired_dogs_l526_526978

-- Definitions of the given conditions
def minutes_per_short_haired_dog : Nat := 10
def short_haired_dogs : Nat := 6
def total_time_minutes : Nat := 4 * 60
def twice_as_long (n : Nat) : Nat := 2 * n

-- Define the problem
def full_haired_dogs : Nat :=
  let short_haired_total_time := short_haired_dogs * minutes_per_short_haired_dog
  let remaining_time := total_time_minutes - short_haired_total_time
  remaining_time / (twice_as_long minutes_per_short_haired_dog)

-- Theorem statement
theorem find_full_haired_dogs : 
  full_haired_dogs = 9 :=
by
  sorry

end find_full_haired_dogs_l526_526978


namespace sum_of_a_b_either_1_or_neg1_l526_526150

theorem sum_of_a_b_either_1_or_neg1 (a b : ℝ) (h1 : a + a = 0) (h2 : b * b = 1) : a + b = 1 ∨ a + b = -1 :=
by {
  sorry
}

end sum_of_a_b_either_1_or_neg1_l526_526150


namespace value_of_a_minus_b_l526_526046

theorem value_of_a_minus_b 
  (a b : ℤ)
  (h1 : 1010 * a + 1014 * b = 1018)
  (h2 : 1012 * a + 1016 * b = 1020) : 
  a - b = -3 :=
sorry

end value_of_a_minus_b_l526_526046


namespace sum_of_possible_ks_l526_526452

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l526_526452


namespace missing_dimension_of_soap_box_l526_526657

theorem missing_dimension_of_soap_box 
  (volume_carton : ℕ) 
  (volume_soap_box : ℕ)
  (number_of_boxes : ℕ)
  (x : ℕ) 
  (h1 : volume_carton = 25 * 48 * 60) 
  (h2 : volume_soap_box = x * 6 * 5)
  (h3: number_of_boxes = 300)
  (h4 : number_of_boxes * volume_soap_box = volume_carton) : 
  x = 8 := by 
  sorry

end missing_dimension_of_soap_box_l526_526657


namespace girls_in_class4_1_l526_526176

theorem girls_in_class4_1 (total_students grade: ℕ)
    (total_girls: ℕ)
    (students_class4_1: ℕ)
    (boys_class4_2: ℕ)
    (h1: total_students = 72)
    (h2: total_girls = 35)
    (h3: students_class4_1 = 36)
    (h4: boys_class4_2 = 19) :
    (total_girls - (total_students - students_class4_1 - boys_class4_2) = 18) :=
by
    sorry

end girls_in_class4_1_l526_526176


namespace fraction_difference_l526_526689

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l526_526689


namespace relationship_m_n_l526_526340

theorem relationship_m_n (m n : ℝ) (h : 0.2^m < 0.2^n) : m > n :=
sorry

end relationship_m_n_l526_526340


namespace find_k_l526_526944

theorem find_k (x y k : ℤ) 
  (h1 : 2 * x - y = 5 * k + 6) 
  (h2 : 4 * x + 7 * y = k) 
  (h3 : x + y = 2023) : 
  k = 2022 := 
  by 
    sorry

end find_k_l526_526944


namespace find_ellipse_find_point_E_l526_526876

noncomputable theory
open Real

-- Definitions
def ellipse_equation (a b t : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity (a b : ℝ) : ℝ := 
  (sqrt (a^2 - b^2)) / a

def circle_tangent_line (a : ℝ) : Prop := 
  let circle_radius := a in
  let line_coeff := (2, -sqrt 2, 6) in -- line: 2x - sqrt(2)y + 6 = 0
  abs (2 * (6/sqrt(6)) - sqrt 2 * 0 + 6) / sqrt ((2^2) + (-(sqrt 2))^2) = circle_radius

def intersection_condition (k : ℝ) : ℝ × ℝ → Prop
| (x, y) := (1 + 3 * k^2) * x^2 - 12 * k^2 * x + 12 * k^2 - 6 = 0

def constant_expression (m x1 x2 k : ℝ) : ℝ :=
  (x1-m) * (x2-m) + k^2 * (x1-2) * (x2-2)

-- Proof statements
theorem find_ellipse {
  a b : ℝ}
  (ha : a > 0)
  (hb : 0 < b)
  (hab : a > b)
  (h_ecc : eccentricity a b = sqrt 6 / 3)
  (h_tangent : circle_tangent_line a) : 
  a = sqrt 6 ∧ b = sqrt 2 ∧ ellipse_equation a b 1 x y=1 
:= sorry

theorem find_point_E {
  a b : ℝ}
  (ha : ellipse_equation a b 1 x y=1)
  (hab : a > b > 0)
  (h_ecc : eccentricity a b = sqrt 6 / 3)
  (h_tangent : circle_tangent_line a)
  {k m : ℝ} (h_intersection : ∀ x1 x2, intersection_condition k (x1, y1) ∧ intersection_condition k (x2, y2))
  (h_const : ∀ x1 x2, constant_expression m x1 x2 k = -5 / 9) :
  m = 7/3 ∧ constant_expression m x1 x2 k = -5/ 9
:= sorry

end find_ellipse_find_point_E_l526_526876


namespace prime_product_conditions_l526_526503

theorem prime_product_conditions (A B : ℕ) (hA : nat.prime A) (hB : nat.prime B)
  (hABdiff : nat.prime (A - B)) (hABsum : nat.prime (A + B))
  (hposA : 0 < A) (hposB : 0 < B) :
  (A * B * (A - B) * (A + B)).even ∧ (A * B * (A - B) * (A + B)) % 3 = 0 :=
sorry

end prime_product_conditions_l526_526503


namespace distance_AB_l526_526088

-- Definitions for the curve C and the line l in polar coordinates.
def curve_c (ρ θ : ℝ) : Prop :=
  ρ^2 = 15 / (1 + 2 * (Real.cos θ)^2)

def line_l (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ + π/3) = Real.sqrt 3

-- Parametric equations of the line l.
def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (-(1/2) * t, Real.sqrt 3 + (Real.sqrt 3 / 2) * t)

-- Proving the distance |AB|
theorem distance_AB : ∀ (t1 t2 : ℝ), 
  (3 * ((-(1/2) * t1)^2) + ((Real.sqrt 3 + (Real.sqrt 3 / 2) * t1))^2 = 15) → 
  (3 * ((-(1/2) * t2)^2) + ((Real.sqrt 3 + (Real.sqrt 3 / 2) * t2))^2 = 15) →
  abs(t1 - t2) = 6 := 
by
  sorry

end distance_AB_l526_526088


namespace intersection_A_B_l526_526383

open Set Int

def A : Set ℝ := {x : ℝ | (x + 1) * (x - 2) > 0}

def B : Set ℤ := {x : ℤ | x^2 - 9 ≤ 0}

theorem intersection_A_B :
  A ∩ (B : Set ℝ) = ({-3, -2, 3} : Set ℝ) :=
sorry

end intersection_A_B_l526_526383


namespace midpoint_IQ_l526_526873

-- Definitions of the problem
variables {I I1 I2 I3 P Q A A' B B' C C' : Point}
variables {triangle_ABC : Triangle}
variables {circle_I1 circle_I2 circle_I3 circle_I1' circle_I2' circle_I3' : Circle}

-- Given conditions
def condition1 : triangle_ABC.is_non_isosceles_acute ∧ triangle_ABC.incenter = I := sorry
def condition2 : triangle_ABC.midpoint_BC = A' ∧ triangle_ABC.midpoint_CA = B' ∧ triangle_ABC.midpoint_AB = C' := sorry
def condition3 : triangle_ABC.excircle_opposite_A = circle_I1 ∧ triangle_ABC.excircle_opposite_B = circle_I2 ∧ triangle_ABC.excircle_opposite_C = circle_I3 := sorry
def condition4 : circle_I1.reflected_over A' = circle_I1' ∧ circle_I2.reflected_over B' = circle_I2' ∧ circle_I3.reflected_over C' = circle_I3' := sorry
def condition5 : radical_center circle_I1 circle_I2 circle_I3 = P ∧ radical_center circle_I1' circle_I2' circle_I3' = Q := sorry

-- Statement to prove
theorem midpoint_IQ : P = midpoint I Q :=
by {
  intros,
  apply condition1,
  apply condition2,
  apply condition3,
  apply condition4,
  apply condition5,
  sorry
}

end midpoint_IQ_l526_526873


namespace exists_horizontal_chord_l526_526334

theorem exists_horizontal_chord (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_eq : f 0 = f 1) : ∃ n : ℕ, n ≥ 1 ∧ ∃ x : ℝ, 0 ≤ x ∧ x + 1/n ≤ 1 ∧ f x = f (x + 1/n) :=
by
  sorry

end exists_horizontal_chord_l526_526334


namespace sum_possible_k_l526_526461

theorem sum_possible_k (j k : ℕ) (hcond : 1 / j + 1 / k = 1 / 4) (hj : 0 < j) (hk : 0 < k) :
  j ≠ k → ∑ k in {k | ∃ j, 1 / j + 1 / k = 1 / 4}, k = 51 :=
by
  sorry

end sum_possible_k_l526_526461


namespace millicent_fraction_books_l526_526386

variable (M H : ℝ)
variable (F : ℝ)

-- Conditions
def harold_has_half_books (M H : ℝ) : Prop := H = (1 / 2) * M
def harold_brings_one_third_books (M H : ℝ) : Prop := (1 / 3) * H = (1 / 6) * M
def new_library_capacity (M F : ℝ) : Prop := (1 / 6) * M + F * M = (5 / 6) * M

-- Target Proof Statement
theorem millicent_fraction_books (M H F : ℝ) 
    (h1 : harold_has_half_books M H) 
    (h2 : harold_brings_one_third_books M H) 
    (h3 : new_library_capacity M F) : 
    F = 2 / 3 :=
sorry

end millicent_fraction_books_l526_526386


namespace value_of_a8_l526_526800

theorem value_of_a8 (a : ℕ → ℝ) :
  (1 + x) ^ 10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x) ^ 2 + a 3 * (1 - x) ^ 3 +
  a 4 * (1 - x) ^ 4 + a 5 * (1 - x) ^ 5 + a 6 * (1 - x) ^ 6 + a 7 * (1 - x) ^ 7 + 
  a 8 * (1 - x) ^ 8 + a 9 * (1 - x) ^ 9 + a 10 * (1 - x) ^ 10 → 
  a 8 = 180 :=
by
  sorry

end value_of_a8_l526_526800


namespace find_n_l526_526013

variable (n : ℤ)

def lcm_condition1 := nat.lcm 30 n = 90
def lcm_condition2 := nat.lcm n 45 = 180

theorem find_n (h1 : lcm_condition1 n) (h2 : lcm_condition2 n) : n = 36 :=
sorry

end find_n_l526_526013


namespace red_shirts_count_l526_526339

theorem red_shirts_count :
  ∀ (total blue_fraction green_fraction : ℕ),
    total = 60 →
    blue_fraction = total / 3 →
    green_fraction = total / 4 →
    (total - (blue_fraction + green_fraction)) = 25 :=
by
  intros total blue_fraction green_fraction h_total h_blue h_green
  rw [h_total, h_blue, h_green]
  norm_num
  sorry

end red_shirts_count_l526_526339


namespace repeating_seventy_two_exceeds_seventy_two_l526_526736

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l526_526736


namespace extreme_points_interval_l526_526557

-- Define the function
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + x

-- Define the derivative of the function
def f' (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

-- Main statement to prove
theorem extreme_points_interval (a : ℝ) :
  (∀ x > 0, f' a x = 0) →
  a ∈ Set.Ioo (-∞) (-√3) := 
sorry

end extreme_points_interval_l526_526557


namespace valid_colorings_count_l526_526752

def Color := {c : String // c = "Red" ∨ c = "White" ∨ c = "Blue"}

structure Vertex :=
(color : Color)

-- Definition of Hexagon and Triangle adjacencies
structure Hexagon :=
  (v1 v2 v3 v4 v5 v6 : Vertex)
  (h_adj_12 : v1 ≠ v2)
  (h_adj_23 : v2 ≠ v3)
  (h_adj_34 : v3 ≠ v4)
  (h_adj_45 : v4 ≠ v5)
  (h_adj_56 : v5 ≠ v6)
  (h_adj_61 : v6 ≠ v1)

structure Triangle :=
  (v1 v2 v3 : Vertex)
  (t_adj_12 : v1 ≠ v2)
  (t_adj_23 : v2 ≠ v3)
  (t_adj_31 : v3 ≠ v1)


structure HexagonTriangle :=
  (hex : Hexagon)
  (tri : Triangle)
  (shared_vertices_1 : hex.v1 = tri.v1)
  (shared_vertices_2 : hex.v3 = tri.v2)
  (shared_vertices_3 : hex.v5 = tri.v3)

noncomputable def count_valid_colorings : Nat :=
  384

theorem valid_colorings_count :
  ∃ (colorings : HexagonTriangle → Nat), colorings = count_valid_colorings :=
begin
  use (λ _, 384),
  refl,
end

#check valid_colorings_count

end valid_colorings_count_l526_526752


namespace correct_conclusions_count_l526_526333

theorem correct_conclusions_count : 
  let p := λ (x : ℝ), -(x + 1)^2 + 3,
      ax_symmetry := ∀ (x : ℝ), x = -1,
      vertex := (-1, 3),
      decrease_x_gt_1 := ∀ x, (x > 1) → (p x > p (x + 1))
  in 
  (p (0) < p (1)) ∧ (¬ax_symmetry (1)) ∧ vertex = (-1, 3) ∧ decrease_x_gt_1 → 
  3 := sorry

end correct_conclusions_count_l526_526333


namespace EG_perpendicular_HF_l526_526485

-- Define a cyclic quadrilateral (cyclic quadrilateral) and tangential (tangential quadrilateral)
variables {A B C D E F G H : Type}

-- Assuming A, B, C, D lie on a common circle (cyclic quadrilateral) and have a common inscribed circle (tangential quadrilateral)
def cyclic_tangential_quadrilateral (A B C D E F G H : Type) :=
  (cyclic_quadrilateral A B C D) ∧ (tangential_quadrilateral A B C D E F G H)

-- Define points of tangency with sides AB, BC, CD, and DA
def points_of_tangency (A B C D E F G H : Type) :=
  (tangency_point A B E) ∧ (tangency_point B C F) ∧ (tangency_point C D G) ∧ (tangency_point D A H)

-- State the theorem to prove EG is perpendicular to HF
theorem EG_perpendicular_HF (A B C D E F G H : Type)
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : tangential_quadrilateral A B C D E F G H)
  (h3 : points_of_tangency A B C D E F G H) :
  perpendicular (line_through_points E G) (line_through_points H F) :=
begin
  sorry
end

end EG_perpendicular_HF_l526_526485


namespace compare_values_l526_526806

variable (a b c : ℝ)

noncomputable def a_def : ℝ := 0.1 * Real.exp 0.1
noncomputable def b_def : ℝ := 1 / 9
noncomputable def c_def : ℝ := - Real.log 0.9

theorem compare_values : 
  c < a ∧ a < b := 
begin
  let a := 0.1 * Real.exp 0.1,
  let b := 1 / 9,
  let c := - Real.log 0.9,
  have h1 : c < a, sorry,
  have h2 : a < b, sorry,
  exact ⟨h1, h2⟩
end

end compare_values_l526_526806


namespace convex_function_is_linear_l526_526314

theorem convex_function_is_linear (f : ℝ → ℝ) :
  (∀ a b p : ℝ, f (p * a + (1 - p) * b) ≤ p * f(a) + (1 - p) * f(b)) →
  ∃ A B : ℝ, ∀ x : ℝ, f x = A * x + B :=
begin
  sorry
end

end convex_function_is_linear_l526_526314


namespace smallest_n_divisible_by_23_l526_526789

theorem smallest_n_divisible_by_23 :
  ∃ n : ℕ, (n^3 + 12 * n^2 + 15 * n + 180) % 23 = 0 ∧
            ∀ m : ℕ, (m^3 + 12 * m^2 + 15 * m + 180) % 23 = 0 → n ≤ m :=
sorry

end smallest_n_divisible_by_23_l526_526789


namespace sum_extrema_g_l526_526490

def g (x : ℝ) : ℝ := |x - 3| + |2 * x - 8| - |x - 5|

theorem sum_extrema_g :
  (∃ min_val max_val : ℝ, (1 ≤ min_val ∧ min_val ≤ 10) ∧ (1 ≤ max_val ∧ max_val ≤ 10) 
  ∧ (∀ x, 1 ≤ x ∧ x ≤ 10 → (g(min_val) ≤ g(x)) ∧ (g(x) ≤ g(max_val))) 
  ∧ g(min_val) + g(max_val) = 26) := sorry

end sum_extrema_g_l526_526490


namespace smallest_number_to_add_for_divisibility_l526_526195

theorem smallest_number_to_add_for_divisibility (n : ℕ) (h : n + 27461 % 9 = 27461) : 
  9 - (27461 % 9) = 7 :=
by
  have rem : 27461 % 9 = 2 := rfl
  rw [rem]
  exact rfl

end smallest_number_to_add_for_divisibility_l526_526195


namespace compare_constants_l526_526864

noncomputable def a : ℝ := 0.1 * real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - real.log 0.9

theorem compare_constants : c < a ∧ a < b :=
by sorry

end compare_constants_l526_526864


namespace fraction_difference_is_correct_l526_526724

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l526_526724


namespace angle_DSC_eq_150_l526_526350

open EuclideanGeometry

noncomputable def angle_ABC : ℝ := 90
noncomputable def angle_equilateral : ℝ := 60
noncomputable def square_side : ℝ := 1

/- Assume a square ABCD with each side of length 1. S is a point inside the square such that triangle ABS is equilateral. -/
variable (A B C D S : Point)

axiom square (A B C D : Point) : square A B C D
axiom inside_square (S : Point) : S ∈ interior ((triangle A B C D).to_real_area)

axiom equilateral_triangle_ABS (A B S : Point) : equilateral_triangle A B S -- A triangle with all sides and angles equal

theorem angle_DSC_eq_150 (A B C D S : Point) 
  (h1: square A B C D) 
  (h2: S ∈ interior (triangle A B C D).to_real_area) 
  (h3: equilateral_triangle A B S) : 
  angle D S C = 150 :=
sorry

end angle_DSC_eq_150_l526_526350


namespace sin_theta_between_vectors_find_m_perpendicular_l526_526916

variables (a b : ℝ × ℝ)
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

def sin_theta (a b : ℝ × ℝ) : ℝ :=
  let cos_theta := dot_product a b / (magnitude a * magnitude b)
  Real.sqrt (1 - cos_theta ^ 2)

def perp (u v : ℝ × ℝ) : Prop := dot_product u v = 0

theorem sin_theta_between_vectors : 
  let a := (3 : ℝ, -4 : ℝ)
  let b := (1 : ℝ, 2 : ℝ)
  sin_theta a b = 2 * Real.sqrt 5 / 5 := sorry

theorem find_m_perpendicular :
  let a := (3 : ℝ, -4 : ℝ)
  let b := (1 : ℝ, 2 : ℝ)
  ∃ m : ℝ, perp (m • a - b) (a + b) ∧ m = 0 := sorry

end sin_theta_between_vectors_find_m_perpendicular_l526_526916


namespace problem_statement_l526_526883

theorem problem_statement (a b : ℝ) (h1 : 2^a = 10) (h2 : 5^b = 10) : (1 / a) + (1 / b) = 1 :=
sorry

end problem_statement_l526_526883


namespace find_a_value_l526_526971

noncomputable def find_a (a : ℝ) : Prop :=
  (a > 0) ∧ (1 / 3 = 2 / a)

theorem find_a_value (a : ℝ) (h : find_a a) : a = 6 :=
sorry

end find_a_value_l526_526971


namespace perpendicular_eq_l526_526875

-- Define the points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 1 (-1)
def B := Point.mk (-2) 1
def C := Point.mk 3 (-5)

-- Define the midpoint M of segment AC
def M : Point :=
  let x_M := (A.x + C.x) / 2
  let y_M := (A.y + C.y) / 2
  Point.mk x_M y_M

-- Define the slope of BM
def slope (P1 P2 : Point) : ℝ := 
  (P2.y - P1.y) / (P2.x - P1.x)

def slope_BM := slope B M

-- Define the equation of the perpendicular from A
def perpendicular_slope (m : ℝ) : ℝ :=
  -1 / m

def slope_AN := perpendicular_slope slope_BM

-- Define the equation of Line AN using point-slope form
def LineAN (P : Point) (m : ℝ) : ℝ × ℝ × ℝ :=
  (m, -1, m * (-P.x) + P.y)

-- The equation of the perpendicular from A
def LineAN_equation := LineAN A slope_AN

-- Proving the equation is equivalent to x - y - 2 = 0
theorem perpendicular_eq : LineAN_equation = (1, -1, -2) := by
  -- Proof steps would go here
  sorry

end perpendicular_eq_l526_526875


namespace major_axis_length_of_ellipse_l526_526023

theorem major_axis_length_of_ellipse :
  ∀ {y x : ℝ},
  (y^2 / 25 + x^2 / 15 = 1) → 
  2 * Real.sqrt 25 = 10 :=
by
  intro y x h
  sorry

end major_axis_length_of_ellipse_l526_526023


namespace excess_common_fraction_l526_526717

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l526_526717


namespace circle_eq_and_distance_product_l526_526084

open Real

noncomputable def point_P := (1, 1 : ℝ)
noncomputable def inclination_angle := π / 4
noncomputable def polar_eq_circle (ρ θ : ℝ) := ρ = 4 * sin θ

theorem circle_eq_and_distance_product :
  (∃ (x y : ℝ), y = 2 ∧ x^2 + (y-2)^2 = 4) ∧
  (∀ (l : ℝ → ℝ × ℝ) (t : ℝ), 
    l t = (1 + (sqrt 2) / 2 * t, 1 + (sqrt 2) / 2 * t : ℝ) →
    l = λ t, (1 + (sqrt 2) / 2 * t, 1 + (sqrt 2) / 2 * t : ℝ) →
    ((√2) * (-(√2)) = 2)) :=
begin
  sorry
end

end circle_eq_and_distance_product_l526_526084


namespace north_pond_duck_estimate_l526_526125

theorem north_pond_duck_estimate :
  let M_LM := 100 in
  let P_LM := 75 in
  let M_NP := 2 * M_LM + 6 in
  let P_NP := 4 * M_LM in
  ∀ T_NP : ℕ, T_NP ≤ Int.natAbs (Int.sqrt (M_NP * P_NP)) →
  M_NP + P_NP + T_NP = 893 :=
begin
  intros M_LM P_LM M_NP P_NP T_NP h,
  have M_LM_def : M_LM = 100 := rfl,
  have P_LM_def : P_LM = 75 := rfl,
  have M_NP_def : M_NP = 2 * M_LM + 6 := rfl,
  have P_NP_def : P_NP = 4 * M_LM := rfl,
  calc
    M_NP + P_NP + T_NP = 206 + 400 + 287 : by sorry
                      ... = 893 : by sorry
end

end north_pond_duck_estimate_l526_526125


namespace parameterization_of_line_l526_526163

theorem parameterization_of_line (s l : ℝ) : 
  let t := (1 : ℝ),
      point := (⟨-3 + l, -3⟩ : ℝ × ℝ),
      y_eq := (2 / 3) * point.1 + 5,
      initial_point := (⟨-3, s⟩ : ℝ × ℝ),
      direction := (⟨l, -6⟩ : ℝ × ℝ),
      param_point := initial_point + t • direction 
  in s = 3 ∧ l = -9 ∧ point.2 = y_eq := 
by
  sorry

end parameterization_of_line_l526_526163


namespace irrational_approximation_l526_526134

theorem irrational_approximation (p q : ℤ) (hq : q ≠ 0) : abs (real.sqrt 2 - (p / q : ℝ)) > 1 / (3 * q^2) :=
sorry

end irrational_approximation_l526_526134


namespace smallest_prime_with_digit_sum_23_l526_526615

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Prime p ∧ digit_sum p = 23 ∧ ∀ q : ℕ, Prime q ∧ digit_sum q = 23 → p ≤ q :=
by
  sorry

end smallest_prime_with_digit_sum_23_l526_526615


namespace equivalent_expression_l526_526278

variable (x y : ℝ)

def is_positive_real (r : ℝ) : Prop := r > 0

theorem equivalent_expression 
  (hx : is_positive_real x) 
  (hy : is_positive_real y) : 
  (Real.sqrt (Real.sqrt (x ^ 2 * Real.sqrt (y ^ 3)))) = x ^ (1 / 2) * y ^ (1 / 12) :=
by
  sorry

end equivalent_expression_l526_526278


namespace minimum_value_ge_100_minimum_value_eq_100_l526_526486

noncomputable def minimum_value_expression (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 10)^2

theorem minimum_value_ge_100 (α β : ℝ) : minimum_value_expression α β ≥ 100 :=
  sorry

theorem minimum_value_eq_100 (α β : ℝ)
  (hα : 3 * Real.cos α + 4 * Real.sin β = 7)
  (hβ : 3 * Real.sin α + 4 * Real.cos β = 10) :
  minimum_value_expression α β = 100 :=
  sorry

end minimum_value_ge_100_minimum_value_eq_100_l526_526486


namespace non_congruent_squares_in_6x6_grid_l526_526392

theorem non_congruent_squares_in_6x6_grid : 
  let grid_size := 6
  let regular_squares := (let f := λ n, (grid_size - n) ^ 2 in f 1 + f 2 + f 3 + f 4 + f 5),
  let diagonal_squares := (let f := λ n, (grid_size - (n + 1)) ^ 2 in f 1 + f 2 + f 3)
  in regular_squares + diagonal_squares = 105
:= by
  let grid_size := 6
  let f n := (grid_size - n) ^ 2
  have regular_squares : ∀ n, f 1 + f 2 + f 3 + f 4 + f 5 = 25 + 16 + 9 + 4 + 1 := sorry
  let diagonal_f n := (grid_size - (n + 1)) ^ 2
  have diagonal_squares : ∀ n, diagonal_f 1 + diagonal_f 2 + diagonal_f 3 = 25 + 16 + 9 := sorry
  have total_squares : regular_squares + diagonal_squares = 25 + 16 + 9 + 4 + 1 + 25 + 16 + 9 := sorry
  exact (regular_squares + diagonal_squares = 105) sorry

end non_congruent_squares_in_6x6_grid_l526_526392


namespace even_square_is_even_l526_526007

theorem even_square_is_even (a : ℤ) (h : Even (a^2)) : Even a :=
sorry

end even_square_is_even_l526_526007


namespace solve_for_C_l526_526351

instance : Nonempty (Real) := ⟨0⟩

variable (A B C : ℝ)

def vec_m := (sqrt 3 * Real.sin A, Real.sin B)
def vec_n := (Real.cos B, sqrt 3 * Real.cos A)
def dot_product := (vec_m.1 * vec_n.1) + (vec_m.2 * vec_n.2)
def condition := dot_product = 1 + Real.cos (A + B)

theorem solve_for_C (h : condition A B) : C = (2 * Real.pi) / 3 := 
  sorry

end solve_for_C_l526_526351


namespace jason_current_cards_l526_526980

-- Define the initial number of Pokemon cards Jason had.
def initial_cards : ℕ := 9

-- Define the number of Pokemon cards Jason gave to his friends.
def given_away : ℕ := 4

-- Prove that the number of Pokemon cards he has now is 5.
theorem jason_current_cards : initial_cards - given_away = 5 := by
  sorry

end jason_current_cards_l526_526980


namespace largest_unique_digit_number_sum_37_l526_526322

theorem largest_unique_digit_number_sum_37 :
  ∃ (n : ℕ), (∀ i j, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)) ∧
             (n.digits.sum = 37) ∧
             (∀ m, (∀ i j, i ≠ j → (m / 10^i % 10) ≠ (m / 10^j % 10)) ∧ m.digits.sum = 37 → m ≤ n) :=
sorry

end largest_unique_digit_number_sum_37_l526_526322


namespace cos_of_tan_l526_526410

theorem cos_of_tan (A B C : Type) [InnerProductSpace ℝ Type] [RightTriangle A B C] (angle_a : A = 90°) (tan_c : tan C = 4) : cos C = (√17) / 17 :=
by
  sorry

end cos_of_tan_l526_526410


namespace quadratic_inequality_solution_l526_526040

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ ax^2 + bx + c > 0) :
  ∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - bx + c > 0 := 
sorry

end quadratic_inequality_solution_l526_526040


namespace perpendicular_line_to_line_l526_526012

variable (l m : Type) [Line l] [Line m]
variable (α β : Type) [Plane α] [Plane β]

-- Definitions of perpendicular and subset relationships
variables [Perpendicular l α] [Parallel α β] [Subset m β]

theorem perpendicular_line_to_line {l : Type} [Line l] {m : Type} [Line m] 
  {α : Type} [Plane α] {β : Type} [Plane β] 
  [Perpendicular l α] [Parallel α β] [Subset m β] : Perpendicular l m := 
  sorry -- proof will go here

end perpendicular_line_to_line_l526_526012


namespace outdoor_section_length_l526_526664

theorem outdoor_section_length (W : ℝ) (A : ℝ) (hW : W = 4) (hA : A = 24) : ∃ L : ℝ, A = W * L ∧ L = 6 := 
by
  use 6
  sorry

end outdoor_section_length_l526_526664


namespace area_of_trapezium_l526_526776

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
by sorry

end area_of_trapezium_l526_526776


namespace calculation_of_factorial_expression_l526_526283

theorem calculation_of_factorial_expression :
  6! - 5 * 5! - 5! = 0 := 
  sorry

end calculation_of_factorial_expression_l526_526283


namespace find_inheritance_l526_526987

noncomputable def inheritance (y : ℝ) : Prop :=
  let federal_tax := 0.25 * y
  let remaining_after_federal := y - federal_tax
  let state_tax := 0.15 * remaining_after_federal
  let total_tax := federal_tax + state_tax
  total_tax = 15000

theorem find_inheritance : ∃ y : ℝ, inheritance y ∧ y = 41379 :=
by
  use 41379
  unfold inheritance
  have h : 0.25 * 41379 + 0.15 * (41379 - 0.25 * 41379) = 15000 := by norm_num
  exact ⟨h, rfl⟩

end find_inheritance_l526_526987


namespace registration_methods_count_l526_526933

theorem registration_methods_count (n m : ℕ) : n = 4 ∧ m = 3 → (m ^ n = 81) :=
by
  { rintros ⟨hn, hm⟩,
    rw [hm, hn],
    exact pow_succ m (n - 1), sorry }

end registration_methods_count_l526_526933


namespace oranges_sold_in_the_morning_eq_30_l526_526184

variable (O : ℝ)  -- Denote the number of oranges Wendy sold in the morning

-- Conditions as assumptions
def price_per_apple : ℝ := 1.5
def price_per_orange : ℝ := 1
def morning_apples_sold : ℝ := 40
def afternoon_apples_sold : ℝ := 50
def afternoon_oranges_sold : ℝ := 40
def total_sales_for_day : ℝ := 205

-- Prove that O, satisfying the given conditions, equals 30
theorem oranges_sold_in_the_morning_eq_30 (h : 
    (morning_apples_sold * price_per_apple) +
    (O * price_per_orange) +
    (afternoon_apples_sold * price_per_apple) +
    (afternoon_oranges_sold * price_per_orange) = 
    total_sales_for_day
  ) : O = 30 :=
by
  sorry

end oranges_sold_in_the_morning_eq_30_l526_526184


namespace log_a_one_zero_log_a_a_one_l526_526529

variable (a : ℝ)

theorem log_a_one_zero (h1 : a > 0) (h2 : a ≠ 1) : log a 1 = 0 :=
sorry

theorem log_a_a_one (h1 : a > 0) (h2 : a ≠ 1) : log a a = 1 :=
sorry

end log_a_one_zero_log_a_a_one_l526_526529


namespace sum_of_possible_ks_l526_526429

theorem sum_of_possible_ks : 
  (∃ (j k : ℕ), (1 < j) ∧ (1 < k) ∧ j ≠ k ∧ ((1/j : ℝ) + (1/k : ℝ) = (1/4))) → 
  (∑ k in {20, 12, 8, 6, 5}, k) = 51 :=
begin
  sorry
end

end sum_of_possible_ks_l526_526429


namespace repeating_decimal_exceeds_finite_decimal_by_l526_526728

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l526_526728


namespace number_of_lockers_l526_526762

theorem number_of_lockers (cost_per_digit : ℝ) (total_cost : ℝ) :
  cost_per_digit = 0.03 ∧ total_cost = 326.91 → 
  ∃ n : ℕ, n = 3000 ∧ 
    total_cost = 
      ∑ i in finset.filter (λ x, x < 10) (finset.range (n + 1)), (cost_per_digit * (string.length (to_string i))) +
      ∑ i in finset.filter (λ x, x ≥ 10 ∧ x < 100) (finset.range (n + 1)), (cost_per_digit * (string.length (to_string i))) +
      ∑ i in finset.filter (λ x, x ≥ 100 ∧ x < 1000) (finset.range (n + 1)), (cost_per_digit * (string.length (to_string i))) +
      ∑ i in finset.filter (λ x, x ≥ 1000) (finset.range (n + 1)), (cost_per_digit * (string.length (to_string i))) :=
begin
  sorry
end

end number_of_lockers_l526_526762


namespace repeating_decimal_exceeds_finite_decimal_by_l526_526731

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l526_526731


namespace repeatingDecimal_exceeds_l526_526702

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l526_526702


namespace profit_percentage_with_discount_l526_526259

variable (costPrice sellingPrice discount : ℝ)
variable (originalProfitPercentage discountedProfitPercentage : ℝ)

-- Conditions
def condition1 : discount = 0.05 := sorry
def condition2 : originalProfitPercentage = 0.34 := sorry
def condition3 : costPrice = 100 := sorry
def condition4 : sellingPrice = costPrice * (1 + originalProfitPercentage) := sorry

-- Statement to prove
theorem profit_percentage_with_discount 
  (h₁ : condition1) 
  (h₂ : condition2) 
  (h₃ : condition3)
  (h₄ : condition4) :
  discountedProfitPercentage = 27.3 := sorry

end profit_percentage_with_discount_l526_526259


namespace infinite_series_sum_l526_526997

noncomputable def compute_sum (p q : ℝ) (h1 : 0 < p) (h2 : 0 < q) (h3 : p > 2 * q) : ℝ :=
  ∑' n, 1 / ([(2 * n - 1) * p - (2 * n - 2) * q] * [(2 * n + 1) * p - 2 * n * q])

theorem infinite_series_sum (p q : ℝ) (h1 : 0 < p) (h2 : 0 < q) (h3 : p > 2 * q) :
  compute_sum p q h1 h2 h3 = 1 / ((p - 2 * q) * p) :=
by
  sorry

end infinite_series_sum_l526_526997


namespace find_number_l526_526402

theorem find_number : ∀ (x : ℝ), (0.15 * 0.30 * 0.50 * x = 99) → (x = 4400) :=
by
  intro x
  intro h
  sorry

end find_number_l526_526402


namespace chandler_tickets_total_cost_l526_526292

theorem chandler_tickets_total_cost :
  let movie_ticket_cost := 30
  let num_movie_tickets := 8
  let num_football_tickets := 5
  let num_concert_tickets := 3
  let num_theater_tickets := 4
  let theater_ticket_cost := 40
  let discount := 0.10
  let total_movie_cost := num_movie_tickets * movie_ticket_cost
  let football_ticket_cost := total_movie_cost / 2
  let total_football_cost := num_football_tickets * football_ticket_cost
  let concert_ticket_cost := football_ticket_cost - 10
  let total_concert_cost := num_concert_tickets * concert_ticket_cost
  let discounted_theater_ticket_cost := theater_ticket_cost * (1 - discount)
  let total_theater_cost := num_theater_tickets * discounted_theater_ticket_cost
  let total_cost := total_movie_cost + total_football_cost + total_concert_cost + total_theater_cost
  total_cost = 1314 := by
  sorry

end chandler_tickets_total_cost_l526_526292


namespace clare_bought_loaves_l526_526293

-- Define the given conditions
def initial_amount : ℕ := 47
def remaining_amount : ℕ := 35
def cost_per_loaf : ℕ := 2
def cost_per_carton : ℕ := 2
def number_of_cartons : ℕ := 2

-- Required to prove the number of loaves of bread bought by Clare
theorem clare_bought_loaves (initial_amount remaining_amount cost_per_loaf cost_per_carton number_of_cartons : ℕ) 
    (h1 : initial_amount = 47) 
    (h2 : remaining_amount = 35) 
    (h3 : cost_per_loaf = 2) 
    (h4 : cost_per_carton = 2) 
    (h5 : number_of_cartons = 2) : 
    (initial_amount - remaining_amount - cost_per_carton * number_of_cartons) / cost_per_loaf = 4 :=
by sorry

end clare_bought_loaves_l526_526293


namespace jenna_dice_rolls_l526_526934

theorem jenna_dice_rolls :
  let seq_prob := (choose 17 3) * (1/6)^3 * (5/6)^15 * (5/6) * (1/6)
  in seq_prob = (680 * 5^16) / 6^20 :=
by sorry

end jenna_dice_rolls_l526_526934


namespace orthocenter_of_triangle_ABC_l526_526962

open Point

-- Definition of the points A, B, and C.
def A : Point ℝ := (2, 3, 4)
def B : Point ℝ := (6, 4, 2)
def C : Point ℝ := (4, 5, 6)

-- Definition of the orthocenter H of the triangle ABC.
def H : Point ℝ := (4, 3, 2)

-- Lean statement to prove that H is the orthocenter of triangle ABC.
theorem orthocenter_of_triangle_ABC : 
  orthocenter A B C = H :=
sorry

end orthocenter_of_triangle_ABC_l526_526962


namespace popped_kernels_second_bag_l526_526129

theorem popped_kernels_second_bag:
  let k1 := 60 in                                   -- Number of popped kernels in the first bag
  let total_k1 := 75 in                             -- Total number of kernels in the first bag
  let k3 := 82 in                                   -- Number of popped kernels in the third bag
  let total_k3 := 100 in                            -- Total number of kernels in the third bag
  let total_k2 := 50 in                             -- Total number of kernels in the second bag
  let avg_percentage := 82 in                       -- Average percentage of popped kernels
  let total_percentage := avg_percentage * 3 in     -- Total percentage for three bags
  (k1 * 100 / total_k1 + avg_percentage + k3 * 100 / total_k3 - total_percentage = 0) → -- Condition
  ((k1 * 100 / total_k1 + avg_percentage + k3 * 100 / total_k3 - total_percentage) = 0) :=
by
  sorry

end popped_kernels_second_bag_l526_526129


namespace part1_part2_l526_526033

noncomputable def f (a x : ℝ) : ℝ :=
  a * real.exp (-x) + x - 2

theorem part1 (a : ℝ) (x : ℝ) (h : a = 2) (hx : -1 ≤ x ∧ x ≤ 3) :
  f a x ∈ set.Icc (real.log 2 - 1) (2 * real.exp 1 - 3) := sorry

theorem part2 (a x1 x2 : ℝ) (h1 : f a x1 = 0) (h2 : f a x2 = 0) (h3 : x1 * x2 < 0) :
  0 < a ∧ a < 2 ∧ x1 + x2 > 2 * real.log a := sorry

end part1_part2_l526_526033


namespace number_is_square_l526_526327

theorem number_is_square (n : ℕ) (h : (n^5 + n^4 + 1).factors.length = 6) : ∃ k : ℕ, n^3 - n + 1 = k^2 :=
begin
  sorry
end

end number_is_square_l526_526327


namespace compare_constants_l526_526862

noncomputable def a : ℝ := 0.1 * real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - real.log 0.9

theorem compare_constants : c < a ∧ a < b :=
by sorry

end compare_constants_l526_526862


namespace shaded_area_l526_526081

-- Define the conditions
def square_side := 8 -- side length of the square in cm
def circle_radius := 3 -- radius of each circle in cm

-- Define the total area of the square
def square_area := square_side * square_side -- Area of the square A_square = 64 cm^2

-- Define the area of one circle
def circle_area := Real.pi * (circle_radius * circle_radius) -- Area of one circle, A_circle = 9π cm^2

-- Define the total area covered by four circles
def total_circle_area := 4 * circle_area -- Total area covered by four circles, 4 * 9π = 36π cm^2

-- Prove the area of the shaded region
theorem shaded_area : 
  (square_area - total_circle_area) = 64 - 36 * Real.pi :=
by 
  sorry

end shaded_area_l526_526081


namespace median_of_set_l526_526565

theorem median_of_set (y : ℝ) (h : (90 + 88 + 85 + 86 + 84 + y) / 6 = 87) : 
  median ({90, 88, 85, 86, 84, y} : set ℝ) = 87 :=
sorry

end median_of_set_l526_526565


namespace perimeter_of_sector_l526_526889

theorem perimeter_of_sector (r : ℝ) (area : ℝ) (perimeter : ℝ) 
  (hr : r = 1) (ha : area = π / 3) : perimeter = (2 * π / 3) + 2 :=
by
  -- You can start the proof here
  sorry

end perimeter_of_sector_l526_526889


namespace compare_constants_l526_526838

theorem compare_constants :
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  in c < a ∧ a < b :=
by
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  sorry

end compare_constants_l526_526838


namespace problem_statement_l526_526370

def sequence_a (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, 2 * S n = a (n + 1) + 2 * (n - 2)

def initial_condition (a : ℕ → ℝ) : Prop :=
  a 1 = 4

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) - 1 = r * (a n - 1)

def sequence_b (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = (a n - 1) / (a n * a (n + 1))

def T_n (b : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, b i

def T_n_less_than_1_div_8 (b : ℕ → ℝ) : Prop :=
  ∀ n, T_n b n < 1 / 8

theorem problem_statement (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) :
  sequence_a a S →
  initial_condition a →
  geometric_sequence a ∧ T_n_less_than_1_div_8 b :=
by
  sorry

end problem_statement_l526_526370


namespace monotonic_decrease_interval_l526_526559

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonic_decrease_interval : 
  ∀ x ∈ Ioo 0 (Real.exp (-1)), deriv f x < 0 :=
by 
  sorry

end monotonic_decrease_interval_l526_526559


namespace range_of_a_l526_526880

def f (x : ℝ) := x^2 - 2 * x
def g (a x : ℝ) := a * x + 2

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∀ x1 : ℝ, x1 ∈ set.Icc (-1) 2 → ∃ x2 : ℝ, x2 ∈ set.Icc (-1) 2 ∧ f x1 = g a x2) ↔ 3 ≤ a :=
by
  sorry

end range_of_a_l526_526880


namespace compare_a_b_c_l526_526812

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b :=
by
  have : c < b := sorry,
  have : a < b := sorry,
  have : a > c := sorry,
  exact ⟨this.2, this.1⟩

end compare_a_b_c_l526_526812


namespace first_student_time_l526_526226

theorem first_student_time
  (n : ℕ)
  (avg_last_three avg_all : ℕ)
  (h_n : n = 4)
  (h_avg_last_three : avg_last_three = 35)
  (h_avg_all : avg_all = 30) :
  let total_time_all := n * avg_all in
  let total_time_last_three := 3 * avg_last_three in
  (total_time_all - total_time_last_three) = 15 :=
by
  let total_time_all := 4 * 30
  let total_time_last_three := 3 * 35
  show total_time_all - total_time_last_three = 15
  sorry

end first_student_time_l526_526226


namespace sum_of_all_possible_k_values_l526_526444

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l526_526444


namespace committee_problem_solution_l526_526240

def committee_problem : Prop :=
  let total_committees := Nat.choose 15 5
  let zero_profs_committees := Nat.choose 8 5
  let one_prof_committees := (Nat.choose 7 1) * (Nat.choose 8 4)
  let undesirable_committees := zero_profs_committees + one_prof_committees
  let desired_committees := total_committees - undesirable_committees
  desired_committees = 2457

theorem committee_problem_solution : committee_problem :=
by
  sorry

end committee_problem_solution_l526_526240


namespace inequality_proof_l526_526858

-- Define constants
def a : ℝ := 0.1 * Real.exp 0.1
def b : ℝ := 1 / 9
def c : ℝ := -Real.log 0.9

-- Assert the inequalities
theorem inequality_proof : c < a ∧ a < b :=
by sorry

end inequality_proof_l526_526858


namespace AM_eq_AN_l526_526277

noncomputable def problem_conditions (A B C O F G M N : Type) [group A] [group B] [group C] [group O] [group F] [group G] [group M] [group N] :
  (∃ (BAC_not_eq_60 : Type) 
     (BD CE circumcircle : Type)
     (DE : Type),
    (tangent_from_B : Type) 
    (tangent_from_C : Type)
    (BD_eq_CE_eq_BC : BD = CE ∧ CE = BC)
    (DE_extends_AB_AC_at_FG : DE)
    (CF_int_BD_at_M : Type)
    (CE_int_BG_at_N : Type) 
     : (Type)) :=
begin
  sorry
end

theorem AM_eq_AN (A B C O F G M N : ℝ) 
  (BAC_not_eq_60 : A ≠ 60) 
  (BD CE circumcircle : ℝ)
  (tangent_from_B : ℝ) 
  (tangent_from_C : ℝ)
  (DE : ℝ)
  (BD_eq_CE_eq_BC : BD = CE ∧ CE = BC)
  (DE_extends_AB_AC_at_FG : DE)
  (CF_int_BD_at_M : ℝ)
  (CE_int_BG_at_N : ℝ) 
  : 
  (AM = AN) :=
by sorry

end AM_eq_AN_l526_526277


namespace tom_initial_money_l526_526178

-- Defining the given values
def super_nintendo_value : ℝ := 150
def store_percentage : ℝ := 0.80
def nes_price : ℝ := 160
def game_value : ℝ := 30
def change_received : ℝ := 10

-- Calculate the credit received for the Super Nintendo
def credit_received := store_percentage * super_nintendo_value

-- Calculate the remaining amount Tom needs to pay for the NES after using the credit
def remaining_amount := nes_price - credit_received

-- Calculate the total amount Tom needs to pay, including the game value
def total_amount_needed := remaining_amount + game_value

-- Proving that the initial money Tom gave is $80
theorem tom_initial_money : total_amount_needed + change_received = 80 :=
by
    sorry

end tom_initial_money_l526_526178


namespace range_of_x_l526_526379

theorem range_of_x (x : ℝ) (h0 : 0 < x) (h1 : 10 ≠ x): 
  (∃ n m k : ℝ, ({0, 1, real.log x} = {n, m, k} ∧ n ≠ m ∧ m ≠ k ∧ n ≠ k)) → 
  x ∈ set.Ioo 0 1 ∪ set.Ioo 1 10 ∪ set.Ioi 10 :=
by
  sorry

end range_of_x_l526_526379


namespace yellow_dandelions_day_before_yesterday_l526_526246

theorem yellow_dandelions_day_before_yesterday :
  ∀ (yellow_yesterday white_yesterday yellow_today white_today : ℕ),
    yellow_yesterday = 20 →
    white_yesterday = 14 →
    yellow_today = 15 →
    white_today = 11 →
    ∃ yellow_day_before_yesterday : ℕ,
      yellow_day_before_yesterday = white_yesterday + white_today :=
by sorry

end yellow_dandelions_day_before_yesterday_l526_526246


namespace samplingIsSystematic_l526_526759

open BigOperators

/-- Define what it means for an ID sequence to end in a specific digit. -/
def endsInFive (id : ℕ) : Prop := id % 10 = 5

/-- Define what it means for a selection to be systematic sampling. -/
def isSystematicSampling (f : ℕ → Prop) : Prop :=
  ∃ k : ℕ, k ≠ 0 ∧ ∀ n : ℕ, f (n * k + 5)

/-- Given condition: selection of students with ID ending in 5 from each class is used. -/
axiom selectionCondition : ∀ id : ℕ, id % 10 = 5 → True

/-- The main theorem stating that the given sampling method is indeed systematic sampling. -/
theorem samplingIsSystematic : isSystematicSampling (λ id, id % 10 = 5) :=
  sorry

end samplingIsSystematic_l526_526759


namespace bus_stop_time_l526_526213

theorem bus_stop_time (speed_without_stoppages speed_with_stoppages : ℝ) (h1: speed_without_stoppages = 48) (h2: speed_with_stoppages = 24) :
  ∃ (minutes_stopped_per_hour : ℝ), minutes_stopped_per_hour = 30 :=
by
  sorry

end bus_stop_time_l526_526213


namespace count_prime_divisors_2310_l526_526397

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_prime_divisors_2310 :
  ∃ (S : Finset ℕ), (∀ x ∈ S, is_prime x ∧ x ∣ 2310) ∧ S.card = 5 :=
begin
  sorry
end

end count_prime_divisors_2310_l526_526397


namespace intersecting_lines_parallel_to_plane_perpendicular_to_third_line_l526_526001

variables (l m n : ℝ)
variables (α β : Plane)

theorem intersecting_lines_parallel_to_plane_perpendicular_to_third_line
  (intersect : l ≠ m)
  (l_parallel_alpha : ∀ P, P ∈ l → P ∈ α)
  (m_parallel_alpha : ∀ P, P ∈ m → P ∈ α)
  (n_perpendicular_l : ∀ P, P ∈ n → ∀ Q, Q ∈ l → P = Q → false)
  (n_perpendicular_m : ∀ P, P ∈ n → ∀ Q, Q ∈ m → P = Q → false) :
  ∀ P, P ∈ n → ∀ Q, Q ∈ α → P = Q → false :=
by
  sorry

end intersecting_lines_parallel_to_plane_perpendicular_to_third_line_l526_526001


namespace right_angled_triangle_l526_526411

variable {α : Type*} [InnerProductSpace ℝ α]

theorem right_angled_triangle {A B C : α}
  (h : ⟪B - A, B - A - (C - A)⟫ = 0) :
  ∃ D E : α, is_right_triangle A B C :=
begin
  sorry
end

end right_angled_triangle_l526_526411


namespace sum_first_100_even_integers_l526_526409

theorem sum_first_100_even_integers : 
  let x := (100 / 2) * (2 + 200) in
  x = 10100 :=
by
  sorry

end sum_first_100_even_integers_l526_526409


namespace no_odd_m_solution_l526_526146

theorem no_odd_m_solution : ∀ (m n : ℕ), 0 < m → 0 < n → (5 * n = m * n - 3 * m) → ¬ Odd m :=
by
  intros m n hm hn h_eq
  sorry

end no_odd_m_solution_l526_526146


namespace right_pans_weight_count_ge_left_l526_526080

theorem right_pans_weight_count_ge_left (L R : Multiset ℕ) (h1 : ∀ (x ∈ L), ∃ (k : ℕ), x = 2^k) 
  (h2 : L.sum = R.sum) (h3 : ∀ (x ∈ (L : Multiset ℕ)), ∀ (y ∈ (L : Multiset ℕ)), x ≠ y) : 
  L.card ≤ R.card := 
by
  sorry

end right_pans_weight_count_ge_left_l526_526080


namespace statement_A_statement_B_statement_D_l526_526897

variable {A B C a b c : ℝ} [triangle : A + B + C = π]

-- Statement A: If A > B, then sin A > sin B
theorem statement_A (h1 : A > B) : sin A > sin B := sorry

-- Statement B: If cos A / a = cos B / b = cos C / c, then triangle ABC is equilateral
theorem statement_B (h2 : cos A / a = cos B / b ∧ cos B / b = cos C / c) : A = B ∧ B = C := sorry

-- Statement D: If b cos C + c cos B = a sin A, then triangle ABC is a right triangle
theorem statement_D (h3 : b * cos C + c * cos B = a * sin A) : A = π / 2 ∨ B = π / 2 ∨ C = π / 2 := sorry

end statement_A_statement_B_statement_D_l526_526897


namespace total_hooligans_l526_526966

def hooligans_problem (X Y : ℕ) : Prop :=
  (X * Y = 365) ∧ (X + Y = 78 ∨ X + Y = 366)

theorem total_hooligans (X Y : ℕ) (h : hooligans_problem X Y) : X + Y = 78 ∨ X + Y = 366 :=
  sorry

end total_hooligans_l526_526966


namespace y2_minus_x2_l526_526887

theorem y2_minus_x2 (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (h1 : 56 ≤ x + y) (h2 : x + y ≤ 59) (h3 : 9 < 10 * x) (h4 : 10 * x < 91 * y) : y^2 - x^2 = 177 :=
by
  sorry

end y2_minus_x2_l526_526887


namespace AI_length_l526_526111

open Real

variables {A B C I D : Type} [IsTriangle A B C]
variables (R r : ℝ) (α : ℝ)
variables (incircle : Incenter A B C = I) (circumcircle : ∀ {D}, OnCircumcircle A B C D → OnSegment A I D)

theorem AI_length (α : ℝ) (r : ℝ) : AI = r / sin (α / 2) :=
sorry

end AI_length_l526_526111


namespace integral_f_l526_526031

noncomputable def f (x : ℝ) : ℝ := Real.exp (|x|)

theorem integral_f :
  ∫ x in -2..4, f x = Real.exp 4 + Real.exp 2 - 2 :=
by
  sorry

end integral_f_l526_526031


namespace triangle_area_l526_526671

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a + b > c) (h3 : a + c > b) (h4 : b + c > a) : 
  a = 3 ∧ b = 4 ∧ c = 5 ∨ a = 4 ∧ b = 3 ∧ c = 5 ∨ a = 5 ∧ b = 4 ∧ c = 3 ∨
  a = 5 ∧ b = 3 ∧ c = 4 ∨ a = 4 ∧ b = 5 ∧ c = 3 ∨ a = 3 ∧ b = 5 ∧ c = 4 → 
  (1 / 2 : ℝ) * ↑a * ↑b = 6 := by
  sorry

end triangle_area_l526_526671


namespace total_worth_is_correct_l526_526522

-- Define the conditions
def rows : ℕ := 4
def gold_bars_per_row : ℕ := 20
def worth_per_gold_bar : ℕ := 20000

-- Define the calculated values
def total_gold_bars : ℕ := rows * gold_bars_per_row
def total_worth_of_gold_bars : ℕ := total_gold_bars * worth_per_gold_bar

-- Theorem statement to prove the correct total worth
theorem total_worth_is_correct : total_worth_of_gold_bars = 1600000 := by
  sorry

end total_worth_is_correct_l526_526522


namespace max_sum_of_squares_l526_526753

open Real

theorem max_sum_of_squares 
  (a : ℝ) (α : ℝ) 
  (hα : 0 < α ∧ α < π / 2) -- α is an acute angle
  -- Assuming a valid acute-give triangle with sides a, b, c with given angle α
  : ∃ b c, b ^ 2 + c ^ 2 ≤ a ^ 2 / (2 * sin (α / 2) ^ 2) ∧
    (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos α) := 
sorry

end max_sum_of_squares_l526_526753


namespace radius_of_tangent_circles_l526_526588

theorem radius_of_tangent_circles (r : ℝ) :
  let ellipse_eq := ∀ x y : ℝ, x^2 + 4 * y^2 = 8
  let circle_eq := ∀ x y : ℝ, (x - r)^2 + y^2 = r^2
  (∀ x y : ℝ, ellipse_eq x y → circle_eq x y) →
  r = (Real.sqrt 6) / 2 := sorry

end radius_of_tangent_circles_l526_526588


namespace cube_value_proportional_l526_526653

theorem cube_value_proportional (side_length1 side_length2 : ℝ) (volume1 volume2 : ℝ) (value1 value2 : ℝ) :
  side_length1 = 4 → volume1 = side_length1 ^ 3 → value1 = 500 →
  side_length2 = 6 → volume2 = side_length2 ^ 3 → value2 = value1 * (volume2 / volume1) →
  value2 = 1688 :=
by
  sorry

end cube_value_proportional_l526_526653


namespace f_of_f_4_l526_526373

noncomputable def f : ℝ → ℝ 
| x => if x > 2 then log 2 x else 2 ^ x - 1

theorem f_of_f_4 : f (f 4) = 3 := by
  sorry

end f_of_f_4_l526_526373


namespace fraction_difference_is_correct_l526_526720

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l526_526720


namespace arithmetic_sequence_fifth_term_l526_526408

variable (a d : ℕ)

-- Conditions
def condition1 := (a + d) + (a + 3 * d) = 10
def condition2 := a + (a + 2 * d) = 8

-- Fifth term calculation
def fifth_term := a + 4 * d

theorem arithmetic_sequence_fifth_term (h1 : condition1 a d) (h2 : condition2 a d) : fifth_term a d = 7 :=
by
  sorry

end arithmetic_sequence_fifth_term_l526_526408


namespace count_positive_integers_satisfying_inequality_l526_526925

theorem count_positive_integers_satisfying_inequality :
  { n : ℕ | 0 < n ∧ 1 + sqrt (n^2 - 9 * n + 20) > sqrt (n^2 - 7 * n + 12) }.card = 4 :=
sorry

end count_positive_integers_satisfying_inequality_l526_526925


namespace find_scalar_t_l526_526358

-- Define the problem statement
theorem find_scalar_t
  (a b : Vector ℝ 3) -- Let a and b be vectors in 3D space
  (h1 : ¬ is_collinear ℝ (Set.insert 0 (Set.insert a (Set.singleton b)))) 
  (h2 : ∃ k : ℝ, a - t • b = k • (2 • a + b)) : t = -1 / 2 := 
sorry

end find_scalar_t_l526_526358


namespace odd_function_value_l526_526995

theorem odd_function_value (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_fx : ∀ x : ℝ, x ≤ 0 → f x = 2 * x ^ 2 - x) :
  f 1 = -3 := 
sorry

end odd_function_value_l526_526995


namespace trapezium_area_l526_526779

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem trapezium_area :
  (1/2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
  sorry

end trapezium_area_l526_526779


namespace compare_a_b_c_l526_526813

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b :=
by
  have : c < b := sorry,
  have : a < b := sorry,
  have : a > c := sorry,
  exact ⟨this.2, this.1⟩

end compare_a_b_c_l526_526813


namespace fraction_difference_l526_526694

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l526_526694


namespace find_value_l526_526801

theorem find_value (x y: ℤ) (h1: 3^x = 27^(y + 1)) (h2: 16^y = 2^(x - 8)) : 2 * x + y = -29 := by
  sorry

end find_value_l526_526801


namespace sequence_converges_to_idempotent_l526_526479

variable {n : Type*} [Fintype n] [DecidableEq n]

noncomputable def A (R : Type*) [Ring R] := Matrix n n R

theorem sequence_converges_to_idempotent (A : A ℝ) (k : ℕ)
  (hA : 3 • (A * A * A) = A * A + A + 1) :
  ∃ (B : A ℝ), B * B = B ∧ ∀ ε > 0, ∃ N : ℕ, ∀ m ≥ N, ∥ (A ^ m) - B ∥ < ε :=
  sorry

end sequence_converges_to_idempotent_l526_526479


namespace sum_possible_values_k_l526_526467

open Nat

theorem sum_possible_values_k (j k : ℕ) (h : (1 / j : ℚ) + 1 / k = 1 / 4) : 
  ∃ ks : List ℕ, (∀ k' ∈ ks, ∃ j', (1 / j' : ℚ) + 1 / k' = 1 / 4) ∧ ks.sum = 51 :=
by
  sorry

end sum_possible_values_k_l526_526467


namespace factor_difference_of_squares_l526_526766

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l526_526766


namespace area_enclosed_by_circle_l526_526606

theorem area_enclosed_by_circle :
  let f := fun x y => (x - 1)^2 + (y - 1)^2 - x - y in
  (∀ x y, f x y = 0 → (x - 1)^2 + (y - 1)^2 = x + y) →
  let radius := (1 : ℝ) / (2 : ℝ).sqrt in
  let area := Real.pi * radius^2 in
  area = (Real.pi) / 2 := by
  sorry

end area_enclosed_by_circle_l526_526606


namespace yellow_dandelions_day_before_yesterday_l526_526247

theorem yellow_dandelions_day_before_yesterday :
  ∀ (yellow_yesterday white_yesterday yellow_today white_today : ℕ),
    yellow_yesterday = 20 →
    white_yesterday = 14 →
    yellow_today = 15 →
    white_today = 11 →
    ∃ yellow_day_before_yesterday : ℕ,
      yellow_day_before_yesterday = white_yesterday + white_today :=
by sorry

end yellow_dandelions_day_before_yesterday_l526_526247


namespace continuous_polynomial_continuous_cosecant_l526_526527

-- Prove that the function \( f(x) = 2x^2 - 1 \) is continuous on \(\mathbb{R}\)
theorem continuous_polynomial : Continuous (fun x : ℝ => 2 * x^2 - 1) :=
sorry

-- Prove that the function \( g(x) = (\sin x)^{-1} \) is continuous on \(\mathbb{R}\) \setminus \(\{ k\pi \mid k \in \mathbb{Z} \} \)
theorem continuous_cosecant : ∀ x : ℝ, x ∉ Set.range (fun k : ℤ => k * Real.pi) → ContinuousAt (fun x : ℝ => (Real.sin x)⁻¹) x :=
sorry

end continuous_polynomial_continuous_cosecant_l526_526527


namespace binom_7_2_eq_21_l526_526297

open Nat

theorem binom_7_2_eq_21 : binomial 7 2 = 21 := 
by sorry

end binom_7_2_eq_21_l526_526297


namespace distance_from_center_to_point_l526_526284

noncomputable def center_of_circle : ℝ × ℝ :=
  let x := 4
  let y := -5
  (x, y)

def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 8 * x - 10 * y + 20

def point : ℝ × ℝ := (-3, 2)

def distance_between_points (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_from_center_to_point : distance_between_points center_of_circle point = 7 * Real.sqrt 2 :=
by
  sorry

end distance_from_center_to_point_l526_526284


namespace part_one_part_two_part_three_l526_526348

open Real

-- Define the sequence according to the given conditions
def seq (n : ℕ) : ℝ :=
  if n = 1 then -1
  else if n = 2 then 1
  else if (n-1) % 2 = 0 then (2 + (-1)^(n-3)) / 2 * seq (n-2)
  else (2 + (-1)^(n-2)) / 2 * seq (n-2)

-- Part 1: Prove a_5 + a_6 = 2
theorem part_one : seq 5 + seq 6 = 2 := by
  sorry

-- Define S_n (sum of the first n terms of the sequence)
def S (n : ℕ) : ℝ :=
  (sum (finset.range n).map seq)

-- Part 2: Prove S_n expression
theorem part_two : ∀ n : ℕ, 
  (if n % 2 = 0 then S n = 2 * (3/2)^(n/2) + 2 * (1/2)^(n/2) - 4 
  else S n = 3 * (3/2)^((n-1)/2) + 2 * (1/2)^((n-1)/2) - 4) := by
  sorry

-- Define b_n = a_{2n-1} + a_{2n}
def b (n : ℕ) : ℝ := seq (2 * n - 1) + seq (2 * n)

-- Part 3: Prove the unique solution for (i, j, k)
theorem part_three : ∃ (i j k : ℕ), i < j ∧ j < k ∧ b i = 2 * b j - b k := by
  exact ⟨1, 2, 3, by decide⟩

end part_one_part_two_part_three_l526_526348


namespace taxi_fare_l526_526172

theorem taxi_fare (x : ℝ) (h : x > 3) : 
  let starting_price := 6
  let additional_fare_per_km := 1.4
  let fare := starting_price + additional_fare_per_km * (x - 3)
  fare = 1.4 * x + 1.8 :=
by
  sorry

end taxi_fare_l526_526172


namespace renovation_cost_distribution_l526_526242

/-- A mathematical proof that if Team A works alone for 3 weeks, followed by both Team A and Team B working together, and the total renovation cost is 4000 yuan, then the payment should be distributed equally between Team A and Team B, each receiving 2000 yuan. -/
theorem renovation_cost_distribution :
  let time_A := 18
  let time_B := 12
  let initial_time_A := 3
  let total_cost := 4000
  ∃ x, (1 / time_A * (x + initial_time_A) + 1 / time_B * x = 1) ∧
       let work_A := 1 / time_A * (x + initial_time_A)
       let work_B := 1 / time_B * x
       work_A = work_B ∧
       total_cost / 2 = 2000 :=
by
  sorry

end renovation_cost_distribution_l526_526242


namespace kevin_fifteenth_finger_l526_526556

-- Define the function f as given by the conditions
def f : ℕ → ℕ 
| 5 := 4
| 4 := 3
| 3 := 6
| 6 := 5
| _ := 0  -- Default case, not needed for this proof

-- Define the sequence generated by repeatedly applying f starting from 5
noncomputable def sequence : ℕ → ℕ 
| 0 := 5
| (n + 1) := f (sequence n)

-- The theorem to prove that the 15th term in the sequence is 3
theorem kevin_fifteenth_finger : sequence 14 = 3 :=
sorry  -- Proof not required

end kevin_fifteenth_finger_l526_526556


namespace min_translation_symmetry_l526_526361

theorem min_translation_symmetry (a : ℝ) (varphi : ℝ) (h1 : ∀ x, f(x) = a * sin x + cos x)
  (h2 : x = π/4) (h3 : varphi > 0)
  (h4 : ∀ x, f(x + varphi) = f(-x)) :
  varphi = 3 * π / 4 := sorry

end min_translation_symmetry_l526_526361


namespace sum_D_1_to_200_l526_526105

def D (n : ℕ) : ℕ :=
  n.digits.sum.filter (λ d, d % 3 = 0).sum

theorem sum_D_1_to_200 : (∑ n in finset.range 201, D n) = 1080 := 
sorry

end sum_D_1_to_200_l526_526105


namespace probability_of_10_heads_in_12_flips_l526_526598

open_locale big_operators

noncomputable def calculate_probability : ℕ → ℕ → ℚ := 
  λ n k, (nat.choose n k : ℚ) / (2 ^ n)

theorem probability_of_10_heads_in_12_flips :
  calculate_probability 12 10 = 66 / 4096 :=
by
  sorry

end probability_of_10_heads_in_12_flips_l526_526598


namespace problem1_l526_526639

theorem problem1 
  (h0 : ∫ x in 0..(real.pi / 2), real.sqrt 2 * real.sin (x + real.pi / 4) = 2) : 
  let m := 2 in 
  binomial_coefficient 6 2 * (-2) ^ 2 = 60 := 
by sorry

end problem1_l526_526639


namespace carolyn_fewer_stickers_l526_526281

theorem carolyn_fewer_stickers :
  let belle_stickers := 97
  let carolyn_stickers := 79
  carolyn_stickers < belle_stickers →
  belle_stickers - carolyn_stickers = 18 :=
by
  intros
  sorry

end carolyn_fewer_stickers_l526_526281


namespace max_possible_percentage_l526_526227

theorem max_possible_percentage (p_wi : ℝ) (p_fs : ℝ) (h_wi : p_wi = 0.4) (h_fs : p_fs = 0.7) :
  ∃ p_both : ℝ, p_both = min p_wi p_fs ∧ p_both = 0.4 :=
by
  sorry

end max_possible_percentage_l526_526227


namespace first_student_time_l526_526223

-- Define the conditions
def num_students := 4
def avg_last_three := 35
def avg_all := 30
def total_time_all := num_students * avg_all
def total_time_last_three := (num_students - 1) * avg_last_three

-- State the theorem
theorem first_student_time : (total_time_all - total_time_last_three) = 15 :=
by
  -- Proof is skipped
  sorry

end first_student_time_l526_526223


namespace non_congruent_squares_6x6_grid_l526_526396

def is_square {α : Type} [linear_ordered_field α] (a b c d : (α × α)) : Prop :=
-- A function to check if four points form a square (not implemented here)
sorry

def count_non_congruent_squares (n : ℕ) : ℕ :=
-- Place calculations for counting non-congruent squares on an n x n grid (not implemented here)
sorry

theorem non_congruent_squares_6x6_grid :
  count_non_congruent_squares 6 = 128 :=
sorry

end non_congruent_squares_6x6_grid_l526_526396


namespace repeating_seventy_two_exceeds_seventy_two_l526_526741

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l526_526741


namespace sum_of_possible_ks_l526_526428

theorem sum_of_possible_ks : 
  (∃ (j k : ℕ), (1 < j) ∧ (1 < k) ∧ j ≠ k ∧ ((1/j : ℝ) + (1/k : ℝ) = (1/4))) → 
  (∑ k in {20, 12, 8, 6, 5}, k) = 51 :=
begin
  sorry
end

end sum_of_possible_ks_l526_526428


namespace find_f_prime_at_2_l526_526368

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x * f' 2

theorem find_f_prime_at_2 (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = 3 * x^2 + 2 * x * (f' 2)) :
  f' 2 = -12 :=
by {
  -- Definition of f
  sorry
}

end find_f_prime_at_2_l526_526368


namespace prove_total_weekly_allowance_l526_526230

noncomputable def total_weekly_allowance : ℕ :=
  let students := 200
  let group1 := students * 45 / 100
  let group2 := students * 30 / 100
  let group3 := students * 15 / 100
  let group4 := students - group1 - group2 - group3  -- Remaining students
  let daily_allowance := group1 * 6 + group2 * 4 + group3 * 7 + group4 * 10
  daily_allowance * 7

theorem prove_total_weekly_allowance :
  total_weekly_allowance = 8330 := by
  sorry

end prove_total_weekly_allowance_l526_526230


namespace molecular_weight_is_correct_l526_526305

-- Define the masses of the individual isotopes
def H1 : ℕ := 1
def H2 : ℕ := 2
def O : ℕ := 16
def C : ℕ := 13
def N : ℕ := 15
def S : ℕ := 33

-- Define the molecular weight calculation
def molecular_weight : ℕ := (2 * H1) + H2 + O + C + N + S

-- The goal is to prove that the calculated molecular weight is 81
theorem molecular_weight_is_correct : molecular_weight = 81 :=
by 
  sorry

end molecular_weight_is_correct_l526_526305


namespace correct_answer_l526_526772

-- Define the given sentence structure with placeholders
def sentence_structure (a b : String) :=
  "Farmers have benefited from " ++ a ++ " is called a water buffalo bank " ++ b ++ " a water buffalo results in every farmer owning one and harvesting more with less human labor."

-- Choices for the blanks
def choice1 := ("what", "where")
def choice2 := ("that", "which")
def choice3 := ("which", "which")
def choice4 := ("which", "where")

-- Correct Answer
def correct_choice := ("what", "where")

-- Statement to prove that correct_choice correctly fills the blanks
theorem correct_answer : sentence_structure (fst correct_choice) (snd correct_choice) =
                         "Farmers have benefited from what is called a water buffalo bank where a water buffalo results in every farmer owning one and harvesting more with less human labor." :=
by
  rfl -- The proof is trivial because correct_choice directly results in the sentence.


end correct_answer_l526_526772


namespace simplify_polynomials_l526_526145

theorem simplify_polynomials (x : ℝ) :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 3 * x - 15) = x^2 + 5 * x + 10 :=
by 
  sorry

end simplify_polynomials_l526_526145


namespace range_of_reciprocal_sum_l526_526049

theorem range_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
    4 ≤ (1/x + 1/y) :=
by
  sorry

end range_of_reciprocal_sum_l526_526049


namespace compare_a_b_c_l526_526818

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b :=
by
  have : c < b := sorry,
  have : a < b := sorry,
  have : a > c := sorry,
  exact ⟨this.2, this.1⟩

end compare_a_b_c_l526_526818


namespace intersection_Q_l526_526472

-- Given conditions
variables {A B C G H Q : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables [AddCommGroup G] [AddCommGroup H] [AddCommGroup Q]
variables [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ G] [Module ℝ H] [Module ℝ Q]

-- Defining points G and H
def point_G (A B : A) : A := (2/5) • A + (3/5) • B
def point_H (B C : B) : B := (3/4) • B + (1/4) • C

-- The theorem statement
theorem intersection_Q (A B C : Q) :
  let G := point_G A B in
  let H := point_H B C in
  ∃ Q, Q = (2/7) • A + (27/70) • B + (1/14) • C :=
sorry

end intersection_Q_l526_526472


namespace axis_of_symmetry_sin_l526_526157

theorem axis_of_symmetry_sin : 
  ∃ (x : ℝ), f x = sin (2 * x + π / 3) ∧ (∃ k : ℤ, x = k * π / 2 - π / 12) :=
by
  -- Define the function formally
  let f : ℝ → ℝ := (λ x, sin (2 * x + π / 3))
  -- State the axis of symmetry equation
  have symmetry : ∃ x : ℝ, ∃ k : ℤ, x = k * π / 2 - π / 12
  { use π / 12,
    use 0,
    ring, }
  use π / 12,
  split,
  { rw symmetry, },
  { exact symmetry, }
  sorry

end axis_of_symmetry_sin_l526_526157


namespace translated_line_tangent_to_curve_l526_526180

theorem translated_line_tangent_to_curve (x y λ : ℝ)
  (h1 : x - 2 * y + λ - 3 = 0)
  (h2 : x^2 + y^2 + 2 * x - 4 * y = 0) :
  λ = 13 ∨ λ = 3 :=
sorry

end translated_line_tangent_to_curve_l526_526180


namespace factorize_3x_squared_minus_7x_minus_6_factorize_6x_squared_minus_7x_minus_5_l526_526313

theorem factorize_3x_squared_minus_7x_minus_6 (x : ℝ) :
  3 * x^2 - 7 * x - 6 = (x - 3) * (3 * x + 2) :=
sorry

theorem factorize_6x_squared_minus_7x_minus_5 (x : ℝ) :
  6 * x^2 - 7 * x - 5 = (2 * x + 1) * (3 * x - 5) :=
sorry

end factorize_3x_squared_minus_7x_minus_6_factorize_6x_squared_minus_7x_minus_5_l526_526313


namespace find_xyz_l526_526498

theorem find_xyz
  (a b c x y z : ℂ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0)
  (h7 : a = (b + c) / (x - 3))
  (h8 : b = (a + c) / (y - 3))
  (h9 : c = (a + b) / (z - 3))
  (h10 : x * y + x * z + y * z = 10)
  (h11 : x + y + z = 6) :
  x * y * z = 10 :=
sorry

end find_xyz_l526_526498


namespace converse_of_polygon_angle_equality_imp_parallel_sides_l526_526953

theorem converse_of_polygon_angle_equality_imp_parallel_sides
    (P : Type) [polygon P] (h_convex: convex P) 
    (h_angles : ∀ A B C D : P, ↔ ∠ A = ∠ C ∧ ∠ B = ∠ D) :
    (∀ A B C D : P, A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A → (AB ∥ CD ∧ AD ∥ BC)) :=
begin
  sorry
end

end converse_of_polygon_angle_equality_imp_parallel_sides_l526_526953


namespace binom_7_2_eq_21_l526_526296

open Nat

theorem binom_7_2_eq_21 : binomial 7 2 = 21 := 
by sorry

end binom_7_2_eq_21_l526_526296


namespace compare_constants_l526_526866

noncomputable def a : ℝ := 0.1 * real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - real.log 0.9

theorem compare_constants : c < a ∧ a < b :=
by sorry

end compare_constants_l526_526866


namespace polynomial_inequality_l526_526112

variables {R : Type*} [ordered_ring R] [linear_ordered_field R]

theorem polynomial_inequality (P : polynomial R) (n : ℕ)
  (h_monic : P.monic)
  (h_degree : P.degree = n)
  (h_roots : ∀ x : R, is_root P x → ∃ r : R, x = -r ∧ r > 0)
  (h_P0 : P.eval 0 = 1) :
  P.eval 2 ≥ 3^n :=
sorry

end polynomial_inequality_l526_526112


namespace time_for_trains_to_cross_each_other_in_same_direction_l526_526591

-- Definitions and conditions
def length_train1 : ℝ := 220
def length_train2 : ℝ := 250
def time_train1_crossing_man : ℝ := 12
def time_train2_crossing_man : ℝ := 18

-- Speeds of the trains
def speed_train1 : ℝ := length_train1 / time_train1_crossing_man
def speed_train2 : ℝ := length_train2 / time_train2_crossing_man

-- Relative speed when trains move in the same direction
def relative_speed : ℝ := speed_train1 - speed_train2

-- Total length to be covered when crossing each other
def total_length : ℝ := length_train1 + length_train2

-- Theorem statement
theorem time_for_trains_to_cross_each_other_in_same_direction : total_length / relative_speed = 105.86 := by
  sorry

end time_for_trains_to_cross_each_other_in_same_direction_l526_526591


namespace equal_amounts_of_tea_and_milk_l526_526582

variable (V : Type) [Inhabited V] [Field V]
variable (milk tea : V)
variable (v : V) -- Assuming v is the volume of one spoonful

def pour_milk_to_tea (milk_volume tea_volume : V) (spoonfuls : V) : V :=
  tea_volume + spoonfuls

def pour_mix_back_to_milk (milk_volume tea_volume : V) (spoonfuls : V) : (V × V) :=
  let mixed_volume := tea_volume * spoonfuls / (tea_volume + spoonfuls)
  (milk_volume - mixed_volume, tea_volume - mixed_volume)

theorem equal_amounts_of_tea_and_milk (milk_volume tea_volume : V) (spoonfuls : V) :
  ∀ (mixture_in_tea cup_mixture: V), 
  cup_mixture = pour_milk_to_tea milk_volume tea_volume spoonfuls →
  let (new_milk_volume, new_tea_volume) := pour_mix_back_to_milk milk_volume tea_volume spoonfuls in
    new_milk_volume + new_tea_volume = milk_volume + tea_volume :=
  sorry

end equal_amounts_of_tea_and_milk_l526_526582


namespace minimum_total_votes_l526_526959

theorem minimum_total_votes (V : ℕ) 
  (A_votes : Float) (B_votes : Float) (C_votes : Float) (D_votes : Float)
  (margin : Float)
  (hA: A_votes = 0.35 * V) 
  (hB: B_votes = 0.25 * V) 
  (hC: C_votes = 0.20 * V) 
  (hD: D_votes = 0.20 * V)
  (h_margin: hA - hB = 500) :
  V = 5000 := 
by
  sorry

end minimum_total_votes_l526_526959


namespace cider_production_l526_526385

theorem cider_production (gd_pint : ℕ) (pl_pint : ℕ) (gs_pint : ℕ) (farmhands : ℕ) (gd_rate : ℕ) (pl_rate : ℕ) (gs_rate : ℕ) (work_hours : ℕ) 
  (gd_total : ℕ) (pl_total : ℕ) (gs_total : ℕ) (gd_ratio : ℕ) (pl_ratio : ℕ) (gs_ratio : ℕ) 
  (gd_pint_val : gd_pint = 20) (pl_pint_val : pl_pint = 40) (gs_pint_val : gs_pint = 30)
  (farmhands_val : farmhands = 6) (gd_rate_val : gd_rate = 120) (pl_rate_val : pl_rate = 240) (gs_rate_val : gs_rate = 180) 
  (work_hours_val : work_hours = 5) 
  (gd_total_val : gd_total = farmhands * work_hours * gd_rate) 
  (pl_total_val : pl_total = farmhands * work_hours * pl_rate) 
  (gs_total_val : gs_total = farmhands * work_hours * gs_rate) 
  (gd_ratio_val : gd_ratio = 1) (pl_ratio_val : pl_ratio = 2) (gs_ratio_val : gs_ratio = 3/2) 
  (ratio_condition : gd_total / gd_ratio = pl_total / pl_ratio ∧ pl_total / pl_ratio = gs_total / gs_ratio) : 
  (gd_total / gd_pint) = 180 := 
sorry

end cider_production_l526_526385


namespace bridge_weight_requirement_l526_526982

def weight_soda_can : ℕ := 12
def weight_empty_soda_can : ℕ := 2
def num_soda_cans : ℕ := 6

def weight_empty_other_can : ℕ := 3
def num_other_cans : ℕ := 2

def wind_force_eq_soda_cans : ℕ := 2

def total_weight_bridge_must_hold : ℕ :=
  weight_soda_can * num_soda_cans + weight_empty_soda_can * num_soda_cans +
  weight_empty_other_can * num_other_cans +
  wind_force_eq_soda_cans * (weight_soda_can + weight_empty_soda_can)

theorem bridge_weight_requirement :
  total_weight_bridge_must_hold = 118 :=
by
  unfold total_weight_bridge_must_hold weight_soda_can weight_empty_soda_can num_soda_cans
    weight_empty_other_can num_other_cans wind_force_eq_soda_cans
  sorry

end bridge_weight_requirement_l526_526982


namespace average_lifespan_of_sampled_products_l526_526241

theorem average_lifespan_of_sampled_products:
  let n1 := 25
  let n2 := 50
  let n3 := 25
  let lifespan1 := 980
  let lifespan2 := 1020
  let lifespan3 := 1032
  let total_samples := 100
  let total_lifespan := (n1 * lifespan1) + (n2 * lifespan2) + (n3 * lifespan3)
  let average_lifespan := total_lifespan / total_samples
  average_lifespan = 1013 :=
by
  -- Definitions and calculations according to the problem conditions
  let n1 := 25
  let n2 := 50
  let n3 := 25
  let lifespan1 := 980
  let lifespan2 := 1020
  let lifespan3 := 1032
  let total_samples := 100
  let total_lifespan := (n1 * lifespan1) + (n2 * lifespan2) + (n3 * lifespan3)
  let average_lifespan := total_lifespan / total_samples
  -- Assert the average lifespan equality
  have h1 : total_lifespan = 24500 + 51000 + 25800 := by sorry
  have h2 : total_lifespan = 101300 := by sorry
  have h3 : average_lifespan = 101300 / 100 := by sorry
  show average_lifespan = 1013, from h3


end average_lifespan_of_sampled_products_l526_526241


namespace smallest_positive_period_of_f_l526_526115

variables {R : Type*} [OrderedAddCommGroup R] [TopologicalSpace R]
variables {f g : R → R} {c : R}

theorem smallest_positive_period_of_f (f_even : ∀ x, f (x) = f (-x))
  (g_odd : ∀ x, g (x) = -g (-x))
  (f_eq_neg_g_shift : ∀ x, f x = -g (x + c))
  (c_pos : c > 0) :
  ∀ x, f x = f (x + 4 * c) :=
by
  sorry

end smallest_positive_period_of_f_l526_526115


namespace repeatingDecimal_exceeds_l526_526701

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l526_526701


namespace sum_not_equals_any_l526_526578

-- Define the nine special natural numbers a1 to a9
def a1 (k : ℕ) : ℕ := (10^k - 1) / 9
def a2 (m : ℕ) : ℕ := 2 * (10^m - 1) / 9
def a3 (p : ℕ) : ℕ := 3 * (10^p - 1) / 9
def a4 (q : ℕ) : ℕ := 4 * (10^q - 1) / 9
def a5 (r : ℕ) : ℕ := 5 * (10^r - 1) / 9
def a6 (s : ℕ) : ℕ := 6 * (10^s - 1) / 9
def a7 (t : ℕ) : ℕ := 7 * (10^t - 1) / 9
def a8 (u : ℕ) : ℕ := 8 * (10^u - 1) / 9
def a9 (v : ℕ) : ℕ := 9 * (10^v - 1) / 9

-- Statement of the problem
theorem sum_not_equals_any (k m p q r s t u v : ℕ) :
  ¬ (a1 k = a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a2 m = a1 k + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a3 p = a1 k + a2 m + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a4 q = a1 k + a2 m + a3 p + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a5 r = a1 k + a2 m + a3 p + a4 q + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a6 s = a1 k + a2 m + a3 p + a4 q + a5 r + a7 t + a8 u + a9 v) ∧
  ¬ (a7 t = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a8 u + a9 v) ∧
  ¬ (a8 u = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a9 v) ∧
  ¬ (a9 v = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u) :=
  sorry

end sum_not_equals_any_l526_526578


namespace domain_R_range_a_domain_interval_value_a_l526_526160

section Problem1

variable {a : ℝ}

def f (x : ℝ) : ℝ := sqrt((1 - a^2) * x^2 + 3 * (1 - a) * x + 6)

theorem domain_R_range_a :
  (∀ x : ℝ, (1 - a^2) * x^2 + 3 * (1 - a) * x + 6 ≥ 0) ↔ (- 5 / 11 ≤ a ∧ a ≤ 1) := 
by 
  sorry

end Problem1

section Problem2

variable {a : ℝ}

def f (x : ℝ) : ℝ := sqrt((1 - a^2) * x^2 + 3 * (1 - a) * x + 6)

theorem domain_interval_value_a :
  (∀ x ∈ set.Icc (-2 : ℝ) 1, (1 - a^2) * x^2 + 3 * (1 - a) * x + 6 ≥ 0) ↔ (a = 2) :=
by 
  sorry

end Problem2

end domain_R_range_a_domain_interval_value_a_l526_526160


namespace james_initial_coins_l526_526979

noncomputable def initial_coins (initial_price new_fraction_increase coins_sold total_money_received : ℕ) := 
  total_money_received / (initial_price + (initial_price * new_fraction_increase))

theorem james_initial_coins :
  initial_coins 15 (2/3 : ℚ) 12 300 = 20 :=
by 
sorry

end james_initial_coins_l526_526979


namespace B_pow_2019_l526_526478

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![1 / 2, 0, -Real.sqrt 3 / 2],
    ![0, 1, 0],
    ![Real.sqrt 3 / 2, 0, 1 / 2]
  ]

theorem B_pow_2019 :
  B ^ 2019 = ![
    ![0, 0, -1],
    ![0, 1, 0],
    ![1, 0, 0]
  ] :=
by
  -- Proof is omitted
  sorry

end B_pow_2019_l526_526478


namespace repeating_seventy_two_exceeds_seventy_two_l526_526735

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l526_526735


namespace Bryan_did_258_pushups_l526_526686

-- Define the conditions
def sets : ℕ := 15
def pushups_per_set : ℕ := 18
def pushups_fewer_last_set : ℕ := 12

-- Define the planned total push-ups
def planned_total_pushups : ℕ := sets * pushups_per_set

-- Define the actual push-ups in the last set
def last_set_pushups : ℕ := pushups_per_set - pushups_fewer_last_set

-- Define the total push-ups Bryan did
def total_pushups : ℕ := (sets - 1) * pushups_per_set + last_set_pushups

-- The theorem to prove
theorem Bryan_did_258_pushups :
  total_pushups = 258 := by
  sorry

end Bryan_did_258_pushups_l526_526686


namespace gold_bars_total_worth_l526_526523

theorem gold_bars_total_worth :
  let rows := 4
  let bars_per_row := 20
  let worth_per_bar : ℕ := 20000
  let total_bars := rows * bars_per_row
  let total_worth := total_bars * worth_per_bar
  total_worth = 1600000 :=
by
  sorry

end gold_bars_total_worth_l526_526523


namespace construct_triangle_exists_l526_526592

noncomputable def constructs_triangle 
    (A B C C1 B1 : Point) 
    (AB CC1: Line) 
    (c h : ℝ) 
    (φ: ℝ) : Prop :=
    (dist A B = c) ∧
    (altitude C C1 AB = h) ∧
    (angle A B1 = φ) ∧
    (angle B1 B C = angle B A C)

theorem construct_triangle_exists 
    (A B C C1 B1 : Point) 
    (AB CC1: Line) 
    (c h φ : ℝ) 
    (h1: 0 < φ ∧ φ < π) 
    (hc: 0 < c) 
    (hh : 0 < h) :
    ∃ C, constructs_triangle A B C C1 B1 AB CC1 c h φ := 
sorry

end construct_triangle_exists_l526_526592


namespace cosine_neg_480_l526_526307

theorem cosine_neg_480 : real.cos (-480 * real.pi / 180) = -1 / 2 :=
by
  -- Using the property of cosine
  have h1 : real.cos (-480 * real.pi / 180) = real.cos (480 * real.pi / 180),
    from real.cos_neg (480 * real.pi / 180),
  -- Using the periodicity of cosine
  have h2 : real.cos (480 * real.pi / 180) = real.cos (120 * real.pi / 180),
    from real.cos_add_two_pi_mul (480 * real.pi / 180) (-(2 * real.pi)), -- Since 480 = 360 + 120,
  -- Using the special angle value
  have h3 : real.cos (120 * real.pi / 180) = -1 / 2,
    from real.cos_pi_div_three_sub_cos_pi,
  -- Combining the results
  rw [h1, h2, h3],
  sorry

end cosine_neg_480_l526_526307


namespace find_length_of_BC_l526_526089

noncomputable section

open Classical

variables {A B C M N D E F: Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variable (ΔABC : Triangle A B C)

-- Conditions
def angle_A_90 (Δ : Triangle A B C) : Prop :=
  Δ.angle A = 90

def AB_eq_AC (Δ : Triangle A B C) : Prop :=
  Δ.length B = Δ.length C

def midpoint_M (Δ : Triangle A B C) (M : A) : Prop :=
  isMidpoint M Δ.B Δ.C

def midpoint_N (Δ : Triangle A B C) (N : A) : Prop :=
  isMidpoint N Δ.A Δ.C

def point_on_segment_MN (Δ : Triangle A B C) (M N D : A) : Prop :=
  D ∈ segment M N ∧ D ≠ M ∧ D ≠ N

def extensions_intersect (B D C E A F : A): Prop :=
  intersects (lineThrough B D) E ∧ intersects (lineThrough C D) F ∧ E ∈ line C A ∧ F ∈ line A B

def condition (Δ : Triangle A B C) (B E : A) (F : A) : Prop :=
  1 / dist Δ.B E + 1 / dist Δ.C F = 3 / 4

-- The statement we need to prove
theorem find_length_of_BC : 
  ∀ (ΔABC : Triangle A B C) (AB_eq_AC : AB_eq_AC ΔABC) (k : A → B → Real)
  (angle_A_90 ΔABC) (midpoint_M) (N moidpoint_N) 
  (Extensions_intersect)(condition) 
  , ΔABC.length Δ.B Δ.C = 4 * Real.sqrt 2 := sorry

end find_length_of_BC_l526_526089


namespace differentiate_and_inequality_l526_526019

variable {f : ℝ → ℝ}
variable (a b : ℝ)

theorem differentiate_and_inequality
  (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
  (h_ineq : ∀ x : ℝ, x * (deriv f x) > - (f x))
  (h_ab : a > b) :
  a * f a > b * f b := 
sorry

end differentiate_and_inequality_l526_526019


namespace sequence_s_convergent_l526_526482

open Nat Real

def is_strictly_increasing (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n < seq (n + 1)

def sequence_s (a : ℕ → ℕ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range k, 1 / (Nat.lcm (a i) (a (i + 1)))

theorem sequence_s_convergent (a : ℕ → ℕ) (h : is_strictly_increasing a) :
  ∃ l, Filter.Tendsto (sequence_s a) Filter.atTop (Filter.tendsto_const_nhds l) :=
sorry

end sequence_s_convergent_l526_526482


namespace find_number_l526_526642

theorem find_number (x : ℝ) (h : 0.833 * x = -60) : x ≈ -72.02 :=
sorry

end find_number_l526_526642


namespace exists_triangle_with_sin_angles_l526_526518

theorem exists_triangle_with_sin_angles (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a^4 + b^4 + c^4 + 4*a^2*b^2*c^2 = 2 * (a^2*b^2 + a^2*c^2 + b^2*c^2)) : 
    ∃ (α β γ : ℝ), α + β + γ = Real.pi ∧ Real.sin α = a ∧ Real.sin β = b ∧ Real.sin γ = c :=
by
  sorry

end exists_triangle_with_sin_angles_l526_526518


namespace collinear_proof_l526_526344

theorem collinear_proof 
  (A B C I U V X Y W Z : Type) 
  [Triangle A B C] 
  [Incenter I A B C] 
  [Circumcircle Φ A B C] 
  (h1 : ⦃l : Line⦄, passes_through I l → perpendicular_to l (C, I))
  (h2 : l.intersect_segment BC U) 
  (h3 : l.intersect_arc BC V) 
  (h4 : ∀ (l' : Line), passes_through U l' ∧ parallel_to l' (A, I) → l'.intersect_segment AV X)
  (h5 : ∀v, passes_through V v ∧ parallel_to v (A, I) → v.intersect_segment AB Y)
  (midpoint_AX_W : midpoint W A X)
  (midpoint_BC_Z : midpoint Z B C)
  (collinear_I_X_Y : collinear I X Y) :
  collinear I W Z :=
sorry

end collinear_proof_l526_526344


namespace proof_problem_l526_526835

-- Define the constants a, b, c
def a := 0.1 * Real.exp 0.1
def b := 1 / 9
def c := -Real.log 0.9

-- State the theorem to prove
theorem proof_problem : c < a ∧ a < b := 
by
  sorry

end proof_problem_l526_526835


namespace parallelogram_area_l526_526743

variables (p q : ℝ^3)
variables a b : ℝ^3

-- Conditions
def a := 2 * p - 3 * q
def b := 3 * p + q
def norm_p : ℝ := 4
def norm_q : ℝ := 1
def angle_pq : ℝ := ∏/6

-- Statement to be proved
theorem parallelogram_area : |a × b| = 22 :=
by
  sorry

end parallelogram_area_l526_526743


namespace sum_of_squares_sines_l526_526287

theorem sum_of_squares_sines : ∑ k in finset.range 89, real.sin ((k + 1) * real.pi / 180) ^ 2 = 44.5 :=
by
  sorry

end sum_of_squares_sines_l526_526287


namespace tan_identity_cot_identity_l526_526136

-- Statement for the first part of the problem
theorem tan_identity (α β γ n : ℤ) (h : α + β + γ = n * π) : 
  tan α + tan β + tan γ = tan α * tan β * tan γ :=
sorry

-- Statement for the second part of the problem
theorem cot_identity (α β γ n : ℤ) (h : α + β + γ = n * π + π / 2) :
  cot α + cot β + cot γ = cot α * cot β * cot γ :=
sorry

end tan_identity_cot_identity_l526_526136


namespace vector_z_eq_a_value_l526_526363

section
variables {a : ℝ}

def z1 := 2 * a + complex.I * 6
def z2 := -1 + complex.I

def vector_z := z2 - z1

theorem vector_z_eq : vector_z = -1 - 2 * a - complex.I * 5 :=
by
  sorry

theorem a_value (H : -5 = (1/2) * (-1 - 2 * a)) : a = 4.5 :=
by
  sorry
end

end vector_z_eq_a_value_l526_526363


namespace ellipse_line_intersection_slope_l526_526371

theorem ellipse_line_intersection_slope (m n : ℝ) (A B : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : 0 < m) (h₂ : 0 < n)
  (h₃ : A = (x₁, y₁)) (h₄ : B = (x₂, y₂))
  (h₅ : m * x₁^2 + n * y₁^2 = 1) (h₆ : m * x₂^2 + n * y₂^2 = 1)
  (h₇ : y₁ = 1 - x₁) (h₈ : y₂ = 1 - x₂)
  (h_slope : (y₁ + y₂) / (x₁ + x₂) = sqrt 2 / 2):
  n / m = sqrt 2 := 
sorry

end ellipse_line_intersection_slope_l526_526371


namespace vision_approximation_l526_526538

noncomputable
def five_point_to_decimal (L: ℝ) : ℝ := 10 ^ (L - 5)

theorem vision_approximation : 
  ∀ L : ℝ, L = 4.9 → (five_point_to_decimal L) ≈ 0.8 :=
by
  intros L hL
  rw hL
  sorry

end vision_approximation_l526_526538


namespace sum_of_x_values_l526_526199

theorem sum_of_x_values (x : ℝ) (h : Real.sqrt ((x + 5)^2) = 9) :
  {x | Real.sqrt ((x + 5)^2) = 9}.sum = -10 :=
by
  sorry

end sum_of_x_values_l526_526199


namespace number_of_students_l526_526413

variables (T S n : ℕ)

-- 1. The teacher's age is 24 years more than the average age of the students.
def condition1 : Prop := T = S / n + 24

-- 2. The teacher's age is 20 years more than the average age of everyone present.
def condition2 : Prop := T = (T + S) / (n + 1) + 20

-- Proving that the number of students in the classroom is 5 given the conditions.
theorem number_of_students (h1 : condition1 T S n) (h2 : condition2 T S n) : n = 5 :=
by sorry

end number_of_students_l526_526413


namespace final_speed_train_l526_526644

theorem final_speed_train
  (u : ℝ) (a : ℝ) (t : ℕ) :
  u = 0 → a = 1 → t = 20 → u + a * t = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end final_speed_train_l526_526644


namespace coefficient_of_x5_in_expansion_l526_526190

theorem coefficient_of_x5_in_expansion :
  let a := (x : ℝ),
      b := (3 * Real.sqrt 2),
      n := 9,
      k := 4 in
  let binom := (Nat.choose n k) in
  let term := binom * a ^ (n - k) * b ^ k in
  term = 40824 * a ^ 5 := by
    sorry

end coefficient_of_x5_in_expansion_l526_526190


namespace find_ellipse_properties_l526_526890

structure Ellipse (a b : ℝ) (h : a > b ∧ b > 0) :=
  (equation : ∀ x y : ℝ, (x^2) / a^2 + (y^2) / b^2 = 1)

def line_l (x y : ℝ) := y = x + 2

structure Circle (radius : ℝ) :=
  (center : ℝ × ℝ)
  (equation : ∀ x y : ℝ, (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2)

def points_on_ellipse (a b : ℝ) (x0 y0 : ℝ) (h : x0 > 0 ∧ y0 > 0 ∧ y0 = k * x0) : Prop :=
  (x0^2) / a^2 + (y0^2) / b^2 = 1

def dot_product (AX AY BX BY : ℝ) := AX * BX + AY * BY

def max_area (a b : ℝ) (k : ℝ) (x0 y0 : ℝ) (A D : ℝ × ℝ) (h : y0 = k * x0 ∧ D.1 = A.1 ∧ D.2 = -A.2) :=
  ∃ x0 y0 : ℝ, (points_on_ellipse a b x0 y0 ∧ 
    y0 = k * x0 ∧ ∀ B : ℝ × ℝ, B.1 = sqrt(2) ∧ B.2 = 1 ∧ dot_product x0 y0 B.1 B.2 = sqrt(6) ∧
    ∀ k : ℝ, ∃ k = sqrt(2) ∧ 
    let A = (x0, y0) in ∀ D : ℝ × ℝ, D.1 = A.1 ∧ D.2 = -A.2 →
    max (1 / 2 * x0 * 2 * y0) = sqrt(6) / 2)

theorem find_ellipse_properties :
  ∃ (a b : ℝ) (k : ℝ) (x0 y0 : ℝ), (Ellipse a b) (points_on_ellipse a b x0 y0)
  ∧ (k = sqrt(2))
  ∧ max_area a b k x0 y0 (x0, y0) :=
sorry

end find_ellipse_properties_l526_526890


namespace length_of_AC_l526_526416

open Real

def euclidean_distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem length_of_AC :
  let A := (0, 0)
  let B := (5, 12)
  let Z := (5, 0)
  let C := (26, 12)
  let AB := euclidean_distance A B
  let ZC := euclidean_distance Z C
  let AZ := euclidean_distance A Z
  (AB = 13) ∧ (ZC = 25) ∧ (AZ = 5) →
  abs (euclidean_distance A C - 18.4) < 0.1 :=
begin
  sorry
end

end length_of_AC_l526_526416


namespace compare_values_l526_526808

variable (a b c : ℝ)

noncomputable def a_def : ℝ := 0.1 * Real.exp 0.1
noncomputable def b_def : ℝ := 1 / 9
noncomputable def c_def : ℝ := - Real.log 0.9

theorem compare_values : 
  c < a ∧ a < b := 
begin
  let a := 0.1 * Real.exp 0.1,
  let b := 1 / 9,
  let c := - Real.log 0.9,
  have h1 : c < a, sorry,
  have h2 : a < b, sorry,
  exact ⟨h1, h2⟩
end

end compare_values_l526_526808


namespace smallest_root_floor_l526_526996

noncomputable def g (x : ℝ) : ℝ := Real.cos x + 3 * Real.sin x + 4 * Real.cot x

theorem smallest_root_floor (s : ℝ) 
  (h1 : ∀ x > 0, g x ≠ 0 → x ≥ s) 
  (h2 : s > 0) 
  (h3 : g s = 0) : 
  ⌊s⌋ = 2 :=
sorry

end smallest_root_floor_l526_526996


namespace contractor_fired_two_people_l526_526652

theorem contractor_fired_two_people
  (total_days : ℕ) (initial_people : ℕ) (days_worked : ℕ) (fraction_completed : ℚ)
  (remaining_days : ℕ) (people_fired : ℕ)
  (h1 : total_days = 100)
  (h2 : initial_people = 10)
  (h3 : days_worked = 20)
  (h4 : fraction_completed = 1/4)
  (h5 : remaining_days = 75)
  (h6 : remaining_days + days_worked = total_days)
  (h7 : people_fired = initial_people - 8) :
  people_fired = 2 :=
  sorry

end contractor_fired_two_people_l526_526652


namespace probability_between_5_over_2_and_7_over_2_l526_526020

noncomputable def X : ℝ → ℝ := sorry  -- Define the random variable X

axiom normal_distribution (μ σ : ℝ) : Type := sorry  -- Axiom for normal distribution

-- Assume X follows a normal distribution with μ = 3, σ = 1/2
def X_distribution : normal_distribution 3 (1/2) := sorry

-- Given P(X > 7/2) = 0.1587
axiom P_X_greater_than_7_over_2 : (P (X > 7 / 2)) = 0.1587 := sorry

-- We need to prove P(5/2 ≤ X ≤ 7/2) = 0.6826
theorem probability_between_5_over_2_and_7_over_2 :
  P (5 / 2 ≤ X ∧ X ≤ 7 / 2) = 0.6826 := by
  sorry

end probability_between_5_over_2_and_7_over_2_l526_526020


namespace fraction_of_field_planted_is_correct_l526_526771

-- Definitions for given conditions

def leg1 : ℕ := 5
def leg2 : ℕ := 12
def hypotenuse : ℕ := 13 -- This is derived from the Pythagorean theorem

def side_length_of_square : ℝ := 21 / 17
def area_of_triangle : ℝ := 30 
def radius_of_circular_path : ℝ := 1
def area_of_square : ℝ := (21 / 17) ^ 2
def area_of_circular_path : ℝ := π -- πr^2 with r=1

-- Fraction of the field that is planted

def planted_fraction : ℝ := (area_of_triangle - area_of_square - area_of_circular_path) / area_of_triangle

theorem fraction_of_field_planted_is_correct :
  planted_fraction = (30 - (21 / 17)^2 - π) / 30 :=
by
  sorry

end fraction_of_field_planted_is_correct_l526_526771


namespace max_value_F_l526_526000

def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := (1/4) * (x - 1)^2 - 1
def F (x : ℝ) : ℝ := f x - g x

theorem max_value_F : ∀ (x : ℝ), (F 2 = Real.log 2 + 3 / 4) :=
by
  sorry

end max_value_F_l526_526000


namespace solve_fraction_zero_l526_526064

theorem solve_fraction_zero (x : ℝ) (h : (x + 5) / (x - 2) = 0) : x = -5 :=
by
  sorry

end solve_fraction_zero_l526_526064


namespace coordinates_of_P_l526_526133

-- Define the conditions and the question as a Lean theorem
theorem coordinates_of_P (m : ℝ) (P : ℝ × ℝ) (h1 : P = (m + 3, m + 1)) (h2 : P.2 = 0) :
  P = (2, 0) := 
sorry

end coordinates_of_P_l526_526133


namespace polynomial_zeros_condition_l526_526116

theorem polynomial_zeros_condition (k : ℕ) (P : Polynomial ℝ) (h_deg : P.degree = k) 
(h_distinct : ∀ a₁ a₂, a₁ ≠ a₂ → IsRoot P a₁ → IsRoot P a₂ → False) 
(h_condition : ∀ a, IsRoot P a → P.eval (a + 1) = 1) :
∃ c : ℝ, ∀ a, IsRoot P a → P.eval (a + 1) = 1 := 
sorry

end polynomial_zeros_condition_l526_526116


namespace find_a6_l526_526879

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {a₁ : ℝ}

/-- The sequence is a geometric sequence -/
axiom geom_seq (n : ℕ) : a n = a₁ * q ^ (n - 1)

/-- The sum of the first three terms is 168 -/
axiom sum_of_first_three_terms : a₁ + a₁ * q + a₁ * q ^ 2 = 168

/-- The difference between the 2nd and the 5th terms is 42 -/
axiom difference_a2_a5 : a₁ * q - a₁ * q ^ 4 = 42

theorem find_a6 : a 6 = 3 :=
by
  -- Proof goes here
  sorry

end find_a6_l526_526879


namespace total_area_correct_l526_526238

noncomputable def radius : ℝ := 7
noncomputable def width : ℝ := 2 * radius
noncomputable def length : ℝ := 3 * width
noncomputable def rect_area : ℝ := length * width
noncomputable def square_side : ℝ := radius * Real.sqrt 2
noncomputable def square_area : ℝ := square_side ^ 2
noncomputable def total_area : ℝ := rect_area + square_area

theorem total_area_correct : total_area = 686 := 
by
  -- Definitions provided above represent the problem's conditions
  -- The value calculated manually is 686
  -- Proof steps skipped for initial statement creation
  sorry

end total_area_correct_l526_526238


namespace average_grade_two_year_period_l526_526263

theorem average_grade_two_year_period :
  let year1_courses := 5
      year1_average := 40
      year2_courses := 6
      year2_average := 100
      total_points := year1_courses * year1_average + year2_courses * year2_average
      total_courses := year1_courses + year2_courses
      overall_average := total_points / total_courses
  in overall_average = 72.7 := by
  sorry

end average_grade_two_year_period_l526_526263


namespace projection_of_relative_displacement_l526_526589

-- Definitions of the displacements of particles A and B
def sA : ℝ × ℝ := (2, 10)
def sB : ℝ × ℝ := (4, 3)

-- Computed relative displacement
def s : ℝ × ℝ := (sB.1 - sA.1, sB.2 - sA.2)

-- Inner product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Norm of a vector
noncomputable def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Projection of vector a on vector b
noncomputable def projection (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (norm b)

-- Statement we need to prove
theorem projection_of_relative_displacement :
  projection s sB = -13 / 5 := by
  sorry

end projection_of_relative_displacement_l526_526589


namespace origin_distance_moved_is_correct_l526_526299

noncomputable def original_center : ℝ × ℝ := (3, 1)
noncomputable def original_radius : ℝ := 3

noncomputable def dilated_center : ℝ × ℝ := (8, 4)
noncomputable def dilated_radius : ℝ := 5

noncomputable def dilation_factor : ℝ :=
  dilated_radius / original_radius

noncomputable def center_of_dilation : ℝ × ℝ := (-2, -3)

noncomputable def initial_distance : ℝ :=
  real.sqrt ((-2)^2 + (-3)^2)

noncomputable def final_distance : ℝ :=
  dilation_factor * initial_distance

noncomputable def distance_moved : ℝ :=
  final_distance - initial_distance

theorem origin_distance_moved_is_correct :
  distance_moved = (2 / 3) * real.sqrt 13 :=
sorry

end origin_distance_moved_is_correct_l526_526299


namespace triangle_angle_bisectors_l526_526308

theorem triangle_angle_bisectors (α β γ : ℝ) 
  (h1 : α + β + γ = 180)
  (h2 : α = 100) 
  (h3 : β = 30) 
  (h4 : γ = 50) :
  ∃ α' β' γ', α' = 40 ∧ β' = 65 ∧ γ' = 75 :=
sorry

end triangle_angle_bisectors_l526_526308


namespace fraction_difference_is_correct_l526_526725

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l526_526725


namespace yellow_dandelions_day_before_yesterday_l526_526248

-- Define the problem in terms of conditions and conclusion
theorem yellow_dandelions_day_before_yesterday
  (yellow_yesterday : ℕ) (white_yesterday : ℕ)
  (yellow_today : ℕ) (white_today : ℕ) :
  yellow_yesterday = 20 → white_yesterday = 14 →
  yellow_today = 15 → white_today = 11 →
  (let yellow_day_before_yesterday := white_yesterday + white_today
  in yellow_day_before_yesterday = 25) :=
by
  intros h1 h2 h3 h4
  let yellow_day_before_yesterday := white_yesterday + white_today
  show yellow_day_before_yesterday = 25
  exact (by sorry)

end yellow_dandelions_day_before_yesterday_l526_526248


namespace mean_equivalence_l526_526564

theorem mean_equivalence :
  (20 + 30 + 40) / 3 = (23 + 30 + 37) / 3 :=
by sorry

end mean_equivalence_l526_526564


namespace cannot_be_value_of_A_plus_P_l526_526162

theorem cannot_be_value_of_A_plus_P (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (a_neq_b: a ≠ b) :
  let A : ℕ := a * b
  let P : ℕ := 2 * a + 2 * b
  A + P ≠ 102 :=
by
  sorry

end cannot_be_value_of_A_plus_P_l526_526162


namespace compare_a_b_c_l526_526850

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l526_526850


namespace sum_possible_k_l526_526456

theorem sum_possible_k (j k : ℕ) (hcond : 1 / j + 1 / k = 1 / 4) (hj : 0 < j) (hk : 0 < k) :
  j ≠ k → ∑ k in {k | ∃ j, 1 / j + 1 / k = 1 / 4}, k = 51 :=
by
  sorry

end sum_possible_k_l526_526456


namespace distinct_integers_sum_l526_526050

theorem distinct_integers_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_product : a * b * c * d = 357) : a + b + c + d = 28 :=
by
  sorry

end distinct_integers_sum_l526_526050


namespace age_of_oldest_child_l526_526553

theorem age_of_oldest_child (a1 a2 a3 x : ℕ) (h1 : a1 = 5) (h2 : a2 = 7) (h3 : a3 = 10) (h_avg : (a1 + a2 + a3 + x) / 4 = 8) : x = 10 :=
by
  sorry

end age_of_oldest_child_l526_526553


namespace b_2016_eq_neg_4_l526_526970

def b : ℕ → ℤ
| 0     => 1
| 1     => 5
| (n+2) => b (n+1) - b n

theorem b_2016_eq_neg_4 : b 2015 = -4 :=
sorry

end b_2016_eq_neg_4_l526_526970


namespace similar_triangles_l526_526173

theorem similar_triangles 
  (O A B C A1 B1 C1 A2 B2 C2 : Type)
  (homothety : ∀ X, X ∈ {A, B, C} → X → X → Prop)
  (A_eqn : homothety A A1)
  (B_eqn : homothety B B1)
  (C_eqn : homothety C C1)
  (parallelogram_OAA1A2 : O + A1 - O = A2)
  (parallelogram_OBB1B2 : O + B1 - O = B2)
  (parallelogram_OCC1C2 : O + C1 - O = C2)
  : similar_triangle ABC A2B2C2 := 
sorry

end similar_triangles_l526_526173


namespace sum_of_xs_eq_459_by_10_l526_526331

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x
noncomputable def frac (x : ℝ) : ℝ := x - floor x

theorem sum_of_xs_eq_459_by_10 :
  let X := {x : ℝ | 0 ≤ x ∧ x ≤ 10 ∧ floor x * frac x * ceil x = 1} in
  (∑ x in X, x) = 459 / 10 :=
by
  sorry

end sum_of_xs_eq_459_by_10_l526_526331


namespace number_of_students_playing_both_l526_526949

theorem number_of_students_playing_both (
  (N : ℕ) (H : ℕ) (B : ℕ) (neither : ℕ) 
  (hN : N = 50) 
  (hH : H = 30) 
  (hB : B = 35) 
  (hneither : neither = 10)
  ) : 
  (H + B - (N - neither)) = 25 :=
by {
  rw [hN, hH, hB, hneither],
  norm_num,
  sorry
}

end number_of_students_playing_both_l526_526949


namespace sum_of_possible_ks_l526_526430

theorem sum_of_possible_ks : 
  (∃ (j k : ℕ), (1 < j) ∧ (1 < k) ∧ j ≠ k ∧ ((1/j : ℝ) + (1/k : ℝ) = (1/4))) → 
  (∑ k in {20, 12, 8, 6, 5}, k) = 51 :=
begin
  sorry
end

end sum_of_possible_ks_l526_526430


namespace elective_combination_or_permutation_l526_526236

/--
 In a school that offers 3 elective courses of type A and 4 elective courses of type B,
 a student needs to choose 3 courses in total, with at least one course from each type.
 Prove that this is a combination problem and not a permutation problem.
-/
theorem elective_combination_or_permutation
  (courses_A courses_B : Type)
  [Finite courses_A] [Finite courses_B]
  (num_A : Nat) (num_B : Nat)
  (hA : Finite.card courses_A = 3) (hB : Finite.card courses_B = 4)
  (h_combination_not_permutation : ∀ (selections : Finset (courses_A ⊕ courses_B)),
    selections.card = 3 → 
    (∃a, ∃b, a ∈ selections ∧ b ∈ selections ∧ a ≠ b) → 
    (selection_method = "Combination")) :
  selection_method = "Combination" :=
sorry

end elective_combination_or_permutation_l526_526236


namespace excess_common_fraction_l526_526713

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l526_526713


namespace probability_male_is_2_5_l526_526234

variable (num_male_students num_female_students : ℕ)

def total_students (num_male_students num_female_students : ℕ) : ℕ :=
  num_male_students + num_female_students

def probability_of_male (num_male_students num_female_students : ℕ) : ℚ :=
  num_male_students / (total_students num_male_students num_female_students : ℚ)

theorem probability_male_is_2_5 :
  probability_of_male 2 3 = 2 / 5 := by
    sorry

end probability_male_is_2_5_l526_526234


namespace quadrilateral_is_rhombus_l526_526106

/-- Given a convex quadrilateral ABCD with intersection of diagonals at O, 
    if the perimeters of triangles ABO, BCO, CDO, and DAO are equal, 
    then ABCD is a rhombus. -/
theorem quadrilateral_is_rhombus (A B C D O : Point ) 
  (h_O : O = intersection (diagonal A C) (diagonal B D))
  (h_perimeters : perimeter (triangle A B O) = perimeter (triangle B C O) ∧
                  perimeter (triangle B C O) = perimeter (triangle C D O) ∧
                  perimeter (triangle C D O) = perimeter (triangle D A O)) 
  : is_rhombus A B C D :=
sorry

end quadrilateral_is_rhombus_l526_526106


namespace mrs_sheridan_total_cats_l526_526124

-- Definitions from the conditions
def original_cats : Nat := 17
def additional_cats : Nat := 14

-- The total number of cats is the sum of the original and additional cats
def total_cats : Nat := original_cats + additional_cats

-- Statement to prove
theorem mrs_sheridan_total_cats : total_cats = 31 := by
  sorry

end mrs_sheridan_total_cats_l526_526124


namespace sum_possible_k_l526_526455

theorem sum_possible_k (j k : ℕ) (hcond : 1 / j + 1 / k = 1 / 4) (hj : 0 < j) (hk : 0 < k) :
  j ≠ k → ∑ k in {k | ∃ j, 1 / j + 1 / k = 1 / 4}, k = 51 :=
by
  sorry

end sum_possible_k_l526_526455


namespace rhombus_area_l526_526155

-- Define the diagonals of the rhombus
def d1 : ℝ := 22
def d2 : ℝ := 30

-- Define the formula for the area of a rhombus
def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- State the theorem
theorem rhombus_area :
  area_of_rhombus d1 d2 = 330 := by
  -- Proof is omitted as required
  sorry

end rhombus_area_l526_526155


namespace rectangle_ratio_l526_526790

noncomputable def side_length (s : ℝ) : Prop :=
  ∃ large_square rectangle : ℝ,
  large_square = 3 * s ∧
  (rectangle.side_length.width = s ∧ rectangle.side_length.length = 3 * s)

theorem rectangle_ratio (s : ℝ) (rectangle : { width : ℝ × length : ℝ }) 
  (h : rectangle.width = s ∧ rectangle.length = 3 * s) :
  (rectangle.length / rectangle.width) = 3 := by
  sorry

end rectangle_ratio_l526_526790


namespace symmetric_point_l526_526324

/-- Find the point M' that is symmetric to the point M with respect to the plane. -/
theorem symmetric_point (M : ℝ × ℝ × ℝ) (plane_coeffs : ℝ × ℝ × ℝ) (d : ℝ) :
  let M' := (-1, 0, 1) in
  M = (0, 2, 1) → plane_coeffs = (2, 4, 0) → d = -3 →
  M' = (-2 * plane_coeffs.1 * (M.1 - plane_coeffs.1 * d) / (plane_coeffs.1 ^ 2 + plane_coeffs.2 ^ 2) ,
        -2 * plane_coeffs.2 * (M.2 - plane_coeffs.2 * d) / (plane_coeffs.1 ^ 2 + plane_coeffs.2 ^ 2) ,
        M.3) :=
begin
  intros,
  sorry,
end

end symmetric_point_l526_526324


namespace non_congruent_squares_in_6_by_6_grid_l526_526388

theorem non_congruent_squares_in_6_by_6_grid : 
  let n := 6 in
  (∑ i in Range (n-1), (n-i)*(n-i) + 
   ∑ i in Range (n-1), (n-i)*(n-i) + 
   ∑ i in Range (5), 5 * 6 - i)%nat = 155 :=
by 
  let n := 6
  have h1 : ∑ i in Range (n-1), (n - i)^2 = 
    (n-1)^2 + (n-2)^2 + ... + 1 = 25 + 16 + 9 + 4 + 1 := sorry
  have h2 : ∑ i in Range (n-1), (n - i)^2 = 
    (5)^2 + (4)^2 + ... + 1^2 = 25 + 16 + 9 + 4 + 1 := sorry
  have h3 : ∑ i in Range (5), 5 * (6 - i) = 
    (5*5) + (5*4) := sorry
  show (∑ i in Range (n-1), (n-i)*(n-i) + 
       ∑ i in Range (n-1), (n-i)*(n-i) + 
       ∑ i in Range (5), 5 * (6 - i)) = 155 := by sorry

end non_congruent_squares_in_6_by_6_grid_l526_526388


namespace compare_a_b_c_l526_526846

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l526_526846


namespace concyclic_points_l526_526990

open Set Finset

def midpoint_of_arc (A B C : Point) (Gamma : Circle) : Prop :=
  is_midpoint A (arc B C Gamma) ∧ 
  ¬ (B ∈ arc B C Gamma ∧ C ∈ arc B C Gamma)

def are_chords (A D E : Point) (Gamma : Circle) : Prop :=
  is_chord A D Gamma ∧ is_chord A E Gamma

theorem concyclic_points (Γ : Circle) (B C A D E F G : Point) 
  (h1 : is_chord B C Γ)
  (h2 : midpoint_of_arc A B C Γ)
  (h3 : are_chords A D E Γ)
  (h4 : intersection_chord A D B C = F)
  (h5 : intersection_chord A E B C = G) :
  cyclic [D, E, F, G] :=
sorry

end concyclic_points_l526_526990


namespace spherical_distance_between_A_and_B_l526_526509

-- Declare the necessary variables
variables (R : ℝ) 

-- Define the conditions
def latitude_circle_radius := R / 2
def arc_length_AB := π * R / 2
def spherical_central_angle := π / 3

-- State the proposition to prove
theorem spherical_distance_between_A_and_B (R : ℝ) :
  (spherical_distance : ℝ) := 
  spherical_central_angle * R = π * R / 3 := 
  sorry

end spherical_distance_between_A_and_B_l526_526509


namespace gcd_factorials_l526_526608

def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

noncomputable def gcd (a b : ℕ) := Nat.gcd a b

theorem gcd_factorials :
  gcd (fact 6) ((fact 9) / (fact 4)) = 480 := by
  sorry

end gcd_factorials_l526_526608


namespace units_digit_42_3_plus_27_2_l526_526200

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_42_3_plus_27_2 : units_digit (42^3 + 27^2) = 7 :=
by
  sorry

end units_digit_42_3_plus_27_2_l526_526200


namespace max_value_of_y_over_x_l526_526048

theorem max_value_of_y_over_x (x y : ℝ) (h1 : x^2 + y^2 - 4 * x + 1 = 0) 
  (h2 : ∃ z : ℝ, y = z^2) : ∃ k : ℝ, y = k * x ∧ -real.sqrt 3 ≤ k ∧ k ≤ real.sqrt 3 ∧ (∀ m : ℝ, y = m * x → m ≤ real.sqrt 3) :=
by
  sorry

end max_value_of_y_over_x_l526_526048


namespace part1_part2_l526_526221

variable {f : ℝ → ℝ}

-- Assume the conditions
axiom f_domain : ∀ x > 0, f x
axiom f_value_at_2 : f 2 = 1
axiom f_multiplication : ∀ x y > 0, f (x * y) = f x + f y
axiom f_increasing : ∀ x1 x2 > 0, x2 > x1 → f x2 > f x1

-- Define the proof problem
theorem part1 : f 1 = 0 ∧ f 4 = 2 ∧ f 8 = 3 :=
  sorry

theorem part2 : ∀ x > 2, x ≤ 4 → f x + f (x - 2) ≤ 3 :=
  sorry

end part1_part2_l526_526221


namespace asha_borrow_father_amount_l526_526682

def asha_borrow_conditions (B M G S F Total Remaining: ℤ) : Prop :=
  Total = B + F + M + G + S ∧ Remaining = Total - (3 * Total / 4)

theorem asha_borrow_father_amount :
  ∃ F : ℤ, asha_borrow_conditions 20 30 70 100 F 260 65 ∧ F = 40 :=
by {
  use 40,
  unfold asha_borrow_conditions,
  split,
  {
    refl,
  },
  {
    norm_num,
  }
}

end asha_borrow_father_amount_l526_526682


namespace eccentricity_of_ellipse_intersection_point_of_lines_l526_526352

def ellipse (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def distance_from_origin_to_line (a b : ℝ) (c : ℝ) (h : c = sqrt (a^2 - b^2)) : ℝ :=
  b * c / a

theorem eccentricity_of_ellipse (a b : ℝ) (h₁ : a > b) (h₂ : b > 0)
  (h₃ : distance_from_origin_to_line a b (sqrt (a^2 - b^2)) = sqrt 2 / 2 * b) :
  sqrt (a^2 - b^2) / a = sqrt 2 / 2 := 
sorry

theorem intersection_point_of_lines (k : ℝ) (a : ℝ) (h₁ : a > 2) (h₂ : a = 2 * sqrt 2)
  (l : a = 2) (line : ∀ x y : ℝ, y = k * x + 4 → Prop) 
  (C : ∀ x y : ℝ, ellipse a 2 h₁ h₂ x y → ellipse a 2 h₁ l x y)
  (M N : ∀ x y : ℝ, line x y → Prop) :
  ∀ G : ℝ × ℝ, G = (0, 1) := 
sorry

end eccentricity_of_ellipse_intersection_point_of_lines_l526_526352


namespace bucket_B_more_than_C_l526_526580

-- Define the number of pieces of fruit in bucket B as a constant
def B := 12

-- Define the number of pieces of fruit in bucket C as a constant
def C := 9

-- Define the number of pieces of fruit in bucket A based on B
def A := B + 4

-- Define the total number of pieces of fruit in all three buckets
def total_fruit := A + B + C

-- Prove that bucket B has 3 more pieces of fruit than bucket C
theorem bucket_B_more_than_C : B - C = 3 := by
  -- sorry is used to skip the proof
  sorry

end bucket_B_more_than_C_l526_526580


namespace total_revenue_correct_l526_526511

noncomputable def total_revenue_with_discounts (kittens_sold puppies_sold rabbits_sold guinea_pigs_sold : ℕ)
  (kitten_price puppy_price rabbit_price guinea_pig_price : ℕ) (discount_rate : ℚ) : ℤ :=
let revenue_without_discounts := kittens_sold * kitten_price + puppies_sold * puppy_price + rabbits_sold * rabbit_price + guinea_pigs_sold * guinea_pig_price in
let number_of_discounts := puppies_sold in
let price_combination := kitten_price + puppy_price in
let discount_per_combination := price_combination * discount_rate in
let total_discount := number_of_discounts * discount_per_combination in
revenue_without_discounts - total_discount

theorem total_revenue_correct :
  total_revenue_with_discounts 10 8 4 6 80 150 45 30 (10/100 : ℚ) = 2176 :=
by
  sorry

end total_revenue_correct_l526_526511


namespace compare_abc_l526_526823
open Real

noncomputable def a : ℝ := 0.1 * exp(0.1)
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - log 0.9

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l526_526823


namespace percent_gain_on_sheep_transaction_l526_526250

theorem percent_gain_on_sheep_transaction (c : ℝ) :
  let total_cost := 900 * c,
      revenue_850 := 900 * c,
      price_per_sheep := revenue_850 / 850,
      revenue_50 := 50 * price_per_sheep,
      total_revenue := revenue_850 + revenue_50,
      profit := total_revenue - total_cost,
      percentage_gain := (profit / total_cost) * 100
  in percentage_gain = 5.88 := 
by {
  sorry
}

end percent_gain_on_sheep_transaction_l526_526250


namespace compare_values_l526_526810

variable (a b c : ℝ)

noncomputable def a_def : ℝ := 0.1 * Real.exp 0.1
noncomputable def b_def : ℝ := 1 / 9
noncomputable def c_def : ℝ := - Real.log 0.9

theorem compare_values : 
  c < a ∧ a < b := 
begin
  let a := 0.1 * Real.exp 0.1,
  let b := 1 / 9,
  let c := - Real.log 0.9,
  have h1 : c < a, sorry,
  have h2 : a < b, sorry,
  exact ⟨h1, h2⟩
end

end compare_values_l526_526810


namespace line_segment_parameters_l526_526788

theorem line_segment_parameters :
  ∃ (a b c d : ℝ), 
    (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (t = 0 → (at + b, ct + d) = (1, -3)) ∧ (t = 1 → (at + b, ct + d) = (-4, 9))) ∧
    (a^2 + b^2 + c^2 + d^2 = 179) :=
by
  use -5, 1, 12, -3
  split
  · intro t
    intro ht
    split
    · intro ht_zero
      rw [ht_zero]
      simp
    · intro ht_one
      rw [ht_one]
      simp
  · sorry

end line_segment_parameters_l526_526788


namespace sum_of_all_possible_k_values_l526_526441

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l526_526441


namespace triangle_equilateral_of_medians_and_angles_l526_526093

theorem triangle_equilateral_of_medians_and_angles (ABC : Triangle) (A B C F E : Point)
  (h_median_AF : is_median A F C B) (h_median_CE : is_median C E A B)
  (h_angle_BAF : ∠ BAF = 30 °) (h_angle_BCE : ∠ BCE = 30 °) : 
  is_equilateral ABC :=
sorry

end triangle_equilateral_of_medians_and_angles_l526_526093


namespace find_a_plus_c_l526_526067

noncomputable def a b c : ℝ

axiom geometric_sequence (a b c : ℝ) : b^2 = a * c
axiom sin_B (B : ℝ) : sin B = 5 / 13
axiom cos_B (a b c : ℝ) : cos B = 12 / (a * c)
axiom ac_value (a c : ℝ) : a * c = 13
axiom triangle_abc (a b c B : ℝ) : b^2 = a^2 + c^2 - 2 * a * c * cos B

theorem find_a_plus_c (a c : ℝ) : a + c = 3 * real.sqrt 7 :=
by
  sorry

end find_a_plus_c_l526_526067


namespace rabbit_distribution_problem_l526_526137

-- Define the rabbits and stores
constant rabbits : Fin 6
constant stores : Fin 5

-- Define the parent and child relationships
constant is_parent : Fin 6 → Bool

-- Define the distribution function
noncomputable def distribute_rabbits (distribution : Fin 6 → Fin 5) : Prop :=
  ∀ i j : Fin 6, (is_parent i = is_parent j → distribution i ≠ distribution j) ∧ (∀ s : Fin 5, ∃! r : Fin 6, distribution r = s → r < 3)

-- Statement of the problem
theorem rabbit_distribution_problem :
  (∃ distribution : Fin 6 → Fin 5, distribute_rabbits distribution) → ∃! (ways : ℕ), ways = 446 :=
sorry

end rabbit_distribution_problem_l526_526137


namespace number_solution_l526_526210

theorem number_solution (x : ℝ) : (x / 5 + 4 = x / 4 - 4) → x = 160 := by
  intros h
  sorry

end number_solution_l526_526210


namespace sum_possible_values_k_l526_526463

open Nat

theorem sum_possible_values_k (j k : ℕ) (h : (1 / j : ℚ) + 1 / k = 1 / 4) : 
  ∃ ks : List ℕ, (∀ k' ∈ ks, ∃ j', (1 / j' : ℚ) + 1 / k' = 1 / 4) ∧ ks.sum = 51 :=
by
  sorry

end sum_possible_values_k_l526_526463


namespace piravena_trip_distance_l526_526130

theorem piravena_trip_distance : 
  ∀ (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z],
  dist X Z = 4000 ∧ dist X Y = 5000 ∧ (∀ A B C : Type, 
  right_triangle A B C → (dist B C) ^ 2 = (dist A B) ^ 2 - (dist A C) ^ 2) →
  ∃ (XZ YZ XY : ℝ), XZ = dist X Z ∧ YZ = dist Y Z ∧ XY = dist X Y ∧ 
  right_triangle X Y Z ∧ XY = 5000 ∧ XZ = 4000 ∧ YZ = 3000 ∧ 
  XY + XZ + YZ = 12000 := 
by 
  sorry


end piravena_trip_distance_l526_526130


namespace coefficient_of_x5_in_expansion_l526_526189

theorem coefficient_of_x5_in_expansion :
  let a := (x : ℝ),
      b := (3 * Real.sqrt 2),
      n := 9,
      k := 4 in
  let binom := (Nat.choose n k) in
  let term := binom * a ^ (n - k) * b ^ k in
  term = 40824 * a ^ 5 := by
    sorry

end coefficient_of_x5_in_expansion_l526_526189


namespace zero_possible_values_of_k_l526_526685

theorem zero_possible_values_of_k :
  ∃ k_value_count : ℕ, (∀ p q : ℕ, Prime p → Prime q → p + q = 57 → k_value_count = 0) ∧ k_value_count = 0 :=
by
  use 0
  intro p q hp hq hsum
  sorry

end zero_possible_values_of_k_l526_526685


namespace bart_total_earnings_l526_526280

def earnings_per_survey_per_question (rate_per_question : ℝ) (questions_per_survey : ℕ) : ℝ :=
  rate_per_question * (questions_per_survey : ℝ)

def total_earnings_per_day (rate_per_question : ℝ) (questions_per_survey : ℕ) (surveys_completed : ℕ) : ℝ :=
  (earnings_per_survey_per_question rate_per_question questions_per_survey) * (surveys_completed : ℝ)

theorem bart_total_earnings :
  let monday_earnings := total_earnings_per_day 0.20 10 3 in
  let tuesday_rate := 0.20 + 0.05 in
  let tuesday_earnings := total_earnings_per_day tuesday_rate 12 4 in
  let wednesday_earnings := total_earnings_per_day 0.10 15 5 in
  monday_earnings + tuesday_earnings + wednesday_earnings = 25.50 :=
by
  let monday_earnings := total_earnings_per_day 0.20 10 3
  let tuesday_rate := 0.20 + 0.05
  let tuesday_earnings := total_earnings_per_day tuesday_rate 12 4
  let wednesday_earnings := total_earnings_per_day 0.10 15 5
  have h : monday_earnings + tuesday_earnings + wednesday_earnings = 25.50 := sorry
  exact h

end bart_total_earnings_l526_526280


namespace repeating_decimal_difference_l526_526706

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l526_526706


namespace part_a_part_b_l526_526629

-- Placeholder definition for the number of angles function.
-- Replace with actual implementation
def angles (n : Nat) : Nat := sorry

-- Placeholder definition for checking the number of angles in a specific figure.
-- Replace with actual implementation
def angles_figure (f : Nat → Type) (n : Nat) : Nat := sorry

-- Part (a)
theorem part_a (k : Nat) (n : Nat) (h : n = 2 * k) : angles(n) ≤ 2 * k - 1 :=
sorry

-- Part (b)
theorem part_b : ∃ f, angles_figure f 100 = 3 :=
sorry

end part_a_part_b_l526_526629


namespace initial_percentage_increase_l526_526166

variable (P : ℝ) (x : ℝ)

theorem initial_percentage_increase :
  (P * (1 + x / 100) * 1.3 = P * 1.625) → (x = 25) := by
  sorry

end initial_percentage_increase_l526_526166


namespace find_smallest_not_good_l526_526290

def is_good (S : set ℕ) : Prop :=
  ∃ (A B : set ℕ), (A ∪ B = S) ∧ (∀ a b c ∈ A, ¬(a^b = c)) ∧ (∀ a b c ∈ B, ¬(a^b = c))

noncomputable def smallest_not_good : ℕ :=
  Inf {n : ℕ | ¬ is_good ({x : ℕ | 2 ≤ x ∧ x ≤ n})}

theorem find_smallest_not_good : smallest_not_good = 65536 :=
sorry

end find_smallest_not_good_l526_526290


namespace no_prime_ratio_circle_l526_526605

theorem no_prime_ratio_circle (A : Fin 2007 → ℕ) :
  ¬ (∀ i : Fin 2007, (∃ p : ℕ, Nat.Prime p ∧ (p = A i / A ((i + 1) % 2007) ∨ p = A ((i + 1) % 2007) / A i))) := by
  sorry

end no_prime_ratio_circle_l526_526605


namespace measure_angle_BCA_l526_526071

-- Definitions
variables {A B C D M O : Type} -- Points in the plane
variables (angle : A → A → A → Real) (midpoint : A → A → A → Prop)
variables (intersect : A → A → A → A → A → Prop)

-- Given conditions
axiom convex_quadrilateral (A B C D : Type) : Prop
axiom is_midpoint {A D M : Type} : midpoint A D M
axiom intersection_point {A C B M O : Type} : intersect A C B M O
axiom angle_ABM : angle A B M = 55
axiom angle_AMB : angle A M B = 70
axiom angle_BOC : angle B O C = 80
axiom angle_ADC : angle A D C = 60

-- Question
theorem measure_angle_BCA (A B C D M O : Type)
  [convex_quadrilateral A B C D] 
  [midpoint A D M] 
  [intersect A C B M O]
  (angle_ABM : angle A B M = 55)
  (angle_AMB : angle A M B = 70)
  (angle_BOC : angle B O C = 80)
  (angle_ADC : angle A D C = 60) :
  angle B C A = 35 := 
sorry

end measure_angle_BCA_l526_526071


namespace compare_a_b_c_l526_526817

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b :=
by
  have : c < b := sorry,
  have : a < b := sorry,
  have : a > c := sorry,
  exact ⟨this.2, this.1⟩

end compare_a_b_c_l526_526817


namespace exists_even_p_and_odd_pq_l526_526679

noncomputable def even_function (p : ℝ → ℝ) : Prop :=
∀ x, p x = p (-x)

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = -f (-x)

theorem exists_even_p_and_odd_pq : 
  ∃ (p q : ℝ → ℝ), even_function p ∧ odd_function (λ x, p (q x)) :=
by
  let p : ℝ → ℝ := λ x, Real.cos x
  let q : ℝ → ℝ := λ x, (Real.pi.div 2) - x
  use [p, q]
  sorry

end exists_even_p_and_odd_pq_l526_526679


namespace trapezium_area_l526_526780

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem trapezium_area :
  (1/2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
  sorry

end trapezium_area_l526_526780


namespace find_f_23pi_over_6_l526_526017

def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < π then 0 
  else sorry

lemma f_property (x : ℝ) : f (x + π) = f x + Real.sin x :=
  sorry

theorem find_f_23pi_over_6 :
  f (23 * π / 6) = 1 / 2 :=
  by
    sorry

end find_f_23pi_over_6_l526_526017


namespace probability_only_one_l526_526070

-- Define the probabilities
def P_A : ℚ := 1 / 2
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Define the complement probabilities
def not_P (P : ℚ) : ℚ := 1 - P
def P_not_A := not_P P_A
def P_not_B := not_P P_B
def P_not_C := not_P P_C

-- Expressions for probabilities where only one student solves the problem
def only_A_solves : ℚ := P_A * P_not_B * P_not_C
def only_B_solves : ℚ := P_B * P_not_A * P_not_C
def only_C_solves : ℚ := P_C * P_not_A * P_not_B

-- Total probability that only one student solves the problem
def P_only_one : ℚ := only_A_solves + only_B_solves + only_C_solves

-- The theorem to prove that the total probability matches
theorem probability_only_one : P_only_one = 11 / 24 := by
  sorry

end probability_only_one_l526_526070


namespace sin_theta_eq_2_sqrt_5_div_5_find_m_l526_526919

-- Part 1: Prove sin θ = 2√5 / 5
def vector_a : ℝ × ℝ := (3, -4)
def vector_b : ℝ × ℝ := (1, 2)

theorem sin_theta_eq_2_sqrt_5_div_5 (θ : ℝ) :
  sin θ = (2 * Real.sqrt 5) / 5 ↔
  let a := vector_a
  let b := vector_b
  θ = Real.arcsin ((-b.1 * a.2 + a.1 * b.2) / (Real.sqrt ((a.1 ^ 2 + a.2 ^ 2) * (b.1 ^ 2 + b.2 ^ 2))))
:= sorry

-- Part 2: Prove m = 0
theorem find_m (m : ℝ) :
  (let a := vector_a in
   let b := vector_b in
   (fun m : ℝ => (m * a.1 - b.1) * (a.1 + b.1) + (m * a.2 - b.2) * (a.2 + b.2)) m = 0) ↔ m = 0
:= sorry

end sin_theta_eq_2_sqrt_5_div_5_find_m_l526_526919


namespace find_m_and_n_l526_526563

theorem find_m_and_n :
  ∃ m n : ℚ, 
    (∀ x : ℚ, ⋍ (y : ℚ, y = (3/4 : ℚ) * x - (5 : ℚ)) 
      ∧ x = -3 + t * n
      ∧ y = m + t * 6 
      → x = -3 ∧ y = m) -> 
      (m = -29/4 ∧ n = -11/3) :=
by sorry

end find_m_and_n_l526_526563


namespace minimum_product_OP_OQ_l526_526372

theorem minimum_product_OP_OQ (a b : ℝ) (P Q : ℝ × ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : P ≠ Q) (h4 : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1) (h5 : Q.1 ^ 2 / a ^ 2 + Q.2 ^ 2 / b ^ 2 = 1)
  (h6 : P.1 * Q.1 + P.2 * Q.2 = 0) :
  (P.1 ^ 2 + P.2 ^ 2) * (Q.1 ^ 2 + Q.2 ^ 2) ≥ (2 * a ^ 2 * b ^ 2 / (a ^ 2 + b ^ 2)) :=
by sorry

end minimum_product_OP_OQ_l526_526372


namespace non_congruent_squares_6x6_grid_l526_526394

def is_square {α : Type} [linear_ordered_field α] (a b c d : (α × α)) : Prop :=
-- A function to check if four points form a square (not implemented here)
sorry

def count_non_congruent_squares (n : ℕ) : ℕ :=
-- Place calculations for counting non-congruent squares on an n x n grid (not implemented here)
sorry

theorem non_congruent_squares_6x6_grid :
  count_non_congruent_squares 6 = 128 :=
sorry

end non_congruent_squares_6x6_grid_l526_526394


namespace repeating_decimal_difference_l526_526708

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l526_526708


namespace initial_investment_B_l526_526645

theorem initial_investment_B (A_initial : ℝ) (B : ℝ) (total_profit : ℝ) (A_profit : ℝ) 
(A_withdraw : ℝ) (B_advance : ℝ) : 
  A_initial = 3000 → B_advance = 1000 → A_withdraw = 1000 → total_profit = 756 → A_profit = 288 → 
  (8 * A_initial + 4 * (A_initial - A_withdraw)) / (8 * B + 4 * (B + B_advance)) = A_profit / (total_profit - A_profit) → 
  B = 4000 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end initial_investment_B_l526_526645


namespace option_B_valid_option_D_valid_l526_526025

theorem option_B_valid (a b x₀ y₀ : ℝ) (h₀ : a > b) (h₁ : b > 0) 
  (h_cond : x₀^2 / a^2 + y₀^2 / b^2 > 1) : 
  let area := (3 * Real.sqrt 3 / 4) * a * b in
  ∃ (L : ℝ → ℝ → Prop), ∀ x y, L x y → area = (3 * Real.sqrt 3 / 4) * a * b :=
by sorry

theorem option_D_valid (a b x₀ y₀ : ℝ) (h₀ : a > b) (h₁ : b > 0) 
  (h_cond : x₀^2 / a^2 + y₀^2 / b^2 = 1) : 
  ∃ (L : ℝ → ℝ → Prop), ∃ (square : ℝ × ℝ → Prop), 
    ∀ x y, L x y → square (x, y) :=
by sorry

end option_B_valid_option_D_valid_l526_526025


namespace trapezium_area_proof_l526_526782

-- Define the lengths of the parallel sides and the distance between them
def a : ℝ := 20
def b : ℝ := 18
def h : ℝ := 15

-- Define the area of the trapezium
def area_of_trapezium (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

-- State the theorem to be proved
theorem trapezium_area_proof : area_of_trapezium a b h = 285 := by
  sorry

end trapezium_area_proof_l526_526782


namespace find_a_l526_526871

-- Define the variables
variables (m d a b : ℝ)

-- State the main theorem with conditions
theorem find_a (h : m = d * a * b / (a - b)) (h_ne : m ≠ d * b) : a = m * b / (m - d * b) :=
sorry

end find_a_l526_526871


namespace hyperbola_center_midpoint_l526_526659

/-- The center of a hyperbola given its foci -/
theorem hyperbola_center_midpoint (p₁ p₂ : (ℝ × ℝ)) : (p₁ = (3, 6)) → (p₂ = (11, 10)) → 
  ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2) = (7, 8) :=
begin
  intros h₁ h₂,
  rw [h₁, h₂],
  simp,
end

end hyperbola_center_midpoint_l526_526659


namespace smallest_r_is_fraction_inverse_l526_526100

theorem smallest_r_is_fraction_inverse (a b c d : ℕ) (r : ℚ) (ha : 2 ≤ a) (hb : 2 ≤ b)
  (h_gcd : Nat.gcd a b = 1) (hc : c ≤ a) (hd : d ≤ b) (hr : r = (a * d - b * c) / b / d) :
  (∃ k : ℕ, r = 1 / k) ∧ (1 / r).denominator = 1 := 
sorry

end smallest_r_is_fraction_inverse_l526_526100


namespace domain_of_f_2x_minus_1_l526_526016

def domain_of_f_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, -1 < x ∧ x < 1 → ∃ y, f x = y

def domain_of_f_composition (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x ∧ x < 1 → ∃ y, f (2 * x - 1) = y

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  domain_of_f_in_interval f → domain_of_f_composition f :=
begin
  sorry
end

end domain_of_f_2x_minus_1_l526_526016


namespace odd_divisors_iff_perfect_square_l526_526635

theorem odd_divisors_iff_perfect_square (n : ℕ) (h : 0 < n) : (∃ k : ℕ, k * k = n) ↔ (∃ d : ℕ, ∃ l : ℕ, l < d ∧ d * (d + 1) = DivisorCount n) := 
sorry

end odd_divisors_iff_perfect_square_l526_526635


namespace right_triangle_leg_length_l526_526061

theorem right_triangle_leg_length
  (a : ℕ) (c : ℕ) (h₁ : a = 8) (h₂ : c = 17) :
  ∃ b : ℕ, a^2 + b^2 = c^2 ∧ b = 15 :=
by
  sorry

end right_triangle_leg_length_l526_526061


namespace fraction_zero_implies_x_is_minus_5_l526_526066

theorem fraction_zero_implies_x_is_minus_5 (x : ℝ) (h1 : (x + 5) / (x - 2) = 0) (h2 : x ≠ 2) : x = -5 := 
by
  sorry

end fraction_zero_implies_x_is_minus_5_l526_526066


namespace framed_painting_ratio_l526_526661

theorem framed_painting_ratio
    (w h : ℕ) 
    (w = 20) 
    (h = 30)
    (x : ℕ) 
    (frame_condition : ∀ x, 3 * x = 3 * x)
    (frame_area_condition : (20 + 2 * x) * (30 + 6 * x) = 3 * (20 * 30)) :
  (20 + 2 * x) / (30 + 6 * x) = 1 / 2 :=
by 
  sorry

end framed_painting_ratio_l526_526661


namespace sum_is_correct_l526_526898

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the second derivative of f
def f'' (x : ℝ) : ℝ := 6*x - 6

-- Given symmetry point
def symmetry_center (x0 : ℝ) := f'' x0 = 0

-- Define the sum to be proven
def sum_f : ℝ :=
  ∑ i in Finset.range 4030, f (i.succ / 2015)

-- The main goal
theorem sum_is_correct :
  sum_f = -8058 :=
sorry

end sum_is_correct_l526_526898


namespace sixthRootsSumOfAngles_l526_526756

noncomputable def sixthRootAnglesSum (z : ℂ) (θ : ℝ) (n : ℕ) : ℝ :=
  (List.sum (List.map (λ k, (θ + 360 * k) / n) [0, 1, 2, 3, 4, 5]))

theorem sixthRootsSumOfAngles :
  ∑ k in [0, 1, 2, 3, 4, 5], (90 + 360 * k) / 6 = 990 :=
by
  sorry

end sixthRootsSumOfAngles_l526_526756


namespace complex_point_location_l526_526011

def i : ℂ := complex.I

lemma power_of_i_2017_2018 :
  1 + i^2017 - i^2018 = 2 + i :=
by {
  have h1 : i^2017 = i,
  { 
    -- Use the periodicity: i^2017 = i^(4*504 + 1) = i
    sorry
  },
  have h2 : i^2018 = -1,
  {
    -- Use the periodicity: i^2018 = i^(4*504 + 2) = -1
    sorry
  },
  -- Combine the results
  calc
    1 + i^2017 - i^2018 = 1 + i - (-1) : by rw [h1, h2]
    ... = 1 + i + 1           : by rw sub_neg_eq_add
    ... = 2 + i              : by ring
}

lemma location_of_point_2_1 :
  ∃ x y : ℝ, 2 + i = complex.mk x y ∧ x > 0 ∧ y > 0 :=
by {
  use 2,
  use 1,
  split,
  { 
    -- Show the complex number is (2, 1)
    exact rfl 
  },
  split,
  {
    -- Show x > 0
    linarith
  },
  {
    -- Show y > 0
    linarith
  }
}

theorem complex_point_location :
  ∃ x y : ℝ, (1 + i^2017 - i^2018 = complex.mk x y ∧ x > 0 ∧ y > 0) :=
begin
  have h : 1 + i^2017 - i^2018 = 2 + i,
  {
    -- Use the previous lemma to get the value
    exact power_of_i_2017_2018 
  },
  rw h,
  exact location_of_point_2_1
end

end complex_point_location_l526_526011


namespace find_a_l526_526910

theorem find_a (a : ℝ) : (∀ x : ℝ, sqrt (x + a) >= x) ∧ (interval_length |a| 4) ↔ (a = 4/9 ∨ a = (1 - sqrt 5) / 8) :=
sorry

end find_a_l526_526910


namespace summer_camp_skills_l526_526417

theorem summer_camp_skills
  (x y z a b c : ℕ)
  (h1 : x + y + z + a + b + c = 100)
  (h2 : y + z + c = 42)
  (h3 : z + x + b = 65)
  (h4 : x + y + a = 29) :
  a + b + c = 64 :=
by sorry

end summer_camp_skills_l526_526417


namespace coefficient_x5_in_expansion_l526_526185

theorem coefficient_x5_in_expansion :
  (∃ c : ℤ, c = binomial 9 4 * (3 * real.sqrt 2) ^ 4 ∧ c = 40824) → true := 
by
  intro h,
  have : 40824 = 40824, from rfl,
  exact h sorry

end coefficient_x5_in_expansion_l526_526185


namespace range_of_a_plus_3b_l526_526003

theorem range_of_a_plus_3b :
  ∀ (a b : ℝ),
    -1 ≤ a + b ∧ a + b ≤ 1 ∧ 1 ≤ a - 2 * b ∧ a - 2 * b ≤ 3 →
    -11 / 3 ≤ a + 3 * b ∧ a + 3 * b ≤ 7 / 3 :=
by
  sorry

end range_of_a_plus_3b_l526_526003


namespace simple_interest_rate_l526_526168

theorem simple_interest_rate (P T A R : ℝ) (hT : T = 15) (hA : A = 4 * P)
  (hA_simple_interest : A = P + (P * R * T / 100)) : R = 20 :=
by
  sorry

end simple_interest_rate_l526_526168


namespace geometric_sequence_general_term_l526_526968

noncomputable def general_term (n : ℕ) := (1 / 2) ^ (n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℚ) (q : ℚ)
  (h_seq : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 + a 4 = 5 / 8)
  (h3 : a 3 = 1 / 4)
  (hq : q < 1) :
  ∀ n, a n = (1 / 2) ^ (n - 1) :=
by 
  sorry

end geometric_sequence_general_term_l526_526968


namespace tangent_of_angle_C_l526_526420

theorem tangent_of_angle_C (A B C : ℝ) (hABC : A + B + C = π)
  (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (h_condition : |cos A ^ 2 - 1 / 4| + (tan B - sqrt 3) ^ 2 = 0) :
  tan C = sqrt 3 :=
  sorry

end tangent_of_angle_C_l526_526420


namespace problem_solution_l526_526302

noncomputable def nth_term_mod (n : ℕ) : ℕ :=
  let required_pos := 2500
  let sequence_value := Nat.ceil ((-1 + Real.sqrt (1 + 4 * required_pos)) / 2).toNat
  sequence_value % 7

theorem problem_solution : nth_term_mod 2500 = 1 :=
by
  sorry

end problem_solution_l526_526302


namespace max_loss_Cara_Janet_l526_526169

theorem max_loss_Cara_Janet 
    (total_money : ℕ) 
    (ratio_Cara : ℕ) 
    (ratio_Janet : ℕ) 
    (ratio_total : ℕ)
    (price_high : ℝ)
    (price_low : ℝ)
    (commission : ℝ) 
    (total_earning_if_sold : ℝ) 
    : (total_money = 110) →
      (ratio_Cara = 4) →
      (ratio_Janet = 5) →
      (ratio_total = 22) →
      (price_high = 2) →
      (price_low = 0.8 * price_high) → -- 80% of high price
      (commission = 0.1) →
      (total_earning_if_sold = 22 * price_low * (1 - commission)) →
      (let combined_money := (ratio_Cara + ratio_Janet) * (total_money / ratio_total),
           max_loss := combined_money - total_earning_if_sold in
       max_loss = 13.32) :=
by
  intros,
  sorry

end max_loss_Cara_Janet_l526_526169


namespace find_m_l526_526891

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * x^b

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, 0 < x → x < y → f y < f x

theorem find_m (m : ℝ) :
  let f : ℝ → ℝ := λ x, (m^2 - m - 1) * x^(m^2 + m - 3) in
  is_power_function f ∧ is_decreasing f → m = -1 :=
begin
  sorry
end

end find_m_l526_526891


namespace problem_solution_l526_526159

variable (f : ℝ → ℝ)
variable (x₁ x₂ : ℝ)

def problem_conditions : Prop :=
  (∀ x : ℝ, f (4 - x) = -f x) ∧
  (∀ x : ℝ, 2 < x → monotone f) ∧
  (x₁ + x₂ < 4) ∧
  ((x₁ - 2) * (x₂ - 2) < 0)

theorem problem_solution (h : problem_conditions f x₁ x₂) : f x₁ + f x₂ < 0 :=
sorry

end problem_solution_l526_526159


namespace fixed_point_through_given_parabola_locus_of_projection_l526_526913

-- Definitions based on given conditions in the problem
def parabola (p : ℝ) : set (ℝ × ℝ) := {point | ∃ y : ℝ, point = (y^2 / (2 * p), y)}

def is_perpendicular (a b : (ℝ × ℝ)) : Prop :=
  let ⟨x1, y1⟩ := a
  let ⟨x2, y2⟩ := b
  x1 ≠ 0 ∧ x2 ≠ 0 ∧ y1 / x1 * y2 / x2 = -1

-- Fixed point problem (Part I)
theorem fixed_point_through_given_parabola 
  (p : ℝ) (A B : ℝ × ℝ) (HaA : A ∈ parabola p) (HbB : B ∈ parabola p)
  (perpOA : is_perpendicular (fst A, snd A) (fst B, snd B)) :
    ∃ M₀: ℝ × ℝ, M₀ = (2 * p, 0) ∧ line A B = line (2 * p, 0) := sorry

-- Locus of projection problem (Part II)
theorem locus_of_projection (p : ℝ) (H : ℝ × ℝ) (A B: ℝ × ℝ) 
  (HaA : A ∈ parabola p) (HbB : B ∈ parabola p) (perpOA : is_perpendicular A B)
  (projectionH : H = proj_AB_O (line A B) (0, 0)):
    (fst H)^2 + (snd H)^2 - 2 * p * (fst H) = 0 := sorry

-- Projection function (not defined in detail)
noncomputable def proj_AB_O (L : ℝ × ℝ → Prop) (O : ℝ × ℝ) : ℝ × ℝ := sorry

end fixed_point_through_given_parabola_locus_of_projection_l526_526913


namespace inequality_proof_l526_526854

-- Define constants
def a : ℝ := 0.1 * Real.exp 0.1
def b : ℝ := 1 / 9
def c : ℝ := -Real.log 0.9

-- Assert the inequalities
theorem inequality_proof : c < a ∧ a < b :=
by sorry

end inequality_proof_l526_526854


namespace geometric_mean_sine_inequality_in_triangle_l526_526975

theorem geometric_mean_sine_inequality_in_triangle
  (A B C: ℝ)
  (a b c d x: ℝ)
  (h1: ∀ (Δ : Triangle), Δ.has_point D (segment AB))
  (h2: D = point_on_segment AB x)
  (h3: CD^2 = AD * BD := by library_search):
  sin A * sin B ≤ sin^2 (C / 2) :=
sorry

end geometric_mean_sine_inequality_in_triangle_l526_526975


namespace excess_common_fraction_l526_526718

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l526_526718


namespace total_worth_is_correct_l526_526521

-- Define the conditions
def rows : ℕ := 4
def gold_bars_per_row : ℕ := 20
def worth_per_gold_bar : ℕ := 20000

-- Define the calculated values
def total_gold_bars : ℕ := rows * gold_bars_per_row
def total_worth_of_gold_bars : ℕ := total_gold_bars * worth_per_gold_bar

-- Theorem statement to prove the correct total worth
theorem total_worth_is_correct : total_worth_of_gold_bars = 1600000 := by
  sorry

end total_worth_is_correct_l526_526521


namespace domain_of_f_l526_526319

theorem domain_of_f (x : ℝ) : (1 < x ∧ x < 4) ↔ ∃ y : ℝ, f y = lg (4 - y) + 1 / (sqrt (y - 1)) ∧ 1 < y ∧ y < 4 :=
by
  sorry

end domain_of_f_l526_526319


namespace integral_min_value_correct_l526_526786

noncomputable def integral_min_value (a b : ℝ) : ℝ :=
  ∫ x in 0..1, (Real.sqrt x - (a + b * x))^2

theorem integral_min_value_correct :
  (∫ x in 0..1, (Real.sqrt x - ((4:ℝ)/15 + (4:ℝ)/5 * x))^2) = (1:ℝ) / 450 :=
by
  sorry

end integral_min_value_correct_l526_526786


namespace financier_invariant_l526_526967

theorem financier_invariant (D A : ℤ) (hD : D = 1 ∨ D = 10 * (A - 1) + D ∨ D = D - 1 + 10 * A)
  (hA : A = 0 ∨ A = A + 10 * (1 - D) ∨ A = A - 1):
  (D - A) % 11 = 1 := 
sorry

end financier_invariant_l526_526967


namespace quadrilateral_axis_of_symmetry_passe_vertex_two_m_gon_axis_of_symmetry_passes_vertex_l526_526517

-- Define a quadrilateral and a 2m-gon with certain symmetry properties
structure Quadrilateral (V : Type) :=
(vertices : V)
(has_symmetry : Prop)
(axis_of_symmetry : V → V → Prop)

structure TwoMGon (V : Type) :=
(vertices : V)
(has_symmetry : Prop)
(axis_of_symmetry : V → V → Prop)

-- The problem requires proving the symmetry properties of quadrilaterals and 2m-gons
theorem quadrilateral_axis_of_symmetry_passe_vertex (V : Type) [Quadrilateral V] :
  ∀ v1 v2 : V, axis_of_symmetry v1 v2 → axis_of_symmetry v2 v3 :=
sorry

theorem two_m_gon_axis_of_symmetry_passes_vertex (V : Type) [TwoMGon V] :
  ∀ v1 v2 : V, axis_of_symmetry v1 v2 → axis_of_symmetry v2 v3 :=
sorry

end quadrilateral_axis_of_symmetry_passe_vertex_two_m_gon_axis_of_symmetry_passes_vertex_l526_526517


namespace compare_constants_l526_526841

theorem compare_constants :
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  in c < a ∧ a < b :=
by
  let a := 0.1 * Real.exp 0.1
  let b := 1 / 9
  let c := -Real.log 0.9
  sorry

end compare_constants_l526_526841


namespace find_positive_integer_pairs_l526_526315

noncomputable def satisfies_conditions (m n : ℕ) : Prop :=
  1 ≤ m^n - n^m ∧ m^n - n^m ≤ m * n

theorem find_positive_integer_pairs :
  { (m, n) | ∀ (m n : ℕ), satisfies_conditions m n → 
    (m, n) = (m, 1) ∧ 2 ≤ m ∨ 
    (m, n) = (2, 5) ∨ 
    (m, n) = (3, 2) } :=
by
  sorry

end find_positive_integer_pairs_l526_526315


namespace solve_fraction_zero_l526_526063

theorem solve_fraction_zero (x : ℝ) (h : (x + 5) / (x - 2) = 0) : x = -5 :=
by
  sorry

end solve_fraction_zero_l526_526063


namespace socks_total_is_51_l526_526096

-- Define initial conditions for John and Mary
def john_initial_socks : Nat := 33
def john_thrown_away_socks : Nat := 19
def john_new_socks : Nat := 13

def mary_initial_socks : Nat := 20
def mary_thrown_away_socks : Nat := 6
def mary_new_socks : Nat := 10

-- Define the total socks function
def total_socks (john_initial john_thrown john_new mary_initial mary_thrown mary_new : Nat) : Nat :=
  (john_initial - john_thrown + john_new) + (mary_initial - mary_thrown + mary_new)

-- Statement to prove
theorem socks_total_is_51 : 
  total_socks john_initial_socks john_thrown_away_socks john_new_socks 
              mary_initial_socks mary_thrown_away_socks mary_new_socks = 51 := 
by
  sorry

end socks_total_is_51_l526_526096


namespace smallest_n_l526_526265

theorem smallest_n (n : ℕ) (h1: n ≥ 100) (h2: n ≤ 999) 
  (h3: (n + 5) % 8 = 0) (h4: (n - 8) % 5 = 0) : 
  n = 123 :=
sorry

end smallest_n_l526_526265


namespace equal_left_right_in_ten_l526_526218

variable (boots : Fin 30 → Bool) -- True means a left boot, False means a right boot

noncomputable def count_left (l r : Nat) : Nat :=
  (Fin.range (r - l + 1)).count (λ i => boots ⟨l + i, sorry⟩)

theorem equal_left_right_in_ten :
  (∃ k : Fin 21, count_left boots k.val (k.val + 9) = 5) :=
by {
  have h_total : (Fin.range 30).count (λ i => boots i) = 15 := sorry,
  have h : (∀ n : Fin (30-10+1), 0 <= count_left boots n.val (n.val + 9) <= 10) := sorry,
  -- Proof details skipped
  sorry,
}

end equal_left_right_in_ten_l526_526218


namespace area_of_enclosing_square_l526_526607

theorem area_of_enclosing_square (r : ℝ) (h : r = 5) : ∃ A : ℝ, A = 100 :=
by 
  let d := 2 * r
  let s := d
  let A := s ^ 2
  have hA : A = 100, from sorry
  exact ⟨A, hA⟩

end area_of_enclosing_square_l526_526607


namespace car_rental_cost_elena_l526_526310

-- Define the conditions
def daily_rental_cost (days : ℕ) := 30 * days
def mileage_cost (miles : ℕ) := 0.25 * miles
def total_cost (days : ℕ) (miles : ℕ) := daily_rental_cost days + mileage_cost miles

-- Prove that Elena pays $215 given the conditions
theorem car_rental_cost_elena : total_cost 3 500 = 215 := by
  sorry

end car_rental_cost_elena_l526_526310


namespace series_sum_l526_526493

noncomputable def r : ℝ :=
classical.some (exists_unique (λ x : ℝ, 0 < x ∧ x^3 + (2/5) * x - 1 = 0))

theorem series_sum (r_pos : 0 < r) (h : r^3 + (2/5) * r - 1 = 0) :
  let S := ∑' n, (n + 1) * r^(3 * n + 2) in
  S = 25 / 4 :=
sorry

end series_sum_l526_526493


namespace area_of_square_efgh_proof_l526_526228

noncomputable def area_of_square_efgh : ℝ :=
  let original_square_side_length := 3
  let radius_of_circles := (3 * Real.sqrt 2) / 2
  let efgh_side_length := original_square_side_length + 2 * radius_of_circles 
  efgh_side_length ^ 2

theorem area_of_square_efgh_proof :
  area_of_square_efgh = 27 + 18 * Real.sqrt 2 :=
by
  sorry

end area_of_square_efgh_proof_l526_526228


namespace area_region_T_l526_526140

-- Define the structure of the rhombus
structure Rhombus :=
  (P Q R S : Point)
  (side_length : ℝ)
  (angle_Q : ℝ)
  (is_right_angle : angle_Q = 90)
  (len_consistent : side_length = 3)

-- Define the region T within the rhombus PQRS
def region_T (r : Rhombus) : Set Point :=
  {p : Point | dist p r.Q < dist p r.P ∧ dist p r.Q < dist p r.R ∧ dist p r.Q < dist p r.S}

-- Main theorem stating the area of region T
theorem area_region_T (r : Rhombus) (h : r.is_right_angle) (h_len : r.len_consistent) :
  area (region_T r) = 2.25 :=
sorry

end area_region_T_l526_526140


namespace rectangular_field_area_l526_526153

-- Definitions from conditions in step a)
def length_of_field (L : ℝ) := L
def breadth_of_field (L : ℝ) := 0.60 * L
def perimeter (L : ℝ) := 2 * L + 2 * breadth_of_field L
def area_of_field (L : ℝ) := L * breadth_of_field L

-- Correct answer in step b)
theorem rectangular_field_area :
  (∃ L : ℝ, perimeter L = 800) → (∃ L : ℝ, area_of_field L = 37500) :=
begin
  intro h,
  cases h with L hL,
  use L,
  sorry
end

end rectangular_field_area_l526_526153


namespace cistern_empty_time_l526_526239

theorem cistern_empty_time (T : ℝ) : 
  (∀ (filling_rate emptying_rate net_rate: ℝ),
    filling_rate = 1 / 2 →
    emptying_rate = 1 / T →
    net_rate = 1 / 4 →
    filling_rate - emptying_rate = net_rate
  ) → 
  T = 4 :=
by
  intro H
  have h1 : 1 / 2 - 1 / T = 1 / 4 := by apply H; norm_num
  sorry

end cistern_empty_time_l526_526239


namespace value_of_f_l526_526374

noncomputable def f : ℝ → ℝ
| x := if h : x ≥ 3 then (1 / 3) ^ x else f (x + 1)

theorem value_of_f (x : ℝ) (hx : x = 2 + Real.log 2 / Real.log 3) : 
  f x = 1 / 54 := by
  sorry

end value_of_f_l526_526374


namespace original_number_of_cats_l526_526674

theorem original_number_of_cats (C : ℕ) : 
  (C - 600) / 2 = 600 → C = 1800 :=
by
  sorry

end original_number_of_cats_l526_526674


namespace pablo_blocks_taller_l526_526128

theorem pablo_blocks_taller (x : ℕ) : 
  let stack1 := 5 in
  let stack2 := 5 + x in
  let stack3 := x in
  let stack4 := x + 5 in
  stack1 + stack2 + stack3 + stack4 = 21 →
  x = 2 :=
by
  sorry

end pablo_blocks_taller_l526_526128


namespace xiao_wang_ways_to_make_8_cents_l526_526634

theorem xiao_wang_ways_to_make_8_cents :
  (∃ c1 c2 c5 : ℕ, c1 ≤ 8 ∧ c2 ≤ 4 ∧ c5 ≤ 1 ∧ c1 + 2 * c2 + 5 * c5 = 8) → (number_of_ways_to_make_8_cents = 7) :=
sorry

end xiao_wang_ways_to_make_8_cents_l526_526634


namespace part1_part2_l526_526083

variable (a b c p r r_A r_B r_C : ℝ)
variable (ΔABC : ∀ (A B C : Type), A → B → C → Prop) -- Triangle ABC

-- Conditions
variables (h1 : ∀ A B C, ΔABC A B C → ∃ (C : ℝ), C = 90)
variables (h2 : ∀ (A B C : ℝ), ΔABC A B C → A = a → B = b → C = c)
variables (h3 : p = (a + b + c) / 2)
variables (h4 : r = (a + b - c) / 2)
variables (h5 : r_A = p - a)
variables (h6 : r_B = p - b)
variables (h7 : r_C = p - c)

-- Proof problem: Part 1
theorem part1 : p * (p - c) = (p - a) * (p - b) := sorry

-- Proof problem: Part 2
theorem part2 : r + r_A + r_B + r_C = 2 * p := sorry

end part1_part2_l526_526083


namespace area_of_trapezium_l526_526777

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
by sorry

end area_of_trapezium_l526_526777


namespace sum_of_possible_k_l526_526434

theorem sum_of_possible_k (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  j > 0 ∧ k > 0 → 
  ∑ i in {20, 12, 8, 6, 5}.to_finset, i = 51 :=
by
  sorry

end sum_of_possible_k_l526_526434


namespace paint_remaining_after_two_days_l526_526656

-- Define the conditions
def original_paint_amount := 1
def paint_used_day1 := original_paint_amount * (1/4)
def remaining_paint_after_day1 := original_paint_amount - paint_used_day1
def paint_used_day2 := remaining_paint_after_day1 * (1/2)
def remaining_paint_after_day2 := remaining_paint_after_day1 - paint_used_day2

-- Theorem to be proved
theorem paint_remaining_after_two_days :
  remaining_paint_after_day2 = (3/8) * original_paint_amount := sorry

end paint_remaining_after_two_days_l526_526656


namespace non_congruent_squares_in_6_by_6_grid_l526_526389

theorem non_congruent_squares_in_6_by_6_grid : 
  let n := 6 in
  (∑ i in Range (n-1), (n-i)*(n-i) + 
   ∑ i in Range (n-1), (n-i)*(n-i) + 
   ∑ i in Range (5), 5 * 6 - i)%nat = 155 :=
by 
  let n := 6
  have h1 : ∑ i in Range (n-1), (n - i)^2 = 
    (n-1)^2 + (n-2)^2 + ... + 1 = 25 + 16 + 9 + 4 + 1 := sorry
  have h2 : ∑ i in Range (n-1), (n - i)^2 = 
    (5)^2 + (4)^2 + ... + 1^2 = 25 + 16 + 9 + 4 + 1 := sorry
  have h3 : ∑ i in Range (5), 5 * (6 - i) = 
    (5*5) + (5*4) := sorry
  show (∑ i in Range (n-1), (n-i)*(n-i) + 
       ∑ i in Range (n-1), (n-i)*(n-i) + 
       ∑ i in Range (5), 5 * (6 - i)) = 155 := by sorry

end non_congruent_squares_in_6_by_6_grid_l526_526389


namespace sum_of_x_satisfying_sqrt_eq_9_l526_526196

theorem sum_of_x_satisfying_sqrt_eq_9 :
  (∀ x : ℝ, sqrt ((x + 5)^2) = 9 → x = 4 ∨ x = -14) →
  (4 + (-14) = -10) :=
by
  intros h
  have h1 : sqrt ((4 + 5)^2) = 9 := by norm_num
  have h2 : sqrt ((-14 + 5)^2) = 9 := by norm_num
  apply h 4 h1
  apply h (-14) h2
  norm_num
  sorry

end sum_of_x_satisfying_sqrt_eq_9_l526_526196


namespace trig_identity_l526_526803

theorem trig_identity (x : ℝ) (h : Real.sin (x + π / 3) = 1 / 3) : 
  Real.sin ((5 * π) / 3 - x) - Real.cos (2 * x - π / 3) = 4 / 9 := 
by 
  sorry

end trig_identity_l526_526803


namespace julia_parrot_weeks_l526_526986

theorem julia_parrot_weeks :
  (∃ P : ℕ, ∀ (total_weeks : ℕ) (rabbit_weeks : ℕ) (total_cost : ℕ) (rabbit_weekly_cost : ℕ) (total_weekly_cost : ℕ),
    total_weeks = 5 ∧ total_cost = 114 ∧ rabbit_weekly_cost = 12 ∧ total_weekly_cost = 30 →
    (total_cost - rabbit_weeks * rabbit_weekly_cost) / (total_weekly_cost - rabbit_weekly_cost) = P) →
  ∃ P : ℕ, P = 3 :=
begin
  intro h,
  cases h with P hP,
  use 3,
  specialize hP 5 5 114 12 30,
  simp at hP,
  exact hP,
end

end julia_parrot_weeks_l526_526986


namespace paths_have_same_parity_l526_526676

variable {V : Type}
variable [DecidableEq V]
variable (G : SimpleGraph V)
variable [Fintype V]

/-- All vertices of the map have even degree -/
def even_degree_vertices : Prop :=
  ∀ v : V, even (G.degree v)

/-- Definition of Eulerian path assuming the graph has at least one Eulerian path -/
def is_Eulerian_path (G : SimpleGraph V) (p : list V) : Prop :=
  (p.head? = some (p.head!) ∧ p.ilast? = some (p.ilast!)) ∧
  ∀ v ∈ p, even (G.degree v)

/-- The main theorem stating the problem requirements and conclusion -/
theorem paths_have_same_parity
  (h_even : even_degree_vertices G)
  (S0 S1 : V)
  (p q : ℕ)
  (path1 : list V) 
  (path2 : list V)
  (h_path1 : is_Eulerian_path G path1)
  (h_p : path1.length = p)
  (h_path2 : is_Eulerian_path G path2)
  (h_q : path2.length = q) :
  (p % 2 = q % 2) := 
by
  sorry

end paths_have_same_parity_l526_526676


namespace length_of_AC_l526_526513

noncomputable def length_AC (α : ℝ) : ℝ :=
sqrt (128 - 128 * cos (α / 8))

theorem length_of_AC (A B C : Point) (r : ℝ) (AB : ℝ) (ratio : ℕ) (h_circle : is_on_circle A r)
  (h_circle_B : is_on_circle B r) (h_radius : r = 8) (h_AB : dist A B = 10)
  (h_arc : arc_division C A B ratio (minor_arc := true) (ratio := 1 / 3)) : 
  dist A C = length_AC ?α :=
sorry

end length_of_AC_l526_526513


namespace sqrt_operations_validity_l526_526623

theorem sqrt_operations_validity :
  ¬(sqrt 2 + sqrt 3 = sqrt 5) ∧
  ¬(sqrt 20 = 2 * sqrt 10) ∧
  (sqrt 3 * sqrt 5 = sqrt 15) ∧
  ¬(sqrt ((-3 : ℤ)^2) = -3) :=
by
  sorry

end sqrt_operations_validity_l526_526623


namespace complex_number_simplification_l526_526745

theorem complex_number_simplification (i : ℂ) (hi : i^2 = -1) : i - (1 / i) = 2 * i :=
by
  sorry

end complex_number_simplification_l526_526745


namespace compare_abc_l526_526827
open Real

noncomputable def a : ℝ := 0.1 * exp(0.1)
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - log 0.9

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l526_526827


namespace shanna_tomato_ratio_l526_526144

-- Define the initial conditions
def initial_tomato_plants : ℕ := 6
def initial_eggplant_plants : ℕ := 2
def initial_pepper_plants : ℕ := 4
def pepper_plants_died : ℕ := 1
def vegetables_per_plant : ℕ := 7
def total_vegetables_harvested : ℕ := 56

-- Define the number of tomato plants that died
def tomato_plants_died (total_vegetables : ℕ) (veg_per_plant : ℕ) (initial_tomato : ℕ) 
  (initial_eggplant : ℕ) (initial_pepper : ℕ) (pepper_died : ℕ) : ℕ :=
  let surviving_plants := total_vegetables / veg_per_plant
  let surviving_pepper := initial_pepper - pepper_died
  let surviving_tomato := surviving_plants - (initial_eggplant + surviving_pepper)
  initial_tomato - surviving_tomato

-- Define the ratio
def ratio_tomato_plants_died_to_initial (tomato_died : ℕ) (initial_tomato : ℕ) : ℚ :=
  (tomato_died : ℚ) / (initial_tomato : ℚ)

theorem shanna_tomato_ratio :
  ratio_tomato_plants_died_to_initial (tomato_plants_died total_vegetables_harvested vegetables_per_plant 
    initial_tomato_plants initial_eggplant_plants initial_pepper_plants pepper_plants_died) initial_tomato_plants 
  = 1 / 2 := by
  sorry

end shanna_tomato_ratio_l526_526144


namespace p_at_zero_l526_526492

-- We state the conditions: p is a polynomial of degree 6, and p(3^n) = 1/(3^n) for n = 0 to 6
def p : Polynomial ℝ := sorry

axiom p_degree : p.degree = 6
axiom p_values : ∀ (n : ℕ), n ≤ 6 → p.eval (3^n) = 1 / (3^n)

-- We want to prove that p(0) = 29523 / 2187
theorem p_at_zero : p.eval 0 = 29523 / 2187 := by sorry

end p_at_zero_l526_526492


namespace sum_of_possible_k_l526_526436

theorem sum_of_possible_k (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  j > 0 ∧ k > 0 → 
  ∑ i in {20, 12, 8, 6, 5}.to_finset, i = 51 :=
by
  sorry

end sum_of_possible_k_l526_526436


namespace circle_center_sum_l526_526303

theorem circle_center_sum {h k : ℝ} :
  (∀ x y : ℝ, x^2 + y^2 - 8 * x - 14 * y + 55 = ((x - h)^2 + (y - k)^2) - h^2 - k^2 + 55) →
  (h = 4) → (k = 7) → h + k = 11 :=
by
  intros h_eq k_eq
  rw [h_eq, k_eq]
  norm_num
  sorry

end circle_center_sum_l526_526303


namespace tetrahedron_net_exists_l526_526747

-- Definitions for the conditions given in the problem
def is_tetrahedron (tri : Type) : Prop :=
  ∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ (a^2 + b^2 = c^2)

-- The main statement to prove
theorem tetrahedron_net_exists : is_tetrahedron ℕ → True :=
by {
  intro h,
  exact True.intro -- Placeholder to skip the proof
}

end tetrahedron_net_exists_l526_526747


namespace find_a_l526_526939

theorem find_a (a x y : ℝ)
    (h1 : a * x - 5 * y = 5)
    (h2 : x / (x + y) = 5 / 7)
    (h3 : x - y = 3) :
    a = 3 := 
by 
  sorry

end find_a_l526_526939


namespace expression_value_l526_526757

theorem expression_value :
  (100 - (3000 - 300) + (3000 - (300 - 100)) = 200) := by
  sorry

end expression_value_l526_526757


namespace find_constant_term_l526_526885

theorem find_constant_term (a : ℝ) (h : a > 0)
  (hexp : ∃ (r : ℕ), (∑ i in finset.range (6 + 1), 
    (nat.choose 6 i) * (-1)^i * a^(6 - i) * (x)^(3 * i - 6) / 2) = 15) : a = 1 :=
sorry

end find_constant_term_l526_526885


namespace compare_values_l526_526811

variable (a b c : ℝ)

noncomputable def a_def : ℝ := 0.1 * Real.exp 0.1
noncomputable def b_def : ℝ := 1 / 9
noncomputable def c_def : ℝ := - Real.log 0.9

theorem compare_values : 
  c < a ∧ a < b := 
begin
  let a := 0.1 * Real.exp 0.1,
  let b := 1 / 9,
  let c := - Real.log 0.9,
  have h1 : c < a, sorry,
  have h2 : a < b, sorry,
  exact ⟨h1, h2⟩
end

end compare_values_l526_526811


namespace range_of_m_l526_526881

-- Definitions of propositions
def is_circle (m : ℝ) : Prop :=
  ∃ x y : ℝ, (x - m)^2 + y^2 = 2 * m - m^2 ∧ 2 * m - m^2 > 0

def is_hyperbola_eccentricity_in_interval (m : ℝ) : Prop :=
  1 < Real.sqrt (1 + m / 5) ∧ Real.sqrt (1 + m / 5) < 2

-- Proving the main statement
theorem range_of_m (m : ℝ) (h1 : is_circle m ∨ is_hyperbola_eccentricity_in_interval m)
  (h2 : ¬ (is_circle m ∧ is_hyperbola_eccentricity_in_interval m)) : 2 ≤ m ∧ m < 15 :=
sorry

end range_of_m_l526_526881


namespace circle_a_ge_3_l526_526964

noncomputable def circle_center_radius (a : ℝ) : Prop :=
∃ (x y : ℝ), (x - a)^2 + (y + a - 3)^2 = 1

noncomputable def point_on_circle (a x y : ℝ) : Prop :=
(x - a)^2 + (y + a - 3)^2 = 1 

noncomputable def distance_OM (a : ℝ) : ℝ :=
real.sqrt (a^2 + (a - 3)^2)

noncomputable def circle_condition (a : ℝ) : Prop := 
∀ (x y : ℝ), point_on_circle a x y → (distance_OM a - 1 ≥ 2)

theorem circle_a_ge_3 (a : ℝ) (h : circle_center_radius a) (a_gt_zero : a > 0) : 
  (a ≥ 3) :=
sorry

end circle_a_ge_3_l526_526964


namespace sqrt_addition_l526_526201

theorem sqrt_addition :
  (Real.sqrt (49 + 81) + Real.sqrt (36 - 9) = Real.sqrt 130 + 3 * Real.sqrt 3) := 
by sorry

end sqrt_addition_l526_526201


namespace quadratic_to_vertex_form_l526_526179

-- Define the given quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Define the vertex form of the quadratic function.
def vertex_form (x : ℝ) : ℝ := (x - 1)^2 + 2

-- State the equivalence we want to prove.
theorem quadratic_to_vertex_form :
  ∀ x : ℝ, quadratic_function x = vertex_form x :=
by
  intro x
  show quadratic_function x = vertex_form x
  sorry

end quadratic_to_vertex_form_l526_526179


namespace meet_time_approx_l526_526561

/-
  Conditions:
  - The jogging track in a sports complex is 1000 meters in circumference.
  - Deepak's speed is 20 km/hr.
  - His wife's speed is 17 km/hr.
  - They start from the same point and walk in opposite directions.
-/
noncomputable def relative_speed_kmh (v1 v2 : ℕ) : ℕ := v1 + v2

noncomputable def relative_speed_mps (v_kmh : ℕ) : ℚ :=
  (v_kmh * 1000) / 3600

noncomputable def time_to_meet (circumference_m : ℕ) (speed_mps : ℚ) : ℚ :=
  circumference_m / speed_mps

theorem meet_time_approx : 
  time_to_meet 1000 (relative_speed_mps (relative_speed_kmh 20 17)) ≈ 97.28 :=
by
  unfold relative_speed_kmh
  unfold relative_speed_mps
  unfold time_to_meet
  sorry

end meet_time_approx_l526_526561


namespace part1_part2_l526_526419

-- Definitions corresponding to the conditions
def angle_A := 35
def angle_B1 := 40
def three_times_angle_triangle (A B C : ℕ) : Prop :=
  A + B + C = 180 ∧ (A = 3 * B ∨ B = 3 * A ∨ C = 3 * A ∨ A = 3 * C ∨ B = 3 * C ∨ C = 3 * B)

-- Part 1: Checking if triangle ABC is a "three times angle triangle".
theorem part1 : three_times_angle_triangle angle_A angle_B1 (180 - angle_A - angle_B1) :=
  sorry

-- Definitions corresponding to the new conditions
def angle_B2 := 60

-- Part 2: Finding the smallest interior angle in triangle ABC.
theorem part2 (angle_A angle_C : ℕ) :
  three_times_angle_triangle angle_A angle_B2 angle_C → (angle_A = 20 ∨ angle_A = 30 ∨ angle_C = 20 ∨ angle_C = 30) :=
  sorry

end part1_part2_l526_526419


namespace coefficient_x5_in_expansion_l526_526192

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binomial_coefficient n k + binomial_coefficient n (k + 1)

open_locale big_operators

theorem coefficient_x5_in_expansion :
  ∀ (x : ℝ), (x + 3 * real.sqrt 2)^9.coeff 5 = 20412 :=
begin
  -- Your proof goes here.
  sorry
end

end coefficient_x5_in_expansion_l526_526192


namespace cake_divided_into_equal_parts_l526_526648

theorem cake_divided_into_equal_parts (cake_weight : ℕ) (pierre : ℕ) (nathalie : ℕ) (parts : ℕ) 
  (hw_eq : cake_weight = 400)
  (hp_eq : pierre = 100)
  (pn_eq : pierre = 2 * nathalie)
  (parts_eq : cake_weight / nathalie = parts)
  (hparts_eq : parts = 8) :
  cake_weight / nathalie = 8 := 
by
  sorry

end cake_divided_into_equal_parts_l526_526648


namespace fruitful_quadruple_ad_bc_multiple_2019_l526_526289

theorem fruitful_quadruple_ad_bc_multiple_2019 (a b c d : ℕ) (h : ∀ᶠ m in Filter.atTop, Nat.gcd (a * m + b) (c * m + d) = 2019) :
  ∃ k : ℕ, |(a * d - b * c)| = 2019 * k :=
begin
  sorry,
end

end fruitful_quadruple_ad_bc_multiple_2019_l526_526289


namespace range_of_b_l526_526010

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x - a) / (2^x + 1)
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := Real.log (x^2 - b)

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x1 x2 : ℝ, f x1 a ≤ g x2 b) → b ≤ -Real.exp 1 :=
by
  sorry

end range_of_b_l526_526010


namespace shells_ratio_correct_l526_526505

noncomputable def number_of_shells_and_ratio : ℤ :=
  let d := 15 
  let m := 4 * d 
  let a := m + 20 
  let t := 195 
  let v := t - (d + m + a)
  v / a

theorem shells_ratio_correct :
  let d := 15 
  let m := 4 * d 
  let a := m + 20 
  let t := 195 
  let v := t - (d + m + a)
  v / a = 1 / 2 := 
by
  rw [number_of_shells_and_ratio]
  sorry

end shells_ratio_correct_l526_526505


namespace range_of_k_l526_526377

def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x → x < y → y < b → f x ≤ f y

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x - Real.log x

theorem range_of_k (k : ℝ) :
  is_increasing_on_interval (f k) 1 (Real.infinity) ↔ k ≥ 1 :=
sorry

end range_of_k_l526_526377


namespace find_exponent_l526_526054

theorem find_exponent (y : ℕ) (h : (1/8) * (2: ℝ)^36 = (2: ℝ)^y) : y = 33 :=
by sorry

end find_exponent_l526_526054


namespace fill_grid_even_possible_l526_526497

theorem fill_grid_even_possible (n : ℕ) (h1 : n > 1) : 
  (∀ (grid : ℕ → ℕ → ℕ), (∀ i j, grid i j ≤ n^2 ∧ grid i j ≥ 1) → 
  (∀ i j, (grid i j + 1 = grid i (j+1) ∨ grid i j + 1 = grid (i+1) j) →
  (∀ i j, ∀ k l, (k ≠ i ∧ l ≠ j ∧ grid i j % n = grid k l % n))) ↔ n % 2 = 0 :=
sorry

end fill_grid_even_possible_l526_526497


namespace evens_in_triangle_l526_526077

theorem evens_in_triangle (a : ℕ → ℕ → ℕ) (h : ∀ i j, a i.succ j = (a i (j - 1) + a i j + a i (j + 1)) % 2) :
  ∀ n ≥ 2, ∃ j, a n j % 2 = 0 :=
  sorry

end evens_in_triangle_l526_526077


namespace johns_new_time_l526_526985

-- Define the initial times of John's attempts
def initial_times : List ℕ := [120, 125, 112, 140, 130]

-- Define the new known time from the additional two attempts
def new_time1 : ℕ := 122

-- Define the median of the new sequence after two additional attempts
def new_median : ℕ := 125

-- Assert that the new time x should satisfy the median condition
theorem johns_new_time (x : ℕ) (times : List ℕ := initial_times ++ [new_time1, x]) 
  (sorted_times : List ℕ := times.qsort (· ≤ ·)) : 
  sorted_times.nth 3 = some new_median → x = 124 :=
sorry

end johns_new_time_l526_526985


namespace sum_reciprocal_squares_roots_l526_526495

-- Define the polynomial P(X) = X^3 - 3X - 1
noncomputable def P (X : ℂ) : ℂ := X^3 - 3 * X - 1

-- Define the roots of the polynomial
variables (r1 r2 r3 : ℂ)

-- State that r1, r2, and r3 are roots of the polynomial
variable (hroots : P r1 = 0 ∧ P r2 = 0 ∧ P r3 = 0)

-- Vieta's formulas conditions for the polynomial P
variable (hvieta : r1 + r2 + r3 = 0 ∧ r1 * r2 + r1 * r3 + r2 * r3 = -3 ∧ r1 * r2 * r3 = 1)

-- The sum of the reciprocals of the squares of the roots
theorem sum_reciprocal_squares_roots : (1 / r1^2) + (1 / r2^2) + (1 / r3^2) = 9 := 
sorry

end sum_reciprocal_squares_roots_l526_526495


namespace _l526_526552

open EuclideanGeometry

noncomputable def orthocenter (A B C H : Point) : Prop :=
  is_perpendicular A H C B ∧ is_perpendicular B H A C ∧ is_perpendicular C H B A

noncomputable def midpoint (P Q M : Point) : Prop :=
  2 • M = P + Q

noncomputable theorem perpendicular_midpoints_to_altitudes {A B C H D E X Y : Point}
  (h_orthocenter : orthocenter A B C H)
  (h_altitude_AD : is_perpendicular A D B C)
  (h_altitude_BE : is_perpendicular B E A C)
  (h_midpoint_X : midpoint A B X)
  (h_midpoint_Y : midpoint C H Y) :
  is_perpendicular X Y D E :=
sorry

end _l526_526552


namespace correct_number_of_sandwiches_l526_526154

def num_breads := 5
def num_meats := 6
def num_cheeses := 4

def total_sandwiches := num_breads * num_meats * num_cheeses

def sandwiches_turkey_swiss := num_breads
def sandwiches_multigrain_turkey := num_cheeses

def sandwiches_bob_can_order := total_sandwiches - sandwiches_turkey_swiss - sandwiches_multigrain_turkey

theorem correct_number_of_sandwiches :
  sandwiches_bob_can_order = 111 :=
by {
  have h_total: total_sandwiches = 120 := rfl,
  have h_turkey_swiss: sandwiches_turkey_swiss = 5 := rfl,
  have h_multigrain_turkey: sandwiches_multigrain_turkey = 4 := rfl,
  unfold sandwiches_bob_can_order,
  rw [h_total, h_turkey_swiss, h_multigrain_turkey],
  norm_num,
  exact rfl,
}

end correct_number_of_sandwiches_l526_526154


namespace worker_daily_wage_after_increase_l526_526672

theorem worker_daily_wage_after_increase (original_wage : ℝ) (increase_percentage : ℝ) :
  original_wage = 28 → increase_percentage = 0.50 → 
  original_wage + (increase_percentage * original_wage) = 42 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end worker_daily_wage_after_increase_l526_526672


namespace sum_of_possible_ks_l526_526449

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l526_526449


namespace bowling_ball_surface_area_l526_526646

def diameter := (17 / 2) -- The diameter of the bowling ball in inches

def radius := diameter / 2 -- Radius derived from the diameter

def surface_area := 4 * Real.pi * radius^2 -- Surface area formula for a sphere

theorem bowling_ball_surface_area : surface_area = 289 * Real.pi / 4 := by
  sorry

end bowling_ball_surface_area_l526_526646


namespace length_of_chord_EF_l526_526469

theorem length_of_chord_EF 
  (r : ℝ) 
  (A B C D G E F : Point) 
  (O N P : Circle) 
  (h1 : diameter O = AB) 
  (h2 : diameter N = BC) 
  (h3 : diameter P = CD) 
  (h4 : radius O = r) 
  (h5 : radius N = r) 
  (h6 : radius P = r) 
  (h7 : tangent AG P G) 
  (h8 : intersects AG N E F) 
  (h9 : AB = BC) 
  (h10 : BC = CD)
  (h11 : r = 20) :
  length E F = 10 * (sqrt 7) := sorry

end length_of_chord_EF_l526_526469


namespace discount_allowed_l526_526628

variable (CP : ℝ)

def MP := 1.4 * CP
def SP := 0.99 * CP

theorem discount_allowed : MP - SP = 0.41 * CP :=
by 
  rw [MP, SP]
  sorry

end discount_allowed_l526_526628


namespace find_y_l526_526534

theorem find_y (t : ℝ) (x : ℝ := 3 - 2 * t) (y : ℝ := 5 * t + 6) (h : x = 1) : y = 11 :=
by
  sorry

end find_y_l526_526534


namespace repeating_decimal_exceeds_finite_decimal_by_l526_526734

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l526_526734


namespace repeating_decimal_exceeds_finite_decimal_by_l526_526727

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end repeating_decimal_exceeds_finite_decimal_by_l526_526727


namespace inequality_proof_l526_526856

-- Define constants
def a : ℝ := 0.1 * Real.exp 0.1
def b : ℝ := 1 / 9
def c : ℝ := -Real.log 0.9

-- Assert the inequalities
theorem inequality_proof : c < a ∧ a < b :=
by sorry

end inequality_proof_l526_526856


namespace visitors_answered_questionnaire_l526_526415

theorem visitors_answered_questionnaire (V : ℕ) (h : (3 / 4 : ℝ) * V = (V : ℝ) - 110) : V = 440 :=
sorry

end visitors_answered_questionnaire_l526_526415


namespace repeating_seventy_two_exceeds_seventy_two_l526_526740

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l526_526740


namespace cos_double_angle_l526_526888
noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

-- Conditions
axiom α_acute : 0 < α ∧ α < π / 2
axiom β_acute : 0 < β ∧ β < π / 2
axiom tan_beta : Real.tan β = 4 / 3
axiom cos_sum_zero : Real.cos(α + β) = 0

-- The theorem to prove
theorem cos_double_angle : Real.cos (2 * α) = 7 / 25 :=
by
  sorry

end cos_double_angle_l526_526888


namespace partition_pairs_exist_l526_526791

def partition (S : Set ℕ) := { π : Set (Set ℕ) // (∀ A ∈ π, A ≠ ∅) ∧ (∀ A B ∈ π, A ≠ B → A ∩ B = ∅) ∧ (⋃₀ π = S) }

noncomputable def size_of_subset (π : partition (Set.univ \ (Finset.range 9).toSet)) (x : ℕ) : ℕ :=
  (π.1.filter (λ s, x ∈ s)).singleton_extraction.some.size

theorem partition_pairs_exist : 
  ∀ (π π' : partition (Set.univ \ (Finset.range 9).toSet)),
  ∃ (x y : ℕ), x ≠ y ∧ size_of_subset π x = size_of_subset π y ∧ size_of_subset π' x = size_of_subset π' y :=
  by sorry

end partition_pairs_exist_l526_526791


namespace substitution_result_l526_526617

-- Conditions
def eq1 (x y : ℝ) : Prop := y = 2 * x - 3
def eq2 (x y : ℝ) : Prop := x - 2 * y = 6

-- The statement to be proven
theorem substitution_result (x y : ℝ) (h1 : eq1 x y) : (x - 4 * x + 6 = 6) :=
by sorry

end substitution_result_l526_526617


namespace repeatingDecimal_exceeds_l526_526700

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l526_526700


namespace ratio_area_shaded_face_l526_526665

theorem ratio_area_shaded_face 
  (V_original : ℝ) (V_removed : ℝ)
  (h1 : V_original = 1000) 
  (h2 : V_removed = 64) : 

  let a := (real.cbrt V_original) -- side of the original cube
  let b := (real.cbrt V_removed) -- side of the removed cube
  let total_surface_area := 3 * (a * a) + 3 * (b * b) + 3 * (a * a - b * b) 
  let shaded_face_area := b * b
  in (shaded_face_area / total_surface_area) = 2 / 75 :=
  sorry

end ratio_area_shaded_face_l526_526665


namespace proof_problem_l526_526830

-- Define the constants a, b, c
def a := 0.1 * Real.exp 0.1
def b := 1 / 9
def c := -Real.log 0.9

-- State the theorem to prove
theorem proof_problem : c < a ∧ a < b := 
by
  sorry

end proof_problem_l526_526830


namespace distance_BC_l526_526972

def point (α : Type*) := α × α × α

noncomputable def distance_squared {α : Type*} [semiring α] (p1 p2 : point α) : α :=
  let (x1, y1, z1) := p1 in
  let (x2, y2, z2) := p2 in
  (x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2

noncomputable def distance {α : Type*} [field α] (p1 p2 : point α) : α :=
  (distance_squared p1 p2).sqrt

def A : point ℝ := (real.sqrt 2, 1, real.sqrt 3)
def B : point ℝ := (0, 1, real.sqrt 3)
def C : point ℝ := (real.sqrt 2, 0, real.sqrt 3)

theorem distance_BC : distance B C = real.sqrt 3 :=
by
  unfold distance,
  unfold distance_squared,
  simp,
  sorry

end distance_BC_l526_526972


namespace sum_possible_values_k_l526_526464

open Nat

theorem sum_possible_values_k (j k : ℕ) (h : (1 / j : ℚ) + 1 / k = 1 / 4) : 
  ∃ ks : List ℕ, (∀ k' ∈ ks, ∃ j', (1 / j' : ℚ) + 1 / k' = 1 / 4) ∧ ks.sum = 51 :=
by
  sorry

end sum_possible_values_k_l526_526464


namespace repeating_decimal_difference_l526_526704

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l526_526704


namespace quotient_DE_AE_correct_l526_526156

-- Definitions of conditions. Adjust definition names as appropriate.
variables {A B C D E F G : Type}
variables [add_comm_group A] [vector_space ℝ A]
variables (AB : A) (BC : A) (CD : A) (DA : A)
variables (AE : A) (BE : A) (GF : A)
variables (BG EF : A)

-- Assuming parallelogram and intersections
axiom parallelogram_ABCD : AB + BC = CD + DA
axiom straight_lines_AE_BE : line A E ∧ line B E
axiom intersection_F : intersect (line B E) (line CD) = F
axiom intersection_G : intersect (line B E) (line AC) = G
axiom BG_eq_EF : BG = EF

-- Defining the quotient
noncomputable
def quotient_DE_AE := 3 - sqrt 5 / 2

-- Proof Statement
theorem quotient_DE_AE_correct : DE / AE = quotient_DE_AE :=
sorry

end quotient_DE_AE_correct_l526_526156


namespace find_line_AB_angle_equality_l526_526908

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the coordinates of the points
def P : ℝ × ℝ := (2, 3)

-- Equation representing the line AB
def line_AB (x y : ℝ) : Prop := 2 * x - 3 * y = 2

-- The conditions: Foci points F1 and F2, and the relationship
variables (F1 F2 : ℝ × ℝ)

-- Problem statement for question 1
theorem find_line_AB :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧
  line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧ A ≠ B :=
sorry

-- Problem statement for question 2
theorem angle_equality :
  ∀ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧
  line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧ A ≠ B →
  let θ_1 := (angle_between (F1, P) (P, A)) in
  let θ_2 := (angle_between (F2, P) (P, B)) in
  θ_1 = θ_2 :=
sorry

end find_line_AB_angle_equality_l526_526908


namespace student_vision_data_correct_l526_526551

noncomputable def student_vision_decimal_approx : ℝ :=
  let L := 4.9
  let V := 10^(-(L - 5))
  approx := 0.8
  sqrt10 := 10^(1/10)
  1 / sqrt10

theorem student_vision_data_correct :
  (L = 4.9) →
  (L = 5 + log10 V) →
  (sqrt10 ≈ 1.259) →
  (V ≈ 0.8) :=
  by
    sorry

end student_vision_data_correct_l526_526551


namespace gcd_15_70_l526_526744

theorem gcd_15_70 : Int.gcd 15 70 = 5 := by
  sorry

end gcd_15_70_l526_526744


namespace repeatingDecimal_exceeds_l526_526698

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l526_526698


namespace inequality_holds_l526_526516

theorem inequality_holds (x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℝ) 
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4)
  (h5 : 0 < x5) (h6 : 0 < x6) (h7 : 0 < x7) (h8 : 0 < x8) 
  (h9 : 0 < x9) :
  (x1 - x3) / (x1 * x3 + 2 * x2 * x3 + x2^2) +
  (x2 - x4) / (x2 * x4 + 2 * x3 * x4 + x3^2) +
  (x3 - x5) / (x3 * x5 + 2 * x4 * x5 + x4^2) +
  (x4 - x6) / (x4 * x6 + 2 * x5 * x6 + x5^2) +
  (x5 - x7) / (x5 * x7 + 2 * x6 * x7 + x6^2) +
  (x6 - x8) / (x6 * x8 + 2 * x7 * x8 + x7^2) +
  (x7 - x9) / (x7 * x9 + 2 * x8 * x9 + x8^2) +
  (x8 - x1) / (x8 * x1 + 2 * x9 * x1 + x9^2) +
  (x9 - x2) / (x9 * x2 + 2 * x1 * x2 + x1^2) ≥ 0 := 
sorry

end inequality_holds_l526_526516


namespace nested_sqrt_eq_l526_526051

theorem nested_sqrt_eq (x : ℝ) (h : x ≥ 0) : sqrt (x * sqrt (x * sqrt (x * sqrt x))) = (x ^ 9) ^ (1 / 4) :=
by
  sorry

end nested_sqrt_eq_l526_526051


namespace average_salary_is_8000_l526_526152

def average_salary_all_workers (A : ℝ) :=
  let total_workers := 30
  let technicians := 10
  let technician_salary := 12000
  let rest_workers := total_workers - technicians
  let rest_salary := 6000
  let total_salary := (technicians * technician_salary) + (rest_workers * rest_salary)
  A = total_salary / total_workers

theorem average_salary_is_8000 : average_salary_all_workers 8000 :=
by
  sorry

end average_salary_is_8000_l526_526152


namespace minimize_f_at_centroid_minimum_f_value_l526_526099

structure Triangle (α : Type*) [MetricSpace α] :=
(A B C : α)

def centroid {α : Type*} [MetricSpace α] (T : Triangle α) : α := 
(p (T.A + T.B + T.C) / 3)

def distance {α : Type*} [MetricSpace α] (P Q : α) : ℝ := 
dist P Q

def f {α : Type*} [MetricSpace α] (T : Triangle α) (P : α) (G : α) : ℝ :=
(distance P T.A) * (distance G T.A) + 
(distance P T.B) * (distance G T.B) + 
(distance P T.C) * (distance G T.C)

theorem minimize_f_at_centroid {α : Type*} [MetricSpace α] 
(T : Triangle α) : ∀ P : α, 
f T P (centroid T) ≥ f T (centroid T) (centroid T) :=
sorry

theorem minimum_f_value {α : Type*} 
(T : Triangle ℝ) (a b c : ℝ) (h₁ : dist T.A T.B = a) (h₂ : dist T.B T.C = b) (h₃ : dist T.C T.A = c) : 
f T (centroid T) (centroid T) = (a^2 + b^2 + c^2) / 3 :=
sorry

end minimize_f_at_centroid_minimum_f_value_l526_526099


namespace coefficient_x5_in_expansion_l526_526187

theorem coefficient_x5_in_expansion :
  (∃ c : ℤ, c = binomial 9 4 * (3 * real.sqrt 2) ^ 4 ∧ c = 40824) → true := 
by
  intro h,
  have : 40824 = 40824, from rfl,
  exact h sorry

end coefficient_x5_in_expansion_l526_526187


namespace sum_of_possible_ks_l526_526448

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l526_526448


namespace vertex_sum_constant_l526_526680

theorem vertex_sum_constant (x y z : ℕ) :
  let A := x + (y + z),
      B := y + (z + x),
      C := z + (x + y) in 
  A = B ∧ B = C := 
by
  sorry

end vertex_sum_constant_l526_526680


namespace circle_intersection_points_l526_526042

noncomputable def circles : Prop :=
  ∃ m c : ℝ,
    (∃ A : ℝ × ℝ, A = (1, 3)) ∧
    (∃ B : ℝ × ℝ, B = (m, 1)) ∧
    (∀ x y : ℝ, centers_on_line : x ∈ {c | 2 * c.1 - c.2 + c.2 = 0}) ∧
    m + c = 1

theorem circle_intersection_points : circles :=
sorry

end circle_intersection_points_l526_526042


namespace repeating_decimal_difference_l526_526709

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l526_526709


namespace find_abc_l526_526992

open Real

theorem find_abc {a b c : ℝ}
  (h1 : b + c = 16)
  (h2 : c + a = 17)
  (h3 : a + b = 18) :
  a * b * c = 606.375 :=
sorry

end find_abc_l526_526992


namespace even_square_is_even_l526_526008

theorem even_square_is_even (a : ℤ) (h : Even (a^2)) : Even a :=
sorry

end even_square_is_even_l526_526008


namespace sum_of_all_possible_k_values_l526_526442

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l526_526442


namespace problem1_problem2_l526_526034

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem problem1 (x : ℝ) : f(x) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3 :=
by
  sorry

theorem problem2 (a : ℝ) : (∃ x : ℝ, f(x) < a^2 - 3 * a) ↔ a < -1 ∨ a > 4 :=
by
  sorry

end problem1_problem2_l526_526034


namespace function_increasing_and_decreasing_intervals_l526_526906

noncomputable def f (x : ℝ) : ℝ := x ^ 3 - (1 / 3) * x ^ 2 - (1 / 2) * x

theorem function_increasing_and_decreasing_intervals :
  let a := 1 / 3
  let b := -1 / 2
  ∃ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = x ^ 3 - 3 * a * x ^ 2 + 2 * b * x) ∧ 
    f 1 = -1 ∧ 
    f' 1 = 0 ∧
    ((∀ x < -1 / 3, f' x > 0) ∧ 
     (∀ x > 1, f' x > 0) ∧ 
     (∀ x ∈ (-1 / 3, 1), f' x < 0)) := 
by 
  sorry

end function_increasing_and_decreasing_intervals_l526_526906


namespace tall_tuna_ratio_l526_526148

theorem tall_tuna_ratio (T : ℕ) (h1 : T + 144 = 432) : T / 144 = 2 :=
by
  have h2 : T = 288 :=
    calc
      T = 432 - 144 : by linarith
  have h3 : T / 144 = 288 / 144 :=
    calc
      T / 144 = 288 / 144 : by rw h2
  rw h3
  exact nat.div_self 144

end tall_tuna_ratio_l526_526148


namespace non_congruent_squares_6x6_grid_l526_526395

def is_square {α : Type} [linear_ordered_field α] (a b c d : (α × α)) : Prop :=
-- A function to check if four points form a square (not implemented here)
sorry

def count_non_congruent_squares (n : ℕ) : ℕ :=
-- Place calculations for counting non-congruent squares on an n x n grid (not implemented here)
sorry

theorem non_congruent_squares_6x6_grid :
  count_non_congruent_squares 6 = 128 :=
sorry

end non_congruent_squares_6x6_grid_l526_526395


namespace transform_CD_eq_C_l526_526624

noncomputable def point := (ℝ × ℝ)

def C := (3, -2) : point
def C' := (-3, 2) : point
def D := (4, -5) : point
def D' := (-4, 5) : point

def rotation_180 (p : point) : point := (-p.1, -p.2)

theorem transform_CD_eq_C'D' :
  rotation_180 C = C' ∧ rotation_180 D = D' :=
by
  -- proof here
  sorry

end transform_CD_eq_C_l526_526624


namespace min_value_of_f_find_A_l526_526376

def f (x : ℝ) : ℝ := 2 * cos x * (cos x + sqrt 3 * sin x) - 1

theorem min_value_of_f : ∃ x : ℝ, f(x) = -2 := 
by
  sorry

variables {A B C a b c : ℝ}
variable (h1 : f(C / 2) = 2)
variable (h2 : a * b = c^2)

theorem find_A (h1 : f(C / 2) = 2) (h2 : a * b = c^2) : A = π / 3 :=
by
  sorry

end min_value_of_f_find_A_l526_526376


namespace cool_is_periodic_sin_is_periodic_not_cool_l526_526501

-- Definitions and Conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f(x) = -f(-x)

def cool (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, is_even (λ x, f(x + a)) ∧ is_odd (λ x, f(x + b))

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ p > 0, ∀ x, f(x + p) = f(x)

-- Statement to prove
theorem cool_is_periodic (f : ℝ → ℝ) : cool f → is_periodic f :=
sorry

-- Example of a periodic function that is not cool
def not_cool_example : ℝ → ℝ := sin

theorem sin_is_periodic_not_cool : is_periodic not_cool_example ∧ ¬ cool not_cool_example :=
sorry

end cool_is_periodic_sin_is_periodic_not_cool_l526_526501


namespace mikes_ride_miles_l526_526122

-- Define the given conditions.
def cost_per_mile : ℕ := 0.25
def annie_miles : ℕ := 22
def starting_fee : ℕ := 2.50
def bridge_toll : ℕ := 5.00

-- Define the cost function for Mike and Annie.
def mike_cost (M : ℕ) : ℕ :=
  starting_fee + cost_per_mile * M

def annie_cost : ℕ :=
  starting_fee + bridge_toll + cost_per_mile * annie_miles

-- State the theorem to be proved.
theorem mikes_ride_miles (M : ℕ) : M = 32 :=
by
  sorry

end mikes_ride_miles_l526_526122


namespace binom_7_2_eq_21_l526_526295

open Nat

theorem binom_7_2_eq_21 : binomial 7 2 = 21 := 
by sorry

end binom_7_2_eq_21_l526_526295


namespace cost_of_superman_game_l526_526585

-- Define the costs as constants
def cost_batman_game : ℝ := 13.60
def total_amount_spent : ℝ := 18.66

-- Define the theorem to prove the cost of the Superman game
theorem cost_of_superman_game : total_amount_spent - cost_batman_game = 5.06 :=
by
  sorry

end cost_of_superman_game_l526_526585


namespace find_s_t_u_l526_526487

variables (a b c : ℝ^3) (s t u : ℝ)

-- Conditions
axiom orthogonal_unit_vectors : ∀ v₁ v₂ : ℝ^3, v₁ ≠ v₂ → v₁ • v₂ = 0
axiom unit_vectors : ∀ v : ℝ^3, v.norm = 1
axiom vector_equation : b = s * (a × b) + t * (b × c) + u * (c × a)
axiom dot_product_condition : b • (a × c) = -1

-- Claim
theorem find_s_t_u : s + t + u = 1 := 
sorry

end find_s_t_u_l526_526487


namespace angle_between_apothem_and_base_plane_l526_526317

-- The problem definition and conditions
variables (α : ℝ)

-- The correct answer as the conclusion
theorem angle_between_apothem_and_base_plane (α : ℝ) :
  ∃ θ : ℝ, θ = arctan ((cot α + sqrt (cot α ^ 2 - 8)) / 2) ∨ θ = arctan ((cot α - sqrt (cot α ^ 2 - 8)) / 2) :=
sorry

end angle_between_apothem_and_base_plane_l526_526317


namespace speedboat_catchup_time_ship_catchup_time_l526_526237

-- Define the speeds of the speedboat and the river current
variables (v1 v2 : ℝ)

-- Define distances and times
@[inline]
def distance_speedboat := (v1 - v2) * 1 + v2 * 1
@[inline]
def speed_turnaround := v1 + v2
@[inline]
def time_to_catch_speedboat := distance_speedboat v1 v2 / (speed_turnaround - v2)

-- The statement asserting that the time to catch up is 1 hour
theorem speedboat_catchup_time (v1 v2 : ℝ) : time_to_catch_speedboat v1 v2 = 1 :=
by {
  unfold time_to_catch_speedboat distance_speedboat speed_turnaround,
  sorry
}

-- Similarly, for the ship
variables (v1_ship v2_ship : ℝ)

def distance_ship := (v1_ship - v2_ship) * 1 + v2_ship * 1
def speed_turnaround_ship := v1_ship + v2_ship
def time_to_catch_ship := distance_ship v1_ship v2_ship / (speed_turnaround_ship - v2_ship)

theorem ship_catchup_time (v1_ship v2_ship : ℝ) : time_to_catch_ship v1_ship v2_ship = 1 :=
by {
  unfold time_to_catch_ship distance_ship speed_turnaround_ship,
  sorry
}

end speedboat_catchup_time_ship_catchup_time_l526_526237


namespace volume_difference_pi_l526_526291

-- Cara's and Dana's cylinder volumes, and the final calculation

noncomputable def volume_difference : ℝ :=
let r_C := 7 / (2 * Real.pi) in
let V_C := Real.pi * r_C^2 * 10 in
let r_D := 10 / (2 * Real.pi) in
let V_D := Real.pi * r_D^2 * 7 in
Real.pi * ((V_D - V_C) / Real.pi)

theorem volume_difference_pi : volume_difference = 26.25 :=
by
  sorry

end volume_difference_pi_l526_526291


namespace luis_drives_10_miles_in_15_minutes_l526_526504

-- Defining the constants given in the conditions
def distance_in_2_hours : ℝ := 80 -- Luis drives 80 miles
def time_in_2_hours : ℝ := 2 -- Luis drives in 2 hours
def time_in_15_minutes : ℝ := 15 / 60 -- 15 minutes converted to hours

-- Defining derived constants
def speed : ℝ := distance_in_2_hours / time_in_2_hours

-- The goal is to prove that Luis drives 10 miles in 15 minutes
theorem luis_drives_10_miles_in_15_minutes : speed * time_in_15_minutes = 10 :=
by
  sorry

end luis_drives_10_miles_in_15_minutes_l526_526504


namespace k_value_correct_l526_526921

theorem k_value_correct (k : ℚ) : 
  let f (x : ℚ) := 4 * x^2 - 3 * x + 5
  let g (x : ℚ) := x^2 + k * x - 8
  (f 5 - g 5 = 20) -> k = 53 / 5 :=
by
  intro h
  sorry

end k_value_correct_l526_526921


namespace angle_C_value_sides_a_b_l526_526947

variables (A B C : ℝ) (a b c : ℝ)

-- First part: Proving the value of angle C
theorem angle_C_value
  (h1 : 2*Real.cos (A/2)^2 + (Real.cos B - Real.sqrt 3 * Real.sin B) * Real.cos C = 1)
  : C = Real.pi / 3 :=
sorry

-- Second part: Proving the values of a and b given c and the area
theorem sides_a_b
  (c : ℝ)
  (h2 : c = 2)
  (h3 : C = Real.pi / 3)
  (area : ℝ)
  (h4 : area = Real.sqrt 3)
  (h5 : 1/2 * a * b * Real.sin C = Real.sqrt 3)
  : a = 2 ∧ b = 2 :=
sorry

end angle_C_value_sides_a_b_l526_526947


namespace inverse_of_f_at_neg2_l526_526941

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the property of the inverse function we need to prove
theorem inverse_of_f_at_neg2 : f (-(3/2)) = -2 :=
  by
    -- Placeholder for the proof
    sorry

end inverse_of_f_at_neg2_l526_526941


namespace remaining_integers_after_removal_l526_526754

theorem remaining_integers_after_removal : 
  let T := (finset.range 100).map (nat.succ)
  in (T.filter (λ n, ¬(n % 2 = 0 ∨ n % 3 = 0 ∨ n % 5 = 0))).card = 26 :=
by 
  sorry

end remaining_integers_after_removal_l526_526754


namespace values_of_a2_a3_general_formula_integer_pairs_l526_526349

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (n : ℕ)

-- Conditions
axiom a1 : a 1 = -2
axiom recurrence_relation : ∀ n : ℕ, 0 < n → a (n + 1) + 3 * S n + 2 = 0
axiom sum_S : ∀ n : ℕ, 0 < n → S n = ∑ i in finset.range n, a (i + 1)

-- Proof 1: Values of a_2 and a_3
theorem values_of_a2_a3 : a 2 = 4 ∧ a 3 = -8 := by sorry

-- Proof 2: General formula for the sequence
theorem general_formula : ∀ n : ℕ, 0 < n → a n = (-2)^n := by sorry

-- Proof 3: Integer pairs (m, n) satisfying a particular equation
theorem integer_pairs (m : ℤ) : ∃ n : ℕ, 0 < n ∧ a n ^ 2 - m * a n = 4 * m + 8 := by sorry

end values_of_a2_a3_general_formula_integer_pairs_l526_526349


namespace a_max_1995_a_min_1995_a_zero_count_1995_l526_526500

def a : ℕ → ℤ
| 1 := 0
| n := if n > 1 then a (n / 2) + (-1) ^ (n * (n - 1) / 2) else 0

-- Proving the maximum value of a(n) for n < 1996 is 9
theorem a_max_1995 : ∃ n < 1996, a n = 9 :=
by sorry

-- Proving the minimum value of a(n) for n < 1996 is -10
theorem a_min_1995 : ∃ n < 1996, a n = -10 :=
by sorry

-- Proving the number of terms a(n) equal to 0 for n < 1996 is 346
theorem a_zero_count_1995 : (finset.filter (λ n, a n = 0) (finset.range 1995)).card = 346 :=
by sorry

end a_max_1995_a_min_1995_a_zero_count_1995_l526_526500


namespace compare_a_b_c_l526_526845

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l526_526845


namespace rate_per_kg_mangoes_l526_526678

theorem rate_per_kg_mangoes:
  (74 * 6 + 9 * x = 975) → x = 59 :=
begin
  sorry
end

end rate_per_kg_mangoes_l526_526678


namespace shaded_region_area_l526_526787

def area_of_shaded_region : ℝ :=
  let square_area := 40 * 40
  let triangle1_rectangle_area := 30 * 30
  let triangle1_area := (1 / 2) * 30 * 10
  let triangle2_area := (1 / 2) * 30 * 20
  square_area - (triangle1_rectangle_area + triangle1_area + triangle2_area)

theorem shaded_region_area :
  let square_corners := [(0,0), (40,0), (40,40), (0,40)]
  let shaded_vertices := [(0,0), (10,0), (40,30), (40,40), (30,40), (0,20)]
  area_of_shaded_region = 250 := by
  sorry

end shaded_region_area_l526_526787


namespace ratio_L_d_eq_2sqrt3_l526_526670

-- Definitions from the conditions
def thin_uniform_rod (m L : ℝ) : Prop := m > 0 ∧ L > 0
def rotational_inertia_center (m L : ℝ) : ℝ := (1 / 12) * m * L^2

-- The statement to prove
theorem ratio_L_d_eq_2sqrt3 (m L d : ℝ) (h1 : thin_uniform_rod m L) (h2 : rotational_inertia_center m L = m * d^2) : L / d = 2 * Real.sqrt 3 := by
  sorry

end ratio_L_d_eq_2sqrt3_l526_526670


namespace bags_of_white_flour_l526_526625

theorem bags_of_white_flour (total_flour wheat_flour : ℝ) (h1 : total_flour = 0.3) (h2 : wheat_flour = 0.2) : 
  total_flour - wheat_flour = 0.1 :=
by
  sorry

end bags_of_white_flour_l526_526625


namespace compare_constants_l526_526860

noncomputable def a : ℝ := 0.1 * real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - real.log 0.9

theorem compare_constants : c < a ∧ a < b :=
by sorry

end compare_constants_l526_526860


namespace first_student_time_l526_526225

theorem first_student_time
  (n : ℕ)
  (avg_last_three avg_all : ℕ)
  (h_n : n = 4)
  (h_avg_last_three : avg_last_three = 35)
  (h_avg_all : avg_all = 30) :
  let total_time_all := n * avg_all in
  let total_time_last_three := 3 * avg_last_three in
  (total_time_all - total_time_last_three) = 15 :=
by
  let total_time_all := 4 * 30
  let total_time_last_three := 3 * 35
  show total_time_all - total_time_last_three = 15
  sorry

end first_student_time_l526_526225


namespace driving_machine_power_l526_526655

noncomputable def power_of_flywheel_driving_machine 
  (radius : ℝ) (mass : ℝ) (rpm : ℝ) (time_minutes : ℝ) : ℝ :=
  let angular_speed := rpm * (2 * Real.pi) / 60
  let tangential_speed := radius * angular_speed
  let kinetic_energy := (1 / 2) * mass * tangential_speed^2
  let time_seconds := time_minutes * 60
  let power_watts := kinetic_energy / time_seconds
  let horsepower := power_watts / 746
  horsepower

theorem driving_machine_power : 
  power_of_flywheel_driving_machine 3 6000 800 3 ≈ 1431 :=
by
  sorry

end driving_machine_power_l526_526655


namespace unique_triplet_satisfying_conditions_l526_526135

theorem unique_triplet_satisfying_conditions :
  ∃! (a b c: ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧
                 (c ∣ a * b + 1) ∧
                 (b ∣ c * a + 1) ∧
                 (a ∣ b * c + 1) ∧
                 a = 2 ∧ b = 3 ∧ c = 7 :=
by
  sorry

end unique_triplet_satisfying_conditions_l526_526135


namespace magnitude_squared_complex_l526_526288

noncomputable def complex_number := Complex.mk 3 (-4)
noncomputable def squared_complex := complex_number * complex_number

theorem magnitude_squared_complex : Complex.abs squared_complex = 25 :=
by
  sorry

end magnitude_squared_complex_l526_526288


namespace log_sub_l526_526311

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem log_sub : log3 81 - log3 (1/9) = 6 := by
  have h1 : log3 81 = 4 := by
    unfold log3
    have : Real.log 81 = Real.log (3^4) := by
      rw [Real.log_pow]
    rw [Real.log_pow, Real.log three, mul_div_cancel_left]
    exact Real.log_pos (by norm_num : (3:ℝ) > 0)
  have h2 : log3 (1/9) = -2 := by
    unfold log3
    have : Real.log (1/9) = Real.log (3^(-2)) := by
      rw [Real.log_div]
      rw [Real.log_one, Real.log_pow]
    rw [Real.log_div, Real.log_pow, Real.log_three, Real.log_one, mul_div_cancel_left]
    exact Real.log_pos (by norm_num : (3:ℝ) > 0)
  rw [h1, h2]
  norm_num

end log_sub_l526_526311


namespace cryptarithm_proof_l526_526316

def digit : Type := Fin 10 

variables (A R K : digit)

-- conditions: different letters correspond to different digits
axiom h₁ : A ≠ R
axiom h₂ : A ≠ K
axiom h₃ : R ≠ K

-- Solve for specific values
axiom val_A : A = 1
axiom val_R : R = 4
axiom val_K : K = 7

-- the cryptarithm equation ARKA + RKA + KA + A = 2014
theorem cryptarithm_proof 
  (A R K : digit) 
  (h₁ : A ≠ R) 
  (h₂ : A ≠ K) 
  (h₃ : R ≠ K) 
  (val_A : A = 1) 
  (val_R : R = 4) 
  (val_K : K = 7) : 
  (1000 * A + 100 * R + 10 * K + A) + 
  (100 * R + 10 * K + A) + 
  (10 * K + A) + 
  A = 2014 :=
by {
  sorry -- proof not required
}

end cryptarithm_proof_l526_526316


namespace lily_pad_growth_rate_l526_526076

theorem lily_pad_growth_rate 
  (day_37_covers_full : ℕ → ℝ)
  (day_36_covers_half : ℕ → ℝ)
  (exponential_growth : day_37_covers_full = 2 * day_36_covers_half) :
  (2 - 1) / 1 * 100 = 100 :=
by sorry

end lily_pad_growth_rate_l526_526076


namespace pyramid_volume_of_cube_l526_526104

noncomputable def volume_of_pyramid {V : Type*} [normed_add_comm_group V] [normed_space ℝ V] (A B G E : V) :=
  1 / 3 * abs ((1 / 2) * ((B - A) × (G - A)).norm) * (E - A).norm

theorem pyramid_volume_of_cube (A B G E : ℝ^3) :
  ((B - A) = (2, 0, 0)) ∧ ((G - A) = (2, 2, 1)) ∧ ((E - A) = (0, 0, 2)) →
  volume_of_pyramid A B G E = (2 * Real.sqrt 5) / 3 :=
by
  intros
  sorry

end pyramid_volume_of_cube_l526_526104


namespace population_net_change_l526_526510

theorem population_net_change :
  ∀ (P : ℚ), let P₁ := P * (6/5) in let P₂ := P₁ * (6/5) in 
            let P₃ := P₂ * (7/10) in let P₄ := P₃ * (7/10) in 
            ((P₄ - P) / P * 100) = -29 := 
by 
  sorry

end population_net_change_l526_526510


namespace find_extremum_l526_526057

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x

theorem find_extremum (a : ℝ) (h : Deriv (f x a) 1 = 0) : a = Real.exp 1 :=
by sorry

end find_extremum_l526_526057


namespace remainder_of_6_pow_1234_mod_13_l526_526306

theorem remainder_of_6_pow_1234_mod_13 : 6 ^ 1234 % 13 = 10 := 
by 
  sorry

end remainder_of_6_pow_1234_mod_13_l526_526306


namespace first_reoccurrence_line_l526_526562

def gcd (a b : ℕ) := nat.gcd a b
def lcm (a b : ℕ) := a * b / gcd a b

theorem first_reoccurrence_line (lst1 : list char) (lst2 : list ℕ) (h1 : lst1.length = 7) (h2 : lst2.length = 4) (line_length : ℕ) :
  lcm lst1.length lst2.length = 28 :=
by
  sorry

end first_reoccurrence_line_l526_526562


namespace sum_of_possible_ks_l526_526453

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l526_526453


namespace intersection_M_N_l526_526039

def M := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def N := { y : ℝ | y > 0 }

theorem intersection_M_N : (M ∩ N) = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l526_526039


namespace inequality_proof_l526_526399

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) : (1 / x) < (1 / y) :=
by
  sorry

end inequality_proof_l526_526399


namespace shaded_area_is_18pi_l526_526555

structure Heptagon (α : Type*) :=
(side_length : ℝ)
(radius : ℝ)
(vertices : ℕ := 7)
(arcs : ℕ := 7)

def total_shaded_area (H : Heptagon ℝ) : ℝ :=
  let interior_angle_sum := (H.vertices - 2) * 180 in
  let total_angle := H.vertices * 360 in
  let angle_difference := total_angle - interior_angle_sum in
  let equivalent_circles := angle_difference / 360 in
  equivalent_circles * π * H.radius ^ 2

noncomputable def heptagon_example : Heptagon ℝ :=
  { side_length := 4, radius := 2 }

theorem shaded_area_is_18pi :
  total_shaded_area heptagon_example = 18 * π :=
by
  sorry

end shaded_area_is_18pi_l526_526555


namespace approximate_root_bisection_l526_526183

noncomputable theory

open real

def f (x : ℝ) : ℝ := x^2 - 2

theorem approximate_root_bisection :
  ∃ x ∈ Ioo (1:ℝ) 2, |x - 1.4| < 0.1 ∧ f x = 0 :=
sorry

end approximate_root_bisection_l526_526183


namespace sum_of_possible_k_l526_526438

theorem sum_of_possible_k (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  j > 0 ∧ k > 0 → 
  ∑ i in {20, 12, 8, 6, 5}.to_finset, i = 51 :=
by
  sorry

end sum_of_possible_k_l526_526438


namespace books_received_l526_526141

theorem books_received (students : ℕ) (books_per_student : ℕ) (books_fewer : ℕ) (expected_books : ℕ) (received_books : ℕ) :
  students = 20 →
  books_per_student = 15 →
  books_fewer = 6 →
  expected_books = students * books_per_student →
  received_books = expected_books - books_fewer →
  received_books = 294 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end books_received_l526_526141


namespace lava_lamp_arrangement_probability_l526_526121

theorem lava_lamp_arrangement_probability :
  let total_arrangements := (nat.choose 12 4) * (nat.choose 8 4)
    -- Total ways to arrange lamps
  let total_turned_on := nat.choose 12 6
    -- Total ways to choose 6 lamps to be turned on
  let favorable_arrangements := (nat.choose 10 4) * (nat.choose 6 3)
    -- Ways to arrange with leftmost green and off, rightmost blue and on
  let favorable_turned_on := nat.choose 11 5
    -- Ways to turn on 5 more lamps given the rightmost blue lamp is already on
  let favorable_outcomes := favorable_arrangements * favorable_turned_on
  let total_outcomes := total_arrangements * total_turned_on
  in (favorable_outcomes / total_outcomes:ℚ) = 80 / 1313 :=
by sorry

end lava_lamp_arrangement_probability_l526_526121


namespace money_left_after_distributions_and_donations_l526_526662

theorem money_left_after_distributions_and_donations 
  (total_income : ℕ)
  (percent_to_children : ℕ)
  (percent_to_each_child : ℕ)
  (number_of_children : ℕ)
  (percent_to_wife : ℕ)
  (percent_to_orphan_house : ℕ)
  (remaining_income_percentage : ℕ)
  (children_distribution : ℕ → ℕ → ℕ)
  (wife_distribution : ℕ → ℕ)
  (calculate_remaining : ℕ → ℕ → ℕ)
  (calculate_donation : ℕ → ℕ → ℕ)
  (calculate_money_left : ℕ → ℕ → ℕ)
  (income : ℕ := 400000)
  (result : ℕ := 57000) :
  children_distribution percent_to_each_child number_of_children = 60 →
  percent_to_wife = 25 →
  remaining_income_percentage = 15 →
  percent_to_orphan_house = 5 →
  wife_distribution percent_to_wife = 100000 →
  calculate_remaining 100 85 = 15 →
  calculate_donation percent_to_orphan_house (calculate_remaining 100 85 * total_income) = 3000 →
  calculate_money_left (calculate_remaining 100 85 * total_income) 3000 = result →
  total_income = income →
  income - (60 * income / 100 + 25 * income / 100 + 5 * (15 * income / 100) / 100) = result
  :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end money_left_after_distributions_and_donations_l526_526662


namespace fraction_difference_is_correct_l526_526726

-- Convert repeating decimal to fraction
def repeating_to_fraction : ℚ := 0.72 + 72 / 9900

-- Convert 0.72 to fraction
def decimal_to_fraction : ℚ := 72 / 100

-- Define the difference between the fractions
def fraction_difference := repeating_to_fraction - decimal_to_fraction

-- Final statement to prove
theorem fraction_difference_is_correct : fraction_difference = 2 / 275 := by
  sorry

end fraction_difference_is_correct_l526_526726


namespace fraction_difference_l526_526687

-- Definitions for the problem conditions
def repeatingDecimal72 := 8 / 11
def decimal72 := 18 / 25

-- Statement that needs to be proven
theorem fraction_difference : repeatingDecimal72 - decimal72 = 2 / 275 := 
by 
  sorry

end fraction_difference_l526_526687


namespace desired_overall_percentage_l526_526666

-- Define the scores in the three subjects
def score1 := 50
def score2 := 70
def score3 := 90

-- Define the expected overall percentage
def expected_overall_percentage := 70

-- The main theorem to prove
theorem desired_overall_percentage :
  (score1 + score2 + score3) / 3 = expected_overall_percentage :=
by
  sorry

end desired_overall_percentage_l526_526666


namespace local_maximum_at_neg2_l526_526139

noncomputable def y (x : ℝ) : ℝ :=
  (1/3) * x^3 - 4 * x + 4

theorem local_maximum_at_neg2 :
  ∃ x : ℝ, x = -2 ∧ 
           y x = 28/3 ∧
           (∀ ε > 0, ∃ δ > 0, ∀ z, abs (z + 2) < δ → y z < y (-2)) := by
  sorry

end local_maximum_at_neg2_l526_526139


namespace min_tan_BAD_in_right_triangle_l526_526473

def triangle_with_angle_and_side (A B C D : Type*) [triangle A B C]
  (h_angle_C : angle A C B = 90) (h_BC : segment B C = 6)
  (h_ratio_BD_DC : ∃ (r : ℝ), ratio (segment B D) (segment D C) = r ∧ r = 2/1) : Prop :=
  ∃ (min_tan_BAD : ℝ), min_tan_BAD = 1

theorem min_tan_BAD_in_right_triangle (A B C D : Type*) [triangle A B C]
  (h_angle_C : angle A C B = 90) (h_BC : segment B C = 6)
  (h_ratio_BD_DC : ∃ (r : ℝ), ratio (segment B D) (segment D C) = r ∧ r = 2/1) :
  ∃ (min_tan_BAD : ℝ), min_tan_BAD = 1 :=
sorry

end min_tan_BAD_in_right_triangle_l526_526473


namespace game_points_l526_526266

def f : ℕ → ℕ := λ n,
  if n % 3 = 0 then 9
  else if n % 2 = 0 then 3
  else 1

def Allie_points : ℕ := f 6 + f 3 + f 4
def Betty_points : ℕ := f 1 + f 2 + f 5 + f 6
def product_points : ℕ := Allie_points * Betty_points

theorem game_points :
  product_points = 294 :=
by
  -- Proof omitted
  sorry

end game_points_l526_526266


namespace polygon_diagonals_l526_526337

-- Lean statement of the problem

theorem polygon_diagonals (n : ℕ) (h : n - 3 = 2018) : n = 2021 :=
  by sorry

end polygon_diagonals_l526_526337


namespace func_symmetry_monotonicity_range_of_m_l526_526558

open Real

theorem func_symmetry_monotonicity (f : ℝ → ℝ)
  (h1 : ∀ x, f (3 + x) = f (1 - x))
  (h2 : ∀ x1 x2, 2 < x1 → 2 < x2 → (f x1 - f x2) / (x1 - x2) > 0) :
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∀ x1 x2, (x1 > 2 ∧ x2 > 2 → f x1 < f x2 → x1 < x2) ∧
            (x2 > 2 ∧ x1 > x2 → f x2 < f x1 → x2 < x1)) := 
sorry

theorem range_of_m (f : ℝ → ℝ)
  (h : ∀ θ : ℝ, f (cos θ ^ 2 + 2 * (m : ℝ) ^ 2 + 2) < f (sin θ + m ^ 2 - 3 * m - 2)) :
  ∀ m, (3 - sqrt 42) / 6 < m ∧ m < (3 + sqrt 42) / 6 :=
sorry

end func_symmetry_monotonicity_range_of_m_l526_526558


namespace sum_of_all_possible_k_values_l526_526446

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l526_526446


namespace repeating_decimal_difference_l526_526710

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l526_526710


namespace functional_eq_f_l526_526993

noncomputable def f : ℝ → ℝ := λ x, (10 * x + 21 + Real.sqrt 41) / (1 + Real.sqrt 41)

theorem functional_eq_f : (∀ x y : ℝ, (f x * f y - f (x * y)) / 5 = x + y + 2) :=
by
  sorry

end functional_eq_f_l526_526993


namespace student_vision_data_correct_l526_526548

noncomputable def student_vision_decimal_approx : ℝ :=
  let L := 4.9
  let V := 10^(-(L - 5))
  approx := 0.8
  sqrt10 := 10^(1/10)
  1 / sqrt10

theorem student_vision_data_correct :
  (L = 4.9) →
  (L = 5 + log10 V) →
  (sqrt10 ≈ 1.259) →
  (V ≈ 0.8) :=
  by
    sorry

end student_vision_data_correct_l526_526548


namespace total_stamps_l526_526520

def c : ℕ := 578833
def bw : ℕ := 523776
def total : ℕ := 1102609

theorem total_stamps : c + bw = total := 
by 
  sorry

end total_stamps_l526_526520


namespace complex_subtraction_l526_526021

open Complex

def z1 : ℂ := 3 + 4 * I
def z2 : ℂ := 1 + I

theorem complex_subtraction : z1 - z2 = 2 + 3 * I := by
  sorry

end complex_subtraction_l526_526021


namespace ABC_value_l526_526138

noncomputable def A := 11
noncomputable def B := 5
noncomputable def C := 5

theorem ABC_value :
  let expr := (2 + Real.sqrt 5) / (3 - Real.sqrt 5),
      A := ((11:ℝ) / 4),
      B := ((5:ℝ) / 4),
      C := 5 in
  (expr * (3 + Real.sqrt 5) / (3 + Real.sqrt 5) = (11 + 5 * Real.sqrt 5) / 4) ∧
  (C = 5) ∧
  (A * 4 = 11) ∧
  (B * 4 = 5) ∧
  (ABC = A * B * C) :=
by
  sorry

end ABC_value_l526_526138


namespace vision_data_approximation_l526_526540

theorem vision_data_approximation :
  ∀ (L V : ℝ), L = 4.9 → (L = 5 + log10 V) → (0.8 ≤ V ∧ V ≤ 0.8 * 1.0001) := 
by
  intros L V hL hRel
  have hlog : log10 V = -0.1 := 
    by rw [←hRel, hL]; ring
  have hV : V = 10 ^ (-0.1) :=
    by rw [←hlog]; exact (real.rpow10_log10 V)
  rw hV
  have hApprox : 10 ^ (-0.1) ≈ 0.8 := sorry -- This is the approximation step
  exact hApprox

end vision_data_approximation_l526_526540


namespace total_toothpicks_correct_l526_526595

def number_of_horizontal_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
(height + 1) * width

def number_of_vertical_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
(height) * (width + 1)

def total_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
number_of_horizontal_toothpicks height width + number_of_vertical_toothpicks height width

theorem total_toothpicks_correct:
  total_toothpicks 30 15 = 945 :=
by
  sorry

end total_toothpicks_correct_l526_526595


namespace unique_digit_B_l526_526956

open Finset

theorem unique_digit_B : 
  ∀ (A B C D E F : ℕ), 
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ 
    (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ 
    (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ 
    (D ≠ E) ∧ (D ≠ F) ∧ 
    (E ≠ F) ∧
    (A ∈ range 2 6) ∧ (B ∈ range 2 6) ∧ (C ∈ range 2 6) ∧
    (D ∈ range 2 6) ∧ (E ∈ range 2 6) ∧ (F ∈ range 2 6) ∧
    (A + B + C) + (A + B + E + F) + (C + D + E) + (B + D + F) + (C + F) = 65 
    → B = 4 :=
by sorry

end unique_digit_B_l526_526956


namespace ratio_of_dividends_l526_526235

-- Definitions based on conditions
def expected_earnings : ℝ := 0.80
def actual_earnings : ℝ := 1.10
def additional_per_increment : ℝ := 0.04
def increment_size : ℝ := 0.10

-- Definition for the base dividend D which remains undetermined
variable (D : ℝ)

-- Stating the theorem
theorem ratio_of_dividends 
  (h1 : actual_earnings = 1.10)
  (h2 : expected_earnings = 0.80)
  (h3 : additional_per_increment = 0.04)
  (h4 : increment_size = 0.10) :
  let additional_earnings := actual_earnings - expected_earnings
  let increments := additional_earnings / increment_size
  let additional_dividend := increments * additional_per_increment
  let total_dividend := D + additional_dividend
  let ratio := total_dividend / actual_earnings
  ratio = (D + 0.12) / 1.10 :=
by
  sorry

end ratio_of_dividends_l526_526235


namespace compare_a_b_c_l526_526844

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l526_526844


namespace triangle_property_l526_526367

theorem triangle_property
  (A B C : ℝ)
  (a b c : ℝ)
  (R : ℝ)
  (hR : R = Real.sqrt 3)
  (h1 : a * Real.sin C + Real.sqrt 3 * c * Real.cos A = 0)
  (h2 : b + c = Real.sqrt 11)
  (htri : a / Real.sin A = 2 * R ∧ b / Real.sin B = 2 * R ∧ c / Real.sin C = 2 * R):
  a = 3 ∧ (1 / 2 * b * c * Real.sin A = Real.sqrt 3 / 2) := 
sorry

end triangle_property_l526_526367


namespace december_sales_fraction_l526_526214

-- Definitions from the problem statement
variable (A : ℝ) -- average monthly sales total for January through November
variable (total_sales_jan_nov : ℝ) := 11 * A -- total sales for January through November
variable (december_sales : ℝ) := 7 * A -- sales total for December
variable (total_sales_year : ℝ) := total_sales_jan_nov + december_sales -- total sales for the year

-- The proof statement
theorem december_sales_fraction :
    (december_sales / total_sales_year) = 7 / 18 :=
by
  sorry

end december_sales_fraction_l526_526214


namespace smallest_number_of_eggs_l526_526207

-- Definitions based on the given problem conditions
def num_containers (c : ℕ) : Prop := c > ceil(106 / 12)

def total_eggs (c : ℕ) : ℕ := 12 * c - 6

-- Statement of the math proof problem
theorem smallest_number_of_eggs (c : ℕ) (hc : num_containers c) : total_eggs c = 102 :=
by
  sorry

end smallest_number_of_eggs_l526_526207


namespace even_x_pairs_l526_526323

theorem even_x_pairs (x y : ℕ) : 
  (∃ n ∈ Nat.even, 4 * x + 7 * y = 600 ∧ 0 < x ∧ 0 < y) 
  → (∃ k : ℕ, k ≤ 10 ∧ x = 14 * k + 6 ∧ y = 82 - 8 * k) :=
sorry

end even_x_pairs_l526_526323


namespace sum_possible_values_k_l526_526468

open Nat

theorem sum_possible_values_k (j k : ℕ) (h : (1 / j : ℚ) + 1 / k = 1 / 4) : 
  ∃ ks : List ℕ, (∀ k' ∈ ks, ∃ j', (1 / j' : ℚ) + 1 / k' = 1 / 4) ∧ ks.sum = 51 :=
by
  sorry

end sum_possible_values_k_l526_526468


namespace sin_maximum_value_l526_526282

theorem sin_maximum_value (c : ℝ) :
  (∀ x : ℝ, x = -π/4 → 3 * Real.sin (2 * x + c) = 3) → c = π :=
by
 sorry

end sin_maximum_value_l526_526282


namespace true_propositions_about_z_l526_526920

def z : ℂ := (2 * complex.I) / (-1 - complex.I)

theorem true_propositions_about_z :
  (z^2 = 2 * complex.I) ∧ (z.im = -1) :=
by
  sorry

end true_propositions_about_z_l526_526920


namespace algebraic_fraction_l526_526620

theorem algebraic_fraction (x : ℝ) (h1 : 1 / 3 = 1 / 3) 
(h2 : x / Real.pi = x / Real.pi) 
(h3 : 2 / (x + 3) = 2 / (x + 3))
(h4 : (x + 2) / 3 = (x + 2) / 3) 
: 
2 / (x + 3) = 2 / (x + 3) := sorry

end algebraic_fraction_l526_526620


namespace find_n_square_expr_is_square_l526_526988

/-- The statement of the mathematically equivalent proof problem -/
theorem find_n_square_expr_is_square (n : ℕ) (h_pos : n > 0)
  (d : ℕ → ℕ)
  (h_d : d n = (∑ i in finset.filter (λ (x : ℕ), x ∣ n) (finset.range (n + 1)), 1)) :
  n = 4 ∨ n = 100 ↔ (∃ k : ℕ, k^2 = n) := sorry

end find_n_square_expr_is_square_l526_526988


namespace total_eyes_l526_526044

def boys := 23
def girls := 18
def cats := 10
def spiders := 5

def boy_eyes := 2
def girl_eyes := 2
def cat_eyes := 2
def spider_eyes := 8

theorem total_eyes : (boys * boy_eyes) + (girls * girl_eyes) + (cats * cat_eyes) + (spiders * spider_eyes) = 142 := by
  sorry

end total_eyes_l526_526044


namespace ellipse_properties_l526_526877

-- Definitions for the ellipse and related conditions
def ellipse_centered_origin_major_axis_x (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

def vertex_one_ (b : ℝ) : Prop :=
  ∃ (x : ℝ), (x = 0 ∧ y = sqrt 5)

def eccentricity_ (a c : ℝ) : Prop :=
  ∃ (c : ℝ), (e = sqrt 6 / 6 = c / a)

def foci_ (F1 F2 : ℝ × ℝ) : Prop := sorry

def point_on_ellipse (M : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_centered_origin_major_axis_x 6 5 x y

-- The theorem to prove
theorem ellipse_properties :
  ∃ (a b : ℝ), ellipse_centered_origin_major_axis_x a b ∧ vertex_one_ b ∧
  eccentricity_ a (sqrt 6 / 6) ∧ 
  (∀ (F1 F2 : ℝ × ℝ), foci_ F1 F2 → vertex_one_ b ∧ eccentricity_ a (sqrt 6 / 6) →
    ∀ M : ℝ × ℝ, point_on_ellipse M →
    max_area_triangle_m_F1_F2 = sqrt 5 ∧
    ¬ exists (P : ℝ × ℝ), point_on_ellipse P ∧
    dot_product (P - F1) (P - F2) = 0)
:= sorry

end ellipse_properties_l526_526877


namespace nonnegative_expression_l526_526335

noncomputable def num (x : ℝ) : ℝ := x - 15 * x^2 + 50 * x^3 - 10 * x^4
noncomputable def denom (x : ℝ) : ℝ := 8 - 3 * x^3

theorem nonnegative_expression (x : ℝ) : 0 ≤ num x / denom x ↔ 0 ≤ x ∧ x < 2 * real.cbrt 3 :=
by
  sorry

end nonnegative_expression_l526_526335


namespace sin_theta_eq_neg_one_ninth_l526_526802

theorem sin_theta_eq_neg_one_ninth 
(θ : ℝ)
(h : Real.cos (Real.pi / 4 - θ / 2) = 2 / 3) :
  Real.sin θ = -1 / 9 := 
by
  sorry

end sin_theta_eq_neg_one_ninth_l526_526802


namespace sin_theta_between_vectors_find_m_perpendicular_l526_526917

variables (a b : ℝ × ℝ)
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

def sin_theta (a b : ℝ × ℝ) : ℝ :=
  let cos_theta := dot_product a b / (magnitude a * magnitude b)
  Real.sqrt (1 - cos_theta ^ 2)

def perp (u v : ℝ × ℝ) : Prop := dot_product u v = 0

theorem sin_theta_between_vectors : 
  let a := (3 : ℝ, -4 : ℝ)
  let b := (1 : ℝ, 2 : ℝ)
  sin_theta a b = 2 * Real.sqrt 5 / 5 := sorry

theorem find_m_perpendicular :
  let a := (3 : ℝ, -4 : ℝ)
  let b := (1 : ℝ, 2 : ℝ)
  ∃ m : ℝ, perp (m • a - b) (a + b) ∧ m = 0 := sorry

end sin_theta_between_vectors_find_m_perpendicular_l526_526917


namespace no_solution_A_eq_B_l526_526382

theorem no_solution_A_eq_B (a : ℝ) (h1 : a = 2 * a) (h2 : a ≠ 2) : false := by
  sorry

end no_solution_A_eq_B_l526_526382


namespace cannot_fit_480_pictures_l526_526535

theorem cannot_fit_480_pictures 
  (A_capacity : ℕ) (B_capacity : ℕ) (C_capacity : ℕ) 
  (n_A : ℕ) (n_B : ℕ) (n_C : ℕ) 
  (total_pictures : ℕ) : 
  A_capacity = 12 → B_capacity = 18 → C_capacity = 24 → 
  n_A = 6 → n_B = 4 → n_C = 3 → 
  total_pictures = 480 → 
  A_capacity * n_A + B_capacity * n_B + C_capacity * n_C < total_pictures :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end cannot_fit_480_pictures_l526_526535


namespace excess_common_fraction_l526_526715

theorem excess_common_fraction :
  let x := (72 / 10^2 + 7.2 * (10 / 10^2) * (10^{-2} / (1 - 10^{-2}))) in
  x - (72 / 100) = 8 / 1100 :=
by
  sorry

end excess_common_fraction_l526_526715


namespace find_magnitude_and_side_find_side_c_l526_526028

def f (x : ℝ) : ℝ := (√3) * sin x * cos x + (cos x)^2 - (1/2)

theorem find_magnitude_and_side (a b B c : ℝ) (k : ℤ) 
  (ha : a = √3) (hb : b = 1) 
  (hB : f B = 1) :
  B = π / 6 ∨ B = π / 6 + k * π :=
  sorry

theorem find_side_c (a b B c : ℝ) 
  (ha : a = √3) (hb : b = 1) 
  (hB : B = π / 6) :
  c = 1 ∨ c = 2 :=
  sorry

end find_magnitude_and_side_find_side_c_l526_526028


namespace non_congruent_squares_in_6x6_grid_l526_526393

theorem non_congruent_squares_in_6x6_grid : 
  let grid_size := 6
  let regular_squares := (let f := λ n, (grid_size - n) ^ 2 in f 1 + f 2 + f 3 + f 4 + f 5),
  let diagonal_squares := (let f := λ n, (grid_size - (n + 1)) ^ 2 in f 1 + f 2 + f 3)
  in regular_squares + diagonal_squares = 105
:= by
  let grid_size := 6
  let f n := (grid_size - n) ^ 2
  have regular_squares : ∀ n, f 1 + f 2 + f 3 + f 4 + f 5 = 25 + 16 + 9 + 4 + 1 := sorry
  let diagonal_f n := (grid_size - (n + 1)) ^ 2
  have diagonal_squares : ∀ n, diagonal_f 1 + diagonal_f 2 + diagonal_f 3 = 25 + 16 + 9 := sorry
  have total_squares : regular_squares + diagonal_squares = 25 + 16 + 9 + 4 + 1 + 25 + 16 + 9 := sorry
  exact (regular_squares + diagonal_squares = 105) sorry

end non_congruent_squares_in_6x6_grid_l526_526393


namespace gary_asparagus_l526_526338

/-- Formalization of the problem -/
theorem gary_asparagus (A : ℝ) (ha : 700 * 0.50 = 350) (hg : 40 * 2.50 = 100) (hw : 630 = 3 * A + 350 + 100) : A = 60 :=
by
  sorry

end gary_asparagus_l526_526338


namespace compare_abc_l526_526821
open Real

noncomputable def a : ℝ := 0.1 * exp(0.1)
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - log 0.9

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l526_526821


namespace trapezium_area_proof_l526_526781

-- Define the lengths of the parallel sides and the distance between them
def a : ℝ := 20
def b : ℝ := 18
def h : ℝ := 15

-- Define the area of the trapezium
def area_of_trapezium (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

-- State the theorem to be proved
theorem trapezium_area_proof : area_of_trapezium a b h = 285 := by
  sorry

end trapezium_area_proof_l526_526781


namespace triangle_area_is_4_l526_526426

variable {PQ RS : ℝ} -- lengths of PQ and RS respectively
variable {area_PQRS area_PQS : ℝ} -- areas of the trapezoid and triangle respectively

-- Given conditions
@[simp]
def trapezoid_area_is_12 (area_PQRS : ℝ) : Prop :=
  area_PQRS = 12

@[simp]
def RS_is_twice_PQ (PQ RS : ℝ) : Prop :=
  RS = 2 * PQ

-- To prove: the area of triangle PQS is 4 given the conditions
theorem triangle_area_is_4 (h1 : trapezoid_area_is_12 area_PQRS)
                          (h2 : RS_is_twice_PQ PQ RS)
                          (h3 : area_PQRS = 3 * area_PQS) : area_PQS = 4 :=
by
  sorry

end triangle_area_is_4_l526_526426


namespace triangle_side_c_triangle_sum_ab_l526_526068

theorem triangle_side_c (a b : ℝ) (A B C : ℝ) (h : a ≠ b) (h1 : 2 * sin (A - B) = a * sin A - b * sin B) :
  c = 2 := sorry

theorem triangle_sum_ab (a b c : ℝ) (A B C : ℝ) (h_area : 0.5 * a * b * sin C = 1) (h_tanc : tan C = 2) :
  a + b = 1 + sqrt 5 := sorry

end triangle_side_c_triangle_sum_ab_l526_526068


namespace airplane_speeds_l526_526587

theorem airplane_speeds (v : ℝ) 
  (h1 : 2.5 * v + 2.5 * 250 = 1625) : 
  v = 400 := 
sorry

end airplane_speeds_l526_526587


namespace rate_of_interest_approx_23_22_percent_l526_526406

theorem rate_of_interest_approx_23_22_percent (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) (CI : ℝ) :
  CI = P * ((1 + r / (100 * n)) ^ (n * t) - 1) →
  CI = P / 3 →
  n = 2 →
  t = 5 →
  r ≈ 23.22 :=
  by sorry

end rate_of_interest_approx_23_22_percent_l526_526406


namespace smallest_possible_value_of_y_l526_526418

noncomputable def smallest_prime_y (x y : ℕ) : Prop :=
  prime x ∧ prime y ∧ 2 * x + y = 180

theorem smallest_possible_value_of_y :
  ∃ y : ℕ, (∃ x : ℕ, smallest_prime_y x y) ∧ y = 101 := 
by
  sorry

end smallest_possible_value_of_y_l526_526418


namespace solve_x_log_x_eq_x_cubed_over_100_l526_526571

theorem solve_x_log_x_eq_x_cubed_over_100 :
  {x : ℝ | x > 0 ∧ x ^ real.log10 x = x^3 / 100} = {10, 100} :=
sorry

end solve_x_log_x_eq_x_cubed_over_100_l526_526571


namespace minimum_distance_between_squirrels_l526_526590

-- Define given conditions
variables {A B O : Point}
noncomputable def AO : ℝ := 120
noncomputable def BO : ℝ := 80
noncomputable def angle_AOB : ℝ := 60

-- Define the proof problem
theorem minimum_distance_between_squirrels :
  ∀ (d : ℝ),
  ∃ m : ℝ, m = 20 * Real.sqrt 3 ∧
  (distance (squirrel_position A O d) (squirrel_position B O d) ≥ m) :=
sorry

end minimum_distance_between_squirrels_l526_526590


namespace rhombus_side_length_l526_526300

theorem rhombus_side_length (d : ℝ) (K : ℝ) (h1 : 3 * d) (h2 : K = (1/2) * d * 3 * d) : 
  let s := sqrt (5 * K / 3) in s^2 = 5 * K / 3 :=
sorry

end rhombus_side_length_l526_526300


namespace Milton_loan_difference_l526_526123

-- Define the conditions for the problem
def principal : ℝ := 8000
def annual_interest_rate : ℝ := 0.15
def duration : ℕ := 3

-- Define the future value calculation with semi-annual compounding
def future_value_semi_annual (P: ℝ) (r: ℝ) (t: ℕ) : ℝ :=
  P * (1 + r / 2) ^ (2 * t)

-- Define the future value calculation with annual compounding
def future_value_annual (P: ℝ) (r: ℝ) (t: ℕ) : ℝ :=
  P * (1 + r) ^ t

-- Define the difference calculation
def difference_in_future_value (P: ℝ) (r: ℝ) (t: ℕ) : ℝ :=
  future_value_annual P r t - future_value_semi_annual P r t

-- State the theorem to prove the difference is 81.92
theorem Milton_loan_difference (P: ℝ := principal) (r: ℝ := annual_interest_rate) (t: ℕ := duration) :
  difference_in_future_value P r t = 81.92 :=
by
  sorry

end Milton_loan_difference_l526_526123


namespace algebraic_fraction_l526_526621

theorem algebraic_fraction (x : ℝ) (h1 : 1 / 3 = 1 / 3) 
(h2 : x / Real.pi = x / Real.pi) 
(h3 : 2 / (x + 3) = 2 / (x + 3))
(h4 : (x + 2) / 3 = (x + 2) / 3) 
: 
2 / (x + 3) = 2 / (x + 3) := sorry

end algebraic_fraction_l526_526621


namespace john_made_additional_shots_l526_526477

variable (initial_shots additional_shots : ℕ)
variable (initial_percentage additional_percentage : ℝ)

-- Conditions translated to Lean definitions
def initial_made_shots (initial_shots : ℕ) (initial_percentage : ℝ) : ℕ :=
  (initial_percentage * initial_shots).toNat

def total_made_shots (initial_shots additional_shots : ℕ) (additional_percentage : ℝ) : ℕ :=
  (additional_percentage * (initial_shots + additional_shots)).toNat

-- The main theorem
theorem john_made_additional_shots :
  initial_shots = 30 →
  additional_shots = 10 →
  initial_percentage = 0.40 →
  additional_percentage = 0.44 →
  (total_made_shots 30 10 0.44) - (initial_made_shots 30 0.40) = 6 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end john_made_additional_shots_l526_526477


namespace prove_expression_l526_526746

def given_expression : ℤ := -4 + 6 / (-2)

theorem prove_expression : given_expression = -7 := 
by 
  -- insert proof here
  sorry

end prove_expression_l526_526746


namespace repeating_seventy_two_exceeds_seventy_two_l526_526738

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l526_526738


namespace sum_of_possible_ks_l526_526454

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l526_526454


namespace student_vision_data_correct_l526_526549

noncomputable def student_vision_decimal_approx : ℝ :=
  let L := 4.9
  let V := 10^(-(L - 5))
  approx := 0.8
  sqrt10 := 10^(1/10)
  1 / sqrt10

theorem student_vision_data_correct :
  (L = 4.9) →
  (L = 5 + log10 V) →
  (sqrt10 ≈ 1.259) →
  (V ≈ 0.8) :=
  by
    sorry

end student_vision_data_correct_l526_526549


namespace problem1_problem2_l526_526353

variable (a b c : ℝ) (F1 F2 A B G M N P Q : ℝ × ℝ) 
variable (S S1 S2 λ : ℝ)

def equation_ellipse : Prop := 
  ∃ a b c : ℝ, 0 < b ∧ b < a ∧ 2 * c = 2 * sqrt 3 ∧ a = 2 ∧ b = 1 ∧ 
  ∀ x y : ℝ, (x, y) ∈ { (x, y) | (x^2 / a^2 + y^2 / b^2 = 1) } ↔ 
  (x^2 / 4 + y^2 = 1)

def lambda_exists : Prop := 
  (∃ G : ℝ × ℝ, G = (1, 0)) ∧ 
  (∃ l : ℝ → ℝ, l = (λ _: ℝ, 4)) ∧ 
  (∀ λ x : ℝ, λ = 2 ∧ λ * sqrt (S1 * S2) - S = 0)

theorem problem1 : equation_ellipse := 
  sorry

theorem problem2 : lambda_exists := 
  sorry

end problem1_problem2_l526_526353


namespace sin_theta_eq_2_sqrt_5_div_5_find_m_l526_526918

-- Part 1: Prove sin θ = 2√5 / 5
def vector_a : ℝ × ℝ := (3, -4)
def vector_b : ℝ × ℝ := (1, 2)

theorem sin_theta_eq_2_sqrt_5_div_5 (θ : ℝ) :
  sin θ = (2 * Real.sqrt 5) / 5 ↔
  let a := vector_a
  let b := vector_b
  θ = Real.arcsin ((-b.1 * a.2 + a.1 * b.2) / (Real.sqrt ((a.1 ^ 2 + a.2 ^ 2) * (b.1 ^ 2 + b.2 ^ 2))))
:= sorry

-- Part 2: Prove m = 0
theorem find_m (m : ℝ) :
  (let a := vector_a in
   let b := vector_b in
   (fun m : ℝ => (m * a.1 - b.1) * (a.1 + b.1) + (m * a.2 - b.2) * (a.2 + b.2)) m = 0) ↔ m = 0
:= sorry

end sin_theta_eq_2_sqrt_5_div_5_find_m_l526_526918


namespace relationship_among_abc_l526_526018

noncomputable def f (x : ℝ) : ℝ := 2 ^ (|x|) - 1

def a : ℝ := f (Real.log 3 / Real.log 0.5)
def b : ℝ := f (Real.log 5 / Real.log 2)
def c : ℝ := f 0

theorem relationship_among_abc : b > a ∧ a > c := by 
  sorry

end relationship_among_abc_l526_526018


namespace probability_13_letters_5_abroad_l526_526760

noncomputable def poisson_pdf (μ k : ℕ) : ℝ :=
  (μ ^ k) / (Nat.factorial k) * Real.exp (-μ)

theorem probability_13_letters_5_abroad :
  let λ1 : ℕ := 4; 
  let λ2 : ℕ := 2;
  let μ1 : ℕ := λ1 * 2;
  let μ2 : ℕ := λ2 * 2;
  poisson_pdf μ1 8 * poisson_pdf μ2 5 ≈ 0.0218 :=
by
  sorry

end probability_13_letters_5_abroad_l526_526760


namespace prob_twins_street_l526_526216

variable (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

theorem prob_twins_street : p ≠ 1 → real := sorry

end prob_twins_street_l526_526216


namespace cost_of_outfit_l526_526675

theorem cost_of_outfit (P T J : ℝ) 
  (h1 : 4 * P + 8 * T + 2 * J = 2400)
  (h2 : 2 * P + 14 * T + 3 * J = 2400)
  (h3 : 3 * P + 6 * T = 1500) :
  P + 4 * T + J = 860 := 
sorry

end cost_of_outfit_l526_526675


namespace minimal_n_is_40_l526_526380

def sequence_minimal_n (p : ℝ) (a : ℕ → ℝ) : Prop :=
  a 1 = p ∧
  a 2 = p + 1 ∧
  (∀ n, n ≥ 1 → a (n + 2) - 2 * a (n + 1) + a n = n - 20) ∧
  (∀ n, a n ≥ p) -- Since minimal \(a_n\) implies non-negative with given \(a_1, a_2\)

theorem minimal_n_is_40 (p : ℝ) (a : ℕ → ℝ) (h : sequence_minimal_n p a) : ∃ n, n = 40 ∧ (∀ m, n ≠ m → a n ≤ a m) :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end minimal_n_is_40_l526_526380


namespace sphere_surface_area_l526_526171

-- Define the radius
def radius : ℝ := 4

-- Define the surface area formula
def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Theorem statement
theorem sphere_surface_area : surface_area radius = 64 * Real.pi :=
by 
  sorry

end sphere_surface_area_l526_526171


namespace first_student_time_l526_526224

-- Define the conditions
def num_students := 4
def avg_last_three := 35
def avg_all := 30
def total_time_all := num_students * avg_all
def total_time_last_three := (num_students - 1) * avg_last_three

-- State the theorem
theorem first_student_time : (total_time_all - total_time_last_three) = 15 :=
by
  -- Proof is skipped
  sorry

end first_student_time_l526_526224


namespace compare_values_l526_526807

variable (a b c : ℝ)

noncomputable def a_def : ℝ := 0.1 * Real.exp 0.1
noncomputable def b_def : ℝ := 1 / 9
noncomputable def c_def : ℝ := - Real.log 0.9

theorem compare_values : 
  c < a ∧ a < b := 
begin
  let a := 0.1 * Real.exp 0.1,
  let b := 1 / 9,
  let c := - Real.log 0.9,
  have h1 : c < a, sorry,
  have h2 : a < b, sorry,
  exact ⟨h1, h2⟩
end

end compare_values_l526_526807


namespace pq_iff_l526_526108

variables {a b : ℝ}

def non_collinear (a b : ℝ) : Prop := ¬ (a = 0 ∨ b = 0)

def proposition_p (a b : ℝ) : Prop := a * b > 0

def proposition_q (a b : ℝ) : Prop := ∃ θ : ℝ, θ > 0 ∧ θ < π / 2 ∧ cos θ = a * b / (abs a * abs b)

theorem pq_iff (a b : ℝ) (h_non_collinear : non_collinear a b) :
  proposition_p a b ↔ proposition_q a b :=
sorry

end pq_iff_l526_526108


namespace cone_slant_height_l526_526572

theorem cone_slant_height {CSA r : ℝ} (h_CSA : CSA = 367.5663404700058) (h_r : r = 9) :
  ∃ l : ℝ, l = 13 :=
by
  -- Grabbing the constant value of pi to use in the CSA formula
  let pi := Real.pi
  -- Calculate slant height using the CSA formula: CSA = π * r * l
  have h_l : l = CSA / (pi * r), by sorry
  -- Substitute the given values and calculate
  have h_values : l = 367.5663404700058 / (pi * 9), by boring_runs
  -- Check if the calculated value is approximately 13
  use 13
  linarith

end cone_slant_height_l526_526572


namespace first_player_winning_strategy_l526_526957

theorem first_player_winning_strategy :
  ∃ (move_count : ℕ) (compartment_from : ℕ),
    initial_conf = [0, 2, 0, 0] →
    winning_strategy first_player initial_conf final_conf →
    move_count = 2 ∧ compartment_from = 1 :=
begin
  let initial_conf := [0, 2, 0, 0],
  let final_conf := [2, 0, 0, 0],
  let first_player := "A",
  sorry,
end

end first_player_winning_strategy_l526_526957


namespace find_bird_values_l526_526094

noncomputable def bird_problem (x y z : ℕ) : Prop :=
  29 + x - y + z / 3 = 42.7

theorem find_bird_values (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : bird_problem x y z :=
  sorry

end find_bird_values_l526_526094


namespace neon_signs_blink_together_l526_526581

-- Define the time intervals for the blinks
def blink_interval1 : ℕ := 7
def blink_interval2 : ℕ := 11
def blink_interval3 : ℕ := 13

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- State the theorem
theorem neon_signs_blink_together : Nat.lcm (Nat.lcm blink_interval1 blink_interval2) blink_interval3 = 1001 := by
  sorry

end neon_signs_blink_together_l526_526581


namespace contradiction_even_odd_l526_526584

theorem contradiction_even_odd (a b c : ℕ) (h1 : (a % 2 = 1 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ c % 2 = 1) ∨ (b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1)) :
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) :=
by
  -- proof by contradiction
  sorry

end contradiction_even_odd_l526_526584


namespace central_cell_value_l526_526336

theorem central_cell_value (a1 a2 a3 a4 a5 a6 a7 a8 C : ℕ) 
  (h1 : a1 + a3 + C = 13) (h2 : a2 + a4 + C = 13)
  (h3 : a5 + a7 + C = 13) (h4 : a6 + a8 + C = 13)
  (h5 : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 40) : 
  C = 3 := 
sorry

end central_cell_value_l526_526336


namespace probability_heads_exactly_10_out_of_12_flips_l526_526602

theorem probability_heads_exactly_10_out_of_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := Nat.choose 12 10
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = (66 : ℝ) / 4096 :=
by
  let total_outcomes := 2^12
  let favorable_outcomes := Nat.choose 12 10
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  have total_outcomes_val : total_outcomes = 4096 := by norm_num
  have favorable_outcomes_val : favorable_outcomes = 66 := by norm_num
  have probability_val : probability = (66 : ℝ) / 4096 := by rw [favorable_outcomes_val, total_outcomes_val]
  exact probability_val

end probability_heads_exactly_10_out_of_12_flips_l526_526602


namespace non_congruent_squares_in_6_by_6_grid_l526_526390

theorem non_congruent_squares_in_6_by_6_grid : 
  let n := 6 in
  (∑ i in Range (n-1), (n-i)*(n-i) + 
   ∑ i in Range (n-1), (n-i)*(n-i) + 
   ∑ i in Range (5), 5 * 6 - i)%nat = 155 :=
by 
  let n := 6
  have h1 : ∑ i in Range (n-1), (n - i)^2 = 
    (n-1)^2 + (n-2)^2 + ... + 1 = 25 + 16 + 9 + 4 + 1 := sorry
  have h2 : ∑ i in Range (n-1), (n - i)^2 = 
    (5)^2 + (4)^2 + ... + 1^2 = 25 + 16 + 9 + 4 + 1 := sorry
  have h3 : ∑ i in Range (5), 5 * (6 - i) = 
    (5*5) + (5*4) := sorry
  show (∑ i in Range (n-1), (n-i)*(n-i) + 
       ∑ i in Range (n-1), (n-i)*(n-i) + 
       ∑ i in Range (5), 5 * (6 - i)) = 155 := by sorry

end non_congruent_squares_in_6_by_6_grid_l526_526390


namespace equal_radii_l526_526475

noncomputable def square {A B C D : Type*} (ABCD : A × B × C × D) := sorry

noncomputable def circle_tangent {A B C D : Type*} (ABCD : A × B × C × D)
  (C1 C2 C3 : Type*) (r1 r2 r3 : ℝ)
  (tangent1 : C1 × C2 × C3 → Prop)
  (tangent2 : C1 × C2 × C3 → Prop)
  (side_tangent : A × B × C × D × C1 × C2 × C3 → Prop) :=
  sorry

theorem equal_radii {A B C D : Type*} (ABCD : A × B × C × D)
  (C1 C2 C3 : Type*) (r1 r2 r3 : ℝ)
  (h_square : square ABCD)
  (tangent_circles : circle_tangent ABCD C1 C2 C3 r1 r2 r3
    (λ ⟨c1, c2, c3⟩, ∀ {x : C1 × C2 × C3}, c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3)
    (λ ⟨c1, c2, c3⟩, ∀ {x : C1 × C2 × C3}, ¬(c1 = c2) ∧ ¬(c2 = c3) ∧ ¬(c1 = c3))
    (λ ⟨a, b, c, d, c1, c2, c3⟩, ∀ {x : A × B × C × D × C1 × C2 × C3}, c1 ≠ c2)) :
  r1 = r2 ∨ r2 = r3 ∨ r1 = r3 :=
sorry

end equal_radii_l526_526475


namespace total_people_l526_526955

-- Define the conditions as constants
def B : ℕ := 50
def S : ℕ := 70
def B_inter_S : ℕ := 20

-- Total number of people in the group
theorem total_people : B + S - B_inter_S = 100 := by
  sorry

end total_people_l526_526955


namespace vision_data_l526_526547

theorem vision_data (L V : ℝ) (approx : ℝ) (h1 : L = 5 + real.log10 V) (h2 : L = 4.9) (h3 : approx = 1.259) : 
  V = 0.8 := 
sorry

end vision_data_l526_526547


namespace sum_possible_k_l526_526457

theorem sum_possible_k (j k : ℕ) (hcond : 1 / j + 1 / k = 1 / 4) (hj : 0 < j) (hk : 0 < k) :
  j ≠ k → ∑ k in {k | ∃ j, 1 / j + 1 / k = 1 / 4}, k = 51 :=
by
  sorry

end sum_possible_k_l526_526457


namespace Alyssa_next_year_games_l526_526267

theorem Alyssa_next_year_games 
  (games_this_year : ℕ) 
  (games_last_year : ℕ) 
  (total_games : ℕ) 
  (games_up_to_this_year : ℕ)
  (total_up_to_next_year : ℕ) 
  (H1 : games_this_year = 11)
  (H2 : games_last_year = 13)
  (H3 : total_up_to_next_year = 39)
  (H4 : games_up_to_this_year = games_this_year + games_last_year) :
  total_up_to_next_year - games_up_to_this_year = 15 :=
by
  sorry

end Alyssa_next_year_games_l526_526267


namespace charlie_extra_fee_l526_526748

-- Conditions
def data_limit_week1 : ℕ := 2 -- in GB
def data_limit_week2 : ℕ := 3 -- in GB
def data_limit_week3 : ℕ := 2 -- in GB
def data_limit_week4 : ℕ := 1 -- in GB

def additional_fee_week1 : ℕ := 12 -- dollars per GB
def additional_fee_week2 : ℕ := 10 -- dollars per GB
def additional_fee_week3 : ℕ := 8 -- dollars per GB
def additional_fee_week4 : ℕ := 6 -- dollars per GB

def data_used_week1 : ℕ := 25 -- in 0.1 GB
def data_used_week2 : ℕ := 40 -- in 0.1 GB
def data_used_week3 : ℕ := 30 -- in 0.1 GB
def data_used_week4 : ℕ := 50 -- in 0.1 GB

-- Additional fee calculation
def extra_data_fee := 
  let extra_data_week1 := max (data_used_week1 - data_limit_week1 * 10) 0
  let extra_fee_week1 := extra_data_week1 * additional_fee_week1 / 10
  let extra_data_week2 := max (data_used_week2 - data_limit_week2 * 10) 0
  let extra_fee_week2 := extra_data_week2 * additional_fee_week2 / 10
  let extra_data_week3 := max (data_used_week3 - data_limit_week3 * 10) 0
  let extra_fee_week3 := extra_data_week3 * additional_fee_week3 / 10
  let extra_data_week4 := max (data_used_week4 - data_limit_week4 * 10) 0
  let extra_fee_week4 := extra_data_week4 * additional_fee_week4 / 10
  extra_fee_week1 + extra_fee_week2 + extra_fee_week3 + extra_fee_week4

-- The math proof problem
theorem charlie_extra_fee : extra_data_fee = 48 := sorry

end charlie_extra_fee_l526_526748


namespace alcohol_percentage_x_l526_526260

-- Definitions derived directly from the conditions
def volume_y := 750 -- volume in milliliters
def volume_x := 250 -- volume in milliliters
def alcohol_percentage_y := 0.30 -- 30% in decimal
def target_alcohol_percentage := 0.25 -- 25% in decimal
def total_volume := volume_y + volume_x

-- Lean statement to prove the percentage of alcohol by volume in solution x
theorem alcohol_percentage_x :
  ∃ P, ((alcohol_percentage_y * volume_y) + (P * volume_x) = target_alcohol_percentage * total_volume) ∧ P = 0.10 :=
by
  sorry

end alcohol_percentage_x_l526_526260


namespace smallest_among_5_neg7_0_neg53_l526_526270

-- Define the rational numbers involved as constants
def a : ℚ := 5
def b : ℚ := -7
def c : ℚ := 0
def d : ℚ := -5 / 3

-- Define the conditions as separate lemmas
lemma positive_greater_than_zero (x : ℚ) (hx : x > 0) : x > c := by sorry
lemma zero_greater_than_negative (x : ℚ) (hx : x < 0) : c > x := by sorry
lemma compare_negative_by_absolute_value (x y : ℚ) (hx : x < 0) (hy : y < 0) (habs : |x| > |y|) : x < y := by sorry

-- Prove the main assertion
theorem smallest_among_5_neg7_0_neg53 : 
    b < a ∧ b < c ∧ b < d := by
    -- Here we apply the defined conditions to show b is the smallest
    sorry

end smallest_among_5_neg7_0_neg53_l526_526270


namespace probability_heads_exactly_10_out_of_12_flips_l526_526604

theorem probability_heads_exactly_10_out_of_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := Nat.choose 12 10
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = (66 : ℝ) / 4096 :=
by
  let total_outcomes := 2^12
  let favorable_outcomes := Nat.choose 12 10
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  have total_outcomes_val : total_outcomes = 4096 := by norm_num
  have favorable_outcomes_val : favorable_outcomes = 66 := by norm_num
  have probability_val : probability = (66 : ℝ) / 4096 := by rw [favorable_outcomes_val, total_outcomes_val]
  exact probability_val

end probability_heads_exactly_10_out_of_12_flips_l526_526604


namespace stride_difference_is_seven_l526_526764

noncomputable def elmer_strides := 40
noncomputable def oscar_leaps := 15
noncomputable def number_of_poles := 31
noncomputable def mile_in_feet := 5280

def stride_difference := 
  let gaps := number_of_poles - 1
  let total_strides := elmer_strides * gaps
  let total_leaps := oscar_leaps * gaps
  let elmer_stride_length := mile_in_feet / total_strides
  let oscar_leap_length := mile_in_feet / total_leaps
  oscar_leap_length - elmer_stride_length

theorem stride_difference_is_seven :
  stride_difference ≈ 7 := 
by
  sorry

end stride_difference_is_seven_l526_526764


namespace last_four_digits_of_series_l526_526496

def f (n : ℕ) : ℕ := 3 * n^2 - 3 * n + 1

theorem last_four_digits_of_series : 
  (f 1 + f 2 + f 3 + ... + f 2010) % 10000 = 1000 := by
  sorry

end last_four_digits_of_series_l526_526496


namespace christine_travel_distance_l526_526749

def speed (t : ℝ) : ℝ := 5 * t + 15

theorem christine_travel_distance : ∫ t in 0..4, speed t = 100 := 
by sorry

end christine_travel_distance_l526_526749


namespace find_f_2_l526_526893

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def f (a b : ℝ) (x : ℝ) := (a * x + b) / (x^2 + 1)

theorem find_f_2 (a b : ℝ) 
  (h_odd : is_odd_function (f a b)) 
  (h_value : f a b (1 / 2) = 2 / 5) : 
  f a b 2 = 2 / 5 :=
by
  sorry

end find_f_2_l526_526893


namespace triangle_circumradius_sqrt3_triangle_area_l526_526364

variables {a b c : ℝ} {A B C : ℝ} {R : ℝ}
variables (triangle_ABC : a = 2 * R * sin A ∧ c = 2 * R * sin C ∧ b = 2 * R * sin B)

theorem triangle_circumradius_sqrt3 (ha : a * sin C + sqrt 3 * c * cos A = 0) (hR : R = sqrt 3)
  (hsinC_nonzero : sin C ≠ 0) : a = 3 :=
begin
  sorry
end

theorem triangle_area (ha : a = 3) (hb : b + c = sqrt 11) (hA : A = 2 * real.pi / 3) : 
  (1/2)*b*c*sin A = sqrt 3 / 2 :=
begin
  sorry
end

end triangle_circumradius_sqrt3_triangle_area_l526_526364


namespace problem1_problem2_l526_526637

-- Problem 1
theorem problem1 : abs (-3) - real.sqrt 12 + 2 * real.sin (real.pi / 6) + (-1 : ℤ)^2021 = 3 - 2 * real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem2 {x : ℝ} : (2 * x + 3) / (x - 2) - 1 = (x - 1) / (2 - x) → x = -2 := by
  sorry

end problem1_problem2_l526_526637


namespace compare_constants_l526_526865

noncomputable def a : ℝ := 0.1 * real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := - real.log 0.9

theorem compare_constants : c < a ∧ a < b :=
by sorry

end compare_constants_l526_526865


namespace repeating_decimal_difference_l526_526703

theorem repeating_decimal_difference :
  let x := 72 / 99 in
  let y := 72 / 100 in
  x - y = 2 / 275 := 
by
  -- we will add a proof later
  sorry

end repeating_decimal_difference_l526_526703


namespace find_f_2_l526_526892

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def f (a b : ℝ) (x : ℝ) := (a * x + b) / (x^2 + 1)

theorem find_f_2 (a b : ℝ) 
  (h_odd : is_odd_function (f a b)) 
  (h_value : f a b (1 / 2) = 2 / 5) : 
  f a b 2 = 2 / 5 :=
by
  sorry

end find_f_2_l526_526892


namespace difference_between_percent_and_fraction_l526_526530

-- Define the number
def num : ℕ := 140

-- Define the percentage and fraction calculations
def percent_65 (n : ℕ) : ℕ := (65 * n) / 100
def fraction_4_5 (n : ℕ) : ℕ := (4 * n) / 5

-- Define the problem's conditions and the required proof
theorem difference_between_percent_and_fraction : 
  percent_65 num ≤ fraction_4_5 num ∧ (fraction_4_5 num - percent_65 num = 21) :=
by
  sorry

end difference_between_percent_and_fraction_l526_526530


namespace repeatingDecimal_exceeds_l526_526695

noncomputable def repeatingDecimalToFraction (d : ℚ) : ℚ := 
    -- Function to convert repeating decimal to fraction
    if d = 0.99999 then 1 else (d * 100 - d) / 99  -- Simplified conversion for demonstration

def decimalToFraction (d : ℚ) : ℚ :=
    -- Function to convert decimal to fraction
    if d = 0.72 then 18 / 25 else 0  -- Replace with actual conversion

theorem repeatingDecimal_exceeds (x y : ℚ) (hx : repeatingDecimalToFraction x = 8/11) (hy : decimalToFraction y = 18/25):
    x - y = 2 / 275 :=
by
    sorry

end repeatingDecimal_exceeds_l526_526695


namespace equilateral_triangle_cut_l526_526421

variable (A B C : Type) [triangle A B C]
variable {H : Type} [orthocenter H A B C]
variable {O : Type} [circumcenter O A B C]
variable {G : Type} [centroid G A B C]
variable {M : Type} [is_intersection M (line H O) (line A B)]
variable {N : Type} [is_intersection N (line H O) (line A C)]
variable {angle_A : Type} [angle A = 60]
variable {acute_triangle : Type} [is_acute_triangle A B C]

theorem equilateral_triangle_cut :
  angle A = 60 -> is_acute_triangle A B C ->
  cuts_off_equilateral_triangle (line O G) :=
by
  sorry

end equilateral_triangle_cut_l526_526421


namespace range_of_a_l526_526943

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x + a| + |x - 2| + a < 2010) ↔ a < 1006 :=
sorry

end range_of_a_l526_526943


namespace compare_values_l526_526809

variable (a b c : ℝ)

noncomputable def a_def : ℝ := 0.1 * Real.exp 0.1
noncomputable def b_def : ℝ := 1 / 9
noncomputable def c_def : ℝ := - Real.log 0.9

theorem compare_values : 
  c < a ∧ a < b := 
begin
  let a := 0.1 * Real.exp 0.1,
  let b := 1 / 9,
  let c := - Real.log 0.9,
  have h1 : c < a, sorry,
  have h2 : a < b, sorry,
  exact ⟨h1, h2⟩
end

end compare_values_l526_526809


namespace hyperbola_distance_focus_asymptote_l526_526318

theorem hyperbola_distance_focus_asymptote :
  let a := 2
  let b := Real.sqrt 2
  let c := Real.sqrt (a^2 + b^2)
  let focus := (c, 0)
  let asymptote_slope := b / a
  ∀ (focus : ℝ × ℝ) (asymptote_slope b real.sqrt 2 / a real.sqrt 2) x,
    Real.abs ((asymptote_slope * focus.1 - focus.2) / Real.sqrt (asymptote_slope ^ 2 + 1)) = Real.sqrt 2 := sorry

end hyperbola_distance_focus_asymptote_l526_526318


namespace cyclist_avg_speed_first_part_l526_526245

/-
Given the following conditions:
1. Total distance for the first part: 9 km.
2. Total distance for the second part: 11 km.
3. Average speed for the second part: 9 km/hr.
4. Total average speed for the entire trip: 9.8019801980198 km/hr.

Prove that the average speed during the first part of the trip is 11.03 km/hr.
-/
theorem cyclist_avg_speed_first_part :
  ∃ (v : ℝ), 
    (∀ (d1 d2 s2 s_avg t1 t2 : ℝ),
      d1 = 9 →
      d2 = 11 →
      s2 = 9 →
      s_avg = 9.8019801980198 →
      t1 = d1 / v →
      t2 = d2 / s2 →
      s_avg = (d1 + d2) / (t1 + t2) → 
      v = 11.03) :=
begin
  use 11.03, -- the speed during the first part
  intros d1 d2 s2 s_avg t1 t2 h_d1 h_d2 h_s2 h_s_avg h_t1 h_t2 h_s_avg_eq,
  sorry
end

end cyclist_avg_speed_first_part_l526_526245


namespace find_f_at_six_l526_526252

theorem find_f_at_six (f : ℝ → ℝ) (h : ∀ x : ℝ, f (4 * x - 2) = x^2 - x + 2) : f 6 = 3.75 :=
by
  sorry

end find_f_at_six_l526_526252


namespace min_value_sum_l526_526398

theorem min_value_sum (n : ℕ) (h : n ≥ 2) (x : Fin n → ℝ) (h₀ : ∀ i, 0 < x i)
    (h₁ : ∑ i, (x i) ^ 2 = 1) :
    (∑ i, (x i) ^ 5 / (∑ j, x j - x i)) ≥ 1 / (n * (n - 1)) :=
sorry

end min_value_sum_l526_526398


namespace carton_weight_l526_526262

theorem carton_weight :
  ∀ (x : ℝ),
  (12 * 4 + 16 * x = 96) → 
  x = 3 :=
by
  intros x h
  sorry

end carton_weight_l526_526262


namespace no_real_solution_f_of_f_f_eq_x_l526_526036

-- Defining the quadratic polynomial f(x) = ax^2 + bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Stating the main theorem
theorem no_real_solution_f_of_f_f_eq_x (a b c : ℝ) (h : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x :=
by 
  -- Proof will go here
  sorry

end no_real_solution_f_of_f_f_eq_x_l526_526036


namespace tan_2A_l526_526900

theorem tan_2A (A B C : ℝ) (a b c : ℝ) (S : ℝ) (cos_A sin_A : ℝ)
  (h1 : ∀ B C, S = (1 / 2) * b * c * sin_A)
  (h2 : ∀ B C, S = b * c * cos_A)
  (h3 : cos_A = 1 / 2 * sin_A)
  (h4 : tan_A = 2) :
  tan (2 * A) = -4 / 3 :=
sorry

end tan_2A_l526_526900


namespace complex_pure_imaginary_l526_526405

theorem complex_pure_imaginary (a : ℝ) 
  (h1 : a^2 + 2*a - 3 = 0) 
  (h2 : a + 3 ≠ 0) : 
  a = 1 := 
by
  sorry

end complex_pure_imaginary_l526_526405


namespace max_circles_equidistant_l526_526355

-- Define the conditions
variables (P1 P2 P3 P4 : Point)

def no_three_collinear (P1 P2 P3 P4 : Point) : Prop :=
  ¬ (Collinear P1 P2 P3) ∧ ¬ (Collinear P1 P2 P4) ∧ ¬ (Collinear P1 P3 P4) ∧ ¬ (Collinear P2 P3 P4)

def not_all_on_same_circle (P1 P2 P3 P4 : Point) : Prop :=
  ¬ (∃ k : Circle, P1 ∈ k ∧ P2 ∈ k ∧ P3 ∈ k ∧ P4 ∈ k)

-- Lean statement capturing the problem conditions and conclusion
theorem max_circles_equidistant (P1 P2 P3 P4 : Point) :
  no_three_collinear P1 P2 P3 P4 →
  not_all_on_same_circle P1 P2 P3 P4 →
  ∃ (k_set : Finset Circle), k_set.card = 7 ∧
    ∀ k ∈ k_set, equidistant_from_all P1 P2 P3 P4 k :=
sorry

end max_circles_equidistant_l526_526355


namespace slope_range_of_inclination_angle_l526_526202

theorem slope_range_of_inclination_angle (θ : ℝ) (k : ℝ) 
  (hθ1 : 60 ≤ θ) (hθ2 : θ ≤ 135) (hk : k = Real.tan θ) : 
  k ∈ Iic (-1) ∪ Ici (Real.sqrt 3) :=
sorry

end slope_range_of_inclination_angle_l526_526202


namespace lumberjack_question_l526_526254

def logs_per_tree (total_firewood : ℕ) (firewood_per_log : ℕ) (trees_chopped : ℕ) : ℕ :=
  total_firewood / firewood_per_log / trees_chopped

theorem lumberjack_question : logs_per_tree 500 5 25 = 4 := by
  sorry

end lumberjack_question_l526_526254


namespace rate_calculation_l526_526194

variable (SI P T : ℝ) (R : ℝ)

def simple_interest_rate (P T SI : ℝ) : ℝ := (SI * 100) / (P * T)

theorem rate_calculation (hSI : SI = 250) (hP : P = 1500) (hT : T = 5) :
  simple_interest_rate P T SI = 3.33 :=
by
  sorry

end rate_calculation_l526_526194


namespace petya_wins_l526_526575

def circuit_game (n : Nat) :=
  n = 2000 ∧ ∀ (cuts_vasya cuts_petya : ℕ), cuts_vasya = 1 ∧ (cuts_petya = 1 ∨ cuts_petya = 3)

theorem petya_wins : circuit_game 2000 →
  ∃ strategy_petya, ∀ strategy_vasya, petya_wins_strategy strategy_petya strategy_vasya
  := by sorry


end petya_wins_l526_526575


namespace coefficient_x5_in_expansion_l526_526193

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binomial_coefficient n k + binomial_coefficient n (k + 1)

open_locale big_operators

theorem coefficient_x5_in_expansion :
  ∀ (x : ℝ), (x + 3 * real.sqrt 2)^9.coeff 5 = 20412 :=
begin
  -- Your proof goes here.
  sorry
end

end coefficient_x5_in_expansion_l526_526193


namespace cheaper_to_buy_more_l526_526149

def C (n : ℕ) : ℕ := 
if 1 ≤ n ∧ n ≤ 20 then 15 * n
else if 21 ≤ n ∧ n ≤ 40 then 13 * n - 10
else if 41 ≤ n ∧ n ≤ 60 then 12 * n
else if 61 ≤ n then 11 * n
else 0 -- Default case which will not be used

theorem cheaper_to_buy_more (n : ℕ) : 
 ( ∃ m : ℕ, m > n ∧ C(m) < C(n) ) → n ∈ {19, 20, 38, 39, 40, 59, 60} ∨ n ∈ {21, 41, 61} →
 n ∈ finset.range 61 → set.card {n | n ∈ {19, 20, 38, 39, 40, 59, 60}} = 7 :=
sorry

end cheaper_to_buy_more_l526_526149


namespace seq_36_eq_363_l526_526401

def seq_fn : ℕ → ℕ
| 1 := 6
| 2 := 63
| 3 := 363
| 4 := 3634
| 5 := 3645
| 6 := 365
| n := if n > 6 then 363 else 0

theorem seq_36_eq_363 : seq_fn 36 = 363 := 
by
  -- Provide proof here or use sorry
  sorry

end seq_36_eq_363_l526_526401


namespace correct_derivative_option_l526_526203

-- Define the derivative expressions for each option
def option_A := deriv (λ x : ℝ, 3 * x^2 + 2) = λ x, 6 * x
def option_B := deriv (λ x : ℝ, sin x) = λ x, cos x
def option_C := deriv (λ x : ℝ, - (1 / x)) = λ x, 1 / x^2
def option_D := deriv (λ x : ℝ, 2 * exp x) = λ x, log 2 * (2 * exp x)

-- The theorem stating that the correct option is C
theorem correct_derivative_option : option_C := 
by sorry

end correct_derivative_option_l526_526203


namespace value_of_A_l526_526638

theorem value_of_A (A B C D : ℕ) (h1 : A * B = 60) (h2 : C * D = 60) (h3 : A - B = C + D) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : A ≠ D) (h7 : B ≠ C) (h8 : B ≠ D) (h9 : C ≠ D) : A = 20 :=
by sorry

end value_of_A_l526_526638


namespace smallest_possible_value_of_ratio_l526_526219

/-
We say that a polygon  P  is inscribed in another polygon  Q  when all the vertices of  P  belong to the perimeter of  Q. We also say in this case that  Q  is circumscribed to  P . 
Given a triangle  T  with vertices A(0, a), B(-b, 0), and C(c, 0), let  ℓ  be the largest side of a square inscribed in  T  and  L  is the shortest side of a square circumscribed to  T.
Find the smallest possible value of the ratio  L/ℓ.
-/

/-- 
  Given a triangle T, prove that the smallest possible value of the ratio L/ℓ, 
  where ℓ is the largest side of a square inscribed in T and L is the shortest side of a square circumscribed around T, is 2. 
-/
theorem smallest_possible_value_of_ratio (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) : ∃ (L ℓ : ℝ), 
  ( ∀ ℓ, ℓ is the largest side of a square inscribed in △ABC) ∧
  ( ∀ L, L is the shortest side of a square circumscribed around △ABC) ∧
  (L/ℓ = 2) := 
sorry

end smallest_possible_value_of_ratio_l526_526219


namespace find_a9_for_geo_seq_l526_526899

noncomputable def geo_seq_a_3_a_13_positive_common_ratio_2 (a_3 a_9 a_13 : ℕ) : Prop :=
  (a_3 * a_13 = 16) ∧ (a_3 > 0) ∧ (a_9 > 0) ∧ (a_13 > 0) ∧ (forall (n₁ n₂ : ℕ), a_9 = a_3 * 2 ^ 6)

theorem find_a9_for_geo_seq (a_3 a_9 a_13 : ℕ) 
  (h : geo_seq_a_3_a_13_positive_common_ratio_2 a_3 a_9 a_13) :
  a_9 = 8 :=
  sorry

end find_a9_for_geo_seq_l526_526899


namespace circle_projection_eccentricity_l526_526182

theorem circle_projection_eccentricity (M N : Plane) (theta : ℝ) (a : ℝ) 
  (intersect_angle : M.angle_with N = theta) : 
  (eccentricity (ellipse_of_projection M N (circle M a))) = sin theta := 
  sorry

end circle_projection_eccentricity_l526_526182


namespace find_projection_l526_526896

def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_v := v.1^2 + v.2^2
  (dot_uv / norm_v * v.1, dot_uv / norm_v * v.2)

variable (u : ℝ × ℝ)
variable (cond1 : proj (2, -2) u = (1/2, -1/2))

theorem find_projection :
  proj (5, 3) u = (1, -1) :=
sorry

end find_projection_l526_526896


namespace triangle_side_lengths_l526_526586

theorem triangle_side_lengths :
  (angle BAC = 60) →
  right_triangle AMN →
  centers O1 O2 →
  points_of_tangency BC P R →
  (right_triangle O1 O2 Q) →
  Q_on O2R Q →
  (O1O2^2 = O1Q^2 + QO2^2) →
  (16 = O1Q^2 + 4) →
  (O1Q = 2 * sqrt 3) →
  (angle O2 O1 Q = 30) →
  (angle O1 O2 Q = 60) →
  (angle RO2C = 75) →
  (angle O2CR = 15) →
  (angle BCA = 30) →
  (angle ABC = 90) →
  (RC = 6 + 3 * sqrt 3) →
  BC = (7 + 5 * sqrt 3) ∧
  AC = (30 + 14 * sqrt 3) / 3 ∧
  AB = (15 + 7 * sqrt 3) / 3 :=
  by
    sorry -- Proof omitted

end triangle_side_lengths_l526_526586
