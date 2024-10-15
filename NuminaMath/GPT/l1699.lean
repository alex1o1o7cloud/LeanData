import Mathlib

namespace NUMINAMATH_GPT_last_three_digits_of_5_power_odd_l1699_169907

theorem last_three_digits_of_5_power_odd (n : ℕ) (h : n % 2 = 1) : (5 ^ n) % 1000 = 125 :=
sorry

end NUMINAMATH_GPT_last_three_digits_of_5_power_odd_l1699_169907


namespace NUMINAMATH_GPT_proposition_B_correct_l1699_169975

theorem proposition_B_correct : ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x :=
by
  sorry

end NUMINAMATH_GPT_proposition_B_correct_l1699_169975


namespace NUMINAMATH_GPT_circle_center_eq_l1699_169920

theorem circle_center_eq (x y : ℝ) :
    (x^2 + y^2 - 2*x + y + 1/4 = 0) → (x = 1 ∧ y = -1/2) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_eq_l1699_169920


namespace NUMINAMATH_GPT_prob_diff_colors_correct_l1699_169918

noncomputable def total_outcomes : ℕ :=
  let balls_pocket1 := 2 + 3 + 5
  let balls_pocket2 := 2 + 4 + 4
  balls_pocket1 * balls_pocket2

noncomputable def favorable_outcomes_same_color : ℕ :=
  let white_balls := 2 * 2
  let red_balls := 3 * 4
  let yellow_balls := 5 * 4
  white_balls + red_balls + yellow_balls

noncomputable def prob_same_color : ℚ :=
  favorable_outcomes_same_color / total_outcomes

noncomputable def prob_different_color : ℚ :=
  1 - prob_same_color

theorem prob_diff_colors_correct :
  prob_different_color = 16 / 25 :=
by sorry

end NUMINAMATH_GPT_prob_diff_colors_correct_l1699_169918


namespace NUMINAMATH_GPT_range_of_m_l1699_169908

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + |x - 1| > m) → m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1699_169908


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1699_169958

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 1)
  (h3 : a 3 = 11)
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = d) : d = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1699_169958


namespace NUMINAMATH_GPT_dad_strawberries_final_weight_l1699_169943

variable {M D : ℕ}

theorem dad_strawberries_final_weight :
  M + D = 22 →
  36 - M + 30 + D = D' →
  D' = 46 :=
by
  intros h h1
  sorry

end NUMINAMATH_GPT_dad_strawberries_final_weight_l1699_169943


namespace NUMINAMATH_GPT_complement_set_l1699_169973

def U := {x : ℝ | x > 0}
def A := {x : ℝ | x > 2}
def complement_U_A := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem complement_set :
  {x : ℝ | x ∈ U ∧ x ∉ A} = complement_U_A :=
sorry

end NUMINAMATH_GPT_complement_set_l1699_169973


namespace NUMINAMATH_GPT_non_arithmetic_sequence_l1699_169921

theorem non_arithmetic_sequence (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) :
    (∀ n, S_n n = n^2 + 2 * n - 1) →
    (∀ n, a_n n = if n = 1 then S_n 1 else S_n n - S_n (n - 1)) →
    ¬(∀ d, ∀ n, a_n (n+1) = a_n n + d) :=
by
  intros hS ha
  sorry

end NUMINAMATH_GPT_non_arithmetic_sequence_l1699_169921


namespace NUMINAMATH_GPT_simplify_frac_op_l1699_169937

-- Definition of the operation *
def frac_op (a b c d : ℚ) : ℚ := (a * c) * (d / (b + 1))

-- Proof problem stating the specific operation result
theorem simplify_frac_op :
  frac_op 5 11 9 4 = 15 :=
by
  sorry

end NUMINAMATH_GPT_simplify_frac_op_l1699_169937


namespace NUMINAMATH_GPT_find_initial_shells_l1699_169913

theorem find_initial_shells (x : ℕ) (h : x + 23 = 28) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_shells_l1699_169913


namespace NUMINAMATH_GPT_tens_of_80_tens_of_190_l1699_169930

def tens_place (n : Nat) : Nat :=
  (n / 10) % 10

theorem tens_of_80 : tens_place 80 = 8 := 
  by
  sorry

theorem tens_of_190 : tens_place 190 = 9 := 
  by
  sorry

end NUMINAMATH_GPT_tens_of_80_tens_of_190_l1699_169930


namespace NUMINAMATH_GPT_correct_transformation_l1699_169935

theorem correct_transformation (x : ℝ) : x^2 - 10 * x - 1 = 0 → (x - 5)^2 = 26 :=
  sorry

end NUMINAMATH_GPT_correct_transformation_l1699_169935


namespace NUMINAMATH_GPT_area_of_plot_l1699_169925

def cm_to_miles (a : ℕ) : ℕ := a * 9

def miles_to_acres (b : ℕ) : ℕ := b * 640

theorem area_of_plot :
  let bottom := 12
  let top := 18
  let height := 10
  let area_cm2 := ((bottom + top) * height) / 2
  let area_miles2 := cm_to_miles area_cm2
  let area_acres := miles_to_acres area_miles2
  area_acres = 864000 :=
by
  sorry

end NUMINAMATH_GPT_area_of_plot_l1699_169925


namespace NUMINAMATH_GPT_parallelogram_s_value_l1699_169990

noncomputable def parallelogram_area (s : ℝ) : ℝ :=
  s * 2 * (s / Real.sqrt 2)

theorem parallelogram_s_value (s : ℝ) (h₀ : parallelogram_area s = 8 * Real.sqrt 2) : 
  s = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_s_value_l1699_169990


namespace NUMINAMATH_GPT_factorial_less_power_l1699_169911

open Nat

noncomputable def factorial_200 : ℕ := 200!

noncomputable def power_100_200 : ℕ := 100 ^ 200

theorem factorial_less_power : factorial_200 < power_100_200 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_factorial_less_power_l1699_169911


namespace NUMINAMATH_GPT_jim_taxi_distance_l1699_169981

theorem jim_taxi_distance (initial_fee charge_per_segment total_charge : ℝ) (segment_len_miles : ℝ)
(init_fee_eq : initial_fee = 2.5)
(charge_per_seg_eq : charge_per_segment = 0.35)
(total_charge_eq : total_charge = 5.65)
(segment_length_eq : segment_len_miles = 2/5):
  let charge_for_distance := total_charge - initial_fee
  let num_segments := charge_for_distance / charge_per_segment
  let total_miles := num_segments * segment_len_miles
  total_miles = 3.6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jim_taxi_distance_l1699_169981


namespace NUMINAMATH_GPT_evaluate_expression_l1699_169963

theorem evaluate_expression : 
  (2 ^ 2015 + 2 ^ 2013 + 2 ^ 2011) / (2 ^ 2015 - 2 ^ 2013 + 2 ^ 2011) = 21 / 13 := 
by 
 sorry

end NUMINAMATH_GPT_evaluate_expression_l1699_169963


namespace NUMINAMATH_GPT_greatest_integer_less_than_M_over_100_l1699_169983

theorem greatest_integer_less_than_M_over_100
  (h : (1/(Nat.factorial 3 * Nat.factorial 18) + 1/(Nat.factorial 4 * Nat.factorial 17) + 
        1/(Nat.factorial 5 * Nat.factorial 16) + 1/(Nat.factorial 6 * Nat.factorial 15) + 
        1/(Nat.factorial 7 * Nat.factorial 14) + 1/(Nat.factorial 8 * Nat.factorial 13) + 
        1/(Nat.factorial 9 * Nat.factorial 12) + 1/(Nat.factorial 10 * Nat.factorial 11) = 
        1/(Nat.factorial 2 * Nat.factorial 19) * (M : ℚ))) :
  ⌊M / 100⌋ = 499 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_less_than_M_over_100_l1699_169983


namespace NUMINAMATH_GPT_find_percentage_l1699_169967

theorem find_percentage (P : ℝ) (h1 : (3 / 5) * 150 = 90) (h2 : (P / 100) * 90 = 36) : P = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l1699_169967


namespace NUMINAMATH_GPT_derek_age_l1699_169976

theorem derek_age (C D E : ℝ) (h1 : C = 4 * D) (h2 : E = D + 5) (h3 : C = E) : D = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_derek_age_l1699_169976


namespace NUMINAMATH_GPT_max_m_value_l1699_169945

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem max_m_value 
  (t : ℝ) 
  (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) <= x) : m ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_m_value_l1699_169945


namespace NUMINAMATH_GPT_part1_part2_l1699_169968

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1699_169968


namespace NUMINAMATH_GPT_percentage_third_day_l1699_169948

def initial_pieces : ℕ := 1000
def percentage_first_day : ℝ := 0.10
def percentage_second_day : ℝ := 0.20
def pieces_left_after_third_day : ℕ := 504

theorem percentage_third_day :
  let pieces_first_day := initial_pieces * percentage_first_day
  let remaining_after_first_day := initial_pieces - pieces_first_day
  let pieces_second_day := remaining_after_first_day * percentage_second_day
  let remaining_after_second_day := remaining_after_first_day - pieces_second_day
  let pieces_third_day := remaining_after_second_day - pieces_left_after_third_day
  (pieces_third_day / remaining_after_second_day * 100 = 30) :=
by
  sorry

end NUMINAMATH_GPT_percentage_third_day_l1699_169948


namespace NUMINAMATH_GPT_below_sea_level_representation_l1699_169944

def depth_below_sea_level (depth: Int → Prop) :=
  ∀ x, depth x → x < 0

theorem below_sea_level_representation (D: Int) (A: Int) 
  (hA: A = 9050) (hD: D = 10907) 
  (above_sea_level: Int → Prop) (below_sea_level: Int → Prop)
  (h1: ∀ y, above_sea_level y → y > 0):
  below_sea_level (-D) :=
by
  sorry

end NUMINAMATH_GPT_below_sea_level_representation_l1699_169944


namespace NUMINAMATH_GPT_recurring_decimal_fraction_l1699_169909

theorem recurring_decimal_fraction (h54 : (0.54 : ℝ) = 54 / 99) (h18 : (0.18 : ℝ) = 18 / 99) :
    (0.54 / 0.18 : ℝ) = 3 := 
by
  sorry

end NUMINAMATH_GPT_recurring_decimal_fraction_l1699_169909


namespace NUMINAMATH_GPT_digitalEarthFunctions_l1699_169916

axiom OptionA (F : Type) : Prop
axiom OptionB (F : Type) : Prop
axiom OptionC (F : Type) : Prop
axiom OptionD (F : Type) : Prop

axiom isRemoteSensing (F : Type) : OptionA F
axiom isGIS (F : Type) : OptionB F
axiom isGPS (F : Type) : OptionD F

theorem digitalEarthFunctions {F : Type} : OptionC F :=
sorry

end NUMINAMATH_GPT_digitalEarthFunctions_l1699_169916


namespace NUMINAMATH_GPT_find_k_l1699_169932

noncomputable def parabola_k : ℝ := 4

theorem find_k (k : ℝ) (h1 : ∀ x, y = k^2 - x^2) (h2 : k > 0)
    (h3 : ∀ A D : (ℝ × ℝ), A = (-k, 0) ∧ D = (k, 0))
    (h4 : ∀ V : (ℝ × ℝ), V = (0, k^2))
    (h5 : 2 * (2 * k + k^2) = 48) : k = 4 :=
  sorry

end NUMINAMATH_GPT_find_k_l1699_169932


namespace NUMINAMATH_GPT_even_function_a_eq_neg1_l1699_169933

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * (Real.exp x + a * Real.exp (-x))

/-- Given that the function f(x) = x(e^x + a e^{-x}) is an even function, prove that a = -1. -/
theorem even_function_a_eq_neg1 (a : ℝ) (h : ∀ x : ℝ, f a x = f a (-x)) : a = -1 :=
sorry

end NUMINAMATH_GPT_even_function_a_eq_neg1_l1699_169933


namespace NUMINAMATH_GPT_arrow_reading_l1699_169901

-- Define the interval and values within it
def in_range (x : ℝ) : Prop := 9.75 ≤ x ∧ x ≤ 10.00
def closer_to_990 (x : ℝ) : Prop := |x - 9.90| < |x - 9.875|

-- The main theorem statement expressing the problem
theorem arrow_reading (x : ℝ) (hx1 : in_range x) (hx2 : closer_to_990 x) : x = 9.90 :=
by sorry

end NUMINAMATH_GPT_arrow_reading_l1699_169901


namespace NUMINAMATH_GPT_base_seven_to_ten_l1699_169900

theorem base_seven_to_ten : 
  (7 * 7^4 + 6 * 7^3 + 5 * 7^2 + 4 * 7^1 + 3 * 7^0) = 19141 := 
by 
  sorry

end NUMINAMATH_GPT_base_seven_to_ten_l1699_169900


namespace NUMINAMATH_GPT_value_of_f_at_13_over_2_l1699_169962

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_at_13_over_2
  (h1 : ∀ x : ℝ , f (-x) = -f (x))
  (h2 : ∀ x : ℝ , f (x - 2) = f (x + 2))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f (x) = -x^2) :
  f (13 / 2) = 9 / 4 :=
sorry

end NUMINAMATH_GPT_value_of_f_at_13_over_2_l1699_169962


namespace NUMINAMATH_GPT_intersection_distance_l1699_169977

theorem intersection_distance (p q : ℕ) (h1 : p = 65) (h2 : q = 2) :
  p - q = 63 := 
by
  sorry

end NUMINAMATH_GPT_intersection_distance_l1699_169977


namespace NUMINAMATH_GPT_daily_wage_of_man_l1699_169980

-- Define the wages for men and women
variables (M W : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 24 * M + 16 * W = 11600
def condition2 : Prop := 12 * M + 37 * W = 11600

-- The theorem we want to prove
theorem daily_wage_of_man (h1 : condition1 M W) (h2 : condition2 M W) : M = 350 :=
by
  sorry

end NUMINAMATH_GPT_daily_wage_of_man_l1699_169980


namespace NUMINAMATH_GPT_find_counterfeit_two_weighings_l1699_169929

-- defining the variables and conditions
variable (coins : Fin 7 → ℝ)
variable (real_weight : ℝ)
variable (fake_weight : ℝ)
variable (is_counterfeit : Fin 7 → Prop)

-- conditions
axiom counterfeit_weight_diff : ∀ i, is_counterfeit i ↔ (coins i = fake_weight)
axiom consecutive_counterfeits : ∃ (start : Fin 7), ∀ i, (start ≤ i ∧ i < start + 4) → is_counterfeit (i % 7)
axiom weight_diff : fake_weight < real_weight

-- Theorem statement
theorem find_counterfeit_two_weighings : 
  (coins (1 : Fin 7) + coins (2 : Fin 7) = coins (4 : Fin 7) + coins (5 : Fin 7)) →
  is_counterfeit (6 : Fin 7) ∧ is_counterfeit (7 : Fin 7) := 
sorry

end NUMINAMATH_GPT_find_counterfeit_two_weighings_l1699_169929


namespace NUMINAMATH_GPT_Willy_Lucy_more_crayons_l1699_169979

def Willy_initial : ℕ := 1400
def Lucy_initial : ℕ := 290
def Max_crayons : ℕ := 650
def Willy_giveaway_percent : ℚ := 25 / 100
def Lucy_giveaway_percent : ℚ := 10 / 100

theorem Willy_Lucy_more_crayons :
  let Willy_remaining := Willy_initial - Willy_initial * Willy_giveaway_percent
  let Lucy_remaining := Lucy_initial - Lucy_initial * Lucy_giveaway_percent
  Willy_remaining + Lucy_remaining - Max_crayons = 661 := by
  sorry

end NUMINAMATH_GPT_Willy_Lucy_more_crayons_l1699_169979


namespace NUMINAMATH_GPT_sum_tens_ones_digits_3_plus_4_power_17_l1699_169986

def sum_of_digits (n : ℕ) : ℕ :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit

theorem sum_tens_ones_digits_3_plus_4_power_17 :
  sum_of_digits ((3 + 4) ^ 17) = 7 :=
  sorry

end NUMINAMATH_GPT_sum_tens_ones_digits_3_plus_4_power_17_l1699_169986


namespace NUMINAMATH_GPT_lcm_of_numbers_is_91_l1699_169984

def ratio (a b : ℕ) (p q : ℕ) : Prop := p * b = q * a

theorem lcm_of_numbers_is_91 (a b : ℕ) (h_ratio : ratio a b 7 13) (h_gcd : Nat.gcd a b = 15) :
  Nat.lcm a b = 91 := 
by sorry

end NUMINAMATH_GPT_lcm_of_numbers_is_91_l1699_169984


namespace NUMINAMATH_GPT_quadratic_roots_r_l1699_169964

theorem quadratic_roots_r (a b m p r : ℚ) :
  (∀ x : ℚ, x^2 - m * x + 3 = 0 → (x = a ∨ x = b)) →
  (∀ x : ℚ, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a + 1)) →
  r = 19 / 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_r_l1699_169964


namespace NUMINAMATH_GPT_digit_difference_l1699_169993

theorem digit_difference (X Y : ℕ) (h1 : 0 ≤ X ∧ X ≤ 9) (h2 : 0 ≤ Y ∧ Y ≤ 9) (h3 : (10 * X + Y) - (10 * Y + X) = 54) : X - Y = 6 :=
sorry

end NUMINAMATH_GPT_digit_difference_l1699_169993


namespace NUMINAMATH_GPT_least_total_bananas_is_1128_l1699_169991

noncomputable def least_total_bananas : ℕ :=
  let b₁ := 252
  let b₂ := 252
  let b₃ := 336
  let b₄ := 288
  b₁ + b₂ + b₃ + b₄

theorem least_total_bananas_is_1128 :
  least_total_bananas = 1128 :=
by
  sorry

end NUMINAMATH_GPT_least_total_bananas_is_1128_l1699_169991


namespace NUMINAMATH_GPT_alice_lost_second_game_l1699_169926

/-- Alice, Belle, and Cathy had an arm-wrestling contest. In each game, two girls wrestled, while the third rested.
After each game, the winner played the next game against the girl who had rested.
Given that Alice played 10 times, Belle played 15 times, and Cathy played 17 times; prove Alice lost the second game. --/

theorem alice_lost_second_game (alice_plays : ℕ) (belle_plays : ℕ) (cathy_plays : ℕ) :
  alice_plays = 10 → belle_plays = 15 → cathy_plays = 17 → 
  ∃ (lost_second_game : String), lost_second_game = "Alice" := by
  intros hA hB hC
  sorry

end NUMINAMATH_GPT_alice_lost_second_game_l1699_169926


namespace NUMINAMATH_GPT_stratified_sampling_total_students_sampled_l1699_169985

theorem stratified_sampling_total_students_sampled 
  (seniors juniors freshmen : ℕ)
  (sampled_freshmen : ℕ)
  (ratio : ℚ)
  (h_freshmen : freshmen = 1500)
  (h_sampled_freshmen_ratio : sampled_freshmen = 75)
  (h_seniors : seniors = 1000)
  (h_juniors : juniors = 1200)
  (h_ratio : ratio = (sampled_freshmen : ℚ) / (freshmen : ℚ))
  (h_freshmen_ratio : ratio * (freshmen : ℚ) = sampled_freshmen) :
  let sampled_juniors := ratio * (juniors : ℚ)
  let sampled_seniors := ratio * (seniors : ℚ)
  sampled_freshmen + sampled_juniors + sampled_seniors = 185 := sorry

end NUMINAMATH_GPT_stratified_sampling_total_students_sampled_l1699_169985


namespace NUMINAMATH_GPT_geom_series_sum_l1699_169955

noncomputable def geom_sum (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1) 

theorem geom_series_sum (S : ℕ) (a r n : ℕ) (eq1 : a = 1) (eq2 : r = 3)
  (eq3 : 19683 = a * r^(n-1)) (S_eq : S = geom_sum a r n) : 
  S = 29524 :=
by
  sorry

end NUMINAMATH_GPT_geom_series_sum_l1699_169955


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l1699_169917

theorem arithmetic_geometric_sequence (d : ℤ) (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n, a n * a (n + 1) = a (n - 1) * a (n + 2)) :
  a 2017 = 1 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l1699_169917


namespace NUMINAMATH_GPT_sum_series_div_3_powers_l1699_169903

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end NUMINAMATH_GPT_sum_series_div_3_powers_l1699_169903


namespace NUMINAMATH_GPT_num_invalid_d_l1699_169997

noncomputable def square_and_triangle_problem (d : ℕ) : Prop :=
  ∃ a b : ℕ, 3 * a - 4 * b = 1989 ∧ a - b = d ∧ b > 0

theorem num_invalid_d : ∀ (d : ℕ), (d ≤ 663) → ¬ square_and_triangle_problem d :=
by {
  sorry
}

end NUMINAMATH_GPT_num_invalid_d_l1699_169997


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1699_169942

open Set

-- Define the sets U and A with their respective elements in the real numbers
def U : Set ℝ := Icc 0 1
def A : Set ℝ := Ico 0 1

-- State the theorem
theorem complement_of_A_in_U : (U \ A) = {1} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1699_169942


namespace NUMINAMATH_GPT_total_cost_is_58_l1699_169995

-- Define the conditions
def cost_per_adult : Nat := 22
def cost_per_child : Nat := 7
def number_of_adults : Nat := 2
def number_of_children : Nat := 2

-- Define the theorem to prove the total cost
theorem total_cost_is_58 : number_of_adults * cost_per_adult + number_of_children * cost_per_child = 58 :=
by
  -- Steps of proof will go here
  sorry

end NUMINAMATH_GPT_total_cost_is_58_l1699_169995


namespace NUMINAMATH_GPT_all_real_possible_values_l1699_169996

theorem all_real_possible_values 
  (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 1) : 
  ∃ r : ℝ, r = (a^4 + b^4 + c^4) / (ab + bc + ca) :=
sorry

end NUMINAMATH_GPT_all_real_possible_values_l1699_169996


namespace NUMINAMATH_GPT_option_C_is_correct_l1699_169923

theorem option_C_is_correct (a b c : ℝ) (h : a > b) : c - a < c - b := 
by
  linarith

end NUMINAMATH_GPT_option_C_is_correct_l1699_169923


namespace NUMINAMATH_GPT_trigonometric_values_l1699_169934

theorem trigonometric_values (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 1 / 3) 
  (h2 : Real.cos x - Real.cos y = 1 / 5) : 
  Real.cos (x + y) = 208 / 225 ∧ Real.sin (x - y) = -15 / 17 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_values_l1699_169934


namespace NUMINAMATH_GPT_increase_in_area_l1699_169915

theorem increase_in_area (a : ℝ) : 
  let original_radius := 3
  let new_radius := original_radius + a
  let original_area := π * original_radius ^ 2
  let new_area := π * new_radius ^ 2
  new_area - original_area = π * (3 + a) ^ 2 - 9 * π := 
by
  sorry

end NUMINAMATH_GPT_increase_in_area_l1699_169915


namespace NUMINAMATH_GPT_mia_weight_l1699_169931

theorem mia_weight (a m : ℝ) (h1 : a + m = 220) (h2 : m - a = 2 * a) : m = 165 :=
sorry

end NUMINAMATH_GPT_mia_weight_l1699_169931


namespace NUMINAMATH_GPT_factorize_cubic_l1699_169912

theorem factorize_cubic (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end NUMINAMATH_GPT_factorize_cubic_l1699_169912


namespace NUMINAMATH_GPT_smallest_n_l1699_169922

theorem smallest_n (n : ℕ) : (n > 0) ∧ (2^n % 30 = 1) → n = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_smallest_n_l1699_169922


namespace NUMINAMATH_GPT_total_flowers_sold_l1699_169904

def flowers_sold_on_monday : ℕ := 4
def flowers_sold_on_tuesday : ℕ := 8
def flowers_sold_on_friday : ℕ := 2 * flowers_sold_on_monday

theorem total_flowers_sold : flowers_sold_on_monday + flowers_sold_on_tuesday + flowers_sold_on_friday = 20 := by
  sorry

end NUMINAMATH_GPT_total_flowers_sold_l1699_169904


namespace NUMINAMATH_GPT_max_profit_at_nine_l1699_169939

noncomputable def profit (x : ℝ) : ℝ := - (1 / 3) * x^3 + 81 * x - 23

theorem max_profit_at_nine :
  ∃ x : ℝ, x = 9 ∧ ∀ (ε : ℝ), ε > 0 → 
  (profit (9 - ε) < profit 9 ∧ profit (9 + ε) < profit 9) := 
by
  sorry

end NUMINAMATH_GPT_max_profit_at_nine_l1699_169939


namespace NUMINAMATH_GPT_region_in_quadrants_l1699_169992

theorem region_in_quadrants (x y : ℝ) :
  (y > 3 * x) → (y > 5 - 2 * x) → (x > 0 ∧ y > 0) :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_region_in_quadrants_l1699_169992


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l1699_169988

theorem min_value_reciprocal_sum (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 2) : 
  (∃ c : ℝ, c = (1/x) + (1/y) + (1/z) ∧ c ≥ 9/2) :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l1699_169988


namespace NUMINAMATH_GPT_union_of_sets_l1699_169978

theorem union_of_sets (A B : Set ℤ) (hA : A = {-1, 3}) (hB : B = {2, 3}) : A ∪ B = {-1, 2, 3} := 
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l1699_169978


namespace NUMINAMATH_GPT_no_fib_right_triangle_l1699_169966

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

theorem no_fib_right_triangle (n : ℕ) : 
  ¬ (fibonacci n)^2 + (fibonacci (n+1))^2 = (fibonacci (n+2))^2 := 
by 
  sorry

end NUMINAMATH_GPT_no_fib_right_triangle_l1699_169966


namespace NUMINAMATH_GPT_z_in_fourth_quadrant_l1699_169959

-- Given complex numbers z1 and z2
def z1 : ℂ := 3 - 2 * Complex.I
def z2 : ℂ := 1 + Complex.I

-- Define the multiplication of z1 and z2
def z : ℂ := z1 * z2

-- Prove that z is located in the fourth quadrant
theorem z_in_fourth_quadrant : z.re > 0 ∧ z.im < 0 :=
by
  -- Construction and calculations skipped for the math proof,
  -- the result should satisfy the conditions for being in the fourth quadrant
  sorry

end NUMINAMATH_GPT_z_in_fourth_quadrant_l1699_169959


namespace NUMINAMATH_GPT_infinite_arith_prog_contains_infinite_nth_powers_l1699_169938

theorem infinite_arith_prog_contains_infinite_nth_powers
  (a d : ℕ) (n : ℕ) 
  (h_pos: 0 < d) 
  (h_power: ∃ k : ℕ, ∃ m : ℕ, a + k * d = m^n) :
  ∃ infinitely_many k : ℕ, ∃ m : ℕ, a + k * d = m^n :=
sorry

end NUMINAMATH_GPT_infinite_arith_prog_contains_infinite_nth_powers_l1699_169938


namespace NUMINAMATH_GPT_value_of_b_l1699_169956

theorem value_of_b : (15^2 * 9^2 * 356 = 6489300) :=
by 
  sorry

end NUMINAMATH_GPT_value_of_b_l1699_169956


namespace NUMINAMATH_GPT_money_left_after_distributions_and_donations_l1699_169928

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

end NUMINAMATH_GPT_money_left_after_distributions_and_donations_l1699_169928


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l1699_169972

-- Defining the conditions as assumptions
variables (l w h : ℝ) 
variable (lw_eq : l * w = 10)
variable (wh_eq : w * h = 14)
variable (lh_eq : l * h = 35)

-- Stating the theorem to prove
theorem volume_of_rectangular_prism : l * w * h = 70 :=
by
  have lw := lw_eq
  have wh := wh_eq
  have lh := lh_eq
  sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l1699_169972


namespace NUMINAMATH_GPT_solve_system_of_equations_l1699_169952

def sys_eq1 (x y : ℝ) : Prop := 6 * (1 - x) ^ 2 = 1 / y
def sys_eq2 (x y : ℝ) : Prop := 6 * (1 - y) ^ 2 = 1 / x

theorem solve_system_of_equations (x y : ℝ) :
  sys_eq1 x y ∧ sys_eq2 x y ↔
  ((x = 3 / 2 ∧ y = 2 / 3) ∨
   (x = 2 / 3 ∧ y = 3 / 2) ∨
   (x = 1 / 6 * (4 + 2 ^ (2 / 3) + 2 ^ (4 / 3)) ∧ y = 1 / 6 * (4 + 2 ^ (2 / 3) + 2 ^ (4 / 3)))) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1699_169952


namespace NUMINAMATH_GPT_mandy_cinnamon_amount_correct_l1699_169994

def mandy_cinnamon_amount (nutmeg : ℝ) (cinnamon : ℝ) : Prop :=
  cinnamon = nutmeg + 0.17

theorem mandy_cinnamon_amount_correct :
  mandy_cinnamon_amount 0.5 0.67 :=
by
  sorry

end NUMINAMATH_GPT_mandy_cinnamon_amount_correct_l1699_169994


namespace NUMINAMATH_GPT_notebook_problem_l1699_169999

/-- Conditions:
1. If each notebook costs 3 yuan, 6 more notebooks can be bought.
2. If each notebook costs 5 yuan, there is a 30-yuan shortfall.

We need to show:
1. The total number of notebooks \( x \).
2. The number of 3-yuan notebooks \( n_3 \). -/
theorem notebook_problem (x y n3 : ℕ) (h1 : y = 3 * x + 18) (h2 : y = 5 * x - 30) (h3 : 3 * n3 + 5 * (x - n3) = y) :
  x = 24 ∧ n3 = 15 :=
by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_notebook_problem_l1699_169999


namespace NUMINAMATH_GPT_print_rolls_sold_l1699_169949

-- Defining the variables and conditions
def num_sold := 480
def total_amount := 2340
def solid_price := 4
def print_price := 6

-- Proposed theorem statement
theorem print_rolls_sold (S P : ℕ) (h1 : S + P = num_sold) (h2 : solid_price * S + print_price * P = total_amount) : P = 210 := sorry

end NUMINAMATH_GPT_print_rolls_sold_l1699_169949


namespace NUMINAMATH_GPT_find_pyramid_volume_l1699_169919

noncomputable def volume_of_pyramid (α β R : ℝ) : ℝ :=
  (1/3) * R^3 * (Real.sin α)^2 * Real.cos (α/2) * Real.tan (Real.pi / 4 - α/2) * Real.tan β

theorem find_pyramid_volume (α β R : ℝ) 
  (base_isosceles : ∀ {a b c : ℝ}, a = b) -- Represents the isosceles triangle condition
  (dihedral_angles_equal : ∀ {angle : ℝ}, angle = β) -- Dihedral angle at the base
  (circumcircle_radius : {radius : ℝ // radius = R}) -- Radius of the circumcircle
  (height_through_point : true) -- Condition: height passes through a point inside the triangle
  :
  volume_of_pyramid α β R = (1/3) * R^3 * (Real.sin α)^2 * Real.cos (α/2) * Real.tan (Real.pi / 4 - α/2) * Real.tan β :=
by {
  sorry
}

end NUMINAMATH_GPT_find_pyramid_volume_l1699_169919


namespace NUMINAMATH_GPT_smaller_than_neg3_l1699_169961

theorem smaller_than_neg3 :
  (∃ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3) ∧ ∀ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3 → x = -5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_than_neg3_l1699_169961


namespace NUMINAMATH_GPT_find_k_inv_h_8_l1699_169947

variable (h k : ℝ → ℝ)

-- Conditions
axiom h_inv_k_x (x : ℝ) : h⁻¹ (k x) = 3 * x - 4
axiom h_3x_minus_4 (x : ℝ) : k x = h (3 * x - 4)

-- The statement we want to prove
theorem find_k_inv_h_8 : k⁻¹ (h 8) = 8 := 
  sorry

end NUMINAMATH_GPT_find_k_inv_h_8_l1699_169947


namespace NUMINAMATH_GPT_large_envelopes_count_l1699_169970

theorem large_envelopes_count
  (total_letters : ℕ) (small_envelope_letters : ℕ) (letters_per_large_envelope : ℕ)
  (H1 : total_letters = 80)
  (H2 : small_envelope_letters = 20)
  (H3 : letters_per_large_envelope = 2) :
  (total_letters - small_envelope_letters) / letters_per_large_envelope = 30 :=
sorry

end NUMINAMATH_GPT_large_envelopes_count_l1699_169970


namespace NUMINAMATH_GPT_average_speed_is_69_l1699_169954

-- Definitions for the conditions
def distance_hr1 : ℕ := 90
def distance_hr2 : ℕ := 30
def distance_hr3 : ℕ := 60
def distance_hr4 : ℕ := 120
def distance_hr5 : ℕ := 45
def total_distance : ℕ := distance_hr1 + distance_hr2 + distance_hr3 + distance_hr4 + distance_hr5
def total_time : ℕ := 5

-- The theorem to be proven
theorem average_speed_is_69 :
  (total_distance / total_time) = 69 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_is_69_l1699_169954


namespace NUMINAMATH_GPT_number_of_faces_of_prism_proof_l1699_169936

noncomputable def number_of_faces_of_prism (n : ℕ) : ℕ := 2 + n

theorem number_of_faces_of_prism_proof (n : ℕ) (E_p E_py : ℕ) (h1 : E_p + E_py = 30) (h2 : E_p = 3 * n) (h3 : E_py = 2 * n) :
  number_of_faces_of_prism n = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_faces_of_prism_proof_l1699_169936


namespace NUMINAMATH_GPT_domain_of_composite_function_l1699_169902

theorem domain_of_composite_function :
  ∀ (f : ℝ → ℝ), (∀ x, -1 ≤ x ∧ x ≤ 3 → ∃ y, f x = y) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f (2*x - 1) = y) :=
by
  intros f domain_f x hx
  sorry

end NUMINAMATH_GPT_domain_of_composite_function_l1699_169902


namespace NUMINAMATH_GPT_jade_transactions_l1699_169951

-- Definitions for each condition
def transactions_mabel : ℕ := 90
def transactions_anthony : ℕ := transactions_mabel + (transactions_mabel / 10)
def transactions_cal : ℕ := 2 * transactions_anthony / 3
def transactions_jade : ℕ := transactions_cal + 17

-- The theorem stating that Jade handled 83 transactions
theorem jade_transactions : transactions_jade = 83 := by
  sorry

end NUMINAMATH_GPT_jade_transactions_l1699_169951


namespace NUMINAMATH_GPT_convert_to_base8_l1699_169960

theorem convert_to_base8 (n : ℕ) (h : n = 1024) : 
  (∃ (d3 d2 d1 d0 : ℕ), n = d3 * 8^3 + d2 * 8^2 + d1 * 8^1 + d0 * 8^0 ∧ d3 = 2 ∧ d2 = 0 ∧ d1 = 0 ∧ d0 = 0) :=
by
  sorry

end NUMINAMATH_GPT_convert_to_base8_l1699_169960


namespace NUMINAMATH_GPT_solve_quadratic_eq_solve_cubic_eq_l1699_169941

-- Problem 1: Solve (x-1)^2 = 9
theorem solve_quadratic_eq (x : ℝ) (h : (x - 1) ^ 2 = 9) : x = 4 ∨ x = -2 := 
by 
  sorry

-- Problem 2: Solve (x+3)^3 / 3 - 9 = 0
theorem solve_cubic_eq (x : ℝ) (h : (x + 3) ^ 3 / 3 - 9 = 0) : x = 0 := 
by 
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_solve_cubic_eq_l1699_169941


namespace NUMINAMATH_GPT_ramu_repairs_cost_l1699_169924

theorem ramu_repairs_cost :
  ∃ R : ℝ, 64900 - (42000 + R) = (29.8 / 100) * (42000 + R) :=
by
  use 8006.16
  sorry

end NUMINAMATH_GPT_ramu_repairs_cost_l1699_169924


namespace NUMINAMATH_GPT_x_is_integer_l1699_169969

theorem x_is_integer 
  (x : ℝ)
  (h1 : ∃ a : ℤ, a = x^1960 - x^1919)
  (h2 : ∃ b : ℤ, b = x^2001 - x^1960) :
  ∃ k : ℤ, x = k :=
sorry

end NUMINAMATH_GPT_x_is_integer_l1699_169969


namespace NUMINAMATH_GPT_find_constants_l1699_169927

def N : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![0, 4]
]

def I : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![1, 0],
  ![0, 1]
]

theorem find_constants (c d : ℚ) : 
  (N⁻¹ = c • N + d • I) ↔ (c = -1/12 ∧ d = 7/12) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l1699_169927


namespace NUMINAMATH_GPT_kids_go_to_camp_l1699_169982

variable (total_kids staying_home going_to_camp : ℕ)

theorem kids_go_to_camp (h1 : total_kids = 313473) (h2 : staying_home = 274865) (h3 : going_to_camp = total_kids - staying_home) :
  going_to_camp = 38608 :=
by
  sorry

end NUMINAMATH_GPT_kids_go_to_camp_l1699_169982


namespace NUMINAMATH_GPT_area_hexagon_DEFD_EFE_l1699_169974

variable (D E F D' E' F' : Type)
variable (perimeter_DEF : ℝ) (radius_circumcircle : ℝ)
variable (area_hexagon : ℝ)

theorem area_hexagon_DEFD_EFE' (h1 : perimeter_DEF = 42)
    (h2 : radius_circumcircle = 7)
    (h_def : area_hexagon = 147) :
  area_hexagon = 147 := 
sorry

end NUMINAMATH_GPT_area_hexagon_DEFD_EFE_l1699_169974


namespace NUMINAMATH_GPT_group_elements_eq_one_l1699_169910
-- Import the entire math library

-- Define the main theorem
theorem group_elements_eq_one 
  {G : Type*} [Group G] 
  (a b : G) 
  (h1 : a * b^2 = b^3 * a) 
  (h2 : b * a^2 = a^3 * b) : 
  a = 1 ∧ b = 1 := 
  by 
  sorry

end NUMINAMATH_GPT_group_elements_eq_one_l1699_169910


namespace NUMINAMATH_GPT_rectangle_area_invariant_l1699_169965

theorem rectangle_area_invariant (l w : ℝ) (A : ℝ) 
  (h0 : A = l * w)
  (h1 : A = (l + 3) * (w - 1))
  (h2 : A = (l - 1.5) * (w + 2)) :
  A = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_invariant_l1699_169965


namespace NUMINAMATH_GPT_min_value_of_function_l1699_169906

theorem min_value_of_function (x : ℝ) (h : x > 2) : (x + 1 / (x - 2)) ≥ 4 :=
  sorry

end NUMINAMATH_GPT_min_value_of_function_l1699_169906


namespace NUMINAMATH_GPT_tiffany_total_bags_l1699_169905

def initial_bags : ℕ := 10
def found_on_tuesday : ℕ := 3
def found_on_wednesday : ℕ := 7
def total_bags : ℕ := 20

theorem tiffany_total_bags (initial_bags : ℕ) (found_on_tuesday : ℕ) (found_on_wednesday : ℕ) (total_bags : ℕ) :
    initial_bags + found_on_tuesday + found_on_wednesday = total_bags :=
by
  sorry

end NUMINAMATH_GPT_tiffany_total_bags_l1699_169905


namespace NUMINAMATH_GPT_range_of_a_l1699_169950

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ (Set.Iio (-3) ∪ Set.Ioi 1) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1699_169950


namespace NUMINAMATH_GPT_quadrant_angle_l1699_169971

theorem quadrant_angle (θ : ℝ) (k : ℤ) (h_theta : 0 < θ ∧ θ < 90) : 
  ((180 * k + θ) % 360 < 90) ∨ (180 * k + θ) % 360 ≥ 180 ∧ (180 * k + θ) % 360 < 270 :=
sorry

end NUMINAMATH_GPT_quadrant_angle_l1699_169971


namespace NUMINAMATH_GPT_derivative_at_0_l1699_169989

noncomputable def f : ℝ → ℝ
| x => if x = 0 then 0 else x^2 * Real.exp (|x|) * Real.sin (1 / x^2)

theorem derivative_at_0 : deriv f 0 = 0 := by
  sorry

end NUMINAMATH_GPT_derivative_at_0_l1699_169989


namespace NUMINAMATH_GPT_polynomial_transformation_l1699_169987

theorem polynomial_transformation :
  ∀ (a h k : ℝ), (8 * x^2 - 24 * x - 15 = a * (x - h)^2 + k) → a + h + k = -23.5 :=
by
  intros a h k h_eq
  sorry

end NUMINAMATH_GPT_polynomial_transformation_l1699_169987


namespace NUMINAMATH_GPT_derivative_f_at_1_l1699_169957

-- Define the function f(x) = x * ln(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem to prove f'(1) = 1
theorem derivative_f_at_1 : (deriv f 1) = 1 :=
sorry

end NUMINAMATH_GPT_derivative_f_at_1_l1699_169957


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_a3_a4_a5_l1699_169998

variable {a : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_a3_a4_a5
  (ha : is_arithmetic_sequence a d)
  (h : a 2 + a 3 + a 4 = 12) : 
  (7 * (a 0 + a 6)) / 2 = 28 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_a3_a4_a5_l1699_169998


namespace NUMINAMATH_GPT_cost_formula_l1699_169914

-- Given Conditions
def flat_fee := 5  -- flat service fee in cents
def first_kg_cost := 12  -- cost for the first kilogram in cents
def additional_kg_cost := 5  -- cost for each additional kilogram in cents

-- Integer weight in kilograms
variable (P : ℕ)

-- Total cost calculation proof problem
theorem cost_formula : ∃ C, C = flat_fee + first_kg_cost + additional_kg_cost * (P - 1) → C = 5 * P + 12 :=
by
  sorry

end NUMINAMATH_GPT_cost_formula_l1699_169914


namespace NUMINAMATH_GPT_no_equal_partition_product_l1699_169940

theorem no_equal_partition_product (n : ℕ) (h : n > 1) : 
  ¬ ∃ A B : Finset ℕ, 
    (A ∪ B = (Finset.range n).erase 0 ∧ A ∩ B = ∅ ∧ (A ≠ ∅) ∧ (B ≠ ∅) 
    ∧ A.prod id = B.prod id) := 
sorry

end NUMINAMATH_GPT_no_equal_partition_product_l1699_169940


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1699_169946

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + 4 * x - 5 > 0} = {x : ℝ | x < -5} ∪ {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1699_169946


namespace NUMINAMATH_GPT_value_of_a_100_l1699_169953

open Nat

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (succ k) => sequence k + 4

theorem value_of_a_100 : sequence 99 = 397 := by
  sorry

end NUMINAMATH_GPT_value_of_a_100_l1699_169953
